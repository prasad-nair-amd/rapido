#!/usr/bin/env python3
"""Web console for rapido: drive remote collection over SSH from a browser form,
watch live status while it runs, and open the generated HTML report when done.

Run with: python rapido-console.py [--host 127.0.0.1] [--port 5000]

Nothing here is persisted across a job: passwords/key paths live only in the
paramiko client for the lifetime of that job's worker thread, and the in-memory
job registry is dropped when the process exits (no database).
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, request, send_file

import rapido_ai_audit as ai_audit_module
import rapido_remote as remote

REPO_ROOT = Path(__file__).resolve().parent
COLLECT_SCRIPT = REPO_ROOT / "rapido-collect.py"
REPORT_SCRIPT = REPO_ROOT / "rapido-report.py"
RUNS_DIR = REPO_ROOT / "console_runs"

app = Flask(__name__)

# job_id -> job state dict. Guarded by _JOBS_LOCK since the SSE route (main thread)
# reads it while the worker thread (spawned per job) writes it.
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

# Checkbox name -> rapido-collect.py flag. "gpu_health" has no flag of its own --
# RAS/telemetry already ride along with -g (rapido-collect.py:4152-4177) -- so it
# maps to the same flag as "gpu"; either checkbox alone is enough to trigger -g.
_SECTION_FLAGS = {
    "cpu": "-c",
    "gpu": "-g",
    "gpu_health": "-g",
    "rocm": "-r",
    "network": "-n",
    "bmc": "-b",
    "platform": "-t",
    "microbenchmarks": "-m",
}


def _job_flags(sections: List[str], p2p: bool, full: bool) -> List[str]:
    flags = sorted({_SECTION_FLAGS[s] for s in sections if s in _SECTION_FLAGS})
    if "-m" in flags:
        if p2p:
            flags.append("--p2p")
        if full:
            flags.append("--full")
    return flags


def _new_job(
    host_configs: List[remote.HostConfig], sections: List[str], p2p: bool, full: bool, ai_audit: bool
) -> str:
    job_id = remote.new_job_id()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "running",
            "queue": queue.Queue(),
            "report_path": None,
            "audit_path": None,
            "error": None,
        }
    thread = threading.Thread(
        target=_run_job, args=(job_id, host_configs, sections, p2p, full, ai_audit), daemon=True
    )
    thread.start()
    return job_id


def _emit(job_id: str, host_label: str, message: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return
    stamp = time.strftime("%H:%M:%S")
    job["queue"].put({"host": host_label, "line": f"[{stamp}] {message}"})


def _finish(
    job_id: str,
    status: str,
    report_path: Optional[str] = None,
    audit_path: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["status"] = status
        job["report_path"] = report_path
        job["audit_path"] = audit_path
        job["error"] = error
    job["queue"].put(None)  # sentinel: tells the SSE route the stream is done


def _collect_from_host(
    job_id: str, host_index: int, config: remote.HostConfig, flags: List[str], local_dir: Path
) -> Path:
    label = config.label()
    _emit(job_id, label, f"Connecting to {config.host}...")
    client = remote.connect(config)
    try:
        # host_index disambiguates the remote workdir and local JSON filename so a
        # two-host run against the same address (e.g. a self-comparison test) can't
        # have host 2 clobber host 1's files.
        job_scoped_id = f"{job_id}-h{host_index}"
        remote.check_python3(client)
        remote_dir = remote.remote_workdir(config.username, job_scoped_id)
        _emit(job_id, label, "Uploading rapido-collect.py...")
        remote.upload_collector(client, str(COLLECT_SCRIPT), remote_dir)

        _emit(job_id, label, f"Running rapido-collect.py {' '.join(flags)} ...")
        remote_json = remote.run_collect(
            client, str(COLLECT_SCRIPT), remote_dir, flags, job_scoped_id,
            on_line=lambda line: _emit(job_id, label, line),
        )

        _emit(job_id, label, "Collection finished, retrieving JSON...")
        local_json = local_dir / f"host{host_index}_{label.replace(':', '_')}.json"
        remote.fetch_file(client, remote_json, str(local_json))
        _emit(job_id, label, f"Retrieved {local_json.name}")
        return local_json
    finally:
        remote.cleanup_remote_dir(client, remote.remote_workdir(config.username, job_scoped_id))
        client.close()


def _run_job(
    job_id: str,
    host_configs: List[remote.HostConfig],
    sections: List[str],
    p2p: bool,
    full: bool,
    ai_audit: bool,
) -> None:
    flags = _job_flags(sections, p2p, full)
    local_dir = RUNS_DIR / job_id
    local_dir.mkdir(parents=True, exist_ok=True)

    json_paths: List[Path] = []
    try:
        for host_index, config in enumerate(host_configs, start=1):
            json_paths.append(_collect_from_host(job_id, host_index, config, flags, local_dir))
    except remote.RemoteError as e:
        _emit(job_id, "console", f"ERROR: {e}")
        _finish(job_id, "failed", error=str(e))
        return
    except Exception as e:  # noqa: BLE001 - surface any unexpected failure to the UI
        _emit(job_id, "console", f"ERROR: unexpected failure: {e}")
        _finish(job_id, "failed", error=str(e))
        return

    _emit(job_id, "console", "Generating report...")
    report_path = local_dir / "report.html"
    try:
        _generate_report(json_paths, report_path)
    except Exception as e:  # noqa: BLE001
        _emit(job_id, "console", f"ERROR: report generation failed: {e}")
        _finish(job_id, "failed", error=str(e))
        return

    _emit(job_id, "console", "Report ready.")

    audit_path: Optional[Path] = None
    if ai_audit:
        # A failed audit step must not fail the whole job -- the mechanical report
        # is already generated and valuable on its own, so any Claude CLI problem
        # (not found, not logged in, timeout) is surfaced as a warning line only.
        try:
            _emit(job_id, "console", "Generating AI audit summary via Claude CLI...")
            claude_path = ai_audit_module.find_claude_cli()
            if not claude_path:
                raise ai_audit_module.AuditError(
                    "Claude CLI not found. Install Claude Code, or run its one-time /login."
                )
            summary = ai_audit_module.generate_audit_summary(json_paths, claude_path)
            audit_path = local_dir / "ai_audit.txt"
            audit_path.write_text(summary, encoding="utf-8")
            _splice_audit_into_report(report_path, summary)
            _emit(job_id, "console", "AI audit summary ready.")
        except ai_audit_module.AuditError as e:
            _emit(job_id, "console", f"WARNING: AI audit summary skipped: {e}")
            audit_path = None
        except Exception as e:  # noqa: BLE001
            _emit(job_id, "console", f"WARNING: AI audit summary failed unexpectedly: {e}")
            audit_path = None

    _finish(job_id, "done", report_path=str(report_path), audit_path=str(audit_path) if audit_path else None)


def _generate_report(json_paths: List[Path], report_path: Path) -> None:
    if len(json_paths) == 1:
        cmd = [sys.executable, str(REPORT_SCRIPT), "-i", str(json_paths[0]), "-o", str(report_path)]
    else:
        cmd = [
            sys.executable, str(REPORT_SCRIPT),
            "-f1", str(json_paths[0]), "-f2", str(json_paths[1]),
            "-o", str(report_path),
        ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "rapido-report.py failed")


def _markdown_lite_to_html(text: str) -> str:
    """Render the handful of Markdown constructs Claude's prose reliably uses
    (#/##/### headers, **bold**, --- rules, "- " bullet lists, blank-line
    paragraphs) as HTML. Not a general Markdown parser -- just enough so the
    audit card doesn't show raw '##'/'**' syntax to the reader.
    """
    def inline(s: str) -> str:
        s = escape(s)
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

    lines = text.split("\n")
    out: List[str] = []
    para: List[str] = []
    in_list = False

    def flush_para():
        if para:
            out.append(f"<p>{'<br>'.join(inline(p) for p in para)}</p>")
            para.clear()

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_para()
            close_list()
            continue
        header_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if header_match:
            flush_para()
            close_list()
            level = min(len(header_match.group(1)) + 2, 6)  # keep below the card's own <h2>
            out.append(f"<h{level}>{inline(header_match.group(2))}</h{level}>")
            continue
        if stripped in ("---", "***", "___"):
            flush_para()
            close_list()
            out.append("<hr>")
            continue
        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet_match:
            flush_para()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(bullet_match.group(1))}</li>")
            continue
        close_list()
        para.append(stripped)

    flush_para()
    close_list()
    return "".join(out)


def _splice_audit_into_report(report_path: Path, summary: str) -> None:
    """Insert the AI-generated narrative as its own first tab in the
    already-generated report.html -- rapido-report.py itself knows nothing
    about this step, so the splice happens here as a post-process rather than
    by touching that script's generation logic (tab bar markup/JS, ids, and
    active-state handling all mirror what rapido-report.py already emits for
    CPU/GPU/etc. tabs).
    """
    html = report_path.read_text(encoding="utf-8")

    tabs_marker = '<div class="tabs" id="tabs-container">'
    tabs_idx = html.find(tabs_marker)
    if tabs_idx == -1:
        return
    content_marker = 'class="tab-content'
    content_idx = html.find(content_marker, tabs_idx)
    if content_idx == -1:
        return
    content_div_start = html.rfind("<div", tabs_idx, content_idx)
    if content_div_start == -1:
        return

    # Demote whatever tab/content the report picked as its default -- only the
    # new AI-audit tab should be active on load. There's at most one of each
    # (rapido-report.py only ever marks the first populated section active).
    html = (
        html[:tabs_idx]
        + html[tabs_idx:content_div_start].replace('class="tab active"', 'class="tab"', 1)
        + html[content_div_start:]
    )
    first_content_active = html.find('tab-content active"', content_div_start)
    if first_content_active != -1:
        html = (
            html[:first_content_active]
            + 'tab-content"'
            + html[first_content_active + len('tab-content active"'):]
        )

    tab_button = (
        '<button class="tab active" onclick="openTab(event, \'ai_audit\')" '
        'draggable="true" data-tab="ai_audit">AI Audit Summary</button>'
    )
    html = html[:tabs_idx + len(tabs_marker)] + tab_button + html[tabs_idx + len(tabs_marker):]

    # Recompute content_div_start -- the tab button insertion shifted every
    # offset after tabs_idx, including the one found above.
    content_idx = html.find(content_marker, tabs_idx)
    content_div_start = html.rfind("<div", tabs_idx, content_idx)

    body_html = _markdown_lite_to_html(summary)
    content_div = (
        '<div id="ai_audit" class="tab-content active">'
        '<div class="ai-audit">'
        '<div class="ai-audit-caption">Generated by Claude from this collection\'s JSON. '
        'Treat as an assistive first read, not a substitute for the data below.</div>'
        f'<div class="ai-audit-body">{body_html}</div>'
        '</div>'
        '</div>'
    )
    html = html[:content_div_start] + content_div + html[content_div_start:]

    style_marker = "</style>"
    style_idx = html.find(style_marker)
    if style_idx != -1:
        css = (
            ".ai-audit { background: #eef4fb; border-left: 4px solid #1565c0; color: #1a2733; "
            "padding: 14px 18px; border-radius: 4px; margin-bottom: 16px; }"
            ".ai-audit-caption { font-size: 0.8rem; color: #4a5a68; margin-bottom: 10px; }"
            ".ai-audit-body p { margin: 0 0 10px 0; line-height: 1.5; }"
            ".ai-audit-body h3, .ai-audit-body h4, .ai-audit-body h5 { margin: 14px 0 6px 0; color: #0d47a1; }"
            ".ai-audit-body ul { margin: 0 0 10px 0; padding-left: 20px; }"
            ".ai-audit-body li { margin-bottom: 4px; line-height: 1.5; }"
            ".ai-audit-body hr { border: none; border-top: 1px solid #c8d6e5; margin: 12px 0; }"
        )
        html = html[:style_idx] + css + html[style_idx:]

    report_path.write_text(html, encoding="utf-8")


def _host_config_from_form(payload: Dict[str, Any], prefix: str) -> Optional[remote.HostConfig]:
    host = (payload.get(f"{prefix}_host") or "").strip()
    if not host:
        return None
    auth_mode = payload.get(f"{prefix}_auth_mode")
    password = payload.get(f"{prefix}_password") or None
    key_path = payload.get(f"{prefix}_key_path") or None
    # Contents of a key file picked via the browser's native file input --
    # browsers never expose that file's real filesystem path to page JS, only its
    # name, so "Browse..." ships the file's text instead of a path.
    key_contents = payload.get(f"{prefix}_key_contents") or None
    key_passphrase = payload.get(f"{prefix}_key_passphrase") or None
    if auth_mode == "key":
        password = None
    else:
        key_path = None
        key_contents = None
        key_passphrase = None
    return remote.HostConfig(
        host=host,
        port=int(payload.get(f"{prefix}_port") or 22),
        username=(payload.get(f"{prefix}_username") or "").strip(),
        password=password,
        key_path=key_path,
        key_contents=key_contents,
        key_passphrase=key_passphrase,
    )


@app.route("/")
def index() -> str:
    return _INDEX_HTML


@app.route("/jobs", methods=["POST"])
def create_job() -> Any:
    payload = request.get_json(force=True, silent=True) or {}

    host1 = _host_config_from_form(payload, "host1")
    if host1 is None:
        return {"error": "Host 1 is required."}, 400

    host_configs = [host1]
    if payload.get("host2_enabled"):
        host2 = _host_config_from_form(payload, "host2")
        if host2 is None:
            return {"error": "Host 2 is enabled but no hostname was given."}, 400
        host_configs.append(host2)

    sections = [s for s in _SECTION_FLAGS if payload.get(f"section_{s}")]
    if not sections:
        return {"error": "Select at least one section to collect."}, 400

    p2p = bool(payload.get("option_p2p"))
    full = bool(payload.get("option_full"))
    ai_audit = bool(payload.get("option_ai_audit"))

    job_id = _new_job(host_configs, sections, p2p, full, ai_audit)
    return {"job_id": job_id}


@app.route("/events/<job_id>")
def events(job_id: str) -> Response:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return Response("job not found", status=404)

    def stream():
        q: queue.Queue = job["queue"]
        while True:
            item = q.get()
            if item is None:
                with _JOBS_LOCK:
                    status = job["status"]
                    report_path = job["report_path"]
                    audit_path = job.get("audit_path")
                    error = job["error"]
                payload = {"done": True, "status": status, "error": error}
                if report_path:
                    payload["report_url"] = f"/jobs/{job_id}/report"
                if audit_path:
                    payload["audit_url"] = f"/jobs/{job_id}/audit"
                yield f"data: {json.dumps(payload)}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(stream(), mimetype="text/event-stream")


@app.route("/jobs/<job_id>/report")
def get_report(job_id: str) -> Any:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None or not job.get("report_path"):
        return "report not ready", 404
    return send_file(job["report_path"])


@app.route("/jobs/<job_id>/audit")
def get_audit(job_id: str) -> Any:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None or not job.get("audit_path"):
        return "audit summary not available", 404
    return send_file(job["audit_path"], mimetype="text/plain")


_SECTION_LABELS = [
    ("cpu", "CPU"),
    ("gpu", "GPU"),
    ("gpu_health", "GPU Health"),
    ("rocm", "ROCm"),
    ("network", "Network"),
    ("bmc", "BMC"),
    ("platform", "Platform"),
    ("microbenchmarks", "Microbenchmarks"),
]

_SECTION_CHECKBOXES = "\n".join(
    f'''<label class="check"><input type="checkbox" name="section_{key}" value="1"> {escape(label)}</label>'''
    for key, label in _SECTION_LABELS
)

_INDEX_HTML = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Rapido Console</title>
<style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #f4f5f7; margin: 0; padding: 24px; color: #1a1a1a; }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
    .subtitle {{ color: #666; margin-bottom: 24px; font-size: 0.9rem; }}
    .card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
    .card h2 {{ font-size: 1.05rem; margin-top: 0; }}
    .row {{ display: flex; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }}
    .row label {{ display: block; font-size: 0.8rem; color: #555; margin-bottom: 3px; }}
    .row .field {{ flex: 1; min-width: 160px; }}
    input[type=text], input[type=password], input[type=number] {{
        width: 100%; padding: 7px 9px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-size: 0.9rem;
    }}
    .auth-toggle {{ display: flex; gap: 16px; margin-bottom: 8px; font-size: 0.85rem; }}
    .key-path-hint {{ margin-top: 6px; font-size: 0.78rem; color: #777; display: flex; align-items: center; gap: 6px; }}
    .key-path-hint input {{ flex: 1; padding: 5px 7px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.85rem; }}
    .sections {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }}
    .check {{ font-size: 0.9rem; display: flex; align-items: center; gap: 6px; }}
    .sub-options {{ margin-top: 8px; padding-left: 4px; font-size: 0.85rem; color: #444; display: none; }}
    .sub-options.visible {{ display: flex; gap: 16px; }}
    #host2Block {{ display: none; }}
    #host2Block.visible {{ display: block; }}
    button.primary {{
        background: #c0392b; color: #fff; border: none; border-radius: 6px; padding: 10px 22px;
        font-size: 0.95rem; cursor: pointer;
    }}
    button.primary:disabled {{ background: #999; cursor: not-allowed; }}
    #status {{ background: #1e1e1e; color: #d4d4d4; border-radius: 6px; padding: 12px 14px; font-family: Consolas, monospace;
        font-size: 0.82rem; max-height: 360px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; display: none; }}
    #status .line {{ margin-bottom: 2px; }}
    #status .host-tag {{ color: #4fc3f7; }}
    #reportLink {{ display: none; margin-top: 14px; }}
    #reportLink a {{ background: #2e7d32; color: #fff; padding: 9px 18px; border-radius: 6px; text-decoration: none; font-size: 0.9rem; margin-right: 10px; }}
    #auditAnchor {{ display: none; background: #1565c0 !important; }}
    .ai-audit-caption {{ font-size: 0.78rem; color: #777; margin-top: 6px; }}
    #errorBox {{ display: none; background: #fdecea; border-left: 4px solid #c62828; color: #5f1a16; padding: 10px 14px; border-radius: 4px; margin-top: 12px; font-size: 0.88rem; }}
</style>
</head>
<body>
<div class="container">
    <h1>Rapido Console</h1>
    <div class="subtitle">Collect server info from remote hosts over SSH and generate an HTML report.</div>

    <div class="card">
        <h2>Host 1</h2>
        {{HOST_BLOCK_1}}
    </div>

    <div class="card">
        <label class="check"><input type="checkbox" id="host2Enabled"> Enable second host (for a comparison report)</label>
        <div id="host2Block">
            <h2>Host 2</h2>
            {{HOST_BLOCK_2}}
        </div>
    </div>

    <div class="card">
        <h2>Sections to collect</h2>
        <div class="sections">
            {_SECTION_CHECKBOXES}
        </div>
        <div class="sub-options" id="microSubOptions">
            <label class="check"><input type="checkbox" id="optionP2p"> Include P2P bandwidth (--p2p)</label>
            <label class="check"><input type="checkbox" id="optionFull"> Include library GEMM + RCCL (--full)</label>
        </div>
    </div>

    <div class="card">
        <label class="check"><input type="checkbox" id="optionAiAudit"> Generate AI audit summary (uses local Claude CLI)</label>
        <div class="ai-audit-caption">Requires running <code>claude /login</code> once on this machine beforehand. Adds a narrative summary to the report; failures here don't stop the job.</div>
    </div>

    <div class="card">
        <button class="primary" id="startBtn" onclick="startJob()">Start Collection</button>
        <div id="status"></div>
        <div id="errorBox"></div>
        <div id="reportLink"><a id="reportAnchor" href="#" target="_blank">View report</a> <a id="auditAnchor" href="#" target="_blank">View AI audit summary</a></div>
    </div>
</div>

<script>
function readFileAsText(file) {{
    return new Promise(function(resolve, reject) {{
        const reader = new FileReader();
        reader.onload = function() {{ resolve(reader.result); }};
        reader.onerror = function() {{ reject(reader.error); }};
        reader.readAsText(file);
    }});
}}

async function hostBlock(prefix) {{
    const fileInput = document.getElementById(prefix + '_key_file');
    let keyContents = '';
    if (fileInput && fileInput.files.length > 0) {{
        keyContents = await readFileAsText(fileInput.files[0]);
    }}
    return {{
        host: document.getElementById(prefix + '_host').value.trim(),
        port: document.getElementById(prefix + '_port').value || 22,
        username: document.getElementById(prefix + '_username').value.trim(),
        auth_mode: document.querySelector('input[name="' + prefix + '_auth"]:checked').value,
        password: document.getElementById(prefix + '_password').value,
        key_path: document.getElementById(prefix + '_key_path').value.trim(),
        key_contents: keyContents,
        key_passphrase: document.getElementById(prefix + '_key_passphrase').value,
    }};
}}

document.getElementById('host2Enabled').addEventListener('change', function() {{
    document.getElementById('host2Block').classList.toggle('visible', this.checked);
}});

document.querySelectorAll('input[name="host1_auth"], input[name="host2_auth"]').forEach(function(radio) {{
    radio.addEventListener('change', function() {{
        const prefix = this.name.replace('_auth', '');
        const isKey = this.value === 'key';
        document.getElementById(prefix + '_password_field').style.display = isKey ? 'none' : 'block';
        document.getElementById(prefix + '_key_field').style.display = isKey ? 'block' : 'none';
    }});
}});

document.querySelector('input[name="section_microbenchmarks"]').addEventListener('change', function() {{
    document.getElementById('microSubOptions').classList.toggle('visible', this.checked);
}});

async function buildPayload() {{
    const h1 = await hostBlock('host1');
    const payload = {{
        host1_host: h1.host, host1_port: h1.port, host1_username: h1.username,
        host1_auth_mode: h1.auth_mode, host1_password: h1.password, host1_key_path: h1.key_path,
        host1_key_contents: h1.key_contents, host1_key_passphrase: h1.key_passphrase,
        host2_enabled: document.getElementById('host2Enabled').checked,
    }};
    if (payload.host2_enabled) {{
        const h2 = await hostBlock('host2');
        payload.host2_host = h2.host; payload.host2_port = h2.port; payload.host2_username = h2.username;
        payload.host2_auth_mode = h2.auth_mode; payload.host2_password = h2.password; payload.host2_key_path = h2.key_path;
        payload.host2_key_contents = h2.key_contents; payload.host2_key_passphrase = h2.key_passphrase;
    }}
    document.querySelectorAll('.sections input[type=checkbox]').forEach(function(cb) {{
        payload[cb.name] = cb.checked;
    }});
    payload.option_p2p = document.getElementById('optionP2p').checked;
    payload.option_full = document.getElementById('optionFull').checked;
    payload.option_ai_audit = document.getElementById('optionAiAudit').checked;
    return payload;
}}

function appendStatus(host, line) {{
    const box = document.getElementById('status');
    box.style.display = 'block';
    const div = document.createElement('div');
    div.className = 'line';
    div.innerHTML = '<span class="host-tag">[' + host + ']</span> ' + line.replace(/&/g, '&amp;').replace(/</g, '&lt;');
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}}

async function startJob() {{
    const btn = document.getElementById('startBtn');
    const errorBox = document.getElementById('errorBox');
    const reportLink = document.getElementById('reportLink');
    errorBox.style.display = 'none';
    reportLink.style.display = 'none';
    document.getElementById('status').innerHTML = '';
    btn.disabled = true;

    let payload;
    try {{
        payload = await buildPayload();
    }} catch (err) {{
        errorBox.textContent = 'Could not read the identity key file: ' + err;
        errorBox.style.display = 'block';
        btn.disabled = false;
        return;
    }}

    fetch('/jobs', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload),
    }}).then(function(res) {{
        return res.json().then(function(data) {{ return {{ ok: res.ok, data: data }}; }});
    }}).then(function(result) {{
        if (!result.ok) {{
            errorBox.textContent = result.data.error || 'Failed to start job.';
            errorBox.style.display = 'block';
            btn.disabled = false;
            return;
        }}
        const jobId = result.data.job_id;
        const source = new EventSource('/events/' + jobId);
        source.onmessage = function(evt) {{
            const msg = JSON.parse(evt.data);
            if (msg.done) {{
                source.close();
                btn.disabled = false;
                if (msg.status === 'done' && msg.report_url) {{
                    document.getElementById('reportAnchor').href = msg.report_url;
                    reportLink.style.display = 'block';
                    const auditAnchor = document.getElementById('auditAnchor');
                    if (msg.audit_url) {{
                        auditAnchor.href = msg.audit_url;
                        auditAnchor.style.display = 'inline-block';
                    }} else {{
                        auditAnchor.style.display = 'none';
                    }}
                }} else {{
                    errorBox.textContent = msg.error || 'Job failed.';
                    errorBox.style.display = 'block';
                }}
                return;
            }}
            appendStatus(msg.host, msg.line);
        }};
        source.onerror = function() {{
            appendStatus('console', 'Connection to server lost.');
            source.close();
            btn.disabled = false;
        }};
    }}).catch(function(err) {{
        errorBox.textContent = 'Request failed: ' + err;
        errorBox.style.display = 'block';
        btn.disabled = false;
    }});
}}
</script>
</body>
</html>
"""


def _host_form_html(prefix: str) -> str:
    return f"""
        <div class="row">
            <div class="field"><label>Hostname / IP</label><input type="text" id="{prefix}_host" placeholder="10.0.0.5"></div>
            <div class="field" style="max-width:100px;"><label>Port</label><input type="number" id="{prefix}_port" value="22"></div>
            <div class="field"><label>Username</label><input type="text" id="{prefix}_username" placeholder="root"></div>
        </div>
        <div class="auth-toggle">
            <label><input type="radio" name="{prefix}_auth" value="password" checked> Password</label>
            <label><input type="radio" name="{prefix}_auth" value="key"> Identity key file</label>
        </div>
        <div class="row" id="{prefix}_password_field">
            <div class="field"><label>Password</label><input type="password" id="{prefix}_password"></div>
        </div>
        <div class="row" id="{prefix}_key_field" style="display:none;">
            <div class="field">
                <label>Identity key file</label>
                <input type="file" id="{prefix}_key_file">
                <div class="key-path-hint">or type a path valid on the machine running this console: <input type="text" id="{prefix}_key_path" placeholder="C:\\path\\to\\key"></div>
            </div>
            <div class="field" style="max-width:200px;"><label>Passphrase (if any)</label><input type="password" id="{prefix}_key_passphrase"></div>
        </div>
    """


_INDEX_HTML = _INDEX_HTML.replace("{HOST_BLOCK_1}", _host_form_html("host1"))
_INDEX_HTML = _INDEX_HTML.replace("{HOST_BLOCK_2}", _host_form_html("host2"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rapido web console")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    RUNS_DIR.mkdir(exist_ok=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
