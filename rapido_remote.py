#!/usr/bin/env python3
"""SSH/SFTP orchestration for rapido-console.py.

Kept separate from the Flask routing in rapido-console.py so the remote-execution
logic can be exercised (and read) without pulling in Flask. Nothing here persists
credentials to disk or to any log -- a caller-supplied password or key path lives
only in the paramiko client object for the lifetime of one job.
"""
import io
import os
import posixpath
import stat
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import paramiko


class RemoteError(Exception):
    """Raised for any connect/exec/transfer failure, with a message safe to show
    directly in the console's live-status feed (never includes the password)."""


@dataclass
class HostConfig:
    host: str
    port: int = 22
    username: str = ""
    password: Optional[str] = None
    key_path: Optional[str] = None
    # Raw PEM text read client-side from a browser file picker. Browsers never
    # expose a chosen file's real filesystem path to page JS (only its name), so
    # "browse for a key" has to ship the key's contents instead of a path --
    # key_path remains for the case where the operator types a path that's valid
    # on the machine running the console itself.
    key_contents: Optional[str] = None
    key_passphrase: Optional[str] = None

    def label(self) -> str:
        return self.host


def _load_private_key(text: str, passphrase: Optional[str] = None) -> paramiko.PKey:
    """Parse PEM key text of an unknown type by trying each key class paramiko
    supports, mirroring what look_for_keys does for on-disk keys."""
    # DSSKey (DSA) was dropped in paramiko 5.0; only try it if this install still
    # has it, so this keeps working across older paramiko versions too.
    key_classes = [paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey]
    if hasattr(paramiko, "DSSKey"):
        key_classes.append(paramiko.DSSKey)
    needs_passphrase = False
    last_err: Optional[Exception] = None
    for cls in key_classes:
        try:
            return cls.from_private_key(io.StringIO(text), password=passphrase)
        except paramiko.PasswordRequiredException as e:
            needs_passphrase = True
            last_err = e
        except (paramiko.SSHException, ValueError) as e:
            last_err = e
    if needs_passphrase and not passphrase:
        raise RemoteError("This identity key is passphrase-protected; enter its passphrase.")
    raise RemoteError(f"Could not parse the identity key: {last_err}")


def connect(config: HostConfig, timeout: int = 15) -> paramiko.SSHClient:
    """Open an SSH connection using either a password or a key (file path or
    browsed-in contents), never more than one of those.

    AutoAddPolicy accepts unknown host keys rather than prompting -- there is no
    interactive terminal on the other end of a web request to answer a prompt on,
    and this tool targets lab/test nodes the operator already has out-of-band
    access to, not arbitrary internet hosts.
    """
    key_given = bool(config.key_path) or bool(config.key_contents)
    if config.password and key_given:
        raise RemoteError("Provide either a password or an identity key, not both.")
    if not config.password and not key_given:
        raise RemoteError("A password or an identity key is required.")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        if config.key_contents:
            pkey = _load_private_key(config.key_contents, config.key_passphrase)
            client.connect(
                hostname=config.host,
                port=config.port,
                username=config.username,
                pkey=pkey,
                timeout=timeout,
                banner_timeout=timeout,
                auth_timeout=timeout,
                allow_agent=False,
                look_for_keys=False,
            )
        elif config.key_path:
            client.connect(
                hostname=config.host,
                port=config.port,
                username=config.username,
                key_filename=config.key_path,
                passphrase=config.key_passphrase,
                timeout=timeout,
                banner_timeout=timeout,
                auth_timeout=timeout,
            )
        else:
            client.connect(
                hostname=config.host,
                port=config.port,
                username=config.username,
                password=config.password,
                timeout=timeout,
                banner_timeout=timeout,
                auth_timeout=timeout,
                allow_agent=False,
                look_for_keys=False,
            )
    except paramiko.AuthenticationException as e:
        raise RemoteError(f"Authentication failed for {config.username}@{config.host}: {e}") from e
    except (paramiko.SSHException, OSError) as e:
        raise RemoteError(f"Could not connect to {config.host}:{config.port}: {e}") from e
    return client


def remote_workdir(username: str, job_id: str) -> str:
    """POSIX path on the remote host used to stage rapido-collect.py and its output
    for this job. Under the user's home dir so no remote-side permission is needed
    beyond what the login account already has."""
    return posixpath.join(f"/home/{username}", ".rapido-console", job_id)


_BENCHMARK_SOURCES = [
    "gpu_p2p_bandwidth.cpp",
    "gpu_kernel_benchmarks.cpp",
    "gpu_host_bandwidth.cpp",
    "gpu_topology.cpp",
    "gpu_gemm_library.cpp",
    "gpu_rccl_collectives.cpp",
]


def upload_collector(client: paramiko.SSHClient, local_path: str, remote_dir: str) -> str:
    """SFTP the local rapido-collect.py into remote_dir (created if missing) so the
    console never depends on a copy that may already exist -- and may be stale --
    on the target host. Returns the remote path.

    Also uploads the .cpp benchmark sources that live alongside rapido-collect.py in
    this repo, since rapido-collect.py compiles them on demand (via hipcc) from its
    own script_dir -- without them, -m/--p2p/--full fail remotely with a "Source
    file not found" error even though the collector script itself is present. Any
    source missing locally is skipped rather than failing the whole upload -- a
    repo checkout without microbenchmark sources should still work for other
    sections.
    """
    sftp = client.open_sftp()
    try:
        _mkdir_p(sftp, remote_dir)
        remote_path = posixpath.join(remote_dir, "rapido-collect.py")
        sftp.put(local_path, remote_path)

        local_dir = os.path.dirname(local_path)
        for name in _BENCHMARK_SOURCES:
            local_cpp = os.path.join(local_dir, name)
            if os.path.isfile(local_cpp):
                sftp.put(local_cpp, posixpath.join(remote_dir, name))
        return remote_path
    except (paramiko.SSHException, OSError) as e:
        raise RemoteError(f"Failed to upload rapido-collect.py: {e}") from e
    finally:
        sftp.close()


def _mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    path = ""
    for part in parts:
        path += "/" + part
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)


def check_python3(client: paramiko.SSHClient) -> None:
    """Fail fast and clearly if python3 is not on the remote PATH, rather than
    letting the collector invocation hang or fail with a cryptic exit status."""
    _, stdout, _ = client.exec_command("command -v python3", timeout=15)
    found = stdout.channel.recv_exit_status() == 0
    if not found:
        raise RemoteError("python3 was not found on the remote host's PATH.")


def run_collect(
    client: paramiko.SSHClient,
    remote_script_path: str,
    remote_dir: str,
    flags: List[str],
    tag: str,
    on_line: Callable[[str], None],
) -> str:
    """Run `python3 rapido-collect.py <flags> -v --tag <tag>` in remote_dir, streaming
    stdout+stderr line-by-line through on_line, and return the remote path of the JSON
    it wrote. Raises RemoteError if the process exits nonzero or never prints the
    "Server information saved to:" line rapido-collect.py already emits unconditionally.
    """
    flag_str = " ".join(flags)
    command = f"cd {_shquote(remote_dir)} && python3 rapido-collect.py {flag_str} -v --tag {_shquote(tag)}"
    on_line(f"$ {command}")

    stdin, stdout, stderr = client.exec_command(command, timeout=None, get_pty=False)
    stdin.close()
    channel = stdout.channel
    channel.setblocking(0)

    saved_path: Optional[str] = None
    buf_out, buf_err = "", ""
    while True:
        made_progress = False
        if channel.recv_ready():
            chunk = channel.recv(4096).decode("utf-8", errors="replace")
            buf_out += chunk
            made_progress = True
        if channel.recv_stderr_ready():
            chunk = channel.recv_stderr(4096).decode("utf-8", errors="replace")
            buf_err += chunk
            made_progress = True

        while "\n" in buf_out:
            line, buf_out = buf_out.split("\n", 1)
            on_line(line)
            marker = "Server information saved to: "
            if marker in line:
                saved_path = line.split(marker, 1)[1].strip()
        while "\n" in buf_err:
            line, buf_err = buf_err.split("\n", 1)
            on_line(f"[stderr] {line}")

        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        if not made_progress:
            time.sleep(0.2)

    if buf_out:
        on_line(buf_out)
    if buf_err:
        on_line(f"[stderr] {buf_err}")

    exit_status = channel.recv_exit_status()
    if exit_status != 0:
        raise RemoteError(f"rapido-collect.py exited with status {exit_status}")
    if not saved_path:
        raise RemoteError(
            "rapido-collect.py finished but did not report an output file "
            "(check the collection warnings above)."
        )
    # rapido-collect.py prints the path it was given via --output/--output-dir,
    # which is relative to remote_dir (the cwd the command ran in) unless the
    # caller passed an absolute path -- resolve it so SFTP (whose own cwd is
    # the login home, not remote_dir) can find the file.
    if not posixpath.isabs(saved_path):
        saved_path = posixpath.join(remote_dir, saved_path)
    return saved_path


def fetch_file(client: paramiko.SSHClient, remote_path: str, local_path: str) -> None:
    sftp = client.open_sftp()
    try:
        sftp.get(remote_path, local_path)
    except (paramiko.SSHException, OSError) as e:
        raise RemoteError(f"Failed to retrieve {remote_path}: {e}") from e
    finally:
        sftp.close()


def cleanup_remote_dir(client: paramiko.SSHClient, remote_dir: str) -> None:
    """Best-effort removal of the per-job staging directory. Failure here is not
    fatal to the job -- it just leaves a stray directory under ~/.rapido-console/
    on the remote host -- so errors are swallowed rather than raised."""
    try:
        client.exec_command(f"rm -rf {_shquote(remote_dir)}", timeout=15)
    except (paramiko.SSHException, OSError):
        pass


def _shquote(value: str) -> str:
    """Minimal POSIX single-quote escaping for building remote shell commands."""
    return "'" + value.replace("'", "'\\''") + "'"


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]
