#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keys used to pair corresponding items between two JSON files (order matters).
_ITEM_PAIR_KEYS: Tuple[str, ...] = ("Section", "Name", "Interface", "Adapter")


def _tab_button_html(tab_id: str, label: str, first_tab: str, has: bool) -> str:
    """Build a tab button; separate from the main f-string template to avoid backslashes in nested f-expressions."""
    if not has:
        return ""
    active = " active" if first_tab == tab_id else ""
    return (
        f'<button class="tab{active}" onclick="openTab(event, \'{tab_id}\')" '
        f'draggable="true" data-tab="{tab_id}">{label}</button>'
    )


def _item_pair_key(item: Any) -> str:
    """Stable string key to match list items between two reports."""
    if isinstance(item, dict):
        for k in _ITEM_PAIR_KEYS:
            if k in item and item[k] is not None:
                return f"{k}:{item[k]!s}"
        try:
            blob = json.dumps(item, sort_keys=True, default=str)
            h = hashlib.md5(blob.encode("utf-8")).hexdigest()[:16]
            return f"__content:{h}"
        except Exception:
            return "__empty__"
    return f"__other:{item!s}"


def _strip_section_field(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in item.items() if k != "Section"}


def _canonical_equal(a: Any, b: Any) -> bool:
    """Deep equality suitable for JSON-like structures (sorted keys for dicts)."""
    if type(a) is not type(b):
        # Allow int/float equivalence
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) == float(b)
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_canonical_equal(a[k], b[k]) for k in sorted(a.keys()))
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_canonical_equal(x, y) for x, y in zip(a, b))
    return a == b


def _pair_section_items(items1: List[Any], items2: List[Any]) -> List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    """
    Pair dict items from two lists by _item_pair_key (FIFO when duplicate keys).
    Unmatched items pair with None. Non-dict entries are paired by order at the end.
    """
    from collections import defaultdict

    queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    nondict2: List[Any] = []
    for it in items2:
        if isinstance(it, dict):
            queues[_item_pair_key(it)].append(it)
        else:
            nondict2.append(it)

    pairs: List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []
    nondict1: List[Any] = []
    for it in items1:
        if isinstance(it, dict):
            k = _item_pair_key(it)
            if queues.get(k):
                pairs.append((it, queues[k].pop(0)))
            else:
                pairs.append((it, None))
        else:
            nondict1.append(it)

    for k, q in queues.items():
        for rest in q:
            pairs.append((None, rest))

    for a, b in zip(nondict1, nondict2):
        pairs.append(({"__scalar__": a}, {"__scalar__": b}))
    for extra in nondict1[len(nondict2) :]:
        pairs.append(({"__scalar__": extra}, None))
    for extra in nondict2[len(nondict1) :]:
        pairs.append((None, {"__scalar__": extra}))

    return pairs


def _render_dict_as_table(data: Dict[str, Any]) -> str:
    """Render a dictionary as an HTML table."""
    rows = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            rows.append(
                f"<tr><th>{escape(str(key))}</th><td>{_value_to_html(value)}</td></tr>"
            )
        else:
            rows.append(
                f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            )
    return "<table class='info-table'>" + "".join(rows) + "</table>"


def _render_list_as_cards(items: List[Any], title: str = "") -> str:
    """Render a list of dictionaries as cards."""
    if not items:
        return "<p class='no-data'>No data available</p>"

    cards = []
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            # Check for Section field first (for ROCm data), then other identifiers
            card_title = item.get("Section") or item.get("Name") or item.get("Interface") or item.get("Adapter") or f"{title} {idx + 1}"

            # Create a copy of the item without the Section field for display
            display_item = {k: v for k, v in item.items() if k != "Section"}

            card_content = _render_dict_as_table(display_item)
            cards.append(
                f"<div class='card'>"
                f"<div class='card-header'>{escape(str(card_title))}</div>"
                f"<div class='card-body'>{card_content}</div>"
                f"</div>"
            )
        else:
            cards.append(f"<div class='card'><div class='card-body'>{escape(str(item))}</div></div>")

    return "".join(cards)


def _value_to_html(value: Any) -> str:
    """Recursively convert Python data structures to HTML fragments."""
    if isinstance(value, dict):
        return _render_dict_as_table(value)
    if isinstance(value, list):
        items = "".join(f"<li>{_value_to_html(item)}</li>" for item in value)
        return "<ul>" + items + "</ul>"
    if value is None:
        return escape("None")
    return escape(str(value))


def _scalar_diff_fragment(value: Any) -> str:
    if value is None:
        return "<span class='diff-absent'>—</span>"
    return escape(str(value))


def _diff_cell_pair_for_value(left: Any, right: Any) -> Tuple[str, str]:
    """Return (html_left, html_right) for one logical field, with diff highlighting."""
    if isinstance(left, dict) and isinstance(right, dict):
        tl, tr = _render_dict_tables_diff(left, right)
        return (tl, tr)
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            f"<span class='diff-cell diff-changed'>{_value_to_html(left) if left is not None else _scalar_diff_fragment(None)}</span>",
            f"<span class='diff-cell diff-changed'>{_value_to_html(right) if right is not None else _scalar_diff_fragment(None)}</span>",
        )
    if isinstance(left, list) and isinstance(right, list):
        if _canonical_equal(left, right):
            inner = _value_to_html(left)
            return (
                f"<span class='diff-cell diff-same'>{inner}</span>",
                f"<span class='diff-cell diff-same'>{inner}</span>",
            )
        return (
            f"<span class='diff-cell diff-changed'>{_value_to_html(left)}</span>",
            f"<span class='diff-cell diff-changed'>{_value_to_html(right)}</span>",
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            f"<span class='diff-cell diff-changed'>{_value_to_html(left)}</span>",
            f"<span class='diff-cell diff-changed'>{_value_to_html(right)}</span>",
        )
    if left is None and right is None:
        s = "<span class='diff-cell diff-same'>—</span>"
        return (s, s)
    if left is None or right is None:
        return (
            f"<span class='diff-cell diff-changed'>{_scalar_diff_fragment(left)}</span>",
            f"<span class='diff-cell diff-changed'>{_scalar_diff_fragment(right)}</span>",
        )
    if _canonical_equal(left, right):
        inner = escape(str(left))
        return (f"<span class='diff-cell diff-same'>{inner}</span>", f"<span class='diff-cell diff-same'>{inner}</span>")
    return (
        f"<span class='diff-cell diff-changed'>{escape(str(left))}</span>",
        f"<span class='diff-cell diff-changed'>{escape(str(right))}</span>",
    )


def _render_dict_tables_diff(
    left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]
) -> Tuple[str, str]:
    """Pair of HTML tables with per-cell diff classes (same key order on both sides)."""
    ld = left or {}
    rd = right or {}
    keys = sorted(set(ld.keys()) | set(rd.keys()))
    rows_l: List[str] = []
    rows_r: List[str] = []
    for key in keys:
        hl, hr = _diff_cell_pair_for_value(ld.get(key), rd.get(key))
        rows_l.append(f"<tr><th>{escape(str(key))}</th><td>{hl}</td></tr>")
        rows_r.append(f"<tr><th>{escape(str(key))}</th><td>{hr}</td></tr>")
    return (
        "<table class='info-table diff-table'>" + "".join(rows_l) + "</table>",
        "<table class='info-table diff-table'>" + "".join(rows_r) + "</table>",
    )


def _card_header_title(item: Optional[Dict[str, Any]], title: str, idx: int) -> str:
    if not item:
        return f"{title} (unmatched)"
    if "__scalar__" in item and len(item) == 1:
        return str(item["__scalar__"])
    return str(
        item.get("Section")
        or item.get("Name")
        or item.get("Interface")
        or item.get("Adapter")
        or f"{title} {idx + 1}"
    )


def _render_list_as_cards_diff(items1: List[Any], items2: List[Any], title: str) -> str:
    """Side-by-side cards with row-level diff highlighting for paired items."""
    pairs = _pair_section_items(items1, items2)
    if not pairs:
        return "<p class='no-data'>No data available</p>"

    blocks: List[str] = []
    for idx, (left, right) in enumerate(pairs):
        header_l = _card_header_title(left if isinstance(left, dict) else None, title, idx)
        header_r = _card_header_title(right if isinstance(right, dict) else None, title, idx)

        if left is None and right is not None:
            body_r = _render_dict_as_table(_strip_section_field(right)) if isinstance(right, dict) else _value_to_html(right)
            blocks.append(
                "<div class='comparison-pair'>"
                "<div class='comparison-column diff-column'>"
                "<div class='card card-diff-unpaired'><div class='card-header card-diff-missing-side'>"
                f"{escape(header_l)}</div>"
                "<div class='card-body'><p class='diff-unpaired-note'>No matching entry in File 1</p></div></div></div>"
                "<div class='comparison-column diff-column'>"
                f"<div class='card card-diff-only-right'><div class='card-header'>{escape(header_r)}</div>"
                f"<div class='card-body'>{body_r}</div></div>"
                "</div></div>"
            )
            continue
        if right is None and left is not None:
            body_l = _render_dict_as_table(_strip_section_field(left)) if isinstance(left, dict) else _value_to_html(left)
            blocks.append(
                "<div class='comparison-pair'>"
                "<div class='comparison-column diff-column'>"
                f"<div class='card card-diff-only-left'><div class='card-header'>{escape(header_l)}</div>"
                f"<div class='card-body'>{body_l}</div></div></div>"
                "<div class='comparison-column diff-column'>"
                "<div class='card card-diff-unpaired'><div class='card-header card-diff-missing-side'>"
                f"{escape(header_r)}</div>"
                "<div class='card-body'><p class='diff-unpaired-note'>No matching entry in File 2</p></div></div>"
                "</div></div>"
            )
            continue

        if not isinstance(left, dict) or not isinstance(right, dict):
            blocks.append(
                "<div class='comparison-pair'>"
                "<div class='comparison-column diff-column'>"
                f"<div class='card'><div class='card-body'>{_value_to_html(left)}</div></div></div>"
                "<div class='comparison-column diff-column'>"
                f"<div class='card'><div class='card-body'>{_value_to_html(right)}</div></div></div>"
                "</div>"
            )
            continue

        tl, tr = _render_dict_tables_diff(_strip_section_field(left), _strip_section_field(right))
        pair_cls = "comparison-pair"
        if not _canonical_equal(_strip_section_field(left), _strip_section_field(right)):
            pair_cls += " comparison-pair-changed"
        blocks.append(
            f"<div class='{pair_cls}'>"
            "<div class='comparison-column diff-column'>"
            f"<div class='card'><div class='card-header'>{escape(header_l)}</div>"
            f"<div class='card-body'>{tl}</div></div></div>"
            "<div class='comparison-column diff-column'>"
            f"<div class='card'><div class='card-header'>{escape(header_r)}</div>"
            f"<div class='card-body'>{tr}</div></div>"
            "</div></div>"
        )

    return "".join(blocks)


def _extract_section_data(data: Dict[str, Any], section: str) -> List[Dict[str, Any]]:
    """Extract data for a specific section (cpu, gpu, network) from the JSON."""
    section_data = data.get(section, {})

    # Flatten the section data
    all_items = []
    if isinstance(section_data, dict):
        for _, items in section_data.items():
            if isinstance(items, list):
                all_items.extend(items)
            elif isinstance(items, dict):
                all_items.append(items)
    elif isinstance(section_data, list):
        all_items = section_data

    return all_items


def _render_comparison_section(data1: Optional[Dict[str, Any]], data2: Optional[Dict[str, Any]],
                                section: str, section_title: str) -> str:
    """Render a comparison section with paired cards and per-field diff highlighting."""
    file1_items = _extract_section_data(data1, section) if data1 else []
    file2_items = _extract_section_data(data2, section) if data2 else []

    if not file1_items and not file2_items:
        return "<p class='no-data'>No data available</p>"

    diff_body = _render_list_as_cards_diff(file1_items, file2_items, section_title)

    return (
        "<div class='comparison-diff-wrap'>"
        "<div class='comparison-file-labels'>"
        "<div class='comparison-file-label'><h3>File 1</h3></div>"
        "<div class='comparison-file-label'><h3>File 2</h3></div>"
        "</div>"
        f"{diff_body}"
        "</div>"
    )


def _render_single_section(data: Dict[str, Any], section: str, section_title: str) -> str:
    """Render a single section without comparison."""
    items = _extract_section_data(data, section)

    if items:
        return _render_list_as_cards(items, section_title)
    else:
        return "<p class='no-data'>No data available</p>"


def generate_comparison_html(file1_path: Optional[Path], file2_path: Optional[Path], output_path: Path) -> None:
    """Generate HTML comparison report from two JSON files."""

    # Load JSON files
    data1 = None
    data2 = None

    if file1_path and file1_path.exists():
        try:
            with file1_path.open("r", encoding="utf-8") as f:
                data1 = json.load(f)
        except json.JSONDecodeError as e:
            print(f"\n ERROR: Invalid JSON in file: {file1_path}")
            print(f"   Issue: {e.msg}")
            print(f"   Location: Line {e.lineno}, Column {e.colno}")
            print(f"\n   This usually means:")
            print(f"   - The data collection was interrupted (Ctrl+C, crash, or system shutdown)")
            print(f"   - The file is corrupted or incomplete")
            print(f"\n   Solutions:")
            print(f"   1. Re-run data collection: python rapido-collect.py")
            print(f"   2. Check disk space and system logs")
            print(f"   3. Try collecting specific sections only (e.g., -c -g)")
            print()
            raise SystemExit(1)
        except Exception as e:
            print(f"\n ERROR: Failed to read file: {file1_path}")
            print(f"   Reason: {str(e)}")
            raise SystemExit(1)

    if file2_path and file2_path.exists():
        try:
            with file2_path.open("r", encoding="utf-8") as f:
                data2 = json.load(f)
        except json.JSONDecodeError as e:
            print(f"\n ERROR: Invalid JSON in file: {file2_path}")
            print(f"   Issue: {e.msg}")
            print(f"   Location: Line {e.lineno}, Column {e.colno}")
            print(f"\n   This usually means:")
            print(f"   - The data collection was interrupted (Ctrl+C, crash, or system shutdown)")
            print(f"   - The file is corrupted or incomplete")
            print(f"\n   Solutions:")
            print(f"   1. Re-run data collection: python rapido-collect.py")
            print(f"   2. Check disk space and system logs")
            print(f"   3. Try collecting specific sections only (e.g., -c -g)")
            print()
            raise SystemExit(1)
        except Exception as e:
            print(f"\n ERROR: Failed to read file: {file2_path}")
            print(f"   Reason: {str(e)}")
            raise SystemExit(1)

    if not data1 and not data2:
        raise ValueError("At least one valid JSON file must be provided")

    # Determine if we're doing comparison or single file
    is_comparison = data1 is not None and data2 is not None

    # Get file names for display
    file1_name = file1_path.name if file1_path else "N/A"
    file2_name = file2_path.name if file2_path else "N/A"

    # Extract command lines from metadata
    command_line1 = ""
    command_line2 = ""
    if data1 and "_metadata" in data1:
        command_line1 = data1["_metadata"].get("command_line", "")
    if data2 and "_metadata" in data2:
        command_line2 = data2["_metadata"].get("command_line", "")

    # Check which sections have data in either file
    has_cpu = False
    has_gpu = False
    has_network = False
    has_bmc = False
    has_rocm = False
    has_microbenchmarks = False
    
    if is_comparison:
        cpu1 = _extract_section_data(data1, "cpu") if data1 else []
        cpu2 = _extract_section_data(data2, "cpu") if data2 else []
        has_cpu = bool(cpu1 or cpu2)
        
        gpu1 = _extract_section_data(data1, "gpu") if data1 else []
        gpu2 = _extract_section_data(data2, "gpu") if data2 else []
        has_gpu = bool(gpu1 or gpu2)
        
        network1 = _extract_section_data(data1, "network") if data1 else []
        network2 = _extract_section_data(data2, "network") if data2 else []
        has_network = bool(network1 or network2)
        
        bmc1 = _extract_section_data(data1, "bmc") if data1 else []
        bmc2 = _extract_section_data(data2, "bmc") if data2 else []
        has_bmc = bool(bmc1 or bmc2)
        
        rocm1 = _extract_section_data(data1, "rocm") if data1 else []
        rocm2 = _extract_section_data(data2, "rocm") if data2 else []
        has_rocm = bool(rocm1 or rocm2)
        
        microbench1 = _extract_section_data(data1, "microbenchmarks") if data1 else []
        microbench2 = _extract_section_data(data2, "microbenchmarks") if data2 else []
        has_microbenchmarks = bool(microbench1 or microbench2)
    else:
        active_data = data1 or data2
        has_cpu = bool(_extract_section_data(active_data, "cpu"))
        has_gpu = bool(_extract_section_data(active_data, "gpu"))
        has_network = bool(_extract_section_data(active_data, "network"))
        has_bmc = bool(_extract_section_data(active_data, "bmc"))
        has_rocm = bool(_extract_section_data(active_data, "rocm"))
        has_microbenchmarks = bool(_extract_section_data(active_data, "microbenchmarks"))

    # Determine which tab should be active by default (first available tab)
    first_tab = None
    if has_cpu:
        first_tab = "cpu"
    elif has_gpu:
        first_tab = "gpu"
    elif has_rocm:
        first_tab = "rocm"
    elif has_network:
        first_tab = "network"
    elif has_bmc:
        first_tab = "bmc"
    elif has_microbenchmarks:
        first_tab = "microbenchmarks"
    
    # Generate tab content only for sections that have data
    cpu_content = ""
    gpu_content = ""
    network_content = ""
    bmc_content = ""
    rocm_content = ""
    microbenchmarks_content = ""
    
    if is_comparison:
        if has_cpu:
            cpu_content = _render_comparison_section(data1, data2, "cpu", "CPU")
        if has_gpu:
            gpu_content = _render_comparison_section(data1, data2, "gpu", "GPU")
        if has_network:
            network_content = _render_comparison_section(data1, data2, "network", "Network")
        if has_bmc:
            bmc_content = _render_comparison_section(data1, data2, "bmc", "BMC")
        if has_rocm:
            rocm_content = _render_comparison_section(data1, data2, "rocm", "ROCm")
        if has_microbenchmarks:
            microbenchmarks_content = _render_comparison_section(data1, data2, "microbenchmarks", "Microbenchmarks")
    else:
        active_data = data1 or data2
        if has_cpu:
            cpu_content = _render_single_section(active_data, "cpu", "CPU")
        if has_gpu:
            gpu_content = _render_single_section(active_data, "gpu", "GPU")
        if has_network:
            network_content = _render_single_section(active_data, "network", "Network")
        if has_bmc:
            bmc_content = _render_single_section(active_data, "bmc", "BMC")
        if has_rocm:
            rocm_content = _render_single_section(active_data, "rocm", "ROCm")
        if has_microbenchmarks:
            microbenchmarks_content = _render_single_section(active_data, "microbenchmarks", "Microbenchmarks")

    # HTML template with tabs
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMD Rapido Server Information{' Comparison' if is_comparison else ''}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}

        .file-names {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 15px;
            font-size: 0.9rem;
        }}

        .file-info {{
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 4px;
        }}

        .command-toggle {{
            margin-top: 15px;
            cursor: pointer;
            font-size: 0.85rem;
            opacity: 0.8;
            user-select: none;
            transition: opacity 0.2s;
        }}

        .command-toggle:hover {{
            opacity: 1;
        }}

        .command-section {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            margin-top: 10px;
        }}

        .command-section.expanded {{
            max-height: 300px;
        }}

        .command-box {{
            background: rgba(0,0,0,0.2);
            padding: 10px 15px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            text-align: left;
            margin: 10px auto;
            max-width: 90%;
            word-break: break-all;
        }}

        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            padding: 0 30px;
            position: relative;
        }}

        .tabs::after {{
            content: "Drag tabs to reorder";
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.75rem;
            color: #6c757d;
            opacity: 0.6;
            pointer-events: none;
        }}

        .tab {{
            padding: 15px 30px;
            cursor: move;
            border: none;
            background: none;
            font-size: 1rem;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
            user-select: none;
        }}

        .tab:hover {{
            background: rgba(102, 126, 234, 0.1);
        }}

        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            background: white;
        }}

        .tab.dragging {{
            opacity: 0.5;
            transform: scale(0.95);
        }}

        .tab.drag-over {{
            background: rgba(102, 126, 234, 0.2);
            border-left: 3px solid #667eea;
        }}

        .tab-content {{
            display: none;
            padding: 30px;
            animation: fadeIn 0.3s;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        .comparison-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .comparison-column {{
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 20px;
            background: #fafafa;
        }}

        .comparison-column h3 {{
            color: #495057;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}

        .comparison-diff-wrap {{
            width: 100%;
        }}

        .diff-legend {{
            font-size: 0.9rem;
            color: #495057;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 16px;
            line-height: 1.5;
        }}

        .header .diff-legend {{
            margin: 18px auto 0;
            max-width: 900px;
            text-align: left;
            color: #212529;
            background: rgba(255, 255, 255, 0.95);
            border-color: rgba(0, 0, 0, 0.08);
        }}

        .diff-legend-swatch {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            margin: 0 4px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .diff-legend-changed {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
        }}

        .diff-legend-same {{
            background: #d4edda;
            border: 1px solid #28a745;
            color: #155724;
        }}

        .diff-legend-absent {{
            background: #e2e3e5;
            border: 1px solid #adb5bd;
            color: #383d41;
        }}

        .comparison-file-labels {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 12px;
        }}

        .comparison-file-label h3 {{
            color: #495057;
            margin: 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #667eea;
        }}

        .comparison-pair {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            align-items: start;
            margin-bottom: 20px;
        }}

        .comparison-pair-changed .card {{
            box-shadow: 0 0 0 2px rgba(255, 193, 7, 0.35);
        }}

        .diff-column {{
            min-width: 0;
        }}

        .diff-table td .diff-table {{
            margin-top: 6px;
            font-size: 0.95em;
        }}

        .diff-cell {{
            display: inline-block;
            width: 100%;
        }}

        .diff-cell.diff-changed {{
            background: #fff8e1;
            border-radius: 4px;
            padding: 4px 6px;
            box-shadow: inset 0 0 0 1px rgba(255, 193, 7, 0.45);
        }}

        .diff-cell.diff-same {{
            border-radius: 4px;
            padding: 2px 4px;
        }}

        .diff-absent {{
            color: #6c757d;
            font-weight: 600;
        }}

        .card-diff-only-left {{
            border-left: 4px solid #fd7e14;
        }}

        .card-diff-only-right {{
            border-left: 4px solid #0d6efd;
        }}

        .card-diff-unpaired {{
            opacity: 0.95;
        }}

        .card-diff-missing-side {{
            background: #6c757d !important;
        }}

        .diff-unpaired-note {{
            margin: 0;
            color: #6c757d;
            font-style: italic;
        }}

        .card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        .card-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 16px;
            font-weight: 600;
            font-size: 1.05rem;
        }}

        .card-body {{
            padding: 16px;
        }}

        .info-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .info-table th,
        .info-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e9ecef;
            text-align: left;
            vertical-align: top;
        }}

        .info-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
            width: 35%;
        }}

        .info-table td {{
            color: #212529;
        }}

        .info-table tr:last-child th,
        .info-table tr:last-child td {{
            border-bottom: none;
        }}

        .no-data {{
            text-align: center;
            padding: 40px;
            color: #6c757d;
            font-style: italic;
        }}

        ul {{
            margin: 0;
            padding-left: 20px;
        }}

        ul li {{
            margin: 5px 0;
        }}

        @media (max-width: 768px) {{
            .comparison-container {{
                grid-template-columns: 1fr;
            }}

            .comparison-file-labels,
            .comparison-pair {{
                grid-template-columns: 1fr;
            }}

            .tabs {{
                overflow-x: auto;
            }}

            .tab {{
                white-space: nowrap;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AMD Rapido Server Information{' Comparison' if is_comparison else ''}</h1>
            {f'''<div class="file-names">
                <div class="file-info"><strong>File 1:</strong> {escape(file1_name)}</div>
                <div class="file-info"><strong>File 2:</strong> {escape(file2_name)}</div>
            </div>''' if is_comparison else f'<div class="file-names"><div class="file-info">{escape(file1_name or file2_name)}</div></div>'}
            {f'''<div class="diff-legend" role="note"><strong>Diff view:</strong> <span class="diff-legend-swatch diff-legend-changed">Changed</span> <span class="diff-legend-swatch diff-legend-same">Same</span> <span class="diff-legend-swatch diff-legend-absent">Missing</span> · Cards are matched by Section / Name / Interface / Adapter when present.</div>''' if is_comparison else ''}
            {f'''<div class="command-toggle" onclick="toggleCommand()">▼ Show collection command{"s" if is_comparison else ""}</div>
            <div class="command-section" id="commandSection">
                {f'<div style="margin-bottom: 10px;"><strong>File 1 Collection:</strong></div><div class="command-box">{escape(command_line1)}</div>' if command_line1 else ''}
                {f'<div style="margin-bottom: 10px; margin-top: 15px;"><strong>File 2 Collection:</strong></div><div class="command-box">{escape(command_line2)}</div>' if command_line2 and is_comparison else ''}
            </div>''' if (command_line1 or command_line2) else ''}
        </div>

        <div class="tabs" id="tabs-container">
            {_tab_button_html("cpu", "CPU", first_tab, has_cpu)}
            {_tab_button_html("gpu", "GPU", first_tab, has_gpu)}
            {_tab_button_html("rocm", "ROCm", first_tab, has_rocm)}
            {_tab_button_html("network", "Network", first_tab, has_network)}
            {_tab_button_html("bmc", "BMC", first_tab, has_bmc)}
            {_tab_button_html("microbenchmarks", "Microbenchmarks", first_tab, has_microbenchmarks)}
        </div>

        {f'<div id="cpu" class="tab-content{" active" if first_tab == "cpu" else ""}">{cpu_content}</div>' if has_cpu else ''}
        {f'<div id="gpu" class="tab-content{" active" if first_tab == "gpu" else ""}">{gpu_content}</div>' if has_gpu else ''}
        {f'<div id="rocm" class="tab-content{" active" if first_tab == "rocm" else ""}">{rocm_content}</div>' if has_rocm else ''}
        {f'<div id="network" class="tab-content{" active" if first_tab == "network" else ""}">{network_content}</div>' if has_network else ''}
        {f'<div id="bmc" class="tab-content{" active" if first_tab == "bmc" else ""}">{bmc_content}</div>' if has_bmc else ''}
        {f'<div id="microbenchmarks" class="tab-content{" active" if first_tab == "microbenchmarks" else ""}">{microbenchmarks_content}</div>' if has_microbenchmarks else ''}
    </div>

    <script>
        function openTab(evt, tabName) {{
            // Hide all tab contents
            var tabContents = document.getElementsByClassName("tab-content");
            for (var i = 0; i < tabContents.length; i++) {{
                tabContents[i].classList.remove("active");
            }}

            // Remove active class from all tabs
            var tabs = document.getElementsByClassName("tab");
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove("active");
            }}

            // Show current tab and mark it as active
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }}

        // Drag and Drop functionality for tabs
        let draggedElement = null;

        function initDragAndDrop() {{
            const tabsContainer = document.getElementById('tabs-container');
            const tabs = tabsContainer.querySelectorAll('.tab');

            tabs.forEach(tab => {{
                tab.addEventListener('dragstart', handleDragStart);
                tab.addEventListener('dragend', handleDragEnd);
                tab.addEventListener('dragover', handleDragOver);
                tab.addEventListener('drop', handleDrop);
                tab.addEventListener('dragenter', handleDragEnter);
                tab.addEventListener('dragleave', handleDragLeave);
            }});

            // Restore saved tab order from localStorage
            restoreTabOrder();
        }}

        function handleDragStart(e) {{
            draggedElement = this;
            this.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/html', this.innerHTML);
        }}

        function handleDragEnd(e) {{
            this.classList.remove('dragging');

            // Remove drag-over class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {{
                tab.classList.remove('drag-over');
            }});

            // Save the new tab order to localStorage
            saveTabOrder();
        }}

        function handleDragOver(e) {{
            if (e.preventDefault) {{
                e.preventDefault();
            }}
            e.dataTransfer.dropEffect = 'move';
            return false;
        }}

        function handleDragEnter(e) {{
            if (this !== draggedElement) {{
                this.classList.add('drag-over');
            }}
        }}

        function handleDragLeave(e) {{
            this.classList.remove('drag-over');
        }}

        function handleDrop(e) {{
            if (e.stopPropagation) {{
                e.stopPropagation();
            }}

            if (draggedElement !== this) {{
                // Get the container
                const container = document.getElementById('tabs-container');

                // Get all tabs
                const allTabs = Array.from(container.children);

                // Get positions
                const draggedIndex = allTabs.indexOf(draggedElement);
                const targetIndex = allTabs.indexOf(this);

                // Reorder elements
                if (draggedIndex < targetIndex) {{
                    container.insertBefore(draggedElement, this.nextSibling);
                }} else {{
                    container.insertBefore(draggedElement, this);
                }}
            }}

            this.classList.remove('drag-over');
            return false;
        }}

        function saveTabOrder() {{
            const tabsContainer = document.getElementById('tabs-container');
            const tabs = tabsContainer.querySelectorAll('.tab');
            const order = Array.from(tabs).map(tab => tab.getAttribute('data-tab'));
            localStorage.setItem('tabOrder', JSON.stringify(order));
        }}

        function restoreTabOrder() {{
            const savedOrder = localStorage.getItem('tabOrder');
            if (!savedOrder) {{
                return;
            }}

            try {{
                const order = JSON.parse(savedOrder);
                const tabsContainer = document.getElementById('tabs-container');
                const tabs = Array.from(tabsContainer.querySelectorAll('.tab'));

                // Create a map of tab elements by their data-tab attribute
                const tabMap = {{}};
                tabs.forEach(tab => {{
                    tabMap[tab.getAttribute('data-tab')] = tab;
                }});

                // Reorder tabs according to saved order
                order.forEach(tabName => {{
                    if (tabMap[tabName]) {{
                        tabsContainer.appendChild(tabMap[tabName]);
                    }}
                }});
            }} catch (e) {{
                console.error('Error restoring tab order:', e);
            }}
        }}

        function toggleCommand() {{
            const section = document.getElementById('commandSection');
            const toggle = document.querySelector('.command-toggle');
            if (section.classList.contains('expanded')) {{
                section.classList.remove('expanded');
                const originalText = toggle.textContent.replace('▲ Hide', '▼ Show');
                toggle.textContent = originalText;
            }} else {{
                section.classList.add('expanded');
                const hiddenText = toggle.textContent.replace('▼ Show', '▲ Hide');
                toggle.textContent = hiddenText;
            }}
        }}

        // Initialize drag and drop when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            initDragAndDrop();
        }});
    </script>
</body>
</html>"""

    # Write HTML file
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"HTML comparison report generated: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSON server info files into an HTML comparison report with tabs."
    )
    parser.add_argument(
        "-f1",
        "--file1",
        help="Path to the first JSON file for comparison.",
    )
    parser.add_argument(
        "-f2",
        "--file2",
        help="Path to the second JSON file for comparison.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to a single JSON input file (legacy mode, use -f1 instead).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="rapido-report.html",
        help="Path for the generated HTML file (default: rapido-report.html).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Handle file paths
    file1_path = None
    file2_path = None

    if args.file1:
        file1_path = Path(args.file1).expanduser().resolve()
        if not file1_path.exists():
            raise FileNotFoundError(f"File 1 not found: {file1_path}")
    elif args.input:
        # Legacy mode: single file input
        file1_path = Path(args.input).expanduser().resolve()
        if not file1_path.exists():
            raise FileNotFoundError(f"Input file not found: {file1_path}")

    if args.file2:
        file2_path = Path(args.file2).expanduser().resolve()
        if not file2_path.exists():
            raise FileNotFoundError(f"File 2 not found: {file2_path}")

    if not file1_path and not file2_path:
        raise ValueError("At least one input file must be specified using -f1, -f2, or -i")

    output_path = Path(args.output).expanduser().resolve()

    generate_comparison_html(file1_path, file2_path, output_path)


if __name__ == "__main__":
    main()
