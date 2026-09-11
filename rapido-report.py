#!/usr/bin/env python3
import argparse
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional


def _render_dict_as_table(data: Dict[str, Any], comparison_data: Optional[Dict[str, Any]] = None) -> str:
    """Render a dictionary as an HTML table, highlighting differences if comparison data provided."""
    rows = []
    for key, value in data.items():
        # Underscore-prefixed keys (e.g. "_raw") are machine-readable side-channel
        # data for charting, not display rows.
        if isinstance(key, str) and key.startswith("_"):
            continue

        # Check if this value differs from comparison data
        is_different = False
        if comparison_data is not None:
            comp_value = comparison_data.get(key)
            is_different = _values_differ(value, comp_value)

        highlight_class = ' class="highlight-diff"' if is_different else ''
        # Marked on the row so the "Differences only" toolbar filter can hide
        # matching rows. Only meaningful when comparison data was supplied.
        row_attr = f" data-diff='{1 if is_different else 0}'" if comparison_data is not None else ""

        if isinstance(value, (dict, list)):
            comp_nested = comparison_data.get(key) if comparison_data else None
            rows.append(
                f"<tr{row_attr}><th>{escape(str(key))}</th><td{highlight_class}>{_value_to_html(value, comp_nested)}</td></tr>"
            )
        else:
            rows.append(
                f"<tr{row_attr}><th>{escape(str(key))}</th><td{highlight_class}>{escape(str(value))}</td></tr>"
            )
    return "<table class='info-table'>" + "".join(rows) + "</table>"


def _render_list_as_cards(items: List[Any], title: str = "", comparison_items: Optional[List[Any]] = None) -> str:
    """Render a list of dictionaries as cards, optionally comparing with another list."""
    if not items:
        return "<p class='no-data'>No data available</p>"

    cards = []
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            # Check for Section field first (for ROCm data), then other identifiers
            card_title = item.get("Section") or item.get("Name") or item.get("Interface") or item.get("Adapter") or f"{title} {idx + 1}"

            # Create a copy of the item without the Section field for display
            display_item = {k: v for k, v in item.items() if k != "Section"}

            # Find matching comparison item by card_title or index
            comparison_item = None
            if comparison_items:
                # Try to match by the same identifier
                for comp_item in comparison_items:
                    if isinstance(comp_item, dict):
                        comp_title = comp_item.get("Section") or comp_item.get("Name") or comp_item.get("Interface") or comp_item.get("Adapter")
                        if comp_title == card_title:
                            comparison_item = {k: v for k, v in comp_item.items() if k != "Section"}
                            break
                # Fallback to index-based matching
                if comparison_item is None and idx < len(comparison_items):
                    if isinstance(comparison_items[idx], dict):
                        comparison_item = {k: v for k, v in comparison_items[idx].items() if k != "Section"}

            card_content = _render_dict_as_table(display_item, comparison_item)
            # data-has-diff lets the toolbar's "Differences only" filter work
            # without re-deriving the comparison client-side. A card with no
            # counterpart at all is itself a difference - it exists in one file
            # only - so it must not be filtered away as "same".
            unmatched = comparison_items is not None and comparison_item is None
            has_diff = "1" if unmatched or 'class="highlight-diff"' in card_content else "0"
            cards.append(
                f"<div class='card' data-has-diff='{has_diff}'>"
                f"<div class='card-header' onclick='toggleCard(this)' onkeydown='cardKey(event, this)' role='button' tabindex='0'>"
                f"{escape(str(card_title))}</div>"
                f"<div class='card-body'>{card_content}</div>"
                f"</div>"
            )
        else:
            cards.append(f"<div class='card'><div class='card-body'>{escape(str(item))}</div></div>")

    return "".join(cards)


def _values_differ(value1: Any, value2: Any) -> bool:
    """Check if two values are different, handling various data types."""
    if value1 is None and value2 is None:
        return False
    if value1 is None or value2 is None:
        return True

    # Normalize string comparison (handle different types)
    str1 = str(value1).strip()
    str2 = str(value2).strip()

    # For numeric comparisons, try to compare as numbers
    try:
        # Extract numbers from strings like "16 GB" or "3.5 GHz"
        import re
        num1 = re.findall(r'[-+]?\d*\.?\d+', str1)
        num2 = re.findall(r'[-+]?\d*\.?\d+', str2)
        if num1 and num2:
            return num1 != num2
    except:
        pass

    return str1 != str2


def _value_to_html(value: Any, comparison_value: Any = None) -> str:
    """Recursively convert Python data structures to HTML fragments."""
    if isinstance(value, dict):
        comp_dict = comparison_value if isinstance(comparison_value, dict) else None
        return _render_dict_as_table(value, comp_dict)
    if isinstance(value, list):
        comp_list = comparison_value if isinstance(comparison_value, list) else None
        items = []
        for idx, item in enumerate(value):
            comp_item = comp_list[idx] if comp_list and idx < len(comp_list) else None
            items.append(f"<li>{_value_to_html(item, comp_item)}</li>")
        return "<ul>" + "".join(items) + "</ul>"
    return escape(str(value))


# ---------------------------------------------------------------------------
# Microbenchmark dashboard
#
# Charts are emitted as inline SVG built here in Python: no JS charting library,
# so the report stays a single self-contained file that renders with no network.
# Everything below consumes the "_raw" numeric blocks that rapido-collect.py
# attaches to its benchmark cards; cards without "_raw" are simply skipped.
# ---------------------------------------------------------------------------

# A GPU whose result deviates from the node median by more than this is flagged.
# In a healthy node of identical parts the spread is a couple of percent; 15%
# is comfortably outside run-to-run noise but still catches a half-speed link.
OUTLIER_THRESHOLD = 0.15

# Cool (slow) to warm (fast). Interpolated between for the heatmap fill.
_HEAT_COLORS = [(13, 71, 161), (2, 136, 209), (0, 150, 136), (255, 179, 0), (216, 67, 21)]


def _is_num(value: Any) -> bool:
    """True for a real, finite number.

    A stray NaN or inf in the JSON (json.load accepts both) would otherwise
    propagate into the colour maths and abort the whole report, so numbers are
    checked with this rather than a bare isinstance everywhere below.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _heat_color(fraction: float) -> str:
    """Map 0.0-1.0 onto the heatmap gradient, returning a CSS rgb() string."""
    if not math.isfinite(fraction):
        fraction = 0.0
    fraction = min(max(fraction, 0.0), 1.0)
    pos = fraction * (len(_HEAT_COLORS) - 1)
    low = int(pos)
    high = min(low + 1, len(_HEAT_COLORS) - 1)
    t = pos - low
    r, g, b = (round(_HEAT_COLORS[low][i] + (_HEAT_COLORS[high][i] - _HEAT_COLORS[low][i]) * t)
               for i in range(3))
    return f"rgb({r},{g},{b})"


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _find_raw(items: List[Any], section_contains: str) -> List[Dict[str, Any]]:
    """Return the '_raw' blocks of every card whose Section contains the given text.

    Section titles carry the GPU index ("GPU 3 Kernel Benchmarks"), so this
    matches on the stable part of the title rather than the whole string.
    """
    found = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("_raw"), dict):
            continue
        if section_contains in str(item.get("Section", "")):
            found.append(item["_raw"])
    return found


def _render_p2p_heatmap(raw: Dict[str, Any]) -> str:
    """Render the N x N P2P bandwidth matrix as a colour-scaled SVG grid."""
    ids = raw.get("gpu_ids") or []
    matrix = raw.get("bandwidth_gbps") or []
    if not ids or not matrix:
        return ""

    n = len(ids)
    measured = [v for row in matrix for v in row if _is_num(v)]
    if not measured:
        return ""

    # Per-hop interconnect type, when the collector managed to read it. The
    # interesting case is the minority one: a single PCIe hop in an otherwise
    # all-XGMI mesh explains a slow link far better than the bandwidth alone.
    links = raw.get("link_types") or []

    def link_of(i: int, j: int) -> str:
        if i < len(links) and isinstance(links[i], list) and j < len(links[i]):
            value = links[i][j]
            if isinstance(value, str) and value:
                return value
        return ""

    seen_links = [link_of(i, j) for i in range(n) for j in range(n) if i != j and link_of(i, j)]
    dominant = max(set(seen_links), key=seen_links.count) if seen_links else ""
    # Scale from zero rather than from the observed minimum. Min-max scaling
    # would stretch a couple of percent of run-to-run noise across the whole
    # gradient, making a healthy uniform node look alarmingly varied.
    low, high = 0.0, max(measured)
    span = high or 1.0

    cell = 58 if n <= 10 else 40
    label_w, label_h = 58, 30
    width = label_w + n * cell + 8
    height = label_h + n * cell + 48

    parts = [
        f"<svg class='chart' viewBox='0 0 {width} {height}' width='100%' "
        f"role='img' aria-label='GPU peer-to-peer bandwidth matrix in GB/s'>"
    ]
    for j, dst in enumerate(ids):
        x = label_w + j * cell + cell / 2
        parts.append(f"<text x='{x:.1f}' y='{label_h - 10}' class='ax' text-anchor='middle'>{escape(str(dst))}</text>")
    for i, src in enumerate(ids):
        y = label_h + i * cell + cell / 2
        parts.append(f"<text x='{label_w - 10}' y='{y + 4:.1f}' class='ax' text-anchor='end'>GPU {escape(str(src))}</text>")

    for i in range(n):
        row = matrix[i] if i < len(matrix) else []
        for j in range(n):
            value = row[j] if j < len(row) else None
            x = label_w + j * cell
            y = label_h + i * cell
            # Cell labels are white by default, which disappears on the light
            # diagonal and "not measured" fills, so those get dark text.
            ink = "#ffffff"
            if i == j:
                fill, text, title = "#e9ecef", "—", f"GPU {ids[i]} (self)"
                ink = "#343a40"
            elif _is_num(value):
                fill = _heat_color((value - low) / span)
                text = f"{value:.0f}"
                title = f"GPU {ids[i]} → GPU {ids[j]}: {value:.2f} GB/s"
            else:
                fill, text, title = "#f8d7da", "n/a", f"GPU {ids[i]} → GPU {ids[j]}: not measured"
                ink = "#842029"

            link = link_of(i, j) if i != j else ""
            if link:
                title += f" over {link}"
            # Only the odd hop out gets a printed tag; tagging all 56 cells of a
            # uniform mesh would be noise. The dominant type is named in the caption.
            tag = ""
            if link and dominant and link != dominant:
                tag = (f"<text x='{x + (cell - 2) / 2:.1f}' y='{y + cell - 8:.1f}' "
                       f"class='cell link' fill='{ink}' text-anchor='middle'>{escape(link)}</text>")

            parts.append(
                f"<g><title>{escape(title)}</title>"
                f"<rect x='{x}' y='{y}' width='{cell - 2}' height='{cell - 2}' rx='3' fill='{fill}'/>"
                f"<text x='{x + (cell - 2) / 2:.1f}' y='{y + cell / 2 + 4:.1f}' class='cell' "
                f"fill='{ink}' text-anchor='middle'>{escape(text)}</text>{tag}</g>"
            )

    # Gradient legend. Sized to the grid so it cannot run past the viewBox on a
    # small node; the swatch count follows from the width rather than the reverse.
    ly = label_h + n * cell + 16
    legend_w = min(200, n * cell)
    steps = max(8, int(legend_w // 5))
    swatch = legend_w / steps
    for step in range(steps):
        parts.append(
            f"<rect x='{label_w + step * swatch:.2f}' y='{ly}' width='{swatch:.2f}' height='12' "
            f"fill='{_heat_color(step / (steps - 1))}'/>"
        )
    parts.append(f"<text x='{label_w}' y='{ly + 28}' class='ax'>{low:.0f} GB/s</text>")
    parts.append(f"<text x='{label_w + legend_w}' y='{ly + 28}' class='ax' text-anchor='end'>{high:.0f} GB/s</text>")
    parts.append("</svg>")
    return "".join(parts)


def _render_bar_chart(labels: List[str], values: List[Optional[float]], unit: str) -> str:
    """Horizontal bars with a median marker; bars far off the median are flagged."""
    present = [v for v in values if _is_num(v)]
    if not present:
        return ""
    high = max(present) or 1.0
    median = _median(present)

    row_h, bar_w, label_w = 26, 300, 92
    width = label_w + bar_w + 96
    height = len(values) * row_h + 12

    parts = [f"<svg class='chart' viewBox='0 0 {width} {height}' width='100%' role='img'>"]
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = idx * row_h + 6
        parts.append(
            f"<text x='0' y='{y + 14}' class='ax'>{escape(str(label))}</text>"
        )
        if not _is_num(value):
            parts.append(f"<text x='{label_w}' y='{y + 14}' class='ax'>not measured</text>")
            continue
        w = max((value / high) * bar_w, 1.0)
        off = median and abs(value - median) / median > OUTLIER_THRESHOLD
        fill = "#d84315" if off else "#4c6ef5"
        parts.append(
            f"<rect x='{label_w}' y='{y + 2}' width='{w:.1f}' height='{row_h - 10}' rx='2' fill='{fill}'/>"
            f"<text x='{label_w + w + 6:.1f}' y='{y + 14}' class='val'>{value:,.2f}"
            f"{' ⚠' if off else ''}</text>"
        )
    if median is not None and high:
        mx = label_w + (median / high) * bar_w
        parts.append(
            f"<line class='med' x1='{mx:.1f}' y1='4' x2='{mx:.1f}' y2='{height - 6}' "
            f"stroke-width='1' stroke-dasharray='3,3'><title>"
            f"median {median:,.2f} {escape(unit)}</title></line>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _uniformity_warnings(metric: str, labels: List[str], values: List[Optional[float]], unit: str) -> List[str]:
    """Describe every GPU deviating from the node median by more than the threshold."""
    present = [v for v in values if _is_num(v)]
    median = _median(present)
    if median is None or median == 0 or len(present) < 3:
        return []
    warnings = []
    for label, value in zip(labels, values):
        if not _is_num(value):
            continue
        delta = (value - median) / median
        if abs(delta) > OUTLIER_THRESHOLD:
            warnings.append(
                f"{label}: {metric} {value:,.2f} {unit} is {delta * 100:+.1f}% "
                f"vs. node median {median:,.2f} {unit}"
            )
    return warnings


# (raw key, chart heading, unit) for the per-GPU bar charts.
_GPU_BAR_METRICS = [
    ("memory_bandwidth_gbps", "Memory Bandwidth", "GB/s"),
    ("gemm_fp32_gflops", "GEMM FP32", "GFLOPS"),
    ("gemm_fp64_gflops", "GEMM FP64", "GFLOPS"),
    ("fma_tflops", "FMA Throughput", "TFLOPS"),
    ("vector_add_gflops", "Vector Add", "GFLOPS"),
    ("convolution_gflops", "1D Convolution", "GFLOPS"),
]

_HOST_BAR_METRICS = [
    ("h2d_pinned_gbps", "Host→Device (Pinned)", "GB/s"),
    ("d2h_pinned_gbps", "Device→Host (Pinned)", "GB/s"),
]


_NO_DASH = "<p class='no-data'>No chartable benchmark data</p>"

# How the collector's failure reasons read to someone looking at the report.
_FAILURE_REASONS = {
    "not_found": "not installed",
    "timeout": "timed out",
    "error": "returned an error",
}


def _render_command_failures(sources: List[Any]) -> str:
    """Banner listing the commands that failed during collection, and why.

    rapido-collect.py records every failure with a reason, but a reader looking at a
    gap in the report otherwise cannot tell whether the tool was missing, hung, or
    errored. `sources` is a list of (label, data) pairs; the label is only printed in
    comparison mode, where a failure may apply to just one of the two files.
    """
    blocks: List[str] = []
    for label, data in sources:
        if not isinstance(data, dict):
            continue
        failures = data.get("command_failures")
        if not isinstance(failures, dict) or not failures:
            continue
        rows = []
        for command, info in failures.items():
            info = info if isinstance(info, dict) else {}
            reason = _FAILURE_REASONS.get(info.get("reason"), info.get("reason") or "failed")
            detail = info.get("detail") or ""
            rows.append(
                f"<li><code>{escape(str(command))}</code> — {escape(reason)}"
                + (f" <span class='detail'>({escape(str(detail))})</span>" if detail else "")
                + "</li>"
            )
        heading = f"Collection warnings ({escape(label)})" if label else "Collection warnings"
        blocks.append(
            f"<div class='failures'><strong>{heading}</strong>"
            f"<p class='caption'>These commands did not complete, so the sections that "
            f"depend on them may be missing or incomplete.</p><ul>{''.join(rows)}</ul></div>"
        )
    return "".join(blocks)


def _render_microbench_dashboard(items: List[Any]) -> str:
    """Build the charts panel that heads the Microbenchmarks tab.

    Returns "" when the JSON predates the "_raw" numerics, so older files still
    render exactly as before, just without charts.
    """
    panels: List[str] = []
    warnings: List[str] = []

    p2p = _find_raw(items, "GPU P2P Bandwidth Matrix")
    if p2p:
        heatmap = _render_p2p_heatmap(p2p[0])
        if heatmap:
            mean = p2p[0].get("mean_gbps")
            low = p2p[0].get("min_gbps")
            caption = "Peer-to-peer bandwidth, source GPU (rows) to destination GPU (columns)."
            # Name the interconnect, and call out any hop that is not on it: on an
            # Instinct node a stray PCIe link where XGMI is expected is a real defect.
            link_rows = p2p[0].get("link_types") or []
            kinds: Dict[str, int] = {}
            for i, row in enumerate(link_rows):
                if not isinstance(row, list):
                    continue
                for j, kind in enumerate(row):
                    if i != j and isinstance(kind, str) and kind:
                        kinds[kind] = kinds.get(kind, 0) + 1
            if kinds:
                if len(kinds) == 1:
                    only = next(iter(kinds))
                    caption += f" All {kinds[only]} links are {escape(only)}."
                else:
                    main = max(kinds, key=lambda k: kinds[k])
                    others = ", ".join(f"{count}x {escape(k)}"
                                       for k, count in sorted(kinds.items()) if k != main)
                    caption += (f" Mostly {escape(main)} ({kinds[main]} links); "
                                f"labelled cells differ: {others}.")
                    warnings.append(
                        f"Mixed GPU interconnect: {others} alongside {kinds[main]}x {main}"
                    )
            if mean and low and (mean - low) / mean > OUTLIER_THRESHOLD:
                warnings.append(
                    f"Slowest P2P link {low:.2f} GB/s is {((low - mean) / mean) * 100:+.1f}% "
                    f"vs. the mean of {mean:.2f} GB/s"
                )
            panels.append(
                f"<div class='panel'><h4>P2P Bandwidth Heatmap</h4>"
                f"<p class='caption'>{caption}</p>{heatmap}</div>"
            )

    def bar_panels(raws: List[Dict[str, Any]], metrics, label_prefix: str) -> None:
        if not raws:
            return
        raws = sorted(raws, key=lambda r: (r.get("gpu_id") if isinstance(r.get("gpu_id"), int) else 0))
        labels = [f"GPU {r.get('gpu_id', '?')}" for r in raws]
        for key, heading, unit in metrics:
            values = [r.get(key) for r in raws]
            chart = _render_bar_chart(labels, values, unit)
            if not chart:
                continue
            panels.append(
                f"<div class='panel'><h4>{escape(heading)} <span class='unit'>({escape(unit)})</span></h4>{chart}</div>"
            )
            warnings.extend(_uniformity_warnings(f"{label_prefix}{heading}", labels, values, unit))

    bar_panels(_find_raw(items, "Kernel Benchmarks"), _GPU_BAR_METRICS, "")
    bar_panels(_find_raw(items, "Host Transfer Bandwidth"), _HOST_BAR_METRICS, "Host transfer ")

    if not panels:
        return ""

    warn_html = ""
    if warnings:
        rows = "".join(f"<li>{escape(w)}</li>" for w in warnings)
        warn_html = (
            f"<div class='uniformity-warn'><strong>Intra-node uniformity: "
            f"{len(warnings)} outlier(s) beyond ±{OUTLIER_THRESHOLD * 100:.0f}% of the median</strong>"
            f"<ul>{rows}</ul></div>"
        )

    return (
        "<div class='dashboard'>"
        "<div class='dashboard-header'>Microbenchmark Dashboard</div>"
        f"{warn_html}"
        f"<div class='panels'>{''.join(panels)}</div>"
        "</div>"
    )


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
    """Render a comparison section with two files side by side, highlighting differences."""
    file1_items = _extract_section_data(data1, section) if data1 else []
    file2_items = _extract_section_data(data2, section) if data2 else []

    # Render with cross-comparison for highlighting
    file1_html = _render_list_as_cards(file1_items, section_title, file2_items) if file1_items else "<p class='no-data'>No data available</p>"
    file2_html = _render_list_as_cards(file2_items, section_title, file1_items) if file2_items else "<p class='no-data'>No data available</p>"

    return (
        "<div class='comparison-container'>"
        "<div class='comparison-column'>"
        f"<h3>File 1</h3>"
        f"{file1_html}"
        "</div>"
        "<div class='comparison-column'>"
        f"<h3>File 2</h3>"
        f"{file2_html}"
        "</div>"
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

    # Collection problems, surfaced at the top of the report. The collector records
    # why each command failed precisely so the reader can tell a missing tool apart
    # from a hung or erroring one instead of just seeing an absent section.
    failures_banner = _render_command_failures(
        [(file1_name, data1), (file2_name, data2)] if is_comparison else
        [("", data1 or data2)]
    )

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
            # Dashboards sit side by side above the cards, mirroring the card layout.
            dash1 = _render_microbench_dashboard(microbench1)
            dash2 = _render_microbench_dashboard(microbench2)
            dashboards = ""
            if dash1 or dash2:
                dashboards = (
                    "<div class='comparison-container'>"
                    f"<div class='comparison-column'><h3>File 1</h3>{dash1 or _NO_DASH}</div>"
                    f"<div class='comparison-column'><h3>File 2</h3>{dash2 or _NO_DASH}</div>"
                    "</div>"
                )
            microbenchmarks_content = dashboards + _render_comparison_section(
                data1, data2, "microbenchmarks", "Microbenchmarks")
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
            microbench_items = _extract_section_data(active_data, "microbenchmarks")
            microbenchmarks_content = (
                _render_microbench_dashboard(microbench_items)
                + _render_single_section(active_data, "microbenchmarks", "Microbenchmarks")
            )

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

        .highlight-diff {{
            background-color: #fff9c4;
            font-weight: 500;
        }}

        /* ---- Toolbar ---- */
        .toolbar {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
            padding: 12px 30px;
            background: #f1f3f5;
            border-bottom: 1px solid #dee2e6;
        }}

        .tb-search {{
            flex: 1 1 240px;
            min-width: 180px;
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 0.9rem;
            background: white;
            color: #212529;
        }}

        .tb-btn {{
            padding: 8px 14px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background: white;
            color: #495057;
            font-size: 0.88rem;
            cursor: pointer;
        }}

        .tb-btn:hover {{
            background: #e9ecef;
        }}

        .tb-check {{
            font-size: 0.88rem;
            color: #495057;
            cursor: pointer;
            user-select: none;
        }}

        .tb-count {{
            font-size: 0.82rem;
            color: #6c757d;
            margin-left: auto;
        }}

        .card-header {{
            cursor: pointer;
        }}

        .card.collapsed .card-body {{
            display: none;
        }}

        .filtered-out {{
            display: none !important;
        }}

        /* ---- Dark mode ---- */
        body.dark {{
            background: #16181d;
        }}

        body.dark .container,
        body.dark .card,
        body.dark .dashboard,
        body.dark .tb-search,
        body.dark .tb-btn {{
            background: #1f2228;
            color: #e4e6eb;
        }}

        body.dark .comparison-column,
        body.dark .toolbar {{
            background: #24272e;
            border-color: #3a3f47;
        }}

        body.dark .info-table th {{
            background: #2a2e36;
            color: #c9ccd1;
        }}

        body.dark .info-table td,
        body.dark .comparison-column h3,
        body.dark .panel h4,
        body.dark .tb-check {{
            color: #e4e6eb;
        }}

        body.dark .info-table th,
        body.dark .info-table td,
        body.dark .card,
        body.dark .dashboard {{
            border-color: #3a3f47;
        }}

        body.dark .highlight-diff {{
            background-color: #4d4426;
            color: #ffeaa0;
        }}

        body.dark .chart text.ax,
        body.dark .chart text.val {{
            fill: #c9ccd1;
        }}

        body.dark .chart line.med {{
            stroke: #8b9199;
        }}

        body.dark .dashboard {{
            background: #21252b;
        }}

        body.dark .panel .unit,
        body.dark .panel .caption {{
            color: #9aa0a6;
        }}

        body.dark .uniformity-warn {{
            background: #3a2a18;
            color: #f3c99a;
        }}

        body.dark .failures {{
            background: #3a1e1c;
            color: #f4b6b0;
        }}

        body.dark .failures code {{
            background: rgba(255, 255, 255, 0.1);
        }}

        /* ---- Microbenchmark dashboard ---- */
        .dashboard {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            margin-bottom: 24px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        .dashboard-header {{
            background: linear-gradient(135deg, #232526 0%, #414345 100%);
            color: white;
            padding: 12px 16px;
            font-weight: 600;
            font-size: 1.05rem;
        }}

        .panels {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
            padding: 16px;
        }}

        .panel h4 {{
            margin: 0 0 4px 0;
            color: #343a40;
            font-size: 0.95rem;
        }}

        .panel .unit {{
            color: #6c757d;
            font-weight: 400;
        }}

        .panel .caption {{
            margin: 0 0 8px 0;
            color: #6c757d;
            font-size: 0.82rem;
        }}

        .chart {{
            display: block;
            max-width: 100%;
            height: auto;
            overflow: visible;
        }}

        .chart text.ax {{
            font: 11px system-ui, sans-serif;
            fill: #495057;
        }}

        .chart text.val {{
            font: 11px system-ui, sans-serif;
            fill: #212529;
        }}

        /* Cell fill colour is set per-cell inline, since it depends on how
           light the swatch underneath it is. */
        .chart text.cell {{
            font: 11px system-ui, sans-serif;
            font-weight: 600;
        }}

        /* Interconnect tag under the number, only drawn on the odd hop out. */
        .chart text.cell.link {{
            font-size: 8px;
            font-weight: 700;
            letter-spacing: 0.03em;
        }}

        .chart line.med {{
            stroke: #495057;
        }}

        .uniformity-warn {{
            background: #fff4e5;
            border-left: 4px solid #d84315;
            margin: 16px 16px 0 16px;
            padding: 12px 16px;
            color: #5f2c0a;
            font-size: 0.9rem;
        }}

        .uniformity-warn ul {{
            margin: 8px 0 0 0;
        }}

        .failures {{
            background: #fdecea;
            border-left: 4px solid #c62828;
            border-radius: 4px;
            margin-bottom: 16px;
            padding: 12px 16px;
            color: #5f1a16;
            font-size: 0.9rem;
        }}

        .failures ul {{
            margin: 8px 0 0 18px;
        }}

        .failures li {{
            margin-bottom: 3px;
        }}

        .failures code {{
            background: rgba(0, 0, 0, 0.06);
            border-radius: 3px;
            padding: 1px 5px;
            font-size: 0.85rem;
        }}

        .failures .detail {{
            opacity: 0.8;
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
            {f'''<div class="command-toggle" onclick="toggleCommand()">▼ Show collection command{"s" if is_comparison else ""}</div>
            <div class="command-section" id="commandSection">
                {f'<div style="margin-bottom: 10px;"><strong>File 1 Collection:</strong></div><div class="command-box">{escape(command_line1)}</div>' if command_line1 else ''}
                {f'<div style="margin-bottom: 10px; margin-top: 15px;"><strong>File 2 Collection:</strong></div><div class="command-box">{escape(command_line2)}</div>' if command_line2 and is_comparison else ''}
            </div>''' if (command_line1 or command_line2) else ''}
        </div>

        {failures_banner}

        <div class="tabs" id="tabs-container">
            {f'<button class="tab{" active" if first_tab == "cpu" else ""}" onclick="openTab(event, ' + "'cpu'" + ')" draggable="true" data-tab="cpu">CPU</button>' if has_cpu else ''}
            {f'<button class="tab{" active" if first_tab == "gpu" else ""}" onclick="openTab(event, ' + "'gpu'" + ')" draggable="true" data-tab="gpu">GPU</button>' if has_gpu else ''}
            {f'<button class="tab{" active" if first_tab == "rocm" else ""}" onclick="openTab(event, ' + "'rocm'" + ')" draggable="true" data-tab="rocm">ROCm</button>' if has_rocm else ''}
            {f'<button class="tab{" active" if first_tab == "network" else ""}" onclick="openTab(event, ' + "'network'" + ')" draggable="true" data-tab="network">Network</button>' if has_network else ''}
            {f'<button class="tab{" active" if first_tab == "bmc" else ""}" onclick="openTab(event, ' + "'bmc'" + ')" draggable="true" data-tab="bmc">BMC</button>' if has_bmc else ''}
            {f'<button class="tab{" active" if first_tab == "microbenchmarks" else ""}" onclick="openTab(event, ' + "'microbenchmarks'" + ')" draggable="true" data-tab="microbenchmarks">Microbenchmarks</button>' if has_microbenchmarks else ''}
        </div>

        <div class="toolbar">
            <input type="search" id="cardSearch" class="tb-search" placeholder="Search cards and values…"
                   aria-label="Filter cards" oninput="applyFilters()">
            {'''<label class="tb-check"><input type="checkbox" id="diffOnly" onchange="applyFilters()"> Differences only</label>''' if is_comparison else ''}
            <button class="tb-btn" onclick="toggleAllCards()" id="collapseAllBtn">Collapse all</button>
            <button class="tb-btn" onclick="toggleDarkMode()" id="darkBtn">Dark mode</button>
            <span class="tb-count" id="filterCount"></span>
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

        // ---- Toolbar: search, diff-only, collapse, dark mode ----

        // Combined filter pass. Runs over the active tab only, since the other
        // tabs are display:none anyway and a full-document pass is wasteful on
        // reports with hundreds of ROCm package rows.
        function applyFilters() {{
            const term = (document.getElementById('cardSearch').value || '').toLowerCase().trim();
            const diffBox = document.getElementById('diffOnly');
            const diffOnly = diffBox ? diffBox.checked : false;
            const active = document.querySelector('.tab-content.active');
            if (!active) return;

            let shown = 0;
            const cards = active.querySelectorAll('.card');
            cards.forEach(card => {{
                // Row-level diff filtering first, so the card's visible text
                // reflects what the search then matches against.
                card.querySelectorAll('tr[data-diff]').forEach(row => {{
                    row.classList.toggle('filtered-out', diffOnly && row.dataset.diff !== '1');
                }});

                let visible = true;
                if (diffOnly && card.dataset.hasDiff === '0') visible = false;
                if (visible && term) {{
                    // Match against what is actually on screen: textContent would
                    // still include rows the diff filter just hid.
                    let text = card.querySelector('.card-header').textContent;
                    card.querySelectorAll('tr').forEach(row => {{
                        if (!row.classList.contains('filtered-out')) text += ' ' + row.textContent;
                    }});
                    visible = text.toLowerCase().indexOf(term) !== -1;
                }}
                card.classList.toggle('filtered-out', !visible);
                if (visible) shown++;
            }});

            const counter = document.getElementById('filterCount');
            counter.textContent = (term || diffOnly)
                ? shown + ' of ' + cards.length + ' cards shown'
                : '';
        }}

        function toggleCard(header) {{
            header.parentElement.classList.toggle('collapsed');
        }}

        // Space/Enter on a focused header, since the header is a div with
        // role="button" and gets no native keyboard activation.
        function cardKey(event, header) {{
            if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {{
                event.preventDefault();
                toggleCard(header);
            }}
        }}

        function toggleAllCards() {{
            const active = document.querySelector('.tab-content.active');
            if (!active) return;
            const cards = active.querySelectorAll('.card');
            // Derive the direction from this tab's own state rather than from the
            // button label, which is shared across tabs.
            let collapsed = 0;
            cards.forEach(card => {{ if (card.classList.contains('collapsed')) collapsed++; }});
            const collapse = collapsed < cards.length;
            cards.forEach(card => {{ card.classList.toggle('collapsed', collapse); }});
            syncCollapseLabel();
        }}

        // Label reflects what the button will do next for the active tab.
        function syncCollapseLabel() {{
            const btn = document.getElementById('collapseAllBtn');
            const active = document.querySelector('.tab-content.active');
            if (!btn || !active) return;
            const cards = active.querySelectorAll('.card');
            let collapsed = 0;
            cards.forEach(card => {{ if (card.classList.contains('collapsed')) collapsed++; }});
            btn.textContent = (cards.length && collapsed === cards.length) ? 'Expand all' : 'Collapse all';
        }}

        function toggleDarkMode() {{
            const on = document.body.classList.toggle('dark');
            document.getElementById('darkBtn').textContent = on ? 'Light mode' : 'Dark mode';
            try {{ localStorage.setItem('rapidoDark', on ? '1' : '0'); }} catch (e) {{}}
        }}

        // Deep links: #gpu etc. select that tab on load, and selecting a tab
        // updates the fragment so the URL can be shared.
        function selectTabByName(name) {{
            // Tab ids are plain slugs; anything else is not a tab and must not
            // reach querySelector, where a quote would throw a syntax error.
            if (!/^[A-Za-z0-9_-]+$/.test(name)) return false;
            const btn = document.querySelector('.tab[data-tab="' + name + '"]');
            const panel = document.getElementById(name);
            if (!btn || !panel) return false;
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            panel.classList.add('active');
            btn.classList.add('active');
            return true;
        }}

        document.addEventListener('DOMContentLoaded', function() {{
            initDragAndDrop();

            try {{
                if (localStorage.getItem('rapidoDark') === '1') toggleDarkMode();
            }} catch (e) {{}}

            const hash = (location.hash || '').replace('#', '');
            if (hash) selectTabByName(hash);
            syncCollapseLabel();

            // Keep the fragment in sync and re-apply filters on tab switches.
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.addEventListener('click', function() {{
                    const name = this.getAttribute('data-tab');
                    // replaceState throws on a file:// URL in some browsers;
                    // a failed deep link must not take the rest of this handler
                    // (and therefore the filters) down with it.
                    if (name) {{
                        try {{ history.replaceState(null, '', '#' + name); }} catch (e) {{}}
                    }}
                    syncCollapseLabel();
                    applyFilters();
                }});
            }});
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
