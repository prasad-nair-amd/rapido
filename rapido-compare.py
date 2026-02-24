#!/usr/bin/env python3
import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    return escape(str(value))


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
    """Render a comparison section with two files side by side."""
    file1_items = _extract_section_data(data1, section) if data1 else []
    file2_items = _extract_section_data(data2, section) if data2 else []

    file1_html = _render_list_as_cards(file1_items, section_title) if file1_items else "<p class='no-data'>No data available</p>"
    file2_html = _render_list_as_cards(file2_items, section_title) if file2_items else "<p class='no-data'>No data available</p>"

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
        with file1_path.open("r", encoding="utf-8") as f:
            data1 = json.load(f)

    if file2_path and file2_path.exists():
        with file2_path.open("r", encoding="utf-8") as f:
            data2 = json.load(f)

    if not data1 and not data2:
        raise ValueError("At least one valid JSON file must be provided")

    # Determine if we're doing comparison or single file
    is_comparison = data1 is not None and data2 is not None

    # Get file names for display
    file1_name = file1_path.name if file1_path else "N/A"
    file2_name = file2_path.name if file2_path else "N/A"

    # Generate tab content
    if is_comparison:
        cpu_content = _render_comparison_section(data1, data2, "cpu", "CPU")
        gpu_content = _render_comparison_section(data1, data2, "gpu", "GPU")
        network_content = _render_comparison_section(data1, data2, "network", "Network")
        rocm_content = _render_comparison_section(data1, data2, "rocm", "ROCm")
        microbenchmarks_content = _render_comparison_section(data1, data2, "microbenchmarks", "GPU Microbenchmarks")
    else:
        active_data = data1 or data2
        cpu_content = _render_single_section(active_data, "cpu", "CPU")
        gpu_content = _render_single_section(active_data, "gpu", "GPU")
        network_content = _render_single_section(active_data, "network", "Network")
        rocm_content = _render_single_section(active_data, "rocm", "ROCm")
        microbenchmarks_content = _render_single_section(active_data, "microbenchmarks", "GPU Microbenchmarks")

    # HTML template with tabs
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMD Rapido Server Information Comparison</title>
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

        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            padding: 0 30px;
        }}

        .tab {{
            padding: 15px 30px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1rem;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }}

        .tab:hover {{
            background: rgba(102, 126, 234, 0.1);
        }}

        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            background: white;
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
            <h1>AMD Rapido Server Information {'Comparison' if is_comparison else 'Report'}</h1>
            {f'''<div class="file-names">
                <div class="file-info"><strong>File 1:</strong> {escape(file1_name)}</div>
                <div class="file-info"><strong>File 2:</strong> {escape(file2_name)}</div>
            </div>''' if is_comparison else f'<div class="file-names"><div class="file-info">{escape(file1_name or file2_name)}</div></div>'}
        </div>

        <div class="tabs">
            <button class="tab active" onclick="openTab(event, 'cpu')">CPU</button>
            <button class="tab" onclick="openTab(event, 'gpu')">GPU</button>
            <button class="tab" onclick="openTab(event, 'network')">Network</button>
            <button class="tab" onclick="openTab(event, 'rocm')">ROCm</button>
            <button class="tab" onclick="openTab(event, 'microbenchmarks')">GPU Microbenchmarks</button>
        </div>

        <div id="cpu" class="tab-content active">
            {cpu_content}
        </div>

        <div id="gpu" class="tab-content">
            {gpu_content}
        </div>

        <div id="network" class="tab-content">
            {network_content}
        </div>

        <div id="rocm" class="tab-content">
            {rocm_content}
        </div>

        <div id="microbenchmarks" class="tab-content">
            {microbenchmarks_content}
        </div>
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
        default="comparison.html",
        help="Path for the generated HTML file (default: comparison.html).",
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
