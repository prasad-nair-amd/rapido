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
        </div>

        <div class="tabs" id="tabs-container">
            {f'<button class="tab{" active" if first_tab == "cpu" else ""}" onclick="openTab(event, \'cpu\')" draggable="true" data-tab="cpu">CPU</button>' if has_cpu else ''}
            {f'<button class="tab{" active" if first_tab == "gpu" else ""}" onclick="openTab(event, \'gpu\')" draggable="true" data-tab="gpu">GPU</button>' if has_gpu else ''}
            {f'<button class="tab{" active" if first_tab == "rocm" else ""}" onclick="openTab(event, \'rocm\')" draggable="true" data-tab="rocm">ROCm</button>' if has_rocm else ''}
            {f'<button class="tab{" active" if first_tab == "network" else ""}" onclick="openTab(event, \'network\')" draggable="true" data-tab="network">Network</button>' if has_network else ''}
            {f'<button class="tab{" active" if first_tab == "bmc" else ""}" onclick="openTab(event, \'bmc\')" draggable="true" data-tab="bmc">BMC</button>' if has_bmc else ''}
            {f'<button class="tab{" active" if first_tab == "microbenchmarks" else ""}" onclick="openTab(event, \'microbenchmarks\')" draggable="true" data-tab="microbenchmarks">Microbenchmarks</button>' if has_microbenchmarks else ''}
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
