"""
Interactive HTML and Plotly Visualizer for TileLens Roofline and Memory Tile Simulation.
"""

import json
from typing import List, Optional
from tilelens.hardware.spec import HardwareSpec, Precision
from tilelens.core.roofline import RooflineResult, BoundType
from tilelens.core.tile_simulator import TileSimulationResult


def generate_interactive_dashboard_html(
    roofline_results: List[RooflineResult],
    tile_results: Optional[List[TileSimulationResult]] = None,
    title: str = "TileLens Hardware-Software Co-Design Dashboard",
) -> str:
    """
    Generates a standalone, beautiful HTML dashboard with interactive SVG/Canvas Roofline curves,
    SRAM memory hierarchy breakdowns, and hardware comparison charts.
    """
    # Prepare JSON serializable data for embedded interactive charts
    devices_data = []
    for r in roofline_results:
        devices_data.append({
            "name": r.hardware_name,
            "precision": r.precision.value.upper(),
            "peak_tflops": r.peak_tflops,
            "bandwidth_gbs": r.memory_bandwidth_gbs,
            "ridge_point": r.ridge_point,
            "op_intensity": r.operational_intensity,
            "attainable_tflops": r.attainable_tflops,
            "exec_time_ms": r.execution_time_ms,
            "bound_type": r.bound_type.value,
            "compute_efficiency": r.compute_efficiency_pct,
            "bandwidth_utilization": r.bandwidth_utilization_pct,
        })

    tiles_data = []
    if tile_results:
        for t in tile_results:
            tiles_data.append({
                "hardware": t.hardware_name,
                "gemm": f"{t.gemm_m}x{t.gemm_n}x{t.gemm_k}",
                "tile_m": t.tile_config.tile_m,
                "tile_n": t.tile_config.tile_n,
                "tile_k": t.tile_config.tile_k,
                "stages": t.tile_config.pipeline_stages,
                "sram_req_kb": t.total_sram_required_kb,
                "sram_limit_kb": t.hardware_sram_limit_kb,
                "sram_util_pct": t.sram_utilization_pct,
                "sram_overflow": t.sram_overflow,
                "grid_tiles": t.total_grid_tiles,
                "waves": t.total_waves,
                "hbm_mb": t.actual_hbm_bytes / 1e6,
                "ideal_hbm_mb": t.ideal_hbm_bytes / 1e6,
                "reuse_factor": t.tile_reuse_factor,
            })

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-heading: #58a6ff;
            --accent-green: #2ea043;
            --accent-red: #f85149;
            --accent-purple: #bc8cff;
            --accent-orange: #d29922;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            margin: 0;
            color: var(--text-heading);
            font-size: 28px;
            font-weight: 700;
        }}
        .badge {{
            background: #21262d;
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 13px;
            color: var(--accent-purple);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .card h3 {{
            margin-top: 0;
            color: #f0f6fc;
            font-size: 18px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            font-size: 14px;
        }}
        .metric-val {{
            font-weight: 600;
            color: #f0f6fc;
        }}
        .tag-compute {{
            color: var(--accent-green);
            background: rgba(46, 160, 67, 0.15);
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .tag-memory {{
            color: var(--accent-red);
            background: rgba(248, 81, 73, 0.15);
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .progress-bar-bg {{
            background: #21262d;
            border-radius: 6px;
            height: 10px;
            overflow: hidden;
            margin-top: 4px;
        }}
        .progress-bar-fill {{
            height: 100%;
            background: var(--text-heading);
            border-radius: 6px;
        }}
        #roofline-plot {{
            width: 100%;
            height: 520px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            margin-bottom: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: #8b949e;
            font-weight: 600;
            background: #161b22;
        }}
        tr:hover {{
            background: #21262d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🔍 {title}</h1>
            <p style="margin: 4px 0 0 0; color: #8b949e; font-size: 14px;">
                Semiconductor Hardware-Software Co-Design & Memory Tile Flow Inspector
            </p>
        </div>
        <span class="badge">TileLens v0.1.0 (Open Source)</span>
    </div>

    <!-- Roofline Plot Container -->
    <div class="card" style="padding: 16px; margin-bottom: 24px;">
        <h3 style="margin-bottom: 12px;">📊 Multi-Hardware Theoretical Roofline Model</h3>
        <div id="roofline-plot"></div>
    </div>

    <!-- Device Cards Grid -->
    <h3 style="color: #f0f6fc; margin-bottom: 16px;">⚡ Evaluated Accelerator Profiles</h3>
    <div class="grid">
"""
    for d in devices_data:
        bound_class = "tag-compute" if d["bound_type"] == "COMPUTE_BOUND" else "tag-memory"
        bound_label = "Compute-Bound" if d["bound_type"] == "COMPUTE_BOUND" else "Memory-Bound"
        html_template += f"""
        <div class="card">
            <h3>{d["name"]} <span style="font-size: 12px; color: #8b949e;">({d["precision"]})</span></h3>
            <div class="metric-row">
                <span>Bottleneck Regime:</span>
                <span class="{bound_class}">{bound_label}</span>
            </div>
            <div class="metric-row">
                <span>Attainable Compute:</span>
                <span class="metric-val">{d["attainable_tflops"]:,.1f} TFLOPs ({d["compute_efficiency"]:.1f}% peak)</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {min(100.0, d['compute_efficiency'])}%;"></div>
            </div>
            <div class="metric-row" style="margin-top: 14px;">
                <span>Operational Intensity:</span>
                <span class="metric-val">{d["op_intensity"]:.1f} FLOPs/Byte</span>
            </div>
            <div class="metric-row">
                <span>Hardware Ridge Point:</span>
                <span class="metric-val">{d["ridge_point"]:.1f} FLOPs/Byte</span>
            </div>
            <div class="metric-row">
                <span>HBM Bandwidth:</span>
                <span class="metric-val">{d["bandwidth_gbs"]:,.0f} GB/s ({d["bandwidth_utilization"]:.1f}% util)</span>
            </div>
            <div class="metric-row">
                <span>Execution Time:</span>
                <span class="metric-val" style="color: var(--accent-orange);">{d["exec_time_ms"]:.4f} ms</span>
            </div>
        </div>
"""

    if tiles_data:
        html_template += """
    </div>
    <!-- Tile Memory Hierarchy Table -->
    <div class="card" style="margin-bottom: 24px;">
        <h3>🧱 On-Chip SRAM & Tile Memory Allocation</h3>
        <table>
            <thead>
                <tr>
                    <th>Hardware</th>
                    <th>GEMM Size</th>
                    <th>Tile Config (Tm,Tn,Tk)</th>
                    <th>Stages</th>
                    <th>SRAM Required / Core</th>
                    <th>SRAM Limit</th>
                    <th>Utilization</th>
                    <th>HBM Traffic</th>
                    <th>Data Reuse Factor</th>
                </tr>
            </thead>
            <tbody>
"""
        for t in tiles_data:
            overflow_style = "color: var(--accent-red); font-weight: bold;" if t["sram_overflow"] else ""
            html_template += f"""
                <tr>
                    <td><b>{t["hardware"]}</b></td>
                    <td>{t["gemm"]}</td>
                    <td>({t["tile_m"]}, {t["tile_n"]}, {t["tile_k"]})</td>
                    <td>{t["stages"]}</td>
                    <td style="{overflow_style}">{t["sram_req_kb"]:.1f} KB</td>
                    <td>{t["sram_limit_kb"]:.1f} KB</td>
                    <td style="{overflow_style}">{t["sram_util_pct"]:.1f}%</td>
                    <td>{t["hbm_mb"]:.2f} MB</td>
                    <td>{t["reuse_factor"]:.2f}x</td>
                </tr>
"""
        html_template += """
            </tbody>
        </table>
    </div>
"""
    else:
        html_template += "\n    </div>\n"

    # Embedded Plotly JavaScript
    html_template += f"""
    <script>
        const devices = {json.dumps(devices_data)};
        
        // Generate Roofline Traces
        const plotData = [];
        const colors = ['#58a6ff', '#2ea043', '#f85149', '#bc8cff', '#d29922', '#39c5bb'];
        
        // X-axis range for intensity (from 0.1 to 10000 FLOPs/Byte)
        const x_vals = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];

        devices.forEach((dev, idx) => {{
            const color = colors[idx % colors.length];
            const peak_tflops = dev.peak_tflops;
            const bw_tbs = dev.bandwidth_gbs / 1000.0;
            
            // Roofline curve points
            const y_vals = x_vals.map(x => Math.min(peak_tflops, x * bw_tbs));

            // Hardware Roofline line
            plotData.push({{
                x: x_vals,
                y: y_vals,
                mode: 'lines',
                name: `${{dev.name}} (Peak: ${{peak_tflops.toLocaleString()}} TF, ${{dev.bandwidth_gbs}} GB/s)`,
                line: {{ color: color, width: 2.5 }}
            }});

            // Operating Point Marker
            plotData.push({{
                x: [dev.op_intensity],
                y: [dev.attainable_tflops],
                mode: 'markers+text',
                name: `${{dev.name}} Workload`,
                text: [`${{dev.name}} (${{dev.attainable_tflops.toFixed(1)}} TF)`],
                textposition: 'top right',
                marker: {{
                    size: 12,
                    color: color,
                    symbol: dev.bound_type === 'COMPUTE_BOUND' ? 'circle' : 'diamond',
                    line: {{ color: '#ffffff', width: 1.5 }}
                }}
            }});
        }});

        const layout = {{
            paper_bgcolor: '#161b22',
            plot_bgcolor: '#0d1117',
            xaxis: {{
                title: 'Operational Intensity (FLOPs / Byte)',
                type: 'log',
                gridcolor: '#30363d',
                color: '#c9d1d9'
            }},
            yaxis: {{
                title: 'Attainable Throughput (TFLOPs)',
                type: 'log',
                gridcolor: '#30363d',
                color: '#c9d1d9'
            }},
            margin: {{ l: 60, r: 40, t: 30, b: 60 }},
            legend: {{
                font: {{ color: '#c9d1d9', size: 11 }},
                bgcolor: 'rgba(22, 27, 34, 0.8)',
                bordercolor: '#30363d',
                borderwidth: 1
            }}
        }};

        Plotly.newPlot('roofline-plot', plotData, layout, {{ responsive: true }});
    </script>
</body>
</html>
"""
    return html_template


def save_html_report(
    filepath: str,
    roofline_results: List[RooflineResult],
    tile_results: Optional[List[TileSimulationResult]] = None,
    title: str = "TileLens Hardware-Software Co-Design Dashboard",
) -> str:
    """Saves the generated dashboard to an HTML file."""
    html_content = generate_interactive_dashboard_html(roofline_results, tile_results, title)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    return filepath
