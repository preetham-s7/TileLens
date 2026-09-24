"""
Command-Line Interface (CLI) for TileLens with optional Rich styling and graceful zero-dependency fallback.
"""

import argparse
import math
import sys
from typing import List, Optional

# Ensure safe UTF-8 terminal encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

from tilelens.hardware.database import (
    HARDWARE_DATABASE,
    ALIAS_MAP,
    get_hardware,
    list_available_hardware,
)
from tilelens.hardware.spec import Precision
from tilelens.core.roofline import RooflineModel, BoundType
from tilelens.core.tile_simulator import TileConfig, TileSimulator
from tilelens.core.analyzer import PerformanceAnalyzer
from tilelens.viz.visualizer import save_html_report


def print_msg(msg: str):
    if HAS_RICH and console:
        console.print(msg)
    else:
        # Strip simple rich tags for plain output
        clean_msg = (
            msg.replace("[bold cyan]", "").replace("[/bold cyan]", "")
            .replace("[bold magenta]", "").replace("[/bold magenta]", "")
            .replace("[bold green]", "").replace("[/bold green]", "")
            .replace("[bold yellow]", "").replace("[/bold yellow]", "")
            .replace("[bold red]", "").replace("[/bold red]", "")
            .replace("[bold white]", "").replace("[/bold white]", "")
            .replace("[bold]", "").replace("[/bold]", "")
            .replace("[yellow]", "").replace("[/yellow]", "")
            .replace("[green]", "").replace("[/green]", "")
            .replace("[cyan]", "").replace("[/cyan]", "")
            .replace("[magenta]", "").replace("[/magenta]", "")
            .replace("[red]", "").replace("[/red]", "")
            .replace("[dim]", "").replace("[/dim]", "")
        )
        try:
            print(clean_msg)
        except UnicodeEncodeError:
            print(clean_msg.encode("ascii", "replace").decode("ascii"))


def handle_list_hardware(args):
    """Prints a styled table of all hardware accelerators across Mobile, Laptop, and Cloud."""
    if HAS_RICH and console:
        table = Table(
            title="[bold magenta]TileLens Hardware Database (Mobile, Laptop PCs & Cloud AI)[/bold magenta]",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Device Alias", style="yellow")
        table.add_column("Name & Vendor", style="white")
        table.add_column("Class / Target", style="cyan")
        table.add_column("Process", justify="right")
        table.add_column("Memory BW", justify="right", style="cyan")
        table.add_column("SRAM / Core", justify="right", style="magenta")
        table.add_column("Peak Compute / NPU", justify="right", style="bold green")

        for key, spec in HARDWARE_DATABASE.items():
            bf16_val = spec.compute.dense_tflops.get(Precision.BF16, spec.compute.dense_tflops.get(Precision.INT8, 0.0))
            fp8_val = spec.compute.dense_tflops.get(Precision.FP8, 0.0)
            compute_str = f"{bf16_val:,.1f} TF"
            if fp8_val > 0:
                compute_str += f" | {fp8_val:,.0f} FP8"
            if spec.npu_tops:
                compute_str += f" | {spec.npu_tops:.0f} NPU"

            table.add_row(
                key,
                f"{spec.name} ({spec.vendor})",
                spec.target_device,
                f"{spec.process_node_nm}nm",
                f"{spec.memory.hbm_bandwidth_gbs:,.0f} GB/s",
                f"{spec.memory.sram_per_core_kb:,.0f} KB",
                compute_str,
            )
        console.print(table)
        console.print(
            f"[dim]Tip: Inspect chip blueprint: [yellow]tilelens blueprint -d a17[/yellow] or [yellow]tilelens blueprint -d m4[/yellow][/dim]\n"
        )
    else:
        print_msg("\n" + "=" * 110)
        print_msg("TileLens Hardware Database (Mobile, Laptop PCs & Cloud AI)")
        print_msg("=" * 110)
        print_msg(f"{'Key / Alias':<22} | {'Name':<24} | {'Target Class':<20} | {'Memory BW':<12} | {'Peak Compute'}")
        print_msg("-" * 110)
        for key, spec in HARDWARE_DATABASE.items():
            bf16_val = spec.compute.dense_tflops.get(Precision.BF16, spec.compute.dense_tflops.get(Precision.INT8, 0.0))
            fp8_val = spec.compute.dense_tflops.get(Precision.FP8, 0.0)
            compute_str = f"{bf16_val:,.1f} TF"
            if fp8_val > 0:
                compute_str += f" | {fp8_val:,.0f} FP8"
            if spec.npu_tops:
                compute_str += f" | {spec.npu_tops:.0f} NPU"
            print_msg(
                f"{key:<22} | {spec.name:<24} | {spec.target_device:<20} | "
                f"{spec.memory.hbm_bandwidth_gbs:,.0f} GB/s".ljust(12) + f" | {compute_str}"
            )
        print_msg("=" * 110 + "\n")


def handle_blueprint(args):
    """Inspects a chip's silicon blueprint, architectural blocks, and operational capabilities."""
    try:
        hw = get_hardware(args.device)
    except KeyError as e:
        print_msg(f"[bold red]Error:[/bold red] {e}")
        return

    if HAS_RICH and console:
        console.print()
        console.print(
            Panel(
                f"[bold cyan]Chip:[/bold cyan] {hw.name} ({hw.vendor})\n"
                f"[bold cyan]Target:[/bold cyan] [bold green]{hw.target_device}[/bold green] | [bold cyan]Architecture:[/bold cyan] {hw.architecture}\n"
                f"[bold cyan]Process Node:[/bold cyan] {hw.process_node_nm}nm | [bold cyan]Transistors:[/bold cyan] {hw.transistor_count_billion or 'N/A'} Billion | [bold cyan]Die Area:[/bold cyan] {hw.die_size_mm2 or 'N/A'} mm²\n"
                f"[bold cyan]Packaging:[/bold cyan] {hw.packaging or 'Monolithic'} | [bold cyan]TDP:[/bold cyan] {hw.tdp_watts}W\n"
                f"[bold cyan]Memory:[/bold cyan] {hw.memory.hbm_bandwidth_gbs:,.0f} GB/s ({hw.memory.hbm_capacity_gb} GB {hw.memory_type or 'DRAM'}, {hw.memory_bus_width_bits or 'N/A'}-bit bus)\n"
                f"[bold cyan]On-Chip SRAM:[/bold cyan] {hw.memory.sram_per_core_kb:,.0f} KB/core (Total: {hw.memory.total_sram_mb:.1f} MB) | L2/SLC Cache: {hw.memory.l2_cache_mb} MB",
                title=f"🔍 Chip Silicon Blueprint: {hw.name}",
                border_style="magenta",
            )
        )

        b_table = Table(title="[bold magenta]Microarchitectural Blueprint Functional Blocks[/bold magenta]", box=box.ROUNDED, header_style="bold cyan")
        b_table.add_column("Count", justify="right", style="yellow")
        b_table.add_column("Block Name", style="bold white")
        b_table.add_column("Domain", style="cyan")
        b_table.add_column("Area %", justify="right", style="green")
        b_table.add_column("Description & Key Specs", style="dim white")

        for b in hw.blueprint_blocks:
            specs_str = " | ".join([f"{k}: {v}" for k, v in b.specs.items()])
            desc_full = f"{b.description}" + (f" [{specs_str}]" if specs_str else "")
            b_table.add_row(
                f"{b.count}x" if b.count > 1 else "1x",
                b.name,
                b.category.upper(),
                f"{b.area_percent:.1f}%" if b.area_percent > 0 else "-",
                desc_full,
            )
        console.print(b_table)

        console.print("\n[bold yellow]⚡ What Else This Chip Does (Operational Capabilities):[/bold yellow]")
        for cap in hw.capabilities:
            console.print(f"  [bold green]✓[/bold green] {cap}")
        console.print()
    else:
        print_msg("\n" + hw.blueprint_summary() + "\n")


def handle_serve(args):
    """Starts a local HTTP server so any mobile phone and laptop PC can use TileLens."""
    import http.server
    import socketserver
    import socket
    import webbrowser
    import os

    port = args.port
    host = args.host

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=workspace_dir, **kwargs)

        def log_message(self, format, *args):
            pass

    server_address = (host, port)
    try:
        # Allow immediate port reuse
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(server_address, QuietHandler)
    except OSError as e:
        print_msg(f"[bold red]Error starting server on port {port}:[/bold red] {e}")
        return

    local_url = f"http://localhost:{port}/index.html"
    network_url = f"http://{local_ip}:{port}/index.html"

    if HAS_RICH and console:
        console.print(
            Panel.fit(
                f"[bold green]TileLens Universal Cross-Device Web App is Running![/bold green]\n\n"
                f"💻 [bold cyan]Laptop / Desktop PC:[/bold cyan]  [underline yellow]{local_url}[/underline yellow]\n"
                f"📱 [bold cyan]Mobile Phone (Wi-Fi):[/bold cyan] [underline yellow]{network_url}[/underline yellow]\n\n"
                f"[dim]Supports auto-detection of your device chip, interactive silicon blueprint, and roofline analysis.[/dim]\n"
                f"[bold white]Press Ctrl+C to stop the server.[/bold white]",
                title="🔍 TileLens Universal Cross-Device Server",
                border_style="cyan",
            )
        )
    else:
        print_msg("\n" + "=" * 70)
        print_msg("TileLens Universal Cross-Device Server is running!")
        print_msg(f"Laptop / Desktop PC: {local_url}")
        print_msg(f"Mobile Phone (Wi-Fi): {network_url}")
        print_msg("Press Ctrl+C to stop the server.")
        print_msg("=" * 70 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(local_url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print_msg("\n[yellow]Shutting down TileLens server...[/yellow]")
        httpd.server_close()


def handle_roofline(args):
    """Evaluates an arbitrary roofline point for a given device."""
    try:
        hw = get_hardware(args.device)
    except KeyError as e:
        print_msg(f"[bold red]Error:[/bold red] {e}")
        return

    precision = Precision(args.precision.lower())
    model = RooflineModel(hw, precision=precision)
    result = model.evaluate(args.flops, args.bytes)

    if HAS_RICH and console:
        console.print(Panel(result.summary(), title=f"Roofline Analysis: {hw.name}", border_style="cyan"))
    else:
        print_msg("\n" + result.summary() + "\n")


def handle_gemm(args):
    """Runs full tile simulation and co-design bottleneck analysis for a GEMM workload."""
    try:
        hw = get_hardware(args.device)
    except KeyError as e:
        print_msg(f"[bold red]Error:[/bold red] {e}")
        return

    precision = Precision(args.precision.lower())
    tile_cfg = TileConfig(
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        pipeline_stages=args.stages,
        precision=precision,
    )

    analyzer = PerformanceAnalyzer(hw)
    report = analyzer.analyze_gemm(args.m, args.n, args.k, tile_cfg)

    rf = report.roofline_result
    tr = report.tile_result

    if HAS_RICH and console:
        console.print()
        console.print(
            Panel.fit(
                f"[bold cyan]Hardware:[/bold cyan] {hw.name} ({hw.architecture})\n"
                f"[bold cyan]GEMM Dimensions:[/bold cyan] M={args.m}, N={args.n}, K={args.k} | Precision: [bold green]{precision.value.upper()}[/bold green]\n"
                f"[bold cyan]Tile Configuration:[/bold cyan] Tm={args.tile_m}, Tn={args.tile_n}, Tk={args.tile_k} (Stages={args.stages})",
                title="TileLens Co-Design Performance Analyzer",
                border_style="magenta",
            )
        )

        r_table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
        r_table.add_column("Metric", style="white")
        r_table.add_column("Value", style="bold green", justify="right")
        r_table.add_column("Hardware Bound / Ceiling", style="yellow", justify="right")

        bound_color = "green" if rf.bound_type == BoundType.COMPUTE_BOUND else "red"
        r_table.add_row("Operational Intensity", f"{rf.operational_intensity:.2f} FLOPs/Byte", f"Ridge: {rf.ridge_point:.2f} FLOPs/Byte")
        r_table.add_row("Bottleneck Classification", f"[{bound_color}]{rf.bound_type.value}[/{bound_color}]", "-")
        r_table.add_row("Attainable Throughput", f"{rf.attainable_tflops:,.2f} TFLOPs", f"Peak: {rf.peak_tflops:,.2f} TFLOPs")
        r_table.add_row("Compute Efficiency", f"{rf.compute_efficiency_pct:.1f}%", "100.0%")
        r_table.add_row("HBM Bandwidth Utilized", f"{rf.bandwidth_utilization_pct:.1f}%", f"{rf.memory_bandwidth_gbs:,.0f} GB/s")
        r_table.add_row("Ideal Execution Time", f"{rf.execution_time_ms:.4f} ms", "-")
        console.print(r_table)

        s_table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta")
        s_table.add_column("SRAM & Concurrency Metric", style="white")
        s_table.add_column("Value", style="bold cyan", justify="right")

        sram_style = "bold red" if tr.sram_overflow else "bold green"
        s_table.add_row("SRAM Required / SM Core", f"[{sram_style}]{tr.total_sram_required_kb:.1f} KB[/{sram_style}] (Limit: {tr.hardware_sram_limit_kb:.1f} KB)")
        s_table.add_row("SRAM Utilization", f"[{sram_style}]{tr.sram_utilization_pct:.1f}%[/{sram_style}]")
        s_table.add_row("Concurrent Blocks per SM", f"{tr.active_blocks_per_sm}")
        s_table.add_row("Total Grid Tiles & Waves", f"{tr.total_grid_tiles} blocks ({tr.total_waves:.2f} waves)")
        s_table.add_row("HBM Data Traffic (Actual / Ideal)", f"{tr.actual_hbm_bytes / 1e6:,.2f} MB / {tr.ideal_hbm_bytes / 1e6:,.2f} MB")
        s_table.add_row("Memory Data Reuse Amplification", f"{tr.tile_reuse_factor:.2f}x")
        console.print(s_table)
    else:
        print_msg("\n" + report.summary())

    if HAS_RICH and console:
        console.print("\n[bold yellow]Hardware Diagnostics & Actionable Co-Design Tips:[/bold yellow]")
        if not report.suggestions:
            console.print("  [bold green][OK] Configuration is well-optimized for this hardware.[/bold green]")
        else:
            for s in report.suggestions:
                sev_color = "red" if s.severity == "CRITICAL" else ("yellow" if s.severity == "WARNING" else "blue")
                console.print(f"  [{sev_color}][{s.severity}][/{sev_color}] [bold]{s.category}:[/bold] {s.message}")
                console.print(f"     [green]↳ Action Item:[/green] {s.action_item}\n")

        if report.optimal_tile_config:
            opt = report.optimal_tile_config
            console.print(
                f"[bold green]Recommended Optimal Tile:[/bold green] Tm={opt.tile_m}, Tn={opt.tile_n}, Tk={opt.tile_k} (Stages={opt.pipeline_stages})\n"
            )


def handle_compare(args):
    """Compares a GEMM kernel across multiple GPU and TPU architectures."""
    device_names = [d.strip() for d in args.devices.split(",")]
    precision = Precision(args.precision.lower())

    roofline_results = []
    tile_results = []
    comparison_rows = []

    for name in device_names:
        try:
            hw = get_hardware(name)
        except KeyError:
            print_msg(f"[yellow]Warning: Device '{name}' not found, skipping.[/yellow]")
            continue

        tile_cfg = TileConfig(tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k, precision=precision)
        sim = TileSimulator(hw)
        tr = sim.simulate_gemm(args.m, args.n, args.k, tile_cfg)
        
        rf_model = RooflineModel(hw, precision=precision)
        rf = rf_model.evaluate(2.0 * args.m * args.n * args.k, tr.actual_hbm_bytes)

        roofline_results.append(rf)
        tile_results.append(tr)
        comparison_rows.append((hw, rf, tr))

    if HAS_RICH and console:
        table = Table(
            title=f"[bold magenta]Multi-Hardware Comparison: GEMM ({args.m}x{args.n}x{args.k}) [{precision.value.upper()}][/bold magenta]",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Hardware", style="bold white")
        table.add_column("Type", style="yellow")
        table.add_column("HBM BW", justify="right")
        table.add_column("Attainable TFLOPs", justify="right", style="bold green")
        table.add_column("Compute Util", justify="right")
        table.add_column("Intensity", justify="right")
        table.add_column("Bound Type", justify="center")
        table.add_column("Exec Time", justify="right", style="magenta")

        for hw, rf, tr in comparison_rows:
            bound_str = "[green]Compute[/green]" if rf.bound_type == BoundType.COMPUTE_BOUND else "[red]Memory[/red]"
            table.add_row(
                hw.name,
                hw.hardware_type.value,
                f"{hw.memory.hbm_bandwidth_gbs:,.0f} GB/s",
                f"{rf.attainable_tflops:,.1f} TF",
                f"{rf.compute_efficiency_pct:.1f}%",
                f"{rf.operational_intensity:.1f} FLOPs/B",
                bound_str,
                f"{rf.execution_time_ms:.4f} ms",
            )
        console.print(table)
    else:
        print_msg("\n" + "=" * 105)
        print_msg(f"Multi-Hardware Comparison: GEMM ({args.m}x{args.n}x{args.k}) [{precision.value.upper()}]")
        print_msg("=" * 105)
        print_msg(f"{'Hardware':<26} | {'Type':<6} | {'HBM BW':<10} | {'Attainable TF':<14} | {'Compute %':<10} | {'Intensity':<12} | {'Exec Time'}")
        print_msg("-" * 105)
        for hw, rf, tr in comparison_rows:
            bound_tag = "Compute" if rf.bound_type == BoundType.COMPUTE_BOUND else "Memory"
            print_msg(
                f"{hw.name:<26} | {hw.hardware_type.value:<6} | {hw.memory.hbm_bandwidth_gbs:,.0f} GB/s".ljust(10) + f" | "
                f"{rf.attainable_tflops:>11.1f} TF | {rf.compute_efficiency_pct:>8.1f}% | {rf.operational_intensity:>8.1f} FLOP/B | {rf.execution_time_ms:.4f} ms ({bound_tag})"
            )
        print_msg("=" * 105 + "\n")

    if args.output_html:
        save_html_report(args.output_html, roofline_results, tile_results)
        print_msg(f"[bold green][OK] Interactive visual dashboard saved to:[/bold green] {args.output_html}")


def handle_export_viz(args):
    """Generates an interactive HTML dashboard with Plotly Roofline curves."""
    device_names = [d.strip() for d in args.devices.split(",")]
    precision = Precision(args.precision.lower())

    roofline_results = []
    tile_results = []

    for name in device_names:
        try:
            hw = get_hardware(name)
            tile_cfg = TileConfig(tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k, precision=precision)
            sim = TileSimulator(hw)
            tr = sim.simulate_gemm(args.m, args.n, args.k, tile_cfg)
            rf_model = RooflineModel(hw, precision=precision)
            rf = rf_model.evaluate(2.0 * args.m * args.n * args.k, tr.actual_hbm_bytes)
            roofline_results.append(rf)
            tile_results.append(tr)
        except KeyError as e:
            print_msg(f"[yellow]Warning:[/yellow] {e}")

    filepath = save_html_report(args.output, roofline_results, tile_results, title=args.title)
    print_msg(f"[bold green][OK] Interactive dashboard exported successfully:[/bold green] {filepath}")


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(
        description="TileLens: Hardware-Software Co-Design, Chip Blueprint Inspector & Roofline Visualizer for Mobile, Laptop PCs & Cloud AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: list-hardware
    subparsers.add_parser("list-hardware", help="List all registered Mobile, Laptop PC, and Cloud AI chips.")

    # Command: blueprint
    p_blue = subparsers.add_parser("blueprint", help="Inspect a chip's silicon blueprint, architectural blocks, and capabilities.")
    p_blue.add_argument("--device", "-d", default="a17", help="Target chip alias (e.g. a17, m4, rtx4060, h100, b200, snapdragon)")

    # Command: serve
    p_srv = subparsers.add_parser("serve", help="Launch the TileLens web application for mobile phones and laptops.")
    p_srv.add_argument("--port", type=int, default=8080, help="Port to listen on")
    p_srv.add_argument("--host", default="0.0.0.0", help="Host interface (0.0.0.0 for LAN/mobile access)")
    p_srv.add_argument("--no-browser", action="store_true", help="Do not automatically open browser on host")

    # Command: roofline
    p_roof = subparsers.add_parser("roofline", help="Calculate standalone theoretical Roofline limit.")
    p_roof.add_argument("--device", "-d", default="h100", help="Target hardware (e.g. a17, m4, h100, tpu_v5e, b200)")
    p_roof.add_argument("--precision", "-p", default="bf16", help="Precision format (fp32, bf16, fp16, fp8, int8)")
    p_roof.add_argument("--flops", type=float, required=True, help="Total FLOPs executed")
    p_roof.add_argument("--bytes", type=float, required=True, help="Total bytes transferred from HBM/DRAM")

    # Command: gemm
    p_gemm = subparsers.add_parser("gemm", help="Simulate GEMM tile flow and analyze hardware bottlenecks.")
    p_gemm.add_argument("-M", "-m", dest="m", type=positive_integer, default=4096, help="GEMM M dimension")
    p_gemm.add_argument("-N", "-n", dest="n", type=positive_integer, default=4096, help="GEMM N dimension")
    p_gemm.add_argument("-K", "-k", dest="k", type=positive_integer, default=4096, help="GEMM K dimension")
    p_gemm.add_argument("--device", "-d", default="h100", help="Target hardware alias (e.g. a17, m4, h100, tpu_v5e)")
    p_gemm.add_argument("--precision", "-p", default="bf16", help="Precision format")
    p_gemm.add_argument("--tile-m", type=positive_integer, default=128, help="Tile M dimension")
    p_gemm.add_argument("--tile-n", type=positive_integer, default=128, help="Tile N dimension")
    p_gemm.add_argument("--tile-k", type=positive_integer, default=64, help="Tile K dimension")
    p_gemm.add_argument("--stages", type=positive_integer, default=2, help="Pipeline multi-buffering stages (2, 3, 4)")

    # Command: compare
    p_comp = subparsers.add_parser("compare", help="Compare GEMM performance across multiple GPUs, Mobile SoCs & TPUs.")
    p_comp.add_argument("-M", "-m", dest="m", type=positive_integer, default=4096, help="GEMM M dimension")
    p_comp.add_argument("-N", "-n", dest="n", type=positive_integer, default=4096, help="GEMM N dimension")
    p_comp.add_argument("-K", "-k", dest="k", type=positive_integer, default=4096, help="GEMM K dimension")
    p_comp.add_argument("--devices", default="a17,m4,rtx4060,h100", help="Comma-separated device list")
    p_comp.add_argument("--precision", "-p", default="bf16", help="Precision format")
    p_comp.add_argument("--tile-m", type=positive_integer, default=128, help="Tile M dimension")
    p_comp.add_argument("--tile-n", type=positive_integer, default=128, help="Tile N dimension")
    p_comp.add_argument("--tile-k", type=positive_integer, default=64, help="Tile K dimension")
    p_comp.add_argument("--output-html", "-o", default=None, help="Optional path to export interactive HTML dashboard")

    # Command: export-viz
    p_viz = subparsers.add_parser("export-viz", help="Export an interactive HTML dashboard.")
    p_viz.add_argument("-M", "-m", dest="m", type=positive_integer, default=4096, help="GEMM M dimension")
    p_viz.add_argument("-N", "-n", dest="n", type=positive_integer, default=4096, help="GEMM N dimension")
    p_viz.add_argument("-K", "-k", dest="k", type=positive_integer, default=4096, help="GEMM K dimension")
    p_viz.add_argument("--devices", default="a17,m4,h100,b200,rtx4090", help="Comma-separated device list")
    p_viz.add_argument("--precision", "-p", default="bf16", help="Precision format")
    p_viz.add_argument("--tile-m", type=positive_integer, default=128, help="Tile M")
    p_viz.add_argument("--tile-n", type=positive_integer, default=128, help="Tile N")
    p_viz.add_argument("--tile-k", type=positive_integer, default=64, help="Tile K")
    p_viz.add_argument("--output", "-o", default="tilelens_dashboard.html", help="Output HTML filepath")
    p_viz.add_argument("--title", default="TileLens Hardware Co-Design Roofline Dashboard", help="Dashboard title")

    args = parser.parse_args()

    if args.command == "list-hardware":
        handle_list_hardware(args)
    elif args.command == "blueprint":
        handle_blueprint(args)
    elif args.command == "serve":
        handle_serve(args)
    elif args.command == "roofline":
        handle_roofline(args)
    elif args.command == "gemm":
        handle_gemm(args)
    elif args.command == "compare":
        handle_compare(args)
    elif args.command == "export-viz":
        handle_export_viz(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
