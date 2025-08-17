"""Command-line interface for the Spiking-FPGA-Toolchain."""

import click
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

from spiking_fpga.core import FPGATarget
from spiking_fpga.network_compiler import compile_network
from spiking_fpga.models.optimization import OptimizationLevel
from spiking_fpga.security import SecureCompiler, VulnerabilityScanner
from spiking_fpga.scalability import DistributedCompiler, ResourceManager


@click.group()
@click.version_option()
def main():
    """🧠 Spiking-FPGA-Toolchain: Compile spiking neural networks to FPGA hardware."""
    pass


@main.command()
@click.argument('network_file', type=click.Path(exists=True, path_type=Path))
@click.option('--target', '-t', 
              type=click.Choice([t.value for t in FPGATarget]),
              default=FPGATarget.ARTIX7_35T.value,
              help='Target FPGA platform')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default=Path('./output'),
              help='Output directory for generated HDL')
@click.option('--optimization-level', '-O', type=int, default=2,
              help='Optimization level (0-3)')
@click.option('--synthesis/--no-synthesis', default=False,
              help='Run FPGA synthesis after HDL generation')
@click.option('--power-budget', type=float,
              help='Power budget in milliwatts')
@click.option('--secure/--no-secure', default=True,
              help='Use secure compilation with sandboxing')
@click.option('--skip-security-scan/--security-scan', default=False,
              help='Skip security vulnerability scanning')
@click.option('--distributed/--local', default=False,
              help='Use distributed compilation for better performance')
@click.option('--max-workers', type=int,
              help='Maximum worker threads for distributed compilation')
def compile(network_file: Path, target: str, output: Optional[Path], 
           optimization_level: int, synthesis: bool, power_budget: Optional[float],
           secure: bool, skip_security_scan: bool, distributed: bool, 
           max_workers: Optional[int]):
    """Compile a spiking neural network to FPGA hardware."""
    fpga_target = FPGATarget(target)
    opt_level = OptimizationLevel(optimization_level)
    
    click.echo(f"🚀 Compiling {network_file.name} for {fpga_target.value}")
    click.echo(f"📊 Optimization level: {opt_level.name}")
    click.echo(f"📁 Output directory: {output}")
    if secure:
        click.echo("🔒 Using secure compilation with sandboxing")
    if distributed:
        click.echo("🚀 Using distributed compilation for enhanced performance")
    
    try:
        if distributed:
            # Use distributed compilation
            dist_compiler = DistributedCompiler({
                'max_workers': max_workers,
                'enable_cache': True
            })
            dist_compiler.start()
            
            try:
                # Load network configuration
                import yaml
                with open(network_file, 'r') as f:
                    network_config = yaml.safe_load(f)
                
                # Submit compilation task
                task_id = dist_compiler.submit_compilation(
                    network_config=network_config,
                    target=target,
                    optimization_level=optimization_level,
                    power_budget_mw=power_budget,
                    run_synthesis=synthesis,
                    output_dir=str(output)
                )
                
                click.echo(f"🔄 Submitted task {task_id}")
                
                # Wait for completion with progress updates
                import time
                start_time = time.time()
                while True:
                    status = dist_compiler.get_status(task_id)
                    elapsed = time.time() - start_time
                    
                    if status == 'completed':
                        result = dist_compiler.get_result(task_id)
                        break
                    elif status == 'running':
                        click.echo(f"⏱️  Compiling... ({elapsed:.1f}s)")
                    
                    time.sleep(2)
                    
                    if elapsed > 1200:  # 20 minute timeout
                        raise TimeoutError("Distributed compilation timed out")
                
            finally:
                dist_compiler.stop()
                
        elif secure:
            # Use secure compiler
            secure_compiler = SecureCompiler()
            result = secure_compiler.secure_compile(
                network_file=network_file,
                target=target,
                output_dir=output,
                optimization_level=optimization_level,
                power_budget_mw=power_budget,
                run_synthesis=synthesis
            )
        else:
            # Standard compilation
            result = compile_network(
                network=network_file,
                target=fpga_target,
                output_dir=output,
                optimization_level=opt_level,
                power_budget_mw=power_budget,
                run_synthesis=synthesis
            )
        
        if result.success:
            click.echo("\n✅ Compilation successful!")
            click.echo(f"📈 Resource usage:")
            click.echo(f"   Neurons: {result.resource_estimate.neurons}")
            click.echo(f"   Synapses: {result.resource_estimate.synapses}")
            click.echo(f"   LUTs: {result.resource_estimate.luts}")
            click.echo(f"   BRAM: {result.resource_estimate.bram_kb:.2f} KB")
            
            click.echo(f"\n📝 Generated files:")
            for name, path in result.hdl_files.items():
                click.echo(f"   {name}: {path}")
                
            if result.synthesis_result:
                if result.synthesis_result.success:
                    click.echo("\n🔧 Synthesis completed successfully")
                else:
                    click.echo("\n❌ Synthesis failed")
                    for error in result.synthesis_result.errors:
                        click.echo(f"   Error: {error}")
            
            # Security validation results
            if secure and hasattr(result, 'security_validation'):
                sec_val = result.security_validation
                click.echo(f"\n🔒 Security validation:")
                click.echo(f"   Input validated: {'✅' if sec_val['input_validated'] else '❌'}")
                click.echo(f"   Sandbox used: {'✅' if sec_val['sandbox_used'] else '❌'}")
                click.echo(f"   Output files validated: {sec_val['output_validated']}")
                click.echo(f"   Compilation time: {sec_val['compilation_time']:.2f}s")
            
            # Run security scan if requested
            if not skip_security_scan and result.success:
                click.echo("\n🛡️  Running security vulnerability scan...")
                scanner = VulnerabilityScanner()
                scan_results = scanner.scan_hdl_directory(output)
                security_score = scanner.calculate_security_score(scan_results)
                
                click.echo(f"   Security score: {security_score['score']}/100 (Grade: {security_score['grade']})")
                click.echo(f"   Issues found: {security_score['total_issues']}")
                if security_score['high_severity'] > 0:
                    click.echo(f"   ⚠️  High severity issues: {security_score['high_severity']}")
                
                if security_score['score'] < 80:
                    click.echo("   💡 Consider reviewing security recommendations")
                    
        else:
            click.echo("\n❌ Compilation failed!")
            for error in result.errors:
                click.echo(f"   Error: {error}")
                
    except Exception as e:
        click.echo(f"\n💥 Unexpected error: {str(e)}")
        raise click.ClickException(str(e))


@main.command()
@click.argument('network_file', type=click.Path(exists=True, path_type=Path))
def analyze(network_file: Path):
    """Analyze a network file and show structure information."""
    from spiking_fpga.compiler.frontend import parse_network_file
    from spiking_fpga.network_compiler import NetworkCompiler
    
    click.echo(f"🔍 Analyzing network: {network_file.name}")
    
    try:
        network = parse_network_file(network_file)
        
        click.echo(f"\n📋 Network Information:")
        click.echo(f"   Name: {network.name}")
        click.echo(f"   Description: {network.description or 'N/A'}")
        click.echo(f"   Timestep: {network.timestep} ms")
        
        click.echo(f"\n🧠 Network Structure:")
        click.echo(f"   Total neurons: {len(network.neurons)}")
        click.echo(f"   Total synapses: {len(network.synapses)}")
        click.echo(f"   Layers: {len(network.layers)}")
        
        click.echo(f"\n📊 Layer Details:")
        for layer in network.layers:
            click.echo(f"   {layer.layer_id}: {layer.layer_type.value} ({layer.size} {layer.neuron_type} neurons)")
        
        # Check for issues
        issues = network.validate_network()
        if issues:
            click.echo(f"\n⚠️  Validation Issues:")
            for issue in issues:
                click.echo(f"   - {issue}")
        else:
            click.echo(f"\n✅ Network validation passed")
        
        # Show resource estimates for different targets
        click.echo(f"\n📈 Resource Estimates:")
        for target in [FPGATarget.ARTIX7_35T, FPGATarget.ARTIX7_100T]:
            compiler = NetworkCompiler(target)
            estimate = compiler.estimate_resources(network)
            utilization = estimate.utilization_percentage(target.resources)
            
            click.echo(f"   {target.value}:")
            click.echo(f"     Logic: {utilization.get('logic', 0):.1f}%")
            click.echo(f"     Memory: {utilization.get('memory', 0):.1f}%")
            click.echo(f"     DSP: {utilization.get('dsp', 0):.1f}%")
            
    except Exception as e:
        click.echo(f"❌ Analysis failed: {str(e)}")
        raise click.ClickException(str(e))


@main.command()
@click.argument('network_file', type=click.Path(exists=True, path_type=Path))
@click.argument('target', type=click.Choice([t.value for t in FPGATarget]))
def suggest(network_file: Path, target: str):
    """Suggest optimizations for a network on the target platform."""
    from spiking_fpga.compiler.frontend import parse_network_file
    from spiking_fpga.network_compiler import NetworkCompiler
    
    fpga_target = FPGATarget(target)
    
    click.echo(f"🎯 Analyzing {network_file.name} for {fpga_target.value}")
    
    try:
        network = parse_network_file(network_file)
        compiler = NetworkCompiler(fpga_target)
        suggestions = compiler.suggest_optimizations(network)
        
        click.echo(f"\n💡 Optimization Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            click.echo(f"   {i}. {suggestion}")
            
    except Exception as e:
        click.echo(f"❌ Analysis failed: {str(e)}")
        raise click.ClickException(str(e))


@main.command()
@click.option('--target', '-t', 
              type=click.Choice([t.value for t in FPGATarget]),
              help='Show resources for specific target')
def resources(target: Optional[str]):
    """Show FPGA resource information."""
    if target:
        fpga_target = FPGATarget(target)
        click.echo(f"📊 Resources for {fpga_target.value} ({fpga_target.vendor}):")
        click.echo(f"🔧 Toolchain: {fpga_target.toolchain}")
        click.echo("📈 Resource limits:")
        for key, value in fpga_target.resources.items():
            if isinstance(value, (int, float)):
                if key.endswith('_kb'):
                    click.echo(f"   {key}: {value:,.1f} KB")
                else:
                    click.echo(f"   {key}: {value:,}")
            else:
                click.echo(f"   {key}: {value}")
    else:
        click.echo("🎯 Supported FPGA targets:")
        for fpga_target in FPGATarget:
            vendor_icon = "🔶" if fpga_target.vendor == "xilinx" else "🔷"
            click.echo(f"   {vendor_icon} {fpga_target.value} ({fpga_target.vendor})")


@main.command()
@click.option('--auto-fix/--no-auto-fix', default=True,
              help='Automatically fix issues found during validation')
@click.option('--quick/--full', default=False,
              help='Quick validation (skip toolchain checks)')
def validate(auto_fix: bool, quick: bool):
    """Validate development environment setup with intelligent auto-fixing."""
    click.echo("🔍 Validating environment...")
    issues_found = []
    fixes_applied = []
    
    # Check Python environment
    try:
        import numpy, yaml, pydantic, networkx
        click.echo("✅ Python dependencies OK")
    except ImportError as e:
        click.echo(f"❌ Missing Python dependency: {e}")
        issues_found.append(f"Missing dependency: {e}")
        if auto_fix:
            click.echo("🔧 Auto-fixing: Installing missing dependencies...")
            import subprocess
            try:
                subprocess.run(["pip", "install", str(e).split("'")[1]], check=True, capture_output=True)
                fixes_applied.append(f"Installed {e}")
                click.echo("✅ Dependency installed successfully")
            except subprocess.CalledProcessError:
                click.echo("❌ Failed to install dependency automatically")
    
    if not quick:
        # Check FPGA toolchains
        from spiking_fpga.compiler.backend import VivadoBackend, QuartusBackend
        
        vivado = VivadoBackend()
        if vivado.is_available():
            click.echo("✅ Vivado toolchain available")
        else:
            click.echo("⚠️  Vivado toolchain not found")
            issues_found.append("Vivado toolchain not available")
        
        quartus = QuartusBackend()
        if quartus.is_available():
            click.echo("✅ Quartus toolchain available")
        else:
            click.echo("⚠️  Quartus toolchain not found")
            issues_found.append("Quartus toolchain not available")
    
    # Check example files
    example_file = Path(__file__).parent.parent.parent / "examples" / "simple_mnist.yaml"
    if example_file.exists():
        click.echo("✅ Example networks available")
    else:
        click.echo("⚠️  Example networks not found")
        issues_found.append("Example networks missing")
        if auto_fix:
            click.echo("🔧 Auto-fixing: Creating example network...")
            try:
                example_file.parent.mkdir(parents=True, exist_ok=True)
                example_content = """# Auto-generated MNIST example network
name: simple_mnist
description: Basic MNIST classifier for validation
timestep: 1.0

layers:
  - layer_id: input
    layer_type: input
    size: 784
    neuron_type: poisson
  - layer_id: hidden
    layer_type: hidden  
    size: 300
    neuron_type: lif
    parameters:
      tau_m: 20.0
      v_thresh: 1.0
  - layer_id: output
    layer_type: output
    size: 10
    neuron_type: lif

connections:
  - source: input
    target: hidden
    connectivity: sparse_random
    sparsity: 0.1
    weight_range: [0.1, 0.8]
  - source: hidden
    target: output
    connectivity: sparse_random
    sparsity: 0.3
    weight_range: [0.2, 1.0]
"""
                example_file.write_text(example_content)
                fixes_applied.append("Created example network")
                click.echo("✅ Example network created successfully")
            except Exception as e:
                click.echo(f"❌ Failed to create example network: {e}")
    
    # Performance validation
    click.echo("\n🚀 Performance validation...")
    try:
        import time
        start = time.time()
        from spiking_fpga.performance_optimizer import create_optimized_compiler
        compiler = create_optimized_compiler(FPGATarget.ARTIX7_35T)
        load_time = time.time() - start
        
        if load_time < 2.0:
            click.echo("✅ Fast compiler initialization")
        else:
            click.echo(f"⚠️  Slow initialization ({load_time:.2f}s)")
            issues_found.append(f"Slow initialization: {load_time:.2f}s")
    except Exception as e:
        click.echo(f"❌ Performance validation failed: {e}")
    
    # Summary
    click.echo(f"\n🎯 Environment validation complete")
    if issues_found:
        click.echo(f"⚠️  Issues found: {len(issues_found)}")
        for issue in issues_found:
            click.echo(f"   - {issue}")
    if fixes_applied:
        click.echo(f"🔧 Auto-fixes applied: {len(fixes_applied)}")
        for fix in fixes_applied:
            click.echo(f"   ✅ {fix}")
    
    if not issues_found:
        click.echo("🎉 Environment fully validated!")
    elif auto_fix and fixes_applied:
        click.echo("✨ Environment validated with auto-fixes applied!")


@main.command()
@click.option('--target', '-t', 
              type=click.Choice([t.value for t in FPGATarget]),
              default=FPGATarget.ARTIX7_35T.value,
              help='Target FPGA platform for benchmarking')
@click.option('--size', type=click.Choice(['small', 'medium', 'large']), 
              default='medium',
              help='Benchmark network size')
@click.option('--duration', type=int, default=10,
              help='Benchmark duration in seconds')
def benchmark(target: str, size: str, duration: int):
    """Run intelligent performance benchmarks with adaptive optimization."""
    fpga_target = FPGATarget(target)
    
    size_configs = {
        'small': {'neurons': 1000, 'layers': [100, 800, 100]},
        'medium': {'neurons': 10000, 'layers': [784, 1200, 10]},
        'large': {'neurons': 100000, 'layers': [1024, 8000, 1000, 100]}
    }
    
    config = size_configs[size]
    
    click.echo(f"🚀 Running {size} benchmark on {fpga_target.value}")
    click.echo(f"🧠 Network: {config['neurons']} neurons, {len(config['layers'])} layers")
    click.echo(f"⏱️  Duration: {duration}s")
    
    try:
        from spiking_fpga.performance_optimizer import AdaptivePerformanceController
        from spiking_fpga.utils.monitoring import SystemResourceMonitor
        import time
        
        # Create benchmark network
        from spiking_fpga.models.network import Network
        network = Network(
            name=f"benchmark_{size}",
            description=f"Automated benchmark network ({size})",
            timestep=1.0
        )
        
        # Initialize performance monitoring
        perf_controller = AdaptivePerformanceController()
        monitor = SystemResourceMonitor()
        
        click.echo("\n📊 Starting benchmark...")
        start_time = time.time()
        monitor.start_monitoring()
        
        # Compile with adaptive optimization
        from spiking_fpga.network_compiler import compile_network
        result = compile_network(
            network=network,
            target=fpga_target,
            optimization_level=3,
            adaptive_optimization=True
        )
        
        compile_time = time.time() - start_time
        
        # Get performance metrics
        metrics = monitor.get_metrics()
        optimization_suggestions = perf_controller.analyze_performance(metrics)
        
        click.echo("\n📈 Benchmark Results:")
        click.echo(f"   Compilation time: {compile_time:.2f}s")
        click.echo(f"   Peak memory usage: {metrics.get('peak_memory_mb', 0):.1f} MB")
        click.echo(f"   Average CPU usage: {metrics.get('avg_cpu_percent', 0):.1f}%")
        
        if result.success:
            click.echo(f"   Resource utilization:")
            estimate = result.resource_estimate
            resources = fpga_target.resources
            
            lut_util = (estimate.luts / resources.get('logic_cells', 1)) * 100
            bram_util = (estimate.bram_kb / resources.get('bram_kb', 1)) * 100
            
            click.echo(f"     LUTs: {lut_util:.1f}%")
            click.echo(f"     BRAM: {bram_util:.1f}%")
            
            if lut_util > 80 or bram_util > 80:
                click.echo("⚠️  High resource utilization detected")
        
        if optimization_suggestions:
            click.echo("\n💡 Optimization Suggestions:")
            for suggestion in optimization_suggestions[:3]:  # Top 3
                click.echo(f"   • {suggestion}")
        
        # Adaptive learning from benchmark
        perf_controller.learn_from_benchmark(config, metrics, result)
        click.echo("\n🧠 Performance patterns learned for future optimizations")
        
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {str(e)}")
        raise click.ClickException(str(e))


@main.command()
@click.option('--watch/--no-watch', default=False,
              help='Continuously monitor system')
@click.option('--interval', type=int, default=5,
              help='Monitoring interval in seconds')
def monitor(watch: bool, interval: int):
    """Monitor system resources and compiler performance."""
    from spiking_fpga.utils.monitoring import SystemResourceMonitor
    import time
    
    monitor = SystemResourceMonitor()
    
    if watch:
        click.echo("👁️  Starting continuous monitoring (Ctrl+C to stop)")
        click.echo(f"📊 Update interval: {interval}s")
        
        try:
            while True:
                metrics = monitor.get_current_metrics()
                click.clear()
                click.echo("🖥️  System Resources:")
                click.echo(f"   CPU: {metrics.get('cpu_percent', 0):.1f}%")
                click.echo(f"   Memory: {metrics.get('memory_percent', 0):.1f}%")
                click.echo(f"   Disk: {metrics.get('disk_percent', 0):.1f}%")
                
                # Check for compilation processes
                active_compilations = monitor.get_active_compilations()
                if active_compilations:
                    click.echo("\n🔄 Active Compilations:")
                    for comp in active_compilations:
                        click.echo(f"   • {comp['network']} ({comp['elapsed']:.1f}s)")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            click.echo("\n👋 Monitoring stopped")
    else:
        metrics = monitor.get_current_metrics()
        click.echo("📊 Current System Status:")
        click.echo(f"   CPU: {metrics.get('cpu_percent', 0):.1f}%")
        click.echo(f"   Memory: {metrics.get('memory_percent', 0):.1f}%")
        click.echo(f"   Available Memory: {metrics.get('available_memory_gb', 0):.1f} GB")
        click.echo(f"   Disk Usage: {metrics.get('disk_percent', 0):.1f}%")
        
        # Show recommendations
        recommendations = monitor.get_performance_recommendations()
        if recommendations:
            click.echo("\n💡 Performance Recommendations:")
            for rec in recommendations:
                click.echo(f"   • {rec}")


@main.command()
@click.argument('hdl_directory', type=click.Path(exists=True, path_type=Path))
@click.option('--output-format', type=click.Choice(['text', 'json', 'html']), 
              default='text',
              help='Output format for scan results')
@click.option('--save-report', type=click.Path(path_type=Path),
              help='Save detailed report to file')
@click.option('--fail-on-high/--no-fail-on-high', default=False,
              help='Exit with error code if high-severity issues found')
def security_scan(hdl_directory: Path, output_format: str, 
                 save_report: Optional[Path], fail_on_high: bool):
    """Run comprehensive security vulnerability scan on HDL files."""
    click.echo(f"🛡️  Scanning {hdl_directory} for security vulnerabilities...")
    
    try:
        scanner = VulnerabilityScanner()
        
        # Run comprehensive scan
        scan_results = scanner.scan_hdl_directory(hdl_directory)
        security_score = scanner.calculate_security_score(scan_results)
        
        # Display results
        click.echo(f"\n📊 Scan Results:")
        click.echo(f"   Files scanned: {scan_results['files_scanned']}")
        click.echo(f"   Total issues: {scan_results['total_issues']}")
        click.echo(f"   Security score: {security_score['score']}/100 (Grade: {security_score['grade']})")
        
        # Severity breakdown
        click.echo(f"\n🚨 Issue Severity:")
        click.echo(f"   High: {security_score['high_severity']}")
        click.echo(f"   Medium: {security_score['medium_severity']}")
        click.echo(f"   Low: {security_score['low_severity']}")
        
        # Top vulnerabilities
        if scan_results['total_issues'] > 0:
            summary = scan_results.get('summary', {})
            common_issues = summary.get('common_issues', [])
            
            if common_issues:
                click.echo(f"\n🔍 Most Common Issues:")
                for issue_type, count in common_issues:
                    click.echo(f"   {issue_type}: {count} occurrences")
            
            # Most vulnerable files
            vulnerable_files = summary.get('most_vulnerable_files', [])
            if vulnerable_files:
                click.echo(f"\n📁 Most Vulnerable Files:")
                for file_path, issue_count in vulnerable_files[:3]:
                    click.echo(f"   {Path(file_path).name}: {issue_count} issues")
        
        # Additional system security checks
        click.echo(f"\n🔧 System Security Checks:")
        
        file_perm_issues = scanner.check_file_permissions(hdl_directory)
        if file_perm_issues:
            click.echo(f"   File permissions: {len(file_perm_issues)} issues")
        else:
            click.echo("   File permissions: ✅ OK")
        
        process_issues = scanner.check_process_isolation()
        if process_issues:
            click.echo(f"   Process isolation: {len(process_issues)} issues")
        else:
            click.echo("   Process isolation: ✅ OK")
        
        network_issues = scanner.check_network_exposure()
        if network_issues:
            click.echo(f"   Network exposure: {len(network_issues)} issues")
        else:
            click.echo("   Network exposure: ✅ OK")
        
        temp_issues = scanner.check_temporary_files()
        if temp_issues:
            click.echo(f"   Temporary files: {len(temp_issues)} issues")
        else:
            click.echo("   Temporary files: ✅ OK")
        
        # Save detailed report if requested
        if save_report:
            detailed_report = {
                'scan_results': scan_results,
                'security_score': security_score,
                'system_checks': {
                    'file_permissions': file_perm_issues,
                    'process_isolation': process_issues,
                    'network_exposure': network_issues,
                    'temporary_files': temp_issues
                }
            }
            
            if output_format == 'json':
                import json
                with open(save_report, 'w') as f:
                    json.dump(detailed_report, f, indent=2, default=str)
            elif output_format == 'html':
                # Generate HTML report
                html_content = _generate_html_report(detailed_report)
                with open(save_report, 'w') as f:
                    f.write(html_content)
            else:
                # Text format
                with open(save_report, 'w') as f:
                    f.write(f"Security Scan Report\n")
                    f.write(f"===================\n\n")
                    f.write(f"Directory: {hdl_directory}\n")
                    f.write(f"Scan Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Security Score: {security_score['score']}/100\n\n")
                    # Add detailed findings...
            
            click.echo(f"\n📄 Detailed report saved to: {save_report}")
        
        # Recommendations
        if security_score['score'] < 90:
            click.echo(f"\n💡 Security Recommendations:")
            if security_score['high_severity'] > 0:
                click.echo("   • Address high-severity vulnerabilities immediately")
            if security_score['medium_severity'] > 5:
                click.echo("   • Review medium-severity issues for potential fixes")
            if file_perm_issues:
                click.echo("   • Fix file permission issues")
            if process_issues or network_issues:
                click.echo("   • Review system security configuration")
            
            click.echo("   • Consider using secure compilation mode")
            click.echo("   • Regular security scans in CI/CD pipeline")
        
        # Exit with error if high-severity issues found and flag set
        if fail_on_high and security_score['high_severity'] > 0:
            click.echo(f"\n❌ Exiting with error due to {security_score['high_severity']} high-severity issues")
            raise click.ClickException("Security scan failed due to high-severity vulnerabilities")
        
        if security_score['score'] >= 90:
            click.echo("\n🎉 Excellent security posture!")
        elif security_score['score'] >= 80:
            click.echo("\n✅ Good security posture with room for improvement")
        else:
            click.echo("\n⚠️  Security improvements needed")
            
    except Exception as e:
        click.echo(f"❌ Security scan failed: {str(e)}")
        raise click.ClickException(str(e))


def _generate_html_report(report_data: Dict[str, Any]) -> str:
    """Generate HTML security report."""
    import time
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .score {{ font-size: 24px; font-weight: bold; }}
        .high {{ color: #d32f2f; }}
        .medium {{ color: #f57c00; }}
        .low {{ color: #388e3c; }}
        .section {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Security Scan Report</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div class="score">Security Score: {report_data['security_score']['score']}/100 
        (Grade: {report_data['security_score']['grade']})</div>
    </div>
    
    <div class="section">
        <h2>📊 Summary</h2>
        <p>Files scanned: {report_data['scan_results']['files_scanned']}</p>
        <p>Total issues: {report_data['scan_results']['total_issues']}</p>
        <p class="high">High severity: {report_data['security_score']['high_severity']}</p>
        <p class="medium">Medium severity: {report_data['security_score']['medium_severity']}</p>
        <p class="low">Low severity: {report_data['security_score']['low_severity']}</p>
    </div>
    
    <!-- Add more sections for detailed findings -->
    
</body>
</html>
"""
    return html


@main.command()
@click.argument('batch_config', type=click.Path(exists=True, path_type=Path))
@click.option('--max-parallel', type=int, default=4,
              help='Maximum parallel compilations')
@click.option('--output-base', type=click.Path(path_type=Path), 
              default=Path('./batch_output'),
              help='Base output directory for batch compilation')
@click.option('--resource-optimization/--no-resource-optimization', default=True,
              help='Enable intelligent resource optimization')
def batch_compile(batch_config: Path, max_parallel: int, output_base: Path,
                 resource_optimization: bool):
    """Perform batch compilation with intelligent resource management."""
    click.echo(f"🚀 Starting batch compilation from {batch_config.name}")
    click.echo(f"📊 Max parallel: {max_parallel}")
    click.echo(f"📁 Output base: {output_base}")
    
    try:
        # Load batch configuration
        import yaml
        import time
        import json
        
        with open(batch_config, 'r') as f:
            config = yaml.safe_load(f)
        
        tasks = config.get('tasks', [])
        if not tasks:
            raise click.ClickException("No tasks found in batch configuration")
        
        click.echo(f"📋 Found {len(tasks)} compilation tasks")
        
        # Initialize distributed compiler and resource manager
        dist_compiler = DistributedCompiler({
            'max_workers': max_parallel,
            'enable_cache': True,
            'target_cpu_usage': 70.0
        })
        
        resource_manager = ResourceManager({
            'memory_pool_gb': 16,  # Configurable
            'cpu_pool_percent': 80.0
        })
        
        dist_compiler.start()
        resource_manager.start_monitoring()
        
        try:
            if resource_optimization:
                click.echo("🧠 Optimizing resource allocation...")
                
                # Predict resource needs
                network_configs = []
                targets = []
                
                for task in tasks:
                    with open(task['network_file'], 'r') as f:
                        network_config = yaml.safe_load(f)
                    network_configs.append(network_config)
                    targets.append(task['target'])
                
                resource_predictions = resource_manager.predict_resource_needs(
                    network_configs, targets)
                
                # Get optimal batching
                batches = resource_manager.recommend_batch_scheduling(resource_predictions)
                
                click.echo(f"📦 Optimized into {len(batches)} resource-balanced batches")
                
                # Execute batches
                all_results = []
                for batch_idx, batch_sessions in enumerate(batches):
                    click.echo(f"\n🔄 Executing batch {batch_idx + 1}/{len(batches)}")
                    
                    # Submit batch tasks
                    batch_task_ids = []
                    for i, session_id in enumerate(batch_sessions):
                        task_idx = int(session_id.split('_')[1])
                        task = tasks[task_idx]
                        
                        # Create output directory
                        task_output = output_base / f"task_{task_idx}"
                        task_output.mkdir(parents=True, exist_ok=True)
                        
                        # Load network config
                        with open(task['network_file'], 'r') as f:
                            network_config = yaml.safe_load(f)
                        
                        # Submit compilation
                        task_id = dist_compiler.submit_compilation(
                            network_config=network_config,
                            target=task['target'],
                            optimization_level=task.get('optimization_level', 2),
                            output_dir=str(task_output),
                            **task.get('options', {})
                        )
                        batch_task_ids.append(task_id)
                    
                    # Wait for batch completion
                    batch_results = dist_compiler.wait_for_batch(batch_task_ids, timeout=1800)
                    all_results.extend(batch_results)
                    
                    # Show batch results
                    successful = sum(1 for r in batch_results if r.success)
                    click.echo(f"   ✅ {successful}/{len(batch_results)} tasks completed successfully")
            
            else:
                # Simple parallel execution without optimization
                click.echo("⚡ Executing with simple parallelization...")
                
                batch_task_ids = []
                for i, task in enumerate(tasks):
                    # Create output directory
                    task_output = output_base / f"task_{i}"
                    task_output.mkdir(parents=True, exist_ok=True)
                    
                    # Load network config
                    with open(task['network_file'], 'r') as f:
                        network_config = yaml.safe_load(f)
                    
                    # Submit compilation
                    task_id = dist_compiler.submit_compilation(
                        network_config=network_config,
                        target=task['target'],
                        optimization_level=task.get('optimization_level', 2),
                        output_dir=str(task_output),
                        **task.get('options', {})
                    )
                    batch_task_ids.append(task_id)
                
                # Wait for all completions
                all_results = dist_compiler.wait_for_batch(batch_task_ids, timeout=3600)
            
            # Generate comprehensive report
            successful_tasks = sum(1 for r in all_results if r.success)
            total_duration = sum(r.duration for r in all_results)
            avg_duration = total_duration / len(all_results) if all_results else 0
            
            click.echo(f"\n📊 Batch Compilation Summary:")
            click.echo(f"   Total tasks: {len(tasks)}")
            click.echo(f"   Successful: {successful_tasks}")
            click.echo(f"   Failed: {len(tasks) - successful_tasks}")
            click.echo(f"   Total time: {total_duration:.1f}s")
            click.echo(f"   Average per task: {avg_duration:.1f}s")
            
            # Performance statistics
            perf_stats = dist_compiler.get_performance_stats()
            if perf_stats:
                click.echo(f"   Cache hit rate: {perf_stats.get('cache_hit_rate', 0):.1f}%")
                click.echo(f"   Throughput: {len(tasks) / (total_duration / max_parallel):.2f} tasks/min")
            
            # Resource statistics
            resource_stats = resource_manager.get_resource_statistics()
            click.echo(f"   Peak memory usage: {resource_stats['system_resources']['memory_used_percent']:.1f}%")
            click.echo(f"   Peak CPU usage: {resource_stats['system_resources']['cpu_used_percent']:.1f}%")
            
            # Save detailed report
            report_file = output_base / "batch_report.json"
            report_data = {
                'summary': {
                    'total_tasks': len(tasks),
                    'successful_tasks': successful_tasks,
                    'total_duration': total_duration,
                    'avg_duration': avg_duration
                },
                'task_results': [
                    {
                        'task_id': r.task_id,
                        'success': r.success,
                        'duration': r.duration,
                        'errors': r.errors,
                        'worker_id': r.worker_id
                    }
                    for r in all_results
                ],
                'performance_stats': perf_stats,
                'resource_stats': resource_stats
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            click.echo(f"\n📄 Detailed report saved to: {report_file}")
            
            if successful_tasks == len(tasks):
                click.echo("🎉 All tasks completed successfully!")
            elif successful_tasks > 0:
                click.echo("⚠️  Some tasks failed - check individual outputs")
            else:
                click.echo("❌ All tasks failed")
                raise click.ClickException("Batch compilation failed")
        
        finally:
            dist_compiler.stop()
            resource_manager.stop_monitoring()
            
    except Exception as e:
        click.echo(f"❌ Batch compilation failed: {str(e)}")
        raise click.ClickException(str(e))


@main.command()
@click.option('--format', type=click.Choice(['text', 'json']), default='text',
              help='Output format')
def system_status(format: str):
    """Show comprehensive system and resource status."""
    try:
        # Get system resources
        import psutil
        import time
        
        # CPU information
        cpu_info = {
            'cores': psutil.cpu_count(),
            'usage_percent': psutil.cpu_percent(interval=1),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown'
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'usage_percent': memory.percent
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            'total_gb': disk.total / (1024**3),
            'free_gb': disk.free / (1024**3),
            'usage_percent': (disk.used / disk.total) * 100
        }
        
        # Check toolchain availability
        toolchains = {}
        try:
            from spiking_fpga.compiler.backend import VivadoBackend, QuartusBackend
            vivado = VivadoBackend()
            quartus = QuartusBackend()
            
            toolchains['vivado'] = vivado.is_available()
            toolchains['quartus'] = quartus.is_available()
        except ImportError:
            toolchains['vivado'] = False
            toolchains['quartus'] = False
        
        # System load
        try:
            load_avg = os.getloadavg()
            load_info = {
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2]
            }
        except (AttributeError, OSError):
            load_info = {'1min': 0, '5min': 0, '15min': 0}
        
        status_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cpu': cpu_info,
            'memory': memory_info,
            'disk': disk_info,
            'load_average': load_info,
            'toolchains': toolchains
        }
        
        if format == 'json':
            import json
            click.echo(json.dumps(status_data, indent=2))
        else:
            # Text format
            click.echo("🖥️  System Status")
            click.echo("=" * 50)
            click.echo(f"📅 Timestamp: {status_data['timestamp']}")
            click.echo()
            
            click.echo("💻 CPU:")
            click.echo(f"   Cores: {cpu_info['cores']}")
            click.echo(f"   Usage: {cpu_info['usage_percent']:.1f}%")
            click.echo(f"   Frequency: {cpu_info['frequency_mhz']} MHz")
            click.echo()
            
            click.echo("🧠 Memory:")
            click.echo(f"   Total: {memory_info['total_gb']:.1f} GB")
            click.echo(f"   Available: {memory_info['available_gb']:.1f} GB")
            click.echo(f"   Usage: {memory_info['usage_percent']:.1f}%")
            click.echo()
            
            click.echo("💾 Disk:")
            click.echo(f"   Total: {disk_info['total_gb']:.1f} GB")
            click.echo(f"   Free: {disk_info['free_gb']:.1f} GB")
            click.echo(f"   Usage: {disk_info['usage_percent']:.1f}%")
            click.echo()
            
            click.echo("📊 Load Average:")
            click.echo(f"   1 min: {load_info['1min']:.2f}")
            click.echo(f"   5 min: {load_info['5min']:.2f}")
            click.echo(f"   15 min: {load_info['15min']:.2f}")
            click.echo()
            
            click.echo("🔧 FPGA Toolchains:")
            for toolchain, available in toolchains.items():
                status_icon = "✅" if available else "❌"
                click.echo(f"   {toolchain.capitalize()}: {status_icon}")
            
            # Performance recommendations
            recommendations = []
            if cpu_info['usage_percent'] > 90:
                recommendations.append("High CPU usage - consider reducing parallel compilation")
            if memory_info['usage_percent'] > 90:
                recommendations.append("High memory usage - close unnecessary applications")
            if disk_info['usage_percent'] > 90:
                recommendations.append("Low disk space - clean up temporary files")
            if not any(toolchains.values()):
                recommendations.append("No FPGA toolchains detected - install Vivado or Quartus")
            
            if recommendations:
                click.echo("\n💡 Recommendations:")
                for rec in recommendations:
                    click.echo(f"   • {rec}")
            else:
                click.echo("\n🎉 System is running optimally!")
    
    except Exception as e:
        click.echo(f"❌ Failed to get system status: {str(e)}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    main()