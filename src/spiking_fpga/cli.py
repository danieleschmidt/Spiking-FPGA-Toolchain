"""Command-line interface for the Spiking-FPGA-Toolchain."""

import click
from pathlib import Path
from typing import Optional

from spiking_fpga.core import FPGATarget
from spiking_fpga.network_compiler import compile_network
from spiking_fpga.models.optimization import OptimizationLevel


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
def compile(network_file: Path, target: str, output: Optional[Path], 
           optimization_level: int, synthesis: bool, power_budget: Optional[float]):
    """Compile a spiking neural network to FPGA hardware."""
    fpga_target = FPGATarget(target)
    opt_level = OptimizationLevel(optimization_level)
    
    click.echo(f"🚀 Compiling {network_file.name} for {fpga_target.value}")
    click.echo(f"📊 Optimization level: {opt_level.name}")
    click.echo(f"📁 Output directory: {output}")
    
    try:
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
def validate():
    """Validate development environment setup."""
    click.echo("🔍 Validating environment...")
    
    # Check Python environment
    try:
        import numpy, yaml, click, pydantic, networkx
        click.echo("✅ Python dependencies OK")
    except ImportError as e:
        click.echo(f"❌ Missing Python dependency: {e}")
    
    # Check FPGA toolchains
    from spiking_fpga.compiler.backend import VivadoBackend, QuartusBackend
    
    vivado = VivadoBackend()
    if vivado.is_available():
        click.echo("✅ Vivado toolchain available")
    else:
        click.echo("⚠️  Vivado toolchain not found")
    
    quartus = QuartusBackend()
    if quartus.is_available():
        click.echo("✅ Quartus toolchain available")
    else:
        click.echo("⚠️  Quartus toolchain not found")
    
    # Check example files
    example_file = Path(__file__).parent.parent.parent / "examples" / "simple_mnist.yaml"
    if example_file.exists():
        click.echo("✅ Example networks available")
    else:
        click.echo("⚠️  Example networks not found")
    
    click.echo("\n🎯 Environment validation complete")


if __name__ == '__main__':
    main()