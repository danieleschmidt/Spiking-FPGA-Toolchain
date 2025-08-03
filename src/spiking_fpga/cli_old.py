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
    """Spiking-FPGA-Toolchain: Compile spiking neural networks to FPGA hardware."""
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
@click.option('--target', '-t', 
              type=click.Choice([t.value for t in FPGATarget]),
              help='Show resources for specific target')
def resources(target: Optional[str]):
    """Show FPGA resource information."""
    if target:
        fpga_target = FPGATarget(target)
        click.echo(f"Resources for {fpga_target.value}:")
        for key, value in fpga_target.resources.items():
            click.echo(f"  {key}: {value}")
    else:
        click.echo("Supported FPGA targets:")
        for fpga_target in FPGATarget:
            click.echo(f"  {fpga_target.value} ({fpga_target.vendor})")


@main.command()
def validate():
    """Validate development environment setup."""
    click.echo("Validating environment...")
    click.echo("✓ Python environment OK")
    click.echo("⚠ FPGA toolchains not implemented yet")
    click.echo("⚠ HDL simulation not implemented yet")


if __name__ == '__main__':
    main()