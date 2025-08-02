"""Command-line interface for the Spiking-FPGA-Toolchain."""

import click
from pathlib import Path
from typing import Optional

from spiking_fpga.core import FPGATarget


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
              help='Output directory for generated HDL')
@click.option('--optimization-level', '-O', type=int, default=2,
              help='Optimization level (0-3)')
def compile(network_file: Path, target: str, output: Optional[Path], optimization_level: int):
    """Compile a spiking neural network to FPGA hardware."""
    click.echo(f"Compiling {network_file} for {target}")
    click.echo("This is a placeholder - implementation coming in Phase 1")
    
    fpga_target = FPGATarget(target)
    click.echo(f"Target resources: {fpga_target.resources}")


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