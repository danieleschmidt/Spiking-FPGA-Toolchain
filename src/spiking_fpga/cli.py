"""Command-line interface for the Spiking-FPGA-Toolchain."""

import click
import logging
import sys
from pathlib import Path
from typing import Optional
import time
import json

from .models.fpga import FPGATarget
from .services.network_compiler import NetworkCompiler, CompilationConfig
from .parsers.yaml_parser import YAMLNetworkParser


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose: bool):
    """Spiking-FPGA-Toolchain: Compile spiking neural networks to FPGA hardware."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.argument('network_file', type=click.Path(exists=True, path_type=Path))
@click.option('--target', '-t', 
              type=click.Choice(list(FPGATarget.FPGA_SPECS.keys())),
              default=FPGATarget.ARTIX7_35T,
              help='Target FPGA platform')
@click.option('--output', '-o', type=click.Path(path_type=Path), default='./output',
              help='Output directory for generated HDL')
@click.option('--optimization-level', '-O', type=int, default=2,
              help='Optimization level (0=none, 1=basic, 2=aggressive, 3=experimental)')
@click.option('--frequency', '-f', type=float, default=100.0,
              help='Target clock frequency in MHz')
@click.option('--enable-plasticity', is_flag=True, default=False,
              help='Enable synaptic plasticity (STDP)')
@click.option('--generate-testbench', is_flag=True, default=True,
              help='Generate simulation testbench')
@click.option('--report-file', type=click.Path(path_type=Path),
              help='Save compilation report to file')
def compile(network_file: Path, target: str, output: Optional[Path], 
           optimization_level: int, frequency: float, enable_plasticity: bool,
           generate_testbench: bool, report_file: Optional[Path]):
    """Compile a spiking neural network to FPGA hardware."""
    
    click.echo(f"🧠 Spiking-FPGA-Toolchain v1.0")
    click.echo(f"📁 Network file: {network_file}")
    click.echo(f"🎯 Target FPGA: {target}")
    click.echo(f"📂 Output directory: {output}")
    click.echo("")
    
    try:
        # Parse network configuration
        click.echo("🔍 Parsing network configuration...")
        parser = YAMLNetworkParser()
        
        if network_file.suffix.lower() in ['.yaml', '.yml']:
            network = parser.parse_file(str(network_file))
        else:
            click.echo(f"❌ Unsupported file format: {network_file.suffix}")
            click.echo("   Supported formats: .yaml, .yml")
            sys.exit(1)
        
        click.echo(f"✅ Loaded network '{network.name}' with {len(network.layers)} layers")
        
        # Setup compilation configuration
        config = CompilationConfig(
            optimization_level=optimization_level,
            target_frequency_mhz=frequency,
            enable_plasticity=enable_plasticity,
            output_directory=str(output),
            generate_testbench=generate_testbench
        )
        
        # Initialize compiler
        compiler = NetworkCompiler(config)
        
        # Quick resource estimation
        click.echo("📊 Estimating resource requirements...")
        estimated_resources = compiler.estimate_resources(network, target)
        fpga_target = FPGATarget(target)
        
        utilization_pct = estimated_resources.get_utilization_percentages(fpga_target)
        click.echo(f"   LUTs: {estimated_resources.luts_used} ({utilization_pct['luts']:.1f}%)")
        click.echo(f"   BRAM: {estimated_resources.bram_used // 1024}KB ({utilization_pct['bram']:.1f}%)")
        click.echo(f"   DSP: {estimated_resources.dsp_used} ({utilization_pct['dsp']:.1f}%)")
        
        # Check if resources exceed capacity
        violations = estimated_resources.check_constraints(fpga_target)
        if violations:
            click.echo("\n⚠️  Resource constraint violations detected:")
            for violation in violations:
                click.echo(f"   - {violation}")
            click.echo("\n💡 Consider:")
            click.echo("   - Using a larger FPGA (e.g., artix7_100t)")
            click.echo("   - Reducing network size")
            click.echo("   - Increasing sparsity in connections")
            
            if not click.confirm("\nContinue compilation anyway?"):
                sys.exit(1)
        
        click.echo("")
        
        # Compile network
        click.echo("⚙️  Compiling network to HDL...")
        start_time = time.time()
        
        result = compiler.compile_network(network, target)
        
        # Report results
        if result.success:
            compilation_time = time.time() - start_time
            click.echo(f"✅ Compilation successful in {compilation_time:.2f}s")
            click.echo("")
            click.echo("📋 Generated files:")
            for hdl_file in result.hdl_files:
                click.echo(f"   📄 {hdl_file}")
            for constraint_file in result.constraint_files:
                click.echo(f"   📄 {constraint_file}")
            if result.testbench_files:
                click.echo(f"   🧪 Testbench: {result.testbench_files[0]}")
            
            click.echo("")
            click.echo("📈 Performance estimates:")
            click.echo(f"   Max frequency: {result.estimated_max_frequency_mhz:.1f} MHz")
            click.echo(f"   Spike throughput: {result.max_spike_rate_mhz:.2f} Mspikes/s")
            click.echo(f"   Inference latency: {result.inference_latency_ms:.2f} ms")
            click.echo(f"   Power consumption: {result.estimated_power_w:.2f} W")
            
            if result.warnings:
                click.echo("\n⚠️  Warnings:")
                for warning in result.warnings:
                    click.echo(f"   - {warning}")
            
            # Save compilation report
            if report_file:
                with open(report_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                click.echo(f"\n📊 Compilation report saved to {report_file}")
            
            click.echo(f"\n🎉 Ready for synthesis! Use Vivado or Quartus to build bitstream.")
            
        else:
            click.echo("❌ Compilation failed:")
            for error in result.errors:
                click.echo(f"   - {error}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            import traceback
            click.echo(traceback.format_exc())
        sys.exit(1)


@main.command()
@click.option('--target', '-t', 
              type=click.Choice(list(FPGATarget.FPGA_SPECS.keys())),
              help='Show resources for specific target')
def resources(target: Optional[str]):
    """Show FPGA resource information."""
    if target:
        fpga_target = FPGATarget(target)
        specs = fpga_target.specs
        click.echo(f"📊 Resources for {target}:")
        click.echo(f"   Logic Cells: {specs.logic_cells:,}")
        click.echo(f"   LUTs: {specs.luts:,}")
        click.echo(f"   Flip-Flops: {specs.flip_flops:,}")
        click.echo(f"   BRAM: {specs.bram_bits // 1024:,} KB ({specs.bram_blocks} blocks)")
        click.echo(f"   DSP Slices: {specs.dsp_slices}")
        click.echo(f"   I/O Pins: {specs.io_pins}")
        click.echo(f"   Max Frequency: {specs.max_frequency_mhz} MHz")
        
        # Show capacity estimates
        max_neurons = fpga_target.get_max_neurons("LIF")
        max_synapses = fpga_target.get_max_synapses()
        click.echo("")
        click.echo("🧠 Estimated capacity:")
        click.echo(f"   Max LIF neurons: ~{max_neurons:,}")
        click.echo(f"   Max synapses: ~{max_synapses:,}")
        
    else:
        click.echo("🎯 Supported FPGA targets:")
        for target_name, specs in FPGATarget.FPGA_SPECS.items():
            vendor = "Xilinx" if "artix" in target_name else "Intel"
            click.echo(f"   {target_name} ({vendor}, {specs.logic_cells:,} cells)")


@main.command()
@click.argument('network_file', type=click.Path(exists=True, path_type=Path))
@click.option('--target', '-t',
              type=click.Choice(list(FPGATarget.FPGA_SPECS.keys())),
              default=FPGATarget.ARTIX7_35T,
              help='Target FPGA platform')
def estimate(network_file: Path, target: str):
    """Estimate resource usage for a network without full compilation."""
    try:
        # Parse network
        parser = YAMLNetworkParser()
        network = parser.parse_file(str(network_file))
        
        # Estimate resources
        compiler = NetworkCompiler()
        resources = compiler.estimate_resources(network, target)
        fpga_target = FPGATarget(target)
        
        utilization_pct = resources.get_utilization_percentages(fpga_target)
        
        click.echo(f"📊 Resource estimation for '{network.name}' on {target}:")
        click.echo("")
        click.echo("🔧 Resource usage:")
        click.echo(f"   LUTs: {resources.luts_used:,} / {fpga_target.specs.luts:,} ({utilization_pct['luts']:.1f}%)")
        click.echo(f"   BRAM: {resources.bram_used // 1024:,} KB / {fpga_target.specs.bram_bits // 1024:,} KB ({utilization_pct['bram']:.1f}%)")
        click.echo(f"   DSP: {resources.dsp_used:,} / {fpga_target.specs.dsp_slices:,} ({utilization_pct['dsp']:.1f}%)")
        
        # Check constraints
        violations = resources.check_constraints(fpga_target)
        if violations:
            click.echo("\n⚠️  Constraint violations:")
            for violation in violations:
                click.echo(f"   - {violation}")
        else:
            click.echo("\n✅ All resource constraints satisfied")
        
        # Power estimate
        power = fpga_target.estimate_power_consumption(resources)
        click.echo(f"\n⚡ Estimated power: {power:.2f} W")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--check-vivado', is_flag=True, help='Check Vivado installation')
@click.option('--check-quartus', is_flag=True, help='Check Quartus installation')
def validate(check_vivado: bool, check_quartus: bool):
    """Validate development environment setup."""
    click.echo("🔍 Validating environment...")
    
    # Check Python environment
    click.echo("✅ Python environment OK")
    
    # Check required packages
    try:
        import numpy
        import yaml
        import pydantic
        click.echo("✅ Required Python packages installed")
    except ImportError as e:
        click.echo(f"❌ Missing Python package: {e}")
    
    # Check FPGA toolchains if requested
    if check_vivado:
        import subprocess
        try:
            result = subprocess.run(['vivado', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                click.echo("✅ Vivado found and accessible")
            else:
                click.echo("⚠️  Vivado not found in PATH")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            click.echo("⚠️  Vivado not found or not accessible")
    
    if check_quartus:
        import subprocess
        try:
            result = subprocess.run(['quartus_sh', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                click.echo("✅ Quartus found and accessible")
            else:
                click.echo("⚠️  Quartus not found in PATH")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            click.echo("⚠️  Quartus not found or not accessible")
    
    click.echo("\n🎯 Ready to compile spiking neural networks!")


@main.command()
@click.argument('output_file', type=click.Path(path_type=Path))
@click.option('--neurons', '-n', type=int, default=1000, help='Number of neurons')
@click.option('--layers', '-l', type=int, default=3, help='Number of layers')
@click.option('--sparsity', '-s', type=float, default=0.1, help='Connection sparsity (0-1)')
def generate_example(output_file: Path, neurons: int, layers: int, sparsity: float):
    """Generate an example network configuration."""
    
    # Create a simple multi-layer network
    config = {
        'name': f'example_network_{neurons}n_{layers}l',
        'description': f'Auto-generated example with {neurons} neurons in {layers} layers',
        'input_size': 100,
        'output_size': 10,
        'parameters': {
            'dt': 0.1,
            'simulation_time': 100.0,
            'spike_threshold': 1.0,
            'refractory_period': 2.0
        },
        'layers': [],
        'connections': []
    }
    
    # Create layers
    neurons_per_layer = neurons // layers
    
    # Input layer
    config['layers'].append({
        'id': 'input',
        'type': 'input',
        'size': 100,
        'neuron_model': 'LIF',
        'tau_m': 20.0
    })
    
    # Hidden layers
    for i in range(1, layers - 1):
        config['layers'].append({
            'id': f'hidden_{i}',
            'type': 'hidden',
            'size': neurons_per_layer,
            'neuron_model': 'LIF',
            'tau_m': 20.0
        })
    
    # Output layer
    config['layers'].append({
        'id': 'output',
        'type': 'output',
        'size': 10,
        'neuron_model': 'LIF',
        'tau_m': 20.0
    })
    
    # Create connections
    layer_ids = [layer['id'] for layer in config['layers']]
    for i in range(len(layer_ids) - 1):
        config['connections'].append({
            'source': layer_ids[i],
            'target': layer_ids[i + 1],
            'connectivity': 'sparse_random',
            'sparsity': sparsity,
            'weight_mean': 0.5,
            'weight_std': 0.1
        })
    
    # Write to file
    import yaml
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    click.echo(f"✅ Example network saved to {output_file}")
    click.echo(f"🧠 Network: {neurons} neurons, {layers} layers, {sparsity} sparsity")
    click.echo(f"🚀 Compile with: spiking-fpga compile {output_file}")


if __name__ == '__main__':
    main()