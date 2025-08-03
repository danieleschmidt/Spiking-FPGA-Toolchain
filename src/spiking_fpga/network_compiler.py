"""Main network compiler interface for the Spiking-FPGA-Toolchain."""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass

from .core import FPGATarget
from .models.network import Network
from .models.optimization import OptimizationLevel, ResourceEstimate
from .compiler.frontend import parse_network_file, get_parser
from .compiler.optimizer import OptimizationPipeline
from .compiler.backend import HDLGenerator, HDLGenerationConfig, VivadoBackend, QuartusBackend, SynthesisResult


@dataclass
class CompilationConfig:
    """Configuration for network compilation."""
    
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    clock_frequency: int = 100_000_000  # 100 MHz
    power_budget_mw: Optional[float] = None
    debug_enabled: bool = False
    generate_reports: bool = True
    run_synthesis: bool = False


@dataclass 
class CompilationResult:
    """Results from network compilation."""
    
    success: bool
    network: Network
    optimized_network: Network
    hdl_files: Dict[str, Path]
    resource_estimate: ResourceEstimate
    optimization_stats: Dict[str, Any]
    synthesis_result: Optional[SynthesisResult] = None
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
    
    def generate_reports(self, output_dir: Union[str, Path]) -> None:
        """Generate comprehensive compilation reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Resource utilization report
        self._generate_resource_report(output_dir / "resource_utilization.txt")
        
        # Network analysis report
        self._generate_network_report(output_dir / "network_analysis.txt")
        
        # Optimization report
        self._generate_optimization_report(output_dir / "optimization_summary.txt")
        
        if self.synthesis_result:
            self._generate_synthesis_report(output_dir / "synthesis_results.txt")
    
    def _generate_resource_report(self, file_path: Path) -> None:
        """Generate resource utilization report."""
        with open(file_path, 'w') as f:
            f.write("=== FPGA Resource Utilization Report ===\\n\\n")
            f.write(f"Total Neurons: {self.resource_estimate.neurons}\\n")
            f.write(f"Total Synapses: {self.resource_estimate.synapses}\\n")
            f.write(f"LUTs Required: {self.resource_estimate.luts}\\n")
            f.write(f"Registers Required: {self.resource_estimate.registers}\\n")
            f.write(f"BRAM Usage: {self.resource_estimate.bram_kb:.2f} KB\\n")
            f.write(f"DSP Slices: {self.resource_estimate.dsp_slices}\\n")
    
    def _generate_network_report(self, file_path: Path) -> None:
        """Generate network analysis report."""
        with open(file_path, 'w') as f:
            f.write("=== Network Analysis Report ===\\n\\n")
            f.write(f"Network Name: {self.network.name}\\n")
            f.write(f"Description: {self.network.description or 'N/A'}\\n")
            f.write(f"Total Layers: {len(self.network.layers)}\\n")
            f.write(f"Timestep: {self.network.timestep} ms\\n\\n")
            
            f.write("Layer Details:\\n")
            for layer in self.network.layers:
                f.write(f"  Layer {layer.layer_id}: {layer.layer_type.value}, ")
                f.write(f"Size: {layer.size}, Type: {layer.neuron_type}\\n")
    
    def _generate_optimization_report(self, file_path: Path) -> None:
        """Generate optimization summary report."""
        with open(file_path, 'w') as f:
            f.write("=== Optimization Summary Report ===\\n\\n")
            f.write(f"Optimization Level: {self.optimization_stats.get('optimization_level', 'N/A')}\\n")
            
            if 'pass_statistics' in self.optimization_stats:
                f.write("\\nOptimization Passes Applied:\\n")
                for pass_stat in self.optimization_stats['pass_statistics']:
                    f.write(f"  - {pass_stat.get('pass_name', 'Unknown')}\\n")
            
            if 'optimization_metrics' in self.optimization_stats:
                metrics = self.optimization_stats['optimization_metrics']
                f.write("\\nOptimization Results:\\n")
                f.write(f"  Neurons Reduced: {metrics.get('neuron_reduction', 0)}\\n")
                f.write(f"  Synapses Reduced: {metrics.get('synapse_reduction', 0)}\\n")
                f.write(f"  LUTs Saved: {metrics.get('lut_reduction', 0)}\\n")
    
    def _generate_synthesis_report(self, file_path: Path) -> None:
        """Generate synthesis results report."""
        with open(file_path, 'w') as f:
            f.write("=== Synthesis Results Report ===\\n\\n")
            f.write(f"Synthesis Success: {self.synthesis_result.success}\\n\\n")
            
            if self.synthesis_result.resource_utilization:
                f.write("Resource Utilization:\\n")
                for resource, value in self.synthesis_result.resource_utilization.items():
                    f.write(f"  {resource}: {value:.2f}%\\n")
            
            if self.synthesis_result.timing_summary:
                f.write("\\nTiming Summary:\\n")
                for metric, value in self.synthesis_result.timing_summary.items():
                    f.write(f"  {metric}: {value}\\n")


class NetworkCompiler:
    """Main compiler for spiking neural networks to FPGA hardware."""
    
    def __init__(self, target: FPGATarget):
        self.target = target
        self.optimization_pipeline = OptimizationPipeline()
        self.hdl_generator = HDLGenerator()
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend based on target
        if target.vendor == "xilinx":
            self.backend = VivadoBackend()
        elif target.vendor == "intel":
            self.backend = QuartusBackend()
        else:
            raise ValueError(f"Unsupported vendor: {target.vendor}")
    
    def compile(self, network: Union[Network, str, Path], 
               output_dir: Union[str, Path],
               config: CompilationConfig = None) -> CompilationResult:
        """Compile a spiking neural network to FPGA hardware."""
        if config is None:
            config = CompilationConfig()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting compilation for target: {self.target.value}")
        
        try:
            # Parse network if needed
            if isinstance(network, (str, Path)):
                self.logger.info(f"Parsing network from file: {network}")
                parsed_network = parse_network_file(Path(network))
            else:
                parsed_network = network
            
            # Validate network
            issues = parsed_network.validate_network()
            if issues:
                self.logger.warning(f"Network validation issues: {issues}")
            
            # Optimize network
            self.logger.info("Running optimization pipeline")
            optimized_network, optimization_stats = self.optimization_pipeline.optimize(
                network=parsed_network,
                target_platform=self.target.resources,
                optimization_level=config.optimization_level
            )
            
            # Generate HDL
            self.logger.info("Generating HDL code")
            hdl_config = HDLGenerationConfig(
                clock_frequency=config.clock_frequency,
                debug_enabled=config.debug_enabled
            )
            self.hdl_generator.config = hdl_config
            hdl_files = self.hdl_generator.generate(optimized_network, output_dir / "hdl")
            
            # Estimate final resources
            resource_estimate = self.optimization_pipeline.resource_optimizer.estimate_resources(
                optimized_network, self.optimization_pipeline.neuron_models
            )
            
            # Run synthesis if requested
            synthesis_result = None
            if config.run_synthesis:
                if self.backend.is_available():
                    self.logger.info("Running FPGA synthesis")
                    synthesis_result = self.backend.synthesize(
                        hdl_files, self.target, output_dir / "build"
                    )
                else:
                    self.logger.warning(f"Backend {self.target.toolchain} not available")
            
            # Create result
            result = CompilationResult(
                success=True,
                network=parsed_network,
                optimized_network=optimized_network,
                hdl_files=hdl_files,
                resource_estimate=resource_estimate,
                optimization_stats=optimization_stats,
                synthesis_result=synthesis_result
            )
            
            # Generate reports if requested
            if config.generate_reports:
                self.logger.info("Generating compilation reports")
                result.generate_reports(output_dir / "reports")
            
            self.logger.info("Compilation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Compilation failed: {str(e)}")
            return CompilationResult(
                success=False,
                network=network if isinstance(network, Network) else Network(name="failed"),
                optimized_network=Network(name="failed"),
                hdl_files={},
                resource_estimate=ResourceEstimate(),
                optimization_stats={},
                errors=[str(e)]
            )
    
    def add_optimization_pass(self, pass_obj) -> None:
        """Add a custom optimization pass."""
        self.optimization_pipeline.pass_manager.add_pass(pass_obj)
    
    def register_neuron_model(self, name: str, model) -> None:
        """Register a custom neuron model."""
        self.optimization_pipeline.register_neuron_model(name, model)
        self.hdl_generator.register_neuron_model(name, model)
    
    def estimate_resources(self, network: Network) -> ResourceEstimate:
        """Estimate FPGA resource usage without compilation."""
        return self.optimization_pipeline.resource_optimizer.estimate_resources(
            network, self.optimization_pipeline.neuron_models
        )
    
    def suggest_optimizations(self, network: Network) -> List[str]:
        """Suggest optimizations for the given network."""
        return self.optimization_pipeline.suggest_optimizations(
            network, self.target.resources
        )


def compile_network(network: Union[Network, str, Path],
                   target: FPGATarget,
                   output_dir: Union[str, Path] = "./output",
                   optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
                   power_budget_mw: Optional[float] = None,
                   run_synthesis: bool = False) -> CompilationResult:
    """Compile a spiking neural network to FPGA hardware.
    
    This is the main entry point for network compilation.
    
    Args:
        network: Network object or path to network definition file
        target: Target FPGA platform
        output_dir: Directory for generated files
        optimization_level: Level of optimization to apply
        power_budget_mw: Power budget in milliwatts (optional)
        run_synthesis: Whether to run FPGA synthesis
    
    Returns:
        CompilationResult with generated files and statistics
    
    Example:
        >>> from spiking_fpga import compile_network, FPGATarget
        >>> result = compile_network(
        ...     "my_network.yaml",
        ...     FPGATarget.ARTIX7_35T,
        ...     optimization_level=OptimizationLevel.AGGRESSIVE
        ... )
        >>> if result.success:
        ...     print(f"Generated HDL files: {list(result.hdl_files.keys())}")
    """
    config = CompilationConfig(
        optimization_level=optimization_level,
        power_budget_mw=power_budget_mw,
        run_synthesis=run_synthesis
    )
    
    compiler = NetworkCompiler(target)
    return compiler.compile(network, output_dir, config)