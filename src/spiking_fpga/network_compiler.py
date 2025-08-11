"""Main network compiler interface for the Spiking-FPGA-Toolchain."""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass
from datetime import datetime

from .core import FPGATarget
from .models.network import Network
from .models.optimization import OptimizationLevel, ResourceEstimate
from .compiler.frontend import parse_network_file, get_parser
from .compiler.optimizer import OptimizationPipeline
from .compiler.backend import HDLGenerator, HDLGenerationConfig, VivadoBackend, QuartusBackend, SynthesisResult
from .utils import (
    StructuredLogger, configure_logging, CompilationTracker,
    NetworkValidator, ConfigurationValidator, FileValidator, ValidationResult,
    HealthMonitor, PerformanceTimer, CircuitBreaker, CompilationMetrics
)


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
    
    def __init__(self, target: FPGATarget, enable_monitoring: bool = True, 
                 log_level: str = "INFO", log_file: Optional[Path] = None):
        self.target = target
        self.optimization_pipeline = OptimizationPipeline()
        self.hdl_generator = HDLGenerator()
        
        # Initialize robust logging and monitoring
        self.logger = configure_logging(log_level, log_file)
        self.compilation_tracker = CompilationTracker(self.logger)
        
        # Initialize validators
        self.network_validator = NetworkValidator()
        self.config_validator = ConfigurationValidator()
        self.file_validator = FileValidator()
        
        # Initialize monitoring
        self.health_monitor = None
        if enable_monitoring:
            metrics_file = log_file.parent / "metrics.jsonl" if log_file else None
            self.health_monitor = HealthMonitor(self.logger, metrics_file)
            self.health_monitor.start_monitoring()
        
        # Initialize backend with circuit breaker protection
        self.circuit_breaker = CircuitBreaker(logger=self.logger)
        if target.vendor == "xilinx":
            self.backend = VivadoBackend()
        elif target.vendor == "intel":
            self.backend = QuartusBackend()
        else:
            raise ValueError(f"Unsupported vendor: {target.vendor}")
        
        self.logger.info("NetworkCompiler initialized", 
                        target=target.value, 
                        vendor=target.vendor,
                        monitoring_enabled=enable_monitoring)
    
    def compile(self, network: Union[Network, str, Path], 
               output_dir: Union[str, Path],
               config: CompilationConfig = None) -> CompilationResult:
        """Compile a spiking neural network to FPGA hardware."""
        if config is None:
            config = CompilationConfig()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize compilation metrics
        optimization_level_name = (
            config.optimization_level.name 
            if hasattr(config.optimization_level, 'name') 
            else str(config.optimization_level)
        )
        
        compilation_metrics = CompilationMetrics(
            network_name="unknown",
            target=self.target.value,
            start_time=datetime.utcnow(),
            optimization_level=optimization_level_name
        )
        
        try:
            # Start compilation tracking
            self.compilation_tracker.start_compilation("network", self.target.value)
            
            # Parse and validate network file if needed
            with PerformanceTimer("network_parsing", self.logger) as parsing_timer:
                if isinstance(network, (str, Path)):
                    network_path = Path(network)
                    
                    # Validate file first
                    file_validation = self.file_validator.validate_network_file(network_path)
                    if not file_validation.valid:
                        compilation_metrics.success = False
                        compilation_metrics.error_message = "; ".join(file_validation.issues)
                        return self._create_error_result(network, file_validation.issues)
                    
                    self.logger.info("Parsing network from file", file_path=str(network_path))
                    parsed_network = parse_network_file(network_path)
                    compilation_metrics.network_name = parsed_network.name
                else:
                    parsed_network = network
                    compilation_metrics.network_name = parsed_network.name
            
            compilation_metrics.parsing_time_seconds = parsing_timer.elapsed_seconds()
            
            # Comprehensive network validation
            with PerformanceTimer("network_validation", self.logger) as validation_timer:
                network_validation = self.network_validator.validate_network(parsed_network)
                
                # Log validation results
                if network_validation.warnings:
                    self.logger.warning("Network validation warnings", 
                                      warnings=network_validation.warnings)
                
                if network_validation.recommendations:
                    self.logger.info("Network optimization recommendations", 
                                   recommendations=network_validation.recommendations)
                
                if not network_validation.valid:
                    compilation_metrics.success = False  
                    compilation_metrics.error_message = "; ".join(network_validation.issues)
                    return self._create_error_result(parsed_network, network_validation.issues)
            
            # Update compilation metrics
            compilation_metrics.neuron_count = len(parsed_network.neurons)
            compilation_metrics.synapse_count = len(parsed_network.synapses)
            compilation_metrics.layer_count = len(parsed_network.layers)
            
            # Validate configuration
            config_validation = self.config_validator.validate_optimization_config(
                config.optimization_level
            )
            if not config_validation.valid:
                return self._create_error_result(parsed_network, config_validation.issues)
            
            # Run optimization pipeline
            with PerformanceTimer("network_optimization", self.logger) as optimization_timer:
                self.logger.info("Running optimization pipeline")
                optimized_network, optimization_stats = self.optimization_pipeline.optimize(
                    network=parsed_network,
                    target_platform=self.target.resources,
                    optimization_level=config.optimization_level
                )
            
            compilation_metrics.optimization_time_seconds = optimization_timer.elapsed_seconds()
            
            # Update optimization metrics  
            if "optimization_metrics" in optimization_stats:
                metrics = optimization_stats["optimization_metrics"]
                compilation_metrics.synapses_pruned = metrics.get("synapse_reduction", 0)
                compilation_metrics.neurons_clustered = metrics.get("neuron_reduction", 0)
            
            # Validate resource constraints
            with PerformanceTimer("resource_estimation", self.logger) as estimation_timer:
                resource_estimate = self.optimization_pipeline.resource_optimizer.estimate_resources(
                    optimized_network, self.optimization_pipeline.neuron_models
                )
                
                # Update compilation metrics
                compilation_metrics.estimated_luts = resource_estimate.luts
                compilation_metrics.estimated_memory_kb = resource_estimate.bram_kb
                compilation_metrics.estimated_dsp_slices = resource_estimate.dsp_slices
                
                # Validate target compatibility
                target_validation = self.config_validator.validate_fpga_target(
                    self.target, resource_estimate
                )
                
                if target_validation.warnings:
                    self.logger.warning("Target resource warnings", 
                                      warnings=target_validation.warnings)
                
                if not target_validation.valid:
                    self.logger.error("Network exceeds target FPGA resources", 
                                    issues=target_validation.issues)
                    # Continue anyway but log the issues
            
            # Generate HDL
            self.logger.info("Generating HDL code")
            hdl_config = HDLGenerationConfig(
                clock_frequency=config.clock_frequency,
                debug_enabled=config.debug_enabled
            )
            self.hdl_generator.config = hdl_config
            
            with PerformanceTimer("hdl_generation", self.logger) as hdl_timer:
                hdl_files = self.hdl_generator.generate(optimized_network, output_dir / "hdl")
            
            compilation_metrics.hdl_generation_time_seconds = hdl_timer.elapsed_seconds()
            
            # Run synthesis if requested
            synthesis_result = None
            if config.run_synthesis:
                with PerformanceTimer("synthesis", self.logger) as synthesis_timer:
                    if self.backend.is_available():
                        self.logger.info("Running FPGA synthesis")
                        try:
                            synthesis_result = self.circuit_breaker.call(
                                self.backend.synthesize, hdl_files, self.target, output_dir / "build"
                            )
                        except Exception as e:
                            self.logger.error("Synthesis failed", error=str(e))
                            synthesis_result = None
                    else:
                        self.logger.warning("Backend not available", 
                                          toolchain=self.target.toolchain)
                
                compilation_metrics.synthesis_time_seconds = synthesis_timer.elapsed_seconds()
            
            # Create successful result
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
            
            # Mark compilation as successful
            compilation_metrics.success = True
            compilation_metrics.end_time = datetime.utcnow()
            
            # Track completion
            if self.health_monitor:
                self.health_monitor.add_compilation_metrics(compilation_metrics)
            
            self.compilation_tracker.finish_compilation(True)
            self.logger.info("Compilation completed successfully", 
                           duration_seconds=compilation_metrics.duration_seconds())
            
            return result
            
        except Exception as e:
            # Mark compilation as failed
            compilation_metrics.success = False
            compilation_metrics.error_message = str(e)
            compilation_metrics.end_time = datetime.utcnow()
            
            if self.health_monitor:
                self.health_monitor.add_compilation_metrics(compilation_metrics)
            
            self.compilation_tracker.finish_compilation(False, str(e))
            self.logger.error("Compilation failed", 
                            error=str(e),
                            duration_seconds=compilation_metrics.duration_seconds())
            
            return self._create_error_result(network, [str(e)])
    
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
    
    def _create_error_result(self, network: Union[Network, str, Path], 
                           errors: List[str]) -> CompilationResult:
        """Create a CompilationResult for errors."""
        if isinstance(network, Network):
            network_obj = network
        else:
            network_obj = Network(name="failed_network")
        
        return CompilationResult(
            success=False,
            network=network_obj,
            optimized_network=Network(name="failed"),
            hdl_files={},
            resource_estimate=ResourceEstimate(),
            optimization_stats={},
            errors=errors
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health and performance status."""
        if self.health_monitor:
            return {
                "system_health": self.health_monitor.get_current_health(),
                "compilation_stats": self.health_monitor.get_compilation_stats()
            }
        return {"monitoring_disabled": True}
    
    def __del__(self):
        """Cleanup monitoring on destruction."""
        if hasattr(self, 'health_monitor') and self.health_monitor:
            self.health_monitor.stop_monitoring()


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