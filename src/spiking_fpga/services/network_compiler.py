"""
Main network compilation service that orchestrates the SNN-to-FPGA pipeline.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..models.network import SNNNetwork
from ..models.fpga import FPGATarget, ResourceUtilization, TimingConstraints
from .hdl_generator import HDLGenerator
from .resource_mapper import ResourceMapper
from .optimization_pipeline import OptimizationPipeline


@dataclass
class CompilationConfig:
    """Configuration for network compilation."""
    optimization_level: int = 2  # 0=none, 1=basic, 2=aggressive, 3=experimental
    target_frequency_mhz: float = 100.0
    enable_plasticity: bool = False
    enable_monitoring: bool = True
    output_directory: str = "./output"
    
    # Optimization passes to enable
    enable_spike_compression: bool = True
    enable_synapse_pruning: bool = True
    enable_neuron_clustering: bool = False
    synapse_pruning_threshold: float = 0.01
    
    # HDL generation options
    hdl_language: str = "verilog"  # "verilog" or "vhdl"
    generate_testbench: bool = True
    add_debug_probes: bool = False


@dataclass
class CompilationResult:
    """Results from network compilation."""
    success: bool = False
    network: Optional[SNNNetwork] = None
    fpga_target: Optional[FPGATarget] = None
    resource_utilization: Optional[ResourceUtilization] = None
    timing_constraints: Optional[TimingConstraints] = None
    
    # Generated files
    hdl_files: List[str] = field(default_factory=list)
    constraint_files: List[str] = field(default_factory=list)
    testbench_files: List[str] = field(default_factory=list)
    
    # Compilation metrics
    compilation_time_s: float = 0.0
    estimated_max_frequency_mhz: float = 0.0
    estimated_power_w: float = 0.0
    
    # Performance estimates
    max_spike_rate_mhz: float = 0.0
    inference_latency_ms: float = 0.0
    
    # Error/warning messages
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'success': self.success,
            'resource_utilization': {
                'luts_used': self.resource_utilization.luts_used if self.resource_utilization else 0,
                'bram_used': self.resource_utilization.bram_used if self.resource_utilization else 0,
                'dsp_used': self.resource_utilization.dsp_used if self.resource_utilization else 0,
            } if self.resource_utilization else None,
            'timing': {
                'target_frequency_mhz': self.timing_constraints.clock_frequency_mhz if self.timing_constraints else 0,
                'estimated_max_frequency_mhz': self.estimated_max_frequency_mhz,
            },
            'performance': {
                'max_spike_rate_mhz': self.max_spike_rate_mhz,
                'inference_latency_ms': self.inference_latency_ms,
                'estimated_power_w': self.estimated_power_w,
            },
            'files': {
                'hdl_files': self.hdl_files,
                'constraint_files': self.constraint_files,
                'testbench_files': self.testbench_files,
            },
            'compilation_time_s': self.compilation_time_s,
            'errors': self.errors,
            'warnings': self.warnings
        }


class NetworkCompiler:
    """Main compiler that transforms SNN specifications into FPGA implementations."""
    
    def __init__(self, config: Optional[CompilationConfig] = None):
        self.config = config or CompilationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize compilation pipeline components
        self.optimizer = OptimizationPipeline()
        self.resource_mapper = ResourceMapper()
        self.hdl_generator = HDLGenerator()
        
        # Register optimization passes based on config
        self._setup_optimization_passes()
    
    def _setup_optimization_passes(self):
        """Configure optimization passes based on compilation config."""
        if self.config.optimization_level == 0:
            return  # No optimizations
        
        if self.config.enable_spike_compression:
            self.optimizer.add_pass('spike_compression')
        
        if self.config.enable_synapse_pruning:
            self.optimizer.add_pass('synapse_pruning', 
                                  threshold=self.config.synapse_pruning_threshold)
        
        if self.config.optimization_level >= 2:
            if self.config.enable_neuron_clustering:
                self.optimizer.add_pass('neuron_clustering')
            self.optimizer.add_pass('memory_optimization')
        
        if self.config.optimization_level >= 3:
            self.optimizer.add_pass('pipeline_balancing')
            self.optimizer.add_pass('power_optimization')
    
    def compile_network(self, network: SNNNetwork, fpga_target: str) -> CompilationResult:
        """
        Main compilation entry point.
        
        Args:
            network: SNN network specification
            fpga_target: Target FPGA platform (e.g., 'artix7_35t')
            
        Returns:
            CompilationResult with generated files and metrics
        """
        import time
        start_time = time.time()
        
        result = CompilationResult()
        result.network = network
        
        try:
            # Validate network
            self.logger.info("Validating network specification...")
            if not network.validate_network():
                result.errors.append("Network validation failed")
                return result
            
            # Initialize FPGA target
            self.logger.info(f"Targeting FPGA: {fpga_target}")
            target = FPGATarget(fpga_target)
            result.fpga_target = target
            
            # Setup timing constraints
            timing = TimingConstraints(
                clock_frequency_mhz=self.config.target_frequency_mhz,
                pipeline_stages=3
            )
            timing.validate_constraints(target)
            result.timing_constraints = timing
            
            # Optimize network
            self.logger.info("Running optimization passes...")
            optimized_network = self.optimizer.optimize(network)
            
            # Map to FPGA resources
            self.logger.info("Mapping network to FPGA resources...")
            placement_result = self.resource_mapper.map_network(optimized_network, target)
            result.resource_utilization = placement_result.resource_usage
            
            # Check resource constraints
            violations = result.resource_utilization.check_constraints(target)
            if violations:
                result.errors.extend(violations)
                return result
            
            # Generate HDL
            self.logger.info("Generating HDL implementation...")
            hdl_result = self.hdl_generator.generate_implementation(
                optimized_network, target, placement_result, timing
            )
            
            # Create output directory
            output_path = Path(self.config.output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Write generated files
            result.hdl_files = self._write_hdl_files(hdl_result, output_path)
            result.constraint_files = self._write_constraint_files(hdl_result, output_path)
            
            if self.config.generate_testbench:
                result.testbench_files = self._write_testbench_files(hdl_result, output_path)
            
            # Calculate performance estimates
            result.estimated_max_frequency_mhz = self._estimate_max_frequency(
                result.resource_utilization, target
            )
            result.estimated_power_w = target.estimate_power_consumption(result.resource_utilization)
            result.max_spike_rate_mhz = timing.calculate_throughput(
                sum(layer.size for layer in network.layers)
            ) / 1e6
            result.inference_latency_ms = self._estimate_inference_latency(network, timing)
            
            # Write compilation report
            self._write_compilation_report(result, output_path)
            
            result.success = True
            self.logger.info(f"Compilation completed successfully in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Compilation failed: {str(e)}")
            result.errors.append(str(e))
            result.success = False
        
        result.compilation_time_s = time.time() - start_time
        return result
    
    def _write_hdl_files(self, hdl_result: Dict, output_path: Path) -> List[str]:
        """Write HDL files to output directory."""
        hdl_files = []
        
        for module_name, module_code in hdl_result.get('modules', {}).items():
            filename = f"{module_name}.v"
            file_path = output_path / filename
            with open(file_path, 'w') as f:
                f.write(module_code)
            hdl_files.append(str(file_path))
        
        # Write top-level module
        if 'top_module' in hdl_result:
            top_file = output_path / "snn_top.v"
            with open(top_file, 'w') as f:
                f.write(hdl_result['top_module'])
            hdl_files.append(str(top_file))
        
        return hdl_files
    
    def _write_constraint_files(self, hdl_result: Dict, output_path: Path) -> List[str]:
        """Write timing and placement constraint files."""
        constraint_files = []
        
        # Xilinx XDC constraints
        if 'xdc_constraints' in hdl_result:
            xdc_file = output_path / "timing_constraints.xdc"
            with open(xdc_file, 'w') as f:
                f.write(hdl_result['xdc_constraints'])
            constraint_files.append(str(xdc_file))
        
        # Intel SDC constraints
        if 'sdc_constraints' in hdl_result:
            sdc_file = output_path / "timing_constraints.sdc"
            with open(sdc_file, 'w') as f:
                f.write(hdl_result['sdc_constraints'])
            constraint_files.append(str(sdc_file))
        
        return constraint_files
    
    def _write_testbench_files(self, hdl_result: Dict, output_path: Path) -> List[str]:
        """Write testbench files for simulation."""
        testbench_files = []
        
        if 'testbench' in hdl_result:
            tb_file = output_path / "snn_testbench.v"
            with open(tb_file, 'w') as f:
                f.write(hdl_result['testbench'])
            testbench_files.append(str(tb_file))
        
        # Write stimulus files
        if 'stimulus_data' in hdl_result:
            stimulus_file = output_path / "stimulus_data.mem"
            with open(stimulus_file, 'w') as f:
                for data_line in hdl_result['stimulus_data']:
                    f.write(f"{data_line}\n")
            testbench_files.append(str(stimulus_file))
        
        return testbench_files
    
    def _write_compilation_report(self, result: CompilationResult, output_path: Path):
        """Write detailed compilation report."""
        report_file = output_path / "compilation_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Also write human-readable report
        report_txt = output_path / "compilation_report.txt"
        with open(report_txt, 'w') as f:
            f.write("Spiking FPGA Toolchain - Compilation Report\n")
            f.write("=" * 50 + "\n\n")
            
            if result.network:
                f.write(f"Network: {result.network.name}\n")
                f.write(f"Layers: {len(result.network.layers)}\n")
                f.write(f"Connections: {len(result.network.connections)}\n\n")
            
            if result.fpga_target:
                f.write(f"Target FPGA: {result.fpga_target.target}\n\n")
            
            if result.resource_utilization:
                f.write("Resource Utilization:\n")
                f.write(f"  LUTs: {result.resource_utilization.luts_used}\n")
                f.write(f"  BRAM (bits): {result.resource_utilization.bram_used}\n")
                f.write(f"  DSP Slices: {result.resource_utilization.dsp_used}\n\n")
            
            f.write("Performance Estimates:\n")
            f.write(f"  Max Spike Rate: {result.max_spike_rate_mhz:.2f} Mspikes/s\n")
            f.write(f"  Inference Latency: {result.inference_latency_ms:.2f} ms\n")
            f.write(f"  Estimated Power: {result.estimated_power_w:.2f} W\n\n")
            
            f.write(f"Compilation Time: {result.compilation_time_s:.2f} seconds\n")
            
            if result.errors:
                f.write("\nErrors:\n")
                for error in result.errors:
                    f.write(f"  - {error}\n")
            
            if result.warnings:
                f.write("\nWarnings:\n")
                for warning in result.warnings:
                    f.write(f"  - {warning}\n")
    
    def _estimate_max_frequency(self, utilization: ResourceUtilization, target: FPGATarget) -> float:
        """Estimate maximum achievable frequency based on resource utilization."""
        # Simple model: frequency decreases with utilization due to routing delays
        base_freq = target.specs.max_frequency_mhz
        
        # Calculate utilization factor (0-1)
        lut_util = utilization.luts_used / target.specs.luts
        bram_util = utilization.bram_used / target.specs.bram_bits
        
        # Routing delays increase with utilization
        routing_factor = 1.0 - 0.3 * max(lut_util, bram_util)
        
        return base_freq * routing_factor
    
    def _estimate_inference_latency(self, network: SNNNetwork, timing: TimingConstraints) -> float:
        """Estimate inference latency for the network."""
        # Simple model: latency = pipeline_depth + simulation_time
        pipeline_latency = (timing.pipeline_stages / timing.clock_frequency_mhz) * 1000  # ms
        simulation_latency = network.parameters.simulation_time
        
        return pipeline_latency + simulation_latency
    
    def register_custom_neuron_model(self, model_name: str, model_class: type):
        """Register a custom neuron model for compilation."""
        self.hdl_generator.register_neuron_model(model_name, model_class)
    
    def get_supported_targets(self) -> List[str]:
        """Get list of supported FPGA targets."""
        return list(FPGATarget.FPGA_SPECS.keys())
    
    def estimate_resources(self, network: SNNNetwork, fpga_target: str) -> ResourceUtilization:
        """Quick resource estimation without full compilation."""
        target = FPGATarget(fpga_target)
        return self.resource_mapper.estimate_resources(network, target)