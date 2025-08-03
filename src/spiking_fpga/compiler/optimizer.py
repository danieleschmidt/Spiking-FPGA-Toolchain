"""Optimization pipeline for SNN compilation."""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from ..models.network import Network
from ..models.optimization import (
    OptimizationPass, OptimizationLevel, ResourceEstimate,
    SynapseProning, SpikeCompression, NeuronClustering, ResourceOptimizer
)
from ..models.neuron_models import NeuronModel, LIFNeuron, IzhikevichNeuron, AdaptiveLIFNeuron


@dataclass
class PassConfiguration:
    """Configuration for an optimization pass."""
    
    pass_type: str
    enabled: bool = True
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class PassManager:
    """Manages optimization passes and their execution order."""
    
    def __init__(self):
        self.passes: List[OptimizationPass] = []
        self.pass_stats: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_pass(self, pass_obj: OptimizationPass) -> None:
        """Add an optimization pass to the pipeline."""
        self.passes.append(pass_obj)
        self.logger.info(f"Added optimization pass: {pass_obj.name}")
    
    def configure_passes(self, level: OptimizationLevel, 
                        custom_config: List[PassConfiguration] = None) -> None:
        """Configure passes based on optimization level."""
        self.passes.clear()
        
        if level == OptimizationLevel.NONE:
            return
        
        # Default pass configurations by level
        if level.value >= 1:  # Basic
            self.add_pass(SynapseProning(weight_threshold=0.01))
        
        if level.value >= 2:  # Aggressive
            self.add_pass(NeuronClustering(cluster_size=16))
            self.add_pass(SynapseProning(weight_threshold=0.005))  # More aggressive pruning
        
        if level.value >= 3:  # Maximum
            self.add_pass(SpikeCompression(compression_ratio=0.7))
            self.add_pass(NeuronClustering(cluster_size=32))  # Larger clusters
        
        # Apply custom configurations if provided
        if custom_config:
            self._apply_custom_config(custom_config)
    
    def _apply_custom_config(self, config: List[PassConfiguration]) -> None:
        """Apply custom pass configuration."""
        for pass_config in config:
            if not pass_config.enabled:
                # Remove disabled passes
                self.passes = [p for p in self.passes if p.name != pass_config.pass_type]
            else:
                # Update parameters for enabled passes
                for pass_obj in self.passes:
                    if pass_obj.name == pass_config.pass_type:
                        # Update pass parameters (implementation specific)
                        break
    
    def run_passes(self, network: Network) -> Tuple[Network, List[Dict[str, Any]]]:
        """Run all configured optimization passes."""
        current_network = network
        all_stats = []
        
        for i, pass_obj in enumerate(self.passes):
            self.logger.info(f"Running pass {i+1}/{len(self.passes)}: {pass_obj.name}")
            
            try:
                current_network, pass_stats = pass_obj.apply(current_network)
                pass_stats["pass_name"] = pass_obj.name
                pass_stats["pass_index"] = i
                all_stats.append(pass_stats)
                
                self.logger.info(f"Pass completed: {pass_obj.name}")
                
            except Exception as e:
                self.logger.error(f"Pass failed: {pass_obj.name} - {str(e)}")
                raise RuntimeError(f"Optimization pass '{pass_obj.name}' failed: {str(e)}")
        
        self.pass_stats = all_stats
        return current_network, all_stats


class OptimizationPipeline:
    """Complete optimization pipeline for SNN compilation."""
    
    def __init__(self):
        self.pass_manager = PassManager()
        self.resource_optimizer = ResourceOptimizer()
        self.neuron_models = {
            "LIF": LIFNeuron(),
            "Izhikevich": IzhikevichNeuron(), 
            "AdaptiveLIF": AdaptiveLIFNeuron(),
        }
        self.logger = logging.getLogger(__name__)
    
    def register_neuron_model(self, name: str, model: NeuronModel) -> None:
        """Register a custom neuron model."""
        self.neuron_models[name] = model
        self.logger.info(f"Registered neuron model: {name}")
    
    def optimize(self, network: Network, target_platform: Dict[str, Any],
                optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
                custom_passes: List[PassConfiguration] = None) -> Tuple[Network, Dict[str, Any]]:
        """Run complete optimization pipeline."""
        
        self.logger.info(f"Starting optimization pipeline for network: {network.name}")
        self.logger.info(f"Optimization level: {optimization_level.name}")
        self.logger.info(f"Target platform: {target_platform}")
        
        # Initial resource estimation
        initial_estimate = self.resource_optimizer.estimate_resources(network, self.neuron_models)
        self.logger.info(f"Initial resource estimate: {initial_estimate}")
        
        # Configure and run optimization passes
        self.pass_manager.configure_passes(optimization_level, custom_passes)
        optimized_network, pass_stats = self.pass_manager.run_passes(network)
        
        # Final resource estimation
        final_estimate = self.resource_optimizer.estimate_resources(optimized_network, self.neuron_models)
        self.logger.info(f"Final resource estimate: {final_estimate}")
        
        # Check resource constraints
        constraints = self.resource_optimizer.check_resource_constraints(
            final_estimate, target_platform
        )
        
        # Calculate optimization metrics
        optimization_stats = {
            "optimization_level": optimization_level.name,
            "target_platform": target_platform,
            "initial_estimate": {
                "neurons": initial_estimate.neurons,
                "synapses": initial_estimate.synapses,
                "luts": initial_estimate.luts,
                "registers": initial_estimate.registers,
                "bram_kb": initial_estimate.bram_kb,
                "dsp_slices": initial_estimate.dsp_slices,
            },
            "final_estimate": {
                "neurons": final_estimate.neurons,
                "synapses": final_estimate.synapses,
                "luts": final_estimate.luts,
                "registers": final_estimate.registers,
                "bram_kb": final_estimate.bram_kb,
                "dsp_slices": final_estimate.dsp_slices,
            },
            "resource_constraints": constraints,
            "utilization": final_estimate.utilization_percentage(target_platform),
            "pass_statistics": pass_stats,
            "optimization_metrics": {
                "neuron_reduction": initial_estimate.neurons - final_estimate.neurons,
                "synapse_reduction": initial_estimate.synapses - final_estimate.synapses,
                "lut_reduction": initial_estimate.luts - final_estimate.luts,
                "memory_reduction": initial_estimate.bram_kb - final_estimate.bram_kb,
            }
        }
        
        # Warn if resource constraints are violated
        if not constraints["overall_ok"]:
            self.logger.warning("Optimized network exceeds target platform resources!")
            for resource, ok in constraints.items():
                if not ok and resource != "overall_ok":
                    utilization = optimization_stats["utilization"].get(resource.replace("_ok", ""), 0)
                    self.logger.warning(f"  {resource}: {utilization:.1f}% utilization")
        
        self.logger.info("Optimization pipeline completed successfully")
        
        return optimized_network, optimization_stats
    
    def suggest_optimizations(self, network: Network, 
                            target_platform: Dict[str, Any]) -> List[str]:
        """Suggest optimizations based on network characteristics."""
        suggestions = []
        
        # Analyze network structure
        total_neurons = len(network.neurons)
        total_synapses = len(network.synapses)
        avg_fanout = total_synapses / total_neurons if total_neurons > 0 else 0
        
        # Estimate resources without optimization
        estimate = self.resource_optimizer.estimate_resources(network, self.neuron_models)
        utilization = estimate.utilization_percentage(target_platform)
        
        # Generate suggestions based on analysis
        if utilization.get("logic", 0) > 80:
            suggestions.append("Consider neuron clustering to reduce logic utilization")
        
        if utilization.get("memory", 0) > 75:
            suggestions.append("Consider synapse pruning to reduce memory usage")
            if avg_fanout > 50:
                suggestions.append("High connectivity detected - aggressive pruning recommended")
        
        if utilization.get("dsp", 0) > 90:
            suggestions.append("DSP utilization high - consider simpler neuron models")
        
        if total_synapses > 100000:
            suggestions.append("Large network detected - spike compression could improve bandwidth")
        
        if len(suggestions) == 0:
            suggestions.append("Network should fit target platform with basic optimization")
        
        return suggestions