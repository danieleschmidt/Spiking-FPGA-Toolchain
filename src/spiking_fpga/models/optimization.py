"""Optimization passes and resource estimation for SNN compilation."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from .network import Network, Synapse
from .neuron_models import NeuronModel


class OptimizationLevel(Enum):
    """Optimization levels for compilation."""
    NONE = 0
    BASIC = 1  
    AGGRESSIVE = 2
    MAXIMUM = 3


@dataclass
class ResourceEstimate:
    """FPGA resource usage estimate."""
    
    luts: int = 0
    registers: int = 0
    bram_kb: float = 0.0
    dsp_slices: int = 0
    neurons: int = 0
    synapses: int = 0
    
    def __add__(self, other: 'ResourceEstimate') -> 'ResourceEstimate':
        """Add two resource estimates."""
        return ResourceEstimate(
            luts=self.luts + other.luts,
            registers=self.registers + other.registers,
            bram_kb=self.bram_kb + other.bram_kb,
            dsp_slices=self.dsp_slices + other.dsp_slices,
            neurons=self.neurons + other.neurons,
            synapses=self.synapses + other.synapses
        )
    
    def utilization_percentage(self, target_resources: Dict[str, Any]) -> Dict[str, float]:
        """Calculate utilization percentage for a target platform."""
        utilization = {}
        
        if "logic_cells" in target_resources:
            utilization["logic"] = (self.luts / target_resources["logic_cells"]) * 100
        elif "logic_elements" in target_resources:
            utilization["logic"] = (self.luts / target_resources["logic_elements"]) * 100
            
        if "bram_kb" in target_resources:
            utilization["memory"] = (self.bram_kb / target_resources["bram_kb"]) * 100
        elif "m10k_blocks" in target_resources:
            # Each M10K block is ~10KB
            utilization["memory"] = (self.bram_kb / (target_resources["m10k_blocks"] * 10)) * 100
            
        if "dsp_slices" in target_resources:
            utilization["dsp"] = (self.dsp_slices / target_resources["dsp_slices"]) * 100
        elif "dsp_blocks" in target_resources:
            utilization["dsp"] = (self.dsp_slices / target_resources["dsp_blocks"]) * 100
            
        return utilization


class OptimizationPass:
    """Base class for optimization passes."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def apply(self, network: Network) -> Tuple[Network, Dict[str, Any]]:
        """Apply optimization pass to network."""
        raise NotImplementedError("Subclasses must implement apply method")


class SynapseProning(OptimizationPass):
    """Remove synapses with weights below threshold."""
    
    def __init__(self, weight_threshold: float = 0.01):
        super().__init__(
            "Synapse Pruning",
            f"Remove synapses with |weight| < {weight_threshold}"
        )
        self.weight_threshold = weight_threshold
    
    def apply(self, network: Network) -> Tuple[Network, Dict[str, Any]]:
        """Apply synapse pruning optimization."""
        original_count = len(network.synapses)
        
        # Filter out weak synapses
        strong_synapses = [
            s for s in network.synapses 
            if abs(s.weight) >= self.weight_threshold
        ]
        
        # Create optimized network
        optimized_network = Network(
            name=network.name + "_pruned",
            description=network.description,
            layers=network.layers.copy(),
            neurons=network.neurons.copy(),
            synapses=strong_synapses,
            timestep=network.timestep
        )
        
        stats = {
            "original_synapses": original_count,
            "pruned_synapses": len(strong_synapses),
            "reduction_percentage": ((original_count - len(strong_synapses)) / original_count) * 100,
            "weight_threshold": self.weight_threshold
        }
        
        return optimized_network, stats


class SpikeCompression(OptimizationPass):
    """Compress spike representation for efficient routing."""
    
    def __init__(self, compression_ratio: float = 0.8):
        super().__init__(
            "Spike Compression",
            f"Compress spike data to {compression_ratio:.1%} of original size"
        )
        self.compression_ratio = compression_ratio
    
    def apply(self, network: Network) -> Tuple[Network, Dict[str, Any]]:
        """Apply spike compression optimization."""
        # This is a conceptual optimization - in practice would modify
        # the HDL generation to use compressed spike packets
        
        stats = {
            "compression_ratio": self.compression_ratio,
            "bandwidth_reduction": (1 - self.compression_ratio) * 100,
            "method": "temporal_coding"
        }
        
        return network, stats


class NeuronClustering(OptimizationPass):
    """Group neurons into clusters for resource sharing."""
    
    def __init__(self, cluster_size: int = 16):
        super().__init__(
            "Neuron Clustering", 
            f"Group neurons into clusters of {cluster_size}"
        )
        self.cluster_size = cluster_size
    
    def apply(self, network: Network) -> Tuple[Network, Dict[str, Any]]:
        """Apply neuron clustering optimization."""
        total_neurons = len(network.neurons)
        num_clusters = (total_neurons + self.cluster_size - 1) // self.cluster_size
        
        # Create clustered network (conceptual - would modify HDL generation)
        stats = {
            "total_neurons": total_neurons,
            "cluster_size": self.cluster_size,
            "num_clusters": num_clusters,
            "resource_sharing": True,
            "memory_reduction": min(50, (self.cluster_size / 4) * 10)  # Estimated savings
        }
        
        return network, stats


class ResourceOptimizer:
    """Analyzes and optimizes network resource usage."""
    
    def __init__(self):
        self.passes = {
            "pruning": SynapseProning,
            "compression": SpikeCompression, 
            "clustering": NeuronClustering
        }
    
    def estimate_resources(self, network: Network, 
                          neuron_models: Dict[str, NeuronModel]) -> ResourceEstimate:
        """Estimate total resource usage for network."""
        estimate = ResourceEstimate()
        
        # Estimate neuron resources
        for neuron in network.neurons:
            if neuron.neuron_type in neuron_models:
                model = neuron_models[neuron.neuron_type]
                neuron_resources = model.get_resource_estimate()
                estimate.luts += neuron_resources.get("luts", 0)
                estimate.registers += neuron_resources.get("registers", 0)
                estimate.dsp_slices += neuron_resources.get("dsp_slices", 0)
        
        # Estimate synapse memory requirements
        # Each synapse requires: weight (16 bits) + metadata (16 bits) = 32 bits
        synapse_memory_bits = len(network.synapses) * 32
        estimate.bram_kb = synapse_memory_bits / (8 * 1024)  # Convert to KB
        
        # Add routing and control overhead (empirical estimates)
        routing_overhead = max(100, len(network.neurons) * 2)  # LUTs for routing
        estimate.luts += routing_overhead
        estimate.registers += max(50, len(network.neurons))    # Control registers
        
        estimate.neurons = len(network.neurons)
        estimate.synapses = len(network.synapses)
        
        return estimate
    
    def optimize_for_target(self, network: Network, target_resources: Dict[str, Any],
                           optimization_level: OptimizationLevel) -> Tuple[Network, Dict[str, Any]]:
        """Optimize network to fit target FPGA resources."""
        optimized_network = network
        optimization_stats = {}
        
        if optimization_level == OptimizationLevel.NONE:
            return network, {"level": "none", "passes": []}
        
        applied_passes = []
        
        # Apply optimization passes based on level
        if optimization_level.value >= 1:
            # Basic optimization: prune weak synapses
            pruning_pass = SynapseProning(weight_threshold=0.01)
            optimized_network, pruning_stats = pruning_pass.apply(optimized_network)
            applied_passes.append({"pass": "pruning", "stats": pruning_stats})
        
        if optimization_level.value >= 2:
            # Aggressive optimization: add clustering
            clustering_pass = NeuronClustering(cluster_size=16)
            optimized_network, clustering_stats = clustering_pass.apply(optimized_network)
            applied_passes.append({"pass": "clustering", "stats": clustering_stats})
        
        if optimization_level.value >= 3:
            # Maximum optimization: add compression
            compression_pass = SpikeCompression(compression_ratio=0.7)
            optimized_network, compression_stats = compression_pass.apply(optimized_network)
            applied_passes.append({"pass": "compression", "stats": compression_stats})
        
        optimization_stats = {
            "level": optimization_level.name.lower(),
            "passes": applied_passes,
            "target_resources": target_resources
        }
        
        return optimized_network, optimization_stats
    
    def check_resource_constraints(self, estimate: ResourceEstimate, 
                                 target_resources: Dict[str, Any]) -> Dict[str, bool]:
        """Check if resource estimate fits within target constraints."""
        utilization = estimate.utilization_percentage(target_resources)
        
        constraints = {}
        constraints["logic_ok"] = utilization.get("logic", 0) <= 90  # 90% max utilization
        constraints["memory_ok"] = utilization.get("memory", 0) <= 85  # 85% max utilization  
        constraints["dsp_ok"] = utilization.get("dsp", 0) <= 95      # 95% max utilization
        constraints["overall_ok"] = all(constraints.values())
        
        return constraints