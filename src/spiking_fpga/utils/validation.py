"""Validation utilities for network and configuration validation."""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import re
import json
import yaml

from ..models.network import Network
from ..core import FPGATarget
from ..models.optimization import OptimizationLevel, ResourceEstimate


@dataclass 
class ValidationResult:
    """Result of validation with issues and recommendations."""
    
    valid: bool
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    def __post_init__(self):
        if not self.issues:
            self.issues = []
        if not self.warnings:
            self.warnings = []
        if not self.recommendations:
            self.recommendations = []
    
    def add_issue(self, message: str):
        """Add a validation issue."""
        self.issues.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        """Add a validation warning."""
        self.warnings.append(message)
    
    def add_recommendation(self, message: str):
        """Add a recommendation."""
        self.recommendations.append(message)


class NetworkValidator:
    """Comprehensive network validation."""
    
    def __init__(self):
        self.min_neurons = 1
        self.max_neurons = 1_000_000
        self.min_synapses = 0
        self.max_synapses = 10_000_000
        self.max_fanout = 1000
        self.max_fanin = 1000
    
    def validate_network(self, network: Network) -> ValidationResult:
        """Perform comprehensive network validation."""
        result = ValidationResult(valid=True, issues=[], warnings=[], recommendations=[])
        
        # Basic structure validation
        self._validate_structure(network, result)
        
        # Connectivity validation
        self._validate_connectivity(network, result)
        
        # Neuron parameters validation
        self._validate_neuron_parameters(network, result)
        
        # Performance recommendations
        self._generate_recommendations(network, result)
        
        return result
    
    def _validate_structure(self, network: Network, result: ValidationResult):
        """Validate basic network structure."""
        # Check neuron count
        neuron_count = len(network.neurons)
        if neuron_count < self.min_neurons:
            result.add_issue(f"Network has too few neurons: {neuron_count} < {self.min_neurons}")
        elif neuron_count > self.max_neurons:
            result.add_issue(f"Network has too many neurons: {neuron_count} > {self.max_neurons}")
        
        # Check synapse count
        synapse_count = len(network.synapses)
        if synapse_count > self.max_synapses:
            result.add_issue(f"Network has too many synapses: {synapse_count} > {self.max_synapses}")
        
        # Check layer structure
        if not network.layers:
            result.add_issue("Network has no layers defined")
        
        layer_sizes = [layer.size for layer in network.layers]
        if any(size <= 0 for size in layer_sizes):
            result.add_issue("All layers must have positive size")
        
        # Check for empty layers
        empty_layers = [i for i, layer in enumerate(network.layers) if layer.size == 0]
        if empty_layers:
            result.add_warning(f"Empty layers found: {empty_layers}")
    
    def _validate_connectivity(self, network: Network, result: ValidationResult):
        """Validate network connectivity."""
        if not network.synapses:
            result.add_warning("Network has no synaptic connections")
            return
        
        # Check for orphaned neurons
        connected_neurons = set()
        for synapse in network.synapses:
            connected_neurons.add(synapse.pre_neuron_id)
            connected_neurons.add(synapse.post_neuron_id)
        
        all_neuron_ids = {n.neuron_id for n in network.neurons}
        orphaned = all_neuron_ids - connected_neurons
        if orphaned:
            result.add_warning(f"Found {len(orphaned)} orphaned neurons with no connections")
        
        # Check fanout/fanin
        fanout_counts = {}
        fanin_counts = {}
        
        for synapse in network.synapses:
            fanout_counts[synapse.pre_neuron_id] = fanout_counts.get(synapse.pre_neuron_id, 0) + 1
            fanin_counts[synapse.post_neuron_id] = fanin_counts.get(synapse.post_neuron_id, 0) + 1
        
        # Check for excessive fanout
        high_fanout = [(nid, count) for nid, count in fanout_counts.items() if count > self.max_fanout]
        if high_fanout:
            result.add_warning(f"Neurons with high fanout (>{self.max_fanout}): {len(high_fanout)}")
        
        # Check for excessive fanin
        high_fanin = [(nid, count) for nid, count in fanin_counts.items() if count > self.max_fanin]
        if high_fanin:
            result.add_warning(f"Neurons with high fanin (>{self.max_fanin}): {len(high_fanin)}")
    
    def _validate_neuron_parameters(self, network: Network, result: ValidationResult):
        """Validate neuron model parameters."""
        for neuron in network.neurons:
            if neuron.neuron_type == "LIF":
                params = neuron.parameters
                if "v_thresh" in params and "v_reset" in params:
                    if params["v_thresh"] <= params["v_reset"]:
                        result.add_issue(f"Neuron {neuron.neuron_id}: threshold must be > reset voltage")
                
                if "tau_m" in params and params["tau_m"] <= 0:
                    result.add_issue(f"Neuron {neuron.neuron_id}: membrane time constant must be positive")
    
    def _generate_recommendations(self, network: Network, result: ValidationResult):
        """Generate performance recommendations."""
        neuron_count = len(network.neurons)
        synapse_count = len(network.synapses)
        
        if synapse_count > 0:
            avg_connectivity = synapse_count / (neuron_count ** 2) if neuron_count > 0 else 0
            
            if avg_connectivity > 0.5:
                result.add_recommendation("High connectivity detected - consider sparse connections for better performance")
            elif avg_connectivity < 0.01:
                result.add_recommendation("Very sparse connectivity - network may have limited expressiveness")
        
        # Check for unbalanced layer sizes
        if len(network.layers) > 1:
            layer_sizes = [layer.size for layer in network.layers]
            max_size = max(layer_sizes)
            min_size = min(layer_sizes)
            
            if max_size / min_size > 100:
                result.add_recommendation("Large variation in layer sizes - consider more balanced architecture")


class ConfigurationValidator:
    """Validate compilation configurations."""
    
    def validate_fpga_target(self, target: FPGATarget, 
                           resource_estimate: ResourceEstimate) -> ValidationResult:
        """Validate FPGA target compatibility."""
        result = ValidationResult(valid=True, issues=[], warnings=[], recommendations=[])
        
        target_resources = target.resources
        utilization = resource_estimate.utilization_percentage(target_resources)
        
        # Check resource utilization
        logic_util = utilization.get("logic", 0)
        memory_util = utilization.get("memory", 0)
        dsp_util = utilization.get("dsp", 0)
        
        if logic_util > 100:
            result.add_issue(f"Logic utilization ({logic_util:.1f}%) exceeds target capacity")
        elif logic_util > 90:
            result.add_warning(f"High logic utilization ({logic_util:.1f}%) - synthesis may fail")
        elif logic_util > 80:
            result.add_recommendation(f"Consider optimization to reduce logic usage ({logic_util:.1f}%)")
        
        if memory_util > 100:
            result.add_issue(f"Memory utilization ({memory_util:.1f}%) exceeds target capacity")
        elif memory_util > 85:
            result.add_warning(f"High memory utilization ({memory_util:.1f}%) - consider pruning")
        
        if dsp_util > 100:
            result.add_issue(f"DSP utilization ({dsp_util:.1f}%) exceeds target capacity")
        elif dsp_util > 95:
            result.add_warning(f"High DSP utilization ({dsp_util:.1f}%) - consider simpler neuron models")
        
        return result
    
    def validate_optimization_config(self, level: OptimizationLevel, 
                                   custom_params: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate optimization configuration."""
        result = ValidationResult(valid=True, issues=[], warnings=[], recommendations=[])
        
        if custom_params:
            # Validate custom optimization parameters
            if "weight_threshold" in custom_params:
                threshold = custom_params["weight_threshold"]
                if not 0 < threshold < 1:
                    result.add_issue("Weight threshold must be between 0 and 1")
            
            if "cluster_size" in custom_params:
                cluster_size = custom_params["cluster_size"]
                if not isinstance(cluster_size, int) or cluster_size < 1:
                    result.add_issue("Cluster size must be a positive integer")
                elif cluster_size > 64:
                    result.add_warning("Large cluster sizes may reduce parallelism")
        
        return result


class FileValidator:
    """Validate input files and formats."""
    
    def __init__(self):
        self.supported_extensions = {".yaml", ".yml", ".json"}
    
    def validate_network_file(self, file_path: Path) -> ValidationResult:
        """Validate network definition file."""
        result = ValidationResult(valid=True, issues=[], warnings=[], recommendations=[])
        
        # Check file exists
        if not file_path.exists():
            result.add_issue(f"File does not exist: {file_path}")
            return result
        
        # Check file extension
        if file_path.suffix.lower() not in self.supported_extensions:
            result.add_issue(f"Unsupported file format: {file_path.suffix}")
            return result
        
        # Check file is readable
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except PermissionError:
            result.add_issue(f"Cannot read file: {file_path}")
            return result
        except Exception as e:
            result.add_issue(f"Error reading file: {str(e)}")
            return result
        
        # Validate file format
        try:
            if file_path.suffix.lower() in {".yaml", ".yml"}:
                yaml.safe_load(content)
            elif file_path.suffix.lower() == ".json":
                json.loads(content)
        except Exception as e:
            result.add_issue(f"Invalid file format: {str(e)}")
            return result
        
        # Check file size
        file_size = len(content)
        if file_size > 10 * 1024 * 1024:  # 10MB
            result.add_warning("Large network file - parsing may be slow")
        
        return result


def validate_identifier(name: str) -> bool:
    """Validate that a name is a valid identifier."""
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for filesystem safety."""
    # Remove/replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized or "unnamed"