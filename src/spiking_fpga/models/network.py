"""
Spiking Neural Network data models and structures.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from enum import Enum
import numpy as np
from pydantic import BaseModel, validator, Field


class LayerType(str, Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"
    RESERVOIR = "reservoir"


class ConnectivityPattern(str, Enum):
    FULL = "full"
    SPARSE_RANDOM = "sparse_random"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    SMALL_WORLD = "small_world"
    CUSTOM = "custom"


@dataclass
class NetworkParameters:
    """Global network parameters for SNN simulation."""
    dt: float = 0.1  # Time step in milliseconds
    simulation_time: float = 100.0  # Total simulation time in ms
    spike_threshold: float = 1.0  # Global spike threshold
    resting_potential: float = 0.0  # Resting membrane potential
    reset_potential: float = 0.0  # Post-spike reset potential
    refractory_period: float = 2.0  # Refractory period in ms
    
    def validate_parameters(self) -> bool:
        """Validate parameter ranges for FPGA implementation."""
        if not (0.01 <= self.dt <= 10.0):
            raise ValueError(f"dt must be between 0.01-10.0ms, got {self.dt}")
        if not (0.1 <= self.spike_threshold <= 10.0):
            raise ValueError(f"spike_threshold must be between 0.1-10.0, got {self.spike_threshold}")
        return True


class Layer(BaseModel):
    """Represents a layer of neurons in the SNN."""
    id: str
    layer_type: LayerType
    size: int = Field(..., gt=0, description="Number of neurons in layer")
    neuron_model: str = "LIF"
    
    # LIF-specific parameters
    tau_m: float = Field(20.0, gt=0, description="Membrane time constant (ms)")
    tau_syn: float = Field(5.0, gt=0, description="Synaptic time constant (ms)")
    capacitance: float = Field(1.0, gt=0, description="Membrane capacitance (nF)")
    
    # Plasticity parameters
    plasticity_enabled: bool = False
    stdp_a_plus: float = 0.1
    stdp_a_minus: float = 0.12
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    
    @validator('size')
    def validate_size_for_fpga(cls, v):
        """Ensure layer size is reasonable for FPGA implementation."""
        if v > 100000:
            raise ValueError(f"Layer size {v} exceeds FPGA capacity (max 100K neurons)")
        return v
    
    def estimate_resources(self) -> Dict[str, int]:
        """Estimate FPGA resources needed for this layer."""
        # Rough estimates for Artix-7 resources per neuron
        luts_per_neuron = 8 if self.plasticity_enabled else 4
        bram_bits_per_neuron = 64 if self.plasticity_enabled else 32
        
        return {
            'luts': self.size * luts_per_neuron,
            'bram_bits': self.size * bram_bits_per_neuron,
            'dsp_slices': self.size // 100 if self.plasticity_enabled else 0
        }


class Connection(BaseModel):
    """Represents connections between layers."""
    source_layer: str
    target_layer: str
    connectivity: ConnectivityPattern
    weight_distribution: str = "uniform"
    
    # Connection parameters
    sparsity: float = Field(0.1, ge=0.0, le=1.0, description="Connection sparsity (0-1)")
    weight_mean: float = 0.5
    weight_std: float = 0.1
    delay_mean: float = 1.0  # Synaptic delay in ms
    delay_std: float = 0.2
    
    # Custom connectivity matrix (for CUSTOM pattern)
    weight_matrix: Optional[np.ndarray] = None
    
    @validator('sparsity')
    def validate_sparsity(cls, v):
        """Ensure sparsity is suitable for FPGA memory constraints."""
        if v > 0.5:
            raise ValueError(f"Sparsity {v} too high for efficient FPGA implementation")
        return v
    
    def generate_weight_matrix(self, source_size: int, target_size: int) -> np.ndarray:
        """Generate weight matrix based on connectivity pattern."""
        if self.connectivity == ConnectivityPattern.CUSTOM:
            if self.weight_matrix is None:
                raise ValueError("Custom connectivity requires weight_matrix")
            return self.weight_matrix
            
        weights = np.zeros((source_size, target_size))
        
        if self.connectivity == ConnectivityPattern.FULL:
            # All-to-all connectivity
            weights = np.random.normal(self.weight_mean, self.weight_std, 
                                     (source_size, target_size))
            
        elif self.connectivity == ConnectivityPattern.SPARSE_RANDOM:
            # Sparse random connectivity
            n_connections = int(self.sparsity * source_size * target_size)
            for _ in range(n_connections):
                i = np.random.randint(0, source_size)
                j = np.random.randint(0, target_size)
                weights[i, j] = np.random.normal(self.weight_mean, self.weight_std)
                
        elif self.connectivity == ConnectivityPattern.NEAREST_NEIGHBOR:
            # Local connectivity pattern
            radius = max(1, int(np.sqrt(self.sparsity) * min(source_size, target_size) / 2))
            for i in range(source_size):
                for j in range(max(0, i-radius), min(target_size, i+radius+1)):
                    if np.random.random() < self.sparsity:
                        weights[i, j] = np.random.normal(self.weight_mean, self.weight_std)
        
        return weights
    
    def estimate_memory_usage(self, source_size: int, target_size: int) -> int:
        """Estimate BRAM usage for synaptic weights (in bits)."""
        if self.connectivity == ConnectivityPattern.FULL:
            n_synapses = source_size * target_size
        else:
            n_synapses = int(self.sparsity * source_size * target_size)
        
        # 16-bit fixed-point weights + addressing overhead
        bits_per_synapse = 16 + np.ceil(np.log2(source_size + target_size))
        return int(n_synapses * bits_per_synapse)


class SNNNetwork(BaseModel):
    """Complete Spiking Neural Network specification."""
    name: str
    description: Optional[str] = ""
    parameters: NetworkParameters = NetworkParameters()
    layers: List[Layer] = []
    connections: List[Connection] = []
    
    # Input/output specifications
    input_size: int = Field(..., gt=0)
    output_size: int = Field(..., gt=0)
    
    def validate_network(self) -> bool:
        """Comprehensive network validation."""
        self.parameters.validate_parameters()
        
        # Check layer references in connections
        layer_ids = {layer.id for layer in self.layers}
        for conn in self.connections:
            if conn.source_layer not in layer_ids:
                raise ValueError(f"Source layer '{conn.source_layer}' not found")
            if conn.target_layer not in layer_ids:
                raise ValueError(f"Target layer '{conn.target_layer}' not found")
        
        # Validate input/output layer sizes
        input_layers = [l for l in self.layers if l.layer_type == LayerType.INPUT]
        output_layers = [l for l in self.layers if l.layer_type == LayerType.OUTPUT]
        
        if not input_layers:
            raise ValueError("Network must have at least one input layer")
        if not output_layers:
            raise ValueError("Network must have at least one output layer")
            
        total_input = sum(l.size for l in input_layers)
        total_output = sum(l.size for l in output_layers)
        
        if total_input != self.input_size:
            raise ValueError(f"Input layer sizes ({total_input}) don't match input_size ({self.input_size})")
        if total_output != self.output_size:
            raise ValueError(f"Output layer sizes ({total_output}) don't match output_size ({self.output_size})")
        
        return True
    
    def get_layer_by_id(self, layer_id: str) -> Optional[Layer]:
        """Find layer by ID."""
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None
    
    def estimate_total_resources(self) -> Dict[str, int]:
        """Estimate total FPGA resources needed."""
        total_resources = {'luts': 0, 'bram_bits': 0, 'dsp_slices': 0}
        
        # Sum layer resources
        for layer in self.layers:
            layer_resources = layer.estimate_resources()
            for resource, amount in layer_resources.items():
                total_resources[resource] += amount
        
        # Add connection memory requirements
        for conn in self.connections:
            source_layer = self.get_layer_by_id(conn.source_layer)
            target_layer = self.get_layer_by_id(conn.target_layer)
            if source_layer and target_layer:
                total_resources['bram_bits'] += conn.estimate_memory_usage(
                    source_layer.size, target_layer.size)
        
        return total_resources
    
    def get_execution_order(self) -> List[str]:
        """Determine layer execution order for pipeline scheduling."""
        # Simple topological sort based on connections
        in_degree = {layer.id: 0 for layer in self.layers}
        
        for conn in self.connections:
            in_degree[conn.target_layer] += 1
        
        queue = [layer_id for layer_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Update in-degrees for connected layers
            for conn in self.connections:
                if conn.source_layer == current:
                    in_degree[conn.target_layer] -= 1
                    if in_degree[conn.target_layer] == 0:
                        queue.append(conn.target_layer)
        
        if len(execution_order) != len(self.layers):
            raise ValueError("Network contains cycles - cannot determine execution order")
        
        return execution_order