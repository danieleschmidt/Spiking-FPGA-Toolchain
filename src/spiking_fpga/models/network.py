"""Network models and data structures for spiking neural networks."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, validator


class ConnectionType(Enum):
    """Types of synaptic connections."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    ELECTRICAL = "electrical"


class LayerType(Enum):
    """Types of neural layers."""
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"
    RECURRENT = "recurrent"


@dataclass
class Synapse:
    """Represents a synaptic connection between neurons."""
    
    pre_neuron_id: int
    post_neuron_id: int
    weight: float
    delay: float = 1.0  # in timesteps
    connection_type: ConnectionType = ConnectionType.EXCITATORY
    plasticity_enabled: bool = False
    
    def __post_init__(self):
        """Validate synapse parameters."""
        if self.weight == 0.0:
            raise ValueError("Synapse weight cannot be zero")
        if self.delay <= 0:
            raise ValueError("Synapse delay must be positive")


@dataclass
class Neuron:
    """Represents a single neuron with its parameters."""
    
    neuron_id: int
    layer_id: int
    neuron_type: str = "LIF"
    parameters: Dict[str, float] = field(default_factory=dict)
    position: Optional[tuple] = None
    
    def __post_init__(self):
        """Set default parameters based on neuron type."""
        if not self.parameters:
            if self.neuron_type == "LIF":
                self.parameters = {
                    "v_thresh": 1.0,
                    "v_reset": 0.0,
                    "tau_m": 20.0,
                    "v_rest": 0.0,
                }
            elif self.neuron_type == "Izhikevich":
                self.parameters = {
                    "a": 0.02,
                    "b": 0.2,
                    "c": -65.0,
                    "d": 8.0,
                }


@dataclass
class Layer:
    """Represents a layer of neurons with common properties."""
    
    layer_id: int
    layer_type: LayerType
    size: int
    neuron_type: str = "LIF"
    parameters: Dict[str, float] = field(default_factory=dict)
    
    def create_neurons(self) -> List[Neuron]:
        """Create neurons for this layer."""
        neurons = []
        for i in range(self.size):
            neuron = Neuron(
                neuron_id=i,
                layer_id=self.layer_id,
                neuron_type=self.neuron_type,
                parameters=self.parameters.copy()
            )
            neurons.append(neuron)
        return neurons


class Network(BaseModel):
    """Complete spiking neural network definition."""
    
    name: str = Field(..., description="Network name")
    description: Optional[str] = Field(None, description="Network description")
    layers: List[Layer] = Field(default_factory=list, description="Network layers")
    neurons: List[Neuron] = Field(default_factory=list, description="All neurons")
    synapses: List[Synapse] = Field(default_factory=list, description="All synapses")
    timestep: float = Field(1.0, description="Simulation timestep in ms")
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('timestep')
    def validate_timestep(cls, v):
        if v <= 0:
            raise ValueError('Timestep must be positive')
        return v
    
    def add_layer(self, layer_type: LayerType, size: int, 
                  neuron_type: str = "LIF", **parameters) -> int:
        """Add a new layer to the network."""
        layer_id = len(self.layers)
        layer = Layer(
            layer_id=layer_id,
            layer_type=layer_type,
            size=size,
            neuron_type=neuron_type,
            parameters=parameters
        )
        self.layers.append(layer)
        
        # Create and add neurons
        layer_neurons = layer.create_neurons()
        neuron_offset = len(self.neurons)
        for neuron in layer_neurons:
            neuron.neuron_id = neuron_offset + neuron.neuron_id
        self.neurons.extend(layer_neurons)
        
        return layer_id
    
    def connect_layers(self, pre_layer_id: int, post_layer_id: int,
                      connectivity_pattern: str = "all_to_all",
                      weight_distribution: str = "uniform",
                      weight_params: Dict[str, float] = None,
                      sparsity: float = 1.0) -> None:
        """Connect two layers with specified pattern."""
        if weight_params is None:
            weight_params = {"min": 0.1, "max": 1.0}
        
        pre_neurons = [n for n in self.neurons if n.layer_id == pre_layer_id]
        post_neurons = [n for n in self.neurons if n.layer_id == post_layer_id]
        
        if connectivity_pattern == "all_to_all":
            for pre_neuron in pre_neurons:
                for post_neuron in post_neurons:
                    if np.random.random() < sparsity:
                        weight = self._sample_weight(weight_distribution, weight_params)
                        synapse = Synapse(
                            pre_neuron_id=pre_neuron.neuron_id,
                            post_neuron_id=post_neuron.neuron_id,
                            weight=weight
                        )
                        self.synapses.append(synapse)
        
        elif connectivity_pattern == "one_to_one":
            if len(pre_neurons) != len(post_neurons):
                raise ValueError("One-to-one connectivity requires equal layer sizes")
            
            for pre_neuron, post_neuron in zip(pre_neurons, post_neurons):
                weight = self._sample_weight(weight_distribution, weight_params)
                synapse = Synapse(
                    pre_neuron_id=pre_neuron.neuron_id,
                    post_neuron_id=post_neuron.neuron_id,
                    weight=weight
                )
                self.synapses.append(synapse)
    
    def _sample_weight(self, distribution: str, params: Dict[str, float]) -> float:
        """Sample synaptic weight from specified distribution."""
        if distribution == "uniform":
            return np.random.uniform(params["min"], params["max"])
        elif distribution == "normal":
            return np.random.normal(params["mean"], params["std"])
        elif distribution == "constant":
            return params["value"]
        else:
            raise ValueError(f"Unknown weight distribution: {distribution}")
    
    def get_total_neurons(self) -> int:
        """Get total number of neurons in the network."""
        return len(self.neurons)
    
    def get_total_synapses(self) -> int:
        """Get total number of synapses in the network."""
        return len(self.synapses)
    
    def get_connectivity_matrix(self) -> np.ndarray:
        """Get sparse connectivity matrix representation."""
        n_neurons = self.get_total_neurons()
        connectivity = np.zeros((n_neurons, n_neurons))
        
        for synapse in self.synapses:
            connectivity[synapse.pre_neuron_id, synapse.post_neuron_id] = synapse.weight
        
        return connectivity
    
    def validate_network(self) -> List[str]:
        """Validate network structure and return list of issues."""
        issues = []
        
        # Check for orphaned neurons
        connected_neurons = set()
        for synapse in self.synapses:
            connected_neurons.add(synapse.pre_neuron_id)
            connected_neurons.add(synapse.post_neuron_id)
        
        all_neuron_ids = {n.neuron_id for n in self.neurons}
        orphaned = all_neuron_ids - connected_neurons
        if orphaned:
            issues.append(f"Orphaned neurons (no connections): {orphaned}")
        
        # Check for invalid synapse connections
        for synapse in self.synapses:
            if synapse.pre_neuron_id not in all_neuron_ids:
                issues.append(f"Synapse references invalid pre-neuron ID: {synapse.pre_neuron_id}")
            if synapse.post_neuron_id not in all_neuron_ids:
                issues.append(f"Synapse references invalid post-neuron ID: {synapse.post_neuron_id}")
        
        # Check for self-connections
        self_connections = [s for s in self.synapses if s.pre_neuron_id == s.post_neuron_id]
        if self_connections:
            issues.append(f"Found {len(self_connections)} self-connections")
        
        return issues