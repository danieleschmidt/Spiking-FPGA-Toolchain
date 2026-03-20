"""
NeuronMapper: Assigns neurons to logical cores (Loihi-style).
Each core has a fixed neuron capacity. Layers are partitioned across cores.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from spiking_fpga.graph import SpikingGraph, NeuronLayer

CORE_NEURON_CAPACITY = 1024  # Loihi-2 style: 1024 neurons/core


@dataclass
class CoreAllocation:
    core_id: int
    layer_name: str
    neuron_start: int  # first neuron index (within layer)
    neuron_end: int    # last neuron index (exclusive)

    @property
    def size(self) -> int:
        return self.neuron_end - self.neuron_start


@dataclass
class MappingResult:
    graph_name: str
    core_allocations: list[CoreAllocation] = field(default_factory=list)
    total_cores: int = 0
    utilization: float = 0.0   # neurons_used / (cores * capacity)

    def cores_for_layer(self, layer_name: str) -> list[CoreAllocation]:
        return [a for a in self.core_allocations if a.layer_name == layer_name]

    def core_count(self) -> int:
        return len({a.core_id for a in self.core_allocations})


class NeuronMapper:
    """
    Maps neurons to Loihi-style logical cores.
    Strategy: sequential partition — fill one core before starting the next.
    """
    def __init__(self, core_capacity: int = CORE_NEURON_CAPACITY):
        self.core_capacity = core_capacity

    def map(self, graph: SpikingGraph) -> MappingResult:
        result = MappingResult(graph_name=graph.name)
        current_core = 0
        current_offset = 0
        total_neurons_assigned = 0

        for layer in graph.layers:
            remaining = layer.neuron_count
            neuron_idx = 0
            while remaining > 0:
                space_in_core = self.core_capacity - current_offset
                chunk = min(remaining, space_in_core)
                alloc = CoreAllocation(
                    core_id=current_core,
                    layer_name=layer.name,
                    neuron_start=neuron_idx,
                    neuron_end=neuron_idx + chunk
                )
                result.core_allocations.append(alloc)
                neuron_idx += chunk
                total_neurons_assigned += chunk
                remaining -= chunk
                current_offset += chunk
                if current_offset >= self.core_capacity:
                    current_core += 1
                    current_offset = 0

        result.total_cores = current_core + (1 if current_offset > 0 else 0)
        total_capacity = result.total_cores * self.core_capacity
        result.utilization = total_neurons_assigned / total_capacity if total_capacity > 0 else 0.0
        return result
