"""
SpikingGraph: Parse and represent SpikeFormer-style Spiking Neural Network models.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NeuronLayer:
    name: str
    neuron_count: int
    neuron_type: str  # "LIF", "IF", "ALIF"
    threshold: float = 1.0
    decay: float = 0.9
    refractory: int = 2  # refractory period in timesteps


@dataclass
class SynapticConnection:
    source: str
    target: str
    weights: list  # list of floats
    delay: int = 1   # spike delay in timesteps
    connection_type: str = "dense"  # "dense" or "sparse"


@dataclass
class SpikingGraph:
    name: str
    layers: list[NeuronLayer] = field(default_factory=list)
    connections: list[SynapticConnection] = field(default_factory=list)
    timesteps: int = 16
    input_layer: Optional[str] = None
    output_layer: Optional[str] = None

    @classmethod
    def from_dict(cls, spec: dict) -> "SpikingGraph":
        """Parse from a SpikeFormer-style dict spec."""
        g = cls(
            name=spec.get("name", "unnamed_snn"),
            timesteps=spec.get("timesteps", 16)
        )
        for layer_spec in spec.get("layers", []):
            layer = NeuronLayer(
                name=layer_spec["name"],
                neuron_count=layer_spec["neuron_count"],
                neuron_type=layer_spec.get("type", "LIF"),
                threshold=layer_spec.get("threshold", 1.0),
                decay=layer_spec.get("decay", 0.9),
                refractory=layer_spec.get("refractory", 2)
            )
            g.layers.append(layer)

        for conn_spec in spec.get("connections", []):
            n_src = next((l.neuron_count for l in g.layers if l.name == conn_spec["source"]), 1)
            n_tgt = next((l.neuron_count for l in g.layers if l.name == conn_spec["target"]), 1)
            # Initialize weights: use provided or random-like (deterministic)
            if "weights" in conn_spec:
                weights = conn_spec["weights"]
            else:
                # Deterministic pseudo-random weights
                seed_val = hash(conn_spec["source"] + conn_spec["target"]) % 10000
                weights = [((seed_val * (i + 1)) % 1000) / 1000.0 - 0.5
                           for i in range(n_src * n_tgt)]
            conn = SynapticConnection(
                source=conn_spec["source"],
                target=conn_spec["target"],
                weights=weights,
                delay=conn_spec.get("delay", 1),
                connection_type=conn_spec.get("type", "dense")
            )
            g.connections.append(conn)

        if g.layers:
            g.input_layer = spec.get("input_layer", g.layers[0].name)
            g.output_layer = spec.get("output_layer", g.layers[-1].name)

        return g

    @classmethod
    def from_json(cls, json_str: str) -> "SpikingGraph":
        return cls.from_dict(json.loads(json_str))

    def get_layer(self, name: str) -> Optional[NeuronLayer]:
        for l in self.layers:
            if l.name == name:
                return l
        return None

    def get_connections_from(self, source: str) -> list[SynapticConnection]:
        return [c for c in self.connections if c.source == source]

    def total_neurons(self) -> int:
        return sum(l.neuron_count for l in self.layers)

    def total_synapses(self) -> int:
        return sum(len(c.weights) for c in self.connections)

    def layer_names(self) -> list[str]:
        return [l.name for l in self.layers]
