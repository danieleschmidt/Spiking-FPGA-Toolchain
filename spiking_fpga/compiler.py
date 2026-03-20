"""
End-to-end SNN compiler: SpikingGraph → HDL binary (Verilog pseudocode).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from spiking_fpga.graph import SpikingGraph
from spiking_fpga.mapper import NeuronMapper, MappingResult
from spiking_fpga.encoder import SynapseEncoder, EncodedNetwork
from spiking_fpga.hdl import HDLGenerator


@dataclass
class CompilationResult:
    graph_name: str
    mapping: MappingResult
    encoded: EncodedNetwork
    hdl: str
    total_cores: int
    total_neurons: int
    total_ops: int
    utilization: float

    def save_hdl(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.hdl)
        return p


class SNNCompiler:
    """
    End-to-end compiler from SpikingGraph to Verilog-like HDL.
    Pipeline: parse → map → encode → generate HDL
    """
    def __init__(self, core_capacity: int = 1024, weight_bits: int = 8):
        self.mapper = NeuronMapper(core_capacity)
        self.encoder = SynapseEncoder(weight_bits)
        self.hdl_gen = HDLGenerator(weight_bits=weight_bits)

    def compile(self, graph: SpikingGraph) -> CompilationResult:
        mapping = self.mapper.map(graph)
        encoded = self.encoder.encode(graph, mapping)
        hdl = self.hdl_gen.generate(graph, mapping, encoded)
        return CompilationResult(
            graph_name=graph.name,
            mapping=mapping,
            encoded=encoded,
            hdl=hdl,
            total_cores=mapping.total_cores,
            total_neurons=graph.total_neurons(),
            total_ops=encoded.total_ops,
            utilization=mapping.utilization
        )

    def compile_from_dict(self, spec: dict) -> CompilationResult:
        graph = SpikingGraph.from_dict(spec)
        return self.compile(graph)


# Demo 3-layer SNN spec
DEMO_3LAYER_SPEC = {
    "name": "spikeformer_3layer",
    "timesteps": 16,
    "layers": [
        {"name": "input", "neuron_count": 16, "type": "LIF", "threshold": 1.0, "decay": 0.9},
        {"name": "hidden", "neuron_count": 32, "type": "LIF", "threshold": 0.8, "decay": 0.95},
        {"name": "output", "neuron_count": 8, "type": "IF", "threshold": 1.2, "decay": 1.0},
    ],
    "connections": [
        {"source": "input", "target": "hidden", "delay": 1},
        {"source": "hidden", "target": "output", "delay": 2},
    ],
    "input_layer": "input",
    "output_layer": "output"
}


def demo_compile() -> CompilationResult:
    """Run demo compilation of a 3-layer SNN."""
    compiler = SNNCompiler(core_capacity=32)  # small cores for demo
    return compiler.compile_from_dict(DEMO_3LAYER_SPEC)
