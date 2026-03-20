"""
SynapseEncoder: Encodes synaptic weights as event-driven spike routing ops.
Produces a list of SpikeOp instructions for the HDL backend.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from spiking_fpga.graph import SpikingGraph, SynapticConnection
from spiking_fpga.mapper import MappingResult


@dataclass
class SpikeOp:
    """A single spike routing operation."""
    op_type: str          # "ROUTE", "WEIGHT", "DELAY", "INHIBIT"
    src_core: int
    src_neuron: int
    dst_core: int
    dst_neuron: int
    weight: float
    delay: int
    comment: str = ""

    def to_asm(self) -> str:
        """Convert to assembly-like representation."""
        return (f"{self.op_type} src={self.src_core}:{self.src_neuron} "
                f"dst={self.dst_core}:{self.dst_neuron} "
                f"w={self.weight:.4f} d={self.delay} ; {self.comment}")


@dataclass
class EncodedNetwork:
    graph_name: str
    ops: list[SpikeOp]
    total_ops: int
    inhibitory_ops: int
    excitatory_ops: int

    def ops_for_connection(self, src_layer: str, dst_layer: str) -> list[SpikeOp]:
        return [op for op in self.ops if src_layer in op.comment and dst_layer in op.comment]


class SynapseEncoder:
    """Encodes synaptic connections as event-driven spike ops."""

    def __init__(self, weight_precision: int = 8):
        self.weight_precision = weight_precision  # bits
        self._quant_levels = 2 ** weight_precision - 1

    def _quantize_weight(self, w: float) -> float:
        """Quantize weight to fixed-point precision."""
        clamped = max(-1.0, min(1.0, w))
        quantized = round(clamped * self._quant_levels) / self._quant_levels
        return quantized

    def encode(self, graph: SpikingGraph, mapping: MappingResult) -> EncodedNetwork:
        ops = []
        # Build neuron → (core_id, local_idx) lookup
        neuron_to_core: dict[str, dict[int, tuple[int, int]]] = {}
        for alloc in mapping.core_allocations:
            if alloc.layer_name not in neuron_to_core:
                neuron_to_core[alloc.layer_name] = {}
            for local_n in range(alloc.size):
                global_n = alloc.neuron_start + local_n
                neuron_to_core[alloc.layer_name][global_n] = (alloc.core_id, local_n)

        for conn in graph.connections:
            src_layer = graph.get_layer(conn.source)
            dst_layer = graph.get_layer(conn.target)
            if not src_layer or not dst_layer:
                continue

            n_src = src_layer.neuron_count
            n_dst = dst_layer.neuron_count
            expected = n_src * n_dst
            weights = conn.weights[:expected] + [0.0] * max(0, expected - len(conn.weights))

            for i in range(n_src):
                for j in range(n_dst):
                    w = self._quantize_weight(weights[i * n_dst + j])
                    if abs(w) < 1e-6:
                        continue
                    src_core, src_local = neuron_to_core.get(conn.source, {}).get(i, (0, i))
                    dst_core, dst_local = neuron_to_core.get(conn.target, {}).get(j, (0, j))
                    op_type = "ROUTE" if w > 0 else "INHIBIT"
                    ops.append(SpikeOp(
                        op_type=op_type,
                        src_core=src_core, src_neuron=src_local,
                        dst_core=dst_core, dst_neuron=dst_local,
                        weight=w, delay=conn.delay,
                        comment=f"{conn.source}->{conn.target}"
                    ))

        excitatory = sum(1 for op in ops if op.op_type == "ROUTE")
        inhibitory = sum(1 for op in ops if op.op_type == "INHIBIT")
        return EncodedNetwork(
            graph_name=graph.name,
            ops=ops,
            total_ops=len(ops),
            inhibitory_ops=inhibitory,
            excitatory_ops=excitatory
        )
