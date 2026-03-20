import pytest
from spiking_fpga.graph import SpikingGraph
from spiking_fpga.mapper import NeuronMapper
from spiking_fpga.encoder import SynapseEncoder, SpikeOp

SPEC = {
    "name": "enc_test",
    "layers": [
        {"name": "a", "neuron_count": 4, "type": "LIF"},
        {"name": "b", "neuron_count": 4, "type": "LIF"},
    ],
    "connections": [
        {"source": "a", "target": "b", "weights": [0.5, -0.3, 0.0, 0.8,
                                                     0.1, 0.9, -0.6, 0.2,
                                                     0.4, -0.1, 0.7, -0.8,
                                                     0.3, 0.5, -0.4, 0.6]},
    ],
}


def _build_encoded(spec=SPEC, capacity=64):
    g = SpikingGraph.from_dict(spec)
    mapper = NeuronMapper(core_capacity=capacity)
    mapping = mapper.map(g)
    encoder = SynapseEncoder(weight_precision=8)
    return encoder.encode(g, mapping), g


def test_ops_generated():
    enc, g = _build_encoded()
    assert enc.total_ops > 0


def test_excitatory_inhibitory_sum():
    enc, g = _build_encoded()
    assert enc.excitatory_ops + enc.inhibitory_ops == enc.total_ops


def test_zero_weights_skipped():
    # weight of 0.0 should not produce an op
    spec = {
        "name": "zero",
        "layers": [
            {"name": "x", "neuron_count": 1, "type": "LIF"},
            {"name": "y", "neuron_count": 1, "type": "LIF"},
        ],
        "connections": [{"source": "x", "target": "y", "weights": [0.0]}],
    }
    enc, g = _build_encoded(spec)
    assert enc.total_ops == 0


def test_positive_weight_is_route():
    spec = {
        "name": "pos",
        "layers": [
            {"name": "x", "neuron_count": 1, "type": "LIF"},
            {"name": "y", "neuron_count": 1, "type": "LIF"},
        ],
        "connections": [{"source": "x", "target": "y", "weights": [0.5]}],
    }
    enc, g = _build_encoded(spec)
    assert enc.excitatory_ops == 1
    assert enc.ops[0].op_type == "ROUTE"


def test_negative_weight_is_inhibit():
    spec = {
        "name": "neg",
        "layers": [
            {"name": "x", "neuron_count": 1, "type": "LIF"},
            {"name": "y", "neuron_count": 1, "type": "LIF"},
        ],
        "connections": [{"source": "x", "target": "y", "weights": [-0.5]}],
    }
    enc, g = _build_encoded(spec)
    assert enc.inhibitory_ops == 1
    assert enc.ops[0].op_type == "INHIBIT"


def test_to_asm():
    op = SpikeOp(op_type="ROUTE", src_core=0, src_neuron=1,
                 dst_core=1, dst_neuron=2, weight=0.5, delay=1, comment="test")
    asm = op.to_asm()
    assert "ROUTE" in asm
    assert "src=0:1" in asm
    assert "dst=1:2" in asm


def test_weight_quantization():
    encoder = SynapseEncoder(weight_precision=8)
    q = encoder._quantize_weight(0.5)
    # should round to nearest 1/255 step
    assert abs(q - round(0.5 * 255) / 255) < 1e-9


def test_weight_clamped():
    encoder = SynapseEncoder(weight_precision=8)
    assert encoder._quantize_weight(2.0) <= 1.0
    assert encoder._quantize_weight(-2.0) >= -1.0


def test_ops_for_connection():
    enc, g = _build_encoded()
    ops = enc.ops_for_connection("a", "b")
    assert len(ops) == enc.total_ops  # all ops are a->b
