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
        {"source": "a", "target": "b",
         "weights": [0.5, -0.3, 0.1, 0.8,
                     -0.2, 0.9, -0.6, 0.4,
                     0.3, -0.1, 0.7, -0.8,
                     0.6, 0.2, -0.4, 0.5]},
    ],
}


def _encode(spec=SPEC, capacity=64):
    g = SpikingGraph.from_dict(spec)
    mapper = NeuronMapper(core_capacity=capacity)
    mapping = mapper.map(g)
    encoder = SynapseEncoder(weight_precision=8)
    return encoder.encode(g, mapping)


def test_encode_produces_ops():
    enc = _encode()
    assert enc.total_ops > 0


def test_ops_have_correct_types():
    enc = _encode()
    for op in enc.ops:
        assert op.op_type in ("ROUTE", "INHIBIT")


def test_spike_op_to_asm():
    op = SpikeOp(op_type="ROUTE", src_core=0, src_neuron=1,
                 dst_core=1, dst_neuron=2, weight=0.5, delay=1, comment="a->b")
    asm = op.to_asm()
    assert "src=" in asm
    assert "dst=" in asm


def test_weight_quantization():
    encoder = SynapseEncoder(weight_precision=8)
    q = encoder._quantize_weight(0.777)
    assert -1.0 <= q <= 1.0
