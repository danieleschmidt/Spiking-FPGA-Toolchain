import pytest
from spiking_fpga.graph import SpikingGraph, NeuronLayer, SynapticConnection

SIMPLE_SPEC = {
    "name": "test_net",
    "timesteps": 8,
    "layers": [
        {"name": "in", "neuron_count": 4, "type": "LIF", "threshold": 1.0},
        {"name": "out", "neuron_count": 2, "type": "IF", "threshold": 0.8},
    ],
    "connections": [
        {"source": "in", "target": "out", "delay": 1},
    ],
    "input_layer": "in",
    "output_layer": "out",
}


def test_from_dict_basic():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.name == "test_net"
    assert g.timesteps == 8
    assert len(g.layers) == 2
    assert len(g.connections) == 1


def test_layer_properties():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    in_layer = g.get_layer("in")
    assert in_layer is not None
    assert in_layer.neuron_count == 4
    assert in_layer.neuron_type == "LIF"
    assert in_layer.threshold == 1.0


def test_input_output_layers():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.input_layer == "in"
    assert g.output_layer == "out"


def test_total_neurons():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.total_neurons() == 6  # 4 + 2


def test_total_synapses():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    # 4 * 2 = 8 weights for dense in->out
    assert g.total_synapses() == 8


def test_connections_from():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    conns = g.get_connections_from("in")
    assert len(conns) == 1
    assert conns[0].target == "out"


def test_layer_names():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.layer_names() == ["in", "out"]


def test_deterministic_weights():
    g1 = SpikingGraph.from_dict(SIMPLE_SPEC)
    g2 = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g1.connections[0].weights == g2.connections[0].weights


def test_explicit_weights():
    spec = {
        "name": "w_net",
        "layers": [
            {"name": "a", "neuron_count": 2, "type": "LIF"},
            {"name": "b", "neuron_count": 2, "type": "LIF"},
        ],
        "connections": [
            {"source": "a", "target": "b", "weights": [0.5, -0.3, 0.1, 0.8]},
        ],
    }
    g = SpikingGraph.from_dict(spec)
    assert g.connections[0].weights == [0.5, -0.3, 0.1, 0.8]


def test_from_json():
    import json
    g = SpikingGraph.from_json(json.dumps(SIMPLE_SPEC))
    assert g.name == "test_net"


def test_get_layer_missing():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.get_layer("nonexistent") is None


def test_default_input_output():
    spec = {
        "name": "auto",
        "layers": [
            {"name": "first", "neuron_count": 3, "type": "LIF"},
            {"name": "last", "neuron_count": 2, "type": "LIF"},
        ],
        "connections": [],
    }
    g = SpikingGraph.from_dict(spec)
    assert g.input_layer == "first"
    assert g.output_layer == "last"
