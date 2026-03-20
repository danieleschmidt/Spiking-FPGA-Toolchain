import pytest
from spiking_fpga.graph import SpikingGraph

SIMPLE_SPEC = {
    "name": "test_net",
    "timesteps": 8,
    "layers": [
        {"name": "in",     "neuron_count": 4, "type": "LIF"},
        {"name": "hidden", "neuron_count": 8, "type": "LIF"},
        {"name": "out",    "neuron_count": 2, "type": "IF"},
    ],
    "connections": [
        {"source": "in",     "target": "hidden"},
        {"source": "hidden", "target": "out"},
    ],
}


def test_parse_layers():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert len(g.layers) == 3


def test_parse_connections():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert len(g.connections) == 2


def test_total_neurons():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.total_neurons() == 14  # 4 + 8 + 2


def test_total_synapses():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.total_synapses() > 0


def test_layer_names():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.layer_names() == ["in", "hidden", "out"]


def test_get_layer():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    layer = g.get_layer("hidden")
    assert layer is not None
    assert layer.neuron_count == 8


def test_get_layer_missing():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.get_layer("nonexistent") is None


def test_connections_from():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    conns = g.get_connections_from("in")
    assert len(conns) == 1
    assert conns[0].target == "hidden"


def test_neuron_type():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.layers[2].neuron_type == "IF"


def test_timesteps():
    g = SpikingGraph.from_dict(SIMPLE_SPEC)
    assert g.timesteps == 8
