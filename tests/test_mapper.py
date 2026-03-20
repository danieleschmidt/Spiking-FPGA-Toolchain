import pytest
from spiking_fpga.graph import SpikingGraph
from spiking_fpga.mapper import NeuronMapper, MappingResult, CORE_NEURON_CAPACITY

SPEC = {
    "name": "mapper_test",
    "layers": [
        {"name": "in", "neuron_count": 16, "type": "LIF"},
        {"name": "hidden", "neuron_count": 32, "type": "LIF"},
        {"name": "out", "neuron_count": 8, "type": "IF"},
    ],
    "connections": [],
}


def test_basic_mapping():
    g = SpikingGraph.from_dict(SPEC)
    mapper = NeuronMapper(core_capacity=32)
    result = mapper.map(g)
    assert result.total_cores > 0
    assert len(result.core_allocations) >= 3


def test_all_neurons_covered():
    g = SpikingGraph.from_dict(SPEC)
    mapper = NeuronMapper(core_capacity=32)
    result = mapper.map(g)
    total = sum(a.size for a in result.core_allocations)
    assert total == g.total_neurons()


def test_large_layer_splits_across_cores():
    spec = {
        "name": "big",
        "layers": [{"name": "huge", "neuron_count": 2048, "type": "LIF"}],
        "connections": [],
    }
    g = SpikingGraph.from_dict(spec)
    mapper = NeuronMapper(core_capacity=1024)
    result = mapper.map(g)
    assert result.total_cores == 2
    assert result.core_count() == 2


def test_utilization_range():
    g = SpikingGraph.from_dict(SPEC)
    mapper = NeuronMapper(core_capacity=32)
    result = mapper.map(g)
    assert 0.0 < result.utilization <= 1.0


def test_cores_for_layer():
    g = SpikingGraph.from_dict(SPEC)
    mapper = NeuronMapper(core_capacity=32)
    result = mapper.map(g)
    hidden_cores = result.cores_for_layer("hidden")
    assert len(hidden_cores) >= 1
    total_hidden = sum(a.size for a in hidden_cores)
    assert total_hidden == 32


def test_exact_fit():
    spec = {
        "name": "exact",
        "layers": [{"name": "layer", "neuron_count": 32, "type": "LIF"}],
        "connections": [],
    }
    g = SpikingGraph.from_dict(spec)
    mapper = NeuronMapper(core_capacity=32)
    result = mapper.map(g)
    assert result.total_cores == 1
    assert abs(result.utilization - 1.0) < 1e-9


def test_single_neuron():
    spec = {
        "name": "tiny",
        "layers": [{"name": "single", "neuron_count": 1, "type": "LIF"}],
        "connections": [],
    }
    g = SpikingGraph.from_dict(spec)
    mapper = NeuronMapper(core_capacity=1024)
    result = mapper.map(g)
    assert result.total_cores == 1
    assert result.core_allocations[0].size == 1
