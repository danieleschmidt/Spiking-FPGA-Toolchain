import pytest
from spiking_fpga.graph import SpikingGraph
from spiking_fpga.mapper import NeuronMapper


def _make_graph(counts, name="test"):
    spec = {
        "name": name,
        "layers": [{"name": f"l{i}", "neuron_count": n, "type": "LIF"} for i, n in enumerate(counts)],
        "connections": [],
    }
    return SpikingGraph.from_dict(spec)


def test_single_layer_single_core():
    g = _make_graph([50])
    mapper = NeuronMapper(core_capacity=100)
    result = mapper.map(g)
    assert result.total_cores == 1


def test_layer_exceeds_core_splits():
    g = _make_graph([25])
    mapper = NeuronMapper(core_capacity=10)
    result = mapper.map(g)
    assert result.total_cores == 3  # ceil(25/10)


def test_utilization_full():
    g = _make_graph([10])
    mapper = NeuronMapper(core_capacity=10)
    result = mapper.map(g)
    assert abs(result.utilization - 1.0) < 1e-9


def test_multiple_layers():
    g = _make_graph([4, 8, 2])
    mapper = NeuronMapper(core_capacity=1024)
    result = mapper.map(g)
    assert result.total_cores >= 1
    total = sum(a.size for a in result.core_allocations)
    assert total == 14


def test_cores_for_layer():
    g = _make_graph([4, 8, 2])
    mapper = NeuronMapper(core_capacity=1024)
    result = mapper.map(g)
    allocs = result.cores_for_layer("l1")
    assert len(allocs) >= 1
    assert all(a.layer_name == "l1" for a in allocs)
    assert sum(a.size for a in allocs) == 8
