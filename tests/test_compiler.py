import pytest
from spiking_fpga.compiler import SNNCompiler, demo_compile, DEMO_3LAYER_SPEC
from spiking_fpga.graph import SpikingGraph


def test_demo_compile_runs():
    result = demo_compile()
    assert result is not None
    assert result.graph_name == "spikeformer_3layer"


def test_demo_compile_cores():
    result = demo_compile()
    assert result.total_cores > 0


def test_demo_compile_neurons():
    result = demo_compile()
    assert result.total_neurons == 56  # 16 + 32 + 8


def test_demo_compile_ops():
    result = demo_compile()
    assert result.total_ops > 0


def test_hdl_output_contains_verilog():
    result = demo_compile()
    assert "`timescale" in result.hdl
    assert "module snn_top_" in result.hdl
    assert "endmodule" in result.hdl


def test_hdl_contains_core_modules():
    result = demo_compile()
    assert "module snn_core_" in result.hdl


def test_compile_from_dict():
    compiler = SNNCompiler(core_capacity=64)
    result = compiler.compile_from_dict(DEMO_3LAYER_SPEC)
    assert result.graph_name == "spikeformer_3layer"
    assert result.total_neurons == 56


def test_utilization_valid():
    result = demo_compile()
    assert 0.0 < result.utilization <= 1.0


def test_save_hdl(tmp_path):
    result = demo_compile()
    out_path = tmp_path / "output.v"
    saved = result.save_hdl(str(out_path))
    assert saved.exists()
    content = saved.read_text()
    assert "module" in content


def test_single_layer_compile():
    spec = {
        "name": "single",
        "layers": [{"name": "only", "neuron_count": 8, "type": "LIF"}],
        "connections": [],
    }
    compiler = SNNCompiler()
    result = compiler.compile_from_dict(spec)
    assert result.total_neurons == 8
    assert result.total_ops == 0


def test_large_network():
    spec = {
        "name": "large",
        "layers": [
            {"name": "in", "neuron_count": 784, "type": "LIF"},
            {"name": "h1", "neuron_count": 512, "type": "LIF"},
            {"name": "out", "neuron_count": 10, "type": "IF"},
        ],
        "connections": [
            {"source": "in", "target": "h1", "delay": 1},
            {"source": "h1", "target": "out", "delay": 1},
        ],
    }
    compiler = SNNCompiler(core_capacity=1024)
    result = compiler.compile_from_dict(spec)
    assert result.total_neurons == 1306
    assert result.total_ops > 0
