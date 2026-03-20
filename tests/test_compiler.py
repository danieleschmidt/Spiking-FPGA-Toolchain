import pytest
from spiking_fpga.compiler import SNNCompiler, demo_compile, DEMO_3LAYER_SPEC


def test_demo_compile_runs():
    result = demo_compile()
    assert result.total_neurons > 0
    assert result.hdl != ""


def test_hdl_contains_module():
    result = demo_compile()
    assert "module" in result.hdl


def test_hdl_contains_timescale():
    result = demo_compile()
    assert "timescale" in result.hdl


def test_compile_from_dict():
    compiler = SNNCompiler(core_capacity=32)
    result = compiler.compile_from_dict(DEMO_3LAYER_SPEC)
    assert result.graph_name == "spikeformer_3layer"


def test_total_neurons_correct():
    result = demo_compile()
    assert result.total_neurons == 56  # 16 + 32 + 8


def test_save_hdl(tmp_path):
    result = demo_compile()
    out = tmp_path / "net.v"
    result.save_hdl(str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_utilization_between_0_and_1():
    result = demo_compile()
    assert 0.0 <= result.utilization <= 1.0


def test_compile_ops_count():
    result = demo_compile()
    assert result.total_ops >= 0
