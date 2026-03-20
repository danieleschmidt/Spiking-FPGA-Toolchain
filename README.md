# Spiking-FPGA-Toolchain

A compiler toolchain that maps **SpikeFormer-style Spiking Neural Networks** onto Loihi-inspired FPGA cores, emitting synthesizable Verilog HDL.

## Pipeline

```
SpikingGraph          ← parse layers + synaptic connections from JSON/dict
     │
     ▼
NeuronMapper          ← assign neurons to logical cores (1024 neurons/core, Loihi-style)
     │
     ▼
SynapseEncoder        ← quantize weights → event-driven SpikeOp instructions (ROUTE/INHIBIT)
     │
     ▼
HDLGenerator          ← emit per-core Verilog modules + synapse routing table + top-level
     │
     ▼
output.v              ← synthesizable Verilog
```

## Quick Start

```python
from spiking_fpga.compiler import SNNCompiler

spec = {
    "name": "mnist_snn",
    "timesteps": 16,
    "layers": [
        {"name": "input",  "neuron_count": 784, "type": "LIF", "threshold": 1.0},
        {"name": "hidden", "neuron_count": 512, "type": "LIF", "threshold": 0.8},
        {"name": "output", "neuron_count": 10,  "type": "IF",  "threshold": 1.2},
    ],
    "connections": [
        {"source": "input",  "target": "hidden", "delay": 1},
        {"source": "hidden", "target": "output", "delay": 2},
    ],
}

compiler = SNNCompiler(core_capacity=1024, weight_bits=8)
result = compiler.compile_from_dict(spec)
print(f"Cores: {result.total_cores}, Neurons: {result.total_neurons}, Ops: {result.total_ops}")
result.save_hdl("output/mnist_snn.v")
```

## Demo

```python
from spiking_fpga.compiler import demo_compile
result = demo_compile()
print(result.hdl[:800])
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Test

```bash
pytest tests/ -v
```
