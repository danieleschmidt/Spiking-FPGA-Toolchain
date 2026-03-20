# Spiking-FPGA-Toolchain

A compiler toolchain for **SpikeFormer → Loihi-style FPGA binaries**.

Converts spiking neural network (SNN) model specifications into Verilog HDL targeting Loihi-inspired neuromorphic cores on FPGA.

## Pipeline

```
SpikeFormer spec (JSON/dict)
        │
        ▼
   SpikingGraph         ← parse layers + connections
        │
        ▼
   NeuronMapper         ← assign neurons to logical cores (Loihi-style)
        │
        ▼
   SynapseEncoder       ← encode weights as event-driven SpikeOps
        │
        ▼
   HDLGenerator         ← emit Verilog modules per core + routing table
        │
        ▼
   output.v             ← synthesizable Verilog (+ top-level module)
```

## Quick Start

```python
from spiking_fpga.compiler import SNNCompiler

spec = {
    "name": "my_snn",
    "timesteps": 16,
    "layers": [
        {"name": "input",  "neuron_count": 28*28, "type": "LIF", "threshold": 1.0},
        {"name": "hidden", "neuron_count": 512,   "type": "LIF", "threshold": 0.8},
        {"name": "output", "neuron_count": 10,    "type": "IF",  "threshold": 1.2},
    ],
    "connections": [
        {"source": "input",  "target": "hidden", "delay": 1},
        {"source": "hidden", "target": "output", "delay": 2},
    ],
    "input_layer": "input",
    "output_layer": "output"
}

compiler = SNNCompiler(core_capacity=1024)
result = compiler.compile_from_dict(spec)
print(f"Cores: {result.total_cores}, Neurons: {result.total_neurons}, Ops: {result.total_ops}")
result.save_hdl("output/my_snn.v")
```

## Demo

```python
from spiking_fpga.compiler import demo_compile
result = demo_compile()
print(result.hdl[:500])
```

## Modules

| Module | Description |
|--------|-------------|
| `spiking_fpga.graph` | `SpikingGraph`, `NeuronLayer`, `SynapticConnection` — parse SNN specs |
| `spiking_fpga.mapper` | `NeuronMapper` — assign neurons to Loihi-style cores |
| `spiking_fpga.encoder` | `SynapseEncoder` — produce quantized `SpikeOp` instructions |
| `spiking_fpga.hdl` | `HDLGenerator` — emit Verilog-like HDL per core + top-level |
| `spiking_fpga.compiler` | `SNNCompiler` — end-to-end compile pipeline |

## Neuron Models

- **LIF** (Leaky Integrate-and-Fire): membrane decays each timestep
- **IF** (Integrate-and-Fire): no decay
- **ALIF** (Adaptive LIF): planned

## Core Architecture (Loihi-style)

- Default capacity: **1024 neurons/core**
- Weights quantized to **8-bit fixed-point** (configurable)
- Spike routing via event-driven `ROUTE`/`INHIBIT` ops
- Each core maps to a standalone Verilog `module`

## Install

```bash
pip install -e .
```

## Test

```bash
pytest tests/ -v
```
