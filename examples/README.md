# Example Networks

This directory contains example spiking neural network configurations that demonstrate the capabilities of the Spiking-FPGA-Toolchain.

## Getting Started

These examples will be functional once the core implementation is complete (Phase 1, Q2 2025). For now, they serve as:
- Design references for the configuration format
- Test cases for the parser implementation
- Templates for new users

## Available Examples

### `simple_mnist.yaml`
A basic MNIST digit classifier with:
- 784 input neurons (28×28 pixels)
- 300 hidden LIF neurons
- 10 output neurons
- Sparse random connectivity

**Target Platform**: Artix-7 35T  
**Estimated Resources**: ~15% LUT utilization  
**Expected Performance**: ~5M spikes/second

### Future Examples (Coming Soon)

- `dvs_gesture_recognition.yaml` - DVS camera gesture recognition
- `cartpole_control.yaml` - Reinforcement learning control task
- `loihi_benchmark.yaml` - Loihi-compatible benchmark network
- `adaptive_learning.yaml` - Network with STDP learning

## Usage

Once the toolchain is implemented, you'll be able to compile these examples:

```bash
# Compile for Artix-7 35T
spiking-fpga compile simple_mnist.yaml --target artix7_35t

# Generate synthesis reports
spiking-fpga compile simple_mnist.yaml --target artix7_35t --reports

# Target different FPGA
spiking-fpga compile simple_mnist.yaml --target cyclone5_gx
```

## Configuration Format

The YAML configuration format includes:

- **Network Structure**: Layers, neurons, and connectivity patterns
- **Neuron Models**: LIF, Izhikevich, and custom models
- **Compilation Settings**: Target platform, optimization level
- **Simulation Parameters**: Duration, timestep, encoding methods

## Contributing Examples

To contribute new example networks:

1. Create a YAML configuration following the format in `simple_mnist.yaml`
2. Add documentation explaining the network purpose and expected performance
3. Include expected resource utilization for different FPGA targets
4. Provide references to relevant research papers if applicable

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.