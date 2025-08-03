"""
Network configuration fixtures for testing.
"""

import pytest
from src.spiking_fpga.models.network import SNNNetwork, Layer, Connection, NetworkParameters, LayerType, ConnectivityPattern


@pytest.fixture
def simple_lif_network():
    """Simple 3-layer LIF network for basic testing."""
    parameters = NetworkParameters(
        dt=0.1,
        simulation_time=100.0,
        spike_threshold=1.0
    )
    
    layers = [
        Layer(
            id="input",
            layer_type=LayerType.INPUT,
            size=100,
            neuron_model="LIF",
            tau_m=20.0
        ),
        Layer(
            id="hidden",
            layer_type=LayerType.HIDDEN,
            size=200,
            neuron_model="LIF",
            tau_m=20.0
        ),
        Layer(
            id="output",
            layer_type=LayerType.OUTPUT,
            size=10,
            neuron_model="LIF",
            tau_m=20.0
        )
    ]
    
    connections = [
        Connection(
            source_layer="input",
            target_layer="hidden",
            connectivity=ConnectivityPattern.SPARSE_RANDOM,
            sparsity=0.1,
            weight_mean=0.5
        ),
        Connection(
            source_layer="hidden",
            target_layer="output",
            connectivity=ConnectivityPattern.SPARSE_RANDOM,
            sparsity=0.2,
            weight_mean=0.8
        )
    ]
    
    return SNNNetwork(
        name="simple_test_network",
        description="Simple 3-layer network for testing",
        parameters=parameters,
        layers=layers,
        connections=connections,
        input_size=100,
        output_size=10
    )


@pytest.fixture
def complex_network():
    """Complex multi-layer network with plasticity for advanced testing."""
    parameters = NetworkParameters(
        dt=0.1,
        simulation_time=200.0,
        spike_threshold=1.2,
        refractory_period=2.5
    )
    
    layers = [
        Layer(
            id="input",
            layer_type=LayerType.INPUT,
            size=784,  # MNIST-like input
            neuron_model="LIF",
            tau_m=20.0
        ),
        Layer(
            id="hidden1",
            layer_type=LayerType.HIDDEN,
            size=400,
            neuron_model="LIF",
            tau_m=20.0,
            plasticity_enabled=True,
            stdp_a_plus=0.1,
            stdp_a_minus=0.12
        ),
        Layer(
            id="hidden2",
            layer_type=LayerType.HIDDEN,
            size=200,
            neuron_model="AdaptiveLIF",
            tau_m=25.0,
            plasticity_enabled=True
        ),
        Layer(
            id="output",
            layer_type=LayerType.OUTPUT,
            size=10,
            neuron_model="LIF",
            tau_m=15.0
        )
    ]
    
    connections = [
        Connection(
            source_layer="input",
            target_layer="hidden1",
            connectivity=ConnectivityPattern.SPARSE_RANDOM,
            sparsity=0.05,
            weight_mean=0.3,
            weight_std=0.1
        ),
        Connection(
            source_layer="hidden1",
            target_layer="hidden2",
            connectivity=ConnectivityPattern.SPARSE_RANDOM,
            sparsity=0.15,
            weight_mean=0.6,
            weight_std=0.2
        ),
        Connection(
            source_layer="hidden2",
            target_layer="output",
            connectivity=ConnectivityPattern.FULL,
            weight_mean=0.8,
            weight_std=0.15
        ),
        # Recurrent connection
        Connection(
            source_layer="hidden1",
            target_layer="hidden1",
            connectivity=ConnectivityPattern.SPARSE_RANDOM,
            sparsity=0.02,
            weight_mean=0.2,
            weight_std=0.05
        )
    ]
    
    return SNNNetwork(
        name="complex_test_network",
        description="Complex multi-layer network with plasticity",
        parameters=parameters,
        layers=layers,
        connections=connections,
        input_size=784,
        output_size=10
    )


@pytest.fixture
def invalid_network():
    """Network with validation errors for testing error handling."""
    # Network with mismatched input/output sizes
    layers = [
        Layer(
            id="input",
            layer_type=LayerType.INPUT,
            size=50,  # Doesn't match input_size
            neuron_model="LIF"
        ),
        Layer(
            id="output", 
            layer_type=LayerType.OUTPUT,
            size=5,   # Doesn't match output_size
            neuron_model="LIF"
        )
    ]
    
    connections = [
        Connection(
            source_layer="nonexistent",  # Invalid layer reference
            target_layer="output",
            connectivity=ConnectivityPattern.FULL
        )
    ]
    
    return SNNNetwork(
        name="invalid_network",
        layers=layers,
        connections=connections,
        input_size=100,  # Mismatch with actual input layer size
        output_size=10   # Mismatch with actual output layer size
    )


@pytest.fixture
def sample_networks():
    """Collection of sample networks for testing."""
    return {
        'mnist_classifier': {
            'name': 'mnist_classifier',
            'description': 'MNIST digit classifier',
            'input_size': 784,
            'output_size': 10,
            'parameters': {
                'dt': 0.1,
                'simulation_time': 100.0,
                'spike_threshold': 1.0
            },
            'layers': [
                {
                    'id': 'input',
                    'type': 'input',
                    'size': 784,
                    'neuron_model': 'LIF',
                    'tau_m': 20.0
                },
                {
                    'id': 'hidden',
                    'type': 'hidden', 
                    'size': 300,
                    'neuron_model': 'LIF',
                    'tau_m': 20.0
                },
                {
                    'id': 'output',
                    'type': 'output',
                    'size': 10,
                    'neuron_model': 'LIF',
                    'tau_m': 20.0
                }
            ],
            'connections': [
                {
                    'source': 'input',
                    'target': 'hidden',
                    'connectivity': 'sparse_random',
                    'sparsity': 0.1,
                    'weight_mean': 0.5
                },
                {
                    'source': 'hidden',
                    'target': 'output',
                    'connectivity': 'full',
                    'weight_mean': 0.8
                }
            ]
        },
        'dvs_gesture': {
            'name': 'dvs_gesture_recognition',
            'description': 'DVS gesture recognition network',
            'input_size': 16384,  # 128x128
            'output_size': 11,
            'parameters': {
                'dt': 0.1,
                'simulation_time': 150.0
            },
            'layers': [
                {
                    'id': 'input',
                    'type': 'input',
                    'size': 16384,
                    'neuron_model': 'LIF'
                },
                {
                    'id': 'conv1',
                    'type': 'hidden',
                    'size': 4096,
                    'neuron_model': 'LIF'
                },
                {
                    'id': 'conv2', 
                    'type': 'hidden',
                    'size': 1024,
                    'neuron_model': 'LIF'
                },
                {
                    'id': 'fc',
                    'type': 'hidden',
                    'size': 256,
                    'neuron_model': 'LIF'
                },
                {
                    'id': 'output',
                    'type': 'output',
                    'size': 11,
                    'neuron_model': 'LIF'
                }
            ],
            'connections': [
                {
                    'source': 'input',
                    'target': 'conv1',
                    'connectivity': 'sparse_random',
                    'sparsity': 0.05
                },
                {
                    'source': 'conv1',
                    'target': 'conv2',
                    'connectivity': 'sparse_random',
                    'sparsity': 0.1
                },
                {
                    'source': 'conv2',
                    'target': 'fc',
                    'connectivity': 'sparse_random',
                    'sparsity': 0.2
                },
                {
                    'source': 'fc',
                    'target': 'output',
                    'connectivity': 'full'
                }
            ]
        }
    }