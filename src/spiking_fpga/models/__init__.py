"""
Data models for Spiking Neural Network representation and FPGA resource modeling.
"""

from .network import SNNNetwork, Layer, Connection, NetworkParameters
from .neuron import LIFNeuron, NeuronModel, SynapticConnection
from .fpga import FPGATarget, ResourceUtilization, TimingConstraints
from .spike import SpikeEvent, AERPacket, SpikeBuffer

__all__ = [
    'SNNNetwork', 'Layer', 'Connection', 'NetworkParameters',
    'LIFNeuron', 'NeuronModel', 'SynapticConnection', 
    'FPGATarget', 'ResourceUtilization', 'TimingConstraints',
    'SpikeEvent', 'AERPacket', 'SpikeBuffer'
]