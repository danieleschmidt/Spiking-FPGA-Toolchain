"""
JSON network configuration parser.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from .yaml_parser import YAMLNetworkParser
from ..models.network import SNNNetwork


class JSONNetworkParser(YAMLNetworkParser):
    """Parser for JSON network configuration files (inherits YAML parser logic)."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def parse_file(self, json_file: str) -> SNNNetwork:
        """
        Parse SNN network from JSON file.
        
        Args:
            json_file: Path to JSON configuration file
            
        Returns:
            SNNNetwork object
        """
        json_path = Path(json_file)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        return self.parse_config(config)
    
    def write_config(self, network: SNNNetwork, output_file: str):
        """
        Write SNNNetwork to JSON configuration file.
        
        Args:
            network: SNNNetwork to serialize
            output_file: Output JSON file path
        """
        config = self._network_to_config_dict(network)
        
        # Write to file with pretty formatting
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Network configuration written to {output_file}")
    
    def _network_to_config_dict(self, network: SNNNetwork) -> Dict[str, Any]:
        """Convert SNNNetwork to configuration dictionary."""
        config = {
            'name': network.name,
            'description': network.description,
            'input_size': network.input_size,
            'output_size': network.output_size,
            'parameters': {
                'dt': network.parameters.dt,
                'simulation_time': network.parameters.simulation_time,
                'spike_threshold': network.parameters.spike_threshold,
                'resting_potential': network.parameters.resting_potential,
                'reset_potential': network.parameters.reset_potential,
                'refractory_period': network.parameters.refractory_period
            },
            'layers': [],
            'connections': []
        }
        
        # Serialize layers
        for layer in network.layers:
            layer_config = {
                'id': layer.id,
                'type': layer.layer_type.value,
                'size': layer.size,
                'neuron_model': layer.neuron_model,
                'tau_m': layer.tau_m,
                'tau_syn': layer.tau_syn,
                'capacitance': layer.capacitance
            }
            
            if layer.plasticity_enabled:
                layer_config['plasticity'] = {
                    'enabled': True,
                    'a_plus': layer.stdp_a_plus,
                    'a_minus': layer.stdp_a_minus,
                    'tau_plus': layer.stdp_tau_plus,
                    'tau_minus': layer.stdp_tau_minus
                }
            
            config['layers'].append(layer_config)
        
        # Serialize connections
        for conn in network.connections:
            conn_config = {
                'source': conn.source_layer,
                'target': conn.target_layer,
                'connectivity': conn.connectivity.value,
                'weight_distribution': conn.weight_distribution,
                'sparsity': conn.sparsity,
                'weight_mean': conn.weight_mean,
                'weight_std': conn.weight_std,
                'delay_mean': conn.delay_mean,
                'delay_std': conn.delay_std
            }
            
            config['connections'].append(conn_config)
        
        return config