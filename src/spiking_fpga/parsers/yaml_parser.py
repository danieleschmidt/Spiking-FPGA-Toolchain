"""
YAML network configuration parser.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..models.network import SNNNetwork, Layer, Connection, NetworkParameters, LayerType, ConnectivityPattern


class YAMLNetworkParser:
    """Parser for YAML network configuration files."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_file(self, yaml_file: str) -> SNNNetwork:
        """
        Parse SNN network from YAML file.
        
        Args:
            yaml_file: Path to YAML configuration file
            
        Returns:
            SNNNetwork object
        """
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_file}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return self.parse_config(config)
    
    def parse_config(self, config: Dict[str, Any]) -> SNNNetwork:
        """
        Parse SNN network from configuration dictionary.
        
        Args:
            config: Network configuration dictionary
            
        Returns:
            SNNNetwork object
        """
        self.logger.info("Parsing network configuration...")
        
        # Extract basic network info
        network_name = config.get('name', 'unnamed_network')
        description = config.get('description', '')
        
        # Parse network parameters
        params_config = config.get('parameters', {})
        parameters = self._parse_parameters(params_config)
        
        # Parse layers
        layers_config = config.get('layers', [])
        layers = self._parse_layers(layers_config)
        
        # Parse connections
        connections_config = config.get('connections', [])
        connections = self._parse_connections(connections_config)
        
        # Extract input/output sizes
        input_size = config.get('input_size')
        output_size = config.get('output_size')
        
        if input_size is None:
            # Infer from input layers
            input_layers = [l for l in layers if l.layer_type == LayerType.INPUT]
            input_size = sum(l.size for l in input_layers)
        
        if output_size is None:
            # Infer from output layers
            output_layers = [l for l in layers if l.layer_type == LayerType.OUTPUT]
            output_size = sum(l.size for l in output_layers)
        
        network = SNNNetwork(
            name=network_name,
            description=description,
            parameters=parameters,
            layers=layers,
            connections=connections,
            input_size=input_size,
            output_size=output_size
        )
        
        # Validate the parsed network
        network.validate_network()
        
        self.logger.info(f"Successfully parsed network '{network_name}' with {len(layers)} layers")
        return network
    
    def _parse_parameters(self, params_config: Dict[str, Any]) -> NetworkParameters:
        """Parse global network parameters."""
        return NetworkParameters(
            dt=params_config.get('dt', 0.1),
            simulation_time=params_config.get('simulation_time', 100.0),
            spike_threshold=params_config.get('spike_threshold', 1.0),
            resting_potential=params_config.get('resting_potential', 0.0),
            reset_potential=params_config.get('reset_potential', 0.0),
            refractory_period=params_config.get('refractory_period', 2.0)
        )
    
    def _parse_layers(self, layers_config: List[Dict[str, Any]]) -> List[Layer]:
        """Parse network layers."""
        layers = []
        
        for layer_config in layers_config:
            layer_id = layer_config.get('id')
            if not layer_id:
                raise ValueError("Layer missing required 'id' field")
            
            layer_type = LayerType(layer_config.get('type', 'hidden'))
            size = layer_config.get('size')
            if not size or size <= 0:
                raise ValueError(f"Layer '{layer_id}' missing or invalid 'size' field")
            
            # Parse neuron model and parameters
            neuron_model = layer_config.get('neuron_model', 'LIF')
            
            # LIF-specific parameters
            tau_m = layer_config.get('tau_m', 20.0)
            tau_syn = layer_config.get('tau_syn', 5.0)
            capacitance = layer_config.get('capacitance', 1.0)
            
            # Plasticity parameters
            plasticity_config = layer_config.get('plasticity', {})
            plasticity_enabled = plasticity_config.get('enabled', False)
            stdp_a_plus = plasticity_config.get('a_plus', 0.1)
            stdp_a_minus = plasticity_config.get('a_minus', 0.12)
            stdp_tau_plus = plasticity_config.get('tau_plus', 20.0)
            stdp_tau_minus = plasticity_config.get('tau_minus', 20.0)
            
            layer = Layer(
                id=layer_id,
                layer_type=layer_type,
                size=size,
                neuron_model=neuron_model,
                tau_m=tau_m,
                tau_syn=tau_syn,
                capacitance=capacitance,
                plasticity_enabled=plasticity_enabled,
                stdp_a_plus=stdp_a_plus,
                stdp_a_minus=stdp_a_minus,
                stdp_tau_plus=stdp_tau_plus,
                stdp_tau_minus=stdp_tau_minus
            )
            
            layers.append(layer)
        
        return layers
    
    def _parse_connections(self, connections_config: List[Dict[str, Any]]) -> List[Connection]:
        """Parse network connections."""
        connections = []
        
        for conn_config in connections_config:
            source_layer = conn_config.get('source')
            target_layer = conn_config.get('target')
            
            if not source_layer or not target_layer:
                raise ValueError("Connection missing 'source' or 'target' field")
            
            connectivity = ConnectivityPattern(conn_config.get('connectivity', 'sparse_random'))
            
            # Weight distribution parameters
            weight_dist = conn_config.get('weight_distribution', 'uniform')
            sparsity = conn_config.get('sparsity', 0.1)
            weight_mean = conn_config.get('weight_mean', 0.5)
            weight_std = conn_config.get('weight_std', 0.1)
            
            # Delay parameters
            delay_mean = conn_config.get('delay_mean', 1.0)
            delay_std = conn_config.get('delay_std', 0.2)
            
            connection = Connection(
                source_layer=source_layer,
                target_layer=target_layer,
                connectivity=connectivity,
                weight_distribution=weight_dist,
                sparsity=sparsity,
                weight_mean=weight_mean,
                weight_std=weight_std,
                delay_mean=delay_mean,
                delay_std=delay_std
            )
            
            connections.append(connection)
        
        return connections
    
    def write_config(self, network: SNNNetwork, output_file: str):
        """
        Write SNNNetwork to YAML configuration file.
        
        Args:
            network: SNNNetwork to serialize
            output_file: Output YAML file path
        """
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
        
        # Write to file
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Network configuration written to {output_file}")
    
    def validate_yaml_schema(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate YAML configuration against expected schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required top-level fields
        required_fields = ['layers']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate layers
        if 'layers' in config:
            for i, layer in enumerate(config['layers']):
                if not isinstance(layer, dict):
                    errors.append(f"Layer {i} must be a dictionary")
                    continue
                
                # Check required layer fields
                if 'id' not in layer:
                    errors.append(f"Layer {i} missing required 'id' field")
                if 'size' not in layer:
                    errors.append(f"Layer {i} missing required 'size' field")
                elif not isinstance(layer['size'], int) or layer['size'] <= 0:
                    errors.append(f"Layer {i} 'size' must be a positive integer")
                
                # Check layer type
                if 'type' in layer and layer['type'] not in [t.value for t in LayerType]:
                    errors.append(f"Layer {i} has invalid type: {layer['type']}")
        
        # Validate connections
        if 'connections' in config:
            layer_ids = {layer.get('id') for layer in config.get('layers', [])}
            
            for i, conn in enumerate(config['connections']):
                if not isinstance(conn, dict):
                    errors.append(f"Connection {i} must be a dictionary")
                    continue
                
                # Check required connection fields
                if 'source' not in conn:
                    errors.append(f"Connection {i} missing required 'source' field")
                elif conn['source'] not in layer_ids:
                    errors.append(f"Connection {i} references unknown source layer: {conn['source']}")
                
                if 'target' not in conn:
                    errors.append(f"Connection {i} missing required 'target' field")
                elif conn['target'] not in layer_ids:
                    errors.append(f"Connection {i} references unknown target layer: {conn['target']}")
                
                # Check connectivity pattern
                if 'connectivity' in conn and conn['connectivity'] not in [p.value for p in ConnectivityPattern]:
                    errors.append(f"Connection {i} has invalid connectivity: {conn['connectivity']}")
                
                # Check sparsity range
                if 'sparsity' in conn:
                    sparsity = conn['sparsity']
                    if not isinstance(sparsity, (int, float)) or not (0.0 <= sparsity <= 1.0):
                        errors.append(f"Connection {i} sparsity must be between 0.0 and 1.0")
        
        return errors