"""Frontend parsers for different SNN description formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml
import json
from pydantic import BaseModel, Field

from ..models.network import Network, LayerType, Neuron, Synapse, Layer


class NetworkParser(ABC):
    """Abstract base class for network parsers."""
    
    @abstractmethod
    def parse(self, source: Union[str, Path, Dict]) -> Network:
        """Parse network description into internal representation."""
        pass
    
    @abstractmethod
    def validate(self, source: Union[str, Path, Dict]) -> List[str]:
        """Validate network description and return list of issues."""
        pass


class NetworkDefinition(BaseModel):
    """Pydantic model for YAML network definitions."""
    
    name: str = Field(..., description="Network name")
    description: Optional[str] = Field(None, description="Network description")
    timestep: float = Field(1.0, description="Simulation timestep in ms")
    
    layers: List[Dict[str, Any]] = Field(..., description="Layer definitions")
    connections: List[Dict[str, Any]] = Field(default_factory=list, description="Connection specifications")
    
    class Config:
        extra = "forbid"


class YAMLNetworkParser(NetworkParser):
    """Parser for YAML-based network definitions."""
    
    def parse(self, source: Union[str, Path, Dict]) -> Network:
        """Parse YAML network definition."""
        if isinstance(source, (str, Path)):
            with open(source, 'r') as f:
                data = yaml.safe_load(f)
        else:
            data = source
        
        # Validate using Pydantic
        net_def = NetworkDefinition(**data)
        
        # Create network
        network = Network(
            name=net_def.name,
            description=net_def.description,
            timestep=net_def.timestep
        )
        
        # Add layers
        layer_map = {}
        for layer_spec in net_def.layers:
            layer_type = LayerType(layer_spec["type"])
            layer_id = network.add_layer(
                layer_type=layer_type,
                size=layer_spec["size"],
                neuron_type=layer_spec.get("neuron_type", "LIF"),
                **layer_spec.get("parameters", {})
            )
            layer_map[layer_spec["name"]] = layer_id
        
        # Add connections
        for conn_spec in net_def.connections:
            pre_layer = layer_map[conn_spec["from"]]
            post_layer = layer_map[conn_spec["to"]]
            
            network.connect_layers(
                pre_layer_id=pre_layer,
                post_layer_id=post_layer,
                connectivity_pattern=conn_spec.get("pattern", "all_to_all"),
                weight_distribution=conn_spec.get("weight_distribution", "uniform"),
                weight_params=conn_spec.get("weight_params", {"min": 0.1, "max": 1.0}),
                sparsity=conn_spec.get("sparsity", 1.0)
            )
        
        return network
    
    def validate(self, source: Union[str, Path, Dict]) -> List[str]:
        """Validate YAML network definition."""
        issues = []
        
        try:
            if isinstance(source, (str, Path)):
                with open(source, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                data = source
            
            # Validate with Pydantic
            NetworkDefinition(**data)
            
            # Additional semantic validation
            if "layers" in data:
                layer_names = [layer["name"] for layer in data["layers"]]
                if len(layer_names) != len(set(layer_names)):
                    issues.append("Duplicate layer names found")
                
                # Check connection references
                if "connections" in data:
                    for conn in data["connections"]:
                        if conn["from"] not in layer_names:
                            issues.append(f"Connection references unknown layer: {conn['from']}")
                        if conn["to"] not in layer_names:
                            issues.append(f"Connection references unknown layer: {conn['to']}")
        
        except Exception as e:
            issues.append(f"Parsing error: {str(e)}")
        
        return issues


class PyNNParser(NetworkParser):
    """Parser for PyNN network descriptions (placeholder)."""
    
    def parse(self, source: Union[str, Path, Dict]) -> Network:
        """Parse PyNN network definition."""
        # This would integrate with PyNN to extract network structure
        # For now, return a simple example network
        
        network = Network(name="PyNN_Network")
        
        # Add basic layers
        input_layer = network.add_layer(LayerType.INPUT, 100, "LIF")
        hidden_layer = network.add_layer(LayerType.HIDDEN, 200, "LIF") 
        output_layer = network.add_layer(LayerType.OUTPUT, 10, "LIF")
        
        # Connect layers
        network.connect_layers(input_layer, hidden_layer, sparsity=0.5)
        network.connect_layers(hidden_layer, output_layer, sparsity=0.8)
        
        return network
    
    def validate(self, source: Union[str, Path, Dict]) -> List[str]:
        """Validate PyNN network definition."""
        return ["PyNN parser not fully implemented yet"]


class Brian2Parser(NetworkParser):
    """Parser for Brian2 network descriptions (placeholder)."""
    
    def parse(self, source: Union[str, Path, Dict]) -> Network:
        """Parse Brian2 network definition."""
        # This would integrate with Brian2 to extract network structure
        
        network = Network(name="Brian2_Network")
        
        # Add example structure
        input_layer = network.add_layer(LayerType.INPUT, 784, "LIF")  # MNIST input
        hidden_layer = network.add_layer(LayerType.HIDDEN, 300, "LIF")
        output_layer = network.add_layer(LayerType.OUTPUT, 10, "LIF")
        
        network.connect_layers(input_layer, hidden_layer, sparsity=0.3)
        network.connect_layers(hidden_layer, output_layer, sparsity=0.9)
        
        return network
    
    def validate(self, source: Union[str, Path, Dict]) -> List[str]:
        """Validate Brian2 network definition."""
        return ["Brian2 parser not fully implemented yet"]


class JSONNetworkParser(NetworkParser):
    """Parser for JSON-based network definitions."""
    
    def parse(self, source: Union[str, Path, Dict]) -> Network:
        """Parse JSON network definition."""
        if isinstance(source, (str, Path)):
            with open(source, 'r') as f:
                data = json.load(f)
        else:
            data = source
        
        # Convert to YAML format and use existing parser
        yaml_parser = YAMLNetworkParser()
        return yaml_parser.parse(data)
    
    def validate(self, source: Union[str, Path, Dict]) -> List[str]:
        """Validate JSON network definition."""
        yaml_parser = YAMLNetworkParser()
        return yaml_parser.validate(source)


def get_parser(format_type: str) -> NetworkParser:
    """Get appropriate parser for the given format."""
    parsers = {
        "yaml": YAMLNetworkParser,
        "yml": YAMLNetworkParser, 
        "json": JSONNetworkParser,
        "pynn": PyNNParser,
        "brian2": Brian2Parser,
    }
    
    if format_type.lower() not in parsers:
        raise ValueError(f"Unsupported format: {format_type}")
    
    return parsers[format_type.lower()]()


def parse_network_file(file_path: Path) -> Network:
    """Parse network from file, auto-detecting format."""
    suffix = file_path.suffix.lower()
    format_map = {
        ".yaml": "yaml",
        ".yml": "yaml", 
        ".json": "json",
    }
    
    if suffix not in format_map:
        raise ValueError(f"Cannot detect format for file: {file_path}")
    
    parser = get_parser(format_map[suffix])
    return parser.parse(file_path)