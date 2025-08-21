"""Input sanitization and validation for enhanced security."""

import re
import os
from pathlib import Path
from typing import Dict, Any, List, Union
import yaml
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Comprehensive input sanitization for all user inputs."""
    
    # Security patterns to detect potential threats
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Directory traversal
        r'__import__',  # Python imports
        r'eval\s*\(',  # Code evaluation
        r'exec\s*\(',  # Code execution
        r'subprocess',  # System calls
        r'os\.system',  # System commands
        r'open\s*\(',  # File operations
        r'<script',  # XSS attempts
        r'javascript:',  # JS injection
        r'vbscript:',  # VB injection
        r'data:.*base64',  # Data URLs
    ]
    
    # Safe file extensions
    SAFE_EXTENSIONS = {'.yaml', '.yml', '.json', '.txt', '.md'}
    
    # Maximum file sizes (bytes)
    MAX_NETWORK_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_CONFIG_FILE_SIZE = 1 * 1024 * 1024    # 1MB
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                  for pattern in self.DANGEROUS_PATTERNS]
    
    def sanitize_string(self, input_str: str, max_length: int = 1000) -> str:
        """Sanitize a string input for safety."""
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        # Length check
        if len(input_str) > max_length:
            raise ValueError(f"Input too long (max {max_length} characters)")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(input_str):
                logger.warning(f"Potentially dangerous pattern detected: {pattern.pattern}")
                raise ValueError(f"Input contains potentially dangerous content")
        
        # Remove null bytes and other control characters
        sanitized = input_str.replace('\x00', '').replace('\r', '').strip()
        
        return sanitized
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and sanitize file paths."""
        path = Path(file_path)
        
        # Convert to absolute path and resolve
        try:
            abs_path = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid file path: {e}")
        
        # Check if path exists
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {abs_path}")
        
        # Check if it's a file (not directory)
        if not abs_path.is_file():
            raise ValueError(f"Path is not a file: {abs_path}")
        
        # Check file extension
        if abs_path.suffix not in self.SAFE_EXTENSIONS:
            raise ValueError(f"Unsafe file extension: {abs_path.suffix}")
        
        # Check file size
        file_size = abs_path.stat().st_size
        if abs_path.suffix in {'.yaml', '.yml', '.json'}:
            max_size = self.MAX_NETWORK_FILE_SIZE
        else:
            max_size = self.MAX_CONFIG_FILE_SIZE
            
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size} bytes (max {max_size})")
        
        # Check for symbolic links (security risk)
        if abs_path.is_symlink():
            logger.warning(f"Symbolic link detected: {abs_path}")
            raise ValueError("Symbolic links not allowed for security")
        
        return abs_path
    
    def validate_network_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize network configuration."""
        if not isinstance(config, dict):
            raise ValueError("Network config must be a dictionary")
        
        # Required fields
        required_fields = ['name', 'layers']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Sanitize string fields
        string_fields = ['name', 'description']
        for field in string_fields:
            if field in config and isinstance(config[field], str):
                config[field] = self.sanitize_string(config[field])
        
        # Validate layers
        if not isinstance(config['layers'], list):
            raise ValueError("Layers must be a list")
        
        if len(config['layers']) > 100:  # Reasonable limit
            raise ValueError("Too many layers (max 100)")
        
        for i, layer in enumerate(config['layers']):
            if not isinstance(layer, dict):
                raise ValueError(f"Layer {i} must be a dictionary")
            
            # Validate layer size
            if 'size' in layer:
                if not isinstance(layer['size'], int) or layer['size'] <= 0:
                    raise ValueError(f"Layer {i} size must be positive integer")
                if layer['size'] > 1000000:  # 1M neurons max per layer
                    raise ValueError(f"Layer {i} too large (max 1M neurons)")
        
        # Validate connections if present
        if 'connections' in config:
            if not isinstance(config['connections'], list):
                raise ValueError("Connections must be a list")
            
            if len(config['connections']) > 1000:  # Reasonable limit
                raise ValueError("Too many connections (max 1000)")
        
        return config
    
    def validate_fpga_target(self, target: str) -> str:
        """Validate FPGA target string."""
        if not isinstance(target, str):
            raise ValueError("FPGA target must be a string")
        
        # Whitelist of valid targets
        valid_targets = {
            'artix7_35t', 'artix7_100t', 
            'cyclone5_gx', 'cyclone5_gt'
        }
        
        target = target.lower().strip()
        if target not in valid_targets:
            raise ValueError(f"Invalid FPGA target: {target}")
        
        return target
    
    def validate_optimization_level(self, level: Union[int, str]) -> int:
        """Validate optimization level."""
        try:
            level = int(level)
        except (ValueError, TypeError):
            raise ValueError("Optimization level must be an integer")
        
        if level < 0 or level > 3:
            raise ValueError("Optimization level must be 0-3")
        
        return level
    
    def validate_power_budget(self, power_mw: Union[float, int, None]) -> float:
        """Validate power budget in milliwatts."""
        if power_mw is None:
            return None
        
        try:
            power_mw = float(power_mw)
        except (ValueError, TypeError):
            raise ValueError("Power budget must be a number")
        
        if power_mw <= 0:
            raise ValueError("Power budget must be positive")
        
        if power_mw > 100000:  # 100W max
            raise ValueError("Power budget too high (max 100W)")
        
        return power_mw
    
    def scan_file_content(self, file_path: Path) -> bool:
        """Scan file content for potential security issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            raise ValueError(f"File contains non-UTF-8 content: {file_path}")
        except Exception as e:
            raise ValueError(f"Cannot read file: {e}")
        
        # Check file content against dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                logger.warning(f"Dangerous pattern in file {file_path}: {pattern.pattern}")
                return False
        
        # Check for extremely large files
        if len(content) > self.MAX_NETWORK_FILE_SIZE:
            raise ValueError(f"File content too large: {len(content)} bytes")
        
        # Validate YAML/JSON structure
        if file_path.suffix in {'.yaml', '.yml'}:
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML format: {e}")
        
        return True