"""
Configuration manager for handling settings from environment variables, files, and defaults.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from dataclasses import dataclass, field


@dataclass
class FPGASettings:
    """FPGA-specific configuration settings."""
    vivado_path: str = "/opt/Xilinx/Vivado/2024.2"
    quartus_path: str = "/opt/intelFPGA/23.1std/quartus"
    default_target: str = "artix7_35t"
    default_frequency_mhz: float = 100.0
    enable_resource_checking: bool = True
    resource_warning_threshold: int = 80


@dataclass
class CompilationSettings:
    """Compilation-specific configuration settings."""
    default_optimization_level: int = 2
    hdl_output_dir: str = "./output"
    enable_parallel_compilation: bool = True
    enable_plasticity_default: bool = False
    generate_testbench_default: bool = True
    enable_caching: bool = True
    cache_expiration_hours: int = 24


@dataclass
class LoggingSettings:
    """Logging configuration settings."""
    log_level: str = "INFO"
    log_file: str = "./logs/spiking_fpga.log"
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Settings:
    """Main application settings container."""
    fpga: FPGASettings = field(default_factory=FPGASettings)
    compilation: CompilationSettings = field(default_factory=CompilationSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    
    # Directories
    temp_dir: str = "./tmp"
    cache_dir: str = "./cache"
    data_dir: str = "./data"
    
    # Security settings
    enable_input_validation: bool = True
    max_network_size: int = 1000000
    max_file_size_bytes: int = 104857600  # 100MB
    
    # Performance settings
    enable_performance_monitoring: bool = True
    enable_resource_tracking: bool = True


class ConfigManager:
    """Manages application configuration from multiple sources."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self._settings = Settings()
        
        # Load configuration in order of precedence
        self._load_default_config()
        if config_file:
            self._load_config_file(config_file)
        self._load_environment_variables()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_default_config(self):
        """Load default configuration values."""
        # Default values are already set in the Settings dataclass
        self.logger.debug("Loaded default configuration")
    
    def _load_config_file(self, config_file: str):
        """Load configuration from file (YAML or JSON)."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self._update_settings_from_dict(config_data)
            self.logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_file}: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # FPGA settings
            'VIVADO_PATH': ('fpga', 'vivado_path'),
            'QUARTUS_PATH': ('fpga', 'quartus_path'),
            'DEFAULT_FPGA_TARGET': ('fpga', 'default_target'),
            'DEFAULT_TARGET_FREQ': ('fpga', 'default_frequency_mhz'),
            'RESOURCE_WARNING_THRESHOLD': ('fpga', 'resource_warning_threshold'),
            
            # Compilation settings
            'DEFAULT_OPTIMIZATION_LEVEL': ('compilation', 'default_optimization_level'),
            'HDL_OUTPUT_DIR': ('compilation', 'hdl_output_dir'),
            'ENABLE_PARALLEL_COMPILATION': ('compilation', 'enable_parallel_compilation'),
            'ENABLE_CACHING': ('compilation', 'enable_caching'),
            'CACHE_EXPIRATION_HOURS': ('compilation', 'cache_expiration_hours'),
            
            # Logging settings
            'LOG_LEVEL': ('logging', 'log_level'),
            'LOG_FILE': ('logging', 'log_file'),
            
            # Directory settings
            'TEMP_DIR': (None, 'temp_dir'),
            'CACHE_DIR': (None, 'cache_dir'),
            'DATA_DIR': (None, 'data_dir'),
            
            # Security settings
            'MAX_NETWORK_SIZE': (None, 'max_network_size'),
            'MAX_FILE_SIZE_BYTES': (None, 'max_file_size_bytes'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert value to appropriate type
                    converted_value = self._convert_env_value(value)
                    
                    if section:
                        section_obj = getattr(self._settings, section)
                        setattr(section_obj, key, converted_value)
                    else:
                        setattr(self._settings, key, converted_value)
                    
                    self.logger.debug(f"Set {env_var} = {converted_value}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to set {env_var}: {e}")
        
        self.logger.debug("Loaded environment variable configuration")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def _update_settings_from_dict(self, config_data: Dict[str, Any]):
        """Update settings from configuration dictionary."""
        for key, value in config_data.items():
            if hasattr(self._settings, key):
                if isinstance(value, dict):
                    # Update nested settings
                    section_obj = getattr(self._settings, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(section_obj, sub_key):
                            setattr(section_obj, sub_key, sub_value)
                else:
                    # Update top-level settings
                    setattr(self._settings, key, value)
    
    def _create_directories(self):
        """Create necessary directories based on configuration."""
        directories = [
            self._settings.temp_dir,
            self._settings.cache_dir,
            self._settings.data_dir,
            self._settings.compilation.hdl_output_dir,
            os.path.dirname(self._settings.logging.log_file)
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_settings(self) -> Settings:
        """Get current settings."""
        return self._settings
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self._settings
        
        try:
            for k in keys:
                value = getattr(value, k)
            return value
        except AttributeError:
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot-notation key."""
        keys = key.split('.')
        target = self._settings
        
        # Navigate to the parent object
        for k in keys[:-1]:
            target = getattr(target, k)
        
        # Set the final value
        setattr(target, keys[-1], value)
    
    def save_config(self, output_file: str):
        """Save current configuration to file."""
        config_data = self._settings_to_dict()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Configuration saved to {output_file}")
    
    def _settings_to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        result = {}
        
        for field_name in self._settings.__dataclass_fields__:
            value = getattr(self._settings, field_name)
            
            if hasattr(value, '__dataclass_fields__'):
                # Convert nested dataclass to dict
                nested_dict = {}
                for nested_field in value.__dataclass_fields__:
                    nested_dict[nested_field] = getattr(value, nested_field)
                result[field_name] = nested_dict
            else:
                result[field_name] = value
        
        return result
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        errors = []
        
        # Validate FPGA paths
        if self._settings.fpga.vivado_path:
            vivado_path = Path(self._settings.fpga.vivado_path)
            if not vivado_path.exists():
                errors.append(f"Vivado path does not exist: {vivado_path}")
        
        if self._settings.fpga.quartus_path:
            quartus_path = Path(self._settings.fpga.quartus_path)
            if not quartus_path.exists():
                errors.append(f"Quartus path does not exist: {quartus_path}")
        
        # Validate optimization level
        if not (0 <= self._settings.compilation.default_optimization_level <= 3):
            errors.append("Optimization level must be between 0 and 3")
        
        # Validate frequency
        if not (1.0 <= self._settings.fpga.default_frequency_mhz <= 1000.0):
            errors.append("Default frequency must be between 1 and 1000 MHz")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self._settings.logging.log_level not in valid_log_levels:
            errors.append(f"Log level must be one of: {valid_log_levels}")
        
        if errors:
            for error in errors:
                self.logger.error(f"Config validation error: {error}")
            return False
        
        return True
    
    def get_fpga_tool_path(self, vendor: str) -> Optional[str]:
        """Get FPGA tool path for specific vendor."""
        if vendor.lower() == 'xilinx':
            return self._settings.fpga.vivado_path
        elif vendor.lower() == 'intel':
            return self._settings.fpga.quartus_path
        else:
            return None
    
    def get_compilation_cache_config(self) -> Dict[str, Any]:
        """Get compilation cache configuration."""
        return {
            'enabled': self._settings.compilation.enable_caching,
            'cache_dir': self._settings.cache_dir,
            'expiration_hours': self._settings.compilation.cache_expiration_hours
        }


# Global configuration manager instance
_config_manager = None

def get_config(config_file: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager