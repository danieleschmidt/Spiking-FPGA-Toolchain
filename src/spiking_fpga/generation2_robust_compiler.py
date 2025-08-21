"""Generation 2 Robust Network Compiler with enhanced error handling and security."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime

from .core import FPGATarget
from .models.network import Network
from .models.optimization import OptimizationLevel, ResourceEstimate
from .network_compiler import NetworkCompiler, CompilationConfig, CompilationResult
from .reliability.advanced_error_recovery import AdvancedErrorRecovery
from .security.enhanced_security_framework import EnhancedSecurityFramework
from .utils import configure_logging, StructuredLogger

logger = logging.getLogger(__name__)


@dataclass
class RobustCompilationConfig(CompilationConfig):
    """Enhanced compilation configuration with robustness features."""
    
    # Security settings
    enable_input_validation: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_hour: int = 100
    
    # Error recovery settings
    enable_error_recovery: bool = True
    max_recovery_attempts: int = 3
    create_backup_outputs: bool = True
    
    # Monitoring settings
    enable_security_monitoring: bool = True
    security_log_file: Optional[Path] = None
    error_log_file: Optional[Path] = None
    
    # Validation settings
    strict_validation: bool = True
    allowed_file_extensions: List[str] = None
    max_file_size_mb: int = 10
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.yaml', '.yml', '.json']


class Generation2RobustCompiler(NetworkCompiler):
    """Enhanced compiler with comprehensive robustness features."""
    
    def __init__(self, target: FPGATarget, 
                 security_config: Optional[RobustCompilationConfig] = None,
                 log_level: str = "INFO",
                 log_file: Optional[Path] = None):
        
        # Initialize base compiler
        super().__init__(target, enable_monitoring=True, log_level=log_level, log_file=log_file)
        
        # Initialize robustness components
        self.security_config = security_config or RobustCompilationConfig()
        
        # Enhanced error recovery
        if self.security_config.enable_error_recovery:
            error_log = self.security_config.error_log_file or (log_file.parent / "errors.jsonl" if log_file else None)
            self.error_recovery = AdvancedErrorRecovery(error_log)
            self._register_custom_recovery_strategies()
        else:
            self.error_recovery = None
            
        # Enhanced security framework
        if self.security_config.enable_security_monitoring:
            security_log = self.security_config.security_log_file or (log_file.parent / "security.jsonl" if log_file else None)
            self.security_framework = EnhancedSecurityFramework(security_log)
        else:
            self.security_framework = None
            
        logger.info("Generation2RobustCompiler initialized", extra={
            "target": target.value,
            "error_recovery_enabled": self.security_config.enable_error_recovery,
            "security_monitoring_enabled": self.security_config.enable_security_monitoring
        })
        
    def _register_custom_recovery_strategies(self):
        """Register custom recovery strategies for FPGA compilation."""
        from .reliability.advanced_error_recovery import RecoveryStrategy
        
        # HDL generation recovery
        self.error_recovery.register_strategy(RecoveryStrategy(
            name="hdl_generation_recovery",
            handler=self._hdl_generation_recovery,
            max_attempts=2,
            applicable_errors=["HDLGenerationError", "TemplateError"],
            priority=7
        ))
        
        # Synthesis recovery
        self.error_recovery.register_strategy(RecoveryStrategy(
            name="synthesis_recovery", 
            handler=self._synthesis_recovery,
            max_attempts=1,
            applicable_errors=["SynthesisError", "ToolchainError"],
            priority=6
        ))
        
        # Resource estimation recovery
        self.error_recovery.register_strategy(RecoveryStrategy(
            name="resource_recovery",
            handler=self._resource_recovery,
            max_attempts=2,
            applicable_errors=["ResourceEstimationError"],
            priority=5
        ))
        
    def compile_robust(self, network: Union[Network, str, Path, Dict[str, Any]], 
                      output_dir: Union[str, Path],
                      config: Optional[RobustCompilationConfig] = None) -> CompilationResult:
        """Compile with enhanced robustness features."""
        
        if config is None:
            config = self.security_config
            
        try:
            # Security validation
            if config.enable_input_validation and self.security_framework:
                self._validate_inputs(network, output_dir, config)
                
            # Rate limiting check
            if config.enable_rate_limiting and self.security_framework:
                client_id = self._get_client_identifier()
                if self.security_framework.check_rate_limit(client_id):
                    raise ValueError("Rate limit exceeded. Please try again later.")
                    
            # Create backup directory if enabled
            backup_dir = None
            if config.create_backup_outputs:
                backup_dir = self._create_backup_directory(output_dir)
                
            # Attempt compilation with error recovery
            result = self._compile_with_recovery(network, output_dir, config)
            
            # Validate outputs
            if result.success:
                self._validate_compilation_outputs(result, output_dir)
                
            # Create backup if successful
            if result.success and backup_dir:
                self._backup_outputs(output_dir, backup_dir)
                
            return result
            
        except Exception as e:
            logger.error(f"Robust compilation failed: {e}")
            
            # Attempt error recovery
            if self.error_recovery:
                recovery_success = self.error_recovery.handle_error(e, "robust_compiler")
                if recovery_success:
                    # Retry compilation once after recovery
                    logger.info("Retrying compilation after error recovery")
                    try:
                        return self._compile_with_recovery(network, output_dir, config)
                    except Exception as retry_error:
                        logger.error(f"Retry compilation failed: {retry_error}")
                        
            # Return failure result
            return CompilationResult(
                success=False,
                network=network if isinstance(network, Network) else Network(name="failed"),
                optimized_network=Network(name="failed"),
                hdl_files={},
                resource_estimate=ResourceEstimate(0, 0, 0, 0, 0, 0),
                optimization_stats={},
                errors=[str(e)]
            )
            
    def _validate_inputs(self, network: Any, output_dir: Any, config: RobustCompilationConfig):
        """Validate all inputs for security."""
        # Validate network input
        if isinstance(network, dict):
            self.security_framework.validate_and_sanitize(network, "network_definition")
        elif isinstance(network, (str, Path)):
            file_path = Path(network)
            if not self.security_framework.validate_file_path(file_path):
                raise ValueError(f"Invalid or unsafe file path: {file_path}")
                
            # Check file size
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > config.max_file_size_mb:
                    raise ValueError(f"File too large: {file_size_mb:.1f}MB (max: {config.max_file_size_mb}MB)")
                    
            # Check file extension
            if file_path.suffix not in config.allowed_file_extensions:
                raise ValueError(f"File extension not allowed: {file_path.suffix}")
                
        # Validate output directory
        output_path = Path(output_dir)
        if not self.security_framework.validate_file_path(output_path):
            raise ValueError(f"Invalid or unsafe output directory: {output_path}")
            
    def _get_client_identifier(self) -> str:
        """Get client identifier for rate limiting."""
        # In a real implementation, this would be based on request context
        # For now, use a simple identifier
        import socket
        return f"local_{socket.gethostname()}"
        
    def _create_backup_directory(self, output_dir: Union[str, Path]) -> Path:
        """Create backup directory."""
        output_path = Path(output_dir)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_dir = output_path.parent / f"{output_path.name}_backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        return backup_dir
        
    def _compile_with_recovery(self, network: Any, output_dir: Any, 
                              config: RobustCompilationConfig) -> CompilationResult:
        """Compile with error recovery capabilities."""
        try:
            return super().compile(network, output_dir, config)
        except Exception as e:
            if self.error_recovery:
                recovery_success = self.error_recovery.handle_error(e, "compilation")
                if recovery_success:
                    # Retry the compilation
                    return super().compile(network, output_dir, config)
            raise
            
    def _validate_compilation_outputs(self, result: CompilationResult, output_dir: Union[str, Path]):
        """Validate compilation outputs."""
        output_path = Path(output_dir)
        
        # Check that expected files were generated
        expected_files = ["hdl/snn_top.v", "hdl/lif_neuron.v"]
        missing_files = []
        
        for expected_file in expected_files:
            file_path = output_path / expected_file
            if not file_path.exists():
                missing_files.append(expected_file)
                
        if missing_files:
            logger.warning(f"Missing expected output files: {missing_files}")
            
        # Validate HDL files for basic syntax
        for hdl_file in result.hdl_files.values():
            if hdl_file.exists() and hdl_file.suffix == '.v':
                self._validate_verilog_syntax(hdl_file)
                
    def _validate_verilog_syntax(self, hdl_file: Path):
        """Basic Verilog syntax validation."""
        try:
            with open(hdl_file, 'r') as f:
                content = f.read()
                
            # Basic checks
            if 'module' not in content:
                logger.warning(f"No module declaration found in {hdl_file}")
            if 'endmodule' not in content:
                logger.warning(f"No endmodule found in {hdl_file}")
                
        except Exception as e:
            logger.warning(f"Could not validate Verilog file {hdl_file}: {e}")
            
    def _backup_outputs(self, output_dir: Union[str, Path], backup_dir: Path):
        """Create backup of successful compilation outputs."""
        import shutil
        try:
            output_path = Path(output_dir)
            for item in output_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, backup_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, backup_dir / item.name, dirs_exist_ok=True)
            logger.info(f"Created backup at: {backup_dir}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            
    def _hdl_generation_recovery(self, error_context) -> bool:
        """Recovery strategy for HDL generation errors."""
        try:
            logger.info("Attempting HDL generation recovery")
            # Could implement template fallbacks, different HDL variants, etc.
            return False  # Simplified for now
        except Exception:
            return False
            
    def _synthesis_recovery(self, error_context) -> bool:
        """Recovery strategy for synthesis errors.""" 
        try:
            logger.info("Attempting synthesis recovery")
            # Could implement different synthesis options, constraint adjustments, etc.
            return False  # Simplified for now
        except Exception:
            return False
            
    def _resource_recovery(self, error_context) -> bool:
        """Recovery strategy for resource estimation errors."""
        try:
            logger.info("Attempting resource estimation recovery")
            # Could implement fallback estimation methods
            return True  # Allow graceful degradation
        except Exception:
            return False
            
    def get_robustness_statistics(self) -> Dict[str, Any]:
        """Get comprehensive robustness statistics."""
        stats = {
            "compiler_type": "Generation2RobustCompiler",
            "target": self.target.value
        }
        
        if self.error_recovery:
            stats["error_recovery"] = self.error_recovery.get_error_statistics()
            
        if self.security_framework:
            stats["security"] = self.security_framework.get_security_summary()
            
        if hasattr(self, 'health_monitor') and self.health_monitor:
            stats["health"] = {
                "monitoring_active": True,
                "uptime_hours": (datetime.utcnow() - self.health_monitor.start_time).total_seconds() / 3600
            }
            
        return stats


def compile_network_robust(network: Union[Network, str, Path, Dict[str, Any]],
                          target: FPGATarget,
                          output_dir: Union[str, Path] = "./output_gen2",
                          optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
                          power_budget_mw: Optional[float] = None,
                          run_synthesis: bool = False,
                          enable_security: bool = True,
                          enable_error_recovery: bool = True) -> CompilationResult:
    """Compile with Generation 2 robustness features.
    
    Args:
        network: Network object, path to network definition file, or network dictionary
        target: Target FPGA platform
        output_dir: Directory for generated files
        optimization_level: Level of optimization to apply
        power_budget_mw: Power budget in milliwatts (optional)
        run_synthesis: Whether to run FPGA synthesis
        enable_security: Enable security monitoring and validation
        enable_error_recovery: Enable advanced error recovery
    
    Returns:
        CompilationResult with generated files and statistics
    """
    config = RobustCompilationConfig(
        optimization_level=optimization_level,
        power_budget_mw=power_budget_mw,
        run_synthesis=run_synthesis,
        enable_input_validation=enable_security,
        enable_security_monitoring=enable_security,
        enable_error_recovery=enable_error_recovery
    )
    
    compiler = Generation2RobustCompiler(target)
    return compiler.compile_robust(network, output_dir, config)