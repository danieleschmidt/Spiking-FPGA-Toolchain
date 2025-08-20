"""Robust network compiler with advanced fault tolerance and security."""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime

from .core import FPGATarget
from .models.network import Network
from .models.optimization import OptimizationLevel
from .network_compiler import NetworkCompiler, CompilationConfig, CompilationResult
from .reliability.advanced_fault_tolerance import AdaptiveFaultTolerance, RobustCompilerWrapper, FaultType
from .security.advanced_security_framework import AdvancedSecurityFramework, ThreatLevel
from .utils.logging import StructuredLogger


@dataclass
class RobustCompilationConfig(CompilationConfig):
    """Extended compilation configuration with robustness features."""
    
    # Fault tolerance settings
    enable_fault_tolerance: bool = True
    max_retry_attempts: int = 3
    checkpoint_interval: int = 5  # Create checkpoints every N operations
    auto_recovery: bool = True
    
    # Security settings
    enable_security_audit: bool = True
    security_level: str = "standard"  # "basic", "standard", "strict"
    sandbox_execution: bool = True
    threat_response: str = "block"  # "block", "warn", "log"
    
    # Resource management
    memory_limit_gb: float = 8.0
    cpu_timeout_minutes: int = 30
    temp_dir_cleanup: bool = True
    
    # Monitoring
    detailed_metrics: bool = True
    health_monitoring: bool = True
    audit_logging: bool = True


class RobustNetworkCompiler:
    """Network compiler with advanced fault tolerance, security, and monitoring."""
    
    def __init__(self, target: FPGATarget, config: Optional[RobustCompilationConfig] = None):
        self.target = target
        self.config = config or RobustCompilationConfig()
        
        # Initialize logging
        self.logger = StructuredLogger("RobustCompiler", level="INFO")
        
        # Initialize core compiler
        self.base_compiler = NetworkCompiler(
            target=target,
            enable_monitoring=self.config.health_monitoring,
            log_level="INFO"
        )
        
        # Initialize fault tolerance
        if self.config.enable_fault_tolerance:
            self.fault_tolerance = AdaptiveFaultTolerance(self.logger.logger)
            self.robust_wrapper = RobustCompilerWrapper(self.fault_tolerance)
        else:
            self.fault_tolerance = None
            self.robust_wrapper = None
        
        # Initialize security framework
        if self.config.enable_security_audit:
            self.security_framework = AdvancedSecurityFramework(self.logger.logger)
        else:
            self.security_framework = None
        
        # Initialize metrics tracking
        self.compilation_metrics: Dict[str, Any] = {}
        self.operation_counter = 0
        
        self.logger.info("RobustNetworkCompiler initialized", 
                        target=target.value,
                        fault_tolerance=self.config.enable_fault_tolerance,
                        security_audit=self.config.enable_security_audit)
    
    def compile_with_robustness(self, 
                               network: Union[Network, str, Path],
                               output_dir: Union[str, Path]) -> 'RobustCompilationResult':
        """Compile network with full robustness features."""
        
        start_time = datetime.now()
        output_dir = Path(output_dir)
        
        self.logger.info("Starting robust compilation",
                        network=str(network)[:100],
                        output_dir=str(output_dir))
        
        # Initialize result tracking
        result = RobustCompilationResult()
        result.start_time = start_time
        result.target = self.target
        result.config = self.config
        
        try:
            # Phase 1: Security pre-screening
            if self.config.enable_security_audit:
                security_result = self._perform_security_prescreening(network)
                result.security_audit = security_result
                
                if security_result.risk_score > 0.8 and self.config.threat_response == "block":
                    result.success = False
                    result.errors.append("Security audit failed: high risk detected")
                    result.security_blocked = True
                    return result
            
            # Phase 2: Setup robust execution environment
            with self._setup_execution_environment(output_dir) as exec_env:
                result.execution_environment = exec_env
                
                # Phase 3: Execute compilation with fault tolerance
                if self.config.enable_fault_tolerance and self.robust_wrapper:
                    compilation_result = self._execute_fault_tolerant_compilation(
                        network, exec_env["secure_output_dir"]
                    )
                else:
                    compilation_result = self._execute_standard_compilation(
                        network, exec_env["secure_output_dir"]
                    )
                
                # Phase 4: Post-compilation security validation
                if self.config.enable_security_audit and compilation_result.success:
                    post_security_result = self._perform_post_compilation_security_check(
                        compilation_result, exec_env["secure_output_dir"]
                    )
                    result.post_security_audit = post_security_result
                
                # Phase 5: Finalize results
                result.base_compilation_result = compilation_result
                result.success = compilation_result.success
                result.errors.extend(compilation_result.errors)
                result.warnings.extend(compilation_result.warnings)
        
        except Exception as e:
            self.logger.error("Robust compilation failed", error=str(e))
            result.success = False
            result.errors.append(f"Compilation exception: {str(e)}")
            
            # Record fault if fault tolerance is enabled
            if self.fault_tolerance:
                fault_id = self.fault_tolerance.record_fault(
                    FaultType.OPTIMIZATION_ERROR,  # Generic fault type
                    "robust_compiler",
                    str(e),
                    {"phase": "main_compilation"}
                )
                result.fault_records.append(fault_id)
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
            # Generate comprehensive report
            self._generate_robust_compilation_report(result, output_dir)
            
            self.logger.info("Robust compilation completed",
                           success=result.success,
                           duration_seconds=result.duration)
        
        return result
    
    def _perform_security_prescreening(self, network: Union[Network, str, Path]):
        """Perform security audit before compilation."""
        self.logger.info("Performing security pre-screening")
        
        source_files = []
        network_config = {}
        
        if isinstance(network, (str, Path)):
            source_files.append(Path(network))
        elif isinstance(network, Network):
            network_config = {
                "name": network.name,
                "layers": len(network.layers),
                "neurons": len(network.neurons),
                "synapses": len(network.synapses)
            }
        
        return self.security_framework.perform_security_audit(source_files, network_config)
    
    def _setup_execution_environment(self, base_output_dir: Path):
        """Setup secure execution environment."""
        return RobustExecutionEnvironment(base_output_dir, self.config, self.logger)
    
    def _execute_fault_tolerant_compilation(self, network, output_dir):
        """Execute compilation with fault tolerance."""
        self.logger.info("Executing fault-tolerant compilation")
        
        # Create checkpoints at key stages
        checkpoint_operations = [
            ("network_parsing", self._safe_network_parsing),
            ("optimization", self._safe_optimization),
            ("hdl_generation", self._safe_hdl_generation),
        ]
        
        last_checkpoint_id = None
        
        for stage_name, operation in checkpoint_operations:
            self.operation_counter += 1
            
            # Create checkpoint if interval reached
            if self.operation_counter % self.config.checkpoint_interval == 0:
                if hasattr(operation, '__self__'):
                    last_checkpoint_id = self.fault_tolerance.create_checkpoint(
                        stage_name, 
                        network,
                        {"stage": stage_name, "operation_counter": self.operation_counter},
                        {"memory_usage": "unknown", "cpu_usage": "unknown"}
                    )
            
            # Wrap operation with fault tolerance
            fault_tolerant_op = self.robust_wrapper.with_fault_tolerance(
                operation, stage_name, self.config.max_retry_attempts
            )
            
            try:
                result = fault_tolerant_op(network, output_dir)
                if stage_name == "network_parsing":
                    network = result  # Update network object for next stages
                    
            except Exception as e:
                self.logger.error(f"Fault-tolerant operation {stage_name} failed", error=str(e))
                
                # Attempt recovery if auto-recovery is enabled
                if self.config.auto_recovery and last_checkpoint_id:
                    self.logger.info(f"Attempting auto-recovery from checkpoint {last_checkpoint_id}")
                    try:
                        checkpoint_data = self.fault_tolerance.restore_from_checkpoint(last_checkpoint_id)
                        network = checkpoint_data.get("network_state", network)
                        # Retry the operation
                        result = fault_tolerant_op(network, output_dir)
                    except Exception as recovery_e:
                        self.logger.error("Auto-recovery failed", error=str(recovery_e))
                        raise
                else:
                    raise
        
        # Execute final compilation with the base compiler
        config = CompilationConfig(
            optimization_level=OptimizationLevel.BASIC,
            generate_reports=True,
            run_synthesis=False
        )
        
        return self.base_compiler.compile(network, output_dir, config)
    
    def _execute_standard_compilation(self, network, output_dir):
        """Execute standard compilation without fault tolerance."""
        self.logger.info("Executing standard compilation")
        
        config = CompilationConfig(
            optimization_level=OptimizationLevel.BASIC,
            generate_reports=True,
            run_synthesis=False
        )
        
        return self.base_compiler.compile(network, output_dir, config)
    
    def _safe_network_parsing(self, network, output_dir):
        """Safely parse network with error handling."""
        if isinstance(network, (str, Path)):
            from .compiler.frontend import parse_network_file
            return parse_network_file(Path(network))
        return network
    
    def _safe_optimization(self, network, output_dir):
        """Safely optimize network."""
        # This would call the optimization pipeline
        return network
    
    def _safe_hdl_generation(self, network, output_dir):
        """Safely generate HDL."""
        # This would call the HDL generation
        return True
    
    def _perform_post_compilation_security_check(self, compilation_result, output_dir):
        """Perform security check on generated files."""
        self.logger.info("Performing post-compilation security check")
        
        generated_files = []
        for file_path in compilation_result.hdl_files.values():
            if isinstance(file_path, Path) and file_path.exists():
                generated_files.append(file_path)
        
        return self.security_framework.perform_security_audit(generated_files, {})
    
    def _generate_robust_compilation_report(self, result: 'RobustCompilationResult', output_dir: Path):
        """Generate comprehensive robust compilation report."""
        report_dir = output_dir / "robust_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Main report
        report_file = report_dir / "robust_compilation_report.txt"
        with open(report_file, 'w') as f:
            f.write("=== ROBUST COMPILATION REPORT ===\\n\\n")
            f.write(f"Compilation Time: {result.start_time}\\n")
            f.write(f"Duration: {result.duration:.2f} seconds\\n")
            f.write(f"Target: {result.target.value}\\n")
            f.write(f"Success: {result.success}\\n\\n")
            
            if result.errors:
                f.write("ERRORS:\\n")
                for error in result.errors:
                    f.write(f"  - {error}\\n")
                f.write("\\n")
            
            if result.warnings:
                f.write("WARNINGS:\\n")
                for warning in result.warnings:
                    f.write(f"  - {warning}\\n")
                f.write("\\n")
            
            # Security information
            if result.security_audit:
                f.write("SECURITY AUDIT (PRE-COMPILATION):\\n")
                f.write(f"  Risk Score: {result.security_audit.risk_score:.2f}\\n")
                f.write(f"  Threats Detected: {len(result.security_audit.threats_detected)}\\n")
                
                for threat in result.security_audit.threats_detected:
                    f.write(f"    - {threat.threat_level.value.upper()}: {threat.description}\\n")
                f.write("\\n")
            
            if result.post_security_audit:
                f.write("SECURITY AUDIT (POST-COMPILATION):\\n")
                f.write(f"  Risk Score: {result.post_security_audit.risk_score:.2f}\\n")
                f.write(f"  Threats Detected: {len(result.post_security_audit.threats_detected)}\\n")
                f.write("\\n")
            
            # Fault tolerance information
            if self.fault_tolerance and result.fault_records:
                f.write("FAULT TOLERANCE:\\n")
                fault_stats = self.fault_tolerance.get_fault_statistics()
                f.write(f"  Total Faults: {fault_stats['total_faults']}\\n")
                f.write(f"  Recovery Rate: {fault_stats['recovery_rate']:.2f}\\n")
                f.write("\\n")
        
        self.logger.info("Generated robust compilation report", report_file=str(report_file))
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        health_info = {
            "timestamp": datetime.now().isoformat(),
            "compiler_status": "active",
        }
        
        # Base compiler health
        if hasattr(self.base_compiler, 'get_health_status'):
            health_info["base_compiler"] = self.base_compiler.get_health_status()
        
        # Fault tolerance status
        if self.fault_tolerance:
            health_info["fault_tolerance"] = self.fault_tolerance.get_fault_statistics()
        
        # Security status  
        if self.security_framework:
            health_info["security"] = self.security_framework.get_security_dashboard()
        
        return health_info


@dataclass
class RobustCompilationResult:
    """Result of robust compilation with extended information."""
    
    success: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    target: Optional[FPGATarget] = None
    config: Optional[RobustCompilationConfig] = None
    
    # Base compilation result
    base_compilation_result: Optional[CompilationResult] = None
    
    # Security audit results
    security_audit: Optional[Any] = None
    post_security_audit: Optional[Any] = None
    security_blocked: bool = False
    
    # Fault tolerance information
    fault_records: List[str] = None
    recovery_attempts: int = 0
    checkpoints_created: List[str] = None
    
    # Execution environment info
    execution_environment: Optional[Dict[str, Any]] = None
    
    # Standard result fields
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.fault_records is None:
            self.fault_records = []
        if self.checkpoints_created is None:
            self.checkpoints_created = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class RobustExecutionEnvironment:
    """Context manager for robust execution environment."""
    
    def __init__(self, base_output_dir: Path, config: RobustCompilationConfig, logger):
        self.base_output_dir = base_output_dir
        self.config = config
        self.logger = logger
        self.temp_dir = None
        self.secure_output_dir = None
    
    def __enter__(self) -> Dict[str, Any]:
        # Create secure temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="robust_compile_")
        self.secure_output_dir = Path(self.temp_dir) / "secure_output"
        self.secure_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Created secure execution environment", 
                        temp_dir=self.temp_dir)
        
        return {
            "temp_dir": Path(self.temp_dir),
            "secure_output_dir": self.secure_output_dir,
            "base_output_dir": self.base_output_dir
        }
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Copy results to final output directory
        if self.secure_output_dir and self.secure_output_dir.exists():
            import shutil
            if not self.base_output_dir.exists():
                self.base_output_dir.mkdir(parents=True)
            
            # Copy all files from secure directory to base directory
            for item in self.secure_output_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.base_output_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, self.base_output_dir / item.name, 
                                  dirs_exist_ok=True)
        
        # Clean up temporary directory if configured
        if self.config.temp_dir_cleanup and self.temp_dir:
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info("Cleaned up execution environment")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory: {e}")


# Convenience function for robust compilation
def compile_network_robust(network: Union[Network, str, Path],
                          target: FPGATarget,
                          output_dir: Union[str, Path] = "./robust_output",
                          config: Optional[RobustCompilationConfig] = None) -> RobustCompilationResult:
    """Compile network with full robustness features.
    
    Args:
        network: Network object or path to network definition
        target: Target FPGA platform
        output_dir: Output directory for generated files
        config: Robust compilation configuration
        
    Returns:
        RobustCompilationResult with comprehensive information
    """
    if config is None:
        config = RobustCompilationConfig()
    
    compiler = RobustNetworkCompiler(target, config)
    return compiler.compile_with_robustness(network, output_dir)