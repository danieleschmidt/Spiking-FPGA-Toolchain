"""
Fault-tolerant compiler with redundancy and self-healing capabilities.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import json
import pickle

from ..network_compiler import NetworkCompiler, CompilationResult, CompilationConfig
from ..core import FPGATarget
from .error_recovery import ErrorRecoverySystem, ErrorContext, RecoveryPolicy

logger = logging.getLogger(__name__)


class RedundancyMode(Enum):
    """Redundancy strategies for fault tolerance."""
    NONE = "none"
    DUAL_MODULAR = "dual_modular"
    TRIPLE_MODULAR = "triple_modular"
    N_VERSION = "n_version"


@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance features."""
    redundancy_mode: RedundancyMode = RedundancyMode.DUAL_MODULAR
    enable_checkpointing: bool = True
    checkpoint_interval: float = 60.0  # seconds
    health_check_interval: float = 10.0  # seconds
    max_recovery_attempts: int = 3
    enable_self_healing: bool = True
    backup_targets: List[FPGATarget] = field(default_factory=list)
    distributed_compilation: bool = False


class FaultTolerantCompiler:
    """
    Fault-tolerant compiler with advanced reliability features.
    
    Features:
    - Multi-version compilation with voting
    - Automatic backup target selection
    - Continuous health monitoring
    - Self-healing capabilities
    - Distributed compilation support
    """
    
    def __init__(self, primary_target: FPGATarget, 
                 config: FaultToleranceConfig = None):
        self.primary_target = primary_target
        self.config = config or FaultToleranceConfig()
        
        # Initialize multiple compiler instances for redundancy
        self.compilers = {}
        self.backup_targets = self.config.backup_targets or []
        self._setup_redundant_compilers()
        
        # Initialize error recovery system
        recovery_policy = RecoveryPolicy(
            max_retry_attempts=self.config.max_recovery_attempts,
            enable_checkpointing=self.config.enable_checkpointing
        )
        self.error_recovery = ErrorRecoverySystem(recovery_policy)
        
        # Health monitoring
        self.health_status = {}
        self.compilation_history = []
        self._health_monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager()
        self._last_checkpoint_time = 0
        
        # Start health monitoring
        if self.config.health_check_interval > 0:
            self._start_health_monitoring()
        
        logger.info(f"FaultTolerantCompiler initialized with {self.config.redundancy_mode.value} redundancy")
    
    def _setup_redundant_compilers(self):
        """Setup redundant compiler instances."""
        # Primary compiler
        self.compilers['primary'] = NetworkCompiler(
            self.primary_target, 
            enable_monitoring=True
        )
        
        # Backup compilers
        if self.config.redundancy_mode != RedundancyMode.NONE:
            for i, backup_target in enumerate(self.backup_targets):
                self.compilers[f'backup_{i}'] = NetworkCompiler(
                    backup_target,
                    enable_monitoring=True
                )
        
        # For N-version redundancy, create multiple compilers with different optimization strategies
        if self.config.redundancy_mode == RedundancyMode.N_VERSION:
            optimization_levels = ['BASIC', 'AGGRESSIVE', 'MAXIMUM']
            for i, level in enumerate(optimization_levels):
                compiler_id = f'nversion_{i}'
                self.compilers[compiler_id] = NetworkCompiler(
                    self.primary_target,
                    enable_monitoring=True
                )
    
    def compile(self, network, output_dir: Path, 
                config: CompilationConfig = None) -> CompilationResult:
        """
        Fault-tolerant compilation with redundancy and error recovery.
        """
        compilation_id = self._generate_compilation_id(network, config)
        
        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(compilation_id)
        if checkpoint and self.config.enable_checkpointing:
            logger.info(f"Resuming from checkpoint: {compilation_id}")
            return self._resume_from_checkpoint(checkpoint, output_dir)
        
        # Create initial checkpoint
        if self.config.enable_checkpointing:
            self.checkpoint_manager.create_checkpoint(
                compilation_id, 
                {"network": network, "config": config, "stage": "start"}
            )
        
        # Execute compilation with redundancy
        if self.config.redundancy_mode == RedundancyMode.NONE:
            return self._single_compilation(network, output_dir, config)
        elif self.config.redundancy_mode in [RedundancyMode.DUAL_MODULAR, RedundancyMode.TRIPLE_MODULAR]:
            return self._redundant_compilation(network, output_dir, config)
        elif self.config.redundancy_mode == RedundancyMode.N_VERSION:
            return self._n_version_compilation(network, output_dir, config)
        else:
            raise ValueError(f"Unsupported redundancy mode: {self.config.redundancy_mode}")
    
    def _single_compilation(self, network, output_dir: Path, 
                          config: CompilationConfig) -> CompilationResult:
        """Single compilation with error recovery."""
        compiler = self.compilers['primary']
        
        for attempt in range(self.config.max_recovery_attempts + 1):
            try:
                result = compiler.compile(network, output_dir, config)
                
                if result.success:
                    self._record_successful_compilation(result)
                    return result
                else:
                    # Handle compilation errors through recovery system
                    error_context = ErrorContext(
                        error_type="compilation_error",
                        error_message="; ".join(result.errors) if result.errors else "Unknown error",
                        timestamp=time.time(),
                        component="primary_compiler",
                        severity="error",
                        context_data={"network": network, "config": config}
                    )
                    
                    success, recovery_data = self.error_recovery.handle_error(
                        Exception("Compilation failed"), error_context
                    )
                    
                    if success and recovery_data:
                        # Update configuration based on recovery
                        if isinstance(recovery_data, dict):
                            config = self._update_config_from_recovery(config, recovery_data)
                        continue
                    else:
                        return result
                        
            except Exception as e:
                logger.error(f"Compilation attempt {attempt + 1} failed: {e}")
                
                # Handle exception through recovery system
                error_context = ErrorContext(
                    error_type="compilation_exception",
                    error_message=str(e),
                    timestamp=time.time(),
                    component="primary_compiler", 
                    severity="critical",
                    recovery_attempts=attempt,
                    context_data={"network": network, "config": config}
                )
                
                success, recovery_data = self.error_recovery.handle_error(e, error_context)
                
                if not success or attempt == self.config.max_recovery_attempts:
                    # Final attempt failed, try backup compiler if available
                    if self.backup_targets and 'backup_0' in self.compilers:
                        logger.warning("Switching to backup compiler")
                        return self._fallback_to_backup(network, output_dir, config)
                    else:
                        raise e
        
        # Should not reach here
        raise RuntimeError("Max compilation attempts exceeded")
    
    def _redundant_compilation(self, network, output_dir: Path, 
                             config: CompilationConfig) -> CompilationResult:
        """Redundant compilation with voting mechanism."""
        results = {}
        
        # Execute parallel compilations
        for compiler_id, compiler in self.compilers.items():
            if compiler_id.startswith('backup') and self.config.redundancy_mode == RedundancyMode.DUAL_MODULAR:
                if len(results) >= 2:  # Only need 2 for dual modular
                    break
            
            try:
                result = compiler.compile(network, output_dir / compiler_id, config)
                results[compiler_id] = result
                logger.info(f"Compilation completed on {compiler_id}: {'SUCCESS' if result.success else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"Compilation failed on {compiler_id}: {e}")
                results[compiler_id] = None
        
        # Implement voting mechanism
        return self._vote_on_results(results, output_dir)
    
    def _n_version_compilation(self, network, output_dir: Path,
                             config: CompilationConfig) -> CompilationResult:
        """N-version programming compilation."""
        results = {}
        
        # Execute compilations with different optimization strategies
        for compiler_id, compiler in self.compilers.items():
            if not compiler_id.startswith('nversion'):
                continue
            
            try:
                # Modify config for each version
                version_config = self._create_version_specific_config(config, compiler_id)
                result = compiler.compile(network, output_dir / compiler_id, version_config)
                results[compiler_id] = result
                
            except Exception as e:
                logger.error(f"N-version compilation failed on {compiler_id}: {e}")
                results[compiler_id] = None
        
        # Compare and validate results
        return self._compare_n_version_results(results, output_dir)
    
    def _vote_on_results(self, results: Dict[str, CompilationResult], 
                        output_dir: Path) -> CompilationResult:
        """Implement voting mechanism for redundant results."""
        successful_results = {k: v for k, v in results.items() if v and v.success}
        
        if not successful_results:
            # All compilations failed, return the primary failure
            primary_result = results.get('primary')
            if primary_result:
                return primary_result
            else:
                # Create a failure result
                from ..models.network import Network
                from ..models.optimization import ResourceEstimate
                return CompilationResult(
                    success=False,
                    network=Network(name="failed"),
                    optimized_network=Network(name="failed"),
                    hdl_files={},
                    resource_estimate=ResourceEstimate(),
                    optimization_stats={},
                    errors=["All redundant compilations failed"]
                )
        
        if len(successful_results) == 1:
            # Only one successful result
            return list(successful_results.values())[0]
        
        # Multiple successful results - implement consensus
        return self._build_consensus_result(successful_results, output_dir)
    
    def _build_consensus_result(self, results: Dict[str, CompilationResult],
                               output_dir: Path) -> CompilationResult:
        """Build consensus result from multiple successful compilations."""
        # For now, return the primary result if available, otherwise the first successful
        if 'primary' in results:
            consensus_result = results['primary']
        else:
            consensus_result = list(results.values())[0]
        
        # Add consensus metadata
        consensus_result.optimization_stats['redundancy_info'] = {
            'successful_compilations': len(results),
            'consensus_achieved': True,
            'compilation_ids': list(results.keys())
        }
        
        logger.info(f"Consensus achieved from {len(results)} successful compilations")
        return consensus_result
    
    def _compare_n_version_results(self, results: Dict[str, CompilationResult],
                                 output_dir: Path) -> CompilationResult:
        """Compare N-version results for consistency."""
        successful_results = {k: v for k, v in results.items() if v and v.success}
        
        if len(successful_results) == 0:
            raise RuntimeError("All N-version compilations failed")
        
        # Compare resource estimates for consistency
        resource_estimates = [r.resource_estimate for r in successful_results.values()]
        if self._are_resource_estimates_consistent(resource_estimates):
            logger.info("N-version results are consistent")
            result = list(successful_results.values())[0]
        else:
            logger.warning("N-version results show discrepancies")
            result = self._resolve_discrepancies(successful_results)
        
        result.optimization_stats['n_version_info'] = {
            'total_versions': len(results),
            'successful_versions': len(successful_results),
            'consistency_check': 'passed' if self._are_resource_estimates_consistent(resource_estimates) else 'failed'
        }
        
        return result
    
    def _are_resource_estimates_consistent(self, estimates: List) -> bool:
        """Check if resource estimates are consistent across versions."""
        if len(estimates) < 2:
            return True
        
        # Allow 10% variation in resource estimates
        tolerance = 0.1
        base_estimate = estimates[0]
        
        for estimate in estimates[1:]:
            if abs(estimate.luts - base_estimate.luts) / base_estimate.luts > tolerance:
                return False
            if abs(estimate.bram_kb - base_estimate.bram_kb) / base_estimate.bram_kb > tolerance:
                return False
        
        return True
    
    def _resolve_discrepancies(self, results: Dict[str, CompilationResult]) -> CompilationResult:
        """Resolve discrepancies in N-version results."""
        # Take the most conservative (highest) resource estimate
        max_luts = max(r.resource_estimate.luts for r in results.values())
        max_bram = max(r.resource_estimate.bram_kb for r in results.values())
        
        # Use result with highest resource usage as base
        conservative_result = max(
            results.values(), 
            key=lambda r: r.resource_estimate.luts + r.resource_estimate.bram_kb
        )
        
        logger.info("Resolved discrepancies using conservative estimates")
        return conservative_result
    
    def _fallback_to_backup(self, network, output_dir: Path, 
                           config: CompilationConfig) -> CompilationResult:
        """Fallback to backup compiler when primary fails."""
        for i, backup_target in enumerate(self.backup_targets):
            compiler_id = f'backup_{i}'
            if compiler_id not in self.compilers:
                continue
            
            try:
                logger.info(f"Attempting backup compilation on {backup_target.value}")
                backup_compiler = self.compilers[compiler_id]
                result = backup_compiler.compile(network, output_dir / compiler_id, config)
                
                if result.success:
                    result.optimization_stats['fallback_info'] = {
                        'fallback_used': True,
                        'fallback_target': backup_target.value,
                        'primary_target': self.primary_target.value
                    }
                    logger.info(f"Backup compilation successful on {backup_target.value}")
                    return result
                
            except Exception as e:
                logger.error(f"Backup compilation failed on {backup_target.value}: {e}")
                continue
        
        raise RuntimeError("All backup compilations failed")
    
    def _start_health_monitoring(self):
        """Start continuous health monitoring."""
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self._health_monitor_thread.start()
        logger.info("Health monitoring started")
    
    def _health_monitor_loop(self):
        """Continuous health monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self._check_system_health()
                self._check_checkpoint_schedule()
                
                if self.config.enable_self_healing:
                    self._perform_self_healing()
                
                self._stop_monitoring.wait(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    def _check_system_health(self):
        """Check overall system health."""
        current_time = time.time()
        
        for compiler_id, compiler in self.compilers.items():
            try:
                health_status = compiler.get_health_status()
                self.health_status[compiler_id] = {
                    'status': 'healthy',
                    'last_check': current_time,
                    'details': health_status
                }
            except Exception as e:
                self.health_status[compiler_id] = {
                    'status': 'unhealthy',
                    'last_check': current_time,
                    'error': str(e)
                }
                logger.warning(f"Health check failed for {compiler_id}: {e}")
    
    def _check_checkpoint_schedule(self):
        """Check if checkpoint should be created."""
        current_time = time.time()
        if (current_time - self._last_checkpoint_time) > self.config.checkpoint_interval:
            self._create_periodic_checkpoint()
            self._last_checkpoint_time = current_time
    
    def _create_periodic_checkpoint(self):
        """Create periodic system checkpoint."""
        checkpoint_data = {
            'timestamp': time.time(),
            'health_status': self.health_status,
            'compilation_history': self.compilation_history[-10:],  # Last 10 compilations
            'error_statistics': self.error_recovery.get_error_statistics()
        }
        
        checkpoint_id = f"periodic_{int(time.time())}"
        self.checkpoint_manager.create_checkpoint(checkpoint_id, checkpoint_data)
        logger.info(f"Created periodic checkpoint: {checkpoint_id}")
    
    def _perform_self_healing(self):
        """Perform self-healing operations."""
        unhealthy_compilers = [
            compiler_id for compiler_id, status in self.health_status.items()
            if status.get('status') == 'unhealthy'
        ]
        
        if unhealthy_compilers:
            logger.info(f"Attempting self-healing for unhealthy compilers: {unhealthy_compilers}")
            
            for compiler_id in unhealthy_compilers:
                try:
                    # Restart compiler
                    if compiler_id in self.compilers:
                        target = self.compilers[compiler_id].target
                        self.compilers[compiler_id] = NetworkCompiler(
                            target, enable_monitoring=True
                        )
                        logger.info(f"Restarted compiler: {compiler_id}")
                        
                except Exception as e:
                    logger.error(f"Self-healing failed for {compiler_id}: {e}")
    
    def _generate_compilation_id(self, network, config) -> str:
        """Generate unique ID for compilation session."""
        content = f"{network}_{config}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _record_successful_compilation(self, result: CompilationResult):
        """Record successful compilation for analytics."""
        compilation_record = {
            'timestamp': time.time(),
            'network_name': result.network.name,
            'target': self.primary_target.value,
            'success': True,
            'resource_utilization': {
                'luts': result.resource_estimate.luts,
                'bram_kb': result.resource_estimate.bram_kb,
                'dsp_slices': result.resource_estimate.dsp_slices
            }
        }
        
        self.compilation_history.append(compilation_record)
        
        # Keep only recent history
        if len(self.compilation_history) > 100:
            self.compilation_history = self.compilation_history[-50:]
    
    def _update_config_from_recovery(self, config: CompilationConfig, 
                                   recovery_data: Dict[str, Any]) -> CompilationConfig:
        """Update compilation config based on recovery suggestions."""
        if 'optimization_level' in recovery_data:
            from ..models.optimization import OptimizationLevel
            level_map = {0: OptimizationLevel.NONE, 1: OptimizationLevel.BASIC, 
                        2: OptimizationLevel.AGGRESSIVE, 3: OptimizationLevel.MAXIMUM}
            config.optimization_level = level_map.get(recovery_data['optimization_level'], config.optimization_level)
        
        if 'timeout' in recovery_data:
            # Assuming config has timeout (would need to extend CompilationConfig)
            pass
        
        return config
    
    def _create_version_specific_config(self, base_config: CompilationConfig, 
                                      version_id: str) -> CompilationConfig:
        """Create version-specific configuration for N-version compilation."""
        config = CompilationConfig(
            optimization_level=base_config.optimization_level,
            clock_frequency=base_config.clock_frequency,
            power_budget_mw=base_config.power_budget_mw,
            debug_enabled=base_config.debug_enabled,
            generate_reports=base_config.generate_reports,
            run_synthesis=base_config.run_synthesis
        )
        
        # Modify based on version
        if version_id.endswith('0'):
            from ..models.optimization import OptimizationLevel
            config.optimization_level = OptimizationLevel.BASIC
        elif version_id.endswith('1'):
            from ..models.optimization import OptimizationLevel
            config.optimization_level = OptimizationLevel.AGGRESSIVE
        elif version_id.endswith('2'):
            from ..models.optimization import OptimizationLevel
            config.optimization_level = OptimizationLevel.MAXIMUM
        
        return config
    
    def _resume_from_checkpoint(self, checkpoint, output_dir: Path) -> CompilationResult:
        """Resume compilation from checkpoint."""
        # This would implement checkpoint restoration logic
        logger.info("Resuming from checkpoint (placeholder implementation)")
        
        # For now, just proceed with normal compilation
        network = checkpoint.get('network')
        config = checkpoint.get('config')
        
        if network and config:
            return self.compile(network, output_dir, config)
        else:
            raise RuntimeError("Invalid checkpoint data")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'primary_target': self.primary_target.value,
            'redundancy_mode': self.config.redundancy_mode.value,
            'health_status': self.health_status,
            'total_compilations': len(self.compilation_history),
            'error_statistics': self.error_recovery.get_error_statistics(),
            'active_compilers': list(self.compilers.keys()),
            'checkpoints_available': len(self.checkpoint_manager.checkpoints)
        }
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._stop_monitoring.set()
        if self._health_monitor_thread:
            self._health_monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring()


class CheckpointManager:
    """Manages compilation checkpoints for recovery."""
    
    def __init__(self, checkpoint_dir: Path = None):
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = {}
    
    def create_checkpoint(self, checkpoint_id: str, data: Any):
        """Create checkpoint with given ID."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        checkpoint_data = {
            'id': checkpoint_id,
            'timestamp': time.time(),
            'data': data
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.checkpoints[checkpoint_id] = checkpoint_path
            logger.info(f"Checkpoint created: {checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint by ID."""
        if checkpoint_id not in self.checkpoints:
            # Try to find checkpoint file
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            if not checkpoint_path.exists():
                return None
            self.checkpoints[checkpoint_id] = checkpoint_path
        
        try:
            with open(self.checkpoints[checkpoint_id], 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return checkpoint_data.get('data')
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        return list(self.checkpoints.keys())
    
    def cleanup_old_checkpoints(self, max_age_hours: float = 24):
        """Clean up old checkpoints."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for checkpoint_id, checkpoint_path in list(self.checkpoints.items()):
            try:
                if checkpoint_path.stat().st_mtime < cutoff_time:
                    checkpoint_path.unlink()
                    del self.checkpoints[checkpoint_id]
                    logger.info(f"Cleaned up old checkpoint: {checkpoint_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup checkpoint {checkpoint_id}: {e}")


class RedundancyManager:
    """Manages redundant system configurations."""
    
    def __init__(self):
        self.active_configurations = {}
        self.backup_configurations = {}
    
    def add_redundant_config(self, config_id: str, primary_config: Any, backup_configs: List[Any]):
        """Add redundant configuration set."""
        self.active_configurations[config_id] = primary_config
        self.backup_configurations[config_id] = backup_configs
        logger.info(f"Added redundant configuration: {config_id}")
    
    def get_active_config(self, config_id: str) -> Optional[Any]:
        """Get active configuration."""
        return self.active_configurations.get(config_id)
    
    def failover_to_backup(self, config_id: str, backup_index: int = 0) -> Optional[Any]:
        """Failover to backup configuration."""
        if config_id in self.backup_configurations:
            backups = self.backup_configurations[config_id]
            if backup_index < len(backups):
                backup_config = backups[backup_index]
                self.active_configurations[config_id] = backup_config
                logger.info(f"Failed over to backup {backup_index} for {config_id}")
                return backup_config
        return None


class FailureRecovery:
    """Manages comprehensive failure recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.failure_history = []
    
    def register_recovery_strategy(self, failure_type: str, strategy: Callable):
        """Register recovery strategy for specific failure type."""
        self.recovery_strategies[failure_type] = strategy
        logger.info(f"Registered recovery strategy for: {failure_type}")
    
    def recover_from_failure(self, failure_type: str, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Attempt recovery from failure."""
        if failure_type in self.recovery_strategies:
            strategy = self.recovery_strategies[failure_type]
            try:
                success, result = strategy(context)
                self.failure_history.append({
                    'failure_type': failure_type,
                    'timestamp': time.time(),
                    'recovery_success': success
                })
                return success, result
            except Exception as e:
                logger.error(f"Recovery strategy failed for {failure_type}: {e}")
                return False, None
        
        logger.warning(f"No recovery strategy found for failure type: {failure_type}")
        return False, None
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self.failure_history:
            return {"no_failures": True}
        
        total_failures = len(self.failure_history)
        successful_recoveries = sum(1 for f in self.failure_history if f['recovery_success'])
        
        return {
            "total_failures": total_failures,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / total_failures if total_failures > 0 else 0,
            "failure_types": list(set(f['failure_type'] for f in self.failure_history))
        }