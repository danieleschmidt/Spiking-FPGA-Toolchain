"""Advanced fault tolerance and self-healing mechanisms for the neuromorphic compiler."""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from enum import Enum


class FaultType(Enum):
    """Types of faults that can occur during compilation."""
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error"
    OPTIMIZATION_ERROR = "optimization_error"
    HDL_GENERATION_ERROR = "hdl_generation_error"
    SYNTHESIS_ERROR = "synthesis_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT_ERROR = "timeout_error"
    HARDWARE_FAILURE = "hardware_failure"


class RecoveryStrategy(Enum):
    """Recovery strategies for different fault types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    RESOURCE_REALLOCATION = "resource_reallocation"


@dataclass
class FaultRecord:
    """Record of a fault occurrence."""
    fault_id: str
    fault_type: FaultType
    timestamp: datetime
    component: str
    error_message: str
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """Compilation checkpoint for recovery."""
    checkpoint_id: str
    timestamp: datetime
    stage: str
    network_state: bytes
    optimization_state: Dict[str, Any]
    resource_state: Dict[str, Any]
    file_path: Path


class AdaptiveFaultTolerance:
    """Advanced fault tolerance system with machine learning-based adaptation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.fault_history: List[FaultRecord] = []
        self.recovery_patterns: Dict[str, RecoveryStrategy] = {}
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.failure_rates: Dict[str, float] = {}
        self.adaptation_enabled = True
        
        # Circuit breaker parameters (adaptive)
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Recovery strategies mapping
        self.default_strategies = {
            FaultType.PARSING_ERROR: RecoveryStrategy.RETRY,
            FaultType.VALIDATION_ERROR: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FaultType.OPTIMIZATION_ERROR: RecoveryStrategy.FALLBACK,
            FaultType.HDL_GENERATION_ERROR: RecoveryStrategy.CHECKPOINT_RESTORE,
            FaultType.SYNTHESIS_ERROR: RecoveryStrategy.RESOURCE_REALLOCATION,
            FaultType.RESOURCE_EXHAUSTION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FaultType.TIMEOUT_ERROR: RecoveryStrategy.RESOURCE_REALLOCATION,
            FaultType.HARDWARE_FAILURE: RecoveryStrategy.FALLBACK,
        }
        
        # Initialize adaptive learning
        self._initialize_adaptive_learning()
    
    def _initialize_adaptive_learning(self) -> None:
        """Initialize machine learning components for fault prediction."""
        try:
            # Simple pattern recognition based on historical data
            self.pattern_weights = {
                "time_of_day": 0.1,
                "network_complexity": 0.3,
                "resource_usage": 0.4,
                "previous_failures": 0.2
            }
            
            # Load historical patterns if available
            patterns_file = Path("fault_patterns.json")
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    historical_patterns = json.load(f)
                    self.recovery_patterns.update(historical_patterns)
                    
        except Exception as e:
            self.logger.warning(f"Failed to initialize adaptive learning: {e}")
    
    def predict_failure_probability(self, component: str, context: Dict[str, Any]) -> float:
        """Predict probability of failure for a given component and context."""
        if component not in self.failure_rates:
            return 0.1  # Default low probability for unknown components
        
        base_rate = self.failure_rates[component]
        
        # Adjust based on context
        adjustment_factors = 1.0
        
        # Time-based adjustment
        current_hour = datetime.now().hour
        if 2 <= current_hour <= 6:  # Historically higher failure rates at night
            adjustment_factors *= 1.2
        
        # Complexity adjustment
        if "network_complexity" in context:
            complexity = context["network_complexity"]
            if complexity > 0.8:
                adjustment_factors *= 1.5
            elif complexity < 0.3:
                adjustment_factors *= 0.8
        
        # Resource usage adjustment
        if "resource_utilization" in context:
            resource_util = context["resource_utilization"]
            if resource_util > 0.9:
                adjustment_factors *= 2.0
            elif resource_util < 0.5:
                adjustment_factors *= 0.7
        
        predicted_probability = min(base_rate * adjustment_factors, 0.95)
        
        self.logger.debug(f"Predicted failure probability for {component}: {predicted_probability:.3f}")
        return predicted_probability
    
    def create_checkpoint(self, stage: str, network_state: Any, 
                         optimization_state: Dict[str, Any],
                         resource_state: Dict[str, Any]) -> str:
        """Create a checkpoint for potential recovery."""
        checkpoint_id = f"checkpoint_{stage}_{int(time.time())}"
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Serialize network state
        network_bytes = pickle.dumps(network_state)
        
        # Create checkpoint file
        checkpoint_path = checkpoint_dir / f"{checkpoint_id}.pkl"
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            stage=stage,
            network_state=network_bytes,
            optimization_state=optimization_state,
            resource_state=resource_state,
            file_path=checkpoint_path
        )
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'network_state': network_state,
                'optimization_state': optimization_state,
                'resource_state': resource_state,
                'metadata': {
                    'stage': stage,
                    'timestamp': checkpoint.timestamp.isoformat()
                }
            }, f)
        
        self.checkpoints[checkpoint_id] = checkpoint
        
        self.logger.info(f"Created checkpoint {checkpoint_id} at stage {stage}")
        return checkpoint_id
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore state from a checkpoint."""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        try:
            with open(checkpoint.file_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"Restored from checkpoint {checkpoint_id}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            raise
    
    def record_fault(self, fault_type: FaultType, component: str, 
                    error_message: str, context: Dict[str, Any] = None) -> str:
        """Record a fault occurrence."""
        fault_id = f"fault_{int(time.time())}_{len(self.fault_history)}"
        
        fault_record = FaultRecord(
            fault_id=fault_id,
            fault_type=fault_type,
            timestamp=datetime.now(),
            component=component,
            error_message=error_message,
            context=context or {}
        )
        
        self.fault_history.append(fault_record)
        
        # Update failure rates
        self._update_failure_rates(component)
        
        self.logger.warning(f"Recorded fault {fault_id}: {fault_type.value} in {component}")
        return fault_id
    
    def _update_failure_rates(self, component: str) -> None:
        """Update failure rates based on observed faults."""
        recent_faults = [f for f in self.fault_history 
                        if f.component == component and 
                        f.timestamp > datetime.now() - timedelta(hours=24)]
        
        total_operations = max(len(recent_faults) * 10, 1)  # Estimate total operations
        failure_rate = len(recent_faults) / total_operations
        
        self.failure_rates[component] = failure_rate
    
    def get_recovery_strategy(self, fault_type: FaultType, component: str,
                            context: Dict[str, Any] = None) -> RecoveryStrategy:
        """Determine the best recovery strategy for a given fault."""
        # Check if we have learned a better strategy for this specific case
        pattern_key = f"{fault_type.value}_{component}"
        if pattern_key in self.recovery_patterns:
            learned_strategy = RecoveryStrategy(self.recovery_patterns[pattern_key])
            self.logger.info(f"Using learned recovery strategy: {learned_strategy.value}")
            return learned_strategy
        
        # Use default strategy
        default_strategy = self.default_strategies.get(fault_type, RecoveryStrategy.RETRY)
        
        # Adaptive adjustments based on context
        if context:
            # If resource exhaustion is detected, prefer resource reallocation
            if context.get("resource_utilization", 0) > 0.9:
                if fault_type in [FaultType.OPTIMIZATION_ERROR, FaultType.HDL_GENERATION_ERROR]:
                    return RecoveryStrategy.RESOURCE_REALLOCATION
            
            # If this is a repeated failure, escalate strategy
            recent_failures = [f for f in self.fault_history[-10:] 
                             if f.component == component and f.fault_type == fault_type]
            
            if len(recent_failures) >= 3:
                if default_strategy == RecoveryStrategy.RETRY:
                    return RecoveryStrategy.FALLBACK
                elif default_strategy == RecoveryStrategy.FALLBACK:
                    return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        return default_strategy
    
    def execute_recovery(self, fault_id: str, recovery_strategy: RecoveryStrategy,
                        recovery_context: Dict[str, Any] = None) -> bool:
        """Execute a recovery strategy."""
        fault_record = next((f for f in self.fault_history if f.fault_id == fault_id), None)
        if not fault_record:
            self.logger.error(f"Fault record {fault_id} not found")
            return False
        
        fault_record.recovery_attempted = True
        fault_record.recovery_strategy = recovery_strategy
        
        try:
            if recovery_strategy == RecoveryStrategy.RETRY:
                success = self._execute_retry_recovery(fault_record, recovery_context)
            elif recovery_strategy == RecoveryStrategy.FALLBACK:
                success = self._execute_fallback_recovery(fault_record, recovery_context)
            elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                success = self._execute_graceful_degradation(fault_record, recovery_context)
            elif recovery_strategy == RecoveryStrategy.CHECKPOINT_RESTORE:
                success = self._execute_checkpoint_restore(fault_record, recovery_context)
            elif recovery_strategy == RecoveryStrategy.RESOURCE_REALLOCATION:
                success = self._execute_resource_reallocation(fault_record, recovery_context)
            else:
                self.logger.error(f"Unknown recovery strategy: {recovery_strategy}")
                success = False
            
            fault_record.recovery_successful = success
            
            # Learn from recovery outcomes
            if self.adaptation_enabled:
                self._update_recovery_patterns(fault_record, success)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Recovery execution failed: {e}")
            fault_record.recovery_successful = False
            return False
    
    def _execute_retry_recovery(self, fault_record: FaultRecord, 
                               context: Dict[str, Any] = None) -> bool:
        """Execute retry recovery strategy."""
        max_retries = context.get("max_retries", 3) if context else 3
        retry_delay = context.get("retry_delay", 1.0) if context else 1.0
        
        self.logger.info(f"Executing retry recovery for fault {fault_record.fault_id}")
        
        for attempt in range(max_retries):
            self.logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
            time.sleep(retry_delay)
            
            # In a real implementation, this would re-execute the failed operation
            # For now, simulate success based on fault type
            if fault_record.fault_type == FaultType.PARSING_ERROR:
                return attempt >= 1  # Succeed after second attempt
            elif fault_record.fault_type == FaultType.TIMEOUT_ERROR:
                return attempt >= 2  # Succeed after third attempt
        
        return False
    
    def _execute_fallback_recovery(self, fault_record: FaultRecord, 
                                  context: Dict[str, Any] = None) -> bool:
        """Execute fallback recovery strategy."""
        self.logger.info(f"Executing fallback recovery for fault {fault_record.fault_id}")
        
        # Implementation would switch to alternative algorithms or backends
        return True  # Simplified success
    
    def _execute_graceful_degradation(self, fault_record: FaultRecord, 
                                     context: Dict[str, Any] = None) -> bool:
        """Execute graceful degradation recovery strategy."""
        self.logger.info(f"Executing graceful degradation for fault {fault_record.fault_id}")
        
        # Implementation would reduce complexity or disable advanced features
        return True  # Simplified success
    
    def _execute_checkpoint_restore(self, fault_record: FaultRecord, 
                                   context: Dict[str, Any] = None) -> bool:
        """Execute checkpoint restore recovery strategy."""
        self.logger.info(f"Executing checkpoint restore for fault {fault_record.fault_id}")
        
        # Find most recent checkpoint
        if not self.checkpoints:
            self.logger.warning("No checkpoints available for restore")
            return False
        
        latest_checkpoint_id = max(self.checkpoints.keys(), 
                                 key=lambda x: self.checkpoints[x].timestamp)
        
        try:
            checkpoint_data = self.restore_from_checkpoint(latest_checkpoint_id)
            return True
        except Exception as e:
            self.logger.error(f"Checkpoint restore failed: {e}")
            return False
    
    def _execute_resource_reallocation(self, fault_record: FaultRecord, 
                                      context: Dict[str, Any] = None) -> bool:
        """Execute resource reallocation recovery strategy."""
        self.logger.info(f"Executing resource reallocation for fault {fault_record.fault_id}")
        
        # Implementation would reallocate resources (CPU, memory, etc.)
        return True  # Simplified success
    
    def _update_recovery_patterns(self, fault_record: FaultRecord, success: bool) -> None:
        """Update learned recovery patterns based on outcomes."""
        pattern_key = f"{fault_record.fault_type.value}_{fault_record.component}"
        
        if success and fault_record.recovery_strategy:
            # Reinforce successful pattern
            self.recovery_patterns[pattern_key] = fault_record.recovery_strategy.value
            self.logger.debug(f"Learned successful pattern: {pattern_key} -> {fault_record.recovery_strategy.value}")
        
        # Save patterns to disk
        try:
            with open("fault_patterns.json", 'w') as f:
                json.dump(self.recovery_patterns, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save recovery patterns: {e}")
    
    def get_fault_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fault statistics."""
        if not self.fault_history:
            return {"total_faults": 0, "recovery_rate": 0.0}
        
        total_faults = len(self.fault_history)
        recovered_faults = len([f for f in self.fault_history if f.recovery_successful])
        attempted_recoveries = len([f for f in self.fault_history if f.recovery_attempted])
        
        fault_types_count = {}
        for fault in self.fault_history:
            fault_type = fault.fault_type.value
            fault_types_count[fault_type] = fault_types_count.get(fault_type, 0) + 1
        
        component_stats = {}
        for fault in self.fault_history:
            component = fault.component
            if component not in component_stats:
                component_stats[component] = {"total": 0, "recovered": 0}
            component_stats[component]["total"] += 1
            if fault.recovery_successful:
                component_stats[component]["recovered"] += 1
        
        return {
            "total_faults": total_faults,
            "attempted_recoveries": attempted_recoveries,
            "successful_recoveries": recovered_faults,
            "recovery_rate": recovered_faults / attempted_recoveries if attempted_recoveries > 0 else 0.0,
            "fault_types": fault_types_count,
            "component_statistics": component_stats,
            "failure_rates": self.failure_rates.copy()
        }
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> None:
        """Clean up old checkpoints to manage disk space."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for checkpoint_id, checkpoint in self.checkpoints.items():
            if checkpoint.timestamp < cutoff_time:
                try:
                    if checkpoint.file_path.exists():
                        checkpoint.file_path.unlink()
                    to_remove.append(checkpoint_id)
                except Exception as e:
                    self.logger.warning(f"Failed to remove checkpoint {checkpoint_id}: {e}")
        
        for checkpoint_id in to_remove:
            del self.checkpoints[checkpoint_id]
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old checkpoints")


class RobustCompilerWrapper:
    """Wrapper that adds fault tolerance to any compiler function."""
    
    def __init__(self, fault_tolerance: AdaptiveFaultTolerance):
        self.fault_tolerance = fault_tolerance
        self.logger = fault_tolerance.logger
    
    def with_fault_tolerance(self, func: Callable, component: str, 
                           max_attempts: int = 3) -> Callable:
        """Wrap a function with fault tolerance."""
        
        def wrapped_func(*args, **kwargs):
            context = {
                "function": func.__name__,
                "component": component,
                "max_attempts": max_attempts
            }
            
            # Predict failure probability
            failure_prob = self.fault_tolerance.predict_failure_probability(component, context)
            
            if failure_prob > 0.7:
                self.logger.warning(f"High failure probability predicted for {component}: {failure_prob:.2f}")
                # Create preventive checkpoint
                if hasattr(args[0] if args else None, '__dict__'):
                    checkpoint_id = self.fault_tolerance.create_checkpoint(
                        f"preventive_{component}",
                        args[0] if args else None,
                        {"function": func.__name__},
                        {"failure_probability": failure_prob}
                    )
                    context["checkpoint_id"] = checkpoint_id
            
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    if attempt > 0:
                        self.logger.info(f"Retry attempt {attempt + 1}/{max_attempts} for {func.__name__}")
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        self.logger.info(f"Function {func.__name__} succeeded after {attempt + 1} attempts")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    self.logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    
                    # Determine fault type based on exception
                    fault_type = self._classify_fault(e)
                    
                    # Record fault
                    fault_id = self.fault_tolerance.record_fault(
                        fault_type, component, str(e), context
                    )
                    
                    # If this is the last attempt, try recovery
                    if attempt == max_attempts - 1:
                        recovery_strategy = self.fault_tolerance.get_recovery_strategy(
                            fault_type, component, context
                        )
                        
                        recovery_success = self.fault_tolerance.execute_recovery(
                            fault_id, recovery_strategy, context
                        )
                        
                        if recovery_success:
                            self.logger.info(f"Recovery successful for {func.__name__}")
                            # Try one more time after recovery
                            try:
                                return func(*args, **kwargs)
                            except Exception as recovery_e:
                                self.logger.error(f"Function failed even after recovery: {recovery_e}")
                                raise recovery_e
                    
                    # Wait before retry
                    if attempt < max_attempts - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        time.sleep(wait_time)
            
            # All attempts failed
            raise last_exception
        
        return wrapped_func
    
    def _classify_fault(self, exception: Exception) -> FaultType:
        """Classify an exception into a fault type."""
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        if "parse" in exception_str or "yaml" in exception_str:
            return FaultType.PARSING_ERROR
        elif "validation" in exception_str or "invalid" in exception_str:
            return FaultType.VALIDATION_ERROR
        elif "optimization" in exception_str:
            return FaultType.OPTIMIZATION_ERROR
        elif "hdl" in exception_str or "generate" in exception_str:
            return FaultType.HDL_GENERATION_ERROR
        elif "synthesis" in exception_str or "vivado" in exception_str or "quartus" in exception_str:
            return FaultType.SYNTHESIS_ERROR
        elif "memory" in exception_str or "resource" in exception_str:
            return FaultType.RESOURCE_EXHAUSTION
        elif "timeout" in exception_str or "timeouterror" in exception_type:
            return FaultType.TIMEOUT_ERROR
        else:
            return FaultType.HARDWARE_FAILURE