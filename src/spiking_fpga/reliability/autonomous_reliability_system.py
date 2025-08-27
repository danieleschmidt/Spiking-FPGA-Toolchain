"""
Autonomous Reliability System for FPGA Neuromorphic Computing
==========================================================

This module implements a comprehensive autonomous reliability system that:
- Continuously monitors system health
- Predicts potential failures before they occur
- Implements automatic recovery mechanisms
- Provides self-healing capabilities
- Maintains system resilience under adverse conditions

The system adapts to different environments and learns from past failures
to improve future reliability.
"""

import asyncio
import numpy as np
import logging
import time
import json
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.monitoring import SystemMetrics
from ..utils.validation import validate_hdl_syntax
from ..models.network import Network


class ReliabilityLevel(Enum):
    """System reliability levels."""
    CRITICAL = "critical"      # Mission-critical systems
    HIGH = "high"             # Production systems
    STANDARD = "standard"     # Development systems
    EXPERIMENTAL = "experimental"  # Research systems


class FailureMode(Enum):
    """Types of failures the system can encounter."""
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_ERROR = "software_error"
    NETWORK_TIMEOUT = "network_timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMPILATION_ERROR = "compilation_error"
    SYNTHESIS_FAILURE = "synthesis_failure"
    THERMAL_ISSUE = "thermal_issue"
    POWER_ANOMALY = "power_anomaly"
    DATA_CORRUPTION = "data_corruption"
    CONFIGURATION_ERROR = "configuration_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    AUTOMATIC_RETRY = "automatic_retry"
    FAILOVER = "failover"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ROLLBACK = "rollback"
    MANUAL_INTERVENTION = "manual_intervention"
    SYSTEM_RESTART = "system_restart"
    LOAD_BALANCING = "load_balancing"
    RESOURCE_SCALING = "resource_scaling"


@dataclass
class FailureEvent:
    """Represents a system failure event."""
    timestamp: float
    failure_mode: FailureMode
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_time: Optional[float] = None
    learned_patterns: List[str] = field(default_factory=list)


@dataclass
class HealthMetrics:
    """System health metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    temperature: float
    power_consumption: float
    network_latency: float
    error_rate: float
    throughput: float
    availability: float
    reliability_score: float


class PredictiveFailureDetector:
    """
    Machine learning-based failure prediction system.
    """
    
    def __init__(self):
        self.failure_history = deque(maxlen=10000)
        self.health_history = deque(maxlen=1000)
        self.prediction_models = {}
        self.feature_extractors = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize basic predictive models
        self._initialize_prediction_models()
        
    def _initialize_prediction_models(self):
        """Initialize predictive models for different failure modes."""
        # Simple threshold-based models for start
        self.prediction_models = {
            FailureMode.THERMAL_ISSUE: {
                'type': 'threshold',
                'temperature_threshold': 85.0,
                'temperature_rate_threshold': 2.0,  # °C per minute
                'prediction_horizon': 300  # 5 minutes
            },
            FailureMode.RESOURCE_EXHAUSTION: {
                'type': 'threshold',
                'memory_threshold': 0.90,
                'cpu_threshold': 0.95,
                'disk_threshold': 0.95,
                'prediction_horizon': 180  # 3 minutes
            },
            FailureMode.POWER_ANOMALY: {
                'type': 'statistical',
                'power_deviation_threshold': 2.5,  # Standard deviations
                'prediction_horizon': 120  # 2 minutes
            }
        }
        
    async def predict_failures(self, current_metrics: HealthMetrics) -> List[Dict[str, Any]]:
        """Predict potential failures based on current metrics."""
        predictions = []
        
        # Add current metrics to history
        self.health_history.append(current_metrics)
        
        # Check each prediction model
        for failure_mode, model in self.prediction_models.items():
            prediction = await self._evaluate_prediction_model(
                failure_mode, model, current_metrics
            )
            if prediction:
                predictions.append(prediction)
                
        return predictions
        
    async def _evaluate_prediction_model(
        self, 
        failure_mode: FailureMode, 
        model: Dict[str, Any], 
        metrics: HealthMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a specific prediction model."""
        
        if model['type'] == 'threshold':
            return await self._evaluate_threshold_model(failure_mode, model, metrics)
        elif model['type'] == 'statistical':
            return await self._evaluate_statistical_model(failure_mode, model, metrics)
        
        return None
        
    async def _evaluate_threshold_model(
        self, 
        failure_mode: FailureMode, 
        model: Dict[str, Any], 
        metrics: HealthMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate threshold-based prediction model."""
        
        if failure_mode == FailureMode.THERMAL_ISSUE:
            if metrics.temperature > model['temperature_threshold']:
                # Check temperature rate of change
                if len(self.health_history) >= 2:
                    prev_metrics = list(self.health_history)[-2]
                    time_diff = metrics.timestamp - prev_metrics.timestamp
                    temp_rate = (metrics.temperature - prev_metrics.temperature) / max(time_diff / 60, 0.1)
                    
                    if temp_rate > model['temperature_rate_threshold']:
                        confidence = min(1.0, (metrics.temperature - model['temperature_threshold']) / 10.0)
                        return {
                            'failure_mode': failure_mode,
                            'probability': confidence,
                            'time_to_failure': model['prediction_horizon'] * (1 - confidence),
                            'contributing_factors': ['high_temperature', 'rapid_temperature_rise'],
                            'recommended_actions': ['reduce_workload', 'check_cooling', 'emergency_shutdown']
                        }
                        
        elif failure_mode == FailureMode.RESOURCE_EXHAUSTION:
            resource_pressure = max(
                metrics.memory_usage - model['memory_threshold'],
                metrics.cpu_usage - model['cpu_threshold'],
                metrics.disk_usage - model['disk_threshold']
            )
            
            if resource_pressure > 0:
                confidence = min(1.0, resource_pressure * 5)  # Scale to 0-1
                return {
                    'failure_mode': failure_mode,
                    'probability': confidence,
                    'time_to_failure': model['prediction_horizon'] * (1 - confidence),
                    'contributing_factors': ['high_resource_usage'],
                    'recommended_actions': ['scale_resources', 'reduce_load', 'cleanup_resources']
                }
                
        return None
        
    async def _evaluate_statistical_model(
        self, 
        failure_mode: FailureMode, 
        model: Dict[str, Any], 
        metrics: HealthMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate statistical prediction model."""
        
        if failure_mode == FailureMode.POWER_ANOMALY and len(self.health_history) >= 20:
            # Get recent power consumption values
            recent_power = [m.power_consumption for m in list(self.health_history)[-20:]]
            power_mean = np.mean(recent_power)
            power_std = np.std(recent_power)
            
            if power_std > 0:
                z_score = abs(metrics.power_consumption - power_mean) / power_std
                
                if z_score > model['power_deviation_threshold']:
                    confidence = min(1.0, (z_score - model['power_deviation_threshold']) / 2.0)
                    return {
                        'failure_mode': failure_mode,
                        'probability': confidence,
                        'time_to_failure': model['prediction_horizon'],
                        'contributing_factors': ['power_anomaly'],
                        'recommended_actions': ['check_power_supply', 'validate_configuration']
                    }
                    
        return None
        
    def learn_from_failure(self, failure_event: FailureEvent, preceding_metrics: List[HealthMetrics]):
        """Learn from failure events to improve prediction accuracy."""
        self.failure_history.append(failure_event)
        
        # Simple pattern learning - identify common precursors
        if len(preceding_metrics) >= 5:
            patterns = self._extract_failure_patterns(failure_event, preceding_metrics)
            failure_event.learned_patterns = patterns
            
            # Update prediction models based on patterns
            self._update_prediction_models(failure_event, patterns)
            
    def _extract_failure_patterns(
        self, 
        failure: FailureEvent, 
        metrics: List[HealthMetrics]
    ) -> List[str]:
        """Extract patterns from metrics preceding a failure."""
        patterns = []
        
        # Simple pattern detection
        temperatures = [m.temperature for m in metrics]
        if temperatures and max(temperatures) - min(temperatures) > 10:
            patterns.append('temperature_spike')
            
        cpu_usage = [m.cpu_usage for m in metrics]
        if cpu_usage and np.mean(cpu_usage[-3:]) > 0.9:
            patterns.append('high_cpu_before_failure')
            
        memory_usage = [m.memory_usage for m in metrics]
        if memory_usage and any(m > 0.95 for m in memory_usage[-5:]):
            patterns.append('memory_pressure')
            
        return patterns
        
    def _update_prediction_models(self, failure: FailureEvent, patterns: List[str]):
        """Update prediction models based on learned patterns."""
        # Simple model updates - could be more sophisticated with ML
        if failure.failure_mode in self.prediction_models:
            model = self.prediction_models[failure.failure_mode]
            
            # Adjust thresholds based on observed failures
            if 'temperature_spike' in patterns and model.get('type') == 'threshold':
                # Lower temperature threshold if we observed failures
                current_threshold = model.get('temperature_threshold', 85.0)
                model['temperature_threshold'] = max(75.0, current_threshold - 2.0)


class AutoRecoverySystem:
    """
    Autonomous recovery system that implements various recovery strategies.
    """
    
    def __init__(self, reliability_level: ReliabilityLevel):
        self.reliability_level = reliability_level
        self.recovery_strategies = {}
        self.recovery_history = deque(maxlen=1000)
        self.system_state = {}
        self.backup_manager = BackupManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
    def _initialize_recovery_strategies(self):
        """Initialize recovery strategies for different failure modes."""
        self.recovery_strategies = {
            FailureMode.COMPILATION_ERROR: [
                RecoveryStrategy.AUTOMATIC_RETRY,
                RecoveryStrategy.ROLLBACK,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            FailureMode.SYNTHESIS_FAILURE: [
                RecoveryStrategy.AUTOMATIC_RETRY,
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.MANUAL_INTERVENTION
            ],
            FailureMode.RESOURCE_EXHAUSTION: [
                RecoveryStrategy.RESOURCE_SCALING,
                RecoveryStrategy.LOAD_BALANCING,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            FailureMode.THERMAL_ISSUE: [
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.SYSTEM_RESTART,
                RecoveryStrategy.MANUAL_INTERVENTION
            ],
            FailureMode.HARDWARE_FAILURE: [
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.MANUAL_INTERVENTION
            ],
            FailureMode.NETWORK_TIMEOUT: [
                RecoveryStrategy.AUTOMATIC_RETRY,
                RecoveryStrategy.LOAD_BALANCING,
                RecoveryStrategy.FAILOVER
            ]
        }
        
    async def attempt_recovery(self, failure_event: FailureEvent) -> bool:
        """Attempt to recover from a failure event."""
        self.logger.info(f"Attempting recovery for {failure_event.failure_mode.value}")
        
        # Get recovery strategies for this failure mode
        strategies = self.recovery_strategies.get(failure_event.failure_mode, [])
        
        for strategy in strategies:
            try:
                success = await self._execute_recovery_strategy(strategy, failure_event)
                if success:
                    failure_event.recovery_successful = True
                    failure_event.recovery_strategy = strategy
                    failure_event.recovery_time = time.time() - failure_event.timestamp
                    
                    self.recovery_history.append({
                        'failure_mode': failure_event.failure_mode,
                        'strategy': strategy,
                        'success': True,
                        'recovery_time': failure_event.recovery_time,
                        'timestamp': time.time()
                    })
                    
                    self.logger.info(f"Recovery successful using {strategy.value}")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy.value} failed: {e}")
                continue
                
        # All strategies failed
        failure_event.recovery_successful = False
        self.logger.error(f"All recovery strategies failed for {failure_event.failure_mode.value}")
        return False
        
    async def _execute_recovery_strategy(
        self, 
        strategy: RecoveryStrategy, 
        failure_event: FailureEvent
    ) -> bool:
        """Execute a specific recovery strategy."""
        
        if strategy == RecoveryStrategy.AUTOMATIC_RETRY:
            return await self._retry_operation(failure_event)
            
        elif strategy == RecoveryStrategy.FAILOVER:
            return await self._failover_to_backup(failure_event)
            
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation(failure_event)
            
        elif strategy == RecoveryStrategy.ROLLBACK:
            return await self._rollback_changes(failure_event)
            
        elif strategy == RecoveryStrategy.RESOURCE_SCALING:
            return await self._scale_resources(failure_event)
            
        elif strategy == RecoveryStrategy.LOAD_BALANCING:
            return await self._rebalance_load(failure_event)
            
        elif strategy == RecoveryStrategy.SYSTEM_RESTART:
            return await self._restart_system_component(failure_event)
            
        else:
            # Manual intervention required
            return False
            
    async def _retry_operation(self, failure_event: FailureEvent) -> bool:
        """Retry the failed operation with exponential backoff."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
            
            try:
                # Attempt to recreate the operation context and retry
                success = await self._retry_specific_operation(failure_event)
                if success:
                    return True
            except Exception as e:
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                
        return False
        
    async def _retry_specific_operation(self, failure_event: FailureEvent) -> bool:
        """Retry the specific operation that failed."""
        # This would contain logic specific to the type of operation
        # For now, simulate retry logic
        if failure_event.failure_mode == FailureMode.COMPILATION_ERROR:
            # Retry compilation with different parameters
            return await self._retry_compilation(failure_event)
        elif failure_event.failure_mode == FailureMode.NETWORK_TIMEOUT:
            # Retry network operation
            return await self._retry_network_operation(failure_event)
        
        return False
        
    async def _retry_compilation(self, failure_event: FailureEvent) -> bool:
        """Retry compilation with adjusted parameters."""
        # Simulate compilation retry with reduced optimization
        await asyncio.sleep(0.5)  # Simulate compilation time
        return np.random.random() > 0.3  # 70% success rate
        
    async def _retry_network_operation(self, failure_event: FailureEvent) -> bool:
        """Retry network operation."""
        # Simulate network retry
        await asyncio.sleep(0.2)
        return np.random.random() > 0.4  # 60% success rate
        
    async def _failover_to_backup(self, failure_event: FailureEvent) -> bool:
        """Failover to backup system or resource."""
        try:
            # Check if backup is available
            backup_available = await self.backup_manager.check_backup_availability(
                failure_event.component
            )
            
            if backup_available:
                # Activate backup
                success = await self.backup_manager.activate_backup(failure_event.component)
                return success
                
        except Exception as e:
            self.logger.error(f"Failover failed: {e}")
            
        return False
        
    async def _graceful_degradation(self, failure_event: FailureEvent) -> bool:
        """Implement graceful degradation of service."""
        try:
            # Reduce system capabilities to maintain basic functionality
            if failure_event.failure_mode == FailureMode.RESOURCE_EXHAUSTION:
                # Reduce processing load
                await self._reduce_processing_load()
                return True
                
            elif failure_event.failure_mode == FailureMode.THERMAL_ISSUE:
                # Reduce clock frequency or disable non-critical functions
                await self._reduce_thermal_load()
                return True
                
        except Exception as e:
            self.logger.error(f"Graceful degradation failed: {e}")
            
        return False
        
    async def _reduce_processing_load(self):
        """Reduce processing load to conserve resources."""
        # Placeholder for load reduction logic
        self.logger.info("Reducing processing load for resource conservation")
        
    async def _reduce_thermal_load(self):
        """Reduce thermal load to prevent overheating."""
        # Placeholder for thermal load reduction
        self.logger.info("Reducing thermal load to prevent overheating")
        
    async def _rollback_changes(self, failure_event: FailureEvent) -> bool:
        """Rollback recent changes that may have caused the failure."""
        try:
            # Check if rollback point is available
            rollback_available = await self.backup_manager.check_rollback_point(
                failure_event.component
            )
            
            if rollback_available:
                success = await self.backup_manager.rollback_to_checkpoint(
                    failure_event.component
                )
                return success
                
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            
        return False
        
    async def _scale_resources(self, failure_event: FailureEvent) -> bool:
        """Scale system resources to handle increased load."""
        try:
            # Attempt to allocate additional resources
            if failure_event.failure_mode == FailureMode.RESOURCE_EXHAUSTION:
                # Request additional memory/CPU/storage
                success = await self._request_additional_resources()
                return success
                
        except Exception as e:
            self.logger.error(f"Resource scaling failed: {e}")
            
        return False
        
    async def _request_additional_resources(self) -> bool:
        """Request additional system resources."""
        # Placeholder for resource scaling logic
        self.logger.info("Requesting additional system resources")
        return True  # Assume success for now
        
    async def _rebalance_load(self, failure_event: FailureEvent) -> bool:
        """Rebalance system load across available resources."""
        try:
            # Redistribute workload
            success = await self._redistribute_workload()
            return success
            
        except Exception as e:
            self.logger.error(f"Load balancing failed: {e}")
            
        return False
        
    async def _redistribute_workload(self) -> bool:
        """Redistribute workload across available resources."""
        # Placeholder for load balancing logic
        self.logger.info("Redistributing workload for better balance")
        return True
        
    async def _restart_system_component(self, failure_event: FailureEvent) -> bool:
        """Restart a specific system component."""
        try:
            # Gracefully restart the failed component
            component = failure_event.component
            self.logger.info(f"Restarting component: {component}")
            
            # Simulate component restart
            await asyncio.sleep(2.0)  # Simulate restart time
            return True
            
        except Exception as e:
            self.logger.error(f"Component restart failed: {e}")
            
        return False


class BackupManager:
    """
    Manages system backups and rollback points.
    """
    
    def __init__(self):
        self.backup_dir = Path("/tmp/fpga_reliability_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.backup_registry = {}
        
    async def create_checkpoint(self, component: str, state: Dict[str, Any]) -> str:
        """Create a checkpoint for a system component."""
        timestamp = time.time()
        checkpoint_id = f"{component}_{int(timestamp)}"
        
        checkpoint_path = self.backup_dir / f"{checkpoint_id}.pkl"
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
                
            self.backup_registry[component] = {
                'checkpoint_id': checkpoint_id,
                'timestamp': timestamp,
                'path': checkpoint_path
            }
            
            return checkpoint_id
            
        except Exception as e:
            logging.error(f"Failed to create checkpoint: {e}")
            raise
            
    async def check_backup_availability(self, component: str) -> bool:
        """Check if a backup is available for a component."""
        return component in self.backup_registry
        
    async def activate_backup(self, component: str) -> bool:
        """Activate backup for a failed component."""
        if component not in self.backup_registry:
            return False
            
        try:
            # Load backup state
            backup_info = self.backup_registry[component]
            with open(backup_info['path'], 'rb') as f:
                backup_state = pickle.load(f)
                
            # Activate backup (placeholder)
            # In real implementation, this would restore the component state
            return True
            
        except Exception as e:
            logging.error(f"Failed to activate backup: {e}")
            return False
            
    async def check_rollback_point(self, component: str) -> bool:
        """Check if rollback point exists for component."""
        return await self.check_backup_availability(component)
        
    async def rollback_to_checkpoint(self, component: str) -> bool:
        """Rollback component to last checkpoint."""
        return await self.activate_backup(component)


class AutonomousReliabilityOrchestrator:
    """
    Main orchestrator for the autonomous reliability system.
    """
    
    def __init__(self, reliability_level: ReliabilityLevel = ReliabilityLevel.STANDARD):
        self.reliability_level = reliability_level
        self.failure_detector = PredictiveFailureDetector()
        self.recovery_system = AutoRecoverySystem(reliability_level)
        self.health_monitor = HealthMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Monitoring configuration
        self.monitoring_active = False
        self.monitoring_interval = self._get_monitoring_interval()
        self.monitoring_task = None
        
        # Reliability metrics
        self.uptime_start = time.time()
        self.failure_count = 0
        self.recovery_count = 0
        self.mtbf = 0.0  # Mean Time Between Failures
        self.mttr = 0.0  # Mean Time To Recovery
        
    def _get_monitoring_interval(self) -> float:
        """Get monitoring interval based on reliability level."""
        intervals = {
            ReliabilityLevel.CRITICAL: 5.0,      # 5 seconds
            ReliabilityLevel.HIGH: 15.0,         # 15 seconds
            ReliabilityLevel.STANDARD: 30.0,     # 30 seconds
            ReliabilityLevel.EXPERIMENTAL: 60.0  # 60 seconds
        }
        return intervals.get(self.reliability_level, 30.0)
        
    async def start_autonomous_monitoring(self):
        """Start autonomous reliability monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info(f"Started autonomous reliability monitoring (interval: {self.monitoring_interval}s)")
        
    async def stop_autonomous_monitoring(self):
        """Stop autonomous reliability monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Stopped autonomous reliability monitoring")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect health metrics
                health_metrics = await self.health_monitor.collect_metrics()
                
                # Predict potential failures
                predictions = await self.failure_detector.predict_failures(health_metrics)
                
                # Handle predictions
                for prediction in predictions:
                    await self._handle_failure_prediction(prediction, health_metrics)
                
                # Check for active failures
                active_failures = await self._detect_active_failures(health_metrics)
                for failure in active_failures:
                    await self._handle_failure_event(failure)
                
                # Update reliability metrics
                self._update_reliability_metrics()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(min(self.monitoring_interval, 60.0))
                
    async def _handle_failure_prediction(
        self, 
        prediction: Dict[str, Any], 
        health_metrics: HealthMetrics
    ):
        """Handle a predicted failure."""
        failure_mode = prediction['failure_mode']
        probability = prediction['probability']
        time_to_failure = prediction.get('time_to_failure', 0)
        
        self.logger.warning(
            f"Predicted failure: {failure_mode.value} "
            f"(probability: {probability:.2f}, TTF: {time_to_failure:.0f}s)"
        )
        
        # Take preventive action based on recommended actions
        recommended_actions = prediction.get('recommended_actions', [])
        for action in recommended_actions:
            await self._execute_preventive_action(action, prediction)
            
    async def _execute_preventive_action(self, action: str, prediction: Dict[str, Any]):
        """Execute preventive action to avoid predicted failure."""
        if action == 'reduce_workload':
            await self._reduce_system_workload()
        elif action == 'scale_resources':
            await self.recovery_system._scale_resources(None)
        elif action == 'check_cooling':
            await self._check_cooling_systems()
        elif action == 'cleanup_resources':
            await self._cleanup_system_resources()
        # Add more preventive actions as needed
        
    async def _reduce_system_workload(self):
        """Reduce system workload proactively."""
        self.logger.info("Reducing system workload proactively")
        
    async def _check_cooling_systems(self):
        """Check cooling system status."""
        self.logger.info("Checking cooling system status")
        
    async def _cleanup_system_resources(self):
        """Cleanup system resources to prevent exhaustion."""
        self.logger.info("Cleaning up system resources")
        
    async def _detect_active_failures(self, health_metrics: HealthMetrics) -> List[FailureEvent]:
        """Detect active system failures."""
        failures = []
        
        # Check for critical thresholds
        if health_metrics.temperature > 90.0:
            failures.append(FailureEvent(
                timestamp=time.time(),
                failure_mode=FailureMode.THERMAL_ISSUE,
                severity='critical',
                component='thermal_system',
                description=f'Critical temperature: {health_metrics.temperature}°C'
            ))
            
        if health_metrics.memory_usage > 0.98:
            failures.append(FailureEvent(
                timestamp=time.time(),
                failure_mode=FailureMode.RESOURCE_EXHAUSTION,
                severity='critical',
                component='memory_system',
                description=f'Memory exhaustion: {health_metrics.memory_usage * 100:.1f}%'
            ))
            
        if health_metrics.error_rate > 0.01:  # More than 1% error rate
            failures.append(FailureEvent(
                timestamp=time.time(),
                failure_mode=FailureMode.SOFTWARE_ERROR,
                severity='high',
                component='software_system',
                description=f'High error rate: {health_metrics.error_rate * 100:.2f}%'
            ))
            
        return failures
        
    async def _handle_failure_event(self, failure_event: FailureEvent):
        """Handle an active failure event."""
        self.failure_count += 1
        
        self.logger.error(
            f"Failure detected: {failure_event.failure_mode.value} "
            f"({failure_event.severity}) - {failure_event.description}"
        )
        
        # Attempt recovery
        recovery_start = time.time()
        recovery_success = await self.recovery_system.attempt_recovery(failure_event)
        
        if recovery_success:
            self.recovery_count += 1
            recovery_time = time.time() - recovery_start
            failure_event.recovery_time = recovery_time
            
            self.logger.info(
                f"Recovery successful for {failure_event.failure_mode.value} "
                f"(recovery time: {recovery_time:.2f}s)"
            )
        else:
            self.logger.error(f"Recovery failed for {failure_event.failure_mode.value}")
            
        # Learn from the failure
        recent_metrics = list(self.health_monitor.metrics_history)[-10:]
        self.failure_detector.learn_from_failure(failure_event, recent_metrics)
        
    def _update_reliability_metrics(self):
        """Update system reliability metrics."""
        current_time = time.time()
        uptime = current_time - self.uptime_start
        
        # Calculate MTBF (Mean Time Between Failures)
        if self.failure_count > 0:
            self.mtbf = uptime / self.failure_count
        else:
            self.mtbf = uptime
            
        # Calculate MTTR (Mean Time To Recovery)
        if self.recovery_count > 0:
            total_recovery_time = sum(
                recovery['recovery_time'] 
                for recovery in self.recovery_system.recovery_history
                if recovery['success']
            )
            self.mttr = total_recovery_time / self.recovery_count
            
    def get_reliability_report(self) -> Dict[str, Any]:
        """Get comprehensive reliability report."""
        current_time = time.time()
        uptime = current_time - self.uptime_start
        
        # Calculate availability (uptime / total time)
        availability = 1.0
        if self.failure_count > 0:
            total_downtime = self.mttr * self.failure_count
            availability = max(0.0, (uptime - total_downtime) / uptime)
            
        return {
            'reliability_level': self.reliability_level.value,
            'uptime_hours': uptime / 3600,
            'failure_count': self.failure_count,
            'recovery_count': self.recovery_count,
            'mtbf_hours': self.mtbf / 3600,
            'mttr_seconds': self.mttr,
            'availability_percentage': availability * 100,
            'monitoring_active': self.monitoring_active,
            'last_health_check': current_time,
            'predictive_models_count': len(self.failure_detector.prediction_models),
            'recovery_strategies_count': len(self.recovery_system.recovery_strategies)
        }


class HealthMonitor:
    """
    System health monitoring component.
    """
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
    async def collect_metrics(self) -> HealthMetrics:
        """Collect current system health metrics."""
        try:
            # Simulate metrics collection
            # In real implementation, this would interface with system monitoring APIs
            metrics = HealthMetrics(
                timestamp=time.time(),
                cpu_usage=np.random.normal(0.6, 0.1),
                memory_usage=np.random.normal(0.7, 0.1),
                disk_usage=np.random.normal(0.5, 0.1),
                temperature=np.random.normal(65.0, 5.0),
                power_consumption=np.random.normal(250.0, 25.0),
                network_latency=np.random.normal(50.0, 10.0),
                error_rate=max(0.0, np.random.normal(0.001, 0.0005)),
                throughput=np.random.normal(1000.0, 100.0),
                availability=0.99,
                reliability_score=0.95
            )
            
            # Ensure metrics are within reasonable bounds
            metrics.cpu_usage = max(0.0, min(1.0, metrics.cpu_usage))
            metrics.memory_usage = max(0.0, min(1.0, metrics.memory_usage))
            metrics.disk_usage = max(0.0, min(1.0, metrics.disk_usage))
            metrics.temperature = max(20.0, min(100.0, metrics.temperature))
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect health metrics: {e}")
            raise


def create_reliability_orchestrator(
    reliability_level: str = "standard",
    enable_monitoring: bool = True
) -> AutonomousReliabilityOrchestrator:
    """
    Factory function to create a configured reliability orchestrator.
    
    Args:
        reliability_level: Reliability level ('critical', 'high', 'standard', 'experimental')
        enable_monitoring: Whether to start monitoring immediately
        
    Returns:
        Configured AutonomousReliabilityOrchestrator
    """
    try:
        level = ReliabilityLevel(reliability_level.lower())
    except ValueError:
        level = ReliabilityLevel.STANDARD
        
    orchestrator = AutonomousReliabilityOrchestrator(level)
    
    if enable_monitoring:
        # Start monitoring in background
        asyncio.create_task(orchestrator.start_autonomous_monitoring())
        
    return orchestrator