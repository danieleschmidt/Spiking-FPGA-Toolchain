"""
Auto-scaling system with predictive resource management.
"""

import time
import threading
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operation."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    COMPILATION_NODES = "compilation_nodes"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK_BANDWIDTH = "network_bandwidth"


@dataclass
class ScalingMetric:
    """Represents a metric used for scaling decisions."""
    name: str
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    aggregation_period: float = 300.0  # 5 minutes


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    timestamp: float
    resource_type: ResourceType
    direction: ScalingDirection
    old_capacity: int
    new_capacity: int
    trigger_metrics: Dict[str, float]
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class ScalingPolicy:
    """Defines scaling behavior and constraints."""
    resource_type: ResourceType
    min_capacity: int = 1
    max_capacity: int = 10
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    cooldown_period: float = 300.0  # 5 minutes
    enable_predictive_scaling: bool = True
    enable_reactive_scaling: bool = True
    
    # Advanced settings
    target_utilization: float = 0.7  # 70%
    scale_up_threshold: float = 0.8  # 80%
    scale_down_threshold: float = 0.3  # 30%


class ResourcePredictor:
    """
    Machine learning-based resource demand predictor.
    
    Uses time series analysis and pattern recognition to predict
    future resource needs based on historical data.
    """
    
    def __init__(self, prediction_horizon: float = 1800.0):  # 30 minutes
        self.prediction_horizon = prediction_horizon
        self.historical_data: List[Tuple[float, Dict[str, float]]] = []
        self.patterns: Dict[str, Any] = {}
        self.seasonal_patterns: Dict[str, List[float]] = {}
        self.trend_analysis: Dict[str, float] = {}
        
        # Simple moving average parameters
        self.short_window = 12  # 1 hour with 5-minute intervals
        self.long_window = 48   # 4 hours with 5-minute intervals
    
    def add_observation(self, metrics: Dict[str, float]):
        """Add observation to historical data."""
        timestamp = time.time()
        self.historical_data.append((timestamp, metrics.copy()))
        
        # Limit history to prevent memory growth
        cutoff_time = timestamp - 86400 * 7  # Keep 1 week
        self.historical_data = [
            (ts, data) for ts, data in self.historical_data 
            if ts > cutoff_time
        ]
        
        # Update patterns if we have enough data
        if len(self.historical_data) >= self.long_window:
            self._update_patterns()
    
    def predict_demand(self, metric_name: str, horizon_seconds: float = None) -> float:
        """Predict future demand for a specific metric."""
        if not self.historical_data:
            return 0.0
        
        horizon = horizon_seconds or self.prediction_horizon
        
        # Get recent values for the metric
        recent_values = [
            data.get(metric_name, 0.0) 
            for _, data in self.historical_data[-self.short_window:]
        ]
        
        if not recent_values:
            return 0.0
        
        # Simple prediction combining trend and seasonal patterns
        current_value = recent_values[-1]
        
        # Trend component
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
            periods_ahead = horizon / 300.0  # Assuming 5-minute intervals
            trend_prediction = current_value + (trend * periods_ahead)
        else:
            trend_prediction = current_value
        
        # Seasonal component
        seasonal_adjustment = self._get_seasonal_adjustment(metric_name, horizon)
        
        # Combined prediction
        predicted_value = trend_prediction + seasonal_adjustment
        
        # Apply bounds (can't be negative, and limit extreme predictions)
        predicted_value = max(0.0, min(predicted_value, current_value * 3.0))
        
        logger.debug(f"Predicted {metric_name}: {predicted_value:.2f} (current: {current_value:.2f})")
        
        return predicted_value
    
    def _update_patterns(self):
        """Update trend and seasonal patterns."""
        for metric_name in self._get_all_metrics():
            values = [
                data.get(metric_name, 0.0)
                for _, data in self.historical_data
            ]
            
            if len(values) >= self.long_window:
                # Calculate trend
                self.trend_analysis[metric_name] = self._calculate_trend(values)
                
                # Update seasonal patterns
                self._update_seasonal_pattern(metric_name, values)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(values)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        
        if n * sum_xx - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope
    
    def _update_seasonal_pattern(self, metric_name: str, values: List[float]):
        """Update seasonal pattern for a metric."""
        # Simple daily pattern (assuming 5-minute intervals)
        daily_periods = 288  # 24 hours * 60 minutes / 5 minutes
        
        if len(values) < daily_periods:
            return
        
        # Group by time of day
        daily_averages = []
        for period in range(daily_periods):
            period_values = [
                values[i] for i in range(period, len(values), daily_periods)
            ]
            if period_values:
                daily_averages.append(np.mean(period_values))
            else:
                daily_averages.append(0.0)
        
        self.seasonal_patterns[metric_name] = daily_averages
    
    def _get_seasonal_adjustment(self, metric_name: str, horizon_seconds: float) -> float:
        """Get seasonal adjustment for prediction."""
        if metric_name not in self.seasonal_patterns:
            return 0.0
        
        pattern = self.seasonal_patterns[metric_name]
        if not pattern:
            return 0.0
        
        # Calculate which period the prediction horizon falls into
        current_time = time.time()
        future_time = current_time + horizon_seconds
        
        # Get time of day for both current and future
        current_period = int((current_time % 86400) / 300) % len(pattern)  # 5-minute periods
        future_period = int((future_time % 86400) / 300) % len(pattern)
        
        current_seasonal = pattern[current_period]
        future_seasonal = pattern[future_period]
        
        # Return the difference
        return future_seasonal - current_seasonal
    
    def _get_all_metrics(self) -> List[str]:
        """Get all metric names from historical data."""
        metrics = set()
        for _, data in self.historical_data:
            metrics.update(data.keys())
        return list(metrics)
    
    def get_prediction_confidence(self, metric_name: str) -> float:
        """Get confidence level for predictions."""
        if len(self.historical_data) < self.short_window:
            return 0.0
        
        # Simple confidence based on data availability and trend stability
        data_confidence = min(1.0, len(self.historical_data) / self.long_window)
        
        # Trend stability (lower variance = higher confidence)
        recent_values = [
            data.get(metric_name, 0.0)
            for _, data in self.historical_data[-self.short_window:]
        ]
        
        if len(recent_values) > 1:
            variance = np.var(recent_values)
            mean_value = np.mean(recent_values)
            coefficient_of_variation = variance / (mean_value + 1e-6)  # Avoid division by zero
            stability_confidence = 1.0 / (1.0 + coefficient_of_variation)
        else:
            stability_confidence = 0.5
        
        return (data_confidence + stability_confidence) / 2.0


class AutoScaler:
    """
    Intelligent auto-scaling system with machine learning-based predictions.
    
    Features:
    - Predictive scaling based on historical patterns
    - Reactive scaling based on current metrics
    - Multi-metric decision making
    - Configurable scaling policies
    - Built-in safeguards and cooldown periods
    """
    
    def __init__(self, scaling_policies: List[ScalingPolicy] = None):
        self.scaling_policies = scaling_policies or []
        self.scaling_history: List[ScalingEvent] = []
        self.current_capacity: Dict[ResourceType, int] = {}
        self.last_scaling_time: Dict[ResourceType, float] = {}
        self.scaling_metrics: Dict[str, ScalingMetric] = {}
        
        # Resource predictor
        self.predictor = ResourcePredictor()
        
        # Scaling callbacks
        self.scaling_callbacks: Dict[ResourceType, Callable] = {}
        
        # Monitoring
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        # Initialize capacities
        for policy in self.scaling_policies:
            self.current_capacity[policy.resource_type] = policy.min_capacity
            self.last_scaling_time[policy.resource_type] = 0.0
        
        logger.info("AutoScaler initialized")
    
    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add a scaling policy."""
        self.scaling_policies.append(policy)
        self.current_capacity[policy.resource_type] = policy.min_capacity
        self.last_scaling_time[policy.resource_type] = 0.0
        logger.info(f"Added scaling policy for {policy.resource_type.value}")
    
    def register_scaling_callback(self, resource_type: ResourceType, callback: Callable):
        """Register callback for scaling operations."""
        self.scaling_callbacks[resource_type] = callback
        logger.info(f"Registered scaling callback for {resource_type.value}")
    
    def add_scaling_metric(self, metric: ScalingMetric):
        """Add a metric for scaling decisions."""
        self.scaling_metrics[metric.name] = metric
        logger.info(f"Added scaling metric: {metric.name}")
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update current metric values."""
        current_time = time.time()
        
        with self._lock:
            # Update scaling metrics
            for name, value in metrics.items():
                if name in self.scaling_metrics:
                    self.scaling_metrics[name].current_value = value
            
            # Add to predictor
            self.predictor.add_observation(metrics)
            
            # Check if scaling is needed
            self._evaluate_scaling_decisions()
    
    def _evaluate_scaling_decisions(self):
        """Evaluate if scaling is needed based on current and predicted metrics."""
        current_time = time.time()
        
        for policy in self.scaling_policies:
            if not self._is_scaling_allowed(policy, current_time):
                continue
            
            scaling_decision = self._make_scaling_decision(policy)
            
            if scaling_decision != ScalingDirection.STABLE:
                self._execute_scaling(policy, scaling_decision, current_time)
    
    def _make_scaling_decision(self, policy: ScalingPolicy) -> ScalingDirection:
        """Make scaling decision for a specific policy."""
        current_capacity = self.current_capacity.get(policy.resource_type, policy.min_capacity)
        
        # Get relevant metrics for this resource type
        relevant_metrics = self._get_relevant_metrics(policy.resource_type)
        
        if not relevant_metrics:
            return ScalingDirection.STABLE
        
        # Reactive scaling based on current metrics
        reactive_decision = self._reactive_scaling_decision(relevant_metrics, policy)
        
        # Predictive scaling based on predictions
        predictive_decision = ScalingDirection.STABLE
        if policy.enable_predictive_scaling:
            predictive_decision = self._predictive_scaling_decision(relevant_metrics, policy)
        
        # Combine decisions (prioritize scale up, be conservative on scale down)
        if reactive_decision == ScalingDirection.UP or predictive_decision == ScalingDirection.UP:
            if current_capacity < policy.max_capacity:
                return ScalingDirection.UP
        elif reactive_decision == ScalingDirection.DOWN and predictive_decision != ScalingDirection.UP:
            if current_capacity > policy.min_capacity:
                return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    def _reactive_scaling_decision(self, metrics: List[ScalingMetric], policy: ScalingPolicy) -> ScalingDirection:
        """Make reactive scaling decision based on current metrics."""
        weighted_utilization = 0.0
        total_weight = 0.0
        
        scale_up_signals = 0
        scale_down_signals = 0
        
        for metric in metrics:
            if metric.current_value > metric.threshold_up:
                scale_up_signals += metric.weight
            elif metric.current_value < metric.threshold_down:
                scale_down_signals += metric.weight
            
            # Calculate weighted utilization
            utilization = metric.current_value / metric.threshold_up if metric.threshold_up > 0 else 0
            weighted_utilization += utilization * metric.weight
            total_weight += metric.weight
        
        if total_weight > 0:
            avg_utilization = weighted_utilization / total_weight
            
            if avg_utilization > policy.scale_up_threshold:
                return ScalingDirection.UP
            elif avg_utilization < policy.scale_down_threshold:
                return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    def _predictive_scaling_decision(self, metrics: List[ScalingMetric], policy: ScalingPolicy) -> ScalingDirection:
        """Make predictive scaling decision based on forecasted metrics."""
        scale_up_confidence = 0.0
        total_confidence = 0.0
        
        for metric in metrics:
            # Get prediction for the metric
            predicted_value = self.predictor.predict_demand(metric.name)
            confidence = self.predictor.get_prediction_confidence(metric.name)
            
            if confidence < 0.3:  # Low confidence threshold
                continue
            
            # Check if predicted value would trigger scaling
            if predicted_value > metric.threshold_up:
                scale_up_confidence += confidence * metric.weight
            
            total_confidence += confidence * metric.weight
        
        # Make decision based on confidence-weighted predictions
        if total_confidence > 0:
            avg_scale_up_confidence = scale_up_confidence / total_confidence
            
            if avg_scale_up_confidence > 0.6:  # High confidence threshold for predictive scaling
                return ScalingDirection.UP
        
        return ScalingDirection.STABLE
    
    def _get_relevant_metrics(self, resource_type: ResourceType) -> List[ScalingMetric]:
        """Get metrics relevant to a specific resource type."""
        # This could be made configurable in the future
        relevant_patterns = {
            ResourceType.COMPILATION_NODES: ["cpu_utilization", "queue_length", "active_jobs"],
            ResourceType.MEMORY: ["memory_utilization", "memory_usage"],
            ResourceType.STORAGE: ["disk_utilization", "io_wait"],
            ResourceType.NETWORK_BANDWIDTH: ["network_utilization", "network_latency"]
        }
        
        patterns = relevant_patterns.get(resource_type, [])
        return [
            metric for name, metric in self.scaling_metrics.items()
            if any(pattern in name.lower() for pattern in patterns)
        ]
    
    def _is_scaling_allowed(self, policy: ScalingPolicy, current_time: float) -> bool:
        """Check if scaling is allowed (not in cooldown period)."""
        last_scaling = self.last_scaling_time.get(policy.resource_type, 0.0)
        return (current_time - last_scaling) > policy.cooldown_period
    
    def _execute_scaling(self, policy: ScalingPolicy, direction: ScalingDirection, current_time: float):
        """Execute scaling operation."""
        current_capacity = self.current_capacity[policy.resource_type]
        
        if direction == ScalingDirection.UP:
            new_capacity = min(policy.max_capacity, current_capacity + policy.scale_up_increment)
        else:  # ScalingDirection.DOWN
            new_capacity = max(policy.min_capacity, current_capacity - policy.scale_down_increment)
        
        if new_capacity == current_capacity:
            return  # No change needed
        
        # Create scaling event
        scaling_event = ScalingEvent(
            timestamp=current_time,
            resource_type=policy.resource_type,
            direction=direction,
            old_capacity=current_capacity,
            new_capacity=new_capacity,
            trigger_metrics={
                name: metric.current_value
                for name, metric in self.scaling_metrics.items()
            }
        )
        
        try:
            # Execute scaling callback
            if policy.resource_type in self.scaling_callbacks:
                callback = self.scaling_callbacks[policy.resource_type]
                success = callback(current_capacity, new_capacity, direction)
                scaling_event.success = success
                
                if success:
                    self.current_capacity[policy.resource_type] = new_capacity
                    self.last_scaling_time[policy.resource_type] = current_time
                    
                    logger.info(f"Scaled {policy.resource_type.value} {direction.value}: {current_capacity} -> {new_capacity}")
                else:
                    scaling_event.error_message = "Scaling callback returned False"
                    logger.warning(f"Scaling {policy.resource_type.value} failed: callback returned False")
            else:
                scaling_event.success = False
                scaling_event.error_message = "No scaling callback registered"
                logger.warning(f"No scaling callback registered for {policy.resource_type.value}")
        
        except Exception as e:
            scaling_event.success = False
            scaling_event.error_message = str(e)
            logger.error(f"Scaling {policy.resource_type.value} failed: {e}")
        
        # Record scaling event
        with self._lock:
            self.scaling_history.append(scaling_event)
            
            # Limit history size
            if len(self.scaling_history) > 1000:
                self.scaling_history = self.scaling_history[-500:]
    
    def start_monitoring(self, interval: float = 60.0):
        """Start background monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Monitoring already started")
            return
        
        def monitor_loop():
            while not self._stop_monitoring.is_set():
                try:
                    # This would integrate with system monitoring
                    # For now, we rely on external metric updates
                    pass
                except Exception as e:
                    logger.error(f"Auto-scaler monitoring error: {e}")
                
                self._stop_monitoring.wait(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started auto-scaling monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped auto-scaling monitoring")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and history."""
        with self._lock:
            recent_events = [
                {
                    'timestamp': event.timestamp,
                    'resource_type': event.resource_type.value,
                    'direction': event.direction.value,
                    'old_capacity': event.old_capacity,
                    'new_capacity': event.new_capacity,
                    'success': event.success,
                    'error': event.error_message
                }
                for event in self.scaling_history[-10:]  # Last 10 events
            ]
            
            return {
                'current_capacity': {
                    rt.value: capacity 
                    for rt, capacity in self.current_capacity.items()
                },
                'scaling_policies': len(self.scaling_policies),
                'recent_scaling_events': recent_events,
                'total_scaling_events': len(self.scaling_history),
                'predictor_confidence': {
                    metric_name: self.predictor.get_prediction_confidence(metric_name)
                    for metric_name in self.scaling_metrics.keys()
                }
            }
    
    def force_scaling(self, resource_type: ResourceType, new_capacity: int) -> bool:
        """Force scaling to a specific capacity (bypass policies)."""
        if resource_type not in self.current_capacity:
            return False
        
        current_capacity = self.current_capacity[resource_type]
        current_time = time.time()
        
        direction = ScalingDirection.UP if new_capacity > current_capacity else ScalingDirection.DOWN
        
        scaling_event = ScalingEvent(
            timestamp=current_time,
            resource_type=resource_type,
            direction=direction,
            old_capacity=current_capacity,
            new_capacity=new_capacity,
            trigger_metrics={"forced": True}
        )
        
        try:
            if resource_type in self.scaling_callbacks:
                callback = self.scaling_callbacks[resource_type]
                success = callback(current_capacity, new_capacity, direction)
                scaling_event.success = success
                
                if success:
                    self.current_capacity[resource_type] = new_capacity
                    self.last_scaling_time[resource_type] = current_time
                    logger.info(f"Forced scaling {resource_type.value}: {current_capacity} -> {new_capacity}")
                    
                    with self._lock:
                        self.scaling_history.append(scaling_event)
                    
                    return True
        
        except Exception as e:
            scaling_event.error_message = str(e)
            logger.error(f"Forced scaling failed: {e}")
        
        with self._lock:
            self.scaling_history.append(scaling_event)
        
        return False