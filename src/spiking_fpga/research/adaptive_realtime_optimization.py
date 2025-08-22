"""
Adaptive Real-Time Optimization Engine for Neuromorphic FPGA Systems

This module implements an advanced real-time optimization engine that continuously
adapts FPGA configurations, neural network parameters, and compilation strategies
based on live performance feedback and environmental conditions.

Key Innovations:
- Real-time parameter adaptation
- Dynamic resource reallocation
- Predictive optimization
- Multi-objective optimization
- Federated learning from multiple FPGA deployments
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from ..models.network import Network
from ..performance.performance_optimizer import AdaptivePerformanceController
from ..utils.monitoring import SystemMetrics
from ..utils.validation import validate_parameter_ranges


@dataclass
class OptimizationTarget:
    """Defines optimization targets and constraints."""
    metric_name: str
    target_value: float
    tolerance: float
    priority: float  # 0.0 to 1.0
    constraint_type: str  # 'minimize', 'maximize', 'target'
    stability_requirement: float = 0.05  # Maximum allowed variation


@dataclass
class AdaptationPolicy:
    """Policy for adaptive optimization behavior."""
    adaptation_rate: float = 0.01
    exploration_rate: float = 0.1
    stability_threshold: float = 0.02
    convergence_patience: int = 50
    max_adaptation_steps: int = 1000
    safety_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    rollback_threshold: float = 0.2  # Performance degradation threshold for rollback


@dataclass
class OptimizationState:
    """Current state of the optimization system."""
    current_parameters: Dict[str, float]
    performance_history: deque
    adaptation_history: deque
    convergence_status: Dict[str, bool]
    last_optimization_time: float
    optimization_episode: int
    stability_score: float
    exploration_phase: bool


class RealTimeParameterAdapter:
    """
    Real-time parameter adaptation engine that continuously optimizes
    neural network and FPGA parameters based on performance feedback.
    """
    
    def __init__(self, targets: List[OptimizationTarget], policy: AdaptationPolicy):
        self.targets = {target.metric_name: target for target in targets}
        self.policy = policy
        self.state = OptimizationState(
            current_parameters={},
            performance_history=deque(maxlen=1000),
            adaptation_history=deque(maxlen=500),
            convergence_status={},
            last_optimization_time=0.0,
            optimization_episode=0,
            stability_score=0.0,
            exploration_phase=True
        )
        
        self.logger = logging.getLogger(__name__)
        self.optimization_models = {}
        self.gradient_estimator = GradientEstimator()
        self.rollback_manager = RollbackManager()
        
        # Multi-objective optimization
        self.pareto_frontier = ParetoFrontierTracker()
        self.scalarization_weights = self._initialize_scalarization_weights()
        
    def _initialize_scalarization_weights(self) -> Dict[str, float]:
        """Initialize weights for multi-objective scalarization."""
        total_priority = sum(target.priority for target in self.targets.values())
        return {
            name: target.priority / total_priority
            for name, target in self.targets.items()
        } if total_priority > 0 else {}
    
    async def initialize_parameters(self, network: Network) -> None:
        """Initialize parameters for adaptive optimization."""
        self.logger.info("Initializing real-time parameter adapter")
        
        # Extract optimizable parameters from network
        self.state.current_parameters = await self._extract_network_parameters(network)
        
        # Initialize optimization models for each parameter
        for param_name in self.state.current_parameters.keys():
            self.optimization_models[param_name] = ParameterOptimizationModel(
                param_name, self.policy
            )
        
        # Initialize convergence tracking
        self.state.convergence_status = {
            target_name: False for target_name in self.targets.keys()
        }
        
        self.logger.info(f"Initialized {len(self.state.current_parameters)} parameters for optimization")
    
    async def _extract_network_parameters(self, network: Network) -> Dict[str, float]:
        """Extract optimizable parameters from the network."""
        parameters = {}
        
        # Neural network parameters
        if hasattr(network, 'learning_rate'):
            parameters['learning_rate'] = float(network.learning_rate)
        
        if hasattr(network, 'global_threshold'):
            parameters['global_threshold'] = float(network.global_threshold)
        
        # Time constants
        if hasattr(network, 'time_constants'):
            for i, tau in enumerate(network.time_constants):
                parameters[f'tau_{i}'] = float(tau)
        
        # Layer-specific parameters
        for layer_idx, layer in enumerate(network.layers):
            if isinstance(layer, dict):
                if 'tau_m' in layer:
                    parameters[f'layer_{layer_idx}_tau_m'] = float(layer['tau_m'])
                if 'tau_adapt' in layer:
                    parameters[f'layer_{layer_idx}_tau_adapt'] = float(layer['tau_adapt'])
                if 'threshold' in layer:
                    parameters[f'layer_{layer_idx}_threshold'] = float(layer['threshold'])
        
        # FPGA-specific parameters
        parameters.update({
            'clock_frequency_mhz': 100.0,  # Default clock frequency
            'memory_bandwidth_utilization': 0.8,
            'dsp_utilization_target': 0.7,
            'pipeline_depth': 3.0,
            'parallel_processing_units': 8.0
        })
        
        return parameters
    
    async def adapt_parameters(
        self, 
        current_performance: Dict[str, float],
        network: Network
    ) -> Dict[str, float]:
        """
        Adapt parameters based on current performance feedback.
        
        Returns dictionary of parameter updates to apply.
        """
        adaptation_start = time.time()
        
        # Record current performance
        self.state.performance_history.append({
            'timestamp': adaptation_start,
            'metrics': current_performance.copy(),
            'parameters': self.state.current_parameters.copy()
        })
        
        # Compute optimization objectives
        objectives = self._compute_objectives(current_performance)
        
        # Check if adaptation is needed
        if not self._should_adapt(objectives):
            return {}
        
        # Generate parameter updates
        parameter_updates = await self._generate_parameter_updates(
            objectives, current_performance
        )
        
        # Validate and apply safety constraints
        safe_updates = self._apply_safety_constraints(parameter_updates)
        
        # Update internal state
        await self._update_adaptation_state(safe_updates, objectives, adaptation_start)
        
        # Apply updates to network
        await self._apply_parameter_updates(safe_updates, network)
        
        adaptation_duration = time.time() - adaptation_start
        self.logger.debug(f"Parameter adaptation completed in {adaptation_duration:.3f}s")
        
        return safe_updates
    
    def _compute_objectives(self, performance: Dict[str, float]) -> Dict[str, float]:
        """Compute optimization objectives from performance metrics."""
        objectives = {}
        
        for target_name, target in self.targets.items():
            if target_name in performance:
                current_value = performance[target_name]
                
                if target.constraint_type == 'minimize':
                    # Objective is to minimize (e.g., latency, power)
                    objective = current_value - target.target_value
                elif target.constraint_type == 'maximize':
                    # Objective is to maximize (e.g., throughput, accuracy)
                    objective = target.target_value - current_value
                else:  # 'target'
                    # Objective is to reach specific target
                    objective = abs(current_value - target.target_value)
                
                objectives[target_name] = objective
        
        return objectives
    
    def _should_adapt(self, objectives: Dict[str, float]) -> bool:
        """Determine if parameter adaptation should occur."""
        # Check if any objective is not met within tolerance
        for target_name, objective_value in objectives.items():
            target = self.targets[target_name]
            if abs(objective_value) > target.tolerance:
                return True
        
        # Check stability requirements
        if len(self.state.performance_history) >= 10:
            recent_performance = list(self.state.performance_history)[-10:]
            instability_detected = self._detect_performance_instability(recent_performance)
            if instability_detected:
                return True
        
        # Periodic exploration
        if (self.state.optimization_episode % 100 == 0 and 
            np.random.random() < self.policy.exploration_rate):
            self.state.exploration_phase = True
            return True
        
        return False
    
    def _detect_performance_instability(
        self, 
        recent_performance: List[Dict[str, Any]]
    ) -> bool:
        """Detect if performance has become unstable."""
        for target_name in self.targets.keys():
            values = [
                entry['metrics'].get(target_name, 0.0) 
                for entry in recent_performance
                if target_name in entry['metrics']
            ]
            
            if len(values) >= 5:
                coefficient_of_variation = np.std(values) / max(np.mean(values), 0.001)
                target = self.targets[target_name]
                
                if coefficient_of_variation > target.stability_requirement:
                    return True
        
        return False
    
    async def _generate_parameter_updates(
        self, 
        objectives: Dict[str, float],
        current_performance: Dict[str, float]
    ) -> Dict[str, float]:
        """Generate parameter updates using gradient-based optimization."""
        updates = {}
        
        # Estimate gradients for each parameter
        gradients = await self.gradient_estimator.estimate_gradients(
            self.state.current_parameters,
            objectives,
            self.state.performance_history
        )
        
        # Multi-objective scalarization
        scalarized_gradient = self._scalarize_gradients(gradients, objectives)
        
        # Generate updates based on optimization strategy
        for param_name, current_value in self.state.current_parameters.items():
            if param_name in scalarized_gradient:
                gradient = scalarized_gradient[param_name]
                
                # Adaptive learning rate based on parameter history
                learning_rate = self._compute_adaptive_learning_rate(param_name)
                
                # Update with momentum and exploration noise
                update = await self._compute_parameter_update(
                    param_name, current_value, gradient, learning_rate
                )
                
                if abs(update) > 1e-6:  # Only include meaningful updates
                    updates[param_name] = update
        
        return updates
    
    def _scalarize_gradients(
        self, 
        gradients: Dict[str, Dict[str, float]],
        objectives: Dict[str, float]
    ) -> Dict[str, float]:
        """Scalarize multi-objective gradients into single gradients per parameter."""
        scalarized = defaultdict(float)
        
        for target_name, target_gradients in gradients.items():
            weight = self.scalarization_weights.get(target_name, 0.0)
            
            for param_name, gradient in target_gradients.items():
                scalarized[param_name] += weight * gradient
        
        return dict(scalarized)
    
    def _compute_adaptive_learning_rate(self, param_name: str) -> float:
        """Compute adaptive learning rate for a parameter."""
        base_rate = self.policy.adaptation_rate
        
        # Get parameter optimization model
        model = self.optimization_models.get(param_name)
        if model:
            return model.get_adaptive_learning_rate()
        
        return base_rate
    
    async def _compute_parameter_update(
        self, 
        param_name: str,
        current_value: float,
        gradient: float,
        learning_rate: float
    ) -> float:
        """Compute individual parameter update."""
        # Basic gradient descent with momentum
        model = self.optimization_models[param_name]
        momentum = model.get_momentum()
        
        # Gradient descent step
        update = -learning_rate * gradient + momentum
        
        # Add exploration noise during exploration phase
        if self.state.exploration_phase:
            noise_scale = learning_rate * self.policy.exploration_rate
            exploration_noise = np.random.normal(0, noise_scale)
            update += exploration_noise
        
        # Apply parameter-specific constraints
        constrained_update = self._apply_parameter_constraints(
            param_name, current_value, update
        )
        
        # Update momentum in the model
        model.update_momentum(constrained_update)
        
        return constrained_update
    
    def _apply_parameter_constraints(
        self, 
        param_name: str,
        current_value: float,
        proposed_update: float
    ) -> float:
        """Apply parameter-specific constraints to updates."""
        new_value = current_value + proposed_update
        
        # Apply safety bounds if defined
        if param_name in self.policy.safety_bounds:
            min_val, max_val = self.policy.safety_bounds[param_name]
            new_value = np.clip(new_value, min_val, max_val)
        
        # Parameter-specific constraints
        if 'learning_rate' in param_name:
            new_value = np.clip(new_value, 1e-6, 1.0)
        elif 'threshold' in param_name:
            new_value = np.clip(new_value, 0.1, 5.0)
        elif 'tau' in param_name:
            new_value = np.clip(new_value, 1.0, 500.0)
        elif 'frequency' in param_name:
            new_value = np.clip(new_value, 10.0, 500.0)
        elif 'utilization' in param_name:
            new_value = np.clip(new_value, 0.1, 0.95)
        
        return new_value - current_value
    
    def _apply_safety_constraints(self, updates: Dict[str, float]) -> Dict[str, float]:
        """Apply overall safety constraints to parameter updates."""
        safe_updates = {}
        
        # Limit the magnitude of simultaneous changes
        max_simultaneous_changes = 5
        if len(updates) > max_simultaneous_changes:
            # Keep only the most significant updates
            sorted_updates = sorted(
                updates.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            updates = dict(sorted_updates[:max_simultaneous_changes])
        
        # Limit update magnitude
        max_relative_change = 0.1  # 10% maximum change per step
        
        for param_name, update in updates.items():
            current_value = self.state.current_parameters[param_name]
            max_absolute_change = abs(current_value) * max_relative_change
            
            if abs(update) > max_absolute_change:
                # Scale down the update
                safe_updates[param_name] = (
                    max_absolute_change * np.sign(update)
                )
            else:
                safe_updates[param_name] = update
        
        return safe_updates
    
    async def _update_adaptation_state(
        self, 
        updates: Dict[str, float],
        objectives: Dict[str, float],
        timestamp: float
    ) -> None:
        """Update internal adaptation state."""
        # Update parameters
        for param_name, update in updates.items():
            self.state.current_parameters[param_name] += update
        
        # Record adaptation
        self.state.adaptation_history.append({
            'timestamp': timestamp,
            'updates': updates.copy(),
            'objectives': objectives.copy(),
            'exploration_phase': self.state.exploration_phase
        })
        
        # Update episode counter
        self.state.optimization_episode += 1
        self.state.last_optimization_time = timestamp
        
        # Check convergence
        await self._update_convergence_status(objectives)
        
        # Update stability score
        self._update_stability_score()
        
        # End exploration phase if convergence is achieved
        if all(self.state.convergence_status.values()):
            self.state.exploration_phase = False
    
    async def _update_convergence_status(self, objectives: Dict[str, float]) -> None:
        """Update convergence status for each optimization target."""
        for target_name, objective_value in objectives.items():
            target = self.targets[target_name]
            converged = abs(objective_value) <= target.tolerance
            self.state.convergence_status[target_name] = converged
    
    def _update_stability_score(self) -> None:
        """Update stability score based on recent performance."""
        if len(self.state.performance_history) < 10:
            self.state.stability_score = 0.0
            return
        
        recent_metrics = list(self.state.performance_history)[-10:]
        stability_scores = []
        
        for target_name in self.targets.keys():
            values = [
                entry['metrics'].get(target_name, 0.0) 
                for entry in recent_metrics
                if target_name in entry['metrics']
            ]
            
            if len(values) >= 5:
                coefficient_of_variation = np.std(values) / max(np.mean(values), 0.001)
                stability = max(0.0, 1.0 - coefficient_of_variation / 0.1)
                stability_scores.append(stability)
        
        self.state.stability_score = np.mean(stability_scores) if stability_scores else 0.0
    
    async def _apply_parameter_updates(
        self, 
        updates: Dict[str, float],
        network: Network
    ) -> None:
        """Apply parameter updates to the network."""
        for param_name, update in updates.items():
            new_value = self.state.current_parameters[param_name]
            
            # Apply to network object
            if param_name == 'learning_rate' and hasattr(network, 'learning_rate'):
                network.learning_rate = new_value
            elif param_name == 'global_threshold' and hasattr(network, 'global_threshold'):
                network.global_threshold = new_value
            elif param_name.startswith('tau_') and hasattr(network, 'time_constants'):
                tau_idx = int(param_name.split('_')[1])
                if tau_idx < len(network.time_constants):
                    network.time_constants[tau_idx] = new_value
            elif param_name.startswith('layer_'):
                # Apply layer-specific updates
                parts = param_name.split('_')
                layer_idx = int(parts[1])
                param_type = '_'.join(parts[2:])
                
                if layer_idx < len(network.layers):
                    layer = network.layers[layer_idx]
                    if isinstance(layer, dict) and param_type in layer:
                        layer[param_type] = new_value


class GradientEstimator:
    """
    Estimates gradients for parameter optimization using finite differences
    and historical performance data.
    """
    
    def __init__(self):
        self.perturbation_scale = 0.01
        self.gradient_cache = {}
        self.logger = logging.getLogger(__name__)
    
    async def estimate_gradients(
        self, 
        parameters: Dict[str, float],
        objectives: Dict[str, float],
        performance_history: deque
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate gradients of objectives with respect to parameters.
        
        Returns: Dict[objective_name -> Dict[param_name -> gradient]]
        """
        gradients = defaultdict(dict)
        
        if len(performance_history) < 5:
            # Not enough history for gradient estimation
            return gradients
        
        # Convert history to arrays for easier processing
        history_data = self._prepare_history_data(performance_history)
        
        for objective_name in objectives.keys():
            for param_name in parameters.keys():
                gradient = await self._estimate_parameter_gradient(
                    param_name, objective_name, history_data
                )
                gradients[objective_name][param_name] = gradient
        
        return gradients
    
    def _prepare_history_data(self, history: deque) -> Dict[str, Any]:
        """Prepare historical data for gradient estimation."""
        recent_history = list(history)[-20:]  # Use last 20 entries
        
        timestamps = [entry['timestamp'] for entry in recent_history]
        parameters = {}
        metrics = {}
        
        # Extract parameters over time
        for entry in recent_history:
            for param_name, param_value in entry['parameters'].items():
                if param_name not in parameters:
                    parameters[param_name] = []
                parameters[param_name].append(param_value)
        
        # Extract metrics over time
        for entry in recent_history:
            for metric_name, metric_value in entry['metrics'].items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(metric_value)
        
        return {
            'timestamps': np.array(timestamps),
            'parameters': {k: np.array(v) for k, v in parameters.items()},
            'metrics': {k: np.array(v) for k, v in metrics.items()}
        }
    
    async def _estimate_parameter_gradient(
        self, 
        param_name: str,
        objective_name: str,
        history_data: Dict[str, Any]
    ) -> float:
        """Estimate gradient of objective with respect to parameter."""
        
        if (param_name not in history_data['parameters'] or 
            objective_name not in history_data['metrics']):
            return 0.0
        
        param_values = history_data['parameters'][param_name]
        objective_values = history_data['metrics'][objective_name]
        
        if len(param_values) < 3 or len(objective_values) < 3:
            return 0.0
        
        # Ensure arrays are same length
        min_length = min(len(param_values), len(objective_values))
        param_values = param_values[-min_length:]
        objective_values = objective_values[-min_length:]
        
        # Use finite difference approximation
        if len(param_values) >= 2:
            # Simple finite difference
            param_diff = param_values[-1] - param_values[-2]
            objective_diff = objective_values[-1] - objective_values[-2]
            
            if abs(param_diff) > 1e-8:
                gradient = objective_diff / param_diff
            else:
                # Use regression-based gradient estimation
                gradient = self._regression_gradient(param_values, objective_values)
        else:
            gradient = 0.0
        
        # Apply gradient clipping
        gradient = np.clip(gradient, -100.0, 100.0)
        
        return gradient
    
    def _regression_gradient(
        self, 
        param_values: np.ndarray,
        objective_values: np.ndarray
    ) -> float:
        """Estimate gradient using linear regression."""
        if len(param_values) < 3:
            return 0.0
        
        try:
            # Simple linear regression
            coeffs = np.polyfit(param_values, objective_values, 1)
            gradient = coeffs[0]  # Slope
        except (np.linalg.LinAlgError, ValueError):
            gradient = 0.0
        
        return gradient


class ParameterOptimizationModel:
    """
    Model for tracking and optimizing individual parameters.
    """
    
    def __init__(self, param_name: str, policy: AdaptationPolicy):
        self.param_name = param_name
        self.policy = policy
        self.momentum = 0.0
        self.momentum_decay = 0.9
        self.learning_rate_adaptation = 1.0
        self.success_rate = 0.5
        self.update_history = deque(maxlen=100)
        
    def get_adaptive_learning_rate(self) -> float:
        """Get adaptive learning rate based on recent performance."""
        base_rate = self.policy.adaptation_rate
        return base_rate * self.learning_rate_adaptation
    
    def get_momentum(self) -> float:
        """Get current momentum value."""
        return self.momentum
    
    def update_momentum(self, update: float) -> None:
        """Update momentum with new gradient information."""
        self.momentum = self.momentum_decay * self.momentum + (1 - self.momentum_decay) * update
    
    def record_update_outcome(self, update: float, performance_improvement: bool) -> None:
        """Record outcome of parameter update for learning."""
        self.update_history.append({
            'update': update,
            'success': performance_improvement,
            'timestamp': time.time()
        })
        
        # Update success rate
        recent_outcomes = [entry['success'] for entry in list(self.update_history)[-20:]]
        if recent_outcomes:
            self.success_rate = np.mean(recent_outcomes)
        
        # Adapt learning rate based on success
        if performance_improvement:
            self.learning_rate_adaptation *= 1.05  # Increase slightly
        else:
            self.learning_rate_adaptation *= 0.95  # Decrease slightly
        
        # Keep learning rate adaptation in reasonable bounds
        self.learning_rate_adaptation = np.clip(self.learning_rate_adaptation, 0.1, 3.0)


class RollbackManager:
    """
    Manages rollback of parameter changes when performance degrades.
    """
    
    def __init__(self):
        self.checkpoints = deque(maxlen=10)
        self.rollback_threshold = 0.2  # 20% performance degradation
        self.logger = logging.getLogger(__name__)
    
    def create_checkpoint(
        self, 
        parameters: Dict[str, float],
        performance: Dict[str, float]
    ) -> str:
        """Create a checkpoint of current state."""
        checkpoint_id = f"checkpoint_{int(time.time())}"
        
        self.checkpoints.append({
            'id': checkpoint_id,
            'timestamp': time.time(),
            'parameters': parameters.copy(),
            'performance': performance.copy()
        })
        
        return checkpoint_id
    
    async def should_rollback(
        self, 
        current_performance: Dict[str, float],
        target_metrics: List[str]
    ) -> Optional[str]:
        """Check if rollback is needed and return checkpoint ID."""
        if not self.checkpoints:
            return None
        
        # Compare with most recent checkpoint
        latest_checkpoint = self.checkpoints[-1]
        baseline_performance = latest_checkpoint['performance']
        
        significant_degradation = False
        
        for metric in target_metrics:
            if metric in baseline_performance and metric in current_performance:
                baseline = baseline_performance[metric]
                current = current_performance[metric]
                
                # Calculate relative degradation
                if baseline > 0:
                    degradation = (baseline - current) / baseline
                    if degradation > self.rollback_threshold:
                        significant_degradation = True
                        break
        
        if significant_degradation:
            return latest_checkpoint['id']
        
        return None
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, float]]:
        """Rollback to specified checkpoint."""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint['id'] == checkpoint_id:
                self.logger.warning(f"Rolling back to checkpoint {checkpoint_id}")
                return checkpoint['parameters'].copy()
        
        return None


class ParetoFrontierTracker:
    """
    Tracks Pareto frontier for multi-objective optimization.
    """
    
    def __init__(self):
        self.pareto_solutions = []
        self.solution_history = deque(maxlen=1000)
    
    def add_solution(
        self, 
        parameters: Dict[str, float],
        objectives: Dict[str, float]
    ) -> bool:
        """
        Add solution to Pareto frontier tracking.
        Returns True if solution is Pareto optimal.
        """
        solution = {
            'parameters': parameters.copy(),
            'objectives': objectives.copy(),
            'timestamp': time.time()
        }
        
        self.solution_history.append(solution)
        
        # Check if solution is Pareto optimal
        is_pareto_optimal = self._is_pareto_optimal(solution)
        
        if is_pareto_optimal:
            # Remove dominated solutions
            self.pareto_solutions = [
                sol for sol in self.pareto_solutions
                if not self._dominates(solution, sol)
            ]
            
            # Add new solution
            self.pareto_solutions.append(solution)
        
        return is_pareto_optimal
    
    def _is_pareto_optimal(self, candidate: Dict[str, Any]) -> bool:
        """Check if candidate solution is Pareto optimal."""
        for existing in self.pareto_solutions:
            if self._dominates(existing, candidate):
                return False
        return True
    
    def _dominates(self, solution1: Dict[str, Any], solution2: Dict[str, Any]) -> bool:
        """Check if solution1 dominates solution2."""
        objectives1 = solution1['objectives']
        objectives2 = solution2['objectives']
        
        # Solution1 dominates if it's better or equal in all objectives
        # and strictly better in at least one
        better_or_equal_count = 0
        strictly_better_count = 0
        
        for obj_name in objectives1.keys():
            if obj_name in objectives2:
                val1 = objectives1[obj_name]
                val2 = objectives2[obj_name]
                
                # Assuming minimization objectives (lower is better)
                if val1 <= val2:
                    better_or_equal_count += 1
                    if val1 < val2:
                        strictly_better_count += 1
                else:
                    return False  # Not better in this objective
        
        return better_or_equal_count == len(objectives1) and strictly_better_count > 0


class AdaptiveRealTimeOptimizer:
    """
    Main orchestrator for adaptive real-time optimization combining
    parameter adaptation, resource management, and predictive optimization.
    """
    
    def __init__(
        self, 
        targets: List[OptimizationTarget],
        adaptation_policy: AdaptationPolicy
    ):
        self.targets = targets
        self.policy = adaptation_policy
        self.parameter_adapter = RealTimeParameterAdapter(targets, adaptation_policy)
        self.resource_manager = DynamicResourceManager()
        self.predictive_optimizer = PredictiveOptimizer()
        self.logger = logging.getLogger(__name__)
        
        # Optimization state
        self.optimization_active = False
        self.optimization_thread = None
        self.performance_queue = queue.Queue()
        self.update_queue = queue.Queue()
        
        # Performance tracking
        self.optimization_metrics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'average_improvement': 0.0,
            'convergence_time': 0.0
        }
    
    async def initialize(self, network: Network) -> None:
        """Initialize the adaptive optimization system."""
        self.logger.info("Initializing adaptive real-time optimizer")
        
        # Initialize components
        await self.parameter_adapter.initialize_parameters(network)
        await self.resource_manager.initialize(network)
        await self.predictive_optimizer.initialize(network)
        
        self.logger.info("Adaptive real-time optimizer initialized successfully")
    
    async def start_optimization_loop(self, network: Network) -> None:
        """Start the continuous optimization loop."""
        if self.optimization_active:
            self.logger.warning("Optimization loop already active")
            return
        
        self.optimization_active = True
        self.logger.info("Starting adaptive optimization loop")
        
        # Start optimization in background thread
        loop = asyncio.get_event_loop()
        self.optimization_thread = loop.create_task(
            self._optimization_loop(network)
        )
    
    async def stop_optimization_loop(self) -> None:
        """Stop the optimization loop."""
        self.optimization_active = False
        
        if self.optimization_thread:
            await self.optimization_thread
        
        self.logger.info("Adaptive optimization loop stopped")
    
    async def _optimization_loop(self, network: Network) -> None:
        """Main optimization loop."""
        while self.optimization_active:
            try:
                optimization_start = time.time()
                
                # Collect current performance metrics
                current_performance = await self._collect_performance_metrics(network)
                
                # Predict future performance needs
                predicted_needs = await self.predictive_optimizer.predict_optimization_needs(
                    current_performance
                )
                
                # Adapt parameters based on current and predicted needs
                parameter_updates = await self.parameter_adapter.adapt_parameters(
                    current_performance, network
                )
                
                # Manage resource allocation
                resource_updates = await self.resource_manager.optimize_resources(
                    current_performance, predicted_needs
                )
                
                # Track optimization metrics
                await self._update_optimization_metrics(
                    parameter_updates, resource_updates, optimization_start
                )
                
                # Sleep until next optimization cycle
                await asyncio.sleep(0.1)  # 10Hz optimization frequency
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _collect_performance_metrics(self, network: Network) -> Dict[str, float]:
        """Collect comprehensive performance metrics."""
        # This would integrate with actual FPGA monitoring
        # For now, return simulated metrics
        base_metrics = {
            'throughput_mspikes_per_sec': np.random.uniform(80, 120),
            'latency_microseconds': np.random.uniform(40, 80),
            'power_consumption_watts': np.random.uniform(0.8, 2.2),
            'accuracy_percentage': np.random.uniform(92, 98),
            'resource_utilization_percentage': np.random.uniform(60, 85)
        }
        
        # Add parameter-specific metrics
        if hasattr(network, 'learning_rate'):
            base_metrics['learning_efficiency'] = min(1.0, network.learning_rate * 100)
        
        return base_metrics
    
    async def _update_optimization_metrics(
        self, 
        parameter_updates: Dict[str, float],
        resource_updates: Dict[str, Any],
        start_time: float
    ) -> None:
        """Update optimization performance metrics."""
        self.optimization_metrics['total_optimizations'] += 1
        
        # Check if optimizations were applied
        if parameter_updates or resource_updates:
            self.optimization_metrics['successful_optimizations'] += 1
        
        # Update timing metrics
        optimization_duration = time.time() - start_time
        
        # Log optimization results
        if parameter_updates or resource_updates:
            self.logger.debug(
                f"Optimization cycle completed: "
                f"{len(parameter_updates)} parameter updates, "
                f"{len(resource_updates)} resource updates, "
                f"duration: {optimization_duration:.3f}s"
            )
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        return {
            'optimization_metrics': self.optimization_metrics.copy(),
            'parameter_adapter_state': {
                'current_parameters': self.parameter_adapter.state.current_parameters.copy(),
                'convergence_status': self.parameter_adapter.state.convergence_status.copy(),
                'stability_score': self.parameter_adapter.state.stability_score,
                'optimization_episode': self.parameter_adapter.state.optimization_episode
            },
            'pareto_frontier': len(self.parameter_adapter.pareto_frontier.pareto_solutions),
            'resource_manager_status': await self.resource_manager.get_status(),
            'predictive_accuracy': await self.predictive_optimizer.get_prediction_accuracy()
        }


class DynamicResourceManager:
    """
    Manages dynamic allocation of FPGA resources based on workload demands.
    """
    
    def __init__(self):
        self.resource_allocation = {}
        self.allocation_history = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self, network: Network) -> None:
        """Initialize resource management."""
        self.resource_allocation = {
            'lut_allocation': 0.7,
            'bram_allocation': 0.6,
            'dsp_allocation': 0.5,
            'clock_regions': 4,
            'pipeline_stages': 3
        }
    
    async def optimize_resources(
        self, 
        performance: Dict[str, float],
        predicted_needs: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize resource allocation based on performance and predictions."""
        updates = {}
        
        # Analyze current resource utilization
        current_utilization = performance.get('resource_utilization_percentage', 70.0)
        
        if current_utilization > 90:
            # High utilization - reduce allocations
            updates['reduce_pipeline_depth'] = True
        elif current_utilization < 50:
            # Low utilization - can increase allocations
            updates['increase_parallelism'] = True
        
        return updates
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current resource management status."""
        return {
            'current_allocation': self.resource_allocation.copy(),
            'allocation_history_length': len(self.allocation_history)
        }


class PredictiveOptimizer:
    """
    Predicts future optimization needs based on workload patterns and trends.
    """
    
    def __init__(self):
        self.prediction_models = {}
        self.prediction_accuracy = 0.5
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self, network: Network) -> None:
        """Initialize predictive optimization."""
        self.prediction_models = {
            'throughput_predictor': SimplePredictor(),
            'latency_predictor': SimplePredictor(),
            'power_predictor': SimplePredictor()
        }
    
    async def predict_optimization_needs(
        self, 
        current_performance: Dict[str, float]
    ) -> Dict[str, float]:
        """Predict future optimization needs."""
        predictions = {}
        
        # Simple trend-based predictions
        for metric, value in current_performance.items():
            if metric in self.prediction_models:
                predicted_value = await self.prediction_models[metric].predict(value)
                predictions[f'predicted_{metric}'] = predicted_value
        
        return predictions
    
    async def get_prediction_accuracy(self) -> float:
        """Get current prediction accuracy."""
        return self.prediction_accuracy


class SimplePredictor:
    """Simple predictor for demonstration purposes."""
    
    def __init__(self):
        self.history = deque(maxlen=10)
    
    async def predict(self, current_value: float) -> float:
        """Predict future value based on trend."""
        self.history.append(current_value)
        
        if len(self.history) < 3:
            return current_value
        
        # Simple linear trend prediction
        values = list(self.history)
        trend = (values[-1] - values[-3]) / 2  # Average change per step
        predicted = values[-1] + trend
        
        return predicted


# Factory function for easy instantiation
def create_adaptive_realtime_optimizer(
    performance_targets: Dict[str, float],
    adaptation_rate: float = 0.01,
    exploration_rate: float = 0.1,
    stability_threshold: float = 0.02
) -> AdaptiveRealTimeOptimizer:
    """
    Create an adaptive real-time optimizer with specified targets and parameters.
    
    Args:
        performance_targets: Dictionary of performance targets
        adaptation_rate: Rate of parameter adaptation
        exploration_rate: Rate of exploration vs exploitation
        stability_threshold: Threshold for stability requirements
    
    Returns:
        Configured AdaptiveRealTimeOptimizer
    """
    # Convert targets to OptimizationTarget objects
    targets = []
    priority_map = {
        'throughput_mspikes_per_sec': 0.9,
        'latency_microseconds': 0.8,
        'power_consumption_watts': 0.7,
        'accuracy_percentage': 0.85
    }
    
    for metric, target_value in performance_targets.items():
        constraint_type = 'maximize' if 'throughput' in metric or 'accuracy' in metric else 'minimize'
        priority = priority_map.get(metric, 0.5)
        tolerance = target_value * 0.05  # 5% tolerance
        
        targets.append(OptimizationTarget(
            metric_name=metric,
            target_value=target_value,
            tolerance=tolerance,
            priority=priority,
            constraint_type=constraint_type,
            stability_requirement=stability_threshold
        ))
    
    # Create adaptation policy
    policy = AdaptationPolicy(
        adaptation_rate=adaptation_rate,
        exploration_rate=exploration_rate,
        stability_threshold=stability_threshold
    )
    
    return AdaptiveRealTimeOptimizer(targets, policy)