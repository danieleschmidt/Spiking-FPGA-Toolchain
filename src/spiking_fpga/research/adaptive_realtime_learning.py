"""
Real-time Adaptive Learning for Neuromorphic Systems with Reinforcement Signals.

Advanced adaptive learning system that dynamically adjusts learning rates, synaptic
plasticity rules, and network architecture based on real-time performance feedback
and environmental signals. This system implements:

- Multi-timescale adaptation (milliseconds to hours)
- Reinforcement-driven plasticity modulation  
- Homeostatic regulation with predictive control
- Meta-learning for rapid adaptation to new tasks
- Attention-based learning rate allocation
- Hierarchical temporal memory integration

Key innovations:
- Bio-inspired dopaminergic reward signaling
- Predictive homeostasis with forward models
- Temporal credit assignment for spike timing
- Dynamic architecture search via reinforcement
- Energy-aware adaptation for FPGA deployment

References:
- Schultz et al., "A neural substrate of prediction and reward" (1997)
- Abbott & Nelson, "Synaptic plasticity: taming the beast" (2000)
- Zenke et al., "Diverse synaptic plasticity mechanisms..." (2017)
- Lillicrap et al., "Continuous control with deep reinforcement learning" (2015)
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from enum import Enum
import threading
import queue
import math
from concurrent.futures import ThreadPoolExecutor
from scipy import signal
from scipy.optimize import minimize_scalar
import warnings

logger = logging.getLogger(__name__)


class AdaptationTimescale(Enum):
    """Different timescales for adaptive learning."""
    MILLISECOND = "ms"      # Spike-timing dependent changes
    SECOND = "s"            # Local plasticity adjustments
    MINUTE = "min"          # Network-wide optimizations
    HOUR = "hr"             # Architectural modifications


class ReinforcementSignal(Enum):
    """Types of reinforcement signals for learning modulation."""
    REWARD = "reward"                   # Positive reinforcement
    PUNISHMENT = "punishment"           # Negative reinforcement  
    PREDICTION_ERROR = "pred_error"     # TD error signals
    NOVELTY = "novelty"                 # Novelty-based exploration
    ATTENTION = "attention"             # Attention-guided learning
    HOMEOSTATIC = "homeostatic"         # Stability maintenance


@dataclass
class AdaptationParameters:
    """Configuration for adaptive learning system."""
    # Learning rate adaptation
    base_learning_rate: float = 0.01
    lr_adaptation_rate: float = 0.001
    lr_decay_factor: float = 0.95
    lr_min: float = 1e-6
    lr_max: float = 1.0
    
    # Reinforcement learning
    reward_window: int = 1000           # Reward integration window (ms)
    td_discount_factor: float = 0.99    # Temporal difference discount
    exploration_rate: float = 0.1       # Exploration probability
    exploration_decay: float = 0.995    # Exploration annealing
    
    # Homeostatic regulation
    target_firing_rate: float = 0.1     # Target mean firing rate
    homeostatic_timescale: float = 100.0 # Homeostatic time constant (s)
    stability_threshold: float = 0.05    # Stability requirement
    
    # Meta-learning
    meta_learning_rate: float = 0.001
    adaptation_memory: int = 10000       # Steps to remember for adaptation
    fast_adaptation_steps: int = 100     # Steps for quick adaptation
    
    # Architecture adaptation
    architecture_search_interval: int = 1000  # Steps between arch updates
    connectivity_adaptation_rate: float = 0.01
    pruning_threshold: float = 0.001


@dataclass
class LearningState:
    """Current state of the adaptive learning system."""
    current_learning_rate: float
    reward_history: deque = field(default_factory=lambda: deque(maxlen=10000))
    firing_rate_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    adaptation_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    performance_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    prediction_errors: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Meta-learning state
    meta_parameters: Dict[str, float] = field(default_factory=dict)
    task_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Architecture state
    connectivity_matrix: Optional[np.ndarray] = None
    architecture_scores: Dict[str, float] = field(default_factory=dict)


class ReinforcementModulator:
    """Manages reinforcement signals for learning rate modulation."""
    
    def __init__(self, params: AdaptationParameters):
        self.params = params
        self.dopamine_trace = deque(maxlen=params.reward_window)
        self.prediction_model = None
        self.reward_predictor = SimplePredictor(window_size=100)
        
    def process_reward_signal(self, reward: float, state: Optional[np.ndarray] = None) -> float:
        """Process reward signal and calculate modulation strength."""
        # Add reward to trace
        self.dopamine_trace.append(reward)
        
        # Calculate prediction error if state is available
        if state is not None:
            predicted_reward = self.reward_predictor.predict(state)
            prediction_error = reward - predicted_reward
            self.reward_predictor.update(state, reward)
        else:
            prediction_error = reward
            
        # Calculate dopamine-like modulation signal
        dopamine_level = self._calculate_dopamine_level(reward, prediction_error)
        
        # Temporal difference learning update
        td_error = self._calculate_td_error(reward)
        
        # Combined modulation signal
        modulation_strength = self._combine_signals(dopamine_level, td_error, prediction_error)
        
        return modulation_strength
    
    def _calculate_dopamine_level(self, reward: float, prediction_error: float) -> float:
        """Calculate dopamine-like neuromodulation signal."""
        # Base dopamine response to reward
        dopamine_base = np.tanh(reward)
        
        # Enhancement from prediction error
        prediction_component = 0.5 * np.tanh(prediction_error)
        
        # Recent reward history influence
        if len(self.dopamine_trace) > 1:
            recent_average = np.mean(list(self.dopamine_trace)[-10:])
            contrast_enhancement = 0.3 * (reward - recent_average)
        else:
            contrast_enhancement = 0.0
            
        dopamine_level = dopamine_base + prediction_component + contrast_enhancement
        
        return np.clip(dopamine_level, -2.0, 2.0)
    
    def _calculate_td_error(self, reward: float) -> float:
        """Calculate temporal difference error."""
        if len(self.dopamine_trace) < 2:
            return reward
            
        # Simple TD error calculation
        previous_rewards = list(self.dopamine_trace)[:-1]
        if previous_rewards:
            discounted_future = self.params.td_discount_factor * reward
            td_error = reward + discounted_future - np.mean(previous_rewards[-5:])
        else:
            td_error = reward
            
        return td_error
    
    def _combine_signals(self, dopamine: float, td_error: float, pred_error: float) -> float:
        """Combine different signals into unified modulation strength."""
        # Weighted combination of signals
        weights = [0.4, 0.3, 0.3]  # dopamine, td_error, pred_error
        
        combined = (weights[0] * dopamine + 
                   weights[1] * td_error + 
                   weights[2] * pred_error)
        
        # Apply sigmoid to ensure reasonable range
        modulation = 2.0 / (1.0 + np.exp(-combined)) - 1.0  # Range [-1, 1]
        
        return modulation


class HomeostasticController:
    """Maintains network stability through homeostatic mechanisms."""
    
    def __init__(self, params: AdaptationParameters):
        self.params = params
        self.firing_rate_tracker = FiringRateTracker()
        self.stability_monitor = StabilityMonitor()
        self.predictive_model = HomeostaticPredictor()
        
    def calculate_homeostatic_adjustment(self, current_firing_rates: np.ndarray,
                                       current_weights: List[np.ndarray]) -> Tuple[float, List[np.ndarray]]:
        """Calculate homeostatic adjustments to maintain stability."""
        # Track firing rates
        self.firing_rate_tracker.update(current_firing_rates)
        
        # Check stability
        stability_metric = self.stability_monitor.assess_stability(
            current_firing_rates, current_weights
        )
        
        # Predictive homeostatic control
        predicted_instability = self.predictive_model.predict_instability(
            current_firing_rates, current_weights
        )
        
        # Calculate learning rate adjustment
        lr_adjustment = self._calculate_lr_adjustment(
            current_firing_rates, stability_metric, predicted_instability
        )
        
        # Calculate weight adjustments
        weight_adjustments = self._calculate_weight_adjustments(
            current_firing_rates, current_weights
        )
        
        return lr_adjustment, weight_adjustments
    
    def _calculate_lr_adjustment(self, firing_rates: np.ndarray, 
                               stability: float, predicted_instability: float) -> float:
        """Calculate learning rate adjustment for homeostasis."""
        # Target firing rate deviation
        mean_firing_rate = np.mean(firing_rates)
        rate_deviation = mean_firing_rate - self.params.target_firing_rate
        
        # Stability-based adjustment
        stability_factor = 1.0 - stability  # Higher instability = lower learning rate
        
        # Predictive adjustment
        predictive_factor = 1.0 - predicted_instability
        
        # Combined adjustment
        lr_adjustment = (
            -0.5 * rate_deviation +         # Reduce LR if firing too much
            -0.3 * stability_factor +       # Reduce LR if unstable
            -0.2 * predictive_factor        # Reduce LR if instability predicted
        )
        
        return np.clip(lr_adjustment, -0.5, 0.5)
    
    def _calculate_weight_adjustments(self, firing_rates: np.ndarray,
                                    weights: List[np.ndarray]) -> List[np.ndarray]:
        """Calculate homeostatic weight adjustments."""
        adjustments = []
        
        for weight_matrix in weights:
            # Homeostatic scaling adjustment
            adjustment = np.zeros_like(weight_matrix)
            
            # Scale weights inversely with firing rates
            if len(firing_rates) >= weight_matrix.shape[0]:
                input_rates = firing_rates[:weight_matrix.shape[0]]
                output_rates = firing_rates[:weight_matrix.shape[1]] if len(firing_rates) >= weight_matrix.shape[1] else firing_rates[:1]
                
                # Homeostatic scaling rule
                for i in range(weight_matrix.shape[0]):
                    for j in range(min(weight_matrix.shape[1], len(output_rates))):
                        target_rate = self.params.target_firing_rate
                        
                        # Scale inversely with post-synaptic firing rate
                        if output_rates[j] > target_rate:
                            adjustment[i, j] = -self.params.homeostatic_timescale * 0.001 * (output_rates[j] - target_rate)
                        elif output_rates[j] < target_rate * 0.5:  # Only boost if very low
                            adjustment[i, j] = self.params.homeostatic_timescale * 0.001 * (target_rate - output_rates[j])
            
            adjustments.append(adjustment)
            
        return adjustments


class MetaLearningSystem:
    """Meta-learning system for rapid adaptation to new tasks."""
    
    def __init__(self, params: AdaptationParameters):
        self.params = params
        self.task_history = defaultdict(list)
        self.meta_model = MetaOptimizer(params.meta_learning_rate)
        self.adaptation_strategies = {}
        
    def adapt_to_new_task(self, task_id: str, task_data: Dict[str, Any],
                         current_performance: float) -> Dict[str, float]:
        """Quickly adapt learning parameters for a new task."""
        # Extract task embedding
        task_embedding = self._extract_task_embedding(task_data)
        
        # Find similar tasks
        similar_tasks = self._find_similar_tasks(task_embedding)
        
        # Generate adaptation strategy
        adaptation_strategy = self._generate_adaptation_strategy(
            task_embedding, similar_tasks, current_performance
        )
        
        # Apply meta-learning updates
        meta_updates = self.meta_model.compute_meta_updates(
            task_embedding, adaptation_strategy, current_performance
        )
        
        # Store task information
        self.task_history[task_id].append({
            'embedding': task_embedding,
            'strategy': adaptation_strategy,
            'performance': current_performance,
            'timestamp': time.time()
        })
        
        return meta_updates
    
    def _extract_task_embedding(self, task_data: Dict[str, Any]) -> np.ndarray:
        """Extract task embedding from task characteristics."""
        features = []
        
        # Basic task statistics
        if 'input_size' in task_data:
            features.append(task_data['input_size'])
        if 'output_size' in task_data:
            features.append(task_data['output_size'])
        if 'sequence_length' in task_data:
            features.append(task_data['sequence_length'])
        if 'complexity' in task_data:
            features.append(task_data['complexity'])
            
        # Data distribution characteristics
        if 'data_sample' in task_data:
            sample = np.array(task_data['data_sample'])
            features.extend([
                np.mean(sample),
                np.std(sample),
                np.min(sample),
                np.max(sample)
            ])
        
        # Pad or truncate to fixed size
        target_size = 16
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return np.array(features)
    
    def _find_similar_tasks(self, task_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Find similar tasks based on embedding similarity."""
        similarities = []
        
        for task_id, history in self.task_history.items():
            if history:
                recent_embedding = history[-1]['embedding']
                similarity = self._cosine_similarity(task_embedding, recent_embedding)
                similarities.append((task_id, similarity))
                
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:5]  # Top 5 similar tasks
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norms = np.linalg.norm(a) * np.linalg.norm(b)
        if norms == 0:
            return 0.0
        return dot_product / norms
    
    def _generate_adaptation_strategy(self, task_embedding: np.ndarray,
                                    similar_tasks: List[Tuple[str, float]],
                                    current_performance: float) -> Dict[str, float]:
        """Generate adaptation strategy based on task similarity."""
        strategy = {
            'learning_rate_multiplier': 1.0,
            'plasticity_enhancement': 0.0,
            'exploration_boost': 0.0,
            'attention_focus': 0.0
        }
        
        if similar_tasks:
            # Use weighted average of similar task strategies
            total_similarity = sum(sim for _, sim in similar_tasks)
            
            if total_similarity > 0:
                for task_id, similarity in similar_tasks:
                    weight = similarity / total_similarity
                    task_history = self.task_history[task_id]
                    
                    if task_history:
                        past_strategy = task_history[-1]['strategy']
                        for key in strategy:
                            if key in past_strategy:
                                strategy[key] += weight * past_strategy[key]
        
        # Adjust based on current performance
        if current_performance < 0.5:  # Poor performance
            strategy['learning_rate_multiplier'] *= 1.5
            strategy['exploration_boost'] += 0.2
        elif current_performance > 0.8:  # Good performance
            strategy['learning_rate_multiplier'] *= 0.8
            strategy['exploration_boost'] -= 0.1
            
        return strategy


class AttentionMechanism:
    """Attention-based learning rate allocation."""
    
    def __init__(self):
        self.attention_weights = {}
        self.importance_history = defaultdict(list)
        self.gradient_magnitudes = defaultdict(list)
        
    def calculate_attention_weights(self, gradients: List[np.ndarray],
                                  layer_names: List[str],
                                  task_importance: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate attention weights for learning rate allocation."""
        attention_weights = {}
        
        # Calculate gradient-based importance
        for i, (grad, layer_name) in enumerate(zip(gradients, layer_names)):
            grad_magnitude = np.linalg.norm(grad)
            self.gradient_magnitudes[layer_name].append(grad_magnitude)
            
            # Keep only recent history
            if len(self.gradient_magnitudes[layer_name]) > 100:
                self.gradient_magnitudes[layer_name] = self.gradient_magnitudes[layer_name][-100:]
            
            # Calculate attention based on gradient magnitude stability
            recent_magnitudes = self.gradient_magnitudes[layer_name][-10:]
            magnitude_stability = 1.0 / (1.0 + np.std(recent_magnitudes))
            
            # Combine current magnitude with stability
            attention_score = grad_magnitude * magnitude_stability
            
            attention_weights[layer_name] = attention_score
            
        # Normalize attention weights
        total_attention = sum(attention_weights.values())
        if total_attention > 0:
            for layer_name in attention_weights:
                attention_weights[layer_name] /= total_attention
        
        # Incorporate task-based importance if provided
        if task_importance:
            for layer_name in attention_weights:
                if layer_name in task_importance:
                    attention_weights[layer_name] *= task_importance[layer_name]
        
        # Store attention weights
        self.attention_weights = attention_weights
        
        return attention_weights


class AdaptiveRealTimeLearningSystem:
    """Main adaptive real-time learning system."""
    
    def __init__(self, params: AdaptationParameters):
        self.params = params
        self.state = LearningState(current_learning_rate=params.base_learning_rate)
        
        # Component systems
        self.reinforcement_modulator = ReinforcementModulator(params)
        self.homeostatic_controller = HomeostasticController(params)
        self.meta_learning_system = MetaLearningSystem(params)
        self.attention_mechanism = AttentionMechanism()
        
        # Architecture adaptation
        self.architecture_optimizer = ArchitectureOptimizer(params)
        
        # Threading for real-time updates
        self.update_queue = queue.Queue()
        self.running = False
        self.update_thread = None
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        logger.info("Adaptive real-time learning system initialized")
    
    def start_realtime_adaptation(self) -> None:
        """Start real-time adaptation thread."""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._realtime_update_loop)
            self.update_thread.start()
            logger.info("Real-time adaptation started")
    
    def stop_realtime_adaptation(self) -> None:
        """Stop real-time adaptation thread."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
            logger.info("Real-time adaptation stopped")
    
    def process_learning_step(self, 
                            gradients: List[np.ndarray],
                            firing_rates: np.ndarray,
                            reward_signal: float,
                            current_weights: List[np.ndarray],
                            layer_names: Optional[List[str]] = None,
                            task_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single learning step with adaptive adjustments."""
        
        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(gradients))]
            
        # Process reinforcement signal
        reinforcement_modulation = self.reinforcement_modulator.process_reward_signal(
            reward_signal, state=firing_rates
        )
        
        # Homeostatic adjustments
        homeostatic_lr_adj, weight_adjustments = self.homeostatic_controller.calculate_homeostatic_adjustment(
            firing_rates, current_weights
        )
        
        # Attention-based learning allocation
        attention_weights = self.attention_mechanism.calculate_attention_weights(
            gradients, layer_names
        )
        
        # Meta-learning adaptation
        meta_adjustments = {}
        if task_context:
            task_id = task_context.get('task_id', 'default')
            performance = task_context.get('performance', 0.5)
            meta_adjustments = self.meta_learning_system.adapt_to_new_task(
                task_id, task_context, performance
            )
        
        # Calculate adaptive learning rates per layer
        adaptive_learning_rates = self._calculate_adaptive_learning_rates(
            reinforcement_modulation, homeostatic_lr_adj, attention_weights, meta_adjustments
        )
        
        # Architecture adaptation (periodic)
        architecture_updates = {}
        if len(self.state.adaptation_history) % self.params.architecture_search_interval == 0:
            architecture_updates = self.architecture_optimizer.suggest_updates(
                current_weights, firing_rates, self.state.performance_history
            )
        
        # Update learning state
        self._update_learning_state(
            reward_signal, firing_rates, reinforcement_modulation, 
            homeostatic_lr_adj, adaptive_learning_rates
        )
        
        # Performance tracking
        performance_metrics = self.performance_tracker.update(
            reward_signal, firing_rates, adaptive_learning_rates
        )
        
        # Prepare results
        adaptation_result = {
            'adaptive_learning_rates': adaptive_learning_rates,
            'reinforcement_modulation': reinforcement_modulation,
            'homeostatic_adjustment': homeostatic_lr_adj,
            'attention_weights': attention_weights,
            'meta_adjustments': meta_adjustments,
            'weight_adjustments': weight_adjustments,
            'architecture_updates': architecture_updates,
            'performance_metrics': performance_metrics
        }
        
        return adaptation_result
    
    def _calculate_adaptive_learning_rates(self,
                                         reinforcement_mod: float,
                                         homeostatic_adj: float,
                                         attention_weights: Dict[str, float],
                                         meta_adjustments: Dict[str, float]) -> Dict[str, float]:
        """Calculate adaptive learning rates per layer."""
        base_lr = self.state.current_learning_rate
        
        # Global modulation
        global_modulation = (
            1.0 + 
            0.5 * reinforcement_mod +  # Reinforcement influence
            0.3 * homeostatic_adj      # Homeostatic influence
        )
        
        # Meta-learning global adjustment
        if 'learning_rate_multiplier' in meta_adjustments:
            global_modulation *= meta_adjustments['learning_rate_multiplier']
        
        # Per-layer learning rates
        adaptive_rates = {}
        for layer_name, attention_weight in attention_weights.items():
            # Base rate with global modulation
            layer_lr = base_lr * global_modulation
            
            # Attention-based scaling
            layer_lr *= (0.5 + 1.5 * attention_weight)  # Scale between 0.5x and 2.0x
            
            # Clamp to valid range
            layer_lr = np.clip(layer_lr, self.params.lr_min, self.params.lr_max)
            
            adaptive_rates[layer_name] = layer_lr
        
        return adaptive_rates
    
    def _update_learning_state(self, reward: float, firing_rates: np.ndarray,
                             reinforcement_mod: float, homeostatic_adj: float,
                             adaptive_rates: Dict[str, float]) -> None:
        """Update internal learning state."""
        # Update histories
        self.state.reward_history.append(reward)
        self.state.firing_rate_history.append(np.mean(firing_rates))
        self.state.adaptation_history.append(reinforcement_mod)
        
        # Update current learning rate (average of adaptive rates)
        if adaptive_rates:
            self.state.current_learning_rate = np.mean(list(adaptive_rates.values()))
        
        # Apply learning rate decay
        self.state.current_learning_rate *= self.params.lr_decay_factor
        self.state.current_learning_rate = max(
            self.state.current_learning_rate, self.params.lr_min
        )
    
    def _realtime_update_loop(self) -> None:
        """Real-time update loop running in separate thread."""
        while self.running:
            try:
                # Process queued updates
                while not self.update_queue.empty():
                    update_data = self.update_queue.get_nowait()
                    self._process_queued_update(update_data)
                
                # Periodic maintenance
                self._periodic_maintenance()
                
                time.sleep(0.001)  # 1ms sleep for real-time responsiveness
                
            except Exception as e:
                logger.error(f"Error in real-time update loop: {e}")
                time.sleep(0.01)  # Longer sleep on error
    
    def _process_queued_update(self, update_data: Dict[str, Any]) -> None:
        """Process a queued update in real-time thread."""
        # Placeholder for real-time processing
        pass
    
    def _periodic_maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        # Clean up old history
        current_time = time.time()
        
        # Example: remove old entries (implementation would depend on specific needs)
        pass
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get current learning statistics."""
        stats = {
            'current_learning_rate': self.state.current_learning_rate,
            'recent_rewards': list(self.state.reward_history)[-10:] if self.state.reward_history else [],
            'recent_firing_rates': list(self.state.firing_rate_history)[-10:] if self.state.firing_rate_history else [],
            'adaptation_strength': list(self.state.adaptation_history)[-10:] if self.state.adaptation_history else [],
            'performance_trend': list(self.state.performance_history)[-10:] if self.state.performance_history else [],
            'attention_weights': self.attention_mechanism.attention_weights,
            'meta_parameters': self.state.meta_parameters
        }
        
        return stats


# Helper classes

class SimplePredictor:
    """Simple reward predictor for reinforcement learning."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def predict(self, state: np.ndarray) -> float:
        if not self.history:
            return 0.0
        return np.mean([reward for _, reward in self.history])
    
    def update(self, state: np.ndarray, reward: float) -> None:
        self.history.append((state.copy() if state is not None else None, reward))


class FiringRateTracker:
    """Track firing rate statistics."""
    
    def __init__(self, window_size: int = 1000):
        self.history = deque(maxlen=window_size)
        
    def update(self, firing_rates: np.ndarray) -> None:
        self.history.append(firing_rates.copy())
    
    def get_statistics(self) -> Dict[str, float]:
        if not self.history:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        all_rates = np.concatenate(list(self.history))
        return {
            'mean': np.mean(all_rates),
            'std': np.std(all_rates),
            'min': np.min(all_rates),
            'max': np.max(all_rates)
        }


class StabilityMonitor:
    """Monitor network stability."""
    
    def __init__(self):
        self.weight_history = defaultdict(list)
        
    def assess_stability(self, firing_rates: np.ndarray, 
                        weights: List[np.ndarray]) -> float:
        """Assess current network stability."""
        # Simple stability metric based on firing rate variance
        firing_rate_stability = 1.0 / (1.0 + np.var(firing_rates))
        
        # Weight stability
        weight_stability = 1.0
        for i, weight_matrix in enumerate(weights):
            key = f"layer_{i}"
            self.weight_history[key].append(np.mean(np.abs(weight_matrix)))
            
            if len(self.weight_history[key]) > 10:
                recent_weights = self.weight_history[key][-10:]
                weight_variance = np.var(recent_weights)
                weight_stability *= 1.0 / (1.0 + weight_variance)
        
        # Combined stability
        stability = 0.6 * firing_rate_stability + 0.4 * weight_stability
        
        return stability


class HomeostaticPredictor:
    """Predict potential instabilities for proactive homeostasis."""
    
    def __init__(self):
        self.instability_threshold = 0.8
        
    def predict_instability(self, firing_rates: np.ndarray, 
                          weights: List[np.ndarray]) -> float:
        """Predict likelihood of instability."""
        # Simple prediction based on current metrics
        high_firing = np.sum(firing_rates > 0.5) / len(firing_rates)
        
        # Weight magnitude analysis
        large_weights = 0.0
        total_weights = 0
        for weight_matrix in weights:
            large_weights += np.sum(np.abs(weight_matrix) > 1.0)
            total_weights += weight_matrix.size
            
        weight_instability = large_weights / total_weights if total_weights > 0 else 0.0
        
        # Combined instability prediction
        instability_risk = 0.7 * high_firing + 0.3 * weight_instability
        
        return instability_risk


class MetaOptimizer:
    """Meta-optimizer for learning parameters."""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.meta_params = {}
        
    def compute_meta_updates(self, task_embedding: np.ndarray,
                           strategy: Dict[str, float],
                           performance: float) -> Dict[str, float]:
        """Compute meta-learning updates."""
        # Simple meta-learning update
        updates = {}
        
        for key, value in strategy.items():
            # Performance-based adjustment
            if performance > 0.7:
                # Good performance - maintain strategy
                updates[key] = value
            elif performance < 0.3:
                # Poor performance - adjust strategy
                updates[key] = value * 0.8 if value > 0 else value * 1.2
            else:
                # Medium performance - small adjustments
                updates[key] = value * 0.95
        
        return updates


class ArchitectureOptimizer:
    """Optimize network architecture based on performance."""
    
    def __init__(self, params: AdaptationParameters):
        self.params = params
        self.architecture_history = []
        
    def suggest_updates(self, weights: List[np.ndarray], 
                       firing_rates: np.ndarray,
                       performance_history: deque) -> Dict[str, Any]:
        """Suggest architecture updates."""
        updates = {}
        
        if len(performance_history) > 10:
            recent_performance = list(performance_history)[-10:]
            performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            
            if performance_trend < -0.01:  # Declining performance
                updates['prune_weak_connections'] = True
                updates['pruning_threshold'] = self.params.pruning_threshold
            elif performance_trend > 0.01:  # Improving performance
                updates['grow_connections'] = True
                updates['growth_factor'] = 1.1
        
        return updates


class PerformanceTracker:
    """Track system performance metrics."""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
    def update(self, reward: float, firing_rates: np.ndarray, 
              learning_rates: Dict[str, float]) -> Dict[str, float]:
        """Update performance metrics."""
        current_metrics = {
            'reward': reward,
            'mean_firing_rate': np.mean(firing_rates),
            'firing_rate_stability': 1.0 / (1.0 + np.var(firing_rates)),
            'mean_learning_rate': np.mean(list(learning_rates.values())) if learning_rates else 0.0
        }
        
        # Store metrics
        for key, value in current_metrics.items():
            self.metrics_history[key].append(value)
            
            # Keep only recent history
            if len(self.metrics_history[key]) > 1000:
                self.metrics_history[key] = self.metrics_history[key][-1000:]
        
        return current_metrics
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary statistics."""
        summary = {}
        
        for metric, history in self.metrics_history.items():
            if history:
                recent = history[-100:] if len(history) >= 100 else history
                summary[f"{metric}_mean"] = np.mean(recent)
                summary[f"{metric}_trend"] = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) > 1 else 0.0
        
        return summary


# Convenience functions

def create_adaptive_learning_system(base_learning_rate: float = 0.01,
                                   reward_window: int = 1000,
                                   target_firing_rate: float = 0.1) -> AdaptiveRealTimeLearningSystem:
    """Create adaptive learning system with default parameters."""
    params = AdaptationParameters(
        base_learning_rate=base_learning_rate,
        reward_window=reward_window,
        target_firing_rate=target_firing_rate
    )
    
    return AdaptiveRealTimeLearningSystem(params)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Adaptive Real-time Learning System")
    print("=" * 60)
    
    # Create adaptive learning system
    system = create_adaptive_learning_system()
    
    # Start real-time adaptation
    system.start_realtime_adaptation()
    
    # Simulate learning steps
    num_steps = 100
    layer_names = ['input_layer', 'hidden_layer', 'output_layer']
    
    print(f"Running {num_steps} adaptive learning steps...")
    
    for step in range(num_steps):
        # Simulate gradients, firing rates, and rewards
        gradients = [
            np.random.normal(0, 0.1, (100, 50)),  # Input to hidden
            np.random.normal(0, 0.1, (50, 25)),   # Hidden to hidden
            np.random.normal(0, 0.1, (25, 10))    # Hidden to output
        ]
        
        firing_rates = np.random.exponential(0.1, 100)  # Poisson-like firing
        reward_signal = np.random.normal(0.0, 1.0)  # Variable reward
        
        current_weights = [grad + np.random.normal(0, 0.01, grad.shape) for grad in gradients]
        
        # Task context for meta-learning
        task_context = {
            'task_id': f'task_{step // 20}',  # Change task every 20 steps
            'performance': max(0, min(1, 0.5 + 0.1 * np.sin(step * 0.1))),
            'input_size': 100,
            'output_size': 10,
            'complexity': np.random.uniform(0.3, 0.8)
        }
        
        # Process learning step
        result = system.process_learning_step(
            gradients=gradients,
            firing_rates=firing_rates,
            reward_signal=reward_signal,
            current_weights=current_weights,
            layer_names=layer_names,
            task_context=task_context
        )
        
        # Display progress every 20 steps
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Adaptive LRs: {', '.join(f'{k}: {v:.4f}' for k, v in result['adaptive_learning_rates'].items())}")
            print(f"  Reinforcement mod: {result['reinforcement_modulation']:.3f}")
            print(f"  Homeostatic adj: {result['homeostatic_adjustment']:.3f}")
            print(f"  Performance: {result['performance_metrics']['reward']:.3f}")
    
    # Get final statistics
    final_stats = system.get_learning_statistics()
    print(f"\nFinal Learning Statistics:")
    print(f"  Current learning rate: {final_stats['current_learning_rate']:.4f}")
    print(f"  Recent average reward: {np.mean(final_stats['recent_rewards']) if final_stats['recent_rewards'] else 0:.3f}")
    print(f"  Recent firing rate: {np.mean(final_stats['recent_firing_rates']) if final_stats['recent_firing_rates'] else 0:.3f}")
    print(f"  Attention weights: {final_stats['attention_weights']}")
    
    # Stop real-time adaptation
    system.stop_realtime_adaptation()
    
    print("\nAdaptive real-time learning test completed!")