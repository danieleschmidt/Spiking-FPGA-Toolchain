"""Advanced performance optimization with ML-driven adaptation and caching."""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
import hashlib
import queue
import statistics
from enum import Enum
import math
import gc


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"
    ADAPTIVE = "adaptive"


class CacheStrategy(Enum):
    """Caching strategies for performance optimization."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"


@dataclass
class PerformanceProfile:
    """Performance profile for a compilation task."""
    task_type: str
    input_hash: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hits: int
    cache_misses: int
    optimization_level: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion."""
    suggestion_type: str
    description: str
    expected_improvement: float
    implementation_effort: str  # "low", "medium", "high"
    priority_score: float
    code_example: Optional[str] = None


class IntelligentCache:
    """ML-driven adaptive caching system with multiple strategies."""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 logger: Optional[logging.Logger] = None):
        self.max_size = max_size
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache storage
        self.cache_data: Dict[str, Any] = {}
        self.access_counts: Dict[str, int] = {}
        self.access_times: Dict[str, datetime] = {}
        self.cache_scores: Dict[str, float] = {}
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        # ML components
        self.access_patterns: Dict[str, List[float]] = {}
        self.prediction_accuracy = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info(f"Initialized IntelligentCache with {strategy.value} strategy")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent tracking."""
        with self._lock:
            if key in self.cache_data:
                # Update access patterns
                self._update_access_pattern(key)
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.access_times[key] = datetime.now()
                self.hit_count += 1
                
                self.logger.debug(f"Cache hit for key: {key[:20]}")
                return self.cache_data[key]
            else:
                self.miss_count += 1
                self.logger.debug(f"Cache miss for key: {key[:20]}")
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache with intelligent eviction."""
        with self._lock:
            # Check if cache is full
            if len(self.cache_data) >= self.max_size and key not in self.cache_data:
                self._evict_item()
            
            self.cache_data[key] = value
            self.access_counts[key] = 1
            self.access_times[key] = datetime.now()
            
            # Initialize access pattern
            if key not in self.access_patterns:
                self.access_patterns[key] = []
            
            # Calculate initial cache score
            self.cache_scores[key] = self._calculate_cache_score(key, value)
            
            self.logger.debug(f"Cached item with key: {key[:20]}")
    
    def _update_access_pattern(self, key: str) -> None:
        """Update access pattern for predictive caching."""
        now = datetime.now()
        
        if key in self.access_times:
            time_since_last = (now - self.access_times[key]).total_seconds()
            
            # Store inter-access time
            if key not in self.access_patterns:
                self.access_patterns[key] = []
            
            self.access_patterns[key].append(time_since_last)
            
            # Keep only recent patterns
            if len(self.access_patterns[key]) > 20:
                self.access_patterns[key] = self.access_patterns[key][-20:]
    
    def _calculate_cache_score(self, key: str, value: Any) -> float:
        """Calculate cache score for intelligent eviction."""
        score = 0.0
        
        # Recency score
        recency = (datetime.now() - self.access_times[key]).total_seconds()
        recency_score = 1.0 / (1.0 + recency / 3600)  # Decay over hours
        
        # Frequency score
        frequency_score = self.access_counts.get(key, 1) / 10.0
        
        # Size score (prefer smaller items)
        try:
            size_bytes = len(pickle.dumps(value))
            size_score = 1.0 / (1.0 + size_bytes / (1024 * 1024))  # MB normalization
        except:
            size_score = 0.5  # Default for non-serializable items
        
        # Predictive score based on access patterns
        predictive_score = self._calculate_predictive_score(key)
        
        # Weighted combination
        score = (0.3 * recency_score + 0.3 * frequency_score + 
                0.2 * size_score + 0.2 * predictive_score)
        
        return score
    
    def _calculate_predictive_score(self, key: str) -> float:
        """Calculate predictive score based on access patterns."""
        if key not in self.access_patterns or len(self.access_patterns[key]) < 3:
            return 0.5  # Neutral score for insufficient data
        
        pattern = self.access_patterns[key]
        
        # Calculate pattern regularity
        if len(pattern) >= 3:
            avg_interval = statistics.mean(pattern)
            std_interval = statistics.stdev(pattern) if len(pattern) > 1 else avg_interval
            
            # Regular patterns get higher scores
            regularity = 1.0 / (1.0 + std_interval / (avg_interval + 1))
            
            # Recent access prediction
            time_since_last = (datetime.now() - self.access_times[key]).total_seconds()
            predicted_next_access = avg_interval
            
            if abs(time_since_last - predicted_next_access) < avg_interval * 0.5:
                return regularity * 1.2  # Boost score if access is predicted soon
            else:
                return regularity * 0.8
        
        return 0.5
    
    def _evict_item(self) -> None:
        """Intelligently evict item based on strategy."""
        if not self.cache_data:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            oldest_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            
        elif self.strategy == CacheStrategy.ADAPTIVE or self.strategy == CacheStrategy.PREDICTIVE:
            # Evict lowest scoring item
            if self.cache_scores:
                # Recalculate scores before eviction
                for key in list(self.cache_scores.keys()):
                    if key in self.cache_data:
                        self.cache_scores[key] = self._calculate_cache_score(key, self.cache_data[key])
                
                oldest_key = min(self.cache_scores.keys(), key=lambda k: self.cache_scores[k])
            else:
                oldest_key = next(iter(self.cache_data.keys()))
        
        else:  # TTL or default
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove item
        del self.cache_data[oldest_key]
        del self.access_counts[oldest_key]
        del self.access_times[oldest_key]
        if oldest_key in self.cache_scores:
            del self.cache_scores[oldest_key]
        
        self.eviction_count += 1
        self.logger.debug(f"Evicted cache item: {oldest_key[:20]}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
            "cache_size": len(self.cache_data),
            "max_size": self.max_size,
            "strategy": self.strategy.value
        }
    
    def predict_future_accesses(self, horizon_minutes: int = 60) -> List[str]:
        """Predict which cache items will be accessed in the future."""
        predictions = []
        horizon_seconds = horizon_minutes * 60
        
        for key, pattern in self.access_patterns.items():
            if len(pattern) >= 3 and key in self.cache_data:
                avg_interval = statistics.mean(pattern)
                time_since_last = (datetime.now() - self.access_times[key]).total_seconds()
                
                # Predict next access time
                predicted_next = avg_interval - time_since_last
                
                if 0 < predicted_next <= horizon_seconds:
                    confidence = 1.0 / (1.0 + statistics.stdev(pattern) / avg_interval)
                    predictions.append((key, predicted_next, confidence))
        
        # Sort by predicted time and confidence
        predictions.sort(key=lambda x: (x[1], -x[2]))
        return [pred[0] for pred in predictions]
    
    def cleanup_expired(self, max_age_hours: float = 24) -> int:
        """Clean up expired cache entries."""
        expired_keys = []
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            for key, access_time in self.access_times.items():
                if access_time < cutoff_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self.cache_data:
                    del self.cache_data[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                if key in self.access_times:
                    del self.access_times[key]
                if key in self.cache_scores:
                    del self.cache_scores[key]
        
        self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)


class PerformanceProfiler:
    """Advanced performance profiler with ML-driven insights."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.profiles: List[PerformanceProfile] = []
        self.optimization_suggestions: List[OptimizationSuggestion] = []
        
        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        # ML components
        self.performance_models: Dict[str, Any] = {}
        
        self.logger.info("PerformanceProfiler initialized")
    
    def profile_execution(self, func: Callable, task_type: str, 
                         input_data: Any = None, context: Dict[str, Any] = None) -> Tuple[Any, PerformanceProfile]:
        """Profile function execution with detailed metrics."""
        import psutil
        import os
        
        # Generate input hash
        input_hash = self._generate_input_hash(input_data)
        
        # Get initial system state
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        initial_cpu_percent = process.cpu_percent()
        
        # Execute function with timing
        start_time = time.time()
        try:
            result = func(input_data) if input_data is not None else func()
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Get final system state
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        final_cpu_percent = process.cpu_percent()
        
        memory_usage = max(final_memory - initial_memory, 0)
        cpu_usage = (initial_cpu_percent + final_cpu_percent) / 2
        
        # Create performance profile
        profile = PerformanceProfile(
            task_type=task_type,
            input_hash=input_hash,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            cache_hits=0,  # Will be updated by cache
            cache_misses=0,  # Will be updated by cache
            optimization_level="none",
            timestamp=datetime.now(),
            context=context or {}
        )
        
        # Add success/error information
        profile.context["success"] = success
        if error:
            profile.context["error"] = error
        
        self.profiles.append(profile)
        
        # Update baselines
        self._update_baselines(profile)
        
        # Generate optimization suggestions
        self._analyze_performance_and_suggest(profile)
        
        self.logger.info(f"Profiled {task_type}: {execution_time:.3f}s, {memory_usage:.1f}MB")
        
        return result, profile
    
    def _generate_input_hash(self, input_data: Any) -> str:
        """Generate hash for input data."""
        try:
            if input_data is None:
                return "none"
            
            # Convert to string representation
            if hasattr(input_data, '__dict__'):
                data_str = str(sorted(input_data.__dict__.items()))
            elif isinstance(input_data, (dict, list, tuple)):
                data_str = str(input_data)
            else:
                data_str = str(input_data)
            
            return hashlib.md5(data_str.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def _update_baselines(self, profile: PerformanceProfile) -> None:
        """Update performance baselines for task types."""
        task_type = profile.task_type
        
        if task_type not in self.baselines:
            self.baselines[task_type] = {
                "execution_time": [],
                "memory_usage": [],
                "cpu_usage": []
            }
        
        # Add new measurements
        self.baselines[task_type]["execution_time"].append(profile.execution_time)
        self.baselines[task_type]["memory_usage"].append(profile.memory_usage)
        self.baselines[task_type]["cpu_usage"].append(profile.cpu_usage)
        
        # Keep only recent measurements
        for metric in self.baselines[task_type]:
            if len(self.baselines[task_type][metric]) > 100:
                self.baselines[task_type][metric] = self.baselines[task_type][metric][-100:]
    
    def _analyze_performance_and_suggest(self, profile: PerformanceProfile) -> None:
        """Analyze performance and generate optimization suggestions."""
        task_type = profile.task_type
        suggestions = []
        
        # Get baseline statistics
        if task_type in self.baselines:
            baseline_times = self.baselines[task_type]["execution_time"]
            baseline_memory = self.baselines[task_type]["memory_usage"]
            
            if len(baseline_times) >= 5:
                avg_time = statistics.mean(baseline_times)
                avg_memory = statistics.mean(baseline_memory)
                
                # Check for performance regressions
                if profile.execution_time > avg_time * 1.5:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type="performance_regression",
                        description="Execution time significantly higher than baseline",
                        expected_improvement=50.0,
                        implementation_effort="medium",
                        priority_score=8.0,
                        code_example="Consider caching intermediate results or algorithm optimization"
                    ))
                
                # Check for memory issues
                if profile.memory_usage > avg_memory * 2.0:
                    suggestions.append(OptimizationSuggestion(
                        suggestion_type="memory_optimization",
                        description="Memory usage significantly higher than baseline",
                        expected_improvement=30.0,
                        implementation_effort="medium",
                        priority_score=7.0,
                        code_example="Consider streaming processing or memory pooling"
                    ))
        
        # Task-specific suggestions
        if task_type == "hdl_generation":
            if profile.execution_time > 10.0:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="parallelization",
                    description="HDL generation could benefit from parallel processing",
                    expected_improvement=60.0,
                    implementation_effort="high",
                    priority_score=9.0,
                    code_example="Use ThreadPoolExecutor to parallelize module generation"
                ))
        
        elif task_type == "optimization":
            if profile.execution_time > 30.0:
                suggestions.append(OptimizationSuggestion(
                    suggestion_type="algorithm_improvement",
                    description="Optimization taking too long, consider heuristics",
                    expected_improvement=70.0,
                    implementation_effort="high",
                    priority_score=8.5,
                    code_example="Implement early termination or approximation algorithms"
                ))
        
        # Add suggestions to list
        self.optimization_suggestions.extend(suggestions)
        
        # Keep only recent suggestions
        if len(self.optimization_suggestions) > 50:
            self.optimization_suggestions = self.optimization_suggestions[-50:]
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get comprehensive performance insights."""
        if not self.profiles:
            return {"status": "no_data"}
        
        # Calculate overall statistics
        total_executions = len(self.profiles)
        recent_profiles = self.profiles[-20:]  # Last 20 executions
        
        avg_execution_time = statistics.mean([p.execution_time for p in recent_profiles])
        avg_memory_usage = statistics.mean([p.memory_usage for p in recent_profiles])
        
        # Task type breakdown
        task_breakdown = {}
        for profile in recent_profiles:
            task_type = profile.task_type
            if task_type not in task_breakdown:
                task_breakdown[task_type] = {
                    "count": 0,
                    "avg_time": 0,
                    "avg_memory": 0
                }
            
            task_breakdown[task_type]["count"] += 1
            task_breakdown[task_type]["avg_time"] += profile.execution_time
            task_breakdown[task_type]["avg_memory"] += profile.memory_usage
        
        # Calculate averages
        for task_type, stats in task_breakdown.items():
            count = stats["count"]
            stats["avg_time"] /= count
            stats["avg_memory"] /= count
        
        # Top optimization opportunities
        top_suggestions = sorted(self.optimization_suggestions, 
                               key=lambda x: x.priority_score, reverse=True)[:5]
        
        return {
            "total_executions": total_executions,
            "average_execution_time": avg_execution_time,
            "average_memory_usage": avg_memory_usage,
            "task_breakdown": task_breakdown,
            "top_optimization_suggestions": [
                {
                    "type": s.suggestion_type,
                    "description": s.description,
                    "expected_improvement": f"{s.expected_improvement}%",
                    "effort": s.implementation_effort,
                    "priority": s.priority_score
                } for s in top_suggestions
            ]
        }
    
    def benchmark_against_baseline(self, task_type: str) -> Dict[str, float]:
        """Benchmark current performance against baseline."""
        if task_type not in self.baselines:
            return {"status": "no_baseline"}
        
        recent_profiles = [p for p in self.profiles[-10:] if p.task_type == task_type]
        if not recent_profiles:
            return {"status": "no_recent_data"}
        
        baseline_times = self.baselines[task_type]["execution_time"]
        recent_times = [p.execution_time for p in recent_profiles]
        
        baseline_avg = statistics.mean(baseline_times)
        recent_avg = statistics.mean(recent_times)
        
        performance_ratio = recent_avg / baseline_avg if baseline_avg > 0 else 1.0
        performance_change = (performance_ratio - 1.0) * 100  # Percentage change
        
        return {
            "baseline_avg_time": baseline_avg,
            "recent_avg_time": recent_avg,
            "performance_change_percent": performance_change,
            "status": "improved" if performance_change < -5 else "degraded" if performance_change > 5 else "stable"
        }


class AdaptivePerformanceOptimizer:
    """ML-driven adaptive performance optimizer."""
    
    def __init__(self, cache: IntelligentCache, profiler: PerformanceProfiler,
                 logger: Optional[logging.Logger] = None):
        self.cache = cache
        self.profiler = profiler
        self.logger = logger or logging.getLogger(__name__)
        
        # Optimization state
        self.current_level = OptimizationLevel.BASIC
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Performance targets
        self.performance_targets = {
            "max_execution_time": 60.0,  # seconds
            "max_memory_usage": 1024.0,  # MB
            "min_cache_hit_rate": 0.7
        }
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.2
        
        self.logger.info("AdaptivePerformanceOptimizer initialized")
    
    def optimize_function(self, func: Callable, task_type: str, 
                         input_data: Any = None) -> Any:
        """Optimize function execution with adaptive strategies."""
        
        # Check cache first
        cache_key = self._generate_cache_key(func, input_data)
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            self.logger.debug(f"Retrieved {task_type} from cache")
            return cached_result
        
        # Apply optimization strategy based on current level
        optimized_func = self._apply_optimization_strategy(func, task_type)
        
        # Execute with profiling
        result, profile = self.profiler.profile_execution(
            optimized_func, task_type, input_data
        )
        
        # Cache result if successful
        if profile.context.get("success", True):
            self.cache.put(cache_key, result)
        
        # Adapt optimization level based on performance
        self._adapt_optimization_level(profile)
        
        return result
    
    def _generate_cache_key(self, func: Callable, input_data: Any) -> str:
        """Generate cache key for function and input."""
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        input_hash = self.profiler._generate_input_hash(input_data)
        return f"{func_name}_{input_hash}_{self.current_level.value}"
    
    def _apply_optimization_strategy(self, func: Callable, task_type: str) -> Callable:
        """Apply optimization strategy based on current level."""
        
        if self.current_level == OptimizationLevel.NONE:
            return func
        
        elif self.current_level == OptimizationLevel.BASIC:
            return self._apply_basic_optimizations(func, task_type)
        
        elif self.current_level == OptimizationLevel.AGGRESSIVE:
            return self._apply_aggressive_optimizations(func, task_type)
        
        elif self.current_level == OptimizationLevel.EXTREME:
            return self._apply_extreme_optimizations(func, task_type)
        
        elif self.current_level == OptimizationLevel.ADAPTIVE:
            return self._apply_adaptive_optimizations(func, task_type)
        
        else:
            return func
    
    def _apply_basic_optimizations(self, func: Callable, task_type: str) -> Callable:
        """Apply basic optimization strategies."""
        
        def optimized_func(*args, **kwargs):
            # Memory optimization: force garbage collection
            gc.collect()
            
            # Execute original function
            result = func(*args, **kwargs)
            
            # Clean up after execution
            gc.collect()
            
            return result
        
        return optimized_func
    
    def _apply_aggressive_optimizations(self, func: Callable, task_type: str) -> Callable:
        """Apply aggressive optimization strategies."""
        
        def optimized_func(*args, **kwargs):
            # Memory optimization
            gc.collect()
            
            # Parallel execution for suitable tasks
            if task_type in ["layer_optimization", "hdl_generation"]:
                return self._parallel_execute(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return optimized_func
    
    def _apply_extreme_optimizations(self, func: Callable, task_type: str) -> Callable:
        """Apply extreme optimization strategies."""
        
        def optimized_func(*args, **kwargs):
            # All previous optimizations plus:
            
            # Pre-compute and cache intermediate results
            intermediate_cache_key = f"intermediate_{task_type}_{hash(str(args))}"
            intermediate_result = self.cache.get(intermediate_cache_key)
            
            if intermediate_result is not None:
                self.logger.debug("Using cached intermediate result")
                return intermediate_result
            
            # Execute with all optimizations
            gc.collect()
            result = self._parallel_execute(func, *args, **kwargs)
            
            # Cache intermediate result
            self.cache.put(intermediate_cache_key, result)
            
            return result
        
        return optimized_func
    
    def _apply_adaptive_optimizations(self, func: Callable, task_type: str) -> Callable:
        """Apply ML-driven adaptive optimizations."""
        
        # Analyze historical performance for this task type
        task_profiles = [p for p in self.profiler.profiles if p.task_type == task_type]
        
        if len(task_profiles) >= 5:
            avg_time = statistics.mean([p.execution_time for p in task_profiles[-5:]])
            
            # Choose optimization level based on performance history
            if avg_time > 30.0:
                return self._apply_extreme_optimizations(func, task_type)
            elif avg_time > 10.0:
                return self._apply_aggressive_optimizations(func, task_type)
            else:
                return self._apply_basic_optimizations(func, task_type)
        else:
            return self._apply_basic_optimizations(func, task_type)
    
    def _parallel_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with parallel optimization where applicable."""
        
        # For now, just execute normally
        # In a real implementation, this would analyze the function
        # and apply parallel execution strategies
        return func(*args, **kwargs)
    
    def _adapt_optimization_level(self, profile: PerformanceProfile) -> None:
        """Adapt optimization level based on performance feedback."""
        
        # Check if performance meets targets
        meets_time_target = profile.execution_time <= self.performance_targets["max_execution_time"]
        meets_memory_target = profile.memory_usage <= self.performance_targets["max_memory_usage"]
        
        cache_stats = self.cache.get_statistics()
        meets_cache_target = cache_stats["hit_rate"] >= self.performance_targets["min_cache_hit_rate"]
        
        performance_score = sum([meets_time_target, meets_memory_target, meets_cache_target]) / 3.0
        
        # Adaptation logic
        if performance_score < 0.5 and self.current_level != OptimizationLevel.EXTREME:
            # Performance is poor, increase optimization level
            self._increase_optimization_level()
            self.logger.info(f"Increased optimization level to {self.current_level.value}")
        
        elif performance_score > 0.8 and self.current_level != OptimizationLevel.BASIC:
            # Performance is good, might decrease optimization level to save resources
            if len(self.optimization_history) >= 5:
                recent_scores = [h["performance_score"] for h in self.optimization_history[-5:]]
                if all(score > 0.8 for score in recent_scores):
                    self._decrease_optimization_level()
                    self.logger.info(f"Decreased optimization level to {self.current_level.value}")
        
        # Record adaptation history
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "optimization_level": self.current_level.value,
            "performance_score": performance_score,
            "execution_time": profile.execution_time,
            "memory_usage": profile.memory_usage
        })
        
        # Keep only recent history
        if len(self.optimization_history) > 50:
            self.optimization_history = self.optimization_history[-50:]
    
    def _increase_optimization_level(self) -> None:
        """Increase optimization level."""
        level_progression = [
            OptimizationLevel.NONE,
            OptimizationLevel.BASIC,
            OptimizationLevel.AGGRESSIVE,
            OptimizationLevel.EXTREME,
            OptimizationLevel.ADAPTIVE
        ]
        
        current_index = level_progression.index(self.current_level)
        if current_index < len(level_progression) - 1:
            self.current_level = level_progression[current_index + 1]
    
    def _decrease_optimization_level(self) -> None:
        """Decrease optimization level."""
        level_progression = [
            OptimizationLevel.NONE,
            OptimizationLevel.BASIC,
            OptimizationLevel.AGGRESSIVE,
            OptimizationLevel.EXTREME,
            OptimizationLevel.ADAPTIVE
        ]
        
        current_index = level_progression.index(self.current_level)
        if current_index > 1:  # Don't go below BASIC
            self.current_level = level_progression[current_index - 1]
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and recommendations."""
        cache_stats = self.cache.get_statistics()
        performance_insights = self.profiler.get_performance_insights()
        
        return {
            "current_optimization_level": self.current_level.value,
            "cache_statistics": cache_stats,
            "performance_insights": performance_insights,
            "adaptation_history": self.optimization_history[-10:],  # Last 10 adaptations
            "recommendations": self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        cache_stats = self.cache.get_statistics()
        
        if cache_stats["hit_rate"] < 0.5:
            recommendations.append("Consider increasing cache size or improving cache key generation")
        
        if len(self.profiler.profiles) >= 5:
            recent_profiles = self.profiler.profiles[-5:]
            avg_time = statistics.mean([p.execution_time for p in recent_profiles])
            avg_memory = statistics.mean([p.memory_usage for p in recent_profiles])
            
            if avg_time > 60.0:
                recommendations.append("Execution times are high - consider parallel processing")
            
            if avg_memory > 1024.0:
                recommendations.append("Memory usage is high - consider streaming or chunking")
        
        if not recommendations:
            recommendations.append("Performance appears optimal with current configuration")
        
        return recommendations


# Convenience function to create optimized performance system
def create_performance_optimization_system(cache_size: int = 1000,
                                         cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                                         logger: Optional[logging.Logger] = None) -> Tuple[IntelligentCache, PerformanceProfiler, AdaptivePerformanceOptimizer]:
    """Create a complete performance optimization system."""
    
    cache = IntelligentCache(max_size=cache_size, strategy=cache_strategy, logger=logger)
    profiler = PerformanceProfiler(logger=logger)
    optimizer = AdaptivePerformanceOptimizer(cache, profiler, logger=logger)
    
    return cache, profiler, optimizer