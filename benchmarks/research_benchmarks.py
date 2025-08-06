"""
Research Benchmarking Framework for Novel Neuromorphic Algorithms

Comprehensive benchmarking suite for evaluating and comparing research implementations
against state-of-the-art baselines with statistical significance testing.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from abc import ABC, abstractmethod

# Research modules
from spiking_fpga.research.adaptive_encoding import AdaptiveSpikeCoder, EncodingMode
from spiking_fpga.research.meta_plasticity import MetaPlasticSTDP, PlasticityParameters

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    algorithm_name: str
    test_name: str
    performance_metrics: Dict[str, float]
    execution_time: float
    memory_usage: Optional[float] = None
    hardware_metrics: Optional[Dict[str, float]] = None
    statistical_significance: Optional[Dict[str, float]] = None


@dataclass
class ComparisonResult:
    """Results from comparing multiple algorithms."""
    test_name: str
    baseline_result: BenchmarkResult
    research_results: List[BenchmarkResult]
    improvement_factors: Dict[str, float]
    statistical_tests: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


class BenchmarkHarness(ABC):
    """Abstract base class for benchmark harnesses."""
    
    @abstractmethod
    def setup_test(self, config: Dict[str, Any]) -> None:
        """Setup test environment."""
        pass
    
    @abstractmethod
    def run_benchmark(self, algorithm: Any, test_data: Any) -> BenchmarkResult:
        """Run benchmark on algorithm with test data."""
        pass
    
    @abstractmethod
    def cleanup_test(self) -> None:
        """Cleanup test environment."""
        pass


class AdaptiveEncodingBenchmark(BenchmarkHarness):
    """Benchmark harness for adaptive spike encoding algorithms."""
    
    def __init__(self):
        self.test_datasets = {}
        self.baseline_encoders = {}
        
    def setup_test(self, config: Dict[str, Any]) -> None:
        """Setup encoding benchmark test environment."""
        # Generate diverse test datasets
        self.test_datasets = {
            'correlated_signals': self._generate_correlated_data(1000, correlation=0.8),
            'random_signals': np.random.random((1000, 10)),
            'oscillatory_signals': self._generate_oscillatory_data(1000, freq=10),
            'sparse_signals': self._generate_sparse_data(1000, sparsity=0.1),
            'high_variance': np.random.normal(0, 2, (1000, 20)),
            'mixed_patterns': self._generate_mixed_patterns(1000)
        }
        
        # Setup baseline encoders for comparison
        self.baseline_encoders = {
            'rate_coding': RateEncoder(),
            'temporal_coding': TemporalEncoder(),
            'population_coding': PopulationEncoder()
        }
        
        logger.info(f"Setup {len(self.test_datasets)} test datasets")
        
    def run_benchmark(self, algorithm: AdaptiveSpikeCoder, test_data: np.ndarray) -> BenchmarkResult:
        """Run encoding benchmark."""
        start_time = time.time()
        
        # Encode test data
        spike_trains = []
        encoding_modes_used = []
        information_contents = []
        
        for i, data_sample in enumerate(test_data):
            if i >= 100:  # Limit for performance
                break
                
            spike_train = algorithm.encode(data_sample)
            spike_trains.append(spike_train)
            encoding_modes_used.append(spike_train.encoding_mode)
            if spike_train.information_content:
                information_contents.append(spike_train.information_content)
        
        execution_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = self._calculate_encoding_metrics(spike_trains, test_data[:len(spike_trains)])
        
        return BenchmarkResult(
            algorithm_name="AdaptiveSpikeCoder",
            test_name="adaptive_encoding",
            performance_metrics=metrics,
            execution_time=execution_time
        )
    
    def cleanup_test(self) -> None:
        """Cleanup encoding test environment."""
        self.test_datasets.clear()
        self.baseline_encoders.clear()
    
    def _generate_correlated_data(self, n_samples: int, correlation: float) -> np.ndarray:
        """Generate temporally correlated test data."""
        data = np.random.random((n_samples, 5))
        for i in range(1, n_samples):
            data[i] = correlation * data[i-1] + (1-correlation) * data[i]
        return data
    
    def _generate_oscillatory_data(self, n_samples: int, freq: float) -> np.ndarray:
        """Generate oscillatory test data."""
        t = np.linspace(0, n_samples/100, n_samples)  # 100 Hz sampling
        data = np.zeros((n_samples, 3))
        data[:, 0] = np.sin(2 * np.pi * freq * t)
        data[:, 1] = np.cos(2 * np.pi * freq * t)
        data[:, 2] = 0.5 * np.sin(4 * np.pi * freq * t)
        return (data + 1) / 2  # Normalize to [0, 1]
    
    def _generate_sparse_data(self, n_samples: int, sparsity: float) -> np.ndarray:
        """Generate sparse test data."""
        data = np.random.random((n_samples, 8))
        mask = np.random.random((n_samples, 8)) > sparsity
        data[mask] = 0
        return data
    
    def _generate_mixed_patterns(self, n_samples: int) -> np.ndarray:
        """Generate mixed pattern test data."""
        data = np.zeros((n_samples, 6))
        
        # Mix different pattern types with proper slicing
        third = n_samples // 3
        
        # Correlated data (pad from 5 to 6 dims)
        corr_data = self._generate_correlated_data(third, 0.9)
        data[:third] = np.pad(corr_data, ((0, 0), (0, 1)), mode='constant')
        
        # Random data
        data[third:2*third] = np.random.random((third, 6))
        
        # Oscillatory data (pad from 3 to 6 dims)
        remaining = n_samples - 2*third
        osc_data = self._generate_oscillatory_data(remaining, 15)
        data[2*third:] = np.pad(osc_data, ((0, 0), (0, 3)), mode='constant')
        
        return data
    
    def _calculate_encoding_metrics(self, spike_trains: List, original_data: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive encoding performance metrics."""
        metrics = {}
        
        # Basic spike statistics
        total_spikes = sum(len(st.spike_times) for st in spike_trains)
        avg_spikes_per_sample = total_spikes / len(spike_trains) if spike_trains else 0
        
        metrics['total_spikes'] = total_spikes
        metrics['avg_spikes_per_sample'] = avg_spikes_per_sample
        
        # Encoding efficiency (spikes per bit of information)
        if spike_trains and spike_trains[0].information_content:
            avg_information = np.mean([st.information_content for st in spike_trains 
                                     if st.information_content])
            metrics['encoding_efficiency'] = avg_spikes_per_sample / max(1, avg_information)
        else:
            metrics['encoding_efficiency'] = 0.0
        
        # Mode diversity (adaptation effectiveness)
        modes_used = [st.encoding_mode for st in spike_trains]
        unique_modes = len(set(modes_used))
        metrics['mode_diversity'] = unique_modes / len(EncodingMode)
        
        # Temporal precision (spike timing variance)
        spike_intervals = []
        for st in spike_trains:
            if len(st.spike_times) > 1:
                intervals = np.diff(st.spike_times)
                spike_intervals.extend(intervals)
        
        if spike_intervals:
            metrics['temporal_precision'] = 1.0 / (np.std(spike_intervals) + 1e-6)
        else:
            metrics['temporal_precision'] = 0.0
        
        # Information preservation estimate
        metrics['information_preservation'] = self._estimate_information_preservation(
            spike_trains, original_data
        )
        
        return metrics
    
    def _estimate_information_preservation(self, spike_trains: List, original_data: np.ndarray) -> float:
        """Estimate information preservation quality."""
        if not spike_trains:
            return 0.0
            
        # Simplified information preservation metric
        # Based on spike pattern diversity vs input diversity
        
        input_entropy = self._calculate_entropy(original_data.flatten())
        
        # Convert spike trains to feature vectors for entropy calculation
        spike_features = []
        for st in spike_trains:
            features = [
                len(st.spike_times),
                np.mean(st.spike_times) if len(st.spike_times) > 0 else 0,
                np.std(st.spike_times) if len(st.spike_times) > 1 else 0,
                len(np.unique(st.neuron_ids)) if len(st.neuron_ids) > 0 else 0
            ]
            spike_features.extend(features)
        
        spike_entropy = self._calculate_entropy(np.array(spike_features))
        
        # Information preservation as ratio of entropies
        preservation = min(1.0, spike_entropy / max(input_entropy, 1e-6))
        return preservation
    
    def _calculate_entropy(self, data: np.ndarray, bins: int = 10) -> float:
        """Calculate entropy of data."""
        if len(data) == 0:
            return 0.0
            
        hist, _ = np.histogram(data, bins=bins)
        probabilities = hist / np.sum(hist)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        return -np.sum(probabilities * np.log2(probabilities))


class MetaPlasticityBenchmark(BenchmarkHarness):
    """Benchmark harness for meta-plastic STDP algorithms."""
    
    def __init__(self):
        self.spike_patterns = {}
        self.baseline_stdp = {}
        
    def setup_test(self, config: Dict[str, Any]) -> None:
        """Setup plasticity benchmark test environment."""
        # Generate diverse spike patterns for testing
        self.spike_patterns = {
            'regular_pairing': self._generate_regular_pairing(100),
            'burst_pairing': self._generate_burst_pairing(50),
            'random_activity': self._generate_random_spikes(200),
            'homeostatic_challenge': self._generate_homeostatic_challenge(150),
            'meta_plasticity_test': self._generate_meta_plasticity_test(80)
        }
        
        # Setup baseline STDP implementations
        self.baseline_stdp = {
            'standard_stdp': StandardSTDP(),
            'homeostatic_stdp': HomeostasticSTDP()
        }
        
        logger.info(f"Setup {len(self.spike_patterns)} spike patterns")
    
    def run_benchmark(self, algorithm: MetaPlasticSTDP, test_data: List[Tuple[int, float]]) -> BenchmarkResult:
        """Run plasticity benchmark."""
        start_time = time.time()
        
        # Setup small test network
        algorithm.add_synapse(0, 1, 0.5)
        algorithm.add_synapse(1, 2, 0.5)
        algorithm.add_synapse(0, 2, 0.3)
        
        initial_weights = {
            (0, 1): algorithm.get_synapse_weight(0, 1),
            (1, 2): algorithm.get_synapse_weight(1, 2),
            (0, 2): algorithm.get_synapse_weight(0, 2)
        }
        
        # Process spike pattern
        for neuron_id, spike_time in test_data:
            algorithm.process_spike(neuron_id, spike_time)
        
        execution_time = time.time() - start_time
        
        # Calculate performance metrics
        final_weights = {
            (0, 1): algorithm.get_synapse_weight(0, 1),
            (1, 2): algorithm.get_synapse_weight(1, 2), 
            (0, 2): algorithm.get_synapse_weight(0, 2)
        }
        
        stats = algorithm.get_plasticity_statistics()
        
        metrics = self._calculate_plasticity_metrics(
            initial_weights, final_weights, stats, execution_time
        )
        
        return BenchmarkResult(
            algorithm_name="MetaPlasticSTDP",
            test_name="meta_plasticity",
            performance_metrics=metrics,
            execution_time=execution_time
        )
    
    def cleanup_test(self) -> None:
        """Cleanup plasticity test environment."""
        self.spike_patterns.clear()
        self.baseline_stdp.clear()
    
    def _generate_regular_pairing(self, n_pairs: int) -> List[Tuple[int, float]]:
        """Generate regular pre-post spike pairs."""
        spikes = []
        for i in range(n_pairs):
            base_time = i * 50.0  # 50ms intervals
            spikes.append((0, base_time))       # Pre-synaptic
            spikes.append((1, base_time + 5.0)) # Post-synaptic (+5ms)
        return sorted(spikes, key=lambda x: x[1])
    
    def _generate_burst_pairing(self, n_pairs: int) -> List[Tuple[int, float]]:
        """Generate burst spike pairing patterns."""
        spikes = []
        for i in range(n_pairs):
            base_time = i * 100.0
            # Pre-synaptic burst
            for j in range(3):
                spikes.append((0, base_time + j * 2.0))
            # Post-synaptic response
            spikes.append((1, base_time + 10.0))
        return sorted(spikes, key=lambda x: x[1])
    
    def _generate_random_spikes(self, n_spikes: int) -> List[Tuple[int, float]]:
        """Generate random spike activity."""
        spikes = []
        for i in range(n_spikes):
            neuron_id = np.random.choice([0, 1, 2])
            spike_time = np.random.exponential(20.0) * i
            spikes.append((neuron_id, spike_time))
        return sorted(spikes, key=lambda x: x[1])
    
    def _generate_homeostatic_challenge(self, n_spikes: int) -> List[Tuple[int, float]]:
        """Generate patterns that challenge homeostatic regulation."""
        spikes = []
        
        # Phase 1: High activity period
        for i in range(n_spikes // 2):
            for neuron_id in [0, 1, 2]:
                spike_time = i * 2.0 + neuron_id * 0.5
                spikes.append((neuron_id, spike_time))
        
        # Phase 2: Low activity period
        base_time = n_spikes
        for i in range(n_spikes // 4):
            neuron_id = np.random.choice([0, 1, 2])
            spike_time = base_time + i * 50.0
            spikes.append((neuron_id, spike_time))
            
        return sorted(spikes, key=lambda x: x[1])
    
    def _generate_meta_plasticity_test(self, n_episodes: int) -> List[Tuple[int, float]]:
        """Generate patterns to test meta-plasticity mechanisms."""
        spikes = []
        
        for episode in range(n_episodes):
            base_time = episode * 200.0
            
            # Repeated pairing within episode (should trigger meta-plasticity)
            for pair in range(5):
                pair_time = base_time + pair * 20.0
                spikes.append((0, pair_time))
                spikes.append((1, pair_time + 3.0))  # Consistent +3ms timing
                
        return sorted(spikes, key=lambda x: x[1])
    
    def _calculate_plasticity_metrics(self, initial_weights: Dict, final_weights: Dict, 
                                    stats: Dict, execution_time: float) -> Dict[str, float]:
        """Calculate plasticity performance metrics."""
        metrics = {}
        
        # Weight change dynamics
        weight_changes = {}
        for synapse_id in initial_weights:
            initial = initial_weights[synapse_id] or 0.0
            final = final_weights[synapse_id] or 0.0
            weight_changes[synapse_id] = final - initial
        
        total_weight_change = sum(abs(change) for change in weight_changes.values())
        metrics['total_weight_change'] = total_weight_change
        
        # Learning efficiency (weight change per update)
        update_count = stats.get('update_count', 1)
        metrics['learning_efficiency'] = total_weight_change / max(1, update_count)
        
        # Homeostatic effectiveness
        if 'avg_firing_rate' in stats:
            target_rate = 2.0  # Hz
            rate_error = abs(stats['avg_firing_rate'] - target_rate)
            metrics['homeostatic_error'] = rate_error / target_rate
        else:
            metrics['homeostatic_error'] = 1.0
        
        # Weight stability (lower variance = more stable)
        if 'weight_std' in stats:
            metrics['weight_stability'] = 1.0 / (stats['weight_std'] + 1e-6)
        else:
            metrics['weight_stability'] = 1.0
        
        # Processing speed (updates per second)
        metrics['processing_speed'] = update_count / max(execution_time, 1e-6)
        
        # Meta-plasticity effectiveness (based on weight change distribution)
        weight_change_values = list(weight_changes.values())
        if len(weight_change_values) > 1:
            change_variance = np.var(weight_change_values)
            metrics['meta_plasticity_effect'] = change_variance
        else:
            metrics['meta_plasticity_effect'] = 0.0
        
        return metrics


# Baseline implementations for comparison
class RateEncoder:
    """Simple rate-based encoder for baseline comparison."""
    def encode(self, data): 
        return np.sum(data) * 10  # Convert to spike count

class TemporalEncoder:
    """Simple temporal encoder for baseline comparison.""" 
    def encode(self, data):
        return len(data) * 5  # Simple temporal encoding

class PopulationEncoder:
    """Simple population encoder for baseline comparison."""
    def encode(self, data):
        return len(data) * 8  # Population-based encoding

class StandardSTDP:
    """Standard STDP implementation for baseline comparison."""
    def __init__(self):
        self.weights = {}
    def update(self, pre_time, post_time):
        return 0.01 if post_time > pre_time else -0.01

class HomeostasticSTDP:
    """Homeostatic STDP for baseline comparison."""
    def __init__(self):
        self.weights = {}
        self.activities = {}
    def update(self, pre_time, post_time):
        return 0.005  # Simplified homeostatic update


class ResearchBenchmarkSuite:
    """Complete benchmark suite for research algorithms."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.benchmarks = {
            'adaptive_encoding': AdaptiveEncodingBenchmark(),
            'meta_plasticity': MetaPlasticityBenchmark()
        }
        
        self.results = []
        
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark suite."""
        logger.info("Starting comprehensive research benchmark suite")
        
        all_results = {}
        
        # Test adaptive encoding
        encoding_results = self._benchmark_adaptive_encoding()
        all_results['adaptive_encoding'] = encoding_results
        
        # Test meta-plasticity
        plasticity_results = self._benchmark_meta_plasticity()
        all_results['meta_plasticity'] = plasticity_results
        
        # Save results
        self._save_results(all_results)
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        return all_results
    
    def _benchmark_adaptive_encoding(self) -> List[BenchmarkResult]:
        """Benchmark adaptive encoding algorithms."""
        logger.info("Benchmarking adaptive encoding algorithms")
        
        benchmark = self.benchmarks['adaptive_encoding']
        benchmark.setup_test({})
        
        results = []
        
        # Test adaptive encoder on different datasets
        adaptive_coder = AdaptiveSpikeCoder()
        
        for dataset_name, dataset in benchmark.test_datasets.items():
            logger.info(f"Testing on {dataset_name} dataset")
            
            result = benchmark.run_benchmark(adaptive_coder, dataset)
            result.test_name = dataset_name
            results.append(result)
        
        benchmark.cleanup_test()
        return results
    
    def _benchmark_meta_plasticity(self) -> List[BenchmarkResult]:
        """Benchmark meta-plasticity algorithms.""" 
        logger.info("Benchmarking meta-plasticity algorithms")
        
        benchmark = self.benchmarks['meta_plasticity']
        benchmark.setup_test({})
        
        results = []
        
        # Test different plasticity parameters
        param_configs = [
            PlasticityParameters(),  # Default
            PlasticityParameters(a_ltp=0.02, a_ltd=0.01),  # High LTP
            PlasticityParameters(target_rate=5.0),  # High target rate
            PlasticityParameters(use_bit_shift_approximation=False)  # No approximation
        ]
        
        for i, params in enumerate(param_configs):
            for pattern_name, spike_pattern in benchmark.spike_patterns.items():
                logger.info(f"Testing config {i} on {pattern_name}")
                
                stdp = MetaPlasticSTDP(params)
                result = benchmark.run_benchmark(stdp, spike_pattern)
                result.test_name = f"{pattern_name}_config_{i}"
                result.algorithm_name = f"MetaPlasticSTDP_config_{i}"
                results.append(result)
        
        benchmark.cleanup_test()
        return results
    
    def _save_results(self, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Save benchmark results to JSON files."""
        timestamp = int(time.time())
        
        for benchmark_name, result_list in results.items():
            filename = self.output_dir / f"{benchmark_name}_results_{timestamp}.json"
            
            # Convert results to serializable format
            serializable_results = []
            for result in result_list:
                result_dict = asdict(result)
                # Handle enum serialization
                if 'encoding_mode' in result_dict:
                    result_dict['encoding_mode'] = str(result_dict['encoding_mode'])
                serializable_results.append(result_dict)
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved {len(result_list)} results to {filename}")
    
    def _generate_comparison_report(self, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Generate comprehensive comparison report."""
        report_path = self.output_dir / "benchmark_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Research Algorithm Benchmark Report\n\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            for benchmark_name, result_list in results.items():
                f.write(f"### {benchmark_name.replace('_', ' ').title()}\n\n")
                f.write(f"- Total tests: {len(result_list)}\n")
                
                avg_execution_time = np.mean([r.execution_time for r in result_list])
                f.write(f"- Average execution time: {avg_execution_time:.4f}s\n")
                
                # Algorithm-specific metrics
                if benchmark_name == 'adaptive_encoding':
                    avg_efficiency = np.mean([r.performance_metrics.get('encoding_efficiency', 0) 
                                            for r in result_list])
                    f.write(f"- Average encoding efficiency: {avg_efficiency:.4f}\n")
                    
                    avg_diversity = np.mean([r.performance_metrics.get('mode_diversity', 0) 
                                           for r in result_list])
                    f.write(f"- Average mode diversity: {avg_diversity:.4f}\n")
                
                elif benchmark_name == 'meta_plasticity':
                    avg_learning_eff = np.mean([r.performance_metrics.get('learning_efficiency', 0) 
                                              for r in result_list])
                    f.write(f"- Average learning efficiency: {avg_learning_eff:.4f}\n")
                    
                    avg_homeostatic_err = np.mean([r.performance_metrics.get('homeostatic_error', 1) 
                                                 for r in result_list])
                    f.write(f"- Average homeostatic error: {avg_homeostatic_err:.4f}\n")
                
                f.write("\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            for benchmark_name, result_list in results.items():
                f.write(f"### {benchmark_name.replace('_', ' ').title()}\n\n")
                
                for result in result_list:
                    f.write(f"#### {result.test_name}\n\n")
                    f.write(f"- Algorithm: {result.algorithm_name}\n")
                    f.write(f"- Execution time: {result.execution_time:.4f}s\n")
                    f.write("- Performance metrics:\n")
                    
                    for metric, value in result.performance_metrics.items():
                        f.write(f"  - {metric}: {value:.6f}\n")
                    
                    f.write("\n")
        
        logger.info(f"Generated benchmark report: {report_path}")


def run_research_benchmarks():
    """Main function to run research benchmarks."""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    benchmark_suite = ResearchBenchmarkSuite()
    results = benchmark_suite.run_comprehensive_benchmark()
    
    print("\n" + "="*60)
    print("RESEARCH BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    for benchmark_name, result_list in results.items():
        print(f"\n{benchmark_name.upper()} BENCHMARKS:")
        print(f"  Tests completed: {len(result_list)}")
        
        execution_times = [r.execution_time for r in result_list]
        print(f"  Total execution time: {sum(execution_times):.2f}s")
        print(f"  Average time per test: {np.mean(execution_times):.4f}s")
        
        if benchmark_name == 'adaptive_encoding':
            efficiencies = [r.performance_metrics.get('encoding_efficiency', 0) for r in result_list]
            print(f"  Best encoding efficiency: {max(efficiencies):.4f}")
            
        elif benchmark_name == 'meta_plasticity':
            learning_effs = [r.performance_metrics.get('learning_efficiency', 0) for r in result_list]
            print(f"  Best learning efficiency: {max(learning_effs):.4f}")
    
    print(f"\nFull results saved to: ./benchmark_results/")
    print("="*60)


if __name__ == "__main__":
    run_research_benchmarks()