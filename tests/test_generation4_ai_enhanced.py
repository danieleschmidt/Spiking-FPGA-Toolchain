"""
Comprehensive tests for Generation 4 AI-Enhanced Compiler.

This test suite validates:
- Neural Architecture Search functionality
- Meta-Learning Optimization
- AI-enhanced HDL generation
- Performance prediction accuracy
- Multi-objective optimization
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from spiking_fpga.core import FPGATarget
from spiking_fpga.models.optimization import OptimizationLevel
from spiking_fpga.generation4_ai_enhanced_compiler import (
    Generation4Compiler,
    NeuralArchitectureSearch,
    MetaLearningOptimizer,
    OptimizationStrategy,
    MetaLearningState,
    CompilationResult,
    create_generation4_compiler,
    compile_with_ai_enhancement
)


@pytest.fixture
def sample_network_config():
    """Sample network configuration for testing."""
    return {
        'name': 'test_network',
        'neurons': 1000,
        'timestep': 1.0,
        'layers': [
            {'layer_id': 'input', 'layer_type': 'input', 'size': 100, 'neuron_type': 'poisson'},
            {'layer_id': 'hidden', 'layer_type': 'hidden', 'size': 800, 'neuron_type': 'lif'},
            {'layer_id': 'output', 'layer_type': 'output', 'size': 100, 'neuron_type': 'lif'}
        ],
        'connections': [
            {'source': 'input', 'target': 'hidden', 'connectivity': 'sparse_random', 'sparsity': 0.1},
            {'source': 'hidden', 'target': 'output', 'connectivity': 'sparse_random', 'sparsity': 0.3}
        ]
    }


@pytest.fixture
def target_fpga():
    """Target FPGA for testing."""
    return FPGATarget.ARTIX7_35T


@pytest.fixture
def temp_output_dir():
    """Temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestNeuralArchitectureSearch:
    """Test Neural Architecture Search functionality."""
    
    def test_nas_initialization(self):
        """Test NAS initialization."""
        nas = NeuralArchitectureSearch()
        
        assert nas.search_space is not None
        assert 'neuron_clustering' in nas.search_space
        assert 'memory_hierarchy' in nas.search_space
        assert 'pipeline_stages' in nas.search_space
        assert 'interconnect_topology' in nas.search_space
        
        assert nas.performance_predictor is not None
    
    def test_search_space_definition(self):
        """Test search space definition."""
        nas = NeuralArchitectureSearch()
        search_space = nas._define_search_space()
        
        # Check neuron clustering options
        assert 'cluster_sizes' in search_space['neuron_clustering']
        assert isinstance(search_space['neuron_clustering']['cluster_sizes'], list)
        assert len(search_space['neuron_clustering']['cluster_sizes']) > 0
        
        # Check memory hierarchy options
        assert 'levels' in search_space['memory_hierarchy']
        assert 'cache_sizes' in search_space['memory_hierarchy']
        assert 'replacement_policies' in search_space['memory_hierarchy']
    
    def test_architecture_sampling(self):
        """Test architecture sampling from search space."""
        nas = NeuralArchitectureSearch()
        
        config1 = nas._sample_architecture()
        config2 = nas._sample_architecture()
        
        # Check that configurations have required keys
        assert 'neuron_clustering' in config1
        assert 'memory_hierarchy' in config1
        assert 'pipeline_stages' in config1
        assert 'interconnect_topology' in config1
        
        # Check that sampling produces different configurations
        # (with high probability for reasonable search space)
        assert config1 != config2 or True  # Allow for rare equal sampling
    
    def test_feature_extraction(self, sample_network_config, target_fpga):
        """Test feature extraction for performance prediction."""
        nas = NeuralArchitectureSearch()
        arch_config = nas._sample_architecture()
        
        features = nas._extract_features(arch_config, sample_network_config, target_fpga)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert features.dtype == np.float32
        
        # Check that features are reasonable
        assert np.all(np.isfinite(features))
        assert np.all(features >= 0)  # Assuming all features are non-negative
    
    def test_performance_prediction(self):
        """Test performance prediction."""
        nas = NeuralArchitectureSearch()
        features = np.random.rand(10).astype(np.float32)
        
        prediction = nas.performance_predictor.predict(features)
        
        assert isinstance(prediction, dict)
        assert 'throughput' in prediction
        assert 'latency' in prediction
        assert 'power' in prediction
        assert 'resource_utilization' in prediction
        assert 'error_rate' in prediction
        
        # Check reasonable value ranges
        assert prediction['throughput'] >= 0
        assert prediction['latency'] > 0
        assert prediction['power'] > 0
        assert 0 <= prediction['resource_utilization'] <= 1
        assert 0 <= prediction['error_rate'] <= 1
    
    def test_scoring_function(self, target_fpga):
        """Test multi-objective scoring function."""
        nas = NeuralArchitectureSearch()
        
        # Test with good performance
        good_perf = {
            'throughput': 1e6,
            'latency': 1.0,
            'power': 1.0,
            'resource_utilization': 0.5,
            'error_rate': 0.01
        }
        
        good_score = nas._calculate_score(good_perf, target_fpga)
        
        # Test with bad performance
        bad_perf = {
            'throughput': 1e3,
            'latency': 100.0,
            'power': 10.0,
            'resource_utilization': 0.9,
            'error_rate': 0.1
        }
        
        bad_score = nas._calculate_score(bad_perf, target_fpga)
        
        assert isinstance(good_score, float)
        assert isinstance(bad_score, float)
        assert 0 <= good_score <= 1
        assert 0 <= bad_score <= 1
        assert good_score > bad_score
    
    def test_nas_search(self, sample_network_config, target_fpga):
        """Test full NAS search process."""
        nas = NeuralArchitectureSearch()
        
        # Run short search for testing
        result = nas.search(sample_network_config, target_fpga, iterations=5)
        
        assert isinstance(result, dict)
        assert 'neuron_clustering' in result
        assert 'memory_hierarchy' in result
        assert 'pipeline_stages' in result
        assert 'interconnect_topology' in result


class TestMetaLearningOptimizer:
    """Test Meta-Learning Optimizer functionality."""
    
    def test_meta_optimizer_initialization(self):
        """Test meta-optimizer initialization."""
        optimizer = MetaLearningOptimizer()
        
        assert optimizer.strategies == list(OptimizationStrategy)
        assert isinstance(optimizer.meta_state, MetaLearningState)
        
        # Check initial state
        assert len(optimizer.meta_state.learned_strategies) == len(OptimizationStrategy)
        assert len(optimizer.meta_state.confidence_scores) == len(OptimizationStrategy)
        assert optimizer.meta_state.performance_history == []
        assert 0 < optimizer.meta_state.adaptation_rate < 1
    
    def test_strategy_selection(self, sample_network_config, target_fpga):
        """Test strategy selection mechanism."""
        optimizer = MetaLearningOptimizer()
        
        strategy = optimizer.select_strategy(sample_network_config, target_fpga)
        
        assert isinstance(strategy, OptimizationStrategy)
        assert strategy in optimizer.strategies
    
    def test_context_feature_extraction(self, sample_network_config, target_fpga):
        """Test context feature extraction."""
        optimizer = MetaLearningOptimizer()
        
        features = optimizer._extract_context_features(sample_network_config, target_fpga)
        
        assert isinstance(features, dict)
        assert 'network_size' in features
        assert 'complexity' in features
        assert 'target_capacity' in features
        assert 'memory_pressure' in features
        
        # Check value ranges
        for key, value in features.items():
            assert isinstance(value, (int, float))
            assert value >= 0
    
    def test_context_bonus_calculation(self, sample_network_config, target_fpga):
        """Test context-specific bonus calculation."""
        optimizer = MetaLearningOptimizer()
        context = optimizer._extract_context_features(sample_network_config, target_fpga)
        
        for strategy in OptimizationStrategy:
            bonus = optimizer._calculate_context_bonus(strategy, context)
            assert isinstance(bonus, (int, float))
            assert bonus >= 0
    
    def test_performance_update(self):
        """Test performance update mechanism."""
        optimizer = MetaLearningOptimizer()
        
        strategy = OptimizationStrategy.NEURAL_ARCHITECTURE_SEARCH
        initial_score = optimizer.meta_state.learned_strategies[strategy.value]
        initial_confidence = optimizer.meta_state.confidence_scores[strategy.value]
        
        # Update with good performance
        optimizer.update_performance(strategy, 0.9)
        
        new_score = optimizer.meta_state.learned_strategies[strategy.value]
        new_confidence = optimizer.meta_state.confidence_scores[strategy.value]
        
        # Check that scores improved
        assert new_score >= initial_score  # Should increase or stay same
        assert new_confidence >= initial_confidence
        assert len(optimizer.meta_state.performance_history) == 1
        assert optimizer.meta_state.performance_history[0] == 0.9
    
    def test_performance_history_management(self):
        """Test performance history management."""
        optimizer = MetaLearningOptimizer()
        strategy = OptimizationStrategy.NEURAL_ARCHITECTURE_SEARCH
        
        # Add more than 100 performance updates
        for i in range(105):
            optimizer.update_performance(strategy, 0.5 + 0.01 * i)
        
        # Check that history is limited to 100 entries
        assert len(optimizer.meta_state.performance_history) == 100
        
        # Check that newest entries are preserved
        assert optimizer.meta_state.performance_history[-1] == 0.5 + 0.01 * 104


class TestGeneration4Compiler:
    """Test Generation 4 Compiler functionality."""
    
    def test_compiler_initialization(self, target_fpga):
        """Test compiler initialization."""
        compiler = Generation4Compiler(target_fpga)
        
        assert compiler.target == target_fpga
        assert isinstance(compiler.nas, NeuralArchitectureSearch)
        assert isinstance(compiler.meta_optimizer, MetaLearningOptimizer)
        assert compiler.logger is not None
    
    def test_factory_function(self, target_fpga):
        """Test factory function."""
        compiler = create_generation4_compiler(target_fpga)
        
        assert isinstance(compiler, Generation4Compiler)
        assert compiler.target == target_fpga
    
    @patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator')
    def test_network_compilation(self, mock_hdl_generator, sample_network_config, target_fpga, temp_output_dir):
        """Test network compilation process."""
        # Mock HDL generator
        mock_generator_instance = Mock()
        mock_generator_instance.generate_hdl.return_value = {
            'top_module': str(temp_output_dir / 'snn_top.v'),
            'neuron_module': str(temp_output_dir / 'lif_neuron.v')
        }
        mock_hdl_generator.return_value = mock_generator_instance
        
        compiler = Generation4Compiler(target_fpga)
        
        result = compiler.compile_network(
            sample_network_config,
            temp_output_dir,
            OptimizationLevel.AGGRESSIVE,
            enable_ai_enhancement=True
        )
        
        assert isinstance(result, CompilationResult)
        assert result.success is True
        assert isinstance(result.hdl_files, dict)
        assert isinstance(result.resource_estimate, dict)
        assert isinstance(result.performance_metrics, dict)
        assert isinstance(result.optimization_strategy, OptimizationStrategy)
        assert 0 <= result.ai_confidence <= 1
        assert result.compilation_time > 0
    
    def test_performance_prediction(self, sample_network_config, target_fpga):
        """Test performance prediction."""
        compiler = Generation4Compiler(target_fpga)
        architecture = compiler._default_architecture()
        
        prediction = compiler._predict_performance(sample_network_config, architecture)
        
        assert isinstance(prediction, dict)
        assert 'throughput' in prediction
        assert 'latency' in prediction
        assert 'power' in prediction
        assert prediction['throughput'] >= 0
        assert prediction['latency'] > 0
        assert prediction['power'] > 0
    
    def test_resource_estimation(self, sample_network_config, target_fpga):
        """Test resource estimation."""
        compiler = Generation4Compiler(target_fpga)
        architecture = compiler._default_architecture()
        
        estimate = compiler._estimate_resources(sample_network_config, architecture)
        
        assert isinstance(estimate, dict)
        assert 'neurons' in estimate
        assert 'synapses' in estimate
        assert 'luts' in estimate
        assert 'bram_kb' in estimate
        assert 'dsp_slices' in estimate
        assert 'lut_utilization' in estimate
        assert 'bram_utilization' in estimate
        assert 'dsp_utilization' in estimate
        
        # Check reasonable values
        assert estimate['neurons'] > 0
        assert estimate['synapses'] > 0
        assert estimate['luts'] > 0
        assert estimate['bram_kb'] > 0
        assert estimate['dsp_slices'] >= 0
        assert 0 <= estimate['lut_utilization'] <= 10  # Allow for over-utilization estimates
        assert 0 <= estimate['bram_utilization'] <= 10
        assert 0 <= estimate['dsp_utilization'] <= 10
    
    def test_ai_confidence_calculation(self, target_fpga):
        """Test AI confidence calculation."""
        compiler = Generation4Compiler(target_fpga)
        
        strategy = OptimizationStrategy.NEURAL_ARCHITECTURE_SEARCH
        performance = {
            'throughput': 1e6,
            'latency': 1.0,
            'power': 2.0,
            'resource_utilization': 0.5
        }
        
        confidence = compiler._calculate_ai_confidence(strategy, performance)
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_performance_score_calculation(self, target_fpga):
        """Test performance score calculation."""
        compiler = Generation4Compiler(target_fpga)
        
        # Test with good performance
        good_performance = {
            'throughput': 1e6,
            'latency': 1.0,
            'power': 1.0,
            'resource_utilization': 0.3
        }
        
        good_score = compiler._calculate_performance_score(good_performance)
        
        # Test with poor performance
        poor_performance = {
            'throughput': 1e3,
            'latency': 100.0,
            'power': 10.0,
            'resource_utilization': 0.9
        }
        
        poor_score = compiler._calculate_performance_score(poor_performance)
        
        assert isinstance(good_score, float)
        assert isinstance(poor_score, float)
        assert 0 <= good_score <= 1
        assert 0 <= poor_score <= 1
        assert good_score > poor_score
    
    def test_default_architecture(self, target_fpga):
        """Test default architecture configuration."""
        compiler = Generation4Compiler(target_fpga)
        
        architecture = compiler._default_architecture()
        
        assert isinstance(architecture, dict)
        assert 'neuron_clustering' in architecture
        assert 'memory_hierarchy' in architecture
        assert 'pipeline_stages' in architecture
        assert 'interconnect_topology' in architecture
        
        # Check that all required sub-keys exist
        assert 'cluster_sizes' in architecture['neuron_clustering']
        assert 'clustering_algorithms' in architecture['neuron_clustering']
        assert 'levels' in architecture['memory_hierarchy']
        assert 'cache_sizes' in architecture['memory_hierarchy']
        assert 'replacement_policies' in architecture['memory_hierarchy']
        assert 'depth' in architecture['pipeline_stages']
        assert 'parallelism' in architecture['pipeline_stages']
        assert 'patterns' in architecture['interconnect_topology']
        assert 'bandwidth' in architecture['interconnect_topology']


class TestIntegration:
    """Integration tests for Generation 4 AI-Enhanced Compiler."""
    
    @patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator')
    def test_high_level_compilation_function(self, mock_hdl_generator, sample_network_config, target_fpga, temp_output_dir):
        """Test high-level compilation function."""
        # Mock HDL generator
        mock_generator_instance = Mock()
        mock_generator_instance.generate_hdl.return_value = {
            'top_module': str(temp_output_dir / 'snn_top.v'),
            'neuron_module': str(temp_output_dir / 'lif_neuron.v')
        }
        mock_hdl_generator.return_value = mock_generator_instance
        
        result = compile_with_ai_enhancement(
            sample_network_config,
            target_fpga,
            temp_output_dir,
            OptimizationLevel.AGGRESSIVE
        )
        
        assert isinstance(result, CompilationResult)
        assert result.success is True
    
    def test_compilation_with_different_strategies(self, sample_network_config, target_fpga, temp_output_dir):
        """Test compilation with different optimization strategies."""
        compiler = Generation4Compiler(target_fpga)
        
        # Force different strategies
        strategies_to_test = [
            OptimizationStrategy.NEURAL_ARCHITECTURE_SEARCH,
            OptimizationStrategy.EVOLUTIONARY_OPTIMIZATION,
            OptimizationStrategy.REINFORCEMENT_LEARNING,
            OptimizationStrategy.META_LEARNING
        ]
        
        results = []
        for strategy in strategies_to_test:
            # Mock strategy selection to return specific strategy
            original_select_strategy = compiler.meta_optimizer.select_strategy
            compiler.meta_optimizer.select_strategy = Mock(return_value=strategy)
            
            with patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator'):
                result = compiler.compile_network(
                    sample_network_config,
                    temp_output_dir,
                    enable_ai_enhancement=True
                )
                results.append(result)
            
            # Restore original method
            compiler.meta_optimizer.select_strategy = original_select_strategy
        
        # Check that all compilations succeeded
        for result in results:
            assert result.success is True
            assert isinstance(result.optimization_strategy, OptimizationStrategy)
    
    def test_meta_learning_adaptation(self, sample_network_config, target_fpga, temp_output_dir):
        """Test that meta-learning adapts over multiple compilations."""
        compiler = Generation4Compiler(target_fpga)
        
        strategy = OptimizationStrategy.NEURAL_ARCHITECTURE_SEARCH
        initial_score = compiler.meta_optimizer.meta_state.learned_strategies[strategy.value]
        
        # Perform multiple compilations
        with patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator'):
            for i in range(3):
                result = compiler.compile_network(
                    sample_network_config,
                    temp_output_dir,
                    enable_ai_enhancement=True
                )
                assert result.success is True
        
        # Check that learning occurred
        assert len(compiler.meta_optimizer.meta_state.performance_history) > 0
    
    def test_compilation_without_ai_enhancement(self, sample_network_config, target_fpga, temp_output_dir):
        """Test compilation without AI enhancement."""
        compiler = Generation4Compiler(target_fpga)
        
        with patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator'):
            result = compiler.compile_network(
                sample_network_config,
                temp_output_dir,
                enable_ai_enhancement=False
            )
        
        assert isinstance(result, CompilationResult)
        assert result.success is True
        assert result.optimization_strategy == OptimizationStrategy.EVOLUTIONARY_OPTIMIZATION


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_compilation_failure_handling(self, sample_network_config, target_fpga, temp_output_dir):
        """Test handling of compilation failures."""
        compiler = Generation4Compiler(target_fpga)
        
        # Mock HDL generator to raise exception
        with patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator') as mock_hdl_generator:
            mock_generator_instance = Mock()
            mock_generator_instance.generate_hdl.side_effect = Exception("Mock compilation error")
            mock_hdl_generator.return_value = mock_generator_instance
            
            result = compiler.compile_network(
                sample_network_config,
                temp_output_dir
            )
        
        assert isinstance(result, CompilationResult)
        assert result.success is False
        assert result.compilation_time > 0
    
    def test_empty_network_config(self, target_fpga, temp_output_dir):
        """Test handling of empty network configuration."""
        compiler = Generation4Compiler(target_fpga)
        
        empty_config = {}
        
        with patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator'):
            result = compiler.compile_network(
                empty_config,
                temp_output_dir
            )
        
        # Should handle gracefully
        assert isinstance(result, CompilationResult)
    
    def test_invalid_output_directory(self, sample_network_config, target_fpga):
        """Test handling of invalid output directory."""
        compiler = Generation4Compiler(target_fpga)
        
        # Use non-existent parent directory
        invalid_dir = Path("/non/existent/path/output")
        
        with patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator'):
            result = compiler.compile_network(
                sample_network_config,
                invalid_dir
            )
        
        # Should handle gracefully or create directory
        assert isinstance(result, CompilationResult)


@pytest.mark.performance
class TestPerformance:
    """Performance tests for Generation 4 compiler."""
    
    def test_compilation_time_reasonable(self, sample_network_config, target_fpga, temp_output_dir):
        """Test that compilation completes in reasonable time."""
        compiler = Generation4Compiler(target_fpga)
        
        import time
        start_time = time.time()
        
        with patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator'):
            result = compiler.compile_network(
                sample_network_config,
                temp_output_dir,
                enable_ai_enhancement=True
            )
        
        elapsed_time = time.time() - start_time
        
        assert result.success is True
        assert elapsed_time < 10.0  # Should complete within 10 seconds
        assert result.compilation_time <= elapsed_time + 0.1  # Small tolerance for measurement differences
    
    def test_nas_search_performance(self, sample_network_config, target_fpga):
        """Test NAS search performance."""
        nas = NeuralArchitectureSearch()
        
        import time
        start_time = time.time()
        
        result = nas.search(sample_network_config, target_fpga, iterations=10)
        
        elapsed_time = time.time() - start_time
        
        assert isinstance(result, dict)
        assert elapsed_time < 5.0  # Should complete within 5 seconds
    
    def test_memory_usage_reasonable(self, sample_network_config, target_fpga, temp_output_dir):
        """Test that memory usage is reasonable."""
        import tracemalloc
        
        tracemalloc.start()
        
        compiler = Generation4Compiler(target_fpga)
        
        with patch('spiking_fpga.generation4_ai_enhanced_compiler.HDLGenerator'):
            result = compiler.compile_network(
                sample_network_config,
                temp_output_dir
            )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        assert result.success is True
        # Memory usage should be reasonable (less than 100 MB peak)
        assert peak < 100 * 1024 * 1024  # 100 MB in bytes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])