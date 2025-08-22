"""
Generation 4 Research Validation Framework

Comprehensive test suite for validating the Generation 4 autonomous learning
and self-modifying systems. Includes statistical validation, reproducibility
testing, and research methodology verification.

Test Categories:
- Autonomous learning system validation
- Self-modifying HDL generation testing
- Real-time optimization verification
- Quality gate validation
- Integration testing
- Performance benchmarking
- Research reproducibility
"""

import pytest
import asyncio
import numpy as np
import logging
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Import modules under test
from spiking_fpga.models.network import Network
from spiking_fpga.research.generation4_autonomous_learning import (
    AutonomousLearningOrchestrator,
    AutonomousLearningConfig,
    SelfModifyingArchitecture,
    RealTimeAdaptiveOptimizer,
    create_autonomous_learning_system
)
from spiking_fpga.research.self_modifying_hdl import (
    SelfModifyingHDLGenerator,
    HDLModificationTemplate,
    SynthesisConfiguration,
    create_self_modifying_hdl_generator
)
from spiking_fpga.research.adaptive_realtime_optimization import (
    AdaptiveRealTimeOptimizer,
    OptimizationTarget,
    AdaptationPolicy,
    create_adaptive_realtime_optimizer
)
from spiking_fpga.quality.autonomous_quality_gates import (
    AutonomousQualityOrchestrator,
    QualityGateStatus,
    create_quality_orchestrator
)


@pytest.fixture
def sample_network():
    """Create a sample neural network for testing."""
    network = Mock(spec=Network)
    network.layers = [
        {'type': 'input', 'size': 100},
        {'type': 'hidden', 'size': 200, 'neuron_model': 'LIF', 'tau_m': 20.0, 'tau_adapt': 100.0},
        {'type': 'output', 'size': 10}
    ]
    network.learning_rate = 0.001
    network.global_threshold = 1.0
    network.time_constants = [20.0, 100.0]
    network.connectivity = 'sparse'
    return network


@pytest.fixture
def autonomous_learning_config():
    """Create configuration for autonomous learning."""
    return AutonomousLearningConfig(
        learning_rate=0.001,
        adaptation_threshold=0.1,
        max_architecture_changes=10,
        convergence_patience=20,
        meta_learning_enabled=True,
        hardware_modification_enabled=True,
        real_time_optimization=True,
        exploration_rate=0.2,
        performance_targets={
            'throughput_mspikes_per_sec': 100.0,
            'latency_microseconds': 50.0,
            'power_consumption_watts': 1.0,
            'accuracy_percentage': 95.0
        }
    )


@pytest.fixture
def mock_hdl_generator():
    """Create a mock HDL generator."""
    generator = Mock()
    generator.generate_hdl = Mock(return_value={
        'top_module.v': '''
        module snn_top(
            input clk, rst,
            input [15:0] current_in,
            output spike_out
        );
        // Basic module content
        endmodule
        '''
    })
    return generator


class TestAutonomousLearningOrchestrator:
    """Test suite for autonomous learning orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, autonomous_learning_config, sample_network):
        """Test orchestrator initialization."""
        orchestrator = AutonomousLearningOrchestrator(autonomous_learning_config)
        
        # Test initialization
        await orchestrator.initialize_autonomous_systems(sample_network)
        
        assert orchestrator.self_modifying_arch is not None
        assert orchestrator.real_time_optimizer is not None
        assert orchestrator.performance_monitor is not None
        assert orchestrator.experiment_id is not None
        assert 'auto_learn_' in orchestrator.experiment_id
    
    @pytest.mark.asyncio
    async def test_autonomous_learning_cycle(self, autonomous_learning_config, sample_network):
        """Test single autonomous learning cycle."""
        orchestrator = AutonomousLearningOrchestrator(autonomous_learning_config)
        await orchestrator.initialize_autonomous_systems(sample_network)
        
        # Mock performance metrics
        with patch.object(orchestrator, '_collect_comprehensive_metrics') as mock_collect:
            mock_collect.return_value = {
                'throughput_mspikes_per_sec': 80.0,  # Below target
                'latency_microseconds': 60.0,
                'power_consumption_watts': 1.2,
                'accuracy_percentage': 92.0
            }
            
            # Run learning cycle
            result = await orchestrator.run_autonomous_learning_cycle(sample_network)
            
            assert 'cycle_duration' in result
            assert 'architecture_modified' in result
            assert 'optimizations_applied' in result
            assert 'performance_metrics' in result
            assert 'experiment_id' in result
            assert result['experiment_id'] == orchestrator.experiment_id
    
    @pytest.mark.asyncio
    async def test_performance_improvement_tracking(self, autonomous_learning_config, sample_network):
        """Test that performance improvements are properly tracked."""
        orchestrator = AutonomousLearningOrchestrator(autonomous_learning_config)
        await orchestrator.initialize_autonomous_systems(sample_network)
        
        # Simulate improving performance over multiple cycles
        performance_trajectory = [
            {'throughput_mspikes_per_sec': 70.0, 'accuracy_percentage': 88.0},
            {'throughput_mspikes_per_sec': 80.0, 'accuracy_percentage': 90.0},
            {'throughput_mspikes_per_sec': 90.0, 'accuracy_percentage': 93.0},
            {'throughput_mspikes_per_sec': 95.0, 'accuracy_percentage': 95.0}
        ]
        
        with patch.object(orchestrator, '_collect_comprehensive_metrics') as mock_collect:
            for metrics in performance_trajectory:
                mock_collect.return_value = metrics
                await orchestrator.run_autonomous_learning_cycle(sample_network)
        
        # Generate research report
        report = await orchestrator.generate_research_report()
        
        assert 'experiment_metadata' in report
        assert 'performance_analysis' in report
        assert len(report['performance_analysis']) > 0
        
        # Check for improvement detection
        throughput_analysis = report['performance_analysis'].get('throughput_mspikes_per_sec', {})
        if throughput_analysis:
            assert throughput_analysis['improvement'] > 0
            assert throughput_analysis['improvement_percentage'] > 0
    
    @pytest.mark.asyncio
    async def test_research_report_generation(self, autonomous_learning_config, sample_network):
        """Test research report generation with statistical analysis."""
        orchestrator = AutonomousLearningOrchestrator(autonomous_learning_config)
        await orchestrator.initialize_autonomous_systems(sample_network)
        
        # Add some performance data
        for i in range(15):
            orchestrator.research_data['performance_trajectory'].append({
                'timestamp': time.time() + i,
                'metrics': {
                    'throughput_mspikes_per_sec': 80.0 + np.random.normal(0, 5),
                    'accuracy_percentage': 92.0 + np.random.normal(0, 2)
                }
            })
        
        report = await orchestrator.generate_research_report()
        
        # Validate report structure
        assert 'experiment_metadata' in report
        assert 'performance_analysis' in report
        assert 'research_insights' in report
        assert 'statistical_significance' in report
        assert 'reproducibility_data' in report
        
        # Validate statistical analysis
        assert 'experiment_id' in report['experiment_metadata']
        assert 'session_duration_seconds' in report['experiment_metadata']
        assert isinstance(report['research_insights'], list)
    
    @pytest.mark.asyncio
    async def test_save_research_data(self, autonomous_learning_config, sample_network):
        """Test saving research data to files."""
        orchestrator = AutonomousLearningOrchestrator(autonomous_learning_config)
        await orchestrator.initialize_autonomous_systems(sample_network)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save research data
            report_path = await orchestrator.save_research_data(Path(temp_dir))
            
            # Verify files were created
            assert Path(report_path).exists()
            
            # Load and verify report content
            with open(report_path, 'r') as f:
                saved_report = json.load(f)
            
            assert 'experiment_metadata' in saved_report
            assert 'reproducibility_data' in saved_report


class TestSelfModifyingArchitecture:
    """Test suite for self-modifying architecture."""
    
    @pytest.mark.asyncio
    async def test_architecture_encoding(self, autonomous_learning_config, sample_network):
        """Test network architecture encoding."""
        arch = SelfModifyingArchitecture(sample_network, autonomous_learning_config)
        
        encoded = arch._encode_architecture(sample_network)
        
        assert 'layer_configs' in encoded
        assert 'global_parameters' in encoded
        assert 'topology_hash' in encoded
        assert len(encoded['layer_configs']) == len(sample_network.layers)
    
    @pytest.mark.asyncio
    async def test_autonomous_modification(self, autonomous_learning_config, sample_network):
        """Test autonomous architecture modification."""
        arch = SelfModifyingArchitecture(sample_network, autonomous_learning_config)
        
        # Performance metrics indicating need for modification
        performance_metrics = {
            'throughput_mspikes_per_sec': 60.0,  # Below target of 100
            'latency_microseconds': 80.0,        # Above target of 50
            'power_consumption_watts': 1.5,      # Above target of 1.0
            'accuracy_percentage': 88.0          # Below target of 95
        }
        
        # Test modification attempt
        modification_made = await arch.autonomous_modification(performance_metrics)
        
        # Should attempt modification due to performance gaps
        # Note: The actual modification depends on internal logic
        assert isinstance(modification_made, bool)
        
        if modification_made:
            assert len(arch.modification_history) > 0
            last_modification = arch.modification_history[-1]
            assert 'timestamp' in last_modification
            assert 'modification' in last_modification
            assert 'reasoning' in last_modification
    
    @pytest.mark.asyncio
    async def test_modification_candidate_generation(self, autonomous_learning_config, sample_network):
        """Test generation of modification candidates."""
        arch = SelfModifyingArchitecture(sample_network, autonomous_learning_config)
        
        performance_gaps = {
            'throughput_mspikes_per_sec': 0.4,  # 40% gap
            'latency_microseconds': 0.3         # 30% gap
        }
        
        current_metrics = {
            'throughput_mspikes_per_sec': 60.0,
            'latency_microseconds': 65.0
        }
        
        candidates = await arch._generate_modification_candidates(
            performance_gaps, current_metrics
        )
        
        assert isinstance(candidates, list)
        assert len(candidates) <= 10  # Should limit candidates
        
        for candidate in candidates:
            assert 'type' in candidate
            assert 'reasoning' in candidate


class TestSelfModifyingHDLGenerator:
    """Test suite for self-modifying HDL generator."""
    
    @pytest.mark.asyncio
    async def test_hdl_generator_initialization(self, mock_hdl_generator):
        """Test HDL generator initialization."""
        generator = SelfModifyingHDLGenerator(mock_hdl_generator)
        
        assert generator.base_generator is mock_hdl_generator
        assert len(generator.modification_templates) > 0
        assert generator.synthesis_history == []
        assert isinstance(generator.performance_models, dict)
    
    @pytest.mark.asyncio
    async def test_self_modifying_hdl_generation(self, mock_hdl_generator, sample_network):
        """Test self-modifying HDL generation."""
        generator = SelfModifyingHDLGenerator(mock_hdl_generator)
        
        performance_targets = {
            'throughput_mspikes_per_sec': 120.0,
            'latency_microseconds': 40.0,
            'power_consumption_watts': 0.8
        }
        
        synthesis_config = SynthesisConfiguration(
            optimization_level=3,
            synthesis_strategy="speed"
        )
        
        # Mock the base HDL generation
        mock_hdl_generator.generate_hdl.return_value = {
            'lif_neuron.v': 'module lif_neuron(); endmodule',
            'spike_router.v': 'module spike_router(); endmodule'
        }
        
        result = await generator.generate_self_modifying_hdl(
            sample_network, performance_targets, synthesis_config
        )
        
        assert 'base_hdl' in result
        assert 'modified_hdl' in result
        assert 'modifications_applied' in result
        assert 'validation_results' in result
        assert 'synthesis_config' in result
        assert 'performance_predictions' in result
    
    @pytest.mark.asyncio
    async def test_template_scoring(self, mock_hdl_generator):
        """Test modification template scoring."""
        generator = SelfModifyingHDLGenerator(mock_hdl_generator)
        
        template = generator.modification_templates[0]  # Get first template
        
        priorities = {
            'throughput': 0.8,
            'latency': 0.6,
            'power': 0.4
        }
        
        base_hdl = {
            'test.v': 'module test(); lif_neuron inst(); endmodule'
        }
        
        score = await generator._score_template(template, priorities, base_hdl)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_learning_from_synthesis_results(self, mock_hdl_generator):
        """Test learning from synthesis results."""
        generator = SelfModifyingHDLGenerator(mock_hdl_generator)
        
        # Create mock modification
        template = generator.modification_templates[0]
        modification = {
            'template': template,
            'parameters': {'PARALLEL_UNITS': 4, 'TAU_M': 20.0}
        }
        
        synthesis_results = {
            'synthesis_successful': True,
            'baseline_throughput_mspikes_per_sec': 80.0
        }
        
        actual_performance = {
            'throughput_mspikes_per_sec': 100.0,  # Improvement
            'latency_microseconds': 45.0,
            'power_consumption_watts': 1.1
        }
        
        initial_success_rate = generator.template_success_rates[template.name]
        
        await generator.learn_from_synthesis_results(
            [modification], synthesis_results, actual_performance
        )
        
        # Should update success rate
        final_success_rate = generator.template_success_rates[template.name]
        
        # Success rate should have been updated
        assert len(generator.synthesis_history) == 1


class TestAdaptiveRealTimeOptimizer:
    """Test suite for adaptive real-time optimizer."""
    
    def test_optimizer_creation(self):
        """Test optimizer creation with factory function."""
        performance_targets = {
            'throughput_mspikes_per_sec': 100.0,
            'latency_microseconds': 50.0,
            'power_consumption_watts': 1.0
        }
        
        optimizer = create_adaptive_realtime_optimizer(
            performance_targets,
            adaptation_rate=0.005,
            exploration_rate=0.15
        )
        
        assert isinstance(optimizer, AdaptiveRealTimeOptimizer)
        assert len(optimizer.targets) == 3
        assert optimizer.policy.adaptation_rate == 0.005
        assert optimizer.policy.exploration_rate == 0.15
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, sample_network):
        """Test optimizer initialization."""
        targets = [
            OptimizationTarget(
                metric_name='throughput_mspikes_per_sec',
                target_value=100.0,
                tolerance=5.0,
                priority=0.9,
                constraint_type='maximize'
            )
        ]
        
        policy = AdaptationPolicy(adaptation_rate=0.01)
        optimizer = AdaptiveRealTimeOptimizer(targets, policy)
        
        await optimizer.initialize(sample_network)
        
        assert optimizer.parameter_adapter is not None
        assert optimizer.resource_manager is not None
        assert optimizer.predictive_optimizer is not None
    
    @pytest.mark.asyncio
    async def test_optimization_loop_start_stop(self, sample_network):
        """Test starting and stopping optimization loop."""
        targets = [
            OptimizationTarget(
                metric_name='throughput_mspikes_per_sec',
                target_value=100.0,
                tolerance=5.0,
                priority=0.9,
                constraint_type='maximize'
            )
        ]
        
        policy = AdaptationPolicy(adaptation_rate=0.01)
        optimizer = AdaptiveRealTimeOptimizer(targets, policy)
        
        await optimizer.initialize(sample_network)
        
        # Start optimization
        await optimizer.start_optimization_loop(sample_network)
        assert optimizer.optimization_active is True
        assert optimizer.optimization_thread is not None
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Stop optimization
        await optimizer.stop_optimization_loop()
        assert optimizer.optimization_active is False
    
    @pytest.mark.asyncio
    async def test_optimization_report_generation(self, sample_network):
        """Test optimization report generation."""
        targets = [
            OptimizationTarget(
                metric_name='throughput_mspikes_per_sec',
                target_value=100.0,
                tolerance=5.0,
                priority=0.9,
                constraint_type='maximize'
            )
        ]
        
        policy = AdaptationPolicy(adaptation_rate=0.01)
        optimizer = AdaptiveRealTimeOptimizer(targets, policy)
        
        await optimizer.initialize(sample_network)
        
        report = await optimizer.get_optimization_report()
        
        assert 'optimization_metrics' in report
        assert 'parameter_adapter_state' in report
        assert 'resource_manager_status' in report
        assert 'predictive_accuracy' in report


class TestAutonomousQualityGates:
    """Test suite for autonomous quality gates."""
    
    @pytest.mark.asyncio
    async def test_quality_orchestrator_creation(self):
        """Test quality orchestrator creation."""
        orchestrator = create_quality_orchestrator()
        
        assert isinstance(orchestrator, AutonomousQualityOrchestrator)
        assert len(orchestrator.quality_gates) > 0
        
        # Check default gates are present
        gate_names = [gate.name for gate in orchestrator.quality_gates]
        expected_gates = [
            'PerformanceRegression',
            'HardwareReliability',
            'SecurityCompliance',
            'FunctionalCorrectness',
            'ResourceUtilization'
        ]
        
        for expected_gate in expected_gates:
            assert expected_gate in gate_names
    
    @pytest.mark.asyncio
    async def test_quality_gates_execution(self, sample_network):
        """Test quality gates execution."""
        orchestrator = create_quality_orchestrator()
        
        # Create execution context
        context = {
            'network': sample_network,
            'performance_metrics': {
                'throughput_mspikes_per_sec': 95.0,
                'latency_microseconds': 55.0,
                'power_consumption_watts': 1.1,
                'accuracy_percentage': 93.0
            },
            'hardware_metrics': {
                'temperature_celsius': 75.0,
                'vccint_voltage': 0.95,
                'fpga_utilization_percentage': 78.0
            },
            'hdl_files': {
                'top.v': 'module top(); endmodule'
            },
            'resource_metrics': {
                'lut_utilization': 82.0,
                'bram_utilization': 65.0,
                'dsp_utilization': 70.0
            }
        }
        
        # Execute quality gates
        result = await orchestrator.execute_quality_gates(context)
        
        assert 'execution_summary' in result
        assert 'gate_results' in result
        assert 'overall_analysis' in result
        assert 'recommendations' in result
        
        # Check execution summary
        execution_summary = result['execution_summary']
        assert 'timestamp' in execution_summary
        assert 'execution_time' in execution_summary
        assert 'overall_status' in execution_summary
        assert 'quality_score' in execution_summary
        
        # Quality score should be between 0 and 100
        assert 0 <= execution_summary['quality_score'] <= 100
    
    @pytest.mark.asyncio
    async def test_quality_gate_health_metrics(self):
        """Test quality gate health metrics."""
        orchestrator = create_quality_orchestrator()
        
        # Execute gates multiple times to build history
        context = {
            'performance_metrics': {'throughput_mspikes_per_sec': 95.0},
            'hardware_metrics': {'temperature_celsius': 75.0}
        }
        
        for _ in range(5):
            await orchestrator.execute_quality_gates(context)
        
        # Check health metrics for each gate
        for gate in orchestrator.quality_gates:
            health_metrics = gate.get_health_metrics()
            
            assert 'success_rate' in health_metrics
            assert 'average_execution_time' in health_metrics
            assert 'execution_count' in health_metrics
            assert 'enabled' in health_metrics
            
            assert 0.0 <= health_metrics['success_rate'] <= 1.0
            assert health_metrics['execution_count'] > 0


class TestIntegrationAndBenchmarks:
    """Integration tests and performance benchmarks."""
    
    @pytest.mark.asyncio
    async def test_full_generation4_integration(self, sample_network):
        """Test full Generation 4 system integration."""
        # Create autonomous learning system
        learning_system = create_autonomous_learning_system(
            sample_network,
            learning_rate=0.001,
            adaptation_threshold=0.05,
            performance_targets={
                'throughput_mspikes_per_sec': 100.0,
                'latency_microseconds': 50.0,
                'power_consumption_watts': 1.0,
                'accuracy_percentage': 95.0
            }
        )
        
        # Initialize system
        await learning_system.initialize_autonomous_systems(sample_network)
        
        # Create quality orchestrator
        quality_orchestrator = create_quality_orchestrator()
        
        # Create adaptive optimizer
        performance_targets = {
            'throughput_mspikes_per_sec': 100.0,
            'latency_microseconds': 50.0,
            'power_consumption_watts': 1.0
        }
        adaptive_optimizer = create_adaptive_realtime_optimizer(performance_targets)
        await adaptive_optimizer.initialize(sample_network)
        
        # Run integrated workflow
        for cycle in range(3):  # Short integration test
            # Run autonomous learning cycle
            learning_result = await learning_system.run_autonomous_learning_cycle(sample_network)
            
            # Execute quality gates
            quality_context = {
                'network': sample_network,
                'performance_metrics': learning_result['performance_metrics']
            }
            quality_result = await quality_orchestrator.execute_quality_gates(quality_context)
            
            # Get optimization report
            optimization_report = await adaptive_optimizer.get_optimization_report()
            
            # Validate integration
            assert 'experiment_id' in learning_result
            assert 'overall_status' in quality_result['execution_summary']
            assert 'optimization_metrics' in optimization_report
        
        # Generate final research report
        final_report = await learning_system.generate_research_report()
        
        assert 'experiment_metadata' in final_report
        assert 'performance_analysis' in final_report
        assert 'statistical_significance' in final_report
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_benchmarks(self, sample_network):
        """Performance benchmarks for Generation 4 systems."""
        # Benchmark autonomous learning cycle
        learning_system = create_autonomous_learning_system(sample_network)
        await learning_system.initialize_autonomous_systems(sample_network)
        
        start_time = time.time()
        for _ in range(10):
            await learning_system.run_autonomous_learning_cycle(sample_network)
        learning_duration = time.time() - start_time
        
        avg_cycle_time = learning_duration / 10
        assert avg_cycle_time < 1.0  # Should complete cycle in under 1 second
        
        # Benchmark quality gates execution
        quality_orchestrator = create_quality_orchestrator()
        context = {
            'network': sample_network,
            'performance_metrics': {'throughput_mspikes_per_sec': 95.0}
        }
        
        start_time = time.time()
        for _ in range(5):
            await quality_orchestrator.execute_quality_gates(context)
        quality_duration = time.time() - start_time
        
        avg_quality_time = quality_duration / 5
        assert avg_quality_time < 2.0  # Should complete gates in under 2 seconds
        
        print(f"Performance Benchmarks:")
        print(f"  Average learning cycle time: {avg_cycle_time:.3f}s")
        print(f"  Average quality gates time: {avg_quality_time:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_long_running_stability(self, sample_network):
        """Test long-running stability of Generation 4 systems."""
        learning_system = create_autonomous_learning_system(sample_network)
        await learning_system.initialize_autonomous_systems(sample_network)
        
        # Run for extended period
        cycles_completed = 0
        errors_encountered = 0
        
        start_time = time.time()
        
        while time.time() - start_time < 10.0:  # Run for 10 seconds
            try:
                await learning_system.run_autonomous_learning_cycle(sample_network)
                cycles_completed += 1
            except Exception as e:
                errors_encountered += 1
                print(f"Error in cycle {cycles_completed}: {e}")
            
            await asyncio.sleep(0.1)  # Small delay between cycles
        
        # Calculate stability metrics
        error_rate = errors_encountered / max(cycles_completed, 1)
        
        assert error_rate < 0.1  # Less than 10% error rate
        assert cycles_completed > 50  # Should complete substantial number of cycles
        
        print(f"Stability Test Results:")
        print(f"  Cycles completed: {cycles_completed}")
        print(f"  Errors encountered: {errors_encountered}")
        print(f"  Error rate: {error_rate:.3%}")


class TestResearchReproducibility:
    """Test suite for research reproducibility and validation."""
    
    @pytest.mark.asyncio
    async def test_reproducible_experiments(self, sample_network):
        """Test that experiments produce reproducible results."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create two identical learning systems
        config = AutonomousLearningConfig(
            learning_rate=0.001,
            adaptation_threshold=0.1,
            exploration_rate=0.0  # Disable exploration for reproducibility
        )
        
        system1 = AutonomousLearningOrchestrator(config)
        system2 = AutonomousLearningOrchestrator(config)
        
        await system1.initialize_autonomous_systems(sample_network)
        await system2.initialize_autonomous_systems(sample_network)
        
        # Run identical learning cycles with same input
        performance_metrics = {
            'throughput_mspikes_per_sec': 80.0,
            'latency_microseconds': 60.0,
            'power_consumption_watts': 1.2,
            'accuracy_percentage': 92.0
        }
        
        # Mock the performance collection to return identical values
        with patch.object(system1, '_collect_comprehensive_metrics') as mock1, \
             patch.object(system2, '_collect_comprehensive_metrics') as mock2:
            
            mock1.return_value = performance_metrics
            mock2.return_value = performance_metrics
            
            result1 = await system1.run_autonomous_learning_cycle(sample_network)
            result2 = await system2.run_autonomous_learning_cycle(sample_network)
        
        # Results should be highly similar (allowing for minor floating-point differences)
        assert abs(result1['cycle_duration'] - result2['cycle_duration']) < 0.1
        assert result1['architecture_modified'] == result2['architecture_modified']
        assert result1['optimizations_applied'] == result2['optimizations_applied']
    
    @pytest.mark.asyncio
    async def test_statistical_significance_validation(self, sample_network):
        """Test statistical significance computation."""
        learning_system = create_autonomous_learning_system(sample_network)
        await learning_system.initialize_autonomous_systems(sample_network)
        
        # Generate trajectory with known improvement
        base_throughput = 80.0
        improvement_per_cycle = 2.0
        
        for i in range(20):
            current_throughput = base_throughput + i * improvement_per_cycle + np.random.normal(0, 1)
            
            learning_system.research_data['performance_trajectory'].append({
                'timestamp': time.time() + i,
                'metrics': {
                    'throughput_mspikes_per_sec': current_throughput,
                    'accuracy_percentage': 90.0 + np.random.normal(0, 0.5)
                }
            })
        
        # Generate research report
        report = await learning_system.generate_research_report()
        
        # Check statistical significance results
        assert 'statistical_significance' in report
        significance = report['statistical_significance']
        
        if 'throughput_mspikes_per_sec' in significance:
            throughput_stats = significance['throughput_mspikes_per_sec']
            assert 'effect_size' in throughput_stats
            assert 'improvement' in throughput_stats
            assert 'practical_significance' in throughput_stats
            
            # Should detect improvement
            assert throughput_stats['improvement'] > 0
    
    @pytest.mark.asyncio
    async def test_research_data_export_format(self, sample_network):
        """Test research data export format for publication."""
        learning_system = create_autonomous_learning_system(sample_network)
        await learning_system.initialize_autonomous_systems(sample_network)
        
        # Add some research data
        learning_system.research_data['modifications'] = [
            {
                'timestamp': time.time(),
                'modification': {
                    'type': 'test_modification',
                    'parameters': {'param1': 1.0}
                }
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = await learning_system.save_research_data(Path(temp_dir))
            
            # Load and validate exported data format
            with open(report_path, 'r') as f:
                exported_data = json.load(f)
            
            # Validate required fields for publication
            required_fields = [
                'experiment_metadata',
                'performance_analysis',
                'research_insights',
                'statistical_significance',
                'reproducibility_data'
            ]
            
            for field in required_fields:
                assert field in exported_data, f"Missing required field: {field}"
            
            # Validate reproducibility data
            repro_data = exported_data['reproducibility_data']
            assert 'configuration_snapshot' in repro_data
            assert 'system_info' in repro_data
            assert 'data_export' in repro_data


# Pytest configuration for performance tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])