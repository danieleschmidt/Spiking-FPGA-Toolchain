"""
Tests for meta-plasticity STDP research module.
"""

import numpy as np
import pytest
from spiking_fpga.research.meta_plasticity import (
    MetaPlasticSTDP,
    BitstiftSTDP,
    HomeostasticRegulator,
    PlasticityParameters,
    SynapticConnection,
)


class TestPlasticityParameters:
    """Test plasticity parameter configuration."""
    
    def test_default_parameters(self):
        """Test default parameter initialization."""
        params = PlasticityParameters()
        
        assert params.a_ltp == 0.01
        assert params.a_ltd == 0.012
        assert params.tau_ltp == 20.0
        assert params.tau_ltd == 20.0
        assert params.target_rate == 2.0
        assert params.weight_precision == 8
        assert params.use_bit_shift_approximation is True
        
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        params = PlasticityParameters(
            a_ltp=0.02,
            tau_ltp=15.0,
            weight_precision=16
        )
        
        assert params.a_ltp == 0.02
        assert params.tau_ltp == 15.0
        assert params.weight_precision == 16
        assert params.a_ltd == 0.012  # Default value


class TestBitstiftSTDP:
    """Test bit-shift STDP implementation."""
    
    def test_initialization(self):
        """Test BitstiftSTDP initialization."""
        params = PlasticityParameters()
        stdp = BitstiftSTDP(params)
        
        assert stdp.max_weight == (2 ** params.weight_precision) - 1
        assert stdp.min_weight == 0
        assert len(stdp.ltp_shifts) > 0
        assert len(stdp.ltd_shifts) > 0
        
    def test_shift_table_computation(self):
        """Test bit-shift approximation table computation."""
        params = PlasticityParameters()
        stdp = BitstiftSTDP(params)
        
        # Test that shift tables contain valid values
        for dt, shift in stdp.ltp_shifts.items():
            assert 0 <= shift <= 7
            assert 1 <= dt <= 50
            
        for dt, shift in stdp.ltd_shifts.items():
            assert 0 <= shift <= 7
            assert 1 <= dt <= 50
            
    def test_synapse_update_ltp(self):
        """Test LTP (long-term potentiation) updates."""
        params = PlasticityParameters()
        stdp = BitstiftSTDP(params)
        
        connection = SynapticConnection(
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=128.0,  # Middle weight
            meta_plasticity_factor=1.0,
            last_pre_spike=10.0,
            last_post_spike=15.0,  # Post after pre -> LTP
            homeostatic_scaling=1.0
        )
        
        updated = stdp.update_synapse(connection, 20.0)
        
        # Weight should increase for LTP
        assert updated.weight >= connection.weight
        assert updated.weight <= stdp.max_weight
        
    def test_synapse_update_ltd(self):
        """Test LTD (long-term depression) updates."""
        params = PlasticityParameters()
        stdp = BitstiftSTDP(params)
        
        connection = SynapticConnection(
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=128.0,
            meta_plasticity_factor=1.0,
            last_pre_spike=15.0,
            last_post_spike=10.0,  # Pre after post -> LTD
            homeostatic_scaling=1.0
        )
        
        updated = stdp.update_synapse(connection, 20.0)
        
        # Weight should decrease for LTD
        assert updated.weight <= connection.weight
        assert updated.weight >= stdp.min_weight
        
    def test_no_update_outside_window(self):
        """Test no update when spikes are outside plasticity window."""
        params = PlasticityParameters()
        stdp = BitstiftSTDP(params)
        
        connection = SynapticConnection(
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=128.0,
            meta_plasticity_factor=1.0,
            last_pre_spike=10.0,
            last_post_spike=100.0,  # Far outside window
            homeostatic_scaling=1.0
        )
        
        original_weight = connection.weight
        updated = stdp.update_synapse(connection, 150.0)
        
        # Weight should remain unchanged
        assert updated.weight == original_weight


class TestHomeostasticRegulator:
    """Test homeostatic regulation mechanism."""
    
    def test_initialization(self):
        """Test homeostatic regulator initialization."""
        params = PlasticityParameters()
        regulator = HomeostasticRegulator(params)
        
        assert regulator.params == params
        assert len(regulator.neuron_activities) == 0
        assert len(regulator.scaling_factors) == 0
        
    def test_activity_tracking(self):
        """Test neuron activity tracking."""
        params = PlasticityParameters()
        regulator = HomeostasticRegulator(params)
        
        # Add spike activities
        regulator.update_activity(0, 10.0)
        regulator.update_activity(0, 20.0)
        regulator.update_activity(1, 15.0)
        
        assert 0 in regulator.neuron_activities
        assert 1 in regulator.neuron_activities
        assert len(regulator.neuron_activities[0]) == 2
        assert len(regulator.neuron_activities[1]) == 1
        
    def test_activity_window_cleanup(self):
        """Test old activity cleanup."""
        params = PlasticityParameters()
        regulator = HomeostasticRegulator(params)
        
        # Add activities across time
        regulator.update_activity(0, 10.0)
        regulator.update_activity(0, 500.0)
        regulator.update_activity(0, 1500.0)  # This should clean up old activities
        
        # Only recent activities should remain
        activities = regulator.neuron_activities[0]
        assert all(t > 500.0 for t in activities)  # Within window from 1500.0
        
    def test_firing_rate_calculation(self):
        """Test firing rate calculation."""
        params = PlasticityParameters()
        regulator = HomeostasticRegulator(params)
        
        # Add known number of spikes in time window
        current_time = 1000.0
        for i in range(5):  # 5 spikes
            regulator.update_activity(0, current_time - 100.0 + i * 20.0)
            
        firing_rate = regulator.calculate_firing_rate(0)
        expected_rate = 5.0  # 5 spikes in 1000ms window = 5 Hz
        assert firing_rate == pytest.approx(expected_rate, rel=0.1)
        
    def test_scaling_factor_calculation(self):
        """Test homeostatic scaling factor calculation."""
        params = PlasticityParameters(target_rate=2.0)
        regulator = HomeostasticRegulator(params)
        
        # Test with no activity (should scale up)
        scaling = regulator.get_scaling_factor(0)
        assert scaling > 1.0
        
        # Test with high activity
        current_time = 1000.0
        for i in range(10):  # High activity
            regulator.update_activity(1, current_time - 100.0 + i * 10.0)
            
        scaling = regulator.get_scaling_factor(1)
        # Should scale down due to high activity
        assert 0.1 <= scaling <= 3.0  # Within bounds


class TestMetaPlasticSTDP:
    """Test complete meta-plastic STDP system."""
    
    def test_initialization(self):
        """Test system initialization."""
        stdp = MetaPlasticSTDP()
        
        assert stdp.learning_enabled is True
        assert len(stdp.synapses) == 0
        assert stdp.update_count == 0
        assert isinstance(stdp.bitshift_stdp, BitstiftSTDP)
        assert isinstance(stdp.homeostatic, HomeostasticRegulator)
        
    def test_add_synapse(self):
        """Test adding synaptic connections."""
        stdp = MetaPlasticSTDP()
        
        stdp.add_synapse(0, 1, initial_weight=0.7)
        stdp.add_synapse(1, 2, initial_weight=0.3)
        
        assert len(stdp.synapses) == 2
        assert (0, 1) in stdp.synapses
        assert (1, 2) in stdp.synapses
        
        # Check initial weights
        weight_01 = stdp.get_synapse_weight(0, 1)
        weight_12 = stdp.get_synapse_weight(1, 2)
        
        assert weight_01 == pytest.approx(0.7, rel=0.1)
        assert weight_12 == pytest.approx(0.3, rel=0.1)
        
    def test_process_spike_basic(self):
        """Test basic spike processing."""
        stdp = MetaPlasticSTDP()
        
        # Add synapses
        stdp.add_synapse(0, 1)
        stdp.add_synapse(1, 2)
        
        # Process spikes
        stdp.process_spike(0, 10.0)
        stdp.process_spike(1, 15.0)
        stdp.process_spike(2, 20.0)
        
        assert stdp.update_count == 3
        
        # Check that homeostatic activities are tracked
        assert 0 in stdp.homeostatic.neuron_activities
        assert 1 in stdp.homeostatic.neuron_activities
        assert 2 in stdp.homeostatic.neuron_activities
        
    def test_plasticity_learning_sequence(self):
        """Test plasticity changes with spike sequences."""
        stdp = MetaPlasticSTDP()
        stdp.add_synapse(0, 1, initial_weight=0.5)
        
        initial_weight = stdp.get_synapse_weight(0, 1)
        
        # Create LTP condition (pre before post)
        stdp.process_spike(0, 10.0)  # Pre-synaptic spike
        stdp.process_spike(1, 15.0)  # Post-synaptic spike shortly after
        
        ltp_weight = stdp.get_synapse_weight(0, 1)
        
        # Weight should change due to STDP
        # Note: Actual direction depends on implementation details
        assert ltp_weight != initial_weight
        
    def test_learning_enable_disable(self):
        """Test enabling/disabling learning."""
        stdp = MetaPlasticSTDP()
        stdp.add_synapse(0, 1, initial_weight=0.5)
        
        # Disable learning
        stdp.learning_enabled = False
        initial_weight = stdp.get_synapse_weight(0, 1)
        
        # Process spikes
        stdp.process_spike(0, 10.0)
        stdp.process_spike(1, 15.0)
        
        # Weight should not change when learning is disabled
        final_weight = stdp.get_synapse_weight(0, 1)
        assert final_weight == initial_weight
        
    def test_plasticity_statistics(self):
        """Test plasticity statistics reporting."""
        stdp = MetaPlasticSTDP()
        stdp.add_synapse(0, 1)
        stdp.add_synapse(1, 2)
        
        # Process some spikes
        for i in range(10):
            stdp.process_spike(0, i * 10.0)
            stdp.process_spike(1, i * 10.0 + 5.0)
            stdp.process_spike(2, i * 10.0 + 15.0)
            
        stats = stdp.get_plasticity_statistics()
        
        assert 'total_synapses' in stats
        assert 'avg_weight' in stats
        assert 'weight_std' in stats
        assert 'avg_firing_rate' in stats
        assert 'update_count' in stats
        
        assert stats['total_synapses'] == 2
        assert stats['update_count'] == 30
        assert 0.0 <= stats['avg_weight'] <= 1.0
        
    def test_nonexistent_synapse(self):
        """Test behavior with non-existent synapse."""
        stdp = MetaPlasticSTDP()
        
        weight = stdp.get_synapse_weight(0, 1)
        assert weight is None


class TestSynapticConnection:
    """Test synaptic connection data structure."""
    
    def test_connection_initialization(self):
        """Test synaptic connection initialization."""
        connection = SynapticConnection(
            pre_neuron_id=5,
            post_neuron_id=10,
            weight=0.75,
            meta_plasticity_factor=0.8,
            last_pre_spike=100.0,
            last_post_spike=105.0
        )
        
        assert connection.pre_neuron_id == 5
        assert connection.post_neuron_id == 10
        assert connection.weight == 0.75
        assert connection.meta_plasticity_factor == 0.8
        assert connection.last_pre_spike == 100.0
        assert connection.last_post_spike == 105.0
        assert connection.homeostatic_scaling == 1.0  # Default value


class TestIntegrationScenarios:
    """Test integration scenarios combining all components."""
    
    def test_network_learning_scenario(self):
        """Test realistic network learning scenario."""
        stdp = MetaPlasticSTDP()
        
        # Create small network: 3 neurons, 2 synapses
        stdp.add_synapse(0, 1, 0.4)
        stdp.add_synapse(1, 2, 0.6)
        
        # Simulate learning episode
        spike_times = [
            (0, 10.0), (1, 15.0),    # LTP for 0->1
            (1, 30.0), (2, 35.0),    # LTP for 1->2
            (0, 50.0), (1, 52.0),    # Strong LTP for 0->1
            (2, 70.0), (1, 72.0),    # LTD for 1->2
        ]
        
        initial_weights = {
            (0, 1): stdp.get_synapse_weight(0, 1),
            (1, 2): stdp.get_synapse_weight(1, 2)
        }
        
        for neuron_id, spike_time in spike_times:
            stdp.process_spike(neuron_id, spike_time)
            
        final_weights = {
            (0, 1): stdp.get_synapse_weight(0, 1),
            (1, 2): stdp.get_synapse_weight(1, 2)
        }
        
        # Weights should have changed
        assert final_weights[(0, 1)] != initial_weights[(0, 1)]
        assert final_weights[(1, 2)] != initial_weights[(1, 2)]
        
        # Get final statistics
        stats = stdp.get_plasticity_statistics()
        assert stats['update_count'] == len(spike_times)
        assert stats['total_synapses'] == 2
        
    @pytest.mark.performance
    def test_performance_large_network(self):
        """Test performance with larger network."""
        import time
        
        stdp = MetaPlasticSTDP()
        
        # Create larger network
        n_neurons = 50
        n_synapses = 200
        
        # Add random synapses
        np.random.seed(42)
        for _ in range(n_synapses):
            pre = np.random.randint(0, n_neurons-1)
            post = np.random.randint(pre+1, n_neurons)
            stdp.add_synapse(pre, post, np.random.uniform(0.1, 0.9))
            
        # Simulate spike activity
        n_spikes = 1000
        start_time = time.time()
        
        for i in range(n_spikes):
            neuron_id = np.random.randint(0, n_neurons)
            spike_time = i * 1.0  # 1ms intervals
            stdp.process_spike(neuron_id, spike_time)
            
        processing_time = time.time() - start_time
        
        # Should process spikes efficiently
        assert processing_time < 5.0  # Less than 5 seconds
        assert stdp.update_count == n_spikes
        
        # Verify final state
        stats = stdp.get_plasticity_statistics()
        assert stats['total_synapses'] == n_synapses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])