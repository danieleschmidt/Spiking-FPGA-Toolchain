"""
Tests for adaptive spike encoding research module.
"""

import numpy as np
import pytest
from spiking_fpga.research.adaptive_encoding import (
    AdaptiveSpikeCoder,
    MultiModalEncoder,
    EncodingMode,
    SpikeTrain,
)


class TestAdaptiveSpikeCoder:
    """Test suite for adaptive spike coding algorithms."""

    def test_adaptive_encoding_basic(self):
        """Test basic adaptive encoding functionality."""
        coder = AdaptiveSpikeCoder()
        
        # Test with different input patterns
        test_inputs = [
            np.array([0.1, 0.5, 0.9]),  # Low-high variation
            np.array([0.5, 0.52, 0.48, 0.51]),  # High correlation
            np.sin(np.linspace(0, 4*np.pi, 20)),  # Oscillatory
        ]
        
        for input_data in test_inputs:
            spike_train = coder.encode(input_data)
            
            assert isinstance(spike_train, SpikeTrain)
            assert len(spike_train.spike_times) > 0
            assert len(spike_train.neuron_ids) > 0
            assert spike_train.encoding_mode in EncodingMode
            
    def test_encoding_mode_selection(self):
        """Test that different input patterns select appropriate encoding modes."""
        coder = AdaptiveSpikeCoder()
        
        # High correlation input should prefer temporal pattern
        correlated_input = np.array([0.1, 0.12, 0.11, 0.13, 0.12])
        stats = coder._calculate_input_statistics(correlated_input)
        mode = coder._select_optimal_encoding(stats)
        
        # The selection logic should handle various cases
        assert mode in EncodingMode
        
        # High variance input
        high_var_input = np.array([0.1, 0.9, 0.2, 0.8, 0.0, 1.0])
        stats = coder._calculate_input_statistics(high_var_input)
        mode = coder._select_optimal_encoding(stats)
        
        assert mode in EncodingMode
        
    def test_temporal_pattern_encoding(self):
        """Test temporal pattern encoding specifically."""
        coder = AdaptiveSpikeCoder()
        input_data = np.array([0.2, 0.5, 0.8])
        
        spike_train = coder._encode_temporal_pattern(input_data)
        
        assert isinstance(spike_train, SpikeTrain)
        assert len(spike_train.spike_times) > 0
        assert len(spike_train.neuron_ids) == len(spike_train.spike_times)
        
        # Check that different neurons have spikes at different time ranges
        unique_neurons = np.unique(spike_train.neuron_ids)
        assert len(unique_neurons) == len(input_data)
        
    def test_population_vector_encoding(self):
        """Test population vector encoding specifically."""
        coder = AdaptiveSpikeCoder()
        input_data = np.array([0.3, 0.7])
        
        spike_train = coder._encode_population_vector(input_data)
        
        assert isinstance(spike_train, SpikeTrain)
        assert len(spike_train.spike_times) > 0
        
        # Should have multiple neurons per input value (population)
        unique_neurons = np.unique(spike_train.neuron_ids)
        assert len(unique_neurons) > len(input_data)
        
    def test_burst_coding_encoding(self):
        """Test burst coding encoding specifically."""
        coder = AdaptiveSpikeCoder()
        input_data = np.array([0.1, 0.6, 0.9])
        
        spike_train = coder._encode_burst_coding(input_data)
        
        assert isinstance(spike_train, SpikeTrain)
        assert len(spike_train.spike_times) > 0
        
        # Should have multiple spikes per neuron (bursts)
        for neuron_id in np.unique(spike_train.neuron_ids):
            neuron_spikes = spike_train.spike_times[spike_train.neuron_ids == neuron_id]
            # Each neuron should have multiple spikes (burst pattern)
            assert len(neuron_spikes) >= 2
            
    def test_adaptation_learning(self):
        """Test that the encoder adapts over time."""
        coder = AdaptiveSpikeCoder()
        
        # Encode multiple similar patterns
        pattern = np.array([0.1, 0.5, 0.9])
        for _ in range(10):
            spike_train = coder.encode(pattern)
            assert len(coder.encoding_history) <= 10
        
        # History should be maintained
        assert len(coder.encoding_history) == 10
        
        # Test history truncation
        for _ in range(1000):
            coder.encode(pattern)
        
        assert len(coder.encoding_history) <= coder.adaptation_window

    def test_input_statistics_calculation(self):
        """Test input statistics calculation."""
        coder = AdaptiveSpikeCoder()
        
        # Test with known input patterns
        constant_input = np.array([0.5, 0.5, 0.5, 0.5])
        stats = coder._calculate_input_statistics(constant_input)
        
        assert stats['variance'] == pytest.approx(0.0, abs=1e-10)
        assert stats['temporal_correlation'] == 0.0  # NaN -> 0.0
        
        # Random input
        random_input = np.random.random(20)
        stats = coder._calculate_input_statistics(random_input)
        
        assert 'variance' in stats
        assert 'temporal_correlation' in stats
        assert 'spectral_concentration' in stats
        assert all(isinstance(v, (int, float)) for v in stats.values())


class TestMultiModalEncoder:
    """Test suite for multi-modal encoder wrapper."""
    
    def test_multi_modal_basic(self):
        """Test basic multi-modal encoder functionality."""
        encoder = MultiModalEncoder()
        
        input_data = np.array([0.2, 0.5, 0.8])
        spike_train = encoder.encode(input_data)
        
        assert isinstance(spike_train, SpikeTrain)
        assert len(spike_train.spike_times) > 0
        
    def test_adaptation_metrics(self):
        """Test adaptation metrics reporting."""
        encoder = MultiModalEncoder()
        
        # Initial metrics
        metrics = encoder.get_adaptation_metrics()
        assert 'total_encodings' in metrics
        assert metrics['total_encodings'] == 0
        
        # After encoding
        encoder.encode(np.array([0.1, 0.5]))
        metrics = encoder.get_adaptation_metrics()
        assert metrics['total_encodings'] == 1


@pytest.fixture
def sample_spike_train():
    """Fixture providing a sample spike train for testing."""
    return SpikeTrain(
        spike_times=np.array([1.0, 5.0, 12.0, 25.0]),
        neuron_ids=np.array([0, 0, 1, 1]),
        encoding_mode=EncodingMode.TEMPORAL_PATTERN,
        information_content=10.5
    )


def test_spike_train_structure(sample_spike_train):
    """Test spike train data structure."""
    assert len(sample_spike_train.spike_times) == 4
    assert len(sample_spike_train.neuron_ids) == 4
    assert sample_spike_train.encoding_mode == EncodingMode.TEMPORAL_PATTERN
    assert sample_spike_train.information_content == 10.5


class TestEncodingModeEnum:
    """Test encoding mode enumeration."""
    
    def test_encoding_modes_exist(self):
        """Test that all expected encoding modes are defined."""
        expected_modes = [
            'TEMPORAL_PATTERN',
            'POPULATION_VECTOR', 
            'BURST_CODING'
        ]
        
        for mode_name in expected_modes:
            assert hasattr(EncodingMode, mode_name)
            
    def test_encoding_mode_values(self):
        """Test encoding mode string values."""
        assert EncodingMode.TEMPORAL_PATTERN.value == "temporal_pattern"
        assert EncodingMode.POPULATION_VECTOR.value == "population_vector"
        assert EncodingMode.BURST_CODING.value == "burst_coding"


class TestEncodingQualityAndPerformance:
    """Test encoding quality metrics and performance."""
    
    def test_encoding_information_preservation(self):
        """Test that encoding preserves input information structure."""
        coder = AdaptiveSpikeCoder()
        
        # Test with structured input
        structured_input = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        spike_train = coder.encode(structured_input)
        
        # Should have spikes from all input elements
        unique_neurons = len(np.unique(spike_train.neuron_ids))
        
        # Depending on encoding mode, should relate to input size
        if spike_train.encoding_mode == EncodingMode.TEMPORAL_PATTERN:
            assert unique_neurons == len(structured_input)
        elif spike_train.encoding_mode == EncodingMode.POPULATION_VECTOR:
            assert unique_neurons > len(structured_input)  # Population coding
        else:  # BURST_CODING
            assert unique_neurons == len(structured_input)
            
    def test_encoding_scalability(self):
        """Test encoding with different input sizes."""
        coder = AdaptiveSpikeCoder()
        
        input_sizes = [1, 5, 10, 50, 100]
        
        for size in input_sizes:
            input_data = np.random.random(size)
            spike_train = coder.encode(input_data)
            
            # Should produce reasonable number of spikes
            assert len(spike_train.spike_times) > 0
            assert len(spike_train.spike_times) < size * 100  # Reasonable upper bound
            
            # Spike times should be reasonable
            assert np.all(spike_train.spike_times >= 0)
            assert np.max(spike_train.spike_times) < 10000  # Within reasonable time window
            
    @pytest.mark.performance
    def test_encoding_performance(self):
        """Test encoding performance with large inputs."""
        import time
        
        coder = AdaptiveSpikeCoder()
        large_input = np.random.random(1000)
        
        start_time = time.time()
        spike_train = coder.encode(large_input)
        encoding_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert encoding_time < 1.0  # Less than 1 second
        assert len(spike_train.spike_times) > 0
        
    def test_encoding_determinism(self):
        """Test encoding determinism with same inputs."""
        coder1 = AdaptiveSpikeCoder()
        coder2 = AdaptiveSpikeCoder()
        
        # Note: Due to randomness in encoding, this tests consistency of mode selection
        input_data = np.array([0.1, 0.5, 0.9])
        
        stats1 = coder1._calculate_input_statistics(input_data)
        stats2 = coder2._calculate_input_statistics(input_data)
        
        # Statistics should be identical for same input
        assert stats1['variance'] == pytest.approx(stats2['variance'])
        assert stats1['temporal_correlation'] == pytest.approx(stats2['temporal_correlation'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])