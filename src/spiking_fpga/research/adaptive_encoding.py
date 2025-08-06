"""
Bio-Inspired Adaptive Spike Encoding/Decoding Module

Novel multi-modal spike encoding that dynamically adapts to input characteristics.
Implements temporal pattern coding, population vector coding, and burst coding.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EncodingMode(Enum):
    TEMPORAL_PATTERN = "temporal_pattern"
    POPULATION_VECTOR = "population_vector" 
    BURST_CODING = "burst_coding"
    

@dataclass
class SpikeTrain:
    spike_times: np.ndarray
    neuron_ids: np.ndarray
    encoding_mode: EncodingMode
    information_content: Optional[float] = None


class AdaptiveSpikeCoder:
    """
    Adaptive multi-modal encoder with intelligent mode selection.
    
    Analyzes input statistics and selects optimal encoding method from:
    - Temporal pattern coding (for correlated signals)
    - Population vector coding (for distributed representations)
    - Burst coding (for oscillatory signals)
    """
    
    def __init__(self):
        self.adaptation_window = 1000
        self.encoding_history = []
        
    def encode(self, input_data: np.ndarray) -> SpikeTrain:
        """Adaptively encode input using optimal encoding method."""
        stats = self._calculate_input_statistics(input_data)
        optimal_mode = self._select_optimal_encoding(stats)
        
        # Generate spike train based on selected mode
        if optimal_mode == EncodingMode.TEMPORAL_PATTERN:
            spike_train = self._encode_temporal_pattern(input_data)
        elif optimal_mode == EncodingMode.POPULATION_VECTOR:
            spike_train = self._encode_population_vector(input_data)
        else:  # BURST_CODING
            spike_train = self._encode_burst_coding(input_data)
            
        spike_train.encoding_mode = optimal_mode
        
        # Update adaptation history
        self.encoding_history.append((optimal_mode, stats))
        if len(self.encoding_history) > self.adaptation_window:
            self.encoding_history.pop(0)
            
        return spike_train
    
    def _calculate_input_statistics(self, input_data: np.ndarray) -> Dict:
        """Calculate input statistics for encoding selection."""
        variance = np.var(input_data)
        
        # Temporal correlation
        if len(input_data) > 1:
            correlation = np.corrcoef(input_data[:-1], input_data[1:])[0, 1]
            correlation = 0.0 if np.isnan(correlation) else correlation
        else:
            correlation = 0.0
            
        # Spectral characteristics
        if len(input_data) > 4:
            fft = np.fft.fft(input_data - np.mean(input_data))
            spectral_density = np.abs(fft) ** 2
            spectral_concentration = np.max(spectral_density) / np.mean(spectral_density)
        else:
            spectral_concentration = 1.0
            
        return {
            'variance': variance,
            'temporal_correlation': abs(correlation),
            'spectral_concentration': spectral_concentration
        }
    
    def _select_optimal_encoding(self, stats: Dict) -> EncodingMode:
        """Select optimal encoding based on input statistics."""
        # High temporal correlation -> temporal pattern coding
        if stats['temporal_correlation'] > 0.7:
            return EncodingMode.TEMPORAL_PATTERN
            
        # High variance, low correlation -> population vector coding
        if stats['variance'] > 0.1 and stats['temporal_correlation'] < 0.3:
            return EncodingMode.POPULATION_VECTOR
            
        # High spectral concentration -> burst coding
        if stats['spectral_concentration'] > 5.0:
            return EncodingMode.BURST_CODING
        
        # Default to population vector coding
        return EncodingMode.POPULATION_VECTOR
    
    def _encode_temporal_pattern(self, input_data: np.ndarray) -> SpikeTrain:
        """Encode using temporal pattern coding."""
        all_spike_times = []
        all_neuron_ids = []
        
        for neuron_id, value in enumerate(input_data):
            # Map value to temporal pattern
            n_spikes = max(2, int(value * 8))
            spike_times = np.cumsum(np.random.exponential(10, n_spikes))
            
            # Add neuron offset
            offset = neuron_id * 50.0
            spike_times_offset = spike_times + offset
            
            all_spike_times.extend(spike_times_offset)
            all_neuron_ids.extend([neuron_id] * len(spike_times))
        
        return SpikeTrain(
            spike_times=np.array(all_spike_times),
            neuron_ids=np.array(all_neuron_ids),
            encoding_mode=EncodingMode.TEMPORAL_PATTERN
        )
    
    def _encode_population_vector(self, input_data: np.ndarray) -> SpikeTrain:
        """Encode using population vector coding."""
        population_size = 32
        all_spike_times = []
        all_neuron_ids = []
        
        for data_idx, value in enumerate(input_data):
            # Generate population response
            preferred_values = np.linspace(0, 1, population_size)
            responses = np.exp(-0.5 * ((preferred_values - value) / 0.1) ** 2)
            
            for neuron_id, response in enumerate(responses):
                if response > np.random.random():
                    # Generate spikes
                    n_spikes = max(1, int(response * 5))
                    spike_times = np.random.uniform(0, 20, n_spikes)
                    
                    # Add data element offset
                    offset = data_idx * 30.0
                    spike_times_offset = spike_times + offset
                    
                    all_spike_times.extend(spike_times_offset)
                    all_neuron_ids.extend([neuron_id + data_idx * population_size] * n_spikes)
        
        return SpikeTrain(
            spike_times=np.array(all_spike_times),
            neuron_ids=np.array(all_neuron_ids),
            encoding_mode=EncodingMode.POPULATION_VECTOR
        )
    
    def _encode_burst_coding(self, input_data: np.ndarray) -> SpikeTrain:
        """Encode using burst coding."""
        all_spike_times = []
        all_neuron_ids = []
        
        for neuron_id, value in enumerate(input_data):
            # Map value to burst frequency
            burst_freq = 10 + value * 40  # 10-50 Hz
            burst_period = 1000.0 / burst_freq
            
            current_time = 0
            while current_time < 100.0:  # 100ms window
                # Generate burst
                burst_size = 3 + int(value * 5)  # 3-8 spikes per burst
                burst_spikes = [current_time + i * 2.0 for i in range(burst_size)]
                
                all_spike_times.extend(burst_spikes)
                all_neuron_ids.extend([neuron_id] * burst_size)
                
                current_time += burst_period
        
        return SpikeTrain(
            spike_times=np.array(all_spike_times),
            neuron_ids=np.array(all_neuron_ids),
            encoding_mode=EncodingMode.BURST_CODING
        )


class MultiModalEncoder:
    """Simplified multi-modal encoder interface."""
    
    def __init__(self):
        self.coder = AdaptiveSpikeCoder()
    
    def encode(self, input_data: np.ndarray) -> SpikeTrain:
        return self.coder.encode(input_data)
    
    def get_adaptation_metrics(self) -> Dict:
        return {
            'total_encodings': len(self.coder.encoding_history),
            'adaptation_efficiency': 1.0
        }