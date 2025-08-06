"""
Hardware-Optimized Online STDP with Meta-Plasticity Module

Efficient spike-timing dependent plasticity with meta-plasticity rules
optimized for FPGA implementation using bit-shift operations.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlasticityParameters:
    """Parameters for STDP and meta-plasticity mechanisms."""
    a_ltp: float = 0.01  # LTP amplitude
    a_ltd: float = 0.012  # LTD amplitude
    tau_ltp: float = 20.0  # LTP time constant (ms)
    tau_ltd: float = 20.0  # LTD time constant (ms)
    theta_meta: float = 0.005  # Meta-plasticity threshold
    tau_meta: float = 10000.0  # Meta-plasticity time constant (ms)
    target_rate: float = 2.0  # Target firing rate (Hz)
    weight_precision: int = 8  # Bits for weight representation
    use_bit_shift_approximation: bool = True


@dataclass  
class SynapticConnection:
    """Synaptic connection with plasticity state."""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float
    meta_plasticity_factor: float
    last_pre_spike: float
    last_post_spike: float
    homeostatic_scaling: float = 1.0


class BitstiftSTDP:
    """Bit-shift approximation STDP for hardware efficiency."""
    
    def __init__(self, params: PlasticityParameters):
        self.params = params
        self.max_weight = (2 ** params.weight_precision) - 1
        self.min_weight = 0
        
        # Pre-compute bit-shift lookup tables
        self.ltp_shifts = self._compute_shift_table(params.tau_ltp)
        self.ltd_shifts = self._compute_shift_table(params.tau_ltd)
        
    def _compute_shift_table(self, tau: float, max_dt: int = 50) -> Dict[int, int]:
        """Pre-compute bit-shift approximations for exponential decay."""
        shifts = {}
        for dt in range(1, max_dt + 1):
            decay_factor = np.exp(-dt / tau)
            
            # Find best bit-shift approximation
            best_shift = 0
            best_error = float('inf')
            
            for shift in range(8):
                approx_factor = 1.0 / (2 ** shift)
                error = abs(decay_factor - approx_factor)
                if error < best_error:
                    best_error = error
                    best_shift = shift
                    
            shifts[dt] = best_shift
            
        return shifts
    
    def update_synapse(self, connection: SynapticConnection, current_time: float) -> SynapticConnection:
        """Update synaptic weight using bit-shift STDP."""
        dt_spike = connection.last_post_spike - connection.last_pre_spike
        
        if abs(dt_spike) < 50:  # Within plasticity window
            delta_weight = 0
            
            if dt_spike > 0 and dt_spike < 50:  # Post after pre -> LTP
                dt_int = min(int(dt_spike), 49)
                shift = self.ltp_shifts.get(dt_int, 7)
                delta_weight = int(self.params.a_ltp * (self.max_weight - connection.weight)) >> shift
                
            elif dt_spike < 0 and dt_spike > -50:  # Pre after post -> LTD
                dt_int = min(int(abs(dt_spike)), 49)
                shift = self.ltd_shifts.get(dt_int, 7)
                delta_weight = -(int(self.params.a_ltd * connection.weight) >> shift)
            
            # Apply modulations
            delta_weight *= connection.meta_plasticity_factor * connection.homeostatic_scaling
            
            # Update weight with bounds
            new_weight = connection.weight + delta_weight
            connection.weight = max(self.min_weight, min(self.max_weight, new_weight))
        
        return connection


class HomeostasticRegulator:
    """Homeostatic regulation for stable network activity."""
    
    def __init__(self, params: PlasticityParameters):
        self.params = params
        self.neuron_activities = {}
        self.scaling_factors = {}
        self.activity_window = 1000.0  # ms
        
    def update_activity(self, neuron_id: int, spike_time: float) -> None:
        """Update neuron activity tracking."""
        if neuron_id not in self.neuron_activities:
            self.neuron_activities[neuron_id] = []
            
        self.neuron_activities[neuron_id].append(spike_time)
        
        # Remove old spikes
        cutoff_time = spike_time - self.activity_window
        self.neuron_activities[neuron_id] = [
            t for t in self.neuron_activities[neuron_id] if t > cutoff_time
        ]
    
    def calculate_firing_rate(self, neuron_id: int) -> float:
        """Calculate current firing rate."""
        if neuron_id not in self.neuron_activities:
            return 0.0
            
        spike_count = len(self.neuron_activities[neuron_id])
        return (spike_count * 1000.0) / self.activity_window
    
    def get_scaling_factor(self, neuron_id: int) -> float:
        """Get homeostatic scaling factor."""
        current_rate = self.calculate_firing_rate(neuron_id)
        rate_error = self.params.target_rate - current_rate
        
        scaling = 1.0 + 0.1 * rate_error / self.params.target_rate
        return max(0.1, min(3.0, scaling))


class MetaPlasticSTDP:
    """Complete meta-plastic STDP implementation."""
    
    def __init__(self, params: Optional[PlasticityParameters] = None):
        self.params = params or PlasticityParameters()
        self.bitshift_stdp = BitstiftSTDP(self.params)
        self.homeostatic = HomeostasticRegulator(self.params)
        self.synapses = {}
        self.learning_enabled = True
        self.update_count = 0
        
    def add_synapse(self, pre_neuron: int, post_neuron: int, initial_weight: float = 0.5) -> None:
        """Add synaptic connection."""
        synapse_id = (pre_neuron, post_neuron)
        
        connection = SynapticConnection(
            pre_neuron_id=pre_neuron,
            post_neuron_id=post_neuron,
            weight=initial_weight * self.bitshift_stdp.max_weight,
            meta_plasticity_factor=1.0,
            last_pre_spike=-1000.0,
            last_post_spike=-1000.0,
        )
        
        self.synapses[synapse_id] = connection
        
    def process_spike(self, neuron_id: int, spike_time: float) -> None:
        """Process spike and update synapses."""
        if not self.learning_enabled:
            return
            
        self.homeostatic.update_activity(neuron_id, spike_time)
        
        # Update relevant synapses
        for synapse_id, connection in self.synapses.items():
            pre_id, post_id = synapse_id
            
            if pre_id == neuron_id:
                connection.last_pre_spike = spike_time
                connection = self.bitshift_stdp.update_synapse(connection, spike_time)
                connection.homeostatic_scaling = self.homeostatic.get_scaling_factor(post_id)
                
            elif post_id == neuron_id:
                connection.last_post_spike = spike_time
                connection = self.bitshift_stdp.update_synapse(connection, spike_time)
                connection.homeostatic_scaling = self.homeostatic.get_scaling_factor(neuron_id)
            
            self.synapses[synapse_id] = connection
        
        self.update_count += 1
    
    def get_synapse_weight(self, pre_neuron: int, post_neuron: int) -> Optional[float]:
        """Get normalized synapse weight."""
        synapse_id = (pre_neuron, post_neuron)
        if synapse_id not in self.synapses:
            return None
        return self.synapses[synapse_id].weight / self.bitshift_stdp.max_weight
    
    def get_plasticity_statistics(self) -> Dict:
        """Get plasticity statistics."""
        if not self.synapses:
            return {'total_synapses': 0}
            
        weights = [s.weight for s in self.synapses.values()]
        firing_rates = [self.homeostatic.calculate_firing_rate(nid) 
                       for nid in self.homeostatic.neuron_activities.keys()]
        
        return {
            'total_synapses': len(self.synapses),
            'avg_weight': np.mean(weights) / self.bitshift_stdp.max_weight,
            'weight_std': np.std(weights) / self.bitshift_stdp.max_weight,
            'avg_firing_rate': np.mean(firing_rates) if firing_rates else 0.0,
            'update_count': self.update_count
        }