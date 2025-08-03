"""Runtime manager for coordinating FPGA operations."""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import time
import threading
import logging
from enum import Enum

from .interface import FPGAInterface, SpikePacket, FPGAStatus
from .buffers import SpikeBuffer
from .monitoring import PerformanceMonitor, NetworkMonitor


class RuntimeState(Enum):
    """Runtime system states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class RuntimeConfig:
    """Configuration for runtime manager."""
    
    spike_buffer_size: int = 10000
    monitoring_interval_ms: int = 100
    max_spike_rate_hz: int = 1000000  # 1M spikes/second
    enable_monitoring: bool = True
    enable_logging: bool = True
    auto_restart_on_error: bool = True
    performance_logging_interval: int = 10  # seconds


@dataclass
class NetworkConfiguration:
    """Network configuration for runtime."""
    
    total_neurons: int
    input_neurons: List[int] = field(default_factory=list)
    output_neurons: List[int] = field(default_factory=list)
    simulation_timestep_ms: float = 1.0
    max_simulation_time_ms: float = 1000.0


class RuntimeManager:
    """Manages FPGA runtime operations and coordination."""
    
    def __init__(self, fpga_interface: FPGAInterface, config: RuntimeConfig = None):
        self.fpga_interface = fpga_interface
        self.config = config or RuntimeConfig()
        
        # Runtime state
        self.state = RuntimeState.IDLE
        self.network_config: Optional[NetworkConfiguration] = None
        
        # Buffers and monitoring
        self.input_buffer = SpikeBuffer(self.config.spike_buffer_size)
        self.output_buffer = SpikeBuffer(self.config.spike_buffer_size)
        
        if self.config.enable_monitoring:
            self.performance_monitor = PerformanceMonitor()
            self.network_monitor = NetworkMonitor()
        else:
            self.performance_monitor = None
            self.network_monitor = None
        
        # Threading
        self.worker_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Event callbacks
        self.spike_callbacks: List[Callable[[SpikePacket], None]] = []
        self.error_callbacks: List[Callable[[str], None]] = []
        
        self.logger = logging.getLogger(__name__)
    
    def initialize(self, network_config: NetworkConfiguration) -> bool:
        """Initialize runtime with network configuration."""
        self.logger.info("Initializing runtime manager")
        self.state = RuntimeState.INITIALIZING
        
        try:
            # Store network configuration
            self.network_config = network_config
            
            # Connect to FPGA
            if not self.fpga_interface.connect():
                raise RuntimeError("Failed to connect to FPGA")
            
            # Configure network on FPGA
            self._configure_fpga_network()
            
            # Start monitoring if enabled
            if self.config.enable_monitoring:
                self._start_monitoring()
            
            self.state = RuntimeState.IDLE
            self.logger.info("Runtime manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize runtime: {str(e)}")
            self.state = RuntimeState.ERROR
            self._trigger_error_callbacks(str(e))
            return False
    
    def start_simulation(self) -> bool:
        """Start network simulation on FPGA."""
        if self.state != RuntimeState.IDLE:
            self.logger.warning(f"Cannot start simulation in state: {self.state}")
            return False
        
        self.logger.info("Starting network simulation")
        self.state = RuntimeState.RUNNING
        
        try:
            # Start FPGA network
            if not self.fpga_interface.write_register(0x00, 0x1):  # Enable bit
                raise RuntimeError("Failed to start FPGA network")
            
            # Start worker thread for spike processing
            self.shutdown_event.clear()
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            
            self.logger.info("Network simulation started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start simulation: {str(e)}")
            self.state = RuntimeState.ERROR
            self._trigger_error_callbacks(str(e))
            return False
    
    def stop_simulation(self) -> bool:
        """Stop network simulation."""
        if self.state != RuntimeState.RUNNING:
            self.logger.warning(f"Cannot stop simulation in state: {self.state}")
            return False
        
        self.logger.info("Stopping network simulation")
        
        try:
            # Stop FPGA network
            self.fpga_interface.write_register(0x00, 0x0)  # Disable bit
            
            # Signal shutdown and wait for worker thread
            self.shutdown_event.set()
            if self.worker_thread:
                self.worker_thread.join(timeout=5.0)
            
            self.state = RuntimeState.IDLE
            self.logger.info("Network simulation stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop simulation: {str(e)}")
            self.state = RuntimeState.ERROR
            return False
    
    def inject_spike(self, neuron_id: int, timestamp: Optional[int] = None) -> bool:
        """Inject spike into specific neuron."""
        if self.state != RuntimeState.RUNNING:
            return False
        
        if timestamp is None:
            timestamp = int(time.time() * 1000)  # Current time in ms
        
        packet = SpikePacket(timestamp=timestamp, neuron_id=neuron_id)
        
        # Add to input buffer
        success = self.input_buffer.add_spike(packet)
        if not success:
            self.logger.warning(f"Input buffer full, dropping spike for neuron {neuron_id}")
        
        return success
    
    def inject_spike_batch(self, spikes: List[Tuple[int, Optional[int]]]) -> int:
        """Inject multiple spikes. Returns number successfully injected."""
        success_count = 0
        current_time = int(time.time() * 1000)
        
        for neuron_id, timestamp in spikes:
            if timestamp is None:
                timestamp = current_time
            
            if self.inject_spike(neuron_id, timestamp):
                success_count += 1
        
        return success_count
    
    def get_output_spikes(self, max_count: int = 100) -> List[SpikePacket]:
        """Get output spikes from buffer."""
        return self.output_buffer.get_spikes(max_count)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get runtime performance statistics."""
        if not self.performance_monitor:
            return {}
        
        stats = self.performance_monitor.get_stats()
        
        # Add FPGA status
        fpga_status = self.fpga_interface.get_status()
        stats.update({
            "fpga_cycle_count": fpga_status.cycle_count,
            "fpga_network_active": fpga_status.network_active,
            "fpga_total_spikes": fpga_status.total_spikes,
            "fpga_error_count": fpga_status.error_count,
        })
        
        return stats
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network activity statistics."""
        if not self.network_monitor:
            return {}
        
        return self.network_monitor.get_stats()
    
    def register_spike_callback(self, callback: Callable[[SpikePacket], None]) -> None:
        """Register callback for received spikes."""
        self.spike_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for errors."""
        self.error_callbacks.append(callback)
    
    def shutdown(self) -> None:
        """Shutdown runtime manager."""
        self.logger.info("Shutting down runtime manager")
        self.state = RuntimeState.SHUTDOWN
        
        # Stop simulation if running
        if self.state == RuntimeState.RUNNING:
            self.stop_simulation()
        
        # Shutdown monitoring
        if self.monitoring_thread:
            self.shutdown_event.set()
            self.monitoring_thread.join(timeout=2.0)
        
        # Disconnect from FPGA
        self.fpga_interface.disconnect()
        
        self.logger.info("Runtime manager shutdown complete")
    
    def _configure_fpga_network(self) -> None:
        """Configure network parameters on FPGA."""
        if not self.network_config:
            raise RuntimeError("No network configuration available")
        
        # Write network configuration to FPGA registers
        self.fpga_interface.write_register(0x10, self.network_config.total_neurons)
        
        # Configure timestep (convert ms to clock cycles)
        timestep_cycles = int(self.network_config.simulation_timestep_ms * 100000)  # 100MHz clock
        self.fpga_interface.write_register(0x14, timestep_cycles)
        
        self.logger.info(f"Configured FPGA for {self.network_config.total_neurons} neurons")
    
    def _worker_loop(self) -> None:
        """Main worker loop for spike processing."""
        self.logger.info("Worker thread started")
        
        while not self.shutdown_event.is_set():
            try:
                # Process input spikes (send to FPGA)
                self._process_input_spikes()
                
                # Process output spikes (receive from FPGA)
                self._process_output_spikes()
                
                # Update monitoring
                if self.performance_monitor:
                    self.performance_monitor.update()
                
                # Small delay to prevent busy waiting
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {str(e)}")
                if not self.config.auto_restart_on_error:
                    self.state = RuntimeState.ERROR
                    break
        
        self.logger.info("Worker thread stopped")
    
    def _process_input_spikes(self) -> None:
        """Process spikes from input buffer and send to FPGA."""
        spikes = self.input_buffer.get_spikes(100)  # Process up to 100 spikes
        
        for spike in spikes:
            success = self.fpga_interface.send_spike(spike)
            if self.performance_monitor and success:
                self.performance_monitor.record_spike_sent()
    
    def _process_output_spikes(self) -> None:
        """Receive spikes from FPGA and add to output buffer."""
        spikes = self.fpga_interface.receive_batch_spikes(100, timeout_ms=1)
        
        for spike in spikes:
            self.output_buffer.add_spike(spike)
            
            if self.performance_monitor:
                self.performance_monitor.record_spike_received()
            
            if self.network_monitor:
                self.network_monitor.record_spike(spike.neuron_id, spike.timestamp)
            
            # Trigger callbacks
            for callback in self.spike_callbacks:
                try:
                    callback(spike)
                except Exception as e:
                    self.logger.error(f"Error in spike callback: {str(e)}")
    
    def _start_monitoring(self) -> None:
        """Start monitoring thread."""
        if self.monitoring_thread:
            return
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Monitoring thread started")
    
    def _monitoring_loop(self) -> None:
        """Monitoring loop for performance metrics."""
        last_log_time = time.time()
        
        while not self.shutdown_event.wait(self.config.monitoring_interval_ms / 1000.0):
            try:
                current_time = time.time()
                
                # Log performance stats periodically
                if (current_time - last_log_time) >= self.config.performance_logging_interval:
                    stats = self.get_performance_stats()
                    self.logger.info(f"Performance stats: {stats}")
                    last_log_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
    
    def _trigger_error_callbacks(self, error_message: str) -> None:
        """Trigger all registered error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error_message)
            except Exception as e:
                self.logger.error(f"Error in error callback: {str(e)}")