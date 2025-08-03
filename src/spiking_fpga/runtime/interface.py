"""FPGA interface and communication protocols."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import struct
import time
import logging


class CommunicationProtocol(Enum):
    """Supported communication protocols."""
    AXI4_LITE = "axi4_lite"
    UART = "uart"
    PCIE = "pcie"
    JTAG = "jtag"
    USB = "usb"


@dataclass
class SpikePacket:
    """Spike data packet for communication."""
    
    timestamp: int
    neuron_id: int
    spike_data: Optional[bytes] = None
    packet_type: str = "spike"
    
    def to_bytes(self) -> bytes:
        """Convert spike packet to bytes for transmission."""
        # Format: [timestamp:32][neuron_id:32][type:8][data_length:8][data:variable]
        header = struct.pack('<III', self.timestamp, self.neuron_id, 
                           ord(self.packet_type[0]) if self.packet_type else 0)
        if self.spike_data:
            header += struct.pack('<I', len(self.spike_data)) + self.spike_data
        else:
            header += struct.pack('<I', 0)
        return header
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'SpikePacket':
        """Create spike packet from received bytes."""
        timestamp, neuron_id, ptype, data_len = struct.unpack('<IIII', data[:16])
        spike_data = data[16:16+data_len] if data_len > 0 else None
        
        return cls(
            timestamp=timestamp,
            neuron_id=neuron_id,
            spike_data=spike_data,
            packet_type=chr(ptype) if ptype > 0 else "spike"
        )


@dataclass
class FPGAStatus:
    """FPGA runtime status information."""
    
    network_active: bool = False
    cycle_count: int = 0
    total_spikes: int = 0
    error_count: int = 0
    temperature: Optional[float] = None
    voltage: Optional[float] = None
    frequency_mhz: Optional[float] = None
    utilization_percent: Optional[float] = None


class FPGAInterface(ABC):
    """Abstract base class for FPGA communication interfaces."""
    
    def __init__(self, device_path: str):
        self.device_path = device_path
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to FPGA."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to FPGA."""
        pass
    
    @abstractmethod
    def send_spike(self, packet: SpikePacket) -> bool:
        """Send spike packet to FPGA."""
        pass
    
    @abstractmethod
    def receive_spike(self, timeout_ms: int = 1000) -> Optional[SpikePacket]:
        """Receive spike packet from FPGA."""
        pass
    
    @abstractmethod
    def write_register(self, address: int, value: int) -> bool:
        """Write to FPGA register."""
        pass
    
    @abstractmethod
    def read_register(self, address: int) -> Optional[int]:
        """Read from FPGA register."""
        pass
    
    @abstractmethod
    def get_status(self) -> FPGAStatus:
        """Get FPGA runtime status."""
        pass
    
    def send_batch_spikes(self, packets: List[SpikePacket]) -> int:
        """Send multiple spike packets. Returns number successfully sent."""
        success_count = 0
        for packet in packets:
            if self.send_spike(packet):
                success_count += 1
            else:
                self.logger.warning(f"Failed to send spike packet for neuron {packet.neuron_id}")
        return success_count
    
    def receive_batch_spikes(self, max_count: int, timeout_ms: int = 1000) -> List[SpikePacket]:
        """Receive multiple spike packets."""
        packets = []
        start_time = time.time()
        
        while len(packets) < max_count:
            remaining_time = max(0, timeout_ms - int((time.time() - start_time) * 1000))
            if remaining_time == 0:
                break
                
            packet = self.receive_spike(remaining_time)
            if packet:
                packets.append(packet)
            else:
                break
        
        return packets


class AXI4LiteInterface(FPGAInterface):
    """AXI4-Lite interface implementation (placeholder for actual hardware)."""
    
    def __init__(self, device_path: str = "/dev/mem"):
        super().__init__(device_path)
        self.base_address = 0x40000000  # Example base address
        self.spike_tx_reg = 0x00
        self.spike_rx_reg = 0x04
        self.control_reg = 0x08
        self.status_reg = 0x0C
    
    def connect(self) -> bool:
        """Connect to AXI4-Lite interface."""
        try:
            # In a real implementation, this would open memory-mapped device
            self.logger.info(f"Connecting to AXI4-Lite interface at {self.device_path}")
            self.is_connected = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to AXI4-Lite: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from AXI4-Lite interface."""
        self.is_connected = False
        self.logger.info("Disconnected from AXI4-Lite interface")
        return True
    
    def send_spike(self, packet: SpikePacket) -> bool:
        """Send spike packet via AXI4-Lite."""
        if not self.is_connected:
            return False
        
        try:
            # Convert packet to 32-bit word for transmission
            spike_word = (packet.timestamp << 16) | (packet.neuron_id & 0xFFFF)
            return self.write_register(self.spike_tx_reg, spike_word)
        except Exception as e:
            self.logger.error(f"Failed to send spike: {str(e)}")
            return False
    
    def receive_spike(self, timeout_ms: int = 1000) -> Optional[SpikePacket]:
        """Receive spike packet via AXI4-Lite."""
        if not self.is_connected:
            return None
        
        try:
            # Poll for new spike data
            start_time = time.time()
            while (time.time() - start_time) * 1000 < timeout_ms:
                spike_word = self.read_register(self.spike_rx_reg)
                if spike_word and spike_word != 0:
                    timestamp = (spike_word >> 16) & 0xFFFF
                    neuron_id = spike_word & 0xFFFF
                    return SpikePacket(timestamp=timestamp, neuron_id=neuron_id)
                time.sleep(0.001)  # 1ms polling interval
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to receive spike: {str(e)}")
            return None
    
    def write_register(self, address: int, value: int) -> bool:
        """Write to memory-mapped register."""
        # Placeholder implementation - would use actual memory mapping
        self.logger.debug(f"AXI Write: addr=0x{address:08x}, value=0x{value:08x}")
        return True
    
    def read_register(self, address: int) -> Optional[int]:
        """Read from memory-mapped register."""
        # Placeholder implementation - would use actual memory mapping
        self.logger.debug(f"AXI Read: addr=0x{address:08x}")
        return 0  # Dummy value
    
    def get_status(self) -> FPGAStatus:
        """Get FPGA status via AXI4-Lite."""
        if not self.is_connected:
            return FPGAStatus()
        
        try:
            status_word = self.read_register(self.status_reg)
            return FPGAStatus(
                network_active=bool(status_word & 0x1),
                cycle_count=(status_word >> 1) & 0x7FFFFFFF,
                total_spikes=0,  # Would read from separate register
                error_count=0,   # Would read from separate register
            )
        except Exception as e:
            self.logger.error(f"Failed to get status: {str(e)}")
            return FPGAStatus()


class UARTInterface(FPGAInterface):
    """UART interface implementation."""
    
    def __init__(self, device_path: str = "/dev/ttyUSB0", baudrate: int = 115200):
        super().__init__(device_path)
        self.baudrate = baudrate
        self.serial_port = None
    
    def connect(self) -> bool:
        """Connect to UART interface."""
        try:
            import serial
            self.serial_port = serial.Serial(self.device_path, self.baudrate, timeout=1)
            self.is_connected = True
            self.logger.info(f"Connected to UART at {self.device_path}")
            return True
        except ImportError:
            self.logger.error("pyserial not installed - cannot use UART interface")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to UART: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from UART interface."""
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
        self.is_connected = False
        self.logger.info("Disconnected from UART interface")
        return True
    
    def send_spike(self, packet: SpikePacket) -> bool:
        """Send spike packet via UART."""
        if not self.is_connected or not self.serial_port:
            return False
        
        try:
            data = packet.to_bytes()
            self.serial_port.write(data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send spike via UART: {str(e)}")
            return False
    
    def receive_spike(self, timeout_ms: int = 1000) -> Optional[SpikePacket]:
        """Receive spike packet via UART."""
        if not self.is_connected or not self.serial_port:
            return None
        
        try:
            self.serial_port.timeout = timeout_ms / 1000.0
            data = self.serial_port.read(16)  # Minimum packet size
            if len(data) >= 16:
                return SpikePacket.from_bytes(data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to receive spike via UART: {str(e)}")
            return None
    
    def write_register(self, address: int, value: int) -> bool:
        """Write register via UART command protocol."""
        if not self.is_connected or not self.serial_port:
            return False
        
        try:
            cmd = struct.pack('<BII', 0x01, address, value)  # Write command
            self.serial_port.write(cmd)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write register via UART: {str(e)}")
            return False
    
    def read_register(self, address: int) -> Optional[int]:
        """Read register via UART command protocol."""
        if not self.is_connected or not self.serial_port:
            return None
        
        try:
            cmd = struct.pack('<BI', 0x02, address)  # Read command
            self.serial_port.write(cmd)
            
            response = self.serial_port.read(4)
            if len(response) == 4:
                return struct.unpack('<I', response)[0]
            return None
        except Exception as e:
            self.logger.error(f"Failed to read register via UART: {str(e)}")
            return None
    
    def get_status(self) -> FPGAStatus:
        """Get FPGA status via UART."""
        # Read multiple status registers
        status = FPGAStatus()
        
        control_reg = self.read_register(0x00)
        if control_reg is not None:
            status.network_active = bool(control_reg & 0x1)
        
        cycle_reg = self.read_register(0x04)
        if cycle_reg is not None:
            status.cycle_count = cycle_reg
        
        return status


def create_interface(protocol: CommunicationProtocol, 
                    device_path: str, **kwargs) -> FPGAInterface:
    """Factory function to create appropriate FPGA interface."""
    
    if protocol == CommunicationProtocol.AXI4_LITE:
        return AXI4LiteInterface(device_path)
    elif protocol == CommunicationProtocol.UART:
        baudrate = kwargs.get('baudrate', 115200)
        return UARTInterface(device_path, baudrate)
    else:
        raise ValueError(f"Unsupported communication protocol: {protocol}")