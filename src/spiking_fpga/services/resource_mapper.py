"""
Resource mapping and placement service for FPGA implementation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

from ..models.network import SNNNetwork, Layer, Connection
from ..models.fpga import FPGATarget, ResourceUtilization, ResourceEstimator


@dataclass
class PlacementResult:
    """Results from resource mapping and placement."""
    success: bool = False
    resource_usage: Optional[ResourceUtilization] = None
    layer_placements: Dict[str, Dict] = None
    routing_delays: Dict[str, float] = None
    placement_score: float = 0.0
    violations: List[str] = None
    
    def __post_init__(self):
        if self.layer_placements is None:
            self.layer_placements = {}
        if self.routing_delays is None:
            self.routing_delays = {}
        if self.violations is None:
            self.violations = []


class ResourceMapper:
    """Maps SNN networks to FPGA resources with placement optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def map_network(self, network: SNNNetwork, target: FPGATarget) -> PlacementResult:
        """
        Map network to FPGA resources with optimized placement.
        
        Args:
            network: SNN network to map
            target: Target FPGA platform
            
        Returns:
            PlacementResult with resource assignments and utilization
        """
        self.logger.info(f"Mapping network '{network.name}' to {target.target}")
        
        result = PlacementResult()
        
        try:
            # Estimate total resource requirements
            total_resources = self.estimate_resources(network, target)
            result.resource_usage = total_resources
            
            # Check if network fits on target FPGA
            violations = total_resources.check_constraints(target)
            if violations:
                result.violations = violations
                self.logger.warning(f"Resource constraints violated: {violations}")
                # Continue with placement but mark as failed
                result.success = False
            else:
                result.success = True
            
            # Perform layer placement optimization
            layer_placements = self._optimize_layer_placement(network, target, total_resources)
            result.layer_placements = layer_placements
            
            # Estimate routing delays between layers
            routing_delays = self._estimate_routing_delays(network, layer_placements, target)
            result.routing_delays = routing_delays
            
            # Calculate placement quality score
            result.placement_score = self._calculate_placement_score(
                network, layer_placements, routing_delays, total_resources, target
            )
            
            self.logger.info(f"Placement completed with score {result.placement_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Resource mapping failed: {str(e)}")
            result.success = False
            result.violations.append(str(e))
        
        return result
    
    def estimate_resources(self, network: SNNNetwork, target: FPGATarget) -> ResourceUtilization:
        """Quick resource estimation for network on target FPGA."""
        # Convert network to dictionary format for ResourceEstimator
        network_spec = {
            'layers': [
                {
                    'id': layer.id,
                    'size': layer.size,
                    'neuron_model': layer.neuron_model,
                    'plasticity_enabled': layer.plasticity_enabled
                }
                for layer in network.layers
            ],
            'connections': [
                {
                    'source_layer': conn.source_layer,
                    'target_layer': conn.target_layer,
                    'connectivity': conn.connectivity.value,
                    'sparsity': conn.sparsity,
                    'plasticity_enabled': getattr(conn, 'plasticity_enabled', False)
                }
                for conn in network.connections
            ]
        }
        
        return ResourceEstimator.estimate_total_resources(network_spec)
    
    def _optimize_layer_placement(self, network: SNNNetwork, target: FPGATarget,
                                total_resources: ResourceUtilization) -> Dict[str, Dict]:
        """Optimize placement of layers on FPGA fabric."""
        placements = {}
        
        # Simple grid-based placement algorithm
        # Divide FPGA into regions and assign layers
        
        # Calculate FPGA grid dimensions based on LUT distribution
        fabric_width = int(math.sqrt(target.specs.luts))
        fabric_height = target.specs.luts // fabric_width
        
        # Sort layers by size (largest first for better placement)
        sorted_layers = sorted(network.layers, key=lambda l: l.size, reverse=True)
        
        # Track occupied regions
        occupied_regions = set()
        
        for i, layer in enumerate(sorted_layers):
            # Estimate region size needed for this layer
            layer_resources = ResourceEstimator.estimate_neuron_resources(
                layer.size, layer.neuron_model
            )
            
            # Calculate region dimensions
            luts_needed = layer_resources.luts_used
            region_width = min(fabric_width, int(math.sqrt(luts_needed)) + 1)
            region_height = min(fabric_height, (luts_needed // region_width) + 1)
            
            # Find best position for this layer
            best_x, best_y = self._find_best_placement_position(
                region_width, region_height, fabric_width, fabric_height,
                occupied_regions, network, layer
            )
            
            # Record placement
            placements[layer.id] = {
                'x_start': best_x,
                'y_start': best_y,
                'x_end': best_x + region_width - 1,
                'y_end': best_y + region_height - 1,
                'width': region_width,
                'height': region_height,
                'luts_used': luts_needed,
                'center_x': best_x + region_width // 2,
                'center_y': best_y + region_height // 2
            }
            
            # Mark regions as occupied
            for x in range(best_x, best_x + region_width):
                for y in range(best_y, best_y + region_height):
                    occupied_regions.add((x, y))
        
        return placements
    
    def _find_best_placement_position(self, width: int, height: int,
                                    fabric_width: int, fabric_height: int,
                                    occupied: set, network: SNNNetwork,
                                    layer: Layer) -> Tuple[int, int]:
        """Find optimal placement position for a layer."""
        best_x, best_y = 0, 0
        best_score = float('inf')
        
        # Try all possible positions
        for x in range(fabric_width - width + 1):
            for y in range(fabric_height - height + 1):
                # Check if position is available
                position_available = True
                for dx in range(width):
                    for dy in range(height):
                        if (x + dx, y + dy) in occupied:
                            position_available = False
                            break
                    if not position_available:
                        break
                
                if position_available:
                    # Calculate placement score (minimize routing distance)
                    score = self._calculate_position_score(
                        x, y, width, height, network, layer
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_x, best_y = x, y
        
        return best_x, best_y
    
    def _calculate_position_score(self, x: int, y: int, width: int, height: int,
                                network: SNNNetwork, layer: Layer) -> float:
        """Calculate quality score for a layer placement position."""
        center_x = x + width // 2
        center_y = y + height // 2
        
        total_distance = 0.0
        connection_count = 0
        
        # Calculate average distance to connected layers (simplified)
        for conn in network.connections:
            if conn.source_layer == layer.id or conn.target_layer == layer.id:
                # For now, use heuristic that prefers center placement
                # In real implementation, would consider actual layer positions
                center_distance = math.sqrt((center_x - 50)**2 + (center_y - 50)**2)
                total_distance += center_distance
                connection_count += 1
        
        # Prefer positions that minimize average routing distance
        if connection_count > 0:
            return total_distance / connection_count
        else:
            # For unconnected layers, prefer center placement
            return math.sqrt((center_x - 50)**2 + (center_y - 50)**2)
    
    def _estimate_routing_delays(self, network: SNNNetwork,
                               placements: Dict[str, Dict],
                               target: FPGATarget) -> Dict[str, float]:
        """Estimate routing delays between layers based on placement."""
        delays = {}
        
        for conn in network.connections:
            source_placement = placements.get(conn.source_layer)
            target_placement = placements.get(conn.target_layer)
            
            if source_placement and target_placement:
                # Calculate Manhattan distance between layer centers
                dx = abs(source_placement['center_x'] - target_placement['center_x'])
                dy = abs(source_placement['center_y'] - target_placement['center_y'])
                manhattan_distance = dx + dy
                
                # Estimate delay based on distance (rough approximation)
                # Typical routing delay: ~0.1ns per LUT hop
                routing_delay_ns = manhattan_distance * 0.1
                
                # Add base logic delay
                base_delay_ns = 2.0
                total_delay_ns = base_delay_ns + routing_delay_ns
                
                connection_key = f"{conn.source_layer}->{conn.target_layer}"
                delays[connection_key] = total_delay_ns
        
        return delays
    
    def _calculate_placement_score(self, network: SNNNetwork,
                                 placements: Dict[str, Dict],
                                 routing_delays: Dict[str, float],
                                 resources: ResourceUtilization,
                                 target: FPGATarget) -> float:
        """Calculate overall placement quality score."""
        score = 0.0
        
        # Factor 1: Resource utilization efficiency (0-100)
        utilization_pct = resources.get_utilization_percentages(target)
        lut_utilization = utilization_pct['luts']
        bram_utilization = utilization_pct['bram']
        
        # Prefer 60-80% utilization (good balance of efficiency and timing)
        optimal_utilization = 70.0
        lut_score = 100 - abs(lut_utilization - optimal_utilization)
        bram_score = 100 - abs(bram_utilization - optimal_utilization)
        utilization_score = (lut_score + bram_score) / 2
        
        # Factor 2: Routing efficiency (minimize average delay)
        if routing_delays:
            avg_delay = sum(routing_delays.values()) / len(routing_delays)
            # Convert to score (lower delay = higher score)
            routing_score = max(0, 100 - avg_delay * 10)  # Rough scaling
        else:
            routing_score = 100  # No connections = perfect routing
        
        # Factor 3: Placement compactness (minimize area)
        if placements:
            total_area = 0
            for placement in placements.values():
                area = placement['width'] * placement['height']
                total_area += area
            
            # Calculate fabric utilization
            fabric_area = int(math.sqrt(target.specs.luts)) ** 2
            area_efficiency = (total_area / fabric_area) * 100
            
            # Prefer compact placements but not too dense
            compactness_score = max(0, 100 - abs(area_efficiency - 50))
        else:
            compactness_score = 0
        
        # Weighted combination of factors
        score = (
            0.4 * utilization_score +
            0.4 * routing_score +
            0.2 * compactness_score
        )
        
        return score
    
    def generate_placement_report(self, result: PlacementResult,
                                network: SNNNetwork, target: FPGATarget) -> str:
        """Generate detailed placement report."""
        report = []
        report.append("FPGA Placement and Routing Report")
        report.append("=" * 50)
        report.append("")
        
        if result.resource_usage:
            util_pct = result.resource_usage.get_utilization_percentages(target)
            report.append("Resource Utilization:")
            report.append(f"  LUTs: {result.resource_usage.luts_used:,} / {target.specs.luts:,} ({util_pct['luts']:.1f}%)")
            report.append(f"  BRAM: {result.resource_usage.bram_used//1024:,} KB / {target.specs.bram_bits//1024:,} KB ({util_pct['bram']:.1f}%)")
            report.append(f"  DSP: {result.resource_usage.dsp_used:,} / {target.specs.dsp_slices:,} ({util_pct['dsp']:.1f}%)")
            report.append("")
        
        if result.layer_placements:
            report.append("Layer Placements:")
            for layer_id, placement in result.layer_placements.items():
                report.append(f"  {layer_id}:")
                report.append(f"    Position: ({placement['x_start']},{placement['y_start']}) to ({placement['x_end']},{placement['y_end']})")
                report.append(f"    Size: {placement['width']} x {placement['height']}")
                report.append(f"    LUTs: {placement['luts_used']:,}")
            report.append("")
        
        if result.routing_delays:
            report.append("Routing Delays:")
            for connection, delay in result.routing_delays.items():
                report.append(f"  {connection}: {delay:.2f} ns")
            report.append("")
        
        report.append(f"Placement Score: {result.placement_score:.2f}/100")
        
        if result.violations:
            report.append("")
            report.append("Constraint Violations:")
            for violation in result.violations:
                report.append(f"  - {violation}")
        
        return "\n".join(report)