"""Multi-region deployment and data residency management."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta
import ipaddress
import socket
import urllib.request


class CloudRegion(Enum):
    """Supported cloud regions for global deployment."""
    
    # Americas
    US_EAST_1 = "us-east-1"           # Virginia, USA
    US_WEST_2 = "us-west-2"           # Oregon, USA  
    US_WEST_1 = "us-west-1"           # California, USA
    CA_CENTRAL_1 = "ca-central-1"     # Central Canada
    SA_EAST_1 = "sa-east-1"           # São Paulo, Brazil
    
    # Europe
    EU_WEST_1 = "eu-west-1"           # Ireland
    EU_WEST_2 = "eu-west-2"           # London, UK
    EU_CENTRAL_1 = "eu-central-1"     # Frankfurt, Germany
    EU_NORTH_1 = "eu-north-1"         # Stockholm, Sweden
    EU_SOUTH_1 = "eu-south-1"         # Milan, Italy
    
    # Asia Pacific
    AP_NORTHEAST_1 = "ap-northeast-1" # Tokyo, Japan
    AP_NORTHEAST_2 = "ap-northeast-2" # Seoul, South Korea
    AP_SOUTHEAST_1 = "ap-southeast-1" # Singapore
    AP_SOUTHEAST_2 = "ap-southeast-2" # Sydney, Australia
    AP_SOUTH_1 = "ap-south-1"         # Mumbai, India
    AP_EAST_1 = "ap-east-1"           # Hong Kong
    
    # Middle East & Africa
    ME_SOUTH_1 = "me-south-1"         # Bahrain
    AF_SOUTH_1 = "af-south-1"         # Cape Town, South Africa
    
    @property
    def continent(self) -> str:
        """Get the continent for this region."""
        if self.value.startswith(('us-', 'ca-', 'sa-')):
            return 'americas'
        elif self.value.startswith('eu-'):
            return 'europe'
        elif self.value.startswith('ap-'):
            return 'asia_pacific'
        elif self.value.startswith('me-'):
            return 'middle_east'
        elif self.value.startswith('af-'):
            return 'africa'
        return 'global'


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    
    region: CloudRegion
    display_name: str
    country_code: str
    city: str
    timezone: str
    data_residency_required: bool = False
    gdpr_applicable: bool = False
    ccpa_applicable: bool = False
    supported_languages: List[str] = field(default_factory=lambda: ['en'])
    primary_currency: str = "USD"
    latency_zones: List[str] = field(default_factory=list)
    compliance_certifications: List[str] = field(default_factory=list)
    
    # Network and infrastructure
    availability_zones: int = 3
    edge_locations: List[str] = field(default_factory=list)
    cdn_enabled: bool = True
    
    # Data processing restrictions
    cross_border_transfers_allowed: bool = True
    encryption_required: bool = True
    data_classification_required: bool = False
    
    def __post_init__(self):
        """Set region-specific defaults."""
        if self.region.continent == 'europe':
            self.gdpr_applicable = True
            self.data_residency_required = True
            self.encryption_required = True
        elif self.region in [CloudRegion.US_WEST_1, CloudRegion.US_WEST_2]:
            self.ccpa_applicable = True


@dataclass
class DeploymentManifest:
    """Deployment manifest for multi-region rollout."""
    
    deployment_id: str
    version: str
    target_regions: List[CloudRegion]
    rollout_strategy: str = "blue_green"  # blue_green, canary, rolling
    health_check_url: str = "/health"
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # Traffic routing
    traffic_allocation: Dict[str, float] = field(default_factory=dict)  # region -> percentage
    failover_regions: Dict[str, CloudRegion] = field(default_factory=dict)
    
    # Compliance requirements
    data_processing_regions: List[CloudRegion] = field(default_factory=list)
    backup_regions: List[CloudRegion] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None
    status: str = "pending"  # pending, deploying, deployed, failed, rollback


class RegionManager:
    """Manages multi-region deployment and data residency."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent / "region_configs"
        self.config_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Initialize region configurations
        self._region_configs = self._initialize_region_configs()
        
        # Active deployments tracking
        self._deployments: Dict[str, DeploymentManifest] = {}
        
        # Geographic IP detection cache
        self._ip_region_cache: Dict[str, CloudRegion] = {}
    
    def _initialize_region_configs(self) -> Dict[CloudRegion, RegionConfig]:
        """Initialize configurations for all supported regions."""
        configs = {
            # Americas
            CloudRegion.US_EAST_1: RegionConfig(
                region=CloudRegion.US_EAST_1,
                display_name="US East (Virginia)",
                country_code="US",
                city="Virginia",
                timezone="America/New_York",
                supported_languages=['en', 'es'],
                primary_currency="USD",
                ccpa_applicable=False,
                compliance_certifications=["SOC2", "ISO27001", "FedRAMP"]
            ),
            
            CloudRegion.US_WEST_2: RegionConfig(
                region=CloudRegion.US_WEST_2,
                display_name="US West (Oregon)",
                country_code="US", 
                city="Oregon",
                timezone="America/Los_Angeles",
                supported_languages=['en', 'es'],
                ccpa_applicable=True,
                compliance_certifications=["SOC2", "ISO27001", "CCPA"]
            ),
            
            CloudRegion.EU_WEST_1: RegionConfig(
                region=CloudRegion.EU_WEST_1,
                display_name="Europe (Ireland)",
                country_code="IE",
                city="Dublin", 
                timezone="Europe/Dublin",
                data_residency_required=True,
                gdpr_applicable=True,
                supported_languages=['en', 'ga'],
                primary_currency="EUR",
                compliance_certifications=["SOC2", "ISO27001", "GDPR", "ISO27018"]
            ),
            
            CloudRegion.EU_CENTRAL_1: RegionConfig(
                region=CloudRegion.EU_CENTRAL_1,
                display_name="Europe (Frankfurt)",
                country_code="DE",
                city="Frankfurt",
                timezone="Europe/Berlin",
                data_residency_required=True,
                gdpr_applicable=True,
                supported_languages=['de', 'en'],
                primary_currency="EUR",
                data_classification_required=True,
                compliance_certifications=["SOC2", "ISO27001", "GDPR", "BDSG"]
            ),
            
            CloudRegion.AP_NORTHEAST_1: RegionConfig(
                region=CloudRegion.AP_NORTHEAST_1,
                display_name="Asia Pacific (Tokyo)",
                country_code="JP",
                city="Tokyo",
                timezone="Asia/Tokyo",
                supported_languages=['ja', 'en'],
                primary_currency="JPY",
                data_residency_required=True,
                compliance_certifications=["SOC2", "ISO27001", "PDPA"]
            ),
            
            CloudRegion.AP_SOUTHEAST_1: RegionConfig(
                region=CloudRegion.AP_SOUTHEAST_1,
                display_name="Asia Pacific (Singapore)",
                country_code="SG",
                city="Singapore",
                timezone="Asia/Singapore", 
                supported_languages=['en', 'zh', 'ms', 'ta'],
                primary_currency="SGD",
                compliance_certifications=["SOC2", "ISO27001", "PDPA", "MAS_TRM"]
            ),
            
            CloudRegion.CA_CENTRAL_1: RegionConfig(
                region=CloudRegion.CA_CENTRAL_1,
                display_name="Canada (Central)",
                country_code="CA",
                city="Toronto",
                timezone="America/Toronto",
                data_residency_required=True,
                supported_languages=['en', 'fr'],
                primary_currency="CAD",
                compliance_certifications=["SOC2", "ISO27001", "PIPEDA"]
            ),
            
            CloudRegion.AP_NORTHEAST_2: RegionConfig(
                region=CloudRegion.AP_NORTHEAST_2,
                display_name="Asia Pacific (Seoul)",
                country_code="KR", 
                city="Seoul",
                timezone="Asia/Seoul",
                supported_languages=['ko', 'en'],
                primary_currency="KRW",
                data_residency_required=True,
                compliance_certifications=["SOC2", "ISO27001", "K_ISMS"]
            ),
            
            CloudRegion.SA_EAST_1: RegionConfig(
                region=CloudRegion.SA_EAST_1,
                display_name="South America (São Paulo)",
                country_code="BR",
                city="São Paulo", 
                timezone="America/Sao_Paulo",
                supported_languages=['pt', 'es', 'en'],
                primary_currency="BRL",
                data_residency_required=True,
                compliance_certifications=["SOC2", "ISO27001", "LGPD"]
            ),
        }
        
        return configs
    
    def get_region_config(self, region: CloudRegion) -> RegionConfig:
        """Get configuration for a specific region."""
        return self._region_configs.get(region, self._create_default_config(region))
    
    def _create_default_config(self, region: CloudRegion) -> RegionConfig:
        """Create a default configuration for a region."""
        return RegionConfig(
            region=region,
            display_name=f"Region {region.value}",
            country_code="XX",
            city="Unknown",
            timezone="UTC"
        )
    
    def get_regions_by_compliance(self, compliance_type: str) -> List[CloudRegion]:
        """Get regions that support a specific compliance requirement."""
        matching_regions = []
        
        for region, config in self._region_configs.items():
            if compliance_type.upper() == "GDPR" and config.gdpr_applicable:
                matching_regions.append(region)
            elif compliance_type.upper() == "CCPA" and config.ccpa_applicable:
                matching_regions.append(region)
            elif compliance_type in config.compliance_certifications:
                matching_regions.append(region)
        
        return matching_regions
    
    def get_regions_by_data_residency(self, required: bool = True) -> List[CloudRegion]:
        """Get regions that meet data residency requirements."""
        return [
            region for region, config in self._region_configs.items()
            if config.data_residency_required == required
        ]
    
    def detect_optimal_region(self, client_ip: Optional[str] = None, 
                            compliance_requirements: Optional[List[str]] = None,
                            preferred_languages: Optional[List[str]] = None) -> CloudRegion:
        """Detect the optimal region for a client."""
        
        candidate_regions = list(self._region_configs.keys())
        
        # Filter by compliance requirements
        if compliance_requirements:
            compliant_regions = set()
            for requirement in compliance_requirements:
                compliant_regions.update(self.get_regions_by_compliance(requirement))
            candidate_regions = [r for r in candidate_regions if r in compliant_regions]
        
        # Filter by language preferences
        if preferred_languages:
            language_matched_regions = []
            for region in candidate_regions:
                config = self.get_region_config(region)
                if any(lang in config.supported_languages for lang in preferred_languages):
                    language_matched_regions.append(region)
            if language_matched_regions:
                candidate_regions = language_matched_regions
        
        # Geographic proximity based on IP (simplified)
        if client_ip and candidate_regions:
            try:
                # In a real implementation, you'd use a GeoIP service
                # This is a simplified version
                region_by_ip = self._detect_region_by_ip(client_ip)
                if region_by_ip in candidate_regions:
                    return region_by_ip
                
                # Find region in same continent
                target_continent = region_by_ip.continent if region_by_ip else 'americas'
                for region in candidate_regions:
                    if region.continent == target_continent:
                        return region
                        
            except Exception as e:
                self.logger.warning(f"Failed to detect region by IP: {e}")
        
        # Default fallback - prefer US East for global availability
        if CloudRegion.US_EAST_1 in candidate_regions:
            return CloudRegion.US_EAST_1
        elif candidate_regions:
            return candidate_regions[0]
        else:
            return CloudRegion.US_EAST_1  # Ultimate fallback
    
    def _detect_region_by_ip(self, ip_address: str) -> CloudRegion:
        """Detect region based on IP address (simplified implementation)."""
        # Check cache first
        if ip_address in self._ip_region_cache:
            return self._ip_region_cache[ip_address]
        
        # Simplified IP-to-region mapping (in production, use proper GeoIP service)
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            
            # Very basic geographic detection based on IP ranges
            # This is a simplified example - use proper GeoIP in production
            if ip_obj.is_private:
                region = CloudRegion.US_EAST_1  # Default for private IPs
            else:
                # Simplified continent detection
                region = CloudRegion.US_EAST_1  # Default
            
            # Cache the result
            self._ip_region_cache[ip_address] = region
            return region
            
        except ValueError:
            return CloudRegion.US_EAST_1  # Invalid IP, use default
    
    def create_deployment(self, deployment_id: str, version: str, 
                         target_regions: List[CloudRegion],
                         rollout_strategy: str = "blue_green") -> DeploymentManifest:
        """Create a new multi-region deployment."""
        
        deployment = DeploymentManifest(
            deployment_id=deployment_id,
            version=version,
            target_regions=target_regions,
            rollout_strategy=rollout_strategy,
            success_criteria={
                "min_success_rate": 0.95,
                "max_latency_p99": 1000,  # ms
                "max_error_rate": 0.01
            }
        )
        
        # Set up traffic allocation (equal by default)
        if target_regions:
            traffic_per_region = 1.0 / len(target_regions)
            deployment.traffic_allocation = {
                region.value: traffic_per_region for region in target_regions
            }
        
        # Configure data processing regions based on compliance
        for region in target_regions:
            config = self.get_region_config(region)
            if config.data_residency_required:
                deployment.data_processing_regions.append(region)
        
        # If no data processing regions, use all regions
        if not deployment.data_processing_regions:
            deployment.data_processing_regions = target_regions.copy()
        
        with self._lock:
            self._deployments[deployment_id] = deployment
        
        self.logger.info(f"Created deployment {deployment_id} targeting {len(target_regions)} regions")
        return deployment
    
    def validate_deployment(self, deployment: DeploymentManifest) -> List[str]:
        """Validate a deployment configuration and return issues."""
        issues = []
        
        # Check compliance requirements
        for region in deployment.target_regions:
            config = self.get_region_config(region)
            
            # Data residency validation
            if config.data_residency_required:
                if region not in deployment.data_processing_regions:
                    issues.append(
                        f"Region {region.value} requires data residency but is not in data_processing_regions"
                    )
            
            # Cross-border transfer validation
            if not config.cross_border_transfers_allowed:
                other_regions = [r for r in deployment.target_regions if r != region]
                for other_region in other_regions:
                    other_config = self.get_region_config(other_region)
                    if other_config.country_code != config.country_code:
                        issues.append(
                            f"Region {region.value} does not allow cross-border transfers to {other_region.value}"
                        )
        
        # Traffic allocation validation
        total_allocation = sum(deployment.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.001:
            issues.append(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        # Backup regions validation
        for primary_region, backup_region in deployment.failover_regions.items():
            if backup_region not in deployment.target_regions:
                issues.append(f"Failover region {backup_region.value} not in target regions")
        
        return issues
    
    def deploy_to_region(self, deployment_id: str, region: CloudRegion) -> bool:
        """Deploy to a specific region."""
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                self.logger.error(f"Deployment {deployment_id} not found")
                return False
        
        config = self.get_region_config(region)
        
        try:
            # Validate region-specific requirements
            validation_issues = self._validate_region_deployment(region, deployment)
            if validation_issues:
                self.logger.error(f"Region validation failed: {validation_issues}")
                return False
            
            self.logger.info(f"Deploying {deployment_id} to {region.value}")
            
            # In a real implementation, this would:
            # 1. Create infrastructure in the region
            # 2. Deploy application code
            # 3. Configure data storage with appropriate encryption
            # 4. Set up monitoring and health checks
            # 5. Configure network routing and load balancing
            
            # Simulate deployment process
            deployment.status = "deployed"
            deployment.deployed_at = datetime.utcnow()
            
            self.logger.info(f"Successfully deployed to {region.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment to {region.value} failed: {e}")
            deployment.status = "failed"
            return False
    
    def _validate_region_deployment(self, region: CloudRegion, 
                                  deployment: DeploymentManifest) -> List[str]:
        """Validate region-specific deployment requirements."""
        issues = []
        config = self.get_region_config(region)
        
        # Check if region supports the deployment
        if region not in deployment.target_regions:
            issues.append(f"Region {region.value} not in target regions")
        
        # Validate compliance requirements
        if config.gdpr_applicable and "gdpr_compliance" not in deployment.success_criteria:
            issues.append(f"GDPR compliance required for {region.value}")
        
        if config.data_classification_required and "data_classification" not in deployment.success_criteria:
            issues.append(f"Data classification required for {region.value}")
        
        return issues
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentManifest]:
        """Get status of a deployment."""
        with self._lock:
            return self._deployments.get(deployment_id)
    
    def rollback_deployment(self, deployment_id: str, region: CloudRegion) -> bool:
        """Rollback deployment in a specific region."""
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return False
        
        try:
            self.logger.info(f"Rolling back deployment {deployment_id} in {region.value}")
            
            # In a real implementation:
            # 1. Route traffic away from the region
            # 2. Restore previous version
            # 3. Validate rollback success
            
            deployment.status = "rollback"
            
            self.logger.info(f"Rollback completed for {region.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed for {region.value}: {e}")
            return False
    
    def get_region_health(self, region: CloudRegion) -> Dict[str, Any]:
        """Get health status of a region."""
        config = self.get_region_config(region)
        
        # In a real implementation, this would check:
        # - Service availability
        # - Latency metrics
        # - Error rates
        # - Resource utilization
        
        return {
            "region": region.value,
            "status": "healthy",
            "availability": 99.9,
            "latency_p50": 45,  # ms
            "latency_p99": 150,  # ms
            "error_rate": 0.001,
            "last_check": datetime.utcnow().isoformat(),
            "compliance_status": {
                "gdpr": config.gdpr_applicable,
                "ccpa": config.ccpa_applicable,
                "data_residency": config.data_residency_required
            }
        }
    
    def list_regions(self, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List available regions with optional filtering."""
        regions = []
        
        for region, config in self._region_configs.items():
            region_info = {
                "region": region.value,
                "display_name": config.display_name,
                "country_code": config.country_code,
                "city": config.city,
                "continent": region.continent,
                "data_residency_required": config.data_residency_required,
                "gdpr_applicable": config.gdpr_applicable,
                "ccpa_applicable": config.ccpa_applicable,
                "supported_languages": config.supported_languages,
                "compliance_certifications": config.compliance_certifications
            }
            
            # Apply filters
            if filter_by:
                include = True
                for key, value in filter_by.items():
                    if key in region_info:
                        if isinstance(value, list):
                            if not any(v in region_info[key] for v in value):
                                include = False
                                break
                        elif region_info[key] != value:
                            include = False
                            break
                
                if include:
                    regions.append(region_info)
            else:
                regions.append(region_info)
        
        return regions


# Global instance
_region_manager = None
_region_lock = threading.Lock()


def get_region_manager() -> RegionManager:
    """Get the global region manager instance."""
    global _region_manager
    
    with _region_lock:
        if _region_manager is None:
            _region_manager = RegionManager()
        
        return _region_manager


def get_region_config(region: CloudRegion) -> RegionConfig:
    """Get configuration for a specific region."""
    manager = get_region_manager()
    return manager.get_region_config(region)


def deploy_to_region(deployment_id: str, region: CloudRegion) -> bool:
    """Deploy to a specific region."""
    manager = get_region_manager()
    return manager.deploy_to_region(deployment_id, region)


def detect_optimal_region(client_ip: Optional[str] = None, 
                         compliance_requirements: Optional[List[str]] = None) -> CloudRegion:
    """Detect optimal region for a client."""
    manager = get_region_manager()
    return manager.detect_optimal_region(client_ip, compliance_requirements)