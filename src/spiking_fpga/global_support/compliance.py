"""Global compliance management with automated data residency and cross-border transfer controls."""

import logging
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import json
from pathlib import Path

from .regions import CloudRegion, RegionManager, get_region_manager
from ..enterprise.compliance_framework import ComplianceRegulation, DataSubject


class DataClassification(Enum):
    """Data classification levels for compliance."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class TransferMechanism(Enum):
    """Legal mechanisms for cross-border data transfers."""
    ADEQUACY_DECISION = "adequacy_decision"    # EU adequacy decision
    STANDARD_CONTRACTUAL_CLAUSES = "scc"      # Standard Contractual Clauses
    BINDING_CORPORATE_RULES = "bcr"           # Binding Corporate Rules
    CERTIFICATION = "certification"           # Certification schemes
    CODE_OF_CONDUCT = "code_of_conduct"      # Codes of conduct
    CONSENT = "consent"                       # User consent
    CONTRACT_PERFORMANCE = "contract"         # Contract performance
    VITAL_INTERESTS = "vital_interests"       # Vital interests
    PUBLIC_INTEREST = "public_interest"       # Public interest


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for compliance."""
    
    record_id: str
    data_subject_id: str
    processing_purpose: str
    data_categories: List[str]
    processing_regions: List[CloudRegion]
    storage_regions: List[CloudRegion]
    data_classification: DataClassification
    
    # Legal basis and consent
    legal_basis: str
    consent_obtained: bool
    consent_timestamp: Optional[datetime] = None
    
    # Cross-border transfers
    cross_border_transfers: List[Dict[str, Any]] = field(default_factory=list)
    transfer_mechanisms: List[TransferMechanism] = field(default_factory=list)
    
    # Retention and lifecycle
    retention_period_days: int = 365
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    deletion_scheduled: Optional[datetime] = None
    
    # Security and encryption
    encryption_in_transit: bool = True
    encryption_at_rest: bool = True
    pseudonymization: bool = False
    anonymization: bool = False


@dataclass
class DataResidencyRequirement:
    """Data residency requirement for a specific regulation."""
    
    regulation: ComplianceRegulation
    applicable_regions: List[CloudRegion]
    prohibited_regions: List[CloudRegion] = field(default_factory=list)
    required_safeguards: List[str] = field(default_factory=list)
    transfer_restrictions: Dict[str, Any] = field(default_factory=dict)
    
    def allows_processing_in_region(self, region: CloudRegion) -> bool:
        """Check if data processing is allowed in a specific region."""
        if self.prohibited_regions and region in self.prohibited_regions:
            return False
        
        if self.applicable_regions and region not in self.applicable_regions:
            return False
        
        return True


class GlobalComplianceManager:
    """Manages global compliance requirements and data residency."""
    
    def __init__(self, region_manager: Optional[RegionManager] = None):
        self.region_manager = region_manager or get_region_manager()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Compliance tracking
        self._processing_records: Dict[str, DataProcessingRecord] = {}
        self._data_subject_regions: Dict[str, CloudRegion] = {}
        
        # Residency requirements by regulation
        self._residency_requirements = self._initialize_residency_requirements()
        
        # Cross-border transfer approvals
        self._transfer_approvals: Dict[str, Dict[str, Any]] = {}
        
        # Compliance monitoring
        self._compliance_violations: List[Dict[str, Any]] = []
        self._monitoring_active = True
        
    def _initialize_residency_requirements(self) -> Dict[ComplianceRegulation, DataResidencyRequirement]:
        """Initialize data residency requirements for regulations."""
        requirements = {}
        
        # GDPR Requirements
        eu_regions = [
            CloudRegion.EU_WEST_1, CloudRegion.EU_WEST_2, 
            CloudRegion.EU_CENTRAL_1, CloudRegion.EU_NORTH_1,
            CloudRegion.EU_SOUTH_1
        ]
        
        requirements[ComplianceRegulation.GDPR] = DataResidencyRequirement(
            regulation=ComplianceRegulation.GDPR,
            applicable_regions=eu_regions,
            required_safeguards=[
                "encryption_at_rest",
                "encryption_in_transit", 
                "access_logging",
                "data_minimization",
                "purpose_limitation"
            ],
            transfer_restrictions={
                "adequacy_required": True,
                "scc_required": True,
                "consent_for_sensitive": True,
                "data_protection_impact_assessment": True
            }
        )
        
        # CCPA Requirements  
        us_regions = [
            CloudRegion.US_WEST_1, CloudRegion.US_WEST_2,
            CloudRegion.US_EAST_1
        ]
        
        requirements[ComplianceRegulation.CCPA] = DataResidencyRequirement(
            regulation=ComplianceRegulation.CCPA,
            applicable_regions=us_regions,
            required_safeguards=[
                "encryption_at_rest",
                "access_logging",
                "data_deletion_capability"
            ],
            transfer_restrictions={
                "disclosure_notification": True,
                "opt_out_required": True
            }
        )
        
        # Canada PIPEDA
        ca_regions = [CloudRegion.CA_CENTRAL_1]
        
        requirements[ComplianceRegulation.PIPEDA] = DataResidencyRequirement(
            regulation=ComplianceRegulation.PIPEDA,
            applicable_regions=ca_regions,
            required_safeguards=[
                "encryption_at_rest",
                "encryption_in_transit",
                "consent_management",
                "breach_notification"
            ]
        )
        
        # Brazil LGPD
        br_regions = [CloudRegion.SA_EAST_1]
        
        requirements[ComplianceRegulation.LGPD] = DataResidencyRequirement(
            regulation=ComplianceRegulation.LGPD,
            applicable_regions=br_regions,
            required_safeguards=[
                "encryption_at_rest", 
                "data_minimization",
                "consent_management",
                "data_controller_designation"
            ]
        )
        
        return requirements
    
    def register_data_processing(self, data_subject: DataSubject, 
                                processing_purpose: str,
                                data_categories: List[str],
                                processing_regions: List[CloudRegion],
                                data_classification: DataClassification = DataClassification.INTERNAL,
                                retention_days: int = 365) -> str:
        """Register a data processing activity."""
        
        record_id = f"proc_{data_subject.subject_id}_{int(datetime.utcnow().timestamp())}"
        
        # Determine storage regions based on compliance requirements
        storage_regions = self._determine_compliant_storage_regions(
            data_subject, processing_regions, data_classification
        )
        
        record = DataProcessingRecord(
            record_id=record_id,
            data_subject_id=data_subject.subject_id,
            processing_purpose=processing_purpose,
            data_categories=data_categories,
            processing_regions=processing_regions,
            storage_regions=storage_regions,
            data_classification=data_classification,
            legal_basis="consent",  # Default, should be specified
            consent_obtained=data_subject.consent_preferences.get("data_processing", False),
            retention_period_days=retention_days
        )
        
        # Check for cross-border transfers
        cross_border_transfers = self._identify_cross_border_transfers(
            data_subject, processing_regions, storage_regions
        )
        
        if cross_border_transfers:
            record.cross_border_transfers = cross_border_transfers
            record.transfer_mechanisms = self._determine_transfer_mechanisms(
                data_subject, cross_border_transfers
            )
        
        with self._lock:
            self._processing_records[record_id] = record
            self._data_subject_regions[data_subject.subject_id] = self._get_data_subject_region(data_subject)
        
        # Validate compliance
        violations = self._validate_processing_compliance(record)
        if violations:
            self.logger.warning(f"Compliance violations for {record_id}: {violations}")
            self._compliance_violations.extend(violations)
        
        self.logger.info(f"Registered data processing: {record_id}")
        return record_id
    
    def _determine_compliant_storage_regions(self, data_subject: DataSubject,
                                           processing_regions: List[CloudRegion],
                                           data_classification: DataClassification) -> List[CloudRegion]:
        """Determine compliant storage regions for data subject."""
        
        # Get applicable regulations for data subject
        applicable_regulations = self._get_applicable_regulations(data_subject)
        
        # Find regions that satisfy all applicable regulations
        compliant_regions = set(processing_regions)
        
        for regulation in applicable_regulations:
            requirement = self._residency_requirements.get(regulation)
            if requirement:
                # Filter regions that meet this regulation's requirements
                regulation_compliant = [
                    region for region in compliant_regions
                    if requirement.allows_processing_in_region(region)
                ]
                compliant_regions = set(regulation_compliant)
        
        # Ensure at least one region is available
        if not compliant_regions and processing_regions:
            # Fallback to processing regions with additional safeguards
            self.logger.warning(f"No fully compliant regions found, using processing regions with safeguards")
            compliant_regions = set(processing_regions)
        
        return list(compliant_regions)
    
    def _get_applicable_regulations(self, data_subject: DataSubject) -> List[ComplianceRegulation]:
        """Get regulations applicable to a data subject."""
        regulations = []
        
        # Determine regulations based on data subject's location/citizenship
        subject_region = self._get_data_subject_region(data_subject)
        region_config = self.region_manager.get_region_config(subject_region)
        
        if region_config.gdpr_applicable:
            regulations.append(ComplianceRegulation.GDPR)
        
        if region_config.ccpa_applicable:
            regulations.append(ComplianceRegulation.CCPA)
        
        # Add other regulations based on region
        if subject_region == CloudRegion.CA_CENTRAL_1:
            regulations.append(ComplianceRegulation.PIPEDA)
        elif subject_region == CloudRegion.SA_EAST_1:
            regulations.append(ComplianceRegulation.LGPD)
        
        return regulations
    
    def _get_data_subject_region(self, data_subject: DataSubject) -> CloudRegion:
        """Determine data subject's primary region."""
        # Map region codes to CloudRegions
        region_mapping = {
            'EU': CloudRegion.EU_WEST_1,
            'US': CloudRegion.US_EAST_1, 
            'CA': CloudRegion.CA_CENTRAL_1,
            'BR': CloudRegion.SA_EAST_1,
            'JP': CloudRegion.AP_NORTHEAST_1,
            'SG': CloudRegion.AP_SOUTHEAST_1,
            'KR': CloudRegion.AP_NORTHEAST_2,
        }
        
        return region_mapping.get(data_subject.region, CloudRegion.US_EAST_1)
    
    def _identify_cross_border_transfers(self, data_subject: DataSubject,
                                       processing_regions: List[CloudRegion],
                                       storage_regions: List[CloudRegion]) -> List[Dict[str, Any]]:
        """Identify cross-border data transfers."""
        
        subject_region = self._get_data_subject_region(data_subject)
        subject_config = self.region_manager.get_region_config(subject_region)
        
        transfers = []
        all_regions = set(processing_regions + storage_regions)
        
        for region in all_regions:
            region_config = self.region_manager.get_region_config(region)
            
            # Check if this constitutes a cross-border transfer
            if region_config.country_code != subject_config.country_code:
                transfer = {
                    "from_country": subject_config.country_code,
                    "to_country": region_config.country_code,
                    "from_region": subject_region.value,
                    "to_region": region.value,
                    "transfer_type": "processing" if region in processing_regions else "storage",
                    "adequacy_status": self._check_adequacy_status(subject_region, region),
                    "requires_safeguards": True
                }
                transfers.append(transfer)
        
        return transfers
    
    def _check_adequacy_status(self, from_region: CloudRegion, to_region: CloudRegion) -> str:
        """Check adequacy status for cross-border transfer."""
        from_config = self.region_manager.get_region_config(from_region)
        to_config = self.region_manager.get_region_config(to_region)
        
        # EU adequacy decisions (simplified)
        eu_adequate_countries = ['CA', 'CH', 'IL', 'NZ', 'JP', 'KR', 'GB']
        
        if from_config.gdpr_applicable:
            if to_config.country_code in eu_adequate_countries:
                return "adequate"
            else:
                return "not_adequate"
        
        # Other adequacy frameworks can be added here
        return "not_applicable"
    
    def _determine_transfer_mechanisms(self, data_subject: DataSubject, 
                                     transfers: List[Dict[str, Any]]) -> List[TransferMechanism]:
        """Determine appropriate transfer mechanisms for cross-border transfers."""
        mechanisms = []
        
        for transfer in transfers:
            if transfer["adequacy_status"] == "adequate":
                mechanisms.append(TransferMechanism.ADEQUACY_DECISION)
            else:
                # Default to Standard Contractual Clauses for non-adequate transfers
                mechanisms.append(TransferMechanism.STANDARD_CONTRACTUAL_CLAUSES)
                
                # Add consent if required for certain data types
                if data_subject.consent_preferences.get("cross_border_transfer", False):
                    mechanisms.append(TransferMechanism.CONSENT)
        
        return list(set(mechanisms))  # Remove duplicates
    
    def _validate_processing_compliance(self, record: DataProcessingRecord) -> List[Dict[str, Any]]:
        """Validate compliance of a data processing record."""
        violations = []
        
        # Get data subject and applicable regulations
        data_subject_region = self._data_subject_regions.get(record.data_subject_id)
        if not data_subject_region:
            return violations
        
        applicable_regulations = []
        region_config = self.region_manager.get_region_config(data_subject_region)
        
        if region_config.gdpr_applicable:
            applicable_regulations.append(ComplianceRegulation.GDPR)
        if region_config.ccpa_applicable:
            applicable_regulations.append(ComplianceRegulation.CCPA)
        
        # Validate against each regulation
        for regulation in applicable_regulations:
            requirement = self._residency_requirements.get(regulation)
            if not requirement:
                continue
            
            # Check storage regions compliance
            for storage_region in record.storage_regions:
                if not requirement.allows_processing_in_region(storage_region):
                    violations.append({
                        "type": "data_residency_violation",
                        "regulation": regulation.value,
                        "message": f"Data storage in {storage_region.value} violates {regulation.value} requirements",
                        "record_id": record.record_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Check cross-border transfer compliance
            for transfer in record.cross_border_transfers:
                if transfer["adequacy_status"] == "not_adequate" and not record.transfer_mechanisms:
                    violations.append({
                        "type": "transfer_violation",
                        "regulation": regulation.value,
                        "message": f"Cross-border transfer to {transfer['to_country']} without adequate safeguards",
                        "record_id": record.record_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Check consent requirements
            if regulation == ComplianceRegulation.GDPR and not record.consent_obtained:
                violations.append({
                    "type": "consent_violation",
                    "regulation": regulation.value,
                    "message": "GDPR requires consent for personal data processing",
                    "record_id": record.record_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return violations
    
    def check_data_residency(self, data_subject_id: str, target_regions: List[CloudRegion]) -> Dict[str, Any]:
        """Check if data can be stored/processed in target regions."""
        
        result = {
            "allowed_regions": [],
            "prohibited_regions": [], 
            "required_safeguards": [],
            "transfer_mechanisms_needed": [],
            "compliance_issues": []
        }
        
        # Get data subject's region
        subject_region = self._data_subject_regions.get(data_subject_id)
        if not subject_region:
            result["compliance_issues"].append("Data subject region not found")
            return result
        
        # Get applicable regulations
        region_config = self.region_manager.get_region_config(subject_region)
        applicable_regulations = []
        
        if region_config.gdpr_applicable:
            applicable_regulations.append(ComplianceRegulation.GDPR)
        if region_config.ccpa_applicable:
            applicable_regulations.append(ComplianceRegulation.CCPA)
        
        # Check each target region
        for target_region in target_regions:
            target_config = self.region_manager.get_region_config(target_region)
            region_allowed = True
            region_safeguards = set()
            region_mechanisms = set()
            
            # Check against each regulation
            for regulation in applicable_regulations:
                requirement = self._residency_requirements.get(regulation)
                if not requirement:
                    continue
                
                if not requirement.allows_processing_in_region(target_region):
                    region_allowed = False
                    result["compliance_issues"].append(
                        f"Region {target_region.value} prohibited by {regulation.value}"
                    )
                else:
                    # Add required safeguards
                    region_safeguards.update(requirement.required_safeguards)
                    
                    # Check for cross-border transfer
                    if target_config.country_code != region_config.country_code:
                        adequacy = self._check_adequacy_status(subject_region, target_region)
                        if adequacy == "not_adequate":
                            region_mechanisms.add(TransferMechanism.STANDARD_CONTRACTUAL_CLAUSES.value)
            
            if region_allowed:
                result["allowed_regions"].append(target_region.value)
                result["required_safeguards"].extend(list(region_safeguards))
                result["transfer_mechanisms_needed"].extend(list(region_mechanisms))
            else:
                result["prohibited_regions"].append(target_region.value)
        
        # Remove duplicates
        result["required_safeguards"] = list(set(result["required_safeguards"]))
        result["transfer_mechanisms_needed"] = list(set(result["transfer_mechanisms_needed"]))
        
        return result
    
    def validate_cross_border_transfer(self, from_region: CloudRegion, to_region: CloudRegion,
                                     data_classification: DataClassification = DataClassification.INTERNAL,
                                     transfer_mechanisms: Optional[List[TransferMechanism]] = None) -> Dict[str, Any]:
        """Validate a proposed cross-border data transfer."""
        
        result = {
            "allowed": False,
            "adequacy_status": "unknown",
            "required_mechanisms": [],
            "recommended_safeguards": [],
            "compliance_notes": []
        }
        
        from_config = self.region_manager.get_region_config(from_region)
        to_config = self.region_manager.get_region_config(to_region)
        
        # Same country - generally allowed
        if from_config.country_code == to_config.country_code:
            result["allowed"] = True
            result["adequacy_status"] = "same_country"
            result["compliance_notes"].append("Transfer within same country")
            return result
        
        # Check adequacy status
        adequacy = self._check_adequacy_status(from_region, to_region)
        result["adequacy_status"] = adequacy
        
        # Determine required mechanisms
        if adequacy == "adequate":
            result["allowed"] = True
            result["required_mechanisms"] = [TransferMechanism.ADEQUACY_DECISION.value]
        else:
            # Need additional safeguards
            if transfer_mechanisms:
                provided_mechanisms = [m.value if isinstance(m, TransferMechanism) else m 
                                     for m in transfer_mechanisms]
                
                # Check if provided mechanisms are sufficient
                if TransferMechanism.STANDARD_CONTRACTUAL_CLAUSES.value in provided_mechanisms:
                    result["allowed"] = True
                    result["required_mechanisms"] = [TransferMechanism.STANDARD_CONTRACTUAL_CLAUSES.value]
                elif TransferMechanism.CONSENT.value in provided_mechanisms:
                    result["allowed"] = True
                    result["required_mechanisms"] = [TransferMechanism.CONSENT.value]
                    result["compliance_notes"].append("User consent required for transfer")
            else:
                result["required_mechanisms"] = [TransferMechanism.STANDARD_CONTRACTUAL_CLAUSES.value]
        
        # Add safeguards based on data classification
        if data_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
            result["recommended_safeguards"].extend([
                "encryption_at_rest",
                "encryption_in_transit", 
                "access_logging",
                "data_minimization"
            ])
        
        if data_classification == DataClassification.TOP_SECRET:
            result["allowed"] = False
            result["compliance_notes"].append("Top secret data transfers require special authorization")
        
        return result
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get a summary of compliance status."""
        with self._lock:
            total_records = len(self._processing_records)
            active_violations = len([v for v in self._compliance_violations 
                                   if (datetime.utcnow() - datetime.fromisoformat(v["timestamp"])).days < 30])
            
            # Count records by region
            region_counts = {}
            for record in self._processing_records.values():
                for region in record.storage_regions:
                    region_counts[region.value] = region_counts.get(region.value, 0) + 1
            
            # Count cross-border transfers
            cross_border_count = sum(
                1 for record in self._processing_records.values() 
                if record.cross_border_transfers
            )
            
            return {
                "total_processing_records": total_records,
                "active_compliance_violations": active_violations,
                "cross_border_transfers": cross_border_count,
                "data_storage_by_region": region_counts,
                "monitoring_status": "active" if self._monitoring_active else "inactive",
                "last_updated": datetime.utcnow().isoformat()
            }
    
    def generate_compliance_report(self, regulation: ComplianceRegulation, 
                                 period_days: int = 30) -> Dict[str, Any]:
        """Generate a compliance report for a specific regulation."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        
        # Filter records and violations for the period
        relevant_records = [
            record for record in self._processing_records.values()
            if record.created_at >= cutoff_date
        ]
        
        relevant_violations = [
            violation for violation in self._compliance_violations
            if (datetime.fromisoformat(violation["timestamp"]) >= cutoff_date and 
                violation["regulation"] == regulation.value)
        ]
        
        # Calculate compliance metrics
        total_records = len(relevant_records)
        violation_count = len(relevant_violations)
        compliance_rate = ((total_records - violation_count) / total_records * 100) if total_records > 0 else 100
        
        report = {
            "regulation": regulation.value,
            "report_period": {
                "start_date": cutoff_date.isoformat(),
                "end_date": datetime.utcnow().isoformat(),
                "period_days": period_days
            },
            "metrics": {
                "total_records": total_records,
                "violation_count": violation_count,
                "compliance_rate_percent": compliance_rate
            },
            "violations_by_type": {},
            "recommendations": [],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Group violations by type
        for violation in relevant_violations:
            violation_type = violation["type"]
            if violation_type not in report["violations_by_type"]:
                report["violations_by_type"][violation_type] = []
            report["violations_by_type"][violation_type].append(violation)
        
        # Generate recommendations
        if violation_count > 0:
            report["recommendations"].extend([
                "Review and update data processing agreements",
                "Implement additional technical safeguards",
                "Conduct compliance training for staff",
                "Review data retention policies"
            ])
        
        return report


# Global instance
_compliance_manager = None
_compliance_lock = threading.Lock()


def get_compliance_manager() -> GlobalComplianceManager:
    """Get the global compliance manager instance.""" 
    global _compliance_manager
    
    with _compliance_lock:
        if _compliance_manager is None:
            _compliance_manager = GlobalComplianceManager()
        
        return _compliance_manager


def check_data_residency(data_subject_id: str, target_regions: List[CloudRegion]) -> Dict[str, Any]:
    """Check data residency compliance for target regions."""
    manager = get_compliance_manager()
    return manager.check_data_residency(data_subject_id, target_regions)


def validate_cross_border_transfer(from_region: CloudRegion, to_region: CloudRegion,
                                  data_classification: DataClassification = DataClassification.INTERNAL) -> Dict[str, Any]:
    """Validate cross-border transfer compliance."""
    manager = get_compliance_manager()
    return manager.validate_cross_border_transfer(from_region, to_region, data_classification)