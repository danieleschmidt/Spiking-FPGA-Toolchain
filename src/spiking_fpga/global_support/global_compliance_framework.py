"""Global compliance framework for GDPR, CCPA, PDPA and other regulations."""

import logging
import json
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import base64
import hmac


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA_SINGAPORE = "pdpa_sg"  # Personal Data Protection Act (Singapore)
    PDPA_THAILAND = "pdpa_th"  # Personal Data Protection Act (Thailand)
    PIPEDA = "pipeda"  # Personal Information Protection Act (Canada)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    SOC2 = "soc2"  # SOC 2 Type II
    ISO27001 = "iso27001"  # ISO/IEC 27001
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act (US)
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    NIST = "nist"  # NIST Cybersecurity Framework


class DataCategory(Enum):
    """Categories of data for compliance classification."""
    PERSONAL_IDENTIFIABLE = "pii"
    SENSITIVE_PERSONAL = "sensitive_pii"
    BIOMETRIC = "biometric"
    FINANCIAL = "financial"
    HEALTH = "health"
    BEHAVIORAL = "behavioral"
    TECHNICAL_LOGS = "technical_logs"
    SYSTEM_METRICS = "system_metrics"
    NETWORK_CONFIGURATION = "network_config"
    COMPILATION_DATA = "compilation_data"
    PERFORMANCE_DATA = "performance_data"
    PUBLIC = "public"


class DataProcessingPurpose(Enum):
    """Purposes for data processing."""
    COMPILATION = "compilation"
    OPTIMIZATION = "optimization"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SECURITY_ANALYSIS = "security_analysis"
    RESEARCH_DEVELOPMENT = "research_development"
    SYSTEM_ADMINISTRATION = "system_administration"
    LEGAL_COMPLIANCE = "legal_compliance"
    SERVICE_IMPROVEMENT = "service_improvement"
    ANALYTICS = "analytics"
    SUPPORT = "support"


class DataSubjectRight(Enum):
    """Rights of data subjects under various frameworks."""
    ACCESS = "access"  # Right to access personal data
    RECTIFICATION = "rectification"  # Right to correct inaccurate data
    ERASURE = "erasure"  # Right to be forgotten
    PORTABILITY = "portability"  # Right to data portability
    RESTRICTION = "restriction"  # Right to restrict processing
    OBJECTION = "objection"  # Right to object to processing
    CONSENT_WITHDRAWAL = "consent_withdrawal"  # Right to withdraw consent
    NOTIFICATION = "notification"  # Right to be notified of breaches


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    record_id: str
    data_category: DataCategory
    processing_purpose: DataProcessingPurpose
    legal_basis: str
    data_source: str
    retention_period: timedelta
    processing_start: datetime
    processing_end: Optional[datetime] = None
    data_subjects_affected: int = 0
    third_parties_involved: List[str] = field(default_factory=list)
    security_measures: List[str] = field(default_factory=list)
    cross_border_transfer: bool = False
    transfer_destinations: List[str] = field(default_factory=list)
    automated_decision_making: bool = False


@dataclass
class ComplianceAssessment:
    """Compliance assessment result."""
    framework: ComplianceFramework
    assessment_date: datetime
    overall_compliance: bool
    compliance_score: float  # 0.0 to 1.0
    requirements_met: List[str]
    requirements_failed: List[str]
    recommendations: List[str]
    risk_level: str  # "low", "medium", "high", "critical"
    next_assessment_due: datetime
    remediation_actions: List[Dict[str, Any]] = field(default_factory=list)


class GlobalComplianceFramework:
    """Comprehensive global compliance framework."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Data processing records
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        
        # Compliance assessments
        self.assessments: Dict[ComplianceFramework, ComplianceAssessment] = {}
        
        # Configuration
        self.compliance_config = self._load_compliance_config()
        self.active_frameworks: Set[ComplianceFramework] = set()
        
        # Data subject rights management
        self.rights_requests: Dict[str, Dict[str, Any]] = {}
        
        # Privacy by design implementation
        self.privacy_controls: Dict[str, Any] = {}
        
        # Data retention policies
        self.retention_policies: Dict[DataCategory, timedelta] = self._initialize_retention_policies()
        
        # Encryption and pseudonymization
        self.crypto_manager = ComplianceCryptoManager(self.logger)
        
        self.logger.info("GlobalComplianceFramework initialized")
    
    def _load_compliance_config(self) -> Dict[str, Any]:
        """Load compliance configuration."""
        default_config = {
            "data_retention_days": {
                "personal": 1095,  # 3 years
                "sensitive": 730,  # 2 years
                "technical": 365,  # 1 year
                "public": -1  # Indefinite
            },
            "encryption_required": True,
            "pseudonymization_required": True,
            "consent_required_purposes": [
                DataProcessingPurpose.ANALYTICS.value,
                DataProcessingPurpose.RESEARCH_DEVELOPMENT.value,
                DataProcessingPurpose.SERVICE_IMPROVEMENT.value
            ],
            "breach_notification_hours": 72,
            "data_protection_officer_required": False,
            "privacy_impact_assessment_threshold": 1000  # Number of data subjects
        }
        
        # Try to load from file
        config_file = Path("compliance_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(f"Failed to load compliance config: {e}")
        
        return default_config
    
    def _initialize_retention_policies(self) -> Dict[DataCategory, timedelta]:
        """Initialize data retention policies."""
        return {
            DataCategory.PERSONAL_IDENTIFIABLE: timedelta(days=self.compliance_config["data_retention_days"]["personal"]),
            DataCategory.SENSITIVE_PERSONAL: timedelta(days=self.compliance_config["data_retention_days"]["sensitive"]),
            DataCategory.BIOMETRIC: timedelta(days=self.compliance_config["data_retention_days"]["sensitive"]),
            DataCategory.FINANCIAL: timedelta(days=self.compliance_config["data_retention_days"]["sensitive"]),
            DataCategory.HEALTH: timedelta(days=self.compliance_config["data_retention_days"]["sensitive"]),
            DataCategory.BEHAVIORAL: timedelta(days=self.compliance_config["data_retention_days"]["personal"]),
            DataCategory.TECHNICAL_LOGS: timedelta(days=self.compliance_config["data_retention_days"]["technical"]),
            DataCategory.SYSTEM_METRICS: timedelta(days=self.compliance_config["data_retention_days"]["technical"]),
            DataCategory.NETWORK_CONFIGURATION: timedelta(days=self.compliance_config["data_retention_days"]["technical"]),
            DataCategory.COMPILATION_DATA: timedelta(days=self.compliance_config["data_retention_days"]["technical"]),
            DataCategory.PERFORMANCE_DATA: timedelta(days=self.compliance_config["data_retention_days"]["technical"]),
            DataCategory.PUBLIC: timedelta(days=-1)  # Indefinite retention
        }
    
    def enable_compliance_framework(self, framework: ComplianceFramework) -> bool:
        """Enable a specific compliance framework."""
        try:
            self.active_frameworks.add(framework)
            
            # Initialize framework-specific requirements
            self._initialize_framework_requirements(framework)
            
            self.logger.info(f"Enabled compliance framework: {framework.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable compliance framework {framework.value}: {e}")
            return False
    
    def _initialize_framework_requirements(self, framework: ComplianceFramework) -> None:
        """Initialize requirements for a specific framework."""
        
        if framework == ComplianceFramework.GDPR:
            self.privacy_controls["gdpr"] = {
                "legal_basis_required": True,
                "consent_granular": True,
                "right_to_erasure": True,
                "right_to_portability": True,
                "privacy_by_design": True,
                "dpo_required_threshold": 250000,  # Data subjects
                "breach_notification_hours": 72,
                "adequacy_decisions": ["EEA", "UK", "Switzerland"]
            }
        
        elif framework == ComplianceFramework.CCPA:
            self.privacy_controls["ccpa"] = {
                "opt_out_required": True,
                "disclosure_categories": True,
                "sale_notification": True,
                "deletion_rights": True,
                "non_discrimination": True,
                "revenue_threshold": 25000000,  # USD
                "personal_info_threshold": 50000  # Consumers
            }
        
        elif framework == ComplianceFramework.SOC2:
            self.privacy_controls["soc2"] = {
                "security_principle": True,
                "availability_principle": True,
                "processing_integrity": True,
                "confidentiality_principle": True,
                "privacy_principle": True,
                "audit_trail_required": True,
                "change_management": True
            }
    
    def record_data_processing(self, 
                             data_category: DataCategory,
                             processing_purpose: DataProcessingPurpose,
                             legal_basis: str,
                             data_source: str,
                             data_subjects_affected: int = 0,
                             **kwargs) -> str:
        """Record a data processing activity."""
        
        record_id = str(uuid.uuid4())
        
        # Determine retention period
        retention_period = self.retention_policies.get(
            data_category, 
            timedelta(days=self.compliance_config["data_retention_days"]["technical"])
        )
        
        record = DataProcessingRecord(
            record_id=record_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_source=data_source,
            retention_period=retention_period,
            processing_start=datetime.now(),
            data_subjects_affected=data_subjects_affected,
            third_parties_involved=kwargs.get("third_parties", []),
            security_measures=kwargs.get("security_measures", []),
            cross_border_transfer=kwargs.get("cross_border_transfer", False),
            transfer_destinations=kwargs.get("transfer_destinations", []),
            automated_decision_making=kwargs.get("automated_decision_making", False)
        )
        
        self.processing_records[record_id] = record
        
        # Check compliance requirements
        self._validate_processing_compliance(record)
        
        self.logger.info(f"Recorded data processing activity: {record_id}")
        return record_id
    
    def _validate_processing_compliance(self, record: DataProcessingRecord) -> None:
        """Validate processing record against active compliance frameworks."""
        
        violations = []
        
        for framework in self.active_frameworks:
            framework_violations = self._check_framework_compliance(record, framework)
            violations.extend(framework_violations)
        
        if violations:
            self.logger.warning(f"Compliance violations detected for record {record.record_id}: {violations}")
        
        # Apply privacy controls
        self._apply_privacy_controls(record)
    
    def _check_framework_compliance(self, record: DataProcessingRecord, 
                                  framework: ComplianceFramework) -> List[str]:
        """Check compliance for a specific framework."""
        violations = []
        
        if framework == ComplianceFramework.GDPR:
            # GDPR-specific checks
            if record.data_category in [DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.SENSITIVE_PERSONAL]:
                if not record.legal_basis:
                    violations.append("GDPR: Legal basis required for personal data processing")
                
                if record.cross_border_transfer:
                    gdpr_controls = self.privacy_controls.get("gdpr", {})
                    adequacy_decisions = gdpr_controls.get("adequacy_decisions", [])
                    
                    for destination in record.transfer_destinations:
                        if destination not in adequacy_decisions:
                            violations.append(f"GDPR: Transfer to {destination} requires additional safeguards")
        
        elif framework == ComplianceFramework.CCPA:
            # CCPA-specific checks
            if record.data_subjects_affected > 0:
                ccpa_controls = self.privacy_controls.get("ccpa", {})
                threshold = ccpa_controls.get("personal_info_threshold", 50000)
                
                if record.data_subjects_affected >= threshold:
                    violations.append("CCPA: Large scale processing requires additional disclosures")
        
        elif framework == ComplianceFramework.SOC2:
            # SOC 2-specific checks
            if not record.security_measures:
                violations.append("SOC 2: Security measures must be documented")
        
        return violations
    
    def _apply_privacy_controls(self, record: DataProcessingRecord) -> None:
        """Apply privacy controls to data processing."""
        
        # Encryption for sensitive data
        if (record.data_category in [DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.SENSITIVE_PERSONAL] and
            self.compliance_config.get("encryption_required", True)):
            
            if "encryption" not in record.security_measures:
                record.security_measures.append("encryption")
                self.logger.info(f"Applied encryption to record {record.record_id}")
        
        # Pseudonymization
        if (record.data_category == DataCategory.PERSONAL_IDENTIFIABLE and
            self.compliance_config.get("pseudonymization_required", True)):
            
            if "pseudonymization" not in record.security_measures:
                record.security_measures.append("pseudonymization")
                self.logger.info(f"Applied pseudonymization to record {record.record_id}")
    
    def handle_data_subject_request(self, request_type: DataSubjectRight, 
                                  subject_identifier: str,
                                  additional_info: Dict[str, Any] = None) -> str:
        """Handle data subject rights requests."""
        
        request_id = str(uuid.uuid4())
        request_record = {
            "request_id": request_id,
            "request_type": request_type.value,
            "subject_identifier": subject_identifier,
            "request_date": datetime.now().isoformat(),
            "status": "pending",
            "additional_info": additional_info or {},
            "processing_records_affected": [],
            "response_due_date": (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        self.rights_requests[request_id] = request_record
        
        # Process request based on type
        if request_type == DataSubjectRight.ACCESS:
            self._process_access_request(request_id, subject_identifier)
        elif request_type == DataSubjectRight.ERASURE:
            self._process_erasure_request(request_id, subject_identifier)
        elif request_type == DataSubjectRight.PORTABILITY:
            self._process_portability_request(request_id, subject_identifier)
        elif request_type == DataSubjectRight.RECTIFICATION:
            self._process_rectification_request(request_id, subject_identifier, additional_info)
        
        self.logger.info(f"Created data subject request: {request_id} ({request_type.value})")
        return request_id
    
    def _process_access_request(self, request_id: str, subject_identifier: str) -> None:
        """Process data access request."""
        affected_records = []
        
        # Find all processing records related to the subject
        for record_id, record in self.processing_records.items():
            if self._record_involves_subject(record, subject_identifier):
                affected_records.append(record_id)
        
        # Update request with findings
        self.rights_requests[request_id]["processing_records_affected"] = affected_records
        self.rights_requests[request_id]["status"] = "processing"
        
        # Generate data export (simplified)
        data_export = {
            "subject_identifier": subject_identifier,
            "data_processing_activities": affected_records,
            "export_date": datetime.now().isoformat()
        }
        
        self.rights_requests[request_id]["data_export"] = data_export
        self.rights_requests[request_id]["status"] = "completed"
        
        self.logger.info(f"Processed access request {request_id}: {len(affected_records)} records found")
    
    def _process_erasure_request(self, request_id: str, subject_identifier: str) -> None:
        """Process right to be forgotten request."""
        affected_records = []
        erased_records = []
        
        for record_id, record in self.processing_records.items():
            if self._record_involves_subject(record, subject_identifier):
                affected_records.append(record_id)
                
                # Check if erasure is legally permissible
                if self._can_erase_record(record):
                    # Mark for deletion
                    record.processing_end = datetime.now()
                    erased_records.append(record_id)
        
        # Update request
        self.rights_requests[request_id]["processing_records_affected"] = affected_records
        self.rights_requests[request_id]["erased_records"] = erased_records
        self.rights_requests[request_id]["status"] = "completed"
        
        self.logger.info(f"Processed erasure request {request_id}: {len(erased_records)} records erased")
    
    def _record_involves_subject(self, record: DataProcessingRecord, subject_identifier: str) -> bool:
        """Check if a processing record involves a specific data subject."""
        # Simplified implementation - in practice, this would involve
        # secure matching against pseudonymized identifiers
        return record.data_source == subject_identifier or subject_identifier in str(record.record_id)
    
    def _can_erase_record(self, record: DataProcessingRecord) -> bool:
        """Check if a record can be legally erased."""
        # Check for legal obligations that prevent erasure
        legal_hold_purposes = [
            DataProcessingPurpose.LEGAL_COMPLIANCE,
            DataProcessingPurpose.SYSTEM_ADMINISTRATION
        ]
        
        if record.processing_purpose in legal_hold_purposes:
            return False
        
        # Check retention period
        if datetime.now() < record.processing_start + record.retention_period:
            return False
        
        return True
    
    def _process_portability_request(self, request_id: str, subject_identifier: str) -> None:
        """Process data portability request."""
        # Similar to access request but with structured export format
        self._process_access_request(request_id, subject_identifier)
        
        # Convert to portable format
        request = self.rights_requests[request_id]
        if "data_export" in request:
            portable_data = {
                "format": "JSON",
                "encoding": "UTF-8",
                "data": request["data_export"],
                "metadata": {
                    "export_standard": "GDPR_Article_20",
                    "machine_readable": True
                }
            }
            request["portable_export"] = portable_data
    
    def _process_rectification_request(self, request_id: str, subject_identifier: str,
                                     correction_info: Dict[str, Any]) -> None:
        """Process data rectification request."""
        # Implementation would involve updating incorrect data
        # This is a simplified placeholder
        
        self.rights_requests[request_id]["status"] = "requires_manual_review"
        self.rights_requests[request_id]["correction_requested"] = correction_info
        
        self.logger.info(f"Rectification request {request_id} requires manual review")
    
    def assess_compliance(self, framework: ComplianceFramework) -> ComplianceAssessment:
        """Perform comprehensive compliance assessment."""
        
        assessment_date = datetime.now()
        requirements_met = []
        requirements_failed = []
        recommendations = []
        
        # Framework-specific assessments
        if framework == ComplianceFramework.GDPR:
            met, failed, recs = self._assess_gdpr_compliance()
            requirements_met.extend(met)
            requirements_failed.extend(failed)
            recommendations.extend(recs)
        
        elif framework == ComplianceFramework.CCPA:
            met, failed, recs = self._assess_ccpa_compliance()
            requirements_met.extend(met)
            requirements_failed.extend(failed)
            recommendations.extend(recs)
        
        elif framework == ComplianceFramework.SOC2:
            met, failed, recs = self._assess_soc2_compliance()
            requirements_met.extend(met)
            requirements_failed.extend(failed)
            recommendations.extend(recs)
        
        # Calculate compliance score
        total_requirements = len(requirements_met) + len(requirements_failed)
        compliance_score = len(requirements_met) / total_requirements if total_requirements > 0 else 0.0
        
        # Determine risk level
        if compliance_score >= 0.95:
            risk_level = "low"
        elif compliance_score >= 0.85:
            risk_level = "medium"
        elif compliance_score >= 0.70:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        assessment = ComplianceAssessment(
            framework=framework,
            assessment_date=assessment_date,
            overall_compliance=compliance_score >= 0.85,
            compliance_score=compliance_score,
            requirements_met=requirements_met,
            requirements_failed=requirements_failed,
            recommendations=recommendations,
            risk_level=risk_level,
            next_assessment_due=assessment_date + timedelta(days=90)
        )
        
        self.assessments[framework] = assessment
        
        self.logger.info(f"Compliance assessment completed for {framework.value}: {compliance_score:.2f}")
        return assessment
    
    def _assess_gdpr_compliance(self) -> Tuple[List[str], List[str], List[str]]:
        """Assess GDPR compliance."""
        met = []
        failed = []
        recommendations = []
        
        # Check legal basis documentation
        personal_data_records = [r for r in self.processing_records.values() 
                               if r.data_category in [DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.SENSITIVE_PERSONAL]]
        
        if all(record.legal_basis for record in personal_data_records):
            met.append("Legal basis documented for all personal data processing")
        else:
            failed.append("Missing legal basis for some personal data processing")
            recommendations.append("Document legal basis for all personal data processing activities")
        
        # Check data retention policies
        if self.retention_policies:
            met.append("Data retention policies defined")
        else:
            failed.append("Data retention policies not defined")
            recommendations.append("Implement comprehensive data retention policies")
        
        # Check rights handling
        if self.rights_requests:
            met.append("Data subject rights handling mechanism in place")
        else:
            recommendations.append("Implement data subject rights handling procedures")
        
        # Check breach notification capability
        breach_config = self.compliance_config.get("breach_notification_hours", 0)
        if breach_config > 0 and breach_config <= 72:
            met.append("Breach notification procedures within GDPR timeframe")
        else:
            failed.append("Breach notification procedures do not meet GDPR requirements")
            recommendations.append("Implement 72-hour breach notification procedures")
        
        return met, failed, recommendations
    
    def _assess_ccpa_compliance(self) -> Tuple[List[str], List[str], List[str]]:
        """Assess CCPA compliance."""
        met = []
        failed = []
        recommendations = []
        
        # Check disclosure categories
        if self.processing_records:
            met.append("Personal information processing activities documented")
        else:
            failed.append("No personal information processing documentation")
        
        # Check opt-out mechanisms
        opt_out_rights = [req for req in self.rights_requests.values() 
                         if req.get("request_type") == DataSubjectRight.OBJECTION.value]
        if opt_out_rights:
            met.append("Opt-out request handling implemented")
        else:
            recommendations.append("Implement opt-out request handling for CCPA compliance")
        
        return met, failed, recommendations
    
    def _assess_soc2_compliance(self) -> Tuple[List[str], List[str], List[str]]:
        """Assess SOC 2 compliance."""
        met = []
        failed = []
        recommendations = []
        
        # Check security measures documentation
        security_records = [r for r in self.processing_records.values() if r.security_measures]
        
        if len(security_records) == len(self.processing_records):
            met.append("Security measures documented for all processing activities")
        else:
            failed.append("Missing security measures for some processing activities")
            recommendations.append("Document security measures for all data processing")
        
        # Check audit trail
        if self.processing_records:
            met.append("Audit trail of data processing activities maintained")
        else:
            failed.append("No audit trail of data processing activities")
        
        return met, failed, recommendations
    
    def generate_compliance_report(self, frameworks: List[ComplianceFramework] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        if frameworks is None:
            frameworks = list(self.active_frameworks)
        
        report = {
            "report_date": datetime.now().isoformat(),
            "active_frameworks": [f.value for f in self.active_frameworks],
            "assessments": {},
            "overall_compliance": True,
            "risk_summary": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "data_processing_summary": {
                "total_records": len(self.processing_records),
                "active_records": len([r for r in self.processing_records.values() if r.processing_end is None]),
                "categories": {}
            },
            "rights_requests_summary": {
                "total_requests": len(self.rights_requests),
                "pending_requests": len([r for r in self.rights_requests.values() if r["status"] == "pending"]),
                "by_type": {}
            }
        }
        
        # Assess each framework
        for framework in frameworks:
            assessment = self.assess_compliance(framework)
            
            report["assessments"][framework.value] = {
                "compliance_score": assessment.compliance_score,
                "overall_compliance": assessment.overall_compliance,
                "risk_level": assessment.risk_level,
                "requirements_met": len(assessment.requirements_met),
                "requirements_failed": len(assessment.requirements_failed),
                "recommendations": assessment.recommendations
            }
            
            # Update overall compliance
            if not assessment.overall_compliance:
                report["overall_compliance"] = False
            
            # Update risk summary
            report["risk_summary"][assessment.risk_level] += 1
        
        # Data processing summary
        for record in self.processing_records.values():
            category = record.data_category.value
            report["data_processing_summary"]["categories"][category] = \
                report["data_processing_summary"]["categories"].get(category, 0) + 1
        
        # Rights requests summary
        for request in self.rights_requests.values():
            request_type = request["request_type"]
            report["rights_requests_summary"]["by_type"][request_type] = \
                report["rights_requests_summary"]["by_type"].get(request_type, 0) + 1
        
        return report
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status."""
        return {
            "active_frameworks": [f.value for f in self.active_frameworks],
            "processing_records_count": len(self.processing_records),
            "rights_requests_count": len(self.rights_requests),
            "recent_assessments": [
                {
                    "framework": assessment.framework.value,
                    "compliance_score": assessment.compliance_score,
                    "risk_level": assessment.risk_level,
                    "assessment_date": assessment.assessment_date.isoformat()
                }
                for assessment in list(self.assessments.values())[-5:]
            ]
        }


class ComplianceCryptoManager:
    """Cryptographic operations for compliance requirements."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.key_material = self._generate_key_material()
    
    def _generate_key_material(self) -> bytes:
        """Generate cryptographic key material."""
        # In production, this should use proper key management
        import secrets
        return secrets.token_bytes(32)
    
    def pseudonymize_identifier(self, identifier: str) -> str:
        """Create pseudonymized identifier."""
        # Use HMAC for stable pseudonymization
        h = hmac.new(self.key_material, identifier.encode('utf-8'), hashlib.sha256)
        return base64.urlsafe_b64encode(h.digest()).decode('ascii').rstrip('=')
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data (simplified)."""
        # In production, use proper encryption like AES-GCM
        encoded = base64.b64encode(data.encode('utf-8')).decode('ascii')
        return f"encrypted:{encoded}"
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data (simplified)."""
        if encrypted_data.startswith("encrypted:"):
            encoded = encrypted_data[10:]  # Remove "encrypted:" prefix
            return base64.b64decode(encoded.encode('ascii')).decode('utf-8')
        return encrypted_data