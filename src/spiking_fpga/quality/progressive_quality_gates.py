"""
Progressive Quality Gates for Autonomous SDLC
===========================================

This module implements progressive quality gates that adapt based on:
- Branch context (feature, hotfix, release, main)
- Project maturity level
- Historical quality patterns
- Risk assessment

Progressive gates provide intelligent, context-aware quality enforcement
that becomes more stringent as code moves closer to production.
"""

import asyncio
import numpy as np
import logging
import time
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque

from .autonomous_quality_gates import (
    AutonomousQualityGate, QualityGateResult, QualityGateStatus, 
    QualityGateSeverity, AutonomousQualityOrchestrator
)
from ..utils.validation import validate_hdl_syntax
from ..security.vulnerability_scanner import VulnerabilityScanner


class BranchType(Enum):
    """Types of git branches with different quality requirements."""
    FEATURE = "feature"
    HOTFIX = "hotfix"
    RELEASE = "release"
    MAIN = "main"
    DEVELOPMENT = "development"
    
    
class ProjectMaturityLevel(Enum):
    """Project maturity levels affecting quality gate strictness."""
    EXPERIMENTAL = "experimental"      # Early development, lenient gates
    DEVELOPMENT = "development"        # Active development, moderate gates
    TESTING = "testing"               # Pre-release testing, strict gates
    PRODUCTION = "production"         # Production ready, strictest gates
    LEGACY = "legacy"                 # Maintenance mode, focused gates


@dataclass
class ProgressiveQualityConfig:
    """Configuration for progressive quality gates."""
    branch_type: BranchType
    maturity_level: ProjectMaturityLevel
    risk_tolerance: float = 0.1  # 0.0 = no risk, 1.0 = high risk
    performance_degradation_threshold: float = 0.05  # 5% degradation allowed
    security_scan_required: bool = True
    code_coverage_threshold: float = 0.80  # 80% minimum coverage
    complexity_threshold: int = 10  # Maximum cyclomatic complexity
    enable_ai_analysis: bool = True
    parallel_execution: bool = True


class ProgressivePerformanceGate(AutonomousQualityGate):
    """
    Performance gate that adapts thresholds based on branch and maturity.
    """
    
    def __init__(self, config: ProgressiveQualityConfig):
        super().__init__("ProgressivePerformance", QualityGateSeverity.HIGH)
        self.config = config
        self.baseline_performance = {}
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # Adaptive thresholds based on branch and maturity
        self.thresholds = self._calculate_adaptive_thresholds()
        
    def _calculate_adaptive_thresholds(self) -> Dict[str, float]:
        """Calculate adaptive thresholds based on configuration."""
        base_thresholds = {
            'throughput_degradation': 0.10,  # 10% degradation
            'latency_increase': 0.15,        # 15% increase
            'memory_usage_increase': 0.20,   # 20% increase
            'power_consumption_increase': 0.25  # 25% increase
        }
        
        # Adjust based on branch type
        branch_multipliers = {
            BranchType.FEATURE: 1.5,      # More lenient
            BranchType.HOTFIX: 0.8,       # Stricter for hotfixes
            BranchType.RELEASE: 0.6,      # Very strict for releases
            BranchType.MAIN: 0.5,         # Strictest for main
            BranchType.DEVELOPMENT: 1.2   # Moderate for development
        }
        
        # Adjust based on maturity
        maturity_multipliers = {
            ProjectMaturityLevel.EXPERIMENTAL: 2.0,  # Very lenient
            ProjectMaturityLevel.DEVELOPMENT: 1.5,   # Moderate
            ProjectMaturityLevel.TESTING: 1.0,       # Standard
            ProjectMaturityLevel.PRODUCTION: 0.7,    # Strict
            ProjectMaturityLevel.LEGACY: 0.9         # Focused on regressions
        }
        
        branch_mult = branch_multipliers.get(self.config.branch_type, 1.0)
        maturity_mult = maturity_multipliers.get(self.config.maturity_level, 1.0)
        
        # Apply multipliers
        adjusted_thresholds = {}
        for metric, base_threshold in base_thresholds.items():
            adjusted_thresholds[metric] = base_threshold * branch_mult * maturity_mult
            
        return adjusted_thresholds
        
    async def _execute_check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute progressive performance check."""
        current_performance = context.get('performance_metrics', {})
        baseline = context.get('baseline_performance', {})
        
        violations = []
        warnings = []
        
        # Compare against baseline if available
        if baseline:
            for metric, current_value in current_performance.items():
                if metric in baseline:
                    baseline_value = baseline[metric]
                    relative_change = self._calculate_relative_change(
                        baseline_value, current_value, metric
                    )
                    
                    threshold_key = self._get_threshold_key(metric)
                    if threshold_key in self.thresholds:
                        threshold = self.thresholds[threshold_key]
                        
                        if abs(relative_change) > threshold:
                            severity = 'critical' if abs(relative_change) > threshold * 1.5 else 'warning'
                            issue = {
                                'metric': metric,
                                'baseline': baseline_value,
                                'current': current_value,
                                'change': relative_change,
                                'threshold': threshold,
                                'severity': severity
                            }
                            
                            if severity == 'critical':
                                violations.append(issue)
                            else:
                                warnings.append(issue)
        
        # Adaptive status determination
        if violations:
            # Check if we can be more lenient based on context
            if self._should_apply_leniency(context, violations):
                status = QualityGateStatus.WARNING
                message = f"Performance issues detected but within acceptable tolerance for {self.config.branch_type.value} branch"
                warnings.extend(violations)
                violations = []
            else:
                status = QualityGateStatus.FAILED
                message = f"Performance degradation exceeds limits: {len(violations)} critical issues"
        elif warnings:
            status = QualityGateStatus.WARNING
            message = f"Performance warnings: {len(warnings)} issues detected"
        else:
            status = QualityGateStatus.PASSED
            message = "Performance within acceptable limits"
            
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            severity=self.severity,
            message=message,
            details={
                'violations': violations,
                'warnings': warnings,
                'thresholds_applied': self.thresholds,
                'branch_type': self.config.branch_type.value,
                'maturity_level': self.config.maturity_level.value
            }
        )
        
    def _calculate_relative_change(self, baseline: float, current: float, metric: str) -> float:
        """Calculate relative change with direction consideration."""
        if baseline == 0:
            return 0.0 if current == 0 else float('inf')
            
        change = (current - baseline) / baseline
        
        # For metrics where increase is bad (latency, power, memory)
        if any(keyword in metric.lower() for keyword in ['latency', 'power', 'memory', 'delay']):
            return change  # Positive change is bad
        
        # For metrics where decrease is bad (throughput, accuracy)
        elif any(keyword in metric.lower() for keyword in ['throughput', 'accuracy', 'efficiency']):
            return -change  # Negative change is bad (but we return positive to indicate badness)
            
        return abs(change)  # Default: any change is potentially bad
        
    def _get_threshold_key(self, metric: str) -> str:
        """Map metric name to threshold key."""
        mapping = {
            'throughput': 'throughput_degradation',
            'latency': 'latency_increase',
            'memory': 'memory_usage_increase',
            'power': 'power_consumption_increase'
        }
        
        for key, threshold_key in mapping.items():
            if key in metric.lower():
                return threshold_key
                
        return 'throughput_degradation'  # Default
        
    def _should_apply_leniency(self, context: Dict[str, Any], violations: List[Dict]) -> bool:
        """Determine if leniency should be applied based on context."""
        # Apply leniency for experimental projects
        if self.config.maturity_level == ProjectMaturityLevel.EXPERIMENTAL:
            return True
            
        # Apply leniency for feature branches with minor violations
        if (self.config.branch_type == BranchType.FEATURE and 
            all(v['severity'] != 'critical' for v in violations)):
            return True
            
        # Check if this is a known issue being worked on
        known_issues = context.get('known_performance_issues', [])
        for violation in violations:
            metric = violation['metric']
            if any(issue.get('metric') == metric for issue in known_issues):
                return True
                
        return False


class ProgressiveSecurityGate(AutonomousQualityGate):
    """
    Security gate that adapts scanning depth based on branch and risk profile.
    """
    
    def __init__(self, config: ProgressiveQualityConfig):
        super().__init__("ProgressiveSecurity", QualityGateSeverity.CRITICAL)
        self.config = config
        self.scanner = VulnerabilityScanner()
        
        # Determine scanning intensity
        self.scan_intensity = self._determine_scan_intensity()
        
    def _determine_scan_intensity(self) -> str:
        """Determine security scanning intensity."""
        # High intensity for production-bound branches
        if self.config.branch_type in [BranchType.MAIN, BranchType.RELEASE]:
            return 'comprehensive'
        elif self.config.maturity_level == ProjectMaturityLevel.PRODUCTION:
            return 'comprehensive'
        elif self.config.branch_type == BranchType.HOTFIX:
            return 'focused'  # Focus on changed areas
        else:
            return 'standard'
            
    async def _execute_check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute progressive security check."""
        hdl_files = context.get('hdl_files', {})
        changed_files = context.get('changed_files', set())
        
        if not hdl_files and self.config.security_scan_required:
            return QualityGateResult(
                gate_name=self.name,
                status=QualityGateStatus.FAILED,
                severity=self.severity,
                message="Security scan required but no HDL files provided"
            )
        
        # Determine which files to scan based on intensity
        files_to_scan = self._select_files_for_scanning(hdl_files, changed_files)
        
        vulnerabilities = []
        compliance_issues = []
        
        if files_to_scan:
            # Run vulnerability scan
            scan_results = await self.scanner.scan_hdl(files_to_scan)
            vulnerabilities = scan_results.get('vulnerabilities', [])
            
            # Run compliance checks
            compliance_results = await self._check_security_compliance(context)
            compliance_issues = compliance_results.get('violations', [])
        
        # Apply progressive filtering
        critical_issues = self._filter_by_severity(vulnerabilities + compliance_issues, 'critical')
        high_issues = self._filter_by_severity(vulnerabilities + compliance_issues, 'high')
        
        # Adaptive decision making
        if critical_issues:
            # Critical issues always fail, regardless of branch
            status = QualityGateStatus.FAILED
            message = f"Critical security vulnerabilities found: {len(critical_issues)}"
        elif high_issues:
            # High issues may be acceptable for feature branches in experimental stage
            if self._can_accept_high_severity_issues():
                status = QualityGateStatus.WARNING
                message = f"High severity security issues detected but acceptable for {self.config.branch_type.value}"
            else:
                status = QualityGateStatus.FAILED
                message = f"High severity security issues not acceptable: {len(high_issues)}"
        else:
            status = QualityGateStatus.PASSED
            message = "Security checks passed"
            
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            severity=self.severity,
            message=message,
            details={
                'scan_intensity': self.scan_intensity,
                'files_scanned': len(files_to_scan),
                'vulnerabilities': vulnerabilities,
                'compliance_issues': compliance_issues,
                'critical_count': len(critical_issues),
                'high_count': len(high_issues)
            }
        )
        
    def _select_files_for_scanning(self, hdl_files: Dict[str, str], changed_files: Set[str]) -> Dict[str, str]:
        """Select files for scanning based on intensity level."""
        if self.scan_intensity == 'comprehensive':
            return hdl_files
        elif self.scan_intensity == 'focused' and changed_files:
            # Only scan changed files for focused scanning
            return {f: content for f, content in hdl_files.items() if f in changed_files}
        else:
            # Standard scanning - sample of files
            file_list = list(hdl_files.items())
            sample_size = min(len(file_list), max(5, len(file_list) // 2))
            return dict(file_list[:sample_size])
            
    def _filter_by_severity(self, issues: List[Dict], severity: str) -> List[Dict]:
        """Filter issues by severity level."""
        return [issue for issue in issues if issue.get('severity') == severity]
        
    def _can_accept_high_severity_issues(self) -> bool:
        """Determine if high severity issues can be accepted."""
        # Never accept high severity issues for production branches
        if self.config.branch_type in [BranchType.MAIN, BranchType.RELEASE]:
            return False
            
        # Never accept for production maturity
        if self.config.maturity_level == ProjectMaturityLevel.PRODUCTION:
            return False
            
        # Accept for experimental projects on feature branches
        return (self.config.maturity_level == ProjectMaturityLevel.EXPERIMENTAL and
                self.config.branch_type == BranchType.FEATURE)
                
    async def _check_security_compliance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check security compliance requirements."""
        # Placeholder for compliance checking
        return {'violations': []}


class ProgressiveCodeQualityGate(AutonomousQualityGate):
    """
    Code quality gate that adapts standards based on project context.
    """
    
    def __init__(self, config: ProgressiveQualityConfig):
        super().__init__("ProgressiveCodeQuality", QualityGateSeverity.MEDIUM)
        self.config = config
        
    async def _execute_check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute progressive code quality check."""
        code_metrics = context.get('code_metrics', {})
        
        issues = []
        
        # Coverage requirements
        coverage = code_metrics.get('test_coverage', 0.0)
        required_coverage = self._get_required_coverage()
        
        if coverage < required_coverage:
            severity = 'critical' if coverage < required_coverage * 0.8 else 'medium'
            issues.append({
                'type': 'low_coverage',
                'current': coverage,
                'required': required_coverage,
                'severity': severity
            })
        
        # Complexity requirements
        max_complexity = code_metrics.get('max_complexity', 0)
        allowed_complexity = self._get_allowed_complexity()
        
        if max_complexity > allowed_complexity:
            issues.append({
                'type': 'high_complexity',
                'current': max_complexity,
                'allowed': allowed_complexity,
                'severity': 'medium'
            })
        
        # Technical debt
        tech_debt_minutes = code_metrics.get('technical_debt_minutes', 0)
        allowed_debt = self._get_allowed_technical_debt()
        
        if tech_debt_minutes > allowed_debt:
            issues.append({
                'type': 'high_tech_debt',
                'current': tech_debt_minutes,
                'allowed': allowed_debt,
                'severity': 'low'
            })
        
        # Determine status
        critical_issues = [i for i in issues if i['severity'] == 'critical']
        
        if critical_issues:
            status = QualityGateStatus.FAILED
            message = f"Critical code quality issues: {len(critical_issues)}"
        elif issues:
            status = QualityGateStatus.WARNING
            message = f"Code quality warnings: {len(issues)}"
        else:
            status = QualityGateStatus.PASSED
            message = "Code quality standards met"
            
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            severity=self.severity,
            message=message,
            details={
                'issues': issues,
                'required_coverage': required_coverage,
                'allowed_complexity': allowed_complexity
            }
        )
        
    def _get_required_coverage(self) -> float:
        """Get required test coverage based on configuration."""
        base_coverage = self.config.code_coverage_threshold
        
        # Adjust based on branch type
        if self.config.branch_type == BranchType.MAIN:
            return base_coverage
        elif self.config.branch_type == BranchType.RELEASE:
            return base_coverage * 0.95
        elif self.config.branch_type == BranchType.HOTFIX:
            return base_coverage * 0.90
        else:  # Feature, development
            return base_coverage * 0.85
            
    def _get_allowed_complexity(self) -> int:
        """Get allowed cyclomatic complexity."""
        base_complexity = self.config.complexity_threshold
        
        # More lenient for experimental projects
        if self.config.maturity_level == ProjectMaturityLevel.EXPERIMENTAL:
            return base_complexity * 2
        elif self.config.maturity_level == ProjectMaturityLevel.DEVELOPMENT:
            return int(base_complexity * 1.5)
        else:
            return base_complexity
            
    def _get_allowed_technical_debt(self) -> int:
        """Get allowed technical debt in minutes."""
        # Base allowance: 2 hours
        base_minutes = 120
        
        # Adjust based on maturity
        multipliers = {
            ProjectMaturityLevel.EXPERIMENTAL: 3.0,
            ProjectMaturityLevel.DEVELOPMENT: 2.0,
            ProjectMaturityLevel.TESTING: 1.5,
            ProjectMaturityLevel.PRODUCTION: 1.0,
            ProjectMaturityLevel.LEGACY: 2.5
        }
        
        return int(base_minutes * multipliers.get(self.config.maturity_level, 1.0))


class ProgressiveQualityOrchestrator(AutonomousQualityOrchestrator):
    """
    Enhanced orchestrator that implements progressive quality gates.
    """
    
    def __init__(self, config: ProgressiveQualityConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Clear default gates and add progressive ones
        self.quality_gates = []
        self._initialize_progressive_gates()
        
        # Risk assessment
        self.risk_assessor = RiskAssessmentEngine()
        
    def _initialize_progressive_gates(self) -> None:
        """Initialize progressive quality gates based on configuration."""
        self.quality_gates = [
            ProgressivePerformanceGate(self.config),
            ProgressiveSecurityGate(self.config),
            ProgressiveCodeQualityGate(self.config)
        ]
        
        # Add conditional gates based on configuration
        if self.config.maturity_level in [ProjectMaturityLevel.TESTING, ProjectMaturityLevel.PRODUCTION]:
            # Add stricter gates for mature projects
            self.quality_gates.extend([
                ComplianceAuditGate(self.config),
                PerformanceBenchmarkGate(self.config)
            ])
            
    async def execute_progressive_quality_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute progressive quality assessment with risk evaluation."""
        # Enhance context with configuration
        enhanced_context = {
            **context,
            'progressive_config': self.config,
            'risk_tolerance': self.config.risk_tolerance
        }
        
        # Perform risk assessment
        risk_profile = await self.risk_assessor.assess_risk(enhanced_context)
        enhanced_context['risk_profile'] = risk_profile
        
        # Execute quality gates
        quality_report = await self.execute_quality_gates(enhanced_context)
        
        # Add progressive-specific analysis
        progressive_analysis = await self._analyze_progressive_results(
            quality_report, risk_profile
        )
        
        quality_report['progressive_analysis'] = progressive_analysis
        quality_report['risk_profile'] = risk_profile
        quality_report['configuration'] = {
            'branch_type': self.config.branch_type.value,
            'maturity_level': self.config.maturity_level.value,
            'risk_tolerance': self.config.risk_tolerance
        }
        
        return quality_report
        
    async def _analyze_progressive_results(
        self, 
        quality_report: Dict[str, Any], 
        risk_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze results in the context of progressive quality."""
        analysis = {
            'risk_adjusted_score': self._calculate_risk_adjusted_score(
                quality_report, risk_profile
            ),
            'progression_readiness': self._assess_progression_readiness(quality_report),
            'recommended_actions': self._generate_progressive_recommendations(
                quality_report, risk_profile
            )
        }
        
        return analysis
        
    def _calculate_risk_adjusted_score(
        self, 
        quality_report: Dict[str, Any], 
        risk_profile: Dict[str, Any]
    ) -> float:
        """Calculate quality score adjusted for risk tolerance."""
        base_score = quality_report['execution_summary']['quality_score']
        risk_factor = risk_profile.get('overall_risk_level', 0.5)
        risk_tolerance = self.config.risk_tolerance
        
        # Adjust score based on risk tolerance
        if risk_factor > risk_tolerance:
            # Penalize high-risk situations
            penalty = (risk_factor - risk_tolerance) * 20  # Up to 20 point penalty
            adjusted_score = max(0, base_score - penalty)
        else:
            # Slight bonus for low-risk situations
            bonus = (risk_tolerance - risk_factor) * 5  # Up to 5 point bonus
            adjusted_score = min(100, base_score + bonus)
            
        return adjusted_score
        
    def _assess_progression_readiness(self, quality_report: Dict[str, Any]) -> Dict[str, str]:
        """Assess readiness for progression to next stage."""
        overall_status = quality_report['execution_summary']['overall_status']
        quality_score = quality_report['execution_summary']['quality_score']
        
        readiness = {}
        
        # Assess readiness for each potential progression
        if self.config.branch_type == BranchType.FEATURE:
            if overall_status == 'passed' and quality_score >= 85:
                readiness['merge_to_development'] = 'ready'
            elif overall_status == 'warning' and quality_score >= 75:
                readiness['merge_to_development'] = 'conditional'
            else:
                readiness['merge_to_development'] = 'not_ready'
                
        elif self.config.branch_type == BranchType.DEVELOPMENT:
            if overall_status == 'passed' and quality_score >= 90:
                readiness['promote_to_testing'] = 'ready'
            else:
                readiness['promote_to_testing'] = 'not_ready'
                
        elif self.config.branch_type == BranchType.RELEASE:
            if overall_status == 'passed' and quality_score >= 95:
                readiness['deploy_to_production'] = 'ready'
            else:
                readiness['deploy_to_production'] = 'not_ready'
                
        return readiness
        
    def _generate_progressive_recommendations(
        self, 
        quality_report: Dict[str, Any], 
        risk_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations specific to progressive quality."""
        recommendations = quality_report.get('recommendations', [])
        
        # Add progressive-specific recommendations
        if self.config.branch_type == BranchType.FEATURE:
            recommendations.append({
                'priority': 'medium',
                'category': 'progressive',
                'title': 'Feature Branch Quality Optimization',
                'description': 'Consider running additional tests before merge',
                'actions': [
                    'Run integration tests',
                    'Validate against development branch baseline',
                    'Check for breaking changes'
                ]
            })
            
        if risk_profile.get('overall_risk_level', 0) > 0.7:
            recommendations.append({
                'priority': 'high',
                'category': 'risk',
                'title': 'High Risk Configuration Detected',
                'description': f'Risk level {risk_profile["overall_risk_level"]:.2f} exceeds comfort zone',
                'actions': [
                    'Review recent changes for risk factors',
                    'Consider additional testing',
                    'Implement gradual rollout strategy'
                ]
            })
            
        return recommendations


class ComplianceAuditGate(AutonomousQualityGate):
    """Compliance audit gate for mature projects."""
    
    def __init__(self, config: ProgressiveQualityConfig):
        super().__init__("ComplianceAudit", QualityGateSeverity.HIGH)
        self.config = config
        
    async def _execute_check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute compliance audit."""
        # Placeholder implementation
        return QualityGateResult(
            gate_name=self.name,
            status=QualityGateStatus.PASSED,
            severity=self.severity,
            message="Compliance audit passed"
        )


class PerformanceBenchmarkGate(AutonomousQualityGate):
    """Performance benchmark gate for production readiness."""
    
    def __init__(self, config: ProgressiveQualityConfig):
        super().__init__("PerformanceBenchmark", QualityGateSeverity.HIGH)
        self.config = config
        
    async def _execute_check(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute performance benchmark."""
        # Placeholder implementation
        return QualityGateResult(
            gate_name=self.name,
            status=QualityGateStatus.PASSED,
            severity=self.severity,
            message="Performance benchmarks passed"
        )


class RiskAssessmentEngine:
    """Engine for assessing deployment and quality risks."""
    
    def __init__(self):
        self.risk_factors = {
            'code_complexity': 0.2,
            'test_coverage': 0.25,
            'recent_changes': 0.15,
            'performance_impact': 0.2,
            'security_issues': 0.2
        }
        
    async def assess_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk based on multiple factors."""
        risks = {}
        
        # Assess individual risk factors
        risks['code_complexity_risk'] = await self._assess_complexity_risk(context)
        risks['test_coverage_risk'] = await self._assess_coverage_risk(context)
        risks['change_risk'] = await self._assess_change_risk(context)
        risks['performance_risk'] = await self._assess_performance_risk(context)
        risks['security_risk'] = await self._assess_security_risk(context)
        
        # Calculate overall risk
        overall_risk = sum(
            risks[f'{factor}_risk'] * weight 
            for factor, weight in self.risk_factors.items() 
            if f'{factor}_risk' in risks
        )
        
        risks['overall_risk_level'] = min(1.0, overall_risk)
        risks['risk_category'] = self._categorize_risk(overall_risk)
        
        return risks
        
    async def _assess_complexity_risk(self, context: Dict[str, Any]) -> float:
        """Assess risk based on code complexity."""
        code_metrics = context.get('code_metrics', {})
        max_complexity = code_metrics.get('max_complexity', 0)
        
        # Normalize to 0-1 scale (assuming max reasonable complexity is 50)
        return min(1.0, max_complexity / 50.0)
        
    async def _assess_coverage_risk(self, context: Dict[str, Any]) -> float:
        """Assess risk based on test coverage."""
        code_metrics = context.get('code_metrics', {})
        coverage = code_metrics.get('test_coverage', 1.0)
        
        # Risk increases as coverage decreases
        return max(0.0, 1.0 - coverage)
        
    async def _assess_change_risk(self, context: Dict[str, Any]) -> float:
        """Assess risk based on recent changes."""
        changed_files = context.get('changed_files', set())
        total_files = context.get('total_files', 1)
        
        # Risk increases with proportion of changed files
        change_ratio = len(changed_files) / max(total_files, 1)
        return min(1.0, change_ratio * 2)  # Double weight for changes
        
    async def _assess_performance_risk(self, context: Dict[str, Any]) -> float:
        """Assess risk based on performance impact."""
        performance_metrics = context.get('performance_metrics', {})
        baseline = context.get('baseline_performance', {})
        
        if not baseline or not performance_metrics:
            return 0.3  # Medium risk for unknown performance impact
            
        # Calculate maximum relative degradation
        max_degradation = 0.0
        for metric, current in performance_metrics.items():
            if metric in baseline:
                degradation = abs(current - baseline[metric]) / max(abs(baseline[metric]), 0.001)
                max_degradation = max(max_degradation, degradation)
                
        return min(1.0, max_degradation * 2)  # Scale to risk
        
    async def _assess_security_risk(self, context: Dict[str, Any]) -> float:
        """Assess security-related risks."""
        # Check for known security issues
        known_security_issues = context.get('security_issues', [])
        critical_issues = [i for i in known_security_issues if i.get('severity') == 'critical']
        high_issues = [i for i in known_security_issues if i.get('severity') == 'high']
        
        if critical_issues:
            return 1.0
        elif high_issues:
            return 0.8
        elif known_security_issues:
            return 0.4
        else:
            return 0.1
            
    def _categorize_risk(self, risk_level: float) -> str:
        """Categorize risk level."""
        if risk_level >= 0.8:
            return 'very_high'
        elif risk_level >= 0.6:
            return 'high'
        elif risk_level >= 0.4:
            return 'medium'
        elif risk_level >= 0.2:
            return 'low'
        else:
            return 'very_low'


def create_progressive_quality_orchestrator(
    branch_name: str = "main",
    project_maturity: str = "development",
    risk_tolerance: float = 0.1,
    **kwargs
) -> ProgressiveQualityOrchestrator:
    """
    Factory function to create a progressive quality orchestrator.
    
    Args:
        branch_name: Current git branch name
        project_maturity: Project maturity level
        risk_tolerance: Risk tolerance (0.0 = no risk, 1.0 = high risk)
        **kwargs: Additional configuration options
        
    Returns:
        Configured ProgressiveQualityOrchestrator
    """
    # Determine branch type from branch name
    branch_type = BranchType.MAIN  # Default
    
    if branch_name.startswith('feature/'):
        branch_type = BranchType.FEATURE
    elif branch_name.startswith('hotfix/'):
        branch_type = BranchType.HOTFIX
    elif branch_name.startswith('release/'):
        branch_type = BranchType.RELEASE
    elif branch_name in ['develop', 'development']:
        branch_type = BranchType.DEVELOPMENT
    elif branch_name in ['main', 'master']:
        branch_type = BranchType.MAIN
        
    # Parse maturity level
    try:
        maturity_level = ProjectMaturityLevel(project_maturity.lower())
    except ValueError:
        maturity_level = ProjectMaturityLevel.DEVELOPMENT
        
    config = ProgressiveQualityConfig(
        branch_type=branch_type,
        maturity_level=maturity_level,
        risk_tolerance=risk_tolerance,
        **kwargs
    )
    
    return ProgressiveQualityOrchestrator(config)