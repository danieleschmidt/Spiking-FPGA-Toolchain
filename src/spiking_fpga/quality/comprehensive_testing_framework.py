"""Comprehensive testing and validation framework for the neuromorphic toolchain."""

import unittest
import pytest
import time
import tempfile
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import yaml
import hashlib
import statistics
from enum import Enum
import subprocess
import shutil
import gc


class TestSeverity(Enum):
    """Test severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestCategory(Enum):
    """Categories of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    STRESS = "stress"
    REGRESSION = "regression"
    ACCEPTANCE = "acceptance"


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    test_category: TestCategory
    severity: TestSeverity
    passed: bool
    duration: float
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestSuite:
    """Collection of related tests."""
    name: str
    description: str
    category: TestCategory
    tests: List[Callable] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    timeout_seconds: int = 300
    parallel_execution: bool = False


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    report_id: str
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration: float
    test_results: List[TestResult]
    quality_score: float
    compliance_status: Dict[str, bool]
    recommendations: List[str]
    performance_metrics: Dict[str, Any]


class ComprehensiveTestFramework:
    """Advanced testing framework with ML-driven test optimization."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_history: List[ValidationReport] = []
        self.test_environment: Dict[str, Any] = {}
        
        # Performance tracking
        self.test_metrics = TestMetricsTracker(self.logger)
        self.failure_analyzer = TestFailureAnalyzer(self.logger)
        
        # Test optimization
        self.test_optimizer = IntelligentTestOptimizer(self.logger)
        
        # Quality gates
        self.quality_gates = QualityGateValidator(self.logger)
        
        self.logger.info("ComprehensiveTestFramework initialized")
    
    def register_test_suite(self, suite: TestSuite) -> None:
        """Register a test suite."""
        self.test_suites[suite.name] = suite
        self.logger.info(f"Registered test suite: {suite.name} ({len(suite.tests)} tests)")
    
    def run_all_tests(self, parallel: bool = False, 
                     categories: Optional[List[TestCategory]] = None) -> ValidationReport:
        """Run all registered test suites."""
        start_time = datetime.now()
        
        # Filter test suites by category if specified
        suites_to_run = {}
        if categories:
            for name, suite in self.test_suites.items():
                if suite.category in categories:
                    suites_to_run[name] = suite
        else:
            suites_to_run = self.test_suites
        
        self.logger.info(f"Running {len(suites_to_run)} test suites")
        
        # Initialize test environment
        self._setup_test_environment()
        
        all_results = []
        
        try:
            if parallel and len(suites_to_run) > 1:
                # Parallel execution
                all_results = self._run_suites_parallel(suites_to_run)
            else:
                # Sequential execution
                all_results = self._run_suites_sequential(suites_to_run)
        
        finally:
            self._cleanup_test_environment()
        
        # Generate comprehensive report
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        report = self._generate_validation_report(all_results, total_duration)
        
        # Update test history and analytics
        self.test_history.append(report)
        self.test_metrics.record_test_run(report)
        
        # Analyze failures and update optimization
        if report.failed_tests > 0:
            self.failure_analyzer.analyze_failures(report.test_results)
            self.test_optimizer.update_from_failures(report.test_results)
        
        self.logger.info(f"Test execution completed: {report.passed_tests}/{report.total_tests} passed")
        return report
    
    def run_quality_gates(self, compilation_result: Any) -> Dict[str, bool]:
        """Run quality gates validation."""
        return self.quality_gates.validate_compilation_result(compilation_result)
    
    def _setup_test_environment(self) -> None:
        """Setup test environment."""
        self.test_environment = {
            "temp_dir": tempfile.mkdtemp(prefix="neuromorphic_test_"),
            "start_time": datetime.now(),
            "test_data_dir": Path(__file__).parent.parent.parent.parent / "test_data",
            "output_dir": Path(self.test_environment.get("temp_dir", "/tmp")) / "test_outputs"
        }
        
        # Create necessary directories
        Path(self.test_environment["output_dir"]).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Test environment setup: {self.test_environment['temp_dir']}")
    
    def _cleanup_test_environment(self) -> None:
        """Cleanup test environment."""
        if "temp_dir" in self.test_environment:
            try:
                shutil.rmtree(self.test_environment["temp_dir"])
                self.logger.info("Test environment cleaned up")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup test environment: {e}")
    
    def _run_suites_sequential(self, suites: Dict[str, TestSuite]) -> List[TestResult]:
        """Run test suites sequentially."""
        all_results = []
        
        for suite_name, suite in suites.items():
            self.logger.info(f"Running test suite: {suite_name}")
            
            suite_results = self._run_single_suite(suite)
            all_results.extend(suite_results)
            
            # Log suite summary
            passed = sum(1 for r in suite_results if r.passed)
            total = len(suite_results)
            self.logger.info(f"Suite {suite_name}: {passed}/{total} tests passed")
        
        return all_results
    
    def _run_suites_parallel(self, suites: Dict[str, TestSuite]) -> List[TestResult]:
        """Run test suites in parallel."""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=min(len(suites), mp.cpu_count())) as executor:
            # Submit all suites
            future_to_suite = {
                executor.submit(self._run_single_suite, suite): suite_name
                for suite_name, suite in suites.items()
            }
            
            # Collect results
            for future in future_to_suite:
                suite_name = future_to_suite[future]
                try:
                    suite_results = future.result()
                    all_results.extend(suite_results)
                    
                    passed = sum(1 for r in suite_results if r.passed)
                    total = len(suite_results)
                    self.logger.info(f"Suite {suite_name}: {passed}/{total} tests passed")
                    
                except Exception as e:
                    self.logger.error(f"Suite {suite_name} failed: {e}")
                    # Create failure result for the entire suite
                    all_results.append(TestResult(
                        test_name=f"{suite_name}_suite_execution",
                        test_category=TestCategory.INTEGRATION,
                        severity=TestSeverity.HIGH,
                        passed=False,
                        duration=0.0,
                        error_message=str(e)
                    ))
        
        return all_results
    
    def _run_single_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run a single test suite."""
        results = []
        
        # Setup suite
        if suite.setup_func:
            try:
                suite.setup_func(self.test_environment)
            except Exception as e:
                self.logger.error(f"Suite setup failed for {suite.name}: {e}")
                return [TestResult(
                    test_name=f"{suite.name}_setup",
                    test_category=suite.category,
                    severity=TestSeverity.CRITICAL,
                    passed=False,
                    duration=0.0,
                    error_message=str(e)
                )]
        
        # Run tests
        if suite.parallel_execution and len(suite.tests) > 1:
            results = self._run_tests_parallel(suite.tests, suite)
        else:
            results = self._run_tests_sequential(suite.tests, suite)
        
        # Teardown suite
        if suite.teardown_func:
            try:
                suite.teardown_func(self.test_environment)
            except Exception as e:
                self.logger.warning(f"Suite teardown failed for {suite.name}: {e}")
                results.append(TestResult(
                    test_name=f"{suite.name}_teardown",
                    test_category=suite.category,
                    severity=TestSeverity.LOW,
                    passed=False,
                    duration=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def _run_tests_sequential(self, tests: List[Callable], suite: TestSuite) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test_func in tests:
            result = self._execute_single_test(test_func, suite)
            results.append(result)
            
            # Early termination for critical failures
            if not result.passed and result.severity == TestSeverity.CRITICAL:
                self.logger.warning(f"Critical test failure in {suite.name}, stopping suite execution")
                break
        
        return results
    
    def _run_tests_parallel(self, tests: List[Callable], suite: TestSuite) -> List[TestResult]:
        """Run tests in parallel."""
        results = []
        
        max_workers = min(len(tests), 4)  # Limit parallel tests
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self._execute_single_test, test_func, suite): test_func
                for test_func in tests
            }
            
            # Collect results
            for future in future_to_test:
                test_func = future_to_test[future]
                try:
                    result = future.result(timeout=suite.timeout_seconds)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Test execution failed: {test_func.__name__}")
                    results.append(TestResult(
                        test_name=test_func.__name__,
                        test_category=suite.category,
                        severity=TestSeverity.HIGH,
                        passed=False,
                        duration=0.0,
                        error_message=str(e)
                    ))
        
        return results
    
    def _execute_single_test(self, test_func: Callable, suite: TestSuite) -> TestResult:
        """Execute a single test function."""
        test_name = test_func.__name__
        start_time = time.time()
        
        try:
            # Execute test with timeout
            if hasattr(test_func, '__timeout__'):
                timeout = test_func.__timeout__
            else:
                timeout = suite.timeout_seconds
            
            # Get test metadata
            severity = getattr(test_func, '__severity__', TestSeverity.MEDIUM)
            
            # Execute test
            test_result = test_func(self.test_environment)
            
            duration = time.time() - start_time
            
            # Handle different return types
            if isinstance(test_result, bool):
                passed = test_result
                metrics = {}
                warnings = []
            elif isinstance(test_result, dict):
                passed = test_result.get('passed', True)
                metrics = test_result.get('metrics', {})
                warnings = test_result.get('warnings', [])
            else:
                passed = True  # Assume success if no return value
                metrics = {}
                warnings = []
            
            return TestResult(
                test_name=test_name,
                test_category=suite.category,
                severity=severity,
                passed=passed,
                duration=duration,
                metrics=metrics,
                warnings=warnings
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test {test_name} failed: {e}")
            
            return TestResult(
                test_name=test_name,
                test_category=suite.category,
                severity=getattr(test_func, '__severity__', TestSeverity.MEDIUM),
                passed=False,
                duration=duration,
                error_message=str(e)
            )
    
    def _generate_validation_report(self, results: List[TestResult], 
                                  total_duration: float) -> ValidationReport:
        """Generate comprehensive validation report."""
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = sum(1 for r in results if not r.passed)
        
        # Calculate quality score
        if total_tests == 0:
            quality_score = 0.0
        else:
            # Base score from pass rate
            pass_rate = passed_tests / total_tests
            
            # Adjust for severity of failures
            critical_failures = sum(1 for r in results if not r.passed and r.severity == TestSeverity.CRITICAL)
            high_failures = sum(1 for r in results if not r.passed and r.severity == TestSeverity.HIGH)
            
            severity_penalty = (critical_failures * 0.3 + high_failures * 0.15)
            quality_score = max(0.0, pass_rate - severity_penalty)
        
        # Compliance status
        compliance_status = {
            "unit_tests": self._check_test_category_compliance(results, TestCategory.UNIT),
            "integration_tests": self._check_test_category_compliance(results, TestCategory.INTEGRATION),
            "performance_tests": self._check_test_category_compliance(results, TestCategory.PERFORMANCE),
            "security_tests": self._check_test_category_compliance(results, TestCategory.SECURITY)
        }
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, quality_score)
        
        return ValidationReport(
            report_id=f"validation_{int(time.time())}",
            timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=0,  # Not implemented yet
            total_duration=total_duration,
            test_results=results,
            quality_score=quality_score,
            compliance_status=compliance_status,
            recommendations=recommendations,
            performance_metrics=performance_metrics
        )
    
    def _check_test_category_compliance(self, results: List[TestResult], 
                                      category: TestCategory) -> bool:
        """Check compliance for a specific test category."""
        category_results = [r for r in results if r.test_category == category]
        
        if not category_results:
            return False  # No tests in category
        
        # Require at least 85% pass rate for compliance
        pass_rate = sum(1 for r in category_results if r.passed) / len(category_results)
        return pass_rate >= 0.85
    
    def _calculate_performance_metrics(self, results: List[TestResult]) -> Dict[str, Any]:
        """Calculate performance metrics from test results."""
        durations = [r.duration for r in results if r.duration > 0]
        
        if not durations:
            return {}
        
        return {
            "avg_test_duration": statistics.mean(durations),
            "max_test_duration": max(durations),
            "min_test_duration": min(durations),
            "total_test_time": sum(durations),
            "performance_test_results": [
                {
                    "test_name": r.test_name,
                    "duration": r.duration,
                    "metrics": r.metrics
                }
                for r in results if r.test_category == TestCategory.PERFORMANCE
            ]
        }
    
    def _generate_recommendations(self, results: List[TestResult], 
                                quality_score: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Quality score recommendations
        if quality_score < 0.7:
            recommendations.append("Quality score is below threshold. Focus on fixing failing tests.")
        
        # Performance recommendations
        slow_tests = [r for r in results if r.duration > 30.0]
        if slow_tests:
            recommendations.append(f"{len(slow_tests)} tests are running slowly (>30s). Consider optimization.")
        
        # Category-specific recommendations
        security_failures = [r for r in results if r.test_category == TestCategory.SECURITY and not r.passed]
        if security_failures:
            recommendations.append("Security test failures detected. Review security implementations.")
        
        performance_failures = [r for r in results if r.test_category == TestCategory.PERFORMANCE and not r.passed]
        if performance_failures:
            recommendations.append("Performance tests failing. Review optimization strategies.")
        
        # Coverage recommendations
        categories_with_tests = set(r.test_category for r in results)
        all_categories = set(TestCategory)
        missing_categories = all_categories - categories_with_tests
        
        if missing_categories:
            missing_names = [cat.value for cat in missing_categories]
            recommendations.append(f"Missing test coverage for: {', '.join(missing_names)}")
        
        if not recommendations:
            recommendations.append("All tests are performing well. Consider adding more comprehensive tests.")
        
        return recommendations
    
    def get_test_analytics(self) -> Dict[str, Any]:
        """Get comprehensive test analytics."""
        if not self.test_history:
            return {"status": "no_test_history"}
        
        recent_reports = self.test_history[-10:]  # Last 10 test runs
        
        # Trend analysis
        quality_trends = [r.quality_score for r in recent_reports]
        pass_rate_trends = [r.passed_tests / r.total_tests if r.total_tests > 0 else 0 
                           for r in recent_reports]
        
        return {
            "total_test_runs": len(self.test_history),
            "latest_quality_score": recent_reports[-1].quality_score,
            "average_quality_score": statistics.mean(quality_trends),
            "quality_trend": "improving" if quality_trends[-1] > quality_trends[0] else "declining",
            "average_pass_rate": statistics.mean(pass_rate_trends),
            "test_metrics": self.test_metrics.get_summary(),
            "failure_analysis": self.failure_analyzer.get_summary()
        }


class TestMetricsTracker:
    """Tracks detailed test metrics and performance."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics_history: List[Dict[str, Any]] = []
    
    def record_test_run(self, report: ValidationReport) -> None:
        """Record metrics from a test run."""
        metrics = {
            "timestamp": report.timestamp.isoformat(),
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
            "failed_tests": report.failed_tests,
            "quality_score": report.quality_score,
            "total_duration": report.total_duration,
            "avg_test_duration": report.performance_metrics.get("avg_test_duration", 0),
            "category_breakdown": self._analyze_category_breakdown(report.test_results)
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def _analyze_category_breakdown(self, results: List[TestResult]) -> Dict[str, Dict[str, int]]:
        """Analyze test results by category."""
        breakdown = {}
        
        for category in TestCategory:
            category_results = [r for r in results if r.test_category == category]
            if category_results:
                breakdown[category.value] = {
                    "total": len(category_results),
                    "passed": sum(1 for r in category_results if r.passed),
                    "failed": sum(1 for r in category_results if not r.passed)
                }
        
        return breakdown
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-5:]  # Last 5 runs
        
        return {
            "total_recorded_runs": len(self.metrics_history),
            "recent_average_quality": statistics.mean(m["quality_score"] for m in recent_metrics),
            "recent_average_duration": statistics.mean(m["total_duration"] for m in recent_metrics),
            "test_velocity": statistics.mean(m["total_tests"] / m["total_duration"] 
                                           for m in recent_metrics if m["total_duration"] > 0)
        }


class TestFailureAnalyzer:
    """Analyzes test failures to identify patterns and root causes."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.failure_patterns: Dict[str, int] = {}
        self.failure_history: List[Dict[str, Any]] = []
    
    def analyze_failures(self, results: List[TestResult]) -> None:
        """Analyze failed tests for patterns."""
        failed_results = [r for r in results if not r.passed]
        
        for failure in failed_results:
            # Pattern analysis
            pattern_key = f"{failure.test_category.value}_{failure.severity.value}"
            self.failure_patterns[pattern_key] = self.failure_patterns.get(pattern_key, 0) + 1
            
            # Record failure details
            failure_record = {
                "timestamp": failure.timestamp.isoformat(),
                "test_name": failure.test_name,
                "category": failure.test_category.value,
                "severity": failure.severity.value,
                "error_message": failure.error_message,
                "duration": failure.duration
            }
            
            self.failure_history.append(failure_record)
        
        # Keep only recent failures
        if len(self.failure_history) > 200:
            self.failure_history = self.failure_history[-200:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get failure analysis summary."""
        return {
            "failure_patterns": self.failure_patterns,
            "recent_failures": len(self.failure_history),
            "most_common_failure_type": max(self.failure_patterns.keys(), 
                                           key=self.failure_patterns.get) if self.failure_patterns else None
        }


class IntelligentTestOptimizer:
    """ML-driven test optimization and prioritization."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.test_priorities: Dict[str, float] = {}
        self.optimization_history: List[Dict[str, Any]] = []
    
    def update_from_failures(self, results: List[TestResult]) -> None:
        """Update test priorities based on failure patterns."""
        failed_results = [r for r in results if not r.passed]
        
        for failure in failed_results:
            test_name = failure.test_name
            
            # Increase priority for failed tests
            current_priority = self.test_priorities.get(test_name, 0.5)
            
            # Severity-based priority adjustment
            severity_multiplier = {
                TestSeverity.CRITICAL: 2.0,
                TestSeverity.HIGH: 1.5,
                TestSeverity.MEDIUM: 1.2,
                TestSeverity.LOW: 1.1,
                TestSeverity.INFO: 1.05
            }
            
            multiplier = severity_multiplier.get(failure.severity, 1.1)
            new_priority = min(current_priority * multiplier, 1.0)
            
            self.test_priorities[test_name] = new_priority
            
            self.logger.debug(f"Updated priority for {test_name}: {new_priority:.2f}")
    
    def get_test_execution_order(self, test_names: List[str]) -> List[str]:
        """Get optimized test execution order."""
        # Sort tests by priority (highest first)
        prioritized_tests = sorted(
            test_names,
            key=lambda name: self.test_priorities.get(name, 0.5),
            reverse=True
        )
        
        return prioritized_tests


class QualityGateValidator:
    """Validates quality gates for deployment readiness."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
        # Define quality gate thresholds
        self.thresholds = {
            "min_test_coverage": 0.85,
            "max_critical_failures": 0,
            "max_high_failures": 2,
            "min_performance_score": 0.8,
            "max_security_failures": 0
        }
    
    def validate_compilation_result(self, compilation_result: Any) -> Dict[str, bool]:
        """Validate compilation result against quality gates."""
        gates = {}
        
        # Basic success check
        gates["compilation_success"] = getattr(compilation_result, 'success', False)
        
        # Error count checks
        errors = getattr(compilation_result, 'errors', [])
        gates["no_critical_errors"] = len(errors) == 0
        
        # Warning count checks
        warnings = getattr(compilation_result, 'warnings', [])
        gates["acceptable_warning_count"] = len(warnings) <= 5
        
        # Performance checks
        if hasattr(compilation_result, 'resource_estimate'):
            gates["resource_constraints_met"] = True  # Placeholder
        else:
            gates["resource_constraints_met"] = False
        
        # Security checks
        if hasattr(compilation_result, 'security_audit'):
            security_audit = compilation_result.security_audit
            if hasattr(security_audit, 'risk_score'):
                gates["security_risk_acceptable"] = security_audit.risk_score < 0.7
            else:
                gates["security_risk_acceptable"] = True
        else:
            gates["security_risk_acceptable"] = True
        
        return gates


# Utility functions for test decorators
def test_severity(severity: TestSeverity):
    """Decorator to mark test severity."""
    def decorator(func):
        func.__severity__ = severity
        return func
    return decorator


def test_timeout(seconds: int):
    """Decorator to set test timeout."""
    def decorator(func):
        func.__timeout__ = seconds
        return func
    return decorator


# Example test suite creation function
def create_basic_test_suite() -> TestSuite:
    """Create a basic test suite for demonstration."""
    
    @test_severity(TestSeverity.CRITICAL)
    def test_import_basic_modules(test_env):
        """Test that basic modules can be imported."""
        try:
            from spiking_fpga import FPGATarget, compile_network
            return True
        except ImportError as e:
            return {"passed": False, "error": str(e)}
    
    @test_severity(TestSeverity.HIGH)
    def test_fpga_target_enumeration(test_env):
        """Test FPGA target enumeration."""
        from spiking_fpga import FPGATarget
        
        targets = list(FPGATarget)
        assert len(targets) >= 4  # Should have at least 4 targets
        
        # Test properties
        for target in targets:
            assert target.vendor in ["xilinx", "intel"]
            assert target.toolchain in ["vivado", "quartus"]
            assert isinstance(target.resources, dict)
        
        return True
    
    @test_severity(TestSeverity.MEDIUM)
    @test_timeout(30)
    def test_basic_compilation(test_env):
        """Test basic compilation functionality."""
        from spiking_fpga import compile_network, FPGATarget
        
        # Create a minimal test network
        test_network_path = Path(test_env["temp_dir"]) / "test_network.yaml"
        test_network_data = {
            "name": "Test Network",
            "timestep": 1.0,
            "layers": [
                {"name": "input", "type": "input", "size": 5},
                {"name": "output", "type": "output", "size": 2}
            ],
            "connections": [
                {"from": "input", "to": "output", "pattern": "all_to_all"}
            ]
        }
        
        with open(test_network_path, 'w') as f:
            yaml.dump(test_network_data, f)
        
        # Test compilation
        result = compile_network(
            test_network_path,
            FPGATarget.ARTIX7_35T,
            output_dir=test_env["output_dir"],
            run_synthesis=False
        )
        
        return {
            "passed": result.success,
            "metrics": {
                "compilation_time": getattr(result, 'duration', 0),
                "generated_files": len(result.hdl_files) if result.hdl_files else 0
            },
            "warnings": result.warnings if hasattr(result, 'warnings') else []
        }
    
    suite = TestSuite(
        name="basic_functionality",
        description="Basic functionality tests",
        category=TestCategory.UNIT,
        tests=[test_import_basic_modules, test_fpga_target_enumeration, test_basic_compilation]
    )
    
    return suite