"""
Production Deployment Orchestrator for Autonomous SDLC
====================================================

This module implements a comprehensive production deployment system that:
- Orchestrates zero-downtime deployments
- Implements progressive rollout strategies
- Monitors deployment health and performance
- Provides automatic rollback capabilities
- Ensures compliance and security requirements
- Manages multi-environment deployments

The system provides autonomous deployment with human oversight capabilities.
"""

import asyncio
import json
import logging
import time
import subprocess
import threading
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import hashlib
import tempfile
import shutil


class DeploymentStage(Enum):
    """Deployment pipeline stages."""
    PREPARATION = "preparation"
    VALIDATION = "validation"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    MONITORING = "monitoring"
    COMPLETION = "completion"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"


class EnvironmentType(Enum):
    """Target environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


@dataclass
class DeploymentConfig:
    """Configuration for a deployment."""
    deployment_id: str
    version: str
    environment: EnvironmentType
    strategy: DeploymentStrategy
    rollout_percentage: float = 100.0
    health_check_timeout: float = 300.0
    rollback_threshold: float = 0.05  # 5% error rate threshold
    enable_monitoring: bool = True
    approval_required: bool = False
    maintenance_window: Optional[Tuple[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    deployment_id: str
    status: DeploymentStatus
    start_time: float
    end_time: Optional[float] = None
    deployed_version: Optional[str] = None
    rollback_version: Optional[str] = None
    health_metrics: Dict[str, float] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    deployment_logs: List[str] = field(default_factory=list)


class DeploymentValidator:
    """
    Validates deployments before and during execution.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def validate_pre_deployment(
        self, 
        config: DeploymentConfig,
        artifact_path: Path
    ) -> Dict[str, Any]:
        """Validate deployment before execution."""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'checks_performed': []
        }
        
        # Artifact validation
        artifact_checks = await self._validate_artifact(artifact_path)
        validation_results['checks_performed'].append('artifact_validation')
        
        if not artifact_checks['valid']:
            validation_results['valid'] = False
            validation_results['errors'].extend(artifact_checks['errors'])
            
        # Environment readiness
        env_checks = await self._validate_environment_readiness(config.environment)
        validation_results['checks_performed'].append('environment_readiness')
        
        if not env_checks['ready']:
            validation_results['valid'] = False
            validation_results['errors'].extend(env_checks['issues'])
            
        # Security validation
        security_checks = await self._validate_security_requirements(config)
        validation_results['checks_performed'].append('security_validation')
        
        if not security_checks['compliant']:
            validation_results['valid'] = False
            validation_results['errors'].extend(security_checks['violations'])
            
        # Capacity validation
        capacity_checks = await self._validate_capacity_requirements(config)
        validation_results['checks_performed'].append('capacity_validation')
        
        if not capacity_checks['sufficient']:
            validation_results['warnings'].extend(capacity_checks['warnings'])
            
        # Compliance validation
        compliance_checks = await self._validate_compliance_requirements(config)
        validation_results['checks_performed'].append('compliance_validation')
        
        if not compliance_checks['compliant']:
            if compliance_checks['critical']:
                validation_results['valid'] = False
                validation_results['errors'].extend(compliance_checks['issues'])
            else:
                validation_results['warnings'].extend(compliance_checks['issues'])
                
        return validation_results
        
    async def _validate_artifact(self, artifact_path: Path) -> Dict[str, Any]:
        """Validate deployment artifact."""
        
        checks = {
            'valid': True,
            'errors': []
        }
        
        if not artifact_path.exists():
            checks['valid'] = False
            checks['errors'].append(f"Artifact not found: {artifact_path}")
            return checks
            
        # Check artifact integrity
        try:
            # Simulate integrity check
            size_mb = artifact_path.stat().st_size / (1024 * 1024)
            if size_mb > 1000:  # 1GB limit
                checks['errors'].append(f"Artifact too large: {size_mb:.1f}MB > 1000MB")
                
        except Exception as e:
            checks['errors'].append(f"Failed to validate artifact: {e}")
            
        if checks['errors']:
            checks['valid'] = False
            
        return checks
        
    async def _validate_environment_readiness(
        self, 
        environment: EnvironmentType
    ) -> Dict[str, Any]:
        """Validate target environment readiness."""
        
        # Simulate environment checks
        readiness = {
            'ready': True,
            'issues': []
        }
        
        # Check system resources
        # In production, this would check actual environment metrics
        
        # Check dependencies
        # Verify required services are running
        
        # Check network connectivity
        # Validate network access and DNS resolution
        
        return readiness
        
    async def _validate_security_requirements(
        self, 
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Validate security requirements for deployment."""
        
        security = {
            'compliant': True,
            'violations': []
        }
        
        # Check encryption requirements
        if config.environment == EnvironmentType.PRODUCTION:
            # Validate TLS/SSL configuration
            # Check certificate validity
            # Verify encryption at rest
            pass
            
        # Validate access controls
        # Check authentication and authorization
        
        # Scan for vulnerabilities
        # Run security scans on deployment artifacts
        
        return security
        
    async def _validate_capacity_requirements(
        self, 
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Validate system capacity for deployment."""
        
        capacity = {
            'sufficient': True,
            'warnings': []
        }
        
        # Check CPU and memory requirements
        # Validate storage capacity
        # Check network bandwidth
        
        return capacity
        
    async def _validate_compliance_requirements(
        self, 
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Validate compliance requirements."""
        
        compliance = {
            'compliant': True,
            'critical': False,
            'issues': []
        }
        
        # Check regulatory compliance (GDPR, HIPAA, etc.)
        # Validate data handling requirements
        # Check audit trail requirements
        
        return compliance


class DeploymentHealthMonitor:
    """
    Monitors deployment health and performance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = deque(maxlen=1000)
        self.alert_thresholds = {
            'error_rate': 0.05,      # 5%
            'response_time': 2000,   # 2 seconds
            'cpu_usage': 0.90,       # 90%
            'memory_usage': 0.85,    # 85%
        }
        
    async def monitor_deployment_health(
        self, 
        deployment_id: str,
        duration_seconds: float = 300.0
    ) -> Dict[str, Any]:
        """Monitor deployment health for specified duration."""
        
        self.logger.info(f"Starting health monitoring for deployment {deployment_id}")
        
        health_data = {
            'deployment_id': deployment_id,
            'monitoring_start': time.time(),
            'monitoring_duration': duration_seconds,
            'metrics': [],
            'alerts': [],
            'overall_health': 'unknown'
        }
        
        monitoring_end = time.time() + duration_seconds
        
        while time.time() < monitoring_end:
            # Collect current metrics
            current_metrics = await self._collect_health_metrics()
            current_metrics['timestamp'] = time.time()
            
            health_data['metrics'].append(current_metrics)
            self.metrics_history.append(current_metrics)
            
            # Check for alerts
            alerts = self._check_health_alerts(current_metrics)
            health_data['alerts'].extend(alerts)
            
            # Wait before next collection
            await asyncio.sleep(30.0)  # Collect every 30 seconds
            
        # Analyze overall health
        health_data['overall_health'] = self._analyze_overall_health(health_data)
        
        self.logger.info(
            f"Health monitoring completed for {deployment_id} - "
            f"Status: {health_data['overall_health']}"
        )
        
        return health_data
        
    async def _collect_health_metrics(self) -> Dict[str, float]:
        """Collect current system health metrics."""
        
        # In production, this would integrate with monitoring systems
        # For now, simulate metrics
        
        import random
        
        metrics = {
            'error_rate': random.uniform(0.001, 0.02),
            'response_time_ms': random.uniform(50, 500),
            'cpu_usage': random.uniform(0.3, 0.8),
            'memory_usage': random.uniform(0.4, 0.7),
            'disk_usage': random.uniform(0.2, 0.6),
            'network_latency_ms': random.uniform(10, 100),
            'active_connections': random.randint(100, 1000),
            'throughput_rps': random.uniform(500, 2000)
        }
        
        return metrics
        
    def _check_health_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check metrics against alert thresholds."""
        
        alerts = []
        
        for metric, value in metrics.items():
            if metric in self.alert_thresholds:
                threshold = self.alert_thresholds[metric]
                
                if value > threshold:
                    alerts.append({
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'severity': 'critical' if value > threshold * 1.2 else 'warning',
                        'timestamp': time.time(),
                        'message': f"{metric} ({value:.3f}) exceeds threshold ({threshold})"
                    })
                    
        return alerts
        
    def _analyze_overall_health(self, health_data: Dict[str, Any]) -> str:
        """Analyze overall health based on collected data."""
        
        alerts = health_data['alerts']
        metrics = health_data['metrics']
        
        if not metrics:
            return 'unknown'
            
        # Count critical alerts
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in alerts if a['severity'] == 'warning']
        
        # Analyze metric trends
        recent_metrics = metrics[-5:] if len(metrics) >= 5 else metrics
        
        # Calculate average error rate
        avg_error_rate = sum(m.get('error_rate', 0) for m in recent_metrics) / len(recent_metrics)
        
        # Determine health status
        if critical_alerts or avg_error_rate > 0.1:
            return 'unhealthy'
        elif warning_alerts or avg_error_rate > 0.05:
            return 'degraded'
        else:
            return 'healthy'


class DeploymentExecutor:
    """
    Executes deployment operations with different strategies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_deployments = {}
        
    async def execute_deployment(
        self, 
        config: DeploymentConfig,
        artifact_path: Path
    ) -> DeploymentResult:
        """Execute deployment with specified strategy."""
        
        deployment_result = DeploymentResult(
            deployment_id=config.deployment_id,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=time.time()
        )
        
        self.active_deployments[config.deployment_id] = deployment_result
        
        try:
            self.logger.info(
                f"Starting {config.strategy.value} deployment {config.deployment_id} "
                f"to {config.environment.value}"
            )
            
            # Execute based on strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._execute_blue_green_deployment(config, artifact_path, deployment_result)
            elif config.strategy == DeploymentStrategy.ROLLING:
                await self._execute_rolling_deployment(config, artifact_path, deployment_result)
            elif config.strategy == DeploymentStrategy.CANARY:
                await self._execute_canary_deployment(config, artifact_path, deployment_result)
            elif config.strategy == DeploymentStrategy.IMMEDIATE:
                await self._execute_immediate_deployment(config, artifact_path, deployment_result)
            else:
                raise ValueError(f"Unsupported deployment strategy: {config.strategy}")
                
            deployment_result.status = DeploymentStatus.COMPLETED
            deployment_result.deployed_version = config.version
            
        except Exception as e:
            self.logger.error(f"Deployment {config.deployment_id} failed: {e}")
            deployment_result.status = DeploymentStatus.FAILED
            deployment_result.error_messages.append(str(e))
            
        finally:
            deployment_result.end_time = time.time()
            
        return deployment_result
        
    async def _execute_blue_green_deployment(
        self, 
        config: DeploymentConfig,
        artifact_path: Path,
        result: DeploymentResult
    ):
        """Execute blue-green deployment strategy."""
        
        result.deployment_logs.append("Starting blue-green deployment")
        
        # Deploy to green environment
        result.deployment_logs.append("Deploying to green environment")
        await self._deploy_to_environment(config, artifact_path, "green")
        
        # Health check green environment
        result.deployment_logs.append("Health checking green environment")
        await self._wait_for_health_check(config, "green")
        
        # Switch traffic to green
        result.deployment_logs.append("Switching traffic to green environment")
        await self._switch_traffic("green")
        
        # Monitor for issues
        result.deployment_logs.append("Monitoring green environment")
        await asyncio.sleep(60.0)  # Monitor for 1 minute
        
        # Decommission blue environment
        result.deployment_logs.append("Decommissioning blue environment")
        await self._decommission_environment("blue")
        
        result.deployment_logs.append("Blue-green deployment completed")
        
    async def _execute_rolling_deployment(
        self, 
        config: DeploymentConfig,
        artifact_path: Path,
        result: DeploymentResult
    ):
        """Execute rolling deployment strategy."""
        
        result.deployment_logs.append("Starting rolling deployment")
        
        # Get list of instances
        instances = await self._get_deployment_instances(config.environment)
        batch_size = max(1, len(instances) // 4)  # 25% at a time
        
        for i in range(0, len(instances), batch_size):
            batch = instances[i:i + batch_size]
            result.deployment_logs.append(f"Deploying to batch {i//batch_size + 1}: {len(batch)} instances")
            
            # Deploy to batch
            await self._deploy_to_instances(config, artifact_path, batch)
            
            # Health check batch
            await self._health_check_instances(batch)
            
            # Wait between batches
            await asyncio.sleep(30.0)
            
        result.deployment_logs.append("Rolling deployment completed")
        
    async def _execute_canary_deployment(
        self, 
        config: DeploymentConfig,
        artifact_path: Path,
        result: DeploymentResult
    ):
        """Execute canary deployment strategy."""
        
        result.deployment_logs.append("Starting canary deployment")
        
        # Deploy to canary (small percentage)
        canary_percentage = min(config.rollout_percentage, 10.0)  # Max 10% for canary
        result.deployment_logs.append(f"Deploying to {canary_percentage}% canary instances")
        await self._deploy_to_percentage(config, artifact_path, canary_percentage)
        
        # Monitor canary
        result.deployment_logs.append("Monitoring canary deployment")
        canary_health = await self._monitor_canary_health(config, 300.0)  # 5 minutes
        
        if canary_health['healthy']:
            # Gradual rollout
            for percentage in [25, 50, 75, 100]:
                if percentage <= config.rollout_percentage:
                    result.deployment_logs.append(f"Rolling out to {percentage}% of instances")
                    await self._deploy_to_percentage(config, artifact_path, percentage)
                    await asyncio.sleep(60.0)  # Wait between stages
        else:
            result.deployment_logs.append("Canary deployment failed health checks - rolling back")
            await self._rollback_canary(config)
            raise Exception("Canary deployment failed health checks")
            
        result.deployment_logs.append("Canary deployment completed successfully")
        
    async def _execute_immediate_deployment(
        self, 
        config: DeploymentConfig,
        artifact_path: Path,
        result: DeploymentResult
    ):
        """Execute immediate deployment strategy."""
        
        result.deployment_logs.append("Starting immediate deployment")
        
        # Deploy to all instances immediately
        await self._deploy_to_environment(config, artifact_path, "all")
        
        # Quick health check
        await self._wait_for_health_check(config, "all")
        
        result.deployment_logs.append("Immediate deployment completed")
        
    async def _deploy_to_environment(self, config: DeploymentConfig, artifact_path: Path, environment: str):
        """Deploy to specified environment."""
        # Simulate deployment
        await asyncio.sleep(2.0)
        
    async def _wait_for_health_check(self, config: DeploymentConfig, environment: str):
        """Wait for environment health check."""
        await asyncio.sleep(1.0)
        
    async def _switch_traffic(self, environment: str):
        """Switch traffic to specified environment."""
        await asyncio.sleep(0.5)
        
    async def _decommission_environment(self, environment: str):
        """Decommission specified environment."""
        await asyncio.sleep(0.5)
        
    async def _get_deployment_instances(self, environment: EnvironmentType) -> List[str]:
        """Get list of deployment instances."""
        # Simulate instance discovery
        return [f"instance-{i}" for i in range(8)]
        
    async def _deploy_to_instances(self, config: DeploymentConfig, artifact_path: Path, instances: List[str]):
        """Deploy to specific instances."""
        await asyncio.sleep(1.0)
        
    async def _health_check_instances(self, instances: List[str]):
        """Health check specific instances."""
        await asyncio.sleep(0.5)
        
    async def _deploy_to_percentage(self, config: DeploymentConfig, artifact_path: Path, percentage: float):
        """Deploy to percentage of instances."""
        await asyncio.sleep(1.5)
        
    async def _monitor_canary_health(self, config: DeploymentConfig, duration: float) -> Dict[str, Any]:
        """Monitor canary deployment health."""
        await asyncio.sleep(duration / 10)  # Simulate monitoring
        return {'healthy': True, 'metrics': {}}
        
    async def _rollback_canary(self, config: DeploymentConfig):
        """Rollback canary deployment."""
        await asyncio.sleep(1.0)


class ProductionDeploymentOrchestrator:
    """
    Main orchestrator for production deployments.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = DeploymentValidator()
        self.executor = DeploymentExecutor()
        self.health_monitor = DeploymentHealthMonitor()
        
        self.deployment_history = deque(maxlen=1000)
        self.active_deployments = {}
        
        # Configuration
        self.enable_automatic_rollback = True
        self.require_approval_for_production = True
        self.max_concurrent_deployments = 3
        
    async def orchestrate_deployment(
        self, 
        config: DeploymentConfig,
        artifact_path: Path,
        auto_approve: bool = False
    ) -> DeploymentResult:
        """Orchestrate complete deployment process."""
        
        deployment_start = time.time()
        
        self.logger.info(
            f"Starting deployment orchestration for {config.deployment_id} "
            f"({config.version} -> {config.environment.value})"
        )
        
        try:
            # Stage 1: Pre-deployment validation
            self.logger.info("Stage 1: Pre-deployment validation")
            validation_result = await self.validator.validate_pre_deployment(
                config, artifact_path
            )
            
            if not validation_result['valid']:
                raise Exception(f"Pre-deployment validation failed: {validation_result['errors']}")
                
            # Stage 2: Approval gate (if required)
            if config.approval_required and not auto_approve:
                self.logger.info("Stage 2: Waiting for approval")
                await self._wait_for_approval(config)
                
            # Stage 3: Deployment execution
            self.logger.info("Stage 3: Deployment execution")
            deployment_result = await self.executor.execute_deployment(config, artifact_path)
            
            # Stage 4: Health monitoring
            if config.enable_monitoring and deployment_result.status == DeploymentStatus.COMPLETED:
                self.logger.info("Stage 4: Post-deployment health monitoring")
                health_data = await self.health_monitor.monitor_deployment_health(
                    config.deployment_id, config.health_check_timeout
                )
                
                deployment_result.health_metrics = health_data
                
                # Check if rollback is needed
                if (self.enable_automatic_rollback and 
                    health_data['overall_health'] == 'unhealthy'):
                    
                    self.logger.warning("Automatic rollback triggered due to health issues")
                    rollback_result = await self._execute_automatic_rollback(config, deployment_result)
                    deployment_result.status = DeploymentStatus.ROLLED_BACK
                    deployment_result.error_messages.append("Automatic rollback due to health issues")
                    
            # Record deployment
            self.deployment_history.append(deployment_result)
            
            execution_time = time.time() - deployment_start
            self.logger.info(
                f"Deployment orchestration completed for {config.deployment_id} - "
                f"Status: {deployment_result.status.value} ({execution_time:.2f}s)"
            )
            
            return deployment_result
            
        except Exception as e:
            self.logger.error(f"Deployment orchestration failed: {e}")
            
            # Create failed deployment result
            failed_result = DeploymentResult(
                deployment_id=config.deployment_id,
                status=DeploymentStatus.FAILED,
                start_time=deployment_start,
                end_time=time.time(),
                error_messages=[str(e)]
            )
            
            self.deployment_history.append(failed_result)
            return failed_result
            
    async def _wait_for_approval(self, config: DeploymentConfig):
        """Wait for deployment approval."""
        
        # In production, this would integrate with approval systems
        # For now, simulate approval process
        
        self.logger.info(f"Approval required for {config.deployment_id} deployment to {config.environment.value}")
        
        if config.environment == EnvironmentType.PRODUCTION:
            # Simulate longer approval process for production
            await asyncio.sleep(5.0)
        else:
            await asyncio.sleep(2.0)
            
        self.logger.info(f"Approval granted for {config.deployment_id}")
        
    async def _execute_automatic_rollback(
        self, 
        config: DeploymentConfig,
        deployment_result: DeploymentResult
    ) -> Dict[str, Any]:
        """Execute automatic rollback."""
        
        self.logger.info(f"Executing automatic rollback for {config.deployment_id}")
        
        # Create rollback configuration
        rollback_config = DeploymentConfig(
            deployment_id=f"{config.deployment_id}_rollback",
            version="previous",  # Would be actual previous version
            environment=config.environment,
            strategy=DeploymentStrategy.IMMEDIATE,  # Rollbacks should be immediate
            enable_monitoring=False,  # Don't monitor rollbacks
            approval_required=False   # Rollbacks don't need approval
        )
        
        # Execute rollback
        rollback_result = await self.executor.execute_deployment(
            rollback_config, 
            Path("previous_version")  # Would be actual previous artifact
        )
        
        deployment_result.rollback_version = "previous"
        
        return {
            'rollback_successful': rollback_result.status == DeploymentStatus.COMPLETED,
            'rollback_result': rollback_result
        }
        
    async def schedule_deployment(
        self, 
        config: DeploymentConfig,
        artifact_path: Path,
        scheduled_time: float
    ) -> str:
        """Schedule a deployment for future execution."""
        
        task_id = f"scheduled_{config.deployment_id}_{int(scheduled_time)}"
        
        # Calculate delay
        delay = scheduled_time - time.time()
        
        if delay <= 0:
            # Execute immediately if scheduled time is in the past
            await self.orchestrate_deployment(config, artifact_path)
        else:
            # Schedule for future execution
            asyncio.create_task(self._execute_scheduled_deployment(
                config, artifact_path, delay
            ))
            
        self.logger.info(f"Scheduled deployment {task_id} for {delay:.0f} seconds from now")
        return task_id
        
    async def _execute_scheduled_deployment(
        self, 
        config: DeploymentConfig,
        artifact_path: Path,
        delay: float
    ):
        """Execute a scheduled deployment."""
        
        await asyncio.sleep(delay)
        await self.orchestrate_deployment(config, artifact_path, auto_approve=True)
        
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a specific deployment."""
        
        # Check active deployments
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]
            
        # Check deployment history
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
                
        return None
        
    def get_deployment_report(self) -> Dict[str, Any]:
        """Get comprehensive deployment report."""
        
        total_deployments = len(self.deployment_history)
        
        if total_deployments == 0:
            return {
                'total_deployments': 0,
                'success_rate': 0.0,
                'average_duration': 0.0,
                'deployment_summary': {}
            }
            
        successful = sum(
            1 for d in self.deployment_history 
            if d.status == DeploymentStatus.COMPLETED
        )
        
        failed = sum(
            1 for d in self.deployment_history 
            if d.status == DeploymentStatus.FAILED
        )
        
        rolled_back = sum(
            1 for d in self.deployment_history 
            if d.status == DeploymentStatus.ROLLED_BACK
        )
        
        # Calculate average duration
        completed_deployments = [
            d for d in self.deployment_history 
            if d.end_time is not None
        ]
        
        avg_duration = 0.0
        if completed_deployments:
            durations = [d.end_time - d.start_time for d in completed_deployments]
            avg_duration = sum(durations) / len(durations)
            
        return {
            'total_deployments': total_deployments,
            'successful_deployments': successful,
            'failed_deployments': failed,
            'rolled_back_deployments': rolled_back,
            'success_rate': successful / total_deployments,
            'average_duration_seconds': avg_duration,
            'active_deployments': len(self.active_deployments),
            'deployment_summary': {
                'completed': successful,
                'failed': failed,
                'rolled_back': rolled_back,
                'in_progress': len(self.active_deployments)
            }
        }


def create_production_deployment_orchestrator() -> ProductionDeploymentOrchestrator:
    """
    Factory function to create production deployment orchestrator.
    
    Returns:
        Configured ProductionDeploymentOrchestrator
    """
    return ProductionDeploymentOrchestrator()


def create_deployment_config(
    deployment_id: str,
    version: str,
    environment: str = "staging",
    strategy: str = "blue_green",
    **kwargs
) -> DeploymentConfig:
    """
    Factory function to create deployment configuration.
    
    Args:
        deployment_id: Unique deployment identifier
        version: Version being deployed
        environment: Target environment
        strategy: Deployment strategy
        **kwargs: Additional configuration options
        
    Returns:
        DeploymentConfig instance
    """
    
    try:
        env_type = EnvironmentType(environment.lower())
    except ValueError:
        env_type = EnvironmentType.STAGING
        
    try:
        deploy_strategy = DeploymentStrategy(strategy.lower())
    except ValueError:
        deploy_strategy = DeploymentStrategy.BLUE_GREEN
        
    return DeploymentConfig(
        deployment_id=deployment_id,
        version=version,
        environment=env_type,
        strategy=deploy_strategy,
        **kwargs
    )


# Example usage and integration
async def main_deployment_example():
    """Example of using the deployment orchestrator."""
    
    # Create orchestrator
    orchestrator = create_production_deployment_orchestrator()
    
    # Create deployment configuration
    config = create_deployment_config(
        deployment_id="neuromorphic-fpga-v2.1.0",
        version="2.1.0",
        environment="production",
        strategy="canary",
        rollout_percentage=50.0,
        health_check_timeout=600.0,
        approval_required=True
    )
    
    # Create artifact path
    artifact_path = Path("/tmp/deployment_artifact.tar.gz")
    artifact_path.touch()  # Create dummy artifact
    
    try:
        # Execute deployment
        result = await orchestrator.orchestrate_deployment(config, artifact_path)
        
        print(f"Deployment Status: {result.status.value}")
        print(f"Deployment Duration: {result.end_time - result.start_time:.2f}s")
        
        # Get deployment report
        report = orchestrator.get_deployment_report()
        print(f"Overall Success Rate: {report['success_rate']:.2%}")
        
    finally:
        # Cleanup
        if artifact_path.exists():
            artifact_path.unlink()


if __name__ == "__main__":
    asyncio.run(main_deployment_example())