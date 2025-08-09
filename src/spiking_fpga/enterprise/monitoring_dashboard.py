"""
Enterprise Monitoring Dashboard for Neuromorphic Computing

Real-time monitoring and alerting system featuring:
- System health and performance dashboards
- Multi-dimensional metrics collection and visualization
- Intelligent alerting with escalation policies
- Predictive analytics and anomaly detection
- Custom dashboards and reporting
- Integration with enterprise monitoring systems (Prometheus, Grafana, etc.)
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import threading
from collections import defaultdict, deque
import statistics
import numpy as np
from datetime import datetime, timedelta
import uuid
import socket

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(Enum):
    """Alert states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    state: AlertState
    metric_name: str
    threshold_value: float
    current_value: float
    tags: Dict[str, str] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    updated_time: float = field(default_factory=time.time)
    acknowledged_by: Optional[str] = None
    acknowledged_time: Optional[float] = None
    resolved_time: Optional[float] = None


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    dashboard_id: str
    name: str
    description: str
    panels: List[Dict[str, Any]] = field(default_factory=list)
    refresh_interval: int = 30  # seconds
    time_range: str = "1h"
    tags: List[str] = field(default_factory=list)
    created_by: str = "system"
    created_time: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics = defaultdict(lambda: deque(maxlen=retention_hours * 3600))  # 1 point/sec
        self.metric_metadata = {}
        self.lock = threading.Lock()
        
    def record_metric(self, metric: MetricPoint) -> None:
        """Record a metric data point."""
        with self.lock:
            self.metrics[metric.name].append(metric)
            self.metric_metadata[metric.name] = {
                'type': metric.metric_type,
                'last_update': metric.timestamp,
                'tags': metric.tags
            }
            
    def get_metric_history(self, metric_name: str, 
                          start_time: float, end_time: float) -> List[MetricPoint]:
        """Get metric history within time range."""
        with self.lock:
            if metric_name not in self.metrics:
                return []
                
            return [
                point for point in self.metrics[metric_name]
                if start_time <= point.timestamp <= end_time
            ]
            
    def get_current_value(self, metric_name: str) -> Optional[float]:
        """Get current value of metric."""
        with self.lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None
            return self.metrics[metric_name][-1].value
            
    def get_metric_stats(self, metric_name: str, 
                        duration_seconds: int = 3600) -> Dict[str, float]:
        """Get statistical summary of metric over duration."""
        end_time = time.time()
        start_time = end_time - duration_seconds
        
        history = self.get_metric_history(metric_name, start_time, end_time)
        
        if not history:
            return {}
            
        values = [point.value for point in history]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'current': values[-1] if values else 0.0
        }
        
    def get_all_metrics(self) -> List[str]:
        """Get list of all metric names."""
        with self.lock:
            return list(self.metrics.keys())


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.alerts = {}
        self.alert_rules = {}
        self.notification_channels = {}
        self.alert_history = deque(maxlen=10000)
        self.suppression_rules = []
        
    def add_alert_rule(self, rule_name: str, metric_name: str, 
                      condition: str, threshold: float,
                      severity: AlertSeverity = AlertSeverity.WARNING,
                      duration: int = 60) -> None:
        """Add alert rule for metric."""
        self.alert_rules[rule_name] = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq', etc.
            'threshold': threshold,
            'severity': severity,
            'duration': duration,  # seconds
            'last_check': 0,
            'violation_start': None
        }
        
        logger.info(f"Added alert rule: {rule_name}")
        
    def check_alerts(self, metrics_collector: MetricsCollector) -> List[Alert]:
        """Check all alert rules against current metrics."""
        new_alerts = []
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                current_value = metrics_collector.get_current_value(rule['metric_name'])
                
                if current_value is None:
                    continue
                    
                # Check condition
                condition_met = self._evaluate_condition(
                    current_value, rule['condition'], rule['threshold']
                )
                
                if condition_met:
                    if rule['violation_start'] is None:
                        rule['violation_start'] = current_time
                    elif current_time - rule['violation_start'] >= rule['duration']:
                        # Duration exceeded, trigger alert
                        alert = self._create_alert(rule_name, rule, current_value)
                        new_alerts.append(alert)
                        rule['violation_start'] = None  # Reset
                else:
                    # Condition not met, reset violation start
                    rule['violation_start'] = None
                    
                    # Check if we should resolve existing alert
                    existing_alert = self._find_active_alert(rule_name)
                    if existing_alert:
                        self._resolve_alert(existing_alert.alert_id)
                        
                rule['last_check'] = current_time
                
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
                
        return new_alerts
        
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'eq':
            return abs(value - threshold) < 1e-6
        elif condition == 'ne':
            return abs(value - threshold) >= 1e-6
        elif condition == 'gte':
            return value >= threshold
        elif condition == 'lte':
            return value <= threshold
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
            
    def _create_alert(self, rule_name: str, rule: Dict[str, Any], 
                     current_value: float) -> Alert:
        """Create new alert."""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            name=rule_name,
            description=f"Metric {rule['metric_name']} {rule['condition']} {rule['threshold']}",
            severity=rule['severity'],
            state=AlertState.ACTIVE,
            metric_name=rule['metric_name'],
            threshold_value=rule['threshold'],
            current_value=current_value
        )
        
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
        return alert
        
    def _find_active_alert(self, rule_name: str) -> Optional[Alert]:
        """Find active alert for rule."""
        for alert in self.alerts.values():
            if alert.name == rule_name and alert.state == AlertState.ACTIVE:
                return alert
        return None
        
    def _resolve_alert(self, alert_id: str) -> None:
        """Resolve alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.state = AlertState.RESOLVED
            alert.resolved_time = time.time()
            alert.updated_time = time.time()
            
            logger.info(f"Alert resolved: {alert.name}")
            
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge alert."""
        if alert_id not in self.alerts:
            return False
            
        alert = self.alerts[alert_id]
        alert.state = AlertState.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_time = time.time()
        alert.updated_time = time.time()
        
        logger.info(f"Alert acknowledged by {acknowledged_by}: {alert.name}")
        return True
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [
            alert for alert in self.alerts.values()
            if alert.state == AlertState.ACTIVE
        ]
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        active_alerts = self.get_active_alerts()
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
            
        return {
            'total_active': len(active_alerts),
            'by_severity': dict(severity_counts),
            'total_historical': len(self.alert_history),
            'acknowledged_count': len([a for a in active_alerts if a.state == AlertState.ACKNOWLEDGED])
        }


class PerformanceTracker:
    """Track system performance metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.tracking_thread = None
        self.running = False
        
    def start_tracking(self) -> None:
        """Start performance tracking."""
        if self.running:
            return
            
        self.running = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        logger.info("Performance tracking started")
        
    def stop_tracking(self) -> None:
        """Stop performance tracking."""
        self.running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5.0)
        logger.info("Performance tracking stopped")
        
    def _tracking_loop(self) -> None:
        """Main tracking loop."""
        while self.running:
            try:
                current_time = time.time()
                
                # System metrics
                self._collect_system_metrics(current_time)
                
                # Application metrics
                self._collect_application_metrics(current_time)
                
                # FPGA metrics (simulated)
                self._collect_fpga_metrics(current_time)
                
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error(f"Performance tracking error: {e}")
                time.sleep(5)
                
    def _collect_system_metrics(self, timestamp: float) -> None:
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_metric(MetricPoint(
                name="system.cpu.usage_percent",
                value=cpu_percent,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                tags={"host": socket.gethostname()}
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric(MetricPoint(
                name="system.memory.usage_percent",
                value=memory.percent,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                tags={"host": socket.gethostname()}
            ))
            
            self.metrics_collector.record_metric(MetricPoint(
                name="system.memory.available_mb",
                value=memory.available / (1024 * 1024),
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                tags={"host": socket.gethostname()}
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics_collector.record_metric(MetricPoint(
                name="system.disk.usage_percent",
                value=(disk.used / disk.total) * 100,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                tags={"host": socket.gethostname(), "mount": "/"}
            ))
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
    def _collect_application_metrics(self, timestamp: float) -> None:
        """Collect application-specific metrics."""
        # Simulated application metrics
        
        # Task processing rate
        processing_rate = np.random.normal(50, 10)  # Simulated
        self.metrics_collector.record_metric(MetricPoint(
            name="app.tasks.processing_rate",
            value=max(0, processing_rate),
            timestamp=timestamp,
            metric_type=MetricType.GAUGE,
            tags={"component": "orchestrator"}
        ))
        
        # Error rate
        error_rate = max(0, np.random.normal(0.05, 0.02))  # Simulated ~5% error rate
        self.metrics_collector.record_metric(MetricPoint(
            name="app.tasks.error_rate",
            value=error_rate,
            timestamp=timestamp,
            metric_type=MetricType.GAUGE,
            tags={"component": "orchestrator"}
        ))
        
        # Queue depth
        queue_depth = max(0, int(np.random.normal(10, 5)))  # Simulated
        self.metrics_collector.record_metric(MetricPoint(
            name="app.tasks.queue_depth",
            value=queue_depth,
            timestamp=timestamp,
            metric_type=MetricType.GAUGE,
            tags={"component": "orchestrator"}
        ))
        
    def _collect_fpga_metrics(self, timestamp: float) -> None:
        """Collect FPGA-specific metrics."""
        # Simulated FPGA metrics for different devices
        
        fpga_devices = ["fpga_001", "fpga_002", "fpga_003"]
        
        for device_id in fpga_devices:
            # Utilization
            utilization = max(0, min(100, np.random.normal(70, 15)))  # Simulated
            self.metrics_collector.record_metric(MetricPoint(
                name="fpga.utilization.percent",
                value=utilization,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                tags={"device": device_id}
            ))
            
            # Temperature
            temperature = max(20, np.random.normal(65, 10))  # Simulated temperature
            self.metrics_collector.record_metric(MetricPoint(
                name="fpga.temperature.celsius",
                value=temperature,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                tags={"device": device_id}
            ))
            
            # Power consumption
            power = max(5, np.random.normal(15, 3))  # Simulated watts
            self.metrics_collector.record_metric(MetricPoint(
                name="fpga.power.watts",
                value=power,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                tags={"device": device_id}
            ))
            
            # Processing throughput
            throughput = max(0, np.random.normal(1000, 200))  # Simulated ops/sec
            self.metrics_collector.record_metric(MetricPoint(
                name="fpga.throughput.ops_per_second",
                value=throughput,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                tags={"device": device_id}
            ))


class DashboardRenderer:
    """Renders monitoring dashboards."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.dashboards = {}
        
    def create_dashboard(self, config: DashboardConfig) -> str:
        """Create new dashboard."""
        self.dashboards[config.dashboard_id] = config
        logger.info(f"Created dashboard: {config.name}")
        return config.dashboard_id
        
    def render_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Render dashboard data."""
        if dashboard_id not in self.dashboards:
            return {'error': f'Dashboard {dashboard_id} not found'}
            
        config = self.dashboards[dashboard_id]
        
        # Calculate time range
        end_time = time.time()
        if config.time_range == "1h":
            start_time = end_time - 3600
        elif config.time_range == "6h":
            start_time = end_time - 6 * 3600
        elif config.time_range == "24h":
            start_time = end_time - 24 * 3600
        else:
            start_time = end_time - 3600  # Default 1 hour
            
        dashboard_data = {
            'dashboard_id': dashboard_id,
            'name': config.name,
            'description': config.description,
            'time_range': {
                'start': start_time,
                'end': end_time,
                'range': config.time_range
            },
            'panels': []
        }
        
        # Render each panel
        for panel_config in config.panels:
            panel_data = self._render_panel(panel_config, start_time, end_time)
            dashboard_data['panels'].append(panel_data)
            
        return dashboard_data
        
    def _render_panel(self, panel_config: Dict[str, Any], 
                     start_time: float, end_time: float) -> Dict[str, Any]:
        """Render individual dashboard panel."""
        panel_type = panel_config.get('type', 'line_chart')
        metric_name = panel_config.get('metric', '')
        
        if not metric_name:
            return {'error': 'No metric specified for panel'}
            
        # Get metric data
        history = self.metrics_collector.get_metric_history(metric_name, start_time, end_time)
        
        panel_data = {
            'title': panel_config.get('title', metric_name),
            'type': panel_type,
            'metric': metric_name,
            'data_points': len(history)
        }
        
        if panel_type == 'line_chart':
            panel_data['data'] = [
                {'timestamp': point.timestamp, 'value': point.value}
                for point in history
            ]
        elif panel_type == 'single_stat':
            current_value = self.metrics_collector.get_current_value(metric_name)
            stats = self.metrics_collector.get_metric_stats(metric_name, int(end_time - start_time))
            
            panel_data['current_value'] = current_value
            panel_data['stats'] = stats
        elif panel_type == 'table':
            # Group by tags and show latest values
            tag_groups = defaultdict(list)
            for point in history[-50:]:  # Last 50 points
                tag_key = str(sorted(point.tags.items()))
                tag_groups[tag_key].append(point)
                
            table_rows = []
            for tag_key, points in tag_groups.items():
                latest_point = max(points, key=lambda p: p.timestamp)
                table_rows.append({
                    'tags': latest_point.tags,
                    'value': latest_point.value,
                    'timestamp': latest_point.timestamp
                })
                
            panel_data['rows'] = table_rows
            
        return panel_data
        
    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards."""
        return [
            {
                'dashboard_id': config.dashboard_id,
                'name': config.name,
                'description': config.description,
                'panels': len(config.panels),
                'created_by': config.created_by,
                'created_time': config.created_time
            }
            for config in self.dashboards.values()
        ]


class AnomalyDetector:
    """Detect anomalies in metric data."""
    
    def __init__(self):
        self.baselines = {}
        self.detection_models = {}
        
    def learn_baseline(self, metric_name: str, 
                      history: List[MetricPoint],
                      window_size: int = 100) -> None:
        """Learn baseline behavior for metric."""
        if len(history) < window_size:
            logger.warning(f"Insufficient data for baseline learning: {metric_name}")
            return
            
        values = [point.value for point in history[-window_size:]]
        
        self.baselines[metric_name] = {
            'mean': statistics.mean(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'percentile_95': np.percentile(values, 95),
            'percentile_5': np.percentile(values, 5),
            'sample_size': len(values),
            'last_updated': time.time()
        }
        
        logger.info(f"Learned baseline for {metric_name}")
        
    def detect_anomalies(self, metric_name: str, 
                        current_value: float,
                        threshold_std_devs: float = 3.0) -> Dict[str, Any]:
        """Detect if current value is anomalous."""
        if metric_name not in self.baselines:
            return {'is_anomaly': False, 'reason': 'No baseline available'}
            
        baseline = self.baselines[metric_name]
        
        # Z-score based detection
        if baseline['std_dev'] > 0:
            z_score = abs(current_value - baseline['mean']) / baseline['std_dev']
            is_anomaly_zscore = z_score > threshold_std_devs
        else:
            z_score = 0.0
            is_anomaly_zscore = False
            
        # Percentile-based detection
        is_anomaly_percentile = (
            current_value < baseline['percentile_5'] or 
            current_value > baseline['percentile_95']
        )
        
        is_anomaly = is_anomaly_zscore or is_anomaly_percentile
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'current_value': current_value,
            'baseline_mean': baseline['mean'],
            'baseline_std_dev': baseline['std_dev'],
            'percentile_5': baseline['percentile_5'],
            'percentile_95': baseline['percentile_95'],
            'anomaly_type': 'statistical' if is_anomaly_zscore else 'percentile' if is_anomaly_percentile else None
        }
        
    def update_baselines(self, metrics_collector: MetricsCollector) -> None:
        """Update baselines for all tracked metrics."""
        for metric_name in metrics_collector.get_all_metrics():
            # Get recent history
            end_time = time.time()
            start_time = end_time - 24 * 3600  # 24 hours
            history = metrics_collector.get_metric_history(metric_name, start_time, end_time)
            
            if len(history) >= 100:  # Minimum samples for baseline
                self.learn_baseline(metric_name, history)


class MonitoringDashboard:
    """Main monitoring dashboard system."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self.dashboard_renderer = DashboardRenderer(self.metrics_collector)
        self.anomaly_detector = AnomalyDetector()
        
        self.monitoring_thread = None
        self.running = False
        
        # Setup default dashboards
        self._create_default_dashboards()
        
        # Setup default alert rules
        self._create_default_alert_rules()
        
    def start_monitoring(self) -> None:
        """Start monitoring system."""
        if self.running:
            logger.warning("Monitoring already running")
            return
            
        self.running = True
        
        # Start performance tracking
        self.performance_tracker.start_tracking()
        
        # Start monitoring loop
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Monitoring dashboard started")
        
    def stop_monitoring(self) -> None:
        """Stop monitoring system."""
        if not self.running:
            return
            
        self.running = False
        
        # Stop performance tracking
        self.performance_tracker.stop_tracking()
        
        # Wait for monitoring thread
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("Monitoring dashboard stopped")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Check alerts
                new_alerts = self.alert_manager.check_alerts(self.metrics_collector)
                
                # Update anomaly baselines periodically
                if int(time.time()) % 3600 == 0:  # Every hour
                    self.anomaly_detector.update_baselines(self.metrics_collector)
                    
                # Check for anomalies in key metrics
                self._check_anomalies()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
                
    def _check_anomalies(self) -> None:
        """Check for anomalies in key metrics."""
        key_metrics = [
            'system.cpu.usage_percent',
            'system.memory.usage_percent',
            'app.tasks.error_rate',
            'fpga.temperature.celsius'
        ]
        
        for metric_name in key_metrics:
            current_value = self.metrics_collector.get_current_value(metric_name)
            if current_value is not None:
                anomaly_result = self.anomaly_detector.detect_anomalies(metric_name, current_value)
                
                if anomaly_result['is_anomaly']:
                    logger.warning(f"Anomaly detected in {metric_name}: {anomaly_result}")
                    
    def _create_default_dashboards(self) -> None:
        """Create default monitoring dashboards."""
        # System Overview Dashboard
        system_dashboard = DashboardConfig(
            dashboard_id="system_overview",
            name="System Overview",
            description="Overall system health and performance",
            panels=[
                {
                    'title': 'CPU Usage',
                    'type': 'line_chart',
                    'metric': 'system.cpu.usage_percent'
                },
                {
                    'title': 'Memory Usage',
                    'type': 'line_chart',
                    'metric': 'system.memory.usage_percent'
                },
                {
                    'title': 'Current CPU',
                    'type': 'single_stat',
                    'metric': 'system.cpu.usage_percent'
                },
                {
                    'title': 'Task Processing Rate',
                    'type': 'line_chart',
                    'metric': 'app.tasks.processing_rate'
                }
            ]
        )
        self.dashboard_renderer.create_dashboard(system_dashboard)
        
        # FPGA Dashboard
        fpga_dashboard = DashboardConfig(
            dashboard_id="fpga_monitoring",
            name="FPGA Monitoring",
            description="FPGA device status and performance",
            panels=[
                {
                    'title': 'FPGA Utilization',
                    'type': 'table',
                    'metric': 'fpga.utilization.percent'
                },
                {
                    'title': 'FPGA Temperature',
                    'type': 'line_chart',
                    'metric': 'fpga.temperature.celsius'
                },
                {
                    'title': 'FPGA Power',
                    'type': 'line_chart',
                    'metric': 'fpga.power.watts'
                },
                {
                    'title': 'FPGA Throughput',
                    'type': 'line_chart',
                    'metric': 'fpga.throughput.ops_per_second'
                }
            ]
        )
        self.dashboard_renderer.create_dashboard(fpga_dashboard)
        
        # Application Dashboard
        app_dashboard = DashboardConfig(
            dashboard_id="application_metrics",
            name="Application Metrics",
            description="Application-specific monitoring",
            panels=[
                {
                    'title': 'Error Rate',
                    'type': 'line_chart',
                    'metric': 'app.tasks.error_rate'
                },
                {
                    'title': 'Queue Depth',
                    'type': 'single_stat',
                    'metric': 'app.tasks.queue_depth'
                },
                {
                    'title': 'Processing Rate',
                    'type': 'single_stat',
                    'metric': 'app.tasks.processing_rate'
                }
            ]
        )
        self.dashboard_renderer.create_dashboard(app_dashboard)
        
    def _create_default_alert_rules(self) -> None:
        """Create default alert rules."""
        # System alerts
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            "system.cpu.usage_percent",
            "gt", 85.0,
            AlertSeverity.WARNING,
            duration=300  # 5 minutes
        )
        
        self.alert_manager.add_alert_rule(
            "critical_cpu_usage",
            "system.cpu.usage_percent", 
            "gt", 95.0,
            AlertSeverity.CRITICAL,
            duration=60  # 1 minute
        )
        
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            "system.memory.usage_percent",
            "gt", 90.0,
            AlertSeverity.WARNING,
            duration=300
        )
        
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            "app.tasks.error_rate",
            "gt", 0.1,  # 10%
            AlertSeverity.CRITICAL,
            duration=120
        )
        
        # FPGA alerts  
        self.alert_manager.add_alert_rule(
            "fpga_overheating",
            "fpga.temperature.celsius",
            "gt", 85.0,
            AlertSeverity.CRITICAL,
            duration=60
        )
        
        self.alert_manager.add_alert_rule(
            "fpga_high_power",
            "fpga.power.watts", 
            "gt", 25.0,
            AlertSeverity.WARNING,
            duration=300
        )
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Get key metrics
        key_metrics = {}
        for metric in ['system.cpu.usage_percent', 'system.memory.usage_percent', 
                      'app.tasks.processing_rate', 'app.tasks.error_rate']:
            current_value = self.metrics_collector.get_current_value(metric)
            if current_value is not None:
                key_metrics[metric] = current_value
                
        return {
            'monitoring_active': self.running,
            'alerts': alert_summary,
            'key_metrics': key_metrics,
            'total_metrics_tracked': len(self.metrics_collector.get_all_metrics()),
            'dashboards_available': len(self.dashboard_renderer.get_dashboard_list()),
            'timestamp': time.time()
        }
        
    def get_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data."""
        return self.dashboard_renderer.render_dashboard(dashboard_id)
        
    def get_alerts(self, active_only: bool = True) -> List[Alert]:
        """Get system alerts."""
        if active_only:
            return self.alert_manager.get_active_alerts()
        else:
            return list(self.alert_manager.alerts.values())
            
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        return self.alert_manager.acknowledge_alert(alert_id, user)


# Convenience functions

def create_monitoring_dashboard() -> MonitoringDashboard:
    """Create monitoring dashboard instance."""
    return MonitoringDashboard()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create monitoring dashboard
    monitor = create_monitoring_dashboard()
    monitor.start_monitoring()
    
    print("Monitoring dashboard started. Collecting metrics...")
    
    # Let it run for a bit
    time.sleep(30)
    
    # Get system status
    status = monitor.get_system_status()
    print("\nSystem Status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Get dashboard data
    system_dashboard = monitor.get_dashboard("system_overview")
    print("\nSystem Dashboard:")
    print(json.dumps(system_dashboard, indent=2, default=str))
    
    # Get alerts
    alerts = monitor.get_alerts()
    print(f"\nActive Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"- {alert.name}: {alert.description} (Severity: {alert.severity.value})")
        
    # Stop monitoring
    monitor.stop_monitoring()