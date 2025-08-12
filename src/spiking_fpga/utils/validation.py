"""Enhanced validation utilities with comprehensive security and error handling."""

import re
import logging
import hashlib
import json
import yaml
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from enum import Enum
from collections import defaultdict
import os
import stat
from datetime import datetime, timedelta

from ..models.network import Network
from ..core import FPGATarget
from ..models.optimization import OptimizationLevel, ResourceEstimate


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class SecurityThreat(BaseModel):
    """Security threat detection result."""
    
    threat_type: str = Field(..., description="Type of security threat detected")
    severity: str = Field(..., description="Threat severity: low, medium, high, critical")
    location: str = Field(..., description="Location where threat was detected")
    description: str = Field(..., description="Human-readable threat description")
    mitigation: str = Field(..., description="Suggested mitigation")
    timestamp: float = Field(default_factory=time.time)
    
    class Config:
        extra = "forbid"


class SecurityConfig(BaseModel):
    """Enhanced security configuration for validation."""
    
    max_file_size_mb: int = Field(100, description="Maximum allowed file size in MB")
    allowed_file_extensions: List[str] = Field([".yaml", ".yml", ".json"], description="Allowed input file extensions")
    max_neurons: int = Field(1_000_000, description="Maximum number of neurons")
    max_synapses: int = Field(10_000_000, description="Maximum number of synapses")
    max_layers: int = Field(1000, description="Maximum number of layers")
    sanitize_network_names: bool = Field(True, description="Sanitize network names")
    validate_hdl_injection: bool = Field(True, description="Check for HDL injection attacks")
    
    # Enhanced security features
    enable_rate_limiting: bool = Field(True, description="Enable rate limiting for validation requests")
    max_requests_per_minute: int = Field(100, description="Maximum validation requests per minute")
    block_suspicious_patterns: bool = Field(True, description="Block files with suspicious patterns")
    quarantine_threats: bool = Field(True, description="Quarantine files with detected threats")
    audit_log_enabled: bool = Field(True, description="Enable security audit logging")
    require_file_integrity: bool = Field(True, description="Require file integrity verification")
    
    class Config:
        extra = "forbid"


@dataclass 
class ValidationResult:
    """Enhanced validation result with security threat detection."""
    
    valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    security_threats: List[SecurityThreat] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_time_ms: float = 0.0
    
    @property
    def has_critical_threats(self) -> bool:
        """Check if any critical security threats were found."""
        return any(threat.severity == "critical" for threat in self.security_threats)
    
    @property
    def has_high_threats(self) -> bool:
        """Check if any high-severity security threats were found."""
        return any(threat.severity in ["high", "critical"] for threat in self.security_threats)
    
    def add_issue(self, message: str):
        """Add a validation issue."""
        self.issues.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        """Add a validation warning."""
        self.warnings.append(message)
    
    def add_recommendation(self, message: str):
        """Add a recommendation."""
        self.recommendations.append(message)
    
    def add_security_threat(self, threat_type: str, severity: str, location: str, 
                           description: str, mitigation: str):
        """Add a security threat."""
        threat = SecurityThreat(
            threat_type=threat_type,
            severity=severity,
            location=location,
            description=description,
            mitigation=mitigation
        )
        self.security_threats.append(threat)
        
        if severity in ["high", "critical"]:
            self.valid = False


class SecurityValidator:
    """Advanced security validation with threat detection."""
    
    def __init__(self, config: SecurityConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.request_tracker = defaultdict(list)  # Track requests per IP/user
        self.quarantine_dir = Path("/tmp/spiking_fpga_quarantine")
        self.quarantine_dir.mkdir(exist_ok=True)
        
        # Suspicious patterns to detect
        self.hdl_injection_patterns = [
            r'\$[a-zA-Z]+\s*\(',  # SystemVerilog system tasks
            r'initial\s+begin',   # Verilog initial blocks
            r'always\s*@',        # Verilog always blocks
            r'module\s+\w+',      # Module definitions
            r'\`[a-zA-Z_]+',      # Compiler directives
            r'\bfork\b.*\bjoin\b',  # Fork-join blocks
        ]
        
        self.malicious_content_patterns = [
            r'<script[^>]*>.*?</script>',  # JavaScript injection
            r'exec\s*\(',                  # Code execution
            r'eval\s*\(',                  # Eval calls
            r'import\s+os',                # OS imports
            r'subprocess\.',               # Subprocess calls
            r'__import__',                 # Dynamic imports
        ]
    
    def validate_file_security(self, file_path: Path, content: str = None) -> ValidationResult:
        """Comprehensive file security validation."""
        start_time = time.time()
        result = ValidationResult(valid=True)
        
        try:
            # Rate limiting check
            if self.config.enable_rate_limiting:
                if not self._check_rate_limit(str(file_path)):
                    result.add_security_threat(
                        "rate_limit_exceeded", "high", str(file_path),
                        "Too many validation requests from this source",
                        "Wait before retrying or contact administrator"
                    )
                    return result
            
            # File existence and permissions
            self._validate_file_permissions(file_path, result)
            
            # File size validation
            file_size = file_path.stat().st_size if file_path.exists() else 0
            max_size = self.config.max_file_size_mb * 1024 * 1024
            if file_size > max_size:
                result.add_security_threat(
                    "file_too_large", "medium", str(file_path),
                    f"File size {file_size} exceeds limit {max_size}",
                    "Reduce file size or increase limit"
                )
            
            # Content validation if provided
            if content is None and file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    result.add_security_threat(
                        "file_read_error", "high", str(file_path),
                        f"Cannot read file: {str(e)}",
                        "Check file permissions and encoding"
                    )
                    return result
            
            if content:
                # HDL injection detection
                if self.config.validate_hdl_injection:
                    self._detect_hdl_injection(content, str(file_path), result)
                
                # Malicious content detection
                if self.config.block_suspicious_patterns:
                    self._detect_malicious_content(content, str(file_path), result)
                
                # File integrity verification
                if self.config.require_file_integrity:
                    self._verify_file_integrity(file_path, content, result)
            
            # Quarantine threats if configured
            if self.config.quarantine_threats and result.has_high_threats:
                self._quarantine_file(file_path, result)
            
        except Exception as e:
            self.logger.error(f"Security validation error: {str(e)}")
            result.add_security_threat(
                "validation_error", "critical", str(file_path),
                f"Validation failed with error: {str(e)}",
                "Contact system administrator"
            )
        
        finally:
            result.validation_time_ms = (time.time() - start_time) * 1000
            
            # Audit logging
            if self.config.audit_log_enabled:
                self._audit_log(file_path, result)
        
        return result
    
    def _check_rate_limit(self, identifier: str) -> bool:
        """Check rate limiting for validation requests."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old entries
        self.request_tracker[identifier] = [
            timestamp for timestamp in self.request_tracker[identifier]
            if timestamp > minute_ago
        ]
        
        # Check limit
        if len(self.request_tracker[identifier]) >= self.config.max_requests_per_minute:
            return False
        
        # Add current request
        self.request_tracker[identifier].append(now)
        return True
    
    def _validate_file_permissions(self, file_path: Path, result: ValidationResult):
        """Validate file permissions for security."""
        if not file_path.exists():
            return
        
        stat_info = file_path.stat()
        
        # Check for world-writable files
        if stat_info.st_mode & stat.S_IWOTH:
            result.add_security_threat(
                "world_writable", "medium", str(file_path),
                "File is world-writable, potential security risk",
                "Remove world write permissions"
            )
        
        # Check for setuid/setgid bits
        if stat_info.st_mode & (stat.S_ISUID | stat.S_ISGID):
            result.add_security_threat(
                "privileged_file", "high", str(file_path),
                "File has setuid/setgid bits set",
                "Remove privileged bits if not required"
            )
    
    def _detect_hdl_injection(self, content: str, location: str, result: ValidationResult):
        """Detect potential HDL injection attacks."""
        for pattern in self.hdl_injection_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                result.add_security_threat(
                    "hdl_injection", "high", location,
                    f"Potential HDL injection detected: pattern '{pattern}'",
                    "Review file content and remove suspicious HDL code"
                )
    
    def _detect_malicious_content(self, content: str, location: str, result: ValidationResult):
        """Detect potentially malicious content patterns."""
        for pattern in self.malicious_content_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                result.add_security_threat(
                    "malicious_content", "critical", location,
                    f"Malicious content detected: {matches[0][:50]}...",
                    "Remove malicious code and review file source"
                )
    
    def _verify_file_integrity(self, file_path: Path, content: str, result: ValidationResult):
        """Verify file integrity using checksums."""
        try:
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            result.performance_metrics['content_hash'] = content_hash
            
            # Store or verify against known good hashes (simplified implementation)
            hash_file = file_path.with_suffix(file_path.suffix + '.sha256')
            if hash_file.exists():
                with open(hash_file, 'r') as f:
                    expected_hash = f.read().strip()
                if content_hash != expected_hash:
                    result.add_security_threat(
                        "integrity_violation", "high", str(file_path),
                        "File content hash doesn't match expected value",
                        "Verify file source and regenerate hash"
                    )
            
        except Exception as e:
            self.logger.warning(f"Integrity verification failed: {str(e)}")
    
    def _quarantine_file(self, file_path: Path, result: ValidationResult):
        """Quarantine file with detected threats."""
        try:
            quarantine_file = self.quarantine_dir / f"{file_path.name}_{int(time.time())}"
            if file_path.exists():
                quarantine_file.write_text(file_path.read_text())
                self.logger.warning(f"File quarantined: {file_path} -> {quarantine_file}")
                
                # Write threat report
                threat_report = {
                    "original_path": str(file_path),
                    "quarantine_time": datetime.now().isoformat(),
                    "threats": [threat.dict() for threat in result.security_threats]
                }
                
                report_file = quarantine_file.with_suffix('.threat_report.json')
                with open(report_file, 'w') as f:
                    json.dump(threat_report, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Failed to quarantine file: {str(e)}")
    
    def _audit_log(self, file_path: Path, result: ValidationResult):
        """Log security validation results for audit."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "file_path": str(file_path),
            "validation_result": {
                "valid": result.valid,
                "threat_count": len(result.security_threats),
                "high_threats": result.has_high_threats,
                "critical_threats": result.has_critical_threats,
            },
            "validation_time_ms": result.validation_time_ms,
        }
        
        self.logger.info(f"Security validation audit: {json.dumps(audit_entry)}")


class NetworkValidator:
    """Enhanced network validation with security checks."""
    
    def __init__(self, security_config: SecurityConfig = None):
        self.min_neurons = 1
        self.max_neurons = 1_000_000
        self.min_synapses = 0
        self.max_synapses = 10_000_000
        self.max_fanout = 1000
        self.max_fanin = 1000
        
        # Security integration
        self.security_config = security_config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Override limits from security config
        if security_config:
            self.max_neurons = min(self.max_neurons, security_config.max_neurons)
            self.max_synapses = min(self.max_synapses, security_config.max_synapses)
    
    def validate_network(self, network: Network, validate_security: bool = True) -> ValidationResult:
        """Perform comprehensive network validation with security checks."""
        start_time = time.time()
        result = ValidationResult(valid=True)
        
        try:
            # Basic structure validation
            self._validate_structure(network, result)
            
            # Connectivity validation
            self._validate_connectivity(network, result)
            
            # Neuron parameters validation with security checks
            self._validate_neuron_parameters(network, result)
            
            # Security validation if enabled
            if validate_security:
                self._validate_network_security(network, result)
            
            # Performance recommendations
            self._generate_recommendations(network, result)
            
        except Exception as e:
            self.logger.error(f"Network validation error: {str(e)}")
            result.add_issue(f"Validation failed: {str(e)}")
            result.add_security_threat(
                "validation_error", "high", "network_validation",
                f"Unexpected error during validation: {str(e)}",
                "Contact system administrator"
            )
        
        finally:
            result.validation_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _validate_structure(self, network: Network, result: ValidationResult):
        """Validate basic network structure."""
        # Check neuron count
        neuron_count = len(network.neurons)
        if neuron_count < self.min_neurons:
            result.add_issue(f"Network has too few neurons: {neuron_count} < {self.min_neurons}")
        elif neuron_count > self.max_neurons:
            result.add_issue(f"Network has too many neurons: {neuron_count} > {self.max_neurons}")
        
        # Check synapse count
        synapse_count = len(network.synapses)
        if synapse_count > self.max_synapses:
            result.add_issue(f"Network has too many synapses: {synapse_count} > {self.max_synapses}")
        
        # Check layer structure
        if not network.layers:
            result.add_issue("Network has no layers defined")
        
        layer_sizes = [layer.size for layer in network.layers]
        if any(size <= 0 for size in layer_sizes):
            result.add_issue("All layers must have positive size")
        
        # Check for empty layers
        empty_layers = [i for i, layer in enumerate(network.layers) if layer.size == 0]
        if empty_layers:
            result.add_warning(f"Empty layers found: {empty_layers}")
    
    def _validate_connectivity(self, network: Network, result: ValidationResult):
        """Validate network connectivity."""
        if not network.synapses:
            result.add_warning("Network has no synaptic connections")
            return
        
        # Check for orphaned neurons
        connected_neurons = set()
        for synapse in network.synapses:
            connected_neurons.add(synapse.pre_neuron_id)
            connected_neurons.add(synapse.post_neuron_id)
        
        all_neuron_ids = {n.neuron_id for n in network.neurons}
        orphaned = all_neuron_ids - connected_neurons
        if orphaned:
            result.add_warning(f"Found {len(orphaned)} orphaned neurons with no connections")
        
        # Check fanout/fanin
        fanout_counts = {}
        fanin_counts = {}
        
        for synapse in network.synapses:
            fanout_counts[synapse.pre_neuron_id] = fanout_counts.get(synapse.pre_neuron_id, 0) + 1
            fanin_counts[synapse.post_neuron_id] = fanin_counts.get(synapse.post_neuron_id, 0) + 1
        
        # Check for excessive fanout
        high_fanout = [(nid, count) for nid, count in fanout_counts.items() if count > self.max_fanout]
        if high_fanout:
            result.add_warning(f"Neurons with high fanout (>{self.max_fanout}): {len(high_fanout)}")
        
        # Check for excessive fanin
        high_fanin = [(nid, count) for nid, count in fanin_counts.items() if count > self.max_fanin]
        if high_fanin:
            result.add_warning(f"Neurons with high fanin (>{self.max_fanin}): {len(high_fanin)}")
    
    def _validate_network_security(self, network: Network, result: ValidationResult):
        """Perform security validation on network structure."""
        # Check for suspiciously large networks (potential DoS)
        neuron_count = len(network.neurons)
        synapse_count = len(network.synapses)
        
        if neuron_count > self.security_config.max_neurons:
            result.add_security_threat(
                "resource_exhaustion", "high", "network_structure",
                f"Network has {neuron_count} neurons, exceeding security limit {self.security_config.max_neurons}",
                "Reduce network size or increase security limits"
            )
        
        if synapse_count > self.security_config.max_synapses:
            result.add_security_threat(
                "resource_exhaustion", "high", "network_structure", 
                f"Network has {synapse_count} synapses, exceeding security limit {self.security_config.max_synapses}",
                "Reduce network connectivity or increase security limits"
            )
        
        # Check for suspicious network names if sanitization is enabled
        if self.security_config.sanitize_network_names and network.name:
            if not re.match(r'^[a-zA-Z0-9_\-\.\s]+$', network.name):
                result.add_security_threat(
                    "name_injection", "medium", "network_name",
                    f"Network name contains potentially unsafe characters: {network.name}",
                    "Use only alphanumeric characters, underscores, hyphens, dots, and spaces"
                )
        
        # Detect potential algorithmic complexity attacks
        if synapse_count > 0 and neuron_count > 0:
            connectivity_ratio = synapse_count / (neuron_count * neuron_count)
            if connectivity_ratio > 0.8:  # Very dense network
                result.add_security_threat(
                    "algorithmic_complexity", "medium", "network_connectivity",
                    f"Network connectivity ratio {connectivity_ratio:.2f} is very high, may cause performance issues",
                    "Consider sparsifying the network or using more efficient algorithms"
                )
    
    def _validate_neuron_parameters(self, network: Network, result: ValidationResult):
        """Validate neuron model parameters."""
        for neuron in network.neurons:
            if neuron.neuron_type == "LIF":
                params = neuron.parameters
                if "v_thresh" in params and "v_reset" in params:
                    if params["v_thresh"] <= params["v_reset"]:
                        result.add_issue(f"Neuron {neuron.neuron_id}: threshold must be > reset voltage")
                
                if "tau_m" in params and params["tau_m"] <= 0:
                    result.add_issue(f"Neuron {neuron.neuron_id}: membrane time constant must be positive")
    
    def _generate_recommendations(self, network: Network, result: ValidationResult):
        """Generate performance recommendations."""
        neuron_count = len(network.neurons)
        synapse_count = len(network.synapses)
        
        if synapse_count > 0:
            avg_connectivity = synapse_count / (neuron_count ** 2) if neuron_count > 0 else 0
            
            if avg_connectivity > 0.5:
                result.add_recommendation("High connectivity detected - consider sparse connections for better performance")
            elif avg_connectivity < 0.01:
                result.add_recommendation("Very sparse connectivity - network may have limited expressiveness")
        
        # Check for unbalanced layer sizes
        if len(network.layers) > 1:
            layer_sizes = [layer.size for layer in network.layers]
            max_size = max(layer_sizes)
            min_size = min(layer_sizes)
            
            if max_size / min_size > 100:
                result.add_recommendation("Large variation in layer sizes - consider more balanced architecture")


class ConfigurationValidator:
    """Enhanced compilation configuration validation with security checks."""
    
    def __init__(self, security_config: SecurityConfig = None):
        self.security_config = security_config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
    
    def validate_fpga_target(self, target: FPGATarget, 
                           resource_estimate: ResourceEstimate) -> ValidationResult:
        """Validate FPGA target compatibility."""
        result = ValidationResult(valid=True, issues=[], warnings=[], recommendations=[])
        
        target_resources = target.resources
        utilization = resource_estimate.utilization_percentage(target_resources)
        
        # Check resource utilization
        logic_util = utilization.get("logic", 0)
        memory_util = utilization.get("memory", 0)
        dsp_util = utilization.get("dsp", 0)
        
        if logic_util > 100:
            result.add_issue(f"Logic utilization ({logic_util:.1f}%) exceeds target capacity")
        elif logic_util > 90:
            result.add_warning(f"High logic utilization ({logic_util:.1f}%) - synthesis may fail")
        elif logic_util > 80:
            result.add_recommendation(f"Consider optimization to reduce logic usage ({logic_util:.1f}%)")
        
        if memory_util > 100:
            result.add_issue(f"Memory utilization ({memory_util:.1f}%) exceeds target capacity")
        elif memory_util > 85:
            result.add_warning(f"High memory utilization ({memory_util:.1f}%) - consider pruning")
        
        if dsp_util > 100:
            result.add_issue(f"DSP utilization ({dsp_util:.1f}%) exceeds target capacity")
        elif dsp_util > 95:
            result.add_warning(f"High DSP utilization ({dsp_util:.1f}%) - consider simpler neuron models")
        
        return result
    
    def validate_optimization_config(self, level: OptimizationLevel, 
                                   custom_params: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate optimization configuration."""
        result = ValidationResult(valid=True, issues=[], warnings=[], recommendations=[])
        
        if custom_params:
            # Validate custom optimization parameters
            if "weight_threshold" in custom_params:
                threshold = custom_params["weight_threshold"]
                if not 0 < threshold < 1:
                    result.add_issue("Weight threshold must be between 0 and 1")
            
            if "cluster_size" in custom_params:
                cluster_size = custom_params["cluster_size"]
                if not isinstance(cluster_size, int) or cluster_size < 1:
                    result.add_issue("Cluster size must be a positive integer")
                elif cluster_size > 64:
                    result.add_warning("Large cluster sizes may reduce parallelism")
        
        return result


class FileValidator:
    """Enhanced file validation with security checks."""
    
    def __init__(self, security_config: SecurityConfig = None):
        self.supported_extensions = {".yaml", ".yml", ".json"}
        self.security_config = security_config or SecurityConfig()
        self.security_validator = SecurityValidator(self.security_config)
        self.logger = logging.getLogger(__name__)
        
        # Update supported extensions from security config
        if security_config and security_config.allowed_file_extensions:
            self.supported_extensions = set(security_config.allowed_file_extensions)
    
    def validate_network_file(self, file_path: Path, 
                             perform_security_validation: bool = True) -> ValidationResult:
        """Comprehensive network file validation with security checks."""
        start_time = time.time()
        result = ValidationResult(valid=True)
        
        try:
            # Security validation first
            if perform_security_validation:
                security_result = self.security_validator.validate_file_security(file_path)
                
                # Merge security results
                result.security_threats.extend(security_result.security_threats)
                result.performance_metrics.update(security_result.performance_metrics)
                
                if security_result.has_critical_threats:
                    result.valid = False
                    result.add_issue("Critical security threats detected - file validation aborted")
                    return result
                
                # Add security warnings as validation warnings
                for threat in security_result.security_threats:
                    if threat.severity in ["medium", "high"]:
                        result.add_warning(f"Security: {threat.description}")
            
            # Check file exists
            if not file_path.exists():
                result.add_issue(f"File does not exist: {file_path}")
                return result
        
            # Check file extension
            if file_path.suffix.lower() not in self.supported_extensions:
                result.add_issue(f"Unsupported file format: {file_path.suffix}")
                return result
        
            # Check file is readable
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
            except PermissionError:
                result.add_issue(f"Cannot read file: {file_path}")
                return result
            except Exception as e:
                result.add_issue(f"Error reading file: {str(e)}")
                return result
        
            # Validate file format
            try:
                if file_path.suffix.lower() in {".yaml", ".yml"}:
                    yaml.safe_load(content)
                elif file_path.suffix.lower() == ".json":
                    json.loads(content)
            except Exception as e:
                result.add_issue(f"Invalid file format: {str(e)}")
                return result
        
            # Check file size against security config
            file_size = len(content)
            max_size = self.security_config.max_file_size_mb * 1024 * 1024
            if file_size > max_size:
                result.add_issue(f"File too large: {file_size} bytes > {max_size} bytes")
                return result
            elif file_size > max_size * 0.8:  # Warning at 80% of limit
                result.add_warning(f"Large network file ({file_size} bytes) - parsing may be slow")
            
            # Additional content validation
            self._validate_file_content_structure(content, file_path, result)
            
        except Exception as e:
            self.logger.error(f"File validation error: {str(e)}")
            result.add_issue(f"File validation failed: {str(e)}")
        
        finally:
            result.validation_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _validate_file_content_structure(self, content: str, file_path: Path, result: ValidationResult):
        """Validate file content structure and detect anomalies."""
        lines = content.split('\n')
        
        # Check for excessively long lines (potential buffer overflow)
        max_line_length = 10000
        for i, line in enumerate(lines, 1):
            if len(line) > max_line_length:
                result.add_warning(f"Line {i} is very long ({len(line)} chars) - may cause parsing issues")
        
        # Check for excessive nesting in YAML/JSON
        if file_path.suffix.lower() in {".yaml", ".yml"}:
            max_indent = 0
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    max_indent = max(max_indent, indent)
            
            if max_indent > 20:  # Very deep nesting
                result.add_warning(f"Deep nesting detected (max indent: {max_indent}) - may indicate complex or malicious structure")
        
        # Check for binary content in text files
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            result.add_issue("File contains non-UTF-8 content - potential binary data in text file")
        
        # Check for null bytes (potential binary injection)
        if '\x00' in content:
            result.add_security_threat(
                "binary_injection", "high", str(file_path),
                "File contains null bytes - potential binary injection",
                "Remove null bytes or verify file is not corrupted"
            )


def validate_identifier(name: str) -> bool:
    """Validate that a name is a valid identifier."""
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def sanitize_filename(filename: str) -> str:
    """Enhanced filename sanitization for filesystem safety."""
    # Remove/replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing dots, spaces, and hyphens
    sanitized = sanitized.strip('. -')
    
    # Remove consecutive underscores/dots
    sanitized = re.sub(r'[_.]{2,}', '_', sanitized)
    
    # Prevent reserved names on Windows
    reserved_names = {'CON', 'PRN', 'AUX', 'NUL'} | {f'COM{i}' for i in range(1, 10)} | {f'LPT{i}' for i in range(1, 10)}
    if sanitized.upper().split('.')[0] in reserved_names:
        sanitized = f"safe_{sanitized}"
    
    # Limit length and ensure not empty
    if len(sanitized) > 200:  # Conservative limit for cross-platform compatibility
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:200-len(ext)] + ext
    
    return sanitized or "unnamed_file"


def sanitize_identifier(identifier: str) -> str:
    """Sanitize identifier for code generation safety."""
    # Only allow alphanumeric and underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', identifier)
    
    # Ensure it starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = f"id_{sanitized}"
    
    # Limit length
    if len(sanitized) > 64:
        sanitized = sanitized[:64]
    
    return sanitized or "default_id"


class RateLimiter:
    """Thread-safe rate limiter for validation operations."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Clean old requests
            self.requests[identifier] = [
                timestamp for timestamp in self.requests[identifier]
                if timestamp > window_start
            ]
            
            # Check limit
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            
            # Add current request
            self.requests[identifier].append(now)
            return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get number of remaining requests for identifier."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Clean old requests
            self.requests[identifier] = [
                timestamp for timestamp in self.requests[identifier]
                if timestamp > window_start
            ]
            
            return max(0, self.max_requests - len(self.requests[identifier]))


class ValidationEngine:
    """Centralized validation engine with comprehensive security and performance monitoring."""
    
    def __init__(self, security_config: SecurityConfig = None):
        self.security_config = security_config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        self.security_validator = SecurityValidator(self.security_config, self.logger)
        self.network_validator = NetworkValidator(self.security_config)
        self.file_validator = FileValidator(self.security_config)
        self.config_validator = ConfigurationValidator()
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_requests=self.security_config.max_requests_per_minute,
            window_seconds=60
        )
        
        # Metrics tracking
        self.validation_metrics = {
            "total_validations": 0,
            "successful_validations": 0,
            "security_threats_detected": 0,
            "files_quarantined": 0,
            "rate_limited_requests": 0,
        }
    
    def validate_complete_pipeline(self, network_file: Path, target: FPGATarget, 
                                 network: Optional[Network] = None,
                                 client_id: str = "unknown") -> ValidationResult:
        """Perform complete validation pipeline with comprehensive security checks."""
        start_time = time.time()
        
        # Rate limiting check
        if self.security_config.enable_rate_limiting:
            if not self.rate_limiter.is_allowed(client_id):
                result = ValidationResult(valid=False)
                result.add_security_threat(
                    "rate_limit_exceeded", "medium", "validation_request",
                    f"Rate limit exceeded for client {client_id}",
                    "Reduce request frequency or contact administrator"
                )
                self.validation_metrics["rate_limited_requests"] += 1
                return result
        
        # Comprehensive validation result
        combined_result = ValidationResult(valid=True)
        
        try:
            self.validation_metrics["total_validations"] += 1
            
            # 1. File validation with security checks
            file_result = self.file_validator.validate_network_file(network_file)
            self._merge_results(combined_result, file_result)
            
            if combined_result.has_critical_threats:
                self.logger.warning(f"Critical security threats in file validation: {network_file}")
                return combined_result
            
            # 2. Network validation if network object provided
            if network:
                network_result = self.network_validator.validate_network(network)
                self._merge_results(combined_result, network_result)
            
            # 3. FPGA target validation (if network available for resource estimation)
            if network:
                from ..network_compiler import NetworkCompiler
                compiler = NetworkCompiler(target)
                resource_estimate = compiler.estimate_resources(network)
                
                target_result = self.config_validator.validate_fpga_target(target, resource_estimate)
                self._merge_results(combined_result, target_result)
            
            # Update metrics
            if combined_result.valid:
                self.validation_metrics["successful_validations"] += 1
            
            if combined_result.security_threats:
                self.validation_metrics["security_threats_detected"] += len(combined_result.security_threats)
                
                # Count quarantined files
                if any(threat.severity in ["high", "critical"] for threat in combined_result.security_threats):
                    self.validation_metrics["files_quarantined"] += 1
            
        except Exception as e:
            self.logger.error(f"Validation pipeline error: {str(e)}")
            combined_result.add_issue(f"Validation pipeline failed: {str(e)}")
            combined_result.add_security_threat(
                "pipeline_error", "high", "validation_pipeline",
                f"Unexpected error in validation pipeline: {str(e)}",
                "Contact system administrator"
            )
        
        finally:
            combined_result.validation_time_ms = (time.time() - start_time) * 1000
        
        return combined_result
    
    def _merge_results(self, target: ValidationResult, source: ValidationResult):
        """Merge validation results."""
        target.issues.extend(source.issues)
        target.warnings.extend(source.warnings)
        target.recommendations.extend(source.recommendations)
        target.security_threats.extend(source.security_threats)
        target.performance_metrics.update(source.performance_metrics)
        
        if not source.valid:
            target.valid = False
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation engine statistics."""
        return {
            "metrics": self.validation_metrics.copy(),
            "rate_limiter_status": {
                "max_requests_per_minute": self.rate_limiter.max_requests,
                "window_seconds": self.rate_limiter.window_seconds,
            },
            "security_config": {
                "max_file_size_mb": self.security_config.max_file_size_mb,
                "max_neurons": self.security_config.max_neurons,
                "max_synapses": self.security_config.max_synapses,
                "hdl_injection_validation": self.security_config.validate_hdl_injection,
                "quarantine_enabled": self.security_config.quarantine_threats,
            }
        }