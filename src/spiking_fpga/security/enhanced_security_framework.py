"""Enhanced security framework for Generation 2 robustness."""

import hashlib
import hmac
import logging
import secrets
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import threading

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: str
    timestamp: datetime
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    risk_level: str = "low"  # low, medium, high, critical
    details: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class AccessAttempt:
    """Record of access attempts for rate limiting."""
    timestamp: datetime
    source: str
    success: bool
    

class SecurityValidator:
    """Comprehensive security validation system."""
    
    def __init__(self):
        self.blocked_patterns = {
            # Code injection patterns
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'open\s*\(',
            r'file\s*\(',
            
            # Path traversal
            r'\.\./.*',
            r'\.\.\\.*',
            r'/etc/',
            r'/proc/',
            r'/sys/',
            
            # Network and protocol
            r'ftp://',
            r'file://',
            r'gopher://',
            r'ldap://',
            
            # Script injection
            r'<script.*?>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
        }
        
        self.compiled_patterns = {
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.blocked_patterns
        }
        
    def validate_input(self, input_data: Any, context: str = "general") -> List[str]:
        """Validate input for security threats."""
        issues = []
        
        if isinstance(input_data, str):
            issues.extend(self._validate_string(input_data, context))
        elif isinstance(input_data, dict):
            issues.extend(self._validate_dict(input_data, context))
        elif isinstance(input_data, list):
            issues.extend(self._validate_list(input_data, context))
            
        return issues
        
    def _validate_string(self, data: str, context: str) -> List[str]:
        """Validate string input."""
        issues = []
        
        # Check length limits
        if len(data) > 10000:  # Configurable limit
            issues.append(f"Input too long: {len(data)} characters")
            
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(data):
                issues.append(f"Potentially dangerous pattern detected: {pattern.pattern}")
                
        # Check for null bytes
        if '\x00' in data:
            issues.append("Null bytes detected in input")
            
        return issues
        
    def _validate_dict(self, data: Dict[str, Any], context: str) -> List[str]:
        """Validate dictionary input."""
        issues = []
        
        # Check key safety
        for key in data.keys():
            if not isinstance(key, str):
                issues.append(f"Non-string key detected: {type(key)}")
                continue
                
            key_issues = self._validate_string(key, f"{context}_key")
            if key_issues:
                issues.extend([f"Key '{key}': {issue}" for issue in key_issues])
                
        # Check values recursively
        for key, value in data.items():
            value_issues = self.validate_input(value, f"{context}_{key}")
            if value_issues:
                issues.extend([f"Value for '{key}': {issue}" for issue in value_issues])
                
        return issues
        
    def _validate_list(self, data: List[Any], context: str) -> List[str]:
        """Validate list input."""
        issues = []
        
        if len(data) > 1000:  # Configurable limit
            issues.append(f"List too long: {len(data)} items")
            
        for i, item in enumerate(data):
            item_issues = self.validate_input(item, f"{context}_{i}")
            if item_issues:
                issues.extend([f"Item {i}: {issue}" for issue in item_issues])
                
        return issues


class RateLimiter:
    """Rate limiting for API protection."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 60):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.attempts: Dict[str, List[AccessAttempt]] = {}
        self.lock = threading.RLock()
        
    def is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited."""
        with self.lock:
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=self.window_minutes)
            
            # Clean old attempts
            if identifier in self.attempts:
                self.attempts[identifier] = [
                    attempt for attempt in self.attempts[identifier]
                    if attempt.timestamp > window_start
                ]
            else:
                self.attempts[identifier] = []
                
            # Check rate limit
            recent_attempts = len(self.attempts[identifier])
            return recent_attempts >= self.max_requests
            
    def record_attempt(self, identifier: str, success: bool = True):
        """Record an access attempt."""
        with self.lock:
            if identifier not in self.attempts:
                self.attempts[identifier] = []
                
            self.attempts[identifier].append(AccessAttempt(
                timestamp=datetime.utcnow(),
                source=identifier,
                success=success
            ))


class SecureConfigManager:
    """Secure configuration management with encryption."""
    
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
        self._config_cache: Dict[str, Any] = {}
        self.lock = threading.RLock()
        
    def store_config(self, name: str, config: Dict[str, Any], file_path: Path):
        """Store encrypted configuration."""
        try:
            with self.lock:
                config_json = json.dumps(config)
                encrypted_data = self.cipher.encrypt(config_json.encode())
                
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(encrypted_data)
                    
                self._config_cache[name] = config
                logger.info(f"Stored encrypted config: {name}")
                
        except Exception as e:
            logger.error(f"Failed to store config {name}: {e}")
            raise
            
    def load_config(self, name: str, file_path: Path) -> Dict[str, Any]:
        """Load and decrypt configuration."""
        try:
            with self.lock:
                if name in self._config_cache:
                    return self._config_cache[name].copy()
                    
                if not file_path.exists():
                    raise FileNotFoundError(f"Config file not found: {file_path}")
                    
                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()
                    
                decrypted_data = self.cipher.decrypt(encrypted_data)
                config = json.loads(decrypted_data.decode())
                
                self._config_cache[name] = config
                return config.copy()
                
        except Exception as e:
            logger.error(f"Failed to load config {name}: {e}")
            raise


class EnhancedSecurityFramework:
    """Comprehensive security framework for Generation 2."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.validator = SecurityValidator()
        self.rate_limiter = RateLimiter()
        self.config_manager = SecureConfigManager()
        self.security_events: List[SecurityEvent] = []
        self.log_file = log_file
        self.lock = threading.RLock()
        
        # Security monitoring
        self.suspicious_activity_threshold = 5
        self.blocked_sources: Set[str] = set()
        
    def validate_and_sanitize(self, data: Any, context: str = "general") -> Any:
        """Validate and sanitize input data."""
        # Log security event
        self._log_security_event("input_validation", {
            "context": context,
            "data_type": type(data).__name__
        })
        
        # Validate input
        issues = self.validator.validate_input(data, context)
        if issues:
            self._log_security_event("validation_failure", {
                "context": context,
                "issues": issues
            }, risk_level="high")
            raise ValueError(f"Security validation failed: {'; '.join(issues)}")
            
        return data
        
    def check_rate_limit(self, identifier: str) -> bool:
        """Check rate limiting for identifier."""
        if identifier in self.blocked_sources:
            self._log_security_event("blocked_source_attempt", {
                "identifier": identifier
            }, risk_level="critical")
            return True
            
        is_limited = self.rate_limiter.is_rate_limited(identifier)
        if is_limited:
            self._log_security_event("rate_limit_exceeded", {
                "identifier": identifier
            }, risk_level="medium")
            
        self.rate_limiter.record_attempt(identifier, not is_limited)
        return is_limited
        
    def block_source(self, identifier: str, reason: str):
        """Block a source from access."""
        with self.lock:
            self.blocked_sources.add(identifier)
            self._log_security_event("source_blocked", {
                "identifier": identifier,
                "reason": reason
            }, risk_level="high")
            
    def validate_file_path(self, file_path: Path, allowed_dirs: Optional[List[Path]] = None) -> bool:
        """Validate file path for security."""
        try:
            # Resolve path to prevent traversal attacks
            resolved_path = file_path.resolve()
            
            # Check if path is in allowed directories
            if allowed_dirs:
                path_allowed = any(
                    str(resolved_path).startswith(str(allowed_dir.resolve()))
                    for allowed_dir in allowed_dirs
                )
                if not path_allowed:
                    self._log_security_event("unauthorized_path_access", {
                        "path": str(file_path),
                        "resolved_path": str(resolved_path)
                    }, risk_level="high")
                    return False
                    
            # Check for suspicious path components
            path_str = str(resolved_path)
            if any(suspicious in path_str.lower() for suspicious in [
                '/etc/', '/proc/', '/sys/', '\\windows\\', '\\system32\\'
            ]):
                self._log_security_event("suspicious_path_access", {
                    "path": path_str
                }, risk_level="critical")
                return False
                
            return True
            
        except Exception as e:
            self._log_security_event("path_validation_error", {
                "path": str(file_path),
                "error": str(e)
            }, risk_level="medium")
            return False
            
    def _log_security_event(self, event_type: str, details: Dict[str, Any], 
                           risk_level: str = "low"):
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            risk_level=risk_level,
            details=details
        )
        
        with self.lock:
            self.security_events.append(event)
            
        logger.warning(f"Security event: {event_type}", extra={
            "event_type": event_type,
            "risk_level": risk_level,
            "details": details
        })
        
        # Write to security log file
        if self.log_file:
            self._write_security_log(event)
            
    def _write_security_log(self, event: SecurityEvent):
        """Write security event to log file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            log_entry = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "risk_level": event.risk_level,
                "details": event.details
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write security log: {e}")
            
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        with self.lock:
            event_types = {}
            risk_levels = {}
            
            recent_events = [
                e for e in self.security_events
                if e.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]
            
            for event in recent_events:
                event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
                risk_levels[event.risk_level] = risk_levels.get(event.risk_level, 0) + 1
                
            return {
                "total_events_24h": len(recent_events),
                "event_types": event_types,
                "risk_distribution": risk_levels,
                "blocked_sources": len(self.blocked_sources),
                "active_rate_limits": len(self.rate_limiter.attempts)
            }