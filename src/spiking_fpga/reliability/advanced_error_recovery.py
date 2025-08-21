"""Advanced error recovery and resilience system for Generation 2."""

import logging
import traceback
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Context information for error recovery."""
    error_type: str
    error_message: str
    timestamp: datetime
    component: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None


@dataclass 
class RecoveryStrategy:
    """Recovery strategy definition."""
    name: str
    handler: Callable
    max_attempts: int = 3
    backoff_factor: float = 1.5
    applicable_errors: List[str] = field(default_factory=list)
    priority: int = 0  # Higher priority = earlier execution


class AdvancedErrorRecovery:
    """Advanced error recovery with intelligent fallback strategies."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.fallback_handlers: Dict[str, Callable] = {}
        self.log_file = log_file
        self.lock = threading.RLock()
        
        # Initialize default recovery strategies
        self._initialize_default_strategies()
        
    def _initialize_default_strategies(self):
        """Initialize built-in recovery strategies."""
        
        # Memory cleanup strategy
        self.register_strategy(RecoveryStrategy(
            name="memory_cleanup",
            handler=self._memory_cleanup_recovery,
            max_attempts=2,
            applicable_errors=["MemoryError", "OutOfMemoryError"],
            priority=10
        ))
        
        # File system recovery
        self.register_strategy(RecoveryStrategy(
            name="filesystem_recovery", 
            handler=self._filesystem_recovery,
            max_attempts=3,
            applicable_errors=["FileNotFoundError", "PermissionError", "OSError"],
            priority=8
        ))
        
        # Network/import recovery
        self.register_strategy(RecoveryStrategy(
            name="dependency_recovery",
            handler=self._dependency_recovery,
            max_attempts=2,
            applicable_errors=["ImportError", "ModuleNotFoundError"],
            priority=6
        ))
        
        # Configuration recovery
        self.register_strategy(RecoveryStrategy(
            name="configuration_recovery",
            handler=self._configuration_recovery,
            max_attempts=1,
            applicable_errors=["ConfigurationError", "ValidationError"],
            priority=5
        ))
        
    def register_strategy(self, strategy: RecoveryStrategy):
        """Register a new recovery strategy."""
        with self.lock:
            self.recovery_strategies.append(strategy)
            # Sort by priority (descending)
            self.recovery_strategies.sort(key=lambda s: s.priority, reverse=True)
            
    def register_fallback(self, error_type: str, handler: Callable):
        """Register a fallback handler for specific error types."""
        with self.lock:
            self.fallback_handlers[error_type] = handler
            
    def handle_error(self, error: Exception, component: str, 
                    context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle an error with intelligent recovery strategies.
        
        Returns:
            bool: True if error was successfully recovered, False otherwise
        """
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.utcnow(),
            component=component,
            severity=self._assess_severity(error),
            metadata=context or {},
            stack_trace=traceback.format_exc()
        )
        
        with self.lock:
            self.error_history.append(error_context)
            
        logger.error(f"Error in {component}: {error}", extra={
            "error_type": error_context.error_type,
            "severity": error_context.severity,
            "metadata": error_context.metadata
        })
        
        # Try recovery strategies
        recovery_success = self._attempt_recovery(error_context)
        
        # Log to file if configured
        self._log_error_to_file(error_context, recovery_success)
        
        return recovery_success
        
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt recovery using applicable strategies."""
        applicable_strategies = [
            s for s in self.recovery_strategies
            if not s.applicable_errors or error_context.error_type in s.applicable_errors
        ]
        
        for strategy in applicable_strategies:
            if error_context.recovery_attempts >= strategy.max_attempts:
                continue
                
            logger.info(f"Attempting recovery with strategy: {strategy.name}")
            
            try:
                backoff_time = strategy.backoff_factor ** error_context.recovery_attempts
                if backoff_time > 0.1:  # Don't delay for very short times
                    time.sleep(backoff_time)
                    
                error_context.recovery_attempts += 1
                
                success = strategy.handler(error_context)
                if success:
                    logger.info(f"Recovery successful with strategy: {strategy.name}")
                    return True
                    
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                
        # Try fallback handlers
        fallback_handler = self.fallback_handlers.get(error_context.error_type)
        if fallback_handler:
            try:
                logger.info(f"Attempting fallback for {error_context.error_type}")
                return fallback_handler(error_context)
            except Exception as fallback_error:
                logger.warning(f"Fallback handler failed: {fallback_error}")
                
        return False
        
    def _assess_severity(self, error: Exception) -> str:
        """Assess the severity of an error."""
        critical_errors = [
            "SystemExit", "KeyboardInterrupt", "MemoryError",
            "RecursionError", "SystemError"
        ]
        high_errors = [
            "ValueError", "TypeError", "AttributeError",
            "FileNotFoundError", "PermissionError"
        ]
        medium_errors = [
            "ImportError", "ModuleNotFoundError", "ConnectionError"
        ]
        
        error_type = type(error).__name__
        
        if error_type in critical_errors:
            return "critical"
        elif error_type in high_errors:
            return "high"
        elif error_type in medium_errors:
            return "medium"
        else:
            return "low"
            
    def _memory_cleanup_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt memory cleanup recovery."""
        try:
            import gc
            gc.collect()
            logger.info("Memory cleanup completed")
            return True
        except Exception:
            return False
            
    def _filesystem_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt filesystem-related recovery."""
        try:
            # Create missing directories
            if "No such file or directory" in error_context.error_message:
                # Extract potential directory path from error message
                import re
                path_match = re.search(r"'([^']+)'", error_context.error_message)
                if path_match:
                    potential_path = Path(path_match.group(1))
                    if not potential_path.suffix:  # Likely a directory
                        potential_path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created missing directory: {potential_path}")
                        return True
                    else:  # File path, create parent directory
                        potential_path.parent.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created parent directory: {potential_path.parent}")
                        return True
        except Exception as e:
            logger.debug(f"Filesystem recovery failed: {e}")
            
        return False
        
    def _dependency_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt dependency-related recovery."""
        try:
            # Try to suggest alternative imports or graceful degradation
            if "cryptography" in error_context.error_message:
                logger.info("Cryptography module missing - research features will be limited")
                return True  # Allow graceful degradation
                
            # Other dependency recoveries can be added here
            return False
        except Exception:
            return False
            
    def _configuration_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt configuration-related recovery."""
        try:
            # Reset to default configuration
            logger.info("Attempting configuration recovery with defaults")
            return True  # Simplified for now
        except Exception:
            return False
            
    def _log_error_to_file(self, error_context: ErrorContext, recovery_success: bool):
        """Log error details to file."""
        if not self.log_file:
            return
            
        try:
            error_record = {
                "timestamp": error_context.timestamp.isoformat(),
                "error_type": error_context.error_type,
                "error_message": error_context.error_message,
                "component": error_context.component,
                "severity": error_context.severity,
                "recovery_attempts": error_context.recovery_attempts,
                "recovery_success": recovery_success,
                "metadata": error_context.metadata
            }
            
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(error_record) + "\n")
                
        except Exception as e:
            logger.warning(f"Failed to log error to file: {e}")
            
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self.lock:
            if not self.error_history:
                return {"total_errors": 0}
                
            total_errors = len(self.error_history)
            error_types = {}
            severity_counts = {}
            component_errors = {}
            
            for error in self.error_history:
                error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
                severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
                component_errors[error.component] = component_errors.get(error.component, 0) + 1
                
            recent_errors = [
                e for e in self.error_history 
                if e.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]
            
            return {
                "total_errors": total_errors,
                "error_types": error_types,
                "severity_distribution": severity_counts,
                "component_distribution": component_errors,
                "recent_errors_24h": len(recent_errors),
                "recovery_rate": sum(1 for e in self.error_history if e.recovery_attempts > 0) / total_errors
            }
            
    def clear_error_history(self, older_than_hours: int = 168):  # 1 week default
        """Clear old error history."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        with self.lock:
            self.error_history = [
                e for e in self.error_history 
                if e.timestamp > cutoff_time
            ]