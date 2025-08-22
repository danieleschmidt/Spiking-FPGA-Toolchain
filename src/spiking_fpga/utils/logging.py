"""Logging configuration and utilities for the Spiking-FPGA-Toolchain."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


def create_logger(name: str, level: int = logging.INFO, 
                 log_file: Optional[Path] = None) -> logging.Logger:
    """Create a simple logger instance for compatibility."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.hasHandlers():
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class StructuredLogger:
    """Structured logger with JSON output support."""
    
    def __init__(self, name: str, level: int = logging.INFO, 
                 log_file: Optional[Path] = None, structured: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.structured = structured
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(StructuredFormatter() if structured else 
                                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if self.structured and kwargs:
            self._log_structured(logging.INFO, message, kwargs)
        else:
            self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if self.structured and kwargs:
            self._log_structured(logging.WARNING, message, kwargs)
        else:
            self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        if self.structured and kwargs:
            self._log_structured(logging.ERROR, message, kwargs)
        else:
            self.logger.error(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        if self.structured and kwargs:
            self._log_structured(logging.DEBUG, message, kwargs)
        else:
            self.logger.debug(message)
    
    def _log_structured(self, level: int, message: str, data: Dict[str, Any]):
        """Log structured message with additional data."""
        record = {
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "level": logging.getLevelName(level)
        }
        self.logger.log(level, json.dumps(record))


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        try:
            # Try to parse as JSON first
            data = json.loads(record.getMessage())
            return json.dumps(data, separators=(',', ':'))
        except (json.JSONDecodeError, ValueError):
            # Fallback to regular message
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if hasattr(record, 'filename'):
                log_data["file"] = f"{record.filename}:{record.lineno}"
            return json.dumps(log_data, separators=(',', ':'))


def configure_logging(log_level: str = "INFO", log_file: Optional[Path] = None, 
                     structured: bool = True) -> StructuredLogger:
    """Configure application-wide logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    return StructuredLogger("spiking_fpga", level, log_file, structured)


class CompilationTracker:
    """Track compilation metrics and performance."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.start_time = None
        self.metrics = {}
    
    def start_compilation(self, network_name: str, target: str):
        """Start tracking compilation."""
        self.start_time = datetime.utcnow()
        self.metrics = {
            "network_name": network_name,
            "target": target,
            "start_time": self.start_time.isoformat(),
        }
        self.logger.info("Compilation started", **self.metrics)
    
    def add_metric(self, key: str, value: Any):
        """Add a compilation metric."""
        self.metrics[key] = value
    
    def finish_compilation(self, success: bool, error: Optional[str] = None):
        """Finish tracking compilation."""
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        self.metrics.update({
            "success": success,
            "duration_seconds": duration,
            "end_time": end_time.isoformat(),
        })
        
        if error:
            self.metrics["error"] = error
        
        if success:
            self.logger.info("Compilation completed successfully", **self.metrics)
        else:
            self.logger.error("Compilation failed", **self.metrics)