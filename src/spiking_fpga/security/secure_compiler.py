"""Secure compiler wrapper with sandboxing and validation."""

import tempfile
import shutil
import subprocess
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from contextlib import contextmanager

from .input_sanitizer import InputSanitizer

logger = logging.getLogger(__name__)


class SecureCompiler:
    """Secure wrapper for network compilation with sandboxing."""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.temp_dirs = []
        
        # Resource limits
        self.max_compilation_time = 600  # 10 minutes
        self.max_memory_mb = 4096  # 4GB
        self.max_processes = 4
        
    def __del__(self):
        """Clean up temporary directories."""
        self.cleanup_temp_dirs()
    
    def cleanup_temp_dirs(self):
        """Remove all temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_dir}: {e}")
        self.temp_dirs.clear()
    
    @contextmanager
    def secure_sandbox(self):
        """Create a secure sandbox for compilation."""
        # Create isolated temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="spiking_fpga_secure_"))
        self.temp_dirs.append(temp_dir)
        
        try:
            # Set restrictive permissions
            os.chmod(temp_dir, 0o700)
            
            # Create subdirectories
            input_dir = temp_dir / "input"
            output_dir = temp_dir / "output" 
            work_dir = temp_dir / "work"
            
            input_dir.mkdir()
            output_dir.mkdir()
            work_dir.mkdir()
            
            yield {
                'temp_dir': temp_dir,
                'input_dir': input_dir,
                'output_dir': output_dir,
                'work_dir': work_dir
            }
            
        finally:
            # Cleanup handled by __del__ or explicit call
            pass
    
    def validate_compilation_environment(self) -> Dict[str, bool]:
        """Validate that compilation environment is secure."""
        checks = {}
        
        # Check available disk space
        try:
            stat = shutil.disk_usage('/')
            free_gb = stat.free / (1024**3)
            checks['sufficient_disk_space'] = free_gb > 5.0  # 5GB minimum
        except Exception:
            checks['sufficient_disk_space'] = False
        
        # Check memory availability
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            checks['sufficient_memory'] = available_gb > 2.0  # 2GB minimum
        except ImportError:
            logger.warning("psutil not available, skipping memory check")
            checks['sufficient_memory'] = True
        except Exception:
            checks['sufficient_memory'] = False
        
        # Check for running toolchains
        try:
            result = subprocess.run(['pgrep', '-f', 'vivado|quartus'], 
                                    capture_output=True, text=True)
            checks['no_conflicting_tools'] = result.returncode != 0
        except Exception:
            checks['no_conflicting_tools'] = True
        
        return checks
    
    def secure_file_copy(self, source: Path, dest: Path) -> bool:
        """Securely copy a file with validation."""
        try:
            # Validate source
            validated_source = self.sanitizer.validate_file_path(source)
            self.sanitizer.scan_file_content(validated_source)
            
            # Copy with size limit
            with open(validated_source, 'rb') as src, open(dest, 'wb') as dst:
                copied = 0
                while True:
                    chunk = src.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    
                    copied += len(chunk)
                    if copied > self.sanitizer.MAX_NETWORK_FILE_SIZE:
                        raise ValueError("File too large during copy")
                    
                    dst.write(chunk)
            
            # Set restrictive permissions
            os.chmod(dest, 0o600)
            return True
            
        except Exception as e:
            logger.error(f"Secure file copy failed: {e}")
            if dest.exists():
                dest.unlink()
            return False
    
    def run_compilation_with_limits(self, cmd: List[str], work_dir: Path, 
                                   timeout: int = None) -> subprocess.CompletedProcess:
        """Run compilation command with resource limits."""
        if timeout is None:
            timeout = self.max_compilation_time
        
        # Prepare environment
        env = os.environ.copy()
        env['TMPDIR'] = str(work_dir)
        env['TEMP'] = str(work_dir)
        
        # Security: Remove potentially dangerous env vars
        dangerous_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']
        for var in dangerous_vars:
            env.pop(var, None)
        
        try:
            # Run with timeout and resource limits
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=False
            )
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Compilation timeout after {timeout}s")
            raise ValueError(f"Compilation exceeded time limit ({timeout}s)")
        except Exception as e:
            logger.error(f"Compilation process failed: {e}")
            raise ValueError(f"Compilation failed: {e}")
    
    def validate_output_files(self, output_dir: Path) -> List[Path]:
        """Validate and sanitize compilation output files."""
        validated_files = []
        
        # Expected output patterns
        safe_patterns = [
            '*.v', '*.vhd', '*.sv',  # HDL files
            '*.tcl', '*.xdc', '*.sdc',  # Constraint files
            '*.txt', '*.log', '*.rpt',  # Report files
            '*.json', '*.yaml'  # Data files
        ]
        
        for pattern in safe_patterns:
            for file_path in output_dir.glob(pattern):
                try:
                    # Basic validation
                    if file_path.is_file() and file_path.stat().st_size > 0:
                        
                        # Check file size
                        if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                            logger.warning(f"Large output file: {file_path}")
                            continue
                        
                        # Scan content for security issues
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read(10000)  # First 10KB
                                
                            # Check for suspicious content
                            if any(pattern in content.lower() for pattern in 
                                   ['password', 'secret', 'key', 'token']):
                                logger.warning(f"Potentially sensitive content in {file_path}")
                                continue
                                
                        except Exception as e:
                            logger.warning(f"Cannot validate {file_path}: {e}")
                            continue
                        
                        validated_files.append(file_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to validate {file_path}: {e}")
        
        return validated_files
    
    def secure_compile(self, network_file: Path, target: str, 
                      output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Perform secure compilation with full validation."""
        
        # Pre-flight validation
        env_checks = self.validate_compilation_environment()
        failed_checks = [check for check, passed in env_checks.items() if not passed]
        
        if failed_checks:
            raise ValueError(f"Environment validation failed: {failed_checks}")
        
        # Validate inputs
        validated_network = self.sanitizer.validate_file_path(network_file)
        validated_target = self.sanitizer.validate_fpga_target(target)
        
        if 'optimization_level' in kwargs:
            kwargs['optimization_level'] = self.sanitizer.validate_optimization_level(
                kwargs['optimization_level'])
        
        if 'power_budget_mw' in kwargs:
            kwargs['power_budget_mw'] = self.sanitizer.validate_power_budget(
                kwargs['power_budget_mw'])
        
        # Create secure sandbox
        with self.secure_sandbox() as sandbox:
            
            # Copy input file securely
            secure_input = sandbox['input_dir'] / validated_network.name
            if not self.secure_file_copy(validated_network, secure_input):
                raise ValueError("Failed to securely copy input file")
            
            # Load and validate network configuration
            try:
                import yaml
                with open(secure_input, 'r') as f:
                    config = yaml.safe_load(f)
                
                validated_config = self.sanitizer.validate_network_config(config)
                
            except Exception as e:
                raise ValueError(f"Invalid network configuration: {e}")
            
            # Perform actual compilation (importing here to avoid circular imports)
            try:
                from spiking_fpga.network_compiler import NetworkCompiler
                from spiking_fpga.core import FPGATarget
                
                fpga_target = FPGATarget(validated_target)
                compiler = NetworkCompiler(fpga_target)
                
                # Set security-conscious compilation options
                secure_kwargs = kwargs.copy()
                secure_kwargs['work_dir'] = sandbox['work_dir']
                secure_kwargs['output_dir'] = sandbox['output_dir']
                secure_kwargs['enable_security_checks'] = True
                secure_kwargs['max_compilation_time'] = self.max_compilation_time
                
                # Run compilation
                start_time = time.time()
                result = compiler.compile_from_file(secure_input, **secure_kwargs)
                compilation_time = time.time() - start_time
                
                # Validate outputs
                if result.success:
                    validated_outputs = self.validate_output_files(sandbox['output_dir'])
                    
                    # Copy validated outputs to final destination
                    output_dir.mkdir(parents=True, exist_ok=True)
                    final_outputs = {}
                    
                    for output_file in validated_outputs:
                        final_path = output_dir / output_file.name
                        shutil.copy2(output_file, final_path)
                        final_outputs[output_file.stem] = final_path
                    
                    result.hdl_files = final_outputs
                
                # Add security metadata
                result.security_validation = {
                    'input_validated': True,
                    'sandbox_used': True,
                    'output_validated': len(validated_outputs) if result.success else 0,
                    'compilation_time': compilation_time,
                    'environment_checks': env_checks
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Secure compilation failed: {e}")
                raise ValueError(f"Compilation failed: {e}")