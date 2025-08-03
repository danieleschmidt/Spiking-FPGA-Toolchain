"""
Database connection and session management.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List
from contextlib import contextmanager
import threading
import json
from dataclasses import asdict


class DatabaseManager:
    """Manages SQLite database connections and operations."""
    
    def __init__(self, db_path: str = "./data/spiking_fpga.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._local = threading.local()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._initialize_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        
        return self._local.connection
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions with automatic commit/rollback."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database transaction failed: {str(e)}")
            raise
    
    def _initialize_schema(self):
        """Initialize database schema if it doesn't exist."""
        with self.get_session() as conn:
            # Networks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS networks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    config_hash TEXT UNIQUE NOT NULL,
                    config_json TEXT NOT NULL,
                    input_size INTEGER NOT NULL,
                    output_size INTEGER NOT NULL,
                    layer_count INTEGER NOT NULL,
                    connection_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Compilations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compilations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    network_id INTEGER NOT NULL,
                    fpga_target TEXT NOT NULL,
                    optimization_level INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    compilation_time_s REAL,
                    luts_used INTEGER,
                    bram_used INTEGER,
                    dsp_used INTEGER,
                    estimated_freq_mhz REAL,
                    estimated_power_w REAL,
                    max_spike_rate_mhz REAL,
                    inference_latency_ms REAL,
                    hdl_files_json TEXT,
                    errors_json TEXT,
                    warnings_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (network_id) REFERENCES networks (id)
                )
            """)
            
            # Benchmarks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    compilation_id INTEGER NOT NULL,
                    benchmark_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    units TEXT,
                    test_conditions_json TEXT,
                    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (compilation_id) REFERENCES compilations (id)
                )
            """)
            
            # Cache table for compilation artifacts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    data_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    size_bytes INTEGER,
                    hit_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_networks_config_hash ON networks (config_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_compilations_network_target ON compilations (network_id, fpga_target)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_compilation ON benchmarks (compilation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_entries (cache_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries (expires_at)")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results."""
        with self.get_session() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT query and return the new row ID."""
        with self.get_session() as conn:
            cursor = conn.execute(query, params)
            return cursor.lastrowid
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an UPDATE/DELETE query and return affected row count."""
        with self.get_session() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
    
    def vacuum(self):
        """Optimize database by running VACUUM."""
        with self.get_session() as conn:
            conn.execute("VACUUM")
        self.logger.info("Database vacuumed successfully")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_session() as conn:
            stats = {}
            
            # Table counts
            for table in ['networks', 'compilations', 'benchmarks', 'cache_entries']:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Database size
            cursor = conn.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor = conn.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats['database_size_bytes'] = page_count * page_size
            
            # Cache statistics
            cursor = conn.execute("""
                SELECT 
                    SUM(size_bytes) as total_cache_size,
                    SUM(hit_count) as total_hits,
                    COUNT(*) as cache_entries
                FROM cache_entries
            """)
            cache_stats = cursor.fetchone()
            stats.update({
                'cache_size_bytes': cache_stats[0] or 0,
                'cache_total_hits': cache_stats[1] or 0,
                'cache_entries': cache_stats[2] or 0
            })
            
            return stats
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries."""
        with self.get_session() as conn:
            cursor = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at < datetime('now')"
            )
            deleted_count = cursor.rowcount
            
        self.logger.info(f"Cleaned up {deleted_count} expired cache entries")
        return deleted_count
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database."""
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.get_session() as conn:
            backup_conn = sqlite3.connect(str(backup_path))
            conn.backup(backup_conn)
            backup_conn.close()
        
        self.logger.info(f"Database backed up to {backup_path}")


# Global database manager instance
_db_manager = None

def get_database() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def set_database_path(db_path: str):
    """Set custom database path (must be called before first use)."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.close()
    _db_manager = DatabaseManager(db_path)