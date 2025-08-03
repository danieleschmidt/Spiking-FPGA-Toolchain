"""
Cache manager for storing and retrieving compilation artifacts.
"""

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import threading

from ..database.connection import get_database


class CacheManager:
    """Manages caching of compilation artifacts and intermediate results."""
    
    def __init__(self, cache_dir: str = "./cache", enable_persistence: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_persistence = enable_persistence
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache for frequently accessed items
        self._memory_cache: Dict[str, Dict] = {}
        self._cache_lock = threading.RLock()
        
        # Cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_entries': 0,
            'disk_entries': 0
        }
    
    def _generate_cache_key(self, data: Union[str, Dict, Any]) -> str:
        """Generate a unique cache key from input data."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve item from cache."""
        with self._cache_lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if not self._is_expired(entry):
                    entry['last_accessed'] = time.time()
                    entry['hit_count'] += 1
                    self._stats['hits'] += 1
                    return entry['data']
                else:
                    # Remove expired entry
                    del self._memory_cache[key]
            
            # Check persistent cache
            if self.enable_persistence:
                cached_data = self._get_from_database(key)
                if cached_data is not None:
                    # Load into memory cache
                    self._memory_cache[key] = {
                        'data': cached_data,
                        'created_at': time.time(),
                        'last_accessed': time.time(),
                        'hit_count': 1,
                        'ttl': None
                    }
                    self._stats['hits'] += 1
                    return cached_data
            
            # Check file cache
            file_path = self.cache_dir / f"{key}.cache"
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Update access time
                    file_path.touch()
                    
                    # Load into memory cache
                    self._memory_cache[key] = {
                        'data': cached_data,
                        'created_at': time.time(),
                        'last_accessed': time.time(),
                        'hit_count': 1,
                        'ttl': None
                    }
                    
                    self._stats['hits'] += 1
                    return cached_data
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load cache file {file_path}: {e}")
                    file_path.unlink(missing_ok=True)
            
            self._stats['misses'] += 1
            return default
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None, 
            persist: bool = True) -> bool:
        """Store item in cache."""
        with self._cache_lock:
            try:
                # Store in memory cache
                self._memory_cache[key] = {
                    'data': data,
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'hit_count': 0,
                    'ttl': ttl
                }
                
                # Store in persistent cache if enabled
                if self.enable_persistence and persist:
                    self._store_in_database(key, data, ttl)
                
                # Store in file cache as backup
                self._store_in_file(key, data)
                
                # Update statistics
                self._stats['memory_entries'] = len(self._memory_cache)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache data for key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Remove item from cache."""
        with self._cache_lock:
            deleted = False
            
            # Remove from memory cache
            if key in self._memory_cache:
                del self._memory_cache[key]
                deleted = True
            
            # Remove from database
            if self.enable_persistence:
                if self._delete_from_database(key):
                    deleted = True
            
            # Remove from file cache
            file_path = self.cache_dir / f"{key}.cache"
            if file_path.exists():
                file_path.unlink()
                deleted = True
            
            return deleted
    
    def clear(self) -> int:
        """Clear all cached items."""
        with self._cache_lock:
            # Clear memory cache
            memory_count = len(self._memory_cache)
            self._memory_cache.clear()
            
            # Clear database cache
            db_count = 0
            if self.enable_persistence:
                db = get_database()
                db_count = db.execute_update("DELETE FROM cache_entries")
            
            # Clear file cache
            file_count = 0
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
                file_count += 1
            
            total_cleared = memory_count + db_count + file_count
            self.logger.info(f"Cleared {total_cleared} cache entries")
            
            return total_cleared
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        with self._cache_lock:
            expired_count = 0
            
            # Clean memory cache
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
                expired_count += 1
            
            # Clean database cache
            if self.enable_persistence:
                db = get_database()
                expired_count += db.cleanup_expired_cache()
            
            # Clean old file cache (older than 7 days)
            cutoff_time = time.time() - (7 * 24 * 3600)
            for cache_file in self.cache_dir.glob("*.cache"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    expired_count += 1
            
            if expired_count > 0:
                self.logger.info(f"Cleaned up {expired_count} expired cache entries")
            
            return expired_count
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._cache_lock:
            stats = self._stats.copy()
            stats['memory_entries'] = len(self._memory_cache)
            
            # Count file cache entries
            stats['file_entries'] = len(list(self.cache_dir.glob("*.cache")))
            
            # Get database cache stats
            if self.enable_persistence:
                db_stats = get_database().get_database_stats()
                stats['database_entries'] = db_stats.get('cache_entries', 0)
                stats['database_size_bytes'] = db_stats.get('cache_size_bytes', 0)
            
            return stats
    
    def cache_compilation_result(self, network_config: Dict, target: str, 
                               result: Any) -> str:
        """Cache compilation result with automatic key generation."""
        cache_data = {
            'network_config': network_config,
            'target': target,
            'type': 'compilation_result'
        }
        
        key = self._generate_cache_key(cache_data)
        
        if self.set(key, result, ttl=3600):  # 1 hour TTL
            self.logger.debug(f"Cached compilation result with key {key}")
            return key
        
        return ""
    
    def get_compilation_result(self, network_config: Dict, target: str) -> Any:
        """Retrieve cached compilation result."""
        cache_data = {
            'network_config': network_config,
            'target': target,
            'type': 'compilation_result'
        }
        
        key = self._generate_cache_key(cache_data)
        result = self.get(key)
        
        if result is not None:
            self.logger.debug(f"Found cached compilation result for key {key}")
        
        return result
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        if entry.get('ttl') is None:
            return False
        
        age = time.time() - entry['created_at']
        return age > entry['ttl']
    
    def _store_in_database(self, key: str, data: Any, ttl: Optional[int]):
        """Store cache entry in database."""
        try:
            db = get_database()
            
            data_json = json.dumps(data, default=str)
            size_bytes = len(data_json.encode())
            
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            db.execute_insert(
                """INSERT OR REPLACE INTO cache_entries 
                   (cache_key, data_type, data_json, size_bytes, expires_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, type(data).__name__, data_json, size_bytes, expires_at)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store cache entry in database: {e}")
    
    def _get_from_database(self, key: str) -> Any:
        """Retrieve cache entry from database."""
        try:
            db = get_database()
            
            rows = db.execute_query(
                """SELECT data_json, expires_at FROM cache_entries 
                   WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > datetime('now'))""",
                (key,)
            )
            
            if rows:
                # Update hit count and last accessed
                db.execute_update(
                    """UPDATE cache_entries 
                       SET hit_count = hit_count + 1, last_accessed = datetime('now')
                       WHERE cache_key = ?""",
                    (key,)
                )
                
                return json.loads(rows[0]['data_json'])
            
        except Exception as e:
            self.logger.warning(f"Failed to retrieve cache entry from database: {e}")
        
        return None
    
    def _delete_from_database(self, key: str) -> bool:
        """Delete cache entry from database."""
        try:
            db = get_database()
            return db.execute_update("DELETE FROM cache_entries WHERE cache_key = ?", (key,)) > 0
        except Exception as e:
            self.logger.warning(f"Failed to delete cache entry from database: {e}")
            return False
    
    def _store_in_file(self, key: str, data: Any):
        """Store cache entry in file."""
        try:
            file_path = self.cache_dir / f"{key}.cache"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to store cache file: {e}")


# Global cache manager instance
_cache_manager = None

def get_cache() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager