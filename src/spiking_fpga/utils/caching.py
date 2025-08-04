"""Intelligent caching system for compilation artifacts and optimization."""

import pickle
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, TypeVar, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time
from collections import OrderedDict

from .logging import StructuredLogger

T = TypeVar('T')


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    def should_evict(self, max_age_seconds: float) -> bool:
        """Check if entry should be evicted based on age."""
        return (datetime.utcnow() - self.last_accessed).total_seconds() > max_age_seconds


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["size_bytes"] -= entry.size_bytes
                return None
            
            # Update access info
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._stats["hits"] += 1
            return entry.data
    
    def put(self, key: str, data: Any, ttl_seconds: Optional[float] = None) -> None:
        """Put item in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(data))
            except Exception:
                size_bytes = 1024  # Fallback estimate
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats["size_bytes"] -= old_entry.size_bytes
                del self._cache[key]
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.default_ttl_seconds
            )
            
            # Add to cache
            self._cache[key] = entry
            self._stats["size_bytes"] += size_bytes
            
            # Evict if necessary
            while len(self._cache) > self.max_size:
                self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # Remove oldest item
        key, entry = self._cache.popitem(last=False)
        self._stats["evictions"] += 1
        self._stats["size_bytes"] -= entry.size_bytes
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats["size_bytes"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "size_bytes": self._stats["size_bytes"],
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }


class FileSystemCache:
    """Persistent file system cache for large compilation artifacts."""
    
    def __init__(self, cache_dir: Path, max_size_mb: float = 1000.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.logger = StructuredLogger("fs_cache")
        
        # Index file to track cache entries
        self.index_file = self.cache_dir / "cache_index.json"
        self.index = self._load_index()
        
        # Cleanup on startup
        self._cleanup_expired()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if not self.index_file.exists():
            return {}
        
        try:
            with open(self.index_file) as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning("Failed to load cache index", error=str(e))
            return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, default=str)
        except Exception as e:
            self.logger.error("Failed to save cache index", error=str(e))
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from filesystem cache."""
        if key not in self.index:
            return None
        
        entry_info = self.index[key]
        cache_path = self._get_cache_path(key)
        
        # Check if file exists
        if not cache_path.exists():
            del self.index[key]
            self._save_index()
            return None
        
        # Check TTL
        if entry_info.get("ttl_seconds"):
            created_at = datetime.fromisoformat(entry_info["created_at"])
            if (datetime.utcnow() - created_at).total_seconds() > entry_info["ttl_seconds"]:
                self._remove_entry(key)
                return None
        
        # Load data
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update access time
            entry_info["last_accessed"] = datetime.utcnow().isoformat()
            entry_info["access_count"] = entry_info.get("access_count", 0) + 1
            self._save_index()
            
            return data
            
        except Exception as e:
            self.logger.error("Failed to load cache entry", key=key, error=str(e))
            self._remove_entry(key)
            return None
    
    def put(self, key: str, data: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Put item in filesystem cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            # Serialize data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Get file size
            size_bytes = cache_path.stat().st_size
            
            # Update index
            self.index[key] = {
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
                "access_count": 1,
                "size_bytes": size_bytes,
                "ttl_seconds": ttl_seconds,
                "file_path": str(cache_path)
            }
            
            self._save_index()
            
            # Cleanup if over size limit
            self._cleanup_if_needed()
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to save cache entry", key=key, error=str(e))
            return False
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry."""
        if key in self.index:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception:
                    pass
            del self.index[key]
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        expired_keys = []
        
        for key, entry_info in self.index.items():
            if entry_info.get("ttl_seconds"):
                created_at = datetime.fromisoformat(entry_info["created_at"])
                if (datetime.utcnow() - created_at).total_seconds() > entry_info["ttl_seconds"]:
                    expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            self._save_index()
            self.logger.info("Cleaned up expired cache entries", count=len(expired_keys))
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if over size limit."""
        total_size = sum(entry.get("size_bytes", 0) for entry in self.index.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by last accessed time (oldest first)
        entries_by_age = sorted(
            self.index.items(),
            key=lambda x: x[1].get("last_accessed", "1970-01-01T00:00:00")
        )
        
        # Remove entries until under limit
        for key, entry_info in entries_by_age:
            self._remove_entry(key)
            total_size -= entry_info.get("size_bytes", 0)
            
            if total_size <= self.max_size_bytes * 0.8:  # Leave 20% headroom
                break
        
        self._save_index()
        self.logger.info("Cache cleanup completed", remaining_entries=len(self.index))
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for key in list(self.index.keys()):
            self._remove_entry(key)
        self.index.clear()
        self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.get("size_bytes", 0) for entry in self.index.values())
        total_accesses = sum(entry.get("access_count", 0) for entry in self.index.values())
        
        return {
            "entry_count": len(self.index),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization_percent": (total_size / self.max_size_bytes) * 100,
            "total_accesses": total_accesses,
            "average_entry_size_kb": (total_size / len(self.index) / 1024) if self.index else 0,
        }


class CompilationCache:
    """High-level caching for compilation artifacts."""
    
    def __init__(self, cache_dir: Optional[Path] = None, enable_memory_cache: bool = True):
        self.logger = StructuredLogger("compilation_cache")
        
        # Memory cache for small, frequently accessed items
        self.memory_cache = LRUCache(max_size=500, default_ttl_seconds=3600) if enable_memory_cache else None
        
        # Filesystem cache for large artifacts
        if cache_dir:
            self.fs_cache = FileSystemCache(cache_dir, max_size_mb=2000.0)
        else:
            self.fs_cache = None
        
        self.logger.info("Compilation cache initialized",
                        memory_cache_enabled=enable_memory_cache,
                        fs_cache_enabled=cache_dir is not None)
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        # Sort keys for consistent hashing
        key_data = {k: v for k, v in sorted(kwargs.items())}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"
    
    def get_network_analysis(self, network_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached network analysis results."""
        key = self._generate_key("network_analysis", network_hash=network_hash)
        
        # Try memory cache first
        if self.memory_cache:
            result = self.memory_cache.get(key)
            if result is not None:
                return result
        
        # Try filesystem cache
        if self.fs_cache:
            result = self.fs_cache.get(key)
            if result is not None and self.memory_cache:
                # Promote to memory cache
                self.memory_cache.put(key, result, ttl_seconds=1800)
                return result
        
        return None
    
    def put_network_analysis(self, network_hash: str, analysis: Dict[str, Any]) -> None:
        """Cache network analysis results."""
        key = self._generate_key("network_analysis", network_hash=network_hash)
        
        # Store in memory cache
        if self.memory_cache:
            self.memory_cache.put(key, analysis, ttl_seconds=1800)
        
        # Store in filesystem cache
        if self.fs_cache:
            self.fs_cache.put(key, analysis, ttl_seconds=7200)
    
    def get_optimization_result(self, network_hash: str, config_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached optimization results."""
        key = self._generate_key("optimization", 
                                network_hash=network_hash, 
                                config_hash=config_hash)
        
        if self.fs_cache:
            result = self.fs_cache.get(key)
            if result is not None:
                self.logger.info("Cache hit for optimization result", key=key)
                return result
        
        return None
    
    def put_optimization_result(self, network_hash: str, config_hash: str, 
                              result: Dict[str, Any]) -> None:
        """Cache optimization results."""
        key = self._generate_key("optimization", 
                                network_hash=network_hash, 
                                config_hash=config_hash)
        
        if self.fs_cache:
            success = self.fs_cache.put(key, result, ttl_seconds=86400)  # 24 hours
            if success:
                self.logger.info("Cached optimization result", key=key)
    
    def get_hdl_generation(self, network_hash: str, config_hash: str) -> Optional[Dict[str, str]]:
        """Get cached HDL generation results."""
        key = self._generate_key("hdl_generation", 
                                network_hash=network_hash, 
                                config_hash=config_hash)
        
        if self.fs_cache:
            result = self.fs_cache.get(key)
            if result is not None:
                self.logger.info("Cache hit for HDL generation", key=key)
                return result
        
        return None
    
    def put_hdl_generation(self, network_hash: str, config_hash: str, 
                          hdl_files: Dict[str, str]) -> None:
        """Cache HDL generation results."""
        key = self._generate_key("hdl_generation", 
                                network_hash=network_hash, 
                                config_hash=config_hash)
        
        if self.fs_cache:
            success = self.fs_cache.put(key, hdl_files, ttl_seconds=86400)
            if success:
                self.logger.info("Cached HDL generation", key=key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {"enabled": True}
        
        if self.memory_cache:
            stats["memory_cache"] = self.memory_cache.get_stats()
        
        if self.fs_cache:
            stats["filesystem_cache"] = self.fs_cache.get_stats()
        
        return stats
    
    def clear(self) -> None:
        """Clear all caches."""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.fs_cache:
            self.fs_cache.clear()
        
        self.logger.info("All caches cleared")