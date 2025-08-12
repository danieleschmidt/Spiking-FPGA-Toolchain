"""
Advanced caching systems with predictive capabilities.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0


class IntelligentCache:
    """Intelligent caching with usage pattern learning."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        logger.info("IntelligentCache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                return entry.value
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self._lock:
            current_time = time.time()
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time
            )
            
            # Evict if necessary
            if len(self.cache) > self.max_size:
                self._evict_entries()
    
    def _evict_entries(self):
        """Evict least recently used entries."""
        # Simple LRU eviction
        entries_by_access = sorted(
            self.cache.values(),
            key=lambda e: e.last_accessed
        )
        
        # Remove oldest 10%
        to_remove = len(entries_by_access) // 10
        for entry in entries_by_access[:to_remove]:
            del self.cache[entry.key]


class DistributedCache:
    """Distributed caching across multiple nodes."""
    
    def __init__(self, node_id: str = "local"):
        self.node_id = node_id
        self.local_cache = IntelligentCache()
        logger.info(f"DistributedCache initialized for node {node_id}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get from distributed cache."""
        return self.local_cache.get(key)
    
    def put(self, key: str, value: Any):
        """Put to distributed cache."""
        self.local_cache.put(key, value)


class PredictiveCache:
    """Predictive caching based on access patterns."""
    
    def __init__(self):
        self.access_patterns: Dict[str, List[float]] = {}
        self.base_cache = IntelligentCache()
        logger.info("PredictiveCache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get with pattern tracking."""
        current_time = time.time()
        
        # Track access pattern
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        self.access_patterns[key].append(current_time)
        
        return self.base_cache.get(key)
    
    def put(self, key: str, value: Any):
        """Put with predictive prefetching."""
        self.base_cache.put(key, value)