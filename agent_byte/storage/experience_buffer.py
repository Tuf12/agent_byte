"""
Experience buffer with windowing and LRU cache for efficient memory management.

This module provides an efficient buffer for managing experience vectors
with support for streaming, windowing, and caching.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Iterator
from collections import OrderedDict, deque
import logging
import time
import json

class ExperienceBuffer:
    """
    Efficient experience buffer with LRU cache and windowing support.

    Features:
    - LRU cache for recent experiences
    - Sliding window for temporal locality
    - Streaming interface for large datasets
    - Memory-efficient operations
    """

    def __init__(self, max_size: int = 10000, cache_size: int = 1000,
                 window_size: int = 100):
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum number of experiences to store
            cache_size: Size of LRU cache for fast access
            window_size: Size of a sliding window for recent experiences
        """
        self.max_size = max_size
        self.cache_size = cache_size
        self.window_size = window_size
        self.logger = logging.getLogger(self.__class__.__name__)

        # Main storage
        self.experiences = deque(maxlen=max_size)

        # LRU cache for frequently accessed experiences
        self.cache = OrderedDict()

        # Sliding window for recent experiences
        self.window = deque(maxlen=window_size)

        # Statistics
        self.stats = {
            'total_added': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'window_hits': 0,
            'last_update': time.time()
        }

    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> int:
        """
        Add an experience to the buffer.

        Args:
            vector: Experience vector
            metadata: Associated metadata

        Returns:
            Index of the added experience
        """
        experience = {
            'vector': vector.copy(),
            'metadata': metadata.copy(),
            'timestamp': time.time(),
            'index': self.stats['total_added']
        }

        # Add to the main storage
        self.experiences.append(experience)

        # Add to window
        self.window.append(experience)

        # Add to cache if space available
        if len(self.cache) < self.cache_size:
            self._add_to_cache(experience['index'], experience)

        self.stats['total_added'] += 1

        return experience['index']

    def get(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get an experience by index.

        Args:
            index: Experience index

        Returns:
            Experience dict or None if not found
        """
        # Check cache first
        if index in self.cache:
            self.stats['cache_hits'] += 1
            # Move to end (most recently used)
            self.cache.move_to_end(index)
            return self.cache[index]

        self.stats['cache_misses'] += 1

        # Search in the main storage
        for exp in self.experiences:
            if exp['index'] == index:
                # Add to cache
                self._add_to_cache(index, exp)
                return exp

        return None

    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """
        Get n most recent experiences.

        Args:
            n: Number of experiences to get

        Returns:
            List of recent experiences
        """
        return list(self.window)[-n:]

    def search_similar(self, query_vector: np.ndarray, k: int = 5,
                       use_window: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar experiences.

        Args:
            query_vector: Query vector
            k: Number of results
            use_window: Whether to search only in window

        Returns:
            List of similar experiences with similarities
        """
        # Choose search space
        search_space = self.window if use_window else self.experiences

        if use_window:
            self.stats['window_hits'] += 1

        # Calculate similarities
        similarities = []
        query_norm = np.linalg.norm(query_vector)

        for exp in search_space:
            vector = exp['vector']
            vector_norm = np.linalg.norm(vector)

            if query_norm > 0 and vector_norm > 0:
                similarity = np.dot(query_vector, vector) / (query_norm * vector_norm)
                similarities.append({
                    'experience': exp,
                    'similarity': float(similarity)
                })

        # Sort and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:k]

    def get_batch(self, indices: List[int]) -> List[Optional[Dict[str, Any]]]:
        """
        Get multiple experiences by indices.

        Args:
            indices: List of indices

        Returns:
            List of experiences (None for not found)
        """
        return [self.get(idx) for idx in indices]

    def iterate_batches(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate over experiences in batches.

        Args:
            batch_size: Size of each batch

        Yields:
            Batches of experiences
        """
        batch = []
        for exp in self.experiences:
            batch.append(exp)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def sample(self, n: int, recent_bias: float = 0.0) -> List[Dict[str, Any]]:
        """
        Sample n experiences with optional recency bias.

        Args:
            n: Number of samples
            recent_bias: Bias towards recent experiences (0-1)

        Returns:
            Sampled experiences
        """
        if len(self.experiences) <= n:
            return list(self.experiences)

        if recent_bias <= 0:
            # Uniform sampling
            indices = np.random.choice(len(self.experiences), n, replace=False)
            return [self.experiences[i] for i in indices]
        else:
            # Weighted sampling with recency bias
            weights = np.array([
                (1 - recent_bias) + recent_bias * (i / len(self.experiences))
                for i in range(len(self.experiences))
            ])
            weights /= weights.sum()

            indices = np.random.choice(
                len(self.experiences), n, replace=False, p=weights
            )
            return [self.experiences[i] for i in indices]

    def get_vectors_array(self) -> np.ndarray:
        """
        Get all vectors as a numpy array.

        Returns:
            Array of shape (n_experiences, vector_dim)
        """
        if not self.experiences:
            return np.array([])

        return np.array([exp['vector'] for exp in self.experiences])

    def get_metadata_list(self) -> List[Dict[str, Any]]:
        """
        Get all metadata as a list.

        Returns:
            List of metadata dicts
        """
        return [exp['metadata'] for exp in self.experiences]

    def filter_by_metadata(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Filter experiences by metadata attributes.

        Args:
            **kwargs: Metadata key-value pairs to match

        Returns:
            Filtered experiences
        """
        filtered = []
        for exp in self.experiences:
            match = True
            for key, value in kwargs.items():
                if key not in exp['metadata'] or exp['metadata'][key] != value:
                    match = False
                    break
            if match:
                filtered.append(exp)

        return filtered

    def _add_to_cache(self, index: int, experience: Dict[str, Any]):
        """Add experience to LRU cache."""
        # Remove the oldest if at capacity
        if len(self.cache) >= self.cache_size:
            # Remove least recently used (first item)
            self.cache.popitem(last=False)

        self.cache[index] = experience

    def clear_cache(self):
        """Clear the LRU cache."""
        self.cache.clear()
        self.logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        cache_hit_rate = 0.0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (
                    self.stats['cache_hits'] + self.stats['cache_misses']
            )

        return {
            'total_experiences': len(self.experiences),
            'cache_size': len(self.cache),
            'window_size': len(self.window),
            'total_added': self.stats['total_added'],
            'cache_hit_rate': cache_hit_rate,
            'window_hits': self.stats['window_hits'],
            'time_since_update': time.time() - self.stats['last_update']
        }

    def save_to_disk(self, path: str):
        """
        Save buffer to disk.

        Args:
            path: Save path
        """
        data = {
            'experiences': list(self.experiences),
            'stats': self.stats,
            'config': {
                'max_size': self.max_size,
                'cache_size': self.cache_size,
                'window_size': self.window_size
            }
        }

        np.save(path, data)
        self.logger.info(f"Saved buffer to {path}")

    def load_from_disk(self, path: str):
        """
        Load buffer from disk.

        Args:
            path: Load path
        """
        data = np.load(path, allow_pickle=True).item()

        # Restore configuration
        self.max_size = data['config']['max_size']
        self.cache_size = data['config']['cache_size']
        self.window_size = data['config']['window_size']

        # Restore experiences
        self.experiences = deque(data['experiences'], maxlen=self.max_size)

        # Rebuild window
        self.window = deque(
            list(self.experiences)[-self.window_size:],
            maxlen=self.window_size
        )

        # Clear cache (will be rebuilt on access)
        self.cache.clear()

        # Restore stats
        self.stats = data['stats']

        self.logger.info(f"Loaded buffer from {path}")


class StreamingExperienceBuffer(ExperienceBuffer):
    """
    Experience buffer with streaming support for very large datasets.

    Extends ExperienceBuffer with:
    - Disk-backed storage for overflow
    - Streaming iteration
    - Chunked processing
    """

    def __init__(self, max_memory_size: int = 10000,
                 cache_size: int = 1000,
                 window_size: int = 100,
                 disk_path: Optional[str] = None):
        """
        Initialize streaming experience buffer.

        Args:
            max_memory_size: Maximum experiences to keep in memory
            cache_size: Size of LRU cache
            window_size: Size of sliding window
            disk_path: Path for disk-backed storage
        """
        super().__init__(max_memory_size, cache_size, window_size)

        self.disk_path = disk_path
        self.disk_experiences = []
        self.disk_index = 0

        if disk_path:
            from pathlib import Path
            Path(disk_path).mkdir(parents=True, exist_ok=True)

    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> int:
        """
        Add experience with disk overflow support.
        """
        # Check if we need to overflow to disk
        if len(self.experiences) >= self.max_size and self.disk_path:
            # Move the oldest experience to disk
            oldest = self.experiences[0]
            self._save_to_disk(oldest)

        # Add normally
        return super().add(vector, metadata)

    def _save_to_disk(self, experience: Dict[str, Any]):
        """Save experience to disk."""
        if not self.disk_path:
            return

        filename_base = f"{self.disk_path}/exp_{self.disk_index:08d}"

        # Save vector as .npy
        np.save(f"{filename_base}_vector.npy", experience['vector'])

        # Save metadata as JSON
        metadata_with_info = {
            'metadata': experience['metadata'],
            'timestamp': experience['timestamp'],
            'index': experience['index']
        }
        with open(f"{filename_base}_metadata.json", 'w') as f:
            json.dump(metadata_with_info, f)

        self.disk_experiences.append(filename_base)
        self.disk_index += 1

    def iterate_all(self, batch_size: int = 32) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate over all experiences including disk-backed ones.
        """
        # First yield from disk
        if self.disk_experiences:
            batch = []
            for filename_base in self.disk_experiences:
                # Load vector
                vector = np.load(f"{filename_base}_vector.npy")

                # Load metadata
                with open(f"{filename_base}_metadata.json", 'r') as f:
                    meta_info = json.load(f)

                exp = {
                    'vector': vector,
                    'metadata': meta_info['metadata'],
                    'timestamp': meta_info['timestamp'],
                    'index': meta_info['index']
                }
                batch.append(exp)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

        # Then yield from memory
        yield from self.iterate_batches(batch_size)