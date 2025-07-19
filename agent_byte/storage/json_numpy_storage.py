"""
JSON+Numpy storage implementation.

This module provides a file-based storage backend using JSON for metadata
and NumPy for vector data. Enhanced with Sprint 9 continuous action space support.
"""

import json
import threading
from collections import defaultdict

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import time

from .base import StorageBase


class JsonNumpyStorage(StorageBase):
    """
    JSON+Numpy storage implementation.

    Uses JSON files for metadata and separate numpy files for vector data.
    Enhanced with Sprint 9 continuous action space support.
    """

    def __init__(self, base_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize JSON+Numpy storage.

        Args:
            base_path: Base directory for storage
            config: Storage configuration
        """
        super().__init__(config)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.lazy_loading = self.config.get('lazy_loading', True)
        self.memory_limit = self.config.get('memory_limit', 1000)
        self.batch_size = self.config.get('batch_size', 100)
        self.auto_flush_interval = self.config.get('auto_flush_interval', 60)  # seconds

        # In-memory buffers for batch operations
        self._experience_buffers = defaultdict(list)
        self._metadata_buffers = defaultdict(list)
        self._buffer_locks = defaultdict(threading.Lock)

        # Lazy loading state
        self._loaded_vectors = {}
        self._vector_metadata = {}
        self._vector_counts = {}

        # Auto-flush timer
        self._last_flush = time.time()

        self.logger = logging.getLogger(__name__)

    def _get_agent_path(self, agent_id: str) -> Path:
        """Get the directory path for an agent."""
        return self.base_path / "agents" / agent_id

    def _get_env_path(self, agent_id: str, env_id: str) -> Path:
        """Get the directory path for an agent's environment."""
        return self._get_agent_path(agent_id) / "environments" / env_id

    def _ensure_agent_dir(self, agent_id: str) -> Path:
        """Ensure agent directory exists."""
        agent_path = self._get_agent_path(agent_id)
        agent_path.mkdir(parents=True, exist_ok=True)
        return agent_path

    def _ensure_env_dir(self, agent_id: str, env_id: str) -> Path:
        """Ensure environment directory exists."""
        env_path = self._get_env_path(agent_id, env_id)
        env_path.mkdir(parents=True, exist_ok=True)
        return env_path

    def _load_json(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Safely load JSON file."""
        try:
            if not filepath.exists():
                return None
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load JSON from {filepath}: {e}")
            return None

    def _save_json(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Safely save JSON file."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except (IOError, TypeError) as e:
            self.logger.error(f"Failed to save JSON to {filepath}: {e}")
            return False

    def _add_timestamp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add timestamp to data."""
        data['timestamp'] = time.time()
        return data

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        return self._cache.get(key) if self._cache else None

    def _put_in_cache(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if self._cache is not None:
            self._cache[key] = value

    def _validate_vector(self, vector: np.ndarray, expected_dim: int = 256) -> bool:
        """Validate vector dimensions and type."""
        if not isinstance(vector, np.ndarray):
            return False
        if vector.ndim != 1:
            return False
        if vector.shape[0] != expected_dim:
            return False
        if not np.isfinite(vector).all():
            return False
        return True
    # Optimized experience vector storage with batching

    def save_experience_vector(self, agent_id: str, vector: np.ndarray,
                               metadata: Dict[str, Any]) -> bool:
        """Save experience vector for similarity search with batching."""
        if not self._validate_vector(vector):
            return False

        try:
            with self._buffer_locks[agent_id]:
                # Add to buffer
                self._experience_buffers[agent_id].append(vector.copy())
                metadata_with_timestamp = {
                    **metadata,
                    'saved_at': time.time(),
                    'buffer_index': len(self._experience_buffers[agent_id]) - 1
                }
                self._metadata_buffers[agent_id].append(metadata_with_timestamp)

                # Flush if buffer is full or auto-flush interval exceeded
                should_flush = (
                        len(self._experience_buffers[agent_id]) >= self.batch_size or
                        time.time() - self._last_flush > self.auto_flush_interval
                )

                if should_flush:
                    self._flush_buffers(agent_id)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save experience vector: {e}")
            return False

    def _flush_buffers(self, agent_id: str) -> bool:
        """Flush in-memory buffers to disk."""
        try:
            if not self._experience_buffers[agent_id]:
                return True

            agent_path = self._ensure_agent_dir(agent_id)
            exp_path = agent_path / "experiences"
            exp_path.mkdir(exist_ok=True)

            vectors_file = exp_path / "vectors.npy"
            metadata_file = exp_path / "metadata.json"

            # Load existing data
            if vectors_file.exists():
                existing_vectors = np.load(vectors_file)
                existing_metadata = self._load_json(metadata_file) or {"experiences": []}
            else:
                existing_vectors = np.empty((0, 256))
                existing_metadata = {"experiences": []}

            # Prepare new data
            new_vectors = np.array(self._experience_buffers[agent_id])
            new_metadata = self._metadata_buffers[agent_id].copy()

            # Update indices in metadata
            start_index = len(existing_metadata["experiences"])
            for i, metadata in enumerate(new_metadata):
                metadata['index'] = start_index + i

            # Combine data
            if existing_vectors.size > 0:
                combined_vectors = np.vstack([existing_vectors, new_vectors])
            else:
                combined_vectors = new_vectors

            combined_metadata = {
                "experiences": existing_metadata["experiences"] + new_metadata,
                "total_count": len(existing_metadata["experiences"]) + len(new_metadata),
                "last_updated": time.time()
            }

            # Save combined data
            np.save(vectors_file, combined_vectors.astype(np.float32))
            self._save_json(metadata_file, combined_metadata)

            # Update lazy loading cache
            if self.lazy_loading:
                self._loaded_vectors[agent_id] = combined_vectors
                self._vector_metadata[agent_id] = combined_metadata["experiences"]
                self._vector_counts[agent_id] = combined_metadata["total_count"]

            # Clear buffers
            self._experience_buffers[agent_id].clear()
            self._metadata_buffers[agent_id].clear()
            self._last_flush = time.time()

            self.logger.info(f"Flushed {len(new_vectors)} vectors for agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to flush buffers for agent {agent_id}: {e}")
            return False

    def flush_all_buffers(self) -> bool:
        """Manually flush all agent buffers."""
        success = True
        for agent_id in list(self._experience_buffers.keys()):
            with self._buffer_locks[agent_id]:
                if not self._flush_buffers(agent_id):
                    success = False
        return success

    def search_similar_experiences(self, agent_id: str, query_vector: np.ndarray,
                                   k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar experiences using optimized cosine similarity."""
        if not self._validate_vector(query_vector):
            return []

        try:
            # Ensure buffers are flushed first
            with self._buffer_locks[agent_id]:
                if self._experience_buffers[agent_id]:
                    self._flush_buffers(agent_id)

            # Load vectors if not in cache
            if agent_id not in self._loaded_vectors:
                self._load_vectors_lazy(agent_id)

            if agent_id not in self._loaded_vectors or len(self._loaded_vectors[agent_id]) == 0:
                return []

            vectors = self._loaded_vectors[agent_id]
            metadata = self._vector_metadata[agent_id]

            # Normalize query vector
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                normalized_query = query_vector / query_norm
            else:
                normalized_query = query_vector

            # Normalize stored vectors (if not already normalized)
            vector_norms = np.linalg.norm(vectors, axis=1)
            mask = vector_norms > 0
            normalized_vectors = vectors.copy()
            normalized_vectors[mask] = vectors[mask] / vector_norms[mask].reshape(-1, 1)

            # Calculate cosine similarities efficiently
            similarities = np.dot(normalized_vectors, normalized_query)

            # Get top k results
            top_indices = np.argsort(similarities)[-k:][::-1]

            results = []
            for idx in top_indices:
                if idx < len(metadata):
                    result = {
                        'vector': vectors[idx],
                        'metadata': metadata[idx],
                        'similarity': float(similarities[idx])
                    }
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Failed to search experiences: {e}")
            return []

    def _load_vectors_lazy(self, agent_id: str) -> None:
        """Load vectors lazily from disk."""
        try:
            agent_path = self._get_agent_path(agent_id)
            exp_path = agent_path / "experiences"

            vectors_file = exp_path / "vectors.npy"
            metadata_file = exp_path / "metadata.json"

            if vectors_file.exists() and metadata_file.exists():
                vectors = np.load(vectors_file)
                metadata_data = self._load_json(metadata_file)

                if metadata_data and "experiences" in metadata_data:
                    self._loaded_vectors[agent_id] = vectors
                    self._vector_metadata[agent_id] = metadata_data["experiences"]
                    self._vector_counts[agent_id] = metadata_data.get("total_count", len(vectors))
                else:
                    self._loaded_vectors[agent_id] = np.empty((0, 256))
                    self._vector_metadata[agent_id] = []
                    self._vector_counts[agent_id] = 0
            else:
                self._loaded_vectors[agent_id] = np.empty((0, 256))
                self._vector_metadata[agent_id] = []
                self._vector_counts[agent_id] = 0

        except Exception as e:
            self.logger.error(f"Failed to load vectors lazily: {e}")
            self._loaded_vectors[agent_id] = np.empty((0, 256))
            self._vector_metadata[agent_id] = []
            self._vector_counts[agent_id] = 0

    def _update_lazy_cache(self, agent_id: str, vectors: np.ndarray, metadata: List[Dict]):
        """Update lazy loading cache with memory management."""
        # Simple memory management - remove oldest if over limit
        if len(self._loaded_vectors) > self.memory_limit:
            # Remove oldest entry
            oldest_agent = next(iter(self._loaded_vectors))
            del self._loaded_vectors[oldest_agent]
            if oldest_agent in self._vector_metadata:
                del self._vector_metadata[oldest_agent]
            if oldest_agent in self._vector_counts:
                del self._vector_counts[oldest_agent]

        self._loaded_vectors[agent_id] = vectors
        self._vector_metadata[agent_id] = metadata
        self._vector_counts[agent_id] = len(vectors)

    # Rest of the methods remain the same as original JsonNumpyStorage
    # (save_agent_profile, load_agent_profile, save_brain_state, etc.)

    def save_agent_profile(self, agent_id: str, profile: Dict[str, Any]) -> bool:
        """Save agent profile."""
        agent_path = self._ensure_agent_dir(agent_id)
        profile_file = agent_path / "profile.json"

        profile = self._add_timestamp(profile.copy())
        success = self._save_json(profile_file, profile)

        if success:
            cache_key = f"profile_{agent_id}"
            if self._cache and cache_key in self._cache:
                del self._cache[cache_key]

        return success

    def load_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent profile."""
        cache_key = f"profile_{agent_id}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        agent_path = self._get_agent_path(agent_id)
        profile_file = agent_path / "profile.json"

        profile = self._load_json(profile_file)
        if profile:
            self._put_in_cache(cache_key, profile)

        return profile

    def get_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Alias for load_agent_profile for backward compatibility."""
        return self.load_agent_profile(agent_id)

    def save_brain_state(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save neural brain state."""
        env_path = self._ensure_env_dir(agent_id, env_id)
        brain_file = env_path / "brain_state.json"

        data = self._add_timestamp(data.copy())
        return self._save_json(brain_file, data)

    def load_brain_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load neural brain state."""
        env_path = self._get_env_path(agent_id, env_id)
        brain_file = env_path / "brain_state.json"

        return self._load_json(brain_file)

    def save_knowledge(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save symbolic knowledge."""
        env_path = self._ensure_env_dir(agent_id, env_id)
        knowledge_file = env_path / "knowledge.json"

        data = self._add_timestamp(data.copy())
        return self._save_json(knowledge_file, data)

    def load_knowledge(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load symbolic knowledge."""
        env_path = self._get_env_path(agent_id, env_id)
        knowledge_file = env_path / "knowledge.json"

        return self._load_json(knowledge_file)

    def save_autoencoder(self, agent_id: str, env_id: str, state_dict: Dict[str, Any]) -> bool:
        """Save autoencoder state."""
        env_path = self._ensure_env_dir(agent_id, env_id)
        autoencoder_file = env_path / "autoencoder.json"

        state_dict = self._add_timestamp(state_dict.copy())
        return self._save_json(autoencoder_file, state_dict)

    def load_autoencoder(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load autoencoder state."""
        env_path = self._get_env_path(agent_id, env_id)
        autoencoder_file = env_path / "autoencoder.json"

        return self._load_json(autoencoder_file)

    def list_autoencoders(self, agent_id: str) -> List[str]:
        """List all environments with autoencoders for an agent."""
        agent_path = self._get_agent_path(agent_id)
        if not agent_path.exists():
            return []

        autoencoders = []
        envs_path = agent_path / "environments"
        if envs_path.exists():
            for env_dir in envs_path.iterdir():
                if env_dir.is_dir():
                    autoencoder_file = env_dir / "autoencoder.json"
                    if autoencoder_file.exists():
                        autoencoders.append(env_dir.name)

        return autoencoders

    # Sprint 9: Continuous action space support

    def save_continuous_network_state(self, agent_id: str, env_id: str, state: Dict[str, Any]) -> bool:
        """Save continuous network state (SAC, DDPG, etc.)."""
        env_path = self._ensure_env_dir(agent_id, env_id)
        network_file = env_path / "continuous_network.json"

        state = self._add_timestamp(state.copy())
        return self._save_json(network_file, state)

    def load_continuous_network_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load continuous network state."""
        env_path = self._get_env_path(agent_id, env_id)
        network_file = env_path / "continuous_network.json"

        return self._load_json(network_file)

    def list_continuous_networks(self, agent_id: str) -> List[str]:
        """List all environments with continuous networks for an agent."""
        agent_path = self._get_agent_path(agent_id)
        if not agent_path.exists():
            return []

        networks = []
        envs_path = agent_path / "environments"
        if envs_path.exists():
            for env_dir in envs_path.iterdir():
                if env_dir.is_dir():
                    network_file = env_dir / "continuous_network.json"
                    if network_file.exists():
                        networks.append(env_dir.name)

        return networks

    def save_action_adapter_config(self, agent_id: str, env_id: str, config: Dict[str, Any]) -> bool:
        """Save action adapter configuration."""
        env_path = self._ensure_env_dir(agent_id, env_id)
        adapter_file = env_path / "action_adapter.json"

        config = self._add_timestamp(config.copy())
        return self._save_json(adapter_file, config)

    def load_action_adapter_config(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load action adapter configuration."""
        env_path = self._get_env_path(agent_id, env_id)
        adapter_file = env_path / "action_adapter.json"

        return self._load_json(adapter_file)

    def list_action_adapters(self, agent_id: str) -> List[str]:
        """List all environments with action adapters for an agent."""
        agent_path = self._get_agent_path(agent_id)
        if not agent_path.exists():
            return []

        adapters = []
        envs_path = agent_path / "environments"
        if envs_path.exists():
            for env_dir in envs_path.iterdir():
                if env_dir.is_dir():
                    adapter_file = env_dir / "action_adapter.json"
                    if adapter_file.exists():
                        adapters.append(env_dir.name)

        return adapters

    def list_environments(self, agent_id: str) -> List[str]:
        """List all environments for an agent."""
        agent_path = self._get_agent_path(agent_id)
        if not agent_path.exists():
            return []

        environments = []
        envs_path = agent_path / "environments"
        if envs_path.exists():
            for env_dir in envs_path.iterdir():
                if env_dir.is_dir():
                    environments.append(env_dir.name)

        return environments

    def _get_all_agent_ids(self) -> List[str]:
        """Get all agent IDs from storage."""
        agents_path = self.base_path / "agents"
        if not agents_path.exists():
            return []

        return [d.name for d in agents_path.iterdir() if d.is_dir()]

    def close(self) -> None:
        """Close storage and clean up."""
        # Flush all remaining buffers
        self.flush_all_buffers()

        # Clear caches
        if self.lazy_loading:
            self._loaded_vectors.clear()
            self._vector_metadata.clear()
            self._vector_counts.clear()

        # Clear buffers
        self._experience_buffers.clear()
        self._metadata_buffers.clear()

        super().close()
        self.logger.info("OptimizedJsonNumpyStorage closed")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'storage_type': 'OptimizedJsonNumpyStorage',
            'base_path': str(self.base_path),
            'lazy_loading': self.lazy_loading,
            'batch_size': self.batch_size,
            'total_agents': 0,
            'total_environments': 0,
            'total_experience_vectors': 0,
            'buffered_vectors': sum(len(buf) for buf in self._experience_buffers.values()),
            'cached_agents': len(self._loaded_vectors)
        }

        try:
            agents_path = self.base_path / "agents"
            if agents_path.exists():
                agent_dirs = [d for d in agents_path.iterdir() if d.is_dir()]
                stats['total_agents'] = len(agent_dirs)

                for agent_dir in agent_dirs:
                    envs_path = agent_dir / "environments"
                    if envs_path.exists():
                        env_dirs = [d for d in envs_path.iterdir() if d.is_dir()]
                        stats['total_environments'] += len(env_dirs)

                    # Count experience vectors
                    exp_path = agent_dir / "experiences"
                    if exp_path.exists():
                        metadata_file = exp_path / "metadata.json"
                        if metadata_file.exists():
                            metadata = self._load_json(metadata_file)
                            if metadata and "total_count" in metadata:
                                stats['total_experience_vectors'] += metadata["total_count"]

        except Exception as e:
            self.logger.error(f"Error calculating storage stats: {e}")

        return stats
