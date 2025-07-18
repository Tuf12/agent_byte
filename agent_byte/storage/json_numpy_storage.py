"""
JSON+Numpy storage implementation.

This module provides a file-based storage backend using JSON for metadata
and NumPy for vector data. Enhanced with Sprint 9 continuous action space support.
"""

import json
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
        self.lazy_loading = self.config.get('lazy_loading', False)
        self.memory_limit = self.config.get('memory_limit', 1000)


        # Lazy loading state
        self._loaded_vectors = {}
        self._vector_metadata = {}

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

    # Core storage methods

    def save_agent_profile(self, agent_id: str, profile: Dict[str, Any]) -> bool:
        """Save agent profile."""
        agent_path = self._ensure_agent_dir(agent_id)
        profile_file = agent_path / "profile.json"

        # Add timestamp
        profile = self._add_timestamp(profile.copy())

        success = self._save_json(profile_file, profile)
        if success:
            # Clear cache for this profile
            cache_key = f"profile_{agent_id}"
            if self._cache and cache_key in self._cache:
                del self._cache[cache_key]

        return success

    def load_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent profile - primary method matching the interface."""
        cache_key = f"profile_{agent_id}"

        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        agent_path = self._get_agent_path(agent_id)
        profile_file = agent_path / "profile.json"

        profile = self._load_json(profile_file)
        if profile:
            self._put_in_cache(cache_key, profile)

        return profile

    def save_brain_state(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save neural brain state."""
        env_path = self._ensure_env_dir(agent_id, env_id)
        brain_file = env_path / "brain_state.json"

        # Add timestamp
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

        # Add timestamp
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

        # Add timestamp
        state_dict = self._add_timestamp(state_dict.copy())

        return self._save_json(autoencoder_file, state_dict)

    def load_autoencoder(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load autoencoder state."""
        env_path = self._get_env_path(agent_id, env_id)
        autoencoder_file = env_path / "autoencoder.json"

        return self._load_json(autoencoder_file)

    def list_autoencoders(self, agent_id: str) -> List[str]:
        """List all environments with saved autoencoders."""
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

    # NEW: Sprint 9 Continuous Action Space Support

    def save_continuous_network_state(self, agent_id: str, env_id: str,
                                     network_state: Dict[str, Any]) -> bool:
        """Save continuous network state (SAC/DDPG weights and training state)."""
        # Validate network state
        if not self._validate_continuous_network_state(network_state):
            return False

        env_path = self._ensure_env_dir(agent_id, env_id)
        network_file = env_path / "continuous_network.json"

        # Add timestamp
        network_state = self._add_timestamp(network_state.copy())

        success = self._save_json(network_file, network_state)
        if success:
            self.logger.info(f"Saved continuous network state for {agent_id}/{env_id}")

        return success

    def load_continuous_network_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load continuous network state."""
        env_path = self._get_env_path(agent_id, env_id)
        network_file = env_path / "continuous_network.json"

        return self._load_json(network_file)

    def save_action_adapter_config(self, agent_id: str, env_id: str,
                                  adapter_config: Dict[str, Any]) -> bool:
        """Save action adapter configuration."""
        # Validate adapter config
        if not self._validate_action_adapter_config(adapter_config):
            return False

        env_path = self._ensure_env_dir(agent_id, env_id)
        adapter_file = env_path / "action_adapter.json"

        # Add timestamp
        adapter_config = self._add_timestamp(adapter_config.copy())

        success = self._save_json(adapter_file, adapter_config)
        if success:
            self.logger.info(f"Saved action adapter config for {agent_id}/{env_id}")

        return success

    def load_action_adapter_config(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load action adapter configuration."""
        env_path = self._get_env_path(agent_id, env_id)
        adapter_file = env_path / "action_adapter.json"

        return self._load_json(adapter_file)

    def list_continuous_networks(self, agent_id: str) -> List[str]:
        """List all environments with saved continuous networks."""
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

    def list_action_adapters(self, agent_id: str) -> List[str]:
        """List all environments with saved action adapters."""
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

    # Experience vector storage

    def save_experience_vector(self, agent_id: str, vector: np.ndarray,
                             metadata: Dict[str, Any]) -> bool:
        """Save experience vector for similarity search."""
        if not self._validate_vector(vector):
            return False

        try:
            agent_path = self._ensure_agent_dir(agent_id)
            exp_path = agent_path / "experiences"
            exp_path.mkdir(exist_ok=True)

            # Load existing data
            vectors_file = exp_path / "vectors.npy"
            metadata_file = exp_path / "metadata.json"

            if vectors_file.exists():
                existing_vectors = np.load(vectors_file)
                vectors = np.vstack([existing_vectors, vector.reshape(1, -1)])
            else:
                vectors = vector.reshape(1, -1)

            if metadata_file.exists():
                existing_metadata = self._load_json(metadata_file) or {"experiences": []}
            else:
                existing_metadata = {"experiences": []}

            # Add new metadata
            metadata_with_timestamp = {
                **metadata,
                'saved_at': time.time(),
                'index': len(existing_metadata["experiences"])
            }
            existing_metadata["experiences"].append(metadata_with_timestamp)

            # Save updated data
            np.save(vectors_file, vectors.astype(np.float32))
            self._save_json(metadata_file, existing_metadata)

            # Update lazy loading cache if enabled
            if self.lazy_loading:
                self._update_lazy_cache(agent_id, vectors, existing_metadata["experiences"])

            return True

        except Exception as e:
            self.logger.error(f"Failed to save experience vector: {e}")
            return False

    def search_similar_experiences(self, agent_id: str, query_vector: np.ndarray,
                                 k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar experiences using cosine similarity."""
        if not self._validate_vector(query_vector):
            return []

        try:
            if self.lazy_loading:
                return self._search_lazy(agent_id, query_vector, k)
            else:
                return self._search_direct(agent_id, query_vector, k)

        except Exception as e:
            self.logger.error(f"Failed to search experiences: {e}")
            return []

    def _search_direct(self, agent_id: str, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Direct search without lazy loading."""
        agent_path = self._get_agent_path(agent_id)
        exp_path = agent_path / "experiences"

        vectors_file = exp_path / "vectors.npy"
        metadata_file = exp_path / "metadata.json"

        if not vectors_file.exists() or not metadata_file.exists():
            return []

        vectors = np.load(vectors_file)
        metadata_data = self._load_json(metadata_file)

        if vectors.shape[0] == 0 or not metadata_data:
            return []

        # Compute cosine similarities
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []

        similarities = np.dot(vectors, query_vector) / (np.linalg.norm(vectors, axis=1) * query_norm)

        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if idx < len(metadata_data["experiences"]):
                result = {
                    'similarity': float(similarities[idx]),
                    'metadata': metadata_data["experiences"][idx],
                    'vector_index': int(idx)
                }
                results.append(result)

        return results

    def _search_lazy(self, agent_id: str, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search with lazy loading."""
        # Load vectors if not in memory
        if agent_id not in self._loaded_vectors:
            self._load_vectors_lazy(agent_id)

        if agent_id not in self._loaded_vectors:
            return []

        vectors = self._loaded_vectors[agent_id]
        metadata = self._vector_metadata[agent_id]

        if vectors.shape[0] == 0:
            return []

        # Compute similarities
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []

        similarities = np.dot(vectors, query_vector) / (np.linalg.norm(vectors, axis=1) * query_norm)

        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if idx < len(metadata):
                result = {
                    'similarity': float(similarities[idx]),
                    'metadata': metadata[idx],
                    'vector_index': int(idx)
                }
                results.append(result)

        return results

    def _load_vectors_lazy(self, agent_id: str):
        """Load vectors for lazy loading."""
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
                else:
                    self._loaded_vectors[agent_id] = np.empty((0, 256))
                    self._vector_metadata[agent_id] = []
            else:
                self._loaded_vectors[agent_id] = np.empty((0, 256))
                self._vector_metadata[agent_id] = []

        except Exception as e:
            self.logger.error(f"Failed to load vectors lazily: {e}")
            self._loaded_vectors[agent_id] = np.empty((0, 256))
            self._vector_metadata[agent_id] = []

    def _update_lazy_cache(self, agent_id: str, vectors: np.ndarray, metadata: List[Dict]):
        """Update lazy loading cache."""
        # Simple memory management
        if len(self._loaded_vectors) > self.memory_limit:
            # Remove oldest entry
            oldest_agent = next(iter(self._loaded_vectors))
            del self._loaded_vectors[oldest_agent]
            if oldest_agent in self._vector_metadata:
                del self._vector_metadata[oldest_agent]

        self._loaded_vectors[agent_id] = vectors
        self._vector_metadata[agent_id] = metadata

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
        if self.lazy_loading:
            self._loaded_vectors.clear()
            self._vector_metadata.clear()

        super().close()
        self.logger.info("JsonNumpyStorage closed")

    # Additional utility methods

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'storage_type': 'JsonNumpyStorage',
            'base_path': str(self.base_path),
            'lazy_loading': self.lazy_loading,
            'total_agents': 0,
            'total_environments': 0,
            'total_experience_vectors': 0
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
                    exp_path = agent_dir / "experiences" / "vectors.npy"
                    if exp_path.exists():
                        try:
                            vectors = np.load(exp_path)
                            stats['total_experience_vectors'] += len(vectors)
                        except:
                            pass

        except Exception as e:
            self.logger.error(f"Error calculating storage stats: {e}")

        return stats

    def clear_agent_data(self, agent_id: str) -> bool:
        """Clear all data for a specific agent."""
        try:
            agent_path = self._get_agent_path(agent_id)
            if agent_path.exists():
                import shutil
                shutil.rmtree(agent_path)

                # Clear from lazy loading cache
                if agent_id in self._loaded_vectors:
                    del self._loaded_vectors[agent_id]
                if agent_id in self._vector_metadata:
                    del self._vector_metadata[agent_id]

                self.logger.info(f"Cleared all data for agent {agent_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to clear agent data: {e}")
            return False