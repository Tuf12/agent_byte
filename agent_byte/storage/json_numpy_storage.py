"""
JSON + Numpy storage implementation for Agent Byte.

This storage backend uses JSON files for structured data and numpy arrays
for vector storage. It's designed to be straightforward but upgradeable to vector
databases in the future. Now includes autoencoder storage support.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
from datetime import datetime
import os

from .base import StorageBase
from ..analysis.autoencoder import VariationalAutoencoder


class JsonNumpyStorage(StorageBase):
    """
    Storage implementation using JSON files and numpy arrays.

    Directory structure:
    base_path/
    ├── agents/
    │   └── {agent_id}/
    │       ├── profile.json
    │       ├── environments/
    │       │   └── {env_id}/
    │       │       ├── brain_state.json
    │       │       ├── knowledge.json
    │       │       └── autoencoder.json
    │       └── experiences/
    │           ├── vectors.npy
    │           ├── metadata.json
    │           └── index.json
    """

    def __init__(self, base_path: str = "./agent_data", config: Optional[Dict[str, Any]] = None):
        """
        Initialize JSON + Numpy storage.

        Args:
            base_path: Base directory for storage
            config: Additional configuration
        """
        super().__init__(config)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Experience vectors stored in memory for fast similarity search
        self._experience_vectors: Dict[str, List[Dict[str, Any]]] = {}

        # Lazy loading configuration
        self.lazy_loading_enabled = config.get('lazy_loading', True) if config else True
        self.memory_limit = config.get('memory_limit', 10000) if config else 10000
        self.batch_size = config.get('batch_size', 1000) if config else 1000

        # Memory-mapped arrays for large datasets
        self._mmap_arrays: Dict[str, np.memmap] = {}

        # Batch write buffers
        self._write_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._buffer_size = config.get('buffer_size', 100) if config else 100

        # Experience indices for fast lookup
        self._experience_indices: Dict[str, Dict[str, int]] = {}

        self._load_experience_indices()

        self.logger.info(f"Initialized JsonNumpyStorage at {self.base_path} (lazy_loading={self.lazy_loading_enabled})")

    def _get_agent_path(self, agent_id: str) -> Path:
        """Get path for agent directory."""
        return self.base_path / "agents" / agent_id

    def _get_env_path(self, agent_id: str, env_id: str) -> Path:
        """Get path for environment directory."""
        return self._get_agent_path(agent_id) / "environments" / env_id

    def _ensure_directory(self, path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    def save_brain_state(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save neural brain state to JSON file."""
        try:
            env_path = self._get_env_path(agent_id, env_id)
            self._ensure_directory(env_path)

            # Add timestamp
            data = self._add_timestamp(data)

            # Save to file
            file_path = env_path / "brain_state.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Clear cache for this entry
            cache_key = self._get_cache_key("brain", agent_id, env_id)
            if cache_key in self._cache:
                del self._cache[cache_key]

            self.logger.debug(f"Saved brain state for {agent_id}/{env_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save brain state: {str(e)}")
            return False

    def load_brain_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load neural brain state from JSON file."""
        try:
            # Check cache first
            cache_key = self._get_cache_key("brain", agent_id, env_id)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            file_path = self._get_env_path(agent_id, env_id) / "brain_state.json"
            if not file_path.exists():
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Cache the result
            self._put_in_cache(cache_key, data)

            return data

        except Exception as e:
            self.logger.error(f"Failed to load brain state: {str(e)}")
            return None

    def save_knowledge(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save symbolic knowledge to JSON file."""
        try:
            env_path = self._get_env_path(agent_id, env_id)
            self._ensure_directory(env_path)

            # Add timestamp
            data = self._add_timestamp(data)

            # Save to file
            file_path = env_path / "knowledge.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Clear cache
            cache_key = self._get_cache_key("knowledge", agent_id, env_id)
            if cache_key in self._cache:
                del self._cache[cache_key]

            self.logger.debug(f"Saved knowledge for {agent_id}/{env_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save knowledge: {str(e)}")
            return False

    def load_knowledge(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load symbolic knowledge from JSON file."""
        try:
            # Check cache first
            cache_key = self._get_cache_key("knowledge", agent_id, env_id)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            file_path = self._get_env_path(agent_id, env_id) / "knowledge.json"
            if not file_path.exists():
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Cache the result
            self._put_in_cache(cache_key, data)

            return data

        except Exception as e:
            self.logger.error(f"Failed to load knowledge: {str(e)}")
            return None

    def save_autoencoder(self, agent_id: str, env_id: str, state_dict: Dict[str, Any]) -> bool:
        """Save autoencoder state to JSON file."""
        try:
            env_path = self._get_env_path(agent_id, env_id)
            self._ensure_directory(env_path)

            # Add timestamp
            state_dict = self._add_timestamp(state_dict)

            # Save to file
            file_path = env_path / "autoencoder.json"
            with open(file_path, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)

            # Clear cache
            cache_key = self._get_cache_key("autoencoder", agent_id, env_id)
            if cache_key in self._cache:
                del self._cache[cache_key]

            self.logger.debug(f"Saved autoencoder for {agent_id}/{env_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save autoencoder: {str(e)}")
            return False

    def load_autoencoder(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load autoencoder state from JSON file."""
        try:
            # Check cache first
            cache_key = self._get_cache_key("autoencoder", agent_id, env_id)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            file_path = self._get_env_path(agent_id, env_id) / "autoencoder.json"
            if not file_path.exists():
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Cache the result
            self._put_in_cache(cache_key, data)

            return data

        except Exception as e:
            self.logger.error(f"Failed to load autoencoder: {str(e)}")
            return None

    def list_autoencoders(self, agent_id: str) -> List[str]:
        """List all environments with saved autoencoders for an agent."""
        try:
            env_dir = self._get_agent_path(agent_id) / "environments"
            if not env_dir.exists():
                return []

            autoencoders = []
            for env_path in env_dir.iterdir():
                if env_path.is_dir():
                    autoencoder_file = env_path / "autoencoder.json"
                    if autoencoder_file.exists():
                        autoencoders.append(env_path.name)

            return autoencoders

        except Exception as e:
            self.logger.error(f"Failed to list autoencoders: {str(e)}")
            return []

    def save_experience_vector(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save experience vector for similarity search with optimized batch writing."""
        try:
            if not self._validate_vector(vector):
                return False

            # Add to write buffer
            if agent_id not in self._write_buffers:
                self._write_buffers[agent_id] = []

            # Add timestamp and index
            metadata['timestamp'] = datetime.now().isoformat()

            self._write_buffers[agent_id].append({
                'vector': vector.copy(),
                'metadata': metadata.copy()
            })

            # Flush buffer if it reaches the size limit
            if len(self._write_buffers[agent_id]) >= self._buffer_size:
                self._flush_write_buffer(agent_id)

            # Update in-memory index for immediate searches
            if agent_id not in self._experience_vectors:
                self._experience_vectors[agent_id] = []

            self._experience_vectors[agent_id].append({
                'vector': vector,
                'metadata': metadata
            })

            # Limit in-memory vectors
            max_vectors = self.config.get('max_vectors_in_memory', 10000)
            if len(self._experience_vectors[agent_id]) > max_vectors:
                # Keep the most recent vectors in memory
                self._experience_vectors[agent_id] = self._experience_vectors[agent_id][-max_vectors:]

            return True

        except Exception as e:
            self.logger.error(f"Failed to save experience vector: {str(e)}")
            return False

    def _flush_write_buffer(self, agent_id: str):
        """Flush write buffer to disk using optimized batch writing."""
        if agent_id not in self._write_buffers or not self._write_buffers[agent_id]:
            return

        try:
            # Ensure experiences directory exists
            exp_path = self._get_agent_path(agent_id) / "experiences"
            self._ensure_directory(exp_path)

            # Load existing data using memory mapping if available
            vectors_file = exp_path / "vectors.npy"
            metadata_file = exp_path / "metadata.json"
            index_file = exp_path / "index.json"

            # Get current index
            if agent_id not in self._experience_indices:
                self._experience_indices[agent_id] = self._load_experience_index(agent_id)

            current_index = self._experience_indices[agent_id].get('count', 0)

            # Prepare batch data
            batch = self._write_buffers[agent_id]
            new_vectors = np.array([item['vector'] for item in batch])
            new_metadata = [item['metadata'] for item in batch]

            # Update metadata with indices
            for i, meta in enumerate(new_metadata):
                meta['vector_index'] = current_index + i

            if vectors_file.exists() and self.lazy_loading_enabled:
                # Use memory mapping for efficient append
                existing_shape = np.load(vectors_file, mmap_mode='r').shape
                total_vectors = existing_shape[0] + len(batch)

                # Create new memory-mapped array
                new_mmap = np.memmap(vectors_file.with_suffix('.tmp'),
                                     dtype='float32',
                                     mode='w+',
                                     shape=(total_vectors, 256))

                # Copy existing data
                existing_mmap = np.memmap(vectors_file, dtype='float32', mode='r', shape=existing_shape)
                new_mmap[:existing_shape[0]] = existing_mmap
                new_mmap[existing_shape[0]:] = new_vectors

                # Flush and close
                del existing_mmap
                del new_mmap

                # Replace old file
                vectors_file.with_suffix('.tmp').replace(vectors_file)
            else:
                # Standard append for smaller datasets
                if vectors_file.exists():
                    existing = np.load(vectors_file)
                    combined = np.vstack([existing, new_vectors])
                    np.save(vectors_file, combined)
                else:
                    np.save(vectors_file, new_vectors)

            # Append metadata efficiently
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                existing_metadata.extend(new_metadata)
            else:
                existing_metadata = new_metadata

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(existing_metadata, f, indent=2, default=str)

            # Update index
            self._experience_indices[agent_id]['count'] = current_index + len(batch)
            with open(index_file, 'w') as f:
                json.dump(self._experience_indices[agent_id], f)

            # Clear buffer
            self._write_buffers[agent_id].clear()

        except Exception as e:
            self.logger.error(f"Failed to flush write buffer: {str(e)}")

    def search_similar_experiences(self, agent_id: str, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar experiences using optimized cosine similarity."""
        try:
            if not self._validate_vector(query_vector):
                return []

            # Ensure write buffer is flushed
            if agent_id in self._write_buffers and self._write_buffers[agent_id]:
                self._flush_write_buffer(agent_id)

            # Get experiences for this agent
            if agent_id not in self._experience_vectors:
                # Try to load from disk with lazy loading
                self._load_agent_experiences(agent_id)

            if agent_id not in self._experience_vectors or not self._experience_vectors[agent_id]:
                return []

            # Normalize query vector for better similarity scores
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector_normalized = query_vector / query_norm
            else:
                return []

            # Vectorized similarity calculation
            vectors = np.array([exp['vector'] for exp in self._experience_vectors[agent_id]])
            metadatas = [exp['metadata'] for exp in self._experience_vectors[agent_id]]

            # Normalize all vectors
            norms = np.linalg.norm(vectors, axis=1)
            valid_mask = norms > 0
            vectors[valid_mask] = vectors[valid_mask] / norms[valid_mask, np.newaxis]

            # Compute similarities in batch
            similarities = np.dot(vectors, query_vector_normalized)

            # Get top k indices
            top_k_indices = np.argpartition(similarities, -min(k, len(similarities)))[-min(k, len(similarities)):]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

            # Build results
            results = []
            for idx in top_k_indices:
                results.append({
                    'similarity': float(similarities[idx]),
                    'metadata': metadatas[idx],
                    'vector': vectors[idx] * norms[idx]  # Denormalize if needed
                })

            return results

        except Exception as e:
            self.logger.error(f"Failed to search experiences: {str(e)}")
            return []

    def list_environments(self, agent_id: str) -> List[str]:
        """List all environments for an agent."""
        try:
            env_dir = self._get_agent_path(agent_id) / "environments"
            if not env_dir.exists():
                return []

            return [d.name for d in env_dir.iterdir() if d.is_dir()]

        except Exception as e:
            self.logger.error(f"Failed to list environments: {str(e)}")
            return []

    def get_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent profile."""
        try:
            profile_path = self._get_agent_path(agent_id) / "profile.json"
            if not profile_path.exists():
                return None

            with open(profile_path, 'r') as f:
                return json.load(f)

        except Exception as e:
            self.logger.error(f"Failed to load agent profile: {str(e)}")
            return None

    def save_agent_profile(self, agent_id: str, profile: Dict[str, Any]) -> bool:
        """Save agent profile."""
        try:
            agent_path = self._get_agent_path(agent_id)
            self._ensure_directory(agent_path)

            # Add timestamp
            profile = self._add_timestamp(profile)

            profile_path = agent_path / "profile.json"
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save agent profile: {str(e)}")
            return False

    def _load_experience_data(self, agent_id: str) -> tuple:
        """Load experience vectors and metadata from disk with lazy loading."""
        exp_path = self._get_agent_path(agent_id) / "experiences"

        vectors = []
        metadatas = []

        # Load vectors
        vectors_file = exp_path / "vectors.npy"
        if vectors_file.exists():
            if self.lazy_loading_enabled and os.path.getsize(vectors_file) > 100 * 1024 * 1024:  # 100MB
                # Use memory mapping for large files
                self._mmap_arrays[agent_id] = np.memmap(vectors_file, dtype='float32', mode='r')
                vectors = self._mmap_arrays[agent_id]
            else:
                vectors = np.load(vectors_file)

        # Load metadata
        metadata_file = exp_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadatas = json.load(f)

        return vectors, metadatas

    def _load_agent_experiences(self, agent_id: str) -> None:
        """Load agent experiences into memory with lazy loading support."""
        try:
            vectors, metadatas = self._load_experience_data(agent_id)

            if agent_id not in self._experience_vectors:
                self._experience_vectors[agent_id] = []

            # For lazy loading, only load the most recent experiences
            if self.lazy_loading_enabled and len(vectors) > self.memory_limit:
                start_idx = len(vectors) - self.memory_limit
                vectors_subset = vectors[start_idx:]
                metadatas_subset = metadatas[start_idx:]

                for i, (vector, metadata) in enumerate(zip(vectors_subset, metadatas_subset)):
                    self._experience_vectors[agent_id].append({
                        'vector': np.array(vector),
                        'metadata': metadata
                    })
            else:
                # Load all if small enough
                for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
                    self._experience_vectors[agent_id].append({
                        'vector': np.array(vector),
                        'metadata': metadata
                    })

        except Exception as e:
            self.logger.error(f"Failed to load agent experiences: {str(e)}")

    def _load_experience_indices(self) -> None:
        """Load experience indices for all agents on startup."""
        try:
            agents_dir = self.base_path / "agents"
            if not agents_dir.exists():
                return

            # Load indices for each agent
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir():
                    agent_id = agent_dir.name
                    self._experience_vectors[agent_id] = []
                    self._experience_indices[agent_id] = self._load_experience_index(agent_id)

        except Exception as e:
            self.logger.error(f"Failed to load experience indices: {str(e)}")

    def _load_experience_index(self, agent_id: str) -> Dict[str, Any]:
        """Load experience index for an agent."""
        index_file = self._get_agent_path(agent_id) / "experiences" / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {'count': 0}

    def _get_all_agent_ids(self) -> List[str]:
        """Get all agent IDs in storage."""
        try:
            agents_dir = self.base_path / "agents"
            if not agents_dir.exists():
                return []

            return [d.name for d in agents_dir.iterdir() if d.is_dir()]

        except Exception as e:
            self.logger.error(f"Failed to get agent IDs: {str(e)}")
            return []

    def delete_agent(self, agent_id: str) -> bool:
        """Delete all data for an agent."""
        try:
            agent_path = self._get_agent_path(agent_id)
            if agent_path.exists():
                shutil.rmtree(agent_path)

            # Remove from memory
            if agent_id in self._experience_vectors:
                del self._experience_vectors[agent_id]

            if agent_id in self._write_buffers:
                del self._write_buffers[agent_id]

            if agent_id in self._experience_indices:
                del self._experience_indices[agent_id]

            if agent_id in self._mmap_arrays:
                del self._mmap_arrays[agent_id]

            # Clear cache entries for this agent
            cache_keys_to_remove = [k for k in self._cache.keys() if agent_id in k]
            for key in cache_keys_to_remove:
                del self._cache[key]

            self.logger.info(f"Deleted agent: {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete agent: {str(e)}")
            return False

    def close(self):
        """Clean up storage resources."""
        # Flush all write buffers
        for agent_id in list(self._write_buffers.keys()):
            if self._write_buffers[agent_id]:
                self._flush_write_buffer(agent_id)

        # Clean up memory-mapped arrays
        self._mmap_arrays.clear()

        # Call parent cleanup
        super().close()