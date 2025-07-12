

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
    │           └── metadata.json
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
        self._load_experience_indices()

        self.logger.info(f"Initialized JsonNumpyStorage at {self.base_path}")

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
        """Save experience vector for similarity search."""
        try:
            if not self._validate_vector(vector):
                return False

            # Ensure experiences directory exists
            exp_path = self._get_agent_path(agent_id) / "experiences"
            self._ensure_directory(exp_path)

            # Load existing vectors and metadata
            vectors, metadatas = self._load_experience_data(agent_id)

            # Add new experience
            metadata['timestamp'] = datetime.now().isoformat()
            metadata['vector_index'] = len(vectors)

            vectors.append(vector)
            metadatas.append(metadata)

            # Save vectors
            np.save(exp_path / "vectors.npy", np.array(vectors))

            # Save metadata
            with open(exp_path / "metadata.json", 'w') as f:
                json.dump(metadatas, f, indent=2, default=str)

            # Update in-memory index
            if agent_id not in self._experience_vectors:
                self._experience_vectors[agent_id] = []

            self._experience_vectors[agent_id].append({
                'vector': vector,
                'metadata': metadata
            })

            # Limit in-memory vectors
            max_vectors = self.config.get('max_vectors_in_memory', 10000)
            if len(self._experience_vectors[agent_id]) > max_vectors:
                self._experience_vectors[agent_id] = self._experience_vectors[agent_id][-max_vectors:]

            return True

        except Exception as e:
            self.logger.error(f"Failed to save experience vector: {str(e)}")
            return False

    def search_similar_experiences(self, agent_id: str, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar experiences using cosine similarity."""
        try:
            if not self._validate_vector(query_vector):
                return []

            # Get experiences for this agent
            if agent_id not in self._experience_vectors:
                # Try to load from disk
                self._load_agent_experiences(agent_id)

            if agent_id not in self._experience_vectors or not self._experience_vectors[agent_id]:
                return []

            # Calculate similarities
            similarities = []
            query_norm = np.linalg.norm(query_vector)

            for exp in self._experience_vectors[agent_id]:
                exp_vector = exp['vector']
                exp_norm = np.linalg.norm(exp_vector)

                if query_norm > 0 and exp_norm > 0:
                    # Cosine similarity
                    similarity = np.dot(query_vector, exp_vector) / (query_norm * exp_norm)
                    similarities.append({
                        'similarity': float(similarity),
                        'metadata': exp['metadata'],
                        'vector': exp_vector
                    })

            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:k]

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
        """Load experience vectors and metadata from disk."""
        exp_path = self._get_agent_path(agent_id) / "experiences"

        vectors = []
        metadatas = []

        # Load vectors
        vectors_file = exp_path / "vectors.npy"
        if vectors_file.exists():
            vectors = np.load(vectors_file).tolist()

        # Load metadata
        metadata_file = exp_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadatas = json.load(f)

        return vectors, metadatas

    def _load_agent_experiences(self, agent_id: str) -> None:
        """Load agent experiences into memory."""
        try:
            vectors, metadatas = self._load_experience_data(agent_id)

            if agent_id not in self._experience_vectors:
                self._experience_vectors[agent_id] = []

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

            # Don't load all vectors on startup - load on demand
            #  register which agents exist
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir():
                    agent_id = agent_dir.name
                    self._experience_vectors[agent_id] = []

        except Exception as e:
            self.logger.error(f"Failed to load experience indices: {str(e)}")

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

            # Clear cache entries for this agent
            cache_keys_to_remove = [k for k in self._cache.keys() if agent_id in k]
            for key in cache_keys_to_remove:
                del self._cache[key]

            self.logger.info(f"Deleted agent: {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete agent: {str(e)}")
            return False