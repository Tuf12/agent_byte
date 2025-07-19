"""
Vector database storage implementation.

This module provides a vector database-backed storage system for efficient
similarity search. Enhanced with Sprint 9 continuous action space support.
"""

import json
from collections import defaultdict

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import time
import sqlite3

import faiss
print("FAISS GPU available:", faiss.get_num_gpus())

from .base import StorageBase


class VectorDBStorage(StorageBase):
    """
    Vector database storage implementation.

    Uses vector databases (FAISS, ChromaDB, etc.) for efficient similarity search
    while maintaining JSON files for metadata and configuration.
    Enhanced with Sprint 9 continuous action space support.
    """

    def __init__(self, base_path: str, backend: str = "faiss", config: Optional[Dict[str, Any]] = None):
        """
        Initialize vector database storage.

        Args:
            base_path: Base directory for storage
            backend: Vector database backend ("faiss", "chroma", "hybrid")
            config: Storage configuration
        """
        super().__init__(config)
        self._vector_buffers = None
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.backend = backend

        # Initialize vector database
        self._init_vector_db()

        # For metadata storage (using JSON like JsonNumpyStorage)
        self._json_storage_path = self.base_path / "metadata"
        self._json_storage_path.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def _init_vector_db(self):
        """Initialize the vector database backend."""
        try:
            if self.backend == "faiss":
                self._init_faiss()
            elif self.backend == "chroma":
                self._init_chroma()
            elif self.backend == "hybrid":
                self._init_hybrid()
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

        except ImportError as e:
            self.logger.warning(f"Vector database backend '{self.backend}' not available: {e}")
            # Fall back to simple in-memory storage for testing
            self._init_fallback()

    def _init_faiss(self):
        """Initialize FAISS backend (CPU-only for compatibility)."""
        try:
            import faiss
            self.vector_dim = 256

            # Force CPU-only to avoid GPU compatibility issues
            self.index = faiss.IndexFlatIP(self.vector_dim)
            print("FAISS: Using CPU (forced for stability)")

            # Storage for metadata
            self.vector_metadata = {}

            # Load existing index if it exists
            index_file = self.base_path / "faiss_index.bin"
            metadata_file = self.base_path / "faiss_metadata.json"

            if index_file.exists() and metadata_file.exists():
                try:
                    self.index = faiss.read_index(str(index_file))
                    with open(metadata_file, 'r') as f:
                        self.vector_metadata = json.load(f)
                    print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
                except Exception as e:
                    print(f"Failed to load existing index: {e}")

        except ImportError:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

    def _search_hybrid(self, agent_id: str, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search using hybrid backend (FAISS + SQLite)."""
        try:
            if self.index.ntotal == 0:
                return []

            # Search in FAISS for similarity
            search_k = min(k * 3, self.index.ntotal)
            similarities, indices = self.index.search(
                query_vector.reshape(1, -1).astype(np.float32),
                search_k
            )

            # Get metadata from SQLite for matching vectors
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx == -1:
                    break

                # Query SQLite for metadata
                cursor = self.conn.execute(
                    '''SELECT metadata FROM experience_metadata 
                       WHERE agent_id = ? AND vector_index = ?''',
                    (agent_id, int(idx))
                )
                row = cursor.fetchone()

                if row:
                    metadata = json.loads(row[0])
                    results.append({
                        'metadata': metadata,
                        'similarity': float(similarity),
                        'vector_index': int(idx)
                    })

                    if len(results) >= k:
                        break

            return results

        except Exception as e:
            self.logger.error(f"Hybrid search error: {e}")
            return []

    def _init_hybrid(self):
        """Initialize hybrid backend (FAISS + SQLite)."""
        try:
            # Initialize FAISS for vector similarity
            import faiss
            import sqlite3
            self.vector_dim = 256
            self.index = faiss.IndexFlatIP(self.vector_dim)
            self.vector_metadata = {}

            # Initialize SQLite for metadata
            self.db_path = self.base_path / "metadata.db"
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")

            # Create metadata table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS experience_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    vector_index INTEGER NOT NULL,
                    metadata TEXT NOT NULL,
                    saved_at REAL NOT NULL,
                    UNIQUE(agent_id, vector_index)
                )
            ''')
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_vector 
                ON experience_metadata(agent_id, vector_index)
            ''')
            self.conn.commit()

            self.logger.info("Hybrid backend (FAISS + SQLite) initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid backend: {e}")
            # Fall back to FAISS only
            self._init_faiss()

    def _save_vector_hybrid(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save vector using hybrid backend (FAISS + SQLite)."""
        try:
            # Add vector to FAISS index
            self.index.add(vector.reshape(1, -1).astype(np.float32))
            vector_index = self.index.ntotal - 1

            # Store metadata in SQLite
            metadata_json = json.dumps(metadata)
            self.conn.execute(
                '''INSERT INTO experience_metadata 
                   (agent_id, vector_index, metadata, saved_at) 
                   VALUES (?, ?, ?, ?)''',
                (agent_id, vector_index, metadata_json, time.time())
            )
            self.conn.commit()

            return True

        except Exception as e:
            self.logger.error(f"Hybrid save error: {e}")
            return False

    def _init_fallback(self):
        """Initialize fallback in-memory storage."""
        self.vectors_storage = {}
        self.metadata_storage = {}
        self.backend = "fallback"

    # JSON-based metadata methods (similar to JsonNumpyStorage)

    def _get_agent_metadata_path(self, agent_id: str) -> Path:
        """Get the metadata directory path for an agent."""
        return self._json_storage_path / "agents" / agent_id

    def _get_env_metadata_path(self, agent_id: str, env_id: str) -> Path:
        """Get the metadata directory path for an agent's environment."""
        return self._get_agent_metadata_path(agent_id) / "environments" / env_id

    def _ensure_agent_metadata_dir(self, agent_id: str) -> Path:
        """Ensure agent metadata directory exists."""
        agent_path = self._get_agent_metadata_path(agent_id)
        agent_path.mkdir(parents=True, exist_ok=True)
        return agent_path

    def _ensure_env_metadata_dir(self, agent_id: str, env_id: str) -> Path:
        """Ensure environment metadata directory exists."""
        env_path = self._get_env_metadata_path(agent_id, env_id)
        env_path.mkdir(parents=True, exist_ok=True)
        return env_path

    def _load_json_metadata(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Safely load JSON metadata file."""
        try:
            if not filepath.exists():
                return None
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load JSON from {filepath}: {e}")
            return None

    def _save_json_metadata(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Safely save JSON metadata file."""
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
        agent_path = self._ensure_agent_metadata_dir(agent_id)
        profile_file = agent_path / "profile.json"

        # Add timestamp
        profile = self._add_timestamp(profile.copy())

        success = self._save_json_metadata(profile_file, profile)
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

        agent_path = self._get_agent_metadata_path(agent_id)
        profile_file = agent_path / "profile.json"

        profile = self._load_json_metadata(profile_file)
        if profile:
            self._put_in_cache(cache_key, profile)

        return profile

    def save_brain_state(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save neural brain state."""
        env_path = self._ensure_env_metadata_dir(agent_id, env_id)
        brain_file = env_path / "brain_state.json"

        # Add timestamp
        data = self._add_timestamp(data.copy())

        return self._save_json_metadata(brain_file, data)

    def load_brain_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load neural brain state."""
        env_path = self._get_env_metadata_path(agent_id, env_id)
        brain_file = env_path / "brain_state.json"

        return self._load_json_metadata(brain_file)

    def save_knowledge(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save symbolic knowledge."""
        env_path = self._ensure_env_metadata_dir(agent_id, env_id)
        knowledge_file = env_path / "knowledge.json"

        # Add timestamp
        data = self._add_timestamp(data.copy())

        return self._save_json_metadata(knowledge_file, data)

    def load_knowledge(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load symbolic knowledge."""
        env_path = self._get_env_metadata_path(agent_id, env_id)
        knowledge_file = env_path / "knowledge.json"

        return self._load_json_metadata(knowledge_file)

    def save_autoencoder(self, agent_id: str, env_id: str, state_dict: Dict[str, Any]) -> bool:
        """Save autoencoder state."""
        env_path = self._ensure_env_metadata_dir(agent_id, env_id)
        autoencoder_file = env_path / "autoencoder.json"

        # Add timestamp
        state_dict = self._add_timestamp(state_dict.copy())

        return self._save_json_metadata(autoencoder_file, state_dict)

    def load_autoencoder(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load autoencoder state."""
        env_path = self._get_env_metadata_path(agent_id, env_id)
        autoencoder_file = env_path / "autoencoder.json"

        return self._load_json_metadata(autoencoder_file)

    def list_autoencoders(self, agent_id: str) -> List[str]:
        """List all environments with saved autoencoders."""
        agent_path = self._get_agent_metadata_path(agent_id)
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

        env_path = self._ensure_env_metadata_dir(agent_id, env_id)
        network_file = env_path / "continuous_network.json"

        # Add timestamp
        network_state = self._add_timestamp(network_state.copy())

        success = self._save_json_metadata(network_file, network_state)
        if success:
            self.logger.info(f"Saved continuous network state for {agent_id}/{env_id}")

        return success

    def load_continuous_network_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load continuous network state."""
        env_path = self._get_env_metadata_path(agent_id, env_id)
        network_file = env_path / "continuous_network.json"

        return self._load_json_metadata(network_file)

    def save_action_adapter_config(self, agent_id: str, env_id: str,
                                  adapter_config: Dict[str, Any]) -> bool:
        """Save action adapter configuration."""
        # Validate adapter config
        if not self._validate_action_adapter_config(adapter_config):
            return False

        env_path = self._ensure_env_metadata_dir(agent_id, env_id)
        adapter_file = env_path / "action_adapter.json"

        # Add timestamp
        adapter_config = self._add_timestamp(adapter_config.copy())

        success = self._save_json_metadata(adapter_file, adapter_config)
        if success:
            self.logger.info(f"Saved action adapter config for {agent_id}/{env_id}")

        return success

    def load_action_adapter_config(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load action adapter configuration."""
        env_path = self._get_env_metadata_path(agent_id, env_id)
        adapter_file = env_path / "action_adapter.json"

        return self._load_json_metadata(adapter_file)

    def list_continuous_networks(self, agent_id: str) -> List[str]:
        """List all environments with saved continuous networks."""
        agent_path = self._get_agent_metadata_path(agent_id)
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
        agent_path = self._get_agent_metadata_path(agent_id)
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

    # Experience vector storage (vector database specific)

    def save_experience_vector(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save experience vector for similarity search."""
        if not self._validate_vector(vector):
            return False

        try:
            # Normalize vector for cosine similarity
            norm = np.linalg.norm(vector)
            if norm > 0:
                normalized_vector = vector / norm
            else:
                normalized_vector = vector

            if self.backend == "faiss":
                return self._save_vector_faiss(agent_id, normalized_vector, metadata)
            elif self.backend == "chroma":
                return self._save_vector_chroma(agent_id, normalized_vector, metadata)
            elif self.backend == "hybrid":
                return self._save_vector_hybrid(agent_id, normalized_vector, metadata)
            elif self.backend == "fallback":
                return self._save_vector_fallback(agent_id, normalized_vector, metadata)
            else:
                self.logger.error(f"Unsupported backend for vector saving: {self.backend}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to save experience vector: {e}")
            return False

    def flush_all_buffers(self) -> bool:
        """Manually flush all vector buffers."""
        # For non-batching implementations, this is a no-op
        return True

    def _flush_vector_batch(self, agent_id: str) -> bool:
        """Flush vector batch to backend."""
        try:
            vectors = self._vector_buffers[agent_id]
            metadatas = self._metadata_buffers[agent_id]

            if not vectors:
                return True

            if self.backend == "faiss":
                # Batch add to FAISS
                batch_vectors = np.array(vectors).astype(np.float32)
                start_idx = self.index.ntotal
                self.index.add(batch_vectors)

                # Handle metadata in batch
                for i, metadata in enumerate(metadatas):
                    vector_id = f"{agent_id}_{start_idx + i}"
                    self.vector_metadata[vector_id] = {
                        'agent_id': agent_id,
                        'metadata': metadata,
                        'saved_at': time.time()
                    }

            elif self.backend == "hybrid":
                # Batch add to FAISS
                batch_vectors = np.array(vectors).astype(np.float32)
                start_idx = self.index.ntotal
                self.index.add(batch_vectors)

                # Batch insert to SQLite
                batch_data = []
                for i, metadata in enumerate(metadatas):
                    batch_data.append((
                        agent_id,
                        start_idx + i,
                        json.dumps(metadata),
                        time.time()
                    ))

                self.conn.executemany(
                    '''INSERT INTO experience_metadata 
                       (agent_id, vector_index, metadata, saved_at) 
                       VALUES (?, ?, ?, ?)''',
                    batch_data
                )
                self.conn.commit()

            elif self.backend == "chroma":
                # Batch add to ChromaDB
                embeddings = [v.tolist() for v in vectors]
                vector_ids = [f"{agent_id}_{int(time.time() * 1000000)}_{i}" for i in range(len(vectors))]
                metadatas_with_agent = [{
                    'agent_id': agent_id,
                    **metadata,
                    'saved_at': time.time()
                } for metadata in metadatas]

                self.collection.add(
                    embeddings=embeddings,
                    metadatas=metadatas_with_agent,
                    ids=vector_ids
                )

            # Clear buffers
            self._vector_buffers[agent_id].clear()
            self._metadata_buffers[agent_id].clear()

            return True

        except Exception as e:
            self.logger.error(f"Failed to flush vector batch: {e}")
            return False



    def _save_vector_faiss(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save vector using FAISS backend."""
        try:
            # Add vector to index
            self.index.add(vector.reshape(1, -1).astype(np.float32))

            # Store metadata
            vector_id = f"{agent_id}_{self.index.ntotal - 1}"
            self.vector_metadata[vector_id] = {
                'agent_id': agent_id,
                'metadata': metadata,
                'saved_at': time.time()
            }

            # Periodically save index and metadata
            if self.index.ntotal % 100 == 0:
                self._save_faiss_state()

            return True

        except Exception as e:
            self.logger.error(f"FAISS save error: {e}")
            return False

    def _save_vector_chroma(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save vector using ChromaDB backend."""
        try:
            vector_id = f"{agent_id}_{int(time.time() * 1000000)}"  # Unique ID

            self.collection.add(
                embeddings=[vector.tolist()],
                metadatas=[{
                    'agent_id': agent_id,
                    **metadata,
                    'saved_at': time.time()
                }],
                ids=[vector_id]
            )

            return True

        except Exception as e:
            self.logger.error(f"ChromaDB save error: {e}")
            return False

    def _save_vector_fallback(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save vector using fallback in-memory storage."""
        if agent_id not in self.vectors_storage:
            self.vectors_storage[agent_id] = []
            self.metadata_storage[agent_id] = []

        self.vectors_storage[agent_id].append(vector)
        self.metadata_storage[agent_id].append({
            **metadata,
            'saved_at': time.time()
        })

        return True

    def search_similar_experiences(self, agent_id: str, query_vector: np.ndarray,
                                 k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar experiences using vector database."""
        if not self._validate_vector(query_vector):
            return []

        try:
            # Normalize query vector
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                normalized_query = query_vector / norm
            else:
                normalized_query = query_vector

            if self.backend == "faiss":
                return self._search_faiss(agent_id, normalized_query, k)
            elif self.backend == "chroma":
                return self._search_chroma(agent_id, normalized_query, k)
            elif self.backend == "fallback":
                return self._search_fallback(agent_id, normalized_query, k)
            elif self.backend == "hybrid":  # ADD THIS LINE
                return self._search_hybrid(agent_id, normalized_query, k)  # ADD THIS LINE

            else:
                self.logger.error(f"Unsupported backend for search: {self.backend}")
                return []

        except Exception as e:
            self.logger.error(f"Failed to search experiences: {e}")
            return []

    def _search_faiss(self, agent_id: str, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search using FAISS backend."""
        if self.index.ntotal == 0:
            return []

        try:
            # Search in FAISS index
            similarities, indices = self.index.search(
                query_vector.reshape(1, -1).astype(np.float32),
                min(k, self.index.ntotal)
            )

            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= 0:  # Valid index
                    vector_id = f"{agent_id}_{idx}"
                    if vector_id in self.vector_metadata:
                        result = {
                            'similarity': float(sim),
                            'metadata': self.vector_metadata[vector_id]['metadata'],
                            'vector_index': int(idx)
                        }
                        results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"FAISS search error: {e}")
            return []

    def _search_chroma(self, agent_id: str, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search using ChromaDB backend."""
        try:
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=k,
                where={"agent_id": agent_id}
            )

            if not results['metadatas'] or not results['distances']:
                return []

            search_results = []
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                # Convert distance to similarity (ChromaDB returns distances)
                similarity = 1.0 / (1.0 + distance)

                result = {
                    'similarity': similarity,
                    'metadata': {k: v for k, v in metadata.items()
                               if k not in ['agent_id', 'saved_at']},
                    'vector_index': len(search_results)
                }
                search_results.append(result)

            return search_results

        except Exception as e:
            self.logger.error(f"ChromaDB search error: {e}")
            return []

    def _search_fallback(self, agent_id: str, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search using fallback in-memory storage."""
        if agent_id not in self.vectors_storage or not self.vectors_storage[agent_id]:
            return []

        vectors = np.array(self.vectors_storage[agent_id])
        metadata = self.metadata_storage[agent_id]

        # Compute cosine similarities
        similarities = np.dot(vectors, query_vector) / np.linalg.norm(vectors, axis=1)

        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            result = {
                'similarity': float(similarities[idx]),
                'metadata': {k: v for k, v in metadata[idx].items()
                           if k != 'saved_at'},
                'vector_index': int(idx)
            }
            results.append(result)

        return results

    def _save_faiss_state(self):
        """Save FAISS index and metadata to disk."""
        try:
            import faiss

            index_file = self.base_path / "faiss_index.bin"
            metadata_file = self.base_path / "faiss_metadata.json"

            faiss.write_index(self.index, str(index_file))
            with open(metadata_file, 'w') as f:
                json.dump(self.vector_metadata, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save FAISS state: {e}")

    def list_environments(self, agent_id: str) -> List[str]:
        """List all environments for an agent."""
        agent_path = self._get_agent_metadata_path(agent_id)
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
        agents_path = self._json_storage_path / "agents"
        if not agents_path.exists():
            return []

        return [d.name for d in agents_path.iterdir() if d.is_dir()]

    def close(self) -> None:
        """Close storage and clean up."""
        try:
            # Flush any remaining buffers
            if hasattr(self, 'flush_all_buffers'):
                self.flush_all_buffers()

            if self.backend == "faiss":
                self._save_faiss_state()
            elif self.backend == "hybrid":
                self._save_faiss_state()
                if hasattr(self, 'conn'):
                    self.conn.close()
            elif self.backend == "chroma" and hasattr(self, 'chroma_client'):
                pass
            elif self.backend == "fallback":
                self.vectors_storage.clear()
                self.metadata_storage.clear()

        except Exception as e:
            self.logger.error(f"Error during storage close: {e}")

        super().close()
        self.logger.info(f"VectorDBStorage ({self.backend}) closed")

    # Additional utility methods

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'storage_type': 'VectorDBStorage',
            'backend': self.backend,
            'base_path': str(self.base_path),
            'total_agents': 0,
            'total_environments': 0,
            'total_experience_vectors': 0
        }

        try:
            # Count agents and environments from metadata
            agents_path = self._json_storage_path / "agents"
            if agents_path.exists():
                agent_dirs = [d for d in agents_path.iterdir() if d.is_dir()]
                stats['total_agents'] = len(agent_dirs)

                for agent_dir in agent_dirs:
                    envs_path = agent_dir / "environments"
                    if envs_path.exists():
                        env_dirs = [d for d in envs_path.iterdir() if d.is_dir()]
                        stats['total_environments'] += len(env_dirs)

            # Count experience vectors from backend
            if self.backend == "faiss":
                stats['total_experience_vectors'] = self.index.ntotal
            elif self.backend == "chroma":
                stats['total_experience_vectors'] = self.collection.count()
            elif self.backend == "fallback":
                total_vectors = sum(len(vectors) for vectors in self.vectors_storage.values())
                stats['total_experience_vectors'] = total_vectors

        except Exception as e:
            self.logger.error(f"Error calculating storage stats: {e}")

        return stats

    def clear_agent_data(self, agent_id: str) -> bool:
        """Clear all data for a specific agent."""
        try:
            # Clear metadata
            agent_path = self._get_agent_metadata_path(agent_id)
            if agent_path.exists():
                import shutil
                shutil.rmtree(agent_path)

            # Clear vectors from backend
            if self.backend == "faiss":
                # FAISS doesn't support selective deletion easily
                # Would need to rebuild index without agent's vectors
                self.logger.warning(f"FAISS backend doesn't support selective deletion for agent {agent_id}")
            elif self.backend == "chroma":
                try:
                    # Delete all vectors for this agent
                    self.collection.delete(where={"agent_id": agent_id})
                except Exception as e:
                    self.logger.warning(f"ChromaDB deletion error: {e}")
            elif self.backend == "fallback":
                if agent_id in self.vectors_storage:
                    del self.vectors_storage[agent_id]
                if agent_id in self.metadata_storage:
                    del self.metadata_storage[agent_id]

            self.logger.info(f"Cleared data for agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear agent data: {e}")
            return False