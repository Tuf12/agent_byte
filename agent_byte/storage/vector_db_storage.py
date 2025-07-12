"""
Vector database storage implementation for Agent Byte.

This storage backend uses FAISS for efficient similarity search and
ChromaDB for metadata-rich storage, providing a scalable alternative
to the JSON+Numpy implementation.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import shutil
from datetime import datetime
import logging
import pickle
import uuid
import time

# FAISS imports with proper error handling
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None  # Define faiss as None when not available
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

# ChromaDB imports with proper error handling
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
    CHROMA_IMPORT_ERROR = None
except ImportError as import_error:
    CHROMA_AVAILABLE = False
    chromadb = None  # Define chromadb as None when not available
    Settings = None  # Define Settings as None when not available
    CHROMA_IMPORT_ERROR = str(import_error)
    logging.warning(f"ChromaDB not available: {import_error}. Install with: pip install chromadb")
except Exception as unexpected_error:
    CHROMA_AVAILABLE = False
    chromadb = None
    Settings = None
    CHROMA_IMPORT_ERROR = str(unexpected_error)
    logging.error(f"Unexpected error importing ChromaDB: {unexpected_error}")

from .base import StorageBase


class VectorDBStorage(StorageBase):
    """
    Storage implementation using vector databases for scalable similarity search.

    Uses a hybrid approach:
    - FAISS for fast similarity search on experience vectors
    - ChromaDB for metadata-rich storage with filtering
    - JSON files for structured data (brain states, knowledge)

    This provides the best of both worlds: speed and flexibility.
    """

    def __init__(self, base_path: str = "./agent_data_vectordb",
                 config: Optional[Dict[str, Any]] = None,
                 backend: str = "hybrid"):
        """
        Initialize vector database storage.

        Args:
            base_path: Base directory for storage
            config: Additional configuration
            backend: Backend to use ("faiss", "chroma", "hybrid")
        """
        super().__init__(config)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.backend = backend

        # Validate backend availability
        if self.backend == "faiss" and not FAISS_AVAILABLE:
            raise ImportError("FAISS backend requested but FAISS is not installed. Install with: pip install faiss-cpu or faiss-gpu")
        elif self.backend == "chroma" and not CHROMA_AVAILABLE:
            error_msg = "ChromaDB backend requested but ChromaDB is not available"
            if CHROMA_IMPORT_ERROR:
                error_msg += f": {CHROMA_IMPORT_ERROR}"
            raise ImportError(error_msg)
        elif self.backend == "hybrid":
            if not FAISS_AVAILABLE and not CHROMA_AVAILABLE:
                raise ImportError("Hybrid backend requested but neither FAISS nor ChromaDB is installed")

        # Initialize backends based on availability and choice
        self.faiss_indices: Dict[str, Dict[str, Any]] = {}  # agent_id -> faiss.Index
        self.chroma_collections: Dict[str, Any] = {}  # agent_id -> collection
        self.chroma_client = None

        # Batch write buffers for ChromaDB
        self._chroma_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._buffer_size = config.get('buffer_size', 100) if config else 100

        # FAISS batch operations
        self._faiss_buffers: Dict[str, List[Tuple[np.ndarray, Dict[str, Any]]]] = {}
        self._faiss_save_interval = config.get('faiss_save_interval', 1000) if config else 1000

        self._initialize_backends()

        self.logger.info(f"Initialized VectorDBStorage with backend: {self.backend}")

    def _initialize_backends(self):
        """Initialize the chosen backend(s)."""
        if self.backend in ["faiss", "hybrid"] and FAISS_AVAILABLE:
            # FAISS initialization happens per agent
            pass

        if self.backend in ["chroma", "hybrid"] and CHROMA_AVAILABLE and chromadb is not None:
            # Initialize ChromaDB client with optimized settings
            chroma_path = self.base_path / "chroma"
            chroma_path.mkdir(exist_ok=True)

            try:
                # Try with optimized settings first if Settings is available
                if Settings is not None:
                    self.chroma_client = chromadb.PersistentClient(
                        path=str(chroma_path),
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
                else:
                    # Fall back to basic initialization without Settings
                    self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
                self.logger.info("ChromaDB initialized successfully")
            except Exception as init_error:
                self.logger.error(f"Failed to create ChromaDB client: {init_error}")
                raise

    def _get_agent_path(self, agent_id: str) -> Path:
        """Get path for agent directory."""
        return self.base_path / "agents" / agent_id

    def _get_env_path(self, agent_id: str, env_id: str) -> Path:
        """Get path for environment directory."""
        return self._get_agent_path(agent_id) / "environments" / env_id

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    # Implement required methods from base class
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

            # Clear cache
            cache_key = self._get_cache_key("brain", agent_id, env_id)
            if hasattr(self, '_cache') and self._cache and cache_key in self._cache:
                del self._cache[cache_key]

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
            if hasattr(self, '_cache') and self._cache and cache_key in self._cache:
                del self._cache[cache_key]

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

            return True

        except Exception as e:
            self.logger.error(f"Failed to save autoencoder: {str(e)}")
            return False

    def load_autoencoder(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load autoencoder state from JSON file."""
        try:
            file_path = self._get_env_path(agent_id, env_id) / "autoencoder.json"
            if not file_path.exists():
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

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

    def save_experience_vector(self, agent_id: str, vector: np.ndarray,
                               metadata: Dict[str, Any]) -> bool:
        """
        Save experience vector using a vector database for efficient search.

        Uses FAISS for similarity search and ChromaDB for metadata filtering.
        """
        try:
            if not self._validate_vector(vector):
                return False

            # Add timestamp
            metadata['timestamp'] = datetime.now().isoformat()

            # Save using chosen backend with batching
            if self.backend == "faiss" and FAISS_AVAILABLE:
                return self._save_vector_faiss_batch(agent_id, vector, metadata)
            elif self.backend == "chroma" and CHROMA_AVAILABLE:
                return self._save_vector_chroma_batch(agent_id, vector, metadata)
            elif self.backend == "hybrid":
                # Save to both backends with batching
                faiss_success = True
                chroma_success = True

                if FAISS_AVAILABLE:
                    faiss_success = self._save_vector_faiss_batch(agent_id, vector, metadata)
                if CHROMA_AVAILABLE:
                    chroma_success = self._save_vector_chroma_batch(agent_id, vector, metadata)

                return faiss_success and chroma_success
            else:
                self.logger.error(f"Backend {self.backend} not available")
                return False

        except Exception as e:
            self.logger.error(f"Failed to save experience vector: {str(e)}")
            return False

    def _save_vector_faiss_batch(self, agent_id: str, vector: np.ndarray,
                                  metadata: Dict[str, Any]) -> bool:
        """Save vector to FAISS with batching."""
        try:
            # Add to buffer
            if agent_id not in self._faiss_buffers:
                self._faiss_buffers[agent_id] = []

            self._faiss_buffers[agent_id].append((vector.copy(), metadata.copy()))

            # Initialize FAISS index if needed
            if agent_id not in self.faiss_indices:
                self._initialize_faiss_index(agent_id)

            # Flush buffer if it's full or at save interval
            if len(self._faiss_buffers[agent_id]) >= self._buffer_size:
                self._flush_faiss_buffer(agent_id)

            return True

        except Exception as e:
            self.logger.error(f"FAISS batch save failed: {str(e)}")
            return False

    def _flush_faiss_buffer(self, agent_id: str):
        """Flush FAISS buffer to index."""
        if agent_id not in self._faiss_buffers or not self._faiss_buffers[agent_id]:
            return

        try:
            index_data = self.faiss_indices[agent_id]
            index = index_data['index']

            # Prepare batch data
            vectors = np.array([item[0] for item in self._faiss_buffers[agent_id]], dtype=np.float32)

            # Add vectors in batch
            start_idx = index.ntotal
            index.add(vectors)

            # Store metadata
            for i, (_, metadata) in enumerate(self._faiss_buffers[agent_id]):
                index_data['metadata'][start_idx + i] = metadata

            # Clear buffer
            self._faiss_buffers[agent_id].clear()

            # Persist periodically
            if index.ntotal % self._faiss_save_interval == 0:
                self._save_faiss_index(agent_id)

        except Exception as e:
            self.logger.error(f"Failed to flush FAISS buffer: {str(e)}")

    def _save_vector_chroma_batch(self, agent_id: str, vector: np.ndarray,
                                   metadata: Dict[str, Any]) -> bool:
        """Save vector to ChromaDB with batching."""
        try:
            # Add to buffer
            if agent_id not in self._chroma_buffers:
                self._chroma_buffers[agent_id] = []

            # Generate ID
            vector_id = f"{agent_id}_{metadata['timestamp']}_{np.random.randint(1000000)}"

            self._chroma_buffers[agent_id].append({
                'id': vector_id,
                'embedding': vector.tolist(),
                'metadata': metadata
            })

            # Flush buffer if it's full
            if len(self._chroma_buffers[agent_id]) >= self._buffer_size:
                self._flush_chroma_buffer(agent_id)

            return True

        except Exception as e:
            self.logger.error(f"ChromaDB batch save failed: {str(e)}")
            return False

    def _flush_chroma_buffer(self, agent_id: str):
        """Flush ChromaDB buffer."""
        if agent_id not in self._chroma_buffers or not self._chroma_buffers[agent_id]:
            return

        try:
            collection = self._get_chroma_collection(agent_id)

            # Prepare batch data
            ids = [item['id'] for item in self._chroma_buffers[agent_id]]
            embeddings = [item['embedding'] for item in self._chroma_buffers[agent_id]]
            metadatas = [item['metadata'] for item in self._chroma_buffers[agent_id]]

            # Batch insert
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

            # Clear buffer
            self._chroma_buffers[agent_id].clear()

        except Exception as e:
            self.logger.error(f"Failed to flush ChromaDB buffer: {str(e)}")

    def search_similar_experiences(self, agent_id: str, query_vector: np.ndarray,
                                   k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[
        Dict[str, Any]]:
        """
        Search for similar experiences using vector similarity.

        Supports metadata filtering when using ChromaDB backend.
        """
        try:
            if not self._validate_vector(query_vector):
                return []

            # Flush any pending writes
            if self.backend in ["faiss", "hybrid"] and agent_id in self._faiss_buffers:
                self._flush_faiss_buffer(agent_id)
            if self.backend in ["chroma", "hybrid"] and agent_id in self._chroma_buffers:
                self._flush_chroma_buffer(agent_id)

            results = []

            if self.backend == "faiss" and FAISS_AVAILABLE:
                results = self._search_faiss(agent_id, query_vector, k)
            elif self.backend == "chroma" and CHROMA_AVAILABLE:
                results = self._search_chroma(agent_id, query_vector, k, filter_metadata)
            elif self.backend == "hybrid":
                # Use ChromaDB if we have filters, otherwise FAISS for speed
                if filter_metadata and CHROMA_AVAILABLE:
                    results = self._search_chroma(agent_id, query_vector, k, filter_metadata)
                elif FAISS_AVAILABLE:
                    results = self._search_faiss(agent_id, query_vector, k)
                elif CHROMA_AVAILABLE:
                    results = self._search_chroma(agent_id, query_vector, k, filter_metadata)

            return results

        except Exception as e:
            self.logger.error(f"Failed to search experiences: {str(e)}")
            return []

    def _search_faiss(self, agent_id: str, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search using FAISS index with optimizations."""
        if agent_id not in self.faiss_indices:
            self._load_faiss_index(agent_id)

        if agent_id not in self.faiss_indices:
            return []

        index_data = self.faiss_indices[agent_id]
        index = index_data['index']
        metadata_map = index_data['metadata']

        if index.ntotal == 0:
            return []

        # Normalize query for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_normalized = query_vector / query_norm
        else:
            return []

        # Search using inner product (which equals cosine similarity for normalized vectors)
        query_2d = query_normalized.reshape(1, -1).astype(np.float32)
        k = min(k, index.ntotal)

        # IndexFlatIP returns inner products in descending order (highest similarity first)
        similarities, indices = index.search(query_2d, k)

        # Build results - they're already sorted by similarity
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for not found
                continue

            result = {
                'similarity': float(similarity),  # Inner product = cosine similarity for normalized vectors
                'metadata': metadata_map.get(int(idx), {}),
                'vector': None
            }
            results.append(result)

        return results

    def _search_chroma(self, agent_id: str, query_vector: np.ndarray, k: int,
                       filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search using ChromaDB collection with optimizations."""
        collection = self._get_chroma_collection(agent_id)

        # Normalize query
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_normalized = query_vector / query_norm
        else:
            return []

        # Build where clause for filtering
        where_clause = None
        if filter_metadata:
            where_clause = {}
            for key, value in filter_metadata.items():
                where_clause[key] = {"$eq": value}

        # Query
        results = collection.query(
            query_embeddings=[query_normalized.tolist()],
            n_results=k,
            where=where_clause
        )

        # Build return format
        output = []
        if results['distances'] and results['distances'][0]:
            for i, (dist, metadata) in enumerate(zip(results['distances'][0],
                                                     results['metadatas'][0])):
                # Convert distance to similarity
                similarity = 1.0 / (1.0 + dist)

                output.append({
                    'similarity': float(similarity),
                    'metadata': metadata,
                    'vector': None
                })

        return output

    def _initialize_faiss_index(self, agent_id: str):
        """Initialize FAISS index for an agent with optimizations."""
        if faiss is None:
            self.logger.error("FAISS is not available")
            return

        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        index = faiss.IndexFlatIP(256)  # 256-dimensional vectors

        # For larger datasets, could use IndexIVFFlat for faster search
        # nlist = 100  # number of clusters
        # quantizer = faiss.IndexFlatIP(256)
        # index = faiss.IndexIVFFlat(quantizer, 256, nlist, faiss.METRIC_INNER_PRODUCT)

        # Create metadata storage
        self.faiss_indices[agent_id] = {
            'index': index,
            'metadata': {}
        }

        # Try to load existing index
        self._load_faiss_index(agent_id)

    def _get_chroma_collection(self, agent_id: str):
        """Get or create ChromaDB collection for an agent with optimizations."""
        collection_name = f"agent_{agent_id}_experiences"

        if agent_id not in self.chroma_collections:
            # Create collection with optimized settings
            self.chroma_collections[agent_id] = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "agent_id": agent_id,
                    "hnsw:space": "cosine",  # Use cosine similarity
                    "hnsw:construction_ef": 200,  # Higher = better quality, slower build
                    "hnsw:search_ef": 100  # Higher = better quality, slower search
                }
            )

        return self.chroma_collections[agent_id]

    def _save_faiss_index(self, agent_id: str):
        """Persist FAISS index to disk."""
        if agent_id not in self.faiss_indices or faiss is None:
            return

        agent_path = self._get_agent_path(agent_id)
        self._ensure_directory(agent_path)

        # Save index
        index_path = agent_path / "faiss_index.bin"
        faiss.write_index(self.faiss_indices[agent_id]['index'], str(index_path))

        # Save metadata
        metadata_path = agent_path / "faiss_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.faiss_indices[agent_id]['metadata'], f)

    def _load_faiss_index(self, agent_id: str):
        """Load FAISS index from disk."""
        if faiss is None:
            return

        agent_path = self._get_agent_path(agent_id)
        index_path = agent_path / "faiss_index.bin"
        metadata_path = agent_path / "faiss_metadata.pkl"

        if index_path.exists() and metadata_path.exists():
            # Load index
            index = faiss.read_index(str(index_path))

            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.faiss_indices[agent_id] = {
                'index': index,
                'metadata': metadata
            }

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

    def get_storage_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get storage statistics.

        Args:
            agent_id: Specific agent or None for global stats

        Returns:
            Storage statistics
        """
        stats = {
            'backend': self.backend,
            'backends_available': {
                'faiss': FAISS_AVAILABLE,
                'chroma': CHROMA_AVAILABLE
            }
        }

        if agent_id:
            # Agent-specific stats
            stats['agent_id'] = agent_id

            # Include buffer sizes
            if agent_id in self._faiss_buffers:
                stats['faiss_buffer_size'] = len(self._faiss_buffers[agent_id])
            if agent_id in self._chroma_buffers:
                stats['chroma_buffer_size'] = len(self._chroma_buffers[agent_id])

            if agent_id in self.faiss_indices:
                stats['faiss_vectors'] = self.faiss_indices[agent_id]['index'].ntotal

            if agent_id in self.chroma_collections:
                collection = self.chroma_collections[agent_id]
                stats['chroma_vectors'] = collection.count()
        else:
            # Global stats
            stats['total_agents'] = len(self._get_all_agent_ids())
            stats['loaded_indices'] = len(self.faiss_indices)
            stats['loaded_collections'] = len(self.chroma_collections)

        return stats

    def close(self):
        """Clean up storage resources."""
        # Flush all buffers
        for agent_id in list(self._faiss_buffers.keys()):
            if self._faiss_buffers[agent_id]:
                self._flush_faiss_buffer(agent_id)

        for agent_id in list(self._chroma_buffers.keys()):
            if self._chroma_buffers[agent_id]:
                self._flush_chroma_buffer(agent_id)

        # Save all FAISS indices
        for agent_id in self.faiss_indices:
            self._save_faiss_index(agent_id)

        # Clear caches
        self._clear_cache()
        self.faiss_indices.clear()
        self.chroma_collections.clear()

        self.logger.info("VectorDBStorage closed")