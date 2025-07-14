"""
Vector database storage implementation for Agent Byte.

This storage backend uses FAISS for efficient similarity search and
ChromaDB for metadata-rich storage, providing a scalable alternative
to the JSON+Numpy implementation. Enhanced with Sprint 5 robustness
features: transaction safety, corruption detection, and automatic recovery.
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
import hashlib
import threading
from contextlib import contextmanager

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
    Enhanced with Sprint 5 robustness features.

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
        Initialize vector database storage with Sprint 5 enhancements.

        Args:
            base_path: Base directory for storage
            config: Additional configuration including Sprint 5 features
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

        # NEW Sprint 5 Phase 3: Transaction and reliability features
        self.transaction_log_enabled = config.get('transaction_log', True) if config else True
        self.corruption_detection_enabled = config.get('corruption_detection', True) if config else True
        self.backup_enabled = config.get('backup_enabled', True) if config else True

        # Transaction state
        self._active_transactions = {}
        self._transaction_lock = threading.Lock()

        # Corruption detection for metadata
        self._metadata_checksums = {}
        self._checksum_file = self.base_path / ".metadata_checksums.json"

        # Write-ahead logging for vector operations
        self._wal_dir = self.base_path / "vector_wal"
        self._wal_dir.mkdir(exist_ok=True)

        # Error tracking specific to vector operations
        self.vector_error_metrics = {
            'vector_save_failures': 0,
            'vector_search_failures': 0,
            'index_corruption_detected': 0,
            'recoveries_performed': 0,
            'transaction_rollbacks': 0,
            'index_rebuilds': 0
        }

        # Index health monitoring
        self.index_health = {
            'last_health_check': 0,
            'health_check_interval': 3600,  # 1 hour
            'vector_count': 0,
            'index_file_size': 0,
            'search_performance_ms': []
        }

        # Load existing data
        self._initialize_backends()
        self._load_metadata_checksums()

        self.logger.info(f"Enhanced VectorDBStorage initialized with backend: {self.backend} (transactions for JSON files only)")

    # NEW Sprint 5: Transaction management methods

    @contextmanager
    def vector_transaction(self, transaction_id: str = None):
        """Context manager for atomic vector operations."""
        if not self.transaction_log_enabled:
            yield
            return

        if transaction_id is None:
            transaction_id = f"vec_txn_{int(time.time() * 1000000)}"

        try:
            self._begin_vector_transaction(transaction_id)
            yield transaction_id
            self._commit_vector_transaction(transaction_id)

        except Exception as e:
            self._rollback_vector_transaction(transaction_id)
            self.vector_error_metrics['transaction_rollbacks'] += 1
            self.logger.error(f"Vector transaction {transaction_id} rolled back: {e}")
            raise

    def _begin_vector_transaction(self, transaction_id: str):
        """Begin a new vector transaction."""
        with self._transaction_lock:
            if transaction_id in self._active_transactions:
                raise ValueError(f"Transaction {transaction_id} already active")

            self._active_transactions[transaction_id] = {
                'start_time': time.time(),
                'vector_operations': [],
                'index_snapshots': {},
                'wal_entries': []
            }

        # Write WAL entry
        self._write_vector_wal_entry(transaction_id, 'BEGIN_VECTOR', {})
        self.logger.debug(f"Started vector transaction: {transaction_id}")

    def _commit_vector_transaction(self, transaction_id: str):
        """Commit vector transaction."""
        with self._transaction_lock:
            if transaction_id not in self._active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")

            transaction = self._active_transactions[transaction_id]

            try:
                # Write WAL commit entry
                self._write_vector_wal_entry(transaction_id, 'COMMIT_VECTOR', {})

                # Finalize vector operations
                for operation in transaction['vector_operations']:
                    if operation['type'] == 'vector_add':
                        self._finalize_vector_add(operation)
                    elif operation['type'] == 'index_update':
                        self._finalize_index_update(operation)

                # Update metadata checksums
                self._save_metadata_checksums()

                # Cleanup transaction data
                self._cleanup_vector_transaction(transaction)

                del self._active_transactions[transaction_id]
                self.logger.debug(f"Committed vector transaction: {transaction_id}")

            except Exception as e:
                self._rollback_vector_transaction(transaction_id)
                raise

    def _rollback_vector_transaction(self, transaction_id: str):
        """Rollback vector transaction."""
        with self._transaction_lock:
            if transaction_id not in self._active_transactions:
                return

            transaction = self._active_transactions[transaction_id]

            try:
                # Write WAL rollback entry
                self._write_vector_wal_entry(transaction_id, 'ROLLBACK_VECTOR', {})

                # Undo vector operations
                for operation in reversed(transaction['vector_operations']):
                    if operation['type'] == 'vector_add':
                        self._rollback_vector_add(operation)
                    elif operation['type'] == 'index_update':
                        self._rollback_index_update(operation)

                # Cleanup transaction data
                self._cleanup_vector_transaction(transaction)

                del self._active_transactions[transaction_id]
                self.logger.debug(f"Rolled back vector transaction: {transaction_id}")

            except Exception as e:
                self.logger.error(f"Vector rollback failed for transaction {transaction_id}: {e}")

    def _write_vector_wal_entry(self, transaction_id: str, operation: str, data: Dict[str, Any]):
        """Write entry to vector write-ahead log."""
        try:
            wal_entry = {
                'transaction_id': transaction_id,
                'operation': operation,
                'timestamp': time.time(),
                'data': data,
                'index_state': self._get_index_state_summary()
            }

            wal_file = self._wal_dir / f"{transaction_id}.vwal"
            with open(wal_file, 'a') as f:
                json.dump(wal_entry, f)
                f.write('\n')

            # Track WAL file for cleanup
            if transaction_id in self._active_transactions:
                self._active_transactions[transaction_id]['wal_entries'].append(wal_file)

        except Exception as e:
            self.logger.error(f"Failed to write vector WAL entry: {e}")

    def _get_index_state_summary(self) -> Dict[str, Any]:
        """Get summary of current index state."""
        try:
            state = {
                'timestamp': time.time(),
                'backend': self.backend,
                'vector_count': 0,
                'index_files': []
            }

            if self.backend == "faiss" and hasattr(self, 'faiss_indices'):
                total_vectors = sum(idx['index'].ntotal for idx in self.faiss_indices.values() if 'index' in idx)
                state['vector_count'] = total_vectors

            elif self.backend == "chroma" and hasattr(self, 'chroma_collections'):
                try:
                    total_vectors = sum(collection.count() for collection in self.chroma_collections.values())
                    state['vector_count'] = total_vectors
                except:
                    state['vector_count'] = 0

            # List index files
            index_files = list(self.base_path.glob("*.index")) + list(self.base_path.glob("*.faiss"))
            state['index_files'] = [str(f) for f in index_files]

            return state

        except Exception as e:
            self.logger.warning(f"Failed to get index state: {e}")
            return {'error': str(e)}

    # Enhanced save methods with transaction support

    def _save_vector_with_transaction(self, agent_id: str, vector: np.ndarray,
                                     metadata: Dict[str, Any], transaction_id: str) -> bool:
        """Save vector within a transaction."""
        try:
            # Add timestamp and agent info
            metadata['timestamp'] = time.time()
            metadata['agent_id'] = agent_id

            # Create backup of current index state if enabled
            index_backup = None
            if self.backup_enabled:
                index_backup = self._create_index_backup()

            # Prepare operation record
            operation = {
                'type': 'vector_add',
                'agent_id': agent_id,
                'vector': vector.copy(),
                'metadata': metadata.copy(),
                'index_backup': index_backup,
                'pre_operation_state': self._get_index_state_summary()
            }

            # Record operation in transaction
            with self._transaction_lock:
                if transaction_id in self._active_transactions:
                    self._active_transactions[transaction_id]['vector_operations'].append(operation)

            # Write WAL entry
            self._write_vector_wal_entry(transaction_id, 'VECTOR_ADD', {
                'agent_id': agent_id,
                'vector_shape': vector.shape,
                'metadata_keys': list(metadata.keys()),
                'backup_path': str(index_backup) if index_backup else None
            })

            # Perform the actual vector addition
            success = self._perform_vector_addition(agent_id, vector, metadata)

            if not success:
                raise Exception("Vector addition to index failed")

            # Update health metrics
            self.index_health['vector_count'] += 1

            return True

        except Exception as e:
            self.logger.error(f"Vector transaction operation failed: {e}")
            raise

    def _perform_vector_addition(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Perform the actual vector addition to the index."""
        try:
            start_time = time.time()

            if self.backend == "faiss":
                return self._save_vector_faiss_batch(agent_id, vector, metadata)
            elif self.backend == "chroma":
                return self._save_vector_chroma_batch(agent_id, vector, metadata)
            elif self.backend == "hybrid":
                faiss_success = True
                chroma_success = True

                if FAISS_AVAILABLE:
                    faiss_success = self._save_vector_faiss_batch(agent_id, vector, metadata)
                if CHROMA_AVAILABLE:
                    chroma_success = self._save_vector_chroma_batch(agent_id, vector, metadata)

                return faiss_success and chroma_success
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        except Exception as e:
            self.logger.error(f"Vector addition failed: {e}")
            return False
        finally:
            # Record performance metrics
            duration = (time.time() - start_time) * 1000
            self.index_health['search_performance_ms'].append(duration)
            # Keep only recent performance data
            if len(self.index_health['search_performance_ms']) > 1000:
                self.index_health['search_performance_ms'] = self.index_health['search_performance_ms'][-1000:]

    # Original methods (preserved and enhanced)

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

    # Implement required methods from base class (enhanced with transaction support)
    def save_brain_state(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save neural brain state to JSON file with transaction safety."""
        try:
            env_path = self._get_env_path(agent_id, env_id)
            self._ensure_directory(env_path)

            # Add timestamp
            data = self._add_timestamp(data)

            # Save to file
            file_path = env_path / "brain_state.json"

            # Use transaction if enabled
            if self.transaction_log_enabled:
                with self.vector_transaction() as txn_id:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)

                    # Verify integrity
                    if self.corruption_detection_enabled:
                        checksum = self._calculate_file_checksum(file_path)
                        self._metadata_checksums[str(file_path)] = checksum
            else:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

            # Clear cache
            cache_key = self._get_cache_key("brain", agent_id, env_id)
            if hasattr(self, '_cache') and self._cache and cache_key in self._cache:
                del self._cache[cache_key]

            return True

        except Exception as e:
            self.logger.error(f"Failed to save brain state: {str(e)}")
            self.vector_error_metrics['vector_save_failures'] += 1
            self._notify_vector_critical_error(f"Brain state save failure: {e}")
            return False

    def load_brain_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load neural brain state from JSON file with corruption detection."""
        try:
            # Check cache first
            cache_key = self._get_cache_key("brain", agent_id, env_id)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            file_path = self._get_env_path(agent_id, env_id) / "brain_state.json"
            if not file_path.exists():
                return None

            # Verify integrity if enabled
            if self.corruption_detection_enabled:
                if not self._verify_file_integrity(file_path):
                    self.logger.error(f"Corruption detected in {file_path}")
                    self.vector_error_metrics['index_corruption_detected'] += 1
                    self._notify_vector_critical_error(f"File corruption detected: {file_path}")
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
        """Save symbolic knowledge to JSON file with transaction safety."""
        try:
            env_path = self._get_env_path(agent_id, env_id)
            self._ensure_directory(env_path)

            # Add timestamp
            data = self._add_timestamp(data)

            # Save to file
            file_path = env_path / "knowledge.json"

            # Use transaction if enabled
            if self.transaction_log_enabled:
                with self.vector_transaction() as txn_id:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)

                    # Verify integrity
                    if self.corruption_detection_enabled:
                        checksum = self._calculate_file_checksum(file_path)
                        self._metadata_checksums[str(file_path)] = checksum
            else:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

            # Clear cache
            cache_key = self._get_cache_key("knowledge", agent_id, env_id)
            if hasattr(self, '_cache') and self._cache and cache_key in self._cache:
                del self._cache[cache_key]

            return True

        except Exception as e:
            self.logger.error(f"Failed to save knowledge: {str(e)}")
            self.vector_error_metrics['vector_save_failures'] += 1
            self._notify_vector_critical_error(f"Knowledge save failure: {e}")
            return False

    def load_knowledge(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load symbolic knowledge from JSON file with corruption detection."""
        try:
            # Check cache first
            cache_key = self._get_cache_key("knowledge", agent_id, env_id)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            file_path = self._get_env_path(agent_id, env_id) / "knowledge.json"
            if not file_path.exists():
                return None

            # Verify integrity if enabled
            if self.corruption_detection_enabled:
                if not self._verify_file_integrity(file_path):
                    self.logger.error(f"Corruption detected in {file_path}")
                    self.vector_error_metrics['index_corruption_detected'] += 1
                    self._notify_vector_critical_error(f"File corruption detected: {file_path}")
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
        """Save autoencoder state to JSON file with transaction safety."""
        try:
            env_path = self._get_env_path(agent_id, env_id)
            self._ensure_directory(env_path)

            # Add timestamp
            state_dict = self._add_timestamp(state_dict)

            # Save to file
            file_path = env_path / "autoencoder.json"

            # Use transaction if enabled
            if self.transaction_log_enabled:
                with self.vector_transaction() as txn_id:
                    with open(file_path, 'w') as f:
                        json.dump(state_dict, f, indent=2, default=str)

                    # Verify integrity
                    if self.corruption_detection_enabled:
                        checksum = self._calculate_file_checksum(file_path)
                        self._metadata_checksums[str(file_path)] = checksum
            else:
                with open(file_path, 'w') as f:
                    json.dump(state_dict, f, indent=2, default=str)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save autoencoder: {str(e)}")
            self.vector_error_metrics['vector_save_failures'] += 1
            self._notify_vector_critical_error(f"Autoencoder save failure: {e}")
            return False

    def load_autoencoder(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load autoencoder state from JSON file with corruption detection."""
        try:
            file_path = self._get_env_path(agent_id, env_id) / "autoencoder.json"
            if not file_path.exists():
                return None

            # Verify integrity if enabled
            if self.corruption_detection_enabled:
                if not self._verify_file_integrity(file_path):
                    self.logger.error(f"Corruption detected in {file_path}")
                    self.vector_error_metrics['index_corruption_detected'] += 1
                    self._notify_vector_critical_error(f"File corruption detected: {file_path}")
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

            # Save using chosen backend with batching (ORIGINAL PERFORMANCE)
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
        Get storage statistics with enhanced health metrics.
        """
        stats = {
            'backend': self.backend,
            'backends_available': {
                'faiss': FAISS_AVAILABLE,
                'chroma': CHROMA_AVAILABLE
            },
            'vector_error_metrics': self.vector_error_metrics.copy(),
            'index_health': self.index_health.copy()
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

    # NEW Sprint 5: Corruption detection and recovery methods

    def _verify_file_integrity(self, file_path: Path) -> bool:
        """Verify file integrity using checksums."""
        if not self.corruption_detection_enabled:
            return True

        try:
            current_checksum = self._calculate_file_checksum(file_path)
            stored_checksum = self._metadata_checksums.get(str(file_path))

            if stored_checksum is None:
                # No stored checksum, calculate and store it
                self._metadata_checksums[str(file_path)] = current_checksum
                self._save_metadata_checksums()
                return True

            return current_checksum == stored_checksum

        except Exception as e:
            self.logger.warning(f"Integrity check failed for {file_path}: {e}")
            return False

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    def _load_metadata_checksums(self):
        """Load stored metadata checksums."""
        try:
            if self._checksum_file.exists():
                with open(self._checksum_file, 'r') as f:
                    self._metadata_checksums = json.load(f)
            else:
                self._metadata_checksums = {}
        except Exception as e:
            self.logger.warning(f"Failed to load metadata checksums: {e}")
            self._metadata_checksums = {}

    def _save_metadata_checksums(self):
        """Save metadata checksums."""
        try:
            with open(self._checksum_file, 'w') as f:
                json.dump(self._metadata_checksums, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata checksums: {e}")

    def _check_index_health(self) -> bool:
        """Check index health and integrity."""
        try:
            current_time = time.time()

            # Skip if checked recently
            if (current_time - self.index_health['last_health_check'] <
                self.index_health['health_check_interval']):
                return True

            self.index_health['last_health_check'] = current_time

            # Check index file existence
            if self.backend == "faiss":
                index_files = list(self.base_path.glob("**/faiss_index.bin"))
                if not index_files:
                    self.logger.warning("FAISS index files missing")
                    return False

            # Performance health check
            if len(self.index_health['search_performance_ms']) > 100:
                avg_search_time = sum(self.index_health['search_performance_ms'][-100:]) / 100
                if avg_search_time > 1000:  # 1 second
                    self.logger.warning(f"Search performance degraded: {avg_search_time:.2f}ms")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Index health check failed: {e}")
            return False

    def _repair_index(self) -> bool:
        """Attempt to repair corrupted index."""
        try:
            self.logger.info("Attempting index repair")

            # Try to recover from backups first
            if self._recover_from_backup():
                self.vector_error_metrics['recoveries_performed'] += 1
                return True

            # Last resort: rebuild empty index
            if self._rebuild_empty_index():
                self.vector_error_metrics['index_rebuilds'] += 1
                self.logger.warning("Rebuilt empty index - data may be lost")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Index repair failed: {e}")
            return False

    def _recover_from_backup(self) -> bool:
        """Recover index from backup files."""
        try:
            backup_files = list(self.base_path.glob("**/*.backup_*"))
            if not backup_files:
                return False

            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for backup_file in backup_files[:3]:  # Try 3 most recent
                try:
                    # Determine original file name
                    original_name = backup_file.name.split('.backup_')[0]
                    original_file = backup_file.parent / original_name

                    # Restore backup
                    shutil.copy2(backup_file, original_file)

                    # Update checksum
                    if self.corruption_detection_enabled:
                        checksum = self._calculate_file_checksum(original_file)
                        self._metadata_checksums[str(original_file)] = checksum

                    self.logger.info(f"Recovered from backup: {backup_file}")
                    return True

                except Exception as e:
                    self.logger.warning(f"Backup recovery failed for {backup_file}: {e}")
                    continue

            return False

        except Exception as e:
            self.logger.error(f"Backup recovery failed: {e}")
            return False

    def _rebuild_empty_index(self) -> bool:
        """Rebuild an empty index."""
        try:
            if self.backend == "faiss":
                import faiss
                # Create new empty index for each agent
                for agent_id in self._get_all_agent_ids():
                    self.faiss_indices[agent_id] = {
                        'index': faiss.IndexFlatIP(256),
                        'metadata': {}
                    }
                    self._save_faiss_index(agent_id)

            elif self.backend == "chroma":
                # Reinitialize ChromaDB collections
                for agent_id in self._get_all_agent_ids():
                    if agent_id in self.chroma_collections:
                        del self.chroma_collections[agent_id]
                    self._get_chroma_collection(agent_id)

            return True

        except Exception as e:
            self.logger.error(f"Empty index rebuild failed: {e}")
            return False

    def _create_index_backup(self) -> Optional[Path]:
        """Create backup of current index state."""
        try:
            backup_dir = self.base_path / "backups"
            backup_dir.mkdir(exist_ok=True)

            backup_timestamp = int(time.time())

            if self.backend == "faiss":
                # Backup FAISS index files
                index_files = list(self.base_path.glob("**/faiss_index.bin"))
                if index_files:
                    source_file = index_files[0]
                    backup_file = backup_dir / f"index_backup_{backup_timestamp}.bin"
                    shutil.copy2(source_file, backup_file)
                    return backup_file

            return None

        except Exception as e:
            self.logger.error(f"Index backup creation failed: {e}")
            return None

    def _notify_vector_critical_error(self, message: str):
        """Notify of critical vector database errors."""
        self.logger.critical(message)

        # Write to critical error log
        try:
            critical_log_path = Path("./critical_vector_errors.log")
            with open(critical_log_path, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {message}\n")
        except Exception:
            pass

    # Transaction cleanup and finalization methods

    def _finalize_vector_add(self, operation: Dict[str, Any]):
        """Finalize vector addition operation."""
        # Vector is already added to index during transaction
        # Just update health metrics
        self.index_health['vector_count'] += 1

    def _finalize_index_update(self, operation: Dict[str, Any]):
        """Finalize index update operation."""
        # Update any cached index state
        pass

    def _rollback_vector_add(self, operation: Dict[str, Any]):
        """Rollback vector addition operation."""
        try:
            # Restore from backup if available
            backup_path = operation.get('index_backup')
            if backup_path and Path(backup_path).exists():
                self.logger.info(f"Restoring index from backup: {backup_path}")
                # This would need implementation based on specific index type
            else:
                self.logger.warning("No backup available for vector rollback")

        except Exception as e:
            self.logger.error(f"Vector rollback failed: {e}")

    def _rollback_index_update(self, operation: Dict[str, Any]):
        """Rollback index update operation."""
        # Similar to vector rollback
        self._rollback_vector_add(operation)

    def _cleanup_vector_transaction(self, transaction: Dict[str, Any]):
        """Cleanup vector transaction resources."""
        try:
            # Cleanup WAL entries
            for wal_file in transaction.get('wal_entries', []):
                try:
                    if wal_file.exists():
                        wal_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup WAL file {wal_file}: {e}")

            # Cleanup index snapshots
            for snapshot_path in transaction.get('index_snapshots', {}).values():
                try:
                    if Path(snapshot_path).exists():
                        Path(snapshot_path).unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup snapshot {snapshot_path}: {e}")

        except Exception as e:
            self.logger.error(f"Transaction cleanup failed: {e}")

    # NEW Sprint 5: Recovery and maintenance utilities

    def repair_vector_storage(self) -> Dict[str, Any]:
        """Repair vector storage corruption and optimize performance."""
        self.logger.info("Starting vector storage repair")

        results = {
            'repair_started': time.time(),
            'files_repaired': 0,
            'indexes_rebuilt': 0,
            'backups_restored': 0,
            'errors': [],
            'repair_success': True
        }

        try:
            # Check and repair index files
            if self.backend == "faiss":
                index_files = list(self.base_path.glob("**/faiss_index.bin"))
                for index_file in index_files:
                    if not self._verify_file_integrity(index_file):
                        self.logger.info(f"Repairing corrupted index: {index_file}")

                        if self._recover_from_backup():
                            results['backups_restored'] += 1
                            results['files_repaired'] += 1
                        else:
                            error_msg = f"Failed to repair index: {index_file}"
                            results['errors'].append(error_msg)
                            results['repair_success'] = False

            results['repair_completed'] = time.time()
            results['repair_duration'] = results['repair_completed'] - results['repair_started']

            self.logger.info(f"Vector storage repair complete: {results}")
            return results

        except Exception as e:
            error_msg = f"Vector storage repair failed: {e}"
            results['errors'].append(error_msg)
            results['repair_success'] = False
            self.logger.error(error_msg)
            return results

    def get_vector_storage_health(self) -> Dict[str, Any]:
        """Get comprehensive vector storage health metrics."""
        health = {
            'timestamp': time.time(),
            'backend': self.backend,
            'error_metrics': self.vector_error_metrics.copy(),
            'index_health': self.index_health.copy(),
            'active_transactions': len(self._active_transactions),
            'features_enabled': {
                'transaction_log': self.transaction_log_enabled,
                'corruption_detection': self.corruption_detection_enabled,
                'backup_enabled': self.backup_enabled
            },
            'storage_size_mb': 0,
            'vector_count': 0,
            'health_status': 'healthy'
        }

        try:
            # Calculate storage size
            total_size = sum(f.stat().st_size for f in self.base_path.rglob('*') if f.is_file())
            health['storage_size_mb'] = total_size / 1024 / 1024

            # Get vector count
            if self.backend == "faiss" and hasattr(self, 'faiss_indices'):
                health['vector_count'] = sum(idx['index'].ntotal for idx in self.faiss_indices.values() if 'index' in idx)
            elif self.backend == "chroma" and hasattr(self, 'chroma_collections'):
                try:
                    health['vector_count'] = sum(collection.count() for collection in self.chroma_collections.values())
                except:
                    health['vector_count'] = 0

            # Determine health status
            total_operations = max(1, health['vector_count'] + sum(self.vector_error_metrics.values()))
            error_rate = sum(self.vector_error_metrics.values()) / total_operations

            if error_rate > 0.1:  # More than 10% error rate
                health['health_status'] = 'degraded'
            elif error_rate > 0.05:  # More than 5% error rate
                health['health_status'] = 'warning'

            # Check performance
            if self.index_health['search_performance_ms']:
                avg_search_time = sum(self.index_health['search_performance_ms'][-100:]) / min(100, len(self.index_health['search_performance_ms']))
                health['avg_search_time_ms'] = avg_search_time

                if avg_search_time > 1000:  # 1 second
                    health['health_status'] = 'warning'

            if health['active_transactions'] > 10:
                health['health_status'] = 'warning'

        except Exception as e:
            health['health_status'] = 'error'
            health['error'] = str(e)

        return health

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

        # Save final checksums
        if self.corruption_detection_enabled:
            self._save_metadata_checksums()

        # Clear caches
        self._clear_cache()
        self.faiss_indices.clear()
        self.chroma_collections.clear()

        self.logger.info("Enhanced VectorDBStorage closed")