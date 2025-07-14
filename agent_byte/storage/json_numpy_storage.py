"""
JSON + Numpy storage implementation for Agent Byte.

This storage backend uses JSON files for structured data and numpy arrays
for vector storage. It's designed to be straightforward but upgradeable to vector
databases in the future. Now includes autoencoder storage support and Sprint 5
robustness features: transaction safety, corruption detection, and automatic recovery.
"""
import fcntl  # For file locking on Unix systems
import json
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
from datetime import datetime
import os
import time
import hashlib
import tempfile
import threading
from contextlib import contextmanager

from .base import StorageBase
from ..analysis.autoencoder import VariationalAutoencoder


class JsonNumpyStorage(StorageBase):
    """
    Storage implementation using JSON files and numpy arrays.
    Enhanced with Sprint 5 robustness features.

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
    ├── wal/                    # NEW Sprint 5: Write-ahead logging
    ├── .checksums.json         # NEW Sprint 5: File integrity
    └── backups/                # NEW Sprint 5: Automatic backups
    """

    def __init__(self, base_path: str = "./agent_data", config: Optional[Dict[str, Any]] = None):
        """
        Initialize JSON + Numpy storage with Sprint 5 enhancements.

        Args:
            base_path: Base directory for storage
            config: Additional configuration including Sprint 5 features
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

        # NEW Sprint 5 Phase 3: Transaction and reliability features
        self.transaction_log_enabled = config.get('transaction_log', True) if config else True
        self.corruption_detection_enabled = config.get('corruption_detection', True) if config else True
        self.backup_enabled = config.get('backup_enabled', True) if config else True

        # Transaction state
        self._active_transactions = {}
        self._transaction_lock = threading.Lock()

        # Corruption detection
        self._file_checksums = {}
        self._checksum_file = self.base_path / ".checksums.json"

        # Write-ahead logging
        self._wal_dir = self.base_path / "wal"
        self._wal_dir.mkdir(exist_ok=True)

        # Error tracking
        self.error_metrics = {
            'save_failures': 0,
            'load_failures': 0,
            'corruption_detected': 0,
            'recoveries_performed': 0,
            'transaction_rollbacks': 0
        }

        # Load existing data
        self._load_experience_indices()
        self._load_checksums()

        self.logger.info(f"Enhanced JsonNumpyStorage initialized at {self.base_path} (lazy_loading={self.lazy_loading_enabled}, transactions={self.transaction_log_enabled})")

    # NEW Sprint 5: Transaction management methods

    @contextmanager
    def transaction(self, transaction_id: str = None):
        """Context manager for atomic transactions."""
        if not self.transaction_log_enabled:
            yield
            return

        if transaction_id is None:
            transaction_id = f"txn_{int(time.time() * 1000000)}"

        try:
            self._begin_transaction(transaction_id)
            yield transaction_id
            self._commit_transaction(transaction_id)

        except Exception as e:
            self._rollback_transaction(transaction_id)
            self.error_metrics['transaction_rollbacks'] += 1
            self.logger.error(f"Transaction {transaction_id} rolled back: {e}")
            raise

    def _begin_transaction(self, transaction_id: str):
        """Begin a new transaction."""
        with self._transaction_lock:
            if transaction_id in self._active_transactions:
                raise ValueError(f"Transaction {transaction_id} already active")

            self._active_transactions[transaction_id] = {
                'start_time': time.time(),
                'operations': [],
                'temp_files': [],
                'wal_entries': []
            }

        # Write WAL entry
        self._write_wal_entry(transaction_id, 'BEGIN', {})
        self.logger.debug(f"Started transaction: {transaction_id}")

    def _commit_transaction(self, transaction_id: str):
        """Commit transaction by finalizing all operations."""
        with self._transaction_lock:
            if transaction_id not in self._active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")

            transaction = self._active_transactions[transaction_id]

            try:
                # Write WAL commit entry
                self._write_wal_entry(transaction_id, 'COMMIT', {})

                # Finalize all operations
                for operation in transaction['operations']:
                    if operation['type'] == 'file_write':
                        self._finalize_file_write(operation)
                    elif operation['type'] == 'file_delete':
                        self._finalize_file_delete(operation)

                # Update checksums
                self._save_checksums()

                # Cleanup temp files
                for temp_file in transaction['temp_files']:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")

                # Cleanup WAL entries
                self._cleanup_wal_entries(transaction['wal_entries'])

                del self._active_transactions[transaction_id]
                self.logger.debug(f"Committed transaction: {transaction_id}")

            except Exception as e:
                self._rollback_transaction(transaction_id)
                raise

    def _rollback_transaction(self, transaction_id: str):
        """Rollback transaction by undoing all operations."""
        with self._transaction_lock:
            if transaction_id not in self._active_transactions:
                return

            transaction = self._active_transactions[transaction_id]

            try:
                # Write WAL rollback entry
                self._write_wal_entry(transaction_id, 'ROLLBACK', {})

                # Undo operations in reverse order
                for operation in reversed(transaction['operations']):
                    if operation['type'] == 'file_write':
                        self._rollback_file_write(operation)
                    elif operation['type'] == 'file_delete':
                        self._rollback_file_delete(operation)

                # Cleanup temp files
                for temp_file in transaction['temp_files']:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception:
                        pass

                # Cleanup WAL entries
                self._cleanup_wal_entries(transaction['wal_entries'])

                del self._active_transactions[transaction_id]
                self.logger.debug(f"Rolled back transaction: {transaction_id}")

            except Exception as e:
                self.logger.error(f"Rollback failed for transaction {transaction_id}: {e}")

    def _write_wal_entry(self, transaction_id: str, operation: str, data: Dict[str, Any]):
        """Write entry to write-ahead log."""
        try:
            wal_entry = {
                'transaction_id': transaction_id,
                'operation': operation,
                'timestamp': time.time(),
                'data': data
            }

            wal_file = self._wal_dir / f"{transaction_id}.wal"
            with open(wal_file, 'a') as f:
                json.dump(wal_entry, f)
                f.write('\n')

            # Track WAL file for cleanup
            if transaction_id in self._active_transactions:
                self._active_transactions[transaction_id]['wal_entries'].append(wal_file)

        except Exception as e:
            self.logger.error(f"Failed to write WAL entry: {e}")

    def _cleanup_wal_entries(self, wal_files: List[Path]):
        """Cleanup WAL files after transaction completion."""
        for wal_file in wal_files:
            try:
                if wal_file.exists():
                    wal_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup WAL file {wal_file}: {e}")

    def _finalize_file_write(self, operation: Dict[str, Any]):
        """Finalize a file write operation."""
        temp_path = operation['temp_path']
        final_path = operation['final_path']

        # Atomic move
        temp_path.replace(final_path)

        # Update checksum
        if self.corruption_detection_enabled:
            checksum = self._calculate_file_checksum(final_path)
            self._file_checksums[str(final_path)] = checksum

    def _finalize_file_delete(self, operation: Dict[str, Any]):
        """Finalize a file delete operation."""
        file_path = operation['file_path']

        if file_path.exists():
            file_path.unlink()

        # Remove checksum
        if self.corruption_detection_enabled:
            self._file_checksums.pop(str(file_path), None)

    def _rollback_file_write(self, operation: Dict[str, Any]):
        """Rollback a file write operation."""
        final_path = operation['final_path']
        backup_path = operation.get('backup_path')

        try:
            # Remove the target file if it exists
            if final_path.exists():
                final_path.unlink()

            # Restore backup if it exists
            if backup_path and backup_path.exists():
                backup_path.replace(final_path)

        except Exception as e:
            self.logger.error(f"Failed to rollback file operation {final_path}: {e}")

    def _rollback_file_delete(self, operation: Dict[str, Any]):
        """Rollback a file delete operation."""
        file_path = operation['file_path']
        backup_path = operation.get('backup_path')

        # Restore from backup if available
        if backup_path and backup_path.exists():
            backup_path.replace(file_path)

    # Enhanced save methods with transaction support

    def _save_with_transaction(self, file_path: Path, data: Dict[str, Any], transaction_id: str) -> bool:
        """Save data within a transaction."""
        self._ensure_directory(file_path.parent)

        # Add timestamp
        data = self._add_timestamp(data)

        # Create backup if file exists
        backup_path = None
        if file_path.exists() and self.backup_enabled:
            backup_path = file_path.with_suffix(f'.backup_{int(time.time())}')
            shutil.copy2(file_path, backup_path)

        # Write to temporary file first
        temp_path = file_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Verify the written file
            if self.corruption_detection_enabled:
                self._verify_json_file(temp_path)

            # Record operation in transaction
            operation = {
                'type': 'file_write',
                'temp_path': temp_path,
                'final_path': file_path,
                'backup_path': backup_path
            }

            with self._transaction_lock:
                if transaction_id in self._active_transactions:
                    self._active_transactions[transaction_id]['operations'].append(operation)
                    if backup_path:
                        self._active_transactions[transaction_id]['temp_files'].append(backup_path)

            # Write WAL entry
            self._write_wal_entry(transaction_id, 'WRITE', {
                'file_path': str(file_path),
                'temp_path': str(temp_path),
                'backup_path': str(backup_path) if backup_path else None
            })

            # Clear cache for this entry
            cache_key = self._get_cache_key("file", str(file_path))
            if self._cache and cache_key in self._cache:
                del self._cache[cache_key]

            return True

        except Exception as e:
            # Cleanup on error
            if temp_path.exists():
                temp_path.unlink()
            if backup_path and backup_path.exists():
                backup_path.unlink()
            raise

    # Original methods enhanced with transaction support

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
        """Save neural brain state to JSON file with transaction safety."""
        try:
            if self.transaction_log_enabled:
                with self.transaction() as txn_id:
                    return self._save_with_transaction(
                        self._get_env_path(agent_id, env_id) / "brain_state.json",
                        data,
                        txn_id
                    )
            else:
                # Fallback to original method
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
            self.error_metrics['save_failures'] += 1
            self._notify_critical_error(f"Brain state save failure: {e}")
            return False

    def load_brain_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load neural brain state from JSON file with corruption detection."""
        return self._load_with_verification(
            self._get_env_path(agent_id, env_id) / "brain_state.json",
            "brain", agent_id, env_id
        )

    def save_knowledge(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save symbolic knowledge to JSON file with transaction safety."""
        try:
            if self.transaction_log_enabled:
                with self.transaction() as txn_id:
                    return self._save_with_transaction(
                        self._get_env_path(agent_id, env_id) / "knowledge.json",
                        data,
                        txn_id
                    )
            else:
                # Fallback to original method
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
            self.error_metrics['save_failures'] += 1
            self._notify_critical_error(f"Knowledge save failure: {e}")
            return False

    def load_knowledge(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load symbolic knowledge from JSON file with corruption detection."""
        return self._load_with_verification(
            self._get_env_path(agent_id, env_id) / "knowledge.json",
            "knowledge", agent_id, env_id
        )

    def save_autoencoder(self, agent_id: str, env_id: str, state_dict: Dict[str, Any]) -> bool:
        """Save autoencoder state to JSON file with transaction safety."""
        try:
            if self.transaction_log_enabled:
                with self.transaction() as txn_id:
                    return self._save_with_transaction(
                        self._get_env_path(agent_id, env_id) / "autoencoder.json",
                        state_dict,
                        txn_id
                    )
            else:
                # Fallback to original method
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
            self.error_metrics['save_failures'] += 1
            self._notify_critical_error(f"Autoencoder save failure: {e}")
            return False

    def load_autoencoder(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load autoencoder state from JSON file with corruption detection."""
        return self._load_with_verification(
            self._get_env_path(agent_id, env_id) / "autoencoder.json",
            "autoencoder", agent_id, env_id
        )

    def _load_with_verification(self, file_path: Path, data_type: str,
                              agent_id: str = None, env_id: str = None) -> Optional[Dict[str, Any]]:
        """Load data with corruption detection and recovery."""
        try:
            # Check cache first
            if agent_id and env_id:
                cache_key = self._get_cache_key(data_type, agent_id, env_id)
            else:
                cache_key = self._get_cache_key(data_type, str(file_path))

            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            if not file_path.exists():
                return None

            # Verify file integrity if enabled
            if self.corruption_detection_enabled:
                if not self._verify_file_integrity(file_path):
                    self.logger.error(f"Corruption detected in {file_path}")
                    self.error_metrics['corruption_detected'] += 1
                    self._notify_critical_error(f"File corruption detected: {file_path}")

                    # Attempt recovery
                    recovered_data = self._attempt_file_recovery(file_path)
                    if recovered_data:
                        self.error_metrics['recoveries_performed'] += 1
                        return recovered_data
                    else:
                        return None

            # Load and verify JSON structure
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Additional JSON structure verification
            self._verify_json_file(file_path)

            # Cache the result
            self._put_in_cache(cache_key, data)

            return data

        except Exception as e:
            self.logger.error(f"Failed to load {data_type} from {file_path}: {e}")
            self.error_metrics['load_failures'] += 1

            # Attempt recovery
            recovered_data = self._attempt_file_recovery(file_path)
            if recovered_data:
                self.error_metrics['recoveries_performed'] += 1
                return recovered_data

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

    # NEW Sprint 5: Corruption detection and recovery methods

    def _verify_file_integrity(self, file_path: Path) -> bool:
        """Verify file integrity using checksums."""
        if not self.corruption_detection_enabled:
            return True

        try:
            current_checksum = self._calculate_file_checksum(file_path)
            stored_checksum = self._file_checksums.get(str(file_path))

            if stored_checksum is None:
                # No stored checksum, calculate and store it
                self._file_checksums[str(file_path)] = current_checksum
                self._save_checksums()
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

    def _verify_json_file(self, file_path: Path):
        """Verify JSON file structure and content."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Basic structure checks
            if not isinstance(data, dict):
                raise ValueError("JSON root must be a dictionary")

            # Check for required timestamp
            if '_saved_at' not in data:
                self.logger.warning(f"File {file_path} missing timestamp")

            return True

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"JSON verification failed for {file_path}: {e}")

    def _attempt_file_recovery(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Attempt to recover from file corruption."""
        self.logger.info(f"Attempting recovery for {file_path}")

        # Try to find backup files
        backup_pattern = f"{file_path.stem}.backup_*"
        backup_files = list(file_path.parent.glob(backup_pattern))

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for backup_file in backup_files[:3]:  # Try 3 most recent backups
            try:
                self.logger.info(f"Trying backup: {backup_file}")

                # Verify backup integrity
                if self._verify_file_integrity(backup_file):
                    with open(backup_file, 'r') as f:
                        data = json.load(f)

                    # Copy backup to original location
                    shutil.copy2(backup_file, file_path)

                    # Update checksum
                    if self.corruption_detection_enabled:
                        checksum = self._calculate_file_checksum(file_path)
                        self._file_checksums[str(file_path)] = checksum
                        self._save_checksums()

                    self.logger.info(f"Successfully recovered {file_path} from {backup_file}")
                    return data

            except Exception as e:
                self.logger.warning(f"Backup {backup_file} also corrupted: {e}")
                continue

        # Try to recover from WAL if available
        recovered_data = self._recover_from_wal(file_path)
        if recovered_data:
            return recovered_data

        self.logger.error(f"All recovery attempts failed for {file_path}")
        return None

    def _recover_from_wal(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Attempt recovery from write-ahead log."""
        try:
            # Look for recent WAL files that might contain this file
            wal_files = list(self._wal_dir.glob("*.wal"))
            wal_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for wal_file in wal_files[:10]:  # Check 10 most recent WAL files
                try:
                    with open(wal_file, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                if (entry.get('operation') == 'WRITE' and
                                    entry.get('data', {}).get('file_path') == str(file_path)):

                                    # Found a write operation for this file
                                    backup_path = entry.get('data', {}).get('backup_path')
                                    if backup_path and Path(backup_path).exists():
                                        self.logger.info(f"Recovering from WAL backup: {backup_path}")
                                        with open(backup_path, 'r') as bf:
                                            return json.load(bf)

                            except json.JSONDecodeError:
                                continue

                except Exception as e:
                    self.logger.warning(f"Failed to read WAL file {wal_file}: {e}")
                    continue

            return None

        except Exception as e:
            self.logger.error(f"WAL recovery failed: {e}")
            return None

    def _load_checksums(self):
        """Load stored checksums from file."""
        try:
            if self._checksum_file.exists():
                with open(self._checksum_file, 'r') as f:
                    self._file_checksums = json.load(f)
            else:
                self._file_checksums = {}

        except Exception as e:
            self.logger.warning(f"Failed to load checksums: {e}")
            self._file_checksums = {}

    def _save_checksums(self):
        """Save checksums to file."""
        try:
            with open(self._checksum_file, 'w') as f:
                json.dump(self._file_checksums, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save checksums: {e}")

    def _notify_critical_error(self, message: str):
        """Notify user of critical errors."""
        self.logger.critical(message)

        # Write to critical error log
        try:
            critical_log_path = Path("./critical_storage_errors.log")
            with open(critical_log_path, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {message}\n")
        except Exception:
            pass

    # NEW Sprint 5: Recovery and maintenance utilities

    def repair_corruption(self, agent_id: str = None) -> Dict[str, Any]:
        """Scan for and repair file corruption."""
        self.logger.info("Starting corruption repair scan")

        results = {
            'files_scanned': 0,
            'corrupted_files': 0,
            'repaired_files': 0,
            'failed_repairs': 0,
            'errors': []
        }

        try:
            # Determine scope
            if agent_id:
                scan_paths = [self._get_agent_path(agent_id)]
            else:
                agents_dir = self.base_path / "agents"
                scan_paths = [p for p in agents_dir.iterdir() if p.is_dir()] if agents_dir.exists() else []

            for agent_path in scan_paths:
                if not agent_path.is_dir():
                    continue

                # Scan all JSON files
                json_files = list(agent_path.rglob("*.json"))

                for json_file in json_files:
                    results['files_scanned'] += 1

                    try:
                        # Check integrity
                        if not self._verify_file_integrity(json_file):
                            results['corrupted_files'] += 1
                            self.logger.warning(f"Corrupted file detected: {json_file}")

                            # Attempt repair
                            recovered_data = self._attempt_file_recovery(json_file)
                            if recovered_data:
                                results['repaired_files'] += 1
                                self.logger.info(f"Successfully repaired: {json_file}")
                            else:
                                results['failed_repairs'] += 1
                                self.logger.error(f"Failed to repair: {json_file}")

                    except Exception as e:
                        error_msg = f"Error scanning {json_file}: {e}"
                        results['errors'].append(error_msg)
                        self.logger.error(error_msg)

            self.logger.info(f"Repair scan complete: {results}")
            return results

        except Exception as e:
            error_msg = f"Repair scan failed: {e}"
            results['errors'].append(error_msg)
            self.logger.error(error_msg)
            return results

    def get_storage_health(self) -> Dict[str, Any]:
        """Get comprehensive storage health metrics."""
        health = {
            'timestamp': time.time(),
            'error_metrics': self.error_metrics.copy(),
            'active_transactions': len(self._active_transactions),
            'features_enabled': {
                'transaction_log': self.transaction_log_enabled,
                'corruption_detection': self.corruption_detection_enabled,
                'backup_enabled': self.backup_enabled,
                'lazy_loading': self.lazy_loading_enabled
            },
            'storage_size_mb': 0,
            'file_count': 0,
            'checksum_count': len(self._file_checksums),
            'health_status': 'healthy'
        }

        try:
            # Calculate storage size
            total_size = sum(f.stat().st_size for f in self.base_path.rglob('*') if f.is_file())
            health['storage_size_mb'] = total_size / 1024 / 1024

            # Count files
            health['file_count'] = len(list(self.base_path.rglob('*.json')))

            # Determine health status
            error_rate = sum(self.error_metrics.values()) / max(1, health['file_count'])
            if error_rate > 0.1:  # More than 10% error rate
                health['health_status'] = 'degraded'
            elif error_rate > 0.05:  # More than 5% error rate
                health['health_status'] = 'warning'

            if health['active_transactions'] > 10:
                health['health_status'] = 'warning'

        except Exception as e:
            health['health_status'] = 'error'
            health['error'] = str(e)

        return health

    # Original methods (unchanged)

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
            if self._cache:
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

        # Save final checksums
        if self.corruption_detection_enabled:
            self._save_checksums()

        # Call parent cleanup
        super().close()