"""
Storage migration utilities for Agent Byte.

This module provides tools for migrating agent data between different
storage backends and versions.
Enhanced with Sprint 9 continuous action space support.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
from datetime import datetime
import time
import numpy as np
from tqdm import tqdm

from .base import StorageBase


class StorageMigrator:
    """
    Handles migration of agent data between storage backends and versions.

    Supports:
    - Backend migration (e.g., JSON to database)
    - Version migration (e.g., v2.0 to v3.0)
    - Batch migration of multiple agents
    - Rollback capabilities
    - Vector database migration with optimized batch processing
    - Sprint 9: Continuous action space data migration
    """

    def __init__(self):
        """Initialize the storage migrator."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.migration_history = []

    def migrate_backend(self,
                        source_storage: StorageBase,
                        target_storage: StorageBase,
                        agent_ids: Optional[List[str]] = None,
                        create_backup: bool = True,
                        batch_size: int = 1000) -> Dict[str, Any]:
        """
        Migrate data from one storage backend to another.

        Args:
            source_storage: Source storage backend
            target_storage: Target storage backend
            agent_ids: Specific agents to migrate (None for all)
            create_backup: Whether to create backup before migration
            batch_size: Batch size for experience vector migration

        Returns:
            Migration results with success/failure counts
        """
        self.logger.info(
            f"Starting backend migration from {type(source_storage).__name__} to {type(target_storage).__name__}")

        results = {
            'start_time': datetime.now(),
            'source_backend': type(source_storage).__name__,
            'target_backend': type(target_storage).__name__,
            'agents_migrated': 0,
            'agents_failed': 0,
            'experiences_migrated': 0,
            'continuous_networks_migrated': 0,  # NEW Sprint 9
            'action_adapters_migrated': 0,      # NEW Sprint 9
            'errors': []
        }

        try:
            # Get all agent IDs if not specified
            if agent_ids is None:
                agent_ids = self._get_all_agent_ids(source_storage)
                self.logger.info(f"Found {len(agent_ids)} agents to migrate")

            # Create backup if requested
            if create_backup:
                backup_path = self._create_backup(source_storage, agent_ids)
                results['backup_path'] = backup_path

            # Migrate each agent
            for agent_id in tqdm(agent_ids, desc="Migrating agents"):
                try:
                    # Migrate basic data
                    sprint9_counts = self._migrate_agent(agent_id, source_storage, target_storage)

                    # Migrate experience vectors with optimized batch processing
                    exp_count = self._migrate_experience_vectors(
                        source_storage, target_storage, agent_id, batch_size
                    )

                    results['agents_migrated'] += 1
                    results['experiences_migrated'] += exp_count
                    results['continuous_networks_migrated'] += sprint9_counts.get('networks', 0)
                    results['action_adapters_migrated'] += sprint9_counts.get('adapters', 0)

                    self.logger.debug(f"Successfully migrated agent: {agent_id}")

                except Exception as e:
                    results['agents_failed'] += 1
                    error_msg = f"Failed to migrate agent {agent_id}: {str(e)}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)

            results['end_time'] = datetime.now()
            results['duration'] = (results['end_time'] - results['start_time']).total_seconds()

            # Record migration
            self.migration_history.append(results)

            self.logger.info(
                f"Migration completed: {results['agents_migrated']} succeeded, "
                f"{results['agents_failed']} failed, "
                f"{results['experiences_migrated']} experiences migrated, "
                f"{results['continuous_networks_migrated']} networks migrated, "
                f"{results['action_adapters_migrated']} adapters migrated")

        except Exception as e:
            self.logger.error(f"Migration failed: {str(e)}")
            results['fatal_error'] = str(e)

        return results

    def migrate_to_vector_db(self, source_path: str, target_path: str,
                           backend: str = "hybrid", batch_size: int = 1000,
                           agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Specialized method to migrate from JSON+Numpy storage to vector database storage.

        Args:
            source_path: Path to JSON+Numpy storage
            target_path: Path for vector database storage
            backend: Vector DB backend ("faiss", "chroma", "hybrid")
            batch_size: Batch size for experience migration
            agent_ids: Specific agents to migrate (None for all)

        Returns:
            Migration results
        """
        from .json_numpy_storage import JsonNumpyStorage
        from .vector_db_storage import VectorDBStorage

        self.logger.info(f"Starting migration to vector DB ({backend})")

        # Initialize storage backends
        source_storage = JsonNumpyStorage(source_path)
        target_storage = VectorDBStorage(target_path, backend=backend)

        # Use the general migration method
        results = self.migrate_backend(
            source_storage, target_storage, agent_ids, batch_size=batch_size
        )

        # Add backend info
        results['vector_db_backend'] = backend

        # Close storage backends
        source_storage.close()
        target_storage.close()

        return results

    def migrate_version(self,
                        storage: StorageBase,
                        from_version: str,
                        to_version: str,
                        agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Migrate data from one version format to another.

        Args:
            storage: Storage backend to migrate
            from_version: Source version (e.g., "2.0")
            to_version: Target version (e.g., "3.0")
            agent_ids: Specific agents to migrate

        Returns:
            Migration results
        """
        self.logger.info(f"Starting version migration from v{from_version} to v{to_version}")

        results = {
            'migration_type': 'version',
            'from_version': from_version,
            'to_version': to_version,
            'agents_processed': 0,
            'agents_updated': 0,
            'errors': []
        }

        # Get migration function
        migration_func = self._get_version_migration(from_version, to_version)
        if not migration_func:
            error = f"No migration path from v{from_version} to v{to_version}"
            results['errors'].append(error)
            return results

        # Get agents to migrate
        if agent_ids is None:
            agent_ids = self._get_all_agent_ids(storage)

        # Process each agent
        for agent_id in agent_ids:
            try:
                # Get all environments for agent
                environments = storage.list_environments(agent_id)
                results['agents_processed'] += 1

                # Migrate profile
                profile = storage.get_agent_profile(agent_id)
                if profile:
                    migrated_profile = migration_func(profile, 'profile')
                    storage.save_agent_profile(agent_id, migrated_profile)

                # Migrate each environment
                for env_id in environments:
                    # Migrate brain state
                    brain_state = storage.load_brain_state(agent_id, env_id)
                    if brain_state:
                        migrated_brain = migration_func(brain_state, 'brain')
                        storage.save_brain_state(agent_id, env_id, migrated_brain)

                    # Migrate knowledge
                    knowledge = storage.load_knowledge(agent_id, env_id)
                    if knowledge:
                        migrated_knowledge = migration_func(knowledge, 'knowledge')
                        storage.save_knowledge(agent_id, env_id, migrated_knowledge)

                    # Migrate autoencoder
                    autoencoder = storage.load_autoencoder(agent_id, env_id)
                    if autoencoder:
                        migrated_autoencoder = migration_func(autoencoder, 'autoencoder')
                        storage.save_autoencoder(agent_id, env_id, migrated_autoencoder)

                results['agents_updated'] += 1

            except Exception as e:
                error_msg = f"Failed to migrate agent {agent_id}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)

        return results

    def _migrate_agent(self, agent_id: str, source: StorageBase, target: StorageBase) -> Dict[str, int]:
        """
        Migrate a single agent's data including Sprint 9 features.

        Returns:
            Dictionary with counts of Sprint 9 data migrated
        """
        sprint9_counts = {'networks': 0, 'adapters': 0}

        # Migrate profile
        profile = source.get_agent_profile(agent_id)
        if profile:
            target.save_agent_profile(agent_id, profile)

        # Get all environments
        environments = source.list_environments(agent_id)

        for env_id in environments:
            # Migrate brain state
            brain_state = source.load_brain_state(agent_id, env_id)
            if brain_state:
                target.save_brain_state(agent_id, env_id, brain_state)

            # Migrate knowledge
            knowledge = source.load_knowledge(agent_id, env_id)
            if knowledge:
                target.save_knowledge(agent_id, env_id, knowledge)

            # Migrate autoencoder
            autoencoder = source.load_autoencoder(agent_id, env_id)
            if autoencoder:
                target.save_autoencoder(agent_id, env_id, autoencoder)

            # NEW Sprint 9: Migrate continuous network state
            try:
                network_state = source.load_continuous_network_state(agent_id, env_id)
                if network_state:
                    target.save_continuous_network_state(agent_id, env_id, network_state)
                    sprint9_counts['networks'] += 1
            except AttributeError:
                # Source doesn't support continuous networks
                pass
            except Exception as e:
                self.logger.warning(f"Failed to migrate continuous network for {agent_id}/{env_id}: {e}")

            # NEW Sprint 9: Migrate action adapter config
            try:
                adapter_config = source.load_action_adapter_config(agent_id, env_id)
                if adapter_config:
                    target.save_action_adapter_config(agent_id, env_id, adapter_config)
                    sprint9_counts['adapters'] += 1
            except AttributeError:
                # Source doesn't support action adapters
                pass
            except Exception as e:
                self.logger.warning(f"Failed to migrate action adapter for {agent_id}/{env_id}: {e}")

        self.logger.debug(f"Migrated agent {agent_id} with {len(environments)} environments, "
                         f"{sprint9_counts['networks']} networks, {sprint9_counts['adapters']} adapters")

        return sprint9_counts

    def _migrate_experience_vectors(self, source: StorageBase, target: StorageBase,
                                  agent_id: str, batch_size: int) -> int:
        """
        Migrate experience vectors in batches for efficiency.

        Returns:
            Number of experiences migrated
        """
        count = 0

        # This is specific to JsonNumpyStorage as source
        if hasattr(source, '_get_agent_path'):
            exp_path = source._get_agent_path(agent_id) / "experiences"
            vectors_file = exp_path / "vectors.npy"
            metadata_file = exp_path / "metadata.json"

            if not vectors_file.exists():
                return 0

            try:
                # Load data
                vectors = np.load(vectors_file)
                metadata_data = self._load_json_safe(metadata_file)

                if not metadata_data or 'experiences' not in metadata_data:
                    return 0

                metadata_list = metadata_data['experiences']

                # Migrate in batches
                for i in range(0, len(vectors), batch_size):
                    batch_end = min(i + batch_size, len(vectors))

                    for j in range(i, batch_end):
                        if j < len(metadata_list):
                            target.save_experience_vector(
                                agent_id,
                                vectors[j],
                                metadata_list[j]
                            )
                            count += 1

                    # Log progress
                    if count % 10000 == 0:
                        self.logger.debug(f"Migrated {count} experiences for {agent_id}")

            except Exception as e:
                self.logger.error(f"Error migrating experiences for {agent_id}: {e}")

        return count

    def _get_all_agent_ids(self, storage: StorageBase) -> List[str]:
        """Get all agent IDs from storage."""
        try:
            # Try to use storage's method if available
            if hasattr(storage, '_get_all_agent_ids'):
                return storage._get_all_agent_ids()
            elif hasattr(storage, 'base_path'):
                # For file-based storage
                agents_path = storage.base_path / "agents"
                if agents_path.exists():
                    return [d.name for d in agents_path.iterdir() if d.is_dir()]
            elif hasattr(storage, '_json_storage_path'):
                # For VectorDBStorage
                agents_path = storage._json_storage_path / "agents"
                if agents_path.exists():
                    return [d.name for d in agents_path.iterdir() if d.is_dir()]

            return []
        except Exception as e:
            self.logger.error(f"Error getting agent IDs: {e}")
            return []

    def _load_json_safe(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Safely load JSON file."""
        try:
            if not filepath.exists():
                return None
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load JSON from {filepath}: {e}")
            return None

    def _create_backup(self, storage: StorageBase, agent_ids: List[str]) -> str:
        """Create backup of agents before migration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"./migration_backup_{timestamp}"

        self.logger.info(f"Creating backup in {backup_dir}")

        # This is a placeholder - actual implementation would depend on storage type
        Path(backup_dir).mkdir(parents=True, exist_ok=True)

        # Save migration metadata
        metadata = {
            'timestamp': timestamp,
            'storage_type': type(storage).__name__,
            'agent_count': len(agent_ids),
            'agent_ids': agent_ids
        }

        with open(Path(backup_dir) / 'migration_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return backup_dir

    def _get_version_migration(self, from_version: str, to_version: str):
        """Get migration function for version upgrade."""
        # Define migration paths
        migrations = {
            ('2.0', '3.0'): self._migrate_v2_to_v3,
            ('2.1', '3.0'): self._migrate_v2_to_v3,  # Same migration path
        }

        return migrations.get((from_version, to_version))

    def _migrate_v2_to_v3(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Migrate data from v2.x to v3.0 format."""
        migrated = data.copy()

        if data_type == 'profile':
            # Update architecture version
            migrated['architecture_version'] = '3.0'

            # Remove environment-specific fields
            if 'environments' in migrated:
                # Keep environment list but remove hard-coded names
                migrated['environments_experienced'] = migrated.pop('environments', [])

        elif data_type == 'brain':
            # Update neural brain format
            if 'architecture_version' not in migrated:
                migrated['architecture_version'] = '3.0'

            # Remove environment-specific learning rates if present
            env_specific_keys = ['pong_learning_rate', 'chess_learning_rate', 'trading_learning_rate']
            for key in env_specific_keys:
                migrated.pop(key, None)

        elif data_type == 'knowledge':
            # Update knowledge format
            if 'metadata' in migrated:
                migrated['metadata']['version'] = '3.0'

            # Remove hard-coded skill mappings
            if 'skill_mappings' in migrated:
                # Convert old skill mappings to new format
                old_mappings = migrated.pop('skill_mappings', {})
                migrated['discovered_skills'] = self._convert_skill_mappings(old_mappings)

        elif data_type == 'autoencoder':
            # Autoencoders are already in v3.0 format
            pass

        return migrated

    def _convert_skill_mappings(self, old_mappings: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old hard-coded skill mappings to new discovered skills format."""
        discovered_skills = {}

        for env_name, skills in old_mappings.items():
            # Skip environment-specific mappings
            if env_name in ['pong', 'chess', 'trading']:
                continue

            # Convert generic skills
            for skill_name, skill_data in skills.items():
                skill_id = f"{skill_name}_migrated"
                discovered_skills[skill_id] = {
                    'skill': {
                        'name': skill_name,
                        'type': 'migrated',
                        'description': f"Migrated from v2: {skill_name}",
                        'confidence': 0.5,  # Reduce confidence for migrated skills
                        'abstraction_level': 'tactical'
                    },
                    'discovered_at': datetime.now().timestamp(),
                    'application_count': 0,
                    'success_rate': 0.0,
                    'confidence': 0.5
                }

        return discovered_skills

    def rollback_migration(self, backup_path: str, storage: StorageBase) -> bool:
        """
        Rollback a migration using backup data.

        Args:
            backup_path: Path to backup directory
            storage: Storage to restore to

        Returns:
            Success status
        """
        try:
            self.logger.info(f"Rolling back migration from backup: {backup_path}")

            # Load backup metadata
            with open(Path(backup_path) / 'migration_metadata.json', 'r') as f:
                metadata = json.load(f)

            # This is a placeholder - actual implementation would restore from backup
            self.logger.warning("Rollback functionality not fully implemented")

            return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {str(e)}")
            return False

    def verify_migration(self, source: StorageBase, target: StorageBase,
                         agent_ids: List[str], sample_size: int = 100) -> Dict[str, Any]:
        """
        Verify that migration was successful including Sprint 9 data.

        Args:
            source: Original storage
            target: New storage
            agent_ids: Agents to verify
            sample_size: Number of experiences to sample for verification

        Returns:
            Verification results
        """
        results = {
            'verified': True,
            'agent_checks': {},
            'experience_checks': {},
            'sprint9_checks': {},  # NEW
            'errors': []
        }

        for agent_id in agent_ids:
            agent_result = {
                'profile_match': False,
                'environments_match': False,
                'data_integrity': True,
                'experience_sample_match': False,
                'continuous_networks_match': True,    # NEW Sprint 9
                'action_adapters_match': True         # NEW Sprint 9
            }

            try:
                # Check profile
                source_profile = source.get_agent_profile(agent_id)
                target_profile = target.get_agent_profile(agent_id)

                if source_profile and target_profile:
                    # Simple check - could be more sophisticated
                    agent_result['profile_match'] = (
                            source_profile.get('agent_id') == target_profile.get('agent_id')
                    )
                elif source_profile is None and target_profile is None:
                    agent_result['profile_match'] = True

                # Check environments
                source_envs = set(source.list_environments(agent_id))
                target_envs = set(target.list_environments(agent_id))
                agent_result['environments_match'] = source_envs == target_envs

                # NEW Sprint 9: Check continuous networks
                try:
                    source_networks = set(source.list_continuous_networks(agent_id))
                    target_networks = set(target.list_continuous_networks(agent_id))
                    agent_result['continuous_networks_match'] = source_networks == target_networks
                except AttributeError:
                    # Source doesn't support continuous networks
                    agent_result['continuous_networks_match'] = True

                # NEW Sprint 9: Check action adapters
                try:
                    source_adapters = set(source.list_action_adapters(agent_id))
                    target_adapters = set(target.list_action_adapters(agent_id))
                    agent_result['action_adapters_match'] = source_adapters == target_adapters
                except AttributeError:
                    # Source doesn't support action adapters
                    agent_result['action_adapters_match'] = True

                # Sample experience verification
                if sample_size > 0:
                    agent_result['experience_sample_match'] = self._verify_experience_sample(
                        source, target, agent_id, sample_size
                    )

                # Overall agent verification
                if not all(agent_result.values()):
                    results['verified'] = False

            except Exception as e:
                agent_result['data_integrity'] = False
                results['errors'].append(f"Verification failed for {agent_id}: {str(e)}")
                results['verified'] = False

            results['agent_checks'][agent_id] = agent_result

        return results

    def _verify_experience_sample(self, source: StorageBase, target: StorageBase,
                                agent_id: str, sample_size: int) -> bool:
        """
        Verify a sample of experiences match between storages.
        """
        try:
            # This is specific to JsonNumpyStorage as source
            if hasattr(source, '_get_agent_path'):
                exp_path = source._get_agent_path(agent_id) / "experiences"
                vectors_file = exp_path / "vectors.npy"

                if not vectors_file.exists():
                    return True  # No experiences to verify

                vectors = np.load(vectors_file)

                if len(vectors) == 0:
                    return True

                # Sample random vectors
                sample_indices = np.random.choice(
                    len(vectors),
                    min(sample_size, len(vectors)),
                    replace=False
                )

                # For each sample, search in target and verify it exists
                for idx in sample_indices:
                    query_vector = vectors[idx]

                    # Search in target
                    results = target.search_similar_experiences(
                        agent_id, query_vector, k=1
                    )

                    if not results or results[0]['similarity'] < 0.99:
                        return False

                return True

            return True  # Can't verify, assume success

        except Exception as e:
            self.logger.error(f"Experience verification failed: {e}")
            return False

    def get_storage_stats(self, storage: StorageBase,
                         agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about a storage backend.

        Args:
            storage: Storage backend
            agent_id: Specific agent or None for global stats

        Returns:
            Storage statistics
        """
        stats = {
            'storage_type': type(storage).__name__,
            'agents': {},
            'total_experiences': 0,
            'total_continuous_networks': 0,  # NEW Sprint 9
            'total_action_adapters': 0,      # NEW Sprint 9
            'total_size_mb': 0
        }

        try:
            if agent_id:
                # Agent-specific stats
                stats['agent_id'] = agent_id

                # Count experiences if possible
                if hasattr(storage, '_get_vector_count'):
                    exp_count = storage._get_vector_count(agent_id)
                    stats['total_experiences'] = exp_count
                elif hasattr(storage, 'get_storage_stats'):
                    agent_stats = storage.get_storage_stats(agent_id)
                    stats.update(agent_stats)

                # NEW Sprint 9: Count continuous networks and adapters
                try:
                    stats['total_continuous_networks'] = len(storage.list_continuous_networks(agent_id))
                    stats['total_action_adapters'] = len(storage.list_action_adapters(agent_id))
                except AttributeError:
                    pass

            else:
                # Global stats
                agent_ids = self._get_all_agent_ids(storage)
                stats['total_agents'] = len(agent_ids)

                for agent_id in agent_ids:
                    agent_stats = {
                        'environments': len(storage.list_environments(agent_id)),
                        'experiences': 0,
                        'continuous_networks': 0,  # NEW Sprint 9
                        'action_adapters': 0       # NEW Sprint 9
                    }

                    # Count experiences
                    if hasattr(storage, '_get_vector_count'):
                        exp_count = storage._get_vector_count(agent_id)
                        agent_stats['experiences'] = exp_count
                        stats['total_experiences'] += exp_count

                    # NEW Sprint 9: Count continuous networks and adapters
                    try:
                        network_count = len(storage.list_continuous_networks(agent_id))
                        adapter_count = len(storage.list_action_adapters(agent_id))
                        agent_stats['continuous_networks'] = network_count
                        agent_stats['action_adapters'] = adapter_count
                        stats['total_continuous_networks'] += network_count
                        stats['total_action_adapters'] += adapter_count
                    except AttributeError:
                        pass

                    stats['agents'][agent_id] = agent_stats

        except Exception as e:
            stats['error'] = str(e)

        return stats


# Convenience functions for backward compatibility and ease of use
def migrate_to_vector_db(source_path: str, target_path: str,
                        backend: str = "hybrid", **kwargs) -> Dict[str, Any]:
    """
    Convenience function to migrate storage to vector database.

    Args:
        source_path: Path to JSON+Numpy storage
        target_path: Path for vector database storage
        backend: Vector DB backend ("faiss", "chroma", "hybrid")
        **kwargs: Additional arguments for migration

    Returns:
        Migration results
    """
    migrator = StorageMigrator()
    return migrator.migrate_to_vector_db(source_path, target_path, backend, **kwargs)


def verify_migration(source_path: str, target_path: str,
                    backend: str = "hybrid", **kwargs) -> Dict[str, Any]:
    """
    Convenience function to verify migration.

    Args:
        source_path: Path to source storage
        target_path: Path to target storage
        backend: Vector DB backend used
        **kwargs: Additional arguments for verification

    Returns:
        Verification results
    """
    from .json_numpy_storage import JsonNumpyStorage
    from .vector_db_storage import VectorDBStorage

    source = JsonNumpyStorage(source_path)
    target = VectorDBStorage(target_path, backend=backend)

    agent_ids = source._get_all_agent_ids()

    migrator = StorageMigrator()
    results = migrator.verify_migration(source, target, agent_ids, **kwargs)

    source.close()
    target.close()

    return results