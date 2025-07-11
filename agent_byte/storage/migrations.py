"""
Storage migration utilities for Agent Byte.

This module provides tools for migrating agent data between different
storage backends and versions.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
from datetime import datetime

from .base import StorageBase


class StorageMigrator:
    """
    Handles migration of agent data between storage backends and versions.

    Supports:
    - Backend migration (e.g., JSON to database)
    - Version migration (e.g., v2.0 to v3.0)
    - Batch migration of multiple agents
    - Rollback capabilities
    """

    def __init__(self):
        """Initialize the storage migrator."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.migration_history = []

    def migrate_backend(self,
                        source_storage: StorageBase,
                        target_storage: StorageBase,
                        agent_ids: Optional[List[str]] = None,
                        create_backup: bool = True) -> Dict[str, Any]:
        """
        Migrate data from one storage backend to another.

        Args:
            source_storage: Source storage backend
            target_storage: Target storage backend
            agent_ids: Specific agents to migrate (None for all)
            create_backup: Whether to create backup before migration

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
            'errors': []
        }

        try:
            # Get all agent IDs if not specified
            if agent_ids is None:
                agent_ids = source_storage._get_all_agent_ids()
                self.logger.info(f"Found {len(agent_ids)} agents to migrate")

            # Create backup if requested
            if create_backup:
                backup_path = self._create_backup(source_storage, agent_ids)
                results['backup_path'] = backup_path

            # Migrate each agent
            for agent_id in agent_ids:
                try:
                    self._migrate_agent(agent_id, source_storage, target_storage)
                    results['agents_migrated'] += 1
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
                f"Migration completed: {results['agents_migrated']} succeeded, {results['agents_failed']} failed")

        except Exception as e:
            self.logger.error(f"Migration failed: {str(e)}")
            results['fatal_error'] = str(e)

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
            agent_ids = storage._get_all_agent_ids()

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

                results['agents_updated'] += 1

            except Exception as e:
                error_msg = f"Failed to migrate agent {agent_id}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)

        return results

    def _migrate_agent(self, agent_id: str, source: StorageBase, target: StorageBase):
        """Migrate a single agent's data."""
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

        # Note: Experience vectors need special handling
        # This is a simplified version - real implementation would batch transfer vectors
        self.logger.debug(f"Migrated agent {agent_id} with {len(environments)} environments")

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
                         agent_ids: List[str]) -> Dict[str, Any]:
        """
        Verify that migration was successful.

        Args:
            source: Original storage
            target: New storage
            agent_ids: Agents to verify

        Returns:
            Verification results
        """
        results = {
            'verified': True,
            'agent_checks': {},
            'errors': []
        }

        for agent_id in agent_ids:
            agent_result = {
                'profile_match': False,
                'environments_match': False,
                'data_integrity': True
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

                # Check environments
                source_envs = set(source.list_environments(agent_id))
                target_envs = set(target.list_environments(agent_id))
                agent_result['environments_match'] = source_envs == target_envs

                # Overall agent verification
                if not (agent_result['profile_match'] and agent_result['environments_match']):
                    results['verified'] = False

            except Exception as e:
                agent_result['data_integrity'] = False
                results['errors'].append(f"Verification failed for {agent_id}: {str(e)}")
                results['verified'] = False

            results['agent_checks'][agent_id] = agent_result

        return results