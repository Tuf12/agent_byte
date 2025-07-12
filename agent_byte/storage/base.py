"""
Abstract base class for storage implementations.

This module provides the base class that all storage backends must inherit from.
It includes helper methods and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
import logging
from datetime import datetime


from ..core.interfaces import Storage

class StorageBase(Storage):
    """
    Base class for all storage implementations.

    This class provides the interface and common functionality that all
    storage backends must implement to work with Agent Byte.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize storage backend.

        Args:
            config: Configuration dictionary for the storage backend
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache = {} if self.config.get('enable_cache', True) else None

    # Abstract methods that must be implemented

    @abstractmethod
    def save_brain_state(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save neural brain state."""
        pass

    @abstractmethod
    def load_brain_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load neural brain state."""
        pass

    @abstractmethod
    def save_knowledge(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save symbolic knowledge."""
        pass

    @abstractmethod
    def load_knowledge(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load symbolic knowledge."""
        pass

    @abstractmethod
    def save_experience_vector(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save experience vector for similarity search."""
        pass

    @abstractmethod
    def search_similar_experiences(self, agent_id: str, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar experiences."""
        pass

    @abstractmethod
    def list_environments(self, agent_id: str) -> List[str]:
        """List all environments for an agent."""
        pass

    @abstractmethod
    def get_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent profile."""
        pass

    @abstractmethod
    def save_agent_profile(self, agent_id: str, profile: Dict[str, Any]) -> bool:
        """Save agent profile."""
        pass

    # New abstract methods for autoencoder support

    @abstractmethod
    def save_autoencoder(self, agent_id: str, env_id: str, state_dict: Dict[str, Any]) -> bool:
        """
        Save autoencoder state for an environment.

        Args:
            agent_id: Agent identifier
            env_id: Environment identifier
            state_dict: Autoencoder state dictionary

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def load_autoencoder(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """
        Load autoencoder state for an environment.

        Args:
            agent_id: Agent identifier
            env_id: Environment identifier

        Returns:
            Autoencoder state dictionary or None if not found
        """
        pass

    @abstractmethod
    def list_autoencoders(self, agent_id: str) -> List[str]:
        """
        List all environments with saved autoencoders for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of environment IDs with autoencoders
        """
        pass

    # Helper methods available to all implementations

    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        return "_".join(str(arg) for arg in args)

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if enabled."""
        if self._cache is not None:
            return self._cache.get(key)
        return None

    def _put_in_cache(self, key: str, value: Any) -> None:
        """Put value in cache if enabled."""
        if self._cache is not None:
            # Simple cache size management
            max_size = self.config.get('cache_size', 100)
            if len(self._cache) >= max_size:
                # Remove the oldest entry (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = value

    def _clear_cache(self) -> None:
        """Clear the cache."""
        if self._cache is not None:
            self._cache.clear()

    def _add_timestamp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add timestamp to data."""
        data['_saved_at'] = datetime.now().isoformat()
        return data

    def _validate_vector(self, vector: np.ndarray, expected_dim: int = 256) -> bool:
        """
        Validate that a vector has the expected dimensions.

        Args:
            vector: Vector to validate
            expected_dim: Expected number of dimensions

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(vector, np.ndarray):
            self.logger.error(f"Vector must be numpy array, got {type(vector)}")
            return False

        if vector.ndim != 1:
            self.logger.error(f"Vector must be 1-dimensional, got {vector.ndim} dimensions")
            return False

        if vector.shape[0] != expected_dim:
            self.logger.error(f"Vector must have {expected_dim} dimensions, got {vector.shape[0]}")
            return False

        return True

    def migrate_to(self, target_storage: 'StorageBase', agent_ids: Optional[List[str]] = None) -> bool:
        """
        Migrate data to another storage backend.

        Args:
            target_storage: Target storage backend
            agent_ids: Specific agent IDs to migrate (None for all)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all agent IDs if not specified
            if agent_ids is None:
                # This would need to be implemented by subclasses
                self.logger.warning("Migrating all agents - this may take time")
                agent_ids = self._get_all_agent_ids()

            for agent_id in agent_ids:
                self.logger.info(f"Migrating agent: {agent_id}")

                # Migrate profile
                profile = self.get_agent_profile(agent_id)
                if profile:
                    target_storage.save_agent_profile(agent_id, profile)

                # Migrate environments
                environments = self.list_environments(agent_id)
                for env_id in environments:
                    # Migrate brain state
                    brain_state = self.load_brain_state(agent_id, env_id)
                    if brain_state:
                        target_storage.save_brain_state(agent_id, env_id, brain_state)

                    # Migrate knowledge
                    knowledge = self.load_knowledge(agent_id, env_id)
                    if knowledge:
                        target_storage.save_knowledge(agent_id, env_id, knowledge)

                    # Migrate autoencoder
                    autoencoder = self.load_autoencoder(agent_id, env_id)
                    if autoencoder:
                        target_storage.save_autoencoder(agent_id, env_id, autoencoder)

                # Note: Experience vectors would need special handling
                self.logger.info(f"Successfully migrated agent: {agent_id}")

            return True

        except Exception as e:
            self.logger.error(f"Migration failed: {str(e)}")
            return False

    @abstractmethod
    def _get_all_agent_ids(self) -> List[str]:
        """Get all agent IDs in storage (must be implemented by subclasses)."""
        pass

    def close(self) -> None:
        """
        Clean up storage resources.

        Override this method if your storage backend needs cleanup.
        """
        self._clear_cache()
        self.logger.info("Storage backend closed")