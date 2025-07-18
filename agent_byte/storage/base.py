"""
Abstract base class for storage implementations.

This module provides the base class that all storage backends must inherit from.
It includes helper methods and common functionality.
Enhanced with Sprint 9 continuous action space support.
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
    Enhanced with Sprint 9 continuous action space support.
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
    def save_agent_profile(self, agent_id: str, profile: Dict[str, Any]) -> bool:
        """Save agent profile."""
        pass

    @abstractmethod
    def load_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent profile - this is the primary method."""
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

    # NEW: Sprint 9 Continuous Action Space Support
    @abstractmethod
    def save_continuous_network_state(self, agent_id: str, env_id: str,
                                     network_state: Dict[str, Any]) -> bool:
        """
        Save continuous network state (SAC/DDPG weights and training state).

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier
            network_state: Dictionary containing:
                - algorithm: "sac" or "ddpg"
                - state_size: Input state dimension
                - action_size: Output action dimension
                - action_bounds: Action space bounds
                - device: Training device ('cpu' or 'cuda')
                - weights_data: Serialized network weights
                - network_info: Algorithm-specific parameters

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_continuous_network_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """
        Load continuous network state.

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier

        Returns:
            Network state dictionary or None if not found
        """
        pass

    @abstractmethod
    def save_action_adapter_config(self, agent_id: str, env_id: str,
                                  adapter_config: Dict[str, Any]) -> bool:
        """
        Save action adapter configuration.

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier
            adapter_config: Dictionary containing:
                - source_space: Source action space specification
                - target_space: Target action space specification
                - adapter_type: Type of adapter used
                - parameters: Adapter-specific parameters

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_action_adapter_config(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """
        Load action adapter configuration.

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier

        Returns:
            Adapter configuration dictionary or None if not found
        """
        pass

    @abstractmethod
    def list_continuous_networks(self, agent_id: str) -> List[str]:
        """
        List all environments with saved continuous networks for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of environment IDs with continuous networks
        """
        pass

    @abstractmethod
    def list_action_adapters(self, agent_id: str) -> List[str]:
        """
        List all environments with saved action adapters for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of environment IDs with action adapters
        """
        pass

    # Compatibility alias - get_agent_profile calls load_agent_profile
    def get_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Alias for load_agent_profile for backward compatibility."""
        return self.load_agent_profile(agent_id)

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
            expected_dim: Expected vector dimension

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(vector, np.ndarray):
            return False

        if vector.ndim != 1:
            return False

        if vector.shape[0] != expected_dim:
            return False

        if not np.isfinite(vector).all():
            return False

        return True

    def _validate_continuous_network_state(self, network_state: Dict[str, Any]) -> bool:
        """
        Validate continuous network state structure.

        Args:
            network_state: Network state to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic validation - must have algorithm
        if 'algorithm' not in network_state:
            self.logger.warning("Missing required field: algorithm")
            return False

        # Validate algorithm if provided
        valid_algorithms = ['sac', 'ddpg', 'td3', 'ppo_continuous']
        if network_state['algorithm'] not in valid_algorithms:
            self.logger.warning(f"Invalid algorithm: {network_state['algorithm']}")
            return False

        # Optional validation for other fields
        if 'state_size' in network_state:
            if not isinstance(network_state['state_size'], int) or network_state['state_size'] <= 0:
                self.logger.warning(f"Invalid state_size: {network_state['state_size']}")
                return False

        if 'action_size' in network_state:
            if not isinstance(network_state['action_size'], int) or network_state['action_size'] <= 0:
                self.logger.warning(f"Invalid action_size: {network_state['action_size']}")
                return False

        return True

    def _validate_action_adapter_config(self, adapter_config: Dict[str, Any]) -> bool:
        """
        Validate action adapter configuration structure.

        Args:
            adapter_config: Adapter config to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic validation - must have adapter_type
        if 'adapter_type' not in adapter_config:
            self.logger.warning("Missing required field: adapter_type")
            return False

        # Validate adapter type
        valid_types = [
            'discrete_to_continuous',
            'continuous_to_discrete',
            'hybrid',
            'parameterized'
        ]
        if adapter_config['adapter_type'] not in valid_types:
            self.logger.warning(f"Invalid adapter_type: {adapter_config['adapter_type']}")
            return False

        return True

    def close(self) -> None:
        """Close storage backend and clean up resources."""
        if self._cache is not None:
            self._cache.clear()
        self.logger.info("Storage backend closed")