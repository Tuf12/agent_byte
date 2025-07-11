"""
Core interfaces for Agent Byte v3.0

These interfaces define the contracts that all environments and storage backends
must implement to work with the Agent Byte system.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
import numpy as np


class Environment(ABC):
    """
    Base environment interface that all environments must implement.

    This is the "USB specification" that allows Agent Byte to work with
    any environment without knowing its specific details.
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.

        Returns:
            Initial state as a numpy array
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.

        Args:
            action: The action to take (as an integer index)

        Returns:
            Tuple of (next_state, reward, done, info):
                next_state: The state after taking the action
                reward: The reward received
                done: Whether the episode has ended
                info: Additional information from the environment
        """
        pass

    @abstractmethod
    def get_state_size(self) -> int:
        """
        Get the native state size of this environment.

        Returns:
            The number of dimensions in the state vector
        """
        pass

    @abstractmethod
    def get_action_size(self) -> int:
        """
        Get the number of possible actions in this environment.

        Returns:
            The number of discrete actions available
        """
        pass

    @abstractmethod
    def get_id(self) -> str:
        """
        Get a unique identifier for this environment.

        Returns:
            A string identifier (e.g., "CartPole-v1", "CustomEnv-001")
        """
        pass

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get optional metadata about the environment.

        This can include:
        - name: Human-readable name
        - purpose: What the environment is for
        - rules: List of rules/constraints
        - objectives: Success criteria
        - instructions: How to interact with the environment

        Returns:
            Dictionary of metadata or None if not provided
        """
        return None

    def render(self) -> Optional[Any]:
        """
        Render the environment (optional).

        Returns:
            Rendering output or None if not supported
        """
        return None

    def close(self) -> None:
        """
        Clean up environment resources (optional).
        """
        pass


class Storage(ABC):
    """
    Abstract storage interface for all storage backends.

    This interface allows hot-swapping between different storage systems
    (JSON, Database, Vector DB) without changing the agent code.
    """

    @abstractmethod
    def save_brain_state(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """
        Save the neural brain state for a specific agent and environment.

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier
            data: Brain state data to save

        Returns:
            True if save was successful, False otherwise
        """
        pass

    @abstractmethod
    def load_brain_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the neural brain state for a specific agent and environment.

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier

        Returns:
            Brain state data or None if not found
        """
        pass

    @abstractmethod
    def save_knowledge(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """
        Save the symbolic knowledge for a specific agent and environment.

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier
            data: Knowledge data to save

        Returns:
            True if save was successful, False otherwise
        """
        pass

    @abstractmethod
    def load_knowledge(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the symbolic knowledge for a specific agent and environment.

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier

        Returns:
            Knowledge data or None if not found
        """
        pass

    @abstractmethod
    def save_experience_vector(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """
        Save an experience vector for similarity search.

        This enables transfer learning by finding similar experiences.

        Args:
            agent_id: Unique identifier for the agent
            vector: Experience vector (typically 256-dimensional)
            metadata: Additional information about the experience

        Returns:
            True if save was successful, False otherwise
        """
        pass

    @abstractmethod
    def search_similar_experiences(self, agent_id: str, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar experiences using vector similarity.

        This method enables future upgrade to vector databases.

        Args:
            agent_id: Unique identifier for the agent
            query_vector: Vector to search for similar experiences
            k: Number of similar experiences to return

        Returns:
            List of similar experiences with metadata
        """
        pass

    @abstractmethod
    def list_environments(self, agent_id: str) -> List[str]:
        """
        List all environments this agent has learned from.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            List of environment identifiers
        """
        pass

    @abstractmethod
    def get_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the overall profile for an agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Agent profile data or None if not found
        """
        pass

    @abstractmethod
    def save_agent_profile(self, agent_id: str, profile: Dict[str, Any]) -> bool:
        """
        Save the overall profile for an agent.

        Args:
            agent_id: Unique identifier for the agent
            profile: Profile data to save

        Returns:
            True if save was successful, False otherwise
        """
        pass

    def migrate_to(self, target_storage: 'Storage') -> bool:
        """
        Migrate all data to a different storage backend.

        This enables hot-swapping storage systems.

        Args:
            target_storage: The storage backend to migrate to

        Returns:
            True if migration was successful, False otherwise
        """
        # Default implementation - can be overridden
        return False