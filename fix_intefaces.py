"""
Core interfaces for Agent Byte v3.0 with Continuous Action Space Support

These interfaces define the contracts that all environments and storage backends
must implement to work with the Agent Byte system. Now enhanced with support
for both discrete and continuous action spaces.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
from enum import Enum


class ActionSpaceType(Enum):
    """Enumeration of supported action space types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"  # Mix of discrete and continuous
    PARAMETERIZED = "parameterized"  # Neural parameterized actions


class ActionSpace:
    """
    Defines the action space for an environment.

    This class encapsulates all information about what actions
    are possible in an environment.
    """

    def __init__(self,
                 space_type: ActionSpaceType,
                 size: Union[int, Tuple[int, ...]],
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 discrete_actions: Optional[List[str]] = None):
        """
        Initialize action space.

        Args:
            space_type: Type of action space
            size: For discrete: number of actions
                  For continuous: shape of action vector
            bounds: For continuous: (low_bounds, high_bounds) arrays
            discrete_actions: Optional names for discrete actions
        """
        self.space_type = space_type
        self.size = size
        self.bounds = bounds
        self.discrete_actions = discrete_actions

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate action space configuration."""
        if self.space_type == ActionSpaceType.DISCRETE:
            if not isinstance(self.size, int) or self.size < 2:
                raise ValueError("Discrete action space must have integer size >= 2")

        elif self.space_type == ActionSpaceType.CONTINUOUS:
            if self.bounds is None:
                raise ValueError("Continuous action space requires bounds")

            low, high = self.bounds
            if not isinstance(low, np.ndarray) or not isinstance(high, np.ndarray):
                raise ValueError("Bounds must be numpy arrays")

            if low.shape != high.shape:
                raise ValueError("Low and high bounds must have same shape")

            if not isinstance(self.size, (int, tuple)):
                raise ValueError("Continuous action space size must be int or tuple")

    def is_discrete(self) -> bool:
        """Check if action space is discrete."""
        return self.space_type == ActionSpaceType.DISCRETE

    def is_continuous(self) -> bool:
        """Check if action space is continuous."""
        return self.space_type == ActionSpaceType.CONTINUOUS

    def is_hybrid(self) -> bool:
        """Check if action space is hybrid."""
        return self.space_type == ActionSpaceType.HYBRID

    def get_action_dim(self) -> int:
        """Get dimensionality of action space."""
        if self.space_type == ActionSpaceType.DISCRETE:
            return 1  # Single discrete action
        elif self.space_type == ActionSpaceType.CONTINUOUS:
            if isinstance(self.size, int):
                return self.size
            else:
                return np.prod(self.size)
        else:
            # For hybrid/parameterized, return total dimensions
            return getattr(self, 'total_dim', 1)

    def sample(self) -> Union[int, np.ndarray]:
        """Sample a random action from this space."""
        if self.space_type == ActionSpaceType.DISCRETE:
            return np.random.randint(0, self.size)

        elif self.space_type == ActionSpaceType.CONTINUOUS:
            low, high = self.bounds
            return np.random.uniform(low, high)

        else:
            raise NotImplementedError(f"Sampling not implemented for {self.space_type}")

    def contains(self, action: Union[int, np.ndarray]) -> bool:
        """Check if an action is valid in this space."""
        if self.space_type == ActionSpaceType.DISCRETE:
            return isinstance(action, (int, np.integer)) and 0 <= action < self.size

        elif self.space_type == ActionSpaceType.CONTINUOUS:
            if not isinstance(action, np.ndarray):
                action = np.array(action)

            low, high = self.bounds
            return np.all(action >= low) and np.all(action <= high)

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'space_type': self.space_type.value,
            'size': self.size,
        }

        if self.bounds is not None:
            result['bounds'] = {
                'low': self.bounds[0].tolist(),
                'high': self.bounds[1].tolist()
            }

        if self.discrete_actions is not None:
            result['discrete_actions'] = self.discrete_actions

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionSpace':
        """Create ActionSpace from dictionary."""
        space_type = ActionSpaceType(data['space_type'])
        size = data['size']

        bounds = None
        if 'bounds' in data:
            bounds = (
                np.array(data['bounds']['low']),
                np.array(data['bounds']['high'])
            )

        discrete_actions = data.get('discrete_actions')

        return cls(space_type, size, bounds, discrete_actions)


class Environment(ABC):
    """
    Enhanced base environment interface supporting both discrete and continuous actions.

    This is the "USB specification" that allows Agent Byte to work with
    any environment without knowing its specific details. Now supports
    continuous control tasks.
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
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.

        Args:
            action: The action to take
                   - For discrete: integer index
                   - For continuous: numpy array of action values
                   - For hybrid: dict with 'discrete' and 'continuous' keys

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
    def get_action_space(self) -> ActionSpace:
        """
        Get the action space specification for this environment.

        Returns:
            ActionSpace object describing the available actions
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

    # BACKWARD COMPATIBILITY: Keep old discrete-only methods
    def get_action_size(self) -> int:
        """
        Get the number of possible actions (backward compatibility).

        For discrete spaces, returns the number of actions.
        For continuous spaces, returns the dimensionality.

        Returns:
            The action space size
        """
        action_space = self.get_action_space()
        if action_space.is_discrete():
            return action_space.size
        else:
            return action_space.get_action_dim()

    def is_discrete_action_space(self) -> bool:
        """Check if this environment uses discrete actions."""
        return self.get_action_space().is_discrete()

    def is_continuous_action_space(self) -> bool:
        """Check if this environment uses continuous actions."""
        return self.get_action_space().is_continuous()

    def get_action_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get action bounds for continuous action spaces.

        Returns:
            Tuple of (low_bounds, high_bounds) or None for discrete spaces
        """
        action_space = self.get_action_space()
        return action_space.bounds if action_space.is_continuous() else None

    def sample_action(self) -> Union[int, np.ndarray]:
        """Sample a random valid action from the action space."""
        return self.get_action_space().sample()

    def validate_action(self, action: Union[int, np.ndarray]) -> bool:
        """Check if an action is valid for this environment."""
        return self.get_action_space().contains(action)

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get optional metadata about the environment.

        This can include:
        - name: Human-readable name
        - purpose: What the environment is for
        - rules: List of rules/constraints
        - objectives: Success criteria
        - instructions: How to interact with the environment
        - action_space_info: Additional action space details

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

    This interface defines how Agent Byte can store and retrieve:
    - Agent profiles and configurations
    - Experience data (state, action, reward, next_state, done)
    - Neural network weights and training state
    - Knowledge base (skills, patterns, transfer history)
    - Performance metrics and analytics
    """

    @abstractmethod
    def save_agent_profile(self, agent_id: str, profile: Dict[str, Any]) -> bool:
        """
        Save agent profile and configuration.

        Args:
            agent_id: Unique identifier for the agent
            profile: Dictionary containing agent configuration and metadata

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Load agent profile and configuration.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Profile dictionary or None if not found
        """
        pass

    @abstractmethod
    def save_experience(self, agent_id: str, env_id: str,
                        experience: Dict[str, Any]) -> bool:
        """
        Save a single experience tuple.

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier
            experience: Dictionary with keys: state, action, reward, next_state, done

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_experiences(self, agent_id: str, env_id: str,
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load experience data for an agent in a specific environment.

        Args:
            agent_id: Unique identifier for the agent
            env_id: Environment identifier
            limit: Maximum number of experiences to return

        Returns:
            List of experience dictionaries
        """
        pass

    @abstractmethod
    def save_brain_state(self, agent_id: str, brain_type: str,
                         state_data: Dict[str, Any]) -> bool:
        """
        Save neural network or brain state.

        Args:
            agent_id: Unique identifier for the agent
            brain_type: Type of brain ('neural', 'symbolic', 'dual')
            state_data: State information to save

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_brain_state(self, agent_id: str, brain_type: str) -> Optional[Dict[str, Any]]:
        """
        Load neural network or brain state.

        Args:
            agent_id: Unique identifier for the agent
            brain_type: Type of brain ('neural', 'symbolic', 'dual')

        Returns:
            State data dictionary or None if not found
        """
        pass

    @abstractmethod
    def save_knowledge_base(self, agent_id: str,
                            knowledge: Dict[str, Any]) -> bool:
        """
        Save knowledge base (skills, patterns, transfer history).

        Args:
            agent_id: Unique identifier for the agent
            knowledge: Knowledge base dictionary

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_knowledge_base(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Load knowledge base.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Knowledge base dictionary or None if not found
        """
        pass

    @abstractmethod
    def search_similar_experiences(self, agent_id: str,
                                   query_state: np.ndarray,
                                   limit: int = 10,
                                   similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Search for experiences similar to a query state.

        Args:
            agent_id: Unique identifier for the agent
            query_state: State to find similar experiences for
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of similar experience dictionaries with similarity scores
        """
        pass

    def clear_agent_data(self, agent_id: str) -> bool:
        """
        Clear all data for a specific agent (optional).

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            True if cleared successfully, False otherwise
        """
        return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics (optional).

        Returns:
            Dictionary with storage usage information
        """
        return {}


# Utility functions for action space handling
def create_discrete_action_space(num_actions: int,
                                 action_names: Optional[List[str]] = None) -> ActionSpace:
    """
    Create a discrete action space.

    Args:
        num_actions: Number of discrete actions
        action_names: Optional names for each action

    Returns:
        ActionSpace configured for discrete actions
    """
    return ActionSpace(
        space_type=ActionSpaceType.DISCRETE,
        size=num_actions,
        discrete_actions=action_names
    )


def create_continuous_action_space(action_dim: int,
                                   low: Union[float, np.ndarray],
                                   high: Union[float, np.ndarray]) -> ActionSpace:
    """
    Create a continuous action space.

    Args:
        action_dim: Dimensionality of action vector
        low: Lower bounds (scalar or array)
        high: Upper bounds (scalar or array)

    Returns:
        ActionSpace configured for continuous actions
    """
    # Convert scalars to arrays
    if np.isscalar(low):
        low = np.full(action_dim, low, dtype=np.float32)
    if np.isscalar(high):
        high = np.full(action_dim, high, dtype=np.float32)

    return ActionSpace(
        space_type=ActionSpaceType.CONTINUOUS,
        size=action_dim,
        bounds=(low, high)
    )


def create_hybrid_action_space(discrete_size: int,
                               continuous_dim: int,
                               continuous_low: Union[float, np.ndarray],
                               continuous_high: Union[float, np.ndarray]) -> ActionSpace:
    """
    Create a hybrid action space with both discrete and continuous components.

    Args:
        discrete_size: Number of discrete actions
        continuous_dim: Dimensionality of continuous actions
        continuous_low: Lower bounds for continuous actions
        continuous_high: Upper bounds for continuous actions

    Returns:
        ActionSpace configured for hybrid actions
    """
    # Convert scalars to arrays for continuous bounds
    if np.isscalar(continuous_low):
        continuous_low = np.full(continuous_dim, continuous_low, dtype=np.float32)
    if np.isscalar(continuous_high):
        continuous_high = np.full(continuous_dim, continuous_high, dtype=np.float32)

    action_space = ActionSpace(
        space_type=ActionSpaceType.HYBRID,
        size=(discrete_size, continuous_dim),
        bounds=(continuous_low, continuous_high)
    )

    # Add total dimension for hybrid spaces
    action_space.total_dim = 1 + continuous_dim  # 1 for discrete + continuous dims

    return action_space