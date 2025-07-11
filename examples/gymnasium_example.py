"""
Example adapter for Gymnasium environments.

This shows how to wrap ANY Gymnasium environment to work with Agent Byte,
without modifying the agent's core code.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym

from agent_byte.core.interfaces import Environment


class GymnasiumAdapter(Environment):
    """
    Adapter that makes any Gymnasium environment compatible with Agent Byte.

    This adapter wraps Gymnasium environments without requiring any changes
    to either the environment or the agent.
    """

    def __init__(self, gym_env: gym.Env, env_id: Optional[str] = None):
        """
        Initialize the adapter.

        Args:
            gym_env: Any Gymnasium environment instance
            env_id: Optional custom ID (defaults to gym env spec)
        """
        self.env = gym_env
        self._env_id = env_id or getattr(gym_env.spec, 'id', 'unknown_gym_env')

        # Cache environment properties
        self._state_size = self._calculate_state_size()
        self._action_size = self._calculate_action_size()

    def _calculate_state_size(self) -> int:
        """Calculate the state size from observation space."""
        obs_space = self.env.observation_space

        if hasattr(obs_space, 'shape'):
            # Box, MultiBinary, etc.
            return int(np.prod(obs_space.shape))
        elif hasattr(obs_space, 'n'):
            # Discrete
            return 1  # Single discrete value
        else:
            # Try to get a sample and check its size
            try:
                sample = obs_space.sample()
                if isinstance(sample, (list, tuple, np.ndarray)):
                    return len(np.array(sample).flatten())
                else:
                    return 1
            except:
                return 1

    def _calculate_action_size(self) -> int:
        """Calculate the number of actions from action space."""
        action_space = self.env.action_space

        if hasattr(action_space, 'n'):
            # Discrete action space
            return action_space.n
        elif hasattr(action_space, 'shape'):
            # Continuous action space - discretize for now
            # This is a simplification - real implementation might handle this differently
            return 3  # e.g., low, medium, high for each dimension
        else:
            return 2  # Default fallback

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        obs, info = self.env.reset()
        return self._observation_to_array(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Convert discrete action index to gym action if needed
        gym_action = self._convert_action(action)

        # Take step in gym environment
        obs, reward, terminated, truncated, info = self.env.step(gym_action)

        # Convert observation to array
        next_state = self._observation_to_array(obs)

        # Combine terminated and truncated for done signal
        done = terminated or truncated

        # Add termination info to info dict
        info['terminated'] = terminated
        info['truncated'] = truncated

        return next_state, float(reward), done, info

    def get_state_size(self) -> int:
        """Get the state size."""
        return self._state_size

    def get_action_size(self) -> int:
        """Get the action size."""
        return self._action_size

    def get_id(self) -> str:
        """Get environment ID."""
        return self._env_id

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get optional metadata about the environment."""
        metadata = {
            'gym_version': gym.__version__,
            'render_modes': self.env.metadata.get('render_modes', []),
            'reward_range': self.env.reward_range
        }

        # Add observation space info
        obs_space = self.env.observation_space
        if hasattr(obs_space, 'shape'):
            metadata['observation_shape'] = obs_space.shape
        if hasattr(obs_space, 'low') and hasattr(obs_space, 'high'):
            metadata['observation_range'] = (obs_space.low.tolist(), obs_space.high.tolist())

        # Add action space info
        action_space = self.env.action_space
        if hasattr(action_space, 'n'):
            metadata['action_type'] = 'discrete'
            metadata['num_actions'] = action_space.n
        elif hasattr(action_space, 'shape'):
            metadata['action_type'] = 'continuous'
            metadata['action_shape'] = action_space.shape

        # Add environment-specific metadata if available
        if hasattr(self.env, 'spec') and self.env.spec:
            metadata['env_name'] = self.env.spec.id
            metadata['max_episode_steps'] = self.env.spec.max_episode_steps
            metadata['reward_threshold'] = self.env.spec.reward_threshold

        return metadata

    def render(self) -> Optional[Any]:
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    def _observation_to_array(self, obs) -> np.ndarray:
        """Convert any observation to a numpy array."""
        if isinstance(obs, np.ndarray):
            return obs.flatten()
        elif isinstance(obs, (list, tuple)):
            return np.array(obs).flatten()
        elif isinstance(obs, (int, float)):
            return np.array([obs])
        else:
            # Try to convert to array
            try:
                return np.array(obs).flatten()
            except:
                # Last resort - return empty array
                return np.array([0.0])

    def _convert_action(self, action: int):
        """Convert discrete action index to gym action format."""
        action_space = self.env.action_space

        if hasattr(action_space, 'n'):
            # Already discrete - just return
            return action
        elif hasattr(action_space, 'shape'):
            # Continuous action space - need to discretize
            # This is a simplified example - real implementation would be more sophisticated
            if len(action_space.shape) == 1:
                # 1D continuous action
                low = action_space.low[0]
                high = action_space.high[0]

                # Map discrete action to continuous
                if action == 0:
                    return np.array([low])
                elif action == 1:
                    return np.array([(low + high) / 2])
                else:
                    return np.array([high])
            else:
                # Multi-dimensional - return zeros for now
                return np.zeros(action_space.shape)
        else:
            # Unknown action space - return action as-is
            return action


# Example usage
if __name__ == "__main__":
    """Example of how to use the Gymnasium adapter."""

    # Import Agent Byte (this would be from the installed package)
    # from agent_byte import AgentByte

    # Create any Gymnasium environment
    gym_env = gym.make("CartPole-v1")

    # Wrap it with our adapter
    adapted_env = GymnasiumAdapter(gym_env)

    # Now it works with Agent Byte!
    print(f"Environment ID: {adapted_env.get_id()}")
    print(f"State size: {adapted_env.get_state_size()}")
    print(f"Action size: {adapted_env.get_action_size()}")
    print(f"Metadata: {adapted_env.get_metadata()}")

    # Test a few steps
    state = adapted_env.reset()
    print(f"Initial state shape: {state.shape}")

    for i in range(5):
        action = np.random.randint(0, adapted_env.get_action_size())
        next_state, reward, done, info = adapted_env.step(action)
        print(f"Step {i}: action={action}, reward={reward:.2f}, done={done}")

        if done:
            state = adapted_env.reset()
            print("Environment reset")

    adapted_env.close()