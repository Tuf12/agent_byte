"""
Continuous Neural Networks for Agent Byte v3.0

This module implements continuous control neural networks including
Soft Actor-Critic (SAC) for environments with continuous action spaces.
Integrates seamlessly with the existing dual brain architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import logging
from collections import deque
import random

from .interfaces import ActionSpace, ActionSpaceType


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for continuous action spaces.

    Outputs mean and log_std for a Gaussian distribution over actions.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_sizes: Tuple[int, ...] = (256, 256),
                 log_std_bounds: Tuple[float, float] = (-20, 2)):
        """
        Initialize Gaussian policy network.

        Args:
            state_size: Size of state input
            action_size: Size of action output
            hidden_sizes: Sizes of hidden layers
            log_std_bounds: Bounds for log standard deviation
        """
        super(GaussianPolicy, self).__init__()

        self.action_size = action_size
        self.log_std_min, self.log_std_max = log_std_bounds

        # Build network layers
        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_size, action_size)
        self.log_std_head = nn.Linear(prev_size, action_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.

        Args:
            state: Input state tensor

        Returns:
            Tuple of (mean, log_std) tensors
        """
        features = self.shared_layers(state)

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.

        Args:
            state: Input state tensor
            deterministic: If True, return mean action without noise

        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            normal = Normal(mean, std)

            # Reparameterization trick
            x_t = normal.rsample()
            action = torch.tanh(x_t)

            # Calculate log probability with tanh correction
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    """
    Q-value network for continuous action spaces (critic).

    Takes state and action as input, outputs Q-value.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_sizes: Tuple[int, ...] = (256, 256)):
        """
        Initialize Q-network.

        Args:
            state_size: Size of state input
            action_size: Size of action input
            hidden_sizes: Sizes of hidden layers
        """
        super(QNetwork, self).__init__()

        # Build network layers
        layers = []
        prev_size = state_size + action_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size

        # Output single Q-value
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Q-value tensor
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SoftActorCritic:
    """
    Soft Actor-Critic (SAC) implementation for continuous control.

    SAC is a state-of-the-art algorithm for continuous control that
    maximizes both expected return and entropy of the policy.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 action_bounds: Tuple[np.ndarray, np.ndarray],
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_alpha: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: Optional[float] = None,
                 target_entropy: Optional[float] = None,
                 hidden_sizes: Tuple[int, ...] = (256, 256),
                 device: str = 'cpu'):
        """
        Initialize SAC agent.

        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            action_bounds: Tuple of (low, high) action bounds
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            lr_alpha: Learning rate for temperature parameter
            gamma: Discount factor
            tau: Soft update parameter
            alpha: Temperature parameter (if None, will be learned)
            target_entropy: Target entropy (if None, will be set automatically)
            hidden_sizes: Hidden layer sizes
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # Networks
        self.actor = GaussianPolicy(state_size, action_size, hidden_sizes).to(self.device)

        # Two Q-networks for double Q-learning
        self.critic1 = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.critic2 = QNetwork(state_size, action_size, hidden_sizes).to(self.device)

        # Target networks
        self.critic1_target = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.critic2_target = QNetwork(state_size, action_size, hidden_sizes).to(self.device)

        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # Temperature parameter (entropy regularization)
        if alpha is None:
            # Learnable temperature
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.target_entropy = target_entropy if target_entropy else -action_size
        else:
            # Fixed temperature
            self.log_alpha = torch.log(torch.tensor(alpha, device=self.device))
            self.alpha_optimizer = None
            self.target_entropy = None

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = 256

        self.logger = logging.getLogger("SoftActorCritic")

    @property
    def alpha(self) -> float:
        """Get current temperature parameter."""
        return self.log_alpha.exp().item()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state: Current state
            deterministic: If True, return deterministic action

        Returns:
            Selected action scaled to action bounds
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor, deterministic)
            action = action.cpu().numpy()[0]

        # Scale action to bounds
        return self._scale_action(action)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to action bounds."""
        low, high = self.action_bounds
        return low + (action + 1.0) * 0.5 * (high - low)

    def _unscale_action(self, action: np.ndarray) -> np.ndarray:
        """Unscale action from action bounds to [-1, 1]."""
        low, high = self.action_bounds
        return 2.0 * (action - low) / (high - low) - 1.0

    def store_experience(self,
                         state: np.ndarray,
                         action: np.ndarray,
                         reward: float,
                         next_state: np.ndarray,
                         done: bool):
        """Store experience in replay buffer."""
        # Unscale action for storage
        unscaled_action = self._unscale_action(action)

        experience = {
            'state': state,
            'action': unscaled_action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.replay_buffer.append(experience)

    def learn(self) -> Dict[str, float]:
        """
        Update networks using experiences from replay buffer.

        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        states = torch.FloatTensor([e['state'] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e['action'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e['done'] for e in batch]).unsqueeze(1).to(self.device)

        # Update critics
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)

        # Update actor
        actor_loss = self._update_actor(states)

        # Update temperature
        alpha_loss = self._update_alpha(states)

        # Soft update target networks
        self._soft_update_targets()

        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }

    def _update_critics(self,
                        states: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        next_states: torch.Tensor,
                        dones: torch.Tensor) -> float:
        """Update critic networks."""
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Compute target Q-values
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs

            target_q = rewards + self.gamma * (1 - dones.float()) * target_q

        # Compute current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # Compute critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return (critic1_loss + critic2_loss).item() / 2

    def _update_actor(self, states: torch.Tensor) -> float:
        """Update actor network."""
        # Sample actions from current policy
        actions, log_probs = self.actor.sample(states)

        # Compute Q-values for sampled actions
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)

        # Compute actor loss (maximize Q-value and entropy)
        actor_loss = (self.alpha * log_probs - q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, states: torch.Tensor) -> float:
        """Update temperature parameter."""
        if self.alpha_optimizer is None:
            return 0.0

        with torch.no_grad():
            _, log_probs = self.actor.sample(states)

        # Compute alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()

        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def _soft_update_targets(self):
        """Soft update target networks."""
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save model parameters."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.log_alpha = checkpoint['log_alpha']

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])


class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG) implementation.

    Alternative to SAC for continuous control, uses deterministic policy
    with added noise for exploration.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 action_bounds: Tuple[np.ndarray, np.ndarray],
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.001,
                 noise_std: float = 0.1,
                 hidden_sizes: Tuple[int, ...] = (256, 256),
                 device: str = 'cpu'):
        """
        Initialize DDPG agent.

        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            action_bounds: Tuple of (low, high) action bounds
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            tau: Soft update parameter
            noise_std: Standard deviation for exploration noise
            hidden_sizes: Hidden layer sizes
            device: Device to run on
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.device = torch.device(device)

        # Actor networks (deterministic policy)
        self.actor = self._build_actor(hidden_sizes).to(self.device)
        self.actor_target = self._build_actor(hidden_sizes).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks
        self.critic = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.critic_target = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Experience replay
        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = 256

        self.logger = logging.getLogger("DDPG")

    def _build_actor(self, hidden_sizes: Tuple[int, ...]) -> nn.Module:
        """Build deterministic actor network."""
        layers = []
        prev_size = self.state_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size

        # Output layer with tanh activation
        layers.append(nn.Linear(prev_size, self.action_size))
        layers.append(nn.Tanh())

        network = nn.Sequential(*layers)

        # Initialize weights
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

        return network

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using deterministic policy with optional noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if add_noise:
            # Add Gaussian noise for exploration
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        # Scale to action bounds
        return self._scale_action(action)

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to action bounds."""
        low, high = self.action_bounds
        return low + (action + 1.0) * 0.5 * (high - low)

    def _unscale_action(self, action: np.ndarray) -> np.ndarray:
        """Unscale action from action bounds to [-1, 1]."""
        low, high = self.action_bounds
        return 2.0 * (action - low) / (high - low) - 1.0

    def store_experience(self,
                         state: np.ndarray,
                         action: np.ndarray,
                         reward: float,
                         next_state: np.ndarray,
                         done: bool):
        """Store experience in replay buffer."""
        unscaled_action = self._unscale_action(action)

        experience = {
            'state': state,
            'action': unscaled_action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.replay_buffer.append(experience)

    def learn(self) -> Dict[str, float]:
        """Update networks using DDPG algorithm."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)

        states = torch.FloatTensor([e['state'] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e['action'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e['done'] for e in batch]).unsqueeze(1).to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones.float()) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update_targets()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

    def _soft_update_targets(self):
        """Soft update target networks."""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class ContinuousNetworkManager:
    """
    Manager for continuous control networks.

    This class integrates continuous control algorithms with the existing
    Agent Byte architecture, automatically selecting appropriate algorithms
    based on the action space.
    """

    def __init__(self, storage, agent_id, str,
                 state_size: int,
                 action_space: ActionSpace,
                 algorithm: str = "auto",
                 device: str = "auto",
                 **kwargs):
        """
        Initialize continuous network manager.

        Args:
            state_size: Size of state space
            action_space: Action space specification
            algorithm: Algorithm to use ("auto", "sac", "ddpg")
            device: Device to use ("auto", "cpu", "cuda")
            **kwargs: Additional arguments for algorithms
        """
        self.state_size = state_size
        self.action_space = action_space
        self.algorithm = algorithm

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.logger = logging.getLogger("ContinuousNetworkManager")

        # Validate action space
        if not action_space.is_continuous():
            raise ValueError("ContinuousNetworkManager requires continuous action space")

        # Get action space parameters
        self.action_size = action_space.get_action_dim()
        self.action_bounds = action_space.bounds

        # Select and initialize algorithm
        self.network = self._initialize_algorithm(**kwargs)

        self.logger.info(f"Initialized {self.algorithm} on {self.device} device")

        """
                Initialize the storage manager.

                Args:
                    storage: Storage backend instance (JsonNumpyStorage or VectorDBStorage)
                    agent_id: Agent identifier
                """
        self.storage = storage
        self.agent_id = agent_id
        self.logger = logging.getLogger("ContinuousNetworkStorageManager")

    def save_continuous_network(self, env_id: str, network_manager: ContinuousNetworkManager) -> bool:
        """
        Save a continuous network manager to storage.

        Args:
            env_id: Environment identifier
            network_manager: ContinuousNetworkManager instance to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create a temporary file path for the network weights
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Save network weights to temporary file
            network_manager.save_networks(tmp_path)

            # Read the weights data
            with open(tmp_path, 'rb') as f:
                weights_data = f.read()

            # Clean up temporary file
            os.unlink(tmp_path)

            # Convert to base64 for JSON storage
            import base64
            weights_base64 = base64.b64encode(weights_data).decode('utf-8')

            # Create storage-friendly state
            network_state = {
                'algorithm': network_manager.algorithm,
                'state_size': network_manager.state_size,
                'action_size': network_manager.action_size,
                'action_bounds': {
                    'low': network_manager.action_bounds[0].tolist(),
                    'high': network_manager.action_bounds[1].tolist()
                },
                'device': network_manager.device,
                'weights_data': weights_base64,
                'network_info': network_manager.get_network_info()
            }

            # Save through storage backend
            success = self.storage.save_continuous_network_state(
                self.agent_id, env_id, network_state
            )

            if success:
                self.logger.info(f"Saved continuous network for {env_id}")
            else:
                self.logger.error(f"Failed to save continuous network for {env_id}")

            return success

        except Exception as e:
            self.logger.error(f"Error saving continuous network: {str(e)}")
            return False

    def load_continuous_network(self, env_id: str, action_space: ActionSpace) -> Optional[ContinuousNetworkManager]:
        """
        Load a continuous network manager from storage.

        Args:
            env_id: Environment identifier
            action_space: Action space for the network

        Returns:
            ContinuousNetworkManager instance or None if not found
        """
        try:
            # Load state from storage
            network_state = self.storage.load_continuous_network_state(self.agent_id, env_id)

            if network_state is None:
                return None

            # Create network manager with saved configuration
            network_manager = ContinuousNetworkManager(
                state_size=network_state['state_size'],
                action_space=action_space,
                algorithm=network_state['algorithm'],
                device=network_state['device']
            )

            # Restore network weights
            if 'weights_data' in network_state:
                import base64
                import tempfile
                import os

                # Decode weights data
                weights_data = base64.b64decode(network_state['weights_data'])

                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(weights_data)

                # Load weights
                network_manager.load_networks(tmp_path)

                # Clean up
                os.unlink(tmp_path)

            self.logger.info(f"Loaded continuous network for {env_id}")
            return network_manager

        except Exception as e:
            self.logger.error(f"Error loading continuous network: {str(e)}")
            return None

    def list_saved_networks(self) -> List[str]:
        """
        List all environments with saved continuous networks.

        Returns:
            List of environment IDs
        """
        try:
            return self.storage.list_continuous_networks(self.agent_id)
        except Exception as e:
            self.logger.error(f"Error listing networks: {str(e)}")
            return []

    def delete_network(self, env_id: str) -> bool:
        """
        Delete a saved continuous network.

        Args:
            env_id: Environment identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # This would need to be implemented in the storage backends
            # For now, just log that deletion was requested
            self.logger.info(f"Network deletion requested for {env_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting network: {str(e)}")
            return False

    def _initialize_algorithm(self, **kwargs) -> Union[SoftActorCritic, DDPG]:
        """Initialize the selected algorithm."""
        if self.algorithm == "auto":
            # Auto-select algorithm based on action space characteristics
            action_range = np.max(self.action_bounds[1] - self.action_bounds[0])

            if action_range > 10 or self.action_size > 5:
                # Large action spaces: use SAC for better exploration
                self.algorithm = "sac"
            else:
                # Smaller action spaces: DDPG can work well
                self.algorithm = "ddpg"

        if self.algorithm == "sac":
            return SoftActorCritic(
                state_size=self.state_size,
                action_size=self.action_size,
                action_bounds=self.action_bounds,
                device=self.device,
                **kwargs
            )

        elif self.algorithm == "ddpg":
            return DDPG(
                state_size=self.state_size,
                action_size=self.action_size,
                action_bounds=self.action_bounds,
                device=self.device,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the continuous control algorithm."""
        if self.algorithm == "sac":
            return self.network.select_action(state, deterministic)
        else:  # DDPG
            return self.network.select_action(state, not deterministic)  # add_noise = not deterministic

    def store_experience(self,
                         state: np.ndarray,
                         action: np.ndarray,
                         reward: float,
                         next_state: np.ndarray,
                         done: bool):
        """Store experience for learning."""
        self.network.store_experience(state, action, reward, next_state, done)

    def learn(self) -> Dict[str, float]:
        """Update the networks."""
        return self.network.learn()

    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the current network."""
        info = {
            'algorithm': self.algorithm,
            'device': self.device,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'action_bounds': {
                'low': self.action_bounds[0].tolist(),
                'high': self.action_bounds[1].tolist()
            }
        }

        if hasattr(self.network, 'alpha'):
            info['temperature'] = self.network.alpha

        if hasattr(self.network, 'replay_buffer'):
            info['replay_buffer_size'] = len(self.network.replay_buffer)

        return info

    def save_networks(self, filepath: str):
        """Save network parameters."""
        self.network.save(filepath)
        self.logger.info(f"Saved networks to {filepath}")

    def load_networks(self, filepath: str):
        """Load network parameters."""
        self.network.load(filepath)
        self.logger.info(f"Loaded networks from {filepath}")


# Utility functions for integration with existing Agent Byte components
def create_continuous_network(state_size: int,
                              action_space: ActionSpace,
                              algorithm: str = "auto",
                              **kwargs) -> ContinuousNetworkManager:
    """
    Factory function to create continuous control networks.

    Args:
        state_size: Size of state space
        action_space: Action space specification
        algorithm: Algorithm to use ("auto", "sac", "ddpg")
        **kwargs: Additional arguments

    Returns:
        ContinuousNetworkManager instance
    """
    return ContinuousNetworkManager(
        state_size=state_size,
        action_space=action_space,
        algorithm=algorithm,
        **kwargs
    )


def is_continuous_compatible(action_space: ActionSpace) -> bool:
    """
    Check if an action space is compatible with continuous control.

    Args:
        action_space: Action space to check

    Returns:
        True if compatible with continuous control
    """
    return action_space.is_continuous() or action_space.is_hybrid()


def estimate_continuous_performance(action_space: ActionSpace) -> Dict[str, Any]:
    """
    Estimate expected performance characteristics for continuous control.

    Args:
        action_space: Action space to analyze

    Returns:
        Dictionary with performance estimates
    """
    if not is_continuous_compatible(action_space):
        return {'compatible': False, 'reason': 'Not a continuous action space'}

    action_dim = action_space.get_action_dim()

    # Estimate complexity based on action dimensionality
    if action_dim <= 2:
        complexity = "low"
        sample_efficiency = "high"
        recommended_algorithm = "ddpg"
    elif action_dim <= 10:
        complexity = "medium"
        sample_efficiency = "medium"
        recommended_algorithm = "sac"
    else:
        complexity = "high"
        sample_efficiency = "low"
        recommended_algorithm = "sac"

    if action_space.bounds:
        action_range = np.max(action_space.bounds[1] - action_space.bounds[0])
        if action_range > 100:
            complexity = "high"
            sample_efficiency = "low"

    return {
        'compatible': True,
        'complexity': complexity,
        'sample_efficiency': sample_efficiency,
        'recommended_algorithm': recommended_algorithm,
        'action_dimensionality': action_dim,
        'estimated_training_episodes': {
            'low': 1000,
            'medium': 5000,
            'high': 20000
        }[complexity]
    }


# Add this helper method to ContinuousNetworkManager class
def create_storage_manager(self, storage, agent_id: str) -> ContinuousNetworkStorageManager:
    """
    Create a storage manager for this network.

    Args:
        storage: Storage backend instance
        agent_id: Agent identifier

    Returns:
        ContinuousNetworkStorageManager instance
    """
    return ContinuousNetworkStorageManager(storage, agent_id)