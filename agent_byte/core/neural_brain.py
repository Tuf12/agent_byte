"""
Neural brain component for Agent Byte with PyTorch implementation.

This module implements the neural learning component of the dual brain system,
handling pattern recognition, value learning, and experience replay with proper
gradient-based optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import logging
import random

from .network import StandardizedNetwork
from ..storage.base import StorageBase


class NeuralBrain:
    """
    Neural learning engine with pattern tracking using PyTorch.

    This brain learns from experience using neural networks and tracks
    patterns for symbolic interpretation.
    """

    def __init__(self,
                 agent_id: str,
                 action_size: int,
                 storage: StorageBase,
                 config: Dict[str, Any]):
        """
        Initialize the neural brain.

        Args:
            agent_id: Unique identifier for the agent
            action_size: Number of possible actions
            storage: Storage backend for persistence
            config: Configuration parameters
        """
        self.agent_id = agent_id
        self.action_size = action_size
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger(f"NeuralBrain-{agent_id}")

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Neural networks
        self.network = StandardizedNetwork(
            action_size=action_size,
            learning_rate=config.get('learning_rate', 0.001),
            device=self.device
        )
        self.target_network = StandardizedNetwork(
            action_size=action_size,
            learning_rate=config.get('learning_rate', 0.001),
            device=self.device
        )

        # Copy initial weights to target network
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.get('learning_rate', 0.001),
            eps=1e-8
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10000,
            gamma=0.95
        )

        # Experience replay
        self.experience_buffer = deque(
            maxlen=config.get('experience_buffer_size', 5000)
        )
        self.batch_size = config.get('batch_size', 16)

        # Learning metrics
        self.training_steps = 0
        self.target_update_frequency = config.get('target_update_frequency', 1000)
        self.gamma = config.get('gamma', 0.99)

        # Loss function (Huber loss for stability)
        self.criterion = nn.SmoothL1Loss()

        # Pattern tracking
        self.learning_history = []
        self.performance_metrics = {
            'total_reward': 0.0,
            'episode_count': 0,
            'average_loss': 0.0,
            'pattern_stability': 0.0,
            'learning_rate': config.get('learning_rate', 0.001)
        }

        self.logger.info("PyTorch Neural brain initialized")

    def select_action(self, state: np.ndarray, exploration_rate: float) -> Tuple[int, Dict[str, Any]]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: 256-dimensional normalized state
            exploration_rate: Probability of random exploration

        Returns:
            Tuple of (action, decision_info)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get Q-values
        self.network.eval()  # Set to eval mode
        with torch.no_grad():
            q_values = self.network(state_tensor)
        self.network.train()  # Back to train mode

        # Convert to numpy for decision
        q_values_np = q_values.cpu().numpy()[0]

        # Epsilon-greedy selection
        if random.random() < exploration_rate:
            action = random.randint(0, self.action_size - 1)
            decision_type = 'exploration'
        else:
            action = int(np.argmax(q_values_np))
            decision_type = 'exploitation'

        # Create decision info
        decision_info = {
            'q_values': q_values_np.tolist(),
            'selected_action': action,
            'decision_type': decision_type,
            'exploration_rate': exploration_rate,
            'max_q_value': float(np.max(q_values_np)),
            'q_value_std': float(np.std(q_values_np))
        }

        return action, decision_info

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        experience = {
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done
        }

        self.experience_buffer.append(experience)

        # Record decision pattern
        self.network.record_decision(state, action, reward)

        # Update metrics
        self.performance_metrics['total_reward'] += reward

    def learn(self) -> Optional[float]:
        """
        Perform learning update using experience replay with PyTorch.

        Returns:
            Average loss or None if not enough experiences
        """
        if len(self.experience_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.experience_buffer, self.batch_size)

        # Prepare batch tensors
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)

        # Current Q values
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: Use the main network to select actions, target network to evaluate
        with torch.no_grad():
            # Select the best actions using the main network
            next_actions = self.network(next_states).max(1)[1].unsqueeze(1)
            # Evaluate using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            # Compute targets
            targets = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))

        # Compute loss
        loss = self.criterion(current_q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Update learning rate scheduler
        self.scheduler.step()

        # Update training steps
        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            self.logger.debug(f"Updated target network at step {self.training_steps}")

        # Update metrics
        loss_value = loss.item()
        alpha = 0.01
        self.performance_metrics['average_loss'] = (
                (1 - alpha) * self.performance_metrics['average_loss'] + alpha * loss_value
        )
        self.performance_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

        # Track learning progress
        self._track_learning_progress(loss_value)

        return float(loss_value)

    def _track_learning_progress(self, loss: float):
        """Track learning progress for pattern analysis."""
        progress = {
            'training_step': self.training_steps,
            'loss': float(loss),
            'average_reward': self.performance_metrics['total_reward'] / max(1, self.training_steps),
            'buffer_size': len(self.experience_buffer),
            'pattern_summary': self.network.get_pattern_summary(),
            'learning_rate': self.performance_metrics['learning_rate']
        }

        self.learning_history.append(progress)

        # Keep history manageable
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-500:]

    def get_neural_insights(self) -> Dict[str, Any]:
        """
        Get insights about neural learning for symbolic interpretation.

        Returns:
            Dictionary of neural learning insights
        """
        pattern_summary = self.network.get_pattern_summary()

        # Analyze learning trajectory
        learning_trajectory = self._analyze_learning_trajectory()

        # Identify emerging patterns
        emerging_patterns = self._identify_emerging_patterns()

        return {
            'pattern_summary': pattern_summary,
            'learning_trajectory': learning_trajectory,
            'emerging_patterns': emerging_patterns,
            'performance_metrics': self.performance_metrics.copy(),
            'training_steps': self.training_steps,
            'experience_count': len(self.experience_buffer),
            'network_stats': self.network.get_num_parameters()
        }

    def _analyze_learning_trajectory(self) -> Dict[str, Any]:
        """Analyze the trajectory of learning."""
        if len(self.learning_history) < 10:
            return {'status': 'insufficient_data'}

        recent_history = self.learning_history[-100:]

        # Calculate trends
        losses = [h['loss'] for h in recent_history]
        rewards = [h['average_reward'] for h in recent_history]

        # Simple linear regression for trend
        x = np.arange(len(losses))
        loss_trend = np.polyfit(x, losses, 1)[0] if len(losses) > 1 else 0
        reward_trend = np.polyfit(x, rewards, 1)[0] if len(rewards) > 1 else 0

        return {
            'loss_trend': float(loss_trend),
            'reward_trend': float(reward_trend),
            'is_improving': loss_trend < 0 < reward_trend,
            'stability': float(1.0 - np.std(losses[-10:]) / (np.mean(losses[-10:]) + 1e-8)),
            'current_lr': self.performance_metrics['learning_rate']
        }

    def _identify_emerging_patterns(self) -> List[Dict[str, Any]]:
        """Identify emerging behavioral patterns."""
        if not self.network.decision_patterns:
            return []

        recent_decisions = self.network.decision_patterns[-50:]
        patterns = []

        # Pattern 1: Action preferences
        action_counts = {}
        action_rewards = {}

        for decision in recent_decisions:
            action = decision['action']
            reward = decision['reward']

            action_counts[action] = action_counts.get(action, 0) + 1
            if action not in action_rewards:
                action_rewards[action] = []
            action_rewards[action].append(reward)

        # Find dominant actions
        total_actions = sum(action_counts.values())
        for action, count in action_counts.items():
            if count / total_actions > 0.3:  # Action used more than 30%
                avg_reward = np.mean(action_rewards[action])
                patterns.append({
                    'type': 'action_preference',
                    'action': action,
                    'frequency': count / total_actions,
                    'average_reward': float(avg_reward),
                    'confidence': min(1.0, count / 10)  # Confidence based on sample size
                })

        # Pattern 2: State-action correlations
        if len(recent_decisions) >= 20:
            # Simplified correlation analysis
            state_features = []
            actions = []
            rewards = []

            for decision in recent_decisions[-20:]:
                if decision['state_summary']:
                    state_features.append(decision['state_summary']['mean'])
                    actions.append(decision['action'])
                    rewards.append(decision['reward'])

            if state_features:
                # Check if certain state features correlate with actions
                state_features = np.array(state_features)
                if np.std(state_features) > 0.1:
                    patterns.append({
                        'type': 'state_action_correlation',
                        'description': 'State features influence action selection',
                        'strength': float(np.std(state_features)),
                        'confidence': 0.7
                    })

        return patterns

    def save_state(self, env_id: str) -> bool:
        """
        Save neural brain state to storage.

        Args:
            env_id: Environment identifier

        Returns:
            Success status
        """
        try:
            # Prepare brain state
            brain_state = {
                'network_state': self.network.get_state_dict_serializable(),
                'target_network_state': self.target_network.get_state_dict_serializable(),
                'optimizer_state': {
                    'state_dict': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict()
                },
                'training_steps': self.training_steps,
                'performance_metrics': self.performance_metrics,
                'learning_history': self.learning_history[-100:],  # Recent history
                'config': self.config,
                'device': str(self.device)
            }

            return self.storage.save_brain_state(self.agent_id, env_id, brain_state)

        except Exception as e:
            self.logger.error(f"Failed to save neural brain state: {e}")
            return False

    def load_state(self, env_id: str) -> bool:
        """
        Load neural brain state from storage.

        Args:
            env_id: Environment identifier

        Returns:
            Success status
        """
        try:
            brain_state = self.storage.load_brain_state(self.agent_id, env_id)
            if not brain_state:
                return False

            # Load networks
            self.network.load_state_dict_serializable(brain_state['network_state'])
            self.target_network.load_state_dict_serializable(brain_state['target_network_state'])

            # Load optimizer
            if 'optimizer_state' in brain_state:
                self.optimizer.load_state_dict(brain_state['optimizer_state']['state_dict'])
                self.scheduler.load_state_dict(brain_state['optimizer_state']['scheduler_state'])

            # Load metrics
            self.training_steps = brain_state.get('training_steps', 0)
            self.performance_metrics = brain_state.get('performance_metrics', self.performance_metrics)
            self.learning_history = brain_state.get('learning_history', [])

            self.logger.info(f"Loaded neural brain state from {env_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load neural brain state: {e}")
            return False

    def reset_episode(self):
        """Reset episode-specific metrics."""
        self.performance_metrics['episode_count'] += 1

    def enable_transfer_mode(self, freeze_core: bool = True):
        """
        Enable to transfer learning mode.

        Args:
            freeze_core: Whether to freeze core layers
        """
        if freeze_core:
            self.network.freeze_core_layers()
            # Only optimize adapter and output layers
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                lr=self.config.get('learning_rate', 0.001) * 0.1  # Lower LR for fine-tuning
            )
        self.logger.info(f"Transfer mode enabled (freeze_core={freeze_core})")

    def disable_transfer_mode(self):
        """Disable transfer learning mode."""
        self.network.unfreeze_core_layers()
        # Re-initialize optimizer with all parameters
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        self.logger.info("Transfer mode disabled")