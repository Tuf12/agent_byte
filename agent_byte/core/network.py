"""
Standardized neural network architecture for Agent Byte.

This module implements the transferable neural network that maintains
256-dimensional inputs regardless of the environment's native state size.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import time
import logging


class StandardizedNetwork:
    """
    Standardized Neural Network Architecture for Transfer Learning.

    Features:
    - Fixed 256-dimensional input for all environments
    - Transferable core layers
    - Environment-specific adapter layers
    - Pattern tracking for symbolic interpretation
    """

    def __init__(self, action_size: int, learning_rate: float = 0.001):
        """
        Initialize the standardized network.

        Args:
            action_size: Number of actions for the current environment
            learning_rate: Learning rate for network updates
        """
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(self.__class__.__name__)

        # Standardized architecture
        self.input_size = 256
        self.core_sizes = [512, 256, 128]  # Transferable core layers
        self.adapter_size = 64  # Environment adaptation layer

        # Initialize network layers
        self._initialize_core_layers()
        self._initialize_adapter_layer()
        self._initialize_output_layer()

        # Pattern tracking for symbolic interpretation
        self.activation_patterns = []
        self.decision_patterns = []
        self.pattern_history_size = 100

        # Metadata for transfer learning
        self.metadata = {
            'architecture_version': '3.0',
            'created_at': time.time(),
            'training_steps': 0,
            'environments_seen': set(),
            'transfer_count': 0
        }

        self.logger.info(
            f"Initialized StandardizedNetwork: {self.input_size}→{self.core_sizes}→{self.adapter_size}→{self.action_size}")

    def _initialize_core_layers(self):
        """Initialize transferable core feature layers."""
        self.core_layers = []

        layer_sizes = [self.input_size] + self.core_sizes
        for i in range(len(layer_sizes) - 1):
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i]),
                'biases': np.zeros(layer_sizes[i + 1]),
                'layer_type': 'core',
                'transferable': True,
                'activation_stats': {
                    'mean': 0.0,
                    'std': 1.0,
                    'sparsity': 0.0
                }
            }
            self.core_layers.append(layer)

    def _initialize_adapter_layer(self):
        """Initialize environment-specific adapter layer."""
        core_output_size = self.core_sizes[-1]

        self.adapter_layer = {
            'weights': np.random.randn(core_output_size, self.adapter_size) * np.sqrt(2.0 / core_output_size),
            'biases': np.zeros(self.adapter_size),
            'layer_type': 'adapter',
            'transferable': False,
            'environment_specific': True
        }

    def _initialize_output_layer(self):
        """Initialize environment-specific output layer."""
        self.output_layer = {
            'weights': np.random.randn(self.adapter_size, self.action_size) * np.sqrt(2.0 / self.adapter_size),
            'biases': np.zeros(self.action_size),
            'layer_type': 'output',
            'transferable': False,
            'environment_specific': True
        }

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            state: 256-dimensional normalized state

        Returns:
            Q-values for each action
        """
        # Validate input
        if len(state) != self.input_size:
            raise ValueError(f"Expected {self.input_size}-dimensional input, got {len(state)}")

        # Track activations for pattern analysis
        activations = [state.copy()]

        # Forward through core layers
        x = state
        for i, layer in enumerate(self.core_layers):
            z = np.dot(x, layer['weights']) + layer['biases']
            x = self._leaky_relu(z)
            activations.append(x.copy())

            # Update activation statistics
            self._update_activation_stats(layer, x)

        # Store core features for transfer learning
        self.core_features = x.copy()

        # Forward through adapter layer
        adapter_z = np.dot(x, self.adapter_layer['weights']) + self.adapter_layer['biases']
        adapter_out = self._leaky_relu(adapter_z)
        activations.append(adapter_out.copy())

        # Forward through output layer
        output_z = np.dot(adapter_out, self.output_layer['weights']) + self.output_layer['biases']
        q_values = output_z  # Linear output for Q-learning

        # Track patterns for symbolic interpretation
        self._track_activation_pattern(activations)

        return q_values

    def _leaky_relu(self, x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function."""
        return np.where(x > 0, x, alpha * x)

    def _update_activation_stats(self, layer: Dict, activations: np.ndarray):
        """Update activation statistics for a layer."""
        stats = layer['activation_stats']

        # Running average update
        alpha = 0.01
        stats['mean'] = (1 - alpha) * stats['mean'] + alpha * np.mean(activations)
        stats['std'] = (1 - alpha) * stats['std'] + alpha * np.std(activations)
        stats['sparsity'] = (1 - alpha) * stats['sparsity'] + alpha * (np.sum(activations == 0) / len(activations))

    def _track_activation_pattern(self, activations: List[np.ndarray]):
        """Track activation patterns for symbolic interpretation."""
        # Create pattern summary
        pattern = {
            'timestamp': time.time(),
            'layer_stats': []
        }

        for i, activation in enumerate(activations):
            stats = {
                'layer': i,
                'mean': float(np.mean(activation)),
                'std': float(np.std(activation)),
                'max': float(np.max(activation)),
                'min': float(np.min(activation)),
                'sparsity': float(np.sum(activation == 0) / len(activation))
            }
            pattern['layer_stats'].append(stats)

        self.activation_patterns.append(pattern)

        # Keep history manageable
        if len(self.activation_patterns) > self.pattern_history_size:
            self.activation_patterns = self.activation_patterns[-self.pattern_history_size:]

    def record_decision(self, state: np.ndarray, action: int, reward: float):
        """
        Record decision for pattern analysis.

        Args:
            state: State that led to the decision
            action: Action taken
            reward: Reward received
        """
        decision = {
            'timestamp': time.time(),
            'state_summary': {
                'mean': float(np.mean(state)),
                'std': float(np.std(state)),
                'dominant_features': np.argsort(np.abs(state))[-10:].tolist()
            },
            'action': action,
            'reward': reward,
            'core_features_summary': {
                'mean': float(np.mean(self.core_features)),
                'std': float(np.std(self.core_features)),
                'max_activation': float(np.max(self.core_features))
            } if hasattr(self, 'core_features') else None
        }

        self.decision_patterns.append(decision)

        # Keep history manageable
        if len(self.decision_patterns) > self.pattern_history_size:
            self.decision_patterns = self.decision_patterns[-self.pattern_history_size:]

    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get summary of tracked patterns for symbolic interpretation.

        Returns:
            Pattern summary including stability, trends, and features
        """
        if not self.activation_patterns:
            return {'pattern_count': 0}

        # Calculate pattern stability
        recent_patterns = self.activation_patterns[-10:] if len(
            self.activation_patterns) >= 10 else self.activation_patterns

        # Extract means from each layer
        layer_means = []
        for pattern in recent_patterns:
            means = [stat['mean'] for stat in pattern['layer_stats']]
            layer_means.append(means)

        layer_means = np.array(layer_means)
        pattern_stability = 1.0 - np.mean(np.std(layer_means, axis=0))

        # Analyze decision patterns
        recent_decisions = self.decision_patterns[-20:] if len(self.decision_patterns) >= 20 else self.decision_patterns
        action_distribution = {}
        reward_by_action = {}

        for decision in recent_decisions:
            action = decision['action']
            reward = decision['reward']

            action_distribution[action] = action_distribution.get(action, 0) + 1
            if action not in reward_by_action:
                reward_by_action[action] = []
            reward_by_action[action].append(reward)

        # Calculate average rewards
        avg_reward_by_action = {
            action: np.mean(rewards) for action, rewards in reward_by_action.items()
        }

        return {
            'pattern_count': len(self.activation_patterns),
            'decision_count': len(self.decision_patterns),
            'pattern_stability': float(pattern_stability),
            'action_distribution': action_distribution,
            'avg_reward_by_action': avg_reward_by_action,
            'recent_patterns': recent_patterns[-3:],
            'core_features_active': hasattr(self, 'core_features')
        }

    def transfer_core_layers_from(self, source_network: 'StandardizedNetwork'):
        """
        Transfer core layers from another network.

        Args:
            source_network: Network to transfer from
        """
        if self.core_sizes != source_network.core_sizes:
            raise ValueError("Incompatible network architectures")

        # Copy core layer weights
        for i, source_layer in enumerate(source_network.core_layers):
            self.core_layers[i]['weights'] = source_layer['weights'].copy()
            self.core_layers[i]['biases'] = source_layer['biases'].copy()
            self.core_layers[i]['activation_stats'] = source_layer['activation_stats'].copy()

        # Update metadata
        self.metadata['transfer_count'] += 1
        self.metadata['last_transfer'] = time.time()

        self.logger.info("Successfully transferred core layers")

    def get_state_dict(self) -> Dict[str, Any]:
        """Get network state for saving."""
        return {
            'core_layers': [
                {
                    'weights': layer['weights'].tolist(),
                    'biases': layer['biases'].tolist(),
                    'activation_stats': layer['activation_stats']
                }
                for layer in self.core_layers
            ],
            'adapter_layer': {
                'weights': self.adapter_layer['weights'].tolist(),
                'biases': self.adapter_layer['biases'].tolist()
            },
            'output_layer': {
                'weights': self.output_layer['weights'].tolist(),
                'biases': self.output_layer['biases'].tolist()
            },
            'metadata': self.metadata,
            'action_size': self.action_size
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load network state."""
        # Verify compatibility
        if state_dict['action_size'] != self.action_size:
            raise ValueError(f"Action size mismatch: expected {self.action_size}, got {state_dict['action_size']}")

        # Load core layers
        for i, layer_data in enumerate(state_dict['core_layers']):
            self.core_layers[i]['weights'] = np.array(layer_data['weights'])
            self.core_layers[i]['biases'] = np.array(layer_data['biases'])
            self.core_layers[i]['activation_stats'] = layer_data['activation_stats']

        # Load adapter layer
        self.adapter_layer['weights'] = np.array(state_dict['adapter_layer']['weights'])
        self.adapter_layer['biases'] = np.array(state_dict['adapter_layer']['biases'])

        # Load output layer
        self.output_layer['weights'] = np.array(state_dict['output_layer']['weights'])
        self.output_layer['biases'] = np.array(state_dict['output_layer']['biases'])

        # Load metadata
        self.metadata.update(state_dict['metadata'])

        self.logger.info("Successfully loaded network state")