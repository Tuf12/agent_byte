"""
Standardized neural network architecture for Agent Byte using PyTorch.

This module implements the transferable neural network that maintains
256-dimensional inputs regardless of the environment's native state size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import logging


class StandardizedNetwork(nn.Module):
    """
    Standardized Neural Network Architecture for Transfer Learning using PyTorch.

    Features:
    - Fixed 256-dimensional input for all environments
    - Transferable core layers
    - Environment-specific adapter layers
    - Pattern tracking for symbolic interpretation
    - Proper gradient-based learning
    """

    def __init__(self, action_size: int, learning_rate: float = 0.001, device: str = None):
        """
        Initialize the standardized network.

        Args:
            action_size: Number of actions for the current environment
            learning_rate: Learning rate for network updates
            device: Torch device ('cuda' or 'cpu')
        """
        super(StandardizedNetwork, self).__init__()

        self.action_size = action_size
        self.learning_rate = learning_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)

        # Standardized architecture dimensions
        self.input_size = 256
        self.core_sizes = [512, 256, 128]
        self.adapter_size = 64

        # Core transferable layers
        self.core_layers = nn.ModuleList()

        # Build core layers
        layer_sizes = [self.input_size] + self.core_sizes
        for i in range(len(layer_sizes) - 1):
            self.core_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Environment-specific adapter layer
        self.adapter_layer = nn.Linear(self.core_sizes[-1], self.adapter_size)

        # Environment-specific output layer
        self.output_layer = nn.Linear(self.adapter_size, self.action_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)

        # Initialize weights
        self._initialize_weights()

        # Move to a device
        self.to(self.device)

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
            'transfer_count': 0,
            'device': str(self.device)
        }

        # Track activations for pattern analysis
        self.core_features = None
        self.activation_stats = {i: {'mean': 0.0, 'std': 1.0, 'sparsity': 0.0}
                                for i in range(len(self.core_layers))}

        self.logger.info(
            f"Initialized PyTorch StandardizedNetwork: {self.input_size}→{self.core_sizes}→"
            f"{self.adapter_size}→{self.action_size} on {self.device}")

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.core_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.adapter_layer.weight)
        nn.init.zeros_(self.adapter_layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: 256-dimensional normalized state.

        Returns:
            Q-values for each action
        """
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            # Ensure tensor is on a correct device
            state = state.to(self.device)

        # Ensure correct shape
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Validate input dimensions
        if state.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size}-dimensional input, got {state.shape[1]}")

        # Track activations for pattern analysis
        activations = [state.clone().detach().cpu()]

        # Forward through core layers
        x = state
        for i, layer in enumerate(self.core_layers):
            x = layer(x)
            x = F.leaky_relu(x, negative_slope=0.01)

            # Apply dropout on all but last core layer
            if i < len(self.core_layers) - 1:
                x = self.dropout(x)

            # Track activations
            activations.append(x.clone().detach().cpu())

            # Update activation statistics
            self._update_activation_stats(i, x)

        # Store core features for transfer learning
        self.core_features = x.clone().detach()

        # Forward through adapter layer
        x = self.adapter_layer(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        activations.append(x.clone().detach().cpu())

        # Forward through output layer (no activation for Q-values)
        q_values = self.output_layer(x)

        # Track patterns for symbolic interpretation (only during inference)
        if not self.training:
            self._track_activation_pattern(activations)

        return q_values

    def _update_activation_stats(self, layer_idx: int, activations: torch.Tensor):
        """Update activation statistics for a layer."""
        with torch.no_grad():
            stats = self.activation_stats[layer_idx]

            # Running average update
            alpha = 0.01
            current_mean = activations.mean().item()
            current_std = activations.std().item()
            current_sparsity = (activations == 0).float().mean().item()

            stats['mean'] = (1 - alpha) * stats['mean'] + alpha * current_mean
            stats['std'] = (1 - alpha) * stats['std'] + alpha * current_std
            stats['sparsity'] = (1 - alpha) * stats['sparsity'] + alpha * current_sparsity

    def _track_activation_pattern(self, activations: List[torch.Tensor]):
        """Track activation patterns for symbolic interpretation."""
        pattern = {
            'timestamp': time.time(),
            'layer_stats': []
        }

        for i, activation in enumerate(activations):
            # Convert to numpy for stats
            act_np = activation.numpy()
            if act_np.ndim > 1:
                act_np = act_np[0]  # Take the first batch element

            stats = {
                'layer': i,
                'mean': float(np.mean(act_np)),
                'std': float(np.std(act_np)),
                'max': float(np.max(act_np)),
                'min': float(np.min(act_np)),
                'sparsity': float(np.sum(act_np == 0) / len(act_np))
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
                'mean': float(self.core_features.mean().item()),
                'std': float(self.core_features.std().item()),
                'max_activation': float(self.core_features.max().item())
            } if self.core_features is not None else None
        }

        self.decision_patterns.append(decision)

        # Keep history manageable
        if len(self.decision_patterns) > self.pattern_history_size:
            self.decision_patterns = self.decision_patterns[-self.pattern_history_size:]

    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get a summary of tracked patterns for symbolic interpretation.

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
        recent_decisions = self.decision_patterns[-20:] if len(
            self.decision_patterns) >= 20 else self.decision_patterns
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
            'core_features_active': self.core_features is not None
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
        with torch.no_grad():
            for i, (target_layer, source_layer) in enumerate(
                    zip(self.core_layers, source_network.core_layers)):
                target_layer.weight.copy_(source_layer.weight)
                target_layer.bias.copy_(source_layer.bias)

        # Copy activation statistics
        self.activation_stats = source_network.activation_stats.copy()

        # Update metadata
        self.metadata['transfer_count'] += 1
        self.metadata['last_transfer'] = time.time()

        self.logger.info("Successfully transferred core layers")

    def freeze_core_layers(self):
        """Freeze core layers for transfer learning."""
        for layer in self.core_layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.logger.info("Core layers frozen")

    def unfreeze_core_layers(self):
        """Unfreeze core layers for fine-tuning."""
        for layer in self.core_layers:
            for param in layer.parameters():
                param.requires_grad = True
        self.logger.info("Core layers unfrozen")

    def get_state_dict_serializable(self) -> Dict[str, Any]:
        """
        Get network state in a serializable format.

        Returns:
            Dictionary containing all network states
        """
        # Get PyTorch state dict
        pytorch_state = self.state_dict()

        # Convert to serializable format
        serializable_state = {}
        for key, tensor in pytorch_state.items():
            serializable_state[key] = tensor.cpu().numpy().tolist()

        return {
            'pytorch_state': serializable_state,
            'metadata': self.metadata,
            'action_size': self.action_size,
            'activation_stats': self.activation_stats,
            'architecture': {
                'input_size': self.input_size,
                'core_sizes': self.core_sizes,
                'adapter_size': self.adapter_size
            }
        }

    def load_state_dict_serializable(self, state_dict: Dict[str, Any]):
        """
        Load network state from serializable format.

        Args:
            state_dict: Dictionary containing network state
        """
        # Verify compatibility
        if state_dict['action_size'] != self.action_size:
            raise ValueError(
                f"Action size mismatch: expected {self.action_size}, "
                f"got {state_dict['action_size']}")

        # Load PyTorch state
        pytorch_state = {}
        for key, value in state_dict['pytorch_state'].items():
            pytorch_state[key] = torch.tensor(value, device=self.device)

        self.load_state_dict(pytorch_state)

        # Load metadata
        self.metadata.update(state_dict['metadata'])
        self.activation_stats = state_dict.get('activation_stats', self.activation_stats)

        self.logger.info("Successfully loaded network state")

    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters in different parts of the network."""
        core_params = sum(p.numel() for layer in self.core_layers
                         for p in layer.parameters())
        adapter_params = sum(p.numel() for p in self.adapter_layer.parameters())
        output_params = sum(p.numel() for p in self.output_layer.parameters())

        return {
            'core': core_params,
            'adapter': adapter_params,
            'output': output_params,
            'total': core_params + adapter_params + output_params
        }