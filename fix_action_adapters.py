"""
Action Space Adapters for Agent Byte v3.0

These adapters enable conversion between different action space types,
allowing the agent to work with any environment regardless of its
native action space.
"""

import numpy as np
from typing import Union, Dict, Any, List, Tuple, Optional
from abc import ABC, abstractmethod
import logging

from .interfaces import ActionSpace, ActionSpaceType


class ActionAdapter(ABC):
    """
    Base class for action space adapters.

    Action adapters convert between different action space types,
    enabling the agent to work with environments that have different
    action representations.
    """

    def __init__(self, source_space: ActionSpace, target_space: ActionSpace):
        """
        Initialize the action adapter.

        Args:
            source_space: The action space the agent produces
            target_space: The action space the environment expects
        """
        self.source_space = source_space
        self.target_space = target_space
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Validate compatibility
        self._validate_compatibility()

    @abstractmethod
    def _validate_compatibility(self):
        """Validate that the adapter can convert between the spaces."""
        pass

    @abstractmethod
    def adapt_action(self, action: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Convert an action from source space to target space.

        Args:
            action: Action in source space format

        Returns:
            Action in target space format
        """
        pass

    @abstractmethod
    def reverse_adapt_action(self, action: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Convert an action from target space back to source space.

        Args:
            action: Action in target space format

        Returns:
            Action in source space format
        """
        pass

    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about this adapter."""
        return {
            'adapter_type': self.__class__.__name__,
            'source_space': self.source_space.to_dict(),
            'target_space': self.target_space.to_dict(),
            'conversion_loss': self.estimate_conversion_loss()
        }

    def estimate_conversion_loss(self) -> float:
        """
        Estimate information loss from conversion (0.0 = no loss, 1.0 = total loss).

        Returns:
            Estimated conversion loss as a float between 0 and 1
        """
        return 0.0  # Default: no loss


class DiscreteToContiunousAdapter(ActionAdapter):
    """
    Adapter to convert discrete actions to continuous actions.

    This is useful when the agent thinks in discrete terms but
    the environment requires continuous control.
    """

    def __init__(self,
                 source_space: ActionSpace,
                 target_space: ActionSpace,
                 strategy: str = "uniform_grid"):
        """
        Initialize discrete to continuous adapter.

        Args:
            source_space: Discrete action space
            target_space: Continuous action space
            strategy: Conversion strategy ("uniform_grid", "random_sample", "gaussian")
        """
        self.strategy = strategy
        super().__init__(source_space, target_space)

        # Pre-compute action mappings for efficiency
        self._precompute_mappings()

    def _validate_compatibility(self):
        """Validate that source is discrete and target is continuous."""
        if not self.source_space.is_discrete():
            raise ValueError("Source space must be discrete")
        if not self.target_space.is_continuous():
            raise ValueError("Target space must be continuous")

    def _precompute_mappings(self):
        """Pre-compute the mapping from discrete actions to continuous values."""
        num_discrete = self.source_space.size
        target_low, target_high = self.target_space.bounds
        target_dim = self.target_space.get_action_dim()

        self.action_mappings = {}

        if self.strategy == "uniform_grid":
            # Create uniform grid in continuous space
            for discrete_action in range(num_discrete):
                if target_dim == 1:
                    # 1D case: linearly space actions
                    continuous_action = target_low + (discrete_action / (num_discrete - 1)) * (target_high - target_low)
                else:
                    # Multi-dimensional case: use combinations
                    actions_per_dim = int(np.ceil(num_discrete ** (1.0 / target_dim)))
                    indices = self._discrete_to_multi_index(discrete_action, actions_per_dim, target_dim)

                    continuous_action = np.zeros(target_dim)
                    for dim in range(target_dim):
                        continuous_action[dim] = target_low[dim] + (indices[dim] / (actions_per_dim - 1)) * (
                                    target_high[dim] - target_low[dim])

                self.action_mappings[discrete_action] = np.clip(continuous_action, target_low, target_high)

        elif self.strategy == "random_sample":
            # Random sampling within bounds
            np.random.seed(42)  # For reproducibility
            for discrete_action in range(num_discrete):
                continuous_action = np.random.uniform(target_low, target_high)
                self.action_mappings[discrete_action] = continuous_action

        elif self.strategy == "gaussian":
            # Gaussian distribution around grid points
            for discrete_action in range(num_discrete):
                if target_dim == 1:
                    center = target_low + (discrete_action / (num_discrete - 1)) * (target_high - target_low)
                    std = (target_high - target_low) / (4 * num_discrete)  # 4-sigma coverage
                    continuous_action = np.random.normal(center, std)
                else:
                    actions_per_dim = int(np.ceil(num_discrete ** (1.0 / target_dim)))
                    indices = self._discrete_to_multi_index(discrete_action, actions_per_dim, target_dim)

                    continuous_action = np.zeros(target_dim)
                    for dim in range(target_dim):
                        center = target_low[dim] + (indices[dim] / (actions_per_dim - 1)) * (
                                    target_high[dim] - target_low[dim])
                        std = (target_high[dim] - target_low[dim]) / (4 * actions_per_dim)
                        continuous_action[dim] = np.random.normal(center, std)

                self.action_mappings[discrete_action] = np.clip(continuous_action, target_low, target_high)

    def _discrete_to_multi_index(self, discrete_action: int, actions_per_dim: int, target_dim: int) -> List[int]:
        """Convert discrete action to multi-dimensional indices."""
        indices = []
        remaining = discrete_action
        for _ in range(target_dim):
            indices.append(remaining % actions_per_dim)
            remaining //= actions_per_dim
        return indices

    def adapt_action(self, action: int) -> np.ndarray:
        """Convert discrete action to continuous action."""
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Expected discrete action (int), got {type(action)}")

        if action not in self.action_mappings:
            raise ValueError(f"Discrete action {action} not in valid range [0, {self.source_space.size - 1}]")

        return self.action_mappings[action].copy()

    def reverse_adapt_action(self, action: np.ndarray) -> int:
        """Convert continuous action back to closest discrete action."""
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        # Find closest discrete action by Euclidean distance
        min_distance = float('inf')
        closest_discrete = 0

        for discrete_action, continuous_mapping in self.action_mappings.items():
            distance = np.linalg.norm(action - continuous_mapping)
            if distance < min_distance:
                min_distance = distance
                closest_discrete = discrete_action

        return closest_discrete

    def estimate_conversion_loss(self) -> float:
        """Estimate information loss from discretization."""
        target_dim = self.target_space.get_action_dim()
        target_volume = np.prod(self.target_space.bounds[1] - self.target_space.bounds[0])

        # Loss increases with dimensionality and decreases with more discrete actions
        discrete_density = self.source_space.size / target_volume
        loss = 1.0 / (1.0 + discrete_density / target_dim)

        return min(loss, 0.9)  # Cap at 90% loss


class ContinuousToDiscreteAdapter(ActionAdapter):
    """
    Adapter to convert continuous actions to discrete actions.

    This is useful when the agent produces continuous actions but
    the environment only accepts discrete actions.
    """

    def __init__(self,
                 source_space: ActionSpace,
                 target_space: ActionSpace,
                 strategy: str = "quantization"):
        """
        Initialize continuous to discrete adapter.

        Args:
            source_space: Continuous action space
            target_space: Discrete action space
            strategy: Conversion strategy ("quantization", "threshold", "argmax")
        """
        self.strategy = strategy
        super().__init__(source_space, target_space)

        # Set up discretization parameters
        self._setup_discretization()

    def _validate_compatibility(self):
        """Validate that source is continuous and target is discrete."""
        if not self.source_space.is_continuous():
            raise ValueError("Source space must be continuous")
        if not self.target_space.is_discrete():
            raise ValueError("Target space must be discrete")

    def _setup_discretization(self):
        """Set up discretization parameters based on strategy."""
        source_low, source_high = self.source_space.bounds
        source_dim = self.source_space.get_action_dim()
        num_discrete = self.target_space.size

        if self.strategy == "quantization":
            # Divide continuous space into discrete bins
            if source_dim == 1:
                self.bin_edges = np.linspace(source_low[0], source_high[0], num_discrete + 1)
            else:
                # For multi-dimensional, use principal component or simple slicing
                self.bin_edges = []
                actions_per_dim = int(np.ceil(num_discrete ** (1.0 / source_dim)))
                for dim in range(source_dim):
                    edges = np.linspace(source_low[dim], source_high[dim], actions_per_dim + 1)
                    self.bin_edges.append(edges)

        elif self.strategy == "threshold":
            # Use fixed thresholds (works best for 1D)
            if source_dim == 1:
                self.thresholds = np.linspace(source_low[0], source_high[0], num_discrete - 1)
            else:
                # For multi-dim, use simple midpoint thresholds
                self.thresholds = []
                for dim in range(source_dim):
                    mid = (source_low[dim] + source_high[dim]) / 2
                    self.thresholds.append(mid)

        elif self.strategy == "argmax":
            # Treat continuous vector as logits and take argmax
            if source_dim < num_discrete:
                raise ValueError(f"Argmax strategy requires source_dim ({source_dim}) >= target_size ({num_discrete})")

    def adapt_action(self, action: np.ndarray) -> int:
        """Convert continuous action to discrete action."""
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        if self.strategy == "quantization":
            return self._quantize_action(action)
        elif self.strategy == "threshold":
            return self._threshold_action(action)
        elif self.strategy == "argmax":
            return self._argmax_action(action)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _quantize_action(self, action: np.ndarray) -> int:
        """Quantize continuous action into discrete bins."""
        source_dim = self.source_space.get_action_dim()

        if source_dim == 1:
            # 1D case: simple binning
            bin_idx = np.digitize(action[0], self.bin_edges) - 1
            return np.clip(bin_idx, 0, self.target_space.size - 1)
        else:
            # Multi-dimensional case: convert to single index
            actions_per_dim = len(self.bin_edges[0]) - 1
            indices = []

            for dim in range(min(source_dim, len(self.bin_edges))):
                bin_idx = np.digitize(action[dim], self.bin_edges[dim]) - 1
                bin_idx = np.clip(bin_idx, 0, actions_per_dim - 1)
                indices.append(bin_idx)

            # Convert multi-dimensional indices to single discrete action
            discrete_action = 0
            for i, idx in enumerate(indices):
                discrete_action += idx * (actions_per_dim ** i)

            return min(discrete_action, self.target_space.size - 1)

    def _threshold_action(self, action: np.ndarray) -> int:
        """Use threshold-based discretization."""
        source_dim = self.source_space.get_action_dim()

        if source_dim == 1:
            # Count how many thresholds the action exceeds
            return int(np.sum(action[0] > self.thresholds))
        else:
            # For multi-dim, use simple binary encoding
            binary_code = []
            for dim in range(min(source_dim, len(self.thresholds))):
                binary_code.append(1 if action[dim] > self.thresholds[dim] else 0)

            # Convert binary to decimal
            discrete_action = 0
            for i, bit in enumerate(binary_code):
                discrete_action += bit * (2 ** i)

            return min(discrete_action, self.target_space.size - 1)

    def _argmax_action(self, action: np.ndarray) -> int:
        """Take argmax of continuous vector."""
        # Use first num_discrete elements as logits
        logits = action[:self.target_space.size]
        return int(np.argmax(logits))

    def reverse_adapt_action(self, action: int) -> np.ndarray:
        """Convert discrete action back to continuous representation."""
        if self.strategy == "quantization":
            return self._dequantize_action(action)
        elif self.strategy == "threshold":
            return self._dethreshold_action(action)
        elif self.strategy == "argmax":
            return self._deargmax_action(action)

    def _dequantize_action(self, action: int) -> np.ndarray:
        """Convert discrete action back to continuous bin center."""
        source_dim = self.source_space.get_action_dim()
        source_low, source_high = self.source_space.bounds

        if source_dim == 1:
            # Use bin center
            bin_width = (source_high[0] - source_low[0]) / self.target_space.size
            continuous_action = source_low[0] + (action + 0.5) * bin_width
            return np.array([continuous_action])
        else:
            # Multi-dimensional reconstruction
            actions_per_dim = int(np.ceil(self.target_space.size ** (1.0 / source_dim)))
            continuous_action = np.zeros(source_dim)

            remaining = action
            for dim in range(source_dim):
                bin_idx = remaining % actions_per_dim
                remaining //= actions_per_dim

                bin_width = (source_high[dim] - source_low[dim]) / actions_per_dim
                continuous_action[dim] = source_low[dim] + (bin_idx + 0.5) * bin_width

            return continuous_action

    def _dethreshold_action(self, action: int) -> np.ndarray:
        """Convert discrete action back using threshold strategy."""
        source_dim = self.source_space.get_action_dim()
        source_low, source_high = self.source_space.bounds

        if source_dim == 1:
            # Map action back to continuous value
            if action == 0:
                return np.array([(source_low[0] + self.thresholds[0]) / 2])
            elif action == len(self.thresholds):
                return np.array([(self.thresholds[-1] + source_high[0]) / 2])
            else:
                return np.array([(self.thresholds[action - 1] + self.thresholds[action]) / 2])
        else:
            # Multi-dimensional binary decoding
            continuous_action = np.zeros(source_dim)
            for dim in range(min(source_dim, len(self.thresholds))):
                bit = (action >> dim) & 1
                if bit:
                    continuous_action[dim] = (self.thresholds[dim] + source_high[dim]) / 2
                else:
                    continuous_action[dim] = (source_low[dim] + self.thresholds[dim]) / 2

            return continuous_action

    def _deargmax_action(self, action: int) -> np.ndarray:
        """Convert discrete action back to one-hot continuous vector."""
        source_dim = self.source_space.get_action_dim()
        continuous_action = np.zeros(source_dim)

        if action < source_dim:
            continuous_action[action] = 1.0

        return continuous_action

    def estimate_conversion_loss(self) -> float:
        """Estimate information loss from discretization."""
        source_dim = self.source_space.get_action_dim()

        # Loss is higher when mapping high-dimensional continuous to low-dimensional discrete
        if self.strategy == "argmax":
            loss = max(0.0, 1.0 - self.target_space.size / source_dim)
        else:
            # For quantization/threshold, loss depends on discretization resolution
            resolution = self.target_space.size / source_dim
            loss = 1.0 / (1.0 + resolution)

        return min(loss, 0.95)  # Cap at 95% loss


class HybridActionAdapter(ActionAdapter):
    """
    Adapter for hybrid action spaces containing both discrete and continuous components.

    This adapter can handle environments that require both discrete choices
    and continuous parameters.
    """

    def __init__(self,
                 source_space: ActionSpace,
                 target_space: ActionSpace):
        """
        Initialize hybrid action adapter.

        Args:
            source_space: Source action space (any type)
            target_space: Hybrid action space
        """
        super().__init__(source_space, target_space)

        # Extract hybrid components
        self.discrete_size, self.continuous_dim = target_space.size
        self.continuous_bounds = target_space.bounds

        # Create sub-adapters if needed
        self._setup_sub_adapters()

    def _validate_compatibility(self):
        """Validate compatibility with hybrid target space."""
        if not self.target_space.is_hybrid():
            raise ValueError("Target space must be hybrid")

    def _setup_sub_adapters(self):
        """Set up sub-adapters for discrete and continuous components."""
        from .interfaces import create_discrete_action_space, create_continuous_action_space

        # Create component action spaces
        discrete_target = create_discrete_action_space(self.discrete_size)
        continuous_target = create_continuous_action_space(
            self.continuous_dim,
            self.continuous_bounds[0],
            self.continuous_bounds[1]
        )

        self.sub_adapters = {}

        if self.source_space.is_discrete():
            # Split discrete actions between discrete and continuous components
            source_size = self.source_space.size
            discrete_portion = min(source_size // 2, self.discrete_size)

            if discrete_portion > 0:
                discrete_source = create_discrete_action_space(discrete_portion)
                self.sub_adapters['discrete'] = None  # Direct mapping

            if source_size > discrete_portion:
                continuous_source = create_discrete_action_space(source_size - discrete_portion)
                self.sub_adapters['continuous'] = DiscreteToContiunousAdapter(
                    continuous_source, continuous_target
                )

        elif self.source_space.is_continuous():
            # Split continuous vector between discrete and continuous components
            source_dim = self.source_space.get_action_dim()
            discrete_portion = min(source_dim, self.discrete_size)

            if discrete_portion > 0:
                discrete_source = create_continuous_action_space(
                    discrete_portion,
                    self.source_space.bounds[0][:discrete_portion],
                    self.source_space.bounds[1][:discrete_portion]
                )
                self.sub_adapters['discrete'] = ContinuousToDiscreteAdapter(
                    discrete_source, discrete_target
                )

            if source_dim > discrete_portion:
                continuous_source = create_continuous_action_space(
                    source_dim - discrete_portion,
                    self.source_space.bounds[0][discrete_portion:],
                    self.source_space.bounds[1][discrete_portion:]
                )
                # May need scaling/clipping adapter
                self.sub_adapters['continuous'] = None  # Direct mapping for now

    def adapt_action(self, action: Union[int, np.ndarray]) -> Dict[str, Union[int, np.ndarray]]:
        """Convert action to hybrid format."""
        hybrid_action = {'discrete': 0, 'continuous': np.zeros(self.continuous_dim)}

        if self.source_space.is_discrete():
            # Map discrete action to hybrid components
            discrete_action = int(action)

            # Simple strategy: use action modulo for discrete, remainder for continuous
            hybrid_action['discrete'] = discrete_action % self.discrete_size

            if 'continuous' in self.sub_adapters and self.sub_adapters['continuous']:
                continuous_input = discrete_action // self.discrete_size
                hybrid_action['continuous'] = self.sub_adapters['continuous'].adapt_action(continuous_input)
            else:
                # Default continuous values (could be learned)
                hybrid_action['continuous'] = np.zeros(self.continuous_dim)

        elif self.source_space.is_continuous():
            # Split continuous vector
            action = np.array(action)

            if 'discrete' in self.sub_adapters and self.sub_adapters['discrete']:
                discrete_input = action[:self.discrete_size] if len(action) >= self.discrete_size else action
                hybrid_action['discrete'] = self.sub_adapters['discrete'].adapt_action(discrete_input)
            else:
                # Use first element for discrete choice
                hybrid_action['discrete'] = int(np.argmax(action[:self.discrete_size])) if len(
                    action) >= self.discrete_size else 0

            # Use remaining elements for continuous
            if len(action) > self.discrete_size:
                continuous_input = action[self.discrete_size:]
                # Truncate or pad to match continuous_dim
                if len(continuous_input) >= self.continuous_dim:
                    hybrid_action['continuous'] = continuous_input[:self.continuous_dim]
                else:
                    padded = np.zeros(self.continuous_dim)
                    padded[:len(continuous_input)] = continuous_input
                    hybrid_action['continuous'] = padded

                # Clip to bounds
                hybrid_action['continuous'] = np.clip(
                    hybrid_action['continuous'],
                    self.continuous_bounds[0],
                    self.continuous_bounds[1]
                )

        return hybrid_action

    def reverse_adapt_action(self, action: Dict[str, Union[int, np.ndarray]]) -> Union[int, np.ndarray]:
        """Convert hybrid action back to source format."""
        discrete_part = action['discrete']
        continuous_part = action['continuous']

        if self.source_space.is_discrete():
            # Combine discrete and continuous into single discrete action
            discrete_action = discrete_part

            if 'continuous' in self.sub_adapters and self.sub_adapters['continuous']:
                continuous_discrete = self.sub_adapters['continuous'].reverse_adapt_action(continuous_part)
                discrete_action += continuous_discrete * self.discrete_size

            return min(discrete_action, self.source_space.size - 1)

        elif self.source_space.is_continuous():
            # Combine into continuous vector
            source_dim = self.source_space.get_action_dim()
            continuous_action = np.zeros(source_dim)

            # Convert discrete to continuous representation
            if 'discrete' in self.sub_adapters and self.sub_adapters['discrete']:
                discrete_continuous = self.sub_adapters['discrete'].reverse_adapt_action(discrete_part)
                continuous_action[:len(discrete_continuous)] = discrete_continuous
            else:
                # One-hot encoding for discrete part
                if discrete_part < source_dim:
                    continuous_action[discrete_part] = 1.0

            # Add continuous part
            continuous_start = min(self.discrete_size, source_dim)
            continuous_end = min(continuous_start + len(continuous_part), source_dim)
            continuous_action[continuous_start:continuous_end] = continuous_part[:continuous_end - continuous_start]

            return continuous_action

    def estimate_conversion_loss(self) -> float:
        """Estimate information loss from hybrid conversion."""
        # Loss depends on how well source space maps to hybrid components
        discrete_loss = 0.0
        continuous_loss = 0.0

        if 'discrete' in self.sub_adapters and self.sub_adapters['discrete']:
            discrete_loss = self.sub_adapters['discrete'].estimate_conversion_loss()

        if 'continuous' in self.sub_adapters and self.sub_adapters['continuous']:
            continuous_loss = self.sub_adapters['continuous'].estimate_conversion_loss()

        # Average loss weighted by component sizes
        discrete_weight = 1.0 / (1.0 + self.continuous_dim)
        continuous_weight = self.continuous_dim / (1.0 + self.continuous_dim)

        return discrete_weight * discrete_loss + continuous_weight * continuous_loss


class ParameterizedActionAdapter(ActionAdapter):
    """
    Adapter for parameterized actions using neural networks.

    This adapter uses neural networks to learn the mapping between
    action spaces, enabling more sophisticated conversions.
    """

    def __init__(self,
                 source_space: ActionSpace,
                 target_space: ActionSpace,
                 hidden_size: int = 64):
        """
        Initialize parameterized action adapter.

        Args:
            source_space: Source action space
            target_space: Target action space
            hidden_size: Hidden layer size for neural networks
        """
        self.hidden_size = hidden_size
        self.networks_initialized = False

        super().__init__(source_space, target_space)

        # Initialize neural networks for adaptation
        self._initialize_networks()

    def _validate_compatibility(self):
        """Parameterized adapters can work with any space types."""
        pass  # No specific validation needed

    def _initialize_networks(self):
        """Initialize neural networks for action adaptation."""
        try:
            import torch
            import torch.nn as nn

            # Determine input and output sizes
            source_size = self._get_space_size(self.source_space)
            target_size = self._get_space_size(self.target_space)

            # Forward network (source -> target)
            self.forward_net = nn.Sequential(
                nn.Linear(source_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, target_size)
            )

            # Reverse network (target -> source)
            self.reverse_net = nn.Sequential(
                nn.Linear(target_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, source_size)
            )

            self.networks_initialized = True
            self.logger.info("Neural networks initialized for parameterized action adaptation")

        except ImportError:
            self.logger.warning("PyTorch not available, using fallback linear adaptation")
            self.networks_initialized = False

    def _get_space_size(self, space: ActionSpace) -> int:
        """Get the size needed to represent an action space as a vector."""
        if space.is_discrete():
            return space.size  # One-hot encoding
        elif space.is_continuous():
            return space.get_action_dim()
        elif space.is_hybrid():
            discrete_size, continuous_dim = space.size
            return discrete_size + continuous_dim
        else:
            return 1

    def _action_to_vector(self, action: Union[int, np.ndarray], space: ActionSpace) -> np.ndarray:
        """Convert action to vector representation."""
        if space.is_discrete():
            # One-hot encoding
            vector = np.zeros(space.size)
            if isinstance(action, (int, np.integer)):
                vector[action] = 1.0
            return vector

        elif space.is_continuous():
            return np.array(action).flatten()

        elif space.is_hybrid():
            discrete_size, continuous_dim = space.size
            vector = np.zeros(discrete_size + continuous_dim)

            if isinstance(action, dict):
                # One-hot for discrete part
                discrete_action = action['discrete']
                vector[discrete_action] = 1.0

                # Direct for continuous part
                continuous_action = action['continuous']
                vector[discrete_size:] = continuous_action[:continuous_dim]

            return vector

        return np.array([0.0])

    def _vector_to_action(self, vector: np.ndarray, space: ActionSpace) -> Union[int, np.ndarray, Dict]:
        """Convert vector representation back to action."""
        if space.is_discrete():
            return int(np.argmax(vector))

        elif space.is_continuous():
            action = vector[:space.get_action_dim()]
            if space.bounds:
                action = np.clip(action, space.bounds[0], space.bounds[1])
            return action

        elif space.is_hybrid():
            discrete_size, continuous_dim = space.size

            discrete_action = int(np.argmax(vector[:discrete_size]))
            continuous_action = vector[discrete_size:discrete_size + continuous_dim]

            if space.bounds:
                continuous_action = np.clip(continuous_action, space.bounds[0], space.bounds[1])

            return {
                'discrete': discrete_action,
                'continuous': continuous_action
            }

        return 0

    def adapt_action(self, action: Union[int, np.ndarray]) -> Union[int, np.ndarray, Dict]:
        """Convert action using neural network."""
        if not self.networks_initialized:
            # Fallback to simple linear mapping
            return self._linear_adapt_action(action)

        try:
            import torch

            # Convert to vector
            input_vector = self._action_to_vector(action, self.source_space)
            input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                output_tensor = self.forward_net(input_tensor)
                output_vector = output_tensor.squeeze(0).numpy()

            # Convert back to action
            return self._vector_to_action(output_vector, self.target_space)

        except Exception as e:
            self.logger.warning(f"Neural network adaptation failed: {e}, using fallback")
            return self._linear_adapt_action(action)

    def reverse_adapt_action(self, action: Union[int, np.ndarray, Dict]) -> Union[int, np.ndarray]:
        """Convert action back using reverse neural network."""
        if not self.networks_initialized:
            return self._linear_reverse_adapt_action(action)

        try:
            import torch

            # Convert to vector
            input_vector = self._action_to_vector(action, self.target_space)
            input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)

            # Reverse pass
            with torch.no_grad():
                output_tensor = self.reverse_net(input_tensor)
                output_vector = output_tensor.squeeze(0).numpy()

            # Convert back to action
            return self._vector_to_action(output_vector, self.source_space)

        except Exception as e:
            self.logger.warning(f"Neural network reverse adaptation failed: {e}, using fallback")
            return self._linear_reverse_adapt_action(action)

    def _linear_adapt_action(self, action: Union[int, np.ndarray]) -> Union[int, np.ndarray, Dict]:
        """Simple linear fallback adaptation."""
        # This is a simplified fallback - in practice, you might want more sophisticated logic
        if self.target_space.is_discrete() and self.source_space.is_discrete():
            # Simple modulo mapping
            return int(action) % self.target_space.size

        elif self.target_space.is_continuous() and self.source_space.is_discrete():
            # Use discrete to continuous adapter
            fallback_adapter = DiscreteToContiunousAdapter(self.source_space, self.target_space)
            return fallback_adapter.adapt_action(action)

        elif self.target_space.is_discrete() and self.source_space.is_continuous():
            # Use continuous to discrete adapter
            fallback_adapter = ContinuousToDiscreteAdapter(self.source_space, self.target_space)
            return fallback_adapter.adapt_action(action)

        else:
            # Default: return some valid action
            return self.target_space.sample()

    def _linear_reverse_adapt_action(self, action: Union[int, np.ndarray, Dict]) -> Union[int, np.ndarray]:
        """Simple linear fallback reverse adaptation."""
        # Similar fallback logic for reverse direction
        if self.source_space.is_discrete() and self.target_space.is_discrete():
            return int(action) % self.source_space.size
        else:
            return self.source_space.sample()

    def estimate_conversion_loss(self) -> float:
        """Estimate conversion loss for parameterized adapter."""
        if self.networks_initialized:
            # Neural networks can potentially achieve low loss with training
            return 0.1  # Assume good neural network performance
        else:
            # Fallback methods have higher loss
            return 0.5


# Factory function for creating appropriate adapters
def create_action_adapter(source_space: ActionSpace,
                          target_space: ActionSpace,
                          adapter_type: str = "auto",
                          **kwargs) -> ActionAdapter:
    """
    Create an appropriate action adapter for converting between action spaces.

    Args:
        source_space: The action space the agent produces
        target_space: The action space the environment expects
        adapter_type: Type of adapter to create ("auto", "discrete_to_continuous",
                     "continuous_to_discrete", "hybrid", "parameterized")
        **kwargs: Additional arguments for specific adapters

    Returns:
        ActionAdapter instance for converting between the spaces
    """
    if adapter_type == "auto":
        # Automatically select adapter type
        if source_space.is_discrete() and target_space.is_continuous():
            adapter_type = "discrete_to_continuous"
        elif source_space.is_continuous() and target_space.is_discrete():
            adapter_type = "continuous_to_discrete"
        elif target_space.is_hybrid():
            adapter_type = "hybrid"
        elif source_space.space_type == target_space.space_type:
            # Same type, use parameterized for flexible conversion
            adapter_type = "parameterized"
        else:
            adapter_type = "parameterized"

    if adapter_type == "discrete_to_continuous":
        strategy = kwargs.get("strategy", "uniform_grid")
        return DiscreteToContiunousAdapter(source_space, target_space, strategy)

    elif adapter_type == "continuous_to_discrete":
        strategy = kwargs.get("strategy", "quantization")
        return ContinuousToDiscreteAdapter(source_space, target_space, strategy)

    elif adapter_type == "hybrid":
        return HybridActionAdapter(source_space, target_space)

    elif adapter_type == "parameterized":
        hidden_size = kwargs.get("hidden_size", 64)
        return ParameterizedActionAdapter(source_space, target_space, hidden_size)

    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


# Utility function for testing adapter quality
def test_adapter_quality(adapter: ActionAdapter, num_samples: int = 100) -> Dict[str, float]:
    """
    Test the quality of an action adapter by measuring round-trip error.

    Args:
        adapter: The adapter to test
        num_samples: Number of test samples to use

    Returns:
        Dictionary with quality metrics
    """
    source_actions = []
    round_trip_errors = []

    for _ in range(num_samples):
        # Sample action from source space
        original_action = adapter.source_space.sample()
        source_actions.append(original_action)

        try:
            # Forward and reverse adaptation
            adapted_action = adapter.adapt_action(original_action)
            recovered_action = adapter.reverse_adapt_action(adapted_action)

            # Calculate error
            if isinstance(original_action, (int, np.integer)):
                error = float(abs(original_action - recovered_action))
            else:
                error = float(np.linalg.norm(original_action - recovered_action))

            round_trip_errors.append(error)

        except Exception as e:
            # If adaptation fails, record high error
            round_trip_errors.append(float('inf'))

    round_trip_errors = np.array(round_trip_errors)
    valid_errors = round_trip_errors[np.isfinite(round_trip_errors)]

    return {
        'mean_round_trip_error': float(np.mean(valid_errors)) if len(valid_errors) > 0 else float('inf'),
        'max_round_trip_error': float(np.max(valid_errors)) if len(valid_errors) > 0 else float('inf'),
        'success_rate': float(len(valid_errors) / len(round_trip_errors)),
        'estimated_conversion_loss': adapter.estimate_conversion_loss(),
        'num_samples': num_samples
    }