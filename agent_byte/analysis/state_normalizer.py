"""
Universal state normalization system.

This module handles conversion of any environment state to the standardized
256-dimension format, enabling transfer learning across diverse environments.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging


class StateNormalizer:
    """
    Handles conversion of any environment state to standardized 256-dimension format.

    This is the core system that enables transfer learning by ensuring all environments
    speak the same "language" to the neural network.
    """

    def __init__(self, target_dim: int = 256):
        """
        Initialize the state normalizer.

        Args:
            target_dim: Target dimension for normalized states (default: 256)
        """
        self.target_dim = target_dim
        self.logger = logging.getLogger(self.__class__.__name__)

        # State statistics for adaptive normalization
        self.state_stats = {}

    def normalize(self, raw_state: np.ndarray, env_id: str,
                  metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Normalize any environment state to standard dimensions.

        Args:
            raw_state: Environment's native state representation
            env_id: Environment identifier for statistics tracking
            metadata: Optional metadata about state structure

        Returns:
            Normalized 256-dimensional state vector
        """
        # Ensure we have a numpy array
        if not isinstance(raw_state, np.ndarray):
            raw_state = np.array(raw_state, dtype=np.float32)

        # Flatten if multi-dimensional
        if raw_state.ndim > 1:
            raw_state = raw_state.flatten()

        # Initialize normalized state
        normalized = np.zeros(self.target_dim, dtype=np.float32)

        # Update statistics
        self._update_statistics(env_id, raw_state)

        # Apply normalization strategy based on state size
        if len(raw_state) >= self.target_dim:
            # State larger than target - use dimensionality reduction
            normalized = self._reduce_dimensions(raw_state, metadata)
        else:
            # State smaller than target - use intelligent padding
            normalized = self._expand_dimensions(raw_state, metadata)

        # Ensure all values are in reasonable range
        normalized = np.clip(normalized, -10, 10)

        return normalized

    def _reduce_dimensions(self, state: np.ndarray,
                           metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Reduce high-dimensional state to target dimensions.

        Uses intelligent sampling based on metadata if available.
        """
        if metadata and 'important_indices' in metadata:
            # Use provided important indices
            important = metadata['important_indices']
            result = np.zeros(self.target_dim)

            # Fill with important features first
            for i, idx in enumerate(important[:self.target_dim]):
                if idx < len(state):
                    result[i] = state[idx]

            return result
        else:
            # Default: even sampling across state
            indices = np.linspace(0, len(state) - 1, self.target_dim, dtype=int)
            return state[indices]

    def _expand_dimensions(self, state: np.ndarray,
                           metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Expand low-dimensional state to target dimensions.

        Uses intelligent padding with positional encoding.
        """
        result = np.zeros(self.target_dim)
        state_len = len(state)

        # Copy original state
        result[:state_len] = state

        # Apply positional encoding to unused dimensions
        for i in range(state_len, self.target_dim):
            # Sinusoidal positional encoding
            pos_encoding = np.sin(i * 0.01) * 0.1 + np.cos(i * 0.005) * 0.05

            # Add some information about original state size
            size_encoding = np.sin(state_len * 0.1) * 0.05

            result[i] = pos_encoding + size_encoding

        # If metadata provides state structure, use it
        if metadata:
            self._apply_semantic_padding(result, state, metadata)

        return result

    def _apply_semantic_padding(self, result: np.ndarray, original_state: np.ndarray,
                                metadata: Dict[str, Any]) -> None:
        """
        Apply semantic padding based on state metadata.

        This helps preserve meaning across different state sizes.
        """
        state_len = len(original_state)

        # Reserved regions for different types of information
        regions = {
            'position': (0, 32),  # Position information
            'velocity': (32, 64),  # Velocity/movement
            'features': (64, 128),  # General features
            'context': (128, 192),  # Environmental context
            'history': (192, 224),  # Historical information
            'padding': (224, 256)  # Padding region
        }

        # Map state types to regions if provided
        if 'state_types' in metadata:
            for i, state_type in enumerate(metadata['state_types'][:state_len]):
                if state_type in regions:
                    start, end = regions[state_type]
                    # Place this state element in its semantic region
                    if start + i < end:
                        result[start + i] = original_state[i]

    def _update_statistics(self, env_id: str, state: np.ndarray) -> None:
        """
        Update running statistics for adaptive normalization.
        """
        if env_id not in self.state_stats:
            self.state_stats[env_id] = {
                'count': 0,
                'mean': np.zeros_like(state),
                'var': np.zeros_like(state),
                'min': np.full_like(state, np.inf),
                'max': np.full_like(state, -np.inf)
            }

        stats = self.state_stats[env_id]
        stats['count'] += 1

        # Update min/max
        stats['min'] = np.minimum(stats['min'], state)
        stats['max'] = np.maximum(stats['max'], state)

        # Update running mean and variance (Welford's algorithm)
        delta = state - stats['mean']
        stats['mean'] += delta / stats['count']
        delta2 = state - stats['mean']
        stats['var'] += delta * delta2

    def get_denormalizer(self, env_id: str) -> 'StateDenormalizer':
        """
        Get a denormalizer for converting back to original state space.

        Args:
            env_id: Environment identifier

        Returns:
            StateDenormalizer instance
        """
        return StateDenormalizer(self, env_id)

    def get_statistics(self, env_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get normalization statistics for an environment.

        Args:
            env_id: Environment identifier

        Returns:
            Dictionary of statistics or None if not available
        """
        if env_id not in self.state_stats:
            return None

        stats = self.state_stats[env_id]
        if stats['count'] > 1:
            variance = stats['var'] / (stats['count'] - 1)
            std = np.sqrt(variance)
        else:
            std = np.zeros_like(stats['mean'])

        return {
            'mean': stats['mean'].copy(),
            'std': std,
            'min': stats['min'].copy(),
            'max': stats['max'].copy(),
            'count': stats['count']
        }


class StateDenormalizer:
    """
    Converts normalized states back to original environment space.

    This is useful for interpretability and debugging.
    """

    def __init__(self, normalizer: StateNormalizer, env_id: str):
        """
        Initialize denormalizer.

        Args:
            normalizer: Parent normalizer instance
            env_id: Environment identifier
        """
        self.normalizer = normalizer
        self.env_id = env_id
        self.stats = normalizer.get_statistics(env_id)

    def denormalize(self, normalized_state: np.ndarray,
                    original_size: int) -> np.ndarray:
        """
        Convert normalized state back to original dimensions.

        Args:
            normalized_state: 256-dimensional normalized state
            original_size: Original state size

        Returns:
            State in original dimensions
        """
        if original_size >= self.normalizer.target_dim:
            # State was reduced - we can't perfectly reconstruct
            # Return the most important features
            return normalized_state[:original_size]
        else:
            # State was padded - extract original portion
            return normalized_state[:original_size]


class StandardizedStateDimensions:
    """
    Defines the semantic layout of the 256-dimensional state space.

    This ensures similar concepts map to similar positions across environments,
    enabling effective transfer learning.
    """

    # Core positions (0-31): Object/Entity positions and orientations
    ENTITY_POSITIONS = (0, 32)

    # Movement vectors (32-63): Velocities, directions, momentum
    MOVEMENT_VECTORS = (32, 64)

    # Timing features (64-95): Temporal patterns, rhythms, sequences
    TIMING_FEATURES = (64, 96)

    # Strategic context (96-127): Goals, objectives, competitive state
    STRATEGIC_CONTEXT = (96, 128)

    # Performance metrics (128-159): Success rates, efficiency measures
    PERFORMANCE_METRICS = (128, 160)

    # Pattern recognition (160-191): Historical patterns, trend analysis
    PATTERN_RECOGNITION = (160, 192)

    # Meta-learning indicators (192-223): Learning progress, adaptation signals
    META_LEARNING = (192, 224)

    # Environment-specific (224-255): Domain-specific features
    ENVIRONMENT_SPECIFIC = (224, 256)

    @classmethod
    def get_region(cls, region_name: str) -> tuple:
        """Get the index range for a semantic region."""
        return getattr(cls, region_name.upper(), (0, 256))

    @classmethod
    def get_all_regions(cls) -> Dict[str, tuple]:
        """Get all semantic regions."""
        return {
            'entity_positions': cls.ENTITY_POSITIONS,
            'movement_vectors': cls.MOVEMENT_VECTORS,
            'timing_features': cls.TIMING_FEATURES,
            'strategic_context': cls.STRATEGIC_CONTEXT,
            'performance_metrics': cls.PERFORMANCE_METRICS,
            'pattern_recognition': cls.PATTERN_RECOGNITION,
            'meta_learning': cls.META_LEARNING,
            'environment_specific': cls.ENVIRONMENT_SPECIFIC
        }