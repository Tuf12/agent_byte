"""
Configuration classes for Agent Byte v3.0

These classes define all configurable parameters for the agent,
ensuring no hard-coded values in the core implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AgentConfig:
    """Main configuration for Agent Byte."""

    # Neural network parameters
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor

    # Exploration parameters
    exploration_rate: float = 0.8
    exploration_decay: float = 0.995
    min_exploration: float = 0.1

    # Training parameters
    batch_size: int = 16
    target_update_frequency: int = 1000
    experience_buffer_size: int = 5000

    # State normalization
    state_dimensions: int = 256  # Universal state size
    state_clip_range: tuple = (-10, 10)

    # Neural-symbolic integration
    symbolic_decision_threshold: float = 0.3  # When to use symbolic decisions
    pattern_detection_interval: int = 10  # Steps between pattern analysis

    # Storage parameters
    save_interval: int = 100  # Steps between saves
    checkpoint_keep_count: int = 5  # Number of checkpoints to keep

    # Transfer learning
    similarity_threshold: float = 0.7  # For experience matching
    transfer_confidence_threshold: float = 0.6  # Minimum confidence for transfer

    # Performance tracking
    performance_window: int = 100  # Episodes to calculate metrics over

    # NEW: Transfer validation parameters
    transfer_validation_interval: int = 100  # Steps between validations
    validation_window_size: int = 50  # Episodes for performance comparison
    transfer_success_threshold: float = 0.1  # Minimum improvement needed
    adaptive_transfer_enabled: bool = True  # Enable strategy adaptation

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'exploration_rate': self.exploration_rate,
            'exploration_decay': self.exploration_decay,
            'min_exploration': self.min_exploration,
            'batch_size': self.batch_size,
            'target_update_frequency': self.target_update_frequency,
            'experience_buffer_size': self.experience_buffer_size,
            'state_dimensions': self.state_dimensions,
            'state_clip_range': self.state_clip_range,
            'symbolic_decision_threshold': self.symbolic_decision_threshold,
            'pattern_detection_interval': self.pattern_detection_interval,
            'save_interval': self.save_interval,
            'checkpoint_keep_count': self.checkpoint_keep_count,
            'similarity_threshold': self.similarity_threshold,
            'transfer_confidence_threshold': self.transfer_confidence_threshold,
            'performance_window': self.performance_window,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class NetworkConfig:
    """Configuration for the neural network architecture."""

    # Architecture
    input_size: int = 256  # Standardized input
    core_layer_sizes: list = field(default_factory=lambda: [512, 256, 128])
    adapter_size: int = 64

    # Layer parameters
    activation_function: str = 'leaky_relu'
    leaky_relu_alpha: float = 0.01
    weight_init_method: str = 'xavier'

    # Regularization
    dropout_rate: float = 0.0  # Can be added if needed
    l2_regularization: float = 0.0

    # Pattern tracking
    track_patterns: bool = True
    pattern_history_size: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'input_size': self.input_size,
            'core_layer_sizes': self.core_layer_sizes,
            'adapter_size': self.adapter_size,
            'activation_function': self.activation_function,
            'leaky_relu_alpha': self.leaky_relu_alpha,
            'weight_init_method': self.weight_init_method,
            'dropout_rate': self.dropout_rate,
            'l2_regularization': self.l2_regularization,
            'track_patterns': self.track_patterns,
            'pattern_history_size': self.pattern_history_size,
        }


@dataclass
class StorageConfig:
    """Configuration for storage system."""

    # Base paths
    base_path: str = "./agent_data"

    # File organization
    use_compression: bool = True
    file_format: str = "json"  # json, pickle, msgpack

    # Vector storage
    vector_index_method: str = "cosine"  # cosine, euclidean, dot_product
    vector_dimension: int = 256
    max_vectors_in_memory: int = 10000

    # Caching
    enable_cache: bool = True
    cache_size: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'base_path': self.base_path,
            'use_compression': self.use_compression,
            'file_format': self.file_format,
            'vector_index_method': self.vector_index_method,
            'vector_dimension': self.vector_dimension,
            'max_vectors_in_memory': self.max_vectors_in_memory,
            'enable_cache': self.enable_cache,
            'cache_size': self.cache_size,
        }


@dataclass
class EnvironmentMetadata:
    """
    Metadata structure for environments.

    This standardizes how environments describe themselves to the agent.
    """

    # Basic information
    name: str
    env_id: str
    env_type: Optional[str] = None  # game, control, robotics, etc.

    # Understanding
    purpose: Optional[str] = None
    instructions: Optional[list] = field(default_factory=list)
    rules: Optional[list] = field(default_factory=list)

    # Objectives
    objectives: Optional[Dict[str, Any]] = field(default_factory=dict)
    success_criteria: Optional[list] = field(default_factory=list)
    failure_criteria: Optional[list] = field(default_factory=list)

    # State information (optional hints)
    state_labels: Optional[list] = None  # Names for state dimensions
    state_ranges: Optional[list] = None  # Expected ranges
    state_types: Optional[list] = None  # position, velocity, angle, etc.

    # Action information (optional hints)
    action_labels: Optional[list] = None  # Names for actions
    action_effects: Optional[list] = None  # Descriptions of what actions do

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'env_id': self.env_id,
            'env_type': self.env_type,
            'purpose': self.purpose,
            'instructions': self.instructions,
            'rules': self.rules,
            'objectives': self.objectives,
            'success_criteria': self.success_criteria,
            'failure_criteria': self.failure_criteria,
            'state_labels': self.state_labels,
            'state_ranges': self.state_ranges,
            'state_types': self.state_types,
            'action_labels': self.action_labels,
            'action_effects': self.action_effects,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentMetadata':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})