"""
Global constants for Agent Byte.

This module contains all constants used across the system.
NO ENVIRONMENT-SPECIFIC VALUES ALLOWED!
"""

# Architecture constants
STANDARD_STATE_DIMENSION = 256  # Universal state size for all environments
NETWORK_ARCHITECTURE_VERSION = "3.0"

# Neural network architecture
CORE_LAYER_SIZES = [512, 256, 128]  # Transferable core layers
ADAPTER_LAYER_SIZE = 64  # Environment-specific adapter
ACTIVATION_FUNCTION = "leaky_relu"
LEAKY_RELU_ALPHA = 0.01

# Learning parameters (defaults)
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_GAMMA = 0.99  # Discount factor
DEFAULT_EXPLORATION_RATE = 0.8
DEFAULT_EXPLORATION_DECAY = 0.995
DEFAULT_MIN_EXPLORATION = 0.1

# Training parameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_TARGET_UPDATE_FREQUENCY = 1000
DEFAULT_EXPERIENCE_BUFFER_SIZE = 5000
DEFAULT_SAVE_INTERVAL = 100

# Neural-symbolic integration
DEFAULT_SYMBOLIC_DECISION_THRESHOLD = 0.3  # When to trust symbolic decisions
PATTERN_DETECTION_INTERVAL = 10  # Steps between pattern analysis
PATTERN_HISTORY_SIZE = 100  # Maximum patterns to track
DECISION_HISTORY_SIZE = 100  # Maximum decisions to track

# Knowledge system
MIN_SKILL_CONFIDENCE = 0.5  # Minimum confidence for skill to be active
SKILL_DISCOVERY_CONFIDENCE_THRESHOLD = 0.6
SKILL_DISCOVERY_EVIDENCE_THRESHOLD = 3
SKILL_SUCCESS_LEARNING_RATE = 0.1  # For updating skill success rates

# Transfer learning
TRANSFER_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for skill transfer
STRUCTURAL_SIMILARITY_THRESHOLD = 0.6
BEHAVIORAL_SIMILARITY_THRESHOLD = 0.5
SKILL_COMPATIBILITY_THRESHOLD = 0.7

# State interpretation
STATE_HISTORY_SIZE = 1000  # States to keep for analysis
CORRELATION_THRESHOLD = 0.7  # For identifying correlated dimensions
ANOMALY_THRESHOLD = 3.0  # Standard deviations for anomaly detection

# Storage
DEFAULT_STORAGE_PATH = "./agent_data"
CHECKPOINT_KEEP_COUNT = 5
VECTOR_SIMILARITY_METHOD = "cosine"
MAX_VECTORS_IN_MEMORY = 10000

# Performance tracking
PERFORMANCE_WINDOW = 100  # Episodes for calculating metrics
IMPROVEMENT_THRESHOLD = 0.1  # Minimum improvement to consider significant

# Pattern detection thresholds
CONVERGENCE_STABILITY_THRESHOLD = 0.7
OSCILLATION_BALANCE_THRESHOLD = 0.3
EXPLORATION_ENTROPY_THRESHOLD = 0.8
EXPLOITATION_DOMINANCE_THRESHOLD = 0.7

# Abstraction levels
ABSTRACTION_LEVELS = ['concrete', 'tactical', 'strategic', 'meta']

# Decision strategy names
DECISION_STRATEGIES = [
    'skill_based',
    'pattern_based',
    'exploration_based',
    'confidence_based',
    'meta_strategy'
]

# Skill types
SKILL_TYPES = [
    'action_optimization',
    'state_response',
    'reward_seeking',
    'pattern_execution',
    'adaptive_behavior',
    'exploratory_strategy'
]

# Learning phases
LEARNING_PHASES = [
    'initial',
    'early_learning',
    'rapid_learning',
    'refinement',
    'plateau',
    'unstable'
]

# Behavioral states
BEHAVIORAL_STATES = [
    'stable_policy',
    'converging',
    'exploring',
    'transitioning',
    'unknown'
]

# State dimension types
STATE_DIMENSION_TYPES = [
    'constant',
    'binary',
    'discrete',
    'normalized',
    'cyclic',
    'stable_continuous',
    'moderate_continuous',
    'volatile_continuous',
    'unknown'
]

# Transfer strategies
TRANSFER_STRATEGIES = [
    'direct_transfer',
    'scaled_transfer',
    'abstract_transfer',
    'compositional_transfer'
]

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Version information
AGENT_BYTE_VERSION = "3.0.0"
MIN_PYTHON_VERSION = "3.8"

# Limits and boundaries
MAX_EPISODE_STEPS = 10000  # Prevent infinite episodes
MAX_ENVIRONMENTS_PER_AGENT = 100  # Reasonable limit
MAX_SKILLS_PER_ENVIRONMENT = 1000  # Prevent memory issues

# File formats
BRAIN_STATE_FORMAT = "json"
KNOWLEDGE_FORMAT = "json"
VECTOR_STORAGE_FORMAT = "npy"

# Performance optimization
CACHE_ENABLED = True
CACHE_SIZE = 100
ANALYSIS_UPDATE_INTERVAL = 100  # Steps between full analysis updates

# Error handling
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Seconds between retries

# Validation
VALID_ACTIONS_RANGE = (2, 100)  # Min and max action space size
VALID_STATE_RANGE = (1, 10000)  # Min and max state dimensions (before normalization)