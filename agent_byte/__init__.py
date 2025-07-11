# agent_byte/__init__.py
"""
Agent Byte v3.0 - Modular Transfer Learning AI Agent

A completely environment-agnostic AI agent that can learn from any environment
and transfer knowledge between them.
"""

__version__ = "3.0.0"

# Core imports
from .core.agent import AgentByte
from .core.config import AgentConfig, NetworkConfig, StorageConfig, EnvironmentMetadata
from .core.interfaces import Environment, Storage

# Storage imports
from .storage.base import StorageBase
from .storage.json_numpy_storage import JsonNumpyStorage

# Analysis imports
from .analysis.state_normalizer import StateNormalizer
from .analysis.environment_analyzer import EnvironmentAnalyzer

__all__ = [
    # Main class
    "AgentByte",
    
    # Configuration
    "AgentConfig",
    "NetworkConfig", 
    "StorageConfig",
    "EnvironmentMetadata",
    
    # Interfaces
    "Environment",
    "Storage",
    
    # Storage
    "StorageBase",
    "JsonNumpyStorage",
    
    # Analysis
    "StateNormalizer",
    "EnvironmentAnalyzer",
]