# Agent Byte v3.0

A modular, transferable AI agent with neural-symbolic dual brain architecture for reinforcement learning.

## Overview

Agent Byte is an innovative AI agent framework that combines neural networks with symbolic reasoning to create agents that can:
- Learn from any environment without environment-specific code
- Transfer knowledge between different tasks and domains
- Provide interpretable decision-making through symbolic reasoning
- Scale from simple to complex environments

## Key Features

### 🧠 Dual Brain Architecture
- **Neural Brain**: Handles pattern recognition, value estimation, and low-level decision making
- **Symbolic Brain**: Provides high-level reasoning, skill discovery, and interpretable knowledge representation
- **Integrated Decision Making**: Seamlessly combines both approaches for optimal performance

### 🔄 Transfer Learning
- Transfer knowledge between completely different environments
- Discover and reuse skills across tasks
- No retraining from scratch - agents build on prior experience

### 🔌 Environment Agnostic
- Works with any environment through a simple interface
- No hard-coded environment specifics
- Easy adapter pattern for integrating new environments

### 💾 Flexible Storage System
- Pluggable storage backends (JSON/NumPy, databases, vector stores)
- Efficient experience storage and retrieval
- Built for scalability

## Installation

### From PyPI (when published)
```bash
pip install agent-byte
```

### From Source
```bash
git clone https://github.com/Tuf12/agent_byte.git
cd agent_byte
pip install -e .
```

### Dependencies
- Python >= 3.8
- numpy >= 1.19.0
- gymnasium >= 0.28.0
- typing-extensions >= 4.0.0

## Quick Start

```python
from agent_byte import AgentByte
from agent_byte.storage import JsonNumpyStorage
import gymnasium as gym

# Note: The GymnasiumAdapter is provided in examples/gymnasium_example.py
# You can copy it to your project or create your own adapter
from gymnasium_example import GymnasiumAdapter

# Create agent with JSON storage
agent = AgentByte(
    agent_id="my_agent",
    storage=JsonNumpyStorage("./agent_data")
)

# Train on CartPole environment
env = gym.make("CartPole-v1")
adapted_env = GymnasiumAdapter(env)
agent.train(adapted_env, episodes=1000)

# Transfer knowledge to a different environment
env2 = gym.make("MountainCar-v0")
adapted_env2 = GymnasiumAdapter(env2)
agent.transfer_to(adapted_env2)
```

## Architecture

```
AgentByte
├── Core Components
│   ├── Agent: Main orchestrator
│   ├── DualBrain: Integration layer
│   ├── NeuralBrain: Deep learning component
│   └── SymbolicBrain: Reasoning component
├── Analysis Layer
│   ├── EnvironmentAnalyzer: Understanding environments
│   ├── StateInterpreter: Making sense of observations
│   └── StateNormalizer: Standardizing inputs (256-dim vectors)
├── Knowledge System
│   ├── DecisionMaker: Action selection
│   ├── SkillDiscovery: Learning reusable skills
│   └── TransferMapper: Cross-domain knowledge transfer
└── Storage Backend
    └── Pluggable storage (JSON, DB, Vector stores)
```

## Creating Environment Adapters

To use Agent Byte with any environment, create an adapter implementing the `Environment` interface:

```python
from agent_byte.core.interfaces import Environment
import numpy as np

class MyEnvironmentAdapter(Environment):
    def __init__(self, your_env):
        self.env = your_env
    
    def reset(self) -> np.ndarray:
        # Reset environment and return initial state
        return self.env.reset()
    
    def step(self, action: int) -> tuple:
        # Execute action and return (state, reward, done, info)
        return self.env.step(action)
    
    def get_state_size(self) -> int:
        # Return size of state vector
        return self.env.observation_space.shape[0]
    
    def get_action_size(self) -> int:
        # Return number of possible actions
        return self.env.action_space.n
    
    def get_id(self) -> str:
        # Return unique environment identifier
        return "my_environment"
```

See `examples/gymnasium_example.py` for a complete implementation.

## Advanced Usage

### Custom Storage Backend
```python
from agent_byte.storage.base import Storage

class MyVectorDBStorage(Storage):
    def __init__(self, connection_string):
        # Initialize your vector database
        pass
    
    # Implement required storage methods
```

### Configuring the Agent
```python
from agent_byte.core.config import AgentConfig

config = AgentConfig(
    learning_rate=0.001,
    exploration_rate=0.1,
    batch_size=64,
    memory_size=100000,
    # ... other parameters
)

agent = AgentByte(
    agent_id="advanced_agent",
    storage=storage,
    config=config
)
```

## Project Structure
```
agent_byte/
├── core/           # Core agent components
├── analysis/       # Environment analysis tools
├── knowledge/      # Knowledge management system
├── storage/        # Storage backends
├── utils/          # Utilities and constants
└── examples/       # Example implementations
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Roadmap

- [ ] Additional storage backends (PostgreSQL, Pinecone, Weaviate)
- [ ] Multi-agent coordination
- [ ] Advanced transfer learning strategies
- [ ] Web UI for agent monitoring
- [ ] Pre-trained agent zoo

## License

MIT License
