# Agent Byte v3.0 - Complete Implementation Plan

## Project Overview

**Goal**: Refactor Agent Byte into a deployable, environment-agnostic Python package that can learn from any environment and transfer knowledge between them.

**Key Changes from the Current Version**:
- Remove ALL hard-coded environment references
- Create clean separation between core agent and environments
- Implement storage abstraction for JSON/Database/Vector DB
- Build proper interface system for environment compatibility
- Maintain 256-dimension state normalization
- Preserve neural-symbolic dual brain architecture

## Architecture Overview

```
agent_byte/
├── core/
│   ├── __init__.py
│   ├── agent.py              # Main AgentByte class
│   ├── interfaces.py         # Abstract interfaces
│   ├── dual_brain.py         # Neural-symbolic integration
│   ├── neural_brain.py       # Neural learning component
│   ├── symbolic_brain.py     # Symbolic reasoning component
|   |-- network.py            # Maintains 256-dimensional input
│   └── config.py             # Configuration classes
│
├── storage/
│   ├── __init__.py
│   ├── base.py               # Abstract storage interface
│   ├── json_numpy_storage.py # JSON + Numpy implementation
│   └── migrations.py         # Storage migration utilities
│
├── knowledge/
├── __init__.py
├── knowledge_system.py   # Main knowledge management
├── skill_discovery.py    # Dynamic skill discovery
├── pattern_interpreter.py # Neural pattern interpretation (replaces pattern_matcher)
├── decision_maker.py     # Symbolic decision making (new)
└── transfer_mapper.py    # Cross-environment mapping
│
├── analysis/
│   ├── __init__.py
│   ├── environment_analyzer.py  # Auto-detection system
│   ├── state_interpreter.py     # State understanding
│   └── state_normalizer.py      # 256-dim normalization
│
├── utils/
│   ├── __init__.py
│   └── constants.py          # Global constants (NO env-specific!)
│
├── examples/
│   ├── gymnasium_example.py  # How to use with Gymnasium
│   ├── custom_env_example.py # How to use with custom environments
│   └── transfer_example.py   # Transfer learning example
│
├── tests/
│   ├── test_core.py
│   ├── test_storage.py
│   └── test_transfer.py
│
├── setup.py
├── requirements.txt
└── README.md
```

## Core Interfaces

### 1. Environment Interface
```python
# core/interfaces.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np

class Environment(ABC):
    """Base environment interface that all environments must implement"""
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return (next_state, reward, done, info)"""
        pass
    
    @abstractmethod
    def get_state_size(self) -> int:
        """Return the native state size"""
        pass
    
    @abstractmethod
    def get_action_size(self) -> int:
        """Return the number of possible actions"""
        pass
    
    @abstractmethod
    def get_id(self) -> str:
        """Return unique identifier for this environment"""
        pass
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Optional: Return metadata about the environment"""
        return None
```

### 2. Storage Interface
```python
# storage/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class Storage(ABC):
    """Abstract storage interface for all storage backends"""
    
    @abstractmethod
    def save_brain_state(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save neural brain state"""
        pass
    
    @abstractmethod
    def load_brain_state(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load neural brain state"""
        pass
    
    @abstractmethod
    def save_knowledge(self, agent_id: str, env_id: str, data: Dict[str, Any]) -> bool:
        """Save symbolic knowledge"""
        pass
    
    @abstractmethod
    def load_knowledge(self, agent_id: str, env_id: str) -> Optional[Dict[str, Any]]:
        """Load symbolic knowledge"""
        pass
    
    @abstractmethod
    def save_experience_vector(self, agent_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save experience vector for similarity search"""
        pass
    
    @abstractmethod
    def search_similar_experiences(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar experiences (enables vector DB upgrade)"""
        pass
    
    @abstractmethod
    def list_environments(self, agent_id: str) -> List[str]:
        """List all environments this agent has learned from"""
        pass
```

## Implementation Phases

### Phase 1: Core Foundation (Week 1)
1. **Create project structure**
   - Set up all directories and __init__.py files
   - Create setup.py and requirements.txt

2. **Implement core interfaces**
   - Environment interface
   - Storage interface
   - Configuration classes

3. **Build storage layer**
   - JsonNumpyStorage implementation
   - Basic save/load functionality
   - Simple vector similarity search

### Phase 2: Environment Agnostic Agent (Week 2)
1. **Refactor Agent class**
   - Remove ALL hard-coded environment references
   - Implement environment-agnostic training loop
   - Add proper configuration system

2. **State normalization system**
   - Port existing 256-dim normalization
   - Remove hard-coded state interpretations
   - Make it work with any state size

3. **Environment analyzer**
   - Auto-detect state structure
   - Learn action effects through exploration
   - Generate environment metadata

### Phase 3: Dual Brain Migration (Week 3)
1. **Neural Brain refactoring**
   - Remove environment-specific code
   - Generalize learning algorithms
   - Maintain pattern tracking

2. **Symbolic Brain refactoring**
   - Remove hard-coded skill mappings
   - Implement dynamic skill discovery
   - Generalize decision making

3. **Neural-Symbolic Integration**
   - Port pattern interpreter
   - Make knowledge mapper environment-agnostic
   - Preserve dual brain communication

### Phase 4: Knowledge & Transfer System (Week 4)
1. **Knowledge System**
   - Dynamic skill discovery
   - Environment understanding storage
   - Cross-environment knowledge mapping

2. **Transfer Learning**
   - Similarity-based experience search
   - Skill transfer mechanisms
   - Adaptation algorithms

3. **Testing & Examples**
   - Create test environments
   - Write comprehensive examples
   - Document API usage

## Key Refactoring Tasks

### Remove Hard-coded Elements:
1. **TransferableSkillMapper.skill_mappings**
   - Currently has hard-coded env names (pong, chess, trading)
   - Replace with dynamic skill discovery

2. **Environment display names**
   - Remove `_get_environment_display_name()`
   - Use environment metadata instead

3. **Action assumptions**
   - Remove assumptions about 3 actions (up/stay/down)
   - Learn action effects dynamically

4. **State interpretations**
   - Remove hard-coded indices (ball_x, ball_y, etc.)
   - Use environment analyzer to detect

### API Design Decisions:
- **Storage**: Optional parameter, defaults to JsonNumpyStorage
- **Config**: Optional parameter, defaults to standard config
- **Metadata**: Environments can provide optional metadata
- **Transfer**: Explicit transfer_to() method for clarity

## Usage Examples

### Basic Training:

```python
from ZOldAgentFiles.agent_byte import AgentByte, AgentConfig, JsonNumpyStorage
from ZOldAgentFiles.agent_byte import GymnasiumAdapter
import gymnasium as gym

# Create agent with custom config
config = AgentConfig(
    learning_rate=0.001,
    exploration_rate=0.8,
    exploration_decay=0.995
)

agent = AgentByte(
    agent_id="my_agent",
    storage=JsonNumpyStorage("./agent_data"),
    config=config
)

# Train on any Gymnasium environment
env = gym.make("CartPole-v1")
adapted_env = GymnasiumAdapter(env)

# Provide optional metadata for faster learning
metadata = {
    'name': 'CartPole',
    'purpose': 'Balance a pole on a moving cart',
    'rules': [
        'Keep pole upright',
        'Stay within track boundaries'
    ],
    'objectives': {
        'primary': 'Maximize time before falling',
        'success_metric': 'steps survived'
    }
}

agent.train(adapted_env, episodes=1000, env_metadata=metadata)
```

### Transfer Learning:
```python
# After training on CartPole, transfer to MountainCar
env2 = gym.make("MountainCar-v0")
adapted_env2 = GymnasiumAdapter(env2)

# Agent automatically searches for similar experiences
agent.transfer_to(adapted_env2)
agent.train(adapted_env2, episodes=500)  # Faster learning!
```

## Testing Strategy

1. **Mock Environments**
   - Create simple test environments
   - Test various state/action sizes
   - Verify no hard-coding remains

2. **Storage Tests**
   - Test save/load functionality
   - Verify vector similarity search
   - Test storage migration

3. **Transfer Tests**
   - Train on one environment
   - Transfer to different environment
   - Verify skill transfer works

## Migration from Current Code

### What to Keep:
- Neural-symbolic dual brain concept
- 256-dimension normalization logic
- Pattern tracking and interpretation
- Knowledge persistence structure

### What to Change:
- Remove ALL environment-specific code
- Generalize skill definitions
- Make state interpretation dynamic
- Abstract storage layer

### What to Add:
- Proper interfaces
- Environment analyzer
- Storage abstraction
- Dynamic skill discovery
- Comprehensive examples

## Success Criteria

1. **Zero hard-coding**: No environment names or specifics in core code
2. **Universal compatibility**: Works with any environment implementing the interface
3. **Effective transfer**: Demonstrates learning transfer between diverse environments
4. **Clean API**: Simple, intuitive usage for end users
5. **Extensible storage**: Easy to upgrade to vector databases later

## Next Steps

1. Create new repository/directory for clean implementation
2. Set up project structure
3. Start with Phase 1: Core interfaces and storage
4. Reference old code for algorithms but not structure
5. Test each component thoroughly before moving to next phase

## Important Notes

- **No hard-coded training environments** in the core package
- **Adapters are separate** from the core agent (not in core/)
- **Storage is hot-swappable** through the interface
- **All environment knowledge** is learned or provided, never assumed
- **256-dim standard** is preserved but adaptable to any input size