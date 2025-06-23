# Agent_Byte
Adaptive AI agent using symbolic reasoning and reinforcement learning. Currently plays Pong.

# 🧠 Agent Byte v1.2 - Adaptive Learning + Knowledge System Enhanced

> **The Evolution of AI: Where Neural Networks Meet Symbolic Intelligence**

Agent Byte is an advanced AI agent that combines deep reinforcement learning with symbolic reasoning to create an intelligent system capable of understanding, adapting, and excelling across multiple environments. Currently mastering Classic Pong, Agent Byte represents the next generation of AI that doesn't just learn patterns—it understands context, applies strategy, and adapts its learning parameters to each unique environment.

![Agent Byte Demo](https://img.shields.io/badge/Demo-Live-brightgreen) ![Version](https://img.shields.io/badge/version-1.2.1-blue) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 What Makes Agent Byte Special

### 🧩 Dual Brain Architecture
- **Neural Brain**: Deep Q-Learning with Dueling Networks for pattern recognition and real-time decision making
- **Knowledge Brain**: Symbolic reasoning system that understands rules, objectives, and strategic concepts
- **Adaptive Learning**: Environment-specific parameter optimization (gamma, learning rate, exploration)

### 🎯 Intelligent Strategy Application
- Analyzes game situations in real-time
- Applies contextual strategies based on environmental understanding
- Balances symbolic reasoning with neural network decisions
- Tracks strategy effectiveness and optimizes approach

### 👤 User Demo Learning
- Records and learns from human gameplay demonstrations
- Integrates user strategies into AI decision-making
- Quality-based demo selection and weighted learning
- Real-time effectiveness tracking

## 🚀 Quick Start

### Prerequisites
```bash
pip install flask flask-socketio numpy
```

### Installation & Run
```bash
git clone https://github.com/yourusername/agent-byte
cd agent-byte
python app.py
```

Open your browser to `http://localhost:5001` and watch Agent Byte learn!

## 🎮 Current Environment: Enhanced Pong

Agent Byte currently excels at Classic Pong with sophisticated understanding:

### 🧠 Neural Capabilities
- **14-dimensional state space** with center-zero coordinate system
- **Dueling DQN architecture** (14→64→32→16→[V(1)+A(3)])
- **Double DQN** for reduced overestimation bias
- **Experience replay** with user demonstration integration

### 🧩 Symbolic Understanding
Agent Byte understands Pong not just as patterns, but as a strategic game:

```json
{
  "objective": "Score 21 points before opponent",
  "core_skills": [
    "Ball trajectory prediction",
    "Optimal paddle positioning", 
    "Timing and reaction speed",
    "Angle control for returns"
  ],
  "tactical_approaches": [
    "Defensive positioning - stay between ball and goal",
    "Aggressive returns - use paddle edges for angles",
    "Predictive movement - anticipate ball path"
  ]
}
```

### 🔧 Adaptive Learning Parameters
- **Gamma**: 0.90 (optimized for short-term competitive gameplay)
- **Learning Rate**: 0.001 (balanced for stable convergence)
- **Exploration**: 0.8 → 0.1 (aggressive early exploration, refined exploitation)

## 📊 Performance Features

### 🎳 Advanced Reward System
- **Hit-to-Score Bonus**: +1.0 extra reward for scoring immediately after hitting
- **Streak Bonuses**: +0.25 per consecutive successful hit
- **Symbolic Task Mapping**: hit→success, miss→failure, score→completion

### 📈 Real-Time Analytics
- Live strategy effectiveness tracking
- Decision type analysis (🧩 symbolic vs 🧠 neural)
- User demonstration learning metrics
- Adaptive learning parameter monitoring

### 🎯 Performance Metrics
- **Win Rate**: Tracks victories across matches
- **Hit Rate**: Percentage of successful ball interceptions  
- **Knowledge Effectiveness**: How well symbolic strategies perform vs neural decisions
- **Demo Effectiveness**: Impact of user demonstrations on AI performance

## 🏗️ Architecture Deep Dive

### Core Components

#### 1. **Dual Brain System** (`dual_brain_system.py`)
```python
class DualBrainAgent:
    def __init__(self):
        self.brain = AgentBrain()        # Neural learning engine
        self.knowledge = AgentKnowledge() # Symbolic understanding
```

#### 2. **Knowledge System** (`knowledge_system.py`)
```python
class SymbolicDecisionMaker:
    def make_informed_decision(self, state, q_values, context):
        # Analyzes situation and applies appropriate strategy
        return action, reasoning
```

#### 3. **Enhanced Environment** (`pong_environment.py`)
```python
def get_env_context(self):
    # Provides comprehensive environmental understanding
    return {
        "learning_parameters": {...},
        "strategic_concepts": {...},
        "failure_patterns": {...}
    }
```

### 🧩 Knowledge Application Flow

1. **Situation Analysis**: Neural state → symbolic game situation
2. **Strategy Selection**: Context + available strategies → best approach  
3. **Action Modification**: Neural Q-values + symbolic reasoning → final action
4. **Effectiveness Tracking**: Outcome → strategy performance updates

## 🎛️ Interactive Controls

### 🎮 Game Controls
- **Arrow Keys/WASD**: Control your paddle
- **Touch Controls**: Mobile-friendly up/down buttons
- **Speed Controls**: Adjust ball speed (1-20)

### 🔧 Learning Controls
- **Exploration Levels**: High (80%), Medium (50%), Low (20%)
- **Demo Learning**: Toggle user demonstration recording
- **Knowledge System**: Enable/disable symbolic reasoning
- **Adaptive Learning**: Toggle environment-specific parameter optimization

### 💾 Persistence Controls
- **Save Brain**: Preserve neural network weights and symbolic knowledge
- **Smart Reset**: Keep good experiences, increase exploration
- **Create Checkpoint**: Backup current state with timestamp

## 📁 Project Structure

```
agent-byte/
├── agent_byte.py           # Main AI agent with adaptive learning
├── dual_brain_system.py    # Neural + symbolic brain architecture  
├── knowledge_system.py     # Symbolic reasoning and strategy application
├── pong_environment.py     # Enhanced Pong with environmental context
├── app.py                  # Flask server and game coordination
├── templates/
│   └── pong.html          # Web interface with real-time controls
├── agent_brain.json        # Persistent neural learning state
├── agent_knowledge.json    # Persistent symbolic knowledge
└── README.md              # This file
```

## 🔬 Technical Highlights

### Neural Network Architecture
```
Input Layer (14 dims) → Feature Layers (64→32→16) → Dueling Streams
                                                   ├── Value Stream (→1)
                                                   └── Advantage Stream (→3)
```

### Adaptive Learning System
```python
# Environment provides recommended parameters
learning_params = env_context.get('learning_parameters', {})
agent.gamma = learning_params.get('recommended_gamma', 0.99)
agent.learning_rate = learning_params.get('recommended_learning_rate', 0.001)
```

### Knowledge Integration
```python
# Symbolic knowledge modifies neural decisions
if use_symbolic_knowledge:
    action, reasoning = knowledge_system.apply_strategy(state, q_values, context)
    return action, f"🧩 {reasoning}"
else:
    return np.argmax(q_values), "🧠 Neural network decision"
```

## 📊 Sample Performance Data

```json
{
  "games_played": 47,
  "win_rate": 68.1,
  "knowledge_effectiveness": 1.847,
  "symbolic_decisions": 342,
  "neural_decisions": 156,
  "adaptive_learning_enabled": true,
  "current_gamma": 0.90,
  "demo_effectiveness": 0.73
}
```

## 🚀 Future Roadmap

### 🎮 Multi-Environment Support
- **Flappy Bird**: Timing and obstacle avoidance strategies
- **Chess**: Long-term strategic planning
- **Reading Comprehension**: Language understanding and retention
- **Creative Writing**: Style learning and content generation

### 🧠 Enhanced AI Capabilities
- **Meta-Learning**: Learn how to learn across environments
- **Transfer Learning**: Apply Pong strategies to similar games
- **Causal Reasoning**: Understand cause-and-effect relationships
- **Self-Reflection**: Generate insights about own performance

### 🔧 Technical Improvements
- **Multi-Agent Training**: Learn from other AI agents
- **Hierarchical Strategies**: Nested decision-making layers
- **Attention Mechanisms**: Focus on relevant state components
- **Curriculum Learning**: Progressive difficulty training

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **New Environments**: Implement `get_env_context()` for other games
- **Strategy Development**: Add new symbolic reasoning patterns
- **UI Enhancements**: Improve real-time visualization
- **Performance Optimization**: Neural network architecture improvements

## 📜 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Inspired by DeepMind's hybrid neural-symbolic approaches
- Built on principles from Dueling DQN and Double DQN research
- Community feedback from AI/ML practitioners

---

**Agent Byte v1.2** - Where artificial intelligence meets artificial wisdom. Watch as neural networks learn not just what to do, but *why* to do it.

*"The future of AI isn't just about pattern recognition—it's about understanding."*
