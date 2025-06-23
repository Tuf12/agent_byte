# agent_byte.py - Enhanced with Adaptive Learning Parameters + Knowledge System
import numpy as np
import json
import time
import random
import os
from collections import deque
import datetime

# Import the dual brain system and knowledge system
from dual_brain_system import DualBrainAgent, AgentBrain, AgentKnowledge
from knowledge_system import SymbolicDecisionMaker

class MatchLogger:
    # Keep your existing MatchLogger implementation
    def __init__(self, log_filename='agent_byte_match_logs.json'):
        self.log_file = log_filename
        self.current_match = None
        self.all_matches = []
        self.load_match_history()
    
    def load_match_history(self):
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.all_matches = data.get('matches', [])
                print(f"📚 Loaded {len(self.all_matches)} match records from {self.log_file}")
        except Exception as e:
            print(f"⚠️ Could not load match history: {e}")
            self.all_matches = []
    
    def start_match(self, match_id, game_type="unknown"):
        self.current_match = {
            'match_id': match_id,
            'game_type': game_type,
            'start_time': datetime.datetime.now().isoformat(),
            'end_time': None,
            'winner': None,
            'final_score': {'player': 0, 'agent_byte': 0},
            'agent_byte_stats': {
                'total_reward': 0,
                'actions_taken': 0,
                'match_reward': 0,
                'exploration_rate_start': 0,
                'exploration_rate_end': 0,
                'training_steps': 0,
                'target_updates': 0,
                'architecture': 'Agent Byte v1.2 - Adaptive Learning + Knowledge System Enhanced',
                'hit_to_score_bonuses': 0,
                'human_demos_used': 0,
                'user_demos_recorded': 0,
                'demo_learning_weight': 0.3,
                'symbolic_lessons_learned': 0,
                'strategies_discovered': 0,
                'symbolic_decisions_made': 0,
                'neural_decisions_made': 0,
                'knowledge_effectiveness': 0.0,
                # NEW: Adaptive learning parameters tracking
                'gamma_used': 0.99,
                'gamma_source': 'default',
                'learning_rate_used': 0.001,
                'learning_parameters_adapted': False
            },
            'interactions': [],
            'rewards_timeline': [],
            'user_demonstrations': [],
            'symbolic_insights': [],
            'strategic_decisions': [],
            'learning_adaptations': []  # NEW: Track learning parameter changes
        }
        print(f"🆕 Started logging {game_type} match {match_id}")
    
    def log_learning_adaptation(self, adaptation_info):
        """Log adaptive learning parameter changes"""
        if self.current_match:
            adaptation = {
                'timestamp': time.time(),
                'parameter': adaptation_info.get('parameter'),
                'old_value': adaptation_info.get('old_value'),
                'new_value': adaptation_info.get('new_value'),
                'source': adaptation_info.get('source'),
                'rationale': adaptation_info.get('rationale')
            }
            self.current_match['learning_adaptations'].append(adaptation)
    
    def log_symbolic_insight(self, insight_type, content):
        """Log symbolic learning insights"""
        if self.current_match:
            insight = {
                'timestamp': time.time(),
                'type': insight_type,
                'content': content
            }
            self.current_match['symbolic_insights'].append(insight)
    
    def log_strategic_decision(self, decision_info):
        """Log strategic decision made by knowledge system"""
        if self.current_match:
            decision = {
                'timestamp': time.time(),
                'action': decision_info.get('action'),
                'reasoning': decision_info.get('reasoning'),
                'confidence': decision_info.get('confidence'),
                'strategy_used': decision_info.get('strategy_used')
            }
            self.current_match['strategic_decisions'].append(decision)
    
    def update_match_stats(self, stats):
        if self.current_match:
            self.current_match['agent_byte_stats'].update(stats)
    
    def end_match(self, winner, final_scores, final_stats):
        if not self.current_match:
            print("⚠️ No current match to end")
            return
        try:
            self.current_match['end_time'] = datetime.datetime.now().isoformat()
            self.current_match['winner'] = winner
            self.current_match['final_score'] = final_scores or {'player': 0, 'agent_byte': 0}
            if isinstance(final_stats, dict):
                self.current_match['agent_byte_stats'].update(final_stats)
            start = datetime.datetime.fromisoformat(self.current_match['start_time'])
            end = datetime.datetime.fromisoformat(self.current_match['end_time'])
            self.current_match['duration_seconds'] = (end - start).total_seconds()
            self.all_matches.append(self.current_match.copy())
            self.save_match_history()
            print(f"📊 Match {self.current_match['match_id']} completed and logged")
            self.current_match = None
        except Exception as e:
            print(f"❌ Error ending match: {e}")
            self.current_match = None
    
    def save_match_history(self):
        try:
            data = {
                'total_matches': len(self.all_matches),
                'last_updated': datetime.datetime.now().isoformat(),
                'version': 'Agent Byte v1.2 - Adaptive Learning + Knowledge System Enhanced',
                'matches': self.all_matches
            }
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"❌ Could not save match history: {e}")
    
    def get_recent_performance(self, last_n_matches=10):
        if not self.all_matches:
            return None
        recent = self.all_matches[-last_n_matches:]
        total_matches = len(recent)
        wins = sum(1 for match in recent if match['winner'] == 'Agent Byte')
        total_reward = sum(match['agent_byte_stats'].get('match_reward', 0) for match in recent)
        total_hit_bonuses = sum(match['agent_byte_stats'].get('hit_to_score_bonuses', 0) for match in recent)
        total_human_demos = sum(match['agent_byte_stats'].get('human_demos_used', 0) for match in recent)
        total_user_demos = sum(match['agent_byte_stats'].get('user_demos_recorded', 0) for match in recent)
        total_lessons = sum(match['agent_byte_stats'].get('symbolic_lessons_learned', 0) for match in recent)
        total_symbolic_decisions = sum(match['agent_byte_stats'].get('symbolic_decisions_made', 0) for match in recent)
        
        # NEW: Track adaptive learning usage
        gamma_adaptations = sum(1 for match in recent if match['agent_byte_stats'].get('learning_parameters_adapted', False))
        
        return {
            'matches_analyzed': total_matches,
            'recent_win_rate': (wins / total_matches * 100) if total_matches > 0 else 0,
            'recent_hit_rate': sum(match['agent_byte_stats'].get('task_success_rate', 0) for match in recent) / total_matches if total_matches > 0 else 0,
            'avg_reward_per_match': total_reward / total_matches if total_matches > 0 else 0,
            'avg_hit_bonuses_per_match': total_hit_bonuses / total_matches if total_matches > 0 else 0,
            'avg_human_demos_per_match': total_human_demos / total_matches if total_matches > 0 else 0,
            'avg_user_demos_per_match': total_user_demos / total_matches if total_matches > 0 else 0,
            'avg_lessons_per_match': total_lessons / total_matches if total_matches > 0 else 0,
            'avg_symbolic_decisions_per_match': total_symbolic_decisions / total_matches if total_matches > 0 else 0,
            'gamma_adaptations': gamma_adaptations,
            'adaptive_learning_usage_rate': (gamma_adaptations / total_matches * 100) if total_matches > 0 else 0,
            'total_matches': len(self.all_matches),
            'total_wins': sum(1 for match in self.all_matches if match['winner'] == 'Agent Byte')
        }

class DuelingNetwork:
    # Keep your existing DuelingNetwork implementation (unchanged)
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.feature_layers = []
        layer_sizes = [input_size] + hidden_sizes[:-1]
        for i in range(len(layer_sizes) - 1):
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i]),
                'biases': np.zeros(layer_sizes[i+1])
            }
            self.feature_layers.append(layer)
        feature_size = hidden_sizes[-2] if len(hidden_sizes) > 1 else hidden_sizes[0]
        stream_size = hidden_sizes[-1]
        self.value_stream = {
            'weights1': np.random.randn(feature_size, stream_size) * np.sqrt(2.0 / feature_size),
            'biases1': np.zeros(stream_size),
            'weights2': np.random.randn(stream_size, 1) * np.sqrt(2.0 / stream_size),
            'biases2': np.zeros(1)
        }
        self.advantage_stream = {
            'weights1': np.random.randn(feature_size, stream_size) * np.sqrt(2.0 / feature_size),
            'biases1': np.zeros(stream_size),
            'weights2': np.random.randn(stream_size, output_size) * np.sqrt(2.0 / stream_size),
            'biases2': np.zeros(output_size)
        }
        print(f"🧠 Dueling Network: {input_size}→{hidden_sizes}→V(1)+A({output_size})")
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def forward(self, state):
        if len(state.shape) > 1:
            state = state.flatten()
        x = state.copy()
        self.activations = [x]
        for layer in self.feature_layers:
            z = np.dot(x, layer['weights']) + layer['biases']
            x = self.leaky_relu(z)
            self.activations.append(x)
        features = x
        v1 = np.dot(features, self.value_stream['weights1']) + self.value_stream['biases1']
        v1_activated = self.leaky_relu(v1)
        value = np.dot(v1_activated, self.value_stream['weights2']) + self.value_stream['biases2']
        a1 = np.dot(features, self.advantage_stream['weights1']) + self.advantage_stream['biases1']
        a1_activated = self.leaky_relu(a1)
        advantages = np.dot(a1_activated, self.advantage_stream['weights2']) + self.advantage_stream['biases2']
        q_values = value + (advantages - np.mean(advantages))
        self.features = features
        self.v1 = v1
        self.v1_activated = v1_activated
        self.value = value
        self.a1 = a1
        self.a1_activated = a1_activated
        self.advantages = advantages
        return q_values
    
    def copy_weights_from(self, other_network):
        for i, layer in enumerate(other_network.feature_layers):
            self.feature_layers[i]['weights'] = layer['weights'].copy()
            self.feature_layers[i]['biases'] = layer['biases'].copy()
        for key in self.value_stream:
            self.value_stream[key] = other_network.value_stream[key].copy()
        for key in self.advantage_stream:
            self.advantage_stream[key] = other_network.advantage_stream[key].copy()
    
    def soft_update_from(self, other_network, tau=0.001):
        for i, layer in enumerate(other_network.feature_layers):
            self.feature_layers[i]['weights'] = (1 - tau) * self.feature_layers[i]['weights'] + tau * layer['weights']
            self.feature_layers[i]['biases'] = (1 - tau) * self.feature_layers[i]['biases'] + tau * layer['biases']
        for key in self.value_stream:
            self.value_stream[key] = (1 - tau) * self.value_stream[key] + tau * other_network.value_stream[key]
        for key in self.advantage_stream:
            self.advantage_stream[key] = (1 - tau) * self.advantage_stream[key] + tau * other_network.advantage_stream[key]
    
    def update_weights(self, state, target_q_values, action_taken):
        current_q_values = self.forward(state)
        q_error = target_q_values - current_q_values
        q_error = np.clip(q_error, -1.0, 1.0)
        action_error = np.zeros_like(current_q_values)
        action_error[action_taken] = q_error[action_taken]
        self.advantage_stream['weights2'] += self.learning_rate * np.outer(self.a1_activated, action_error)
        self.advantage_stream['biases2'] += self.learning_rate * action_error
        value_error = np.sum(q_error) / len(q_error)
        self.value_stream['weights2'] += self.learning_rate * np.outer(self.v1_activated, [value_error])
        self.value_stream['biases2'] += self.learning_rate * value_error
        return np.mean(q_error ** 2)

class AgentByte:
    """Enhanced Agent Byte with Adaptive Learning Parameters + Dual Brain Architecture + Knowledge System"""
    
    def __init__(self, state_size=14, action_size=3, logger=None, app_name="unknown_game"):
        print("🚀 Agent Byte v1.2 - Adaptive Learning + Knowledge System Enhanced Initializing...")
        
        # Initialize dual brain system
        self.dual_brain = DualBrainAgent()
        self.app_name = app_name
        self.app_context = None
        
        # NEW: Initialize symbolic decision maker
        self.symbolic_decision_maker = SymbolicDecisionMaker()
        
        # Neural network components
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = [64, 32, 16]
        self.main_network = DuelingNetwork(self.state_size, self.hidden_sizes, self.action_size, learning_rate=0.001)
        self.target_network = DuelingNetwork(self.state_size, self.hidden_sizes, self.action_size, learning_rate=0.001)
        self.target_network.copy_weights_from(self.main_network)
        
        # Learning parameters (will be adapted per environment)
        self.target_update_frequency = 1000
        self.soft_update_tau = 0.005
        self.use_soft_updates = True
        self.learning_rate = self.dual_brain.brain.learning_rate
        self.exploration_rate = self.dual_brain.brain.epsilon
        self.exploration_decay = 0.995
        self.min_exploration = 0.1
        self.gamma = self.dual_brain.brain.gamma  # Default gamma from brain, will be overridden per environment
        
        # NEW: Adaptive learning tracking
        self.default_gamma = self.gamma
        self.environment_gamma = None
        self.gamma_source = "default"
        self.learning_parameters_adapted = False
        self.environment_learning_metadata = {}
        
        # Experience and demo buffers
        self.experience_buffer = deque(maxlen=5000)
        self.user_demo_buffer = deque(maxlen=1000)
        self.replay_batch_size = 16
        self.replay_frequency = 4
        self.min_buffer_size = 500
        
        # Demo learning parameters
        self.demo_learning_weight = 0.3
        self.demo_replay_ratio = 0.25
        
        # Performance tracking
        self.games_played = 0
        self.wins = 0
        self.total_reward = 0
        self.match_reward = 0
        self.actions_taken = 0
        self.training_steps = self.dual_brain.brain.training_steps
        self.total_loss = self.dual_brain.brain.total_loss
        self.target_updates = self.dual_brain.brain.target_updates
        self.strategic_moves = 0
        self.hit_to_score_bonuses = 0
        self.total_bonus_reward = 0
        self.human_demos_used = 0
        self.user_demos_recorded = 0
        self.user_demos_processed = 0
        self.double_dqn_improvements = self.dual_brain.brain.double_dqn_improvements
        
        # Knowledge system tracking
        self.symbolic_decisions_made = 0
        self.neural_decisions_made = 0
        self.knowledge_effectiveness = 0.0
        
        # Symbolic learning tracking
        self.lessons_learned_this_match = 0
        self.strategies_discovered_this_match = 0
        
        # State tracking
        self.last_state = None
        self.last_action = None
        
        # Logger
        self.logger = logger or MatchLogger()
        
        print("✅ Agent Byte v1.2 Adaptive Learning + Knowledge System Enhanced Created!")
        print(f"   🧠 Core Brain: {self.training_steps} training steps")
        print(f"   🧩 Knowledge: Symbolic learning + intelligent application")
        print(f"   🎯 Architecture: Neural + Symbolic Decision Making")
        print(f"   👤 Demo Learning: Enhanced with symbolic understanding")
        print(f"   ⚙️ Adaptive Learning: Environment-specific parameter optimization")
        print(f"   🔧 Default Gamma: {self.gamma} (will adapt per environment)")

    def start_new_match(self, game_type="game", env_context=None):
        """Start new match with adaptive learning parameter loading"""
        # Reset match-specific stats
        self.match_reward = 0
        self.actions_taken = 0
        self.hit_to_score_bonuses = 0
        self.human_demos_used = 0
        self.user_demos_recorded = 0
        self.user_demos_processed = 0
        self.lessons_learned_this_match = 0
        self.strategies_discovered_this_match = 0
        self.symbolic_decisions_made = 0
        self.neural_decisions_made = 0
        
        # Reset decision maker history for this match
        self.symbolic_decision_maker.decision_history = []
        
        # Load symbolic context for this game with environmental context
        self.app_context = self.dual_brain.start_session(game_type, env_context=env_context)
        
        # NEW: Adapt learning parameters based on environment context
        self._adapt_learning_parameters(env_context)
        
        # Start logging
        match_id = f"agent_byte_{game_type}_{int(time.time())}"
        self.logger.start_match(match_id, game_type)
        if self.logger.current_match:
            self.logger.current_match['agent_byte_stats']['exploration_rate_start'] = self.exploration_rate
            self.logger.current_match['agent_byte_stats']['gamma_used'] = self.gamma
            self.logger.current_match['agent_byte_stats']['gamma_source'] = self.gamma_source
            self.logger.current_match['agent_byte_stats']['learning_parameters_adapted'] = self.learning_parameters_adapted
            
            # Log environmental context integration
            if env_context:
                self.logger.log_symbolic_insight("env_context_loaded", f"Environment context integrated for {game_type}")
                
                # Log learning parameter adaptations
                if self.learning_parameters_adapted:
                    self.logger.log_learning_adaptation({
                        'parameter': 'gamma',
                        'old_value': self.default_gamma,
                        'new_value': self.gamma,
                        'source': f'environment:{game_type}',
                        'rationale': self.environment_learning_metadata.get('gamma_rationale', 'Environment-specific optimization')
                    })
        
        print(f"🆕 New {game_type} match started with adaptive learning + knowledge system enabled")
        if self.app_context:
            strategies = len(self.app_context.get('strategies', []))
            lessons = len(self.app_context.get('lessons', []))
            print(f"   📚 Available knowledge: {strategies} strategies, {lessons} lessons")
            print(f"   🧩 Knowledge system: Active and ready for intelligent decision making")
            print(f"   ⚙️ Learning parameters: Gamma={self.gamma:.3f} ({self.gamma_source})")
            
            # Show environmental understanding if available
            if env_context:
                print(f"   🌟 Environmental understanding acquired:")
                print(f"      🎯 Objective: {env_context['objective']['primary']}")
                print(f"      📋 Rules: {len(env_context.get('rules', {}))} rule categories understood")
                print(f"      🧠 Strategic concepts: {len(env_context.get('strategic_concepts', {}).get('core_skills', []))} core skills")
                print(f"      💡 Learning recommendations: {sum(len(recs) for recs in env_context.get('learning_recommendations', {}).values())} recommendations")
                
                # Show learning parameter adaptations
                if self.learning_parameters_adapted:
                    print(f"   🔧 Adaptive learning applied:")
                    print(f"      Gamma: {self.default_gamma:.3f} → {self.gamma:.3f}")
                    print(f"      Rationale: {self.environment_learning_metadata.get('gamma_rationale', 'Environment-specific optimization')}")

    def _adapt_learning_parameters(self, env_context):
        """Adapt learning parameters based on environment context"""
        if not env_context:
            self.gamma_source = "default"
            self.learning_parameters_adapted = False
            return
        
        # Extract learning parameters from environment context
        learning_params = env_context.get('learning_parameters', {})
        
        if learning_params:
            print("🔧 Adapting learning parameters for environment...")
            
            # Adapt gamma
            recommended_gamma = learning_params.get('recommended_gamma')
            if recommended_gamma and recommended_gamma != self.gamma:
                old_gamma = self.gamma
                self.environment_gamma = recommended_gamma
                self.gamma = recommended_gamma
                self.gamma_source = f"environment:{env_context.get('name', 'unknown')}"
                self.learning_parameters_adapted = True
                
                gamma_rationale = learning_params.get('gamma_rationale', 'Environment-specific optimization')
                self.environment_learning_metadata['gamma_rationale'] = gamma_rationale
                
                print(f"   🎯 Gamma adapted: {old_gamma:.3f} → {self.gamma:.3f}")
                print(f"   📝 Rationale: {gamma_rationale}")
            
            # Could adapt other parameters here in the future:
            # - learning_rate
            # - exploration parameters
            # - buffer sizes
            # - network architecture
            
            # Store temporal characteristics for potential future optimizations
            temporal_chars = learning_params.get('temporal_characteristics', {})
            if temporal_chars:
                self.environment_learning_metadata.update(temporal_chars)
                print(f"   ⏱️ Temporal characteristics understood:")
                print(f"      Match duration: {temporal_chars.get('match_duration', 'unknown')}")
                print(f"      Feedback immediacy: {temporal_chars.get('feedback_immediacy', 'unknown')}")
                print(f"      Decision frequency: {temporal_chars.get('decision_frequency', 'unknown')}")
        
        else:
            # No learning parameters provided, use defaults
            self.gamma_source = "default"
            self.learning_parameters_adapted = False
            print("   ⚙️ Using default learning parameters (no environment-specific recommendations)")

    def get_action(self, state):
        """ENHANCED action selection with intelligent symbolic knowledge application"""
        try:
            if isinstance(state, (list, tuple)):
                state = np.array(state)
            elif len(state.shape) > 1:
                state = state.flatten()
                
            # Get neural network Q-values
            q_values = self.main_network.forward(state)
            
            # ENHANCED: Use intelligent symbolic decision making
            if self.app_context:
                action, reasoning = self.symbolic_decision_maker.make_informed_decision(
                    state, q_values, self.app_context, self.exploration_rate
                )
                
                # Track decision type
                if "🧩" in reasoning:  # Symbolic decision was made
                    self.symbolic_decisions_made += 1
                    print(f"🎯 {reasoning}")
                    if self.logger and self.logger.current_match:
                        self.logger.log_symbolic_insight("strategic_decision", reasoning)
                        self.logger.log_strategic_decision({
                            'action': action,
                            'reasoning': reasoning,
                            'confidence': 0.8,  # Could extract this from reasoning
                            'strategy_used': reasoning.split("'")[1] if "'" in reasoning else "unknown"
                        })
                else:
                    self.neural_decisions_made += 1
                
            else:
                # Fallback to neural network decision
                if random.random() < self.exploration_rate:
                    action = random.randint(0, self.action_size - 1)
                else:
                    action = np.argmax(q_values)
                reasoning = "🧠 Neural network decision (no context)"
                self.neural_decisions_made += 1
            
            self.last_state = state.copy()
            self.last_action = action
            self.actions_taken += 1
            
            return action
            
        except Exception as e:
            print(f"⚠️ Agent action error: {e}")
            return random.randint(0, self.action_size - 1)

    def learn(self, reward, next_state, done=False):
        """Enhanced learning with adaptive gamma and symbolic insight generation"""
        if self.last_state is None or self.last_action is None:
            return
            
        try:
            if isinstance(next_state, (list, tuple)):
                next_state = np.array(next_state)
            elif len(next_state.shape) > 1:
                next_state = next_state.flatten()
            
            # NEW: Update strategy effectiveness tracking
            self.symbolic_decision_maker.update_strategy_effectiveness(reward)
            
            # Track hit-to-score bonuses and generate lessons
            if reward > 3.5:
                self.hit_to_score_bonuses += 1
                self.total_bonus_reward += (reward - 3.0)
                print(f"🎳 Hit-to-Score Bonus! Total: {self.hit_to_score_bonuses}")
                
                # Generate symbolic lesson
                lesson = f"Hit-to-score combo achieved with reward {reward:.1f} - timing and positioning critical"
                if self.dual_brain.knowledge.add_lesson(self.app_name, lesson):
                    self.lessons_learned_this_match += 1
            
            # Generate lessons based on reward patterns
            if reward > 5.0:
                lesson = f"High reward action ({reward:.1f}) - successful strategy worth repeating"
                if self.dual_brain.knowledge.add_lesson(self.app_name, lesson):
                    self.lessons_learned_this_match += 1
            elif reward < -3.0:
                lesson = f"Poor outcome ({reward:.1f}) - avoid similar action patterns"
                if self.dual_brain.knowledge.add_lesson(self.app_name, lesson):
                    self.lessons_learned_this_match += 1
            
            # Store experience
            experience = {
                'state': self.last_state.copy(),
                'action': self.last_action,
                'reward': reward,
                'next_state': next_state.copy(),
                'done': done,
                'source': 'agent',
                'learning_weight': 1.0
            }
            self.experience_buffer.append(experience)
            self.match_reward += reward
            self.total_reward += reward
            
            # Train networks with adaptive gamma
            if (len(self.experience_buffer) >= self.min_buffer_size and 
                self.training_steps % self.replay_frequency == 0):
                self._train_networks()
            
            # Update target network
            if self.use_soft_updates:
                self.target_network.soft_update_from(self.main_network, self.soft_update_tau)
                self.target_updates += 1
            else:
                if self.training_steps % self.target_update_frequency == 0:
                    self.target_network.copy_weights_from(self.main_network)
                    self.target_updates += 1
            
            self.training_steps += 1
            
            # Update dual brain core learning stats
            self.dual_brain.brain.training_steps = self.training_steps
            self.dual_brain.brain.target_updates = self.target_updates
            self.dual_brain.brain.epsilon = self.exploration_rate
            self.dual_brain.brain.total_loss = self.total_loss
            self.dual_brain.brain.gamma = self.gamma  # Update brain with current gamma
            
            # Decay exploration
            if self.training_steps % 100 == 0:
                if self.exploration_rate > self.min_exploration:
                    self.exploration_rate *= self.exploration_decay
            
            # Calculate knowledge effectiveness
            self._update_knowledge_effectiveness()
            
            # Update match stats including adaptive learning parameters
            if self.logger.current_match:
                current_stats = {
                    'match_reward': self.match_reward,
                    'total_reward': self.total_reward,
                    'actions_taken': self.actions_taken,
                    'training_steps': self.training_steps,
                    'target_updates': self.target_updates,
                    'double_dqn_improvements': self.double_dqn_improvements,
                    'exploration_rate_end': self.exploration_rate,
                    'hit_to_score_bonuses': self.hit_to_score_bonuses,
                    'total_bonus_reward': self.total_bonus_reward,
                    'human_demos_used': self.human_demos_used,
                    'user_demos_recorded': self.user_demos_recorded,
                    'user_demos_processed': self.user_demos_processed,
                    'symbolic_lessons_learned': self.lessons_learned_this_match,
                    'strategies_discovered': self.strategies_discovered_this_match,
                    'symbolic_decisions_made': self.symbolic_decisions_made,
                    'neural_decisions_made': self.neural_decisions_made,
                    'knowledge_effectiveness': self.knowledge_effectiveness,
                    # NEW: Adaptive learning tracking
                    'gamma_used': self.gamma,
                    'gamma_source': self.gamma_source,
                    'learning_parameters_adapted': self.learning_parameters_adapted
                }
                self.logger.update_match_stats(current_stats)
            
            # Periodic reporting with adaptive learning insights
            if self.training_steps % 500 == 0:
                avg_loss = self.total_loss / max(1, self.training_steps)
                strategy_performance = self.symbolic_decision_maker.get_strategy_performance_summary()
                print(f"🚀 Agent step {self.training_steps}:")
                print(f"   🎯 Target updates: {self.target_updates}, Exploration: {self.exploration_rate:.3f}")
                print(f"   🔧 Adaptive Learning: Gamma={self.gamma:.3f} ({self.gamma_source})")
                print(f"   🎳 Hit-to-Score bonuses: {self.hit_to_score_bonuses}")
                print(f"   👤 User demos: {len(self.user_demo_buffer)} available, {self.user_demos_processed} used")
                print(f"   🧩 Symbolic decisions: {self.symbolic_decisions_made}, Neural: {self.neural_decisions_made}")
                print(f"   📊 Knowledge effectiveness: {self.knowledge_effectiveness:.2f}")
                if strategy_performance:
                    print(f"   🎯 Strategy performance: {strategy_performance}")
                
                # Log symbolic insight including adaptive learning
                if self.logger.current_match:
                    insight = f"Training milestone: {self.training_steps} steps, gamma={self.gamma:.3f}, knowledge effectiveness: {self.knowledge_effectiveness:.2f}"
                    self.logger.log_symbolic_insight("training_milestone", insight)
                
        except Exception as e:
            print(f"❌ Learn error: {e}")

    def _update_knowledge_effectiveness(self):
        """Calculate knowledge system effectiveness"""
        strategy_performance = self.symbolic_decision_maker.get_strategy_performance_summary()
        if strategy_performance:
            # Average performance of symbolic vs neural decisions
            symbolic_perf = strategy_performance.get('symbolic', 0)
            neural_perf = strategy_performance.get('neural', 0)
            
            if neural_perf != 0:
                self.knowledge_effectiveness = max(0, symbolic_perf / neural_perf)
            else:
                self.knowledge_effectiveness = 1.0 if symbolic_perf > 0 else 0.0
        else:
            self.knowledge_effectiveness = 0.0

    def _train_networks(self):
        """Enhanced training with adaptive gamma and symbolic context"""
        # Use existing training implementation but with adaptive gamma
        batch_size = min(self.replay_batch_size, len(self.experience_buffer))
        batch = random.sample(list(self.experience_buffer), max(1, batch_size - 1))
        
        if len(self.user_demo_buffer) > 0:
            human_demo = random.choice(list(self.user_demo_buffer))
            batch.append(human_demo)
            self.human_demos_used += 1
            self.user_demos_processed += 1
        
        total_loss = 0
        double_dqn_benefits = 0
        
        for experience in batch:
            current_state = experience['state']
            action = experience['action']
            reward = experience['reward']
            next_state = experience.get('next_state', current_state)
            done = experience.get('done', False)
            
            next_q_main = self.main_network.forward(next_state)
            best_next_action = np.argmax(next_q_main)
            next_q_target = self.target_network.forward(next_state)
            
            if done:
                target_q_value = reward
            else:
                # Use adaptive gamma here!
                target_q_value = reward + self.gamma * next_q_target[best_next_action]
            
            if not done:
                standard_dqn_target = reward + self.gamma * np.max(next_q_target)
                if abs(target_q_value - standard_dqn_target) > 0.1:
                    double_dqn_benefits += 1
            
            current_q_values = self.main_network.forward(current_state)
            target_q_values = current_q_values.copy()
            target_q_values[action] = target_q_value
            
            loss = self.main_network.update_weights(current_state, target_q_values, action)
            total_loss += loss
        
        self.total_loss += total_loss / len(batch)
        self.double_dqn_improvements += double_dqn_benefits

    def get_stats(self):
        """Enhanced stats including adaptive learning parameters and knowledge system metrics"""
        avg_reward_per_game = self.total_reward / max(1, self.games_played)
        win_rate = self.wins / max(1, self.games_played)
        avg_loss = self.total_loss / max(1, self.training_steps)
        
        if self.exploration_rate > 0.6:
            learning_phase = "Exploring"
        elif self.exploration_rate > 0.4:
            learning_phase = "Learning"
        elif self.exploration_rate > 0.25:
            learning_phase = "Optimizing"
        else:
            learning_phase = "Expert"
        
        recent_perf = self.logger.get_recent_performance()
        
        # Get symbolic knowledge summary
        knowledge_summary = {}
        if self.app_name and self.app_context:
            knowledge_summary = {
                'strategies_available': len(self.app_context.get('strategies', [])),
                'lessons_learned': len(self.app_context.get('lessons', [])),
                'reflections_made': len(self.app_context.get('symbolic_reflections', [])),
                'symbolic_win_rate': self.dual_brain.knowledge._calculate_win_rate(self.app_context)
            }
        
        # Get strategy performance summary
        strategy_performance = self.symbolic_decision_maker.get_strategy_performance_summary()
        
        stats = {
            'games_played': int(self.games_played),
            'wins': int(self.wins),
            'win_rate': float(round(win_rate * 100, 1)),
            'exploration_rate': float(round(self.exploration_rate, 3)),
            'avg_reward_per_game': float(round(avg_reward_per_game, 2)),
            'match_reward': float(round(self.match_reward, 1)),
            'total_reward': float(round(self.total_reward, 1)),
            'actions_taken': int(self.actions_taken),
            'training_steps': int(self.training_steps),
            'experience_buffer_size': int(len(self.experience_buffer)),
            'avg_loss': float(round(avg_loss, 4)) if self.training_steps > 0 else 0.0,
            'learning_rate': float(self.learning_rate),
            'learning_phase': learning_phase,
            'strategic_moves': int(self.strategic_moves),
            'hit_to_score_bonuses': int(self.hit_to_score_bonuses),
            'total_bonus_reward': float(round(self.total_bonus_reward, 2)),
            'human_demos_used': int(self.human_demos_used),
            'user_demos_recorded': int(self.user_demos_recorded),
            'user_demos_processed': int(self.user_demos_processed),
            'user_demo_buffer_size': int(len(self.user_demo_buffer)),
            'demo_learning_weight': float(self.demo_learning_weight),
            'demo_replay_ratio': float(self.demo_replay_ratio),
            'architecture': 'Agent Byte v1.2 - Adaptive Learning + Knowledge System Enhanced',
            'target_updates': int(self.target_updates),
            'double_dqn_improvements': int(self.double_dqn_improvements),
            'network_parameters': 15000,
            'ball_tracking_score': float(round(avg_loss, 4)),
            
            # Symbolic learning stats
            'lessons_learned_this_match': int(self.lessons_learned_this_match),
            'strategies_discovered_this_match': int(self.strategies_discovered_this_match),
            
            # Knowledge system stats
            'symbolic_decisions_made': int(self.symbolic_decisions_made),
            'neural_decisions_made': int(self.neural_decisions_made),
            'knowledge_effectiveness': float(round(self.knowledge_effectiveness, 3)),
            'strategy_performance': strategy_performance,
            
            # NEW: Adaptive learning stats
            'gamma': float(round(self.gamma, 4)),
            'gamma_source': self.gamma_source,
            'default_gamma': float(round(self.default_gamma, 4)),
            'environment_gamma': float(round(self.environment_gamma, 4)) if self.environment_gamma else None,
            'learning_parameters_adapted': self.learning_parameters_adapted,
            'environment_learning_metadata': self.environment_learning_metadata,
            
            **knowledge_summary
        }
        
        if recent_perf:
            stats['recent_performance'] = recent_perf
        
        return stats

    def get_detailed_knowledge_analysis(self):
        """Get detailed analysis of knowledge application including adaptive learning"""
        if not self.app_context:
            return "No active knowledge context"
        
        strategy_performance = self.symbolic_decision_maker.get_strategy_performance_summary()
        
        analysis = f"""
🧩 KNOWLEDGE SYSTEM ANALYSIS - Agent Byte v1.2 Adaptive Learning
═══════════════════════════════════════════════════════════════

📊 Strategy Performance:
{self._format_strategy_performance(strategy_performance)}

🎯 Available Knowledge:
   Environmental Strategies: {len(self.app_context.get('environment_context', {}).get('strategic_concepts', {}).get('core_skills', []))}
   Tactical Approaches: {len(self.app_context.get('environment_context', {}).get('tactical_approaches', []))}
   Learned Strategies: {len(self.app_context.get('strategies', []))}
   Lessons Learned: {len(self.app_context.get('lessons', []))}

🔄 Decision Distribution:
   Symbolic Decisions: {self.symbolic_decisions_made} ({(self.symbolic_decisions_made/(max(1, self.symbolic_decisions_made + self.neural_decisions_made))*100):.1f}%)
   Neural Decisions: {self.neural_decisions_made} ({(self.neural_decisions_made/(max(1, self.symbolic_decisions_made + self.neural_decisions_made))*100):.1f}%)

📈 Knowledge Effectiveness: {self.knowledge_effectiveness:.3f}

⚙️ Adaptive Learning Status:
   Current Gamma: {self.gamma:.4f}
   Gamma Source: {self.gamma_source}
   Default Gamma: {self.default_gamma:.4f}
   Parameters Adapted: {'Yes' if self.learning_parameters_adapted else 'No'}
   {f'Environment Gamma: {self.environment_gamma:.4f}' if self.environment_gamma else ''}

📋 Recent Strategic Decisions:
{self._format_recent_decisions()}
        """
        
        return analysis.strip()
    
    def _format_strategy_performance(self, performance):
        if not performance:
            return "   No performance data yet"
        
        lines = []
        for strategy_type, avg_reward in performance.items():
            status = "✅" if avg_reward > 0 else "❌" if avg_reward < -0.5 else "⚖️"
            lines.append(f"   {status} {strategy_type}: {avg_reward:+.3f} avg reward")
        
        return "\n".join(lines)
    
    def _format_recent_decisions(self):
        recent = self.symbolic_decision_maker.decision_history[-5:]
        if not recent:
            return "   No recent decisions"
        
        lines = []
        for i, decision in enumerate(recent, 1):
            if decision['chosen'] == 'symbolic':
                lines.append(f"   {i}. 🧩 {decision['reasoning']}")
            else:
                lines.append(f"   {i}. 🧠 Neural network")
        
        return "\n".join(lines)

    # Keep existing demo learning methods (unchanged)
    def record_user_demo(self, demo_dict):
        """Enhanced user demonstration recording with symbolic insights"""
        try:
            if not all(key in demo_dict for key in ['state', 'action', 'reward', 'source', 'outcome']):
                print(f"❌ Invalid demo data: missing required keys")
                return False
            
            state = np.array(demo_dict["state"])
            if state.shape[0] != self.state_size:
                print(f"❌ Invalid demo state size: {state.shape[0]} != {self.state_size}")
                return False
            
            action = demo_dict["action"]
            if not (0 <= action < self.action_size):
                print(f"❌ Invalid demo action: {action} not in range [0, {self.action_size})")
                return False
            
            reward = demo_dict["reward"]
            outcome = demo_dict["outcome"]
            
            # Generate symbolic insight from demo
            if outcome == "hit" and reward > 1.0:
                lesson = f"User demonstrated successful hit technique with reward {reward:.1f}"
                if self.dual_brain.knowledge.add_lesson(self.app_name, lesson):
                    self.lessons_learned_this_match += 1
            
            enhanced_demo = {
                'state': state,
                'action': action,
                'reward': reward,
                'source': demo_dict["source"],
                'outcome': outcome,
                'timestamp': time.time(),
                'quality_score': self._evaluate_demo_quality(outcome, reward),
                'learning_weight': self._calculate_demo_weight(outcome, reward)
            }
            
            self.user_demo_buffer.append(enhanced_demo)
            self.user_demos_recorded += 1
            
            if self.logger.current_match:
                self.logger.log_user_demonstration(enhanced_demo)
            
            print(f"👤 User demo recorded: Action={action}, Outcome={outcome}, Quality={enhanced_demo['quality_score']:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ Error recording user demo: {e}")
            return False

    def _evaluate_demo_quality(self, outcome, reward):
        """Evaluate the quality of a user demonstration"""
        if outcome == "hit":
            return min(1.0, 0.8 + reward * 0.2)
        elif outcome == "miss":
            return max(0.1, 0.3 + reward * 0.1)
        elif outcome == "positioning":
            return max(0.2, 0.5 + reward * 0.5)
        else:
            return 0.5

    def _calculate_demo_weight(self, outcome, reward):
        """Calculate learning weight for demo"""
        base_weight = self.demo_learning_weight
        
        if outcome == "hit" and reward > 1.0:
            return base_weight * 1.5
        elif outcome == "miss" and reward < 0:
            return base_weight * 0.7
        elif outcome == "positioning":
            return base_weight * 0.8
        else:
            return base_weight

    def _calculate_demo_effectiveness(self):
        """Calculate how effective user demonstrations have been"""
        if self.user_demos_processed == 0:
            return 0.0
        
        usage_rate = self.user_demos_processed / max(1, len(self.user_demo_buffer))
        match_performance = max(0, self.match_reward) / max(1, abs(self.match_reward))
        
        return min(1.0, (usage_rate + match_performance) / 2)

    def end_match(self, winner, final_scores=None, game_stats=None):
        """Enhanced match ending with adaptive learning analysis"""
        try:
            # Determine outcome for symbolic learning
            win = (winner == "Agent Byte")
            outcome_summary = f"{'Victory' if win else 'Defeat'} with reward {self.match_reward:.1f}"
            
            # Generate strategic insights based on performance
            if win and self.match_reward > 10:
                strategy = f"Winning strategy: Achieved {self.match_reward:.1f} reward through consistent play"
                if self.dual_brain.knowledge.add_strategy(self.app_name, strategy):
                    self.strategies_discovered_this_match += 1
            elif not win and self.match_reward < -5:
                lesson = f"Loss pattern: Negative reward {self.match_reward:.1f} indicates strategy adjustment needed"
                if self.dual_brain.knowledge.add_lesson(self.app_name, lesson):
                    self.lessons_learned_this_match += 1
            
            # Record result in symbolic knowledge
            self.dual_brain.knowledge.record_game_result(self.app_name, win, self.match_reward, outcome_summary)
            
            # Generate reflections
            reflections = self.dual_brain.knowledge.reflect_on_performance(self.app_name)
            
            # Analyze knowledge system performance
            strategy_performance = self.symbolic_decision_maker.get_strategy_performance_summary()
            
            # End dual brain session
            self.dual_brain.end_session(win, self.match_reward, outcome_summary)
            
            # Prepare final stats with adaptive learning info
            final_stats = {
                'match_reward': round(self.match_reward, 2),
                'total_reward': round(self.total_reward, 2),
                'actions_taken': self.actions_taken,
                'training_steps': self.training_steps,
                'target_updates': self.target_updates,
                'double_dqn_improvements': self.double_dqn_improvements,
                'exploration_rate_start': getattr(self, 'match_start_exploration', self.exploration_rate),
                'exploration_rate_end': self.exploration_rate,
                'architecture': 'Agent Byte v1.2 - Adaptive Learning + Knowledge System Enhanced',
                'hit_to_score_bonuses': self.hit_to_score_bonuses,
                'total_bonus_reward': round(self.total_bonus_reward, 2),
                'human_demos_used': self.human_demos_used,
                'user_demos_recorded': self.user_demos_recorded,
                'user_demos_processed': self.user_demos_processed,
                'symbolic_lessons_learned': self.lessons_learned_this_match,
                'strategies_discovered': self.strategies_discovered_this_match,
                'reflections_generated': len(reflections),
                'symbolic_decisions_made': self.symbolic_decisions_made,
                'neural_decisions_made': self.neural_decisions_made,
                'knowledge_effectiveness': round(self.knowledge_effectiveness, 3),
                'strategy_performance': strategy_performance,
                # NEW: Adaptive learning final stats
                'gamma_used': round(self.gamma, 4),
                'gamma_source': self.gamma_source,
                'default_gamma': round(self.default_gamma, 4),
                'learning_parameters_adapted': self.learning_parameters_adapted,
                'environment_learning_metadata': self.environment_learning_metadata
            }
            
            if game_stats:
                final_stats.update(game_stats)
            
            # End logging
            self.logger.end_match(winner, final_scores or {'player': 0, 'agent_byte': 0}, final_stats)
            
            self.games_played += 1
            if winner == "Agent Byte":
                self.wins += 1
            
            print(f"🏁 Match ended: {winner} wins!")
            print(f"   💰 Match reward: {self.match_reward:.1f}")
            print(f"   🎳 Hit-to-Score bonuses: {self.hit_to_score_bonuses}")
            print(f"   👤 User demos: recorded={self.user_demos_recorded}, used={self.user_demos_processed}")
            print(f"   🧩 Knowledge system: {self.symbolic_decisions_made} symbolic, {self.neural_decisions_made} neural decisions")
            print(f"   📊 Knowledge effectiveness: {self.knowledge_effectiveness:.2f}")
            print(f"   🔧 Learning: Gamma={self.gamma:.3f} ({self.gamma_source}), Adapted={self.learning_parameters_adapted}")
            print(f"   🤔 Generated {len(reflections)} new reflections")
            if strategy_performance:
                print(f"   🎯 Strategy performance: {strategy_performance}")
            
            return final_stats
            
        except Exception as e:
            print(f"❌ Error in end_match: {e}")
            return None

    def save_brain(self, filename=None):
        """Save dual brain system with adaptive learning metadata"""
        # Save both brains
        success = self.dual_brain.save_all()
        
        if success:
            win_rate = (self.wins / max(1, self.games_played)) * 100
            print(f"💾 Agent Byte v1.2 Adaptive Learning + Knowledge System Enhanced saved!")
            print(f"   📊 Games: {self.games_played}, Win rate: {win_rate:.1f}%")
            print(f"   🧠 Core brain: {self.training_steps} steps, {self.target_updates} updates")
            print(f"   🧩 Knowledge: {len(self.dual_brain.knowledge.knowledge.get('categories', {}).get('games', {}))} environments")
            print(f"   🎯 Knowledge effectiveness: {self.knowledge_effectiveness:.3f}")
            print(f"   🔧 Adaptive learning: Gamma={self.gamma:.3f} ({self.gamma_source})")
        
        return success

    def load_brain(self, filename=None):
        """Load dual brain system (automatically handled during initialization)"""
        # Brain loading is handled by dual brain system initialization
        return True

if __name__ == "__main__":
    print("🧪 Testing Enhanced Agent Byte with Adaptive Learning + Knowledge System...")
    
    # Test with environment context that includes learning parameters
    test_env_context = {
        'name': 'test_pong',
        'learning_parameters': {
            'recommended_gamma': 0.90,
            'gamma_rationale': 'Short-term competitive game with immediate feedback',
            'recommended_learning_rate': 0.001,
            'temporal_characteristics': {
                'match_duration': '2-5 minutes',
                'feedback_immediacy': 'Immediate'
            }
        }
    }
    
    agent = AgentByte(state_size=14, action_size=3, app_name="test_pong")
    agent.start_new_match("test_pong", env_context=test_env_context)
    
    test_state = np.random.random(14)
    
    # Test some actions and learning
    for i in range(10):
        action = agent.get_action(test_state)
        reward = random.uniform(-2, 5)
        agent.learn(reward=reward, next_state=test_state, done=False)
        print(f"Action {i+1}: {action}, Reward: {reward:.2f}")
    
    # Test user demo
    demo_success = agent.record_user_demo({
        'state': test_state.tolist(),
        'action': 1,
        'reward': 1.5,
        'source': 'user_action',
        'outcome': 'hit'
    })
    
    # Show detailed analysis
    print("\n" + "="*50)
    print(agent.get_detailed_knowledge_analysis())
    
    # End match
    agent.end_match("Agent Byte", {'player': 15, 'agent_byte': 21})
    
    # Save everything
    agent.save_brain()
    
    print(f"\n✅ Enhanced Agent Byte v1.2 Adaptive Learning test complete!")
    print(f"🧩 Symbolic decisions: {agent.symbolic_decisions_made}")
    print(f"🧠 Neural decisions: {agent.neural_decisions_made}")
    print(f"📊 Knowledge effectiveness: {agent.knowledge_effectiveness:.3f}")
    print(f"🔧 Gamma adapted: {agent.default_gamma:.3f} → {agent.gamma:.3f} ({agent.gamma_source})")
    print(f"⚙️ Learning parameters adapted: {agent.learning_parameters_adapted}")
