# agent_byte.py - Enhanced Modular Agent with Environment Integration
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
from dueling_network_pytorch import DuelingNetworkPyTorch, DuelingNetworkCompatibility


class MatchLogger:
    """Match logging system for tracking performance across games"""

    def __init__(self, log_filename='agent_matches.json'):
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
                    # Track lifetime total if available
                    self.total_matches_lifetime = data.get('total_matches_all_time', len(self.all_matches))
                    
                    print(f"📚 Loaded {len(self.all_matches)} recent match records from {self.log_file}")
                    if 'total_matches_all_time' in data:
                        print(f"   (Total lifetime matches: {self.total_matches_lifetime})")
        except Exception as e:
            print(f"⚠️ Could not load match history: {e}")
            self.all_matches = []
            self.total_matches_lifetime = 0

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
                'architecture': 'Agent Byte v1.2 - Modular + Adaptive Learning + Knowledge System Enhanced',
                'hit_to_score_bonuses': 0,
                'human_demos_used': 0,
                'user_demos_recorded': 0,
                'demo_learning_weight': 0.3,
                'symbolic_lessons_learned': 0,
                'strategies_discovered': 0,
                'symbolic_decisions_made': 0,
                'neural_decisions_made': 0,
                'knowledge_effectiveness': 0.0,
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
            'learning_adaptations': []
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

    def log_user_demonstration(self, demo_data):
        """Log user demonstration data"""
        if self.current_match:
            demo = {
                'timestamp': time.time(),
                'action': demo_data.get('action'),
                'outcome': demo_data.get('outcome'),
                'reward': demo_data.get('reward'),
                'quality_score': demo_data.get('quality_score'),
                'learning_weight': demo_data.get('learning_weight')
            }
            self.current_match['user_demonstrations'].append(demo)

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
                self.current_match['agent_byte_stats'].update(self._convert_numpy_types(final_stats))
            start = datetime.datetime.fromisoformat(self.current_match['start_time'])
            end = datetime.datetime.fromisoformat(self.current_match['end_time'])
            self.current_match['duration_seconds'] = (end - start).total_seconds()
            self.all_matches.append(self.current_match.copy())
            
            # LIMIT TO 3 MOST RECENT MATCHES to reduce file size
            if len(self.all_matches) > 3:
                self.all_matches = self.all_matches[-3:]
                print(f"🗂️ Match history trimmed to last 3 matches")
            
            self.save_match_history()
            print(f"📊 Match {self.current_match['match_id']} completed and logged")
            self.current_match = None
        except Exception as e:
            print(f"❌ Error ending match: {e}")
            self.current_match = None

    def save_match_history(self):
        try:
            data = {
                'total_matches_all_time': getattr(self, 'total_matches_lifetime', len(self.all_matches)),
                'matches_in_history': len(self.all_matches),
                'last_updated': datetime.datetime.now().isoformat(),
                'version': 'Agent Byte v1.2 - PyTorch + Adaptive Learning + Knowledge System Enhanced',
                'note': 'Only keeping last 3 matches to optimize file size',
                'matches': self._convert_numpy_types(self.all_matches)
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

        gamma_adaptations = sum(
            1 for match in recent if match['agent_byte_stats'].get('learning_parameters_adapted', False))

        return {
            'matches_analyzed': total_matches,
            'recent_win_rate': (wins / total_matches * 100) if total_matches > 0 else 0,
            'recent_hit_rate': sum(match['agent_byte_stats'].get('task_success_rate', 0) for match in
                                   recent) / total_matches if total_matches > 0 else 0,
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

    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable types"""
        import numpy as np
        from typing import Any, Union, Dict, List

        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj.item())  # 🔧 FIX: Use .item() to extract scalar
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj.item())  # 🔧 FIX: Use .item() to extract scalar
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj.item() if hasattr(obj, 'item') else obj)  # 🔧 FIX: Safe bool conversion
        elif isinstance(obj, (int, float, str, bool)):
            return obj  # 🔧 FIX: Pass through regular Python types
        else:
            try:
                # 🔧 FIX: Try to convert unknown numpy types
                if hasattr(obj, 'item'):
                    return obj.item()
                else:
                    return obj
            except (ValueError, TypeError):
                # 🔧 FIX: Fallback for unconvertible types
                return str(obj)

# Legacy NumPy DuelingNetwork - DEPRECATED - Use PyTorch version instead
# Keeping for reference/fallback only
class DuelingNetworkLegacy:
    """Legacy NumPy Dueling DQN Network implementation - DEPRECATED"""
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        print("⚠️ Using legacy NumPy network - Consider upgrading to PyTorch version")
        # ... keeping original implementation for fallback ...
        
class AgentByte:
    """Enhanced Modular Agent Byte with Environment Integration + Dual Brain Architecture + Knowledge System"""
    
    def __init__(self, state_size=14, action_size=3, logger=None, app_name="unknown_game"):
        print("🚀 Agent Byte v1.2 - Modular + Adaptive Learning + Knowledge System Enhanced Initializing...")
        
        # Initialize dual brain system
        self.dual_brain = DualBrainAgent()
        self.app_name = app_name
        self.app_context = None
        
        # NEW: Environment integration for modular behavior
        self.env = None  # Will be set by the coordinator
        self.env_context = None
        self.env_constants = {}
        
        # Initialize symbolic decision maker
        self.symbolic_decision_maker = SymbolicDecisionMaker()
        
        # Neural network components - PyTorch implementation
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = [64, 32, 16]
        
        # Initialize PyTorch networks
        self.pytorch_main_network = DuelingNetworkPyTorch(
            self.state_size, self.hidden_sizes, self.action_size, learning_rate=0.001
        )
        self.pytorch_target_network = DuelingNetworkPyTorch(
            self.state_size, self.hidden_sizes, self.action_size, learning_rate=0.001
        )
        
        # Wrap in compatibility layer for existing code
        self.main_network = DuelingNetworkCompatibility(self.pytorch_main_network)
        self.target_network = DuelingNetworkCompatibility(self.pytorch_target_network)
        
        # Initialize target network with main network weights
        self.target_network.copy_weights_from(self.main_network)
        
        # Neural network save paths
        self.main_network_path = "agent_main_network.pth"
        self.target_network_path = "agent_target_network.pth"
        
        # Try to load existing networks
        self._load_neural_networks()
        
        # Learning parameters (will be adapted per environment)
        self.target_update_frequency = 1000
        self.soft_update_tau = 0.005
        self.use_soft_updates = True
        self.learning_rate = self.dual_brain.brain.learning_rate
        self.exploration_rate = self.dual_brain.brain.epsilon
        self.exploration_decay = 0.995
        self.min_exploration = 0.1
        self.gamma = self.dual_brain.brain.gamma  # Default gamma from brain, will be overridden per environment
        
        # Adaptive learning tracking
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
        
        print("✅ Agent Byte v1.2 Modular + Adaptive Learning + Knowledge System Enhanced Created!")
        print(f"   🧠 Core Brain: {self.training_steps} training steps")
        print(f"   🧩 Knowledge: Symbolic learning + intelligent application")
        print(f"   🎯 Architecture: Neural + Symbolic Decision Making")
        print(f"   👤 Demo Learning: Enhanced with symbolic understanding")
        print(f"   ⚙️ Adaptive Learning: Environment-specific parameter optimization")
        print(f"   🔧 Default Gamma: {self.gamma} (will adapt per environment)")
        print(f"   🏗️ Modular Design: Ready for environment-specific integration")

    def set_environment(self, env):
        """Set the environment instance for modular behavior"""
        self.env = env
        if hasattr(env, 'get_env_context'):
            self.env_context = env.get_env_context()
            print(f"🌟 Environment context loaded: {self.env_context.get('name', 'unknown')}")
        
        if hasattr(env, 'get_environment_specific_constants'):
            self.env_constants = env.get_environment_specific_constants()
            print(f"⚙️ Environment constants loaded: {len(self.env_constants)} parameters")

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

        # Adapt learning parameters based on environment context
        self._adapt_learning_parameters(env_context)

        # Start logging
        match_id = f"agent_byte_{game_type}_{int(time.time())}"
        self.logger.start_match(match_id, game_type)

        #  Use 'agent_stats' instead of 'agent_byte_stats'
        if self.logger.current_match:
            self.logger.current_match['agent_byte_stats']['exploration_rate_start'] = self.exploration_rate
            self.logger.current_match['agent_byte_stats']['gamma_used'] = self.gamma
            self.logger.current_match['agent_byte_stats']['gamma_source'] = self.gamma_source
            self.logger.current_match['agent_byte_stats'][
                'learning_parameters_adapted'] = self.learning_parameters_adapted

            # Log environmental context integration
            if env_context:
                self.logger.log_symbolic_insight("env_context_loaded",
                                                 f"Environment context integrated for {game_type}")

                # Log learning parameter adaptations
                if self.learning_parameters_adapted:
                    self.logger.log_learning_adaptation({
                        'parameter': 'gamma',
                        'old_value': self.default_gamma,
                        'new_value': self.gamma,
                        'source': f'environment:{game_type}',
                        'rationale': self.environment_learning_metadata.get('gamma_rationale',
                                                                            'Environment-specific optimization')
                    })

        print(f"🆕 New {game_type} match started with modular adaptive learning + knowledge system enabled")
        if self.app_context:
            strategies = len(self.app_context.get('strategies', []))
            lessons = len(self.app_context.get('lessons', []))
            print(f"   📚 Available knowledge: {strategies} strategies, {lessons} lessons")
            print(f"   🧩 Knowledge system: Active and ready for intelligent decision making")
            print(f"   ⚙️ Learning parameters: Gamma={self.gamma:.3f} ({self.gamma_source})")
            print(f"   🏗️ Modular integration: Environment-specific behavior enabled")
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
                            'confidence': 0.8,
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
        """Enhanced learning with modular environment integration and adaptive gamma"""
        if self.last_state is None or self.last_action is None:
            return
            
        try:
            if isinstance(next_state, (list, tuple)):
                next_state = np.array(next_state)
            elif len(next_state.shape) > 1:
                next_state = next_state.flatten()
            
            # Update strategy effectiveness tracking
            self.symbolic_decision_maker.update_strategy_effectiveness(reward)
            
            # NEW: Use environment-specific interpretation for learning
            if self.env:
                # Check if environment should generate lessons or strategies
                if hasattr(self.env, 'should_generate_lesson') and self.env.should_generate_lesson(reward):
                    if hasattr(self.env, 'generate_lesson_from_reward'):
                        lesson = self.env.generate_lesson_from_reward(reward)
                        if self.dual_brain.knowledge.add_lesson(self.app_name, lesson):
                            self.lessons_learned_this_match += 1
                            print(f"📚 Environment lesson: {lesson}")
                
                if hasattr(self.env, 'should_generate_strategy') and self.env.should_generate_strategy(reward):
                    if hasattr(self.env, 'generate_strategy_from_performance'):
                        strategy = self.env.generate_strategy_from_performance(
                            self.wins, self.games_played, self.total_reward / max(1, self.games_played)
                        )
                        if self.dual_brain.knowledge.add_strategy(self.app_name, strategy):
                            self.strategies_discovered_this_match += 1
                            print(f"🎯 Environment strategy: {strategy}")
                
                # Track environment-specific bonuses using environment constants
                if self.env_constants:
                    hit_bonus_threshold = self.env_constants.get('hit_to_score_bonus_threshold', 3.5)
                    if reward > hit_bonus_threshold:
                        self.hit_to_score_bonuses += 1
                        bonus_amount = reward - self.env_constants.get('task_completion_bonus', 3.0)
                        self.total_bonus_reward += max(0, bonus_amount)
                        
                        # Use environment-specific interpretation
                        if hasattr(self.env, 'interpret_reward'):
                            interpretation = self.env.interpret_reward(reward)
                            print(f"🎳 {interpretation}! Total bonuses: {self.hit_to_score_bonuses}")
                        else:
                            print(f"🎳 Bonus achieved! Total: {self.hit_to_score_bonuses}")
            
            else:
                # Fallback behavior when no environment is set
                # Use generic thresholds and messages
                if reward > 3.5:
                    self.hit_to_score_bonuses += 1
                    self.total_bonus_reward += max(0, reward - 3.0)
                    lesson = f"High reward combination ({reward:.1f}) - successful strategy worth repeating"
                    if self.dual_brain.knowledge.add_lesson(self.app_name, lesson):
                        self.lessons_learned_this_match += 1
                        print(f"📚 Generic lesson: {lesson}")
                
                if reward > 5.0:
                    strategy = f"High performance strategy: Achieved {reward:.1f} reward through consistent execution"
                    if self.dual_brain.knowledge.add_strategy(self.app_name, strategy):
                        self.strategies_discovered_this_match += 1
                        print(f"🎯 Generic strategy: {strategy}")
            
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
            
            # Update target network with adaptive parameters
            if self.use_soft_updates:
                # Calculate adaptive parameters for soft updates
                win_rate = (self.wins / max(1, self.games_played))
                recent_rewards = [exp.get('reward', 0) for exp in list(self.experience_buffer)[-10:]]
                adaptive_params = self.main_network.get_adaptive_params(win_rate, recent_rewards)
                
                self.target_network.soft_update_from(self.main_network, self.soft_update_tau, adaptive_params)
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
            self.dual_brain.brain.gamma = self.gamma
            
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
                print(f"   🏗️ Environment integration: {'Active' if self.env else 'Inactive'}")
                print(f"   🎳 Bonuses: {self.hit_to_score_bonuses}")
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
            symbolic_perf = strategy_performance.get('symbolic', 0.0)  # 🔧 FIX: Use 0.0 (float)
            neural_perf = strategy_performance.get('neural', 0.0)  # 🔧 FIX: Use 0.0 (float)

            if neural_perf != 0.0:  # 🔧 FIX: Compare with 0.0 (float)
                self.knowledge_effectiveness = max(0.0, symbolic_perf / neural_perf)  # 🔧 FIX: Use 0.0 (float)
            else:
                self.knowledge_effectiveness = 1.0 if symbolic_perf > 0.0 else 0.0  # 🔧 FIX: Use 0.0 and 1.0 (float)
        else:
            self.knowledge_effectiveness = 0.0  # 🔧 FIX: Use 0.0 (float)

    def _train_networks(self):
        """Enhanced training with adaptive learning parameters and symbolic context"""
        # Calculate adaptive parameters based on recent performance
        win_rate = (self.wins / max(1, self.games_played))
        recent_rewards = [exp.get('reward', 0) for exp in list(self.experience_buffer)[-20:]]
        adaptive_params = self.main_network.get_adaptive_params(win_rate, recent_rewards)
        
        # Use existing training implementation but with adaptive parameters
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
            
            # NEW: Use adaptive parameters for weight updates
            loss = self.main_network.update_weights(current_state, target_q_values, action, adaptive_params)
            total_loss += loss
        
        self.total_loss += total_loss / len(batch)
        self.double_dqn_improvements += double_dqn_benefits
        
        # Track performance trend for adaptive learning rate
        if hasattr(self, 'previous_loss'):
            performance_trend = self.previous_loss - (total_loss / len(batch))
            self.main_network.set_adaptive_learning_rate(performance_trend)
        self.previous_loss = total_loss / len(batch)

    def record_user_demo(self, demo_dict):
        """Enhanced user demonstration recording with environment-specific insights"""
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
            
            # NEW: Use environment-specific interpretation for demo learning
            if self.env and hasattr(self.env, 'format_user_demo_outcome'):
                formatted_outcome = self.env.format_user_demo_outcome(outcome, reward)
                print(f"👤 {formatted_outcome}")
            
            # Generate symbolic insight from demo using environment context
            if outcome == "hit" and reward > 1.0:
                if self.env and hasattr(self.env, 'generate_lesson_from_reward'):
                    lesson = self.env.generate_lesson_from_reward(reward, context={'source': 'user_demo'})
                else:
                    lesson = f"User demonstrated successful technique with reward {reward:.1f}"
                
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
            
            # Use environment-specific feedback if available
            if self.env and hasattr(self.env, 'get_performance_feedback_phrase'):
                feedback = self.env.get_performance_feedback_phrase("demo_quality", enhanced_demo['quality_score'] * 100)
                print(f"👤 User demo recorded: Action={action}, Outcome={outcome}, {feedback}")
            else:
                print(f"👤 User demo recorded: Action={action}, Outcome={outcome}, Quality={enhanced_demo['quality_score']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error recording user demo: {e}")
            return False

    def _evaluate_demo_quality(self, outcome, reward):
        """Evaluate the quality of a user demonstration using environment constants if available"""
        if self.env_constants:
            demo_success_reward = self.env_constants.get('demo_success_reward', 1.5)
            demo_failure_penalty = self.env_constants.get('demo_failure_penalty', -0.5)
            positioning_reward = self.env_constants.get('positioning_reward', 0.1)
            
            if outcome == "hit":
                return min(1.0, 0.8 + (reward / demo_success_reward) * 0.2)
            elif outcome == "miss":
                return max(0.1, 0.3 + (reward / abs(demo_failure_penalty)) * 0.1)
            elif outcome == "positioning":
                return max(0.2, 0.5 + (reward / positioning_reward) * 0.5)
        else:
            # Fallback to original logic
            if outcome == "hit":
                return min(1.0, 0.8 + reward * 0.2)
            elif outcome == "miss":
                return max(0.1, 0.3 + reward * 0.1)
            elif outcome == "positioning":
                return max(0.2, 0.5 + reward * 0.5)
        
        return 0.5

    def _calculate_demo_weight(self, outcome, reward):
        """Calculate learning weight for demo using environment constants"""
        base_weight = self.demo_learning_weight
        
        if self.env_constants:
            demo_success_reward = self.env_constants.get('demo_success_reward', 1.5)
            
            if outcome == "hit" and reward > demo_success_reward * 0.7:
                return base_weight * 1.5
            elif outcome == "miss" and reward < 0:
                return base_weight * 0.7
            elif outcome == "positioning":
                return base_weight * 0.8
        else:
            # Fallback to original logic
            if outcome == "hit" and reward > 1.0:
                return base_weight * 1.5
            elif outcome == "miss" and reward < 0:
                return base_weight * 0.7
            elif outcome == "positioning":
                return base_weight * 0.8
        
        return base_weight

    def _calculate_demo_effectiveness(self):
        """Calculate how effective user demonstrations have been"""
        if self.user_demos_processed == 0:
            return 0.0
        
        usage_rate = self.user_demos_processed / max(1, len(self.user_demo_buffer))
        match_performance = max(0, self.match_reward) / max(1, abs(self.match_reward))
        
        return min(1.0, (usage_rate + match_performance) / 2)

    def end_match(self, winner, final_scores=None, game_stats=None):
        """Enhanced match ending with modular environment integration"""
        try:
            # Determine outcome for symbolic learning
            win = (winner == "Agent Byte")
            
            # NEW: Use environment-specific interpretation if available
            if self.env and hasattr(self.env, 'interpret_reward'):
                outcome_summary = self.env.interpret_reward(self.match_reward)
            else:
                outcome_summary = f"{'Victory' if win else 'Defeat'} with reward {self.match_reward:.1f}"
            
            # Generate strategic insights based on performance using environment context
            if win and self.match_reward > 10:
                if self.env and hasattr(self.env, 'generate_strategy_from_performance'):
                    strategy = self.env.generate_strategy_from_performance(
                        self.wins + 1, self.games_played + 1, self.match_reward
                    )
                else:
                    strategy = f"Winning strategy: Achieved {self.match_reward:.1f} reward through consistent play"
                
                if self.dual_brain.knowledge.add_strategy(self.app_name, strategy):
                    self.strategies_discovered_this_match += 1
            
            elif not win and self.match_reward < -5:
                if self.env and hasattr(self.env, 'generate_lesson_from_reward'):
                    lesson = self.env.generate_lesson_from_reward(self.match_reward, context={'match_end': True})
                else:
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
                'architecture': 'Agent Byte v1.2 - Modular + Adaptive Learning + Knowledge System Enhanced',
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
                'gamma_used': round(self.gamma, 4),
                'gamma_source': self.gamma_source,
                'default_gamma': round(self.default_gamma, 4),
                'learning_parameters_adapted': self.learning_parameters_adapted,
                'environment_learning_metadata': self.environment_learning_metadata,
                'environment_integrated': self.env is not None,
                'modular_behavior_active': hasattr(self.env, 'interpret_reward') if self.env else False
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
            print(f"   🏗️ Environment integration: {'Active' if self.env else 'Inactive'}")
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

    def get_stats(self):
        """Enhanced stats including adaptive learning parameters, knowledge system metrics, and environment integration"""
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
            'architecture': 'Agent Byte v1.2 - Modular + Adaptive Learning + Knowledge System Enhanced',
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
            
            # Adaptive learning stats
            'gamma': float(round(self.gamma, 4)),
            'gamma_source': self.gamma_source,
            'default_gamma': float(round(self.default_gamma, 4)),
            'environment_gamma': float(round(self.environment_gamma, 4)) if self.environment_gamma else None,
            'learning_parameters_adapted': self.learning_parameters_adapted,
            'environment_learning_metadata': self.environment_learning_metadata,
            
            # NEW: Modular environment integration stats
            'environment_integrated': self.env is not None,
            'environment_name': self.env_context.get('name', 'unknown') if self.env_context else 'none',
            'environment_constants_loaded': len(self.env_constants),
            'modular_behavior_active': hasattr(self.env, 'interpret_reward') if self.env else False,
            'environment_specific_learning': bool(self.env and hasattr(self.env, 'generate_lesson_from_reward')),
            
            **knowledge_summary
        }
        
        if recent_perf:
            stats['recent_performance'] = recent_perf
        
        return stats

    def get_detailed_knowledge_analysis(self):
        """Get detailed analysis of knowledge application including adaptive learning and environment integration"""
        if not self.app_context:
            return "No active knowledge context"
        
        strategy_performance = self.symbolic_decision_maker.get_strategy_performance_summary()
        
        analysis = f"""
🧩 MODULAR KNOWLEDGE SYSTEM ANALYSIS - Agent Byte v1.2
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

🏗️ Environment Integration:
   Environment Active: {'Yes' if self.env else 'No'}
   Environment Name: {self.env_context.get('name', 'unknown') if self.env_context else 'none'}
   Constants Loaded: {len(self.env_constants)}
   Modular Behavior: {'Active' if (self.env and hasattr(self.env, 'interpret_reward')) else 'Inactive'}
   Environment Learning: {'Active' if (self.env and hasattr(self.env, 'generate_lesson_from_reward')) else 'Inactive'}

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

    def _load_neural_networks(self):
        """Load neural networks if they exist"""
        try:
            main_loaded = self.pytorch_main_network.load_model(self.main_network_path)
            target_loaded = self.pytorch_target_network.load_model(self.target_network_path)
            
            if main_loaded and target_loaded:
                print(f"🧠 Neural networks loaded successfully")
                print(f"   Main: {self.main_network_path}")
                print(f"   Target: {self.target_network_path}")
                return True
            elif main_loaded:
                print(f"🧠 Main network loaded, copying to target network")
                self.target_network.copy_weights_from(self.main_network)
                return True
            else:
                print(f"🆕 No existing neural networks found, starting with random weights")
                return False
        except Exception as e:
            print(f"⚠️ Error loading neural networks: {e}")
            return False
    
    def _save_neural_networks(self):
        """Save neural networks"""
        try:
            self.pytorch_main_network.save_model(self.main_network_path)
            self.pytorch_target_network.save_model(self.target_network_path)
            print(f"💾 Neural networks saved successfully")
            return True
        except Exception as e:
            print(f"❌ Error saving neural networks: {e}")
            return False
    
    def save_brain(self, filename=None):
        """Save dual brain system with neural networks, adaptive learning metadata and environment integration info"""
        # Save neural networks first
        neural_success = self._save_neural_networks()
        
        # Save dual brain system
        brain_success = self.dual_brain.save_all()
        
        if neural_success and brain_success:
            win_rate = (self.wins / max(1, self.games_played)) * 100
            print(f"💾 Agent Byte v1.2 PyTorch + Adaptive Learning + Knowledge System Enhanced saved!")
            print(f"   📊 Games: {self.games_played}, Win rate: {win_rate:.1f}%")
            print(f"   🧠 Core brain: {self.training_steps} steps, {self.target_updates} updates")
            print(f"   🔗 Neural networks: Saved with PyTorch (GPU compatible)")
            print(f"   🧩 Knowledge: {len(self.dual_brain.knowledge.knowledge.get('categories', {}).get('games', {}))} environments")
            print(f"   🎯 Knowledge effectiveness: {self.knowledge_effectiveness:.3f}")
            print(f"   🔧 Adaptive learning: Gamma={self.gamma:.3f} ({self.gamma_source})")
            print(f"   🏗️ Environment integration: {'Active' if self.env else 'Inactive'}")
        
        return neural_success and brain_success

    def load_brain(self, filename=None):
        """Load dual brain system (automatically handled during initialization)"""
        # Brain loading is handled by dual brain system initialization
        return True


# Test section
if __name__ == "__main__":
    print("🧪 Testing Enhanced Modular Agent Byte with Environment Integration...")
    
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
    
    # Test environment integration
    class MockEnvironment:
        def get_environment_specific_constants(self):
            return {
                'hit_to_score_bonus_threshold': 3.5,
                'high_reward_threshold': 5.0,
                'task_completion_bonus': 3.0,
                'demo_success_reward': 1.5,
                'demo_failure_penalty': -0.5,
                'positioning_reward': 0.1
            }
        
        def interpret_reward(self, reward):
            if reward >= 4.0:
                return "Excellent combo execution"
            elif reward >= 1.0:
                return "Successful task completion"
            else:
                return "Learning opportunity"
        
        def generate_lesson_from_reward(self, reward, context=None):
            interpretation = self.interpret_reward(reward)
            source = context.get('source', 'general') if context else 'general'
            return f"Environment lesson ({source}): {interpretation} (reward={reward:.1f})"
        
        def should_generate_lesson(self, reward):
            return abs(reward) > 1.0
        
        def should_generate_strategy(self, reward):
            return reward > 5.0
        
        def generate_strategy_from_performance(self, wins, games, avg_reward):
            win_rate = (wins / max(1, games)) * 100
            if win_rate > 70:
                return f"Dominant strategy: {win_rate:.1f}% win rate with {avg_reward:.1f} avg reward"
            elif win_rate > 50:
                return f"Effective strategy: {win_rate:.1f}% win rate, focus on consistency"
            else:
                return f"Developing strategy: {win_rate:.1f}% win rate, need improvement"
        
        def format_user_demo_outcome(self, outcome, reward):
            return f"Mock environment demo analysis: {outcome} with reward {reward:.2f}"
        
        def get_performance_feedback_phrase(self, metric_type, value):
            if metric_type == "demo_quality":
                if value > 80:
                    return "Excellent technique demonstrated"
                elif value > 60:
                    return "Good form shown"
                elif value > 40:
                    return "Average execution"
                else:
                    return "Needs improvement"
            return f"Performance: {value:.1f}"
    
    # Set mock environment
    mock_env = MockEnvironment()
    agent.set_environment(mock_env)
    
    agent.start_new_match("test_pong", env_context=test_env_context)
    
    test_state = np.random.random(14)
    
    # Test some actions and learning with environment integration
    test_rewards = [5.0, -2.0, 1.5, 4.2, -0.5, 6.1, 0.8]
    for i, reward in enumerate(test_rewards):
        action = agent.get_action(test_state)
        agent.learn(reward=reward, next_state=test_state, done=False)
        print(f"Action {i+1}: {action}, Reward: {reward:.2f}")
    
    # Test user demo with environment integration
    demo_success = agent.record_user_demo({
        'state': test_state.tolist(),
        'action': 1,
        'reward': 1.5,
        'source': 'user_action',
        'outcome': 'hit'
    })
    
    print(f"Demo recording successful: {demo_success}")
    
    # Test another demo with different outcome
    demo_success2 = agent.record_user_demo({
        'state': test_state.tolist(),
        'action': 0,
        'reward': -0.3,
        'source': 'user_action',
        'outcome': 'miss'
    })
    
    print(f"Second demo recording successful: {demo_success2}")
    
    # Show detailed analysis
    print("\n" + "="*70)
    print(agent.get_detailed_knowledge_analysis())
    
    # Show stats
    print("\n" + "="*70)
    stats = agent.get_stats()
    print("📊 AGENT STATS SUMMARY:")
    print(f"   🎮 Games played: {stats['games_played']}")
    print(f"   🏆 Win rate: {stats['win_rate']}%")
    print(f"   🧠 Training steps: {stats['training_steps']}")
    print(f"   🧩 Symbolic decisions: {stats['symbolic_decisions_made']}")
    print(f"   🤖 Neural decisions: {stats['neural_decisions_made']}")
    print(f"   📈 Knowledge effectiveness: {stats['knowledge_effectiveness']}")
    print(f"   🔧 Gamma: {stats['gamma']} ({stats['gamma_source']})")
    print(f"   🏗️ Environment integrated: {stats['environment_integrated']}")
    print(f"   ⚙️ Modular behavior active: {stats['modular_behavior_active']}")
    
    # End match
    final_stats = agent.end_match("Agent Byte", {'player': 15, 'agent_byte': 21})
    
    # Save everything
    save_success = agent.save_brain()
    
    print(f"\n✅ Enhanced Modular Agent Byte v1.2 test complete!")
    print(f"🧩 Symbolic decisions: {agent.symbolic_decisions_made}")
    print(f"🧠 Neural decisions: {agent.neural_decisions_made}")
    print(f"📊 Knowledge effectiveness: {agent.knowledge_effectiveness:.3f}")
    print(f"🔧 Gamma adapted: {agent.default_gamma:.3f} → {agent.gamma:.3f} ({agent.gamma_source})")
    print(f"⚙️ Learning parameters adapted: {agent.learning_parameters_adapted}")
    print(f"🏗️ Environment integration: {'Active' if agent.env else 'Inactive'}")
    print(f"🎯 Modular behavior: {'Working' if hasattr(agent.env, 'interpret_reward') else 'Not available'}")
    print(f"💾 Brain save successful: {save_success}")
    
    if final_stats:
        print(f"📋 Final match stats keys: {len(final_stats)} metrics recorded")
        print(f"🔍 Environment integration recorded: {final_stats.get('environment_integrated', False)}")
        print(f"🎮 Modular behavior active: {final_stats.get('modular_behavior_active', False)}")