# app.py - Modular Coordinator with Adaptive Learning + Knowledge System Integration
import time
import threading
import random
import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import datetime

# Import our modular components
from agent_byte import AgentByte, MatchLogger
from pong_environment import PongEnvironment

class ModularPongGame:
    """Modular Pong Game with Adaptive Learning + Knowledge System Integration"""
    
    def __init__(self):
        print("🎮 Initializing Modular Pong Game with Adaptive Learning + Knowledge System...")
        
        # Create environment with user action tracking
        self.env = PongEnvironment()
        
        # Create match logger
        self.logger = MatchLogger('adaptive_knowledge_enhanced_agent_byte_logs.json')
        
        # Create agent with enhanced adaptive learning + knowledge system
        self.agent = AgentByte(state_size=14, action_size=3, logger=self.logger)
        
        # Game control
        self.running = False
        
        # User demo learning control
        self.demo_learning_enabled = True
        self.demo_recording_active = True
        
        # Knowledge system control
        self.knowledge_system_enabled = True
        
        # NEW: Adaptive learning control
        self.adaptive_learning_enabled = True
        self.adaptive_learning_history = []
        
        # Auto-save
        self.frames_since_save = 0
        self.auto_save_interval = 3600
        
        # Try to load existing brain
        brain_loaded = self.agent.load_brain('adaptive_knowledge_enhanced_agent_byte_v1_2.json')
        if not brain_loaded:
            print("🆕 Starting with fresh Agent Byte brain")
        
        print("✅ Modular Game initialization complete!")
        print("🎯 ENHANCED: Adaptive Learning + Knowledge System + User Demo Learning integrated")
        print("🔧 ADAPTIVE LEARNING SYSTEM:")
        print(f"   Environment-specific parameter optimization")
        print(f"   Automatic gamma adaptation based on game characteristics")
        print(f"   Learning parameter tracking and effectiveness analysis")
        print("🧩 KNOWLEDGE SYSTEM:")
        print(f"   Intelligent strategy selection based on game situation")
        print(f"   Environmental context awareness and failure avoidance")
        print(f"   Strategy performance tracking and optimization")
        print("👤 USER DEMO LEARNING:")
        print(f"   Demo learning weight: {self.agent.demo_learning_weight}")
        print(f"   Demo replay ratio: {self.agent.demo_replay_ratio}")
        print(f"   Demo buffer size: {len(self.agent.user_demo_buffer)}")
        print(f"📊 Match logging: {len(self.logger.all_matches)} previous matches loaded")
    
    def reset_game(self):
        """Reset game for new match with adaptive learning + environmental context integration"""
        self.env.reset_game()
        
        # Get environmental context from the environment
        env_context = self.env.get_env_context()
        
        # Start new match with environmental context and adaptive learning
        self.agent.start_new_match("pong", env_context=env_context)
        
        # Record adaptive learning details for UI
        if self.agent.learning_parameters_adapted:
            adaptation_info = {
                'timestamp': time.time(),
                'environment': 'pong',
                'default_gamma': self.agent.default_gamma,
                'adapted_gamma': self.agent.gamma,
                'gamma_source': self.agent.gamma_source,
                'rationale': self.agent.environment_learning_metadata.get('gamma_rationale', 'Environment-specific optimization')
            }
            self.adaptive_learning_history.append(adaptation_info)
        
        print(f"🆕 Game reset with adaptive learning + knowledge system and environmental context loaded")
        print(f"   🎯 Agent Byte now understands: {env_context['objective']['primary']}")
        print(f"   📚 Pre-loaded with {len(env_context.get('strategic_concepts', {}).get('core_skills', []))} core skills")
        print(f"   🔧 Adaptive learning: Gamma={self.agent.gamma:.3f} ({self.agent.gamma_source})")
        print(f"   🧩 Knowledge system: Ready for intelligent decision making")
    
    def update_game(self):
        """Game update with adaptive learning + knowledge system integration"""
        # Get current state
        current_state = self.env.create_state()
        
        # Get agent action (now with adaptive learning + knowledge system)
        agent_action = self.agent.get_action(current_state)
        
        # Execute step in environment
        next_state, reward, game_ended = self.env.step(agent_action)
        
        # Check for user demonstration to record
        if self.demo_learning_enabled and self.demo_recording_active:
            try:
                user_demo = self.env.evaluate_user_action_outcome()
                if user_demo:
                    success = self.agent.record_user_demo(user_demo)
                    if success:
                        print(f"🎓 User demo learned: {user_demo['outcome']} (reward: {user_demo['reward']:.2f})")
            except Exception as e:
                print(f"❌ Error processing user demo: {e}")
        
        # Agent learns from experience (with adaptive learning + knowledge system effectiveness tracking)
        self.agent.learn(reward=reward, next_state=next_state, done=game_ended)
        
        # Game end processing
        if game_ended:
            if self.env.winner == "Agent Byte":
                final_reward = 10.0
                print(f"🏆 Agent Byte wins! Total wins: {self.agent.wins + 1}/{self.agent.games_played + 1}")
                
                # Balanced learning iterations
                for _ in range(2):
                    self.agent.learn(reward=5.0, next_state=next_state, done=True)
            else:
                final_reward = -10.0
                print(f"😞 Agent Byte loses. Wins: {self.agent.wins}/{self.agent.games_played + 1}")
                print(f"👤 User played well! {self.env.user_hit_count} successful hits this game")
            
            # Final learning with balanced reward
            self.agent.learn(reward=final_reward, next_state=next_state, done=True)
            
            # End match with comprehensive stats including adaptive learning
            try:
                final_scores = {'player': self.env.player_score, 'agent_byte': self.env.ai_score}
                pong_stats = self.env.get_pong_stats()
                
                # Add knowledge system stats
                pong_stats.update({
                    'user_demos_recorded_this_match': self.agent.user_demos_recorded,
                    'user_demos_used_this_match': self.agent.user_demos_processed,
                    'demo_learning_enabled': self.demo_learning_enabled,
                    'demo_recording_active': self.demo_recording_active,
                    'knowledge_system_enabled': self.knowledge_system_enabled,
                    'symbolic_decisions_made': self.agent.symbolic_decisions_made,
                    'neural_decisions_made': self.agent.neural_decisions_made,
                    'knowledge_effectiveness': self.agent.knowledge_effectiveness,
                    # NEW: Adaptive learning stats
                    'adaptive_learning_enabled': self.adaptive_learning_enabled,
                    'gamma_used': self.agent.gamma,
                    'gamma_source': self.agent.gamma_source,
                    'default_gamma': self.agent.default_gamma,
                    'learning_parameters_adapted': self.agent.learning_parameters_adapted,
                    'environment_learning_metadata': self.agent.environment_learning_metadata
                })
                
                self.agent.end_match(self.env.winner, final_scores, pong_stats)
            except Exception as e:
                print(f"❌ Error ending match: {e}")
                self.agent.end_match(self.env.winner or "Unknown", {'player': 0, 'agent_byte': 0})
            
            # Performance analysis
            win_rate = self.agent.wins / self.agent.games_played if self.agent.games_played > 0 else 0
            demo_effectiveness = self.agent._calculate_demo_effectiveness()
            strategy_performance = self.agent.symbolic_decision_maker.get_strategy_performance_summary()
            
            self.running = False
            self.agent.save_brain('adaptive_knowledge_enhanced_agent_byte_v1_2.json')
            
            print(f"📊 Game {self.agent.games_played} Complete:")
            print(f"   🎯 Win rate: {win_rate*100:.1f}%")
            print(f"   🎯 Target updates: {self.agent.target_updates}")
            print(f"   💰 Match reward: {self.agent.match_reward:.1f}")
            print(f"   👤 User demos this match: {self.agent.user_demos_recorded}")
            print(f"   🎓 Demo effectiveness: {demo_effectiveness:.2f}")
            print(f"   🧩 Knowledge decisions: {self.agent.symbolic_decisions_made} symbolic, {self.agent.neural_decisions_made} neural")
            print(f"   📊 Knowledge effectiveness: {self.agent.knowledge_effectiveness:.3f}")
            print(f"   🔧 Adaptive learning: Gamma={self.agent.gamma:.3f} ({self.agent.gamma_source})")
            print(f"   📈 Exploration: {self.agent.exploration_rate:.3f}")
            if strategy_performance:
                print(f"   🎯 Strategy performance: {strategy_performance}")
        
        # Auto-save (less frequent for mobile)
        self.frames_since_save += 1
        if self.frames_since_save >= self.auto_save_interval:
            self.agent.save_brain('adaptive_knowledge_enhanced_agent_byte_v1_2.json')
            self.frames_since_save = 0
    
    def move_player_paddle(self, direction):
        """Move player paddle and record action for learning"""
        # This now automatically records the user action in the environment
        self.env.move_player_paddle(direction)
    
    def set_ball_speed(self, speed):
        """Set ball speed"""
        self.env.set_ball_speed(speed)
    
    def get_game_state(self):
        """Return game state for web interface with adaptive learning + knowledge system stats"""
        game_state = self.env.get_game_state()
        
        # Add agent stats
        agent_stats = self.agent.get_stats()
        
        # Add Pong-specific stats
        pong_stats = self.env.get_pong_stats()
        agent_stats.update(pong_stats)
        
        # Add knowledge system specific stats
        agent_stats.update({
            'demo_learning_enabled': self.demo_learning_enabled,
            'demo_recording_active': self.demo_recording_active,
            'knowledge_system_enabled': self.knowledge_system_enabled,
            'user_demo_buffer_usage': len(self.agent.user_demo_buffer),
            'demo_learning_effectiveness': self.agent._calculate_demo_effectiveness(),
            # NEW: Adaptive learning stats
            'adaptive_learning_enabled': self.adaptive_learning_enabled,
            'adaptive_learning_history': self.adaptive_learning_history[-10:],  # Last 10 adaptations
            'current_gamma_info': {
                'value': self.agent.gamma,
                'source': self.agent.gamma_source,
                'default': self.agent.default_gamma,
                'adapted': self.agent.learning_parameters_adapted,
                'environment_metadata': self.agent.environment_learning_metadata
            }
        })
        
        game_state['ai_stats'] = agent_stats
        
        return game_state
    
    def toggle_demo_learning(self):
        """Toggle user demonstration learning on/off"""
        self.demo_learning_enabled = not self.demo_learning_enabled
        print(f"👤 Demo learning: {'ENABLED' if self.demo_learning_enabled else 'DISABLED'}")
        return self.demo_learning_enabled
    
    def toggle_demo_recording(self):
        """Toggle user action recording on/off"""
        self.demo_recording_active = not self.demo_recording_active
        print(f"📹 Demo recording: {'ACTIVE' if self.demo_recording_active else 'PAUSED'}")
        return self.demo_recording_active
    
    def toggle_knowledge_system(self):
        """Toggle knowledge system on/off"""
        self.knowledge_system_enabled = not self.knowledge_system_enabled
        print(f"🧩 Knowledge system: {'ENABLED' if self.knowledge_system_enabled else 'DISABLED'}")
        return self.knowledge_system_enabled
    
    # NEW: Adaptive learning controls
    def toggle_adaptive_learning(self):
        """Toggle adaptive learning system on/off"""
        self.adaptive_learning_enabled = not self.adaptive_learning_enabled
        self.agent.dual_brain.brain.adaptive_learning_enabled = self.adaptive_learning_enabled
        print(f"🔧 Adaptive learning: {'ENABLED' if self.adaptive_learning_enabled else 'DISABLED'}")
        return self.adaptive_learning_enabled
    
    def get_adaptive_learning_analysis(self):
        """Get detailed adaptive learning analysis"""
        if hasattr(self.agent, 'dual_brain'):
            return self.agent.dual_brain.knowledge.get_adaptive_learning_analysis("pong")
        return "Adaptive learning analysis not available"
    
    def get_gamma_comparison_analysis(self):
        """Get analysis comparing different gamma values used"""
        if not self.adaptive_learning_history:
            return "No adaptive learning history available"
        
        analysis = "🔧 GAMMA ADAPTATION HISTORY:\n"
        for i, adaptation in enumerate(self.adaptive_learning_history[-5:], 1):
            analysis += f"   {i}. {adaptation['default_gamma']:.3f} → {adaptation['adapted_gamma']:.3f} ({adaptation['gamma_source']})\n"
            analysis += f"      Rationale: {adaptation['rationale']}\n"
        
        return analysis
    
    def adjust_demo_parameters(self, weight=None, ratio=None):
        """Adjust demo learning parameters"""
        if weight is not None:
            self.agent.demo_learning_weight = weight
        if ratio is not None:
            self.agent.demo_replay_ratio = ratio
        return {
            'weight': self.agent.demo_learning_weight,
            'ratio': self.agent.demo_replay_ratio
        }
    
    def clear_user_demos(self):
        """Clear user demonstration buffer"""
        cleared_count = len(self.agent.user_demo_buffer)
        self.agent.user_demo_buffer.clear()
        print(f"🗑️ Cleared {cleared_count} user demonstrations")
        return cleared_count
    
    def get_knowledge_analysis(self):
        """Get detailed knowledge system analysis"""
        return self.agent.get_detailed_knowledge_analysis()

# Flask setup with enhanced adaptive learning + knowledge system
app = Flask(__name__)
app.config['SECRET_KEY'] = 'adaptive_knowledge_enhanced_agent_byte_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

print("🚀 Creating Modular Game with Adaptive Learning + Knowledge System...")
game = ModularPongGame()

@app.route('/')
def index():
    return render_template('pong.html')

@socketio.on('start_game')
def start_game():
    game.running = True
    game.reset_game()
    
    def game_loop():
        while game.running:
            game.update_game()
            socketio.emit('game_update', game.get_game_state())
            time.sleep(1/60)  # 60 FPS
    
    threading.Thread(target=game_loop, daemon=True).start()
    emit('game_started')

@socketio.on('stop_game')
def stop_game():
    game.running = False
    emit('game_stopped')

@socketio.on('move_paddle')
def move_paddle(data):
    direction = data.get('direction', 0)
    # This now automatically records user actions for learning
    game.move_player_paddle(direction)

@socketio.on('new_game')
def new_game():
    game.reset_game()
    emit('new_game_started')

# Ball speed controls
@socketio.on('increase_ball_speed')
def increase_ball_speed():
    new_speed = min(20, game.env.ball_speed + 2)
    game.set_ball_speed(new_speed)
    emit('ball_speed_changed', {'speed': new_speed, 'message': f'Ball speed: {new_speed}'})

@socketio.on('decrease_ball_speed')
def decrease_ball_speed():
    new_speed = max(1, game.env.ball_speed - 2)
    game.set_ball_speed(new_speed)
    emit('ball_speed_changed', {'speed': new_speed, 'message': f'Ball speed: {new_speed}'})

@socketio.on('reset_ball_speed')
def reset_ball_speed():
    game.set_ball_speed(game.env.default_ball_speed)
    emit('ball_speed_changed', {'speed': game.env.default_ball_speed, 'message': f'Ball speed reset: {game.env.default_ball_speed}'})

# Exploration controls
@socketio.on('set_exploration_high')
def set_exploration_high():
    game.agent.exploration_rate = 0.8
    emit('exploration_changed', {'rate': game.agent.exploration_rate, 'message': 'Exploration set to HIGH (80%)'})

@socketio.on('set_exploration_medium')
def set_exploration_medium():
    game.agent.exploration_rate = 0.5
    emit('exploration_changed', {'rate': game.agent.exploration_rate, 'message': 'Exploration set to MEDIUM (50%)'})

@socketio.on('set_exploration_low')
def set_exploration_low():
    game.agent.exploration_rate = 0.2
    emit('exploration_changed', {'rate': game.agent.exploration_rate, 'message': 'Exploration set to LOW (20%)'})

# User Demo Learning Controls
@socketio.on('toggle_demo_learning')
def toggle_demo_learning():
    enabled = game.toggle_demo_learning()
    status = "ENABLED" if enabled else "DISABLED"
    emit('demo_learning_toggled', {
        'enabled': enabled, 
        'message': f'👤 User demo learning: {status}'
    })

@socketio.on('toggle_demo_recording')
def toggle_demo_recording():
    active = game.toggle_demo_recording()
    status = "ACTIVE" if active else "PAUSED"
    emit('demo_recording_toggled', {
        'active': active, 
        'message': f'📹 Demo recording: {status}'
    })

@socketio.on('adjust_demo_weight')
def adjust_demo_weight(data):
    weight = data.get('weight', 0.3)
    params = game.adjust_demo_parameters(weight=weight)
    emit('demo_params_changed', {
        'params': params,
        'message': f'👤 Demo learning weight: {params["weight"]:.2f}'
    })

@socketio.on('adjust_demo_ratio')
def adjust_demo_ratio(data):
    ratio = data.get('ratio', 0.25)
    params = game.adjust_demo_parameters(ratio=ratio)
    emit('demo_params_changed', {
        'params': params,
        'message': f'👤 Demo replay ratio: {params["ratio"]:.2f}'
    })

@socketio.on('clear_user_demos')
def clear_user_demos():
    cleared_count = game.clear_user_demos()
    emit('user_demos_cleared', {
        'cleared_count': cleared_count,
        'message': f'🗑️ Cleared {cleared_count} user demonstrations'
    })

@socketio.on('get_demo_stats')
def get_demo_stats():
    stats = {
        'demo_buffer_size': len(game.agent.user_demo_buffer),
        'demos_recorded_this_match': game.agent.user_demos_recorded,
        'demos_used_this_match': game.agent.user_demos_processed,
        'demo_learning_weight': game.agent.demo_learning_weight,
        'demo_replay_ratio': game.agent.demo_replay_ratio,
        'demo_learning_enabled': game.demo_learning_enabled,
        'demo_recording_active': game.demo_recording_active,
        'demo_effectiveness': game.agent._calculate_demo_effectiveness()
    }
    emit('demo_stats_update', stats)

# Knowledge System Controls
@socketio.on('toggle_knowledge_system')
def toggle_knowledge_system():
    enabled = game.toggle_knowledge_system()
    status = "ENABLED" if enabled else "DISABLED"
    emit('knowledge_system_toggled', {
        'enabled': enabled,
        'message': f'🧩 Knowledge system: {status}'
    })

@socketio.on('get_knowledge_analysis')
def get_knowledge_analysis():
    analysis = game.get_knowledge_analysis()
    emit('knowledge_analysis_update', {
        'analysis': analysis,
        'message': 'Knowledge system analysis updated'
    })

@socketio.on('get_strategy_performance')
def get_strategy_performance():
    performance = game.agent.symbolic_decision_maker.get_strategy_performance_summary()
    emit('strategy_performance_update', {
        'performance': performance,
        'message': 'Strategy performance data updated'
    })

# NEW: Adaptive Learning Controls
@socketio.on('toggle_adaptive_learning')
def toggle_adaptive_learning():
    enabled = game.toggle_adaptive_learning()
    status = "ENABLED" if enabled else "DISABLED"
    emit('adaptive_learning_toggled', {
        'enabled': enabled,
        'message': f'🔧 Adaptive learning: {status}'
    })

@socketio.on('get_adaptive_learning_analysis')
def get_adaptive_learning_analysis():
    analysis = game.get_adaptive_learning_analysis()
    emit('adaptive_learning_analysis_update', {
        'analysis': analysis,
        'message': 'Adaptive learning analysis updated'
    })

@socketio.on('get_gamma_comparison')
def get_gamma_comparison():
    comparison = game.get_gamma_comparison_analysis()
    emit('gamma_comparison_update', {
        'comparison': comparison,
        'message': 'Gamma comparison analysis updated'
    })

@socketio.on('get_current_gamma_info')
def get_current_gamma_info():
    gamma_info = {
        'current_gamma': game.agent.gamma,
        'default_gamma': game.agent.default_gamma,
        'gamma_source': game.agent.gamma_source,
        'learning_parameters_adapted': game.agent.learning_parameters_adapted,
        'environment_metadata': game.agent.environment_learning_metadata
    }
    emit('gamma_info_update', gamma_info)

# Agent controls
@socketio.on('save_ai_brain')
def save_ai_brain():
    game.agent.save_brain('adaptive_knowledge_enhanced_agent_byte_v1_2.json')
    stats = game.agent.get_stats()
    emit('ai_brain_saved', {
        'message': f'🧠 Agent Byte v1.2 Adaptive Learning saved! Gamma: {stats.get("gamma", 0):.3f}, Knowledge effectiveness: {stats.get("knowledge_effectiveness", 0):.3f}'
    })

@socketio.on('create_ai_checkpoint')
def create_ai_checkpoint():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f'adaptive_knowledge_enhanced_checkpoint_{timestamp}.json'
    game.agent.save_brain(checkpoint_filename)
    emit('ai_checkpoint_created', {
        'message': f'📸 Adaptive Knowledge Enhanced Checkpoint saved: {checkpoint_filename}'
    })

@socketio.on('smart_reset_ai')
def smart_reset_ai():
    game.agent.exploration_rate = 0.7
    if len(game.agent.experience_buffer) > 50:
        good_experiences = list(game.agent.experience_buffer)[:25]
        game.agent.experience_buffer.clear()
        game.agent.experience_buffer.extend(good_experiences)
    
    # Keep user demos and reset knowledge system tracking for fresh start
    demo_count = len(game.agent.user_demo_buffer)
    game.agent.symbolic_decisions_made = 0
    game.agent.neural_decisions_made = 0
    game.agent.symbolic_decision_maker.decision_history = []
    
    emit('ai_smart_reset', {
        'message': f'🧠 Smart reset applied! Kept {demo_count} demos, reset knowledge tracking. Gamma: {game.agent.gamma:.3f}'
    })

@socketio.on('clear_bad_experiences')
def clear_bad_experiences():
    if len(game.agent.experience_buffer) > 50:
        good_experiences = list(game.agent.experience_buffer)[:25]
        game.agent.experience_buffer.clear()
        game.agent.experience_buffer.extend(good_experiences)
        emit('experiences_cleared', {'message': '🗑️ Cleared recent bad experiences!'})
    else:
        game.agent.experience_buffer.clear()
        emit('experiences_cleared', {'message': '🗑️ Cleared all experiences!'})

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("\n" + "="*80)
    print("🧠 ADAPTIVE LEARNING + KNOWLEDGE ENHANCED AGENT BYTE v1.2")
    print("="*80)
    print("🚀 Server starting on PORT 5001...")
    print("📱 Open browser to: http://localhost:5001")
    print("🤖 Enhanced Agent Byte: Now with ADAPTIVE LEARNING + intelligent symbolic knowledge application!")
    print("🔧 ADAPTIVE LEARNING FEATURES:")
    print("   ✅ Environment-specific parameter optimization (Gamma, Learning Rate, Exploration)")
    print("   ✅ Automatic adaptation based on environment characteristics")
    print("   ✅ Learning parameter effectiveness tracking")
    print("   ✅ Historical adaptation analysis and comparison")
    print("   ✅ Real-time parameter monitoring and adjustment")
    print("🧩 KNOWLEDGE SYSTEM FEATURES:")
    print("   ✅ Intelligent strategy selection based on game situation")
    print("   ✅ Environmental context awareness and rule understanding")
    print("   ✅ Failure pattern avoidance and success pattern application")
    print("   ✅ Strategy performance tracking and optimization")
    print("   ✅ Real-time decision reasoning and explanation")
    print("🎓 USER DEMO LEARNING FEATURES:")
    print("   ✅ Automatic user action recording")
    print("   ✅ Real-time demonstration evaluation")
    print("   ✅ Quality-based demo selection")
    print("   ✅ Configurable learning parameters")
    print("   ✅ Demo effectiveness tracking")
    print("🏗️ INTEGRATION POINTS:")
    print("   📹 User paddle movements → Demo recording")
    print("   🎯 User hits/misses → Learning opportunities")
    print("   🧩 Game situations → Intelligent strategy selection")
    print("   📊 Strategy outcomes → Performance optimization")
    print("   🔧 Environment type → Automatic parameter adaptation")
    print("⚡ ADAPTIVE LEARNING PARAMETERS:")
    print(f"   Pong Gamma: {game.env.recommended_gamma} (short-term focus)")
    print(f"   Default Gamma: {game.agent.default_gamma} (fallback)")
    print(f"   Current Gamma: {game.agent.gamma} ({game.agent.gamma_source})")
    print(f"   Learning Rate: {game.agent.learning_rate}")
    print(f"   Adaptive system: {'ENABLED' if game.adaptive_learning_enabled else 'DISABLED'}")
    print("🎮 HOW ADAPTIVE LEARNING WORKS:")
    print("   1. Environment provides recommended learning parameters")
    print("   2. Agent automatically adapts Gamma, Learning Rate, Exploration")
    print("   3. Parameters are optimized for environment characteristics")
    print("   4. Performance with different parameters is tracked")
    print("   5. Best performing parameters are remembered for future sessions")
    print("   6. Real-time monitoring shows adaptation effectiveness")
    print("🔧 Ball speed controls + Exploration controls + Demo controls + Knowledge controls + Adaptive learning controls")
    print("🏆 First to 21 points wins!")
    print("="*80)
    
    # Show current status
    print(f"🔧 Adaptive Learning Status:")
    print(f"   System: {'ACTIVE' if game.adaptive_learning_enabled else 'INACTIVE'}")
    print(f"   Current Gamma: {game.agent.gamma:.3f} ({game.agent.gamma_source})")
    print(f"   Default Gamma: {game.agent.default_gamma:.3f}")
    print(f"   Parameters Adapted: {'YES' if game.agent.learning_parameters_adapted else 'NO'}")
    if game.agent.environment_learning_metadata:
        print(f"   Adaptation Rationale: {game.agent.environment_learning_metadata.get('gamma_rationale', 'Not available')}")
    
    print(f"🧩 Knowledge System Status:")
    print(f"   System: {'ACTIVE' if game.knowledge_system_enabled else 'INACTIVE'}")
    print(f"   Learning: {'ON' if game.demo_learning_enabled else 'OFF'}")
    print(f"   Recording: {'ACTIVE' if game.demo_recording_active else 'PAUSED'}")
    print(f"   Available demos: {len(game.agent.user_demo_buffer)}")
    
    # Show recent performance
    recent_perf = game.logger.get_recent_performance()
    if recent_perf:
        print(f"📊 Recent Performance (last 10 matches):")
        print(f"   🏆 Win rate: {recent_perf['recent_win_rate']:.1f}%")
        print(f"   💰 Avg reward: {recent_perf['avg_reward_per_match']:.1f}")
        print(f"   👤 Avg user demos: {recent_perf.get('avg_user_demos_per_match', 0):.1f}")
        print(f"   🧩 Avg symbolic decisions: {recent_perf.get('avg_symbolic_decisions_per_match', 0):.1f}")
        print(f"   🔧 Adaptive learning usage: {recent_perf.get('adaptive_learning_usage_rate', 0):.1f}%")
        print(f"   📈 Total matches: {recent_perf['total_matches']}")
        print("="*80)
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
