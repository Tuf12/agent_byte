# dual_brain_system.py - Agent Byte Dual Brain Architecture (Enhanced for Adaptive Learning)
import json
import time
import os
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

class AgentBrain:
    """Core learning engine - environment agnostic with adaptive learning support"""
    
    def __init__(self, brain_file='agent_brain.json'):
        self.brain_file = brain_file
        self.training_steps = 0
        self.learning_rate = 0.001
        self.gamma = 0.99  # Default gamma, will be overridden per environment
        self.epsilon = 0.8
        self.double_dqn_improvements = 0
        self.target_updates = 0
        self.total_loss = 0.0
        self.experience_buffer_size = 0
        self.shared_experience_count = 0
        self.architecture_version = "Agent Byte v1.2 Adaptive Learning + Knowledge System Enhanced"
        self.created_timestamp = time.time()
        self.last_updated = time.time()
        
        # NEW: Adaptive learning tracking
        self.adaptive_learning_enabled = True
        self.environment_adaptations = {}  # Track per-environment adaptations
        self.default_learning_parameters = {
            'gamma': 0.99,
            'learning_rate': 0.001,
            'epsilon_start': 0.8,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.1
        }
        
        # Load existing brain if available
        self.load_brain()
    
    def save_brain(self):
        """Save core learning engine state with adaptive learning metadata"""
        try:
            brain_data = {
                'training_steps': self.training_steps,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'double_dqn_improvements': self.double_dqn_improvements,
                'target_updates': self.target_updates,
                'total_loss': self.total_loss,
                'experience_buffer_size': self.experience_buffer_size,
                'shared_experience_count': self.shared_experience_count,
                'architecture_version': self.architecture_version,
                'created_timestamp': self.created_timestamp,
                'last_updated': time.time(),
                'saved_at': datetime.now().isoformat(),
                # NEW: Adaptive learning metadata
                'adaptive_learning_enabled': self.adaptive_learning_enabled,
                'environment_adaptations': self.environment_adaptations,
                'default_learning_parameters': self.default_learning_parameters
            }
            
            with open(self.brain_file, 'w') as f:
                json.dump(brain_data, f, indent=2)
            
            print(f"🧠 Agent Brain saved: {self.training_steps} steps, {self.target_updates} updates")
            print(f"   🔧 Adaptive learning: {len(self.environment_adaptations)} environment adaptations stored")
            return True
            
        except Exception as e:
            print(f"❌ Error saving brain: {e}")
            return False
    
    def load_brain(self):
        """Load core learning engine state with adaptive learning metadata"""
        try:
            if os.path.exists(self.brain_file):
                with open(self.brain_file, 'r') as f:
                    brain_data = json.load(f)
                
                self.training_steps = brain_data.get('training_steps', 0)
                self.learning_rate = brain_data.get('learning_rate', 0.001)
                self.gamma = brain_data.get('gamma', 0.99)
                self.epsilon = brain_data.get('epsilon', 0.8)
                self.double_dqn_improvements = brain_data.get('double_dqn_improvements', 0)
                self.target_updates = brain_data.get('target_updates', 0)
                self.total_loss = brain_data.get('total_loss', 0.0)
                self.experience_buffer_size = brain_data.get('experience_buffer_size', 0)
                self.shared_experience_count = brain_data.get('shared_experience_count', 0)
                self.created_timestamp = brain_data.get('created_timestamp', time.time())
                
                # NEW: Load adaptive learning metadata
                self.adaptive_learning_enabled = brain_data.get('adaptive_learning_enabled', True)
                self.environment_adaptations = brain_data.get('environment_adaptations', {})
                self.default_learning_parameters = brain_data.get('default_learning_parameters', {
                    'gamma': 0.99,
                    'learning_rate': 0.001,
                    'epsilon_start': 0.8,
                    'epsilon_decay': 0.995,
                    'epsilon_min': 0.1
                })
                
                print(f"🧠 Agent Brain loaded: {self.training_steps} training steps")
                if self.environment_adaptations:
                    print(f"   🔧 Adaptive learning: {len(self.environment_adaptations)} environment adaptations loaded")
                return True
            else:
                print("🆕 No existing brain found, starting fresh")
                return False
                
        except Exception as e:
            print(f"❌ Error loading brain: {e}")
            return False
    
    def record_environment_adaptation(self, environment_name, adaptation_data):
        """Record successful learning parameter adaptations for future use"""
        try:
            if environment_name not in self.environment_adaptations:
                self.environment_adaptations[environment_name] = {
                    'adaptations_count': 0,
                    'successful_adaptations': [],
                    'performance_metrics': {},
                    'last_used': None
                }
            
            adaptation_record = {
                'timestamp': time.time(),
                'parameters': adaptation_data.get('parameters', {}),
                'performance_improvement': adaptation_data.get('performance_improvement', 0),
                'match_duration': adaptation_data.get('match_duration', 0),
                'final_reward': adaptation_data.get('final_reward', 0),
                'win_result': adaptation_data.get('win_result', False)
            }
            
            self.environment_adaptations[environment_name]['successful_adaptations'].append(adaptation_record)
            self.environment_adaptations[environment_name]['adaptations_count'] += 1
            self.environment_adaptations[environment_name]['last_used'] = time.time()
            
            # Keep only recent adaptations (last 10)
            if len(self.environment_adaptations[environment_name]['successful_adaptations']) > 10:
                self.environment_adaptations[environment_name]['successful_adaptations'] = \
                    self.environment_adaptations[environment_name]['successful_adaptations'][-10:]
            
            print(f"🔧 Recorded adaptation for {environment_name}: {adaptation_data.get('parameters', {})}")
            return True
            
        except Exception as e:
            print(f"❌ Error recording environment adaptation: {e}")
            return False
    
    def get_recommended_parameters(self, environment_name):
        """Get recommended learning parameters based on past adaptations"""
        if not self.adaptive_learning_enabled:
            return self.default_learning_parameters.copy()
        
        if environment_name in self.environment_adaptations:
            adaptations = self.environment_adaptations[environment_name]['successful_adaptations']
            if adaptations:
                # Get the most successful recent adaptation
                best_adaptation = max(adaptations, key=lambda x: x.get('performance_improvement', 0))
                recommended = self.default_learning_parameters.copy()
                recommended.update(best_adaptation.get('parameters', {}))
                
                print(f"🎯 Using learned parameters for {environment_name}: {best_adaptation.get('parameters', {})}")
                return recommended
        
        print(f"🔧 Using default parameters for {environment_name} (no adaptations available)")
        return self.default_learning_parameters.copy()

class AgentKnowledge:
    """Symbolic knowledge and environment-specific understanding (Enhanced for Adaptive Learning)"""
    
    def __init__(self, knowledge_file='agent_knowledge.json'):
        self.knowledge_file = knowledge_file
        self.knowledge = {
            "categories": {
                "games": {},
                "activities": {},
                "general_principles": []
            },
            "metadata": {
                "version": "1.2.1",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "knowledge_system_enhanced": True,
                "adaptive_learning_enhanced": True
            }
        }
        
        # Load existing knowledge
        self.load_knowledge()
    
    def load_knowledge(self):
        """Load symbolic knowledge from file"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r') as f:
                    self.knowledge = json.load(f)
                
                # Ensure metadata includes adaptive learning enhancement
                if 'metadata' not in self.knowledge:
                    self.knowledge['metadata'] = {}
                self.knowledge['metadata']['knowledge_system_enhanced'] = True
                self.knowledge['metadata']['adaptive_learning_enhanced'] = True
                self.knowledge['metadata']['version'] = "1.2.1"
                
                print(f"🧩 Agent Knowledge loaded: {len(self.knowledge.get('categories', {}).get('games', {}))} games known")
                return True
            else:
                print("🆕 No existing knowledge found, starting fresh")
                return False
        except Exception as e:
            print(f"❌ Error loading knowledge: {e}")
            return False
    
    def save_knowledge(self):
        """Save symbolic knowledge to file"""
        try:
            self.knowledge['metadata']['last_updated'] = datetime.now().isoformat()
            self.knowledge['metadata']['knowledge_system_enhanced'] = True
            self.knowledge['metadata']['adaptive_learning_enhanced'] = True
            
            with open(self.knowledge_file, 'w') as f:
                json.dump(self.knowledge, f, indent=2)
            
            print(f"🧩 Agent Knowledge saved")
            return True
        except Exception as e:
            print(f"❌ Error saving knowledge: {e}")
            return False
    
    def load_app_context(self, app_name, category="games", env_context=None):
        """Load symbolic data for specific environment with optional environmental context and adaptive learning"""
        try:
            app_context = self.knowledge.get('categories', {}).get(category, {}).get(app_name, None)
            
            if app_context:
                print(f"📖 Loaded existing context for {app_name}:")
                print(f"   Objective: {app_context.get('objective', 'Unknown')}")
                print(f"   Games played: {app_context.get('metrics', {}).get('games_played', 0)}")
                print(f"   Win rate: {self._calculate_win_rate(app_context):.1f}%")
                print(f"   Strategies known: {len(app_context.get('strategies', []))}")
                print(f"   Lessons learned: {len(app_context.get('lessons', []))}")
                
                # Update with new environmental context if provided
                if env_context:
                    updated = self._integrate_env_context(app_context, env_context, app_name, category)
                    if updated:
                        print(f"   🔄 Updated with latest environmental context")
                
                # NEW: Add adaptive learning tracking to existing context
                if 'adaptive_learning_metadata' not in app_context:
                    app_context['adaptive_learning_metadata'] = {
                        'enabled': True,
                        'parameter_adaptations': [],
                        'performance_with_adaptations': {},
                        'environment_specific_optimizations': {}
                    }
                
                return app_context
            else:
                print(f"🆕 No existing context for {app_name}, creating new with environmental context...")
                return self._create_new_app_context(app_name, category, env_context)
                
        except Exception as e:
            print(f"❌ Error loading app context: {e}")
            return None
    
    def _integrate_env_context(self, app_context, env_context, app_name, category):
        """Integrate environmental context into existing app context with adaptive learning support"""
        try:
            updated = False
            
            # Update objective if provided by environment
            if 'objective' in env_context:
                env_objective = env_context['objective']
                if isinstance(env_objective, dict):
                    new_objective = env_objective.get('primary', app_context.get('objective', 'Unknown'))
                else:
                    new_objective = str(env_objective)
                
                if app_context.get('objective') != new_objective:
                    app_context['objective'] = new_objective
                    updated = True
            
            # Update or add environmental rules
            if 'rules' in env_context:
                if 'environmental_rules' not in app_context:
                    app_context['environmental_rules'] = {}
                app_context['environmental_rules'].update(env_context['rules'])
                updated = True
            
            # NEW: Integrate adaptive learning parameters
            if 'learning_parameters' in env_context:
                learning_params = env_context['learning_parameters']
                if 'adaptive_learning_metadata' not in app_context:
                    app_context['adaptive_learning_metadata'] = {
                        'enabled': True,
                        'parameter_adaptations': [],
                        'performance_with_adaptations': {},
                        'environment_specific_optimizations': {}
                    }
                
                # Store environment-recommended parameters
                app_context['adaptive_learning_metadata']['environment_recommendations'] = {
                    'gamma': learning_params.get('recommended_gamma'),
                    'learning_rate': learning_params.get('recommended_learning_rate'),
                    'exploration': learning_params.get('recommended_exploration', {}),
                    'temporal_characteristics': learning_params.get('temporal_characteristics', {}),
                    'last_updated': time.time()
                }
                updated = True
                print(f"   🔧 Adaptive learning parameters integrated")
            
            # Add strategic concepts as initial strategies if not already present
            if 'strategic_concepts' in env_context:
                strategic_info = env_context['strategic_concepts']
                existing_strategies = app_context.get('strategies', [])
                
                # Add core skills as strategies
                for skill in strategic_info.get('core_skills', []):
                    strategy = f"Master {skill.lower()}"
                    if strategy not in existing_strategies:
                        existing_strategies.append(strategy)
                        updated = True
                
                # Add tactical approaches
                for approach in strategic_info.get('tactical_approaches', []):
                    if approach not in existing_strategies:
                        existing_strategies.append(approach)
                        updated = True
                
                app_context['strategies'] = existing_strategies
            
            # Add learning recommendations as lessons if not present
            if 'learning_recommendations' in env_context:
                learning_info = env_context['learning_recommendations']
                existing_lessons = app_context.get('lessons', [])
                
                for phase, recommendations in learning_info.items():
                    for rec in recommendations:
                        lesson = f"[{phase.replace('_', ' ').title()}] {rec}"
                        if lesson not in existing_lessons:
                            existing_lessons.append(lesson)
                            updated = True
                
                app_context['lessons'] = existing_lessons
            
            # Store environmental metadata
            if 'context_metadata' not in app_context:
                app_context['context_metadata'] = {}
            
            app_context['context_metadata'].update({
                'last_env_context_update': env_context.get('context_metadata', {}).get('generated_at', time.time()),
                'environment_version': env_context.get('context_metadata', {}).get('environment_version', 'unknown'),
                'context_integration_timestamp': time.time(),
                'knowledge_system_enhanced': True,
                'adaptive_learning_enhanced': True
            })
            
            # Store full environmental context for reference
            app_context['environment_context'] = env_context
            updated = True
            
            if updated:
                self.knowledge['categories'][category][app_name] = app_context
                self.save_knowledge()
                print(f"🔄 Environmental context integrated for {app_name}")
            
            return updated
            
        except Exception as e:
            print(f"❌ Error integrating environmental context: {e}")
            return False
    
    def _create_new_app_context(self, app_name, category="games", env_context=None):
        """Create new context for unknown environment with environmental context integration and adaptive learning"""
        if category == "games":
            # Start with basic structure
            new_context = {
                "objective": "Learn through gameplay",
                "rules": {},
                "strategies": [],
                "metrics": {
                    "games_played": 0,
                    "wins": 0,
                    "losses": 0,
                    "demo_effectiveness": 0.0,
                    "best_win_streak": 0,
                    "current_streak": 0,
                    "knowledge_system_decisions": 0,
                    "symbolic_effectiveness": 0.0
                },
                "lessons": [],
                "symbolic_reflections": [],
                "performance_history": [],
                "knowledge_system_metadata": {
                    "enabled": True,
                    "created_with_knowledge_system": True,
                    "version": "1.2.1"
                },
                # NEW: Adaptive learning metadata
                "adaptive_learning_metadata": {
                    "enabled": True,
                    "created_with_adaptive_learning": True,
                    "parameter_adaptations": [],
                    "performance_with_adaptations": {},
                    "environment_specific_optimizations": {},
                    "version": "1.2.1"
                }
            }
            
            # Integrate environmental context if provided
            if env_context:
                print(f"🌟 Creating {app_name} context with environmental knowledge and adaptive learning...")
                
                # Set objective from environment
                if 'objective' in env_context:
                    env_objective = env_context['objective']
                    if isinstance(env_objective, dict):
                        new_context['objective'] = env_objective.get('primary', 'Learn through gameplay')
                        
                        # Add additional objectives as initial lessons
                        if 'secondary' in env_objective:
                            new_context['lessons'].append(f"Secondary goal: {env_objective['secondary']}")
                        if 'win_condition' in env_objective:
                            new_context['lessons'].append(f"Victory condition: {env_objective['win_condition']}")
                    else:
                        new_context['objective'] = str(env_objective)
                
                # Add environmental rules
                if 'rules' in env_context:
                    new_context['environmental_rules'] = env_context['rules']
                
                # NEW: Process adaptive learning parameters
                if 'learning_parameters' in env_context:
                    learning_params = env_context['learning_parameters']
                    new_context['adaptive_learning_metadata']['environment_recommendations'] = {
                        'gamma': learning_params.get('recommended_gamma'),
                        'learning_rate': learning_params.get('recommended_learning_rate'),
                        'exploration': learning_params.get('recommended_exploration', {}),
                        'temporal_characteristics': learning_params.get('temporal_characteristics', {}),
                        'environment_complexity': learning_params.get('environment_complexity', {}),
                        'created_at': time.time()
                    }
                    
                    # Add adaptive learning lesson
                    gamma_rationale = learning_params.get('gamma_rationale', 'Environment-specific optimization')
                    new_context['lessons'].append(f"[Adaptive Learning] {gamma_rationale}")
                    
                    print(f"   🔧 Adaptive learning parameters captured:")
                    print(f"      Recommended Gamma: {learning_params.get('recommended_gamma')}")
                    print(f"      Rationale: {gamma_rationale}")
                
                # Extract strategies from strategic concepts
                if 'strategic_concepts' in env_context:
                    strategic_info = env_context['strategic_concepts']
                    
                    # Add core skills as strategies
                    for skill in strategic_info.get('core_skills', []):
                        new_context['strategies'].append(f"Develop {skill.lower()}")
                    
                    # Add tactical approaches
                    for approach in strategic_info.get('tactical_approaches', []):
                        new_context['strategies'].append(approach)
                    
                    # Add success patterns as lessons
                    for pattern in strategic_info.get('success_patterns', []):
                        new_context['lessons'].append(f"Key insight: {pattern}")
                
                # Extract lessons from learning recommendations
                if 'learning_recommendations' in env_context:
                    learning_info = env_context['learning_recommendations']
                    for phase, recommendations in learning_info.items():
                        for rec in recommendations:
                            new_context['lessons'].append(f"[{phase.replace('_', ' ').title()}] {rec}")
                
                # Add failure patterns as lessons to avoid
                if 'failure_patterns' in env_context:
                    for pattern_name, pattern_info in env_context['failure_patterns'].items():
                        lesson = f"Avoid {pattern_name}: {pattern_info.get('description', 'Unknown failure mode')}"
                        solution = pattern_info.get('solution', 'No solution provided')
                        new_context['lessons'].append(f"{lesson} | Solution: {solution}")
                
                # Store environmental metadata
                new_context['context_metadata'] = {
                    'created_with_env_context': True,
                    'env_context_timestamp': env_context.get('context_metadata', {}).get('generated_at', time.time()),
                    'environment_version': env_context.get('context_metadata', {}).get('environment_version', 'unknown'),
                    'integration_timestamp': time.time(),
                    'knowledge_system_enhanced': True,
                    'adaptive_learning_enhanced': True
                }
                
                # Store full environmental context for reference
                new_context['environment_context'] = env_context
                
                print(f"   📝 Extracted {len(new_context['strategies'])} strategies")
                print(f"   📚 Extracted {len(new_context['lessons'])} lessons")
                print(f"   🎯 Set objective: {new_context['objective']}")
                
        else:
            # Non-game context
            new_context = {
                "purpose": "Unknown activity",
                "symbolic_skills": [],
                "performance_metrics": {},
                "knowledge_system_metadata": {
                    "enabled": True,
                    "created_with_knowledge_system": True,
                    "version": "1.2.1"
                },
                "adaptive_learning_metadata": {
                    "enabled": True,
                    "created_with_adaptive_learning": True,
                    "version": "1.2.1"
                }
            }
            
            if env_context:
                # Adapt environmental context for non-game activities
                if 'objective' in env_context:
                    new_context['purpose'] = str(env_context['objective'])
                
                new_context['environment_context'] = env_context
        
        # Save the new context
        if 'categories' not in self.knowledge:
            self.knowledge['categories'] = {}
        if category not in self.knowledge['categories']:
            self.knowledge['categories'][category] = {}
        
        self.knowledge['categories'][category][app_name] = new_context
        self.save_knowledge()
        
        return new_context
    
    def record_adaptive_learning_result(self, app_name, adaptation_data, category="games"):
        """Record the results of adaptive learning parameter usage"""
        try:
            app_context = self.knowledge.get('categories', {}).get(category, {}).get(app_name)
            if not app_context:
                return False
            
            if 'adaptive_learning_metadata' not in app_context:
                app_context['adaptive_learning_metadata'] = {
                    'enabled': True,
                    'parameter_adaptations': [],
                    'performance_with_adaptations': {},
                    'environment_specific_optimizations': {}
                }
            
            # Record the adaptation result
            adaptation_record = {
                'timestamp': time.time(),
                'parameters_used': adaptation_data.get('parameters_used', {}),
                'match_performance': {
                    'win': adaptation_data.get('win', False),
                    'final_reward': adaptation_data.get('final_reward', 0),
                    'duration': adaptation_data.get('duration', 0),
                    'actions_taken': adaptation_data.get('actions_taken', 0)
                },
                'effectiveness_metrics': {
                    'compared_to_default': adaptation_data.get('performance_improvement', 0),
                    'knowledge_system_synergy': adaptation_data.get('knowledge_effectiveness', 0)
                }
            }
            
            app_context['adaptive_learning_metadata']['parameter_adaptations'].append(adaptation_record)
            
            # Keep only recent adaptations
            if len(app_context['adaptive_learning_metadata']['parameter_adaptations']) > 20:
                app_context['adaptive_learning_metadata']['parameter_adaptations'] = \
                    app_context['adaptive_learning_metadata']['parameter_adaptations'][-20:]
            
            # Update performance tracking
            gamma_used = adaptation_data.get('parameters_used', {}).get('gamma', 'unknown')
            gamma_key = f"gamma_{gamma_used:.3f}" if isinstance(gamma_used, (int, float)) else str(gamma_used)
            
            if gamma_key not in app_context['adaptive_learning_metadata']['performance_with_adaptations']:
                app_context['adaptive_learning_metadata']['performance_with_adaptations'][gamma_key] = []
            
            app_context['adaptive_learning_metadata']['performance_with_adaptations'][gamma_key].append({
                'win': adaptation_data.get('win', False),
                'reward': adaptation_data.get('final_reward', 0),
                'timestamp': time.time()
            })
            
            # Save updated knowledge
            self.knowledge['categories'][category][app_name] = app_context
            self.save_knowledge()
            
            print(f"🔧 Recorded adaptive learning result for {app_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error recording adaptive learning result: {e}")
            return False
    
    def get_adaptive_learning_analysis(self, app_name, category="games"):
        """Get analysis of adaptive learning effectiveness for an environment"""
        try:
            app_context = self.knowledge.get('categories', {}).get(category, {}).get(app_name)
            if not app_context or 'adaptive_learning_metadata' not in app_context:
                return "No adaptive learning data available"
            
            metadata = app_context['adaptive_learning_metadata']
            adaptations = metadata.get('parameter_adaptations', [])
            performance_data = metadata.get('performance_with_adaptations', {})
            
            if not adaptations:
                return "No adaptive learning adaptations recorded yet"
            
            # Analyze recent performance
            recent_adaptations = adaptations[-10:]
            avg_performance = sum(a['match_performance']['final_reward'] for a in recent_adaptations) / len(recent_adaptations)
            win_rate = sum(1 for a in recent_adaptations if a['match_performance']['win']) / len(recent_adaptations) * 100
            
            # Find best performing gamma
            best_gamma = None
            best_gamma_performance = -float('inf')
            
            for gamma_key, results in performance_data.items():
                if results:
                    avg_reward = sum(r['reward'] for r in results) / len(results)
                    if avg_reward > best_gamma_performance:
                        best_gamma_performance = avg_reward
                        best_gamma = gamma_key
            
            analysis = f"""
🔧 ADAPTIVE LEARNING ANALYSIS - {app_name}
═══════════════════════════════════════════════

📊 Recent Performance (last {len(recent_adaptations)} matches):
   Average Reward: {avg_performance:.2f}
   Win Rate: {win_rate:.1f}%
   Total Adaptations Recorded: {len(adaptations)}

🎯 Parameter Performance Analysis:
   Best Performing Gamma: {best_gamma} (avg reward: {best_gamma_performance:.2f})
   
📈 Performance by Parameters:"""

            for gamma_key, results in performance_data.items():
                if results:
                    avg_reward = sum(r['reward'] for r in results) / len(results)
                    win_rate_gamma = sum(1 for r in results if r['win']) / len(results) * 100
                    analysis += f"\n   {gamma_key}: {avg_reward:.2f} avg reward, {win_rate_gamma:.1f}% win rate ({len(results)} matches)"
            
            # Environment recommendations
            env_recs = metadata.get('environment_recommendations', {})
            if env_recs:
                analysis += f"\n\n🌟 Environment Recommendations:"
                analysis += f"\n   Recommended Gamma: {env_recs.get('gamma', 'Not specified')}"
                analysis += f"\n   Learning Rate: {env_recs.get('learning_rate', 'Not specified')}"
                
                temporal = env_recs.get('temporal_characteristics', {})
                if temporal:
                    analysis += f"\n   Match Duration: {temporal.get('match_duration', 'Unknown')}"
                    analysis += f"\n   Feedback Type: {temporal.get('feedback_immediacy', 'Unknown')}"
            
            return analysis.strip()
            
        except Exception as e:
            print(f"❌ Error generating adaptive learning analysis: {e}")
            return f"Error analyzing adaptive learning data: {e}"

    # Keep all other existing methods unchanged
    def record_game_result(self, app_name, win, reward, outcome_summary, category="games"):
        """Log match stats back into correct section (Enhanced for adaptive learning)"""
        try:
            # Ensure context exists
            if app_name not in self.knowledge.get('categories', {}).get(category, {}):
                self.load_app_context(app_name, category)
            
            app_context = self.knowledge['categories'][category][app_name]
            metrics = app_context.get('metrics', {})
            
            # Update basic metrics
            metrics['games_played'] = metrics.get('games_played', 0) + 1
            
            if win:
                metrics['wins'] = metrics.get('wins', 0) + 1
                metrics['current_streak'] = metrics.get('current_streak', 0) + 1
                if metrics['current_streak'] > metrics.get('best_win_streak', 0):
                    metrics['best_win_streak'] = metrics['current_streak']
            else:
                metrics['losses'] = metrics.get('losses', 0) + 1
                metrics['current_streak'] = 0
            
            # Record performance history
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'win': win,
                'reward': reward,
                'outcome_summary': outcome_summary,
                'win_rate_at_time': self._calculate_win_rate(app_context)
            }
            
            if 'performance_history' not in app_context:
                app_context['performance_history'] = []
            
            app_context['performance_history'].append(performance_entry)
            
            # Keep only last 50 games for performance history
            if len(app_context['performance_history']) > 50:
                app_context['performance_history'] = app_context['performance_history'][-50:]
            
            # Update the context
            app_context['metrics'] = metrics
            self.knowledge['categories'][category][app_name] = app_context
            
            # Save knowledge
            self.save_knowledge()
            
            win_rate = self._calculate_win_rate(app_context)
            print(f"📊 {app_name} result recorded: {'WIN' if win else 'LOSS'}")
            print(f"   Total games: {metrics['games_played']}, Win rate: {win_rate:.1f}%")
            print(f"   Current streak: {metrics.get('current_streak', 0)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error recording game result: {e}")
            return False
    
    def reflect_on_performance(self, app_name, category="games"):
        """Generate symbolic reflections based on patterns (Enhanced with adaptive learning insights)"""
        try:
            app_context = self.knowledge.get('categories', {}).get(category, {}).get(app_name)
            if not app_context:
                return "No performance data available for reflection"
            
            metrics = app_context.get('metrics', {})
            history = app_context.get('performance_history', [])
            
            reflections = []
            
            # Analyze win rate trends
            win_rate = self._calculate_win_rate(app_context)
            if win_rate > 70:
                reflections.append(f"{app_name} mastery achieved - consistently strong performance with adaptive learning optimization")
            elif win_rate > 50:
                reflections.append(f"{app_name} competency developing - adaptive learning showing positive impact")
            elif win_rate > 30:
                reflections.append(f"{app_name} learning in progress - adaptive learning parameters may need refinement")
            else:
                reflections.append(f"{app_name} requires strategy adjustment - consider different adaptive learning approach")
            
            # Analyze recent performance if enough history
            if len(history) >= 10:
                recent_wins = sum(1 for h in history[-10:] if h['win'])
                recent_win_rate = (recent_wins / 10) * 100
                
                if recent_win_rate > win_rate + 10:
                    reflections.append("Recent improvement trend detected - adaptive learning optimizations are working")
                elif recent_win_rate < win_rate - 10:
                    reflections.append("Recent performance decline - may need to adjust adaptive learning parameters")
            
            # Analyze streaks
            best_streak = metrics.get('best_win_streak', 0)
            if best_streak >= 5:
                reflections.append(f"Demonstrated ability for sustained success (best streak: {best_streak}) - adaptive learning contributing to consistency")
            elif best_streak >= 3:
                reflections.append("Shows potential for consistency - adaptive learning helping maintain successful patterns")
            
            # Generate lesson insights with adaptive learning context
            if len(history) >= 5:
                high_reward_games = [h for h in history if h.get('reward', 0) > 5]
                if high_reward_games:
                    reflections.append("High-reward outcomes correlate with optimized adaptive learning parameters")
            
            # Add adaptive learning specific insights
            adaptive_metadata = app_context.get('adaptive_learning_metadata', {})
            if adaptive_metadata.get('parameter_adaptations'):
                adaptations_count = len(adaptive_metadata['parameter_adaptations'])
                reflections.append(f"Adaptive learning system has made {adaptations_count} parameter optimizations for this environment")
            
            # Add strategic insights based on app type
            if app_name == "pong":
                if win_rate > 60:
                    reflections.append("Pong reflects mastery of timing and prediction enhanced by adaptive gamma optimization - skills transferable to other environments")
                else:
                    reflections.append("Pong performance suggests adaptive learning parameters need environment-specific tuning")
            
            # Store reflections in knowledge
            if 'symbolic_reflections' not in app_context:
                app_context['symbolic_reflections'] = []
            
            # Add new reflections (avoid duplicates)
            new_reflections = []
            existing = set(app_context['symbolic_reflections'])
            for reflection in reflections:
                if reflection not in existing:
                    new_reflections.append(reflection)
                    app_context['symbolic_reflections'].append(reflection)
            
            # Keep only recent reflections
            if len(app_context['symbolic_reflections']) > 10:
                app_context['symbolic_reflections'] = app_context['symbolic_reflections'][-10:]
            
            # Update knowledge
            self.knowledge['categories'][category][app_name] = app_context
            self.save_knowledge()
            
            if new_reflections:
                print(f"🤔 New reflections on {app_name}:")
                for reflection in new_reflections:
                    print(f"   • {reflection}")
            
            return reflections
            
        except Exception as e:
            print(f"❌ Error reflecting on performance: {e}")
            return []
    
    def _calculate_win_rate(self, app_context):
        """Calculate win rate from metrics"""
        metrics = app_context.get('metrics', {})
        games = metrics.get('games_played', 0)
        wins = metrics.get('wins', 0)
        return (wins / max(1, games)) * 100
    
    def add_lesson(self, app_name, lesson, category="games"):
        """Add a learned lesson to the knowledge base (Enhanced for adaptive learning)"""
        try:
            app_context = self.knowledge.get('categories', {}).get(category, {}).get(app_name)
            if not app_context:
                app_context = self.load_app_context(app_name, category)
            
            if 'lessons' not in app_context:
                app_context['lessons'] = []
            
            # Enhance lesson with adaptive learning context
            enhanced_lesson = f"{lesson} [Adaptive Learning + Knowledge System Enhanced]"
            
            # Avoid duplicate lessons
            if enhanced_lesson not in app_context['lessons']:
                app_context['lessons'].append(enhanced_lesson)
                print(f"📚 Adaptive Knowledge System lesson learned for {app_name}: {lesson}")
                
                # Keep only recent lessons
                if len(app_context['lessons']) > 20:
                    app_context['lessons'] = app_context['lessons'][-20:]
                
                self.save_knowledge()
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error adding lesson: {e}")
            return False
    
    def add_strategy(self, app_name, strategy, category="games"):
        """Add a strategy to the knowledge base (Enhanced for adaptive learning)"""
        try:
            app_context = self.knowledge.get('categories', {}).get(category, {}).get(app_name)
            if not app_context:
                app_context = self.load_app_context(app_name, category)
            
            if 'strategies' not in app_context:
                app_context['strategies'] = []
            
            # Enhance strategy with adaptive learning context
            enhanced_strategy = f"{strategy} [Discovered via Adaptive Knowledge System]"
            
            # Avoid duplicate strategies
            if enhanced_strategy not in app_context['strategies']:
                app_context['strategies'].append(enhanced_strategy)
                print(f"🎯 Adaptive Knowledge System strategy added for {app_name}: {strategy}")
                self.save_knowledge()
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error adding strategy: {e}")
            return False

class DualBrainAgent:
    """Complete dual brain system combining core learning + symbolic knowledge (Enhanced for Adaptive Learning)"""
    
    def __init__(self, brain_file='agent_brain.json', knowledge_file='agent_knowledge.json'):
        self.brain = AgentBrain(brain_file)
        self.knowledge = AgentKnowledge(knowledge_file)
        self.current_app = None
        self.current_context = None
        self.current_adaptive_params = None  # NEW: Track current adaptive parameters
        
        print("🧠🧩 Dual Brain Agent Byte v1.2 Adaptive Learning + Knowledge System Enhanced initialized!")
        print(f"   Core Brain: {self.brain.training_steps} training steps")
        print(f"   Knowledge: {len(self.knowledge.knowledge.get('categories', {}).get('games', {}))} environments known")
        print(f"   🔧 Adaptive Learning: {len(self.brain.environment_adaptations)} environment adaptations stored")
        print(f"   🧩 Knowledge System: Active and ready for intelligent decision making")
    
    def start_session(self, app_name, category="games", env_context=None):
        """Start a session in specific environment with adaptive learning and environmental context"""
        self.current_app = app_name
        self.current_context = self.knowledge.load_app_context(app_name, category, env_context)
        
        # NEW: Get recommended adaptive parameters for this environment
        self.current_adaptive_params = self.brain.get_recommended_parameters(app_name)
        
        print(f"🚀 Starting adaptive knowledge-enhanced session: {app_name}")
        if self.current_context:
            print(f"   Loaded context with {len(self.current_context.get('strategies', []))} strategies")
            print(f"   Previous performance: {self.knowledge._calculate_win_rate(self.current_context):.1f}% win rate")
            print(f"   🔧 Recommended parameters: {self.current_adaptive_params}")
            
            if env_context:
                print(f"   🌟 Environmental context integrated:")
                print(f"      Objective: {self.current_context.get('objective', 'Unknown')}")
                if 'environmental_rules' in self.current_context:
                    print(f"      Rules understood: {len(self.current_context['environmental_rules'])} categories")
                if 'environment_context' in self.current_context:
                    env_meta = self.current_context['environment_context'].get('context_metadata', {})
                    print(f"      Environment version: {env_meta.get('environment_version', 'Unknown')}")
                    print(f"      🧩 Knowledge system ready for intelligent application")
                    print(f"      🔧 Adaptive learning optimized for environment characteristics")
        
        return self.current_context
    
    def end_session(self, win, reward, outcome_summary, adaptive_params_used=None):
        """End current session and record results with adaptive learning tracking"""
        if self.current_app:
            # Record game result
            self.knowledge.record_game_result(self.current_app, win, reward, outcome_summary)
            
            # NEW: Record adaptive learning effectiveness if parameters were used
            if adaptive_params_used:
                adaptation_data = {
                    'parameters_used': adaptive_params_used,
                    'win': win,
                    'final_reward': reward,
                    'performance_improvement': 0,  # Could calculate vs baseline
                    'knowledge_effectiveness': 0   # Could get from agent stats
                }
                
                # Record in both brain and knowledge systems
                self.brain.record_environment_adaptation(self.current_app, adaptation_data)
                self.knowledge.record_adaptive_learning_result(self.current_app, adaptation_data)
            
            # Reflect on performance periodically
            games_played = self.current_context.get('metrics', {}).get('games_played', 0)
            if games_played % 5 == 0:  # Reflect every 5 games
                self.knowledge.reflect_on_performance(self.current_app)
            
            print(f"✅ Adaptive knowledge-enhanced session ended for {self.current_app}")
        
        self.current_app = None
        self.current_context = None
        self.current_adaptive_params = None
    
    def save_all(self):
        """Save both brain and knowledge with adaptive learning data"""
        brain_saved = self.brain.save_brain()
        knowledge_saved = self.knowledge.save_knowledge()
        return brain_saved and knowledge_saved
    
    def get_session_summary(self):
        """Get summary of current session knowledge with adaptive learning insights"""
        if self.current_app:
            base_summary = self.knowledge.get_app_summary(self.current_app)
            adaptive_analysis = self.knowledge.get_adaptive_learning_analysis(self.current_app)
            return f"{base_summary}\n\n{adaptive_analysis}"
        return "No active session"
    
    def get_adaptive_learning_status(self):
        """Get current status of adaptive learning system"""
        return {
            'enabled': self.brain.adaptive_learning_enabled,
            'environments_adapted': len(self.brain.environment_adaptations),
            'current_session': self.current_app,
            'current_params': self.current_adaptive_params,
            'total_adaptations': sum(env_data['adaptations_count'] for env_data in self.brain.environment_adaptations.values())
        }

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Enhanced Dual Brain System with Adaptive Learning...")
    
    # Create dual brain agent
    agent = DualBrainAgent()
    
    # Test adaptive learning status
    adaptive_status = agent.get_adaptive_learning_status()
    print(f"🔧 Adaptive Learning Status: {adaptive_status}")
    
    # Create test environment context with learning parameters
    test_env_context = {
        'name': 'test_pong',
        'objective': {'primary': 'Score 21 points before opponent'},
        'learning_parameters': {
            'recommended_gamma': 0.90,
            'gamma_rationale': 'Short-term competitive game with immediate feedback',
            'recommended_learning_rate': 0.001,
            'temporal_characteristics': {
                'match_duration': '2-5 minutes',
                'feedback_immediacy': 'Immediate',
                'decision_frequency': '60 decisions per second'
            }
        },
        'strategic_concepts': {
            'core_skills': ['Ball trajectory prediction', 'Optimal paddle positioning'],
            'tactical_approaches': ['Defensive positioning']
        }
    }
    
    # Start a session with adaptive learning
    context = agent.start_session("test_pong", env_context=test_env_context)
    
    # Simulate some game results with adaptive parameter tracking
    results = [
        (True, 8.5, "Won with adaptive gamma optimization"),
        (False, -2.1, "Lost - adaptive parameters need adjustment"),
        (True, 12.3, "Excellent performance with optimized learning rate"),
        (False, -5.2, "Strategy failure despite adaptive learning"),
        (True, 15.7, "Dominated with perfect adaptive parameter combination")
    ]
    
    for win, reward, summary in results:
        agent.knowledge.record_game_result("test_pong", win, reward, summary)
        
        # Simulate adaptive parameter usage
        adaptive_params_used = {
            'gamma': 0.90,
            'learning_rate': 0.001,
            'source': 'environment_recommendation'
        }
        
        # Record adaptive learning result
        adaptation_data = {
            'parameters_used': adaptive_params_used,
            'win': win,
            'final_reward': reward,
            'performance_improvement': reward if win else 0,
            'knowledge_effectiveness': 0.8 if win else 0.3
        }
        agent.knowledge.record_adaptive_learning_result("test_pong", adaptation_data)
    
    # Add some lessons and strategies with adaptive learning context
    agent.knowledge.add_lesson("test_pong", "Adaptive gamma 0.90 significantly improves short-term decision making")
    agent.knowledge.add_lesson("test_pong", "Environment-specific learning rates optimize convergence speed")
    agent.knowledge.add_strategy("test_pong", "Use recommended gamma for environment type to maximize learning efficiency")
    
    # Generate reflections
    agent.knowledge.reflect_on_performance("test_pong")
    
    # Show comprehensive summary
    print("\n" + "="*80)
    print(agent.get_session_summary())
    
    # Show adaptive learning analysis
    print("\n" + "="*80)
    adaptive_analysis = agent.knowledge.get_adaptive_learning_analysis("test_pong")
    print(adaptive_analysis)
    
    # End session with adaptive learning tracking
    agent.end_session(True, 10.5, "Final game won with optimized adaptive parameters", adaptive_params_used={'gamma': 0.90})
    
    # Save everything
    agent.save_all()
    
    print("\n✅ Enhanced Dual Brain System with Adaptive Learning test complete!")
    print(f"🔧 Final adaptive learning status: {agent.get_adaptive_learning_status()}")
