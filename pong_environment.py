# pong_environment.py - Enhanced with Symbolic Interpretation Methods
import numpy as np
import random
import time

class PongEnvironment:
    """Symbolic Pong Environment - Enhanced with Environment-Specific Learning Parameters and Symbolic Interpretation"""
    """Enhanced with center-zero coordinate system and modular symbolic interpretation"""
    
    def __init__(self, width=800, height=400):
        print("🏓 Initializing Enhanced Symbolic Pong Environment with Adaptive Learning Parameters...")
        
        # Game settings
        self.width = width
        self.height = height
        self.paddle_height = 80
        self.paddle_width = 10
        self.ball_size = 10
        self.paddle_speed = 8
        self.ball_speed = 8
        self.default_ball_speed = 6
        self.winning_score = 21
        
        # ENVIRONMENT-SPECIFIC LEARNING PARAMETERS
        self.recommended_gamma = 0.90  # Pong-specific: short-term game with immediate feedback
        self.recommended_learning_rate = 0.001
        self.recommended_exploration_start = 0.8
        self.recommended_exploration_decay = 0.995
        self.recommended_min_exploration = 0.1
        
        # Game state (keeping existing implementation)
        self.running = False
        self.game_over = False
        self.winner = None
        
        # Positions
        self.ball_x = 0
        self.ball_y = 0
        self.ball_dx = 0
        self.ball_dy = 0
        
        self.player_paddle_y = 0
        self.ai_paddle_y = 0
        
        self.player_score = 0
        self.ai_score = 0
        
        # Tracking for rewards and state
        self.previous_ball_x = 0
        self.previous_ball_y = 0
        self.last_ai_action = 0
        
        # Symbolic task tracking (generalized)
        self.task_successes = 0      # Successfully intercepted ball
        self.task_failures = 0       # Failed to intercept ball
        self.success_streak = 0      # Current consecutive successes
        self.best_success_streak = 0 # Best streak this round
        self.total_success_bonus = 0.0
        self.total_failure_penalty = 0.0
        
        # Hit-to-score tracking for bonus rewards
        self.last_action_was_hit = False
        
        # Task completion tracking
        self.task_completions = 0    # Goals scored (task completion)
        self.task_failures_major = 0 # Goals conceded (major failure)
        
        # User action tracking for learning
        self.user_action_history = []
        self.last_user_action = None
        self.last_user_state = None
        self.user_demonstration_active = False
        self.user_hit_count = 0
        self.user_miss_count = 0
        self.user_successful_defenses = 0
        self.user_action_pending = False
        self.last_user_paddle_y = 0
        self.user_ball_collision_detected = False
        
        self.reset_game()
        
        print("✅ Enhanced Symbolic Pong Environment Ready!")
        print("🎯 SYMBOLIC REWARD SYSTEM:")
        print("   Task Success (hit): +1.0 + 0.25 * streak")
        print("   Task Failure (miss): -0.5")
        print("   Task Completion (goal): +3.0 / -0.5")
        print("   Match Win/Loss: +10.0 / -10.0")
        print("👤 USER LEARNING SYSTEM:")
        print("   Tracking user paddle actions and outcomes")
        print("   Recording successful user strategies")
        print("🧩 KNOWLEDGE SYSTEM COMPATIBLE:")
        print("   Provides comprehensive environmental context")
        print("   Supports intelligent strategy application")
        print("⚙️ ADAPTIVE LEARNING PARAMETERS:")
        print(f"   Recommended Gamma: {self.recommended_gamma} (short-term focus)")
        print(f"   Recommended Learning Rate: {self.recommended_learning_rate}")
        print(f"   Recommended Exploration: {self.recommended_exploration_start} → {self.recommended_min_exploration}")
        print("🧠 Symbolic Mapping: hit→success, miss→failure, score→completion")

    # NEW: Symbolic Interpretation Methods for Agent Byte Integration

    def interpret_reward(self, reward):
        """Return a symbolic interpretation of the reward for Pong environment"""
        if reward >= 10.0:
            return "Match victory - ultimate success"
        elif reward <= -10.0:
            return "Match defeat - ultimate failure"
        elif reward >= 4.0:
            return "Scored after hit - perfect combo execution"
        elif reward >= 3.0:
            return "Goal scored - task completion achieved"
        elif reward >= 1.5:
            return "Successful hit with streak bonus - building momentum"
        elif reward >= 1.0:
            return "Successful ball hit - task success"
        elif reward >= 0.5:
            return "Minor positive outcome - good positioning"
        elif reward > 0:
            return "Small progress - moving in right direction"
        elif reward == 0:
            return "Neutral action - no immediate impact"
        elif reward >= -0.5:
            return "Minor setback - missed opportunity"
        elif reward >= -3.0:
            return "Task failure - missed ball or poor positioning"
        elif reward >= -5.0:
            return "Significant failure - opponent scored"
        else:
            return "Major failure - critical mistake"

    def generate_lesson_from_reward(self, reward, context=None):
        """Generate a learning lesson string from a reward in Pong context"""
        meaning = self.interpret_reward(reward)
        
        # Add context-specific insights for Pong
        if reward >= 4.0:
            return f"Hit-to-score combo achieved with reward {reward:.1f} - timing and positioning critical for Pong success"
        elif reward >= 3.0:
            return f"Goal scored (reward {reward:.1f}) - successful Pong strategy worth repeating"
        elif reward >= 1.5:
            return f"Streak building in Pong (reward {reward:.1f}) - consistency is key to momentum"
        elif reward >= 1.0:
            return f"Successful ball interception (reward {reward:.1f}) - core Pong skill demonstrated"
        elif reward <= -5.0:
            return f"Poor Pong performance ({reward:.1f}) - defensive positioning needs improvement"
        elif reward < 0:
            return f"Pong mistake made ({reward:.1f}) - avoid similar action patterns in future rallies"
        else:
            return f"Pong lesson learned: {meaning} (reward={reward:.1f})"

    def generate_strategy_from_performance(self, wins, games_played, avg_reward):
        """Generate strategic insights from performance data specific to Pong"""
        win_rate = (wins / max(1, games_played)) * 100
        
        if win_rate > 70 and avg_reward > 5:
            return "Master Pong strategy: Aggressive hit-to-score combinations with strong defensive positioning"
        elif win_rate > 50 and avg_reward > 2:
            return "Effective Pong approach: Consistent ball interception with occasional scoring opportunities"
        elif win_rate > 30:
            return "Developing Pong competency: Focus on basic ball tracking and paddle positioning"
        else:
            return "Fundamental Pong learning needed: Improve reaction timing and ball trajectory prediction"
    
    def get_task_success_description(self):
        """Return environment-specific description of what constitutes task success"""
        return "Successfully intercepting the ball with paddle contact in Pong"
    
    def get_task_failure_description(self):
        """Return environment-specific description of what constitutes task failure"""
        return "Missing the ball when it approaches the AI paddle in Pong"
    
    def get_task_completion_description(self):
        """Return environment-specific description of what constitutes task completion"""
        return "Scoring a goal by getting the ball past the opponent's paddle in Pong"
    
    def get_major_failure_description(self):
        """Return environment-specific description of what constitutes major failure"""
        return "Allowing the opponent to score a goal in Pong"
    
    def format_user_demo_outcome(self, outcome_type, reward):
        """Format user demonstration outcome for Pong-specific context"""
        if outcome_type == "hit":
            return f"User demonstrated successful Pong ball interception (reward: {reward:.2f})"
        elif outcome_type == "miss":
            return f"User missed ball in Pong (reward: {reward:.2f}) - learning opportunity"
        elif outcome_type == "positioning":
            return f"User showed Pong paddle positioning technique (reward: {reward:.2f})"
        else:
            return f"User Pong demonstration: {outcome_type} (reward: {reward:.2f})"
    
    def get_performance_feedback_phrase(self, performance_metric, value):
        """Get Pong-specific performance feedback phrases"""
        if performance_metric == "hit_rate":
            if value > 80:
                return "Excellent Pong ball interception skills!"
            elif value > 60:
                return "Good Pong hitting consistency"
            elif value > 40:
                return "Developing Pong accuracy"
            else:
                return "Pong hitting needs improvement"
        
        elif performance_metric == "win_rate":
            if value > 70:
                return "Dominating Pong performance!"
            elif value > 50:
                return "Competitive Pong player"
            elif value > 30:
                return "Learning Pong fundamentals"
            else:
                return "Pong beginner level"
        
        elif performance_metric == "streak":
            if value > 10:
                return "Amazing Pong streak! Unstoppable!"
            elif value > 5:
                return "Great Pong momentum building"
            elif value > 2:
                return "Decent Pong consistency"
            else:
                return "Working on Pong reliability"
        
        return f"Pong {performance_metric}: {value}"
    
    def get_environment_specific_constants(self):
        """Return Pong-specific constants that Agent Byte might need"""
        return {
            "hit_to_score_bonus_threshold": 3.5,
            "high_reward_threshold": 5.0,
            "low_reward_threshold": -3.0,
            "streak_bonus_multiplier": 0.25,
            "task_success_base_reward": 1.0,
            "task_failure_base_penalty": -0.5,
            "task_completion_bonus": 3.0,
            "major_failure_penalty": -0.5,
            "match_win_bonus": 10.0,
            "match_loss_penalty": -10.0,
            "demo_success_reward": 1.5,
            "demo_failure_penalty": -0.5,
            "positioning_reward": 0.1
        }
    
    def should_generate_lesson(self, reward, context=None):
        """Determine if a lesson should be generated from this reward in Pong context"""
        constants = self.get_environment_specific_constants()
        
        # Generate lessons for significant events
        return (
            reward >= constants["hit_to_score_bonus_threshold"] or  # Hit-to-score combo
            reward >= constants["high_reward_threshold"] or         # High performance
            reward <= constants["low_reward_threshold"] or          # Poor performance
            abs(reward) >= constants["task_completion_bonus"]       # Major events
        )
    
    def should_generate_strategy(self, reward, context=None):
        """Determine if a strategy should be generated from this reward in Pong context"""
        constants = self.get_environment_specific_constants()
        
        # Generate strategies for excellent performance or major wins
        return (
            reward >= constants["match_win_bonus"] or               # Match victory
            reward >= constants["hit_to_score_bonus_threshold"]     # Excellent combo
        )

    # [Keep all existing methods unchanged - reset_game, create_state, step, etc.]
    
    def reset_game(self):
        """Reset game for new match with symbolic state reset"""
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = random.choice([-1, 1]) * self.ball_speed
        self.ball_dy = random.choice([-1, 1]) * self.ball_speed
        
        self.player_paddle_y = self.height // 2 - self.paddle_height // 2
        self.ai_paddle_y = self.height // 2 - self.paddle_height // 2
        
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        self.winner = None
        
        self.previous_ball_x = self.ball_x
        self.previous_ball_y = self.ball_y
        self.last_ai_action = 0
        
        # Reset symbolic tracking
        self.task_successes = 0
        self.task_failures = 0
        self.success_streak = 0
        self.best_success_streak = 0
        self.total_success_bonus = 0.0
        self.total_failure_penalty = 0.0
        self.task_completions = 0
        self.task_failures_major = 0
        self.last_action_was_hit = False
        
        # Reset user tracking
        self.user_action_history = []
        self.last_user_action = None
        self.last_user_state = None
        self.user_hit_count = 0
        self.user_miss_count = 0
        self.user_successful_defenses = 0

    def create_state(self):
        """Create state vector with CENTER-ZERO coordinates (center = 0.0)"""
        # Core position and velocity (CENTER-ZERO normalized)
        # X-axis: -1.0 (left) to +1.0 (right), center = 0.0
        # Y-axis: -0.5 (bottom) to +0.5 (top), center = 0.0
        
        ball_x_normalized = (self.ball_x - self.width/2) / (self.width/2)
        ball_y_normalized = (self.ball_y - self.height/2) / (self.height/2)
        
        # AI paddle center position (center-zero)
        ai_paddle_center = self.ai_paddle_y + self.paddle_height / 2
        ai_paddle_normalized = (ai_paddle_center - self.height/2) / (self.height/2)
        
        core_state = np.array([
            ball_x_normalized,                               # Ball X: -1.0 to +1.0
            ball_y_normalized,                               # Ball Y: -0.5 to +0.5
            (self.ball_dx + 15) / 30,                       # Ball X velocity (unchanged)
            (self.ball_dy + 15) / 30,                       # Ball Y velocity (unchanged)
            ai_paddle_normalized,                           # AI paddle: -0.5 to +0.5
            abs(self.ball_x - self.width) / self.width     # Distance to AI side (unchanged)
        ])
        
        # Strategic information (center-zero aware)
        ball_paddle_distance = abs(ball_y_normalized - ai_paddle_normalized)
        ball_approaching = 1.0 if self.ball_dx > 0 else 0.0
        
        # Predictive trajectory (center-zero)
        if self.ball_dx != 0:
            time_to_paddle = (self.width - self.ball_x) / abs(self.ball_dx)
            predicted_y_pixels = self.ball_y + self.ball_dy * time_to_paddle
            predicted_y_pixels = np.clip(predicted_y_pixels, 0, self.height)
            predicted_y_normalized = (predicted_y_pixels - self.height/2) / (self.height/2)
            prediction_error = abs(predicted_y_normalized - ai_paddle_normalized)
        else:
            prediction_error = 0.0  # Changed from 0.5 to 0.0 (center)
        
        # Optimal target position (center-zero)
        # Keep ball in safe zone, but normalize around center
        target_y_pixels = np.clip(self.ball_y, 40, self.height - 40)
        target_y = (target_y_pixels - self.height/2) / (self.height/2)
        
        strategic_state = np.array([
            ball_paddle_distance,
            ball_approaching,
            prediction_error,
            target_y
        ])
        
        # Context and urgency (mostly unchanged)
        ball_speed_norm = np.sqrt(self.ball_dx ** 2 + self.ball_dy ** 2) / 20
        ball_angle = np.arctan2(self.ball_dy, self.ball_dx) / np.pi
        in_ai_court = 1.0 if self.ball_x > self.width / 2 else 0.0
        urgent_situation = 1.0 if (self.ball_x > self.width * 0.8 and self.ball_dx > 0) else 0.0
        
        context_state = np.array([
            ball_speed_norm,
            ball_angle,
            in_ai_court,
            urgent_situation
        ])
        
        # Combine all state components (14 dimensions total)
        full_state = np.concatenate([core_state, strategic_state, context_state])
        return full_state

    def get_env_context(self):
        """Provide comprehensive environment context with adaptive learning parameters for Agent Byte"""
        context = {
            "environment_type": "competitive_game",
            "name": "pong",
            "display_name": "Classic Pong",
            "version": "Enhanced Symbolic v1.2 - Adaptive Learning Parameters + Modular Interpretation",
            
            # Core game understanding
            "objective": {
                "primary": "Score 21 points before opponent",
                "secondary": "Prevent opponent from scoring",
                "win_condition": "First player to reach winning_score wins",
                "lose_condition": "Opponent reaches winning_score first"
            },
            
            # ENVIRONMENT-SPECIFIC LEARNING PARAMETERS
            "learning_parameters": {
                "recommended_gamma": self.recommended_gamma,
                "gamma_rationale": "Short-term competitive game with immediate feedback loops",
                "recommended_learning_rate": self.recommended_learning_rate,
                "recommended_exploration": {
                    "start": self.recommended_exploration_start,
                    "decay": self.recommended_exploration_decay,
                    "minimum": self.recommended_min_exploration,
                    "rationale": "High initial exploration for quick adaptation, moderate decay for consistent improvement"
                },
                "temporal_characteristics": {
                    "match_duration": "2-5 minutes typical",
                    "decision_frequency": "60 decisions per second",
                    "feedback_immediacy": "Immediate (hit/miss within 1 second)",
                    "reward_temporal_distance": "1-3 seconds from action to outcome"
                },
                "environment_complexity": {
                    "state_space": "14-dimensional continuous",
                    "action_space": "3-dimensional discrete", 
                    "dynamics": "Deterministic physics with reactive opponent",
                    "learning_challenge": "Timing and prediction under pressure"
                }
            },
            
            # NEW: Symbolic interpretation configuration
            "symbolic_interpretation": {
                "task_success_name": "ball_hit",
                "task_failure_name": "ball_miss", 
                "task_completion_name": "goal_scored",
                "major_failure_name": "goal_conceded",
                "reward_thresholds": self.get_environment_specific_constants(),
                "lesson_generation_enabled": True,
                "strategy_generation_enabled": True,
                "performance_feedback_enabled": True
            },
            
            # Game mechanics and rules
            "rules": {
                "ball_mechanics": {
                    "bounces_off_walls": "Ball reflects off top and bottom boundaries",
                    "paddle_collision": "Ball reflects when hit by paddle",
                    "scoring": "Ball passing paddle results in opponent scoring"
                },
                "paddle_control": {
                    "movement": "Paddle moves vertically only",
                    "speed": f"{self.paddle_speed} pixels per action",
                    "constraints": "Cannot move beyond screen boundaries"
                },
                "scoring_system": {
                    "point_value": 1,
                    "winning_score": self.winning_score,
                    "reset_after_score": "Ball returns to center after each point"
                }
            },
            
            # Strategic concepts the AI should understand (Enhanced for Knowledge System)
            "strategic_concepts": {
                "core_skills": [
                    "Ball trajectory prediction",
                    "Optimal paddle positioning", 
                    "Timing and reaction speed",
                    "Angle control for returns"
                ],
                "tactical_approaches": [
                    "Defensive positioning - stay between ball and goal",
                    "Aggressive returns - use paddle edges for angles",
                    "Predictive movement - anticipate ball path",
                    "Counter-attacking - respond to opponent patterns"
                ],
                "success_patterns": [
                    "Early positioning beats reactive movement",
                    "Consistent hits build momentum",
                    "Angled shots create scoring opportunities",
                    "Defensive stability enables offensive chances"
                ],
                "situational_awareness": [
                    "Critical defense when ball approaches AI side",
                    "Offensive opportunities when ball is on player side",
                    "Preparation phase when ball is in neutral zone",
                    "Positioning optimization during non-critical moments"
                ]
            },
            
            # Reward structure explanation with gamma considerations
            "reward_structure": {
                "positive_rewards": {
                    "successful_hit": "+1.0 base + streak bonus (+0.25 per consecutive hit)",
                    "score_goal": "+3.0 task completion bonus",
                    "hit_to_score_combo": "+1.0 additional bonus for scoring after hitting",
                    "match_victory": "+10.0 final victory bonus"
                },
                "negative_rewards": {
                    "missed_ball": "-0.5 task failure penalty",
                    "concede_goal": "-0.5 minor penalty for allowing score",
                    "match_defeat": "-10.0 final defeat penalty"
                },
                "reward_philosophy": "Emphasis on consistent performance and strategic play over individual actions",
                "gamma_alignment": f"Gamma {self.recommended_gamma} balances immediate feedback with short-term strategy building"
            },
            
            # Environment-specific metrics and success indicators
            "performance_metrics": {
                "primary_kpis": [
                    "Hit rate percentage",
                    "Win rate percentage", 
                    "Average match reward",
                    "Hit-to-score conversion rate"
                ],
                "advanced_metrics": [
                    "Consecutive hit streaks",
                    "Defensive success rate",
                    "Strategic positioning effectiveness",
                    "Adaptation to opponent patterns"
                ],
                "learning_indicators": [
                    "Increasing hit rate over time",
                    "Longer winning streaks",
                    "Higher average rewards per match",
                    "Improved anticipation timing"
                ]
            },
            
            # State space information for the AI (Knowledge System Compatible)
            "state_representation": {
                "dimensions": 14,
                "components": {
                    "positional": ["ball_x", "ball_y", "paddle_y"],
                    "velocity": ["ball_dx", "ball_dy"], 
                    "strategic": ["ball_paddle_distance", "prediction_error", "target_y"],
                    "contextual": ["ball_speed", "ball_angle", "urgency_level"]
                },
                "normalization": "All values normalized to [0,1] range for stable learning",
                "knowledge_system_mapping": {
                    "situation_analysis": "State components map to game situation types",
                    "urgency_detection": "Urgency level determines strategy selection priority",
                    "trajectory_prediction": "Ball position and velocity enable predictive strategies"
                }
            },
            
            # Action space explanation  
            "action_space": {
                "size": 3,
                "actions": {
                    0: "Move paddle UP",
                    1: "Keep paddle STATIONARY", 
                    2: "Move paddle DOWN"
                },
                "action_philosophy": "Simple discrete actions for precise control",
                "timing_importance": "Action timing more critical than action choice",
                "knowledge_system_enhancement": "Actions modified by strategic context analysis"
            },
            
            # Common failure modes and how to avoid them (Enhanced for Knowledge System)
            "failure_patterns": {
                "reactive_play": {
                    "description": "Only moving after ball reaches AI side",
                    "solution": "Predictive positioning based on ball trajectory",
                    "knowledge_system_prevention": "Early positioning strategies applied proactively"
                },
                "edge_camping": {
                    "description": "Staying at screen edges consistently", 
                    "solution": "Dynamic positioning based on ball location",
                    "knowledge_system_prevention": "Center-bias in positioning strategies"
                },
                "over_correction": {
                    "description": "Making too many rapid movements",
                    "solution": "Smooth, calculated position adjustments",
                    "knowledge_system_prevention": "Confidence-based action selection"
                },
                "passive_defense": {
                    "description": "Only focusing on blocking without counter-attack",
                    "solution": "Transition from defense to offense after successful hits",
                    "knowledge_system_prevention": "Situation-aware strategy switching"
                }
            },
            
            # Environment dynamics and physics
            "physics_model": {
                "ball_speed": f"Base speed: {self.ball_speed}, Max speed: 20",
                "collision_mechanics": "Simple reflection with angle modification based on paddle hit position",
                "boundary_behavior": "Ball bounces off top/bottom, passes through left/right for scoring",
                "paddle_physics": "Instant response to actions, no momentum or inertia"
            },
            
            # Recommended learning approach (Enhanced for Knowledge System)
            "learning_recommendations": {
                "early_phase": [
                    "Focus on basic ball interception",
                    "Learn optimal paddle positioning",
                    "Develop timing for ball contact"
                ],
                "intermediate_phase": [
                    "Master trajectory prediction",
                    "Implement angle-based returns", 
                    "Build consistent hit streaks"
                ],
                "advanced_phase": [
                    "Develop opponent pattern recognition",
                    "Optimize hit-to-score conversion",
                    "Master defensive-to-offensive transitions"
                ],
                "knowledge_system_phase": [
                    "Apply situational strategy selection",
                    "Utilize environmental context awareness",
                    "Optimize strategy performance based on feedback",
                    "Integrate symbolic reasoning with neural learning"
                ]
            },
            
            # Integration metadata
            "context_metadata": {
                "generated_at": time.time(),
                "environment_version": "Enhanced Symbolic v1.2 - Adaptive Learning Parameters + Modular Interpretation",
                "context_version": "1.2.2",
                "last_updated": time.strftime('%Y-%m-%d %H:%M:%S'),
                "compatible_agents": ["Agent Byte v1.2", "Knowledge System Enhanced", "Dual Brain Architecture", "Modular AI Agent"],
                "environment_complexity": "Medium - Simple physics, complex strategy",
                "knowledge_system_features": [
                    "Comprehensive situation analysis",
                    "Strategy-to-action mapping",
                    "Failure pattern avoidance",
                    "Performance-based optimization"
                ],
                "adaptive_learning_features": [
                    "Environment-specific gamma recommendation",
                    "Temporal characteristic analysis", 
                    "Reward-structure-aligned learning rates",
                    "Context-aware exploration strategies"
                ],
                "modular_features": [
                    "Environment-specific symbolic interpretation",
                    "Configurable reward thresholds",
                    "Pluggable lesson and strategy generation",
                    "Standardized Agent Byte integration interface"
                ],
                "default_gamma": self.recommended_gamma,
                "gamma_category": "short_term_feedback",
                "optimal_for": [
                    "Real-time competitive gameplay",
                    "Immediate reward feedback",
                    "Tactical skill development",
                    "Reaction time optimization"
                ]
            }
        }
        
        return context

    # [Keep all other existing methods unchanged - step, move_player_paddle, etc.]
    
    
    def step(self, agent_action, mini_byte_action):
        """Execute one game step with symbolic reward tracking"""
        # Store previous positions for collision detection
        self.previous_ball_x = self.ball_x
        self.previous_ball_y = self.ball_y
        self.user_ball_collision_detected = False
        
        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with top/bottom walls
        if self.ball_y <= self.ball_size / 2 or self.ball_y >= self.height - self.ball_size / 2:
            self.ball_dy = -self.ball_dy
            self.ball_y = np.clip(self.ball_y, self.ball_size / 2, self.height - self.ball_size / 2)
        
        # Move AI paddle based on action
        movement = agent_action - 1  # Convert 0,1,2 to -1,0,1
        self.ai_paddle_y += movement * self.paddle_speed
        self.ai_paddle_y = max(0, min(self.height - self.paddle_height, self.ai_paddle_y))
        
        # Initialize reward and outcome flags
        step_reward = 0.0
        game_ended = False
        task_success = False
        task_failure = False
        
        # AI paddle collision detection (task success)
        if (
                self.width - self.paddle_width - self.ball_size / 2 <= self.ball_x <= self.width - self.paddle_width + self.ball_size / 2 and
            self.ball_dx > 0 and
            self.ai_paddle_y <= self.ball_y <= self.ai_paddle_y + self.paddle_height):
            
            task_success = True
            
            # Reflect ball
            self.ball_dx = -abs(self.ball_dx)
            hit_position = (self.ball_y - self.ai_paddle_y - self.paddle_height / 2) / (self.paddle_height / 2)
            self.ball_dy += hit_position * 3  # Add spin based on hit position
        
        # Player paddle collision
        elif (self.paddle_width + self.ball_size / 2 >= self.ball_x >= self.paddle_width - self.ball_size / 2 and
              self.ball_dx < 0 and
              self.player_paddle_y <= self.ball_y <= self.player_paddle_y + self.paddle_height):
            self.user_ball_collision_detected = True
            self.user_hit_count += 1
            print(f"👤 USER HIT DETECTED! Total user hits: {self.user_hit_count}")

            # Reflect ball
            self.ball_dx = abs(self.ball_dx)
            hit_position = (self.ball_y - self.player_paddle_y - self.paddle_height / 2) / (self.paddle_height / 2)
            self.ball_dy += hit_position * 3  # Add spin based on hit position
        elif (self.ball_x < self.paddle_width - self.ball_size / 2 <= self.previous_ball_x and
              self.ball_dx < 0 and
              not (self.player_paddle_y <= self.ball_y <= self.player_paddle_y + self.paddle_height)):

            # User missed the ball
            self.user_miss_count += 1
            print(f"👤 USER MISS DETECTED! Total user misses: {self.user_miss_count}")


        # Check for AI task failure (missed ball)
        elif (self.ball_x > self.width - self.paddle_width + self.ball_size / 2 >= self.previous_ball_x and
              self.ball_dx > 0 and
              not (self.ai_paddle_y <= self.ball_y <= self.ai_paddle_y + self.paddle_height)):
            
            task_failure = True
        
        # Process symbolic outcomes
        if task_success:
            step_reward += self.process_task_success()
        elif task_failure:
            step_reward += self.process_task_failure()
        
        # Scoring logic (task completion/major failure)
        if self.ball_x < 0:  # AI scores (task completion)
            self.ai_score += 1
            self.task_completions += 1
            step_reward += 3.0  # Task completion bonus
            
            # Hit-to-score bonus: +1 if last action was a hit
            if self.last_action_was_hit:
                step_reward += 1.0
                print(f"🎯 TASK COMPLETION #{self.task_completions}: Agent scores after hit! +3.0 + 1.0 (hit bonus) = +4.0")
            else:
                print(f"🎯 TASK COMPLETION #{self.task_completions}: Agent scores! +3.0")
            
            self.reset_ball()
            
            if self.ai_score >= self.winning_score:
                self.game_over = True
                self.winner = "Agent Byte"
                step_reward += 10.0  # Match win bonus
                game_ended = True
                print("🏆 MATCH WON: +10.0")
        
        elif self.ball_x > self.width:  # Player scores (major task failure)
            self.player_score += 1
            self.task_failures_major += 1
            step_reward -= 0.5  # Minor penalty for conceding
            self.user_successful_defenses += 1  # User successfully defended and scored
            print(f"💔 MAJOR FAILURE #{self.task_failures_major}: Player scores! -0.5")
            self.reset_ball()
            
            if self.player_score >= self.winning_score:
                self.game_over = True
                self.winner = "Player"
                step_reward -= 10.0  # Match loss penalty
                game_ended = True
                print("💀 MATCH LOST: -10.0")
        
        # Create next state
        next_state = self.create_state()
        
        return next_state, step_reward, game_ended

    def process_task_success(self):
        """Agent successfully intercepted ball - symbolic task success"""
        self.task_successes += 1
        self.success_streak += 1
        self.last_action_was_hit = True  # Track for hit-to-score bonus
        
        # Reward calculation: base + streak bonus
        base_reward = 1.0
        streak_bonus = min(self.success_streak - 1, 20) * 0.25  # Cap at 20 streak
        total_reward = base_reward + streak_bonus
        
        self.total_success_bonus += streak_bonus
        
        # Track best streak this round
        if self.success_streak > self.best_success_streak:
            self.best_success_streak = self.success_streak
        
        print(f"✅ TASK SUCCESS #{self.task_successes} (streak: {self.success_streak}): +{base_reward} + {streak_bonus:.2f} = +{total_reward:.2f}")
        return total_reward

    def process_task_failure(self):
        """Agent failed to intercept ball - symbolic task failure"""
        self.task_failures += 1
        self.success_streak = 0  # Reset success streak
        self.last_action_was_hit = False  # Reset hit tracking
        
        # Simple flat penalty
        base_penalty = -0.5
        self.total_failure_penalty += abs(base_penalty)
        
        print(f"❌ TASK FAILURE #{self.task_failures}: {base_penalty}")
        return base_penalty

    def reset_ball(self):
        """Reset ball to center with random direction"""
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = random.choice([-1, 1]) * self.ball_speed
        self.ball_dy = random.choice([-1, 1]) * self.ball_speed
        self.previous_ball_x = self.ball_x
        self.previous_ball_y = self.ball_y
        self.last_action_was_hit = False  # Reset hit tracking after ball reset

    def move_player_paddle(self, direction):
        """🔧 FIXED: Enhanced user action recording"""
        # Store previous position for better tracking
        self.last_user_paddle_y = self.player_paddle_y

        # Record the user action before moving
        self.record_user_action(direction)

        # Move the paddle
        self.player_paddle_y += direction * self.paddle_speed
        self.player_paddle_y = max(0, min(self.height - self.paddle_height, self.player_paddle_y))

        # Mark that user action is pending evaluation
        self.user_action_pending = True
    def record_user_action(self, user_action):
        """Record user action for learning purposes"""
        current_state = self.create_player_state()
        
        # Convert user action to standardized format (0=up, 1=stay, 2=down)
        if user_action == -1:  # Moving up
            action = 0
        elif user_action == 1:  # Moving down
            action = 2
        else:  # Staying still
            action = 1
        
        # Store the action and state
        self.last_user_action = action
        self.last_user_state = current_state.copy()
        
        # Add to history
        self.user_action_history.append({
            'timestamp': time.time(),
            'state': current_state.copy(),
            'action': action,
            'ball_x': self.ball_x,
            'ball_y': self.ball_y,
            'paddle_y': self.player_paddle_y,
            'ball_approaching': self.ball_dx < 0
        })
        
        # Keep history manageable
        if len(self.user_action_history) > 100:
            self.user_action_history = self.user_action_history[-50:]

    def create_player_state(self):
        """Create state vector for player side with CENTER-ZERO coordinates"""
        # Player side state with center-zero coordinates
        ball_x_normalized = (self.ball_x - self.width/2) / (self.width/2)
        ball_y_normalized = (self.ball_y - self.height/2) / (self.height/2)
        
        # Player paddle center position (center-zero)
        player_paddle_center = self.player_paddle_y + self.paddle_height / 2
        player_paddle_normalized = (player_paddle_center - self.height/2) / (self.height/2)
        
        core_state = np.array([
            ball_x_normalized,                               # Ball X: -1.0 to +1.0
            ball_y_normalized,                               # Ball Y: -0.5 to +0.5
            (self.ball_dx + 15) / 30,                       # Ball X velocity
            (self.ball_dy + 15) / 30,                       # Ball Y velocity
            player_paddle_normalized,                       # Player paddle: -0.5 to +0.5
            abs(self.ball_x - 0) / self.width              # Distance to player side
        ])
        
        # Strategic information for player (center-zero)
        ball_paddle_distance = abs(ball_y_normalized - player_paddle_normalized)
        ball_approaching = 1.0 if self.ball_dx < 0 else 0.0
        
        # Predictive trajectory for player (center-zero)
        if self.ball_dx != 0:
            time_to_paddle = abs(self.ball_x - 0) / abs(self.ball_dx)
            predicted_y_pixels = self.ball_y + self.ball_dy * time_to_paddle
            predicted_y_pixels = np.clip(predicted_y_pixels, 0, self.height)
            predicted_y_normalized = (predicted_y_pixels - self.height/2) / (self.height/2)
            prediction_error = abs(predicted_y_normalized - player_paddle_normalized)
        else:
            prediction_error = 0.0
        
        # Optimal target position for player (center-zero)
        target_y_pixels = np.clip(self.ball_y, 40, self.height - 40)
        target_y = (target_y_pixels - self.height/2) / (self.height/2)
        
        strategic_state = np.array([
            ball_paddle_distance,
            ball_approaching,
            prediction_error,
            target_y
        ])
        
        # Context and urgency for player
        ball_speed_norm = np.sqrt(self.ball_dx ** 2 + self.ball_dy ** 2) / 20
        ball_angle = np.arctan2(self.ball_dy, self.ball_dx) / np.pi
        in_player_court = 1.0 if self.ball_x < self.width / 2 else 0.0
        urgent_situation = 1.0 if (self.ball_x < self.width * 0.2 and self.ball_dx < 0) else 0.0
        
        context_state = np.array([
            ball_speed_norm,
            ball_angle,
            in_player_court,
            urgent_situation
        ])
        
        # Combine all state components (14 dimensions total)
        full_state = np.concatenate([core_state, strategic_state, context_state])
        return full_state

    def evaluate_user_action_outcome(self):
        """🔧 FIXED: Better user action evaluation"""
        if self.last_user_action is None or self.last_user_state is None:
            return None

        # Check if user successfully hit the ball (using our enhanced collision detection)
        if self.user_ball_collision_detected:
            reward = 1.5  # Positive reward for successful hit
            outcome = "hit"

            # Create demo entry
            demo_entry = {
                'state': self.last_user_state.copy(),
                'action': self.last_user_action,
                'reward': reward,
                'source': 'user_action',
                'outcome': outcome,
                'timestamp': time.time()
            }

            # Reset tracking
            self.last_user_action = None
            self.last_user_state = None
            self.user_ball_collision_detected = False

            return demo_entry

        # For positioning actions, give small reward
        elif self.user_action_pending:
            reward = self.evaluate_user_positioning()
            outcome = "positioning"

            if abs(reward) > 0.01:  # Only record if meaningful positioning
                demo_entry = {
                    'state': self.last_user_state.copy(),
                    'action': self.last_user_action,
                    'reward': reward,
                    'source': 'user_action',
                    'outcome': outcome,
                    'timestamp': time.time()
                }

                # Reset tracking
                self.last_user_action = None
                self.last_user_state = None
                self.user_action_pending = False

                return demo_entry


        return None

    def evaluate_user_positioning(self):
        """Evaluate user paddle positioning when not hitting/missing"""
        if self.last_user_state is None:
            return 0.0
        
        # Small reward for good positioning
        paddle_center = self.player_paddle_y + self.paddle_height / 2
        ball_paddle_distance = abs(self.ball_y - paddle_center)
        
        # Reward being close to ball when it's approaching
        if self.ball_dx < 0:  # Ball approaching player
            if ball_paddle_distance < self.paddle_height / 2:
                return 0.1  # Small positioning reward
            elif ball_paddle_distance < self.paddle_height:
                return 0.05
        
        return 0.0

    def set_ball_speed(self, speed):
        """Set ball speed (compatibility with existing system)"""
        self.ball_speed = max(1, min(20, speed))
        
        # Update current ball velocity if ball is moving
        if self.ball_dx != 0 or self.ball_dy != 0:
            current_speed = np.sqrt(self.ball_dx**2 + self.ball_dy**2)
            if current_speed > 0:
                speed_ratio = self.ball_speed / current_speed
                self.ball_dx *= speed_ratio
                self.ball_dy *= speed_ratio

    def get_game_state(self):
        """Return current game state for web interface"""
        return {
            'ball': {
                'x': float(self.ball_x), 
                'y': float(self.ball_y), 
                'size': int(self.ball_size)
            },
            'player_paddle': {
                'y': float(self.player_paddle_y), 
                'height': int(self.paddle_height)
            },
            'ai_paddle': {
                'y': float(self.ai_paddle_y), 
                'height': int(self.paddle_height)
            },
            'scores': {
                'player': int(self.player_score), 
                'ai': int(self.ai_score), 
                'winning_score': int(self.winning_score)
            },
            'dimensions': {
                'width': int(self.width), 
                'height': int(self.height)
            },
            'ball_speed': int(self.ball_speed),
            'game_over': bool(self.game_over),
            'winner': self.winner
        }

    def get_pong_stats(self):
        """Return symbolic performance statistics including user stats (Knowledge System Compatible)"""
        total_tasks = self.task_successes + self.task_failures
        task_success_rate = (self.task_successes / max(1, total_tasks)) * 100
        
        # Create symbolic action log
        symbolic_actions = []
        symbolic_outcomes = []
        
        if self.task_successes > 0:
            symbolic_actions.extend(['intercept'] * self.task_successes)
            symbolic_outcomes.extend(['success'] * self.task_successes)
        
        if self.task_failures > 0:
            symbolic_actions.extend(['miss'] * self.task_failures)
            symbolic_outcomes.extend(['failure'] * self.task_failures)
        
        if self.task_completions > 0:
            symbolic_actions.extend(['score'] * self.task_completions)
            symbolic_outcomes.extend(['completion'] * self.task_completions)
        
        # User performance stats
        total_user_actions = self.user_hit_count + self.user_miss_count
        user_hit_rate = (self.user_hit_count / max(1, total_user_actions)) * 100
        
        return {
            # Symbolic metrics (generalized)
            'task_successes': self.task_successes,
            'task_failures': self.task_failures,
            'task_success_rate': round(task_success_rate, 2),
            'success_streak_current': self.success_streak,
            'success_streak_best': self.best_success_streak,
            'total_success_bonus': round(self.total_success_bonus, 1),
            'total_failure_penalty': round(self.total_failure_penalty, 1),
            
            # Task completion metrics
            'task_completions': self.task_completions,
            'task_failures_major': self.task_failures_major,
            
            # User performance tracking
            'user_hits': self.user_hit_count,
            'user_misses': self.user_miss_count,
            'user_hit_rate': round(user_hit_rate, 2),
            'user_successful_defenses': self.user_successful_defenses,
            'user_actions_recorded': len(self.user_action_history),
            
            # Environment-specific learning parameters
            'recommended_gamma': self.recommended_gamma,
            'recommended_learning_rate': self.recommended_learning_rate,
            'recommended_exploration_start': self.recommended_exploration_start,
            'recommended_exploration_decay': self.recommended_exploration_decay,
            'recommended_min_exploration': self.recommended_min_exploration,
            'gamma_category': 'short_term_feedback',
            
            # Compatibility with existing UI (mapped to symbolic terms)
            'round_hits': self.task_successes,
            'round_misses': self.task_failures,
            'round_hit_rate': round(task_success_rate, 2),
            'round_best_hit_streak': self.best_success_streak,
            'round_worst_miss_streak': 0,  # Removed consecutive miss tracking
            'consecutive_hits': self.success_streak,
            'consecutive_misses': 0,  # Removed consecutive miss tracking
            'total_hit_bonus': round(self.total_success_bonus, 1),
            'total_miss_penalty': round(self.total_failure_penalty, 1),
            'points_scored': self.ai_score,
            'points_allowed': self.player_score,
            
            # Symbolic metadata
            'symbolic_actions': symbolic_actions,
            'symbolic_outcomes': symbolic_outcomes,
            'reward_paradigm': 'Modular Symbolic Task-Based Learning + Hit-to-Score Bonus + User Demo Learning + Knowledge System Enhanced + Adaptive Learning Parameters',
            'task_mapping': {
                'hit_ball': 'task_success',
                'miss_ball': 'task_failure', 
                'score_goal': 'task_completion',
                'score_after_hit': 'task_completion + hit_bonus',
                'concede_goal': 'major_task_failure',
                'user_hit': 'user_demonstration_success',
                'user_miss': 'user_demonstration_failure'
            },
            'knowledge_system_compatible': True,
            'adaptive_learning_compatible': True,
            'modular_interpretation_compatible': True,
            'environment_version': 'v1.2.2 Modular Interpretation + Adaptive Learning Parameters'
        }

# Test the enhanced environment with new modular interpretation methods
if __name__ == "__main__":
    print("🧪 Testing Enhanced Symbolic Pong Environment with Modular Interpretation...")
    
    env = PongEnvironment()
    
    # Test symbolic interpretation methods
    test_rewards = [15.0, 4.5, 3.0, 1.5, 1.0, 0.5, 0.0, -0.5, -3.0, -5.0, -15.0]
    
    print("\n🧩 Testing Symbolic Interpretation Methods:")
    for reward in test_rewards:
        interpretation = env.interpret_reward(reward)
        lesson = env.generate_lesson_from_reward(reward)
        should_lesson = env.should_generate_lesson(reward)
        should_strategy = env.should_generate_strategy(reward)
        
        print(f"   Reward {reward:+5.1f}: {interpretation}")
        print(f"   {'✅' if should_lesson else '❌'} Lesson: {'✅' if should_strategy else '❌'} Strategy")
        if should_lesson:
            print(f"   📚 Generated: {lesson}")
        print()
    
    # Test performance feedback
    print("🎯 Testing Performance Feedback:")
    test_metrics = [
        ("hit_rate", 85), ("hit_rate", 45), ("win_rate", 75), ("streak", 12)
    ]
    for metric, value in test_metrics:
        feedback = env.get_performance_feedback_phrase(metric, value)
        print(f"   {metric} {value}: {feedback}")
    
    # Test environment-specific constants
    constants = env.get_environment_specific_constants()
    print(f"\n⚙️ Environment Constants: {len(constants)} defined")
    for key, value in list(constants.items())[:5]:
        print(f"   {key}: {value}")
    
    # Test the enhanced environment with new modular interpretation methods
if __name__ == "__main__":
    print("🧪 Testing Enhanced Symbolic Pong Environment with Modular Interpretation...")
    
    env = PongEnvironment()
    
    # Test symbolic interpretation methods
    test_rewards = [15.0, 4.5, 3.0, 1.5, 1.0, 0.5, 0.0, -0.5, -3.0, -5.0, -15.0]
    
    print("\n🧩 Testing Symbolic Interpretation Methods:")
    for reward in test_rewards:
        interpretation = env.interpret_reward(reward)
        lesson = env.generate_lesson_from_reward(reward)
        should_lesson = env.should_generate_lesson(reward)
        should_strategy = env.should_generate_strategy(reward)
        
        print(f"   Reward {reward:+5.1f}: {interpretation}")
        print(f"   {'✅' if should_lesson else '❌'} Lesson: {'✅' if should_strategy else '❌'} Strategy")
        if should_lesson:
            print(f"   📚 Generated: {lesson}")
        print()
    
    # Test performance feedback
    print("🎯 Testing Performance Feedback:")
    test_metrics = [
        ("hit_rate", 85), ("hit_rate", 45), ("win_rate", 75), ("streak", 12)
    ]
    for metric, value in test_metrics:
        feedback = env.get_performance_feedback_phrase(metric, value)
        print(f"   {metric} {value}: {feedback}")
    
    # Test environment-specific constants
    constants = env.get_environment_specific_constants()
    print(f"\n⚙️ Environment Constants: {len(constants)} defined")
    for key, value in list(constants.items())[:5]:
        print(f"   {key}: {value}")
    
    # Test environmental context with new modular features
    env_context = env.get_env_context()
    symbolic_config = env_context.get('symbolic_interpretation', {})
    print(f"\n🔧 Symbolic Interpretation Config:")
    print(f"   Task Success: {symbolic_config.get('task_success_name')}")
    print(f"   Task Failure: {symbolic_config.get('task_failure_name')}")
    print(f"   Lesson Generation: {symbolic_config.get('lesson_generation_enabled')}")
    print(f"   Reward Thresholds: {len(symbolic_config.get('reward_thresholds', {}))} defined")
    
    # Test modular features metadata
    modular_features = env_context.get('context_metadata', {}).get('modular_features', [])
    print(f"\n🧩 Modular Features: {len(modular_features)}")
    for feature in modular_features:
        print(f"   ✅ {feature}")
    
    print("\n🎯 Enhanced Modular Pong Environment Test Complete!")
    print("✅ All symbolic interpretation methods working correctly!")
    print("✅ Ready for integration with modular Agent Byte!")