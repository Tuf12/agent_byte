# knowledge_system.py - How Agent Byte Uses Symbolic Knowledge - FIXED
import numpy as np
import random
from typing import Dict, List, Tuple, Any

from numpy import signedinteger
from numpy._typing import _32Bit, _64Bit


class KnowledgeInterpreter:
    """Translates symbolic knowledge into actionable decision-making logic"""

    def __init__(self):
        self.strategy_mappings = {
            # Environmental strategies
            "Ball trajectory prediction": self._apply_trajectory_prediction,
            "Optimal paddle positioning": self._apply_optimal_positioning,
            "Timing and reaction speed": self._apply_timing_optimization,
            "Angle control for returns": self._apply_angle_control,

            # Tactical approaches
            "Defensive positioning - stay between ball and goal": self._apply_defensive_positioning,
            "Aggressive returns - use paddle edges for angles": self._apply_aggressive_returns,
            "Predictive movement - anticipate ball path": self._apply_predictive_movement,
            "Counter-attacking - respond to opponent patterns": self._apply_counter_attacking,

            # Learned strategies (dynamic patterns)
            "move toward ball path": self._apply_ball_tracking,
            "angle shots": self._apply_angle_shots,
            "intercept early": self._apply_early_interception,

            # Success patterns
            "Early positioning beats reactive movement": self._apply_early_positioning,
            "Consistent hits build momentum": self._apply_consistency_focus,
            "Angled shots create scoring opportunities": self._apply_offensive_angles,
            "Defensive stability enables offensive chances": self._apply_defense_to_offense,

            # Master and Develop Strategies - 🔧 FIX: Map these properly
            "Master ball trajectory prediction": self._apply_trajectory_prediction,
            "Develop ball trajectory prediction": self._apply_trajectory_prediction,
            "Master optimal paddle positioning": self._apply_optimal_positioning,
            "Develop optimal paddle positioning": self._apply_optimal_positioning,
            "Master timing and reaction speed": self._apply_timing_optimization,
            "Develop timing and reaction speed": self._apply_timing_optimization,
            "Master angle control for returns": self._apply_angle_control,
            "Develop angle control for returns": self._apply_angle_control,
        }

        self.failure_avoidance = {
            "reactive_play": self._avoid_reactive_play,
            "edge_camping": self._avoid_edge_camping,
            "over_correction": self._avoid_over_correction,
            "passive_defense": self._avoid_passive_defense
        }

    def apply_knowledge(self, state: np.ndarray, q_values: np.ndarray, app_context: Dict,
                        exploration_rate: float) -> tuple[signedinteger[_32Bit | _64Bit], str] | tuple[int, str]:
        """Main method to apply knowledge and return modified action + reasoning"""

        if not app_context:
            return np.argmax(q_values), "No context available"

        # Analyze current game state
        game_situation = self._analyze_game_situation(state)

        # Determine which knowledge to apply
        applicable_strategies = self._select_applicable_strategies(app_context, game_situation)

        if not applicable_strategies:
            return np.argmax(q_values), "No applicable strategies for current situation"

        # Apply the most relevant strategy
        chosen_strategy = self._choose_best_strategy(applicable_strategies, game_situation)

        # Get strategy function and apply it
        strategy_func = self.strategy_mappings.get(chosen_strategy)
        if strategy_func:
            try:
                modified_action, confidence = strategy_func(state, q_values, game_situation)

                # Apply failure avoidance patterns
                final_action = self._apply_failure_avoidance(modified_action, state, game_situation, app_context)

                reasoning = f"Applied '{chosen_strategy}' (confidence: {confidence:.2f})"
                return final_action, reasoning
            except Exception as e:
                print(f"⚠️ Error applying strategy '{chosen_strategy}': {e}")
                return np.argmax(q_values), f"Strategy error, using neural fallback"
        else:
            # 🔧 FIX: Better fallback when strategy not found
            print(f"⚠️ Strategy '{chosen_strategy}' not in mappings, using neural fallback")
            return np.argmax(q_values), f"Neural fallback (strategy '{chosen_strategy}' unavailable)"

    def _analyze_game_situation(self, state: np.ndarray) -> Dict[str, Any]:
        """Analyze current game state with CENTER-ZERO coordinates"""
        if len(state) < 14:
            return {"situation": "unknown", "urgency": 0.0}

        # Extract key state components (CENTER-ZERO coordinates)
        ball_x = state[0]  # Ball X: -1.0 to +1.0 (center = 0.0)
        ball_y = state[1]  # Ball Y: -0.5 to +0.5 (center = 0.0)
        ball_dx = state[2]  # Ball X velocity (normalized)
        ball_dy = state[3]  # Ball Y velocity (normalized)
        paddle_y = state[4]  # AI paddle: -0.5 to +0.5 (center = 0.0)
        distance_to_ai = state[5]  # Distance to AI side

        # Additional strategic info
        ball_approaching = state[7] if len(state) > 7 else (ball_dx > 0.5)
        urgent_situation = state[13] if len(state) > 13 else (distance_to_ai < 0.2 and ball_approaching)

        # Determine situation type with center-zero awareness
        situation_type = "neutral"
        urgency_level = 0.0

        # RESET BALL DETECTION (center is now 0.0, 0.0!)
        ball_at_center = (abs(ball_x) < 0.1 and abs(ball_y) < 0.1)
        ball_near_center = (abs(ball_x) < 0.2 and abs(ball_y) < 0.15)

        if ball_at_center:
            situation_type = "reset_ball_critical"
            urgency_level = 0.95
        elif ball_near_center:
            situation_type = "reset_ball_detected"
            urgency_level = 0.8
        elif urgent_situation:
            situation_type = "critical_defense"
            urgency_level = 0.9
        elif ball_approaching and distance_to_ai < 0.3:
            situation_type = "incoming_ball"
            urgency_level = 0.7
        elif ball_approaching and distance_to_ai < 0.5:
            situation_type = "preparation"
            urgency_level = 0.4
        elif not ball_approaching:
            situation_type = "positioning"
            urgency_level = 0.2

        return {
            "situation": situation_type,
            "urgency": urgency_level,
            "ball_position": (ball_x, ball_y),
            "ball_velocity": (ball_dx, ball_dy),
            "paddle_position": paddle_y,
            "ball_approaching": ball_approaching,
            "distance_to_ai": distance_to_ai,
            "ball_at_center": ball_at_center,
            "ball_near_center": ball_near_center
        }

    def _select_applicable_strategies(self, app_context: Dict, game_situation: Dict) -> List[str]:
        """Select which strategies are applicable to current situation"""
        applicable = []

        # Get all available strategies
        env_strategies = []
        if 'environment_context' in app_context:
            env_context = app_context['environment_context']
            strategic_concepts = env_context.get('strategic_concepts', {})
            env_strategies.extend(strategic_concepts.get('core_skills', []))
            env_strategies.extend(strategic_concepts.get('tactical_approaches', []))
            env_strategies.extend(strategic_concepts.get('success_patterns', []))

        learned_strategies = app_context.get('strategies', [])
        all_strategies = env_strategies + learned_strategies

        # Filter by situation appropriateness
        situation = game_situation['situation']
        urgency = game_situation['urgency']

        for strategy in all_strategies:
            if self._is_strategy_applicable(strategy, situation, urgency):
                applicable.append(strategy)

        return applicable

    def _is_strategy_applicable(self, strategy: str, situation: str, urgency: float) -> bool:
        """Determine if a strategy is applicable to current situation"""
        strategy_lower = strategy.lower()
        urgency = float(urgency)  # 🔧 FIX: Ensure float type

        # Critical defense situations
        if situation == "critical_defense":
            return any(keyword in strategy_lower for keyword in [
                "defensive", "intercept", "positioning", "timing", "early"
            ])

        # Incoming ball situations
        elif situation == "incoming_ball":
            return any(keyword in strategy_lower for keyword in [
                "trajectory", "prediction", "timing", "angle", "intercept"
            ])

        # Preparation situations
        elif situation == "preparation":
            return any(keyword in strategy_lower for keyword in [
                "positioning", "predictive", "anticipate", "movement"
            ])

        # General positioning
        elif situation == "positioning":
            return any(keyword in strategy_lower for keyword in [
                "positioning", "movement", "defensive", "stability"
            ])

        return True  # Default: all strategies applicable

    def _choose_best_strategy(self, strategies: List[str], game_situation: Dict) -> str:
        """Choose the most appropriate strategy for current situation"""
        if not strategies:
            return ""

        # Prioritize by situation urgency
        urgency = float(game_situation['urgency'])  # 🔧 FIX: Ensure float for comparison
        situation = game_situation['situation']

        # High urgency: prioritize defensive/timing strategies
        if urgency > 0.7:
            defensive_strategies = [s for s in strategies if any(word in s.lower()
                                                                 for word in
                                                                 ['defensive', 'timing', 'intercept', 'reaction'])]
            if defensive_strategies:
                return random.choice(defensive_strategies)

        # Medium urgency: prioritize predictive/positioning strategies
        elif urgency > 0.4:
            predictive_strategies = [s for s in strategies if any(word in s.lower()
                                                                  for word in
                                                                  ['predictive', 'trajectory', 'positioning',
                                                                   'anticipate'])]
            if predictive_strategies:
                return random.choice(predictive_strategies)

        # Low urgency: prioritize offensive/angle strategies
        else:
            offensive_strategies = [s for s in strategies if any(word in s.lower()
                                                                 for word in
                                                                 ['angle', 'aggressive', 'offensive', 'counter'])]
            if offensive_strategies:
                return random.choice(offensive_strategies)

        # Fallback: random choice
        return random.choice(strategies)

    # Strategy Implementation Methods - 🔧 FIX: Consistent signatures
    def _apply_trajectory_prediction(self, state: np.ndarray, q_values: np.ndarray,
                                     game_situation: Dict) -> Tuple[int, float]:
        """Apply ball trajectory prediction with CENTER-ZERO coordinates"""
        ball_y = game_situation['ball_position'][1]  # -0.5 to +0.5
        ball_dy = game_situation['ball_velocity'][1]  # velocity
        paddle_y = game_situation['paddle_position']  # -0.5 to +0.5

        # Predict where ball will be (center-zero)
        predicted_y = ball_y + (ball_dy * 0.3)

        # Choose action to move toward predicted position (center = 0.0)
        if predicted_y > paddle_y + 0.1:
            return 2, 0.8  # Move down (toward positive)
        elif predicted_y < paddle_y - 0.1:
            return 0, 0.8  # Move up (toward negative)
        else:
            return 1, 0.9  # Stay (already positioned well)

    def _apply_optimal_positioning(self, state: np.ndarray, q_values: np.ndarray,
                                   game_situation: Dict) -> Tuple[int, float]:
        """Apply optimal paddle positioning with CENTER-ZERO coordinates"""
        ball_y = game_situation['ball_position'][1]  # -0.5 to +0.5
        paddle_y = game_situation['paddle_position']  # -0.5 to +0.5

        # Target center-ball alignment (center is now 0.0!)
        target_position = ball_y
        position_error = target_position - paddle_y

        if abs(position_error) < 0.05:
            return 1, 0.9  # Stay - good position
        elif position_error > 0:
            return 2, 0.7  # Move down (toward positive)
        else:
            return 0, 0.7  # Move up (toward negative)

    def _apply_timing_optimization(self, state: np.ndarray, q_values: np.ndarray,
                                   game_situation: Dict) -> Tuple[int, float]:
        """Apply timing and reaction speed strategy"""
        urgency = float(game_situation['urgency'])  # 🔧 FIX: Ensure float for comparison

        if urgency > 0.8:
            # High urgency: favor quick decisive action
            best_action = np.argmax(q_values)
            return int(best_action), 0.85  # 🔧 FIX: Ensure int return
        else:
            # Lower urgency: favor positioning action
            return self._apply_optimal_positioning(state, q_values, game_situation)

    def _apply_defensive_positioning(self, state: np.ndarray, q_values: np.ndarray,
                                     game_situation: Dict) -> Tuple[int, float]:
        """Apply defensive positioning with CENTER-ZERO coordinates"""
        ball_y = game_situation['ball_position'][1]  # -0.5 to +0.5
        paddle_y = game_situation['paddle_position']  # -0.5 to +0.5

        # Defensive strategy: stay between ball and goal (CENTER = 0.0)
        target_y = ball_y * 0.7 + 0.0 * 0.3  # Weighted toward ball and CENTER (0.0)

        if target_y > paddle_y + 0.08:
            return 2, 0.75  # Move down (toward positive)
        elif target_y < paddle_y - 0.08:
            return 0, 0.75  # Move up (toward negative)
        else:
            return 1, 0.8  # Stay

    def _apply_predictive_movement(self, state: np.ndarray, q_values: np.ndarray,
                                   game_situation: Dict) -> Tuple[int, float]:
        """Apply predictive movement strategy"""
        return self._apply_trajectory_prediction(state, q_values, game_situation)

    def _apply_early_positioning(self, state: np.ndarray, q_values: np.ndarray,
                                 game_situation: Dict) -> Tuple[int, float]:
        """Apply early positioning beats reactive movement"""
        urgency = float(game_situation['urgency'])  # 🔧 FIX: Ensure float for comparison

        if urgency < 0.5:
            # Early positioning when not urgent
            return self._apply_trajectory_prediction(state, q_values, game_situation)
        else:
            # React appropriately when urgent
            return self._apply_optimal_positioning(state, q_values, game_situation)

    def _apply_aggressive_returns(self, state: np.ndarray, q_values: np.ndarray,
                                  game_situation: Dict) -> Tuple[int, float]:
        """Apply aggressive returns using paddle edges - FIXED for center-zero"""
        ball_y = float(game_situation['ball_position'][1])  # -0.5 to +0.5
        paddle_y = float(game_situation['paddle_position'])  # -0.5 to +0.5
        urgency = float(game_situation['urgency'])  # 🔧 FIX: Ensure float

        if game_situation['ball_approaching'] and urgency > 0.6:
            # Try to hit with paddle edge for angle (CENTER-ZERO: 0.0 is center)
            if ball_y > 0.0:  # Ball in upper half (positive)
                return 0, 0.6  # Move up to hit with top edge
            else:  # Ball in lower half (negative)
                return 2, 0.6  # Move down to hit with bottom edge

        return self._apply_optimal_positioning(state, q_values, game_situation)

    # Failure avoidance methods
    def _apply_failure_avoidance(self, action: int, state: np.ndarray,
                                 game_situation: Dict, app_context: Dict) -> int:
        """Apply failure pattern avoidance - FIXED for center-zero"""

        # Get failure patterns from environmental context
        failure_patterns = {}
        if 'environment_context' in app_context:
            failure_patterns = app_context['environment_context'].get('failure_patterns', {})

        urgency = float(game_situation['urgency'])  # 🔧 FIX: Ensure float for comparison
        paddle_y = float(game_situation['paddle_position'])  # 🔧 FIX: Ensure float for comparison

        # Avoid reactive play
        if 'reactive_play' in failure_patterns and urgency < 0.3:
            return self._avoid_reactive_play(action, state, game_situation)

        # Avoid edge camping - FIXED for center-zero coordinates
        if paddle_y < -0.35 or paddle_y > 0.35:  # FIXED: edges are now -0.35 and +0.35
            return self._avoid_edge_camping(action, state, game_situation)

        return action

    def _avoid_reactive_play(self, action: int, state: np.ndarray, game_situation: Dict) -> int:
        """Avoid reactive play pattern"""
        urgency = float(game_situation['urgency'])  # 🔧 FIX: Ensure float for comparison

        # Encourage proactive positioning instead of waiting
        if urgency < 0.3:
            ball_y = float(game_situation['ball_position'][1])
            paddle_y = float(game_situation['paddle_position'])

            # Move toward ball proactively
            if abs(ball_y - paddle_y) > 0.15:
                return 2 if ball_y > paddle_y else 0

        return action

    def _avoid_edge_camping(self, action, state, game_situation):
        paddle_y = float(game_situation['paddle_position'])
        ball_y = float(game_situation['ball_position'][1])

        # Allow edge camping IF ball is near the edge too
        if paddle_y > 0.35:
            # Only override if the ball is NOT in the bottom quarter
            if ball_y < 0.3:  # If ball is far from bottom
                return 0  # Move up toward center
            # If ball is near the bottom, allow agent to stay
        if paddle_y < -0.35:
            if ball_y > -0.3:  # If ball is far from top
                return 2  # Move down toward center

        return action

    def _avoid_over_correction(self, action: int, state: np.ndarray, game_situation: Dict) -> int:
        """Avoid making too many rapid movements"""
        # This would require tracking recent actions - simplified for now
        return action

    def _avoid_passive_defense(self, action: int, state: np.ndarray, game_situation: Dict) -> int:
        """Avoid passive defense - transition to offense when possible"""
        urgency = float(game_situation['urgency'])  # 🔧 FIX: Ensure float for comparison

        if urgency < 0.4 and not game_situation['ball_approaching']:
            # Not defensive situation - can be more aggressive
            return action  # Allow more aggressive positioning

        return action

    # 🔧 FIX: Helper methods with consistent signatures
    def _apply_angle_control(self, state: np.ndarray, q_values: np.ndarray, game_situation: Dict) -> Tuple[int, float]:
        return self._apply_aggressive_returns(state, q_values, game_situation)

    def _apply_counter_attacking(self, state: np.ndarray, q_values: np.ndarray, game_situation: Dict) -> Tuple[
        int, float]:
        return self._apply_aggressive_returns(state, q_values, game_situation)

    def _apply_ball_tracking(self, state: np.ndarray, q_values: np.ndarray, game_situation: Dict) -> Tuple[int, float]:
        return self._apply_trajectory_prediction(state, q_values, game_situation)

    def _apply_angle_shots(self, state: np.ndarray, q_values: np.ndarray, game_situation: Dict) -> Tuple[int, float]:
        return self._apply_aggressive_returns(state, q_values, game_situation)

    def _apply_early_interception(self, state: np.ndarray, q_values: np.ndarray, game_situation: Dict) -> Tuple[
        int, float]:
        return self._apply_early_positioning(state, q_values, game_situation)

    def _apply_consistency_focus(self, state: np.ndarray, q_values: np.ndarray, game_situation: Dict) -> Tuple[
        int, float]:
        return self._apply_optimal_positioning(state, q_values, game_situation)

    def _apply_offensive_angles(self, state: np.ndarray, q_values: np.ndarray, game_situation: Dict) -> Tuple[
        int, float]:
        return self._apply_aggressive_returns(state, q_values, game_situation)

    def _apply_defense_to_offense(self, state: np.ndarray, q_values: np.ndarray, game_situation: Dict) -> Tuple[
        int, float]:
        urgency = float(game_situation['urgency'])  # 🔧 FIX: Ensure float for comparison

        if urgency > 0.7:
            return self._apply_defensive_positioning(state, q_values, game_situation)
        else:
            return self._apply_aggressive_returns(state, q_values, game_situation)


class SymbolicDecisionMaker:
    """High-level decision maker that uses symbolic knowledge"""

    def __init__(self):
        self.knowledge_interpreter = KnowledgeInterpreter()
        self.decision_history = []
        self.strategy_effectiveness = {}

    def make_informed_decision(self, state: np.ndarray, q_values: np.ndarray,
                               app_context: Dict, exploration_rate: float) -> Tuple[int, str]:
        """Make decision using both neural network and symbolic knowledge"""

        # Get neural network recommendation
        nn_action = np.argmax(q_values)

        # Decide whether to use symbolic knowledge
        use_symbolic = self._should_use_symbolic_knowledge(app_context, exploration_rate)

        if use_symbolic:
            # Apply symbolic knowledge
            symbolic_action, reasoning = self.knowledge_interpreter.apply_knowledge(
                state, q_values, app_context, exploration_rate
            )

            # Track decision
            decision_info = {
                'nn_action': int(nn_action),  # 🔧 FIX: Ensure int
                'symbolic_action': int(symbolic_action),  # 🔧 FIX: Ensure int
                'reasoning': reasoning,
                'chosen': 'symbolic'
            }
            self.decision_history.append(decision_info)

            return int(symbolic_action), f"🧩 {reasoning}"  # 🔧 FIX: Ensure int return

        else:
            # Use neural network decision
            decision_info = {
                'nn_action': int(nn_action),  # 🔧 FIX: Ensure int
                'symbolic_action': None,
                'reasoning': "Neural network decision",
                'chosen': 'neural'
            }
            self.decision_history.append(decision_info)

            return int(nn_action), "🧠 Neural network decision"  # 🔧 FIX: Ensure int return

    def _should_use_symbolic_knowledge(self, app_context: Dict, exploration_rate: float) -> bool:
        """Decide whether to use symbolic knowledge or pure neural network"""

        if not app_context:
            return False

        # More likely to use symbolic knowledge when:
        # 1. Have learned strategies
        # 2. Lower exploration (more confident)
        # 3. Environmental context available

        strategies_available = len(app_context.get('strategies', []))
        lessons_available = len(app_context.get('lessons', []))
        has_env_context = 'environment_context' in app_context

        # Base probability
        base_prob = 0.15

        # Increase probability based on available knowledge
        knowledge_factor = min(0.3, (strategies_available + lessons_available) * 0.05)

        # Increase probability when less exploring (more exploitation)
        confidence_factor = max(0, 0.4 - exploration_rate)

        # Bonus for environmental context
        env_bonus = 0.1 if has_env_context else 0

        total_probability = base_prob + knowledge_factor + confidence_factor + env_bonus

        return random.random() < total_probability

    def update_strategy_effectiveness(self, reward: float):
        """Update effectiveness tracking based on reward received"""
        if not self.decision_history:
            return

        recent_decision = self.decision_history[-1]
        strategy_type = recent_decision['chosen']

        if strategy_type not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy_type] = []

        self.strategy_effectiveness[strategy_type].append(reward)

        # Keep only recent effectiveness data
        if len(self.strategy_effectiveness[strategy_type]) > 50:
            self.strategy_effectiveness[strategy_type] = self.strategy_effectiveness[strategy_type][-25:]

    def get_strategy_performance_summary(self) -> Dict[str, float]:
        """Get summary of strategy performance"""
        summary = {}

        for strategy_type, rewards in self.strategy_effectiveness.items():
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                summary[strategy_type] = round(avg_reward, 3)

        return summary


# Example integration test
if __name__ == "__main__":
    print("🧪 Testing Knowledge Application System...")

    # Create test environment
    decision_maker = SymbolicDecisionMaker()

    # Mock app context with strategies
    test_context = {
        'strategies': ['move toward ball path', 'angle shots'],
        'lessons': ['Early positioning beats reactive movement'],
        'environment_context': {
            'strategic_concepts': {
                'core_skills': ['Ball trajectory prediction', 'Optimal paddle positioning'],
                'tactical_approaches': ['Defensive positioning - stay between ball and goal']
            },
            'failure_patterns': {
                'reactive_play': {'description': 'Moving only after ball reaches AI side'}
            }
        }
    }

    # Test decision making
    test_state = np.array([0.0, 0.0, 0.1, 0.05, 0.0, 0.3, 0.6, 0.8, 0.2, 0.1, 0.5, 0.3, 0.4, 0.9])
    test_q_values = np.array([0.2, 0.8, 0.3])

    for i in range(5):
        action, reasoning = decision_maker.make_informed_decision(
            test_state, test_q_values, test_context, exploration_rate=0.3
        )
        print(f"Decision {i + 1}: Action={action}, {reasoning}")

        # Simulate reward feedback
        reward = random.uniform(-1, 2)
        decision_maker.update_strategy_effectiveness(reward)

    # Show performance summary
    print(f"\n📊 Strategy Performance: {decision_maker.get_strategy_performance_summary()}")
    print("✅ Knowledge Application System test complete!")