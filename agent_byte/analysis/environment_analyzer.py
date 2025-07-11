"""
Environment analyzer for automatic environment understanding.

This module implements the auto-detection system that learns about environments
through exploration and analysis, without requiring hard-coded knowledge.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import defaultdict, deque

from ..core.interfaces import Environment


class EnvironmentAnalyzer:
    """
    Analyzes environments to automatically detect their characteristics.

    This system learns about environments through exploration, detecting:
    - State structure and meaning
    - Action effects
    - Reward patterns
    - Objectives and rules
    """

    def __init__(self, exploration_episodes: int = 10):
        """
        Initialize the environment analyzer.

        Args:
            exploration_episodes: Number of episodes to explore for analysis
        """
        self.exploration_episodes = exploration_episodes
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze(self, env: Environment,
                provided_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze an environment to understand its characteristics.

        Args:
            env: Environment to analyze
            provided_metadata: Optional metadata provided about the environment

        Returns:
            Comprehensive analysis of the environment
        """
        self.logger.info(f"Analyzing environment: {env.get_id()}")

        # Start with provided metadata if available
        analysis = {
            'env_id': env.get_id(),
            'state_size': env.get_state_size(),
            'action_size': env.get_action_size(),
            'provided_metadata': provided_metadata or {}
        }

        # Get environment's own metadata if available
        env_metadata = env.get_metadata()
        if env_metadata:
            analysis['env_metadata'] = env_metadata

        # Perform exploratory analysis
        exploration_data = self._explore_environment(env)

        # Analyze different aspects
        analysis['state_analysis'] = self._analyze_state_structure(exploration_data)
        analysis['action_analysis'] = self._analyze_action_effects(exploration_data)
        analysis['reward_analysis'] = self._analyze_reward_patterns(exploration_data)
        analysis['dynamics_analysis'] = self._analyze_dynamics(exploration_data)

        # Infer high-level characteristics
        analysis['inferred_objectives'] = self._infer_objectives(exploration_data, analysis)
        analysis['inferred_rules'] = self._infer_rules(exploration_data, analysis)
        analysis['environment_type'] = self._classify_environment_type(analysis)

        # Generate understanding summary
        analysis['understanding'] = self._generate_understanding(analysis)

        return analysis

    def _explore_environment(self, env: Environment) -> Dict[str, List]:
        """
        Explore the environment to collect data for analysis.
        """
        exploration_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'transitions': [],
            'episodes': []
        }

        for episode in range(self.exploration_episodes):
            episode_data = {
                'states': [],
                'actions': [],
                'rewards': [],
                'length': 0,
                'total_reward': 0
            }

            state = env.reset()
            done = False
            step_count = 0

            while not done and step_count < 1000:  # Prevent infinite episodes
                # Exploration policy: random actions
                action = np.random.randint(0, env.get_action_size())

                next_state, reward, done, info = env.step(action)

                # Record transition
                transition = {
                    'state': state.copy(),
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done,
                    'info': info
                }

                exploration_data['states'].append(state)
                exploration_data['actions'].append(action)
                exploration_data['rewards'].append(reward)
                exploration_data['next_states'].append(next_state)
                exploration_data['dones'].append(done)
                exploration_data['transitions'].append(transition)

                episode_data['states'].append(state)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['length'] += 1
                episode_data['total_reward'] += reward

                state = next_state
                step_count += 1

            exploration_data['episodes'].append(episode_data)

        return exploration_data

    def _analyze_state_structure(self, data: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze the structure and meaning of state dimensions.
        """
        states = np.array(data['states'])
        if len(states) == 0:
            return {}

        analysis = {
            'num_dimensions': states.shape[1],
            'dimension_stats': [],
            'dimension_types': [],
            'correlations': None
        }

        # Analyze each dimension
        for dim in range(states.shape[1]):
            dim_values = states[:, dim]

            stats = {
                'index': dim,
                'mean': float(np.mean(dim_values)),
                'std': float(np.std(dim_values)),
                'min': float(np.min(dim_values)),
                'max': float(np.max(dim_values)),
                'range': float(np.max(dim_values) - np.min(dim_values))
            }

            # Infer dimension type based on characteristics
            dim_type = self._infer_dimension_type(dim_values, data)
            stats['inferred_type'] = dim_type

            analysis['dimension_stats'].append(stats)
            analysis['dimension_types'].append(dim_type)

        # Analyze correlations between dimensions
        if states.shape[0] > 10:
            analysis['correlations'] = np.corrcoef(states.T)

        return analysis

    def _infer_dimension_type(self, values: np.ndarray, data: Dict[str, List]) -> str:
        """
        Infer what type of information a state dimension represents.
        """
        # Check if it's constant
        if np.std(values) < 0.001:
            return 'constant'

        # Check if it's binary
        unique_values = np.unique(values)
        if len(unique_values) == 2:
            return 'binary'

        # Check if it's discrete
        if len(unique_values) < 10 and all(v == int(v) for v in unique_values):
            return 'discrete'

        # Check if it's bounded
        value_range = np.max(values) - np.min(values)
        if -1.1 <= np.min(values) and np.max(values) <= 1.1:
            return 'normalized'

        # Check if it changes smoothly (likely position/velocity)
        if len(values) > 10:
            diffs = np.diff(values)
            smoothness = np.mean(np.abs(diffs)) / (value_range + 1e-8)
            if smoothness < 0.1:
                return 'position'
            elif smoothness < 0.3:
                return 'velocity'

        # Check if it's cyclic (angles)
        if value_range > 5 and np.abs(values[0] - values[-1]) < value_range * 0.1:
            return 'cyclic'

        return 'continuous'

    def _analyze_action_effects(self, data: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze what effects each action has on the state.
        """
        analysis = {
            'num_actions': len(set(data['actions'])),
            'action_effects': {},
            'action_frequencies': defaultdict(int)
        }

        # Count action frequencies
        for action in data['actions']:
            analysis['action_frequencies'][action] += 1

        # Analyze effect of each action
        for action in range(analysis['num_actions']):
            effects = []

            for trans in data['transitions']:
                if trans['action'] == action:
                    state_change = trans['next_state'] - trans['state']
                    effects.append(state_change)

            if effects:
                effects = np.array(effects)
                mean_effect = np.mean(effects, axis=0)
                std_effect = np.std(effects, axis=0)

                # Find dimensions most affected by this action
                significant_dims = np.where(np.abs(mean_effect) > 0.1)[0]

                analysis['action_effects'][action] = {
                    'mean_state_change': mean_effect,
                    'std_state_change': std_effect,
                    'primary_affected_dims': significant_dims.tolist(),
                    'effect_magnitude': float(np.linalg.norm(mean_effect))
                }

        return analysis

    def _analyze_reward_patterns(self, data: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze reward patterns to understand objectives.
        """
        rewards = np.array(data['rewards'])

        analysis = {
            'reward_stats': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'positive_ratio': float(np.sum(rewards > 0) / len(rewards))
            },
            'reward_types': self._classify_reward_structure(rewards),
            'episode_patterns': []
        }

        # Analyze episode-level patterns
        for ep in data['episodes']:
            ep_rewards = ep['rewards']
            if ep_rewards:
                pattern = {
                    'total_reward': ep['total_reward'],
                    'length': ep['length'],
                    'avg_reward': ep['total_reward'] / ep['length'],
                    'final_reward': ep_rewards[-1]
                }
                analysis['episode_patterns'].append(pattern)

        return analysis

    def _classify_reward_structure(self, rewards: np.ndarray) -> str:
        """
        Classify the type of reward structure.
        """
        unique_rewards = np.unique(rewards)

        if len(unique_rewards) == 1:
            return 'constant'
        elif len(unique_rewards) == 2:
            if 0 in unique_rewards:
                return 'sparse_binary'
            else:
                return 'binary'
        elif len(unique_rewards) < 10:
            return 'discrete'
        else:
            # Check if mostly sparse
            zero_ratio = np.sum(rewards == 0) / len(rewards)
            if zero_ratio > 0.9:
                return 'sparse_continuous'
            else:
                return 'dense_continuous'

    def _analyze_dynamics(self, data: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze environment dynamics (deterministic vs stochastic).
        """
        analysis = {
            'appears_deterministic': True,
            'state_transition_variance': [],
            'periodic_patterns': False
        }

        # Group transitions by (state, action) pairs
        transition_groups = defaultdict(list)
        for trans in data['transitions']:
            # Create a hashable key for (state, action)
            state_key = tuple(np.round(trans['state'], 3))
            key = (state_key, trans['action'])
            transition_groups[key].append(trans['next_state'])

        # Check variance in transitions
        variances = []
        for key, next_states in transition_groups.items():
            if len(next_states) > 1:
                next_states = np.array(next_states)
                variance = np.mean(np.var(next_states, axis=0))
                variances.append(variance)

                if variance > 0.01:
                    analysis['appears_deterministic'] = False

        if variances:
            analysis['state_transition_variance'] = variances

        return analysis

    def _infer_objectives(self, data: Dict[str, List], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer the objectives of the environment.
        """
        objectives = {
            'primary': 'Unknown',
            'success_indicators': [],
            'failure_indicators': []
        }

        reward_analysis = analysis['reward_analysis']

        # Infer based on reward patterns
        if reward_analysis['reward_types'] == 'sparse_binary':
            objectives['primary'] = 'Reach a specific goal state'
            objectives['success_indicators'] = ['Positive reward received']
        elif reward_analysis['reward_types'] == 'dense_continuous':
            if reward_analysis['reward_stats']['mean'] > 0:
                objectives['primary'] = 'Maximize cumulative reward'
            else:
                objectives['primary'] = 'Survive as long as possible'

        # Check for terminal states
        terminal_rewards = [data['rewards'][i] for i, done in enumerate(data['dones']) if done]
        if terminal_rewards:
            avg_terminal = np.mean(terminal_rewards)
            if avg_terminal < -1:
                objectives['failure_indicators'].append('Large negative reward at termination')
            elif avg_terminal > 1:
                objectives['success_indicators'].append('Large positive reward at termination')

        return objectives

    def _infer_rules(self, data: Dict[str, List], analysis: Dict[str, Any]) -> List[str]:
        """
        Infer rules and constraints of the environment.
        """
        rules = []

        # Check state bounds
        state_analysis = analysis['state_analysis']
        for dim_stat in state_analysis['dimension_stats']:
            if dim_stat['inferred_type'] == 'normalized':
                rules.append(
                    f"State dimension {dim_stat['index']} is bounded between {dim_stat['min']:.2f} and {dim_stat['max']:.2f}")

        # Check action effects
        action_analysis = analysis['action_analysis']
        for action, effects in action_analysis['action_effects'].items():
            if effects['effect_magnitude'] > 0.5:
                rules.append(
                    f"Action {action} has significant state impact (magnitude: {effects['effect_magnitude']:.2f})")

        # Check termination conditions
        terminal_states = [data['states'][i] for i, done in enumerate(data['dones']) if done]
        if terminal_states:
            rules.append(f"Environment can terminate (observed {len(terminal_states)} terminations)")

        return rules

    def _classify_environment_type(self, analysis: Dict[str, Any]) -> str:
        """
        Classify the type of environment based on analysis.
        """
        state_size = analysis['state_size']
        action_size = analysis['action_size']
        reward_type = analysis['reward_analysis']['reward_types']

        # Simple heuristics for classification
        if action_size == 2:
            if state_size < 10:
                return 'simple_control'
            else:
                return 'complex_control'
        elif action_size <= 5:
            if reward_type in ['sparse_binary', 'sparse_continuous']:
                return 'goal_oriented'
            else:
                return 'continuous_control'
        else:
            return 'complex_discrete'

    def _generate_understanding(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate human-readable understanding of the environment.
        """
        understanding = {
            'summary': f"Environment with {analysis['state_size']} state dimensions and {analysis['action_size']} actions",
            'dynamics': 'Deterministic' if analysis['dynamics_analysis']['appears_deterministic'] else 'Stochastic',
            'reward_structure': analysis['reward_analysis']['reward_types'],
            'complexity': 'Low' if analysis['state_size'] < 10 else 'High'
        }

        return understanding