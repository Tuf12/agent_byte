"""
State interpreter for understanding environment states.

This module analyzes and interprets state vectors to extract meaningful
information without any hard-coded environment knowledge.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import deque


class StateInterpreter:
    """
    Interprets environment states to extract meaningful features.

    This interpreter analyzes state patterns to understand:
    - State structure and dimensions
    - Dynamic vs. static components
    - Correlations and relationships
    - Semantic meaning through observation
    """

    def __init__(self, state_history_size: int = 1000):
        """
        Initialize the state interpreter.

        Args:
            state_history_size: Number of states to keep in history
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state_history_size = state_history_size

        # State analysis data
        self.state_history = deque(maxlen=state_history_size)
        self.dimension_stats = {}
        self.dimension_interpretations = {}
        self.state_clusters = []

        # Analysis results cache
        self.analysis_cache = {
            'last_update': 0,
            'dimension_types': {},
            'relationships': {},
            'semantic_groups': []
        }

    def add_state(self, state: np.ndarray, reward: float = 0.0,
                  action: Optional[int] = None):
        """
        Add a state to the interpreter for analysis.

        Args:
            state: State vector to analyze
            reward: Reward associated with this state
            action: Action taken from this state
        """
        state_record = {
            'state': state.copy(),
            'reward': reward,
            'action': action,
            'timestamp': len(self.state_history)
        }

        self.state_history.append(state_record)

        # Update dimension statistics incrementally
        self._update_dimension_stats(state)

        # Periodically update full analysis
        if len(self.state_history) % 100 == 0:
            self._update_analysis()

    def interpret_state(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Interpret a single state vector.

        Args:
            state: State to interpret

        Returns:
            Interpretation including features and semantic meaning
        """
        interpretation = {'raw_state': state, 'dimension_count': len(state), 'features': {
            'mean': float(np.mean(state)),
            'std': float(np.std(state)),
            'min': float(np.min(state)),
            'max': float(np.max(state)),
            'zero_count': int(np.sum(state == 0)),
            'sparsity': float(np.sum(state == 0) / len(state))
        }, 'anomalies': [], 'semantic_interpretation': {}}

        # Extract basic features

        # Dimension-wise interpretation
        dim_interpretations = []
        for i, value in enumerate(state):
            dim_type = self._get_dimension_type(i)
            dim_interpretation = {
                'index': i,
                'value': float(value),
                'type': dim_type,
                'normalized_value': self._normalize_dimension_value(i, value)
            }

            # Check for anomalies
            if self._is_anomalous_value(i, value):
                interpretation['anomalies'].append({
                    'dimension': i,
                    'value': float(value),
                    'reason': 'Outside normal range'
                })

            dim_interpretations.append(dim_interpretation)

        interpretation['dimension_interpretations'] = dim_interpretations

        # Semantic interpretation
        interpretation['semantic_interpretation'] = self._generate_semantic_interpretation(state)

        return interpretation

    def analyze_state_structure(self) -> Dict[str, Any]:
        """
        Analyze the overall structure of states seen so far.

        Returns:
            Comprehensive analysis of state structure
        """
        if len(self.state_history) < 10:
            return {'status': 'insufficient_data'}

        states = np.array([record['state'] for record in self.state_history])

        analysis = {
            'dimension_count': states.shape[1],
            'sample_count': states.shape[0],
            'dimension_analysis': {},
            'correlations': {},
            'semantic_groups': [],
            'dynamics': {}
        }

        # Analyze each dimension
        for dim in range(states.shape[1]):
            dim_values = states[:, dim]

            dim_analysis = {
                'type': self._classify_dimension_type(dim_values),
                'stats': {
                    'mean': float(np.mean(dim_values)),
                    'std': float(np.std(dim_values)),
                    'min': float(np.min(dim_values)),
                    'max': float(np.max(dim_values))
                },
                'variability': float(np.std(dim_values) / (np.mean(np.abs(dim_values)) + 1e-8)),
                'is_static': bool(np.std(dim_values) < 0.001),
                'is_discrete': self._is_discrete(dim_values)
            }

            analysis['dimension_analysis'][dim] = dim_analysis

        # Analyze correlations
        if states.shape[0] > 20:
            correlation_matrix = np.corrcoef(states.T)
            analysis['correlations'] = self._extract_significant_correlations(correlation_matrix)

        # Identify semantic groups
        analysis['semantic_groups'] = self._identify_semantic_groups(states)

        # Analyze dynamics
        analysis['dynamics'] = self._analyze_dynamics(states)

        # Cache the analysis
        self.analysis_cache['last_update'] = len(self.state_history)
        self.analysis_cache['dimension_types'] = {
            dim: info['type'] for dim, info in analysis['dimension_analysis'].items()
        }
        self.analysis_cache['semantic_groups'] = analysis['semantic_groups']

        return analysis

    def _update_dimension_stats(self, state: np.ndarray):
        """Update running statistics for each dimension."""
        for i, value in enumerate(state):
            if i not in self.dimension_stats:
                self.dimension_stats[i] = {
                    'count': 0,
                    'mean': 0.0,
                    'M2': 0.0,  # For variance calculation
                    'min': value,
                    'max': value,
                    'unique_values': set()
                }

            stats = self.dimension_stats[i]
            stats['count'] += 1

            # Update mean and variance using Welford's algorithm
            delta = value - stats['mean']
            stats['mean'] += delta / stats['count']
            delta2 = value - stats['mean']
            stats['M2'] += delta * delta2

            # Update min/max
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)

            # Track unique values (for discrete detection)
            if stats['count'] < 100:  # Only track early on
                stats['unique_values'].add(float(value))

    def _update_analysis(self):
        """Periodically update full analysis."""
        self.analyze_state_structure()

    def _classify_dimension_type(self, values: np.ndarray) -> str:
        """Classify what type of information a dimension represents."""
        # Check if constant
        if np.std(values) < 0.001:
            return 'constant'

        # Check if binary
        unique_values = np.unique(values)
        if len(unique_values) == 2:
            return 'binary'

        # Check if discrete
        if self._is_discrete(values):
            return 'discrete'

        # Check if bounded
        value_range = np.max(values) - np.min(values)
        if -1.1 <= np.min(values) and np.max(values) <= 1.1:
            return 'normalized'

        # Check if cyclic (e.g., angles)
        if self._is_cyclic(values):
            return 'cyclic'

        # Check variability patterns
        variability = np.std(values) / (np.mean(np.abs(values)) + 1e-8)

        if variability < 0.1:
            return 'stable_continuous'
        elif variability < 0.5:
            return 'moderate_continuous'
        else:
            return 'volatile_continuous'

    def _is_discrete(self, values: np.ndarray) -> bool:
        """Check if values are discrete."""
        unique_values = np.unique(values)

        # If few unique values relative to samples
        if len(unique_values) < len(values) * 0.1:
            return True

        # Check if all values are close to integers
        if all(np.abs(v - round(v)) < 0.001 for v in unique_values[:20]):
            return True

        return False

    def _is_cyclic(self, values: np.ndarray) -> bool:
        """Check if values represent cyclic data (like angles)."""
        # Simple heuristic: check if values wrap around
        if len(values) < 50:
            return False

        # Look for sudden jumps that might indicate wrapping
        diffs = np.diff(values)
        large_jumps = np.abs(diffs) > (np.max(values) - np.min(values)) * 0.8

        return np.sum(large_jumps) > len(values) * 0.02

    def _get_dimension_type(self, dim: int) -> str:
        """Get a cached dimension type."""
        if dim in self.analysis_cache['dimension_types']:
            return self.analysis_cache['dimension_types'][dim]

        # Fallback to simple classification
        if dim in self.dimension_stats:
            stats = self.dimension_stats[dim]
            if stats['count'] > 1:
                variance = stats['M2'] / (stats['count'] - 1)
                if variance < 0.001:
                    return 'constant'

        return 'unknown'

    def _normalize_dimension_value(self, dim: int, value: float) -> float:
        """Normalize a dimension value based on observed range."""
        if dim not in self.dimension_stats:
            return 0.0

        stats = self.dimension_stats[dim]
        value_range = stats['max'] - stats['min']

        if value_range < 0.001:
            return 0.0

        return (value - stats['min']) / value_range

    def _is_anomalous_value(self, dim: int, value: float) -> bool:
        """Check if a value is anomalous for a dimension."""
        if dim not in self.dimension_stats or self.dimension_stats[dim]['count'] < 20:
            return False

        stats = self.dimension_stats[dim]
        if stats['count'] > 1:
            variance = stats['M2'] / (stats['count'] - 1)
            std = np.sqrt(variance)

            # Check if value is more than 3 standard deviations from mean
            if np.abs(value - stats['mean']) > 3 * std:
                return True

        return False

    def _extract_significant_correlations(self, corr_matrix: np.ndarray,
                                          threshold: float = 0.7) -> Dict[str, List[Tuple[int, int, float]]]:
        """Extract significant correlations from correlation matrix."""
        correlations = {
            'positive': [],
            'negative': []
        }

        n_dims = corr_matrix.shape[0]

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                corr = corr_matrix[i, j]

                if corr > threshold:
                    correlations['positive'].append((i, j, float(corr)))
                elif corr < -threshold:
                    correlations['negative'].append((i, j, float(corr)))

        # Sort by correlation strength
        correlations['positive'].sort(key=lambda x: abs(x[2]), reverse=True)
        correlations['negative'].sort(key=lambda x: abs(x[2]), reverse=True)

        return correlations

    def _identify_semantic_groups(self, states: np.ndarray) -> List[Dict[str, Any]]:
        """Identify groups of dimensions that might represent related information."""
        groups = []

        # Group 1: Position-like dimensions (low variability, continuous)
        position_dims = []
        for dim, analysis in self.analysis_cache.get('dimension_types', {}).items():
            if analysis in ['stable_continuous', 'normalized']:
                position_dims.append(dim)

        if position_dims:
            groups.append({
                'name': 'position_like',
                'dimensions': position_dims,
                'description': 'Dimensions with stable, continuous values'
            })

        # Group 2: Velocity-like dimensions (moderate variability)
        velocity_dims = []
        for dim, analysis in self.analysis_cache.get('dimension_types', {}).items():
            if analysis == 'moderate_continuous':
                velocity_dims.append(dim)

        if velocity_dims:
            groups.append({
                'name': 'velocity_like',
                'dimensions': velocity_dims,
                'description': 'Dimensions with moderate variability'
            })

        # Group 3: Discrete state indicators
        discrete_dims = []
        for dim, analysis in self.analysis_cache.get('dimension_types', {}).items():
            if analysis in ['binary', 'discrete']:
                discrete_dims.append(dim)

        if discrete_dims:
            groups.append({
                'name': 'state_indicators',
                'dimensions': discrete_dims,
                'description': 'Discrete or binary state indicators'
            })

        return groups

    def _analyze_dynamics(self, states: np.ndarray) -> Dict[str, Any]:
        """Analyze dynamic properties of state evolution."""
        if len(states) < 10:
            return {}

        dynamics = {
            'change_rates': {},
            'periodicities': {},
            'trends': {}
        }

        # Analyze change rates for each dimension
        for dim in range(states.shape[1]):
            dim_values = states[:, dim]

            # Skip constant dimensions
            if np.std(dim_values) < 0.001:
                continue

            # Calculate change rate
            changes = np.abs(np.diff(dim_values))
            avg_change = np.mean(changes)

            dynamics['change_rates'][dim] = {
                'average': float(avg_change),
                'variability': float(np.std(changes))
            }

            # Simple trend detection
            if len(dim_values) > 20:
                # Linear regression for trend
                x = np.arange(len(dim_values))
                slope = np.polyfit(x, dim_values, 1)[0]

                dynamics['trends'][dim] = {
                    'slope': float(slope),
                    'direction': 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable')
                }

        return dynamics

    def _generate_semantic_interpretation(self, state: np.ndarray) -> Dict[str, Any]:
        """Generate semantic interpretation of state."""
        interpretation = {}

        # Use semantic groups if available
        for group in self.analysis_cache.get('semantic_groups', []):
            group_name = group['name']
            group_dims = group['dimensions']

            if group_dims:
                # Extract values for this group
                group_values = [state[dim] for dim in group_dims if dim < len(state)]

                if group_values:
                    interpretation[group_name] = {
                        'values': group_values,
                        'summary': {
                            'mean': float(np.mean(group_values)),
                            'magnitude': float(np.linalg.norm(group_values))
                        }
                    }

        return interpretation

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of state interpretation findings."""
        if len(self.state_history) < 10:
            return {'status': 'insufficient_data'}

        # Get latest analysis
        analysis = self.analyze_state_structure()

        summary = {
            'states_analyzed': len(self.state_history),
            'dimension_count': analysis['dimension_count'],
            'dimension_types': {},
            'static_dimensions': [],
            'dynamic_dimensions': [],
            'correlated_dimensions': [],
            'semantic_understanding': []
        }

        # Summarize dimension types
        type_counts = {}
        for dim, dim_info in analysis['dimension_analysis'].items():
            dim_type = dim_info['type']
            type_counts[dim_type] = type_counts.get(dim_type, 0) + 1

            if dim_info['is_static']:
                summary['static_dimensions'].append(dim)
            else:
                summary['dynamic_dimensions'].append(dim)

        summary['dimension_types'] = type_counts

        # Add correlated dimensions
        if 'correlations' in analysis:
            for corr_list in analysis['correlations'].values():
                for dim1, dim2, corr_value in corr_list[:5]:  # Top 5
                    summary['correlated_dimensions'].append({
                        'dims': (dim1, dim2),
                        'correlation': corr_value
                    })

        # Add semantic understanding
        for group in analysis.get('semantic_groups', []):
            summary['semantic_understanding'].append({
                'group': group['name'],
                'dimension_count': len(group['dimensions']),
                'description': group['description']
            })

        return summary