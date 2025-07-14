"""
Transfer mapper for cross-environment knowledge transfer.

This module handles the mapping and adaptation of knowledge between
different environments without any hard-coded assumptions.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import time

from numpy import floating


class TransferMapper:
    """
    Maps knowledge between environments for transfer learning.

    This mapper identifies which skills and patterns can transfer between
    environments and how they need to be adapted.
    """

    def __init__(self):
        """Initialize the transfer mapper."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Transfer history
        self.transfer_history = []
        self.transfer_success_metrics = {}

        # NEW: Add validation tracking
        self.validation_metrics = {}
        self.transfer_validators = {}

        # Similarity thresholds
        self.thresholds = {
            'structural_similarity': 0.6,
            'behavioral_similarity': 0.5,
            'skill_compatibility': 0.7,
            'minimum_confidence': 0.5
        }

        # Adaptation strategies
        self.adaptation_strategies = {
            'direct_transfer': self._direct_transfer,
            'scaled_transfer': self._scaled_transfer,
            'abstract_transfer': self._abstract_transfer,
            'compositional_transfer': self._compositional_transfer
        }

    def map_knowledge(self, source_knowledge: Dict[str, Any],
                      target_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map knowledge from source to target environment.

        Args:
            source_knowledge: Knowledge from source environment including
                - skills
                - patterns
                - performance metrics
                - environmental understanding
            target_analysis: Analysis of target environment

        Returns:
            Mapped knowledge package for target environment
        """
        mapped_knowledge = {
            'transferable_skills': {},
            'adapted_patterns': [],
            'transfer_confidence': 0.0,
            'adaptation_recommendations': [],
            'mapping_metadata': {
                'source_env': source_knowledge.get('environment_id', 'unknown'),
                'target_env': target_analysis.get('env_id', 'unknown'),
                'timestamp': time.time()
            }
        }

        # Analyze environment compatibility
        compatibility = self._analyze_environment_compatibility(
            source_knowledge, target_analysis
        )

        # Map skills
        skill_mappings = self._map_skills(
            source_knowledge.get('transferable_skills', {}),
            target_analysis,
            compatibility
        )
        mapped_knowledge['transferable_skills'] = skill_mappings['mapped_skills']

        # Map patterns
        pattern_mappings = self._map_patterns(
            source_knowledge.get('patterns', []),
            target_analysis,
            compatibility
        )
        mapped_knowledge['adapted_patterns'] = pattern_mappings

        # Calculate overall transfer confidence
        mapped_knowledge['transfer_confidence'] = self._calculate_transfer_confidence(
            compatibility, skill_mappings, pattern_mappings
        )

        # Generate adaptation recommendations
        mapped_knowledge['adaptation_recommendations'] = self._generate_adaptation_recommendations(
            compatibility, skill_mappings
        )

        # Record transfer attempt
        self._record_transfer(mapped_knowledge)

        return mapped_knowledge

    def _analyze_environment_compatibility(self, source: Dict[str, Any],
                                           target: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compatibility between source and target environments."""
        compatibility = {}

        # Action space compatibility
        source_actions = source.get('action_size', 0)
        target_actions = target.get('action_size', 0)

        if source_actions > 0 and target_actions > 0:
            action_compat = min(source_actions, target_actions) / max(source_actions, target_actions)
        else:
            action_compat = 0.0

        compatibility['action_space_compatibility'] = action_compat

        # State space compatibility
        source_states = source.get('state_size', 0)
        target_states = target.get('state_size', 0)

        if source_states > 0 and target_states > 0:
            state_compat = min(source_states, target_states) / max(source_states, target_states)
        else:
            state_compat = 0.0

        compatibility['state_space_compatibility'] = state_compat

        # Reward type compatibility
        source_reward = source.get('reward_type', '')
        target_reward_analysis = target.get('reward_analysis', {})
        target_reward = target_reward_analysis.get('reward_types', '')

        if source_reward and target_reward:
            reward_compat = 1.0 if source_reward == target_reward else 0.5
        else:
            reward_compat = 0.5

        compatibility['reward_compatibility'] = reward_compat

        # Environment type compatibility
        source_env = source.get('environment_type', '')
        target_env = target.get('environment_type', '')

        if source_env and target_env:
            env_compat = 1.0 if source_env == target_env else 0.3
        else:
            env_compat = 0.5

        compatibility['environment_type_compatibility'] = env_compat

        # Add behavioral similarity (CRITICAL FIX for missing key)
        compatibility['behavioral_similarity'] = min(1.0, (action_compat + state_compat + env_compat) / 3.0)

        # Calculate overall compatibility with balanced weighting
        scores = [action_compat, state_compat, reward_compat, env_compat]
        weights = [0.3, 0.3, 0.2, 0.2]

        compatibility['overall_compatibility'] = sum(s * w for s, w in zip(scores, weights))

        return compatibility

    def _map_skills(self, source_skills: Dict[str, Any], target_analysis: Dict[str, Any],
                    compatibility: Dict[str, Any]) -> Dict[str, Any]:
        """Map skills from source to target environment."""
        mapped_skills = {}
        mapping_metadata = {
            'total_source_skills': len(source_skills),
            'successfully_mapped': 0,
            'mapping_strategies_used': {}
        }

        for skill_id, skill_data in source_skills.items():
            # Check if skill meets minimum confidence
            if skill_data.get('confidence', 0) < self.thresholds['minimum_confidence']:
                continue

            # Select mapping strategy based on skill type and compatibility
            strategy = self._select_mapping_strategy(skill_data, compatibility)

            # Apply mapping strategy
            mapped_skill = self.adaptation_strategies[strategy](
                skill_data, target_analysis, compatibility
            )

            if mapped_skill:
                mapped_skills[skill_id] = mapped_skill
                mapping_metadata['successfully_mapped'] += 1

                # Track strategy usage
                if strategy not in mapping_metadata['mapping_strategies_used']:
                    mapping_metadata['mapping_strategies_used'][strategy] = 0
                mapping_metadata['mapping_strategies_used'][strategy] += 1

        return {
            'mapped_skills': mapped_skills,
            'metadata': mapping_metadata
        }

    def _map_patterns(self, source_patterns: List[Dict[str, Any]], target_analysis: Dict[str, Any],
                      compatibility: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map behavioral patterns to target environment."""
        mapped_patterns = []

        for pattern in source_patterns:
            # Only map stable patterns
            if pattern.get('stability', 0) < 0.5:
                continue

            mapped_pattern = {
                'original_pattern': pattern,
                'adaptation_needed': compatibility['overall_compatibility'] < 0.8,
                'confidence_factor': compatibility['overall_compatibility'],
                'recommendations': []
            }

            # Add specific recommendations based on a pattern type
            pattern_type = pattern.get('type', 'unknown')

            if pattern_type == 'action_sequence' and compatibility['action_space_compatibility'] < 1.0:
                mapped_pattern['recommendations'].append(
                    'Action sequences may need adjustment for different action space'
                )

            if pattern_type == 'state_response' and compatibility['state_space_compatibility'] < 1.0:
                mapped_pattern['recommendations'].append(
                    'State-based responses should be validated in new state space'
                )

            mapped_patterns.append(mapped_pattern)

        return mapped_patterns

    def _select_mapping_strategy(self, skill_data: Dict[str, Any],
                                 compatibility: Dict[str, Any]) -> str:
        """Select the appropriate mapping strategy for a skill."""
        skill = skill_data.get('skill', {})
        skill_type = skill.get('type', 'unknown')
        abstraction_level = skill.get('abstraction_level', 'concrete')

        # High compatibility and concrete skills: direct transfer
        if (compatibility['overall_compatibility'] > 0.8 and
                abstraction_level == 'concrete'):
            return 'direct_transfer'

        # High abstraction skills: abstract transfer
        elif abstraction_level in ['strategic', 'meta']:
            return 'abstract_transfer'

        # Medium compatibility: scaled transfer
        elif compatibility['overall_compatibility'] > 0.5:
            return 'scaled_transfer'

        # Low compatibility: compositional transfer
        else:
            return 'compositional_transfer'

    def _direct_transfer(self, skill_data: Dict[str, Any], target_analysis: Dict[str, Any],
                         compatibility: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Direct transfer with minimal adaptation."""
        transferred_skill = skill_data.copy()

        # Reduce confidence slightly for safety
        transferred_skill['confidence'] *= 0.9
        transferred_skill['transfer_type'] = 'direct'
        transferred_skill['source_compatibility'] = compatibility['overall_compatibility']

        return transferred_skill

    def _scaled_transfer(self, skill_data: Dict[str, Any], target_analysis: Dict[str, Any],
                         compatibility: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transfer with scaled confidence based on compatibility."""
        transferred_skill = skill_data.copy()

        # Scale confidence by compatibility
        transferred_skill['confidence'] *= compatibility['overall_compatibility']
        transferred_skill['transfer_type'] = 'scaled'
        transferred_skill['source_compatibility'] = compatibility['overall_compatibility']

        # Add adaptation notes
        transferred_skill['adaptation_notes'] = []

        if compatibility['action_space_compatibility'] < 1.0:
            transferred_skill['adaptation_notes'].append(
                'Action mappings may need adjustment'
            )

        if compatibility['state_space_compatibility'] < 1.0:
            transferred_skill['adaptation_notes'].append(
                'State interpretation requires validation'
            )

        return transferred_skill

    def _abstract_transfer(self, skill_data: Dict[str, Any], target_analysis: Dict[str, Any],
                           compatibility: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transfer abstract skills with environment-specific grounding needed."""
        skill = skill_data.get('skill', {})

        # Only transfer if skill is abstract enough
        if skill.get('abstraction_level') not in ['strategic', 'meta']:
            return None

        transferred_skill = skill_data.copy()

        # Reduce confidence more for abstract transfers
        transferred_skill['confidence'] *= 0.7
        transferred_skill['transfer_type'] = 'abstract'
        transferred_skill['requires_grounding'] = True

        # Add grounding recommendations
        transferred_skill['grounding_recommendations'] = [
            'Validate abstract concepts in new environment',
            'Map high-level strategies to specific actions',
            'Monitor performance to refine abstractions'
        ]

        return transferred_skill

    def _compositional_transfer(self, skill_data: Dict[str, Any], target_analysis: Dict[str, Any],
                                compatibility: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transfer by decomposing skills into components."""
        skill = skill_data.get('skill', {})

        # Extract transferable components
        components = []

        # Check if skill has clear parameters that can be decomposed
        parameters = skill.get('parameters', {})

        if 'expected_reward' in parameters and compatibility['reward_structure_similarity'] > 0.5:
            components.append({
                'component': 'reward_seeking',
                'confidence': compatibility['reward_structure_similarity']
            })

        if skill.get('type') in ['exploratory_strategy', 'adaptive_behavior']:
            # These are generally transferable
            components.append({
                'component': 'behavioral_strategy',
                'confidence': 0.6
            })

        if not components:
            return None

        transferred_skill = skill_data.copy()
        transferred_skill['confidence'] *= 0.5  # Low confidence for compositional
        transferred_skill['transfer_type'] = 'compositional'
        transferred_skill['components'] = components
        transferred_skill['requires_recomposition'] = True

        return transferred_skill

    def _calculate_transfer_confidence(self, compatibility: Dict[str, Any],
                                       skill_mappings: Dict[str, Any],
                                       pattern_mappings: List[Dict[str, Any]]) -> floating | float:
        """Calculate overall confidence in the transfer."""
        factors = [compatibility['overall_compatibility']]

        # Environment compatibility factor

        # Skill transfer success factor
        if skill_mappings['metadata']['total_source_skills'] > 0:
            skill_success_rate = (
                    skill_mappings['metadata']['successfully_mapped'] /
                    skill_mappings['metadata']['total_source_skills']
            )
            factors.append(skill_success_rate)

        # Pattern transfer factor
        if pattern_mappings:
            pattern_confidence = np.mean([
                p['confidence_factor'] for p in pattern_mappings
            ])
            factors.append(pattern_confidence)

        # Calculate weighted confidence
        if factors:
            return np.mean(factors)
        else:
            return 0.0

    def _generate_adaptation_recommendations(self, compatibility: Dict[str, Any],
                                             skill_mappings: Dict[str, Any]) -> List[str]:
        """Generate recommendations for successful transfer."""
        recommendations = []

        # Based on compatibility
        if compatibility['overall_compatibility'] < 0.5:
            recommendations.append(
                'Low environment compatibility - expect significant adaptation period'
            )

        if compatibility['action_space_compatibility'] < 0.8:
            recommendations.append(
                'Action space differs - validate action mappings early'
            )

        if compatibility['state_space_compatibility'] < 0.8:
            recommendations.append(
                'State space differs - monitor state interpretation carefully'
            )

        # Based on skill mappings
        strategies_used = skill_mappings['metadata']['mapping_strategies_used']

        if 'abstract_transfer' in strategies_used:
            recommendations.append(
                'Abstract skills transferred - ground them through experience'
            )

        if 'compositional_transfer' in strategies_used:
            recommendations.append(
                'Skills decomposed for transfer - recompose through learning'
            )

        # General recommendations
        if compatibility['behavioral_similarity'] > 0.7:
            recommendations.append(
                'High behavioral similarity - leverage existing strategies'
            )
        else:
            recommendations.append(
                'Different behavioral requirements - explore new strategies'
            )

        return recommendations

    def _record_transfer(self, mapped_knowledge: Dict[str, Any]):
        """Record transfer attempt for learning."""
        transfer_record = {
            'source_env': mapped_knowledge['mapping_metadata']['source_env'],
            'target_env': mapped_knowledge['mapping_metadata']['target_env'],
            'timestamp': mapped_knowledge['mapping_metadata']['timestamp'],
            'transfer_confidence': mapped_knowledge['transfer_confidence'],
            'skills_transferred': len(mapped_knowledge['transferable_skills']),
            'patterns_transferred': len(mapped_knowledge['adapted_patterns'])
        }

        self.transfer_history.append(transfer_record)

        # Update success metrics (to be updated with actual performance later)
        env_pair = f"{transfer_record['source_env']}->{transfer_record['target_env']}"
        if env_pair not in self.transfer_success_metrics:
            self.transfer_success_metrics[env_pair] = {
                'attempts': 0,
                'total_confidence': 0.0,
                'total_skills': 0
            }

        metrics = self.transfer_success_metrics[env_pair]
        metrics['attempts'] += 1
        metrics['total_confidence'] += transfer_record['transfer_confidence']
        metrics['total_skills'] += transfer_record['skills_transferred']

    def update_transfer_outcome(self, source_env: str, target_env: str,
                                performance_improvement: float,
                                detailed_metrics: Dict[str, Any] = None):
        """Update transfer success metrics with actual performance."""
        env_pair = f"{source_env}->{target_env}"

        if env_pair in self.transfer_success_metrics:
            metrics = self.transfer_success_metrics[env_pair]
            if 'performance_improvements' not in metrics:
                metrics['performance_improvements'] = []

            metrics['performance_improvements'].append(performance_improvement)

            # Calculate average improvement
            metrics['average_improvement'] = np.mean(metrics['performance_improvements'])

            # NEW: Add detailed metrics if provided
            if detailed_metrics:
                metrics['detailed_performance'] = detailed_metrics
                metrics['adaptation_time'] = detailed_metrics.get('adaptation_time', 0)
                metrics['skill_retention'] = detailed_metrics.get('skill_retention', 0.0)

            # Store validation metrics
            if env_pair not in self.validation_metrics:
                self.validation_metrics[env_pair] = {}

            self.validation_metrics[env_pair]['last_validation'] = {
                'performance_improvement': performance_improvement,
                'timestamp': time.time(),
                'detailed_metrics': detailed_metrics
            }
    def validate_transfer(self, source_env: str, target_env: str,
                          transferred_skills: Dict[str, Any],
                          target_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Validate transferred skills performance."""
        env_pair = f"{source_env}->{target_env}"

        # Get baseline from when transfer was applied
        baseline = self.validation_metrics.get(env_pair, {}).get('baseline_performance', {})

        # Calculate performance delta
        performance_delta = target_performance.get('average_reward', 0) - baseline.get('average_reward', 0)
        adaptation_time = target_performance.get('step_count', 0) - baseline.get('step_count', 0)

        # Validate each transferred skill
        skill_validations = {}
        for skill_id, skill_data in transferred_skills.items():
            skill_validations[skill_id] = {
                'predicted_performance': skill_data.get('confidence', 0),
                'actual_performance': skill_data.get('success_rate', 0),
                'validation_status': 'success' if skill_data.get('success_rate', 0) > 0.5 else 'needs_adaptation'
            }

        validation_result = {
            'performance_delta': performance_delta,
            'adaptation_time': adaptation_time,
            'overall_success': performance_delta > 0.1,
            'skill_validations': skill_validations,
            'recommendations': self._generate_validation_recommendations(skill_validations)
        }

        # Update transfer success metrics
        self.update_transfer_outcome(source_env, target_env, performance_delta, validation_result)

        return validation_result

    def get_adaptive_transfer_strategy(self, source_env: str, target_env: str) -> str:
        """Select best transfer strategy based on validation history."""
        env_pair = f"{source_env}->{target_env}"

        if env_pair in self.transfer_success_metrics:
            metrics = self.transfer_success_metrics[env_pair]

            # Check which strategies worked best
            if 'strategy_performance' in metrics:
                best_strategy = max(metrics['strategy_performance'],
                                    key=lambda x: metrics['strategy_performance'][x])
                return best_strategy

        # Default strategy selection based on environment compatibility
        return 'direct_transfer'  # fallback to existing logic


    def get_transfer_recommendations(self, source_env: str,
                                     available_targets: List[str]) -> List[Dict[str, Any]]:
        """Get recommendations for the best transfer targets."""
        recommendations = []

        for target in available_targets:
            env_pair = f"{source_env}->{target}"

            recommendation = {
                'target_environment': target,
                'transfer_confidence': 0.5,  # Default
                'recommendation_score': 0.5,
                'rationale': []
            }

            # Use historical data if available
            if env_pair in self.transfer_success_metrics:
                metrics = self.transfer_success_metrics[env_pair]

                if metrics['attempts'] > 0:
                    avg_confidence = metrics['total_confidence'] / metrics['attempts']
                    recommendation['transfer_confidence'] = avg_confidence

                    if 'average_improvement' in metrics:
                        recommendation['historical_improvement'] = metrics['average_improvement']
                        recommendation['recommendation_score'] = (
                                avg_confidence * 0.5 +
                                min(1.0, metrics['average_improvement']) * 0.5
                        )

                    recommendation['rationale'].append(
                        f"Based on {metrics['attempts']} previous transfers"
                    )

            recommendations.append(recommendation)

        # Sort by recommendation score
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)

        return recommendations

    def _generate_validation_recommendations(self, skill_validations: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        successful_skills = [sid for sid, val in skill_validations.items()
                             if val['validation_status'] == 'success']
        failed_skills = [sid for sid, val in skill_validations.items()
                         if val['validation_status'] == 'needs_adaptation']

        if len(successful_skills) > len(failed_skills):
            recommendations.append(f"Transfer largely successful: {len(successful_skills)} skills working well")
        else:
            recommendations.append(f"Transfer needs adaptation: {len(failed_skills)} skills underperforming")

        if failed_skills:
            recommendations.append("Consider retraining underperforming transferred skills")

        if successful_skills:
            recommendations.append("Successful skills can be used as foundation for further learning")

        return recommendations



