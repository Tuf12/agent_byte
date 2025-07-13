"""
Dynamic skill discovery system for Agent Byte.

This module discovers skills from interpreted patterns without any
hard-coded environment knowledge, enabling true environment-agnostic learning.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import uuid
import time
import logging
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from collections import defaultdict, Counter
import torch


class SkillDiscovery:
    """
    Discovers skills dynamically from pattern interpretations.

    Skills are abstract capabilities that emerge from successful patterns,
    completely independent of specific environments.
    """

    def __init__(self):
        """Initialize the skill discovery system."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Skill templates based on abstract behavioral patterns
        self.skill_templates = {
            'action_optimization': self._discover_action_optimization,
            'state_response': self._discover_state_response,
            'reward_seeking': self._discover_reward_seeking,
            'pattern_execution': self._discover_pattern_execution,
            'adaptive_behavior': self._discover_adaptive_behavior,
            'exploratory_strategy': self._discover_exploratory_strategy
        }

        # Skill abstraction levels
        self.abstraction_levels = ['concrete', 'tactical', 'strategic', 'meta']

        # Discovery thresholds
        self.confidence_threshold = 0.6
        self.evidence_threshold = 3

        # Clustering support for skill discovery
        self.experience_buffer = []
        self.clustering_interval = 100
        self.experiences_since_clustering = 0
        self.min_cluster_size = 5
    def _add_to_experience_buffer(self, latent_state: np.ndarray, action: int,
                                 reward: float, pattern_interpretation: Dict[str, Any]):
        """Add experience to clustering buffer."""
        experience = {
            'latent_state': latent_state,
            'action': action,
            'reward': reward,
            'pattern_interpretation': pattern_interpretation,
            'timestamp': time.time()
        }
        self.experience_buffer.append(experience)
        self.experiences_since_clustering += 1

        # Keep buffer size manageable
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]


    def _cluster_based_discovery(self) -> List[Dict[str, Any]]:
        """Discover skills through clustering experiences."""
        if len(self.experience_buffer) < self.min_cluster_size:
            return []

        skills = []

        try:
            # Prepare data for clustering
            states = np.array([exp['latent_state'] for exp in self.experience_buffer])
            actions = np.array([exp['action'] for exp in self.experience_buffer])
            rewards = np.array([exp['reward'] for exp in self.experience_buffer])

            # Combine state and action for clustering
            features = np.column_stack([states, actions.reshape(-1, 1)])

            # Use KMeans clustering
            n_clusters = min(8, max(2, len(self.experience_buffer) // 20))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)

            # Analyze each cluster
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_experiences = [self.experience_buffer[i] for i in range(len(self.experience_buffer)) if cluster_mask[i]]

                if len(cluster_experiences) >= self.min_cluster_size:
                    # Analyze cluster
                    cluster_actions = [exp['action'] for exp in cluster_experiences]
                    cluster_rewards = [exp['reward'] for exp in cluster_experiences]

                    dominant_action = max(set(cluster_actions), key=cluster_actions.count)
                    avg_reward = np.mean(cluster_rewards)

                    if avg_reward > 0:  # Only consider positive clusters
                        skill = {
                            'name': f'cluster_skill_{cluster_id}',
                            'type': 'cluster_behavior',
                            'description': f'Learned behavior from cluster {cluster_id}',
                            'confidence': min(1.0, len(cluster_experiences) / 50.0),
                            'abstraction_level': 'tactical',
                            'discovered_via': 'clustering',
                            'timestamp': time.time(),
                            'evidence': {
                                'cluster_id': cluster_id,
                                'cluster_size': len(cluster_experiences),
                                'dominant_action': dominant_action,
                                'average_reward': avg_reward
                            },
                            'parameters': {
                                'cluster_center': kmeans.cluster_centers_[cluster_id].tolist(),
                                'dominant_action': dominant_action
                            }
                        }
                        skill['id'] = self._generate_skill_id(skill)
                        skills.append(skill)

        except Exception as e:
            self.logger.warning(f"Clustering-based discovery failed: {e}")

        return skills



    def discover(self, pattern_interpretation: Dict[str, Any],
                 existing_skills: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Discover new skills from pattern interpretation.

        Args:
            pattern_interpretation: Interpreted patterns from neural brain
            existing_skills: Currently known skills

        Returns:
            List of newly discovered skills
        """
        discovered_skills = []

        # Try each skill discovery template
        for template_name, discover_func in self.skill_templates.items():
            skills = discover_func(pattern_interpretation, existing_skills)
            for skill in skills:
                if self._is_novel_skill(skill, existing_skills):
                    # Assign unique ID
                    skill['id'] = self._generate_skill_id(skill)
                    skill['discovered_via'] = template_name
                    skill['timestamp'] = time.time()
                    discovered_skills.append(skill)

        # Add clustering-based discovery
        if self.experiences_since_clustering >= self.clustering_interval and len(self.experience_buffer) >= 50:
            clustering_skills = self._cluster_based_discovery()
            discovered_skills.extend(clustering_skills)
            self.experiences_since_clustering = 0

        # Attempt to abstract existing skills
        abstract_skills = self._discover_abstract_skills(
            pattern_interpretation, existing_skills
        )
        discovered_skills.extend(abstract_skills)

        return discovered_skills

    def _discover_action_optimization(self, interpretation: Dict[str, Any],
                                      existing_skills: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover skills related to optimizing actions."""
        skills = []

        # Look for successful action patterns
        for indicator in interpretation.get('skill_indicators', []):
            if indicator['type'] == 'successful_action_pattern':
                confidence = indicator.get('confidence', 0)
                if confidence >= self.confidence_threshold:
                    skill = {
                        'name': f'optimize_action_{indicator["evidence"]["action"]}',
                        'type': 'action_optimization',
                        'description': indicator['description'],
                        'confidence': confidence,
                        'abstraction_level': 'concrete',
                        'evidence': indicator['evidence'],
                        'conditions': {
                            'applicable_when': 'similar_state_encountered',
                            'expected_outcome': 'positive_reward'
                        },
                        'parameters': {
                            'target_action': indicator['evidence']['action'],
                            'expected_reward': indicator['evidence']['average_reward']
                        }
                    }
                    skills.append(skill)

        # Look for exploitation patterns
        for pattern in interpretation.get('detected_patterns', []):
            if pattern['name'] == 'exploitation' and pattern['confidence'] >= self.confidence_threshold:
                details = pattern.get('details', {})
                skill = {
                    'name': 'focused_exploitation',
                    'type': 'action_optimization',
                    'description': 'Consistently exploit best-known action',
                    'confidence': pattern['confidence'],
                    'abstraction_level': 'tactical',
                    'evidence': details,
                    'conditions': {
                        'applicable_when': 'high_confidence_action_available',
                        'expected_outcome': 'consistent_positive_reward'
                    },
                    'parameters': {
                        'dominant_action': details.get('dominant_action'),
                        'dominance_ratio': details.get('dominance', 0)
                    }
                }
                skills.append(skill)

        return skills

    def _discover_state_response(self, interpretation: Dict[str, Any],
                                 existing_skills: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover skills related to responding to states."""
        skills = []

        # Look for adaptation patterns
        for pattern in interpretation.get('detected_patterns', []):
            if pattern['name'] == 'adaptation' and pattern['confidence'] >= self.confidence_threshold:
                skill = {
                    'name': 'adaptive_response',
                    'type': 'state_response',
                    'description': 'Adapt actions based on state changes',
                    'confidence': pattern['confidence'],
                    'abstraction_level': 'tactical',
                    'evidence': pattern.get('details', {}),
                    'conditions': {
                        'applicable_when': 'state_features_correlate_with_actions',
                        'expected_outcome': 'improved_performance'
                    },
                    'parameters': {
                        'correlation_strength': pattern.get('details', {}).get('correlation_strength', 0)
                    }
                }
                skills.append(skill)

        # Look for state-dependent indicators
        for indicator in interpretation.get('skill_indicators', []):
            if indicator['type'] == 'state_action_correlation':
                skill = {
                    'name': 'state_aware_decision',
                    'type': 'state_response',
                    'description': 'Make decisions based on state features',
                    'confidence': indicator.get('confidence', 0.5),
                    'abstraction_level': 'concrete',
                    'evidence': indicator.get('evidence', {}),
                    'conditions': {
                        'applicable_when': 'state_features_available',
                        'expected_outcome': 'contextual_action_selection'
                    },
                    'parameters': {
                        'feature_importance': 'learned_from_experience'
                    }
                }
                if skill['confidence'] >= self.confidence_threshold:
                    skills.append(skill)

        return skills

    def _discover_reward_seeking(self, interpretation: Dict[str, Any],
                                 existing_skills: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover skills related to seeking rewards."""
        skills = []

        # Look for reward maximization indicators
        for indicator in interpretation.get('skill_indicators', []):
            if indicator['type'] == 'reward_maximization':
                skill = {
                    'name': 'reward_maximizer',
                    'type': 'reward_seeking',
                    'description': indicator['description'],
                    'confidence': indicator['confidence'],
                    'abstraction_level': 'concrete',
                    'evidence': indicator['evidence'],
                    'conditions': {
                        'applicable_when': 'action_available',
                        'expected_outcome': 'high_reward'
                    },
                    'parameters': {
                        'target_action': indicator['evidence']['action'],
                        'expected_reward': indicator['evidence']['average_reward']
                    }
                }
                skills.append(skill)

        # Look for improvement patterns
        for pattern in interpretation.get('detected_patterns', []):
            if pattern['name'] == 'improvement' and pattern['confidence'] >= self.confidence_threshold:
                skill = {
                    'name': 'continuous_improvement',
                    'type': 'reward_seeking',
                    'description': 'Consistently improving reward acquisition',
                    'confidence': pattern['confidence'],
                    'abstraction_level': 'strategic',
                    'evidence': pattern.get('details', {}),
                    'conditions': {
                        'applicable_when': 'learning_active',
                        'expected_outcome': 'increasing_rewards'
                    },
                    'parameters': {
                        'improvement_rate': pattern.get('details', {}).get('reward_trend', 0)
                    }
                }
                skills.append(skill)

        return skills

    def _discover_pattern_execution(self, interpretation: Dict[str, Any],
                                    existing_skills: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover skills related to executing patterns."""
        skills = []

        # Look for stable behavior patterns
        for indicator in interpretation.get('skill_indicators', []):
            if indicator['type'] == 'stable_behavior':
                skill = {
                    'name': 'pattern_consistency',
                    'type': 'pattern_execution',
                    'description': indicator['description'],
                    'confidence': indicator['confidence'],
                    'abstraction_level': 'tactical',
                    'evidence': indicator['evidence'],
                    'conditions': {
                        'applicable_when': 'familiar_situation',
                        'expected_outcome': 'predictable_performance'
                    },
                    'parameters': {
                        'pattern_count': indicator['evidence']['pattern_count'],
                        'consistency_level': indicator['confidence']
                    }
                }
                skills.append(skill)

        # Look for convergence patterns
        for pattern in interpretation.get('detected_patterns', []):
            if pattern['name'] == 'convergence':
                skill = {
                    'name': 'policy_convergence',
                    'type': 'pattern_execution',
                    'description': 'Converged to stable behavioral policy',
                    'confidence': pattern['confidence'],
                    'abstraction_level': 'strategic',
                    'evidence': pattern.get('details', {}),
                    'conditions': {
                        'applicable_when': 'sufficient_experience',
                        'expected_outcome': 'optimal_behavior'
                    },
                    'parameters': {
                        'stability': pattern.get('details', {}).get('stability', 0),
                        'pattern_stability': pattern.get('details', {}).get('pattern_stability', 0)
                    }
                }
                skills.append(skill)

        return skills

    def _discover_adaptive_behavior(self, interpretation: Dict[str, Any],
                                    existing_skills: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover skills related to adaptive behavior."""
        skills = []

        # Check behavioral state transitions
        if interpretation.get('behavioral_state') == 'transitioning':
            skill = {
                'name': 'behavioral_flexibility',
                'type': 'adaptive_behavior',
                'description': 'Ability to transition between behavioral states',
                'confidence': interpretation.get('confidence', 0.5),
                'abstraction_level': 'strategic',
                'evidence': {
                    'current_state': interpretation['behavioral_state'],
                    'detected_patterns': len(interpretation.get('detected_patterns', []))
                },
                'conditions': {
                    'applicable_when': 'environment_changes',
                    'expected_outcome': 'maintained_performance'
                },
                'parameters': {
                    'flexibility_score': interpretation.get('confidence', 0.5)
                }
            }
            if skill['confidence'] >= self.confidence_threshold:
                skills.append(skill)

        # Look for oscillation as adaptive behavior
        for pattern in interpretation.get('detected_patterns', []):
            if pattern['name'] == 'oscillation':
                skill = {
                    'name': 'balanced_exploration',
                    'type': 'adaptive_behavior',
                    'description': 'Balance between multiple strategies',
                    'confidence': pattern['confidence'],
                    'abstraction_level': 'tactical',
                    'evidence': pattern.get('details', {}),
                    'conditions': {
                        'applicable_when': 'multiple_viable_options',
                        'expected_outcome': 'robust_performance'
                    },
                    'parameters': {
                        'balance_ratio': pattern.get('details', {}).get('action_balance', 0)
                    }
                }
                skills.append(skill)

        return skills

    def _discover_exploratory_strategy(self, interpretation: Dict[str, Any],
                                       existing_skills: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover skills related to exploration."""
        skills = []

        # Look for exploration patterns
        for pattern in interpretation.get('detected_patterns', []):
            if pattern['name'] == 'exploration':
                skill = {
                    'name': 'systematic_exploration',
                    'type': 'exploratory_strategy',
                    'description': 'Systematic exploration of action space',
                    'confidence': pattern['confidence'],
                    'abstraction_level': 'tactical',
                    'evidence': pattern.get('details', {}),
                    'conditions': {
                        'applicable_when': 'uncertainty_high',
                        'expected_outcome': 'knowledge_acquisition'
                    },
                    'parameters': {
                        'exploration_entropy': pattern.get('details', {}).get('entropy', 0),
                        'coverage': pattern.get('details', {}).get('normalized_entropy', 0)
                    }
                }
                skills.append(skill)

        # Check learning phase for exploration
        if interpretation.get('learning_phase') in ['early_learning', 'rapid_learning']:
            skill = {
                'name': 'learning_phase_exploration',
                'type': 'exploratory_strategy',
                'description': 'Active exploration during learning phase',
                'confidence': 0.7,  # Default confidence for phase-based skills
                'abstraction_level': 'strategic',
                'evidence': {
                    'learning_phase': interpretation['learning_phase']
                },
                'conditions': {
                    'applicable_when': 'new_environment_or_situation',
                    'expected_outcome': 'rapid_learning'
                },
                'parameters': {
                    'phase': interpretation['learning_phase']
                }
            }
            skills.append(skill)

        return skills

    def _discover_abstract_skills(self, interpretation: Dict[str, Any],
                                  existing_skills: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover abstract skills by combining existing skills."""
        abstract_skills = []

        # Group existing skills by type
        skills_by_type = {}
        for skill_data in existing_skills.values():
            skill = skill_data.get('skill', {})
            skill_type = skill.get('type', 'unknown')
            if skill_type not in skills_by_type:
                skills_by_type[skill_type] = []
            skills_by_type[skill_type].append(skill)

        # Look for meta-skills
        if len(skills_by_type) >= 3:
            # Multi-strategy skill
            total_skills = sum(len(skills) for skills in skills_by_type.values())
            if total_skills >= 5:
                meta_skill = {
                    'name': 'multi_strategy_mastery',
                    'type': 'meta_skill',
                    'description': 'Ability to employ multiple strategies',
                    'confidence': 0.8,
                    'abstraction_level': 'meta',
                    'evidence': {
                        'skill_types': list(skills_by_type.keys()),
                        'total_skills': total_skills
                    },
                    'conditions': {
                        'applicable_when': 'complex_environment',
                        'expected_outcome': 'versatile_performance'
                    },
                    'parameters': {
                        'strategy_diversity': len(skills_by_type),
                        'skill_count': total_skills
                    },
                    'id': self._generate_skill_id({'name': 'multi_strategy_mastery'})
                }
                abstract_skills.append(meta_skill)

        # Look for skill combinations
        if 'action_optimization' in skills_by_type and 'state_response' in skills_by_type:
            combined_skill = {
                'name': 'contextual_optimization',
                'type': 'combined_skill',
                'description': 'Optimize actions based on state context',
                'confidence': 0.7,
                'abstraction_level': 'strategic',
                'evidence': {
                    'component_skills': ['action_optimization', 'state_response']
                },
                'conditions': {
                    'applicable_when': 'state_dependent_rewards',
                    'expected_outcome': 'context_aware_performance'
                },
                'parameters': {
                    'integration_level': 'high'
                },
                'id': self._generate_skill_id({'name': 'contextual_optimization'})
            }
            abstract_skills.append(combined_skill)

        return abstract_skills

    def _is_novel_skill(self, skill: Dict[str, Any], existing_skills: Dict[str, Any]) -> bool:
        """Check if a skill is novel compared to existing skills."""
        skill_name = skill['name']
        skill_type = skill['type']

        for existing_skill_data in existing_skills.values():
            existing_skill = existing_skill_data.get('skill', {})
            # Check for same name or very similar skills
            if existing_skill.get('name') == skill_name:
                return False

            # Check for same type with similar parameters
            if existing_skill.get('type') == skill_type:
                existing_params = existing_skill.get('parameters', {})
                new_params = skill.get('parameters', {})

                # Simple similarity check
                if self._are_parameters_similar(existing_params, new_params):
                    return False

        return True

    def _are_parameters_similar(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
        """Check if two parameter sets are similar."""
        if not params1 or not params2:
            return False

        # Check if they have mostly the same keys
        keys1 = set(params1.keys())
        keys2 = set(params2.keys())
        common_keys = keys1.intersection(keys2)

        if len(common_keys) < min(len(keys1), len(keys2)) * 0.7:
            return False

        # Check if values are similar for common keys
        similar_count = 0
        for key in common_keys:
            val1 = params1[key]
            val2 = params2[key]

            if type(val1) == type(val2):
                if isinstance(val1, (int, float)):
                    if abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-8) < 0.2:
                        similar_count += 1
                elif val1 == val2:
                    similar_count += 1

        return similar_count >= len(common_keys) * 0.7

    def _generate_skill_id(self, skill: Dict[str, Any]) -> str:
        """Generate unique ID for a skill."""
        # Use combination of name and type for uniqueness
        base = f"{skill['name']}_{skill.get('type', 'unknown')}"
        # Add short UUID for guaranteed uniqueness
        return f"{base}_{str(uuid.uuid4())[:8]}"

    def analyze_skill_relationships(self, skills: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between discovered skills."""
        relationships = {
            'skill_hierarchy': {},
            'complementary_skills': [],
            'conflicting_skills': [],
            'skill_progression': []
        }

        # Group by abstraction level
        by_level = {}
        for skill_id, skill_data in skills.items():
            skill = skill_data.get('skill', {})
            level = skill.get('abstraction_level', 'concrete')
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(skill_id)

        relationships['skill_hierarchy'] = by_level

        # Find complementary skills
        skill_list = list(skills.values())
        for i in range(len(skill_list)):
            for j in range(i + 1, len(skill_list)):
                skill1 = skill_list[i].get('skill', {})
                skill2 = skill_list[j].get('skill', {})

                if self._are_skills_complementary(skill1, skill2):
                    relationships['complementary_skills'].append({
                        'skill1': skill1['name'],
                        'skill2': skill2['name'],
                        'reason': 'different_types_same_goal'
                    })

        return relationships

    def _are_skills_complementary(self, skill1: Dict[str, Any], skill2: Dict[str, Any]) -> bool:
        """Check if two skills are complementary."""
        # Different types but same abstraction level often complement
        if (skill1.get('type') != skill2.get('type') and
                skill1.get('abstraction_level') == skill2.get('abstraction_level')):
            return True

        # Action optimization and state response complement each other
        types = {skill1.get('type'), skill2.get('type')}
        if types == {'action_optimization', 'state_response'}:
            return True

        return False

    def categorize_skill(self, skill: Dict[str, Any]) -> str:
        """
        Categorize skill into chess-inspired taxonomy.

        Args:
            skill: Skill to categorize

        Returns:
            Category name (tactical/positional/strategic)
        """
        # Default to tactical
        category = 'tactical'

        # Analyze skill characteristics
        confidence = skill.get('confidence', 0)
        abstraction = skill.get('abstraction_level', 'concrete')
        evidence = skill.get('evidence', {})

        # Strategic skills: high abstraction, high confidence, broad impact
        if abstraction in ['strategic', 'meta'] and confidence > 0.8:
            category = 'strategic'

        # Positional skills: medium abstraction, pattern-based
        elif abstraction == 'tactical' and confidence > 0.6:
            if 'behavioral_state' in evidence or 'learning_phase' in evidence:
                category = 'positional'

        # Tactical skills: concrete, immediate reward
        else:
            if evidence.get('average_reward', 0) > 0:
                category = 'tactical'

        return category

    def score_skill(self, skill: Dict[str, Any]) -> float:
        """
        Score skill quality for ranking and selection.

        Args:
            skill: Skill to score

        Returns:
            Score value (0-1)
        """
        score = 0.0

        # Base score from confidence
        score += skill.get('confidence', 0) * 0.4

        # Success rate contribution
        if 'success_rate' in skill:
            score += skill['success_rate'] * 0.3

        # Evidence quality
        evidence = skill.get('evidence', {})
        if 'cluster_size' in evidence:
            # Larger clusters are more reliable
            cluster_score = min(1.0, evidence['cluster_size'] / 100)
            score += cluster_score * 0.2

        if 'average_reward' in evidence:
            # Normalize reward contribution
            reward_score = min(1.0, max(0.0, evidence['average_reward'] / 10.0))
            score += reward_score * 0.1

        # Abstraction level bonus
        abstraction_scores = {
            'concrete': 0.0,
            'tactical': 0.05,
            'strategic': 0.1,
            'meta': 0.15
        }
        score += abstraction_scores.get(skill.get('abstraction_level', 'concrete'), 0)

        return min(1.0, score)

    def get_skill_hierarchy(self, skills: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize skills into hierarchical taxonomy.

        Args:
            skills: Dictionary of all skills

        Returns:
            Hierarchical organization by category
        """
        hierarchy = {
            'tactical': [],
            'positional': [],
            'strategic': []
        }

        for skill_id, skill_data in skills.items():
            skill = skill_data.get('skill', {})
            category = self.categorize_skill(skill)
            score = self.score_skill(skill)

            skill_info = {
                'id': skill_id,
                'name': skill['name'],
                'score': score,
                'confidence': skill.get('confidence', 0),
                'applications': skill_data.get('application_count', 0)
            }

            hierarchy[category].append(skill_info)

        # Sort each category by score
        for category in hierarchy:
            hierarchy[category].sort(key=lambda x: x['score'], reverse=True)

        return hierarchy