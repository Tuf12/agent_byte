"""
Symbolic brain component for Agent Byte.

This module implements the symbolic reasoning component of the dual brain system,
handling knowledge representation, skill discovery, and high-level decision making.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import time

from ..storage.base import StorageBase
from ..knowledge.skill_discovery import SkillDiscovery
from ..knowledge.pattern_interpreter import PatternInterpreter
from ..knowledge.decision_maker import SymbolicDecisionMaker


class SymbolicBrain:
    """
    Symbolic reasoning engine with dynamic skill discovery.

    This brain interprets neural patterns, discovers skills, and makes
    high-level decisions based on learned knowledge.
    """

    def __init__(self,
                 agent_id: str,
                 storage: StorageBase,
                 config: Dict[str, Any]):
        """
        Initialize the symbolic brain.

        Args:
            agent_id: Unique identifier for the agent
            storage: Storage backend for persistence
            config: Configuration parameters
        """
        self.agent_id = agent_id
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger(f"SymbolicBrain-{agent_id}")

        # Knowledge components
        self.skill_discovery = SkillDiscovery()
        self.pattern_interpreter = PatternInterpreter()
        self.decision_maker = SymbolicDecisionMaker()

        # Knowledge base
        self.knowledge = {
            'discovered_skills': {},
            'skill_applications': [],
            'environmental_understanding': {},
            'decision_history': [],
            'meta_knowledge': {
                'total_decisions': 0,
                'successful_decisions': 0,
                'skill_discovery_count': 0,
                'last_updated': time.time()
            }
        }

        # Current context
        self.current_context = {
            'environment_analysis': None,
            'recent_patterns': [],
            'active_skills': []
        }

        self.logger.info("Symbolic brain initialized")

    def interpret_neural_patterns(self, neural_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret patterns from neural brain.

        Args:
            neural_insights: Insights from neural brain

        Returns:
            Symbolic interpretation of patterns
        """
        # Use pattern interpreter to understand neural patterns
        interpretation = self.pattern_interpreter.interpret(neural_insights)

        # Update recent patterns
        self.current_context['recent_patterns'].append({
            'timestamp': time.time(),
            'interpretation': interpretation,
            'neural_summary': neural_insights.get('pattern_summary', {})
        })

        # Keep recent patterns manageable
        if len(self.current_context['recent_patterns']) > 20:
            self.current_context['recent_patterns'] = self.current_context['recent_patterns'][-20:]

        return interpretation

    def discover_skills(self, pattern_interpretation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Discover new skills from interpreted patterns.

        Args:
            pattern_interpretation: Interpreted neural patterns

        Returns:
            List of discovered skills
        """
        # Use skill discovery to find new skills
        discovered = self.skill_discovery.discover(
            pattern_interpretation,
            self.knowledge['discovered_skills']
        )

        # Add new skills to knowledge base
        for skill in discovered:
            skill_id = skill['id']
            if skill_id not in self.knowledge['discovered_skills']:
                self.knowledge['discovered_skills'][skill_id] = {
                    'skill': skill,
                    'discovered_at': time.time(),
                    'application_count': 0,
                    'success_rate': 0.0,
                    'confidence': skill.get('confidence', 0.5)
                }

                self.knowledge['meta_knowledge']['skill_discovery_count'] += 1
                self.logger.info(f"Discovered new skill: {skill['name']}")

        return discovered

    def make_decision(self, state: np.ndarray, q_values: np.ndarray,
                      exploration_rate: float) -> Tuple[Optional[int], Dict[str, Any]]:
        """
        Make symbolic decision based on current knowledge.

        Args:
            state: Current state
            q_values: Q-values from neural network
            exploration_rate: Current exploration rate

        Returns:
            Tuple of (action or None, decision_info)
        """
        # Prepare decision context
        context = {
            'state': state,
            'q_values': q_values,
            'exploration_rate': exploration_rate,
            'discovered_skills': self.knowledge['discovered_skills'],
            'recent_patterns': self.current_context['recent_patterns'],
            'environment_analysis': self.current_context['environment_analysis']
        }

        # Use decision maker
        action, reasoning, confidence = self.decision_maker.decide(context)

        # Create decision info
        decision_info = {
            'symbolic_action': action,
            'reasoning': reasoning,
            'confidence': confidence,
            'skills_available': len(self.knowledge['discovered_skills']),
            'decision_type': 'symbolic' if action is not None else 'deferred'
        }

        # Record decision
        self._record_decision(decision_info)

        return action, decision_info

    def _record_decision(self, decision_info: Dict[str, Any]):
        """Record decision for learning."""
        decision_record = {
            'timestamp': time.time(),
            'decision_info': decision_info,
            'context_summary': {
                'skills_available': len(self.knowledge['discovered_skills']),
                'patterns_analyzed': len(self.current_context['recent_patterns'])
            }
        }

        self.knowledge['decision_history'].append(decision_record)
        self.knowledge['meta_knowledge']['total_decisions'] += 1

        # Keep history manageable
        if len(self.knowledge['decision_history']) > 100:
            self.knowledge['decision_history'] = self.knowledge['decision_history'][-100:]

    def update_skill_application(self, skill_id: str, success: bool, reward: float):
        """
        Update skill application results.

        Args:
            skill_id: Skill that was applied
            success: Whether application was successful
            reward: Reward received
        """
        if skill_id in self.knowledge['discovered_skills']:
            skill_data = self.knowledge['discovered_skills'][skill_id]
            skill_data['application_count'] += 1

            # Update success rate
            alpha = 0.1  # Learning rate for success rate
            current_rate = skill_data['success_rate']
            skill_data['success_rate'] = (1 - alpha) * current_rate + alpha * (1.0 if success else 0.0)

            # Update confidence based on results
            if success and reward > 0:
                skill_data['confidence'] = min(1.0, skill_data['confidence'] * 1.1)
            elif not success:
                skill_data['confidence'] = max(0.1, skill_data['confidence'] * 0.9)

            # Record application
            self.knowledge['skill_applications'].append({
                'skill_id': skill_id,
                'timestamp': time.time(),
                'success': success,
                'reward': reward
            })

            # Keep applications manageable
            if len(self.knowledge['skill_applications']) > 500:
                self.knowledge['skill_applications'] = self.knowledge['skill_applications'][-500:]

    def set_environment_analysis(self, analysis: Dict[str, Any]):
        """
        Set environment analysis for context.

        Args:
            analysis: Environment analysis from analyzer
        """
        self.current_context['environment_analysis'] = analysis

        # Extract environmental understanding
        understanding = {
            'state_structure': analysis.get('state_analysis', {}),
            'action_effects': analysis.get('action_analysis', {}),
            'reward_patterns': analysis.get('reward_analysis', {}),
            'environment_type': analysis.get('environment_type', 'unknown')
        }

        env_id = analysis.get('env_id', 'unknown')
        self.knowledge['environmental_understanding'][env_id] = understanding

        self.logger.info(f"Set environment analysis for {env_id}")

    def get_transferable_knowledge(self) -> Dict[str, Any]:
        """
        Get knowledge that can transfer to other environments.

        Returns:
            Transferable knowledge package
        """
        # Filter skills by confidence and success rate
        transferable_skills = {}
        for skill_id, skill_data in self.knowledge['discovered_skills'].items():
            if skill_data['confidence'] > 0.6 and skill_data['success_rate'] > 0.5:
                transferable_skills[skill_id] = {
                    'skill': skill_data['skill'],
                    'confidence': skill_data['confidence'],
                    'success_rate': skill_data['success_rate'],
                    'abstraction_level': skill_data['skill'].get('abstraction_level', 'low')
                }

        return {
            'transferable_skills': transferable_skills,
            'meta_knowledge': self.knowledge['meta_knowledge'].copy(),
            'skill_count': len(transferable_skills)
        }

    def integrate_transferred_knowledge(self, transferred_knowledge: Dict[str, Any]):
        """
        Integrate knowledge transferred from another environment.

        Args:
            transferred_knowledge: Knowledge package from another environment
        """
        transferred_skills = transferred_knowledge.get('transferable_skills', {})

        for skill_id, skill_data in transferred_skills.items():
            if skill_id not in self.knowledge['discovered_skills']:
                # Add transferred skill with reduced confidence
                self.knowledge['discovered_skills'][skill_id] = {
                    'skill': skill_data['skill'],
                    'discovered_at': time.time(),
                    'application_count': 0,
                    'success_rate': 0.0,
                    'confidence': skill_data['confidence'] * 0.7,  # Reduce confidence for new environment
                    'transferred': True,
                    'source_confidence': skill_data['confidence']
                }

                self.logger.info(f"Integrated transferred skill: {skill_data['skill']['name']}")

    def get_symbolic_insights(self) -> Dict[str, Any]:
        """
        Get insights about symbolic reasoning.

        Returns:
            Dictionary of symbolic insights
        """
        # Calculate skill effectiveness
        skill_effectiveness = {}
        for skill_id, skill_data in self.knowledge['discovered_skills'].items():
            if skill_data['application_count'] > 0:
                skill_effectiveness[skill_id] = {
                    'name': skill_data['skill']['name'],
                    'effectiveness': skill_data['success_rate'] * skill_data['confidence'],
                    'applications': skill_data['application_count']
                }

        # Sort by effectiveness
        top_skills = sorted(
            skill_effectiveness.items(),
            key=lambda x: x[1]['effectiveness'],
            reverse=True
        )[:5]

        return {
            'total_skills': len(self.knowledge['discovered_skills']),
            'active_skills': len([s for s in self.knowledge['discovered_skills'].values()
                                  if s['confidence'] > 0.5]),
            'top_skills': dict(top_skills),
            'decision_success_rate': (
                    self.knowledge['meta_knowledge']['successful_decisions'] /
                    max(1, self.knowledge['meta_knowledge']['total_decisions'])
            ),
            'environments_understood': len(self.knowledge['environmental_understanding']),
            'recent_discoveries': self._get_recent_discoveries()
        }

    def _get_recent_discoveries(self) -> List[Dict[str, Any]]:
        """Get recently discovered skills."""
        recent = []
        current_time = time.time()

        for skill_id, skill_data in self.knowledge['discovered_skills'].items():
            time_since_discovery = current_time - skill_data['discovered_at']
            if time_since_discovery < 3600:  # Last hour
                recent.append({
                    'name': skill_data['skill']['name'],
                    'time_ago': time_since_discovery,
                    'confidence': skill_data['confidence']
                })

        return recent

    def save_state(self, env_id: str) -> bool:
        """
        Save symbolic brain state to storage.

        Args:
            env_id: Environment identifier

        Returns:
            Success status
        """
        try:
            # Update timestamp
            self.knowledge['meta_knowledge']['last_updated'] = time.time()

            # Prepare state for saving
            knowledge_state = {
                'knowledge': self.knowledge,
                'config': self.config
            }

            return self.storage.save_knowledge(self.agent_id, env_id, knowledge_state)

        except Exception as e:
            self.logger.error(f"Failed to save symbolic brain state: {e}")
            return False

    def load_state(self, env_id: str) -> bool:
        """
        Load symbolic brain state from storage.

        Args:
            env_id: Environment identifier

        Returns:
            Success status
        """
        try:
            knowledge_state = self.storage.load_knowledge(self.agent_id, env_id)
            if not knowledge_state:
                return False

            # Load knowledge
            self.knowledge = knowledge_state.get('knowledge', self.knowledge)

            self.logger.info(f"Loaded symbolic brain state from {env_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load symbolic brain state: {e}")
            return False