"""
Dual brain integration for Agent Byte.

This module integrates the neural and symbolic brains into a unified
decision-making system with bidirectional communication.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from .neural_brain import NeuralBrain
from .symbolic_brain import SymbolicBrain
from ..storage.base import StorageBase


class DualBrain:
    """
    Integrated dual brain system combining neural and symbolic reasoning.

    This class manages the interaction between the neural brain (pattern learning)
    and symbolic brain (high-level reasoning) to create an intelligent agent.
    """

    def __init__(self,
                 agent_id: str,
                 action_size: int,
                 storage: StorageBase,
                 config: Dict[str, Any]):
        """
        Initialize the dual brain system.

        Args:
            agent_id: Unique identifier for the agent
            action_size: Number of possible actions
            storage: Storage backend for persistence
            config: Configuration parameters
        """
        self.agent_id = agent_id
        self.action_size = action_size
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger(f"DualBrain-{agent_id}")

        # Initialize both brains
        self.neural_brain = NeuralBrain(agent_id, action_size, storage, config)
        self.symbolic_brain = SymbolicBrain(agent_id, storage, config)

        # Integration state
        self.integration_state = {
            'decisions_made': 0,
            'neural_decisions': 0,
            'symbolic_decisions': 0,
            'integration_active': True,
            'last_sync': None
        }

        # Current episode state
        self.current_experience = None
        self.episode_active = False

        self.logger.info("Dual brain system initialized")

    def set_environment_analysis(self, analysis: Dict[str, Any]):
        """
        Set environment analysis for both brains.

        Args:
            analysis: Environment analysis from analyzer
        """
        # Symbolic brain uses full analysis
        self.symbolic_brain.set_environment_analysis(analysis)

        # Neural brain might use simplified version
        self.logger.info("Environment analysis set for dual brain")

    def decide(self, state: np.ndarray, exploration_rate: float) -> int:
        """
        Make a decision using the dual brain system.

        Args:
            state: Current state (256-dimensional)
            exploration_rate: Current exploration rate

        Returns:
            Selected action
        """
        # Get neural decision
        neural_action, neural_info = self.neural_brain.select_action(state, exploration_rate)

        # Get neural insights for symbolic brain
        neural_insights = self.neural_brain.get_neural_insights()

        # Symbolic brain interprets patterns
        pattern_interpretation = self.symbolic_brain.interpret_neural_patterns(neural_insights)

        # Discover new skills if any
        discovered_skills = self.symbolic_brain.discover_skills(pattern_interpretation)

        # Get symbolic decision
        q_values = np.array(neural_info['q_values'])
        symbolic_action, symbolic_info = self.symbolic_brain.make_decision(
            state, q_values, exploration_rate
        )

        # Integration decision
        if symbolic_action is not None and symbolic_info['confidence'] > self.config.get('symbolic_decision_threshold',
                                                                                         0.3):
            # Use symbolic decision
            final_action = symbolic_action
            decision_source = 'symbolic'
            self.integration_state['symbolic_decisions'] += 1

            self.logger.debug(f"Using symbolic decision: {symbolic_info['reasoning']}")
        else:
            # Use neural decision
            final_action = neural_action
            decision_source = 'neural'
            self.integration_state['neural_decisions'] += 1

        # Update integration state
        self.integration_state['decisions_made'] += 1

        # Store current experience info
        self.current_experience = {
            'state': state,
            'action': final_action,
            'decision_source': decision_source,
            'neural_info': neural_info,
            'symbolic_info': symbolic_info,
            'discovered_skills': discovered_skills
        }

        return final_action

    def learn(self, experience: Dict[str, Any]):
        """
        Learn from experience using both brains.

        Args:
            experience: Experience dictionary with state, action, reward, next_state, done
        """
        # Store experience in neural brain
        self.neural_brain.store_experience(
            experience['state'],
            experience['action'],
            experience['reward'],
            experience['next_state'],
            experience['done']
        )

        # Neural learning
        loss = self.neural_brain.learn()

        # Update symbolic brain based on results
        if self.current_experience:
            # Check if a skill was applied
            symbolic_info = self.current_experience.get('symbolic_info', {})
            if symbolic_info.get('decision_type') == 'symbolic':
                # Determine success based on reward
                success = experience['reward'] > 0

                # Update skill application
                # (In a real implementation, we'd track which specific skill was used)
                # For now, we'll update based on the reasoning
                reasoning = symbolic_info.get('reasoning', '')
                if 'skill:' in reasoning:
                    # Extract skill ID from reasoning (simplified)
                    skill_id = reasoning.split('skill:')[1].split()[0]
                    self.symbolic_brain.update_skill_application(
                        skill_id, success, experience['reward']
                    )

        # Periodic synchronization
        if self.integration_state['decisions_made'] % 100 == 0:
            self._synchronize_brains()

    def _synchronize_brains(self):
        """Synchronize knowledge between brains."""
        # Get neural insights
        neural_insights = self.neural_brain.get_neural_insights()

        # Symbolic brain interprets and discovers
        pattern_interpretation = self.symbolic_brain.interpret_neural_patterns(neural_insights)
        discovered_skills = self.symbolic_brain.discover_skills(pattern_interpretation)

        if discovered_skills:
            self.logger.info(f"Synchronization discovered {len(discovered_skills)} new skills")

        self.integration_state['last_sync'] = self.integration_state['decisions_made']

    def start_episode(self):
        """Start a new episode."""
        self.episode_active = True
        self.neural_brain.reset_episode()

    def end_episode(self):
        """End current episode."""
        self.episode_active = False

        # Final synchronization
        self._synchronize_brains()

    def get_insights(self) -> Dict[str, Any]:
        """
        Get insights from the dual brain system.

        Returns:
            Combined insights from both brains
        """
        neural_insights = self.neural_brain.get_neural_insights()
        symbolic_insights = self.symbolic_brain.get_symbolic_insights()

        # Calculate integration metrics
        total_decisions = max(1, self.integration_state['decisions_made'])
        neural_ratio = self.integration_state['neural_decisions'] / total_decisions
        symbolic_ratio = self.integration_state['symbolic_decisions'] / total_decisions

        return {
            'neural': neural_insights,
            'symbolic': symbolic_insights,
            'integration': {
                'total_decisions': self.integration_state['decisions_made'],
                'neural_ratio': neural_ratio,
                'symbolic_ratio': symbolic_ratio,
                'balance': 1.0 - abs(neural_ratio - symbolic_ratio),
                'last_sync': self.integration_state['last_sync']
            }
        }

    def save_state(self, env_id: str) -> bool:
        """
        Save dual brain state.

        Args:
            env_id: Environment identifier

        Returns:
            Success status
        """
        # Save both brain states
        neural_saved = self.neural_brain.save_state(env_id)
        symbolic_saved = self.symbolic_brain.save_state(env_id)

        return neural_saved and symbolic_saved

    def load_state(self, env_id: str) -> bool:
        """
        Load dual brain state.

        Args:
            env_id: Environment identifier

        Returns:
            Success status
        """
        # Load both brain states
        neural_loaded = self.neural_brain.load_state(env_id)
        symbolic_loaded = self.symbolic_brain.load_state(env_id)

        return neural_loaded and symbolic_loaded

    def prepare_for_transfer(self, target_env_id: str) -> Dict[str, Any]:
        """
        Prepare knowledge for transfer to another environment.

        Args:
            target_env_id: Target environment ID

        Returns:
            Transfer package
        """
        # Get transferable knowledge from symbolic brain
        symbolic_knowledge = self.symbolic_brain.get_transferable_knowledge()

        # Get neural patterns that might transfer
        neural_insights = self.neural_brain.get_neural_insights()
        pattern_summary = neural_insights.get('pattern_summary', {})

        return {
            'source_agent': self.agent_id,
            'symbolic_knowledge': symbolic_knowledge,
            'neural_patterns': {
                'pattern_stability': pattern_summary.get('pattern_stability', 0),
                'action_preferences': pattern_summary.get('avg_reward_by_action', {}),
                'learning_trajectory': neural_insights.get('learning_trajectory', {})
            },
            'integration_metrics': {
                'balance': self.get_insights()['integration']['balance'],
                'total_experience': neural_insights.get('training_steps', 0)
            }
        }

    def apply_transfer(self, transfer_package: Dict[str, Any]):
        """
        Apply transferred knowledge from another environment.

        Args:
            transfer_package: Transfer package from another environment
        """
        # Apply symbolic knowledge
        symbolic_knowledge = transfer_package.get('symbolic_knowledge', {})
        self.symbolic_brain.integrate_transferred_knowledge(symbolic_knowledge)

        # Neural patterns inform initial biases
        neural_patterns = transfer_package.get('neural_patterns', {})

        # Log transfer
        self.logger.info(
            f"Applied transfer from {transfer_package.get('source_agent', 'unknown')}: "
            f"{symbolic_knowledge.get('skill_count', 0)} skills transferred"
        )