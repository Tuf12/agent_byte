"""
Unit tests for Agent Byte knowledge system.

This test suite verifies the knowledge components including pattern interpretation,
skill discovery (both template and clustering-based), skill classifiers, and
decision making.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import time
from typing import Dict, Any, List

from agent_byte.knowledge import (
    PatternInterpreter,
    SkillDiscovery,
    SymbolicDecisionMaker,
    TransferMapper,
    SkillClassifier,
    SkillClassifierManager
)
from agent_byte.knowledge.knowledge_system import KnowledgeSystem


class TestPatternInterpreter(unittest.TestCase):
    """Test pattern interpretation functionality."""

    def setUp(self):
        self.interpreter = PatternInterpreter()

    def test_basic_interpretation(self):
        """Test basic pattern interpretation."""
        neural_insights = {
            'pattern_summary': {
                'pattern_stability': 0.8,
                'action_distribution': {0: 10, 1: 20, 2: 5},
                'avg_reward_by_action': {0: 0.5, 1: 1.5, 2: -0.5}
            },
            'learning_trajectory': {
                'loss_trend': -0.01,
                'reward_trend': 0.05,
                'stability': 0.7,
                'is_improving': True
            },
            'emerging_patterns': [
                {
                    'type': 'action_preference',
                    'action': 1,
                    'frequency': 0.6,
                    'average_reward': 1.5,
                    'confidence': 0.8
                }
            ],
            'performance_metrics': {
                'total_reward': 100,
                'episode_count': 50
            }
        }

        interpretation = self.interpreter.interpret(neural_insights)

        # Check basic structure
        self.assertIn('detected_patterns', interpretation)
        self.assertIn('behavioral_state', interpretation)
        self.assertIn('learning_phase', interpretation)
        self.assertIn('skill_indicators', interpretation)
        self.assertIn('confidence', interpretation)

        # Check pattern detection
        patterns = [p['name'] for p in interpretation['detected_patterns']]
        self.assertIn('improvement', patterns)
        self.assertIn('convergence', patterns)

        # Check skill indicators
        self.assertGreater(len(interpretation['skill_indicators']), 0)

    def test_cluster_context_interpretation(self):
        """Test interpretation with cluster context."""
        neural_insights = {
            'pattern_summary': {'pattern_stability': 0.5},
            'learning_trajectory': {'status': 'insufficient_data'},
            'emerging_patterns': [],
            'performance_metrics': {}
        }

        cluster_context = {
            'cluster_id': 3,
            'confidence': 0.9,
            'recommended_action': 2,
            'expected_reward': 1.2,
            'action_confidence': 0.85
        }

        interpretation = self.interpreter.interpret(neural_insights, cluster_context)

        # Check cluster context is included
        self.assertEqual(interpretation['cluster_context'], cluster_context)

        # Check cluster-based skill indicators
        skill_types = [ind['type'] for ind in interpretation['skill_indicators']]
        self.assertIn('cluster_membership', skill_types)
        self.assertIn('cluster_action_preference', skill_types)


class TestSkillDiscovery(unittest.TestCase):
    """Test skill discovery functionality."""

    def setUp(self):
        self.skill_discovery = SkillDiscovery()

    def test_template_based_discovery(self):
        """Test traditional template-based skill discovery."""
        pattern_interpretation = {
            'detected_patterns': [
                {'name': 'exploitation', 'confidence': 0.8, 'details': {'dominant_action': 1, 'dominance': 0.9}}
            ],
            'behavioral_state': 'stable_policy',
            'learning_phase': 'refinement',
            'skill_indicators': [
                {
                    'type': 'successful_action_pattern',
                    'description': 'Action 1 yields positive rewards',
                    'confidence': 0.7,
                    'evidence': {'action': 1, 'frequency': 0.6, 'average_reward': 1.5}
                }
            ],
            'confidence': 0.75
        }

        existing_skills = {}

        discovered = self.skill_discovery.discover(pattern_interpretation, existing_skills)

        self.assertGreater(len(discovered), 0)

        # Check skill structure
        for skill in discovered:
            self.assertIn('id', skill)
            self.assertIn('name', skill)
            self.assertIn('type', skill)
            self.assertIn('confidence', skill)
            self.assertIn('discovered_via', skill)
            self.assertIn('timestamp', skill)

    def test_clustering_based_discovery(self):
        """Test clustering-based skill discovery."""
        # Prepare experience buffer with synthetic data
        for i in range(100):
            latent_state = np.random.randn(16)  # 16D latent space
            action = i % 3
            reward = 1.0 if action == 1 else -0.5

            pattern_interpretation = {
                'behavioral_state': 'exploring',
                'learning_phase': 'early_learning',
                'confidence': 0.5
            }

            self.skill_discovery._add_to_experience_buffer(
                latent_state, action, reward, pattern_interpretation
            )

        # Force clustering
        self.skill_discovery.experiences_since_clustering = self.skill_discovery.clustering_interval

        existing_skills = {}
        discovered = self.skill_discovery.discover({}, existing_skills)

        # Should discover skills from clustering
        cluster_skills = [s for s in discovered if s.get('discovered_via') == 'clustering']
        self.assertGreater(len(cluster_skills), 0)

        # Check cluster skill structure
        for skill in cluster_skills:
            self.assertIn('cluster_id', skill['evidence'])
            self.assertIn('cluster_size', skill['evidence'])
            self.assertIn('dominant_action', skill['evidence'])
            self.assertIn('average_reward', skill['evidence'])

    def test_skill_categorization(self):
        """Test chess-inspired skill categorization."""
        tactical_skill = {
            'confidence': 0.5,
            'abstraction_level': 'concrete',
            'evidence': {'average_reward': 0.5}
        }

        strategic_skill = {
            'confidence': 0.9,
            'abstraction_level': 'strategic',
            'evidence': {}
        }

        self.assertEqual(self.skill_discovery.categorize_skill(tactical_skill), 'tactical')
        self.assertEqual(self.skill_discovery.categorize_skill(strategic_skill), 'strategic')

    def test_skill_scoring(self):
        """Test skill scoring system."""
        skill = {
            'confidence': 0.8,
            'success_rate': 0.7,
            'abstraction_level': 'tactical',
            'evidence': {
                'cluster_size': 50,
                'average_reward': 5.0
            }
        }

        score = self.skill_discovery.score_skill(skill)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.5)  # Should be relatively high


class TestSkillClassifier(unittest.TestCase):
    """Test skill classifier functionality."""

    def setUp(self):
        self.classifier = SkillClassifier("test_skill", model_type="logistic")

    def test_training_sample_collection(self):
        """Test adding training samples."""
        initial_count = self.classifier.training_metrics['total_samples']

        # Add positive sample
        state = np.random.randn(256)
        self.classifier.add_training_sample(state, applied=True, success=True, reward=1.0)

        self.assertEqual(
            self.classifier.training_metrics['total_samples'],
            initial_count + 1
        )
        self.assertEqual(self.classifier.training_metrics['positive_samples'], 1)

        # Add negative sample
        self.classifier.add_training_sample(state, applied=False, success=False, reward=-1.0)

        self.assertEqual(self.classifier.training_metrics['negative_samples'], 1)

    def test_prediction_before_training(self):
        """Test prediction before training returns neutral result."""
        state = np.random.randn(256)
        confidence, should_apply = self.classifier.predict_applicability(state)

        self.assertEqual(confidence, 0.5)
        self.assertFalse(should_apply)
        self.assertFalse(self.classifier.is_trained)

    def test_training_and_prediction(self):
        """Test training and prediction workflow."""
        # FIX: Set random seed for deterministic test results
        np.random.seed(42)

        # Generate synthetic training data
        for i in range(60):
            state = np.random.randn(256)
            # Create pattern: positive samples have positive first feature
            state[0] = 1.0 if i < 30 else -1.0
            label = i < 30  # First 30 are positive

            self.classifier.add_training_sample(
                state,
                applied=label,
                success=label,
                reward=1.0 if label else -1.0
            )

        # Train classifier
        accuracy = self.classifier.train(force=True)
        self.assertIsNotNone(accuracy)
        self.assertGreater(accuracy, 0.5)  # Should learn the pattern
        self.assertTrue(self.classifier.is_trained)

        # Test prediction on new samples (also with fixed seed for consistency)
        np.random.seed(123)  # Different seed for test data
        positive_state = np.random.randn(256)
        positive_state[0] = 2.0
        conf_pos, should_apply_pos = self.classifier.predict_applicability(positive_state)

        negative_state = np.random.randn(256)
        negative_state[0] = -2.0
        conf_neg, should_apply_neg = self.classifier.predict_applicability(negative_state)

        # DEBUG: Print the actual values
        print(f"DEBUG: conf_pos={conf_pos:.3f}, should_apply_pos={should_apply_pos}")
        print(f"DEBUG: conf_neg={conf_neg:.3f}, should_apply_neg={should_apply_neg}")
        print(f"DEBUG: threshold={self.classifier.confidence_threshold}")

        # Positive state should have higher confidence
        self.assertGreater(conf_pos, conf_neg)
        self.assertTrue(should_apply_pos)
        self.assertFalse(should_apply_neg)

    def test_neural_classifier(self):
        """Test neural network classifier."""
        neural_classifier = SkillClassifier("test_neural", model_type="neural")

        # Add training samples
        for i in range(100):
            state = np.random.randn(256)
            label = i < 50
            neural_classifier.add_training_sample(
                state,
                applied=label,
                success=label,
                reward=1.0 if label else -1.0
            )

        # Train
        accuracy = neural_classifier.train(force=True)
        self.assertIsNotNone(accuracy)
        self.assertTrue(neural_classifier.is_trained)


class TestSkillClassifierManager(unittest.TestCase):
    """Test skill classifier manager."""

    def setUp(self):
        self.manager = SkillClassifierManager(retrain_interval=10)

    def test_classifier_creation(self):
        """Test automatic classifier creation."""
        classifier = self.manager.get_or_create_classifier("skill_1")
        self.assertIsInstance(classifier, SkillClassifier)

        # Getting again should return same instance
        classifier2 = self.manager.get_or_create_classifier("skill_1")
        self.assertIs(classifier, classifier2)

    def test_experience_management(self):
        """Test adding experiences through manager."""
        state = np.random.randn(256)

        self.manager.add_experience(
            "skill_1", state, applied=True, success=True, reward=1.0
        )

        classifier = self.manager.classifiers["skill_1"]
        self.assertEqual(classifier.training_metrics['total_samples'], 1)

    def test_prediction_aggregation(self):
        """Test predicting applicable skills."""
        # FIX: Set random seed for deterministic results
        np.random.seed(42)

        # Create and train multiple classifiers
        for skill_id in ["skill_1", "skill_2", "skill_3"]:
            classifier = self.manager.get_or_create_classifier(skill_id)

            # Add synthetic data
            for i in range(60):
                state = np.random.randn(256)
                state[int(skill_id[-1])] = 1.0 if i < 30 else -1.0
                label = i < 30

                classifier.add_training_sample(
                    state, applied=label, success=label,
                    reward=1.0 if label else -1.0
                )

            classifier.train(force=True)

        # Test prediction
        test_state = np.random.randn(256)
        test_state[1] = 2.0  # Should favor skill_1

        predictions = self.manager.predict_applicable_skills(
            test_state,
            ["skill_1", "skill_2", "skill_3"],
            threshold=0.0
        )

        self.assertEqual(len(predictions), 3)
        self.assertEqual(predictions[0][0], "skill_1")  # Should be ranked first

    def test_periodic_retraining(self):
        """Test periodic retraining functionality."""
        # Create classifier with data
        classifier = self.manager.get_or_create_classifier("skill_1")
        for i in range(60):
            state = np.random.randn(256)
            classifier.add_training_sample(
                state, applied=True, success=True, reward=1.0
            )

        # Simulate episodes
        for _ in range(9):
            self.manager.periodic_retrain()

        self.assertEqual(self.manager.episodes_since_retrain, 9)
        self.assertFalse(classifier.is_trained)

        # 10th episode should trigger retraining
        self.manager.periodic_retrain()

        self.assertEqual(self.manager.episodes_since_retrain, 0)
        self.assertTrue(classifier.is_trained)


class TestSymbolicDecisionMaker(unittest.TestCase):
    """Test symbolic decision making."""

    def setUp(self):
        self.decision_maker = SymbolicDecisionMaker()

    def test_strategy_selection(self):
        """Test decision strategy selection."""
        # Minimal context
        minimal_skills = {}
        minimal_patterns = []

        strategy_name, _ = self.decision_maker._select_strategy(
            minimal_skills, minimal_patterns, 0.8
        )
        self.assertEqual(strategy_name, 'exploration_based')

        # Rich context
        rich_skills = {
            f"skill_{i}": {'confidence': 0.8} for i in range(5)
        }
        rich_patterns = [{'pattern': i} for i in range(5)]

        strategy_name, _ = self.decision_maker._select_strategy(
            rich_skills, rich_patterns, 0.1
        )
        self.assertIn(strategy_name, ['skill_based', 'pattern_based'])

    def test_skill_based_decision_with_classifiers(self):
        """Test skill-based decision using classifier predictions."""
        context = {
            'state': np.random.randn(256),
            'q_values': np.array([0.1, 0.5, 0.3]),
            'exploration_rate': 0.1,
            'discovered_skills': {
                'skill_1': {
                    'skill': {
                        'name': 'test_skill',
                        'type': 'action_optimization',
                        'description': 'Test skill',
                        'parameters': {'target_action': 1}
                    },
                    'confidence': 0.8
                }
            },
            'recent_patterns': [],
            'environment_analysis': {},
            'skill_predictions': [('skill_1', 0.9)]  # High confidence prediction
        }

        action, reasoning, confidence = self.decision_maker.decide(context)

        self.assertEqual(action, 1)  # Should select predicted skill's action
        self.assertIn('Classifier selected', reasoning)
        self.assertEqual(confidence, 0.9)


class TestKnowledgeSystem(unittest.TestCase):
    """Test integrated knowledge system."""

    def setUp(self):
        self.knowledge_system = KnowledgeSystem()

    def test_neural_insights_processing(self):
        """Test processing neural insights through the system."""
        neural_insights = {
            'pattern_summary': {
                'pattern_stability': 0.7,
                'action_distribution': {0: 5, 1: 15, 2: 3}
            },
            'learning_trajectory': {
                'is_improving': True,
                'stability': 0.6
            },
            'emerging_patterns': [
                {
                    'type': 'action_preference',
                    'action': 1,
                    'average_reward': 2.0,
                    'confidence': 0.8
                }
            ],
            'performance_metrics': {}
        }

        result = self.knowledge_system.process_neural_insights('test_env', neural_insights)

        self.assertIn('pattern_interpretation', result)
        self.assertIn('discovered_skills', result)
        self.assertIn('skill_relationships', result)
        self.assertIn('recommendations', result)

    def test_knowledge_transfer(self):
        """Test knowledge transfer between environments."""
        # Create skills in source environment
        self.knowledge_system._initialize_environment_knowledge('source_env')
        source_knowledge = self.knowledge_system.knowledge_base['environments']['source_env']

        source_knowledge['discovered_skills']['skill_1'] = {
            'skill': {
                'name': 'transferable_skill',
                'type': 'action_optimization',
                'abstraction_level': 'tactical'
            },
            'confidence': 0.8,
            'success_rate': 0.9,
            'application_count': 50
        }

        # Prepare transfer package
        transfer_package = self.knowledge_system.prepare_transfer_package('source_env')

        self.assertIn('transferable_skills', transfer_package)
        self.assertGreater(len(transfer_package['transferable_skills']), 0)

        # Apply to target
        target_analysis = {
            'env_id': 'target_env',
            'state_size': 10,
            'action_size': 3
        }

        result = self.knowledge_system.apply_transfer('target_env', transfer_package, target_analysis)

        self.assertIn('transferable_skills', result)
        self.assertGreater(result['transfer_confidence'], 0)


class TestTransferMapper(unittest.TestCase):
    """Test transfer mapping functionality."""

    def setUp(self):
        self.transfer_mapper = TransferMapper()

    def test_environment_compatibility(self):
        """Test environment compatibility analysis."""
        source_knowledge = {
            'action_size': 3,
            'state_size': 10,
            'reward_type': 'dense_continuous',
            'environment_type': 'control'
        }

        target_analysis = {
            'action_size': 3,
            'state_size': 12,
            'reward_analysis': {'reward_types': 'dense_continuous'},
            'environment_type': 'control'
        }

        compatibility = self.transfer_mapper._analyze_environment_compatibility(
            source_knowledge, target_analysis
        )

        self.assertIn('action_space_compatibility', compatibility)
        self.assertIn('state_space_compatibility', compatibility)
        self.assertIn('overall_compatibility', compatibility)

        # Should have high compatibility
        self.assertGreater(compatibility['overall_compatibility'], 0.7)

    def test_skill_mapping_strategies(self):
        """Test different skill mapping strategies."""
        skill_data = {
            'skill': {
                'type': 'action_optimization',
                'abstraction_level': 'concrete'
            },
            'confidence': 0.8
        }

        high_compatibility = {'overall_compatibility': 0.9}
        low_compatibility = {'overall_compatibility': 0.4}

        # Test strategy selection
        high_strategy = self.transfer_mapper._select_mapping_strategy(
            skill_data, high_compatibility
        )
        self.assertEqual(high_strategy, 'direct_transfer')

        low_strategy = self.transfer_mapper._select_mapping_strategy(
            skill_data, low_compatibility
        )
        self.assertEqual(low_strategy, 'compositional_transfer')


def run_all_tests():
    """Run all knowledge system tests."""
    test_classes = [
        TestPatternInterpreter,
        TestSkillDiscovery,
        TestSkillClassifier,
        TestSkillClassifierManager,
        TestSymbolicDecisionMaker,
        TestKnowledgeSystem,
        TestTransferMapper
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


# Add this new test class to your test_knowledge.py file:

class TestTransferValidation(unittest.TestCase):
    """Test Sprint 6 transfer validation functionality."""

    def setUp(self):
        self.knowledge_system = KnowledgeSystem()
        self.transfer_mapper = TransferMapper()

    def test_transfer_validation_pipeline(self):
        """Test the complete transfer validation pipeline."""
        # Setup source environment with skills
        self.knowledge_system._initialize_environment_knowledge('source_env')
        source_env = self.knowledge_system.knowledge_base['environments']['source_env']

        # Add a transferable skill
        source_env['discovered_skills']['test_skill'] = {
            'skill': {
                'name': 'test_transferable_skill',
                'type': 'action_optimization',
                'abstraction_level': 'tactical'
            },
            'confidence': 0.8,
            'success_rate': 0.7,
            'application_count': 20
        }

        # Prepare and apply transfer
        transfer_package = self.knowledge_system.prepare_transfer_package('source_env')
        target_analysis = {
            'env_id': 'target_env',
            'state_size': 10,
            'action_size': 3
        }

        result = self.knowledge_system.apply_transfer('target_env', transfer_package, target_analysis)

        # Verify pending validation was created
        target_env = self.knowledge_system.knowledge_base['environments']['target_env']
        self.assertIn('pending_validation', target_env)
        self.assertIn('transferred_skills', target_env['pending_validation'])

    def test_validate_active_transfers(self):
        """Test the validate_active_transfers method."""
        # Setup environment with pending validation
        self.knowledge_system._initialize_environment_knowledge('test_env')
        env_knowledge = self.knowledge_system.knowledge_base['environments']['test_env']

        # Add transferred skill and pending validation
        env_knowledge['discovered_skills']['transferred_skill'] = {
            'skill': {'name': 'test_skill'},
            'confidence': 0.8,
            'success_rate': 0.6,
            'application_count': 10
        }

        env_knowledge['pending_validation'] = {
            'pre_transfer_performance': {'average_reward': 0.5, 'step_count': 100},
            'transferred_skills': {'transferred_skill': {}},
            'transfer_timestamp': time.time(),
            'source_environment': 'source_env'
        }

        # Test validation with improved performance
        current_performance = {
            'average_reward': 0.8,  # Improvement
            'episode_length': 50,
            'step_count': 150,
            'timestamp': time.time()
        }

        result = self.knowledge_system.validate_active_transfers('test_env', current_performance)

        # Should return validation result
        self.assertIsNotNone(result)
        self.assertIn('overall_success', result)
        self.assertIn('performance_delta', result)

    def test_transfer_validation_report(self):
        """Test transfer validation reporting."""
        # Setup some transfer metrics
        env_pair = "source_env->test_env"
        self.knowledge_system.transfer_mapper.transfer_success_metrics[env_pair] = {
            'attempts': 5,
            'total_confidence': 3.5,
            'total_skills': 10,
            'average_improvement': 0.3
        }

        report = self.knowledge_system.get_transfer_validation_report('test_env')

        self.assertIn('target_environment', report)
        self.assertIn('transfer_metrics', report)
        self.assertIn('total_transfers_to_env', report)
        self.assertEqual(report['target_environment'], 'test_env')

    def test_transfer_mapper_validation(self):
        """Test TransferMapper validation functionality."""
        # Test skills with different success rates
        transferred_skills = {
            'good_skill': {
                'confidence': 0.8,
                'success_rate': 0.7  # Good performance
            },
            'bad_skill': {
                'confidence': 0.6,
                'success_rate': 0.3  # Poor performance
            }
        }

        target_performance = {
            'average_reward': 1.0,
            'step_count': 200
        }

        # Mock baseline in validation_metrics
        env_pair = "source->target"
        self.transfer_mapper.validation_metrics[env_pair] = {
            'baseline_performance': {
                'average_reward': 0.5,
                'step_count': 100
            }
        }

        result = self.transfer_mapper.validate_transfer(
            'source', 'target', transferred_skills, target_performance
        )

        # Check validation result structure
        self.assertIn('performance_delta', result)
        self.assertIn('overall_success', result)
        self.assertIn('skill_validations', result)
        self.assertIn('recommendations', result)

        # Performance should show improvement
        self.assertEqual(result['performance_delta'], 0.5)  # 1.0 - 0.5
        self.assertTrue(result['overall_success'])  # > 0.1 threshold

        # Check skill validations
        self.assertEqual(result['skill_validations']['good_skill']['validation_status'], 'success')
        self.assertEqual(result['skill_validations']['bad_skill']['validation_status'], 'needs_adaptation')

    def test_adaptive_transfer_strategy(self):
        """Test adaptive transfer strategy selection."""
        # Setup historical performance data
        env_pair = "env1->env2"
        self.transfer_mapper.transfer_success_metrics[env_pair] = {
            'strategy_performance': {
                'direct_transfer': 0.8,
                'scaled_transfer': 0.6,
                'abstract_transfer': 0.4
            }
        }

        strategy = self.transfer_mapper.get_adaptive_transfer_strategy('env1', 'env2')

        # Should select the best performing strategy
        self.assertEqual(strategy, 'direct_transfer')

    def test_validation_recommendations(self):
        """Test validation recommendation generation."""
        skill_validations = {
            'skill1': {'validation_status': 'success'},
            'skill2': {'validation_status': 'success'},
            'skill3': {'validation_status': 'needs_adaptation'}
        }

        recommendations = self.transfer_mapper._generate_validation_recommendations(skill_validations)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # Should indicate mostly successful transfer
        rec_text = ' '.join(recommendations)
        self.assertIn('successful', rec_text.lower())


# Also update the run_all_tests function to include the new test class:
def run_all_tests():
    """Run all knowledge system tests."""
    test_classes = [
        TestPatternInterpreter,
        TestSkillDiscovery,
        TestSkillClassifier,
        TestSkillClassifierManager,
        TestSymbolicDecisionMaker,
        TestKnowledgeSystem,
        TestTransferMapper,
        TestTransferValidation  # NEW: Add Sprint 6 tests
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()
if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)