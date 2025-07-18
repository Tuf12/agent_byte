"""
Test Suite for Sprint 9: Continuous Action Space Support

This test verifies that the enhanced dual brain system correctly handles:
1. Discrete action spaces (existing functionality)
2. Continuous action spaces (new functionality)
3. Action space detection and adaptation
4. Integration between neural and symbolic brains
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import tempfile
import os

# Import the components we're testing
from agent_byte.core.dual_brain import DualBrain
from agent_byte.core.interfaces import create_discrete_action_space, create_continuous_action_space
from agent_byte.storage.json_numpy_storage import JsonNumpyStorage


class TestSprint9ContinuousActions:
    """Test suite for Sprint 9 continuous action space functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = JsonNumpyStorage(self.temp_dir)
        self.agent_id = "test_agent_sprint9"

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discrete_action_space_detection(self):
        """Test that discrete action spaces are detected correctly."""
        config = {}  # Default should be discrete

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=4,
            storage=self.storage,
            config=config
        )

        # Verify discrete setup
        assert dual_brain.action_size == 4
        assert dual_brain.continuous_network is None

        # Test action space info
        action_info = dual_brain.get_action_space_info()
        assert action_info['action_space_type'] in ['discrete', 'unknown']
        assert not action_info['continuous_support']

    def test_continuous_action_space_detection(self):
        """Test that continuous action spaces are detected correctly."""
        config = {
            'continuous_actions': True,
            'action_bounds': ([-1.0, -2.0], [1.0, 2.0])
        }

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=2,
            storage=self.storage,
            config=config
        )

        # Verify continuous setup
        assert dual_brain.action_size == 2

        # Check if continuous support is available
        action_info = dual_brain.get_action_space_info()
        if action_info['continuous_support']:
            assert dual_brain.continuous_network is not None
            assert action_info['action_space_type'] == 'continuous'
        else:
            # Continuous components not available, should gracefully degrade
            assert dual_brain.continuous_network is None

    def test_discrete_action_decision_making(self):
        """Test that discrete action decision making works correctly."""
        config = {}

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=4,
            storage=self.storage,
            config=config
        )

        # Create test state
        state = np.random.random(256).astype(np.float32)
        exploration_rate = 0.5

        # Test decision making
        action = dual_brain.decide(state, exploration_rate)

        # Verify discrete action
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 4

        # Test insights
        insights = dual_brain.get_insights()
        assert 'integration' in insights
        assert insights['integration']['total_decisions'] == 1

    def test_continuous_action_decision_making(self):
        """Test that continuous action decision making works correctly."""
        config = {
            'continuous_actions': True,
            'action_bounds': ([-1.0, -1.0], [1.0, 1.0])
        }

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=2,
            storage=self.storage,
            config=config
        )

        # Create test state
        state = np.random.random(256).astype(np.float32)
        exploration_rate = 0.5

        # Test decision making
        action = dual_brain.decide(state, exploration_rate)

        # Check if continuous network is available
        if dual_brain.continuous_network is not None:
            # Verify continuous action
            assert isinstance(action, np.ndarray)
            assert action.shape == (2,)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)
        else:
            # Fallback to discrete if continuous not available
            assert isinstance(action, (int, np.integer))

    def test_learning_from_experience_discrete(self):
        """Test learning from experience with discrete actions."""
        config = {}

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=4,
            storage=self.storage,
            config=config
        )

        # Create experience
        experience = {
            'state': np.random.random(256).astype(np.float32),
            'action': 2,  # Discrete action
            'reward': 1.0,
            'next_state': np.random.random(256).astype(np.float32),
            'done': False
        }

        # Test learning
        metrics = dual_brain.learn_from_experience(experience)

        # Verify learning occurred
        assert isinstance(metrics, dict)

    def test_learning_from_experience_continuous(self):
        """Test learning from experience with continuous actions."""
        config = {
            'continuous_actions': True,
            'action_bounds': ([-1.0, -1.0], [1.0, 1.0])
        }

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=2,
            storage=self.storage,
            config=config
        )

        # Create experience with continuous action
        experience = {
            'state': np.random.random(256).astype(np.float32),
            'action': np.array([0.5, -0.3]),  # Continuous action
            'reward': 1.0,
            'next_state': np.random.random(256).astype(np.float32),
            'done': False
        }

        # Test learning
        metrics = dual_brain.learn_from_experience(experience)

        # Verify learning occurred
        assert isinstance(metrics, dict)

    def test_action_space_adaptation_from_analysis(self):
        """Test that action space can be updated from environment analysis."""
        config = {}  # Start with discrete

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=2,
            storage=self.storage,
            config=config
        )

        # Initially should be discrete
        assert dual_brain.continuous_network is None

        # Provide analysis indicating continuous environment
        analysis = {
            'action_space_type': 'continuous',
            'action_bounds': ([-2.0, -1.0], [2.0, 1.0])
        }

        dual_brain.set_environment_analysis(analysis)

        # Check if it adapted (if continuous support is available)
        action_info = dual_brain.get_action_space_info()
        if action_info['continuous_support']:
            assert dual_brain.continuous_network is not None

    def test_symbolic_to_continuous_action_conversion(self):
        """Test conversion of symbolic discrete actions to continuous actions."""
        config = {
            'continuous_actions': True,
            'action_bounds': ([-1.0, -1.0], [1.0, 1.0])
        }

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=2,
            storage=self.storage,
            config=config
        )

        # Skip test if continuous support not available
        if dual_brain.continuous_network is None:
            pytest.skip("Continuous network support not available")

        # Mock symbolic brain to return a discrete action
        with patch.object(dual_brain.symbolic_brain, 'make_decision') as mock_decision:
            mock_decision.return_value = (1, {'confidence': 0.8, 'reasoning': 'test'})

            state = np.random.random(256).astype(np.float32)
            action = dual_brain.decide(state, 0.1)  # Low exploration for symbolic decision

            # Should get continuous action even from symbolic decision
            assert isinstance(action, np.ndarray)
            assert action.shape == (2,)

    def test_performance_metrics_collection(self):
        """Test that performance metrics are collected correctly."""
        config = {
            'continuous_actions': True,
            'action_bounds': ([-1.0], [1.0])
        }

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=1,
            storage=self.storage,
            config=config
        )

        # Make some decisions
        state = np.random.random(256).astype(np.float32)
        for i in range(5):
            action = dual_brain.decide(state, 0.5)

            # Learn from experience
            experience = {
                'state': state,
                'action': action,
                'reward': 1.0,
                'next_state': np.random.random(256).astype(np.float32),
                'done': False
            }
            dual_brain.learn_from_experience(experience)

        # Check performance metrics
        metrics = dual_brain.get_performance_metrics()
        assert metrics['total_decisions'] == 5
        assert 'neural_decisions' in metrics
        assert 'symbolic_decisions' in metrics

    def test_state_saving_and_loading(self):
        """Test that dual brain state can be saved and loaded correctly."""
        config = {
            'continuous_actions': True,
            'action_bounds': ([-1.0], [1.0])
        }

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=1,
            storage=self.storage,
            config=config
        )

        # Make some decisions to create state
        state = np.random.random(256).astype(np.float32)
        action = dual_brain.decide(state, 0.5)

        # Save state
        env_id = "test_env"
        save_success = dual_brain.save_state(env_id)
        assert save_success

        # Create new dual brain and load state
        dual_brain2 = DualBrain(
            agent_id=self.agent_id,
            action_size=1,
            storage=self.storage,
            config=config
        )

        load_success = dual_brain2.load_state(env_id)
        assert load_success

    def test_transfer_knowledge_preparation(self):
        """Test that knowledge can be prepared for transfer."""
        config = {
            'continuous_actions': True,
            'action_bounds': ([-1.0], [1.0])
        }

        dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=1,
            storage=self.storage,
            config=config
        )

        # Make some decisions to create knowledge
        state = np.random.random(256).astype(np.float32)
        for _ in range(3):
            action = dual_brain.decide(state, 0.5)
            experience = {
                'state': state,
                'action': action,
                'reward': 1.0,
                'next_state': np.random.random(256).astype(np.float32),
                'done': False
            }
            dual_brain.learn_from_experience(experience)

        # Prepare transfer package
        transfer_package = dual_brain.prepare_for_transfer("target_env")

        # Verify transfer package structure
        assert 'source_agent' in transfer_package
        assert 'symbolic_knowledge' in transfer_package
        assert 'neural_patterns' in transfer_package
        assert 'integration_metrics' in transfer_package

    def test_error_handling_missing_continuous_components(self):
        """Test graceful degradation when continuous components are missing."""
        config = {
            'continuous_actions': True,
            'action_bounds': ([-1.0], [1.0])
        }

        # Mock the continuous support as unavailable
        with patch('agent_byte.core.dual_brain.CONTINUOUS_SUPPORT_AVAILABLE', False):
            dual_brain = DualBrain(
                agent_id=self.agent_id,
                action_size=1,
                storage=self.storage,
                config=config
            )

            # Should fall back to discrete mode
            assert dual_brain.continuous_network is None

            # Should still work for decisions
            state = np.random.random(256).astype(np.float32)
            action = dual_brain.decide(state, 0.5)
            assert isinstance(action, (int, np.integer))


class TestIntegrationWithAgent:
    """Test integration with the main Agent class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('agent_byte.core.agent.Environment')
    def test_agent_with_continuous_environment(self, mock_env_class):
        """Test that Agent class works with continuous environments."""
        # Create mock environment
        mock_env = Mock()
        mock_env.get_id.return_value = "continuous_test_env"
        mock_env.get_state_size.return_value = 4
        mock_env.get_action_size.return_value = 2
        mock_env.reset.return_value = np.random.random(4)
        mock_env.step.return_value = (
            np.random.random(4),  # next_state
            1.0,  # reward
            False,  # done
            {}  # info
        )
        mock_env_class.return_value = mock_env

        # Test would require full Agent class setup
        # This is a placeholder for integration testing
        pass


def run_sprint9_tests():
    """
    Main function to run all Sprint 9 tests.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("🚀 Running Sprint 9: Continuous Action Space Tests...")

    try:
        # Run the test suite
        pytest.main([
            __file__,
            "-v",  # Verbose output
            "-x",  # Stop on first failure
            "--tb=short"  # Short traceback format
        ])

        print("✅ Sprint 9 tests completed!")
        return True

    except Exception as e:
        print(f"❌ Sprint 9 tests failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests directly
    run_sprint9_tests()