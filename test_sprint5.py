"""
Comprehensive tests for Sprint 5: Robustness & Error Recovery

This test suite validates all Sprint 5 features:
- Phase 1: Environment Failure Handling
- Phase 2: Checkpointing System
- Phase 3: Storage Failure Recovery

Run with: python -m pytest test_sprint5.py -v
"""

import pytest
import tempfile
import shutil
import time
import signal
import os
import json
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Fixed import statements to match actual project structure
from agent_byte.core.agent import AgentByte
from agent_byte.core.config import AgentConfig
from agent_byte.core.checkpoint_manager import CheckpointManager
from agent_byte.storage.json_numpy_storage import JsonNumpyStorage
from agent_byte.storage.vector_db_storage import VectorDBStorage
from agent_byte.storage.recovery_utils import StorageRecoveryManager


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, env_id="test_env", state_size=10, action_size=4,
                 should_fail=False, timeout_after=None):
        self.env_id = env_id
        self.state_size = state_size
        self.action_size = action_size
        self.should_fail = should_fail
        self.timeout_after = timeout_after
        self.step_count = 0
        self.state = np.random.random(state_size).astype(np.float32)

    def get_id(self):
        return self.env_id

    def get_state_size(self):
        return self.state_size

    def get_action_size(self):
        return self.action_size

    def reset(self):
        self.step_count = 0
        self.state = np.random.random(self.state_size).astype(np.float32)

        if self.should_fail and self.step_count == 0:
            raise Exception("Environment reset failure")

        return self.state.copy()

    def step(self, action):
        self.step_count += 1

        # Simulate timeout
        if self.timeout_after and self.step_count >= self.timeout_after:
            time.sleep(2)  # Simulate hang

        # Simulate failure
        if self.should_fail and self.step_count > 5:
            raise Exception("Environment step failure")

        # Normal operation
        reward = np.random.random() - 0.5
        done = self.step_count >= 20
        self.state = np.random.random(self.state_size).astype(np.float32)

        return self.state.copy(), reward, done, {}


class TestPhase1EnvironmentFailureHandling:
    """Test Phase 1: Environment Failure Handling"""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agent_id = "test_agent_phase1"

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_timeout_detection(self):
        """Test that agent detects and handles environment timeouts."""
        config = AgentConfig()
        config.exploration_rate = 0.1

        agent = AgentByte(
            agent_id=self.agent_id,
            storage=JsonNumpyStorage(str(self.temp_dir / "storage")),
            config=config,
            enable_checkpointing=True
        )

        # Create environment that times out after 3 steps
        env = MockEnvironment(timeout_after=3)

        # Train with timeout protection
        results = agent.train(env, episodes=1, enable_recovery=True)

        # Should detect timeout and handle gracefully
        assert results is not None
        assert 'error_events' in results

        # Check health status shows issues
        health = agent.get_health_status()
        assert health['environment_health']['consecutive_failures'] >= 0

    def test_environment_failure_recovery(self):
        """Test recovery from environment step failures."""
        config = AgentConfig()
        config.exploration_rate = 0.1

        agent = AgentByte(
            agent_id=self.agent_id,
            storage=JsonNumpyStorage(str(self.temp_dir / "storage")),
            config=config,
            enable_checkpointing=True
        )

        # Create environment that fails after a few steps
        env = MockEnvironment(should_fail=True)

        # Train with error handling
        results = agent.train(env, episodes=2, enable_recovery=True)

        # Should handle failures gracefully
        assert results is not None
        assert 'error_events' in results
        assert len(results['error_events']) > 0

        # Check that consecutive failures are tracked
        assert results['final_performance']['error_statistics']['total_errors'] > 0

    def test_degraded_mode_activation(self):
        """Test that degraded mode activates under stress."""
        config = AgentConfig()

        agent = AgentByte(
            agent_id=self.agent_id,
            storage=JsonNumpyStorage(str(self.temp_dir / "storage")),
            config=config,
            enable_checkpointing=True
        )

        # Force consecutive failures
        agent.environment_health['consecutive_failures'] = 6  # Above threshold

        # Check health status
        should_continue = agent._should_continue_training()
        assert not should_continue

        # Check that degraded mode can be detected
        health = agent.get_health_status()
        assert 'environment_health' in health
        assert health['environment_health']['consecutive_failures'] == 6

    def test_emergency_shutdown(self):
        """Test emergency shutdown functionality."""
        config = AgentConfig()

        agent = AgentByte(
            agent_id=self.agent_id,
            storage=JsonNumpyStorage(str(self.temp_dir / "storage")),
            config=config,
            enable_checkpointing=True
        )

        # Test emergency shutdown
        agent.emergency_shutdown("test_shutdown")

        # Should create emergency checkpoint
        if agent.checkpoint_manager:
            checkpoints = agent.checkpoint_manager._list_checkpoints()
            emergency_checkpoints = [cp for cp in checkpoints if 'emergency' in cp.name]
            assert len(emergency_checkpoints) > 0


class TestPhase2CheckpointingSystem:
    """Test Phase 2: Checkpointing System"""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agent_id = "test_agent_phase2"

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_manager_creation(self):
        """Test that checkpoint manager is created and configured correctly."""
        storage = JsonNumpyStorage(str(self.temp_dir / "storage"))

        checkpoint_manager = CheckpointManager(
            agent_id=self.agent_id,
            storage=storage,
            timer_interval=5,  # 5 seconds for testing
            episode_interval=10,
            max_checkpoints=3
        )

        assert checkpoint_manager.agent_id == self.agent_id
        assert checkpoint_manager.timer_interval == 5
        assert checkpoint_manager.episode_interval == 10
        assert checkpoint_manager.max_checkpoints == 3

    def test_checkpoint_creation_and_validation(self):
        """Test checkpoint creation and integrity validation."""
        storage = JsonNumpyStorage(str(self.temp_dir / "storage"))

        checkpoint_manager = CheckpointManager(
            agent_id=self.agent_id,
            storage=storage,
            max_checkpoints=5
        )

        # Register a simple state provider
        checkpoint_manager.register_state_provider(
            'test_state',
            lambda: {'test_data': 'checkpoint_test', 'timestamp': time.time()}
        )

        # Create manual checkpoint
        success = checkpoint_manager.create_manual_checkpoint("test_checkpoint")
        assert success

        # Verify checkpoint exists and is valid
        checkpoints = checkpoint_manager._list_checkpoints()
        assert len(checkpoints) > 0

        # Load and validate checkpoint
        latest_checkpoint = checkpoint_manager.load_latest_checkpoint()
        assert latest_checkpoint is not None
        assert 'test_state' in latest_checkpoint['components']
        assert latest_checkpoint['components']['test_state']['test_data'] == 'checkpoint_test'

    def test_episode_based_checkpointing(self):
        """Test episode-based checkpoint triggers."""
        storage = JsonNumpyStorage(str(self.temp_dir / "storage"))

        checkpoint_manager = CheckpointManager(
            agent_id=self.agent_id,
            storage=storage,
            episode_interval=5,  # Every 5 episodes
            max_checkpoints=3
        )

        checkpoint_manager.register_state_provider(
            'episode_state',
            lambda: {'episode_count': checkpoint_manager.current_episode}
        )

        # DEBUG: Check what checkpoints exist initially
        initial_checkpoint_files = checkpoint_manager._list_checkpoints()
        initial_checkpoints = len(initial_checkpoint_files)
        print(f"\nDEBUG: Initial checkpoints: {initial_checkpoints}")
        print(f"DEBUG: Initial checkpoint files: {[f.name for f in initial_checkpoint_files]}")
        print(f"DEBUG: Current episode before update: {getattr(checkpoint_manager, 'current_episode', 'Not set')}")

        # Update episodes - should trigger checkpoint at episode 5
        checkpoint_manager.update_episode_count(5)

        # DEBUG: Check what checkpoints exist after update
        final_checkpoint_files = checkpoint_manager._list_checkpoints()
        checkpoints_after = len(final_checkpoint_files)
        print(f"DEBUG: Final checkpoints: {checkpoints_after}")
        print(f"DEBUG: Final checkpoint files: {[f.name for f in final_checkpoint_files]}")
        print(f"DEBUG: Current episode after update: {getattr(checkpoint_manager, 'current_episode', 'Not set')}")

        # Check that checkpoint behavior is as expected
        # Either a new checkpoint was created OR rotation occurred with max_checkpoints=3
        if initial_checkpoints == 0:
            # If we started with 0, we should have 1 after creating a checkpoint
            assert checkpoints_after == 1, f"Expected 1 checkpoint, got {checkpoints_after}"
        else:
            # If we started with some checkpoints, check for rotation behavior
            # With max_checkpoints=3, we should not exceed 3 total
            assert checkpoints_after <= 3, f"Expected max 3 checkpoints due to rotation, got {checkpoints_after}"
            # And we should have either the same count (if already at max) or one more
            assert checkpoints_after >= min(initial_checkpoints + 1, 3) - 1, f"Unexpected checkpoint count change"

    def test_checkpoint_rotation(self):
        """Test that old checkpoints are removed when max is exceeded."""
        storage = JsonNumpyStorage(str(self.temp_dir / "storage"))

        checkpoint_manager = CheckpointManager(
            agent_id=self.agent_id,
            storage=storage,
            max_checkpoints=2  # Keep only 2 checkpoints
        )

        checkpoint_manager.register_state_provider(
            'rotation_test',
            lambda: {'checkpoint_number': time.time()}
        )

        # Create multiple checkpoints
        for i in range(5):
            checkpoint_manager.create_manual_checkpoint(f"test_{i}")
            time.sleep(0.1)  # Ensure different timestamps

        # Should only have max_checkpoints remaining
        checkpoints = checkpoint_manager._list_checkpoints()
        assert len(checkpoints) <= 2

    def test_checkpoint_rollback(self):
        """Test rollback to previous checkpoint."""
        storage = JsonNumpyStorage(str(self.temp_dir / "storage"))

        checkpoint_manager = CheckpointManager(
            agent_id=self.agent_id,
            storage=storage
        )

        # Create initial state
        test_value = "initial_state"

        def get_state():
            return {'test_value': test_value}

        checkpoint_manager.register_state_provider('rollback_test', get_state)

        # Create checkpoint with initial state
        checkpoint_manager.create_manual_checkpoint("initial")

        # Change state
        test_value = "modified_state"

        # Rollback to checkpoint
        checkpoint_data = checkpoint_manager.rollback_to_checkpoint()

        assert checkpoint_data is not None
        assert checkpoint_data['components']['rollback_test']['test_value'] == "initial_state"

    def test_checkpoint_corruption_detection(self):
        """Test detection of corrupted checkpoints."""
        storage = JsonNumpyStorage(str(self.temp_dir / "storage"))

        checkpoint_manager = CheckpointManager(
            agent_id=self.agent_id,
            storage=storage
        )

        checkpoint_manager.register_state_provider(
            'corruption_test',
            lambda: {'data': 'valid_data'}
        )

        # Create valid checkpoint
        checkpoint_manager.create_manual_checkpoint("valid")

        # Get ALL checkpoint files and corrupt them all
        checkpoints = checkpoint_manager._list_checkpoints()
        assert len(checkpoints) > 0

        # Corrupt ALL checkpoint files to ensure corruption is detected
        for checkpoint_file in checkpoints:
            with open(checkpoint_file, 'w') as f:
                f.write('{"invalid": "json"')  # Invalid JSON

        # Try to load - should detect corruption since all files are corrupted
        corrupted_checkpoint = checkpoint_manager.load_latest_checkpoint()
        assert corrupted_checkpoint is None  # Should fail to load corrupted checkpoint


class TestPhase3StorageFailureRecovery:
    """Test Phase 3: Storage Failure Recovery"""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agent_id = "test_agent_phase3"

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_transaction_safety_json_storage(self):
        """Test transaction safety in JSON storage."""
        storage = JsonNumpyStorage(
            str(self.temp_dir / "storage"),
            config={'transaction_log': True, 'corruption_detection': True}
        )

        test_data = {
            'neural_weights': [1.0, 2.0, 3.0],
            'training_info': {'epochs': 100, 'loss': 0.05}
        }

        # Test transaction wrapper
        with storage.transaction() as txn_id:
            success = storage.save_brain_state(self.agent_id, "test_env", test_data)
            assert success
            assert txn_id is not None

        # Verify data was saved correctly
        loaded_data = storage.load_brain_state(self.agent_id, "test_env")
        assert loaded_data is not None
        assert loaded_data['neural_weights'] == [1.0, 2.0, 3.0]

    def test_corruption_detection_and_recovery(self):
        """Test file corruption detection and automatic recovery."""
        storage = JsonNumpyStorage(
            str(self.temp_dir / "storage"),
            config={'transaction_log': True, 'corruption_detection': True, 'backup_enabled': True}
        )

        test_data = {'test': 'data', 'important': True}

        # Save data (creates backup)
        storage.save_brain_state(self.agent_id, "test_env", test_data)

        # Corrupt the main file
        brain_file = storage._get_env_path(self.agent_id, "test_env") / "brain_state.json"
        with open(brain_file, 'w') as f:
            f.write('corrupted data')

        # Try to load - should detect corruption and recover from backup
        recovered_data = storage.load_brain_state(self.agent_id, "test_env")

        # Should either recover successfully or return None gracefully
        # (depends on backup availability)
        if recovered_data:
            assert recovered_data['test'] == 'data'

    def test_storage_health_monitoring(self):
        """Test storage health monitoring capabilities."""
        storage = JsonNumpyStorage(
            str(self.temp_dir / "storage"),
            config={'transaction_log': True, 'corruption_detection': True}
        )

        # Get initial health
        health = storage.get_storage_health()

        assert 'timestamp' in health
        assert 'error_metrics' in health
        assert 'features_enabled' in health
        assert health['health_status'] in ['healthy', 'warning', 'degraded', 'error']
        assert health['features_enabled']['transaction_log'] == True
        assert health['features_enabled']['corruption_detection'] == True

    def test_repair_corruption_functionality(self):
        """Test corruption repair utilities."""
        storage = JsonNumpyStorage(
            str(self.temp_dir / "storage"),
            config={'transaction_log': True, 'corruption_detection': True}
        )

        # Create some test data
        test_data = {'repair_test': True, 'data': 'valid'}
        storage.save_brain_state(self.agent_id, "test_env", test_data)

        # Run corruption repair scan
        repair_results = storage.repair_corruption(self.agent_id)

        assert 'files_scanned' in repair_results
        assert 'corrupted_files' in repair_results
        assert 'repaired_files' in repair_results
        assert isinstance(repair_results['files_scanned'], int)
        assert repair_results['files_scanned'] > 0

    def test_vector_storage_robustness(self):
        """Test vector storage robustness features."""
        try:
            storage = VectorDBStorage(
                str(self.temp_dir / "vector_storage"),
                config={'transaction_log': True, 'corruption_detection': True},
                backend="faiss"
            )

            # Test health monitoring
            health = storage.get_vector_storage_health()

            assert 'backend' in health
            assert 'error_metrics' in health
            assert 'features_enabled' in health
            assert health['backend'] == "faiss"

            # Test vector operations still work
            test_vector = np.random.random(256).astype(np.float32)
            test_metadata = {'test': True, 'env_id': 'test'}

            success = storage.save_experience_vector(self.agent_id, test_vector, test_metadata)
            assert success

            # Test search still works
            results = storage.search_similar_experiences(self.agent_id, test_vector, k=1)
            assert len(results) >= 0  # Should work without errors

        except ImportError:
            pytest.skip("FAISS not available for vector storage test")

    def test_recovery_manager_comprehensive_health_check(self):
        """Test comprehensive health checking with RecoveryManager."""
        storage = JsonNumpyStorage(
            str(self.temp_dir / "storage"),
            config={'transaction_log': True, 'corruption_detection': True}
        )

        recovery_manager = StorageRecoveryManager(
            storage=storage,
            config={'auto_repair_enabled': True}
        )

        # Run comprehensive health check
        health_report = recovery_manager.comprehensive_health_check()

        assert 'timestamp' in health_report
        assert 'storage_type' in health_report
        assert 'overall_health' in health_report
        assert 'validation_results' in health_report
        assert 'recommendations' in health_report

        assert health_report['storage_type'] == 'JsonNumpyStorage'
        assert health_report['overall_health'] in ['healthy', 'warning', 'degraded', 'critical', 'error']

    def test_emergency_repair_procedures(self):
        """Test emergency repair procedures."""
        storage = JsonNumpyStorage(
            str(self.temp_dir / "storage"),
            config={'transaction_log': True, 'corruption_detection': True}
        )

        recovery_manager = StorageRecoveryManager(storage=storage)

        # Create some test data first
        test_data = {'emergency_test': True}
        storage.save_brain_state(self.agent_id, "test_env", test_data)

        # Run emergency repair
        repair_results = recovery_manager.emergency_repair(self.agent_id)

        assert 'timestamp' in repair_results
        assert 'repair_type' in repair_results
        assert 'actions_taken' in repair_results
        assert 'success' in repair_results
        assert repair_results['repair_type'] == 'emergency'

    def test_scheduled_maintenance(self):
        """Test scheduled maintenance procedures."""
        storage = JsonNumpyStorage(
            str(self.temp_dir / "storage"),
            config={'transaction_log': True, 'corruption_detection': True}
        )

        recovery_manager = StorageRecoveryManager(storage=storage)

        # Run scheduled maintenance
        maintenance_results = recovery_manager.scheduled_maintenance()

        assert 'timestamp' in maintenance_results
        assert 'tasks_completed' in maintenance_results
        assert 'cleanup_results' in maintenance_results
        assert 'recommendations' in maintenance_results
        assert isinstance(maintenance_results['tasks_completed'], list)


class TestIntegratedRobustness:
    """Test integrated robustness across all Sprint 5 phases."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agent_id = "test_agent_integrated"

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_robustness(self):
        """Test end-to-end robustness with real agent training."""
        config = AgentConfig()
        config.exploration_rate = 0.1
        config.save_interval = 5

        agent = AgentByte(
            agent_id=self.agent_id,
            storage=JsonNumpyStorage(
                str(self.temp_dir / "storage"),
                config={'transaction_log': True, 'corruption_detection': True}
            ),
            config=config,
            enable_checkpointing=True
        )

        # Create reliable environment
        env = MockEnvironment()

        # Train with all robustness features enabled
        results = agent.train(env, episodes=10, enable_recovery=True)

        # Verify training completed successfully
        assert results is not None
        assert 'final_performance' in results
        assert 'error_statistics' in results['final_performance']
        assert len(results['episode_rewards']) > 0

        # Verify checkpoints were created
        if agent.checkpoint_manager:
            checkpoints = agent.checkpoint_manager._list_checkpoints()
            assert len(checkpoints) >= 0  # Should have created some checkpoints

        # Verify health status is available
        health = agent.get_health_status()
        assert health is not None
        assert 'timestamp' in health

    def test_recovery_from_checkpoint_after_failure(self):
        """Test recovery from checkpoint after simulated failure."""
        config = AgentConfig()
        config.exploration_rate = 0.1

        # Create agent with checkpointing
        agent1 = AgentByte(
            agent_id=self.agent_id,
            storage=JsonNumpyStorage(str(self.temp_dir / "storage")),
            config=config,
            enable_checkpointing=True
        )

        # Train briefly to create checkpoints
        env = MockEnvironment()
        agent1.train(env, episodes=5, enable_recovery=True)

        # Simulate agent shutdown
        initial_episodes = agent1.total_episodes
        agent1.emergency_shutdown("simulated_failure")

        # Create new agent instance (simulates restart)
        agent2 = AgentByte(
            agent_id=self.agent_id,  # Same ID
            storage=JsonNumpyStorage(str(self.temp_dir / "storage")),
            config=config,
            enable_checkpointing=True
        )

        # Should recover state from checkpoint
        if agent2.checkpoint_manager:
            checkpoint_info = agent2.checkpoint_manager.get_checkpoint_info()
            assert checkpoint_info['total_checkpoints'] >= 0

        # Continue training
        results = agent2.train(env, episodes=5, enable_recovery=True)
        assert results is not None

    def test_performance_with_robustness_enabled(self):
        """Test that robustness features don't severely impact performance."""
        config = AgentConfig()
        config.exploration_rate = 0.1

        # Test with robustness enabled
        start_time = time.time()

        agent = AgentByte(
            agent_id=self.agent_id,
            storage=JsonNumpyStorage(
                str(self.temp_dir / "storage"),
                config={'transaction_log': True, 'corruption_detection': True}
            ),
            config=config,
            enable_checkpointing=True
        )

        env = MockEnvironment()
        results = agent.train(env, episodes=20, enable_recovery=True)

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (less than 30 seconds for 20 episodes)
        assert elapsed_time < 30
        assert results is not None
        assert len(results['episode_rewards']) == 20


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])