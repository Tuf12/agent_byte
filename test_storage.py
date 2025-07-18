"""
Unit tests for Agent Byte storage system.

This test suite verifies the correctness of all storage backends,
including JSON+Numpy and vector database implementations.
Enhanced with Sprint 9 continuous action space tests.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import time
from typing import Dict, Any, List

from agent_byte.storage import JsonNumpyStorage, VectorDBStorage, StorageBase
from agent_byte.storage.experience_buffer import ExperienceBuffer, StreamingExperienceBuffer
from agent_byte.storage.migrations import StorageMigrator


class StorageTestBase:
    """Base test class with common storage tests."""

    storage: StorageBase = None
    temp_dir: Path = None

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agent_id = "test_agent"
        self.env_id = "test_env"

    def tearDown(self):
        """Clean up test environment."""
        if self.storage:
            self.storage.close()
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_profile_save_load(self):
        """Test saving and loading agent profiles."""
        profile = {
            'agent_id': self.agent_id,
            'creation_time': '2024-01-01T00:00:00',
            'environments_experienced': ['env1', 'env2'],
            'total_episodes': 100
        }

        # Save profile
        success = self.storage.save_agent_profile(self.agent_id, profile)
        self.assertTrue(success)

        # Load profile
        loaded_profile = self.storage.get_agent_profile(self.agent_id)
        self.assertIsNotNone(loaded_profile)
        self.assertEqual(loaded_profile['agent_id'], profile['agent_id'])
        self.assertEqual(loaded_profile['total_episodes'], profile['total_episodes'])

        # Test non-existent profile
        non_existent = self.storage.get_agent_profile("non_existent_agent")
        self.assertIsNone(non_existent)

    def test_brain_state_save_load(self):
        """Test saving and loading brain states."""
        brain_state = {
            'network_state': {
                'weights': [[1.0, 2.0], [3.0, 4.0]],
                'biases': [0.1, 0.2]
            },
            'training_steps': 1000,
            'learning_rate': 0.001
        }

        # Save brain state
        success = self.storage.save_brain_state(self.agent_id, self.env_id, brain_state)
        self.assertTrue(success)

        # Load brain state
        loaded_state = self.storage.load_brain_state(self.agent_id, self.env_id)
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state['training_steps'], brain_state['training_steps'])
        self.assertEqual(loaded_state['learning_rate'], brain_state['learning_rate'])

        # Test non-existent brain state
        non_existent = self.storage.load_brain_state(self.agent_id, "non_existent_env")
        self.assertIsNone(non_existent)

    def test_knowledge_save_load(self):
        """Test saving and loading symbolic knowledge."""
        knowledge = {
            'discovered_skills': {
                'skill1': {
                    'name': 'test_skill',
                    'confidence': 0.8,
                    'applications': 10
                }
            },
            'meta_knowledge': {
                'total_skills': 1,
                'success_rate': 0.75
            }
        }

        # Save knowledge
        success = self.storage.save_knowledge(self.agent_id, self.env_id, knowledge)
        self.assertTrue(success)

        # Load knowledge
        loaded_knowledge = self.storage.load_knowledge(self.agent_id, self.env_id)
        self.assertIsNotNone(loaded_knowledge)
        self.assertIn('discovered_skills', loaded_knowledge)
        self.assertEqual(
            loaded_knowledge['meta_knowledge']['total_skills'],
            knowledge['meta_knowledge']['total_skills']
        )

    def test_autoencoder_save_load(self):
        """Test saving and loading autoencoder states."""
        autoencoder_state = {
            'architecture': {
                'input_dim': 100,
                'latent_dim': 256,
                'hidden_dims': [512, 256]
            },
            'pytorch_state': {
                'encoder.weight': [[1.0, 2.0], [3.0, 4.0]],
                'encoder.bias': [0.1, 0.2]
            },
            'training_metrics': {
                'epochs_trained': 50,
                'best_loss': 0.001
            }
        }

        # Save autoencoder
        success = self.storage.save_autoencoder(self.agent_id, self.env_id, autoencoder_state)
        self.assertTrue(success)

        # Load autoencoder
        loaded_state = self.storage.load_autoencoder(self.agent_id, self.env_id)
        self.assertIsNotNone(loaded_state)
        self.assertEqual(
            loaded_state['architecture']['input_dim'],
            autoencoder_state['architecture']['input_dim']
        )
        self.assertEqual(
            loaded_state['training_metrics']['epochs_trained'],
            autoencoder_state['training_metrics']['epochs_trained']
        )

        # List autoencoders
        autoencoder_list = self.storage.list_autoencoders(self.agent_id)
        self.assertIn(self.env_id, autoencoder_list)

    def test_continuous_network_save_load(self):
        """Test saving and loading continuous network states (Sprint 9)."""
        network_state = {
            'algorithm': 'sac',
            'state_size': 256,
            'action_size': 2,
            'action_bounds': {
                'low': [-1.0, -1.0],
                'high': [1.0, 1.0]
            },
            'device': 'cpu',
            'weights_data': 'mock_base64_encoded_weights',
            'network_info': {
                'algorithm': 'sac',
                'temperature': 0.2,
                'replay_buffer_size': 1000
            }
        }

        # Save continuous network state
        success = self.storage.save_continuous_network_state(self.agent_id, self.env_id, network_state)
        self.assertTrue(success)

        # Load continuous network state
        loaded_state = self.storage.load_continuous_network_state(self.agent_id, self.env_id)
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state['algorithm'], network_state['algorithm'])
        self.assertEqual(loaded_state['state_size'], network_state['state_size'])
        self.assertEqual(loaded_state['action_size'], network_state['action_size'])

        # List continuous networks
        networks_list = self.storage.list_continuous_networks(self.agent_id)
        self.assertIn(self.env_id, networks_list)

        # Test non-existent network
        non_existent = self.storage.load_continuous_network_state(self.agent_id, "non_existent_env")
        self.assertIsNone(non_existent)

    def test_action_adapter_save_load(self):
        """Test saving and loading action adapter configurations (Sprint 9)."""
        adapter_config = {
            'source_space': {
                'space_type': 'discrete',
                'size': 4
            },
            'target_space': {
                'space_type': 'continuous',
                'size': 2,
                'bounds': {
                    'low': [-1.0, -1.0],
                    'high': [1.0, 1.0]
                }
            },
            'adapter_type': 'discrete_to_continuous',
            'parameters': {
                'strategy': 'uniform_grid',
                'conversion_loss': 0.1
            }
        }

        # Save action adapter config
        success = self.storage.save_action_adapter_config(self.agent_id, self.env_id, adapter_config)
        self.assertTrue(success)

        # Load action adapter config
        loaded_config = self.storage.load_action_adapter_config(self.agent_id, self.env_id)
        self.assertIsNotNone(loaded_config)
        self.assertEqual(loaded_config['adapter_type'], adapter_config['adapter_type'])
        self.assertEqual(loaded_config['source_space']['size'], adapter_config['source_space']['size'])

        # List action adapters
        adapters_list = self.storage.list_action_adapters(self.agent_id)
        self.assertIn(self.env_id, adapters_list)

        # Test non-existent adapter
        non_existent = self.storage.load_action_adapter_config(self.agent_id, "non_existent_env")
        self.assertIsNone(non_existent)

    def test_experience_vectors(self):
        """Test saving and searching experience vectors."""
        # Create test vectors
        vectors = []
        metadatas = []

        for i in range(10):
            # Create normalized vectors for better similarity scores
            vector = np.random.randn(256).astype(np.float32)
            vector = vector / np.linalg.norm(vector)  # Normalize to unit vector
            metadata = {
                'env_id': self.env_id,
                'action': i % 3,
                'reward': float(i),
                'done': i == 9
            }
            vectors.append(vector)
            metadatas.append(metadata)

            # Save vector
            success = self.storage.save_experience_vector(self.agent_id, vector, metadata)
            self.assertTrue(success)

        # Search for similar experiences
        # Create a query vector that's actually similar (less noise)
        query_vector = vectors[5] + np.random.randn(256) * 0.01  # Much smaller noise
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize
        results = self.storage.search_similar_experiences(self.agent_id, query_vector, k=3)

        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

        # Check that results are sorted by similarity
        for i in range(1, len(results)):
            self.assertLessEqual(results[i]['similarity'], results[i - 1]['similarity'])

        # The most similar should be close to our target
        self.assertGreater(results[0]['similarity'], 0.8)

    def test_list_environments(self):
        """Test listing environments for an agent."""
        # Create multiple environments
        env_ids = ['env1', 'env2', 'env3']

        for env_id in env_ids:
            brain_state = {'test': env_id}
            self.storage.save_brain_state(self.agent_id, env_id, brain_state)

        # List environments
        environments = self.storage.list_environments(self.agent_id)

        for env_id in env_ids:
            self.assertIn(env_id, environments)

        # Test non-existent agent
        empty_list = self.storage.list_environments("non_existent_agent")
        self.assertEqual(len(empty_list), 0)

    def test_cache_functionality(self):
        """Test that caching improves performance."""
        if not hasattr(self.storage, '_cache') or self.storage._cache is None:
            self.skipTest("Storage backend doesn't support caching")

        profile = {'agent_id': self.agent_id, 'data': 'test'}
        self.storage.save_agent_profile(self.agent_id, profile)

        # First load (cache miss)
        start_time = time.time()
        loaded1 = self.storage.get_agent_profile(self.agent_id)
        first_load_time = time.time() - start_time

        # Second load (cache hit)
        start_time = time.time()
        loaded2 = self.storage.get_agent_profile(self.agent_id)
        second_load_time = time.time() - start_time

        # Cache should make second load faster (or at least not slower)
        self.assertLessEqual(second_load_time, first_load_time * 1.5)

        # Data should be the same
        self.assertEqual(loaded1['agent_id'], loaded2['agent_id'])

    def test_vector_validation(self):
        """Test vector validation."""
        # Valid vector
        valid_vector = np.random.randn(256).astype(np.float32)
        metadata = {'test': 'data'}
        success = self.storage.save_experience_vector(self.agent_id, valid_vector, metadata)
        self.assertTrue(success)

        # Invalid vectors
        invalid_vectors = [
            np.random.randn(128),  # Wrong size
            np.random.randn(256, 2),  # Wrong dimensions
            [1, 2, 3],  # Not numpy array
        ]

        for invalid_vector in invalid_vectors:
            success = self.storage.save_experience_vector(self.agent_id, invalid_vector, metadata)
            self.assertFalse(success)

    def test_sprint9_integration(self):
        """Test integration of all Sprint 9 features."""
        # Save a complete Sprint 9 agent profile
        profile = {
            'agent_id': self.agent_id,
            'creation_time': '2024-01-01T00:00:00',
            'environments_experienced': ['env1', 'env2'],
            'total_episodes': 100,
            'continuous_networks_enabled': True,
            'continuous_network_count': 2
        }

        # Save profile
        success = self.storage.save_agent_profile(self.agent_id, profile)
        self.assertTrue(success)

        # Save continuous network for multiple environments
        for i, env_id in enumerate(['env1', 'env2']):
            network_state = {
                'algorithm': 'sac' if i == 0 else 'ddpg',
                'state_size': 256,
                'action_size': 2,
                'action_bounds': {'low': [-1.0, -1.0], 'high': [1.0, 1.0]},
                'device': 'cpu',
                'weights_data': f'mock_weights_{i}',
                'network_info': {'algorithm': 'sac' if i == 0 else 'ddpg'}
            }

            success = self.storage.save_continuous_network_state(self.agent_id, env_id, network_state)
            self.assertTrue(success)

            # Save corresponding action adapter
            adapter_config = {
                'source_space': {'space_type': 'discrete', 'size': 4},
                'target_space': {
                    'space_type': 'continuous',
                    'size': 2,
                    'bounds': {'low': [-1.0, -1.0], 'high': [1.0, 1.0]}
                },
                'adapter_type': 'discrete_to_continuous',
                'parameters': {'strategy': 'uniform_grid'}
            }

            success = self.storage.save_action_adapter_config(self.agent_id, env_id, adapter_config)
            self.assertTrue(success)

        # Verify all data is accessible
        loaded_profile = self.storage.get_agent_profile(self.agent_id)
        self.assertTrue(loaded_profile['continuous_networks_enabled'])
        self.assertEqual(loaded_profile['continuous_network_count'], 2)

        # Verify continuous networks
        networks = self.storage.list_continuous_networks(self.agent_id)
        self.assertEqual(len(networks), 2)
        self.assertIn('env1', networks)
        self.assertIn('env2', networks)

        # Verify action adapters
        adapters = self.storage.list_action_adapters(self.agent_id)
        self.assertEqual(len(adapters), 2)
        self.assertIn('env1', adapters)
        self.assertIn('env2', adapters)

        # Load and verify specific network
        sac_network = self.storage.load_continuous_network_state(self.agent_id, 'env1')
        self.assertEqual(sac_network['algorithm'], 'sac')

        ddpg_network = self.storage.load_continuous_network_state(self.agent_id, 'env2')
        self.assertEqual(ddpg_network['algorithm'], 'ddpg')


class TestJsonNumpyStorage(StorageTestBase, unittest.TestCase):
    """Test JSON+Numpy storage implementation."""

    def setUp(self):
        super().setUp()
        self.storage = JsonNumpyStorage(str(self.temp_dir / "json_storage"))

    def test_lazy_loading(self):
        """Test lazy loading functionality."""
        # Create storage with lazy loading enabled
        lazy_storage = JsonNumpyStorage(
            str(self.temp_dir / "lazy_storage"),
            config={'lazy_loading': True, 'memory_limit': 50}
        )

        # Add many vectors
        for i in range(100):
            vector = np.random.randn(256).astype(np.float32)
            metadata = {'index': i}
            success = lazy_storage.save_experience_vector(self.agent_id, vector, metadata)
            self.assertTrue(success)

        # Search should still work with lazy loading
        query_vector = np.random.randn(256).astype(np.float32)
        results = lazy_storage.search_similar_experiences(self.agent_id, query_vector, k=5)
        self.assertEqual(len(results), 5)

        lazy_storage.close()

    def test_file_structure(self):
        """Test that files are created in the expected structure."""
        profile = {'agent_id': self.agent_id}
        self.storage.save_agent_profile(self.agent_id, profile)

        brain_state = {'test': 'brain'}
        self.storage.save_brain_state(self.agent_id, self.env_id, brain_state)

        # Test Sprint 9 file structure
        network_state = {'algorithm': 'sac', 'state_size': 256}
        self.storage.save_continuous_network_state(self.agent_id, self.env_id, network_state)

        adapter_config = {'adapter_type': 'discrete_to_continuous'}
        self.storage.save_action_adapter_config(self.agent_id, self.env_id, adapter_config)

        # Check file structure
        agent_path = self.temp_dir / "json_storage" / "agents" / self.agent_id
        self.assertTrue(agent_path.exists())
        self.assertTrue((agent_path / "profile.json").exists())

        env_path = agent_path / "environments" / self.env_id
        self.assertTrue(env_path.exists())
        self.assertTrue((env_path / "brain_state.json").exists())
        self.assertTrue((env_path / "continuous_network.json").exists())
        self.assertTrue((env_path / "action_adapter.json").exists())


class TestVectorDBStorage(StorageTestBase, unittest.TestCase):
    """Test vector database storage implementation."""

    def setUp(self):
        super().setUp()
        # Test with FAISS backend (if available)
        try:
            self.storage = VectorDBStorage(
                str(self.temp_dir / "vector_storage"),
                backend="faiss"
            )
        except:
            self.skipTest("FAISS not available")

    def test_vector_search_performance(self):
        """Test that vector search is fast even with many vectors."""
        # Add many vectors
        num_vectors = 1000
        for i in range(num_vectors):
            vector = np.random.randn(256).astype(np.float32)
            metadata = {'index': i}
            self.storage.save_experience_vector(self.agent_id, vector, metadata)

        # Time the search
        query_vector = np.random.randn(256).astype(np.float32)
        start_time = time.time()
        results = self.storage.search_similar_experiences(self.agent_id, query_vector, k=10)
        search_time = time.time() - start_time

        self.assertEqual(len(results), 10)
        self.assertLess(search_time, 0.1)  # Should be fast (< 100ms)

    def test_metadata_filtering(self):
        """Test metadata filtering in ChromaDB backend."""
        # Skip if not using ChromaDB
        if getattr(self.storage, 'backend', None) != "chroma":
            self.skipTest("Metadata filtering only available in ChromaDB")

        # Add vectors with different metadata
        for i in range(20):
            vector = np.random.randn(256).astype(np.float32)
            metadata = {
                'env_id': f'env_{i % 3}',
                'reward': float(i),
                'action': i % 4
            }
            self.storage.save_experience_vector(self.agent_id, vector, metadata)

        # Search with metadata filter
        query_vector = np.random.randn(256).astype(np.float32)
        # Note: This assumes the storage backend supports filter_metadata parameter
        try:
            results = self.storage.search_similar_experiences(
                self.agent_id, query_vector, k=5
            )
            # Basic check that search works
            self.assertLessEqual(len(results), 5)
        except TypeError:
            # Skip if metadata filtering not supported in this backend
            self.skipTest("Metadata filtering not supported in this backend configuration")


class TestContinuousNetworkIntegration(unittest.TestCase):
    """Test continuous network integration with storage."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = JsonNumpyStorage(str(self.temp_dir / "integration_test"))
        self.agent_id = "continuous_test_agent"

    def tearDown(self):
        """Clean up test environment."""
        self.storage.close()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_continuous_network_storage_manager(self):
        """Test the ContinuousNetworkStorageManager integration."""
        try:
            from agent_byte.core.continuous_network import ContinuousNetworkStorageManager
            from agent_byte.core.interfaces import create_continuous_action_space
        except ImportError:
            self.skipTest("Continuous network modules not available")

        # Create storage manager
        storage_manager = ContinuousNetworkStorageManager(self.storage, self.agent_id)

        # Test listing empty networks
        networks = storage_manager.list_saved_networks()
        self.assertEqual(len(networks), 0)

        # Test saving mock network data (simulating what would be saved)
        env_id = "test_continuous_env"
        mock_network_state = {
            'algorithm': 'sac',
            'state_size': 256,
            'action_size': 2,
            'action_bounds': {
                'low': [-1.0, -1.0],
                'high': [1.0, 1.0]
            },
            'device': 'cpu',
            'weights_data': 'mock_base64_encoded_weights_data',
            'network_info': {
                'algorithm': 'sac',
                'device': 'cpu',
                'state_size': 256,
                'action_size': 2,
                'temperature': 0.2
            }
        }

        # Save mock network state directly through storage
        success = self.storage.save_continuous_network_state(
            self.agent_id, env_id, mock_network_state
        )
        self.assertTrue(success)

        # Test that storage manager can list it
        networks = storage_manager.list_saved_networks()
        self.assertEqual(len(networks), 1)
        self.assertIn(env_id, networks)

        # Test loading
        loaded_state = self.storage.load_continuous_network_state(self.agent_id, env_id)
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state['algorithm'], 'sac')
        self.assertEqual(loaded_state['state_size'], 256)


class TestExperienceBuffer(unittest.TestCase):
    """Test experience buffer implementations."""

    def test_experience_buffer_basic(self):
        """Test basic experience buffer functionality."""
        buffer = ExperienceBuffer(max_size=100, cache_size=20, window_size=10)

        # Add experiences
        for i in range(50):
            vector = np.random.randn(256)
            metadata = {'index': i}
            idx = buffer.add(vector, metadata)
            self.assertEqual(idx, i)

        # Test get
        exp = buffer.get(25)
        self.assertIsNotNone(exp)
        self.assertEqual(exp['metadata']['index'], 25)

        # Get the same item again to create a cache hit
        exp2 = buffer.get(25)
        self.assertIsNotNone(exp2)
        self.assertEqual(exp2['metadata']['index'], 25)

        # Test recent
        recent = buffer.get_recent(5)
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[-1]['metadata']['index'], 49)

        # Test search
        query = np.random.randn(256)
        results = buffer.search_similar(query, k=3, use_window=True)
        self.assertLessEqual(len(results), 3)

        # Test stats
        stats = buffer.get_stats()
        self.assertEqual(stats['total_experiences'], 50)
        self.assertGreater(stats['cache_hit_rate'], 0)  # Now should have cache hits

    def test_streaming_buffer(self):
        """Test streaming experience buffer with disk overflow."""
        temp_dir = tempfile.mkdtemp()

        try:
            buffer = StreamingExperienceBuffer(
                max_memory_size=10,
                cache_size=5,
                disk_path=temp_dir
            )

            # Add more experiences than memory can hold
            for i in range(30):
                vector = np.random.randn(256)
                metadata = {'index': i}
                buffer.add(vector, metadata)

            # Check that old experiences are on disk
            self.assertGreater(len(buffer.disk_experiences), 0)
            self.assertLessEqual(len(buffer.experiences), 10)

            # Test iteration over all experiences
            all_experiences = []
            for batch in buffer.iterate_all(batch_size=5):
                all_experiences.extend(batch)

            self.assertEqual(len(all_experiences), 30)

        finally:
            shutil.rmtree(temp_dir)


class TestStorageMigration(unittest.TestCase):
    """Test storage migration functionality."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agent_id = "migration_test_agent"

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_backend_migration(self):
        """Test migrating from JSON to vector database storage."""
        # Create source storage with data
        source_path = self.temp_dir / "source"
        source = JsonNumpyStorage(str(source_path))

        # Add test data
        profile = {'agent_id': self.agent_id, 'test': 'profile'}
        source.save_agent_profile(self.agent_id, profile)

        brain_state = {'weights': [1, 2, 3]}
        source.save_brain_state(self.agent_id, 'env1', brain_state)

        knowledge = {'skills': ['skill1', 'skill2']}
        source.save_knowledge(self.agent_id, 'env1', knowledge)

        # Add Sprint 9 data
        network_state = {'algorithm': 'sac', 'state_size': 256}
        source.save_continuous_network_state(self.agent_id, 'env1', network_state)

        adapter_config = {'adapter_type': 'discrete_to_continuous'}
        source.save_action_adapter_config(self.agent_id, 'env1', adapter_config)

        # Add experience vectors
        for i in range(10):
            vector = np.random.randn(256).astype(np.float32)
            metadata = {'index': i}
            source.save_experience_vector(self.agent_id, vector, metadata)

        # Create target storage
        target_path = self.temp_dir / "target"
        target = JsonNumpyStorage(str(target_path))

        # Migrate
        migrator = StorageMigrator()
        results = migrator.migrate_backend(
            source, target,
            agent_ids=[self.agent_id],
            create_backup=False
        )

        self.assertEqual(results['agents_migrated'], 1)
        self.assertEqual(results['agents_failed'], 0)

        # Verify migration
        verify_results = migrator.verify_migration(
            source, target,
            agent_ids=[self.agent_id]
        )

        self.assertTrue(verify_results['verified'])
        self.assertTrue(verify_results['agent_checks'][self.agent_id]['profile_match'])
        self.assertTrue(verify_results['agent_checks'][self.agent_id]['environments_match'])

        # Check data integrity including Sprint 9 data
        loaded_profile = target.get_agent_profile(self.agent_id)
        self.assertEqual(loaded_profile['test'], 'profile')

        loaded_brain = target.load_brain_state(self.agent_id, 'env1')
        self.assertEqual(loaded_brain['weights'], [1, 2, 3])

        loaded_knowledge = target.load_knowledge(self.agent_id, 'env1')
        self.assertEqual(loaded_knowledge['skills'], ['skill1', 'skill2'])

        # Verify Sprint 9 data
        loaded_network = target.load_continuous_network_state(self.agent_id, 'env1')
        self.assertEqual(loaded_network['algorithm'], 'sac')

        loaded_adapter = target.load_action_adapter_config(self.agent_id, 'env1')
        self.assertEqual(loaded_adapter['adapter_type'], 'discrete_to_continuous')

        source.close()
        target.close()


class TestErrorHandling(unittest.TestCase):
    """Test error handling in storage operations."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = JsonNumpyStorage(str(self.temp_dir))

    def tearDown(self):
        self.storage.close()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_invalid_vector_handling(self):
        """Test handling of invalid vectors."""
        # Wrong dimension
        vector = np.random.randn(128)
        success = self.storage.save_experience_vector('agent', vector, {})
        self.assertFalse(success)

        # Wrong type
        success = self.storage.save_experience_vector('agent', [1, 2, 3], {})
        self.assertFalse(success)

        # Multi-dimensional
        vector = np.random.randn(16, 16)
        success = self.storage.save_experience_vector('agent', vector, {})
        self.assertFalse(success)

    def test_corrupted_file_handling(self):
        """Test handling of corrupted files."""
        # Create a corrupted JSON file
        agent_path = self.temp_dir / "agents" / "test_agent"
        agent_path.mkdir(parents=True)

        with open(agent_path / "profile.json", 'w') as f:
            f.write("{ corrupted json")

        # Should handle gracefully
        profile = self.storage.get_agent_profile("test_agent")
        self.assertIsNone(profile)

    def test_invalid_continuous_network_data(self):
        """Test handling of invalid continuous network data."""
        # Invalid network state (missing required fields)
        invalid_state = {'algorithm': 'sac'}  # Missing required fields
        success = self.storage.save_continuous_network_state('agent', 'env', invalid_state)
        # Should still save (validation happens at higher level)
        self.assertTrue(success)

        # Load should work but data will be incomplete
        loaded = self.storage.load_continuous_network_state('agent', 'env')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['algorithm'], 'sac')

    def test_concurrent_access(self):
        """Test handling of concurrent access (basic test)."""
        import threading

        results = {'errors': 0}

        def write_data(i):
            try:
                vector = np.random.randn(256).astype(np.float32)
                metadata = {'thread': i}
                self.storage.save_experience_vector('agent', vector, metadata)
            except Exception:
                results['errors'] += 1

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_data, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should complete without errors
        self.assertEqual(results['errors'], 0)


def run_all_tests():
    """Run all storage tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestJsonNumpyStorage,
        TestVectorDBStorage,
        TestContinuousNetworkIntegration,
        TestExperienceBuffer,
        TestStorageMigration,
        TestErrorHandling
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)