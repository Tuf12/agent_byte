"""
Test script for Sprint 2: Autoencoder State Normalization

This script tests the autoencoder implementation including:
- VAE training and compression
- State normalization with autoencoders
- Storage integration
- Transfer learning compatibility
"""

import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Import our modules (adjust paths as needed)
from agent_byte.analysis.autoencoder import VariationalAutoencoder, AutoencoderTrainer
from agent_byte.analysis.state_normalizer import StateNormalizer
from agent_byte.storage.json_numpy_storage import JsonNumpyStorage


class TestEnvironment:
    """Mock environment for testing."""

    def __init__(self, env_id: str, state_size: int):
        self.env_id = env_id
        self.state_size = state_size

    def get_id(self) -> str:
        return self.env_id

    def get_state_size(self) -> int:
        return self.state_size

    def generate_states(self, n_samples: int) -> np.ndarray:
        """Generate synthetic states for testing."""
        # Create states with some structure
        states = []
        for i in range(n_samples):
            # Add some patterns to make compression meaningful
            t = i / n_samples * 2 * np.pi
            state = np.zeros(self.state_size)

            # Add different patterns to different parts
            if self.state_size >= 10:
                state[0] = np.sin(t)  # Sine wave
                state[1] = np.cos(t)  # Cosine wave
                state[2:5] = np.random.randn(3) * 0.1  # Small noise
                state[5] = np.sin(2 * t)  # Higher frequency
                if self.state_size > 10:
                    state[10:] = np.random.randn(self.state_size - 10) * 0.5
            else:
                state = np.random.randn(self.state_size)

            states.append(state)

        return np.array(states)


def test_vae_basic():
    """Test basic VAE functionality."""
    print("\n=== Testing Basic VAE Functionality ===")

    # Create VAE
    input_dim = 20
    latent_dim = 10
    vae = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

    # Test forward pass - ensure tensor is on same device as model
    test_input = torch.randn(5, input_dim).to(vae.device)
    reconstruction, original, mu, log_var = vae(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent log_var shape: {log_var.shape}")

    # Test loss calculation
    loss, components = vae.loss_function(reconstruction, original, mu, log_var)
    print(f"Loss components: {components}")

    # Test compress/decompress
    state = np.random.randn(input_dim)
    compressed = vae.compress(state)
    decompressed = vae.decompress(compressed)

    print(f"Original state shape: {state.shape}")
    print(f"Compressed shape: {compressed.shape}")
    print(f"Decompressed shape: {decompressed.shape}")
    print(f"Reconstruction error: {np.mean((state - decompressed) ** 2):.6f}")

    return True


def test_autoencoder_training():
    """Test autoencoder training process."""
    print("\n=== Testing Autoencoder Training ===")

    # Create trainer
    trainer = AutoencoderTrainer(latent_dim=256)

    # Generate training data
    env = TestEnvironment("test_env", state_size=50)
    states = env.generate_states(2000)

    print(f"Training data shape: {states.shape}")

    # Train autoencoder
    autoencoder = trainer.train_autoencoder(env.get_id(), states, validation_split=0.2)

    # Check training metrics
    metrics = autoencoder.training_metrics
    print(f"Training metrics:")
    print(f"  Epochs trained: {metrics['epochs_trained']}")
    print(f"  Best loss: {metrics['best_loss']:.6f}")
    print(f"  Final reconstruction loss: {metrics['reconstruction_loss']:.6f}")
    print(f"  Final KL loss: {metrics['kl_loss']:.6f}")

    # Test reconstruction quality
    test_states = states[:10]
    reconstruction_errors = []
    for state in test_states:
        error = autoencoder.get_reconstruction_error(state)
        reconstruction_errors.append(error)

    avg_error = np.mean(reconstruction_errors)
    print(f"Average reconstruction error on test states: {avg_error:.6f}")

    # Analyze latent space
    analysis = trainer.analyze_latent_space(env.get_id(), states[:100])
    print(f"Latent space analysis:")
    print(f"  Active dimensions: {analysis['active_dimensions']}/{analysis['latent_dim']}")
    print(f"  Total variance: {analysis['total_variance']:.6f}")
    print(f"  Sparsity: {analysis['sparsity']:.4f}")

    return autoencoder, avg_error < 0.1  # Success if low reconstruction error


def test_state_normalizer_integration():
    """Test StateNormalizer with autoencoder support."""
    print("\n=== Testing State Normalizer Integration ===")

    # Create normalizer with autoencoder enabled
    normalizer = StateNormalizer(target_dim=256, use_autoencoder=True)

    # Test with different sized environments
    test_cases = [
        ("small_env", 10),
        ("medium_env", 50),
        ("large_env", 500)
    ]

    results = {}

    for env_id, state_size in test_cases:
        print(f"\nTesting {env_id} with state size {state_size}")

        # Generate states
        env = TestEnvironment(env_id, state_size)
        states = env.generate_states(1500)  # Need 1000+ for autoencoder training

        # Track normalization methods
        methods_used = set()

        # Normalize states and track method evolution
        for i, state in enumerate(states):
            normalized = normalizer.normalize(state, env_id)
            method = normalizer.normalization_methods.get(env_id, 'unknown')
            methods_used.add(method)

            # Check dimensions
            assert len(normalized) == 256, f"Wrong normalized dimension: {len(normalized)}"

            # Check range
            assert np.all(np.abs(normalized) <= 10), "Values outside expected range"

        # Get normalization info
        info = normalizer.get_normalization_info(env_id)
        print(f"  Methods used: {methods_used}")
        print(f"  Final method: {info['method']}")
        print(f"  Has autoencoder: {info['has_autoencoder']}")

        if info['has_autoencoder']:
            print(f"  Autoencoder epochs: {info['autoencoder_metrics']['epochs_trained']}")
            print(f"  Best loss: {info['autoencoder_metrics']['best_loss']:.6f}")

            if 'latent_analysis' in info:
                print(f"  Active latent dims: {info['latent_analysis']['active_dimensions']}/256")

        results[env_id] = info

    return normalizer, results


def test_storage_integration():
    """Test storage of autoencoders."""
    print("\n=== Testing Storage Integration ===")

    # Create temporary storage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JsonNumpyStorage(temp_dir)
        agent_id = "test_agent"

        # Create and train an autoencoder
        trainer = AutoencoderTrainer(latent_dim=256)
        env_id = "storage_test_env"
        states = np.random.randn(1500, 30)

        autoencoder = trainer.train_autoencoder(env_id, states)

        # Save through storage
        state_dict = autoencoder.get_state_dict_serializable()
        success = storage.save_autoencoder(agent_id, env_id, state_dict)
        print(f"Save autoencoder: {'Success' if success else 'Failed'}")

        # List autoencoders
        autoencoder_list = storage.list_autoencoders(agent_id)
        print(f"Listed autoencoders: {autoencoder_list}")

        # Load back
        loaded_state = storage.load_autoencoder(agent_id, env_id)
        print(f"Load autoencoder: {'Success' if loaded_state else 'Failed'}")

        if loaded_state:
            # Create new autoencoder and load state
            arch = loaded_state['architecture']
            new_autoencoder = VariationalAutoencoder(
                input_dim=arch['input_dim'],
                latent_dim=arch['latent_dim'],
                hidden_dims=arch['hidden_dims']
            )
            new_autoencoder.load_state_dict_serializable(loaded_state)

            # Test that it works
            test_state = states[0]
            original_compressed = autoencoder.compress(test_state)
            loaded_compressed = new_autoencoder.compress(test_state)

            diff = np.mean(np.abs(original_compressed - loaded_compressed))
            print(f"Compression difference after load: {diff:.8f}")

            return diff < 1e-6  # Should be identical

    return False


def test_transfer_compatibility():
    """Test autoencoder compatibility checking for transfer learning."""
    print("\n=== Testing Transfer Learning Compatibility ===")

    normalizer = StateNormalizer(target_dim=256, use_autoencoder=True)

    # Train autoencoders for different environments
    envs = [
        ("env_A_v1", 30),
        ("env_A_v2", 30),  # Same size as v1
        ("env_B", 50),     # Different size
    ]

    # Train autoencoders
    for env_id, state_size in envs:
        states = TestEnvironment(env_id, state_size).generate_states(1500)
        for state in states:
            normalizer.normalize(state, env_id)

    # Check compatibility
    print("\nChecking autoencoder compatibility:")

    # Create a new environment with same size as env_A
    new_env = TestEnvironment("env_A_v3", 30)

    compatible_count = 0
    if normalizer.autoencoder_trainer:
        for existing_env_id in normalizer.autoencoder_trainer.autoencoders:
            ae = normalizer.autoencoder_trainer.autoencoders[existing_env_id]
            if ae.input_dim == new_env.get_state_size():
                print(f"  {existing_env_id}: Compatible (input_dim={ae.input_dim})")
                compatible_count += 1
            else:
                print(f"  {existing_env_id}: Incompatible (input_dim={ae.input_dim})")

    print(f"\nFound {compatible_count} compatible autoencoders for transfer")

    return compatible_count > 0


def visualize_compression(autoencoder: VariationalAutoencoder, states: np.ndarray):
    """Visualize compression and reconstruction quality."""
    print("\n=== Visualizing Compression Results ===")

    # Select a few states to visualize
    n_samples = min(5, len(states))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        state = states[i]

        # Compress and decompress
        compressed = autoencoder.compress(state)
        reconstructed = autoencoder.decompress(compressed)

        # Original state
        axes[i, 0].plot(state)
        axes[i, 0].set_title(f"Original State {i}")
        axes[i, 0].set_xlabel("Dimension")
        axes[i, 0].set_ylabel("Value")

        # Compressed representation (first 50 dims)
        axes[i, 1].plot(compressed[:50])
        axes[i, 1].set_title(f"Compressed (first 50 of {len(compressed)} dims)")
        axes[i, 1].set_xlabel("Latent Dimension")
        axes[i, 1].set_ylabel("Value")

        # Reconstruction
        axes[i, 2].plot(state, label="Original", alpha=0.7)
        axes[i, 2].plot(reconstructed, label="Reconstructed", alpha=0.7)
        axes[i, 2].set_title(f"Reconstruction (MSE: {np.mean((state - reconstructed)**2):.6f})")
        axes[i, 2].set_xlabel("Dimension")
        axes[i, 2].set_ylabel("Value")
        axes[i, 2].legend()

    plt.tight_layout()
    plt.savefig("autoencoder_compression_results.png")
    print("Saved visualization to autoencoder_compression_results.png")
    plt.close()


def run_all_tests():
    """Run all Sprint 2 tests."""
    print("=" * 60)
    print("Sprint 2: Autoencoder State Normalization - Test Suite")
    print("=" * 60)

    # Test 1: Basic VAE
    try:
        test_vae_basic()
        print("✓ Basic VAE test passed")
    except Exception as e:
        print(f"✗ Basic VAE test failed: {e}")
        return False

    # Test 2: Autoencoder Training
    try:
        autoencoder, success = test_autoencoder_training()
        if success:
            print("✓ Autoencoder training test passed")

            # Visualize results
            env = TestEnvironment("viz_env", 50)
            states = env.generate_states(100)
            visualize_compression(autoencoder, states[:5])
        else:
            print("✗ Autoencoder training test failed: High reconstruction error")
            return False
    except Exception as e:
        print(f"✗ Autoencoder training test failed: {e}")
        return False

    # Test 3: State Normalizer Integration
    try:
        normalizer, results = test_state_normalizer_integration()
        print("✓ State normalizer integration test passed")
    except Exception as e:
        print(f"✗ State normalizer integration test failed: {e}")
        return False

    # Test 4: Storage Integration
    try:
        if test_storage_integration():
            print("✓ Storage integration test passed")
        else:
            print("✗ Storage integration test failed")
            return False
    except Exception as e:
        print(f"✗ Storage integration test failed: {e}")
        return False

    # Test 5: Transfer Compatibility
    try:
        if test_transfer_compatibility():
            print("✓ Transfer compatibility test passed")
        else:
            print("✗ Transfer compatibility test failed")
            return False
    except Exception as e:
        print(f"✗ Transfer compatibility test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("All Sprint 2 tests passed! ✓")
    print("=" * 60)

    return True


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()

    if success:
        print("\nSprint 2 implementation is working correctly!")
        print("The autoencoder-based state normalization system is ready for use.")
    else:
        print("\nSome tests failed. Please check the implementation.")