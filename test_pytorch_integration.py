"""
Test PyTorch integration for Agent Byte v3.0

This module tests the PyTorch implementation of the neural network
and learning system.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to a path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agent_byte.core.network import StandardizedNetwork
from agent_byte.core.neural_brain import NeuralBrain
from agent_byte.storage.json_numpy_storage import JsonNumpyStorage


class TestPyTorchNetwork:
    """Test the PyTorch StandardizedNetwork implementation."""

    def test_network_initialization(self):
        """Test network initializes correctly."""
        action_size = 4
        network = StandardizedNetwork(action_size=action_size)

        # Check architecture
        assert network.input_size == 256
        assert network.action_size == action_size
        assert len(network.core_layers) == 3

        # Check a device
        assert network.device.type in ['cpu', 'cuda']

        # Check parameter count
        param_count = network.get_num_parameters()
        assert param_count['total'] > 0
        print(f"Network parameters: {param_count}")

    def test_forward_pass(self):
        """Test forward pass through the network."""
        network = StandardizedNetwork(action_size=4)

        # Test with numpy array
        state_np = np.random.randn(256).astype(np.float32)
        q_values = network(state_np)

        assert isinstance(q_values, torch.Tensor)
        assert q_values.shape == (1, 4)

        # Test with batch
        batch_states = torch.randn(16, 256)
        batch_q_values = network(batch_states)
        assert batch_q_values.shape == (16, 4)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        network = StandardizedNetwork(action_size=3)
        network.train()

        # Forward pass
        state = torch.randn(1, 256, requires_grad=True)
        q_values = network(state)

        # Compute loss (ensure target is on the same device as q_values)
        target = torch.tensor([[1.0, 0.0, -1.0]]).to(q_values.device)
        loss = torch.nn.functional.mse_loss(q_values, target)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_pattern_tracking(self):
        """Test activation pattern tracking."""
        network = StandardizedNetwork(action_size=4)
        network.eval()  # Patterns only tracked in eval mode

        # Run multiple forward passes
        for i in range(5):
            state = np.random.randn(256)
            network(state)
            network.record_decision(state, i % 4, np.random.randn())

        # Check patterns were recorded
        pattern_summary = network.get_pattern_summary()
        assert pattern_summary['pattern_count'] >= 5
        assert pattern_summary['decision_count'] >= 5
        assert 'pattern_stability' in pattern_summary

    def test_transfer_learning(self):
        """Test transfer learning functionality."""
        source_network = StandardizedNetwork(action_size=4)
        target_network = StandardizedNetwork(action_size=6)  # Different action size

        # Modify source weights
        with torch.no_grad():
            for layer in source_network.core_layers:
                layer.weight.add_(1.0)

        # Transfer core layers
        target_network.transfer_core_layers_from(source_network)

        # Check weights were transferred
        for source_layer, target_layer in zip(source_network.core_layers,
                                            target_network.core_layers):
            assert torch.allclose(source_layer.weight, target_layer.weight)

    def test_freeze_unfreeze(self):
        """Test freezing and unfreezing layers."""
        network = StandardizedNetwork(action_size=4)

        # Freeze core layers
        network.freeze_core_layers()
        for layer in network.core_layers:
            for param in layer.parameters():
                assert not param.requires_grad

        # Adapter and output should still be trainable
        assert network.adapter_layer.weight.requires_grad
        assert network.output_layer.weight.requires_grad

        # Unfreeze
        network.unfreeze_core_layers()
        for layer in network.core_layers:
            for param in layer.parameters():
                assert param.requires_grad

    def test_serialization(self):
        """Test network state serialization."""
        network = StandardizedNetwork(action_size=5)

        # Get state
        state_dict = network.get_state_dict_serializable()

        # Check structure
        assert 'pytorch_state' in state_dict
        assert 'metadata' in state_dict
        assert 'architecture' in state_dict

        # Create new network and load state
        new_network = StandardizedNetwork(action_size=5)
        new_network.load_state_dict_serializable(state_dict)

        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
                network.named_parameters(),
                new_network.named_parameters()):
            assert torch.allclose(param1, param2), f"Mismatch in {name1}"


class TestNeuralBrain:
    """Test the PyTorch NeuralBrain implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = JsonNumpyStorage("./test_agent_data")
        self.config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'batch_size': 4,
            'experience_buffer_size': 100
        }

    def test_brain_initialization(self):
        """Test neural brain initialization."""
        brain = NeuralBrain(
            agent_id="test_agent",
            action_size=4,
            storage=self.storage,
            config=self.config
        )

        assert brain.network is not None
        assert brain.target_network is not None
        assert brain.optimizer is not None
        assert brain.scheduler is not None

    def test_action_selection(self):
        """Test action selection."""
        brain = NeuralBrain("test", 4, self.storage, self.config)

        state = np.random.randn(256)

        # Test exploration
        action, info = brain.select_action(state, exploration_rate=1.0)
        assert 0 <= action < 4
        assert info['decision_type'] == 'exploration'

        # Test exploitation
        action, info = brain.select_action(state, exploration_rate=0.0)
        assert 0 <= action < 4
        assert info['decision_type'] == 'exploitation'
        assert 'q_values' in info

    def test_experience_storage(self):
        """Test experience storage."""
        brain = NeuralBrain("test", 3, self.storage, self.config)

        # Store experiences
        for i in range(10):
            state = np.random.randn(256)
            next_state = np.random.randn(256)
            brain.store_experience(state, i % 3, 0.1 * i, next_state, False)

        assert len(brain.experience_buffer) == 10

    def test_learning_step(self):
        """Test actual learning happens."""
        brain = NeuralBrain("test", 3, self.storage, self.config)

        # Store enough experiences
        for i in range(20):
            state = np.random.randn(256)
            next_state = np.random.randn(256)
            reward = 1.0 if i % 3 == 0 else -0.1
            brain.store_experience(state, i % 3, reward, next_state, False)

        # Get initial network weights
        initial_weights = {}
        for name, param in brain.network.named_parameters():
            initial_weights[name] = param.clone()

        # Perform learning
        losses = []
        for _ in range(5):
            loss = brain.learn()
            if loss is not None:
                losses.append(loss)

        assert len(losses) > 0, "No learning occurred"

        # Check weights changed
        weights_changed = False
        for name, param in brain.network.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "Network weights did not change during learning"

    def test_target_network_update(self):
        """Test target network updates."""
        config = self.config.copy()
        config['target_update_frequency'] = 5
        brain = NeuralBrain("test", 3, self.storage, config)

        # Store experiences
        for i in range(20):
            state = np.random.randn(256)
            next_state = np.random.randn(256)
            brain.store_experience(state, i % 3, 0.1, next_state, False)

        # Get initial target weights
        initial_target = {k: v.clone() for k, v in brain.target_network.state_dict().items()}

        # Track successful learning steps
        successful_learns = 0

        # Learn until we've had enough successful updates
        for i in range(20):  # Increased iterations
            loss = brain.learn()
            if loss is not None:
                successful_learns += 1

        # Check that we had enough successful learns to trigger update
        assert successful_learns >= config['target_update_frequency'], \
            f"Only had {successful_learns} successful learns, need at least {config['target_update_frequency']}"

        # Check target network was updated
        current_target = brain.target_network.state_dict()
        weights_updated = False
        for key in initial_target:
            if not torch.allclose(initial_target[key], current_target[key]):
                weights_updated = True
                break

        assert weights_updated, f"Target network was not updated after {successful_learns} learns"

    def test_save_load_state(self):
        """Test saving and loading brain state."""
        brain = NeuralBrain("test", 4, self.storage, self.config)

        # Train a bit
        for i in range(20):
            state = np.random.randn(256)
            next_state = np.random.randn(256)
            brain.store_experience(state, i % 4, 0.1, next_state, False)

        for _ in range(5):
            brain.learn()

        # Save state
        success = brain.save_state("test_env")
        assert success

        # Create new brain and load state
        new_brain = NeuralBrain("test", 4, self.storage, self.config)
        success = new_brain.load_state("test_env")
        assert success

        # Verify state match
        assert new_brain.training_steps == brain.training_steps

        # Verify network weights match
        for (name1, param1), (name2, param2) in zip(
                brain.network.named_parameters(),
                new_brain.network.named_parameters()):
            assert torch.allclose(param1, param2), f"Mismatch in {name1}"

    def test_transfer_mode(self):
        """Test transfer learning mode."""
        brain = NeuralBrain("test", 4, self.storage, self.config)

        # Enable transfer mode
        brain.enable_transfer_mode(freeze_core=True)

        # Check core layers are frozen
        for layer in brain.network.core_layers:
            for param in layer.parameters():
                assert not param.requires_grad

        # Disable transfer mode
        brain.disable_transfer_mode()

        # Check all layers are trainable
        for param in brain.network.parameters():
            assert param.requires_grad


def test_gradient_computation():
    """Integration test for gradient computation."""
    # Create minimal setup
    storage = JsonNumpyStorage("./test_gradient")
    config = {'learning_rate': 0.01, 'batch_size': 2}
    brain = NeuralBrain("grad_test", 2, storage, config)

    # Create specific experiences
    state1 = np.zeros(256)
    state1[0] = 1.0  # Distinctive feature

    state2 = np.zeros(256)
    state2[1] = 1.0  # Different feature

    # Action 0 from state1 gives reward 1
    brain.store_experience(state1, 0, 1.0, state2, False)
    # Action 1 from state1 gives reward -1
    brain.store_experience(state1, 1, -1.0, state2, False)

    # Learn multiple times
    losses = []
    for _ in range(50):
        loss = brain.learn()
        if loss:
            losses.append(loss)

    # Check that network learned
    with torch.no_grad():
        q_values = brain.network(torch.FloatTensor(state1).unsqueeze(0).to(brain.device))
        q_values_np = q_values.cpu().numpy()[0]

    # Action 0 should have higher Q-value than action 1
    assert q_values_np[0] > q_values_np[1], \
        f"Network did not learn preference: Q(s,a0)={q_values_np[0]}, Q(s,a1)={q_values_np[1]}"

    print(f"Learning successful! Q-values: {q_values_np}")
    print(f"Average loss over training: {np.mean(losses):.4f}")


if __name__ == "__main__":
    # Run basic tests
    print("Testing PyTorch Network Implementation...")
    network_test = TestPyTorchNetwork()
    network_test.test_network_initialization()
    network_test.test_forward_pass()
    network_test.test_gradient_flow()
    print("✓ Network tests passed!")

    print("\nTesting Neural Brain Implementation...")
    brain_test = TestNeuralBrain()
    brain_test.setup_method()
    brain_test.test_brain_initialization()
    brain_test.test_action_selection()
    brain_test.test_learning_step()
    print("✓ Brain tests passed!")

    print("\nTesting Gradient Computation...")
    test_gradient_computation()
    print("✓ Gradient computation test passed!")

    print("\nAll PyTorch integration tests passed! 🎉")