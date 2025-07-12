"""
Autoencoder-based state compression for Agent Byte.

This module implements variational autoencoders (VAE) for compressing
environment states into the standardized 256-dimensional representation,
enabling nonlinear compression and better transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from pathlib import Path


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for state compression.

    This VAE learns to compress environment-specific state representations
    into a standardized 256-dimensional latent space that captures the
    essential information for decision-making.
    """

    def __init__(self, input_dim: int, latent_dim: int = 256,
                 hidden_dims: List[int] = None, device: str = None):
        """
        Initialize the Variational Autoencoder.

        Args:
            input_dim: Dimension of the input state
            latent_dim: Dimension of the latent representation (default 256)
            hidden_dims: List of hidden layer dimensions
            device: Torch device ('cuda' or 'cpu')
        """
        super(VariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(f"VAE-{input_dim}to{latent_dim}")

        # Default architecture if isn't specified
        if hidden_dims is None:
            # Scale hidden layers based on input size
            if input_dim <= 50:
                hidden_dims = [128, 64]
            elif input_dim <= 200:
                hidden_dims = [256, 128]
            else:
                hidden_dims = [512, 256, 128]

        self.hidden_dims = hidden_dims

        # Build encoder
        encoder_layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.LeakyReLU(0.2)
            ])

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build decoder
        decoder_layers = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        dims = hidden_dims[::-1]  # Reverse for decoder
        for i in range(len(dims) - 1):
            decoder_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.LeakyReLU(0.2)
            ])

        decoder_layers.extend([
            nn.Linear(dims[-1], input_dim),
            nn.Tanh()  # Assuming normalized inputs
        ])

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

        # Move to a device
        self.to(self.device)

        # Training metrics
        self.training_metrics = {
            'epochs_trained': 0,
            'best_loss': float('inf'),
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'last_updated': time.time()
        }

        self.logger.info(f"Initialized VAE: {input_dim}→{hidden_dims}→{latent_dim} on {self.device}")

    @staticmethod
    def _init_weights(module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space parameters.

        Args:
            x: Input state tensor

        Returns:
            Tuple of (mu, log_var) for the latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed state.

        Args:
            z: Latent vector

        Returns:
            Reconstructed state
        """
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input state

        Returns:
            Tuple of (reconstruction, input, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, x, mu, log_var

    @staticmethod
    def loss_function(reconstruction: torch.Tensor, original: torch.Tensor,
                      mu: torch.Tensor, log_var: torch.Tensor,
                      beta: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate VAE loss (reconstruction + KL divergence).

        Args:
            reconstruction: Reconstructed state
            original: Original state
            mu: Latent mean
            log_var: Latent log variance
            beta: Weight for KL divergence term (beta-VAE)

        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, original, reduction='mean')

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        loss_components = {
            'reconstruction': recon_loss.item(),
            'kl_divergence': kl_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_components

    def compress(self, state: np.ndarray) -> np.ndarray:
        """
        Compress state to latent representation (inference mode).

        Args:
            state: State to compress

        Returns:
            256-dimensional latent representation
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state.to(self.device)

            # Ensure correct shape
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            # Encode to latent space (use mean for deterministic encoding)
            mu, _ = self.encode(state_tensor)

            # Return as numpy array
            return mu.cpu().numpy().squeeze()

    def decompress(self, latent: np.ndarray) -> np.ndarray:
        """
        Decompress latent representation to state (inference mode).

        Args:
            latent: Latent representation

        Returns:
            Reconstructed state
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            if isinstance(latent, np.ndarray):
                latent_tensor = torch.FloatTensor(latent).to(self.device)
            else:
                latent_tensor = latent.to(self.device)

            # Ensure correct shape
            if latent_tensor.dim() == 1:
                latent_tensor = latent_tensor.unsqueeze(0)

            # Decode from latent space
            reconstruction = self.decode(latent_tensor)

            # Return as numpy array
            return reconstruction.cpu().numpy().squeeze()

    def get_reconstruction_error(self, state: np.ndarray) -> float:
        """
        Calculate reconstruction error for a state.

        Args:
            state: State to test

        Returns:
            Reconstruction error (MSE)
        """
        self.eval()
        with torch.no_grad():
            # Compress and decompress
            latent = self.compress(state)
            reconstruction = self.decompress(latent)

            # Calculate MSE
            error = np.mean((state - reconstruction) ** 2)
            return float(error)

    def update_metrics(self, loss_components: Dict[str, float]):
        """Update training metrics."""
        self.training_metrics['reconstruction_loss'] = loss_components['reconstruction']
        self.training_metrics['kl_loss'] = loss_components['kl_divergence']
        self.training_metrics['last_updated'] = time.time()

        if loss_components['total'] < self.training_metrics['best_loss']:
            self.training_metrics['best_loss'] = loss_components['total']

    def get_state_dict_serializable(self) -> Dict[str, Any]:
        """Get state dict in serializable format."""
        pytorch_state = self.state_dict()
        serializable_state = {}

        for key, tensor in pytorch_state.items():
            serializable_state[key] = tensor.cpu().numpy().tolist()

        return {
            'pytorch_state': serializable_state,
            'architecture': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'hidden_dims': self.hidden_dims
            },
            'training_metrics': self.training_metrics
        }

    def load_state_dict_serializable(self, state_dict: Dict[str, Any]):
        """Load state dict from serializable format."""
        # Verify architecture compatibility
        arch = state_dict['architecture']
        if (arch['input_dim'] != self.input_dim or
            arch['latent_dim'] != self.latent_dim):
            raise ValueError("Incompatible autoencoder architecture")

        # Load PyTorch state
        pytorch_state = {}
        for key, value in state_dict['pytorch_state'].items():
            pytorch_state[key] = torch.tensor(value, device=self.device)

        self.load_state_dict(pytorch_state)

        # Load metrics
        self.training_metrics = state_dict.get('training_metrics', self.training_metrics)


class AutoencoderTrainer:
    """
    Trainer for environment-specific autoencoders.

    Handles training, validation, and management of autoencoders
    for different environments.
    """

    def __init__(self, latent_dim: int = 256, device: str = None):
        """
        Initialize the autoencoder trainer.

        Args:
            latent_dim: Target latent dimension (default 256)
            device: Torch device
        """
        self.latent_dim = latent_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger("AutoencoderTrainer")

        # Store trained autoencoders
        self.autoencoders = {}

        # Training configuration
        self.training_config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'early_stopping_patience': 10,
            'beta': 1.0,  # Beta for beta-VAE
            'min_samples': 1000  # Minimum samples needed for training
        }

    def train_autoencoder(self, env_id: str, states: np.ndarray,
                         validation_split: float = 0.2) -> VariationalAutoencoder:
        """
        Train an autoencoder for a specific environment.

        Args:
            env_id: Environment identifier
            states: Array of states from the environment
            validation_split: Fraction of data for validation

        Returns:
            Trained autoencoder
        """
        self.logger.info(f"Training autoencoder for {env_id} with {len(states)} states")

        # Determine input dimension
        input_dim = states.shape[1] if states.ndim > 1 else states.shape[0]

        # Create autoencoder
        autoencoder = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            device=self.device
        )

        # Prepare data
        dataset = torch.FloatTensor(states).to(self.device)
        n_samples = len(dataset)
        n_val = int(n_samples * validation_split)

        # Split data
        indices = torch.randperm(n_samples)
        train_data = dataset[indices[n_val:]]
        val_data = dataset[indices[:n_val]]

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            autoencoder.parameters(),
            lr=self.training_config['learning_rate']
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.training_config['epochs']):
            # Training phase
            autoencoder.train()
            train_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                # Extract tensor from batch tuple
                batch_data = batch[0]

                # Forward pass
                recon, original, mu, log_var = autoencoder(batch_data)

                # Calculate loss
                loss, loss_components = autoencoder.loss_function(
                    recon, original, mu, log_var,
                    beta=self.training_config['beta']
                )

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            autoencoder.eval()
            with torch.no_grad():
                val_recon, val_original, val_mu, val_log_var = autoencoder(val_data)
                val_loss, val_components = autoencoder.loss_function(
                    val_recon, val_original, val_mu, val_log_var,
                    beta=self.training_config['beta']
                )

            # Update metrics
            autoencoder.update_metrics(val_components)
            autoencoder.training_metrics['epochs_trained'] = epoch + 1

            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                    f"Val Loss: {val_loss.item():.4f} "
                    f"(Recon: {val_components['reconstruction']:.4f}, "
                    f"KL: {val_components['kl_divergence']:.4f})"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.training_config['early_stopping_patience']:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Store trained autoencoder
        self.autoencoders[env_id] = autoencoder

        # Calculate final statistics
        reconstruction_errors = []
        with torch.no_grad():
            for state in val_data[:100]:  # Sample for statistics
                error = autoencoder.get_reconstruction_error(state.cpu().numpy())
                reconstruction_errors.append(error)

        avg_error = np.mean(reconstruction_errors)
        self.logger.info(
            f"Training complete for {env_id}. "
            f"Average reconstruction error: {avg_error:.6f}"
        )

        return autoencoder

    def get_or_train_autoencoder(self, env_id: str, states: Optional[np.ndarray] = None) -> Optional[VariationalAutoencoder]:
        """
        Get an existing autoencoder or train a new one if needed.

        Args:
            env_id: Environment identifier
            states: States for training (if needed)

        Returns:
            Autoencoder or None if insufficient data
        """
        # Return existing if available
        if env_id in self.autoencoders:
            return self.autoencoders[env_id]

        # Train new one if we have data
        if states is not None and len(states) >= self.training_config['min_samples']:
            return self.train_autoencoder(env_id, states)

        self.logger.warning(
            f"Insufficient data for {env_id} "
            f"({len(states) if states is not None else 0} < {self.training_config['min_samples']})"
        )
        return None

    def save_autoencoder(self, env_id: str, path: str):
        """Save autoencoder to file."""
        if env_id not in self.autoencoders:
            raise ValueError(f"No autoencoder found for {env_id}")

        autoencoder = self.autoencoders[env_id]
        state_dict = autoencoder.get_state_dict_serializable()

        # Save using torch
        torch.save(state_dict, path)
        self.logger.info(f"Saved autoencoder for {env_id} to {path}")

    def load_autoencoder(self, env_id: str, path: str) -> VariationalAutoencoder:
        """Load autoencoder from a file."""
        # Load state dict
        state_dict = torch.load(path, map_location=self.device)

        # Create autoencoder with correct architecture
        arch = state_dict['architecture']
        autoencoder = VariationalAutoencoder(
            input_dim=arch['input_dim'],
            latent_dim=arch['latent_dim'],
            hidden_dims=arch['hidden_dims'],
            device=self.device
        )

        # Load weights
        autoencoder.load_state_dict_serializable(state_dict)

        # Store in cache
        self.autoencoders[env_id] = autoencoder

        self.logger.info(f"Loaded autoencoder for {env_id} from {path}")
        return autoencoder

    def analyze_latent_space(self, env_id: str, states: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the latent space representation for an environment.

        Args:
            env_id: Environment identifier
            states: States to analyze

        Returns:
            Analysis results
        """
        if env_id not in self.autoencoders:
            return {'error': 'No autoencoder found for environment'}

        autoencoder = self.autoencoders[env_id]

        # Encode states
        latent_representations = []
        for state in states:
            latent = autoencoder.compress(state)
            latent_representations.append(latent)

        latent_representations = np.array(latent_representations)

        # Analyze latent space
        analysis = {
            'latent_dim': autoencoder.latent_dim,
            'dimension_stats': [],
            'total_variance': float(np.var(latent_representations)),
            'sparsity': float(np.mean(np.abs(latent_representations) < 0.01))
        }

        # Per-dimension analysis
        for i in range(autoencoder.latent_dim):
            dim_values = latent_representations[:, i]
            analysis['dimension_stats'].append({
                'index': i,
                'mean': float(np.mean(dim_values)),
                'std': float(np.std(dim_values)),
                'min': float(np.min(dim_values)),
                'max': float(np.max(dim_values)),
                'active': float(np.std(dim_values)) > 0.1
            })

        # Count active dimensions
        analysis['active_dimensions'] = sum(
            1 for stat in analysis['dimension_stats'] if stat['active']
        )

        return analysis