"""
Main Agent Byte class - the core of the modular AI agent.

This module implements the main agent that is completely environment-agnostic
and supports transfer learning across any environment. Now with autoencoder
support for better state representation learning.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import logging
import time
from datetime import datetime
import uuid

from numpy import floating

from .interfaces import Environment, Storage
from .config import AgentConfig, NetworkConfig, EnvironmentMetadata
from ..analysis.state_normalizer import StateNormalizer
from ..analysis.environment_analyzer import EnvironmentAnalyzer
from ..storage.json_numpy_storage import JsonNumpyStorage
from .dual_brain import DualBrain

class AgentByte:
    """
    The main Agent Byte class - a modular, transferable AI agent.

    This agent can learn from any environment implementing the Environment interface
    and transfer knowledge between different environments. Now supports autoencoder-based
    state compression for improved representation learning.
    """

    def __init__(self,
                 agent_id: str,
                 storage: Optional[Storage] = None,
                 config: Optional[AgentConfig] = None,
                 storage_backend: str = "json"):
        """
        Initialize Agent Byte.

        Args:
            agent_id: Unique identifier for this agent
            storage: Storage backend (defaults to JsonNumpyStorage)
            config: Agent configuration (defaults to AgentConfig)
            storage_backend: If storage is None, which backend to use ("json" or "vectordb")
        """
        self.agent_id = agent_id

        # Initialize storage with appropriate backend
        if storage is None:
            if storage_backend == "vectordb":
                from ..storage.vector_db_storage import VectorDBStorage
                self.storage = VectorDBStorage(f"./agent_data_vectordb/{agent_id}")
            else:
                self.storage = JsonNumpyStorage(f"./agent_data/{agent_id}")
        else:
            self.storage = storage

        self.config = config or AgentConfig()

        # Set up logging
        self.logger = logging.getLogger(f"AgentByte-{agent_id}")

        # Initialize components with autoencoder support
        self.state_normalizer = StateNormalizer(
            self.config.state_dimensions,
            use_autoencoder=True  # Enable autoencoder by default
        )
        self.environment_analyzer = EnvironmentAnalyzer()

        # Agent state
        self.current_environment = None
        self.environments_experienced = set()
        self.total_episodes = 0
        self.creation_time = datetime.now()

        # Load or create agent profile
        self._load_or_create_profile()

        # Neural network and brain components will be initialized when needed
        self.network = None
        self.dual_brain = None

        # Neural network and brain components will be initialized when needed
        self.dual_brain = None
        self.current_env_analysis = None

        # Load existing autoencoders
        self._load_autoencoders()

        self.logger.info(f"Agent Byte {agent_id} initialized with autoencoder support")

    def _load_autoencoders(self):
        """Load any existing autoencoders for this agent."""
        try:
            autoencoder_envs = self.storage.list_autoencoders(self.agent_id)
            for env_id in autoencoder_envs:
                self.state_normalizer.load_autoencoder(self.storage, self.agent_id, env_id)
                self.logger.info(f"Loaded autoencoder for environment: {env_id}")
        except Exception as e:
            self.logger.warning(f"Could not load autoencoders: {e}")

    def _load_or_create_profile(self):
        """Load existing agent profile or create new one."""
        profile = self.storage.get_agent_profile(self.agent_id)

        if profile:
            self.environments_experienced = set(profile.get('environments_experienced', []))
            self.total_episodes = profile.get('total_episodes', 0)
            self.creation_time = datetime.fromisoformat(profile.get('creation_time', datetime.now().isoformat()))
            self.logger.info(f"Loaded existing profile: {len(self.environments_experienced)} environments experienced")
        else:
            # Create a new profile
            self._save_profile()
            self.logger.info("Created new agent profile")

    def _save_profile(self):
        """Save agent profile to storage."""
        profile = {
            'agent_id': self.agent_id,
            'creation_time': self.creation_time.isoformat(),
            'environments_experienced': list(self.environments_experienced),
            'total_episodes': self.total_episodes,
            'config': self.config.to_dict(),
            'last_updated': datetime.now().isoformat(),
            'autoencoder_enabled': self.state_normalizer.use_autoencoder
        }

        self.storage.save_agent_profile(self.agent_id, profile)

    def train(self,
              env: Environment,
              episodes: int,
              env_metadata: Optional[Dict[str, Any]] = None,
              session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the agent on any environment.

        Args:
            env: Environment implementing the Environment interface
            episodes: Number of episodes to train
            env_metadata: Optional metadata about the environment
            session_id: Optional session identifier

        Returns:
            Training results and statistics
        """
        session_id = session_id or str(uuid.uuid4())
        env_id = env.get_id()

        self.logger.info(f"Starting training session {session_id} on {env_id}")

        # Analyze environment if first time
        if env_id not in self.environments_experienced:
            self.logger.info(f"First time encountering {env_id}, analyzing...")
            analysis = self.environment_analyzer.analyze(env, env_metadata)
            self._process_environment_analysis(env_id, analysis)
            self.environments_experienced.add(env_id)

        # Initialize components for this environment
        self._initialize_for_environment(env)

        # Training statistics
        training_stats = {
            'session_id': session_id,
            'environment': env_id,
            'episodes': episodes,
            'start_time': time.time(),
            'episode_rewards': [],
            'episode_lengths': [],
            'learning_events': [],
            'autoencoder_status': self._get_autoencoder_status(env_id)
        }

        # Main training loop
        for episode in range(episodes):
            episode_reward, episode_length = self._train_episode(env, episode)

            training_stats['episode_rewards'].append(episode_reward)
            training_stats['episode_lengths'].append(episode_length)

            # Log progress
            if (episode + 1) % max(1, episodes // 10) == 0:
                avg_reward = np.mean(training_stats['episode_rewards'][-100:])
                self.logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")

            # Save periodically
            if (episode + 1) % self.config.save_interval == 0:
                self._save_progress(env_id)

        # Final save including autoencoders
        self._save_progress(env_id)
        self.total_episodes += episodes
        self._save_profile()

        # Calculate final statistics
        training_stats['end_time'] = time.time()
        training_stats['duration'] = training_stats['end_time'] - training_stats['start_time']
        training_stats['final_performance'] = {
            'mean_reward': float(np.mean(training_stats['episode_rewards'])),
            'std_reward': float(np.std(training_stats['episode_rewards'])),
            'mean_length': float(np.mean(training_stats['episode_lengths'])),
            'improvement': self._calculate_improvement(training_stats['episode_rewards'])
        }

        # Update autoencoder status
        training_stats['autoencoder_status_final'] = self._get_autoencoder_status(env_id)

        self.logger.info(f"Training completed: {training_stats['final_performance']}")

        return training_stats

    def _get_autoencoder_status(self, env_id: str) -> Dict[str, Any]:
        """Get the status of autoencoder for an environment."""
        return self.state_normalizer.get_normalization_info(env_id)

    def _train_episode(self, env: Environment, episode_num: int) -> Tuple[float, int]:
        """
        Train a single episode.

        Args:
            env: Environment to train on
            episode_num: Current episode number

        Returns:
            Tuple of (episode_reward, episode_length)
        """
        # Start an episode in dual brain
        if self.dual_brain is not None:
            self.dual_brain.start_episode()

        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done and episode_length < 10000:  # Prevent infinite episodes
            # Normalize state (may use autoencoder)
            normalized_state = self.state_normalizer.normalize(
                state, env.get_id()
            )

            # Get action from agent
            action = self._select_action(normalized_state, env)

            # Take action in environment
            next_state, reward, done, info = env.step(action)

            # Store experience
            self._store_experience(
                state, action, reward, next_state, done, env.get_id()
            )

            # Learn from experience
            self._learn(env.get_id())

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

            # Update exploration rate
            if self.config.exploration_rate > self.config.min_exploration:
                self.config.exploration_rate *= self.config.exploration_decay

                # End episode in dual brain
                if self.dual_brain is not None:
                    self.dual_brain.end_episode()

        return episode_reward, episode_length

    def _select_action(self, normalized_state: np.ndarray, env: Environment) -> int:
        """
        Select action using neural-symbolic dual brain.
        """
        # Initialize dual brain if needed
        if self.dual_brain is None:
            self._initialize_dual_brain(env.get_action_size())

        # Get latent state if autoencoder is available
        latent_state = None
        env_id = env.get_id()
        if (self.state_normalizer.use_autoencoder and
                self.state_normalizer.autoencoder_trainer and
                env_id in self.state_normalizer.autoencoder_trainer.autoencoders):
            autoencoder = self.state_normalizer.autoencoder_trainer.autoencoders[env_id]
            # Get the raw state for autoencoder
            raw_state = env.reset() if not hasattr(self, '_last_raw_state') else self._last_raw_state
            latent_state = autoencoder.compress(raw_state)

        # Make a decision using dual brain
        action = self.dual_brain.decide(normalized_state, self.config.exploration_rate, latent_state)

        return action

    def _initialize_dual_brain(self, action_size: int):
        """Initialize the dual brain system."""
        self.dual_brain = DualBrain(
            agent_id=self.agent_id,
            action_size=action_size,
            storage=self.storage,
            config=self.config.to_dict()
        )

        # Set environment analysis if available
        if self.current_env_analysis:
            self.dual_brain.set_environment_analysis(self.current_env_analysis)

        self.logger.info(f"Initialized dual brain with action size {action_size}")

    def _store_experience(self, state: np.ndarray, action: int, reward: float,
                          next_state: np.ndarray, done: bool, env_id: str):
        """
        Store experience for learning.

        """
        # Normalize states for storage (using autoencoder if available)
        normalized_state = self.state_normalizer.normalize(state, env_id)
        normalized_next_state = self.state_normalizer.normalize(next_state, env_id)

        # Create experience vector for similarity search
        experience_vector = np.concatenate([
            normalized_state[:128],  # First half of state
            normalized_next_state[:128]  # First half of the next state
        ])

        metadata = {
            'env_id': env_id,
            'action': action,
            'reward': reward,
            'done': done,
            'timestamp': time.time(),
            'normalization_method': self.state_normalizer.normalization_methods.get(env_id, 'unknown')
        }

        # Save experience vector
        self.storage.save_experience_vector(
            self.agent_id, experience_vector, metadata
        )

        # Store for dual brain learning
        self._last_experience = {
            'state': normalized_state,
            'action': action,
            'reward': reward,
            'next_state': normalized_next_state,
            'done': done
        }

    def _learn(self, env_id: str):
        """
        Learn from experiences using dual brain.
        """
        if self.dual_brain is not None and hasattr(self, '_last_experience'):
            self.dual_brain.learn(self._last_experience)

    def transfer_to(self, target_env: Environment,
                    initial_episodes: int = 10) -> Dict[str, Any]:
        """
        Transfer learned knowledge to a new environment.

        Args:
            target_env: Target environment to transfer to
            initial_episodes: Number of episodes to initially explore

        Returns:
            Transfer learning results
        """
        target_id = target_env.get_id()
        self.logger.info(f"Transferring knowledge to {target_id}")

        transfer_results = {
            'source_environments': list(self.environments_experienced),
            'target_environment': target_id,
            'transfer_time': time.time(),
            'similar_experiences_found': 0,
            'transferred_skills': [],
            'autoencoder_compatibility': self._check_autoencoder_compatibility(target_env)
        }

        # Prepare transfer package if dual brain exists
        if self.dual_brain is not None:
            transfer_package = self.dual_brain.prepare_for_transfer(target_id)
            transfer_results['transfer_package'] = transfer_package
            transfer_results['transferred_skills'] = list(transfer_package.get(
                'symbolic_knowledge', {}
            ).get('transferable_skills', {}).keys())

        # Analyze target environment
        analysis = self.environment_analyzer.analyze(target_env)
        self._process_environment_analysis(target_id, analysis)

        # Find similar experiences from other environments
        test_state = target_env.reset()
        normalized_test = self.state_normalizer.normalize(test_state, target_id)

        similar_experiences = self.storage.search_similar_experiences(
            self.agent_id, normalized_test, k=20
        )

        transfer_results['similar_experiences_found'] = len(similar_experiences)

        # Log similar experiences
        for exp in similar_experiences[:5]:
            self.logger.info(
                f"Similar experience from {exp['metadata']['env_id']}: "
                f"similarity={exp['similarity']:.3f}, reward={exp['metadata']['reward']:.2f}, "
                f"normalization={exp['metadata'].get('normalization_method', 'unknown')}"
            )

        # Perform initial training with transfer knowledge
        self.logger.info(f"Performing {initial_episodes} initial episodes on target")
        training_results = self.train(target_env, initial_episodes)

        transfer_results['initial_performance'] = training_results['final_performance']

        return transfer_results

    def _check_autoencoder_compatibility(self, target_env: Environment) -> Dict[str, Any]:
        """Check if we can reuse autoencoders from similar environments."""
        target_id = target_env.get_id()
        target_state_size = target_env.get_state_size()

        compatibility = {
            'target_state_size': target_state_size,
            'compatible_autoencoders': [],
            'exact_match': False
        }

        # Check existing autoencoders
        for env_id in self.state_normalizer.autoencoder_trainer.autoencoders:
            autoencoder = self.state_normalizer.autoencoder_trainer.autoencoders[env_id]
            if autoencoder.input_dim == target_state_size:
                compatibility['compatible_autoencoders'].append({
                    'env_id': env_id,
                    'input_dim': autoencoder.input_dim,
                    'training_metrics': autoencoder.training_metrics
                })
                if env_id == target_id:
                    compatibility['exact_match'] = True

        return compatibility

    def _process_environment_analysis(self, env_id: str, analysis: Dict[str, Any]):
        """Process and store environment analysis result."""
        # Extract key information
        understanding = {
            'env_id': env_id,
            'analyzed_at': datetime.now().isoformat(),
            'state_size': analysis['state_size'],
            'action_size': analysis['action_size'],
            'environment_type': analysis['environment_type'],
            'reward_structure': analysis['reward_analysis']['reward_types'],
            'objectives': analysis['inferred_objectives'],
            'rules': analysis['inferred_rules'],
            'understanding': analysis['understanding']
        }

        # Store in knowledge base (placeholder for now)
        self.logger.info(f"Processed analysis for {env_id}: {understanding['environment_type']}")
        # Store for dual brain
        self.current_env_analysis = analysis

        # Update dual brain if it exists
        if self.dual_brain is not None:
            self.dual_brain.set_environment_analysis(analysis)

    def _initialize_for_environment(self, env: Environment):
        """Initialize components for a specific environment."""
        env_id = env.get_id()
        self.current_environment = env_id

        # Neural network and dual brain initialization will be added in Phase 3
        self.logger.info(f"Initialized components for {env_id}")

    # Update _save_progress to include dual brain and autoencoders:
    def _save_progress(self, env_id: str):
        """Save current training progress."""
        if self.dual_brain is not None:
            self.dual_brain.save_state(env_id)
            self.logger.debug(f"Saved dual brain state for {env_id}")

        # Save autoencoders
        self.state_normalizer.save_autoencoders(self.storage, self.agent_id)
        self.logger.debug(f"Saved autoencoders")

    def _calculate_improvement(self, rewards: List[float]) -> float:
        """Calculate normalized improvement over training.
        Returns:
            Relative improvement: (late_mean - early_mean) / abs(early_mean)
            0.0 if not enough data or if both means are zero.
        """
        if len(rewards) < 20:
            return 0.0  # Not enough data for meaningful trend

        n = int(max(1, len(rewards) // 10))

        early_mean = np.mean(rewards[:n])
        late_mean = np.mean(rewards[-n:])

        eps = 1e-8  # Small value to avoid zero division

        if abs(early_mean) < eps:
            if abs(late_mean) < eps:
                return 0.0  # No change, all rewards are near zero
            else:
                return float(np.sign(late_mean))  # Sudden jump from zero to nonzero
        return (late_mean - early_mean) / (abs(early_mean) + eps)

    # Add method to get insights:
    def get_insights(self) -> Dict[str, Any]:
        """Get insights from the agent's learning."""
        insights = {
            'agent_id': self.agent_id,
            'environments_experienced': list(self.environments_experienced),
            'total_episodes': self.total_episodes,
            'normalization_methods': {}
        }

        # Add normalization info for each environment
        for env_id in self.environments_experienced:
            insights['normalization_methods'][env_id] = self.state_normalizer.get_normalization_info(env_id)

        if self.dual_brain is not None:
            insights['dual_brain'] = self.dual_brain.get_insights()

        return insights

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'total_episodes': self.total_episodes,
            'environments_experienced': list(self.environments_experienced),
            'creation_time': self.creation_time.isoformat(),
            'current_environment': self.current_environment,
            'exploration_rate': self.config.exploration_rate,
            'config': self.config.to_dict(),
            'autoencoder_enabled': self.state_normalizer.use_autoencoder,
            'autoencoders_trained': len(self.state_normalizer.autoencoder_trainer.autoencoders) if self.state_normalizer.autoencoder_trainer else 0
        }

    def reset_exploration(self):
        """Reset exploration rate to initial value."""
        self.config.exploration_rate = AgentConfig().exploration_rate
        self.logger.info(f"Reset exploration rate to {self.config.exploration_rate}")

    def save(self):
        """Save agent state."""
        self._save_profile()
        if self.current_environment:
            self._save_progress(self.current_environment)
        self.logger.info("Agent state saved")