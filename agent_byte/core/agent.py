"""
Main Agent Byte class - the core of the modular AI agent.

This module implements the main agent that is completely environment-agnostic
and supports transfer learning across any environment. Now enhanced with
robust error handling, checkpointing, and recovery mechanisms (Sprint 5).
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import logging
import time
from datetime import datetime
import uuid
import signal
import threading

# NEW Sprint 5 imports
try:
    import psutil  # For system monitoring
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system monitoring disabled")

from numpy import floating

from .interfaces import Environment, Storage
from .config import AgentConfig, NetworkConfig, EnvironmentMetadata
from ..analysis.state_normalizer import StateNormalizer
from ..analysis.environment_analyzer import EnvironmentAnalyzer
from ..storage.json_numpy_storage import JsonNumpyStorage
from .dual_brain import DualBrain
from .checkpoint_manager import CheckpointManager  # NEW Sprint 5


class AgentByte:
    """
    The main Agent Byte class - a modular, transferable AI agent.

    This agent can learn from any environment implementing the Environment interface
    and transfer knowledge between different environments. Enhanced with robust
    error handling, automatic checkpointing, and recovery mechanisms.
    """

    def __init__(self,
                 agent_id: str,
                 storage: Optional[Storage] = None,
                 config: Optional[AgentConfig] = None,
                 storage_backend: str = "json",
                 enable_checkpointing: bool = True):  # NEW Sprint 5 parameter
        """
        Initialize Agent Byte with enhanced robustness features.

        Args:
            agent_id: Unique identifier for this agent
            storage: Storage backend (defaults to JsonNumpyStorage)
            config: Agent configuration (defaults to AgentConfig)
            storage_backend: If storage is None, which backend to use ("json" or "vectordb")
            enable_checkpointing: Enable automatic checkpointing and recovery
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

        # Neural network and brain components will be initialized when needed
        self.dual_brain = None
        self.current_env_analysis = None

        # NEW Sprint 5: Checkpoint manager
        self.checkpoint_manager = None
        if enable_checkpointing:
            try:
                self.checkpoint_manager = CheckpointManager(
                    agent_id=self.agent_id,
                    storage=self.storage,
                    timer_interval=900,  # 15 minutes
                    episode_interval=500,
                    max_checkpoints=10
                )
                self._register_checkpoint_providers()
            except Exception as e:
                self.logger.warning(f"Checkpoint manager initialization failed: {e}")
                self.checkpoint_manager = None

        # NEW Sprint 5: Environment health monitoring
        self.environment_health = {
            'consecutive_failures': 0,
            'last_successful_step': time.time(),
            'timeout_threshold': 300,  # 5 minutes
            'max_step_time': 30,  # 30 seconds per step
            'failure_threshold': 5
        }

        # NEW Sprint 5: Training session state
        self.training_session = {
            'active': False,
            'start_time': None,
            'current_environment': None,
            'episodes_completed': 0,
            'last_save_time': time.time()
        }

        # NEW Sprint 5: Error recovery state
        self.recovery_state = {
            'recovery_attempts': 0,
            'max_recovery_attempts': 3,
            'last_recovery_time': None,
            'degraded_mode': False
        }

        # Load or create agent profile
        self._load_or_create_profile()

        # NEW Sprint 5: Training state tracking
        self._current_training_stats = {}  # For passing stats to _train_episode
        self._last_raw_state = None  # Track raw state for autoencoder

        # Load existing autoencoders
        self._load_autoencoders()

        self.logger.info(f"Agent Byte {agent_id} initialized with checkpointing: {enable_checkpointing}")


    def _load_autoencoders(self):
        """Load any existing autoencoders for this agent."""
        try:
            # Type hint for IDE to recognize the interface
            storage: Storage = self.storage
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

    # NEW Sprint 5: Checkpoint management methods
    def _register_checkpoint_providers(self):
        """Register functions that provide state for checkpointing."""
        if not self.checkpoint_manager:
            return

        # Agent configuration and profile
        self.checkpoint_manager.register_state_provider(
            'agent_profile',
            self._get_checkpoint_profile_state
        )

        # Training session state
        self.checkpoint_manager.register_state_provider(
            'training_session',
            lambda: self.training_session.copy()
        )

        # System metrics
        self.checkpoint_manager.register_state_provider(
            'system_metrics',
            self._get_system_metrics
        )

    def _get_checkpoint_profile_state(self) -> Dict[str, Any]:
        """Get agent profile state for checkpointing (consolidates with existing profile logic)."""
        # Reuse existing profile logic
        return {
            'agent_id': self.agent_id,
            'environments_experienced': list(self.environments_experienced),
            'total_episodes': self.total_episodes,
            'creation_time': self.creation_time.isoformat(),
            'config': self.config.to_dict(),
            'normalization_methods': getattr(self.state_normalizer, 'normalization_methods', {}),
            'autoencoder_enabled': self.state_normalizer.use_autoencoder,
            'current_environment': self.current_environment
        }

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics for monitoring."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return {
                    'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(),
                    'open_files': len(process.open_files()),
                    'timestamp': time.time()
                }
            else:
                return {'timestamp': time.time(), 'psutil_unavailable': True}
        except Exception:
            return {'timestamp': time.time(), 'error': 'metrics_unavailable'}

    def train(self,
              env: Environment,
              episodes: int,
              env_metadata: Optional[Dict[str, Any]] = None,
              session_id: Optional[str] = None,
              enable_recovery: bool = True) -> Dict[str, Any]:  # NEW Sprint 5 parameter
        """
        Train the agent on any environment with enhanced error handling and checkpointing.

        Args:
            env: Environment implementing the Environment interface
            episodes: Number of episodes to train
            env_metadata: Optional metadata about the environment
            session_id: Optional session identifier
            enable_recovery: Enable automatic recovery from checkpoints

        Returns:
            Training results and statistics including error metrics
        """
        session_id = session_id or str(uuid.uuid4())
        env_id = env.get_id()

        # NEW Sprint 5: Initialize training session
        self.training_session.update({
            'active': True,
            'start_time': time.time(),
            'current_environment': env_id,
            'episodes_completed': 0,
            'session_id': session_id
        })

        self.logger.info(f"Starting enhanced training session {session_id} on {env_id}")

        # NEW Sprint 5: Attempt recovery if enabled
        if enable_recovery and self.checkpoint_manager:
            self._attempt_recovery(env_id)

        # Enhanced training statistics (extends original statistics)
        training_stats = {
            'session_id': session_id,
            'environment': env_id,
            'episodes': episodes,
            'start_time': time.time(),
            'episode_rewards': [],
            'episode_lengths': [],
            'learning_events': [],
            'autoencoder_status': self._get_autoencoder_status(env_id),
            # NEW Sprint 5: Error and health tracking
            'error_events': [],
            'recovery_events': [],
            'checkpoint_events': [],
            'health_metrics': []
        }

        try:
            # Analyze environment if first time (with NEW Sprint 5 error handling)
            if env_id not in self.environments_experienced:
                self._safe_environment_analysis(env, env_metadata, training_stats)

            # Initialize components for this environment (with NEW Sprint 5 error handling)
            self._safe_initialize_for_environment(env, training_stats)

            # NEW Sprint 5: Register dual brain state provider after initialization
            if self.checkpoint_manager and self.dual_brain:
                self.checkpoint_manager.register_state_provider(
                    'dual_brain',
                    lambda: self._get_dual_brain_checkpoint_state(env_id)
                )

            # Main training loop with NEW Sprint 5 enhanced error handling
            for episode in range(episodes):
                try:
                    episode_start_time = time.time()

                    # NEW Sprint 5: Check system health before episode
                    if not self._check_system_health():
                        self.logger.warning("System health check failed, enabling degraded mode")
                        self.recovery_state['degraded_mode'] = True

                    # NEW Sprint 5: Store training stats for the episode method to access
                    self._current_training_stats = training_stats

                    # Train episode (keeping original method signature)
                    episode_reward, episode_length = self._train_episode(env, episode)

                    episode_duration = time.time() - episode_start_time

                    # Record episode results
                    training_stats['episode_rewards'].append(episode_reward)
                    training_stats['episode_lengths'].append(episode_length)

                    # NEW Sprint 5: Update session state
                    self.training_session['episodes_completed'] = episode + 1

                    # NEW Sprint 5: Update checkpoint manager with episode count
                    if self.checkpoint_manager:
                        self.checkpoint_manager.update_episode_count(self.total_episodes + episode + 1)

                    # NEW Sprint 5: Health monitoring
                    self._update_health_metrics(episode_duration, training_stats)

                    # NEW Sprint 5: Reset failure count on successful episode
                    self.environment_health['consecutive_failures'] = 0
                    self.environment_health['last_successful_step'] = time.time()

                    # Log progress
                    if (episode + 1) % max(1, episodes // 10) == 0:
                        avg_reward = np.mean(training_stats['episode_rewards'][-100:])
                        self.logger.info(
                            f"Episode {episode + 1}/{episodes}, "
                            f"Avg Reward: {avg_reward:.2f}, "
                            f"Duration: {episode_duration:.2f}s"
                        )

                    # Periodic saves with NEW Sprint 5 error handling
                    if (episode + 1) % self.config.save_interval == 0:
                        self._safe_save_progress(env_id, training_stats)

                except Exception as e:
                    # NEW Sprint 5: Enhanced error handling
                    self._handle_training_error(e, episode, training_stats)
                    if not self._should_continue_training():
                        break

            # Final save with NEW Sprint 5 error handling
            self._safe_save_progress(env_id, training_stats)

        except Exception as e:
            # NEW Sprint 5: Critical error handling
            self.logger.error(f"Critical training failure: {e}")
            training_stats['error_events'].append({
                'type': 'critical_failure',
                'error': str(e),
                'timestamp': time.time()
            })

            # NEW Sprint 5: Attempt emergency checkpoint
            if self.checkpoint_manager:
                self.checkpoint_manager.create_manual_checkpoint('emergency')

        finally:
            # Always cleanup training session
            self.training_session['active'] = False
            self.total_episodes += self.training_session.get('episodes_completed', 0)
            self._save_profile()

        # Calculate final statistics (enhanced with error metrics)
        training_stats['end_time'] = time.time()
        training_stats['duration'] = training_stats['end_time'] - training_stats['start_time']
        training_stats['final_performance'] = self._calculate_performance_with_errors(training_stats)

        self.logger.info(f"Enhanced training completed: {training_stats['final_performance']}")
        return training_stats

    def _train_episode(self, env: Environment, episode_num: int) -> Tuple[float, int]:
        """
        Train a single episode with enhanced robustness (keeps original signature).

        Args:
            env: Environment to train on
            episode_num: Current episode number

        Returns:
            Tuple of (episode_reward, episode_length)
        """
        # NEW Sprint 5: Access training stats from instance variable
        training_stats = getattr(self, '_current_training_stats', {})

        # NEW Sprint 5: Timeout and safety parameters
        episode_timeout = self.environment_health['timeout_threshold']
        step_timeout = self.environment_health['max_step_time']
        episode_start = time.time()

        try:
            # Start an episode in dual brain
            if self.dual_brain is not None:
                self.dual_brain.start_episode()

            # NEW Sprint 5: Reset environment with timeout protection
            state = self._safe_env_reset(env, step_timeout)
            self._last_raw_state = state.copy()  # Store for autoencoder

            done = False
            episode_reward = 0.0
            episode_length = 0
            last_step_time = time.time()  # NEW Sprint 5

            while not done and episode_length < 10000:  # Prevent infinite episodes
                # NEW Sprint 5: Check for episode timeout
                if time.time() - episode_start > episode_timeout:
                    self.logger.warning(f"Episode {episode_num} timeout after {episode_timeout}s")
                    if training_stats:
                        training_stats.setdefault('error_events', []).append({
                            'type': 'episode_timeout',
                            'episode': episode_num,
                            'duration': time.time() - episode_start,
                            'timestamp': time.time()
                        })
                    break

                # NEW Sprint 5: Check for step timeout
                if time.time() - last_step_time > step_timeout:
                    self.logger.warning(f"Step timeout in episode {episode_num}")
                    if training_stats:
                        training_stats.setdefault('error_events', []).append({
                            'type': 'step_timeout',
                            'episode': episode_num,
                            'step': episode_length,
                            'timestamp': time.time()
                        })
                    break

                # Normalize state (NEW Sprint 5: with error handling)
                try:
                    normalized_state = self.state_normalizer.normalize(state, env.get_id())
                except Exception as e:
                    self.logger.error(f"State normalization failed: {e}")
                    normalized_state = np.zeros(self.config.state_dimensions)

                # Get action from agent (NEW Sprint 5: with timeout protection)
                try:
                    action = self._select_action_with_timeout(normalized_state, env, step_timeout)
                except TimeoutError:
                    action = np.random.randint(0, env.get_action_size())
                    self.logger.warning(f"Action selection timeout, using random action: {action}")
                except Exception as e:
                    self.logger.error(f"Action selection failed: {e}")
                    action = np.random.randint(0, env.get_action_size())

                # Take action in environment (NEW Sprint 5: with error handling)
                try:
                    next_state, reward, done, info = env.step(action)

                    # NEW Sprint 5: Validate step results
                    if not self._validate_step_results(next_state, reward, done):
                        self.logger.warning("Invalid step results detected")
                        reward = 0.0
                        if next_state is None:
                            next_state = state.copy()

                except Exception as e:
                    self.logger.error(f"Environment step failed: {e}")
                    self.environment_health['consecutive_failures'] += 1
                    next_state = state.copy()
                    reward = 0.0
                    done = True

                # Store experience (NEW Sprint 5: with error handling)
                try:
                    self._store_experience(state, action, reward, next_state, done, env.get_id())
                except Exception as e:
                    self.logger.error(f"Experience storage failed: {e}")

                # Learn from experience (NEW Sprint 5: with error handling)
                try:
                    self._learn(env.get_id())
                except Exception as e:
                    self.logger.error(f"Learning step failed: {e}")

                # Update state
                state = next_state
                self._last_raw_state = next_state.copy()  # Store for autoencoder
                episode_reward += reward
                episode_length += 1
                last_step_time = time.time()  # NEW Sprint 5

                # Update exploration rate
                if self.config.exploration_rate > self.config.min_exploration:
                    self.config.exploration_rate *= self.config.exploration_decay

            # End episode in dual brain
            if self.dual_brain is not None:
                self.dual_brain.end_episode()

            # NEW Sprint 6: Validate transfers periodically
            if (episode_num % self.config.transfer_validation_interval == 0 and
                    self.dual_brain and self.dual_brain.symbolic_brain and
                    hasattr(self.dual_brain.symbolic_brain, 'knowledge_system')):

                current_performance = {
                    'average_reward': episode_reward,
                    'episode_length': episode_length,
                    'step_count': episode_num,
                    'timestamp': time.time()
                }

                validation_result = self.dual_brain.symbolic_brain.knowledge_system.validate_active_transfers(
                    env.get_id(), current_performance
                )

                if validation_result:
                    self.logger.info(f"Transfer validation completed: {validation_result['overall_success']}")

            return episode_reward, episode_length

        except Exception as e:
            # NEW Sprint 5: Episode-level error handling
            self.logger.error(f"Episode {episode_num} failed with error: {e}")
            self.environment_health['consecutive_failures'] += 1
            return 0.0, 0

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
            # Use stored raw state, fallback to current normalized state
            raw_state = self._last_raw_state if self._last_raw_state is not None else normalized_state
            latent_state = autoencoder.compress(raw_state)

        # Make a decision using dual brain
        action = self.dual_brain.decide(normalized_state, self.config.exploration_rate, latent_state)

        return action

    # NEW Sprint 5: Enhanced action selection with timeout
    def _select_action_with_timeout(self, normalized_state: np.ndarray,
                                   env: Environment, timeout: float) -> int:
        """Select action with timeout protection."""
        start_time = time.time()

        try:
            # Initialize dual brain if needed
            if self.dual_brain is None:
                self._initialize_dual_brain(env.get_action_size())

            # Get latent state if autoencoder is available
            latent_state = None
            env_id = env.get_id()
            if (self.state_normalizer.use_autoencoder and
                    self.state_normalizer.autoencoder_trainer and
                    env_id in self.state_normalizer.autoencoder_trainer.autoencoders):
                try:
                    autoencoder = self.state_normalizer.autoencoder_trainer.autoencoders[env_id]
                    raw_state = getattr(self, '_last_raw_state', normalized_state)
                    latent_state = autoencoder.compress(raw_state)
                except Exception as e:
                    self.logger.warning(f"Autoencoder compression failed: {e}")

            # Make decision using dual brain
            action = self.dual_brain.decide(
                normalized_state,
                self.config.exploration_rate,
                latent_state
            )

            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError("Action selection timeout")

            return action

        except TimeoutError:
            raise
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            return np.random.randint(0, env.get_action_size())

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
        if (self.state_normalizer.autoencoder_trainer and
            hasattr(self.state_normalizer.autoencoder_trainer, 'autoencoders')):
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
        self.logger.info(f"Initialized components for {env_id}")

    def _save_progress(self, env_id: str):
        """Save current training progress."""
        if self.dual_brain is not None:
            self.dual_brain.save_state(env_id)
            self.logger.debug(f"Saved dual brain state for {env_id}")

        # Save autoencoders
        self.state_normalizer.save_autoencoders(self.storage, self.agent_id)
        self.logger.debug(f"Saved autoencoders")

    def _calculate_improvement(self, rewards: List[float]) -> float:
        """Calculate normalized improvement over training."""
        if len(rewards) < 20:
            return 0.0

        n = int(max(1, len(rewards) // 10))
        early_mean = np.mean(rewards[:n])
        late_mean = np.mean(rewards[-n:])
        eps = 1e-8

        if abs(early_mean) < eps:
            if abs(late_mean) < eps:
                return 0.0
            else:
                return float(np.sign(late_mean))
        return (late_mean - early_mean) / (abs(early_mean) + eps)

    def _get_autoencoder_status(self, env_id: str) -> Dict[str, Any]:
        """Get the status of autoencoder for an environment."""
        return self.state_normalizer.get_normalization_info(env_id)

    # NEW Sprint 5: Robustness and error handling methods

    def _safe_env_reset(self, env: Environment, timeout: float) -> np.ndarray:
        """Reset environment with timeout protection."""
        try:
            start_time = time.time()
            state = env.reset()

            if time.time() - start_time > timeout:
                self.logger.warning("Environment reset timeout")

            return state

        except Exception as e:
            self.logger.error(f"Environment reset failed: {e}")
            return np.zeros(env.get_state_size())

    def _validate_step_results(self, next_state, reward, done) -> bool:
        """Validate step results from environment."""
        try:
            if next_state is None:
                return False
            if not isinstance(reward, (int, float)) or np.isnan(reward):
                return False
            if not isinstance(done, bool):
                return False
            return True
        except Exception:
            return False

    def _handle_training_error(self, error: Exception, episode: int, training_stats: Dict[str, Any]):
        """Handle general training errors."""
        self.environment_health['consecutive_failures'] += 1

        error_event = {
            'type': 'training_error',
            'error': str(error),
            'episode': episode,
            'consecutive_failures': self.environment_health['consecutive_failures'],
            'timestamp': time.time()
        }

        training_stats.setdefault('error_events', []).append(error_event)
        self.logger.error(f"Training error in episode {episode}: {error}")

    def _should_continue_training(self) -> bool:
        """Determine if training should continue based on error state."""
        if self.environment_health['consecutive_failures'] >= self.environment_health['failure_threshold']:
            return False
        if self.recovery_state['recovery_attempts'] >= self.recovery_state['max_recovery_attempts']:
            return False
        return True

    def _check_system_health(self) -> bool:
        """Check system health metrics."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024

                if memory_mb > 8192:  # 8GB threshold
                    self.logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                    return False

                # Check disk space
                disk_usage = psutil.disk_usage(
                    self.storage.base_path if hasattr(self.storage, 'base_path') else '.'
                )
                free_gb = disk_usage.free / 1024 / 1024 / 1024

                if free_gb < 1.0:  # 1GB threshold
                    self.logger.error(f"Low disk space: {free_gb:.1f}GB")
                    self._notify_critical_error(f"Low disk space: {free_gb:.1f}GB")
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return True

    def _update_health_metrics(self, episode_duration: float, training_stats: Dict[str, Any]):
        """Update health metrics tracking."""
        health_metric = {
            'timestamp': time.time(),
            'episode_duration': episode_duration,
            'memory_usage_mb': 0,
            'consecutive_failures': self.environment_health['consecutive_failures'],
            'degraded_mode': self.recovery_state['degraded_mode']
        }

        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                health_metric['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
        except Exception:
            pass

        training_stats.setdefault('health_metrics', []).append(health_metric)

        # Keep only recent metrics
        if len(training_stats['health_metrics']) > 100:
            training_stats['health_metrics'] = training_stats['health_metrics'][-100:]

    def _safe_save_progress(self, env_id: str, training_stats: Dict[str, Any]):
        """Save progress with error handling."""
        try:
            if self.dual_brain is not None:
                success = self.dual_brain.save_state(env_id)
                if not success:
                    raise Exception("Dual brain save failed")

            self.state_normalizer.save_autoencoders(self.storage, self.agent_id)
            self.training_session['last_save_time'] = time.time()
            self.logger.debug(f"Saved progress for {env_id}")

        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
            training_stats.setdefault('error_events', []).append({
                'type': 'save_failure',
                'error': str(e),
                'timestamp': time.time()
            })
            self._notify_critical_error(f"Save failure: {e}")

    def _safe_environment_analysis(self, env: Environment, env_metadata: Optional[Dict[str, Any]],
                                  training_stats: Dict[str, Any]):
        """Safely analyze environment with error handling."""
        try:
            self.logger.info(f"Analyzing environment: {env.get_id()}")
            analysis = self.environment_analyzer.analyze(env, env_metadata)
            self._process_environment_analysis(env.get_id(), analysis)
            self.environments_experienced.add(env.get_id())

        except Exception as e:
            self.logger.error(f"Environment analysis failed: {e}")
            training_stats.setdefault('error_events', []).append({
                'type': 'analysis_failure',
                'error': str(e),
                'timestamp': time.time()
            })

    def _safe_initialize_for_environment(self, env: Environment, training_stats: Dict[str, Any]):
        """Safely initialize components for environment."""
        try:
            env_id = env.get_id()
            self.current_environment = env_id
            self.logger.info(f"Initializing components for {env_id}")

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            training_stats.setdefault('error_events', []).append({
                'type': 'initialization_failure',
                'error': str(e),
                'timestamp': time.time()
            })

    def _attempt_recovery(self, env_id: str):
        """Attempt to recover from a previous session."""
        if not self.checkpoint_manager:
            return

        try:
            checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()

            if checkpoint_data:
                self.logger.info(f"Attempting recovery from checkpoint: {checkpoint_data['checkpoint_id']}")

                # Restore agent state
                if 'agent_profile' in checkpoint_data['components']:
                    profile_data = checkpoint_data['components']['agent_profile']
                    self.total_episodes = profile_data.get('total_episodes', 0)
                    self.environments_experienced = set(profile_data.get('environments_experienced', []))

                # Restore training session state
                if 'training_session' in checkpoint_data['components']:
                    session_data = checkpoint_data['components']['training_session']
                    self.training_session.update(session_data)

                self.recovery_state['recovery_attempts'] += 1
                self.recovery_state['last_recovery_time'] = time.time()

                self.logger.info("Recovery completed successfully")

        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            self.recovery_state['recovery_attempts'] += 1

    def _get_dual_brain_checkpoint_state(self, env_id: str) -> Dict[str, Any]:
        """Get dual brain state for checkpointing."""
        if not self.dual_brain:
            return {}

        try:
            insights = self.dual_brain.get_insights()
            return {
                'env_id': env_id,
                'insights': insights,
                'integration_state': self.dual_brain.integration_state.copy(),
                'episode_active': self.dual_brain.episode_active,
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"Failed to get dual brain state: {e}")
            return {'error': str(e)}

    def _notify_critical_error(self, message: str):
        """Notify user of critical errors."""
        self.logger.critical(message)

        # Write to critical error log
        try:
            from pathlib import Path
            critical_log_path = Path("./critical_errors.log")
            with open(critical_log_path, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - Agent {self.agent_id}: {message}\n")
        except Exception:
            pass

    def _calculate_performance_with_errors(self, training_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final performance with error statistics (enhanced version)."""
        performance = {}

        try:
            # Original performance metrics
            if training_stats['episode_rewards']:
                performance.update({
                    'mean_reward': float(np.mean(training_stats['episode_rewards'])),
                    'std_reward': float(np.std(training_stats['episode_rewards'])),
                    'improvement': self._calculate_improvement(training_stats['episode_rewards'])
                })

            if training_stats['episode_lengths']:
                performance['mean_length'] = float(np.mean(training_stats['episode_lengths']))

            # NEW Sprint 5: Error statistics
            error_events = training_stats.get('error_events', [])
            performance['error_statistics'] = {
                'total_errors': len(error_events),
                'error_types': {},
                'consecutive_failures': self.environment_health['consecutive_failures'],
                'degraded_mode_used': self.recovery_state['degraded_mode'],
                'recovery_attempts': self.recovery_state['recovery_attempts']
            }

            # Count error types
            for error_event in error_events:
                error_type = error_event['type']
                performance['error_statistics']['error_types'][error_type] = \
                    performance['error_statistics']['error_types'].get(error_type, 0) + 1

        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            performance = {'error': str(e)}

        return performance

    # NEW Sprint 5: Operational methods

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health and status information."""
        status = {
            'agent_id': self.agent_id,
            'timestamp': time.time(),
            'training_active': self.training_session['active'],
            'environment_health': self.environment_health.copy(),
            'recovery_state': self.recovery_state.copy(),
            'checkpoint_info': None
        }

        if self.checkpoint_manager:
            status['checkpoint_info'] = self.checkpoint_manager.get_checkpoint_info()

        status['system_metrics'] = self._get_system_metrics()
        return status

    def emergency_shutdown(self, reason: str = "user_requested"):
        """Perform emergency shutdown with checkpoint creation."""
        self.logger.warning(f"Emergency shutdown requested: {reason}")

        try:
            self.training_session['active'] = False

            if self.checkpoint_manager:
                success = self.checkpoint_manager.create_manual_checkpoint(f'emergency_{reason}')
                if success:
                    self.logger.info("Emergency checkpoint created")
                else:
                    self.logger.error("Emergency checkpoint failed")

            self._save_profile()

            if self.checkpoint_manager:
                self.checkpoint_manager.shutdown()

            self.logger.info("Emergency shutdown completed")

        except Exception as e:
            self.logger.error(f"Emergency shutdown failed: {e}")

    # Original methods (preserved)

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
        stats = {
            'agent_id': self.agent_id,
            'total_episodes': self.total_episodes,
            'environments_experienced': list(self.environments_experienced),
            'creation_time': self.creation_time.isoformat(),
            'current_environment': self.current_environment,
            'exploration_rate': self.config.exploration_rate,
            'config': self.config.to_dict(),
            'autoencoder_enabled': self.state_normalizer.use_autoencoder,
            'autoencoders_trained': 0
        }

        # Add autoencoder count
        if (self.state_normalizer.autoencoder_trainer and
            hasattr(self.state_normalizer.autoencoder_trainer, 'autoencoders')):
            stats['autoencoders_trained'] = len(self.state_normalizer.autoencoder_trainer.autoencoders)

        return stats

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

    def __del__(self):
        """Enhanced cleanup on object destruction."""
        try:
            if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager:
                self.checkpoint_manager.shutdown()
        except:
            pass