"""
Checkpoint manager for Agent Byte robustness and error recovery.

This module handles periodic state snapshots, rollback functionality,
and automatic recovery mechanisms for the Agent Byte system.
"""

import os
import json
import time
import signal
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import shutil

from ..storage.base import StorageBase


class CheckpointManager:
    """
    Manages checkpoints for Agent Byte with automatic recovery capabilities.

    Features:
    - Timer-based checkpointing (every 15 minutes)
    - Episode-based checkpointing (every 500 episodes)
    - Checkpoint-on-exit (signal handlers)
    - Checkpoint rotation (keep 10 recent)
    - Integrity validation
    - Automatic recovery
    """

    def __init__(self,
                 agent_id: str,
                 storage: StorageBase,
                 checkpoint_dir: str = "./checkpoints",
                 timer_interval: int = 900,  # 15 minutes
                 episode_interval: int = 500,
                 max_checkpoints: int = 10):
        """
        Initialize checkpoint manager.

        Args:
            agent_id: Agent identifier
            storage: Storage backend
            checkpoint_dir: Directory for checkpoint files
            timer_interval: Seconds between timer checkpoints (900 = 15 min)
            episode_interval: Episodes between episode checkpoints
            max_checkpoints: Maximum checkpoints to keep
        """
        self.agent_id = agent_id
        self.storage = storage
        self.checkpoint_dir = Path(checkpoint_dir) / agent_id
        self.timer_interval = timer_interval
        self.episode_interval = episode_interval
        self.max_checkpoints = max_checkpoints

        self.logger = logging.getLogger(f"CheckpointManager-{agent_id}")

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.last_timer_checkpoint = time.time()
        self.last_episode_checkpoint = 0
        self.current_episode = 0
        self.checkpointing_enabled = True

        # Timer thread for periodic checkpoints
        self.timer_thread = None
        self.shutdown_requested = False

        # Callbacks for getting agent state
        self.state_providers: Dict[str, Callable] = {}

        # Metrics
        self.metrics = {
            'total_checkpoints': 0,
            'timer_checkpoints': 0,
            'episode_checkpoints': 0,
            'exit_checkpoints': 0,
            'failed_checkpoints': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'corruption_detected': 0
        }

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        # Start timer thread
        self._start_timer_thread()

        self.logger.info(f"CheckpointManager initialized: timer={timer_interval}s, episodes={episode_interval}")

    def register_state_provider(self, component_name: str, provider_func: Callable) -> None:
        """
        Register a function that provides state for a component.

        Args:
            component_name: Name of component (e.g., 'dual_brain', 'config')
            provider_func: Function that returns component state
        """
        self.state_providers[component_name] = provider_func
        self.logger.debug(f"Registered state provider: {component_name}")

    def update_episode_count(self, episode: int) -> None:
        """Update current episode count for episode-based checkpointing."""
        self.current_episode = episode

        # Check if episode-based checkpoint is needed
        if (episode > 0 and
                episode % self.episode_interval == 0 and
                episode > self.last_episode_checkpoint):
            self._create_checkpoint('episode', episode=episode)

    def create_manual_checkpoint(self, reason: str = 'manual') -> bool:
        """
        Create a manual checkpoint.

        Args:
            reason: Reason for checkpoint

        Returns:
            Success status
        """
        return self._create_checkpoint(reason)

    def _create_checkpoint(self, checkpoint_type: str, **metadata) -> bool:
        """
        Create a checkpoint with current agent state.

        Args:
            checkpoint_type: Type of checkpoint ('timer', 'episode', 'exit', 'manual')
            **metadata: Additional metadata

        Returns:
            Success status
        """
        if not self.checkpointing_enabled:
            return False

        try:
            timestamp = datetime.now()
            checkpoint_id = f"{checkpoint_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

            # Collect state from all providers
            checkpoint_data = {
                'checkpoint_id': checkpoint_id,
                'agent_id': self.agent_id,
                'timestamp': timestamp.isoformat(),
                'checkpoint_type': checkpoint_type,
                'episode': self.current_episode,
                'metadata': metadata,
                'components': {}
            }

            # Gather state from registered providers
            for component_name, provider_func in self.state_providers.items():
                try:
                    component_state = provider_func()
                    checkpoint_data['components'][component_name] = component_state
                except Exception as e:
                    self.logger.error(f"Failed to get state from {component_name}: {e}")
                    return False

            # Calculate checksum for integrity verification
            checkpoint_data['checksum'] = self._calculate_checksum(checkpoint_data)

            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            # Update tracking
            if checkpoint_type == 'timer':
                self.last_timer_checkpoint = time.time()
                self.metrics['timer_checkpoints'] += 1
            elif checkpoint_type == 'episode':
                self.last_episode_checkpoint = self.current_episode
                self.metrics['episode_checkpoints'] += 1
            elif checkpoint_type == 'exit':
                self.metrics['exit_checkpoints'] += 1

            self.metrics['total_checkpoints'] += 1

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            self.logger.info(f"Created {checkpoint_type} checkpoint: {checkpoint_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            self.metrics['failed_checkpoints'] += 1
            return False

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent valid checkpoint.

        Returns:
            Checkpoint data or None if no valid checkpoint found
        """
        checkpoints = self._list_checkpoints()

        if not checkpoints:
            self.logger.info("No checkpoints found")
            return None

        # Try checkpoints from newest to oldest
        for checkpoint_file in checkpoints:
            try:
                checkpoint_data = self._load_and_validate_checkpoint(checkpoint_file)
                if checkpoint_data:
                    self.logger.info(f"Loaded checkpoint: {checkpoint_data['checkpoint_id']}")
                    self.metrics['successful_recoveries'] += 1
                    return checkpoint_data

            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
                continue

        self.logger.error("No valid checkpoints found for recovery")
        self.metrics['failed_recoveries'] += 1
        return None

    def _load_and_validate_checkpoint(self, checkpoint_file: Path) -> Optional[Dict[str, Any]]:
        """
        Load and validate a specific checkpoint file.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            Validated checkpoint data or None if invalid
        """
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Validate structure
            required_fields = ['checkpoint_id', 'agent_id', 'timestamp', 'components', 'checksum']
            for field in required_fields:
                if field not in checkpoint_data:
                    self.logger.warning(f"Checkpoint {checkpoint_file} missing field: {field}")
                    return None

            # Validate agent ID
            if checkpoint_data['agent_id'] != self.agent_id:
                self.logger.warning(f"Checkpoint {checkpoint_file} for different agent")
                return None

            # Validate checksum
            stored_checksum = checkpoint_data.pop('checksum')
            calculated_checksum = self._calculate_checksum(checkpoint_data)

            if stored_checksum != calculated_checksum:
                self.logger.error(f"Checkpoint {checkpoint_file} failed integrity check")
                self.metrics['corruption_detected'] += 1
                return None

            # Restore checksum to data
            checkpoint_data['checksum'] = stored_checksum

            return checkpoint_data

        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_file}: {e}")
            return None

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate SHA256 checksum of checkpoint data."""
        # Create a copy without checksum field
        data_copy = {k: v for k, v in data.items() if k != 'checksum'}

        # Convert to JSON string for consistent hashing
        json_str = json.dumps(data_copy, sort_keys=True, default=str)

        return hashlib.sha256(json_str.encode()).hexdigest()

    def _list_checkpoints(self) -> List[Path]:
        """List all checkpoint files sorted by modification time (newest first)."""
        checkpoint_files = list(self.checkpoint_dir.glob("*.json"))

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return checkpoint_files

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond the retention limit."""
        checkpoints = self._list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            to_remove = checkpoints[self.max_checkpoints:]

            for checkpoint_file in to_remove:
                try:
                    checkpoint_file.unlink()
                    self.logger.debug(f"Removed old checkpoint: {checkpoint_file.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")

    def _timer_thread_worker(self) -> None:
        """Worker function for timer-based checkpointing thread."""
        while not self.shutdown_requested:
            try:
                # Sleep in small intervals to allow for clean shutdown
                for _ in range(self.timer_interval):
                    if self.shutdown_requested:
                        break
                    time.sleep(1)

                if not self.shutdown_requested and self.checkpointing_enabled:
                    elapsed = time.time() - self.last_timer_checkpoint
                    if elapsed >= self.timer_interval:
                        self._create_checkpoint('timer')

            except Exception as e:
                self.logger.error(f"Timer checkpoint thread error: {e}")
                time.sleep(60)  # Back off on error

    def _start_timer_thread(self) -> None:
        """Start the timer-based checkpointing thread."""
        if self.timer_thread is None or not self.timer_thread.is_alive():
            self.timer_thread = threading.Thread(
                target=self._timer_thread_worker,
                name=f"CheckpointTimer-{self.agent_id}",
                daemon=True
            )
            self.timer_thread.start()
            self.logger.debug("Started timer checkpoint thread")

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, creating exit checkpoint...")
            self._create_checkpoint('exit', signal=signum)
            self.shutdown()

        try:
            signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Termination
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)  # Hangup (Unix)
        except Exception as e:
            self.logger.warning(f"Could not register signal handlers: {e}")

    def enable_checkpointing(self) -> None:
        """Enable automatic checkpointing."""
        self.checkpointing_enabled = True
        self.logger.info("Checkpointing enabled")

    def disable_checkpointing(self) -> None:
        """Disable automatic checkpointing."""
        self.checkpointing_enabled = False
        self.logger.info("Checkpointing disabled")

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints."""
        checkpoints = self._list_checkpoints()

        info = {
            'total_checkpoints': len(checkpoints),
            'latest_checkpoint': None,
            'checkpoint_sizes': [],
            'metrics': self.metrics.copy()
        }

        if checkpoints:
            latest = checkpoints[0]
            stat = latest.stat()

            info['latest_checkpoint'] = {
                'filename': latest.name,
                'size_bytes': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'age_seconds': time.time() - stat.st_mtime
            }

            # Calculate total checkpoint storage usage
            total_size = sum(cp.stat().st_size for cp in checkpoints)
            info['total_size_bytes'] = total_size
            info['average_size_bytes'] = total_size / len(checkpoints)

        return info

    def rollback_to_checkpoint(self, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Rollback to a specific checkpoint or the latest one.

        Args:
            checkpoint_id: Specific checkpoint ID, or None for latest

        Returns:
            Checkpoint data for rollback or None if failed
        """
        try:
            if checkpoint_id:
                # Find specific checkpoint
                checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
                if not checkpoint_file.exists():
                    self.logger.error(f"Checkpoint {checkpoint_id} not found")
                    return None

                checkpoint_data = self._load_and_validate_checkpoint(checkpoint_file)
            else:
                # Load latest checkpoint
                checkpoint_data = self.load_latest_checkpoint()

            if checkpoint_data:
                self.logger.info(f"Rolling back to checkpoint: {checkpoint_data['checkpoint_id']}")

                # Update episode tracking
                self.current_episode = checkpoint_data.get('episode', 0)
                self.last_episode_checkpoint = max(0, self.current_episode - self.episode_interval)

                return checkpoint_data
            else:
                self.logger.error("Failed to load checkpoint for rollback")
                return None

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return None

    def shutdown(self) -> None:
        """Shutdown checkpoint manager gracefully."""
        self.logger.info("Shutting down checkpoint manager...")

        # Stop timer thread
        self.shutdown_requested = True

        if self.timer_thread and self.timer_thread.is_alive():
            self.timer_thread.join(timeout=5)

        # Create final checkpoint
        if self.checkpointing_enabled:
            self._create_checkpoint('exit', reason='shutdown')

        self.logger.info("Checkpoint manager shutdown complete")

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup