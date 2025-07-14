"""
Recovery utilities for Agent Byte storage systems.

This module provides comprehensive tools for detecting, diagnosing, and repairing
storage corruption, as well as utilities for data migration and backup management.
Sprint 5 Phase 3 - Storage Failure Recovery completion.
"""

import json
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import time

from .base import StorageBase
from .json_numpy_storage import JsonNumpyStorage
from .vector_db_storage import VectorDBStorage


class StorageRecoveryManager:
    """
    Comprehensive storage recovery and maintenance manager.

    Provides tools for:
    - Corruption detection and repair
    - Data validation and integrity checking
    - Backup management and restoration
    - Performance optimization
    - Health monitoring and alerting
    """

    def __init__(self, storage: StorageBase, config: Optional[Dict[str, Any]] = None):
        """
        Initialize recovery manager.

        Args:
            storage: Storage backend to manage
            config: Recovery configuration options
        """
        self.storage = storage
        self.config = config or {}
        self.logger = logging.getLogger(f"RecoveryManager-{storage.__class__.__name__}")

        # Recovery configuration
        self.max_repair_attempts = self.config.get('max_repair_attempts', 3)
        self.backup_retention_days = self.config.get('backup_retention_days', 30)
        self.health_check_interval = self.config.get('health_check_interval', 3600)  # 1 hour
        self.auto_repair_enabled = self.config.get('auto_repair_enabled', True)

        # Recovery state
        self.last_health_check = 0
        self.recovery_history = []
        self.critical_errors = []

        # Metrics
        self.recovery_metrics = {
            'total_repairs': 0,
            'successful_repairs': 0,
            'failed_repairs': 0,
            'corrupted_files_detected': 0,
            'backups_restored': 0,
            'data_loss_incidents': 0
        }

        self.logger.info("Storage recovery manager initialized")

    def comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of storage system.

        Returns:
            Detailed health report with issues and recommendations
        """
        self.logger.info("Starting comprehensive health check")
        start_time = time.time()

        report = {
            'timestamp': start_time,
            'storage_type': self.storage.__class__.__name__,
            'overall_health': 'healthy',
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'metrics': {},
            'validation_results': {},
            'performance_analysis': {}
        }

        try:
            # 1. Storage-specific validation
            if isinstance(self.storage, JsonNumpyStorage):
                report['validation_results'] = self._validate_json_storage()
            elif isinstance(self.storage, VectorDBStorage):
                report['validation_results'] = self._validate_vector_storage()

            # 2. File system health
            report['filesystem_health'] = self._check_filesystem_health()

            # 3. Performance analysis
            report['performance_analysis'] = self._analyze_performance()

            # 4. Backup status
            report['backup_status'] = self._check_backup_status()

            # 5. Security analysis
            report['security_analysis'] = self._check_security_issues()

            # 6. Determine overall health
            report['overall_health'] = self._determine_overall_health(report)

            # 7. Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)

            report['check_duration'] = time.time() - start_time
            self.last_health_check = start_time

            self.logger.info(f"Health check completed in {report['check_duration']:.2f}s: {report['overall_health']}")
            return report

        except Exception as e:
            report['overall_health'] = 'error'
            report['critical_issues'].append(f"Health check failed: {e}")
            self.logger.error(f"Health check failed: {e}")
            return report

    def _validate_json_storage(self) -> Dict[str, Any]:
        """Validate JSON storage structure and integrity."""
        results = {
            'files_checked': 0,
            'corrupted_files': 0,
            'missing_files': 0,
            'structural_issues': 0,
            'checksum_mismatches': 0,
            'issues': []
        }

        try:
            # Get storage base path
            if hasattr(self.storage, 'base_path'):
                base_path = Path(self.storage.base_path)
            else:
                return results

            # Check all JSON files
            json_files = list(base_path.rglob("*.json"))

            for json_file in json_files:
                results['files_checked'] += 1

                try:
                    # Check file readability
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    # Validate structure based on file type
                    if json_file.name == 'profile.json':
                        self._validate_profile_structure(data, json_file, results)
                    elif json_file.name == 'brain_state.json':
                        self._validate_brain_state_structure(data, json_file, results)
                    elif json_file.name == 'knowledge.json':
                        self._validate_knowledge_structure(data, json_file, results)

                    # Check checksum if available
                    if hasattr(self.storage, '_verify_file_integrity'):
                        if not self.storage._verify_file_integrity(json_file):
                            results['checksum_mismatches'] += 1
                            results['issues'].append(f"Checksum mismatch: {json_file}")

                except json.JSONDecodeError as e:
                    results['corrupted_files'] += 1
                    results['issues'].append(f"JSON corruption in {json_file}: {e}")
                except FileNotFoundError:
                    results['missing_files'] += 1
                    results['issues'].append(f"Missing file: {json_file}")
                except Exception as e:
                    results['structural_issues'] += 1
                    results['issues'].append(f"Validation error in {json_file}: {e}")

            return results

        except Exception as e:
            results['issues'].append(f"JSON storage validation failed: {e}")
            return results

    def _validate_profile_structure(self, data: Dict[str, Any], file_path: Path, results: Dict[str, Any]):
        """Validate agent profile structure."""
        required_fields = ['agent_id', 'creation_time', 'total_episodes', 'environments_experienced']

        for field in required_fields:
            if field not in data:
                results['structural_issues'] += 1
                results['issues'].append(f"Missing profile field '{field}' in {file_path}")

    def _validate_brain_state_structure(self, data: Dict[str, Any], file_path: Path, results: Dict[str, Any]):
        """Validate brain state structure."""
        if 'neural_brain' not in data and 'symbolic_brain' not in data:
            results['structural_issues'] += 1
            results['issues'].append(f"Brain state missing neural/symbolic data in {file_path}")

    def _validate_knowledge_structure(self, data: Dict[str, Any], file_path: Path, results: Dict[str, Any]):
        """Validate knowledge structure."""
        if 'skills' not in data and 'patterns' not in data:
            results['structural_issues'] += 1
            results['issues'].append(f"Knowledge missing skills/patterns data in {file_path}")

    def _validate_vector_storage(self) -> Dict[str, Any]:
        """Validate vector storage integrity."""
        if hasattr(self.storage, 'validate_vector_storage'):
            return self.storage.validate_vector_storage()
        else:
            return {'validation_skipped': 'Vector storage validation not available'}

    def _check_filesystem_health(self) -> Dict[str, Any]:
        """Check filesystem health and capacity."""
        health = {
            'disk_usage': {},
            'permissions_ok': True,
            'free_space_gb': 0,
            'issues': []
        }

        try:
            if hasattr(self.storage, 'base_path'):
                base_path = Path(self.storage.base_path)

                # Check disk usage
                import shutil
                total, used, free = shutil.disk_usage(base_path)

                health['disk_usage'] = {
                    'total_gb': total / (1024 ** 3),
                    'used_gb': used / (1024 ** 3),
                    'free_gb': free / (1024 ** 3),
                    'usage_percent': (used / total) * 100
                }

                health['free_space_gb'] = health['disk_usage']['free_gb']

                # Check if low on space
                if health['disk_usage']['usage_percent'] > 95:
                    health['issues'].append("Critical: Less than 5% disk space remaining")
                elif health['disk_usage']['usage_percent'] > 90:
                    health['issues'].append("Warning: Less than 10% disk space remaining")

                # Check write permissions
                test_file = base_path / ".write_test"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                except Exception:
                    health['permissions_ok'] = False
                    health['issues'].append("No write permissions to storage directory")

            return health

        except Exception as e:
            health['issues'].append(f"Filesystem check failed: {e}")
            return health

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze storage performance metrics."""
        analysis = {
            'avg_operation_time_ms': 0,
            'slow_operations_detected': False,
            'performance_trend': 'stable',
            'bottlenecks': [],
            'metrics_available': False
        }

        try:
            # Check if storage has performance metrics
            if hasattr(self.storage, 'get_storage_health'):
                health = self.storage.get_storage_health()
                analysis['metrics_available'] = True

                # Analyze search performance for vector storage
                if 'avg_search_time_ms' in health:
                    analysis['avg_operation_time_ms'] = health['avg_search_time_ms']

                    if health['avg_search_time_ms'] > 1000:
                        analysis['slow_operations_detected'] = True
                        analysis['bottlenecks'].append("Slow search operations detected")

                # Check error rates
                if 'error_metrics' in health:
                    error_metrics = health['error_metrics']
                    total_errors = sum(error_metrics.values())

                    if total_errors > 10:
                        analysis['bottlenecks'].append("High error rate detected")

            return analysis

        except Exception as e:
            analysis['bottlenecks'].append(f"Performance analysis failed: {e}")
            return analysis

    def _check_backup_status(self) -> Dict[str, Any]:
        """Check backup availability and freshness."""
        status = {
            'backup_files_found': 0,
            'newest_backup_age_hours': None,
            'oldest_backup_age_hours': None,
            'backup_coverage': 'unknown',
            'issues': []
        }

        try:
            if hasattr(self.storage, 'base_path'):
                base_path = Path(self.storage.base_path)

                # Find backup files
                backup_files = list(base_path.rglob("*.backup_*"))
                status['backup_files_found'] = len(backup_files)

                if backup_files:
                    # Calculate backup ages
                    current_time = time.time()
                    backup_ages = [(current_time - f.stat().st_mtime) / 3600 for f in backup_files]

                    status['newest_backup_age_hours'] = min(backup_ages)
                    status['oldest_backup_age_hours'] = max(backup_ages)

                    # Check if backups are recent
                    if status['newest_backup_age_hours'] > 24:
                        status['issues'].append("No recent backups found (older than 24 hours)")

                    # Estimate coverage
                    critical_files = len(list(base_path.rglob("*.json")))
                    if len(backup_files) >= critical_files * 0.5:
                        status['backup_coverage'] = 'good'
                    elif len(backup_files) >= critical_files * 0.2:
                        status['backup_coverage'] = 'partial'
                    else:
                        status['backup_coverage'] = 'poor'
                        status['issues'].append("Insufficient backup coverage")
                else:
                    status['backup_coverage'] = 'none'
                    status['issues'].append("No backup files found")

            return status

        except Exception as e:
            status['issues'].append(f"Backup status check failed: {e}")
            return status

    def _check_security_issues(self) -> Dict[str, Any]:
        """Check for security-related storage issues."""
        security = {
            'file_permissions_secure': True,
            'sensitive_data_exposed': False,
            'access_control_issues': [],
            'recommendations': []
        }

        try:
            if hasattr(self.storage, 'base_path'):
                base_path = Path(self.storage.base_path)

                # Check file permissions
                for file_path in base_path.rglob("*"):
                    if file_path.is_file():
                        mode = file_path.stat().st_mode

                        # Check if world-readable (basic check)
                        if mode & 0o004:
                            security['file_permissions_secure'] = False
                            security['access_control_issues'].append(f"World-readable file: {file_path}")

                # Check for sensitive files in wrong locations
                for json_file in base_path.rglob("*.json"):
                    if json_file.name == 'profile.json':
                        # Check if profile contains sensitive information
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)

                            # Look for potentially sensitive keys
                            sensitive_keys = ['password', 'token', 'key', 'secret']
                            for key in data.keys():
                                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                                    security['sensitive_data_exposed'] = True
                                    security['access_control_issues'].append(f"Sensitive data in profile: {key}")
                        except:
                            pass

            # Generate security recommendations
            if not security['file_permissions_secure']:
                security['recommendations'].append("Review and restrict file permissions")

            if security['sensitive_data_exposed']:
                security['recommendations'].append("Encrypt or remove sensitive data from storage")

            return security

        except Exception as e:
            security['access_control_issues'].append(f"Security check failed: {e}")
            return security

    def _determine_overall_health(self, report: Dict[str, Any]) -> str:
        """Determine overall health status from report components."""
        critical_issues = report.get('critical_issues', [])
        validation_results = report.get('validation_results', {})
        filesystem_health = report.get('filesystem_health', {})

        # Check for critical issues
        if critical_issues:
            return 'critical'

        # Check validation results
        if validation_results.get('corrupted_files', 0) > 0:
            return 'degraded'

        # Check filesystem health
        fs_issues = filesystem_health.get('issues', [])
        if any('Critical' in issue for issue in fs_issues):
            return 'critical'
        elif any('Warning' in issue for issue in fs_issues):
            return 'warning'

        # Check validation warnings
        if (validation_results.get('checksum_mismatches', 0) > 0 or
                validation_results.get('structural_issues', 0) > 0):
            return 'warning'

        return 'healthy'

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on health report."""
        recommendations = []

        # Critical issues
        if report.get('overall_health') == 'critical':
            recommendations.append("URGENT: Run emergency_repair() immediately")

        # Corruption issues
        validation = report.get('validation_results', {})
        if validation.get('corrupted_files', 0) > 0:
            recommendations.append("Run repair_corruption() to fix corrupted files")

        # Performance issues
        performance = report.get('performance_analysis', {})
        if performance.get('slow_operations_detected'):
            recommendations.append("Consider storage optimization or migration to faster backend")

        # Backup issues
        backup_status = report.get('backup_status', {})
        if backup_status.get('backup_coverage') in ['poor', 'none']:
            recommendations.append("Implement regular backup strategy")

        # Disk space issues
        filesystem = report.get('filesystem_health', {})
        disk_usage = filesystem.get('disk_usage', {})
        if disk_usage.get('usage_percent', 0) > 90:
            recommendations.append("Free up disk space or add storage capacity")

        # Security issues
        security = report.get('security_analysis', {})
        recommendations.extend(security.get('recommendations', []))

        return recommendations

    def emergency_repair(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform emergency repair for critical storage issues.

        Args:
            agent_id: Specific agent to repair, or None for all agents

        Returns:
            Repair results with detailed actions taken
        """
        self.logger.warning("Starting emergency repair procedure")
        start_time = time.time()

        repair_results = {
            'timestamp': start_time,
            'agent_id': agent_id,
            'repair_type': 'emergency',
            'actions_taken': [],
            'files_repaired': 0,
            'data_recovered': 0,
            'data_lost': 0,
            'success': False,
            'errors': []
        }

        try:
            # 1. Create emergency backup
            backup_result = self._create_emergency_backup(agent_id)
            repair_results['actions_taken'].append(f"Emergency backup: {backup_result}")

            # 2. Repair storage-specific issues
            if isinstance(self.storage, JsonNumpyStorage):
                json_repair = self._emergency_repair_json_storage(agent_id)
                repair_results.update(json_repair)
            elif isinstance(self.storage, VectorDBStorage):
                vector_repair = self._emergency_repair_vector_storage(agent_id)
                repair_results.update(vector_repair)

            # 3. Validate repairs
            validation_result = self._validate_emergency_repairs(agent_id)
            repair_results['validation'] = validation_result

            repair_results['success'] = validation_result.get('all_critical_issues_resolved', False)
            repair_results['repair_duration'] = time.time() - start_time

            # Update metrics
            self.recovery_metrics['total_repairs'] += 1
            if repair_results['success']:
                self.recovery_metrics['successful_repairs'] += 1
            else:
                self.recovery_metrics['failed_repairs'] += 1

            self.logger.info(
                f"Emergency repair completed in {repair_results['repair_duration']:.2f}s: {repair_results['success']}")
            return repair_results

        except Exception as e:
            repair_results['errors'].append(f"Emergency repair failed: {e}")
            repair_results['success'] = False
            self.logger.error(f"Emergency repair failed: {e}")
            return repair_results

    def _create_emergency_backup(self, agent_id: Optional[str] = None) -> str:
        """Create emergency backup before repair operations."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if hasattr(self.storage, 'base_path'):
                base_path = Path(self.storage.base_path)
                backup_dir = base_path.parent / f"emergency_backup_{timestamp}"
                backup_dir.mkdir(exist_ok=True)

                if agent_id:
                    # Backup specific agent
                    agent_path = base_path / agent_id
                    if agent_path.exists():
                        shutil.copytree(agent_path, backup_dir / agent_id)
                else:
                    # Backup everything
                    for item in base_path.iterdir():
                        if item.is_dir():
                            shutil.copytree(item, backup_dir / item.name)
                        else:
                            shutil.copy2(item, backup_dir / item.name)

                return f"Created at {backup_dir}"
            else:
                return "Backup not supported for this storage type"

        except Exception as e:
            return f"Backup failed: {e}"

    def _emergency_repair_json_storage(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Emergency repair for JSON storage issues."""
        results = {
            'json_files_repaired': 0,
            'checksum_fixes': 0,
            'structure_fixes': 0
        }

        try:
            # Use storage's built-in repair if available
            if hasattr(self.storage, 'repair_corruption'):
                repair_result = self.storage.repair_corruption(agent_id)
                results.update(repair_result)

            return results

        except Exception as e:
            results['errors'] = [f"JSON storage repair failed: {e}"]
            return results

    def _emergency_repair_vector_storage(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Emergency repair for vector storage issues."""
        results = {
            'indexes_repaired': 0,
            'vectors_recovered': 0,
            'metadata_fixed': 0
        }

        try:
            # Use storage's built-in repair if available
            if hasattr(self.storage, 'repair_vector_storage'):
                repair_result = self.storage.repair_vector_storage()
                results.update(repair_result)

            return results

        except Exception as e:
            results['errors'] = [f"Vector storage repair failed: {e}"]
            return results

    def _validate_emergency_repairs(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate that emergency repairs were successful."""
        try:
            # Re-run health check to verify repairs
            health_report = self.comprehensive_health_check()

            validation = {
                'overall_health_after_repair': health_report['overall_health'],
                'critical_issues_remaining': len(health_report['critical_issues']),
                'warnings_remaining': len(health_report['warnings']),
                'all_critical_issues_resolved': len(health_report['critical_issues']) == 0
            }

            return validation

        except Exception as e:
            return {
                'validation_error': f"Repair validation failed: {e}",
                'all_critical_issues_resolved': False
            }

    def scheduled_maintenance(self) -> Dict[str, Any]:
        """
        Perform scheduled maintenance tasks.

        Returns:
            Maintenance results and recommendations
        """
        self.logger.info("Starting scheduled maintenance")
        start_time = time.time()

        maintenance_results = {
            'timestamp': start_time,
            'tasks_completed': [],
            'tasks_failed': [],
            'optimizations_applied': [],
            'cleanup_results': {},
            'recommendations': []
        }

        try:
            # 1. Health check
            health_report = self.comprehensive_health_check()
            maintenance_results['health_status'] = health_report['overall_health']

            # 2. Cleanup old files
            cleanup_result = self._cleanup_old_files()
            maintenance_results['cleanup_results'] = cleanup_result
            maintenance_results['tasks_completed'].append('file_cleanup')

            # 3. Optimize storage
            optimization_result = self._optimize_storage()
            maintenance_results['optimizations_applied'] = optimization_result
            maintenance_results['tasks_completed'].append('storage_optimization')

            # 4. Update checksums
            if hasattr(self.storage, '_save_checksums'):
                self.storage._save_checksums()
                maintenance_results['tasks_completed'].append('checksum_update')

            # 5. Generate maintenance recommendations
            maintenance_results['recommendations'] = self._generate_maintenance_recommendations(health_report)

            maintenance_results['maintenance_duration'] = time.time() - start_time
            self.logger.info(f"Scheduled maintenance completed in {maintenance_results['maintenance_duration']:.2f}s")

            return maintenance_results

        except Exception as e:
            maintenance_results['tasks_failed'].append(f"Maintenance failed: {e}")
            self.logger.error(f"Scheduled maintenance failed: {e}")
            return maintenance_results

    def _cleanup_old_files(self) -> Dict[str, Any]:
        """Clean up old backup and temporary files."""
        cleanup = {
            'backups_removed': 0,
            'temp_files_removed': 0,
            'space_freed_mb': 0,
            'errors': []
        }

        try:
            if hasattr(self.storage, 'base_path'):
                base_path = Path(self.storage.base_path)

                # Clean old backups
                cutoff_date = datetime.now() - timedelta(days=self.backup_retention_days)
                cutoff_timestamp = cutoff_date.timestamp()

                for backup_file in base_path.rglob("*.backup_*"):
                    try:
                        if backup_file.stat().st_mtime < cutoff_timestamp:
                            size_mb = backup_file.stat().st_size / (1024 * 1024)
                            backup_file.unlink()
                            cleanup['backups_removed'] += 1
                            cleanup['space_freed_mb'] += size_mb
                    except Exception as e:
                        cleanup['errors'].append(f"Failed to remove backup {backup_file}: {e}")

                # Clean temporary files
                for temp_file in base_path.rglob("*.tmp"):
                    try:
                        size_mb = temp_file.stat().st_size / (1024 * 1024)
                        temp_file.unlink()
                        cleanup['temp_files_removed'] += 1
                        cleanup['space_freed_mb'] += size_mb
                    except Exception as e:
                        cleanup['errors'].append(f"Failed to remove temp file {temp_file}: {e}")

            return cleanup

        except Exception as e:
            cleanup['errors'].append(f"Cleanup failed: {e}")
            return cleanup

    def _optimize_storage(self) -> List[str]:
        """Apply storage optimizations."""
        optimizations = []

        try:
            # Storage-specific optimizations
            if hasattr(self.storage, '_optimize_vector_storage'):
                self.storage._optimize_vector_storage()
                optimizations.append('vector_storage_optimized')

            # Generic optimizations
            if hasattr(self.storage, '_cache') and self.storage._cache:
                self.storage._clear_cache()
                optimizations.append('cache_cleared')

            return optimizations

        except Exception as e:
            optimizations.append(f"Optimization failed: {e}")
            return optimizations

    def _generate_maintenance_recommendations(self, health_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on maintenance results."""
        recommendations = []

        # Copy recommendations from health report
        recommendations.extend(health_report.get('recommendations', []))

        # Add maintenance-specific recommendations
        performance = health_report.get('performance_analysis', {})
        if performance.get('slow_operations_detected'):
            recommendations.append("Schedule more frequent maintenance to improve performance")

        backup_status = health_report.get('backup_status', {})
        if backup_status.get('backup_files_found', 0) < 5:
            recommendations.append("Increase backup frequency")

        return list(set(recommendations))  # Remove duplicates

    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get comprehensive recovery and maintenance metrics."""
        return {
            'recovery_metrics': self.recovery_metrics.copy(),
            'last_health_check': self.last_health_check,
            'critical_errors_count': len(self.critical_errors),
            'recovery_history_count': len(self.recovery_history),
            'auto_repair_enabled': self.auto_repair_enabled,
            'config': {
                'max_repair_attempts': self.max_repair_attempts,
                'backup_retention_days': self.backup_retention_days,
                'health_check_interval': self.health_check_interval
            }
        }