#!/usr/bin/env python3
"""
Test script to verify fixes for the specific failing test cases.
"""

import tempfile
import shutil
import numpy as np
from pathlib import Path


# Test the specific failing scenarios
def test_invalid_continuous_network_validation():
    """Test that invalid continuous network data is handled according to test expectations."""
    print("=== Testing Invalid Continuous Network Validation ===")

    from agent_byte.storage import JsonNumpyStorage

    temp_dir = tempfile.mkdtemp()
    try:
        storage = JsonNumpyStorage(temp_dir)

        # Invalid network state (missing required fields) - should still save per test expectation
        invalid_state = {'algorithm': 'sac'}  # Missing other fields
        success = storage.save_continuous_network_state('agent', 'env', invalid_state)

        print(f"Saved invalid state: {success}")

        if success:
            # Try to load it back
            loaded = storage.load_continuous_network_state('agent', 'env')
            print(f"Loaded back: {loaded is not None}")
            if loaded:
                print(f"Algorithm: {loaded.get('algorithm')}")

        storage.close()
        return success

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)


def test_file_structure():
    """Test that Sprint 9 files are created correctly."""
    print("\n=== Testing File Structure ===")

    from agent_byte.storage import JsonNumpyStorage

    temp_dir = tempfile.mkdtemp()
    try:
        storage = JsonNumpyStorage(temp_dir)

        agent_id = "test_agent"
        env_id = "test_env"

        # Save basic data
        profile = {'agent_id': agent_id}
        success1 = storage.save_agent_profile(agent_id, profile)
        print(f"Profile saved: {success1}")

        brain_state = {'test': 'brain'}
        success2 = storage.save_brain_state(agent_id, env_id, brain_state)
        print(f"Brain state saved: {success2}")

        # Save Sprint 9 data
        network_state = {'algorithm': 'sac', 'state_size': 256}
        success3 = storage.save_continuous_network_state(agent_id, env_id, network_state)
        print(f"Network state saved: {success3}")

        adapter_config = {'adapter_type': 'discrete_to_continuous'}
        success4 = storage.save_action_adapter_config(agent_id, env_id, adapter_config)
        print(f"Adapter config saved: {success4}")

        # Check file structure
        storage_path = Path(temp_dir)
        agent_path = storage_path / "agents" / agent_id
        env_path = agent_path / "environments" / env_id

        print(f"Agent path exists: {agent_path.exists()}")
        print(f"Profile file exists: {(agent_path / 'profile.json').exists()}")
        print(f"Env path exists: {env_path.exists()}")
        print(f"Brain state file exists: {(env_path / 'brain_state.json').exists()}")
        print(f"Network file exists: {(env_path / 'continuous_network.json').exists()}")
        print(f"Adapter file exists: {(env_path / 'action_adapter.json').exists()}")

        storage.close()

        # Return True if all Sprint 9 files exist
        return (env_path / 'continuous_network.json').exists() and (env_path / 'action_adapter.json').exists()

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)


def test_migration_verification():
    """Test that migration verification works with Sprint 9 data."""
    print("\n=== Testing Migration Verification ===")

    from agent_byte.storage import JsonNumpyStorage
    from agent_byte.storage.migrations import StorageMigrator

    temp_dir = tempfile.mkdtemp()
    try:
        # Create source with Sprint 9 data
        source_path = Path(temp_dir) / "source"
        source = JsonNumpyStorage(str(source_path))

        agent_id = "test_agent"

        # Add basic data
        profile = {'agent_id': agent_id, 'test': 'profile'}
        source.save_agent_profile(agent_id, profile)

        brain_state = {'weights': [1, 2, 3]}
        source.save_brain_state(agent_id, 'env1', brain_state)

        # Add Sprint 9 data
        network_state = {'algorithm': 'sac', 'state_size': 256}
        source.save_continuous_network_state(agent_id, 'env1', network_state)

        adapter_config = {'adapter_type': 'discrete_to_continuous'}
        source.save_action_adapter_config(agent_id, 'env1', adapter_config)

        # Add experience vectors
        for i in range(10):
            vector = np.random.randn(256).astype(np.float32)
            metadata = {'index': i}
            source.save_experience_vector(agent_id, vector, metadata)

        # Create target and migrate
        target_path = Path(temp_dir) / "target"
        target = JsonNumpyStorage(str(target_path))

        migrator = StorageMigrator()
        results = migrator.migrate_backend(
            source, target,
            agent_ids=[agent_id],
            create_backup=False
        )

        print(f"Migration results: {results['agents_migrated']} migrated, {results['agents_failed']} failed")
        print(f"Networks migrated: {results['continuous_networks_migrated']}")
        print(f"Adapters migrated: {results['action_adapters_migrated']}")

        # Verify migration
        verify_results = migrator.verify_migration(
            source, target,
            agent_ids=[agent_id]
        )

        print(f"Verification: {verify_results['verified']}")
        agent_check = verify_results['agent_checks'][agent_id]
        print(f"Profile match: {agent_check['profile_match']}")
        print(f"Environments match: {agent_check['environments_match']}")
        print(f"Networks match: {agent_check['continuous_networks_match']}")
        print(f"Adapters match: {agent_check['action_adapters_match']}")

        if verify_results['errors']:
            print(f"Errors: {verify_results['errors']}")

        source.close()
        target.close()

        return verify_results['verified']

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all specific tests."""
    print("🧪 Testing Specific Failing Test Cases")
    print("=" * 50)

    tests = [
        ("Invalid Network Validation", test_invalid_continuous_network_validation),
        ("File Structure", test_file_structure),
        ("Migration Verification", test_migration_verification)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status} - {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ ERROR - {test_name}: {e}")

    print("\n" + "=" * 50)
    print("📊 Summary:")

    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")

    passed = sum(results.values())
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)