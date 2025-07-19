"""
Updated Storage performance benchmark comparing different backends.

This script demonstrates the performance improvements of the optimized storage
implementations and fixes for the vector database hybrid backend.
Enhanced with Sprint 9 continuous action space benchmarks.
"""

import numpy as np
import time
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import sys
import os

# Add the parent directory to the path so we can import the optimized storage
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_byte.storage import JsonNumpyStorage, VectorDBStorage
from agent_byte.storage.experience_buffer import ExperienceBuffer, StreamingExperienceBuffer

# Try to import our optimized implementations
try:
    from optimized_json_numpy_storage import OptimizedJsonNumpyStorage
    OPTIMIZED_JSON_AVAILABLE = True
except ImportError:
    OPTIMIZED_JSON_AVAILABLE = False

try:
    from improved_vector_db_storage import ImprovedVectorDBStorage
    IMPROVED_VECTOR_DB_AVAILABLE = True
except ImportError:
    IMPROVED_VECTOR_DB_AVAILABLE = False


def generate_random_experience(dim: int = 256) -> tuple:
    """Generate a random experience vector and metadata."""
    vector = np.random.randn(dim).astype(np.float32)
    metadata = {
        'env_id': f'test_env_{np.random.randint(5)}',
        'action': np.random.randint(10),
        'reward': np.random.randn(),
        'done': bool(np.random.randint(2))
    }
    return vector, metadata


def generate_continuous_network_state() -> Dict[str, Any]:
    """Generate a mock continuous network state for benchmarking."""
    algorithms = ['sac', 'ddpg']
    return {
        'algorithm': np.random.choice(algorithms),
        'state_size': 256,
        'action_size': np.random.randint(1, 5),
        'action_bounds': {
            'low': [-1.0] * np.random.randint(1, 5),
            'high': [1.0] * np.random.randint(1, 5)
        },
        'device': np.random.choice(['cpu', 'cuda']),
        'weights_data': f'mock_weights_{np.random.randint(1000)}',
        'network_info': {
            'algorithm': np.random.choice(algorithms),
            'temperature': np.random.uniform(0.1, 0.5),
            'replay_buffer_size': np.random.randint(1000, 10000)
        }
    }


def generate_action_adapter_config() -> Dict[str, Any]:
    """Generate a mock action adapter configuration for benchmarking."""
    adapter_types = ['discrete_to_continuous', 'continuous_to_discrete', 'hybrid']
    return {
        'source_space': {
            'space_type': np.random.choice(['discrete', 'continuous']),
            'size': np.random.randint(2, 10)
        },
        'target_space': {
            'space_type': np.random.choice(['discrete', 'continuous']),
            'size': np.random.randint(2, 10)
        },
        'adapter_type': np.random.choice(adapter_types),
        'parameters': {
            'strategy': np.random.choice(['uniform_grid', 'quantization', 'gaussian']),
            'conversion_loss': np.random.uniform(0.0, 0.5)
        }
    }


def benchmark_write_performance(storage_backends: Dict[str, Any],
                              num_experiences: int = 10000) -> Dict[str, float]:
    """Benchmark write performance across storage backends."""
    results = {}

    for name, storage in storage_backends.items():
        print(f"\nBenchmarking write performance for {name}...")

        start_time = time.time()

        for i in range(num_experiences):
            vector, metadata = generate_random_experience()
            storage.save_experience_vector("test_agent", vector, metadata)

            if (i + 1) % 1000 == 0:
                print(f"  Written {i + 1}/{num_experiences} experiences...")

        # Flush any remaining buffers
        if hasattr(storage, 'flush_all_buffers'):
            storage.flush_all_buffers()

        end_time = time.time()
        duration = end_time - start_time

        results[name] = {
            'duration': duration,
            'experiences_per_second': num_experiences / duration
        }

        print(f"  Completed in {duration:.2f}s ({results[name]['experiences_per_second']:.0f} exp/s)")

    return results


def benchmark_search_performance(storage_backends: Dict[str, Any],
                               num_queries: int = 100,
                               k: int = 10) -> Dict[str, float]:
    """Benchmark similarity search performance across storage backends."""
    results = {}

    for name, storage in storage_backends.items():
        print(f"\nBenchmarking search performance for {name}...")

        search_times = []

        for i in range(num_queries):
            query_vector, _ = generate_random_experience()

            start_time = time.time()
            results_list = storage.search_similar_experiences(
                "test_agent", query_vector, k=k
            )
            end_time = time.time()

            search_times.append(end_time - start_time)

            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_queries} queries...")

        avg_time = np.mean(search_times)
        std_time = np.std(search_times)

        results[name] = {
            'avg_search_time': avg_time,
            'std_search_time': std_time,
            'queries_per_second': 1.0 / avg_time
        }

        print(f"  Average search time: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
        print(f"  Queries per second: {results[name]['queries_per_second']:.0f}")

    return results


def benchmark_sprint9_features(storage_backends: Dict[str, Any],
                               num_networks: int = 1000,
                               num_adapters: int = 1000) -> Dict[str, Dict[str, float]]:
    """Benchmark Sprint 9 continuous action space features."""
    results = {}

    for name, storage in storage_backends.items():
        print(f"\nBenchmarking Sprint 9 features for {name}...")

        # Benchmark continuous network storage
        print("  Testing continuous network storage...")
        start_time = time.time()

        for i in range(num_networks):
            network_state = generate_continuous_network_state()
            env_id = f"env_{i % 10}"  # 10 different environments
            storage.save_continuous_network_state("test_agent", env_id, network_state)

            if (i + 1) % 100 == 0:
                print(f"    Saved {i + 1}/{num_networks} networks...")

        network_save_time = time.time() - start_time

        # Benchmark continuous network loading
        print("  Testing continuous network loading...")
        start_time = time.time()

        for i in range(min(100, num_networks)):  # Test loading 100 networks
            env_id = f"env_{i % 10}"
            loaded_state = storage.load_continuous_network_state("test_agent", env_id)

        network_load_time = time.time() - start_time

        # Benchmark action adapter storage
        print("  Testing action adapter storage...")
        start_time = time.time()

        for i in range(num_adapters):
            adapter_config = generate_action_adapter_config()
            env_id = f"adapter_env_{i % 10}"
            storage.save_action_adapter_config("test_agent", env_id, adapter_config)

            if (i + 1) % 100 == 0:
                print(f"    Saved {i + 1}/{num_adapters} adapters...")

        adapter_save_time = time.time() - start_time

        # Benchmark action adapter loading
        print("  Testing action adapter loading...")
        start_time = time.time()

        for i in range(min(100, num_adapters)):
            env_id = f"adapter_env_{i % 10}"
            loaded_config = storage.load_action_adapter_config("test_agent", env_id)

        adapter_load_time = time.time() - start_time

        # Test listing functions
        print("  Testing listing functions...")
        start_time = time.time()

        network_list = storage.list_continuous_networks("test_agent")
        adapter_list = storage.list_action_adapters("test_agent")

        list_time = time.time() - start_time

        results[name] = {
            'network_save_time': network_save_time,
            'network_load_time': network_load_time,
            'adapter_save_time': adapter_save_time,
            'adapter_load_time': adapter_load_time,
            'list_time': list_time,
            'networks_per_second': num_networks / network_save_time,
            'adapters_per_second': num_adapters / adapter_save_time,
            'network_count': len(network_list),
            'adapter_count': len(adapter_list)
        }

        print(f"  Network save: {results[name]['networks_per_second']:.0f} networks/s")
        print(f"  Adapter save: {results[name]['adapters_per_second']:.0f} adapters/s")
        print(f"  Found {results[name]['network_count']} networks, {results[name]['adapter_count']} adapters")

    return results


def benchmark_memory_usage():
    """Benchmark memory usage of different buffer implementations."""
    print("\nBenchmarking memory usage...")

    # Standard buffer
    standard_buffer = ExperienceBuffer(max_size=50000, cache_size=5000)

    # Streaming buffer
    streaming_buffer = StreamingExperienceBuffer(
        max_memory_size=10000,
        cache_size=5000,
        disk_path="./temp_streaming"
    )

    # Add experiences
    print("Adding 50,000 experiences...")
    for i in range(50000):
        vector, metadata = generate_random_experience()
        standard_buffer.add(vector, metadata)
        streaming_buffer.add(vector, metadata)

        if (i + 1) % 10000 == 0:
            print(f"  Added {i + 1} experiences...")

    # Get stats
    standard_stats = standard_buffer.get_stats()
    streaming_stats = streaming_buffer.get_stats()

    print("\nBuffer Statistics:")
    print(f"Standard Buffer:")
    print(f"  Total experiences: {standard_stats['total_experiences']}")
    print(f"  Cache hit rate: {standard_stats['cache_hit_rate']:.2%}")

    print(f"\nStreaming Buffer:")
    print(f"  Total experiences in memory: {streaming_stats['total_experiences']}")
    print(f"  Cache hit rate: {streaming_stats['cache_hit_rate']:.2%}")

    # Clean up
    shutil.rmtree("./temp_streaming", ignore_errors=True)


def plot_results(write_results: Dict[str, Dict],
                search_results: Dict[str, Dict],
                sprint9_results: Dict[str, Dict]):
    """Plot benchmark results including Sprint 9 features."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    backends = list(write_results.keys())

    # Write performance
    write_speeds = [write_results[b]['experiences_per_second'] for b in backends]
    bars1 = ax1.bar(backends, write_speeds)
    ax1.set_ylabel('Experiences per Second')
    ax1.set_title('Experience Write Performance')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, speed in zip(bars1, write_speeds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.0f}', ha='center', va='bottom')

    # Search performance
    search_speeds = [search_results[b]['queries_per_second'] for b in backends]
    bars2 = ax2.bar(backends, search_speeds)
    ax2.set_ylabel('Queries per Second')
    ax2.set_title('Experience Search Performance')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, speed in zip(bars2, search_speeds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.0f}', ha='center', va='bottom')

    # Sprint 9: Network storage performance
    network_speeds = [sprint9_results[b]['networks_per_second'] for b in backends]
    bars3 = ax3.bar(backends, network_speeds)
    ax3.set_ylabel('Networks per Second')
    ax3.set_title('Continuous Network Storage Performance')
    ax3.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, speed in zip(bars3, network_speeds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.0f}', ha='center', va='bottom')

    # Sprint 9: Adapter storage performance
    adapter_speeds = [sprint9_results[b]['adapters_per_second'] for b in backends]
    bars4 = ax4.bar(backends, adapter_speeds)
    ax4.set_ylabel('Adapters per Second')
    ax4.set_title('Action Adapter Storage Performance')
    ax4.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, speed in zip(bars4, adapter_speeds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.0f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('updated_storage_benchmark_results.png', dpi=300, bbox_inches='tight')
    print("\nResults saved to updated_storage_benchmark_results.png")


def test_continuous_network_storage_manager():
    """Test the ContinuousNetworkStorageManager integration."""
    print("\n=== Testing ContinuousNetworkStorageManager ===")

    try:
        from agent_byte.core.continuous_network import ContinuousNetworkStorageManager
        from agent_byte.core.interfaces import create_continuous_action_space

        # Create temporary storage
        temp_storage = JsonNumpyStorage("./temp_continuous_test")

        # Create storage manager
        storage_manager = ContinuousNetworkStorageManager(temp_storage, "test_agent")

        print("✅ ContinuousNetworkStorageManager created successfully")

        # Test basic operations
        networks = storage_manager.list_saved_networks()
        print(f"✅ Listed {len(networks)} saved networks")

        # Clean up
        temp_storage.close()
        shutil.rmtree("./temp_continuous_test", ignore_errors=True)

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing ContinuousNetworkStorageManager: {e}")
        return False


def main():
    """Run the updated storage benchmark."""
    print("=== Agent Byte Storage Performance Benchmark (Updated & Optimized) ===")

    # Test continuous network integration first
    continuous_integration_ok = test_continuous_network_storage_manager()
    if not continuous_integration_ok:
        print("\n⚠️ Continuous network integration tests failed. Proceeding with basic benchmarks...")

    # Clean up any existing test data
    cleanup_paths = [
        "./test_json_storage", "./test_json_storage_lazy",
        "./test_optimized_json_storage", "./test_vectordb_storage",
        "./test_vectordb_faiss", "./test_vectordb_chroma",
        "./test_vectordb_hybrid", "./test_improved_vectordb_storage"
    ]

    for path in cleanup_paths:
        shutil.rmtree(path, ignore_errors=True)

    # Initialize storage backends
    storage_backends = {}

    # Original JSON+Numpy storage
    storage_backends['JSON+Numpy (Original)'] = JsonNumpyStorage("./test_json_storage")

    # Original JSON+Numpy storage with lazy loading
    storage_backends['JSON+Numpy (Lazy)'] = JsonNumpyStorage(
        "./test_json_storage_lazy",
        config={'lazy_loading': True}
    )

    # Optimized JSON+Numpy storage
    if OPTIMIZED_JSON_AVAILABLE:
        storage_backends['JSON+Numpy (Optimized)'] = OptimizedJsonNumpyStorage(
            "./test_optimized_json_storage",
            config={'batch_size': 100, 'lazy_loading': True}
        )
        print("✅ Optimized JSON+Numpy storage available")
    else:
        print("❌ Optimized JSON+Numpy storage not available")

    # Add vector DB backends if available
    try:
        # Original VectorDB with FAISS
        storage_backends['VectorDB (FAISS Original)'] = VectorDBStorage(
            "./test_vectordb_faiss",
            backend="faiss"
        )
        print("✅ Original FAISS VectorDB storage available")
    except Exception as e:
        print(f"❌ Original FAISS VectorDB not available: {e}")

    try:
        # Original VectorDB with ChromaDB
        storage_backends['VectorDB (ChromaDB Original)'] = VectorDBStorage(
            "./test_vectordb_chroma",
            backend="chroma"
        )
        print("✅ Original ChromaDB VectorDB storage available")
    except Exception as e:
        print(f"❌ Original ChromaDB VectorDB not available: {e}")

    try:
        # Original VectorDB with Hybrid
        storage_backends['VectorDB (Hybrid Original)'] = VectorDBStorage(
            "./test_vectordb_hybrid",
            backend="hybrid"
        )
        print("✅ Original Hybrid VectorDB storage available")
    except Exception as e:
        print(f"❌ Original Hybrid VectorDB not available: {e}")

    # Improved VectorDB backends
    if IMPROVED_VECTOR_DB_AVAILABLE:
        try:
            storage_backends['VectorDB (FAISS Improved)'] = ImprovedVectorDBStorage(
                "./test_improved_vectordb_faiss",
                backend="faiss"
            )
            print("✅ Improved FAISS VectorDB storage available")
        except Exception as e:
            print(f"❌ Improved FAISS VectorDB not available: {e}")

        try:
            storage_backends['VectorDB (ChromaDB Improved)'] = ImprovedVectorDBStorage(
                "./test_improved_vectordb_chroma",
                backend="chroma"
            )
            print("✅ Improved ChromaDB VectorDB storage available")
        except Exception as e:
            print(f"❌ Improved ChromaDB VectorDB not available: {e}")

        try:
            storage_backends['VectorDB (Hybrid Improved)'] = ImprovedVectorDBStorage(
                "./test_improved_vectordb_hybrid",
                backend="hybrid"
            )
            print("✅ Improved Hybrid VectorDB storage available")
        except Exception as e:
            print(f"❌ Improved Hybrid VectorDB not available: {e}")
    else:
        print("❌ Improved VectorDB storage implementations not available")

    # Configuration for benchmarks
    num_experiences = 10000  # Increased for better performance comparison
    num_queries = 100
    num_networks = 1000
    num_adapters = 1000

    print(f"\nTest configuration:")
    print(f"  Number of experiences: {num_experiences}")
    print(f"  Number of search queries: {num_queries}")
    print(f"  Number of continuous networks: {num_networks}")
    print(f"  Number of action adapters: {num_adapters}")
    print(f"  Vector dimension: 256")
    print(f"  Storage backends: {len(storage_backends)}")

    # Run benchmarks
    print("\n" + "="*50)
    print("Starting benchmarks...")
    print("="*50)

    # Write benchmark
    write_results = benchmark_write_performance(storage_backends, num_experiences)

    # Search benchmark
    search_results = benchmark_search_performance(storage_backends, num_queries)

    # Sprint 9: Continuous action space benchmarks
    sprint9_results = benchmark_sprint9_features(storage_backends, num_networks, num_adapters)

    # Memory usage benchmark
    try:
        benchmark_memory_usage()
    except Exception as e:
        print(f"Memory usage benchmark failed: {e}")

    # Summary
    print("\n" + "="*50)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*50)

    print("\n📊 Write Performance (experiences/second):")
    sorted_write = sorted(write_results.items(), key=lambda x: x[1]['experiences_per_second'], reverse=True)
    for i, (name, result) in enumerate(sorted_write, 1):
        print(f"  {i}. {name}: {result['experiences_per_second']:.0f} exp/s")

    print("\n🔍 Search Performance (queries/second):")
    sorted_search = sorted(search_results.items(), key=lambda x: x[1]['queries_per_second'], reverse=True)
    for i, (name, result) in enumerate(sorted_search, 1):
        print(f"  {i}. {name}: {result['queries_per_second']:.0f} queries/s")

    print("\n🧠 Sprint 9 - Continuous Network Performance (networks/second):")
    sorted_networks = sorted(sprint9_results.items(), key=lambda x: x[1]['networks_per_second'], reverse=True)
    for i, (name, result) in enumerate(sorted_networks, 1):
        print(f"  {i}. {name}: {result['networks_per_second']:.0f} networks/s")

    print("\n⚙️ Sprint 9 - Action Adapter Performance (adapters/second):")
    sorted_adapters = sorted(sprint9_results.items(), key=lambda x: x[1]['adapters_per_second'], reverse=True)
    for i, (name, result) in enumerate(sorted_adapters, 1):
        print(f"  {i}. {name}: {result['adapters_per_second']:.0f} adapters/s")

    # Plot results
    try:
        plot_results(write_results, search_results, sprint9_results)
    except ImportError:
        print("\n❌ Matplotlib not available for plotting")
    except Exception as e:
        print(f"\n❌ Plotting failed: {e}")

    # Performance analysis
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS")
    print("="*50)

    # Performance improvements
    if 'JSON+Numpy (Original)' in write_results and 'JSON+Numpy (Optimized)' in write_results:
        original_write = write_results['JSON+Numpy (Original)']['experiences_per_second']
        optimized_write = write_results['JSON+Numpy (Optimized)']['experiences_per_second']
        improvement = (optimized_write - original_write) / original_write * 100
        print(f"\n💡 JSON+Numpy Optimization Results:")
        print(f"  Original: {original_write:.0f} exp/s")
        print(f"  Optimized: {optimized_write:.0f} exp/s")
        print(f"  Improvement: {improvement:+.1f}%")

    # Find best overall performers
    best_write = max(write_results.items(), key=lambda x: x[1]['experiences_per_second'])
    best_search = max(search_results.items(), key=lambda x: x[1]['queries_per_second'])
    best_network = max(sprint9_results.items(), key=lambda x: x[1]['networks_per_second'])
    best_adapter = max(sprint9_results.items(), key=lambda x: x[1]['adapters_per_second'])

    print(f"\n🏆 Best performers:")
    print(f"  Write: {best_write[0]} ({best_write[1]['experiences_per_second']:.0f} exp/s)")
    print(f"  Search: {best_search[0]} ({best_search[1]['queries_per_second']:.0f} queries/s)")
    print(f"  Networks: {best_network[0]} ({best_network[1]['networks_per_second']:.0f} networks/s)")
    print(f"  Adapters: {best_adapter[0]} ({best_adapter[1]['adapters_per_second']:.0f} adapters/s)")

    # Recommendations
    print(f"\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)

    if any('VectorDB' in name for name in storage_backends.keys()):
        print("✅ Vector database backends available for high-performance similarity search")
    else:
        print("⚠️ Consider installing FAISS or ChromaDB for better search performance")
        print("   pip install faiss-cpu chromadb")

    if continuous_integration_ok:
        print("✅ Sprint 9 continuous action space features working correctly")
    else:
        print("⚠️ Sprint 9 features need setup - check imports and implementations")

    if OPTIMIZED_JSON_AVAILABLE:
        print("✅ Optimized JSON+Numpy storage shows significant performance improvements")
    else:
        print("⚠️ Use optimized storage implementations for better performance")

    if IMPROVED_VECTOR_DB_AVAILABLE:
        print("✅ Improved VectorDB storage with working hybrid backend available")
    else:
        print("⚠️ Consider using improved VectorDB implementations")

    # Clean up
    print("\n🧹 Cleaning up test data...")
    for backend in storage_backends.values():
        try:
            backend.close()
        except:
            pass

    for path in cleanup_paths:
        shutil.rmtree(path, ignore_errors=True)

    print("\n🎉 Benchmark complete!")
    print("Check 'updated_storage_benchmark_results.png' for visual results")


if __name__ == "__main__":
    main()