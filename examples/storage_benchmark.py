"""
Storage performance benchmark comparing different backends.

This script demonstrates the performance improvements of using
vector database storage over JSON+Numpy storage.
"""

import numpy as np
import time
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

from agent_byte.storage import JsonNumpyStorage, VectorDBStorage
from agent_byte.storage.experience_buffer import ExperienceBuffer, StreamingExperienceBuffer


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
                search_results: Dict[str, Dict]):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Write performance
    backends = list(write_results.keys())
    write_speeds = [write_results[b]['experiences_per_second'] for b in backends]

    ax1.bar(backends, write_speeds)
    ax1.set_ylabel('Experiences per Second')
    ax1.set_title('Write Performance')
    ax1.tick_params(axis='x', rotation=45)

    # Search performance
    search_speeds = [search_results[b]['queries_per_second'] for b in backends]

    ax2.bar(backends, search_speeds)
    ax2.set_ylabel('Queries per Second')
    ax2.set_title('Search Performance')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('storage_benchmark_results.png')
    print("\nResults saved to storage_benchmark_results.png")


def main():
    """Run the storage benchmark."""
    print("=== Agent Byte Storage Performance Benchmark ===")

    # Clean up any existing test data
    for path in ["./test_json_storage", "./test_vectordb_storage"]:
        shutil.rmtree(path, ignore_errors=True)

    # Initialize storage backends
    storage_backends = {
        'JSON+Numpy': JsonNumpyStorage("./test_json_storage"),
        'JSON+Numpy (Lazy)': JsonNumpyStorage(
            "./test_json_storage_lazy",
            config={'lazy_loading': True}
        )
    }

    # Add vector DB backends if available
    try:
        storage_backends['VectorDB (FAISS)'] = VectorDBStorage(
            "./test_vectordb_faiss",
            backend="faiss"
        )
    except:
        print("FAISS not available, skipping...")

    try:
        storage_backends['VectorDB (ChromaDB)'] = VectorDBStorage(
            "./test_vectordb_chroma",
            backend="chroma"
        )
    except:
        print("ChromaDB not available, skipping...")

    try:
        storage_backends['VectorDB (Hybrid)'] = VectorDBStorage(
            "./test_vectordb_hybrid",
            backend="hybrid"
        )
    except:
        print("Hybrid backend not available, skipping...")

    # Run benchmarks
    num_experiences = 10000
    num_queries = 100

    print(f"\nTest configuration:")
    print(f"  Number of experiences: {num_experiences}")
    print(f"  Number of search queries: {num_queries}")
    print(f"  Vector dimension: 256")

    # Write benchmark
    write_results = benchmark_write_performance(storage_backends, num_experiences)

    # Search benchmark
    search_results = benchmark_search_performance(storage_backends, num_queries)

    # Memory usage benchmark
    benchmark_memory_usage()

    # Summary
    print("\n=== Summary ===")
    print("\nWrite Performance (experiences/second):")
    for name, result in write_results.items():
        print(f"  {name}: {result['experiences_per_second']:.0f}")

    print("\nSearch Performance (queries/second):")
    for name, result in search_results.items():
        print(f"  {name}: {result['queries_per_second']:.0f}")

    # Plot results
    try:
        plot_results(write_results, search_results)
    except ImportError:
        print("\nMatplotlib not available for plotting")

    # Clean up
    print("\nCleaning up test data...")
    for backend in storage_backends.values():
        backend.close()

    for path in ["./test_json_storage", "./test_json_storage_lazy",
                 "./test_vectordb_faiss", "./test_vectordb_chroma",
                 "./test_vectordb_hybrid"]:
        shutil.rmtree(path, ignore_errors=True)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()