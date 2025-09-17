import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Our existing modules
from helpers import PATH_DATA
from datagen import get_decks, save_decks, load_decks
from gen_data import generate_seeds, deck_from_seed, compute_scores_from_seeds


# ----------------------------
# Benchmark utilities
# ----------------------------
def benchmark(func, *args, repeat: int = 5, **kwargs):
    """Run function multiple times and return mean, std of runtime (seconds)."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def memory_usage_bytes(arr: np.ndarray) -> int:
    """Return memory usage of NumPy array in bytes."""
    return arr.nbytes


# ----------------------------
# Run tests
# ----------------------------
def run_tests(n_decks: int = 2_000_000, seed: int = 12345, batch_size: int = 100_000):
    """
    Run reproducibility + performance tests for both datagen and seed pipelines.
    """
    results = []

    # =====================================================
    # 1. Deck pipeline (datagen.py)
    # =====================================================
    mean_gen, std_gen = benchmark(get_decks, n_decks, seed)
    sample_decks = get_decks(n_decks, seed)
    mem_bytes = memory_usage_bytes(sample_decks)

    results.append({
        "Pipeline": "Decks",
        "Test": "Deck Generation",
        "Num Decks": n_decks,
        "Batch Size": batch_size,
        "Mean Time (s)": mean_gen,
        "Std Time (s)": std_gen,
        "Memory (MB)": mem_bytes / 1e6
    })

    mean_write, std_write = benchmark(save_decks, sample_decks, seed, batch_size, "tmp_decks")
    results.append({
        "Pipeline": "Decks",
        "Test": "Disk Write",
        "Num Decks": n_decks,
        "Batch Size": batch_size,
        "Mean Time (s)": mean_write,
        "Std Time (s)": std_write,
        "Memory (MB)": mem_bytes / 1e6
    })

    batch_file = os.path.join(PATH_DATA, "tmp_decks_0.npy")
    mean_read, std_read = benchmark(np.load, batch_file)
    results.append({
        "Pipeline": "Decks",
        "Test": "Disk Read",
        "Num Decks": n_decks,
        "Batch Size": batch_size,
        "Mean Time (s)": mean_read,
        "Std Time (s)": std_read,
        "Memory (MB)": mem_bytes / 1e6
    })

    for f in Path(PATH_DATA).glob("tmp_decks*"):
        f.unlink()

    # =====================================================
    # 2. Seed pipeline (gen_data.py)
    # =====================================================
    n_seeds = n_decks  # one seed per deck

    mean_seed, std_seed = benchmark(generate_seeds, n_seeds, seed)
    seeds = generate_seeds(n_seeds, seed)
    mem_bytes_seeds = memory_usage_bytes(seeds)

    results.append({
        "Pipeline": "Seeds",
        "Test": "Seed Generation",
        "Num Decks": n_decks,
        "Batch Size": "-",
        "Mean Time (s)": mean_seed,
        "Std Time (s)": std_seed,
        "Memory (MB)": mem_bytes_seeds / 1e6
    })

    mean_rebuild, std_rebuild = benchmark(deck_from_seed, int(seeds[0]))
    results.append({
        "Pipeline": "Seeds",
        "Test": "Deck Rebuild (from seed)",
        "Num Decks": 1,
        "Batch Size": "-",
        "Mean Time (s)": mean_rebuild,
        "Std Time (s)": std_rebuild,
        "Memory (MB)": 52 / 1e6  # trivial single deck memory
    })

    # Example scoring: sum of 1s
    def example_score_fn(deck):
        return np.sum(deck)

    mean_score, std_score = benchmark(compute_scores_from_seeds, seeds[:10_000], example_score_fn)
    scores = compute_scores_from_seeds(seeds[:10_000], example_score_fn)
    mem_bytes_scores = memory_usage_bytes(scores)

    results.append({
        "Pipeline": "Seeds",
        "Test": "Compute Scores",
        "Num Decks": n_decks,
        "Batch Size": "-",
        "Mean Time (s)": mean_score,
        "Std Time (s)": std_score,
        "Memory (MB)": mem_bytes_scores / 1e6
    })

    # =====================================================
    # Save results
    # =====================================================
    df = pd.DataFrame(results)
    out_file = Path(PATH_DATA) / "benchmark_results.csv"
    df.to_csv(out_file, index=False)

    print("\nBenchmark Results:")
    print(df.to_string(index=False))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    run_tests()
