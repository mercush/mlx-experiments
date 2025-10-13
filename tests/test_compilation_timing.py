"""
Test to measure compilation benefits for each function in sparse_dbm.py
"""
import mlx.core as mx
import time
import consts
from sparse_dbm import (
    ising_bernoulli,
    sample_block_inner,
    block_gibbs_step,
    block_gibbs_sampler,
    compute_correlation,
    compute_reconstruction_error,
)


def time_function(func, *args, num_runs=10, warmup_runs=10):
    """
    Time a function execution.

    Args:
        func: Function to time
        *args: Arguments to pass to the function
        num_runs: Number of times to run for timing
        warmup_runs: Number of warmup runs (not timed)

    Returns:
        avg_time: Average execution time in seconds
    """
    # Warmup runs
    for _ in range(warmup_runs):
        result = func(*args)
        mx.eval(result)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.time()
        result = func(*args)
        mx.eval(result)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return avg_time, min_time, max_time


def test_ising_bernoulli():
    print("\n" + "="*80)
    print("Testing: ising_bernoulli")
    print("="*80)

    p = mx.array([0.5] * consts.num_neurons, dtype=consts.dtype)

    # Uncompiled version
    print("\nUncompiled version:")
    avg, min_t, max_t = time_function(ising_bernoulli, p)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")

    # Compiled version
    print("\nCompiled version:")
    compiled_func = mx.compile(ising_bernoulli)
    avg, min_t, max_t = time_function(compiled_func, p)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")


def test_sample_block_inner():
    print("\n" + "="*80)
    print("Testing: sample_block_inner")
    print("="*80)

    weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons], dtype=consts.dtype)
    biases = mx.zeros(consts.num_neurons, dtype=consts.dtype)
    clamps = mx.zeros(consts.num_neurons, dtype=consts.dtype)
    model_config = mx.ones(consts.num_neurons, dtype=consts.dtype)
    block_mask = consts.block_masks[0]

    # Uncompiled version
    print("\nUncompiled version:")
    avg, min_t, max_t = time_function(sample_block_inner, weights, biases, clamps, model_config, block_mask)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")

    # Compiled version
    print("\nCompiled version:")
    compiled_func = mx.compile(sample_block_inner)
    avg, min_t, max_t = time_function(compiled_func, weights, biases, clamps, model_config, block_mask)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")


def test_block_gibbs_step():
    print("\n" + "="*80)
    print("Testing: block_gibbs_step")
    print("="*80)

    batch_size = consts.batch_size
    weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons], dtype=consts.dtype)
    biases = mx.zeros(consts.num_neurons, dtype=consts.dtype)
    clamps = mx.zeros((batch_size, consts.num_neurons), dtype=consts.dtype)
    model_config = mx.ones((batch_size, consts.num_neurons), dtype=consts.dtype)

    # Uncompiled version
    print("\nUncompiled version:")
    avg, min_t, max_t = time_function(block_gibbs_step, weights, biases, clamps, model_config)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")

    # Compiled version
    print("\nCompiled version:")
    compiled_func = mx.compile(block_gibbs_step)
    avg, min_t, max_t = time_function(compiled_func, weights, biases, clamps, model_config)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")


def test_block_gibbs_sampler():
    print("\n" + "="*80)
    print("Testing: block_gibbs_sampler")
    print("="*80)

    batch_size = consts.batch_size
    weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons], dtype=consts.dtype)
    biases = mx.zeros(consts.num_neurons, dtype=consts.dtype)
    clamps = mx.zeros((batch_size, consts.num_neurons), dtype=consts.dtype)

    # Uncompiled version
    print("\nUncompiled version:")
    avg, min_t, max_t = time_function(block_gibbs_sampler, weights, biases, clamps, num_runs=5)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")

    # Compiled version
    print("\nCompiled version:")
    compiled_func = mx.compile(block_gibbs_sampler)
    avg, min_t, max_t = time_function(compiled_func, weights, biases, clamps, num_runs=5)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")


def test_compute_correlation():
    print("\n" + "="*80)
    print("Testing: compute_correlation")
    print("="*80)

    batch_size = consts.batch_size
    model_config = mx.random.normal((batch_size, consts.num_neurons), dtype=consts.dtype)

    # Uncompiled version
    print("\nUncompiled version:")
    avg, min_t, max_t = time_function(compute_correlation, model_config)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")

    # Compiled version
    print("\nCompiled version:")
    compiled_func = mx.compile(compute_correlation)
    avg, min_t, max_t = time_function(compiled_func, model_config)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")


def test_compute_reconstruction_error():
    print("\n" + "="*80)
    print("Testing: compute_reconstruction_error")
    print("="*80)

    batch_size = consts.batch_size
    weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons], dtype=consts.dtype)
    biases = mx.zeros(consts.num_neurons, dtype=consts.dtype)
    clamps = mx.random.normal((batch_size, consts.num_neurons), dtype=consts.dtype)

    # Uncompiled version
    print("\nUncompiled version:")
    avg, min_t, max_t = time_function(compute_reconstruction_error, weights, biases, clamps, num_runs=3)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")

    # Compiled version
    print("\nCompiled version:")
    compiled_func = mx.compile(compute_reconstruction_error)
    avg, min_t, max_t = time_function(compiled_func, weights, biases, clamps, num_runs=3)
    print(f"  Average: {avg*1000:.2f}ms")
    print(f"  Min: {min_t*1000:.2f}ms")
    print(f"  Max: {max_t*1000:.2f}ms")


def print_summary():
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey findings:")
    print("- Functions with @mx.compile should show speedup on compiled version")
    print("- If times are similar, compilation may not be helping")
    print("- Large speedups indicate successful compilation")
    print("- Check if functions are being recompiled due to shape changes")
    print("\nReasons for poor compilation performance:")
    print("1. Shape changes between calls (forces recompilation)")
    print("2. Too many small operations (compilation overhead > benefit)")
    print("3. Python loops inside compiled functions (not optimized)")
    print("4. Dynamic behavior that can't be optimized")
    print("="*80)


if __name__ == "__main__":
    print("Compilation Timing Test Suite")
    print(f"Configuration: {consts.num_neurons} neurons, batch_size={consts.batch_size}")
    print(f"sample_iter={consts.sample_iter}, dtype={consts.dtype}")

    # Run all tests
    test_ising_bernoulli()
    test_sample_block_inner()
    test_block_gibbs_step()
    test_block_gibbs_sampler()
    test_compute_correlation()
    test_compute_reconstruction_error()

    print_summary()
