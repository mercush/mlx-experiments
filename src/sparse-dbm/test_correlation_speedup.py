import mlx.core as mx
import numpy as np
import time
import sparse
import consts

@mx.compile
def correlation_without_kernel(X: mx.array, mask_rows: mx.array, mask_cols: mx.array) -> mx.array:
    """
    Compute correlation matrix without specialized kernel.
    Computes full correlation matrix and then masks it.
    
    PERFORMANCE NOTE: This is slow because it computes the full correlation matrix
    (X.T @ X) which is O(n_features^2 * batch_size) operations, even though we only
    need a small subset of values. For a 4264x4264 matrix with 60,808 non-zeros,
    this computes ~2.3 billion operations when only ~7.8 million are needed.
    
    Returns only the data values at mask positions (not a full Matrix object).
    """
    batch_size = X.shape[0]
    # Compute full correlation matrix: X.T @ X / batch_size
    # This is the bottleneck: O(n_features^2 * batch_size) operations
    corr_full = (X.T @ X) / batch_size
    
    # Extract values at mask positions using advanced indexing
    # This is relatively fast (~1ms) compared to the matrix multiplication
    mask_values = corr_full[mask_rows, mask_cols]
    
    return mask_values

def benchmark_correlation():
    """
    Benchmark correlation matrix computation with and without specialized kernel.
    Always runs the full experiment for consistent benchmarking.
    """
    print("=" * 80)
    print("Correlation Matrix Computation Benchmark")
    print("=" * 80)
    print()
    
    # Get the graph mask from consts
    graph_mask_dense = consts.graph_mask
    n_features = graph_mask_dense.shape[0]
    
    # Convert dense mask to sparse Matrix (graph_mask is already bfloat16)
    print("Converting dense mask to sparse format...")
    start_convert = time.time()
    mask = sparse.from_dense(graph_mask_dense, dtype=mx.bfloat16)
    convert_time = time.time() - start_convert
    print(f"  Conversion took: {convert_time:.2f} seconds")
    print()
    
    print(f"Graph mask shape: {graph_mask_dense.shape}")
    print(f"Number of non-zero entries in mask: {mask.nnz}")
    print(f"Mask sparsity: {1.0 - mask.nnz / (n_features * n_features):.4f}")
    print()
    
    # Create a batch of iid normal samples
    batch_size = 128
    X = mx.random.normal((batch_size, n_features), dtype=mx.bfloat16)
    
    print(f"Input matrix X shape: {X.shape}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Warmup runs for JIT compilation
    warmup_iterations = 10
    print(f"Warming up JIT (running {warmup_iterations} iterations for each path)...")
    start_warmup = time.time()
    for _ in range(warmup_iterations):
        result_with = sparse.masked_correlation(X, mask)
        mx.eval(result_with.data)  # Ensure computation completes
    for _ in range(warmup_iterations):
        result_without_data = correlation_without_kernel(X, mask.rows, mask.cols)
        mx.eval(result_without_data)  # Ensure computation completes
    warmup_time = time.time() - start_warmup
    print(f"Warmup complete. (took {warmup_time:.2f} seconds)")
    print()
    
    # Benchmark with specialized kernel
    print("Benchmarking with specialized kernel...")
    num_iterations_fast = 5
    times_with_kernel = []
    
    for _ in range(num_iterations_fast):
        start = time.time()
        result_with_kernel = sparse.masked_correlation(X, mask)
        mx.eval(result_with_kernel.data)  # Ensure computation completes
        end = time.time()
        times_with_kernel.append(end - start)
    
    avg_time_with_kernel = np.mean(times_with_kernel)
    std_time_with_kernel = np.std(times_with_kernel)
    
    print(f"  Average time: {avg_time_with_kernel * 1000:.4f} ms")
    print(f"  Std deviation: {std_time_with_kernel * 1000:.4f} ms")
    print(f"  Min time: {min(times_with_kernel) * 1000:.4f} ms")
    print(f"  Max time: {max(times_with_kernel) * 1000:.4f} ms")
    print()
    
    # Benchmark without specialized kernel
    print("Benchmarking without specialized kernel (full correlation + masking)...")
    print("  NOTE: This computes the full 4264x4264 correlation matrix, which is")
    print("  why it's slower than the specialized kernel that only computes needed values.")
    times_without_kernel = []
    
    for _ in range(5):
        start = time.time()
        result_without_kernel_data = correlation_without_kernel(X, mask.rows, mask.cols)
        mx.eval(result_without_kernel_data)  # Ensure computation completes
        end = time.time()
        times_without_kernel.append(end - start)
    
    avg_time_without_kernel = np.mean(times_without_kernel)
    std_time_without_kernel = np.std(times_without_kernel)
    
    print(f"  Average time: {avg_time_without_kernel * 1000:.4f} ms")
    print(f"  Std deviation: {std_time_without_kernel * 1000:.4f} ms")
    print(f"  Min time: {min(times_without_kernel) * 1000:.4f} ms")
    print(f"  Max time: {max(times_without_kernel) * 1000:.4f} ms")
    print()
    
    # Calculate speedup
    speedup = avg_time_without_kernel / avg_time_with_kernel
    
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"With specialized kernel:    {avg_time_with_kernel * 1000:.4f} ms")
    print(f"Without specialized kernel: {avg_time_without_kernel * 1000:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print()
    print("Performance Analysis:")
    print(f"  - Full correlation matrix computation: ~{avg_time_without_kernel * 1000:.2f} ms")
    print(f"    (Computes {n_features}x{n_features} = {n_features*n_features:,} values)")
    print(f"  - Specialized kernel: ~{avg_time_with_kernel * 1000:.2f} ms")
    print(f"    (Only computes {mask.nnz:,} needed values)")
    print(f"  - Operations ratio: {(n_features*n_features) / mask.nnz:.1f}x more operations")
    print("=" * 80)
    
    # Verify results match
    print()
    print("Verifying results match...")
    start_verify = time.time()
    # Get final results for comparison
    result_with_kernel_final = sparse.masked_correlation(X, mask)
    result_without_kernel_final = correlation_without_kernel(X, mask.rows, mask.cols)
    mx.eval(result_with_kernel_final.data)
    mx.eval(result_without_kernel_final)
    
    # Compute difference vectorized (much faster than looping with .item())
    diff = mx.abs(result_with_kernel_final.data - result_without_kernel_final)
    max_diff = mx.max(diff).item()
    verify_time = time.time() - start_verify
    print(f"  Verification took: {verify_time:.2f} seconds")
    
    print(f"Maximum difference between results: {max_diff:.8f}")
    # bfloat16 has lower precision, so use a more relaxed tolerance
    if max_diff < 1e-2:
        print("✓ Results match within numerical precision!")
    else:
        print("⚠ Results differ - this may indicate a bug or numerical precision issue.")
    print()

if __name__ == "__main__":
    try:
        benchmark_correlation()
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
