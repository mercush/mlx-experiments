import mlx.core as mx
import sparse
import numpy as np

def test_sparse_matmul():
    """Test that sparse matrix-vector multiplication works correctly."""
    print("Testing sparse matrix-vector multiplication...")

    # Create a small sparse matrix
    rows = mx.array([0, 0, 1, 2, 2], dtype=mx.int32)
    cols = mx.array([0, 2, 1, 0, 2], dtype=mx.int32)
    data = mx.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=mx.bfloat16)

    sparse_mat = sparse.Matrix(rows, cols, data, (3, 3), mx.bfloat16)

    # Create a vector
    vec = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)

    # Sparse matrix-vector multiply
    result_sparse = sparse_mat @ vec

    # Dense comparison
    dense_mat = sparse_mat.to_dense()
    result_dense = dense_mat @ vec

    print(f"Sparse result: {result_sparse}")
    print(f"Dense result:  {result_dense}")
    print(f"Match: {mx.allclose(result_sparse, result_dense, atol=1e-2)}")

    return mx.allclose(result_sparse, result_dense, atol=1e-2)

def test_masked_correlation():
    """Test that masked correlation works correctly."""
    print("\nTesting masked correlation...")

    # Create a small mask
    mask_dense = mx.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ], dtype=mx.bfloat16)

    mask_sparse = sparse.from_dense(mask_dense, dtype=mx.bfloat16)

    # Create a small data matrix (batch_size=4, n_features=3)
    X = mx.random.normal((4, 3), dtype=mx.bfloat16)

    # Compute correlation with sparse kernel
    result_sparse = sparse.masked_correlation(X, mask_sparse)

    # Compute correlation with dense method
    batch_size = X.shape[0]
    corr_full = (X.T @ X) / batch_size
    result_dense = mask_dense * corr_full

    # Extract values at mask positions
    sparse_values = result_sparse.to_dense()

    print(f"Sparse correlation shape: {result_sparse.shape}")
    print(f"Sparse correlation nnz: {result_sparse.nnz}")
    print(f"Max difference: {mx.max(mx.abs(sparse_values - result_dense)).item():.6f}")
    print(f"Match: {mx.allclose(sparse_values, result_dense, atol=1e-2)}")

    return mx.allclose(sparse_values, result_dense, atol=1e-2)

def test_sparse_operations():
    """Test sparse matrix operations used in training."""
    print("\nTesting sparse matrix operations...")

    # Create two sparse matrices with same structure
    rows = mx.array([0, 0, 1, 2], dtype=mx.int32)
    cols = mx.array([0, 1, 1, 2], dtype=mx.int32)
    data1 = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.bfloat16)
    data2 = mx.array([0.5, 1.0, 1.5, 2.0], dtype=mx.bfloat16)

    mat1 = sparse.Matrix(rows, cols, data1, (3, 3), mx.bfloat16)
    mat2 = sparse.Matrix(rows, cols, data2, (3, 3), mx.bfloat16)

    # Test element-wise data operations
    sum_data = mat1.data + mat2.data
    diff_data = mat1.data - mat2.data
    scaled_data = mat1.data * 2.0

    print(f"Sum data: {sum_data}")
    print(f"Diff data: {diff_data}")
    print(f"Scaled data: {scaled_data}")

    # Create new sparse matrix with summed data
    mat_sum = sparse.Matrix(rows, cols, sum_data, (3, 3), mx.bfloat16)

    # Verify
    dense1 = mat1.to_dense()
    dense2 = mat2.to_dense()
    dense_sum = dense1 + dense2
    sparse_sum_dense = mat_sum.to_dense()

    print(f"Dense sum:\n{dense_sum}")
    print(f"Sparse sum (as dense):\n{sparse_sum_dense}")
    print(f"Match: {mx.allclose(sparse_sum_dense, dense_sum, atol=1e-2)}")

    return mx.allclose(sparse_sum_dense, dense_sum, atol=1e-2)

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sparse DBM Operations")
    print("=" * 60)

    test1 = test_sparse_matmul()
    test2 = test_masked_correlation()
    test3 = test_sparse_operations()

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Sparse matmul: {'PASS' if test1 else 'FAIL'}")
    print(f"  Masked correlation: {'PASS' if test2 else 'FAIL'}")
    print(f"  Sparse operations: {'PASS' if test3 else 'FAIL'}")
    print("=" * 60)

    if test1 and test2 and test3:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
