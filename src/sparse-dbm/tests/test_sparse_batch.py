"""
Test sparse matrix @ batch of vectors operation.
"""
import mlx.core as mx
import sparse
import numpy as np

def test_sparse_batch_matmul():
    """Test sparse matrix @ batch of vectors works correctly."""
    print("Testing sparse matrix @ batch of vectors...")

    # Create a small sparse matrix
    rows = mx.array([0, 0, 1, 2, 2], dtype=mx.int32)
    cols = mx.array([0, 2, 1, 0, 2], dtype=mx.int32)
    data = mx.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=mx.bfloat16)

    sparse_mat = sparse.Matrix(rows, cols, data, (3, 3), mx.bfloat16)

    # Create a batch of vectors (batch_size=4, num_features=3)
    batch = mx.array([
        [1.0, 2.0, 3.0],
        [0.5, 1.0, 1.5],
        [2.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=mx.bfloat16)

    print(f"Sparse matrix shape: {sparse_mat.shape}")
    print(f"Batch shape: {batch.shape}")

    # Sparse matrix @ batch.T, then transpose back
    # (3, 3) @ (3, 4) = (3, 4), then transpose to (4, 3)
    result_sparse = (sparse_mat @ batch.T).T

    # Dense comparison
    dense_mat = sparse_mat.to_dense()
    result_dense = (dense_mat @ batch.T).T

    print(f"Sparse result shape: {result_sparse.shape}")
    print(f"Dense result shape: {result_dense.shape}")
    print(f"\nSparse result:\n{result_sparse}")
    print(f"\nDense result:\n{result_dense}")
    print(f"\nMax difference: {mx.max(mx.abs(result_sparse - result_dense)).item():.6f}")
    print(f"Match: {mx.allclose(result_sparse, result_dense, atol=1e-2)}")

    return mx.allclose(result_sparse, result_dense, atol=1e-2)

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sparse Matrix @ Batch Operation")
    print("=" * 60)

    passed = test_sparse_batch_matmul()

    print("\n" + "=" * 60)
    if passed:
        print("✓ Test PASSED!")
    else:
        print("✗ Test FAILED!")
    print("=" * 60)
