import mlx.core as mx
import numpy as np
import sparse

def test_sparse_vector_matmul():
    """Test sparse matrix-vector multiplication."""
    print("Testing sparse matrix-vector multiplication...")

    # Create a simple sparse matrix
    # [[1, 0, 2],
    #  [0, 3, 0],
    #  [4, 0, 5]]
    rows = mx.array([0, 0, 1, 2, 2], dtype=mx.int32)
    cols = mx.array([0, 2, 1, 0, 2], dtype=mx.int32)
    data = mx.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=mx.bfloat16)

    mat = sparse.Matrix(rows, cols, data, (3, 3), dtype=mx.bfloat16)

    # Test vector
    vec = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)

    # Compute using sparse kernel
    result_sparse = mat @ vec

    # Compute using dense for comparison
    result_dense = mat.to_dense() @ vec

    print(f"Sparse result: {result_sparse}")
    print(f"Dense result: {result_dense}")
    # Use relaxed tolerance for bfloat16
    match = mx.allclose(result_sparse, result_dense, rtol=1e-2, atol=1e-2)
    print(f"Match: {match}")
    print()

    return match

def test_masked_correlation():
    """Test masked correlation computation."""
    print("Testing masked correlation...")

    # Create a batch of data
    batch_size = 4
    n_features = 5
    X = mx.random.normal((batch_size, n_features), dtype=mx.bfloat16)

    # Create a mask (only compute correlation for some pairs)
    mask_rows = mx.array([0, 0, 1, 2, 3], dtype=mx.int32)
    mask_cols = mx.array([0, 1, 1, 3, 4], dtype=mx.int32)
    mask_data = mx.ones(5, dtype=mx.bfloat16)
    mask = sparse.Matrix(mask_rows, mask_cols, mask_data, (n_features, n_features), dtype=mx.bfloat16)

    # Compute using sparse kernel
    result_sparse = sparse.masked_correlation(X, mask)

    # Compute using dense for comparison (convert to bfloat16 for fair comparison)
    corr_dense = ((X.T @ X) / batch_size).astype(mx.bfloat16)

    # Extract values at mask positions
    print("Sparse correlation values:")
    for i in range(mask.nnz):
        r, c = int(mask_rows[i].item()), int(mask_cols[i].item())
        sparse_val = float(result_sparse.data[i].item())
        dense_val = float(corr_dense[r, c].item())
        print(f"  Position ({r}, {c}): sparse={sparse_val:.4f}, dense={dense_val:.4f}")

    # Check if they match using vectorized comparison (much faster)
    # Extract dense values at mask positions using advanced indexing
    dense_values = corr_dense[mask_rows, mask_cols]
    # Use relaxed tolerance for bfloat16
    all_match = mx.allclose(result_sparse.data, dense_values, rtol=1e-2, atol=1e-2)
    print(f"All values match: {bool(all_match.item())}")
    print()

    return bool(all_match.item())

def test_from_dense():
    """Test creating sparse matrix from dense array."""
    print("Testing from_dense conversion...")

    dense = mx.array([[1.0, 0.0, 2.0],
                      [0.0, 3.0, 0.0],
                      [4.0, 0.0, 5.0]], dtype=mx.bfloat16)

    sparse_mat = sparse.from_dense(dense, dtype=mx.bfloat16)

    print(f"Original dense shape: {dense.shape}")
    print(f"Sparse matrix nnz: {sparse_mat.nnz}")
    print(f"Rows: {sparse_mat.rows}")
    print(f"Cols: {sparse_mat.cols}")
    print(f"Data: {sparse_mat.data}")

    # Convert back to dense
    reconstructed = sparse_mat.to_dense()

    # Use relaxed tolerance for bfloat16
    match = mx.allclose(dense, reconstructed, rtol=1e-2, atol=1e-2)
    print(f"Reconstructed matches original: {match}")
    print()

    return match

if __name__ == "__main__":
    print("=" * 60)
    print("Running sparse matrix tests")
    print("=" * 60)
    print()

    try:
        test1 = test_from_dense()
        test2 = test_sparse_vector_matmul()
        test3 = test_masked_correlation()

        print("=" * 60)
        print("Test Results:")
        print(f"  from_dense: {'PASS' if test1 else 'FAIL'}")
        print(f"  sparse_vector_matmul: {'PASS' if test2 else 'FAIL'}")
        print(f"  masked_correlation: {'PASS' if test3 else 'FAIL'}")
        print("=" * 60)

        if test1 and test2 and test3:
            print("All tests PASSED!")
        else:
            print("Some tests FAILED!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
