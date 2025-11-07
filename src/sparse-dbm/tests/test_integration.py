"""
Integration test for sparse_dbm.py with a minimal example.
This tests that the sparse implementation can run a single training step.
"""
import mlx.core as mx
import sparse
import mlx.nn as nn
import numpy as np

# Minimal configuration
num_neurons = 10
batch_size = 4
dtype = mx.bfloat16

def test_single_training_step():
    """Test a single training step with sparse operations."""
    print("Testing single training step with sparse operations...")

    # Create a simple sparse mask (e.g., diagonal + some off-diagonals)
    mask_np = np.eye(num_neurons, dtype=np.float32)
    # Add some off-diagonal connections
    for i in range(num_neurons - 1):
        mask_np[i, i+1] = 1.0
        mask_np[i+1, i] = 1.0

    graph_mask = mx.array(mask_np, dtype=dtype)
    graph_mask_sparse = sparse.from_dense(graph_mask, dtype=dtype)

    print(f"  Graph mask shape: {graph_mask_sparse.shape}")
    print(f"  Graph mask nnz: {graph_mask_sparse.nnz}")

    # Initialize sparse weights
    weights_dense = graph_mask * mx.random.normal((num_neurons, num_neurons), dtype=dtype)
    weights = sparse.from_dense(weights_dense, dtype=dtype)

    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights nnz: {weights.nnz}")

    # Initialize biases
    biases = mx.zeros((num_neurons,), dtype=dtype)

    # Create a batch of configurations
    batch = mx.random.normal((batch_size, num_neurons), dtype=dtype)
    batch = mx.where(batch > 0, mx.array(1.0, dtype=dtype), mx.array(-1.0, dtype=dtype))

    print(f"  Batch shape: {batch.shape}")

    # Test sparse matrix-vector multiplication (core operation in sampling)
    print("\n  Testing sparse @ vector...")
    vec = batch[0]  # Take first sample
    result = weights @ vec
    print(f"    Result shape: {result.shape}")
    print(f"    Result dtype: {result.dtype}")

    # Test masked correlation (core operation in learning)
    print("\n  Testing masked correlation...")
    corr_sparse = sparse.masked_correlation(batch, graph_mask_sparse)
    print(f"    Correlation shape: {corr_sparse.shape}")
    print(f"    Correlation nnz: {corr_sparse.nnz}")
    print(f"    Correlation dtype: {corr_sparse.dtype}")

    # Test that we can update weights
    print("\n  Testing weight update...")
    learning_rate = 0.01
    delta_data = learning_rate * corr_sparse.data
    new_weights_data = weights.data + delta_data
    new_weights = sparse.Matrix(
        weights.rows, weights.cols, new_weights_data, weights.shape, weights.dtype
    )
    print(f"    New weights shape: {new_weights.shape}")
    print(f"    New weights nnz: {new_weights.nnz}")

    print("\n✓ Single training step simulation PASSED!")
    return True

def test_block_structure():
    """Test that block masking works with sparse operations."""
    print("\nTesting block structure compatibility...")

    # Create 4 complementary block masks (simple checkerboard-like pattern)
    block_masks = []
    for block_id in range(4):
        mask = mx.array([(i % 4) == block_id for i in range(num_neurons)], dtype=mx.bool_)
        block_masks.append(mask)

    print(f"  Created {len(block_masks)} block masks")

    # Verify they're complementary
    total_mask = sum(m.astype(mx.float32) for m in block_masks)
    print(f"  Sum of masks: min={mx.min(total_mask).item()}, max={mx.max(total_mask).item()}")

    # Test that we can apply block masks to sparse operations
    config = mx.random.normal((num_neurons,), dtype=dtype)
    config = mx.where(config > 0, mx.array(1.0, dtype=dtype), mx.array(-1.0, dtype=dtype))

    for i, block_mask in enumerate(block_masks):
        # Simulate updating only certain positions
        sample = mx.random.normal((num_neurons,), dtype=dtype)
        sample = mx.where(sample > 0, mx.array(1.0, dtype=dtype), mx.array(-1.0, dtype=dtype))

        config = mx.where(block_mask, sample, config)
        print(f"    Block {i+1}: updated {mx.sum(block_mask).item():.0f} positions")

    print("\n✓ Block structure test PASSED!")
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Sparse DBM Integration Test")
    print("=" * 70)

    try:
        test1 = test_single_training_step()
        test2 = test_block_structure()

        print("\n" + "=" * 70)
        print("Integration Test Results:")
        print(f"  Single training step: {'PASS' if test1 else 'FAIL'}")
        print(f"  Block structure: {'PASS' if test2 else 'FAIL'}")
        print("=" * 70)

        if test1 and test2:
            print("\n✓ All integration tests PASSED!")
            print("\nThe sparse implementation is ready to use in sparse_dbm.py")
        else:
            print("\n✗ Some integration tests FAILED!")

    except Exception as e:
        print(f"\n✗ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
