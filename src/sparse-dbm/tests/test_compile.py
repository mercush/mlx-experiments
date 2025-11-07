"""
Quick test to verify the sparse_dbm.py compiles and runs one training step.
"""
import mlx.core as mx
import sparse
import sys

# Import consts first to get configuration
import consts

# Now import the training functions
from sparse_dbm import TrainingState, train_step

def test_compile():
    """Test that train_step compiles with the new TrainingState structure."""
    print("Testing compilation with sparse matrices...")

    # Convert graph mask to sparse format
    print("  Converting graph mask to sparse format...")
    graph_mask_sparse = sparse.from_dense(consts.graph_mask, dtype=consts.dtype)
    print(f"  Graph mask: {graph_mask_sparse.shape}, nnz: {graph_mask_sparse.nnz}")

    # Initialize weights (sparse)
    print("  Initializing weights...")
    weights_dense = consts.graph_mask * mx.random.normal(
        [consts.num_neurons, consts.num_neurons], dtype=consts.dtype
    )
    weights_sparse = sparse.from_dense(weights_dense, dtype=consts.dtype)
    biases = consts.random_idx_transform @ consts.visible_bias_init

    # Initialize state with sparse weight components
    print("  Creating initial state...")
    state = TrainingState(
        weight_rows=weights_sparse.rows,
        weight_cols=weights_sparse.cols,
        weight_data=weights_sparse.data,
        biases=biases,
        weight_vel_data=mx.zeros_like(weights_sparse.data),
        biases_vel=mx.zeros_like(biases),
        error=mx.array(0.0),
    )

    # Get a small batch
    print("  Preparing batch...")
    batch_size = min(consts.batch_size, 2)  # Use small batch for quick test
    batch_img = consts.random_transform_train_imgs[:batch_size]
    batch_label = consts.random_transform_train_labels[:batch_size]

    print("  Running first train_step (compiling + executing)...")
    result = train_step(state, batch_img, batch_label)
    mx.eval(result)
    # mx.compile returns a tuple, not a NamedTuple - convert it back
    state = TrainingState(*result)
    print(f"  ✓ First step complete! Error: {state.error.item():.6f}")

    print("  Running second train_step (should be cached)...")
    result = train_step(state, batch_img, batch_label)
    mx.eval(result)
    state = TrainingState(*result)
    print(f"  ✓ Second step complete! Error: {state.error.item():.6f}")

    print("\n✓ Compilation test PASSED!")
    return True

if __name__ == "__main__":
    try:
        success = test_compile()
        if success:
            print("\nThe sparse implementation compiles successfully!")
            print("You can now run the full training with: python sparse_dbm.py")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Compilation test FAILED with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
