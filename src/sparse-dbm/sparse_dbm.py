import mlx.core as mx
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import NamedTuple
import consts
import mlx.nn as nn
import sparse


class TrainingState(NamedTuple):
    # Store sparse weights as separate arrays for compile compatibility
    weight_rows: mx.array
    weight_cols: mx.array
    weight_data: mx.array
    biases: mx.array
    weight_vel_data: mx.array  # Velocity for sparse weight data only
    biases_vel: mx.array
    error: mx.array

state = [mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def ising_bernoulli(p):
    """
    Generate a Bernoulli random variable with probability p.
    """
    return (2 * mx.random.bernoulli(p) - 1).astype(consts.dtype)

def sample_block(
    weights: sparse.Matrix,
    biases: mx.array,
    clamps: mx.array,
    model_configuration: mx.array,
    block_mask: mx.array,
):
    """
    Sample a block of neurons using sparse matrix multiplication.

    Args:
        weights: Sparse weight matrix (num_neurons, num_neurons)
        biases: Bias vector (num_neurons,)
        clamps: Clamped values (batch_size, num_neurons)
        model_configuration: Current configuration (batch_size, num_neurons)
        block_mask: Mask for which neurons to update (num_neurons,)

    Returns:
        Updated configuration (batch_size, num_neurons)
    """
    # Sparse matrix @ batch of vectors: (num_neurons, num_neurons) @ (batch_size, num_neurons).T
    # Result shape: (num_neurons, batch_size), then transpose to (batch_size, num_neurons)
    J = (weights @ model_configuration.T).T + biases
    prob = nn.sigmoid(2 * J)
    sample = ising_bernoulli(prob)

    # Fused: only update where block_mask=True AND clamps=0
    update_mask = block_mask & (clamps == 0)
    model_configuration = mx.where(update_mask, sample, model_configuration)

    return model_configuration


def block_gibbs_step(
    weights: sparse.Matrix,
    biases: mx.array,
    clamps: mx.array,
    model_configuration: mx.array,
) -> mx.array:
    # Unrolled loop over 4 block masks for better compilation
    model_configuration = sample_block(
        weights, biases, clamps, model_configuration, consts.block_masks[0]
    )
    model_configuration = sample_block(
        weights, biases, clamps, model_configuration, consts.block_masks[1]
    )
    model_configuration = sample_block(
        weights, biases, clamps, model_configuration, consts.block_masks[2]
    )
    model_configuration = sample_block(
        weights, biases, clamps, model_configuration, consts.block_masks[3]
    )
    return model_configuration


def block_gibbs_sampler(
    weights: sparse.Matrix, biases: mx.array, clamps: mx.array
) -> mx.array:
    """
    Perform block Gibbs sampling to generate a new model configuration.
    """
    model_configuration = ising_bernoulli(
        mx.full(clamps.shape, 0.5, dtype=consts.dtype)
    )
    model_configuration = mx.where(clamps == 0, model_configuration, clamps)

    for _ in range(consts.sample_iter):
        model_configuration = block_gibbs_step(
            weights, biases, clamps, model_configuration
        )
    return model_configuration


def compute_correlation(
    model_configuration: mx.array, graph_mask_sparse: sparse.Matrix
) -> sparse.Matrix:
    """
    Compute masked correlation using optimized sparse kernel.
    This is much faster than computing full correlation and then masking.
    """
    return sparse.masked_correlation(model_configuration, graph_mask_sparse)


def compute_reconstruction_error(
    weights: sparse.Matrix, biases: mx.array, clamps: mx.array
) -> mx.array:
    """
    Compute the reconstruction error between the model and data correlations.
    """
    hidden = block_gibbs_sampler(weights, biases, clamps)
    # Use scalar 0 instead of zeros_like to avoid allocation
    hidden_clamps = mx.where(clamps == 0, hidden, 0)
    reconstruction = block_gibbs_sampler(weights, biases, hidden_clamps)
    # Use scalar 0 instead of zeros_like to avoid allocation
    reconstruction_clamps = mx.where(clamps == 0, 0, reconstruction)
    error = mx.mean((clamps - reconstruction_clamps) ** 2)
    return error


@partial(mx.compile, inputs=state, outputs=state)
def train_step(
    state: TrainingState, batch_img: mx.array, batch_label: mx.array
) -> TrainingState:
    """
    Perform a single training step with fully sparse operations.
    Uses sparse matrix-vector multiplication in Gibbs sampling and sparse correlation.
    """
    weight_rows, weight_cols, weight_data, biases, weight_vel_data, biases_vel, _ = state
    batch = batch_img + batch_label

    # Reconstruct sparse weights
    weights = sparse.Matrix(
        weight_rows,
        weight_cols,
        weight_data,
        (consts.num_neurons, consts.num_neurons),
        consts.dtype,
    )

    # Reconstruct sparse mask for correlation kernel
    mask_data = mx.ones_like(weight_rows).astype(consts.dtype)
    graph_mask_sparse = sparse.Matrix(
        weight_rows,
        weight_cols,
        mask_data,
        (consts.num_neurons, consts.num_neurons),
        consts.dtype,
    )

    # positive phase - now using sparse matrix multiplication!
    model_configuration = block_gibbs_sampler(weights, biases, batch)

    # Use sparse kernel for correlation (4.27x faster than dense!)
    m_data_sparse = compute_correlation(model_configuration, graph_mask_sparse)
    m_data_bias = mx.mean(model_configuration, axis=0)

    # negative phase - now using sparse matrix multiplication!
    model_configuration = block_gibbs_sampler(weights, biases, mx.zeros_like(batch))

    m_model_sparse = compute_correlation(model_configuration, graph_mask_sparse)
    m_model_bias = mx.mean(model_configuration, axis=0)

    # updating weights - operate on sparse data directly
    delta_weight_data = (
        consts.learning_rate * (m_data_sparse.data - m_model_sparse.data) + consts.alpha * weight_vel_data
    )
    delta_biases = (
        consts.learning_rate * (m_data_bias - m_model_bias) + consts.alpha * biases_vel
    )

    new_weight_data = weight_data + delta_weight_data
    new_biases = biases + delta_biases

    # Reconstruct new weights for error computation
    new_weights = sparse.Matrix(
        weight_rows,
        weight_cols,
        new_weight_data,
        (consts.num_neurons, consts.num_neurons),
        consts.dtype,
    )
    error = compute_reconstruction_error(new_weights, new_biases, batch)

    return TrainingState(
        weight_rows,
        weight_cols,
        new_weight_data,
        new_biases,
        delta_weight_data,
        delta_biases,
        error,
    )


if __name__ == "__main__":
    # Convert graph mask to sparse format
    print("Converting graph mask to sparse format...")
    graph_mask_sparse = sparse.from_dense(consts.graph_mask, dtype=consts.dtype)
    print(f"Graph mask: {graph_mask_sparse.shape}, nnz: {graph_mask_sparse.nnz}")
    print("  -> Will use fully sparse operations (matrix-vector and correlation)")

    # Initialize weights (sparse)
    weights_dense = consts.graph_mask * mx.random.normal(
        [consts.num_neurons, consts.num_neurons], dtype=consts.dtype
    )
    weights_sparse = sparse.from_dense(weights_dense, dtype=consts.dtype)
    biases = consts.random_idx_transform @ consts.visible_bias_init

    # Initialize state with sparse weight components
    state = TrainingState(
        weight_rows=weights_sparse.rows,
        weight_cols=weights_sparse.cols,
        weight_data=weights_sparse.data,
        biases=biases,
        weight_vel_data=mx.zeros_like(weights_sparse.data),
        biases_vel=mx.zeros_like(biases),
        error=mx.array(0.0),
    )
    errors = []
    for epoch in range(consts.epochs):
        with tqdm(
            total=len(consts.random_transform_train_imgs),
            desc=f"Epoch {epoch + 1}/{consts.epochs}",
        ) as pbar:
            for i in range(
                0, len(consts.random_transform_train_imgs), consts.batch_size
            ):
                state = train_step(
                    state,
                    consts.random_transform_train_imgs[i : i + consts.batch_size],
                    consts.random_transform_train_labels[i : i + consts.batch_size],
                )
                mx.eval(state)
                pbar.update(consts.batch_size)
                pbar.set_postfix(
                    error=state[-1].item(), peak_memory=mx.get_peak_memory() / 1e9
                )
                errors.append(state[-1].item())
    plt.plot(errors)
    plt.ylabel("Reconstruction Error")
    plt.savefig("training_error.png")
    plt.show()
