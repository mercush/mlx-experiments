import mlx.core as mx
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import NamedTuple
import consts
import mlx.nn as nn

class TrainingState(NamedTuple):
    weights: mx.array
    biases: mx.array
    weight_vel: mx.array
    biases_vel: mx.array
    error: mx.array

@mx.compile
def ising_bernoulli(p):
    """
    Generate a Bernoulli random variable with probability p.
    """
    return (2 * mx.random.bernoulli(p) - 1).astype(consts.dtype)

def sample_block_inner(
    weights: mx.array,
    biases: mx.array,
    clamps: mx.array,
    model_configuration: mx.array,
    block_mask: mx.array,
):
    # TODO: optimize weights here so that it's sparse
    J = weights @ model_configuration + biases
    prob = nn.sigmoid(2 * J)
    sample = ising_bernoulli(prob)

    # Fused: only update where block_mask=True AND clamps=0
    update_mask = block_mask & (clamps == 0)
    model_configuration = mx.where(update_mask, sample, model_configuration)

    return model_configuration

sample_block = mx.vmap(sample_block_inner, in_axes=(None, None, 0, 0, None))

def block_gibbs_step(
    weights: mx.array,
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
    weights: mx.array, biases: mx.array, clamps: mx.array
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

def compute_correlation(model_configuration: mx.array) -> mx.array:
    return consts.graph_mask * (model_configuration.T @ model_configuration) / consts.batch_size

def compute_reconstruction_error(
    weights: mx.array, biases: mx.array, clamps: mx.array
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

@mx.compile
def train_step(
    state: TrainingState, batch_img: mx.array, batch_label: mx.array
) -> TrainingState:
    """
    Perform a single training step.
    """
    weights, biases, weight_vel, biases_vel, _ = state
    batch = batch_img + batch_label
    # positive phase
    model_configuration = block_gibbs_sampler(weights, biases, batch)

    m_data = compute_correlation(model_configuration)
    m_data_bias = mx.mean(model_configuration, axis=0)

    # negative phase
    model_configuration = block_gibbs_sampler(weights, biases, mx.zeros_like(batch))

    m_model = compute_correlation(model_configuration)
    m_model_bias = mx.mean(model_configuration, axis=0)

    # updating weights
    delta_weights = (
        consts.learning_rate * (m_data - m_model) + consts.alpha * weight_vel
    )
    delta_biases = (
        consts.learning_rate * (m_data_bias - m_model_bias) + consts.alpha * biases_vel
    )

    new_weights, new_biases = weights + delta_weights, biases + delta_biases
    error = compute_reconstruction_error(new_weights, new_biases, batch)

    return TrainingState(new_weights, new_biases, delta_weights, delta_biases, error)


if __name__ == "__main__":
    weights = consts.graph_mask * mx.random.normal(
        [consts.num_neurons, consts.num_neurons]
    )
    biases = consts.random_idx_transform @ consts.visible_bias_init

    state = TrainingState(
        weights=weights,
        biases=biases,
        weight_vel=mx.zeros_like(weights),
        biases_vel=mx.zeros_like(biases),
        error=mx.array(0.0),
    )
    errors = []
    for epoch in range(consts.epochs):
        with tqdm(
            total=len(consts.random_transform_train_imgs),
            desc=f"Epoch {epoch+1}/{consts.epochs}"
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
                pbar.set_postfix(error=state[4].item(), peak_memory=mx.get_peak_memory() / 1e9)
                errors.append(state[4].item())
    plt.plot(errors)
    plt.ylabel('Reconstruction Error')
    plt.savefig('training_error.png')
    plt.show()
