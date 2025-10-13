import mlx.core as mx
import pytest
import tracemalloc
from typing import Callable, Any
import consts
from sparse_dbm import (
    ising_bernoulli,
    sample_block_inner,
    sample_block,
    block_gibbs_step,
    block_gibbs_sampler,
    compute_correlation,
    compute_reconstruction_error,
    train_step,
    TrainingState,
)


class MemoryProfiler:
    """Context manager to track memory usage during function execution."""

    def __init__(self, name: str):
        self.name = name
        self.peak_memory = 0
        self.memory_diff = 0

    def __enter__(self):
        tracemalloc.start()
        mx.reset_peak_memory()
        return self

    def __exit__(self, *_):
        self.peak_memory = mx.get_peak_memory() / (1024**2)  # Convert to MB
        _, peak = tracemalloc.get_traced_memory()
        self.memory_diff = peak / (1024**2)  # Convert to MB
        tracemalloc.stop()
        print(f"\n{self.name}:")
        print(f"  Peak GPU memory: {self.peak_memory:.2f} MB")
        print(f"  Peak CPU memory: {self.memory_diff:.2f} MB")


def measure_memory(func: Callable, *args, **kwargs) -> tuple[Any, float, float]:
    """
    Measure memory usage of a function call.
    Returns: (result, gpu_memory_mb, cpu_memory_mb)
    """
    tracemalloc.start()
    mx.reset_peak_memory()

    result = func(*args, **kwargs)
    mx.eval(result)  # Force evaluation

    gpu_memory = mx.get_peak_memory() / (1024**2)
    _, peak = tracemalloc.get_traced_memory()
    cpu_memory = peak / (1024**2)
    tracemalloc.stop()

    return result, gpu_memory, cpu_memory


class TestSamplingFunctions:
    """Test sampling functions and their memory usage."""

    def test_ising_bernoulli_memory(self):
        """Test memory usage of ising_bernoulli function."""
        with MemoryProfiler("ising_bernoulli (small)"):
            p = mx.array([0.5] * 100)
            result = ising_bernoulli(p)
            mx.eval(result)

        with MemoryProfiler("ising_bernoulli (large - full network)"):
            p = mx.array([0.5] * consts.num_neurons)
            result = ising_bernoulli(p)
            mx.eval(result)

        assert result.shape == (consts.num_neurons,)
        assert mx.all((result == 1) | (result == -1))

    def test_sample_block_inner_memory(self):
        """Test memory usage of single block sampling."""
        batch_size = consts.batch_size
        weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons])
        biases = mx.zeros((consts.num_neurons,))
        clamps = mx.zeros((batch_size, consts.num_neurons))
        model_config = mx.ones((batch_size, consts.num_neurons))
        block_mask = consts.block_masks[0]

        with MemoryProfiler("sample_block_inner"):
            result = sample_block_inner(
                weights, biases, clamps[0], model_config[0], block_mask
            )
            mx.eval(result)

        assert result.shape == (consts.num_neurons,)

    def test_sample_block_vmapped_memory(self):
        """Test memory usage of vmapped block sampling (CRITICAL - this may show high memory)."""
        batch_size = consts.batch_size
        weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons])
        biases = mx.zeros(consts.num_neurons)
        clamps = mx.zeros((batch_size, consts.num_neurons))
        model_config = mx.ones((batch_size, consts.num_neurons))
        block_mask = consts.block_masks[0]

        with MemoryProfiler(f"sample_block (vmapped over batch_size={batch_size})"):
            result = sample_block(
                weights, biases, clamps, model_config, block_mask
            )
            mx.eval(result)

        assert result.shape == (batch_size, consts.num_neurons)

    def test_block_gibbs_step_memory(self):
        """Test memory usage of full Gibbs step (4 blocks)."""
        batch_size = consts.batch_size
        weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons])
        biases = mx.zeros(consts.num_neurons)
        clamps = mx.zeros((batch_size, consts.num_neurons))
        model_config = mx.ones((batch_size, consts.num_neurons))

        with MemoryProfiler(f"block_gibbs_step (4 blocks, batch_size={batch_size})"):
            result = block_gibbs_step(weights, biases, clamps, model_config)
            mx.eval(result)

        assert result.shape == (batch_size, consts.num_neurons)

    def test_block_gibbs_sampler_memory(self):
        """Test memory usage of full Gibbs sampler (multiple iterations)."""
        batch_size = consts.batch_size
        weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons])
        biases = mx.zeros(consts.num_neurons)
        clamps = mx.zeros((batch_size, consts.num_neurons))

        with MemoryProfiler(f"block_gibbs_sampler ({consts.sample_iter} iterations, batch_size={batch_size})"):
            result = block_gibbs_sampler(weights, biases, clamps)
            mx.eval(result)

        assert result.shape == (batch_size, consts.num_neurons)


class TestCorrelationFunctions:
    """Test correlation computation and identify memory bottlenecks."""

    def test_corr_matrix_memory(self):
        """Test memory usage of correlation matrix (CRITICAL - outer product is expensive)."""
        batch_size = consts.batch_size
        model_config = mx.random.normal((batch_size, consts.num_neurons))

        print(f"\nCorrelation matrix size: ({consts.num_neurons}, {consts.num_neurons})")
        print(f"Expected memory per matrix: {consts.num_neurons**2 * 2 / (1024**2):.2f} MB (bfloat16)")
        print(f"Total for batch of {batch_size}: {batch_size * consts.num_neurons**2 * 2 / (1024**2):.2f} MB")

        with MemoryProfiler(f"corr_matrix (vmapped, batch_size={batch_size})"):
            result = compute_correlation(model_config)
            mx.eval(result)

        assert result.shape == (consts.num_neurons, consts.num_neurons)

    def test_compute_correlation_memory(self):
        """Test full correlation computation with graph masking."""
        batch_size = consts.batch_size
        model_config = mx.random.normal((batch_size, consts.num_neurons))

        with MemoryProfiler(f"compute_correlation (batch_size={batch_size})"):
            result = compute_correlation(model_config)
            mx.eval(result)

        assert result.shape == (consts.num_neurons, consts.num_neurons)

class TestReconstructionError:
    """Test reconstruction error computation."""

    def test_reconstruction_error_memory(self):
        """Test memory usage of reconstruction error (runs 2 Gibbs samplers)."""
        batch_size = consts.batch_size
        weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons])
        biases = mx.zeros(consts.num_neurons)
        clamps = mx.random.normal((batch_size, consts.num_neurons))

        with MemoryProfiler(f"compute_reconstruction_error (2x Gibbs sampler, batch_size={batch_size})"):
            result = compute_reconstruction_error(weights, biases, clamps)
            mx.eval(result)

        assert isinstance(result.item(), float)


class TestTrainingStep:
    """Test full training step and identify cumulative memory usage."""

    def test_train_step_memory(self):
        """Test memory usage of complete training step."""
        batch_size = consts.batch_size
        weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons], dtype=consts.dtype)
        biases = mx.zeros(consts.num_neurons, dtype=consts.dtype)

        state = TrainingState(
            weights=weights,
            biases=biases,
            weight_vel=mx.zeros_like(weights),
            biases_vel=mx.zeros_like(biases),
            error=mx.array(0.0),
        )

        batch_img = consts.random_transform_train_imgs[:batch_size]
        batch_label = consts.random_transform_train_labels[:batch_size]
        print("batch_img:", batch_img.shape, batch_img.dtype)

        with MemoryProfiler(f"train_step (full training step, batch_size={batch_size})"):
            new_state = train_step(state, batch_img, batch_label)
            mx.eval(new_state)

    def test_train_step_phases_memory(self):
        """Break down memory usage by training phase."""
        batch_size = consts.batch_size
        weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons], dtype=consts.dtype)
        biases = mx.zeros(consts.num_neurons, dtype=consts.dtype)

        batch_img = consts.random_transform_train_imgs[:batch_size]
        batch_label = consts.random_transform_train_labels[:batch_size]
        batch = batch_img + batch_label

        # Phase 1: Positive phase sampling
        with MemoryProfiler("Phase 1: Positive Gibbs sampling"):
            model_config_pos = block_gibbs_sampler(weights, biases, batch)
            mx.eval(model_config_pos)

        # Phase 2: Positive phase correlation
        with MemoryProfiler("Phase 2: Positive correlation computation"):
            m_data = compute_correlation(model_config_pos)
            m_data_bias = mx.mean(model_config_pos, axis=0)
            mx.eval(m_data, m_data_bias)

        # Phase 3: Negative phase sampling
        with MemoryProfiler("Phase 3: Negative Gibbs sampling"):
            model_config_neg = block_gibbs_sampler(weights, biases, mx.zeros_like(batch))
            mx.eval(model_config_neg)

        # Phase 4: Negative phase correlation
        with MemoryProfiler("Phase 4: Negative correlation computation"):
            m_model = compute_correlation(model_config_neg)
            m_model_bias = mx.mean(model_config_neg, axis=0)
            mx.eval(m_model, m_model_bias)

        # Phase 5: Weight update
        with MemoryProfiler("Phase 5: Weight update"):
            delta_weights = consts.learning_rate * (m_data - m_model)
            delta_biases = consts.learning_rate * (m_data_bias - m_model_bias)
            new_weights = weights + delta_weights
            new_biases = biases + delta_biases
            mx.eval(new_weights, new_biases)

        # Phase 6: Reconstruction error
        with MemoryProfiler("Phase 6: Reconstruction error computation"):
            error = compute_reconstruction_error(new_weights, new_biases, batch)
            mx.eval(error)


class TestMemoryScaling:
    """Test how memory scales with different parameters."""

    def test_batch_size_scaling(self):
        """Test memory scaling with different batch sizes."""
        weights = consts.graph_mask * mx.random.normal([consts.num_neurons, consts.num_neurons])
        biases = mx.zeros(consts.num_neurons)

        for batch_size in [1, 2, 4, 8, 16]:
            clamps = mx.zeros((batch_size, consts.num_neurons))

            _, gpu_mem, cpu_mem = measure_memory(
                block_gibbs_sampler, weights, biases, clamps
            )

            print(f"Batch size {batch_size:2d}: GPU={gpu_mem:6.2f} MB, CPU={cpu_mem:6.2f} MB")

    def test_correlation_batch_scaling(self):
        """Test correlation memory scaling with batch size."""
        print("\nCorrelation computation memory scaling:")

        for batch_size in [1, 2, 4, 8, 16]:
            model_config = mx.random.normal((batch_size, consts.num_neurons))

            _, gpu_mem, cpu_mem = measure_memory(
                compute_correlation, model_config
            )

            print(f"Batch size {batch_size:2d}: GPU={gpu_mem:6.2f} MB, CPU={cpu_mem:6.2f} MB")


if __name__ == "__main__":
    print("=" * 80)
    print("MEMORY PROFILING TEST SUITE FOR SPARSE-DBM")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Network size: {consts.num_neurons} neurons")
    print(f"  Batch size: {consts.batch_size}")
    print(f"  Sample iterations: {consts.sample_iter}")
    print(f"  Data type: {consts.dtype}")
    print(f"  Number of edges: {consts.num_edges}")
    print("=" * 80)

    # Run tests
    pytest.main([__file__, "-v", "-s"])
