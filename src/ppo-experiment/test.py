import mlx.core as mx
from tqdm import tqdm
import mlx.nn as nn
import matplotlib.pyplot as plt

BOUNDS = mx.array([-1, 1])
LR = 0.0001
N = 1000
K = 100000 # number of steps between pi_old updates
M = 1 # number of pi_old updates


def normal_logpdf(samples, mu, sigma):
    """Compute the log probability density of a normal distribution."""
    return (
        -0.5 * ((samples - mu) / sigma) ** 2
        - mx.log(sigma)
        - 0.5 * mx.log(mx.array(2 * mx.pi))
    )

def grad_step(params, params_old):
    def loss_fn(params):
        """Calculate the loss function for the PPO algorithm."""
        samples = mx.random.normal(loc=params_old[0], scale=params_old[1], shape=(N,))
        sample_logprob = normal_logpdf(samples, params_old[0], params_old[1])
        reward = mx.where(
            (BOUNDS[0] < samples) & (samples < BOUNDS[1]), 10, 0
        )
        target_logprob = (
            normal_logpdf(samples, params[0], params[1])
        )
        log_ratio = sample_logprob + reward - target_logprob
        loss = mx.mean(reward * log_ratio)
        return loss

    lval, lgrad = mx.value_and_grad(loss_fn)(params)
    params = params - LR * lgrad
    return params


def test_ppo():
    params_old = mx.array([0.0, 1.0])
    params = mx.array([0.0, 1.0])
    pbar = tqdm(range(M * K))
    for _ in range(M):
        for _ in range(K):
            params = grad_step(params, params_old)
            mx.eval(params)            
            pbar.update(1)
            pbar.set_postfix(mu=f"{params[0].item():.4f}", sigma=f"{params[1].item():.4f}")
        params_old = params

    # Plot the final distribution
    x = mx.linspace(-2, 2, 1000)
    final_dist = mx.exp(normal_logpdf(x, params[0], params[1]))
    plt.plot(x.tolist(), final_dist.tolist())
    plt.title("Final Distribution after PPO")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.show()


if __name__ == "__main__":
    test_ppo()
