import mlx.core as mx
import mlx.nn as nn

def normal_logpdf(samples, mu, sigma):
    return (
        -0.5 * ((samples - mu) / sigma) ** 2
        - mx.log(sigma)
        - 0.5 * mx.log(mx.array(2 * mx.pi))
    )

# Setup: params_old = [0, 1], test with params = [0, 3]
params_old = mx.array([0.0, 1.0])
params = mx.array([0.0, 3.0])

# Sample from params_old
samples = mx.random.normal(loc=0.0, scale=1.0, shape=(10000,))

sample_logprob = normal_logpdf(samples, params_old[0], params_old[1])
target_logprob = normal_logpdf(samples, params[0], params[1])

# What kl_div_loss computes
kl_loss = nn.losses.kl_div_loss(sample_logprob, target_logprob)
print(f"kl_div_loss(sample_logprob, target_logprob): {kl_loss}")

# What kl_div_loss actually does:
# kl = E[exp(target) * (target - input)]
manual_kl = mx.mean(mx.exp(target_logprob) * (target_logprob - sample_logprob))
print(f"Manual: E[p_new(x) * (log p_new(x) - log p_old(x))]: {manual_kl}")

# What you probably WANT (standard KL divergence):
# KL(p_old || p_new) = E_{x~p_old}[log p_old(x) - log p_new(x)]
standard_kl = mx.mean(sample_logprob - target_logprob)
print(f"Standard KL(p_old || p_new): {standard_kl}")

# Analytical KL for Gaussians: log(σ_new/σ_old) + σ_old²/(2σ_new²) - 1/2
analytical_kl = mx.log(3.0/1.0) + (1.0**2)/(2 * 3.0**2) - 0.5
print(f"Analytical KL(N(0,1) || N(0,3)): {analytical_kl}")

print("\n--- Why kl_div_loss is wrong for this use case ---")
print(f"exp(target_logprob) = p_new(x) weights the samples")
print(f"When sigma_new > sigma_old, tail samples get MORE weight")
print(f"This biases the optimization to increase sigma!")
