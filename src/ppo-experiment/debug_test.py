import mlx.core as mx
import mlx.nn as nn

# Simple test to understand what's happening
params_old = mx.array([0.0, 1.0])
params = mx.array([0.0, 1.0])

# Draw samples from params_old
samples = mx.random.normal(loc=params_old[0], scale=params_old[1], shape=(1000,))

def normal_logpdf(samples, mu, sigma):
    """Compute the log probability density of a normal distribution."""
    return (
        -0.5 * ((samples - mu) / sigma) ** 2
        - mx.log(sigma)
        - 0.5 * mx.log(mx.array(2 * mx.pi))
    )

# Compute log probs
sample_logprob = normal_logpdf(samples, params[0], params[1])
target_logprob = normal_logpdf(samples, params_old[0], params_old[1])

print("When params = params_old:")
print(f"  sample_logprob mean: {mx.mean(sample_logprob)}")
print(f"  target_logprob mean: {mx.mean(target_logprob)}")
print(f"  Difference: {mx.mean(target_logprob - sample_logprob)}")

# Compute KL loss
loss = nn.losses.kl_div_loss(sample_logprob, target_logprob)
print(f"  KL loss: {loss}")

# Now try with different params
params_test = mx.array([0.0, 0.5])
sample_logprob_test = normal_logpdf(samples, params_test[0], params_test[1])
loss_test = nn.losses.kl_div_loss(sample_logprob_test, target_logprob)
print(f"\nWhen params = [0.0, 0.5] (smaller sigma):")
print(f"  KL loss: {loss_test}")

# Check gradient
def loss_fn(params):
    sample_logprob = normal_logpdf(samples, params[0], params[1])
    target_logprob = normal_logpdf(samples, params_old[0], params_old[1])
    loss = nn.losses.kl_div_loss(sample_logprob, target_logprob)
    return loss

params_reset = mx.array([0.0, 1.0])
lval, lgrad = mx.value_and_grad(loss_fn)(params_reset)
print(f"\nGradient at params = [0.0, 1.0]:")
print(f"  grad: {lgrad}")
print(f"  loss: {lval}")

# Try at a different point
params_test2 = mx.array([0.1, 0.8])
lval2, lgrad2 = mx.value_and_grad(loss_fn)(params_test2)
print(f"\nGradient at params = [0.1, 0.8]:")
print(f"  grad: {lgrad2}")
print(f"  loss: {lval2}")
