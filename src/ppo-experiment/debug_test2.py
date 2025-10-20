import mlx.core as mx
import mlx.nn as nn

# Test: What if we draw samples from params_old instead?
params_old = mx.array([0.0, 1.0])
params = mx.array([0.0, 1.0])

def normal_logpdf(samples, mu, sigma):
    """Compute the log probability density of a normal distribution."""
    return (
        -0.5 * ((samples - mu) / sigma) ** 2
        - mx.log(sigma)
        - 0.5 * mx.log(mx.array(2 * mx.pi))
    )

# Draw samples from params_old (FIXED distribution)
samples = mx.random.normal(loc=params_old[0], scale=params_old[1], shape=(10000,))

def loss_fn(params):
    sample_logprob = normal_logpdf(samples, params[0], params[1])
    target_logprob = normal_logpdf(samples, params_old[0], params_old[1])
    loss = nn.losses.kl_div_loss(sample_logprob, target_logprob)
    return loss

# Gradient when params = params_old
params_test = mx.array([0.0, 1.0])
lval, lgrad = mx.value_and_grad(loss_fn)(params_test)
print(f"Samples from params_old = [0.0, 1.0]")
print(f"Gradient at params = [0.0, 1.0]:")
print(f"  grad: {lgrad}")
print(f"  loss: {lval}")

# Compare: what if samples come from current params?
print(f"\n--- Now compare when samples come from current params ---")

def loss_fn_biased(params):
    # Draw fresh samples from params each time
    samples_biased = mx.random.normal(loc=params[0], scale=params[1], shape=(10000,))
    sample_logprob = normal_logpdf(samples_biased, params[0], params[1])
    target_logprob = normal_logpdf(samples_biased, params_old[0], params_old[1])
    loss = nn.losses.kl_div_loss(sample_logprob, target_logprob)
    return loss

# But wait - we CAN'T differentiate through the sampling!
# This would give zero gradient
# Let's instead show what YOUR code does: sample first, THEN differentiate

for test_params in [[0.0, 1.0], [0.0, 0.8], [0.0, 0.5]]:
    params_test = mx.array(test_params)
    # Sample from THIS specific params value
    samples_from_this = mx.random.normal(loc=params_test[0], scale=params_test[1], shape=(10000,))

    def loss_fn_at_this_point(p):
        sample_logprob = normal_logpdf(samples_from_this, p[0], p[1])
        target_logprob = normal_logpdf(samples_from_this, params_old[0], params_old[1])
        loss = nn.losses.kl_div_loss(sample_logprob, target_logprob)
        return loss

    lval, lgrad = mx.value_and_grad(loss_fn_at_this_point)(params_test)
    print(f"\nSamples drawn from {test_params}:")
    print(f"  Gradient evaluated at {test_params}: {lgrad}")
    print(f"  Loss: {lval}")
