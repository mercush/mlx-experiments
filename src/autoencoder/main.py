import mlx.core as mx
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlx.optimizers as optim
import mlx.nn as nn

BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 1000
mx.random.seed(500)

def sample(num_samples=BATCH_SIZE):
    r = mx.random.uniform(shape=(num_samples, 1))
    return mx.concat([mx.cos(2 * mx.pi * r), mx.sin(2 * mx.pi * r)], axis=1)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def __call__(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# Training loop
def train(model, optimizer):
    for _ in tqdm(range(EPOCHS)):
        x = sample()
        
        def loss_fn(model):
            x_recon = model(x)
            return mx.mean(mx.square(x - x_recon))
        
        _, grads = nn.value_and_grad(model, loss_fn)(model)
        
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

def generate_embedding_from_angle(inputs, model):
    circle_points = mx.concat([mx.cos(2 * mx.pi * inputs), mx.sin(2 * mx.pi * inputs)], axis=1)
    encoded = model.encoder(circle_points)
    return encoded

def viz_data():
    x = sample(100)
    plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 0].tolist(), x[:, 1].tolist(), label='Original Data', alpha=0.5)
    plt.legend()
    plt.show()

def viz_encoding(model):
    inputs = mx.linspace(0, 1, 1000).reshape((-1, 1))
    encoded = generate_embedding_from_angle(inputs, model)
    plt.figure(figsize=(8, 8))
    plt.scatter(inputs.tolist(), encoded.tolist(), label='Encoding', alpha=0.5)
    plt.legend()
    plt.show()

def viz_decoder(model):
    inputs = mx.linspace(0, 1, 1000).reshape((-1, 1))
    encoded = generate_embedding_from_angle(inputs, model)
    bounds = mx.min(encoded).item(), mx.max(encoded).item()
    inputs = mx.linspace(bounds[0], bounds[1], 1000).reshape((-1, 1))
    decoded = model.decoder(inputs)
    plt.figure(figsize=(8, 8))
    plt.scatter(decoded[:, 0].tolist(), decoded[:, 1].tolist(), label='Reconstructed with embedding only', alpha=0.5)
    plt.legend()
    plt.show()

def viz_reconstruction(model):
    inputs = sample(1000)
    decoded = model(inputs)
    plt.figure(figsize=(8, 8))
    plt.scatter(decoded[:, 0].tolist(), decoded[:, 1].tolist(), label='Reconstructed', alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    model = Autoencoder()
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    train(model, optimizer)
    viz_data()
    viz_encoding(model)
    viz_reconstruction(model)
    viz_decoder(model)
