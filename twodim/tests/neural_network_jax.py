import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax.scipy.signal import convolve2d
from jax.scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import flax.linen as nn
import optax

def top_hat_2d(x, y, x0, y0, wx, wy):
    return jnp.where((jnp.abs(x - x0) <= wx / 2) & (jnp.abs(y - y0) <= wy / 2), 1, 0)

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y):
    return jnp.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

def convolved_model(xy, x0, y0, sigma_x, sigma_y, wx, wy, a, b, c):
    x, y = xy
    dx, dy = x[1, 0] - x[0, 0], y[0, 1] - y[0, 0]  # Grid spacing
    
    top_hat = top_hat_2d(x, y, x0, y0, wx, wy)
    gaussian = gaussian_2d(x, y, x0, y0, sigma_x, sigma_y)
    
    gaussian /= jnp.sum(gaussian) * dx * dy
    convolved = convolve2d(top_hat, gaussian, mode='same', boundary='fill', fillvalue=0)
    
    convolved *= 1 / jnp.sum(convolved) * dx * dy
    
    background = a * x + b * y + c
    
    return convolved.ravel() + background.ravel()

def fit_convolved_model(x, y, data):
    x_grid, y_grid = jnp.meshgrid(x, y, indexing='ij')
    xy = (x_grid, y_grid)
    
    def loss(params):
        return jnp.sum((convolved_model(xy, *params) - data.ravel()) ** 2)
    
    initial_guess = jnp.array([jnp.mean(x), jnp.mean(y), 1.0, 1.0, 2.0, 2.0, 0, 0, jnp.mean(data)])
    result = minimize(loss, initial_guess, method='BFGS')
    
    return result.x

def generate_synthetic_data(size=1000, grid_size=(32, 32)):
    key = jax.random.PRNGKey(0)
    x = jnp.linspace(-5, 5, grid_size[0])
    y = jnp.linspace(-5, 5, grid_size[1])
    x_grid, y_grid = jnp.meshgrid(x, y, indexing='ij')
    data = []
    labels = []
    sigma_center = 0.2
    
    for i in range(size):
        key, subkey = jax.random.split(key)
        x0, y0 = jax.random.normal(subkey, (2,)) * sigma_center
        key, subkey = jax.random.split(key)
        sigma_x, sigma_y = jax.random.uniform(subkey, (2,), minval=0.5, maxval=2)
        key, subkey = jax.random.split(key)
        wx, wy = jax.random.uniform(subkey, (2,), minval=0.5, maxval=3)
        key, subkey = jax.random.split(key)
        a, b, c = jax.random.uniform(subkey, (3,), minval=-1e-3, maxval=1e-3)
        synthetic_image = convolved_model((x_grid, y_grid), x0, y0, sigma_x, sigma_y, wx, wy, a, b, c).reshape(grid_size)
        
        data.append(synthetic_image)
        labels.append([x0, y0, sigma_x, sigma_y, wx, wy])
    
    return jnp.array(data)[..., jnp.newaxis], jnp.array(labels)

class NeuralNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=6)(x)  # 6 output parameters
        return x

def test_model(test_size=100, grid_size=(32, 32)):
    test_data, test_labels = generate_synthetic_data(size=test_size, grid_size=grid_size)
    
    predictions = []
    for i in range(test_size):
        pred = fit_convolved_model(jnp.linspace(-5, 5, grid_size[0]), jnp.linspace(-5, 5, grid_size[1]), test_data[i].squeeze())
        predictions.append(pred)
    predictions = jnp.array(predictions)
    
    mse = jnp.mean((predictions - test_labels) ** 2, axis=0)
    print("Mean Squared Error for each parameter:", mse)
    
    for i in range(5):
        print(f"Test {i+1} - True: {test_labels[i]}, Predicted: {predictions[i]}")
    
    return mse

if __name__ == "__main__":
    grid_size = (32, 32)
    x = jnp.linspace(-5, 5, grid_size[0])
    y = jnp.linspace(-5, 5, grid_size[1])
    test_data, test_labels = generate_synthetic_data(size=10, grid_size=grid_size)
    
    for i in range(len(test_data)):
        pred_params = fit_convolved_model(x, y, test_data[i].squeeze())
        print(f"Sample {i+1}: True Params: {test_labels[i]}, Predicted: {pred_params}")