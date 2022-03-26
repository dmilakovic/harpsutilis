#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:11:36 2022

@author: dmilakov
"""

import jax
import numpy as np
import jax.numpy as jnp
from tinygp import kernels

jax.config.update("jax_enable_x64", True)


ndim = 3
X = np.random.default_rng(1).normal(size=(10, ndim))

# This kernel is equivalent...
scale = 1.5
kernel1 = kernels.Matern32(scale)

# ... to manually scaling the input coordinates
kernel0 = kernels.Matern32()
np.testing.assert_allclose(kernel0(X / scale, X / scale), kernel1(X, X))
#%%
import numpy as np
import matplotlib.pyplot as plt

random = np.random.default_rng(48392)
X = random.uniform(-5, 5, (100, 2))
yerr = 0.1
y = np.sin(X[:, 0]) * np.cos(X[:, 1] + X[:, 0]) + yerr * random.normal(
    size=len(X)
)

# For plotting predictions on a grid
x_grid, y_grid = np.linspace(-5, 5, 100), np.linspace(-5, 5, 50)
x_, y_ = np.meshgrid(x_grid, y_grid)
y_true = np.sin(x_) * np.cos(x_ + y_)
X_pred = np.vstack((x_.flatten(), y_.flatten())).T

# For plotting covariance ellipses
theta = np.linspace(0, 2 * np.pi, 500)[None, :]
ellipse = 0.5 * np.concatenate((np.cos(theta), np.sin(theta)), axis=0)

plt.figure(figsize=(6, 6))
plt.pcolor(x_grid, y_grid, y_true, vmin=y_true.min(), vmax=y_true.max())
plt.scatter(
    X[:, 0], X[:, 1], c=y, ec="black", vmin=y_true.min(), vmax=y_true.max()
)
plt.xlabel("x")
plt.ylabel("y")
_ = plt.title("data")
#%%
import jaxopt
from tinygp import GaussianProcess, kernels, transforms


def train_gp(nparams, build_gp_func):
    @jax.jit
    def loss(params):
        return -build_gp_func(params).log_probability(y)

    params = {
        "log_amp": np.float64(0.0),
        "log_scale": np.zeros(nparams),
    }
    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(params)
    return build_gp_func(soln.params)

def build_gp_corr(params):
    kernel = jnp.exp(params["log_amp"]) * transforms.Cholesky.from_parameters(
        jnp.exp(params["log_scale"][:2]),
        params["log_scale"][2:],
        kernels.ExpSquared(),
    )
    return GaussianProcess(kernel, X, diag=yerr**2)


corr_gp = train_gp(3, build_gp_corr)

y_pred = corr_gp.condition(y, X_pred).gp.loc.reshape(y_true.shape)
xy = corr_gp.kernel.kernel2.factor @ ellipse

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
axes[0].plot(xy[0], xy[1], "--k", lw=0.5)
axes[0].pcolor(x_, y_, y_pred, vmin=y_true.min(), vmax=y_true.max())
axes[0].scatter(
    X[:, 0], X[:, 1], c=y, ec="black", vmin=y_true.min(), vmax=y_true.max()
)
axes[1].pcolor(x_, y_, y_pred - y_true, vmin=y_true.min(), vmax=y_true.max())
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("correlated kernel")
axes[1].set_xlabel("x")
_ = axes[1].set_title("residuals")