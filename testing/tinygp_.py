#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:14:36 2022

@author: dmilakov
"""

import numpy as np
import matplotlib.pyplot as plt

random = np.random.default_rng(42)

t = np.sort(
    np.append(
        random.uniform(0, 3.8, 28),
        random.uniform(5.5, 10, 18),
    )
)
yerr = random.uniform(0.08, 0.22, len(t))
y = (
    0.2 * (t - 5)
    + np.sin(3 * t + 0.1 * (t - 5) ** 2)
    + yerr * random.normal(size=len(t))
)

true_t = np.linspace(-6, 6, 100)
# true_y = 0.2 * (true_t - 5) + np.sin(3 * true_t + 0.1 * (true_t - 5) ** 2)
#%%
# plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-6, 6)
plt.ylim(0,0.25)
_ = plt.title("simulated data")
#%%

# from tinygp import kernels, GaussianProcess

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen.initializers import zeros

import optax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

prior_sigma = 5.0
t=vel1s_; yerr=err1s_; y=flx1s_

def numpyro_model(t, yerr, y=None):
    mean = numpyro.sample("mean", dist.Normal(0.0, prior_sigma))
    jitter = numpyro.sample("jitter", dist.HalfNormal(prior_sigma))

    sigma1 = numpyro.sample("sigma1", dist.HalfNormal(prior_sigma))
    rho1 = numpyro.sample("rho1", dist.HalfNormal(prior_sigma))
    tau = numpyro.sample("tau", dist.HalfNormal(prior_sigma))
    kernel1 = sigma1**2 * kernels.ExpSquared(tau) #* kernels.Cosine(rho1)

    # sigma2 = numpyro.sample("sigma2", dist.HalfNormal(prior_sigma))
    # rho2 = numpyro.sample("rho2", dist.HalfNormal(prior_sigma))
    # kernel2 = sigma2**2 * kernels.Matern32(rho2)

    kernel = kernel1 #+ kernel2
    gp = GaussianProcess(kernel, t, diag=yerr**2 + jitter, mean=mean)
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)

    if y is not None:
        numpyro.deterministic("pred", gp.condition(y, true_t).gp.loc)


nuts_kernel = NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=1000,
    num_chains=2,
    progress_bar=False,
)
rng_key = jax.random.PRNGKey(34923)
#%%
# %%time
mcmc.run(rng_key, t, yerr, y=y)
samples = mcmc.get_samples()
pred = samples["pred"].block_until_ready()  # Blocking to get timing right
#%%
import arviz as az

data = az.from_numpyro(mcmc)
print(az.summary(
    data, var_names=[v for v in data.posterior.data_vars if v != "pred"]
))
#%%
q = np.percentile(pred, [5, 50, 95], axis=0)
plt.fill_between(true_t, q[0], q[2], color="C0", alpha=0.5, label="inference")
plt.plot(true_t, q[1], color="C0", lw=2)
# plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3, label="truth")

plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("x [day]")
plt.ylabel("y [ppm]")
plt.ylim(0,0.2)
plt.xlim(-6,6)
plt.legend()
_ = plt.title("posterior inference")

#%%

class GaussianMeanFunction(dist.Distribution):
    support = dist.constraints.positive
    def __init__(self, L):
        self.L = L
        super().__init__(batch_shape=jnp.shape(L), event_shape=())

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        mu,sigma,amp = self.L
        return 