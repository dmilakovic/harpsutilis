#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:27:41 2022

@author: dmilakov
"""

import jax
import jax.numpy as jnp
import jaxopt
from tinygp import kernels, GaussianProcess, transforms
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


jax.config.update("jax_enable_x64", True)

X, Y, Y_err = jnp.array(np.loadtxt('./data.txt').T)
X_test = jnp.linspace(X.min(),X.max(),400)
#%%
def mean_function(theta, X):
    
    gauss = jnp.exp(
        -0.5 * jnp.square((X - theta["mf_loc"]) / jnp.exp(theta["log_mf_width"]))
        # -0.5 * jnp.square((X - theta["mf_loc"]) / theta["log_mf_width"])
    )
    
    beta = jnp.array([1, gauss])
    return jnp.array([theta["mf_const"],jnp.exp(theta["log_mf_amp"])]) @ beta
    # return theta["mf_amps"] * mod

def build_gp(theta):
    # We want most of our parameters to be positive so we take the `exp` here
    # Note that we're using `jnp` instead of `np`
    amp   = jnp.exp(theta["log_gp_amp"])
    scale = jnp.exp(theta["log_gp_scale"])

    # Construct the kernel by multiplying and adding `Kernel` objects
    k1 = amp * kernels.ExpSquared(scale) # LSF kernel
    # k2 = amps[1] * transforms.Transform(jnp.exp, kernels.ExpSquared(scales[1])) # scatter kernel
    
    kernel = k1 #+ k2 
    # kernel = LatentKernel(k1, k2)
    return GaussianProcess(
        kernel,
        X,
        diag=jnp.exp(theta["log_gp_diag"]),
        mean=partial(mean_function, theta),
    )


@jax.jit
def loss(theta):
    gp = build_gp(theta)
    return -gp.log_probability(Y)

mean_params = {
    "mf_const":1.0,
    "log_mf_amp":np.log(10.),
    # "mf_amps":10.,
    "mf_loc": 0.0,
    # "log_mf_width": np.log(0.5),
    "log_mf_width": 0.5,
}
theta = dict(
    # log_gp_amp=np.log(0.1),
    # log_gp_scale=np.log(3.0),
    # log_gp_diag=np.log(0.05),
    log_gp_diag=jnp.log(Y_err**2),
    log_gp_amp=np.array(1.),
    log_gp_scale=np.array(1.),
    **mean_params
)
loss(theta)

solver = jaxopt.ScipyMinimize(fun=loss)
# solver = jaxopt.GradientDescent(fun=loss)
soln = solver.run(jax.tree_map(jnp.asarray, theta))
print(f"Final negative log likelihood: {soln.state.fun_val}")
#%%
gp = build_gp(soln.params)
_, cond = gp.condition(Y, X_test)

mu = cond.loc
std = np.sqrt(cond.variance)
_, (ax1,ax2) = plt.subplots(2, 1,sharex=True)
ax1.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
ax1.plot(X_test, mu, label="Full model")
for i in [1,3]:
    ax1.fill_between(X_test, mu + i*std, mu - i*std, color="C0", alpha=0.3)

rng_key = jax.random.PRNGKey(55873)
sampled_functions = cond.sample(rng_key,(20,))
for f in sampled_functions:
    ax1.plot(X_test, f, 'k--',alpha=0.1)


# Separate mean and GP
_, cond_nomean = gp.condition(Y, X_test, include_mean=False)

mu_nomean = cond_nomean.loc #+ soln.params["mf_amps"][0] # second term is for nicer plot
std_nomean = np.sqrt(cond_nomean.variance)

# plt.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
ax1.plot(X_test, mu_nomean, c='C0', ls='--', label="GP model")
for i in [1,3]:
    ax1.fill_between(X_test, mu_nomean + i*std_nomean, mu_nomean - i*std_nomean,
                     color="C0", alpha=0.3)
ax1.plot(X_test, jax.vmap(gp.mean_function)(X_test), c="C1", label="Gaussian model")

# Plot residuals
_, cond_predict = gp.condition(Y, X, include_mean=True)
mu_predict = cond_predict.loc # second term is for nicer plot
std_predict = np.sqrt(cond_predict.variance)


Y_tot = jnp.sqrt(std_predict**2 + Y_err**2)
rsd = (mu_predict-Y)/Y_tot
ax2.scatter(X,rsd,marker='.',c='grey')

ax1.set_xlim(X_test.min(), X_test.max())
# ax1.set_ylim(-0.05,0.25)
ax2.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_ylabel("Norm. resids")
_ = ax1.legend()

#%%
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)


def model(x, y=None):
    # The parameters of the GP model
    mf_loc       = numpyro.sample("mf_loc", dist.Normal(0.0, 2.0))
    log_mf_width = numpyro.sample("log_mf_width", dist.HalfNormal(2.0))
    mf_const     = numpyro.sample("mf_const", dist.Uniform(0.0,10.0))
    log_mf_amp   = numpyro.sample("log_mf_amp", dist.HalfNormal(10.0))
    log_gp_diag  = numpyro.sample("log_gp_diag", dist.HalfNormal(5.0))
    log_gp_amp   = numpyro.sample("log_gp_amp", dist.HalfNormal(5.0))
    log_gp_scale = numpyro.sample("log_gp_scale", dist.Uniform(0.0,2.0))
    # sys.exit()
    
    # diag = 1e-5
    theta = dict(
        mf_loc=mf_loc,
        log_mf_width=log_mf_width,
        mf_const=mf_const,
        log_mf_amp=log_mf_amp,
        log_gp_diag=log_gp_diag,
        log_gp_amps=log_gp_amp,
        log_gp_scale=log_gp_scale,
    )
    # Set up the kernel and GP objects for the LSF
    kernel_lsf = jnp.exp(log_gp_amp) * kernels.ExpSquared(log_gp_scale)
    gp_lsf = GaussianProcess(kernel_lsf, 
                             x, 
                             diag=log_gp_diag, 
                             mean=partial(mean_function, theta))
    
    # LSF model
    inferred_lsf = numpyro.sample("gp_lsf", gp_lsf.numpyro_dist())
    
    # Intrinsic scatter
    log_sct_amp   = numpyro.sample("log_sct_amp", dist.HalfNormal(5.0))
    log_sct_scale = numpyro.sample("log_sct_scale", dist.Uniform(0.0,2.))
    log_sct_diag  = numpyro.sample("log_sct_diag", dist.Uniform(0.0,2.0))
    kernel_sct    = jnp.exp(log_sct_amp) * kernels.ExpSquared(log_sct_scale)

    gp_scatter = GaussianProcess(kernel_sct, 
                                 x, 
                                 diag=log_sct_diag, 
                                 mean=0.)
    
    # log scatter model
    log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    
    numpyro.sample("obs", 
                   dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
                   obs=y)
#%% Run the MCMC
nuts_kernel = numpyro.infer.NUTS(model, target_accept_prob=0.9)
mcmc = numpyro.infer.MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=1000,
    num_chains=2,
    progress_bar=False,
)
rng_key = jax.random.PRNGKey(55873)
mcmc.run(rng_key, X, y=Y)
samples = mcmc.get_samples()
#%% Plot MCMC results
q = np.percentile(samples["gp_lsf"], [5, 25, 50, 75, 95], axis=0)
plt.plot(X, q[2], color="C0", label="MCMC inferred model")
plt.fill_between(X, q[0], q[-1], alpha=0.3, lw=0, color="C0")
plt.fill_between(X, q[1], q[-2], alpha=0.3, lw=0, color="C0")
# plt.plot(X, true_log_rate, "--", color="C1", label="true rate")
plt.errorbar(X, Y, Y_err, ls='', marker='.', c='k', label="data")
plt.legend(loc=2)
plt.xlabel("x")
_ = plt.ylabel("counts")
#%% Plot corner plot
import arviz as az

data=az.from_dict(samples)
var_names = ["mf_loc", "log_mf_width", "mf_const", "log_mf_amp",
             "log_gp_amp","log_gp_scale","log_sct_amp","log_sct_scale"]
az.plot_pair(
    data,
    var_names=var_names,
    kind="kde",
    marginals=True,
    point_estimate="median",
    divergences=True,
    textsize=12,
)
#%%  SVI using Autodelta
# import torch
from numpyro.infer.autoguide import AutoDelta
from torch.distributions import constraints

def model_2(x, y=None):
    # The parameters of the GP model
    mf_loc       = jnp.float64(0.0)
    log_mf_width = jnp.float64(0.5)
    mf_const     = jnp.float64(0.0)
    log_mf_amp   = jnp.float64(3.0)
    log_gp_diag  = jnp.float64(-2.0)
    log_gp_amp   = jnp.float64(3.0)
    log_gp_scale = jnp.float64(1.0)
    log_sct_amp   = jnp.float64(1.0)
    log_sct_scale = jnp.float64(1.0)
    log_sct_diag  = jnp.float64(-2.0)
    # sys.exit()
    
    # diag = 1e-5
    theta = dict(
        mf_loc=mf_loc,
        log_mf_width=log_mf_width,
        mf_const=mf_const,
        log_mf_amp=log_mf_amp,
        log_gp_diag=log_gp_diag,
        log_gp_amps=log_gp_amp,
        log_gp_scale=log_gp_scale,
    )
    # Set up the kernel and GP objects for the LSF
    kernel_lsf = jnp.exp(log_gp_amp) * kernels.ExpSquared(log_gp_scale)
    gp_lsf = GaussianProcess(kernel_lsf, 
                             x, 
                             diag=log_gp_diag, 
                             mean=partial(mean_function, theta))
    
    # LSF model
    inferred_lsf = numpyro.sample("gp_lsf", gp_lsf.numpyro_dist())
    
    # Intrinsic scatter
    
    kernel_sct    = jnp.exp(log_sct_amp) * kernels.ExpSquared(log_sct_scale)

    gp_scatter = GaussianProcess(kernel_sct, 
                                 x, 
                                 diag=log_sct_diag, 
                                 mean=0.)
    
    # log scatter model
    log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    
    numpyro.sample("obs", 
                   dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
                   obs=y)
    
# def guide(x, y=None):
#     # register the two variational parameters with Pyro
#     # - both parameters will have initial value 15.0.
#     # - because we invoke constraints.positive, the optimizer
#     # will take gradients on the unconstrained parameters
#     # (which are related to the constrained parameters by a log)
#     mf_loc       = numpyro.param("mf_loc", jnp.float64(0.0))
#     log_mf_width = numpyro.param("log_mf_width", jnp.float64(0.5))
#     mf_const     = numpyro.param("mf_const", 0.0)
#     log_mf_amp   = numpyro.param("log_mf_amp", 10.0)
#     log_gp_diag  = numpyro.param("log_gp_diag", 1e-5)
#     log_gp_amp   = numpyro.param("log_gp_amp", 0.0)
#     log_gp_scale = numpyro.param("log_gp_scale",0.5)
#     # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
#     log_sct_amp   = numpyro.sample("log_sct_amp", dist.HalfNormal(5.0))
#     log_sct_scale = numpyro.sample("log_sct_scale", dist.Uniform(0.0,2.))
#     log_sct_diag  = numpyro.sample("log_sct_diag", dist.Uniform(0.0,2.0))
    

    
#     theta = dict(
#         mf_loc=mf_loc,
#         log_mf_width=log_mf_width,
#         mf_const=mf_const,
#         log_mf_amp=log_mf_amp,
#         log_gp_diag=log_gp_diag,
#         log_gp_amps=log_gp_amp,
#         log_gp_scale=log_gp_scale,
#     )
#     # Set up the kernel and GP objects for the LSF
#     kernel_lsf = jnp.exp(log_gp_amp) * kernels.ExpSquared(log_gp_scale)
#     kernel_sct    = jnp.exp(log_sct_amp) * kernels.ExpSquared(log_sct_scale)
    
    
#     gp_lsf = GaussianProcess(kernel_lsf, 
#                              x, 
#                              diag=log_gp_diag, 
#                              mean=partial(mean_function, theta))
#     gp_scatter = GaussianProcess(kernel_sct, 
#                                  x, 
#                                  diag=log_sct_diag, 
#                                  mean=0.)
    
#     log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
#     # LSF model
#     inferred_lsf = numpyro.sample("gp_lsf", gp_lsf.numpyro_dist())
#     numpyro.sample("lsf_model", 
#                    dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
#                    obs=y)
    
#     # numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))


# data = Y
guide = AutoDelta(model)
optim = numpyro.optim.Adam(0.01)
svi = numpyro.infer.SVI(model, guide, optim, numpyro.infer.Trace_ELBO(10))
results = svi.run(jax.random.PRNGKey(55873), 3000, X, y=Y, progress_bar=True)

