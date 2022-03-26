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

#%%
def get_data(od,N_test=400):
    import harps.lsf as hlsf
    import harps.io as io
    modeller=hlsf.LSFModeller('/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/test.dat',60,70,method='gp',subpix=10,filter=10,numpix=8,iter_solve=1,iter_center=1)

    extensions = ['linelist','flux','background','error','wavereference']
    data, numfiles = io.mread_outfile(modeller._outfile,extensions,701,
                            start=None,stop=None,step=None)
    linelists=data['linelist']
    fluxes=data['flux']
    errors=data['error']
    # backgrounds=data['background']
    backgrounds=None
    wavelengths=data['wavereference']
    orders=np.arange(od,od+1)
    pix3d,vel3d,flx3d,err3d,orders=hlsf.stack('gauss',linelists,fluxes,wavelengths,errors,backgrounds,orders)
    pixl=5000
    pixr=5500

    # pix1s=pix3d[od,pixl:pixr]
    vel1s=vel3d[od,pixl:pixr]
    flx1s=flx3d[od,pixl:pixr]
    err1s=err3d[od,pixl:pixr]

    vel1s_, flx1s_, err1s_ = hlsf.clean_input(vel1s,flx1s,err1s,sort=True,verbose=True,filter=30)
    
    X      = jnp.array(vel1s_)
    Y      = jnp.array(flx1s_*100)
    Y_err  = jnp.array(err1s_*100)
    # Y      = jnp.array([flx1s_,err1s_])
    X_test = jnp.linspace(-5.5, 5.5, N_test)
    return X, Y, Y_err, X_test
X_,Y_,Y_err_,X_test = get_data(65)
#%%
X = X_#[::1]
Y = Y_#[::1]
Y_err = Y_err_#[::1]
#%%
# # And a GP with a general mean function
# def mean_function1(x,mean=0,sigma=1.2,amp=12):
#     return amp * jnp.exp(-jnp.power(((x-mean)/sigma),2.0))

# def mean_function(params, X):
#     mod = jnp.exp(
#         -0.5 * jnp.square((X - params["loc"]) / jnp.exp(params["log_width"]))
#     )
#     beta = jnp.array([1, mod])
#     return params["amps"] @ beta

# mean_params = {
#     "amps": np.array([1, 10]),
#     "loc": 0.0,
#     "log_width": np.log(0.5),
# }

# gp = GaussianProcess(kernel, X, diag=Y_err, mean=partial(mean_function,mean_params))
# y_func = gp.sample(jax.random.PRNGKey(4), shape=(10,))

# # X_grid = np.linspace(-5.5, 5.5, 200)
# model = jax.vmap(partial(mean_function, mean_params))(X_test)

# # Plotting these samples
# _, axes = plt.subplots(1, 1)
# ax = axes
# ax.plot(X, y_func.T, color="k", lw=0.5)
# # ax.plot(X_test, jax.vmap(mean_function1)(X_test), label="mean")
# ax.plot(X_test, jax.vmap(partial(mean_function,mean_params))(X_test), label="mean")
# ax.legend()
# ax.set_xlabel("x")
# _ = ax.set_ylabel("mean function")
# #%%

# _, cond_gp = gp.condition(Y, X_test)

# # The GP object keeps track of its mean and variance, which we can use for
# # plotting confidence intervals
# mu = cond_gp.mean
# std = np.sqrt(cond_gp.variance)
# plt.figure()
# plt.plot(X_test, mu, "C1", label="mean")
# plt.plot(X_test, mu + std, "--C1", label="1-sigma region")
# plt.plot(X_test, mu - std, "--C1")

# # We can also plot samples from the conditional
# y_samp = cond_gp.sample(jax.random.PRNGKey(1), shape=(12,))
# plt.plot(X_test, y_samp[0], "C0", lw=0.5, alpha=0.5, label="samples")
# plt.plot(X_test, y_samp[1:].T, "C0", lw=0.5, alpha=0.5)

# plt.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
# plt.legend(fontsize=10)
# plt.xlim(X_test.min(), X_test.max())
# plt.xlabel("x")
# _ = plt.ylabel("y")
# #%%


# def build_gp(theta, X):

#     # We want most of our parameters to be positive so we take the `exp` here
#     # Note that we're using `jnp` instead of `np`
#     amps = jnp.exp(theta["log_amps"])
#     scales = jnp.exp(theta["log_scales"])

#     # Construct the kernel by multiplying and adding `Kernel` objects
#     k1 = amps[0] * kernels.ExpSquared(scales[0]) # LSF kernel
#     k2 = amps[1] * kernels.ExpSquared(scales[1]) # scatter kernel
    
#     kernel = k1 + k2 

#     return GaussianProcess(
#         kernel, X, diag=jnp.exp(theta["log_diag"]), 
#         mean=mean_function(theta['gaussian_mean'],
#                            theta['gaussian_sigma'],
#                            theta['gaussian_amplitude'])
#     )

# def neg_log_likelihood(theta, X, y):
#     gp = build_gp(theta, X)
#     return -gp.log_probability(y)


# theta_init = {
#     "gaussian_mean": np.float64(0),
#     "gaussian_sigma": np.float64(1),
#     "gaussian_amplitude":np.float64(12),
#     "log_diag": np.log(0.19),
#     "log_amps": np.log([1.0, 0.5]),
#     "log_scales": np.log([1.0,1.0])
# }

# obj = jax.jit(jax.value_and_grad(neg_log_likelihood))

# print(f"Initial negative log likelihood: {obj(theta_init, X, Y)[0]}")
# print(
#     f"Gradient of the negative log likelihood, wrt the parameters:\n{obj(theta_init, X, Y)[1]}"
# )
#%%

# solver = jaxopt.ScipyMinimize(fun=neg_log_likelihood)
# soln = solver.run(theta_init, X=X, y=Y)
# print(f"Final negative log likelihood: {soln.state.fun_val}")

#%%
# gp = build_gp(soln.params, X)
# cond_gp = gp.condition(Y, X_test).gp
# mu, var = cond_gp.loc, cond_gp.variance

# plt.figure()
# plt.plot(X, Y, ".k")
# plt.fill_between(
#     X_test, mu + np.sqrt(var), mu - np.sqrt(var), color="C0", alpha=0.5
# )
# plt.plot(X_test, mu, color="C0", lw=2)
# # We can also plot samples from the conditional
# y_samp = cond_gp.sample(jax.random.PRNGKey(1), shape=(50,))
# plt.plot(X_test, y_samp[0], "C0", lw=0.5, alpha=0.5, label="samples")
# plt.plot(X_test, y_samp[1:].T, "C0", lw=0.5, alpha=0.5)

# plt.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
# plt.legend(fontsize=10)
# plt.xlim(X_test.min(), X_test.max())
# plt.xlabel("x")
# _ = plt.ylabel("y")
# # plt.xlim(t.min(), 2025)
# # plt.xlabel("year")
# # _ = plt.ylabel("CO$_2$ in ppm")

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

def build_gp2(theta):
    # We want most of our parameters to be positive so we take the `exp` here
    # Note that we're using `jnp` instead of `np`
    amps = jnp.exp(theta["log_gp_amps"])
    scales = jnp.exp(theta["log_gp_scales"])

    # Construct the kernel by multiplying and adding `Kernel` objects
    k1 = amps[0] * kernels.ExpSquared(scales[0]) # LSF kernel
    k2 = kernels.ExpSquared(scales[1]) # scatter kernel
    
    kernel = k1 * k2
    return GaussianProcess(
        kernel,
        X,
        diag=jnp.exp(theta["log_gp_diag"]),
        mean=partial(mean_function, theta),
    )





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

@jax.jit
def loss(theta):
    gp = build_gp(theta)
    return -gp.log_probability(Y)
loss(theta)

# rng_key = jax.random.PRNGKey(0)

# from numpyro.infer.util import initialize_model
# init_params, potential_fn_gen, *_ = initialize_model(
#     rng_key,
#     model_numpyro,              # this is your numpyro model
#     model_args=(X,Y),   # add your model arguments here
#     dynamic_args=True,
# )

# solver = jaxopt.ScipyMinimize(fun=potential_fn_gen)
solver = jaxopt.ScipyMinimize(fun=loss)
# solver = jaxopt.GradientDescent(fun=loss)
soln = solver.run(jax.tree_map(jnp.asarray, theta))
print(f"Final negative log likelihood: {soln.state.fun_val}")
#%%
import harps.plotter as hplt

gp = build_gp(soln.params)
_, cond = gp.condition(Y, X_test)

mu = cond.loc
std = np.sqrt(cond.variance)

fig = hplt.Figure2(2,1, figsize=(12,8), height_ratios=[5,1])
ax1 = fig.add_subplot(0,1,0,1)
ax2 = fig.add_subplot(1,2,0,1,sharex=ax1)
# _, (ax1,ax2) = plt.subplots(2, 1,sharex=True)
ax1.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
ax1.plot(X_test, mu, label="Full model")
for i in [1,3]:
    ax1.fill_between(X_test, mu + i*std, mu - i*std, color="C0", alpha=0.3)

# rng_key = jax.random.PRNGKey(55873)
# sampled_functions = cond.sample(rng_key,(20,))
# for f in sampled_functions:
#     ax1.plot(X_test, f, 'k--',alpha=0.1)


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

Xrange = X_test.max() - X_test.min()
ax1.set_xlim(X_test.min()-0.1*Xrange, X_test.max()+0.1*Xrange)
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


def model_numpyro(x, y=None):
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
                                 mean=0)
    
    # log scatter model
    log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    
    numpyro.sample("obs", 
                   dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
                   obs=y)
#%%
# Run the MCMC
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
#%%
q = np.percentile(samples["gp_lsf"], [5, 25, 50, 75, 95], axis=0)
plt.plot(X, q[2], color="C0", label="MCMC inferred model")
plt.fill_between(X, q[0], q[-1], alpha=0.3, lw=0, color="C0")
plt.fill_between(X, q[1], q[-2], alpha=0.3, lw=0, color="C0")
# plt.plot(X, true_log_rate, "--", color="C1", label="true rate")
plt.errorbar(X, Y, Y_err, ls='', marker='.', c='k', label="data")
plt.legend(loc=2)
plt.xlabel("x")
_ = plt.ylabel("counts")
#%%
import arviz as az

data = az.from_numpyro(mcmc)
print(az.summary(
    data, var_names=[v for v in data.posterior.data_vars if v != "pred"]
))
#%% Plot corner plot
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
from numpyro.infer.autoguide import AutoDelta
guide = AutoDelta(model)
optim = numpyro.optim.Adam(0.01)
svi = numpyro.infer.SVI(model, guide, optim, numpyro.infer.Trace_ELBO(10))
results = svi.run(jax.random.PRNGKey(55873), 3000, X, y=Y, progress_bar=True)

#%% Latent kernel

class LatentKernel(kernels.Kernel):
    def __init__(self, kernel1,kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        # self.theta  = theta
    def evaluate(self, X1, X2):
        x1, xerr1 = X1
        x2, xerr2 = X2
        

        # Evaluate the kernel matrix and all of its relevant derivatives
        K1 = self.kernel1.evaluate(x1, x2)
        K2 = self.kernel2.evaluate(x1, x2)
        

        return K1+K2

#%%
