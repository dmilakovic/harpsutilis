#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:01:40 2023

@author: dmilakov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:50:44 2023

@author: dmilakov
"""
import numpyro
import numpyro.distributions as dist
numpyro.set_host_device_count(4)

import jax
import jax.numpy as jnp
import jaxopt
from tinygp import kernels, GaussianProcess, transforms, noise
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
# from scipy.optimize import curve_fit
# from harps.functions import gauss4p
# import harps.lsf.read as hread
# import harps.lsf.gp as lsfgp

jax.config.update("jax_enable_x64", True)


#%%

x,y,y_err=np.loadtxt('/Users/dmilakov/projects/lfc/dataprod/data_od50_segm8.txt').T
x_grid = np.linspace(x.min(), x.max(), 500)
#%%
def build_scatter_GP(theta,X,Y_err=None):
    '''
    Returns Gaussian Process for the intrinsic scatter of points (beyond noise)

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    sct_const  = jnp.exp(theta['sct_log_const'])
    sct_amp    = jnp.exp(theta['sct_log_amp'])
    sct_scale  = jnp.exp(theta['sct_log_scale'])
    if Y_err is not None:
        Noise2d = noise.Diagonal(jnp.power(Y_err,2.))
    else:
        Noise2d = noise.Diagonal(jnp.full_like(X,1e-4))
    sct_kernel = sct_amp * kernels.ExpSquared(sct_scale) #+ kernels.Constant(sct_const)
    # sct_kernel = sct_amp * kernels.Matern52(sct_scale) #+ kernels.Constant(sct_const)
    return GaussianProcess(
        sct_kernel,
        X,
        noise= Noise2d,
        mean = sct_const
    )

def build_LSF_GP(theta_lsf,X,Y=None,Y_err=None):
    '''
    Returns a Gaussian Process for the LSF. If scatter is not None, tries to 
    include a second GP for the intrinsic scatter of datapoints beyond the
    error on each individual point.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    Y_err : TYPE
        DESCRIPTION.
    scatter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    gp_amp   = jnp.exp(theta_lsf['gp_log_amp'])
    gp_scale = jnp.exp(theta_lsf["gp_log_scale"])
    kernel = gp_amp * kernels.ExpSquared(gp_scale) # LSF kernel
    # Various variances (obs=observed, add=constant random noise, tot=total)
    var_add = jnp.exp(theta_lsf['log_var_add']) 
    
    try:
        var_data = jnp.power(Y_err,2.)
    except:
        var_data = jnp.full_like(X, 1e-8)
    var_tot = var_data + var_add
    noise2d = jnp.diag(var_tot)
    Noise2d = noise.Dense(noise2d)
    
    
    return GaussianProcess(
        kernel,
        X,
        noise = Noise2d,
        mean=partial(gaussian_mean_function, theta_lsf),
    )

def gaussian_mean_function(theta, X):
    '''
    Returns the Gaussian profile with parameters encapsulated in dictionary
    theta, evaluated a points in X

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    mean  = theta["mf_loc"]
    sigma = jnp.exp(theta["mf_log_sig"])
    gauss = jnp.exp(-0.5 * jnp.square((X - mean)/sigma)) \
            / jnp.sqrt(2*jnp.pi) / sigma
    beta = jnp.array([gauss,1])
    
    return jnp.array([theta['mf_amp'],theta['mf_const']]) @ beta
#%%

def model_numpyro_sample(x, y=None, y_err=None):
    # The parameters of the GP model
    mf_loc       = numpyro.sample("mf_loc", dist.Normal(0., 0.05))
    mf_sig       = numpyro.sample("mf_sig", dist.Normal(1.4,0.1))
    mf_const     = numpyro.sample("mf_const", dist.Normal(0.0,0.1))
    mf_amp       = numpyro.sample("mf_amp", dist.Normal(100.0,5.))
    log_var_add  = numpyro.sample("log_var_add", dist.Uniform(-15.,-2.))
    gp_amp       = numpyro.sample("gp_amp", dist.TruncatedNormal(0.0,0.5,high=10.))
    gp_scale     = numpyro.sample("gp_scale", dist.TruncatedNormal(1.0,0.2,low=0.4))
    
    # mf_sig       = numpyro.deterministic("mf_sig", jnp.exp(mf_sig))
    mf_log_sig   = numpyro.deterministic("mf_log_sig",jnp.log(mf_sig))
    gp_log_amp   = numpyro.deterministic("gp_log_amp", jnp.log(gp_amp))
    gp_log_scale = numpyro.deterministic("gp_log_scale", jnp.log(gp_scale))
    # diag = 1e-5
    theta = dict(
        mf_amp        = mf_amp,
        mf_loc        = mf_loc,
        mf_log_sig    = mf_log_sig,
        mf_const      = mf_const,
        gp_log_amp    = gp_log_amp,
        gp_log_scale  = gp_log_scale,
        log_var_add   = log_var_add,
    )
    
    # Set up the kernel and GP objects for the LSF
    gp_lsf = build_LSF_GP(theta, x, y, y_err)
    
    # LSF model
    inferred_lsf = numpyro.sample("gp_lsf", gp_lsf.numpyro_dist())
    
    # Intrinsic scatter
    sct_amp       = numpyro.sample("sct_amp", dist.TruncatedNormal(1.,1.0,low=0.0,high=3.))
    sct_scale     = numpyro.sample("sct_scale", dist.TruncatedNormal(1.,0.1,low=0.4))
    sct_log_const = numpyro.sample("sct_log_const", dist.Uniform(-10.0,-5.0))
    
    sct_log_amp   = numpyro.deterministic("sct_log_amp", jnp.log(sct_amp))
    sct_log_scale = numpyro.deterministic("sct_log_scale", jnp.log(sct_scale))
    sct_const     = numpyro.deterministic("sct_const", jnp.exp(sct_log_const))
    
    theta_scatter = dict(
        sct_log_const = sct_log_const,
        sct_log_amp = sct_log_amp,
        sct_log_scale = sct_log_scale,
        )
    
    gp_scatter = build_scatter_GP(theta_scatter, x, y_err)
    # # log scatter model
    log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    g = numpyro.deterministic("g", jnp.exp(log_inferred_sct/2.))
    with numpyro.plate("data",len(x)):
        numpyro.sample("obs", 
                       dist.Normal(inferred_lsf, 
                                    # 5*jnp.mean(y_err)
                                    y_err*g
                                   ),
                       obs=y)
    return None 

#%%
from numpyro.infer import Predictive
# num_observations = 250
num_prior_samples = 100
RNG = jax.random.PRNGKey(5765)
PRIOR_RNG, MCMC_RNG, PRED_RNG = jax.random.split(RNG, 3)
prior = Predictive(model_numpyro_sample, num_samples=num_prior_samples)

# prior_samples = prior(PRIOR_RNG,x,y,y_err)
prior_samples = prior(PRIOR_RNG,x,y_err=y_err)

#%%
plt.errorbar(x,y,y_err,marker='.',ms=2,capsize=2,ls='',color='red')
label = "Prior predictive samples"
pred_y = prior_samples["gp_lsf"]
g = prior_samples['g']
for n in np.random.default_rng(0).integers(len(pred_y), size=100):
    plt.plot(x, pred_y[n], ".", color="C0", alpha=0.1, label=label)
    plt.plot(x, g[n], ".", color="C1", alpha=0.1, label=label)
    label = None
plt.legend();
#%% Run the MCMC
nuts_kernel = numpyro.infer.NUTS(model_numpyro_sample, target_accept_prob=0.9)
mcmc = numpyro.infer.MCMC(
    nuts_kernel,
    num_warmup=50,
    # num_warmup=100,
    num_samples=100,
    # num_samples=100,
    num_chains=2,
    progress_bar=True,
)
# rng_key = jax.random.PRNGKey(55875)
mcmc.run(MCMC_RNG, x, y=y, y_err=y_err)
samples = mcmc.get_samples()
#%%
posterior = Predictive(model_numpyro_sample, mcmc.get_samples())
posterior_samples = posterior(PRED_RNG,x, y_err=y_err)
g = np.median(posterior_samples['g'],axis=0)
plt.errorbar(x,y,g*y_err,marker='.',ms=2,capsize=2,ls='',color='red')
plt.errorbar(x,y,y_err,marker='.',ms=2,capsize=4,ls='',color='red',alpha=0.3)
label = "Posterior predictive samples"
post_pred_y = posterior_samples["obs"]
for n in np.random.default_rng(0).integers(len(post_pred_y), size=100):
    plt.plot(x, post_pred_y[n], ".", color="C0", alpha=0.1, label=label)
    label = None
plt.legend();
#%%
prio_pred_y = prior_samples["g"]
post_pred_y = posterior_samples["g"]
for n in np.random.default_rng(0).integers(len(post_pred_y), size=100):
    plt.plot(x, post_pred_y[n], ".", color="C0", alpha=0.1, label=label)
    plt.plot(x, prio_pred_y[n], ".", color="C1", alpha=0.1, label=label)
    label = None
plt.legend()
#%%
def get_theta(samples,i=None):
    theta = {}
    for key in ['gp_log_amp', 'gp_log_scale', 'log_var_add', 
                'mf_amp', 'mf_const', 'mf_loc', 'mf_log_sig',
                'sct_log_amp', 'sct_log_const', 'sct_log_scale']:
        if i is not None:
            theta[key] = samples[key][i]
        else:
            theta[key] = np.percentile(samples[key],50,axis=0)
    return theta 

def plot_mcmc_model(samples,i=None):
    
    theta = get_theta(samples,i)
    
    lsf_gp = build_LSF_GP(theta, x, y, y_err)
    
    # _, lsf_cond = lsf_gp.condition(y,x_grid)
    mean = lsf_gp.mean
    # std_from_var  = np.sqrt(lsf_cond.variance)

    sct_gp = build_scatter_GP(theta, x)
    # _, sct_cond = sct_gp.condition(y_err,x_grid)
    std_from_sct = sct_gp.mean
    
    if i is not None:
        label=f'Model {i}'
        c=f'C{i}'
    else:
        label='Median parameter model'
        c='C1'
    plt.plot(x,mean,label=label,c=c)
    plt.fill_between(x, mean+std_from_sct, mean-std_from_sct, alpha=0.2,color=c)
    # plt.fill_between(x_grid, mean+std_from_var, mean-std_from_var, alpha=0.2,color=c)
    # plt.plot(X, true_log_rate, "--", color="C1", label="true rate")
    plt.errorbar(x, y, y_err, ls='', marker='.', c='k', label="data")
    plt.legend(loc=2)
    plt.xlabel("x")
    plt.ylabel("Flux")
for i in [None,0,10,100]:
    plot_mcmc_model(samples,i=i)

#%%
def print_result(samples):
    medians = {}
    for key,val in samples.items():
        if key not in ['gp_lsf','gp_sct','pred','obs']:
            medians[key] = np.mean(samples[key])
    values = dict(
                A=medians['mf_amp'], 
                mu=medians['mf_loc'],
                sigma=np.exp(medians['mf_log_sig']), 
                y0=medians['mf_const'],
                a=np.exp(medians['gp_log_amp']), 
                l=np.exp(medians['gp_log_scale']), 
                logvar=medians['log_var_add']/np.log(10.),
              )
    for key,val in values.items():
        print(f"{key:<10s} = {val:8.3f}")
print_result(samples)
    
  #%% plots      
#%%
q = np.percentile(samples["gp_lsf"], [5, 25, 50, 75, 95], axis=0)
plt.errorbar(x,y,y_err,marker='.',ms=2,capsize=2,ls='')
plt.plot(x, q[2], color="C0", label="MCMC inferred model")
plt.fill_between(x, q[0], q[-1], alpha=0.3, lw=0, color="C0")
plt.fill_between(x, q[1], q[-2], alpha=0.3, lw=0, color="C0")

#%%


import arviz as az

data = az.from_numpyro(posterior=mcmc,prior=prior_samples)
print(az.summary(
    data, var_names=[v for v in data.posterior.data_vars if v != "pred"]
))
#%% Plot corner plot
import arviz as az
data=az.from_dict(samples)
var_names = ["mf_loc","mf_sig","mf_amp",
             "gp_amp","gp_scale",
             "sct_amp","sct_scale","sct_log_const"]
az.plot_pair(
    data,
    var_names=var_names,
    kind="kde",
    marginals=True,
    point_estimate="median",
    divergences=True,
    textsize=12,
)
#%%
az.plot_pair(
    az.from_dict(prior_samples),
    var_names=var_names,
    kind="kde",
    marginals=True,
    point_estimate="median",
    divergences=True,
    textsize=12,
)
#%%
import corner
inf_data = az.from_numpyro(mcmc)
corner.corner(
    inf_data,
    var_names=var_names,)
