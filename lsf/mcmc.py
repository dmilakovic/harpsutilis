#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:50:44 2023

@author: dmilakov
"""
import jax
import jax.numpy as jnp
import jaxopt
from tinygp import kernels, GaussianProcess, transforms, noise
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
from harps.functions import gauss4p
import harps.lsf.read as hread
import harps.lsf.gp as lsfgp

jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist


#%%
npix=4096
od = 50; segm = 9
fname = '/Users/dmilakov/projects/lfc/dataprod/v_2.2/output/2018-12-05_0812.dat'
pixl=npix//16*segm
pixr=npix//16*(segm+1)
# pixl=2235
# pixr=2737
# pixl = 3200
# pixr = 3500
scale = 'pix'
# scale = 'velocity'
X_,Y_,Y_err_,fig = hread.get_data(fname,od,pixl,pixr,scale=scale,fittype='gauss',
                                  plot=True,
                                  filter=None)
x = jnp.array(X_)
y = jnp.array(Y_)
y_err = jnp.array(Y_err_)
x_grid = np.linspace(x.min(), x.max(), 500)
#%%

def model_numpyro_sample(x, y=None, y_err=None):
    # The parameters of the GP model
    mf_loc      = numpyro.sample("mf_loc", dist.Normal(0., 0.1))
    mf_log_sig  = numpyro.sample("mf_log_sig", dist.Normal(0.4,0.1))
    mf_const    = numpyro.sample("mf_const", dist.Normal(0.0,0.1))
    mf_amp      = numpyro.sample("mf_amp", dist.Normal(100.0,5.))
    
    # if y_err is not None:
    #     log_gp_diag = jnp.log(y_err**2)
    # else:
    log_var_add  = numpyro.sample("log_var_add", dist.Normal(-6.,1.))
    gp_log_amp   = numpyro.sample("gp_log_amp", dist.Normal(-0.3,0.1))
    gp_log_scale = numpyro.sample("gp_log_scale", dist.Normal(-0.3,0.1))
    # sys.exit()
    mf_sig      = numpyro.deterministic("mf_sig", jnp.exp(mf_log_sig))
    gp_amp      = numpyro.deterministic("gp_amp", jnp.exp(gp_log_amp))
    gp_scale    = numpyro.deterministic("gp_scale", jnp.exp(gp_log_scale))
    # diag = 1e-5
    theta = dict(
        mf_amp        = mf_amp,
        mf_loc        = mf_loc,
        mf_log_sig    = mf_log_sig,
        mf_const      = mf_const,
        gp_log_amp    = gp_log_amp,#popt[0]/5.,
        gp_log_scale  = gp_log_scale,
        log_var_add   = log_var_add,
    )
    
    # Set up the kernel and GP objects for the LSF
    # print(theta)
    gp_lsf = lsfgp.build_LSF_GP(theta, x, y, y_err)
    
    # LSF model
    inferred_lsf = numpyro.sample("gp_lsf", gp_lsf.numpyro_dist())
    
    # Intrinsic scatter
    
    
    # var_data  = jnp.power(S,2.)
    
    sct_log_amp   = numpyro.sample("sct_log_amp", dist.Normal(-6.0,0.5))
    sct_log_scale = numpyro.sample("sct_log_scale", dist.Normal(1.,0.1))
    sct_log_const  = numpyro.sample("sct_log_const", dist.Uniform(-1.0,1.0))
    
    sct_amp      = numpyro.deterministic("sct_amp", jnp.exp(sct_log_amp))
    sct_scale    = numpyro.deterministic("sct_scale", jnp.exp(sct_log_scale))
    sct_const    = numpyro.deterministic("sct_const", jnp.exp(sct_log_const))
    
    theta_scatter = dict(
        sct_log_const = sct_log_const,
        sct_log_amp = sct_log_amp,
        sct_log_scale = sct_log_scale,
        )
    
    gp_scatter = lsfgp.build_scatter_GP(theta_scatter, x, y_err)
    # # log scatter model
    log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    with numpyro.plate("data",len(x)):
        numpyro.sample("obs", 
                       dist.Normal(inferred_lsf, 
                                   jnp.mean(y_err)
                                   # y_err*jnp.exp(log_inferred_sct/2.)
                                   ),
                       obs=y)
    return None 
# def model_numpyro_param(x, y=None, y_err=None):
#     # The parameters of the GP model
#     mf_loc       = numpyro.param("mf_loc", 0., 
#                                   constraint=dist.constraints.real)
#     log_mf_width = numpyro.param("log_mf_width", 1.,
#                                   constraint=dist.constraints.real)
#     mf_const     = numpyro.param("mf_const", 0., 
#                                   constraint=dist.constraints.real)
#     log_mf_amp   = numpyro.param("log_mf_amp", 0., 
#                                   constraint=dist.constraints.real)
#     # if y_err is not None:
#     #     log_gp_diag = jnp.log(y_err**2)
#     # else:
#     log_gp_diag  = numpyro.param("log_gp_diag", 1e-5, 
#                                   constraint=dist.constraints.real)
#     log_gp_amp   = numpyro.param("log_gp_amp", 0., 
#                                   constraint=dist.constraints.real)
#     log_gp_scale = numpyro.param("log_gp_scale", 0., 
#                                   constraint=dist.constraints.real)
#     # sys.exit()
    
#     # diag = 1e-5
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
#     # print(theta)
#     gp_lsf_ = lsfgp.build_LSF_GP(theta, x, y, y_err)
    
#     # LSF model
#     inferred_lsf = numpyro.sample("gp_lsf", gp_lsf_.numpyro_dist())
    
#     # Intrinsic scatter
#     log_sct_amp   = numpyro.sample("log_sct_amp", dist.HalfNormal(5.0))
#     log_sct_scale = numpyro.sample("log_sct_scale", dist.Uniform(0.0,2.))
#     log_sct_diag  = numpyro.sample("log_sct_diag", dist.Uniform(0.0,2.0))
#     kernel_sct    = jnp.exp(log_sct_amp) * kernels.ExpSquared(log_sct_scale)
    
#     gp_scatter = GaussianProcess(kernel_sct, 
#                                  x, 
#                                  diag=log_sct_diag, 
#                                  mean=0.0)
    
#     # log scatter model
#     log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    
#     numpyro.sample("obs", 
#                    dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
#                    obs=y)
    
# def guide_numpyro(x, y=None, y_err=None):
#     # The parameters of the GP model
#     mf_loc       = numpyro.param("mf_loc_mean", 0., 
#                                  constraint=dist.constraints.real)
#     mf_loc       = numpyro.param("mf_loc_sigma", 0., 
#                                  constraint=dist.constraints.real)
#     log_mf_width = numpyro.param("log_mf_width", 1.,
#                                   constraint=dist.constraints.real)
#     mf_const     = numpyro.param("mf_const", 0., 
#                                  constraint=dist.constraints.real)
#     log_mf_amp   = numpyro.param("log_mf_amp", 0., 
#                                  constraint=dist.constraints.real)
#     # if y_err is not None:
#     #     log_gp_diag = jnp.log(y_err**2)
#     # else:
#     log_gp_diag  = numpyro.param("log_gp_diag", 1e-5, 
#                                  constraint=dist.constraints.real)
#     log_gp_amp   = numpyro.param("log_gp_amp", 0., 
#                                  constraint=dist.constraints.real)
#     log_gp_scale = numpyro.param("log_gp_scale", 0., 
#                                  constraint=dist.constraints.real)
#     # sys.exit()
    
#     # diag = 1e-5
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
#     # print(theta)
#     kernel_lsf = jnp.exp(log_gp_amp) * kernels.ExpSquared(log_gp_scale)
#     gp_lsf_ = GaussianProcess(kernel_lsf, 
#                              x, 
#                              diag=log_gp_diag, 
#                              mean=partial(mean_function, theta))
    
#     # LSF model
#     inferred_lsf = numpyro.sample("gp_lsf", gp_lsf_.numpyro_dist())
    
#     # Intrinsic scatter
#     log_sct_amp   = numpyro.sample("log_sct_amp", dist.HalfNormal(5.0))
#     log_sct_scale = numpyro.sample("log_sct_scale", dist.Uniform(0.0,2.))
#     log_sct_diag  = numpyro.sample("log_sct_diag", dist.Uniform(0.0,2.0))
#     kernel_sct    = jnp.exp(log_sct_amp) * kernels.ExpSquared(log_sct_scale)
    
#     gp_scatter = GaussianProcess(kernel_sct, 
#                                  x, 
#                                  diag=log_sct_diag, 
#                                  mean=0.0)
    
#     # log scatter model
#     log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    
#     numpyro.sample("obs", 
#                    dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
#                    obs=y)

#%%
from numpyro.infer import Predictive
# num_observations = 250
num_prior_samples = 1000
RNG = jax.random.PRNGKey(0)
PRIOR_RNG, MCMC_RNG, PRED_RNG = jax.random.split(RNG, 3)
prior = Predictive(model_numpyro_sample, num_samples=num_prior_samples)

# prior_samples = prior(PRIOR_RNG,x,y,y_err)
prior_samples = prior(PRIOR_RNG,x,y_err=y_err)

#%%
plt.errorbar(x,y,y_err,marker='.',ms=2,capsize=2,ls='',color='red')
label = "Prior predictive samples"
pred_y = prior_samples["gp_lsf"]
for n in np.random.default_rng(0).integers(len(pred_y), size=100):
    plt.plot(x, pred_y[n], ".", color="C0", alpha=0.1, label=label)
    label = None
plt.legend();
#%%
# Run the MCMC
nuts_kernel = numpyro.infer.NUTS(model_numpyro_sample, target_accept_prob=0.9)
mcmc = numpyro.infer.MCMC(
    nuts_kernel,
    num_warmup=20,
    # num_warmup=100,
    num_samples=20,
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
plt.errorbar(x,y,y_err,marker='.',ms=2,capsize=2,ls='',color='red')
label = "Posterior predictive samples"
post_pred_y = posterior_samples["obs"]
for n in np.random.default_rng(0).integers(len(post_pred_y), size=100):
    plt.plot(x, post_pred_y[n], ".", color="C0", alpha=0.1, label=label)
    label = None
plt.legend();
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
    
    lsf_gp = lsfgp.build_LSF_GP(theta, x, y, y_err)
    
    _, lsf_cond = lsf_gp.condition(y,x_grid)
    mean = lsf_cond.mean
    std_from_var  = np.sqrt(lsf_cond.variance)

    sct_gp = lsfgp.build_scatter_GP(theta, x)
    _, sct_cond = sct_gp.condition(y_err,x_grid)
    std_from_sct = sct_cond.mean
    
    if i is not None:
        label=f'Model {i}'
        c=f'C{i}'
    else:
        label='Median parameter model'
        c='C1'
    plt.plot(x_grid,mean,label=label,c=c)
    plt.fill_between(x_grid, mean+std_from_sct, mean-std_from_sct, alpha=0.2,color=c)
    # plt.fill_between(x_grid, mean+std_from_var, mean-std_from_var, alpha=0.2,color=c)
    # plt.plot(X, true_log_rate, "--", color="C1", label="true rate")
    plt.errorbar(x, y, y_err, ls='', marker='.', c='k', label="data")
    plt.legend(loc=2)
    plt.xlabel("x")
    plt.ylabel("Flux")
for i in [None,1,10,100]:
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
    
    
#%%
q = np.percentile(samples["gp_lsf"], [5, 25, 50, 75, 95], axis=0)
plt.errorbar(x,y,y_err,marker='.',ms=2,capsize=2,ls='')
plt.plot(x, q[2], color="C0", label="MCMC inferred model")
plt.fill_between(x, q[0], q[-1], alpha=0.3, lw=0, color="C0")
plt.fill_between(x, q[1], q[-2], alpha=0.3, lw=0, color="C0")

theta = {}
for key in ['gp_log_amp', 'gp_log_scale', 'log_var_add', 
            'mf_amp', 'mf_const', 'mf_loc', 'mf_log_sig',
            'sct_log_amp', 'sct_log_const', 'sct_log_scale']:
    theta[key] = np.percentile(samples[key],50,axis=0)
    
#%% plots    

#%%
theta = get_theta(samples)
x_array, log_var, err_log_var = lsfgp.estimate_variance(x,y,y_err, 
                                                        theta, minpts=15, plot=False)
S, S_var = lsfgp.rescale_errors((theta,x_array,log_var,err_log_var), x, y_err)

gp_sct=lsfgp.build_scatter_GP(theta,x_array,err_log_var)
_,sct_cond=gp_sct.condition(log_var,x_grid)
sct_mean = sct_cond.mean
sct_std = np.sqrt(sct_cond.variance)

plt.errorbar(x_array,log_var,err_log_var,marker='o',ls='',capsize=2,c='C0')
# plt.errorbar(x,S,S_var,marker='o',ls='',capsize=2,c='C2')
plt.plot(x_grid,sct_cond.mean,c='C1')
plt.fill_between(x_grid,sct_mean+sct_std,sct_mean-sct_std,alpha=0.2,color='C1')
#%%
import arviz as az

data = az.from_numpyro(mcmc)
print(az.summary(
    data, var_names=[v for v in data.posterior.data_vars if v != "pred"]
))
#%% Plot corner plot
import arviz as az
data=az.from_dict(samples)
var_names = ["mf_loc","mf_log_sig","mf_amp",
             "gp_log_amp","gp_log_scale",
             "sct_log_amp","sct_log_scale","sct_log_const"]
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
    prior_samples,
    var_names=var_names,
    kind="kde",
    marginals=True,
    point_estimate="median",
    divergences=True,
    textsize=12,
)