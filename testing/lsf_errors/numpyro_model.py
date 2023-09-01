#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:55:28 2023

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
from scipy.optimize import curve_fit
from harps.functions import gauss4p
import harps.lsf.read as hread
import harps.lsf.gp as lsfgp

jax.config.update("jax_enable_x64", True)


#%%
filepath = '/Users/dmilakov/projects/lfc/numpyro/errors/line_od50_1161-1172.dat'
# filepath =  '/Users/dmilakov/projects/lfc/numpyro/errors/order50.txt'
x, y, bkg = np.loadtxt(filepath)[:1024].T

#%% model
def model(x,y=None,bkg=None):
    with numpyro.plate("data",len(x)):
        f_i = numpyro.sample('f_i',
                             dist.Uniform(0,5e5))
                             # dist.TruncatedNormal(4e5,np.sqrt(4e5),low=1.0))
        b_i = numpyro.sample('b_i',
                             dist.Uniform(0,5e4))
        #                      dist.TruncatedNormal(2e2,np.sqrt(2e2),low=1.0))
        # obs_f = numpyro.sample("obs_f",dist.Poisson(f_i))
        obs_b = numpyro.sample("obs_b",dist.Poisson(b_i),obs=bkg)
        obs_f = numpyro.sample("obs_f",dist.Poisson(f_i+b_i),obs=y)
        
    f_star = numpyro.deterministic('f_star',jnp.sum(f_i))
    with numpyro.plate("data",len(x)):
        ratio = numpyro.deterministic('psi',(f_i/f_star))
    return

#%%
from numpyro.infer import Predictive
# num_observations = 250
num_prior_samples = 100
RNG = jax.random.PRNGKey(5765)
PRIOR_RNG, MCMC_RNG, PRED_RNG = jax.random.split(RNG, 3)
prior = Predictive(model, num_samples=num_prior_samples)

# prior_samples = prior(PRIOR_RNG,x,y,y_err)
prior_samples = prior(PRIOR_RNG,x,bkg=bkg)

#%%
plt.errorbar(x,y,np.sqrt(y),marker='.',ms=2,capsize=6)
label = "Prior predictive samples"
pred_f = prior_samples["obs_f"]
pred_b = prior_samples['obs_b']
for n in np.random.default_rng(0).integers(len(pred_f), size=100):
    plt.plot(x, pred_f[n], ".", color="C0", alpha=0.1, label=label)
    plt.plot(x, pred_b[n], ".", color="C1", alpha=0.1, label=label)
    label = None
plt.legend();
#%% Run the MCMC
nuts_kernel = numpyro.infer.NUTS(model, target_accept_prob=0.9)
mcmc = numpyro.infer.MCMC(
    nuts_kernel,
    num_warmup=200,
    # num_warmup=100,
    num_samples=1000,
    # num_samples=100,
    num_chains=4,
    progress_bar=True,
)
# rng_key = jax.random.PRNGKey(55875)
mcmc.run(MCMC_RNG, x, y=y, bkg=bkg)
samples = mcmc.get_samples()
mcmc.print_summary()
#%%
posterior = Predictive(model, mcmc.get_samples())
posterior_samples = posterior(PRED_RNG,x, bkg=bkg)
# post_pred_y = np.median(posterior_samples['obs_f'],axis=0)
# plt.errorbar(x,y,g*y_err,marker='.',ms=2,capsize=2,ls='',color='red')
# plt.errorbar(x,y,y_err,marker='.',ms=2,capsize=4,ls='',color='red',alpha=0.3)
plt.errorbar(x,y,np.sqrt(y),marker='.',ms=2,capsize=6,c='red')
label = "Posterior predictive samples"
post_pred_y = posterior_samples["obs_f"]
post_pred_bkg = posterior_samples["obs_b"]
for n in np.random.default_rng(0).integers(len(post_pred_y), size=100):
    plt.plot(x, post_pred_y[n], ".", color="C0", alpha=0.1, label=label)
    plt.plot(x, post_pred_bkg[n], ".", color="C1", alpha=0.1, label=label)
    label = None
plt.legend();
#%%
import arviz as az
import corner
var_names = ['ratio','f_star']
inf_data = az.from_numpyro(mcmc)
corner.corner(
    inf_data,
    var_names=var_names,)
#%%
data=az.from_dict(posterior_samples)
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
f_star = np.sum(y-bkg)
psi_mcmc = np.mean(samples['psi'],axis=0)
sig_mcmc = np.std(samples['psi'],axis=0)

psi_naive = (y-bkg)/f_star
var_naive = 1./f_star**2 * (y + bkg)
sig_naive = np.sqrt(var_naive)

psi_theor = (y-bkg)/f_star
var_theor = psi_theor*(1-psi_theor)/np.sum(y)
sig_theor = np.sqrt(var_theor)
labels = ['psi_mcmc    ',
          'v_mcmc      ',
          'psi_naive   ',
          'v_naive     ',
          'dist.(abs)  ',
          'dist.(sig)  ',
          'psi diff    ',
          'v diff      ']
print(''.join(labels))
for i in range(len(x)):
    p_mcmc = psi_mcmc[i]
    s_mcmc = sig_mcmc[i]
    p_naive = psi_naive[i]
    s_naive = sig_naive[i]
    
    
    
    distance = np.abs(p_mcmc-p_naive)
    residuals = distance/s_mcmc
    print(f'{p_mcmc:<12.3e}'+\
          f'{s_mcmc:<12.3e}'+\
          f'{p_naive:<12.3e}'+\
          f'{s_naive:<12.3e}'+\
          f'{distance:<12.3e}'+\
          f'{residuals:<12.3f}'+\
          f'{np.abs(p_naive/p_mcmc-1):<12.3%}'+\
          f'{np.abs(s_naive/s_mcmc-1):<12.3%}'
          )
#%%
fig, (ax1,ax2) = plt.subplots(1,2,sharex=True)

ax1.errorbar(np.arange(len(x)),psi_mcmc,sig_mcmc,label='MCMC',
             marker='o',capsize=2)
ax1.errorbar(np.arange(len(x)),psi_naive,sig_naive,label='implemented',
             marker='s',capsize=2)
ax1.errorbar(np.arange(len(x)),psi_theor,sig_theor,label='theoretical',
             marker='v',capsize=2)

ax2.scatter(np.arange(len(x)),psi_mcmc/sig_mcmc,label='MCMC',marker='o')
ax2.scatter(np.arange(len(x)),psi_naive/sig_naive,label='implemented',marker='s')
ax2.scatter(np.arange(len(x)),psi_theor/sig_theor,label='theoretical',marker='v')
ax2.scatter(np.arange(len(x)),(y-bkg)/np.sqrt(y+bkg),label='data',marker='*')
ax1.legend()
# plt.scatter(x,(y-bkg)/np.sum(y-bkg),label='data')
#%% Covariances between first centralised moments of 
pixl = 150
pixr = pixl+50
X_ = samples['psi'][:,pixl:pixr]
X = (X_-np.mean(X_,axis=0))/np.std(X_,axis=0)
Y_ = samples['f_i'][:,pixl:pixr]
Y  = (Y_-np.mean(Y_,axis=0))/np.std(Y_,axis=0)
plt.hist2d(np.ravel(X),np.ravel(Y),bins=50,range=((-5,5),(-5,5)),cmap='Greys')
