#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:39:11 2023

A collection of code used to make tests on the proper calculation of scatter

@author: dmilakov
"""

#%%
import numpy as np
import jax
import jax.numpy as jnp
import harps.lsf as hlsf
import harps.lsf.gp as gp
import harps.lsf.read as read
import matplotlib.pyplot as plt
# from .2023-01-18 import get_data
#%%
#%%

ftype='HARPS'
# ftype='ESPRESSO'
if ftype == 'HARPS':
    npix = 4096
    od = 50
    # fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/harps/2015-04-17_1440.dat'
    fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_2.0/2018-12-05_0812.dat'

if ftype == 'ESPRESSO':
    npix = 9111
    od = 120
    fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/single.dat'


seg = 12
pixl=npix//16*seg
pixr=npix//16*(seg+1)
# pixl=2235
# pixr=2737
# pixl = 3200
# pixr = 3500
X_,Y_,Y_err_,_ = read.get_data(fname,od,pixl,pixr,scale='pix',fittype='gauss',
                               filter=None)
X = jnp.array(X_)
Y = jnp.array(Y_)
Y_err = jnp.array(Y_err_)

# plt.figure()
# plt.errorbar(X,Y,Y_err,ls='',marker='.',color='grey')
#%%
LSF_solution = gp.train_LSF_tinygp(X,Y,Y_err,scatter=None)
#%%
# logvar_x,logvar_y,logvar_err=gp.estimate_variance_bin(X,Y,Y_err,LSF_solution,minpts=10,plot=True)
xx, log_sam_err, log_sam_err_err = gp.estimate_excess_error(X,Y,Y_err,LSF_solution,10,True)
#%%
scatter = gp.train_scatter_tinygp(X,Y,Y_err,LSF_solution,include_error=False)
scatter_pars, logvar_x, logvar_y, logvar_y_err = scatter
#%%
plt.figure()
# var_x, var_y, var_y_err = logvar_x,logvar_y,logvar_err
var_y, var_y_err = gp.log2lin(logvar_y,logvar_y_err)
plt.errorbar(logvar_x,var_y,var_y_err,ls='',marker='x',capsize=2)
sct_gp = gp.build_scatter_GP(scatter_pars,logvar_x,Y_err=var_y_err)
_,cond_sct_gp = sct_gp.condition(logvar_y,logvar_x)
predict_logy = cond_sct_gp.mean
predict_logy_err = np.sqrt(cond_sct_gp.mean)

X_grid = np.linspace(X.min(),X.max(), 100)
_, cond_sct_grid = sct_gp.condition(logvar_y,X_grid)
log_f_mean  = cond_sct_grid.mean
log_f_sigma = jnp.sqrt(cond_sct_grid.variance)

f, f_sigma = gp.log2lin(log_f_mean, log_f_sigma)

S, S_var = gp.rescale_errors(scatter,X,Y_err,plot=True)

# # S = f * Y_err
# S = f * Y_err

# def _evaluate(x):
#     value = sct_gp.condition(logvar_y,jnp.atleast_1d(x))[1].mean
#     return value[0]
# deriv = jax.grad(_evaluate)
# dfdx  = deriv(X)
# Variance on S
# S_var = S**2 * sct_cond.variance
# S_var = Y_err**2 * f_sigma**2 
# S_var = S**2 * dfdx**2  * f_sigma**2
# plt.scatter(X, S/Y_err,marker='o',c='r')
plt.errorbar(X, S/Y_err,np.sqrt(S_var)/Y_err,marker='o',c='r',ls='',capsize=2)

# plt.figure()
plt.plot(X_grid,f,'-k')
for i in [1,3]:
    plt.fill_between(X_grid,f-i*f_sigma,f+i*f_sigma,alpha=0.25,color='k')
    
# X_grid = jnp.linspace(X.min(),X.max(),200)
# _, cond_sct_gp_data = sct_gp.condition(var_y,X_grid)
# factor = jnp.exp(cond_sct_gp_data.loc)
# plt.plot(X_grid,factor,'-k')
# plt.yscale('log')
#%%
LSF_solution_wscatter = gp.train_LSF_tinygp(X,Y,Y_err,scatter=scatter)
#%%
import harps.lsf.plot as hlsfplot
fig,ax=plt.subplots(1,1)
hlsfplot.plot_tinygp_model(X,Y,Y_err,LSF_solution_wscatter,ax,scatter)
#%%
fig,ax=plt.subplots(1,1)
hlsfplot.plot_tinygp_model(X,Y,Y_err,LSF_solution,ax,None)

#%%
ax=plt.subplot()
hlsf.plot_variances(ax,X,Y,Y_err,LSF_solution,scatter,yscale='log')
