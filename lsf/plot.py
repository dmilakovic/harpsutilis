#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
from harps import functions as hf
from harps import settings as hs
from harps import io as io
from harps import containers as container
from harps import plotter as hplot
from harps import fit as hfit
#from .gaussprocess_class import HeteroskedasticGaussian
from harps.core import os, np, plt, FITS


import jax
from   jax import jit
import jax.numpy as jnp
import jaxopt
from tinygp import kernels, GaussianProcess, noise

import pyro
import numpyro 
import torch
import numpyro.distributions as dist
from functools import partial 

#import line_profiler

#import errno

from scipy import interpolate
from scipy.optimize import leastsq, brentq, curve_fit, root, newton
import scipy.stats as stats

from matplotlib import ticker
import hashlib

import harps.lsf.gp as lsfgp

def plot_tinygp_model(x,y,y_err,solution,ax,scatter=None):
    X = jnp.array(x)
    Y        = jnp.array(y)
    Y_err    = jnp.array(y_err)
    # Y = jnp.array(y*100)
    # Y_err = jnp.array(y_err*100)
    X_grid = jnp.linspace(X.min(),X.max(),400)
    
    gp = lsfgp.build_LSF_GP(solution,X,Y,Y_err,scatter)
    _, cond = gp.condition(Y, X_grid)

    mu = cond.loc
    std = np.sqrt(cond.variance)
    ax.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
    ax.plot(X_grid, mu, label="Full model")
    for i in [1,3]:
        ax.fill_between(X_grid, mu + i*std, mu - i*std, color="C0", alpha=0.3)


    # Separate mean and GP
    _, cond_nomean = gp.condition(Y, X_grid, include_mean=False)

    mu_nomean = cond_nomean.loc #+ soln.params["mf_amps"][0] # second term is for nicer plot
    std_nomean = np.sqrt(cond_nomean.variance)

    # plt.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
    ax.plot(X_grid, mu_nomean, c='C0', ls='--', label="GP model")
    for i in [1,3]:
        ax.fill_between(X_grid, mu_nomean + i*std_nomean, mu_nomean - i*std_nomean,
                         color="C0", alpha=0.3)
    ax.plot(X_grid, jax.vmap(gp.mean_function)(X_grid), c="C1", label="Gaussian model")

  
    return None

def plot_solution(pix1s,flx1s,err1s,method,dictionary,
                      checksum,save=False,**kwargs):
    plot_subpix_grid = kwargs.pop('plot_subpix',False)
    plot_model       = kwargs.pop('plot_model',True)
    rasterized       = kwargs.pop('rasterized',False)
    plot_sigma       = kwargs.pop('plot_sigma',[1,3])
    plot_gaussian    = kwargs.pop('plot_gaussian',False)
    
    total_shift = -dictionary['totshift']
    centre_error = dictionary['lsfcen_err']
    
    params_LSF = dictionary['solution_LSF']
    N_params   = len(params_LSF)
    full_theta = params_LSF
    
    X = jnp.array(pix1s)
    Y        = jnp.array(flx1s)
    Y_err    = jnp.array(err1s)
    
    try:
        solution_scatter = dictionary['solution_scatter']
        params_sct = solution_scatter[0]#.params
        logvar_x   = solution_scatter[1]
        logvar_y   = solution_scatter[2] 
        logvar_error = solution_scatter[3] 
        scatter    = (params_sct,logvar_x,logvar_y,logvar_error)
        N_params += len(params_sct)
        full_theta.update(params_sct)
        
        gp_scatter = lsfgp.build_scatter_GP(params_sct, logvar_x, logvar_error)
        _, gp_sct_cond = gp_scatter.condition(logvar_y,X)
        # var_scatter = jnp.exp(gp_sct_cond.loc) * Y_err**2
        new_err, new_err_var = lsfgp.rescale_errors(scatter,X,Y_err,plot=False)
        var_data = new_err**2
        
    except:
        scatter = None
        # var_scatter = jnp.zeros_like(Y_err)
    # model_scatter = dictionary['model_scatter']
    # scatter = "True" if model_scatter==True else None
    
    for (p,v) in full_theta.items():
        print(f"{p:<20s} = {v:>8.3f}")
    
    
    
    
    # Condition the model on a dense grid in X
    X_grid = jnp.linspace(X.min(),X.max(),400)
    gp = lsfgp.build_LSF_GP(params_LSF,X,Y,
                      Y_err=Y_err,
                      # Y_err=jnp.zeros_like(Y),
                      scatter=scatter,
                      #scatter=None
                      )
    _, cond = gp.condition(Y, X_grid)
    
    mu = cond.loc
    std = np.sqrt(cond.variance)
    
    plotter = hplot.Figure2(4,2, figsize=(9,8),
                        height_ratios=[5,2,2,2],width_ratios=[5,1])
    
    ax_obs = plotter.add_subplot(0,1,0,1)
    ax_gp  = plotter.add_subplot(1,2,0,1,sharex=ax_obs)
    ax_var = plotter.add_subplot(2,3,0,1,sharex=ax_obs)
    ax_rsd = plotter.add_subplot(3,4,0,1,sharex=ax_obs)
    ax_hst = plotter.add_subplot(3,4,1,2)
    
    for ax in plotter.axes:
        ax.axhline(0,ls=':',c='grey',zorder=5)
    
    # First panel: data, full model and the mean function
    ax_obs.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
    ax_obs.plot(X_grid, mu, label=r"$GP$ model",lw=2,zorder=5)
    for i in np.atleast_1d(plot_sigma):
        ax_obs.fill_between(X_grid, mu + i*std, mu - i*std, color="C0", 
                            alpha=0.15)
    ax_obs.plot(X_grid, jax.vmap(gp.mean_function)(X_grid), c="C1",ls='--',
             label=r"Mean function, $\mu(GP)$",lw=2,zorder=4)   
    
    
    # First panel: random samples from GP posterior 
    # rng_key = jax.random.PRNGKey(55873)
    # sampled_f = cond.sample(rng_key,(20,))
    # for f in sampled_f:
    #     ax_obs.plot(X_grid,f,c='C0',lw=0.5)
    
    # Second panel: the Gaussian process + residuals from Gaussian model
    _, cond_nomean = gp.condition(Y, X_grid, include_mean=False)
    
    mu_nomean = cond_nomean.loc 
    std_nomean = np.sqrt(cond_nomean.variance)
    ax_gp.plot(X_grid, mu_nomean, c='C0', ls='--', label=r"$GP - \mu(GP)$",zorder=5)
    y2lims = [100,-100] # saves y limits for the middle panel
    for i in np.atleast_1d(plot_sigma):
        upper = mu_nomean + i*std_nomean
        lower = mu_nomean - i*std_nomean
        if np.max(lower)<y2lims[0]:
            y2lims[0]=np.min(lower)
        if np.max(upper)>y2lims[1]:
            y2lims[1]=np.max(upper)
        ax_gp.fill_between(X_grid, upper, lower,
                         color="C0", alpha=0.3,zorder=0)
    # Second panel: random samples from GP posterior , no mean function
    rng_key = jax.random.PRNGKey(55873)
    sampled_f = cond_nomean.sample(rng_key,(20,))
    for f in sampled_f:
        ax_gp.plot(X_grid,f,c='C0',lw=0.5)
        
    # Second panel: residuals from gaussian model
    # _, cond_nomean_predict = gp.condition(Y, X, include_mean=False)
    # std_nomean_predict = np.sqrt(cond_nomean_predict.variance)
    cond_nomean_predict, std_nomean_predict = gp.predict(Y,X,
                                                         include_mean=False,
                                                         return_var=True)
    Y_gauss_rsd = Y - jax.vmap(gp.mean_function)(X)
    Y_gauss_err = Y_err
    # Y_gauss_err = jnp.sqrt(Y_err**2 + std_nomean_predict**2)
    ax_gp.errorbar(X, Y_gauss_rsd, Y_gauss_err, marker='.',color='grey',ls='')
    
    # Third panel: variances
    ax_var = plot_variances(ax_var, X,Y,Y_err,params_LSF,scatter=scatter,
                            yscale='log')
    ax_var.legend(bbox_to_anchor=(1.02, 1.00),fontsize=8)
    
    # Fourth left panel: normalised residuals for Gaussian Process
    _, cond_predict = gp.condition(Y, X, include_mean=True)
    mu_model = cond_predict.loc # second term is for nicer plot
    
    # calculate all variances 
    var_data = Y_err **2
    var_model = cond_predict.variance
   
    
    Y_tot = jnp.sqrt(var_model + var_data)
    # Y_tot = Y_err
    # Y_tot = jnp.sqrt(var_predict)
    rsd = (mu_model-Y)/Y_tot
    snr = Y/Y_tot
    ax_rsd.scatter(X,rsd,marker='.',c='grey')
    ax_obs.plot(X,snr)
    
    rsd_to_plot = [rsd]
    # ---------- Single gaussian fit (optional):
    if plot_gaussian:
        # First panel: Fitted Gaussian
        p0 = (np.max(Y),0,np.std(X),0)
        popt,pcov = curve_fit(hf.gauss4p,X._value,Y._value,sigma=Y_err._value,
                              absolute_sigma=False,p0=p0)
        gauss_Y = hf.gauss4p(X_grid,*popt)
        gauss_Y_err = hf.error_from_covar(hf.gauss4p, popt, pcov, X_grid)
        ax_obs.plot(X_grid, gauss_Y, c="C3",ls=':',
                 label="Gaussian model",lw=2,zorder=3)
        for i in [1,3]:
            upper = gauss_Y + i*gauss_Y_err
            lower = gauss_Y - i*gauss_Y_err
            ax_obs.fill_between(X_grid,upper,lower,
                                color='C3',alpha=0.3,zorder=0)
    
        # Fourth left panel: normalised residuals for a single Gaussian function
        gauss_mu_predict  = hf.gauss4p(X, *popt)
        gauss_std_predict = hf.error_from_covar(hf.gauss4p, popt, pcov, X)
        gauss_rsd = (gauss_mu_predict-Y)/gauss_std_predict
        ax_rsd.scatter(X,gauss_rsd,marker='.',c='C3')    
        rsd_to_plot.append(gauss_rsd)
    
    # ---------- Histogram of residuals
    ax3_ylims = ax_rsd.get_ylim()
    colors = ['grey','C3']
    
    for i,rsd_arr in enumerate(rsd_to_plot):
        y_pos = 0.9-0.12*i
        color = colors[i]
        median,upper,lower=plot_histogram(ax_hst, rsd_arr, color, y_pos,
                                          range=ax3_ylims)
    
        for ax in [ax_rsd,ax_hst]:
            [ax.axhline(val,ls=(0,(10,5,10,5)),color=color,lw=0.8) for val in [upper,lower]]
            [ax.axhspan(-1,1,color=color,alpha=0.2)]
    ax_hst.set_ylim(ax3_ylims)
    chisq = np.sum(rsd**2)
    dof   = (len(Y)-N_params)
    aicc  = chisq + 2*len(Y)*N_params / (len(Y)-N_params-1)
    labels = ['Mean func $\mu$',
              'Mean func $\sigma$', 
              'Mean func $A$', 
              'Mean func $y_0$',
              'GP $\sigma$', 
              'GP $l$', 
              'log(rand.var)',
              r'log(<Y_err>)',
              '$N$',
              r'$\nu$',
              r'$\chi^2$',
              r'$\chi^2/\nu$',
              'AICc',
              '-log(probability)',
              'Meas centre',
              '(error)',
              'Gaus centre']
    values = [#np.exp(params_LSF['log_mf_loc']),
              params_LSF['mf_loc'], 
              np.exp(params_LSF['mf_log_sig']), 
              params_LSF['mf_amp'], 
              params_LSF['mf_const'], 
              np.exp(params_LSF['gp_log_amp']),
              np.exp(params_LSF['gp_log_scale']),
              params_LSF['log_var_add'],
              np.log(np.mean(Y_err**2)),
              len(Y),  
              dof, 
              chisq, 
              chisq/dof,
              aicc,
              lsfgp.loss_LSF(params_LSF,X,Y,Y_err,scatter),
              total_shift*1000,
              centre_error*1000]
    units  = [*2*(r'kms$^{-1}$',),
              *3*('arb.',),
              r'kms$^{-1}$',
              *8*('',),
              *3*(r'ms$^{-1}$',)]
    formats = [*8*('9.3f',),
               *2*('5d',),
               *10*('9.3f',),
               *3*('+9.3f')]
    if plot_gaussian:
        values.append(popt[1])
    for i,(l,v,m,u) in enumerate(zip(labels,values,formats,units)):
        text = (f"{l:>20} = {v:>{m}}")
        if len(u)>0:
            text+=f' [{u}]'
        ax_obs.text(1.26,0.9-i*0.08,text,
                 horizontalalignment='right',
                 verticalalignment='center', 
                 transform=ax_obs.transAxes, 
                 fontsize=7)
        print(text)
    
    
    Xlarger = np.max([X.max(), np.abs(X.min())])
    ax_obs.set_xlim(-1.025*Xlarger, 1.025*Xlarger)
    y2lim = np.max([*np.abs(y2lims),*Y_gauss_rsd])
    ax_gp.set_ylim(-1.5*y2lim,1.5*y2lim)
    # ax1.set_ylim(-0.05,0.25)
    plotter.axes[-1].set_xlabel("x "+r'[kms$^{-1}$]')
    ax_obs.set_ylabel("Flux (arbitrary)")
    ax_gp.set_ylabel(r"Data $-$ $\mu(GP)$")
    ax_rsd.set_ylabel("Residuals\n"+r"($\sigma$)")
    ax_hst.set_yticklabels([])
    ax_hst.set_xlabel('#')
    _ = ax_obs.legend()
    _ = ax_gp.legend()
    
    plotter.figure.align_ylabels()
    
    if save:
        figname = '/Users/dmilakov/projects/lfc/plots/lsf/'+\
                  'ESPRESSO_{0}.pdf'.format(checksum)
        plotter.save(figname,rasterized=rasterized)
        _ = plt.close(plotter.figure)   
        return None
    else:
        return plotter
def plot_histogram(ax,rsd_arr,color,text_yposition,range=None):
    # Fourth right panel: a histogram of normalised residuals
    n, bins, _ = ax.hist(np.ravel(rsd_arr),bins=10,range=range,
                          color=color,orientation='horizontal',
                          histtype='step',lw=2)
    # Fourth right panel: print and plot horizontal lines for quantiles 
    median    = np.median(np.ravel(rsd_arr))
    quantiles = np.quantile(np.ravel(rsd_arr),[0.05,0.95])
    lower, upper = quantiles - median
    ax.text(0.85,text_yposition,#
             r'${{{0:+3.1f}}}_{{{1:+3.1f}}}^{{{2:+3.1f}}}$'.format(median,lower,upper),
             horizontalalignment='right',
             verticalalignment='center', 
             transform=ax.transAxes, 
             fontsize=7,
             color=color)
    [ax.axhline(val,ls=(0,(10,5,10,5)),color='grey',lw=0.8) for val in [upper,lower]]
    return median,upper,lower 

def plot_variances(ax, X,Y,Y_err,theta,scatter=None,yscale='log'):
    var_data = jnp.power(Y_err,2)
    var_add = jnp.broadcast_to(jnp.exp(theta['log_var_add']),Y_err.shape)
    var_tot = var_add + var_data
    
    gp = lsfgp.build_LSF_GP(theta,X,Y,Y_err,scatter=scatter)
    _, cond = gp.condition(Y,X,include_mean=True)
    var_mod = cond.variance
    if scatter is not None:
        S, S_var = lsfgp.rescale_errors(scatter, X, Y_err)
        var_new   = jnp.power(S,2.)    
        var_new_err = 2*S*jnp.sqrt(S_var)   
        var_tot   = var_add + var_new
        
            
        
        ax.plot(X,var_new,label='Inferred from data',ls=(0,(5,2,5)),c='C3')
        
        ax.fill_between(X,
                        var_new+var_new_err,
                        var_new-var_new_err,
                        color='C3',alpha=0.3)
        
    ax.scatter(X,var_data,label='Data',marker='.',c='grey',s=2)
    ax.plot(X,var_add,label='Add. variance',ls=(0,(1,2,1,2)),c='C1')
    ax.plot(X,var_tot,label='Total variance',ls='--',c='C4',lw=2.)
    ax.plot(X,var_mod,label='Model variance',ls='-',c='C0',lw=1.)
    ax.legend(fontsize=8)
    yscale_kwargs={}
    if yscale=='log':
        yscale_kwargs.update(dict(nonpositive='clip'))
    ax.set_yscale(yscale,**yscale_kwargs)
    ax.set_ylabel(r'$\sigma^2$')
    return ax

def plot_analytic_lsf(values,ax,title=None,saveto=None,**kwargs):
    nitems = len(values.shape)
    npts   = 500
    x = np.linspace(-6,6,npts)
    plot_components=kwargs.pop('plot_components',False)
    if nitems>0:
        numvals = len(values)
        colors = plt.cm.jet(np.linspace(0,1,numvals))
        for j,item in enumerate(values):
            y = hf.gaussP(x,*item['pars'])
            ax.plot(x,y,lw=2,c=colors[j])
            if plot_components:
                ylist = hf.gaussP(x,*item['pars'],return_components=True)
                [ax.plot(x,y_,lw=0.6,ls='--',c=colors[j]) for y_ in ylist]
    else:            
        y = hf.gaussP(x,*values['pars'])
        ax.plot(x,y,lw=2)
    return ax
    
def plot_gp_lsf(values,ax,title=None,saveto=None,**kwargs):
    nitems = len(values.shape)
    npts   = 500
    x      = np.linspace(-10,10,npts)
    if nitems>0:
        numvals = len(values)
        colors = plt.cm.jet(np.linspace(0,1,numvals))
        for j, item in enumerate(values):
            pass

    return ax


    