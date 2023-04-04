#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
from harps.core import np, plt

import harps.lsf.aux as aux

import jax
import jax.numpy as jnp
import jaxopt
from tinygp import kernels, GaussianProcess, noise

from functools import partial 


from scipy.optimize import curve_fit



@jax.jit
def loss_LSF(theta,X,Y,Y_err,scatter=None):
    gp = build_LSF_GP(theta,X,Y,Y_err,scatter)
    return -gp.log_probability(Y)
@jax.jit
def loss_scatter(theta,X,Y,Y_err):
    gp = build_scatter_GP(theta,X,Y_err)
    return -gp.log_probability(Y)


def train_LSF_tinygp(X,Y,Y_err,scatter=None):
    '''
    Returns parameters which minimise the loss function defined below.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    Y_err : TYPE
        DESCRIPTION.
    scatter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    solution : TYPE
        DESCRIPTION.

    '''
    p0 = (np.max(Y),0,np.std(X),0)
    # plt.errorbar(X,Y,Y_err,marker='o',ls='')
    popt,pcov = curve_fit(hf.gauss4p,X._value,Y._value,sigma=Y_err._value,
                          absolute_sigma=False,p0=p0)
    perr = np.sqrt(np.diag(pcov))
    
    theta = dict(
        mf_amp        = popt[0],
        mf_loc        = popt[1],
        mf_log_sig    = jnp.log(popt[2]),
        mf_const      = popt[3],
        gp_log_amp    = 1.,#popt[0]/5.,
        gp_log_scale  = 0.,
        log_var_add   = -5.,
    )
    kappa = 5
    lower_bounds = dict(
        mf_amp       = popt[0]-kappa*perr[0],
        mf_loc       = popt[1]-kappa*perr[1],
        mf_log_sig   = np.log(popt[2]-kappa*perr[2]),
        mf_const     = popt[3]-kappa*perr[3],
        gp_log_amp   = -2., #popt[0]/3.-kappa*perr[0],
        gp_log_scale = -1.,
        log_var_add  = -15.,
    )
    upper_bounds = dict(
        mf_amp       = popt[0]+kappa*perr[0],
        mf_loc       = popt[1]+kappa*perr[1],
        mf_log_sig   = np.log(popt[2]+kappa*perr[2]),
        mf_const     = popt[3]+kappa*perr[3],
        gp_log_amp   = 2., # popt[0]/3.+kappa*perr[0],
        gp_log_scale = 1.,
        log_var_add  = 1.5,
    )
    # print(popt); print(perr); print(theta)#; sys.exit()
    bounds = (lower_bounds, upper_bounds)
    # print(bounds)
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=partial(loss_LSF,
                                                      X=X,
                                                      Y=Y,
                                                      Y_err=Y_err,
                                                      scatter=scatter),
                                          method="l-bfgs-b")
    solution = lbfgsb.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
    
    # solver = jaxopt.GradientDescent(fun=partial(loss_LSF,
    #                                           X=X,
    #                                           Y=Y,
    #                                           Y_err=Y_err,
    #                                           scatter=scatter
    #                                           ))
    # solution = solver.run(jax.tree_map(jnp.asarray, theta))
    # try:
    #     print(f"Best fit parameters: {solution.params}")
    # except: pass
    try:
        print(f"Final negative log likelihood: {solution.state.fun_val}")
    except: pass
    return solution.params


        
        
        

def estimate_variance_bin(X,Y,Y_err,theta,minpts,plot=False):
    """
    Estimates the variance based on the residuals to the provided GP parameters
    
    The returned variance is in units of data variance! 
    One should multiply this variance with the variance on the data to get
    accurate results. 

    Parameters
    ----------
    X : jax array
        Contains the x-coordinates
    Y : jax array
        Contains the y-coordinates
    Y_err : jax array
        Contains the error on the y-coordinates.
    theta : dictionary
        Contains the LSF hyper-parameters.
    scale : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.
    minpts : TYPE
        DESCRIPTION.

    Returns
    -------
    logvar_x : TYPE
        DESCRIPTION.
    logvar_y : TYPE
        DESCRIPTION.

    """
    gp = build_LSF_GP(theta,X,Y,Y_err,scatter=None)
    _, cond = gp.condition(Y,X)
    mean_lsf = cond.mean
    rsd = jnp.array((Y - mean_lsf)/Y_err)
    # Bin the residuals along the X-axis into (N+1) bins and calculate the
    # standard deviation in each. 
    # N = 40
    # xlims = np.linspace(X.min(),X.max(),N+1)
    # xdist = np.diff(xlims)[0] # size of the bin in km/s
    # xlims = np.linspace(-scale,scale,nbins)
    # step = np.diff(xlims)[0]
    # bin_means takes right edges of pixels
    # means, stds, counts, var_var = aux.bin_means(X._value,rsd._value,
    #                                 xlims,
    #                                 minpts,
    #                                 value='mean',
    #                                 kind='spline',
    #                                 # y_err=Y_err,
    #                                 remove_outliers=True,
    #                                 return_variance_variance=True)
    
    counts, bin_edges = aux.bin_optimally(X,minpts)
    # Define bin centres
    bin_cens = jnp.array((bin_edges[1:]+bin_edges[:-1])/2.)
    
    # Calculate the relevant statistics
    calculate=['mean','std','sam_variance','sam_variance_variance',
               'pop_variance','pop_variance_variance']
    arrays = aux.get_bin_stat(X, rsd, bin_edges,calculate=calculate,
                              remove_outliers=True)
    means = arrays['mean']
    stds  = arrays['std']
    sam_var_ = arrays['sam_variance']
    sam_var_var = arrays['sam_variance_variance']
    pop_var_ = arrays['pop_variance']
    pop_var_var = arrays['pop_variance_variance']
    
    # Remove empty bins
    cut = np.where(pop_var_!=0)[0]
    
    x_array     = bin_cens[cut]
    pop_var     = pop_var_[cut]
    pop_err     = jnp.sqrt(pop_var)  # error = sqrt of population variance
    sam_var     = sam_var_[cut]
    sam_err     = jnp.sqrt(sam_var)
    pop_var_err = jnp.sqrt(pop_var_var[cut])
    pop_err_err = 1./2. / pop_err * pop_var_err
    sam_var_err = jnp.sqrt(sam_var_var[cut])
    sam_err_err = 1./2. / sam_err * sam_var_err
    log_pop_err = jnp.log(pop_err) # log of error (sqrt of population variance)
    
    log_pop_var, log_pop_var_err = lin2log(pop_var,pop_var_err)
    log_sam_var, log_sam_var_err = lin2log(sam_var,sam_var_err)
    # log_pop_err, log_pop_err_err = lin2log(pop_err,pop_err_err)
    # pop_var_err = jnp.sqrt(pop_var_var[cut])
    # log_pop_var_err = jnp.log(pop_var_err)
    
    # var     = y_array**2
    # y_error = jnp.sqrt(pop_var_var[cut]) # error on population variance
    # err_var   = jnp.sqrt(jnp.abs(pop_var_var))[cut] # error on sample variance
    # err_log_var = jnp.abs(1./pop_var) * err_var
    #logvar_err = jnp.log(y_var) # log of variance on sample variance 
    # log_variance is the 
    # calculate variance on sample variance
    
    
    # print(cut, stds[cut], counts)
    if plot:
        plt.figure()
        # plt.errorbar(X,rsd,Y_err,marker='s',ls='',c='C0')
        # plt.scatter(x_array,means[cut],marker='s',s=5)
        plt.scatter(X,rsd,marker='o',s=3,label='rsd')
        plt.errorbar(bin_cens[cut],
                     np.zeros_like(bin_cens[cut]),
                     # means[cut],
                     sam_var[cut],
                     marker='s',ls='',c='red',
                     label = 'means')
        # plt.scatter(xlims-step/2,stds,marker='o',c='red',zorder=10)
        for i in [-1,1]:
            for j in [1]:
            # plt.plot(x_array,i*stds[cut],color='red',lw=2)
                plt.plot(x_array, j*sam_var,color='r',lw=2)
            
                plt.fill_between(x_array, 
                                 j*sam_var + i*sam_var_err,
                                 j*sam_var - i*sam_var_err, 
                                 color='red',alpha=0.3,zorder=10)
        # plt.errorbar(x_array,pop_var,pop_var_err,label='variance',marker='.',
        #              ls='',c='k')
        plt.xlim(-8,7)
        # plt.ylim(-5.,3.)
        # plt.yscale('log')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.errorbar(x_array, log_pop_var, log_pop_var_err)
        plt.errorbar(x_array, log_sam_var, log_sam_var_err)
        plt.xlim(-8,7)
        # plt.ylim(-5.,3.)
        plt.show()
    # x_array - linear
    # log_var = logarithm of the measured extra variance
    # err_log_var = error on the logarithm of the variance 
    # return x_array, log_pop_err, log_pop_err_err
    return x_array, log_sam_var, log_sam_var_err
    # return x_array, log_var, err_log_var
    
    
def get_model(x_test,X,Y,Y_err,theta,scatter=None):
    # print('get_model',*[np.shape(_) for _ in [x_test,X,Y,Y_err]])
    # print('get_model',theta)
    gp = build_LSF_GP(theta,X,Y,Y_err,scatter=None)
    _, cond = gp.condition(Y,x_test)
    model = cond.mean
    var   = jnp.sqrt(cond.variance)
    return model, var
    
    
def get_residuals(X,Y,Y_err,theta,scatter=None):
    '''
    Returns the residuals to the LSF model

    Parameters
    ----------
    X : array-like
        x-coordinates values.
    Y : array-like
        y-coordinates values.
    Y_err : array-like
        Standard deviation (error) on the y-coordinate values.
    theta : dictionary
        Parameters of the LSF model.
    scatter : tuple, optional
        Output of train_scatter_gp. The default is None.

    Returns
    -------
    rsd : TYPE
        Normalised residuals of the data to the model. 
        No rescaling is done internally on the errors. One may modify the
        Y_err before passing it to this function.

    '''
    model, variance = get_model(X, X, Y, Y_err, theta, None)
    rsd = jnp.array((Y - model)/Y_err)
    return rsd
    
def estimate_variance(X,Y,Y_err,theta,minpts,plot=False,ax=None):
    """
    Estimates the variance based on the residuals to the provided GP parameters
    
    The returned variance is in units of data variance! 
    One should multiply this variance with the variance on the data to get
    accurate results. 

    Parameters
    ----------
    X : jax array
        Contains the x-coordinates
    Y : jax array
        Contains the y-coordinates
    Y_err : jax array
        Contains the error on the y-coordinates.
    theta : dictionary
        Contains the LSF hyper-parameters.
    scale : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.
    minpts : TYPE
        DESCRIPTION.

    Returns
    -------
    logvar_x : TYPE
        DESCRIPTION.
    logvar_y : TYPE
        DESCRIPTION.

    """
    
    
    # Optimally bin the counts    
    counts, bin_edges = aux.bin_optimally(X,minpts)
    # Define bin centres
    bin_cens = jnp.array((bin_edges[1:]+bin_edges[:-1])/2.)
    
    rsd = get_residuals(X,Y,Y_err,theta)
    # Calculate the relevant statistics
    calculate=['mean','std','sam_variance','sam_variance_variance',
               'pop_variance','pop_variance_variance']
    arrays = aux.get_bin_stat(X, rsd, bin_edges,calculate=calculate,
                              remove_outliers=True)
    # means = arrays['mean']
    # stds  = arrays['std']
    sam_var_ = arrays['sam_variance']
    sam_var_var = arrays['sam_variance_variance']
    # pop_var_ = arrays['pop_variance']
    # pop_var_var = arrays['pop_variance_variance']
    
    # Remove empty bins
    cut = np.where(sam_var_!=0)[0]
    
    x_array     = bin_cens[cut]
    # pop_var     = pop_var_[cut]
    # pop_err     = jnp.sqrt(pop_var)  # error = sqrt of population variance
    sam_var     = sam_var_[cut]
    sam_err     = jnp.sqrt(sam_var)
    # pop_var_err = jnp.sqrt(pop_var_var[cut])
    sam_var_err = jnp.sqrt(sam_var_var[cut])
    
    log_sam_var, log_sam_var_err = aux.lin2log(sam_var,sam_var_err)
    
    
    plot_flag = plot | (ax is not None)
    if plot_flag:
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots(1)
        ax.scatter(X,rsd,marker='o',s=3,label='rsd')
        ax.errorbar(bin_cens[cut],
                     np.zeros_like(bin_cens[cut]),
                     # means[cut],
                     sam_err[cut],
                     marker='s',ls='',c='red',
                     label = 'means')
        ax.errorbar(bin_cens[cut],sam_var,sam_var_err,marker='x',ls='',
                    capsize=2,c='C1')
        # for i in [-1,1]:
        #     for j in [1]:
        #         ax.plot(x_array, j*sam_var,color='r',lw=2)
            
        #         ax.fill_between(x_array, 
        #                           j*sam_var + i*sam_var_err,
        #                           j*sam_var - i*sam_var_err, 
        #                           color='red',alpha=0.3,zorder=10)
        # plt.xlim(-8,7)
        ax.set_xlabel("Distance from centre (pix)")
        ax.set_ylabel(r"$S^2 (\sigma^2)$")
        ax.legend()
        
    return x_array, log_sam_var, log_sam_var_err

def train_scatter_tinygp(X,Y,Y_err,theta_lsf,minpts=15,
                         include_error=True):
    '''
    Based on Kersting et al. 2007 :
        Most Likely Heteroscedastic Gaussian Process Regression

    '''

    x_array, log_var, err_log_var_ = estimate_variance(X,Y,Y_err,
                                                          theta_lsf,
                                                          minpts,plot=False)
    
    err_log_var = None
    if include_error:
        err_log_var = err_log_var_
    
    # print(f"Optimizing scatter parameters, err_log_variance = {err_log_variance}")
    theta = dict(
        sct_log_const  = -5.0,
        sct_log_amp    = -0.2,
        sct_log_scale  = 0.0,
        sct_log_epsilon0 = -3.,
        )
    lower_bounds = dict(
        sct_log_const  =-10.0,
        sct_log_amp    =-3.0,
        sct_log_scale  =-1.0,
        sct_log_epsilon0 = -15.,
        )
    upper_bounds = dict(
        sct_log_const  = 0.0,
        sct_log_amp    = 1.0,
        sct_log_scale  = 2.0,
        sct_log_epsilon0 = 3.,
        )
    bounds = (lower_bounds, upper_bounds)
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=partial(loss_scatter,
                                                      X=x_array,
                                                      Y=log_var,
                                                      Y_err=err_log_var),
                                          method="l-bfgs-b")
    solution = lbfgsb.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
    # solver = jaxopt.GradientDescent(fun=partial(loss_scatter,
    #                                           X=x_array,
    #                                           Y=log_var,
    #                                           Y_err=err_log_var,
    #                                           ))
    # solution = solver.run(jax.tree_map(jnp.asarray, theta))
    # print("Scatter solution:",solution.params)
    # print(f"Scatter final negative log likelihood: {solution.state.fun_val}")
    return solution.params, x_array, log_var, err_log_var

def get_scatter_covar(X,Y,Y_err,theta_lsf):
    gp = build_LSF_GP(theta_lsf,X,Y,Y_err,scatter=None)
    _, cond = gp.condition(Y,X,include_mean=False)
    # mean_lsf = cond.loc
    # plt.plot(X,mean_lsf)
    return cond.covariance


    


def rescale_errors(scatter,X,Y_err,plot=False,ax=None):
    '''
    Performs error rescaling, as determined by the scatter parameters

    Parameters
    ----------
    scatter : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    Y_err : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    
    
    theta_scatter, logvar_x, logvar_y, logvar_err = scatter
    sct_gp        = build_scatter_GP(theta_scatter,logvar_x,logvar_err)
    _, sct_cond   = sct_gp.condition(logvar_y,X)
    F_mean  = sct_cond.mean
    F_sigma = jnp.sqrt(sct_cond.variance)
    
    S, S_var = transform(X,Y_err,F_mean,F_sigma,sct_gp,logvar_y)
    plot_flag = plot | (ax is not None)
    if plot_flag:
        import matplotlib.ticker as ticker
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots(1)
        X_grid = jnp.linspace(X.min(),X.max(),200)
        
        
        _, sct_cond_grid = sct_gp.condition(logvar_y,X_grid)
        F_mean_grid  = sct_cond_grid.mean
        F_sigma_grid = jnp.sqrt(sct_cond_grid.variance)
        # print(np.shape(F_mean_grid));sys.exit()
        # f_grid, f_var_grid = transform(X_grid,np.full_like(X_grid,1.),
        #                                 F_mean_grid,F_sigma_grid,
        #                                 sct_gp,logvar_y)
        # logvar_grid_y, logvar_grid_err = aux.lin2log(f_grid, np.sqrt(f_var_grid))
        
        
        linvar_y, linvar_err = aux.log2lin(logvar_y, logvar_err)
        ax.errorbar(logvar_x,logvar_y,
                    logvar_err,ls='',capsize=2,marker='s',
                    label='binned')
        
        
        
        ax.plot(X_grid,F_mean_grid,'-C0',label=r'$g(x;\phi_g)$')
        ax.fill_between(X_grid,
                        F_mean_grid + F_sigma_grid, 
                        F_mean_grid - F_sigma_grid, 
                        color='C0',
                        alpha=0.3)
        # ax.scatter(X,(S/Y_err)**2.,c='r',s=2)
        ax.set_ylabel(r'$\log(\frac{S^2}{\sigma^2})$')
        # ax.set_yscale('log')
        ax.set_xlabel('Distance from centre (pix)')
        ax.set_ylim(-1.5, 3.5)
        ax.yaxis.tick_left()
        # ax.yaxis.set_ticks_position('left')
        axr = ax.secondary_yaxis('right', functions=(lambda x: np.exp(x), 
                                                     lambda x: np.log(x)))
        axr.yaxis.set_major_locator(ticker.FixedLocator([1,5,10,15,20]))
        axr.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axr.set_ylabel(r'$S^2 (\sigma^2)$',labelpad=-3)
        # axr.set_yticks([1, 5, 10,20])
        # axr.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.legend()
    
    
    return S, S_var

def F(x,gp,logvar_y):
    '''
    For a scalar input x and a Gaussian Process GP(mu,sigma**2), 
    returns a scalar output GP(mu (x))
    
    Parameters
    ----------
        x : float32
    Output:
        value : float32
    '''
    
    value = gp.condition(logvar_y,jnp.atleast_1d(x))[1].mean
    return value[0]

def transform(x, sigma, GP_mean, GP_sigma, GP, logvar_y):
    '''
    Rescales the old error value at x-coordinate x using the GP mean 
    and sigma evaluated at x.
    
    F ~ GP(mean, sigma^2)
    F(x=x_i) = log( S_i^2 / sigma_i^2 )
    ==> S_i = sqrt( exp( F(x=x_i) ) ) * sigma_i
            = sqrt( exp( GP_mean) ) * sigma_i
    
    
    
    Propagation of error gives:
    sigma(S_i) = | S_i / 2 * d(F)/dx|_{x_i} * GP_sigma |
    
    where
    GP_mean = F(x=x_i)
    GP_sigma = sigma(F(x=x_i)) 

    Parameters
    ----------
    x : float32, array_like
        x-coordinate.
    sigma : float32, array_like
        error on the y-coordinate value at x.
    GP_mean : float32, array_like
        mean of the GP evaluated at x.
    GP_sigma : float32, array_like
        sigma of the GP evaluated at x.

    Returns
    -------
    S : float32, array_like
        rescaled error on the y-coordinate at x.
    S_var : float32, array_like
        variance on the rescaled error due to uncertainty on the GP mean.

    '''
    deriv = jax.grad(partial(F,gp=GP,logvar_y=logvar_y))
    dFdx  = jax.vmap(deriv)(x)
    S = sigma * jnp.sqrt(jnp.exp(GP_mean))
    S_var = jnp.power(S / 2. * dFdx * GP_sigma,2.)
    return S, S_var
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

def build_LSF_GP(theta_lsf,X,Y,Y_err,scatter=None):
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
    
    if scatter is not None:   
        S, S_var = rescale_errors(scatter, X, Y_err)
        var_data  = jnp.power(S,2.)
        # var_sct_matrix = jnp.diag(var_sct)#+inf_var_covar
        # noise2d = jnp.diag(var_add+var_sct)
        # Noise2d    = noise.Dense(
        #                 noise2d + \
        #                 var_sct_matrix
        #                 )
    else:
        var_data = jnp.power(Y_err,2.)
    var_tot = var_data + var_add
    noise2d = jnp.diag(var_tot)
    Noise2d = noise.Dense(noise2d)
    
    
    return GaussianProcess(
        kernel,
        X,
        noise = Noise2d,
        mean=partial(gaussian_mean_function, theta_lsf),
    )

def build_LSF_GP_bk(theta_lsf,X,Y,Y_err,scatter=None):
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
    var_data = jnp.power(Y_err,2)
    var_add = jnp.exp(theta_lsf['log_var_add']) 
    var_tot = var_data + var_add
    noise2d = jnp.diag(var_tot)
    if scatter is not None:   
        # print("Using scatter parameters")
        # sct_sol, x_array, logvar, logvar_err = scatter
        # sct_gp = build_scatter_GP(sct_sol, x_array, logvar_err)
        # _, sct_cond = sct_gp.condition(logvar,X)
        # var_sct    = jnp.exp(sct_cond.loc) * var_data
        S, S_var = rescale_errors(scatter, X, Y_err)
        var_sct  = jnp.power(S,2.)
        var_sct_matrix = jnp.diag(var_sct)#+inf_var_covar
        Noise2d    = noise.Dense(
                        noise2d + \
                        var_sct_matrix
                        )
    else:
        Noise2d = noise.Diagonal(var_data+var_add)
    return GaussianProcess(
        kernel,
        X,
        noise = Noise2d,
        mean=partial(gaussian_mean_function, theta_lsf),
    )





def estimate_centre(X,Y,Y_err,LSF_solution,scatter=None,N=10):
    
    def value_(x):
        _, cond = gp.condition(Y,jnp.array([x]))
        sample = cond.sample(rng_key,shape=())
        return sample[0]
    # @partial(gp=cond,Y=Y)
    def derivative_(x):#,gp,Y,rng_key):
        # return jax.grad(partial(value_,gp=gp,Y=Y,rng_key=rng_key))(x)
        return jax.grad(value_)(x)
    # @jit
    def solve_(rng_key):
        bisect = jaxopt.Bisection(derivative_,-1.,1.)#,gp=gp,Y=Y,rng_key=rng_key)
        return bisect.run().params
    
    if scatter is not None:
        scatter_solution, logvar_x, logvar_y, logvar_y_err  = scatter
        gp = build_LSF_GP(LSF_solution,X,Y,Y_err,
                          (scatter_solution,
                           logvar_x,
                           logvar_y,
                           logvar_y_err
                           )
                          )
    else:
        gp = build_LSF_GP(LSF_solution,X,Y,Y_err)
    
    X_grid  = jnp.linspace(-1,1,100)
    _, cond = gp.condition(Y,X_grid)
    
    
    centres = np.empty(N)
    for i in range(N):
        rng_key = jax.random.PRNGKey(i)
        
        centres[i] = solve_(rng_key)
    mean, sigma = hf.average(centres)
    return mean, sigma
def estimate_centre_anderson(X,Y,Y_err,LSF_solution,scatter=None):
    
    def value_(x):
        _, cond = gp.condition(Y,jnp.array([x]))
        return cond.mean[0]
    # @partial(gp=cond,Y=Y)
    def derivative_(x):#,gp,Y,rng_key):
        # return jax.grad(partial(value_,gp=gp,Y=Y,rng_key=rng_key))(x)
        return jax.grad(value_)(x)
    # @jit
    
    if scatter is not None:
        scatter_solution, logvar_x, logvar_y, logvar_y_err  = scatter
        gp = build_LSF_GP(LSF_solution,X,Y,Y_err,
                          (scatter_solution,
                           logvar_x,
                           logvar_y,
                           logvar_y_err
                           )
                          )
    else:
        gp = build_LSF_GP(LSF_solution,X,Y,Y_err)
    
    vn = value_(-0.5)
    vp = value_(+0.5)
    dn = derivative_(-0.5)
    dp = derivative_(+0.5)
    
    shift = (vp - vn)/(dp + dn)
    
    return shift, 0.