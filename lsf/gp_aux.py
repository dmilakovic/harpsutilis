#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:25:51 2023

@author: dmilakov
"""

parnames_lfc = ['mf_amp','mf_loc','mf_log_sig','mf_const',
                'gp_log_amp','gp_log_scale','log_var_add']
parnames_sct = ['sct_log_amp','sct_log_scale','sct_log_const']
parnames_all = parnames_lfc + parnames_sct

import numpy as np
import harps.lsf.read as hread
import harps.lsf.gp as hlsfgp
import jax
import jaxopt
import jax.numpy as jnp
from functools import partial 
from scipy.optimize import leastsq




def evaluate_GP(GP,y_data,x_test):
    _, cond = GP.condition(y_data,x_test)
    
    mean = cond.mean
    var  = jnp.sqrt(cond.variance)
    
    return mean, var

def build_scatter_GP_from_lsf1s(lsf1s):
    scatter    = hread.scatter_from_lsf1s(lsf1s)
    scatter_gp = hlsfgp.build_scatter_GP(scatter[0],
                                         X=scatter[1],
                                         Y_err=scatter[3])
    return scatter_gp

def evaluate_scatter_GP_from_lsf1s(lsf1s,x_test):
    theta_sct, sct_x, sct_y, sct_yerr  = hread.scatter_from_lsf1s(lsf1s)
    sct_gp = hlsfgp.build_scatter_GP(theta_sct,sct_x,sct_yerr)
   
    return evaluate_GP(sct_gp, sct_y, x_test)


    
def build_LSF_GP_from_lsf1s(lsf1s,return_scatter=False):
    theta_LSF, data_x, data_y, data_yerr = hread.LSF_from_lsf1s(lsf1s)
    scatter  = hread.scatter_from_lsf1s(lsf1s)
    LSF_gp = hlsfgp.build_LSF_GP(theta_LSF, data_x, data_y, data_yerr,
                                 scatter=scatter)
    if return_scatter:
        return LSF_gp, scatter
    else:
        return LSF_gp

def evaluate_LSF_GP_from_lsf1s(lsf1s,x_test):
    theta_LSF, data_x, data_y, data_yerr = hread.LSF_from_lsf1s(lsf1s)
    scatter = hread.scatter_from_lsf1s(lsf1s)
    LSF_gp = hlsfgp.build_LSF_GP(theta_LSF, data_x, data_y, data_yerr,
                                 scatter=scatter)
    
    return evaluate_GP(LSF_gp, data_y, x_test)




def evaluate_lsf1s(lsf1s,x_test):
    return evaluate_LSF_GP_from_lsf1s(lsf1s,x_test)

def get_segment_centres(lsf1d):
    segcens = (lsf1d['pixl']+lsf1d['pixr'])/2
    return segcens


def get_segment_weights(center,lsf1d,N=2):
    segcens   = get_segment_centres(lsf1d)
    # print(segcens)
    segdist   = jnp.diff(segcens)[0] # assumes equally spaced segment centres
    distances = jnp.abs(center-segcens)
    if N>1:
        used  = distances<segdist*(N-1)
    else:
        used  = distances<segdist/2.
    inv_dist  = 1./distances
    
    segments = jnp.where(used)[0]
    weights  = inv_dist[used]/np.sum(inv_dist[used])
    
    return segments, weights

def helper_calculate_average(list_array,weights,N):
    weights_= jnp.vstack([jnp.full(N,w,dtype='float32') for w in weights])
    average = jnp.average(list_array,axis=0,weights=weights_) 
    return average

def helper_extract_params(theta):
    try:
        amp = theta['amp']
        cen = theta['cen']
        wid = jnp.abs(theta['wid'])
    except:
        amp, cen, wid = theta
        wid = jnp.abs(wid)
    return amp,cen,wid
def helper_rescale_errors(theta,x_test,y_err,sct_data,weights):
    
    amp,cen,wid = helper_extract_params(theta)
    x   = jnp.array((x_test-cen) * wid)
    
    S_list = []
    for scatter in sct_data:
        S, S_var = hlsfgp.rescale_errors(scatter,x,y_err,plot=False)
        S_list.append(S)
    average = helper_calculate_average(jnp.array(S_list),
                                       weights,len(x_test))   
     
    return average

def helper_rescale_xarray(theta,x_test):
    amp,cen,wid = helper_extract_params(theta)
    
    x   = jnp.array((x_test-cen) * wid)
    return x

# def helper_sum_errors(term1,term2):
#     squared = jnp.vstack([jnp.power(terms,2,axis=-1)])
#     X = jnp.sqrt(jnp.sum(squared,axis=0))
#     return X
    
def rsd(theta,x_test,y_data,y_err,LSF_data,sct_data,weights):
    model,mod_err = get_model(theta,x_test,LSF_data,weights)
    rescaled_yerr = helper_rescale_errors(theta,x_test, y_err, sct_data, weights)
    error  = jnp.sqrt(jnp.sum([jnp.power(mod_err,2.),
                              jnp.power(rescaled_yerr,2.)],
                              axis=0))
    # error         = helper_sum_errors(mod_err,rescaled_yerr)
    rsd     = (y_data - model)/error
    return rsd
# @jax.jit

    

def loss(theta,x_test,y_data,y_err,lsf1d,N):
    lsf1d = lsf1d[0]
    bary = jnp.average(x_test,weights=y_data)
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    
    chisq   = jnp.sum(rsd(theta,x_test,y_data,y_err,LSF_data,sct_data,weights)**2)
    
    return chisq
# @jax.jit 
# def return_model(theta,x_test,LSF_data,weights,*args):
#     amp,cen,wid = helper_extract_params(theta)
#     x = helper_rescale_xarray(theta, x_test)
    
#     N = len(LSF_data)
#     model_list = []
#     error_list = []
#     for i in range(N):
#         LSF_theta, LSF_x, LSF_y, LSF_yerr = LSF_data[i]
#         mean, error = hlsfgp.get_model(x,LSF_x,LSF_y,LSF_yerr,LSF_theta,
#                                      scatter=None)
#         model_list.append(mean)
#         error_list.append(error)
        
#     model_ = helper_calculate_average(jnp.array(model_list), 
#                                       weights,len(x_test))
#     # error_ = helper_sum_errors(error_list)
#     error_ = jnp.sqrt(jnp.sum(jnp.power(jnp.array(error_list),2.),axis=0))
    
#     normalisation = amp / jnp.max(model_)
#     model = model_ * normalisation
#     error = error_ * normalisation
#     return model, error



def get_params_scipy(lsf1d,x_test,y_data,y_err,interpolate=False):
    
    
    # def residuals(theta):
    #     model_y, model_err = return_model(theta, x_test, theta_LSF, LSF_x,LSF_y,LSF_yerr)
    #     if scatter is not None:
    #         S, S_var = hlsfgp.rescale_errors(scatter,x_test,y_err,plot=False)
    #         error = S
    #     else:
    #         error = y_err
        
    #     rsd     = (y_data - model_y)/error
    #     return rsd
    bary = np.average(x_test,weights=y_data)
    N = 2 if interpolate == True else 1
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    theta = (np.max(y_data),bary,1.)
    # theta_LSF, LSF_x, LSF_y, LSF_yerr = hread.LSF_from_lsf1s(lsf1s)
    # scatter = hread.scatter_from_lsf1s(lsf1s)
    optpars,pcov,infodict,errmsg,ier = leastsq(rsd,x0=theta,
                                               args=(x_test,y_data,y_err,
                                                     LSF_data,sct_data,weights),
                                               full_output=True)
    
    if ier not in [1, 2, 3, 4]:
        print("Optimal parameters not found: " + errmsg)
        optpars = np.full_like(theta,np.nan)
        pcov = None
        success = False
    else:
        success = True
    dof = len(x_test) - len(optpars)
    chisq= loss(optpars,x_test,y_data,y_err,LSF_data,sct_data,jnp.array(weights))
    return optpars, pcov, chisq, dof

# def get_parameters(lsf1d,x_test,y_data,y_err,interpolate=False):
#     bary = np.average(x_test,weights=y_data)
#     N = 2 if interpolate == True else 1
    
def get_model(theta,x_test,LSF_data,weights):
    amp,cen,wid = helper_extract_params(theta)
    x = helper_rescale_xarray(theta, x_test)
    
    model_list = []
    error_list = []
    M = len(x)
    N = len(LSF_data)
    for i in range(N):
        LSF_theta, LSF_x, LSF_y, LSF_yerr = LSF_data[i]
        mean, error = hlsfgp.get_model(x,LSF_x,LSF_y,LSF_yerr,LSF_theta,
                                     scatter=None)
        model_list.append(mean)
        error_list.append(error)
        
    weights_= jnp.vstack([jnp.full(M,w,dtype='float32') \
                          for w in weights])
    model_  = jnp.average(jnp.array(model_list),
                          axis=0,
                          weights=weights_
                          ) 
    
    error_ = jnp.sqrt(jnp.sum(jnp.power(jnp.array(error_list),2.),
                              axis=0)
                      )
    
    normalisation = amp / jnp.max(model_)
    model = model_ * normalisation
    mod_err = error_ * normalisation
    return model, mod_err    

def get_parameters(lsf1d,x_test,y_data,y_err,interpolate=False):
    bary = jnp.average(x_test,weights=y_data)
    
    N = 2 if interpolate == True else 1
    bary = jnp.average(x_test,weights=y_data)
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    
    
    theta = dict(
        amp = jnp.max(y_data),
        cen = bary,
        wid = 1.0
        )
    lower_bounds = dict(
        amp = jnp.max(y_data)*0.8,
        cen = bary-2.0,
        wid = 0.9
        )
    upper_bounds = dict(
        amp = jnp.max(y_data)*1.2,
        cen = bary+2.0,
        wid = 1.1,
        )
    bounds = (lower_bounds, upper_bounds)
    
    kwargs = dict(
        x_test=jnp.array(x_test,dtype='float32'),
        y_data=jnp.array(y_data),
        y_err=jnp.array(y_err),
        lsf1d=tuple(lsf1d,),
        N=N
        # LSF_data=LSF_data,
        # sct_data=sct_data,
        # weights = jnp.array(weights)
        )
    # return theta, kwargs 
    def _get_model(theta,x,LSF_data,weights):
        model_list = []
        error_list = []
        M = len(x)
        N = len(LSF_data)
        for i in range(N):
            LSF_theta, LSF_x, LSF_y, LSF_yerr = LSF_data[i]
            mean, error = hlsfgp.get_model(x,LSF_x,LSF_y,LSF_yerr,LSF_theta,
                                         scatter=None)
            model_list.append(mean)
            error_list.append(error)
            
        weights_= jnp.vstack([jnp.full(M,w,dtype='float64') \
                              for w in weights])
        model_  = jnp.average(jnp.array(model_list),
                              axis=0,
                              weights=weights_
                              ) 
        
        error_ = jnp.sqrt(jnp.sum(jnp.power(jnp.array(error_list),2.),
                                  axis=0)
                          )
        return model_, error_
    
    @jax.jit
    def loss_(theta):
        amp,cen,wid = helper_extract_params(theta)
        x = helper_rescale_xarray(theta, x_test)
        # M = len(x)
        # N = len(LSF_data) # number of LSF models for interpolation
        
        # # Evaluate the N GPs and weight them
        # model_list = []
        # error_list = []
        # for i in range(N):
        #     LSF_theta, LSF_x, LSF_y, LSF_yerr = LSF_data[i]
        #     mean, error = hlsfgp.get_model(x,LSF_x,LSF_y,LSF_yerr,LSF_theta,
        #                                  scatter=None)
        #     model_list.append(mean)
        #     error_list.append(error)
            
        # weights_= jnp.vstack([jnp.full(M,w,dtype='float32') \
        #                       for w in weights])
        # model_  = jnp.average(jnp.array(model_list),
        #                       axis=0,
        #                       weights=weights_
        #                       ) 
        
        # error_ = jnp.sqrt(jnp.sum(jnp.power(jnp.array(error_list),2.),
        #                           axis=0)
        #                   )
        model_, error_ = _get_model(theta,x,LSF_data,weights)
        
        normalisation = amp / jnp.sum(model_)
        model = model_ * normalisation
        mod_err = error_ * normalisation
        
        rescaled_yerr = helper_rescale_errors(theta,x_test, y_err, 
                                              sct_data, weights)
        
        error = jnp.sqrt(jnp.sum(jnp.power(jnp.array(
                                                [mod_err,rescaled_yerr]),
                                            2.),
                                  axis=0)
                          )
        
        rsd     = (y_data - model)/error
        
        return jnp.sum(rsd**2)
        
    
    
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=loss_,
                                          method="l-bfgs-b")
    solution = lbfgsb.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
    
    
    # solver = jaxopt.GradientDescent(fun=partial(loss,
    #                                             x_test=jnp.array(x_test),
    #                                             y_data=jnp.array(y_data),
    #                                             y_err=jnp.array(y_err),
    #                                             theta_LSF=theta_LSF,
    #                                             X = jnp.array(LSF_x),
    #                                             Y = jnp.array(LSF_y),
    #                                             Y_err = jnp.array(LSF_yerr),
    #                                           ))
    # solution = solver.run(jax.tree_map(jnp.asarray, theta))
    
    optpars = solution.params
    dof = len(x_test) - len(optpars)
    chisq= loss_(optpars)
    return optpars, chisq, dof

def get_parameters_opt(lsf1d,x_test,y_data,y_err,interpolate=False):
    import scipy.optimize as optimize
    bary = np.average(x_test,weights=y_data)
    N = 2 if interpolate == True else 1
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    
    
    theta = dict(
        amp = np.max(y_data),
        cen = bary,
        wid = 1.0
        )
    lower_bounds = dict(
        amp = np.max(y_data)*0.8,
        cen = bary-2.0,
        wid = 0.95
        )
    upper_bounds = dict(
        amp = np.max(y_data)*1.2,
        cen = bary+2.0,
        wid = 1.05,
        )
    bounds = (lower_bounds, upper_bounds)
    
    kwargs = dict(
        x_test=np.array(x_test,dtype='float32'),
        y_data=np.array(y_data),
        y_err=np.array(y_err),
        LSF_data=LSF_data,
        sct_data=sct_data,
        weights = np.array(weights)
        )
    optimize.minimize(loss, x0=np.fromiter(theta.values(),dtype='float32'), 
                      args=tuple(kwargs.values()),
                      method='BFGS')
    
    return theta, kwargs 





def extract_LSF_lists(center,lsf1d,N=2):
    return extract_lists('LSF',center,lsf1d,N)

def extract_scatter_lists(center,lsf1d,N=2):
    return extract_lists('scatter',center,lsf1d,N)

def extract_lists(what,center,lsf1d,N=2):
    assert what in ['LSF','scatter']
    segments, weights = get_segment_weights(center,lsf1d,N)
    ll = []
    for segm in segments:
        data = hread.from_lsf1s(lsf1d[segm],what)
        ll.append(data)
    return tuple(ll),weights

def get_integral(optpars,x1l,flx1l,lsf1d,interpolate,M=100):
    N = 2 if interpolate == True else 1
    bary   = jnp.average(x1l,weights=flx1l)
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    x_test = jnp.linspace(x1l.min(), x1l.max(),M)
    model, model_err = get_model(optpars,x_test,LSF_data,weights)
    
    return jnp.trapz(model,x_test)

def fit_lsf2line(x1l,flx1l,bkg1l,err1l,lsf1d,interpolate=True,
        output_model=False,plot=False,*args,**kwargs):
    
    bary   = np.average(x1l,weights=flx1l)
    x_test = jnp.array(x1l,dtype=jnp.float32)
    y_data = jnp.array(flx1l-bkg1l,dtype=jnp.float32)
    y_err  = jnp.array(err1l,dtype=jnp.float32)
    
    try:
        optpars, chisq, dof = get_parameters(lsf1d,x_test,y_data,y_err,
                                             interpolate=interpolate)
        # optpars = get_params_scipy(lsf1s,x_test,y_data,y_err)
        pcov = None
        success = True
    except:
        optpars = dict(amp=np.nan,cen=np.nan,wid=np.nan)
        success = False
    
    if success:   
        amp = optpars['amp']
        cen = optpars['cen']
        wid = optpars['wid']
        
        # chisq   = loss(optpars, x_test,y_data,y_err,
        #                theta_LSF,LSF_x, LSF_y, LSF_yerr)
        # dof  = len(x1l) - (len(optpars)+len(theta_LSF))
        if pcov is not None:
            pcov = pcov*chisq/dof
        else:
            pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
        #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
    else:
        popt = np.full(3,np.nan)
        amp, cen, wid = popt
        pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
        chisq = np.nan
        success=False
        dof  = len(x1l)
    pars    = np.array([amp, cen, wid])
    errors  = np.sqrt(np.diag(pcov))
    chisqnu = chisq/dof
    
    N = 2 if interpolate == True else 1
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    model, model_err = get_model(optpars,x_test,LSF_data,weights)
    
    integral = np.sum(model)
    output_tuple = (success, pars, errors, chisq, chisqnu, integral)
    if plot:
        plot_result(optpars,lsf1d,x1l,flx1l,bkg1l,err1l)
    if output_model:  
        output_tuple = output_tuple + (model,)
    return output_tuple
    
def plot_result(optpars,lsf1d,pix,flux,background,error,interpolate=True):
    import matplotlib.pyplot as plt
    pix = jnp.array(pix)
    
    # theta_LSF, LSF_x, LSF_y, LSF_yerr = hread.LSF_from_lsf1s(lsf1s)
    # print('plot',*[np.shape(_) for _ in [optpars,LSF_x,LSF_y,LSF_yerr]])
    # model   = return_model(optpars,pix,lsf1s)
    bary = np.average(pix,weights=flux)
    N = 2 if interpolate == True else 1
    LSF_data, weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data, weights = extract_lists('scatter',bary,lsf1d,N=N)
    
    
    model,model_err = get_model(optpars,pix,LSF_data,weights)
    rescaled_yerr = helper_rescale_errors(optpars, pix, error,sct_data,weights)
    # full_error    = jnp.sqrt(jnp.sum(jnp.power(jnp.array([model_err,
    #                                                       rescaled_yerr]),2.),
    #                                  axis=0))
    rsd = ((flux-background)-model)/rescaled_yerr
    # print(model)
    # plotter = Figure2(2,1,height_ratios=[3,1])
    # ax0     = plotter.add_subplot(0,1,0,1)
    # ax1     = plotter.add_subplot(1,2,0,1,sharex=ax0)
    fig, (ax1,ax2) = plt.subplots(2,1)
    
    ax1.set_title('fit.lsf')
    ax1.plot(pix,flux-background,label='Flux',drawstyle='steps-mid')
    ax1.plot(pix,model,label='Model',drawstyle='steps-mid')
    
    x_grid = np.linspace(pix.min(),pix.max(),400)
    model_grid,model_grid_err = get_model(optpars,x_grid,LSF_data,weights)
    ax1.plot(x_grid,model_grid,lw=2)
    
    ax2.scatter(pix,rsd,marker='s')
    [ax2.axhline(i,ls='--',lw=1) for i in [-1,0,1]]
    ax2.set_ylim(-5,5)
    ax1.legend()
    
