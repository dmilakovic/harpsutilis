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
    var  = jnp.sqrt(cond.covariance)
    
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

# def get_LSF_model(theta,LSF_model,x_test):
#     amp = theta['amp']
#     cen = theta['cen']
#     wid = jnp.abs(theta['wid'])
    
    
    
#     x     = cen + (x_test* wid)
#     y     = amp * (LSF_model / np.max(LSF_model))
    
#     return x,y

@jax.jit
def loss(theta,x_test,y_data,y_err,theta_LSF,X,Y,Y_err,scatter=None):
    
    # rsd = hlsfgp.get_residuals(x_test, Y, Y_err, theta_LSF)
    
    # model_x, model_y = get_LSF_model(theta, lsf1s, x_test)
    model_y = return_model(theta, x_test, theta_LSF, X, Y, Y_err)
    if scatter is not None:
        S, S_var = hlsfgp.rescale_errors(scatter,x_test,y_err,plot=False)
        error = S
    else:
        error = y_err
    
    rsd     = (y_data - model_y)/error
    chisq   = np.sum(rsd**2)
    
    return chisq

def return_model(theta,x_test,theta_LSF,X,Y,Y_err):
    
    try:
        amp = theta['amp']
        cen = theta['cen']
        wid = jnp.abs(theta['wid'])
    except:
        amp, cen, wid = theta
        wid = jnp.abs(wid)
    
    x   = jnp.array(cen + (x_test * wid))
    
    mean, var = hlsfgp.get_model(x,X,Y,Y_err,theta_LSF,scatter=None)
    # gp_LSF = hlsfgp.build_LSF_GP(theta_LSF, X, Y, Y_err)
    # mean, var = evaluate_GP(gp_LSF,Y, x)
    
    model_y = amp * (mean / np.max(mean))
    return model_y


def get_params_scipy(pix,flux,background,error,lsf1s,
        output_model=False,plot=False,*args,**kwargs):
    
    x_test = jnp.array(pix,dtype=jnp.float32)
    y_data = jnp.array(flux-background,dtype=jnp.float32)
    y_err  = jnp.array(error,dtype=jnp.float32)
    
    
    def residuals(theta):
        model_y = return_model(theta, x_test, theta_LSF, LSF_x,LSF_y,LSF_yerr)
        if scatter is not None:
            S, S_var = hlsfgp.rescale_errors(scatter,x_test,y_err,plot=False)
            error = S
        else:
            error = y_err
        
        rsd     = (y_data - model_y)/error
        return rsd
    
    theta = (np.max(y_data),0.,1.)
    theta_LSF, LSF_x, LSF_y, LSF_yerr = hread.LSF_from_lsf1s(lsf1s)
    scatter = hread.scatter_from_lsf1s(lsf1s)
    optpars,pcov,infodict,errmsg,ier = leastsq(residuals,x0=theta,
                                        full_output=True)
    
    if ier not in [1, 2, 3, 4]:
        print("Optimal parameters not found: " + errmsg)
        optpars = np.full_like(theta,np.nan)
        pcov = None
        success = False
    else:
        success = True
    if success:   
        amp, cen, wid = optpars
        chisq = np.sum(infodict['fvec']**2)
        dof  = (len(x_test) - (len(optpars)+len(theta_LSF)))
        if pcov is not None:
            pcov = pcov*chisq/dof
        else:
            pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
        #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
    else:
        optpars = np.full_like(theta,np.nan)
        amp, cen, wid = optpars
        pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
        chisq = np.nan
        dof  = (len(x_test) - (len(optpars)+len(theta_LSF)))
        success=False
    pars    = np.array([amp, cen, wid])
    errors  = np.sqrt(np.diag(pcov))
    chisqnu = chisq/dof
    
    if plot:
        plot_result(optpars,lsf1s,pix,flux,background,error)
    if output_model:  
        model   = return_model(optpars,x_test,theta_LSF,LSF_x,LSF_y,LSF_yerr)
        return success, pars, errors, chisq, chisqnu, model
    else:
        return success, pars, errors, chisq, chisqnu

def get_parameters(lsf1s,x_test,y_data,y_err):
    
    
    
    theta = dict(
        amp = np.max(y_data),
        cen = 0.0,
        wid = 1.0
        )
    lower_bounds = dict(
        amp = np.max(y_data)*0.95,
        cen = -1.0,
        wid = 0.7
        )
    upper_bounds = dict(
        amp = 2*np.max(y_data),
        cen = 1.0,
        wid = 1.3,
        )
    bounds = (lower_bounds, upper_bounds)
    
    theta_LSF, LSF_x, LSF_y, LSF_yerr = hread.LSF_from_lsf1s(lsf1s)
    scatter = hread.scatter_from_lsf1s(lsf1s)
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=partial(loss,
                                                x_test=jnp.array(x_test),
                                                y_data=jnp.array(y_data),
                                                y_err=jnp.array(y_err),
                                                theta_LSF=theta_LSF,
                                                X = jnp.array(LSF_x),
                                                Y = jnp.array(LSF_y),
                                                Y_err = jnp.array(LSF_yerr),
                                                scatter=scatter),
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
    
    # optpars = solution.params
    # print(optpars, loss(optpars,x_test,y_data,y_err,theta_LSF,LSF_x,LSF_y,LSF_yerr))
    
    # import matplotlib.pyplot as plt
    # from scipy.optimize import curve_fit
    # import harps.functions as hf
    # model = return_model(optpars,x_test,theta_LSF,LSF_x,LSF_y,LSF_yerr)
    # dof = len(model) - (len(optpars)+len(theta_LSF))
    # chisq= loss(optpars,x_test,y_data,y_err,theta_LSF,LSF_x,LSF_y,LSF_yerr,scatter=scatter)
    # print(f"LSF chisq = {chisq/dof}")
    
    # fig, (ax1,ax2) = plt.subplots(2,1)
    # ax1.errorbar(x_test,y_data,y_err,marker='o',ms=2,capsize=2)
    # ax2.scatter(x_test,(y_data-model)/y_err)

    
    # x_grid = np.linspace(x_test.min(),x_test.max(),100)
    # # x_grid = x_test
    # model = return_model(optpars,x_grid,theta_LSF,LSF_x,LSF_y,LSF_yerr)
    # ax1.plot(x_grid,model,lw=3,c='r')
    
    
    
    # p0 = (np.max(y_data),0,np.std(x_test),0)
    # popt,pcov = curve_fit(hf.gauss4p,x_test,y_data,sigma=y_err,
    #                       absolute_sigma=False,p0=p0)
    # gauss_Y = hf.gauss4p(x_test,*popt)
    # # gauss_Y_err = hf.error_from_covar(hf.gauss4p, popt, pcov, x_test)
    # ax1.plot(x_grid, hf.gauss4p(x_grid,*popt), c="C3",ls=':',
    #           label="Gaussian model",lw=2,zorder=3)
    # ax2.scatter(x_test, (gauss_Y-y_data)/y_err)
    # print(f"Gaussian chisq = {np.sum(((gauss_Y-y_data)/y_err)**2)/dof}")
    
    # plt.show()
    return solution.params

def fit_lsf(pix,flux,background,error,lsf1s,
        output_model=False,plot=False,*args,**kwargs):
    """
    lsf1d must be an instance of LSF class and contain only one segment 
    (see harps.lsf)
    """
    # import harps.lsf.gp_aux as gp_aux
    # lsf1s = lsf1s.values
    # print(hread.LSF_from_lsf1s(lsf1s))
    # sys.exit()
    x_test = jnp.array(pix,dtype=jnp.float32)
    y_data = jnp.array(flux-background,dtype=jnp.float32)
    y_err  = jnp.array(error,dtype=jnp.float32)
    
    theta_LSF, LSF_x, LSF_y, LSF_yerr = hread.LSF_from_lsf1s(lsf1s)
    
    # LSF_x = jnp.array(LSF_x)
    # LSF_y = jnp.array(LSF_y)
    # LSF_yerr = jnp.array(LSF_yerr)
    
    try:
        optpars = get_parameters(lsf1s,x_test,y_data,y_err)
        pcov = None
        success = True
    except:
        optpars = dict(amp=np.nan,cen=np.nan,wid=np.nan)
        success = False
    
    if success:   
        amp = optpars['amp']
        cen = optpars['cen']
        wid = optpars['wid']
        
        chisq   = loss(optpars, x_test,y_data,y_err,
                       theta_LSF,LSF_x, LSF_y, LSF_yerr)
        dof  = len(pix) - (len(optpars)+len(theta_LSF))
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
        dof  = len(pix)
    pars    = np.array([amp, cen, wid])
    errors  = np.sqrt(np.diag(pcov))
    chisqnu = chisq/dof
    
    #pars[0]*interpolate.splev(pix+pars[1],splr)+background
    if plot:
        plot_result(optpars,lsf1s,pix,flux,background,error)
    if output_model:  
        model   = return_model(optpars,x_test,theta_LSF,LSF_x,LSF_y,LSF_yerr)
        return success, pars, errors, chisq, chisqnu, model
    else:
        return success, pars, errors, chisq, chisqnu
    
def plot_result(optpars,lsf1s,pix,flux,background,error):
    import matplotlib.pyplot as plt
    theta_LSF, LSF_x, LSF_y, LSF_yerr = hread.LSF_from_lsf1s(lsf1s)
    model   = return_model(optpars,pix,theta_LSF,LSF_x,LSF_y,LSF_yerr)
    # plotter = Figure2(2,1,height_ratios=[3,1])
    # ax0     = plotter.add_subplot(0,1,0,1)
    # ax1     = plotter.add_subplot(1,2,0,1,sharex=ax0)
    fig, (ax1,ax2) = plt.subplots(2,1)
    
    ax1.set_title('fit.lsf')
    ax1.plot(pix,flux-background,label='Flux',drawstyle='steps-mid')
    ax1.plot(pix,model,label='Model',drawstyle='steps-mid')
    
    x_grid = np.linspace(pix.min(),pix.max(),400)
    model_grid = return_model(optpars,x_grid,theta_LSF,LSF_x,LSF_y,LSF_yerr)
    ax1.plot(x_grid,model_grid,lw=2)
    
    ax2.scatter(pix,((flux-background)-model)/error,marker='s')
    [ax2.axhline(i,ls='--',lw=1) for i in [-1,0,1]]
    ax2.set_ylim(-5,5)
    ax1.legend()