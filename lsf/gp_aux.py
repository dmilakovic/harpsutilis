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

# def get_LSF_model(theta,LSF_model,x_test):
#     amp = theta['amp']
#     cen = theta['cen']
#     wid = jnp.abs(theta['wid'])
    
    
    
#     x     = cen + (x_test* wid)
#     y     = amp * (LSF_model / np.max(LSF_model))
    
#     return x,y
def get_segment_centres(lsf1d):
    segcens = (lsf1d['pixl']+lsf1d['pixr'])/2
    # segcens[0]  = lsf1d['pixl'][0]
    # segcens[-1] = lsf1d['pixr'][-1]
    return segcens

def get_boundary_segments(center,lsf1d):
    segcens = get_segment_centres(lsf1d)
    # segcens = get_segment_centres(lsf1d)
    # print(segcens)
    seg_r   = np.digitize(center,segcens)
    #assert seg_r<len(segcens), "Right segment 'too right', {}".format(seg_r)
    if seg_r<len(segcens):
        pass
    else:
        seg_r = len(segcens)-1
    seg_l   = seg_r-1
    return seg_l, seg_r

def get_segment_weights(center,lsf1d,N=2):
    segcens   = get_segment_centres(lsf1d)
    segdist   = np.diff(segcens)[0] # assumes equally spaced segment centres
    distances = np.abs(center-segcens)
    # segments  = np.argsort(distances)[:N]
    # M = 2 if N<2 else N
    if N>1:
        used  = distances<segdist*(N-1)
    else:
        used  = distances<segdist/2.
    inv_dist  = 1./distances
    
    segments = np.where(used)[0]
    weights  = inv_dist[used]/np.sum(inv_dist[used])
    
    # if center>=np.min(segcens) & center<=np.max(segcens):
    #     weights = np.abs((segcens-center)/segdist)
    # f1      = (segcens[seg_r]-center)/(segcens[seg_r]-segcens[seg_l])
    # f2      = (center-segcens[seg_l])/(segcens[seg_r]-segcens[seg_l])
    
    # weights = [f1,f2]
    
    return segments, weights

#% https://github.com/google/jax/issues/4572#issuecomment-709809897
#% https://github.com/google/jax/issues/1922
def some_hash_function(x):
  return int(jnp.sum(x))

class HashableArrayWrapper:
  def __init__(self, val):
    self.val = val
  def __hash__(self):
    return some_hash_function(self.val)
  def __eq__(self, other):
    return (isinstance(other, HashableArrayWrapper) and
            jnp.all(jnp.equal(self.val, other.val)))

def gnool_jit(fun, static_array_argnums=()):
  @partial(jax.jit, static_argnums=static_array_argnums)
  def callee(*args):
    args = list(args)
    for i in static_array_argnums:
      args[i] = args[i].val
    return fun(*args)

  def caller(*args):
    args = list(args)
    for i in static_array_argnums:
      args[i] = HashableArrayWrapper(args[i])
    return callee(*args)

  return caller

@jax.jit
def loss(theta,x_test,y_data,y_err,list_theta_LSF,
         list_LSF_x,list_LSF_y,list_LSF_yerr,weights):
    # N = len(theta_LSF)   
    # models = []
    # for i in range(N):
    #     f   = return_model(theta,x_test,theta_LSF[i], 
    #                        LSF_x[i], LSF_y[i], LSF_yerr[i])
    #     models.append(f)
    # # models = [return_model(theta,x_test,lsf1d[segm]) for segm in segments]
    # weights_= jnp.vstack([jnp.full_like(x_test,w) for w in weights])
    # model  = jnp.average(models,axis=0,weights=weights_)
    model,m_err   = return_model(theta,x_test,list_theta_LSF,
                           list_LSF_x,list_LSF_y,list_LSF_yerr,weights)
    error   = jnp.sqrt(jnp.sum(jnp.array([jnp.power(_,2) for _ in [m_err,y_err]]),axis=0))
    # print(m_err,y_err,error)
    
    rsd     = (y_data - model)/error
    chisq   = jnp.sum(rsd**2)
    
    return chisq
    
# @jax.jit
def return_model(theta,x_test,list_theta_LSF,list_LSF_x,list_LSF_y,
                 list_LSF_yerr,weights):
    # print(lsf1s)
    try:
        amp = theta['amp']
        cen = theta['cen']
        wid = jnp.abs(theta['wid'])
    except:
        amp, cen, wid = theta
        wid = jnp.abs(wid)
    
    x   = jnp.array((x_test-cen) * wid)
    
    M = len(weights)
    if isinstance(list_theta_LSF,list):
        N = len(list_theta_LSF)
    elif isinstance(list_theta_LSF,dict):
        N = 1
        list_theta_LSF = list(list_theta_LSF)
    assert N==M
    # N = len(list_theta_LSF)
    model_list = []
    error_list = []
    # print('amp,cen,wid',amp,cen,wid)
    # print('0',model_list)
    for i in range(N):
        mean, error = hlsfgp.get_model(x,list_LSF_x[i],list_LSF_y[i],
                                     list_LSF_yerr[i],list_theta_LSF[i],
                                     scatter=None)
        model_list.append(mean)
        error_list.append(error)
    weights_= jnp.vstack([jnp.full_like(x_test,w,dtype='float32') for w in weights])
    # print('1',model_list)
    # print('2',weights_)
    model_ = jnp.average(model_list,axis=0,weights=weights_)  
    error_ = jnp.sqrt(jnp.sum(jnp.array([jnp.power(_,2) for _ in error_list]),
                              axis=0))
    
    normalisation = amp / jnp.max(model_)
    model = model_ * normalisation
    error = error_ * normalisation
    return model, error

@jax.jit
def return_model_bk2(theta,x_test,theta_LSF, LSF_x, LSF_y, LSF_yerr):
    # print(lsf1s)
    try:
        amp = theta['amp']
        cen = theta['cen']
        wid = jnp.abs(theta['wid'])
    except:
        amp, cen, wid = theta
        wid = jnp.abs(wid)
    
    x   = jnp.array(cen + (x_test * wid))
    
    mean, var = hlsfgp.get_model(x,LSF_x,LSF_y,LSF_yerr,theta_LSF,scatter=None)
    # gp_LSF = hlsfgp.build_LSF_GP(theta_LSF, X, Y, Y_err)
    # mean, var = evaluate_GP(gp_LSF,Y, x)
    
    model_y = amp * (mean / jnp.max(mean))
    return model_y
    
@jax.jit
def loss_bk(theta,x_test,y_data,y_err,theta_LSF,X,Y,Y_err,scatter=None):
    
    # rsd = hlsfgp.get_residuals(x_test, Y, Y_err, theta_LSF)
    
    # model_x, model_y = get_LSF_model(theta, lsf1s, x_test)
    model_y = return_model_bk(theta, x_test, theta_LSF, X, Y, Y_err)
    if scatter is not None:
        S, S_var = hlsfgp.rescale_errors(scatter,x_test,y_err,plot=False)
        error = S
    else:
        error = y_err
    
    rsd     = (y_data - model_y)/error
    chisq   = np.sum(rsd**2)
    
    return chisq
# @jax.jit
def return_model_bk(theta,x_test,theta_LSF,X,Y,Y_err):
    
    try:
        amp = theta['amp']
        cen = theta['cen']
        wid = jnp.abs(theta['wid'])
    except:
        amp, cen, wid = theta
        wid = jnp.abs(wid)
    
    x   = jnp.array((x_test-cen) * wid)
    
    mean, var = hlsfgp.get_model(x,X,Y,Y_err,theta_LSF,scatter=None)
    # gp_LSF = hlsfgp.build_LSF_GP(theta_LSF, X, Y, Y_err)
    # mean, var = evaluate_GP(gp_LSF,Y, x)
    
    model_y = amp * (mean / np.max(mean))
    
    return model_y


def get_params_scipy(lsf1s,x_test,y_data,y_err,interpolate=False,*args,**kwargs):
    
    # x_test = jnp.array(pix,dtype=jnp.float32)
    # y_data = jnp.array(flux-background,dtype=jnp.float32)
    # y_err  = jnp.array(error,dtype=jnp.float32)
    
    
    def residuals(theta):
        model_y, model_err = return_model(theta, x_test, theta_LSF, LSF_x,LSF_y,LSF_yerr)
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
    optpars,pcov,infodict,errmsg,ier = leastsq(loss,x0=theta,
                                               args=(x_test,y_data,y_err,lsf1s),
                                               full_output=True)
    
    if ier not in [1, 2, 3, 4]:
        print("Optimal parameters not found: " + errmsg)
        optpars = np.full_like(theta,np.nan)
        pcov = None
        success = False
    else:
        success = True
    return optpars
    # if success:   
    #     amp, cen, wid = optpars
    #     chisq = np.sum(infodict['fvec']**2)
    #     dof  = (len(x_test) - (len(optpars)+len(theta_LSF)))
    #     if pcov is not None:
    #         pcov = pcov*chisq/dof
    #     else:
    #         pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
    #     #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
    # else:
    #     optpars = np.full_like(theta,np.nan)
    #     amp, cen, wid = optpars
    #     pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
    #     chisq = np.nan
    #     dof  = (len(x_test) - (len(optpars)+len(theta_LSF)))
    #     success=False
    # pars    = np.array([amp, cen, wid])
    # errors  = np.sqrt(np.diag(pcov))
    # chisqnu = chisq/dof
    
    # if output_model:  
    #     model   = return_model(optpars,x_test,theta_LSF,LSF_x,LSF_y,LSF_yerr)
    #     return success, pars, errors, chisq, chisqnu, model
    # else:
    #     return success, pars, errors, chisq, chisqnu

def get_parameters(lsf1d,x_test,y_data,y_err,interpolate=False):
    
    bary = np.average(x_test,weights=y_data)
    N = 2 if interpolate == True else 1
    _ = extract_LSF_lists(bary,lsf1d,N=N)
    list_theta_LSF, list_LSF_x, list_LSF_y, list_LSF_yerr, weights = _
    
    scatter = None
    # scatter = hread.scatter_from_lsf1s(lsf1s)
    # print(scatter)
    # print(LSF_x)
    # print(scatter)
    
    
    theta = dict(
        amp = np.max(y_data),
        cen = bary,
        wid = 1.0
        )
    lower_bounds = dict(
        amp = np.max(y_data)*0.8,
        cen = bary-1.0,
        wid = 0.9
        )
    upper_bounds = dict(
        amp = np.max(y_data)*1.2,
        cen = bary+1.0,
        wid = 1.1,
        )
    bounds = (lower_bounds, upper_bounds)
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=partial(loss,
                                                x_test=jnp.array(x_test),
                                                y_data=jnp.array(y_data),
                                                y_err=jnp.array(y_err),
                                                list_theta_LSF=list_theta_LSF,
                                                list_LSF_x = list_LSF_x,
                                                list_LSF_y = list_LSF_y,
                                                list_LSF_yerr = list_LSF_yerr,
                                                weights = jnp.array(weights)),
                                          method="l-bfgs-b")
    # lbfgsb = jaxopt.ScipyBoundedMinimize(fun=partial(loss_bk,
    #                                             x_test=jnp.array(x_test),
    #                                             y_data=jnp.array(y_data),
    #                                             y_err=jnp.array(y_err),
    #                                             theta_LSF=theta_LSF,
    #                                             X = jnp.array(LSF_x),
    #                                             Y = jnp.array(LSF_y),
    #                                             Y_err = jnp.array(LSF_yerr),
    #                                             scatter=scatter),
    #                                       method="l-bfgs-b")
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
    # optpars['cen']=optpars['cen']+bary
    # print(optpars, loss(optpars,x_test,y_data,y_err,theta_LSF,LSF_x,LSF_y,LSF_yerr))
    
    # import matplotlib.pyplot as plt
    # from scipy.optimize import curve_fit
    # import harps.functions as hf
    # model = return_model(optpars,x_test,theta_LSF,LSF_x,LSF_y,LSF_yerr)
    dof = len(x_test) - len(optpars) #- np.sum([len(_) for _ in list_theta_LSF])
    chisq= loss(optpars,x_test,y_data,y_err,list_theta_LSF,
                list_LSF_x,list_LSF_y,list_LSF_yerr,jnp.array(weights))
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
    return optpars, chisq, dof

def extract_LSF_lists(center,lsf1d,N=2):
    segments, weights = get_segment_weights(center,lsf1d,N)
    print(center,segments,weights)
    list_theta_LSF = []
    list_LSF_x = []
    list_LSF_y = []
    list_LSF_yerr = []
    
    for segm in segments:
        LSF_i = hread.LSF_from_lsf1s(lsf1d[segm])
        list_theta_LSF.append(LSF_i[0])
        list_LSF_x.append(jnp.array(LSF_i[1]))
        list_LSF_y.append(jnp.array(LSF_i[2]))
        list_LSF_yerr.append(jnp.array(LSF_i[3]))
    return list_theta_LSF, list_LSF_x, list_LSF_y, list_LSF_yerr, weights

def fit_lsf2line(x1l,flx1l,bkg1l,err1l,lsf1d,interpolate=True,
        output_model=False,plot=False,*args,**kwargs):
    
    bary = np.average(x1l,weights=flx1l)
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
    #pars[0]*interpolate.splev(pix+pars[1],splr)+background
    if plot:
        plot_result(optpars,lsf1d,x1l,flx1l,bkg1l,err1l)
    if output_model:  
        N = 2 if interpolate == True else 1
        _ = extract_LSF_lists(bary,lsf1d,N=N)
        list_theta_LSF, list_LSF_x, list_LSF_y, list_LSF_yerr, weights = _
        model,model_err   = return_model(optpars,x_test,list_theta_LSF,
                               list_LSF_x,list_LSF_y,
                               list_LSF_yerr,weights)
        return success, pars, errors, chisq, chisqnu, model
    else:
        return success, pars, errors, chisq, chisqnu
    
def plot_result(optpars,lsf1d,pix,flux,background,error,interpolate=True):
    import matplotlib.pyplot as plt
    pix = jnp.array(pix)
    
    # theta_LSF, LSF_x, LSF_y, LSF_yerr = hread.LSF_from_lsf1s(lsf1s)
    # print('plot',*[np.shape(_) for _ in [optpars,LSF_x,LSF_y,LSF_yerr]])
    # model   = return_model(optpars,pix,lsf1s)
    bary = np.average(pix,weights=flux)
    N = 2 if interpolate == True else 1
    _ = extract_LSF_lists(bary,lsf1d,N=N)
    list_theta_LSF, list_LSF_x, list_LSF_y, list_LSF_yerr, weights = _
    
    model,model_err = return_model(optpars,pix,list_theta_LSF,
                                   list_LSF_x,list_LSF_y,list_LSF_yerr,weights)
    rsd = ((flux-background)-model)/error
    # print(model)
    # plotter = Figure2(2,1,height_ratios=[3,1])
    # ax0     = plotter.add_subplot(0,1,0,1)
    # ax1     = plotter.add_subplot(1,2,0,1,sharex=ax0)
    fig, (ax1,ax2) = plt.subplots(2,1)
    
    ax1.set_title('fit.lsf')
    ax1.plot(pix,flux-background,label='Flux',drawstyle='steps-mid')
    ax1.plot(pix,model,label='Model',drawstyle='steps-mid')
    
    x_grid = np.linspace(pix.min(),pix.max(),400)
    model_grid,model_grid_err = return_model(optpars,x_grid,
                                             list_theta_LSF,list_LSF_x,
                                             list_LSF_y,list_LSF_yerr,weights)
    ax1.plot(x_grid,model_grid,lw=2)
    
    ax2.scatter(pix,rsd,marker='s')
    [ax2.axhline(i,ls='--',lw=1) for i in [-1,0,1]]
    ax2.set_ylim(-5,5)
    ax1.legend()
    
