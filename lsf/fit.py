#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:00:06 2023

@author: dmilakov
"""
import numpy as np
import jax.numpy as jnp
import harps.lsf.read as hread
import harps.lsf.gp as hlsfgp
import harps.containers as container
import harps.progress_bar as progress_bar
import harps.plotter as hplt
import matplotlib.pyplot as plt
import logging
from scipy.optimize import leastsq
import scipy.interpolate as interpolate
from matplotlib import ticker



def line(x1l,flx1l,err1l,bary,LSF1d,scale,interpolate=True,
        output_model=False,output_rsd=False,plot=False,save_fig=None,
        rsd_range=None,
        *args,**kwargs):
    """
    lsf1d must be an instance of LSF class and contain only one segment 
    (see harps.lsf)
    """
    def residuals(x0,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y):
        model_data    = lsf_model(lsf_loc_x,lsf_loc_y,x0,x1l,scale)
        model_scatter = sct_model(sct_loc_x,sct_loc_y,x0,x1l,scale)
#        weights  = np.ones_like(pix)
        weights  = assign_weights(x1l,x0[1],scale)
        
        rescaled_error = err1l 
        # rescaled_error = err1l*model_scatter
        # resid = (flx1l - bkg1l - model_data) / rescaled_error
        resid = (flx1l - model_data) / rescaled_error * weights
        #resid = line_w * (counts- model)
        return resid
    
    centroid = np.average(x1l,weights=flx1l)
    if container.npars==3:
        p0 = (np.max(flx1l),centroid,1.)
    elif container.npars==4:
        p0 = (np.max(flx1l),centroid,1.,0)
    N = 2 if interpolate else 1
    lsf_loc_x,lsf_loc_y = LSF1d.interpolate_lsf(bary,N)
    sct_loc_x,sct_loc_y = LSF1d.interpolate_scatter(bary,N)
    # plt.figure(); plt.plot(sct_loc_x,sct_loc_y)
    
    popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                                        args=(lsf_loc_x,lsf_loc_y,
                                              sct_loc_x,sct_loc_y),
                                        ftol=1e-10,
                                        full_output=True)
    
    if ier not in [1, 2, 3, 4]:
        
        print("Optimal parameters not found: " + errmsg)
        popt = np.full_like(p0,np.nan)
        pcov = None
        success = False
    else:
        success = True
    
    if success:
        # amp, cen, wid, a0, a1 = popt
        cost = np.sum(infodict['fvec']**2)
        dof  = (len(x1l) - len(popt))
        if pcov is not None:
            pcov = pcov*cost/dof
        else:
            pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
        #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
        
    else:
        popt = np.full_like(p0,np.nan)
        pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
        cost = np.nan
        dof  = (len(x1l) - len(popt))
        success=False
    pars    = popt
    errors  = np.sqrt(np.diag(pcov))
    # chisqnu = cost/dof
    
    model    = lsf_model(lsf_loc_x,lsf_loc_y,pars,x1l,scale)
    within   = within_limits(x1l,pars[1],scale)
    
    chisq, dof = get_chisq_dof(x1l,flx1l,err1l,model,pars,scale)
    chisqnu = chisq / dof
    # _        = np.where((rsd_norm<10)&(within))
    # print(_)
    integral = np.sum(model[within])
    output_tuple = (success, pars, errors, cost, chisqnu, integral)
    # print(cost,(len(x1l) - len(popt)),cost/(len(x1l) - len(popt)),chisq,dof,chisq/dof)
    if plot:
        fig = plot_fit(x1l,flx1l,err1l,model=model,#rsd_norm=rsd_norm,
                       pars=pars,scale=scale,
                       lsf_loc_x=lsf_loc_x,lsf_loc_y=lsf_loc_y,
                       rsd_range=rsd_range,
                       **kwargs)
    if output_model:  
        output_tuple =  output_tuple + (model,)
    if output_rsd:  
        output_tuple =  output_tuple + (infodict['fvec'],)
    if plot:
        output_tuple =  output_tuple + (fig,)
        
    return output_tuple

def get_chisq_dof(x1l,flx1l,err1l,model,pars,scale):
    rsd_norm = (model-flx1l)/err1l
    
    within   = within_limits(x1l,pars[1],scale)
    dof      = len(within)-len(pars)
    chisq    = np.sum(rsd_norm[within]**2)
    chisqnu  = chisq/dof
    return chisq,dof

def line_gauss(x1l,flx1l,err1l,bary,LSF1d,scale,interpolate=True,
        output_model=False,output_rsd=False,plot=False,save_fig=None,
        rsd_range=None,
        *args,**kwargs):
    from harps.fit import gauss as fit_gauss
    
    output_gauss = fit_gauss(x1l, flx1l, err1l, 
                       model='SimpleGaussian', 
                       xscale=scale,
                       output_model=False)
    success, pars, errors, chisq, chisqnu,integral = output_gauss
    output_tuple = (success, pars, errors, chisq, chisqnu, integral)
    try:
        A, mu, sigma,e0 = pars
    except:
        A, mu, sigma = pars
    
    model = A*np.exp(-0.5*(x1l-mu)**2/sigma**2)
    try:
        model += e0*x1l
    except:
        pass
    label = r'Gaussian IP'
    rsd_norm = np.abs((model-flx1l)/err1l)
    if plot:
        fig = plot_fit(x1l,flx1l,err1l,model,pars,scale,
                       # rsd_norm=rsd_norm,
                       rsd_range=rsd_range,
                       is_gaussian=True,
                       **kwargs)
    if output_model:  
        output_tuple =  output_tuple + (model,)
    if output_rsd:  
        output_tuple =  output_tuple + ((flx1l-model)/err1l,)
    if plot:
        output_tuple =  output_tuple + (fig,)
        
    return output_tuple

def plot_fit(x1l,flx1l,err1l,model,pars,scale,is_gaussian=False,
             rsd_norm=None,
             rsd_range=None,**kwargs):
    def func_pixel(x):
        return x - pars[1]
    def inverse_pixel(x):
        return x + pars[1]
    def func_velocity(x):
        return (x/pars[1] - 1)*299792.458
    def inverse_velocity(x):
        return pars[1]*(1+x/299792.458)
    
    axes = kwargs.pop('axes',None)
    ax_sent = True if axes is not None else False
    if ax_sent:
        ax1, ax2 = axes
    else:
        default_args = dict(figsize=(5,4.5),height_ratios=[3,1],
                           left=0.12,
                           bottom=0.12,
                           top=0.9,
                           hspace=0.02
            )
        fig_args = {**default_args,**kwargs}
        fig = hplt.Figure2(2,1,**fig_args)
        ax1 = fig.add_subplot(0,1,0,1)
        ax2 = fig.add_subplot(1,2,0,1,sharex=ax1)
    
    
    
    within   = within_limits(x1l,pars[1],scale)
    
    ax1.errorbar(x1l,flx1l,err1l,drawstyle='steps-mid',marker='.',
                 label='Data')
    ax1.plot(x1l[within],model[within],drawstyle='steps-mid',marker='x',lw=3,
             label='Model')
    ax1.axvline(pars[1],ls=':',c='k',lw=2)
    
    
    
    # rsd_norm = ((flx1l-model)/err1l)#[within]
    # dof = np.sum(within)-len(pars)
    # chisq = np.sum(rsd_norm**2)
    # chisqnu = chisq/dof
    rsd_norm = rsd_norm if rsd_norm is not None else (model-flx1l)/err1l
    # dof  = (len(x1l) - len(pars))
    # within   = within_limits(x1l,pars[1],scale)
    # chisq = np.sum(rsd_norm[within]**2)
    # chisqnu = chisq/dof
    chisq, dof = get_chisq_dof(x1l,flx1l,err1l,model,pars,scale)
    chisqnu = chisq / dof
    # print(chisq,dof,chisqnu)
    ax1.text(0.95,0.9,r'$\chi^2_\nu=$'+f'{chisqnu:8.2f}',
             ha='right',va='baseline',
             transform=ax1.transAxes)
    if scale[:3]=='pix':
        dx1, dx2 = 5, 2.5
    elif scale[:3]=='vel':
        dv     = np.array([2,4]) # units km/s
        dx1, dx2 = pars[1] *  dv/299792.458
    ax1.axvspan(pars[1]-dx1,pars[1]+dx1,alpha=0.1)
    ax1.axvspan(pars[1]-dx2,pars[1]+dx2,alpha=0.1)
    ax1.xaxis.tick_bottom()
    
    if 'pix' in scale:
        functions = (func_pixel,inverse_pixel)
        for ax in [ax1,ax2]:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5,integer=True))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    elif 'vel' in scale or 'wav' in scale:
        functions = (func_velocity,inverse_velocity)
        for ax in [ax1,ax2]:
            ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax_top = ax1.secondary_xaxis('top', functions=functions)
    
    ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_top.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_top.set_xlabel(r'$\Delta x$'+f' ({scale[:3]})',labelpad=3)
    
    # ax2.scatter(x1l,infodict['fvec'],label='infodict')
    xgrid = np.linspace(x1l.min(), x1l.max(), 100)
    lsf_loc_x = kwargs.pop('lsf_loc_x',None)
    lsf_loc_y = kwargs.pop('lsf_loc_y',None)
    if lsf_loc_x is not None and lsf_loc_y is not None:
        
        ygrid = lsf_model(lsf_loc_x,lsf_loc_y,pars,xgrid,scale)
        label = r'$\psi(\Delta x)$'
    if is_gaussian:
        try:
            A, mu, sigma,e0 = pars
        except:
            A, mu, sigma = pars
        
        ygrid = A*np.exp(-0.5*(xgrid-mu)**2/sigma**2)
        try:
            ygrid += e0*xgrid
        except:
            pass
        label = r'Gaussian IP'
    ax1.plot(xgrid,ygrid,c='grey',lw=2,ls='--',label=label)    
    ax1.legend(loc='upper left')
    weights = assign_weights(x1l,pars[1],scale)[within]
    # rsd  = (flx1l-model)/err1l
    ax2.scatter(x1l[within],rsd_norm[within],label='rsd',
                edgecolor='k',color='w')
    ax2.scatter(x1l[within],rsd_norm[within],label='rsd',marker='o',
                alpha=weights)
    
    
    ax2.axhspan(-1,1,color='grey',alpha=0.3)
    # ylim = np.min([1.5*np.nanpercentile(np.abs(rsd_norm),95),10.3])
    # ylim = 1.8*np.max(np.abs(rsd_norm))
    ylim = rsd_range if rsd_range is not None else 1.8*np.max(np.abs(rsd_norm[within]))
    # rsd_range = kwargs.pop('rsd_range',False)
    print(ylim,rsd_range)
    # if rsd_range:
        # ylim = rsd_range
    ax2.set_ylim(-ylim,ylim)
    if 'pix' in scale:
        ax2.set_xlabel('Pixel')
    elif 'vel' in scale or 'wav' in scale:
        ax2.set_xlabel(r'Wavelength (\AA)')
    # ax2.set_xlabel(f"{scale.capitalize()}")
    ax1.set_ylabel(r"Intensity ($e^-$)")
    ax2.set_ylabel("Residuals "+r"($\sigma$)")
    
    for x,r,w in zip(x1l[within],rsd_norm[within],weights):
        ax2.text(x,r+0.1*ylim*2,f'{w:.2f}',
                  horizontalalignment='center',
                  verticalalignment='center',
                  fontsize=8)
    if not ax_sent:
        fig.ticks_('major', 1,'y',ticknum=3)
        fig.scinotate(0,'y',)
    # ax2.legend()
        return fig
    else:
        return ax1,ax2
    
    
def lsf_model(lsf_loc_x,lsf_loc_y,pars,xarray,scale):
    """
    Returns the model of the data from the LSF and parameters provided. 
    Does not include the background.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    try:
        amp, cen, wid = pars
    except:
        amp, cen, wid, e0  = pars
    wid   = np.abs(wid)
    x     = lsf_loc_x * wid
    y     = lsf_loc_y / np.max(lsf_loc_y) 
    splr  = interpolate.splrep(x,y) 
    
    if scale[:3]=='pix':
        x_test = xarray-cen
    elif scale[:3]=='vel':
        x_test = (xarray-cen)/cen*299792.458
    try:
        model = amp*interpolate.splev(x_test,splr) + e0*x_test
    except:
        model = amp*interpolate.splev(x_test,splr)
    return model

def sct_model(sct_loc_x,sct_loc_y,pars,xarray,scale):
    """
    Returns the model of the data from the LSF and parameters provided. 
    Does not include the background.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    try:
        amp, cen, wid = pars
    except:
        amp, cen, wid, a0  = pars
    wid   = np.abs(wid)
    x     = sct_loc_x * wid
    y     = sct_loc_y 
    splr  = interpolate.splrep(x,y)
    
    if scale[:3]=='pix':
        x_test = xarray-cen
    elif scale[:3]=='vel':
        x_test = (xarray-cen)/cen*299792.458
    model = interpolate.splev(x_test,splr)
    return np.exp(model/2.)

def within_limits(xarray,center,scale):
    '''
    Returns a boolean array of length len(xarray), indicating whether the 
    xarray values are within fitting limits.

    Parameters
    ----------
    xarray : array-like
        x-coordinates of the line.
    center : scalar
        centre of the line.
    scale : string
        'pixel or 'velocity'.

    Returns
    -------
    array-like
        A boolean array of length len(xarray). Elements equals True when 
        xarray values are within fitting limits.

    '''
    binlims = get_binlimits(xarray, center, scale)
    low  = np.min(binlims)
    high = np.max(binlims)
    return (xarray>=low)&(xarray<=high)

def get_binlimits(xarray,center,scale):
    if scale[:3]=='pix':
        dx = np.array([-5,-2.5,2.5,5]) # units pix
        binlims = dx + center
    elif scale[:3]=='vel':
        varray = (xarray-center)/center * 299792.458 # units km/s
        # dv     = np.array([-5,-2.5,2.5,5]) # units km/s
        dv = np.array([-4,-2,2,4])
        binlims = center * (1 + dv/299792.458) # units wavelength
    return binlims

def assign_weights(xarray,center,scale):
    def f(x,x1,x2): 
        # a linear function going through x1 and x2
        return np.abs((x-x1)/(x2-x1))
    
    weights  = np.zeros_like(xarray,dtype=np.float64)
    binlims = get_binlimits(xarray, center, scale)
        
    idx      = np.digitize(xarray,binlims)
    cut1     = np.where(idx==2)[0]
    cutl     = np.where(idx==1)[0]
    cutr     = np.where(idx==3)[0]
    # ---------------
    # weights are:
    #  = 1,           -2.5<=x<=2.5
    #  = 0,           -5.0>=x & x>=5.0
    #  = linear[0-1]  -5.0<x<-2.5 & 2.5>x>5.0 
    # ---------------
    weights[cutl] = f(xarray[cutl],binlims[0],binlims[1])
    weights[cutr] = f(xarray[cutr],binlims[3],binlims[2])
    weights[cut1] = 1
    return weights