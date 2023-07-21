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



def line(x1l,flx1l,err1l,bary,LSF1d,scale,interpolate=True,
        output_model=False,output_rsd=False,plot=False,*args,**kwargs):
    """
    lsf1d must be an instance of LSF class and contain only one segment 
    (see harps.lsf)
    """
    def residuals(x0,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y):
        model_data    = lsf_model(lsf_loc_x,lsf_loc_y,x0,x1l,scale)
        model_scatter = sct_model(sct_loc_x,sct_loc_y,x0,x1l,scale)
#        weights  = np.ones_like(pix)
        weights  = assign_weights(x1l,x0[1],scale)
        
        rescaled_error = err1l * model_scatter
        # rescaled_error = err1l*model_scatter
        # resid = (flx1l - bkg1l - model_data) / rescaled_error
        resid = (flx1l - model_data) / rescaled_error * weights
        #resid = line_w * (counts- model)
        return resid
    
    centroid = np.average(x1l,weights=flx1l)
    p0 = (np.max(flx1l),centroid,1.)
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
    chisqnu = cost/dof
    
    model   = lsf_model(lsf_loc_x,lsf_loc_y,pars,x1l,scale)
    
    integral = np.sum(model)
    output_tuple = (success, pars, errors, cost, chisqnu, integral)
    
    if plot:
        fig = hplt.Figure2(2,1,figize=(8,4),height_ratios=[3,1])
        ax1 = fig.add_subplot(0,1,0,1)
        ax2 = fig.add_subplot(1,2,0,1,sharex=ax1)
        ax1.errorbar(x1l,flx1l,err1l,drawstyle='steps-mid',capsize=3)
        ax1.plot(x1l,model,drawstyle='steps-mid',marker='x')
        ax1.text(0.8,0.9,r'$\chi^2_\nu=$'+f'{chisqnu:8.2f}',transform=ax1.transAxes)
        # ax1.axvspan(pars[1]-2.5,pars[1]+2.5,alpha=0.1)
        # ax1.axvspan(pars[1]-5,pars[1]+5,alpha=0.1)
        ax2.scatter(x1l,(flx1l-model)/err1l,label='rsd')
        ax2.scatter(x1l,infodict['fvec'],label='infodict')
        xgrid = np.linspace(x1l.min(), x1l.max(), 100)
        ygrid = lsf_model(lsf_loc_x,lsf_loc_y,pars,xgrid,scale)
        ax1.plot(xgrid,ygrid,c='k',lw=2)
        ax1.set_title(scale)
        ax2.legend()
    
    if output_model:  
        output_tuple =  output_tuple + (model,)
    if output_rsd:  
        output_tuple =  output_tuple + (infodict['fvec'],)
    return output_tuple
    
def lsf_model(lsf_loc_x,lsf_loc_y,pars,xarray,scale):
    """
    Returns the model of the data from the LSF and parameters provided. 
    Does not include the background.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    try:
        amp, cen, wid = pars
        e0 = 0.
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

def assign_weights(xarray,center,scale):
    def f(x,x1,x2): 
        # a linear function going through x1 and x2
        return x/(x2-x1)-x1/(x2-x1)
    
    weights  = np.zeros_like(xarray)
    if scale[:3]=='pix':
        dx = [-5,-2.5,2.5,5] # units pix
        binlims = dx + center
    elif scale[:3]=='vel':
        varray = (xarray-center)/center * 299792.458 # units km/s
        dv     = np.array([-4,-2,2,4]) # units km/s
        binlims = center * (1 + dv/299792.458) # units wavelength
        
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
    weights[cut1] = 1
    weights[cutl] = [f(x,binlims[0],binlims[1]) for x in xarray[cutl]]
    weights[cutr] = [f(x,binlims[3],binlims[2]) for x in xarray[cutr]]
    return weights