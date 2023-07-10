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
import matplotlib.pyplot as plt
import logging
from scipy.optimize import leastsq
import scipy.interpolate as interpolate



def line(x1l,flx1l,bkg1l,err1l,LSF1d,interpolate=True,npars=3,
        output_model=False,output_rsd=False,*args,**kwargs):
    """
    lsf1d must be an instance of LSF class and contain only one segment 
    (see harps.lsf)
    """
    def residuals(x0,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y):
        model_data    = lsf_model(lsf_loc_x,lsf_loc_y,x0,x1l)
        model_scatter = sct_model(sct_loc_x,sct_loc_y,x0,x1l)
#        weights  = np.ones_like(pix)
        weights  = assign_weights(x1l-x0[1])
        try:
            rescaled_error = err1l * x0[3] * model_scatter
        except:
            rescaled_error = err1l
        # rescaled_error = err1l*model_scatter
        # resid = (flx1l - bkg1l - model_data) / rescaled_error
        resid = (flx1l - bkg1l - model_data) / rescaled_error * weights
        #resid = line_w * (counts- model)
        return resid
    
    bary = np.average(x1l,weights=flx1l)
    if npars==3:
        p0 = (np.max(flx1l-bkg1l),bary,1.)
    elif npars==4:
        p0 = (np.max(flx1l-bkg1l),bary,1.0,1.,)
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
    
    #pars[0]*interpolate.splev(pix+pars[1],splr)+background
#    plt.figure()
#    plt.title('fit.lsf')
#    plt.plot(pix,flux)
#    plt.plot(pix,model)
    model   = lsf_model(lsf_loc_x,lsf_loc_y,pars,x1l) + bkg1l
    
    integral = np.sum(model)-np.sum(bkg1l)
    output_tuple = (success, pars, errors, cost, chisqnu, integral)
    if output_model:  
        output_tuple =  output_tuple + (model,)
    if output_rsd:  
        output_tuple =  output_tuple + (infodict['fvec'],)
    return output_tuple
    
def lsf_model(lsf_loc_x,lsf_loc_y,pars,pix):
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
    
    x_test = pix-cen
    model = amp*interpolate.splev(x_test,splr)
    return model

def sct_model(sct_loc_x,sct_loc_y,pars,pix):
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
    
    x_test = pix-cen
    model = interpolate.splev(x_test,splr)
    return np.exp(model/2.)

def assign_weights(pixels):
        weights  = np.zeros_like(pixels)
        binlims  = [-5,-2.5,2.5,5]
        idx      = np.digitize(pixels,binlims)
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
        weights[cutl] = 0.4*(5+pixels[cutl])
        weights[cutr] = 0.4*(5-pixels[cutr])
        return weights