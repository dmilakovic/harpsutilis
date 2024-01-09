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
from scipy.optimize import leastsq, least_squares
import scipy.interpolate as interpolate
from matplotlib import ticker


def residuals(pars,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y,
              x,scale,weight=False,obs=None,obs_err=None):
    amp, cen, wid, m, y0 = _unpack_pars(pars)
    
    model_data    = lsf_model(lsf_loc_x,lsf_loc_y,pars,x,scale) + \
                    m * (x-cen) + y0 
    model_scatter = sct_model(sct_loc_x,sct_loc_y,pars,x,scale)
    if not weight:
        within  = within_limits(x,cen,scale)
        weights = np.zeros_like(x)
        weights[within] = 1.
    else:
        weights  = assign_weights(x,cen,scale)
    
    
    if obs is not None:
        resid = (obs - model_data) * weights
        if obs_err is not None:
            rescaled_error = obs_err 
            # rescaled_error = err1l*model_scatter
            resid = resid / rescaled_error 
        return resid.astype(np.float64)
    else:
        return model_data
def residuals2(pars,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y,
              x,scale,weight=False,obs=None,obs_err=None):
    # amp, cen, wid, m, y0 = _unpack_pars(pars)
    cen = _unpack_pars(pars)[1]
    within  = within_limits(x,cen,scale)
    outside = ~within
    
    weights_line = np.zeros_like(x)
    weights_line[outside] = 1.
    if not weight:
        weights_lsf = np.zeros_like(x)
        weights_lsf[within] = 1.
    else:
        weights_lsf  = assign_weights(x,cen,scale)
    
    model_lsf    = lsf_model(lsf_loc_x,lsf_loc_y,pars,x,scale)# * weights_lsf
    # model_scatter = sct_model(sct_loc_x,sct_loc_y,pars,x,scale)
    # 
    
    model_data   = model_lsf #+ model_line
    if len(pars)>3:
        amp, cen, wid, m, y0 = _unpack_pars(pars)
        model_line   = (m * (x - cen) + y0)# * weights_line
        model_data = model_lsf + model_line
    # model_data   = np.vstack([model_lsf, model_line ])
    
    if obs is not None:
        resid = (obs - model_data) * weights_lsf
        # resid = np.vstack([(obs-model_lsf),
        #                    (obs-model_line)])
        if obs_err is not None:
            rescaled_error = obs_err 
            # rescaled_error = err1l*model_scatter
            resid = resid / rescaled_error 
        return resid.flatten()
    else:
        return model_data
    
def line(x1l,flx1l,err1l,bary,LSF1d,scale,weight=True,interpolate=True,
        output_model=False,output_rsd=False,plot=False,save_fig=None,
        rsd_range=None,npars=None,method='lmfit',bounded=False,
        *args,**kwargs):
    """
    lsf1d must be an instance of LSF class and contain only one segment 
    (see harps.lsf)
    """
    
        
    # def residuals_lmfit(pars,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y):
    #     model_data    = lsf_model(lsf_loc_x,lsf_loc_y,pars,x1l,scale)
    #     if weight:
    #         weights  = assign_weights(x1l,pars['cen'],scale)
    #     else:
    #         weights = np.ones_like(x1l)
        
        
    #     rescaled_error = err1l 
    #     resid = (flx1l - model_data) / rescaled_error * weights
    #     return resid
    
    
    
    N = 2 if interpolate else 1
    lsf_loc_x,lsf_loc_y = LSF1d.interpolate_lsf(bary,N)
    sct_loc_x,sct_loc_y = LSF1d.interpolate_scatter(bary,N)
    
    npars = npars if npars is not None else container.npars
    guess_amp = np.max(flx1l)
    guess_cen = np.average(x1l,weights=flx1l)
    p0 = _prepare_pars(npars,method,x1l,flx1l)
    if method=='scipy':
        
        if not bounded:
            popt,pcov,infodict,errmsg,ier = leastsq(residuals2,x0=p0,
                                            args=(lsf_loc_x,lsf_loc_y,
                                                  sct_loc_x,sct_loc_y,
                                                  x1l,scale,weight,
                                                  flx1l,err1l),
                                            ftol=1e-12,
                                            full_output=True)
        else:
            bounds = np.array([(0.8*guess_amp,1.2*guess_amp),
                      (guess_cen-0.5,guess_cen+0.5),
                      (0.9,1.1),
                      (-1e3,1e3),
                      (-1e3,1e3)])
            print(bounds)
            result = least_squares(residuals, x0=p0,
                                    bounds = bounds[:npars].T,
                                    args=(lsf_loc_x,lsf_loc_y,
                                          sct_loc_x,sct_loc_y),
                                    )
            pars = result['x']
            ier = [1]
            errmsg = ''
            print(result)
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
                pcov = np.diag([np.nan for i in range(len(popt))])
            #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
            rsd = infodict['fvec']
            
        else:
            popt = np.full_like(p0,np.nan)
            pcov = np.diag([np.nan for i in range(len(popt))])
            cost = np.nan
            dof  = (len(x1l) - len(popt))
            success=False
        pars    = popt
        errors  = np.sqrt(np.diag(pcov))
        # print(pars,errors)
    # chisqnu = cost/dof
    elif method=='lmfit':
        from lmfit import create_params, fit_report, minimize, Model
        # # fit_params = create_params(amp=guess_amp, cen=guess_cen, 
        # #                            wid=1., slope=0.0, offset=0)
        
        result = minimize(residuals2, p0, 
                        kws=dict(
                            lsf_loc_x=lsf_loc_x,
                            lsf_loc_y=lsf_loc_y,
                            sct_loc_x=sct_loc_x,
                            sct_loc_y=sct_loc_y,
                            x = x1l,
                            scale = scale,
                            obs = flx1l,
                            obs_err = err1l
                            ))
        # model = Model(model_profile) + Model(model_line)
        # result = model.fit(flx1l,params=p0,weights=1./err1l,
        #                    x_array=x1l,
        #                    lsf_loc_x=lsf_loc_x,
        #                    lsf_loc_y=lsf_loc_y,
        #                    scale=scale
        #                    )
        # print(fit_report(result))
        pars_obj = result.params
        pars = _unpack_pars(pars_obj)
        success = result.success
        covar = result.covar
        errors = np.sqrt(np.diag(covar))
        cost   = np.sum(result.residual**2)
        rsd    = result.residual
    # model    = lsf_model(lsf_loc_x,lsf_loc_y,pars,x1l,scale)
    model    = residuals2(pars,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y,
                         x=x1l,scale=scale,weight=weight,
                         obs=None,obs_err=None)
    within   = within_limits(x1l,pars[1],scale)
    
    chisq, dof = get_chisq_dof(x1l,flx1l,err1l,model,pars,scale)
    chisqnu = chisq / dof
    # _        = np.where((rsd_norm<10)&(within))
    # print(_)
    if len(np.shape(model))>1:
        integral = np.sum(model,axis=0)[within]
    else:
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
        output_tuple =  output_tuple + (rsd,)
    if plot:
        output_tuple =  output_tuple + (fig,)
        
    return output_tuple

def get_chisq_dof(x1l,flx1l,err1l,model,pars,scale):
    if len(np.shape(model))>1:
        rsd_norm = (np.sum(model,axis=0)-flx1l)/err1l
    else:
        rsd_norm = (model-flx1l)/err1l
    cen  = _unpack_pars(pars)[1]
    within   = within_limits(x1l,cen,scale)
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
    if len(pars)==5:
        A, mu, sigma, m, y0 = pars
    elif len(pars)==4:
        A, mu, sigma, y0 = pars
        m = 0.
    elif len(pars)==3:
        A, mu, sigma = pars
        m  = 0.
        y0 = 0.
    
    model = A*np.exp(-0.5*(x1l-mu)**2/sigma**2)
    try:
        model += m*(x1l-mu) + y0
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
    if len(np.shape(model))>1:
        for array,cut in zip(model,[within,~within]):
            ax1.plot(x1l[cut],array[cut],
                     drawstyle='steps-mid',marker='x',lw=3,
                     label='Model')
    else:
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
    xgrid = np.linspace(x1l.min(), x1l.max(), 300)
    lsf_loc_x = kwargs.pop('lsf_loc_x',None)
    lsf_loc_y = kwargs.pop('lsf_loc_y',None)
    if lsf_loc_x is not None and lsf_loc_y is not None:
        
        ygrid = lsf_model(lsf_loc_x,lsf_loc_y,pars,xgrid,scale)
        label = r'$\psi(\Delta x)$'
    if is_gaussian:
        pars = _unpack_pars(pars)
        A, mu, sigma = pars[:3]
        m = 0; y0 = 0
        if len(pars)==4:
            y0 = pars[-1]
        elif len(pars)==5:
            m, y0 = pars[3:]
        
        ygrid = A*np.exp(-0.5*(xgrid-mu)**2/sigma**2) + m*(xgrid-mu) + y0
        
        label = r'Gaussian IP'
    ax1.plot(xgrid,ygrid,c='grey',lw=2,ls='--',label=label)    
    # if len(pars)==5:
    #     ax1.plot(xgrid,(xgrid-pars[1])*pars[3] + pars[4],lw=2,ls='--',label='linear')
    ax1.legend(loc='upper left')
    weights = assign_weights(x1l,pars[1],scale)[within]
    # rsd  = (flx1l-model)/err1l
    if len(np.shape(rsd_norm))>1:
        for array,cut in zip(rsd_norm,[within,~within]):
            ax2.scatter(x1l[cut],array[cut],label='rsd',
                        edgecolor='k',color='w')
            ax2.scatter(x1l[cut],array[cut],label='rsd',marker='o',
                        alpha=weights)
    else:
        ax2.scatter(x1l[within],rsd_norm[within],label='rsd',
                    edgecolor='k',color='w')
        ax2.scatter(x1l[within],rsd_norm[within],label='rsd',marker='o',
                    alpha=weights)
    
    
    ax2.axhspan(-1,1,color='grey',alpha=0.3)
    # ylim = np.min([1.5*np.nanpercentile(np.abs(rsd_norm),95),10.3])
    # ylim = 1.8*np.max(np.abs(rsd_norm))
    if len(np.shape(rsd_norm))>1:
        default_ylim = np.max(np.abs(rsd_norm[0][within]))
    else:
        default_ylim = np.max(np.abs(rsd_norm[within]))
    ylim = rsd_range if rsd_range is not None else 1.8*default_ylim
    # rsd_range = kwargs.pop('rsd_range',False)
    # print(ylim,rsd_range)
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
    
    if len(np.shape(rsd_norm))>1:
        rsd_use = rsd_norm[0]
    else:
        rsd_use = rsd_norm
    for x,r,w in zip(x1l[within],rsd_use[within],weights):
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
def _prepare_pars(npars,method,x,y):
    assert npars>2
    guess_amp = 1.1*np.max(y)
    guess_cen = np.average(x,weights=y)
    guess_wid = 1.0
    if method=='scipy':
        p0 = (guess_amp,guess_cen,guess_wid)
        if npars==4:
            p0 = (p0)+(0.,)
        elif npars==5:
            p0 = (p0)+(0.,0.)
    elif method=='lmfit':
        from lmfit import Parameters
        parameters = [('amp',    guess_amp, True, None,None,None,None),
                      ('cen',    guess_cen, True, None,None,None,None),
                      ('wid',    guess_wid, True, None,None,None,None),
                      ('slope',        0.0, True, None,None,None,None),
                      ('offset',       0.0, True, None,None,None,None),
                      ]
        p0 = Parameters()
        # parameter tuples (name, value, vary, min, max, expr, brute_step).
        p0.add_many(*parameters[:npars])
        
    return p0
    
def _unpack_pars(pars):
    try:
        vals = pars.valuesdict()
    except:
        vals = pars
    npars = len(pars)
    if isinstance(vals,dict):
        m = 0.
        y0 = 0.
        if len(vals)==3:
            amp = vals['amp']
            cen = vals['cen']
            wid = vals['wid']
        elif len(vals)==4:
            amp = vals['amp']
            cen = vals['cen']
            wid = vals['wid']
            y0  = vals['offset']
        elif len(vals)==5:
            amp = vals['amp']
            cen = vals['cen']
            wid = vals['wid']
            y0  = vals['offset']
            m   = vals['slope']
        elif len(vals)==2:
            amp = vals['amp']
            cen = vals['cen']
            wid = 1.
    else:
        amp = vals[0]
        cen = vals[1]
        wid = vals[2]
        y0  = 0.
        m   = 0.
        if len(vals)==4:
            y0 = vals[3]
        if len(vals)==5:
            m  = vals[3]
            y0 = vals[4]
    return (amp,cen,wid,m,y0)[:npars]
def lsf_model(lsf_loc_x,lsf_loc_y,pars,xarray,scale):
    """
    Returns the model of the data from the LSF and parameters provided. 
    Does not include the background.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    amp,cen,wid = _unpack_pars(pars)[:3]
    
    wid   = np.abs(wid)
    x     = lsf_loc_x * wid
    y     = lsf_loc_y / np.max(lsf_loc_y) 
    splr  = interpolate.splrep(x,y) 
    
    if scale[:3]=='pix':
        x_test = xarray-cen
    elif scale[:3]=='vel':
        x_test = (xarray-cen)/cen*299792.458
    # model = amp * (m*x_test + y0 + interpolate.splev(x_test,splr))
    model = amp * interpolate.splev(x_test,splr) 
    return model

def sct_model(sct_loc_x,sct_loc_y,pars,xarray,scale):
    """
    Returns the model of the data from the LSF and parameters provided. 
    Does not include the background.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    amp,cen,wid,m,y0 = _unpack_pars(pars)
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