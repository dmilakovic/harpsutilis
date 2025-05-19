#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:02:28 2018

@author: dmilakov
"""
from harps.core import np, interpolate
import harps.functions.spectral as specfunc
import harps.peakdetect as pkd
import matplotlib.pyplot as plt
import harps.progress_bar as progress_bar

# kind = 'spline'
kind = 'linear_smooth'
# kind = 'fit_spline'

def get_linepos_env_bkg(yarray2d,sOrder,plot=False,verbose=False):
    line_positions, env2d, bkg2d = pkd.process_spectrum2d(yarray2d,
                                                          sOrder=sOrder,
                                                      plot_main_details=plot,
                                                      verbose=verbose)
    return line_positions, env2d, bkg2d
    
def get_env_bkg_old(yarray,extrema1d,xarray=None,yerror=None,kind=kind,*args,**kwargs):
    
    xarray = xarray if xarray is not None else np.arange(len(yarray))
    f = kwargs.pop('f',0.01)
    node_dist = kwargs.pop('node_dist',20)
    plot = kwargs.pop('plot',False)
    # maxmin = specfunc.peakdet(yarray, xarray, plot=plot, *args,**kwargs)
    if extrema1d is not None:
        maxima1d,minima1d = extrema1d
    else:
        return np.zeros_like(yarray),np.ones_like(yarray)
    arrays = []
    # plt.figure()
    # plt.plot(xarray,yarray)
   
    for (x,y) in [maxima1d.T,minima1d.T]:
        xint = np.round(x).astype(int)
        # print(xint,y)
        # plt.scatter(x,y,marker='v')
        if   kind == "spline":
            # intfunc = interpolate.splrep(x, yarray[xint])
            intfunc = interpolate.splrep(x, y)
            arr     = interpolate.splev(xarray,intfunc) 
        elif "linear" in kind:
            # intfunc = interpolate.interp1d(x,y,
            #                                bounds_error=False,
            #                                fill_value=0)
            # arr_ = intfunc(xarray)
            # arr_ = np.interp(xarray,x,yarray[x])
            arr_ = np.interp(xarray,x,y)
            if "smooth" in kind:
                window_len = 51
                arr_filtered = pkd._smooth(arr_,window_len,
                                            window='nuttall',mode='valid')
                # arr  = arr_filtered[window_len-1:-(window_len-1)]
                arr = arr_filtered
            else:
                arr = arr_
            
        elif kind == 'fit_spline':
            arr = fit_spline(yarray, xarray, yerror=yerror, 
                             xxtrm=x,yxtrm=y,f=f,
                             node_dist=node_dist,plot=plot,
                             *args, **kwargs)
        arrays.append(arr)
    env, bkg = arrays
    return env,bkg

# def continuum(yarray,yerror,bins=10,frac=0.3,zeta=3,Kf=6,window_len=None):
    

def get_env_bkg1d_from_array(yarray, extrema, kind=kind, *args, **kwargs):
    xarray   = np.arange(len(yarray))
    yerror   = np.sqrt(yarray)
    env, bkg = get_env_bkg(yarray,extrema, xarray,yerror,kind,*args,**kwargs)
    return env, bkg

def get_env_bkg2d_from_array(flux2d,extrema,sOrder=None,eOrder=None,
                             kind=kind,*args,**kwargs):
    nbo,npix = np.shape(flux2d)
    sOrder   = sOrder if sOrder is not None else 39
    eOrder   = eOrder if eOrder is not None else nbo
    orders  = np.arange(sOrder,eOrder)
    env = np.zeros_like(flux2d)
    bkg = np.zeros_like(flux2d)
    for i, od in enumerate(orders):
        maxima1d, minima1d = extrema[0][od], extrema[1][od]
        env1d,bkg1d = get_env_bkg(flux2d[od],(maxima1d,minima1d),
                                  None,np.sqrt(flux2d[od]),
                                  kind,*args,**kwargs)
        env[od] = env1d
        bkg[od] = bkg1d
        progress_bar.update(i/(len(orders)-1),f'Background {od}/{eOrder}')
    return env, bkg

def get_env_bkg1d(spec, order=None, kind=kind, *args, **kwargs):
    yarray   = spec.data[order]
    xarray   = np.arange(spec.npix)
    yerror   = np.sqrt(yarray)
    extrema  = spec.extrema
    env, bkg = get_env_bkg(yarray,extrema,xarray,yerror,kind,*args,**kwargs)
    return env, bkg

def get_env_bkg2d(spec, order=None,kind=kind, *args, **kwargs):
    flux2d = spec['flux']
    extrema = spec['extrema']
    sOrder = spec.sOrder
    eOrder = spec.eOrder
    env, bkg = get_env_bkg2d_from_array(flux2d,extrema,
                                        sOrder=sOrder,eOrder=eOrder,
                                 kind=kind,*args,**kwargs)
    
    # if order is not None:
    #     orders = np.asarray(order)
    # else:
    #     orders  = np.arange(spec.nbo)
    # env = np.zeros_like(spec.data)
    # bkg = np.zeros_like(spec.data)
    # for i, od in enumerate(orders):
    #     if od<spec.sOrder or od>spec.eOrder:
    #         continue
    #     else:
    #         env1d,bkg1d = get_env_bkg1d(spec,od,kind,*args,**kwargs)
    #         env[i] = env1d
    #         bkg[i] = bkg1d
    #     progress_bar.update(i/(len(orders)-1),'Background')
    return env, bkg
def getbkg(yarray,xarray=None,kind=kind,*args,**kwargs):
    """
    Returns the interpolated background between minima in yarray.
    
    Smooths the spectrum using Wiener filtering to detect true minima.
    See peakdetect.py for more information
    """
    xarray = xarray if xarray is not None else np.arange(len(yarray))
    min_idx,_ = specfunc.detect_minima(yarray, xarray, *args,**kwargs)
    min_idx = min_idx.astype(int)
    if   kind == "spline":
        intfunc = interpolate.splrep(min_idx, yarray[min_idx])
        bkg     = interpolate.splev(xarray,intfunc) 
    elif kind == "linear":
        intfunc = interpolate.interp1d(min_idx,yarray[min_idx],
                                       bounds_error=False,
                                       fill_value=0)
        bkg = intfunc(xarray)
    elif kind == 'fit_spline':
        bkg = fit_spline(yarray,xarray,yerror=np.sqrt(yarray),
                         xxtm=min_idx,yxtrm=_,
                         node_dist=100,f=0.1,*args,**kwargs)
        
    return bkg

def get1d(spec, order, kind=kind,*args,**kwargs):
    """
    Returns the background in the echelle order. 
    Default linear interpolation between line minima.
    """
    yarray = spec.data[order]
    xarray = np.arange(spec.npix)
    # window = spec.lfckeys['window_size']
    bkg    = getbkg(yarray,xarray,*args,**kwargs)
    return bkg
    
def get2d(spec, order=None, kind=kind, *args):
    """
    Returns the background for all echelle orders in the spectrum.
    Default linear interpolation between line minima.
    """
    orders     = np.arange(spec.nbo)
    background = np.array([get1d(spec,o,kind) for o in orders])
    return background

def getenv(xarray,yarray,kind='linear',*args,**kwargs):
    """
    Returns the interpolated background between minima in yarray.
    
    Smooths the spectrum using Wiener filtering to detect true minima.
    See peakdetect.py for more information
    """
    xarray = xarray if xarray is not None else np.arange(len(yarray))
    xbkg,ybkg = specfunc.detect_maxima(yarray, xarray, *args,**kwargs)
    if   kind == "spline":
        intfunc = interpolate.splrep(xbkg, ybkg)
        data    = interpolate.splev(xarray,intfunc) 
    elif kind == "linear":
        intfunc = interpolate.interp1d(xbkg,ybkg,
                                       bounds_error=False,
                                       fill_value=0)
        data = intfunc(xarray)
    return data

def getenv1d(spec, order, kind=kind,*args):
    """
    Returns the background in the echelle order. 
    Default linear interpolation between line minima.
    """
    yarray = spec.data[order]
    xarray = np.arange(spec.npix)
    window = spec.lfckeys['window_size']
    env    = getenv(xarray,yarray,window)
    return env
    
def getenv2d(spec, order=None, kind=kind, *args):
    """
    Returns the background for all echelle orders in the spectrum.
    Default linear interpolation between line minima.
    """
    orders   = np.arange(spec.nbo)
    envelope = np.array([getenv1d(spec,o,kind) for o in orders])
    return envelope

def fit_spline(yarray,xarray=None,yerror=None,xxtrm=None,yxtrm=None,
                node_dist=20,f=0.02,plot=False,*args,**kwargs):
    # 2023-07-11
    # with node_dist = 60 the power spectrum of background/envelope was 
    # consistent with pink noise but some remaining structure in flux
    # with 35 
    xarray = xarray if xarray is not None else np.arange(len(yarray))
    yerror = yerror if yerror is not None else np.ones_like(yarray)
    # detect minima 
    # min_idx,_ = hf.detect_minima(yarray, xarray, *args,**kwargs)
    # minima,maxima = hf.peakdet(yarray,*args)
    min_idx = xxtrm.astype(int)
    cut = np.where(min_idx<len(yarray))[0]
    min_idx = min_idx[cut]
    
    # add 2 pixels either side of the minima
    idx = []
    # idx.append(0)
    # idx.append(-1)
    for i in min_idx:
        if i>0 and i<len(yarray):
            pass
        else:
            continue
            
        idx.append(i)
        for j in range(-2,3,1):
            if ((i+j)>0 and (i+j)<len(yarray)-1):
                pass
            else:
                continue
            testval = np.abs(yarray[i+j])
            # only consider the point if its flux is +/- (1+f) away from 
            # the flux of the minimum
            if testval>=(1-f)*yarray[i] and testval<=(1+f)*yarray[i]:
                
                idx.append(i+j)
        # idx.append(np.arange(i-1,i+2,1))
    idx = np.sort(np.hstack(idx).astype(np.int16))
    # nodes = np.arange(node_dist-node_dist/2,len(yarray),node_dist)
    nodes = np.arange(19,len(yarray),node_dist)
    tck,fp,ier,msg = interpolate.splrep(idx, yarray[idx],
                                        w = 1./yerror[idx],
                                        k = 3,
                                        t = nodes,
                                        task=-1,
                                        full_output=True
                                        )
    bkg = interpolate.splev(xarray,tck)
    #print(fp,ier,msg)
    if plot:
        # print(len(idx))
        plt.figure()
        plt.plot(xarray,yarray,drawstyle='steps-mid',lw=0.3)
        plt.errorbar(xarray[idx],yarray[idx],yerror[idx],marker='x',c='r',ls='',alpha=0.3)
        plt.scatter(xarray[min_idx],yarray[min_idx],#yerror[min_idx],
                     marker='v',c='k',alpha=0.3)
        
        
        plt.scatter(tck[0],interpolate.splev(tck[0],tck),marker='o',c='k')
        
        
        
        plt.plot(xarray,bkg,c='r',lw=2.)
    return bkg