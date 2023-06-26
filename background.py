#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:02:28 2018

@author: dmilakov
"""
from harps.core import np, interpolate
import harps.functions as hf
import harps.peakdetect as pkd
import matplotlib.pyplot as plt

kind = 'spline'

def get_env_bkg(yarray,xarray=None,kind=kind,*args,**kwargs):
    xarray = xarray if xarray is not None else np.arange(len(yarray))
    maxmin = hf.detect_maxmin(yarray, xarray, *args,**kwargs)
    if maxmin is not None:
        maxima,minima = maxmin
    else:
        return np.zeros_like(yarray),np.ones_like(yarray)
    arrays = []
    for (x,y) in [maxima,minima]:
        if   kind == "spline":
            intfunc = interpolate.splrep(x, y)
            arr     = interpolate.splev(xarray,intfunc) 
        elif kind == "linear":
            intfunc = interpolate.interp1d(x,y,
                                           bounds_error=False,
                                           fill_value=0)
            arr = intfunc(xarray)
        arrays.append(arr)
    env, bkg = arrays
    return env,bkg

def get_env_bkg1d(spec, order=None, kind=kind, *args, **kwargs):
    yarray   = spec.data[order]
    xarray   = np.arange(spec.npix)
    env, bkg = get_env_bkg(yarray,xarray,kind,*args,**kwargs)
    return env, bkg
def get_env_bkg2d(spec, order=None,kind=kind, *args, **kwargs):
    if order is not None:
        orders = np.asarray(order)
    else:
        orders  = np.arange(spec.nbo)
    env = np.zeros_like(spec.data)
    bkg = np.zeros_like(spec.data)
    for i, od in enumerate(orders):
        if od<spec.sOrder or od>spec.eOrder:
            continue
        else:
            print(od)
            env1d,bkg1d = get_env_bkg1d(spec,od,kind)
            env[i] = env1d
            bkg[i] = bkg1d
        
    return env, bkg
def getbkg(yarray,xarray=None,kind=kind,*args,**kwargs):
    """
    Returns the interpolated background between minima in yarray.
    
    Smooths the spectrum using Wiener filtering to detect true minima.
    See peakdetect.py for more information
    """
    xarray = xarray if xarray is not None else np.arange(len(yarray))
    min_idx,_ = hf.detect_minima(yarray, xarray, *args,**kwargs)
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
    xbkg,ybkg = hf.detect_maxima(yarray, xarray, *args,**kwargs)
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

def fit_spline(yarray,xarray=None,yerror=None,extreme='min',
                node_dist=30,f=0.05,*args,**kwargs):
    xarray = xarray if xarray is not None else np.arange(len(yarray))
    yerror = yerror if yerror is not None else np.ones_like(yarray)
    # detect minima 
    # min_idx,_ = hf.detect_minima(yarray, xarray, *args,**kwargs)
    min_idx,_ = hf.peakdet(yarray,extreme=extreme,remove_false=False,*args)
    min_idx = min_idx.astype(int)
    # add 2 pixels either side of the minima
    idx = []
    for i in min_idx:
        idx.append(i)
        for j in range(-2,3,1):
            testval = np.abs(yarray[i+j])
            if testval>=(1-f)*yarray[i] and testval<=(1+f)*yarray[i]:
                idx.append(i+j)
        # idx.append(np.arange(i-1,i+2,1))
    idx = np.sort(np.hstack(idx).astype(int))
    nodes = np.arange(node_dist,len(yarray)-node_dist,node_dist)
    print(len(idx))
    
    plt.figure()
    plt.plot(xarray,yarray,drawstyle='steps-mid')
    plt.errorbar(xarray[idx],yarray[idx],yerror[idx],marker='x',c='r',ls='',alpha=0.3)
    # plt.plot(xarray[idx],1./yerror[idx])
    
    tck,fp,ier,msg = interpolate.splrep(idx, yarray[idx],
                                            w = 1./yerror[idx]+1.,
                                           k = 3,
                                           # s = 1000,
                                            t = nodes,
                                            task=-1,
                                           full_output=True
                                           )
    print(fp,ier,msg)
    plt.scatter(tck[0],interpolate.splev(tck[0],tck),marker='o',c='k')
    bkg = interpolate.splev(xarray,tck)
    
    
    plt.plot(xarray,bkg,c='r')
    return bkg