#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:02:28 2018

@author: dmilakov
"""
from harps.core import np, interpolate
import harps.functions as hf

def getbkg(yarray,xarray=None,window=3,kind='linear'):
    """
    Returns the interpolated background between minima in yarray.
    
    Smooths the spectrum using Wiener filtering to detect true minima.
    See peakdetect.py for more information
    """
    xarray = xarray if xarray is not None else np.arange(len(yarray))
    xbkg,ybkg = hf.peakdet(yarray, xarray, extreme="min",window=window)
    if   kind == "spline":
        intfunc = interpolate.splrep(xbkg, ybkg)
        bkg     = interpolate.splev(xarray,intfunc) 
    elif kind == "linear":
        intfunc = interpolate.interp1d(xbkg,ybkg,
                                       bounds_error=False,
                                       fill_value=0)
        bkg = intfunc(xarray)
    return bkg

def get1d(spec, order, kind="linear",*args):
    """
    Returns the background in the echelle order. 
    Default linear interpolation between line minima.
    """
    yarray = spec.data[order]
    xarray = np.arange(spec.npix)
    window = spec.lfckeys['window_size']
    bkg    = getbkg(yarray,xarray,window)
    return bkg
    
def get2d(spec, order=None, kind="linear", *args):
    """
    Returns the background for all echelle orders in the spectrum.
    Default linear interpolation between line minima.
    """
    orders     = np.arange(spec.nbo)
    background = np.array([get1d(spec,o,kind) for o in orders])
    return background

def getenv(xarray,yarray,window,kind='linear'):
    """
    Returns the interpolated background between minima in yarray.
    
    Smooths the spectrum using Wiener filtering to detect true minima.
    See peakdetect.py for more information
    """
    xbkg,ybkg = hf.peakdet(yarray, xarray, extreme='max',window=window,
                           method='peakdetect')
    if   kind == "spline":
        intfunc = interpolate.splrep(xbkg, ybkg)
        data    = interpolate.splev(xarray,intfunc) 
    elif kind == "linear":
        intfunc = interpolate.interp1d(xbkg,ybkg,
                                       bounds_error=False,
                                       fill_value=0)
        data = intfunc(xarray)
    return data

#def getenv(xarray,yarray,window,kind='linear'):
#    """
#    Returns the interpolated background between minima in yarray.
#    
#    Smooths the spectrum using Wiener filtering to detect true minima.
#    See peakdetect.py for more information
#    """
#    env = get(xarray,yarray,window,use='max',kind=kind)
#    return env

def getenv1d(spec, order, kind="linear",*args):
    """
    Returns the background in the echelle order. 
    Default linear interpolation between line minima.
    """
    yarray = spec.data[order]
    xarray = np.arange(spec.npix)
    window = spec.lfckeys['window_size']
    env    = getenv(xarray,yarray,window)
    return env
    
def getenv2d(spec, order=None, kind="linear", *args):
    """
    Returns the background for all echelle orders in the spectrum.
    Default linear interpolation between line minima.
    """
    orders   = np.arange(spec.nbo)
    envelope = np.array([getenv1d(spec,o,kind) for o in orders])
    return envelope