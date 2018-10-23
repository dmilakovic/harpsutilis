#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:02:28 2018

@author: dmilakov
"""
from harps.core import np, interpolate
import harps.functions as hf

def getbkg(xarray,yarray,window,kind='linear'):
    """
    Returns the interpolated background between minima in yarray.
    
    Smooths the spectrum using Wiener filtering to detect true minima.
    See peakdetect.py for more information
    """
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
    bkg    = getbkg(xarray,yarray,window)
    return bkg
    
def get2d(spec, order=None, kind="linear", *args):
    """
    Returns the background for all echelle orders in the spectrum.
    Default linear interpolation between line minima.
    """
    orders     = np.arange(spec.nbo)
    background = np.array([get1d(spec,o,kind) for o in orders])
    return background