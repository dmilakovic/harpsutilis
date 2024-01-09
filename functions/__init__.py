#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:43:28 2018

@author: dmilakov
"""
# import numpy as np
# import pandas as pd
# import math
#import xarray as xr
# import sys
# import os
# import re

# from harps import peakdetect as pkd
# from harps import emissionline as emline
from harps import settings as hs
# from harps.constants import c
# import harps.containers as container
# import harps.functions.profile as profile

# from harps.core import welch, logging
import harps.progress_bar as progress_bar

# from scipy.special import erf, wofz, gamma, gammaincc, expn
# from scipy.optimize import minimize, leastsq, curve_fit, brentq
# from scipy.interpolate import splev, splrep
# from scipy.integrate import quad

# from glob import glob

# import numpy as np
# import jax.numpy as jnp
# import jax.scipy as jsp

# from matplotlib import pyplot as plt
#from kapteyn import kmpfit

__version__   = hs.__version__

__all__ = ['aux','lfc','math','outliers','profile','spectral']
# hs.setup_logging()

from harps.functions.aux import *
from harps.functions.lfc import *
from harps.functions.math import *
from harps.functions.outliers import *
from harps.functions.profile import *
from harps.functions.spectral import *


#------------------------------------------------------------------------------
#
#                           P R O G R E S S   B A R 
#
#------------------------------------------------------------------------------
def update_progress(*args,**kwargs):
    progress_bar.update(*args,**kwargs)
#------------------------------------------------------------------------------
#
#                        P H O T O N     N O I S E
#
#------------------------------------------------------------------------------   
# def get_background1d(data,kind="linear",*args):
#     """
#     Returns the background in the echelle order. 
#     Default linear interpolation between line minima.
#     """
#     yarray = np.atleast_1d(data)
#     assert len(np.shape(yarray))==1, "Data is not 1-dimensional"
#     xarray = np.arange(np.size(data))
#     bkg    = get_background(xarray,yarray)
#     return bkg
    
# def get_background2d(data,kind="linear", *args):
#     """
#     Returns the background for all echelle orders in the spectrum.
#     Default linear interpolation between line minima.
#     """
#     orders     = np.shape(data)[0]
#     background = np.array([get_background1d(data[o],kind) \
#                            for o in range(orders)])
#     return background

# def get_background(xarray,yarray,kind='linear'):
#     """
#     Returns the interpolated background between minima in yarray.
    
#     Smooths the spectrum using Wiener filtering to detect true minima.
#     See peakdetect.py for more information
#     """
#     from scipy import interpolate
#     xbkg,ybkg = peakdet(yarray, xarray, extreme="min")
#     if   kind == "spline":
#         intfunc = interpolate.splrep(xbkg, ybkg)
#         bkg     = interpolate.splev(xarray,intfunc) 
#     elif kind == "linear":
#         intfunc = interpolate.interp1d(xbkg,ybkg,
#                                        bounds_error=False,
#                                        fill_value=0)
#         bkg = intfunc(xarray)
#     return bkg

# def get_error2d(data2d):
#     assert len(np.shape(data2d))==2, "Data is not 2-dimensional"
#     data2d  = np.abs(data2d)
#     bkg2d   = get_background2d(data2d)
#     error2d = np.sqrt(np.abs(data2d) + np.abs(bkg2d))
#     return error2d
    
# def get_error1d(data1d,*args):
#     data1d  = np.abs(data1d)
#     bkg1d   = np.abs(get_background1d(data1d,*args))
#     error1d = np.sqrt(data1d + bkg1d)
#     return error1d
# def sigmav(data2d,wavesol2d):
#     """
#     Calculates the limiting velocity precison of all pixels in the spectrum
#     using ThAr wavelengths.
#     """
#     orders  = np.arange(np.shape(data2d)[0])
#     sigma_v = np.array([sigmav1d(data2d[order],wavesol2d[order]) \
#                         for order in orders])
#     return sigma_v

# def sigmav1d(data1d,wavesol1d):
#     """
#     Calculates the limiting velocity precison of all pixels in the order
#     using ThAr wavelengths.
#     """
#     # weights for photon noise calculation
#     # Equation 5 in Murphy et al. 2007, MNRAS 380 p839
#     #pix2d   = np.vstack([np.arange(spec.npix) for o in range(spec.nbo)])
#     error1d = get_error1d(data1d)
#     df_dlbd = derivative1d(data1d,wavesol1d)
#     sigma_v = c*error1d/(wavesol1d*df_dlbd)
#     return sigma_v


