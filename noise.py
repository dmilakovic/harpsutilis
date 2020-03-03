#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:52:42 2020

@author: dmilakov
"""
from harps.core import np
from harps.background import getbkg
from harps.constants import c
import harps.functions as hf

def error1d(data,background=None,*args,**kwargs):
    """
    Returns a 1d array with errors on flux values for this order. 
    Adds the error due to background subtraction in quadrature to the 
    photon counting error.
    """
    data1d  = np.abs(data)
    npts    = np.size(data1d)
    bkg1d   = background if background is not None \
              else np.abs(getbkg(data1d,*args,**kwargs))
    error1d = np.sqrt(data1d + bkg1d)
    return error1d
def sigmav1d(data,error=None,wave=None,background=None,*args,**kwargs):
    """
    Calculates the limiting velocity precison of all pixels in the order
    using ThAr wavelengths.
    """
    wave    = wave if wave is not None else np.arange(len(data))
    error   = error if error is not None else error1d(data,background,*args,**kwargs)
    # weights for photon noise calculation
    # Equation 5 in Murphy et al. 2007, MNRAS 380 p839
    #pix2d   = np.vstack([np.arange(spec.npix) for o in range(spec.nbo)])
    df_dlbd = hf.derivative1d(data,wave)
    sigma_v = c*error/(wave*df_dlbd)
    return sigma_v
def weights1d(data,error=None,wave=None,*args,**kwargs):
    """
    Calculates the photon noise of this order.
    """
    sigma_v = sigmav1d(data,error,wave,*args,**kwargs)
    return (sigma_v/c)**-2

def photon(data,error=None,wave=None,unit='mps',*args,**kwargs):
    """
    Calculates the theoretical limiting velocity precision from the photon 
    noise weights of this order(s).
    """
    weights = weights1d(data,error,wave,*args,**kwargs)
    precision1d = 1./np.sqrt(np.sum(weights))
    precision_total = 1./np.sqrt(np.sum(np.power(precision1d,-2)))
    if unit == 'mps':
        fac = 2.99792458e8
    else:
        fac = 1.
    return precision_total * fac
#def sigmav1d(self,order=None,unit='mps'):
#    """
#    Calculates the theoretical limiting velocity precision from the photon 
#    noise weights of this order(s).
#    """
#    orders = self.prepare_orders(order)
#    precision_order = [1./np.sqrt(np.sum(self.weights[order])) \
#                       for order in orders]
#    precision_total = 1./np.sqrt(np.sum(np.power(precision_order,-2)))
#    if unit == 'mps':
#        fac = 2.99792458e8
#    else:
#        fac = 1.
#    return precision_total * fac