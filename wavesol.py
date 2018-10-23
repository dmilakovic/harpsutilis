#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:40:01 2018

@author: dmilakov
"""

from harps.core import np

import harps.io as io
import harps.functions as hf

def construct_singleorder(coeffs,npix):
    return hf.polynomial(np.arange(npix),*coeffs)
def construct(coeffs,npix):
    #nbo,deg = np.shape(a)
    wavesol = np.array([construct_singleorder(c,npix) for c in coeffs])
    return wavesol
def _to_vacuum(lambda_air,pressure=760,ccdtemp=15):
    """
    Returns vacuum wavelengths.
    
    Args:    
        lambda_air: 1D numpy array
    Returns:
        lambda_vacuum : 1D numpy array
    """
    assert lambda_air.sum()!=0, "Wavelength array is empty."
    index    = 1e-6*pressure*(1.0+(1.049-0.0157*ccdtemp)*1e-6*pressure) \
                /720.883/(1.0+0.003661*ccdtemp) \
                *(64.328+29498.1/(146.0-2**(1e4/lambda_air)) \
                +255.4/(41.0-2**(1e4/lambda_air)))+1.0
    lambda_vacuum = lambda_air*index
    return lambda_vacuum
def _get_wavecoeff_air(filepath):
    ''' 
    Returns coefficients of a third-order polynomial from the FITS file 
    header in a matrix. This procedure is described in the HARPS DRS  
    user manual.
    https://www.eso.org/sci/facilities/lasilla/
            instruments/harps/doc/DRS.pdf
    '''
    def _read_wavecoef1d(order):
        """ 
        Returns ThAr wavelength calibration coefficients saved in the header.
        Returns zeroes when no coefficients are found.
        """
        coeffs = np.zeros(deg+1)
        for i in range(deg+1):                    
            ll    = i + order*(deg+1)
            try:
                coeffs[i] = header["ESO DRS CAL TH COEFF LL{0}".format(ll)]
            except:
                continue
        return coeffs
    
    header = io.read_e2ds_header(filepath)
    meta   = io.read_e2ds_meta(filepath)
    nbo    = meta['nbo']
    deg    = meta['d']
    coeffs = np.array(list(map(_read_wavecoef1d,range(nbo))))
    bad_orders = np.array(list(filter(lambda x: sum(x)==0, coeffs)))
    
    return coeffs, bad_orders

#==============================================================================
    
#               W A V E L E N G T H    S O L U T I O N S  
    
#==============================================================================
def thar(spec,vacuum=True):
    """ 
    Return the ThAr wavelength solution, as saved in the header of the
    e2ds file. 
    """
    coeffs, bad_orders = _get_wavecoeff_air(spec.filepath)
    wavesol_air = construct(coeffs,spec.npix)
    if vacuum==True:
        return _to_vacuum(wavesol_air)
    else:
        return wavesol_air
def comb(spec,vacuum=True):
    linelist = io.read_linelist(spec.filepath)
    return linelist
