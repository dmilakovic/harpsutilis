#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:40:01 2018

@author: dmilakov
"""

from harps.core import np
from harps.constants import c
from harps.spectrum import extract_version

import harps.io as io
import harps.functions as hf
import harps.fit as fit
import harps.containers as container

#==============================================================================
#    
#                   H E L P E R     F U N C T I O N S  
#    
#==============================================================================
def evaluate(coeffs,x=None,startpix=None,endpix=None):
    x = x if x is not None else np.arange(startpix,endpix,1)
#    print("COEFFS = ",coeffs)
#    return np.polyval(coeffs[::-1],x)
    return hf.polynomial(x,*coeffs)
def construct(coeffs,npix):
    """ For ThAr only"""
    #nbo,deg = np.shape(a)
    wavesol = np.array([evaluate(c,startpix=0,endpix=npix) for c in coeffs])
    return wavesol

allowed_calibrators = ['thar','comb']

def get(spec,calibrator,*args,**kwargs):
    assert calibrator in allowed_calibrators
    if calibrator == 'thar':
        return thar(spec,*args,**kwargs)
    elif calibrator == 'comb':
        return comb(spec,*args,**kwargs)
    
#==============================================================================
#    
#               T H O R I U M    A R G O N     F U N C T I O N S  
#    
#==============================================================================
def _refrindex(pressure,ccdtemp):
    index    = 1e-6*pressure*(1.0+(1.049-0.0157*ccdtemp)*1e-6*pressure) \
                /720.883/(1.0+0.003661*ccdtemp) \
                *(64.328+29498.1/(146.0-2**(1e4/lambda_air)) \
                +255.4/(41.0-2**(1e4/lambda_air)))+1.0
    return index
def _to_vacuum(lambda_air,pressure=760,ccdtemp=15):
    """
    Returns wavelengths in vacuum.
    
    Args:    
        lambda_air: 1D numpy array
    Returns:
        lambda_vacuum : 1D numpy array
    """
    assert lambda_air.sum()!=0, "Wavelength array is empty."
    index         = _refrindex(pressure,ccdtemp)
    lambda_vacuum = lambda_air*index
    return lambda_vacuum
def _to_air(lambda_vacuum,pressure=760,ccdtemp=15):
    """
    Returns wavelengths in air.
    
    Args:    
        lambda_air: 1D numpy array
    Returns:
        lambda_vacuum : 1D numpy array
    """
    assert lambda_vacuum.sum()!=0, "Wavelength array is empty."
    index      = _refrindex(pressure,ccdtemp)
    lambda_air = lambda_vacuum/index
    return lambda_air

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
    bad_orders = np.where(np.sum(coeffs,axis=1)==0)[0]
    
    return coeffs, bad_orders
def _get_wavecoeff_vacuum(filepath):
    
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



#==============================================================================
#    
#               L A S E R    F R E Q U E N C Y    C O M B
#
#                            F U N C T I O N S  
#    
#==============================================================================

def comb(spec,version,*args,**kwargs):
    coefficients = _get_wavecoeff_comb(spec,version,*args,**kwargs)
    wavesol_comb = construct_from_combcoeff(coefficients,spec.npix)

    return wavesol_comb

# stopped here, 29 Oct 2018
def residuals(spec,version,fittype='gauss',*args,**kwargs):
    linelist     = spec['linelist']
    coefficients = spec['coeff',version]
    
    centers      = linelist[fittype][:,1]
    wavelengths  = hf.freq_to_lambda(linelist['freq'])
    nlines       = len(linelist)
    residuals    = container.residuals(nlines)
    for coeff in coefficients:
        order = coeff['order']
        segm  = coeff['segm']
        pixl  = coeff['pixl']
        pixr  = coeff['pixr']
        cut   = np.where((linelist['order']==order) & 
                         (centers >= pixl) &
                         (centers <= pixr))
        centsegm = centers[cut]
        wavereal  = wavelengths[cut]
        wavefit   = evaluate(coeff['pars'],centsegm)
        residuals['order'][cut]=order
        residuals['segm'][cut]=segm
        residuals['residual'][cut]=(wavereal-wavefit)/wavereal*c
    return residuals
        
def _get_wavecoeff_comb(spec,version,*args,**kwargs):
    """
    Returns a dictionary with the wavelength solution coefficients derived from
    LFC lines
    """
    linelist = spec['linelist']
#    polyord  = extract_version(version)[0]
    wavesol2d = fit.wavesol(linelist,version,*args,**kwargs)
    return wavesol2d

def construct_order(coeffs,npix):
    wavesol1d  = np.zeros(npix)
    for segment in coeffs:
        pixl = segment['pixl']
        pixr = segment['pixr']
        pars = segment['pars']
        wavesol1d[pixl:pixr] = evaluate(pars,None,pixl,pixr)
    return wavesol1d

#def construct_from_combcoeff1d(coeffs,npix,order):
#    cfs = coeffs[hf.get_extname(order)]
#    wavesol1d = construct_order(cfs,npix) 
#    return wavesol1d

def construct_from_combcoeff(coeffs,npix):
    orders    = np.unique(coeffs['order'])
    nbo       = np.max(orders)+1
    
    wavesol2d = np.zeros((nbo,npix))
    for order in orders:
        coeffs1d = coeffs[np.where(coeffs['order']==order)]
        wavesol2d[order] = construct_order(coeffs1d,npix)
        
    return wavesol2d

#def evaluate(coeffs,centers):
#    np.polyval(coeffs[::-1],centers)