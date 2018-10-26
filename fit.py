#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:07:32 2018

@author: dmilakov
"""

from harps.core import leastsq, odr
from harps.core import np
from harps.core import os
from harps.constants import c

import harps.settings as hs
import harps.emissionline as emline
import harps.containers as container

#==============================================================================
# Assumption: Frequencies are known with 1MHz accuracy
freq_err = 1e6


#==============================================================================
    
#                           G A P S    F I L E   
    
#==============================================================================
def read_gaps(filepath=None):
    if filepath is not None:
        filepath = filepath  
    else:
        filepath = os.path.join(hs.harps_prod,'gapsA.npy')
    gapsfile = np.load(filepath)    
    #orders   = np.array(gapsfile[:,0],dtype='i4')
    #gaps2d   = np.array(gapsfile[:,1:],dtype='f8')
    return gapsfile
    
    #{"ORDER{od:2d}".format(od=od):gaps1d for od,gaps1d in zip(orders,gaps2d)}


def get_gaps(order,filepath=None):
    gapsfile  = read_gaps(filepath)
    selection = np.where(gapsfile[:,0]==order)
    gaps1d    = gapsfile[selection,1:]
    return gaps1d
#==============================================================================
#
#                         L I N E      F I T T I N G                  
#
#==============================================================================

def gauss(x,flux,bkg,error,model='SingleGaussian',output_model=False,
          *args,**kwargs):
    line_model   = getattr(emline,model)
    line         = line_model(x,flux-bkg,error)
    pars, errors = line.fit(bounded=False)
    chisq        = line.rchi2
    if output_model:
        model = line.evaluate()
        return pars, errors, chisq, model
    else:
        return pars, errors, chisq
    
    
#==============================================================================
#
#        W A V E L E N G T H     S O L U T I O N      F I T T I N G                  
#
#==============================================================================
        
def wavesol1d(centers,wavelengths,cerror,werror,polyord,usepatches=True):
    """
    Uses 'patch' to fit polynomials of degree given by polyord keyword.
    
    
    If 'usepatch' is true, divides the data into 8 bins, each 512 pix wide. A
    separate polyonomial solution is derived for each patch.
    """
    if usepatches==True:
        numpatch = 8
    else:
        numpatch = 1
            
    npix = 4096
    
    patchlims = np.linspace(npix//numpatch,npix,numpatch)
    binned = np.digitize(centers,patchlims)
    # new container
    coeffs = container.coeffs(polyord,numpatch)
    for i in range(numpatch):
        sel = np.where(binned==i)
        
        pars, errs = patch(centers[sel],wavelengths[sel],
                               cerror[sel],werror[sel],polyord)
        
        coeffs[i]['pixl'] = patchlims[i]-npix//numpatch
        coeffs[i]['pixr'] = patchlims[i]
        coeffs[i]['pars'] = pars
        coeffs[i]['errs'] = errs
    return coeffs
    
def wavesol(linedict,polyord,fit='gauss',usepatches=True):
    """
    Fits the wavelength solution to the data provided in the linedict.
    Calls 'wavesol1d' for all orders in linedict.
    
    Uses Gaussian profiles as default input.
    
    Returns:
    -------
        wavesol2d : dictionary with coefficients for each order in linedict
        
    """
    wavesol2d = {}
    for extname, linelist in linedict.items():
        centers1d = linelist[fit][:,1]
        cerrors1d = linelist['{fit}_err'.format(fit=fit)][:,1]
        wavelen1d = 1e10*(c/linelist['freq'])
        werrors1d = 1e10*(c/linelist['freq']**2 * freq_err)
        
        wavesol2d[extname] = wavesol1d(centers1d,wavelen1d,
                                       cerrors1d,werrors1d,
                                       polyord,usepatches)
    return wavesol2d
def patch(centers,wavelengths,cerror,werror,polyord):
    """
    Fits a polynomial to the provided data and errors.
    Uses scipy's Orthogonal distance regression package in order to take into
    account the errors in both x and y directions.
    
    Returns:
    -------
        coef : len(polyord) array
        errs : len(polyord) array
    """
    if np.size(centers)>polyord:
        # beta0 is the initial guess
        beta0 = np.polyfit(centers,wavelengths,polyord)[::-1]
        data  = odr.RealData(centers,wavelengths,sx=cerror,sy=werror)
        model = odr.polynomial(order=polyord)
        ODR   = odr.ODR(data,model,beta0=beta0)
        out   = ODR.run()
        pars  = out.beta
        errs  = out.sd_beta
        
    return pars, errs
        