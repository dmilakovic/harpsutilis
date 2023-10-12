#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:27:34 2023

@author: dmilakov
"""

import numpy as np
import harps.functions.spectral as specfunc
import harps.background as bkg

def prepare_data1d(wav1d,flx1d,redisperse=False,bkg1d=None,subbkg=True,
                   velocity_step=0.82,plot=False):
    assert len(flx1d)==len(wav1d)
    npix = len(flx1d)
    
    err1d = np.sqrt(flx1d)
    if redisperse:
        wav1d,flx1d,err1d=specfunc.redisperse1d(wav1d, flx1d, err1d,
                                                velocity_step)
    
    if subbkg:
        if bkg1d is not None:
            if redisperse:
                _,bkg1d,__=specfunc.redisperse1d(wav1d, bkg1d, np.sqrt(bkg1d),
                                                        velocity_step)
        else:
            env1d,bkg1d = bkg.get_env_bkg(flx1d,xarray=None,yerror=err1d,
                                          kind='fit_spline')
        flx1d = flx1d-bkg1d
    
        err1d = np.sqrt(flx1d + bkg1d)
    
    return wav1d,flx1d,err1d

def prepare_data2d(wav2d,flx2d,redisperse=False,bkg2d=None,subbkg=True,
                   velocity_step=0.82,plot=False,sOrder=39,eOrder=None):
    assert np.shape(flx2d)==np.shape(wav2d)
    nbo,npix = np.shape(flx2d)
    
    err2d = np.sqrt(np.abs(flx2d))
    if redisperse:
        wav2d,flx2d,err2d=specfunc.redisperse2d(wav2d, flx2d, err2d,
                                                velocity_step)
    
    if subbkg:
        if bkg2d is not None:
            if redisperse:
                _,bkg2d,__=specfunc.redisperse2d(wav2d, bkg2d, np.sqrt(bkg2d),
                                                        velocity_step)
        else:
            env2d,bkg2d = bkg.get_env_bkg2d_from_array(flx2d,
                                                       sOrder=sOrder,
                                                       eOrder=eOrder,
                                                       kind='fit_spline')
        flx2d = flx2d-bkg2d
    
        err2d = np.sqrt(np.abs(flx2d + bkg2d))
    
    return wav2d,flx2d,err2d

    
    
    
        
    
    
    
    