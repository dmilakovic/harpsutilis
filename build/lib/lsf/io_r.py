#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:14:09 2023

@author: dmilakov
"""

from fitsio import FITS
import os
import harps.settings as hs
import numpy as np
import harps.lsf.classes as lclass

def read_lsf(fibre,specifier,method,version=-1):
    
    # specifier must be either a string (['round','octog']) or a np.datetime64
    # instance. 
    if isinstance(specifier,str):
        shape = specifier[0:5]
    elif isinstance(specifier,np.datetime64):
        if specifier<=np.datetime64('2015-05-01'):
            shape = 'round'
        else:
            shape = 'octog'
    else:
        print("Fibre shape unknown")
    assert shape in ['round','octog']
    # assert method in ['spline','analytic','gp']
    filename ='LSF_{f}_{s}_{m}.fits'.format(f=fibre,s=shape,m=method)
    hdu = FITS(os.path.join(hs.dirnames['lsf'],filename))
    lsf = hdu[-1].read()
    return lclass.LSF(lsf)

def from_file(filepath,nhdu=-1):
    hdu = FITS(filepath)
    lsf = hdu[nhdu].read()
    return lclass.LSF(lsf)