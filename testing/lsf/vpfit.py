#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:11:24 2023

@author: dmilakov
"""

from fitsio import FITS
from spectres import spectres
import numpy as np


def return_ip(lsf2d,wave2d,error2d,regions,dv,subpix):
    '''
    

    Parameters
    ----------
    lsf2d : lsf.container.LSF2d object
        Contains numerical models relevant for this dataset.
    wave2d : 2d-array
        Contains the wavelength solution for this dataset.
    error2d : 2d-array
        Contains the spectral error array for this dataset.
    regions : a list of 2-ple
        A list of tuples (wstart,wend) containing starting and ending 
        wavelengths for which to construct the IP.
    dv : scalar
        The pixel size in km/s.
    subpix : scalar
        The number of subpixel steps. Must be an odd number.

    Returns
    -------
    None.

    '''
    assert subpix%2==1, "Parameter 'subpixel' must be an odd number"
    ip = None
    for i,(wave1,wave2) in enumerate(regions):
        
        cut = np.where((wave2d>=wave1)&(wave2d<=wave2))
        
    
    return ip
    


if __name__=='__main__':
    # do stuff
    pass