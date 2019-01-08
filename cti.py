#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 12:40:02 2019

@author: dmilakov
"""
from harps.core import np

def model(flux,pars=None,sigma=None):
    A, B = pars if pars is not None else (0.26290918, 0.70655505)
    sigmaA, sigmaB = sigma if sigma is not None else (0.05737292, 0.08646136)
    
    x     = np.log10(flux/1e6)
    shift = -(A + B*x)
    noise = np.sqrt(sigmaA**2 + (sigmaB*x)**2)
    return shift, noise