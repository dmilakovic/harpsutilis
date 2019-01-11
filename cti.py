#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 12:40:02 2019

@author: dmilakov
"""
from harps.core import np



def log(flux,fibre=None,pars=None,sigma=None):
    defpars = dict(A=(-0.6750527, -2.16896985), B=(-0.15704125, -0.86041432))
    defsigma= dict(A=(0.06426434, 0.05278),     B=(0.06277495, 0.04825154))
    
    valid = bool(fibre)^(bool(pars)&bool(sigma))
    
    assert valid==True, "Incorrect usage"
    a, b = pars if pars is not None else defpars[fibre]
    sigmaa, sigmab = sigma if sigma is not None else defsigma[fibre]
    
    x     = np.log10(flux/1e6)
    shift = -(a + b*x)
    noise = np.sqrt(sigmaa**2 + (sigmab*x)**2)
    return shift, noise

def exp(flux,fibre=None,pars=None,sigma=None):
    defpars = dict(A=(3.64711121e+00, 7.18010765e+04),
                   B=(2.07251092e+00, 9.08949137e+04))
    defsigma = dict(A=(6.98522127e-02, 2.70313985e+03),
                    B=(9.12855033e-02, 8.82314095e+03))
    
    valid = bool(fibre)^(bool(pars)&bool(sigma))
    
    assert valid==True, "Incorrect usage"
    a, b = pars if pars is not None else defpars[fibre]
    sigmaa, sigmab = sigma if sigma is not None else defsigma[fibre]
    
    x      = np.exp(-flux/b)
    shift  = - a * x
    noise  = x* np.sqrt(sigmaa**2 + (a/b*sigmab)**2)
    return shift, noise

