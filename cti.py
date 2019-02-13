#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 12:40:02 2019

@author: dmilakov
"""
from harps.core import np
#from harps.dataset import methods
methods = ['wavesol','coeff','freq','cent']


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

#defpars = dict(A=(3.64711121e+00, 7.18010765e+04),
#                   B=(2.07251092e+00, 9.08949137e+04))
#defsigma = dict(A=(6.98522127e-02, 2.70313985e+03),
#                B=(9.12855033e-02, 8.82314095e+03))
#exppars = dict(wavesol=dict(A=(3.57303061, 93694.04128564),
#                            B=(1.75453919, 133507.79694895)),
#               coeff=dict(A=(3.91556209, 87679.37158182),
#                          B=(2.07769575, 119736.92684871)),
#               freq=dict(A=(3.86970705, 77536.54916097),
#                         B=(1.73594352, 120012.53545720)),
#               cent=dict(A=(3.87366875, 77492.65889866),
#                         B=(1.74009125, 119776.80525635)))
#expsigma = dict(wavesol=dict(A=(0.07414590, 3900.05439915),
#                             B=(0.06390793, 11672.18368664)),
#                coeff=dict(A=(0.07010021, 3074.92395187),
#                           B=(0.06181612, 8192.22068529)),
#                freq=dict(A=(0.07934977, 2982.39685268),
#                          B=(0.05876572, 9349.65270190)),
#                cent=dict(A=(0.07932868, 2976.28385943),
#                          B=(0.05877583, 9303.63099125)))
#methods = ['wavesol','coeff','freq','cent']
valdtype = np.dtype([('pars','float64',(2,)),('errs','float64',(2,))])
metdtype = np.dtype([(method,valdtype) for method in methods])     
fitdtype = np.dtype([('lsf',metdtype), ('gauss',metdtype)])
pardtype = np.dtype([('A',fitdtype),('B',fitdtype)])

exppars     = np.load('/Users/dmilakov/harps/dataprod/cti/model.npy')[0]


def exp(flux,fibre=None,fittype=None,method=None,pars=None,sigma=None):
    
    
    # either (fittype and method and fibre) or (pars and sigma) provided
    valid = (bool(fittype)&bool(method)&bool(fibre))^(bool(pars)&bool(sigma))
    
    assert valid==True, "Incorrect usage"
    if pars is not None:
        a,b = pars
    else:
        a,b = exppars[fibre][fittype][method]['pars'].T
    if sigma is not None:
        sigmaa,sigmab = sigma
    else:
        sigmaa,sigmab = exppars[fibre][fittype][method]['errs'].T
    
    x      = np.exp(-flux/b)
    shift  = - a * x
    noise  = x* np.sqrt(sigmaa**2 + (a/b*sigmab)**2)
    return shift, noise

