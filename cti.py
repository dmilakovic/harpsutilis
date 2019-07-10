#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 12:40:02 2019

@author: dmilakov
"""
from harps.core import np,os, json
import harps.settings as hs
#from harps.dataset import methods
methods = ['wavesol','coeff','freq','cent']


#def log(flux,fibre=None,pars=None,sigma=None):
#    defpars = dict(A=(-0.6750527, -2.16896985), B=(-0.15704125, -0.86041432))
#    defsigma= dict(A=(0.06426434, 0.05278),     B=(0.06277495, 0.04825154))
#    
#    valid = bool(fibre)^(bool(pars)&bool(sigma))
#    
#    assert valid==True, "Incorrect usage"
#    a, b = pars if pars is not None else defpars[fibre]
#    sigmaa, sigmab = sigma if sigma is not None else defsigma[fibre]
#    
#    x     = np.log10(flux/1e6)
#    shift = -(a + b*x)
#    noise = np.sqrt(sigmaa**2 + (sigmab*x)**2)
#    return shift, noise

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

ctifolder   = hs.dirnames['cti']
ctimodel    = os.path.join(ctifolder,'model.npy')
exppars     = np.load(ctimodel)[0]

#%% Simple model y = A + B*log(x) 
def log_model(xdata,*pars):
    A,B = pars
    return A+B*np.log10(xdata)
#%% Exponential model y = a + b * exp(-x/c)
def exp_model(xdata,*pars):
    a,b,c = pars
    return a + b*np.exp(-xdata/c)
#%%
def exp(flux,fibre=None,fittype=None,method=None,pars=None,sigma=None):
    
    
    # either (fittype and method and fibre) or (pars and sigma) provided
    valid = (bool(fittype)&bool(method)&bool(fibre))^(bool(pars)&bool(sigma))
    
    assert valid==True, "Incorrect usage"
    
    custompars = bool(pars)&bool(sigma)
    if custompars:
        pars  = pars
        sigma = sigma
    else:
#        a,b = exppars[fibre][fittype][method]['pars'].T
        pars,sigma = read_model(fibre,fittype,method,'exp')
        
    a, b, c = pars
    sigmaa, sigmab, sigmac = sigma
    
    x      =  np.exp(-flux/c)
    shift  = - exp_model(flux,*pars)
#    noise  = x* np.sqrt(sigmaa**2 + (a/b*sigmab)**2)
    noise  = np.sqrt(sigmaa**2 + (x*sigmab)**2 + ((flux*x)/c**2*sigmac)**2)
    return shift, noise
def log(flux,fibre=None,fittype=None,method=None,pars=None,sigma=None):
    
    
    # either (fittype and method and fibre) or (pars and sigma) provided
    valid = (bool(fittype)&bool(method)&bool(fibre))^(bool(pars)&bool(sigma))
    
    assert valid==True, "Incorrect usage"
    
    custompars = bool(pars)&bool(sigma)
    if custompars:
        pars  = pars
        sigma = sigma
    else:
#        a,b = exppars[fibre][fittype][method]['pars'].T
        pars,sigma = read_model(fibre,fittype,method,'log')
        
    a, b = pars
    sigmaa, sigmab = sigma
    
    x       = np.log10(flux)
    shift   = -(a + b*x)
    noise   = np.sqrt(sigmaa**2 + (sigmab*x)**2)
    return shift, noise

def from_file(filepath):
    with open(filepath,'r') as file:
        data = json.load(file)
    
    return data

def read_model(fibre,fittype,method,model,version=None):
    dirname = hs.get_dirname('cti',version)
#    filename = 'cti_model_2015-04-17-gray-' +\
    method = 'coeff'
    filename = 'cti_model_2012-02-15-' +\
               '{}_{}_{}_{}.json'.format(fibre,fittype,method,model)
#    print("USING : {}".format(filename))
    filepath = os.path.join(dirname,filename)
    data     = from_file(filepath)
    
    pars     = data['pars']
    errs     = data['errs']
    return pars, errs


