#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:31:14 2023

@author: dmilakov
"""
import numpy as np
from scipy.special import erf, wofz
#------------------------------------------------------------------------------
# 
#                           G A U S S I A N S
#
#------------------------------------------------------------------------------
    
def double_gaussN_erf(x,params):
    
    N = params.shape[0]
    y = np.zeros_like(x,dtype=np.float)
    xb = (x[:-1]+x[1:])/2
    gauss1 = params[['amplitude1','center1','sigma1']]
    gauss2 = params[['amplitude2','center2','sigma2']]
    for i in range(N):
        A1, mu1, sigma1 = gauss1.iloc[i]
        A2, mu2, sigma2 = gauss2.iloc[i]
        
        e11 = erf((xb[:-1]-mu1)/(np.sqrt(2)*sigma1))
        e21 = erf((xb[1:] -mu1)/(np.sqrt(2)*sigma1))
        y[1:-1] += A1*sigma1*np.sqrt(np.pi/2)*(e21-e11)
        
        e12 = erf((xb[:-1]-mu2)/(np.sqrt(2)*sigma2))
        e22 = erf((xb[1:] -mu2)/(np.sqrt(2)*sigma2))
        y[1:-1] += A2*sigma2*np.sqrt(np.pi/2)*(e22-e12)
        
        
    return y
def gaussN_erf(boundaries,params):
    x = np.atleast_2d(boundaries)
    N = params.shape[0]
    y = np.zeros_like(x,dtype=np.float)
    xb = (x[:-1]+x[1:])/2
    for i in range(N):
        A,mu,sigma,A_error,mu_error,sigma_error,pn,ct = params.iloc[i]
        sigma = np.abs(sigma)
        e1 = erf((xb[:-1]-mu)/(np.sqrt(2)*sigma))
        e2 = erf((xb[1:] -mu)/(np.sqrt(2)*sigma))
        y[1:-1] += A*sigma*np.sqrt(np.pi/2)*(e2-e1)
    return y
def integrated_gauss(boundaries,pars):
    ''' 
    The integral of a Gaussian between two points, x1 and x2, is calculated
    as:
        
        Phi(x1,x2) = A * sigma * sqrt(pi/2) * [erf(t2) - erf(t1)]
    
    Where A and sigma are the amplitude and the variance of a Gaussian, 
    and 't' is defined as:
        
        t = (x - mu)/(sqrt(2) * sigma)
    
    Here, mu is the mean of the Gaussian.
    '''
    
    A, mu, sigma = pars
    xb = np.atleast_2d(boundaries)
    e1  = erf((xb[0,:]-mu)/(np.sqrt(2)*sigma))
    e2  = erf((xb[1,:]-mu)/(np.sqrt(2)*sigma))
    y   = A*sigma*np.sqrt(np.pi/2)*(e2-e1)
    
    return y

def gauss4p(x, amplitude, center, sigma, y0 ):
    # Four parameters: amplitude, center, width, y-offset
    #y = np.zeros_like(x,dtype=np.float64)
    #A, mu, sigma, y0 = p
    y = y0+ amplitude/np.sqrt(2*np.pi)/sigma*np.exp((-((x-center)/sigma)**2)/2)
    return y
def gauss3p(x, amplitude, center, sigma):
    # Three parameters: amplitude, center, width
    #y = np.zeros_like(x)
    #A, mu, sigma = p
    y = amplitude/np.sqrt(2*np.pi)/sigma*np.exp((-((x-center)/sigma)**2)/2)
    return y
def gaussN(x, *params):
    # Three parameters: amplitude, center, width
    y = np.zeros_like(x)
    #A, mu, sigma = p
    for i in range(0,len(params),3):
        a = params[i]
        c = params[i+1]
        s = params[i+2]
        y = y + a/np.sqrt(2*np.pi)/s*np.exp((-((x-c)/s)**2)/2.)
    return y
def gaussP(x,*params,xrange=(-5,5),step=1,**kwargs):
    return_components=kwargs.pop('return_components',False)
    return_center=kwargs.pop('return_center',False)
    return_sigma=kwargs.pop('return_sigma',False)
    
    return_tuple = False
    if return_center or return_sigma:
        return_tuple = True
    xmin, xmax = xrange
    size = np.abs(xmax)+np.abs(xmin)
    N       = int((size)/step)   # number of side gaussians 
    sigma0  = step/2.355 
    assert len(params)==N+2, "{0:2d} parameters provided, {1:2d} required ".format(len(params),N+2)
    centers_ = np.linspace(xmin,xmax,N+1)
    centers__= np.delete(centers_,N//2)
    centers  = np.insert(centers__,0,0)
    #print(centers)
    y       = np.zeros_like(x)
    if return_components:
        ylist = []
    for i in range(len(centers)):
        if i == 0:
            sigma = params[0]
            amp   = params[1]
        else:
            sigma = sigma0
            amp   = params[i+1]
        y_ = gauss3p(x,amp,centers[i],sigma)
        y  = y + y_
        if return_components:
            ylist.append(y_)
    
    if not return_components:
        val = y
    else:
        val = ylist
    if return_tuple:
        tupval = (val,)
        if return_center:
            tupval = tupval+(centers,)
        if return_sigma:
            tupval= tupval + (sigma0,)
        return tupval
    else:
        return val
    
###############################################################################
################            LINE PROFILE GENERATION            ################
def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-(x / alpha)**2 * np.log(2))

def L(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def La(x, A, gamma):
    return A * L(x,gamma)

def V(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)
def Va(x, A, alpha, gamma):
    return A * V(x, alpha, gamma)

    