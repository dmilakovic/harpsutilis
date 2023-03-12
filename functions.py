#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:43:28 2018

@author: dmilakov
"""
# import numpy as np
# import pandas as pd
import math
#import xarray as xr
import sys
import os
import re

from harps import peakdetect as pkd
from harps import emissionline as emline
from harps import settings as hs
from harps.constants import c
import harps.containers as container

from harps.core import welch, logging

from scipy.special import erf, wofz, gamma, gammaincc, expn
from scipy.optimize import minimize, leastsq, curve_fit, brentq
from scipy.interpolate import splev, splrep
from scipy.integrate import quad

from glob import glob

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from matplotlib import pyplot as plt
#from kapteyn import kmpfit

__version__   = hs.__version__

hs.setup_logging()
# some shared lists for 'return_empty_dataset' and 'return_empty_dataarray'

#------------------------------------------------------------------------------
# 
#                           P R O P O S A L S
#
#------------------------------------------------------------------------------
def accuracy(w=None,SNR=10,dx=0.829,u=0.9):
    '''
    Returns the rms accuracy [km/s] of a spectral line with SNR=10, 
    pixel size = 0.829 km/s and apsorption strength 90%.
    
    Equation 4 from Cayrel 1988 "Data Analysis"
    
    Parameters
    ----------
    w : float
        The equivalent width of the line (in pix). The default is None.
    SNR : TYPE, optional
        DESCRIPTION. The default is 10.
    dx : TYPE, optional
        DESCRIPTION. The default is 829.
    u : TYPE, optional
        DESCRIPTION. The default is 0.9.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if w is None:
        raise ValueError("No width specified")
    epsilon = 1/SNR
    return np.sqrt(2)/np.pi**0.25 * np.sqrt(w*dx)*epsilon/u
def equivalent_width(SNR,wave,R,dx):
    ''' Cayrel 1988 formula 6
    SNR : rms of the signal to noise of the line
    wave: wavelength of the line
    R   : resolution at wavelength
    dx  : pixel size (in A)
    '''
    FWHM = wave / R
    epsilon=1/SNR
    return 1.5*np.sqrt(FWHM*dx)*epsilon
def PN_Murphy(R,SNR,FWHM):
    '''R = resolution, SNR = peak signal/noise, FWHM = FWHM of line [pix]'''
    FWHM_inst = 2.99792458e8/R 
    dv = 0.41 * FWHM_inst / (SNR * np.sqrt(FWHM))
    return dv
def min_equivalent_width(n,FWHM,SNR):
    return n*FWHM/SNR
def min_SNR(n,FWHM,EW):
    return n*FWHM/EW
def schechter_cdf(m,A=1,beta=2,m0=100,mmin=10,mmax=None,npts=1e4):
    """
    Return the CDF value of a given mass for a set mmin,mmax
    mmax will default to 10 m0 if not specified
    
    Analytic integral of the Schechter function:
        x^-a + exp (-x/m) 
    http://www.wolframalpha.com/input/?i=integral%28x^-a+exp%28-x%2Fm%29+dx%29
    """
    if mmax is None:
        mmax = 10*m0
    
    # integrate the CDF from the minimum to maximum 
    # undefined posint = -m0 * mmax**-beta * (mmax/m0)**beta * scipy.special.gammainc(1-beta, mmax/m0)
    # undefined negint = -m0 * mmin**-beta * (mmin/m0)**beta * scipy.special.gammainc(1-beta, mmin/m0)
    posint = -mmax**(1-beta) * expn(beta, mmax/m0)
    negint = -mmin**(1-beta) * expn(beta, mmin/m0)
    tot = posint-negint

    # normalize by the integral
    # undefined ret = (-m0 * m**-beta * (m/m0)**beta * scipy.special.gammainc(1-beta, m/m0)) / tot
    ret = (-m**(1-beta) * expn(beta, m/m0) - negint)/ tot

    return ret 
def schechter(x,norm,alpha,cutoff):
    return norm*((x/cutoff)**alpha)*np.exp(-x/cutoff)
def schechter_int(x,norm,alpha,cutoff):
    return norm*cutoff**(-alpha+1)*gamma(alpha+1)*gammaincc(alpha+1,x/cutoff)
def delta_x(z1,z2):
    ''' Returns total absorption path between redshifts z1 and z2'''
    def integrand(z):
        return (1+z)**2/np.sqrt(0.3*(1+z)**3+0.7)
    return quad(integrand,z1,z2)
def calc_lambda(x,dx,order,wavl):
    pol=wavl[order.astype(int),:]
    return pol[:,0]+pol[:,1]*x+pol[:,2]*x**2+pol[:,3]*x**3,(pol[:,1]+2*pol[:,2]*x+3*pol[:,3]*x**2)*dx

def chisq(params,x,data,weights=None):
    amp, ctr, sgm = params
    if weights==None:
        weights = np.ones(x.shape)
    fit    = gauss3p(x,amp,ctr,sgm)
    chisq  = ((data - fit)**2/weights).sum()
    return chisq
def lambda_at_vz(v,z,l0):
    '''Returns the wavelength of at redshift z moved by a velocity offset of v [m/s]'''
    return l0*(1+z)*(1+v/c)

def wave_(WL0,CRPIX1,CD1,NPIX):
    wa =np.array([np.power(10,WL0+((i+1)-CRPIX1)*CD1) for i in range(NPIX)])
    return wa
def wave_from_header(header):
    try:
        wl0 = header['crval1']
    except:
        wl0 = np.log10(header['up_wlsrt'])
    crpix1 = header['crpix1']
    cd1    = header['cd1_1']
    npix   = header['naxis1']
    return wave_(wl0,crpix1,cd1,npix,)
    
#------------------------------------------------------------------------------
# 
#                           M A T H E M A T I C S
#
#------------------------------------------------------------------------------
def derivative1d(y,x=None,order=1,method='coeff'):
    ''' Get the first derivative of a 1d array y. 
    Modified from 
    http://stackoverflow.com/questions/18498457/
    numpy-gradient-function-and-numerical-derivatives'''
    # def _contains_nan(array):
    #     return np.any(np.isnan(array))
    # contains_nan = [_contains_nan(array) for array in [y,x]]
    # x = x if x is not None else np.arange(len(y))
    # if any(contains_nan)==True:
    #     return np.zeros_like(y)
    # else:
    #     pass
    if method=='forward':
        dx = np.diff(x,order)
        dy = np.diff(y,order)
        d  = dy/dx
    if method == 'central':
        z1  = np.hstack((y[0], y[:-1]))
        z2  = np.hstack((y[1:], y[-1]))
        dx1 = np.hstack((0, np.diff(x)))
        dx2 = np.hstack((np.diff(x), 0))  
        if np.all(np.asarray(dx1+dx2)==0):
            dx1 = dx2 = np.ones_like(x)/2
        d   = (z2-z1) / (dx2+dx1)
    if method == 'coeff':
        d = derivative(y,x,order)
    return d
def derivative(y_axis,x_axis=None,order=1,accuracy=4):
    if order==1:
        _coeffs = {2:[-1/2,0,1/2],
                  4:[1/12,-2/3,0,2/3,-1/12],
                  6:[-1/60,3/20,-3/4,0,3/4,-3/20,1/60],
                  8:[1/280,-4/105,1/5,-4/5,0,4/5,-1/5,4/105,-1/280]}
    elif order==2:
        _coeffs = {2:[1,-2,1],
                   4:[-1/12, 4/3, -5/2, 4/3, -1/12],
                   6:[1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                   8:[-1/560, 8/315	, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
            }
    x_axis = x_axis if x_axis is not None else jnp.arange(np.shape(y_axis)[-1])
    return (-1)**order*_derivative(y_axis,x_axis,np.array(_coeffs[accuracy]))
# def _derivative(y_axis,x_axis,coeffs):    
#     N        = len(y_axis)
#     pad_width = int(len(coeffs)//2)
#     y_padded = np.pad(y_axis,pad_width,mode='symmetric')
#     x_padded = np.pad(x_axis,pad_width,mode='linear_ramp',
#                       end_values=(-pad_width,N+pad_width-1))
#     xcubed   = np.power(np.diff(x_padded),3)
#     h        = np.insert(xcubed,0,xcubed[0])
#     print(np.shape(y_axis),np.shape(x_axis),np.shape(y_padded),np.shape(h))
#     y_deriv  = np.convolve(y_padded, coeffs, 'same')/h
    
#     return y_deriv[pad_width:-pad_width]
def _derivative(y_axis,x_axis,coeffs):    
    coeffs = np.asarray(coeffs)
    # N        = len(y_axis)
    # pad_width = int(len(coeffs)//2)
    # y_padded = np.pad(y_axis,pad_width,mode='symmetric')
    # x_padded = np.pad(x_axis,pad_width,mode='linear_ramp',
                      # end_values=(-pad_width,N+pad_width-1))
   
    # print(np.shape(y_axis),np.shape(x_axis),np.shape(y_padded),np.shape(h))
    if len(np.shape(y_axis))>1:
        y, x   = jnp.broadcast_arrays(y_axis,x_axis)
        
        
        xcubed   = jnp.power(jnp.diff(x,axis=1),3.0)
        h        = jnp.insert(xcubed,0,xcubed[0][0],axis=1)
        
        L         = np.shape(coeffs)[0]
        coeffs_ = jnp.zeros((L,L))
        coeffs_  = coeffs_.at[L//2].set(coeffs)
        
        y_deriv  = jsp.signal.convolve(y,coeffs_,'same')/h
        # y_deriv  = jsp.signal.convolve2d(coeffs_,y,'same')/h
    else:   
        xcubed   = jnp.power(np.diff(x_axis),3)
        h        = jnp.insert(xcubed,0,xcubed[0])
        y_deriv  = jnp.convolve(y_axis, coeffs, 'same')/h
    
    return y_deriv




def derivative_eval(x,y_array,x_array):
    deriv=derivative1d(y_array,x_array,order=1,method='coeff')
    srep =splrep(x_array,deriv)
    return splev(x,srep) 

def derivative_zero(y_array,x_array,left,right):
    return brentq(derivative_eval,left,right,args=(y_array,x_array))
    # return None
    
    
def error_from_covar(func,pars,covar,x,N=1000):
    samples  = np.random.multivariate_normal(pars,covar,N)
    try:
        values_ = [func(x,*(sample)) for sample in samples]
    except:
        values_ = [func(x,sample) for sample in samples]
    error    = np.std(values_,axis=0)
    return error
def freq_to_lambda(freq):
    return 1e10*c/(freq) #/1e9
def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))
def integer_slice(i, n, m):
    # return nth to mth digit of i (as int)
    l = math.floor(math.log10(i)) + 1
    return i / int(pow(10, l - m)) % int(pow(10, m - n + 1))

def mad(x,around=None):
    ''' Returns median absolute deviation of input array'''
    around = around if around is not None else np.median(x)
    return np.median(np.abs(around-x))
def mean_abs_dev(x):
    ''' Returns the mean absolute deviaton of input array'''
    return np.mean(np.abs(x-np.mean(x)))
def nmoment(x, counts, c, n):
    ''' Calculates the nth moment of x around c using counts as weights'''
    #https://stackoverflow.com/questions/29064053/calculate-moments-mean-variance-of-distribution-in-python
    return np.sum(counts*(x-c)**n) / np.sum(counts)

def polynomial(x, *p):
    y = np.zeros_like(x,dtype=np.float64)
    for i,a in enumerate(p):
        y += a*x**i
    return y
def polyjac(x,*p):
    
#    y = np.zeros((len(p),len(x)),dtype=np.float64)
#    for i,a in enumerate(p):
#        y[i]= i*a*x**(i-1)
    y = np.array([i*a*x**(i-1) for i,a in enumerate(p)])
    return np.atleast_2d(y).T
def rms(x,around_mean=False,axis=None):
    ''' Returns root mean square of input array'''
    mean = np.nanmean(x,axis=axis) if around_mean==True else 0.0
    return np.sqrt(np.nanmean(np.square(x-mean),axis=axis))
def running_mean(x, N,pad_mode='symmetric',convolve_mode='same'):
    
    if convolve_mode=='same':
        x_pad = np.pad(x,N,mode=pad_mode)
        mean = np.convolve(x_pad, np.ones((N,))/N,mode=convolve_mode)
        return mean[N:-N]
    if convolve_mode=='valid':
        return mean
    if convolve_mode=='full':
        mean = np.convolve(x, np.ones((N,))/N,mode=convolve_mode)
        return mean[int(N/2-1):-int(N/2-1)-1]

def running_rms(x, N):
    x2 = np.power(x,2)
    window = np.ones(N)/float(N)
    return np.sqrt(np.convolve(x2, window, 'same'))
def running_std(x, N):
    import pandas as pd
        #return np.convolve(x, np.ones((N,))/N)[(N-1):]
    series = pd.Series(x)
    return series.rolling(N).std()
def round_to_closest(a,b):
    '''
    a (float, array of floats): number to round
    b (float): closest unit to round to
    '''
    if len(np.shape(a))>0:
        return np.array([round(_/b)*b for _ in a])
    else:
        return round(a/b)*b

def sig_clip(v):
       m1=np.mean(v,axis=-1)
       std1=np.std(v-m1,axis=-1)
       m2=np.mean(v[abs(v-m1)<5*std1],axis=-1)
       std2=np.std(v[abs(v-m2)<5*std1],axis=-1)
       m3=np.mean(v[abs(v-m2)<5*std2],axis=-1)
       std3=np.std(v[abs(v-m3)<5*std2],axis=-1)
       return abs(v-m3)<5*std3   
def sigclip1d(v,sigma=3,maxiter=10,converge_num=0.02,plot=False):
    from matplotlib.patches import Polygon
    v    = np.array(v)
    ct   = np.size(v)
    dim  = len(np.shape(v))
    iter = 0; c1 = 1.0; c2=0.0
    while (c1>=c2) and (iter<maxiter):
        lastct = ct
        mean   = np.nanmean(v)
        std    = np.nanstd(v-mean)
        cond   = abs(v-mean)<sigma*std
        cut    = np.where(cond)
        ct     = len(cut[0])
        
        c1     = abs(ct-lastct)
        c2     = converge_num*lastct
        iter  += 1
    if plot:
        if dim == 1:
            plt.figure(figsize=(12,6))
            plt.scatter(np.arange(len(v)),v,s=2,c="C0")        
            plt.scatter(np.arange(len(v))[~cond],v[~cond],
                            s=10,c="C1",marker='x')
            plt.axhline(mean,ls='-',c='r')
            plt.axhline(mean+sigma*std,ls='--',c='r')
            plt.axhline(mean-sigma*std,ls='--',c='r')
        if dim == 2:
            fig, ax = plt.subplots(1)
            im = ax.imshow(v,aspect='auto',
                       vmin=np.percentile(v[cond],3),
                       vmax=np.percentile(v[cond],97),
                       cmap=plt.cm.coolwarm)
            cb = fig.colorbar(im)
            for i,c in enumerate(cond):
                if np.all(c): 
                    continue
                else:
                    print(c)
                    x = [j for j in c if j is False]
                    print(x)
                    ax.add_patch(Polygon([[0, 0], [4, 1.1], [6, 2.5], [2, 1.4]], closed=True,
                      fill=False, hatch='/'))
#            plt.scatter(np.arange(len(v))[~cond],v[~cond],
#                            s=10,c="C1",marker='x')
#            plt.axhline(mean,ls='-',c='r')
#            plt.axhline(mean+sigma*std,ls='--',c='r')
#            plt.axhline(mean-sigma*std,ls='--',c='r')
    return cond

def sigclip2d(v,sigma=5,maxiter=100,converge_num=0.02):
    ct   = np.size(v)
    iter = 0; c1 = 1.0; c2=0.0
    while (c1>=c2) and (iter<maxiter):
        lastct = ct
        mean   = np.mean(v)
        std    = np.std(v-mean)
        cond   = abs(v-mean)<sigma*std
        cut    = np.where(cond)
        ct     = len(cut[0])
        
        c1     = abs(ct-lastct)
        c2     = converge_num*lastct
        iter  += 1
    return cond
def negpos(number):
    return -abs(number),abs(number)
def removenan(array):
    return array[~np.isnan(array)]
def nan_to_num(array):
    finite = np.isfinite(array)
    return array[finite]
def round_up_to_odd(f,b=1):
    return round_to_closest(np.ceil(f) // 2 * 2 + 1 ,b)
def round_up_to_even(f,b=1):
    return round_to_closest(np.ceil(f) // 2 * 2 ,b)
def round_down_to_odd(f,b=1):
    return round_to_closest(np.floor(f) // 2 * 2 + 1 ,b)
def round_down_to_even(f,b=1):
    return round_to_closest(np.floor(f) // 2 * 2 ,b)

def missing_elements(L, start, end):
    """
    https://stackoverflow.com/questions/16974047/
    efficient-way-to-find-missing-elements-in-an-integer-sequence
    """
    if end - start <= 1: 
        if L[end] - L[start] > 1:
            yield from range(L[start] + 1, L[end])
        return

    index = start + (end - start) // 2

    # is the lower half consecutive?
    consecutive_low =  L[index] == L[start] + (index - start)
    if not consecutive_low:
        yield from missing_elements(L, start, index)

    # is the upper part consecutive?
    consecutive_high =  L[index] == L[end] - (end - index)
    if not consecutive_high:
        yield from missing_elements(L, index, end)
def find_missing(integers_list,start=None,limit=None):
    """
    Given a list of integers and optionally a start and an end, finds all
    the integers from start to end that are not in the list.

    'start' and 'end' default respectivly to the first and the last item of the list.

    Doctest:

    >>> find_missing([1,2,3,5,6,7], 1, 7)
    [4]

    >>> find_missing([2,3,6,4,8], 2, 8)
    [5, 7]

    >>> find_missing([1,2,3,4], 1, 4)
    []

    >>> find_missing([11,1,1,2,3,2,3,2,3,2,4,5,6,7,8,9],1,11)
    [10]

    >>> find_missing([-1,0,1,3,7,20], -1, 7)
    [2, 4, 5, 6]

    >>> find_missing([-2,0,3], -5, 2)
    [-5, -4, -3, -1, 1, 2]

    >>> find_missing([2],4,5)
    [4, 5]

    >>> find_missing([3,5,6,7,8], -3, 5)
    [-3, -2, -1, 0, 1, 2, 4]

    >>> find_missing([1,2,4])
    [3]

    """
    # https://codereview.stackexchange.com/a/77890
    start = start if start is not None else integers_list[0]
    limit = limit if limit is not None else integers_list[-1]
    return [i for i in range(start,limit + 1) if i not in integers_list]
def average(values,errors=None):
    """
    Return the weighted average and weighted sample standard deviation.

    values, weights -- Numpy ndarrays with the same shape.

    Assumes that weights contains only integers (e.g. how many samples in each group).

    See also https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights
    """
    weights = np.atleast_1d(errors) if errors is not None else np.ones_like(values)
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    variance = variance*sum(weights)/(sum(weights)-1)
    return (average, np.sqrt(variance))
    
def wmean(values,errors=None):
    # errors = np.atleast_1d(errors) if errors is not None else np.ones_like(values)
    # variance = np.power(errors,2)
    # weights  = 1./variance 
    # mean  = np.nansum(values * weights) / np.nansum(weights)
    # sigma = 1./ np.sqrt(np.sum(weights))
    return average(values,errors)
def aicc(chisq,n,p):
    ''' Returns the Akiake information criterion value 
    chisq = chi square
    n     = number of points
    p     = number of free parameters
    
    '''
    return chisq + 2*p + 2*p*(p+1)/(n-p-1)
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
    
#def HermiteF(x,*params):
#------------------------------------------------------------------------------
# 
#                           P L O T T I N G
#
#------------------------------------------------------------------------------    
def figure(*args, **kwargs):
    return get_fig_axes(*args, **kwargs)

def get_fig_axes(naxes,ratios=None,title=None,sep=0.05,alignment="vertical",
                 figsize=(16,9),sharex=None,sharey=None,grid=None,
                 subtitles=None,presentation=False,enforce_figsize=False,
                 left=0.1,right=0.95,top=0.95,bottom=0.10,**kwargs):
    
    def get_grid(alignment,naxes):
        if alignment=="grid":
            ncols = np.int(round(np.sqrt(naxes)))
            nrows,lr = [np.int(k) for k in divmod(naxes,round(np.sqrt(naxes)))]
            if lr>0:
                nrows += 1     
        elif alignment=="vertical":
            ncols = 1
            nrows = naxes
        elif alignment=="horizontal":
            ncols = naxes
            nrows = 1
        grid = np.array([ncols,nrows],dtype=int)
        return grid
    
    fig         = plt.figure(figsize=figsize)
    if enforce_figsize:
        fig.set_size_inches(figsize)
    # Change color scheme and text size if producing plots for a presentation
    # assuming black background
    if presentation==True:
        spine_col = kwargs.pop('spine_color','w')
        text_size = kwargs.pop('text_size','20')
        hide_spine = kwargs.pop('hide_spine',[])
        spine_lw=kwargs.pop('spine_lw','1')
#        spine_ec=kwargs.pop('spine_ec','k')
    else:
        pass
    
    # Share X axis
    if sharex!=None:
        if type(sharex)==list:
            pass
        else:
            sharex = list(sharex for i in range(naxes))
    elif sharex==None:
        sharex = list(False for i in range(naxes))
    # First item with sharex==True:
    try:
        firstx = sharex.index(True)
    except:
        firstx = None
    # Share Y axis  
    if sharey!=None:
        if type(sharey)==list:
            pass
        else:
            sharey = list(sharey for i in range(naxes))
    elif sharey==None:
        sharey = list(False for i in range(naxes))
    # First item with sharey==True:
    try:
        firsty = sharey.index(True)
    except:
        firsty = None
    
    sharexy = [(sharex[i],sharey[i]) for i in range(naxes)]
    
    # Add title
    if title!=None:
        fig.suptitle(title)
    # Calculate canvas dimensions
    
    # GRID
    if grid==None:
        grid = get_grid(alignment,naxes)
    else:
        grid = np.array(grid,dtype=int)
    ncols,nrows = grid

    if ratios==None:
        ratios = np.array([np.ones(ncols),np.ones(nrows)])
    else:
        if np.size(np.shape(ratios))==1:
            if   alignment == 'vertical':
                ratios = np.array([np.ones(ncols),ratios])
            elif alignment == 'horizontal':
                ratios = np.array([ratios,np.ones(nrows)])
        elif np.size(np.shape(ratios))==2:
            ratios = np.array(ratios).reshape((ncols,nrows))
    top, bottom = (top,bottom)
    left, right = (left,right)
    W, H        = (right-left, top-bottom)
    s           = sep
    #h           = H/naxes - (naxes-1)/naxes*s
    
    h0          = (H - (nrows-1)*s)/np.sum(ratios[1])
    w0          = (W - (ncols-1)*s)/np.sum(ratios[0])
    axes        = []
    axsize      = []
    for c in range(ncols):
        for r in range(nrows):
            ratiosc = ratios[0][:c]
            ratiosr = ratios[1][:r+1]
            w  = ratios[0][c]*w0
            h  = ratios[1][r]*h0
            l  = left + np.sum(ratiosc)*w0 + c*s
            d  = top - np.sum(ratiosr)*h0 - r*s
            size  = [l,d,w,h] 
            axsize.append(size)       
    for i in range(naxes):   
        size   = axsize[i]
        sharex,sharey = sharexy[i]
        if i==0:
            axes.append(fig.add_axes(size))
        else:
            kwargs = {}
            if   (sharex==True  and sharey==False):
                kwargs["sharex"]=axes[firstx]
                #axes.append(fig.add_axes(size,sharex=axes[firstx]))
            elif (sharex==False and sharey==True):
                kwargs["sharey"]=axes[firsty]
                #axes.append(fig.add_axes(size,sharey=axes[firsty]))
            elif (sharex==True  and sharey==True):
                kwargs["sharex"]=axes[firstx]
                kwargs["sharey"]=axes[firsty]
                #axes.append(fig.add_axes(size,sharex=axes[firstx],sharey=axes[firsty]))
            elif (sharex==False and sharey==False): 
                pass
                #axes.append(fig.add_axes(size))
            axes.append(fig.add_axes(size,**kwargs))
    for a in axes:
        a.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))
    if presentation == True:
        for a in axes:
            #plt.setp(tuple(a.spines.values()), edgecolor=spine_ec)
            plt.setp(tuple(a.spines.values()), color=spine_col)
            plt.setp(tuple(a.spines.values()), linewidth=spine_lw)
            
            plt.setp(tuple(a.spines.values()), facecolor=spine_col)
            plt.setp([a.get_xticklines(), a.get_yticklines(),a.get_xticklabels(),a.get_yticklabels()], color=spine_col)
            plt.setp([a.get_xticklabels(),a.get_yticklabels()],size=text_size)
            for s in hide_spine:
                a.spines[s].set_visible(False)
                #plt.setp([a.get_xlabel(),a.get_ylabel()],color=spine_col,size=text_size)
            #plt.setp(a.get_yticklabels(),visible=False)
    else:
        pass
    
    return fig,axes

def make_ticks_sparser(axis,scale='x',ticknum=None,minval=None,maxval=None):
    ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
        ticks on a given scale (x or y)'''
    ticknum = ticknum if ticknum is not None else 4
    if scale=='x':
        if minval is None or maxval is None:
            minval,maxval = axis.get_xlim()
        axis.set_xticks(np.linspace(minval,maxval,ticknum))
    elif scale=='y':
        if minval is None or maxval is None:
            minval,maxval = axis.get_ylim()
        axis.set_yticks(np.linspace(minval,maxval,ticknum))
    return axis
#------------------------------------------------------------------------------
# 
#                       L I S T   M A N I P U L A T I O N
#
#------------------------------------------------------------------------------
def return_basenames(filelist):
    filelist_noext = [os.path.splitext(file)[0] for file in filelist]
    return [os.path.basename(file) for file in filelist_noext]
def return_filelist(dirpath,ftype,fibre,ext='fits'):  
    filename_pattern=os.path.join(dirpath,
                    "*{fbr}_{ftp}.{ext}".format(ftp=ftype,fbr=fibre,ext=ext))
    try:
        filelist=np.array(glob(filename_pattern))
    except:
        raise ValueError("No files of this type were found")
    return filelist

def prepare_orders(order=None):
    '''
    Returns an array or a list containing the input orders.
    '''
    if order is None:
        orders = np.arange(hs.sOrder,hs.eOrder,1)
    else:
        orders = to_list(order)
    return orders


def select_orders(orders):
    use = np.zeros((hs.nOrder,),dtype=bool); use.fill(False)
    for order in range(hs.sOrder,hs.eOrder,1):
        if order in orders:
            o = order - hs.sOrder
            use[o]=True
    col = np.where(use==True)[0]
    return col
def to_list(item):
    """ Pushes item into a list """
    if type(item)==int:
        items = [item]
    elif type(item)==np.int64:
        items = [item]
    elif type(item)==list:
        items = item
    elif type(item)==np.ndarray:
        items = list(item)
    elif type(item)==str or isinstance(item,np.str):
        items = [item]
    elif type(item)==tuple:
        items = [*item]
    elif item is None:
        items = None
    else:
        print('Unsupported type. Type provided:',type(item))
    return items    
def get_dirname(filetype,dirname=None):
    if dirname is not None:
        dirname = dirname
    else:
        dirname = hs.dirnames[filetype]
    print("DIRNAME = ",dirname)
    direxists = os.path.isdir(dirname)
    if not direxists:
        raise ValueError("Directory does not exist")
    else:
        return dirname

def basename_to_datetime(filename):
    ''' 
    Extracts the datetime of HARPS observations from the filename
    Args:
    -----
        filename - str or list
    Returns:
    -----
        datetime - np.datetime64 object or a list of np.datetime64 objects
    '''
    filenames = to_list(filename)
    datetimes = []
    for fn in filenames:
        bn = os.path.splitext(os.path.basename(fn))[0]
        p = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}.\d{3}")
        s = p.search(bn)
        if s:
            dt = np.datetime64(s[0].replace('_',':')) 
        else:
            dt = np.datetime64(None)
        datetimes.append(dt)
    if len(datetimes)==1:
        return datetimes[0]
    else:
        return datetimes
def datetime_to_tuple(datetime):
    def to_tuple(dt):
        return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    datetimes = np.atleast_1d(datetime)
    datelist  = datetimes.tolist()
    dt_tuple  = list(map(to_tuple,datelist))
    return dt_tuple
def datetime_to_record(datetime):
    datetimes = np.atleast_1d(datetime)
    datelist  = datetimes.tolist()
    dt_record = container.datetime(len(datetimes))
    for dtr,dtv in zip(dt_record,datelist):
        dtr['year']  = dtv.year
        dtr['month'] = dtv.month
        dtr['day']   = dtv.day
        dtr['hour']  = dtv.hour
        dtr['min']   = dtv.minute
        dtr['sec']   = dtv.second
    return dt_record
def record_to_datetime(record):
    if record.dtype.fields is None:
        raise ValueError("Input must be a structured numpy array")   
    if isinstance(record,np.void):
        dt = '{0:4}-{1:02}-{2:02}T{3:02}:{4:02}:{5:02}'.format(*record)
        datetime = np.datetime64(dt)
    elif isinstance(record,np.ndarray):
        datetime = np.zeros_like(record,dtype='datetime64[s]')
        for i,rec in enumerate(record):
            dt='{0:4}-{1:02}-{2:02}T{3:02}:{4:02}:{5:02}'.format(*rec)
            datetime[i] = dt
    return datetime
def tuple_to_datetime(value):
    def to_datetime(value):
        return np.datetime64('{0:4}-{1:02}-{2:02}'
                             'T{3:02}:{4:02}:{5:02}'.format(*value))
    values = np.atleast_1d(value)
    datetimes = np.array(list(map(to_datetime,values)))
    return datetimes
from collections import defaultdict
def list_duplicates(seq):
    # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    """ Return a dictionary of duplicates of the input sequence """
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)
def find_nearest(array1,array2):
    ''' UNUSED''' 
    idx = []
    lim = np.median(np.diff(array1))
    for value in array1:
        distances = np.abs(array2-value)
        closest   = distances.min()
        if closest <= lim:
            idc = distances.argmin()
            idx.append(idc)
        else:
            continue
    return array2[idx]

def flatten_list(inlist):
    outlist = [item for sublist in inlist for item in sublist]
    return outlist
def ravel(array,removenan=True):
    a = np.ravel(array)
    if removenan:
        a = a[~np.isnan(a)]
    return a

def read_filelist(filepath):
    if os.path.isfile(filepath):
        mode = 'r+'
    else:
        mode = 'a+'
    filelist=[line.strip('\n') for line in open(filepath,mode)
              if line[0]!='#']
    return filelist
def overlap(a, b):
    # https://www.followthesheep.com/?p=1366
    a1=np.argsort(a)
    b1=np.argsort(b)
    # use searchsorted:
    sort_left_a=a[a1].searchsorted(b[b1], side='left')
    sort_right_a=a[a1].searchsorted(b[b1], side='right')
    #
    sort_left_b=b[b1].searchsorted(a[a1], side='left')
    sort_right_b=b[b1].searchsorted(a[a1], side='right')


    # # which values are in b but not in a?
    # inds_b=(sort_right_a-sort_left_a == 0).nonzero()[0]
    # # which values are in a but not in b?
    # inds_a=(sort_right_b-sort_left_b == 0).nonzero()[0]

    # which values of b are also in a?
    inds_b=(sort_right_a-sort_left_a > 0).nonzero()[0]
    # which values of a are also in b?
    inds_a=(sort_right_b-sort_left_b > 0).nonzero()[0]

    #return a1[inds_a], b1[inds_b]
    return inds_a,inds_b
def wrap_order(order,sOrder,eOrder):
    '''
    Returns an array or a list containing the input orders.
    '''
    orders = np.arange(eOrder)
    select = slice(sOrder,eOrder,1)
    if order is not None:
        select = slice_order(order)
    return orders[select]
def slice_order(order):
    #nbo = self.meta['nbo']
    start = None
    stop  = None
    step  = None
    if isinstance(order,int):
        start = order
        stop = order+1
        step = 1
    elif isinstance(order,tuple):
        numitems = np.shape(order)[0]
        if numitems==3:
            start, stop, step = order
        elif numitems==2:
            start, stop = order
            step = 1
        elif numitems==1:
            start = order
            stop  = order+1
            step  = 1
    return slice(start,stop,step)
import bisect
class Closest:
    """Assumes *no* redundant entries - all inputs must be unique"""
    def __init__(self, numlist=None, firstdistance=0):
        if numlist is None:
            numlist=[]
        self.numindexes = dict((val, n) for n, val in enumerate(numlist))
        self.nums = sorted(self.numindexes)
        self.firstdistance = firstdistance

    def append(self, num):
        if num in self.numindexes:
            raise ValueError("Cannot append '%s' it is already used" % str(num))
        self.numindexes[num] = len(self.nums)
        bisect.insort(self.nums, num)

    def rank(self, target):
        rank = bisect.bisect(self.nums, target)
        if rank == 0:
            pass
        elif len(self.nums) == rank:
            rank -= 1
        else:
            dist1 = target - self.nums[rank - 1]
            dist2 = self.nums[rank] - target
            if dist1 < dist2:
                rank -= 1
        return rank

    def closest(self, target):
        try:
            return self.numindexes[self.nums[self.rank(target)]]
        except IndexError:
            return 0

    def distance(self, target):
        rank = self.rank(target)
        try:
            dist = abs(self.nums[rank] - target)
        except IndexError:
            dist = self.firstdistance
        return dist
from   numpy.lib.recfunctions import append_fields
def stack_arrays(list_of_arrays):
    '''
    Stacks a list of structured arrays, adding a column indicating the position
    in the list.
    '''
    indices  = np.hstack([np.full(len(array),i)  for i,array \
                          in enumerate(list_of_arrays)])
    stacked0 = np.hstack(list_of_arrays)
    stacked  = append_fields(stacked0,'exp',indices,usemask=False)
    return stacked

#------------------------------------------------------------------------------
# 
#                           P E A K     D E T E C T I O N
#
#------------------------------------------------------------------------------


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.
    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
    Returns:
    --------
        mask : A numobservations-length boolean array.
    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def is_outlier_running(points, window=5,thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
#    plt.figure()
#    plt.plot(points,label='data')
    rmean = running_mean(points,window)
    # rmean = running_rms(points,window)
#    rmean = np.percentile(points,85)
    if len(points.shape) == 1:
        points = points[:,None]  
    diff  = np.sum((points-rmean)**2,axis=-1)
    diff  = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
#    plt.plot(rmean,label='rmean')
#    plt.plot(diff,label='diff')
#    plt.plot(modified_z_score,label='z_score')
#    plt.legend()
#    plt.show()
    return modified_z_score > thresh

def is_outlier_bins(points,idx,thresh=3.5):
    outliers = np.zeros_like(points)
    for i in np.unique(idx):
        cut = np.where(idx==i)[0]
        outliers[cut] = is_outlier(points[cut],thresh=thresh)
        # print(is_outlier(points[cut],thresh=thresh))
    return outliers.astype(bool)
        

def is_outlier_original(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def peakdet_limits(y_axis,plot=False):
    freq0, P0    = welch(y_axis,nperseg=300)
    cut = np.where(freq0>0.002)[0]
    freq, P = freq0[cut], P0[cut]
    maxind     = np.argmax(P)
    maxfreq    = freq[maxind]
    
    if plot:
        plt.figure()
        plt.plot(1./freq,P)
        
    
    # maxima and minima in the power spectrum
    maxima, minima = (np.transpose(x) for x in pkd.peakdetect(P,freq,
                                                              lookahead=2))
    minsorter  = np.argsort(minima[0])
    # find the most powerful peak in the power spectrum
    index      = np.searchsorted(minima[0],maxfreq,sorter=minsorter)
    # find minima surrounding the most powerful peak
    minfreq = (minima[0][index-1:index+1])
    try:
        maxdist, mindist = tuple(1./minfreq)
    except:
        maxdist = -1
        mindist = -1
    if plot:
        [plt.axvline(pos,c='C1',ls='--') for pos in tuple(1./minfreq)]
        
    mindist = max(mindist,5)
    maxdist = max(maxdist,10)
    return mindist,maxdist

def remove_false_maxima(x_axis,y_axis,input_xmin,input_ymin,limit,
                        mindist,maxdist,polyord=1,N=None,plot=True):
    '''
    Removes false minima (or maxima) by considering the distances between
    sequential extrema and the characteristics of the data.

    Parameters
    ----------
    x_axis : array-like
        Pixel number or wavelength.
    y_axis : array-like
        Flux or S/N.
    input_xmin : list
        x-coordinates of the extremes in y_axis.
    input_ymin : list
        y-coordinates of the extremes in y_axis.
    limit : float
        largest allowed distance between sequential extremes.
    mindist : float
        smallest allowed distance between sequential extremes.
    maxdist : float
        largest allowed distance between sequential extremes.
    polyord : int, optional
        the order of the polynomial fitted through the distances between
        extremes. The default is 1.
    N : int, optional
        the window (in pixels) over which RMS and mean are calculated and
        compared. The default is approximately len(yaxis)/10, rounded to 
        the closest 100.
    plot : bool, optional
        Plots the results. The default is True.

    Returns
    -------
    xmin : list
        Cleaned x-coordinates of extremes in y_axis.
    ymin : list
        Cleaned y-coordinates of extremes in y_axis.

    '''
    
    outliers   = np.full_like(input_xmin,True)
    N = int((maxdist+mindist)/2.)
    mean_y = running_mean(y_axis, N=N)
    new_xmin = input_xmin
    new_ymin = input_ymin
    j = 0
    if plot:
        fig,ax=figure(4,sharex=True,ratios=[3,1,1,1])
        ax[0].plot(x_axis,y_axis,drawstyle='steps-mid')
        ax[0].plot(x_axis,mean_y,drawstyle='steps-mid',
                    label='Mean over {} pixels'.format(N))
        ax[0].scatter(input_xmin,input_ymin,marker='^',c='red',s=30)
        ax[0].set_ylabel('Data')
        ax[1].set_ylabel('Distances between\nextremes')
        ax[2].set_ylabel("Residuals")
        ax[3].set_ylabel("Flux < mean\nover {} pixels".format(N))
    while sum(outliers)>0 and j<10:
        # print('BEGIN iteration {}, outliers={}/{}'.format(j, sum(outliers), len(outliers)))
        old_xmin = new_xmin
        old_ymin = new_ymin
        # Distances to the left and to the right neighbour
        dist_r = np.diff(old_xmin, append=old_xmin[-1]+mindist)
        dist_l = np.diff(np.roll(old_xmin,-1),prepend=old_xmin[0]-mindist) 
        
        # Due to rolling, last element of dist_l is wrong, set to 0
        dist_l[-1]=0
        
        # Difference in the distances between the left and right neighbour
        dist   = dist_r - dist_l; dist[-1] = 0
        # Fit a polynomial of order polyord to the left and right distances
        # Save residuals to the best fit model into arrays
        arrays = []
        for i,values in enumerate([dist_l,dist_r]):
            keep = (values>mindist) & (values<maxdist)
            pars, cov = np.polyfit(old_xmin[keep],
                                    values[keep],polyord,
                                      cov=True)
            model = np.polyval(pars,old_xmin)
            resid = values-model
            cond_ = np.abs(resid)>limit
            arrays.append(cond_)
            if plot:
                if i == 0:
                    c='b'; marker='<'  
                else:
                    c='r'; marker='>'
                ax[2].scatter(old_xmin,resid,marker=marker,c=c,s=15)
                [ax[2].axhline(l,c='r',lw=2) for l in [-limit,limit]]
        
        # Maxima for which residuals from the left AND from the right are
        # larger than some limit  
        cond0     = np.bitwise_and(*arrays)
        # Maxima for which distances from the immediate left and from the 
        # immediate right neighbour disagree by more than some limit
        cond1     = np.abs(dist)>limit
        # Maxima for which the distance to the left OR the distance to the 
        # right neighbour is smaller than the minimum allowed distance between
        # neighbouring maxima
        cond2     = np.bitwise_or(np.abs(dist_l)<mindist,
                                  np.abs(dist_r)<mindist)
        # Maxima for y_values which are below the running mean y_axis across 
        # N pixels 
        indices   = np.asarray(old_xmin,dtype=np.int16)
        cond3     = old_ymin<1.1*mean_y[indices]
        
        outliers_ = np.bitwise_and(cond1,cond0)
        outliers_ = np.bitwise_and(cond2,outliers_)
        outliers_ = np.bitwise_or(cond3,outliers_)
        cut       = np.where(outliers_==True)[0]
        if len(cut)>0:
            for i in cut:
                if i+2 in cut and outliers_[i]==True and i+2<len(outliers_):
                    # print("i={}, i+2={}".format(dist[i],dist[i+2]))
                    if np.sign(dist[i])!=np.sign(dist[i+2]):
                        # print("Changing values in outliers")
                        outliers_[i] = False
                        outliers_[i+1]=True
                        outliers_[i+2]=False
                else:
                    pass
        outliers = outliers_
        
        
        if plot:
            ax[0].scatter(old_xmin[outliers],old_ymin[outliers],marker='x',
                          c="C{}".format(j))
            ax[1].scatter(old_xmin,dist_l,marker='<',s=15,c='b')
                          # c="C{}".format(j))
            ax[1].scatter(old_xmin,dist_r,marker='>',s=15,c='r')
                            # c="C{}".format(j))
            [ax[1].axhline(l,c='r',lw=2) for l in [mindist,maxdist]]
            ax[2].scatter(old_xmin[cond0],arrays[0][cond0],marker='x',s=15,c='b')
            ax[2].scatter(old_xmin[cond0],arrays[1][cond0],marker='x',s=15,c='r')
            # ax[3].axhline(-limit,c='r',lw=2)
            # ax[3].axhline(limit,c='r',lw=2)
            ax[3].scatter(old_xmin,cond3,marker='x')
        new_xmin = (old_xmin[~outliers])
        new_ymin = (old_ymin[~outliers])
        # print('END iteration {}, outliers={}/{}'.format(j, sum(outliers), len(input_xmin)))
        j+=1
        
    xmin, ymin = new_xmin, new_ymin
    if plot:
        maxima0 = (np.roll(xmin,1)+xmin)/2
        maxima = np.array(maxima0[1:],dtype=np.int)
        [ax[0].axvline(x,ls=':',lw=0.5,c='r') for x in maxima]
        # ax[0].legend()
    
    return xmin,ymin

def remove_false_minima(x_axis,y_axis,input_xmin,input_ymin,limit,
                        mindist,maxdist,polyord=1,N=None,plot=True):
    '''
    DO NOT USE
    
    
    Removes false minima (or maxima) by considering the distances between
    sequential extrema and the characteristics of the data.

    Parameters
    ----------
    x_axis : array-like
        Pixel number or wavelength.
    y_axis : array-like
        Flux or S/N.
    input_xmin : list
        x-coordinates of the extremes in y_axis.
    input_ymin : list
        y-coordinates of the extremes in y_axis.
    limit : float
        largest allowed distance between sequential extremes.
    mindist : float
        smallest allowed distance between sequential extremes.
    maxdist : float
        largest allowed distance between sequential extremes.
    polyord : int, optional
        the order of the polynomial fitted through the distances between
        extremes. The default is 1.
    N : int, optional
        the window (in pixels) over which RMS and mean are calculated and
        compared. The default is approximately len(yaxis)/10, rounded to 
        the closest 100.
    plot : bool, optional
        Plots the results. The default is True.

    Returns
    -------
    xmin : list
        Cleaned x-coordinates of extremes in y_axis.
    ymin : list
        Cleaned y-coordinates of extremes in y_axis.

    '''
    
    new_xmin = input_xmin
    new_ymin = input_ymin
    outliers   = np.full_like(input_xmin,True)
    N = N if N is not None else int(round_to_closest(len(y_axis),1000)/10)
    M = int(N/10)
    mean_arrayN = running_mean(y_axis, N=N)
    mean_arrayM = running_mean(y_axis, N=M)
    rms_arrayM = running_rms(y_axis, N=M)
    y_lt_rms    = y_axis < rms_arrayM
    rms_lt_mean = rms_arrayM < mean_arrayN
    # remove 
    # cond0 = y_lt_rms[np.asarray(old_xmin[1:],dtype=np.int32)]==False
    
    j = 0
    if plot:
        fig,ax=figure(4,sharex=True,ratios=[3,1,1,1])
        ax[0].plot(x_axis,y_axis,drawstyle='steps-mid')
        ax[0].plot(x_axis,rms_arrayM,drawstyle='steps-mid',
                   label='RMS over {} pixels'.format(M))
        ax[0].plot(x_axis,mean_arrayN,drawstyle='steps-mid',
                   label='Mean over {} pixels'.format(N))
        ax[0].scatter(input_xmin,input_ymin,marker='^',c='red',s=8)
        ax[3].plot(x_axis[np.asarray(input_xmin,np.int32)],
                   rms_lt_mean[np.asarray(input_xmin,np.int32)],
                   drawstyle='steps-mid')
        ax[0].set_ylabel('Data')
        ax[1].set_ylabel('Residuals')
        ax[2].set_ylabel("Distances between\nextremes")
        ax[3].set_ylabel("RMS < Mean\nover {} pixels".format(N))
    while sum(outliers)>0 and j<50:
        print('iteration ',j, sum(outliers), len(outliers))
        old_xmin = new_xmin
        old_ymin = new_ymin
        # xpos = old_xmin
        # ypos = old_ymin
        dist = np.diff(old_xmin)
        
        keep = (dist>(mindist-2)) & (dist<(maxdist+2))
        pars = np.polyfit(old_xmin[1:][keep],dist[keep],polyord)
        model = np.polyval(pars,old_xmin[1:])
        
        resids = dist-model
        # outliers are LFC lines which satisfy:
        # 1. Distance between sequential lines is < mindist - 2 pix
        # 2. Residuals to the model going through the points are < 
        # outliers1 = np.logical_and(
                                   # (np.abs(resids)>limit), 
                                   # (dist<(mindist-2)),
                                   # (dist>(maxdist+2))
                                   # )
        # negative 
        # cond1 = dist<(mindist-2)
        # cond2 = dist>(maxdist+5)
        # cond3 = np.abs(resids)>limit
        # cond4 = y_lt_rms[np.asarray(old_xmin[1:],dtype=np.int32)]==True
        # outliers1 = np.logical_or.reduce([cond1,cond2,cond3,cond4])
        # positive
        cond1 = dist>(mindist-2)
        cond2 = dist<(maxdist+5)
        cond3 = np.abs(resids)<limit
        cond4 = y_lt_rms[np.asarray(old_xmin[1:],dtype=np.int32)]==False
        outliers1 = np.logical_and.reduce([cond1,cond2,cond3,cond4])
        # outliers1 = np.logical_or(outliers1,cond4)
        # make outliers a len(xpos) array, taking care to remove/keep the first
        # point
        insert_value = False if outliers1[0]==False else True
        print(*[len(array) for array in [cond1,cond2,cond3,outliers1]])
        outliers2 = np.insert(outliers1,0,insert_value)
        outliers = outliers2
        new_xmin = (old_xmin[~outliers])
        new_ymin = (old_ymin[~outliers])
        if plot:
            ax[0].scatter(old_xmin[outliers2],old_ymin[outliers2],marker='x',s=15,
                          c="C{}".format(j))
            ax[1].scatter(old_xmin[1:],resids,marker='o',s=3,c="C{}".format(j))
            ax[1].scatter(old_xmin[1:][outliers1],resids[outliers1],marker='x',s=15,
                          c="C{}".format(j))
            ax[1].axhline(limit,c='r',lw=2)
            ax[1].axhline(-limit,c='r',lw=2)
            ax[2].scatter(old_xmin[1:],dist,marker='o',s=3,c="C{}".format(j))
            ax[2].axhline(0.9*mindist,c='r',lw=2)
        print('END iteration ',j, sum(outliers), len(outliers))
        j+=1
        
    # good_range = y_axis > mean_arrayM
    # cond4 = good_range[np.asarray(new_xmin,dtype=np.int32)]==True
    # outliers = np.logical_and(~outliers_,cond4)
    # outliers = outliers_
    print(outliers)
    xmin, ymin = new_xmin[~outliers], new_ymin[~outliers]
    if plot:
        maxima0 = (np.roll(xmin,1)+xmin)/2
        maxima = np.array(maxima0[1:],dtype=np.int)
        [ax[0].axvline(x,ls=':',lw=0.5,c='r') for x in maxima]
        ax[0].legend()
    
    return xmin,ymin


        
        
def detect_maxmin(y_axis,x_axis=None,plot=False,*args,**kwargs):
    # check whether there is signal in the data:
    mindist,maxdist = peakdet_limits(y_axis,plot=False)
    if mindist>6 and maxdist>9:
        pass
    else:
        return None
    maxima = peakdet(y_axis,x_axis,extreme='max',*args,**kwargs)
    # define minima as points in between neighbouring maxima
    N,M = np.shape(maxima)
    minima = np.empty((N,M-1))
    minima[0] = np.asarray(maxima[0][:-1]+np.diff(maxima[0])/2,dtype=int)
    minima[1] = y_axis[minima[0].astype(int)]

    if plot:
        plt.figure()
        if x_axis is not None:
            x_axis = x_axis
        else:
            x_axis = np.arange(len(y_axis))
        plt.plot(x_axis,y_axis,drawstyle='steps-mid')
        plt.scatter(maxima[0],maxima[1],c='g',marker='^',label='Maxima')
        plt.scatter(minima[0],minima[1],c='r',marker='o',label='Minima')
        # plt.scatter(minima[0],minima[1],c='k',marker='s',label='Not used minima')
        plt.legend()
    return maxima,minima

def detect_minima(yarray,xarray=None,*args,**kwargs):
    return detect_maxmin(yarray,xarray,*args,**kwargs)[1]

def detect_maxima(yarray,xarray=None,*args,**kwargs):
    return detect_maxmin(yarray,xarray,*args,**kwargs)[0]

def peakdet(y_axis, x_axis = None, y_error = None, extreme='max',
            remove_false=True,method='peakdetect_derivatives',plot=False,
            lookahead=8,delta=0,pad_len=20,window=11,limit=None, logger=None):
    '''
    A more general function to detect minima and maxima in the data
    
    Returns a list of minima or maxima 
    
    https://gist.github.com/sixtenbe/1178136
    '''
    
    
        
    if method=='peakdetect':
        function = pkd.peakdetect
        args = (lookahead,delta)
    elif method == 'peakdetect_fft':
        function = pkd.peakdetect_fft
        args = (pad_len,)
    elif method == 'peakdetect_zero_crossing':
        function = pkd.peakdetect_zero_crossing
        args = (window,)
    elif method == 'peakdetect_derivatives':
        function = pkd.peakdetect_derivatives
        args = (window,)
    if delta == 0:
        if extreme == 'max':
            delta = np.percentile(y_axis,10)
        elif extreme == 'min':
            delta = 0
    if y_error is not None:
        assert len(y_error)==len(y_axis), "y_error not same length as y_axis"
        y_axis = y_axis / y_error
    maxima,minima = [np.array(a) for a 
                     in function(y_axis, x_axis, *args)]
    if extreme == 'max':
        data = np.transpose(maxima)
    elif extreme == 'min':
        data = np.transpose(minima)
    if remove_false:
        limit = limit if limit is not None else 2*window
        # try:
        mindist, maxdist = peakdet_limits(y_axis,plot=plot)
            # return x_axis,y_axis,data[0],data[1],limit,mindist,maxdist
        data = remove_false_maxima(x_axis,y_axis,
                data[0],data[1],limit,mindist,maxdist,plot=plot)
        # except:
            # logger = logger or logging.getLogger(__name__)
            # logger.warning("Could not remove false minima")
        
    return data

def get_time(worktime):
    """
    Returns the work time in hours, minutes, seconds

    Outputs:
    --------
           h : hour
           m : minute
           s : second
    """					
    m,s = divmod(worktime, 60)
    h,m = divmod(m, 60)
    h,m,s = [int(value) for value in (h,m,s)]
    return h,m,s        




def wrap(args):
    function, pars = args
    return function(pars)
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
#------------------------------------------------------------------------------
# 
#                           C O M B     S P E C I F I C
#
#------------------------------------------------------------------------------  


    
def noise_from_linelist(linelist):
    x = (np.sqrt(np.sum(np.power(linelist['noise']/c,-2))))
    return c/x
def remove_bad_fits(linelist,fittype,limit=None,q=None):
    """ 
    Removes lines which have uncertainties in position larger than a given 
    limit.
    """
    limit  = limit if limit is not None else 0.03
    q      = q     if q     is not None else 0.9
    
    field  = '{}_err'.format(fittype)
    values = linelist[field][:,1]
    keep   = np.where(values<=limit)[0]
    frac   = len(keep)/len(values)
    # do not remove more than q*100% of lines
    while frac<q:
        limit  += 0.001
        keep   = np.where(values<=limit)[0]
        frac   = len(keep)/len(values)
#    print(len(keep),len(linelist), "{0:5.3%} removed".format(1-frac))
    return linelist[keep]
def _get_index(centers):
    ''' Input: dataarray with fitted positions of the lines
        Output: 1d array with indices that uniquely identify every line'''
    fac = 10000
    MOD = 2.
    od = centers.od.values[:,np.newaxis]*fac
    centers_round = np.rint(centers.values/MOD)*MOD
    centers_nonan = np.nan_to_num(centers_round)
    ce = np.asarray(centers_nonan,dtype=np.int)
    index0=np.ravel(od+ce)
    mask = np.where(index0%fac==0)
    index0[mask]=999999999
    return index0
def _get_sorted(index1,index2):
    print('len indexes',len(index1),len(index2))
    # lines that are common for both spectra
    intersect=np.intersect1d(index1,index2)
    intersect=intersect[intersect>0]

    indsort=np.argsort(intersect)
    
    argsort1=np.argsort(index1)
    argsort2=np.argsort(index2)
    
    sort1 =np.searchsorted(index1[argsort1],intersect)
    sort2 =np.searchsorted(index2[argsort2],intersect)
    
    return argsort1[sort1],argsort2[sort2]
def average_line_flux(linelist,flux2d,bkg2d=None,orders=None):
    ''' 
    Returns the average line flux per line of an exposure.
    '''
    orders = orders if orders is not None else np.unique(linelist['order'])
    if bkg2d is not None:
        totflux = np.sum(flux2d[orders]-bkg2d[orders])
    else:
        totflux = np.sum(flux2d[orders])
    ll     = container.Generic(linelist)
    nlines = len(ll[orders])
    return totflux/nlines
def make_comb_interpolation(lines_LFC1, lines_LFC2,ftype='gauss'):
    ''' Routine to use the known frequencies and positions of a comb, 
        and its repetition and offset frequencies to build a frequency
        solution by linearly interpolating between individual lines
        
        Arguments must be for the same fibre!
        
        LFC1: known
        LFC2: to be interpolated
        
        Args:
        -----
            lines_LFC1 : lines xarray Dataset for a single exposure
            lines_LFC2 : lines xarray Dataset for a single exposure
    ''' 
    
    freq_LFC1 = lines_LFC1['freq']
    freq_LFC2 = lines_LFC2['freq']

    pos_LFC1  = lines_LFC1[ftype][:,1]
    pos_LFC2  = lines_LFC2['bary']
    #plt.figure(figsize=(12,6))
    minord  = np.max(tuple(np.min(f['order']) for f in [lines_LFC1,lines_LFC2]))
    maxord  = np.min(tuple(np.max(f['order']) for f in [lines_LFC1,lines_LFC2]))
    interpolated = {}
    for od in np.arange(minord,maxord):
        print("Order {0:=>30d}".format(od))
        # get fitted positions of LFC1 and LFC2 lines
        inord1 = np.where(lines_LFC1['order']==od)
        inord2 = np.where(lines_LFC2['order']==od)
        cen1   = pos_LFC1[inord1]
        cen2   = pos_LFC2[inord2]
        x1, x2 = (np.sort(x) for x in [cen1,cen2])
        # find the closest LFC1 line to each LFC2 line in this order 
        freq1  = freq_LFC1[inord1]
        freq2  = freq_LFC2[inord2]
        f1, f2 = (np.sort(f)[::-1] for f in [freq1,freq2])
        vals, bins = f2, f1
        right = np.digitize(vals,bins,right=False)
        print(right)
        fig, ax = figure(2,sharex=True,ratios=[3,1])
        ax[0].set_title("Order = {0:2d}".format(od))
        ax[0].scatter(f1,x1,c="C0",label='LFC1')
        ax[0].scatter(f2,x2,c="C1",label='LFC2')
        interpolated_LFC2 = []
        for x_LFC2,f_LFC2,index_LFC1 in zip(x2,f2,right):
            if index_LFC1 == 0 or index_LFC1>len(bins)-1:
                interpolated_LFC2.append(np.nan)
                continue
            else:
                pass
            
            f_left  = f1[index_LFC1-1]
            x_left  = x1[index_LFC1-1]
            f_right = f1[index_LFC1]
            x_right = x1[index_LFC1]
#            if x_LFC2 > x_right:
#                interpolated_LFC2.append(np.nan)
#                continue
#            else:
#                pass
            
            # fit linear function 
            fitpars = np.polyfit(x=[f_left,f_right],
                                 y=[x_left,x_right],deg=1)
            ax[0].scatter([f_left,f_right],[x_left,x_right],c='C0',marker='x',s=4)
            fspace = np.linspace(f_left,f_right,10)
            xspace = np.linspace(x_left,x_right,10)
            ax[0].plot(fspace,xspace,c='C0')
            x_int   = np.polyval(fitpars,f_LFC2)
            interpolated_LFC2.append(x_int)
            ax[1].scatter([f_LFC2],[(x_LFC2-x_int)*829],c='C1',marker='x',s=4)
            print("{:>3d}".format(index_LFC1),
                  (3*("{:>14.5f}")).format(x_left,x_LFC2,x_right),
                  "x_int = {:>10.5f}".format(x_int),
                  "RV = {:>10.5f}".format((x_LFC2-x_int)*829))
            #print(x_LFC2,interpolated_x)
        interpolated[od] = interpolated_LFC2
        #[plt.axvline(x1,ls='-',c='r') for x1 in f1]
        #[plt.axvline(x2,ls=':',c='g') for x2 in f2]
        break
    return interpolated
def get_comb_offset(source_anchor,source_offset,source_reprate,modefilter):
    m,k     = divmod(round((source_anchor-source_offset)/source_reprate),
                                   modefilter)
    comb_offset = (k-1)*source_reprate + source_offset #+ anchor_offset
#    f0_comb = k*source_reprate + source_offset 
    return comb_offset
#------------------------------------------------------------------------------
#
#                           P R O G R E S S   B A R 
#
#------------------------------------------------------------------------------
def update_progress(progress,name=None,logger=None):
    # https://stackoverflow.com/questions/3160699/python-progress-bar
    barLength = 40 
    status = ""
    name = name if name is not None else ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt\r\n"
    if progress >= 1:
        progress = 1
        status = "Done\r\n"
    block = int(round(barLength*progress))
    mess  = (name,"#"*block + "-"*(barLength-block), progress*100, status)
    text = "\rProgress [{0}]: [{1}] {2:8.3f}% {3}".format(*mess)
    if logger is not None:
        logger.info(text)
    else:
        sys.stdout.write(text)
        sys.stdout.flush()
#------------------------------------------------------------------------------
#
#                        P H O T O N     N O I S E
#
#------------------------------------------------------------------------------   
def get_background1d(data,kind="linear",*args):
    """
    Returns the background in the echelle order. 
    Default linear interpolation between line minima.
    """
    yarray = np.atleast_1d(data)
    assert len(np.shape(yarray))==1, "Data is not 1-dimensional"
    xarray = np.arange(np.size(data))
    bkg    = get_background(xarray,yarray)
    return bkg
    
def get_background2d(data,kind="linear", *args):
    """
    Returns the background for all echelle orders in the spectrum.
    Default linear interpolation between line minima.
    """
    orders     = np.shape(data)[0]
    background = np.array([get_background1d(data[o],kind) \
                           for o in range(orders)])
    return background

def get_background(xarray,yarray,kind='linear'):
    """
    Returns the interpolated background between minima in yarray.
    
    Smooths the spectrum using Wiener filtering to detect true minima.
    See peakdetect.py for more information
    """
    from scipy import interpolate
    xbkg,ybkg = peakdet(yarray, xarray, extreme="min")
    if   kind == "spline":
        intfunc = interpolate.splrep(xbkg, ybkg)
        bkg     = interpolate.splev(xarray,intfunc) 
    elif kind == "linear":
        intfunc = interpolate.interp1d(xbkg,ybkg,
                                       bounds_error=False,
                                       fill_value=0)
        bkg = intfunc(xarray)
    return bkg

def get_error2d(data2d):
    assert len(np.shape(data2d))==2, "Data is not 2-dimensional"
    data2d  = np.abs(data2d)
    bkg2d   = get_background2d(data2d)
    error2d = np.sqrt(np.abs(data2d) + np.abs(bkg2d))
    return error2d
    
def get_error1d(data1d,*args):
    data1d  = np.abs(data1d)
    bkg1d   = np.abs(get_background1d(data1d,*args))
    error1d = np.sqrt(data1d + bkg1d)
    return error1d
def sigmav(data2d,wavesol2d):
    """
    Calculates the limiting velocity precison of all pixels in the spectrum
    using ThAr wavelengths.
    """
    orders  = np.arange(np.shape(data2d)[0])
    sigma_v = np.array([sigmav1d(data2d[order],wavesol2d[order]) \
                        for order in orders])
    return sigma_v

def sigmav1d(data1d,wavesol1d):
    """
    Calculates the limiting velocity precison of all pixels in the order
    using ThAr wavelengths.
    """
    # weights for photon noise calculation
    # Equation 5 in Murphy et al. 2007, MNRAS 380 p839
    #pix2d   = np.vstack([np.arange(spec.npix) for o in range(spec.nbo)])
    error1d = get_error1d(data1d)
    df_dlbd = derivative1d(data1d,wavesol1d)
    sigma_v = c*error1d/(wavesol1d*df_dlbd)
    return sigma_v

# =============================================================================
    
#                    S P E C T R U M     H E L P E R
#                          F U N C T I O N S
    
# =============================================================================
optordsA   = [161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149,
   148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136,
   135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123,
   122, 121, 120, 119, 118, 117, 116, 114, 113, 112, 111, 110, 109,
   108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98,  97,  96,
    95,  94,  93,  92,  91,  90,  89]
optordsB   = [161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149,
   148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136,
   135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123,
   122, 121, 120, 119, 118, 117,      114, 113, 112, 111, 110, 109,
   108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98,  97,  96,
    95,  94,  93,  92,  91,  90,  89]

def prepare_orders(order,nbo,sOrder,eOrder):
    '''
    Returns an array or a list containing the input orders.
    '''
    orders = np.arange(nbo)
    select = slice(sOrder,eOrder,1)
    
    if isinstance(order,list):
        return orders[order]
    elif order is not None:
        select = prepare_slice(order,nbo,sOrder)
    return orders[select]
def prepare_slice(order,nbo,sOrder):
    import numbers
    if isinstance(order,numbers.Integral):
        start = order
        stop = order+1
        step = 1
    elif isinstance(order,tuple):
        range_sent = True
        numitems = np.shape(order)[0]
        if numitems==3:
            start, stop, step = order
        elif numitems==2:
            start, stop = order
            step = 1
        elif numitems==1:
            start = order
            stop  = order+1
            step  = 1
    else:
        start = sOrder
        stop  = nbo
        step  = 1
    return slice(start,stop,step)
#def ord2optord(order,fibre):
#        optord = np.arange(88+self.nbo,88,-1)
#        # fibre A doesn't contain order 115
#        if self.meta['fibre'] == 'A':
#            shift = 1
#        # fibre B doesn't contain orders 115 and 116
#        elif self.meta['fibre'] == 'B':
#            shift = 2
#        cut=np.where(optord>114)
#        optord[cut]=optord[cut]+shift
#        
#        return optord