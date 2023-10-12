#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:27:52 2023

@author: dmilakov
"""
import numpy as np
import math as mth
import scipy.interpolate as interpolate
from scipy.optimize import minimize, leastsq, curve_fit, brentq
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

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
    x = x if x is not None else np.arange(len(y))
    if method=='forward':
        dx = np.diff(x)
        # dx  = np.append(dx_,np.zeros(order))
        dy = np.diff(y,order)
        # dy  = np.append(dy_,np.zeros(order))
        if order==2:
            d   = dy/dx[:-1]
        else:
            d = dy/dx
        d = np.append(d,np.zeros(order))
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
    deriv=derivative(y_array,x_array,order=1,accuracy=8)
    srep =interpolate.splrep(x_array,deriv)
    return interpolate.splev(x,srep) 

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

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))
def integer_slice(i, n, m):
    # return nth to mth digit of i (as int)
    l = mth.floor(mth.log10(i)) + 1
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
    errors = np.atleast_1d(errors) if errors is not None else np.ones_like(values)
    variance = np.power(errors,2)
    weights  = 1./variance 
    mean  = np.nansum(values * weights) / np.nansum(weights)
    sigma = 1./ np.sqrt(np.sum(weights))
    return mean,sigma
    # return average(values,errors)
def aicc(chisq,n,p):
    ''' Returns the Akiake information criterion value 
    chisq = chi square
    n     = number of points
    p     = number of free parameters
    
    '''
    return chisq + 2*p + 2*p*(p+1)/(n-p-1)