#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:33:47 2023

@author: dmilakov
"""
import numpy as np
import jax.numpy as jnp
import harps.functions.math as math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# 
#                              O U T L I E R S
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
    median = jnp.median(points, axis=0)
    diff = jnp.sum((points - median)**2, axis=-1)
    diff = jnp.sqrt(diff)
    med_abs_deviation = jnp.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def is_outlier_running(points, window=5,thresh=3.5,plot=False):
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
    
    # rmean = math.running_mean(points,window)
    rmean = running_mean_exclude_center(points,window)
    # rmean = running_rms(points,window)
#    rmean = np.percentile(points,85)
    if len(points.shape) == 1:
        points = points[:,None]  
    diff  = np.sum((points-rmean)**2,axis=-1)
    diff  = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    if plot:
        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(points,label='data')
        ax1.plot(rmean,label='rmean')
        ax2.plot(modified_z_score,label='z_score')
        ax2.axhline(thresh,ls='--')
        mask = modified_z_score > thresh
        ax1.plot(np.arange(len(points))[mask],
                 points[mask],ls='',marker='x',color='C1',ms=8)
        ax1.legend()
        plt.show()
    return modified_z_score > thresh

def is_outlier_bins(points,idx,thresh=3.5):
    outliers = np.zeros_like(points)
    for i in np.unique(idx):
        cut = np.where(idx==i)[0]
        outliers[cut] = is_outlier(points[cut],thresh=thresh)
        # print(is_outlier(points[cut],thresh=thresh))
    return outliers.astype(bool)

def is_outlier_from_linear(xvals,yvals,idx,yerrs=None,thresh=3.5):
    def func(x,a,b):
        return a*x + b
    outliers = np.zeros_like(yvals)
    for i in np.unique(idx):
        cut = np.where(idx==i)[0]
        sigma = yerrs[cut] if yerrs is not None else None
        pars,pcov = curve_fit(func, 
                              ydata=yvals[cut],
                              xdata=xvals[cut],
                              p0=(1,0),
                              sigma=sigma)
        resids = yvals[cut] - func(xvals[cut],*pars)
        check_array = resids/sigma if sigma is not None else resids
        outliers[cut] = is_outlier(check_array,thresh=thresh)
        # print(is_outlier(points[cut],thresh=thresh))
    return outliers.astype(bool)
        
def is_outlier_from_poly(xvals,yvals,yerrs=None,deg=1,thresh=3.5):
    from functools import partial
    from numpy.polynomial import Polynomial
    # func = partial(np.polyval, deg=deg)
    outliers = np.zeros_like(yvals)
    sigma = yerrs if yerrs is not None else np.ones_like(yvals)
    # pars,pcov = curve_fit(func, 
    #                       ydata=yvals[cut],
    #                       xdata=xvals[cut],
    #                       p0=(1,0),
    #                       sigma=sigma)
    poly = Polynomial.fit(x = xvals,
                          y = yvals,
                          deg = deg,
                          w = 1./sigma)
    resids = yvals - poly(xvals)
    check_array = resids/sigma if sigma is not None else resids
    outliers = is_outlier(check_array,thresh=thresh)
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

def running_mean_exclude_center(array, window_size):
    """
    Calculate the running mean of an array with a specified window size,
    excluding the central point of the window.

    Parameters:
    - array (numpy.ndarray): Input array.
    - window_size (int): Size of the window for calculating the running mean.

    Returns:
    - numpy.ndarray: Array containing the running mean values.
    """
    # Validate window size
    if window_size <= 0 or window_size >= len(array):
        raise ValueError("Window size must be a positive integer less than the length of the array.")

    # Create a copy of the input array to store the running mean values
    running_means = np.zeros_like(array, dtype=float)

    # Calculate the running mean for each element in the array
    for i in range(len(array)):
        # Determine the window boundaries
        start = max(0, i - window_size // 2)
        end = min(len(array), i + window_size // 2 + 1)

        # Exclude the central point of the window
        if window_size % 2 == 0 and start <= i < end:
            # Shift the window to exclude the central point
            end -= 1

        # Calculate the mean of the window excluding the central point
        if start < end:
            window_values = array[start:end]
            window_values = np.delete(window_values, window_size // 2)  # Exclude central point
            running_means[i] = np.mean(window_values)

    return running_means