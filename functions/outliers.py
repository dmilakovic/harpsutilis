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
    rmean = math.running_mean(points,window)
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