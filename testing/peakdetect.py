#!/usr/bin/python2


# Copyright (C) 2016 Sixten Bergman
# License WTFPL
#
# This program is free software. It comes without any warranty, to the extent
# permitted by applicable law. 
# You can redistribute it and/or modify it under the terms of the Do What The
# Fuck You Want To Public License, Version 2, as published by Sam Hocevar. See
# http://www.wtfpl.net/ for more details.
#
# note that the function peakdetect is derived from code which was released to
# public domain see: http://billauer.co.il/peakdet.html
#

import logging
from math import pi, log
import numpy as np
import pylab
import matplotlib.pyplot as plt
import harps.functions.math as mathfunc
from scipy import fft, ifft
from scipy.optimize import curve_fit
from scipy.signal import cspline1d_eval, cspline1d, wiener, nuttall, welch

from sklearn.preprocessing import MinMaxScaler # Switched scaler
from sklearn.cluster import DBSCAN
from sklearn.linear_model import TheilSenRegressor # Robust regressor for spacing trend
from collections import Counter # For finding the main cluster


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


__all__ = [
        "peakdetect",
        "peakdetect_derivatives",
        "peakdetect_fft",
        "peakdetect_parabola",
        "peakdetect_sine",
        "peakdetect_sine_locked",
        "peakdetect_spline",
        "peakdetect_zero_crossing",
        "zero_crossings",
        "zero_crossings_sine_fit"
        ]

def _derivative1d(y_axis, x_axis):
    ''' Get the first derivative of a 1d array y. 
    Modified from 
    http://stackoverflow.com/questions/18498457/
    numpy-gradient-function-and-numerical-derivatives'''
    z1  = np.hstack((y_axis[0], y_axis[:-1]))
    z2  = np.hstack((y_axis[1:], y_axis[-1]))
    dx1 = np.hstack((0, np.diff(x_axis)))
    dx2 = np.hstack((np.diff(x_axis), 0))  
#    if np.all(np.asarray(dx1+dx2)==0):
#        dx1 = dx2 = np.ones_like(x_axis)/2
    d   = (z2-z1) / (dx2+dx1)
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
    x_axis = x_axis if x_axis is not None else np.arange(len(y_axis))
    return (-1)**order*_derivative(y_axis,x_axis,_coeffs[accuracy])


def _derivative(y_axis,x_axis,coeffs):    
    N        = len(y_axis)
    pad_width = int(len(coeffs)//2)
    y_padded = np.pad(y_axis,pad_width,mode='symmetric')
    x_padded = np.pad(x_axis,pad_width,mode='linear_ramp',
                      end_values=(-pad_width,N+pad_width-1))
    xcubed   = np.power(np.diff(x_padded),3)
    h        = np.insert(xcubed,0,xcubed[0])
    y_deriv  = np.convolve(y_padded, coeffs, 'same')/h
    
    return y_deriv[pad_width:-pad_width]
    

def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = np.arange(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")
    
    # cut = np.where(y_axis!=0.0)[0]
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis
    

def _pad(fft_data, pad_len):
    """
    Pads fft data to interpolate in time domain
    
    keyword arguments:
    fft_data -- the fft
    pad_len --  By how many times the time resolution should be increased by
    
    return: padded list
    """
    l = len(fft_data)
    n = _n(l * pad_len)
    fft_data = list(fft_data)
    
    return fft_data[:l // 2] + [0] * (2**n-l) + fft_data[l // 2:]
    
def _n(x):
    """
    Find the smallest value for n, which fulfils 2**n >= x
    
    keyword arguments:
    x -- the value, which 2**n must surpass
    
    return: the integer n
    """
    return int(log(x)/log(2)) + 1
    
    
def _peakdetect_parabola_fitter(raw_peaks, x_axis, y_axis, points):
    """
    Performs the actual parabola fitting for the peakdetect_parabola function.
        
    keyword arguments:
    raw_peaks -- A list of either the maxima or the minima peaks, as given
        by the peakdetect functions, with index used as x-axis
    
    x_axis -- A numpy array of all the x values
    
    y_axis -- A numpy array of all the y values
    
    points -- How many points around the peak should be used during curve
        fitting, must be odd.
    
    
    return: A list giving all the peaks and the fitted waveform, format:
        [[x, y, [fitted_x, fitted_y]]]
        
    """
    func = lambda x, a, tau, c: a * ((x - tau) ** 2) + c
    fitted_peaks = []
    distance = abs(x_axis[raw_peaks[1][0]] - x_axis[raw_peaks[0][0]]) / 4
    for peak in raw_peaks:
        index = peak[0]
        x_data = x_axis[index - points // 2: index + points // 2 + 1]
        y_data = y_axis[index - points // 2: index + points // 2 + 1]
        # get a first approximation of tau (peak position in time)
        tau = x_axis[index]
        # get a first approximation of peak amplitude
        c = peak[1]
        a = np.sign(c) * (-1) * (np.sqrt(abs(c))/distance)**2
        """Derived from ABC formula to result in a solution where A=(rot(c)/t)**2"""
        
        # build list of approximations
        
        p0 = (a, tau, c)
        popt, pcov = curve_fit(func, x_data, y_data, p0)
        # retrieve tau and c i.e x and y value of peak
        x, y = popt[1:3]
        
        # create a high resolution data set for the fitted waveform
        x2 = np.linspace(x_data[0], x_data[-1], points * 10)
        y2 = func(x2, *popt)
        
        fitted_peaks.append([x, y, [x2, y2]])
        
    return fitted_peaks
    
    
def peakdetect_parabole(*args, **kwargs):
    """
    Misspelling of peakdetect_parabola
    function is deprecated please use peakdetect_parabola
    """
    logging.warn("peakdetect_parabole is deprecated due to misspelling use: peakdetect_parabola")
    
    return peakdetect_parabola(*args, **kwargs)
    
    
#def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0.1):
def peakdetect(y_axis, x_axis = None, lookahead = 3, delta=10):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)
    
    lookahead -- distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200) 
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    
    delta -- this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]
    
    
def peakdetect_fft(y_axis, x_axis, pad_len = 20):
    """
    Performs a FFT calculation on the data and zero-pads the results to
    increase the time domain resolution after performing the inverse fft and
    send the data to the 'peakdetect' function for peak 
    detection.
    
    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as the index 50.234 or similar.
    
    Will find at least 1 less peak then the 'peakdetect_zero_crossing'
    function, but should result in a more precise value of the peak as
    resolution has been increased. Some peaks are lost in an attempt to
    minimize spectral leakage by calculating the fft between two zero
    crossings for n amount of signal periods.
    
    The biggest time eater in this function is the ifft and thereafter it's
    the 'peakdetect' function which takes only half the time of the ifft.
    Speed improvements could include to check if 2**n points could be used for
    fft and ifft or change the 'peakdetect' to the 'peakdetect_zero_crossing',
    which is maybe 10 times faster than 'peakdetct'. The pro of 'peakdetect'
    is that it results in one less lost peak. It should also be noted that the
    time used by the ifft function can change greatly depending on the input.
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
    
    pad_len -- By how many times the time resolution should be
        increased by, e.g. 1 doubles the resolution. The amount is rounded up
        to the nearest 2**n amount
        (default: 20)
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    zero_indices = zero_crossings(y_axis, window_len = 11)
    #select a n amount of periods
    last_indice = - 1 - (1 - len(zero_indices) & 1)
    ###
    # Calculate the fft between the first and last zero crossing
    # this method could be ignored if the beginning and the end of the signal
    # are unnecessary as any errors induced from not using whole periods
    # should mainly manifest in the beginning and the end of the signal, but
    # not in the rest of the signal
    # this is also unnecessary if the given data is an amount of whole periods
    ###
    fft_data = fft(y_axis[zero_indices[0]:zero_indices[last_indice]])
    padd = lambda x, c: x[:len(x) // 2] + [0] * c + x[len(x) // 2:]
    n = lambda x: int(log(x)/log(2)) + 1
    # pads to 2**n amount of samples
    fft_padded = padd(list(fft_data), 2 ** 
                n(len(fft_data) * pad_len) - len(fft_data))
    
    # There is amplitude decrease directly proportional to the sample increase
    sf = len(fft_padded) / float(len(fft_data))
    # There might be a leakage giving the result an imaginary component
    # Return only the real component
    y_axis_ifft = ifft(fft_padded).real * sf #(pad_len + 1)
    x_axis_ifft = np.linspace(
                x_axis[zero_indices[0]], x_axis[zero_indices[last_indice]],
                len(y_axis_ifft))
    # get the peaks to the interpolated waveform
    max_peaks, min_peaks = peakdetect(y_axis_ifft, x_axis_ifft, 500,
                                    delta = abs(np.diff(y_axis).max() * 2))
    #max_peaks, min_peaks = peakdetect_zero_crossing(y_axis_ifft, x_axis_ifft)
    
    # store one 20th of a period as waveform data
    data_len = int(np.diff(zero_indices).mean()) / 10
    data_len += 1 - data_len & 1
    
    
    return [max_peaks, min_peaks]
    
    
def peakdetect_parabola(y_axis, x_axis, points = 31):
    """
    Function for detecting local maxima and minima in a signal.
    Discovers peaks by fitting the model function: y = k (x - tau) ** 2 + m
    to the peaks. The amount of points used in the fitting is set by the
    points argument.
    
    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly, if it was returned as index 50.234 or similar.
    
    will find the same amount of peaks as the 'peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
    
    points -- How many points around the peak should be used during curve
        fitting (default: 31)
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # make the points argument odd
    points += 1 - points % 2
    #points += 1 - int(points) & 1 slower when int conversion needed
    
    # get raw peaks
    max_raw, min_raw = peakdetect_zero_crossing(y_axis)
    
    # define output variable
    max_peaks = []
    min_peaks = []
    
    max_ = _peakdetect_parabola_fitter(max_raw, x_axis, y_axis, points)
    min_ = _peakdetect_parabola_fitter(min_raw, x_axis, y_axis, points)
    
    max_peaks = map(lambda x: [x[0], x[1]], max_)
    max_fitted = map(lambda x: x[-1], max_)
    min_peaks = map(lambda x: [x[0], x[1]], min_)
    min_fitted = map(lambda x: x[-1], min_)
    
    return [max_peaks, min_peaks]
    

def peakdetect_sine(y_axis, x_axis, points = 31, lock_frequency = False):
    """
    Function for detecting local maxima and minima in a signal.
    Discovers peaks by fitting the model function:
    y = A * sin(2 * pi * f * (x - tau)) to the peaks. The amount of points used
    in the fitting is set by the points argument.
    
    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as index 50.234 or similar.
    
    will find the same amount of peaks as the 'peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.
    
    The function might have some problems if the sine wave has a
    non-negligible total angle i.e. a k*x component, as this messes with the
    internal offset calculation of the peaks, might be fixed by fitting a 
    y = k * x + m function to the peaks for offset calculation.
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
    
    points -- How many points around the peak should be used during curve
        fitting (default: 31)
    
    lock_frequency -- Specifies if the frequency argument of the model
        function should be locked to the value calculated from the raw peaks
        or if optimization process may tinker with it.
        (default: False)
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # make the points argument odd
    points += 1 - points % 2
    #points += 1 - int(points) & 1 slower when int conversion needed
    
    # get raw peaks
    max_raw, min_raw = peakdetect_zero_crossing(y_axis)
    
    # define output variable
    max_peaks = []
    min_peaks = []
    
    # get global offset
    offset = np.mean([np.mean(max_raw, 0)[1], np.mean(min_raw, 0)[1]])
    # fitting a k * x + m function to the peaks might be better
    #offset_func = lambda x, k, m: k * x + m
    
    # calculate an approximate frequency of the signal
    Hz_h_peak = np.diff(zip(*max_raw)[0]).mean()
    Hz_l_peak = np.diff(zip(*min_raw)[0]).mean()
    Hz = 1 / np.mean([Hz_h_peak, Hz_l_peak])
    
    
    
    # model function
    # if cosine is used then tau could equal the x position of the peak
    # if sine were to be used then tau would be the first zero crossing
    if lock_frequency:
        func = lambda x_ax, A, tau: A * np.sin(
            2 * pi * Hz * (x_ax - tau) + pi / 2)
    else:
        func = lambda x_ax, A, Hz, tau: A * np.sin(
            2 * pi * Hz * (x_ax - tau) + pi / 2)
    #func = lambda x_ax, A, Hz, tau: A * np.cos(2 * pi * Hz * (x_ax - tau))
    
    
    #get peaks
    fitted_peaks = []
    for raw_peaks in [max_raw, min_raw]:
        peak_data = []
        for peak in raw_peaks:
            index = peak[0]
            x_data = x_axis[index - points // 2: index + points // 2 + 1]
            y_data = y_axis[index - points // 2: index + points // 2 + 1]
            # get a first approximation of tau (peak position in time)
            tau = x_axis[index]
            # get a first approximation of peak amplitude
            A = peak[1]
            
            # build list of approximations
            if lock_frequency:
                p0 = (A, tau)
            else:
                p0 = (A, Hz, tau)
            
            # subtract offset from wave-shape
            y_data -= offset
            popt, pcov = curve_fit(func, x_data, y_data, p0)
            # retrieve tau and A i.e x and y value of peak
            x = popt[-1]
            y = popt[0]
            
            # create a high resolution data set for the fitted waveform
            x2 = np.linspace(x_data[0], x_data[-1], points * 10)
            y2 = func(x2, *popt)
            
            # add the offset to the results
            y += offset
            y2 += offset
            y_data += offset
            
            peak_data.append([x, y, [x2, y2]])
       
        fitted_peaks.append(peak_data)
    
    # structure date for output
    max_peaks = map(lambda x: [x[0], x[1]], fitted_peaks[0])
    max_fitted = map(lambda x: x[-1], fitted_peaks[0])
    min_peaks = map(lambda x: [x[0], x[1]], fitted_peaks[1])
    min_fitted = map(lambda x: x[-1], fitted_peaks[1])
    
    
    return [max_peaks, min_peaks]

    
def peakdetect_sine_locked(y_axis, x_axis, points = 31):
    """
    Convenience function for calling the 'peakdetect_sine' function with
    the lock_frequency argument as True.
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
    points -- How many points around the peak should be used during curve
        fitting (default: 31)
    
    return: see the function 'peakdetect_sine'
    """
    return peakdetect_sine(y_axis, x_axis, points, True)
    
    
def peakdetect_spline(y_axis, x_axis, pad_len=20):
    """
    Performs a b-spline interpolation on the data to increase resolution and
    send the data to the 'peakdetect_zero_crossing' function for peak 
    detection.
    
    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as the index 50.234 or similar.
    
    will find the same amount of peaks as the 'peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. 
        x-axis must be equally spaced.
    
    pad_len -- By how many times the time resolution should be increased by,
        e.g. 1 doubles the resolution.
        (default: 20)
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # could perform a check if x_axis is equally spaced
    #if np.std(np.diff(x_axis)) > 1e-15: raise ValueError
    # perform spline interpolations
    dx = x_axis[1] - x_axis[0]
    x_interpolated = np.linspace(x_axis.min(), x_axis.max(), len(x_axis) * (pad_len + 1))
    cj = cspline1d(y_axis)
    y_interpolated = cspline1d_eval(cj, x_interpolated, dx=dx,x0=x_axis[0])
    # get peaks
    max_peaks, min_peaks = peakdetect_zero_crossing(y_interpolated, x_interpolated)
    
    return [max_peaks, min_peaks]
    
def peakdetect_zero_crossing(y_axis, x_axis = None, window = 11):
    """
    Function for detecting local maxima and minima in a signal.
    Discovers peaks by dividing the signal into bins and retrieving the
    maximum and minimum value of each the even and odd bins respectively.
    Division into bins is performed by smoothing the curve and finding the
    zero crossings.
    
    Suitable for repeatable signals, where some noise is tolerated. Executes
    faster than 'peakdetect', although this function will break if the offset
    of the signal is too large. It should also be noted that the first and
    last peak will probably not be found, as this function only can find peaks
    between the first and last zero crossing.
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the position of the peaks. If
        omitted an index of the y_axis is used.
        (default: None)
    
    window -- the dimension of the smoothing window; should be an odd integer
        (default: 11)
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    
    zero_indices = zero_crossings(y_axis, window_len = window)
    period_lengths = np.diff(zero_indices)
            
    bins_y = [y_axis[index:index + diff] for index, diff in 
        zip(zero_indices, period_lengths)]
    bins_x = [x_axis[index:index + diff] for index, diff in 
        zip(zero_indices, period_lengths)]
        
    even_bins_y = bins_y[::2]
    odd_bins_y = bins_y[1::2]
    even_bins_x = bins_x[::2]
    odd_bins_x = bins_x[1::2]
    hi_peaks_x = []
    lo_peaks_x = []
    
    #check if even bin contains maxima
    if abs(even_bins_y[0].max()) > abs(even_bins_y[0].min()):
        hi_peaks = [bin.max() for bin in even_bins_y]
        lo_peaks = [bin.min() for bin in odd_bins_y]
        # get x values for peak
        for bin_x, bin_y, peak in zip(even_bins_x, even_bins_y, hi_peaks):
            hi_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
        for bin_x, bin_y, peak in zip(odd_bins_x, odd_bins_y, lo_peaks):
            lo_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
    else:
        hi_peaks = [bin.max() for bin in odd_bins_y]
        lo_peaks = [bin.min() for bin in even_bins_y]
        # get x values for peak
        for bin_x, bin_y, peak in zip(odd_bins_x, odd_bins_y, hi_peaks):
            hi_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
        for bin_x, bin_y, peak in zip(even_bins_x, even_bins_y, lo_peaks):
            lo_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
    
    max_peaks = [[x, y] for x,y in zip(hi_peaks_x, hi_peaks)]
    min_peaks = [[x, y] for x,y in zip(lo_peaks_x, lo_peaks)]
    
    return [max_peaks, min_peaks]
def peakdetect_derivatives(y_axis, x_axis = None, window_len=None, 
                           super_sample_factor=10,plot=False, deriv_method='coeff',
                           validate=False): 
    """
    Function for detecting extrema in the signal by smoothing the input data
    by a Wiener filter of given window length, then identifying extrema in 
    the signal by looking for sign change in the first derivative of the
    smoothed signal. Maxima are identified by a negative second derivative, 
    whereas minima by a positive one. 
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the position of the peaks. If
        omitted an index of the y_axis is used.
        (default: None)
    
    window_len -- the dimension of the smoothing window; should be an odd 
        integer (default: 3)
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    bins = super_sample_factor
    y_rebinned   = _rebin(y_axis,newshape=(bins*len(y_axis),))
    # xmax = np.max(x_axis); xmin = np.min(x_axis)
    # x_rebinned    = np.linspace(xmin,xmax,len(y_rebinned))/bins - 0.5
    x_rebinned    = np.arange(len(y_rebinned))/bins - 0.5 + np.min(x_axis)
    
    factor = bins/2
    if window_len is not None:
        window_len_ = window_len 
    else:
        window_len_ = get_window(y_axis,plot=False) 
    
    window_len = int(window_len_ * factor)
    mindist = int(window_len_ / 2)
    print(mindist, window_len)
        
    y_filtered_   = _smooth(y_rebinned,window_len,window='nuttall',mode='same')
    y_filtered    = y_filtered_[window_len-1:-(window_len-1)]
    derivative1st = mathfunc.derivative1d(y_filtered,x=None,order=1,
                                          method=deriv_method)
    derivative2nd = mathfunc.derivative1d(y_filtered,x=None,order=2,
                                          method=deriv_method)
    # derivative1st = derivative(y_filtered,order=1,accuracy=4)
    # derivative2nd = derivative(y_filtered,order=2,accuracy=8)
    # extremes = indices where the sign of the derivative changes
    # # indices where the inflection changes 
    # crossings = indices BEFORE sign change
    sign_change = np.diff(np.sign(derivative1st))
    crossings_  = np.where(np.abs(sign_change)==2)[0]
    inside      = np.logical_and((crossings_ >= 0),
                                (crossings_ <= len(y_filtered)-1)
                                )
    crossings  = crossings_[inside]
    
    # compare two points around a crossing and save the one whose 
    # 1st derivative is closer to zero
    extrema    = np.zeros_like(crossings)
    for i,idx in enumerate(crossings):
        left = np.abs(derivative1st[idx])
        right = np.abs(derivative1st[idx+1])
        # left = y_filtered[idx]
        # right = y_filtered[idx+1]
        if left<right:
            extrema[i]=idx
        else:
            extrema[i]=idx+1
        # print(i,left,right,extrema[i])
            
    max_ind = extrema[np.where(derivative2nd[extrema]<0)]
    min_ind = extrema[np.where(derivative2nd[extrema]>0)]
    

    max_peaks_rebinned = [[x,y] for x,y in
                          zip(x_rebinned[max_ind],y_filtered[max_ind])]
    min_peaks_rebinned = [[x,y] for x,y in 
                          zip(x_rebinned[min_ind],y_filtered[min_ind])]
    
    max_peaks_ = [[np.round(x).astype(int),np.round(y).astype(int)] 
                 for x,y in 
                 max_peaks_rebinned if (np.round(x)>=0 and np.round(x)<len(y_axis))]
    min_peaks_ = [[np.round(x).astype(int),np.round(y).astype(int)] 
                 for x,y in 
                 min_peaks_rebinned if (np.round(x)>=0 and np.round(x)<len(y_axis))]
    # max_peaks = _validate(y_axis,max_peaks_,kind='max',mindist=mindist)
    # min_peaks = _validate(y_axis,min_peaks_,kind='min',mindist=mindist,
    #                       verbose=False)
    if validate:
        maxima = _validate(y_axis,max_peaks_rebinned,
                              kind='max',mindist=mindist)
        minima = _validate(y_axis,min_peaks_rebinned,
                              kind='min',mindist=mindist,
                              verbose=False)
    else:
        maxima = np.array(max_peaks_rebinned)
        minima = np.array(min_peaks_rebinned)
    
    if plot:
        nrows = 5
        fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows,1,sharex=True,
                                                 figsize=(6,3+nrows))
        
        for ax,label in zip([ax1,ax2,ax3,ax4,ax5],
                            ['Data',
                             '1st derivative',
                             '2nd derivative',
                             'Distance in x',
                             'Distance in y']):
            ax.set_ylabel(label)
        ax1.plot(x_axis,y_axis,drawstyle='steps-mid')
        ax1.plot(x_rebinned,y_filtered,drawstyle='steps-mid')
        ax2.plot(x_rebinned,derivative1st,drawstyle='steps-mid')
        ax3.plot(x_rebinned,derivative2nd,drawstyle='steps-mid')
        for ax,array in zip([#ax1,
                             ax2,ax3],
                            [#y_rebinned,
                             derivative1st,
                             derivative2nd]):
            ax.scatter(x_rebinned[max_ind],
                       array[max_ind],s=20,marker='^',c='C4',zorder=11)
            ax.scatter(x_rebinned[min_ind],
                       array[min_ind],s=20,marker='v',c='C3',zorder=10)
            ax.axhline(0,ls='--')
        ax1.scatter([_[0] for _ in minima],
                    [_[1] for _ in minima],
                    marker='v',c='C3',zorder=10)
        ax1.scatter([_[0] for _ in maxima],
                    [_[1] for _ in maxima],
                    marker='^',c='C4',zorder=10)
        ax4.scatter([_[0] for _ in minima[:-1]],
                    np.diff([_[0] for _ in minima]),
                    marker='v',c='C3')
        
        ax4.scatter([_[0] for _ in maxima[:-1]],
                    np.diff([_[0] for _ in maxima]),
                    marker='^',c='C4')
        
        ax5.scatter([_[0] for _ in minima[:-1]],
                    np.abs(np.diff([_[1] for _ in minima])),
                    marker='v',c='C3')
        
        ax5.scatter([_[0] for _ in maxima[:-1]],
                    np.abs(np.diff([_[1] for _ in maxima])),
                    marker='^',c='C4')
        
        # ax2.set_ylim(np.nanpercentile(derivative1st,[1,99]))
        # ax3.set_ylim(np.nanpercentile(derivative2nd,[1,99]))
        # fig.tight_layout()
    return [maxima, minima]

def _validate(y_axis,extrema,kind='max',mindist=3,verbose=False):
    from scipy.stats import kstest
    validated = []
    absolute_mindist = 10
    mindist = int(mindist) if int(mindist)>=absolute_mindist else absolute_mindist
    last_xint = 0
    if verbose:
        print('{0:<4s}{1:<6s}{2:<8s}{3:<8s}{4:<6s}{5:<6s}{6:<8s}{7:<6s}{8:<6s}{9:<6s}'.format('i','x','left','right','y','<y>','cond2','cond3','cond4','cond2&3&4'))
    for i,(x,y) in enumerate(extrema):
        xint = np.round(x).astype(int)
        yint = int(y)
        if xint==last_xint:
            continue
        y_segment = y_axis[last_xint:xint]
        
        # cond1 = False 
        # left = y_axis[xint]
        
        # if xint<len(y_axis)-1:
        #     right = y_axis[xint+1]
        #     if kind == 'max': # The point following a maximum should be smaller
        #         cond1 = left>right
        #     elif kind == 'min': # The point following a minimum should be larger
        #         cond1 = left<right
        # else:
        #     cond1 = True
        value = [x,y]
        # print((5*"{:<12.1f}").format(x,xint,last_xint,y,y_axis[xint]))
        cond2 = value not in validated
        # print(xint!=last_xint)
        if len(validated)>0 and xint>0:
            # Check that the data is not consistent with random noise
            # y_span = y_segment
            # mean, std_dev = np.mean(np.sqrt(y_span)), np.std(np.sqrt(y_span))
            # ks_stat, p_value = kstest(y_span, 'norm', args=(mean, std_dev))
            
            
            # print('branch 1', x, xint)
            # Calculate the average of y-axis between the last validated point 
            # and the point under examination
            y_average = np.nanmean(y_segment)
            # Check that the last validated x-coordinate is at least mindist 
            # away from the point under examination 
            cond4 = np.abs(validated[-1][0]-x) > mindist
            # print(f"{i:<6d} {np.abs(validated[-1][0]-x):<8.3f} ß{mindist:<4d} {cond4:<8}")
            
        else: # if this is the first point
            # print('branch 2')
            if xint>0:
                y_average = np.nanmean(y_axis[last_xint:xint])
            else:
                y_average = np.nanmean(y_axis[last_xint:last_xint+mindist]) #if kind=='min' else 0
            cond4 = True
        # The maximum/minimum should be above/below the average for the line
        if kind == 'max': 
            cond3 = y > y_average
        elif kind == 'min':
            cond3 = y < y_average
        
        
        
        if verbose:
            
            print(f'{i:<6d}{x:<6.1f} {xint:<8d} {last_xint:<8d} {y:<6.1f} {y_average:<8.2f} {cond2:<6} {cond3:<6} {cond4:<6} {cond2&cond3&cond4:<6}')
        if cond2 and cond3 and cond4:
            validated.append(value)
            last_xint = xint
    rarray = np.array(validated)
    # x_return = rarray[:,0].astype(int)
    # y_return = rarray[:,1]
    return np.array(validated)

def _fit_clip(x,y,y_err=None,deg=2,maxiter=100):
    assert len(x)==len(y)
    y_err = y_err if y_err is not None else np.ones_like(y)
    assert len(y_err)==len(y)
    
    import copy
    X = copy.deepcopy(x)
    Y = copy.deepcopy(y)
    Y_err = copy.deepcopy(y_err)
    
    keep = np.full_like(x,True,dtype=bool)
    discard = keep
    chisq = 0.
    

    while sum(discard)!=0:
        x_i = X[keep]
        y_i = Y[keep]
        y_err_i = Y_err[keep]
        
        coeff_i, cost, rank, sv, rcond = np.polyfit(x_i, y_i,
                                                    deg = deg, w=1/y_err_i,
                                                    full=True)
        
        y_fit_i = np.polyval(coeff_i, x_i)
        rsd_i = (y_i - y_fit_i) / y_err_i
        
        
    
def _validate_new(y_axis,maxima,minima,mindist=3,verbose=False):
   
    validated = []
    
    x_M, y_M = maxima.T
    x_m, y_m = minima.T
    # Sort all extrema by x-coordinate
    x_extrema = np.hstack([x_M, x_m])
    sorter = np.argsort(x_extrema)
    x_sorted = x_extrema[sorter]
    y_extrema = np.hstack([y_M, y_m])
    y_sorted = y_extrema[sorter]
    
    # ID = +1 if maximum and -1 if minimum
    id_array = np.hstack([np.full_like(x_M,+1),np.full_like(x_m,-1)])
    id_sorted = id_array[sorter]
    
    D_M = np.diff(x_M)    # distances in x between consecutive Maxima
    P_M = np.diff(y_M)    # distances in y between consecutive Maxima
    
    D_m = np.diff(x_m)    # distances in x between consecutive minima
    P_m = np.diff(y_m)    # distances in y between consecutive minima
    
    D = np.diff(x_sorted) # distances in x between consecutive extrema
    P = np.diff(y_sorted) # distances in y between consecutive extrema 
    
    # condition 1
    # the sum of consecutive IDs must be 0, otherwise two extrema of the same 
    # kind appear twice in a row 
    
    # condition 2
    # distance must be at least mindist
    cut_D = np.where(D>mindist)[0]
    
    # condition 3
    # change in y between two consecutive extrema must be at least a factor f
    # smaller than the median change
    median_P = np.median(P)
    cut_P = np.where(P>0.3*median_P)[0]
    

    return np.array(validated)
 
import itertools


def apply_pairwise(values,
                   function):
    """
    Applies function to consecutive pairs of elements from an input list
    """
    def pairwise(iterable):
        """
        s -> (s0,s1), (s1,s2), (s2,s3), ...
        """
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    yield from itertools.chain([None],
                               itertools.starmap(function, pairwise(values)))       
    
def _filter(y_axis, x_axis=None, len=5):
    """
    Smooths the function using Wiener filter of length len (has to be odd).
    """
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    return wiener(y_axis, len)
    
def _smooth(x, window_len=11, window="hanning", mode="valid"):
    """
    smooth the data using a window of the requested size.
    
    This method is based on the convolution of a scaled window on the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    
    keyword arguments:
    x -- the input signal 
    
    window_len -- the dimension of the smoothing window; should be an odd
        integer (default: 11)
    
    window -- the type of window from 'flat', 'hanning', 'hamming', 
        'bartlett', 'blackman', where flat is a moving average
        (default: 'hanning')

    
    return: the smoothed signal
        
    example:

    t = linspace(-2,2,0.1)
    x = sin(t)+randn(len(t))*0.1
    y = _smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, 
    numpy.convolve, scipy.signal.lfilter 
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    if window_len<3:
        return x
    #declare valid windows in a dictionary
    window_funcs = {
        "flat": lambda _len: np.ones(_len, "d"),
        "hanning": np.hanning,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
        "nuttall": nuttall
        }
    
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    try:
        w = window_funcs[window](window_len)
    except KeyError:
        raise ValueError(
            "Window is not one of '{0}', '{1}', '{2}', '{3}', '{4}'".format(
            *window_funcs.keys()))
    
    y = np.convolve(w / w.sum(), s, mode = mode)
    
    return y
def _rebin( a, newshape ):
    '''Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]
    
def get_window(y_axis,plot=False,minimum=8):
    freq0, P0    = welch(y_axis,nperseg=512)
    cut = np.where(freq0>0.02)[0]
    freq, P = freq0[cut], P0[cut]
    maxind     = np.argmax(P)
    maxfreq    = freq[maxind]
    if plot:
        import matplotlib.ticker as ticker
        def one_over(x):
            """Vectorized 1/x, treating x==0 manually"""
            x = np.array(x, float)
            near_zero = np.isclose(x, 0)
            x[near_zero] = np.inf
            x[~near_zero] = 1 / x[~near_zero]
            return x
        
        fig, ax = plt.subplots(1)
        ax.semilogy(1/freq,P)
        ax.semilogy(1/maxfreq,P[maxind],marker='x',c='C1')
        ax2 = ax.secondary_xaxis('top',functions=(one_over,one_over))
        # ax2.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=10))
        ax.set_xlabel("Period (pix)")
        ax2.set_xlabel("Frequency (1/pix)")
        ax.set_ylabel("Power")
        # ax.set_xlim(1e-3,0.51)
        
        
    window = mathfunc.round_down_to_odd(1./maxfreq)
    return window if window>minimum else minimum
    
def zero_crossings(y_axis, window_len = 11, 
    window_f="hanning", offset_corrected=False):
    """
    Algorithm to find zero crossings. Smooths the curve and finds the
    zero-crossings by looking for a sign change.
    
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find zero-crossings
    
    window_len -- the dimension of the smoothing window; should be an odd
        integer (default: 11)
    
    window_f -- the type of window from 'flat', 'hanning', 'hamming', 
        'bartlett', 'blackman' (default: 'hanning')
    
    offset_corrected -- Used for recursive calling to remove offset when needed
    
    
    return: the index for each zero-crossing
    """
    # smooth the curve
    length = len(y_axis)
    
    # discard tail of smoothed signal
    y_axis = _smooth(y_axis, window_len, window_f)[:length]
    indices = np.where(np.diff(np.sign(y_axis)))[0]
    
    # check if zero-crossings are valid
    diff = np.diff(indices)
    if diff.std() / diff.mean() > 0.1:
        #Possibly bad zero crossing, see if it's offsets
        if ((diff[::2].std() / diff[::2].mean()) < 0.1 and 
        (diff[1::2].std() / diff[1::2].mean()) < 0.1 and
        not offset_corrected):
            #offset present attempt to correct by subtracting the average
            offset = np.mean([y_axis.max(), y_axis.min()])
            return zero_crossings(y_axis-offset, window_len, window_f, True)
        #Invalid zero crossings and the offset has been removed
        print(diff.std() / diff.mean())
        print(np.diff(indices))
        raise ValueError(
            "False zero-crossings found, indicates problem {0!s} or {1!s}".format(
            "with smoothing window", "unhandled problem with offset"))
    # check if any zero crossings were found
    if len(indices) < 1:
        raise ValueError("No zero crossings found")
    #remove offset from indices due to filter function when returning
    return indices - (window_len // 2 - 1)
    # used this to test the fft function's sensitivity to spectral leakage
    #return indices + np.asarray(30 * np.random.randn(len(indices)), int)
    
############################Frequency calculation#############################
#    diff = np.diff(indices)
#    time_p_period = diff.mean()
#    
#    if diff.std() / time_p_period > 0.1:
#        raise ValueError(
#            "smoothing window too small, false zero-crossing found")
#    
#    #return frequency
#    return 1.0 / time_p_period
##############################################################################


def zero_crossings_sine_fit(y_axis, x_axis, fit_window = None, smooth_window = 11):
    """
    Detects the zero crossings of a signal by fitting a sine model function
    around the zero crossings:
    y = A * sin(2 * pi * Hz * (x - tau)) + k * x + m
    Only tau (the zero crossing) is varied during fitting.
    
    Offset and a linear drift of offset is accounted for by fitting a linear
    function the negative respective positive raw peaks of the wave-shape and
    the amplitude is calculated using data from the offset calculation i.e.
    the 'm' constant from the negative peaks is subtracted from the positive
    one to obtain amplitude.
    
    Frequency is calculated using the mean time between raw peaks.
    
    Algorithm seems to be sensitive to first guess e.g. a large smooth_window
    will give an error in the results.
    
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the position of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    
    fit_window -- Number of points around the approximate zero crossing that
        should be used when fitting the sine wave. Must be small enough that
        no other zero crossing will be seen. If set to none then the mean
        distance between zero crossings will be used (default: None)
    
    smooth_window -- the dimension of the smoothing window; should be an odd
        integer (default: 11)
    
    
    return: A list containing the positions of all the zero crossings.
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    #get first guess
    zero_indices = zero_crossings(y_axis, window_len = smooth_window)
    #modify fit_window to show distance per direction
    if fit_window == None:
        fit_window = np.diff(zero_indices).mean() // 3
    else:
        fit_window = fit_window // 2
    
    #x_axis is a np array, use the indices to get a subset with zero crossings
    approx_crossings = x_axis[zero_indices]
    
    
    
    #get raw peaks for calculation of offsets and frequency
    raw_peaks = peakdetect_zero_crossing(y_axis, x_axis)
    #Use mean time between peaks for frequency
    ext = lambda x: list(zip(*x)[0])
    _diff = map(np.diff, map(ext, raw_peaks))
    
    
    Hz = 1 / np.mean(map(np.mean, _diff))
    #Hz = 1 / np.diff(approx_crossings).mean() #probably bad precision
    
    
    #offset model function
    offset_func = lambda x, k, m: k * x + m
    k = []
    m = []
    amplitude = []
    
    for peaks in raw_peaks:
        #get peak data as nparray
        x_data, y_data = map(np.asarray, zip(*peaks))
        #x_data = np.asarray(x_data)
        #y_data = np.asarray(y_data)
        #calc first guess
        A = np.mean(y_data)
        p0 = (0, A)
        popt, pcov = curve_fit(offset_func, x_data, y_data, p0)
        #append results
        k.append(popt[0])
        m.append(popt[1])
        amplitude.append(abs(A))
    
    #store offset constants
    p_offset = (np.mean(k), np.mean(m))
    A = m[0] - m[1]
    #define model function to fit to zero crossing
    #y = A * sin(2*pi * Hz * (x - tau)) + k * x + m
    func = lambda x, tau: A * np.sin(2 * pi * Hz * (x - tau)) + offset_func(x, *p_offset)
    
    
    #get true crossings
    true_crossings = []
    for indice, crossing in zip(zero_indices, approx_crossings):
        p0 = (crossing, )
        subset_start = max(indice - fit_window, 0.0)
        subset_end = min(indice + fit_window + 1, len(x_axis) - 1.0)
        x_subset = np.asarray(x_axis[subset_start:subset_end])
        y_subset = np.asarray(y_axis[subset_start:subset_end])
        #fit
        popt, pcov = curve_fit(func, x_subset, y_subset, p0)
        
        true_crossings.append(popt[0])
    
    
    return true_crossings
        
        
    
    
def _test_zero():
    _max, _min = peakdetect_zero_crossing(y,x)
def _test():
    _max, _min = peakdetect(y,x, delta=0.30)
    
    
def _test_graph():
    i = 10000
    x = np.linspace(0,3.7*pi,i)
    y = (0.3*np.sin(x) + np.sin(1.3 * x) + 0.9 * np.sin(4.2 * x) + 0.06 *
    np.random.randn(i))
    y *= -1
    x = range(i)
    
    _max, _min = peakdetect(y,x,750, 0.30)
    xm = [p[0] for p in _max]
    ym = [p[1] for p in _max]
    xn = [p[0] for p in _min]
    yn = [p[1] for p in _min]
    
    plot = pylab.plot(x,y)
    pylab.hold(True)
    pylab.plot(xm, ym, "r+")
    pylab.plot(xn, yn, "g+")
    
    _max, _min = peak_det_bad.peakdetect(y, 0.7, x)
    xm = [p[0] for p in _max]
    ym = [p[1] for p in _max]
    xn = [p[0] for p in _min]
    yn = [p[1] for p in _min]
    pylab.plot(xm, ym, "y*")
    pylab.plot(xn, yn, "k*")
    pylab.show()
    
def _test_graph_cross(window = 11):
    i = 10000
    x = np.linspace(0,8.7*pi,i)
    y = (2*np.sin(x) + 0.006 *
    np.random.randn(i))
    y *= -1
    pylab.plot(x,y)
    #pylab.show()
    
    
    crossings = zero_crossings_sine_fit(y,x, smooth_window = window)
    y_cross = [0] * len(crossings)
    
    
    plot = pylab.plot(x,y)
    pylab.hold(True)
    pylab.plot(crossings, y_cross, "b+")
    pylab.show()
    
def get_window_robust(y_axis, sampling_rate=1.0, plot=False,
                      default_window_period=15, # Default period in pixels
                      min_period_pixels=3,      # Smallest expected feature period
                      max_period_pixels=None,   # Largest relevant period
                      nperseg_min_periods=4):   # Welch nperseg = N * est. period
    """
    Estimates an optimal window length based on the dominant period in y_axis,
    assumed to correspond to LFC line spacing or feature width.

    Args:
        y_axis (np.array): The input signal.
        sampling_rate (float): Sampling rate of y_axis (1.0 if x is pixel index).
        plot (bool): Whether to plot the power spectrum.
        default_window_period (int): Fallback period in pixels if no clear peak.
        min_period_pixels (int): Minimum expected period of features (in pixels).
        max_period_pixels (int): Maximum expected period. If None, defaults to len(y_axis)/3.
        nperseg_min_periods (int): nperseg for welch will be at least this many times
                                   the max_period_pixels, capped by signal length.
    Returns:
        int: Estimated odd window length (period).
    """
    if len(y_axis) < 2 * default_window_period: # Heuristic for very short signals
        # Ensure mathfunc.round_down_to_odd is available
        return mathfunc.round_down_to_odd(max(3, len(y_axis) // 5))

    if max_period_pixels is None:
        max_period_pixels = len(y_axis) / 3.0 # Max period is 1/3 signal length
    max_period_pixels = max(max_period_pixels, min_period_pixels * 2) # Ensure range

    # nperseg for Welch: should be long enough to capture lowest frequency of interest
    nperseg_val = min(len(y_axis), max(256, int(nperseg_min_periods * max_period_pixels)))

    try:
        freqs, Pxx = welch(y_axis, fs=sampling_rate, nperseg=nperseg_val, scaling='density', window='hann')
    except ValueError: # e.g. if nperseg_val > len(y_axis) after all checks
        return mathfunc.round_down_to_odd(default_window_period)


    # Filter frequencies to the range corresponding to expected feature spacings
    # Min period -> Max frequency; Max period -> Min frequency
    # Add a small epsilon to avoid division by zero if max_period_pixels is huge or min_period_pixels is tiny
    min_freq_of_interest = 1.0 / (max_period_pixels + 1e-9)
    max_freq_of_interest = 1.0 / (min_period_pixels + 1e-9)

    # Exclude DC (freqs[0]) and frequencies outside our band of interest
    valid_indices = np.where((freqs > min_freq_of_interest) & (freqs < max_freq_of_interest))[0]

    if len(valid_indices) == 0 or len(Pxx[valid_indices]) == 0:
        period = default_window_period # Fallback
    else:
        # Find the peak within the valid frequency band
        peak_idx_in_valid = np.argmax(Pxx[valid_indices])
        dominant_freq = freqs[valid_indices[peak_idx_in_valid]]

        if dominant_freq <= 1e-9: # Should not happen if min_freq_of_interest > 0
             period = default_window_period
        else:
            period = 1.0 / dominant_freq

    # Sanity check the period
    period = np.clip(period, min_period_pixels, max_period_pixels)
    window_len = mathfunc.round_down_to_odd(int(period))
    window_len = max(3, window_len) # Ensure at least 3

    if plot:
        import matplotlib.pyplot as plt # Keep imports local to plotting block
        # ... (plotting code from your get_window, adapted) ...
        fig, ax = plt.subplots(1, figsize=(8,5))
        ax.semilogy(freqs, Pxx, label='Full Spectrum (Welch)', color='grey', alpha=0.7)
        if len(valid_indices) > 0:
             ax.semilogy(freqs[valid_indices], Pxx[valid_indices], label='Considered Band', color='orange')
             ax.semilogy(dominant_freq, Pxx[valid_indices[peak_idx_in_valid]], 'rx', markersize=10,
                         label=f'Dominant (Period ~{period:.1f}px)')
        ax.set_xlabel(f'Frequency (1/pixels, fs={sampling_rate})')
        ax.set_ylabel('PSD')
        ax.legend()
        ax.set_title(f'Power Spectrum for Window Estimation (Est. Window: {window_len})')
        ax.grid(True, linestyle=':')

        # Secondary x-axis for period
        def freq_to_period(f):
            return np.where(f > 1e-9, 1.0/f, np.inf)
        def period_to_freq(p):
            return np.where(p > 1e-9, 1.0/p, np.inf)

        # secax = ax.secondary_xaxis('top', functions=(freq_to_period, period_to_freq))
        # secax.set_xlabel('Period (pixels)')
        # plt.tight_layout()
        plt.show()

    return window_len


def _ensure_alternation(peaks_x, peaks_y, valleys_x, valleys_y):
    """
    Ensures strict P-V-P-V alternation.
    If P-P, keeps higher. If V-V, keeps lower.
    Returns numpy arrays.
    """
    # Convert inputs to numpy arrays if they are lists
    peaks_x, peaks_y = np.array(peaks_x), np.array(peaks_y)
    valleys_x, valleys_y = np.array(valleys_x), np.array(valleys_y)

    if not len(peaks_x) and not len(valleys_x): # Both empty
        return np.array([]), np.array([]), np.array([]), np.array([])
    if not len(peaks_x): # Only valleys
        return np.array([]), np.array([]), valleys_x, valleys_y
    if not len(valleys_x): # Only peaks
        return peaks_x, peaks_y, np.array([]), np.array([])

    # Combine all features with a type indicator (1 for peak, -1 for valley)
    all_x = np.concatenate((peaks_x, valleys_x))
    all_y = np.concatenate((peaks_y, valleys_y))
    all_types = np.concatenate((np.ones(len(peaks_x)), -np.ones(len(valleys_x))))

    # Sort by x-coordinate
    sorted_indices = np.argsort(all_x)
    sorted_x = all_x[sorted_indices]
    sorted_y = all_y[sorted_indices]
    sorted_types = all_types[sorted_indices]

    final_x_list, final_y_list, final_types_list = [], [], []

    if len(sorted_x) == 0: # Should not happen if checks above are passed
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Add the first feature
    final_x_list.append(sorted_x[0])
    final_y_list.append(sorted_y[0])
    final_types_list.append(sorted_types[0])

    for i in range(1, len(sorted_x)):
        current_x, current_y, current_type = sorted_x[i], sorted_y[i], sorted_types[i]
        last_x, last_y, last_type = final_x_list[-1], final_y_list[-1], final_types_list[-1]

        if current_type != last_type: # Different type, so alternate; add it
            final_x_list.append(current_x)
            final_y_list.append(current_y)
            final_types_list.append(current_type)
        else: # Same type as previous (e.g., P-P or V-V)
            if current_type == 1: # Both are peaks
                if current_y > last_y: # Current peak is higher, replace previous
                    final_x_list[-1], final_y_list[-1] = current_x, current_y
            else: # Both are valleys
                if current_y < last_y: # Current valley is lower, replace previous
                    final_x_list[-1], final_y_list[-1] = current_x, current_y
    
    final_x_arr = np.array(final_x_list)
    final_y_arr = np.array(final_y_list)
    final_types_arr = np.array(final_types_list)

    # Separate back into peaks and valleys
    out_peaks_x = final_x_arr[final_types_arr == 1]
    out_peaks_y = final_y_arr[final_types_arr == 1]
    out_valleys_x = final_x_arr[final_types_arr == -1]
    out_valleys_y = final_y_arr[final_types_arr == -1]
    
    return out_peaks_x, out_peaks_y, out_valleys_x, out_valleys_y


def _calculate_features_for_extrema(
    extrema_x, extrema_y,           # Current features (e.g., peaks_x, peaks_y)
    opposite_x, opposite_y,       # Bracketing features (e.g., valleys_x, valleys_y)
    is_peak,                        # Boolean: True if extrema are peaks
    min_extrema_for_spacing_model=3 # Min number of *extrema_x* to attempt spacing model
):
    """
    Calculates features: x_coord, prominence/depth, spacing_deviation.
    Returns a NumPy array of shape (num_extrema, 3).
    """
    num_extrema = len(extrema_x)
    if num_extrema == 0:
        return np.array([]).reshape(0, 3) # Ensure 3 columns for vstack later

    # 1. Prominence / Depth (delta_y)
    prom_depth_values = np.zeros(num_extrema)
    for i in range(num_extrema):
        x_i, y_i = extrema_x[i], extrema_y[i]
        
        # Find bracketing opposite features
        left_indices = np.where(opposite_x < x_i)[0]
        right_indices = np.where(opposite_x > x_i)[0]

        y_brackets_list = []
        if len(left_indices) > 0: # Has a bracketing feature to the left
            y_brackets_list.append(opposite_y[left_indices[-1]]) # Closest on the left
        if len(right_indices) > 0: # Has a bracketing feature to the right
            y_brackets_list.append(opposite_y[right_indices[0]]) # Closest on the right

        if not y_brackets_list: # No bracketing features found
            prom_depth_values[i] = 0 # Assign 0 prominence/depth
            continue

        y_brackets_arr = np.array(y_brackets_list)
        if is_peak:
            # Prominence: y_peak - max(neighboring_valley_y)
            prom_depth_values[i] = y_i - np.max(y_brackets_arr)
        else:
            # Depth: min(neighboring_peak_y) - y_valley
            prom_depth_values[i] = np.min(y_brackets_arr) - y_i
        
        prom_depth_values[i] = max(0, prom_depth_values[i]) # Ensure non-negative


    # 2. Spacing Deviation (delta_x_dev)
    spacing_dev_values = np.zeros(num_extrema)
    # Need at least min_extrema_for_spacing_model points to define enough spacings for a robust fit.
    # TheilSenRegressor typically needs at least 2 points for X and y for a fit.
    # If num_extrema = 3, spacings = 2, spacing_x_midpoints = 2. This is min for TheilSen.
    if num_extrema >= min_extrema_for_spacing_model:
        spacings = np.diff(extrema_x)
        spacing_x_midpoints = (extrema_x[:-1] + extrema_x[1:]) / 2

        # Ensure enough points for the regressor model
        if len(spacing_x_midpoints) >= 2: # At least 2 midpoints (i.e., 3 extrema)
            model = TheilSenRegressor(random_state=42) # Robust to outliers in spacing
            try:
                model.fit(spacing_x_midpoints.reshape(-1, 1), spacings)
                
                for i in range(num_extrema):
                    devs_list = []
                    # Deviation from spacing with previous point
                    if i > 0:
                        d_prev = extrema_x[i] - extrema_x[i-1]
                        x_mid_prev = (extrema_x[i] + extrema_x[i-1]) / 2
                        d_prev_exp = model.predict(np.array([[x_mid_prev]]))[0]
                        devs_list.append(abs(d_prev - d_prev_exp))
                    
                    # Deviation from spacing with next point
                    if i < num_extrema - 1:
                        d_next = extrema_x[i+1] - extrema_x[i]
                        x_mid_next = (extrema_x[i+1] + extrema_x[i]) / 2
                        d_next_exp = model.predict(np.array([[x_mid_next]]))[0]
                        devs_list.append(abs(d_next - d_next_exp))
                    
                    if devs_list:
                        spacing_dev_values[i] = np.mean(devs_list)
                    # else (single point or failed model), spacing_dev_values[i] remains 0
            except ValueError: 
                # Fit can fail if, e.g., x_midpoints are not diverse enough.
                # In this case, spacing_dev_values remain 0.
                pass 
    
    # Assemble the feature matrix: [x_coordinate, prominence/depth, spacing_deviation]
    feature_matrix = np.vstack([
        extrema_x, 
        prom_depth_values,
        spacing_dev_values
    ]).T
    
    return feature_matrix

# --- Main Integrated Function ---
def peakdetect_derivatives_with_clustering(
    y_axis, x_axis=None,
    super_sample_factor=5,
    window_len_method='auto_robust',
    user_window_len_orig_scale=None,
    deriv_method='coeff',
    first_deriv_zero_threshold_factor=0.05,
    plot=False,
    # get_window_robust parameters
    gw_min_period_pixels=5, gw_max_period_pixels=None, gw_default_window_period=15,
    # DBSCAN Clustering Parameters
    dbscan_eps=0.5,  # DBSCAN epsilon parameter (TUNE CAREFULLY!)
    dbscan_min_samples=5, # DBSCAN min_samples parameter (TUNE!)
    min_features_for_clustering=10, # Min num of peaks/valleys to attempt clustering
    # _calculate_features_for_extrema parameter
    min_extrema_for_spacing_model=3
):
    """
    Peak detection using derivatives, followed by DBSCAN clustering for filtering.
    Uses MinMaxScaler for feature scaling.
    """
    x_axis_orig, y_axis_orig = _datacheck_peakdetect(x_axis, y_axis)

    # 1. Supersample
    if super_sample_factor > 1:
        y_rebinned = _rebin(y_axis_orig, newshape=(super_sample_factor * len(y_axis_orig),))
        x_rebinned = np.linspace(np.min(x_axis_orig), np.max(x_axis_orig), len(y_rebinned))
    else:
        y_rebinned = np.copy(y_axis_orig)
        x_rebinned = np.copy(x_axis_orig)

    # 2. Determine Smoothing Window Length
    if window_len_method == 'auto_robust':
        window_len_on_orig_scale = get_window_robust(
            y_axis_orig, plot=plot, min_period_pixels=gw_min_period_pixels,
            max_period_pixels=gw_max_period_pixels, default_window_period=gw_default_window_period
        )
    # ... (other window_len_method options as in previous response) ...
    else: # Fallback or simplified for example
        window_len_on_orig_scale = get_window_robust(y_axis_orig, plot=plot)


    actual_smoothing_window_len = mathfunc.round_down_to_odd(
        int(window_len_on_orig_scale * super_sample_factor)
    )
    actual_smoothing_window_len = min(actual_smoothing_window_len, mathfunc.round_down_to_odd(len(y_rebinned) // 2 -1))
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    print(f"Orig scale window: {window_len_on_orig_scale}, Rebinned smooth window: {actual_smoothing_window_len}")

    # 3. Smooth and Align (using mode='valid' in your _smooth function)
    y_smoothed_from_smooth_func = _smooth(
        y_rebinned,
        window_len=actual_smoothing_window_len,
        window='nuttall',
        mode="valid"  # <<< ENSURE THIS MODE IS PASSED TO AND USED BY _smooth
    )
    
    # As analyzed:
    # len(y_smoothed_from_smooth_func) = len(y_rebinned) + actual_smoothing_window_len - 1

    # To get the 'centered' part that aligns with y_rebinned (length len(y_rebinned)),
    # we trim (actual_smoothing_window_len - 1) // 2 from each side of this output.
    # This requires actual_smoothing_window_len to be odd.
    
    trim_amount_each_side = (actual_smoothing_window_len - 1) // 2
    
    # Check if the output from _smooth is long enough for trimming
    if len(y_smoothed_from_smooth_func) < (2 * trim_amount_each_side + 1):
        print(f"Warning: Output of _smooth (len {len(y_smoothed_from_smooth_func)}) is too short "
              f"for trimming ({trim_amount_each_side} from each side based on "
              f"actual_smoothing_window_len: {actual_smoothing_window_len}). "
              f"len(y_rebinned): {len(y_rebinned)}. This might indicate an issue in "
              f"_smooth's output length or a very short y_rebinned relative to the window.")
        # Fallback or error handling:
        # If this happens, reliable peak detection is unlikely.
        # Ensure the return format matches what the caller expects for empty/error results.
        return ([[], []], [[], []]) # Example: returns two items, each a list of two empty lists

    y_smoothed = y_smoothed_from_smooth_func[trim_amount_each_side : -trim_amount_each_side]
    
    # After trimming, y_smoothed should have length len(y_rebinned).
    # (L_r + w_len - 1) - 2 * ((w_len - 1)//2) = L_r (if w_len is odd)
    
    x_coords_for_smoothed = x_rebinned

    # Sanity check to catch alignment issues early
    if len(y_smoothed) != len(x_coords_for_smoothed):
        # This could happen if actual_smoothing_window_len somehow ended up even,
        # or if the output length of _smooth was not as expected.
        # Attempt a minor adjustment if off by one (can sometimes happen with convolution length logic)
        if abs(len(y_smoothed) - len(x_coords_for_smoothed)) == 1 and len(y_smoothed) > 0 :
            print(f"Warning: Post-trimming length mismatch. len(y_smoothed)={len(y_smoothed)}, "
                  f"len(x_coords_for_smoothed)={len(x_coords_for_smoothed)}. Attempting minor trim.")
            if len(y_smoothed) > len(x_coords_for_smoothed):
                y_smoothed = y_smoothed[:len(x_coords_for_smoothed)]
            # else: (y_smoothed is shorter) this path is problematic and indicates a deeper issue.
            # For now, we assume y_smoothed won't be shorter if _smooth worked as expected.

        # Final check after potential adjustment
        if len(y_smoothed) != len(x_coords_for_smoothed):
            raise ValueError(
                f"CRITICAL Alignment Error with mode='valid' path: len(y_smoothed)={len(y_smoothed)} "
                f"!= len(x_coords_for_smoothed)={len(x_coords_for_smoothed)}. "
                f"Rebinned len={len(y_rebinned)}, "
                f"Output from _smooth (mode=valid) before trim len={len(y_smoothed_from_smooth_func)}, "
                f"Trim amount each side={trim_amount_each_side}"
            )

    if len(y_smoothed) < 3:
        print("Warning: Smoothed data (after trimming for 'valid' mode) has less than 3 points. Returning empty.")
        return ([[], []], [[], []]) # Ensure correct return format


    # 4. Calculate Derivatives
    derivative1st = mathfunc.derivative1d(y_smoothed, x=None, order=1, method=deriv_method)
    derivative2nd = mathfunc.derivative1d(y_smoothed, x=None, order=2, method=deriv_method)

    # 5. Identify Candidate Extrema (Sensitive primary detection)
    deriv1_thresh = first_deriv_zero_threshold_factor * np.max(np.abs(derivative1st)) if len(derivative1st) > 0 else 1e-9
    candidate_indices = []
    sign_changes = np.diff(np.sign(derivative1st))
    crossings_idx = np.where(np.abs(sign_changes) == 2)[0]
    for idx in crossings_idx:
        candidate_indices.append(idx if np.abs(derivative1st[idx]) < np.abs(derivative1st[idx+1]) else idx+1)
    candidate_indices = sorted(list(set(candidate_indices)))
    candidate_indices = np.array(candidate_indices, dtype=int)

    if len(candidate_indices) == 0: return [[], []]
    
    valid_candidate_indices = candidate_indices[candidate_indices < len(derivative2nd)] # Ensure bounds
    max_ind = valid_candidate_indices[derivative2nd[valid_candidate_indices] < 0]
    min_ind = valid_candidate_indices[derivative2nd[valid_candidate_indices] > 0]

    # Initial candidates from derivatives
    current_peaks_x = x_coords_for_smoothed[max_ind] if len(max_ind) > 0 else np.array([])
    current_peaks_y = y_smoothed[max_ind] if len(max_ind) > 0 else np.array([])
    current_valleys_x = x_coords_for_smoothed[min_ind] if len(min_ind) > 0 else np.array([])
    current_valleys_y = y_smoothed[min_ind] if len(min_ind) > 0 else np.array([])

    # 6. Filter with DBSCAN Clustering
    # Ensure strict alternation before feature calculation for clustering
    current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y = \
        _ensure_alternation(current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y)

    # --- Filter Peaks with Clustering ---
    final_peaks_x, final_peaks_y = current_peaks_x, current_peaks_y # Default if no clustering
    if len(current_peaks_x) >= min_features_for_clustering and len(current_valleys_x) > 0: # Need valleys for prominence
        peak_features = _calculate_features_for_extrema(
            current_peaks_x, current_peaks_y,
            current_valleys_x, current_valleys_y, # Use current valleys for bracketing
            is_peak=True,
            min_extrema_for_spacing_model=min_extrema_for_spacing_model
        )
        if peak_features.shape[0] > 0 and peak_features.shape[1] == 3:
            scaler_peaks = MinMaxScaler() # Using MinMaxScaler
            scaled_peak_features = scaler_peaks.fit_transform(peak_features)
            
            dbscan_p = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            peak_labels = dbscan_p.fit_predict(scaled_peak_features)

            core_point_labels = peak_labels[peak_labels != -1]
            if len(core_point_labels) > 0:
                most_common_label = Counter(core_point_labels).most_common(1)[0][0]
                true_peak_indices = np.where(peak_labels == most_common_label)[0]
                final_peaks_x = current_peaks_x[true_peak_indices]
                final_peaks_y = current_peaks_y[true_peak_indices]
            elif len(peak_labels) > 0: # All outliers or no clusters
                final_peaks_x, final_peaks_y = np.array([]), np.array([])
    
    # Re-alternate with filtered peaks before filtering valleys
    # This is important as valley features depend on bracketing peaks
    current_peaks_x_filt, current_peaks_y_filt, \
    current_valleys_x_interim, current_valleys_y_interim = _ensure_alternation(
        final_peaks_x, final_peaks_y, current_valleys_x, current_valleys_y
    )
    # Use these newly alternated lists for valley filtering
    # (current_valleys_x_interim might be slightly different from original current_valleys_x)


    # --- Filter Valleys with Clustering ---
    final_valleys_x, final_valleys_y = current_valleys_x_interim, current_valleys_y_interim # Default
    if len(current_valleys_x_interim) >= min_features_for_clustering and len(current_peaks_x_filt) > 0: # Need peaks for depth
        valley_features = _calculate_features_for_extrema(
            current_valleys_x_interim, current_valleys_y_interim,
            current_peaks_x_filt, current_peaks_y_filt, # Use filtered peaks for bracketing
            is_peak=False,
            min_extrema_for_spacing_model=min_extrema_for_spacing_model
        )
        if valley_features.shape[0] > 0 and valley_features.shape[1] == 3:
            scaler_valleys = MinMaxScaler() # Using MinMaxScaler
            scaled_valley_features = scaler_valleys.fit_transform(valley_features)

            dbscan_v = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            valley_labels = dbscan_v.fit_predict(scaled_valley_features)
            
            core_point_labels = valley_labels[valley_labels != -1]
            if len(core_point_labels) > 0:
                most_common_label = Counter(core_point_labels).most_common(1)[0][0]
                true_valley_indices = np.where(valley_labels == most_common_label)[0]
                final_valleys_x = current_valleys_x_interim[true_valley_indices]
                final_valleys_y = current_valleys_y_interim[true_valley_indices]
            elif len(valley_labels) > 0: # All outliers
                final_valleys_x, final_valleys_y = np.array([]), np.array([])
    
    # Final alternation pass
    final_peaks_x, final_peaks_y, final_valleys_x, final_valleys_y = \
        _ensure_alternation(final_peaks_x, final_peaks_y, final_valleys_x, final_valleys_y)


    if plot:
        # Adapted plotting from previous response
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
        ax1.plot(x_axis_orig, y_axis_orig, label='Original Data', color='lightgray', alpha=0.7, zorder=1)
        ax1.plot(x_coords_for_smoothed, y_smoothed, label=f'Smoothed (Win={actual_smoothing_window_len})', color='blue', zorder=2)
        
        # Plot initial candidates (before clustering)
        # ax1.scatter(current_peaks_x, current_peaks_y, marker='o', color='pink', s=30, label='Initial Peak Candidates', zorder=3, alpha=0.5)
        # ax1.scatter(current_valleys_x, current_valleys_y, marker='o', color='lightblue', s=30, label='Initial Valley Candidates', zorder=3, alpha=0.5)

        if len(final_peaks_x) > 0:
            ax1.scatter(final_peaks_x, final_peaks_y, marker='^', color='red', s=60, label='Final Peaks (Clustered)', zorder=4)
        if len(final_valleys_x) > 0:
            ax1.scatter(final_valleys_x, final_valleys_y, marker='v', color='green', s=60, label='Final Valleys (Clustered)', zorder=4)
        ax1.legend(loc='upper right')
        ax1.set_ylabel("Flux")

        ax2.plot(x_coords_for_smoothed, derivative1st, label='1st Derivative')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.axhline(deriv1_thresh, color='orange', linestyle=':', label=f'D1 Thresh ({deriv1_thresh:.2e})')
        ax2.axhline(-deriv1_thresh, color='orange', linestyle=':')
        ax2.legend(loc='upper right'); ax2.set_ylabel("1st Deriv")

        ax3.plot(x_coords_for_smoothed, derivative2nd, label='2nd Derivative')
        ax3.axhline(0, color='gray', linestyle='--')
        ax3.legend(loc='upper right'); ax3.set_ylabel("2nd Deriv")
        ax3.set_xlabel("X-coordinate")
        fig.suptitle("Peak Detection with Derivative Method & DBSCAN Clustering", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    return [final_peaks_x.tolist(), final_peaks_y.tolist()], \
           [final_valleys_x.tolist(), final_valleys_y.tolist()]

def filter_extrema_with_clustering(
    initial_peaks_x, initial_peaks_y,
    initial_valleys_x, initial_valleys_y,
    dbscan_eps=0.75,             # DBSCAN epsilon parameter (tune this carefully)
    dbscan_min_samples=5,        # DBSCAN min_samples parameter (tune this)
    n_iterations=1,                # Number of refinement iterations
    min_extrema_for_clustering=10, # Min number of extrema to attempt clustering
    min_extrema_for_spacing_model=3 # Min extrema needed for spacing model fit
):
    """
    Filters peaks and valleys using DBSCAN clustering on a feature space.
    Features: x-coordinate, prominence/depth, spacing deviation.
    Returns two tuples: (filtered_peaks_x, filtered_peaks_y), (filtered_valleys_x, filtered_valleys_y)
    """
    current_peaks_x = np.array(initial_peaks_x)
    current_peaks_y = np.array(initial_peaks_y)
    current_valleys_x = np.array(initial_valleys_x)
    current_valleys_y = np.array(initial_valleys_y)

    # Initial alternation pass
    current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y = \
        _ensure_alternation(current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y)

    for iteration_num in range(n_iterations):
        # --- Filter Peaks ---
        num_current_peaks = len(current_peaks_x)
        if num_current_peaks >= min_extrema_for_clustering:
            peak_features = _calculate_features_for_extrema(
                current_peaks_x, current_peaks_y,
                current_valleys_x, current_valleys_y, # Use current valleys for bracketing
                is_peak=True,
                min_extrema_for_spacing_model=min_extrema_for_spacing_model
            )
            if peak_features.shape[0] > 0 and peak_features.shape[1] == 3:
                scaler_peaks = MinMaxScaler()
                scaled_peak_features = scaler_peaks.fit_transform(peak_features)
                
                # Adjust DBSCAN parameters as needed, possibly separately for peaks/valleys
                dbscan_p = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                peak_labels = dbscan_p.fit_predict(scaled_peak_features)

                # Identify the main cluster (largest non -1 cluster)
                core_point_labels = peak_labels[peak_labels != -1] # Labels of points not marked as outliers
                if len(core_point_labels) > 0:
                    # Find the most frequent label among core points
                    most_common_label = Counter(core_point_labels).most_common(1)[0][0]
                    # Select indices of peaks belonging to this main cluster
                    true_peak_indices = np.where(peak_labels == most_common_label)[0]
                    
                    current_peaks_x = current_peaks_x[true_peak_indices]
                    current_peaks_y = current_peaks_y[true_peak_indices]
                elif len(peak_labels) > 0 : # All points are outliers or no clusters formed
                    current_peaks_x, current_peaks_y = np.array([]), np.array([]) # Remove all
            # If not enough features or calculation failed, keep current peaks for this step
        
        # Ensure alternation after peak filtering, before valley filtering
        current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y = \
            _ensure_alternation(current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y)

        # --- Filter Valleys ---
        num_current_valleys = len(current_valleys_x)
        if num_current_valleys >= min_extrema_for_clustering:
            valley_features = _calculate_features_for_extrema(
                current_valleys_x, current_valleys_y,
                current_peaks_x, current_peaks_y, # Use (potentially filtered) peaks for bracketing
                is_peak=False,
                min_extrema_for_spacing_model=min_extrema_for_spacing_model
            )
            if valley_features.shape[0] > 0 and valley_features.shape[1] == 3:
                scaler_valleys = MinMaxScaler()
                scaled_valley_features = scaler_valleys.fit_transform(valley_features)

                dbscan_v = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                valley_labels = dbscan_v.fit_predict(scaled_valley_features)
                
                core_point_labels = valley_labels[valley_labels != -1]
                if len(core_point_labels) > 0:
                    most_common_label = Counter(core_point_labels).most_common(1)[0][0]
                    true_valley_indices = np.where(valley_labels == most_common_label)[0]
                    
                    current_valleys_x = current_valleys_x[true_valley_indices]
                    current_valleys_y = current_valleys_y[true_valley_indices]
                elif len(valley_labels) > 0: # All points are outliers
                    current_valleys_x, current_valleys_y = np.array([]), np.array([])
        
        # Final alternation for this iteration
        current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y = \
            _ensure_alternation(current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y)
            
    return (list(current_peaks_x), list(current_peaks_y)), \
           (list(current_valleys_x), list(current_valleys_y))
           
           
# --- Visualization Function ---
def visualize_filtering_results(
    x_spectrum_data, y_spectrum_data,
    initial_peaks_x_tuple, initial_peaks_y_tuple,    # These are the raw initial detections
    initial_valleys_x_tuple, initial_valleys_y_tuple, # Before any processing by your filter
    final_peaks_x_list, final_peaks_y_list,          # Output of your filter
    final_valleys_x_list, final_valleys_y_list,
    min_extrema_for_spacing_model_vis=3 # For feature calculation during visualization
):
    """
    Visualizes the spectrum, initial detections, false detections, and final detections.
    Also shows a 3D feature space plot.

    Args:
        x_spectrum_data (list/np.array): X-coordinates of the spectrum.
        y_spectrum_data (list/np.array): Flux values of the spectrum.
        initial_peaks_x_tuple (tuple of lists): Original x-coords of detected peaks.
        initial_peaks_y_tuple (tuple of lists): Original y-coords of detected peaks.
        initial_valleys_x_tuple (tuple of lists): Original x-coords of detected valleys.
        initial_valleys_y_tuple (tuple of lists): Original y-coords of detected valleys.
        final_peaks_x_list (list): Filtered x-coords of peaks.
        final_peaks_y_list (list): Filtered y-coords of peaks.
        final_valleys_x_list (list): Filtered x-coords of valleys.
        final_valleys_y_list (list): Filtered y-coords of valleys.
        min_extrema_for_spacing_model_vis (int): Param for _calculate_features_for_extrema.
    """
    initial_peaks_x = np.array(initial_peaks_x_tuple[0] if isinstance(initial_peaks_x_tuple, tuple) else initial_peaks_x_tuple)
    initial_peaks_y = np.array(initial_peaks_y_tuple[0] if isinstance(initial_peaks_y_tuple, tuple) else initial_peaks_y_tuple)
    initial_valleys_x = np.array(initial_valleys_x_tuple[0] if isinstance(initial_valleys_x_tuple, tuple) else initial_valleys_x_tuple)
    initial_valleys_y = np.array(initial_valleys_y_tuple[0] if isinstance(initial_valleys_y_tuple, tuple) else initial_valleys_y_tuple)
    
    final_peaks_x = np.array(final_peaks_x_list)
    final_peaks_y = np.array(final_peaks_y_list)
    final_valleys_x = np.array(final_valleys_x_list)
    final_valleys_y = np.array(final_valleys_y_list)

    # --- 1. Identify False Detections ---
    # Use a tolerance for floating point comparison if necessary
    # For simplicity, we'll convert to sets of tuples (x,y)
    # Rounding to avoid float precision issues when checking membership
    set_final_peaks = set(zip(np.round(final_peaks_x, decimals=5), np.round(final_peaks_y, decimals=5)))
    set_final_valleys = set(zip(np.round(final_valleys_x, decimals=5), np.round(final_valleys_y, decimals=5)))

    false_peaks_x, false_peaks_y = [], []
    true_initial_peaks_x, true_initial_peaks_y = [], [] # Initial peaks that were kept
    for x, y in zip(initial_peaks_x, initial_peaks_y):
        if (round(x, 5), round(y, 5)) not in set_final_peaks:
            false_peaks_x.append(x)
            false_peaks_y.append(y)
        else:
            true_initial_peaks_x.append(x)
            true_initial_peaks_y.append(y)
    
    false_valleys_x, false_valleys_y = [], []
    true_initial_valleys_x, true_initial_valleys_y = [], [] # Initial valleys that were kept
    for x, y in zip(initial_valleys_x, initial_valleys_y):
        if (round(x, 5), round(y, 5)) not in set_final_valleys:
            false_valleys_x.append(x)
            false_valleys_y.append(y)
        else:
            true_initial_valleys_x.append(x)
            true_initial_valleys_y.append(y)

    # --- 2. 2D Plot: Spectrum with Detections ---
    plt.figure(figsize=(15, 7))
    plt.plot(x_spectrum_data, y_spectrum_data, label='Spectrum Data', color='gray', alpha=0.7, zorder=1)
    
    # Initial Detections (those that were kept, but shown as initial)
    plt.scatter(true_initial_peaks_x, true_initial_peaks_y, color='cyan', marker='o', s=50, label='Initial Peaks (Kept)', zorder=2, alpha=0.6)
    plt.scatter(true_initial_valleys_x, true_initial_valleys_y, color='lime', marker='o', s=50, label='Initial Valleys (Kept)', zorder=2, alpha=0.6)

    # False Detections
    plt.scatter(false_peaks_x, false_peaks_y, color='red', marker='x', s=100, label='False Peaks', zorder=3)
    plt.scatter(false_valleys_x, false_valleys_y, color='magenta', marker='x', s=100, label='False Valleys', zorder=3)

    # Final Detections (can be plotted over the "kept initial" ones for emphasis)
    plt.scatter(final_peaks_x, final_peaks_y, edgecolor='blue', facecolor='none', marker='o', s=120, label='Final Peaks', zorder=4, linewidth=1.5)
    plt.scatter(final_valleys_x, final_valleys_y, edgecolor='green', facecolor='none', marker='o', s=120, label='Final Valleys', zorder=4, linewidth=1.5)
    
    plt.title('Spectrum with Initial, False, and Final Detections')
    plt.xlabel('X-coordinate')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    # --- 3. 3D Feature Space Plot ---
    # We need to calculate features for the initial set of peaks/valleys
    # Apply alternation first to mimic the start of the filtering process
    alt_initial_peaks_x, alt_initial_peaks_y, \
    alt_initial_valleys_x, alt_initial_valleys_y = _ensure_alternation(
        initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y
    )

    if len(alt_initial_peaks_x) > 0 and len(alt_initial_valleys_x) > 0: # Need both for prominence calc
        # Features for initial (alternated) peaks
        features_initial_peaks = _calculate_features_for_extrema(
            alt_initial_peaks_x, alt_initial_peaks_y,
            alt_initial_valleys_x, alt_initial_valleys_y, # Use alternated initial valleys as opposites
            is_peak=True,
            min_extrema_for_spacing_model=min_extrema_for_spacing_model_vis
        )
        if features_initial_peaks.shape[0] > 0:
            scaler_peaks = MinMaxScaler()
            scaled_features_peaks = scaler_peaks.fit_transform(features_initial_peaks)

            # Determine which of these alt_initial_peaks survived
            peak_status = [] # True if kept, False if discarded
            set_final_peaks_coords = set(zip(np.round(final_peaks_x, decimals=5))) # Only x for matching features
            
            # This matching is a bit tricky because _ensure_alternation might merge/remove points.
            # We'll match based on the x-coordinate of the alternated initial peaks.
            current_alt_peaks_set = set(zip(np.round(alt_initial_peaks_x, 5), np.round(alt_initial_peaks_y, 5)))

            for x_alt, y_alt in zip(alt_initial_peaks_x, alt_initial_peaks_y):
                 if (round(x_alt, 5), round(y_alt, 5)) in set_final_peaks:
                     peak_status.append('Kept (Final)') # Was in final_peaks
                 else:
                     peak_status.append('Discarded')
            
            colors_peaks = ['blue' if s == 'Kept (Final)' else 'red' for s in peak_status]

            fig_peaks = plt.figure(figsize=(10, 8))
            ax_peaks = fig_peaks.add_subplot(111, projection='3d')
            # scaled_features_peaks columns: 0:x_coord, 1:prominence, 2:spacing_dev
            scatter_peaks = ax_peaks.scatter(
                scaled_features_peaks[:, 0], # Scaled X
                scaled_features_peaks[:, 1], # Scaled Prominence
                scaled_features_peaks[:, 2], # Scaled Spacing Deviation
                c=colors_peaks,
                marker='o'
            )
            ax_peaks.set_xlabel('Scaled X-coordinate')
            ax_peaks.set_ylabel('Scaled Prominence')
            ax_peaks.set_zlabel('Scaled Spacing Deviation')
            ax_peaks.set_title('3D Feature Space for Initial Peaks (after alternation)')
            # Create dummy scatter artists for legend
            kept_proxy = plt.Line2D([0],[0], linestyle="none", c='blue', marker='o')
            discarded_proxy = plt.Line2D([0],[0], linestyle="none", c='red', marker='o')
            ax_peaks.legend([kept_proxy, discarded_proxy], ['Kept', 'Discarded'], numpoints=1)
            plt.show()

    if len(alt_initial_valleys_x) > 0 and len(alt_initial_peaks_x) > 0: # Need both for depth calc
        # Features for initial (alternated) valleys
        features_initial_valleys = _calculate_features_for_extrema(
            alt_initial_valleys_x, alt_initial_valleys_y,
            alt_initial_peaks_x, alt_initial_peaks_y, # Use alternated initial peaks as opposites
            is_peak=False,
            min_extrema_for_spacing_model=min_extrema_for_spacing_model_vis
        )
        if features_initial_valleys.shape[0] > 0:
            scaler_valleys = MinMaxScaler()
            scaled_features_valleys = scaler_valleys.fit_transform(features_initial_valleys)

            valley_status = []
            current_alt_valleys_set = set(zip(np.round(alt_initial_valleys_x, 5), np.round(alt_initial_valleys_y, 5)))

            for x_alt, y_alt in zip(alt_initial_valleys_x, alt_initial_valleys_y):
                if (round(x_alt, 5), round(y_alt, 5)) in set_final_valleys:
                    valley_status.append('Kept (Final)')
                else:
                    valley_status.append('Discarded')

            colors_valleys = ['green' if s == 'Kept (Final)' else 'magenta' for s in valley_status]

            fig_valleys = plt.figure(figsize=(10, 8))
            ax_valleys = fig_valleys.add_subplot(111, projection='3d')
            scatter_valleys = ax_valleys.scatter(
                scaled_features_valleys[:, 0], # Scaled X
                scaled_features_valleys[:, 1], # Scaled Depth
                scaled_features_valleys[:, 2], # Scaled Spacing Deviation
                c=colors_valleys,
                marker='o'
            )
            ax_valleys.set_xlabel('Scaled X-coordinate')
            ax_valleys.set_ylabel('Scaled Depth')
            ax_valleys.set_zlabel('Scaled Spacing Deviation')
            ax_valleys.set_title('3D Feature Space for Initial Valleys (after alternation)')
            kept_proxy = plt.Line2D([0],[0], linestyle="none", c='green', marker='o')
            discarded_proxy = plt.Line2D([0],[0], linestyle="none", c='magenta', marker='o')
            ax_valleys.legend([kept_proxy, discarded_proxy], ['Kept', 'Discarded'], numpoints=1)
            plt.show()

    
if __name__ == "__main__":
    from math import pi
    import pylab
    
    i = 10000
    x = np.linspace(0,3.7*pi,i)
    y = (0.3*np.sin(x) + np.sin(1.3 * x) + 0.9 * np.sin(4.2 * x) + 0.06 * 
    np.random.randn(i))
    y *= -1
    
    _max, _min = peakdetect(y, x, 750, 0.30)
    xm = [p[0] for p in _max]
    ym = [p[1] for p in _max]
    xn = [p[0] for p in _min]
    yn = [p[1] for p in _min]
    
    plot = pylab.plot(x, y)
    pylab.hold(True)
    pylab.plot(xm, ym, "r+")
    pylab.plot(xn, yn, "g+")
    
    
    pylab.show()
    
    