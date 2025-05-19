#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:34:45 2025

@author: dmilakov
"""

from fitsio import FITS
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

#%%
filename = '/Users/dmilakov/projects/lfc/data/harps/raw/4bruce/HARPS.2015-04-17T00:00:41.445_lfc.fits'
with FITS(filename,'r') as hdul:
    data_blue = hdul[1].read()
    data_red  = hdul[2].read()
    
#%%

cutx = range(2016,2080)

data_cut = data_red[:,cutx]
plt.imshow(data_cut)

#%%

def peakdetect_derivatives(y_axis, x_axis = None, window_len=None, 
                           bins=10,plot=False, deriv_method='coeff'): 
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
    crossings_ = np.where(np.diff(np.sign(derivative1st)))[0]
    inside     = np.logical_and((crossings_ >= 0),(crossings_ <= len(y_filtered)-1))
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
    maxima = _validate(y_axis,max_peaks_rebinned,
                          kind='max',mindist=mindist)
    minima = _validate(y_axis,min_peaks_rebinned,
                          kind='min',mindist=mindist,
                          verbose=True)
    
    if plot:
        fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,figsize=(6,7))
        
        for ax,label in zip([ax1,ax2,ax3,ax4],
                            ['Data',
                             '1st derivative',
                             '2nd derivative',
                             'Distance between minima']):
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
                       array[max_ind],s=20,marker='^',c='C4')
            ax.scatter(x_rebinned[min_ind],
                       array[min_ind],s=20,marker='v',c='C3')
            ax.axhline(0,ls='--')
        ax1.scatter([_[0] for _ in minima],
                    [_[1] for _ in minima],
                    marker='v',c='C2')
        ax1.scatter([_[0] for _ in maxima],
                    [_[1] for _ in maxima],
                    marker='^',c='C3')
        ax4.scatter([_[0] for _ in minima[:-1]],
                    np.diff([_[0] for _ in minima]))
        
        ax4.scatter([_[0] for _ in maxima[:-1]],
                    np.diff([_[0] for _ in maxima]))
        
        # ax2.set_ylim(np.nanpercentile(derivative1st,[1,99]))
        # ax3.set_ylim(np.nanpercentile(derivative2nd,[1,99]))
        # fig.tight_layout()
    return [maxima, minima]


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
    

def _datacheck_peakdetect(x_array, y_array):
    y_shape = np.shape(y_array)
    print(y_shape)
    assert len(y_shape)>-1 and len(y_shape)<=2, 'Array can be 1d or 2d'
    if x_array is None:
        if len(y_shape)==1:
            x_array = np.arange(len(y_array))
        elif len(y_shape)==2:
            x_array = np.tile(np.arange(y_shape[0]),(y_shape[1],1)).T
    
    if np.shape(y_array) != np.shape(x_array):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")
    
    #needs to be a numpy array
    y_array = np.array(y_array)
    x_array = np.array(x_array)
    return x_array, y_array
    

def _smooth(x, window_len=11, window="hanning", mode="valid"):
    """
    Smooth the data using a window of the requested size.
    
    This method is based on the convolution of a scaled window on the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    
    Parameters:
    x -- the input signal (1D or 2D array)
    window_len -- the dimension of the smoothing window; should be an odd
        integer (default: 11)
    window -- the type of window from 'flat', 'hanning', 'hamming', 
        'bartlett', 'blackman', where flat is a moving average
        (default: 'hanning')
    mode -- convolution mode: 'full', 'valid', or 'same' (default: 'valid')
    
    Returns:
    Smoothed signal (same shape as input).
    """
    if x.ndim == 1:
        return _smooth_1d(x, window_len, window, mode)
    elif x.ndim == 2:
        return np.array([_smooth_1d(row, window_len, window, mode) for row in x])
    else:
        raise ValueError("smooth only accepts 1D or 2D arrays.")

def _smooth_1d(x, window_len, window, mode):
    """
    Helper function to smooth 1D data.
    """
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    if window_len < 3:
        return x

    # Declare valid windows in a dictionary
    window_funcs = {
        "flat": lambda _len: np.ones(_len, "d"),
        "hanning": np.hanning,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
    }
    
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    try:
        w = window_funcs[window](window_len)
    except KeyError:
        raise ValueError(
            "Window is not one of {0}".format(", ".join(window_funcs.keys()))
        )
    
    y = signal.convolve2d(w / w.sum(), s, mode=mode)
    
    # Truncate the output to match the input size for 'valid' mode
    if mode == "valid":
        start = window_len // 2
        end = -start if window_len % 2 == 0 else -(start + 1)
        y = y[start:end]
    
    return y

def _smooth_2d(x, window_len, window="hanning", mode="valid"):
    """
    Helper function to smooth 2D data.
    """
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    if window_len < 3:
        return x

    # Declare valid windows in a dictionary
    window_funcs = {
        "flat": lambda _len: np.ones(_len, "d"),
        "hanning": np.hanning,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
    }
    
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    try:
        w_ = window_funcs[window](window_len)
        w  = np.outer(w_,w_)
    except KeyError:
        raise ValueError(
            "Window is not one of {0}".format(", ".join(window_funcs.keys()))
        )
    
    y = signal.convolve2d(w / w.sum(), s, mode=mode)
    
    # Truncate the output to match the input size for 'valid' mode
    if mode == "valid":
        start = window_len // 2
        end = -start if window_len % 2 == 0 else -(start + 1)
        y = y[start:end]
    
    return y

def _rebin( a, newshape ):
    '''Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]