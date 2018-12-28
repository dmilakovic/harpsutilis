#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:37:15 2018

@author: dmilakov
"""

from harps.core import np, plt
from harps.constants import c as cc
import harps.functions as hf

def extract_cen_freq(linelist,fittype):
    """
    Returns centers (of fittype), frequencies, and the photon noise of lines
    in the linelist provided.
    """
    return linelist[fittype][:,1], linelist['freq'], linelist['noise']

def interpolate1d(comb1lines,comb2lines,fittype='gauss'):
    """
    Returns the interpolated frequencies and the photon noise of COMB2 lines 
    using the known positions of COMB2 lines to interpolate between COMB1
    lines. Uses the frequencies and positions of COMB1 lines to perform the 
    interpolation.
    """
    def interpolate_freq(x,nx,index):
        """
        Returns the interpolated frequency and the photon noise of a single 
        line. 
        
        Input: 
        -----
            x (float)   : position of COMB2 line in pixels
            nx (flaot)  : photon noise of the COMB2 line
            index (int) : index of the COMB1 line that is to the right of the 
                          COMB2 line 
        """
        
        if index > 0 and index < len(freq1):
            f1 = freq1[index-1]
            x1 = cen1[index-1]
            f2 = freq1[index]
            x2 = cen1[index]
            
            # Two point form of a line passing through (x1,f1) and (x2,f2):
            # f(x) =  f2 + (x-x2)*(f1-f2)/(x1-x2)
            
            f_int = f1 + (x-x1)*(f2-f1)/(x2-x1)
            #print(x-x1,x2-x,x2-x1)
            # Noise is the square root of the sum of variances
            n1 = noise1[index-1]
            n2 = noise1[index]
            noise = np.sqrt(n1*n1 + n2*n2 + nx*nx)
        else:
            f_int = np.nan
            noise = np.nan
        return f_int, noise
# =============================================================================    
    # COMB1 is used for interpolation of COMB2 lines
    cen1, freq1, noise1 = extract_cen_freq(comb1lines,fittype)
    cen2, freq2, noise2 = extract_cen_freq(comb2lines,fittype)
    bins = np.digitize(cen2,cen1,right=False)
    # COMB2 lines are binned into bins defined by the positions of COMB1 lines
    # COMB1: 0       1       2       3       4       5       6       7       8
    # COMB1: x       x       x       x       x       x       x       x       x
    # COMB2:  ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^
    # bins:   1     1     2     3     4     4     5     6     7     7     8
    # this is an array containing the COMB1 indices before which COMB2 lines 
    # should be inserted     
    # interpolate COMB2 frequencies from the positions of COMB1 and COMB2 lines
    freq_int, noise_int = np.transpose([interpolate_freq(c,n,i) \
                                        for c,n,i in zip(cen2,noise2,bins)])
# =============================================================================   
    
    
    
    
    
    #shift = hf.removenan(-cc*(freq_int-freq2)/freq_int)
    #noise = hf.removenan(noise_int)
    
    #print("Shift = {0:10.5f}+-{1:8.5f} m/s".format(*calculate_shift(shift,noise)))
    return freq_int, noise_int

def interpolate2d(comb1lines,comb2lines,fittype='gauss'):
    
    minord = np.min(tuple(np.min(f['order']) for f in [comb1lines,comb2lines]))
    maxord = np.max(tuple(np.max(f['order']) for f in [comb1lines,comb2lines]))
    
    interpolated_freq  = np.full(len(comb2lines),np.nan)
    interpolated_noise = np.full(len(comb2lines),np.nan)
    for order in range(minord,maxord+1,1):
        inord1 = np.where(comb1lines['order']==order)[0]
        inord2 = np.where(comb2lines['order']==order)[0]
        freq_int, noise_int =interpolate1d(comb1lines[inord1],
                              comb2lines[inord2],
                              fittype)
        interpolated_freq[inord2] = freq_int
        interpolated_noise[inord2] = noise_int
    return interpolated_freq, interpolated_noise

def calculate_shift(shift,noise):
    variance = np.power(noise,2)
    weights  = 1./variance 
    rv_mean  = np.nansum(shift * weights) / np.nansum(weights)
    rv_sigma = 1./ np.sqrt(np.sum(weights))
    
    return rv_mean, rv_sigma

def two_spectra(spec,refspec,fittype='gauss'):
    comb1lines = refspec['linelist']
    comb2lines = spec['linelist']
    return two_linelists(comb1lines,comb2lines)
    
def two_linelists(comb1lines,comb2lines,fittype='gauss'):
    true_cent, true_freq, true_noise  = extract_cen_freq(comb2lines,fittype)
    int_freq, int_noise = interpolate2d(comb1lines,comb2lines,fittype)
    
    shift = hf.removenan(-cc*(int_freq-true_freq)/int_freq)
    noise = hf.removenan(int_noise)

    m = hf.sigclip1d(shift,plot=False)
    rv_mean, rv_sigma = calculate_shift(shift[m],noise[m])
    return rv_mean, rv_sigma