#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:37:15 2018

@author: dmilakov
"""

from harps.core import np, plt
from harps.constants import c as cc
import harps.functions as hf
import harps.fit as fit
import harps.wavesol as ws

def get_index(linelist,fittype='gauss'):
    fac = 10000
    MOD = 2.
    
    centers = linelist[fittype][:,1]
    orders  = linelist['order']*fac
    cround  = np.around(centers/MOD)*MOD
    cint    = np.asarray(cround,dtype=np.int)
    index0  = orders+cint
    return index0

def get_sorted(index1,index2):
    #print('len indexes',len(index1),len(index2))
    # lines that are common for both spectra
    intersect=np.intersect1d(index1,index2)
    intersect=intersect[intersect>0]

    indsort=np.argsort(intersect)
    
    argsort1=np.argsort(index1)
    argsort2=np.argsort(index2)
    
    sort1 =np.searchsorted(index1[argsort1],intersect)
    sort2 =np.searchsorted(index2[argsort2],intersect)
    
    return argsort1[sort1],argsort2[sort2]
def extract_cen_freq(linelist,fittype):
    """
    Returns centers (of fittype), frequencies, and the photon noise of lines
    in the linelist provided.
    """
    return linelist[fittype][:,1], linelist['freq'], linelist['noise']

def interpolate1d(comb1lines,comb2lines,fittype,returns='freq'):
    """
    Returns the interpolated frequencies/centres and the photon noise of COMB2 
    lines using the known positions of COMB2 lines to interpolate between COMB1
    lines. Uses the frequencies and positions of COMB1 lines to perform the 
    interpolation.
    """
    def interpolate_freq(x,nx,index):
        """
        Returns the interpolated frequency and the photon noise of a single 
        line. 
        
        Input: 
        -----
            x (float)   : central position of COMB2 line in pixels
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
            # 2019/01/02: changed + to - for testing purposes, return to +
            f_int = f1 + (x-x1)*(f2-f1)/(x2-x1)
            #print(x-x1,x2-x,x2-x1)
            # Noise is the square root of the sum of variances times the
            # derivation of function w.r.t each variable
            n1 = (1 - (x-x1)/(x2-x1))*noise1[index-1]
            n2 = (x-x1)/(x2-x1)*noise1[index]
            noise = np.sqrt(n1*n1 + n2*n2 + nx*nx)
        else:
            f_int = np.nan
            noise = np.nan
        return f_int, noise
    def interpolate_cen(f,nx,index):
        """
        Returns the interpolated centre and the photon noise of a single 
        line. 
        
        Input: 
        -----
            f (float)   : frequency of COMB2 line in Hz
            nx (flaot)  : photon noise of the COMB2 line
            index (int) : index of the COMB1 line that is to the right of the 
                          COMB2 line 
        """
        
        if index > 0 and index < len(freq1):
            f1 = freq1[index-1]
            x1 = cen1[index-1]
            f2 = freq1[index]
            x2 = cen1[index]
            
            if f>f1 or f<f2:
                #print(" FREQUENCY OUT OF RANGE")
                x_int = np.nan
                noise = np.nan
            else:
                # Two point form of a line passing through (x1,f1) and (x2,f2):
                # x(f) =  x1 + (f-f1)*(x2-x1)/(f2-f1)
                # 2019/01/02: changed + to - for testing purposes, return to +
                x_int = x1 + (f-f1)*(x2-x1)/(f2-f1)
                #print(x-x1,x2-x,x2-x1)
                # Noise is the square root of the sum of variances times the
                # derivation of function w.r.t each variable
                n1 = (1 - (f-f1)/(f2-f1))*noise1[index-1]
                n2 = (f-f1)/(f2-f1)*noise1[index]
                noise = np.sqrt(n1*n1 + n2*n2 + nx*nx)
        else:
            x_int = np.nan
            noise = np.nan
        if x_int<0:
            print((6*('{:14.4e}')).format(x1,x_int,x2,f1,f,f2))
        return x_int, noise
    
    
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
    
# =============================================================================   
    
    if returns=='freq':
        function = interpolate_freq
        values   = cen2
    elif returns=='centre':
        function = interpolate_cen
        values   = freq2
    # interpolate COMB2 frequencies from the positions of COMB1 and COMB2 lines
    intval, intnoise = np.transpose([function(v,n,i) \
                                        for v,n,i in zip(values,noise2,bins)])
    
    #shift = hf.removenan(-cc*(freq_int-freq2)/freq_int)
    #noise = hf.removenan(noise_int)
    
    #print("Shift = {0:10.5f}+-{1:8.5f} m/s".format(*calculate_shift(shift,noise)))
    return intval, intnoise

def interpolate2d(comb1lines,comb2lines,fittype,returns='freq'):
    
    minord = np.max(tuple(np.min(f['order']) for f in [comb1lines,comb2lines]))
    maxord = np.min(tuple(np.max(f['order']) for f in [comb1lines,comb2lines]))
    
    interpolated_vals  = np.full(len(comb2lines),np.nan)
    interpolated_noise = np.full(len(comb2lines),np.nan)
    for order in range(minord,maxord,1):
        inord1 = np.where(comb1lines['order']==order)[0]
        inord2 = np.where(comb2lines['order']==order)[0]
        intval, intnoise = interpolate1d(comb1lines[inord1],
                              comb2lines[inord2],
                              fittype,
                              returns)
        interpolated_vals[inord2] = intval
        interpolated_noise[inord2] = intnoise
    return interpolated_vals, interpolated_noise

def global_shift(shift,noise,sig,plot=False,verbose=False):
    n     = np.where(np.abs(shift)<2.99792458e8)
    
    shift0 = shift[n]
    noise0 = noise[n]
    m     = hf.sigclip1d(shift0,sig,plot=plot)
    shift1 = shift0[m]
    noise1 = noise0[m]
    variance = np.power(noise1,2)
    weights  = 1./variance 
    if verbose:
        argsort = np.argsort(shift1)
        for a, s,w in zip(argsort,shift1[argsort],weights[argsort]):
            print((4*("{:12.4e}")).format(a,s,w, s*w))
    mean  = np.nansum(shift1 * weights) / np.nansum(weights)
    sigma = 1./ np.sqrt(np.sum(weights))
    
    return mean, sigma

def two_spectra(spec,refspec,fittype,sigma):
    comb1lines = refspec['linelist']
    comb2lines = spec['linelist']
    return interpolate(comb1lines,comb2lines,fittype,sigma)
def get_unit(array):
    minexponent = np.nanmin(np.floor(np.log10(array))).astype(int)
    #print(minexponent)
    if minexponent==14:
        unit = 'Hz'
    elif minexponent==8:
        unit = 'MHz'
    elif minexponent==3:
        unit = 'Angstrom'
    elif minexponent==-7:
        unit = 'm'
    elif minexponent>=0 and minexponent<4:
        unit = 'pixel'
    else:
        unit = 'unknown'
    return unit
def from_coefficients(linelist,coeffs,fittype,version,sig,**kwargs):
    sig1d = np.atleast_1d(sig)
    data  = ws.residuals(linelist,coeffs,fittype=fittype,version=version)
    shift = data['residual_mps']
    noise = data['noise']
    res = np.vstack([global_shift(shift,noise,sig,**kwargs) for sig in sig1d])
    return res
    

def interpolate(comb1lines,comb2lines,fittype,sig,use='freq',**kwargs):
    sig1d = np.atleast_1d(sig)
    true_cent, true_freq, true_noise  = extract_cen_freq(comb2lines,fittype)
    intvals, intnoise = interpolate2d(comb1lines,comb2lines,fittype,use)
    if np.any(intvals<0):
        cut = np.where(intvals<0)
        print(cut, comb2lines[cut],intvals[cut])
    unit = get_unit(intvals)
    if unit == 'Hz' or unit=='MHz':
        shift = hf.removenan(-cc*(intvals-true_freq)/intvals)
        noise = hf.removenan(intnoise)
    elif unit == 'pixel':
        cutnan = ~np.isnan(intvals)
        #wave0 = hf.freq_to_lambda(true_freq[cutnan])
        # 'transplant' interpolated centers into linelist of COMB2
        comb2int = np.copy(comb2lines)#[cutnan]
        comb2int[fittype][:,1] = intvals#[cutnan]
        # if coefficients were not given, calculate from COMB1
        version = kwargs.pop('version',501)
        coeffs  = kwargs.pop('coeffs',None)
        if coeffs is None:
            coeffs = fit.dispersion(comb1lines,version,fittype)
        
        wave0  = ws.evaluate2d(coeffs,comb2int[cutnan],fittype)
        wave1  = ws.evaluate2d(coeffs,comb2lines[cutnan],fittype)
        shift  = hf.removenan(cc*(wave1-wave0)/wave0)
        cutinv = np.where(np.abs(shift)<cc)[0]
        shift  = shift[cutinv]
        noise  = intnoise[cutnan][cutinv]
    else:
        raise ValueError("Stopping. Unit {}.".format(unit))
    #noise = hf.removenan(intnoise)
    #m = hf.sigclip1d(shift,**kwargs)
    #print(noise[m])
    #return shift, noise
    res = np.vstack([global_shift(shift,noise,sig,**kwargs) for sig in sig1d])
    return res

def wavesolutions(wavesol1, wavesol2, sig,**kwargs):
    sig1d = np.atleast_1d(sig)
    ws1   = ws.Wavesol(wavesol1)
    ws2   = ws.Wavesol(wavesol2)
    
    diff  = ws2/ws1
    
    shift = hf.ravel(diff.values)
    m     = np.where(shift!=0)[0]
    shift = shift[m]
    noise = np.ones_like(shift)
    res = np.vstack([global_shift(shift,noise,sig,**kwargs) for sig in sig1d])
#    print(res)
#    mean, sigma = res.T
    
    return res