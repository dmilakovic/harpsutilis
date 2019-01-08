#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:26:15 2018

@author: dmilakov
"""

from harps.core import np, pd
from harps.core import curve_fit, leastsq
from harps.core import plt
from harps.constants import c

import harps.settings as hs
import harps.io as io
import harps.functions as hf
import harps.containers as container
import harps.fit as hfit
import harps.emissionline as emline

from numba import jit

quiet = hs.quiet

def _make_extname(order):
    return "ORDER{order:2d}".format(order=order)

def arange_modes(center1d,coeff1d,reprate,anchor):
    """
    Uses the positions of maxima to assign mode numbers to all lines in the 
    echelle order.
    
    Uses the ThAr wavelength calibration to calculate the mode of the central 
    line.
    """
    
    # warn if ThAr solution does not exist for this order:
    if np.all(coeff1d)==0:
        raise UserWarning("ThAr WAVELENGTH SOLUTION DOES NOT EXIST")

    # total number of lines
    nlines = len(center1d)
    # central line
    ref_index = nlines//2
    ref_pixel = center1d[ref_index]
    # calculate frequencies of the central line from ThAr solution
    ref_wave_thar = hf.polynomial(ref_pixel,*coeff1d)
    ref_freq_thar = c/ref_wave_thar*1e10
    # convert frequency into mode number
    decimal_n = ((ref_freq_thar - (anchor))/reprate)
    integer_n = np.rint(decimal_n).astype(np.int16)
    ref_n     = integer_n
    #print("{0:3d}/{1:3d} (pixel={2:8.4f})".format(ref_index,nlines,ref_pixel))
    
    # make a decreasing array of modes, where modes[ref_index]=ref_n:
    aranged  = np.arange(nlines)[::-1]
    shifted  = aranged - (nlines-ref_index-1)
    modes    = shifted+ref_n
    return modes, ref_index

def arange_modes_by_closeness(spec,order):
    """
    Uses the positions of maxima to assign mode numbers to all lines in the 
    echelle order. 
    
    Looks for the line that is 'closest' to the expected wavelength of a mode,
    and uses this line to set the scale for the entire order.
    """
    thar = spec.tharsol[order]
    # warn if ThAr solution does not exist for this order:
    if sum(thar)==0:
        raise UserWarning("ThAr WAVELENGTH SOLUTION DOES NOT EXIST")
        
    
     # LFC keywords
    reprate = spec.lfckeys['comb_reprate']
    anchor  = spec.lfckeys['comb_anchor']
    minima,maxima = get_minmax(spec,order)
    # total number of lines
    nlines = len(maxima)
    # calculate frequencies of all lines from ThAr solution
    maxima_index     = maxima
    maxima_wave_thar = thar[maxima_index]
    maxima_freq_thar = c/maxima_wave_thar*1e10
    # closeness is defined as distance of the known LFC mode to the line 
    # detected on the CCD
    
    decimal_n = ((maxima_freq_thar - (anchor))/reprate)
    integer_n = np.rint(decimal_n).astype(np.int16)
    closeness = np.abs( decimal_n - integer_n )
    # the line closest to the frequency of an LFC mode is the reference:
    ref_index = int(np.argmin(closeness))
    ref_pixel = int(maxima_index[ref_index])
    ref_n     = int(integer_n[ref_index])
    print(ref_index,'\t',nlines)
    ref_freq  = anchor + ref_n * reprate
    ref_wave  = c/ref_freq * 1e10
    # make a decreasing array of modes, where modes[ref_index]=ref_n:
    aranged  = np.arange(nlines)[::-1]
    shifted  = aranged - (nlines-ref_index-1)
    modes    = shifted+ref_n
    return modes, ref_index
def detect1d(spec,order,plot=False,line_model='SingleGaussian',*args,**kwargs):
    """
    Returns a list of all LFC lines and fit parameters in the specified order.
    """
    # LFC keywords
    reprate = spec.lfckeys['comb_reprate']
    anchor  = spec.lfckeys['comb_anchor']
    
    data              = spec.data[order]
    error             = spec.get_error1d(order)
    background        = spec.get_background1d(order)
    pn_weights        = spec.weights1d(order)
    # Mode identification 
    minima,maxima     = get_minmax(spec,order)
    
    nlines            = len(maxima)
    
    # Plot
    if plot:
        plt.figure()
        plt.plot(np.arange(4096),data)
        
    # New data container
    linelist          = container.linelist(nlines)
    linelist['order'] = order
    for i in range(0,nlines,1):
        # mode edges
        lpix, rpix = (minima[i],minima[i+1])
        # barycenter
        pix  = np.arange(lpix,rpix,1)
        flx  = data[lpix:rpix]
        bary = np.sum(flx*pix)/np.sum(flx)
        # segment
        center  = maxima[i]
        local_seg = center//spec.segsize
        # photon noise
        sumw = np.sum(pn_weights[lpix:rpix])
        pn   = (c/np.sqrt(sumw))
        # signal to noise ratio
        err = error[lpix:rpix]
        snr = np.sum(flx)/np.sum(err)
        # background
        bkg = background[lpix:rpix]
               
        linelist[i]['pixl']  = lpix
        linelist[i]['pixr']  = rpix
        linelist[i]['noise'] = pn
        linelist[i]['segm']  = local_seg
        linelist[i]['bary']  = bary
        linelist[i]['snr']   = snr
        
        # fit lines     
        # using 'SingleGaussian' class, extend by one pixel in each direction
        # make sure the do not go out of range
        if lpix==0:
            lpix = 1
        if rpix==4095:
            rpix = 4094 
        
        pixx = np.arange(lpix-1,rpix+1,1)
        flxx = data[lpix-1:rpix+1]
        errx = error[lpix-1:rpix+1]
        bkgx = background[lpix-1:rpix+1]
        
        pars,errs,chisq = hfit.gauss(pixx,flxx,bkgx,errx,
                                     line_model,*args,**kwargs)
        
        linelist[i]['gauss']     = pars
        linelist[i]['gauss_err'] = errs
        linelist[i]['gchisq']    = chisq
    # arange modes  
    coeffs2d = spec.ThAr.coeffs
    coeffs1d = np.ravel(coeffs2d['pars'][order])
    center1d = linelist['gauss'][:,1]
    modes,refline = arange_modes(center1d,coeffs1d,reprate,anchor)
    for i in range(0,nlines,1):
         # mode and frequency of the line
        linelist[i]['mode'] = modes[i]
        linelist[i]['freq'] = anchor + modes[i]*reprate
#        linelist[i]['anchor'] = anchor
#        linelist[i]['reprate'] = reprate
        if plot:
            if i==refline:
                lw = 1; ls = '-'
            else:
                lw = 0.5; ls = '--'
            plt.axvline(center1d[i],c='r',ls=ls,lw=lw) 
    return linelist

def detect(spec,order=None,*args,**kwargs):
    """
    Returns a list of all detected LFC lines in a numpy array defined as 
    linelist in harps.container
    """
    orders = spec.prepare_orders(order)
    if not quiet:
        pbar   = tqdm.tqdm(total=len(orders),desc='Linelist')
    output = []
    for od in orders:
        #pbar.set_description("Order = {od:2d}".format(od=od))
        output.append(detect1d(spec,od,*args,**kwargs))
        if not quiet:
            pbar.update(1) 
    lines2d = np.hstack(output)
    return lines2d

def fit1d(spec,order):
    """
    Wrapper around 'detect1d'. Returns a numpy array defined as linelist in 
    harps.container.
    """
    
    return detect1d(spec,order)
    
def fit(spec,order=None):
    """
    Wrapper around 'detect'. Returns a dictionary.
    """
    return detect(spec,order)
def get_minmax(spec,order,use='minima'):
    """
    Returns the positions of the minima between the LFC lines and the 
    approximated positions of the maxima of the lines.
    """
    assert use in ['minima','maxima']
    # extract arrays
    data = spec.data[order]
    bkg  = spec.get_background1d(order)
    pixels = np.arange(spec.npix)
    
    # determine the positions of minima
    yarray = data-bkg
    kwargs = dict(remove_false=True,
                  method='peakdetect_derivatives',
                  window=spec.lfckeys['window_size'])
    if use=='minima':
        extreme = 'min'
    elif use=='maxima':
        extreme = 'max'
    
    priext_x,priext_y = hf.peakdet(yarray,pixels,extreme=extreme,**kwargs)
    priext = (priext_x).astype(np.int16)
    secext = ((priext+np.roll(priext,1))/2).astype(np.int16)[1:]
    if use == 'minima':
        minima = priext
        maxima = secext
    elif use == 'maxima':
        minima = secext
        maxima = priext
    return minima,maxima
def model(spec,fittype='gauss',line_model=None,nobackground=False):
    """
    Default behaviour is to use SingleGaussian class from EmissionLines
    """
    line_model   = line_model if line_model is not None else hfit.default_line
    linelist     = spec['linelist']
    lineclass    = getattr(emline,line_model)
    numlines     = len(linelist)
    model2d  = np.zeros_like(spec.data)
    for i in range(numlines):
        order = linelist[i]['order']
        pixl = linelist[i]['pixl']
        pixr = linelist[i]['pixr']
        pars = linelist[i][fittype]
        pix  = np.arange(pixl-1,pixr+1)
        line = lineclass()
        
        model2d[order,pixl:pixr] = line.evaluate(pars,pix)
    if nobackground==False:
        bkg2d    = spec.get_background()
        model2d += bkg2d
    return model2d
def model_gauss(spec,*args,**kwargs):
    return model(spec,*args,**kwargs)
    