#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:26:15 2018

@author: dmilakov
"""

from harps.core import np, pd
from harps.core import curve_fit, leastsq
from harps.core import tqdm, plt
from harps.constants import c

import harps.settings as hs
import harps.io as io
import harps.functions as hf
import harps.containers as container
import harps.fit as hfit
import harps.emissionline as emline

from numba import jit

def _make_extname(order):
    return "ORDER{order:2d}".format(order=order)

def arange_modes(spec,order):
    """
    Uses the positions of maxima to assign mode numbers to all lines in the 
    echelle order.
    """
    thar = spec.get_tharsol1d(order)

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
    
    decimal_n = ((maxima_freq_thar - anchor)/reprate)
    integer_n = np.rint(decimal_n).astype(np.int16)
    closeness = np.abs( decimal_n - integer_n )
    # the line closest to the frequency of an LFC mode is the reference:
    ref_index = int(np.argmin(closeness))
    ref_pixel = int(maxima_index[ref_index])
    ref_n     = int(integer_n[ref_index])
    ref_freq  = anchor + ref_n * reprate
    ref_wave  = c/ref_freq * 1e10
    # make a decreasing array of modes, where modes[ref_index]=ref_n:
    aranged  = np.arange(nlines)[::-1]
    shifted  = aranged - (nlines-ref_index-1)
    modes    = shifted+ref_n
    return modes 

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
    pn_weights        = photon_noise_weights1d(spec,order)
    # Mode identification 
    minima,maxima     = get_minmax(spec,order)
    modes             = arange_modes(spec,order)
    nlines            = len(modes)
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
        # frequency of the line
        freq    = anchor + modes[i]*reprate
        
        linelist[i]['pixl']  = lpix
        linelist[i]['pixr']  = rpix
        linelist[i]['mode']  = modes[i]
        linelist[i]['noise'] = pn
        linelist[i]['freq']  = freq
        linelist[i]['segm']  = local_seg
        linelist[i]['bary']  = bary
        linelist[i]['snr']   = snr
        if plot:
            plt.axvline(center,c='r',ls='--',lw=0.5) 
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
    return linelist

def detect(spec,order=None,*args,**kwargs):
    """
    Returns a dictionary with all LFC lines in the provided spectrum.
    """
    orders = spec.prepare_orders(order)
    pbar   = tqdm.tqdm(total=len(orders),desc='Linelist')
    output = []
    for od in orders:
        #pbar.set_description("Order = {od:2d}".format(od=od))
        output.append(detect1d(spec,od,*args,**kwargs))
        pbar.update(1) 
    lines2d = np.hstack(output)
    return lines2d

def fit1d(spec,order):
    """
    Wrapper around 'detect1d'. Returns a list.
    """
    
    return detect1d(spec,order)
    
def fit(spec,order=None):
    """
    Wrapper around 'detect'. Returns a dictionary.
    """
    return detect(spec,order)
def get_minmax(spec,order,fluxcut=2e2):
    """
    Returns the positions of the minima between the LFC lines and the 
    approximated positions of the maxima of the lines.
    """
    # extract arrays
    data = spec.data[order]
    bkg  = spec.get_background1d(order)
    pixels = np.arange(spec.npix)
    
    # determine the positions of minima
    yarray     = data-bkg
    minima_x,minima_y  = hf.peakdet(yarray,pixels,extreme='min',
                        method='peakdetect_derivatives',
                        window=spec.lfckeys['window_size'])
    minima     = (minima_x).astype(np.int16)
    # zeroth order approximation: maxima are equidistant from minima
    maxima0 = ((minima+np.roll(minima,1))/2).astype(np.int16)
    # remove 0th element (between minima[0] and minima[-1]) and reset index
    maxima = maxima0[1:]
    #maxima  = maxima1.reset_index(drop=True)
    return minima,maxima
def model(spec,fittype='gauss',line_model=None,nobackground=False):
    """
    Default behaviour is to use SingleGaussian class from EmissionLines
    """
    line_model   = line_model if line_model is not None else hfit.default_line
    linelist     = spec['linelist']
    print(linelist)
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
    
def sigmav(spec):
    """
    Calculates the limiting velocity precison of all pixels in the spectrum
    using ThAr wavelengths.
    """
    orders  = np.arange(spec.nbo)
    sigma_v = np.array([sigmav1d(spec,order) for order in orders])
    return sigma_v

def sigmav1d(spec,order):
    """
    Calculates the limiting velocity precison of all pixels in the order
    using ThAr wavelengths.
    """
    data    = spec.data[order]
    thar    = spec.get_tharsol()[order]
    err     = spec.get_error1d(order)
    # weights for photon noise calculation
    # Equation 5 in Murphy et al. 2007, MNRAS 380 p839
    #pix2d   = np.vstack([np.arange(spec.npix) for o in range(spec.nbo)])
    df_dlbd = hf.derivative1d(data,thar)
    sigma_v = c*err/(thar*df_dlbd)
    return sigma_v

def photon_noise_weights1d(spec,order):
    """
    Calculates the photon noise of the order.
    """
    sigma_v = sigmav1d(spec,order)
    return (sigma_v/c)**-2

def photon_noise_weights(spec):
    """
    Calculates the photon noise of the entire spectrum.
    """
    sigma_v = sigmav1d(spec)
    return (sigma_v/c)**-2