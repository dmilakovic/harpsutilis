#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:26:15 2018

@author: dmilakov
"""
import numpy as np
import logging
import matplotlib.pyplot as plt

from harps.constants import c
from harps.background import getbkg

import harps.settings as hs
import harps.functions as hf
import harps.containers as container
import harps.fit as hfit
import harps.emissionline as emline
import harps.lsf as hlsf
import harps.lines_aux as laux
#from . import curves as curve
import harps.noise as noise

# from numba import jit
import scipy.stats as stats

quiet = hs.quiet
hs.setup_logging()

def _make_extname(order):
    return "ORDER{order:2d}".format(order=order)

def arange_modes_from_coefficients(center1d,coeff1d,reprate,anchor):
    """
    Uses the positions of maxima to assign mode numbers to all lines in the 
    echelle order.
    
    Uses the ThAr coefficients (in vacuum) to calculate the mode of the central 
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
    integer_n = np.rint(decimal_n).astype(np.int32)
    ref_n     = integer_n
    #print("{0:3d}/{1:3d} (pixel={2:8.4f})".format(ref_index,nlines,ref_pixel))
    
    # make a decreasing array of modes, where modes[ref_index]=ref_n:
    aranged  = np.flip(np.arange(nlines))
    shifted  = aranged - (nlines-ref_index-1)
    modes    = shifted+ref_n
    return modes, ref_index

def arange_modes_from_array(center1d,wave1d,reprate,anchor):
    """
    Uses the positions of maxima to assign mode numbers to all lines in the 
    echelle order.
    
    Uses an array of wavelengths to calculate the mode of the central 
    line.
    """
    # warn if ThAr solution does not exist for this order:


    # total number of lines
    nlines = len(center1d)
    # central line
    ref_index = nlines//2
    ref_pixel = center1d[ref_index]
    # calculate frequencies of the central line from ThAr solution
    ref_wave_thar = wave1d[int(ref_pixel)]
    ref_freq_thar = c/ref_wave_thar*1e10
    # convert frequency into mode number
    decimal_n = ((ref_freq_thar - (anchor))/reprate)
    integer_n = np.rint(decimal_n).astype(np.int32)
    ref_n     = integer_n
    #print("{0:3d}/{1:3d} (pixel={2:8.4f})".format(ref_index,nlines,ref_pixel))
    
    # make a decreasing array of modes, where modes[ref_index]=ref_n:
    aranged  = np.flip(np.arange(nlines))
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
    maxima,minima = get_maxmin1d(spec,order)
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


    
    


def detect1d(spec,order,plot=False,fittype=['gauss'],wavescale=['pix','wav'],
             gauss_model='SimpleGaussian',subbkg=hs.subbkg,divenv=hs.divenv,
             lsf=None,lsf_method='gp',lsf_interpolate=True,
             logger=None,debug=False,*args,**kwargs):
    """
    Returns a list of all LFC lines and fit parameters in the specified order.
    """
    log               = logger or logging.getLogger(__name__)
    # LFC keywords
    reprate           = spec.lfckeys['comb_reprate']
    anchor            = spec.lfckeys['comb_anchor']
    if 'anchor_offset' in list(kwargs.keys()):
        offset = kwargs.pop('anchor_offset')
    else:
        offset  = 0
    window            = spec.lfckeys['window_size']
    
    # Make sure fittype and wavescale can be indexed
    fittype   = np.atleast_1d(fittype)
    wavescale = np.atleast_1d(wavescale)
    
    # Data
    data = laux.prepare_data(spec.flux[order],spec.error[order],
                                    spec.envelope[order],spec.background[order],
                                    subbkg=subbkg,divenv=divenv)
    flx_norm, err_norm, bkg_norm = data
    pn_weights        = spec.weights[order]
    background        = bkg_norm
    envelope          = spec.envelope[order]
    # Mode identification 
    maxima ,minima     = hf.detect_maxmin(flx_norm,None,*args,**kwargs)
    maxima_x, maxima_y = maxima
    minima_x, minima_y = minima
    nlines             = len(minima_x)-1
    if debug:
        log.info("Identified {} LFC lines in order {}".format(nlines,order))
    # Plot
    npix = spec.npix
    if plot:
        plt.figure()
        plt.plot(np.arange(npix),flx_norm)
        plt.vlines(minima_x,0,np.max(flx_norm),linestyles=':',
                   linewidths=0.4,colors='C1')
        
    # New data container
    linelist          = container.linelist(nlines)
    linelist['order'] = order
    linelist['optord']= spec.optical_orders[order]
    for i in range(0,nlines,1):
        # mode edges
        # if i>0:
        #     lpix = int((maxima_x[i-1]+maxima_x[i])/2)
        # else:
        #     lpix = 0
        # if i<nlines-1:
        #     rpix = int((maxima_x[i]+maxima_x[i+1])/2)
        # else:
        #     rpix = spec.npix-1
        lpix, rpix = (int(minima_x[i]),int(minima_x[i+1]))
        # barycenter
        pix  = np.arange(lpix,rpix,1)
        flx  = flx_norm[lpix:rpix]
        if rpix-lpix<=4:
            print(i,lpix,rpix)
            continue
        # bary = centroid, flux weighted mean position
        bary = np.average(pix,weights=flx)
        # bmean = flux weighted mean of two brightest pixels
        s = np.argsort(flx)[-2:]
        bmean = np.average(pix[s],weights=flx[s])
        # skewness
        skew = stats.skew(flx,bias=False)
        # CCD segment assignment (pixel space)
        center  = maxima_x[i]
        local_seg = center//spec.segsize
        # photon noise
        sumw = np.sum(pn_weights[lpix:rpix])
        pn   = (c/np.sqrt(sumw))
        # signal to noise ratio
        err = err_norm[lpix:rpix]
        snr = np.sum(flx)/np.sum(err)
        # background
        bkg = background[lpix:rpix]
        
        
        linelist[i]['pixl']   = lpix
        linelist[i]['pixr']   = rpix
        linelist[i]['noise']  = pn
        linelist[i]['sumbkg'] = np.sum(bkg)
        linelist[i]['sumflx'] = np.sum(flx)
        linelist[i]['segm']   = local_seg
        linelist[i]['bary']   = bary
        linelist[i]['bmean']  = bmean
        linelist[i]['skew']   = skew
        linelist[i]['snr']    = snr
        linelist[i]['id']     = get_line_index(linelist[i])
    if debug:
        log.debug("Lines prepared for fitting using {}".format(fittype))
    # dictionary that contains functions for line profile fitting
    fitfunc = dict(gauss=fit_gauss1d)
    fitargs = dict(gauss=(gauss_model,))
    # print('all fine to here')
    if 'lsf' in fittype:   
        if lsf is not None:
            if isinstance(lsf,str):
                lsf_full  = hlsf.from_file(lsf)
            elif isinstance(lsf,object):
                lsf_full  = lsf
        else:
            lsf_full   = hlsf.read_lsf(spec.meta['fibre'],spec.datetime,lsf_method)
        # interpolation=kwargs.pop('interpolation',True)
        #print(interpolation)
        fitfunc['lsf']=fit_lsf1d
        fitargs['lsf']=(lsf_full,lsf_interpolate)
        
    
    for i,ft in enumerate(fittype):
        for j,ws in enumerate(wavescale):
            if ws == 'pix':
                wave  = np.arange(spec.npix)
            elif ws=='wav':
                wave  = spec.wavereference[order]
            linepars = fitfunc[ft](linelist,wave,flx_norm,err_norm,
                                   *fitargs[ft])
            linelist[f'{ft}_{ws}']          = linepars['pars']
            linelist[f'{ft}_{ws}_err']      = linepars['errs']
            linelist[f'{ft}_{ws}_chisq']    = linepars['chisq']
            linelist[f'{ft}_{ws}_chisqnu']  = linepars['chisqnu']
            linelist['success'][:,i*2+j*1]  = linepars['conv']
            linelist[f'{ft}_{ws}_integral'] = linepars['integral']
            
    # print("Fitting of order {} completed ".format(order))
    # arange modes of lines in the order using ThAr coefficients in vacuum
    wave1d = spec.wavereference[order]
    center1d = linelist['gauss_pix'][:,1]
    modes,refline = arange_modes_from_array(center1d,wave1d,
                                            reprate,anchor+offset)
    for i in range(0,nlines,1):
         # mode and frequency of the line
        linelist[i]['mode'] = modes[i]
        linelist[i]['freq'] = anchor + modes[i]*reprate + offset
#        linelist[i]['anchor'] = anchor
#        linelist[i]['reprate'] = reprate
        if plot:
            if i==refline:
                lw = 1; ls = '-'
            else:
                lw = 0.5; ls = '--'
            plt.axvline(center1d[i],c='r',ls=ls,lw=lw) 
     # fit lines   
    return linelist

def detect(spec,order=None,logger=None,debug=False,*args,**kwargs):
    """
    Returns a list of all detected LFC lines in a numpy array defined as 
    linelist in harps.container
    """
    orders = spec.prepare_orders(order)
    output = []
    msg = 'failed'
    for od in orders:  
        try:
            output.append(detect1d(spec,od,debug=debug,logger=logger,
                                   *args,**kwargs))
            msg = 'successful'
        except:
            pass
        if debug:
            log = logger or logging.getLogger(__name__)
            log.info('Order {} {}'.format(od,msg))
    lines2d = np.hstack(output)
    return lines2d

def detect_from_array(data,wave,reprate,anchor,error1d=None,background1d=None,
             weights1d=None,window=3,plot=False,fittype=['gauss','lsf'],
             wavescale=['pix','wav'],
             gauss_model='SingleGaussian',lsf=None,lsf_method='analytic',
             *args,**kwargs):
    """
    Returns a list of all LFC lines and fit parameters in the specified order.
    """
  
    # Make sure fittype is a list
    fittype = np.atleast_1d(fittype)
    wavescale = np.atleast_1d(wavescale)
    
    # Data
    background = background1d if background1d is not None \
                 else getbkg(data,window=window)
    error      = error1d if error1d is not None \
                 else noise.error1d(data,background=background)
    
    pn_weights = noise.weights1d(data)
    
    # Mode identification 
    maxima,minima     = get_maxmin1d(data,xarray=None,
                                     background=background,window=window)
    
    nlines            = len(minima)-1
    # Plot
    pixel = np.arange(len(data))
    if plot:
        plt.figure()
        plt.plot(pixel,data,drawstyle='steps')
        plt.vlines(pixel[maxima],0,np.max(data),linestyles=':',linewidths=0.4,
                   colors='C1')
        
    # New data container
    linelist          = container.linelist(nlines)
#    linelist['order'] = order
#    linelist['optord']= spec.optical_orders[order]
    for i in range(0,nlines,1):
        # mode edges
        lpix, rpix = (minima[i],minima[i+1])
        # barycenter
        pix  = pixel[lpix:rpix]
        flx  = data[lpix:rpix]
        bary = np.sum(flx*pix)/np.sum(flx)
        # skew = hf.nmoment(pix,flx,bary,3)
        x = np.arange(pixl,pixr,1)
        _ = np.where((x>=bary-4.)&(x<=bary+4.))[0]
        skew = stats.skew(flx[_],bias=False)
        # segment
        center  = maxima[i]
#        local_seg = center//spec.segsize
        # photon noise
        sumw = np.sum(pn_weights[lpix:rpix])
        pn   = (c/np.sqrt(sumw))
        # signal to noise ratio
        err = error[lpix:rpix]
        snr = np.sum(flx)/np.sum(err)
        # background
        bkg = background[lpix:rpix]
        linelist[i]['pixl']   = lpix
        linelist[i]['pixr']   = rpix
        linelist[i]['noise']  = pn
        linelist[i]['sumbkg'] = np.sum(bkg)
        linelist[i]['sumflx'] = np.sum(flx)
#        linelist[i]['segm']   = local_seg
        linelist[i]['bary']   = bary
        linelist[i]['skew']   = skew
        linelist[i]['snr']    = snr
    # dictionary that contains functions
    fitfunc = dict(gauss=fit_gauss1d)
    fitargs = dict(gauss=(gauss_model,))
    if 'lsf' in fittype:   
        if lsf is not None:
            if isinstance(lsf,str):
                lsf_full  = hlsf.from_file(lsf)
            elif isinstance(lsf,object):
                lsf_full  = lsf
        else:
            quit("No LSF model provided. Please provide one and retry.")
        interpolation=kwargs.pop('interpolation',True)
        #print(interpolation)
        fitfunc['lsf']=fit_lsf1d
        fitargs['lsf']=(lsf_full,interpolation,lsf_method)
        
    
    for i,ft in enumerate(fittype):
        for j,ws in enumerate(wavescale):
            if ws == 'pix':
                wave_  = np.arange(len(wave))
            elif ws=='wav':
                wave_  = wave
            linepars = fitfunc[ft](linelist,data,wave_,background,error,*fitargs[ft])
            print(linepars)
            linelist[f'{ft}_{ws}']         = linepars['pars']
            linelist[f'{ft}_{ws}_err']     = linepars['errs']
            linelist[f'{ft}_{ws}_chisq']   = linepars['chisq']
            linelist[f'{ft}_{ws}_chisqnu'] = linepars['chisqnu']
            linelist['success'][:,i*2+j*1] = linepars['conv']

    centers = maxima
#    centers = linelist['gauss'][:,1]
    modes,refline = arange_modes_from_array(centers,wave,reprate,anchor)
    for i in range(0,nlines,1):
         # mode and frequency of the line
        linelist[i]['mode'] = modes[i]
        linelist[i]['freq'] = anchor + modes[i]*reprate
    if plot:
        npix = len(data)
        model1d = np.zeros(npix)
        for i in range(0,nlines,1):
            gauss_model  = gauss_model if gauss_model is not None else hfit.default_line
            lineclass    = getattr(emline,gauss_model)
            line         = lineclass()
            pars         = linelist[i]['gauss']
            lpix, rpix   = linelist[i]['pixl'],linelist[i]['pixr']
            pix          = pixel[lpix:rpix]
            if lpix==0:
                lpix = 1
            if rpix==npix:
                rpix = npix-1 
            
            pix2         = pixel[lpix-1:rpix+1]
            model1d[lpix:rpix] = line.evaluate(pars,pix2)
            if i==refline:
                lw = 1.5; ls = '-'; color = 'k'
            else:
                lw = 0.75; ls = '--'; color='r'
            plt.axvline(pixel[centers[i]],c=color,ls=ls,lw=lw) 
        model1d += background
        try:
            plt.plot(pixel,model1d,drawstyle='steps',c='C1')
        except:
            pass
    return linelist
    
def fit(spec,order=None):
    """
    Wrapper around 'detect'. Returns a dictionary.
    """
    return detect(spec,order)
def fit_gauss1d(linelist,wave,data,error,
                line_model='SingleGaussian',
                *args,**kwargs):

    nlines  = len(linelist)
    linepars = container.linepars(nlines)
    
    npix = len(data)
    
    for i,line in enumerate(linelist):
        # mode edges
        lpix, rpix = (line['pixl'],line['pixr'])
        # fit lines     
        # using 'SingleGaussian' class, extend by one pixel in each direction
        # make sure the do not go out of range
        if lpix==0:
            lpix = 1
        if rpix==npix-1:
            rpix = npix-2 
        if line_model=='SingleGaussian':
            pixx = wave[lpix-1:rpix+1]
            flxx = data[lpix-1:rpix+1]
            errx = error[lpix-1:rpix+1]
        else:
            pixx = wave[lpix:rpix]
            flxx = data[lpix:rpix]
            errx = error[lpix:rpix]
        # bkgx = background[lpix-1:rpix+1]
        # envx = envelope[lpix-1:rpix+1]
        fit_result = hfit.gauss(pixx,flxx,errx,line_model,*args,**kwargs)
        success, pars,errs,chisq,chisqnu,integral = fit_result
        linepars[i]['pars'] = pars
        linepars[i]['errs'] = errs
        linepars[i]['chisq']= chisq
        linepars[i]['chisqnu']=chisqnu
        linepars[i]['conv'] = success
        linepars[i]['integral'] = integral
        # print(i,linepars[i])
    return linepars

def fit_gauss1d_minima(minima,data,wave,background,error,line_model='SingleGaussian',
                *args,**kwargs):

    nlines  = len(minima)-1
    linepars = container.linepars(nlines,3)
    npix = len(data)
    for i in range(nlines):
        # mode edges
        lpix, rpix = int(minima[i]), int(minima[i+1])
        print(lpix,rpix)
        # fit lines     
        # using 'SingleGaussian' class, extend by one pixel in each direction
        # make sure the do not go out of range
        if lpix==0:
            lpix = 1
        if rpix==npix-1:
            rpix = npix-2 
        
        pixx = np.arange(lpix-1,rpix+1,1)
        flxx = data[lpix-1:rpix+1]
        errx = error[lpix-1:rpix+1]
        bkgx = background[lpix-1:rpix+1]
        
        success, pars,errs,chisq, chisqnu = hfit.gauss(pixx,flxx,bkgx,errx,
                                     line_model,*args,**kwargs)
#        p0 = (np.max(flxx),np.mean(pixx),np.std(pixx),np.min(flxx),0e0)
#        pars, cov = curve_fit(curve.gauss5p,pixx,flxx,p0=p0,sigma=errx,
#                              absolute_sigma=True)
#        errs      = np.sqrt(np.diag(cov))
#        resid     = (flxx - curve.gauss5p(pixx,*pars))/errx
#        dof       = len(pixx) - len(pars)
#        chisq     = np.sum(resid**2) / dof
#        success   = 1
        linepars[i]['pars'] = pars
        linepars[i]['errs'] = errs
        linepars[i]['chisq']= chisq
        linepars[i]['chisqnu']= chisqnu
        linepars[i]['conv'] = success
    return linepars
def fit_lsf1d(linelist,wave,data,error,lsf,interpolation=False):
    """
    lsf must be an instance of LSF class with all orders and segments present
    (see harps.lsf)
    """
    nlines  = len(linelist)
    linepars = container.linepars(nlines)   
#    plt.figure()
    # print(lsf)
    # assert method in ['analytic','spline','gp']
    for i,line in enumerate(linelist):
        # mode edges
        lpix, rpix = (line['pixl'],line['pixr'])
        flx  = data[lpix:rpix]
        pix  = np.arange(lpix,rpix,1.) 
        err  = error[lpix:rpix]
        # line center
        cent = line['bary']#line[fittype][1]
        # segment
        order = line['order']
        if interpolation:
            lsf1s = lsf.interpolate(order,cent)
            
        else:
            segm  = line['segm']
            # lsf1s = lsf[order,segm]
            lsf1s = lsf.values[segm]
        # initial guess
        p0   = (np.max(flx),cent,1.)
        # success,pars,errs,chisq,chisqnu,integral,model = hfit.lsf(pix-cent,flx,bkg,err,
        #                                   lsf1s,output_model=True)
        try:
            fit_result = hfit.lsf(pix-cent,flx,err,lsf1s,output_model=True)
            success,pars,errs,chisq,chisqnu,integral,model = fit_result
        except:
            success = False
            pars    = np.full_like(p0,np.nan)
            errs    = np.full_like(p0,np.inf)
            chisq   = -1
            chisqnu = -1
            # print('exception',p0,method,lsf1s)
       
        flux, center, wid = pars
        #center = cent - shift
        linepars[i]['pars']   = pars
        linepars[i]['errs']   = errs
        linepars[i]['chisq']  = chisq
        linepars[i]['chisqnu']= chisqnu
        linepars[i]['conv']   = success
        linepars[i]['integral'] = integral
#        plt.figure()
#        plt.plot(pix,model+bkg,c='C1',label='output model')
#        plt.plot(pix,flx,c='C0',label='data')
#    plt.figure()
#    plt.hist(linepars['chisq'],bins=20)
    
        
    #plt.legend()
    return linepars
def get_maxmin1d(yarray,xarray=None,background=None,use='minima',**kwargs):
    """
    Returns the positions of the minima between the LFC lines and the 
    approximated positions of the maxima of the lines.
    """
    window = kwargs.pop('window',3)
    
    assert use in ['minima','maxima']
     
    if xarray is None:
        xarray = np.arange(len(yarray))
    assert np.shape(xarray)==np.shape(yarray)
    
    # determine the positions of minima
    yarray0 = yarray
    if background is not None:
        yarray0 = yarray - background
        
    kwargs = dict(remove_false=kwargs.pop('remove_false',False),
                  method='peakdetect_derivatives',
                  window=window)
    maxima, minima = hf.detect_maxmin(yarray,xarray,window=window)
#    if use=='minima':
#        extreme = 'min'
#    elif use=='maxima':
#        extreme = 'max'
    
#    priext_x,priext_y = hf.peakdet(yarray0,xarray,extreme=extreme,**kwargs)
#    priext = (priext_x).astype(np.int16)
#    secext = ((priext+np.roll(priext,1))/2).astype(np.int16)[1:]
#    if use == 'minima':
#        minima = priext
#        maxima = secext
#    elif use == 'maxima':
#        minima = secext
#        maxima = priext
    return np.array(maxima[0],dtype=int),np.array(minima[0],dtype=int)
def get_maxmin(spec,order,*args,**kwargs):
    """
    Returns the positions of the minima between the LFC lines and the 
    approximated positions of the maxima of the lines.
    """
    # extract arrays
    data = spec.data[order]
    bkg  = spec.background[order]
    pixels = np.arange(spec.npix)
    
    # determine the positions of minima
    yarray = data-bkg
    # kwargs = dict(remove_false=remove_false,
                  # method='peakdetect_derivatives',
                  # window=11)#spec.lfckeys['window_size'])
    
    maxima,minima = hf.detect_maxmin(yarray,pixels,plot=True,*args,**kwargs)
    
    return minima,maxima

def model(spec,fittype,line_model=None,lsf=None,fibre=None,nobackground=False,
          interpolate_lsf=True):
    """
    Default behaviour is to use SingleGaussian class from EmissionLines.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    line_model   = line_model if line_model is not None else hfit.default_line
    linelist     = spec['linelist']
    lineclass    = getattr(emline,line_model)
    numlines     = len(linelist)
    model2d      = np.zeros_like(spec.data)
    bkg2d        = spec.get_background()
    fibre        = fibre if fibre is not None else 'A'
    if fittype == 'lsf':
        if lsf is not None:
            if isinstance(lsf,str):
                lsf  = hlsf.from_file(lsf)
            elif isinstance(lsf,object):
                lsf  = lsf
        else:
            lsf      = hlsf.read_lsf(fibre,spec.datetime,'gp')
    ftype = f'{fittype}_pix'
    for i in range(numlines):
        order = linelist[i]['order']
        pixl  = linelist[i]['pixl']
        pixr  = linelist[i]['pixr']
        segm  = linelist[i]['segm']
        pars  = linelist[i][ftype]
        
        if fittype == 'gauss':
            pix   = np.arange(pixl-1,pixr+1)
            line  = lineclass()
            model2d[order,pixl:pixr] = line.evaluate(pars,pix)
        elif fittype == 'lsf':
            pix = np.arange(pixl,pixr)
            center = pars[1]
            if np.isfinite(center):
                if interpolate_lsf:
                    lsf1s = hlsf.interpolate_local_spline(lsf,order,center)
                else:
                    lsf1s = lsf[order,segm]
                model2d[order,pixl:pixr] = hfit.lsf_model_spline(lsf1s,pars,pix)
            else:
                continue
    if nobackground==False:
        model2d += bkg2d
    return model2d
def model_gauss(spec,*args,**kwargs):
    return model(spec,'gauss',*args,**kwargs)
def model_lsf(spec,*args,**kwargs):
    return model(spec,'lsf',*args,**kwargs)

def select_order(linelist,order):
    if isinstance(order,slice):
        orders = np.arange(order.start,order.stop+1,order.step)
    orders = np.atleast_1d(orders)
    cut = np.isin(linelist['order'],orders)
    return linelist[cut]
def remove_order(linelist,order):
    if isinstance(order,slice):
        orders = np.arange(order.start,order.stop+1,order.step)
    orders = np.atleast_1d(orders)
    cut = np.isin(linelist['order'],orders)
    return linelist[~cut]
def select(linelist,condict):
    condtup = tuple(linelist[key]==val for key,val in condict.items())
    condition = np.logical_and.reduce(condtup) 
    cut = np.where(condition==True)
    return linelist[cut]

def get_line_index(linelist_like):
    fac = 10000
    MOD = 1
    centers = linelist_like['bary']
    orders  = linelist_like['order']*fac
    cround  = np.round(centers/MOD)*MOD
    cint    = np.asarray(cround,dtype=np.int)
    index0  = orders+cint
    return index0

class Linelist(container.Generic):
    def __init__(self,narray):
        self._values = narray
    def __getitem__(self,item):
        condict, segm_sent = self._extract_item(item)
        return self.select(condict)
    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning order plus segment.
        
        To be used with partial decorator
        """
        condict = {}
        segm_sent = False
        if isinstance(item,dict):
            if len(item)==2: segm_sent=True
            
            condict.update(item)
        else:
            dict_sent=False
            if isinstance(item,tuple):
                
                nitem = len(item) 
                if nitem==2:
                    segm_sent=True
                    order,segm = item
                    
                elif nitem==1:
                    segm_sent=False
                    order = item[0]
            else:
                segm_sent=False
                order=item
            condict['order']=order
            if segm_sent:
                condict['segm']=segm
        return condict, segm_sent
    def __len__(self):
        return len(self.values)
    @property
    def values(self):
        return self._values
    def select(self,condict):
        values  = self.values 
        condtup = tuple(values[key]==val for key,val in condict.items())
        
        condition = np.logical_and.reduce(condtup)
        
        cut = np.where(condition==True)
        return Linelist(values[cut])
    
    