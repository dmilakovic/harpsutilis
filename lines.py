#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:26:15 2018

@author: dmilakov
"""

from harps.core import np, pd
from harps.core import curve_fit, leastsq
from harps.core import plt, interpolate
from harps.constants import c

import harps.settings as hs
import harps.io as io
import harps.functions as hf
import harps.containers as container
import harps.fit as hfit
import harps.emissionline as emline
import harps.lsf as hlsf
import harps.curves as curve

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
    integer_n = np.rint(decimal_n).astype(np.int32)
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
def detect1d(spec,order,plot=False,fittype=['gauss','lsf'],
             gauss_model='SingleGaussian',lsf=None,*args,**kwargs):
    """
    Returns a list of all LFC lines and fit parameters in the specified order.
    """
    # LFC keywords
    reprate = spec.lfckeys['comb_reprate']
    anchor  = spec.lfckeys['comb_anchor']
    offset  = spec.lfckeys['comb_offset']
    
    # Make sure fittype is a list
    fittype = np.atleast_1d(fittype)
    
    # Data
    data              = spec.data[order]
    error             = spec.get_error1d(order)
    background        = spec.get_background1d(order)
    pn_weights        = spec.get_weights1d(order)
    
    # Mode identification 
    minima,maxima     = get_minmax(spec,order,
                                   remove_false=kwargs.pop('remove_false',True))
    
    nlines            = len(maxima)
    
    # Plot
    if plot:
        plt.figure()
        plt.plot(np.arange(4096),data)
        
    # New data container
    linelist          = container.linelist(nlines)
    linelist['order'] = order
    linelist['optord']= spec.optical_orders[order]
    for i in range(0,nlines,1):
        # mode edges
        lpix, rpix = (minima[i],minima[i+1])
        # barycenter
        pix  = np.arange(lpix,rpix,1)
        flx  = data[lpix:rpix]
        bary = np.sum(flx*pix)/np.sum(flx)
        skew = hf.nmoment(pix,flx,bary,3)
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
        linelist[i]['skew']  = skew
        linelist[i]['snr']   = snr
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
            lsf_full   = hlsf.read_lsf(spec.meta['fibre'],spec.datetime)
        fitfunc['lsf']=fit_lsf1d
        fitargs['lsf']=(lsf_full,)
    
    for i,ft in enumerate(fittype):
        linepars = fitfunc[ft](linelist,data,background,error,*fitargs[ft])
        linelist['{}'.format(ft)]           = linepars['pars']
        linelist['{}_err'.format(ft)]       = linepars['errs']
        linelist['{}chisq'.format(ft[0])]   = linepars['chisq']
        linelist['{}chisqnu'.format(ft[0])] = linepars['chisqnu']
        linelist['success'][:,i]            = linepars['conv']
        
    # arange modes  
    coeffs2d = spec.ThAr.coeffs
    coeffs1d = np.ravel(coeffs2d['pars'][order])
    center1d = linelist['gauss'][:,1]
    modes,refline = arange_modes(center1d,coeffs1d,reprate,anchor)
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

def detect(spec,order=None,*args,**kwargs):
    """
    Returns a list of all detected LFC lines in a numpy array defined as 
    linelist in harps.container
    """
    orders = spec.prepare_orders(order)
    output = []
    for od in orders:
        output.append(detect1d(spec,od,*args,**kwargs))
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
def fit_gauss1d(linelist,data,background,error,line_model='SingleGaussian',
                *args,**kwargs):

    nlines  = len(linelist)
    linepars = container.linepars(nlines)
    
    for i,line in enumerate(linelist):
        # mode edges
        lpix, rpix = (line['pixl'],line['pixr'])
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
        
        success, pars,errs,chisq,chisqnu = hfit.gauss(pixx,flxx,bkgx,errx,
                                     line_model,*args,**kwargs)
        
        linepars[i]['pars'] = pars
        linepars[i]['errs'] = errs
        linepars[i]['chisq']= chisq
        linepars[i]['chisqnu']=chisqnu
        linepars[i]['conv'] = success
    
    return linepars

def fit_gauss1d_minima(minima,data,background,error,line_model='SingleGaussian',
                *args,**kwargs):

    nlines  = len(minima)-1
    linepars = container.linepars(nlines,3)
    
    for i in range(nlines):
        # mode edges
        lpix, rpix = int(minima[i]), int(minima[i+1])
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
def fit_lsf1d(linelist,data,background,error,lsf,interpolation=True):
    """
    lsf must be an instance of LSF class with all orders and segments present
    (see harps.lsf)
    """
    nlines  = len(linelist)
    linepars = container.linepars(nlines)   
#    plt.figure()
    for i,line in enumerate(linelist):
        # mode edges
        lpix, rpix = (line['pixl'],line['pixr'])
        flx  = data[lpix:rpix]
        pix  = np.arange(lpix,rpix,1.) 
        bkg  = background[lpix:rpix]
        err  = error[lpix:rpix]
        # line center
        cent = line['bary']#line[fittype][1]
        # segment
        order = line['order']
        if interpolation:
            lsf1s = lsf.interpolate(order,cent)
        else:
            segm  = line['segm']
            lsf1s = lsf[order,segm]
        # initial guess
        p0   = (np.max(flx),cent,1.)
        try:
            success,pars,errs,chisq,chisqnu,model = hfit.lsf(pix,flx,bkg,err,
                                              lsf1s,p0,output_model=True)
        except:
            success = False
            pars    = np.full_like(p0,np.nan)
            errs    = np.full_like(p0,np.inf)
            chisq   = -1
            chisqnu = -1
        flux, center, wid = pars
        #center = cent - shift
        linepars[i]['pars']   = pars
        linepars[i]['errs']   = errs
        linepars[i]['chisq']  = chisq
        linepars[i]['chisqnu']= chisqnu
        linepars[i]['conv']   = success
#        plt.figure()
#        plt.plot(pix,model+bkg,c='C1',label='output model')
#        plt.plot(pix,flx,c='C0',label='data')
#    plt.figure()
#    plt.hist(linepars['chisq'],bins=20)
    
        
    #plt.legend()
    return linepars
def get_minmax1d(yarray,xarray=None,background=None,use='minima',**kwargs):
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
        
    kwargs = dict(remove_false=kwargs.pop('remove_false',True),
                  method='peakdetect_derivatives',
                  window=window)
    if use=='minima':
        extreme = 'min'
    elif use=='maxima':
        extreme = 'max'
    
    priext_x,priext_y = hf.peakdet(yarray0,xarray,extreme=extreme,**kwargs)
    priext = (priext_x).astype(np.int16)
    secext = ((priext+np.roll(priext,1))/2).astype(np.int16)[1:]
    if use == 'minima':
        minima = priext
        maxima = secext
    elif use == 'maxima':
        minima = secext
        maxima = priext
    return minima,maxima
def get_minmax(spec,order,use='minima',remove_false=True):
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
    kwargs = dict(remove_false=remove_false,
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
            lsf      = hlsf.read_lsf(fibre,spec.datetime)
    for i in range(numlines):
        order = linelist[i]['order']
        pixl  = linelist[i]['pixl']
        pixr  = linelist[i]['pixr']
        segm  = linelist[i]['segm']
        pars  = linelist[i][fittype]
        
        if fittype == 'gauss':
            pix   = np.arange(pixl-1,pixr+1)
            line  = lineclass()
            model2d[order,pixl:pixr] = line.evaluate(pars,pix)
        elif fittype == 'lsf':
            pix = np.arange(pixl,pixr)
            center = pars[1]
            if np.isfinite(center):
                if interpolate_lsf:
                    lsf1s = hlsf.interpolate_local(lsf,order,center)
                else:
                    lsf1s = lsf[order,segm]
                model2d[order,pixl:pixr] = hfit.lsf_model(lsf1s,pars,pix)
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

class Linelist(object):
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
    
    