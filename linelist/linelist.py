#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:26:15 2018

@author: dmilakov
"""

from harps.core import np
from harps.core import curve_fit, leastsq
from harps.constants import c

import harps.settings as hs
import harps.functions as hf
import harps.containers as container

def _get_line_minmax(spec,order):
    # extract arrays
    data = spec.data[order]
    bkg  = spec.get_background1d(order)
    pixels = np.arange(spec.npix)
    # photon noise
    #sigma_v= orderdata.sel(ax='sigma_v')
    #pn_weights = (sigma_v/299792458e0)**-2
    
    # determine the positions of minima
    yarray     = data-bkg
    minima_df  = hf.peakdet(yarray,pixels,extreme='min',
                        method='peakdetect_derivatives',
                        window=spec.lfckeys['window_size'])
    minima     = (minima_df.x).astype(np.int16)
    # zeroth order approximation: maxima are equidistant from minima
    maxima0 = ((minima+np.roll(minima,1))/2).astype(np.int16)
    # remove 0th element (between minima[0] and minima[-1]) and reset index
    maxima1 = maxima0[1:]
    maxima  = maxima1.reset_index(drop=True)
    
    return minima,maxima
def _arrange_modes(spec,order):
    thar = spec.get_tharsol1d(order)
    err  = spec.get_error1d(order)  

    # warn if ThAr solution does not exist for this order:
    if sum(thar)==0:
        raise UserWarning("ThAr WAVELENGTH SOLUTION DOES NOT EXIST")
        
    
     # LFC keywords
    reprate = spec.lfckeys['comb_reprate']
    anchor  = spec.lfckeys['comb_anchor']

    minima,maxima = _identify_line_positions(spec,order)
    # total number of lines
    nlines = len(maxima)
    # calculate frequencies of all lines from ThAr solution
    maxima_index     = maxima.values
    maxima_wave_thar = thar[maxima_index]
    maxima_freq_thar = c/maxima_wave_thar*1e10
    # closeness is defined as distance of the known LFC mode to the line 
    # detected on the CCD
    
    decimal_n = ((maxima_freq_thar - anchor)/reprate)
    integer_n = np.rint(decimal_n).astype(np.int16)
    closeness = np.abs( decimal_n - integer_n ).values
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
def detect_lines1d(spec,order):#orderdata,f0_comb,reprate,segsize,pixPerLine,window):
    
    
    linelist = container.linelist(nlines)
    #print("N[ref_index]==ref_n",N[ref_index]==ref_n)
    for i in range(0,nlines,1):
        # array of pixels
        lpix, rpix = (minima.x[i],minima.x[i+1])
        linelist[i]['pixl']=lpix
        linelist[i]['pixr']=rpix
        pix  = np.arange(lpix,rpix,1,dtype=np.int32)
        # sometimes the pix array covers more than can fit into the arr container
        # trim it on both sides until it fits
        if len(pix)>pixPerLine:
            k = 0
            while len(pix)>pixPerLine:
                pix = np.arange(lpix+k,rpix-k,dtype=np.int32)
                k+=1
        # flux, background, flux error
        flux = spec1d[pix]
        bkg  = bkg1d[pix]
        err  = err1d[pix]

        # save values
        val  = {'pix':pix, 
                'flx':flux,
                'bkg':bkg,
                'err':err}
        for ax in val.keys():
            idx  = dict(id=i,pid=np.arange(pix.size),ax=ax)
            try:
                arr['line'].loc[idx] = val[ax]
            except:
                print(np.arange(pix.size))
                print(arr['line'].coords['pid'])
        # barycenter, segment
        bary = np.sum(flux*pix)/np.sum(flux)
        center  = maxima.iloc[i]
        #cen_pix = pix[np.argmax(flux)]
        local_seg = center//segsize
        # photon noise
        sumw = np.sum(pn_weights[pix])
        pn   = (299792458e0/np.sqrt(sumw)).values
        # signal to noise ratio
        snr = np.sum(flux)/np.sum(err)
        # frequency of the line
        freq    = f0_comb + N[i]*reprate
        
        arr['attr'].loc[dict(id=i,att='n')]   = N[i]
        arr['attr'].loc[dict(id=i,att='pn')]  = pn
        arr['attr'].loc[dict(id=i,att='freq')]= freq
        arr['attr'].loc[dict(id=i,att='seg')] = local_seg
        arr['attr'].loc[dict(id=i,att='bary')]= bary
        arr['attr'].loc[dict(id=i,att='snr')] = snr
        
        linelist[i]['numb']  = N[i]
        linelist[i]['noise'] = pn
        linelist[i]['freq']  = freq
        linelist[i]['segm']  = local_seg
        linelist[i]['bary']  = bary
        linelist[i]['snr']   = snr
        # calculate weights in a separate function
    # save the total flux in the order
    #print(linelist)
    arr['stat'].loc[dict(od=order,odpar='sumflux')] = np.sum(spec1d)
    return linelist
def detect_lines2(spec,order,calculate_weights=False):
    '''
    Detects lines present in the echelle order.
  
    The output is saved in a FITS file.
    
    '''
    
    # MAIN PART
    orders = self.prepare_orders(order)
#        self.check_and_load_psf()
    wavesol_thar = self.check_and_get_wavesol('thar')
    if self.lineDetectionPerformed==True:    
        lines  = self.lines
        return lines
    else:
        #print('init lines')
        lines = self.check_and_return_lines()
        lines = self.lines
    
    e2ds = self.get_e2ds(orders)

   
    pbar =tqdm.tqdm(total=1,desc="Detecting lines")
#        #outdata = mp_pool.map(detect_order,[e2ds.sel(od=od) for od in orders])
##        outdata = pool.uimap(detect_order,[(e2ds.sel(od=od),self.f0_comb,self.reprate,self.segsize) for od in orders])
#        outdata = Parallel(n_jobs=hs.nproc)(delayed(detect_order)(e2ds.sel(od=od),self.f0_comb,self.reprate,self.segsize,self.pixPerLine) for od in orders)
    pool1 = Pool(hs.nproc)
    outdata = pool1.map(wrap_detect_order,
                        [(e2ds.sel(od=od),self.f0_comb,self.reprate,
                          self.segsize,self.pixPerLine,self.window) \
                          for od in orders])
    pool1.close()
    pool1.join()
    pbar.update(1)
    pbar.close()
    # SAVE TO LINES HDU
    lines_hdu = self.HDU_get('linelist')
    for linelist,od in zip(outdata,orders):
        #linelist_dtype = hf.return_dtype('linelist')
        #modified_linelist = np.asarray(linelist,dtype=linelist_dtype)
        #self._write_HDU(linelist)
        lines_hdu.write(linelist,extname='ORDER{0:2s}'.format(str(od)))
    lines_hdu.close()
        
    #sys.exit()
    #detected_lines = xr.merge(outdata)
   
    #lines['attr'] = detected_lines['attr']
    #lines['line'] = detected_lines['line']
    #lines['stat'] = detected_lines['stat']
    if calculate_weights:
        psf = self.check_and_load_psf()
        pool2 = Pool(hs.nproc)
        weights = pool2.map(wrap_calculate_line_weights,[(lines.sel(od=od),self.psf,self.pixPerLine) for od in orders])
        print('Weights calculated')
        weights = xr.merge(weights)
        pool2.close()
        pool2.join()
        lines['line'].loc[dict(od=orders,ax='wgt')] = weights['line'].sel(od=orders)
    else:
        pass
    self.linelist = self.HDU['linelist']
    
    self.lineDetectionPerformed=True
    gc.collect()
    return self.linelist