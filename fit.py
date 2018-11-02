#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:07:32 2018

@author: dmilakov
"""

from harps.core import odr, np, os, tqdm, plt, curve_fit

from harps.constants import c

import harps.settings as hs
import harps.emissionline as emline
import harps.containers as container
import harps.functions as hf

from harps.spectrum import extract_version

#==============================================================================
# Assumption: Frequencies are known with 1MHz accuracy
freq_err = 1e6


#==============================================================================
    
#                           G A P S    F I L E   
    
#==============================================================================
def read_gaps(filepath=None):
    if filepath is not None:
        filepath = filepath  
    else:
        filepath = os.path.join(hs.harps_prod,'gapsA.npy')
    gapsfile = np.load(filepath)    
    #orders   = np.array(gapsfile[:,0],dtype='i4')
    #gaps2d   = np.array(gapsfile[:,1:],dtype='f8')
    return gapsfile
    
    #{"ORDER{od:2d}".format(od=od):gaps1d for od,gaps1d in zip(orders,gaps2d)}


def get_gaps(order,filepath=None):
    gapsfile  = read_gaps(filepath)
    orders   = np.array(gapsfile[:,0],dtype='i4')
    gaps2d   = np.array(gapsfile[:,1:],dtype='f8')
    selection = np.where(orders==order)[0]
    gaps1d    = gaps2d[selection]
    return np.ravel(gaps1d)


def introduce_gaps(centers,gaps1d,npix=4096):
    if np.size(gaps1d)==0:
        return centers
    elif np.size(gaps1d)==1:
        gap  = gaps1d
        gaps = np.full((7,),gap)
    else:
        gaps = gaps1d
    centc = np.copy(centers)
    for i,gap in enumerate(gaps):
        ll = (i+1)*npix/(np.size(gaps)+1)
        cut = np.where(centc>ll)[0]
        centc[cut] = centc[cut]-gap
    return centc
#==============================================================================
#
#                         L I N E      F I T T I N G                  
#
#==============================================================================
default_line = 'SingleGaussian'
def gauss(x,flux,bkg,error,model=default_line,output_model=False,
          *args,**kwargs):
    assert np.size(x)==np.size(flux)==np.size(bkg)
    line_model   = getattr(emline,model)
    line         = line_model()    
    try:
        pars, errors = line.fit(x,flux-bkg,error,bounded=False)
    except:
        plt.figure()
        plt.plot(x,flux-bkg)
        plt.plot(x,error)
    chisq        = line.rchi2
    if output_model:
        model = line.evaluate(pars)
        return pars, errors, chisq, model
    else:
        return pars, errors, chisq
    
    
#==============================================================================
#
#        W A V E L E N G T H     S O L U T I O N      F I T T I N G                  
#
#==============================================================================
def wavesol(linelist,version,fit='gauss'):
    """
    Fits the wavelength solution to the data provided in the linedict.
    Calls 'wavesol1d' for all orders in linedict.
    
    Uses Gaussian profiles as default input.
    
    Returns:
    -------
        wavesol2d : dictionary with coefficients for each order in linedict
        
    """
    orders  = np.unique(linelist['order'])
    polyord, gaps, do_segment = extract_version(version)
    wavesolist = []
    pbar       = tqdm.tqdm(total=len(orders),desc="Wavesol")
#    plt.figure()
#    colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
    for i,order in enumerate(orders):
        linelis1d = linelist[np.where(linelist['order']==order)]
        centers1d = linelis1d[fit][:,1]
        cerrors1d = linelis1d['{fit}_err'.format(fit=fit)][:,1]
        wavelen1d = 1e10*(c/linelis1d['freq'])
        werrors1d = 1e10*(c/linelis1d['freq']**2 * freq_err)
        if gaps:
#            centersold = centers1d
            gaps1d  = get_gaps(order,None)
            centers1d = introduce_gaps(centers1d,gaps1d)
#            plt.scatter(centersold,centers1d-centersold,s=2,c=colors[i])
        else:
            pass
        ws1d      = wavesol1d(centers1d,wavelen1d,
                              cerrors1d,werrors1d,
                              version)
        ws1d['order'] = order
        wavesolist.append(ws1d)
        pbar.update(1)
    wavesol2d = np.hstack(wavesolist)
    return wavesol2d
        
def wavesol1d(centers,wavelengths,cerror,werror,version):
    """
    Uses 'segment' to fit polynomials of degree given by polyord keyword.
    
    
    If version=xx1, divides the data into 8 segments, each 512 pix wide. 
    A separate polyonomial solution is derived for each segment.
    """
    polyord, gaps, do_segment = extract_version(version)
    if do_segment==True:
        numsegs = 8
    else:
        numsegs = 1
            
    npix = 4096
    #numlines = len(centers) 
    seglims  = np.linspace(npix//numsegs,npix,numsegs)
    binned   = np.digitize(centers,seglims)
    
    # new container
    coeffs = container.coeffs(polyord,numsegs)
    for i in range(numsegs):
        sel = np.where(binned==i)
        pars, errs = segment(centers[sel],wavelengths[sel],
                               cerror[sel],werror[sel],polyord)
        
        coeffs[i]['pixl'] = seglims[i]-npix//numsegs
        coeffs[i]['pixr'] = seglims[i]
        coeffs[i]['pars'] = pars
        coeffs[i]['errs'] = errs
    return coeffs
    

def segment(centers,wavelengths,cerror,werror,polyord):
    """
    Fits a polynomial to the provided data and errors.
    Uses scipy's Orthogonal distance regression package in order to take into
    account the errors in both x and y directions.
    
    Returns:
    -------
        coef : len(polyord) array
        errs : len(polyord) array
    """
    if np.size(centers)>polyord:
        # beta0 is the initial guess
        beta0 = np.polyfit(centers,wavelengths,polyord)[::-1]                
        data  = odr.RealData(centers,wavelengths,sx=cerror,sy=werror)
        model = odr.polynomial(order=polyord)
        ODR   = odr.ODR(data,model,beta0=beta0)
        out   = ODR.run()
        pars  = out.beta
        errs  = out.sd_beta
    return pars, errs
        