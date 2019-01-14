#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:07:32 2018

@author: dmilakov
"""

from harps.core import odr, np, os, plt, curve_fit, json, interpolate, leastsq

from harps.constants import c

import harps.settings as hs
import harps.emissionline as emline
import harps.containers as container
import harps.functions as hf

quiet = hs.quiet

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
        filepath = '/Users/dmilakov/harps/dataprod/output/v_0.5.6/' +\
        'gaps/2015-04-17_fibreB_v501_gaps.json'
    with open(filepath,'r') as json_file:
        gaps_file = json.load(json_file)
    return gaps_file['gaps_pix']
    
    

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
        centc[cut] = centc[cut]+gap
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
        chisq        = line.rchi2
        model = line.evaluate(pars)
    except:
#        plt.figure()
#        plt.plot(x,flux-bkg)
#        plt.plot(x,error)
        pars   = np.full(3,np.nan)
        errors = np.full(3,np.nan)
        chisq  = np.nan
        model  = np.full_like(flux,np.nan)
    if output_model:
        
        return pars, errors, chisq, model
    else:
        return pars, errors, chisq
def lsf(pix,flux,background,error,weights,lsf,p0,
        output_model=False,*args,**kwargs):
    def residuals(x0,splr):
        # flux, center
        amp, sft = x0
        model = amp * interpolate.splev(pix+sft,splr)
        
        # sigma_tot^2 = sigma_counts^2 + sigma_background^2
        # sigma_counts = sqrt(counts)     sigma_background = sqrt(background)
        #error = np.sqrt(flux + background)
        resid = np.sqrt(weights) * ((flux-background) - model) / error
        #resid = line_w * (counts- model)
        return resid
    
    splr = interpolate.splrep(lsf['x'],lsf['y'])
    
    popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                                        args=(splr,),
                                        full_output=True)
    
    if ier not in [1, 2, 3, 4]:
        print("Optimal parameters not found: " + errmsg)
        popt = np.full_like(p0,np.nan)
        pcov = None
        success = False
    else:
        success = True
    if success:
        
        sft, flx = popt
        cost = np.sum(infodict['fvec']**2)
        dof  = (len(pix) - len(popt))
        if pcov is not None:
            pcov = pcov*cost/dof
        else:
            pcov = np.array([[np.inf,0],[0,np.inf]])
        #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
    pars   = popt
    errors = np.sqrt(np.diag(pcov))
    chisq  = cost/dof
    model = pars[0]*interpolate.splev(pix+pars[1],splr)+background
#    plt.figure()
#    plt.title('fit.lsf')
#    plt.plot(pix,flux)
#    plt.plot(pix,model)
    if output_model:  
        return pars, errors, chisq, model
    else:
        return pars, errors, chisq
    
#==============================================================================
#
#        W A V E L E N G T H     D I S P E R S I O N      F I T T I N G                  
#
#==============================================================================
def dispersion(linelist,version,fit='gauss'):
    """
    Fits the wavelength solution to the data provided in the linelist.
    Calls 'wavesol1d' for all orders in linedict.
    
    Uses Gaussian profiles as default input.
    
    Returns:
    -------
        wavesol2d : dictionary with coefficients for each order in linelist
        
    """
    orders  = np.unique(linelist['order'])
    polyord, gaps, do_segment = hf.extract_version(version)
    disperlist = []
    if not quiet:
        pbar       = tqdm.tqdm(total=len(orders),desc="Wavesol")
#    plt.figure()
#    colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
    for i,order in enumerate(orders):
        linelis1d = linelist[np.where(linelist['order']==order)]
        centers1d = linelis1d[fit][:,1]
        cerrors1d = linelis1d['{fit}_err'.format(fit=fit)][:,1]
        wavelen1d = hf.freq_to_lambda(linelis1d['freq'])
        werrors1d = 1e10*(c/((linelis1d['freq']*1e9)**2 * 1e6))
        if gaps:
#            centersold = centers1d
            gaps1d  = read_gaps(None)
            centers1d = introduce_gaps(centers1d,gaps1d)
#            plt.scatter(centersold,centers1d-centersold,s=2,c=colors[i])
        else:
            pass
        di1d      = dispersion1d(centers1d,wavelen1d,
                              cerrors1d,werrors1d,
                              version)
        di1d['order'] = order
        disperlist.append(di1d)
        if not quiet:
            pbar.update(1)
    dispersion2d = np.hstack(disperlist)
    return dispersion2d
        
def dispersion1d(centers,wavelengths,cerror,werror,version):
    """
    Uses 'segment' to fit polynomials of degree given by polyord keyword.
    
    
    If version=xx1, divides the data into 8 segments, each 512 pix wide. 
    A separate polyonomial solution is derived for each segment.
    """
    polyord, gaps, do_segment = hf.extract_version(version)
    if do_segment==True:
        numsegs = 8
    else:
        numsegs = 1
            
    npix = 4096
    # remove NaN
    centers     = hf.removenan(centers)
    wavelengths = hf.removenan(wavelengths)
    cerror      = hf.removenan(cerror)
    werror      = hf.removenan(werror)
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
    numcen = np.size(centers)
    assert numcen>polyord, "No. centers too low, {}".format(numcen)
#    plt.figure()
#    plt.errorbar(centers,wavelengths,yerr=werror,xerr=cerror,ms=2,ls='',capsize=4)
#    [plt.axvline(512*i,ls='--',lw=0.3,c='k') for i in range(9)]
    if numcen>polyord:
        # beta0 is the initial guess
        beta0 = np.polyfit(centers,wavelengths,polyord)[::-1]                
        data  = odr.RealData(centers,wavelengths,sx=cerror,sy=werror)
        model = odr.polynomial(order=polyord)
        ODR   = odr.ODR(data,model,beta0=beta0)
        out   = ODR.run()
        pars  = out.beta
        errs  = out.sd_beta
#    plt.plot(centers,np.polyval(pars[::-1],centers))
    return pars, errs
        