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
import harps.gaps as hg
quiet = hs.quiet
version = hs.version
#==============================================================================
# Assumption: Frequencies are known with 1MHz accuracy
freq_err = 1e4


#==============================================================================
    
#                           G A P S    F I L E   
    
#==============================================================================
#def read_gaps(filepath=None):
#    if filepath is not None:
#        filepath = filepath  
#    else:
#        dirpath = hs.get_dirname('gaps')
#        filepath = os.path.join(dirpath,'gaps.json')
#    with open(filepath,'r') as json_file:
#        gaps_file = json.load(json_file)
#    gaps = []
#    for block in range(1,4):
#        orders  = gaps_file['orders{}'.format(block)]
#        norders = len(orders)
#        block_gaps = container.gaps(norders)
#        block_gaps['order'] = orders
#        block_gaps['gaps']  = gaps_file['block{}'.format(block)]
#        gaps.append(block_gaps)
#    gaps = np.hstack(gaps)
#    return np.sort(gaps)
#    
#    
#
#def get_gaps(order,filepath=None):
#    gapsfile  = read_gaps(filepath)
#    orders   = np.array(gapsfile[:,0],dtype='i4')
#    gaps2d   = np.array(gapsfile[:,1:],dtype='f8')
#    selection = np.where(orders==order)[0]
#    gaps1d    = gaps2d[selection]
#    return np.ravel(gaps1d)
#
#
#def introduce_gaps(centers,gaps1d,npix=4096):
#    if np.size(gaps1d)==0:
#        return centers
#    elif np.size(gaps1d)==1:
#        gap  = gaps1d
#        gaps = np.full((7,),gap)
#    else:
#        gaps = gaps1d
#    centc = np.copy(centers)
#    
#    for i,gap in enumerate(gaps):
#        ll = (i+1)*npix/(np.size(gaps)+1)
#        cut = np.where(centc>ll)[0]
#        centc[cut] = centc[cut]-gap
#    return centc
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
        success = True
    except:
#        plt.figure()
#        plt.plot(x,flux-bkg)
#        plt.plot(x,error)
        pars   = np.full(3,np.nan)
        errors = np.full(3,np.nan)
        chisq  = np.nan
        model  = np.full_like(flux,np.nan)
        success = False
    if output_model:
        
        return success, pars, errors, chisq, model
    else:
        return success, pars, errors, chisq
def lsf(pix,flux,background,error,lsf1s,p0,
        output_model=False,*args,**kwargs):
    """
    lsf1d must be an instance of LSF class and contain only one segment 
    (see harps.lsf)
    """
    def assign_weights(pixels):
        weights  = np.zeros_like(pixels)
        binlims  = [-5,-2.5,2.5,5]
        idx      = np.digitize(pix,binlims)
        cut1     = np.where(idx==2)[0]
        cutl     = np.where(idx==1)[0]
        cutr     = np.where(idx==3)[0]
        # ---------------
        # weights are:
        #  = 1,           -2.5<=x<=2.5
        #  = 0,           -5.0>=x & x>=5.0
        #  = linear[0-1]  -5.0<x<2.5 & 2.5>x>5.0 
        # ---------------
        weights[cut1] = 1
        weights[cutl] = 0.4*(5+pixels[cutl])
        weights[cutr] = 0.4*(5-pixels[cutr])
        return weights
    def residuals(x0,lsf1s):
        # flux, center
        #amp, sft = x0
        #sftpix   = pix-sft
        model    = lsf_model(lsf1s,x0,pix)#amp * interpolate.splev(sftpix,splr)
        weights  = np.ones_like(pix)
        #weights  = assign_weights(sftpix)
        resid = np.sqrt(weights) * ((flux-background) - model) / error
        #resid = line_w * (counts- model)
        return resid
    
    #splr = interpolate.splrep(lsf1s.x,lsf1s.y)
    
    popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                                        args=(lsf1s,),
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
    else:
        popt = np.full_like(p0,np.nan)
        pcov = np.array([[np.inf,0],[0,np.inf]])
        cost = np.nan
        dof  = (len(pix) - len(popt))
        success=False
    pars   = popt
    errors = np.sqrt(np.diag(pcov))
    chisq  = cost/dof
    model = lsf_model(lsf1s,pars,pix)#pars[0]*interpolate.splev(pix+pars[1],splr)+background
#    plt.figure()
#    plt.title('fit.lsf')
#    plt.plot(pix,flux)
#    plt.plot(pix,model)
    if output_model:  
        return success, pars, errors, chisq, model
    else:
        return success, pars, errors, chisq
def lsf_model(lsf1s,pars,pix):
    """
    Returns the model of the data from the LSF and parameters provided. 
    Does not include the background.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    splr  = interpolate.splrep(lsf1s.x,lsf1s.y)
    model = pars[0]*interpolate.splev(pix-pars[1],splr)
    return model
#==============================================================================
#
#        W A V E L E N G T H     D I S P E R S I O N      F I T T I N G                  
#
#==============================================================================
def dispersion(linelist,version,fittype='gauss'):
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
    if gaps:
        gaps2d     = hg.read_gaps(None)
    plot=False
    if plot and gaps:
        plt.figure()
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
    linelist0 = hf.remove_bad_fits(linelist,fittype)
    for i,order in enumerate(orders):
        linelis1d = linelist0[np.where(linelist0['order']==order)]
        centers1d = linelis1d[fittype][:,1]
        cerrors1d = linelis1d['{fit}_err'.format(fit=fittype)][:,1]
        wavelen1d = hf.freq_to_lambda(linelis1d['freq'])
        werrors1d = 1e10*(c/((linelis1d['freq'])**2)) * freq_err
        if gaps:
            if plot:
                centersold = centers1d
            cut       = np.where(gaps2d['order']==order)
            gaps1d    = gaps2d[cut]['gaps'][0]
            centers1d = hg.introduce_gaps(centers1d,gaps1d)
            if plot:
                plt.scatter(centersold,centers1d-centersold,s=2,c=[colors[i]])
        else:
            pass
        di1d      = dispersion1d(centers1d,wavelen1d,
                              cerrors1d,werrors1d,
                              version)
        di1d['order'] = order
        disperlist.append(di1d)
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
    #centers     = hf.removenan(centers)
    #wavelengths = hf.removenan(wavelengths)
    #cerror      = hf.removenan(cerror)
    #werror      = hf.removenan(werror)
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
    

def segment(centers,wavelengths,cerror,werror,polyord,plot=False):
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
    arenan = np.isnan(centers)
    centers     = centers[~arenan]
    wavelengths = wavelengths[~arenan]
    cerror      = cerror[~arenan]
    werror      = werror[~arenan]
#    if plot:
#        plt.figure()
#        plt.errorbar(centers,wavelengths,yerr=werror,xerr=cerror,ms=2,ls='',capsize=4)
#        [plt.axvline(512*i,ls='--',lw=0.3,c='k') for i in range(9)]
    clip0 = np.full_like(centers,False,dtype='bool')
    clip1 = np.full_like(centers,True,dtype='bool')
    j = 0
    while not np.sum(clip0)==np.sum(clip1) and j<10:
        j+=1
        clip0        = clip1
        centers0     = centers[clip0]
        wavelengths0 = wavelengths[clip0]
        cerror0      = cerror[clip0]
        werror0      = werror[clip0]
        pars,errs    = poly(centers0,wavelengths0,cerror0,werror0,polyord)
        residuals    = wavelengths-np.polyval(pars[::-1],centers)
        clip1        = hf.sigclip1d(residuals,5)
        
        if plot and np.sum(~clip1)>0:
#            plt.figure()
            #plt.plot(centers[clip1],np.polyval(pars[::-1],centers[clip1]))
            plt.scatter(centers,residuals,s=2)
            plt.scatter(centers[~clip1],residuals[~clip1],s=16,marker='x')
        
    return pars, errs
def poly(centers,wavelengths,cerror,werror,polyord):
    numcen = np.size(centers)
    assert numcen>polyord, "No. centers too low, {}".format(numcen)
    # beta0 is the initial guess
    beta0 = np.polyfit(centers,wavelengths,polyord)[::-1]                
    data  = odr.RealData(centers,wavelengths,sx=cerror,sy=werror)
    model = odr.polynomial(order=polyord)
    ODR   = odr.ODR(data,model,beta0=beta0)
    out   = ODR.run()
    pars  = out.beta
    errs  = out.sd_beta
    return pars, errs