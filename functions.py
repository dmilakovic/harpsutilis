#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:43:28 2018

@author: dmilakov
"""
import numpy as np
import pandas as pd
import xarray as xr
import sys
import os

from harps import peakdetect as pkd
from harps import emissionline as emline
from harps import settings as hs

from scipy.special import erf, wofz, gamma, gammaincc, expn
from scipy.optimize import minimize, leastsq, curve_fit
from scipy.integrate import quad

from glob import glob

from matplotlib import pyplot as plt
#from kapteyn import kmpfit

__version__   = hs.__version__
# some shared lists for 'return_empty_dataset' and 'return_empty_dataarray'

################################################################################################################
########################################## F U N C T I O N S ###################################################
################################################################################################################
def accuracy(w=None,SNR=10,dx=829,u=0.9):
    '''
    Returns the rms accuracy of a spectral line with SNR=10, 
    pixel size = 829 m/s and apsorption strength 90%.
    
    Equation 4 from Cayrel 1988 "Data Analysis"
    
    Input: line width in pixels
    '''
    if w is None:
        raise ValueError("No width specified")
    epsilon = 1/SNR
    return np.sqrt(2)/np.pi**0.25 * np.sqrt(w*dx)*epsilon/u
def equivalent_width(SNR,wave,R,dx):
    ''' Cayrel formula 6
    SNR : rms of the signal to noise of the line
    wave: wavelength of the line
    R   : resolution at wavelength
    dx  : pixel size (in A)
    '''
    FWHM = wave / R
    epsilon=1/SNR
    return 1.5*np.sqrt(FWHM*dx)*epsilon
def PN_Murphy(R,SNR,FWHM):
    '''R = resolution, SNR = peak signal/noise, FWHM = FWHM of line [pix]'''
    FWHM_inst = 2.99792458e8/R 
    dv = 0.41 * FWHM_inst / (SNR * np.sqrt(FWHM))
    return dv
def min_equivalent_width(n,FWHM,SNR):
    return n*FWHM/SNR
def min_SNR(n,FWHM,EW):
    return n*FWHM/EW
def schechter_cdf(m,A=1,beta=2,m0=100,mmin=10,mmax=None,npts=1e4):
    """
    Return the CDF value of a given mass for a set mmin,mmax
    mmax will default to 10 m0 if not specified
    
    Analytic integral of the Schechter function:
        x^-a + exp (-x/m) 
    http://www.wolframalpha.com/input/?i=integral%28x^-a+exp%28-x%2Fm%29+dx%29
    """
    if mmax is None:
        mmax = 10*m0
    
    # integrate the CDF from the minimum to maximum 
    # undefined posint = -m0 * mmax**-beta * (mmax/m0)**beta * scipy.special.gammainc(1-beta, mmax/m0)
    # undefined negint = -m0 * mmin**-beta * (mmin/m0)**beta * scipy.special.gammainc(1-beta, mmin/m0)
    posint = -mmax**(1-beta) * expn(beta, mmax/m0)
    negint = -mmin**(1-beta) * expn(beta, mmin/m0)
    tot = posint-negint

    # normalize by the integral
    # undefined ret = (-m0 * m**-beta * (m/m0)**beta * scipy.special.gammainc(1-beta, m/m0)) / tot
    ret = (-m**(1-beta) * expn(beta, m/m0) - negint)/ tot

    return ret 
def schechter(x,norm,alpha,cutoff):
    return norm*((x/cutoff)**alpha)*np.exp(-x/cutoff)
def schechter_int(x,norm,alpha,cutoff):
    return norm*cutoff**(-alpha+1)*gamma(alpha+1)*gammaincc(alpha+1,x/cutoff)
def delta_x(z1,z2):
    ''' Returns total absorption path between redshifts z1 and z2'''
    def integrand(z):
        return (1+z)**2/np.sqrt(0.3*(1+z)**3+0.7)
    return quad(integrand,z1,z2)
def calc_lambda(x,dx,order,wavl):
    pol=wavl[order.astype(int),:]
    return pol[:,0]+pol[:,1]*x+pol[:,2]*x**2+pol[:,3]*x**3,(pol[:,1]+2*pol[:,2]*x+3*pol[:,3]*x**2)*dx

def chisq(params,x,data,weights=None):
    amp, ctr, sgm = params
    if weights==None:
        weights = np.ones(x.shape)
    fit    = gauss3p(x,amp,ctr,sgm)
    chisq  = ((data - fit)**2/weights).sum()
    return chisq
def combine_line_list(theoretical,measured):
    ''' UNUSED'''
    combined = []
    lim = 0.7*np.median(np.diff(theoretical))
    for value in theoretical:
        distances = np.abs(measured-value)
        closest   = distances.min()
        if closest <= lim:
            combined.append(measured[distances.argmin()])
        else:
            combined.append(value)
    return np.array(combined)
def cut_patch(df,i):
    ''' Returns a Pandas Series with the values of wavelengths in patch i'''
    pix = df['pixel']
    cut = np.where((pix>=i*512)&(pix<(1+i)*512))[0]
    print(cut)
    return df.iloc[cut]
def derivative1d(y,x,n=1,method='central'):
    ''' Get the first derivative of a 1d array y. 
    Modified from 
    http://stackoverflow.com/questions/18498457/
    numpy-gradient-function-and-numerical-derivatives'''
    def _contains_nan(array):
        return np.any(np.isnan(array))
    contains_nan = [_contains_nan(array) for array in [y,x]]
    
    if any(contains_nan)==True:
        return np.zeros_like(y)
    else:
        pass
    if method=='forward':
        dx = np.diff(x,n)
        dy = np.diff(y,n)
        d  = dy/dx
    if method == 'central':
        z1  = np.hstack((y[0], y[:-1]))
        z2  = np.hstack((y[1:], y[-1]))
        dx1 = np.hstack((0, np.diff(x)))
        dx2 = np.hstack((np.diff(x), 0))  
        if np.all(np.asarray(dx1+dx2)==0):
            dx1 = dx2 = np.ones_like(x)/2
        d   = (z2-z1) / (dx2+dx1)
    return d
def double_gaussN_erf(x,params):
    if type(x) == pd.Series:
        x = x.values
    else:
        pass
    N = params.shape[0]
    y = np.zeros_like(x,dtype=np.float)
    xb = (x[:-1]+x[1:])/2
    gauss1 = params[['amplitude1','center1','sigma1']]
    gauss2 = params[['amplitude2','center2','sigma2']]
    for i in range(N):
        A1, mu1, sigma1 = gauss1.iloc[i]
        A2, mu2, sigma2 = gauss2.iloc[i]
        
        e11 = erf((xb[:-1]-mu1)/(np.sqrt(2)*sigma1))
        e21 = erf((xb[1:] -mu1)/(np.sqrt(2)*sigma1))
        y[1:-1] += A1*sigma1*np.sqrt(np.pi/2)*(e21-e11)
        
        e12 = erf((xb[:-1]-mu2)/(np.sqrt(2)*sigma2))
        e22 = erf((xb[1:] -mu2)/(np.sqrt(2)*sigma2))
        y[1:-1] += A2*sigma2*np.sqrt(np.pi/2)*(e22-e12)
        
        
    return y
def figure(*args, **kwargs):
    return get_fig_axes(*args, **kwargs)
def basename_to_datetime(filename):
    ''' 
    Extracts the datetime of HARPS observations from the filename
    Args:
    -----
        filename - str or list
    Returns:
    -----
        datetime - np.datetime64 object or a list of np.datetime64 objects
    '''
    filenames = to_list(filename)
    datetimes = []
    for fn in filenames:
        bn = os.path.basename(fn)[:29]
        dt = np.datetime64(bn.split('.')[1].replace('_',':')) 
        datetimes.append(dt)
    if len(datetimes)==1:
        return datetimes[0]
    else:
        return datetimes
def find_nearest(array1,array2):
    ''' UNUSED''' 
    idx = []
    lim = np.median(np.diff(array1))
    for value in array1:
        distances = np.abs(array2-value)
        closest   = distances.min()
        if closest <= lim:
            idc = distances.argmin()
            idx.append(idc)
        else:
            continue
    return array2[idx]
def fit_peak(i,xarray,yarray,yerr,weights,xmin,xmax,dx,order,method='kmpfit',
             model=None,verbose=0):
    '''
    Returns the parameters of the fit for the i-th peak of a single echelle 
    order.
    
    Args:
        xarray:   pixels of the echelle order
        yarray:   flux of the echelle order
        weigths:  weights of individual pixels
        xpos:     positions of individual peaks, determined by get_extreme(max)
        dx:       distances between individual peaks (i.e. np.diff(xpos))
        model:    Gaussian function
        method:   fitting method, default: curve_fit
        
    Returns:
        params:   parameters returned by the fitting procedure
        covar:    covariance matrix of parameters
    '''
    def calculate_photon_noise(weights):
        # FORMULA 10 Bouchy
        return 1./np.sqrt(weights.sum())*299792458e0
        
    # Prepare output array
    # number of parameters
    model = model if model is not None else 'singlegaussian'
    if model=='singlegaussian':
        n = 3
        model_class = emline.SingleGaussian
    elif model=='doublegaussian':
        n = 6
        model_class = emline.DoubleGaussian
    elif model=='simplegaussian':
        n=6
        model_class = emline.SimpleGaussian
    #print(model)
#    dtype = np.dtype([('pars',np.float64,(n,)),
#                      ('errors',np.float64,(n,)),
#                      ('pn',np.float64,(1,)),
#                      ('r2',np.float64,(1,)),
#                      ('cen',np.float64,(1,)),
#                      ('cen_err',np.float64,(1,))])
    coords  = ['amp','cen','sig','amp_err','cen_err','sig_err','chisq','pn']
    results = np.full(len(coords),np.nan)
    arr     = return_empty_dataset(order)
#    results = xr.DataArray(np.full((8,),np.nan),
#a                           coords={'par':coords},
#                           dims='par')
    # Fit only data between the two adjacent minima of the i-th peak
    if i<np.size(xmin)-1:
        cut = xarray.loc[((xarray>=xmin[i])&(xarray<=xmin[i+1])&(yarray>=0))].index
    else:
        #print("Returning results")
        return results

    # If this selection is not an empty set, fit the Gaussian profile
    if verbose>0:
        print("LINE:{0:<5d} cutsize:{1:<5d}".format(i,np.size(cut)))
    if cut.size>6:
        x    = xarray.iloc[cut]#.values
        y    = yarray.iloc[cut]#.values
        ye   = yerr.iloc[cut]
        # barycenter
        b    = np.sum(x*y)/np.sum(y)
        wght = weights[cut]
        wght = wght/wght.sum()
        pn   = calculate_photon_noise(wght)
        ctr  = xmax[i]
        amp  = np.max(yarray.iloc[cut])
        sgm  = 3*dx[i]
        if method == 'curve_fit':              
            guess                          = [amp, ctr, sgm] 
            #print("{} {},{}/{} {} {}".format(order,scale,i,npeaks,guess,cut.size))
            try:
                best_pars, pcov                = curve_fit(model, 
                                                          x, y, 
                                                          p0=guess)
            except:
                return ((-1.0,-1.0,-1.0),np.nan)

        elif method == 'chisq':
            params                      = [amp, ctr, sgm] 
            result                      = minimize(chisq,params,
                                                   args=(x,y,wght))
            best_pars                      = result.x
            

        elif method == 'erfc':
            line   = model_class(x,y,weights=ye)
            if verbose>1:
                print("LINE{0:>5d}".format(i),end='\t')
            try:
                pars, errors = line.fit(bounded=True)
                center       = line.center
                center_error = line.center_error
                rsquared     = line.calc_R2()
                rchi2        = line.rchi2
            except:
                pars, errors = np.full((2,3),np.nan)
                rchi2        = np.inf
            if verbose>1:
                print("ChiSq:{0:<10.5f} R2:{1:<10.5f}".format(line.rchi2,line.R2()))
            if verbose>2:
                columns = ("A1","m1","s1","A2","m2","s2")
                print("LINE{0:>5d}".format(i),(6*"{:>20s}").format(*columns))
                print("{:>9}".format(''),(6*"{:>20.6e}").format(*pars))
            #            line.plot()
#            sys.exit()
        elif method == 'epsf':
            pass
        else:
            sys.exit("Method not recognised!")
        results[0]    = pars[0]
        results[1]    = pars[1]
        results[2]    = pars[2]
        results[3]    = errors[0]
        results[4]    = errors[1]
        results[5]    = errors[2]
        results[6]    = rchi2
        results[7]    = pn
        
        cen = pars[1]
        cen_err = errors[1]
        flx = pars[0]
        flx_err = errors[0]
        freq = np.nan
        
        #fitPars       = ['cen','shift','cen_err','flx','flx_err','chisq','lbd','rsd']

        pars = np.array([cen,np.nan,cen_err,flx,flx_err,
                         chisq,np.nan,np.nan,np.nan])
    return results
    #return np.concatenate((best_pars,np.array([pn])))
def wrap_fit_peak_gauss(pars):
    return fit_peak_gauss(*pars)
def fit_peak_gauss(lines,order,line_id,method='erfc',
             model=None,pixPerLine=22,verbose=0):
    '''
    Returns the parameters of the fit for the i-th peak of a single echelle 
    order.
    
    Args:
        xarray:   pixels of the echelle order
        yarray:   flux of the echelle order
        weigths:  weights of individual pixels
        xpos:     positions of individual peaks, determined by get_extreme(max)
        dx:       distances between individual peaks (i.e. np.diff(xpos))
        model:    Gaussian function
        method:   fitting method, default: curve_fit
        
    Returns:
        params:   parameters returned by the fitting procedure
        covar:    covariance matrix of parameters
    '''
    def calculate_photon_noise(weights):
        # FORMULA 10 Bouchy
        return 1./np.sqrt(weights.sum())*299792458e0
    def my_model(p, x):
       #-----------------------------------------------------------------------
       # This describes the model and its parameters for which we want to find
       # the best fit. 'p' is a sequence of parameters (array/list/tuple).
       #-----------------------------------------------------------------------
       A, mu, sigma = p
       return( A * numpy.exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma))  )
    
    
    def my_residuals(p, data):
       #-----------------------------------------------------------------------
       # This function is the function called by the fit routine in kmpfit
       # It returns a weighted residual. De fit routine calculates the
       # square of these values.
       #-----------------------------------------------------------------------
       x, y, err = data
       return (y-my_model(p,x)) / err
    
    
    def my_derivs(p, data, dflags):
       #-----------------------------------------------------------------------
       # This function is used by the fit routine to find the values for
       # the explicit partial derivatives. Argument 'dflags' is a list
       # with booleans. If an element is True then an explicit partial
       # derivative is required.
       #-----------------------------------------------------------------------
       x, y, err = data
       A, mu, sigma, zerolev = p
       pderiv = numpy.zeros([len(p), len(x)])  # You need to create the required array
       sig2 = sigma*sigma
       sig3 = sig2 * sigma
       xmu  = x-mu
       xmu2 = xmu**2
       expo = numpy.exp(-xmu2/(2.0*sig2))
       fx = A * expo
       for i, flag in enumerate(dflags):
          if flag:
             if i == 0:
                pderiv[0] = expo
             elif i == 1:
                pderiv[1] = fx * xmu/(sig2)
             elif i == 2:
                pderiv[2] = fx * xmu2/(sig3)
       pderiv /= -err
       return pderiv
   
    # Prepare output array
    # number of parameters
    model = model if model is not None else 'simplegaussian'
    
    if model=='singlegaussian':
        n = 3
        model_class = emline.SingleGaussian
    elif model=='singlesimplegaussian':
        n = 3
        model_class = emline.SingleSimpleGaussian
    elif model=='doublegaussian':
        n = 6
        model_class = emline.DoubleGaussian
    elif model=='simplegaussian':
        n=6
        model_class = emline.SimpleGaussian
    
    
    par_arr     = return_empty_dataarray('pars',order,pixPerLine)
    mod_arr     = return_empty_dataarray('model',order,pixPerLine)
    
    # MAIN PART 
    # select single line
    lid       = line_id
    line      = lines.sel(id=lid).dropna('pid','all')
    pid       = line.coords['pid']
    line_x    = line['line'].sel(ax='pix')
    line_y    = line['line'].sel(ax='flx')
    line_yerr = line['line'].sel(ax='err')
    line_bkg  = line['line'].sel(ax='bkg')
 
    # If this selection is not an empty set, fit the Gaussian profile
    if verbose>0:
        print("LINE:{0:<5d} cutsize:{1:<5d}".format(line_id,np.size(line_x)))
    if line_x.size>6:
        
        if method == 'curve_fit':              
            guess                          = [amp, ctr, sgm] 
            #print("{} {},{}/{} {} {}".format(order,scale,i,npeaks,guess,cut.size))
            try:
                best_pars, pcov                = curve_fit(model, 
                                                          x, y, 
                                                          p0=guess)
            except:
                return ((-1.0,-1.0,-1.0),np.nan)

        elif method == 'kmpfit':
            print(method)
            fitobj = kmpfit.Fitter(residuals=my_residuals, deriv=my_derivs, 
                                   data=(line_x.values,
                                         (line_y-line_bkg).values, 
                                         line_yerr.values))
            try:
                fitobj.fit(params0=p0)
            except:
                mes = fitobj.message
                print("Something wrong with fit: ", mes)
                raise SystemExit
            pars    = fitobj.params
            errors  = fitobj.xerror
            rchi2   = fitobj.rchi2_min
            line_model = my_model(pars,line_x)+line_bkg
            print(rchi2)
        elif method == 'erfc':
            # calculate weights by normalising to 1
            
            eline   = model_class(xdata=line_x.values,
                                  ydata=(line_y-line_bkg).values,
                                  yerr=line_yerr.values,
                                  weights=None)
            if verbose>1:
                print("LINE{0:>5d}".format(i),end='\t')
            
            try:
                pars, errors = eline.fit(bounded=False)
                center       = eline.center
                center_error = eline.center_error
                rsquared     = eline.calc_R2()
                rchi2        = eline.rchi2
            except:
                pars, errors = np.full((2,3),np.nan)
                rchi2        = np.inf
            if verbose>1:
                print("ChiSq:{0:<10.5f} R2:{1:<10.5f}".format(eline.rchi2,eline.R2()))
            if verbose>2:
                columns = ("A1","m1","s1","A2","m2","s2")
                print("LINE{0:>5d}".format(i),(6*"{:>20s}").format(*columns))
                print("{:>9}".format(''),(6*"{:>20.6e}").format(*pars))
            #            line.plot()
#            sys.exit()
            line_model = eline.evaluate(pars,clipx=False) + line_bkg
        elif method == 'epsf':
            pass
        else:
            sys.exit("Method not recognised!")
        
        flx, cen, sigma = pars
        flx_err, cen_err, sigma_err = errors
        
        
        # pars: ['cen','cen_err','flx','flx_err','sigma','sigma_err','chisq']
        pars = np.array([cen,cen_err,flx,flx_err,sigma,sigma_err,rchi2])

        par_arr.loc[dict(od=order,id=lid,ft='gauss')] = pars
        mod_arr.loc[dict(od=order,id=lid,pid=pid,ft='gauss')] = line_model
    return par_arr,mod_arr
def flatten_list(inlist):
    outlist = [item for sublist in inlist for item in sublist]
    return outlist
def ravel(array,removenan=True):
    a = np.ravel(array)
    if removenan:
        a = a[~np.isnan(a)]
    return a
def gauss4p(x, amplitude, center, sigma, y0 ):
    # Four parameters: amplitude, center, width, y-offset
    #y = np.zeros_like(x,dtype=np.float64)
    #A, mu, sigma, y0 = p
    y = y0+ amplitude*np.exp((-((x-center)/sigma)**2)/2.)
    return y
def gauss3p(x, amplitude, center, sigma):
    # Three parameters: amplitude, center, width
    #y = np.zeros_like(x)
    #A, mu, sigma = p
    y = amplitude*np.exp((-((x-center)/sigma)**2)/2.)
    return y
def gaussN(x, params):
    N = params.shape[0]
    # Three parameters: amplitude, center, width
    y = np.zeros_like(x)
    #A, mu, sigma = p
    for i in range(N):
        a,c,s,pn,ct = params.iloc[i]
        y = y + a*np.exp((-((x-c)/s)**2)/2.)
    return y
def gaussN_erf(x,params):
    if type(x) == pd.Series:
        x = x.values
    else:
        pass
    N = params.shape[0]
    y = np.zeros_like(x,dtype=np.float)
    xb = (x[:-1]+x[1:])/2
    for i in range(N):
        A,mu,sigma,A_error,mu_error,sigma_error,pn,ct = params.iloc[i]
        sigma = np.abs(sigma)
        e1 = erf((xb[:-1]-mu)/(np.sqrt(2)*sigma))
        e2 = erf((xb[1:] -mu)/(np.sqrt(2)*sigma))
        y[1:-1] += A*sigma*np.sqrt(np.pi/2)*(e2-e1)
    return y
def get_dirname(filetype,dirname=None):
    if dirname is not None:
        dirname = dirname
    else:
        dirname = hs.dirnames[filetype]
    print("DIRNAME = ",dirname)
    direxists = os.path.isdir(dirname)
    if not direxists:
        raise ValueError("Directory does not exist")
    else:
        return dirname
    
def get_fig_axes(naxes,ratios=None,title=None,sep=0.05,alignment="vertical",
                 figsize=(16,9),sharex=None,sharey=None,grid=None,
                 subtitles=None,presentation=False,enforce_figsize=False,
                 left=0.1,right=0.95,top=0.95,bottom=0.10,**kwargs):
    
    def get_grid(alignment,naxes):
        if alignment=="grid":
            ncols = np.int(round(np.sqrt(naxes)))
            nrows,lr = [np.int(k) for k in divmod(naxes,round(np.sqrt(naxes)))]
            if lr>0:
                nrows += 1     
        elif alignment=="vertical":
            ncols = 1
            nrows = naxes
        elif alignment=="horizontal":
            ncols = naxes
            nrows = 1
        grid = np.array([ncols,nrows],dtype=int)
        return grid
    
    fig         = plt.figure(figsize=figsize)
    if enforce_figsize:
        fig.set_size_inches(figsize)
    # Change color scheme and text size if producing plots for a presentation
    # assuming black background
    if presentation==True:
        spine_col = kwargs.pop('spine_color','w')
        text_size = kwargs.pop('text_size','20')
        hide_spine = kwargs.pop('hide_spine',[])
        spine_lw=kwargs.pop('spine_lw','1')
#        spine_ec=kwargs.pop('spine_ec','k')
    else:
        pass
    
    # Share X axis
    if sharex!=None:
        if type(sharex)==list:
            pass
        else:
            sharex = list(sharex for i in range(naxes))
    elif sharex==None:
        sharex = list(False for i in range(naxes))
    # First item with sharex==True:
    try:
        firstx = sharex.index(True)
    except:
        firstx = None
    # Share Y axis  
    if sharey!=None:
        if type(sharey)==list:
            pass
        else:
            sharey = list(sharey for i in range(naxes))
    elif sharey==None:
        sharey = list(False for i in range(naxes))
    # First item with sharey==True:
    try:
        firsty = sharey.index(True)
    except:
        firsty = None
    
    sharexy = [(sharex[i],sharey[i]) for i in range(naxes)]
    
    # Add title
    if title!=None:
        fig.suptitle(title)
    # Calculate canvas dimensions
    
    # GRID
    if grid==None:
        grid = get_grid(alignment,naxes)
    else:
        grid = np.array(grid,dtype=int)
    ncols,nrows = grid

    if ratios==None:
        ratios = np.array([np.ones(ncols),np.ones(nrows)])
    else:
        if np.size(np.shape(ratios))==1:
            if   alignment == 'vertical':
                ratios = np.array([np.ones(ncols),ratios])
            elif alignment == 'horizontal':
                ratios = np.array([ratios,np.ones(nrows)])
        elif np.size(np.shape(ratios))==2:
            ratios = np.array(ratios).reshape((ncols,nrows))
    top, bottom = (top,bottom)
    left, right = (left,right)
    W, H        = (right-left, top-bottom)
    s           = sep
    #h           = H/naxes - (naxes-1)/naxes*s
    
    h0          = (H - (nrows-1)*s)/np.sum(ratios[1])
    w0          = (W - (ncols-1)*s)/np.sum(ratios[0])
    axes        = []
    axsize      = []
    for c in range(ncols):
        for r in range(nrows):
            ratiosc = ratios[0][:c]
            ratiosr = ratios[1][:r+1]
            w  = ratios[0][c]*w0
            h  = ratios[1][r]*h0
            l  = left + np.sum(ratiosc)*w0 + c*s
            d  = top - np.sum(ratiosr)*h0 - r*s
            size  = [l,d,w,h] 
            axsize.append(size)       
    for i in range(naxes):   
        size   = axsize[i]
        sharex,sharey = sharexy[i]
        if i==0:
            axes.append(fig.add_axes(size))
        else:
            kwargs = {}
            if   (sharex==True  and sharey==False):
                kwargs["sharex"]=axes[firstx]
                #axes.append(fig.add_axes(size,sharex=axes[firstx]))
            elif (sharex==False and sharey==True):
                kwargs["sharey"]=axes[firsty]
                #axes.append(fig.add_axes(size,sharey=axes[firsty]))
            elif (sharex==True  and sharey==True):
                kwargs["sharex"]=axes[firstx]
                kwargs["sharey"]=axes[firsty]
                #axes.append(fig.add_axes(size,sharex=axes[firstx],sharey=axes[firsty]))
            elif (sharex==False and sharey==False): 
                pass
                #axes.append(fig.add_axes(size))
            axes.append(fig.add_axes(size,**kwargs))
    for a in axes:
        a.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))
    if presentation == True:
        for a in axes:
            #plt.setp(tuple(a.spines.values()), edgecolor=spine_ec)
            plt.setp(tuple(a.spines.values()), color=spine_col)
            plt.setp(tuple(a.spines.values()), linewidth=spine_lw)
            
            plt.setp(tuple(a.spines.values()), facecolor=spine_col)
            plt.setp([a.get_xticklines(), a.get_yticklines(),a.get_xticklabels(),a.get_yticklabels()], color=spine_col)
            plt.setp([a.get_xticklabels(),a.get_yticklabels()],size=text_size)
            for s in hide_spine:
                a.spines[s].set_visible(False)
                #plt.setp([a.get_xlabel(),a.get_ylabel()],color=spine_col,size=text_size)
            #plt.setp(a.get_yticklabels(),visible=False)
    else:
        pass
    
    return fig,axes
def get_extname(order):
    return "ORDER{od:02d}".format(od=order)
def get_extreme(xarr,yarr,extreme="max",kind="LFC",thresh=0.1):
    ''' Calculates the positions of LFC profile peaks/valleys from data.
    In:
    ---   xarr,yarr (array-like, size=N (number of datapoints))
          extreme(str) = "min" or "max"
    Out:
    ---   peakpos(array-like, size=M (number of detected peaks))'''
    if extreme=='min':
        debugging=False   
    else:
        debugging=False
    if debugging:
        print("EXTREME = {}".format(extreme))
    
    # Calculate the first order numerical derivation of fluxes with respect to wavelength
    dy  = pd.Series(np.gradient(yarr,1,edge_order=2))

    df     = pd.DataFrame({"x":xarr, "xn":xarr.shift(1), "xp":xarr.shift(-1),
                           "y":yarr, "yn":yarr.shift(1), "yp":yarr.shift(-1),
                           "dx":xarr.diff(1), "dy":dy, 
                           "dyn":dy.shift(1)})
    # Find indices where two neighbouring points have opposite dy signs. 
    # We can now identify indices, p, for which i-th element of dy is +, dyn is -, ###and d2s is - (condition for maximum of a function)
    if extreme == "max":
        p = df.loc[(df.dy<=0.)&(df.dyn>0.)].index
    elif extreme == "min":
        p = df.loc[(df.dy>=0.)&(df.dyn<0.)].index
    # Simple linear interpolation to find the position where dx=0.
    xpk0  = (df.x - (df.xn-df.x)/(df.dyn-df.dy)*df.dy)[p].reset_index(drop=True)
    # Remove NaN values   
    xpk1 = xpk0.dropna()
    # Flux at the extremum is estimated as the maximum/minumum value of the two
    # points closest to the extremum
    if extreme == "max":
        ypk1 = df[["y","yn"]].iloc[p].max(axis=1).reset_index(drop=True)
    elif extreme == "min":
        ypk1 = df[["y","yn"]].iloc[p].min(axis=1).reset_index(drop=True)
    if kind == "LFC": 
        if extreme == "max":
            llim      = (df.y.max()-df.y.min())*thresh
            countmask = ypk1.loc[ypk1>=llim].index
            xpk       = xpk1[countmask].reset_index(drop=True)
            ypk       = ypk1[countmask].reset_index(drop=True)
        elif extreme == "min":
            xpk = xpk1
            ypk = ypk1
            pass
    peaks = pd.DataFrame({"x":xpk,"y":ypk})
        
    return peaks
def get_time(worktime):
    """
    Returns the work time in hours, minutes, seconds

    Outputs:
    --------
           h : hour
           m : minute
           s : second
    """					
    m,s = divmod(worktime, 60)
    h,m = divmod(m, 60)
    h,m,s = [int(value) for value in (h,m,s)]
    return h,m,s
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.
    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
    Returns:
    --------
        mask : A numobservations-length boolean array.
    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def is_outlier_running(points, window=5,thresh=1.):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
#    plt.figure()
#    plt.plot(points,label='data')
    rmean = running_mean(points,window)
#    rmean = np.percentile(points,85)
    if len(points.shape) == 1:
        points = points[:,None]  
    diff  = np.sum((points-rmean)**2,axis=-1)
    diff  = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
#    plt.plot(rmean,label='rmean')
#    plt.plot(diff,label='diff')
#    plt.plot(modified_z_score,label='z_score')
#    plt.legend()
#    plt.show()
    return modified_z_score > thresh
def is_outlier_original(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
def is_peak(points):
    """
    Returns a boolean array with True if points are peaks and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    return points>med_abs_deviation
def mad(x):
    ''' Returns median absolute deviation of input array'''
    return np.median(np.abs(np.median(x)-x))
#def remove_false_minima(oldminima,polyord=4,limit=limit):
#        xpos = oldminima.x
#        ypos = oldminima.y
#        diff = np.diff(xpos)
#        pars = np.polyfit(xpos[1:],diff,polyord)
#        model = np.polyval(pars,xpos[1:])
#        resids = diff-model
#        outliers1 = resids<-limit
#        outliers3 = resids>limit
#        print("Non-detections = ", np.sum(outliers3))
#        print("False detections = ",np.sum(outliers1))
#        # make outliers a len(xpos) array
#        outliers2 = np.insert(outliers1,0,False)
#        outliers4 = np.insert(outliers3,0,False)
#        newminima = (oldminima[~outliers2])
#        minima = newminima.reset_index(drop=True)
#        fig,ax=figure(2,sharex=True,ratios=[3,1])
#        ax[0].plot(x_axis,y_axis)
#        ax[0].scatter(xpos,ypos,marker='^',c='C1',s=8)
#        ax[0].scatter(xpos[outliers2],ypos[outliers2],marker='x',c='r',s=15)
#        ax[0].scatter(xpos[outliers4],ypos[outliers4],marker='x',c='C4',s=15)
#        ax[1].scatter(xpos[1:],resids,marker='o',c='C0',s=3)
#        ax[1].scatter(xpos[outliers2],resids[outliers1],marker='x',c='r',s=15)
#        ax[1].scatter(xpos[outliers4],resids[outliers3],marker='x',c='C4',s=15)
#        return minima
def peakdet(y_axis, x_axis = None, extreme='max', method='peakdetect_derivatives',
            lookahead=8, delta=0, pad_len=20, window=7,limit=None):
    '''
    https://gist.github.com/sixtenbe/1178136
    '''
    def remove_false_minima(input_xmin,input_ymin,limit,polyord=1):
        new_xmin = input_xmin
        new_ymin = input_ymin
        outliers   = np.full_like(input_xmin,True)
        j = 0
        plot=0
        if plot:
            fig,ax=figure(2,sharex=True,ratios=[3,1])
            ax[0].plot(x_axis,y_axis)
            ax[0].scatter(input_xmin,input_ymin,marker='^',c='red',s=8)
        while sum(outliers)>0:
            
            #print("j=",j,"N_outliers=",sum(outliers))
            old_xmin = new_xmin
            old_ymin = new_ymin
            xpos = old_xmin
            ypos = old_ymin
            diff = np.diff(xpos)
            pars = np.polyfit(xpos[1:],diff,polyord)
            model = np.polyval(pars,xpos[1:])
            
            resids = diff-model
            outliers1 = np.logical_or((resids<-limit), (diff<window))
            #print(outliers1)
            #outliers3 = resids>limit
            #print("Non-detections = ", np.sum(outliers3))
            #print("False detections = ",np.sum(outliers1))
            # make outliers a len(xpos) array
            outliers2 = np.insert(outliers1,0,False)
            #outliers4 = np.insert(outliers3,0,False)
            outliers = outliers2
            new_xmin = (old_xmin[~outliers])
            new_ymin = (old_ymin[~outliers])
            if plot:
                ax[0].scatter(xpos[outliers2],ypos[outliers2],marker='x',s=15,
                              c="C{}".format(j))
                #ax[0].scatter(xpos[outliers4],ypos[outliers4],marker='x',c='C4',s=15)
                ax[1].scatter(xpos[1:],resids,marker='o',s=3,c="C{}".format(j))
                ax[1].scatter(xpos[outliers2],resids[outliers1],marker='x',s=15,
                              c="C{}".format(j))
                #ax[1].scatter(xpos[1:],diff,marker='^',s=8)
            j+=1
        xmin, ymin = new_xmin, new_ymin
        
        if plot:
            maxima0 = (np.roll(minima.x,1)+minima.x)/2
            maxima = np.array(round(maxima0[1:]),dtype=np.int)
            [ax[0].axvline(x,ls=':',lw=0.5,c='r') for x in maxima]
        
        return xmin,ymin
    
    if method=='peakdetect':
        function = pkd.peakdetect
        args = (lookahead,delta)
    elif method == 'peakdetect_fft':
        function = pkd.peakdetect_fft
        args = (pad_len,)
    elif method == 'peakdetect_zero_crossing':
        function = pkd.peakdetect_zero_crossing
        args = (window,)
    elif method == 'peakdetect_derivatives':
        function = pkd.peakdetect_derivatives
        args = (window,)
    if delta == 0:
        if extreme is 'max':
            delta = np.percentile(y_axis,10)
        elif extreme is 'min':
            delta = 0
    maxima,minima = [np.array(a) for a 
                     in function(y_axis, x_axis, *args)]
    if extreme is 'max':
        return maxima
        #peaks = pd.DataFrame({"x":maxima[:,0],"y":maxima[:,1]})
    elif extreme is 'min':
        #peaks = pd.DataFrame({"x":minima[:,0],"y":minima[:,1]})
        limit = limit if limit is not None else 3*window
        minima = remove_false_minima(minima[:,0],minima[:,1],limit=limit)
        return minima
def peakdet2(xarr,yarr,delta=None,extreme='max'):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
    v = np.asarray(yarr)
    if xarr is None:
        xarr = np.arange(len(v))
    if delta is None:
        delta = np.percentile(v,p)
    if np.isscalar(delta):
        delta = np.full(xarr.shape,delta,dtype=np.float32)
    if len(yarr) != len(xarr):
        sys.exit('Input vectors v and x must have same length')
  
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True
    
    for i in range(len(v)):
        this = v[i]
        d    = delta[i]
        if this > mx:
            mx = this
            mxpos = xarr[i]
        if this < mn:
            mn = this
            mnpos = xarr[i]
        
        if lookformax:
            if this < mx-d:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = xarr[i]
                lookformax = False
        else:
            if this > mn+d:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = xarr[i]
                lookformax = True
       
    maxtab = np.array(maxtab)
    mintab = np.array(mintab)
    if extreme=='max':
        peaks = pd.DataFrame({"x":maxtab[:,0],"y":maxtab[:,1]})
    elif extreme=='min':
        peaks = pd.DataFrame({"x":mintab[:,0],"y":mintab[:,1]})
    return peaks
         
def polynomial(x, *p):
    y = np.zeros_like(x,dtype=np.float64)
    for i,a in enumerate(p):
        y = y + a*x**i
    return y

def rms(x):
    ''' Returns root mean square of input array'''
    return np.sqrt(np.mean(np.square(x)))
def running_mean(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]
#    series = pd.Series(x)
#    return series.rolling(N).mean()
def running_std(x, N):
        #return np.convolve(x, np.ones((N,))/N)[(N-1):]
    series = pd.Series(x)
    return series.rolling(N).std()

def return_basenames(filelist):
    filelist_noext = [os.path.splitext(file)[0] for file in filelist]
    return [os.path.basename(file) for file in filelist_noext]
def return_filelist(dirpath,ftype,fibre,ext='fits'):  
    filename_pattern=os.path.join(dirpath,
                    "*{fbr}_{ftp}.{ext}".format(ftp=ftype,fbr=fibre,ext=ext))
    try:
        filelist=np.array(glob(filename_pattern))
    except:
        raise ValueError("No files of this type were found")
    return filelist
def round_to_closest(a,b):
    return round(a/b)*b

def prepare_orders(order=None):
        '''
        Returns an array or a list containing the input orders.
        '''
        if order is None:
            orders = np.arange(hs.sOrder,hs.eOrder,1)
        else:
            orders = to_list(order)
        return orders

def sig_clip(v):
       m1=np.mean(v,axis=-1)
       std1=np.std(v-m1,axis=-1)
       m2=np.mean(v[abs(v-m1)<5*std1],axis=-1)
       std2=np.std(v[abs(v-m2)<5*std1],axis=-1)
       m3=np.mean(v[abs(v-m2)<5*std2],axis=-1)
       std3=np.std(v[abs(v-m3)<5*std2],axis=-1)
       return abs(v-m3)<5*std3   
def sig_clip2(v,sigma=5,maxiter=100,converge_num=0.02):
    ct   = np.size(v)
    iter = 0; c1 = 1.0; c2=0.0
    while (c1>=c2) and (iter<maxiter):
        lastct = ct
        mean   = np.mean(v,axis=-1)
        std    = np.std(v-mean,axis=-1)
        cond   = abs(v-mean)<sigma*std
        cut    = np.where(cond)
        ct     = len(cut[0])
        
        c1     = abs(ct-lastct)
        c2     = converge_num*lastct
        iter  += 1
    return cond
def select_orders(orders):
    use = np.zeros((hs.nOrder,),dtype=bool); use.fill(False)
    for order in range(hs.sOrder,hs.eOrder,1):
        if order in orders:
            o = order - hs.sOrder
            use[o]=True
    col = np.where(use==True)[0]
    return col
def to_list(item):
    """ Pushes item into a list """
    if type(item)==int:
        items = [item]
    elif type(item)==np.int64:
        items = [item]
    elif type(item)==list:
        items = item
    elif type(item)==np.ndarray:
        items = list(item)
    elif type(item)==str:
        items = [item]
    elif item is None:
        items = None
    else:
        print('Unsupported type. Type provided:',type(item))
    return items
def make_ticks_sparser(axis,scale='x',ticknum=None,minval=None,maxval=None):
    ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
        ticks on a given scale (x or y)'''
    ticknum = ticknum if ticknum is not None else 4
    if scale=='x':
        if minval is None or maxval is None:
            minval,maxval = axis.get_xlim()
        axis.set_xticks(np.linspace(minval,maxval,ticknum))
    elif scale=='y':
        if minval is None or maxval is None:
            minval,maxval = axis.get_ylim()
        axis.set_yticks(np.linspace(minval,maxval,ticknum))
    return axis
def wrap(args):
    function, pars = args
    return function(pars)
###############################################################################
################            LINE PROFILE GENERATION            ################
def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-(x / alpha)**2 * np.log(2))

def L(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def V(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)
###############################################################################
###############################################################################
#############                  COMB MANIPULATION                  #############                        
def _get_index(centers):
    ''' Input: dataarray with fitted positions of the lines
        Output: 1d array with indices that uniquely identify every line'''
    fac = 10000
    MOD = 2.
    od = centers.od.values[:,np.newaxis]*fac
    centers_round = np.rint(centers.values/MOD)*MOD
    centers_nonan = np.nan_to_num(centers_round)
    ce = np.asarray(centers_nonan,dtype=np.int)
    index0=np.ravel(od+ce)
    mask = np.where(index0%fac==0)
    index0[mask]=999999999
    return index0
def _get_sorted(index1,index2):
    print('len indexes',len(index1),len(index2))
    # lines that are common for both spectra
    intersect=np.intersect1d(index1,index2)
    intersect=intersect[intersect>0]

    indsort=np.argsort(intersect)
    
    argsort1=np.argsort(index1)
    argsort2=np.argsort(index2)
    
    sort1 =np.searchsorted(index1[argsort1],intersect)
    sort2 =np.searchsorted(index2[argsort2],intersect)
    
    return argsort1[sort1],argsort2[sort2]
def make_comb_interpolation(lines_LFC1, lines_LFC2,ftype='gauss'):
    ''' Routine to use the known frequencies and positions of a comb, 
        and its repetition and offset frequencies to build a frequency
        solution by linearly interpolating between individual lines
        
        Arguments must be for the same fibre!
        
        LFC1: known
        LFC2: to be interpolated
        
        Args:
        -----
            lines_LFC1 : lines xarray Dataset for a single exposure
            lines_LFC2 : lines xarray Dataset for a single exposure
    ''' 
    
    freq_LFC1 = lines_LFC1['attr'].sel(att='freq').load()
    freq_LFC2 = lines_LFC2['attr'].sel(att='freq').load()
    
#    pos_LFC1  = lines_LFC1['pars'].sel(par='cen',ft=ftype).load()
#    pos_LFC2  = lines_LFC2['pars'].sel(par='cen',ft=ftype).load()
    pos_LFC1  = lines_LFC1['attr'].sel(att='bary').load()
    pos_LFC2  = lines_LFC2['attr'].sel(att='bary').load()
    #plt.figure(figsize=(12,6))
    minord  = np.min(tuple(f.coords['od'] for f in [freq_LFC1,freq_LFC2]))
    maxord  = np.max(tuple(f.coords['od'] for f in [freq_LFC1,freq_LFC2]))
    interpolated = {}
    for od in np.arange(minord,maxord):
        print("Order {0:=>30d}".format(od))
        # get fitted positions of LFC1 and LFC2 lines
        x1 = pos_LFC1.sel(od=od).dropna('id')
        x2 = pos_LFC2.sel(od=od).dropna('id')
        # make sure to use only the lines that were successfully fitted
        #index1=_get_index(x1)
        #index2=_get_index(x2)
        good_LFC1 = x1.coords['id']
        good_LFC2 = x2.coords['id']
        # switch to numpy arrays
        x1 = x1.values
        x2 = x2.values
        x1, x2 = (np.sort(x) for x in [x1,x2])
        # find the closest LFC1 line to each LFC2 line in this order 
        f1 = freq_LFC1.sel(od=od)[good_LFC1].values
        f2 = freq_LFC2.sel(od=od)[good_LFC2].values
        f1, f2 = (np.sort(f) for f in [f1,f2])
        vals, bins = f2, f1
        right = np.digitize(vals,bins,right=False)
        print(right)
        fig, ax = figure(2,sharex=True,ratios=[3,1])
        ax[0].set_title("Order = {0:2d}".format(od))
        ax[0].scatter(f1,x1,c="C0",label='LFC1')
        ax[0].scatter(f2,x2,c="C1",label='LFC2')
        interpolated_LFC2 = []
        for x_LFC2,f_LFC2,index_LFC1 in zip(x2,f2,right):
            if index_LFC1 == 0 or index_LFC1>len(bins)-1:
                interpolated_LFC2.append(np.nan)
                continue
            else:
                pass
            
            f_left  = f1[index_LFC1-1]
            x_left  = x1[index_LFC1-1]
            f_right = f1[index_LFC1]
            x_right = x1[index_LFC1]
#            if x_LFC2 > x_right:
#                interpolated_LFC2.append(np.nan)
#                continue
#            else:
#                pass
            
            # fit linear function 
            fitpars = np.polyfit(x=[f_left,f_right],
                                 y=[x_left,x_right],deg=1)
            ax[0].scatter([f_left,f_right],[x_left,x_right],c='C0',marker='x',s=4)
            fspace = np.linspace(f_left,f_right,10)
            xspace = np.linspace(x_left,x_right,10)
            ax[0].plot(fspace,xspace,c='C0')
            x_int   = np.polyval(fitpars,f_LFC2)
            interpolated_LFC2.append(x_int)
            ax[1].scatter([f_LFC2],[(x_LFC2-x_int)*829],c='C1',marker='x',s=4)
            print("{:>3d}".format(index_LFC1),
                  (3*("{:>14.5f}")).format(x_left,x_LFC2,x_right),
                  "x_int = {:>10.5f}".format(x_int),
                  "RV = {:>10.5f}".format((x_LFC2-x_int)*829))
            #print(x_LFC2,interpolated_x)
        interpolated[od] = interpolated_LFC2
        #[plt.axvline(x1,ls='-',c='r') for x1 in f1]
        #[plt.axvline(x2,ls=':',c='g') for x2 in f2]
    return interpolated