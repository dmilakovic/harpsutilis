#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:43:28 2018

@author: dmilakov
"""
import numpy as np
import pandas as pd
import sys

from harps.peakdetect import peakdetect
#from harps.emissionline import SingleGaussian, SimpleGaussian, DoubleGaussian 
from harps.settings import *

from scipy.special import erf
from scipy.optimize import minimize, leastsq, curve_fit

from matplotlib import pyplot as plt

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
    if method=='forward':
        dx = np.diff(x,n)
        dy = np.diff(y,n)
        d  = dy/dx
    if method == 'central':
        z1  = np.hstack((y[0], y[:-1]))
        z2  = np.hstack((y[1:], y[-1]))
        dx1 = np.hstack((0, np.diff(x)))
        dx2 = np.hstack((np.diff(x), 0))  
        #print("Zeros in dx1+dx2",np.where((dx1+dx2)==0)[0].size)
        #print(z2-z1)
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
def fit_peak(i,xarray,yarray,yerr,weights,xmin,xmax,dx,method='erfc',
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
        model_class = SingleGaussian
    elif model=='doublegaussian':
        n = 6
        model_class = DoubleGaussian
    elif model=='simplegaussian':
        n=6
        model_class = SimpleGaussian
    #print(model)
    dtype = np.dtype([('pars',np.float64,(n,)),
                      ('errors',np.float64,(n,)),
                      ('pn',np.float64,(1,)),
                      ('r2',np.float64,(1,)),
                      ('cen',np.float64,(1,)),
                      ('cen_err',np.float64,(1,))])
    results = np.empty(shape=(1,),dtype=dtype)
    
    # Fit only data between the two adjacent minima of the i-th peak
    if i<np.size(xmin)-1:
        cut = xarray.loc[((xarray>=xmin[i])&(xarray<=xmin[i+1]))].index
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
            pars, errors = line.fit(bounded=True)
            center       = line.center
            center_error = line.center_error
            rsquared     = line.calc_R2()
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
        results['pars']    = pars
        results['pn']      = pn
        results['errors']  = errors
        results['r2']      = rsquared
        results['cen']     = center
        results['cen_err'] = center_error
    return results
    #return np.concatenate((best_pars,np.array([pn])))
def flatten_list(inlist):
    outlist = [item for sublist in inlist for item in sublist]
    return outlist
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
def get_fig_axes(naxes,ratios=None,title=None,sep=0.05,alignment="vertical",
                 figsize=(16,9),sharex=None,sharey=None,grid=None,
                 subtitles=None,presentation=False,
                 left=0.1,right=0.95,top=0.95,bottom=0.08,**kwargs):
    
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
    # Change color scheme and text size if producing plots for a presentation
    # assuming black background
    if presentation==True:
        spine_col = kwargs.pop('spine_color','w')
        text_size = 20
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
    if presentation == True:
        for a in axes:
            plt.setp(tuple(a.spines.values()), color=spine_col)
            plt.setp([a.get_xticklines(), a.get_yticklines(),a.get_xticklabels(),a.get_yticklabels()], color=spine_col)
            plt.setp([a.get_xticklabels(),a.get_yticklabels()],size=text_size)
#            plt.setp([a.get_xlabel(),a.get_ylabel()],color=spine_col,size=text_size)
            #plt.setp(a.get_yticklabels(),visible=False)
    else:
        pass
    
    return fig,axes

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

def peakdet(y_axis, x_axis = None, extreme='max', lookahead=8, delta=0):
    '''
    https://gist.github.com/sixtenbe/1178136
    '''
    if delta == 0:
        if extreme is 'max':
            delta = np.percentile(y_axis,10)
        elif extreme is 'min':
            delta = 0
    maxima,minima = [np.array(a) for a 
                     in peakdetect(y_axis, x_axis, lookahead, delta)]
    if extreme is 'max':
        peaks = pd.DataFrame({"x":maxima[:,0],"y":maxima[:,1]})
    elif extreme is 'min':
        peaks = pd.DataFrame({"x":minima[:,0],"y":minima[:,1]})
    return peaks
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
def select_orders(orders):
    use = np.zeros((nOrder,),dtype=bool); use.fill(False)
    for order in range(sOrder,eOrder,1):
        if order in orders:
            o = order - sOrder
            use[o]=True
    col = np.where(use==True)[0]
    return col