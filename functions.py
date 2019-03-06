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
from harps.constants import c
import harps.containers as container

from harps.core import welch

from scipy.special import erf, wofz, gamma, gammaincc, expn
from scipy.optimize import minimize, leastsq, curve_fit
from scipy.integrate import quad

from glob import glob

from matplotlib import pyplot as plt
#from kapteyn import kmpfit

__version__   = hs.__version__
# some shared lists for 'return_empty_dataset' and 'return_empty_dataarray'

#------------------------------------------------------------------------------
# 
#                           P R O P O S A L S
#
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# 
#                           M A T H E M A T I C S
#
#------------------------------------------------------------------------------
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
def freq_to_lambda(freq):
    return 1e10*c/(freq) #/1e9
def integer_slice(i, n, m):
    # return nth to mth digit of i (as int)
    l = math.floor(math.log10(i)) + 1
    return i / int(pow(10, l - m)) % int(pow(10, m - n + 1))

def mad(x):
    ''' Returns median absolute deviation of input array'''
    return np.median(np.abs(np.median(x)-x))
def polynomial(x, *p):
    y = np.zeros_like(x,dtype=np.float64)
    for i,a in enumerate(p):
        y += a*x**i
    return y

def rms(x,axis=None):
    ''' Returns root mean square of input array'''
    return np.sqrt(np.nanmean(np.square(x),axis=axis))
def running_mean(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]
#    series = pd.Series(x)
#    return series.rolling(N).mean()
def running_std(x, N):
        #return np.convolve(x, np.ones((N,))/N)[(N-1):]
    series = pd.Series(x)
    return series.rolling(N).std()
def round_to_closest(a,b):
    return round(a/b)*b

def sig_clip(v):
       m1=np.mean(v,axis=-1)
       std1=np.std(v-m1,axis=-1)
       m2=np.mean(v[abs(v-m1)<5*std1],axis=-1)
       std2=np.std(v[abs(v-m2)<5*std1],axis=-1)
       m3=np.mean(v[abs(v-m2)<5*std2],axis=-1)
       std3=np.std(v[abs(v-m3)<5*std2],axis=-1)
       return abs(v-m3)<5*std3   
def sigclip1d(v,sigma=3,maxiter=10,converge_num=0.02,plot=False):
    v    = np.array(v)
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
    if plot:
        plt.figure(figsize=(12,6))
        plt.scatter(np.arange(len(v)),v,s=2,c="C0")        
        plt.scatter(np.arange(len(v))[~cond],v[~cond],
                        s=10,c="C1",marker='x')
        plt.axhline(mean,ls='-',c='r')
        plt.axhline(mean+sigma*std,ls='--',c='r')
        plt.axhline(mean-sigma*std,ls='--',c='r')
    return cond
#def remove_outliers(v,sigma=3,maxiter=10,converge_num=0.02,plot=False):
#    v = np.array(v)
#    outliers   = np.full_like(v,True)
#    j = 0
#    while sum(outliers)>0:
#        oldv = v
#        v    = oldv
#        
#        outliers1 = np.logical_and((np.abs(resids)>limit), (diff<0.9*mindist))
def sigclip2d(v,sigma=5,maxiter=100,converge_num=0.02):
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
def negpos(number):
    return -abs(number),abs(number)
def removenan(array):
    return array[~np.isnan(array)]
def nan_to_num(array):
    finite = np.isfinite(array)
    return array[finite]
def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

def missing_elements(L, start, end):
    """
    https://stackoverflow.com/questions/16974047/
    efficient-way-to-find-missing-elements-in-an-integer-sequence
    """
    if end - start <= 1: 
        if L[end] - L[start] > 1:
            yield from range(L[start] + 1, L[end])
        return

    index = start + (end - start) // 2

    # is the lower half consecutive?
    consecutive_low =  L[index] == L[start] + (index - start)
    if not consecutive_low:
        yield from missing_elements(L, start, index)

    # is the upper part consecutive?
    consecutive_high =  L[index] == L[end] - (end - index)
    if not consecutive_high:
        yield from missing_elements(L, index, end)
def find_missing(integers_list,start=None,limit=None):
    """
    Given a list of integers and optionally a start and an end, finds all
    the integers from start to end that are not in the list.

    'start' and 'end' default respectivly to the first and the last item of the list.

    Doctest:

    >>> find_missing([1,2,3,5,6,7], 1, 7)
    [4]

    >>> find_missing([2,3,6,4,8], 2, 8)
    [5, 7]

    >>> find_missing([1,2,3,4], 1, 4)
    []

    >>> find_missing([11,1,1,2,3,2,3,2,3,2,4,5,6,7,8,9],1,11)
    [10]

    >>> find_missing([-1,0,1,3,7,20], -1, 7)
    [2, 4, 5, 6]

    >>> find_missing([-2,0,3], -5, 2)
    [-5, -4, -3, -1, 1, 2]

    >>> find_missing([2],4,5)
    [4, 5]

    >>> find_missing([3,5,6,7,8], -3, 5)
    [-3, -2, -1, 0, 1, 2, 4]

    >>> find_missing([1,2,4])
    [3]

    """
    # https://codereview.stackexchange.com/a/77890
    start = start if start is not None else integers_list[0]
    limit = limit if limit is not None else integers_list[-1]
    return [i for i in range(start,limit + 1) if i not in integers_list]
#------------------------------------------------------------------------------
# 
#                           G A U S S I A N S
#
#------------------------------------------------------------------------------
    
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

#------------------------------------------------------------------------------
# 
#                           P L O T T I N G
#
#------------------------------------------------------------------------------    
def figure(*args, **kwargs):
    return get_fig_axes(*args, **kwargs)

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
#------------------------------------------------------------------------------
# 
#                       L I S T   M A N I P U L A T I O N
#
#------------------------------------------------------------------------------
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

def prepare_orders(order=None):
        '''
        Returns an array or a list containing the input orders.
        '''
        if order is None:
            orders = np.arange(hs.sOrder,hs.eOrder,1)
        else:
            orders = to_list(order)
        return orders


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
    elif type(item)==str or isinstance(item,np.str):
        items = [item]
    elif item is None:
        items = None
    else:
        print('Unsupported type. Type provided:',type(item))
    return items    
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
        bn = os.path.splitext(os.path.basename(fn))[0]
        dt = np.datetime64(bn.split('.')[1].replace('_',':')) 
        datetimes.append(dt)
    if len(datetimes)==1:
        return datetimes[0]
    else:
        return datetimes
def datetime_to_tuple(datetime):
    def to_tuple(dt):
        return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    datetimes = np.atleast_1d(datetime)
    datelist  = datetimes.tolist()
    dt_tuple  = list(map(to_tuple,datelist))
    return dt_tuple
def datetime_to_record(datetime):
    datetimes = np.atleast_1d(datetime)
    datelist  = datetimes.tolist()
    dt_record = container.datetime(len(datetimes))
    for dtr,dtv in zip(dt_record,datelist):
        dtr['year']  = dtv.year
        dtr['month'] = dtv.month
        dtr['day']   = dtv.day
        dtr['hour']  = dtv.hour
        dtr['min']   = dtv.minute
        dtr['sec']   = dtv.second
    return dt_record
def record_to_datetime(record):
    if record.dtype.fields is None:
        raise ValueError("Input must be a structured numpy array")   
    if isinstance(record,np.void):
        dt = '{0:4}-{1:02}-{2:02}T{3:02}:{4:02}:{5:02}'.format(*record)
        datetime = np.datetime64(dt)
    elif isinstance(record,np.ndarray):
        datetime = np.zeros_like(record,dtype='datetime64[s]')
        for i,rec in enumerate(record):
            dt='{0:4}-{1:02}-{2:02}T{3:02}:{4:02}:{5:02}'.format(*rec)
            datetime[i] = dt
    return datetime
def tuple_to_datetime(value):
    def to_datetime(value):
        return np.datetime64('{0:4}-{1:02}-{2:02}'
                             'T{3:02}:{4:02}:{5:02}'.format(*value))
    values = np.atleast_1d(value)
    datetimes = np.array(list(map(to_datetime,values)))
    return datetimes
#def record_to_tuple(record):
#    def to_tuple(rec):
#        return rec['year'],rec['month'],rec['day'], \
#               rec['hour'],rec['min'],rec['sec']
#    if record.dtype.fields is None:
#        raise ValueError("Input must be a structured numpy array")   
#    if isinstance(record,np.void):
#        dt = '{0:4}-{1:02}-{2:02}T{3:02}:{4:02}:{5:02}'.format(*record)
#        datetime = np.datetime64(dt)
#    elif isinstance(record,np.ndarray):
#        datetime = np.zeros_like(record,dtype='datetime64[s]')
#        for i,rec in enumerate(record):
#            dt='{0:4}-{1:02}-{2:02}T{3:02}:{4:02}:{5:02}'.format(*rec)
#            datetime[i] = dt
#    return datetime
    
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

def flatten_list(inlist):
    outlist = [item for sublist in inlist for item in sublist]
    return outlist
def ravel(array,removenan=True):
    a = np.ravel(array)
    if removenan:
        a = a[~np.isnan(a)]
    return a

def read_filelist(filepath):
    if os.path.isfile(filepath):
        mode = 'r+'
    else:
        mode = 'a+'
    filelist=[line.strip('\n') for line in open(filepath,mode)
              if line[0]!='#']
    return filelist
def overlap(a, b):
    # https://www.followthesheep.com/?p=1366
    a1=np.argsort(a)
    b1=np.argsort(b)
    # use searchsorted:
    sort_left_a=a[a1].searchsorted(b[b1], side='left')
    sort_right_a=a[a1].searchsorted(b[b1], side='right')
    #
    sort_left_b=b[b1].searchsorted(a[a1], side='left')
    sort_right_b=b[b1].searchsorted(a[a1], side='right')


    # # which values are in b but not in a?
    # inds_b=(sort_right_a-sort_left_a == 0).nonzero()[0]
    # # which values are in a but not in b?
    # inds_a=(sort_right_b-sort_left_b == 0).nonzero()[0]

    # which values of b are also in a?
    inds_b=(sort_right_a-sort_left_a > 0).nonzero()[0]
    # which values of a are also in b?
    inds_a=(sort_right_b-sort_left_b > 0).nonzero()[0]

    #return a1[inds_a], b1[inds_b]
    return inds_a,inds_b
#------------------------------------------------------------------------------
# 
#                           P E A K     D E T E C T I O N
#
#------------------------------------------------------------------------------
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




def peakdet(y_axis, x_axis = None, extreme='max',remove_false=False,
            method='peakdetect_derivatives',
            lookahead=8, delta=0, pad_len=20, window=7,limit=None):
    '''
    https://gist.github.com/sixtenbe/1178136
    '''
    def remove_false_minima(input_xmin,input_ymin,limit,mindist,polyord=1):
        new_xmin = input_xmin
        new_ymin = input_ymin
        outliers   = np.full_like(input_xmin,True)
        j = 0
        plot=False
        if plot:
            fig,ax=figure(3,sharex=True,ratios=[3,1,1])
            ax[0].plot(x_axis,y_axis)
            ax[0].scatter(input_xmin,input_ymin,marker='^',c='red',s=8)
        while sum(outliers)>0:
            old_xmin = new_xmin
            old_ymin = new_ymin
            xpos = old_xmin
            ypos = old_ymin
            diff = np.diff(xpos)
            pars = np.polyfit(xpos[1:],diff,polyord)
            model = np.polyval(pars,xpos[1:])
            
            resids = diff-model
            outliers1 = np.logical_and((np.abs(resids)>limit), (diff<0.9*mindist))
            # make outliers a len(xpos) array
            outliers2 = np.insert(outliers1,0,False)
            outliers = outliers2
            new_xmin = (old_xmin[~outliers])
            new_ymin = (old_ymin[~outliers])
            if plot:
                ax[0].scatter(xpos[outliers2],ypos[outliers2],marker='x',s=15,
                              c="C{}".format(j))
                ax[1].scatter(xpos[1:],resids,marker='o',s=3,c="C{}".format(j))
                ax[1].scatter(xpos[outliers2],resids[outliers1],marker='x',s=15,
                              c="C{}".format(j))
                ax[2].scatter(xpos[1:],diff,marker='o',s=3,c="C{}".format(j))
                ax[1].axhline(limit,c='r',lw=2)
                ax[1].axhline(-limit,c='r',lw=2)
                ax[2].axhline(0.9*mindist,c='r',lw=2)
            j+=1
        xmin, ymin = new_xmin, new_ymin
        
        if plot:
            maxima0 = (np.roll(xmin,1)+xmin)/2
            maxima = np.array(maxima0[1:],dtype=np.int)
            [ax[0].axvline(x,ls=':',lw=0.5,c='r') for x in maxima]
            
        
        return xmin,ymin
    def limits(y_axis):
        freq, P    = welch(y_axis)
        maxind     = np.argmax(P)
        maxfreq    = freq[maxind]
        # maxima and minima in the power spectrum
        maxima, minima = (np.transpose(x) for x in pkd.peakdetect(P,freq))
        minsorter  = np.argsort(minima[0])
        # the largest period 
        index      = np.searchsorted(minima[0],maxfreq,sorter=minsorter)
        
        minfreq = (minima[0][index-1:index+1])
        maxdist, mindist = tuple(1./minfreq)
        return mindist,maxdist
        
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
        data = np.transpose(maxima)
    elif extreme is 'min':
        data = np.transpose(minima)
    if remove_false:
        limit = limit if limit is not None else 2*window
        mindist, maxdist = limits(y_axis)
        #print(mindist,maxdist)
        data = remove_false_minima(data[0],data[1],limit,mindist)
    return data

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
#------------------------------------------------------------------------------
# 
#                           C O M B     S P E C I F I C
#
#------------------------------------------------------------------------------  
default = 501
def extract_item(item):
    """
    utility function to extract an "item", meaning
    a extension number,name plus version.
    
    To be used with partial decorator
    """
    ver=default
    if isinstance(item,tuple):
        ver_sent=True
        nitem=len(item)
        if nitem == 1:
            ext=item[0]
        elif nitem == 2:
            ext,ver=item
    else:
        ver_sent=False
        ext=item
    return ext,ver,ver_sent
def item_to_version(item=None):
    # IMPORTANT : this function controls the DEFAULT VERSION
    """
    Returns an integer representing the settings provided
    
    Returns the default version if no args provided.
    
    Args:
    -----
    item (dict,int,tuple) : contains information on the version
    
    Returns:
    -------
    version (int): either 1 or a three digit integer in the range 100-511
        If version == 1:
           linear interpolation between individual lines
        If version in 100-511: 
           version = PGS (polynomial order [int], gaps [bool], segmented[bool])
                   
    """
    assert default > 99 and default <1000, "Invalid default version"
    ver = default
    #polyord,gaps,segment = [int((default/10**x)%10) for x in range(3)][::-1]
    polyord,gaps,segment = version_to_pgs(item)
    if isinstance(item,dict):
        polyord = item.pop('polyord',polyord)
        gaps    = item.pop('gaps',gaps)
        segment = item.pop('segment',segment)
        ver     = int("{2:2d}{1:1d}{0:1d}".format(segment,gaps,polyord))
    elif isinstance(item,int) or isinstance(item,np.int64):
        polyord,gaps,segment=version_to_pgs(item)
        ver     = int("{2:2d}{1:1d}{0:1d}".format(segment,gaps,polyord))
    elif isinstance(item,tuple):
        polyord = item[0]
        gaps    = item[1]
        segment = item[2]
        ver     = int("{2:2d}{1:1d}{0:1d}".format(segment,gaps,polyord))
    return ver
def version_to_pgs(ver):  
    #
    
    if isinstance(ver,int) and ver==1:
        polyord = 1
        gaps    = 0
        segment = 0
    elif isinstance(ver,int) and ver>99 and ver<1000:
        dig = np.ceil(np.log10(ver)).astype(int)
        split  = [int((ver/10**x)%10) for x in range(dig)][::-1]
        if dig==3:
            polyord, gaps, segment = split
        elif dig==4:
            polyord = np.sum(i*np.power(10,j) for j,i \
                             in enumerate(split[:2][::-1]))
            gaps    = split[2]
            segment = split[3]
    else:
        polyord,gaps,segment = version_to_pgs(501)
    return polyord,gaps,segment
    
def noise_from_linelist(linelist):
    x = (np.sqrt(np.sum(np.power(linelist['noise']/c,-2))))
    return c/x
def remove_bad_fits(linelist,fittype,limit=0.03):
    """ 
    Removes lines which have uncertainties in position larger than a given 
    limit.
    """
    field  = '{}_err'.format(fittype)
    cut = np.where(linelist[field][:,1]<=limit)[0]
    #print(len(cut),len(linelist), "{0:5.3%} removed".format((len(linelist)-len(cut))/len(linelist)))
    return linelist[cut]
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
def average_line_flux(linelist,flux2d,bkg2d=None):
    orders = np.unique(linelist['order'])
    if bkg2d is not None:
        totflux = np.sum(flux2d[orders]-bkg2d[orders])
    else:
        totflux = np.sum(flux2d[orders])
    nlines = len(linelist)
    return totflux/nlines
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
    
    freq_LFC1 = lines_LFC1['freq']
    freq_LFC2 = lines_LFC2['freq']

    pos_LFC1  = lines_LFC1[ftype][:,1]
    pos_LFC2  = lines_LFC2['bary']
    #plt.figure(figsize=(12,6))
    minord  = np.max(tuple(np.min(f['order']) for f in [lines_LFC1,lines_LFC2]))
    maxord  = np.min(tuple(np.max(f['order']) for f in [lines_LFC1,lines_LFC2]))
    interpolated = {}
    for od in np.arange(minord,maxord):
        print("Order {0:=>30d}".format(od))
        # get fitted positions of LFC1 and LFC2 lines
        inord1 = np.where(lines_LFC1['order']==od)
        inord2 = np.where(lines_LFC2['order']==od)
        cen1   = pos_LFC1[inord1]
        cen2   = pos_LFC2[inord2]
        x1, x2 = (np.sort(x) for x in [cen1,cen2])
        # find the closest LFC1 line to each LFC2 line in this order 
        freq1  = freq_LFC1[inord1]
        freq2  = freq_LFC2[inord2]
        f1, f2 = (np.sort(f)[::-1] for f in [freq1,freq2])
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
        break
    return interpolated

#------------------------------------------------------------------------------
#
#                           P R O G R E S S   B A R 
#
#------------------------------------------------------------------------------
def update_progress(progress,name=None):
    # https://stackoverflow.com/questions/3160699/python-progress-bar
    barLength = 40 
    status = ""
    name = name if name is not None else ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt\r\n"
    if progress >= 1:
        progress = 1
        status = "Done\r\n"
    block = int(round(barLength*progress))
    mess  = (name,"#"*block + "-"*(barLength-block), progress*100, status)
    text = "\rProgress [{0}]: [{1}] {2:8.3f}% {3}".format(*mess)
    sys.stdout.write(text)
    sys.stdout.flush()
#------------------------------------------------------------------------------
#
#                        P H O T O N     N O I S E
#
#------------------------------------------------------------------------------   
def get_background1d(data,kind="linear",*args):
    """
    Returns the background in the echelle order. 
    Default linear interpolation between line minima.
    """
    yarray = np.atleast_1d(data)
    assert len(np.shape(yarray))==1, "Data is not 1-dimensional"
    xarray = np.arange(np.size(data))
    bkg    = get_background(xarray,yarray)
    return bkg
    
def get_background2d(data,kind="linear", *args):
    """
    Returns the background for all echelle orders in the spectrum.
    Default linear interpolation between line minima.
    """
    orders     = np.shape(data)[0]
    background = np.array([get_background1d(data[o],kind) \
                           for o in range(orders)])
    return background

def get_background(xarray,yarray,kind='linear'):
    """
    Returns the interpolated background between minima in yarray.
    
    Smooths the spectrum using Wiener filtering to detect true minima.
    See peakdetect.py for more information
    """
    from scipy import interpolate
    xbkg,ybkg = peakdet(yarray, xarray, extreme="min")
    if   kind == "spline":
        intfunc = interpolate.splrep(xbkg, ybkg)
        bkg     = interpolate.splev(xarray,intfunc) 
    elif kind == "linear":
        intfunc = interpolate.interp1d(xbkg,ybkg,
                                       bounds_error=False,
                                       fill_value=0)
        bkg = intfunc(xarray)
    return bkg

def get_error2d(data2d):
    assert len(np.shape(data2d))==2, "Data is not 2-dimensional"
    data2d  = np.abs(data2d)
    bkg2d   = get_background2d(data2d)
    error2d = np.sqrt(np.abs(data2d) + np.abs(bkg2d))
    return error2d
    
def get_error1d(data1d,*args):
    data1d  = np.abs(data1d)
    bkg1d   = np.abs(get_background1d(data1d,*args))
    error1d = np.sqrt(data1d + bkg1d)
    return error1d
def sigmav(data2d,wavesol2d):
    """
    Calculates the limiting velocity precison of all pixels in the spectrum
    using ThAr wavelengths.
    """
    orders  = np.arange(np.shape(data2d)[0])
    sigma_v = np.array([sigmav1d(data2d[order],wavesol2d[order]) \
                        for order in orders])
    return sigma_v

def sigmav1d(data1d,wavesol1d):
    """
    Calculates the limiting velocity precison of all pixels in the order
    using ThAr wavelengths.
    """
    # weights for photon noise calculation
    # Equation 5 in Murphy et al. 2007, MNRAS 380 p839
    #pix2d   = np.vstack([np.arange(spec.npix) for o in range(spec.nbo)])
    error1d = get_error1d(data1d)
    df_dlbd = derivative1d(data1d,wavesol1d)
    sigma_v = c*error1d/(wavesol1d*df_dlbd)
    return sigma_v

# =============================================================================
    
#                    S P E C T R U M     H E L P E R
#                          F U N C T I O N S
    
# =============================================================================
#def ord2optord(order,fibre):
#        optord = np.arange(88+self.nbo,88,-1)
#        # fibre A doesn't contain order 115
#        if self.meta['fibre'] == 'A':
#            shift = 1
#        # fibre B doesn't contain orders 115 and 116
#        elif self.meta['fibre'] == 'B':
#            shift = 2
#        cut=np.where(optord>114)
#        optord[cut]=optord[cut]+shift
#        
#        return optord