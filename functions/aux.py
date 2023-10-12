#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:40:57 2023

@author: dmilakov
"""

import os, re
import numpy as np
from glob import glob
import harps.settings as hs
import harps.containers as container

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
    elif type(item)==tuple:
        items = [*item]
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
        p = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}.\d{3}")
        s = p.search(bn)
        if s:
            dt = np.datetime64(s[0].replace('_',':')) 
        else:
            dt = np.datetime64(None)
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
from collections import defaultdict
def list_duplicates(seq):
    # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    """ Return a dictionary of duplicates of the input sequence """
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)
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

# import bisect
# class Closest:
#     """Assumes *no* redundant entries - all inputs must be unique"""
#     def __init__(self, numlist=None, firstdistance=0):
#         if numlist is None:
#             numlist=[]
#         self.numindexes = dict((val, n) for n, val in enumerate(numlist))
#         self.nums = sorted(self.numindexes)
#         self.firstdistance = firstdistance

#     def append(self, num):
#         if num in self.numindexes:
#             raise ValueError("Cannot append '%s' it is already used" % str(num))
#         self.numindexes[num] = len(self.nums)
#         bisect.insort(self.nums, num)

#     def rank(self, target):
#         rank = bisect.bisect(self.nums, target)
#         if rank == 0:
#             pass
#         elif len(self.nums) == rank:
#             rank -= 1
#         else:
#             dist1 = target - self.nums[rank - 1]
#             dist2 = self.nums[rank] - target
#             if dist1 < dist2:
#                 rank -= 1
#         return rank

#     def closest(self, target):
#         try:
#             return self.numindexes[self.nums[self.rank(target)]]
#         except IndexError:
#             return 0

#     def distance(self, target):
#         rank = self.rank(target)
#         try:
#             dist = abs(self.nums[rank] - target)
#         except IndexError:
#             dist = self.firstdistance
#         return dist
from   numpy.lib.recfunctions import append_fields
def stack_arrays(list_of_arrays):
    '''
    Stacks a list of structured arrays, adding a column indicating the position
    in the list.
    '''
    indices  = np.hstack([np.full(len(array),i)  for i,array \
                          in enumerate(list_of_arrays)])
    stacked0 = np.hstack(list_of_arrays)
    stacked  = append_fields(stacked0,'exp',indices,usemask=False)
    return stacked


#------------------------------------------------------------------------------
# 
#                            F U N C T I O N S
#
#------------------------------------------------------------------------------



def wrap(args):
    function, pars = args
    return function(pars)



#------------------------------------------------------------------------------
# 
#                                O T H E R
#
#------------------------------------------------------------------------------
def accuracy(w=None,SNR=10,dx=0.829,u=0.9):
    '''
    Returns the rms accuracy [km/s] of a spectral line with SNR=10, 
    pixel size = 0.829 km/s and apsorption strength 90%.
    
    Equation 4 from Cayrel 1988 "Data Analysis"
    
    Parameters
    ----------
    w : float
        The equivalent width of the line (in pix). The default is None.
    SNR : TYPE, optional
        DESCRIPTION. The default is 10.
    dx : TYPE, optional
        DESCRIPTION. The default is 829.
    u : TYPE, optional
        DESCRIPTION. The default is 0.9.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if w is None:
        raise ValueError("No width specified")
    epsilon = 1/SNR
    return np.sqrt(2)/np.pi**0.25 * np.sqrt(w*dx)*epsilon/u
def equivalent_width(SNR,wave,R,dx):
    ''' Cayrel 1988 formula 6
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
    fit    = profile.gauss3p(x,amp,ctr,sgm)
    chisq  = ((data - fit)**2/weights).sum()
    return chisq
def lambda_at_vz(v,z,l0):
    '''Returns the wavelength of at redshift z moved by a velocity offset of v [m/s]'''
    return l0*(1+z)*(1+v/c)


    


