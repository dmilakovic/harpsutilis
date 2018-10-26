#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:34:22 2018

@author: dmilakov
"""
from harps.core import FITS, fits
from harps.core import os

import harps.settings as hs

version = hs.__version__

#==============================================================================
    
#                                 E 2 D S   
    
#==============================================================================
def read_e2ds_data(filepath):
    with FITS(filepath,memmap=False) as hdulist:
        data     = hdulist[0].read()
    return data
def read_e2ds_meta(filepath):
    header   = read_e2ds_header(filepath)
    mjd      = header['MJD-OBS']
    obsdate  = header['DATE-OBS']
    npix     = header["NAXIS1"]
    nbo      = header["ESO DRS CAL LOC NBO"]
    conad    = header["ESO DRS CCD CONAD"]
    try: 
        d    = header["ESO DRS CAL TH DEG LL"]
    except:
        try:
            d  = header["ESO DRS CAL TH DEG X"]
        except:
            d  = 3
            print(filepath)
            UserWarning ("No ThAr calibration attached")
            pass
    date     = header["DATE"]
    exptime  = header["EXPTIME"]
    # Fibre information is not saved in the header, but can be obtained 
    # from the filename 
    fibre    = filepath[-6]
    fibshape = 'octogonal'
    meta     = dict(npix=npix, nbo=nbo, conad=conad, d=d, obsdate=obsdate,
                mjd=mjd, exptime=exptime, fibre=fibre, fibshape=fibshape)
    return meta 
def read_e2ds_header(filepath):
    with FITS(filepath,memmap=False) as hdulist:
        header   = hdulist[0].read_header()
    return header
def read_e2ds(filepath):
    data   = read_e2ds_data(filepath)
    meta   = read_e2ds_meta(filepath)
    header = read_e2ds_header(filepath)
    return data, meta, header

def read_LFC_keywords(filepath,LFC_name):
    with FITS(filepath,memmap=False) as hdulist:
        header   = hdulist[0].read_header()
    
    fr_source = 250e6
    try:
        #offset frequency of the LFC, rounded to 1MHz
        #self.anchor = round(self.header["HIERARCH ESO INS LFC1 ANCHOR"],-6) 
        anchor  = header["ESO INS LFC1 ANCHOR"]
        #repetition frequency of the LFC
        source_reprate = header["ESO INS LFC1 REPRATE"]
    except:
        anchor       = 288059930000000.0 #Hz, HARPS frequency 2016-11-01
    if LFC_name=='HARPS':
        modefilter   = 72
        f0_source    = -50e6 #Hz
        reprate      = modefilter*fr_source #Hz
        pixPerLine   = 22
        # wiener filter window scale
        window       = 3
    elif LFC_name=='FOCES':
        modefilter   = 100
        f0_source    = 20e6 #Hz
        reprate      = modefilter*fr_source #Hz
        anchor       = round(288.08452e12,-6) #Hz 
        # taken from Gaspare's notes on April 2015 run
        pixPerLine   = 35
        # wiener filter window scale
        window       = 5
    #omega_r = 250e6
    m,k            = divmod(
                        round((anchor-f0_source)/fr_source),
                               modefilter)
    f0_comb   = (k-1)*fr_source + f0_source
    
    LFC_keys = dict(name=LFC_name, comb_anchor=f0_comb, window_size=window,
                    source_anchor=anchor, source_reprate=source_reprate, 
                    modefilter=modefilter, comb_reprate=reprate,ppl=pixPerLine)
    return LFC_keys


#==============================================================================
    
#               L I N E L I S T      A N D     W A V E S O L   
    
#==============================================================================
allowed_hdutypes = ['linelist','wavesol','model','coeff']
def new_hdutype(filepath,hdutype,dirpath=None,overwrite=True):
    # ------- Checks 
    assert hdutype in allowed_hdutypes, 'Unrecognized HDU type'
    path = get_hdu_pathname(filepath,hdutype,dirpath)
    if overwrite:
        try:
            os.remove(path)
        except:
            pass
    else:
        pass
    
    
    newhdu = FITS(path,mode='rw')
    return newhdu

def new_linelist(filepath=None,dirpath=None,overwrite=True):
    """ Wrapper around 'new_hdutype' for hdutype='linelist' """
    return new_hdutype(filepath,'linelist',dirpath,overwrite)

def new_wavesol(filepath=None,dirpath=None,overwrite=True):
    """ Wrapper around 'new_hdutype' for hdutype='wavesol' """
    return new_hdutype(filepath,'wavesol',dirpath,overwrite)
def new_coeff(filepath=None,dirpath=None,overwrite=True):
    """ Wrapper around 'new_hdutype' for hdutype='coeff' """
    return new_hdutype(filepath,'coeff',dirpath,overwrite)

def read_hdutype(filepath,hdutype,dirpath=None):
    assert hdutype in allowed_hdutypes, 'Unrecognized HDU type'
    path = get_hdu_pathname(filepath,hdutype,dirpath)
    if os.path.isfile(path):  
        pass
    else:
        raise IOError("File '{f}' does not exist"
                      " for this filepath".format(f=hdutype))
    with FITS(path,mode='r') as hdu:
        datadict = {}
        for h in hdu[1:]:
            extname = h.get_extname()
            data    = h.read()
            datadict[extname]=data
    return datadict
        
def read_linelist(filepath,mode='rw',dirpath=None):
    """ Wrapper around 'read_hdutype' for hdutype='linelist' """
    return read_hdutype(filepath,'linelist',dirpath)
    
def read_wavesol(filepath,mode='rw',dirpath=None):
    """ Wrapper around 'read_hdutype' for hdutype='wavesol' """
    return read_hdutype(filepath,'wavesol',dirpath)

def write_hdutype(filepath,data,hdutype,dirpath=None):
    assert hdutype in allowed_hdutypes, 'Unrecognized HDU type'
    path = get_hdu_pathname(filepath,hdutype,dirpath)
    exists = os.path.isfile(path)
    if exists:
        hdu = read_hdutype(filepath,hdutype,'rw',dirpath)
    else:
        raise IOError()
def get_hdu_pathname(filepath,hdutype,version=version,dirname=None):
    dirname  = get_dirname(hdutype,dirname)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    path     = os.path.join(dirname,basename+'_{}.fits'.format(hdutype))
    return path    
def get_dirname(filetype,version=version,dirname=None):
    ''' Returns the path to the directory with files of the selected type. '''
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