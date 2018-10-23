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
    hdulist  = FITS(filepath,memmap=False)
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
allowed_hdutypes = ['linelist','wavesol']
def new_hdutype(spec,hdutype,dirpath=None,overwrite=True):
    # ------- Checks 
    assert hdutype in allowed_hdutypes, 'Unrecognized HDU type'
    filepath = spec.filepath
    path = get_hdu_pathname(filepath,hdutype,dirpath)
    if overwrite:
        try:
            os.remove(path)
        except:
            pass
    else:
        pass
    # ------- Reads metadata and LFC keywords
    meta = spec.meta
    LFC  = spec.lfckeys
    def make_dict(name,value,comment=''):
        return dict(name=name,value=value,comment=comment)
    def return_header():
        header_names=['Author','version','npix','MJD','obsdate','fibshape',
                      'LFC','omega_r','omega_0',
                      'use_gaps','use_ptch','polyord']
        header_values=['Dinko Milakovic',version,meta['npix'],meta['mjd'],
                       meta['obsdate'],meta['fibshape'],
                       LFC['name'],LFC['reprate'],LFC['f0_comb'],
                       spec.use_gaps,spec.patches,spec.polyord]
        header_comments=['','harps.classes version used',
                         'Number of pixels','Modified Julian Date',
                         'LFC name','Repetition frequency',
                         'Offset frequency','Fibre shape',
                         'Shift lines using gap file',
                         'Fit wavelength solution in 512 pix patches',
                         'Polynomial order of the wavelength solution']
        
        header = [make_dict(n,v,c) for n,v,c in zip(header_names,
                                                    header_values,
                                                    header_comments)]
        return header
    
    newhdu = FITS(path,mode='rw')
    header = return_header()
    newhdu.write([[0,0],[1,1]],header=header,
                 extname='PRIMARY')
    return newhdu
def read_hdutype(filepath,hdutype,mode='rw',dirpath=None):
    assert hdutype in allowed_hdutypes, 'Unrecognized HDU type'
    path = get_hdu_pathname(filepath,hdutype,dirpath)
    if os.path.isfile(path):    
        return FITS(path,mode=mode)
    else: 
        raise IOError("File '{f}' does not exist"
                      " for this filepath".format(f=hdutype))
def read_linelist(filepath,mode='rw',dirpath=None):
    """ Wrapper around 'read_hdutype' for hdutype='linelist' """
    return read_hdutype(filepath,'linelist',mode,dirpath)
    
def read_wavesol(filepath,mode='rw',dirpath=None):
    """ Wrapper around 'read_hdutype' for hdutype='wavesol' """
    return read_hdutype(filepath,'wavesol',mode,dirpath)


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