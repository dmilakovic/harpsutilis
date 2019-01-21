#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:34:22 2018

@author: dmilakov
"""
from harps.core import FITS
from harps.core import os, np

import harps.settings as hs
import harps.functions as hf
#from   harps.lsf import LSF
version = hs.__version__

#==============================================================================
    
#                                 E 2 D S   
    
#==============================================================================
def read_e2ds_data(filepath):
    with FITS(filepath,memmap=False) as hdulist:
        data     = hdulist[0].read()
    return data
def mread_e2ds_data(e2dslist):
    ''' Reads the data of all files in the list provided'''
    data = []
    filelist = read_textfile(e2dslist)
    for i,filepath in enumerate(filelist):
        hf.update_progress(i/(len(filelist)-1),"Read fits")
        with FITS(filepath,memmap=False) as hdulist:
            data.append(hdulist[0].read())
    return data
def read_e2ds_meta(filepath):
    header   = read_e2ds_header(filepath)
    data     = read_e2ds_data(filepath)
    
    mjd      = header['MJD-OBS']
    obsdate  = header['DATE-OBS']
    npix     = header["NAXIS1"]
    nbo      = header["ESO DRS CAL LOC NBO"]
    conad    = header["ESO DRS CCD CONAD"]
    try:
        qcstr = str.strip(header["ESO DRS CAL QC"])
    except:
        qcstr = "UNKNOWN"
    if  qcstr == "PASSED":
        qc = True
    else:
        qc = False
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
               qc=qcstr, mjd=mjd, exptime=exptime, fibre=fibre, 
               fibshape=fibshape)
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

def read_LFC_keywords(filepath,LFC_name,anchor_offset=0):
    with FITS(filepath,memmap=False) as hdulist:
        header   = hdulist[0].read_header()
    
    fr_source = 250e6
    try:
        #offset frequency of the LFC, rounded to 1MHz
        anchor  = header["ESO INS LFC1 ANCHOR"]
        #repetition frequency of the LFC
        source_reprate = 250e6#header["ESO INS LFC1 REPRATE"]
    except:
        anchor         = 288059930000000.0 #Hz, HARPS frequency 2016-11-01
        source_reprate = 250e6
    
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
    #include anchor offset if provided
    anchor = anchor + anchor_offset
    
    m,k            = divmod(
                        round((anchor-f0_source)/fr_source),
                               modefilter)
    f0_comb   = (k-1)*fr_source + f0_source
    LFC_keys = dict(name=LFC_name, comb_anchor=f0_comb, window_size=window,
                    source_anchor=anchor, source_reprate=source_reprate, 
                    modefilter=modefilter, comb_reprate=reprate,ppl=pixPerLine,
                    comb_offset=anchor_offset)
    return LFC_keys
# =============================================================================
#    
#                        O U T    F I L E S
#    
# =============================================================================
    
def _check_if_list(item):
    if isinstance(item,list):
        return item
    else:
        return [item]
    
def read_outfile(filepath,version=501):
    return read_outfile_extension(filepath,['linelist','wavesol_comb'],version)
def read_outfile_header(filepath,extension=0,version=None):
    with FITS(filepath,'r') as fits:
        header = fits[extension,version].read_header()
    return header
def read_outfile_extension(filepath, extension=['wavesol_comb'],version=501):
    extension = _check_if_list(extension)
    data = []
    with FITS(filepath,'r') as fits: 
        for ext in extension:
            if ext == 'linelist':
                    data.append(fits[ext].read())
            else:
                try:
                    data.append(fits[ext,version].read())
                except:
                    raise ValueError("Extension {0}, v{1} "
                                     "could not be found".format(ext,version))
    return tuple(data)
def read_fluxord(filepath):
    header = read_outfile_header(filepath,0,None)
    fluxes = [rec['value'] for rec in header.records() \
                        if 'FLUXORD' in rec['name']]
    return fluxes
def mread_outfile(outlist_filepath,extensions,version=None,**kwargs):
    version = hf.item_to_version(version)
    
    outlist = read_textfile(outlist_filepath,**kwargs)
    
    cache = {ext:[] for ext in np.atleast_1d(extensions)}
    for i,file in enumerate(outlist):
        hf.update_progress(i/(len(outlist)-1),'Read')
        with FITS(file,'r') as fits:
            for ext,lst in cache.items():
                if ext not in ['linelist','weights',
                               'background','flux','error']:
                    data = fits[ext,version].read()
                else:
                    data = fits[ext].read()
                lst.append(data)

    for ext,lst in cache.items():
        cache[ext] = np.array(lst)

    return cache, len(outlist)

#==============================================================================
    
#               L I N E L I S T      A N D     W A V E S O L   
    
#==============================================================================
allowed_hdutypes = ['linelist','flux','background','error','weights',
                    'coeff_gauss','coeff_lsf','wavesol_gauss','wavesol_lsf',
                    'model_gauss','model_lsf','residual_gauss','residual_lsf',
                    'wavesol_2pt_lsf','wavesol_2pt_gauss']
def new_fits(filepath,dirpath=None):
    # ------- Checks 
#    assert hdutype in allowed_hdutypes, 'Unrecognized HDU type'
    path = get_fits_path(filepath,dirpath)
    try:
        os.remove(path)
    except:
        pass
    newfits = FITS(path,mode='rw')
    return newfits

def open_fits(filepath,dirpath=None,mode='rw',overwrite=False):
    fits = read_fits(filepath,dirpath,mode)
    if fits is not None:
        if overwrite==True:
            return new_fits(filepath,dirpath)
        else:
            return fits
    else:
        return new_fits(filepath,dirpath)

def fits_exists(filepath,dirpath=None):
    path = get_fits_path(filepath,dirpath)
    if os.path.isfile(path):
        exists = True
    else:
        exists=False
    return exists

def read_fits(filepath,dirpath=None,mode='rw'):
    try:
        path = get_fits_path(filepath,dirpath)
        hdu  = FITS(path,mode=mode)
    except:
#        raise IOError("File does not exist")
        hdu = None
    return hdu

def read_hdudata(filepath,dirpath=None):
    with get_hdu(filepath,dirpath) as hdu:
        hdudata = {}
        for h in hdu[1:]:
            extname = h.get_extname()
            data    = h.read()
            hdudata[extname]=data
    return hdudata
    

def read_hduext(filepath,extname,dirpath=None):
    assert extname in allowed_hdutypes
    with open_fits(filepath,dirpath) as fits:
        hduext = fits[extname].read()
        return hduext
def write_hdu(filepath,data,extname,header=None,dirpath=None):
    assert extname in allowed_hdutypes
    with open_fits(filepath,dirpath) as fits:
        fits.write(data=data,extname=extname,header=header)
        print(fits)
    return
def get_fits_path(filepath,dirpath=None,version=version):
    dirname  = get_dirpath(version,dirpath)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    newname  = basename.replace('e2ds','out')+'.fits'
    path     = os.path.join(dirname,newname)
    return path    
def get_dirpath(version=version,dirpath=None):
    ''' Returns the path to the directory with files of the selected type. '''
    if dirpath is not None:
        dirpath = dirpath
    else:
        dirpath = hs.dirnames['fits']
    #print("DIRNAME = ",dirpath)
    direxists = os.path.isdir(dirpath)
    if not direxists:
        create=input("Directory {} does not exist. Create? (y/n)".format(dirpath))
        if create == 'y':
            try: 
                hs.make_directory(dirpath)
            except:
                dirpath = hs.harps_fits
        else:
            dirpath = hs.harps_fits
    return dirpath
def get_extnames(filepath,dirpath=None):
    with get_hdu(filepath,dirpath) as hdu:
        extnames = [h.get_extname() for h in hdu[1:]]
    return extnames

#==============================================================================
    
#                          T E X T    F I L E S   
    
#==============================================================================
def read_textfile(filepath,start=None,stop=None,step=None):
    if os.path.isfile(filepath):
        mode = 'r+'
    else:
        mode = 'a+'
    data = [line.strip('\n') for line in open(filepath,mode)
              if line[0]!='#']
    use = slice(start,stop,step)
    return data[use]
def write_textfile(data,filepath,header=None):
    '''
    Writes data to file.
    
    Args:
    -----
        data : str or list
    '''
    
    write_header = True if header is not None else False
    # Make directory if does not exist
    dirname = os.path.dirname(filepath)
    success = hs.make_directory(dirname)
    if success:
        pass
    else:
        raise ValueError("Could not make directory")
    # Check if file exists
    if os.path.isfile(filepath):
        mode = 'a'
    else:
        mode = 'w'
        
    # Write data
    data_out = to_string(data)
    with open(filepath,mode) as outfile:
        if write_header:
            outfile.write(header)
        else:
            pass
        outfile.write(data_out)
        
def to_string(obj,sep='\n'):
    '''
    Separator must be a string
    '''
    assert isinstance(sep,str)==True
    
    if isinstance(obj,str):
        return obj
    elif isinstance(obj,list):
        return sep.join(str(val) for val in obj)
    else:
        return None
        