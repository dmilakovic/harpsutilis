#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:28:45 2019

@author: dmilakov
"""

from   harps.core     import FITS, np, os
from   harps.classes  import Spectrum

import harps.settings as hs
import harps.io       as io
version      = hs.__version__

class ObjectSpec(object):
    def __init__(self,objSpec,overwrite=False,*args,**kwargs):
        self._path2objSpec = objSpec
        
        dirpath       = kwargs.pop('dirpath',None)
        self._outfile = io.get_fits_path('objspec',
                                        self._path2objSpec,
                                        version,dirpath)
        overwrite     = kwargs.pop('overwrite',False)
        primhead      = self.return_header('primary')
        with FITS(self.outfile,'rw',clobber=overwrite) as hdu:
            hdu[0].write_keys(primhead)
        
    def _set_calibration_file(self,LFCspec):
        self._path2LFCSpec = LFCspec
        self._LFCSpec = Spectrum(LFCspec)
        
    @property
    def calibration_spec(self):
        return self._LFCSpec
    
    def return_header(self,hdutype):
        def return_value(name):
            if name=='Simple':
                value = True
            elif name=='Bitpix':
                value = 32
            elif name=='Naxis':
                value = 0
            elif name=='Extend':
                value = True
            elif name=='Author':
                value = 'Dinko Milakovic'
            elif name=='version':
                value = version
            elif name=='fibre':
                value = self.fibre
            return value
        def make_dict(name,value,comment=''):
            return dict(name=name,value=value,comment=comment)
            
        if hdutype == 'primary':
            names = ['Simple','Bitpix','Naxis','Extend','Author',
                     'Object','version']            
        else: 
            names = ['version']
        comments_dict={'Simple':'Conforms to FITS standard',
                  'Bitpix':'Bits per data value',
                  'Naxis':'Number of data axes',
                  'Extend':'FITS dataset may contain extensions',
                  'Author':'',
                  'fibre':'Fibre',
                  'version':'Code version used'}
        values_dict = {name:return_value(name) for name in names}
        
        values   = [values_dict[name] for name in names]
        comments = [comments_dict[name] for name in names]
        header   = [make_dict(n,v,c) for n,v,c in zip(names,values,comments)]
        return FITSHDR(header)