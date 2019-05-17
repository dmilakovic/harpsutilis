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
    
    