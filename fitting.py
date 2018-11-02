#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:07:32 2018

@author: dmilakov
"""

from harps.core import leastsq
from harps.core import np

import harps.emissionline as emline


#==============================================================================
#
#                         L I N E      F I T T I N G                  
#
#==============================================================================

def gauss(x,flux,error,bkg,model='SingleGaussian',*args,**kwargs):
    line_model = getattr(emline,model)
    line = line_model(x,flux-bkg,error)
    pars, parerr = line.fit(bounded=False)
    return pars, parerr
    
    