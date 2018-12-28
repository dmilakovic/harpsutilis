#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:58:21 2018

@author: dmilakov
"""
#==============================================================================

#                             C L A S S E S   

#==============================================================================

from harps.settings import Settings
from harps.process import Process
from harps.spectrum import Spectrum
from harps.manager import Manager
from harps.series import Series
from harps.lsfmodeller import LSFModeller
from harps.plotter import SpectrumPlotter#, LSFPlotter, ManagerPlotter



#==============================================================================

#                        M I S C E L A N E O U S   

#==============================================================================
import numpy as np
import warnings
class mimicKwargs(object):
    def __init__(self, labelLong, labelShort, defautValue,kwargDic):
        if kwargDic.get(labelLong) and kwargDic.get(labelShort):
            warnings.warn("same flag used two times")
        else:
            self.userInput = kwargDic.get(labelLong) or kwargDic.get(labelShort) or defautValue
    def output(self):
        return self.userInput
class ClassCreator(object):
    def __init__(self,nFiles,nOrder,fibre,names):
        dtypes   = {"real":[np.float32 for n in names],"comp":[np.complex64 for n in names]}
        shapes2d = [(nFiles,nOrder) for n in names]
        shapes1d = [(nOrder) for n in names]
        fibres   = list(fibre)
        if 'A' in fibres and 'B' in fibres:
            fibres.append('A-B')
        self.names       = names
        self.real1d      = np.dtype([((names[i],dtypes["real"][i],(shapes1d[i]))) for i in range(len(names))])
        self.real2d      = np.dtype([((names[i],dtypes["real"][i],(shapes2d[i]))) for i in range(len(names))])
        self.comp1d      = np.dtype([((names[i],dtypes["comp"][i],(shapes1d[i]))) for i in range(len(names))])
        self.comp2d      = np.dtype([((names[i],dtypes["comp"][i],(shapes2d[i]))) for i in range(len(names))])
        self.data        = np.dtype({"names":fibres, "formats":[self.real2d for f in fibres]})
        self.median      = np.dtype({"names":fibres, "formats":[self.real1d for f in fibres]})
        self.ftdata      = np.dtype({"names":fibres, "formats":[self.comp2d for f in fibres]})
        self.ftmedian    = np.dtype({"names":fibres, "formats":[self.comp1d for f in fibres]})
        return                
class Datatypes(object):
    def __init__(self,nFiles, nOrder, fibre):
        self.nFiles   = nFiles
        self.nOrder   = nOrder
        self.fibres   = list(fibre)
        
        
    def specdata(self, add_corr=False):
        names    = ["FLX", "ENV", "BKG", "B2E"]
        if add_corr == True:
            names.append("FMB")
        return ClassCreator(self.nFiles,self.nOrder,self.fibres,names)
    def calibrationdata(self):
        names    = ["HARPS", "FOCES", "THAR"]
        return ClassCreator(self.nFiles,self.nOrder,self.fibres,names)
    def custom(self,names):
        return ClassCreator(self.nFiles,self.nOrder,self.fibres,names)
 
