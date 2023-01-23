#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:26:58 2022

@author: dmilakov
"""

import harps.functions as hf
import harps.spectrum as hs
import harps.lsf as hlsf
import harps.lines as lines

import numpy as np
import matplotlib.pyplot as plt

#%%
spec=hc.ESPRESSO('/Users/dmilakov/projects/lfc/data/2019-05-03_ESPRESSO_S2D_LFC_FP_A.fits',
                 '/Users/dmilakov/projects/lfc/data/ESPRESSO_DLL_MATRIX_A.fits',
                 f0=270e6,fr=18e9)

#%%
od = 100

