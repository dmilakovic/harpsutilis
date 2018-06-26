#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:59:15 2018

@author: dmilakov
"""
import os

harps_home   = os.environ['HARPSHOME']
harps_data   = os.environ['HARPSDATA']
harps_dtprod = os.environ['HARPSDATAPROD']

harps_prod   = os.path.join(harps_dtprod,'products')
harps_psf    = os.path.join(harps_prod,'psf_fit')
harps_ws     = os.path.join(harps_prod,'wave_solutions')
harps_lines  = os.path.join(harps_prod,'lines')
harps_plots  = os.path.join(harps_dtprod,'plots')


## 
nproc = 6

## first and last order in a spectrum
chip   = 'both'
if chip == 'red':
    sOrder = 45   
    eOrder = 72
elif chip == 'blue':
    sOrder = 25
    eOrder = 41
elif chip == 'both':
    sOrder = 43
    eOrder = 72
nOrder = eOrder - sOrder
nPix   = 4096
##
