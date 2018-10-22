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
harps_rv     = os.path.join(harps_prod,'rv')
harps_combined = os.path.join(harps_prod,'combined_datasets')
harps_plots  = os.path.join(harps_dtprod,'plots')
harps_sims   = os.path.join(harps_home,'simulations')
harps_fits   = os.path.join(harps_dtprod,'fits')

dirnames = {'home':harps_home,
            'data':harps_data,
            'dtprod':harps_dtprod,
            'prod':harps_prod,
            'psf':harps_psf,
            'wavesol':harps_ws,
            'linelist':harps_fits,
            'lines':harps_lines,
            'plots':harps_plots,
            'simul':harps_sims}

rexp = 1e5

## 
nproc = 4

## first and last order in a spectrum
chip   = 'both'
if chip == 'red':
    sOrder = 45   
    eOrder = 71
elif chip == 'blue':
    sOrder = 25
    eOrder = 41
elif chip == 'both':
    sOrder = 43
    eOrder = 71
nOrder = eOrder - sOrder
nPix   = 4096
##
