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
harps_plots  = os.path.join(harps_dtprod,'plots')


## 
nproc = 10

## first and last order in a spectrum
chip   = 'red'
if chip == 'red':
    sOrder = 45   
    eOrder = 72
elif chip == 'blue':
    sOrder = 25
    eOrder = 41
nOrder = eOrder - sOrder
nPix   = 4096
##
