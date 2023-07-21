#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:16:41 2023

@author: dmilakov
"""

from glob import glob
wavefilelist=glob('/Users/dmilakov/projects/Q0515-4414/data/harps/from_rfc/*_wave_A.fits')
errfilelist=glob('/Users/dmilakov/projects/Q0515-4414/data/harps/from_rfc/*_error_A.fits')

#%%
import harps.lsf.container as hlc
lsfpath = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_2.1/HARPS.2018-12-05T08:12:52_lsf.fits'
version = 911
xrange = 5
dv = 0.83
subpix = 15

wstart_list = [5032.039,
               5096.968,
               5114.793,
               5552.449,
               5581.477,
               6002.597,
               6018.003,
               6124.113
               ]

wend_list =   [
               5034.844,
               5099.808,
               5117.643,
               5555.543,
               5584.587,
               6005.842,
               6021.357,
               6127.525
               ]


for wstart,wend in zip(wstart_list,wend_list):
    wmid = (wstart+wend)/2.
    save_to = '/Users/dmilakov/projects/Q0515-4414/data/harps/ip_files/'+\
              f'Q0515_IP_wav={wmid:5.2f}A.dat'
    hlc.combine_from_list_of_files(lsfpath, version, xrange, dv, subpix,
                                   wstart, wend, wavefilelist,errfilelist,
                                   save=True,
                                   save_to=save_to)
