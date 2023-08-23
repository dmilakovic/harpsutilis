#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:11:24 2023

@author: dmilakov
"""

from fitsio import FITS
import harps.lsf.container as container
from glob import glob
from velplot.fortread import read_fort13
import os


lsfpath = '/Users/dmilakov/projects/lfc/dataprod/v_2.2/lsf/HARPS.2018-12-05T08:12:52_lsf_most_likely.fits'
version = 1
wavefilelist = glob('/Users/dmilakov/projects/Q0515-4414/data/harps/from_rfc/*_wave_A.fits')
errfilelist =  glob('/Users/dmilakov/projects/Q0515-4414/data/harps/from_rfc/*_error_A.fits')

fort13_files = [
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg1/reg1_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg2/reg2_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg3/reg3_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg4/reg4_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg5/reg5_in.13',
    ]
savedir = '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/test_2023-08-21/ip/'


for i,file in enumerate(fort13_files):
    f13_out = read_fort13(file)
    table2  = f13_out[2]
    for j,transition in enumerate(table2):
        wstart = transition['wave_start']
        wend   = transition['wave_end']
        save_path = os.path.join(savedir,f'IP_r{i+1}_s{j+1}.dat')
        container.combine_from_list_of_files(lsfpath,version,
                                             xrange=6, # pixels
                                             dv=0.83, # km/s
                                             subpix=25,
                                             wstart=wstart,
                                             wend=wend,
                                             wavefilelist=wavefilelist,
                                             errfilelist=errfilelist,
                                             save=True,
                                             save_to=save_path)
        print(f'Model for file {i+1} segment {j+1} saved to {save_path}')
