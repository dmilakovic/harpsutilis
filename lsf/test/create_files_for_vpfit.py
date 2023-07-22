#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:16:41 2023

@author: dmilakov
"""

from glob import glob
import os
wavefilelist=glob('/Users/dmilakov/projects/Q0515-4414/data/harps/from_rfc/*_wave_A.fits')
errfilelist=glob('/Users/dmilakov/projects/Q0515-4414/data/harps/from_rfc/*_error_A.fits')

#%%
import harps.lsf.container as hlc
lsfpath = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_2.1/HARPS.2018-12-05T08:12:52_lsf.fits'
version = 911
xrange = 5
dv = 0.83
subpix = 15
#%%
from velplot.fortread import read_fort13

def produce_IPs_from_fort13(fort13_path,dirpath=None,j=None):
    
    
    
    header, comment, table1,table2,table3 = read_fort13(fort13_path)
    wstart_list = table1['wave_start']
    wend_list   = table1['wave_end']
    
    
    
    
    for i,(wstart,wend) in enumerate(zip(wstart_list,wend_list)):
        save_to = os.path.join(dirpath,f'IP_r{j+1}_s{i+1}.dat')
        hlc.combine_from_list_of_files(lsfpath, version, xrange, dv, subpix,
                                       wstart, wend, wavefilelist,errfilelist,
                                       save=True,
                                       save_to=save_to)
        
def produce_IPs_from_list_fort13(fort13_list,dirpath=None):
    for j,file in enumerate(fort13_list):
        produce_IPs_from_fort13(file,dirpath,j)
    #%%
fort13_list = [
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/mvpfit084/5alpha_lfc/reg1/reg1_noip_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/mvpfit084/5alpha_lfc/reg2/reg2_noip_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/mvpfit084/5alpha_lfc/reg3/reg3_noip_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/mvpfit084/5alpha_lfc/reg4/reg4_noip_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/mvpfit084/5alpha_lfc/reg5/reg5_noip_in.13',
    ]
dirpath = '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/mvpfit084/ip/'

produce_IPs_from_list_fort13(fort13_list,dirpath)
