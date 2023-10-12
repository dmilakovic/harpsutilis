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
lsfpath = '/Users/dmilakov/projects/lfc/dataprod/v2.3.5/lsf/HARPS.2018-12-05T08:12:52.040_e2ds_A_lsf.fits'
version = 1
xrange = 5
dv = 0.8299977
subpix = 25
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
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/2023-10-04/reg1/reg1_noip_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/2023-10-04/reg2/reg2_noip_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/2023-10-04/reg3/reg3_noip_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/2023-10-04/reg4/reg4_noip_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/2023-10-04/reg5/reg5_noip_in.13',
    ]
dirpath = '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/2023-10-04/ip/'

produce_IPs_from_list_fort13(fort13_list,dirpath)
