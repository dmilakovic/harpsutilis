#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:59:15 2018

@author: dmilakov
"""
import os, errno

__version__ = '0.5.1'
version     = 'v_{vers}'.format(vers=__version__)

harps_home   = os.environ['HARPSHOME']
harps_data   = os.environ['HARPSDATA']
harps_dtprod = os.environ['HARPSDATAPROD']

def prod_version(version):
    return os.path.join(*[harps_dtprod,version])
harps_prod     = prod_version(version)
harps_psf      = os.path.join(*[harps_prod,'psf_fit'])
harps_ws       = os.path.join(*[prod_version(version),'fits','wavesol'])
harps_lines    = os.path.join(*[harps_prod,'xrlines'])
harps_rv       = os.path.join(*[harps_prod,'rv'])
harps_combined = os.path.join(*[harps_prod,'combined_datasets'])
harps_plots    = os.path.join(*[prod_version(version),'plots'])
harps_sims     = os.path.join(*[harps_home,'simulations'])
harps_linelist = os.path.join(*[prod_version(version),'fits','linelist'])
harps_coeff    = os.path.join(*[prod_version(version),'fits','coeff'])
harps_fits     = os.path.join(*[prod_version(version),'fits'])

dirnames = {'home':harps_home,
            'data':harps_data,
            'dtprod':harps_dtprod,
            'prod':harps_prod,
            'psf':harps_psf,
            'fits':harps_fits,
            'wavesol':harps_ws,
            'linelist':harps_linelist,
            'lines':harps_lines,
            'coeff':harps_coeff,
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
def make_directory(dirpath):
    print("Making directory: ",dirpath)
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return 1
def make_directory_tree(version):
    directories = (harps_prod,harps_psf,harps_ws,harps_linelist,harps_plots)
    [make_directory(d) for d in directories]

