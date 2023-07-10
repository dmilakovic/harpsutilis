#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:46:49 2023

@author: dmilakov
"""
import numpy as np

def prepare_data(flx1d,err1d,env1d,bkg1d,subbkg,divenv):
    f        = flx1d
    sigma_f  = err1d
    var_sum  = (sigma_f/f)**2
    
    if subbkg:
        b  = bkg1d
        e  = env1d - bkg1d
    else:
        b  = np.zeros_like(f)   
        e  = env1d
    if divenv:
        data = (f-b) / (e-b)
    else:
        data = f - b
    
    var_data = np.power(err1d,2.)
    var_bkg  = bkg1d
    var_env  = env1d
    if not subbkg and not divenv:
        var = var_data
    elif subbkg and not divenv:
        var = var_data + bkg1d
    elif subbkg and divenv:
        var = 1./(e-b)**2 * var_data + ((f-e)/(e-b)**2)**2 * var_bkg + \
            ((b-f)/(e-b)**2)**2 * var_env
    elif not subbkg and divenv:
        var = 1./e**2 * var_data + f/e**2 * var_env
    data_error   =  np.sqrt(var)
    
    return data, data_error