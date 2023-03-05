#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:47:30 2023

Reads in the linelist and calls harps.lsf functions to do the LSF modelling

@author: dmilakov
"""
#%%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import harps.lsf as hlsf
import harps.lsf.read as hread
jax.config.update("jax_enable_x64", True)
rng_key = jax.random.PRNGKey(55873)
#%%

ftype='HARPS'
# ftype='ESPRESSO'
if ftype == 'HARPS':
    npix = 4096
    od = 51
    # fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/harps/2015-04-17_1440.dat'
    fname = '/Users/dmilakov/projects/lfc/list/2018-12-05_A.list'
    # fname = '/Users/dmilakov/projects/lfc/list/HARPS2018-12-10T0525.list'
    # checksum = f'{ftype}_'

if ftype == 'ESPRESSO':
    npix = 9111
    od = 120
    fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/single.dat'


segm = 0
pixl=npix//16*segm
pixr=npix//16*(segm+1)
# pixl=2235
# pixr=2737
# pixl = 3200
# pixr = 3500
scale = 'pix'
# scale = 'velocity'
X_,Y_,Y_err_ = hread.get_data(fname,od,pixl,pixr,scale=scale,fittype='gauss',filter=None)
X = jnp.array(X_)
Y = jnp.array(Y_)
Y_err = jnp.array(Y_err_)

plt.figure()
plt.errorbar(X,Y,Y_err,ls='',marker='.',color='grey')
#%%
import harps.lsf.construct as construct
lsf1s_sct = construct.model_1s(X, Y, Y_err, 
                             numiter=2,
                             plot=True, 
                             save_plot=True, 
                             model_scatter=True,
                             metadata=dict(
                                 order=od,
                                 scale=scale,
                                 segment=segm,
                                 iteration=0,
                                 )
                             )
#%%
lsf1s_nosct = construct.model_1s(X, Y, Y_err, 
                             numiter=2,
                             plot=True, 
                             save_plot=True, 
                             model_scatter=False,
                             metadata=dict(
                                 order=od,
                                 scale=scale,
                                 segment=segm,
                                 iteration=0,
                                 )
                             )