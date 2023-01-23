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
import jaxopt
from tinygp import kernels, GaussianProcess, transforms, noise
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit, newton
from harps.functions import gauss4p
import harps.lsf as hlsf
import numpyro
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)
rng_key = jax.random.PRNGKey(55873)
#%%
def get_data(od,pixl,pixr,filter=50):
    import harps.lsf as hlsf
    import harps.io as io
    fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/single.dat'
    modeller=hlsf.LSFModeller(fname,60,70,method='gp',subpix=10,
                              filter=None,numpix=8,iter_solve=1,iter_center=1)

    extensions = ['linelist','flux','background','error','wavereference']
    data, numfiles = io.mread_outfile(modeller._outfile,extensions,701,
                            start=None,stop=None,step=None)
    linelists=data['linelist']
    fluxes=data['flux']
    errors=data['error']
    backgrounds=data['background']
    # backgrounds=None
    wavelengths=data['wavereference']
    orders=np.arange(od,od+1)
    pix3d,vel3d,flx3d,err3d,orders=hlsf.stack('gauss',
                                              linelists,
                                              fluxes,
                                              wavelengths,
                                              errors,
                                              backgrounds,
                                              orders)


    pix1s=pix3d[od,pixl:pixr]
    vel1s=vel3d[od,pixl:pixr]
    flx1s=flx3d[od,pixl:pixr]
    err1s=err3d[od,pixl:pixr]

    # vel1s_ , flx1s_, err1s_ = vel1s, flx1s, err1s
    x = pix1s
    # x = vel1s
    vel1s_, flx1s_, err1s_ = hlsf.clean_input(x,flx1s,err1s,sort=False,
                                              verbose=True,filter=filter)
    
    X      = jnp.array(vel1s_)
    # X      = jnp.array(pix1s)
    Y      = jnp.array(flx1s_*100)
    Y_err  = jnp.array(err1s_*100)
    # Y      = jnp.array([flx1s_,err1s_])
    return X, Y, Y_err, 
# od=102
# od = 120
# seg = 5
# pixl=9111//16*seg
# pixr=9111//16*(seg+1)
# pixl=2235
# pixr=2737
od = 90
pixl = 2944
pixr = 3500
X_,Y_,Y_err_ = get_data(od,pixl,pixr,None)
#%%
X = X_
Y = Y_
Y_err = Y_err_
plt.figure()
plt.errorbar(X,Y,Y_err,ls='',marker='.',color='grey')
#%%
lsf1s = hlsf.construct_lsf1s(X, Y, Y_err, method='tinygp',
                             numiter=2,
                             numpix=10,subpix=4,
                             plot=True, 
                             save_plot=False, 
                             model_scatter=False)