#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 09:26:55 2023

@author: dmilakov
"""
import harps.spectrum as hc
import harps.inout as hio
import harps.lsf.gp_aux as gp_aux
from   harps.lsf.container import LSF2d
import harps.lsf.fit as hlsfit
import numpy as np
import matplotlib.pyplot as plt
import harps.plotter as hplt
from fitsio import FITS

filepath = '/Users/dmilakov/projects/lfc/data/harps/e2ds/2018-12/2018-12-04/'+\
    'HARPS.2018-12-05T08:12:52.040_e2ds_A.fits'
spec=hc.HARPS(filepath,fr=18e9,f0=4.58e9,overwrite=False)
#%%
# order = 50; index = 175 # save
order = 50; index = 20
lsf_filepath = hio.get_fits_path('lsf',filepath)
with FITS(lsf_filepath) as hdul:
    lsf2d = hdul['pixel_model',111].read()
LSF2d_nummodel = LSF2d(lsf2d)
# lsf2d_gp = LSF2d_gp[order].values
# lsf2d_numerical = hlsfit.numerical_model(lsf2d_gp,xrange=(-8,8),subpix=11)
# LSF2d_numerical = LSF2d(lsf2d_numerical)

linelist=spec['linelist']
cut=np.where((linelist['order']==order)&(linelist['index']==index))[0]
line=linelist[cut][0]

pixl = line['pixl']
pixr = line['pixr']
bary = line['bary']

x1l = np.arange(pixl,pixr)
flx1l = spec.flux[order,pixl:pixr]
bkg1l = spec['background'][order,pixl:pixr]
err1l = spec['error'][order,pixl:pixr]
#%%

def fit(x1l,flx1l,bkg1l,err1l,LSF1d_object,npars,interpolate=True):
    output_tuple = hlsfit.line(x1l,flx1l,bkg1l,err1l,LSF1d_object,
                               npars=npars,
                               interpolate=interpolate,
                               output_model=True,
                               output_rsd=True)
    success, pars, errors, cost, chisqnu, integral, model, rsd= output_tuple
    print(pars,errors,chisqnu)
    fig = hplt.Figure2(2,1,figize=(8,4),height_ratios=[3,1])
    ax1 = fig.add_subplot(0,1,0,1)
    ax2 = fig.add_subplot(1,2,0,1,sharex=ax1)
    ax1.errorbar(x1l,flx1l-bkg1l,err1l,drawstyle='steps-mid',capsize=3)
    ax1.plot(x1l,model-bkg1l,drawstyle='steps-mid',marker='x')
    ax1.text(0.8,0.9,r'$\chi^2_\nu=$'+f'{chisqnu:8.2f}',transform=ax1.transAxes)
    ax1.axvspan(pars[1]-2.5,pars[1]+2.5,alpha=0.1)
    ax1.axvspan(pars[1]-5,pars[1]+5,alpha=0.1)
    ax2.scatter(x1l,(model-flx1l)/err1l,label='outside')
    ax2.scatter(x1l,-rsd,label='infodict')
    
    N = 2 if interpolate else 1
    lsf_loc_x,lsf_loc_y = LSF1d_object.interpolate_lsf(pars[1],N)
    sct_loc_x,sct_loc_y = LSF1d_object.interpolate_scatter(pars[1],N)
    xgrid = np.linspace(x1l.min(), x1l.max(), 100)
    ygrid = hlsfit.lsf_model(lsf_loc_x,lsf_loc_y,pars,xgrid)
    ax1.plot(xgrid,ygrid,c='k',lw=2)
for npars in [3,4]:
    fit(x1l,flx1l,bkg1l,err1l,LSF2d_nummodel[order],npars)
#%% scipy.leastsq
# interpolate=True
# optpars, pcov, chisq, dof = gp_aux.get_params_scipy(lsf1d,x1l,flx1l-bkg1l,err1l,
#                                               interpolate=interpolate)
# gp_aux.plot_result(optpars,lsf1d,x1l,flx1l,bkg1l,err1l,interpolate=interpolate)
# #%% jax.jit
# interpolate = True
# optpars, pcov, chisq, dof = gp_aux.get_parameters(lsf1d,x1l,flx1l-bkg1l,err1l,
#                                             interpolate=interpolate)
# gp_aux.plot_result(optpars,lsf1d,x1l,flx1l,bkg1l,err1l,interpolate=interpolate)
# #%% scipy.optimise
# interpolate = True
# optpars, pcov, chisq, dof = gp_aux.get_parameters_opt(lsf1d,x1l,flx1l-bkg1l,err1l,
#                                             interpolate=interpolate)
# gp_aux.plot_result(optpars,lsf1d,x1l,flx1l,bkg1l,err1l,interpolate=interpolate)
# #%%
# output_tuple=gp_aux.fit_lsf2line(x1l,flx1l,bkg1l,err1l,lsf1d,interpolate=True,plot=True)
# success, pars, errors, chisq, chisqnu, integral = output_tuple



