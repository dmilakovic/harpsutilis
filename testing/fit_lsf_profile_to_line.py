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
import harps.lines_aux as laux
import harps.settings as hs
import harps.wavesol as ws
from fitsio import FITS
#%%

filepath = '/Users/dmilakov/projects/lfc/data/harps/e2ds/2018-12/2018-12-04/'+\
    'HARPS.2018-12-05T08:12:52.040_e2ds_A.fits'
blazepath = '/Users/dmilakov/projects/Q0515-4414/data/harps/reduced/blaze/'+\
    'reduced/2018-12-04/HARPS.2018-12-04T20:14:42.379_blaze_A.fits'
filepath = '/Users/dmilakov/projects/lfc/data/harps/e2ds/2018-12/2018-12-03/'+\
    'HARPS.2018-12-04T01:58:51.451_e2ds_A.fits'
lsf_filepath='/Users/dmilakov/projects/lfc/dataprod/v_2.2/lsf/'+\
    'HARPS.2018-12-05T08:12:52_lsf_most_likely.fits'
spec=hc.HARPS(filepath,fr=18e9,f0=4.58e9,overwrite=False,blazepath=blazepath)
#%%
# order = 50; index = 175 # save
version=1
# lsf_filepath = hio.get_fits_path('lsf',filepath)
with FITS(lsf_filepath) as hdul:
    lsf2d_pix = hdul['pixel_model',version].read()
    lsf2d_vel = hdul['velocity_model',version].read()
LSF2d_nummodel_pix = LSF2d(lsf2d_pix)
LSF2d_nummodel_vel = LSF2d(lsf2d_vel)
# lsf2d_gp = LSF2d_gp[order].values
# lsf2d_numerical = hlsfit.numerical_model(lsf2d_gp,xrange=(-8,8),subpix=11)
# LSF2d_numerical = LSF2d(lsf2d_numerical)


linelist=spec['linelist',version]

wav = ws.comb_dispersion(linelist=linelist, 
                           version=701, 
                           fittype='gauss',
                           npix=4096, 
                           nord=72)
#%%
order = 45; index = 20     
cut=np.where((linelist['order']==order)&(linelist['index']==index))[0]
line=linelist[cut][0]

pixl = line['pixl']
pixr = line['pixr']
bary = line['bary']
wav0 = line['gauss_wav'][1]


vel = (wav-wav0)/wav0*299792.458
flx = spec['flux']
bkg = spec['background']
env = spec['envelope']
err = spec['error']

data, data_error, bkg_norm = laux.prepare_data(flx,err,env,bkg,hs.subbkg,hs.divenv)

pix1l = np.arange(pixl,pixr)
flx1l = data[order,pixl:pixr]
err1l = data_error[order,pixl:pixr]
bkg1l = bkg_norm[order,pixl:pixr]
vel1l = wav[order,pixl:pixr]
#%%

def fit(x1l,flx1l,err1l,bary,LSF1d_object,scale,interpolate=True):
    output_tuple = hlsfit.line(x1l,flx1l,err1l,bary,
                               LSF1d_object,
                               scale = scale,
                               interpolate=interpolate,
                               output_model=True,
                               output_rsd=True,
                               plot=True)
    success, pars, errors, cost, chisqnu, integral, model, rsd, fig = output_tuple
    print(pars,errors,chisqnu,integral)
    
    
    
    
fit(pix1l,flx1l,err1l,bary,LSF2d_nummodel_pix[order],scale='pixel')
fit(vel1l,flx1l,err1l,bary,LSF2d_nummodel_vel[order],scale='velocity')
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



