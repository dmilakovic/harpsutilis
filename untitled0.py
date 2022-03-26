#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 19:33:06 2022

@author: dmilakov
"""
import harps.io as io
import harps.lsf as hlsf
import numpy as np
import jax
#%%
modeller=hlsf.LSFModeller('/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/test.dat',60,70,method='gp',subpix=10,filter=10,numpix=8,iter_solve=1,iter_center=1)

extensions = ['linelist','flux','background','error','wavereference']
data, numfiles = io.mread_outfile(modeller._outfile,extensions,701,
                        start=None,stop=None,step=None)
linelists=data['linelist']
fluxes=data['flux']
errors=data['error']
backgrounds=data['background']
# backgrounds=None
wavelengths=data['wavereference']
orders=np.arange(60,151)
pix3d,vel3d,flx3d,err3d,orders=hlsf.stack('gauss',linelists,fluxes,wavelengths,errors,backgrounds,orders)
#%%
od=145
pixl=5000
pixr=5500

pix1s=pix3d[od,pixl:pixr]#[:,0]
vel1s=vel3d[od,pixl:pixr]#[:,0]
flx1s=flx3d[od,pixl:pixr]#[:,0]
err1s=err3d[od,pixl:pixr]#[:,0]

test = False
if test==True:
    cond = np.where(~((pix1s>-2)&(pix1s<2)))
    vel1s=vel3d[od,pixl:pixr][cond]
    flx1s=flx3d[od,pixl:pixr][cond]
    err1s=err3d[od,pixl:pixr][cond]
    

vel1s_, flx1s_, err1s_ = hlsf.clean_input(vel1s,flx1s,err1s,sort=True,
                                          rng_key=jax.random.PRNGKey(55845),
                                          verbose=True,filter=5)
if test:
    vel1s_=np.append(vel1s_,[-0.5,+0.5,0.33])
    flx1s_=np.append(flx1s_,[0.06648128,0.04429982,0.04443524])
    err1s_=np.append(err1s_,[0.0029379,0.00034252,0.0027491])
# plt.errorbar(*[np.ravel(a) for a in [vel1s,flx1s,err1s]],marker='.',ls='')
lsf1s_100 = hlsf.construct_lsf1s(vel1s_,flx1s_,err1s_,'tinygp',
                                 plot=True,
                                 numiter=5,
                                 filter=None,
                                 save_plot=False,
                                model_scatter=False
                                 # model_scatter=True
                                 )
#%%

