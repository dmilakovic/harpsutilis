#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 19:33:06 2022

@author: dmilakov
"""
import harps.io as io
import harps.lsf as hlsf
import numpy as np
#%%
modeller=hlsf.LSFModeller('/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/test.dat',60,70,method='gp',subpix=10,filter=10,numpix=8,iter_solve=1,iter_center=1)

extensions = ['linelist','flux','background','error','wavereference']
data, numfiles = io.mread_outfile(modeller._outfile,extensions,701,
                        start=None,stop=None,step=None)
linelists=data['linelist']
fluxes=data['flux']
errors=data['error']
# backgrounds=data['background']
backgrounds=None
wavelengths=data['wavereference']
orders=np.arange(60,151)
pix3d,vel3d,flx3d,err3d,orders=hlsf.stack('gauss',linelists,fluxes,wavelengths,errors,backgrounds,orders)
#%%
od=100
pixl=2500
pixr=3000

pix1s=pix3d[od,pixl:pixr]
vel1s=vel3d[od,pixl:pixr]
flx1s=flx3d[od,pixl:pixr]
err1s=err3d[od,pixl:pixr]

vel1s_, flx1s_, err1s_ = hlsf.clean_input(vel1s,flx1s,err1s,sort=True,verbose=True,filter=10)
# plt.errorbar(*[np.ravel(a) for a in [vel1s,flx1s,err1s]],marker='.',ls='')
lsf1s_100 = hlsf.construct_lsf1s(vel1s_,flx1s_,err1s_,'tinygp',plot=True,numiter=5,
                                 save_plot=True,filter=None)
