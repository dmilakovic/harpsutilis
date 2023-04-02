#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 17:40:45 2023

@author: dmilakov
"""
import matplotlib.pyplot as plt
from fitsio import FITS
import numpy as np
#%%
# outpath = '/Users/dmilakov/projects/lfc/dataprod/fits/v_2.0/HARPS.2018-12-05T08:12:52.fits'
# outpath = '/Users/dmilakov/projects/lfc/dataprod/fits/v_2.0/HARPS.2018-12-10T05:25:48.fits'
outpath='/Users/dmilakov/projects/lfc/dataprod/from_bb/fits/v_2.0/HARPS.2018-12-05T08:12:52.fits'
hdu=FITS(outpath,'rw',clobber=False)
#%%
firstrow = 0 ; lastrow = 368 # od = 39
# firstrow = 369; lastrow=734 # od = 40
# firstrow = 734; lastrow=1034 # od = 41
# firstrow=3551 ; lastrow=3883 # od = 49
# firstrow=3884; lastrow=4216 # od = 50
# firstrow=4217; lastrow=4546 # od = 51
# firstrow = 3551; lastrow=4546
#%%
plt.figure()
gauss_chisqnu=hdu['linelist'].read(columns='gauss_pix_chisqnu')[firstrow:lastrow]
lsf_chisqnu=hdu['linelist',111].read(columns='lsf_pix_chisqnu')[firstrow:lastrow]
#plt.hist(gauss_chisqnu,histtype='step',bins=50,label='gauss',lw=2)
for it in [211,311]:
    
    lsf_chisqnu_it=hdu['linelist',it].read(columns='lsf_pix_chisqnu')[firstrow:lastrow]
    #diff = lsf_chisqnu_itm1-lsf_chisqnu_it
    plt.scatter(lsf_chisqnu,lsf_chisqnu_it,s=1,label=f"iteration={it}")
plt.legend()
#%%
plt.figure()
gauss_cens=hdu['linelist'].read(columns='gauss_pix')[firstrow:lastrow,1]
for it in [111,211,311,411,511]:#211,311,411]:
    lsf_cens=hdu['linelist',it].read(columns='lsf_pix')[firstrow:lastrow,1]
    diff = (lsf_cens-gauss_cens)*829
    plt.scatter(gauss_cens,diff,ls='-',s=2,label=f"iteration={it}")
[plt.axvline(_,ls=":") for _ in range(0,4097,256)]
plt.xlabel("Line centre (pix)")
plt.ylabel(r"LSF $-$ Gaussian centre"+r" (ms$^{-1}$)")
plt.legend()
#%%
plt.figure()
gauss_chisqnu=hdu['linelist'].read(columns='gauss_pix_chisqnu')[firstrow:lastrow]
lsf_chisqnu=hdu['linelist',111].read(columns='lsf_pix_chisqnu')[firstrow:lastrow]
for it in [211,311,411]:
    if it==211:
        lsf_chisqnu_itm1=lsf_chisqnu
    else:
        lsf_chisqnu_itm1=lsf_chisqnu_it
    lsf_chisqnu_it=hdu['linelist',it].read(columns='lsf_pix_chisqnu')[firstrow:lastrow]
    diff = lsf_chisqnu_itm1-lsf_chisqnu_it
    plt.scatter(lsf_chisqnu,diff,s=(it//100)*2,label=f"iteration={it}")
plt.legend()
#%%
plt.figure()
gauss_chisqnu=hdu['linelist'].read(columns='gauss_pix_chisqnu')[firstrow:lastrow]
# plt.hist(gauss_chisqnu,histtype='step',bins=50,label='Gauss')
for it in [111,211,311,411,511]:
    lsf_chisqnu=hdu['linelist',it].read(columns='lsf_pix_chisqnu')[firstrow:lastrow]
    plt.hist(lsf_chisqnu,histtype='step',bins=50,label=f'iteration={it}',
             lw=(it//100)*1.1,
             range=(0,5))
plt.legend()
#%%
import harps.lsf.aux as aux
plt.figure()
segm = 10
segsize=4096//16
pixl=segm*segsize; pixr = (segm+1)*segsize

x2d=np.vstack([np.arange(4096) for i in range(72)])
flx = hdu['flux'].read()
bkg = hdu['background'].read()
err = hdu['error'].read()
linelist = hdu['linelist',111].read()
# cut = np.where()


for fittype in ['gauss','lsf']:
    pix3d,vel3d,flx3d,err3d,orders=aux.stack(fittype,linelist[firstrow:lastrow],flx,
                            x2d,err,bkg,orders=50)
    plt.errorbar(pix3d[50,pixl:pixr,0],flx3d[50,pixl:pixr,0],
                err3d[50,pixl:pixr,0],marker='.',ls='',capsize=3)