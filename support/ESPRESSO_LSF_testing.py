#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:00:50 2022

@author: dmilakov
"""

import harps.lsf as hlsf
import numpy as np
import jax 

modeller=hlsf.LSFModeller('/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/single.dat',
                          100,101,method='tinygp',
                          subpix=10,filter=2,numpix=8,iter_solve=2,iter_center=2)
wavelengths = modeller['wavereference']
fluxes      = modeller['flux']
backgrounds = modeller['background']
errors      = modeller['error']
linelists   = modeller['linelist']
fittype     = 'gauss'
#%%
pix3d,vel3d,flx3d,err3d,orders = hlsf.stack(fittype,linelists,fluxes,
                                    wavelengths,errors,backgrounds,
                                    modeller._orders)
#%%
# orders = [91,92,93,94,99,101]
# orders = [95,100,105]
# orders = [60,62,65]
orders = [100]
lsf_i    = hlsf.construct_lsf(pix3d,flx3d,err3d,scale='pixel',
                         orders=orders,
                         numseg=modeller._numseg,
                         numpix=modeller._numpix,
                         subpix=modeller._subpix,
                         numiter=modeller._iter_center,
                         method=modeller._method,
                         filter=modeller._filter,
                         verbose=True)
#%%

od=100
seg = 5
pixl=9111//16*seg
pixr=9111//16*(seg+1)

pix1s=pix3d[od,pixl:pixr]#[:,0]
vel1s=vel3d[od,pixl:pixr]#[:,0]
flx1s=flx3d[od,pixl:pixr]#[:,0]
err1s=err3d[od,pixl:pixr]#[:,0]

test = False
if test==True:
    cond = np.where(~((pix1s>-1)&(pix1s<1)))
    vel1s=vel3d[od,pixl:pixr][cond]
    flx1s=flx3d[od,pixl:pixr][cond]
    err1s=err3d[od,pixl:pixr][cond]
    

rng_key=jax.random.PRNGKey(55825) # original
# rng_key=jax.random.PRNGKey(55826)
# rng_key=jax.random.PRNGKey(558257)
# rng_key=jax.random.PRNGKey(55822)
# rng_key=jax.random.PRNGKey(558214)
x1s_, flx1s_, err1s_ = hlsf.clean_input(
                                        # pix1s,
                                        vel1s,
                                        flx1s,err1s,sort=True,
                                          rng_key=rng_key,
                                          verbose=True,filter=None)
if test:
    x1s_=np.append(x1s_,[-0.5,+0.5,0.33])
    flx1s_=np.append(flx1s_,[0.6648128,0.84429982,0.4443524])
    err1s_=np.append(err1s_,[0.029379,0.084252,0.27491])
# plt.errorbar(*[np.ravel(a) for a in [vel1s,flx1s,err1s]],marker='.',ls='')
lsf1s_100 = hlsf.construct_lsf1s(x1s_,flx1s_,err1s_,'tinygp',
                                 plot=True,
                                 numiter=2,
                                 filter=None,
                                 save_plot=False,
                                 model_scatter=False
                                 # model_scatter=True
                                 )
#%%
from matplotlib import ticker
import harps.plotter as hplot
plotter = hplot.Figure2(2,2,left=0.15,bottom=0.15,figsize=(4,3))
figure = plotter.fig
axes = [plotter.ax() for i in range(4)]

figure.text(0.55,0.05,"Distance from center"+r" [kms$^{-1}$]",
            horizontalalignment='center',
            verticalalignment='center')
figure.text(0.05,0.5,"Relative intensity",rotation=90,
            horizontalalignment='center',
            verticalalignment='center')


filelist = ['/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/ESPRESSO_65_vel.fits',
            '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/ESPRESSO_75_vel.fits',
            # '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/ESPRESSO_95_vel.fits',
            '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/ESPRESSO_105_vel.fits',
            '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/ESPRESSO_110_vel.fits'
            ]


# for i,seg in enumerate([2,6,10,12]):
for i,file in enumerate(filelist):
    ax = axes[i]
    # ax = axes[0]
    LSF=hlsf.from_file(file,-1)
    values = LSF.values
    
    ax.plot(values[8]['x'],values[8]['y'])
    ax.set_ylim(-5,100)
    ax.set_xlim(-5,5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5,steps=[1,2,5]))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    #        ax.set_yticklabels([])
    ax.grid(True,ls=':',lw=1,which='both',axis='both')
    
    #%%
from matplotlib import ticker
import harps.plotter as hplot
import numpy as np
import matplotlib.cm as cm
plotter = hplot.Figure2(1,1,left=0.2,bottom=0.2,figsize=(3,2))
figure = plotter.fig
axes = [plotter.ax() for i in range(1)]

figure.text(0.55,0.05,"Distance from center"+r" [pix]",
            horizontalalignment='center',
            verticalalignment='center')
figure.text(0.05,0.5,"Relative intensity",rotation=90,
            horizontalalignment='center',
            verticalalignment='center')

file = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/ESPRESSO_78.fits'
LSF=hlsf.from_file(file,-1)
values = LSF.values
# colors = cm.jet(np.linspace(0,1,4))
colors = ['blue','green','orange','red']
for i,seg in enumerate([2,8,10,13]):
# for i,file in enumerate(filelist):
    # ax = axes[i]
    ax = axes[0]
    pixl = values[seg]['pixl']
    pixr = values[seg]['pixr']
    x    = values[seg]['x']
    y    = values[seg]['y']
    
    ax.plot(x,y/np.max(values['y'])*100, c = colors[i])
    ax.set_ylim(-2,102)
    ax.set_xlim(-5.5,5.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5,steps=[1,2]))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    #        ax.set_yticklabels([])
    ax.grid(True,ls=':',lw=1,which='both',axis='both')
    # ax.legend()
