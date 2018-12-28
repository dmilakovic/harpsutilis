#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:12:31 2018

@author: dmilakov


Compare ThAr solutions. 
"""
import os
import numpy as np
from harps.wavesol import ThAr

import harps.functions as hf
import harps.plotter as hplt
import harps.containers as container
import scipy.stats as stats
import matplotlib.pyplot as plt
from harps.constants import c


#filepath = '/Users/dmilakov/harps/dataprod/input/e2dslist/2015-04-17_all.txt'
filepath = '/Users/dmilakov/observations/HE0515-4414/102.A-0697/data/reduced_e2ds.txt'
tharlist = np.sort(hf.read_filelist(filepath))
  

tharsols = {'A':[],'B':[]}
files    = {'A':[],'B':[]}
lines    = {'A':[],'B':[]}
coeffs   = {'A':[],'B':[]}
refindex=0

for i,file in enumerate(tharlist):
    print("Filename = ",file)
    fibre = os.path.splitext(os.path.basename(file))[0][-1]
    thar = ThAr(file,vacuum=True)
    cff = thar.coeffs
    cf = coeffs[fibre]
    cf.append(cff)
    
    ws = thar()
    wl = tharsols[fibre]
    wl.append(ws)
    
    fl = files[fibre]
    fl.append(file)
#%%  
shift_wave = {'A':[],'B':[]}
for fibre in ['A','B']:
    wavefib  = np.array(tharsols[fibre])
    thar_rv  = container.radial_velocity(len(wavefib))
    mask     = np.zeros_like(wavefib)
    mask[:,48,:]=1
    wavesol  = np.ma.array(wavefib,mask=mask)
    wavesol  = wavesol[:,43:,:]
    waveref  = wavesol[refindex]
    # RV shift in pixel values
    wavediff = (waveref - wavesol)/waveref * c

    for i,file in enumerate(files[fibre]):
        print(i,file,fibre)
        #fibre    = os.path.splitext(os.path.basename(file))[0][-1]
        datetime = hf.basename_to_datetime(str(file))
        clipped  = stats.sigmaclip(wavediff[i]).clipped
        thar_rv[i]['rv'] = np.mean(clipped)
        thar_rv[i]['datetime'] = datetime
        thar_rv[i]['fibre']=fibre
    shift_wave[fibre] = thar_rv
#%% PLOT RV SHIFT
fig = hplt.Figure(1)
dictarr = shift_wave
for fibre in ['A','B']:
    #cut  = np.where(shift['fibre']==fibre)
    data = dictarr[fibre]['rv']
    fig.plot(0,data)
#fig.axes[0].set_ylim(-1.5,1.5)   
fig.plot(0,(dictarr['A']['rv']-dictarr['B']['rv']))      
#%% PLOT 2d IMAGES
ref = 0
fibre = 'A'
lim=100
numexp = len(tharsols[fibre])
for idx in range(numexp):
    rv  = c*(tharsols[fibre][ref]-tharsols[fibre][idx])/tharsols[fibre][ref]
    
    fig,ax = hf.figure(1,figsize=(10,8))
    
    ax[0].set_title("ThAr Exp={0} Ref={1} Fibre={2}".format(idx,ref,fibre))
    ax[0].set_xlabel("Px")
    ax[0].set_ylabel("Order")
    
    plot = ax[0].imshow(rv,aspect='auto',vmin=-lim,vmax=lim)
    
    cbar = fig.colorbar(plot)
    cbar.set_label("RV [m/s]")
    figname = "rv_thar_fib{0}_exp{1:03d}_ref{2:03d}.pdf".format(fibre,idx,ref)
    figpath = os.path.join('/Users/dmilakov/harps/dataprod/plots/tharsol_rv/wavesol',
                           figname)
    #fig.savefig(figpath)
#%%
## PLOT COEFFICIENTS VS TIME
fibre = 'A'

coeffs_fib = np.array(coeffs[fibre])
nspec,nord,nseg,npar = np.shape(coeffs_fib['pars'])

ref = 0
for order in range(nord):
    fig, ax = hf.figure(4,alignment='grid',sep=0.06,
                        title='ORDER = {}'.format(order))
    pix = 2048
    all_pars = np.reshape(coeffs_fib['pars'],(30,72,4))
    lbd0     = hf.polynomial(pix,*all_pars[ref,order])
    
    for i in range(npar):
        pars = all_pars[:,order,i]
        
        if np.all(pars==0):
            continue
        else:
            pars_masked = np.ma.masked_equal(pars,0)
        ax[i].set_title(r"$a_{}$".format(i))
        ax[i].plot(pars_masked,marker='o',ls='',ms=3)
        ymin,ymax = np.percentile(pars_masked,[10,90])
        yran = ymax-ymin
        
        ymean = np.mean(pars_masked)
        ystd  = np.std(pars_masked)
        #ax[i].set_ylim(ymin,ymax)
    figname = "coeff_thar_fib{0}_ord{1}.pdf".format(fibre,order)
    figpath = os.path.join('/Users/dmilakov/harps/dataprod/plots/tharsol_rv/coeff',
                           figname)
    #fig.savefig(figpath)
#%%
## PLOT wavelength of a pixel VS time
fibre = 'B'

coeffs_fib = np.array(coeffs[fibre])['pars'][:,:,0,:]
nspec,nord,npar = np.shape(coeffs_fib)

pixels = [2048]

ref = 100

polyvec=np.vectorize(hf.polynomial)

#for order in range(nord):
for order in range(10,11):  
    for pix in pixels:
        lbd0 = hf.polynomial(pix,*coeffs_fib[ref,order,:])
        pars = coeffs_fib[:,order,:]
        
        fig, ax = hf.figure(1)
        ax[0].set_title("RV for pixel {0}, order {1}, fibre {2}".format(pix,order,fibre))
        ax[0].axhline(0,ls=':',c='k',lw=0.5)
        ax[0].set_xlabel('Exposure number')
        ax[0].set_ylabel('RV [m/s]')
        if np.all(pars==0):
            continue
        else:
            pars_masked = np.ma.masked_equal(pars,0)
            
        lbd = polyvec(pix,*pars_masked.T)
        
        rv = (lbd-lbd0)/lbd0*2.99792458e8
        
        ax[0].plot(rv,marker='o',lw=1,ms=3)
        ymin,ymax = np.percentile(rv,[10,90])
        yran = ymax-ymin
        if ymin!=ymax:
            ax[0].set_ylim(ymin-yran,ymax+yran)
        figname = "rv_thar_fib{0}_ord{1}_pix{2}.pdf".format(fibre,order,pix)
        figpath = os.path.join('/Users/dmilakov/harps/dataprod/plots/tharsol_rv/pix',
                               figname)
        #fig.savefig(figpath)