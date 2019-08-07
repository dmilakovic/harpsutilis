#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:56:06 2019

@author: dmilakov


Script to determine any possible offset in the anchor frequency of HARPS during
2015 04 campaign
"""
import harps.classes   as hc
import harps.wavesol   as ws
import harps.plotter   as hplot
import harps.fit       as fit
import harps.functions as hf
from   harps.constants import c
import harps.compare   as compare
from   harps.containers import Generic
import matplotlib
import numpy as np

#%%
spec1=hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T04_27_14.341_e2ds_A.fits',LFC='HARPS')
spec2=hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T05_10_56.076_e2ds_A.fits',LFC='FOCES')
# spec2 has no attached ThAr calibration
spec2.ThAr = spec1.ThAr
#%%
linelist1 = spec1['linelist']
linelist2 = spec2['linelist']

# RECREATE THE ORIGINAL PROGRAM
#%%
anchorsteps = np.arange(-440e6,440e6,20e6)
orders     = [35,36]#50,51,52,55,65]
names      = ['HARPS','FOCES']
spectra    = {'HARPS':spec1, 'FOCES':spec2}
linelists  = {'HARPS':Generic(spec1['linelist']),
              'FOCES':Generic(spec2['linelist'])}
anchor     = {'HARPS':288060049999000, 'FOCES':288.08452e12-250e6}
repfreq    = {'HARPS':250e6, 'FOCES':250e6}
s          = {'HARPS':72, 'FOCES':100} # line filtering by FP etalons

#%% select lines common to both
ll1, ll2 = compare.overlapping_lines(linelist1,linelist2,'gauss')
cut      = np.where(np.abs(ll1['gauss'][:,1]-ll2['gauss'][:,1])<0.25)[0]
lines_harps,lines_foces = ll1[cut], ll2[cut]
#%%
nlines = len(cut)


rv_dict          = {}
residuals_dict   = {}


medianrvs = np.zeros((nlines,len(anchorsteps)),dtype=float)

for i in range(nlines):
    order = ll1[i]['order']
    m0    = {'HARPS':ll1[i]['index'],'FOCES':ll2[i]['index']}
    r0    = {'HARPS': 0, 'FOCES':0}
    line_wave = {'HARPS':hf.freq_to_lambda(ll1[i]['freq']), 
                 'FOCES':hf.freq_to_lambda(ll2[i]['freq'])}
    
    for ss,anchorstep in enumerate(anchorsteps):
        combwavesol_dict = {}
        for name in names:
            spec       = spectra[name]
            f0         = anchor[name]
            if name=='HARPS':
                f0 = anchor[name]+anchorstep
            spec1d = spec.data[order]
            lines  = linelists[name][order]
            cent1d = lines.values['gauss'][:,1]
            cerr1d = lines.values['gauss_err'][:,1]
            nlines1d = len(lines)
        
            # get the cardinal number of the wanted line in the comb
            m   = m0[name]
            r   = r0[name]
            lw  = line_wave[name]
            
            n   = round((c/lw*1e10-f0)/repfreq[name])
#            print("{:^70}".format(name))
#            print("{0:<40}{1:>20}".format("N:",n))
#            
        
            #print(lw, lp1, l0, lm1)
        
            # create a new array of wavelengths for each comb line
            # n = cardinal number of wanted line, determined above
            # s = 72 (HARPS) or 100 (FOCES)
            # f = lines added "by hand"
            # i = loop iterator
            freq1d = np.array([(f0 + ( n + (m-i)*s[name] + r)*repfreq[name]) for i in range(nlines1d)])
            wave1d = hf.freq_to_lambda(freq1d)
            freq_err = 20e6
            werr1d = 1e10*(c/(freq1d**2)) * freq_err
        
            # arrays to save data to
            combwavesol = np.zeros(4096)
            eval1d      = np.zeros_like(cent1d)
            coeffs      = fit.dispersion1d(cent1d,wave1d,cerr1d,werr1d,601)
            combwavesol = ws.disperse1d(coeffs,4096)

                
            combwavesol_dict[name] = combwavesol
            
            # Residuals
            #residuals              = (wave1d-eval1d)/wave1d*c

            
        rvHARPS_FOCES = (combwavesol_dict['HARPS']-combwavesol_dict['FOCES'])/combwavesol_dict['FOCES']*c
        medianrvs[i,ss] = np.median(rvHARPS_FOCES)
    hf.update_progress((i+1)/nlines)
#%%
import matplotlib.ticker as ticker
plt.style.use('paper')
plotter=hplot.Figure2(1,1,left=0.13)
ax = plotter.add_subplot(0,1,0,1)
#ax.set_title("Step size $\Delta f_0=20$ MHz")

y=medianrvs
textypos=140
ylims=(-100,180)
legpos='lower right'
ylabel='Median velocity shift ' + r'[${\rm ms^{-1}}$]'
N = np.shape(medianrvs)[0]
orders = np.unique(lines_harps['order'])

colors = plt.cm.jet(np.linspace(0, 1, N))

for i in range(N):
    order = lines_harps[i]['order']
#    j = np.argwhere(orders==order)[0]
    ax.scatter(anchorsteps/1e6,y[i],marker='o',lw=1,label="Order = {}".format(order),
               facecolors='None',edgecolors=colors[i],s=80,alpha=0.1)
ax.axhline(0,ls=':',lw=1)
[ax.axvline(-350+250*i,ls=':',lw=1) for i in range(4)]
#[ax.text(-380+250*i,textypos,"{} MHz".format(-350+250*i),rotation=90) for i in range(4)]
ax.set_xlabel("Offset in HARPS $f_0$ [MHz]")
#ax.xaxis.set_minor_locator(ticker.MultipleLocator(125))
ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
ax.set_xticks([-350,-100,150,400])
ax.set_ylabel(ylabel)
plotter.save('/Volumes/dmilakov/harps/dataprod/plots/v_1.1.0/harps_anchor_offset.pdf')
#ax.set_xlim(-500,500)
#ax.set_ylim(ylims)
#ax.set_ylim(ylims)
#ax.legend()