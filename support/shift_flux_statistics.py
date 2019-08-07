#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:26:17 2019

@author: dmilakov

Plot line shifts as a function of flux (to show that CTI correction can not be
done on extracted 2d spectra and individual lines, but needs to be done on raw
frames or average RV shifts of an exposure).


"""
from fitsio import FITS

import os
import numpy as np
import matplotlib
import harps.classes     as hc
import harps.plotter     as hplot
import harps.wavesol     as ws
import harps.containers  as cont
import harps.functions   as hf
import harps.settings    as hs
import scipy.stats       as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from   matplotlib.colorbar import ColorbarBase
from   matplotlib.colors   import Normalize,ListedColormap, LinearSegmentedColormap
from   matplotlib.cm       import ScalarMappable
from   matplotlib          import cm


matplotlib.style.use('paper')
#%%
fittype = 'gauss'
fibre   = 'A'
#%%
dset_dir  = hs.get_dirname('dataset')
dset_name = '2015-04-17-gray-{}.fits'.format(fibre)
dset      = FITS(os.path.join(dset_dir,dset_name))
#%%
dset_dir  = '/Volumes/dmilakov/harps/dataprod/dataset/v_1.0.3'
dset_name = '2012-02-15-{}.fits'.format(fibre)
dset      = FITS(os.path.join(dset_dir,dset_name))
#%%
mlinelist = cont.General(dset['linelist'].read())
coeffs    = cont.General(dset['coeff_{}'.format(fittype),601].read())
#%%
refexp = 0
amps=[]
shifts=[]
lines = []
coeff = coeffs.select(dict(exp=refexp)).values
for exp in range(194):
    linelist = mlinelist.select(dict(exp=exp)).values
    lines.append(linelist)
    
    shift=ws.residuals(linelist,coeff,fittype=fittype,version=601)
    shifts.append(shift['residual_mps'])
    
    amps.append(linelist['{}'.format(fittype)][:,0])
    
rsd = np.hstack(shifts)
flx = np.hstack(amps)
lns = np.hstack(lines)

#%%
m    = hf.sigclip1d(rsd,3)
n    = flx<np.percentile(flx,99.95)
bins = hf.histedges_equalN(flx[m&n],5)
rsd_mean,binedge,binnum = ss.binned_statistic(flx[m],rsd[m],'mean',bins=bins)
rsd_std,binedge,binnum  = ss.binned_statistic(flx[m],rsd[m],'std',bins=bins)
rsd_cnt,binedge,binnum  = ss.binned_statistic(flx[m],rsd[m],'count',bins=bins)

bincen = 0.5*(binedge[1:]+binedge[:-1])

plotter = hplot.Figure2(2,1,height_ratios=[1,3],left=0.12)
ax0     = plotter.add_subplot(0,1,0,1)
ax1     = plotter.add_subplot(1,2,0,1)

xlim = (-1e4,1.1*np.percentile(flx,99.95))

ax0.hist(flx[m],bins=100,histtype='step',lw=2)
ax0.set_ylabel("#")

#plt.scatter(flux[m],rsdg[m],s=1,marker='o',edgecolor='gray',color='None',rasterized=True)
ax1.errorbar(bincen,rsd_mean,yerr=rsd_std,marker='o',c='k',capsize=4)
ax1.axhline(0,ls='--',lw=2,c='k')
ax1.set_xlabel("Line amplitude [counts]")
ax1.set_ylabel("Average line shift [m/s]")
ax1.set_ylim(-8.5,8.5)
ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))


for ax in plotter.axes:
    [ax.axvline(be,ls=':',lw=1) for be in binedge]
    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
ax0.set_xticklabels([])
ax0.yaxis.set_major_locator(ticker.MaxNLocator(2))
ax0.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))

plotter.figure.align_ylabels(plotter.axes)
dirname = os.path.join(*[hs.get_dirname('plots'),'cti'])
figname = 'shift_vs_flux_{}_{}.pdf'.format(fittype,fibre)
figpath = os.path.join(dirname,figname)
#plotter.save(figpath)
#%%
orders = np.unique(lns['order'])

mean_sft2d,xedge,yedge,binnum=ss.binned_statistic_2d(lns['order'][n&m],
                                                     lns['gauss'][n&m][:,1],
                                                     rsd[n&m],
                                                     'mean',bins=[41,100])
xcen=0.5*(xedge[1:]+xedge[:-1])-0.5
ycen=0.5*(yedge[1:]+yedge[:-1])
xx,yy=np.meshgrid(np.flip(ycen),xcen)
plt.figure()
plt.scatter(xx,yy,c=mean_sft2d,cmap='coolwarm',vmin=-10,vmax=10,s=32,marker='s')
plt.colorbar(label='Shift [m/s]')
#plt.xlabel("Pixel")
#plt.ylabel("Diffraction order")

#%% take an example of three lines
order = [49,49,49,32,32,32]
index = [0+2,241-2,20,100,145,190]
markers = ['o','D','s','X','^','*']

### NEW COLORMAP
bottom = cm.get_cmap('Blues', 128)
top    = cm.get_cmap('Oranges_r', 128)

newcolors = np.vstack((bottom(np.linspace(1, 0, 128)),
                       top(np.linspace(1, 0, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')
###
cmap = newcmp#'RdGy'


plotter = hplot.Figure2(1,2,left=0.12,right=0.9,
                        width_ratios=[40,1])
ax      = plotter.add_subplot(0,1,0,1)
cbarax  = plotter.add_subplot(0,1,1,2)
#cutmask = []

inset = plotter.figure.add_axes([0.65,0.15,0.18,0.28])
exp0  = mlinelist.select(dict(exp=0)).values

inset.scatter(exp0[fittype][:,1],exp0['order'],
              c = np.sqrt(exp0[fittype][:,0]),s=0.4,cmap='gray',
              rasterized=True)
inset.set_xticks([])
inset.set_yticks([])
ax.arrow(0.67,0.64,0.14,0.0,transform=plotter.figure.transFigure,head_width=0.007)
ax.arrow(0.642,0.67,0.0,0.24,transform=plotter.figure.transFigure,head_width=0.007)
ax.text(0.74,0.615,r'$\lambda$',transform=plotter.figure.transFigure,fontsize=10)
ax.text(0.62,0.79,r'$\lambda$',transform=plotter.figure.transFigure,fontsize=10)

#inset.set_xticklabels(['blue','red'],fontsize=10)
#inset.set_yticklabels(['blue','red'],fontsize=10)




off = 20
vmin,vmax = -3,3

norm = Normalize(vmin=vmin, vmax=vmax)
cmob = ScalarMappable(norm,cmap)

refline1 = mlinelist.select(dict(exp=0,order=49,index=20))

refline2 = mlinelist.select(dict(exp=0,order=32,index=145))


for k,od,i in zip(np.arange(len(order)),order,index):
    condict = dict(order=od,index=i)
    line  = mlinelist.select(condict)
    cut   = mlinelist.cut(dict(order=od,index=i,exp=0))
#    cutmask.append(cut[0][0])
    
    if od==32:
        refline = refline2
    elif od==49:
        refline = refline1
    else:
        assert(od in order)
    
    refshift = ws.residuals(refline.values,coeff,fittype=fittype,version=601)
    refA     = refline.values['gauss'][:,0]


    A    = line.values[fittype][:,0]
    mean = line.values[fittype][:,1]
    fluxfrac = (A/refA-1)*100
#    colors = cmob.to_rgba(fluxfrac)
    
    shift = ws.residuals(line.values,coeff,fittype=fittype,version=601)
    
    
    censhift = (shift[fittype]-shift[fittype][0])/shift[fittype][0]*829
    y0     = shift['residual_mps']/refshift['residual_mps']
    y      = y0-np.mean(y0)
#    y     = np.diff(line.values['gauss'][:,1])*829
    x     = np.arange(1,len(y)+1)
    colors = cmob.to_rgba(y)
    
    print(k,np.mean(A),np.mean(shift['noise']),hf.rms(y-y.mean()),np.min(fluxfrac),np.max(fluxfrac))
    print(k,'optord = {0:3d}'.format(line.values['optord'][0]),
          'lambda = {0:8.3f}'.format(hf.freq_to_lambda(line.values['freq'][0])))
   

    jump = -off * (k-2)
    ebar = ax.scatter(fluxfrac,
                censhift,#+jump,
#                yerr=shift['noise'],
                marker=markers[k],lw=1,
                edgecolor='grey',
#                elinewidth=0.5,
#                ms=48,
                c=colors,
#                mec = 'C{}'.format(k),
#                ecolor='C{}'.format(k),
#                color=colors,
                vmin = vmin,
                vmax = vmax,
                label="A = {0:8.1e}".format(A[0]),
                rasterized=True)
    
#    ebar = ax.errorbar([4e4],
#                jump,
#                yerr=np.mean(shift['noise']),
#                marker=markers[k],lw=1,
#                capsize=3,
#                elinewidth=2,
#                ls='',
#                c='None',
#                mec = 'None',
#                ecolor='k',
#                rasterized=True)
#    ax.axhline(jump,ls=':',lw=1,c='k')
    inset.scatter(exp0[fittype][cut,1],exp0['order'][cut],
              marker=markers[k],s=32,
              c='k',
              edgecolor='white',
              rasterized=True)
#ax.set_xlim(-25,30)
#ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.set_ylabel("Line shift [m/s]")
ax.set_xlabel("Line amplitude relative to reference [%]")
#ax.set_xscale('log')
#ax.legend(ncol=2)

cb1 = ColorbarBase(ax=cbarax,norm=norm,cmap=cmap,
                   label='Deviation from mean line amplitude [%]')
cbarax.ticklabel_format(scilimits=(-2,4))
cbarax.yaxis.set_major_locator(ticker.MultipleLocator(5))
dirname = os.path.join(*[hs.get_dirname('plots'),'cti'])
figname = 'shift_vs_flux_position_{}_{}.pdf'.format(fittype,fibre)
figpath = os.path.join(dirname,figname)
#plotter.save(figpath)


#%% take an example of three lines
order = [49,49,49,32,32,32]
index = [0+2,241-2,20,100,145,190]
markers = ['o','D','s','X','^','*']


cmap = newcmp#'RdGy'


plotter = hplot.Figure2(1,2,left=0.12,right=0.9,
                        width_ratios=[40,1])
ax      = plotter.add_subplot(0,1,0,1)
cbarax  = plotter.add_subplot(0,1,1,2)
#cutmask = []

inset = plotter.figure.add_axes([0.65,0.65,0.18,0.28])
exp0  = mlinelist.select(dict(exp=0))

inset.scatter(exp0.values[fittype][:,1],exp0.values['order'],
              c = np.sqrt(exp0.values[fittype][:,0]),s=0.4,cmap='gray',
              rasterized=True)
inset.set_xticks([])
inset.set_yticks([])
ax.arrow(0.67,0.64,0.14,0.0,transform=plotter.figure.transFigure,head_width=0.007)
ax.arrow(0.642,0.67,0.0,0.24,transform=plotter.figure.transFigure,head_width=0.007)
ax.text(0.74,0.615,r'$\lambda$',transform=plotter.figure.transFigure,fontsize=10)
ax.text(0.62,0.79,r'$\lambda$',transform=plotter.figure.transFigure,fontsize=10)

#inset.set_xticklabels(['blue','red'],fontsize=10)
#inset.set_yticklabels(['blue','red'],fontsize=10)




off = 20
vmin,vmax = -0.5,0.5

norm = Normalize(vmin=vmin, vmax=vmax)
cmob = ScalarMappable(norm,cmap)


for k,od in enumerate([68]):
    condict = dict(order=od)
    lines  = mlinelist.select(condict)
    reflin = mlinelist.select(dict(exp=0,order=od))
    A0     = np.tile(reflin.values[fittype][:,0],194)
    
    
    A      = lines.values[fittype][:,0]
    pos1    = lines.values[fittype][:,1].reshape(194,-1).mean(axis=0)
    pos    = np.tile(pos1,194)
    fluxfrac = (A/A0)*100
#    colors = cmob.to_rgba(fluxfrac)
    
    shift  = ws.residuals(lines.values,coeff,fittype=fittype,version=601)
    
    
    censhift = (shift[fittype]-shift[fittype][0])/shift[fittype][0]*829
    y0     = shift['residual_mps']#/refshift['residual_mps']
    y      = y0-np.mean(y0)
    y     = (lines.values['gauss'][:,1]-pos)/pos*829
    x     = np.arange(1,len(y)+1)
    colors = cmob.to_rgba(y)
    
    print(k,np.mean(A),np.mean(shift['noise']),hf.rms(y-y.mean()),np.min(fluxfrac),np.max(fluxfrac))
    print(k,'optord = {0:3d}'.format(line.values['optord'][0]),
          'lambda = {0:8.3f}'.format(hf.freq_to_lambda(line.values['freq'][0])))
   

    jump = -off * (k-2)
    ebar = ax.scatter(pos,
                fluxfrac,#+jump,
#                yerr=shift['noise'],
                marker=markers[k],lw=1,
                edgecolor='grey',
#                elinewidth=0.5,
#                ms=48,
                c=colors,
#                mec = 'C{}'.format(k),
#                ecolor='C{}'.format(k),
#                color=colors,
                vmin = vmin,
                vmax = vmax,
                label="A = {0:8.1e}".format(A[0]),
                rasterized=True)
    
#    ebar = ax.errorbar([4e4],
#                jump,
#                yerr=np.mean(shift['noise']),
#                marker=markers[k],lw=1,
#                capsize=3,
#                elinewidth=2,
#                ls='',
#                c='None',
#                mec = 'None',
#                ecolor='k',
#                rasterized=True)
#    ax.axhline(jump,ls=':',lw=1,c='k')
#    cut   = mlinelist.cut(dict(order=od,exp=0))
#    inset.scatter(exp0[fittype][cut,1],exp0['order'][cut],
#              marker=markers[k],s=32,
#              c='k',
#              edgecolor='white',
#              rasterized=True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1024))
ax.set_xlabel("Line position")
ax.set_ylabel("Line amplitude relative to reference [%]")
#ax.set_xscale('log')
#ax.legend(ncol=2)

cb1 = ColorbarBase(ax=cbarax,norm=norm,cmap=cmap,
                   label='Line shift [m/s]')
cbarax.ticklabel_format(scilimits=(-2,4))
cbarax.yaxis.set_major_locator(ticker.MultipleLocator(5))
dirname = os.path.join(*[hs.get_dirname('plots'),'cti'])
figname = 'shift_vs_flux_position_{}_{}.pdf'.format(fittype,fibre)
figpath = os.path.join(dirname,figname)
#plotter.save(figpath)