#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:55:27 2019

@author: dmilakov


Shows the AICc and Chisq distributions for various polynomial models
"""

from fitsio import FITS
import harps.plotter as hplot
import harps.settings as hs
import harps.functions as hf
import numpy as np
import os

import matplotlib
matplotlib.style.use('paper')
matplotlib.use('TKAgg')
#%%

#dsetA = FITS(os.path.join(hs.harps_dset,'2015-04-17-A.fits'))
dsetA=FITS('/lustre/opsw/work/dmilakov/harps/dataprod/dataset/v_1.1.0/2015-04-17-A.fits')
#dsetA=FITS('/Volumes/dmilakov/harps/dataprod/dataset/v_1.0.3/2015-04-17-A.fits')
#%%

chisq    = {}
aicc     = {}
dof      = {}
residarr = {}
#%%
fittype  = 'gauss'
versions = [201,210,301,310,401,410,501,510,601,610,701,710,801,810,
        901,910,1001,1010,1101,1110,1201,1210,1310,1410,1510,
        1610,1710,1810,1910,2010]
segver   = [201,301,401,501,601,701,801,901,1001,1101,1201]
globver  = [310,410,510,610,710,810,910,1010,1110,1210,1310,1410,1510,1610,
            1710,1810,1910,2010]
#versions = [201,210,301,310,401,410,500,501,510,601,610,701,710,800,801,810,
#        901,910,1001,1010,1101,1110,1201,1210,1301,1310,1401,1410,1501,1510,
#        1601,1610,1710,1810,1910,2010]
#%%
#versions = [810,1610,2010]
#residarr = {}
for i,ver in enumerate(versions):
    
    print(ver)
    
    p,g,s = hf.version_to_pgs(ver)
    coeff = dsetA['coeff_{}'.format(fittype),ver].read()
   
    if ver in dof.keys():
        pass
    else:
        dof_  = (p+1)*(1+7*s)
        dof[ver] = dof_
        
    if ver in residarr.keys():
        pass
    else:
        chisq_ = coeff['chisq']
        chisq[ver] = chisq_
        
    if ver in residarr.keys():
        pass
    else:
        aicc_  = coeff['aicc']
        aicc[ver]  = aicc_
        
#    if ver in residarr.keys():
#        pass
#    else:
#        rsd_ = np.hstack(dsetA['residuals_{}'.format(fittype),ver])
#        residarr[ver]=rsd_
    hf.update_progress((i+1)/len(versions),'read')
#chisq = np.array(chisq)
#aicc  = np.array(aicc)
#%%
use = 'mean'
if use == 'mean':
    func = np.mean
elif use == 'median':
    func = np.median
    
print((3*("{:^20s}")).format("VERSION","chisq {}".format(use),"AICc {}".format(use)))
veravailable = list(chisq.keys())
for i,ver in enumerate(veravailable):
    print(("{:^20d}"+2*("{:>20.3f}")).format(ver,func(chisq[ver]),func(aicc[ver])))
#logchisq = {ver:np.log10(arr) for ver,arr in chisq.items()}
#logaicc  = {ver:np.log10(arr) for ver,arr in aicc.items()}


#%%
medchisq = {ver:func(chisq_) for ver,chisq_ in chisq.items()}
medaicc  = {ver:func(aicc_[np.where(aicc_<np.quantile(aicc_,0.99))]) \
            for ver,aicc_ in aicc.items()}



#%%
#from scipy.optimize import curve_fit
#def lorentz(x,*p):
#    I,gamma, x0 = p
#    return I/(np.pi*gamma)*1/(1+((x-x0)/gamma)**2)
#def moffat(x,*p):
#    I, x0, gamma, beta = p
#    return I/(1 + ((x-x0)/gamma)**2)**beta
#xrange  = (-15,15)
#bins    = 2*xrange[1]
#hwhm    = {}
#hwhmplotter  = plot.Figure(1)
#ax_hwhm      = hwhmplotter.axes
#ax_hwhm[0].set_xlabel("Residuals [m/s]")
#
#labels   =["Polyord={}".format(p) for p in [hf.version_to_pgs(ver) for ver in versions]]
#for j,ver in enumerate(versions):
#    
#    data = residarr[ver]['residual_mps']#/residarr[ver]['noise']
#    cut  = np.where(np.abs(data)<xrange[1])[0]
#    data = data[cut]
#    mean = np.mean(data)
#    sigma= np.std(data)
#    vals,lims,obj = ax_hwhm[0].hist(data,bins=bins,range=xrange,
#                           histtype='step',lw=3,alpha=0.5,label=labels[j],
#                             density=False)
#    cens = (lims[1:]+lims[:-1])/2
#    # fit moffat
#    pars,covar = curve_fit(moffat,cens,vals,(np.max(vals),0,2,1))
##    print(pars)
##    print(covar)
#    I, x0, gamma, beta = pars
#    fac = (2*np.sqrt((2**(1/beta))-1))
#    fwhm = gamma*fac
#    hwhm_ = fwhm/2
#    print(("{:^20d}"+"{:>20.3f}").format(ver,hwhm_))
#    hwhm[ver] = hwhm_
#    errs = np.sqrt(np.diag(covar))
#    I_err,x0_err,gamma_err,beta_err = errs
#    fwhm_err = np.sqrt((gamma_err*fac)**2 )
#    
#figname = 'residuals_hwhm_comparison_{}.pdf'.format(fittype)
#dirname = hs.get_dirname('plots')
#hwhmplotter.save(os.path.join(dirname,figname))
#%%
dof_plotter = hplot.Figure2(2,1,left=0.12,hspace=0.3,height_ratios=[1,1])
fig_dof     = dof_plotter.figure
ax2         = dof_plotter.add_subplot(0,1,0,1)
ax1         = dof_plotter.add_subplot(1,2,0,1)
ax_dof      = [ax1, ax2]

print("Using: ",matplotlib.get_backend())

for i,poly in enumerate(['global','seg']):
    if poly == 'seg':
        usever = segver
        label = 'Segmented'
        scale = 'linear'
    elif poly == 'global':
        usever = globver
        label = 'Global'
        scale = 'linear'
    usedof  = np.array([dof[ver] for ver in usever])
    
    if poly == 'seg':
        usepoly = np.array(usedof)/8 - 1
    if poly=='global':
        usepoly = np.array(usedof) -1 
        
    
    useaicc = np.array([medaicc[ver] for ver in usever])
#    usehwhm = [hwhm[ver] for ver in usever]
    minaicc = np.argmin(useaicc)
    notmin  = np.where(np.arange(len(useaicc),dtype=np.int)!=minaicc)[0]
    ax_dof[i].scatter(usepoly[notmin],useaicc[notmin])
    ax_dof[i].scatter(usepoly[minaicc],useaicc[minaicc])
    ax_dof[i].ticklabel_format(axis='y', style='sci', scilimits=(-2, 4))
    
    ax_dof[i].set_xlim(1,21)
    xticks = np.arange(2,22,2)
    ax_dof[i].set_xticks(xticks)
    ax_dof[i].text(0.98,0.8,label,transform=ax_dof[i].transAxes,
              horizontalalignment='right')
    ax_dof[i].set_ylabel("{} AICc".format(use.capitalize()))
    ax_dof[i].set_yscale(scale)
    twiny = ax_dof[i].twiny()
    twiny.set_xticks(xticks)
    twiny.set_xlim(1,21)
    if poly=='seg':
        twiny.set_xticklabels(8*(xticks+1),fontsize=10)
    elif poly=='global':
        twiny.set_xticklabels((xticks+1),fontsize=10)
    print("Using: ",matplotlib.get_backend())

ax_dof[0].set_xlabel("Polynomial degree")
print("Using: ",matplotlib.get_backend())
#ax1.set_yscale('log')
if fittype=='lsf':
    ax1.set_ylim(6e3,1.8e3)
    ax2.set_ylim(130,350)
elif fittype=='gauss':
    ax1.set_ylim(6e2,1.8e3)
    ax2.set_ylim(82,108)
#fig_dof.text(0.03,0.55,'AICc',verticalalignment='center',rotation=90,
#             horizontalalignment='center')
fig_dof.align_ylabels()
figname = '{}_aicc_polynomial_{}.pdf'.format(use,fittype)
dirname = hs.get_dirname('plots')
filepath = os.path.join(*[dirname,'wavesolution','aicc',figname])
dof_plotter.save(filepath)
print("Plot saved to: ",filepath)
