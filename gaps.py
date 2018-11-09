#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:09:27 2018

@author: dmilakov

USAGE : python gaps.py $path/to/settings/file.json

The settings file must have the 'outlist' parameter.

Calculate gaps from 'out' FITS files.

(1) Read the residuals to the LFC wavelength calibration, version 500
    5th order polynomial is the largest order polynomial that is within 64 bits
    for 4096**5
    
(2) Bin the residuals into 8 bins per each 512 pixel segment.
    This would give approximately 5 lines per exposure per bin. 
    
(3) Fit a 3th order polynomial to the binned residuals. With 8 bins, this would
    make the number of degrees of freedom DOF = 8 bins - 4 parameters = 4.
    This seems reasonable, not to overfit the data
    
(4) Gaps are defined as the size of the discontinuity when the polynomials from
    step (3) are evaluated at the border pixel of the segment from the left and
    from the right.
    
    GAP = f_right(x=512*i) - f_left(x=512*i)
    
    x is pixel, i is the segment number (1-8).
    
    
"""

from harps.core import np, sys, os, time, json
from harps.core import curve_fit
from harps.core import FITS

import harps.settings as hs
import harps.functions as hf
import harps.containers as container
from harps.wavesol import evaluate

#%%
def read_data(settings):
    
    
    outlist = hf.read_filelist(settings.outlist)
    
    residlist = []
    linelist  = []
    for file in outlist:
        print("Filename = ",file)
        with FITS(file,'r') as fits:
            hdu_r = fits['residuals',500]
            
            residlist.append(hdu_r.read())
            hdu_l = fits['linelist']
            linelist.append(hdu_l.read())
    residarr = np.hstack(residlist)
    linesarr  = np.hstack(linelist)
    return residarr, linesarr
#%%
def cut_data(residarr,linesarr):
    
    centers0   = linesarr['gauss'][:,1]
    residuals0 = residarr['residual']
    # use only red chip 
    redchip    = np.where(residarr['order']>44)
    centers1   = centers0[redchip]
    residuals1 = residuals0[redchip]
    # remove outliers
    limit      = 200
    cut        = np.where(np.abs(residuals1)<limit)
    residuals  = residuals1[cut]
    centers    = centers1[cut]
    return residuals,centers

def get_binlims(nsegs,nsubins):  
    nbins   = nsegs*nsubins  
    seglims   = np.linspace(0,4096,nsegs+1)
    binlims   = np.linspace(0,4096,nbins+1)
#    bincen   = (binlims[1:]+binlims[:-1])/2
    return seglims,binlims
#%%
def bin_data(residuals,centers,nsegs,nsubins):
    nbins   = nsegs*nsubins  
    seglims   = np.linspace(0,4096,nsegs+1)
    binlims   = np.linspace(0,4096,nbins+1)
    bincen   = (binlims[1:]+binlims[:-1])/2
    
    binned   = np.digitize(centers,binlims)
    binval   = np.zeros_like(bincen)
    binstd   = np.zeros_like(bincen)
    nbins    = len(bincen)
    for i in range(nbins):
        inbin = np.where(binned==i+1)[0]
        binval[i] = np.mean(residuals[inbin])
        binstd[i] = np.std(residuals[inbin])
    return bincen, binval, binstd
#%%
def fit_binned(seglims,bincen,binval,binstd,polyord):
    nsegs    = len(seglims)-1
    coeffs   = container.coeffs(polyord,nsegs)
    fitresids = []
    for i in range(nsegs):
        pixl  = seglims[i]
        pixr  = seglims[i+1]
        
        inseg = np.where((bincen>pixl)&(bincen<pixr))[0]
        binsh = bincen[inseg]-pixl
        res   = curve_fit(hf.polynomial,binsh,binval[inseg],
                          sigma=binstd[inseg], p0 = np.ones(polyord+1))
        pars  = res[0]
        cov   = res[1]
        errs  = [np.sqrt(cov[i][i]) for i in range(polyord+1)]
        resd  = binval[inseg] - evaluate(pars,binsh)
        dof   = len(inseg) - len(pars)
        chi2  = np.sum(resd**2 / binstd[inseg]**2) / dof
        fitresids.append(resd)
        coeffs['segm'][i] = i
        coeffs['pixl'][i] = pixl
        coeffs['pixr'][i] = pixr
        coeffs['pars'][i] = pars
        coeffs['errs'][i] = errs
        coeffs['chi2'][i] = chi2
    fitresids = np.array(fitresids)
    return coeffs,fitresids
#%%  C A L C U L A T E   G A P S
def calculate_gaps(coeffs,nsegs):
    x = np.linspace(0,512,2)
    vals = np.zeros((nsegs,len(x)))
    for i,c in enumerate(coeffs):
        pars    = c['pars']
        vals[i] = hf.polynomial(x,*pars)
    leftvals, rightvals = np.transpose(vals)
    # in units pixel, assuming 1 pix = 829 m/s
    gaps = np.array([leftvals[i+1]-rightvals[i] for i in range(nsegs-1)])/829.
    return gaps, vals
#%%  S A V E   C O E F F I C I E N T S    T O   F I L E
def save_gaps(settings,gaps,bincen,binval,binlims,seglims,polyord,nsegs,nsubins):
    
    gapspath = os.path.join(settings.outdir,settings.selfname+'_gaps.json')
    settings.append('gapspath',gapspath)
    
    data = {"created_by":settings.selfpath,
            "created_on":time.strftime("%Y-%m-%dT%H_%M_%S"),
            "dataset":settings.outlist,
            "bin_limits":list(binlims),
            "bin_centers":list(bincen),
            "bin_values":list(binval),
            "seg_limits":list(seglims),
            "polyord":polyord,
            "gaps_pix":list(gaps),
            "gaps_mps":list(gaps*829.)
            }
    if os.path.isfile(gapspath):
        mode = 'a'
    else:
        mode = 'w'
    with open(gapspath,mode) as file:
        json.dump(data,file,indent=4)
    return

save_gaps(settings,gaps,bincen,binval,binlims,seglims,polyord,nsegs,nsubins)
#%%
def main():
    filepath = '/Users/dmilakov/harps/dataprod/settings/series1_fibreA.json'
    #settings = hs.Settings(sys.argv[1])
    settings = hs.Settings(filepath)
    
    polyord = 3
    
    nsegs = 8
    
    nsubins = 8
    
    residarr, linesarr = read_data(settings)
    
    residuals, centers = cut_data(residarr,linesarr)
    
    bincen, binval, binstd = bin_data(residuals,centers,nsegs,nsubins)
    
    seglims, binlims = get_binlims(nsegs,nsubins)
    
    coeffs, fitres = fit_binned(seglims,bincen,binval,binstd,polyord)
    
    gaps, vals = calculate_gaps(coeffs,nsegs)
    
    save_gaps(settings,gaps,bincen,binval,binlims,seglims,polyord,nsegs,nsubins)
    
    plot_all(bincen,binval,binstd,binlims,coeffs,fitres,vals,seglims,centers,residuals)
    
    return
#%% P L O T    R E S I D U A L S   A N D   F I T S

def plot_all(bincen,binval,binstd,binlims,coeffs,fitres,vals,seglims,centers,residuals):

    fig, ax = hf.figure(2,ratios=[3,1],sharex=True)
    binwid   = np.diff(binlims)/2
    ax[0].errorbar(bincen,binval,yerr=binstd,xerr=binwid,
                   c='C1',marker='s',ms=5,ls='')
    for c in coeffs:
        pixl = c['pixl']
        pars = c['pars']
        xx = np.linspace(0,512,128)
        yy = evaluate(pars,x=xx)
        ax[0].plot(xx+pixl,yy,c='C3',lw=2)
        ax[1].plot()
    [ax[0].axvline(512*i,ls='--',lw=0.3) for i in range(9)]
    ax[1].scatter(bincen,np.ravel(fitres),marker='o',c='C0',s=2)
    ax[1].axhline(0,ls='--',lw=0.5)
    ax[0].scatter(seglims[:-1],vals[:,0],marker='>',c='C4')
    ax[0].scatter(seglims[1:],vals[:,1],marker='<',c='C4')
    ax[0].scatter(centers,residuals,s=1,alpha=0.1)
    
#%% E X A M I N E     R E S I D U A L S   B Y   O R D E R
def plot_by_order(residarr,linesarr):
    
    centers0   = linesarr['gauss'][:,1]
    residuals0 = residarr['residual']
        
    orders = np.arange(60,65).astype(int)
    fig2, ax2  = hf.figure(len(orders),sharex=True,alignment='vertical',figsize=(12,12))
    for i,order in enumerate(orders):
        inorder = np.where(residarr['order']==order)[0]
        print("ORDER {0:>5d} {1:>5.3f} k LINES".format(order,len(inorder)/10e3))
        ax2[i].scatter(centers0[inorder],residuals0[inorder],s=2)
        ax2[i].set_ylim(-60,60)
        [ax2[i].axvline(512*j,ls='--',lw=0.3) for j in range(9)]
plot_by_order(residarr,linesarr)