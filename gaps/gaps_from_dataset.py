#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:09:27 2018

@author: dmilakov

USAGE : 
    
    python gaps $path/to/settings/file.json 



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

from harps.core import np, sys, os, time, json, argparse
from harps.core import curve_fit
from harps.core import FITS

import harps.settings as hs
import harps.functions as hf
import harps.containers as container
from harps.wavesol import evaluate
import harps.dataset as hd

#%%
def read_data(filepath,fittype):
    
    
    dataset = hd.Dataset(filepath)
    residarr = np.hstack(dataset['residuals_{}'.format(fittype),500])
#    residlist = []
#    linelist  = []
#    linenum = 0
#    for i,file in enumerate(outlist):
#        #print("Filename = ",file)
#        with FITS(file,'r') as fits:
#            hdu_r = fits['residuals',500]
#            
#            residlist.append(hdu_r.read())
#            hdu_l = fits['linelist']
#            linelist.append(hdu_l.read())
#            linenum+=len(hdu_l.read())
##        print("{1} Cumulative n of lines {0:>5d}".format(linenum,i))
##        wait=input('Press key')
#            
#    residarr = np.hstack(residlist)
#    linesarr  = np.hstack(linelist)
    
    return residarr#, linesarr
#%%
def cut_data(args,residarr):
    
    centers0   = residarr['gauss']
    residuals0 = residarr['residual']
    # order selection 
    if args.order is not None:
        inorder = np.array([idx for idx in range(len(residarr)) \
                            if residarr['order'][idx] in args.order])
    else:
        #all
        inorder = np.arange(len(centers0))
    # use only red chip 
    redchip    = np.where(residarr['order']>44)[0]
    
    # both conditions
    validord   = np.intersect1d(redchip,inorder)
    centers1   = centers0[validord]
    residuals1 = residuals0[validord]
   
    # remove outliers in amplitude
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
def save_gaps(filepath,gaps,bincen,binval,binlims,seglims,polyord,nsegs,nsubins):
    
    basename = os.path.basename(filepath)
    gapsname = "{}_gaps.json".format(os.path.splitext(basename)[0])
    gapspath = os.path.join(hs.dirnames['gaps'],gapsname)
    
    data = {"created_on":time.strftime("%Y-%m-%dT%H_%M_%S"),
            "dataset":filepath,
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

#save_gaps(settings,gaps,bincen,binval,binlims,seglims,polyord,nsegs,nsubins)
#%%
def main(args):
#    filepath = '/Users/dmilakov/harps/dataprod/settings/series1_fibreA.json'
    print(args)
    filepath = args.file
    polyord = args.polyord
    fittype = args.fittype
    nsegs = 8
    nsubins = args.bins
    

    residarr = read_data(filepath,fittype)
    print("{0:>20s} = {1:8.3f} k".format("TOTAL N OF LINES",
                                         np.size(residarr)/1e3))
    residuals, centers = cut_data(args,residarr)
    
    bincen, binval, binstd = bin_data(residuals,centers,nsegs,nsubins)
    
    seglims, binlims = get_binlims(nsegs,nsubins)
    
    coeffs, fitres = fit_binned(seglims,bincen,binval,binstd,polyord)
    
    gaps, vals = calculate_gaps(coeffs,nsegs)
    
    if not args.dry:
        save_gaps(filepath,gaps,bincen,binval,binlims,
                  seglims,polyord,nsegs,nsubins)
    if args.plot:
        plot_all(args,bincen,binval,binstd,binlims,coeffs,fitres,gaps,
                 vals,seglims,centers,residuals)
    
    return
#%% P L O T    R E S I D U A L S   A N D   F I T S

def plot_all(args,bincen,binval,binstd,binlims,coeffs,
             fitres,gaps,vals,seglims,centers,residuals):
    print("{0:>20s} = {1:8.3f} k".format("LINES USED",np.size(centers)/1e3))
    
    filepath = args.file
    fittype  = args.fittype
    basename = os.path.basename(filepath)
    basenoext= os.path.splitext(basename)[0]
    
    if not args.nofit:
        fig, ax = hf.figure(4,ratios=[[3,1],[3,1]],alignment='grid',
                            sharex=[True,True,False,False],
                            sharey=[True,False,True,False],
                            left=0.12)
        binwid   = np.diff(binlims)/2
        fitres   = np.ravel(fitres)
        ax[0].errorbar(bincen,binval,yerr=binstd,xerr=binwid,
                       c='C1',marker='s',ms=5,ls='',rasterized=True)
        for c in coeffs:
            pixl = c['pixl']
            pars = c['pars']
            xx = np.linspace(0,512,128)
            yy = evaluate(pars,x=xx)
            ax[0].plot(xx+pixl,yy,c='C2',lw=2,zorder=100,rasterized=True)
            ax[1].plot()
    
        ax[1].scatter(bincen,fitres,marker='o',c='C0',s=2,rasterized=True)
        ax[0].scatter(seglims[:-1],vals[:,0],marker='>',c='C2',rasterized=True)
        ax[0].scatter(seglims[1:],vals[:,1],marker='<',c='C2',rasterized=True)
        ax[1].axhline(0,ls='--',lw=0.5)
        
        ax[1].set_ylim(*hf.negpos(1.2*np.percentile(fitres,95)))
        ax[1].set_ylabel("Residuals to \n the fit [m/s]")
        
        ax[2].scatter(seglims[1:-1],gaps*829,marker='s',s=5,rasterized=True)
        [ax[2].axvline(512*i,ls='--',lw=0.3) for i in range(9)]
        ax[3].hist(fitres,bins=5)
    else:
        fig, ax = hf.figure(1,left=0.15)
    [ax[0].axvline(512*i,ls='--',lw=0.3) for i in range(9)]
    # LABELS AND TITLE
    ax[0].set_title("Gap size")
    ax[0].set_ylim(-60,60)
    ax[0].set_ylabel("Residuals to \n wavelength dispersion [m/s]")
    ax[0].set_xlim(-100,4196)
    ax[0].set_xlabel("Pixel")
    
    if args.save_plot:
        # thin out the points to plot to file
        figdir  = os.path.join(hs.dirnames['plots'],'gaps')
        polyord = args.polyord
        nsubins = args.bins
        figname = "{0}_poly={1}_bins={2}_ft={3}.pdf".format(basenoext,polyord,
                   nsubins,fittype)
        figpath = os.path.join(figdir,figname)
        print(figpath)
        ax[0].scatter(centers[::10],residuals[::10],s=1,c='C0',alpha=0.1,
          rasterized=True)
        
        fig.savefig(figpath,rasterized=True)
        print("Figure saved to : {}".format(figpath))
    else:
        ax[0].scatter(centers,residuals,s=1,c='C0',alpha=0.1)
    return
    
#%% E X A M I N E     R E S I D U A L S   B Y   O R D E R
def plot_by_order(residarr,linesarr):
    
    centers0   = linesarr['gauss'][:,1]
    residuals0 = residarr['residual']
        
    orders = np.arange(50,54).astype(int)
    fig2, ax2  = hf.figure(len(orders),sharex=True,alignment='vertical',figsize=(12,12))
    for i,order in enumerate(orders):
        inorder = np.where(residarr['order']==order)[0]
        print("ORDER {0:>5d} {1:>5.3f} k LINES".format(order,len(inorder)/1e3))
        ax2[i].scatter(centers0[inorder],residuals0[inorder],s=2)
        ax2[i].set_ylim(-60,60)
        [ax2[i].axvline(512*j,ls='--',lw=0.3) for j in range(9)]
#plot_by_order(residarr,linesarr)
        
#%% M A I N    P A R T
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate gaps.')
    parser.add_argument('file',type=str, 
                        help='Path to the settings file')
    parser.add_argument('-ft','--fittype',type=str,default='gauss',
                        help="Fittype, default gauss.")
    parser.add_argument('-p','--plot', action='store_true', default=False,
                        help="Plot the gaps model.")
    parser.add_argument('-sp','--save-plot', action='store_true', default=False,
                        help="Save the plot.")
    parser.add_argument('-dry-run','--dry',action='store_true',default=False,
                        help="Do not save output to file.")
    parser.add_argument('-nf','--nofit',action = 'store_true',default=False,
                        help="Do not plot the fit")
    parser.add_argument('-poly','--polyord',type=int,default=3,
                        help="Polynomial order, default 3.")
    parser.add_argument('-b','--bins',type=int,default=8,
                    help="Number of bins in each 512px segment, default 8.")
    parser.add_argument('-o','--order',type=int,nargs='+', default=None,
                        help="Orders to use, default all.")
    args = parser.parse_args()
    main(args)