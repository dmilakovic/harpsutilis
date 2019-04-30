#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:09:27 2018

@author: dmilakov

USAGE : 
    
    python gaps $path/to/settings/file.json 



The settings file must have the 'outlist' parameter.

Calculate gaps from 'out' FITS files.

(1) Read the residuals to the LFC wavelength calibration, version 800
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
import harps.fit as hfit
import harps.containers as container
from   harps.wavesol import evaluate
import harps.dataset as hd
import harps.plotter as plot
nsegs  = 8
#%%
def read_data(filepath,fittype,version):
    
    
    dataset = hd.Dataset(filepath)
    residarr = np.hstack(dataset['residuals_{}'.format(fittype),version])
    
    return residarr
#%%
blocklims = [[61,72],[46,61],[27,45],[0,26]]
#blocklims = [[89,99],[100,114],[116,134],[135,161]]
def cut_data(residarr,block,dist):
    
    centers0   = residarr['gauss']
    residuals0 = residarr['residual_mps']
    
    # use only lines on a given chip block
    low,high = blocklims[block-1]
    cut0     = np.where((residarr['order']>=low)&(residarr['order']<=high))[0]

    centers1   = centers0[cut0]
    residuals1 = residuals0[cut0]
   
    # remove outliers in amplitude
    limit      = 200
    cut1        = np.where(np.abs(residuals1)<limit)
    residuals2  = residuals1[cut1]
    centers2    = centers1[cut1]
   
    # remove points closer than distance 'dist' from the segment edges
    seglims = np.linspace(0,4096,nsegs+1)
    cond    = [(centers2>=(seglims[i+1]-dist))&(centers2<=(seglims[i+1]+dist)) \
                                for i in range(nsegs-1)]
    cut2     = np.logical_or.reduce(cond)
    residuals = residuals2[~cut2]
    centers   = centers2[~cut2]
    
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
def fit_poly(residuals,centers,polyord):
    seglims = np.linspace(0,4096,nsegs+1)
    coeffs   = container.coeffs(polyord,nsegs)
    fitresids = []
    for i in range(nsegs):
        pixl  = seglims[i]
        pixr  = seglims[i+1]
        inseg = np.where((centers>pixl)&(centers<pixr))[0]
        #binsh = bincen[inseg]-pixl
        res   = curve_fit(hf.polynomial,centers[inseg],residuals[inseg],
                          p0 = np.ones(polyord+1))
        pars  = res[0]
        cov   = res[1]
        errs  = [np.sqrt(cov[i][i]) for i in range(polyord+1)]
        resd  = residuals[inseg] - np.polyval(pars[::-1],centers[inseg])
        dof   = len(inseg) - len(pars)
        #chi2  = np.sum(resd**2 / binstd[inseg]**2) / dof
        fitresids.append(resd)
        coeffs['segm'][i] = i
        coeffs['pixl'][i] = pixl
        coeffs['pixr'][i] = pixr
        coeffs['pars'][i] = pars
        coeffs['errs'][i] = errs
        #coeffs['chi2'][i] = chi2
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

#%% 

def calculate_gaps_spline(residuals,centers,nknots):
    nsegs   = 8
    seglims = np.linspace(0,4096,nsegs+1)
    fig     = plot.Figure(1)
    models  = []
    for i in range(nsegs):
        pixl  = seglims[i]
        pixr  = seglims[i+1]
        inseg = np.where((centers>pixl)&(centers<pixr))[0]
        splin = hfit.spline(centers[inseg],residuals[inseg],n_knots=nknots)
        model = splin.predict(np.array([pixl,pixr]))
        #errs  = [np.sqrt(cov[i][i]) for i in range(polyord+1)]
        resd  = residuals[inseg] - splin.predict(centers[inseg])
        #dof   = len(inseg) - len(pars)
        #chi2  = np.sum(resd**2 / binstd[inseg]**2) / dof
        fig.axes[0].scatter(centers[inseg],residuals[inseg],s=2,marker='o')  
        minval,maxval = np.min(centers[inseg]),np.max(centers[inseg])
        X = np.linspace(pixl,pixr,100)
        fig.axes[0].plot(X,splin.predict(X),c='k')  
        [fig.axes[0].axvline(x,ls=':',c='k',lw=1) for x in [minval,maxval]]
        models.append(splin)
    
    vals = np.zeros((nsegs-1,2))
    for i in range(nsegs-1):
        pix     = seglims[i+1]
        model_l = models[i]
        model_r = models[i+1]
        vals[i] = [model_l.predict(pix),model_r.predict(pix)]
    leftvals,rightvals = np.transpose(vals)
    gaps = (rightvals-leftvals)/829.
    return gaps, vals
#%%  S A V E   C O E F F I C I E N T S    T O   F I L E
def save_gaps(filepath,block,gaps,vals,bincen,binval,binlims,seglims,
              polyord,nsegs,nsubins):
    
    basename = os.path.basename(filepath)
    gapsname = "{}_{}_gaps.json".format(os.path.splitext(basename)[0],block)
    gapspath = os.path.join(hs.dirnames['gaps'],gapsname)
    
    data = {"created_on":time.strftime("%Y-%m-%dT%H_%M_%S"),
            "dataset":filepath,
            "bin_limits":list(binlims),
            "bin_centers":list(bincen),
            "bin_values":list(binval),
            "seg_limits":list(seglims),
            "polyord":polyord,
            "gaps_pix":list(gaps),
            "gaps_mps":list(gaps*829.),
            "gaps_left":list(vals[:,0]/829),
            "gaps_right":list(vals[:,1]/829)
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
def print_message(args,gaps):
    
    print("{0:<20} = {1:<20}".format("Version",args.version))
    print("{0:<20} = {1:<20}".format("Block",args.block))
    print("{0:<20} = {1:<20}".format("Exclusion distance",args.dist))
    print("{0:<20} = {1:<20}".format("No. of knees",args.knees))
    print("{0:=>81}".format(""))
    print("{0:<10}".format("Segm"),(7*"{:10d}").format(*np.arange(1,8,1)))
    print("{0:<10}".format("mpix"),(7*"{:10.3f}").format(*gaps*1e3))
    print("{0:<10}".format("m/s"), (7*"{:10.3f}").format(*gaps))
    return
def main(args):
#    filepath = '/Users/dmilakov/harps/dataprod/settings/series1_fibreA.json'
    print(args)
    filepath = args.file
    polyord = args.polyord
    fittype = args.fittype
    nknees  = args.knees
    block   = args.block
    version = args.version
    dist    = args.dist
    residarr = read_data(filepath,fittype,version)
    total    = np.size(residarr)
    print("{0:<20s} = {1:<20.3f}k".format("TOTAL N OF LINES",total/1e3))
    residuals, centers = cut_data(residarr,block,dist)
    used     = np.size(centers)
    
    print("{0:<20s} = {1:<20.3f}k ({2:5.1%})".format("LINES USED",
                                                 used/1e3,
                                                 used/total))
    #bincen, binval, binstd = bin_data(residuals,centers,nsegs,nsubins)
    
    #seglims, binlims = get_binlims(nsegs,nsubins)

    #coeffs, fitres = fit_binned(seglims,bincen,binval,binstd,polyord)
    
    #gaps, vals = calculate_gaps(coeffs,nsegs)
    
    gaps, vals = calculate_gaps_spline(residuals, centers, nknees)
    
    print_message(args,gaps)
    if not args.dry:
        save_gaps(filepath,block,gaps,vals,bincen,binval,binlims,
                  seglims,polyord,nsegs,nsubins)
    if args.plot:
        plot_all(args,bincen,binval,binstd,binlims,coeffs,fitres,gaps,
                 vals,seglims,centers,residuals)
    
    return
#%% P L O T    R E S I D U A L S   A N D   F I T S

def plot_all(args,bincen,binval,binstd,binlims,coeffs,
             fitres,gaps,vals,seglims,centers,residuals,unit='mps'):
    
    
    filepath = args.file
    fittype  = args.fittype
    basename = os.path.basename(filepath)
    basenoext= os.path.splitext(basename)[0]
    stat = False
    
    if unit !='mps':
        binval = binval / 829
        binstd = binstd / 829
        residuals = residuals / 829
    
    if not args.nofit and stat:
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
            if unit != 'mps':
                yy = yy/829
            ax[0].plot(xx+pixl,yy,c='C2',lw=2,zorder=100,rasterized=True)
            ax[1].plot()
    
        ax[1].scatter(bincen,fitres,marker='o',c='C0',s=2,rasterized=True)
        ax[0].scatter(seglims[:-1],vals[:,0],marker='>',c='C2',rasterized=True)
        ax[0].scatter(seglims[1:],vals[:,1],marker='<',c='C2',rasterized=True)
        ax[1].axhline(0,ls='--',lw=0.5)
        
        ax[1].set_ylim(*hf.negpos(1.2*np.percentile(fitres,95)))
        ax[1].set_ylabel("Residuals [m/s]")
        
        ax[2].scatter(seglims[1:-1],gaps*829,marker='s',s=5,rasterized=True)
        [ax[2].axvline(512*i,ls='--',lw=0.3) for i in range(9)]
        ax[3].hist(fitres,bins=5)
    elif not args.nofit and not stat:
        plotter = plot.Figure(1,left=0.12,top=0.95,enforce_size=True)
        fig, ax = plotter.fig, plotter.axes
        binwid   = np.diff(binlims)/2
        fitres   = np.ravel(fitres)
        ax[0].errorbar(bincen,binval,yerr=binstd,xerr=binwid,
                       c='C1',marker='s',ms=5,ls='',rasterized=True)
        for c in coeffs:
            pixl = c['pixl']
            pars = c['pars']
            xx = np.linspace(0,512,128)
            yy = evaluate(pars,x=xx)
            if unit != 'mps':
                yy = yy/829
            ax[0].plot(xx+pixl,yy,c='C1',lw=2,zorder=100,rasterized=True)
    
        #ax[1].scatter(bincen,fitres,marker='o',c='C0',s=2,rasterized=True)
        #ax[0].scatter(seglims[:-1],vals[:,0],marker='>',c='C2',rasterized=True)
        #ax[0].scatter(seglims[1:],vals[:,1],marker='<',c='C2',rasterized=True)
        #ax[1].axhline(0,ls='--',lw=0.5)
        
        #ax[1].set_ylim(*hf.negpos(1.2*np.percentile(fitres,95)))
        #ax[1].set_ylabel("Residuals to \n the fit [m/s]")
        
        #ax[2].scatter(seglims[1:-1],gaps*829,marker='s',s=5,rasterized=True)
        #[ax[2].axvline(512*i,ls='--',lw=0.3) for i in range(9)]
        #ax[3].hist(fitres,bins=5)
    else:
        fig, ax = hf.figure(1,left=0.15)
    [ax[0].axvline(512*i,ls='--',lw=0.3) for i in range(9)]
    # LABELS AND TITLE
    #ax[0].set_title("Gaps, block {}".format(args.block))
    ylim = (-45,45)
    if unit!='mps':
        ylim = (-0.07,0.07)
    ax[0].set_ylim(*ylim)
    ax[0].set_ylabel("Residuals [m/s]")
    ax[0].set_xlim(-200,4296)
    ax[0].set_xlabel("Pixel")
    plotter.ticks(0,'x',5,0,4096)
    plotter.ticks(0,'y',5,-40,40)
    if args.save_plot:
        # thin out the points to plot to file
        figdir  = os.path.join(*[hs.dirnames['plots'],'gaps','v_1.0.1'])
        polyord = args.polyord
        nsubins = args.bins
        block   = args.block
        version = args.version
        figname = "{0}_block={1}_poly={2}".format(basenoext,block,polyord) + \
                  "_bins={0}_ft={1}_ver={2}.pdf".format(nsubins,fittype,version)
        figpath = os.path.join(figdir,figname)
        print(figpath)
        ax[0].scatter(centers,residuals,s=1,c='C0',alpha=0.1,
          rasterized=True)
        
        fig.savefig(figpath,rasterized=True)
        print("Figure saved to : {}".format(figpath))
    else:
        ax[0].scatter(centers,residuals,s=1,c='C0',alpha=0.1,rasterized=True)
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
        ax2[i].set_ylim(-40,40)
        [ax2[i].axvline(512*j,ls='--',lw=0.3) for j in range(9)]
#plot_by_order(residarr,linesarr)
        
#%% M A I N    P A R T
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate gaps.')
    parser.add_argument('file',type=str, 
                        help='Path to the settings file')
    parser.add_argument('block',type=int,
                        help='CCD block')
    parser.add_argument('-v','--version',type=int,default=800,
                        help='Version')
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
    parser.add_argument('-k','--knees',type=int,default=8,
                    help="Number of knees in each 512px segment, default 8.")
    parser.add_argument('-d','--dist',type=int,default=10,
                    help="Pixel distance to be excluded, default 10.")
    parser.add_argument('-o','--order',type=int,nargs='+', default=None,
                        help="Orders to use, default all.")
    args = parser.parse_args()
    main(args)