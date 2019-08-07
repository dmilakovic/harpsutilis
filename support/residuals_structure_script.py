#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:15:39 2019

@author: dmilakov

grouped mean of residuals
"""

import harps.functions as hf
import harps.plotter as plot
import harps.dataset as hd
import numpy as np
import harps.settings as hs
import os
import matplotlib.pyplot as plt
import matplotlib
import argparse
import scipy

from fitsio import FITS

matplotlib.use('TKAgg')
matplotlib.style.use('paper')
#%% GROUP 
blocklims = [[61,72],[46,61],[27,46],[0,27]]
optords   = [161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149,
       148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136,
       135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123,
       122, 121, 120, 119, 118, 117, 116, 114, 113, 112, 111, 110, 109,
       108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98,  97,  96,
        95,  94,  93,  92,  91,  90,  89]
labels = {701:r"Segmented $7^{\rm th}$ order",
              1810:r"Global $18^{\rm th}$ order"}
linestyles = {701:'-',1810:'-'}
markers    = {701:'s',1810:'o'}
colors     = {701:'C0',1810:'C1'}
    
def cut_data(residarr,block,fittype,pex=None):
    '''Returns residuals within one detector block.
    pex controls the how many pixels around segment boundary are ignored'''
    centers0   = residarr[fittype]
    residuals0 = residarr['residual_mps']
    noise0     = residarr['noise']
    
    # use only valid orders 
    low,high = blocklims[block-1]
    sel      = np.where((residarr['order']>=low)&(residarr['order']<high))[0]

    centers1   = centers0[sel]
    residuals1 = residuals0[sel]
    noise1     = noise0[sel]
    # remove outliers in amplitude
    limit      = args.res_exclude
    cut        = np.where(np.abs(residuals1)<limit)
    residuals  = residuals1[cut]
    centers    = centers1[cut]
    noise      = noise1[cut]
    
    if pex is not None and pex !=0:
        boundary = np.linspace(0,4096,9)
        for i,b in enumerate(boundary):
            centers = np.ma.masked_inside(centers,b-pex,b+pex)
            mask    = centers.mask
            residuals = np.ma.masked_array(residuals,mask=mask).compressed()
            noise     = np.ma.masked_array(noise,mask=mask).compressed()
            centers   = centers.compressed()
    return residuals,centers,noise
#%%
def plot_residuals_binned(args,arrays):
    npix=4096
    bins=8*args.bins
    fac =npix//bins
    
    # which detector block
    blocks = [1,2,3]
    # exclude pixels around segment boundaries
    pex   = args.pix_exclude
    
    # bins
    binlims = np.linspace(0+fac,npix,bins) 
    bincens = np.linspace(0+fac/2,npix-fac/2,bins)
    
    # containers
    means   = {}#np.zeros((len(versions),bins))
    errors  = {}#np.zeros((len(versions),bins))
    
    j = 0
    percentiles = [32,68]
    plotter = plot.Figure2(3,1,figsize=(8,9),enforce_figsize=True,top=0.93,bottom=0.06)
    ax0     = plotter.add_subplot(0,1,0,1)
    ax1     = plotter.add_subplot(1,2,0,1,sharey=ax0)
    ax2     = plotter.add_subplot(2,3,0,1,sharey=ax0)
    
    draw_points = args.draw_points
    draw_contours=False
    
    fittype  = args.fittype
    versions = list(arrays.keys())
    for j,block in enumerate(blocks):
        ax = plotter.axes[j]
        for k,ver in enumerate(versions):
            data = arrays[ver]
            resids,centers,noise=cut_data(data, block, fittype, pex)
            # bin, calculate mean and central 68% of residuals 
            means1d,edges,binned = scipy.stats.binned_statistic(centers,resids,
                                                   'mean',bins=bins,range=(0,4096))
            sigma1d,edges,binned = scipy.stats.binned_statistic(centers,resids,
                                                   'std',bins=bins,range=(0,4096))
            bincens = 0.5*(edges[1:]+edges[:-1])
    #        binned  = np.digitize(centers,binlims)
    #        means1d = np.zeros(bins)
    #        sigma1d = np.zeros((bins,len(percentiles)))
    #        for i,b in enumerate(binlims):
    #            cut = np.where(binned==i)[0]
    #            #means[j,i] = np.mean(data['residual'][cut])
    #            #errors[j,i]= np.std(data['residual'][cut])
    #            means1d[i] = np.mean((resids)[cut])
    #            sigma1d[i] = np.percentile((resids)[cut],percentiles) 
            means[ver]=means1d
            errors[ver]=sigma1d
       
            if draw_points:
                plotter.axes[j].scatter(centers[::2],resids[::2],s=1,
                            rasterized=True,alpha=0.1)
                pass
            elif draw_contours:
                H,xedges,yedges = np.histogram2d(centers,resids,bins=bins,range=[[0,4096],[-20,20]],
                                                 normed=True)
                xcen=(xedges[1:]+xedges[:-1])/2
                ycen=(yedges[1:]+yedges[:-1])/2
                X,Y=np.meshgrid(xcen,ycen)
                contours=ax0.contour(X,Y,H.T,np.percentile(H,[68,95]),
                                    linestyles=linestyles[ver],
                                    colors=colors[ver],linewidths=[2])
            else:
                pass
            ax.plot(bincens,means[ver],ls=linestyles[ver],
                        marker=markers[ver],
                     label=labels[ver])
            print('Block {0:1d}, version {1:4d}:\t'.format(block,ver) + \
                  'Mean = {0:8.3f}, RMS = {1:8.3f}, N={2:8d}'.format(np.mean(means1d),hf.rms(means1d),len(resids)))
    
            # plot a the mean error to the left of the main plot as a reference
            ax.errorbar(-80*(k+1),0,yerr=np.mean(sigma1d),
                        c=colors[ver],marker='',capsize=5,elinewidth=3,capthick=3)
        ax.axhline(0,ls='--',c='k',lw=1)
        [ax.axvline((i)*512,ls=':',lw=0.7,c='k') for i in range(9)]
        if draw_points:
            ax.set_ylim(-25,25)
            plotter.ticks(j,'y',5,-20,20)
        else:
            if fittype=='gauss':
                ax.set_ylim(-5,5)
                plotter.ticks(j,'y',5,-4,4)
            else:
                ax.set_ylim(-10,10)
                plotter.ticks(j,'y',5,-8,8)
        plotter.ticks(j,'x',5,0,4096)
        
        if j<2:
            ax.set_xticklabels([])
        if j==1:
            ax.set_ylabel('Residuals [m/s]')
        
    plotter.axes[0].legend(loc='upper center',ncol=2,bbox_to_anchor=(0.5, 1.28))
    plotter.axes[-1].set_xlabel('Pixel')
    #plotter.figure.text(0.04,0.5,'Residuals [m/s]',horizontalalignment='center',
    #                    verticalalignment='center',rotation=90)
    
    
    if args.save_plot:
        fibre = args.fibre
        fittype = args.fittype
        figdir  = os.path.join(*[hs.get_dirname('plots'),
                                 'wavesolution',
                                 'residuals'])
        figname = 'corr_residuals_mean_fibre={0}'.format(fibre) +\
                  '_points={0}_'.format(draw_points) +\
                  'pex={0}_ft={1}.pdf'.format(pex,fittype)
                  
        plotter.save(os.path.join(figdir,figname),rasterized=True)
        print("Plot saved to: ",os.path.join(figdir,figname))
        
#%%
from scipy.optimize import curve_fit
def lorentz(x,*p):
    I,gamma, x0 = p
    return I/(np.pi*gamma)*1/(1+((x-x0)/gamma)**2)
def moffat(x,*p):
    I, x0, gamma, beta = p
    return I/(1 + ((x-x0)/gamma)**2)**beta

def plot_histogram(args,arrays):
    plot_profile=False
    plot_arrows =False
    
    plotter = plot.Figure(1,figsize=(8,6),enforce_figsize=True)
    axes    = plotter.axes
    ax0     = axes[0]
    plotver = [701,1810]
    xrange  = (-20,20)
    bins    = 51#2*xrange[1]
    lslist  = ['-',':']
    q          = [0.025,0.16,0.5,0.84,0.975]
    zorders = {701:0,1810:-10}
    print(("{:<10s}"+len(q)*("{:10.3f}")).format("version",*q))
    for j,ver in enumerate(plotver):
        
        data      = arrays[ver]['residual_mps']#/arrays[ver]['noise']
        cut       = np.where(np.abs(data)<xrange[1])[0]
        data      = data[cut]
        mean      = np.mean(data)
        sigma     = np.std(data)
        quantiles = np.quantile(data,q)
        print(("{:<10d}"+len(q)*("{:10.3f}")).format(ver,*quantiles))
    #    [ax0.axvline(qq,ls=':',lw=1,c=colors[ver]) for qq in quantiles]
        
        vals,lims,obj = ax0.hist(data,bins=bins,range=xrange,histtype='step',
                                 lw=3,color=colors[ver],zorder=zorders[ver],
                                 label=labels[ver],density=False,ls=linestyles[ver])
        cens = (lims[1:]+lims[:-1])/2
        # fill in the central 50 percent of the histogram
    #    ax0.fill_between(cens, 0, vals, interpolate=False,color=colors[ver],
    #                     alpha = 0.3,
    #                where=((cens>=np.percentile(data, 25)) &
    #                       (cens<=np.percentile(data, 75))))
        
        
        # fit lorentzian
        pars,covar = curve_fit(moffat,cens,vals,(np.max(vals),0,2,1))
    #    print(pars)
    #    print(covar)
        I, x0, gamma, beta = pars
        fac = (2*np.sqrt((2**(1/beta))-1))
        fwhm = gamma*fac
        # draw a line marking the width of the central 50 percent
        ax0.errorbar(quantiles[2],1e4*(2-j),capsize=8,capthick=3,
                     xerr=np.array([quantiles[3]-quantiles[1]])/2,yerr=0)
        
        errs = np.sqrt(np.diag(covar))
        I_err,x0_err,gamma_err,beta_err = errs
        fwhm_err = np.sqrt((gamma_err*fac)**2 )
        
        #print((2*("{:7.2f}+-{:7.2f}\t")).format(x0,x0_err,fwhm,fwhm_err))
       
        if plot_profile:
             x = np.linspace(*xrange,61)
             y = moffat(x,*pars)
             ax0.plot(x,y,color=colors[ver])
        arrlength = 5
        offset = 1
        #right side, pointing left
        if plot_arrows:
            for j in [True,False]:
                ax0.arrow(x0+(-1)**j*(fwhm/2+arrlength+offset),
                          I/2,
                          -(-1)**j*(arrlength),
                          0,
                          shape='full',
                          color=colors[ver],
                          width=1e3,head_width=3e3,head_length=1,
                          length_includes_head=True,
                          head_starts_at_zero=False)
                ax0.text(0,I/2,"{0:5.2f}".format(fwhm/2),
                         horizontalalignment='center',
                         verticalalignment='center')
        # left side, pointing right
    #    ax0.arrow(x0-fwhm/2-arrlength,I/2,arrlength,0,shape='full',color=colors[ver],
        #          width=0.0005,head_length=1,head_width=0.003,
        #         length_includes_head=True)
    ax0.legend()
    ax0.set_xlabel('Residuals [m/s]')
    ax0.set_ylabel('Number of lines')
    ax0.set_xlim(-20,25)
    plotter.ticks(0,'y',5,0,2e5)
    #ax0.set_ylim(0,2.7e5)
    
    if args.save_plot:
        fibre   = args.fibre
        fittype = args.fittype
        figdir  = os.path.join(*[hs.get_dirname('plots'),'wavesolution','residuals'])
        figname = 'hist_residuals_{0}_{1}.pdf'.format(fibre,fittype)
        plotter.save(os.path.join(figdir,figname))

#%% Cumulative distribution of residuals as a proxy of model accuracy
def plot_cumulative(args,arrays):
    versions   = list(arrays.keys())
    plotter    = plot.Figure2(1,1)
    ax0        = plotter.add_subplot(0,1,0,1)
    xrange     = (0,20)
    cumulative = True
    histtype   = 'step'
    density    = True
    bins       = 100
    q          = [0.68,0.95]
    colors     = {701:'C0',1810:'C1'}
    for ver in versions:
        data      = np.abs(arrays[ver]['residual_mps'])
        quantiles = np.quantile(data,q)
        ax0.hist(data, range=xrange, cumulative=cumulative, histtype=histtype,
                 bins=bins, density=density,color=colors[ver])
        [ax0.plot((quantiles[i],quantiles[i]),(0,q[i]),c=colors[ver]) for i in range(len(q))]
        
        print("Version {0:4d} : {1:5.3f} ({2:5.3f}) within {3:3.0%} ({4:3.0%})".format(ver,*quantiles,*q))
    if args.save_plot:
        fibre   = args.fibre
        fittype = args.fittype
        figdir  = os.path.join(*[hs.get_dirname('plots'),'wavesolution','residuals'])
        figname = 'cumul_residuals_{0}_{1}.pdf'.format(fibre,fittype)
        plotter.save(os.path.join(figdir,figname))
#%%   
def main(args):
    print(args)
    dset = hd.Dataset(args.filepath)
    fittype = args.fittype
    
    versions = args.version
    
    arrays = {}

    for ver in versions:
        data = np.hstack(dset['residuals_{}'.format(fittype),ver])
        arrays[ver]=data
    
    if 'str' in args.plot or 'structure' in args.plot:
        plot_residuals_binned(args,arrays)
    if 'hist' in args.plot or 'histogram' in args.plot:
        plot_histogram(args,arrays)
    if 'cum' in args.plot or 'cumulative' in args.plot:
        plot_cumulative(args,arrays)
    
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath',type=str,help='Path to the outfile')
    parser.add_argument('fibre',type=str,help='Fibre')
    parser.add_argument('fittype',type=str,help='Fittype for lines')
    parser.add_argument('-v','--version',type=int,nargs='*',default=[701,1810],
                        help='Version of wavelength calibration')
    parser.add_argument('-p','--plot',type=str,nargs='*',
                        default=['str','hist','cum'])
    parser.add_argument('-sp','--save-plot',action='store_true',default=False)
    parser.add_argument('--draw-points',action='store_true',default=False)
    parser.add_argument('-pex','--pix-exclude',type=int,default=32)
    parser.add_argument('-rex','--res-exclude',type=int,default=32)
    parser.add_argument('-b','--bins',type=int,default=8)
    
    args = parser.parse_args()
    main(args)