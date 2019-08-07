#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constructing a CTI model using the RV data from February 2012 data.

Reads in the dataset with RV data and fits an exponential function to the data
(separately for the two fibres)

Results of the fit is saved in variable 'pars'.


On 17 Sep 2018, the best fit to model y = a + b * exp(-x/c) on unbinned data 
gives:

pars:
    {'A': array([-1.79049770e+00,  3.48708998e+00,  5.68084348e+05]),
     'B': array([-3.17763319e+00,  3.91926919e+00,  1.67954405e+06])}
errors:
    {'A': array([1.57374670e-01, 1.28352721e-01, 5.84621606e+04]),
     'B': array([1.08525790e+00, 1.05585390e+00, 6.24570521e+05])}
covar:
    {'A': array([[ 2.47667869e-02, -1.86251487e-02, -8.75678573e+03],
                 [-1.86251487e-02,  1.64744209e-02,  5.91874113e+03],
                 [-8.75678573e+03,  5.91874113e+03,  3.41782422e+09]]),
     'B': array([[ 1.17778470e+00, -1.14549547e+00, -6.75730721e+05],
                 [-1.14549547e+00,  1.11482745e+00,  6.56211741e+05],
                 [-6.75730721e+05,  6.56211741e+05,  3.90088336e+11]])}

Modified from a file created on Tue Sep 11 10:40:34 2018
Modified on Wed Jun 11 2019
@author: dmilakov
"""

import harps.classes   as hc
import harps.functions as hf
import harps.settings  as hs
import harps.dataset   as hd
import harps.plotter   as hplot

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
import json

from scipy.optimize import curve_fit
from scipy.stats    import binned_statistic
#matplotlib.use('TKAgg')
matplotlib.style.use('paper')
#%%
def main(args):
    outfile  = args.file
    fibre    = args.fibre
    method   = args.method
    fittype  = args.fittype
    version  = args.version
    sigma    = args.sigma
    models   = np.atleast_1d(args.model)
    binsize  = args.binsize
    vlim     = args.vlim
    
    series   = hd.Series(outfile,fibre)
    values   = series.__call__('{}_{}'.format(method,fittype),version,
                               vlim=vlim)
    shift    = hd.SeriesVelocityShift(values)
    shift    = shift.correct_time()
    shift.plot(3)
    
    flux, shift, error, minx, maxx = select_data(values,binsize,sigma,True)
    npts = np.size(flux)
    if args.plot:
        plotter = hplot.Figure2(1,1,left=0.12)
        figure  = plotter.figure
        ax0     = plotter.add_subplot(0,1,0,1)
        ax0.errorbar(flux,shift,yerr=error,xerr=[minx,maxx],
                     ls='',marker='o',capsize=3,
                         label='Data')
        ax0.set_xlabel("Average flux per line [counts]")
        ax0.set_ylabel("Velocity shift [m/s]")
        if args.title==True:
            ax0.set_title("Method = {0:10s} version = {1:>4d}".format(method,version))
        ax0.ticklabel_format(axis='x',style='sci',scilimits=(-2,3))
    for model in models:
        if model=='log':
            cti_model = log_model
            p0        = (1,1)
            label     = r"$v(f)=a+b\,\log{f}$"
        elif model == 'exp':
            cti_model = exp_model
            p0        = (2,1e5)
            label     = r"$v(f)=a\,\exp{(-f/b)}$"
        kwargs = dict(p0=p0,absolute_sigma=False)
        if args.binsize>0:
            kwargs['sigma'] = error
        pars, covar = curve_fit(cti_model,flux,shift,**kwargs)
        
        n, p        = len(flux),len(pars)
        predicted   = cti_model(flux,*pars)
        normresid   = (shift-predicted)/error
        chisq       = np.nansum(normresid[np.isfinite(normresid)]**2)
        chisqnu     = chisq/(n-p)
        aicc        = chisq + 2*p + 2*p*(p+1)/(n-p-1)
        print_message(args,model,npts,pars,covar,chisq,chisqnu,aicc)
        if args.plot:
            if args.binsize>0:
                flxnb = values['flux']
                shiftnb, errnb = np.transpose(values['{}sigma'.format(sigma)])
#                flxnb,shiftnb,errnb = select_data(values,False,sigma)
                ax0.scatter(flxnb,shiftnb,marker='o',c='k',s=2)
            
            if not args.nofit:
                x = np.linspace(np.min(flux),np.max(flux),200)
                y = cti_model(x,*pars)
                ax0.plot(x,y,ls='-',label=label)
            ax0.axhline(0,ls=':',lw=1.,c='k')
            ax0.legend()
            if args.save_plot:
                figdir  = os.path.join(*[hs.dirnames['plots'],hs.version,'cti'])
                basename = os.path.basename(outfile)
                basenoext = os.path.splitext(basename)[0]
                figname = "{0}_cti_model={1}_ft={2}".format(basenoext,models,fittype,) + \
                      "_met={0}_sigma={1}".format(method,sigma) + \
                      "_ver={0}_fb={1}.pdf".format(version,fibre)
                figpath = os.path.join(figdir,figname)
                figure.savefig(figpath,rasterized=True)
                print("Figure saved to {}".format(figpath))
        if args.save:
            save(args,model,pars,covar,chisq,chisqnu,aicc)
    if args.plot:
        plt.show()
    
        
def select_data(values,binsize,sigma=None,return_xrange=False):
    if binsize>0:
        sigma  = sigma if sigma is not None else 3
        x      = values['flux']
        y,yerr = values['{}sigma'.format(sigma)].T
        # create bin edges 
        # each series consists of 10 spectra, natural way to bin is by flux
        # right edges are important, so append a large value to the end
        bins   = np.sort(x)[::binsize]
        bins   = np.append(bins,2*np.max(x))
        # use the bins to calculate statistics
        flux,ed,ct   = binned_statistic(x,x,'mean',bins=bins)
        minx,ed,ct   = binned_statistic(x,x,np.min,bins=bins)
        maxx,ed,ct   = binned_statistic(x,x,np.max,bins=bins)
        shift,ed,ct  = binned_statistic(x,y,'mean',bins=bins)
        error,ed,ct  = binned_statistic(x,y,'std',bins=bins)
    else:
        flux     = values['flux']
        minx     = flux
        maxx     = flux
        shift,error  = values['{}sigma'.format(sigma)].T
    # make the highest flux zero-point
    maxflux = np.argmax(flux)
    shift0  = shift[maxflux]
    shift   -= shift0
    if return_xrange:
        return flux, shift, error, flux-minx, maxx-flux
    else:
        return flux, shift, error
def print_message(args,model,npts,pars,covar,chisq,chisqnu,aicc):
    print("{0:=>81}".format(""))
    print("{0:<20} = {1:<20}".format("File",args.file))
    print("{0:<20} = {1:<20}".format("Fibre",args.fibre))
    print("{0:<20} = {1:<20}".format("Method",args.method))
    print("{0:<20} = {1:<20}".format("Fittype",args.fittype))
    print("{0:<20} = {1:<20}".format("Version",args.version))
    print("{0:<20} = {1:<20}".format("Sigma",args.sigma))
    print("{0:<20} = {1:<20}".format("Model",model))
    print("{0:<20} = {1:<20}".format("No. of points",npts))
    print("{0:->81}".format(""))
    npars = len(pars)
    errs  = np.sqrt(np.diag(covar))
    print("{0:<20} = ".format("pars"),(npars*"{:15.5e}").format(*pars))
    print("{0:<20} = ".format("errs"),(npars*"{:15.5e}").format(*errs))
    print("{0:<20} = {1:15.5f}".format("chisq",chisq))
    print("{0:<20} = {1:15.5f}".format("chisqnu",chisqnu))
    print("{0:<20} = {1:15.5f}".format("AICc",aicc))
    return
def save(args,model,pars,covar,chisq,chisqnu,aicc):
    basename  = os.path.basename(args.file)
    basenoext = os.path.splitext(basename)[0]
    filename  = 'cti_model_'+basenoext+ \
                '_{}_{}_{}.json'.format(args.fittype,args.method,model)
    dirname   = hs.get_dirname('cti')
    
    filepath  = os.path.join(dirname,filename)
    if model == 'log':
        function = 'y (x) = A + B * log(x)'
    elif model == 'exp':
        function = 'y (x) = A + B * exp(x/c)'
    data      = {'file'    : args.file,
                 'fibre'   : args.fibre,
                 'method'  : args.method,
                 'fittype' : args.fittype,
                 'version' : args.version,
                 'sigma'   : args.sigma,
                 'model'   : function,
                 'chisq'   : chisq,
                 'chisqnu' : chisqnu,
                 'aicc'    : aicc,
                 'pars'    : list(pars),
                 'errs'    : list(np.sqrt(np.diag(covar)))}
    print(data)
    with open(filepath,mode='w+') as file:
        json.dump(data,file,indent=4)
    print("SAVED TO: {}".format(filepath))
    return
#%% Simple model y = A + B*log(x) 
def log_model(xdata,*pars):
    A,B = pars
    return A+B*np.log10(xdata)
#%% Exponential model y = a + b * exp(-x/c)
def exp_model(xdata,*pars):
    b,c = pars
    return b*np.exp(-xdata/c)


#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate CTI model parameters.')
    parser.add_argument('file',type=str, 
                        help='Path to the output file')
    parser.add_argument('fibre',type=str,
                        help='Fibre')
    parser.add_argument('-v','--version',type=int,default=701,
                        help='Version')
    parser.add_argument('-ft','--fittype',type=str,default='gauss',
                        help="Fittype, default gauss.")
    parser.add_argument('-s','--sigma',type=int,default=3,
                        help='Error on RV. Sigma units.')
    parser.add_argument('-mt','--method',type=str,default='coeff',
                        help="RV method used. Options 'wavesol' (default),  " 
                        "'coeff', 'freq', 'cent'.")
    parser.add_argument('-md','--model',type=str,default=['exp','log'],nargs='*',
                        help="Fitting model used. Options 'exp' (default),  " 
                        "'log'.")
    parser.add_argument('-b','--binsize', type=int, default=0,
                        help="Bin the data before fitting.")
    parser.add_argument('--vlim',type=float,default=None,
                        help="Sets an upper limit in shift values")
    parser.add_argument('-p','--plot', action='store_true', default=False,
                        help="Plot the data and the fit model.")
    parser.add_argument('-sp','--save-plot', action='store_true', default=False,
                        help="Save the plot.")
    parser.add_argument('--notitle', dest='title', action='store_false', 
                        default=True,
                        help="Do not put title on figure.")
    parser.add_argument('--save',action='store_true',default=False,
                        help="Save output to file. Overwrites old files!")
    parser.add_argument('-nf','--nofit',action = 'store_true',default=False,
                        help="Do not plot the fit")
    args = parser.parse_args()
    print(args)
    main(args)  

#%% Fit data to all data points
#pars = {}
#covar = {}
#errors = {}
#
#for fbr in ['A','B']:
#    xdata = db1['stat'].sel(att='average_flux',fb=fbr)
#    ydata = db1['rv'].sel(fb=fbr,par='rv')
#    yerr  = db1['rv'].sel(fb=fbr,par='rv_err')
#    pars_all,covar_all = curve_fit(model,xdata,ydata,sigma=yerr,p0=pars0)
#    print("Parameters through all data points:", (len(pars_all)*("{:<10f}")).format(*pars_all))
#    pars[fbr] = pars_all
#    covar[fbr] = covar_all
#    errors[fbr] = np.sqrt([covar[fbr][i][i] for i in range(len(pars[fbr]))])
##%% Plot
#fig, ax = hf.figure(2)
#c  = {'A':'C0','B':'C1'}
#for fbr in ['A','B']:
#    xdata = db1['stat'].sel(att='average_flux',fb=fbr)
#    ydata = db1['rv'].sel(fb=fbr,par='rv')
#    yerr  = db1['rv'].sel(fb=fbr,par='rv_err')
#    ymodel = model(xdata,*pars[fbr])
#    ax[0].errorbar(xdata,ydata,yerr,marker='x',ms=4,ls='',label=fbr,c=c[fbr])
#    x = np.logspace(np.log10(xdata.min()),
#                    np.log10(xdata.max()),30)
#    ax[0].plot(x,model(x,*pars[fbr]),c=c[fbr],label="Unbinned")
#    ax[1].scatter(xdata,ydata-ymodel,c=c[fbr])
#ax[1].axhline(0,ls=':',lw=0.5,c='k')
#ax[0].legend()
#### END (rest not used)
#
#
##%% Data binning
##time_ends = np.where(np.array(np.diff(db1.time),dtype=np.float64)>1e11)[0]
#times = np.hstack([ydata.time[0],(ydata.time)[9::10]])
#x_binned = xdata.groupby_bins('time',times)
#y_binned = ydata.groupby_bins('time',times)
#
#x_mean, x_std = x_binned.mean(), x_binned.std()
#y_mean, y_std = y_binned.mean(), y_binned.std()
#
##%% Fit data to binned points
#pars_binned, covar_binned = curve_fit(model,x_mean,y_mean,sigma=y_std,p0=pars0)
#print("Parameters through binned points:", "A={0:<10f} B={1:<10f}".format(*pars_binned))
#
##%%
#fig, ax = hf.figure(1)
#ax = ax[0]
#ax.errorbar(x_mean,y_mean,y_std,marker='x',ms=4,ls='',label='')
#x = np.logspace(np.log10(xdata.min()),
#                np.log10(xdata.max()),30)
#
#ax.plot(x,model(x,*pars_all),c='C0',label="Binned")
#
##ax.plot(xdata[a],model(xdata,*pars_all)[a],c='C0',label="Unbinned")
##ax.plot(xdata,model(xdata,*pars_binned),c='C3',label="Binned")
##ax.errorbar(x_mean,y_mean,xerr=x_std,yerr=y_std,marker='o',c='C3',ls='',label='')
##ax.set_xscale('log')
#ax.legend()