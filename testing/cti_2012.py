#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:09:06 2019

@author: dmilakov
"""
import harps.series as ser
from harps.core import np, curve_fit, os
import harps.plotter as plot
import harps.functions as hf
import harps.wavesol as ws
seriesA_2012=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.6/scigarn/2012-02-15_fibreA_v501.dat','A',refindex=0)
seriesB_2012=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.6/scigarn/2012-02-15_fibreB_v501.dat','B',refindex=0)
# exposure 144 is mid-valued in flux in the brightest exposure sub-series (series 15 of the day)
#%%
#tharsolA = ws.ThAr('/Volumes/dmilakov/harps/data/2012-02-15/series01/HARPS.2012-02-15T13_26_45.666_e2ds_A.fits',True)
#tharsolB = ws.ThAr('/Volumes/dmilakov/harps/data/2012-02-15/series01/HARPS.2012-02-15T13_26_45.666_e2ds_B.fits',True)
#coeffsA  = tharsolA.coeffs
#coeffsB  = tharsolB.coeffs
#%%
sigma = 2

pixels = (0*512,8*512)
orders = (41,48)

waveA=seriesA_2012.wavesol(pixels=pixels,orders=orders,sigma=sigma,ravel=True)
waveB=seriesB_2012.wavesol(pixels=pixels,orders=orders,sigma=sigma,ravel=True)

freqA = seriesA_2012.interpolate(use='freq',orders=orders,sigma=sigma)
freqB = seriesB_2012.interpolate(use='freq',orders=orders,sigma=sigma)

coeffA = seriesA_2012.coefficients(orders=orders,sigma=sigma,version=501)
coeffB = seriesB_2012.coefficients(orders=orders,sigma=sigma,version=501)

coeffA_500 = seriesA_2012.coefficients(orders=orders,sigma=sigma,version=500)
coeffB_500 = seriesB_2012.coefficients(orders=orders,sigma=sigma,version=500)
#%%
fibres = ['A','B','A-B']
data = {'A':[waveA,freqA,coeffA,coeffA_500], 
        'B':[waveB,freqB,coeffB,coeffB_500],
        'A-B':[waveA-waveB,freqA-freqB,coeffA-coeffB,coeffA_500-coeffB_500]}
methods = ['COMB segmented', 'Lines', 'Lines + COMB segmented',
           'Lines + COMB unsegmented']
colors = {'COMB segmented':'C0', 
          'Lines':'C1', 
          'Lines + COMB segmented':'C2',
          'Lines + COMB unsegmented':'C3'}
#%% Correct for time drift
def temporal_fit(group1,group2,model='linear'):
    time1  = group1.mean('datetime')
    shift1 = group1.mean('shift')
    time2  = group2.mean('datetime')
    shift2 = group2.mean('shift')
    shiftdelta = shift2-shift1
    timedelta  = (time2-time1)/np.timedelta64(1,'s')
    # shift(t) = shift1 + (shift2-shift1)/(time2-time1)*(t-time1)
    # shift(t) = A + B * t
    A = shift1 - shiftdelta/timedelta
    B = shiftdelta/timedelta
    
    return A, B, time1

def temporal_model(x,A,B):
    return A + B * x


time_plotter=plot.Figure(2,figsize=(16,8))
# dictionary with time-corrected data
data_tc = {}
for i,f in enumerate(['A','B']):
    data_fibre = []
    for d, method in zip(data[f],methods):
        time_bins   = d['datetime'][10::10].values
        time_groups = d.groupby_bins('datetime',time_bins)
        A_t, B_t, time1 = temporal_fit(time_groups['0'],time_groups['14'])
        
        x1 = (d.values['datetime']-d.values['datetime'][0]).astype(np.float64)
        time_plotter.axes[i].plot(x1,d['shift'].values,lw=0.4,
                         label="{0} uncorrected".format(f),marker='o')
        dc = d.correct_time((A_t,B_t),time1,copy=True)
        
        data_fibre.append(dc)
        
        time_plotter.axes[i].plot(x1,dc['shift'].values,lw=0.4,
                         label="{0} corrected".format(f),marker='^')
        dt = np.linspace(0,28e3,100)
        time_plotter.axes[i].plot(dt,temporal_model(dt,A_t,B_t))
        time_plotter.axes[i].axhline(0,ls=':',c='k',lw=0.5)
        time_plotter.axes[i].legend()
    data_tc[f]=data_fibre

#%% Exponential model y = a + b * exp(-x/c)
def model_exp(xdata,*pars):
    a,b = pars
    return a*np.exp(-xdata/b)
#pars0 = (0,10)
#pars_exp, covar_exp = curve_fit(model_exp,x_mean,y_mean,sigma=y_std,p0=pars0)
#print("EXP Parameters through binned points:", "A={0:<10e} B={1:<10e}".format(*pars_exp))

#%% Simple model y = A + B*x', where x is log10
def model_log(xdata,*pars):
    A,B = pars
    return A*(np.log10(xdata)-np.log10(B))
#pars0 = (1,1)
#pars_log, covar_log = curve_fit(model_log,x_mean,y_mean,sigma=y_std,p0=pars0)
#print("LOG Parameters through binned points:", "A={0:<10e} B={1:<10e}".format(*pars_log))
#%%
#def model(xdata):
#    A, B = (0.26290918, 0.70655505)
#    return A + B*np.log10(xdata/1e6)
#%%
model  = model_exp
initp  = [(3,7e4),(2,9e4)]
labels = [ r'$y(x)=a+b\cdot\log{x}$']
plotter=plot.Figure(2,figsize=(12,12),sharex=True,sharey=True)
#data   = {'A':rvwA_2012,'B':rvwB_2012}

use = slice(0,2)
for i,f in enumerate(['A','B']):
    for d,method in zip(data_tc[f][use],methods[use]):
        d.plot(plotter=plotter,axnum=i,c=colors[method],
                  label=method,scale='flux',ls='',marker='o',alpha=0.5,)
#        rvwB.plot(plotter=plotter,axnum=i,
#                  label='B',scale='flux',ls='',m='s',alpha=0.5)
#        diff.plot(plotter=plotter,axnum=i,
#                  label='B-A',scale='flux',ls='',m='o',alpha=0.5)
        x = d['flux'].values
        y = d['shift'].values
        sigma_y = d['noise'].values
        pars, covar = curve_fit(model,x,y,initp[i],10*sigma_y,True)
        print(f,method,pars,np.sqrt(np.diag(covar)))

        #plotter.axes[i].errorbar(x,y,sigma,marker='s',ms=2,
        #            ls='',capsize=2,label=labels[i])
        x_model = np.linspace(d.min('flux'),d.max('flux'),100)
        y_model = model(x_model,*pars)
        plotter.axes[i].plot(x_model,y_model,c=colors[method],lw=2)
#        plotter.axes[i].set_xscale('log')
        #plotter.axes[i].set_yscale('log')
        plotter.axes[i].legend()
        
        d_cticorr=d.correct_cti(f,copy=True)
        d_cticorr.plot(plotter=plotter,axnum=i,
                  label='{} corrected'.format(method),c=colors[method],
                  scale='flux',ls='',marker='^',alpha=0.5)
        
#%%
pixels = (0*512,8*512)
orders = (41,51)
rvwA_2012=seriesA_2012.wavesol(pixels=pixels,orders=orders)
rvwB_2012=seriesB_2012.wavesol(pixels=pixels,orders=orders)
plotter = plot.Figure(2)
data   = {'A':rvwA_2012,'B':rvwB_2012}

for f in ['A','B']:
    data[f].plot(plotter=plotter,label=f)
diff_uncorrected = data['A']-data['B']
diff_uncorrected.plot(plotter=plotter,label='A-B')

#for f in ['A','B']:
#    data[f].plot(plotter=plotter,label=f)
#%% PLOT SHIFT AS A FUNCTION OF SEQUENCE NUMBER
#%  Plot different methods for each fibre

def get_specifier(orders,pixels,sigma):
    specifier = "orders={0}-{1}_".format(*orders) + \
                "pixels={0}-{1}_".format(*pixels) + \
                "{0}sigma".format(sigma)
    return specifier


labels = ['COMB segmented', 'Lines', 'Lines + COMB segmented', 'Lines + COMB unsegmented']

method_plotter = plot.Figure(len(fibres),sharex=True,sharey=True,top=0.85)
method_plotter.fig.suptitle(r"2012-02-17: orders=42-48, "
                            "pixels=0-4096, ${}\sigma$-clipping".format(sigma))

for i,f in enumerate(fibres):
    ax = method_plotter.axes[i]
    ax.text(0.85,0.8,"Fibre {}".format(f),fontsize=12,
                       transform=ax.transAxes)
    for j,method in enumerate(data[f]):
        method.plot(plotter=method_plotter,axnum=i,label=labels[j],legend=False)
method_plotter.axes[0].legend(loc='upper center',bbox_to_anchor=(0.5, 1.4),ncol=len(labels))
#method_plotter.axes[0].set_ylim(-1.5,4)

specifier = get_specifier(orders,pixels,sigma)
figname   = '2012-02_shift_method_comparison_{}.pdf'.format(specifier)
folder    = '/Users/dmilakov/harps/dataprod/plots/CTI/'
method_plotter.fig.savefig(os.path.join(folder,figname))

#%% PLOT SHIFT AS A FUNCTION OF FLUX
flux_plotter = plot.Figure(len(labels),figsize=(16,10),grid=(2,2),
                           sharex=True,sharey=True,top=0.85)
flux_plotter.fig.suptitle(r"2012-02-17: orders={0}-{1}, "
                            "pixels={2}-{3}, ${4}\sigma$-clipping".format(*orders,*pixels,sigma))
colors = {'A':'C0','B':'C3'}
for j,f in enumerate(fibres):
    if f=='A-B': continue
    for i,d in enumerate(data[f]):
        ax = flux_plotter.axes[i]
        ax.text(0.97,0.9,"{}".format(labels[i]),fontsize=10,horizontalalignment='right',
                       transform=ax.transAxes)
        d.plot(plotter=flux_plotter,scale='flux',axnum=i,label=f,
               ls='',legend=False,c=colors[f])
        
        x = d['flux'].values
        y = d['shift'].values
        sigma_y = d['noise'].values
        pars, covar = curve_fit(model_log,x,y,sigma=sigma_y,absolute_sigma=False,
                                p0=(-1.3,-1.5))
        x_model = np.linspace(d.min('flux'),d.max('flux'),100)
        y_model = model_log(x_model,*pars)
        flux_plotter.axes[i].plot(x_model,y_model,lw=2,c=colors[f],alpha=0.5)
flux_plotter.axes[0].legend(loc='upper center',bbox_to_anchor=(1., 1.3),ncol=len(fibres))
flux_plotter.axes[0].set_xscale('log')
flux_plotter.axes[0].set_ylim(-2.,7)


specifier = get_specifier(orders,pixels,sigma)
figname   = '2012-02_shift_vs_flux_method_comparison_{}.pdf'.format(specifier)
folder    = '/Users/dmilakov/harps/dataprod/plots/CTI/'

