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
tharsolA = ws.ThAr('/Volumes/dmilakov/harps/data/2012-02-15/series01/HARPS.2012-02-15T13_26_45.666_e2ds_A.fits',True)
tharsolB = ws.ThAr('/Volumes/dmilakov/harps/data/2012-02-15/series01/HARPS.2012-02-15T13_26_45.666_e2ds_B.fits',True)
coeffsA  = tharsolA.coeffs
coeffsB  = tharsolB.coeffs
#%%
sigma = 2

pixels = (0*512,8*512)
orders = (41,48)

rvwA=seriesA_2012.wavesol(pixels=pixels,orders=orders,sigma=sigma,ravel=True)
rvwB=seriesB_2012.wavesol(pixels=pixels,orders=orders,sigma=sigma,ravel=True)

rvlA = seriesA_2012.interpolate(use='freq',orders=orders,sigma=sigma)
rvlB = seriesB_2012.interpolate(use='freq',orders=orders,sigma=sigma)

rvcA = seriesA_2012.coefficients(orders=orders,sigma=sigma,version=501)
rvcB = seriesB_2012.coefficients(orders=orders,sigma=sigma,version=501)

rvcA_500 = seriesA_2012.coefficients(orders=orders,sigma=sigma,version=500)
rvcB_500 = seriesB_2012.coefficients(orders=orders,sigma=sigma,version=500)

fibres = ['A','B','A-B']
data = {'A':[rvwA,rvlA,rvcA,rvcA_500], 
        'B':[rvwB,rvlB,rvcB,rvcB_500],
        'A-B':[rvwA-rvwB,rvlA-rvlB,rvcA-rvcB,rvcA_500-rvcB_500]}

#%%
def get_specifier(orders,pixels,sigma):
    specifier = "orders={0}-{1}_".format(*orders) + \
                "pixels={0}-{1}_".format(*pixels) + \
                "{0}sigma".format(sigma)
    return specifier
#%% PLOT SHIFT AS A FUNCTION OF SEQUENCE NUMBER
#%  Plot different methods for each fibre



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


#%% Exponential model y = a + b * exp(-x/c)
def model_exp(xdata,*pars):
    a,b = pars
    return a + b*np.exp(-xdata/1e6)
pars0 = (0,10)
pars_exp, covar_exp = curve_fit(model_exp,x_mean,y_mean,sigma=y_std,p0=pars0)
print("EXP Parameters through binned points:", "A={0:<10e} B={1:<10e}".format(*pars_exp))

#%% Simple model y = A + B*x', where x is log10
def model_log(xdata,*pars):
    A,B = pars
    return A+B*np.log10(xdata/1e6)
pars0 = (1,1)
pars_log, covar_log = curve_fit(model_log,x_mean,y_mean,sigma=y_std,p0=pars0)
print("LOG Parameters through binned points:", "A={0:<10e} B={1:<10e}".format(*pars_log))
#%%
#def model(xdata):
#    A, B = (0.26290918, 0.70655505)
#    return A + B*np.log10(xdata/1e6)
#%%
models = [model_log]
initp  = [(2,-1)]
labels = [ r'$y(x)=a+b\cdot\log{x}$']
plotter=plot.Figure(len(models),figsize=(12,12))
data   = {'A':rvwA_2012,'B':rvwB_2012}

for f in ['A','B']:
    for i,model in enumerate(models):
        data[f].plot(plotter=plotter,axnum=i,
                  label=f,scale='flux',ls='',m='o',alpha=0.5)
#        rvwB.plot(plotter=plotter,axnum=i,
#                  label='B',scale='flux',ls='',m='s',alpha=0.5)
#        diff.plot(plotter=plotter,axnum=i,
#                  label='B-A',scale='flux',ls='',m='o',alpha=0.5)
        x = data[f]['flux'].values
        y = data[f]['shift'].values
        sigma = data[f]['noise'].values
        pars, covar = curve_fit(model,x,y,
                                           p0=initp[i])
        print(f,i,model,pars,np.sqrt(np.diag(covar)))

        #plotter.axes[i].errorbar(x,y,sigma,marker='s',ms=2,
        #            ls='',capsize=2,label=labels[i])
        x_model = np.linspace(data[f].min('flux'),data[f].max('flux'),100)
        y_model = model(x_model,*pars)
        plotter.axes[i].plot(x_model,y_model,lw=2)
        plotter.axes[i].set_xscale('log')
        #plotter.axes[0].set_yscale('log')
        plotter.axes[i].legend()
        
        data[f].correct_cti(f)
        data[f].plot(plotter=plotter,axnum=i,
                  label='{} corrected'.format(f),
                  scale='flux',ls='',m='o',alpha=0.5)
        
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
