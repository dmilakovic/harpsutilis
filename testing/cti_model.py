#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:09:06 2019

@author: dmilakov
"""
import harps.series as ser
from harps.core import np, curve_fit
import harps.plotter as plot
import harps.functions as hf
seriesA=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.6/scigarn_201202_fibreA_v501.dat')
seriesB=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.6/scigarn_201202_fibreB_v501.dat')
#%%
pixels = (512,7*512)
orders = (42,50)
rvwA=seriesA.rv_from_wavesol(pixels=pixels,orders=orders)
rvwB=seriesB.rv_from_wavesol(pixels=pixels,orders=orders)

#%%
diff = rvwB-rvwA

groups = np.split(diff,15)

x_mean = np.array([np.mean(group['flux'])  for group in groups])
y_mean = np.array([np.mean(group['shift']) for group in groups])
x_high = np.array([np.max(group['flux'])-np.mean(group['flux'])   for group in groups])
x_low  = np.array([np.mean(group['flux']) - np.min(group['flux'])   for group in groups])
y_std  = np.array([np.std(group['shift'])  for group in groups])
x_std  = np.vstack([x_low,x_high]) 
#%%


#%% Exponential model y = a + b * exp(-x/c)
def model_exp(xdata,*pars):
    a,b,c = pars
    return a + b*np.exp(-xdata/c)
pars0 = (0,10,1e6)
pars_exp, covar_exp = curve_fit(model_exp,x_mean,y_mean,sigma=y_std,p0=pars0)
print("Parameters through binned points:", "A={0:<10e} B={1:<10e} C={2:<10e}".format(*pars_exp))

#%% Simple model y = A + B*x', where x is log10
def model_log(xdata,*pars):
    A,B = pars
    return A+B*np.log10(xdata)
pars0 = (1,1)
pars_log, covar_log = curve_fit(model_log,x_mean,y_mean,sigma=y_std,p0=pars0)
print("Parameters through binned points:", "A={0:<10e} B={1:<10e}".format(*pars_log))
#%%
models = [model_exp,model_log]
initp  = [(0,10,1e6),(1,1)]
labels = [r'$y(x)=a+b\cdot\exp{(-x/c)}$', r'$y(x)=a+b\cdot\log{x}$']
plotter=plot.Figure(2,figsize=(12,12))
    
for i,model in enumerate(models):
    rvwA.plot(plotter=plotter,axnum=i,
              label='A',scale='flux',ls='',m='^',alpha=0.5)
    rvwB.plot(plotter=plotter,axnum=i,
              label='B',scale='flux',ls='',m='s',alpha=0.5)
    diff.plot(plotter=plotter,axnum=i,
              label='B-A',scale='flux',ls='',m='o',alpha=0.5)
    
    pars, covar = curve_fit(model,x_mean,y_mean,sigma=y_std,
                                       p0=initp[i])
    plotter.axes[i].errorbar(x_mean,y_mean,y_std,x_std,marker='s',ms=2,
                ls='',capsize=2,label=labels[i])
    x = np.linspace(np.min(diff['flux']),np.max(diff['flux']),100)
    y = model(x,*pars)
    plotter.axes[i].plot(x,y,lw=2,c='C3')
    plotter.axes[i].set_xscale('log')
    #plotter.axes[0].set_yscale('log')
    plotter.axes[i].legend()