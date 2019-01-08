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
seriesA_2012=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.6/scigarn/2012-02-15_fibreA_v501.dat',refindex=5)
seriesB_2012=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.6/scigarn/2012-02-15_fibreB_v501.dat',refindex=5)
# exposure 144 is mid-valued in flux in the brightest exposure sub-series (series 15 of the day)
#%%
pixels = (0*512,8*512)
orders = (41,51)
rvwA_2012=seriesA_2012.rv_from_wavesol(pixels=pixels,orders=orders)
rvwB_2012=seriesB_2012.rv_from_wavesol(pixels=pixels,orders=orders)

#%% Remove linear trend with time:

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



    #%%
#nominal1 = {'A':rvwA_2012[0:10],'B':rvwB_2012[0:10]}
#nominal2 = {'A':rvwA_2012[-10:],'B':rvwB_2012[-10:]}
data     = {'A':rvwA_2012,'B':rvwB_2012}
plotter_time=plot.Figure(2,figsize=(16,8))
for i,f in enumerate(['A','B']):
    time_bins   = data[f]['datetime'][10::10].values
    time_groups = data[f].groupby_bins('datetime',time_bins)
    A_t, B_t, time1 = temporal_fit(time_groups['0'],time_groups['14'])
    
    x1 = (data[f].values['datetime']-data[f].values['datetime'][0]).astype(np.float64)
    plotter_time.axes[i].plot(x1,data[f]['shift'].values,lw=0.4,label="{0} uncorrected".format(f),marker='o')
    data[f].correct_time((A_t,B_t),time1)
    plotter_time.axes[i].plot(x1,data[f]['shift'].values,lw=0.4,label="{0} corrected".format(f),marker='^')
    dt = np.linspace(0,28e3,100)
    plotter_time.axes[i].plot(dt,temporal_model(dt,A_t,B_t))
    plotter_time.axes[i].axhline(0,ls=':',c='k',lw=0.5)
    plotter_time.axes[i].legend()
#    nominal1[f].plot(plotter=plotter_time,axnum=i,label=f,scale='time',m='^')
#    nominal2[f].plot(plotter=plotter_time,axnum=i,label=f,scale='time',m='o')

##%% NORMALISE THE SHIFT IN EACH FIBRE
## ordinal number of the group to use in each fibre. 
#idx = {'A':'4','B':'3'}
#fluxes = {}
#shifts = {}
#for i,f in enumerate(['A','B']): 
#    time_bins   = data[f]['datetime'][10::10].values
#    time_groups = data[f].groupby_bins('datetime',time_bins)
#    norm_group  = time_groups[idx[f]]
#    mean_flux   = norm_group.mean('flux')
#    mean_shift  = norm_group.mean('shift')
#    
#    fluxes[f] = mean_flux  
#    shifts[f] = mean_shift
#
#plotter=plot.Figure(1,figsize=(16,8),bottom=0.12)
#    
#rvwA_2012.plot(plotter=plotter,axnum=0,
#          label='A',scale='flux',ls='',m='^',alpha=0.5)
#rvwB_2012.plot(plotter=plotter,axnum=0,
#          label='B',scale='flux',ls='',m='s',alpha=0.5)
#normal = ser.RV(np.copy(rvwA_2012.values))
#normal['shift'] = shifts['A']/shifts['B']
#summed = rvwB_2012*normal
#summed.plot(plotter=plotter,axnum=0,
#          label='B, multiplicative factor',scale='flux',ls='',m='s',alpha=0.5)
##%%
#
#
#diff = rvwB_2012-rvwA_2012
#
#times  = diff['datetime'][10::10].values
#groups = diff.groupby_bins('datetime',times)
#
#x_mean = np.array([group.mean('flux') for group in groups.values()])
#y_mean = np.array([group.mean('shift') for group in groups.values()])
#x_high = np.array([group.max('flux')-group.mean('flux')   for group in groups.values()])
#x_low  = np.array([group.mean('flux') - group.min('flux') for group in groups.values()])
#y_std  = np.array([group.std('shift') for group in groups.values()])
#x_std  = np.vstack([x_low,x_high]) 
#%%
plotter=plot.Figure(1,figsize=(16,8),bottom=0.12)
    
rvwA_2012.plot(plotter=plotter,axnum=0,
          label='A',scale='flux',ls='',m='^',alpha=0.5)
rvwB_2012.plot(plotter=plotter,axnum=0,
          label='B',scale='flux',ls='',m='s',alpha=0.5)
#diff.plot(plotter=plotter,axnum=0,
#          label='B-A',scale='flux',ls='',m='o',alpha=0.5)
#plotter.axes[0].scatter(x_mean,y_mean,c='r',marker='x')
#%%


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
models = [model_exp,model_log]
initp  = [(0,10),(2,-1)]
labels = [r'$y(x)=a+b\cdot\exp{(-x/c)}$', r'$y(x)=a+b\cdot\log{x}$']
plotter=plot.Figure(2,figsize=(12,12))
data   = {'A':rvwA_2012,'B':rvwB_2012}

for f in ['A','B']:
    for i,model in enumerate(models):
        print(f,i,model)
        data[f].plot(plotter=plotter,axnum=i,
                  label=f,scale='flux',ls='',m='o',alpha=0.5)
#        rvwB.plot(plotter=plotter,axnum=i,
#                  label='B',scale='flux',ls='',m='s',alpha=0.5)
#        diff.plot(plotter=plotter,axnum=i,
#                  label='B-A',scale='flux',ls='',m='o',alpha=0.5)
        x = data[f]['flux'].values
        y = data[f]['shift'].values
        sigma = data[f]['noise'].values
        pars, covar = curve_fit(model,x,y,sigma=sigma,
                                           p0=initp[i])
        #plotter.axes[i].errorbar(x,y,sigma,marker='s',ms=2,
        #            ls='',capsize=2,label=labels[i])
        x_model = np.linspace(data[f].min('flux'),data[f].max('flux'),100)
        y_model = model(x_model,*pars)
        plotter.axes[i].plot(x_model,y_model,lw=2,c='C3')
        plotter.axes[i].set_xscale('log')
        #plotter.axes[0].set_yscale('log')
        plotter.axes[i].legend()