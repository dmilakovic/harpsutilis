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
import harps.dataset as hd
seriesA_2012=hd.Series('/Users/dmilakov/harps/dataprod/output/v_0.6.0/scigarn/2012/2012-02-15-A.dat','A')
seriesB_2012=hd.Series('/Users/dmilakov/harps/dataprod/output/v_0.6.0/scigarn/2012/2012-02-15-B.dat','B')
# exposure 144 is mid-valued in flux in the brightest exposure sub-series (series 15 of the day)
#%%

waveA=seriesA_2012['wavesol_gauss',501]
waveB=seriesB_2012['wavesol_gauss',501]

freqA = seriesA_2012['freq_gauss',501]
freqB = seriesB_2012['freq_gauss',501]

coeffA = seriesA_2012['coeff_gauss',501]
coeffB = seriesB_2012['coeff_gauss',501]

coeffA_500 = seriesA_2012['coeff_gauss',500]
coeffB_500 = seriesB_2012['coeff_gauss',500]

centA = seriesA_2012['cent_gauss',501]
centB = seriesB_2012['cent_gauss',501]
#%%
fibres = ['A','B','A-B']
data = {'A':[waveA,coeffA,freqA,centA], 
        'B':[waveB,coeffB,freqB,centB], 
        'A-B':[waveA-waveB,coeffA-coeffB,freqA-freqB,centA-centB]}
fittypes = ['lsf']
methods = ['wavesol', 'coeff','freq', 'cent']
colors = {'wavesol':'C0', 
          'coeff':'C1', 
          'freq':'C2',
          'cent':'C3'}
series = {'A':seriesA_2012, 'B':seriesB_2012}
#%% Correct for time drift
def temporal_fit(group1,group2,model='linear',sigma=3):
    time1  = group1.mean('datetime')
    shift1 = group1.mean('{}sigma'.format(sigma))[0]
    time2  = group2.mean('datetime')
    shift2 = group2.mean('{}sigma'.format(sigma))[0]
    print(shift1,shift2)
    shiftdelta = shift2-shift1
    timedelta  = (time2-time1)/np.timedelta64(1,'s')
    # shift(t) = shift1 + (shift2-shift1)/(time2-time1)*(t-time1)
    # shift(t) = A + B * (t-t0)
    A = shift1 #- shiftdelta/timedelta
    B = shiftdelta/timedelta  # (m/s)/s
    
    return A, B, time1

def temporal_model(x,A,B):
    return A + B * x
import itertools

time_plotter=plot.Figure(2,figsize=(16,8))
# dictionary with time-corrected data
data_tc = {}
for i,f in enumerate(['A','B']):
    data_fibre = []
    extensions = [item for item in itertools.product(fittypes,methods)]
    for fittype,method in extensions:
        print(fittype,method)
        data   = series[f]['{}_{}'.format(method,fittype)]
        dtimes = hf.tuple_to_datetime(data.values['datetime'])
        time_bins = dtimes[10::10]
#        time_bins = dtimes#hf.tuple_to_datetime(dtimes)
        time_groups = data.groupby_bins('datetime',time_bins)
        
        #plot mean of groups:
        time_plotter.axes[i].scatter(
                [(time_groups['0'].mean('datetime')-dtimes[0])/np.timedelta64(1,'s'),
                 (time_groups['14'].mean('datetime')-dtimes[0])/np.timedelta64(1,'s')],
                [time_groups['0'].mean('3sigma')[0],
                 time_groups['14'].mean('3sigma')[0]])
        
        A_t, B_t, time1 = temporal_fit(time_groups['0'],time_groups['14'])
        
        x1 = (dtimes - dtimes[0]).astype(np.float64)
        time_plotter.axes[i].plot(x1,data['3sigma'].values[:,0],lw=0.4,ms=4,
                         label="{0} uncorrected".format(method),marker='o')
        dc = data.correct_time((A_t,B_t),time1,copy=True)
        
        data_fibre.append(dc)
        
        time_plotter.axes[i].plot(x1,dc['3sigma'].values[:,0],lw=0.4,ms=4,
                         label="{0} corrected".format(method),marker='^')
        dt = np.linspace(0,26e3,100)
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
data_plotter=plot.Figure(4,figsize=(16,9),sharex=True,
                         sharey=[True,False,True,False],
                         top=0.8,
                         ratios=[3,1,3,1])
data_plotter.figure.suptitle('Flux intensity model')
axes = data_plotter.axes
#data   = {'A':rvwA_2012,'B':rvwB_2012}
scale = 'log'
sigma = 5
key   = '{}sigma'.format(sigma)
parsrec = np.zeros(1,dtype=cti.pardtype)

for i,f in enumerate(['A','B']):
    print((2*("{:^15}")).format('Fibre','Method') + \
          (4*("{:<20}")).format('A_mu','B_mu','A_sig','B_sig'))
    for d,method in zip(data_tc[f],methods):
        axnum = i*2
        ax0 = axes[axnum]
        ax1 = axes[axnum+1]
        d.plot(3,plotter=data_plotter,axnum=axnum,c=colors[method],ms=5,
                  label=method,scale='flux',ls='',marker='o',alpha=0.5,)
#        rvwB.plot(plotter=plotter,axnum=i,
#                  label='B',scale='flux',ls='',m='s',alpha=0.5)
#        diff.plot(plotter=plotter,axnum=i,
#                  label='B-A',scale='flux',ls='',m='o',alpha=0.5)
        x = d['flux'].values
        y, sigma_y = d['5sigma'].values.T
        # remove outliers
        cut = np.where(np.abs(y)<10)
        
        #sigma_y = d['noise'].values
        pars, covar = curve_fit(model,x[cut],y[cut],initp[i])#,#sigma=x[cut]/1e6,
#                                absolute_sigma=False)
        errs = np.sqrt(np.diag(covar))
        
        parsrec[f][fittype][method]['pars'] = pars
        parsrec[f][fittype][method]['errs'] = errs
        print((2*("{:^15}")).format(f,method) + \
              (4*("{:<20.8f}")).format(pars[0],pars[1],errs[0],errs[1]))
        #plotter.axes[i].errorbar(x,y,sigma,marker='s',ms=2,
        #            ls='',capsize=2,label=labels[i])
        x_model = np.linspace(d.min('flux'),d.max('flux'),100)
        y_model = model(x_model,*pars)
        ax0.plot(x_model,y_model,c=colors[method],lw=2)
        ax0.text(0.9,0.8,"Fibre {}".format(f),#fontsize=10,
                horizontalalignment='left',transform=ax0.transAxes)
#        plotter.axes[i].set_xscale('log')
        #plotter.axes[i].set_yscale('log')
        
        ax0.set_xscale(scale)
        d_cticorr=d.correct_cti(method,f,copy=True)
        d_cticorr.plot(5,plotter=data_plotter,axnum=axnum+1,ms=5,
                  label='{} corrected'.format(method),c=colors[method],
                  scale='flux',ls='',marker='^',alpha=0.5)
        if f=='A':
            data_plotter.axes[i].legend(loc='upper center',
                             bbox_to_anchor=(0.5,1.6),ncol=len(method))
figname   = '2012-02_shift_model_{}.pdf'.format(scale)
folder    = '/Users/dmilakov/harps/dataprod/plots/CTI/'
#data_plotter.fig.savefig(os.path.join(folder,figname))

#%% PLOT SHIFT AS A FUNCTION OF SEQUENCE NUMBER
#%  Plot different methods for each fibre

#def get_specifier(orders,pixels,sigma):
#    specifier = "orders={0}-{1}_".format(*orders) + \
#                "pixels={0}-{1}_".format(*pixels) + \
#                "{0}sigma".format(sigma)
#    return specifier
#
#
#labels = ['COMB segmented', 'Lines', 'Lines + COMB segmented', 'Lines + COMB unsegmented']
#
#method_plotter = plot.Figure(len(fibres),sharex=True,sharey=True,top=0.85)
#method_plotter.fig.suptitle(r"2012-02-17: orders=42-48, "
#                            "pixels=0-4096, ${}\sigma$-clipping".format(sigma))
#
#for i,f in enumerate(fibres):
#    ax = method_plotter.axes[i]
#    ax.text(0.85,0.8,"Fibre {}".format(f),fontsize=12,
#                       transform=ax.transAxes)
#    for j,method in enumerate(data[f]):
#        method.plot(plotter=method_plotter,axnum=i,label=labels[j],legend=False)
#method_plotter.axes[0].legend(loc='upper center',bbox_to_anchor=(0.5, 1.4),ncol=len(labels))
##method_plotter.axes[0].set_ylim(-1.5,4)
#
#specifier = get_specifier(orders,pixels,sigma)
#figname   = '2012-02_shift_method_comparison_{}.pdf'.format(specifier)
#folder    = '/Users/dmilakov/harps/dataprod/plots/CTI/'
#method_plotter.fig.savefig(os.path.join(folder,figname))
#
##%% PLOT SHIFT AS A FUNCTION OF FLUX
#flux_plotter = plot.Figure(len(labels),figsize=(16,10),grid=(2,2),
#                           sharex=True,sharey=True,top=0.85)
#flux_plotter.fig.suptitle(r"2012-02-17: orders={0}-{1}, "
#                            "pixels={2}-{3}, ${4}\sigma$-clipping".format(*orders,*pixels,sigma))
#colors = {'A':'C0','B':'C3'}
#for j,f in enumerate(fibres):
#    if f=='A-B': continue
#    for i,d in enumerate(data[f]):
#        ax = flux_plotter.axes[i]
#        ax.text(0.97,0.9,"{}".format(labels[i]),fontsize=10,horizontalalignment='right',
#                       transform=ax.transAxes)
#        d.plot(plotter=flux_plotter,scale='flux',axnum=i,label=f,
#               ls='',legend=False,c=colors[f])
#        
#        x = d['flux'].values
#        y = d['shift'].values
#        sigma_y = d['noise'].values
#        pars, covar = curve_fit(model_log,x,y,sigma=sigma_y,absolute_sigma=False,
#                                p0=(-1.3,-1.5))
#        x_model = np.linspace(d.min('flux'),d.max('flux'),100)
#        y_model = model_log(x_model,*pars)
#        flux_plotter.axes[i].plot(x_model,y_model,lw=2,c=colors[f],alpha=0.5)
#flux_plotter.axes[0].legend(loc='upper center',bbox_to_anchor=(1., 1.3),ncol=len(fibres))
#flux_plotter.axes[0].set_xscale('log')
#flux_plotter.axes[0].set_ylim(-2.,7)
#
#
#specifier = get_specifier(orders,pixels,sigma)
#figname   = '2012-02_shift_vs_flux_method_comparison_{}.pdf'.format(specifier)
#folder    = '/Users/dmilakov/harps/dataprod/plots/CTI/'

