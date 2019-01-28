#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:59:42 2019

@author: dmilakov
"""

import harps.series as ser
from   harps.core import np, curve_fit,os
import harps.plotter as plot
import harps.functions as hf
import harps.settings as hs
#%%
class CacheObject(object):
    def __init__(self):
        self._cache = {}
    def __call__(self,function,*args,**kwargs):
        return function(*args,**kwargs)
    def __getitem__(self,item):
        key, function, kwargs = self._unpack_item(item)
        try:
            data = self._cache[key]
        except:
            data = self.__call__(function,**kwargs)
            self._cache[key] = data
        return data
    def _unpack_item(self,item):
        function = None
        kwargs = {}
        if isinstance(item,tuple):
            assert len(item)>3, "Too many values to unpack"
            if len(item)==3:
                key, function, kwargs = item
            elif len(item)==2:
                key, function = item
            else:
                key = item
        elif isinstance(item,str):
            key = item
        return key, function, kwargs
#%%
fittype='gauss'
version=501
sigma = 4
seriesA_2015=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.7/scigarn/2015-04-17_fibreA_series0103.dat','A',
                   refindex=0,version=version)
seriesB_2015=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.7/scigarn/2015-04-17_fibreB_series0103.dat','B',
                   refindex=0,version=version)
#%%
cache = CacheObject()        

#%% COMPARISON OF METHODS, BOTH FIBRES INDIVIDUALLY
series_dict = [seriesA_2015,seriesB_2015]
for fb,series in zip(['A','B'],series_dict):
    for ft in ['gauss','lsf']:
        wave_gauss=cache['wave_{}_{}'.format(ft,fb),series.wavesol,dict(fittype=ft,sigma=sigma)]   
        lines=cache['lines_{}_{}'.format(ft,fb),series.interpolate,dict(fittype=ft,sigma=sigma,use='freq')]
        lines_cen=cache['lines_cen_{}_{}'.format(ft,fb),series.interpolate,dict(fittype=ft,sigma=sigma,use='centre')]
        coeff=cache['coeff_{}_{}'.format(ft,fb),series.coefficients,dict(fittype=ft,sigma=sigma,version=version)]

#%%
methods = ['wave','lines','lines_cen','coeff']
fittypes = ['gauss','lsf']
fibres   = ['A','B','A-B']
for mt in methods:
    plotter=plot.Figure(3,sharex=True)
    plotter.fig.suptitle(mt)
    for ft in fittypes:
        resA = cache['{}_{}_{}'.format(mt,ft,'A')]
        resB = cache['{}_{}_{}'.format(mt,ft,'B')]
        diff = resA-resB
        resA.plot(plotter=plotter,axnum=0,label=ft)
        resB.plot(plotter=plotter,axnum=1,label=ft)
        diff.plot(plotter=plotter,axnum=2,label=ft)
    for ax,fb in zip(plotter.axes,fibres):
        ax.text(0.04,0.8,fb,fontsize=10,horizontalalignment='left',
                       transform=ax.transAxes)
        
#%%
wgA=cache['wave_gauss_A']
wgB=cache['wave_gauss_B']
wlA=cache['wave_lsf_A']
wlB=cache['wave_lsf_B']
wgdiff=(wgA-wgB)
wldiff=(wlA-wlB)

wgA.plot(plotter=plotter,axnum=0,label='gauss')
wgB.plot(plotter=plotter,axnum=1,label='gauss')
wlA.plot(plotter=plotter,axnum=0,label='lsf')
wlB.plot(plotter=plotter,axnum=1,label='lsf')
wgdiff.plot(plotter=plotter,axnum=2,label='gauss')
wldiff.plot(plotter=plotter,axnum=2,label='lsf')

#%%
    plotter=plot.Figure(2,sharex=True,ratios=[3,1])
    plotter.fig.suptitle('Fibre {}'.format(fibre))
    wave.plot(plotter=plotter,label='wave')
    lines.plot(plotter=plotter,label='lines:freq')
    lines_cen.plot(plotter=plotter,label='lines:centre')
    coeff.plot(plotter=plotter,label='coeff')
    plotter.axes[1].axhline(0,lw=0.7,c='k',ls=':')
    plotter.axes[1].plot((lines-wave).values['shift'],marker='o',ms=3,c='C1',
                  lw=0.5,label='lines:freq-wave')
    plotter.axes[1].plot((lines_cen-wave).values['shift'],marker='o',ms=3,c='C2',
                  lw=0.5,label='lines:centre-wave')
    plotter.axes[1].plot((coeff-wave).values['shift'],marker='o',ms=3,c='C3',
                  lw=0.5,label='coeff-wave')
    dirname = os.path.join(hs.dirnames['plots'],'method_comparison')
    figname = '2015-04-17_comparison_fibre{0}_sigma={1}_fittype={2}_version={3}.pdf'.format(fibre,sigma,fittype,version)
    plotter.fig.savefig(os.path.join(dirname,figname))

