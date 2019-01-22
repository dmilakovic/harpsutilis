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
        assert len(item)<4, "Too many values to unpack"
        function = None
        kwargs = {}
        if len(item)==3:
            key, function, kwargs = item
        elif len(item)==2:
            key, function = item
        else:
            key = item
        return key, function, kwargs
#%%
fittype='gauss'
version=1
seriesA_2015=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.7/scigarn/2015-04-17_fibreA.dat','A',
                   fittype,refindex=0,version=version)
seriesB_2015=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.7/scigarn/2015-04-17_fibreB.dat','B',
                   fittype,refindex=0,version=version)
#%%
cache = CacheObject()        

#%% COMPARISON OF METHODS, BOTH FIBRES INDIVIDUALLY
series_dict = [seriesA_2015,seriesB_2015]
sigma = 4
for fibre,series in zip(['A','B'],series_dict):
    wave=cache['wave_{}'.format(fibre),series.wavesol,dict(sigma=sigma)]   
    lines=cache['lines_{}'.format(fibre),series.interpolate,dict(use='freq')]
    lines_cen=cache['lines_cen_{}'.format(fibre),series.interpolate,dict(use='centre')]
    coeff=cache['coeff_{}'.format(fibre),series.coefficients,dict(version=version)]
    
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
    figname = '2015-04-17_comparison_fibre{0}_sigma={1}.pdf'.format(fibre,sigma)
    #plotter.fig.savefig(os.path.join(dirname,figname))

