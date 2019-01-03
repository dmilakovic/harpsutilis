#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 09:20:54 2019

@author: dmilakov
"""

import harps.series as ser
import harps.plotter as plot

seriesA=ser.Series('/Users/dmilakov/harps/dataprod/output/v_0.5.5/'
                   'fibreA_series0103_scigarn_refexp50.dat',refindex=0)
seriesB=ser.Series('/Users/dmilakov/harps/dataprod/output/v_0.5.5/'
                   'fibreB_series0103_scigarn_refexp50.dat',refindex=0)
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
cache = CacheObject()        
#%% COMPARISON FROM WAVELENGTH SOLUTION 
wave_A=cache['wave_A',seriesA.rv_from_wavesol]   
#wave_A=seriesA.rv_from_wavesol()
wave_B=cache['wave_B',seriesB.rv_from_wavesol]   
#wave_B=seriesB.rv_from_wavesol()
plotter_wave=wave_A.plot(label='A')
wave_B.plot(plotter=plotter_wave,label='B')
(wave_B-wave_A).plot(plotter=plotter_wave,label='B-A')
plotter_wave.figure.suptitle("WAVELENGTH SOLUTIONS")
#%% COMPARISON FROM LINE POSITIONS, INTERPOLATE FREQUENCY
#lines_A=seriesA.interpolate(use='freq')
lines_A=cache['lines_A',seriesA.interpolate,dict(use='freq')]
#lines_B=seriesB.interpolate(use='freq')
lines_B=cache['lines_B',seriesB.interpolate,dict(use='freq')]
plotter_lines_freq=lines_A.plot(label='A')
lines_B.plot(plotter=plotter_lines_freq,label='B')
(lines_B-lines_A).plot(plotter=plotter_lines_freq,label='B-A')
plotter_lines_freq.figure.suptitle('LINES:FREQ')
#%% COMPARISON FROM LINE POSITIONS, INTERPOLATE CENTERS
#lines_cen_A=seriesA.interpolate(use='centre')
#lines_cen_B=seriesB.interpolate(use='centre')
lines_cen_A=cache['lines_cen_A',seriesA.interpolate,dict(use='centre')]
lines_cen_B=cache['lines_cen_B',seriesB.interpolate,dict(use='centre')]
plotter_lines_cen=lines_cen_A.plot(label='A')
lines_cen_B.plot(plotter=plotter_lines_cen,label='B')
(lines_cen_B-lines_cen_A).plot(plotter=plotter_lines_cen,label='B-A')
plotter_lines_cen.figure.suptitle('LINES:CENTRE')
#%% COMPARISON FROM LINE POSITIONS AND COEFFICIENTS
#coeff_A=seriesA.coefficients(version=500)
#coeff_B=seriesB.coefficients(version=500)
coeff_A=cache['coeff_A',seriesA.coefficients,dict(version=500)]
coeff_B=cache['coeff_B',seriesB.coefficients,dict(version=500)]
plotter_coeffs=coeff_A.plot(label='A')
coeff_B.plot(plotter=plotter_coeffs,label='B')
(coeff_B-coeff_A).plot(plotter=plotter_coeffs,label='B-A')
plotter_coeffs.figure.suptitle("COEFFICIENTS")

#%% COMPARISON OF METHODS, BOTH FIBRES INDIVIDUALLY
series_dict = [seriesA,seriesB]
for fibre,series in zip(['A','B'],series_dict):
    wave=cache['wave_{}'.format(fibre),series.rv_from_wavesol]   
    lines=cache['lines_{}'.format(fibre),series.interpolate,dict(use='freq')]
    lines_cen=cache['lines_cen_{}'.format(fibre),series.interpolate,dict(use='centre')]
    coeff=cache['coeff_{}'.format(fibre),series.coefficients,dict(version=500)]
    
    plotter=plot.Figure(2,sharex=True,ratios=[3,1])
    plotter.fig.suptitle('Fibre {}'.format(fibre))
    wave.plot(plotter=plotter,label='wave')
    lines.plot(plotter=plotter,label='lines:freq')
    lines_cen.plot(plotter=plotter,label='lines:centre')
    coeff.plot(plotter=plotter,label='coeff')
    plotter.axes[1].axhline(0,lw=0.7,c='k',ls=':')
    plotter.axes[1].plot((lines-wave).values['rv'],marker='o',ms=3,c='C1',
                  lw=0.5,label='lines:freq-wave')
    plotter.axes[1].plot((lines_cen-wave).values['rv'],marker='o',ms=3,c='C2',
                  lw=0.5,label='lines:centre-wave')
    plotter.axes[1].plot((coeff-wave).values['rv'],marker='o',ms=3,c='C3',
                  lw=0.5,label='coeff-wave')
#%% COMPARISON OF METHODS, B-A
    
