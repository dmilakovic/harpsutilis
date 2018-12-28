#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:37:22 2018

@author: dmilakov
"""

import harps.classes as hc
from   harps.core import np, plt
import harps.functions as hf
import harps.settings as hs
import harps.compare as compare
import harps.containers as container
from   harps.constants import c
import harps.io as io
import scipy.stats as stats
from   harps.plotter import SpectrumPlotter

# =============================================================================
#
#                               S E R I E S
#
# =============================================================================

class Series(object):
    def __init__(self,outfile,refindex=0,version=500):
        self._outfile  = outfile
        self._outlist  = np.sort(io.read_textfile(outfile))
        self._refindex = refindex
        self._version  = 500
        
        self._read_from_file()
        self._cache = {}
        self._results = container.radial_velocity(len(self))
    def __len__(self):
        return len(self._outlist)
    
    def __get__(self,item):
        try:
            data = self._cache[item]
        except:
            if item == 'wavesol':
                data = self.rv_from_wavesol()
            elif item == 'lines':
                data = self.rv_from_lines()
        finally:
            pass
        return data
    def _read_from_file(self):
        ws = []
        ls = []
        dt = []
        
        for i,filepath in enumerate(self._outlist):
            #print("{0:03d}/{1:03d}".format(i,len(self)))
            lines, wavesol = io.read_outfile(filepath,self._version)
            ls.append(lines)
            ws.append(wavesol)
            dt.append(hf.basename_to_datetime(filepath))
            
        self._wavesols  = np.array(ws)
        self._lines     = np.array(ls)
        self._datetimes = np.array(dt)
            
        return
    
    def rv_from_wavesol(self):
        wavesols = self._wavesols
        data     = RV(len(self))
        results  = data._results
        # take only orders 43 and up
        wavesol2d  = wavesols[:,43:,:]
        waveref2d  = wavesol2d[self._refindex]
        # RV shift in pixel values
        wavediff2d = (waveref2d - wavesol2d)/waveref2d * c
        
        for i,dt in enumerate(self._datetimes):
           
            clipped     = stats.sigmaclip(wavediff2d[i]).clipped
            average_rv  = np.average(clipped)
            
            results[i]['rv'] = average_rv
            results[i]['datetime'] = dt
            results[i]['pn'] = 0.0            
#            if plot2d==True:
#                fig,ax=hf.figure(1)
#                ax0 = ax[0].imshow(wavediff2d[i],aspect='auto',vmin=-40,vmax=40)
#                fig.colorbar(ax0)
        self._cache['wavesol']=data
        return data
    
    def rv_from_lines(self):
        lines   = self._lines
        data    = RV(len(self))
        results = data._results
        reflinelist = lines[self._refindex]
        for i,dt in enumerate(self._datetimes):
            results[i]['datetime'] = dt
            if i == self._refindex:
                continue
            linelist = lines[i]
            rv, noise = compare.two_linelists(reflinelist,linelist)
            
            results[i]['rv'] = rv
            results[i]['pn'] = noise
            print(message(i,len(self),rv,noise))
        
        self._cache['lines']=data
        return data
    
class RV(object):
    def __init__(self, nelem):
        self._nelem   = nelem
        self._results = container.radial_velocity(nelem)
    def __str__(self):
        print(self._results)
        return "{0:=>60s}".format("")
    def __len__(self):
        return len(self._nelem)
    
    def __add__(self,item):
        id1  = _intersect(self._results,item)
        id2  = _intersect(item,self._results)
        arr1 = self._results[id1]
        arr2 = item[id2]
        
        data       = RV(len(id1))
        result     = data._results
        result['rv'] = arr1['rv'] + arr2['rv']
        result['pn'] = np.sqrt(arr1['pn']**2 + arr2['pn']**2)
        result['datetime'] = arr1['datetime']
        
        return data
    def __sub__(self,item):
        id1  = _intersect(self._results,item)
        id2  = _intersect(item,self._results)
        arr1 = self._results[id1]
        arr2 = item[id2]
        
        data       = RV(len(id1))
        result     = data._results
        result['rv'] = arr1['rv'] - arr2['rv']
        result['pn'] = np.sqrt(arr1['pn']**2 + arr2['pn']**2)
        result['datetime'] = arr1['datetime']
        
        return data
    def __radd__(self,item):
        
        return self.__add__(self,item)
    def __iadd__(self,item):
        
        idx  = self._intersect(item)
        arr1 = self._results[idx]
        arr2 = item[idx]
        
        result       = container.radial_velocity(len(idx))
        result['rv'] = arr1['rv'] + arr2['rv']
        result['pn'] = np.sqrt(arr1['pn']**2 + arr2['pn']**2)
        result['datetime'] = arr1['datetime'][idx]
        
        return result
    def __mul__(self,item):
        
        idx  = self._intersect(item)
        arr1 = self._results[idx]
        arr2 = item[idx]
        
        result       = container.radial_velocity(len(idx))
        result['rv'] = arr1['rv'] * arr2['rv']
        result['pn'] = np.sqrt(arr1['pn']**2 + arr2['pn']**2)
        result['datetime'] = arr1['datetime'][idx]
        
        return result
    def __rmul__(self,item):
        
        idx  = self._intersect(item)
        arr1 = self._results[idx]
        arr2 = item[idx]
        
        result       = container.radial_velocity(len(idx))
        result['rv'] = arr1['rv'] * arr2['rv']
        result['pn'] = np.sqrt(arr1['pn']**2 + arr2['pn']**2)
        result['datetime'] = arr1['datetime'][idx]
        
        return result
    def __imul__(self,item):
        
        idx  = self._intersect(item)
        arr1 = self._results[idx]
        arr2 = item[idx]
        
        result       = container.radial_velocity(len(idx))
        result['rv'] = arr1['rv'] * arr2['rv']
        result['pn'] = np.sqrt(arr1['pn']**2 + arr2['pn']**2)
        result['datetime'] = arr1['datetime'][idx]
        
        return result
    def __getitem__(self,key):
        return self._results[key]
    def __setitem__(self,key,val):
        self._results[key] = val
        return
    @property
    def result(self):
        return self._results
    
    def plot(self,scale='sequence',plotter=None,**kwargs):
        
        plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
        
        axes    = plotter.axes
        results = self._results
        
        if scale == 'sequence':
            x = np.arange(self._nelem)
        else:
            x = (results['datetime']-results['datetime'][0]).astype(np.float64)
        y     = results['rv']
        yerr  = results['pn']
        axes[0].errorbar(x,y,yerr,lw=0.8,marker='o',ms=2)
        axes[0].axhline(0,ls=':',lw=1,c='k')
        axes[0].set_xlabel(scale.capitalize())
        axes[0].set_ylabel("RV [m/s/]")
        return plotter
def _intersect(array1,array2):
        ''' Returns the index of data points with the same datetime stamp '''
        dt1 = array1['datetime']
        dt2 = set(array2['datetime'])
        idx = np.array([i for i, val in enumerate(dt1) if val in dt2])
        return idx
def message(i,total,rv,noise):
    mess = ("EXP={exp:<5d}/{tot:<5d}".format(exp=i,tot=total) + \
            "{t1:>8s}{rv:>8.3f}".format(t1="RV =",rv=rv) + \
            "{t2:>8s}{pn:>7.3f}".format(t2="PN =",pn=noise))
    return mess