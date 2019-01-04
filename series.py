#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:37:22 2018

@author: dmilakov
"""
import matplotlib
matplotlib.use('GTKAgg')

import harps.classes as hc
from   harps.core import np, plt
import harps.functions as hf
import harps.fit as fit
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
    def __init__(self,outfile,refindex=0,version=501):
        self._outfile  = outfile
        self._outlist  = np.sort(io.read_textfile(outfile))
        self._refindex = refindex
        self._version  = version
        
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
    
    def rv_from_wavesol(self,exposures=None,orders=None,pixels=None):
        wavesols = self._wavesols
        data     = RV(len(self))
        results  = data._values
        # range in exposurs
        if exposures is not None:
            exposures = slice(*exposures) 
            idx  = np.arange(exposures.start,exposures.stop,exposures.step)
        else:
            exposures = slice(None)
            idx  = np.arange(len(self))
        # take only orders 43 and up
        orders = slice(*orders) if orders is not None else slice(43,None,None)
        # range in pixels
        pixels = slice(*pixels) if pixels is not None else slice(None)
        
        wavesol2d  = wavesols[exposures,orders,pixels]
        waveref2d  = wavesol2d[self._refindex]
        # RV shift in pixel values
        wavediff2d = (waveref2d - wavesol2d)/waveref2d * c
        #return wavediff2d
        datetimes = self._datetimes[exposures]
        
        for j,i,dt in zip(np.arange(len(wavediff2d)),idx,datetimes):
           
            clipped, low, upp = stats.sigmaclip(wavediff2d[j])#.clipped
            
            average_rv  = np.average(clipped)
            
            results[i]['shift'] = average_rv
            results[i]['datetime'] = dt
            results[i]['noise'] = 0.0     
            print("{0:5d}{1:10.3f}{2:10.3f}".format(i,average_rv, upp))
#            if plot2d==True:
#                fig,ax=hf.figure(1)
#                ax0 = ax[0].imshow(wavediff2d[i],aspect='auto',vmin=-40,vmax=40)
#                fig.colorbar(ax0)
        self._cache['wavesol']=data
        return data
    def coefficients(self,range=None,version=None,**kwargs):
        
        data    = RV(len(self))
        if range is not None:
            selection = slice(*range)
            idx = np.arange(selection.start,selection.stop,selection.step)
        else:
            selection = slice(None)
            idx = np.arange(len(self))
        lines     = self._lines[selection]
        datetimes = self._datetimes[selection]
        results   = data._values[selection]
        
        reflinelist = lines[self._refindex]
        version     = version if version is not None else self._version
        coeffs      = fit.dispersion(reflinelist,version,'gauss')
        for j,i,l,dt in zip(np.arange(len(lines)),idx,lines,datetimes):
            results[j]['datetime'] = dt
            #reflines = lines[j-1]
            linelist = lines[j]
            rv, noise = compare.from_coefficients(linelist,coeffs,
                                                  **kwargs)
            
            results[j]['shift'] = rv
            results[j]['noise'] = noise
            print(message(i,len(self),rv,noise))
        
        self._cache['coefficients']=data
        return data
    def interpolate(self,use='freq',range=None,**kwargs):
        
        data    = RV(len(self))
        if range is not None:
            selection = slice(*range)
            idx = np.arange(selection.start,selection.stop,selection.step)
        else:
            selection = slice(None)
            idx = np.arange(len(self))
        lines     = self._lines[selection]
        datetimes = self._datetimes[selection]
        results   = data._values[selection]
        
        reflinelist = lines[self._refindex]
        
        for j,i,l,dt in zip(np.arange(len(lines)),idx,lines,datetimes):
            results[j]['datetime'] = dt
            if i == self._refindex:
                continue
            linelist = lines[j]
            rv, noise = compare.interpolate(reflinelist,linelist,
                                              use=use,**kwargs)
            
            results[j]['shift'] = rv
            results[j]['noise'] = noise
            print(message(i,len(self),rv,noise))
        
        self._cache['lines']=data
        return data
    
class RV(object):
    def __init__(self, nelem):
        self._nelem   = nelem
        self._values = container.radial_velocity(nelem)
    def __str__(self):
        print(self._results)
        return "{0:=>60s}".format("")
    def __len__(self):
        return len(self._nelem)
    
    def __add__(self,item):
        id1  = _intersect(self._values,item)
        id2  = _intersect(item,self._values)
        arr1 = self._results[id1]
        arr2 = item[id2]
        
        data       = RV(len(id1))
        result     = data._values
        result['shift'] = arr1['shift'] + arr2['shift']
        result['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        result['datetime'] = arr1['datetime']
        
        return data
    def __sub__(self,item):
        id1  = _intersect(self._values,item)
        id2  = _intersect(item,self._values)
        arr1 = self._values[id1]
        arr2 = item[id2]
        
        data       = RV(len(id1))
        result     = data._values
        result['shift'] = arr1['shift'] - arr2['shift']
        result['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        result['datetime'] = arr1['datetime']
        
        return data
    def __radd__(self,item):
        
        return self.__add__(self,item)
    def __iadd__(self,item):
        
        idx  = self._intersect(item)
        arr1 = self._values[idx]
        arr2 = item[idx]
        
        result       = container.radial_velocity(len(idx))
        result['shift'] = arr1['shift'] + arr2['shift']
        result['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        result['datetime'] = arr1['datetime'][idx]
        
        return result
    def __mul__(self,item):
        
        idx  = self._intersect(item)
        arr1 = self._values[idx]
        arr2 = item[idx]
        
        result       = container.radial_velocity(len(idx))
        result['shift'] = arr1['shift'] * arr2['shift']
        result['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        result['datetime'] = arr1['datetime'][idx]
        
        return result
    def __rmul__(self,item):
        
        idx  = self._intersect(item)
        arr1 = self._values[idx]
        arr2 = item[idx]
        
        result       = container.radial_velocity(len(idx))
        result['shift'] = arr1['shift'] * arr2['shift']
        result['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        result['datetime'] = arr1['datetime'][idx]
        
        return result
    def __imul__(self,item):
        
        idx  = self._intersect(item)
        arr1 = self._values[idx]
        arr2 = item[idx]
        
        result       = container.radial_velocity(len(idx))
        result['shift'] = arr1['shift'] * arr2['shift']
        result['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        result['datetime'] = arr1['datetime'][idx]
        
        return result
    def __getitem__(self,key):
        return self._values[key]
    def __setitem__(self,key,val):
        self._values[key] = val
        return
    @property
    def values(self):
        return self._values
    
    def plot(self,scale='sequence',plotter=None,**kwargs):
        
        plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
        
        axes    = plotter.axes
        results = self._values
        
        if scale == 'sequence':
            x = np.arange(self._nelem)
        else:
            x = (results['datetime']-results['datetime'][0]).astype(np.float64)
        y     = results['shift']
        yerr  = results['noise']
        label = kwargs.pop('label',None)
        axes[0].errorbar(x,y,yerr,lw=0.8,marker='o',ms=2,label=label)
        axes[0].axhline(0,ls=':',lw=1,c='k')
        axes[0].set_xlabel(scale.capitalize())
        axes[0].set_ylabel("RV [m/s]")
        axes[0].legend()
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