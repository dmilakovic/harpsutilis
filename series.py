#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:37:22 2018

@author: dmilakov
"""
#import matplotlib
#matplotlib.use('GTKAgg')

from   harps.core import np
import harps.functions as hf
import harps.fit as fit
import harps.compare as compare
import harps.containers as container
from   harps.constants import c
import harps.io as io
import scipy.stats as stats
from   harps.plotter import SpectrumPlotter
import harps.cti as cti
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
        fl = []
        pn = []
        numfiles = len(self)
        for i,filepath in enumerate(self._outlist):
            #print("{0:03d}/{1:03d}".format(i,len(self)))
            hf.update_progress(i/(numfiles-1))
            lines, wavesol = io.read_outfile(filepath,self._version)
            #header  = io.read_outfile_header(filepath,0)
            fluxes  = np.array(io.read_fluxord(filepath))
            noise   = 299792458./(np.sqrt(np.sum(np.power(lines['noise']/299792458.,-2))))
            ls.append(lines)
            ws.append(wavesol)
            fl.append(fluxes)
            pn.append(noise)
            dt.append(hf.basename_to_datetime(filepath))
            
        self._wavesols  = np.array(ws)
        self._lines     = np.array(ls)
        self._datetimes = np.array(dt)
        self._fluxes    = np.array(fl) 
        self._noises    = np.array(pn)
            
        return
    
    def rv_from_wavesol(self,exposures=None,orders=None,pixels=None,
                        plot2d=False):
        wavesols = self._wavesols
        data     = container.radial_velocity(len(self))
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
        print(exposures,orders,pixels)
        wavesol2d  = wavesols[exposures,orders,pixels]
        waveref2d  = wavesol2d[self._refindex]
        # RV shift in pixel values
        wavediff2d = (waveref2d - wavesol2d)/waveref2d * c
        #return wavediff2d
        datetimes = self._datetimes[exposures]
        # number of lines
        lines     = self._lines[exposures]
        fluxes    = self._fluxes[exposures,orders]
        noises    = self._noises[exposures]
        for j,i,dt in zip(np.arange(len(wavediff2d)),idx,datetimes):
            hf.update_progress(j/(len(wavediff2d)-1))
            clipped, low, upp = stats.sigmaclip(wavediff2d[j])#.clipped
            
            average_rv  = np.average(clipped)
            
            data[i]['shift']    = average_rv
            data[i]['datetime'] = dt
            data[i]['noise']    = noises[j]#np.std(clipped)/np.sqrt(np.size(clipped))
            data[i]['flux']     = np.sum(fluxes[j])/len(lines[j])
            
            #print("{0:5d}{1:10.3f}{2:10.3f}".format(i,results[i]['shift'], upp))
            if plot2d==True:
                fig,ax=hf.figure(1)
                fig.suptitle("Exposure {}".format(i))
                ax0 = ax[0].imshow(wavediff2d[j],aspect='auto',vmin=-40,vmax=40)
                fig.colorbar(ax0)
        self._cache['wavesol']=RV(data)
        return RV(data)
    def coefficients(self,range=None,version=None,**kwargs):
        
        data    = container.radial_velocity(len(self))
        if range is not None:
            selection = slice(*range)
            idx = np.arange(selection.start,selection.stop,selection.step)
        else:
            selection = slice(None)
            idx = np.arange(len(self))
        lines     = self._lines[selection]
        datetimes = self._datetimes[selection]
        #results   = data._values[selection]
        
        reflinelist = lines[self._refindex]
        version     = version if version is not None else self._version
        coeffs      = fit.dispersion(reflinelist,version,'gauss')
        for j,i,l,dt in zip(np.arange(len(lines)),idx,lines,datetimes):
            data[j]['datetime'] = dt
            #reflines = lines[j-1]
            linelist = lines[j]
            rv, noise = compare.from_coefficients(linelist,coeffs,
                                                  **kwargs)
            
            data[j]['shift'] = rv
            data[j]['noise'] = noise
            print(message(i,len(self),rv,noise))
        
        self._cache['coefficients']=RV(data)
        return RV(data)
    def interpolate(self,use='freq',range=None,**kwargs):
        
        data    = container.radial_velocity(len(self))
        if range is not None:
            selection = slice(*range)
            idx = np.arange(selection.start,selection.stop,selection.step)
        else:
            selection = slice(None)
            idx = np.arange(len(self))
        lines     = self._lines[selection]
        datetimes = self._datetimes[selection]
        results   = data[selection]
        
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
        
        self._cache['lines']=RV(results)
        return RV(results)
    
#class RV(object):
#    def __init__(self, nelem):
#        self._nelem   = nelem
#        self._values = container.radial_velocity(nelem)
class RV(object):
    def __init__(self,values):
        self._values = values
        self._nelem  = np.shape(values)[0]
    def __str__(self):
        print(self._results)
        return "{0:=>60s}".format("")
    def __len__(self):
        return self._nelem
    
    def __add__(self,item):
        id1  = _intersect(self._values,item.values)
        id2  = _intersect(item.values,self._values)
        arr1 = self._values[id1]
        arr2 = item.values[id2]
        data       = container.radial_velocity(len(id1))
        
        data['shift'] = arr1['shift'] + arr2['shift']
        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        data['datetime'] = arr1['datetime']
        data['flux']  = arr1['flux']#+arr2['flux'])/2.
        return RV(data)
    def __sub__(self,item):
        id1  = _intersect(self._values,item.values)
        id2  = _intersect(item.values,self._values)
        arr1 = self._values[id1]
        arr2 = item.values[id2]
        
        data       = container.radial_velocity(len(id1))
        
        data['shift'] = arr1['shift'] - arr2['shift']
        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        data['datetime'] = arr1['datetime']
        data['flux']  = (arr1['flux']+arr2['flux'])/2.
        return RV(data)
    def __radd__(self,item):
        
        return self.__add__(self,item)
    def __iadd__(self,item):
        
        id1  = _intersect(self._values,item.values)
        id2  = _intersect(item.values,self._values)
        arr1 = self._values[id1]
        arr2 = item.values[id2]
        
        data       = container.radial_velocity(len(id1))
        data['shift'] = arr1['shift'] + arr2['shift']
        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        data['datetime'] = arr1['datetime']
        data['flux']  = (arr1['flux']+arr2['flux'])/2.
        return RV(data)
    def __mul__(self,item):
        
        id1  = _intersect(self._values,item.values)
        id2  = _intersect(item.values,self._values)
        arr1 = self._values[id1]
        arr2 = item.values[id2]
        
        data       = container.radial_velocity(len(id1))
        data['shift'] = arr1['shift'] * arr2['shift']
        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        data['datetime'] = arr1['datetime']
        data['flux']  = arr1['flux']#+arr2['flux'])/2.
        return RV(data)
    def __rmul__(self,item):
        id1  = _intersect(self._values,item.values)
        id2  = _intersect(item.values,self._values)
        arr1 = self._values[id1]
        arr2 = item.values[id2]
        
        data       = container.radial_velocity(len(id1))
        data['shift'] = arr1['shift'] * arr2['shift']
        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        data['datetime'] = arr1['datetime']
        data['flux']  = arr1['flux']#+arr2['flux'])/2.
        return RV(data)
    def __imul__(self,item):
        
        idx  = self._intersect(item)
        arr1 = self._values[idx]
        arr2 = item.values[idx]
        
        data       = container.radial_velocity(len(idx))
        data['shift'] = arr1['shift'] * arr2['shift']
        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        data['datetime'] = arr1['datetime'][idx]
        data['flux']  = arr1['flux']#+arr2['flux'])/2.
        return RV(data)
    def __getitem__(self,key):
        return RV(self._values[key])
    def __setitem__(self,key,val):
        self._values[key] = val
        return
    @property
    def values(self):
        return self._values
    @property
    def shape(self):
        return np.shape(self.values)
    
    def copy(self):
        return RV(np.copy(self.values))
    
    def _get_values(self,key):
        if key=='datetime':
            return self._values[key].view('i8')
        else:
            return self._values[key]
    def groupby_bins(self,key,bins):
        values0 = self.values[key]
        values  = self._get_values(key)
        if key=='datetime':
            bins = bins.view('i8')
        binned = np.digitize(values,bins)
        groups = {str(bin):self[np.where(binned==bin)] \
                      for bin in np.unique(binned)}
        return groups
    def min(self,key):
        value = np.min(self._get_values(key))
        if key=='datetime':
            return np.datetime64(int(value),'s')
        else:
            return value
    def max(self,key):
        value = np.max(self._get_values(key))
        if key=='datetime':
            return np.datetime64(int(value),'s')
        else:
            return value
    def mean(self,key):
        values = self._get_values(key)
        mean   = np.mean(values)
        if key == 'datetime':
            mean = np.datetime64(int(mean),'s')
        return mean
    def std(self,key):
        values = self._get_values(key)
        std    = np.std(values)
        if key == 'datetime':
            std = np.timedelta64(int(std),'s')
        return std
    
    def correct_cti(self,pars=None,sigma=None):
        values = self.values
        
        flux   = values['flux']
        corr, noise   = cti.model(flux,pars,sigma)
        
        values['shift'] = values['shift']+corr
        return values
    def correct_time(self,pars,datetime,copy=False):
        def model(x,pars):
            A, B = pars
            return A + B * x
        
        if copy:
            values = np.copy(self.values)
        else:
            values = self.values
        timedelta = (values['datetime'] - datetime)/np.timedelta64(1,'s')
        values['shift'] = values['shift'] - model(timedelta,pars)
        if copy:
            return RV(values)
        else:
            return self
    def plot(self,scale='sequence',plotter=None,axnum=0,**kwargs):
        ls = kwargs.pop('ls','-')
        lw = kwargs.pop('lw',0.8)
        m  = kwargs.pop('marker','o')
        ms = kwargs.pop('ms',2)
        a  = kwargs.pop('alpha',1.)
        plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
        
        axes    = plotter.axes
        results = self._values
        
        if scale == 'sequence':
            x = np.arange(self._nelem)
        elif scale == 'flux':
            x = results['flux']
        else:
            x = (results['datetime']-results['datetime'][0]).astype(np.float64)
        y     = results['shift']
        yerr  = results['noise']
        label = kwargs.pop('label',None)
        axes[axnum].errorbar(x,y,yerr,ls=ls,lw=lw,marker=m,
                         ms=ms,alpha=a,label=label)
        axes[axnum].axhline(0,ls=':',lw=1,c='k')
        axes[axnum].set_xlabel(scale.capitalize())
        axes[axnum].set_ylabel("RV [m/s]")
        axes[axnum].legend()
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