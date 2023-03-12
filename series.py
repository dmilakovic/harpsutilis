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
import harps.inout as io
import scipy.stats as stats
from   harps.plotter import SpectrumPlotter
import harps.cti as cti
from   harps.lines import select
import harps.wavesol as ws
# =============================================================================
#
#                               S E R I E S
#
# =============================================================================

class Series(object):
    def __init__(self,outfile,fibre,refindex=0,version=701,read=True):
        self._outfile  = outfile
        self._outlist  = np.sort(io.read_textfile(outfile))
        self._refindex = refindex
        self._version  = version
        #self._sigma    = sigma
        self._fibre    = fibre
        if read:
            self._read_from_file(version)
        #self._cache = {}
        #self._results = container.radial_velocity(len(self))
    def __len__(self):
        return len(self._outlist)
    
    def __get__(self,item):
        try:
            data = self._cache[item]
        except:
            if item == 'wavesol_gauss':
                data = self.rv_from_wavesol()
            elif item == 'linelist':
                data = self.rv_from_lines()
        finally:
            pass
        return data
#    @property 
#    def sigma(self):
#        return self._sigma
    @property
    def version(self):
        return self._version
    def _read_from_file(self,version):
        extensions = [
                      'datetime',
                      'linelist',
                      'flux',
                      'wavesol_gauss',
#                      'wavesol_lsf',
                      'coeff_gauss',
#                      'coeff_lsf',
                      'noise']
        print("Reading version {}".format(version))
        data, numfiles = io.mread_outfile(self._outfile,extensions,
                                          version)
        for key, val in data.items():
            setattr(self,"{}".format(key),val)
        
        return

    def get(self,fittype,exposures=None,orders=None,):
        '''
        Returns the wavesols, lines, fluxes, noises, datetimes for a selection
        of exposures and orders
        '''
        wavesols0  = getattr(self,"wavesol_{}".format(fittype))
        lines0     = self.linelist
        fluxes0    = self.flux
        noises0    = self.noise
        datetimes0 = self.datetime
        
        if exposures is not None:
            exposures = slice(*exposures)
            #idx = np.arange(exposures.start,exposures.stop,exposures.step)
        else:
            exposures = slice(None)
            #idx = np.arange(len(self))
        
        if orders is not None:
            orders = slice(*orders)
            lines  = np.array([select_order(l,orders) \
                               for l in lines0[exposures]])
        else:
            orders = slice(41,None,None)
            lines  = lines0[exposures]
        wavesols  = wavesols0[exposures,orders]
        fluxes    = fluxes0[exposures,orders]
        noises    = noises0[exposures]
        datetimes = datetimes0[exposures]
        
        return wavesols, lines, fluxes, noises, datetimes
    def get_idx(self,exposures):
        if exposures is not None:
            exposures = slice(*exposures)
            idx = np.arange(exposures.start,exposures.stop,exposures.step)
        else:
            exposures = slice(None)
            idx = np.arange(len(self))
        return idx
    def wavesol(self,fittype,sigma,exposures=None,orders=None,pixels=None,
                verbose=False,plot2d=False,**kwargs):
        #sigma    = sigma if sigma is not None else self.sigma
        data     = container.radial_velocity(len(self))
        wavesols,lines,fluxes,noises,datetimes = self.get(fittype,exposures,orders)
        # range in pixels
        pixels = slice(*pixels) if pixels is not None else slice(None)
        
        wavesol2d  = wavesols[pixels]
        waveref2d  = wavesol2d[self._refindex]
        # RV shift in pixel values
        #wavediff2d = (waveref2d - wavesol2d)/waveref2d * c
        idx        = self.get_idx(exposures)
        for j,i,dt in zip(np.arange(len(wavesol2d)),idx,datetimes):
            if i==self._refindex:
                rv    = 0.
                noise = 0.
            else:
                rv, noise = compare.wavesolutions(waveref2d,wavesol2d[j],
                                                    sigma=sigma,**kwargs)
            
            data[i]['shift']    = rv
            data[i]['datetime'] = dt
            data[i]['noise']    = noise
            data[i]['flux']     = np.sum(fluxes[j])/len(lines[j])
            
            if verbose:
                print(message(i,len(self),rv,noise))
            else:
                hf.update_progress((j+1)/len(wavesol2d))
            if plot2d==True:
                fig,ax=hf.figure(1)
                fig.suptitle("Exposure {}".format(i))
                ax0 = ax[0].imshow(wavesol2d[j],aspect='auto',vmin=-40,vmax=40)
                fig.colorbar(ax0)
        data['fibre'] = self._fibre
#        self._cache['wavesol']=RV(data)
        return RV(data)
    def coefficients(self,fittype,version,sigma,exposures=None,orders=None,
                     coeffs=None,verbose=False,**kwargs):
        #sigma       = sigma if sigma is not None else self.sigma
        data        = container.radial_velocity(len(self))
        wavesols,lines,fluxes,noises,datetimes = self.get(fittype,exposures,orders)
        idx         = self.get_idx(exposures)
        reflinelist = self.linelist[self._refindex]
        #version     = version if version is not None else self.version
        if coeffs is None:
            coeffs  = fit.dispersion(reflinelist,version,fittype)
        for j,i,l,dt in zip(np.arange(len(lines)),idx,lines,datetimes):
            data[j]['datetime'] = dt
            #reflines = lines[j-1]
            linelist  = lines[j]
            rv, noise = compare.from_coefficients(linelist,coeffs,
                                                  fittype=fittype,
                                                  sigma=sigma,
                                                  **kwargs)
            data[i]['flux']  = np.sum(fluxes[j])/len(lines[j])
            data[j]['shift'] = rv
            data[j]['noise'] = noise
            if verbose:
                print(message(i,len(self),rv,noise))
            else:
                hf.update_progress((i+1)/len(lines))
        data['fibre'] = self._fibre
        #self._cache['coefficients']=RV(data)
        return RV(data)
    def interpolate(self,fittype,sigma,use='freq',exposures=None,orders=None,
                    verbose=False,**kwargs):
        #sigma   = sigma if sigma is not None else self.sigma
        data    = container.radial_velocity(len(self))
        wavesols,lines,fluxes,noises,datetimes = self.get(fittype,exposures,orders)
        idx  = self.get_idx(exposures)
        
        reflinelist = self.linelist[self._refindex]
        
        for j,i,l,dt in zip(np.arange(len(lines)),idx,lines,datetimes):
            data[j]['datetime'] = dt
            if i == self._refindex:
                continue
            linelist = lines[j]
            rv, noise = compare.interpolate(reflinelist,linelist,
                                            fittype=fittype,
                                            sigma=sigma,
                                            use=use,**kwargs)
            data[j]['flux']  = np.sum(fluxes[j])/len(lines[j])
            data[j]['shift'] = rv
            data[j]['noise'] = noise
            if verbose:
                print(message(i,len(self),rv,noise))
            else:
                hf.update_progress((i+1)/len(lines))
        data['fibre'] = self._fibre
        #self._cache['lines']=RV(data)
        return RV(data)
def get_idx(self,exposures):
    idx = np.arange(exposures.start,exposures.stop,exposures.step)
    return idx

def cut(exposures=None,orders=None,pixels=None):
    exposures = slice(*exposures) if exposures is not None else slice(None)
    orders    = slice(*orders) if orders is not None else slice(41,None,None)
    pixels    = slice(*pixels) if pixels is not None else slice(None)
    return exposures,orders,pixels
def wavesol(wavesols,sigma,refindex=0,datetimes=None,fluxes=None,lines=None,
            exposures=None,orders=None,pixels=None,verbose=False,fibre=None,
            plot2d=False,**kwargs):
    
    exposures, orders, pixels = cut(exposures,orders,pixels)
    data     = container.radial_velocity(np.shape(wavesols)[0])
    wavesol2d  = wavesols[exposures,orders,pixels]
    waveref2d  = wavesol2d[refindex]
    
    for i,expwavesol in enumerate(wavesol2d):
        if i==refindex:
            rv    = 0.
            noise = 0.
        else:
            rv, noise = compare.wavesolutions(waveref2d,expwavesol,
                                                sigma=sigma,**kwargs)
        
        data[i]['shift']    = rv  
        data[i]['noise']    = noise
        if datetimes is not None:
            dt='{0:4}-{1:02}-{2:02}T{3:02}:{4:02}:{5:02}'.format(*datetimes[i])
            data[i]['datetime'] = dt
        if fluxes is not None and lines is not None:
            data[i]['flux']     = np.sum(fluxes[i])/len(lines[i])
        
        if verbose:
            print(message(i,len(wavesol2d),rv,noise))
        else:
            hf.update_progress((i+1)/len(wavesol2d))
        if plot2d==True:
            fig,ax=hf.figure(1)
            fig.suptitle("Exposure {}".format(i))
            ax0 = ax[0].imshow(wavesol2d[i],aspect='auto',vmin=-40,vmax=40)
            fig.colorbar(ax0)
    if fibre is not None:
        data['fibre'] = fibre
    return RV(data)

    
def interpolate(linelist,fittype,sigma,use,refindex=0,datetimes=None,
                exposures=None,orders=None,fibre=None,verbose=False,**kwargs):
    #sigma   = sigma if sigma is not None else self.sigma
    #wavesols,lines,fluxes,noises,datetimes = self.get(fittype,exposures,orders)
    #idx  = self.get_idx(exposures)
    assert use in ['freq','centre']
    if exposures is not None:
        exposures, orders, pixels = cut(exposures,orders,None)
        idx = get_idx(exposures)
    else:
        idx = np.unique(linelist['exp']) 
    numexp = len(idx)
    data   = container.radial_velocity(numexp)
    reflinelist = select(linelist,dict(exp=refindex))
    
    for i in idx:
        if i == refindex:
            continue
        explinelist = select(linelist,dict(exp=i))
        rv, noise = compare.interpolate(reflinelist,explinelist,
                                        fittype=fittype,
                                        sigma=sigma,
                                        use=use,**kwargs)
        #data[j]['flux']  = np.sum(fluxes[j])/len(lines[j])
        
        data[i]['shift'] = rv
        data[i]['noise'] = noise
        if datetimes is not None:
            dt='{0:4}-{1:02}-{2:02}T{3:02}:{4:02}:{5:02}'.format(*datetimes[i])
            data[i]['datetime'] = dt

        if verbose:
            print(message(i,numexp,rv,noise))
        else:
            hf.update_progress((i+1)/numexp)
    if fibre is not None:
        data['fibre'] = fibre
    #self._cache['lines']=RV(data)
    return RV(data)
class RV(object):
    def __init__(self,values):
        self._values = values
        self._nelem  = np.shape(values)[0]
    def __str__(self):
        print(self._results)
        return "{0:=>60s}".format("")
    def __len__(self):
        return self._nelem
    
#    def __calc__(self,item,operation):
#        
#    
    def __add__(self,item):
        id1  = _intersect(self._values,item.values)
        id2  = _intersect(item.values,self._values)
        arr1 = self._values[id1]
        arr2 = item.values[id2]
        
        fibre1 = arr1['fibre'][0]
        fibre2 = arr2['fibre'][0]
        
        data       = container.radial_velocity(len(id1))
        
        data['shift'] = arr1['shift'] + arr2['shift']
        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        data['datetime'] = arr1['datetime']
        data['flux']  = (arr1['flux']+arr2['flux'])/2.
        data['fibre'] = "{0}+{1}".format(fibre1,fibre2)
        return RV(data)
    def __sub__(self,item):
        id1  = _intersect(self._values,item.values)
        id2  = _intersect(item.values,self._values)
        arr1 = self._values[id1]
        arr2 = item.values[id2]
        
        fibre1 = arr1['fibre'][0]
        fibre2 = arr2['fibre'][0]
        
        data       = container.radial_velocity(len(id1))
        
        data['shift'] = arr1['shift'] - arr2['shift']
        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        data['datetime'] = arr1['datetime']
        data['flux']  = (arr1['flux']+arr2['flux'])/2.
        data['fibre'] = "{0}-{1}".format(fibre1,fibre2)
        return RV(data)
#    def __radd__(self,item):
#        
#        return self.__add__(self,item)
#    def __iadd__(self,item):
#        
#        id1  = _intersect(self._values,item.values)
#        id2  = _intersect(item.values,self._values)
#        arr1 = self._values[id1]
#        arr2 = item.values[id2]
#        
#        data       = container.radial_velocity(len(id1))
#        data['shift'] = arr1['shift'] + arr2['shift']
#        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
#        data['datetime'] = arr1['datetime']
#        data['flux']  = (arr1['flux']+arr2['flux'])/2.
#        data['fibre'] = "{0}+{1}".format(arr1['fibre'],arr2['fibre'])
#        return RV(data)
    def __mul__(self,item):
        
        id1  = _intersect(self._values,item.values)
        id2  = _intersect(item.values,self._values)
        arr1 = self._values[id1]
        arr2 = item.values[id2]
        
        fibre1 = arr1['fibre'][0]
        fibre2 = arr2['fibre'][0]
        
        data       = container.radial_velocity(len(id1))
        
        data['shift'] = arr1['shift'] * arr2['shift']
        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
        data['datetime'] = arr1['datetime']
        data['flux']  = (arr1['flux']+arr2['flux'])/2.
        data['fibre'] = "{0}*{1}".format(fibre1,fibre2)
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
    
    def correct_cti(self,fibre,pars=None,sigma=None,copy=False):
        if copy:
            values = np.copy(self.values)
        else:
            values = self.values
        flux   = values['flux']
        corr, noise   = cti.exp(flux,fibre,pars,sigma)
        
        values['shift'] = values['shift']+corr
        if copy:
            return RV(values)
        else:
            return self
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
    def plot(self,scale='sequence',plotter=None,axnum=0,legend=True,**kwargs):
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
                         ms=ms,alpha=a,label=label,**kwargs)
        axes[axnum].axhline(0,ls=':',lw=1,c='k')
        axes[axnum].set_xlabel(scale.capitalize())
        axes[axnum].set_ylabel("RV [m/s]")
        if legend:
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