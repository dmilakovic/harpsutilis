#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:55:42 2019

@author: dmilakov
"""
import harps.compare as compare
import harps.containers as container
from   harps.core import np, os
import harps.cti as cti
import harps.fit as fit
import harps.functions as hf
import harps.inout as io
from   harps.lines import select, Linelist
from   harps.plotter import SpectrumPlotter, Figure2
from   harps.settings import __version__ as hs_version
import harps.wavesol as ws

from   fitsio import FITS, FITSHDR
from   numpy.lib.recfunctions import append_fields
import multiprocessing as mp
import itertools
import time
from   pathos.multiprocessing import ProcessPool
import scipy.stats as stats
import matplotlib.ticker as ticker
import matplotlib.dates as mdates


methods = ['wavesol','coeff','freq','cent']

class Series(object):
    def __init__(self,filepath,overwrite=False,ver=None):
        
        self._infile = filepath
        self._hdu_path = io.get_fits_path('series',filepath,ver)
        primhead = self.return_header('primary')
        with FITS(self._hdu_path,'rw',clobber=overwrite) as hdu:
            hdu[0].write_keys(primhead)
            
    def __getitem__(self,item):
        return
    def __call__(self,item):
        return 
    
    def read(self,extension,version=None,write=False,avflux=True,*args,**kwargs):
        """ 
        
        Read the data from individual FITS files in the text file provided.
        
        
        Args:
        ----
        extension (str, list of str) : extensions to be read from files
        version   (int) : version number, 3 digit (PGS)
                        P = polynomial order
                        G = gaps
                        S = segment
        
        Returns:
        -------
        data (array_like) : requested data
            
        """
        #print(extension,version)
        version       = hf.item_to_version(version)
        #print("updated version", version)
        orders        = kwargs.pop('order',None)
        data,numfiles = io.mread_outfile(self._infile,extension,version,
                                         avflux=avflux,order=orders)
        for key,val in data.items():
            print(key)
            if key =='datetime':
                val = hf.datetime_to_record(val)
            
            elif key not in ['wavesol_gauss','wavesol_lsf',
                             'noise','flux','model_gauss','model_lsf']:
                # stack, keep the exposure number (0-indexed)
                exposures=np.hstack([np.full(len(ls),i) \
                                     for i,ls in enumerate(val)])
                stacked = np.hstack(val)
                val = append_fields(stacked,'exp',exposures,
                                    usemask=False)
                del(stacked)
        return data    
class Series_old(object):
    table = np.dtype([('gauss','float64',(2,)),
                      ('lsf  ','float64',(2,)),
                      ('sigma','u4',())])
    
    def __init__(self,filepath,fibre,overwrite=False,ver=None):
        self._infile    = filepath
        self._dset_path = io.get_fits_path('dataset',filepath,ver)
        self._dataset   = FITS(self._dset_path,'rw')
        self._hdu_path  = io.get_fits_path('series',filepath,ver)
        self.fibre      = fibre
        primhead = self.return_header('primary')
        with FITS(self._hdu_path,'rw',clobber=overwrite) as hdu:
            hdu[0].write_keys(primhead)
        
    @property
    def dataset(self):
        return self._dataset
    def dset_read(self,item):
        extension, version, ver_sent = self._extract_item(item)
        return self.dataset[extension,version].read()
    def wavesol(self,fittype,version,sigma,refindex,**kwargs):
        waves2d = self.dset_read(('wavesol_{}'.format(fittype),version))
        dates   = self.dset_read('datetime')
        fluxes  = self.dset_read('flux')
        noises  = self.dset_read('noise')
        rv      = wavesol(waves2d,fittype,sigma,dates,fluxes,noises,refindex,
                          fibre=self.fibre,**kwargs)
        return rv
    def interpolate(self,use,fittype,version,sigma,refindex,**kwargs):
        linelist = self.dset_read('linelist')
        dates    = self.dset_read('datetime')
        fluxes   = self.dset_read('flux')
        rv       = interpolate(linelist,fittype,sigma,dates,fluxes,use=use,
                          refindex=refindex,fibre=self.fibre)
        return rv
    def interpolate_freq(self,fittype,version,sigma,refindex,**kwargs):
        rv = self.interpolate('freq',fittype,version,sigma,refindex,**kwargs)
        return rv
    def interpolate_cent(self,fittype,version,sigma,refindex,**kwargs):
        rv = self.interpolate('centre',fittype,version,sigma,refindex,**kwargs)
        return rv
    def coefficients(self,fittype,version,sigma,refindex,**kwargs):
        linelist = self.dset_read('linelist')
        dates    = self.dset_read('datetime')
        fluxes   = self.dset_read('flux')
        rv       = coefficients(linelist,fittype,version,sigma,dates,fluxes,
                                refindex=refindex,fibre=self.fibre,**kwargs)
        return rv
    def __getitem__(self,item):
        '''
        Tries reading data from file, otherwise runs __call__. 
        
        Args:
        ----
            datatype (str) :  ['linelist','coeff','model_gauss',
                               'wavesol_comb','wavesol_thar']
        
        Returns:
        -------
            data (array_like) : values of dataset
            
        '''
        ext, ver, versent = self._extract_item(item)
        mess = "Extension {ext:>20}, version {ver:<5}:".format(ext=ext,ver=ver)
        
        try:
            with FITS(self._hdu_path,'rw') as hdu:
                data    = hdu[ext,ver].read()
                mess   += " read from file."
        except:
            data   = self.__call__(ext,ver,write=True)
            mess   += " calculated."
        finally:
            #print(mess)
            pass
        return SeriesVelocityShift(data)
    def __call__(self,ext,version=None,refindex=0,write=False,**kwargs):
        methodfunc = {'wavesol':self.wavesol, 'freq':self.interpolate_freq,
                      'cent':self.interpolate_cent, 'coeff':self.coefficients}
        
        assert ext in ['wavesol_gauss','wavesol_lsf',
                       'freq_gauss','freq_lsf',
                       'cent_gauss','cent_lsf',
                       'coeff_gauss','coeff_lsf']
        
        method,fittype = ext.split('_')
        func = methodfunc[method]
        
        data = func(fittype,version,[1,2,3,4,5],refindex,**kwargs)
        if write:
            with FITS(self._hdu_path,'rw') as hdu:
                
                header = self.return_header(ext)
                hdu.write(data=data,header=header,
                                  extname=ext,extver=version)
        return data
        
    def _extract_item(self,item):
        """
        utility function to extract an "item", meaning
        a extension number,name plus version.
        """
        ver=0.
        if isinstance(item,tuple):
            ver_sent=True
            nitem=len(item)
            if nitem == 1:
                ext=item[0]
            elif nitem == 2:
                ext,ver=item
        else:
            ver_sent=False
            ext=item
        
        ver = hf.item_to_version(ver)
        return ext,ver,ver_sent
    def return_header(self,hdutype):
        def return_value(name):
            if name=='Simple':
                value = True
            elif name=='Bitpix':
                value = 32
            elif name=='Naxis':
                value = 0
            elif name=='Extend':
                value = True
            elif name=='Author':
                value = 'Dinko Milakovic'
            elif name=='version':
                value = hs_version
            elif name=='fibre':
                value = self.fibre
            return value
        def make_dict(name,value,comment=''):
            return dict(name=name,value=value,comment=comment)
            
        if hdutype == 'primary':
            names = ['Simple','Bitpix','Naxis','Extend','Author',
                     'fibre','version']            
        else: 
            names = ['version']
        comments_dict={'Simple':'Conforms to FITS standard',
                  'Bitpix':'Bits per data value',
                  'Naxis':'Number of data axes',
                  'Extend':'FITS dataset may contain extensions',
                  'Author':'',
                  'fibre':'Fibre',
                  'version':'Code version used'}
        values_dict = {name:return_value(name) for name in names}
        
        values   = [values_dict[name] for name in names]
        comments = [comments_dict[name] for name in names]
        header   = [make_dict(n,v,c) for n,v,c in zip(names,values,comments)]
        return FITSHDR(header)
    def plot(self,extension,version,sigma,exposures=None,ax=None,
             scale=None, **kwargs):
        version = hf.item_to_version(version)
        data    = self[extension,version]
        sigma1d = np.atleast_1d(sigma)
        #plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
        
        for sig in sigma1d:
            plotter = data.plot(sig,scale=scale,exposures=exposures,ax=ax,**kwargs)
        

        return plotter

    def process(self,methods,fittypes,versions,nproc=None):
        self.processes = []
        self.queue     = mp.Queue()
        
        def find_last(lst, elm):
          gen = (len(lst) - 1 - i for i, v in enumerate(reversed(lst)) if v == elm)
          return next(gen, None)
      
        def get_nproc(nelem):
            xx = [nelem%i for i in range(1,21)]
            return find_last(xx,0)+1
        iterables = np.array(list(itertools.product(methods,fittypes,versions)))
        nproc = nproc if nproc is not None else get_nproc(len(iterables))
        print("Using {} processors.".format(nproc))
        chunks = np.split(iterables,nproc)
        
        for i,chunk in enumerate(chunks):
            if len(chunk)<1:
                continue
            p = mp.Process(target=self._work_on_chunk,args=((chunk,)))
            hf.update_progress((i+1)/nproc,"chunk") 
            p.start()
            self.processes.append(p)
        for p in self.processes:
            p.join()
            
        while True:
            time.sleep(5)
            if not mp.active_children():
                break

        for i in range(len(iterables)):
            item = self.queue.get()          
            #print('{0:>5d} element extracted'.format(i))
        print("Finished")
        return
    def _work_on_chunk(self,chunk):
        chunk = np.atleast_1d(chunk)
        for i,settuple in enumerate(chunk):
            self._single(settuple)
            self.queue.put(settuple)
            
    def _single(self,settuple):
        method, fittype, version = settuple
        version = int(version)
        extension   = '{m}_{f}'.format(m=method,f=fittype) 
        rv_data  = self[extension,version]
    
def get_combext(fittype):
    ext     = ['wavesol','coeff','residuals']
    fittype = np.atleast_1d(fittype) if fittype is not None else ['gauss','lsf']
    combext = ['{0}_{1}'.format(*item) for item in itertools.product(ext,fittype)]
    return combext
class Dataset(object):
    basext  = ['datetime','linelist','flux','noise']
    combext = ['wavesol_gauss','wavesol_lsf','coeff_gauss','coeff_lsf',
                  'residuals_gauss','residuals_lsf']
    modext  = ['model_gauss','model_lsf']
    def __init__(self,filepath,overwrite=False):
        self._infile = filepath
        self._loaded   = False
        self._outfile = io.get_fits_path('dataset',filepath)
        
        
        #self._hdu = FITS(self._outfile,'rw',clobber=overwrite)
        primhead = self.return_header('primary')
        if not os.path.isfile(self._outfile):
            with FITS(self._outfile,'rw',clobber=overwrite) as hdu:
                hdu[0].write_keys(primhead)
        if overwrite:
            with FITS(self._outfile,'rw',clobber=overwrite) as hdu:
                hdu[0].write_keys(primhead)
                print(hdu)
        #print(FITS(self._outfile,'rw'))
        return 
    def __len__(self):
        return self.numfiles
    
    def __getitem__(self,item):
        '''
        Tries reading data from file, otherwise runs __call__. 
        
        Args:
        ----
            datatype (str) :  ['linelist','coeff','model_gauss',
                               'wavesol_comb','wavesol_thar']
            save     (bool):  saves to the FITS file if true
        
        Returns:
        -------
            data (array_like) : values of dataset
            
        '''
        ext, ver, versent = self._extract_item(item)
        mess = "Extension {ext:>20}, version {ver:<5}:".format(ext=ext,ver=ver)
        
        try:
            with FITS(self._outfile,'w') as hdu:
                data    = hdu[ext,ver].read()
                mess   += " read from file."
        except:
            data   = self.__call__(ext,ver,write=True)[ext]
            mess   += " calculated."
        finally:
            print(mess)
            pass
        return data
    def _extract_item(self,item):
        """
        utility function to extract an "item", meaning
        a extension number,name plus version.
        """
        ver=0.
        if isinstance(item,tuple):
            ver_sent=True
            nitem=len(item)
            if nitem == 1:
                ext=item[0]
            elif nitem == 2:
                ext,ver=item
        else:
            ver_sent=False
            ext=item
        
        ver = hf.item_to_version(ver)
        return ext,ver,ver_sent
    def __call__(self,extension,version=None,write=False,avflux=True,*args,**kwargs):
        """ 
        
        Calculate dataset.
        
        Parameters are dataset name and version. 
        
        Args:
        ----
        dataset (str) : name of the dataset
        version (int) : version number, 3 digit (PGS)
                        P = polynomial order
                        G = gaps
                        S = segment
        
        Returns:
        -------
        data (array_like) : values of dataset
            
        """
        #print(extension,version)
        version       = hf.item_to_version(version)
        #print("updated version", version)
        orders        = kwargs.pop('order',None)
        data,numfiles = io.mread_outfile(self._infile,extension,version,
                                         avflux=avflux,order=orders)
        if write:
            print('Preparing version {}'.format(version))
            
            for key,val in data.items():
                print(key)
                if key =='datetime':
                    val = hf.datetime_to_record(val)
                
                elif key not in ['wavesol_gauss','wavesol_lsf',
                                 'noise','flux','model_gauss','model_lsf']:
                    # stack, keep the exposure number (0-indexed)
                    exposures=np.hstack([np.full(len(ls),i) \
                                         for i,ls in enumerate(val)])
                    stacked = np.hstack(val)
                    val = append_fields(stacked,'exp',exposures,
                                        usemask=False)
                    del(stacked)
                with FITS(self._outfile,'rw') as hdu:
                    print('Writing version {} to file'.format(version))
                    header = self.return_header(key)
                    hdu.write(data=val,header=header,
                              extname=key,extver=version)
        return data
#    def __del__(self):
#        self.hdu.close()
        
    def read(self,version,fittype=None,*args,**kwargs):
        basext = Dataset.basext
        combext = get_combext(fittype)
        self.__call__(basext,write=True,*args,**kwargs)
        for ver in np.atleast_1d(version):
            data = self.__call__(combext,ver,write=True,*args,**kwargs)
            del(data)
        return
    def read_models(self):
        modext = Dataset.modext
        data = self.__call__(modext,write=True)
        return
    def return_header(self,hdutype):
        
        def return_value(name):
            if name=='Simple':
                value = True
            elif name=='Bitpix':
                value = 32
            elif name=='Naxis':
                value = 0
            elif name=='Extend':
                value = True
            elif name=='Author':
                value = 'Dinko Milakovic'
            elif name=='version':
                value = hs_version
            
            return value
        def make_dict(name,value,comment=''):
            return dict(name=name,value=value,comment=comment)
            
        if hdutype == 'primary':
            names = ['Simple','Bitpix','Naxis','Extend','Author','version']            
        else: 
            names = ['version']
        comments_dict={'Simple':'Conforms to FITS standard',
                  'Bitpix':'Bits per data value',
                  'Naxis':'Number of data axes',
                  'Extend':'FITS dataset may contain extensions',
                  'Author':'',
                  'version':'Code version used'}
        values_dict = {name:return_value(name) for name in names}
        
        values   = [values_dict[name] for name in names]
        comments = [comments_dict[name] for name in names]
        header   = [make_dict(n,v,c) for n,v,c in zip(names,values,comments)]
        return FITSHDR(header)
    
#    def mread_outfile(self,version):
#        
#        print("Reading version {}".format(version))
#        data, numfiles = io.mread_outfile(self._outfile,extensions,
#                                          version)
#        setattr(self,'numfiles',numfiles)
#        for key, val in data.items():
#            setattr(self,"{key}_v{ver}".format(key=key,ver=version),val)
#        setattr(self,'_loaded',True)
#        return

    def get(self,fittype,exposures=None,orders=None,):
        '''
        Returns the wavesols, lines, fluxes, noises, datetimes for a selection
        of exposures and orders
        '''
        wavesols0  = self['wavesol_{}'.format(fittype)]
        lines0     = self['linelist']
        fluxes0    = self['flux']
        noises0    = self['noise']
        datetimes0 = self['datetime']
        
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
    @property
    def hdu(self):
        return FITS(self._outfile,'rw')
class SeriesVelocityShift(object):
    def __init__(self,values):
        self._values = values
        try:
            self._nelem = np.shape(values)[0]
        except:
            self._nelem = len(values) 
#    def __str__(self):
#        print(self.values)
#        return "{0:=>80s}".format("")
    def __len__(self):
        return self._nelem   
    def __getitem__(self,item):
        return SeriesVelocityShift(self.values[item])
    def __add__(self,item):
        assert len(self)==len(item), "Unequal lengths"
        selfval = self.values
        itemval = item.values
        #assert selfval.dtype.fields.keys() == itemval.dtype.fields.keys()
        data       = np.zeros_like(self.values)
        if selfval.dtype.fields != None:
            for key in selfval.dtype.fields.keys():
                data[key][:,0] = selfval[:,0]-itemval[:,0]
                data[key][:,1] = np.sqrt(selfval[:,1]**2+itemval[:,1]**2)
        else:
            for key in selfval.dtype.fields.keys():
                data[:,0] = selfval[:,0]-itemval[:,0]
                data[:,1] = np.sqrt(selfval[:,1]**2+itemval[:,1]**2)
        return SeriesVelocityShift(data)
    def __sub__(self,item):
        assert len(self)==len(item), "Unequal lengths"
        selfval = self.values
        itemval = item.values
        #assert selfval.dtype.fields.keys() == itemval.dtype.fields.keys()
        data       = np.copy(self.values)

        if selfval.dtype.fields != None:
            for key in selfval.dtype.fields.keys():
                if key=='datetime' or key=='flux': continue
                data[key][:,0] = selfval[key][:,0]-itemval[key][:,0]
                data[key][:,1] = np.sqrt(selfval[key][:,1]**2 + \
                                         itemval[key][:,1]**2)
        else:
            data[:,0] = selfval[:,0]-itemval[:,0]
            data[:,1] = np.sqrt(selfval[:,1]**2+itemval[:,1]**2)
        return SeriesVelocityShift(data)
    @property
    def values(self):
        return self._values
    @property
    def shape(self):
        return np.shape(self.values)
    
    def copy(self):
        return SeriesVelocityShift(np.copy(self.values))
    def plot(self,sigma,scale=None,exposures=None,ax=None,
            legend=False,dateformat='%Y-%m-%d',**kwargs):
        left = kwargs.pop('left',None)
        ls = kwargs.pop('ls','-')
        lw = kwargs.pop('lw',0.8)
        m  = kwargs.pop('marker','o')
        ms = kwargs.pop('ms',4)
        a  = kwargs.pop('alpha',1.)
        
        return_plotter=False
        if ax is not None:
            ax = ax
        else:
            plotter = Figure2(1,1,left=left)
            ax      = plotter.add_subplot(0,1,0,1)
            return_plotter=True
        #axes   = plotter.axes
        values = self._values
        
        exp     = exposures if exposures is not None else slice(None)
        if not isinstance(exposures,slice):
            try:
                idx = np.arange(len(exp)) 
            except:
                idx = np.arange(len(self))
        else:
            try:
                idx = np.arange(exp.start,exp.stop,exp.step)
            except:
                idx = np.arange(len(self))
        # X-axis is exposures, possible keyword for offset from zero
        of = kwargs.pop('exp_offset',0)
        x0 = np.arange(len(self))+of
        xlabel = 'Exposure'
        # X-axis is average flux per line
        if scale == 'flux':
            x0 = values['flux'][idx]
            ls = ''
            xlabel = 'Average flux per line [counts]'
            ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,3))
            
            
        # X-axis is a time stamp
        elif scale=='datetime':
            datetimes = hf.tuple_to_datetime(values['datetime'])
            x0 = datetimes
            ls = ''
            xlabel = 'Date'
        # Select a subset of exposures (if provided)
        x = x0[idx]
        # Read in RV values and errors
        y,yerr = values['{}sigma'.format(sigma)].T
        label = kwargs.pop('label',None)
        # Plot
        ax.errorbar(x,y[exp],yerr[exp],ls=ls,lw=lw,marker=m,
                         ms=ms,alpha=a,label=label,**kwargs)
        if scale=='datetime':
            ax.xaxis.set_major_formatter(mdates.DateFormatter(dateformat))
        ax.axhline(0,ls=':',lw=1,c='k')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Global velocity shift "+r"[$\rm{ m s^{-1}}$]")
        try:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        except:
            pass
        if legend:
            ax.legend()
        if return_plotter:
            return ax, plotter
        else:
            return ax
    def _get_values(self,key):
#        if key=='datetime':
#            return self._values[key].view('i8')
#        else:
        return self._values[key]
    def groupby_bins(self,key,bins):
        values  = self._get_values(key)
       
        if key=='datetime':
            values = hf.tuple_to_datetime(values).view('i8')
            bins = bins.view('i8')
        
        binned = np.digitize(values,bins)
        groups = {int(bin):self[np.where(binned==bin)] \
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
        
        if key == 'datetime':
            values = hf.tuple_to_datetime(values).view('i8')
            mean   = np.datetime64(int(np.mean(values)),'s')
        else:
            mean   = np.mean(values,axis=0)
        return mean
    def std(self,key):
        values = self._get_values(key)
        std    = np.std(values)
        if key == 'datetime':
            std = np.timedelta64(int(std),'s')
        return std
    
    def correct_cti(self,fibre=None,fittype=None,method=None,
                    pars=None,sigma=None,copy=False):
        if copy:
            values = np.copy(self.values)
        else:
            values = self.values
        flux   = values['flux']
        corr, noise   = cti.exp(flux,fibre,fittype,method,pars,sigma)
        keys = [key for key in values.dtype.fields.keys() if 'sigma' in key]
        for key in keys:
            values[key][:,0] = (values[key][:,0] + corr) - corr[0]
            values[key][:,1] = np.sqrt(values[key][:,1]**2 + noise**2)
        if copy:
            return SeriesVelocityShift(values)
        else:
            return self
    def correct_time(self,pars=None,copy=False):
#        assert isinstance(datetime,np.datetime64), "Incorrect datetime format"
        pars = pars if pars is not None else self.get_time_pars()   
        A, B, datetime0 = pars
        if copy:
            values = np.copy(self.values)
        else:
            values = self.values
        datetimes = hf.tuple_to_datetime(values['datetime'])
        timedelta = (datetimes - datetime0)/np.timedelta64(1,'s')
        keys = [key for key in values.dtype.fields.keys() if 'sigma' in key]
        for key in keys:
            values[key][:,0] = values[key][:,0]-temporal_model(timedelta,A,B)
        if copy:
            return SeriesVelocityShift(values)
        else:
            return self
    def get_time_pars(self,bins=10,plot=False):
        values = self.values
        dtimes = hf.tuple_to_datetime(values['datetime'])
        time_bins = np.append(dtimes[bins::bins],dtimes[-1]+np.timedelta64(1,'D'))
#        time_bins = hf.histedges_equalN(dtimes.astype(np.float64), bins)
        print(time_bins)
        time_groups = self.groupby_bins('datetime',time_bins)
        

        time_pars  = temporal_fit(time_groups[0],time_groups[len(time_bins)-1])
        if plot:
            plotter = Figure2(1,1)
            ax      = plotter.add_subplot(0,1,0,1)
            #plot uncorrected unbinned data
            x1 = (dtimes - dtimes[0]).astype(np.float64)
            ax.plot(x1,values['3sigma'][:,0],lw=0.4,ms=4,
                    label="Uncorrected",marker='o')
            #plot mean of groups:
            ax.plot(
                [(time_groups[0].mean('datetime')-dtimes[0])/np.timedelta64(1,'s'),
                 (time_groups[14].mean('datetime')-dtimes[0])/np.timedelta64(1,'s')],
                [time_groups[0].mean('3sigma')[0],
                 time_groups[14].mean('3sigma')[0]],marker='x',ms=8,c='C1')
        
        return time_pars
        
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
#class RV(object):
#    def __init__(self,values):
#        self._values = values
#        self._nelem  = np.shape(values)[0]
#    def __str__(self):
#        print(self.values)
#        return "{0:=>80s}".format("")
#    def __len__(self):
#        return self._nelem
#    
#    def __getitem__(self,item):
#        return self.values[item]
#    def __add__(self,item):
#        idx1,idx2  = _intersect(self.values,item.values,'datetime')
#        arr1 = self._values[idx1]
#        arr2 = item.values[idx2]
#        
#        fibre1 = arr1['fibre'][0]
#        fibre2 = arr2['fibre'][0]
#        
#        data       = container.radial_velocity(len(id1))
#        
#        data['shift'] = arr1['shift'] + arr2['shift']
#        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
#        data['datetime'] = arr1['datetime']
#        data['fittype']  = arr1['fittype']
#        data['flux']  = (arr1['flux']+arr2['flux'])/2.
#        data['fibre'] = "{0}+{1}".format(fibre1,fibre2)
#        return RV(data)
#    def __sub__(self,item):
#        idx1,idx2  = _intersect(self.values,item.values,'datetime')
#        arr1 = self._values[idx1]
#        arr2 = item.values[idx2]
#        
#        fibre1 = arr1['fibre'][0]
#        fibre2 = arr2['fibre'][0]
#        
#        data       = container.radial_velocity(len(idx1))
#        
#        data['shift'] = arr1['shift'] - arr2['shift']
#        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
#        data['datetime'] = arr1['datetime']
#        data['flux']  = (arr1['flux']+arr2['flux'])/2.
#        data['fibre'] = "{0}-{1}".format(fibre1,fibre2)
#        return RV(data)
#    def __mul__(self,item):
#        
#        idx1,idx2  = _intersect(self.values,item.values,'datetime')
#        arr1 = self._values[idx1]
#        arr2 = item.values[idx2]
#        
#        fibre1 = arr1['fibre'][0]
#        fibre2 = arr2['fibre'][0]
#        
#        data       = container.radial_velocity(len(idx1))
#        
#        data['shift'] = arr1['shift'] * arr2['shift']
#        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
#        data['datetime'] = arr1['datetime']
#        data['flux']  = (arr1['flux']+arr2['flux'])/2.
#        data['fibre'] = "{0}*{1}".format(fibre1,fibre2)
#        return RV(data)
#    def __rmul__(self,item):
#        idx1,idx2  = _intersect(self.values,item.values,'datetime','fittype')
#        arr1 = self._values[idx1]
#        arr2 = item.values[idx2]
#        
#        data       = container.radial_velocity(len(idx1))
#        data['shift'] = arr1['shift'] * arr2['shift']
#        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
#        data['datetime'] = arr1['datetime']
#        data['flux']  = arr1['flux']#+arr2['flux'])/2.
#        return RV(data)
#    def __imul__(self,item):
#        
#        idx1,idx2  = _intersect(self.values,item.values,'datetime','fittype')
#        arr1 = self._values[idx1]
#        arr2 = item.values[idx2]
#        
#        data       = container.radial_velocity(len(idx1))
#        data['shift'] = arr1['shift'] * arr2['shift']
#        data['noise'] = np.sqrt(arr1['noise']**2 + arr2['noise']**2)
#        data['datetime'] = arr1['datetime'][idx]
#        data['flux']  = arr1['flux']#+arr2['flux'])/2.
#        return RV(data)
#    def __getitem__(self,key):
#        return RV(self._values[key])
#    def __setitem__(self,key,val):
#        self._values[key] = val
#        return
#    @property
#    def values(self):
#        return self._values
#    @property
#    def shape(self):
#        return np.shape(self.values)
#    
#    def copy(self):
#        return RV(np.copy(self.values))
#    
#    def _get_values(self,key):
#        if key=='datetime':
#            return self._values[key].view('i8')
#        else:
#            return self._values[key]
#    def groupby_bins(self,key,bins):
#        values0 = self.values[key]
#        values  = self._get_values(key)
#        if key=='datetime':
#            bins = bins.view('i8')
#        binned = np.digitize(values,bins)
#        groups = {str(bin):self[np.where(binned==bin)] \
#                      for bin in np.unique(binned)}
#        return groups
#    def min(self,key):
#        value = np.min(self._get_values(key))
#        if key=='datetime':
#            return np.datetime64(int(value),'s')
#        else:
#            return value
#    def max(self,key):
#        value = np.max(self._get_values(key))
#        if key=='datetime':
#            return np.datetime64(int(value),'s')
#        else:
#            return value
#    def mean(self,key):
#        values = self._get_values(key)
#        mean   = np.mean(values)
#        if key == 'datetime':
#            mean = np.datetime64(int(mean),'s')
#        return mean
#    def std(self,key):
#        values = self._get_values(key)
#        std    = np.std(values)
#        if key == 'datetime':
#            std = np.timedelta64(int(std),'s')
#        return std
#    
#    def correct_cti(self,fibre,pars=None,sigma=None,copy=False):
#        if copy:
#            values = np.copy(self.values)
#        else:
#            values = self.values
#        flux   = values['flux']
#        corr, noise   = cti.exp(flux,fibre,pars,sigma)
#        
#        values['shift'] = values['shift']+corr
#        if copy:
#            return RV(values)
#        else:
#            return self
#    def correct_time(self,pars,datetime,copy=False):
#        def model(x,pars):
#            A, B = pars
#            return A + B * x
#        
#        if copy:
#            values = np.copy(self.values)
#        else:
#            values = self.values
#        timedelta = (values['datetime'] - datetime)/np.timedelta64(1,'s')
#        values['shift'] = values['shift'] - model(timedelta,pars)
#        if copy:
#            return RV(values)
#        else:
#            return self
#    def plot(self,scale='sequence',plotter=None,axnum=0,legend=True,**kwargs):
#        ls = kwargs.pop('ls','-')
#        lw = kwargs.pop('lw',0.8)
#        m  = kwargs.pop('marker','o')
#        ms = kwargs.pop('ms',2)
#        a  = kwargs.pop('alpha',1.)
#        plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
#        
#        axes    = plotter.axes
#        results = self._values
#        
#        if scale == 'sequence':
#            x = np.arange(self._nelem)
#        elif scale == 'flux':
#            x = results['flux']
#        else:
#            x = (results['datetime']-results['datetime'][0]).astype(np.float64)
#        y     = results['shift']
#        yerr  = results['noise']
#        label = kwargs.pop('label',None)
#        axes[axnum].errorbar(x,y,yerr,ls=ls,lw=lw,marker=m,
#                         ms=ms,alpha=a,label=label,**kwargs)
#        axes[axnum].axhline(0,ls=':',lw=1,c='k')
#        axes[axnum].set_xlabel(scale.capitalize())
#        axes[axnum].set_ylabel("RV [m/s]")
#        if legend:
#            axes[axnum].legend()
#        return plotter

def _intersect(array1,array2,*keys):
    ''' Returns the index of data points with the same values of keys '''
    keys = np.atleast_1d(keys)
    assert len(keys)>0, "No key provided"
    lst1 = []
    lst2 = []
    for key in keys:
        ind1,ind2 = hf.overlap(array1[key],array2[key])
        lst1.append(ind1)
        lst2.append(ind2)
    if len(keys)==1:
        idx1 = ind1
        idx2 = ind2
    elif len(keys)==2:
        print(*lst1)
        idx1 = np.intersect1d(*lst1)
        idx2 = np.intersect1d(*lst2)
    else:
        from functools import reduce
        idx1 = reduce(np.intersect1d(*lst1))
        idx2 = reduce(np.intersect1d(*lst2))
    
    
#    dt1 = array1[key]
#    dt2 = set(array2[key])
#    idx = np.array([i for i, val in enumerate(dt1) if val in dt2])
    return idx1,idx2
def message(i,total,rv,noise):
    mess = ("EXP={exp:<5d}/{tot:<5d}".format(exp=i,tot=total) + \
            "{t1:>8s}{rv:>8.3f}".format(t1="RV =",rv=rv) + \
            "{t2:>8s}{pn:>7.3f}".format(t2="PN =",pn=noise))
    return mess
def radvel(nexp,sigma1d):
    dtype = np.dtype([('flux','float64',()),
                      ('datetime','u4',(6,))] + \
                     [('{}sigma'.format(s),'float64',(2,)) for s in sigma1d])
    narray = np.zeros(nexp,dtype=dtype)
    return narray
def get_idx(self,exposures):
    idx = np.arange(exposures.start,exposures.stop,exposures.step)
    return idx

def cut(exposures=None,orders=None,pixels=None):
    exposures = slice(*exposures) if exposures is not None else slice(None)
    orders    = slice(*orders) if orders is not None else slice(29,None,None)
    pixels    = slice(*pixels) if pixels is not None else slice(None)
    return exposures,orders,pixels
def wavesol(wavesols,fittype,sigma,datetimes,fluxes,noises,refindex=0,
            exposures=None,orders=None,pixels=None,verbose=False,fibre=None,
            plot2d=False,**kwargs):
    
    exposures, orders, pixels = cut(exposures,orders,pixels)
    wavesol2d  = wavesols[exposures,orders,pixels]
    waveref2d  = wavesol2d[refindex]
    nexp,nord,npix = np.shape(wavesol2d)
    sigma1d    = np.atleast_1d(sigma)
    data       = radvel(nexp,sigma1d)
    data['flux'] = fluxes
    for i,expwavesol in enumerate(wavesol2d):
        data[i]['datetime'] = tuple(datetimes[i])
        if i==refindex:
            shift = np.zeros_like(sigma1d)
            noise = np.zeros_like(sigma1d)
            res   = np.transpose([shift,noise])
        else:
            res = compare.wavesolutions(waveref2d,expwavesol,
                                                sig=sigma,**kwargs)
        for j,s in enumerate(sigma1d):
            data[i]['{}sigma'.format(s)] = res[j]
            data[i]['{}sigma'.format(s)][1] = noises[i]
        hf.update_progress((i+1)/len(wavesol2d),'wave')
    
    return data

    
def interpolate(linelist,fittype,sigma,datetimes,fluxes,use,refindex=0,
                exposures=None,orders=None,fibre=None,verbose=False,**kwargs):
    assert use in ['freq','centre']
    if exposures is not None:
        exposures, orders, pixels = cut(exposures,orders,None)
        idx = get_idx(exposures)
    else:
        idx = np.unique(linelist['exp']) 
    reflinelist = select(linelist,dict(exp=refindex))
    nexp = len(idx)
    sigma1d    = np.atleast_1d(sigma)
    data       = radvel(nexp,sigma1d)
    data['flux']     = fluxes
    for i in idx:
        data[i]['datetime'] = tuple(datetimes[i])
        if i == refindex:
            continue
        explinelist = select(linelist,dict(exp=i))
        res = compare.interpolate(reflinelist,explinelist,
                                        fittype=fittype,
                                        sig=sigma1d,
                                        use=use,**kwargs)
        for j,s in enumerate(sigma1d):
            data[i]['{}sigma'.format(s)] = res[j]
        hf.update_progress((i+1)/nexp,'{}'.format(use))
    return data
def coefficients(linelist,fittype,version,sigma,datetimes,fluxes,refindex=0,
                coeffs=None,fibre=None,exposures=None,order=None,
                verbose=False,**kwargs):
    if exposures is not None:
        exposures, orders, pixels = cut(exposures,orders,None)
        idx = get_idx(exposures)
    else:
        idx = np.unique(linelist['exp']) 
    # convert the input linelist into lines.Linelist
    ll     = container.Generic(linelist)
    # perform order selection according to 'order' keyword. 
    # defaults to using all available orders 
    available_orders = np.unique(linelist['order'])
    orders = available_orders
    condict0 = dict(exp=refindex)
    if order is not None:
        minord = np.min(available_orders)
        maxord = np.max(available_orders)
        orders = hf.wrap_order(order,minord,maxord)
        condict0.update(order=orders)
    reflinelist = ll.select(condict0)

    nexp = len(idx)
    sigma1d     = np.atleast_1d(sigma)
    data        = radvel(nexp,sigma1d)
    data['flux']     = fluxes
    if coeffs is None:
        coeffs  = ws.get_wavecoeff_comb(reflinelist.values,version,fittype)
    for i in idx:
        data[i]['datetime'] = tuple(datetimes[i])
        condict = dict(exp=i)
        if order is not None:
            condict.update(order=orders)
        linelist1exp = ll.select(condict)
        #reflines = lines[j-1]
        res = compare.from_coefficients(linelist1exp.values,coeffs,
                                              fittype=fittype,
                                              version=version,
                                              sig=sigma1d,
                                              **kwargs)
        #data[i]['flux']  = np.sum(fluxes[j])/len(lines[j])
        for j,s in enumerate(sigma1d):
            data[i]['{}sigma'.format(s)] = res[j]
        hf.update_progress((i+1)/nexp,'coeff')

    return data