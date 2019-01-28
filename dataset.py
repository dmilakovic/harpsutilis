#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:55:42 2019

@author: dmilakov
"""
from   harps.settings import __version__ as version
from   harps.core import np
import harps.io as io
import harps.functions as hf
from   fitsio import FITS, FITSHDR
from   numpy.lib.recfunctions import append_fields

class Dataset(object):
    basext  = ['datetime','linelist','flux','noise']
    combext = ['wavesol_gauss','wavesol_lsf','coeff_gauss','coeff_lsf',
                  'residuals_gauss','residuals_lsf']
    def __init__(self,filepath,overwrite=False):
        self._infile = filepath
        self._loaded   = False
        self._outfile = io.get_fits_path('dataset',filepath)
        
        
        #self._hdu = FITS(self._outfile,'rw',clobber=overwrite)
        if overwrite:
            primhead = self.return_header('primary')
            with FITS(self._outfile,'rw',clobber=overwrite) as hdu:
                print(hdu.mode)
                hdu[0].write_keys(primhead)
                #print(hdu.mode)
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
            with FITS(self._outfile,'rw') as hdu:
                data    = hdu[ext,ver].read()
                mess   += " read from file."
        except:
            data   = self.__call__(ext,ver)[ext]
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
    def __call__(self,extension,version=None,write=False,*args,**kwargs):
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
        
        version = hf.item_to_version(version)
        data,numfiles = io.mread_outfile(self._infile,extension,version)
        if write:
            print('Writing to file')
            with FITS(self._outfile,'rw') as hdu:
            #hdu = self.hdu
                for key,val in data.items():
                    print(key)
                    if key =='datetime':
                        val = hf.datetime_to_record(val)
                    # TO DO: add average fluxes
                    elif key not in ['wavesol_gauss','wavesol_lsf','noise']:
                        # stack, keep the exposure number (0-indexed)
                        exposures=np.hstack([np.full(len(ls),i) \
                                             for i,ls in enumerate(val)])
                        stacked = np.hstack(val)
                        val = append_fields(stacked,'exp',exposures,
                                            usemask=False)
                        del(stacked)
                    header = self.return_header(key)
                    hdu.write(data=val,header=header,
                              extname=key,extver=version)
        return data
#    def __del__(self):
#        self.hdu.close()
        
    def read(self,version):
        basext = Dataset.basext
        self.__call__(basext,write=True)
        for ver in np.atleast_1d(version):
            data = self.__call__(Dataset.combext,ver,write=True)
            del(data)
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
                value = version
            
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
    
    def mread_outfile(self,version):
        
        print("Reading version {}".format(version))
        data, numfiles = io.mread_outfile(self._outfile,extensions,
                                          version)
        setattr(self,'numfiles',numfiles)
        for key, val in data.items():
            setattr(self,"{key}_v{ver}".format(key=key,ver=version),val)
        setattr(self,'_loaded',True)
        return

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