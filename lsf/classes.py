#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
import harps.inout as io
import harps.plotter as hplot
from harps.core import np, FITS

import harps.lsf.aux as aux
import harps.lsf.plot as lsf_plot
import harps.lsf.construct as construct

import jax.numpy as jnp
from tinygp import kernels
import gc


class LSFModeller(object):
    def __init__(self,outfile,sOrder,eOrder,scale,iter_solve=2,iter_center=5,
                 numseg=16,filter=None):
        assert scale in ['velocity','pixel'], "Provided scale unknown." +\
            "Allowed values are 'velocity' and 'pixel'"
        
        self._outfile = outfile
        self._cache = {}
        self._iter_solve  = iter_solve
        self._iter_center = iter_center
        self._numseg  = numseg
        # self._numpix  = numpix
        # self._subpix  = subpix
        self._sOrder  = sOrder
        self._eOrder  = eOrder
        self._orders  = np.arange(sOrder,eOrder)
        # self._method  = method
        self._filter  = filter
        self.iters_done = 0
        self.scale = scale
    def __getitem__(self,extension):
        try:
            data = self._cache[extension]
        except:
            self._read_data_from_file()
            #self._cache.update({extension:data})
            data = self._cache[extension]
        if extension in ['flux','error','background']:
            data = data*100
        return data
    def __setitem__(self,extension,data):
        self._cache.update({extension:data})
    def _read_data_from_file(self,start=None,stop=None,step=None,**kwargs):
        extensions = ['linelist','flux','background','error','wavereference']
        data, numfiles = io.mread_outfile(self._outfile,extensions,701,
                                start=start,stop=stop,step=step)
        self._cache.update(data)
        self.numfiles = numfiles
        return
    
    def __call__(self,model_scatter,filepath=None,verbose=False):
        """ Returns the LSF in an numpy array  """
        
    def stack(self,fittype='lsf'):
        fluxes      = self['flux']
        backgrounds = self['background']
        linelists   = self['linelist']
        errors      = self['error']
        wavelengths = self['wavereference']
        
        return aux.stack(fittype,linelists,fluxes,wavelengths,errors,
                     backgrounds,self._orders)
        
    def save(self,data,filepath,extname,version=None,overwrite=False):
        with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
            hdu.write(data,extname=extname,extver=version)
        hdu.close()
        print("File saved to {}".format(filepath))
        return

    


class SpectralMixture(kernels.Kernel):
    def __init__(self, weight, scale, freq):
        self.weight = jnp.atleast_1d(weight)
        self.scale = jnp.atleast_1d(scale)
        self.freq = jnp.atleast_1d(freq)

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]
        return jnp.sum(
            self.weight
            * jnp.prod(
                jnp.exp(-2 * jnp.pi**2 * tau**2 / self.scale**2)
                * jnp.cos(2 * jnp.pi * self.freq * tau),
                axis=-1,
            )
        )
    
    