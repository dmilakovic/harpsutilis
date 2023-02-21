#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
import harps.io as io
import harps.plotter as hplot
from harps.core import np, FITS

import harps.lsf.aux as aux
import harps.lsf.plot as lsf_plot
import harps.lsf.construct as construct

import jax.numpy as jnp
from tinygp import kernels
import gc

from matplotlib import ticker

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
        scale      = self.scale
        assert scale in ['pixel','velocity']
        wavelengths = self['wavereference']
        fluxes      = self['flux']
        backgrounds = self['background']
        errors      = self['error']
        fittype     = 'lsf'
        for i in range(self._iter_solve):
            linelists = self['linelist']
            if i == 0:
                fittype = 'gauss'
            pix3d,vel3d,flx3d,err3d,orders = aux.stack(fittype,linelists,fluxes,
                                                wavelengths,errors,backgrounds,
                                         self._orders)
            # lsf_i    = construct_lsf(pix3d,flx3d,err3d,self._orders,
            #                          numseg=self._numseg,
            #                          numpix=self._numpix,
            #                          subpix=self._subpix,
            #                          numiter=self._iter_center,
            #                          method=self._method,
            #                          filter=self._filter,
            #                          verbose=verbose)
            lst = []
            for j,od in enumerate(self._orders):
                print("order = {}".format(od))
                plot=True; save_plot=True
                if scale=='pixel':
                    x3d = pix3d
                elif scale=='velocity':
                    x3d = vel3d
                metadata=dict(
                    order=od,
                    scale=scale,
                    model_scatter=model_scatter,
                    
                    )
                lsf1d=(construct.models_1d(x3d[od],flx3d[od],err3d[od],
                                          numseg=self._numseg,
                                          numiter=self._iter_center,
                                          minpts=15,
                                          model_scatter=model_scatter,
                                          minpix=None,maxpix=None,
                                          filter=None,plot=plot,
                                          metadata=metadata,
                                          save_plot=save_plot))
                lsf1d['order'] = od
                lst.append(lsf1d)
                
                if len(orders)>1:
                    hf.update_progress((j+1)/len(orders),'Fit LSF')
                if filepath is not None:
                    self.save(lsf1d,filepath,extname=f"{scale}_{od}",
                              version=f'{j+1:02d}',overwrite=False)
                del(lsf1d)
                gc.collect()
            lsf_i = LSF(np.hstack(lst))
            self._lsf_i = lsf_i
            setattr(self,'lsf_{}'.format(i),lsf_i)
            if i < self._iter_solve-1:
                linelists_i = aux.solve(lsf_i,linelists,fluxes,errors,
                                    backgrounds,fittype,)
                self['linelist'] = linelists_i
            self.iters_done += 1
        lsf_final = lsf_i
        self._lsf_final = lsf_final
        
        return lsf_final
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

    
class LSF(object):
    def __init__(self,narray):
        self._values = narray
    def __getitem__(self,item):
        condict, segm_sent = self._extract_item(item)
        values  = self.values 
        condition = np.logical_and.reduce(tuple(values[key]==val \
                                       for key,val in condict.items()))
        cut = np.where(condition==True)[0]
        
        return LSF(values[cut])

    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning order plus segment.
        
        To be used with partial decorator
        """
        condict = {}
        if isinstance(item,dict):
            if len(item)==2: segm_sent=True
            condict.update(item)
        else:
            dict_sent=False
            if isinstance(item,tuple):
                
                nitem = len(item) 
                if nitem==2:
                    segm_sent=True
                    order,segm = item
                    
                elif nitem==1:
                    segm_sent=False
                    order = item[0]
            else:
                segm_sent=False
                order=item
            condict['order']=order
            if segm_sent:
                condict['segm']=segm
        return condict, segm_sent
    @property
    def values(self):
        return self._values
    @property
    def x(self):
        return self._values['x']
    @property
    def y(self):
        return self._values['y']
    @property
    def deriv(self):
        return self._values['dydx']
    @property
    def pars(self):
        return self._values['pars']
    
    def save(self,filepath,version=None,overwrite=False):
        with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
            hdu.write(self.values,extname='LSF',extver=version)
        hdu.close()
        print("File saved to {}".format(filepath))
        return
    def plot(self,ax=None,title=None,saveto=None,*args,**kwargs):
        if ax is not None:
            ax = ax  
        else:
            plotter = hplot.Figure2(1,1,left=0.08,bottom=0.12)
            figure, ax = plotter.fig, plotter.add_subplot(0,1,0,1)
        # try:
        ax = lsf_plot.plot_spline_lsf(self.values,ax,title,saveto,*args,**kwargs)
        # except:
        #     ax = plot_analytic_lsf(self.values,ax,title,saveto,*args,**kwargs)
        
        ax.set_ylim(-5,100)
        ax.set_xlabel("Distance from center"+r" [kms$^{-1}$]")
        ax.set_ylabel("Relative intensity")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5,steps=[1,2,5]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#        ax.set_yticklabels([])
        ax.grid(True,ls=':',lw=1,which='both',axis='both')

        if title:
            ax.set_title(title)
        if saveto:
            figure.savefig(saveto)
        return ax
    def interpolate(self,order,center):
        
        # if method=='spline':
        return aux.interpolate_local_spline(self,order,center)
        # elif method == 'analytic':
        #     return interpolate_local_analytic(self,order,center)


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
    
    