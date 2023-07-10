#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:54:05 2023

@author: dmilakov
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import harps.plotter as hplot
import harps.lsf.gp as hlsfgp
import harps.containers as container
import harps.progress_bar as progress_bar
import harps.lsf.read as hread
from fitsio import FITS

import jax.numpy as jnp

class LSF1d(object):
    def __init__(self,narray):
        self._values = narray
        self._cache  = {}
    def __len__(self):
        return len(self._values)
    def __getitem__(self,item):
        # condict, segm_sent = self._extract_item(item)
        values  = self.values 
        # condition = np.logical_and.reduce(tuple(values[key]==val \
        #                                for key,val in condict.items()))
        cut = np.where(values['segm']==item)
        return LSF1d(values[cut])

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
    # @property
    # def numerical_model(self,xrange=(-8,8),subpix=11):
    #     try:
    #         nummodel = self._cache['numerical_model']
    #     except:
    #         nummodel = LSF1d(numerical_model(self.values,
    #                                        xrange=xrange,
    #                                        subpix=subpix))
    #         self._cache['numerical_model'] = nummodel
    #     return nummodel
    #     # LSF2d_numerical = LSF(lsf2d_numerical)
    def interpolate_lsf(self,center,N=2):
        lsf1d_num = self.values
        return interpolate_local_lsf(center,lsf1d_num,N=N)
    def interpolate_scatter(self,center,N=2):
        lsf1d_num = self.values
        return interpolate_local_scatter(center,lsf1d_num,N=N)
    
class LSF2d(object):
    def __init__(self,narray):
        self._values = narray
        self._cache  = {}
        
    def __len__(self):
        return len(self._values)
    def __getitem__(self,item):
        condict, segm_sent = self._extract_item(item)
        values  = self.values 
        condition = np.logical_and.reduce(tuple(values[key]==val \
                                       for key,val in condict.items()))
        cut = np.where(condition==True)[0]
        
        return LSF1d(values[cut])

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
    # @property
    # def numerical_model(self,xrange=(-8,8),subpix=11):
    #     try:
    #         nummodel = self._cache['numerical_model']
    #     except:
    #         nummodel = LSF1d(numerical_model(self.values,
    #                                        xrange=xrange,
    #                                        subpix=subpix))
    #         self._cache['numerical_model'] = nummodel
    #     return nummodel
    #     # LSF2d_numerical = LSF(lsf2d_numerical)
    def interpolate(self,order,center,N=2):
        lsf1d_num = self.numerical_model[order].values
        return interpolate_local_lsf(center,lsf1d_num,N=N)


def interpolate_local_lsf(center,lsf1d_num,N=2):
    return interpolate_local(center,'LSF',lsf1d_num,N=2)
    
def interpolate_local_scatter(center,lsf1d_num,N=2):
    return interpolate_local(center,'scatter',lsf1d_num,N=2)
    

def interpolate_local(center,what,lsf1d_num,N=2):
    assert np.isfinite(center)==True, "Center not finite, {}".format(center)
    # values  = lsf1d_num[order].values
    # assert len(values)>0, "No LSF model for order {}".format(order)
    
    indices, weights = get_segment_weights(center, lsf1d_num)
    
    if what=='LSF':
        name = 'y'
    elif what=='scatter':
        name = 'scatter'
    lsf_array = [lsf1d_num[i][name] for i in indices]
    loc_lsf_x    = lsf1d_num[indices[0]]['x']
    loc_lsf_y    = helper_calculate_average(lsf_array,
                                            weights)
    return loc_lsf_x,loc_lsf_y

def helper_calculate_average(list_array,weights):
    N = len(list_array[0])
    weights_= np.vstack([np.full(N,w,dtype='float32') for w in weights])
    average = np.average(list_array,axis=0,weights=weights_) 
    return average



def get_segment_centres(lsf1d):
    segcens = (lsf1d['pixl']+lsf1d['pixr'])/2
    return segcens


def get_segment_weights(center,lsf1d,N=2):
    sorter=np.argsort(lsf1d['segm'])
    segcens   = get_segment_centres(lsf1d[sorter])
    # print(segcens)
    segdist   = jnp.diff(segcens)[0] # assumes equally spaced segment centres
    distances = jnp.abs(center-segcens)
    if N>1:
        used  = distances<segdist*(N-1)
    else:
        used  = distances<segdist/2.
    inv_dist  = 1./distances
    
    segments = jnp.where(used)[0]
    weights  = inv_dist[used]/np.sum(inv_dist[used])
    
    return segments, weights