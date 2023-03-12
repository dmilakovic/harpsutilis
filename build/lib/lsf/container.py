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
import harps.lsf.aux as aux
from fitsio import FITS

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
