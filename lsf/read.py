#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:21:02 2023

@author: dmilakov
"""

import harps.lsf.aux as aux
import harps.lsf.gp_aux as gp_aux
import matplotlib.pyplot as plt
import harps.plotter as hplt
import numpy as np
import jax.numpy as jnp


def get_data(fname,od,pixl,pixr,scale,fittype='gauss',filter=None,plot=True):
    # import harps.lsf as hlsf
    # from harps.lsf.classes import LSFModeller
    import harps.inout as io
    # modeller=LSFModeller(fname,50,72,method='gp',subpix=10,
    #                           filter=None,numpix=8,iter_solve=1,iter_center=1)

    extensions = ['linelist','flux','background','envelope','error','wavereference']
    data, numfiles = io.mread_outfile(fname,extensions,None,
                            start=None,stop=None,step=None)
    linelists=data['linelist']
    fluxes=data['flux']
    errors=data['error']
    backgrounds=data['background']
    envelopes=data['envelope']
    # backgrounds=None
    wavelengths=data['wavereference']
    
    
    
    orders=np.arange(od,od+1)
    pix3d,vel3d,flx3d,err3d,orders=aux.stack(fittype,
                                              linelists,
                                              fluxes,
                                              wavelengths,
                                              errors,
                                              envelopes,
                                              backgrounds,
                                              orders)
    # pix3d,vel3d,flx3d,err3d,orders=aux.stack_div_envelope(fittype,
    #                                           linelists,
    #                                           flx3d_in=fluxes,
    #                                           x3d_in=wavelengths,
    #                                           err3d_in=errors,
    #                                           env3d_in=envelopes,
    #                                           bkg3d_in=backgrounds,
    #                                           orders=orders)


    pix1s=pix3d[od,pixl:pixr]
    vel1s=vel3d[od,pixl:pixr]
    flx1s=flx3d[od,pixl:pixr]
    err1s=err3d[od,pixl:pixr]

    # vel1s_ , flx1s_, err1s_ = vel1s, flx1s, err1s
    x = pix1s
    if scale=='velocity':
        print(scale)
        x = vel1s
        
    x1s_, flx1s_, err1s_ = aux.clean_input(x,flx1s,err1s,sort=True,
                                              verbose=True,filter=filter)
    
    X      = np.array(x1s_)
    # X      = jnp.array(pix1s)
    Y      = np.array(flx1s_)
    Y_err  = np.array(err1s_)
    # Y      = jnp.array([flx1s_,err1s_])
    if plot:
        fig = hplt.Figure2(1, 3,figsize=(10,3),left=0.05,bottom=0.15,right=0.97,
                          wspace=0.2,)
        ax1 = fig.add_subplot(0,1,0,1)
        ax2 = fig.add_subplot(0,1,1,2)
        ax3 = fig.add_subplot(0,1,2,3)
        # ax1,ax2,ax3 = (fig.ax() for i in range(3))
        ax1.plot(wavelengths[0,od,pixl:pixr]/10.,
                 (fluxes-backgrounds)[0,od,pixl:pixr],
                 # drawstyle='steps-mid'
                 )
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Flux (counts)')
        
        ax2.errorbar(X,Y,Y_err,ls='',marker='.',color='k')
        ax2.set_xlabel('Distance from centre (pix)')
        ax2.set_ylabel('Flux (arbitrary)')
        # Representative error bar
        data_yerr = np.median(Y_err)
        # axis_yerr = ax2.transAxes.transform(data_yerr)
        # axCoord = (0.9,0.9)
        # print(ax2.transLimits.transform((0,-1)))
        # axis_to_data = ax2.transLimits.inverted()
        # dataCoord = axis_to_data.transform(axCoord)
        # print(dataCoord)
        xlims = ax2.get_xlim(); x = xlims[0] + 0.9*(xlims[1]-xlims[0])
        ylims = ax2.get_ylim(); y = ylims[0] + 0.9*(ylims[1]-ylims[0])
        ax2.errorbar(x,y,10*data_yerr,fmt='',color='k')
        
        
        ax3.plot(X,Y/Y_err,'.k')
        ax3.set_xlabel('Distance from centre (pix)')
        ax3.set_ylabel('S/N')
        
        fig.ticks_('major', 0,'x',tick_every=0.1)
        fig.scinotate(0, 'y', bracket='round')
        
        return X, Y, Y_err, fig
    else:
        return X, Y, Y_err
    

def parameters_from_lsf1s(lsf1s,parnames=None):
    dictionary = {}
    if parnames is not None:
        parnames = np.atleast_1d(parnames)
    else:
        parnames = gp_aux.parnames_lfc + gp_aux.parnames_sct
    for parname in parnames:
        try:
            dictionary.update({parname:jnp.array(lsf1s[parname][0])})
            
        except:
            try:
                dictionary.update({parname:jnp.array(lsf1s[parname])})
            # print(parname,)
            except:
                continue
    return dictionary

def from_lsf1s(lsf1s,what):
    if what == 'LSF':
        desc = 'data'
        parnames = gp_aux.parnames_lfc
    elif what == 'scatter':
        desc = 'sct'
        parnames = gp_aux.parnames_sct
    
    pars = parameters_from_lsf1s(lsf1s,parnames)
    field_names = [f"{desc}_{coord}" for coord in ['x','y','yerr']]
    x, y, y_err = (field_from_lsf1s(lsf1s,field) for field in field_names)
    return (pars, x, y, y_err)

def field_from_lsf1s(lsf1s,field):
    lsf1s = prepare_lsf1s(lsf1s)
    data = lsf1s[field] 
    cut  = np.where(~np.isnan(data))[0]
    return jnp.array(data[cut],dtype='float32')


def scatter_from_lsf1s(lsf1s):
    scatter = from_lsf1s(lsf1s,'scatter')
    if len(scatter[0])==0:
        scatter = None
    return scatter

def LSF_from_lsf1s(lsf1s):
    return from_lsf1s(lsf1s,'LSF')
    
def prepare_lsf1s(lsf1s):
    test = len(np.shape(lsf1s))
    if test>0:
        return lsf1s[0]
    else:
        return lsf1s
    
#%%% FITS FILES
from fitsio import FITS
