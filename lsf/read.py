#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:21:02 2023

@author: dmilakov
"""
from harps.lsf.classes import LSFModeller
import harps.lsf.aux as aux
import harps.lsf.gp_aux as gp_aux
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


def get_data(fname,od,pixl,pixr,xscale,filter=None):
    # import harps.lsf as hlsf
    import harps.io as io
    # modeller=LSFModeller(fname,50,72,method='gp',subpix=10,
    #                           filter=None,numpix=8,iter_solve=1,iter_center=1)

    extensions = ['linelist','flux','background','error','wavereference']
    data, numfiles = io.mread_outfile(fname,extensions,701,
                            start=None,stop=None,step=None)
    linelists=data['linelist']
    fluxes=data['flux']
    errors=data['error']
    backgrounds=data['background']
    # backgrounds=None
    wavelengths=data['wavereference']
    
    # plt.figure()
    # plt.plot(wavelengths[0,od,pixl:pixr],(fluxes/errors)[0,od,pixl:pixr])
    # plt.ylabel('S/N')
    
    orders=np.arange(od,od+1)
    pix3d,vel3d,flx3d,err3d,orders=aux.stack('gauss',
                                              linelists,
                                              fluxes,
                                              wavelengths,
                                              errors,
                                              backgrounds,
                                              orders)


    pix1s=pix3d[od,pixl:pixr]
    vel1s=vel3d[od,pixl:pixr]
    flx1s=flx3d[od,pixl:pixr]
    err1s=err3d[od,pixl:pixr]

    # vel1s_ , flx1s_, err1s_ = vel1s, flx1s, err1s
    x = pix1s
    if xscale=='vel':
        x = vel1s
        
    vel1s_, flx1s_, err1s_ = aux.clean_input(x,flx1s,err1s,sort=True,
                                              verbose=True,filter=filter)
    
    X      = np.array(vel1s_)
    # X      = jnp.array(pix1s)
    Y      = np.array(flx1s_)
    Y_err  = np.array(err1s_)
    # Y      = jnp.array([flx1s_,err1s_])
    plt.figure(); plt.plot(X,Y/Y_err,'.k'); plt.ylabel('S/N')
    
    return X, Y, Y_err

def parameters_from_lsf1s(lsf1s,parnames=None):
    dictionary = {}
    if parnames is not None:
        parnames = np.atleast_1d(parnames)
    else:
        parnames = gp_aux.parnames_lfc + gp_aux.parnames_sct
    for parname in parnames:
        try:
            dictionary.update({parname:lsf1s[parname]})
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
    data = lsf1s[field] 
    cut  = np.where(~np.isnan(data))[0]
    return np.array(data[cut],dtype='float32')

def scatter_from_lsf1s(lsf1s):
    scatter = from_lsf1s(lsf1s,'scatter')
    if len(scatter[0])==0:
        scatter = None
    return scatter

def LSF_from_lsf1s(lsf1s):
    return from_lsf1s(lsf1s,'LSF')
    
