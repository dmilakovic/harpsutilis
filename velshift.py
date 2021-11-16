#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:15:54 2020

@author: dmilakov
"""
import numpy as np

import harps.functions as hf
import harps.compare as compare
import harps.wavesol as ws
from   harps.lines import select
import harps.containers as container

def message(i,total,rv,noise):
    mess = ("EXP={exp:<5d}/{tot:<5d}".format(exp=i,tot=total) + \
            "{t1:>8s}{rv:>8.3f}".format(t1="RV =",rv=rv) + \
            "{t2:>8s}{pn:>7.3f}".format(t2="PN =",pn=noise))
    return mess
def velarray(n,m):
    dtype = np.dtype([('flux','float64',()),
                      ('datetime','datetime64[s]',()),
                      ('mean','float64',(m)),
                      ('sigma','float64',(m))])
    narray = np.zeros(n,dtype=dtype)
    return narray
def get_idx(self,exposures):
    idx = np.arange(exposures.start,exposures.stop,exposures.step)
    return idx

def cut(exposures=None,orders=None,pixels=None):
    exposures = slice(*exposures) if exposures is not None else slice(None)
    orders    = slice(*orders) if orders is not None else slice(None,None,None)
    pixels    = slice(*pixels) if pixels is not None else slice(None)
    return exposures,orders,pixels
def wavesol(wavesols,fittype,sigma,datetimes=None,fluxes=None,noises=None,refindex=0,
            exposures=None,orders=None,pixels=None,verbose=False,fibre=None,
            plot2d=False,**kwargs):
    exposures = slice(*exposures) if exposures is not None else slice(None)
    pixels    = slice(*pixels) if pixels is not None else slice(None)
    if orders is not None:
        orders = slice(*orders) 
    else:
        orders = np.where(wavesols.sum(axis=0).sum(axis=-1)!=0)[0]
        
    wavesol2d  = wavesols[exposures,orders,pixels]
    waveref2d  = wavesol2d[refindex]
    nexp,nord,npix = np.shape(wavesol2d)
    data       = velarray(nexp,len(np.atleast_1d(sigma)))
    if fluxes is not None:
        data['flux'] = fluxes
    for i,expwavesol in enumerate(wavesol2d):
        if datetimes is not None:
            data[i]['datetime'] = datetimes[i]
        if i==refindex:
            res = (0,0)
        else:
            res = compare.wavesolutions(waveref2d,expwavesol,
                                                sigma=sigma,**kwargs)
        data[i]['mean']  = res[0]
        data[i]['sigma'] = res[1]
        hf.update_progress((i+1)/len(wavesol2d),'wave')
    
    return data

    
def interpolate(linelist,fittype,sigma,datetimes,fluxes,use,refindex=0,
                exposures=None,orders=None,fibre=None,verbose=False,**kwargs):
    assert use in ['freq','centre']
    nexp = len(exposures) if exposures is not None else len(linelist)
    if exposures is not None:
        exposures, orders, pixels = cut(exposures,orders,None)
        idx = get_idx(exposures)
    else:
        idx = np.arange(nexp)
    reflinelist = linelist[refindex]
    
    data       = velarray(nexp,len(np.atleast_1d(sigma)))
    data['flux']     = fluxes
    for i in idx:
        data[i]['datetime'] = datetimes[i]
        if i == refindex:
            continue
        explinelist = linelist[i]
        res = compare.interpolate(reflinelist,explinelist,
                                        fittype=fittype,
                                        sigma=sigma,
                                        use=use,**kwargs)
        data[i]['mean'] = res[0]
        data[i]['sigma'] = res[1]
        hf.update_progress((i+1)/nexp,'{}'.format(use))
    return data
def coefficients(linelist,fittype,version,sigma,datetimes,fluxes,refindex=0,
                coeffs=None,fibre=None,exposures=None,order=None,
                verbose=False,**kwargs):
    # check if linelist is stacked (see output of harps.functions.stack_array)
    # stack otherwise
    if len(linelist.dtype)==0:
        nexp      = len(linelist)
        linelist0 = hf.stack_arrays(linelist)
    else:
        nexp      = len(np.unique(linelist['exp']))
        linelist0 = linelist
        
    if exposures is not None:
        exposures, orders, pixels = cut(exposures,order,None)
        idx = get_idx(exposures)
    else:
        idx = np.arange(nexp)
    # convert the input linelist into lines.Linelist
    ll     = container.Generic(linelist0)
    
    # perform order selection according to 'order' keyword. 
    # defaults to using all available orders 
    available_orders = np.unique(ll.values[refindex]['order'])
    orders = available_orders
    condict0 = dict(exp=refindex)
    if order is not None:
        minord = np.min(available_orders)
        maxord = np.max(available_orders)
        orders = hf.wrap_order(order,minord,maxord)
        condict0.update(order=orders)
    reflinelist      = ll.select(condict0)
    nexp             = len(idx)
    sigma1d          = np.atleast_1d(sigma)
    data             = velarray(nexp,len(sigma1d))
    data['flux']     = fluxes
    data['datetime'] = datetimes
    if coeffs is None:
        coeffs  = ws.get_wavecoeff_comb(reflinelist.values,version,fittype)
    for i in idx:
        # select linelist for this exposure
        condict = dict(exp=i)
        if order is not None:
            condict.update(order=orders)
        linelist1exp = ll.select(condict)
        #reflines = lines[j-1]
        res = compare.from_coefficients(linelist1exp.values,coeffs,
                                              fittype=fittype,
                                              version=version,
                                              sigma=sigma,
                                              **kwargs)
        data[i]['mean']  = res[0]
        data[i]['sigma'] = res[1]
        hf.update_progress((i+1)/nexp,'coeff')

    return data