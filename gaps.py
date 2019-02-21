#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:36:33 2019

@author: dmilakov
"""
import harps.containers as container
import harps.settings   as hs
from   harps.core import np, os, json
#==============================================================================
    
#                           G A P S    F I L E   
    
#==============================================================================
def read_gaps(filepath=None):
    if filepath is not None:
        filepath = filepath  
    else:
        dirpath = hs.get_dirname('gaps')
        filepath = os.path.join(dirpath,'gaps.json')
    with open(filepath,'r') as json_file:
        gaps_file = json.load(json_file)
    gaps = []
    for block in range(1,4):
        orders  = gaps_file['orders{}'.format(block)]
        norders = len(orders)
        block_gaps = container.gaps(norders)
        block_gaps['order'] = orders
        block_gaps['gaps']  = gaps_file['block{}'.format(block)]
        gaps.append(block_gaps)
    gaps = np.hstack(gaps)
    return np.sort(gaps)
    
    

def get_gaps(order,filepath=None):
    gapsfile  = read_gaps(filepath)
    orders   = np.array(gapsfile[:,0],dtype='i4')
    gaps2d   = np.array(gapsfile[:,1:],dtype='f8')
    selection = np.where(orders==order)[0]
    gaps1d    = gaps2d[selection]
    return np.ravel(gaps1d)


def introduce_gaps(centers,gaps1d,npix=4096):
    if np.size(gaps1d)==0:
        return centers
    elif np.size(gaps1d)==1:
        gap  = gaps1d
        gaps = np.full((7,),gap)
    else:
        gaps = gaps1d
    centc = np.copy(centers)
    
    for i,gap in enumerate(gaps):
        ll = (i+1)*npix/(np.size(gaps)+1)
        cut = np.where(centc>ll)[0]
        centc[cut] = centc[cut]-gap
    return centc

def in_linelist(linelist,fittype='gauss',filepath=None,copy=True,npix=4096):
    gaps2d = read_gaps(filepath)
    orders = np.unique(linelist['order'])
    if copy:
        llist = np.copy(linelist)
    else:
        llist = linelist
    for order in orders:
        gaps1d = gaps2d[np.where(gaps2d['order']==order)]['gaps'][0]
        inord  = np.where(linelist['order']==order)[0]
        cent1d = llist[inord][fittype][:,1]
        cent   = introduce_gaps(cent1d,gaps1d,npix)
        llist[inord][fittype][:,1] = cent
    return llist
def in_residuals(residuals,fittype='gauss',filepath=None,npix=4096,copy=True):
    gaps2d = read_gaps(filepath)
    orders = np.unique(residuals['order'])
    if copy:
        rsdlist = np.copy(residuals)
    else:
        rsdlist = residuals
    for order in orders:
        gaps1d = gaps2d[np.where(gaps2d['order']==order)]['gaps'][0]
        inord  = np.where(residuals['order']==order)[0]
        cent1d = rsdlist[inord][fittype]
        cent   = introduce_gaps(cent1d,gaps1d,npix)
        rsdlist[fittype][inord] = cent
    return rsdlist