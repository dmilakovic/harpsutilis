#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:17:38 2018

@author: dmilakov
"""

from harps.core import np

def upper(strlist):
    outlist = []
    for entry in strlist:
        if type(entry)==str:
            outlist.append(str.upper(entry))
        else:
            outlist.append(entry)
    return outlist

#def get_index(hdu,args):
#    if isinstance(args,dict):
#        return index_from_dict(hdu,args)
#    elif isinstance(args,str):
#        return index_from_extname(hdu,args)
#    elif isinstance(args,tuple):
#        hdu0    = hdu
#        extname = args[0]
#        keyvals = args[1]
#        match_extname = index_from_extname(hdu0,extname)
#        hdu1    = hdu0[match_extname]
def 
def index_from_dict(hdu,keyvals):
    """
    Returns the index of the HDU that contains keywords and values provided in 
    keyvals.
    
    Args:
    ----
        hdu (fitsio.FITS) 
        keyvals (dict) : contains keys and values to look for
    Returns:
    -------
        i (int) : index of the HDU
        
        or
        
        None 
    """
    keys0 = upper(list(keyvals.keys()))
    vals0 = upper(list(keyvals.values()))
    keyvals0 = {key:val for key,val in zip(keys0,vals0)}
    indices = []
    for i,h in enumerate(hdu):
        header = h.read_header()
        keys   = upper(list(header.keys()))
        setint = np.intersect1d(keys0,keys)
        if len(setint)>0:
            vals  = upper([header[key] for key in setint])
            vals0 = [keyvals0[key] for key in setint]
            print(vals0,vals)
            good  = np.all(vals==vals0)
            if good:
                indices.append(i)
            else:
                continue
        else:
            continue
    return indices

def index_fom_extname(hdu,extname):
    hdu_extnames  = [h.get_extname() for h in hdu]
    indices       = [i for i,key in enumerate(hdu_extnames) if key==extname]
    return indices