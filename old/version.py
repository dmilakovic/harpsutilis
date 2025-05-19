#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:52:50 2023

@author: dmilakov
"""
import numpy as np

default_wavesol = 701
default_lsf     = 111

def get_version(ftype,**kwargs):
    if ftype=='wavesol':
        default = default_wavesol
    elif ftype=='lsf':
        default = default_lsf
    

def extract_item(item):
    """
    utility function to extract an "item", meaning
    a extension number and name plus version.
    
    To be used with partial decorator
    """
    # ver=default
    ver_sent=False
    if isinstance(item,tuple):
        ver_sent=True
        nitem=len(item)
        if nitem == 1:
            ext=item[0]
        elif nitem == 2:
            ext,ver=item
            if ver is not None:
                ver = item_to_version(ver)
    else:
        ext=item
        ver=None
    
    return ext,ver,ver_sent
def item_to_version(item=None,ftype=None):
    # IMPORTANT : this function controls the DEFAULT VERSION
    """
    Returns an integer representing the settings provided
    
    Returns the default version if no args provided.
    
    Args:
    -----
    item (int) : version
    
    Returns:
    -------
    version (int): either 1 or a three digit integer in the range 100-511
        If version == 1:
           linear interpolation between individual lines
        If version in 100-511: 
           version = PGS (polynomial order [int], gaps [bool], segmented[bool])
                   
    """
    # assert ftype in ['wavesol','lsf']
    # if ftype == 'wavesol':
    #     default = default_wavesol
    # elif ftype == 'lsf':
    #     default = default_lsf
    
    # assert default > 100 and default <1000, "Invalid default version"
    # ver = default
    # print(ver)
    if item is None:
        # val1, val2, val3 =unpack_integer(default)
        # polyord,gaps,segment=version_to_pgs(item)
        # ver     = int(f"{val1:2d}{val2:1d}{val3:1d}")
        ver = None
    elif isinstance(item,dict):
        val1, val2, val3 = unpack_dictionary(item,ftype)
        ver     = int(f"{val1:2d}{val2:1d}{val3:1d}")
    elif isinstance(item,(int,np.integer)):
        if item<100:
            ver = item
        else:
            val1, val2, val3 = unpack_integer(item)
            ver     = int(f"{val1:2d}{val2:1d}{val3:1d}")
    elif isinstance(item,tuple):
        val1 = item[0]
        val2 = item[1]
        val3 = item[2]
        ver  = int(f"{val1:2d}{val2:1d}{val3:1d}")
    return ver

def unpack_dictionary(item_dict, ftype):
    assert ftype in ['wavesol','lsf']
    if ftype == 'wavesol':
        key1, key2, key3 = ('polyord', 'gaps', 'segment')
        default = default_wavesol
    elif ftype == 'lsf':
        key1, key2, key3 = ('iteration', 'model_scatter', 'interpolate')
        default = default_lsf
        
    val1_, val2_, val3_ = unpack_integer(default)
        
    val1 = int(item_dict.get(key1,val1_))
    val2 = int(item_dict.get(key2,val2_))
    val3 = int(item_dict.get(key3,val3_))
    return val1, val2, val3

def unpack_integer(ver):  
    ver = default_wavesol
    if isinstance(ver,(int,np.integer)) and ver<=100:
        val1 = 1
        val2 = 0
        val3 = 0
    elif isinstance(ver,(int,np.integer)) and ver>100 and ver<3000:
        dig = np.ceil(np.log10(ver)).astype(int)
        split  = np.flip([int((ver/10**x)%10) for x in range(dig)])
        if dig==3:
            val1, val2, val3 = split
        elif dig==4:
            val1 = np.sum(i*np.power(10,j) for j,i \
                             in enumerate(split[:2][::-1]))
            val2    = split[2]
            val3 = split[3]
    return val1, val2, val3
    