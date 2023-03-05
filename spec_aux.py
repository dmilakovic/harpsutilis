#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:08:48 2023

@author: dmilakov
"""
import numpy as np

default = 701
def extract_item(item):
    """
    utility function to extract an "item", meaning
    a extension number,name plus version.
    
    To be used with partial decorator
    """
    ver=default
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
    
    return ext,ver,ver_sent
def item_to_version(item=None):
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
    assert default > 100 and default <1000, "Invalid default version"
    ver = default
    polyord,gaps,segment=version_to_pgs(item)
    if item is None:
        # polyord,gaps,segment=version_to_pgs(item)
        ver     = int("{2:2d}{1:1d}{0:1d}".format(segment,gaps,polyord))
    # else:
    #     if item<=10:
    #         ver = item
    #     else:
    #         ver = int("{2:2d}{1:1d}{0:1d}".format(segment,gaps,polyord))
        
    #polyord,gaps,segment = [int((default/10**x)%10) for x in range(3)][::-1]
    # polyord,gaps,segment = version_to_pgs(item)
    # #print("item_to_version",item, type(item))
    elif isinstance(item,dict):
        polyord = item.pop('polyord',polyord)
        gaps    = item.pop('gaps',gaps)
        segment = item.pop('segment',segment)
        ver     = int("{2:2d}{1:1d}{0:1d}".format(segment,gaps,polyord))
    elif isinstance(item,(int,np.integer)):
        if item<100:
            ver = item
        else:
            polyord,gaps,segment=version_to_pgs(item)
            ver     = int("{2:2d}{1:1d}{0:1d}".format(segment,gaps,polyord))
    elif isinstance(item,tuple):
        polyord = item[0]
        gaps    = item[1]
        segment = item[2]
        ver     = int("{2:2d}{1:1d}{0:1d}".format(segment,gaps,polyord))
    return ver
def version_to_pgs(ver):  
    
    #print('version_to_pgs',ver,type(ver))
    
    if isinstance(ver,(int,np.integer)) and ver<=100:
        polyord = 1
        gaps    = 0
        segment = 0
    elif isinstance(ver,(int,np.integer)) and ver>100 and ver<3000:
        dig = np.ceil(np.log10(ver)).astype(int)
        split  = np.flip([int((ver/10**x)%10) for x in range(dig)])
        if dig==3:
            polyord, gaps, segment = split
        elif dig==4:
            polyord = np.sum(i*np.power(10,j) for j,i \
                             in enumerate(split[:2][::-1]))
            gaps    = split[2]
            segment = split[3]
    else:
        polyord,gaps,segment = version_to_pgs(default)
    return polyord,gaps,segment