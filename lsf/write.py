#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:28:20 2023

@author: dmilakov
"""
import harps.settings as hs
import os
from fitsio import FITS

def lsf_to_file(data,filename,extname,version=None,
                  dirpath=None,overwrite=False):
    dirpath=dirpath if dirpath is not None else hs.get_dirname('lsf')
    filepath = os.path.join(dirpath,filename)
    with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
        status = 'failed'
        try:
            hdu['extname'].append(data)
            action = 'append'
            status = 'done'
        except:
            hdu.write(data,extname=extname,extver=version)
            action = 'write'
            status = 'done'
        finally:
            hdu.close()
            print(f"Data {action} to {filepath} {status}.")
    
    return None
    
    