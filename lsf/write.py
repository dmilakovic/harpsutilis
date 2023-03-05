#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:28:20 2023

@author: dmilakov
"""
import harps.settings as hs
import os
from fitsio import FITS

def lsf_to_file(data,filepath,extname,version=None,clobber=False):
    print(filepath)
    # dirpath=dirpath if dirpath is not None else hs.get_dirname('lsf')
    # filepath = os.path.join(dirpath,filename)
    with FITS(filepath,mode='rw',clobber=clobber) as hdu:
        status = 'failed'
        # extver_hasdata = False
        # try:
        #     hdu[extname,version].has_data()
        #     extvar_hasdata = True
        # except:
        #     pass
        
        # if extvar_hasdata & overwrite:
        #     expression = helper_get_expression(data)
        #     rows2del = hdu[extname,version].where()
        
        
        try:
            hdu[extname,version].append(data)
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
    
# def helper_get_expression(data):
#     pairs   = [tuple(entry['order'],entry['segm']) for entry in data]
    
#     expression = ""
#     for od in orders:
#         expression += f" order=={od} &&"
#     for segm in segments:
#         expression += f" segm=={segm} &&"
#     if expression[-2:]=='&&':
#         return expression[:-2]