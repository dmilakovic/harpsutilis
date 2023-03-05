#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:59:44 2023

@author: dmilakov
"""


import harps.spectrum as hc
import harps.lsf.construct as construct
import harps.lsf.aux as aux
import harps.io as hio
#%%
filename = '/Users/dmilakov/projects/lfc/data/harps/e2ds/2018-12/2018-12-04/'+\
            'HARPS.2018-12-05T08:12:52.040_e2ds_A.fits'
spec=hc.HARPS(filename,f0=4.68e9,fr=18e9,overwrite=False)
lsf_filepath=hio.get_fits_path('lsf',spec.filepath)

#%%
spec.process(fittype='gauss')
#%%
order=54
for it in range(1,2,1):
    lsf1d=construct.from_spectrum_1d(spec,order=order,iteration=it,scale='pixel',
                                 iter_center=5,numseg=16,model_scatter=True,
                                 save_fits=True,clobber=False,interpolate=True,
                                 update_linelist=True)