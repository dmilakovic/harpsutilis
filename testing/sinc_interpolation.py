#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:18:44 2023

@author: dmilakov
"""
import harps.spectrum as hc
from fitsio import FITS
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sinc

#%%
spec=hc.HARPS('/Users/dmilakov/projects/lfc/data/harps/e2ds/2018-12/2018-12-03/HARPS.2018-12-04T00:18:00.981_e2ds_A.fits',fr=18e9,f0=4.58e9)


flux=spec.flux
hdul=FITS('/Users/dmilakov/projects/lfc/dataprod/fits/v_2.0/HARPS.2018-12-04T00:18:00.fits')
llist=hdul['linelist'].read()
#%%
numline = 3000
od = llist[numline]['order']
pixl = llist[numline]['pixl']
pixr = llist[numline]['pixr']
bary = llist[numline]['bary']
linef=flux[od,pixl:pixr]
linex=np.arange(pixl,pixr)
plt.plot(linex,linef)
#%%
def pulse(x,shift,amplitude):
    return amplitude*sinc((x-shift))

def vpulse(*args,**kwargs):
    return np.vectorize(pulse)(*args,**kwargs)

def sum_pulses(xarray,samples_x,samples_y):
    return np.sum([vpulse(xarray,sx,sy) for sx,sy in zip(samples_x,samples_y)],axis=0)

def shift_line(xarray,yarray):
    bary = np.average(xarray,weights=yarray)
    return xarray-bary, sum_pulses(xarray,xarray,yarray)
   
#%%
def process_line(llist,numline): 
    od = llist[numline]['order']
    pixl = llist[numline]['pixl']
    pixr = llist[numline]['pixr']
    f_star = llist[numline]['gauss_pix_integral']
    linef=flux[od,pixl:pixr]/f_star
    linex=np.arange(pixl,pixr)
    return shift_line(linex,linef)
#%%
plt.scatter(linex-bary,linef,marker='o')

# for i in range(len(linex)):
#     plt.plot(linex-bary,vpulse(linex,linex[i],linef[i]))

line_x, line_y = shift_line(linex,linef)
# plt.plot(linex-bary,sum_pulses(linex-bary, linex-bary, linef))
plt.plot(line_x, line_y)
#%%
for n in np.arange(3000,3030):
    line_x, line_y = process_line(llist,n)
    plt.scatter(line_x,line_y,s=2)
    
    
    