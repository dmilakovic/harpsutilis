#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:33:59 2018

@author: dmilakov
"""
import numpy as np
import matplotlib.pyplot as plt
from harps.wavesol import ThAr
#%%
wave0=FITS('/Users/dmilakov/harps/server/2015-04-17/HARPS.2015-04-15T19:34:22.221_wave_A.fits')[0].read()
wave1=FITS('/Users/dmilakov/harps/server/2015-04-17/HARPS.2015-04-16T18:48:48.969_wave_A.fits')[0].read()

fig = plt.figure(figsize=(10,10))
im = plt.imshow((wave1-wave0)/wave0*299729458,aspect='auto',vmin=-50,vmax=50)
plt.colorbar(im)
#%%
thar0=ThAr('/Users/dmilakov/harps/server/2015-04-17/HARPS.2015-04-15T19:34:22.221_e2ds_A.fits',False)()
thar1=ThAr('/Users/dmilakov/harps/server/2015-04-17/HARPS.2015-04-16T18:48:48.969_e2ds_A.fits',False)()
fig2 = plt.figure()
im2 = plt.imshow((thar1-thar0)/thar0*299729458,aspect='auto',vmin=-50,vmax=50)
plt.colorbar(im2)
#%%
xrange = (-50,50)
plt.figure(figsize=(9,9))
plt.hist(np.ravel((thar1-thar0)/thar0*299729458),bins=50,range=xrange)
plt.hist(np.ravel((wave1-wave0)/wave0*299729458),bins=50,range=xrange)
#%%
plt.figure(figsize=(16,8))
plt.hist(np.ravel(np.diff(wave0)),bins=200)

plt.figure(figsize=(16,8))
plt.hist(np.ravel(np.diff(thar0)),bins=200)

#%%
plt.figure(figsize=(12,9))
plt.hist(np.diff(wave0)[50],bins=200)
plt.hist(np.diff(thar0)[50],bins=200)
#%%
plt.figure(figsize=(12,9))
plt.plot(np.diff((wave1-wave0)/wave0*299792458)[50],bins=200)
plt.figure(figsize=(12,9))
plt.plot(np.diff((wave1-wave0)/wave0*299792458)[50],marker='o')
plt.figure(figsize=(12,9))
plt.plot(np.diff((wave1-wave0)/wave0*299792458)[50],marker='o',ms=3,lw=0.5)
plt.figure(figsize=(12,9))
plt.plot(np.diff((thar1-thar0)/thar0*299792458)[50],marker='o',ms=3,lw=0.5)