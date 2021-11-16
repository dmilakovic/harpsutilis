#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:16:33 2020

@author: dmilakov
"""

import harps.functions as hf
import harps.wavesol as ws
import harps.lines as hl
import harps.background as bkg
from fitsio import FITS
import numpy as np

c  = 299792458
fr = 18e9
f0 = 7.35e9

hdu = FITS('/Users/dmilakov/harps/espresso/data/'
           'Calibration_S2D_BLAZE_LFC_LFC_A_2020-02-21T13-19-55.322.fits')

spec2d = hdu[1].read()
od = 100

spec1d=spec2d[od]

maxima,minima=hf.detect_minmax(spec1d,window=5)
pix1d_lfc=maxima[0]
N = len(pix1d_lfc)

wave1d_lfc = np.array([c/(f0+n*fr)*1e10 for n in range(30282+N,30282,-1)])

poly=np.polyfit(pix1d_lfc,wave1d_lfc,9)

wave1d=np.polyval(poly,np.arange(len(spec1d)))

error1d=np.sqrt(spec1d)
bkg1d_spline=bkg.getbkg(spec1d,window=5)

linelist=hl.detect_from_array(spec1d,wave1d,fr,f0,error1d,bkg1d_spline,
                              fittype='gauss',plot=True,window=5)

dispersion,coeff=ws.polynomial(linelist,900,full_output=True,npix=len(spec1d))

rsd=ws.residuals(linelist,coeff,300)

plt.figure()