#!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import sys
from astropy.io import fits
import warnings
import scipy.constants as const
from scipy import interpolate
import gc
import datetime

from scipy.optimize import curve_fit
import pandas as pd
import lmfit
from lmfit.models import GaussianModel
from scipy.optimize import minimize
from peakdetect import peakdetect
import xarray as xr
from joblib import Parallel,delayed
import h5py

from scipy.special import erf,erfc
from scipy.linalg import svd
from scipy.optimize import curve_fit, minimize, leastsq, least_squares, OptimizeWarning, fmin_ncg
from scipy.optimize._lsq.least_squares import prepare_bounds
## first and last order in a spectrum
sOrder = 45    
eOrder = 72
nOrder = eOrder - sOrder
nPix   = 4096
##

class Spectrum(object):
    ''' Spectrum object contains functions and methods to read data from a 
        FITS file processed by the HARPS pipeline
    '''
    def __init__(self,filepath=None,ftype='e2ds',
                 header=True,data=True,LFC='HARPS'):
        '''
        Initialise the spectrum object.
        '''
        self.filepath = filepath
        self.name = "HARPS Spectrum"
        self.ftype = ftype
        self.hdulist = []
        self.data    = []
        self.header  = []
        self.bad_orders = []
        self.LFC     = LFC
        self.wavesol = []
        self.wavesol_thar = []
        self.wavesol_LFC = []
        self.fr_source = 250e6 #Hz
        self.f0_source = -50e6 #Hz
        
        self.gapsfile   = np.load("/Users/dmilakov/harps/data/gapsA.npy")
        gaps            = np.zeros(shape=(eOrder+1,7))
        gorders         = np.array(self.gapsfile[:,0],dtype='i4')
        gaps[gorders,:] = np.array(self.gapsfile[:,1:],dtype='f8')
        self.gaps       = gaps
        if header == True:
            self.__read_meta__()
        else:
            pass
        if data == True:
            self.__read_data__()
#            self.norders = self.data.shape[0]
            self.sOrder  = sOrder
        else:
            pass
    def __check_and_load__(self):
        ''' Method to read the header and the data, if not done previously. '''
        try:
            if len(self.data)==0:
                self.__read_data__()
        except:
            pass
        try:
            if len(self.header)==0:
                self.__read_meta__()
        except:
            pass
        return
    def __read_meta__(self):
        ''' Method to read header keywords and save them as properties of the 
            Spectrum class'''	
        self.hdulist = fits.open(self.filepath,memmap=False)
        self.header  = self.hdulist[0].header
        self.npix    = self.header["NAXIS1"]
        self.nbo     = self.header["HIERARCH ESO DRS CAL LOC NBO"]
        self.conad   = self.header["HIERARCH ESO DRS CCD CONAD"]
        
        try:
            self.d   = self.header["HIERARCH ESO DRS CAL TH DEG LL"]
        except:
            try:
                self.d  = self.header["HIERARCH ESO DRS CAL TH DEG X"]
            except:
                self.d  = 3
                print(self.filepath)
                warnings.warn("Something wrong with this file, skipping")
                pass
        self.date    = self.header["DATE"]
        self.exptime = self.header["EXPTIME"]
        try:
            #offset frequency of the LFC, rounded to 1MHz
            self.anchor = round(self.header["HIERARCH ESO INS LFC1 ANCHOR"],-6) 
            #repetition frequency of the LFC
            self.reprate = self.header["HIERARCH ESO INS LFC1 REPRATE"]
        except:
            pass
        if self.LFC=='HARPS':
            self.modefilter   = 72
            self.f0_source    = -50e6 #Hz
            self.reprate      = self.modefilter*250e6 #Hz
        elif self.LFC=='FOCES':
            self.modefilter   = 100
            self.f0_source    = 20e6 #Hz
            self.reprate      = self.modefilter*250e6 #Hz
            self.anchor       = round(288.08452e12,-6) #Hz
        self.omega_r = 250e6
        m,k            = divmod(
                            round((self.anchor-self.f0_source)/self.fr_source),
                                   self.modefilter)
        self.f0_comb   = (k)*self.fr_source + self.f0_source
        # Fibre information is not saved in the header, but can be obtained 
        # from the filename 
        self.fibre   = self.filepath[-6]
    def __read_data__(self,convert_to_e=True):
        ''' Method to read data from the FITS file
        Args:
        ---- 
            convert_to_e : convert the flux to electron counts'''
        if len(self.hdulist)==0:
            self.hdulist = fits.open(self.filepath,memmap=False)
        if   self.ftype=="s1d" or self.ftype=="e2ds":
            data = self.hdulist[0].data.copy()#[sOrder:eOrder]
        elif self.ftype=="":
            data = self.hdulist[1].data.copy()#[sOrder:eOrder]
        
        if convert_to_e is True:
            data = data * self.conad
            self.data_units = "e-"
        else:
            self.data_units = "ADU"
        self.data = data
        return self.data
    def __get_wavesol__(self,calibrator="ThAr",nobackground=True,vacuum=True,
                        orders=None,method='curve_fit',patches=False,gaps=True,
                        polyord=3,**kwargs):
        '''Function to get the wavelength solution.
        Lambda (order, pixel) = Sum{i=0,d} [A(i+order*(d+1))*x^i]
        
        NOTE: Coefficients of the wavelenegth solution in the FITS header file 
        are for wavelengths in air! 
        It is necessary to transform them to vacuum wavelengths by calculating 
        the refractive index of air under standard conditions.
        This is done by the program by default.
        
        Args:
            calibrator: String specifying the calibration method. Options are 
                'ThAr' and 'LFC'.
            nobackground: Boolean, optional, default: False.
                If true, the background is subtracted.
            vacuum: Boolean, optional, default: True. 
                If true, vacuum wavelengths are used. Otherwise, air 
                wavelengths are used.
            LFC: String, optional, default: 'HARPS'. 
                Options are 'HARPS' and 'FOCES'.
            orders : List of integers specifying the echelle orders for 
                calibration, default=None. If not None, calibration will be 
                performed only for specified orders. 
            method: String specifying the method to be used for fitting LFC 
                lines. Options are 'curve_fit', 'lmfit', 'chisq'. 
                Default: 'curve_fit'.
            patches: Boolean. If true, fitting of the wavelength solution is 
                performed in 512-pixel patches. Default: false.
            gaps: Boolean. If true, gaps are introduced in the detected 
                
        Returns:
            wavesol: A 1D or 2D numpy array containing the wavelength solution 
                for all available (or selected) echelle orders in the spectrum. 
            The wavelength solution is also saved into the Spectrum object. It
            is saved as attribute 'wavesol_thar' and 'wavesol_LFC' in the cases
            of 'ThAr' and 'LFC', respectively. 
        '''
        def patch_fit(patch,polyord=3,method='curve_fit'):
            data      = patch.lbd
            x         = patch.pix
                          
            data_axis = np.array(data.values,dtype=np.float64)
            x_axis    = np.array(x.values,dtype=np.float64)
            datanan,xnan = (np.isnan(data_axis).any(),np.isnan(x_axis).any())
            if (datanan==True or xnan==True):
                print("NaN values in data or x")
            if x_axis.size>polyord:
                coef = np.polyfit(x_axis,data_axis,polyord)
                if method == 'curve_fit':
                    coef,pcov = curve_fit(polynomial,x_axis,data_axis,p0=coef[::-1])
                    coef = coef[::-1]
            else: 
                coef = None  
            return coef
        
        def fit_wavesol(pix,lbd,patches=True,polyord=3):
            if patches==True:
                npt = 8
            else:
                npt = 1
            ps = 4096/npt
            
            
            cc     = pd.concat([lbd,pix],
                                axis=1,keys=['lbd','pix'])
            
            cc = cc.dropna(how='any').reset_index(drop=True)
            ws     = np.zeros(4096)
            cf = np.zeros(shape=(npt,polyord+1))
            rs = pd.Series(index=pix.index)
            
            
            for i in range(npt):
                ll,ul     = np.array([i*ps,(i+1)*ps],dtype=np.int)
                patch     = cc.where((cc.pix>=ll)&
                                     (cc.pix<ul)).dropna()
                
                if patch.size>polyord:
                    pixels    = np.arange(ll,ul,1,dtype=np.int)
                    coef      = patch_fit(patch,polyord)
                    if coef is not None:
                        fit       = np.polyval(coef,patch.pix)
                        
                        residuals = np.array(patch.lbd.values-fit,dtype=np.float64)
                        rs.iloc[patch.index]=residuals
                        outliers  = is_outlier(residuals,5)
                        if np.any(outliers)==True:
                            patch['outlier']=outliers
                            newpatch = patch.where(patch.outlier==False).dropna(how='any')
                            coef = patch_fit(newpatch,polyord) 
                        cf[i,:]=coef
                else:
                    ws[ll:ul] = np.nan
                    
                try:
                    ws[ll:ul] = np.polyval(coef,pixels)
                except:
                    ws[ll:ul] = np.nan
            fit = np.polyval(coef,pix)
#            print(rs,fit)
            # residuals are in m/s
            rs = rs/fit*299792458
            
            return ws,cf,rs

            
        def _to_vacuum(lambda_air):
            ''' Returns vacuum wavelengths.
            
            Args:    
                lambda_air: 1D numpy array
            Returns:
                lambda_vacuum : 1D numpy array
            '''
            if lambda_air.sum()==0:
                return
            pressure = 760.0
            temp     = 15
            index    = 1e-6*pressure*(1.0+(1.049-0.0157*temp)*1e-6*pressure) \
                        /720.883/(1.0+0.003661*temp) \
                        *(64.328+29498.1/(146.0-2**(1e4/lambda_air)) \
                        +255.4/(41.0-2**(1e4/lambda_air)))+1.0
            lambda_vacuum = lambda_air*index
            return lambda_vacuum
            
        
        if orders is None:
            if calibrator == "ThAr":
                orders = np.arange(0,self.nbo,1)
            if calibrator == "LFC":
                orders = np.arange(self.sOrder,self.nbo,1)
        def _get_wavecoeff_air():
            ''' 
            Returns coefficients of a third-order polynomial from the FITS file 
            header in a matrix. This procedure is described in the HARPS DRS  
            user manual.
            https://www.eso.org/sci/facilities/lasilla/
                    instruments/harps/doc/DRS.pdf
            '''
            wavecoeff    = np.zeros(shape = (self.nbo, self.d+1, ), 
                                    dtype = np.float64)
            self.bad_orders = []
            for order in orders:
                # Try reading the coefficients for each order. If failed, 
                # classify the order as a 'bad order'.
                for i in range(self.d+1):                    
                    ll    = i + order*(self.d+1)
                    try:
                        coeff = self.header["ESO DRS CAL TH COEFF LL{0}".format(ll)]
                    except:
                        coeff = 0
                    if coeff==0:                         
                        if order not in self.bad_orders:
                            self.bad_orders.append(order)
                    wavecoeff[order,i] = coeff
            return wavecoeff
        def _get_wavecoeff_vacuum():
            ''' 
            Returns coefficients of the third-order polynomial for vacuum.
            '''
            wavecoeff    = np.zeros(shape = (self.nbo, self.d+1, ), 
                                    dtype = np.float64)
            for order in orders:
                wavecoeff_air            = self.wavecoeff_air[order]
                wavecoeff_vac,covariance = curve_fit(polynomial, 
                                                     np.arange(self.npix), 
                                                     self.wavesol_thar[order], 
                                                     p0=wavecoeff_air)
                wavecoeff[order]         = wavecoeff_vac
            return wavecoeff
            
            
        self.__check_and_load__()
        
        # wavesol(72,4096) contains wavelengths for 72 orders and 4096 pixels
        
        if calibrator is "ThAr": 
            # If this routine has not been run previously, read the calibration
            # coefficients from the FITS file. For each order, derive the 
            # calibration in air. If 'vacuum' flag is true, convert the 
            # wavelengths to vacuum wavelengths for each order. 
            # Finally, derive the coefficients for the vacuum solution.
            if (len(self.wavesol_thar)== 0 or self.wavesol_thar.sum()==0):
                wavesol_thar = np.zeros(shape=(self.nbo,self.npix,), 
                                        dtype=np.float64)
                self.wavecoeff_air = _get_wavecoeff_air()
                for order in orders:
                    if self.is_bad_order(order)==False:
                        wavesol_air = np.array(
                                        [np.sum(self.wavecoeff_air[order,i]*pix**i 
                                                for i in range(self.d+1)) 
                                        for pix in range(0,self.npix,1)])
                        if vacuum is True:
                            wavesol_thar[order] = _to_vacuum(wavesol_air)
                            
                        else:
                            wavesol_thar[order] = wavesol_air
                    else:
                        wavesol_thar[order] = np.zeros(self.npix)
                self.wavesol_thar = wavesol_thar
                if vacuum is True:
                    self.wavecoeff_vacuum = _get_wavecoeff_vacuum()
                
            # If this routine has been run previously, there's nothing to do
            else:
                wavesol_thar = self.wavesol_thar
                pass
            
            
        if calibrator == "LFC":
            #print(orders)           
            # Calibration for each order is performed in two steps:
            #   (1) Fitting LFC lines in both pixel and wavelength space
            #   (2) Dividing the 4096 pixel range into 8x512 pixel patches and
            #       fitting a 3rd order polynomial to the positions of the 
            #       peaks
            wavesol_LFC  = np.zeros(shape = (self.nbo,self.npix,), 
                                    dtype = np.float64)
            # Save coeffiecients of the best fit solution
            if patches==True:
                npt = 8
            elif patches==False:
                npt = 1
            if npt == 1:
                wavecoef_LFC = np.zeros(shape = (self.nbo,polyord+1), 
                                        dtype = np.float64)
            else:
                wavecoef_LFC = np.zeros(shape = (self.nbo,polyord+1,npt), 
                                        dtype = np.float64)
            # Save positions of lines
            cc_data      = xr.DataArray(np.full((nOrder,3,500),np.NaN),
                                        dims=['od','typ','val'],
                                        coords=[np.arange(sOrder,eOrder),
                                                ['wave','pix','photon_noise'],
                                                np.arange(500)])
            # Save residuals to the fit
            rsd          = xr.DataArray(np.full((nOrder,500),np.NaN),
                                        dims=['od','val'],
                                        coords=[np.arange(sOrder,eOrder),
                                                np.arange(500)])
            # First, check if the user provided a wavelength solution or 
            # air/vacuum coefficients to be used for LFC calibration. If not 
            # provided, read in ThAr wavelength calibration from the FITS file.
            try:
                self.wavesol_thar     = kwargs['wavesol_thar']
                try:
                    self.wavecoeff_air = kwargs['wavecoeff_air']
                except:
                    self.wavecoeff_air = _get_wavecoeff_air()
                self.wavecoeff_vacuum = _get_wavecoeff_vacuum()
            except:
                self.wavesol_thar=[]
                self.__get_wavesol__(calibrator="ThAr",vacuum=True,oders=None)
                pass
            # Check if the sum of the ThAr solution is different from zero. 
            # If the sum is equal to zero, repeat the ThAr calibration
            # (BUG?)
            if self.wavesol_thar.sum()==0:
                self.__get_wavesol__(calibrator="ThAr",vacuum=True,
                                     orders=None,**kwargs)
            
            # Some series had a different anchor frequency. Additional argument
            # can be passed to the function to tell the program to shift the
            # anchor frequency by a certain amount.
            try:
                anchor_offset = kwargs['anchor_offset']
                #print("Anchor offset = ", anchor_offset)
                self.anchor   = self.anchor+anchor_offset
            except:
                pass
            
            #print(self.LFC, "{}GHz".format(self.f0_comb/1e9))
            for order in orders:
                # Check if the order is listed among bad orders. 
                # If everything is fine, fit the lines of the comb in both 
                # wavelength and pixel space. Every 512th pixel is larger than 
                # the previous 511 pixels. Therefore, divide the entire 4096
                # pixel range into 8 512 pixel wide chunks and fit a separate 
                # wavelength solution to each chunk.
#                print("ORDER = {}".format(order))
                LFC_wavesol = np.zeros(self.npix)
                if self.is_bad_order(order):
                    wavesol_LFC[order] = LFC_wavesol
                    continue
                else:
                    pass
                cc     = self.fit_lines(order,method=method)
                cc_pix = cc['pixel'].dropna()
                cc_lbd = cc['wave'].dropna()
                
                # Include the gaps
                if gaps is True:
                    g0 = self.gaps[order,:]
                    XX = self.introduce_gaps(cc_pix['center'],g0)
                    cc_pix['center'] = XX
                elif gaps is False:
                    pass
                
                cc_data.loc[dict(typ='wave',od=order)][cc_lbd.index.values] = cc_lbd.center_th.values
                cc_data.loc[dict(typ='pix',od=order)][cc_pix.index.values] = cc_pix.center.values   
                cc_data.loc[dict(typ='photon_noise',od=order)][cc_pix.index.values] = cc_pix.photon_noise.values   
                LFC_wavesol,coef,residuals = fit_wavesol(cc_pix['center'],
                                               cc_lbd.center_th,
                                               patches=patches,
                                               polyord=polyord)
                
                wavesol_LFC[order]  = LFC_wavesol
                wavecoef_LFC[order] = coef           
                rsd.loc[dict(od=order)][cc_lbd.index.values] = residuals
            self.wavesol_LFC  = wavesol_LFC
            self.cc_data      = cc_data
            self.wavecoef_LFC = wavecoef_LFC
            self.residuals    = rsd
        #self.wavesol = wavesol
        if calibrator is "ThAr":
            return wavesol_thar
        elif calibrator is "LFC":
            return wavesol_LFC
    def calculate_fourier_transform(self,**kwargs):
        try:    orders = kwargs["order"]
        except: orders = np.arange(self.sOrder,self.nbo,1)
        n       = (2**2)*4096
        freq    = np.fft.rfftfreq(n=n, d=1)
        uppix   = 1./freq
        # we only want to use periods lower that 4096 pixels 
        # (as there's no sense to use more)
        cut     = np.where(uppix<=nPix)
        # prepare object for data input
        datatypes = Datatypes(nFiles=1,
                              nOrder=self.nbo,
                              fibre=self.fibre).specdata(add_corr=True)
        datafft   = np.zeros(shape=uppix.shape, dtype=datatypes.ftdata)
        dtype     = datatypes.names
        for i,o in enumerate(orders): 
            try:
                data = self.data[o]
                env  = self.get_envelope1d(o)
                bkg  = self.get_background1d(o)
                b2e  = bkg/env
                fmb  = data-bkg
                for f in list(self.fibre):
                    datafft[f]["FLX"][:,0,i] = np.fft.rfft(data,n=n)
                    datafft[f]["ENV"][:,0,i] = np.fft.rfft(env,n=n)
                    datafft[f]["BKG"][:,0,i] = np.fft.rfft(bkg,n=n)
                    datafft[f]["B2E"][:,0,i] = np.fft.rfft(b2e,n=n)
                    datafft[f]["FMB"][:,0,i] = np.fft.rfft(fmb,n=n)
            except:
                pass
        self.datafft = datafft[cut]
        self.freq    = uppix[cut]    
    def calculate_photon_noise(self,order=None,return_array=False):
        ''' Calulates the photon noise for the entire exposure. 
        
        Calculates the photon noise for entire frame by method of Bouchy 2003.
        If echelle order is provided, returns the photon noise for the order. 
        Also, if return_array=True, returns a 1D or 2D array containing the 
        photon noise value for each pixel.
        
        Args:
            order: Integer number of the echelle order.
            return_array: Boolean, optional, default=False
        Returns:
            if not return_array:
                photon_noise: Photon noise across the entire frame (if no order
                is specified) or a single echelle order (if order is specified).
            if return_array:
                (photon_noise, photon_noise_array)
        '''
        try:
            weights2d       = self.weights2d
        except:
            weights2d       = self.get_weights2d()
        photon_noise2d      = np.zeros(shape=self.weights2d.shape)
        photon_noise1d      = np.zeros(shape=(self.nbo,))
        # Bouchy Equation (10)
        self.photon_noise2d = 299792458e0/np.sqrt(weights2d)
        for o in range(sOrder,self.nbo,1):
            photon_noise1d[o] = 299792458e0/np.sqrt(weights2d[o].sum())
#            print(o,photon_noise1d[o])
            
        # Remove nan values
        nan_values   = np.where(np.isnan(photon_noise1d))
        photon_noise1d[nan_values] = 0e0
        good_values  = np.where(photon_noise1d!=0)
        
        self.photon_noise1d = photon_noise1d
        #print(self.photon_noise1d.shape, self.photon_noise1d)
        
        
        photon_noise1d      = self.photon_noise1d[good_values]
        # Bouchy Equation (13)
        self.photon_noise   = 1./np.sqrt((photon_noise1d**-2).sum())
        if order is not None:
            if ((type(order)==np.int64)|(type(order)==int)):
                weights1d      = weights2d[order]
                photon_noise1d_1o = 1./np.sqrt(weights1d.sum())*299792458e0
                photon_noise1d_1o_array = 1./np.sqrt(weights1d)*299792458e0
                if not return_array:
                    return photon_noise1d_1o
                else:
                    return (photon_noise1d_1o, photon_noise1d_1o_array)
            elif type(order) is list:
                photon_noise_subdata = self.photon_noise1d[order]
                photon_noise_sub = 1./np.sqrt((photon_noise_subdata**-2).sum())
                if not return_array:
                    return photon_noise_sub
                else:
                    return (photon_noise_sub,photon_noise_subdata)
        else:
            if not return_array:
                return self.photon_noise
            else:
                return (self.photon_noise,self.photon_noise2d)
                
                    
    def extract1d(self,order,scale='pixel',nobackground=False,
                  vacuum=True,**kwargs):
        """ Extracts the 1D spectrum of a specified echelle order from the
            FITS file.
        
            Args:
                order: Integer number of the echelle order in the FITS file.
                scale : String to determine the x-axis, optional, 
                    default: 'pixel'. Options are 'pixel' and 'wave'.
                nobackground: Boolean, optional, default: False.
                        If true, the background is subtracted.
                vacuum: Boolean, optional, default: True. 
                        If true, vacuum wavelengths are used. Otherwise, air 
                        wavelengths are used.
            Returns:
                spec1d : Dictionary with two numpy 1D-arrays.
                
                { scale: wavelengths[A] / pixels,
                 'flux': counts}
        """
        self.__check_and_load__()
        #print(self.filepath,order,scale)
        try: 
            calibrator   = kwargs["calibrator"]
        except:
            calibrator   = "ThAr"
        #print("extract1d",calibrator)
        try:
            self.wavesol = kwargs["wavesol"]
            #print("Wavesolution provided")
        except:    
            if   len(self.wavesol_thar)!=0:
                #print("Existing thar wavesolution")
                wavesol = self.wavesol_thar
            elif len(self.wavesol_thar)==0:
                #print("No existing thar wavesolution")
                wavesol = self.__get_wavesol__(calibrator,orders=[order],
                                               vacuum=vacuum)
          
        wave1d  = pd.Series(wavesol[order])
        pix1d   = pd.Series(np.arange(wave1d.size))
        flux1d  = pd.Series(self.data[order])
        # Assuming the error is simply photon noise
        error1d = pd.Series(np.sqrt(self.data[order]))
        
                
        if nobackground is True:
            if   scale == 'pixel':
                xarray1d = pix1d
            elif scale == 'wave':
                xarray1d = wave1d
            kind      = 'spline'
            minima    = peakdet(flux1d, xarray1d, extreme="min")
            xbkg,ybkg = minima.x, minima.y
            if   kind == "spline":
                coeff       = interpolate.splrep(xbkg, ybkg)
                background  = interpolate.splev(xarray1d,coeff) 
            elif kind == "linear":
                coeff      = interpolate.interp1d(xbkg,ybkg)
                mask       = np.where((xarray1d>=min(xbkg))&
                                      (xarray1d<=max(xbkg)))[0]
                background = coeff(xarray1d[mask])
            flux1d     = flux1d - background
        spec1d  = pd.DataFrame(np.array([pix1d,wave1d,flux1d,error1d]).T,
                               columns=['pixel','wave','flux','error'])
        return spec1d
    def extract2d(self,scale='pixel'):
        """ Extracts the 2D spectrum from the FITS file.
        
            Args:
                scale : String to determine the x-axis. 
                        Options are 'pixel' and 'wave'.
            Returns:
                spec2d : Dictionary with two numpy 2D-arrays.
                
                { scale: wavelengths[A] / pixels,
                 'flux': counts}
        """
        self.__check_and_load__()
        if scale=="wave":
            self.wavesol = self.__get_wavesol__(calibrator="ThAr")
        else:
            pass
        if   scale=='wave':
            wave2d  = self.wavesol
            flux2d  = self.data
            #spec1d = np.stack([wave1d,flux1d])
            spec2d  = dict(wave=wave2d, flux=flux2d)
        elif scale=='pixel':
            pixel2d = np.mgrid[0:np.size(self.data):1, 
                               0:np.size(self.data):1].reshape(2,-1).T
            flux2d  = self.data
            spec2d  = dict(pixel=pixel2d, flux=flux2d)
        return spec2d

    def fit_lines(self,order,nobackground=True,method='erfc'):
        """Fits LFC lines of a single echelle order.
        
        Extracts a 1D spectrum of a selected echelle order and fits a single 
        Gaussian profile to each line, in both wavelength and pixel space. 
        
        Args:
            order: Integer number of the echelle order in the FITS file.
            nobackground: Boolean determining whether background is subtracted 
                before fitting is performed.
            method: String specifying the method to be used for fitting. 
                Options are 'curve_fit', 'lmfit', 'chisq'.
        Returns:
            A dictionary with two pandas DataFrame objects, each containing 
            parameters of the fitted lines. For example:
            
            {'wave': pd.DataFrame(amplitude,center,sigma),
             'pixel: pd.DataFrame(amplitude,center,sigma)}
        """
        
        
        def _remove_poor_fits(input_lines):
            """ Removes poorly fitted lines from the list of fitted lines.
            
            Identifies outliers in sigma parameter and removes those lines from 
            the list.
            
            Args:
                input_lines: Dictionary returned by fit_lines.
            Returns:
                output_lines: Dictionary returned by fit_lines.
            """
            
            xi = []
            output_lines = {}
            for scale,df in input_lines.items():
                sigma = np.array(df.sigma.values,dtype=np.float32)#[1:]
                centd = np.array(df.center.diff().dropna().values,dtype=np.float32)
#                ind   = np.where((is_outlier2(sigma,thresh=4)==True) |  
#                                 (is_outlier2(centd,thresh=4)==True))[0]
                # Outliers in sigma
                ind1  = np.where((is_outlier(sigma)==True))[0]
#                ind2  = np.where((is_outlier(centd)==True))[0]
                # Negative centers
                ind3  = np.where(df.center<0)[0]
                ind   = np.union1d(ind1,ind3)
#                ind   = np.union1d(ind,ind3)
                xi.append(ind)
                
            a1,a2  = xi
            xmatch = np.intersect1d(a2, a1)
            
            for scale,df in input_lines.items():
                newdf = df.drop(df.index[xmatch])
                output_lines[scale] = newdf  
                
            return output_lines
                
        #######################################################################
        #                        MAIN PART OF fit_lines                       #
        #######################################################################
        # Debugging
        plot=False
        
        # Extract data from the fits file
        spec1d  = self.extract1d(order,nobackground=nobackground,vacuum=True)
        
        pn,weights  = self.calculate_photon_noise(order,return_array=True)
        weights = self.get_weights1d(order)
        # Define limits in wavelength and theoretical wavelengths of lines
        maxima  = peakdet(spec1d.flux,spec1d.wave,extreme='max')
        xpeak   = maxima.x
        #print(xpeak)
        nu_min  = 299792458e0/(xpeak.iloc[-1]*1e-10)
        nu_max  = 299792458e0/(xpeak.iloc[0]*1e-10)
        #print(nu_min,nu_max)
        npeaks  = int(round((nu_max-nu_min)/self.reprate))+1
        n_start = int(round((nu_min - self.f0_comb)/self.reprate))
        lbd     = np.array([299792458e0/(self.f0_comb 
               + (n_start+i)*self.reprate)*1e10 for i in range(xpeak.size)][::-1])
        
        # Find the closest detected peak to the theoretical position using the 
        # nearest neighbour method
#        lbd_th   = combine_line_list(lbd,maxima.x.values)
#        lbd_th   = maxima.x.values
        
        
        
        # Invert _vacuum_ wavelenth solution obtained from ThAr and 
        # find theoretical pixel positions of lines 
        try:
            wavecoeff_vacuum = self.wavecoeff_vacuum[order]
        except:
            wavesol          = self.wavesol_thar[order]
            wavecoeff_air    = self.wavecoeff_air[order]
            wavecoeff,pcov   = curve_fit(polynomial3,np.arange(self.npix),
                                         wavesol,
                                         sigma = weights,
                                         p0=wavecoeff_air)
            wavecoeff_vacuum = wavecoeff
        poly1d   = np.poly1d(wavecoeff_vacuum[::-1])
        lbd_th   = lbd
        npeaks   = lbd_th.size
        pix_th   = np.array([(np.poly1d(lbd_th[npeaks-i-1])-poly1d).roots[2].real
                                        for i in range(npeaks)][::-1])
#        print(pix_th.size,lbd_th.size)
        # Define a pandas DataFrame object with theoretical centres of lines
        lines_th = pd.DataFrame({'pixel':pix_th,'wave':lbd_th}) 
        maxima_p = peakdet(spec1d.flux,spec1d.pixel,extreme='max')
        maxima_th = pd.DataFrame({'pixel':maxima_p.x,'wave':maxima.x})
        if   nobackground is False:
            model = gauss4p
        elif nobackground is True:
            model = gauss3p
        #print(lines_th)
        # Perform the fitting        
        lines    = {}
        for scale in ['wave','pixel']:
            if plot:
                plt.figure()
            xarray     = spec1d[scale]
            yarray     = spec1d['flux']
            
            xpos       = maxima_th[scale]
            if   scale == 'wave':
                dxi   = 0.2
            elif scale == 'pixel':
                dxi   = 11.
            dx         = xarray.diff(1).fillna(dxi)
            results = Parallel(n_jobs=-2)(delayed(fit_peak)(i,xarray,yarray,weights,xpos,dx,model,method) for i in range(npeaks))
            lines_fit = pd.DataFrame(results,
                                     index=lines_th.index,
                                     columns=['amplitude','center','sigma','photon_noise'])
            lines_fit['center_th'] = lines_th[scale]
            lines_fit.dropna(axis=0,how='any',inplace=True)            
            lines[scale]        = lines_fit      
        lines = _remove_poor_fits(lines)

        return lines
    def get_background1d(self, order, scale="pixel", kind="linear"):
        '''Function to determine the background of the observations by fitting a cubic spline to the minima of LFC lines'''
        spec1d          = self.extract1d(order=order)
        if scale == "pixel":
            xarray = pd.Series(spec1d.index.values)
        elif scale == "wave":
            xarray = spec1d.wave
        #print(xarray)
        minima          = peakdet(spec1d.flux, xarray, extreme="min")
        xbkg,ybkg       = minima.x, minima.y
        if   kind == "spline":
            coeff       = interpolate.splrep(xbkg, ybkg)
            background  = interpolate.splev(xarray,coeff) 
        elif kind == "linear":
            coeff      = interpolate.interp1d(xbkg,ybkg)
            mask       = np.where((xarray>=min(xbkg))&(xarray<=max(xbkg)))[0]
            background = coeff(xarray[mask])
        del(spec1d); del(xbkg); del(ybkg); del(coeff)
        return background
    def get_envelope1d(self, order, scale="pixel", kind="spline"):
        '''Function to determine the envelope of the observations by fitting a cubic spline to the maxima of LFC lines'''
        key = scale
        spec1d      = self.extract1d(order=order)
        maxima      = peakdet(spec1d["flux"], spec1d[scale], extreme="max")
        xpeak,ypeak = maxima.x, maxima.y
        
        if   kind == "spline":
            coeff       = interpolate.splrep(xpeak, ypeak)
            envelope    = interpolate.splev(spec1d[scale],coeff) 
        elif kind == "linear": 
            coeff    = interpolate.interp1d(xpeak,ypeak)
            mask     = np.where((spec1d[scale]>=min(xpeak))&(spec1d[key]<=max(xpeak)))[0]
            envelope = coeff(spec1d[scale][mask])
        del(spec1d); del(xpeak); del(ypeak); del(coeff)
        return envelope

    def get_rv_diff(self,order,scale="pixel"):
        ''' Function that calculates the RV offset between the line fitted with and without background subtraction'''
        self.__check_and_load__()
        if scale == "pixel":
            f = 826. #826 m/s is the pixel size of HARPS
        elif scale == "wave":
        # TO DO: think of a way to convert wavelengths into velocities. this is a function of wavelength and echelle order. 
            f = 1. 
        lines_withbkg = self.fit_lines(order,scale,nobackground=False)
        lines_nobkg   = self.fit_lines(order,scale,nobackground=True)
        npeaks        = lines_withbkg.size
        delta_rv      = (lines_withbkg["MU"]-lines_nobkg["MU"])*f
        median_rv     = np.nanmedian(delta_rv)
        print("ORDER {0}, median RV displacement = {1}".format(order,median_rv))
        return delta_rv
    def get_wavecoeff(self,medium='vacuum',orders=None):
        self.__check_and_load__()
        if orders==None:
            orders = np.arange(0,self.nbo,1)
        # If attribute exists, return the current value
        attribute = 'wavecoeff_{}'.format(medium)
        if hasattr(self, attribute):
            return getattr(self,attribute)
        # Else, return coefficients in the appropriate medium
        else:
            wavecoeff = np.zeros(shape = (self.nbo, self.d+1, ), 
                                    dtype = np.float64)
            if medium=='air':
                self.bad_orders = []
                for order in orders:
                    # Try reading the coefficients for each order. If failed, 
                    # classify the order as a 'bad order'.
                    for i in range(self.d+1):                    
                        ll    = i + order*(self.d+1)
                        try:
                            coeff = self.header["ESO DRS CAL TH COEFF LL{0}".format(ll)]
                        except:
                            coeff = 0
                        if coeff==0:                         
                            if order not in self.bad_orders:
                                self.bad_orders.append(order)
                        wavecoeff[order,i] = coeff
            elif medium=='vacuum':
                wavecoeff_air = self.get_wavecoeff('air',orders=orders)
                for order in orders:
                    wc_air            = wavecoeff_air[order]
                    wavecoeff_vac,covariance = curve_fit(polynomial, 
                                                         np.arange(self.npix), 
                                                         self.wavesol_thar[order], 
                                                         p0=wc_air)
                    wavecoeff[order]         = wavecoeff_vac
            setattr(self,attribute,wavecoeff)
            return wavecoeff
    def get_weights1d(self,order,calibrator="ThAr"):
        ''' 
        Return weights of individual pixels for a single 1d echelle order 
        (Bouchy 2001)
        '''
        spec1d        = self.extract1d(order=order,nobackground=False)
        wavesol       = self.__get_wavesol__(calibrator)*1e-10 # meters
        diff          = np.diff(wavesol[order])
        #dlambda       = np.insert(diff,0,diff[0])
        dlambda       = np.gradient(wavesol[order])
        dflux         = np.gradient(spec1d['flux'])#,dlambda)
        df_dl         = derivative1d(spec1d['flux'].values,wavesol[order])
        #print(dflux)
        weights1d     = wavesol[order]**2 * (df_dl)**2 / spec1d['flux']
        return weights1d
    def get_weights2d(self,calibrator="ThAr"):
        ''' 
        Return weights of individual pixels for the entire 2d spectrum
        (Bouchy 2001) 
        '''
        spec2d_data    = self.data#[sOrder:eOrder] ##### FIND A WAY TO REMOVE ROWS WITH ZEROS
        #rows, cols    = np.nonzero(spec2d_data)
        spec2d         = self.data#[rows,cols]
        wavesol2d      = self.__get_wavesol__(calibrator)#[sOrder:eOrder]
        
        #remove zeros
        remove_zeros=False
        if remove_zeros:
            zerorows       = np.where(wavesol2d.any(axis=1)==False)[0]
            wavesol        = np.delete(wavesol2d,zerorows,axis=0)
            spec2d         = np.delete(spec2d_data,zerorows,axis=0)
            orders         = np.delete(self.nbo,1,zerorows,axis=0)
        else:
            wavesol        = wavesol2d
            spec2d         = spec2d_data
            orders         = np.arange(self.nbo)
        
        #diff           = np.diff(wavesol)
        #dlambda        = np.insert(diff,0,diff[:,0],axis=1)
        #dflux          = np.gradient(spec2d,axis=1)#,dlambda,axis=1)
        df_dl          = np.zeros(shape=spec2d.shape)
        for order in orders:
            if self.is_bad_order(order)==True:
                df_dl[order] = 1.0
            elif wavesol[order].sum() == 0.:
                df_dl[order] = 1.0
            else:    
                df_dl[order] = derivative1d(spec2d[order],wavesol[order])                    
        zeros          = np.where(spec2d==0)
        spec2d[zeros]  = np.inf                    
        weights2d      = (wavesol**2 * df_dl**2) / np.abs(spec2d)
        cut            = np.where(weights2d == 0.)
        weights2d[cut] = np.inf
        self.weights2d = np.zeros(shape=self.data.shape)
        self.weights2d[orders,:] = weights2d
        
        return self.weights2d
    def introduce_gaps(self,x,gaps):
        xc = x.copy()
        if np.size(gaps)==1:
            gap  = gaps
            gaps = np.full((7,),gap)
        for i,gap in enumerate(gaps):
            ll = (i+1)*self.npix/(np.size(gaps)+1)
            cut = np.where(x>ll)[0]
            xc.iloc[cut] = xc.iloc[cut]-gap
        return xc
    def is_bad_order(self,order):
        if order in self.bad_orders: 
            return True
        else:
            return False
    def plot(self,order,nobackground=False,scale='wave',fit=False,naxes=1,
             ratios=None,title=None,sep=0.05,alignment="vertical",
             figsize=(16,9),sharex=None,sharey=None,**kwargs):
        #if hasattr(self,'figure'):
        #    pass
        #else:
        fig, axes = get_fig_axes(naxes,ratios=ratios,title=title,
                                 sep=sep,alignment=alignment,
                                 figsize=figsize,sharex=sharex,
                                 sharey=sharey,**kwargs)
        self.figure, self.axes = fig, axes
    
        spec1d = self.extract1d(order,nobackground=nobackground)
        x      = spec1d[scale]
        y      = spec1d.flux
        self.axes[0].plot(x,y,label='Data')
        if fit==True:
            fit_lines = self.fit_lines(order,nobackground=nobackground)
            self.axes[0].plot(x,gaussN(x,fit_lines[scale]),label='Fit')
        self.axes[0].legend()
        self.figure.show()
class EmissionLine(object):
    ''' Class with functions to fit LFC lines as pure Gaussians'''
    def __init__(self,xdata,ydata,yerr=None, weights=None, scale=None):
        ''' Initialize the object using measured data
        
        Args:
        ----
            xdata: 1d array of x-axis data (pixels or wavelengts)
            ydata: 1d array of y-axis data (electron counts)
            weights:  1d array of weights calculated using Bouchy method
        '''
#        if scale is 'wave':
#            self.xdata = 100*xdata
#        else:
        self.xdata   = xdata
        self.scale   = scale
        self.xbounds = (self.xdata[:-1]+self.xdata[1:])/2
        self.ydata   = ydata
        yerr         = yerr if yerr is not None else np.sqrt(np.abs(self.ydata))
        weights      = weights if weights is not None else 1/yerr
        self.yerr    = yerr
        self.weights = weights
        
        
        self.success = False                       
    def _initialize_parameters(self):
        ''' Method to initialize parameters from data (pre-fit)
        Returns:
        ----
            p0: tuple with inital (amplitude, mean, sigma) values
        '''
        yarg = np.argmax(self.ydata)
        A0 = self.ydata[yarg]
        
        m0 = np.mean(self.xdata)
        s0 = np.sqrt(np.var(self.xdata))
        y0 = 0e0
        p0 = (A0,m0,s0)
        self.initial_parameters = p0
        return p0
    def _get_fit_parameters(self):
        ''' Method to check whether the fit has been successfully performed.
        If the fit was successful, returns the fit values. Otherwise, the 
        method performs the fitting procedure and returns the fit values.
        Returns:
        -------
            pfit: tuple with fitted (amplitude, mean, sigma) values
        '''
        if self.success == True:
            pfit = self.fit_parameters
        else:
            p0 = self._initialize_parameters()
            pfit, pcov = self.fit(p0)
        return pfit
    def _wrap_jac(self):
        if weights is None:
            def jac_wrapped(params):
                return self.jacobian(params,self.xdata)
        else:
            def jac_wrapped(params):
                return self.weights[:, np.newaxis] * np.asarray(self.jacobian(params,self.xdata))
        return jac_wrapped
    def residuals(self,pars,weights=None):
        ''' Returns the residuals of individual data points to the model.
        Args:
        ----
            pars: tuple (amplitude, mean, sigma) of the model
        Returns:
        -------
             1d array (len = len(xdata)) of residuals
        '''
#        model = self.model(*pars)
        cdata = self.ydata[1:-1]
        weights = weights if weights is not None else self.weights[1:-1]
        return weights * (self.model(*pars) - cdata)
    def chisq(self,pars,weights=None):
        ''' Returns the chi-square of data points to the model.
        Args:
        ----
            pars: tuple (amplitude, mean, sigma) of the model
        Returns:
        -------
            chisq
        '''
        return (self.residuals(pars)**2).sum()
    def evaluate(self,p,x=None):
        ''' Returns the evaluated Gaussian function along the provided `x' and 
        for the provided Gaussian parameters `p'. 
        
        Args:
        ---- 
            x: 1d array along which to evaluate the Gaussian. Defaults to xdata
            p: tuple (amplitude, mean, sigma) of Gaussian parameters. 
               Defaults to the fit parameter values.
        '''
        if x is None:
            x = self.xdata
            xb = self.xbounds
        else:
            x = x
            xb = (x[:-1]+x[1:])/2
        A, mu, sigma = p if p is not None else self._get_fit_parameters()
        y = A * np.exp(-1/2*((x-mu)/sigma)**2) 
        
#        e1 = erf((xb[:-1]-mu)/(np.sqrt(2)*sigma))
#        e2 = erf((xb[1:] -mu)/(np.sqrt(2)*sigma))
#        y  = A*sigma*np.sqrt(np.pi/2)*(e2-e1)
        return y
    def jacobian(self,p,x0=None):
        A, mu, sigma = p
        x = x0 if x0 is not None else self.xdata[1:-1]
        y = self.evaluate(p,x) if x0 is not None else self.ydata[1:-1]
        dfdp = np.array([y/A,
                         y*(x-mu)/(sigma**2),
                         y*(x-mu)**2/(sigma**3)]).T
        return dfdp
    def hessian(self,p,x=None):
        A, mu, sigma = p
        x = x if x is not None else self.xdata[1:-1]
        y = self.evaluate(p,x) if x is not None else self.ydata[1:-1]
        N = len(x)
        n = len(p)
        hes = np.zeros((n,n,N))
        hes[0,0] = 0
        hes[0,1] = y/A*(x-mu)/sigma**2
        hes[0,2] = y/A*(x-mu)**2/sigma**3
        hes[1,0] = hes[0,1]
        hes[1,1] = y*((x-mu)**2/sigma**4 - 1/sigma**2)
        hes[1,2] = y*((x-mu)**3/sigma**5 - 2*(x-mu)/sigma**3)
        hes[2,0] = hes[0,2]
        hes[2,1] = hes[1,2]
        hes[2,2] = y*((x-mu)**4/sigma**6 - 3*(x-mu)**2/sigma**4)
        self.hess = hes
        return hes
    def fit_ncg(self,p0):
        if p0 is None:
            p0 = self._initialize_parameters()
        minimize(self.chisq,p0,method='Newton-CG',jac=self.jacobian)
            
    def fit(self,p0=None,absolute_sigma=True, bounds=(-np.inf, np.inf), 
            method=None, check_finite=True, **kwargs):
        ''' Performs the fitting of a Gaussian to the data. Acts as a wrapper 
        around the scipy.optimize `leastsq' function that minimizes the chisq 
        of the fit. 
        
        The function at each point is evaluated as an integral of the Gaussian 
        between the edges of the pixels (in case of wavelengths, boundary is 
        assumed to be in the midpoint between wavelength values). 
        
        The function calculates the fit parameters and the fit errors from the 
        covariance matrix provided by `leastsq'.
        
        Args:
        ----
            p0: tuple (amplitude, mean, sigma) with the initial guesses. 
                If None, is calculated from the data.
                
        Returns:
        -------
            pfit: tuple (amplitude, mean, sigma) of best fit parameters
            pcov: covariance matrix between the best fit parameters
            
        Optional:
        --------
            return_full: Returns full output. Defaults to False
        '''
        
        if p0 is None:
            p0 = self._initialize_parameters()
        p0 = np.atleast_1d(p0)
        n = p0.size    
        lb, ub = prepare_bounds(bounds, n)
        bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
        if method is None:
            if bounded_problem:
                method = 'trf'
            else:
                method = 'lm'
    
        if method == 'lm' and bounded_problem:
            raise ValueError("Method 'lm' only works for unconstrained problems. "
                             "Use 'trf' or 'dogbox' instead.")
        # NaNs can not be handled
        if check_finite:
            self.ydata = np.asarray_chkfinite(self.ydata)
        else:
            self.ydata = np.asarray(self.ydata)
    
        if isinstance(self.xdata, (list, tuple, np.ndarray)):
            # `xdata` is passed straight to the user-defined `f`, so allow
            # non-array_like `xdata`.
            if check_finite:
                self.xdata = np.asarray_chkfinite(self.xdata)
            else:
                self.xdata = np.asarray(self.xdata)
    
        if method != 'lm':
            jac = '2-point'
        print("Method:",method)    
        if method == 'lm':    
            return_full = kwargs.pop('full_output', False)
#            wrapped_jac = self._wrap_jac()
            res = leastsq(self.residuals,p0,Dfun=None,full_output=1)#,col_deriv=True,**kwargs)
            pfit, pcov, infodict, errmsg, ier = res
            cost = np.sum(infodict['fvec']**2)
            if ier not in [1, 2, 3, 4]:
                raise RuntimeError("Optimal parameters not found: " + errmsg)
            else:
                success = True
        else:
            res = least_squares(self.residuals, p0, jac=jac, bounds=bounds, method=method,
                                **kwargs)
            if not res.success:
                raise RuntimeError("Optimal parameters not found: " + res.message)
    
            cost = 2 * res.cost  # res.cost is half sum of squares!
            pfit = res.x
            
            success = res.success
            # Do Moore-Penrose inverse discarding zero singular values.
            _, s, VT = svd(res.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[:s.size]
            pcov = np.dot(VT.T / s**2, VT)
            return_full = False    
        warn_cov = False
#        absolute_sigma=False
        dof  = (len(self.ydata) - len(pfit))
        
#        if self.scale is 'wave':
#            pfit[1] = pfit[1]/100
#            pfit[2] = pfit[2]/100
                
        if pcov is None:
            # indeterminate covariance
            pcov = np.zeros((len(pfit), len(pfit)), dtype=np.float)
            pcov.fill(np.inf)
            warn_cov = True         
        elif not absolute_sigma:
            if len(self.ydata) > len(pfit):
                
                #s_sq = cost / (self.ydata.size - pfit.size)
                s_sq = cost / dof
                pcov = pcov * s_sq
            else:
                pcov.fill(self.inf)
                warn_cov = True
        if warn_cov:
            warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning)    
        error = [] 
        for i in range(len(pfit)):
            try:
              error.append(np.absolute(pcov[i][i])**0.5)
            except:
              error.append( 0.00 )
        
        self.fit_parameters = pfit
        self.covar     = pcov
        self.rchi2     = cost / dof
        self.dof       = dof
        self.fit_error = np.array(error)
#        self.infodict = infodict
#        self.errmsg = errmsg
        self.success = success
        self.cost = cost
        if return_full:
            return pfit, pcov, infodict, errmsg, ier
        else:
            return pfit, pcov
    def model(self,A,mu,sigma):
        ''' Calculates the expected electron counts by assuming:
            (1) The PSF is a Gaussian function,
            (2) The number of photons falling on each pixel is equal to the 
                integral of the PSF between the pixel edges. (In the case of 
                wavelengths, the edges are calculated as midpoints between
                the wavelength of each pixel.)
        
        The integral of a Gaussian between two points, x1 and x2, is calculated
        as:
            
            Phi(x1,x2) = A * sigma * sqrt(pi/2) * [erf(t2) - erf(t1)]
        
        Where A and sigma are the amplitude and the variance of a Gaussian, 
        and `t' is defined as:
            
            t = (x - mu)/(sqrt(2) * sigma)
        
        Here, mu is the mean of the Gaussian.
        '''
        xb = self.xbounds
        e1 = erf((xb[:-1]-mu)/(np.sqrt(2)*sigma))
        e2 = erf((xb[1:] -mu)/(np.sqrt(2)*sigma))
        print("E1:",e1)
        print("E2:",e2)
        y  = A*sigma*np.sqrt(np.pi/2)*(e2-e1)
        return y
    def plot(self,fit=True,cofidence_intervals=True):
        ''' Plots the line as a histogram of electron counts with corresponding
        errors. If `fit' flag is True, plots the result of the fitting 
        procedure.
        
        
        '''
        fig,ax = h.get_fig_axes(1,figsize=(9,9))
        self.fig = fig
        self.ax_list  = ax
        widths = np.diff(self.xdata)[:-1]
        ax[0].bar(self.xdata[1:-1],self.ydata[1:-1],
                  widths,align='center',alpha=0.3,color='#1f77b4')
        ax[0].errorbar(self.xdata[1:-1],self.ydata[1:-1],
                       yerr=self.yerr[1:-1],fmt='o',color='#1f77b4')
        if fit is True:
            p = self._get_fit_parameters()
            xeval = np.linspace(np.min(self.xdata),np.max(self.xdata),100)
            yeval = self.evaluate(p,xeval)
            fit = True
        ax[0].plot(xeval,yeval,color='#ff7f0e')
        ax[0].plot([p[1],p[1]],[0,p[0]],ls=':',color='#ff7f0e')
        if cofidence_intervals is True and fit is True:
            y,ylow,yhigh = self.confidence_band(xeval,confprob=0.05)
            ax[0].fill_between(xeval,ylow,yhigh,alpha=0.5,color='#ff7f0e')
            y,ylow,yhigh = self.confidence_band(xeval,confprob=0.32)
            ax[0].fill_between(xeval,ylow,yhigh,alpha=0.2,
                              color='#ff7f0e')
        
        
        return
    def confidence_band(self, x, confprob=0.05, absolute_sigma=True):
        from scipy.stats import t
        # Given the confidence probability confprob = 100(1-alpha)
        # we derive for alpha: alpha = 1 - confprob
        alpha = 1.0 - confprob
        prb = 1.0 - alpha/2
        tval = t.ppf(prb, self.dof) #degrees of freedom
                    
        C = self.covar
       
        p = self._get_fit_parameters()
        n = len(p)              # Number of parameters from covariance matrix
        N = len(x)
        if absolute_sigma:
            covscale = 1.0
        else:
            covscale = self.rchi2
          
        y = self.evaluate(p,x)
        dfdp = self.jacobian(p,x).T
        df2 = np.zeros(N)
        for j in range(n):
            for k in range(n):
                df2 += dfdp[j]*dfdp[k]*C[j,k]
        df = np.sqrt(covscale*df2)
#        df = np.sqrt(df2)
        
        delta = tval * df
        upperband = y + delta
        lowerband = y - delta
        return y, upperband, lowerband       
        
class Worker(object):   
    def __init__(self,filename=None,mode=None,manager=None,
                 refA=None,refB=None,orders=None):
        self.filename = filename
        self.open_file(self.filename,mode)
        self.manager = manager
        print(self.file)
        eo = self.check_exists("orders")
        if eo == False:
            if not orders:
                orders = np.arange(sOrder,eOrder)
            self.file.create_dataset("orders",data=orders)
        if not refA:
            return "Reference wavelength solution for fibre A not given"
        if not refB:
            return "Reference wavelength solution for fibre B not given"
            
    def is_open(self):
        return self.open
    def open_file(self,filename=None,mode=None):
        if not filename:
            filename = self.filename
        e = os.path.isfile(filename)
        if not mode:
            if e:
                mode = "r+"
            else:
                mode = "w"
        print(filename,mode)
        self.file = h5py.File(filename,mode)
        self.open = True
    def dump_to_file(self):
        self.file.flush()
        return
    def close_file(self):
        self.file.close()
        self.open = False
        return
    def check_exists(self,node):
        e = False
        ds = "{}".format(node)
        if ds in self.file:
            e = True
        return e 
    
    def run(self,i):
        o = self.is_open()
        if o == False:
            self.open_file()
        else:
            pass
        e = self.check_exists("{}".format(i))
        
        nodes = ["{}/{}/{}".format(i,f,t) for f in ["A","B"] 
                 for t in ["wavesol_LFC","rv","weights","lines","coef"]]
        ne = [self.check_exists(node) for node in nodes]
           
        filelim = filelim = {'A':self.manager.file_paths['A'][93], 
                             'B':self.manager.file_paths['B'][93]}
        if ((e == False) or (np.all(ne)==False)):
            fileA = self.manager.file_paths['A'][i]
            if fileA < filelim['A']:
                LFC1 = 'FOCES'
                LFC2 = 'FOCES'
                anchor_offset=0e0
            else:
                LFC1 = 'FOCES'
                LFC2 = 'HARPS'
                anchor_offset=-100e6
            fileB = self.manager.file_paths['B'][i]
            print(i,fileA,LFC1)
            print(i,fileB,LFC2)
            specA = Spectrum(fileA,data=True,LFC=LFC1)
            specB = Spectrum(fileB,data=True,LFC=LFC2)
    
            wavesolA = specA.__get_wavesol__(calibrator='LFC',
                                  wavesol_thar=tharA,
                                  wavecoeff_air=wavecoeff_airA)
            wavesolB = specB.__get_wavesol__(calibrator='LFC',
                                  anchor_offset=anchor_offset,
                                  wavesol_thar=tharB,
                                  #orders=np.arange(sOrder+1,eOrder-1),
                                  wavecoeff_air=wavecoeff_airB)
            
            rvA      = (wavesolA[sOrder:eOrder] - wavesol_refA)/wavesol_refA * c
            rvB      = (wavesolB[sOrder:eOrder] - wavesol_refB)/wavesol_refB * c
            
              
            weightsA = specA.get_weights2d()[sOrder:eOrder]
            weightsB = specB.get_weights2d()[sOrder:eOrder]
            
            linesA   = specA.cc_data.values
            linesB   = specB.cc_data.values
            
            coefsA   = specA.wavecoef_LFC
            coefsB   = specB.wavecoef_LFC
            
            nodedata = [wavesolA,rvA,weightsA,linesA,coefsA,
                        wavesolB,rvB,weightsB,linesB,coefsB]
            for node,data in zip(nodes,nodedata):
                node_exists = self.check_exists(node)
                if node_exists==False:
                    self.file.create_dataset(node,data=data)
                    self.file.flush()
                else:
                    pass
            
        return       
class Manager(object):
    '''
    Manager is able to find the files on local drive, read the selected data, 
    and perform bulk operations on the data.
    '''
    def __init__(self,date=None,year=None,month=None,day=None,
                 begin=None,end=None,sequence=None,get_file_paths=True):
        '''
        date(yyyy-mm-dd)
        begin(yyyy-mm-dd)
        end(yyyy-mm-dd)
        sequence(day,sequence)
        '''
        self.file_paths = []
        self.spectra    = []
        #harpsDataFolder = os.path.join("/Volumes/home/dmilakov/harps","data")
        harpsDataFolder = os.path.join("/Volumes/home/dmilakov/harps","data")
        self.harpsdir   = harpsDataFolder
        if sequence!=None:
            if type(sequence)==tuple:
                sequence_list_filepath = os.path.join('/Volumes/home/dmilakov/harps/aux/COMB_April2015/','day{}_seq{}.list'.format(*sequence))
                self.sequence_list_filepath = [sequence_list_filepath]
                self.sequence = [sequence]
            elif type(sequence)==list:
                self.sequence_list_filepath = []
                self.sequence = sequence
                for item in sequence:
                    sequence_list_filepath = os.path.join('/Volumes/home/dmilakov/harps/aux/COMB_April2015/','day{}_seq{}.list'.format(*item))
                    self.sequence_list_filepath.append(sequence_list_filepath)
        if sequence == None:
            self.sequence_list_filepath = None
            if   date==None and (year!=None and month!=None and day!=None) and (begin==None or end==None):
                self.dates = ["{y:4d}-{m:02d}-{d:02d}".format(y=year,m=month,d=day)]
                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
            elif date!=None:
                self.dates = [date]
                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
            elif date==None and (year==None or month==None or day==None) and (begin==None or end==None):
                raise ValueError("Invalid date input. Expected format is 'yyyy-mm-dd'.")
            elif (begin!=None and end!=None):
                by,bm,bd       = tuple(int(val) for val in begin.split('-'))
                ey,em,ed       = tuple(int(val) for val in end.split('-'))
                print(by,bm,bd)
                print(ey,em,ed)
                self.begindate = datetime.datetime.strptime(begin, "%Y-%m-%d")
                self.enddate   = datetime.datetime.strptime(end, "%Y-%m-%d")
                self.dates     = []
                def daterange(start_date, end_date):
                    for n in range(int ((end_date - start_date).days)):
                        yield start_date + datetime.timedelta(n)
                for single_date in daterange(self.begindate, self.enddate):
                    self.dates.append(single_date.strftime("%Y-%m-%d"))
                
            self.datadirlist  = []
            
            for date in self.dates:
                #print(date)
                datadir = os.path.join(harpsDataFolder,date)
                if os.path.isdir(datadir):
                    self.datadirlist.append(datadir)
        self.orders = np.arange(sOrder,eOrder,1)
        if get_file_paths:
            self.get_file_paths(fibre='AB')
        
    def get_file_paths(self, fibre, ftype='e2ds',**kwargs):
        '''
        Function to find fits files of input type and input date in the 
        $HARPS environment variable.
        
        INPUT:
        ------
        date  = "yyyy-mm-dd" - string
        type  = ("e2ds", "s1d") - string
        fibre = ("A", "B", "AB") - string
        
        Additional arguments:
        condition = ("filename")
            "filename": requires additional arguments "first" and "last" 
            (filename)
        OUTPUT:
        ------
        fileList = list of fits file paths in the HARPS data folder for a 
                   specific date
        '''
        self.fibre = fibre
        self.ftype = ftype
        #if os.path.isdir(self.datadir)==False: raise OSError("Folder not found")
        filePaths        = {}
        
        if self.sequence_list_filepath:  
            
            if type(self.sequence_list_filepath)==list:    
                sequence_list = []
                for item,seq in zip(self.sequence_list_filepath,self.sequence):
                    with open(item) as sl:            
                        for line in sl:
                            sequence_list.append([seq,line[0:29]])
                    for fbr in list(fibre):
                        fitsfilepath_list = []
                        for seq,item in sequence_list:
                            date    = item.split('.')[1][0:10]
                            datadir = "{date}_seq{n}".format(date=item.split('.')[1][0:10],n=seq[1])
                            time    = item.split('T')[1].split(':')
                            fitsfilepath = os.path.join(self.harpsdir,datadir,
                                            "HARPS.{date}T{h}_{m}_{s}_{ft}_{f}.fits".format(date=date,h=time[0],m=time[1],s=time[2],ft=ftype,f=fbr))
                            fitsfilepath_list.append(fitsfilepath)
                            #print(date,time,fitsfilepath,os.path.isfile(fitsfilepath))
                        filePaths[fbr] = fitsfilepath_list
        if not self.sequence_list_filepath:
            for fbr in list(fibre):
                nestedlist = []
                for datadir in self.datadirlist:
                    try:
                        datefilepaths=np.array(glob(os.path.join(datadir,"*{ftp}*{fbr}.fits".format(ftp=ftype,fbr=fbr))))
                    except:
                        raise ValueError("No files of this type were found")
                    if "condition" in kwargs.keys():
                        self.condition = {"condition":kwargs["condition"]}
                        if kwargs["condition"] == "filename":
                            self.condition["first"] = kwargs["first"]
                            self.condition["last"]  = kwargs["last"]
                            ff = np.where(datefilepaths==os.path.join(datadir,
                                    "{base}_{ftp}_{fbr}.fits".format(base=kwargs["first"],ftp=self.ftype,fbr=fbr)))[0][0]
                            lf = np.where(datefilepaths==os.path.join(datadir,
                                    "{base}_{ftp}_{fbr}.fits".format(base=kwargs["last"],ftp=self.ftype,fbr=fbr)))[0][0]
                            selection = datefilepaths[ff:lf]
                            nestedlist.append(selection)
                    else:
                        nestedlist.append(datefilepaths)
                flatlist       = [item for sublist in nestedlist for item in sublist]   
                filePaths[fbr] = flatlist
        self.file_paths = filePaths
        self.numfiles = [np.size(filePaths[fbr]) for fbr in list(fibre)]
        return 
    def get_spectra(self, fibre, ftype='e2ds', header=False,data=False):
        # DOESN'T WORK PROPERLY!! DO NOT USE
        '''
        Function to get a list of Spectrum class objects for manipulation
        '''
        if not self.file_paths:
            print("Fetching file paths")
            self.get_file_paths(fibre=fibre, ftype=ftype)
        else:
            pass
        spectra = {}
        for fbr in list(fibre):
            fbr_list    = self.file_paths[fbr]
            fbr_spectra = []
            for path in fbr_list:
                spectrum = Spectrum(filepath=path,ftype=ftype,header=header,data=data)
                fbr_spectra.append(spectrum)
            spectra[fbr] = fbr_spectra
        self.spectra = spectra
        return self.spectra
    def get_spectrum(self,ftype,fibre,header=False,data=False):
        return 0
    def read_data(self, filename="datacube", **kwargs):
        try:    
            fibre       = kwargs["fibre"]
            self.fibre  = fibre
        except: fibre   = self.fibre
        try:    
            orders      = kwargs["orders"]
            self.orders = orders
        except: orders  = self.orders
        # if there are conditions, change the filename to reflect the conditions
        try:
            for key,val in self.condition.items():
                filename = filename+"_{key}={val}".format(key=key,val=val)
        except:
            pass
        if not self.datadirlist:
            self.get_file_paths(fibre=fibre)
        # CREATE DATACUBE IF IT DOES NOT EXIST
        if   len(self.dates)==1:
            self.datafilepath = os.path.join(self.datadirlist[0],
                                         "{name}_{fibre}.npy".format(name=filename, fibre=fibre))
        elif len(self.dates)>1:
            self.datafilepath = os.path.join(self.harpsdir,
                                         "{name}_{fibre}_{begin}_{end}.npy".format(name=filename, fibre=fibre,
                                             begin=self.begindate.strftime("%Y-%m-%d"), end=self.enddate.strftime("%Y-%m-%d")))
        #self.datafilepath = os.path.join(self.harpsdir,"2015-04-18/datacube_condition=filename_first=HARPS.2015-04-18T01_35_46.748_last=HARPS.2015-04-18T13_40_42.580_AB.npy")
        if os.path.isfile(self.datafilepath)==False:
            #sys.exit()
            print("Data at {date} is not prepared. Processing...".format(date=self.dates))
            self.reduce_data(fibre=fibre, filename=filename)
        else:
            pass        
        
        datainfile  = np.load(self.datafilepath)
        self.dtype  = [datainfile.dtype.fields[f][0].names for f in list(fibre)][0]
        self.nfiles = [np.shape(datainfile[f]["FLX"])[1] for f in list(fibre)]
        if kwargs["orders"]:
            col         = select_orders(orders)
            subdtype  = Datatypes(nOrder=len(orders),nFiles=self.nfiles[0],fibre=fibre).specdata(add_corr=True)
            data        = np.empty(shape=datainfile.shape, dtype=subdtype.data)
            for f in list(fibre):
                for dt in self.dtype:
                    data[f][dt] = datainfile[f][dt][:,:,col]
            
        else:
            data    = datainfile
        self.data   = data
        
        
        return
    def reduce_data(self, fibre, ftype='e2ds', filename="datacube", **kwargs):
        ''' Subroutine which prepares data for easier handling. 
        Subroutine reads all spectra contained in Manager.file_paths and extracts detected counts for all orders (FLX), fits the 
        envelope (ENV) and background (BKG) with a cubic spline, calculates the background-to-envelope ratio (B2R). The subroutine 
        removes the background from original detected signal (FMB). This data is finally saved into a numpy pickled file.'''        
        if np.size(self.file_paths)>0:
            pass
        else:
            self.get_file_paths(fibre=fibre, ftype=ftype, **kwargs)
        
        # SOME PARAMETERS 
        #nPix    = 4096               # number of pixels in image
        #sOrder  = 40                # first order in image
        #eOrder  = 72                # last order in image
        nOrder  = eOrder-sOrder     # number of orders in image
        
        #if   len(self.dates)==1:
        #    self.datafilepath = os.path.join(self.datadirlist[0],
        #                                 "{name}_{fibre}.npy".format(name=filename, fibre=fibre))
        #elif len(self.dates)>1:
        #    self.datafilepath = os.path.join(self.harpsdir,
        #                                 "{name}_{fibre}_{begin}_{end}.npy".format(name=filename, fibre=fibre, 
        #                                    begin=self.begindate.strftime("%Y-%m-%d"), end=self.enddate.strftime("%Y-%m-%d")))
        fibres  = list(fibre)
        nFibres = len(fibres)
        nFiles  = len(self.file_paths[fibres[0]])
        print("Found {} files".format(nFiles))

        data    = np.zeros((nPix,),dtype=Datatypes(nFiles=nFiles,nOrder=nOrder,fibre=fibre).specdata(add_corr=True).data)
        #print(np.shape(data))
        
        for f in fibres:
            nFiles = np.size(self.file_paths[f])
            for e in range(nFiles):
                spec = Spectrum(filepath=self.file_paths[f][e],ftype='e2ds',header=True)
                print(self.file_paths[f][e])
                for order in range(sOrder,eOrder-1,1):
                    o = order-sOrder
                    envelope   = spec.get_envelope1d(order=order,scale='pixel',kind='spline')
                    background = spec.get_background1d(order=order,scale='pixel',kind='spline')
                    b2eRatio   = (background / envelope)
                    #print(np.shape(envelope),np.shape(background),np.shape(b2eRatio), np.shape(data))
                    #print(f,e,o, np.shape(envelope), np.shape(data[f]["ENV"]))
                    data[f]["FLX"][:,e,o] = spec.extract1d(order=order)['flux']
                    data[f]["ENV"][:,e,o] = envelope
                    data[f]["BKG"][:,e,o] = background
                    data[f]["B2E"][:,e,o] = b2eRatio
                    data[f]["FMB"][:,e,o] = data[f]["FLX"][:,e,o] - background
                    del(envelope); del(background); del(b2eRatio)
                    gc.collect()
                del(spec)
        # SAVE TO FILE
        
        np.save(self.datafilepath,data)
        print("Data saved to {0}".format(self.datafilepath))
        return
    def select_file_subset(self,condition,**kwargs):
        '''
        Select only those files which fulfill the condition. 
        Condition keyword options:
            filename: selects files between the two file (base)names. Requires additional keywords "first" and "last".

        Returns a new Manager class object, with only selected filenames
        '''
        if not self.file_paths:
            self.get_file_paths(self.fibre)
        selection = {}
        if condition == "filename":
            for f in list(self.fibre):
                filenames = np.array(self.file_paths[f])
                print(os.path.join(self.datadir,
                                "{base}_{ft}_{f}.fits".format(base=kwargs["first"],ft=self.ftype,f=f)))
                ff = np.where(filenames==os.path.join(self.datadir,
                                "{base}_{ft}_{f}.fits".format(base=kwargs["first"],ft=self.ftype,f=f)))[0][0]
                lf = np.where(filenames==os.path.join(self.datadir,
                                "{base}_{ft}_{f}.fits".format(base=kwargs["last"],ft=self.ftype,f=f)))[0][0]
                print(ff,lf)
                selection[f] = list(filenames[ff:lf])
        newManager = Manager(date=self.dates[0])
        newManager.fibre      = self.fibre
        newManager.ftype      = self.ftype
        newManager.file_paths = selection
        return newManager
    def calculate_medians(self,use="data",**kwargs):
        '''
        This subroutine calculates the medians for user-selected datatypes and orders, or for all data handled by the manager.
        The operations are done using data in Manager.data and a new attribute, Manager.mediandata, is created by this subroutine.
        '''
        try:    fibre  = kwargs["fibre"]
        except: fibre  = self.fibre
        try:    dtype  = kwargs["dtype"]
        except: dtype  = self.dtype
        try:    orders = kwargs["orders"]
        except: 
            try: orders = self.orders
            except: print("Keyword 'orders' not specified")
        try:    errors = kwargs["errors"]
        except: errors = False
        
        datatypes  = Datatypes().specdata(self.nfiles[0],nOrder=len(orders),fibre=fibre, add_corr=True)
        if   use == "data":
            dtuse = datatypes.median
            data = self.data
        elif use == "fourier":
            dtuse = datatypes.ftmedian
            try:
                data = self.ftdata
            except:
                self.calculate_fft(**kwargs)
                data = self.datafft
        # if 'error' keyword is true, calculate 16th,50th,84th percentile
        data50p = np.empty(shape=data.shape, dtype=dtuse)
        if   errors == True:
            q = [50,16,84]
            data16p = np.empty(shape=data.shape, dtype=dtuse)
            data84p = np.empty(shape=data.shape, dtype=dtuse)
        # else, calculate only the 50th percentile (median)
        else:
            q = [50]
        # for compatibility with other parts of the code, it is necessary to reshape the arrays (only needed if using a single order)
        # reshaping from (npix,) -> (npix,1)
        if np.size(orders)==1:
            data50p=data50p[:,np.newaxis]
            if errors == True:
                data16p = data16p[:,np.newaxis]
                data84p = data84p[:,np.newaxis]

        # make a selection on orders
        col = select_orders(orders)
        # now use the selection to define a new object which contains median data
        
        for f in list(fibre):
            for dt in dtype:
                #print(f,dt,data[f][dt].shape)
                subdata = data[f][dt]#[:,:,col]
                auxdata = np.nanpercentile(subdata,q=q,axis=1)
                print(auxdata.shape)
                if   errors == True:
                    data50p[f][dt] = auxdata[0]
                    data16p[f][dt] = auxdata[1]
                    data84p[f][dt] = auxdata[2]
                elif errors == False:
                    data50p[f][dt] = auxdata
        if   use == "data":
            self.data50p = data50p
            if errors == True:
                self.data84p = data84p
                self.data16p = data16p
        elif use == "fourier":
            self.datafft50p = data50p
            if errors == True:
                self.datafft84p = data84p
                self.datafft16p = data16p
        return
    def calculate_fft(self,**kwargs):
        try:    fibre  = kwargs["fibre"]
        except: fibre  = self.fibre
        try:    dtype  = kwargs["dtype"]
        except: dtype  = self.dtype
        #try:    orders = kwargs["orders"]
        #except: 
        #    try: 
        orders = self.orders
        #    except: print("Keyword 'orders' not specified")
        ############### FREQUENCIES ###############
        n       = (2**2)*4096
        freq    = np.fft.rfftfreq(n=n, d=1)
        uppix   = 1./freq
        # we only want to use periods lower that 4096 pixels (as there's no sense to use more)
        cut     = np.where(uppix<=nPix)
        # prepare object for data input
        datatypes = Datatypes(nFiles=self.nfiles[0],nOrder=np.size(orders),fibre=fibre).specdata(add_corr=True)
        datafft   = np.zeros(shape=uppix.shape, dtype=datatypes.ftdata)
        for f in list(fibre):
            for dt in dtype: 
                subdata = self.data[f][dt]
                #print(f,dt,np.shape(subdata))
                for i,o in enumerate(orders):
                    for e in range(subdata.shape[1]):
                        datafft[f][dt][:,e,i] = np.fft.rfft(subdata[:,e,i],n=n)
        self.datafft = datafft[cut]
        self.freq    = uppix[cut]
        return

class ClassCreator(object):
    def __init__(self,nFiles,nOrder,fibre,names):
        dtypes   = {"real":[np.float32 for n in names],"comp":[np.complex64 for n in names]}
        shapes2d = [(nFiles,nOrder) for n in names]
        shapes1d = [(nOrder) for n in names]
        fibres   = list(fibre)
        if 'A' in fibres and 'B' in fibres:
            fibres.append('A-B')
        self.names       = names
        self.real1d      = np.dtype([((names[i],dtypes["real"][i],(shapes1d[i]))) for i in range(len(names))])
        self.real2d      = np.dtype([((names[i],dtypes["real"][i],(shapes2d[i]))) for i in range(len(names))])
        self.comp1d      = np.dtype([((names[i],dtypes["comp"][i],(shapes1d[i]))) for i in range(len(names))])
        self.comp2d      = np.dtype([((names[i],dtypes["comp"][i],(shapes2d[i]))) for i in range(len(names))])
        self.data        = np.dtype({"names":fibres, "formats":[self.real2d for f in fibres]})
        self.median      = np.dtype({"names":fibres, "formats":[self.real1d for f in fibres]})
        self.ftdata      = np.dtype({"names":fibres, "formats":[self.comp2d for f in fibres]})
        self.ftmedian    = np.dtype({"names":fibres, "formats":[self.comp1d for f in fibres]})
        return                
class Datatypes(object):
    def __init__(self,nFiles, nOrder, fibre):
        self.nFiles   = nFiles
        self.nOrder   = nOrder
        self.fibres   = list(fibre)
        
        
    def specdata(self, add_corr=False):
        names    = ["FLX", "ENV", "BKG", "B2E"]
        if add_corr == True:
            names.append("FMB")
        return ClassCreator(self.nFiles,self.nOrder,self.fibres,names)
    def calibrationdata(self):
        names    = ["HARPS", "FOCES", "THAR"]
        return ClassCreator(self.nFiles,self.nOrder,self.fibres,names)
    def custom(self,names):
        return ClassCreator(self.nFiles,self.nOrder,self.fibres,names)
 
class Colours(object):
    def __init__(self):
        #colors=["windows blue", "amber", "pale red", "faded green", "light lavender", "bright orange", "ocean", "saffron"]
        self.palette = [(0.21568627450980393, 0.47058823529411764, 0.7490196078431373),
                        (0.996078431372549, 0.7019607843137254, 0.03137254901960784),
                        (0.8509803921568627, 0.32941176470588235, 0.30196078431372547),
                        (0.4823529411764706, 0.6980392156862745, 0.4549019607843137),
                        (0.8745098039215686, 0.7725490196078432, 0.996078431372549),
                        (1.0, 0.3568627450980392, 0.0),
                        (0.00392156862745098, 0.4823529411764706, 0.5725490196078431),
                        (0.996078431372549, 0.6980392156862745, 0.03529411764705882)]
        
class Plotter(object):
    """ IDEA: separate class for plotting data"""
    def __init__(self,plot_object,figsize=(16,9),**kwargs):
        if   plot_object.__class__ == Manager:
            self.manager = plot_object
            self.plot_object_class = Manager#.__class__
            self.fibre = self.manager.fibre
            self.orders = self.manager.orders
        if   plot_object.__class__ == Spectrum:
            self.spectrum = plot_object
            self.plot_object_class = Spectrum#.__class__
            self.fibre = plot_object.filepath[-6]
        self.fig = plt.figure(figsize=figsize)
        self.defaultparams = (0.1,0.1,0.85,0.85)	# position and size of the canvas
        self.fontsize=12
        self.axes = []
        colours      = [Colours().palette for i in range(20)]
        self.colours = [item for sublist in colours for item in sublist]
        try:
            self.dtype   = kwargs["dtype"]
        except:
            if self.plot_object_class == Manager:
                self.dtype   = self.manager.dtype
            elif self.plot_object_class == Spectrum:
                self.dtype   = ["FLX","ENV","BKG","B2E","FMB"]
        #self.datatypes = Datatypes(self.manager.nfiles[0],nOrder=self.manager.orders,fibre=self.manager.fibre, add_corr=True)
    
        
        
    def create_canvas(self,ctype,size,**kwargs):
        if ctype == "SPECTRUM":
            self.axes.append(self.fig.add_axes(size,**kwargs))
        if ctype == "FOURIER":
            self.axes.append(self.fig.add_axes(size,**kwargs))
        if ctype == "RV":
            self.axes.append(self.fig.add_axes(size,**kwargs))
    def plot(self,dtype,ctype,**kwargs):
        ctype = ctype.upper()
        #additional plot arguments
        try:
            fibre  = list(kwargs["fibre"])
        except: print("Please select fibre(s).")
        try:
            labels = kwargs["legend"]
        except: pass
        try: orders = kwargs["orders"]
        except: 
            try:
                orders = self.orders
            except:
                print("Please specify orders.")
        try: median = kwargs["median"]
        except: median = False
        
        self.get_plot_params(dtype=self.dtype,orders=orders)
        
        if not self.axes:
            naxes = len(fibre) #number of axes
            top, bottom = (0.95,0.08)
            left, right = (0.1,0.95)
            W, H        = (right-left, top-bottom)
            s           = 0.05
            h           = H/naxes - (naxes-1)/naxes*s
            for i in range(naxes):
                down    = top - (i+1)*h - i*s
                size = [left,down,W,h]
                if i==0:
                    self.create_canvas(ctype,size)
                if i>0:
                    self.create_canvas(ctype,size,sharex=self.axes[0],sharey=self.axes[0])
        
            #labels = [np.arange(np.shape(data[f])[1]) for f in fibre]    
        ylims = []
        if ctype=="SPECTRUM":
            for fn,f in enumerate(fibre):
                ax = self.axes[fn]
                for dt in dtype:
                    for i,o in enumerate(orders): 
                        pargs = self.plot_params[f][dt][i]
                        print(pargs["label"])
                        if self.plot_object_class == Manager:
                            if   median == True:
                                data = self.manager.data50p[f][dt][:,i]
                                if dt=="B2E":
                                    data = data*100.
                            elif median == False:
                                data = self.manager.data[f][dt][:,i]
                        elif self.plot_object_class == Spectrum:
                            spec1d  = self.spectrum.extract1d(o)
                            env     = self.spectrum.get_envelope1d(o)
                            bkg     = self.spectrum.get_background1d(o)
                            b2e     = bkg/env
                            
                            fmb         = spec1d['flux']-bkg
                            if   dt == "FLX":
                                data = spec1d['flux']
                            elif dt == "ENV":
                                data = env
                            elif dt == "BKG":
                                data = bkg
                            elif dt == "B2E":
                                data = b2e
                            elif dt == "FMB":
                                data = fmb 
                        try:
                            ax.plot(data, **pargs)
                        except:
                            print("Something went wrong")
                        del(pargs)
                    
                        ylims.append(1.5*np.percentile(data,98))
                    #print(np.percentile(self.manager.data[f][dt],98))
                ax.set_xlim(0,4096)
            print(ylims)
            self.axes[-1].set_xlabel("Pixel")
            self.axes[-1].set_ylim(0,max(ylims))
            
        if ctype=="FOURIER":
            #print("Fourier")
            lst = {"real":'-', "imag":'-.'}
            for fn,f in enumerate(fibre):
                ax = self.axes[fn]
                for dt in dtype:
                    for i,o in enumerate(orders):
                        pargs = self.plot_params[f][dt][i]
                        print(f,dt,i,o,pargs)
                        if self.plot_object_class == Manager:
                            if   median == True:
                                data = self.manager.datafft50p[f][dt][:,i]
                                freq = self.manager.freq
                            elif median == False:
                                data = self.manager.datafft[f][dt][:,i]
                                freq = self.manager.freq
                        elif self.plot_object_class == Spectrum:
                            data = self.spectrum.calculate_fourier_transform()
                            freq = self.spectrum.freq
                        #print(data.real.shape, self.manager.freq.shape)
                        #try:
                        ax.plot(freq, data.real,lw=2.,**pargs)
                        #   print("Plotted")
                        #except:
                        #   print("Something went wrong")                
                ax.set_xscale('log')
            self.axes[-1].set_xlabel("Period [Pixel$^{-1}$]")
        if ctype == "RV":
            bins = 100
            fs = 12 #25 for posters
            # An example of three data sets to compare
            labels = [str(o) for o in orders]
            data_sets = [self.spectrum.get_rv_diff(o) for o in orders]
          
            # Computed quantities to aid plotting
            #hist_range = (np.min([np.min(dd) for dd in data_sets]),
            #              np.max([np.max(dd) for dd in data_sets]))
            hist_range = (-5,5)
            binned_data_sets = [np.histogram(d, range=hist_range, bins=bins)[0]
                                for d in data_sets]
            binned_maximums = np.max(binned_data_sets, axis=1)
            y_locations = np.linspace(0, 1.8*sum(binned_maximums), np.size(binned_maximums))           
            # The bin_edges are the same for all of the histograms
            bin_edges = np.linspace(hist_range[0], hist_range[1], bins+1)
            centers = .5 * (bin_edges + np.roll(bin_edges, 1))[1:]
            widths = np.diff(bin_edges)
            # Cycle through and plot each histogram
            for ax in self.axes:
                ax.axvline(color='k',ls=':',lw=1.5)
                i=0
                for y_loc, binned_data in zip(y_locations, binned_data_sets):
                    #ax.barh(centers, binned_data, height=heights, left=lefts)
                    ax.bar(left=centers,height=binned_data,bottom=y_loc, width=widths,lw=0,color=self.colours[0],align='center')
                    ax.axhline(xmin=0.05,xmax=0.95,y=y_loc,ls="-",color='k')
                    dt = np.where((data_sets[i]>hist_range[0])&(data_sets[i]<hist_range[1]))[0]
                    #ax.annotate("{0:5.3f}".format(median_rv[i]),xy=(median_rv[i],y_loc),xytext=(median_rv[i],y_loc+binned_maximums[i]))
                    for dd in data_sets[i][dt]:
                        ax.plot((dd,dd),(y_loc-2,y_loc-7),color=self.colours[0])
                    i+=1
                
                ax.set_yticks(y_locations)
                ax.set_yticklabels(labels)
                ax.yaxis.set_tick_params(labelsize=fs)
                ax.xaxis.set_tick_params(labelsize=fs)
                
                ax.set_xlabel("Radial velocity [m/s]",fontsize=fs)
                ax.set_ylabel("Echelle order",fontsize=fs)
                
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.tick_params(axis='both', direction='out')
                ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
                ax.get_yaxis().tick_left()
                ax.set_ylim(-10,2.1*sum(binned_maximums))
            plt.show()

        for ax in self.axes:
            ax.legend()
    
        plt.show()
        return
    def get_plot_params(self,orders,**kwargs):
        fibre  = list(self.fibre)
        try:
            dtype = kwargs["dtype"]
        except:
            dtype = self.manager.dtype
        real1d = np.dtype([((dtype[i],list,1)) for i in range(np.size(dtype))])
        self.plot_params = np.empty(shape=np.shape(orders), dtype=np.dtype({"names":fibre, "formats":[real1d for f in fibre]}))
        lstyles = {"FLX":"-","ENV":"--","BKG":":","B2E":"-","FMB":"-"}
        lwidths = {"FLX":1.0,"ENV":2.0,"BKG":2.0,"B2E":1.5,"FMB":1.0}
        #print(fibre,type(orders),orders.shape,self.plot_params)
        for f in fibre:
            j=0
            for dt in dtype:
                k=0
                for i,o in enumerate(orders):
                    label = "{f} {d} {o}".format(f=f,d=dt,o=o)
                    #print(f,dt,o,"label=",label)
                    c=self.colours[i]
                    #print(lstyles[dt])
                    pargs = {"label":label, "c":c,"ls":lstyles[dt], "lw":lwidths[dt]}
                    
                    self.plot_params[f][dt][i] = pargs
                    del(pargs)
                    k+=2
                j+=5
        return
class SelectionFrame(object):
    def __init__(self,xpoints,ypoints):
        self._x_data, self._y_data = [xpoints,ypoints]
        self.peaks                 = peakdet(self._y_data,self._x_data)
        self._x_peaks, self._y_peaks = self.peaks.x, self.peaks.y
        self.create_main_panel()
        self.draw_figure()
        self._is_pick_started = False
        self._picked_indices = None
        self._is_finished = False
        
    def create_main_panel(self):
        self.dpi    = 300
        self.fig    = plt.figure(figsize=(16,9))
        self.canvas = self.fig.canvas
        self.axes   = self.fig.add_axes([0.05,0.05,0.9,0.9])
        #self.toolbar = wxagg.NavigationToolbar2WxAgg(self.canvas)
        #self.vbox = wx.BoxSizer(wx.VERTICAL)
        #self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        #self.vbox.AddSpacer(25)
        #self.vbox.Add(self.toolbar, 0, wx.EXPAND)
        #self.panel.SetSizer(self.vbox)
        #self.vbox.Fit(self)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    def draw_figure(self):
        self.axes.clear()
        
        self.axes.plot(self._x_data, self._y_data, picker=False)
        self.axes.scatter(self._x_peaks, self._y_peaks, c='r', lw=0., picker=2)
        
        self.canvas.draw()
    def on_exit(self, event):
        self.Destroy()

    def picked_points(self):
        if self._picked_indices is None:
            return None
        else:
            return [ [self._x_data[i], self._y_data[i]]
                    for i in self._picked_indices ]

    def on_pick(self, event):
        if not self._is_pick_started:
            self._picked_indices = []
            self._is_pick_started = True

        for index in event.ind:
            if index not in self._picked_indices:
                self._picked_indices.append(index)
                self.axes.scatter(self._x_peaks[index], self._y_peaks[index], c='r', m='d',lw=0.)
                self.canvas.draw_idle()
        print(self.picked_points())

    def on_key(self, event):
        """If the user presses the Escape key then stop picking points and
        reset the list of picked points."""
        if 'r' == event.key:
            self._is_pick_started = False
            self._picked_indices = None
        if 'enter' == event.key:
            print("Selection done")
            self.fig.canvas.mpl_disconnect(self.fig.canvas.mpl_connect('pick_event', self.on_pick))
            self.fig.canvas.mpl_disconnect(self.fig.canvas.mpl_connect('key_press_event', self.on_key))
        if 'escape' == event.key:
            self._is_finished=True
        return
    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.fig.canvas.mpl_connect('key_press_event', self.press))
    def pick_lambda(self):
        ll = np.zeros(shape=(np.size(self._picked_points)))
        for i in range(ll.size):
            l = input("{}, lambda = ",format(i))
            ll[i] = l
        self._picked_lambda = ll
        return
################################################################################################################
########################################## F U N C T I O N S ###################################################
################################################################################################################
def chisq(params,x,data,weights=None):
    amp, ctr, sgm = params
    if weights==None:
        weights = np.ones(x.shape)
    fit    = gauss3p(x,amp,ctr,sgm)
    chisq  = ((data - fit)**2/weights).sum()
    return chisq
def combine_line_list(theoretical,measured):
    combined = []
    lim = 0.7*np.median(np.diff(theoretical))
    for value in theoretical:
        distances = np.abs(measured-value)
        closest   = distances.min()
        if closest <= lim:
            combined.append(measured[distances.argmin()])
        else:
            combined.append(value)
    return np.array(combined)
def cut_patch(df,i):
    ''' Returns a Pandas Series with the values of wavelengths in patch i'''
    pix = df['pixel']
    cut = np.where((pix>=i*512)&(pix<(1+i)*512))[0]
    print(cut)
    return df.iloc[cut]
def derivative1d(y,x,n=1,method='central'):
    ''' Get the first derivative of a 1d array y. 
    Modified from 
    http://stackoverflow.com/questions/18498457/
    numpy-gradient-function-and-numerical-derivatives'''
    if method=='forward':
        dx = np.diff(x,n)
        dy = np.diff(y,n)
        d  = dy/dx
    if method == 'central':
        z1  = np.hstack((y[0], y[:-1]))
        z2  = np.hstack((y[1:], y[-1]))
        dx1 = np.hstack((0, np.diff(x)))
        dx2 = np.hstack((np.diff(x), 0))  
        #print("Zeros in dx1+dx2",np.where((dx1+dx2)==0)[0].size)
        #print(z2-z1)
        d   = (z2-z1) / (dx2+dx1)
    return d
def find_nearest(array1,array2):
    idx = []
    lim = np.median(np.diff(array1))
    for value in array1:
        distances = np.abs(array2-value)
        closest   = distances.min()
        if closest <= lim:
            idc = distances.argmin()
            idx.append(idc)
        else:
            continue
    return array2[idx]
def fit_peak(i,xarray,yarray,weights,xpos,dx,model,method='erfc'):
    '''
    Returns the parameters of the fit for the i-th peak of a single echelle 
    order.
    
    Args:
        xarray:   pixels of the echelle order
        yarray:   flux of the echelle order
        weigths:  weights of individual pixels
        xpos:     positions of individual peaks, determined by get_extreme(max)
        dx:       distances between individual peaks (i.e. np.diff(xpos))
        model:    Gaussian function
        method:   fitting method, default: curve_fit
        
    Returns:
        params:   parameters returned by the fitting procedure
    '''
    def calculate_photon_noise(weights):
        return 1./np.sqrt(weights.sum())*299792458e0
        
    # Fit only data within a certain distance from the i-th peak
    # Fit data that are inside the range [(x(i-1)+x(i))/2, (x(i)+x(i+1))/2]
    if   i == 0:
        cut = xarray.loc[((xarray>=(3*xpos[i]-xpos[i+1])/2.)&
                          (xarray<=(xpos[i+1]+xpos[i])/2.))].index
    elif i<=np.size(xpos)-2:
        cut = xarray.loc[((xarray>=(xpos[i-1]+xpos[i])/2.)&
                          (xarray<=(xpos[i+1]+xpos[i])/2.))].index
    elif i == np.size(xpos)-1:
        cut = xarray.loc[((xarray>=(xpos[i-1]+xpos[i])/2.)&
                          (xarray<=(3*xpos[i]-xpos[i-1])/2.))].index
    # If this selection is not an empty set, fit the Gaussian profile
    if cut.size>4:
        x    = xarray.iloc[cut]#.values
        y    = yarray.iloc[cut]#.values
        wght = weights[cut]
        pn   = calculate_photon_noise(wght)
        ctr  = xpos[i]
        amp  = np.max(yarray.iloc[cut])
        sgm  = 3*dx[i]
        #print("{},{}/{}".format(scale,i,npeaks))
        if method=='lmfit':
            gmodel            = lmfit.Model(gauss3p)
            params            = gmodel.make_params(center=ctr,
                                                   amplitude=amp,
                                                   sigma=sgm)
            result            = gmodel.fit(y,params,
                                           x=x,
                                           weights=wght)
#                    lines_fit.iloc[i] = result.best_values
            best_pars         = result.best_values.values()
        elif method == 'curve_fit':              
            guess                          = [amp, ctr, sgm] 
            #print("{} {},{}/{} {} {}".format(order,scale,i,npeaks,guess,cut.size))
            try:
                best_pars, pcov                = curve_fit(model, 
                                                          x, y, 
                                                          p0=guess)
            except:
                return ((-1.0,-1.0,-1.0),np.nan)
#                    lines_fit.iloc[i]['amplitude'] = best_pars[0]
#                    lines_fit.iloc[i]['center']    = best_pars[1]
#                    lines_fit.iloc[i]['sigma']     = best_pars[2]
        elif method == 'chisq':
            params                      = [amp, ctr, sgm] 
            result                      = minimize(chisq,params,
                                                   args=(x,y,wght))
            best_pars                      = result.x
#                    lines_fit.iloc[i]['amplitude'] = result.x[0]
#                    lines_fit.iloc[i]['center']    = result.x[1]
#                    lines_fit.iloc[i]['sigma']     = result.x[2]
        elif method == 'erfc':
            params = [amp,ctr,sgm]
            line   = EmissionLine(x,y,yerr=None,weights=wght)
            best_pars, pcov = line.fit()
        else:
            sys.exit("Method not recognised!")
    return np.concatenate((best_pars,np.array([pn])))
def flatten_list(inlist):
    outlist = [item for sublist in inlist for item in sublist]
    return outlist
def gauss4p(x, amplitude, center, sigma, y0 ):
    # Four parameters: amplitude, center, width, y-offset
    #y = np.zeros_like(x,dtype=np.float64)
    #A, mu, sigma, y0 = p
    y = y0+ amplitude*np.exp((-((x-center)/sigma)**2)/2.)
    return y
def gauss3p(x, amplitude, center, sigma):
    # Three parameters: amplitude, center, width
    #y = np.zeros_like(x)
    #A, mu, sigma = p
    y = amplitude*np.exp((-((x-center)/sigma)**2)/2.)
    return y
def gaussN(x, params):
    N = params.shape[0]
    # Three parameters: amplitude, center, width
    y = np.zeros_like(x)
    #A, mu, sigma = p
    for i in range(N):
        a,c,s,ct = params.iloc[i]
        y = y + a*np.exp((-((x-c)/s)**2)/2.)
    return y
def get_fig_axes(naxes,ratios=None,title=None,sep=0.05,alignment="vertical",
                 figsize=(16,9),sharex=None,sharey=None,grid=None,
                 subtitles=None,presentation=False,
                 left=0.1,right=0.95,top=0.95,bottom=0.08,**kwargs):
    
    def get_grid(alignment,naxes):
        if alignment=="grid":
            ncols = np.int(round(np.sqrt(naxes)))
            nrows,lr = [np.int(k) for k in divmod(naxes,round(np.sqrt(naxes)))]
            if lr>0:
                nrows += 1     
        elif alignment=="vertical":
            ncols = 1
            nrows = naxes
        elif alignment=="horizontal":
            ncols = naxes
            nrows = 1
        grid = np.array([ncols,nrows],dtype=int)
        return grid
    
    fig         = plt.figure(figsize=figsize)
    
    # Change color scheme and text size if producing plots for a presentation
    if presentation==True:
        spine_col = 'w'
        text_size = 20
    else:
        pass
    
    # Share X axis
    if sharex!=None:
        if type(sharex)==list:
            pass
        else:
            sharex = list(sharex for i in range(naxes))
    elif sharex==None:
        sharex = list(False for i in range(naxes))
    # First item with sharex==True:
    try:
        firstx = sharex.index(True)
    except:
        firstx = None
    # Share Y axis  
    if sharey!=None:
        if type(sharey)==list:
            pass
        else:
            sharey = list(sharey for i in range(naxes))
    elif sharey==None:
        sharey = list(False for i in range(naxes))
    # First item with sharey==True:
    try:
        firsty = sharey.index(True)
    except:
        firsty = None
    
    sharexy = [(sharex[i],sharey[i]) for i in range(naxes)]
    
    # Add title
    if title!=None:
        fig.suptitle(title)
    # Calculate canvas dimensions
    
    # GRID
    if grid==None:
        grid = get_grid(alignment,naxes)
    else:
        grid = np.array(grid,dtype=int)
    ncols,nrows = grid

    if ratios==None:
        ratios = np.array([np.ones(ncols),np.ones(nrows)])
    else:
        if np.size(np.shape(ratios))==1:
            if   alignment == 'vertical':
                ratios = np.array([np.ones(ncols),ratios])
            elif alignment == 'horizontal':
                ratios = np.array([ratios,np.ones(nrows)])
        elif np.size(np.shape(ratios))==2:
            ratios = np.array(ratios).reshape((ncols,nrows))
    top, bottom = (top,bottom)
    left, right = (left,right)
    W, H        = (right-left, top-bottom)
    s           = sep
    #h           = H/naxes - (naxes-1)/naxes*s
    
    h0          = (H - (nrows-1)*s)/np.sum(ratios[1])
    w0          = (W - (ncols-1)*s)/np.sum(ratios[0])
    axes        = []
    axsize      = []

    for c in range(ncols):
        for r in range(nrows):
            ratiosc = ratios[0][:c]
            ratiosr = ratios[1][:r+1]
            w  = ratios[0][c]*w0
            h  = ratios[1][r]*h0
            l  = left + np.sum(ratiosc)*w0 + c*s
            d  = top - np.sum(ratiosr)*h0 - r*s
            size  = [l,d,w,h] 
            axsize.append(size)       
    for i in range(naxes):   
        size   = axsize[i]
        sharex,sharey = sharexy[i]
        if i==0:
            axes.append(fig.add_axes(size))
        else:
            kwargs = {}
            if   (sharex==True  and sharey==False):
                kwargs["sharex"]=axes[firstx]
                #axes.append(fig.add_axes(size,sharex=axes[firstx]))
            elif (sharex==False and sharey==True):
                kwargs["sharey"]=axes[firsty]
                #axes.append(fig.add_axes(size,sharey=axes[firsty]))
            elif (sharex==True  and sharey==True):
                kwargs["sharex"]=axes[firstx]
                kwargs["sharey"]=axes[firsty]
                #axes.append(fig.add_axes(size,sharex=axes[firstx],sharey=axes[firsty]))
            elif (sharex==False and sharey==False): 
                pass
                #axes.append(fig.add_axes(size))
            axes.append(fig.add_axes(size,**kwargs))
    if presentation == True:
        for a in axes:
            plt.setp(tuple(a.spines.values()), color=spine_col)
            plt.setp([a.get_xticklines(), a.get_yticklines(),a.get_xticklabels(),a.get_yticklabels()], color=spine_col)
            plt.setp([a.get_xticklabels(),a.get_yticklabels()],size=text_size)
#            plt.setp([a.get_xlabel(),a.get_ylabel()],color=spine_col,size=text_size)
            #plt.setp(a.get_yticklabels(),visible=False)
    else:
        pass
    
    return fig,axes

def get_extreme(xarr,yarr,extreme="max",kind="LFC",thresh=0.1):
    ''' Calculates the positions of LFC profile peaks from data.
    In:
    ---   xarr,yarr (array-like, size=N (number of datapoints))
          extreme(str) = "min" or "max"
    Out:
    ---   peakpos(array-like, size=M (number of detected peaks))'''
    if extreme=='min':
        debugging=False   
    else:
        debugging=False
    if debugging:
        print("EXTREME = {}".format(extreme))
    
    # Calculate the first order numerical derivation of fluxes with respect to wavelength
    dy  = pd.Series(np.gradient(yarr,1,edge_order=2))

    df     = pd.DataFrame({"x":xarr, "xn":xarr.shift(1), "xp":xarr.shift(-1),
                           "y":yarr, "yn":yarr.shift(1), "yp":yarr.shift(-1),
                           "dx":xarr.diff(1), "dy":dy, 
                           "dyn":dy.shift(1)})
    # Find indices where two neighbouring points have opposite dy signs. 
    # We can now identify indices, p, for which i-th element of dy is +, dyn is -, ###and d2s is - (condition for maximum of a function)
    if extreme == "max":
        p = df.loc[(df.dy<=0.)&(df.dyn>0.)].index
    elif extreme == "min":
        p = df.loc[(df.dy>=0.)&(df.dyn<0.)].index
    # Simple linear interpolation to find the position where dx=0.
    xpk0  = (df.x - (df.xn-df.x)/(df.dyn-df.dy)*df.dy)[p].reset_index(drop=True)
    # Remove NaN values   
    xpk1 = xpk0.dropna()
    # Flux at the extremum is estimated as the maximum/minumum value of the two
    # points closest to the extremum
    if extreme == "max":
        ypk1 = df[["y","yn"]].iloc[p].max(axis=1).reset_index(drop=True)
    elif extreme == "min":
        ypk1 = df[["y","yn"]].iloc[p].min(axis=1).reset_index(drop=True)
    if kind == "LFC": 
        if extreme == "max":
            llim      = (df.y.max()-df.y.min())*thresh
            countmask = ypk1.loc[ypk1>=llim].index
            xpk       = xpk1[countmask].reset_index(drop=True)
            ypk       = ypk1[countmask].reset_index(drop=True)
        elif extreme == "min":
            xpk = xpk1
            ypk = ypk1
            pass
    peaks = pd.DataFrame({"x":xpk,"y":ypk})
        
    return peaks
def get_time(worktime):
    """
    Returns the work time in hours, minutes, seconds

    Outputs:
    --------
           h : hour
           m : minute
           s : second
    """					
    m,s = divmod(worktime, 60)
    h,m = divmod(m, 60)
    h,m,s = [int(value) for value in (h,m,s)]
    return h,m,s
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.
    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
    Returns:
    --------
        mask : A numobservations-length boolean array.
    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def is_outlier2(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    median = np.median(points, axis=0)
    if len(points.shape) == 1:
        points = points[:,None]
    
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(np.abs(diff))

    med_abs_deviation = np.median(diff)

    modified_z_score = (0.6745 * diff) / med_abs_deviation
    flag = np.isnan(modified_z_score).any()
    if flag == True:
        nans = np.where(np.isnan(modified_z_score)==True)[0]
        modified_z_score[nans] = np.Inf
                        
    
    return modified_z_score > thresh
def is_outlier_running(points, window=5,thresh=1.):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
#    plt.figure()
#    plt.plot(points,label='data')
    rmean = runningMeanFast(points,window)
#    rmean = np.percentile(points,85)
    if len(points.shape) == 1:
        points = points[:,None]  
    diff  = np.sum((points-rmean)**2,axis=-1)
    diff  = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
#    plt.plot(rmean,label='rmean')
#    plt.plot(diff,label='diff')
#    plt.plot(modified_z_score,label='z_score')
#    plt.legend()
#    plt.show()
    return modified_z_score > thresh
def is_outlier_original(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
def is_peak(points):
    """
    Returns a boolean array with True if points are peaks and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    return points>med_abs_deviation
def mad(x):
    ''' Returns median absolute deviation of input array'''
    return np.median(np.abs(np.median(x)-x))
def nearest_neighbors(x, y) :
    x, y = map(np.asarray, (x, y))
    y = y.copy()
    y_idx = np.arange(len(y))
    nearest_neighbor = np.empty((len(x),), dtype=np.intp)
    for j, xj in enumerate(x) :
        idx = np.argmin(np.abs(y - xj))
        nearest_neighbor[j] = y_idx[idx]
        y = np.delete(y, idx)
        y_idx = np.delete(y_idx, idx)

    return nearest_neighbor  
def peakdet(y_axis, x_axis = None, extreme='max', lookahead=8, delta=0):
    '''
    https://gist.github.com/sixtenbe/1178136
    '''
    if delta == 0:
        if extreme is 'max':
            delta = np.percentile(y_axis,10)
        elif extreme is 'min':
            delta = 0
    maxima,minima = [np.array(a) for a 
                     in peakdetect(y_axis, x_axis, lookahead, delta)]
    if extreme is 'max':
        peaks = pd.DataFrame({"x":maxima[:,0],"y":maxima[:,1]})
    elif extreme is 'min':
        peaks = pd.DataFrame({"x":minima[:,0],"y":minima[:,1]})
    return peaks
def peakdet2(xarr,yarr,delta=None,extreme='max'):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
    v = np.asarray(yarr)
    if xarr is None:
        xarr = np.arange(len(v))
    if delta is None:
#        delta = runningMeanFast(v,35)
        delta = np.percentile(v,p)
    if np.isscalar(delta):
        delta = np.full(xarr.shape,delta,dtype=np.float32)
    if len(yarr) != len(xarr):
        sys.exit('Input vectors v and x must have same length')
  
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True
    
    for i in range(len(v)):
        this = v[i]
        d    = delta[i]
        if this > mx:
            mx = this
            mxpos = xarr[i]
        if this < mn:
            mn = this
            mnpos = xarr[i]
        
        if lookformax:
            if this < mx-d:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = xarr[i]
                lookformax = False
        else:
            if this > mn+d:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = xarr[i]
                lookformax = True
       
    maxtab = np.array(maxtab)
    mintab = np.array(mintab)
    if extreme=='max':
        peaks = pd.DataFrame({"x":maxtab[:,0],"y":maxtab[:,1]})
    elif extreme=='min':
        peaks = pd.DataFrame({"x":mintab[:,0],"y":mintab[:,1]})
    return peaks
         
def polynomial(x, *p):
    y = np.zeros_like(x,dtype=np.float64)
    for i,a in enumerate(p):
        y = y + a*x**i
    return y
def polynomial1(x, a0,a1):
    return a0 + a1*x 
def polynomial2(x, a0,a1,a2):
    return a0 + a1*x + a2*x**2 
def polynomial3(x, a0,a1,a2,a3):
    return a0 + a1*x + a2*x**2 + a3*x**3
def polynomial4(x, a0,a1,a2,a3,a4):
    return a0 + a1*x + a2*x**2 + a3*x**3 +a4*x**4
def polynomial5(x, a0,a1,a2,a3,a4,a5):
    return a0 + a1*x + a2*x**2 + a3*x**3 +a4*x**4 + a5*x**5
def rms(x):
    ''' Returns root mean square of input array'''
    return np.sqrt(np.mean(np.square(x)))
def runningMeanFast(x, N):
        return np.convolve(x, np.ones((N,))/N)[(N-1):]
def select_orders(orders):
    use = np.zeros((nOrder,),dtype=bool); use.fill(False)
    for order in range(sOrder,eOrder,1):
        if order in orders:
            o = order - sOrder
            use[o]=True
    col = np.where(use==True)[0]
    return col