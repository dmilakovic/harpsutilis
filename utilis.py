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
import urllib

import pandas as pd
#import lmfit
#from lmfit.models import GaussianModel
from harps.peakdetect import peakdetect
import xarray as xr
from joblib import Parallel,delayed
import h5py

from scipy.special import erf,erfc
from scipy.linalg import svd
from scipy.optimize import curve_fit, fsolve, newton, brentq
from scipy.optimize import minimize, leastsq, least_squares, OptimizeWarning, fmin_ncg
from scipy.optimize._lsq.least_squares import prepare_bounds
from scipy import odr
## IMPORT ENVIRONMENT VARIABLES AND USE THEM FOR OUTPUT
harps_home   = os.environ['HARPSHOME']
harps_data   = os.environ['HARPSDATA']
harps_dtprod = os.environ['HARPSDATAPROD']

harps_prod   = os.path.join(harps_dtprod,'products')
harps_plots  = os.path.join(harps_dtprod,'plots')


## 
nproc = 10

## first and last order in a spectrum
chip   = 'red'
if chip == 'red':
    sOrder = 42   
    eOrder = 72
elif chip == 'blue':
    sOrder = 25
    eOrder = 41
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
        self.wavesol_thar = None
        self.wavesol_LFC  = None
        self.fr_source = 250e6 #Hz
        self.f0_source = -50e6 #Hz
        gapsfilepath    = os.path.join(harps_prod,'gapsA.npy')
        self.gapsfile   = np.load(gapsfilepath)
        gaps            = np.zeros(shape=(eOrder+1,7))
        gorders         = np.array(self.gapsfile[:,0],dtype='i4')
        gaps[gorders,:] = np.array(self.gapsfile[:,1:],dtype='f8')
        self.gaps       = gaps
        
        self.polyord    = 8
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
        
        try:
            anchor = self.anchor
        except:
            self.__read_LFC_keywords__()
        return
    def check_and_get_wavesol(self,calibrator='LFC',orders=None):
        ''' Check and retrieve the wavelength calibration'''
        wavesol_name = 'wavesol_{cal}'.format(cal=calibrator)
        exists_calib = False if getattr(self,wavesol_name) is None else True
               
        # Run wavelength calibration if the wavelength calibration has not yet 
        # been performed  
        if exists_calib == False:
            wavesol = self.__get_wavesol__(calibrator,orders=orders)
        else:
            # Load the existing wavelength calibration and check if all wanted
            # orders have been calibrated
            wavesol = getattr(self,wavesol_name)
            ws_exists_all = np.all(wavesol[orders])
            if ws_exists_all == False:
                wavesol = self.__get_wavesol__(calibrator,orders=orders)
        return wavesol
    def check_and_get_comb_lines(self,calibrator='LFC',orders=None):
        ''' Check and retrieve the positions of lines '''
        
        # Check if the Spectrum instance already has lines
        exists_lines = hasattr(self,'lines')
        if exists_lines == False:
            wavesol = self.check_and_get_wavesol(calibrator,orders)
            lines = self.lines
        else:
            lines = self.lines
            lines_exist_all = np.isnan(lines.sel(typ='pix',od=orders)).any()
            if lines_exist_all == True:
                wavesol = self.check_and_get_wavesol('LFC',orders)
                lines = self.lines
        return lines
    def check_and_load_psf(self):
        exists_psf = hasattr(self,'psf')
        if not exists_psf:
            self.load_psf()
        else:
            pass
        segments        = np.unique(self.psf.coords['seg'].values)
        N_seg           = len(segments)
        segment_limits  = sl = np.linspace(0,4096,N_seg+1)
        segment_centers = sc = (sl[1:]+sl[:-1])/2
        segment_centers[0] = 0
        segment_centers[-1] = 4096
        
        self.segments        = segments
        self.nsegments       = N_seg
        self.segment_centers = segment_centers
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
        # Fibre information is not saved in the header, but can be obtained 
        # from the filename 
        self.fibre   = self.filepath[-6]
    def __read_LFC_keywords__(self):
        try:
            #offset frequency of the LFC, rounded to 1MHz
            self.anchor = round(self.header["HIERARCH ESO INS LFC1 ANCHOR"],-6) 
            #repetition frequency of the LFC
            self.reprate = self.header["HIERARCH ESO INS LFC1 REPRATE"]
        except:
            self.anchor       = 288059930000000.0 #Hz, HARPS frequency 2016-11-01
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
        
        
    def __read_data__(self,convert_to_e=True):
        ''' Method to read data from the FITS file
        Args:
        ---- 
            convert_to_e : convert the flux to electron counts'''
        if len(self.hdulist)==0:
            self.hdulist = fits.open(self.filepath,memmap=False)
        if   self.ftype=="s1d" or self.ftype=="e2ds":
            data = self.hdulist[0].data.copy()
        elif self.ftype=="":
            data = self.hdulist[1].data.copy()
        
        if convert_to_e is True:
            data = data * self.conad
            self.data_units = "e-"
        else:
            self.data_units = "ADU"
        self.data = data
        return self.data
    def __get_wavesol__(self,calibrator="ThAr",nobackground=True,vacuum=True,
                        orders=None,method='erfc',model=None,
                        patches=False,gaps=True,
                        polyord=8,**kwargs):
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
        def patch_fit(patch,polyord,method='ord'):
            data      = patch.lbd
            data_err  = patch.lbd_err
            x         = patch.pix
            x_err     = patch.pix_err
#            print(data, data_err)
#            print(x,x_err)              
            data_axis = np.array(data.values,dtype=np.float64)
            data_err  = np.array(data_err.values,dtype=np.float64)
            x_axis    = np.array(x.values,dtype=np.float64)
            x_err     = np.array(x_err.values,dtype=np.float64)
            datanan,xnan = (np.isnan(data_axis).any(),np.isnan(x_axis).any())
            if (datanan==True or xnan==True):
                print("NaN values in data or x")
            if x_axis.size>self.polyord:
                coef = np.polyfit(x_axis,data_axis,self.polyord)
                if method == 'curve_fit':
                    coef,pcov = curve_fit(polynomial,x_axis,data_axis,p0=coef[::-1])
                    coef = coef[::-1]
                    coef_err = []
                if method == 'ord':
                    data  = odr.RealData(x_axis,data_axis,sx=x_err,sy=data_err)
                    model = odr.polynomial(order=self.polyord)
                    fit   = odr.ODR(data,model,beta0=coef)
                    out   = fit.run()
                    coef  = out.beta
                    coef_err = out.sd_beta
            else: 
                coef = None  
                coef_err = None
            return coef,coef_err
        
        def fit_wavesol(pix,lbd,pix_err,lbd_err,patches=True):
            if patches==True:
                npt = 8
            else:
                npt = 1
            ps = 4096/npt
            
            
            cc     = pd.concat([lbd,pix,lbd_err,pix_err],
                                axis=1,keys=['lbd','pix','lbd_err','pix_err'])
            
            cc = cc.dropna(how='any').reset_index(drop=True)
            ws     = np.zeros(4096)
            # coefficients and residuals
            cf = np.zeros(shape=(npt,self.polyord+1))
            rs = pd.Series(index=pix.index)
            
            
            for i in range(npt):
                ll,ul     = np.array([i*ps,(i+1)*ps],dtype=np.int)
                patch     = cc.where((cc.pix>=ll)&
                                     (cc.pix<ul)).dropna()
                
                if patch.size>self.polyord:
                    pixels        = np.arange(ll,ul,1,dtype=np.int)
                    coef,coef_err = patch_fit(patch,self.polyord)
                    if coef is not None:
                        fit       = np.polyval(coef,patch.pix)
                        
                        residuals = np.array(patch.lbd.values-fit,dtype=np.float64)
                        rs.iloc[patch.index]=residuals
                        outliers  = is_outlier(residuals,5)
                        if np.any(outliers)==True:
                            patch['outlier']=outliers
                            newpatch = patch.where(patch.outlier==False).dropna(how='any')
                            coef,coef_err = patch_fit(newpatch,self.polyord) 
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
        # self.polyord = polyord
        # wavesol(72,4096) contains wavelengths for 72 orders and 4096 pixels
        
        if calibrator is "ThAr": 
            # If this routine has not been run previously, read the calibration
            # coefficients from the FITS file. For each order, derive the 
            # calibration in air. If 'vacuum' flag is true, convert the 
            # wavelengths to vacuum wavelengths for each order. 
            # Finally, derive the coefficients for the vacuum solution.

            # If wavesol_thar has not been initialised:
            
            if (self.wavesol_thar is None or self.wavesol_thar.sum()==0):
                wavesol_thar = np.zeros(shape=(self.nbo,self.npix,), 
                                        dtype=np.float64)
            else:
                wavesol_thar = self.wavesol_thar
            ws_thar_exists_all = np.all(wavesol_thar[orders])
            if ws_thar_exists_all == False:
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
                
            # If this routine has been run previously, check 
            
                


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
                wavecoef_LFC = np.zeros(shape = (self.nbo,self.polyord+1,npt), 
                                        dtype = np.float64)
            else:
                wavecoef_LFC = np.zeros(shape = (self.nbo,self.polyord+1,npt), 
                                        dtype = np.float64)
            # Save positions of lines
            if method =='erfc':
                model = model if model is not None else 'singlegaussian'
            if model == 'singlegaussian':
                dset_names = ['wave','pix','photon_noise','R2',
                                'cen','amp','sig','cen_err']
            else:
                dset_names   = ['wave','pix','photon_noise','R2',
                                'cen1','cen2','amp1','amp2','sig1','sig2',
                                'cen','cen_err']
            cc_data      = xr.DataArray(np.full((nOrder,len(dset_names),500),np.NaN),
                                        dims=['od','typ','val'],
                                        coords=[np.arange(sOrder,eOrder),
                                                dset_names,
                                                np.arange(500)])
            # Save residuals to the fit
            rsd          = xr.DataArray(np.full((nOrder,500),np.NaN),
                                        dims=['od','val'],
                                        coords=[np.arange(sOrder,eOrder),
                                                np.arange(500)])
            # Check if a ThAr calibration is attached to the Spectrum.
            # Priority given to ThAr calibration provided directly to the 
            # function. If none given, see if one is already attached to the 
            # Spectrum. If also none, run __get_wavesol__('ThAr')
            
            kwarg_wavesol_thar = kwargs.get('wavesol_thar',False)
            if kwarg_wavesol_thar != False:
                self.wavesol_thar     = kwarg_wavesol_thar
                self.wavecoef_air = kwargs.pop('wavecoeff_air',_get_wavecoeff_air())
                
#                try:
#                    self.wavecoeff_air = kwargs['wavecoeff_air']
#                except:
#                    self.wavecoeff_air = _get_wavecoeff_air()
                self.wavecoeff_vacuum = _get_wavecoeff_vacuum()
            elif self.wavesol_thar is not None:
                pass
            else:
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
                print("ORDER = ",order)
                lines     = self.fit_lines(order,scale='pixel',method=method)
                lines     = lines.dropna()
                
                # Include the gaps
                if gaps is True:
                    g0 = self.gaps[order,:]
                    XX = self.introduce_gaps(lines['center'],g0)
                    lines['center'] = XX
                elif gaps is False:
                    pass
                indx = lines.index.values
                cc_data.loc[dict(typ='wave',od=order)][indx] = lines.th_wave.values
                cc_data.loc[dict(typ='pix',od=order)][indx] = lines.center.values   
                cc_data.loc[dict(typ='photon_noise',od=order)][indx] = lines.photon_noise.values   
                cc_data.loc[dict(typ='R2',od=order)][indx] = lines.r2.values 
                if ((model == 'doublegaussian')or(model=='simplegaussian')):     
                    cc_data.loc[dict(typ='cen1',od=order)][indx] = lines.center1.values  
                    cc_data.loc[dict(typ='cen2',od=order)][indx] = lines.center2.values            
                    cc_data.loc[dict(typ='sig1',od=order)][indx] = lines.sigma1.values
                    cc_data.loc[dict(typ='sig2',od=order)][indx] = lines.sigma2.values
                    cc_data.loc[dict(typ='amp1',od=order)][indx] = lines.amplitude1.values
                    cc_data.loc[dict(typ='amp2',od=order)][indx] = lines.amplitude2.values
                elif model == 'simplegaussian':
                    cc_data.loc[dict(typ='sig',od=order)][indx]  = lines.sigma.values
                cc_data.loc[dict(typ='cen', od=order)][indx] = lines.center.values
                cc_data.loc[dict(typ='cen_err', od=order)][indx] = lines.center_err.values
                LFC_wavesol,coef,residuals = fit_wavesol(
                                               lines['center'],
                                               lines.th_wave,
                                               lines['center_err'],
                                               pd.Series(np.zeros_like(lines.th_wave)),
                                               patches=patches
                                               )
                
                wavesol_LFC[order]  = LFC_wavesol
                wavecoef_LFC[order] = coef.T        
                rsd.loc[dict(od=order)][indx] = residuals
            self.wavesol_LFC  = wavesol_LFC
            self.lines        = cc_data
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
#        dtype     = datatypes.names
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
#        photon_noise2d      = np.zeros(shape=self.weights2d.shape)
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
                
    def cut_lines(self,order,nobackground=True,vacuum=True,
                  columns=['pixel','flux']):
        ''' Returns a tuple of dictionaries. The keys of each dictionary consist
            of order numbers. The values of in the dictionary are list of arrays. 
            List items are for each LFC line. (Needs better description)
            
            Output:
            -------
            ({order:list of arrays for order in orders} for column in columns)
        '''                
            
        orders = self.prepare_orders(order)
       
        dicts  = {col:{} for col in columns}

        for order in orders:
            
            lists = {col:[] for col in columns}
            # Extract data from the fits file
            spec1d  = self.extract1d(order=order,
                                     nobackground=nobackground,
                                     vacuum=vacuum,
                                     columns=columns)
            if (('pixel' in columns) and ('flux' in columns)):
                xarray  = spec1d.pixel
                yarray  = spec1d.flux
            else:
                raise ValueError("No pixel or flux columns")
            
            # find minima and number of LFC lines                
            minima  = peakdet(yarray,xarray,extreme='min')
            xmin    = minima.x
            npeaks  = np.size(xmin)-1
            
            for i in range(npeaks):
                index = xarray.loc[((xarray>=xmin[i])&(xarray<=xmin[i+1]))].index
                cut   = spec1d.loc[index]
                for col in columns:
                    l = lists[col]
                    if col == 'bary':
                        x = cut['pixel']
                        y = cut['flux']
                        b = np.sum(x*y) / np.sum(y)
                        l.append(b)
                    else:
                        
                        l.append(cut[col].values)
           
                    
            for col in columns:
                dicts[col][order]=lists[col]   
        return tuple(dicts[col] for col in columns)
        
    def extract1d(self,order,scale='pixel',nobackground=False,
                  vacuum=True,columns=['pixel','wave','flux','error'],**kwargs):
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
        include = {}
        if 'wave' in columns:
            calibrator   = kwargs.get('calibrator','ThAr')
            #print("extract1d",calibrator)
            wavesol      = kwargs.get('wavesol',None)
            if wavesol is None:  
    #            print(self.wavesol_thar, calibrator)
                if (self.wavesol_thar is None or np.sum(self.wavesol_thar[order])==0):
                    #print("No existing thar wavesolution for this order")
                    wavesol = self.__get_wavesol__(calibrator,orders=[order],
                                                   vacuum=vacuum)            
                else:
                    #print("Existing thar wavesolution for this order")
                    wavesol = self.wavesol_thar  
            wave1d  = pd.Series(wavesol[order])
            include['wave']=wave1d
        if 'pixel' in columns:
            pix1d   = pd.Series(np.arange(4096),dtype=np.float64)
            include['pixel']=pix1d
        if 'flux' in columns:
            flux1d  = pd.Series(self.data[order])
            
        if 'error' in columns:
            # Assuming the error is simply photon noise
            error1d = pd.Series(np.sqrt(self.data[order]))
            include['error']=error1d
                
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
        if nobackground is True:
            flux1d     = flux1d - background
        if 'flux' in columns:
            include['flux']=flux1d
        if 'bkg' in columns:
            bkg1d = pd.Series(background)
            include['bkg']=bkg1d
        spec1d  = pd.DataFrame.from_dict(include)
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

    def fit_lines(self,order,nobackground=True,method='erfc',model=None,
                  scale='pixel',remove_poor_fits=False,verbose=0):
        """Fits LFC lines of a single echelle order.
        
        Extracts a 1D spectrum of a selected echelle order and fits a single 
        Gaussian profile to each line, in both wavelength and pixel space. 
        
        Args:
            order: Integer number of the echelle order in the FITS file.
            nobackground: Boolean determining whether background is subtracted 
                before fitting is performed.
            method: String specifying the method to be used for fitting. 
                Options are 'curve_fit', 'lmfit', 'chisq', 'erfc'.
            remove_poor_fits: If true, removes the fits which are classified
                as outliers in their sigma values.
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
            #output_lines = {}
            df = input_lines
            #for scale,df in input_lines.items():
            if 'sigma1' in df.columns:
                sigma = np.array(df.sigma1.values,dtype=np.float32)#[1:]
            else:
                sigma = np.array(df.sigma.values,dtype=np.float32)
#                centd = np.array(df.center.diff().dropna().values,dtype=np.float32)
#                ind   = np.where((is_outlier2(sigma,thresh=4)==True) |  
#                                 (is_outlier2(centd,thresh=4)==True))[0]
            # Outliers in sigma
            ind1  = np.where((is_outlier(sigma)==True))[0]
#                ind2  = np.where((is_outlier(centd)==True))[0]
            # Negative centers
            if 'center1' in df.columns:    
                ind3  = np.where(df.center1<0)[0]
            else:
                ind3  = np.where(df.center<0)[0]
            ind   = np.union1d(ind1,ind3)
            xi.append(ind)
                
            #a1,a2  = xi
            #xmatch = np.intersect1d(a2, a1)
            xmatch=[0]
            #for scale,df in input_lines.items():
            newdf = df.drop(df.index[xmatch])
            output_lines = newdf  
                
            return output_lines
                
        #######################################################################
        #                        MAIN PART OF fit_lines                       #
        #######################################################################
        # Debugging
        plot=False
        if verbose>0:
            print("ORDER:{0:<5d} Bkground:{1:<5b} Method:{2:<5s}".format(order,
                  not nobackground, method))
        # Determine which scales to use
        scale = ['wave','pixel'] if scale is None else [scale]
        
        # Extract data from the fits file
        spec1d  = self.extract1d(order,nobackground=nobackground,vacuum=True)
        
        pn,weights  = self.calculate_photon_noise(order,return_array=True)
        weights = self.get_weights1d(order)
        # Define limits in wavelength and theoretical wavelengths of lines
        maxima  = peakdet(spec1d.flux,spec1d.wave,extreme='max')
        minima  = peakdet(spec1d.flux,spec1d.pixel,extreme='min')
        xpeak   = maxima.x
        nu_min  = 299792458e0/(xpeak.iloc[-1]*1e-10)
        nu_max  = 299792458e0/(xpeak.iloc[0]*1e-10)
        #print(nu_min,nu_max)
        npeaks  = int(round((nu_max-nu_min)/self.reprate))+1
        n_start = int(round((nu_min - self.f0_comb)/self.reprate))
        lbd     = np.array([299792458e0/(self.f0_comb 
               + (n_start+i)*self.reprate)*1e10 for i in range(xpeak.size)][::-1])
        if verbose>1:
            print("Npeaks:{0:<5}".format(npeaks))
        
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
        #print(lines_th)
        # Perform the fitting        
        #lines    = {}
        #for scale in scale:
        if plot:
            plt.figure()
        xarray     = spec1d['pixel']
        yarray     = spec1d['flux']
        yerror     = spec1d['error']
        xmax       = maxima_th['pixel']
        xmin       = minima.x
        nminima    = minima.index.size
        nmaxima    = maxima.index.size
        print(nminima,nmaxima)
        dxi   = 11.
        dx         = xarray.diff(1).fillna(dxi)
        if verbose>2:
            print('Fitting {}'.format(scale))
        
        # model
        model = model if model is not None else 'singlegaussian'
        results = Parallel(n_jobs=1)(delayed(fit_peak)(i,xarray,yarray,yerror,weights,xmin,xmax,dx,method,model) for i in range(nminima))
        results = np.array(results)
      
        parameters = results['pars'].squeeze(axis=1)
        errors     = results['errors'].squeeze(axis=1)
        photon_nse = results['pn'].squeeze(axis=1)
        center     = results['cen'].squeeze(axis=1)
        center_err = results['cen_err'].squeeze(axis=1)
        rsquared   = results['r2'].squeeze(axis=1)
        N = results.shape[0]
        M = parameters.shape[1]
        
        
        #print(np.shape(parameters),np.shape(errors),np.shape(photon_nse))
        line_results = np.concatenate((parameters,errors,photon_nse,rsquared,center,center_err),axis=1)
        if model == 'singlegaussian':
            columns = ['amplitude','cen','sigma',
                       'amplitude_error','cen_error','sigma_error',
                       'photon_noise','r2','center','center_err']
        elif ((model == 'doublegaussian') or (model=='simplegaussian')):
            columns = ['amplitude1','center1','sigma1',
                      'amplitude2','center2','sigma2',
                      'amplitude1_error','center1_error','sigma1_error',
                      'amplitude2_error','center2_error','sigma2_error',
                      'photon_noise','r2','center','center_err']
        lines_fit = pd.DataFrame(line_results,
                                 index=np.arange(0,nminima,1),#lines_th.index,
                                 columns=columns)
        # make sure sigma values are positive!
        if model == 'singlegaussian':
            lines_fit.sigma = lines_fit.sigma.abs()
        elif ((model == 'doublegaussian') or (model=='simplegaussian')):
            lines_fit.sigma1 = lines_fit.sigma1.abs()
            lines_fit.sigma2 = lines_fit.sigma2.abs()
        lines_fit['th_wave'] = lines_th['wave']
        lines_fit['th_pixel']  = lines_th['pixel']
        lines_fit.dropna(axis=0,how='any',inplace=True)            
#        lines[scale]        = lines_fit   
        if remove_poor_fits == True:
            if verbose>2:
                print('Removing poor fits')
            lines_fit = _remove_poor_fits(lines_fit)
        else:
            pass

        return lines_fit
    def fit_lines1d(self,order,nobackground=True,method='epsf',model=None,
                  scale='pixel',vacuum=True,remove_poor_fits=False,verbose=0):
       
        self.check_and_load_psf()
        sc        = self.segment_centers
        segsize   = 4096//self.nsegments
        pixels    = self.psf.coords['pix']
        pixelbins = (pixels[1:]+pixels[:-1])/2
        
        def get_line_weights(line_x,center):
            
            weights = xr.DataArray(np.full_like(pixels,np.nan),coords=[pixels],dims=['pix'])
            
            pixels0 = line_x - center
            pix = pixels[np.digitize(pixels0,pixelbins,right=True)]
            # central 2.5 pixels on each side have weights = 1
            central_pix = pix[np.where(abs(pix)<=2.5)[0]]
            weights.loc[dict(pix=central_pix)]=1.0
            # pixels outside of 5.5 have weights = 0
            outer_pix   = pix[np.where(abs(pix)>=5.5)[0]]
            weights.loc[dict(pix=outer_pix)]=0.0
            # pixels with 2.5<abs(pix)<5.5 have weights between 0 and 1, linear
            midleft_pix  = pix[np.where((pix>-5.5)&(pix<-2.5))[0]]
            midleft_w   = np.array([(x+5.5)/3 for x in midleft_pix])
            
            midright_pix = pix[np.where((pix>2.5)&(pix<5.5))[0]]
            midright_w   = np.array([(-x+5.5)/3 for x in midright_pix])
            
            weights.loc[dict(pix=midleft_pix)] =midleft_w
            weights.loc[dict(pix=midright_pix)]=midright_w
            return weights.dropna('pix').values
        def residuals(x0,pixels,counts,weights,background,splr):
            ''' Modela parameters are estimated shift of the line center from 
                the brightest pixel and the line flux. 
                Input:
                ------
                   x0        : shift, flux
                   pixels    : pixels of the line
                   counts    : detected e- for each pixel
                   weights   : weights of each pixel (see 'get_line_weights')
                   background: estimated background contamination in e- 
                   splr      : spline representation of the ePSF
                Output:
                -------
                   residals  : residuals of the model
            '''
            sft, flux = x0
            model = flux * interpolate.splev(pixels+sft,splr) 
            resid = np.sqrt(line_w) * ((counts-background) - model)/np.sum(counts)
            #resid = line_w * (counts- model)
            return resid
        
        # Determine which scales to use
        scale = ['wave','pixel'] if scale is None else [scale]
        if verbose>0:
            print("ORDER:{0:<5d} Bkground:{1:<5b} Method:{2:<5s}".format(order,
                  not nobackground, method))
 
        # Cut the lines
        pixel, flux, error, bkgr, bary = self.cut_lines(order, nobackground=nobackground,
                  vacuum=vacuum,columns=['pixel', 'flux', 'error', 'bkg', 'bary'])
        pixel = pixel[order]
        flux  = flux[order]
        error = error[order]
        bkgr  = bkgr[order]
        bary  = bary[order]
        nlines = len(pixel)
        
        params = ['cen','cen_err','flux','flux_err','shift','phase','b','chisq']
        lines = xr.DataArray(data=np.zeros((nlines,len(params))),
                             coords = [np.arange(nlines),params],
                             dims = ['id','par'])
        
        for n in range(nlines):
            line_x = pixel[n]
            line_y = flux[n]
            line_b = bkgr[n]
               
            cen_pix = line_x[np.argmax(line_y)]
            line_w = get_line_weights(line_x,cen_pix)
            
            local_seg = cen_pix//segsize
            psf_x, psf_y = self.get_local_psf(cen_pix,order=order,seg=local_seg)
            
            psf_rep  = interpolate.splrep(psf_x,psf_y)
            p0 = (0,np.max(line_y))
            
            popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                                    args=(line_x,line_y,line_w,line_b,psf_rep),
                                    full_output=True)
            print(n,np.sum(infodict['fvec']**2)/(len(line_x)-len(popt)))
            if ier not in [1, 2, 3, 4]:
                print("Optimal parameters not found: " + errmsg)
                popt = np.full_like(p0,np.nan)
                pcov = None
                success = False
            else:
                success = True
            if success:
                
                sft, flx = popt
                cost = np.sum(infodict['fvec']**2)
                dof  = (len(line_x) - len(popt))
                if pcov is not None:
                    pcov = pcov*cost/dof
                else:
                    pcov = np.array([[np.inf,0],[0,np.inf]])
            cen              = line_x[np.argmax(line_y)]-sft
            cen_err, flx_err = [np.sqrt(pcov[i][i]) for i in range(2)]
            phi              = cen - int(cen+0.5)
            b                = bary[n]
            
            pars = np.array([cen,cen_err,flx,flx_err, sft,phi,b,cost/dof])
            lines.loc[dict(id=n)] = pars
        return lines
    def get_average_profile(self,order,nobackground=True):
        # Extract data from the fits file
        spec1d  = self.extract1d(order,nobackground=nobackground,vacuum=True)
        
        pn,weights  = self.calculate_photon_noise(order,return_array=True)
        weights     = self.get_weights1d(order)
        # Define limits in wavelength and theoretical wavelengths of lines
        maxima      = peakdet(spec1d.flux,spec1d.wave,extreme='max')
        minima      = peakdet(spec1d.flux,spec1d.pixel,extreme='min')
        xpeak       = maxima.x
        nu_min      = 299792458e0/(xpeak.iloc[-1]*1e-10)
        nu_max      = 299792458e0/(xpeak.iloc[0]*1e-10)
        #print(nu_min,nu_max)
        npeaks      = int(round((nu_max-nu_min)/self.reprate))+1
        
        
        xarray = spec1d.pixel
        yarray = spec1d.flux
        xmin   = minima.x
        
        xdata = []
        ydata = []
        
#        data  = xr.DataArray(np.zeros())
        for n in range(npeaks-2):
            # cut the lines and combine into a single profile
            cut    = np.where((xarray>=xmin[n])&(xarray<=xmin[n+1]))[0]
            line_x = xarray[cut].values
            line_y = yarray[cut].values
            
            xn = np.linspace(-5,5,cut.size)
            yn = line_y
            
            xdata.append(xn)
            ydata.append(yn)
#        return pd.Panel(np.transpose([xdata,ydata]),columns=['x','y'])
        return xdata,ydata
        
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
    def get_barycenters(self,order,nobackground=True,vacuum=True):
        xdata, ydata = self.cut_lines(order,nobackground=nobackground,vacuum=vacuum)    
        barycenters  = {}
        orders = self.prepare_orders(order)
        
        for order in orders:
            barycenters_order = []
            for i in range(np.size(xdata[order])):
                xline = xdata[order][i]
                yline = ydata[order][i]
                
                b     = np.sum(xline * yline) / np.sum(yline)
                barycenters_order.append(b)
                
            barycenters[order] = barycenters_order
        return barycenters
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
    def get_extremes(self, order, scale="pixel", extreme="max"):
        '''Function to determine the envelope of the observations by fitting a cubic spline to the maxima of LFC lines'''
        spec1d      = self.extract1d(order=order,columns=[scale,'flux'])
        extremes    = peakdet(spec1d["flux"], spec1d[scale], extreme=extreme)
        return extremes
    def get_distortions(self,order=None,calibrator='LFC'):
        ''' 
        Returns the difference between the theoretical ('real') wavelength of 
        LFC lines and the wavelength interpolated from the wavelength solution.
        Returned array is in units metres per second (m/s).
        '''
        orders = self.prepare_orders(order)
        nOrder = len(orders)
        dist   = xr.DataArray(np.full((nOrder,3,500),np.NaN),
                              dims=['od','typ','val'],
                              coords=[orders,
                                      ['wave','pix','rv'],
                                      np.arange(500)])
        for i,order in enumerate(orders):
            data  = self.check_and_get_comb_lines('LFC',orders)
            wav0  = data.sel(typ='wave',od=order)#.dropna('val')
            pix0  = data.sel(typ='pix',od=order)#.dropna('val')
            if calibrator == 'ThAr':
                coeff = self.wavecoeff_vacuum[order]
            elif calibrator == 'LFC':
                coeff = self.wavecoef_LFC[order][::-1]
            wav1 = polynomial(pix0,*coeff)
            rv   = (wav1-wav0)/wav0 * 299792458.
            dist.loc[dict(typ='pix',od=order)]=pix0
            dist.loc[dict(typ='wave',od=order)]=wav0
            dist.loc[dict(typ='rv',od=order)]=rv
        return dist
    def get_local_psf(self,pix,order,seg):
        self.check_and_load_psf()
        sc       = self.segment_centers
        def return_closest_segments(pix):
            sg_right  = int(np.digitize(pix,sc))
            sg_left   = sg_right-1
            return sg_left,sg_right
        
        sgl, sgr = return_closest_segments(pix)
        f1 = (sc[sgr]-pix)/(sc[sgr]-sc[sgl])
        f2 = (pix-sc[sgl])/(sc[sgr]-sc[sgl])
        
        epsf_x  = self.psf.sel(ax='x',od=order,seg=seg).dropna('pix')+pix
        epsf_1 = self.psf.sel(ax='y',od=order,seg=sgl).dropna('pix')
        epsf_2 = self.psf.sel(ax='y',od=order,seg=sgr).dropna('pix')
        epsf_y = f1*epsf_1 + f2*epsf_2 
        
        xc     = epsf_y.coords['pix']
        epsf_x  = self.psf.sel(ax='x',od=order,seg=seg,pix=xc)+pix
        
        return epsf_x, epsf_y
    def get_psf(self,order,seg):
        self.check_and_load_psf()
        order = self.prepare_orders(order)
        seg   = self.to_list(seg)
        #epsf_x = self.psf.sel(seg=seg,od=order,ax='x').dropna('pix','all')
        #epsf_y = self.psf.sel(seg=seg,od=order,ax='y').dropna('pix','all')
        psf = self.psf.sel(seg=seg,od=order,ax='y')
        return psf
    def get_residuals(self,order=None):
        orders  = self.prepare_orders(order)
                
        lines = self.check_and_get_comb_lines(calibrator='LFC',orders=orders)
        
        resids  = self.residuals
        
        #pos_pix = lines.sel(typ='pix',od=orders)
        selected_res = resids.sel(od=orders)
        return selected_res
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
#        npeaks        = lines_withbkg.size
        delta_rv      = (lines_withbkg["MU"]-lines_nobkg["MU"])*f
        median_rv     = np.nanmedian(delta_rv)
        print("ORDER {0}, median RV displacement = {1}".format(order,median_rv))
        return delta_rv
    def get_wavecoeff(self,medium='vacuum',orders=None):
        self.__check_and_load__()
        if orders is None:
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
        
        Formula 8
        '''
        spec1d        = self.extract1d(order=order,nobackground=False)
        wavesol       = self.__get_wavesol__(calibrator)*1e-10 # meters
#        diff          = np.diff(wavesol[order])
        #dlambda       = np.insert(diff,0,diff[0])
#        dlambda       = np.gradient(wavesol[order])
#        dflux         = np.gradient(spec1d['flux'])#,dlambda)
        df_dl         = derivative1d(spec1d['flux'].values,wavesol[order])
        #print(dflux)
        weights1d     = wavesol[order]**2 * (df_dl)**2 / (spec1d['flux'])
        return weights1d
    def get_weights2d(self,calibrator="ThAr"):
        ''' 
        Return weights of individual pixels for the entire 2d spectrum
        (Bouchy 2001) 
        '''
        spec2d_data    = self.data ##### FIND A WAY TO REMOVE ROWS WITH ZEROS
        spec2d         = self.data
        wavesol2d      = self.__get_wavesol__(calibrator)
        
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
    def load_psf(self,filepath=None):
        if filepath is not None:
            filepath = filepath
        else:
            filepath = os.path.join(harps_dtprod,'epsf','harps_A.nc')
        
        data = xr.open_dataset(filepath)
        epsf = data['epsf'].sel(ax=['x','y'])
        self.psf = epsf
        return
    def plot_spectrum(self,order=None,nobackground=False,scale='wave',fit=False,
             confidence_intervals=False,legend=False,
             naxes=1,ratios=None,title=None,sep=0.05,alignment="vertical",
             figsize=(16,9),sharex=None,sharey=None,plotter=None,**kwargs):
        '''
        Plots the spectrum. 
        
        Args:
        ----
            order:          integer of list or orders to be plotted
            nobackground:   boolean, subtracts the background
            scale:          'wave' or 'pixel'
            fit:            boolean, fits the lines and shows the fits
            
        Returns:
        --------
            plotter:    Plotter Class object
        '''
        
        if plotter is None:
            plotter = SpectrumPlotter(bottom=0.12,**kwargs)
        else:
            pass
        figure, axes = plotter.figure, plotter.axes
    
        orders = self.prepare_orders(order)
        for order in orders:
            spec1d = self.extract1d(order,nobackground=nobackground)
            x      = spec1d[scale]
            y      = spec1d.flux
            axes[0].plot(x,y,label='Data')
            if fit==True:
                fit_lines = self.fit_lines(order,scale=scale,nobackground=nobackground)
                self.axes[0].plot(x,double_gaussN_erf(x,fit_lines[scale]),label='Fit')
        if legend:
            axes[0].legend()
        figure.show()
        return plotter
    def plot_distortions(self,order=None,kind='lines',plotter=None,
                         show=True,**kwargs):
        '''
        Plots the distortions in the CCD through two channels:
        kind = 'lines' plots the difference between LFC theoretical wavelengths
        and the value inferred from the ThAr wavelength solution. 
        kind = 'wavesol' plots the difference between the LFC and the ThAr
        wavelength solutions.
        
        Args:
        ----
            order:      integer of list or orders to be plotted
            kind:       'lines' or 'wavesol'
            plotter:    Plotter Class object (allows plotting multiple spectra
                            in a single panel)
            show:       boolean
        Returns:
        --------
            plotter:    Plotter Class object
        '''
        if plotter is None:
            plotter = SpectrumPlotter(bottom=0.12,**kwargs)
        else:
            pass
        figure, axes = plotter.figure, plotter.axes
        axes[0].set_ylabel('$\Delta$=(ThAr - LFC)')
        axes[0].set_xlabel('Pixel')
        orders = self.prepare_orders(order)
        
        
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker = kwargs.get('marker','x')
        
        plotargs = {'ms':2,'marker':marker}
        for i,order in enumerate(orders):
            if kind == 'lines':
                data  = self.check_and_get_comb_lines('LFC',orders)
                wav   = data.sel(typ='wave',od=order).dropna('val')
                pix   = data.sel(typ='pix',od=order).dropna('val')
                coeff = self.wavecoeff_vacuum[order][::-1]
                thar  = np.polyval(coeff,pix)
                plotargs['ls']=''
            elif kind == 'wavesol':
                wav   = self.wavesol_LFC[order]
                pix   = np.arange(self.npix)
                thar  = self.wavesol_thar[order]
                plotargs['ls']='-'
                plotargs['ms']=0
            rv  = (thar-wav)/wav * 299792458.
            if len(orders)>5:
                plotargs['color']=colors[i]
            axes[0].plot(pix,rv,**plotargs)
        [axes[0].axvline(512*(i+1),lw=0.3,ls='--') for i in range (8)]
        if show == True: figure.show() 
        return plotter
    def plot_residuals(self,order=None,calibrator='LFC',mean=True,
                       plotter=None,show=True,**kwargs):
        '''
        Plots the residuals of LFC lines to the wavelength solution. 
        
        Args:
        ----
            order:      integer of list or orders to be plotted
            calibrator: 'LFC' or 'ThAr'
            mean:       boolean, plots the running mean of width 5. Window size
                         can be changed using the keyword 'window'
            plotter:    Plotter Class object (allows plotting multiple spectra
                            in a single panel)
            show:       boolean
        Returns:
        --------
            plotter:    Plotter Class object
        '''
        if plotter is None:
            plotter = SpectrumPlotter(bottom=0.12,**kwargs)
        else:
            pass
        figure, axes = plotter.figure, plotter.axes
        
        orders = self.prepare_orders(order)
                
        lines = self.check_and_get_comb_lines(calibrator,orders)
        
        resids  = self.residuals
        
        pos_pix = lines.sel(typ='pix',od=orders)
        pos_res = resids.sel(od=orders)
        
        
        
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker = kwargs.get('marker','x')
        
        plotargs = {'s':2,'marker':marker}
        for i,order in enumerate(orders):
            pix = pos_pix.sel(od=order)
            res = pos_res.sel(od=order)
            if len(orders)>5:
                plotargs['color']=colors[i]
            axes[0].scatter(pix,res,**plotargs)
            if mean==True:
                meanplotargs={'lw':0.8}
                w  = kwargs.get('window',5)                
                rm = running_mean(res,w)
                if len(orders)>5:
                    meanplotargs['color']=colors[i]
                axes[0].plot(pix,rm,**meanplotargs)
        [axes[0].axvline(512*(i+1),lw=0.3,ls='--') for i in range (8)]
        if show == True: figure.show() 
        return plotter
    def plot_histogram(self,kind,order=None,show=True,plotter=None,**kwargs):
        '''
        Plots a histogram of residuals of LFC lines to the wavelength solution 
        (kind = 'residuals') or a histogram of R2 goodness-of-fit estimators 
        (kind = 'R2').
        
        Args:
        ----
            kind:       'residuals' or 'R2'
            order:      integer or list of orders to be plotted
            plotter:    Plotter Class object (allows plotting multiple spectra
                            in a single panel)
            show:       boolean
        Returns:
        --------
            plotter:    Plotter Class object
        '''
        if kind not in ['residuals','R2']:
            raise ValueError('No histogram type specified \n \
                              Valid options: \n \
                              \t residuals \n \
                              \t R2')
        else:
            pass
        
        orders = self.prepare_orders(order)
            
        N = len(orders)
        if plotter is None:
            plotter = SpectrumPlotter(naxes=N,alignment='grid',**kwargs)
        else:
            pass
        figure, axes = plotter.figure, plotter.axes
        lines = self.check_and_get_comb_lines(orders=orders)
        if kind == 'residuals':
            data     = self.residuals
            normed   = True
        elif kind == 'R2':
            data     = lines.sel(typ='R2')
            normed   = False
        bins    = kwargs.get('bins',10)
        for i,order in enumerate(orders):
            selection = data.sel(od=order).dropna('val').values
            axes[i].hist(selection,bins=bins,normed=normed)
            if kind == 'residuals':
                mean = np.mean(selection)
                std  = np.std(selection)
                A    = 1./np.sqrt(2*np.pi*std**2)
                x    = np.linspace(np.min(selection),np.max(selection),100)
                y    = A*np.exp(-0.5*((x-mean)/std)**2)
                axes[i].plot(x,y,color='#ff7f0e')
                axes[i].plot([mean,mean],[0,A],color='#ff7f0e',ls='--')
                axes[i].text(0.8, 0.95,r"$\mu={0:8.3e}$".format(mean), 
                            horizontalalignment='center',
                            verticalalignment='center',transform=axes[i].transAxes)
                axes[i].text(0.8, 0.9,r"$\sigma={0:8.3f}$".format(std), 
                            horizontalalignment='center',
                            verticalalignment='center',transform=axes[i].transAxes)
        if show == True: figure.show() 
        return plotter
    def plot_wavesolution(self,calibrator='LFC',order=None,nobackground=True,
                       plotter=None,naxes=1,ratios=None,title=None,sep=0.05,
                       figsize=(16,9),alignment="vertical",
                       sharex=None,sharey=None,show=True,**kwargs):
        '''
        Plots the wavelength solution of the spectrum for the provided orders.
        '''
        
        if plotter is None:
            plotter = SpectrumPlotter(naxes=naxes,ratios=ratios,title=title,
                                  sep=sep,figsize=figsize,alignment=alignment,
                                  sharex=sharex,sharey=sharey,**kwargs)
        else:
            pass
        figure, axes = plotter.figure, plotter.axes
        
        orders = self.prepare_orders(order)
        
        # Check and retrieve the wavelength calibration
        wavesol_name = 'wavesol_{cal}'.format(cal=calibrator)
        exists_calib = hasattr(self,wavesol_name)
        if exists_calib == False:
            wavesol = self.__get_wavesol__(calibrator,orders=orders)
        else:
            wavesol = getattr(self,wavesol_name)
            
        # Check and retrieve the positions of lines 
        exists_lines = hasattr(self,'lines')
        if exists_lines == False:
            wavesol = self.__get_wavesol__(calibrator)
            lines = self.lines
        else:
            lines = self.lines
            
        # Select line data        
        pos_pix = lines.sel(typ='pix')
        pos_wav = lines.sel(typ='wave')
        
        # Manage colors
        #cmap   = plt.get_cmap('viridis')
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker = kwargs.get('marker','x')
        ms     = kwargs.get('markersize',5)
        
        # Plot the line through the points?
        plotline = kwargs.get('plot_line',True)
        for i,order in enumerate(orders):
            pix = pos_pix.sel(od=order)
            wav = pos_wav.sel(od=order)
            axes[0].scatter(pix,wav,s=ms,color=colors[i],marker=marker)
            if plotline == True:
                axes[0].plot(wavesol[order],color=colors[i])
        
        if show == True: figure.show() 
        return plotter
    def prepare_orders(self,order):
        '''
        Returns an array or a list containing the input orders.
        '''
        if order is None:
            orders = np.arange(self.sOrder,self.nbo,1)
        else:
            orders = self.to_list(order)
        return orders
    def to_list(self,item):
        if type(item)==int:
            items = [item]
        elif type(item)==np.int64:
            items = [item]
        elif type(item)==list:
            items = item
        elif type(item)==np.ndarray:
            items = list(item)
        elif type(item) == None:
            items = None
        else:
            print('Unsupported type. Type provided:',type(item))
        return items
class EmissionLine(object):
    def __init__(self,xdata,ydata,yerr=None,weights=None,
                 absolute_sigma=True,bounds=None):
        ''' Initialize the object using measured data
        
        Args:
        ----
            xdata: 1d array of x-axis data (pixels or wavelengts)
            ydata: 1d array of y-axis data (electron counts)
            weights:  1d array of weights calculated using Bouchy method
            kind: 'emission' or 'absorption'
        '''
        def _unwrap_array_(array):
            if type(array)==pd.Series:
                narray = array.values
            elif type(array)==np.ndarray:
                narray = array
            return narray
            
            
        self.xdata       = _unwrap_array_(xdata)
        self.xbounds     = (self.xdata[:-1]+self.xdata[1:])/2
        self.ydata       = _unwrap_array_(ydata)
        yerr             = yerr if yerr is not None else np.sqrt(np.abs(self.ydata))
        weights          = weights if weights is not None else yerr #np.ones_like(xdata)
        self.yerr        = _unwrap_array_(yerr)
        self.weights     = _unwrap_array_(weights)
        self.success     = False
        self.sigmabound = 2*np.std(self.xdata)#/3   
        self.bounds      = bounds
        
        self.barycenter = np.sum(self.xdata*self.ydata)/np.sum(self.ydata)
#        self.model_class = model_class(xdata,ydata,yerr,weights)
#        self.model       = self.model_class.model
#        self.jacobian    = self.model_class.jacobian
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
            errors = self.fit_errors
        else:
            p0 = self._initialize_parameters()
            pars, errors = self.fit(p0)
            pfit = self.fit_parameters
            errors = self.fit_errors
        return pfit, errors
    def _get_gauss_parameters(self):
        ''' Method to check whether the fit has been successfully performed.
        If the fit was successful, returns the fit values. Otherwise, the 
        method performs the fitting procedure and returns the fit values.
        Returns:
        -------
            pfit: tuple with fitted (amplitude, mean, sigma) values
        '''
        if self.success == True:
            pars = self.gauss_parameters
            errors = self.gauss_errors
        else:
            p0 = self._initialize_parameters()
            pars, errors = self.fit(p0)
        return pars, errors  
    
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
        return weights * (self.model(pars) - cdata)
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
    def calc_R2(self,pars=None,weights=None):
        ''' Returns the R^2 estimator of goodness of fit to the model.
        Args:
        ----
            pars: tuple (A1, mu1, sigma1, fA, fm, sigma2) of the model
        Returns:
        -------
            chisq
        '''
        if pars is None:
            pars = self._get_gauss_parameters()[0]
        cdata = self.ydata[1:-1]
        weights = weights if weights is not None else self.weights[1:-1]
        SSR = 1 - np.sum(self.residuals(pars,weights)**2/np.std(cdata))
        SST = np.sum(weights*(cdata - np.mean(cdata))**2)
        rsq = 1 - SSR/SST
        return rsq
    
    def evaluate(self,pars=None,x=None,separate=False,clipx=True):
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
        pars = pars if pars is not None else self._get_gauss_parameters()
        
#        g1 = A1 * np.exp(-1/2*((x-mu1)/sigma1)**2)[1:-1]
#        g2 = A2 * np.exp(-1/2*((x-mu2)/sigma2)**2)[1:-1]
        p  = np.reshape(pars,(-1,3))
        N  = p.shape[0]
        Y  = []
        #print(p)
        for i in range(N):
            A, mu, sigma = p[i]
            #e11  = erf((xb[:-1]-mu)/(np.sqrt(2)*sigma))
            #e21  = erf((xb[1:] -mu)/(np.sqrt(2)*sigma))
            #y    = A*sigma*np.sqrt(np.pi/2)*(e21-e11)
            y = A * np.exp(-1/2*((x-mu)/sigma)**2)#[1:-1]
            if clipx:
                y=y[1:-1]
            Y.append(y)
        
        
        if separate:
            return tuple(Y)
        else:
            return np.sum(Y,axis=0)
    def fit(self,p0=None,absolute_sigma=True, bounded=True,
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
        
        if bounded == True:
            if self.bounds is None:
                bounds = self._initialize_bounds()
            else:
                bounds = self.bounds
        else:
            bounds=(-np.inf, np.inf)
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
        if method == 'lm':    
            return_full = kwargs.pop('full_output', False)
#            wrapped_jac = self._wrap_jac()
            res = leastsq(self.residuals,p0,Dfun=None,full_output=1)#,col_deriv=True,**kwargs)
            pfit, pcov, infodict, errmsg, ier = res
            cost = np.sum(infodict['fvec']**2)
            if ier not in [1, 2, 3, 4]:
                #raise RuntimeError("Optimal parameters not found: " + errmsg)
                pfit = np.full_like(p0,np.nan)
                pcov = None
                success = False
            else:
                success = True
        else:
            #print('Bounded problem')
            res = least_squares(self.residuals, p0, jac=self.jacobian, bounds=bounds, method=method,
                                **kwargs)
            if not res.success:
                #raise RuntimeError("Optimal parameters not found: " + res.message)
                pfit = np.full_like(p0,np.nan)
                pcov = None
                success = False
                cost = np.inf
            else:
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
        error_fit_pars = [] 
        for i in range(len(pfit)):
            try:
              error_fit_pars.append(np.absolute(pcov[i][i])**0.5)
            except:
              error_fit_pars.append( 0.00 )
        
        # From fit parameters, calculate the parameters of the two gaussians 
        # and corresponding errors
        if success == True:
            if (self.__class__ == SingleGaussian):
                gauss_parameters = pfit
                gauss_errors     = error_fit_pars
                
                fit_parameters   = pfit
                fit_errors       = error_fit_pars
                
            elif (self.__class__ == DoubleGaussian or
                  self.__class__ == SimpleGaussian):
                A1,m1,s1,A2,m2,s2 = pfit
                error_A1,error_m1,error_s1,error_A2,error_m2,error_s2 = error_fit_pars
                
                # Make the component with the smaller mean to be m1 and the  
                # component with the larger mean to be m2. (i.e. m1<m2)
                
                if m1<m2:
                    gp_c1 = A1, m1, s1
                    gp_c2 = A2, m2, s2
                    gp_c1error = error_A1, error_m1, error_s1
                    gp_c2error = error_A2, error_m2, error_s2
                    
                    
                elif m1>m2:
                    gp_c1 = A2, m2, s2
                    gp_c2 = A1, m1, s1
                    gp_c1error = error_A2, error_m2, error_s2  
                    gp_c2error = error_A1, error_m1, error_s1            
                else:
                    print("m1=m2 ?", m1==m2)
                gauss_parameters = np.array([*gp_c1,*gp_c2])
                gauss_errors     = np.array([*gp_c1error,*gp_c2error])
                fit_parameters   = pfit
                fit_errors       = error_fit_pars
            
        else:
            gauss_parameters = np.full_like(pfit,np.nan)
            gauss_errors     = np.full_like(pfit,np.nan)
            fit_parameters   = pfit
            fit_errors       = error_fit_pars
            
        self.covar     = pcov
        self.rchi2     = cost / dof
        self.dof       = dof
        
        
        self.gauss_parameters = gauss_parameters
        self.gauss_errors     = gauss_errors
        
        
        self.fit_parameters   = fit_parameters
        self.fit_errors       = fit_errors
        
        self.center           = self.calculate_center(gauss_parameters)
        #self.center_error     = self.calculate_center_uncertainty(pfit,pcov)
        self.center_error     = 0.
        self.center_mass      = np.sum(self.weights*self.xdata*self.ydata)/np.sum(self.weights*self.ydata)
        
        
        
#        self.infodict = infodict
#        self.errmsg = errmsg
        self.success = success
        self.cost = cost
        if return_full:
            return gauss_parameters, gauss_errors, infodict, errmsg, ier
        else:
            return gauss_parameters, gauss_errors
    
    def plot(self,fit=True,cofidence_intervals=True,ax=None,**kwargs):
        ''' Plots the line as a histogram of electron counts with corresponding
        errors. If `fit' flag is True, plots the result of the fitting 
        procedure.
        
        
        '''
        import matplotlib.transforms as mtransforms
        if ax is None:
            fig,ax = get_fig_axes(1,figsize=(9,9),bottom=0.12,left=0.15,**kwargs)
            self.fig = fig
        elif type(ax) == plt.Axes:
            ax = [ax]
        elif type(ax) == list:
            pass
        self.ax_list  = [ax]
        widths = np.diff(self.xdata)[:-1]
        ax[0].bar(self.xdata[1:-1],self.ydata[1:-1],
                  widths,align='center',alpha=0.3,color='C0')
        ax[0].errorbar(self.xdata[1:-1],self.ydata[1:-1],
                       yerr=self.yerr[1:-1],fmt='o',color='C0')
        yeval = np.zeros_like(self.ydata)
        if fit is True:
            p,pe = self._get_gauss_parameters()
#            xeval = np.linspace(np.min(self.xdata),np.max(self.xdata),100)
            xeval = self.xdata
            if self.__class__ == SingleGaussian:
                yeval = self.evaluate(p,xeval,False)
            elif (self.__class__ == DoubleGaussian or
                  self.__class__ == SimpleGaussian):
                y1,y2 = self.evaluate(p,xeval,True)
                yeval = y1+y2
            xeval = xeval[1:-1]
            if (self.__class__ == DoubleGaussian or
                self.__class__ == SimpleGaussian):
                ax[0].plot(xeval,y1,color='C2',lw=0.7,ls='--')
                ax[0].plot(xeval,y2,color='C2',lw=0.7,ls='--')  
                A1, m1, s1, A2, m2, s2 = p
                if ((m1>np.min(self.xdata))&(m1<np.max(self.xdata))):
                    ax[0].plot([m1,m1], [0,A1],ls='--',lw=0.7,color='C2')
                if ((m2>np.min(self.xdata))&(m2<np.max(self.xdata))):
                    ax[0].plot([m2,m2], [0,A2],ls='--',lw=0.7,color='C2')
            fit = True
            color = kwargs.pop('color','C1')
            label = kwargs.pop('label',None)
            ax[0].plot(xeval,yeval,color=color,marker='o',label=label)
            
              
            # calculate the center of the line and the 1-sigma uncertainty
#            cenx = self.center
#            ceny = self.evaluate(p,np.array([m1,cenx,m2]),ptype='gauss')
#            ax[0].plot([cenx,cenx],[0,ceny[1]],ls='--',lw=1,c='C1')
#            cend = self.center_error
            
            # shade the area around the center of line (1-sigma uncertainty)
#            xcenval = np.linspace(cenx-cend,cenx+cend,100)
#            ycenval = self.evaluate(p,xcenval,ptype='gauss')
#            ax[0].fill_between(xcenval,0,ycenval,color='C1',alpha=0.4,
#              where=((xcenval>=cenx-cend)&(xcenval<=cenx+cend)))
        if cofidence_intervals is True and fit is True:
            xeval = self.xdata
            y,ylow,yhigh = self.confidence_band(xeval,confprob=0.05)
            ax[0].fill_between(xeval[1:-1],ylow,yhigh,alpha=0.5,color='#ff7f0e')
            y,ylow,yhigh = self.confidence_band(xeval,confprob=0.32)
            ax[0].fill_between(xeval[1:-1],ylow,yhigh,alpha=0.2,
                              color='#ff7f0e')
        ymax = np.max([1.2*np.percentile(yeval,95),1.2*np.max(self.ydata)])
        ax[0].set_ylim(-np.percentile(yeval,20),ymax)
        ax[0].set_xlabel('Pixel')
        ax[0].set_ylabel('Counts')
        ax[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        return
    def confidence_band(self, x, confprob=0.05, absolute_sigma=False):
        
        # https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html \
                            #confidence-and-prediction-intervals
        from scipy.stats import t
        # Given the confidence probability confprob = 100(1-alpha)
        # we derive for alpha: alpha = 1 - confprob
        alpha = 1.0 - confprob
        prb = 1.0 - alpha/2
        tval = t.ppf(prb, self.dof) #degrees of freedom
                    
        C = self.covar
       
        p,pe = self._get_fit_parameters()
        n = len(p)              # Number of parameters from covariance matrix
        N = len(x)
        if absolute_sigma:
            covscale = 1.0
        else:
            covscale = self.rchi2 * self.dof
          
        y = self.evaluate(p,x)
        
        # If the x array is larger than xdata, provide new weights
        # for all points in x by linear interpolation
        int_function = interpolate.interp1d(self.xdata,self.weights)
        weights      = int_function(x)
        dfdp = self.jacobian(p,x,weights).T
        
        df2 = np.zeros(N-2)
        for j in range(n):
            for k in range(n):
                df2 += dfdp[j]*dfdp[k]*C[j,k]
#        df2 = np.dot(np.dot(dfdp.T,self.covar),dfdp).sum(axis=1)
        df = np.sqrt(covscale*df2)
        delta = tval * df
        upperband = y + delta
        lowerband = y - delta
        return y, upperband, lowerband       
    def calculate_center_uncertainty(self,pfit=None,covar=None,N=200):
        ''' 
        Returns the standard deviation of centres drawn from a random sample.
        
        Draws N samples by randomly sampling the provided fit parameteres and
        corresponding errors to construct N models of the line. Line centre 
        is calculated for each of the N models.         
        '''
        pfit = pfit if pfit is not None else self._get_fit_parameters()[0]    
        C = covar if covar is not None else self.covar
        # It is not possible to calculate center uncertainty if the covariance 
        # matrix contains infinite values
        if np.isinf(C).any() == True:
            return -1
        else:
            pass
        mdgN  = np.random.multivariate_normal(mean=pfit,cov=C,size=N)
#        cut   = np.where(mdgN[:,3]>0)[0]
#        mdgN  = mdgN[cut]
        centers = np.zeros(mdgN.size)
        for i,pars in enumerate(mdgN):
            pgauss_i    = pars
            centers[i]  = self.calculate_center(pgauss_i)
        return centers.std()    
class SingleGaussian(EmissionLine):
    def model(self,pars,separate=False):
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
        and 't' is defined as:
            
            t = (x - mu)/(sqrt(2) * sigma)
        
        Here, mu is the mean of the Gaussian.
        '''
        xb  = self.xbounds
        A, mu, sigma = pars
        e1  = erf((xb[:-1]-mu)/(np.sqrt(2)*sigma))
        e2  = erf((xb[1:] -mu)/(np.sqrt(2)*sigma))
        y   = A*sigma*np.sqrt(np.pi/2)*(e2-e1)
        
        return y
    def _fitpars_to_gausspars(self,pfit):
        '''
        Transforms fit parameteres into gaussian parameters.
        '''
        return pfit                
    def _initialize_parameters(self):
        ''' Method to initialize parameters from data (pre-fit)
        Returns:
        ----
            p0: tuple with inital (amplitude, mean, sigma) values
        '''
        A0 = np.percentile(self.ydata,90)
        
        m0 = np.percentile(self.xdata,45)
        s0 = np.sqrt(np.var(self.xdata))/3
        p0 = (A0,m0,s0)
        self.initial_parameters = p0
        return p0
    def _initialize_bounds(self):
        ''' Method to initialize bounds from data (pre-fit)
        Returns:
        ----
            (lb,ub): tuple with bounds on the fitting parameters
        '''

        
        lb = (np.min(self.ydata), np.min(self.xdata), 0)
        ub = (np.max(self.ydata), np.max(self.xdata), self.sigmabound)
        self.bounds = (lb,ub)
        return (lb,ub)
    
    def jacobian(self,fitpars,x0=None,weights=None):
        '''
        Returns the Jacobian matrix of the __fitting__ function. 
        In the case x0 and weights are not provided, uses inital values.
        '''
        # Be careful not to put gaussian parameters instead of fit parameters!
        A, mu, sigma = fitpars
        weights = weights[1:-1] if weights is not None else self.weights[1:-1]
        if x0 is None:
            x = self.xdata#[1:-1]
        else:
            x = x0#[1:-1]
        y = self.evaluate(fitpars,x) 
        x = x[1:-1]
        dfdp = np.array([y/A,
                         y*(x-mu)/(sigma**2),
                         y*(x-mu)**2/(sigma**3)]).T
        return weights[:,None]*dfdp
    def calculate_center(self,pgauss=None):
        '''
        Returns the x coordinate of the line center.
        Calculates the line center by solving for CDF(x_center)=0.5
        '''
        pgauss = pgauss if pgauss is not None else self._get_gauss_parameters()[0]
        A,m,s = pgauss
        
        def eq(x):
            cdf =  0.5*erfc((m-x)/(s*np.sqrt(2)))
            return  cdf - 0.5
        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
        return x
    
    
class DoubleGaussian(EmissionLine):
    def model(self,pars):
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
        and 't' is defined as:
            
            t = (x - mu)/(sqrt(2) * sigma)
        
        Here, mu is the mean of the Gaussian.
        '''
        xb  = self.xbounds
        #A1,mu1,sigma1,fA,fm,sigma2 = pars
        #A2  = A1*fA
        #mu2 = mu1 + fm*np.max([sigma1,sigma2])
        #A1,mu1,sigma1,A2,mu2,sigma2 = self._fitpars_to_gausspars(pars)
        A1,mu1,sigma1,A2,mu2,sigma2 = pars
        
        e11  = erf((xb[:-1]-mu1)/(np.sqrt(2)*sigma1))
        e21  = erf((xb[1:] -mu1)/(np.sqrt(2)*sigma1))
        y1   = A1*sigma1*np.sqrt(np.pi/2)*(e21-e11)
        
        e12  = erf((xb[:-1]-mu2)/(np.sqrt(2)*sigma2))
        e22  = erf((xb[1:] -mu2)/(np.sqrt(2)*sigma2))
        y2   = A2*sigma2*np.sqrt(np.pi/2)*(e22-e12)
        
        return y1+y2
    
    def _initialize_parameters(self):
        ''' Method to initialize parameters from data (pre-fit)
        Returns:
        ----
            p0: tuple with inital (amplitude, mean, sigma) values
        '''
        A0 = np.percentile(self.ydata,90)
        
        m0 = np.percentile(self.xdata,50)
        s0 = np.sqrt(np.var(self.xdata))/3
        p0 = (A0,m0,s0,A0,m0,s0)
        self.initial_parameters = p0
        return p0
    def _initialize_bounds(self):
        ''' Method to initialize bounds from data (pre-fit)
        Returns:
        ----
            (lb,ub): tuple with bounds on the fitting parameters
        '''
        # ORIGINAL CONSTRAINTS
        lb = (np.min(self.ydata), np.min(self.xdata), 0,
              0, -3, 0)
        ub = (np.max(self.ydata), np.max(self.xdata), self.sigmabound,
              1, 3, self.sigmabound)
        
        # NO CONSTRAINTS
#        lb = (0., -np.inf, 0,         0, -np.inf, 0)
#        ub = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf )
        
#        GASPARE'S CONSTRAINTS
#        lb = (0, -np.inf, 0,
#              0, -3, 0)
#        ub = (np.inf,  np.inf, np.inf,
#              1, 3, np.inf)
        #  CONSTRAINTS
#        lb = (np.min(self.ydata), -np.inf, 0,         0, -np.inf, 0)
#        ub = (np.max(self.ydata), np.inf, np.inf, np.inf, np.inf, np.inf )
        
#        lb = (0,-np.inf,0, 0,-np.inf,0)
#        ub = (np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)
        
        return (lb,ub)
    def jacobian(self,pars,x0=None,weights=None):
        '''
        Returns the Jacobian matrix of the __fitting__ function. 
        In the case x0 and weights are not provided, uses inital values.
        '''
        # Be careful not to put gaussian parameters instead of fit parameters!
        A1, mu1, sigma1, A2, mu2, sigma2 = pars
        
        weights = weights[1:-1] if weights is not None else self.weights[1:-1]
        if x0 is None:
            x = self.xdata#[1:-1]
            #y = self.ydata[1:-1]
        else:
            x = x0#[1:-1]
        y1,y2 = self.evaluate(pars,x,separate=True) 
        #y = A * np.exp(-1/2*((x-mu)/sigma)**2) 
        x = x[1:-1]
        dfdp = np.array([y1/A1,
                         y1*(x-mu1)/(sigma1**2),
                         y1*(x-mu1)**2/(sigma1**3),
                         y2/A1,
                         y2*(x-mu2)/(sigma2**2),
                         y2*(x-mu2)**2/(sigma2**3)]).T
        return weights[:,None]*dfdp
    def calculate_center(self,pars=None):
        '''
        Returns the x coordinate of the line center.
        Calculates the line center by solving for CDF(x_center)=0.5
        '''
        pars = pars if pars is not None else self._get_gauss_parameters()[0]
        A1,m1,s1,A2,m2,s2 = pars
        print(pars)
        def eq(x):
            cdf =  0.5*erfc((m1-x)/(s1*np.sqrt(2))) + \
                  0.5*erfc((m2-x)/(s2*np.sqrt(2)))
            return  cdf/2 - 0.5
        print(eq(np.min(self.xdata)),eq(np.max(self.xdata)))
        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
        return x 
class SimpleGaussian(DoubleGaussian):
    def model(self,pars):
        ''' Calculates the expected electron counts by assuming:
            (1) The PSF is a Gaussian function,
            (2) The number of photons falling on each pixel is equal to the 
                integral of the PSF between the pixel edges. (In the case of 
                wavelengths, the edges are calculated as midpoints between
                the wavelength of each pixel.)
        
        '''
        x  = self.xdata[1:-1]
#        A1,mu1,sigma1,fA,fm,sigma2 = pars
        A1,mu1,sigma1,A2,mu2,sigma2 = pars
        
        y1   = A1*np.exp(-0.5*(x-mu1)**2/sigma1**2)
        y2   = A2*np.exp(-0.5*(x-mu2)**2/sigma2**2)
                
        return y1+y2

    def evaluate(self,pars,x=None,separate=False,ptype='gauss'):
        ''' Returns the evaluated Gaussian function along the provided `x' and 
        for the provided Gaussian parameters `p'. 
        
        Args:
        ---- 
            x: 1d array along which to evaluate the Gaussian. Defaults to xdata
            p: tuple (amplitude, mean, sigma) of Gaussian parameters. 
               Defaults to the fit parameter values.
        '''
        if x is None:
            x = self.xdata[1:-1]
            xb = self.xbounds
        else:
            x = x[1:-1]
            xb = (x[:-1]+x[1:])/2
        p = pars if pars is not None else self._get_gauss_parameters()[0]
        
#        g1 = A1 * np.exp(-1/2*((x-mu1)/sigma1)**2)[1:-1]
#        g2 = A2 * np.exp(-1/2*((x-mu2)/sigma2)**2)[1:-1]
        pi = np.reshape(p,(-1,3))
        N    = pi.shape[0]
        Y    = []
        for i in range(N):
            A, mu, sigma = pi[i]
            y    = A*np.exp(-0.5*(x-mu)**2/sigma**2)
            Y.append(y)
        
        
        if separate:
            return tuple(Y)
        else:
            return np.sum(Y,axis=0)
    def calculate_center(self,pgauss=None):
        '''
        Returns the x coordinate of the line center.
        Calculates the line center by solving for CDF(x_center)=0.5
        '''
        pgauss = pgauss if pgauss is not None else self._get_gauss_parameters()[0]
        A1,m1,s1,A2,m2,s2 = pgauss
#        print(pgauss)
        def eq(x):
            cdf =  0.5*erfc((m1-x)/(s1*np.sqrt(2))) + \
                  0.5*erfc((m2-x)/(s2*np.sqrt(2)))
            return  cdf - 1.5
        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
        return x 
class SpectralLine2(object):
    ''' Class with functions to fit LFC lines as pure Gaussians'''
    def __init__(self,xdata,ydata,kind='emission',yerr=None, weights=None,
                 absolute_sigma=True):
        ''' Initialize the object using measured data
        
        Args:
        ----
            xdata: 1d array of x-axis data (pixels or wavelengts)
            ydata: 1d array of y-axis data (electron counts)
            weights:  1d array of weights calculated using Bouchy method
            kind: 'emission' or 'absorption'
        '''
        def _unwrap_array_(array):
            if type(array)==pd.Series:
                narray = array.values
            elif type(array)==np.ndarray:
                narray = array
            return narray
            
            
        self.xdata   = _unwrap_array_(xdata)
        self.xbounds = (self.xdata[:-1]+self.xdata[1:])/2
        self.ydata   = _unwrap_array_(ydata)
        self.kind    = kind
        yerr         = yerr if yerr is not None else np.sqrt(np.abs(self.ydata))
        weights      = weights if weights is not None else yerr #np.ones_like(xdata)
        self.yerr    = _unwrap_array_(yerr)
        self.weights = _unwrap_array_(weights)
        
        
        self.success = False   
    def _fitpars_to_gausspars(self,pfit):
        '''
        Transforms fit parameteres into gaussian parameters.
        '''
        A1, m1, s1, fA, fm, s2 = pfit
        A2 = fA*A1
        D  = np.max([s1,s2])
        m2 = m1 + fm*D    
        return (A1,m1,s1,A2,m2,s2)                
    def _initialize_parameters(self):
        ''' Method to initialize parameters from data (pre-fit)
        Returns:
        ----
            p0: tuple with inital (amplitude, mean, sigma) values
        '''
        A0 = np.percentile(self.ydata,90)
        
        m0 = np.percentile(self.xdata,45)
        D  = np.mean(np.diff(self.xdata))
        s0 = np.sqrt(np.var(self.xdata))/3
        p0 = (A0,m0,s0,0.9,D,s0)
        self.initial_parameters = p0
        return p0
    def _initialize_bounds(self):
        ''' Method to initialize bounds from data (pre-fit)
        Returns:
        ----
            (lb,ub): tuple with bounds on the fitting parameters
        '''
        std = np.std(self.xdata)#/3
        if self.kind == 'emission':
            Amin = 0.
            Amax = np.max(self.ydata)
        elif self.kind == 'absorption':
            Amin = np.min(self.ydata)
            Amax = 0.
        peak = peakdetect(self.ydata,self.xdata,lookahead=2,delta=0)[0][0][0]
        lb = (Amin, np.min(self.xdata), 0,
              0, -2, 0)
        ub = (Amax, np.max(self.xdata), std,
              1, 2, std)
        self.bounds = (lb,ub)
        return (lb,ub)
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
            errors = self.fit_errors
        else:
            p0 = self._initialize_parameters()
            pars, errors = self.fit(p0)
            pfit = self.fit_parameters
            errors = self.fit_errors
        return pfit, errors
    def _get_gauss_parameters(self):
        ''' Method to check whether the fit has been successfully performed.
        If the fit was successful, returns the fit values. Otherwise, the 
        method performs the fitting procedure and returns the fit values.
        Returns:
        -------
            pfit: tuple with fitted (amplitude, mean, sigma) values
        '''
        if self.success == True:
            pars = self.gauss_parameters
            errors = self.gauss_errors
        else:
            p0 = self._initialize_parameters()
            pars, errors = self.fit(p0)
        return pars, errors
    
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
    def R2(self,pars=None,weights=None):
        ''' Returns the R^2 estimator of goodness of fit to the model.
        Args:
        ----
            pars: tuple (A1, mu1, sigma1, fA, fm, sigma2) of the model
        Returns:
        -------
            chisq
        '''
        if pars is None:
            pars = self._get_fit_parameters()[0]
        cdata = self.ydata[1:-1]
        weights = weights if weights is not None else self.weights[1:-1]
        SSR = 1 - np.sum(self.residuals(pars,weights)**2/np.std(cdata))
        SST = np.sum(weights*(cdata - np.mean(cdata))**2)
        rsq = 1 - SSR/SST
        return rsq
    def log_prior(self,pars=None):
        if pars is None:
            pars = self._get_fit_parameters()[0]
        A1,m1,s1,A2,m2,s2 = pars
        xmin = np.min(self.xdata)
        xmax = np.max(self.xdata)
        D = max([s1,s2])
        if ((s1<0) or (s2<0) or (A1<0) or (A2<0) or 
            (m1<xmin) or (m1>xmax) or (m2<xmin) or (m2>xmax)):
            return -np.inf # log(0)
        else:
            return - np.log(s1) - np.log(s2) - np.log(A1) - np.log(A2) 
    def log_likelihood(self,pars=None):
        if pars is None:
            pars = self._get_fit_parameters()[0]
        A1,m1,s1,A2,m2,s2 = pars
        y_model = self.model(theta,x)
        return np.sum(-0.5*np.log(2*np.pi*y_model) - (y[1:-1]-y_model)**2 / (2*y_model))
    def log_posterior(theta,x,y):
        lnprior = log_prior(theta,x)
        if lnprior == -np.inf:
            return -np.inf
        else:
            return lnprior + log_likelihood(theta,x,y)
    def evaluate(self,p,x=None,separate=False,ptype='gauss'):
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
        if ptype=='gauss':
            A1, mu1, sigma1, A2, mu2, sigma2 = p if p is not None else self._get_gauss_parameters()
        elif ptype=='fit':
            A1, mu1, sigma1, fA, fm, sigma2 = p if p is not None else self._get_fit_parameters()
            A1, mu1, sigma1, A2, mu2, sigma2 = self._fitpars_to_gausspars(p)
#        g1 = A1 * np.exp(-1/2*((x-mu1)/sigma1)**2)[1:-1]
#        g2 = A2 * np.exp(-1/2*((x-mu2)/sigma2)**2)[1:-1]
        
        e11  = erf((xb[:-1]-mu1)/(np.sqrt(2)*sigma1))
        e21  = erf((xb[1:] -mu1)/(np.sqrt(2)*sigma1))
        y1   = A1*sigma1*np.sqrt(np.pi/2)*(e21-e11)
        
        e12  = erf((xb[:-1]-mu2)/(np.sqrt(2)*sigma2))
        e22  = erf((xb[1:] -mu2)/(np.sqrt(2)*sigma2))
        y2   = A2*sigma2*np.sqrt(np.pi/2)*(e22-e12)
        
        
        if separate:
            return y1,y2
        else:
            return y1+y2
    def jacobian(self,fitpars,x0=None,weights=None):
        '''
        Returns the Jacobian matrix of the __fitting__ function. 
        In the case x0 and weights are not provided, uses inital values.
        '''
        # Be careful not to put gaussian parameters instead of fit parameters!
        A1, mu1, sigma1, fA, fm, sigma2 = fitpars
        D   = np.max([sigma1,sigma2])
        mu2 = mu1 + D*fm
        weights = weights[1:-1] if weights is not None else self.weights[1:-1]
        if x0 is None:
            x = self.xdata#[1:-1]
            #y = self.ydata[1:-1]
        else:
            x = x0#[1:-1]
        y1,y2 = self.evaluate(fitpars,x,separate=True,ptype='fit') 
        #y = A * np.exp(-1/2*((x-mu)/sigma)**2) 
        x = x[1:-1]
        dfdp = np.array([y1/A1 + y2/A1,
                         y1*(x-mu1)/(sigma1**2) + y2*(x-mu2)/(sigma2**2),
                         y1*(x-mu1)**2/(sigma1**3),
                         y2/fA,
                         y2*(x-mu2)/(sigma2**2)*D,
                         y2*(x-mu2)**2/(sigma2**3)]).T
        return weights[:,None]*dfdp
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
    def model(self,A1,mu1,sigma1,fA,fm,sigma2):
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
        and 't' is defined as:
            
            t = (x - mu)/(sqrt(2) * sigma)
        
        Here, mu is the mean of the Gaussian.
        '''
        xb  = self.xbounds
        
        A2  = A1*fA
        mu2 = mu1 + fm*np.max([sigma1,sigma2])
        
        e11  = erf((xb[:-1]-mu1)/(np.sqrt(2)*sigma1))
        e21  = erf((xb[1:] -mu1)/(np.sqrt(2)*sigma1))
        y1   = A1*sigma1*np.sqrt(np.pi/2)*(e21-e11)
        
        e12  = erf((xb[:-1]-mu2)/(np.sqrt(2)*sigma2))
        e22  = erf((xb[1:] -mu2)/(np.sqrt(2)*sigma2))
        y2   = A2*sigma2*np.sqrt(np.pi/2)*(e22-e12)
        
        return y1+y2
    def fit(self,p0=None,absolute_sigma=True, bounded=True,
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
        
        if bounded == True:
            bounds = self._initialize_bounds()
        else:
            bounds=(-np.inf, np.inf)
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
        if method == 'lm':    
            return_full = kwargs.pop('full_output', False)
#            wrapped_jac = self._wrap_jac()
            res = leastsq(self.residuals,p0,Dfun=None,full_output=1)#,col_deriv=True,**kwargs)
            pfit, pcov, infodict, errmsg, ier = res
            cost = np.sum(infodict['fvec']**2)
            if ier not in [1, 2, 3, 4]:
                #raise RuntimeError("Optimal parameters not found: " + errmsg)
                pfit = np.full_like(p0,np.nan)
                pcov = None
                success = False
            else:
                success = True
        else:
            res = least_squares(self.residuals, p0, jac=self.jacobian, bounds=bounds, method=method,
                                **kwargs)
            if not res.success:
                #raise RuntimeError("Optimal parameters not found: " + res.message)
                pfit = np.full_like(p0,np.nan)
                pcov = None
                success = False
                cost = np.inf
            else:
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
        error_fit_pars = [] 
        for i in range(len(pfit)):
            try:
              error_fit_pars.append(np.absolute(pcov[i][i])**0.5)
            except:
              error_fit_pars.append( 0.00 )
        
        # From fit parameters, calculate the parameters of the two gaussians 
        # and corresponding errors
        if success == True:
            A1, m1, s1, fA, fm, s2 = pfit
            
            A2 = fA*A1
            D  = max([s1,s2])
            m2 = m1 + fm*D
            
            error_A1, error_m1, error_s1, error_fA, error_fm, error_s2 = error_fit_pars
            
            error_A2 = np.sqrt((A2/A1*error_A1)**2 +  (A2/fA*error_fA)**2)
            if D == s1:
                error_D = error_s1
            elif D==s2:
                error_D = error_s2
            error_m2 = np.sqrt(error_m1**2 + error_D**2)
            
            # Make the component with the smaller mean to be m1 and the  
            # component with the larger mean to be m2. (i.e. m1<m2)
            
            if m1<m2:
                gp_c1 = A1, m1, s1
                gp_c2 = A2, m2, s2
                gp_c1error = error_A1, error_m1, error_s1
                gp_c2error = error_A2, error_m2, error_s2
                
                
            elif m1>m2:
                gp_c1 = A2, m2, s2
                gp_c2 = A1, m1, s1
                gp_c1error = error_A2, error_m2, error_s2  
                gp_c2error = error_A1, error_m1, error_s1            
            else:
                print("m1=m2 ?", m1==m2)
            gauss_parameters = np.array([*gp_c1,*gp_c2])
            gauss_errors     = np.array([*gp_c1error,*gp_c2error])
            fit_parameters   = pfit
            fit_errors       = error_fit_pars
            
        else:
            gauss_parameters = np.full_like(pfit,np.nan)
            gauss_errors     = np.full_like(pfit,np.nan)
            fit_parameters   = pfit
            fit_errors       = error_fit_pars
            
        self.covar     = pcov
        self.rchi2     = cost / dof
        self.dof       = dof
        
        
        self.gauss_parameters = gauss_parameters
        self.gauss_errors     = gauss_errors
        
        
        self.fit_parameters   = fit_parameters
        self.fit_errors       = fit_errors
        
        self.center           = self.calculate_center(gauss_parameters)
#        self.center_error     = self.calculate_center_uncertainty(pfit,pcov)
        self.center_mass      = np.sum(self.weights*self.xdata*self.ydata)/np.sum(self.weights*self.ydata)
        
        
        
#        self.infodict = infodict
#        self.errmsg = errmsg
        self.success = success
        self.cost = cost
        if return_full:
            return gauss_parameters, gauss_errors, infodict, errmsg, ier
        else:
            return gauss_parameters, gauss_errors
    def calculate_center(self,pgauss=None):
        '''
        Returns the x coordinate of the line center.
        Calculates the line center by solving for CDF(x_center)=0.5
        '''
        pgauss = pgauss if pgauss is not None else self._get_gauss_parameters()[0]
        A1,m1,s1,A2,m2,s2 = pgauss
        
        def eq(x):
            cdf =  0.5*erfc((m1-x)/(s1*np.sqrt(2))) + \
                  0.5*erfc((m2-x)/(s2*np.sqrt(2)))
            return  cdf/2 - 0.5
        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
        return x
    def calculate_center_uncertainty(self,pfit=None,covar=None,N=200):
        ''' 
        Returns the standard deviation of centres drawn from a random sample.
        
        Draws N samples by randomly sampling the provided fit parameteres and
        corresponding errors to construct N models of the line. Line centre 
        is calculated for each of the N models.         
        '''
        pfit = pfit if pfit is not None else self._get_fit_parameters()[0]    
        C = covar if covar is not None else self.covar
        # It is not possible to calculate center uncertainty if the covariance 
        # matrix contains infinite values
        if np.isinf(C).any() == True:
            return -1
        else:
            pass
        mdgN  = np.random.multivariate_normal(mean=pfit,cov=C,size=N)
        cut   = np.where(mdgN[:,3]>0)[0]
        mdgN  = mdgN[cut]
        centers = np.zeros(cut.size)
        for i,pars in enumerate(mdgN):
            pgauss_i    = self._fitpars_to_gausspars(pars)
            centers[i]  = self.calculate_center(pgauss_i)
        return centers.std()
    def calculate_photon_noise(self):
        '''INCORRECT'''
        deriv   = derivative1d(self.ydata,self.xdata)
        weights = pd.Series(deriv**2*829**2/self.ydata)
        weights = weights.replace([np.inf,-np.inf],np.nan)
        weights = weights.dropna()
        return 1./np.sqrt(weights.sum())*299792458e0
    def plot(self,fit=True,cofidence_intervals=True,ax=None,**kwargs):
        ''' Plots the line as a histogram of electron counts with corresponding
        errors. If `fit' flag is True, plots the result of the fitting 
        procedure.
        
        
        '''
        import matplotlib.transforms as mtransforms
        if ax is None:
            fig,ax = get_fig_axes(1,figsize=(9,9),bottom=0.12,left=0.15,**kwargs)
            self.fig = fig
        self.ax_list  = ax
        widths = np.diff(self.xdata)[:-1]
        ax[0].bar(self.xdata[1:-1],self.ydata[1:-1],
                  widths,align='center',alpha=0.3,color='#1f77b4')
        ax[0].errorbar(self.xdata[1:-1],self.ydata[1:-1],
                       yerr=self.yerr[1:-1],fmt='o',color='#1f77b4')
        yeval = np.zeros_like(self.ydata)
        if fit is True:
            p,pe = self._get_gauss_parameters()
#            xeval = np.linspace(np.min(self.xdata),np.max(self.xdata),100)
            xeval = self.xdata
            y1,y2 = self.evaluate(p,xeval,True,ptype='gauss')
            yeval = y1+y2
            fit = True
            xeval = xeval[1:-1]
            ax[0].plot(xeval,yeval,color='#ff7f0e',marker='o')
            ax[0].plot(xeval,y1,color='#2ca02c',lw=0.7,ls='--')
            ax[0].plot(xeval,y2,color='#2ca02c',lw=0.7,ls='--')
            A1, m1, s1, A2, m2, s2 = p
            ax[0].plot([m1,m1], [0,A1],ls='--',lw=0.7,color='#2ca02c')
            ax[0].plot([m2,m2], [0,A2],ls='--',lw=0.7,color='#2ca02c')
              
            # calculate the center of the line and the 1-sigma uncertainty
#            cenx = self.center
#            ceny = self.evaluate(p,np.array([m1,cenx,m2]),ptype='gauss')
#            ax[0].plot([cenx,cenx],[0,ceny[1]],ls='--',lw=1,c='C1')
#            cend = self.center_error
            
            # shade the area around the center of line (1-sigma uncertainty)
#            xcenval = np.linspace(cenx-cend,cenx+cend,100)
#            ycenval = self.evaluate(p,xcenval,ptype='gauss')
#            ax[0].fill_between(xcenval,0,ycenval,color='C1',alpha=0.4,
#              where=((xcenval>=cenx-cend)&(xcenval<=cenx+cend)))
        if cofidence_intervals is True and fit is True:
            xeval = self.xdata
            y,ylow,yhigh = self.confidence_band(xeval,confprob=0.05)
            ax[0].fill_between(xeval[1:-1],ylow,yhigh,alpha=0.5,color='#ff7f0e')
            y,ylow,yhigh = self.confidence_band(xeval,confprob=0.32)
            ax[0].fill_between(xeval[1:-1],ylow,yhigh,alpha=0.2,
                              color='#ff7f0e')
        ymax = np.max([1.2*np.percentile(yeval,95),1.2*np.max(self.ydata)])
        ax[0].set_ylim(-np.percentile(yeval,20),ymax)
        ax[0].set_xlabel('Pixel')
        ax[0].set_ylabel('Counts')
        ax[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        return
    def confidence_band(self, x, confprob=0.05, absolute_sigma=False):
        
        # https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html \
                            #confidence-and-prediction-intervals
        from scipy.stats import t
        # Given the confidence probability confprob = 100(1-alpha)
        # we derive for alpha: alpha = 1 - confprob
        alpha = 1.0 - confprob
        prb = 1.0 - alpha/2
        tval = t.ppf(prb, self.dof) #degrees of freedom
                    
        C = self.covar
       
        p,pe = self._get_fit_parameters()
        n = len(p)              # Number of parameters from covariance matrix
        N = len(x)
        if absolute_sigma:
            covscale = 1.0
        else:
            covscale = self.rchi2 * self.dof
          
        y = self.evaluate(p,x,ptype='fit')
        
        # If the x array is larger than xdata, provide new weights
        # for all points in x by linear interpolation
        int_function = interpolate.interp1d(self.xdata,self.weights)
        weights      = int_function(x)
        dfdp = self.jacobian(p,x,weights).T
        
        df2 = np.zeros(N-2)
        for j in range(n):
            for k in range(n):
                df2 += dfdp[j]*dfdp[k]*C[j,k]
#        df2 = np.dot(np.dot(dfdp.T,self.covar),dfdp).sum(axis=1)
        df = np.sqrt(covscale*df2)
        delta = tval * df
        upperband = y + delta
        lowerband = y - delta
        return y, upperband, lowerband       
        
class Worker(object):   
    def __init__(self,filename=None,mode=None,manager=None,
                 orders=None):
        self.filename = filename
        self.open_file(self.filename,mode)
        self.manager = manager
        print(self.file)
        eo = self.check_exists("orders")
        if eo == False:
            if not orders:
                orders = np.arange(sOrder,eOrder)
            self.file.create_dataset("orders",data=orders)
        self.manager.get_file_paths('AB')
        
        
        print('Worker initialised')  
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
    def distortion_node(self,i,fibre='AB'):
        return ["{}/{}/{}".format(i,f,t) for f in list(fibre) 
                                 for t in ['wave','pix','rv']]
    def do_wavelength_stability(self,refA=None,refB=None,tharA=None,tharB=None,
                                LFC_A0='FOCES',LFC_B0='FOCES',filelim=None):
        if not refA:
            return "Reference wavelength solution for fibre A not given"
        if not refB:
            return "Reference wavelength solution for fibre B not given"
            
        o = self.is_open()
        if o == False:
            self.open_file()
        else:
            pass
        
        fileA0 = self.manager.file_paths['A'][0]
        fileB0 = self.manager.file_paths['B'][0]
        specA0 = Spectrum(fileA0,data=True,LFC=LFC_A0)
        specB0 = Spectrum(fileB0,data=True,LFC=LFC_B0)
        tharA  = specA0.__get_wavesol__('ThAr')
        tharB  = specB0.__get_wavesol__('ThAr')
        wavecoeff_airA = specA0.wavecoeff_air
        wavecoeff_airB = specB0.wavecoeff_air
        wavesol_refA  = specA0.__get_wavesol__('LFC')
        wavesol_refB  = specB0.__get_wavesol__('LFC')
        
        numfiles = self.manager.numfiles[0]
        for i in range(numfiles):
            e = self.check_exists("{}".format(i))
            
            nodes = ["{}/{}/{}".format(i,f,t) for f in ["A","B"] 
                     for t in ["wavesol_LFC","rv","weights","lines","coef"]]
            ne = [self.check_exists(node) for node in nodes]
               
            # THIS SECTION IS MEANT TO WORK FOR DATA FROM APRIL 17th ONLY!!
            filelim = {'A':self.manager.file_paths['A'][93], 
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
                
                rvA      = (wavesolA[sOrder:eOrder] - wavesol_refA)/wavesol_refA * 299792458
                rvB      = (wavesolB[sOrder:eOrder] - wavesol_refB)/wavesol_refB * 299792458
                
                  
                weightsA = specA.get_weights2d()[sOrder:eOrder]
                weightsB = specB.get_weights2d()[sOrder:eOrder]
                
                linesA   = specA.lines.values
                linesB   = specB.lines.values
                
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
    def do_distortion_calculation(self,fibre='AB'):
        
        o = self.is_open()
        if o == False:
            self.open_file()
        else:
            pass
        
        
        for fi,f in enumerate(list(fibre)):
            numfiles = self.manager.numfiles[fi]
            for i in range(numfiles):   
                e = self.check_exists("{}".format(i))
                
                nodes = self.distortion_node(i,fibre=f)
                ne = [self.check_exists(node) for node in nodes]
#                print(nodes)
#                print(e, ne)
                if ((e == False) or (np.all(ne)==False)):
                    print("Working on {}/{}".format(i+1,numfiles))
                    spec = Spectrum(self.manager.file_paths[f][i],LFC='HARPS')
                    spec.polyord = 8
                    spec.__get_wavesol__('LFC',gaps=False,patches=False)
                    #spec.plot_distortions(plotter=plotter,kind='lines',show=False)
                    dist = spec.get_distortions()
                    
                    wav  = dist.sel(typ='wave')
                    pix  = dist.sel(typ='pix')
                    rv   = dist.sel(typ='rv')
                    nodedata = [wav,pix,rv]
                    self.save_nodes(nodes,nodedata)
        return
    def read_distortion_file(self,filename=None):
        if filename is None:
            filename = self.filename
        self.open_file()
        
        l    = len(self.file)
        data = xr.DataArray(np.full((l,2,3,nOrder,500),np.nan),
                            dims=['fn','fbr','typ','od','val'],
                            coords = [np.arange(l),
                                      ['A','B'],
                                      ['wave','pix','rv'],
                                      np.arange(sOrder,eOrder),
                                      np.arange(500)])
        for i in range(l):
            nodes = self.distortion_node(i,fibre='AB')
            for node in nodes:
                print(node)
                e = self.check_exists(node)
                if e == True:
                    fn,fbr,typ = node.split('/')
                    fn = int(fn)
                    if fbr == 'A':
                        ods=np.arange(sOrder,eOrder)
                    elif fbr == 'B':
                        ods=np.arange(sOrder,eOrder-1)
                    data.loc[dict(fn=fn,fbr=fbr,typ=typ,od=ods)] = self.file[node][...]
        self.distortion_data = data
        return data  
    def save_nodes(self,nodenames,nodedata):
        for node,data in zip(nodenames,nodedata):
            node_exists = self.check_exists(node)
#            print(node,node_exists)
            if node_exists==False:
                print('Saving node:',node)
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
                 begin=None,end=None,run=None,sequence=None,get_file_paths=True):
        '''
        date(yyyy-mm-dd)
        begin(yyyy-mm-dd)
        end(yyyy-mm-dd)
        sequence(day,sequence)
        '''
        gaspare_url     = 'http://people.sc.eso.org/~glocurto/COMB/'
        
        self.file_paths = []
        self.spectra    = []
        #harpsDataFolder = os.path.join("/Volumes/home/dmilakov/harps","data")
        harpsDataFolder = harps_data#os.path.join("/Volumes/home/dmilakov/harps","data")
        self.harpsdir   = harpsDataFolder
        if sequence!=None:
            run = run if run is not None else ValueError("No run selected")
            
            if type(sequence)==tuple:
                sequence_list_filepath = urllib.parse.urljoin(gaspare_url,'/COMB_{}/day{}_seq{}.list'.format(run,*sequence),True)
                self.sequence_list_filepath = [sequence_list_filepath]
                self.sequence = [sequence]
            elif type(sequence)==list:
                self.sequence_list_filepath = []
                self.sequence = sequence
                for item in sequence:
                    sequence_list_filepath = urllib.parse.urljoin(gaspare_url,'/COMB_{}/day{}_seq{}.list'.format(run,*item))
                    self.sequence_list_filepath.append(sequence_list_filepath)
        if sequence == None:
            self.sequence_list_filepath = None
            if   date==None and (year!=None and month!=None and day!=None) and (begin==None or end==None):
                self.dates = ["{y:4d}-{m:02d}-{d:02d}".format(y=year,m=month,d=day)]
                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
            elif date!=None:
                self.dates = [date]
                
                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
                print(self.datadir)
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
            print(self.sequence_list_filepath)
            if type(self.sequence_list_filepath)==list:    
                sequence_list = []
                
                for item,seq in zip(self.sequence_list_filepath,self.sequence):
                    #wp  = os.path.join(gaspare_url,'COMB_{}'.format(run))
                    req = urllib.request.Request(item)
                    res = urllib.request.urlopen(req)
                    htmlBytes = res.read()
                    htmlStr   = htmlBytes.decode('utf8').split('\n')
                    sequence_list = htmlStr[:-1]
                    print(sequence_list)
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
        #nOrder  = eOrder-sOrder     # number of orders in image
        
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
class SpectrumPlotter(object):
    def __init__(self,naxes=1,ratios=None,title=None,sep=0.05,figsize=(16,9),
                 alignment="vertical",sharex=None,sharey=None,**kwargs):
        fig, axes = get_fig_axes(naxes,ratios=ratios,title=title,
                                 sep=sep,alignment=alignment,
                                 figsize=figsize,sharex=sharex,
                                 sharey=sharey,**kwargs)
        self.figure = fig
        self.axes   = axes   
        
        
class ManagerPlotter(object):
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
def accuracy(w=None,SNR=10,dx=829,u=0.9):
    '''
    Returns the rms accuracy of a spectral line with SNR=10, 
    pixel size = 829 m/s and apsorption strength 90%.
    
    Equation 4 from Cayrel 1988 "Data Analysis"
    '''
    if w is None:
        raise ValueError("No width specified")
    epsilon = 1/SNR
    return np.sqrt(2)/np.pi**0.25 * np.sqrt(w*dx)*epsilon/u
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
def double_gaussN_erf(x,params):
    if type(x) == pd.Series:
        x = x.values
    else:
        pass
    N = params.shape[0]
    y = np.zeros_like(x,dtype=np.float)
    xb = (x[:-1]+x[1:])/2
    gauss1 = params[['amplitude1','center1','sigma1']]
    gauss2 = params[['amplitude2','center2','sigma2']]
    for i in range(N):
        A1, mu1, sigma1 = gauss1.iloc[i]
        A2, mu2, sigma2 = gauss2.iloc[i]
        
        e11 = erf((xb[:-1]-mu1)/(np.sqrt(2)*sigma1))
        e21 = erf((xb[1:] -mu1)/(np.sqrt(2)*sigma1))
        y[1:-1] += A1*sigma1*np.sqrt(np.pi/2)*(e21-e11)
        
        e12 = erf((xb[:-1]-mu2)/(np.sqrt(2)*sigma2))
        e22 = erf((xb[1:] -mu2)/(np.sqrt(2)*sigma2))
        y[1:-1] += A2*sigma2*np.sqrt(np.pi/2)*(e22-e12)
        
        
    return y
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
def fit_peak(i,xarray,yarray,yerr,weights,xmin,xmax,dx,method='erfc',
             model=None,verbose=0):
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
        covar:    covariance matrix of parameters
    '''
    def calculate_photon_noise(weights):
        # FORMULA 10 Bouchy
        return 1./np.sqrt(weights.sum())*299792458e0
        
    # Prepare output array
    # number of parameters
    model = model if model is not None else 'singlegaussian'
    if model=='singlegaussian':
        n = 3
        model_class = SingleGaussian
    elif model=='doublegaussian':
        n = 6
        model_class = DoubleGaussian
    elif model=='simplegaussian':
        n=6
        model_class = SimpleGaussian
    #print(model)
    dtype = np.dtype([('pars',np.float64,(n,)),
                      ('errors',np.float64,(n,)),
                      ('pn',np.float64,(1,)),
                      ('r2',np.float64,(1,)),
                      ('cen',np.float64,(1,)),
                      ('cen_err',np.float64,(1,))])
    results = np.empty(shape=(1,),dtype=dtype)
    
    # Fit only data between the two adjacent minima of the i-th peak
    if i<np.size(xmin)-1:
        cut = xarray.loc[((xarray>=xmin[i])&(xarray<=xmin[i+1]))].index
    else:
        #print("Returning results")
        return results

    # If this selection is not an empty set, fit the Gaussian profile
    if verbose>0:
        print("LINE:{0:<5d} cutsize:{1:<5d}".format(i,np.size(cut)))
    if cut.size>6:
        x    = xarray.iloc[cut]#.values
        y    = yarray.iloc[cut]#.values
        ye   = yerr.iloc[cut]
       
        wght = weights[cut]
        wght = wght/wght.sum()
        pn   = calculate_photon_noise(wght)
        ctr  = xmax[i]
        amp  = np.max(yarray.iloc[cut])
        sgm  = 3*dx[i]
        if method == 'curve_fit':              
            guess                          = [amp, ctr, sgm] 
            #print("{} {},{}/{} {} {}".format(order,scale,i,npeaks,guess,cut.size))
            try:
                best_pars, pcov                = curve_fit(model, 
                                                          x, y, 
                                                          p0=guess)
            except:
                return ((-1.0,-1.0,-1.0),np.nan)

        elif method == 'chisq':
            params                      = [amp, ctr, sgm] 
            result                      = minimize(chisq,params,
                                                   args=(x,y,wght))
            best_pars                      = result.x
            

        elif method == 'erfc':
            line   = model_class(x,y,weights=ye)
            if verbose>1:
                print("LINE{0:>5d}".format(i),end='\t')
            pars, errors = line.fit(bounded=True)
            center       = line.center
            center_error = line.center_error
            rsquared     = line.calc_R2()
            if verbose>1:
                print("ChiSq:{0:<10.5f} R2:{1:<10.5f}".format(line.rchi2,line.R2()))
            if verbose>2:
                columns = ("A1","m1","s1","A2","m2","s2")
                print("LINE{0:>5d}".format(i),(6*"{:>20s}").format(*columns))
                print("{:>9}".format(''),(6*"{:>20.6e}").format(*pars))
            #            line.plot()
#            sys.exit()
        elif method == 'epsf':
            pass
        else:
            sys.exit("Method not recognised!")
        results['pars']    = pars
        results['pn']      = pn
        results['errors']  = errors
        results['r2']      = rsquared
        results['cen']     = center
        results['cen_err'] = center_error
    return results
    #return np.concatenate((best_pars,np.array([pn])))
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
        a,c,s,pn,ct = params.iloc[i]
        y = y + a*np.exp((-((x-c)/s)**2)/2.)
    return y
def gaussN_erf(x,params):
    if type(x) == pd.Series:
        x = x.values
    else:
        pass
    N = params.shape[0]
    y = np.zeros_like(x,dtype=np.float)
    xb = (x[:-1]+x[1:])/2
    for i in range(N):
        A,mu,sigma,A_error,mu_error,sigma_error,pn,ct = params.iloc[i]
        sigma = np.abs(sigma)
        e1 = erf((xb[:-1]-mu)/(np.sqrt(2)*sigma))
        e2 = erf((xb[1:] -mu)/(np.sqrt(2)*sigma))
        y[1:-1] += A*sigma*np.sqrt(np.pi/2)*(e2-e1)
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
    # assuming black background
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
    rmean = running_mean(points,window)
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
def polynomial3(x, a0,a1,a2,a3):
    return a0 + a1*x + a2*x**2 + a3*x**3

def rms(x):
    ''' Returns root mean square of input array'''
    return np.sqrt(np.mean(np.square(x)))
def running_mean(x, N):
        return np.convolve(x, np.ones((N,))/N)[(N-1):]
def select_orders(orders):
    use = np.zeros((nOrder,),dtype=bool); use.fill(False)
    for order in range(sOrder,eOrder,1):
        if order in orders:
            o = order - sOrder
            use[o]=True
    col = np.where(use==True)[0]
    return col