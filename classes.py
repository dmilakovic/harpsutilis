#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:58:21 2018

@author: dmilakov
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import gc
import os
import warnings
import urllib
import datetime
import dill as pickle
import tqdm
import errno
import sys
import time


from glob import glob
from astropy.io import fits
from scipy.optimize import curve_fit, leastsq, newton
from scipy import odr, interpolate
from joblib import Parallel,delayed
#from pathos.pools import ProcessPool
from pathos.pools import ProcessPool, ParallelPool
from multiprocessing import Pool
from scipy.signal import find_peaks_cwt as findpeaks

from harps import functions as hf
from harps import settings as hs

__version__ = '0.4.13'

harps_home   = hs.harps_home
harps_data   = hs.harps_data
harps_dtprod = hs.harps_dtprod
harps_plots  = hs.harps_plots
harps_prod   = hs.harps_prod

sOrder       = hs.sOrder
eOrder       = hs.eOrder
nOrder       = eOrder-sOrder




class Spectrum(object):
    ''' Spectrum object contains functions and methods to read data from a 
        FITS file processed by the HARPS pipeline
    '''
    def __init__(self,filepath=None,LFC='HARPS',ftype='e2ds',
                 header=True,readdata=True):
        '''
        Initialise the spectrum object.
        '''
        self.filepath = filepath
        self.name = "HARPS Spectrum"
        self.datetime = np.datetime64(os.path.basename(filepath).split('.')[1].replace('_',':')) 
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
        self.lines      = None
        
        self.use_gaps       = False
        self.patches    = True
        # polynomial order = self.polyord+1
        self.polyord    = 7
        
        self.segsize    = 4096//16 #pixel
        
        self.lineDetectionPerformed=False
        self.lineFittingPerformed = dict(epsf=False,gauss=False)
        if header == True:
            self.__read_meta__()
        else:
            pass
        if readdata == True:
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

    def check_and_get_wavesol(self,calibrator='LFC',order=None):
        ''' Check and retrieve the wavelength calibration'''
        wavesol_name = 'wavesol_{cal}'.format(cal=calibrator)
        exists_calib = False if getattr(self,wavesol_name) is None else True
        #print("{} calibration exists = {}".format(calibrator,exists_calib))
        if calibrator=='thar': calibrator='ThAr'
        # Run wavelength calibration if the wavelength calibration has not yet 
        # been performed  
        orders = self.prepare_orders(order)
        if exists_calib == False:
            wavesol = self.__get_wavesol__(calibrator,orders=orders)
        else:
            # Load the existing wavelength calibration and check if all wanted
            # orders have been calibrated
            wavesol = getattr(self,wavesol_name)
            ws_exists_all = np.all(wavesol.sel(od=orders))
            if ws_exists_all == False:
                wavesol = self.__get_wavesol__(calibrator,orders=orders)
        return wavesol
    def check_and_get_comb_lines(self,calibrator='LFC',orders=None):
        ''' Check and retrieve the positions of lines '''
        
        # Check if the Spectrum instance already has the attribute lines
        exists_lines = True if self.lines is not None else False
        if exists_lines == False:
            wavesol = self.check_and_get_wavesol(calibrator,orders)
            del(wavesol)
            lines = self.lines
        else:
            lines = self.lines
            orders = self.prepare_orders(orders)
            list_order = []
            for order in orders:
                # check if all values in 'pix' axis of the lines are nan
                # if they are not, the order exists
                exists_order = ~xr.ufuncs.isnan(lines['line'].sel(ax='pix',od=order).dropna('id','all')).all()
                list_order.append(exists_order.values)
            lines_exist_all = np.any(list_order)
            if lines_exist_all == True:
                #wavesol = self.check_and_get_wavesol('LFC',orders)
                lines = self.lines
            else:
                # run line detection on missing orders
                lines = self.detect_lines(order=orders)
        return lines
    def check_and_return_lines(self):
        self.__check_and_load__()
        
        existLines = True if self.lines is not None else False
        if not existLines:
            order = self.prepare_orders(None)
            lines = hf.return_empty_dataset(order,self.pixPerLine)
        
            self.lines = lines
        else:
            lines = self.lines
        return lines
    def check_and_load_psf(self,filepath=None):
        exists_psf = hasattr(self,'psf')
        
        if not exists_psf:
            self.load_psf(filepath)
        else:
            pass
        segments        = np.unique(self.psf.coords['seg'].values)
        N_seg           = len(segments)
        # segment limits
        sl              = np.linspace(0,4096,N_seg+1)
        # segment centers
        sc              = (sl[1:]+sl[:-1])/2
        sc[0] = 0
        sc[-1] = 4096
        
        self.segments        = segments
        self.nsegments       = N_seg
        self.segsize         = self.npix//N_seg
        self.segment_centers = sc
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
                warnings.warn("No ThAr calibration attached")
                pass
        self.date    = self.header["DATE"]
        self.exptime = self.header["EXPTIME"]
        # Fibre information is not saved in the header, but can be obtained 
        # from the filename 
        self.fibre   = self.filepath[-6]
        self.fibre_shape = 'octogonal'
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
            self.reprate      = self.modefilter*self.fr_source #Hz
            self.pixPerLine   = 22
            # wiener filter window scale
            self.window       = 3
        elif self.LFC=='FOCES':
            self.modefilter   = 100
            self.f0_source    = 20e6 #Hz
            self.reprate      = self.modefilter*self.fr_source #Hz
            self.anchor       = round(288.08452e12,-6) #Hz
            self.pixPerLine   = 35
            # wiener filter window scale
            self.window       = 5
        self.omega_r = 250e6
        m,k            = divmod(
                            round((self.anchor-self.f0_source)/self.fr_source),
                                   self.modefilter)
        self.f0_comb   = (k)*self.fr_source + self.f0_source
        return
        
    def __read_data__(self,flux_electrons=True):
        ''' Method to read data from the FITS file
        Args:
        ---- 
            flux_electrons : flux is in electron counts'''
        if len(self.hdulist)==0:
            self.hdulist = fits.open(self.filepath,memmap=False)
        if   self.ftype=="s1d" or self.ftype=="e2ds":
            data = self.hdulist[0].data.copy()
        elif self.ftype=="":
            data = self.hdulist[1].data.copy()
        if flux_electrons == True:
            data = data * self.conad
            self.fluxu = "e-"
        else:
            self.fluxu = "ADU"
        self.data = data
        return self.data
    def __get_wavesol__(self,calibrator="ThAr",nobackground=True,vacuum=True,
                        orders=None,fittype=['epsf','gauss'],model=None,
                        patches=None,gaps=None,
                        polyord=None,**kwargs):
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
        
        if orders is None:
            if calibrator == "ThAr":
                orders = np.arange(0,self.nbo,1)
            if calibrator == "LFC":
                orders = np.arange(self.sOrder,self.nbo,1)
                
        patches = patches if patches is not None else self.patches
        gaps    = gaps if gaps is not None else self.use_gaps
        def patch_fit(patch,polyord=None,fit_method='curve_fit'):
            ''' Fits a given patch with a polynomial function'''
            polyord = polyord if polyord is not None else self.polyord
            pix     = patch['pars'].sel(par='cen',ft=ftype)
            pix_err = patch['pars'].sel(par='cen_err',ft=ftype)
            freq    = patch['attr'].sel(att='freq')
            freq_err= patch['attr'].sel(att='freq_err')
            lbd     = 299792458e0/freq*1e10
            lbd_err = 299792458e0/freq_err*1e10
              
            data_axis = np.array(lbd.values,dtype=np.float64)
            data_err  = np.array(lbd_err.values,dtype=np.float64)
            x_axis    = np.array(pix.values,dtype=np.float64)
            x_err     = np.array(pix_err.values,dtype=np.float64)
            datanan,xnan = (np.isnan(data_axis).any(),np.isnan(x_axis).any())
            
            if (datanan==True or xnan==True):
                print("NaN values in data or x")
            if x_axis.size>polyord:
                coef = np.polyfit(x_axis,data_axis,polyord)
                if fit_method == 'curve_fit':
                    coef,pcov = curve_fit(hf.polynomial,x_axis,data_axis,p0=coef[::-1])
                    coef = coef[::-1]
                    coef_err = []
                if fit_method == 'ord':
                    data  = odr.RealData(x_axis,data_axis,sx=x_err,sy=data_err)
                    model = odr.polynomial(order=polyord)
                    fit   = odr.ODR(data,model,beta0=coef)
                    out   = fit.run()
                    coef  = out.beta
                    coef_err = out.sd_beta
                if fit_method == 'spline':
                    spline_rep = interpolate.splrep(x=x_axis,y=data_axis,w=1./data_err)
                    
                    
            else: 
                coef = None  
                coef_err = None
            return coef,coef_err
        def patch_fit_spline(patch,ftype):
            ''' Fits a given patch with a spline function'''
            pix     = patch['pars'].sel(par='cen',ft=ftype)
            pix_err = patch['pars'].sel(par='cen_err',ft=ftype)
            freq    = patch['attr'].sel(att='freq')
            freq_err= patch['attr'].sel(att='freq_err')
            lbd     = 299792458e0/freq*1e10
            lbd_err = 299792458e0/freq_err*1e10
              
            data_axis = np.array(lbd.values,dtype=np.float64)
            data_err  = np.array(lbd_err.values,dtype=np.float64)
            x_axis    = np.array(pix.values,dtype=np.float64)
            x_err     = np.array(pix_err.values,dtype=np.float64)
            datanan,xnan = (np.isnan(data_axis).any(),np.isnan(x_axis).any())
            if (datanan==True or xnan==True):
                print("NaN values in data or x")
            
            spline_rep = interpolate.splrep(x=x_axis,y=data_axis)#,w=1./data_err)
            return spline_rep
        
        def fit_wavesol(lines_in_order,ftype,patches,fit_method='polyfit'):
            # perform the fitting in patches?
            # npt = number of patches
            if patches==True:
                npt = 8
            else:
                npt = 1
            # patch size in pixels
            ps = 4096/npt
            
            
            numlines = len(lines_in_order.id)
            # extract fitted line positions and errors
            pix     = lines_in_order['pars'].sel(par='cen',ft=ftype)#.dropna('id','all')
            
            ws     = np.zeros(self.npix)
            # coefficients and residuals
            cf = np.zeros(shape=(npt,polyord+1))
            # new xr.DataArray
            dims = ['id','par']
            pars = ['rsd','lbd','lbd_err']
            da = xr.DataArray(np.full((numlines,len(pars)),np.nan),
                              coords = [lines_in_order.id,
                                        pars],
                              dims=dims)
            
            # do fit for each patch
            for i in range(npt):
                # lower and upper limit in pixel for the patch
                ll,ul     = np.array([i*ps,(i+1)*ps],dtype=np.int)
                pixels    = np.arange(ll,ul,1,dtype=np.int)

                # select lines in this pixel range
                patch     = lines_in_order.where((pix>=ll)&
                                     (pix<ul)).dropna('id','all')
                patch_id  = patch.coords['id']
                # polynomial order must be lower than the number of points
                # used for fitting
                if fit_method != 'spline':
                    if patch_id.size>polyord:
                        
                        coef,coef_err = patch_fit(patch,polyord)
                        if coef is not None:
                            # centers [pix]
                            centers = patch['pars'].sel(par='cen',ft=ftype)
                            center_error = patch['pars'].sel(par='cen_err',ft=ftype)
                            # calculate wavelength according to the fit
                            lbd_c  = np.polyval(coef,centers)
                            # calculate the residual to the known wavelength [m/s] 
                            freq2lbd = 299792458e0/patch['attr'].sel(att='freq')*1e10
                            resid    = (freq2lbd.values-lbd_c)/freq2lbd.values*299792458e0
                            # calculate wavelength error
                            icoef = coef[::-1]
                            lbd_e = np.sum([(j+1)*icoef[j+1]*centers**(j) for j in range(np.shape(coef)[0]-1)],axis=0) * \
                                    center_error
                            # save data
                            da.loc[dict(id=patch_id,par='rsd')]     = resid
                            da.loc[dict(id=patch_id,par='lbd')]     = lbd_c
                            da.loc[dict(id=patch_id,par='lbd_err')] = lbd_e
                            
                            cf[i,:]=coef[::-1]
                    else:
                        ws[ll:ul] = np.nan
                    try:
                        ws[ll:ul] = np.polyval(coef,pixels)
                    except:
                        ws[ll:ul] = np.nan
                else:
                    splrep = patch_fit_spline(patch,ftype)
                    
                    fit_lbd   = interpolate.splev(patch['pars'].sel(par='cen',ft=ftype),splrep)
                    freq2lbd = 299792458e0/patch['attr'].sel(att='freq')*1e10
                    resid     = (freq2lbd.values-fit_lbd)/freq2lbd.values*299792458e0
                    da.loc[dict(id=patch_id,par='rsd')] = np.array(resid,dtype=np.float64)
                    da.loc[dict(id=patch_id,par='lbd')] = np.array(fit_lbd,dtype=np.float64)
                    ws[ll:ul] = interpolate.splev(pixels,splrep)
                    
            #fit = np.polyval(coef,pix)
#            print(rs,fit)
            # residuals are in m/s
            #rs = rs/fit*299792458
            return ws,cf,da

            
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
                        self.tharcalib_flag = True
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
                wavecoeff_vac,covariance = curve_fit(hf.polynomial, 
                                                     np.arange(self.npix), 
                                                     self.wavesol_thar.sel(od=order), 
                                                     p0=wavecoeff_air)
                wavecoeff[order]         = wavecoeff_vac
            return wavecoeff
            
            
        self.__check_and_load__()
        polyord = polyord if polyord is not None else self.polyord

        # wavesol(72,4096) contains wavelengths for 72 orders and 4096 pixels
        
        if type(fittype)==list:
            pass
        elif type(fittype)==str and fittype in ['epsf','gauss']:
            fittype = [fittype]
        else:
            fittype = ['epsf','gauss']
        if calibrator is "ThAr": 
            # If this routine has not been run previously, read the calibration
            # coefficients from the FITS file. For each order, derive the 
            # calibration in air. If 'vacuum' flag is true, convert the 
            # wavelengths to vacuum wavelengths for each order. 
            # Finally, derive the coefficients for the vacuum solution.

            # If wavesol_thar has not been initialised:
            
            
            
            if (self.wavesol_thar is None or self.wavesol_thar.sum()==0):
                wavesol_thar = xr.DataArray(np.full((self.nbo,self.npix),np.nan),
                                       coords = [np.arange(self.nbo),
                                                 np.arange(self.npix)],
                                       dims = ['od','pix'])
            else:
                wavesol_thar = self.wavesol_thar
            ws_thar_exists_all = np.all(~np.isnan(wavesol_thar.sel(od=orders)))
            if ws_thar_exists_all == False:
                self.wavecoeff_air = _get_wavecoeff_air()
                for order in orders:
                    if self.is_bad_order(order)==False:
                        wavesol_air = np.array(
                                        [np.sum(self.wavecoeff_air[order,i]*pix**i 
                                                for i in range(self.d+1)) 
                                        for pix in range(0,self.npix,1)])
                        if vacuum is True:
                            wavesol_thar.loc[dict(od=order)] = _to_vacuum(wavesol_air)
                            
                        else:
                            wavesol_thar.loc[dict(od=order)] = wavesol_air
                    else:
                        wavesol_thar.loc[dict(od=order)] = np.zeros(self.npix)
                self.wavesol_thar = wavesol_thar
                if vacuum is True:
                    self.wavecoeff_vacuum = _get_wavecoeff_vacuum()
                
            # If this routine has been run previously, check 
            self.wavesol_thar = wavesol_thar
            return wavesol_thar
                


        if calibrator == "LFC":
            #print(orders)           
            # Calibration for each order is performed in two steps:
            #   (1) Fitting LFC lines in both pixel and wavelength space
            #   (2) Dividing the 4096 pixel range into 8x512 pixel patches and
            #       fitting a 3rd order polynomial to the positions of the 
            #       peaks
            
            # Save positions of lines
            #if method == 'epsf':
            for ftype in fittype:
                lines = self.fit_lines(orders,fittype=ftype)
            
            # Check if a ThAr calibration is attached to the Spectrum.
            # Priority given to ThAr calibration provided directly to the 
            # function. If none given, see if one is already attached to the 
            # Spectrum. If also none, run __get_wavesol__('ThAr')
            
#            kwarg_wavesol_thar = kwargs.get('wavesol_thar',None)
#            if kwarg_wavesol_thar is not None:
#                # if ThAr calibration is provided, use it
#                self.wavesol_thar = kwarg_wavesol_thar
#                self.wavecoef_air = kwargs.pop('wavecoeff_air',_get_wavecoeff_air())
#                
##                try:
##                    self.wavecoeff_air = kwargs['wavecoeff_air']
##                except:
##                    self.wavecoeff_air = _get_wavecoeff_air()
#                self.wavecoeff_vacuum = _get_wavecoeff_vacuum()
            if self.wavesol_thar is not None:
                # if ThAr calibration is attached to the spectrum, pass
                pass
            else:
                # if ThAr calibration is not provided nor attached, retrieve it
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
            wavesol_LFC = xr.DataArray(np.full((2,self.nbo,self.npix),np.nan),
                                       coords = [['epsf','gauss'],
                                                 np.arange(self.nbo),
                                                 np.arange(self.npix)],
                                       dims = ['ft','od','pix'],
                                       name='wavesol')
            for ftype in fittype:
#                wavesol_LFC  = np.zeros(shape = (self.nbo,self.npix,), 
#                                    dtype = np.float64)
                # Save coeffiecients of the best fit solution
                if patches==True:
                    npt = 8
                elif patches==False:
                    npt = 1
                wavecoef_LFC = xr.DataArray(np.full((2,self.nbo,npt,polyord+1),np.nan),
                                       coords = [['epsf','gauss'],
                                                 np.arange(self.nbo),
                                                 np.arange(npt),
                                                 np.arange(polyord+1)],
                                       dims = ['ft','od','patch','pod'],
                                       name='coef')
#                if npt == 1:
#                    wavecoef_LFC = np.zeros(shape = (self.nbo,polyord+1,npt), 
#                                            dtype = np.float64)
#                else:
#                    wavecoef_LFC = np.zeros(shape = (self.nbo,polyord+1,npt), 
#                                            dtype = np.float64)
                    
                
                progress = tqdm.tqdm(total=len(orders),
                                    desc="Fit wavesol {0:>5s}".format(ftype))
                for order in orders:
                    # Check if the order is listed among bad orders. 
                    # If everything is fine, fit the lines of the comb in both 
                    # wavelength and pixel space. Every 512th pixel is larger than 
                    # the previous 511 pixels. Therefore, divide the entire 4096
                    # pixel range into 8 512 pixel wide chunks and fit a separate 
                    # wavelength solution to each chunk.
    #                print("ORDER = {}".format(order))
                    LFC_wavesol_singleorder = np.zeros(self.npix)
                    if self.is_bad_order(order):
                        wavesol_LFC.loc[dict(od=order,ft=ftype)] = np.zeros(self.npix)
                        continue
                    else:
                        pass
                    #print("ORDER = ",order)
    #                if method=='epsf':
    #                    lines_in_order = lines['pars'].sel(od=order,ft='epsf').dropna('id','all')#     = self.fit_lines(order,scale='pixel',method=method)
    #                elif method=='gauss':
    #                    lines_in_order = lines['gauss'].sel(od=order).dropna('id','all')
                    lines_in_order = lines.sel(od=order).dropna('id','all')
                    
                    
                    # Include the gaps
                    if gaps is True:
                        g0 = self.gaps[order,:]
                        old_cen = lines_in_order['pars'].sel(par='cen',ft=fittype)
                        new_cen = self.introduce_gaps(lines_in_order['pars'].sel(par='cen',ft=fittype),g0)
                        print(old_cen-new_cen)
                        lines_in_order['pars'].loc[dict(par='cen',ft=fittype)] = new_cen
                    elif gaps is False:
                        pass
                    
                    LFC_ws,coef,da = fit_wavesol(
                                                   lines_in_order,
                                                   ftype=ftype,
                                                   patches=patches
                                                   )
                    
                    wavesol_LFC.loc[dict(ft=ftype,od=order)]  = LFC_ws
                    wavecoef_LFC.loc[dict(ft=ftype,od=order)] = coef  
                    ids                 = da.coords['id']
                    #if method=='epsf':
                    lines['wave'].loc[dict(od=order,id=ids,ft=ftype,wav='val')] = da.sel(par='lbd')
                    lines['wave'].loc[dict(od=order,id=ids,ft=ftype,wav='err')] = da.sel(par='lbd_err')
                    lines['wave'].loc[dict(od=order,id=ids,ft=ftype,wav='rsd')] = da.sel(par='rsd')
                    
                    #elif method =='gauss':
                    #    lines['gauss'].loc[dict(od=order,id=ids,par='lbd')] = lbds
                    #    lines['gauss'].loc[dict(od=order,id=ids,par='rsd')] = resids
                #wavesol_LFC_dict[ftype] = wavesol_LFC
                    progress.update(1)
                progress.close
            self.wavesol_LFC  = wavesol_LFC
            #self.lines        = cc_data
            self.wavecoef_LFC = wavecoef_LFC
            self.LFCws         = xr.merge([wavesol_LFC,wavecoef_LFC])
            #self.residuals    = rsd
        #self.wavesol = wavesol
        
            return wavesol_LFC
    def calc_lambda(self,ft='epsf',orders=None):
        ''' Returns wavelength and wavelength error for the lines using 
            polynomial coefficients in wavecoef_LFC.
            
            Adapted from HARPS mai_compute_drift.py'''
        if orders is not None:
            orders = orders
        else:
            orders = np.arange(self.sOrder,self.nbo,1)
        lines = self.check_and_return_lines()
        ws    = self.check_and_get_wavesol()
        wc    = self.wavecoef_LFC
        
        x     = lines['pars'].sel(par='cen',od=orders,ft=ft).values
        x_err = lines['pars'].sel(par='cen_err',od=orders,ft=ft).values
        c     = wc.sel(patch=0,od=orders,ft=ft).values
        # wavelength of lines
        wave  = np.sum([c[:,i]*(x.T**i) for i in range(c.shape[1])],axis=0).T
        # wavelength errors
        dwave = np.sum([(i+1)*c[:,i+1]*(x.T**(i+1)) \
                        for i in range(c.shape[1]-1)],axis=0).T * x_err
        return wave,dwave
    def calculate_fourier_transform(self,**kwargs):
        try:    orders = kwargs["order"]
        except: orders = np.arange(self.sOrder,self.nbo,1)
        n       = (2**2)*4096
        freq    = np.fft.rfftfreq(n=n, d=1)
        uppix   = 1./freq
        # we only want to use periods lower that 4096 pixels 
        # (as there's no sense to use more)
        cut     = np.where(uppix<=self.npix)
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
            minima  = hf.peakdet(yarray,xarray,extreme='min')
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
    def get_e2ds(self,order=None):
        '''
        Returns an xarray DataArray object with: flux, background, flux error,
        wavelength, and photon noise contribution for each pixel.
        
        Args:
            order : int, list of int, None - orders of the DataArray object
        Returns:
            e2ds  : xarray DataArray
        '''
        orders = self.prepare_orders(order)
        
        spec2d = self.extract2d()
        bkg2d  = self.get_background2d()
        err2d  = np.sqrt(np.abs(spec2d)+np.abs(bkg2d))
        #wave2d = xr.DataArray(wavesol_thar,coords=spec2d.coords)
        wave2d = self.check_and_get_wavesol('thar')
        
        # weights for photon noise calculation
        # Equation 5 in Murphy et al. 2007, MNRAS 380 p839
        pix2d   = np.vstack([np.arange(self.npix) for o in range(self.nbo)])
        df_dlbd = np.vstack([hf.derivative1d(spec2d[o],wave2d[o]) \
                                             for o in range(self.nbo)])
        sigma_v = 299792458e0*err2d/(wave2d*df_dlbd)
        e2ds = xr.concat([spec2d,bkg2d,err2d,wave2d,sigma_v],
                         pd.Index(['flx','bkg','err','wave','sigma_v'],
                                  name='ax'))
        e2ds.name = 'e2ds'
        
        # truncate data below sOrder:
        e2ds = e2ds[:,sOrder:self.nbo,:]
        self.e2ds = e2ds
        return e2ds
    
    def detect_lines(self,order=None,calculate_weights=False):
        '''
        This method determines the minima of flux between the LFC lines
        and updates self.lines with the position, flux, background, flux error
        and barycenter of each line.
        '''
        
        # MAIN PART
        orders = self.prepare_orders(order)
#        self.check_and_load_psf()
        wavesol_thar = self.check_and_get_wavesol('thar')
        if self.lineDetectionPerformed==True:    
            lines  = self.lines
            return lines
        else:
            #print('init lines')
            lines = self.check_and_return_lines()
            lines = self.lines
        
        e2ds = self.get_e2ds(orders)

       
        pbar =tqdm.tqdm(total=1,desc="Detecting lines")
        start = time.time()
#        #outdata = mp_pool.map(detect_order,[e2ds.sel(od=od) for od in orders])
##        outdata = pool.uimap(detect_order,[(e2ds.sel(od=od),self.f0_comb,self.reprate,self.segsize) for od in orders])
#        outdata = Parallel(n_jobs=hs.nproc)(delayed(detect_order)(e2ds.sel(od=od),self.f0_comb,self.reprate,self.segsize,self.pixPerLine) for od in orders)
        pool1 = Pool(hs.nproc)
        outdata = pool1.map(wrap_detect_order,
                            [(e2ds.sel(od=od),self.f0_comb,self.reprate,
                              self.segsize,self.pixPerLine,self.window) \
                              for od in orders])
        pool1.close()
        pool1.join()
        end  = time.time()
        pbar.update(1)
        pbar.close()
        detected_lines = xr.merge(outdata)
       
        lines['attr'] = detected_lines['attr']
        lines['line'] = detected_lines['line']
        lines['stat'] = detected_lines['stat']
        if calculate_weights:
            psf = self.check_and_load_psf()
            pool2 = Pool(hs.nproc)
            weights = pool2.map(wrap_calculate_line_weights,[(lines.sel(od=od),self.psf,self.pixPerLine) for od in orders])
            print('Weights calculated')
            weights = xr.merge(weights)
            pool2.close()
            pool2.join()
            lines['line'].loc[dict(od=orders,ax='wgt')] = weights['line'].sel(od=orders)
        else:
            pass
        self.lines = lines
        
        self.lineDetectionPerformed=True
        gc.collect()
        return lines
    
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
                if (self.wavesol_thar is None or np.sum(self.wavesol_thar.sel(od=order))==0):
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
            error1d = pd.Series(np.sqrt(np.abs(self.data[order])))
            include['error']=error1d
                
        if   scale == 'pixel':
            xarray1d = pix1d
        elif scale == 'wave':
            xarray1d = wave1d
        kind      = 'spline'
        minima    = hf.peakdet(flux1d, xarray1d, extreme="min")
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
    def extract2d(self):
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
#        if scale=="wave":
#            self.wavesol = self.__get_wavesol__(calibrator="ThAr")
#        else:
#            pass
#        if   scale=='wave':
#            wave2d  = self.wavesol
#            flux2d  = self.data
#            #spec1d = np.stack([wave1d,flux1d])
#            spec2d  = dict(wave=wave2d, flux=flux2d)
#        elif scale=='pixel':
#            
        spec2d  = xr.DataArray(self.data,
                               coords=[np.arange(self.nbo),
                                       np.arange(self.npix)],
                               dims=['od','pix'])
        return spec2d
    def fit_single_line(self,line,psf=None):
        def residuals(x0,pixels,counts,weights,background,splr):
            ''' Model parameters are estimated shift of the line center from 
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
            # sigma_tot^2 = sigma_counts^2 + sigma_background^2
            # sigma_counts = sqrt(counts)     sigma_background = sqrt(background)
            error = np.sqrt(counts + background)
            resid = np.sqrt(weights) * ((counts-background) - model) / error
#            resid = counts/np.sum(counts) * ((counts-background) - model) / error
            #resid = line_w * (counts- model)
            return resid
        def get_local_psf(pix,order,seg,mixing=True):
            ''' Returns local ePSF at a given pixel of the echelle order
            '''
            segments        = np.unique(psf.coords['seg'].values)
            N_seg           = len(segments)
            # segment limits
            sl              = np.linspace(0,4096,N_seg+1)
            # segment centers
            sc              = (sl[1:]+sl[:-1])/2
            sc[0] = 0
            sc[-1] = 4096
           
            def return_closest_segments(pix):
                sg_right  = int(np.digitize(pix,sc))
                sg_left   = sg_right-1
                return sg_left,sg_right
            
            sgl, sgr = return_closest_segments(pix)
            f1 = (sc[sgr]-pix)/(sc[sgr]-sc[sgl])
            f2 = (pix-sc[sgl])/(sc[sgr]-sc[sgl])
            
            #epsf_x  = psf.sel(ax='x',od=order,seg=seg).dropna('pix')+pix
            
            epsf_1 = psf.sel(ax='y',od=order,seg=sgl).dropna('pix')
            epsf_2 = psf.sel(ax='y',od=order,seg=sgr).dropna('pix')
            
            if mixing == True:
                epsf_y = f1*epsf_1 + f2*epsf_2 
            else:
                epsf_y = epsf_1
            
            xc     = epsf_y.coords['pix']
            epsf_x  = psf.sel(ax='x',od=order,seg=seg,pix=xc)+pix
            #qprint(epsf_x,epsf_y)
            return epsf_x, epsf_y
        # MAIN PART 
        
        if psf is None:
            self.check_and_load_psf()
            psf=self.psf
        else:
            pass
        # select single line
        #lid       = line_id
        line      = line.dropna('pid','all')
        pid       = line.coords['pid']
        lid       = int(line.coords['id'])
    
        line_x    = line['line'].sel(ax='pix')
        line_y    = line['line'].sel(ax='flx')
        line_w    = line['line'].sel(ax='wgt')
        #print("Read the data for line {}".format(lid))
        # fitting fails if some weights are NaN. To avoid this:
        weightIsNaN = np.any(np.isnan(line_w))
        if weightIsNaN:
            whereNaN  = np.isnan(line_w)
            line_w[whereNaN] = 0e0
            #print('Corrected weights')
        line_bkg  = line['line'].sel(ax='bkg')
        line_bary = line['attr'].sel(att='bary')
        cen_pix   = line_x[np.argmax(line_y)]
        #freq      = line['attr'].sel(att='freq')
        #print('Attributes ok')
        #lbd       = line['attr'].sel(att='lbd')
        # get local PSF and the spline representation of it
        order        = int(line.coords['od'])
        loc_seg      = line['attr'].sel(att='seg')
        psf_x, psf_y = self.get_local_psf(line_bary,order=order,seg=loc_seg)
        
        psf_rep  = interpolate.splrep(psf_x,psf_y)
        #print('Local PSF interpolated')
        # fit the line for flux and position
        #arr    = hf.return_empty_dataset(order,pixPerLine)
        try: pixPerLine = self.pixPerLine
        except: 
            self.__read_LFC_keywords__()
            pixPerLine = self.pixPerLine
        par_arr     = hf.return_empty_dataarray('pars',order,pixPerLine)
        mod_arr     = hf.return_empty_dataarray('model',order,pixPerLine)
        p0 = (5e-1,np.percentile(line_y,90))

        
        # GAUSSIAN ESTIMATE
        g0 = (np.nanpercentile(line_y,90),float(line_bary),1.3)
        gausp,gauscov=curve_fit(hf.gauss3p,p0=g0,
                            xdata=line_x,ydata=line_y)
        Amp, mu, sigma = gausp
        p0 = (0.01,Amp)
        popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                                args=(line_x,line_y,line_w,line_bkg,psf_rep),
                                full_output=True,
                                ftol=1e-5)
        
        if ier not in [1, 2, 3, 4]:
            print("Optimal parameters not found: " + errmsg)
            popt = np.full_like(p0,np.nan)
            pcov = None
            success = False
        else:
            success = True
       
        if success:
            
            sft, flux = popt
            line_model = flux * interpolate.splev(line_x+sft,psf_rep) + line_bkg
            cost   = np.sum(infodict['fvec']**2)
            dof    = (len(line_x) - len(popt))
            rchisq = cost/dof
            if pcov is not None:
                pcov = pcov*rchisq
            else:
                pcov = np.array([[np.inf,0],[0,np.inf]])
            cen              = cen_pix+sft
            cen              = line_bary - sft
            cen_err, flx_err = [np.sqrt(pcov[i][i]) for i in range(2)]
  
            #pars = np.array([cen,sft,cen_err,flux,flx_err,rchisq,np.nan,np.nan])
            pars = np.array([cen,cen_err,flx,flx_err,sigma,sigma_err,rchi2])
        else:
            pars = np.full(len(hf.fitPars),np.nan)
       
        par_arr.loc[dict(od=order,id=lid,ft='epsf')] = pars
        mod_arr.loc[dict(od=order,id=lid,
                         pid=line_model.coords['pid'],ft='epsf')] = line_model

        return par_arr,mod_arr
#        
    def fit_lines(self,order=None,fittype='epsf',nobackground=True,model=None,
                  remove_poor_fits=False,verbose=0,njobs=hs.nproc):
        ''' Calls one of the specialised line fitting routines.
        '''
        # Was the fitting already performed?
        if self.lineFittingPerformed[fittype] == True:
            return self.lines
        
        # Prepare orders
        orders = self.prepare_orders(order)
        
        # Select method
        if fittype == 'epsf':
            self.check_and_load_psf()
            function = wrap_fit_epsf
#            function = wrap_fit_single_line
        elif fittype == 'gauss':
            function = hf.wrap_fit_peak_gauss
        
        # Check if the lines were detected, run 'detect_lines' if not
        self.check_and_return_lines()
        if self.lineDetectionPerformed==True:
            detected_lines = self.lines
            if detected_lines is None:
                if fittype == 'epsf':
                    cw=True
                elif fittype == 'gauss':
                    cw=False
                detected_lines = self.detect_lines(order,calculate_weights=cw)
            else:
                pass
        else:
            if fittype == 'epsf':
                cw=True
            elif fittype == 'gauss':
                cw=False
            detected_lines = self.detect_lines(order,calculate_weights=cw)
        lines = detected_lines
        orders    = self.prepare_orders(order)
        linesID   = self.lines.coords['id']
        
        list_of_order_linefits = []
        list_of_order_models = []
        
        progress = tqdm.tqdm(total=len(orders),
                             desc='Fitting lines {0:>5s}'.format(fittype))
        
        start = time.time()
#        mp_pool = ProcessPool()
#        mp_pool.nproc      = 1
        pool3 = Pool(hs.nproc)
        for order in orders:
            
            progress.update(1)
            order_data = detected_lines.sel(od=order).dropna('id','all')
            lines_in_order = order_data.coords['id']
            numlines       = np.size(lines_in_order)
            
            if fittype == 'epsf':
#                output = Parallel(n_jobs=njobs)(delayed(function)(order_data,order,lid,self.psf,self.pixPerLine) for lid in range(numlines))
#                results = mp_pool.map(self.fit_single_line,[order_data.sel(id=lid) for lid in range(numlines)])
#                output = pool.map(function,[(order_data,order,lid,self.psf,self.pixPerLine) for lid in range(numlines)])
#                results = Parallel(n_jobs=hs.nproc)(delayed(function)(order_data.sel(id=lid),self.psf,self.pixPerLine) for lid in range(numlines))
                results = pool3.map(function,
                                    [(order_data.sel(id=lid),self.psf,self.pixPerLine) for lid in range(numlines)])
                time.sleep(1)
            elif fittype == 'gauss':
#                results = Parallel(n_jobs=hs.nproc)(delayed(function)(order_data,order,i,'erfc','singlegaussian',self.pixPerLine,0) for i in range(numlines))
                results = pool3.map(function,[(order_data,order,i,'erfc','singlesimplegaussian',self.pixPerLine,0) for i in range(numlines)])
            
            parameters,models = zip(*results)
            order_fit = xr.merge(parameters)
            order_models = xr.merge(models)
            list_of_order_linefits.append(order_fit['pars'])
            list_of_order_models.append(order_models['model'])
            gc.collect()
            #order_linedata = order_fit['line']
            #list_of_order_linedata.append(order_linedata)
#            mp_pool.close()
#            mp_pool.restart()
        pool3.close()
        pool3.join()
        progress.close()
        fits = xr.merge(list_of_order_linefits)
        models =xr.merge(list_of_order_models)
        #lines = xr.merge([fits,models])
        lines['pars'].loc[dict(ft=fittype,od=orders)]  = fits['pars'].sel(ft=fittype)
        lines['model'].loc[dict(ft=fittype,od=orders)] = models['model'].sel(ft=fittype)
        self.lines = lines
        self.lineDetectionPerformed = True
        self.lineFittingPerformed[fittype]=True
#        pool.close()
#        del(pool)
        return lines
        
    def fit_lines_gaussian1d(self,order,nobackground=True,method='erfc',model=None,
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
            ind1  = np.where((hf.is_outlier(sigma)==True))[0]
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
        maxima  = hf.peakdet(spec1d.flux,spec1d.wave,extreme='max')
        minima  = hf.peakdet(spec1d.flux,spec1d.pixel,extreme='min')
        xpeak   = maxima.x
        nu_min  = 299792458e0/(xpeak.iloc[-1]*1e-10)
        nu_max  = 299792458e0/(xpeak.iloc[0]*1e-10)
#        print(nu_min,nu_max)
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
            wavecoeff,pcov   = curve_fit(hf.polynomial3,np.arange(self.npix),
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
        maxima_p = hf.peakdet(spec1d.flux,spec1d.pixel,extreme='max')
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
        #print(nminima,nmaxima)
        dxi   = 11.
        dx         = xarray.diff(1).fillna(dxi)
        if verbose>2:
            print('Fitting {}'.format(scale))
        
        # model
        model = model if model is not None else 'singlegaussian'
        results = Parallel(n_jobs=hs.nproc)(delayed(hf.fit_peak)(i,xarray,yarray,yerror,weights,xmin,xmax,dx,method,model) for i in range(nminima))
        results = np.array(results)
      
        parameters = results['pars'].squeeze(axis=1)
        errors     = results['errors'].squeeze(axis=1)
        photon_nse = results['pn'].squeeze(axis=1)
        center     = results['cen'].squeeze(axis=1)
        center_err = results['cen_err'].squeeze(axis=1)
        rsquared   = results['r2'].squeeze(axis=1)
        #N = results.shape[0]
        #M = parameters.shape[1]
        
        
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
    def fit_lines_gaussian2d(self,order=None,nobackground=True,method='erfc',
                  model=None,scale='pixel',remove_poor_fits=False,verbose=0):
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
        # Have lines been fitted already?
        self.check_and_return_lines()
        if self.lineDetectionPerformed==True:
            detected_lines = self.lines
            if detected_lines is None:
                detected_lines = self.detect_lines(order)
            else:
                pass
        else:
            detected_lines = self.detect_lines(order)        
        orders    = self.prepare_orders(order)
        linesID   = self.lines.coords['id']
        
        # contains a list of DataArrays, each one containing line fit params
        list_of_order_fits = []
        for order in orders:
            order_data = detected_lines.sel(od=order).dropna('id','all')
            lines_in_order = order_data.coords['id']
            numlines       = np.size(lines_in_order)
            if verbose>1:
                print("Npeaks:{0:<5}".format(numlines))
        
        

            model = model if model is not None else 'singlegaussian'
            output = Parallel(n_jobs=hs.nproc)(delayed(hf.fit_peak_gauss)(order_data,order,i,method,model) for i in range(numlines))
            # output is a list of xr.DataArrays containing line fit params
            # for this order
            order_fit = xr.merge(output)
            list_of_order_fits.append(order_fit)

        fits = xr.merge(list_of_order_fits)
        #fits.rename({'pars':'gauss'})
        lines_gaussian = xr.merge([detected_lines,fits])
        #self.lines_gaussian = lines_gaussian
        return lines_gaussian
    def fit_lines1d(self,order,nobackground=False,method='epsf',model=None,
                  scale='pixel',vacuum=True,remove_poor_fits=False,verbose=0):
        # load PSF and detect lines
        self.check_and_load_psf()
        self.check_and_return_lines()
        
        #sc        = self.segment_centers
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
            resid = np.sqrt(line_w) * ((counts-background) - model)/np.sqrt(np.abs(counts))
            #resid = line_w * (counts- model)
            return resid
        
        # Determine which scales to use
        scale = ['wave','pixel'] if scale is None else [scale]
        if verbose>0:
            print("ORDER:{0:<5d} Bkground:{1:<5b} Method:{2:<5s}".format(order,
                  not nobackground, method))
        # Prepare orders
        orders = self.prepare_orders(order)
        #lines = self.check_and_return_lines()
       
            
        # Cut the lines
        
        pixel, flux, error, bkgr, bary = self.cut_lines(orders, nobackground=nobackground,
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
            local_seg = cen_pix//segsize
            psf_x, psf_y = self.get_local_psf(cen_pix,order=order,seg=local_seg)
            
            line_w = get_line_weights(line_x,cen_pix)
            psf_rep  = interpolate.splrep(psf_x,psf_y)
            p0 = (0,np.max(line_y))
            
            popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                                    args=(line_x,line_y,line_w,line_b,psf_rep),
                                    full_output=True)
            
            if ier not in [1, 2, 3, 4]:
                print("Optimal parameters not found: " + errmsg)
                popt = np.full_like(p0,np.nan)
                pcov = None
                success = False
            else:
                success = True
            if success:
                
                sft, flx = popt
                cost   = np.sum(infodict['fvec']**2)
                dof    = (len(line_x) - len(popt))
                rchisq = cost/dof
                if pcov is not None:
                    pcov = pcov*rchisq
                else:
                    pcov = np.array([[np.inf,0],[0,np.inf]])
                cen              = line_x[np.argmax(line_y)]-sft
                cen_err, flx_err = [np.sqrt(pcov[i][i]) for i in range(2)]
                phi              = cen - int(cen+0.5)
                b                = bary[n]
                pars = np.array([cen,cen_err,flx,flx_err, sft,phi,b,rchisq])
                model = flx * interpolate.splev(pixels+sft,psf_rep) 
            else:
                pars = np.full(8,np.nan)
                model = np.full_like(pixels,np.nan)
        #lines.loc[dict(id=n)]
        lines.loc[dict(id=n)] = pars
        return lines
    def fit_lines2d(self,order=None):
        self.check_and_load_psf()
        self.check_and_return_lines()
        if self.lineDetectionPerformed==True:
            detected_lines = self.lines
            if detected_lines is None:
                detected_lines = self.detect_lines(order)
            else:
                pass
        else:
            detected_lines = self.detect_lines(order)
        orders    = self.prepare_orders(order)
        linesID   = self.lines.coords['id']
        
        list_of_order_fits = []
        for order in orders:
            order_data = detected_lines.sel(od=order).dropna('id','all')
            lines_in_order = order_data.coords['id']
            numlines       = np.size(lines_in_order)
            output = Parallel(n_jobs=hs.nproc)(delayed(fit_epsf)(order_data,order,lid,self.psf) for lid in range(numlines))
#            print(order,np.shape(output))
#            array = np.array(output)
            order_fit = xr.merge(output)
            list_of_order_fits.append(order_fit)
        fits = xr.merge(list_of_order_fits)
        lines = xr.merge([detected_lines,fits])
        self.lines = lines
        self.lineDetectionPerformed = True
        return lines

        
        
    def get_average_profile(self,order,nobackground=True):
        # Extract data from the fits file
        spec1d  = self.extract1d(order,nobackground=nobackground,vacuum=True)
        
        pn,weights  = self.calculate_photon_noise(order,return_array=True)
        #weights     = self.get_weights1d(order)
        # Define limits in wavelength and theoretical wavelengths of lines
        maxima      = hf.peakdet(spec1d.flux,spec1d.wave,extreme='max')
        minima      = hf.peakdet(spec1d.flux,spec1d.pixel,extreme='min')
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
            #line_x = xarray[cut].values
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
            xarray = np.arange(self.npix)
        elif scale == "wave":
            xarray = spec1d.wave
        #print(xarray)
        yarray          = self.data[order]
        minima          = hf.peakdet(yarray, xarray, extreme="min",
                                     delta=np.percentile(yarray,5))
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
    def get_background2d(self,orders=None,kind='linear'):
        orders = self.prepare_orders(orders)
        spec2d = self.extract2d()
        bkg2d  = spec2d.copy()
        pixels = spec2d.coords['pix']
        for order in orders:
            flux            = spec2d.sel(od=order)
            minima          = hf.peakdet(flux, pixels, extreme="min")
            xbkg,ybkg       = minima.x, minima.y
            if   kind == "spline":
                coeff       = interpolate.splrep(xbkg, ybkg)
                background  = interpolate.splev(pixels,coeff) 
            elif kind == "linear":
                coeff      = interpolate.interp1d(xbkg,ybkg)
                valid      = pixels.clip(min(xbkg),max(xbkg))
                background = coeff(valid)
            bkg2d.loc[dict(od=order)]=background
        self.background = bkg2d
        return bkg2d
        
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
        maxima      = hf.peakdet(spec1d["flux"], spec1d[scale], extreme="max")
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
    def get_envelope2d(self,orders=None,kind='linear'):
        orders = self.prepare_orders(orders)
        spec2d = self.extract2d()
        env2d  = spec2d.copy()
        pixels = spec2d.coords['pix']
        for order in orders:
            flux            = spec2d.sel(od=order)
            maxima          = hf.peakdet(flux, pixels, extreme="max")
            xenv,yenv       = maxima.x, maxima.y
            if   kind == "spline":
                coeff       = interpolate.splrep(xenv, yenv)
                background  = interpolate.splev(pixels,coeff) 
            elif kind == "linear":
                coeff      = interpolate.interp1d(xenv,yenv)
                valid      = pixels.clip(min(xenv),max(xenv))
                background = coeff(valid)
            env2d.loc[dict(od=order)]=background
        self.envelope = env2d
        return env2d
    def get_extremes(self, order, scale="pixel", extreme="max"):
        '''Function to determine the envelope of the observations by fitting a cubic spline to the maxima of LFC lines'''
        spec1d      = self.extract1d(order=order,columns=[scale,'flux'])
        extremes    = hf.peakdet(spec1d["flux"], spec1d[scale], extreme=extreme)
        return extremes
    def get_distortions(self,order=None,calibrator='LFC',ft='epsf'):
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
            freq0 = data['attr'].sel(att='freq',od=order)#.dropna('val')
            wav0  = 299792458*1e10/freq0
            pix0  = data['pars'].sel(par='cen',od=order)#.dropna('val')
            if calibrator == 'ThAr':
                coeff = self.wavecoeff_vacuum[order]
            elif calibrator == 'LFC':
                coeff = self.wavecoef_LFC.sel(od=order,ft=ft)[::-1]
            wav1 = hf.polynomial(pix0,*coeff)
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
        del(lines)
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
                    wavecoeff_vac,covariance = curve_fit(hf.polynomial, 
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
        df_dl         = hf.derivative1d(spec1d['flux'].values,wavesol[order])
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
                df_dl[order] = hf.derivative1d(spec2d[order],wavesol[order])                    
        zeros          = np.where(spec2d==0)
        spec2d[zeros]  = np.inf                    
        weights2d      = (wavesol**2 * df_dl**2) / np.abs(spec2d)
        cut            = np.where(weights2d == 0.)
        weights2d[cut] = np.inf
        self.weights2d = np.zeros(shape=self.data.shape)
        self.weights2d[orders,:] = weights2d
        
        return self.weights2d
    def introduce_gaps(self,x,gaps):
        xc = np.copy(x)
        if np.size(gaps)==1:
            gap  = gaps
            gaps = np.full((7,),gap)
        for i,gap in enumerate(gaps):
            ll = (i+1)*self.npix/(np.size(gaps)+1)
            cut = np.where(x>ll)[0]
            xc[cut] = xc[cut]-gap
        return xc
    def is_bad_order(self,order):
        if order in self.bad_orders: 
            return True
        else:
            return False
    
    def load_lines(self,dirname=None,replace=True):
        dirname = dirname if dirname is not None else hs.harps_lines
        direxists = os.path.isdir(dirname)
        if not direxists:
            raise ValueError("Directory does not exist")
        else:
            pass
        basename = os.path.basename(self.filepath)[:-5]
        path     = os.path.join(dirname,basename+'_lines.nc')
        
        try:
            lines    = xr.open_dataset(path)
            print('Lines loaded from: {}'.format(path))
        except:
            return None
        if replace == True:
            self.lines = lines
            self.lineDetectionPerformed=True
            self.lineFittingPerformed['gauss']=True
            self.lineFittingPerformed['epsf'] =True
        else:
            pass
        return lines
    def load_psf(self,filepath=None,fibre_shape=None):
        if fibre_shape is None:
            fibre_shape = self.fibre_shape
        else:
            fibre_shape = 'octogonal'
        if filepath is not None:
            filepath = filepath
        else:
            if self.LFC == 'HARPS':
                filepath = os.path.join(hs.harps_psf,
                                    'fibre{}'.format(self.fibre),
                                    'harps{}_{}.nc'.format(self.fibre,fibre_shape))
            elif self.LFC == 'FOCES':
                filepath = os.path.join(hs.harps_psf,
                                    'fibre{}'.format(self.fibre),
                                    'foces{}_{}.nc'.format(self.fibre,'round'))
        
        data = xr.open_dataset(filepath)
        epsf = data['epsf'].sel(ax=['x','y'])
        self.psf = epsf
        return epsf
    def load_wavesol(self,dirname=None,replace=True):
        dirname = dirname if dirname is not None else hs.harps_ws
        direxists = os.path.isdir(dirname)
        if not direxists:
            raise ValueError("Directory does not exist")
        else:
            pass
        basename = os.path.basename(self.filepath)[:-5]
        path     = os.path.join(dirname,basename+'_LFCws.nc')
        
        try:
            LFCws = xr.open_dataset(path)
            wavesol_LFC = LFCws['wavesol']
            wavecoef_LFC = LFCws['coef']
            print('Wavesol loaded from: {}'.format(path))
        except:
            return None
        if replace == True:
            self.wavesol_LFC = wavesol_LFC
            self.wavecoef_LFC = wavecoef_LFC
            self.LFCws       = LFCws
        else:
            pass
        return LFCws
    def plot_spectrum(self,order=None,nobackground=False,scale='pixel',
             fit=False,fittype='epsf',confidence_intervals=False,legend=False,
             naxes=1,ratios=None,title=None,sep=0.05,
             figsize=(16,9),plotter=None,axnum=None,
             **kwargs):
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
            plotter = SpectrumPlotter(naxes,figsize=figsize,sep=sep,
                                      title=title,ratios=ratios,**kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        figure, axes = plotter.figure, plotter.axes
    
        
        orders = self.prepare_orders(order)
        if fit==True:
            self.check_and_get_comb_lines(orders=orders)
            lines = self.lines
            linesID = lines.coords['id'].values
            if type(fittype) == list:
                fittype = fittype
            elif type(fittype) == str:
                fittype = [fittype]
        for order in orders:
            spec1d = self.extract1d(order,nobackground=nobackground)
            x      = spec1d[scale]
            y      = spec1d.flux
            yerr   = spec1d.error
            
            axes[ai].errorbar(x,y,yerr=yerr,label='Data',capsize=3,capthick=0.3,
                ms=10,elinewidth=0.3,color='C0',zorder=100)
            if fit==True:   
                fittype = hf.to_list(fittype)
                for lid in linesID:
                    if scale == 'wave':
                        line_x = lines['line'].sel(od=order,id=lid,ax='wave')
                    elif scale == 'pixel':
                        line_x = lines['line'].sel(od=order,id=lid,ax='pix')
                    if len(line_x.dropna('pid','all')) == 0:
                        continue
                    else: pass
                    models = []
                    colors = []
                    if 'epsf' in fittype:
                        line_m = lines['model'].sel(od=order,id=lid,ft='epsf')
                        if nobackground:
                            bkg = lines['line'].sel(od=order,id=lid,ax='bkg')
                            line_m = line_m - bkg
                        models.append(line_m)
                        colors.append('C1')
                    if 'gauss' in fittype:
                        line_m = lines['model'].sel(od=order,id=lid,ft='gauss')
                        models.append(line_m)
                        colors.append('C2')
#                    print(line_x,line_m)
                    #print(models)
                    for model,col in zip(models,colors):
                        axes[ai].scatter(line_x,model,marker='X',s=10,color=col)
                #fit_lines = self.fit_lines(order,scale=scale,nobackground=nobackground)
                #self.axes[0].plot(x,double_gaussN_erf(x,fit_lines[scale]),label='Fit')
        axes[ai].set_xlabel('Pixel')
        axes[ai].set_ylabel('Flux [$e^-$]')
        m = hf.round_to_closest(np.max(y),hs.rexp)
        axes[ai].set_yticks(np.linspace(0,m,3))
        if legend:
            axes[ai].legend()
        figure.show()
        return plotter
    def plot_distortions(self,order=None,kind='lines',plotter=None,axnum=None,
                         fittype='epsf',show=True,**kwargs):
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
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        figure, axes = plotter.figure, plotter.axes
        axes[ai].set_ylabel('$\Delta x$=(ThAr - LFC) [m/s]')
        axes[ai].set_xlabel('Pixel')
        orders = self.prepare_orders(order)
        
        
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker = kwargs.get('marker','x')
        
        plotargs = {'ms':2,'marker':marker}
        for i,order in enumerate(orders):
            if kind == 'lines':
                data  = self.check_and_get_comb_lines('LFC',orders)
                freq  = data['attr'].sel(att='freq',od=order).dropna('id')
                wav   = 299792458*1e10/freq
                
                # alternatively, use interpolated value of lambda
                wav   = data['wave'].sel(wav='val',od=order,ft=fittype).dropna('id')
                pix   = data['pars'].sel(par='cen',od=order,ft=fittype).dropna('id')
                coeff = self.wavecoeff_vacuum[order][::-1]
                thar  = np.polyval(coeff,pix.values)
                plotargs['ls']=''
            elif kind == 'wavesol':
                wav   = self.wavesol_LFC.sel(ft=fittype,od=order)
                pix   = np.arange(self.npix)
                thar  = self.wavesol_thar[order]
                plotargs['ls']='-'
                plotargs['ms']=0
            rv  = (thar-wav)/wav * 299792458e0
            if len(orders)>5:
                plotargs['color']=colors[i]
            axes[ai].plot(pix,rv,**plotargs)
        [axes[ai].axvline(512*(i+1),lw=0.3,ls='--') for i in range (8)]
        if show == True: figure.show() 
        return plotter
    def plot_line(self,order,line_id,fittype='epsf',center=True,residuals=False,
                  plotter=None,axnum=None,title=None,figsize=(12,12),show=True,
                  **kwargs):
        ''' Plots the selected line and the models with corresponding residuals
        (optional).'''
        naxes = 1 if residuals is False else 2
        left  = 0.15 if residuals is False else 0.2
        ratios = None if residuals is False else [4,1]
        if plotter is None:
            plotter = SpectrumPlotter(naxes=naxes,title=title,figsize=figsize,
                                      ratios=ratios,sharex=False,
                                      left=left,bottom=0.18,**kwargs)
            
        else:
            pass
        ai = axnum if axnum is not None else 0
        figure, axes = plotter.figure, plotter.axes
        # Load line data
        lines  = self.check_and_return_lines()
        line   = lines.sel(od=order,id=line_id)
        models = lines['model'].sel(od=order,id=line_id)
        pix    = line['line'].sel(ax='pix')
        flx    = line['line'].sel(ax='flx')
        err    = line['line'].sel(ax='err')
        # save residuals for later use in setting limits on y axis if needed
        if residuals:
            resids = []
        # Plot measured line
        axes[ai].errorbar(pix,flx,yerr=err,ls='',color='C0',marker='o',zorder=0)
        axes[ai].bar(pix,flx,width=1,align='center',color='C0',alpha=0.3)
        axes[ai].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        # Plot models of the line
        if type(fittype)==list:
            pass
        elif type(fittype)==str and fittype in ['epsf','gauss']:
            fittype = [fittype]
        else:
            fittype = ['epsf','gauss']
        # handles and labels
        labels  = []
        for j,ft in enumerate(fittype):
            if ft == 'epsf':
                label = 'LSF'
                c   = 'C1'
                m   = 's'
            elif ft == 'gauss':
                label = 'Gauss'
                c   = 'C2'
                m   = '^'
            labels.append(label)
            axes[ai].plot(pix,models.sel(ft=ft),ls='-',color=c,marker=m,label=ft)
            if residuals:
                rsd        = (flx-models.sel(ft=ft))/err
                resids.append(rsd)
                axes[ai+1].scatter(pix,rsd,color=c,marker=m)
        # Plot centers
            if center:
                
                cen = line['pars'].sel(par='cen',ft=ft)
                axes[ai].axvline(cen,ls='--',c=c)
        # Makes plot beautiful
        
        axes[ai].set_ylabel('Flux\n[$e^-$]')
        rexp = hs.rexp
        m   = hf.round_to_closest(np.max(flx.dropna('pid').values),rexp)
#        axes[ai].set_yticks(np.linspace(0,m,3))
        hf.make_ticks_sparser(axes[ai],'y',3,0,m)
        # Handles and labels
        handles, oldlabels = axes[ai].get_legend_handles_labels()
        axes[ai].legend(handles,labels)
        if residuals:
            axes[ai+1].axhline(0,ls='--',lw=0.7)
            axes[ai+1].set_ylabel('Residuals\n[$\sigma$]')
            # make ylims symmetric
            lim = 1.2*np.nanpercentile(np.abs(resids),100)
            lim = np.max([5,lim])
            axes[ai+1].set_ylim(-lim,lim)
            axes[ai].set_xticklabels(axes[ai].get_xticklabels(),fontsize=1)
            axes[ai+1].set_xlabel('Pixel')
            axes[ai+1].axhspan(-3,3,alpha=0.3)
        else:
            axes[ai].set_xlabel('Pixel')
            

        
        if show == True: figure.show()
        return plotter
    def plot_linefit_residuals(self,order=None,hist=False,plotter=None,
                               axnum=None,show=True,lines=None,fittype='epsf',
                               **kwargs):
        ''' Plots the residuals of the line fits as either a function of 
            position on the CCD or a produces a histogram of values'''
        
        if hist == False:
            figsize = (12,9)
        else: 
            figsize = (9,9)
        if plotter is None:
            plotter=SpectrumPlotter(figsize=figsize,bottom=0.12,**kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        
        figure,axes = plotter.figure, plotter.axes
        
        orders = self.prepare_orders(order)
        if lines is None:
            lines = self.check_and_return_lines()
        else:
            lines = lines
        try:
            centers = lines['pars'].sel(od=orders,par='cen')
        except:
            centers = lines['gauss'].sel(od=orders,par='cen')
        pixel   = lines['line'].sel(od=orders,ax='pix')
        data    = lines['line'].sel(od=orders,ax='flx')
        model   = lines['model'].sel(od=orders,ft=fittype)
        fitresids = data - model
        if hist == True:
            bins = kwargs.get('bins',30)
            xrange = kwargs.get('range',None)
            log  = kwargs.get('log',False)
            label = kwargs.get('label',fittype)
            alpha = kwargs.get('alpha',1.)
            fitresids1d = np.ravel(fitresids)
            fitresids1d = fitresids1d[~np.isnan(fitresids1d)]
            axes[ai].hist(fitresids1d,bins=bins,range=xrange,log=log,
                label=label,alpha=alpha)
            axes[ai].set_ylabel('Number of lines')
            axes[ai].set_xlabel('Residuals [$e^-$]')
        else:
            if len(orders)>5:
                colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
            else:
                colors = ["C{}".format(n) for n in range(6)]
            marker     = kwargs.get('marker','o')
            markersize = kwargs.get('markersize',2)
            alpha      = kwargs.get('alpha',1.)
            plotargs = {'s':markersize,'marker':marker,'alpha':alpha}
            for o,order in enumerate(orders):
                ord_pix    = np.ravel(pixel.sel(od=order))
                ord_pix    = ord_pix[~np.isnan(ord_pix)]
                
                ord_fitrsd = np.ravel(fitresids.sel(od=order))
                ord_fitrsd = ord_fitrsd[~np.isnan(ord_fitrsd)]
                axes[ai].scatter(ord_pix,ord_fitrsd,**plotargs,color=colors[o])
            [axes[ai].axvline(512*(i),ls=':',lw=0.3) for i in range(9)]
        if show == True: figure.show() 
        return plotter
    def plot_residuals(self,order=None,calibrator='LFC',mean=False,
                       fittype='epsf',plotter=None,axnum=None,show=True,
                       photon_noise=False,**kwargs):
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
            bottom  = kwargs.get('bottom',0.12)
            plotter = SpectrumPlotter(bottom=bottom,**kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        figure, axes = plotter.figure, plotter.axes
        
        orders = self.prepare_orders(order)
                
        lines = self.check_and_return_lines()
        
#        resids  = lines['pars'].sel(par='rsd',od=orders)
        
        pos_pix   = lines['pars'].sel(par='cen',od=orders,ft=fittype)
        pos_res   = lines['wave'].sel(wav='rsd',od=orders,ft=fittype)
        pho_noise = lines['attr'].sel(att='pn',od=orders)
        
        
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker     = kwargs.get('marker','x')
        markersize = kwargs.get('markersize',2)
        alpha      = kwargs.get('alpha',1.)
        color      = kwargs.get('color',None)
        plotargs = {'s':markersize,'marker':marker,'alpha':alpha}
        for i,order in enumerate(orders):
            pix = pos_pix.sel(od=order)
            res = pos_res.sel(od=order)
            if len(orders)>5:
                plotargs['color']=color if color is not None else colors[i]
                
            if not photon_noise:
                axes[ai].scatter(pix,res,**plotargs)
            else:
                pn = pho_noise.sel(od=order)
                axes[ai].errorbar(pix,y=res,yerr=pn,ls='--',lw=0.3,**plotargs)
            if mean==True:
                meanplotargs={'lw':0.8}
                w  = kwargs.get('window',5)                
                rm = hf.running_mean(res,w)
                if len(orders)>5:
                    meanplotargs['color']=colors[i]
                axes[ai].plot(pix,rm,**meanplotargs)
        [axes[ai].axvline(512*(i),lw=0.3,ls='--') for i in range (9)]
        axes[ai]=hf.make_ticks_sparser(axes[ai],'x',9,0,4096)
        axes[ai].set_xlabel('Pixel')
        axes[ai].set_ylabel('Residuals [m/s]')
        if show == True: figure.show() 
        return plotter
    def plot_histogram(self,kind,order=None,separate=False,fittype='epsf',
                       show=True,plotter=None,axnum=None,**kwargs):
        '''
        Plots a histogram of residuals of LFC lines to the wavelength solution 
        (kind = 'residuals') or a histogram of R2 goodness-of-fit estimators 
        (kind = 'R2').
        
        Args:
        ----
            kind:       'residuals' or 'chisq'
            order:      integer or list of orders to be plotted
            plotter:    Plotter Class object (allows plotting multiple spectra
                            in a single panel)
            show:       boolean
        Returns:
        --------
            plotter:    Plotter Class object
        '''
        if kind not in ['residuals','chisq']:
            raise ValueError('No histogram type specified \n \
                              Valid options: \n \
                              \t residuals \n \
                              \t R2')
        else:
            pass
        
        histrange = kwargs.pop('range',None)
        normed    = kwargs.pop('normed',False)
        orders = self.prepare_orders(order)
            
        N = len(orders)
        if plotter is None:
            if separate == True:
                plotter = SpectrumPlotter(naxes=N,alignment='grid',**kwargs)
            elif separate == False:
                plotter = SpectrumPlotter(naxes=1,**kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        figure, axes = plotter.figure, plotter.axes
        lines = self.check_and_get_comb_lines(orders=orders)
        
        # plot residuals or chisq
        if kind == 'residuals':
            data     = lines['wave'].sel(wav='rsd',ft=fittype)
        elif kind == 'chisq':
            data     = lines['pars'].sel(par='chisq',ft=fittype)
        bins    = kwargs.get('bins',10)
        alpha   = kwargs.get('alpha',1.0)
        if separate == True:
            for i,order in enumerate(orders):
                selection = data.sel(od=order).dropna('id').values
                axes[i].hist(selection,bins=bins,normed=normed,range=histrange,
                             alpha=alpha)
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
        elif separate == False:
            selection = np.ravel(data.dropna('id').values)
            print(selection)
            axes[0].hist(selection,bins=bins,normed=normed,range=histrange,
                         alpha=alpha)
            if kind == 'residuals':
                mean = np.mean(selection)
                std  = np.std(selection)
                A    = 1./np.sqrt(2*np.pi*std**2)
                x    = np.linspace(np.min(selection),np.max(selection),100)
                y    = A*np.exp(-0.5*((x-mean)/std)**2)
                axes[ai].plot(x,y,color='C1')
                axes[ai].plot([mean,mean],[0,A],color='C1',ls='--')
                axes[ai].text(0.8, 0.95,r"$\mu={0:8.3e}$".format(mean), 
                            horizontalalignment='center',
                            verticalalignment='center',transform=axes[0].transAxes)
                axes[ai].text(0.8, 0.9,r"$\sigma={0:8.3f}$".format(std), 
                            horizontalalignment='center',
                            verticalalignment='center',transform=axes[0].transAxes)
            axes[ai].set_xlabel("{}".format(kind))
            axes[ai].set_ylabel('Number of lines')
        if show == True: figure.show() 
        return plotter
    def plot_psf(self,order=None,seg=None,plotter=None,psf=None,spline=False,
                       show=True,**kwargs):
        if psf is None:
            self.check_and_load_psf()
            psf = self.psf
            
        if order is None:
            orders = psf.od.values
        else:
            orders = hf.to_list(order)
            
        if seg is None:
            segments = psf.seg.values
        else:
            segments = hf.to_list(seg)
        nseg = len(segments)
        
            
        if plotter is None:
            plotter = SpectrumPlotter(1,bottom=0.12,**kwargs)
#            figure, axes = hf.get_fig_axes(len(orders),bottom=0.12,
#                                              alignment='grid',**kwargs)
        else:
            pass
        figure, axes = plotter.figure, plotter.axes
        
                
        lines = self.check_and_return_lines()
        if nseg>4:    
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0,1,nseg))
        else:
            colors = ["C{0:d}".format(i) for i in range(10)]
        for i,order in enumerate(orders):
            for j,s in enumerate(segments):
                axes[i].scatter(psf.sel(od=order,ax='x',seg=s),
                                psf.sel(od=order,ax='y',seg=s),
                                marker='X',color=colors[j],
                                edgecolor='k',linewidth=0.1)
                if spline:
                    psf_x = psf.sel(od=order,ax='x',seg=s).dropna('pix')
                    psf_y = psf.sel(od=order,ax='y',seg=s).dropna('pix')
                    splrep=interpolate.splrep(psf_x,psf_y)
                    psfpix = psf_x.coords['pix']
                    minpix,maxpix = np.min(psfpix),np.max(psfpix)
                    x = np.linspace(minpix,maxpix,50)
                    y = interpolate.splev(x,splrep)
                    axes[i].plot(x,y,color=colors[j])
                
        if show == True: figure.show()
        return plotter
    def plot_shift(self,order=None,p1='epsf',p2='gauss',
                   plotter=None,axnum=None,show=True,**kwargs):
        ''' Plots the shift between the selected estimators of the
            line centers '''
        if plotter is None:
            plotter = SpectrumPlotter(bottom=0.12,**kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        figure, axes = plotter.figure, plotter.axes
        
        orders = self.prepare_orders(order)
                
        lines = self.check_and_return_lines()
        
        def get_center_estimator(p):
            if p == 'epsf':
                cen = lines['pars'].sel(par='cen',ft='epsf',od=orders)
                label = 'cen_{psf}'
            elif p == 'gauss':
                cen = lines['pars'].sel(par='cen',ft='gauss',od=orders)
                label = 'cen_{gauss}'
            elif p == 'bary':
                cen = lines['attr'].sel(att='bary',od=orders)
                label = 'b'
            return cen, label
        
        cen1,label1  = get_center_estimator(p1)
        cen2,label2  = get_center_estimator(p2)
        bary,labelb  = get_center_estimator('bary')
        delta = cen1 - cen2 
        
        shift = delta * 829
        axes[ai].set_ylabel('[m/s]')
        
        axes[ai].scatter(bary,shift,marker='o',s=2,label="${0} - {1}$".format(label1,label2))
        axes[ai].set_xlabel('Line barycenter [pix]')
        axes[ai].legend()
        
        if show == True: figure.show()
        return plotter
    
    def plot_wavesolution(self,calibrator='LFC',order=None,nobackground=True,
                       plotter=None,axnum=None,naxes=1,ratios=None,title=None,
                       sep=0.05,figsize=(16,9),fittype='epsf',
                       alignment="vertical",sharex=None,sharey=None,show=True,
                       **kwargs):
        '''
        Plots the wavelength solution of the spectrum for the provided orders.
        '''
        
        if plotter is None:
            plotter = SpectrumPlotter(naxes=naxes,ratios=ratios,title=title,
                                  sep=sep,figsize=figsize,alignment=alignment,
                                  sharex=sharex,sharey=sharey,**kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        figure, axes = plotter.figure, plotter.axes
        
        orders = self.prepare_orders(order)
        # Make fittype a list
        fittype = hf.to_list(fittype)
        # Check and retrieve the wavelength calibration
        wavesol_name = 'wavesol_{cal}'.format(cal=calibrator)
        calib_attribute = getattr(self,wavesol_name)
        if calib_attribute is None:
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
            
        # Retrieve line wavelengths    
        pos_freq = lines['attr'].sel(att='freq')
        pos_wav  = (299792458e0/pos_freq)*1e10
        
        # Manage colors
        #cmap   = plt.get_cmap('viridis')
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker = kwargs.get('marker','x')
        ms     = kwargs.get('markersize',5)
        ls     = {'epsf':'-','gauss':'--'}
        # Plot the line through the points?
        plotline = kwargs.get('plot_line',True)
        # Select line data    
        for ft in fittype:
            pos_pix  = lines['pars'].sel(par='cen',ft=ft)
            # Do plotting
            for i,order in enumerate(orders):
                pix = pos_pix.sel(od=order).dropna('id','all')
                wav = pos_wav.sel(od=order).dropna('id','all')
                axes[ai].scatter(pix,wav,s=ms,color=colors[i],marker=marker)
                if plotline == True:
                    axes[ai].plot(wavesol.sel(ft=ft,od=order),color=colors[i],ls=ls[ft])
        axes[ai].set_xlabel('Pixel')
        axes[ai].set_ylabel('Wavelength [$\AA$]')
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
    def save_dataset(self,dataset,dtype=None,dirname=None,replace=False):
        dtype = dtype if dtype in ['lines','LFCws'] \
            else UserWarning('Data type unknown')
        
        if dirname is not None:
            dirname = dirname
        else:
            if dtype == 'lines':
                dirname = hs.harps_lines
            elif dtype == 'LFCws':
                dirname = hs.harps_ws
        
        direxists = os.path.isdir(dirname)
        if not direxists:
            raise ValueError("Directory does not exist")
        else:
            pass
        basename = os.path.basename(self.filepath)[:-5]
        path     = os.path.join(dirname,basename+'_{}.nc'.format(dtype))
        
        dataset    = self.include_attributes(dataset)
        if replace==True:
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            dataset.to_netcdf(path,engine='netcdf4')
            print('Dataset saved to: {}'.format(path))
        except:
            print('Dataset could not be saved to {}'.format(path))
    def save_wavesol(self,dirname=None,replace=False):
#        dirname = dirname if dirname is not None else hs.harps_ws
#        direxists = os.path.isdir(dirname)
#        if not direxists:
#            raise ValueError("Directory does not exist")
#        else:
#            pass
#        basename = os.path.basename(self.filepath)[:-5]
#        path     = os.path.join(dirname,basename+'_LFCws.nc')
        
        wavesol_LFC  = self.check_and_get_wavesol(calibrator='LFC')
#        wavecoef_LFC = self.wavecoef_LFC
#        dataset      = xr.merge([wavesol_LFC,wavecoef_LFC])
        LFCws        = self.LFCws
        self.save_dataset(LFCws,dtype='LFCws',dirname=dirname,replace=replace)
        
#        dataset      = self.include_attributes(dataset)
#        print(dataset)
#        dataset.to_netcdf(path,engine='netcdf4')
#        print('Wavesolution saved to: {}'.format(path))
    def save_lines(self,dirname=None,replace=False):
#        dirname = dirname if dirname is not None else hs.harps_lines
#        direxists = os.path.isdir(dirname)
#        if not direxists:
#            raise ValueError("Directory does not exist")
#        else:
#            pass
#        basename = os.path.basename(self.filepath)[:-5]
#        path     = os.path.join(dirname,basename+'_lines.nc')
        
        lines    = self.check_and_get_comb_lines()
        self.save_dataset(lines,dtype='lines',dirname=dirname,replace=replace)
#        lines    = self.include_attributes(lines)
#        if replace==True:
#            try:
#                os.remove(path)
#            except OSError:
#                pass
#        try:
#            lines.to_netcdf(path,engine='netcdf4')
#            print('Lines saved to: {}'.format(path))
#        except:
#            print('Lines could not be saved to {}'.format(path))
    def include_attributes(self,xarray_object):
        '''
        Saves selected attributes of the Spectrum class to the xarray_object
        provided.
        '''
        
        xarray_object.attrs['LFC'] = self.LFC
        xarray_object.attrs['fr_source'] = self.fr_source
        xarray_object.attrs['f0_source'] = self.f0_source
        xarray_object.attrs['fibreshape'] = self.fibre_shape
        
        xarray_object.attrs['gaps'] = int(self.use_gaps)
        xarray_object.attrs['patches'] = int(self.patches)
        xarray_object.attrs['polyord'] = self.polyord

        xarray_object.attrs['harps.classes version'] = __version__
        return xarray_object
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
###############################################################################
##############################   MANAGER   ####################################
###############################################################################
class Manager(object):
    '''
    Manager is able to find the files on local drive, read the selected data, 
    and perform bulk operations on the data.
    '''
    def __init__(self,
                 date=None,year=None,month=None,day=None,
                 begin=None,end=None,
                 run=None,sequence=None,
                 dirpath=None,
                 filelist=None,
                 get_file_paths=True):
        '''
        Ways to initialize object:
            (1) Provide a single date
            (2) Provide a begin and end date
            (3) Provide a run and sequence ID
            (4) Provide a path to a directory
            (5) Provide a path to a list of files to use
        date(yyyy-mm-dd)
        begin(yyyy-mm-dd)
        end(yyyy-mm-dd)
        sequence(day,sequence)
        '''
        def get_init_method():
            if date!=None or (year!=None and month!=None and day!=None):
                method = 1
            elif begin!=None and end!=None:
                method = 2
            elif run!=None and sequence!=None:
                method = 3
            elif dirpath!=None:
                method = 4
            elif filelist!=None:
                method = 5
            return method
        
        baseurl     = 'http://people.sc.eso.org/%7Eglocurto/COMB/'
        
        self.file_paths = []
        self.spectra    = []
        #harpsDataFolder = os.path.join("/Volumes/home/dmilakov/harps","data")
        harpsDataFolder = harps_data#os.path.join("/Volumes/home/dmilakov/harps","data")
        self.harpsdir   = harpsDataFolder
        self.datadir_list = []
        
        method = get_init_method()
#        print(method)
        self.method = method

        if method==1:
            self.sequence_list_filepath = None
            if   date==None and (year!=None and month!=None and day!=None):
                self.dates = ["{y:4d}-{m:02d}-{d:02d}".format(y=year,m=month,d=day)]
                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
            elif date!=None:
                self.dates = [date]
                
                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
                print(self.datadir)
            elif date==None and (year==None or month==None or day==None) and (begin==None or end==None):
                raise ValueError("Invalid date input. Expected format is 'yyyy-mm-dd'.")
        elif method==2:
            
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
                
            
            
            for date in self.dates:
                #print(date)
                datadir = os.path.join(harpsDataFolder,date)
                if os.path.isdir(datadir):
                    self.datadir_list.append(datadir)
        if method==3:
            run = run if run is not None else ValueError("No run selected")
            
            if type(sequence)==tuple:
                sequence_list_filepath = baseurl+'COMB_{}/day{}_seq{}.list'.format(run,*sequence)
                print(sequence_list_filepath)
                self.sequence_list_filepath = [sequence_list_filepath]
                self.sequence = [sequence]
            elif type(sequence)==list:
                self.sequence_list_filepath = []
                self.sequence = sequence
                for item in sequence:
                    sequence_list_filepath = baseurl+'COMB_{}/day{}_seq{}.list'.format(run,*item)
                    self.sequence_list_filepath.append(sequence_list_filepath)
                    
        elif method == 4:
            if type(dirpath)==str: 
                self.datadir_list.append(dirpath)
            elif type(dirpath)==list:
                for d in dirpath:
                    self.datadir_list.append(d)
        elif method == 5:
            path_list = pd.read_csv(filelist,comment='#',names=['file'])
            absolute = np.all([os.path.exists(filepath) for filepath in path_list.file])
            if not absolute:
                pass
                
#        else:
#            self.sequence_list_filepath = None
#            if   date==None and (year!=None and month!=None and day!=None) and (begin==None or end==None):
#                self.dates = ["{y:4d}-{m:02d}-{d:02d}".format(y=year,m=month,d=day)]
#                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
#            elif date!=None:
#                self.dates = [date]
#                
#                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
#                print(self.datadir)
#            elif date==None and (year==None or month==None or day==None) and (begin==None or end==None):
#                raise ValueError("Invalid date input. Expected format is 'yyyy-mm-dd'.")
#            elif (begin!=None and end!=None):
#                by,bm,bd       = tuple(int(val) for val in begin.split('-'))
#                ey,em,ed       = tuple(int(val) for val in end.split('-'))
#                print(by,bm,bd)
#                print(ey,em,ed)
#                self.begindate = datetime.datetime.strptime(begin, "%Y-%m-%d")
#                self.enddate   = datetime.datetime.strptime(end, "%Y-%m-%d")
#                self.dates     = []
#                def daterange(start_date, end_date):
#                    for n in range(int ((end_date - start_date).days)):
#                        yield start_date + datetime.timedelta(n)
#                for single_date in daterange(self.begindate, self.enddate):
#                    self.dates.append(single_date.strftime("%Y-%m-%d"))
#                
#            self.datadirlist  = []
#            
#            for date in self.dates:
#                #print(date)
#                datadir = os.path.join(harpsDataFolder,date)
#                if os.path.isdir(datadir):
#                    self.datadirlist.append(datadir)
        self.orders = np.arange(sOrder,eOrder,1)
        if get_file_paths==True and len(self.file_paths)==0 :
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
        
        if self.method==3:  
            # Run ID and sequence provided
            print(self.sequence_list_filepath)
            if type(self.sequence_list_filepath)==list:    
                # list to save paths to files on disk
                sequence_list = []
                
                for item,seq in zip(self.sequence_list_filepath,self.sequence):
                    # read files in the sequence from the internet
                    req = urllib.request.Request(item)
                    res = urllib.request.urlopen(req)
                    htmlBytes = res.read()
                    htmlStr   = htmlBytes.decode('utf8').split('\n')
                    filenamelist  = htmlStr[:-1]
                    # append filenames to a list
                    for filename in filenamelist:
                        sequence_list.append([seq,filename[0:29]])
                    # use the list to construct filepaths to files on disk
                    for fbr in list(fibre):
                        fitsfilepath_list = []
                        for seq,item in sequence_list:
                            date    = item.split('.')[1][0:10]
                            datadir = os.path.join("{date}".format(date=item.split('.')[1][0:10]),
                                                   "series {n:02d}".format(n=seq[1]))
                            time    = item.split('T')[1].split(':')
                            fitsfilepath = os.path.join(self.harpsdir,datadir,
                                            "HARPS.{date}T{h}_{m}_{s}_{ft}_{f}.fits".format(date=date,h=time[0],m=time[1],s=time[2],ft=ftype,f=fbr))
                            fitsfilepath_list.append(fitsfilepath)
                            #print(date,time,fitsfilepath,os.path.isfile(fitsfilepath))
                        filePaths[fbr] = fitsfilepath_list
        elif self.method==5:
            pass
        else:
            for fbr in list(fibre):
                nestedlist = []
                for datadir in self.datadir_list:
                    try:
                        files_in_dir=np.array(glob(os.path.join(datadir,"*{ftp}*{fbr}.fits".format(ftp=ftype,fbr=fbr))))
                    except:
                        raise ValueError("No files of this type were found")
                    if "condition" in kwargs.keys():
                        self.condition = {"condition":kwargs["condition"]}
                        if kwargs["condition"] == "filename":
                            self.condition["first"] = kwargs["first"]
                            self.condition["last"]  = kwargs["last"]
                            ff = np.where(files_in_dir==os.path.join(datadir,
                                    "{base}_{ftp}_{fbr}.fits".format(base=kwargs["first"],ftp=self.ftype,fbr=fbr)))[0][0]
                            lf = np.where(files_in_dir==os.path.join(datadir,
                                    "{base}_{ftp}_{fbr}.fits".format(base=kwargs["last"],ftp=self.ftype,fbr=fbr)))[0][0]
                            selection = files_in_dir[ff:lf]
                            nestedlist.append(selection)
                    else:
                        nestedlist.append(files_in_dir)
                flatlist       = [item for sublist in nestedlist for item in sublist]   
                filePaths[fbr] = flatlist
              
        
        self.file_paths = filePaths
        basenames       = [[os.path.basename(file)[:-5] for file in filePaths[f]] for f in list(fibre)]
        self.basenames  = dict(zip(list(fibre),basenames))
        datetimes      = [[np.datetime64(bn.split('.')[1].replace('_',':')) for bn in self.basenames[fbr]] for fbr in list(fibre)]
        self.datetimes = dict(zip(list(fibre),datetimes))
        self.numfiles = [np.size(filePaths[fbr]) for fbr in list(fibre)]
        if np.sum(self.numfiles)==0:
            raise UserWarning("No files found in the specified location")
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
    def read_data(self,dtype=None,fibre='AB',dirname=None,autoclose=True,
                  engine=None):
        ''' Reads lines and wavelength solutions of the spectra '''
        if dtype in ['lines','LFCws']:
            dtype = dtype
        else:
            raise UserWarning('Data type unknown')
        
        # make the fibre argument a list
        fibres = list(fibre)
        if len(self.file_paths) == 0:
            self.get_file_paths(fibre=fibre)
        else:
            pass
        # get dirnames
        if dirname is not None:
            dirnames = dirname
        else:
            if dtype == 'lines':
                dirnames = dict(zip(fibres,[hs.harps_lines for fbr in fibres]))
            elif dtype == 'LFCws':
                dirnames = dict(zip(fibres,[hs.harps_ws for fbr in fibres]))
            else:
                raise UserWarning("Uknown")
                
#        dirnames = dirname if dirname is not None else dict(zip(fibres,[hs.harps_lines for fbr in fibres]))
        if type(dirname)==dict:
            pass
        elif type(dirname)==str:
            dirnames = dict(zip(fibres,[dirname for fbr in fibres]))
        
        # get basename
        basenames = self.basenames
        
        filenames = [[os.path.join(dirnames[fbr],b+'_{}.nc'.format(dtype)) \
                                   for b in basenames[fbr]] for fbr in fibres]
        filelist = dict(zip(fibres,filenames))
        ll = []
        for fbr in fibres:
            idx   = pd.Index(self.datetimes[fbr],name='time')
            data_fibre = xr.open_mfdataset(filelist[fbr],
                                           concat_dim=idx,
                                           engine=engine,
                                           autoclose=autoclose)
            data_fibre = data_fibre.sel(od=np.arange(46,50))
            data_fibre = data_fibre.sortby('time')
            data_fibre.expand_dims('fb')
            ll.append(data_fibre)
        if len(fibres)>1:
            data = xr.concat(ll,dim=pd.Index(['A','B'],name='fb'))
        else:
            data = ll[0]
            
        return data
    def read_lines(self,fibre='AB',dirname=None,**kwargs):
        self.lines = self.read_data(dtype='lines',fibre=fibre,
                                    dirname=dirname,**kwargs)
        return self.lines
    def read_wavesol(self,fibre='AB',dirname=None,**kwargs):
        self.LFCws = self.read_data(dtype='LFCws',fibre=fibre,
                                    dirname=dirname,**kwargs)
        return self.LFCws

    def _get_index(self,centers):
        ''' Input: dataarray with fitted positions of the lines
            Output: 1d array with indices that uniquely identify every line'''
        fac = 10000
        MOD = 2.
        od = centers.od.values[:,np.newaxis]*fac
        centers_round = np.rint(centers.values/MOD)*MOD
        centers_nonan = np.nan_to_num(centers_round)
        ce = np.asarray(centers_nonan,dtype=np.int)
        index0=np.ravel(od+ce)
        mask = np.where(index0%fac==0)
        index0[mask]=999999999
        return index0
    def _get_sorted(self,index1,index2):
        print('len indexes',len(index1),len(index2))
        # lines that are common for both spectra
        intersect=np.intersect1d(index1,index2)
        intersect=intersect[intersect>0]
    
        indsort=np.argsort(intersect)
        
        argsort1=np.argsort(index1)
        argsort2=np.argsort(index2)
        
        sort1 =np.searchsorted(index1[argsort1],intersect)
        sort2 =np.searchsorted(index2[argsort2],intersect)
        
        return argsort1[sort1],argsort2[sort2]
    
    def calc_lambda(self,ft='gauss',orders=None,iref=0,flim=2e3):
        ''' Returns wavelength and wavelength error for the lines using 
            polynomial coefficients in wavecoef_LFC.
            
            Adapted from HARPS mai_compute_drift.py'''
        if orders is not None:
            orders = orders
        else:
            orders = np.arange(self.sOrder,self.nbo,1)
        lines0 = self.lines#.load()
        
        # wavelength calibrations
        #ws    = self.check_and_get_wavesol()
        wc    = self.LFCws['coef']
        # coordinates
        times = lines0.coords['time']
        ids   = lines0.coords['id']
        fibre = lines0.coords['fb']
        nspec = len(lines0.coords['time'])
        
        # select only lines which have fluxes above the limit
        lines0 = lines0.where(lines0['pars'].sel(par='flx')>flim)
        
        
        # new dataset
        
        dims_wave   = ['fb','time','od','id','val']
        dims_rv     = ['fb','time','par']
        dims_att    = ['fb','time','att']
        shape_wave  = (len(fibre),nspec,len(orders),400,2)
        shape_rv    = (len(fibre),nspec,2)
        shape_att   = (len(fibre),nspec,2)
        variables   = {'wave':(dims_wave,np.full(shape_wave,np.nan)),
                       'rv':(dims_rv,np.full(shape_rv,np.nan)),
                       'stat':(dims_att,np.full(shape_att,np.nan))}
        coords      = {'fb':fibre.values,
                       'time':times.values,
                       'od':orders,
                       'id':ids.values,
                       'att':['nlines','average_flux'],
                       'val':['wav','dwav'],
                       'par':['rv','rv_err']}
        dataset     = xr.Dataset(variables,coords)

        # try using thar_coef
        #workdir = harps_data
        workdir = '/Volumes/dmilakov/harps/data'
        specpath=['2012-02-15','series02','HARPS.2012-02-15T13_48_55.530_e2ds_A.fits']
        spec0_path = os.path.join(workdir,*specpath)
        spec0 = Spectrum(spec0_path)
        thar  = spec0.__get_wavesol__('ThAr')
        coef  = spec0.wavecoeff_air[orders]
       
        for fbr in fibre:
            for j in range(nspec):
                print(fbr.values,j)
                idx = dict(fb=fbr,time=times[j],od=orders)
                # all lines in this order, fittype, exposure, fibre
                l0 = lines0.sel(od=orders,ft=ft,time=times[j],fb=fbr)
                # centers and errors
                x     = (l0['pars'].sel(par='cen')).values
                x_err = (l0['pars'].sel(par='cen_err')).values
               
#                coef  = wc.sel(patch=0,od=orders,
#                               ft=ft,time=times[iref],
#                               fb=fbr).values
                # wavelength of lines
                wave = np.sum([coef[:,i]*(x.T**i) \
                               for i in range(np.shape(coef)[-1])],axis=0).T
                dataset['wave'].loc[dict(fb=fbr,time=times[j],od=orders,val='wav')] = wave
             
                # wavelength errors
                dwave = np.sum([(i+1)*coef[:,i+1]*(x.T**(i)) \
                                for i in range(np.shape(coef)[-1]-1)],axis=0).T*x_err
                #print(np.shape(hf.ravel(x)),np.shape(hf.ravel(x_err)))
                dataset['wave'].loc[dict(fb=fbr,time=times[j],od=orders,val='dwav')] = dwave
          
        self.dataset=dataset
        return dataset
    def calc_rv(self,orders=None,iref=0,sigma=5,ft='gauss'):
        plot=False
        
        if orders is not None:
            orders = orders
        else:
            orders = np.arange(self.sOrder,self.nbo,1)
        dataset = self.dataset
        lines = self.lines
        times = dataset.coords['time']
        ids   = dataset.coords['id']
        fibre = dataset.coords['fb']
        nspec = len(times)
#        nspec = 10
        # lines
        lines = self.lines
        
        for fbr in fibre:
            
            cen_ref = lines['pars'].sel(fb=fbr,time=times[iref],
                            od=orders,par='cen',ft=ft)
            index_ref = self._get_index(cen_ref)
            wavref = dataset['wave'].sel(fb=fbr,time=times[iref],
                                       od=orders,val='wav')
            dwref  = dataset['wave'].sel(fb=fbr,time=times[iref],
                                       od=orders,val='dwav')

            wavref = np.ravel(wavref)
            dwref  = np.ravel(dwref)
            for j in range(nspec):
                total_flux = lines['stat'].sel(fb=fbr,time=times[j],
                                  od=orders,odpar='sumflux')
                total_flux = np.sum(total_flux)
                cen   = lines['pars'].sel(fb=fbr,time=times[j],
                            od=orders,par='cen',ft=ft)
                index = self._get_index(cen)
                
                
                wav1  = dataset['wave'].sel(fb=fbr,time=times[j],
                                          od=orders,val='wav')
                dwav1 = dataset['wave'].sel(fb=fbr,time=times[j],
                                          od=orders,val='dwav')

                wav1  = np.ravel(wav1)
                dwav1 = np.ravel(dwav1)
                # global shift
                c=2.99792458e8
                sel1,sel2 = self._get_sorted(index,index_ref)
                v=hf.ravel(c*(wav1[sel1]-wavref[sel2])/wav1[sel1])
                dwav=hf.ravel(np.sqrt((c*dwref[sel2]/wavref[sel2])**2+\
                                      (c*dwav1[sel1]/wav1[sel1])**2))
                
                if j!=iref:
                    m=hf.sig_clip2(v,sigma)
                else:
                    m=np.arange(len(v))
   	  
                average_flux = total_flux / len(m)
                global_dv    = np.sum(v[m]/(dwav[m])**2)/np.sum(1/(dwav[m])**2)
                global_sig_dv= (np.sum(1/(dwav[m])**2))**(-0.5)
                print(fbr.values,j,len(v), sum(m), global_dv, global_sig_dv)
                if plot:
                    plt.figure()
                    plt.title("fbr={}, spec={}".format(fbr.values,times[j].values))
                    plt.scatter(np.arange(len(v)),v,s=2)
                    plt.scatter(np.arange(len(v))[m],v[m],s=2)
                    #plt.plot(freq1d[sel1]-freq1d_ref[sel2])
                dataset['rv'].loc[dict(fb=fbr,time=times[j],par='rv')] = global_dv
                dataset['rv'].loc[dict(fb=fbr,time=times[j],par='rv_err')] = global_sig_dv
                dataset['stat'].loc[dict(fb=fbr,time=times[j],att='nlines')] = len(m)
                dataset['stat'].loc[dict(fb=fbr,time=times[j],att='average_flux')] = average_flux
        self.dataset=dataset
        return self.dataset
    
    def plot_rv_time(self,fittype='gauss',method=1): 
        dv = self.dataset
        
        plotter = SpectrumPlotter(bottom=0.12)
        fig, ax = plotter.figure, plotter.axes
        t = dv.coords['time'].values
        t = np.arange(len(t))
        AmB= dv['rv'].sel(par='rv',fb='A') - \
             dv['rv'].sel(par='rv',fb='B')
        AmB_err = np.sqrt(np.sum([dv['rv'].sel(par='rv_err',fb=f)**2 \
                                  for f in ['A','B']],axis=0))
        ax[0].errorbar(t,AmB,AmB_err,label='A-B',ls='',marker='x',ms=5)
        for f in ['A','B']:  
            x = dv['stat'].sel(fb=f,att='average_flux')
            y = dv['rv'].sel(par='rv',fb=f)
            y_err = dv['rv'].sel(par='rv_err',fb=f)
            ax[0].errorbar(t,y,y_err,label=f,ls='',marker='o',ms=2)
        ax[0].axhline(0,ls='--',c='k',lw=0.7)
        [ax[0].axvline((10*i),ls=':',lw=0.5,c='C0') for i in range(len(t)//10)]
        ax[0].legend()
        ax[0].set_ylim(-6,10)
        ax[0].set_xlabel("Exposure number")
        ax[0].set_ylabel("Shift [m/s]")
        return plotter
    def plot_rv_flux(self,fittype='gauss',method=1): 
        dv = self.dataset
        
        plotter = SpectrumPlotter()
        fig, ax = plotter.figure, plotter.axes
       
        AmB= dv['rv'].sel(par='rv',fb='A') - \
             dv['rv'].sel(par='rv',fb='B')
        AmB_err = np.sqrt(np.sum([dv['rv'].sel(par='rv_err',fb=f)**2 \
                                  for f in ['A','B']],axis=0))
        x0 = np.average(dv['stat'].sel(att='average_flux'),axis=0)
        ax[0].errorbar(x0,AmB,AmB_err,label='A-B',ls='',marker='x',ms=5)
#        for f in ['A','B']:  
#            x = dv['stat'].sel(fb=f,att='average_flux')
#            y = dv['rv'].sel(par='rv',fb=f)
#            y_err = dv['rv'].sel(par='rv_err',fb=f)
#            ax[0].errorbar(x,y,y_err,label=f,ls='',marker='o',ms=2)
        ax[0].axhline(0,ls='--',c='k',lw=0.7)
        ax[0].legend()
        return plotter
    def get_spectrum(self,ftype,fibre,header=False,data=False):
        return 0
#    def read_data(self, filename="datacube", **kwargs):
#        try:    
#            fibre       = kwargs["fibre"]
#            self.fibre  = fibre
#        except: fibre   = self.fibre
#        try:    
#            orders      = kwargs["orders"]
#            self.orders = orders
#        except: orders  = self.orders
#        # if there are conditions, change the filename to reflect the conditions
#        try:
#            for key,val in self.condition.items():
#                filename = filename+"_{key}={val}".format(key=key,val=val)
#        except:
#            pass
#        if not self.datadirlist:
#            self.get_file_paths(fibre=fibre)
#        # CREATE DATACUBE IF IT DOES NOT EXIST
#        if   len(self.dates)==1:
#            self.datafilepath = os.path.join(self.datadirlist[0],
#                                         "{name}_{fibre}.npy".format(name=filename, fibre=fibre))
#        elif len(self.dates)>1:
#            self.datafilepath = os.path.join(self.harpsdir,
#                                         "{name}_{fibre}_{begin}_{end}.npy".format(name=filename, fibre=fibre,
#                                             begin=self.begindate.strftime("%Y-%m-%d"), end=self.enddate.strftime("%Y-%m-%d")))
#        #self.datafilepath = os.path.join(self.harpsdir,"2015-04-18/datacube_condition=filename_first=HARPS.2015-04-18T01_35_46.748_last=HARPS.2015-04-18T13_40_42.580_AB.npy")
#        if os.path.isfile(self.datafilepath)==False:
#            #sys.exit()
#            print("Data at {date} is not prepared. Processing...".format(date=self.dates))
#            self.reduce_data(fibre=fibre, filename=filename)
#        else:
#            pass        
#        
#        datainfile  = np.load(self.datafilepath)
#        self.dtype  = [datainfile.dtype.fields[f][0].names for f in list(fibre)][0]
#        self.nfiles = [np.shape(datainfile[f]["FLX"])[1] for f in list(fibre)]
#        if kwargs["orders"]:
#            col         = hf.select_orders(orders)
#            subdtype  = Datatypes(nOrder=len(orders),nFiles=self.nfiles[0],fibre=fibre).specdata(add_corr=True)
#            data        = np.empty(shape=datainfile.shape, dtype=subdtype.data)
#            for f in list(fibre):
#                for dt in self.dtype:
#                    data[f][dt] = datainfile[f][dt][:,:,col]
#            
#        else:
#            data    = datainfile
#        self.data   = data
#        
        
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
        #nFibres = len(fibres)
        nFiles  = len(self.file_paths[fibres[0]])
        print("Found {} files".format(nFiles))

        data    = np.zeros((hs.nPix,),dtype=Datatypes(nFiles=nFiles,nOrder=nOrder,fibre=fibre).specdata(add_corr=True).data)
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
        #col = hf.select_orders(orders)
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
        cut     = np.where(uppix<=hs.nPix)
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


###############################################################################
###########################    LSF MODELLER   #################################
###############################################################################
class LSFModeller(object):
    def __init__(self,manager,orders=None,specnum=10,segnum=16,subnum=4,
                 niter=4,fibre='AB',LFC='HARPS',fibreshape='round'):
        ''' Initializes the LSF Modeller
        
        Args:
        -----
            manager: Manager class object with paths to LFC spectra
            orders:  (scalar or list) echelle orders for which to 
                     perform LSF modelling
            specnum: (scalar) number of spectra to use 
            segnum:  (scalar) number of subdivisions of 4096 pixel 
            niter:   (scalar) number of iterations for LSF modelling
            fibre:   (str) fibres to perform LSF modelling 
            LFC:     (str) HARPS or FOCES
            fibreshape: (str) shape of the fibre (round or octagonal)
            
        '''
        if orders is not None:
            orders = hf.to_list(orders)
        else:
            orders = np.arange(hs.sOrder,hs.eOrder,1)
            
        self.manager = manager
        self.orders  = orders
        self.specnum = specnum
        self.segnum  = segnum
        self.subnum  = subnum
        self.LFC     = LFC
        self.niter   = niter
        self.fibres  = list(fibre)
        self.fibreshape = fibreshape
        
        self.interpolate=True
        self.fit_gaussians=False
        
        self.topdir = os.path.join(hs.harps_prod,'psf_fit')
        
        self.savedir  = os.path.join(self.topdir,'April2015_2')
        
        def mkdir_p(path):
            try:
                os.makedirs(path,exist_ok=True)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(path):
                    pass
                else:
                    raise
                    
        for fibre in self.fibres:
            fibredir = os.path.join(self.savedir,'fibre{}'.format(fibre))
            mkdir_p(fibredir)
        
    def return_eLSF(self,fibre,order):
        ''' Performs effective LSF reconstruction in totality'''
        manager = self.manager
        niter   = self.niter
        orders  = order#self.orders
        segnum  = self.segnum
        #subnum  = self.subnum
        specnum = self.specnum
        
        interpolate_local_psf = self.interpolate
        fit_gaussians = self.fit_gaussians
        
        #orders = orders if orders is not None else [45]
        
        data0   = self.initialize_dataset(order)
        data    = self.stack_lines_from_spectra(data0,fibre,first_iteration=True,fit_gaussians=fit_gaussians) 
        # j counts iterations
        j = 0
        # 
        data_with_pars = data_with_eLSF = data_recentered = data
        plot_elsf = False
        plot_cen  = False
        if plot_elsf:
            fig_elsf,ax_elsf = hf.get_fig_axes(segnum,alignment='grid',title='LSF iteration')
        if plot_cen:
            fig_cen,ax_cen = hf.get_fig_axes(1,title='Centeroid shifts')
        # iteratively derive LSF 
        while j < niter:
            data_with_eLSF  = self.construct_eLSF(data_recentered)
            data_with_pars  = self.solve_line_positions(data_with_eLSF,interpolate_local_psf)
            data_recentered = self.stack_lines_from_spectra(data_with_pars,fibre,False)       
            
            j +=1
        final_data = data_recentered
        return final_data
    def run(self):
        orders = self.orders
        fibres = self.fibres
        for fibre in list(fibres):
        
            for order in orders:
                
                filepath = self.get_filepath(order,fibre)
                #print(filepath)
                fileexists = os.path.isfile(filepath)
                if fileexists == True:
                    print('FIBRE {0}, ORDER {1} {2:>10}'.format(fibre,order,'exists'))
                    continue
                else:
                    print('FIBRE {0}, ORDER {1} {2:>10}'.format(fibre,order,'working'))
                    pass
                
                data=self.return_eLSF(fibre,order)
                self.save2file(data,fibre)
        return data
                
            
    def initialize_dataset(self,order):
        ''' Returns a new xarray dataset object of given shape.'''
#        orders  = order #self.orders
        specnum = self.specnum
        segnum  = self.segnum
        subnum  = self.subnum
        nOrders = 1
       
        # number of pixels each eLSF comprises of
        npix   = 17
        # make the subsampled grid where eLSF will be tabulated
        a      = divmod(npix,2)
        xrange = (-a[0],a[0]+a[1])
        pixels    = np.arange(xrange[0],xrange[1],1/subnum)
        # assume each segment contains 60 lines (not true for large segments!)
        lines_per_seg = 60
        # create a multi-index for data storage
        mdix      = pd.MultiIndex.from_product([np.arange(segnum),
                                            np.arange(specnum),
                                            np.arange(lines_per_seg)],
                                names=['sg','sp','id'])
        ndix      = specnum*segnum*lines_per_seg
        # axes for each line
        # x = x coords in eLSF reference frame
        # y = eLSF estimates in eLSF reference frame
        # pos = x coords on CCD
        # flx = counts extracted from CCD
        # err = sqrt(flx), backgroud included
        # lsf = eLSF estimate for this line
        # rsd = residuals between the model and flux
        # der = derivatives of eLSF
        # w   = weigths 
        # mod = model of the line
        axes   = ['x','y','pos','flx','err','bkg','lsf','rsd','der','w','mod']
        n_axes = len(axes)
        # values for each parameter
        values = ['cen','cen_err','flx','flx_err','sft','phi','bary','cen_1g']
        n_vals = len(values)
        # create xarray Dataset object to save the data
        data0   = xr.Dataset({'line': (['od','pix','ax','idx'], np.full((nOrders,npix*subnum,n_axes,ndix),np.nan)),
                         'resd': (['od','seg','pix'],np.full((nOrders,segnum,npix*subnum),np.nan)),
                         'elsf': (['od','seg','pix','ax'], np.full((nOrders,segnum,npix*subnum,n_axes),np.nan)),
                         'shft': (['od','seg'], np.full((nOrders,segnum),np.nan)),
                         'pars': (['od','idx','val'], np.full((nOrders,ndix,n_vals),np.nan)),
                         'gauss': (['ng','od','pix','ax','idx'], np.full((2,nOrders,npix*subnum,n_axes,ndix),np.nan))},
                         coords={'od' :[order], 
                                 'idx':mdix, 
                                 'pix':pixels,
                                 'seg':np.arange(segnum),
                                 'ax' :axes,
                                 'val':values,
                                 'ng' :[1,2]})
    
        return data0
    def stack_lines_from_spectra(self,data,fibre='A',first_iteration=None,fit_gaussians=False):
        ''' Stacks LFC lines along their determined centre
        
            Stacks the LFC lines along their centre (or barycentre) using all the 
            spectra in the provided Manager object. Returns updated xarray dataset 
            (provided by the keyword data).
        '''
        manager = self.manager
        def get_idxs(barycenters,order,nspec):
            '''Returns a list of (segment,spectrum,index) for a given order.'''
            segs=np.asarray(np.array(barycenters[order])//s,np.int32)
            seg,frq = np.unique(segs,return_counts=True)
            nums=np.concatenate([np.arange(f) for s,f in zip(seg,frq)])
            idxs = [(s, nspec, i) for s,i in zip(segs,nums)]
            return idxs
        def return_n_filepaths(manager,N,fibre,skip=5):
            ''' Returns a list of length N with paths to HARPS spectra contained
                in the Manager object. Skips files so to improve dithering of
                lines.
            '''
            i = 0
            files = []
            while len(files) < N:
                spec = Spectrum(manager.file_paths[fibre][skip*i+1])
                spec.__get_wavesol__('ThAr')
                if np.sum(spec.wavesol_thar)==0:
                    pass
                else:
                    files.append(spec.filepath)
                i+=1
                
            return files
        if first_iteration == None:
            # check if data['pars'] is empty
            if np.size(data['pars'].dropna('val','all')) == 0:
                first_iteration = True
            else:
                first_iteration = False
            
        orders          = data.coords['od'].values
        pixels          = data.coords['pix'].values
        specnum         = np.unique(data.coords['sp'].values).size
        pix_step        = pixels[1]-pixels[0]
        pixelbins       = (pixels[1:]+pixels[:-1])/2
        segments        = np.unique(data.coords['seg'].values)
        N_seg           = len(segments)
        s               = 4096//N_seg
        pbar            = tqdm.tqdm(total=(specnum*len(orders)),desc="Centering spectra")
        files           = return_n_filepaths(manager,specnum,fibre)
        for i_spec, file in enumerate(files):
            #print("SPEC {0} {1}".format(fibre,i_spec+1))
            # use every 5th spectrum to improve the sampling of the PSF
            spec = Spectrum(file,LFC='FOCES')
            
            xdata,ydata,edata,bdata,barycenters =spec.cut_lines(orders,nobackground=False,
                                              columns=['pixel','flux','error','bkg','bary'])
            
            for o,order in enumerate(orders):
                idxs = get_idxs(barycenters,order,i_spec)
                numlines = len(barycenters[order])
                if first_iteration:
                    maxima      = spec.get_extremes(order,scale='pixel',extreme='max')['y']
                    lines_1g    = spec.fit_lines(order,fittype='gauss')
                # stack individual lines
                for i in range(numlines):
                    
                    line_pix = xdata[order][i]
                    line_flx = ydata[order][i]
                    line_err = edata[order][i]
                    line_bkg = bdata[order][i]
                    line_flx_nobkg = line_flx-line_bkg
                    idx = idxs[i]
                    # cen is the center of the ePSF!
                    # all lines are aligned so that their cen aligns
                    if first_iteration:
                        b        = barycenters[order][i]
                        cen      = barycenters[order][i]
                        cen_err  = 0
                        flux     = np.max(line_flx)#maxima.iloc[i]
                        flux_err = np.sqrt(flux)
                        shift    = 0
                        phase    = cen - int(cen+0.5)
                        try: cen_g1   = lines_1g.center.iloc[i]
                        except: cen_g1 = np.nan
                        pars     = (cen,cen_err,flux,flux_err,shift,phase,b,cen_g1)
                    else:
                        pars = data['pars'].sel(od=order,idx=idx).values
                        cen,cen_err,flux,flux_err,shift,phase,b,cen_g1 = pars
                    
                    data['pars'].loc[dict(idx=idx,od=order)] = np.array(pars)
                    
                    #-------- MOVING TO A COMMON FRAME--------
                    xline0 = line_pix - cen
                    pix = pixels[np.digitize(xline0,pixelbins,right=True)]
                    
                    #-------- LINE POSITIONS & FLUX --------
                    # first clear the previous estimates of where the line is located
                    # (new estimate of the line center might make the values in the  
                    # 'pix' array different from before, but this is not erased
                    # with each new iteration)
                    
                    data['line'].loc[dict(ax='pos',od=order,idx=idx)]=np.nan
                    data['line'].loc[dict(ax='flx',od=order,idx=idx)]=np.nan
                    data['line'].loc[dict(ax='err',od=order,idx=idx)]=np.nan
                    data['line'].loc[dict(ax='bkg',od=order,idx=idx)]=np.nan
                    data['line'].loc[dict(ax='x',od=order,idx=idx)]  =np.nan
                    data['line'].loc[dict(ax='w',od=order,idx=idx)]  =np.nan
                    # --------------------------------------
                    # Save new values
                    data['line'].loc[dict(ax='pos',od=order,idx=idx,pix=pix)]=line_pix
                    data['line'].loc[dict(ax='flx',od=order,idx=idx,pix=pix)]=line_flx
                    data['line'].loc[dict(ax='err',od=order,idx=idx,pix=pix)]=line_err
                    data['line'].loc[dict(ax='bkg',od=order,idx=idx,pix=pix)]=line_bkg
                    data['line'].loc[dict(ax='x',od=order,idx=idx,pix=pix)]  =line_pix-cen
                    
                    # --------WEIGHTS--------
                    # central 2.5 pixels on each side have weights = 1
                    central_pix = pix[np.where(abs(pix)<=2.5)[0]]
                    data['line'].loc[dict(ax='w',od=order,idx=idx,pix=central_pix)]=1.0
                    # pixels outside of 5.5 have weights = 0
                    outer_pix   = pix[np.where(abs(pix)>=5.5)[0]]
                    data['line'].loc[dict(ax='w',od=order,idx=idx,pix=outer_pix)]=0.0
                    # pixels with 2.5<abs(pix)<5.5 have weights between 0 and 1, linear
                    midleft_pix  = pix[np.where((pix>-5.5)&(pix<-2.5))[0]]
                    midleft_w   = np.array([(x+5.5)/3 for x in midleft_pix])
                    
                    midright_pix = pix[np.where((pix>2.5)&(pix<5.5))[0]]
                    midright_w   = np.array([(-x+5.5)/3 for x in midright_pix])
                    
                    data['line'].loc[dict(ax='w',od=order,idx=idx,pix=midleft_pix)] =midleft_w
                    data['line'].loc[dict(ax='w',od=order,idx=idx,pix=midright_pix)]=midright_w
                    
                    #-------- NORMALISE LINE ------
                    data['line'].loc[dict(ax='y',od=order,idx=idx,pix=pix)]  =(line_flx_nobkg)/np.sum(line_flx_nobkg)
                pbar.update(1)
        pbar.close()
        return data
    def construct_eLSF(self,data):
        n_iter   = self.niter
        orders   = data.coords['od'].values
        segments = np.unique(data.coords['seg'].values)
        N_seg    = len(segments)
        pixels   = data.coords['pix'].values
        N_sub    = round(len(pixels)/(pixels.max()-pixels.min()))
        
        clip     = 2.5
        
        plot = False
        pbar     = tqdm.tqdm(total=(len(orders)*N_seg),desc='Constructing eLSF')
        if plot:
            fig, ax = hf.get_fig_axes(N_seg,alignment='grid')
        for o,order in enumerate(orders):
            for n in segments:
                j = 0
                # select flux data for all lines in the n-th segment and the right order
                # drop all NaN values in pixel and 
                segment = data['line'].sel(sg=n,od=order).dropna('pix','all').dropna('idx','all')
                # extract data in x and y, corresponding coordinates and line idx
                y_data = segment.sel(ax='y')#.dropna('pix','all').dropna('idx','all')
                x_data = segment.sel(ax='x')#.dropna('pix','all').dropna('idx','all')
                x_coords = y_data.coords['pix'].values
                line_idx = [(n,*t) for t in y_data.coords['idx'].values]
                #line_idx = y_data.coords['idx'].values
                # initialise effective LSF of this segment as null values    
                data['elsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] = 0
                data['elsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')] = x_coords
                delta_x    = 0
                sum_deltax = 0
                
                
                while j<n_iter:
                    if np.isnan(delta_x):
                        print("delta_x is NaN!")
                        
                        return data
                    # read the latest eLSF array for this order and segment, drop NaNs
                    elsf_y  = data['elsf'].sel(od=order,seg=n,ax='y',pix=x_coords)
                    elsf_x  = data['elsf'].sel(od=order,seg=n,ax='x',pix=x_coords)
                    # construct the spline using x_coords and current eLSF, 
                    # evaluate eLSF for all points and save values and residuals
                    splr = interpolate.splrep(elsf_x.values,elsf_y.values)                    
                    sple = interpolate.splev(x_data.values,splr)
    #                print(sple)
                    #print(len(line_idx),len(sple))
                    rsd  = (y_data-sple)
                    data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='lsf')] = sple
                    data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='rsd')] = rsd.values
                    # calculate the mean of the residuals between the samplings and eLSF
                    testbins  = np.array([(l,u) for l,u in zip(x_coords-1/N_sub,x_coords+1/N_sub)])
                               
                    
                    rsd_muarr = np.zeros_like(x_coords)
                    rsd_sigarr = np.zeros_like(x_coords)
                    #-------- ITERATIVELY REJECT SAMPLIN MORE THAN 2.5sigma FROM THE MEAN --------
                    for i in range(rsd_muarr.size):
                        llim, ulim = testbins[i]
                        rsd_cut = rsd.where((x_data>llim)&(x_data<=ulim)).dropna('pix','all')     
                        if rsd_cut.size == 0:
                                break
                        sigma_old = 999
                        dsigma    = 999
                        while dsigma>1e-2:
                            mu = rsd_cut.mean(skipna=True).values
                            sigma = rsd_cut.std(skipna=True).values
                            if ((sigma == 0) or (np.isnan(sigma)==True)):
                                break
                            rsd_cut.clip(mu-sigma*clip,mu+sigma*clip).dropna('idx','all')
                            
                            dsigma = (sigma_old-sigma)/sigma
                            sigma_old = sigma
                        rsd_muarr[i]  =   mu
                        rsd_sigarr[i] =   sigma
                    rsd_mean = xr.DataArray(rsd_muarr,coords=[x_coords],dims=['pix']).dropna('pix','all')
                    #rsd_sigma= xr.DataArray(rsd_sigarr,coords=[x_coords],dims=['pix']).dropna('pix','all')
                    #rsd_coords = rsd_mean.coords['pix']
                    #print(rsd_coords==x_coords)
                    # adjust current model of the eLSF by the mean of the residuals
                    data['elsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')]  = x_coords
                    data['elsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] += rsd_mean
                    # re-read the new eLSF model: 
                    elsf_y = data['elsf'].sel(od=order,seg=n,ax='y',pix=x_coords)
                    elsf_x = data['elsf'].sel(od=order,seg=n,ax='x',pix=x_coords)
                    elsf_c = elsf_y.coords['pix']
                    # calculate the derivative of the new eLSF model
                    elsf_der = xr.DataArray(hf.derivative1d(elsf_y.values,elsf_x.values),coords=[elsf_c],dims=['pix'])
                    data['elsf'].loc[dict(od=order,seg=n,ax='der',pix=elsf_c)] =elsf_der
                    # calculate the shift to be applied to all samplings
                    # evaluate at pixel e
                    e = 0.5
                    elsf_neg     = elsf_y.sel(pix=-e,method='nearest').values
                    elsf_pos     = elsf_y.sel(pix=e,method='nearest').values
                    elsf_der_neg = elsf_der.sel(pix=-e,method='nearest').values
                    elsf_der_pos = elsf_der.sel(pix=e,method='nearest').values
                    delta_x      = (elsf_pos-elsf_neg)/(elsf_der_pos-elsf_der_neg)
  
                    if plot:
   
                        ax[n].scatter(elsf_x.values,elsf_y.values,marker='s',s=10,c='C{}'.format(j+1)) 
                        ax[n].axvline(0,ls='--',lw=1,c='C0')
                        ax[n].scatter(x_data.values,y_data.values,s=1,c='C{}'.format(j),marker='s',alpha=0.5)
  
                            
                    j+=1               
                    # shift the sampling by delta_x for the next iteration
                    x_data += delta_x
                    # add delta_x to total shift over all iterations
                    sum_deltax += delta_x
                    if np.isnan(delta_x):
                        
                        print("delta_x is NaN!")
                        print(x_data)
                data['shft'].loc[dict(seg=n,od=order)] = sum_deltax
               
                pbar.update(1)
                
        pbar.close()
        
        return data
    def solve_line_positions(self,data,interpolate_local_psf=True):
        ''' Solves for the flux of the line and the shift (Delta x) from the center
        of the brightest pixel'''
       
        
        orders          = data.coords['od'].values
        pixels          = data.coords['pix'].values
        midx            = data.coords['idx'].values
        segments        = np.unique(data.coords['seg'].values)
        segnum          = len(segments)
        s               = 4096//segnum
        segment_limits  = sl = np.linspace(0,4096,segnum+1)
        segment_centers = sc = (sl[1:]+sl[:-1])/2
        segment_centers[0] = 0
        segment_centers[-1] = 4096
        def residuals(x0,pixels,counts,weights,background,splr):
            # center, flux
            sft, flux = x0
            model = flux * interpolate.splev(pixels+sft,splr)
            # sigma_tot^2 = sigma_counts^2 + sigma_background^2
            # sigma_counts = sqrt(counts)     sigma_background = sqrt(background)
            error = np.sqrt(counts + background)
            resid = np.sqrt(line_w) * ((counts-background) - model) / error
            #resid = line_w * (counts- model)
            return resid
            
        for order in orders:
            for idx in midx:
                sg,sp,lid = idx
                line_pars = data['pars'].sel(idx=idx,od=order).values
                cen,cen_err, flx, flx_err, dx, phi, b, cen_1g = line_pars
                p0 = (dx,flx)
                if np.isnan(p0).any() == True:
                    continue
                line   = data['line'].sel(idx=idx,od=order)
                line_x = line.sel(ax='pos').dropna('pix')
                lcoords=line_x.coords['pix']
                line_x = line_x.values
                line_y = line.sel(ax='flx').dropna('pix').values
                line_b = line.sel(ax='bkg').dropna('pix').values
                line_w = line.sel(ax='w').dropna('pix').values
                if ((len(line_x)==0)or(len(line_y)==0)or(len(line_w)==0)):
                    continue
                cen_pix = line_x[np.argmax(line_y)]  
                
                elsf_x  = data['elsf'].sel(ax='x',od=order,seg=sg).dropna('pix')+cen_pix
                
                #---------- CONSTRUCT A LOCAL LSF ----------
                # find in which segment the line falls
                sg2     = np.digitize(cen_pix,segment_centers)
                sg1     = sg2-1
                elsf1   = data['elsf'].sel(ax='y',od=order,seg=sg1).dropna('pix') 
                elsf2   = data['elsf'].sel(ax='y',od=order,seg=sg2).dropna('pix')
                if interpolate_local_psf:
                    f1 = (sc[sg2]-cen_pix)/(sc[sg2]-sc[sg1])
                    f2 = (cen_pix-sc[sg1])/(sc[sg2]-sc[sg1])
                    elsf_y  = f1*elsf1 + f2*elsf2  
                else:
                    elsf_y  = data['elsf'].sel(ax='y',od=order,seg=sg).dropna('pix') 
                elsf_x  = elsf_x.sel(pix=elsf_y.coords['pix'])
                splr    = interpolate.splrep(elsf_x.values,elsf_y.values)
                popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                                                       args=(line_x,line_y,line_w,line_b,splr),
                                                       full_output=True)
                
                if ier not in [1, 2, 3, 4]:
                    print((3*("{:<3d}")).format(*idx),"Optimal parameters not found: " + errmsg)
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
                    #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
                else:
                    continue
                #print('CHISQ = {0:15.5f}'.format(cost/dof))
                cen = line_x[np.argmax(line_y)]-sft
                cen_err, flx_err = [np.sqrt(pcov[i][i]) for i in range(2)]
                phi = cen - int(cen+0.5)
                save_pars = np.array([cen,cen_err,flx,flx_err, sft,phi,b,cen_1g])
                data['pars'].loc[dict(idx=idx,od=order)] = save_pars
                data['line'].loc[dict(idx=idx,od=order,ax='lsf',pix=elsf_y.coords['pix'])]=elsf_y
                
                # calculate residuals:
                model = flx * interpolate.splev(line_x+sft,splr) 
                resid = (line_y-line_b) - model
                data['line'].loc[dict(idx=idx,od=order,ax='mod',pix=lcoords)]=model
                data['line'].loc[dict(idx=idx,od=order,ax='rsd',pix=lcoords)]=resid
                
        return data
    def get_filepath(self,order,fibre):
        dirpath  = os.path.join(self.savedir,'fibre{}'.format(fibre))
        basepath = '{LFC}_fib{fb}{sh}_order_{od}.nc'.format(LFC=self.LFC,
                                                         fb=fibre,
                                                         sh=self.fibreshape,
                                                         od=order)
        filepath = os.path.join(dirpath,basepath)
        return filepath
    def save2file(self,data,fibre):
#        fibres = self.fibres
        order = int(data.coords['od'].values)
        filepath = self.get_filepath(order,fibre)
        data4file = data.unstack('idx')
        data4file.attrs['LFC'] = self.LFC
        data4file.attrs['fibreshape'] = self.fibreshape
        data4file.attrs['interpolate'] = int(self.interpolate)
        data4file.attrs['fit_gaussians'] = int(self.fit_gaussians)
        data4file.to_netcdf(filepath)
        print("Saved to {}".format(filepath))
        
        return
###############################################################################
##############################   PLOTTER   ####################################
###############################################################################
class SpectrumPlotter(object):
    def __init__(self,naxes=1,ratios=None,title=None,sep=0.05,figsize=(16,9),
                 alignment="vertical",sharex=None,sharey=None,**kwargs):
        fig, axes = hf.get_fig_axes(naxes,ratios=ratios,title=title,
                                 sep=sep,alignment=alignment,
                                 figsize=figsize,sharex=sharex,
                                 sharey=sharey,**kwargs)
        self.figure = fig
        self.axes   = axes   
    def show(self):
        self.figure.show()
        return
class LSFPlotter(object):
    def __init__(self,filepath):
        lsf = xr.open_dataset(filepath)
        self.lsf = lsf
        
    def initilize_plotter(self,naxes=1,ratios=None,title=None,sep=0.05,figsize=(9,9),
                 alignment="vertical",sharex=None,sharey=None,**kwargs):
        fig, axes = hf.get_fig_axes(naxes,ratios=ratios,title=title,
                                 sep=sep,alignment=alignment,
                                 figsize=figsize,sharex=sharex,
                                 sharey=sharey,**kwargs)
        self.figure = fig
        self.axes   = axes  
        
        return
    def plot_epsf(self,order=None,seg=None,plot_lsf=True,plot_points=False,fig=None,):
        ''' Plots the full data, including the lines used for reconstruction'''
        data = self.lsf
        if order is None:
            orders   = data.coords['od'].values
        else:
            orders   = hf.to_list(order)
        print(orders)
        if seg is None:    
            segments = np.unique(data.coords['seg'].values)
        else:
            segments = hf.to_list(seg)
        ids = np.unique(data.coords['id'].values)
        #sgs = np.unique(data.coords['sg'].values)
        sgs = segments
        sps = np.unique(data.coords['sp'].values)
        
        midx  = pd.MultiIndex.from_product([sgs,sps,np.arange(60)],
                                names=['sg','sp','id'])
        if fig is None:
            fig,ax = hf.get_fig_axes(1,figsize=(9,9))
        else:
            fig = fig
            ax  = fig.get_axes()
        for order in orders:

            
            for s in sgs:
                if plot_points:
                    data_s = data['shft'].sel(od=order,seg=s)
                    for lid in range(60):
                        data_x = data['line'].sel(ax='x',od=order,sg=s,id=lid).dropna('pix','all')
                        pix = data_x.coords['pix']
                        if np.size(pix)>0:
                            data_y = data['line'].sel(ax='y',od=order,sg=s,id=lid,pix=pix)
                            print(np.shape(data_x),np.shape(data_y),np.shape(data_s))
                            ax[0].scatter(data_x+data_s,data_y,s=5,c='C0',marker='s',alpha=0.3)
                        else:
                            continue
                if plot_lsf:
                    epsf_x = data['epsf'].sel(ax='x',od=order,seg=s).dropna('pix','all')
                    epsf_y = data['epsf'].sel(ax='y',od=order,seg=s).dropna('pix','all')
                    ax[0].scatter(epsf_x,epsf_y,marker='x',c='C1') 
        ax[0].set_xlabel('Pixel')
        ax[0].set_yticks([])
        return fig 
    def plot_psf(self,psf_ds,order=None,fig=None,**kwargs):
        '''Plots only the LSF'''
        if order is not None:
            orders = hf.to_list(order)
        else:
            orders   = psf_ds.coords['od'].values
        segments = np.unique(psf_ds.coords['seg'].values)
        # provided figure?
        if fig is None:
            fig,ax = hf.get_fig_axes(len(segments),alignment='grid')
        else:
            fig = fig
            ax  = fig.get_axes()
        # colormap
        if len(orders)>5:
            cmap = plt.get_cmap('jet')
        else:
            cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0,1,len(orders)))
        # line parameters
        ls = kwargs.pop('ls','')
        lw = kwargs.pop('lw',0.3)
        for o,order in enumerate(orders):
            for n in segments:
                epsf_y = psf_ds.sel(ax='y',seg=n,od=order).dropna('pix','all')
                epsf_x = psf_ds.sel(ax='x',seg=n,od=order).dropna('pix','all')
                #print(epsf_y, epsf_x)
                ax[n].plot(epsf_x,epsf_y,c=colors[o],ls=ls,lw=lw,marker='x',ms=3) 
                ax[n].set_title('{0}<pix<{1}'.format(n*256,(n+1)*256), )
        return fig 
    def to_list(self,item):
        if type(item)==str:
            items = [item]
        elif type(item) == list:
            items = item
        elif item is None:
            items = []
        elif type(item)==int:
            items = [item]
        else:
            print("Type provided {}".format(type(item)))
        return items
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
###############################################################################
###########################   MISCELANEOUS   ##################################
###############################################################################
class mimicKwargs(object):
    def __init__(self, labelLong, labelShort, defautValue,kwargDic):
        if kwargDic.get(labelLong) and kwargDic.get(labelShort):
            warnings.warn("same flag used two times")
        else:
            self.userInput = kwargDic.get(labelLong) or kwargDic.get(labelShort) or defautValue
    def output(self):
        return self.userInput
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
def wrap_fit_epsf(pars):
    return fit_epsf(*pars)
def fit_epsf(line,psf,pixPerLine):
    def residuals(x0,pixels,counts,weights,background,splr):
        ''' Model parameters are estimated shift of the line center from 
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
        # sigma_tot^2 = sigma_counts^2 + sigma_background^2
        # sigma_counts = sqrt(counts)     sigma_background = sqrt(background)
        error = np.sqrt(counts + background)
        resid = np.sqrt(weights) * ((counts-background) - model) / error
        #resid = line_w * (counts- model)
        return resid
    def get_local_psf(pix,order,seg):
        ''' Returns local ePSF at a given pixel of the echelle order
        '''
        segments        = np.unique(psf.coords['seg'].values)
        N_seg           = len(segments)
        # segment limits
        sl              = np.linspace(0,4096,N_seg+1)
        # segment centers
        sc              = (sl[1:]+sl[:-1])/2
        sc[0] = 0
        sc[-1] = 4096
       
        def return_closest_segments(pix):
            sg_right  = int(np.digitize(pix,sc))
            sg_left   = sg_right-1
            return sg_left,sg_right
        
        sgl, sgr = return_closest_segments(pix)
        f1 = (sc[sgr]-pix)/(sc[sgr]-sc[sgl])
        f2 = (pix-sc[sgl])/(sc[sgr]-sc[sgl])
        
        #epsf_x  = psf.sel(ax='x',od=order,seg=seg).dropna('pix')+pix
        epsf_1 = psf.sel(ax='y',od=order,seg=sgl).dropna('pix')
        epsf_2 = psf.sel(ax='y',od=order,seg=sgr).dropna('pix')
        
        epsf_y = f1*epsf_1 + f2*epsf_2 
       
        xc     = epsf_y.coords['pix']
        if len(xc)==0:
            print(lid,"No pixels in xc, ",len(xc))
            print(epsf_1.coords['pix'])
            print(epsf_2.coords['pix'])
#            from IPython.core.debugger import Tracer
#
#            print(lid,psf_x)
#            print(lid,psf_y)
#            Tracer()()
        epsf_x  = psf.sel(ax='x',od=order,seg=seg,pix=xc)+pix
        #qprint(epsf_x,epsf_y)f
        return epsf_x, epsf_y
    # MAIN PART 
    
    line      = line.dropna('pid','all')
    pid       = line.coords['pid']
    lid       = int(line.coords['id'])
    order     = int(line.coords['od'])
    line_x    = line['line'].sel(ax='pix')
    line_y    = line['line'].sel(ax='flx')
    line_w    = line['line'].sel(ax='wgt')
    line_bkg  = line['line'].sel(ax='bkg')
    line_bary = line['attr'].sel(att='bary')
    cen_pix   = line_x[np.argmax(line_y)]
    loc_seg   = line['attr'].sel(att='seg')
    freq      = line['attr'].sel(att='freq')
    #lbd       = line['attr'].sel(att='lbd')
    
    # get local PSF and the spline representation of it
    psf_x, psf_y = get_local_psf(cen_pix,order=order,seg=loc_seg)
    try:
        psf_rep  = interpolate.splrep(psf_x,psf_y)
    except:
        from IPython.core.debugger import Tracer
        print(lid,psf_x)
        print(lid,psf_y)
        Tracer()()
        
    
    # fit the line for flux and position
    par_arr     = hf.return_empty_dataarray('pars',order,pixPerLine)
    mod_arr     = hf.return_empty_dataarray('model',order,pixPerLine)
    p0 = (-1e-1,np.percentile(line_y,80))
#            print(line_x,line_y,line_w)
#            print(line_b,p0)
    popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                            args=(line_x,line_y,line_w,line_bkg,psf_rep),
                            full_output=True)
    cen, flx = popt
    line_model = flx * interpolate.splev(line_x+cen,psf_rep) + line_bkg
    if ier not in [1, 2, 3, 4]:
        print("Optimal parameters not found: " + errmsg)
        popt = np.full_like(p0,np.nan)
        pcov = None
        success = False
    else:
        success = True
    if success:
        
        sft, flx = popt
        cost   = np.sum(infodict['fvec']**2)
        dof    = (len(line_x) - len(popt))
        rchisq = cost/dof
        if pcov is not None:
            pcov = pcov*rchisq
        else:
            pcov = np.array([[np.inf,0],[0,np.inf]])
        cen              = line_x[np.argmax(line_y)]-sft
        cen_err, flx_err = [np.sqrt(pcov[i][i]) for i in range(2)]
        #phi              = cen - int(cen+0.5)
        b                = line_bary
        pars = np.array([cen,sft,cen_err,flx,flx_err,rchisq,np.nan,np.nan])
    else:
        pars = np.full(len(hf.fitPars),np.nan)
    # pars: ['cen','cen_err','flx','flx_err','chisq','rsd']
    # attr: ['bary','freq','freq_err','lbd','seg']
    
    
    # Save all the data back
    par_arr.loc[dict(od=order,id=lid,ft='epsf')] = pars
    mod_arr.loc[dict(od=order,id=lid,
                     pid=line_model.coords['pid'],ft='epsf')] = line_model

    return par_arr,mod_arr
def wrap_fit_single_line(pars):
    return fit_single_line(*pars)
def fit_single_line(line,psf,pixPerLine):
    def residuals(x0,pixels,counts,weights,background,splr):
        ''' Model parameters are estimated shift of the line center from 
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
        # sigma_tot^2 = sigma_counts^2 + sigma_background^2
        # sigma_counts = sqrt(counts)     sigma_background = sqrt(background)
        error = np.sqrt(counts + background)
        resid = np.sqrt(weights) * ((counts-background) - model) / error
#            resid = counts/np.sum(counts) * ((counts-background) - model) / error
        #resid = line_w * (counts- model)
        return resid
    def get_local_psf(pix,order,seg,mixing=True):
        ''' Returns local ePSF at a given pixel of the echelle order
        '''
        #print(pix,order,seg)
        segments        = np.unique(psf.coords['seg'].values)
        N_seg           = len(segments)
        seg             = int(seg)
        # segment limits
        sl              = np.linspace(0,4096,N_seg+1)
        # segment centers
        sc              = (sl[1:]+sl[:-1])/2
        sc[0] = 0
        sc[-1] = 4096
       
        def return_closest_segments(pix):
            sg_right  = int(np.digitize(pix,sc))
            sg_left   = sg_right-1
            return sg_left,sg_right
        
        sgl, sgr = return_closest_segments(pix)
        f1 = (sc[sgr]-pix)/(sc[sgr]-sc[sgl])
        f2 = (pix-sc[sgl])/(sc[sgr]-sc[sgl])
        
        #epsf_x  = psf.sel(ax='x',od=order,seg=seg).dropna('pix')+pix
        
        epsf_1 = psf.sel(ax='y',od=order,seg=sgl).dropna('pix')
        epsf_2 = psf.sel(ax='y',od=order,seg=sgr).dropna('pix')
        
        if mixing == True:
            epsf_y = f1*epsf_1 + f2*epsf_2 
        else:
            epsf_y = epsf_1
        
        xc     = epsf_y.coords['pix']
        epsf_x  = psf.sel(ax='x',od=order,seg=seg,pix=xc)+pix
        #print(epsf_x.values,epsf_y.values)
        return epsf_x, epsf_y
    # MAIN PART 
    
    
    # select single line
    #lid       = line_id
    line      = line.dropna('pid','all')
    pid       = line.coords['pid']
    lid       = int(line.coords['id'])

    line_x    = line['line'].sel(ax='pix')
    line_y    = line['line'].sel(ax='flx')
    line_w    = line['line'].sel(ax='wgt')
    #print("Read the data for line {}".format(lid))
    # fitting fails if some weights are NaN. To avoid this:
    weightIsNaN = np.any(np.isnan(line_w))
    if weightIsNaN:
        whereNaN  = np.isnan(line_w)
        line_w[whereNaN] = 0e0
        #print('Corrected weights')
    line_bkg  = line['line'].sel(ax='bkg')
    line_bary = line['attr'].sel(att='bary')
    cen_pix   = line_x[np.argmax(line_y)]
    #freq      = line['attr'].sel(att='freq')
    #print('Attributes ok')
    #lbd       = line['attr'].sel(att='lbd')
    # get local PSF and the spline representation of it
    order        = int(line.coords['od'])
    loc_seg      = line['attr'].sel(att='seg')
    psf_x, psf_y = get_local_psf(line_bary,order=order,seg=loc_seg)
    
    psf_rep  = interpolate.splrep(psf_x,psf_y)
    #print('Local PSF interpolated')
    # fit the line for flux and position
    #arr    = hf.return_empty_dataset(order,pixPerLine)
    
    par_arr     = hf.return_empty_dataarray('pars',order,pixPerLine)
    mod_arr     = hf.return_empty_dataarray('model',order,pixPerLine)
    p0 = (5e-1,np.percentile(line_y,90))

    
    # GAUSSIAN ESTIMATE
    g0 = (np.nanpercentile(line_y,90),float(line_bary),1.3)
    gausp,gauscov=curve_fit(hf.gauss3p,p0=g0,
                        xdata=line_x,ydata=line_y)
    Amp, mu, sigma = gausp
    p0 = (0.01,Amp)
    popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                            args=(line_x,line_y,line_w,line_bkg,psf_rep),
                            full_output=True,
                            ftol=1e-5)
    
    if ier not in [1, 2, 3, 4]:
        print("Optimal parameters not found: " + errmsg)
        popt = np.full_like(p0,np.nan)
        pcov = None
        success = False
    else:
        success = True
   
    if success:
        
        sft, flux = popt
        line_model = flux * interpolate.splev(line_x+sft,psf_rep) + line_bkg
        cost   = np.sum(infodict['fvec']**2)
        dof    = (len(line_x) - len(popt))
        rchisq = cost/dof
        if pcov is not None:
            pcov = pcov*rchisq
        else:
            pcov = np.array([[np.inf,0],[0,np.inf]])
        cen              = cen_pix+sft
        cen              = line_bary - sft
        cen_err, flx_err = [np.sqrt(pcov[i][i]) for i in range(3)]
        sigma
#        pars = np.array([cen,sft,cen_err,flux,flx_err,rchisq,np.nan,np.nan])
        pars = np.array([cen,cen_err,flx,flx_err,sigma,sigma_err,rchi2])
    else:
        pars = np.full(len(hf.fitPars),np.nan)
   
    par_arr.loc[dict(od=order,id=lid,ft='epsf')] = pars
    mod_arr.loc[dict(od=order,id=lid,
                     pid=line_model.coords['pid'],ft='epsf')] = line_model

    return par_arr,mod_arr
def wrap_calculate_line_weights(pars):
    return calculate_line_weights(*pars)
def calculate_line_weights(subdata,psf,pixPerLine):
    '''
    Uses the barycenters of lines to populate the weight axis 
    of data['line']
    '''
    
    order  = int(subdata.coords['od'])
    
    # read PSF pixel values and create bins
    psfPixels    = psf.coords['pix']
    psfPixelBins = (psfPixels[1:]+psfPixels[:-1])/2
    
    # create container for weights
    linesID      = subdata.coords['id']
    # shift line positions to PSF reference frame
   
    linePixels0 = subdata['line'].sel(ax='pix') - \
                  subdata['attr'].sel(att='bary')
    arr = hf.return_empty_dataset(order,pixPerLine)
    for lid in linesID:                    
        line1d = linePixels0.sel(id=lid).dropna('pid')
        if len(line1d) == 0:
            continue
        else:
            pass
        weights = xr.DataArray(np.full_like(psfPixels,np.nan),
                               coords=[psfPixels.coords['pix']],
                               dims = ['pid'])
        # determine which PSF pixel each line pixel falls in
        dig = np.digitize(line1d,psfPixelBins,right=True)
        
        pix = psfPixels[dig]
        # central 2.5 pixels on each side have weights = 1
        central_pix = pix[np.where(abs(pix)<=2.5)[0]]
        # pixels outside of 5.5 have weights = 0
        outer_pix   = pix[np.where(abs(pix)>=4.5)[0]]
        # pixels with 2.5<abs(pix)<5.5 have weights between 0 and 1, linear
        midleft_pix  = pix[np.where((pix>=-4.5)&(pix<-2.5))[0]]
        midleft_w   = np.array([(x+5.5)/3 for x in midleft_pix])
        
        midright_pix = pix[np.where((pix>2.5)&(pix<=4.5))[0]]
        midright_w   = np.array([(-x+5.5)/3 for x in midright_pix])
        
        weights.loc[dict(pid=central_pix)] =1.0
        weights.loc[dict(pid=outer_pix)]   =0.0
        weights.loc[dict(pid=midleft_pix)] =midleft_w
        weights.loc[dict(pid=midright_pix)]=midright_w
        #print(weights.values)
        weights = weights.dropna('pid')
        #print(len(weights))
        sel = dict(od=order,id=lid,ax='wgt',pid=np.arange(len(weights)))
        arr['line'].loc[sel]=weights.values
    return arr['line'].sel(ax='wgt')
def wrap_detect_order(pars):
    return detect_order(*pars) 
def detect_order(subdata,f0_comb,reprate,segsize,pixPerLine,window):
    #print("f0={0:>5.1f} GHz\tfr={1:>5.1} GHz".format(f0_comb/1e9,reprate/1e9))
    # debugging
    verbose=1
    plot=0
    # main
    order  = int(subdata.coords['od'])
    arr    = hf.return_empty_dataset(order,pixPerLine)
    spec1d = subdata.sel(ax='flx')
    bkg1d  = subdata.sel(ax='bkg')
    err1d  = subdata.sel(ax='err')
    pixels = np.arange(4096)
    wave1d = subdata.sel(ax='wave')
    #print("Order = {} \t Wave1d.sum = {}".format(order,wave1d.sum().values))
    # wavelength solution exists?
    if wave1d.sum()==0:
        warnings.warn("ThAr WAVELENGTH SOLUTION DOES NOT EXIST")
        return arr
    # photon noise
    sigma_v= subdata.sel(ax='sigma_v')
    pn_weights = (sigma_v/299792458e0)**-2
    
    
    # 
    yarray = spec1d-bkg1d
    minima = hf.peakdet(yarray,pixels,extreme='min',
                        method='peakdetect_derivatives',window=window)
    # number of lines is the number of minima detected - 1
    npeaks1 = len(minima.x)-1
    
#    maxima = hf.peakdet(yarray,pixels,extreme='max',
#                        method='peakdetect_derivatives',window=window)
#    
#    # use only lines with flux in maxima > fluxlim
#    fluxlim = 3e3
    #maxima = maxima.where(maxima.y>fluxlim).dropna()

    first  = int(round((minima.x.iloc[-1]+minima.x.iloc[-2])/2))
    last   = int(round((minima.x.iloc[0]+minima.x.iloc[1])/2))

    nu_min  = (299792458e0/(wave1d[first]*1e-10)).values
    nu_max  = (299792458e0/(wave1d[last]*1e-10)).values
    
    n_start = int(round((nu_min - f0_comb)/reprate))
    n_end   = int(round((nu_max - f0_comb)/reprate))
    npeaks2  = (n_end-n_start)+1
    
    if plot:
        fig,ax = hf.figure(2,sharex=True,figsize=(12,6))
        ax[1].plot((0,4096),(0,0),ls='--',c='k',lw=0.5)
        ax[0].plot(pixels,yarray)
    # do quality control if npeaks1 != npeaks2
    
    if verbose>1:
        message=('Order = {2:>2d} '
                 'detected = {0:>8d} '
                 'inferred = {1:>8d}').format(npeaks1,npeaks2,order)
        print(message)
    if npeaks1!=npeaks2:
        delta = abs(npeaks1-npeaks2)
        if delta>50:
            raise UserWarning("{} lines difference. Wrong LFC?".format(delta))
        if npeaks1>npeaks2:
            if verbose>0:
                warnings.warn('{0:3d} more lines detected than inferred.'
                              ' Order={1:2d}'.format(npeaks1-npeaks2,order))
            # look for outliers in the distance between positions of minima
            # the difference should be a smoothly varying function of pixel 
            # number (modelled as a polynomial function). 
            oldminima = minima
            xpos = oldminima.x
            diff = np.diff(xpos)
            pars = np.polyfit(xpos[1:],diff,4)
            model = np.polyval(pars,xpos[1:])
            resids = diff-model
            outliers1 = resids<-5
            if plot:
                linepos2=((oldminima.x+np.roll(oldminima.x,1))/2)[1:]
                [ax[0].axvline(lp,lw=0.3,ls=':',c='C2') for lp in linepos2]
                ax[1].scatter(xpos[1:],resids,c='C0')
                ax[1].scatter(xpos[1:][outliers1],resids[outliers1],
                              marker='x',c='r')
            # make outliers a len(xpos) array
            outliers2 = np.insert(outliers1,0,False)
            newminima = (oldminima[~outliers2])
            minima = newminima.reset_index()
            npeaks1=len(minima.x)-1
            if verbose>0:
                message=('CORRECTED detected = {0:>8d} '
                         'inferred = {1:>8d}').format(npeaks1,npeaks2)
                print(message)
        elif npeaks1<npeaks2:
            if verbose>0:
                warnings.warn('{0:3d} fewer lines detected than inferred.'
                              ' Order={1:2d}'.format(npeaks2-npeaks1,order))
        npeaks = min(npeaks1,npeaks2)
    else:
        npeaks=npeaks2
    if plot:
        ax[0].plot(minima.x,minima.y,ls='',marker='x',markersize=2,c='r')
        #[0].plot(maxima.x,maxima.y,ls='',marker='x',markersize=2,c='g')
        linepos=((minima.x+np.roll(minima.x,1))/2)[1:]
        [ax[0].axvline(lp,lw=0.3,ls=':',c='C1') for lp in linepos]
        
        
    
    
    freq1d_asc  = np.array([(f0_comb+(n_start+j)*reprate) \
                         for j in range(npeaks2)])
    # in decreasing order (wavelength increases for every element, i.e. 
    # every element is redder)
    freq1d_dsc  = np.array([(f0_comb+(n_end-j)*reprate) \
                         for j in range(npeaks2)])
    
    freq1d=freq1d_asc
    
    # iterate over lines
    for i in range(0,npeaks-1,1):
        if verbose>3:
            print(i,len(minima.x))
        # array of pixels
        lpix, upix = (minima.x[i],minima.x[i+1])
        pix  = np.arange(lpix,upix,1,dtype=np.int32)
        # sometimes the pix array covers more than can fit into the arr container
        # trim it on both sides until it fits
        if len(pix)>pixPerLine:
            k = 0
            while len(pix)>pixPerLine:
                pix = np.arange(lpix+k,upix-k,dtype=np.int32)
                k+=1
        # flux, background, flux error
        flux = spec1d[pix]
        bkg  = bkg1d[pix]
        err  = err1d[pix]

        # save values
        val  = {'pix':pix, 
                'flx':flux,
                'bkg':bkg,
                'err':err}
        for ax in val.keys():
            idx  = dict(id=i,pid=np.arange(pix.size),ax=ax)
            try:
                arr['line'].loc[idx] = val[ax]
            except:
                print(np.arange(pix.size))
                print(arr['line'].coords['pid'])
        # barycenter, segment
        bary = np.sum(flux*pix)/np.sum(flux)
        cen_pix = pix[np.argmax(flux)]
        local_seg = cen_pix//segsize
        # photon noise
        sumw = np.sum(pn_weights[pix])
        pn   = (299792458e0/np.sqrt(sumw)).values
        # signal to noise ratio
        snr = np.sum(flux)/np.sum(err)
        # frequency of the line
        center  = int(round((lpix+upix)/2))
        freq0   = (299792458e0/(wave1d[center]*1e-10)).values
        n       = int(round((freq0 - f0_comb)/reprate))
        freq    = f0_comb + n*reprate
        
        arr['attr'].loc[dict(id=i,att='pn')]  = pn
        arr['attr'].loc[dict(id=i,att='freq')]= freq
        arr['attr'].loc[dict(id=i,att='seg')] = local_seg
        arr['attr'].loc[dict(id=i,att='bary')]= bary
        arr['attr'].loc[dict(id=i,att='snr')] = snr
        # calculate weights in a separate function
    # save the total flux in the order
    arr['stat'].loc[dict(od=order,odpar='sumflux')] = np.sum(spec1d)
    return arr