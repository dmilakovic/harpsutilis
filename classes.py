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

from glob import glob
from astropy.io import fits
from scipy.optimize import curve_fit, leastsq
from scipy import odr, interpolate
from joblib import Parallel,delayed


import harps.functions as funcs
import harps.settings as settings

harps_home   = settings.harps_home
harps_data   = settings.harps_data
harps_dtprod = settings.harps_dtprod
harps_plots  = settings.harps_plots
harps_prod   = settings.harps_prod

sOrder       = settings.sOrder
eOrder       = settings.eOrder
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
        self.polyord    = 8
        
        self.lineDetectionPerformed=False
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
    def return_empty_dataset(self,order=None):
        linesPerOrder = 400
        if self.LFC == 'HARPS':
            pixPerLine    = 22
        elif self.LFC == 'FOCES':
            pixPerLine    = 35
        lineAxes      = ['pix','flx','bkg','err','rsd','wgt','mod','wave']
        linePars      = ['bary','cen','cen_err','flx','flx_err',
                         'freq','freq_err','chisq','seg']
        if order is None:
            shape_data    = (linesPerOrder,len(lineAxes),pixPerLine)
            shape_pars    = (linesPerOrder,len(linePars))
            data_vars     = {'line':(['id','ax','pid'],np.full(shape_data,np.nan)),
                             'pars':(['id','par'],np.full(shape_pars,np.nan))}
            data_coords   = {'id':np.arange(linesPerOrder),
                             'pid':np.arange(pixPerLine),
                             'ax':lineAxes,
                             'par':linePars}
        else:
            orders        = self.prepare_orders(order)
            
            shape_data    = (len(orders),linesPerOrder,len(lineAxes),pixPerLine)
            shape_pars    = (len(orders),linesPerOrder,len(linePars))
            data_vars     = {'line':(['od','id','ax','pid'],np.full(shape_data,np.nan)),
                             'pars':(['od','id','par'],np.full(shape_pars,np.nan))}
#            if len(orders) ==1: orders = orders[0]
            data_coords   = {'od':orders,
                             'id':np.arange(linesPerOrder),
                             'pid':np.arange(pixPerLine),
                             'ax':lineAxes,
                             'par':linePars}
        dataset       = xr.Dataset(data_vars,data_coords)
        self.linesPerOrder = linesPerOrder
        self.pixPerLine    = pixPerLine
        return dataset
    def check_and_get_wavesol(self,calibrator='LFC',orders=None):
        ''' Check and retrieve the wavelength calibration'''
        wavesol_name = 'wavesol_{cal}'.format(cal=calibrator)
        exists_calib = False if getattr(self,wavesol_name) is None else True
        if calibrator=='thar': calibrator='ThAr'
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
                # run line detection
                lines = self.detect_lines(order=orders)
        return lines
    def check_and_initialize_lines(self):
        existLines = True if self.lines is not None else False
        if not existLines:
            order = self.prepare_orders(None)
            lines = self.return_empty_dataset(order)
        
            self.lines = lines
        else:
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
                        orders=None,method='erfc',model=None,
                        patches=False,gaps=False,
                        polyord=4,**kwargs):
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
        def patch_fit(patch,polyord,method='curve_fit'):
            ''' Fits a given patch with a polynomial function'''
            pix     = patch.sel(par='cen')
            pix_err = patch.sel(par='cen_err')
            freq    = patch.sel(par='freq')
            freq_err= patch.sel(par='freq_err')
            lbd     = 299792458e0/freq*1e10
            lbd_err = 299792458e0/freq_err*1e10
#            print(data, data_err)
#            print(x,x_err)  
              
            data_axis = np.array(lbd.values,dtype=np.float64)
            data_err  = np.array(lbd_err.values,dtype=np.float64)
            x_axis    = np.array(pix.values,dtype=np.float64)
            x_err     = np.array(pix_err.values,dtype=np.float64)
            datanan,xnan = (np.isnan(data_axis).any(),np.isnan(x_axis).any())
            if (datanan==True or xnan==True):
                print("NaN values in data or x")
            if x_axis.size>self.polyord:
                coef = np.polyfit(x_axis,data_axis,self.polyord)
                if method == 'curve_fit':
                    coef,pcov = curve_fit(funcs.polynomial,x_axis,data_axis,p0=coef[::-1])
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
        
        def fit_wavesol(lines_in_order,patches=True):
            # perform the fitting in patches?
            # npt = number of patches
            if patches==True:
                npt = 8
            else:
                npt = 1
            ps = 4096/npt
            
            numlines = len(lines_in_order.coords['id'])
            # extract fitted line positions and errors
            pix     = lines_in_order.sel(par='cen')
            pix_err = lines_in_order.sel(par='cen_err')
            freq    = lines_in_order.sel(par='freq')
            lbd     = 299792458e0/freq*1e10
            
            
            ws     = np.zeros(self.npix)
            # coefficients and residuals
            cf = np.zeros(shape=(npt,self.polyord+1))
#            rs = pd.Series(index=np.arange(self.npix))
            rs = xr.DataArray(np.full_like(pix,np.nan),
                              coords=[np.arange(numlines)],
                              dims = ['id'])
            
            # do fit for each patch
            for i in range(npt):
                # lower and upper limit in pixel for the patch
                ll,ul     = np.array([i*ps,(i+1)*ps],dtype=np.int)
                # select lines in this pixel range
                patch     = lines_in_order.where((pix>=ll)&
                                     (pix<ul)).dropna('id','all')
                patch_id  = patch.coords['id']
                # polynomial order must be lower than the number of points
                # used for fitting
                if patch.size>self.polyord:
                    pixels        = np.arange(ll,ul,1,dtype=np.int)
                    coef,coef_err = patch_fit(patch,self.polyord)
                    if coef is not None:
                        fit_lbd   = np.polyval(coef,patch.sel(par='cen'))
                        patch_lbd = 299792458e0/patch.sel(par='freq')*1e10
                        resid     = (patch_lbd.values-fit_lbd)/patch_lbd.values*299792458e0
                        rs.loc[dict(id=patch_id)] = np.array(resid,dtype=np.float64)
                        
                        #rs.iloc[patch.index]=residuals
#                        outliers  = funcs.is_outlier(residuals,5)
#                        if np.any(outliers)==True:
#                            patch['outlier']=outliers
#                            newpatch = patch.where(patch.outlier==False).dropna(how='any')
#                            coef,coef_err = patch_fit(newpatch,self.polyord) 
                        cf[i,:]=coef
                else:
                    ws[ll:ul] = np.nan
                    
                try:
                    ws[ll:ul] = np.polyval(coef,pixels)
                except:
                    ws[ll:ul] = np.nan
            #fit = np.polyval(coef,pix)
#            print(rs,fit)
            # residuals are in m/s
            #rs = rs/fit*299792458
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
                wavecoeff_vac,covariance = curve_fit(funcs.polynomial, 
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
            lines = self.fit_lines2d(orders)
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
                lines_in_order = lines['pars'].sel(od=order).dropna('id','all')#     = self.fit_lines(order,scale='pixel',method=method)
                
                # STOPPED HERE 22nd MARCH 2018
                
                # Include the gaps
                if gaps is True:
                    g0 = self.gaps[order,:]
                    new_cen = self.introduce_gaps(lines_in_order.sel(par='cen'),g0)
                    lines_in_order.loc[dict(par='cen')] = new_cen
                elif gaps is False:
                    pass
  
                LFC_wavesol,coef,residuals = fit_wavesol(
                                               lines_in_order,
                                               patches=patches
                                               )
                
                wavesol_LFC[order]  = LFC_wavesol
                wavecoef_LFC[order] = coef.T     
                ids                 = residuals.coords['id']
                lines['pars'].loc[dict(od=order,id=ids,par='rsd')] = residuals
            self.wavesol_LFC  = wavesol_LFC
            #self.lines        = cc_data
            self.wavecoef_LFC = wavecoef_LFC
            #self.residuals    = rsd
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
            minima  = funcs.peakdet(yarray,xarray,extreme='min')
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
    def detect_lines(self,order=None,calculate_weights=True):
        '''
        This method determines the minima of flux between the LFC lines
        and updates self.lines with the position, flux, background, flux error
        and barycenter of each line.
        '''
        
        def calculate_line_weights(orders):
            '''
            Uses the barycenters of lines to populate the weight axis 
            of data['line']
            '''
            # read PSF pixel values and create bins
            psfPixels    = self.psf.coords['pix']
            psfPixelBins = (psfPixels[1:]+psfPixels[:-1])/2
            
            # create container for weights
            linesID      = self.lines.coords['id']
            # shift line positions to PSF reference frame
           
            linePixels0 = lines['line'].sel(ax='pix') - \
                          lines['pars'].sel(par='bary')
            for od in orders:
                for lid in linesID:                    
                    line1d = linePixels0.sel(od=od,id=lid).dropna('pid')
                    weights = xr.DataArray(np.full_like(psfPixels,np.nan),
                                           coords=[psfPixels.coords['pix']],
                                           dims = ['pid'])
                    # determine which PSF pixel each line pixel falls in
                    dig = np.digitize(line1d,psfPixelBins,right=True)
                    
                    pix = psfPixels[dig]
                    # central 2.5 pixels on each side have weights = 1
                    central_pix = pix[np.where(abs(pix)<=2.5)[0]]
                    # pixels outside of 5.5 have weights = 0
                    outer_pix   = pix[np.where(abs(pix)>=5.5)[0]]
                    # pixels with 2.5<abs(pix)<5.5 have weights between 0 and 1, linear
                    midleft_pix  = pix[np.where((pix>-5.5)&(pix<-2.5))[0]]
                    midleft_w   = np.array([(x+5.5)/3 for x in midleft_pix])
                    
                    midright_pix = pix[np.where((pix>2.5)&(pix<5.5))[0]]
                    midright_w   = np.array([(-x+5.5)/3 for x in midright_pix])
                    
                    weights.loc[dict(pid=central_pix)] =1.0
                    weights.loc[dict(pid=outer_pix)]   =0.0
                    weights.loc[dict(pid=midleft_pix)] =midleft_w
                    weights.loc[dict(pid=midright_pix)]=midright_w
                    weights = weights.dropna('pid')
                    sel = dict(od=od,id=lid,ax='wgt',pid=np.arange(len(weights)))
                    lines['line'].loc[sel]=weights.values
            return 
        def detect_order(subdata,order):
#            print(subdata)
            spec1d = subdata.sel(ax='flx')
            bkg1d  = subdata.sel(ax='bkg')
            err1d  = subdata.sel(ax='err')
            pixels = np.arange(self.npix)
            wave1d = subdata.sel(ax='wave')
            minima = funcs.peakdet(spec1d,pixels,extreme='min')
            
            npeaks = len(minima.x)-1
            arr    = self.return_empty_dataset(None)
            
            maxima = funcs.peakdet(spec1d,pixels,extreme='max')
            nmaxima = len(maxima)-1
#            first  = int(maxima.x.iloc[-1])
            first  = int(round((minima.x.iloc[-1]+minima.x.iloc[-2])/2))
            last   = int(maxima.x.iloc[0])
            #plt.figure()
            #plt.title(order)
            #plt.plot(wave1d,spec1d)
            
            nu_min  = (299792458e0/(wave1d[first]*1e-10)).values
            nu_max  = (299792458e0/(wave1d[last]*1e-10)).values
            #print(nu_min,nu_max)
            #npeaks2  = int(round((nu_max-nu_min)/self.reprate))+1
            #print(npeaks,nmaxima)
            n_start = int(round((nu_min - self.f0_comb)/self.reprate))
            # in inverse order (wavelength decreases for every element)
            freq1d  = np.array([(self.f0_comb+(n_start+j)*self.reprate) \
                                 for j in range(len(minima))])
            #[plt.axvline(299792458*1e10/f,ls=':',c='r',lw=0.5) for f in freq1d]
            #plt.axvline(299792458*1e10/nu_min,ls=':',c='r',lw=0.5)
            for i in range(npeaks,0,-1):
                # array of pixels
                lpix, upix = (minima.x[i-1],minima.x[i])
                pix  = np.arange(lpix,upix,1,dtype=np.int32)
                # flux, background, flux error
                flux = spec1d[pix]
                bkg  = bkg1d[pix]
                err  = err1d[pix]
                #print(np.arange(pix.size))
                # save values
                val  = {'pix':pix, 
                        'flx':flux,
                        'bkg':bkg,
                        'err':err}
                for ax in val.keys():
                    idx  = dict(id=i-1,pid=np.arange(pix.size),ax=ax)
                    arr['line'].loc[idx] = val[ax]
                
                # barycenter, segment
                bary = np.sum(flux*pix)/np.sum(flux)
                cen_pix = pix[np.argmax(flux)]
                local_seg = cen_pix//self.segsize
                
                arr['pars'].loc[dict(id=i-1,par='freq')]= freq1d[npeaks-i]
                arr['pars'].loc[dict(id=i-1,par='seg')] = local_seg
                arr['pars'].loc[dict(id=i-1,par='bary')]= bary
                # calculate weights in a separate function
            return arr
        def organise_data():
            spec2d = self.extract2d()
            bkg2d  = self.get_background2d()
            err2d  = np.sqrt(spec2d)
            wave2d = xr.DataArray(wavesol_thar,coords=spec2d.coords)
            
            data = xr.concat([spec2d,bkg2d,err2d,wave2d],
                             pd.Index(['flx','bkg','err','wave'],name='ax'))
            return data
        
        # MAIN PART
        orders = self.prepare_orders(order)
        self.check_and_load_psf()
        wavesol_thar = self.check_and_get_wavesol('thar')
        if self.lineDetectionPerformed==True:    
            lines  = self.lines
            return lines
        else:
            print('init lines')
            lines = self.check_and_initialize_lines()
            lines = self.lines
        
        e2ds = organise_data()
        e2ds.name = 'e2ds'
        # truncate data below sOrder:
        e2ds = e2ds[:,sOrder:self.nbo,:]
        
        # merge lines and e2ds
        #data = xr.merge([e2ds,lines])
        
        for od in orders:
            indata  = e2ds.sel(od=od)
            outdata = indata.pipe(detect_order,od)
            lines['pars'].loc[dict(od=od)] = outdata['pars']
            lines['line'].loc[dict(od=od)] = outdata['line']
        #lines = xr.apply_ufunc(detect_order,e2ds.groupby('od'),dask='parallelized')
        if calculate_weights:
            calculate_line_weights(orders)
        else:
            pass
        self.lineDetectionPerformed=True
#        self.lineDetectionPerformed=False
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
        minima    = funcs.peakdet(flux1d, xarray1d, extreme="min")
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
            ind1  = np.where((funcs.is_outlier(sigma)==True))[0]
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
        maxima  = funcs.peakdet(spec1d.flux,spec1d.wave,extreme='max')
        minima  = funcs.peakdet(spec1d.flux,spec1d.pixel,extreme='min')
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
            wavecoeff,pcov   = curve_fit(funcs.polynomial3,np.arange(self.npix),
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
        maxima_p = funcs.peakdet(spec1d.flux,spec1d.pixel,extreme='max')
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
        results = Parallel(n_jobs=1)(delayed(funcs.fit_peak)(i,xarray,yarray,yerror,weights,xmin,xmax,dx,method,model) for i in range(nminima))
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
    def fit_lines1d(self,order,nobackground=False,method='epsf',model=None,
                  scale='pixel',vacuum=True,remove_poor_fits=False,verbose=0):
        # load PSF and detect lines
        self.check_and_load_psf()
        self.check_and_initialize_lines()
        
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
        #lines = self.check_and_initialize_lines()
       
            
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
        self.check_and_initialize_lines()
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
            output = Parallel(n_jobs=-1)(delayed(fit)(order_data,order,lid,self.psf) for lid in range(numlines))
#            print(order,np.shape(output))
#            array = np.array(output)
            order_fit = xr.merge(output)
            list_of_order_fits.append(order_fit)
        fits = xr.merge(list_of_order_fits)
        lines = xr.merge([detected_lines,fits])
        self.lines = lines
        return lines

#            lines['pars'].loc[dict(od=order,id=lines_in_order)] = fitpars
#            lines['line'].loc[dict(od=order,id=lines_in_order,ax='mod')] = models
#        print("Lines fitted")
#        self.lines = lines
#        return self.lines
                #lines['pars']
        
        
    def get_average_profile(self,order,nobackground=True):
        # Extract data from the fits file
        spec1d  = self.extract1d(order,nobackground=nobackground,vacuum=True)
        
        pn,weights  = self.calculate_photon_noise(order,return_array=True)
        #weights     = self.get_weights1d(order)
        # Define limits in wavelength and theoretical wavelengths of lines
        maxima      = funcs.peakdet(spec1d.flux,spec1d.wave,extreme='max')
        minima      = funcs.peakdet(spec1d.flux,spec1d.pixel,extreme='min')
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
            xarray = pd.Series(spec1d.index.values)
        elif scale == "wave":
            xarray = spec1d.wave
        #print(xarray)
        minima          = funcs.peakdet(spec1d.flux, xarray, extreme="min")
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
            minima          = funcs.peakdet(flux, pixels, extreme="min")
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
        maxima      = funcs.peakdet(spec1d["flux"], spec1d[scale], extreme="max")
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
            maxima          = funcs.peakdet(flux, pixels, extreme="max")
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
        extremes    = funcs.peakdet(spec1d["flux"], spec1d[scale], extreme=extreme)
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
            freq0 = data['pars'].sel(par='freq',od=order)#.dropna('val')
            wav0  = 299792458*1e10/freq0
            pix0  = data['pars'].sel(par='cen',od=order)#.dropna('val')
            if calibrator == 'ThAr':
                coeff = self.wavecoeff_vacuum[order]
            elif calibrator == 'LFC':
                coeff = self.wavecoef_LFC[order][::-1]
            wav1 = funcs.polynomial(pix0,*coeff)
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
                    wavecoeff_vac,covariance = curve_fit(funcs.polynomial, 
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
        df_dl         = funcs.derivative1d(spec1d['flux'].values,wavesol[order])
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
                df_dl[order] = funcs.derivative1d(spec2d[order],wavesol[order])                    
        zeros          = np.where(spec2d==0)
        spec2d[zeros]  = np.inf                    
        weights2d      = (wavesol**2 * df_dl**2) / np.abs(spec2d)
        cut            = np.where(weights2d == 0.)
        weights2d[cut] = np.inf
        self.weights2d = np.zeros(shape=self.data.shape)
        self.weights2d[orders,:] = weights2d
        
        return self.weights2d
    def introduce_gaps(self,x,gaps):
        xc = np.empty_like(x)
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
            yerr   = spec1d.error
            axes[0].errorbar(x,y,yerr=yerr,label='Data',capsize=3,capthick=0.3,
                ms=10,elinewidth=0.3)
            if fit==True:
                self.check_and_get_comb_lines()
                lines = self.lines
                linesID = lines.coords['id'].values
                for lid in linesID:
                    if scale == 'wave':
                        line_x = lines['line'].sel(od=order,id=lid,ax='wave')
                    elif scale == 'pixel':
                        line_x = lines['line'].sel(od=order,id=lid,ax='pix')
                    line_m = lines['line'].sel(od=order,id=lid,ax='mod')
                    axes[0].scatter(line_x,line_m,c='C1',marker='X',s=10)
                #fit_lines = self.fit_lines(order,scale=scale,nobackground=nobackground)
                #self.axes[0].plot(x,double_gaussN_erf(x,fit_lines[scale]),label='Fit')
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
                freq  = data['pars'].sel(par='freq',od=order).dropna('id')
                wav   = 299792458*1e10/freq
                pix   = data['pars'].sel(par='cen',od=order).dropna('id')
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
                
        lines = self.check_and_initialize_lines()
        
#        resids  = lines['pars'].sel(par='rsd',od=orders)
        
        pos_pix = lines['pars'].sel(par='cen',od=orders)
        pos_res = lines['pars'].sel(par='rsd',od=orders)
        
        
        
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker     = kwargs.get('marker','x')
        markersize = kwargs.get('markersize',2)
        alpha      = kwargs.get('alpha',1.)
        plotargs = {'s':markersize,'marker':marker,'alpha':alpha}
        for i,order in enumerate(orders):
            pix = pos_pix.sel(od=order)
            res = pos_res.sel(od=order)
            if len(orders)>5:
                plotargs['color']=colors[i]
            axes[0].scatter(pix,res,**plotargs)
            if mean==True:
                meanplotargs={'lw':0.8}
                w  = kwargs.get('window',5)                
                rm = funcs.running_mean(res,w)
                if len(orders)>5:
                    meanplotargs['color']=colors[i]
                axes[0].plot(pix,rm,**meanplotargs)
        [axes[0].axvline(512*(i+1),lw=0.3,ls='--') for i in range (8)]
        axes[0].set_xlabel('Pixel')
        axes[0].set_ylabel('Residuals [m/s]')
        if show == True: figure.show() 
        return plotter
    def plot_histogram(self,kind,order=None,separate=False,
                       show=True,plotter=None,**kwargs):
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
        figure, axes = plotter.figure, plotter.axes
        lines = self.check_and_get_comb_lines(orders=orders)
        
        # plot residuals or chisq
        if kind == 'residuals':
            data     = lines['pars'].sel(par='rsd')
        elif kind == 'chisq':
            data     = lines['pars'].sel(par='chisq')
            
        bins    = kwargs.get('bins',10)
        if separate == True:
            for i,order in enumerate(orders):
                selection = data.sel(od=order).dropna('id').values
                axes[i].hist(selection,bins=bins,normed=normed,range=histrange)
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
            axes[0].hist(selection,bins=bins,normed=normed,range=histrange)
            if kind == 'residuals':
                mean = np.mean(selection)
                std  = np.std(selection)
                A    = 1./np.sqrt(2*np.pi*std**2)
                x    = np.linspace(np.min(selection),np.max(selection),100)
                y    = A*np.exp(-0.5*((x-mean)/std)**2)
                axes[0].plot(x,y,color='C1')
                axes[0].plot([mean,mean],[0,A],color='C1',ls='--')
                axes[0].text(0.8, 0.95,r"$\mu={0:8.3e}$".format(mean), 
                            horizontalalignment='center',
                            verticalalignment='center',transform=axes[0].transAxes)
                axes[0].text(0.8, 0.9,r"$\sigma={0:8.3f}$".format(std), 
                            horizontalalignment='center',
                            verticalalignment='center',transform=axes[0].transAxes)
            axes[0].set_xlabel("{}".format(kind))
            axes[0].set_ylabel('Number of lines')
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
            
        # Select line data        
        pos_pix  = lines['pars'].sel(par='cen')
        pos_freq = lines['pars'].sel(par='freq')
        pos_wav  = (299792458e0/pos_freq)*1e10
        
        # Manage colors
        #cmap   = plt.get_cmap('viridis')
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker = kwargs.get('marker','x')
        ms     = kwargs.get('markersize',5)
        
        # Plot the line through the points?
        plotline = kwargs.get('plot_line',True)
        for i,order in enumerate(orders):
            pix = pos_pix.sel(od=order).dropna('id','all')
            wav = pos_wav.sel(od=order).dropna('id','all')
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
###############################################################################
##############################   MANAGER   ####################################
###############################################################################
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
        baseurl     = 'http://people.sc.eso.org/%7Eglocurto/COMB/'
        
        self.file_paths = []
        self.spectra    = []
        #harpsDataFolder = os.path.join("/Volumes/home/dmilakov/harps","data")
        harpsDataFolder = harps_data#os.path.join("/Volumes/home/dmilakov/harps","data")
        self.harpsdir   = harpsDataFolder
        if sequence!=None:
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
            col         = funcs.select_orders(orders)
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
        #nFibres = len(fibres)
        nFiles  = len(self.file_paths[fibres[0]])
        print("Found {} files".format(nFiles))

        data    = np.zeros((settings.nPix,),dtype=Datatypes(nFiles=nFiles,nOrder=nOrder,fibre=fibre).specdata(add_corr=True).data)
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
        #col = funcs.select_orders(orders)
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
        cut     = np.where(uppix<=settings.nPix)
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
##############################   PLOTTER   ####################################
###############################################################################
class SpectrumPlotter(object):
    def __init__(self,naxes=1,ratios=None,title=None,sep=0.05,figsize=(16,9),
                 alignment="vertical",sharex=None,sharey=None,**kwargs):
        fig, axes = funcs.get_fig_axes(naxes,ratios=ratios,title=title,
                                 sep=sep,alignment=alignment,
                                 figsize=figsize,sharex=sharex,
                                 sharey=sharey,**kwargs)
        self.figure = fig
        self.axes   = axes   
    def show(self):
        self.figure.show()
        return
        
        
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
def fit(lines,order,lid,psf):
    def return_empty_dataset(order=None):
        orders        = [order]
        linesPerOrder = 400
        pixPerLine    = 22
        # lineAxes : pixel, flux, background, flux error, residual, weight
        #            best fit model, wavelength
        lineAxes      = ['pix','flx','bkg','err','rsd','wgt','mod','wave']
        # linePars : barycenter, best fit center, best fit center error, 
        #            best fit flux, best fit flux error, frequency, 
        #            frequency error, reduced chi square, segment number,
        #            residual of the wavelength to the wavelength solution fit
        linePars      = ['bary','cen','cen_err','flx','flx_err',
                         'freq','freq_err','chisq','seg','rsd']
        shape_data    = (1,linesPerOrder,len(lineAxes),pixPerLine)
        shape_pars    = (1,linesPerOrder,len(linePars))
        data_vars     = {'line':(['od','id','ax','pid'],np.full(shape_data,np.nan)),
                         'pars':(['od','id','par'],np.full(shape_pars,np.nan))}
#            if len(orders) ==1: orders = orders[0]
        data_coords   = {'od':orders,
                         'id':np.arange(linesPerOrder),
                         'pid':np.arange(pixPerLine),
                         'ax':lineAxes,
                         'par':linePars}
        dataset       = xr.Dataset(data_vars,data_coords)
        return dataset
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
        resid = np.sqrt(weights) * ((counts-background) - model)/np.sqrt(np.abs(counts))
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
        
        epsf_x  = psf.sel(ax='x',od=order,seg=seg).dropna('pix')+pix
        epsf_1 = psf.sel(ax='y',od=order,seg=sgl).dropna('pix')
        epsf_2 = psf.sel(ax='y',od=order,seg=sgr).dropna('pix')
        epsf_y = f1*epsf_1 + f2*epsf_2 
        
        xc     = epsf_y.coords['pix']
        epsf_x  = psf.sel(ax='x',od=order,seg=seg,pix=xc)+pix
        
        return epsf_x, epsf_y
    # MAIN PART 
    line      = lines.sel(id=lid).dropna('pid','all')
    pid       = line.coords['pid']
    line_x    = line['line'].sel(ax='pix')
    line_y    = line['line'].sel(ax='flx')
    line_w    = line['line'].sel(ax='wgt')
    line_bkg  = line['line'].sel(ax='bkg')
    line_bary = line['pars'].sel(par='bary')
    cen_pix   = line_x[np.argmax(line_y)]
    loc_seg   = line['pars'].sel(par='seg')
    freq      = line['pars'].sel(par='freq')
    
    psf_x, psf_y = get_local_psf(cen_pix,order=order,seg=loc_seg)
    psf_rep  = interpolate.splrep(psf_x,psf_y)
    
    #
    arr    = return_empty_dataset(order)
    p0 = (0,np.max(line_y))
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
        pars = np.array([b, cen,cen_err,flx,flx_err,
                         freq,1e3,rchisq,loc_seg,np.nan])
    else:
        pars = np.full(len(linePars),np.nan)
    arr['pars'].loc[dict(id=lid)]=pars
#    print(np.shape(arr['line'].loc[dict(id=lid,ax='mod',pid=pid)]))
#    print(np.shape(line_model))
    arr['line'].loc[dict(od=order,id=lid,ax='mod',pid=np.arange(len(line_model)))]=line_model
    
    return arr