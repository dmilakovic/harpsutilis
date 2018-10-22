#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:45:04 2018

@author: dmilakov
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import gc
import os
import warnings
import dill as pickle
import tqdm
import sys
import time


from astropy.io import fits
from fitsio import FITS
from scipy.optimize import curve_fit, leastsq
from scipy import odr, interpolate
from multiprocessing import Pool

from harps import functions as hf
from harps import settings as hs
allowed_hdutypes = ['linelist','wavesol']

__version__ = '0.5.1'

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
                 header=True,readdata=True,read_LFC=True,initialize_fits=False):
        '''
        Initialise the spectrum object.
        '''
        self.filepath = filepath
        self.name = "HARPS Spectrum"
        self.datetime = np.datetime64(os.path.basename(filepath).split('.')[1].replace('_',':')) 
        self.HDU   = {'linelist':None,'wavesol':None}
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
        if read_LFC == True:
            self.__read_LFC_keywords__()
        else:
            pass
        if initialize_fits==True:
            self._initialize_FITS()
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
        
        existLines = hasattr(self,'linelist') 
        if not existLines:
            order = self.prepare_orders(None)
            lines = hf.return_empty_dataset(order,self.pixPerLine)
            linelist = self.HDU_get('linelist')
            self.linelist = linelist
        else:
            linelist = self.linelist
        return linelist
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
    
    def HDU_exists(self,hdutype):
        hdus = getattr(self,'HDU')
        hdu = hdus[hdutype]
        if hdu is not None and len(hdu)>0:
            return True
        else:
            return False
    def HDU_get(self,hdutype,**kwargs):
        ''' kwargs: 
            ------
                mode : 'r','rw'
        '''
        exists = self.HDU_exists(hdutype)
        if exists:       
            return self.HDU_read(hdutype,**kwargs)
        else:
            return self.HDU_new(hdutype)
            
    def HDU_new(self,hdutype,dirname=None,overwrite=True,mode='rw'):
        path     = self.HDU_pathname(hdutype,dirname)
        self.hdu_path = path
        if overwrite == True:
            try:
                os.remove(path)
            except OSError:
                pass
        else:
            pass
        newhdu   = HDU(path,mode)
        newhdu.write_primary(self)
        self.HDU.update({hdutype:path})
        return newhdu
    def HDU_pathname(self,hdutype,dirname=None):
        dirname  = hf.get_dirname(hdutype,dirname)
        basename = os.path.splitext(os.path.basename(self.filepath))[0]
        path     = os.path.join(dirname,basename+'_{}.fits'.format(hdutype))
        return path
    def HDU_read(self,hdutype,dirpath=None,mode='rw'):
        print("READING ",hdutype)
        if dirpath is not None:
            hdu_path = self.HDU_pathname(hdutype,dirpath)
        else:
            hdu_path = self.HDU[hdutype]
        hdu = FITS(hdu_path,mode)
        return hdu
    
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
            #self.anchor = round(self.header["HIERARCH ESO INS LFC1 ANCHOR"],-6) 
            self.anchor = self.header["HIERARCH ESO INS LFC1 ANCHOR"]
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
            # taken from Gaspare's notes on April 2015 run
            self.pixPerLine   = 35
            # wiener filter window scale
            self.window       = 5
        self.omega_r = 250e6
        m,k            = divmod(
                            round((self.anchor-self.f0_source)/self.fr_source),
                                   self.modefilter)
        self.f0_comb   = (k-1)*self.fr_source + self.f0_source
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
            minima  = hf.peakdet(yarray,xarray,extreme='min',window=self.window)
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
        A method to detect lines present in the echelle order.
        
        
        The output is saved in a FITS file.
        
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
        pbar.update(1)
        pbar.close()
        # SAVE TO LINES HDU
        lines_hdu = self.HDU_get('linelist')
        for linelist,od in zip(outdata,orders):
            #linelist_dtype = hf.return_dtype('linelist')
            #modified_linelist = np.asarray(linelist,dtype=linelist_dtype)
            #self._write_HDU(linelist)
            lines_hdu.write(linelist,extname='ORDER{0:2s}'.format(str(od)))
        lines_hdu.close()
            
        #sys.exit()
        #detected_lines = xr.merge(outdata)
       
        #lines['attr'] = detected_lines['attr']
        #lines['line'] = detected_lines['line']
        #lines['stat'] = detected_lines['stat']
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
        self.linelist = self.HDU['linelist']
        
        self.lineDetectionPerformed=True
        gc.collect()
        return self.linelist
    
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
        minima    = hf.peakdet(flux1d, xarray1d, extreme="min",
                               window=self.window,**kwargs)
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
    def extract2d(self,order=None):
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
    def fit_lines(self,order=None,fittype='epsf',nobackground=True,model=None,
                  remove_poor_fits=False,verbose=0,njobs=hs.nproc):
        ''' Calls one of the specialised line fitting routines.
        '''
        # Was the fitting already performed?
        if self.lineFittingPerformed[fittype] == True:
            return self.HDU_get('linelist')
        else:
            pass
        
        # Select method
        if fittype == 'epsf':
            self.check_and_load_psf()
            function = wrap_fit_epsf
#            function = wrap_fit_single_line
        elif fittype == 'gauss':
            function = hf.wrap_fit_peak_gauss
        
        # Check if the lines were detected, run 'detect_lines' if not
        self.check_and_return_lines()

        linelist_hdu = self.HDU_get('linelist')
        orders    = self.prepare_orders(order)
        #linesID   = self.lines.coords['id']
        
        
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
#        
#    def fit_lines(self,order=None,fittype='epsf',nobackground=True,model=None,
#                  remove_poor_fits=False,verbose=0,njobs=hs.nproc):
#        ''' Calls one of the specialised line fitting routines.
#        '''
#        # Was the fitting already performed?
#        if self.lineFittingPerformed[fittype] == True:
#            return self.linelist
#        
#        
#        # Select method
#        if fittype == 'epsf':
#            self.check_and_load_psf()
#            function = wrap_fit_epsf
##            function = wrap_fit_single_line
#        elif fittype == 'gauss':
#            function = hf.wrap_fit_peak_gauss
#        
#        # Check if the lines were detected, run 'detect_lines' if not
#        self.check_and_return_lines()
##        if self.lineDetectionPerformed==True:
##            linelist = self.HDU_pathname('linelist')
##            if linelist is None:
##                if fittype == 'epsf':
##                    cw=True
##                elif fittype == 'gauss':
##                    cw=False
##                linelist = self.detect_lines(order,calculate_weights=cw)
##            else:
##                pass
##        else:
##            if fittype == 'epsf':
##                cw=True
##            elif fittype == 'gauss':
##                cw=False
##            linelist = self.detect_lines(order,calculate_weights=cw)
##        lines = linelist
#        linelist_hdu = self.HDU_get('linelist')
#        orders    = self.prepare_orders(order)
#        #linesID   = self.lines.coords['id']
#        
#        
#        list_of_order_linefits = []
#        list_of_order_models = []
#        
#        progress = tqdm.tqdm(total=len(orders),
#                             desc='Fitting lines {0:>5s}'.format(fittype))
#        
#        start = time.time()
##        mp_pool = ProcessPool()
##        mp_pool.nproc      = 1
#        pool3 = Pool(hs.nproc)
#        for order in orders:
#            
#            progress.update(1)
#            order_data = detected_lines.sel(od=order).dropna('id','all')
#            lines_in_order = order_data.coords['id']
#            numlines       = np.size(lines_in_order)
#            
#            if fittype == 'epsf':
##                output = Parallel(n_jobs=njobs)(delayed(function)(order_data,order,lid,self.psf,self.pixPerLine) for lid in range(numlines))
##                results = mp_pool.map(self.fit_single_line,[order_data.sel(id=lid) for lid in range(numlines)])
##                output = pool.map(function,[(order_data,order,lid,self.psf,self.pixPerLine) for lid in range(numlines)])
##                results = Parallel(n_jobs=hs.nproc)(delayed(function)(order_data.sel(id=lid),self.psf,self.pixPerLine) for lid in range(numlines))
#                results = pool3.map(function,
#                                    [(order_data.sel(id=lid),self.psf,self.pixPerLine) for lid in range(numlines)])
#                time.sleep(1)
#            elif fittype == 'gauss':
##                results = Parallel(n_jobs=hs.nproc)(delayed(function)(order_data,order,i,'erfc','singlegaussian',self.pixPerLine,0) for i in range(numlines))
#                results = pool3.map(function,[(order_data,order,i,'erfc','singlesimplegaussian',self.pixPerLine,0) for i in range(numlines)])
#            
#            parameters,models = zip(*results)
#            order_fit = xr.merge(parameters)
#            order_models = xr.merge(models)
#            list_of_order_linefits.append(order_fit['pars'])
#            list_of_order_models.append(order_models['model'])
#            gc.collect()
#           
#        pool3.close()
#        pool3.join()
#        progress.close()
#        fits = xr.merge(list_of_order_linefits)
#        models =xr.merge(list_of_order_models)
#        #lines = xr.merge([fits,models])
#        lines['pars'].loc[dict(ft=fittype,od=orders)]  = fits['pars'].sel(ft=fittype)
#        lines['model'].loc[dict(ft=fittype,od=orders)] = models['model'].sel(ft=fittype)
#        self.lines = lines
#        self.lineDetectionPerformed = True
#        self.lineFittingPerformed[fittype]=True
##        pool.close()
##        del(pool)
#        return lines
        
#    def fit_lines_gaussian1d(self,order,nobackground=True,method='erfc',model=None,
#                  scale='pixel',remove_poor_fits=False,verbose=0):
#        """Fits LFC lines of a single echelle order.
#        
#        Extracts a 1D spectrum of a selected echelle order and fits a single 
#        Gaussian profile to each line, in both wavelength and pixel space. 
#        
#        Args:
#            order: Integer number of the echelle order in the FITS file.
#            nobackground: Boolean determining whether background is subtracted 
#                before fitting is performed.
#            method: String specifying the method to be used for fitting. 
#                Options are 'curve_fit', 'lmfit', 'chisq', 'erfc'.
#            remove_poor_fits: If true, removes the fits which are classified
#                as outliers in their sigma values.
#        Returns:
#            A dictionary with two pandas DataFrame objects, each containing 
#            parameters of the fitted lines. For example:
#            
#            {'wave': pd.DataFrame(amplitude,center,sigma),
#             'pixel: pd.DataFrame(amplitude,center,sigma)}
#        """
#        
#        
#        def _remove_poor_fits(input_lines):
#            """ Removes poorly fitted lines from the list of fitted lines.
#            
#            Identifies outliers in sigma parameter and removes those lines from 
#            the list.
#            
#            Args:
#                input_lines: Dictionary returned by fit_lines.
#            Returns:
#                output_lines: Dictionary returned by fit_lines.
#            """
#            
#            xi = []
#            #output_lines = {}
#            df = input_lines
#            #for scale,df in input_lines.items():
#            if 'sigma1' in df.columns:
#                sigma = np.array(df.sigma1.values,dtype=np.float32)#[1:]
#            else:
#                sigma = np.array(df.sigma.values,dtype=np.float32)
##                centd = np.array(df.center.diff().dropna().values,dtype=np.float32)
##                ind   = np.where((is_outlier2(sigma,thresh=4)==True) |  
##                                 (is_outlier2(centd,thresh=4)==True))[0]
#            # Outliers in sigma
#            ind1  = np.where((hf.is_outlier(sigma)==True))[0]
##                ind2  = np.where((is_outlier(centd)==True))[0]
#            # Negative centers
#            if 'center1' in df.columns:    
#                ind3  = np.where(df.center1<0)[0]
#            else:
#                ind3  = np.where(df.center<0)[0]
#            ind   = np.union1d(ind1,ind3)
#            xi.append(ind)
#                
#            #a1,a2  = xi
#            #xmatch = np.intersect1d(a2, a1)
#            xmatch=[0]
#            #for scale,df in input_lines.items():
#            newdf = df.drop(df.index[xmatch])
#            output_lines = newdf  
#                
#            return output_lines
#                
#        #######################################################################
#        #                        MAIN PART OF fit_lines                       #
#        #######################################################################
#        # Debugging
#        plot=False
#        if verbose>0:
#            print("ORDER:{0:<5d} Bkground:{1:<5b} Method:{2:<5s}".format(order,
#                  not nobackground, method))
#        # Determine which scales to use
#        scale = ['wave','pixel'] if scale is None else [scale]
#        
#        # Extract data from the fits file
#        spec1d  = self.extract1d(order,nobackground=nobackground,vacuum=True)
#        
#        pn,weights  = self.calculate_photon_noise(order,return_array=True)
#        weights = self.get_weights1d(order)
#        # Define limits in wavelength and theoretical wavelengths of lines
#        maxima  = hf.peakdet(spec1d.flux,spec1d.wave,extreme='max',
#                             window=self.window)
#        minima  = hf.peakdet(spec1d.flux,spec1d.pixel,extreme='min',
#                             window=self.window)
#        xpeak   = maxima.x
#        nu_min  = 299792458e0/(xpeak.iloc[-1]*1e-10)
#        nu_max  = 299792458e0/(xpeak.iloc[0]*1e-10)
##        print(nu_min,nu_max)
#        npeaks  = int(round((nu_max-nu_min)/self.reprate))+1
#        n_start = int(round((nu_min - self.f0_comb)/self.reprate))
#        lbd     = np.array([299792458e0/(self.f0_comb 
#               + (n_start+i)*self.reprate)*1e10 for i in range(xpeak.size)][::-1])
#        if verbose>1:
#            print("Npeaks:{0:<5}".format(npeaks))
#        
#        # Find the closest detected peak to the theoretical position using the 
#        # nearest neighbour method
##        lbd_th   = combine_line_list(lbd,maxima.x.values)
##        lbd_th   = maxima.x.values
#        
#        
#        
#        # Invert _vacuum_ wavelenth solution obtained from ThAr and 
#        # find theoretical pixel positions of lines 
#        try:
#            wavecoeff_vacuum = self.wavecoeff_vacuum[order]
#        except:
#            wavesol          = self.wavesol_thar[order]
#            wavecoeff_air    = self.wavecoeff_air[order]
#            wavecoeff,pcov   = curve_fit(hf.polynomial3,np.arange(self.npix),
#                                         wavesol,
#                                         sigma = weights,
#                                         p0=wavecoeff_air)
#            wavecoeff_vacuum = wavecoeff
#        poly1d   = np.poly1d(wavecoeff_vacuum[::-1])
#        lbd_th   = lbd
#        npeaks   = lbd_th.size
#        pix_th   = np.array([(np.poly1d(lbd_th[npeaks-i-1])-poly1d).roots[2].real
#                                        for i in range(npeaks)][::-1])
##        print(pix_th.size,lbd_th.size)
#        # Define a pandas DataFrame object with theoretical centres of lines
#        lines_th = pd.DataFrame({'pixel':pix_th,'wave':lbd_th}) 
#        maxima_p = hf.peakdet(spec1d.flux,spec1d.pixel,extreme='max')
#        maxima_th = pd.DataFrame({'pixel':maxima_p.x,'wave':maxima.x})
#        #print(lines_th)
#        # Perform the fitting        
#        #lines    = {}
#        #for scale in scale:
#        if plot:
#            plt.figure()
#        xarray     = spec1d['pixel']
#        yarray     = spec1d['flux']
#        yerror     = spec1d['error']
#        xmax       = maxima_th['pixel']
#        xmin       = minima.x
#        nminima    = minima.index.size
#        nmaxima    = maxima.index.size
#        #print(nminima,nmaxima)
#        dxi   = 11.
#        dx         = xarray.diff(1).fillna(dxi)
#        if verbose>2:
#            print('Fitting {}'.format(scale))
#        
#        # model
#        model = model if model is not None else 'singlegaussian'
#        results = Parallel(n_jobs=hs.nproc)(delayed(hf.fit_peak)(i,xarray,yarray,yerror,weights,xmin,xmax,dx,method,model) for i in range(nminima))
#        results = np.array(results)
#      
#        parameters = results['pars'].squeeze(axis=1)
#        errors     = results['errors'].squeeze(axis=1)
#        photon_nse = results['pn'].squeeze(axis=1)
#        center     = results['cen'].squeeze(axis=1)
#        center_err = results['cen_err'].squeeze(axis=1)
#        rsquared   = results['r2'].squeeze(axis=1)
#        #N = results.shape[0]
#        #M = parameters.shape[1]
#        
#        
#        #print(np.shape(parameters),np.shape(errors),np.shape(photon_nse))
#        line_results = np.concatenate((parameters,errors,photon_nse,rsquared,center,center_err),axis=1)
#        if model == 'singlegaussian':
#            columns = ['amplitude','cen','sigma',
#                       'amplitude_error','cen_error','sigma_error',
#                       'photon_noise','r2','center','center_err']
#        elif ((model == 'doublegaussian') or (model=='simplegaussian')):
#            columns = ['amplitude1','center1','sigma1',
#                      'amplitude2','center2','sigma2',
#                      'amplitude1_error','center1_error','sigma1_error',
#                      'amplitude2_error','center2_error','sigma2_error',
#                      'photon_noise','r2','center','center_err']
#        lines_fit = pd.DataFrame(line_results,
#                                 index=np.arange(0,nminima,1),#lines_th.index,
#                                 columns=columns)
#        # make sure sigma values are positive!
#        if model == 'singlegaussian':
#            lines_fit.sigma = lines_fit.sigma.abs()
#        elif ((model == 'doublegaussian') or (model=='simplegaussian')):
#            lines_fit.sigma1 = lines_fit.sigma1.abs()
#            lines_fit.sigma2 = lines_fit.sigma2.abs()
#        lines_fit['th_wave'] = lines_th['wave']
#        lines_fit['th_pixel']  = lines_th['pixel']
#        lines_fit.dropna(axis=0,how='any',inplace=True)            
##        lines[scale]        = lines_fit   
#        if remove_poor_fits == True:
#            if verbose>2:
#                print('Removing poor fits')
#            lines_fit = _remove_poor_fits(lines_fit)
#        else:
#            pass
#
#        return lines_fit
#    def fit_lines_gaussian2d(self,order=None,nobackground=True,method='erfc',
#                  model=None,scale='pixel',remove_poor_fits=False,verbose=0):
#        """Fits LFC lines of a single echelle order.
#        
#        Extracts a 1D spectrum of a selected echelle order and fits a single 
#        Gaussian profile to each line, in both wavelength and pixel space. 
#        
#        Args:
#            order: Integer number of the echelle order in the FITS file.
#            nobackground: Boolean determining whether background is subtracted 
#                before fitting is performed.
#            method: String specifying the method to be used for fitting. 
#                Options are 'curve_fit', 'lmfit', 'chisq', 'erfc'.
#            remove_poor_fits: If true, removes the fits which are classified
#                as outliers in their sigma values.
#        Returns:
#            A dictionary with two pandas DataFrame objects, each containing 
#            parameters of the fitted lines. For example:
#            
#            {'wave': pd.DataFrame(amplitude,center,sigma),
#             'pixel: pd.DataFrame(amplitude,center,sigma)}
#        """
#        
#                
#        #######################################################################
#        #                        MAIN PART OF fit_lines                       #
#        #######################################################################
#        # Debugging
#        plot=False
#        if verbose>0:
#            print("ORDER:{0:<5d} Bkground:{1:<5b} Method:{2:<5s}".format(order,
#                  not nobackground, method))
#        # Determine which scales to use
#        scale = ['wave','pixel'] if scale is None else [scale]
#        # Have lines been fitted already?
#        self.check_and_return_lines()
#        if self.lineDetectionPerformed==True:
#            detected_lines = self.lines
#            if detected_lines is None:
#                detected_lines = self.detect_lines(order)
#            else:
#                pass
#        else:
#            detected_lines = self.detect_lines(order)        
#        orders    = self.prepare_orders(order)
#        linesID   = self.lines.coords['id']
#        
#        # contains a list of DataArrays, each one containing line fit params
#        list_of_order_fits = []
#        for order in orders:
#            order_data = detected_lines.sel(od=order).dropna('id','all')
#            lines_in_order = order_data.coords['id']
#            numlines       = np.size(lines_in_order)
#            if verbose>1:
#                print("Npeaks:{0:<5}".format(numlines))
#        
#        
#
#            model = model if model is not None else 'singlegaussian'
#            output = Parallel(n_jobs=hs.nproc)(delayed(hf.fit_peak_gauss)(order_data,order,i,method,model) for i in range(numlines))
#            # output is a list of xr.DataArrays containing line fit params
#            # for this order
#            order_fit = xr.merge(output)
#            list_of_order_fits.append(order_fit)
#
#        fits = xr.merge(list_of_order_fits)
#        #fits.rename({'pars':'gauss'})
#        lines_gaussian = xr.merge([detected_lines,fits])
#        #self.lines_gaussian = lines_gaussian
#        return lines_gaussian
#    def fit_lines1d(self,order,nobackground=False,method='epsf',model=None,
#                  scale='pixel',vacuum=True,remove_poor_fits=False,verbose=0):
#        # load PSF and detect lines
#        self.check_and_load_psf()
#        self.check_and_return_lines()
#        
#        #sc        = self.segment_centers
#        segsize   = 4096//self.nsegments
#        pixels    = self.psf.coords['pix']
#        pixelbins = (pixels[1:]+pixels[:-1])/2
#        
#        def get_line_weights(line_x,center):
#            
#            weights = xr.DataArray(np.full_like(pixels,np.nan),coords=[pixels],dims=['pix'])
#            
#            pixels0 = line_x - center
#            pix = pixels[np.digitize(pixels0,pixelbins,right=True)]
#            # central 2.5 pixels on each side have weights = 1
#            central_pix = pix[np.where(abs(pix)<=2.5)[0]]
#            weights.loc[dict(pix=central_pix)]=1.0
#            # pixels outside of 5.5 have weights = 0
#            outer_pix   = pix[np.where(abs(pix)>=5.5)[0]]
#            weights.loc[dict(pix=outer_pix)]=0.0
#            # pixels with 2.5<abs(pix)<5.5 have weights between 0 and 1, linear
#            midleft_pix  = pix[np.where((pix>-5.5)&(pix<-2.5))[0]]
#            midleft_w   = np.array([(x+5.5)/3 for x in midleft_pix])
#            
#            midright_pix = pix[np.where((pix>2.5)&(pix<5.5))[0]]
#            midright_w   = np.array([(-x+5.5)/3 for x in midright_pix])
#            
#            weights.loc[dict(pix=midleft_pix)] =midleft_w
#            weights.loc[dict(pix=midright_pix)]=midright_w
#            return weights.dropna('pix').values
#        def residuals(x0,pixels,counts,weights,background,splr):
#            ''' Modela parameters are estimated shift of the line center from 
#                the brightest pixel and the line flux. 
#                Input:
#                ------
#                   x0        : shift, flux
#                   pixels    : pixels of the line
#                   counts    : detected e- for each pixel
#                   weights   : weights of each pixel (see 'get_line_weights')
#                   background: estimated background contamination in e- 
#                   splr      : spline representation of the ePSF
#                Output:
#                -------
#                   residals  : residuals of the model
#            '''
#            sft, flux = x0
#            model = flux * interpolate.splev(pixels+sft,splr) 
#            resid = np.sqrt(line_w) * ((counts-background) - model)/np.sqrt(np.abs(counts))
#            #resid = line_w * (counts- model)
#            return resid
#        
#        # Determine which scales to use
#        scale = ['wave','pixel'] if scale is None else [scale]
#        if verbose>0:
#            print("ORDER:{0:<5d} Bkground:{1:<5b} Method:{2:<5s}".format(order,
#                  not nobackground, method))
#        # Prepare orders
#        orders = self.prepare_orders(order)
#        #lines = self.check_and_return_lines()
#       
#            
#        # Cut the lines
#        
#        pixel, flux, error, bkgr, bary = self.cut_lines(orders, nobackground=nobackground,
#                  vacuum=vacuum,columns=['pixel', 'flux', 'error', 'bkg', 'bary'])
#        pixel = pixel[order]
#        flux  = flux[order]
#        error = error[order]
#        bkgr  = bkgr[order]
#        bary  = bary[order]
#        nlines = len(pixel)
#        params = ['cen','cen_err','flux','flux_err','shift','phase','b','chisq']
#        lines = xr.DataArray(data=np.zeros((nlines,len(params))),
#                             coords = [np.arange(nlines),params],
#                             dims = ['id','par'])
#        
#        for n in range(nlines):
#            line_x = pixel[n]
#            line_y = flux[n]
#            line_b = bkgr[n]
#            cen_pix = line_x[np.argmax(line_y)]
#            local_seg = cen_pix//segsize
#            psf_x, psf_y = self.get_local_psf(cen_pix,order=order,seg=local_seg)
#            
#            line_w = get_line_weights(line_x,cen_pix)
#            psf_rep  = interpolate.splrep(psf_x,psf_y)
#            p0 = (0,np.max(line_y))
#            
#            popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
#                                    args=(line_x,line_y,line_w,line_b,psf_rep),
#                                    full_output=True)
#            
#            if ier not in [1, 2, 3, 4]:
#                print("Optimal parameters not found: " + errmsg)
#                popt = np.full_like(p0,np.nan)
#                pcov = None
#                success = False
#            else:
#                success = True
#            if success:
#                
#                sft, flx = popt
#                cost   = np.sum(infodict['fvec']**2)
#                dof    = (len(line_x) - len(popt))
#                rchisq = cost/dof
#                if pcov is not None:
#                    pcov = pcov*rchisq
#                else:
#                    pcov = np.array([[np.inf,0],[0,np.inf]])
#                cen              = line_x[np.argmax(line_y)]-sft
#                cen_err, flx_err = [np.sqrt(pcov[i][i]) for i in range(2)]
#                phi              = cen - int(cen+0.5)
#                b                = bary[n]
#                pars = np.array([cen,cen_err,flx,flx_err, sft,phi,b,rchisq])
#                model = flx * interpolate.splev(pixels+sft,psf_rep) 
#            else:
#                pars = np.full(8,np.nan)
#                model = np.full_like(pixels,np.nan)
#        #lines.loc[dict(id=n)]
#        lines.loc[dict(id=n)] = pars
#        return lines
#    def fit_lines2d(self,order=None):
#        self.check_and_load_psf()
#        self.check_and_return_lines()
#        if self.lineDetectionPerformed==True:
#            detected_lines = self.lines
#            if detected_lines is None:
#                detected_lines = self.detect_lines(order)
#            else:
#                pass
#        else:
#            detected_lines = self.detect_lines(order)
#        orders    = self.prepare_orders(order)
#        linesID   = self.lines.coords['id']
#        
#        list_of_order_fits = []
#        for order in orders:
#            order_data = detected_lines.sel(od=order).dropna('id','all')
#            lines_in_order = order_data.coords['id']
#            numlines       = np.size(lines_in_order)
#            output = Parallel(n_jobs=hs.nproc)(delayed(fit_epsf)(order_data,order,lid,self.psf) for lid in range(numlines))
##            print(order,np.shape(output))
##            array = np.array(output)
#            order_fit = xr.merge(output)
#            list_of_order_fits.append(order_fit)
#        fits = xr.merge(list_of_order_fits)
#        lines = xr.merge([detected_lines,fits])
#        self.lines = lines
#        self.lineDetectionPerformed = True
#        return lines
#
#        
#        
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
        
    def get_background1d(self, order, scale="pixel", kind="linear",*args):
        '''Function to determine the background of the observations by fitting a cubic spline to the minima of LFC lines'''
        spec1d          = self.extract1d(order=order)
        if scale == "pixel":
            xarray = np.arange(self.npix)
        elif scale == "wave":
            xarray = spec1d.wave
        #print(xarray)
        yarray          = self.data[order]
        minima          = hf.peakdet(yarray, xarray, extreme="min",
                                     window=self.window,**kwargs)
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
    def get_background2d(self,orders=None,kind='linear',**kwargs):
        orders = self.prepare_orders(orders)
        spec2d = self.extract2d()
        bkg2d  = spec2d.copy()
        pixels = spec2d.coords['pix']
        for order in orders:
            flux            = spec2d.sel(od=order)
            minima          = hf.peakdet(flux, pixels, extreme="min",
                                         window=self.window, **kwargs)
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
    def get_envelope1d(self, order, scale="pixel", kind="spline",**kwargs):
        '''Function to determine the envelope of the observations by fitting 
            a cubic spline or a straight line to the maxima of LFC lines'''
        key = scale
        spec1d      = self.extract1d(order=order)
        maxima      = hf.peakdet(spec1d["flux"], spec1d[scale], extreme="max",
                                 **kwargs)
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
    def get_envelope2d(self,orders=None,kind='linear',**kwargs):
        orders = self.prepare_orders(orders)
        spec2d = self.extract2d()
        env2d  = spec2d.copy()
        pixels = spec2d.coords['pix']
        for order in orders:
            flux            = spec2d.sel(od=order)
            maxima          = hf.peakdet(flux, pixels, extreme="max",**kwargs)
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
        extremes    = hf.peakdet(spec1d["flux"], spec1d[scale], extreme=extreme,
                                 limit=2*self.window)
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
                #wav   = data['wave'].sel(wav='val',od=order,ft=fittype).dropna('id')
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
            print(len(thar),len(wav))
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

        
        wavesol_LFC  = self.check_and_get_wavesol(calibrator='LFC')

        LFCws        = self.LFCws
        self.save_dataset(LFCws,dtype='LFCws',dirname=dirname,replace=replace)
        

    def save_lines(self,dirname=None,replace=False):

        lines    = self.check_and_get_comb_lines()
        self.save_dataset(lines,dtype='lines',dirname=dirname,replace=replace)

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
    
class HDU(FITS):
    def write_primary(self,spec):
        ''' Wrapper around fitsio FITS class'''
        
        def make_dict(name,value,comment=''):
            return dict(name=name,value=value,comment=comment)
        def return_header():
            header_names=['Author','version','npix','MJD','LFC','omega_r',
                          'omega_0','fibshape','use_gaps','use_ptch','polyord']
            header_values=['Dinko Milakovic',__version__,spec.npix,00000,
                           spec.LFC,spec.reprate,spec.f0_comb,spec.fibre_shape,
                           spec.use_gaps,spec.patches,spec.polyord]
            header_comments=['','harps.classes version used',
                             'Number of pixels','Modified Julian Date',
                             'LFC name','Repetition frequency',
                             'Offset frequency','Fibre shape',
                             'Shift lines using gap file',
                             'Fit wavelength solution in 512 pix patches',
                             'Polynomial order of the wavelength solution']
            
            header = [make_dict(n,v,c) for n,v,c in zip(header_names,
                                                        header_values,
                                                        header_comments)]
            return header
        
        
        header   = return_header()
        self.write(np.arange(2*2).reshape(2,-1),header=header,
                     extname='PRIMARY')
        
        return  
###############################################################################
###########################   MISCELANEOUS   ##################################
###############################################################################    
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

def detect_order(orderdata,f0_comb,reprate,segsize,pixPerLine,window):
    # speed of light
    c = 2.99792458e8
    # metadata and data container
    order = int(orderdata.coords['od'])
    arr   = hf.return_empty_dataset(order,pixPerLine)
    
    # extract arrays
    spec1d = orderdata.sel(ax='flx')
    wave1d = orderdata.sel(ax='wave')
    bkg1d  = orderdata.sel(ax='bkg')
    err1d  = orderdata.sel(ax='err')
    pixels = np.arange(4096)
    # photon noise
    sigma_v= orderdata.sel(ax='sigma_v')
    pn_weights = (sigma_v/299792458e0)**-2
    
    # warn if ThAr solution does not exist for this order:
    if wave1d.sum()==0:
        warnings.warn("ThAr WAVELENGTH SOLUTION DOES NOT EXIST")
        return arr
    # determine the positions of minima
    yarray = spec1d-bkg1d
    raw_minima = hf.peakdet(yarray,pixels,extreme='min',
                        method='peakdetect_derivatives',
                        window=window)
    minima = raw_minima
    # zeroth order approximation: maxima are equidistant from minima
    maxima0 = ((minima.x+np.roll(minima.x,1))/2).astype(np.int16)
    # remove 0th element (between minima[0] and minima[-1]) and reset index
    maxima1 = maxima0[1:]
    maxima  = maxima1.reset_index(drop=True)
    # first order approximation: maxima are closest to the brightest pixel 
    # between minima
    #maxima0 = []
    # total number of lines
    nlines = len(maxima)
    # calculate frequencies of all lines from ThAr solution
    maxima_index     = maxima.values
    maxima_wave_thar = wave1d[maxima_index]
    maxima_freq_thar = 2.99792458e8/maxima_wave_thar*1e10
    # closeness of all maxima to the known modes:
    decimal_n = ((maxima_freq_thar - f0_comb)/reprate)
    integer_n = np.rint(decimal_n).astype(np.int16)
    closeness = np.abs( decimal_n - integer_n ).values
    # the line closest to the frequency of an LFC mode is the reference:
    ref_index = int(np.argmin(closeness))
    ref_pixel = int(maxima_index[ref_index])
    ref_n     = int(integer_n[ref_index])
    ref_freq  = f0_comb + ref_n * reprate
    ref_wave  = c/ref_freq * 1e10
    # make a decreasing array of Ns, where N[ref_index]=ref_n:
    aranged  = np.arange(nlines)[::-1]
    shifted  = aranged - (nlines-ref_index-1)
    N        = shifted+ref_n
    
    linelist = hf.return_empty_linelist(nlines)
    #print("N[ref_index]==ref_n",N[ref_index]==ref_n)
    for i in range(0,nlines,1):
        # array of pixels
        lpix, rpix = (minima.x[i],minima.x[i+1])
        linelist[i]['pixl']=lpix
        linelist[i]['pixr']=rpix
        pix  = np.arange(lpix,rpix,1,dtype=np.int32)
        # sometimes the pix array covers more than can fit into the arr container
        # trim it on both sides until it fits
        if len(pix)>pixPerLine:
            k = 0
            while len(pix)>pixPerLine:
                pix = np.arange(lpix+k,rpix-k,dtype=np.int32)
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
        center  = maxima.iloc[i]
        #cen_pix = pix[np.argmax(flux)]
        local_seg = center//segsize
        # photon noise
        sumw = np.sum(pn_weights[pix])
        pn   = (299792458e0/np.sqrt(sumw)).values
        # signal to noise ratio
        snr = np.sum(flux)/np.sum(err)
        # frequency of the line
        freq    = f0_comb + N[i]*reprate
        
        arr['attr'].loc[dict(id=i,att='n')]   = N[i]
        arr['attr'].loc[dict(id=i,att='pn')]  = pn
        arr['attr'].loc[dict(id=i,att='freq')]= freq
        arr['attr'].loc[dict(id=i,att='seg')] = local_seg
        arr['attr'].loc[dict(id=i,att='bary')]= bary
        arr['attr'].loc[dict(id=i,att='snr')] = snr
        
        linelist[i]['numb']  = N[i]
        linelist[i]['noise'] = pn
        linelist[i]['freq']  = freq
        linelist[i]['segm']  = local_seg
        linelist[i]['bary']  = bary
        linelist[i]['snr']   = snr
        # calculate weights in a separate function
    # save the total flux in the order
    #print(linelist)
    arr['stat'].loc[dict(od=order,odpar='sumflux')] = np.sum(spec1d)
    return linelist
#    # first = pixel of the center of the rightmost line
#    # last = pixel of the center of the leftmost line
#    first = maxima.iloc[-1]
#    last  = maxima.iloc[0]
#    # calculate their frequencies using ThAr as reference
#    nu_right = (299792458e0/(wave1d[first]*1e-10)).values
#    nu_left  = (299792458e0/(wave1d[last]*1e-10)).values
#    # calculate cardinal number from frequencies
#    n_right  = int(round((nu_right - f0_comb)/reprate))
#    n_left   = int(round((nu_left - f0_comb)/reprate))
#    
#    freq_dsc    = np.array([(f0_comb+(n_left-j)*reprate) \
#                         for j in range(nlines)])
#    freq_asc = np.array([(f0_comb+(n_right+j)*reprate) \
#                         for j in range(nlines)])
#    fig,ax=hf.figure(2,sharex=True,ratios=[3,1])
#    ax[0].plot(wave1d,spec1d-bkg1d)
#    ax[0].axvline(ref_wave,ls=':',c='C1')
#    wave_maxima = wave1d[maxima_index]
#    flux_maxima = (spec1d-bkg1d)[maxima_index]
#    ax[0].scatter(wave_maxima,flux_maxima,marker='x',c='r')
#    [ax[0].axvline(299792458e0/f*1e10,ls=':',c='C1',lw=0.8) for f in freq_dsc]
#    [ax[0].axvline(299792458e0/f*1e10,ls='--',c='C2',lw=0.5) for f in freq_asc]
#    ax[1].scatter(wave_maxima,closeness,marker='o',s=4)
#    ax[1].scatter(wave_maxima[ref_index],closeness[ref_index],marker='x',s=10,c='r')
#    return minima
def detect_order_old(subdata,f0_comb,reprate,segsize,pixPerLine,window):
    #print("f0={0:>5.1f} GHz\tfr={1:>5.1} GHz".format(f0_comb/1e9,reprate/1e9))
    # debugging
    verbose=2
    plot=0
    # read in data to manipulate
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
    
    
    # determine the positions of minima
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
#    #maxima = maxima.where(maxima.y>fluxlim).dropna()
    # zeroth order approximation: maxima are equidistant from minima
    maxima0 = ((minima.x+np.roll(minima.x,1))/2).astype(np.int16)
    # remove 0th element (falls between minima[0] and minima[-1]) and reset index
    maxima1 = maxima0[1:]
    maxima  = maxima1.reset_index(drop=True)
    # first = pixel of the center of the rightmost line
    #first  = int(round((minima.x.iloc[-1]+minima.x.iloc[-2])/2))
    first = maxima.iloc[-1]
    # last = pixel of the center of the leftmost line
    #last   = int(round((minima.x.iloc[0]+minima.x.iloc[1])/2))
    last  = maxima.iloc[0]
    # wavelengths of maxima:
    maxima_int = maxima.values
    wave_maxima = wave1d[maxima_int]
    # convert to frequencies:
    freq_maxima = (2.99792458e8/wave_maxima*1e10).values
    # closeness of all maxima to the known modes:
    decimal_n = (freq_maxima - f0_comb)/reprate
    integer_n = np.rint(decimal_n).astype(np.int16)
    closeness = abs( decimal_n - integer_n )
    # the reference line is the one that is closest to the known mode
    # reference index is index in the list of maxima, not pixel
    ref_index = np.argmin(closeness)
    #print(maxima)
    print(ref_index)
    ref_pixel = int(maxima.iloc[ref_index])
    print(ref_pixel)
    nu_min  = (299792458e0/(wave1d[first]*1e-10)).values
    nu_max  = (299792458e0/(wave1d[last]*1e-10)).values
    nu_ref  = (299792458e0/(wave1d[ref_pixel]*1e-10)).values
    
    n_right = int(round((nu_min - f0_comb)/reprate))
    n_left  = int(round((nu_max - f0_comb)/reprate))
    n_ref   = int(round((nu_ref - f0_comb)/reprate))
    print(n_ref,integer_n[ref_index])
    print(n_left, n_ref, n_right)
    print("left = {}".format(n_ref+ref_index+1), 
          "right = {}".format(n_ref-(npeaks1-ref_index)+1))
    npeaks2  = (n_left-n_right)+1
    if plot:
        fig,ax = hf.figure(2,sharex=True,figsize=(16,9),ratios=[3,1])
        ax[0].set_title("Order = {0:2d}".format(order))
        ax[1].plot((0,4096),(0,0),ls='--',c='k',lw=0.5)
        ax[0].plot(pixels,yarray)
        ax[0].plot(minima.x,minima.y,ls='',marker='^',markersize=5,c='C1')
    # do quality control if npeaks1 != npeaks2
    
    if verbose>1:
        message=('Order = {0:>2d} '
                 'detected = {1:>8d} '
                 'inferred = {2:>8d} '
                 'start = {3:>8d}').format(order,npeaks1,npeaks2,n_right)
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
            ypos = oldminima.y
            diff = np.diff(xpos)
            pars = np.polyfit(xpos[1:],diff,2)
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
            oldminima = minima
            xpos = oldminima.x
            ypos = oldminima.y
            diff = np.diff(xpos)
            pars = np.polyfit(xpos[1:],diff,4)
            model = np.polyval(pars,xpos[1:])
            resids = diff-model
            outliers1 = resids>+5
            if plot:
                linepos2=((oldminima.x+np.roll(oldminima.x,1))/2)[1:]
                [ax[0].axvline(lp,lw=0.3,ls=':',c='C2') for lp in linepos2]
                ax[1].scatter(xpos[1:],resids,c='C0')
                ax[1].scatter(xpos[1:][outliers1],resids[outliers1],
                              marker='x',c='r')
                
        npeaks = min(npeaks1,npeaks2)
    else:
        npeaks=npeaks2
    if plot:
        
        #[0].plot(maxima.x,maxima.y,ls='',marker='x',markersize=2,c='g')
        linepos=((minima.x+np.roll(minima.x,1))/2)[1:]
        [ax[0].axvline(lp,lw=0.3,ls=':',c='C1') for lp in linepos]
        try:
            ax[0].plot(xpos[outliers2],ypos[outliers2],
                      ls='',marker='x',markersize=5,c='r')
        except:
            pass
    # with new minima, calculate the first and last n in the order
    # first = rightmost line
    first  = int(round((minima.x.iloc[-1]+minima.x.iloc[-2])/2))
    # last = leftmost line
    last   = int(round((minima.x.iloc[0]+minima.x.iloc[1])/2))
    nu_min  = (299792458e0/(wave1d[first]*1e-10)).values
    nu_max  = (299792458e0/(wave1d[last]*1e-10)).values
    nu_ref  = (299792458e0/(wave1d[ref_pixel]*1e-10)).values
    
    n_right = int(round((nu_min - f0_comb)/reprate))
    n_left  = int(round((nu_max - f0_comb)/reprate))
    
    freq1d_asc  = np.array([(f0_comb+(n_right+j)*reprate) \
                         for j in range(npeaks2)])
    # in decreasing order (wavelength increases for every element, i.e. 
    # every element is redder)
    freq1d_dsc  = np.array([(f0_comb+(n_left-j)*reprate) \
                         for j in range(npeaks2)])
    
    freq1d=freq1d_dsc
    
    # iterate over lines
    for i in range(0,npeaks,1):
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
        center  = maxima.iloc[i]
        #cen_pix = pix[np.argmax(flux)]
        local_seg = center//segsize
        # photon noise
        sumw = np.sum(pn_weights[pix])
        pn   = (299792458e0/np.sqrt(sumw)).values
        # signal to noise ratio
        snr = np.sum(flux)/np.sum(err)
        # frequency of the line
        
        freq0   = (299792458e0/(wave1d[center]*1e-10)).values
        n       = int(round((freq0 - f0_comb)/reprate))
#        n = n_left-i
        print("closeness: ",((freq0 - f0_comb)/reprate - \
                             round((freq0 - f0_comb)/reprate)),
                closeness[i])
        print("n_start+i={0:<5d} "
              "n(ThAr)={1:<5d} "
              "delta(n) = {2:<5d}".format(n_left-i, 
                                         n, (n_left-i-n)))
        freq    = f0_comb + n*reprate
#        freq    = freq1d[i]
        arr['attr'].loc[dict(id=i,att='n')]   = n
        arr['attr'].loc[dict(id=i,att='pn')]  = pn
        arr['attr'].loc[dict(id=i,att='freq')]= freq
        arr['attr'].loc[dict(id=i,att='seg')] = local_seg
        arr['attr'].loc[dict(id=i,att='bary')]= bary
        arr['attr'].loc[dict(id=i,att='snr')] = snr
        # calculate weights in a separate function
    # save the total flux in the order
    arr['stat'].loc[dict(od=order,odpar='sumflux')] = np.sum(spec1d)
    
    return arr