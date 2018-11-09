#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:40:01 2018

@author: dmilakov
"""

from harps.core import np
from harps.constants import c
#from harps.spectrum import extract_version

import harps.io as io
import harps.functions as hf
import harps.fit as fit
import harps.containers as container

#==============================================================================
#    
#                   H E L P E R     F U N C T I O N S  
#    
#==============================================================================
def evaluate(pars,x=None,startpix=None,endpix=None):
    x = x if x is not None else np.arange(startpix,endpix,1)
    return hf.polynomial(x,*pars)
def construct(coeffs,npix):
    """ For ThAr only"""
    #nbo,deg = np.shape(a)
    wavesol = np.array([evaluate(c,startpix=0,endpix=npix) for c in coeffs])
    return wavesol

allowed_calibrators = ['thar','comb']

class Wavesol(object):
    def __init__(self,**kwargs):
        
        self._cache    = {}
    
    def __getitem__(self,item):
        if item in self._cache:
            result = self._cache[item]
        else:
            if item == 'thar':
                result = self.thar
            self._cache[item] = result
        return result
    
    def __call__(self,calibrator,*args,**kwargs):
        print(calibrator)
        print(*args)
        print(**kwargs)
        assert calibrator in allowed_calibrators
        if calibrator=='thar':
            thar = ThAr(*args,**kwargs)()
            return thar
        elif calibrator == 'comb':
            comb = Comb(*args,**kwargs)()
#            version = version if version is not None else self.spectrum.version
            return comb
    def _extract_item(self,item):
        """
        utility function to extract an "item", meaning
        a extension number,name plus version.
        """
        ver=None
        if isinstance(item,tuple):
            ver_sent=True
            nitem=len(item)
            if nitem == 1:
                calibrator=item[0]
            elif nitem == 2:
                calibrator,ver=item
        else:
            ver_sent=False
            calibrator=item
        return calibrator,ver,ver_sent
        
        
    def thar(filepath,vacuum=True):
        tharsol = ThAr(filepath,vacuum)
        return tharsol()
    def comb(spec,version):
        combsol = Comb(spec,version)
        return combsol()
    
def _refrindex(pressure,ccdtemp,wavelength):
    index    = 1e-6*pressure*(1.0+(1.049-0.0157*ccdtemp)*1e-6*pressure) \
                /720.883/(1.0+0.003661*ccdtemp) \
                *(64.328+29498.1/(146.0-2**(1e4/wavelength)) \
                +255.4/(41.0-2**(1e4/wavelength)))+1.0
    return index
def _to_vacuum(lambda_air,pressure=760,ccdtemp=15):
    """
    Returns wavelengths in vacuum.
    
    Args:    
        lambda_air: 1D numpy array
    Returns:
        lambda_vacuum : 1D numpy array
    """
    if np.sum(lambda_air)!=0:
        index = _refrindex(pressure,ccdtemp,lambda_air)
    else:
        index = 1
    lambda_vacuum = lambda_air*index
    return lambda_vacuum
def _to_air(lambda_vacuum,pressure=760,ccdtemp=15):
    """
    Returns wavelengths in air.
    
    Args:    
        lambda_air: 1D numpy array
    Returns:
        lambda_vacuum : 1D numpy array
    """
    assert lambda_vacuum.sum()!=0, "Wavelength array is empty."
    index      = _refrindex(pressure,ccdtemp,lambda_vacuum)
    lambda_air = lambda_vacuum/index
    return lambda_air
#==============================================================================
#    
#               T H O R I U M    A R G O N     F U N C T I O N S  
#    
#==============================================================================   
class ThAr(object):
    def __init__(self,filepath,vacuum):
        self._filepath = filepath
        self._vacuum   = vacuum
        pass
    def __call__(self):
        dispers2d, bad_orders = self._thar(self._filepath,self._vacuum)
        self.bad_orders = bad_orders
        return dispers2d
    @staticmethod
    def _get_wavecoeff_air(filepath):
        ''' 
        Returns coefficients of a third-order polynomial from the FITS file 
        header in a matrix. This procedure is described in the HARPS DRS  
        user manual.
        https://www.eso.org/sci/facilities/lasilla/
                instruments/harps/doc/DRS.pdf
        '''
        def _read_wavecoef1d(order):
            """ 
            Returns ThAr wavelength calibration coefficients saved in the header.
            Returns zeroes when no coefficients are found.
            """
            coeffs = np.zeros(deg+1)
            for i in range(deg+1):                    
                ll    = i + order*(deg+1)
                try:
                    coeffs[i] = header["ESO DRS CAL TH COEFF LL{0}".format(ll)]
                except:
                    continue
            return coeffs
        
        header = io.read_e2ds_header(filepath)
        meta   = io.read_e2ds_meta(filepath)
        nbo    = meta['nbo']
        deg    = meta['d']
        coeffs = np.array(list(map(_read_wavecoef1d,range(nbo))))
        bad_orders = np.where(np.sum(coeffs,axis=1)==0)[0]
        
        return coeffs, bad_orders
    def get_wavecoeff_vacuum(self):    
        wavesol_vacuum = self._thar(self._filepath,True)
        meta   = io.read_e2ds_meta(self._filepath)
        nbo    = meta['nbo']
        deg    = meta['d']
        npix   = meta['npix'] 
        coeff2d  = np.zeros((nbo,deg+1))
        for order in range(nbo):
            coeff2d[order] = np.polyfit(np.arange(npix),
                                        wavesol_vacuum[order],
                                        deg)[::-1]
        return coeff2d
    @staticmethod
    def _thar(filepath,vacuum=True,npix=4096):
        """ 
        Return the ThAr wavelength solution, as saved in the header of the
        e2ds file. 
        """
        coeffs, bad_orders = ThAr._get_wavecoeff_air(filepath)
        wavesol_air = construct(coeffs,npix)
        if vacuum==True:
            return _to_vacuum(wavesol_air), bad_orders
        else:
            return wavesol_air, bad_orders



#==============================================================================
#    
#               L A S E R    F R E Q U E N C Y    C O M B
#
#                            F U N C T I O N S  
#    
#==============================================================================
class Comb(object):
    def __init__(self,spec,version,fittype='gauss'):
        self._spectrum = spec
        self._version  = version
        self._fittype  = fittype
    def __call__(self):
        return self._comb(self._version,self._fittype)
    
    
    def _comb(self,version,fittype='gauss'):
        spec         = self._spectrum
        coefficients = self.get_wavecoeff_comb()
        wavesol_comb = self._construct_from_combcoeff(coefficients,spec.npix)
    
        return wavesol_comb
    
    # stopped here, 29 Oct 2018
    def residuals(self,*args,**kwargs):
        spec         = self._spectrum
        version      = self._version
        fittype      = self._fittype
        linelist     = spec['linelist']
        coefficients = spec['coeff',version]
        
        centers      = linelist[fittype][:,1]
        wavelengths  = hf.freq_to_lambda(linelist['freq'])
        nlines       = len(linelist)
        residuals    = container.residuals(nlines)
        for coeff in coefficients:
            order = coeff['order']
            segm  = coeff['segm']
            pixl  = coeff['pixl']
            pixr  = coeff['pixr']
            cut   = np.where((linelist['order']==order) & 
                             (centers >= pixl) &
                             (centers <= pixr))
            centsegm = centers[cut]
            wavereal  = wavelengths[cut]
            wavefit   = evaluate(coeff['pars'],centsegm)
            residuals['order'][cut]=order
            residuals['segm'][cut]=segm
            residuals['residual'][cut]=(wavereal-wavefit)/wavereal*c
        residuals['gauss'] = centers

        return residuals
            
    def get_wavecoeff_comb(self):
        """
        Returns a dictionary with the wavelength solution coefficients derived from
        LFC lines
        """
        spec      = self._spectrum
        version   = self._version
        fittype   = self._fittype
        linelist  = spec['linelist']
        wavesol2d = fit.dispersion(linelist,version,fittype)
        return wavesol2d
    def _construct_order(self,coeffs,npix):
        wavesol1d  = np.zeros(npix)
        for segment in coeffs:
            pixl = segment['pixl']
            pixr = segment['pixr']
            pars = segment['pars']
            wavesol1d[pixl:pixr] = evaluate(pars,None,pixl,pixr)
        return wavesol1d
    
    #def construct_from_combcoeff1d(coeffs,npix,order):
    #    cfs = coeffs[hf.get_extname(order)]
    #    wavesol1d = construct_order(cfs,npix) 
    #    return wavesol1d
    def _construct_from_combcoeff(self,coeffs,npix):
        orders    = np.unique(coeffs['order'])
        nbo       = np.max(orders)+1
        
        wavesol2d = np.zeros((nbo,npix))
        for order in orders:
            coeffs1d = coeffs[np.where(coeffs['order']==order)]
            wavesol2d[order] = self._construct_order(coeffs1d,npix)
            
        return wavesol2d

