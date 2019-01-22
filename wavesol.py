#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:40:01 2018

@author: dmilakov
"""

from harps.core import np, plt
from harps.constants import c
#from harps.spectrum import extract_version

import harps.io as io
import harps.functions as hf
import harps.fit as fit
import harps.containers as container
import harps.plotter as plot

#==============================================================================
#    
#                   H E L P E R     F U N C T I O N S  
#    
#==============================================================================
def evaluate(pars,x=None,startpix=None,endpix=None):
    if startpix and endpix:
        assert startpix<endpix, "Starting pixel larger than ending pixel"
    x = x if x is not None else np.arange(startpix,endpix,1)
    return np.polyval(pars[::-1],x)

def evaluate2d(coefficients,linelist,fittype='gauss',errors=False):
    """
    Returns 1d array of wavelength of all lines from linelist, as calculated
    from the coefficients. 
    """
    centers = linelist[fittype][:,1]
    centerr = linelist['{0}_err'.format(fittype)][:,1]
    wave    = np.zeros(len(centers)) 
    waverr  = np.zeros(len(centers))
    for coeff in coefficients:
        order = coeff['order']
        pixl  = coeff['pixl']
        pixr  = coeff['pixr']
        cut   = np.where((linelist['order']==order) & 
                         (centers >= pixl) &
                         (centers <= pixr))
        centsegm = centers[cut]
        pars     = coeff['pars']
        wavesegm = evaluate(pars,centsegm)
        wave[cut] = wavesegm
        if errors:
            derivpars = (np.arange(len(pars))*pars)[1:]
            waverr[cut] = evaluate(derivpars,centsegm)
    if errors:
        return wave, waverr
    else:
        return wave
def dispersion(coeffs2d,npix):
    wavesol = disperse2d(coeffs2d,npix)
    return wavesol

def disperse1d(coeffs,npix):
    wavesol1d  = np.zeros(npix)
    for segment in coeffs:
        pixl = segment['pixl']
        pixr = segment['pixr']
        pars = segment['pars']
        wavesol1d[pixl:pixr] = evaluate(pars,None,pixl,pixr)
    return wavesol1d
def disperse2d(coeffs,npix):
    orders    = np.unique(coeffs['order'])
    nbo       = np.max(orders)+1
    
    wavesol2d = np.zeros((nbo,npix))
    for order in orders:
        coeffs1d = coeffs[np.where(coeffs['order']==order)]
        wavesol2d[order] = disperse1d(coeffs1d,npix)
        
    return wavesol2d

def construct(coeffs,npix):
    """ For ThAr only"""
    #nbo,deg = np.shape(a)
    wavesol = np.array([evaluate(c,startpix=0,endpix=npix) for c in coeffs])
    
    return wavesol


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

def residuals(linelist,coefficients,fittype='gauss'):
    centers      = linelist[fittype][:,1]
    photnoise    = linelist['noise']
    wavelengths  = hf.freq_to_lambda(linelist['freq'])
    nlines       = len(linelist)
    result       = container.residuals(nlines)
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
        result['order'][cut]=order
        result['segm'][cut]=segm
        result['residual'][cut]=(wavefit-wavereal)/wavereal*c
        result['noise'][cut] = photnoise[cut]
    result[fittype] = centers
    
    return result
def twopoint_coeffs(linelist,fittype='gauss',exclude_gaps=True,*args,**kwargs):
    """ Uses the input array to return the coefficients of the wavelength 
    calibration by interpolating between neighbouring comb lines.
    """
    def exclude(coeffs,lim):
        cut = ((coeffs['pixl']<=lim)&(coeffs['pixr']>=lim))
        #print(lim,np.sum(cut))
        return coeffs[~cut]
    MOD    = 2
    numseg = len(linelist)-1
    coeffs = container.coeffs(1,numseg) 
    for i in range(numseg):
        if linelist['order'][i]!=linelist['order'][i+1]:
            continue
        order = linelist['order'][i]
        left  = linelist[fittype][i,1]
        right = linelist[fittype][i+1,1]
        if left>right:
            continue
        if np.isfinite(left) and np.isfinite(right):
            pass
        else:
            continue
        pixl  = left#np.int(np.around(left/MOD)*MOD)
        pixr  = right#np.int(np.around(right/MOD)*MOD)
        
        waveL = hf.freq_to_lambda(linelist['freq'][i])
        waveR = hf.freq_to_lambda(linelist['freq'][i+1])
        # y(x) = a0 + a1*x
        a0    = waveL - (waveR-waveL)/(pixr-pixl)*pixl
        a1    = (waveR-waveL)/(pixr-pixl)
        coeffs[i]['order'] = order
        coeffs[i]['segm']  = i
        coeffs[i]['pixl']  = pixl#np.int(np.around(left/MOD)*MOD)#pixl
        coeffs[i]['pixr']  = pixr#np.int(np.around(right*MOD)/MOD)#pixr
        coeffs[i]['pars']  = [a0,a1]
    if exclude_gaps:
        seglims = np.linspace(512*1,512*8,8)
        coeffs0 = np.copy(coeffs)
        for lim in seglims:
            coeffs0  = exclude(coeffs0,lim)
        coeffs = coeffs0
    #dispersion = disperse2d(coeffs,npix)
    #np.place(dispersion,dispersion==0,np.nan)
    return coeffs
def twopoint(linelist,fittype='gauss',npix=4096,full_output=False,
             exclude_gaps=True,*args,**kwargs):
    """ Uses the input array to return the coefficients of the wavelength 
    calibration by interpolating between neighbouring comb lines.
    """
    
    coeffs = twopoint_coeffs(linelist,fittype,exclude_gaps,*args,**kwargs)
    dispersion = disperse2d(coeffs,npix)
    np.place(dispersion,dispersion==0,np.nan)
    if full_output:
        return dispersion, coeffs
    else:
        return dispersion
def polynomial(linelist,version=500,fittype='gauss',npix=4096,
               full_output=False,*args,**kwargs):
    coeffs = fit.dispersion(linelist,version,fittype)
    dispersion = disperse2d(coeffs,npix)
    
    if full_output:
        return dispersion, coeffs
    else:
        return dispersion
    
allowed_calibrators = ['thar','comb']
#==============================================================================
#    
#                       W A V E S O L    C L A S S  
#    
#==============================================================================
class Wavesol(object):
    __metric = {"A":1,"m":1e10,'m/s':1}
    def __init__(self,narray,unit='A'):
        self._values  = narray
        self._unit    = unit
    def __getitem__(self,item):
        order,pix, pix_sent = self._extract_item(item)
        
        values = self.values
        if pix_sent:
            return Wavesol(values[order,pix],self.unit)
        else:
            return Wavesol(values[order],self.unit)
        
    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning order plus pixel limits.
        
        To be used with partial decorator
        """
        pix_sent = False
        pix      = None
        if isinstance(item,tuple):
            if len(item)==2: 
                pix_sent=True
                order = item[0]
                pix   = item[1]
            if len(item)==1:
                order = item
        else:
            order = item
        return order, pix, pix_sent
    def __add__(self,other):
        values0 = self._into_angstrom() 
        values1 = other._into_angstrom()
        
        return Wavesol((values0+values1)/Wavesol.__metric[self.unit],self.unit)
    def __sub__(self,other):
        values0 = self._into_angstrom() 
        values1 = other._into_angstrom()
        
        return Wavesol((values0-values1)/Wavesol.__metric[self.unit],self.unit)
    def __neg__(self):
        return Wavesol(-self.values,self.unit)
    def __pos__(self):
        return Wavesol(self.values,self.unit)
    def __truediv__(self,other):
        dv      = np.zeros_like(self.values)
        values0 = self._into_angstrom() 
        values1 = other._into_angstrom()
        cut0    = values0!=0
        cut1    = values1!=0
        cut     = np.where(cut0&cut1)
        
        dv[cut] = (values1[cut]-values0[cut])/values1[cut] * c
        return Wavesol(dv,'m/s')
    def _into_angstrom(self):
        return self.values * Wavesol.__metric[self.unit]
    
    @property
    def values(self):
        return self._values
    @property
    def unit(self):
        return self._unit
    @property
    def shape(self):
        return np.shape(self.values)
    def plot(self,plot2d=False,*args,**kwargs):
        figure = plot.Figure(1)
        fig    = figure.fig
        ax     = figure.axes
        values = self.values
        sumord = np.sum(values,axis=1)
        orders = np.where(sumord!=0)[0]
        
        if plot2d:
            
            image = ax[0].imshow(values,aspect='auto')
            cbar  = fig.colorbar(image)
            cbar.set_label(self.unit)
        else:
            numord = len(orders)
            colors = plt.cm.Vega10(np.linspace(0,1,10))
            if numord>5:
                colors = plt.cm.jet(np.linspace(0,1,numord))
            for i,order in enumerate(orders):
                if np.sum(values[order])==0:
                    continue
                ax[0].plot(values[order],c=colors[i],lw=0.8)
        return figure
                
#==============================================================================
#    
#               T H O R I U M    A R G O N     F U N C T I O N S  
#    
#==============================================================================   
class ThAr(object):
    def __init__(self,filepath,vacuum):
        self._filepath  = filepath
        self._vacuum    = vacuum
        self._coeffs_air = None 
        self._coeffs_vac = None
        self._bad_orders = None
        self._qc         = None
        pass
    def __call__(self):
        dispers2d, bad_orders, qc = self._thar(self._filepath,self._vacuum)
        self._bad_orders = bad_orders
        self._qc          = qc
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
            coeff1d = container.coeffs(deg,1)
            coeff1d['order'] = order
            coeff1d['pixl']  = 0
            coeff1d['pixr']  = 4095
            for i in range(deg+1):                    
                ll    = i + order*(deg+1)
                try:
                    a = header["ESO DRS CAL TH COEFF LL{0}".format(ll)]
                    coeff1d['pars'][0,i] = a
                except:
                    continue
            return coeff1d
        
        header = io.read_e2ds_header(filepath)
        meta   = io.read_e2ds_meta(filepath)
        nbo    = meta['nbo']
        deg    = meta['d']
        coeffs = np.vstack([_read_wavecoef1d(order) for order in range(nbo)])
        bad_orders = np.unique(np.where(np.sum(coeffs['pars'],axis=1)==0)[0])
        qc     = meta['qc']
        return coeffs, bad_orders, qc
    def get_wavecoeff_vacuum(self):    
        def get1d(order):
            
            if order not in bad_orders:
                pars   = np.polyfit(np.arange(npix),
                                    wavesol_vacuum[order],
                                    deg)
            else:
                pars = np.zeros((1,deg+1))
            return (order,0,0,4095,0,np.flip(pars),np.zeros_like(pars))
        
        wavesol_vacuum, bad_orders, qc = self._thar(self._filepath,True)
        meta   = io.read_e2ds_meta(self._filepath)
        nbo    = meta['nbo']
        deg    = meta['d']
        npix   = meta['npix'] 

        coeff2d = container.coeffs(deg,nbo)
        for order in range(nbo):
            coeff2d[order] = get1d(order)
        
        return coeff2d
    @staticmethod
    def _thar(filepath,vacuum=True,npix=4096):
        """ 
        Return the ThAr wavelength solution, as saved in the header of the
        e2ds file. 
        """
        coeffs, bad_orders, qc = ThAr._get_wavecoeff_air(filepath)
        wavesol_air = construct(coeffs['pars'][:,0,:],npix)
        if vacuum==True:
            return _to_vacuum(wavesol_air), bad_orders, qc
        else:
            return wavesol_air, bad_orders, qc
    @property
    def coeffs(self):
        if self._vacuum==False:
            if self._coeffs_air is not None:
                pass
            else:
                coeffs,bad_orders,qc = ThAr._get_wavecoeff_air(self._filepath)
                self._coeffs_air = coeffs
            return self._coeffs_air
        if self._vacuum==True:
            if self._coeffs_vac is not None:
                pass
            else:
                self._coeffs_vac = self.get_wavecoeff_vacuum()
            return self._coeffs_vac

    @property
    def bad_orders(self):
        if self._bad_orders is not None:
            pass
        else:
            coeffs,bad_orders,qc  = ThAr._get_wavecoeff_air(self._filepath)
            self._bad_orders = bad_orders
        return self._bad_orders
    @property
    def qc(self):
        if self._qc is not None:
            pass
        else:
            coeffs,bad_orders,qc = ThAr._get_wavecoeff_air(self._filepath)
            self._qc = qc
        return self._qc
        
#==============================================================================
#    
#               L A S E R    F R E Q U E N C Y    C O M B
#
#                            F U N C T I O N S  
#    
#==============================================================================
def get_wavecoeff_comb(linelist,version,fittype):
    """
    Returns a dictionary with the wavelength solution coefficients derived from
    LFC lines
    """
    if version==1:
        coeffs2d = twopoint_coeffs(linelist,fittype)
    else:
        coeffs2d = fit.dispersion(linelist,version,fittype)
    return coeffs2d

def comb_dispersion(linelist,version,fittype,npix,*args,**kwargs):
    if version==1:
        wavesol_comb = twopoint(linelist,fittype,npix,*args,**kwargs)
    else:
        wavesol_comb = polynomial(linelist,version,fittype,npix,*args,**kwargs)
    #coeffs2d = get_wavecoeff_comb(linelist,version,fittype)
    #wavesol_comb = dispersion(coeffs2d,npix)
    return wavesol_comb


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
    def dispersion(self,version=None,fittype=None):
        version = version if version is not None else self._version
        fittype = fittype if fittype is not None else self._fittype
        return self._comb(version,fittype)
    # stopped here, 29 Oct 2018
    def residuals(self,*args,**kwargs):
        spec         = self._spectrum
        version      = self._version
        fittype      = self._fittype
        linelist     = spec['linelist']
        coefficients = spec['coeff',version]
        
        polyord, gaps, segmented = hf.extract_version(version)
        if gaps:
            gaps1d = fit.read_gaps()
            centers_w_gaps = fit.introduce_gaps(linelist[fittype][:,1],gaps1d)
            linelist[fittype][:,1] = centers_w_gaps
        resid  = residuals(linelist,coefficients)

        return resid
            
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

    @property
    def coeffs(self):
        return self.get_wavecoeff_comb()