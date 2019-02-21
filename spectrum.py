#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:45:04 2018

@author: dmilakov
"""
from harps.core import sys
from harps.core import np, xr
from harps.core import os
from harps.core import leastsq, curve_fit,  interpolate
from harps.core import fits, FITS, FITSHDR
from harps.core import plt
from harps.core import warnings

#from multiprocessing import Pool

from harps import functions as hf
from harps import settings as hs
from harps import io
from harps import wavesol as ws
from harps import background
from harps import lines

from harps.constants import c

from harps.plotter import SpectrumPlotter, Figure

version      = hs.__version__
harps_home   = hs.harps_home
harps_data   = hs.harps_data
harps_dtprod = hs.harps_dtprod
harps_plots  = hs.harps_plot
harps_prod   = hs.harps_prod

sOrder       = hs.sOrder
eOrder       = hs.eOrder
nOrder       = eOrder-sOrder



class Spectrum(object):
    ''' Spectrum object contains functions and methods to read data from a 
        FITS file processed by the HARPS pipeline
    '''
    def __init__(self,filepath=None,LFC='HARPS',model='SingleGaussian',
                 overwrite=False,ftype=None,sOrder=None,eOrder=None,**kwargs):
        '''
        Initialise the spectrum object.
        '''
        self.filepath = filepath
        self.name     = "HARPS Spectrum"
        self.lfcname  = LFC
        basename_str  = os.path.basename(filepath)
        filename_str  = os.path.splitext(basename_str)[0]
        filetype_str  = filename_str.split('_')[1]
        self.filetype = ftype if ftype is not None else filetype_str
        self.data     = io.read_e2ds_data(filepath)
        self.hdrmeta  = io.read_e2ds_meta(filepath)
        self.header   = io.read_e2ds_header(filepath)
        # include anchor offset if provided (in Hz)
        anchor_offset = kwargs.pop('anchor_offset',0)
        self.lfckeys  = io.read_LFC_keywords(filepath,LFC,anchor_offset)
        self.meta     = self.hdrmeta
        
        
        self.npix     = self.meta['npix']
        self.nbo      = self.meta['nbo']
        self.d        = self.meta['d']
        self.sOrder   = sOrder if sOrder is not None else hs.sOrder
        self.eOrder   = eOrder if eOrder is not None else self.meta['nbo']
        
        self.model    = model
        
        self.version  = self._item_to_version()
        versiondict   = self._version_to_dict(self.version)
        self.polyord  = versiondict['polyord']
        self.gaps     = versiondict['gaps']
        self.segment  = versiondict['segment']
        
        
        self.segsize  = self.npix//16 #pixel
        varmeta       = dict(sOrder=self.sOrder,polyord=self.polyord,
                             gaps=self.gaps,segment=self.segment,
                             segsize=self.segsize,model=self.model)
        self.meta.update(varmeta)
        
        self._cache   = {}
        try:
            self.ThAr = ws.ThAr(self.filepath,vacuum=True)
        except:
            self.ThAr = None
            
            
        self.datetime = np.datetime64(self.meta['obsdate'])
        dirpath       = kwargs.pop('dirpath',None)
        self._outfits = io.get_fits_path('fits',filepath,version,dirpath)
        self._hdu     = FITS(self._outfits,'rw',clobber=overwrite)
        self.write_primaryheader(self._hdu)
        #self.wavesol  = Wavesol(self)
        
        
        
        
        return
    def __getitem__(self,item):
        '''
        Tries reading data from file, otherwise runs __call__. 
        
        Args:
        ----
            datatype (str) :  ['linelist','coeff','model_gauss',
                               'wavesol_comb','wavesol_thar']
            save     (bool):  saves to the FITS file if true
        
        Returns:
        -------
            data (array_like) : values of dataset
            
        '''
        ext, ver, versent = self._extract_item(item)
        mess = "Extension {ext:>20}, version {ver:<5}:".format(ext=ext,ver=ver)
        hdu  = self._hdu
        try:
            data    = hdu[ext,ver].read()
            mess   += " read from file."
        except:
            data   = self.__call__(ext,ver)
            header = self.return_header(ext)
            hdu.write(data=data,header=header,extname=ext,extver=ver)
            mess   += " calculated."
        finally:
            print(mess)
            pass
        return data

    def __str__(self):
        meta     = self.meta
        dirname  = os.path.dirname(self.filepath)
        basename = os.path.basename(self.filepath)
        mess =  "{0:^80s} \n".format("S P E C T R U M")+\
                "{0:-^80s} \n".format("")+\
                "{0:<20s}:{1:>60s}\n".format("Directory",dirname)+\
                "{0:<20s}:{1:>60s}\n".format("File",basename)+\
                "{0:<20s}:{1:>60s}\n".format("LFC",self.lfcname)+\
                "{0:<20s}:{1:>60s}\n".format("Obsdate",meta['obsdate'])+\
                "{0:<20s}:{1:>60s}\n".format("Model",meta['model'])
        return mess
    
    def __call__(self,dataset,version=None,write=False,*args,**kwargs):
        """ 
        
        Calculate dataset.
        
        Parameters are dataset name and version. 
        
        Args:
        ----
        dataset (str) : name of the dataset
        version (int) : version number, 3 digit (PGS)
                        P = polynomial order
                        G = gaps
                        S = segment
        
        Returns:
        -------
        data (array_like) : values of dataset
            
        """
        def funcargs(name):
            if name =='coeff_gauss':
                args = (self['linelist'],version,'gauss')
            if name =='coeff_lsf':
                args = (self['linelist'],version,'lsf')
            elif name=='wavesol_gauss':
                args = (self['linelist'],version,'gauss',self.npix)
            elif name=='wavesol_lsf':
                args = (self['linelist'],version,'lsf',self.npix)
            elif name=='residuals_gauss':
                args = (self['linelist'],self['coeff_gauss',version],
                        version,'gauss')
            elif name=='residuals_lsf':
                args = (self['linelist'],self['coeff_lsf',version],
                        version,'lsf')
            elif name=='wavesol_2pt_gauss':
                args = (self['linelist'],'gauss',self.npix)
            elif name=='wavesol_2pt_lsf':
                args = (self['linelist'],'lsf',self.npix)
            return args
        assert dataset in io.allowed_hdutypes, "Allowed: {}".format(io.allowed_hdutypes)
        version = self._item_to_version(version)
        functions = {'linelist':lines.detect,
                     'coeff_gauss':ws.get_wavecoeff_comb,
                     'coeff_lsf':ws.get_wavecoeff_comb,
                     'wavesol_gauss':ws.comb_dispersion,
                     'wavesol_lsf':ws.comb_dispersion,
                     'model_gauss':lines.model_gauss,
                     'model_lsf':lines.model_lsf,
                     'residuals_gauss':ws.residuals,
                     'residuals_lsf':ws.residuals,
                     'wavesol_2pt_gauss':ws.twopoint,
                     'wavesol_2pt_lsf':ws.twopoint,
                     'weights':self.get_weights2d,
                     'error':self.get_error2d,
                     'background':self.get_background,
                     'envelope':self.get_envelope}
        if dataset in ['coeff_gauss','coeff_lsf',
                       'wavesol_gauss','wavesol_lsf',
                       'residuals_gauss','residuals_lsf',
                       'wavesol_2pt_gauss','wavesol_2pt_lsf']:
            data = functions[dataset](*funcargs(dataset))
        elif dataset in ['weights','background','error','envelope']:
            data = functions[dataset]()
        elif dataset in ['linelist','model_gauss','model_lsf']:
            data = functions[dataset](self,*args,**kwargs)
        elif dataset in ['flux']:
            data = getattr(self,'data')
        if write:
            hdu = self._hdu
            header = self.return_header(dataset)
            hdu.write(data=data,header=header,extname=dataset,extver=version)
        return data
    def __del__(self):
        """
        Closes the output HDU for this object.
        """
        self._hdu.close()
        return
    @staticmethod
    def _version_to_dict(ver):
        """ 
        Converts the integer representation of settings into a dictionary.
        
        Args:
        ----
        ver (int) : version number
        
        Returns:
        -------
        dict : (polyord,gaps,segment)
        """
        if isinstance(ver,int) and ver>99 and ver<1000:
            split  = [int((ver/10**x)%10) for x in range(3)][::-1]
            polyord, gaps, segment = split
        return dict(polyord=polyord,gaps=gaps,segment=segment)
    def _item_to_version(self,item=None):
        """
        Returns an integer representing the settings provided
        
        Returns the default version if no args provided.
        
        Args:
        -----
        item (dict,int,tuple) : contains information on the version
        
        Returns:
        -------
        version (int): 
        """

        return hf.item_to_version(item)
    
    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning
        a extension number,name plus version.
        """
        ver=0.
        if isinstance(item,tuple):
            ver_sent=True
            nitem=len(item)
            if nitem == 1:
                ext=item[0]
            elif nitem == 2:
                ext,ver=item
        else:
            ver_sent=False
            ext=item
        
        ver = hf.item_to_version(ver)
        return ext,ver,ver_sent
    def write(self,item):
        """
        Writes the input item (extension plus version) to the output HDU file.
        Equivalent to __call__(item,write=True).
        """
        ext, ver, versent = self._extract_item(item)
        hdu    = self._hdu
        data   = self.__call__(ext,ver)
        header = self.return_header(ext)
        if versent:
            hdu.write(data=data,header=header,extname=ext,extver=ver)
        else:
            hdu.write(data=data,header=header,extname=ext)
        return data
    def write_primaryheader(self,hdu):
        ''' Writes the spectrum metadata to the HDU header'''
        header = self.return_header('primary')
        hdu[0].write_keys(header)
        return 
    def return_header(self,extension):
        """
        Returns a FITSHDR object with header information for this extension.
        """
        meta = self.meta
        LFC  = self.lfckeys
        # ------- Reads metadata and LFC keywords
        
        def return_value(name):
            if name=='Simple':
                value = True
            elif name=='Bitpix':
                value = 32
            elif name=='Naxis':
                value = 0
            elif name=='Extend':
                value = True
            elif name=='Author':
                value = 'Dinko Milakovic'
            elif name=='version':
                value = version
            elif name=='npix':
                value = meta['npix']
            elif name=='mjd':
                value = meta['mjd']
            elif name=='date-obs':
                value = meta['obsdate']
            elif name=='fibshape':
                value = meta['fibshape']
            elif name=='lfc':
                value = LFC['name'],
            elif name=='reprate':
                value = LFC['comb_reprate']
            elif name=='anchor':
                value = LFC['comb_anchor']
            elif name=='gaps':
                value = meta['gaps']
            elif name=='segment':
                value = meta['segment']
            elif name=='polyord':
                value = meta['polyord']
            elif name=='model':
                value = meta['model']
            elif name=='totflux':
                value = np.sum(self.data)
            elif name=='totnoise':
                value = self.sigmav()
            return value
        def make_dict(name,value,comment=''):
            return dict(name=name,value=value,comment=comment)
            
        if extension == 'primary':
            names = ['Simple','Bitpix','Naxis','Extend','Author',
                     'npix','mjd','date-obs','fibshape','totflux']
            
        elif extension == 'linelist':
            names = ['version','totflux']
        elif extension in ['wavesol_gauss','wavesol_lsf']:
            names = ['lfc','anchor','reprate','gaps','segment','polyord']
        elif extension in ['coeff_gauss','coeff_lsf']:
            names = ['gaps','segment','polyord']
        elif extension in ['model_gauss', 'model_lsf']:
            names = ['model']
        elif extension in ['residuals_gauss', 'residuals_lsf']:
            names = ['lfc','anchor','reprate','gaps','segment','polyord']
        elif extension in ['wavesol_2pt_gauss','wavesol_2pt_lsf']:
            names = ['lfc','anchor','reprate']
        elif extension == 'weights':
            names = ['version','lfc']
        elif extension in ['flux','error','background','envelope']:
            names = ['totflux']
        else:
            raise UserWarning("HDU type not recognised")

        comments_dict={'Simple':'Conforms to FITS standard',
                  'Bitpix':'Bits per data value',
                  'Naxis':'Number of data axes',
                  'Extend':'FITS dataset may contain extensions',
                  'Author':'',
                  'version':'Code version used',
                  'npix':'Number of pixels',
                  'mjd':'Modified Julian Date',
                  'date-obs':'Date of observation',
                  'fibshape':'Fibre shape',
                  'lfc':'LFC name',
                  'reprate':'LFC repetition frequency',
                  'anchor':'LFC offset frequency',
                  'gaps':'Shift lines using gap file',
                  'segment':'Fit wavelength solution in 512 pix segments',
                  'polyord':'Polynomial order of the wavelength solution',
                  'model':'EmissionLine class used to fit lines',
                  'totflux':'Total flux in the exposure',
                  'totnoise':'Photon noise of the exposure [m/s]'}
        values_dict = {name:return_value(name) for name in names}
        if extension=='primary':
            for order in range(self.nbo):
                name = 'fluxord{0:02d}'.format(order)
                names.append(name)
                values_dict[name] = np.sum(self.data[order])
                comments_dict[name] = "Total flux in order {0:02d}".format(order)
        
        values   = [values_dict[name] for name in names]
        comments = [comments_dict[name] for name in names]
        header   = [make_dict(n,v,c) for n,v,c in zip(names,values,comments)]
        return FITSHDR(header)


    @property
    def error(self,*args):
        """
        Returns the 2d error on flux values for the entire exposure. Caches the
        array.
        """
        try:
            error2d = self._cache['error2d']
        except:
            error2d = self.get_error2d(*args)  
            self._cache['error2d']=error2d
        return error2d
    def get_error2d(self):
        """
        Returns a 2d array with errors on flux values. Adds the error due to 
        background subtraction in quadrature to the photon counting error.
        """
        data2d  = np.abs(self.data)
        bkg2d   = background.get2d(self)
        error2d = np.sqrt(np.abs(data2d) + np.abs(bkg2d))
        return error2d
    
    def get_error1d(self,order,*args):
        """
        Returns a 1d array with errors on flux values for this order. 
        Adds the error due to background subtraction in quadrature to the 
        photon counting error.
        """
        data1d  = np.abs(self.data[order])
        bkg1d   = np.abs(background.get1d(self,order,*args))
        error1d = np.sqrt(data1d + bkg1d)
        return error1d
    @property
    def background(self):
        """
        Returns the 2d background model for the entire exposure. Caches the
        array.
        """
        try:
            bkg2d = self._cache['background2d']
        except:
            bkg2d = background.get2d(self)
            self._cache['background2d']=bkg2d
        return bkg2d
    def get_background(self,*args):
        """
        Returns the 2d background model for the entire exposure. 
        """
        return background.get2d(self,*args)
    
    def get_background1d(self,order,*args):
        """
        Returns the 1d background model for this order. 
        """
        return background.get1d(self,order,*args)
    @property
    def envelope(self):
        """
        Returns the 2d background model for the entire exposure. Caches the
        array.
        """
        try:
            env2d = self._cache['envelope2d']
        except:
            env2d = background.getenv2d(self)
            self._cache['envelope2d']=env2d
        return env2d
    def get_envelope(self,*args):
        """
        Returns the 2d background model for the entire exposure. 
        """
        return background.getenv2d(self,*args)
    
    def get_envelope1d(self,order,*args):
        """
        Returns the 1d background model for this order. 
        """
        return background.getenv1d(self,order,*args)
    
    @property
    def weights(self):
        """
        Returns the photon noise weights for each pixel in the spectrum
        """
        try:
            weights2d = self._cache['weights2d']
        except:
            weights2d = self.get_weights2d()  
            self._cache['weights2d']=weights2d
        return weights2d
    def get_weights1d(self,order):
        """
        Calculates the photon noise of this order.
        """
        sigma_v = self.sigmav1d(order)
        return (sigma_v/c)**-2
    
    def get_weights2d(self):
        """
        Calculates the photon noise of the entire spectrum.
        """
        sigmav2d = self.sigmav2d()
        return (sigmav2d/c)**-2
    def sigmav(self,order=None,unit='mps'):
        """
        Calculates the theoretical limiting velocity precision from the photon 
        noise weights of this order(s).
        """
        orders = self.prepare_orders(order)
        precision_order = [1./np.sqrt(np.sum(self.get_weights1d(order))) \
                           for order in orders]
        precision_total = 1./np.sqrt(np.sum(np.power(precision_order,-2)))
        if unit == 'mps':
            fac = 2.99792458e8
        else:
            fac = 1.
        return precision_total * fac
    def sigmav2d(self):
        """
        Calculates the limiting velocity precison of all pixels in the spectrum
        using ThAr wavelengths.
        """
        orders  = np.arange(self.nbo)
        sigma_v = np.array([self.sigmav1d(order) for order in orders])
        return sigma_v

    def sigmav1d(self,order):
        """
        Calculates the limiting velocity precison of all pixels in the order
        using ThAr wavelengths.
        """
        data    = self.data[order]
        thar    = self.tharsol[order]
        err     = self.error[order]
        # weights for photon noise calculation
        # Equation 5 in Murphy et al. 2007, MNRAS 380 p839
        #pix2d   = np.vstack([np.arange(spec.npix) for o in range(spec.nbo)])
        df_dlbd = hf.derivative1d(data,thar)
        sigma_v = c*err/(thar*df_dlbd)
        return sigma_v
    
    @property
    def tharsol(self):
        """
        Returns the 2d ThAr wavelength calibration for this exposure. Caches
        the array.
        """
        thardisp2d = self._tharsol()
        self._cache['tharsol'] = thardisp2d
        return thardisp2d

    @property
    def ThAr(self):
        return self._tharsol
    @ThAr.setter
    def ThAr(self,tharsol):
        """ Input is a wavesol.ThAr object """
        self._tharsol = tharsol
        
#    @property
#    def combsol(self):
#        combdisp2d = ws.Comb(self,self._item_to_version())
#        self._cache['combsol'] = combdisp2d
#        return combdisp2d
#    @combsol.setter
#    def combsol(self,combsol):
#        """ Input is a wavesol.Comb object """
#        self._combsol = combsol
#        
   

    def fit_lines(self,order=None,*args,**kwargs):
        """
        Performs line fitting for the order(s) provided.
        """
        orders = self.prepare_orders(order)
        if len(orders)==1:
            linelist = lines.fit1d(self,orders[0])
            return linelist
        else:
            linedict = lines.fit(self,orders)
            return linedict

    
    def plot_spectrum(self,*args,**kwargs):
        '''
        Plots the spectrum. 
        
        Args:
        ----
            order:          integer of list or orders to be plotted
            nobackground:   boolean, subtracts the background
            scale:          'pixel', 'combsol' or 'tharsol'
            model:          boolean, fits the lines and shows the fits
            
        Returns:
        --------
            plotter:    Plotter Class object
        '''
        
        
        if self.filetype=='s1d':
            plotter = self.plot_spectrum_s1d(*args,**kwargs)
        else :
            plotter = self.plot_spectrum_e2ds(*args,**kwargs)
        return plotter
    def plot_spectrum_s1d(self,plotter=None,*args,**kwargs):
        """
        Plots the spectrum if file type is s1d.
        """
        # ----------------------      READ ARGUMENTS     ----------------------
        
        scale   = kwargs.pop('scale','pixel')
        ai      = kwargs.pop('axnum', 0)
        legend  = kwargs.pop('legend',False)
        plotter = plotter if plotter is not None else Figure(1,**kwargs)
        figure  = plotter.figure
        axes    = plotter.axes
        
        item    = kwargs.pop('version',None)
        version = self._item_to_version(item)
        if scale=='pixel':
            x1d    = np.arange(self.npix)
            xlabel = 'Pixel'
        else:
            x1d    = self.comb.dispersion(version)
            xlabel = 'Wavelength [A]'
        y1d = self.data
        axes[ai].errorbar(x1d,y1d,yerr=0,label='Data',capsize=3,capthick=0.3,
                ms=10,elinewidth=0.3,color='C0',zorder=100)  
        axes[ai].set_xlabel(xlabel)
        axes[ai].set_ylabel('Counts')
        m = hf.round_to_closest(np.max(y1d),hs.rexp)
        axes[ai].set_yticks(np.linspace(0,m,3))
        if legend:
            axes[ai].legend()
        figure.show()
    def plot_spectrum_e2ds(self,order=None,plotter=None,**kwargs):
        """
        Plots the spectrum if file type is e2ds.
        
        Args:
        -----
            order:          integer or list/array or orders to be plotted, 
                            default None.
            plotter:        harps.Figure object (opt). Default None.
            nobackground:   boolean (opt). Subtracts the background, 
                            default false.
            scale:          str (opt). Allowed values 'pixel', 'combsol' and
                            'tharsol', default 'pixel'.
            model:          boolean (opt). Plots the line fits, default false.
            fittype:        str, list of strings (opt). Allowed values are
                            'lsf' and 'gauss' (default).
            ai:             int (opt). Sets the axes index for plotting, 
                            default 0.
            legend:         bool (opt). Shows the legend if true, default true.
            kind:           str (opt). Sets the plot command. Allowed values 
                            are 'errorbar', 'line', 'points', default 'errorbar'
            show_background bool(opt). Plots the background if true, default
                            false.
        
        """
        # ----------------------      READ ARGUMENTS     ----------------------
        
        nobkg   = kwargs.pop('nobackground',False)
        scale   = kwargs.pop('scale','pixel')
        model   = kwargs.pop('model',False)
        fittype = kwargs.pop('fittype','gauss')
        ai      = kwargs.pop('axnum', 0)
        legend  = kwargs.pop('legend',True)
        kind    = kwargs.pop('kind','errorbar')
        shwbkg  = kwargs.pop('show_background',False)
        plotter = plotter if plotter is not None else Figure(1,**kwargs)
        figure  = plotter.fig
        axes    = plotter.axes
        
        
        orders  = self.prepare_orders(order)
        # ----------------------        READ DATA        ----------------------
        
        if model==True:
            model2d = self['model_{ft}'.format(ft=fittype)]
        item    = kwargs.pop('version',None)
        version = self._item_to_version(item)
        assert scale in ['pixel','combsol','tharsol']
        if scale=='pixel':
            x2d    = np.vstack([np.arange(self.npix) for i in range(self.nbo)])
            xlabel = 'Pixel'
        elif scale=='combsol':
            x2d    = self['wavesol_{}'.format(fittype),version]
            xlabel = r'Wavelength [$\rm{\AA}$]'
        elif scale=='tharsol':
            x2d    = self.tharsol
            xlabel = r'Wavelength [$\rm{\AA}$]'
        for order in orders:
            x      = x2d[order]
            y      = self.data[order]
            if nobkg:
                bkg = self.get_background1d(order)
                y = y-bkg 
            yerr   = self.get_error1d(order)
            if kind=='errorbar':
                axes[ai].errorbar(x,y,yerr=yerr,label='Flux',capsize=3,
                    capthick=0.3,ms=10,elinewidth=0.3,zorder=100,#color='C0',
                    rasterized=True)
            elif kind=='points':
                axes[ai].plot(x,y,label='Flux',ls='',marker='o',
                    ms=10,color='C0',zorder=100,rasterized=True)
                
            else:
                axes[ai].plot(x,y,label='Flux',ls='-',zorder=100,#color='C0',
                    rasterized=True)
            if model==True:   
                model1d = model2d[order]
                axes[ai].plot(x,model1d,c='C1',
                             label='Model {}'.format(fittype),)
            if shwbkg==True:
                bkg1d = self.get_background1d(order)
                axes[ai].plot(x,bkg1d,label='Background',ls='-',color='C3',
                    zorder=100,rasterized=True)
        axes[ai].set_xlabel(xlabel)
        axes[ai].set_ylabel('Counts')
        m = hf.round_to_closest(np.max(y),hs.rexp)
        axes[ai].set_yticks(np.linspace(0,m,3))
        if legend:
            handles,labels = axes[ai].get_legend_handles_labels()
            axes[ai].legend(handles[:2],labels[:2])
        figure.show()
        return plotter
    def plot_distortions(self,order=None,kind='lines',plotter=None,**kwargs):
        '''
        Plots the distortions in the CCD in two varieties:
        kind = 'lines' plots the difference between LFC theoretical wavelengths
        and the value inferred from the ThAr wavelength solution. 
        kind = 'wavesol' plots the difference between the LFC and the ThAr
        wavelength solutions.
        
        Args:
        ----
            order:      integer of list or orders to be plotted
            kind:       'lines' or 'wavesol'
            plotter:    Figure class object from harps.plotter (opt), 
                        default None.
        Returns:
        --------
            plotter:    Figure class object
        '''
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        fittype = kwargs.pop('fittype','gauss')
        ai      = kwargs.pop('axnum', 0)
        marker  = kwargs.get('marker','x')
        plotter = plotter if plotter is not None else Figure(1,**kwargs)
        axes    = plotter.axes
        # ----------------------        PLOT DATA        ----------------------
        
        
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        plotargs = {'ms':2,'marker':marker}
        
        if kind == 'lines':
            plotargs['ls']=''
            
            data  = self['linelist']
            wave  = hf.freq_to_lambda(data['freq'])
            cens  = data['{}'.format(fittype)][:,1]
            coeff = self._tharsol.get_wavecoeff_vacuum()
            
            for i,order in enumerate(orders):
                if len(orders)>5:
                    plotargs['color']=colors[i]
                cut  = np.where(data['order']==order)
                pars = coeff[order]['pars']
                #print(cens[cut])
                thar = np.polyval(pars[::-1],cens[cut])
                #print(order,thar,wave[cut])
                rv   = (wave[cut]-thar)/wave[cut] * c
                axes[ai].plot(cens[cut],rv,**plotargs)
        elif kind == 'wavesol':
            plotargs['ls']='-'
            plotargs['ms']=0
                
            version = kwargs.pop('version',self._item_to_version(None))
            wave = self['wavesol_{}'.format(fittype),version]
            thar = self.tharsol
            plotargs['ls']='--'
            for i,order in enumerate(orders):
                plotargs['color']=colors[i]
                rv = (wave[order]-thar[order])/wave[order] * c
                axes[ai].plot(rv,**plotargs)
            
        [axes[ai].axvline(512*(i+1),lw=0.3,ls='--') for i in range (8)]
        axes[ai].set_ylabel('$\Delta x$=(ThAr - LFC) [m/s]')
        axes[ai].set_xlabel('Pixel')
        return plotter
    def plot_line(self,order,lineid,fittype='gauss',center=True,residuals=False,
                  plotter=None,axnum=None,title=None,figsize=(12,12),show=True,
                  **kwargs):
        ''' Plots the selected line and the models with corresponding residuals
        (optional).'''
        naxes = 1 if residuals is False else 2
        left  = 0.15 if residuals is False else 0.2
        ratios = None if residuals is False else [4,1]
        if plotter is None:
            plotter = Figure(naxes=naxes,title=title,figsize=figsize,
                                      ratios=ratios,sharex=False,
                                      left=left,bottom=0.18,**kwargs)
            
        else:
            pass
        ai = axnum if axnum is not None else 0
        figure, axes = plotter.figure, plotter.axes
        # Load line data
        linelist  = self['linelist']
        line      = linelist[np.where((linelist['order']==order) & \
                                      (linelist['index']==lineid))]
        pixl      = line['pixl'][0]
        pixr      = line['pixr'][0]
        print(line['lsf'])
        pix       = np.arange(pixl,pixr)
        
        flux      = self['flux'][order,pixl:pixr]
        error     = self['error'][order,pixl:pixr]
        
        
        # save residuals for later use in setting limits on y axis if needed
        if residuals:
            resids = []
        # Plot measured line
        axes[ai].errorbar(pix,flux,yerr=error,ls='',color='C0',marker='o',zorder=0)
        axes[ai].bar(pix,flux,width=1,align='center',color='C0',alpha=0.3)
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
        for j,ft in enumerate(np.atleast_1d(fittype)):
            if ft == 'lsf':
                label = 'LSF'
                c   = 'C1'
                m   = 's'
            elif ft == 'gauss':
                label = 'Gauss'
                c   = 'C2'
                m   = '^'
            labels.append(label)
            model     = self['model_{ft}'.format(ft=ft)][order,pixl:pixr]
            axes[ai].plot(pix,model,ls='-',color=c,marker=m,label=ft)
            if residuals:
                rsd        = (flux-model)/error
                resids.append(rsd)
                axes[ai+1].scatter(pix,rsd,color=c,marker=m)
        # Plot centers
            if center:
                
                cen = line[ft][0][1]
                axes[ai].axvline(cen,ls='--',c=c)
        # Makes plot beautiful
        
        axes[ai].set_ylabel('Flux\n[$e^-$]')
        rexp = hs.rexp
        m   = hf.round_to_closest(np.max(flux),rexp)
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
                               axnum=None,fittype='gauss',
                               **kwargs):
        ''' Plots the residuals of the line fits as either a function of 
            position on the CCD or a produces a histogram of values'''
        
        if hist == False:
            figsize = (12,9)
        else: 
            figsize = (9,9)
        if plotter is None:
            plotter=Figure(1,figsize=figsize,bottom=0.12,left=0.15,
                                    **kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        
        figure,axes = plotter.figure, plotter.axes
        
        orders = self.prepare_orders(order)
        plot2d = True if len(orders)>1 else False
        data   = self.data
        model  = self['model_{ft}'.format(ft=fittype)]
        resids = (data - model)[orders]
        if hist == True:
            bins = kwargs.get('bins',30)
            xrange = kwargs.get('range',None)
            log  = kwargs.get('log',False)
            label = kwargs.get('label',fittype)
            alpha = kwargs.get('alpha',1.)
            fitresids1d = np.ravel(resids)
            axes[ai].hist(fitresids1d,bins=bins,range=xrange,log=log,
                label=label,alpha=alpha)
            axes[ai].set_ylabel('Number of lines')
            axes[ai].set_xlabel('Residuals [$e^-$]')
        else:
            if plot2d:
                from matplotlib.colors import Normalize
                sig       = np.std(resids)
                normalize = Normalize(-sig,sig,False)
                
                img = axes[ai].imshow(resids,aspect='auto',norm=normalize,
                        extent=[0,4096,self.nbo,self.sOrder])
                cbar      = plt.colorbar(img)
                cbar.set_label('Residuals [$e^-$]')
                axes[ai].set_ylabel('Order')
                axes[ai].set_xlabel('Pixel')
            else:
                colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
                for i,order in enumerate(orders):
                    axes[ai].scatter(np.arange(self.npix),resids[i],
                        s=1,color=colors[i])
                axes[ai].set_xlabel('Pixel')
                axes[ai].set_ylabel('Residuals [$e^-$]')
        return plotter
    def plot_residuals(self,order=None,fittype='gauss',version=None,
                       plotter=None,**kwargs):
        '''
        Plots the residuals of LFC lines to the wavelength solution. 
        
        Args:
        ----
            order:      integer or list/array or orders to be plotted, default 
                        None
            fittype:    str or list of str (opt). Sets the fit model, default
                        'gauss'
            plotter:    Figure class object from harps.plotter (opt), 
                        default None.
        Returns:
        --------
            plotter:    Figure class object
        '''
        # ----------------------      READ ARGUMENTS     ----------------------
        
        
        version = hf.item_to_version(version)
        phtnois = kwargs.pop('photon_noise',False)
        ai      = kwargs.pop('axnum', 0)
        mean    = kwargs.pop('mean',False)
        plotter = plotter if plotter is not None else Figure(1,**kwargs)
        axes    = plotter.axes
        # ----------------------        READ DATA        ----------------------
        linelist  = self['linelist']
        if order is not None:
            orders    = np.atleast_1d(order)
        else:
            orders = np.unique(linelist['order'])
        centers2d = linelist[fittype][:,1]
        
        noise     = linelist['noise']
        coeffs    = ws.get_wavecoeff_comb(linelist,version,fittype)
        residua2d = ws.residuals(linelist,coeffs,version,fittype)
        print(len(residua2d),len(linelist))
        # ----------------------      PLOT SETTINGS      ----------------------
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker     = kwargs.pop('marker','x')
        markersize = kwargs.pop('markersize',2)
        alpha      = kwargs.pop('alpha',1.)
        color      = kwargs.pop('color',None)
        plotargs = {'s':markersize,'marker':marker,'alpha':alpha}
        # ----------------------       PLOT DATA         ----------------------
        for i,order in enumerate(orders):
            cutcen = np.where(linelist['order']==order)[0]
            cent1d = centers2d[cutcen]
#            cutres = np.where(residua2d['order']==order)[0]
            resi1d = residua2d['residual'][cutcen]
            if len(orders)>5:
                plotargs['color']=color if color is not None else colors[i]
                
            if not phtnois:
                axes[ai].scatter(cent1d,resi1d,**plotargs)
            else:
                pn = noise[cutcen]
                axes[ai].errorbar(cent1d,y=resi1d,yerr=pn,
                                    ls='--',lw=0.3,**plotargs)
            if mean==True:
                meanplotargs={'lw':0.8}
                w  = kwargs.get('window',5)                
                rm = hf.running_mean(resi1d,w)
                if len(orders)>5:
                    meanplotargs['color']=colors[i]
                axes[ai].plot(cent1d,rm,**meanplotargs)
        [axes[ai].axvline(512*(i),lw=0.3,ls='--') for i in range (9)]
        axes[ai]=hf.make_ticks_sparser(axes[ai],'x',9,0,4096)
        axes[ai].set_xlabel('Pixel')
        axes[ai].set_ylabel('Residuals [m/s]')
        axes[ai].set_title("Version PGS = {v:3d}".format(v=version))
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
            order:      integer or list/array of orders to be plotted
            plotter:    Figure class object from harps.plotter (opt), 
                        default None.
            show:       boolean
        Returns:
        --------
            plotter:    Figure class object
        '''
        if kind not in ['residual','gchisq']:
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
                plotter = Figure(1,naxes=N,alignment='grid',**kwargs)
            elif separate == False:
                plotter = Figure(1,naxes=1,**kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        figure, axes = plotter.figure, plotter.axes
        
        # plot residuals or chisq
        if kind == 'residual':
            data = self['residuals']
        elif kind == 'gchisq':
            data = self['linelist']
        bins    = kwargs.get('bins',10)
        alpha   = kwargs.get('alpha',1.0)
        if separate == True:
            for i,order in enumerate(orders):
                cut = np.where(data['order']==order)
                sel = data[kind][cut]
                axes[i].hist(sel,bins=bins,normed=normed,range=histrange,
                             alpha=alpha)
                if kind == 'residual':
                    mean = np.mean(sel)
                    std  = np.std(sel)
                    A    = 1./np.sqrt(2*np.pi*std**2)
                    x    = np.linspace(np.min(sel),np.max(sel),100)
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
            sel = data[kind]
            print(sel)
            axes[0].hist(sel,bins=bins,normed=normed,range=histrange,
                         alpha=alpha)
            if kind == 'residual':
                mean = np.mean(sel)
                std  = np.std(sel)
                A    = 1./np.sqrt(2*np.pi*std**2)
                x    = np.linspace(np.min(sel),np.max(sel),100)
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
        figure.show() 
        return plotter
    
    def plot_shift(self,order=None,p1='lsf',p2='gauss',
                   plotter=None,axnum=None,show=True,**kwargs):
        ''' 
        Plots the shift between the selected estimators of the line centers.
            
        Args:
        -----
            order   : int or list/array of integers, default None.
            p1      : primary estimator, default 'lsf'.
            p2      : secondary estimator, default 'gauss'.
            plotter : Figure class object from harps.plotter (opt), 
                        default None.
        '''
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        ai      = kwargs.pop('axnum', 0)
        plotter = plotter if plotter is not None else Figure(1,**kwargs)
        axes    = plotter.axes
        
        #
        linelist = lines.Linelist(self['linelist'])
        
        def get_center_estimator(linelist1d,p):
            if p == 'lsf':
                cen = linelist1d.values['lsf'][:,1]
                label = 'cen_{lsf}'
            elif p == 'gauss':
                cen = linelist1d.values['gauss'][:,1]
                label = 'cen_{gauss}'
            elif p == 'bary':
                cen = linelist1d.values['bary']
                label = 'b'
            return cen, label
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        
        for i,order in enumerate(orders):
            linelist1d   = linelist[order]
            cen1,label1  = get_center_estimator(linelist1d,p1)
            cen2,label2  = get_center_estimator(linelist1d,p2)
            bary,labelb  = get_center_estimator(linelist1d,'bary')
            delta = cen1 - cen2 
            
            shift = delta * 829
            
            axes[ai].scatter(bary,shift,marker='o',s=2,c=[colors[i]],
                    label="${0} - {1}$".format(label1,label2))
        axes[ai].set_ylabel('[m/s]')
        axes[ai].set_xlabel('Line barycenter [pix]')
        #axes[ai].legend()
        
        return plotter
    
    def plot_wavesolution(self,order=None,calibrator='comb',
                          fittype=['gauss','lsf'],version=None,plotter=None,
                          **kwargs):
        '''
        Plots the wavelength solution of the spectrum for the provided orders.
        '''
        
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        ai      = kwargs.pop('axnum', 0)
        plotter = plotter if plotter is not None else Figure(1,**kwargs)
        axes    = plotter.axes
        # ----------------------        READ DATA        ----------------------
        
        
        fittype = np.atleast_1d(fittype)
        # Check and retrieve the wavelength calibration
        
        linelist = self['linelist']
        
        frequencies = linelist['freq'] 
        wavelengths = hf.freq_to_lambda(frequencies)
        # Manage colors
        #cmap   = plt.get_cmap('viridis')
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker = kwargs.get('marker','x')
        ms     = kwargs.get('markersize',5)
        ls     = {'lsf':'--','gauss':'-'}
        # Plot the line through the points?
        plotline = kwargs.get('plot_line',True)
        # Select line data    
        for ft in fittype:
            if plotline == True:
                
                if calibrator == 'comb':
                    wavesol = self['wavesol_{}'.format(ft),version]
                else:
                    wavesol = self.tharsol
            centers = linelist[ft][:,1]
            # Do plotting
            for i,order in enumerate(orders):
                cut = np.where(linelist['order']==order)
                pix = centers[cut]
                wav = wavelengths[cut]
                axes[ai].plot(pix,wav,color=colors[i],
                    ls='',ms=ms,marker=marker)
                if plotline == True:
                    axes[ai].plot(wavesol[order],color=colors[i],ls=ls[ft],
                        lw=0.8)
        axes[ai].set_xlabel('Pixel')
        axes[ai].set_ylabel('Wavelength [$\AA$]')
        return plotter
    def prepare_orders(self,order):
        '''
        Returns an array or a list containing the input orders.
        '''
        nbo = self.meta['nbo']
        if order is None:
            orders = np.arange(self.sOrder,nbo,1)
        else:
            orders = hf.to_list(order)
        return orders
    @property
    def optical_orders(self):
        optord = np.arange(88+self.nbo,88,-1)
        # fibre A doesn't contain order 115
        if self.meta['fibre'] == 'A':
            shift = 1
        # fibre B doesn't contain orders 115 and 116
        elif self.meta['fibre'] == 'B':
            shift = 2
        cut=np.where(optord>114)
        optord[cut]=optord[cut]+shift
        
        return optord
    def _slice(self,order):
        nbo = self.meta['nbo']
        if isinstance(order,int):
            start = order
            stop = order+1
            step = 1
        elif isinstance(order,tuple):
            range_sent = True
            numitems = np.shape(order)[0]
            if numitems==3:
                start, stop, step = order
            elif numitems==2:
                start, stop = order
                step = 1
            elif numitems==1:
                start = order
                stop  = order+1
                step  = 1
        else:
            start = self.sOrder
            stop  = nbo
            step  = 1
        return slice(start,stop,step)
    
