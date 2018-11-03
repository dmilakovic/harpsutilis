#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:45:04 2018

@author: dmilakov
"""
from harps.core import sys
from harps.core import np, pd, xr
from harps.core import os, tqdm, gc, time
from harps.core import leastsq, curve_fit, odr, interpolate
from harps.core import fits, FITS, FITSHDR
from harps.core import plt
from harps.core import warnings

from multiprocessing import Pool

from harps import functions as hf
from harps import settings as hs
from harps import io
from harps import wavesol
from harps import background
from harps import lines

from harps.constants import c

from harps.plotter import SpectrumPlotter

version      = hs.__version__
harps_home   = hs.harps_home
harps_data   = hs.harps_data
harps_dtprod = hs.harps_dtprod
harps_plots  = hs.harps_plots
harps_prod   = hs.harps_prod

sOrder       = hs.sOrder
eOrder       = hs.eOrder
nOrder       = eOrder-sOrder

def extract_version(ver):
    if isinstance(ver,int) and ver>99 and ver<1000:
        split  = [int((ver/10**x)%10) for x in range(3)][::-1]
        polyord, gaps, segment = split
#    return dict(polyord=polyord,gaps=gaps,segment=segment)
        return polyord,gaps,segment

class Spectrum(object):
    ''' Spectrum object contains functions and methods to read data from a 
        FITS file processed by the HARPS pipeline
    '''
    def __init__(self,filepath=None,LFC='HARPS',model='SingleGaussian',
                 **kwargs):
        '''
        Initialise the spectrum object.
        '''
        self.filepath = filepath
        self.name     = "HARPS Spectrum"
        self.lfcname  = LFC
        self.data     = io.read_e2ds_data(filepath)
        self.hdrmeta  = io.read_e2ds_meta(filepath)
        self.header   = io.read_e2ds_header(filepath)
        self.lfckeys  = io.read_LFC_keywords(filepath,LFC)
        self.meta     = self.hdrmeta
        
        
        self.npix     = self.meta['npix']
        self.nbo      = self.meta['nbo']
        self.d        = self.meta['d']
        self.sOrder   = hs.sOrder
        self.eOrder   = self.meta['nbo']
        
        self.model    = model
        
        version       = self._item_to_version()
        self.polyord  = version[0]
        self.gaps     = version[1]
        self.segment  = version[2]
        
        
        self.segsize  = self.npix//16 #pixel
        varmeta       = dict(sOrder=self.sOrder,polyord=self.polyord,
                             gaps=self.gaps,segment=self.segment,
                             segsize=self.segsize,model=self.model)
        self.meta.update(varmeta)
        
        self.datetime = np.datetime64(self.meta['obsdate'])
        
        self.outfits  = io.get_fits_path(filepath)
        clobber = kwargs.pop('clobber',False)
        self.hdu      = FITS(self.outfits,'rw',clobber=clobber)
        self.write_primaryheader(self.hdu)
        try:
            self._tharsol = wavesol.thar(self,vacuum=True)
        except:
            self._tharsol = None
        return
    def __getitem__(self,item):
        '''
        Tries reading data from file, otherwise runs appropriate function. 
        
        Args:
        ----
            datatype (str) :  ['linelist','coeff','model_gauss',
                               'wavesol_comb','wavesol_thar']
            save     (bool):  saves to the FITS file if true
        
        Returns:
        -------
            data : np.array
            
        '''
        ext, ver, versent = self._extract_item(item)
        mess = "Extension {ext:>10}, version {ver:<5}:".format(ext=ext,ver=ver)
        hdu  = self.hdu
        try:
            data   = hdu[ext,ver].read()
            mess   += "read from file."
        except:
            data   = self.calculate(ext,ver)
            header = self.return_header(ext)
            hdu.write(data=data,header=header,extname=ext,extver=ver)
            mess   += "calculated."
        print(mess)
        return data

    def __str__(self):
        meta     = self.meta
        dirname  = os.path.dirname(self.filepath)
        basename = os.path.basename(self.filepath)
        mess =  "{0:^80s} \n".format("SPECTRUM")+\
                "{0:-^80s} \n".format("")+\
                "{0:<20s}:{1:>60s}\n".format("Directory",dirname)+\
                "{0:<20s}:{1:>60s}\n".format("File",basename)+\
                "{0:<20s}:{1:>60s}\n".format("LFC",self.lfcname)+\
                "{0:<20s}:{1:>60s}\n".format("Obsdate",meta['obsdate'])+\
                "{0:<20s}:{1:>60s}\n".format("Model",meta['model'])
        return mess
    @staticmethod
    def _version_to_dict(ver):
        if isinstance(ver,int) and ver>99 and ver<1000:
            split  = [int((ver/10**x)%10) for x in range(3)][::-1]
            polyord, gaps, segment = split
        return dict(polyord=polyord,gaps=gaps,segment=segment)
    def _item_to_version(self,item=None):
        polyord = 3#self.polyord
        gaps    = 0#self.gaps
        segment = 0#self.segment
     
        if isinstance(item,dict):
            polyord = item.pop('polyord',polyord)
            gaps    = item.pop('use_gaps',gaps)
            segment = item.pop('use_ptch',segment)
            ver     = int("{2:1d}{1:1d}{0:1d}".format(segment,gaps,polyord))
        elif isinstance(item,int) and item>99 and item<1000:
            split   = [int((item/10**x)%10) for x in range(3)][::-1]
            polyord = split[0]
            gaps    = split[1]
            segment = split[2]
        elif isinstance(item,tuple):
            polyord = item[0]
            gaps    = item[1]
            segment = item[2]
        ver     = int("{2:1d}{1:1d}{0:1d}".format(segment,gaps,polyord))
        
        return ver
    
    def calculate(self,datatype,version=None,*args,**kwargs):
        
        assert datatype in io.allowed_hdutypes
        version = self._item_to_version(version)
        functions = {'linelist':lines.detect,
                     'coeff':wavesol._get_wavecoeff_comb,
                     'wavesol_comb':wavesol.comb,
                     'model_gauss':lines.model_gauss,
                     'residuals':wavesol.residuals}
        if datatype in ['coeff','wavesol_comb','residuals']:
            data = functions[datatype](self,version,*args,**kwargs)
        elif datatype in ['linelist','model_gauss']:
            data = functions[datatype](self,*args,**kwargs)
        return data
    def _extract_item(self,item):
        """
        utility function to extract an "item", meaning
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
        
        ver = self._item_to_version(ver)
        return ext,ver,ver_sent
    
    def write_primaryheader(self,hdu):
        ''' Writes the spectrum metadata to the HDU header'''
        header = self.return_header('primary')
        hdu[0].write_keys(header)
        return 
    def return_header(self,hdutype):
        meta = self.meta
        LFC  = self.lfckeys
        # ------- Reads metadata and LFC keywords
        
        def make_dict(name,value,comment=''):
            return dict(name=name,value=value,comment=comment)
        
        if hdutype == 'primary':
            names = ['Simple','Bitpix','Naxis','Extend','Author',
                     'npix','mjd','date-obs','fibshape']
        elif hdutype == 'linelist':
            names = ['version']
        elif hdutype == 'wavesol_comb':
            names = ['lfc','anchor','reprate','gaps','segment','polyord']
        elif hdutype == 'coeff':
            names = ['gaps','segment','polyord']
        elif hdutype == 'model_gauss':
            names = ['model']
        elif hdutype == 'residuals':
            names = ['lfc','anchor','reprate','gaps','segment','polyord']
        else:
            raise UserWarning("HDU type not recognised")

        values_dict={'Simple':True,
                'Bitpix':32,
                'Naxis':0,
                'Extend':True,
                'Author':'Dinko Milakovic',
                'version':version,
                'npix':meta['npix'],
                'mjd':meta['mjd'],
                'date-obs':meta['obsdate'],
                'fibshape':meta['fibshape'],
                'lfc':LFC['name'],
                'reprate':LFC['comb_reprate'],
                'anchor':LFC['comb_anchor'],
                'gaps':meta['gaps'],
                'segment':meta['segment'],
                'polyord':meta['polyord'],
                'model':meta['model']}
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
                  'model':'EmissionLine class used to fit lines'}
        
        
        values   = [values_dict[name] for name in names]
        comments = [comments_dict[name] for name in names]
        header   = [make_dict(n,v,c) for n,v,c in zip(names,values,comments)]
        return FITSHDR(header)
        
        
    


#    def check_and_load_psf(self,filepath=None):
#        exists_psf = hasattr(self,'psf')
#        
#        if not exists_psf:
#            self.load_psf(filepath)
#        else:
#            pass
#        segments        = np.unique(self.psf.coords['seg'].values)
#        N_seg           = len(segments)
#        # segment limits
#        sl              = np.linspace(0,4096,N_seg+1)
#        # segment centers
#        sc              = (sl[1:]+sl[:-1])/2
#        sc[0] = 0
#        sc[-1] = 4096
#        
#        self.segments        = segments
#        self.nsegments       = N_seg
#        self.segsize         = self.npix//N_seg
#        self.segment_centers = sc
#        return
#    


    
    
    def get_error(self,*args):
        return self.get_error2d(*args)
    
    def get_error2d(self,*args):
        try:
            error2d = getattr(self,'error2d')
        except:
            data2d  = np.abs(self.data)
            bkg2d   = background.get2d(self,*args)
            error2d = np.sqrt(np.abs(data2d) + np.abs(bkg2d))
        return error2d
    
    def get_error1d(self,order,*args):
        data1d  = np.abs(self.data[order])
        bkg1d   = np.abs(background.get1d(self,order,*args))
        error1d = np.sqrt(data1d + bkg1d)
        return error1d
    
    def get_background(self,*args):
        return background.get2d(self,*args)
    
    def get_background1d(self,order,*args):
        return background.get1d(self,order,*args)
    
    def get_tharsol1d(self,order,*args):
        tharsol = wavesol.thar(self,*args)
        return tharsol[order]
    def get_tharsol(self,*args):
        return wavesol.thar(self,*args)
    @property
    def tharsol(self):
        return self._tharsol
    @tharsol.setter
    def tharsol(self,spec):
        self._tharsol = wavesol.thar(spec)
        
        
    def get_wavesol(self,calibrator,*args,**kwargs):
        wavesol_cal = "wavesol_{cal}".format(cal=calibrator)
        if hasattr(self,wavesol_cal):
            ws = getattr(self,wavesol_cal)
        else:
            
            ws = wavesol.get(self,calibrator,*args,**kwargs)
            setattr(self,wavesol_cal,ws)
        return ws
    def fit_lines(self,order=None,*args,**kwargs):
        orders = self.prepare_orders(order)
        if len(orders)==1:
            linelist = lines.fit1d(self,orders[0])
            return linelist
        else:
            linedict = lines.fit(self,orders)
            return linedict

#    
#    
#    def calc_lambda(self,ft='epsf',orders=None):
#        ''' Returns wavelength and wavelength error for the lines using 
#            polynomial coefficients in wavecoef_LFC.
#            
#            Adapted from HARPS mai_compute_drift.py'''
#        if orders is not None:
#            orders = orders
#        else:
#            orders = np.arange(self.sOrder,self.nbo,1)
#        lines = self.check_and_return_lines()
#        ws    = self.check_and_get_wavesol()
#        wc    = self.wavecoef_LFC
#        
#        x     = lines['pars'].sel(par='cen',od=orders,ft=ft).values
#        x_err = lines['pars'].sel(par='cen_err',od=orders,ft=ft).values
#        c     = wc.sel(patch=0,od=orders,ft=ft).values
#        # wavelength of lines
#        wave  = np.sum([c[:,i]*(x.T**i) for i in range(c.shape[1])],axis=0).T
#        # wavelength errors
#        dwave = np.sum([(i+1)*c[:,i+1]*(x.T**(i+1)) \
#                        for i in range(c.shape[1]-1)],axis=0).T * x_err
#        return wave,dwave


#    def fit_single_line(self,line,psf=None):
#        def residuals(x0,pixels,counts,weights,background,splr):
#            ''' Model parameters are estimated shift of the line center from 
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
#            # sigma_tot^2 = sigma_counts^2 + sigma_background^2
#            # sigma_counts = sqrt(counts)     sigma_background = sqrt(background)
#            error = np.sqrt(counts + background)
#            resid = np.sqrt(weights) * ((counts-background) - model) / error
##            resid = counts/np.sum(counts) * ((counts-background) - model) / error
#            #resid = line_w * (counts- model)
#            return resid
#        def get_local_psf(pix,order,seg,mixing=True):
#            ''' Returns local ePSF at a given pixel of the echelle order
#            '''
#            segments        = np.unique(psf.coords['seg'].values)
#            N_seg           = len(segments)
#            # segment limits
#            sl              = np.linspace(0,4096,N_seg+1)
#            # segment centers
#            sc              = (sl[1:]+sl[:-1])/2
#            sc[0] = 0
#            sc[-1] = 4096
#           
#            def return_closest_segments(pix):
#                sg_right  = int(np.digitize(pix,sc))
#                sg_left   = sg_right-1
#                return sg_left,sg_right
#            
#            sgl, sgr = return_closest_segments(pix)
#            f1 = (sc[sgr]-pix)/(sc[sgr]-sc[sgl])
#            f2 = (pix-sc[sgl])/(sc[sgr]-sc[sgl])
#            
#            #epsf_x  = psf.sel(ax='x',od=order,seg=seg).dropna('pix')+pix
#            
#            epsf_1 = psf.sel(ax='y',od=order,seg=sgl).dropna('pix')
#            epsf_2 = psf.sel(ax='y',od=order,seg=sgr).dropna('pix')
#            
#            if mixing == True:
#                epsf_y = f1*epsf_1 + f2*epsf_2 
#            else:
#                epsf_y = epsf_1
#            
#            xc     = epsf_y.coords['pix']
#            epsf_x  = psf.sel(ax='x',od=order,seg=seg,pix=xc)+pix
#            #qprint(epsf_x,epsf_y)
#            return epsf_x, epsf_y
#        # MAIN PART 
#        
#        if psf is None:
#            self.check_and_load_psf()
#            psf=self.psf
#        else:
#            pass
#        # select single line
#        #lid       = line_id
#        line      = line.dropna('pid','all')
#        pid       = line.coords['pid']
#        lid       = int(line.coords['id'])
#    
#        line_x    = line['line'].sel(ax='pix')
#        line_y    = line['line'].sel(ax='flx')
#        line_w    = line['line'].sel(ax='wgt')
#        #print("Read the data for line {}".format(lid))
#        # fitting fails if some weights are NaN. To avoid this:
#        weightIsNaN = np.any(np.isnan(line_w))
#        if weightIsNaN:
#            whereNaN  = np.isnan(line_w)
#            line_w[whereNaN] = 0e0
#            #print('Corrected weights')
#        line_bkg  = line['line'].sel(ax='bkg')
#        line_bary = line['attr'].sel(att='bary')
#        cen_pix   = line_x[np.argmax(line_y)]
#        #freq      = line['attr'].sel(att='freq')
#        #print('Attributes ok')
#        #lbd       = line['attr'].sel(att='lbd')
#        # get local PSF and the spline representation of it
#        order        = int(line.coords['od'])
#        loc_seg      = line['attr'].sel(att='seg')
#        psf_x, psf_y = self.get_local_psf(line_bary,order=order,seg=loc_seg)
#        
#        psf_rep  = interpolate.splrep(psf_x,psf_y)
#        #print('Local PSF interpolated')
#        # fit the line for flux and position
#        #arr    = hf.return_empty_dataset(order,pixPerLine)
#        try: pixPerLine = self.pixPerLine
#        except: 
#            self.__read_LFC_keywords__()
#            pixPerLine = self.pixPerLine
#        par_arr     = hf.return_empty_dataarray('pars',order,pixPerLine)
#        mod_arr     = hf.return_empty_dataarray('model',order,pixPerLine)
#        p0 = (5e-1,np.percentile(line_y,90))
#
#        
#        # GAUSSIAN ESTIMATE
#        g0 = (np.nanpercentile(line_y,90),float(line_bary),1.3)
#        gausp,gauscov=curve_fit(hf.gauss3p,p0=g0,
#                            xdata=line_x,ydata=line_y)
#        Amp, mu, sigma = gausp
#        p0 = (0.01,Amp)
#        popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
#                                args=(line_x,line_y,line_w,line_bkg,psf_rep),
#                                full_output=True,
#                                ftol=1e-5)
#        
#        if ier not in [1, 2, 3, 4]:
#            print("Optimal parameters not found: " + errmsg)
#            popt = np.full_like(p0,np.nan)
#            pcov = None
#            success = False
#        else:
#            success = True
#       
#        if success:
#            
#            sft, flux = popt
#            line_model = flux * interpolate.splev(line_x+sft,psf_rep) + line_bkg
#            cost   = np.sum(infodict['fvec']**2)
#            dof    = (len(line_x) - len(popt))
#            rchisq = cost/dof
#            if pcov is not None:
#                pcov = pcov*rchisq
#            else:
#                pcov = np.array([[np.inf,0],[0,np.inf]])
#            cen              = cen_pix+sft
#            cen              = line_bary - sft
#            cen_err, flx_err = [np.sqrt(pcov[i][i]) for i in range(2)]
#  
#            #pars = np.array([cen,sft,cen_err,flux,flx_err,rchisq,np.nan,np.nan])
#            pars = np.array([cen,cen_err,flx,flx_err,sigma,sigma_err,rchi2])
#        else:
#            pars = np.full(len(hf.fitPars),np.nan)
#       
#        par_arr.loc[dict(od=order,id=lid,ft='epsf')] = pars
#        mod_arr.loc[dict(od=order,id=lid,
#                         pid=line_model.coords['pid'],ft='epsf')] = line_model
#
#        return par_arr,mod_arr
#    def fit_lines(self,order=None,fittype='epsf',nobackground=True,model=None,
#                  remove_poor_fits=False,verbose=0,njobs=hs.nproc):
#        ''' Calls one of the specialised line fitting routines.
#        '''
#        # Was the fitting already performed?
#        if self.lineFittingPerformed[fittype] == True:
#            return self.HDU_get('linelist')
#        else:
#            pass
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
#
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


    
    def is_bad_order(self,order):
        if order in self.bad_orders: 
            return True
        else:
            return False
    

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

    def plot_spectrum(self,order=None,plotter=None,**kwargs):
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
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        nobkg   = kwargs.pop('nobackground',False)
        scale   = kwargs.pop('scale','pixel')
        model   = kwargs.pop('model',False)
        fittype = kwargs.pop('fittype','gauss')
        ai      = kwargs.pop('axnum', 0)
        legend  = kwargs.pop('legend',False)
        plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
        figure  = plotter.figure
        axes    = plotter.axes
        # ----------------------        READ DATA        ----------------------
        
        if model==True:
            model2d = self['model_{ft}'.format(ft=fittype)]
        if scale=='pixel':
            x2d    = np.vstack([np.arange(self.npix) for i in range(self.nbo)])
            xlabel = 'Pixel'
        else:
            x2d    = self.get_wavesol('thar')
            xlabel = 'Wavelength [A]'
        for order in orders:
            x      = x2d[order]
            y      = self.data[order]
            if nobkg:
                bkg = self.get_background1d(order)
                y = y-bkg 
            yerr   = self.get_error1d(order)
            
            axes[ai].errorbar(x,y,yerr=yerr,label='Data',capsize=3,capthick=0.3,
                ms=10,elinewidth=0.3,color='C0',zorder=100)
            if model==True:   
                model1d = model2d[order]
                axes[ai].plot(x,model1d,label='Model',c='C1')
               
        axes[ai].set_xlabel(xlabel)
        axes[ai].set_ylabel('Flux [$e^-$]')
        m = hf.round_to_closest(np.max(y),hs.rexp)
        axes[ai].set_yticks(np.linspace(0,m,3))
        if legend:
            axes[ai].legend()
        figure.show()
        return plotter
    def plot_distortions(self,order=None,kind='lines',plotter=None,**kwargs):
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
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        fittype = kwargs.pop('fittype','gauss')
        ai      = kwargs.pop('axnum', 0)
        marker  = kwargs.get('marker','x')
        plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
        axes    = plotter.axes
        # ----------------------        PLOT DATA        ----------------------
        
        
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        plotargs = {'ms':2,'marker':marker}
        
        if kind == 'lines':
            plotargs['ls']=''
            
            data  = self['linelist']
            wave  = hf.freq_to_lambda(data['freq'])
            cens  = data['{}'.format(fittype)][:,1]
            coeff = wavesol._get_wavecoeff_vacuum(self)
            
            for i,order in enumerate(orders):
                if len(orders)>5:
                    plotargs['color']=colors[i]
                cut  = np.where(data['order']==order)
                thar = np.polyval(coeff[order][::-1],cens[cut])
                #print(order,thar,wave[cut])
                rv   = (wave[cut]-thar)/wave[cut] * c
                axes[ai].plot(cens[cut],rv,**plotargs)
        elif kind == 'wavesol':
            plotargs['ls']='-'
            plotargs['ms']=0
                
            version = kwargs.pop('version',self._item_to_version(None))
            wave = self['wavesol_comb',version]
            thar = wavesol.thar(self)
            plotargs['ls']='--'
            for i,order in enumerate(orders):
                plotargs['color']=colors[i]
                rv = (wave[order]-thar[order])/wave[order] * c
                axes[ai].plot(rv,**plotargs)
            
        [axes[ai].axvline(512*(i+1),lw=0.3,ls='--') for i in range (8)]
        axes[ai].set_ylabel('$\Delta x$=(ThAr - LFC) [m/s]')
        axes[ai].set_xlabel('Pixel')
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
                               axnum=None,plot2d=True,fittype='gauss',
                               **kwargs):
        ''' Plots the residuals of the line fits as either a function of 
            position on the CCD or a produces a histogram of values'''
        
        if hist == False:
            figsize = (12,9)
        else: 
            figsize = (9,9)
        if plotter is None:
            plotter=SpectrumPlotter(figsize=figsize,bottom=0.12,left=0.15,
                                    **kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        
        figure,axes = plotter.figure, plotter.axes
        
        orders = self.prepare_orders(order)
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
    def plot_residuals(self,order=None,calibrator='comb',fittype='gauss',
                       version=None,plotter=None,**kwargs):
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
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        
        version = self._item_to_version(version)
        phtnois = kwargs.pop('photon_noise',False)
        ai      = kwargs.pop('axnum', 0)
        mean    = kwargs.pop('mean',False)
        plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
        axes    = plotter.axes
        # ----------------------        READ DATA        ----------------------
        linelist = self['linelist']
 
        centers2d = linelist[fittype][:,1]
        
        noise     = linelist['noise']
        residua2d = self['residuals',version]
        
        # ----------------------      PLOT SETTINGS      ----------------------
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker     = kwargs.pop('marker','x')
        markersize = kwargs.pop('markersize',2)
        alpha      = kwargs.pop('alpha',1.)
        color      = kwargs.pop('color',None)
        plotargs = {'s':markersize,'marker':marker,'alpha':alpha}
        # ----------------------       PLOT DATA         ----------------------
        for i,order in enumerate(orders):
            cutcen = np.where(linelist['order']==order)
            cent1d = centers2d[cutcen]
            cutres = np.where(residua2d['order']==order)
            resi1d = residua2d['residual'][cutres]
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
                plotter = SpectrumPlotter(naxes=N,alignment='grid',**kwargs)
            elif separate == False:
                plotter = SpectrumPlotter(naxes=1,**kwargs)
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
                
        linelist = self['linelist']
        
        def get_center_estimator(p):
            if p == 'lsf':
                cen = linelist['lsf'][:,1]
                label = 'cen_{lsf}'
            elif p == 'gauss':
                cen = linelist['gauss'][:,1]
                label = 'cen_{gauss}'
            elif p == 'bary':
                cen = linelist['bary']
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
        
        figure.show()
        return plotter
    
    def plot_wavesolution(self,calibrator='comb',order=None,plotter=None,
                          **kwargs):
        '''
        Plots the wavelength solution of the spectrum for the provided orders.
        '''
        
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        fittype = kwargs.pop('fittype','gauss')
        ai      = kwargs.pop('axnum', 0)
        plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
        axes    = plotter.axes
        # ----------------------        READ DATA        ----------------------
        
        
        fittype = hf.to_list(fittype)
        # Check and retrieve the wavelength calibration
        wavesol = self['wavesol_comb']
        linelist = self['linelist']
        
        frequencies = linelist['freq'] 
        wavelengths = hf.freq_to_lambda(frequencies)
        # Manage colors
        #cmap   = plt.get_cmap('viridis')
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker = kwargs.get('marker','x')
        ms     = kwargs.get('markersize',5)
        ls     = {'epsf':'--','gauss':'-'}
        # Plot the line through the points?
        plotline = kwargs.get('plot_line',True)
        # Select line data    
        for ft in fittype:
            centers  = linelist[ft][:,1]
            # Do plotting
            for i,order in enumerate(orders):
                cut = np.where(linelist['order']==order)
                pix = centers[cut]
                wav = wavelengths[cut]
                axes[ai].scatter(pix,wav,s=ms,color=colors[i],marker=marker)
                if plotline == True:
                    axes[ai].plot(wavesol[order],color=colors[i],ls=ls[ft],lw=0.5)
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
