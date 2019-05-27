#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:28:45 2019

@author: dmilakov
"""

from   harps.core     import FITS, FITSHDR, np, os
from   harps.classes  import Spectrum

import harps.settings  as hs
import harps.io        as io
import harps.functions as hf

from   astropy.coordinates import SkyCoord, EarthLocation
from   astropy.time        import Time
from   astropy             import units as u

version      = hs.__version__

class Object(object):
    def __init__(self,objSpec,overwrite=False,*args,**kwargs):
        self._path_to_objSpec = objSpec
        self.basename = os.path.basename(self._path_to_objSpec)
        self.dirname  = os.path.dirname(self._path_to_objSpec)
        
        
        self.header   = io.read_e2ds_header(self._path_to_objSpec)
        # Convert data into electrons
        self._conad   = self.header['HIERARCH ESO DRS CCD CONAD']
        self.flux     = io.read_e2ds_data(self._path_to_objSpec)*self._conad
        
        self._cache = {}
        self._blazecorrected = False
        
        self.correct_blaze()
        
        self._barycorrected = False
        
    def _set_calibration_file(self,LFCspec):
        self._path_to_LFCSpec = LFCspec
        self._LFCSpec = Spectrum(LFCspec)
        
    @property
    def calibration_file(self):
        return self._path_to_LFCSpec
    @calibration_file.setter
    def calibration_file(self,filepath):
        self._path_to_LFCSpec = filepath
        return
    @property
    def wave(self):
        try:
            wavesol = self._cache['wave']
        except:
            lfcspec = Spectrum(self.calibration_file)
            self.calibration_spec = lfcspec
            
            wavesol0 = lfcspec['wavesol_lsf',501]
            # apply barycentric correction
            berv     = self.berv
            wavesol  = wavesol0/(1+berv/299792458.)
            self._cache['wave'] = wavesol
            self._barycorrected = True
        return wavesol
        
    def return_header(self):
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
            elif name=='Object':
                value = self.header['OBJECT']
            elif name=='Exposure':
                value = self.basename
            elif name=='version':
                value = version
            elif name=='Calib':
                value = os.path.basename(self.calibration_file)
            elif name=='Blaze':
                value = self.header['HIERARCH ESO DRS BLAZE FILE']
            elif name=='barycorr':
                value = self._barycorrected
            elif name=='blazcorr':
                value = self._blazecorrected
            elif name=='midobs':
                value = str(self.midobs)
            elif name=='berv':
                value = self.berv
            return value
        def make_dict(name,value,comment=''):
            return dict(name=name,value=value,comment=comment)
            
        names = ['Author','Object','Exposure','Blaze','Calib','version',
                 'barycorr','blazcorr','midobs','berv']            
        
        comments_dict={'Simple':'Conforms to FITS standard',
                  'Bitpix':'Bits per data value',
                  'Naxis':'Number of data axes',
                  'Extend':'FITS dataset may contain extensions',
                  'Author':'',
                  'Object':'Target name',
                  'Exposure':'Original file',
                  'version':'Code version used',
                  'Blaze':'Blaze file',
                  'Calib':'Calibration file',
                  'barycorr':'Barycentric correction applied',
                  'blazcorr':'Blaze corrected',
                  'midobs':'Flux weighted midpoint of observation',
                  'berv':'Barycentric Earth Radial Velocity at midobs'}
        values_dict = {name:return_value(name) for name in names}
        
        values   = [values_dict[name] for name in names]
        comments = [comments_dict[name] for name in names]
        header   = [make_dict(n,v,c) for n,v,c in zip(names,values,comments)]
        return FITSHDR(header)
    
    @property
    def blaze(self):
        '''
        Returns a 2d array containing the blaze function. 
        
        Finds the relevant blaze file name in the header, after which the file
        is read in. Assumes a specific directory structure. 
        '''
        try: 
            blaze2d = self._cache['blaze2d']
        except:
            blaze_filename = self.header['HIERARCH ESO DRS BLAZE FILE']
            blaze_datetime = hf.basename_to_datetime(blaze_filename)
            datedir        = np.datetime_as_string(blaze_datetime)[0:10]
            topdir         = os.path.split(self.dirname)[0]
            #print(topdir,datedir,blaze_filename)
            path_to_blaze  = os.path.join(*[topdir,datedir,blaze_filename])
            
            with FITS(path_to_blaze,mode='r') as hdu:
                blaze2d = hdu[0].read()
            self._cache['blaze2d'] = blaze2d
        return blaze2d
    def correct_blaze(self):
        '''
        Performs blaze correction on the file. Returns None.
        '''
        if not self._blazecorrected:
            blaze      = self.blaze
            self.flux  = self.flux/blaze
            self._blazecorrected = True
        else:
            pass
        return None
    @property 
    def error(self):
        try:
            noise = self._cache['noise']
        except:
            signal = self.flux
            npix   = 16 # 16, private communciation with G. Lo Curto
            ron    = self.header['HIERARCH ESO DRS CCD SIGDET']
            expt   = self.header['EXPTIME']
            dc     = 0.8 # 0.5 - 1 e-/pix/h Private communication with G. Lo Curto
            noise  = np.sqrt(np.abs(signal) + npix*ron**2 * npix*dc*expt/3600)
            self._cache['noise'] = noise
        return noise
    @property
    def midobs(self):
        '''
        Returns the midpoint of observations in the form of a numpy.datetime64 
        scalar. Uses the start time of the observations and the exposure 
        length for the calculation. Both values are read from the header.
        '''
        try:
            midobs = self._cache['midobs']
        except:
            dateobs = self.header['DATE-OBS']
            exptime = self.header['EXPTIME']
            meanobs = self.header['HIERARCH ESO INS DET1 TMMEAN']
            midobs  = np.datetime64(dateobs) + \
                      meanobs * np.timedelta64(np.int(exptime),'s')
            self._cache['midobs'] = midobs
        return midobs
    @property
    def geoloc(self):
        '''
        Returns an astropy EarthLocation object with observatory coordinates.
        Observatory coordinates are read from the header.
        '''
        try:
            geoloc = self._cache['geoloc']
        except:
            lat  = self.header['HIERARCH ESO TEL GEOLAT']
            lon  = self.header['HIERARCH ESO TEL GEOLON']
            elev = self.header['HIERARCH ESO TEL GEOELEV']
            geoloc = EarthLocation.from_geodetic(lat=lat*u.deg,
                                                 lon=lon*u.deg, 
                                                 height=elev*u.m)
            self._cache['geoloc'] = geoloc
        return geoloc
    @property 
    def skycoord(self):
        '''
        Returns an astropy SkyCoord object with the target coordinates.
        Target coordinates are read from the header.
        '''
        try:
            skycoord = self._cache['skycoord']
        except:
            ra       = self.header['RA']
            dec      = self.header['DEC']
            skycoord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            self._cache['skycoord'] = skycoord
        return skycoord
    @property
    def borv(self):
        ''' 
        Returns barycentric observatory radial velocity at the flux-weighted 
        midpoint of observations in units m/s. 
            
        Reads in the target and observatory locations from the header, 
        then uses astropy `radial_velocity_correction' to calculate the
        barycentric correction. 
        '''
        try:
            berv = self._cache['borv']
        except:
            skycoord = self.skycoord
            geoloc   = self.geoloc
            obstime  = Time(self.midobs)
            berv_    = skycoord.radial_velocity_correction('barycentric',
                                                           obstime=obstime,
                                                           location=geoloc)
            berv     = np.float64(berv_)
            self._cache['borv'] = berv
        return berv
            
    @property
    def berv(self):
        ''' 
        Returns barycentric observatory radial velocity at the flux-weighted 
        midpoint of observations in units m/s. 
            
        Reads the value from the header. 
        '''
        try:
            berv = self._cache['berv']
        except:
            berv = self.header['HIERARCH ESO DRS BERV']*1000
            self._cache['berv'] = berv
        return berv
    def save(self,dirname=None,overwrite=False):
        dirname = dirname if dirname is not None else io.get_dirpath('objspec')
        
        extensions = ['wave','error','flux']
        for ext in extensions:
            img      = getattr(self,ext)
            header   = self.return_header()
            basename = str.replace(self.basename,'e2ds',ext)
            filename = os.path.join(dirname,basename)
            with FITS(filename,'rw',clobber=overwrite) as hdu:
                hdu.write(img)
                hdu[-1].write_keys(header)

                print(ext,img.shape,img.dtype)
        return
        