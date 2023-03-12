#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:18:41 2018

@author: dmilakov

If 'import harps.process' fails with the message
QXcbConnection: Could not connect to display localhost:13.0

try one of the two:
    
import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pylab
import matplotlib; matplotlib.use('nbAgg'); import matplotlib.pylab

"""

# from   core import np, mp, json, os, gc, glob, time, plt
import numpy as np
import json, os, glob, gc, time
import multiprocessing as mp
import matplotlib.pyplot as plt
import ray
from   harps.settings import __version__ as hs_version
import logging
from   numpy.lib.recfunctions import append_fields

import harps.spectrum as hspec
from   harps.wavesol import ThAr
from   harps.decorators import memoize

import harps.inout as io
import harps.functions as hf
import harps.settings as hs
import harps.velshift as vs

from fitsio import FITS, FITSHDR

#logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
#                    datefmt = '%Y-%m-%d %H:%M:%S')
__version__      = hs.__version__

class Process(object):
    def __init__(self,setting_json):
        self._settings = hs.Settings(setting_json)
        self.init_logger()
        self.logger = logging.getLogger(__name__)
        # -----------------   M A I N   -----------------
        self.logger.info("INITIALIZING NEW PROCESS")
        self.logger.info("Using {}".format(self.settings['selfpath']))
        
        
        self.logger.info("Saving input file path to {}.json".format(self.json_file))
        self._settings.update(dict(outlist=self.output_file))
        self._settings.write()
        self._filelist = None
        
        try:
            versions  = np.atleast_1d(self.settings['version'])
        except:
            versdic = dict(polyord=self.settings['polyord'],
                           gaps=self.settings['gaps'],
                           segment=self.settings['segment'])
            versions = np.atleast_1d(hf.item_to_version(versdic))
        self._versions = versions
        self.logger.info('Version '+(len(versions)*('{0:4d}')).format(*self.version))  
        # --- ThAr spectrum
        self._wavereference = self.settings['wavereference']
        self.logger.info('ThAr wavereference from {}'.format(self.wavereference))
        # --- orders
        try:
            orders = self.settings['orders']
            self.orders = orders
            self.sOrder = np.min(orders)
            self.eOrder = np.max(orders)
            msg1 = "{}".format(self.sOrder)
            msg2 = "{}".format(self.eOrder)
        except:
            try:
                self.sOrder = self.settings['sOrder'] 
                msg1 = "{}".format(self.sOrder)
            except:
                self.sOrder = None
                msg1 = "{} (default in harps.settings)".format(hs.sOrder)
            
            try:
                self.eOrder = self.settings['eOrder']
                msg2 = "{}".format(self.eOrder)
            except:
                self.eOrder = None
                msg2 = "read from FITS file header"
            
        finally:
            self.logger.info("Starting / ending order: " + \
                             "{} / {}".format(msg1,msg2))
        # --- remove false lines
        try:
            self.remove_false_lines = self.settings['remove_false_lines']
        except:
            self.remove_false_lines = True
        finally:
            msg3 = self.remove_false_lines
            self.logger.info("Try to remove false minima: {}".format(msg3))
        # --- LFC frequencies
        try:
            self.offset = self.settings['f0']
            self.reprate = self.settings['fr']
            self.logger.info("Anchor / repetition frequencies: " + \
                             "{0:4.2f}GHz".format(self.offset/1e9) + " / " +\
                             "{0:4.2f}GHz".format(self.reprate/1e9))
        except:
            try:
                self.lfc = self.settings['LFC']
            except:
                raise ValueError("No information on LFC. Provide f0 and fr or "
                                 "LFC name")
        # --- fittype
        self.fittype = np.atleast_1d(self.settings['fittype'])
        self.logger.info("Line-spread function: " + \
                         (len(self.fittype)*("{}")).format(*self.fittype))
        
        # --- fittype
        try:
            self.debug = self.settings['debug']
        except:
            self.debug = False
        self.logger.info("Debug: {}".format(self.debug))
        
        # --- do all comb specific calculations?
        try:
            self.do_comb_specific = self.settings['do_comb_specific']
        except:
            self.do_comb_specific = True
        self.logger.info("Perform wavelength calibration: {}".format(self.do_comb_specific))
        # --- overwrite
        self.overwrite  = self.settings['overwrite']
        self.logger.info('OVERWRITE existing files {}'.format(self.overwrite))
        # --- output file
        self.open_outfits() 
        
    def __len__(self):
        '''
        Returns the number of files that still need to be processed. 
        This number is equal to all files if rewrite is true.
        '''
        try:
            return self._numfiles
        except:
            fl = self.filelist
            self._numfiles = len(fl)
            return len(fl)
    
    def __call__(self,nproc=None,*args,**kwargs):
        '''
        Process exposures provided in the input file and produce files needed
        for other computations.
        Wrapper around self.run
        '''
        return run(self.filelist,self.settings)
    
    def init_logger(self):
        '''
        Sets up the logging files.
        '''
        hs.setup_logging()     
        return

    @property
    def settings(self):
        '''
        Returns a dictionary of all values in the Settings file.
        '''
        return self._settings.__dict__
        
    @settings.setter
    def settings(self,filepath):
        '''
        Sets the Settings object to 'filepath'.
        '''
        self._settings = hs.Settings(filepath)
    @property
    def output_dir(self):
        '''
        Returns the path to the directory containing the 'output.dat' file.
        '''
        return self.settings['outdir']
    @property
    def output_file(self):
        '''
        Returns the path to the 'output.dat' file
        '''
        return os.path.join(self.output_dir,self.json_file+'.dat')
    @property
    def json_file(self):
        '''
        Returns the basename of the json file containing the settings.
        '''
        return self.settings['selfname']
    @property
    def e2dslist(self):
        '''
        Returns a list of paths to all e2ds files contained in 'e2dslist'.
        '''
        return io.read_textfile(self.settings['e2dslist'])
    @property
    def wavereference(self):
        '''
        Returns the path to the e2ds file from which ThAr calibration
        is read.
        '''
        return self._wavereference
    @property
    def nproc(self):
        '''
        Returns the number of processors used to process the files.
        '''
        try:
            return self._nproc
        except:
            self.nproc=None
            return self._nproc
    @nproc.setter
    def nproc(self,value):
        '''
        Sets the number of processors used to process the files.
        
        If None provided, tries reading 'nproc' from the settings file. 
        Uses half of the system processors if this value is not given.
        
        It also checks that not more processors are used than there are files
        to process and that this number is also not smaller than one.
        '''
        if value is None:
            try:
                _nproc   = self.settings['nproc']
            except:
                _nproc   = mp.cpu_count()//2
        else:
            _nproc = value
        # do not use more processors than files
        if len(self)<_nproc:
            _nproc = len(self)  
        # but use more than 1  
        if _nproc < 1:
            _nproc = mp.cpu_count()//2
            self.logger.info('Invalid number of processors provided. ' + \
                             'Using half of the system processors ({})'.format(_nproc))
        self._nproc = _nproc
        return self._nproc
    def open_outfits(self):
        '''
        Checks if the output file exists and creates it if it does not.
        '''
        write_header = False
        # make folder if does not exist
        success = hs.make_directory(self.output_dir)
        if success:
            pass
        else:
            raise ValueError("Could not make directory")
        # append if file exists, make new if not
        if not self.overwrite and os.path.isfile(self.output_file):
            mode = 'a'
        else:
            mode = 'w'
            write_header = True
        with open(self.output_file,mode) as outfile:
            if write_header:
                outfile.write("# Created by "
                              "{}\n".format(self.settings['selfpath']))
            self.logger.info("Opened {} file "
                             ", mode '{}'".format(self.output_file,mode))
        return    

    @property
    def filelist(self):
        '''
        Returns a list of e2ds files which still need to be processed. If 
        overwrite is True, returns the full input file list.
        '''
        logger = logging.getLogger(__name__+'.filelist')
        e2dslist = self.settings['e2dslist']
        logger.info("Reading filelist from {}".format(e2dslist))
        todo_full = np.sort(self.e2dslist)
        if self.overwrite :
            ff = todo_full
        else:
            logger.info("{} files to process".format(len(todo_full)))
            done_full = np.sort(io.read_textfile(self.output_file))
            logger.info("{} already processed".format(len(done_full)))
            done_base = np.array([get_base(f) for f in done_full])
            todo_base = np.array([get_base(f) for f in todo_full])
            index     = ~np.isin(todo_base,done_base)
            todo_now  = todo_full[index]
            ff = todo_now
        
        self._filelist = ff
        self._numfiles = len(ff)
        return self._filelist
        
  
    @property
    def version(self):
        '''
        Returns a list of versions specified in the settings file.
        '''
        return self._versions
    
    def run(self,nproc=None):
        '''
        Starts the calculations on input e2ds files according to the settings 
        provided. Keeps a log.
        
        Args:
        ----
            nproc: int, number of processors to use. Uses half the available 
                   system processors if None provided.
        '''
        # https://stackoverflow.com/questions/6672525/multiprocessing-queue-in-python
        # answer by underrun
        self.nproc = nproc
    
        files   = self.filelist
        
        # --- log something    
        logger = logging.getLogger(__name__+'.run')
        if not len(files)==0 :
            logger.info("Running {} files ".format(len(files)) + \
                             "on {} processors".format(self.nproc))
        else:
            logger.info("All data already processed, exiting")
            return
        start       = time.time()
        logger.info("Start time {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        
        # -- divide files into chunks and send each into multiprocessing queue
        chunks = np.array_split(files,self.nproc)
        
        self.processes = []
        queue     = mp.JoinableQueue()
        for i in range(self.nproc):
            if len(chunks[i])<1:
                continue
            p = mp.Process(target=self._work_on_chunk_process,args=(queue,))
            p.deamon = True
            p.start()
            self.processes.append(p)
        
        for chunk in chunks:
            queue.put(chunk)
        queue.join()
        for proc in self.processes:
            queue.put(None)
        queue.join()

        for p in self.processes:
            p.join()
        # p = mp.Pool(self.nproc)
        # p.map(self._single_file,files)
        # p.close()
        end       = time.time()
        worktime  = end - start
        logger.info("End time {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        logger.info("Total time "
                    "{0:02d}h {1:02d}m {2:02d}s".format(*hf.get_time(worktime)))
        return None
    
        
    def _item_to_version(self,item=None):
        return hf.item_to_version(item)
    
    def _read_filelist(self,filepath):
        '''
        Wrapper around harps.io.read_textfile.
        '''
        return io.read_textfile(filepath)
    
def run(filelist,settings):
    result = []
    nprocs = settings['nproc']
    # chunks = np.array_split(files,nproc)
    for file in filelist:
        result.append([_single_file(file,settings)])
    return result
def _spec_kwargs(settings):
    '''
    Returns a dictionary of keywords and correspodning values that are 
    provided to harps.spectrum.Spectrum class inside self._single_file. 
    The keywords are hard coded, values should be given in the settings 
    file.
    '''
    
    kwargs = {}
    
    keywords = ['f0','fr','debug','dirpath','overwrite','sOrder','eOrder',
                'wavereference']
    
    for key in keywords:
        try:
            kwargs[key] = settings[key]
        except:
            kwargs[key] = None
    return kwargs
   
    
def _single_file(filepath,settings):
    '''
    Main routine to analyse e2ds files. 
    
    Performs line identification and fitting as well as wavelength 
    calibration. Uses provided settings to set the range of echelle orders
    to analyse, line-spread function model, ThAr calibration, etc. 
    Keeps a log.
    
    Args:
    ----
        filepath (str): path to the e2ds file
    '''
    def get_item(spec,item,version,**kwargs):
        try:
            itemdata = spec[item,version]
            message  = 'saved'
            #print("FILE {}, ext {} success".format(filepath,item))
            del(itemdata)
        except:
            message  = 'failed, trying with __call__(write=True)'
            try:
                itemdata = spec(item,version,write=True)
                del(itemdata)
            except:
                message = 'FAILED'
            
        finally:
            logger.info("SPECTRUM {}".format(get_base(filepath)) +\
                        " item {}".format(item.upper()) +\
                        " version {}".format(version) +\
                        " {}".format(message))
        return
    def comb_specific(fittype):
        comb_items = ['coeff','wavesol','residuals','model']
        return ['{}_{}'.format(item,fittype) for item in comb_items]
    logger    = logging.getLogger(__name__+'.single_file')
    versions  = np.atleast_1d(settings['version'])
    
    speckwargs = _spec_kwargs(settings) 
    print(speckwargs)
    if settings['LFC']=="ESPRESSO":
        spec  = hspec.ESPRESSO(filepath,**speckwargs)
    elif settings['LFC']=="HARPS":
        spec  = hspec.HARPS(filepath,**speckwargs)
        # replace ThAr with reference
        spec.wavereference_object = ThAr(settings['wavereference'],
                                  vacuum=True)

    try:
        lsfpath = settings['lsf']
    except:
        lsfpath = None
    linelist = spec('linelist',order=(settings['sOrder'],
                                      settings['eOrder']),write=True,
                    fittype=settings['fittype'],
                    lsf=lsfpath,
                    remove_false=settings['remove_false_lines'])
 
    
    basic    = ['flux','error','envelope','background','weights',
                'noise','wavereference'] 
    for item in basic:
        get_item(spec,item,None)
    if settings['do_comb_specific']:
        combitems = []
        for fittype in np.atleast_1d(settings['fittype']):
            combitems = combitems + comb_specific(fittype) 
        for item in combitems:
            if item in ['model_lsf','model_gauss']:
                get_item(spec,item,None,
                         lsf=lsfpath)
            else:
                for version in versions:
                    get_item(spec,item,version)
            pass
    else:
        pass
        
        
    savepath = spec._outpath + '\n'
    with open(settings['outlist'],'a+') as outfile:
        outfile.write(savepath)
    logger.info('Spectrum {} FINISHED'.format(get_base(filepath)))
    del(spec); 
    
    return savepath
def _work_on_chunk_process(self,queue):
    '''
    Takes an item (list of files to process) from the queue and runs 
    _single_file on each file. Keeps a log.
    '''
    sentinel = None
    while True:
        chunk_ = queue.get()
        # only continue if provided with a list
        if not (isinstance(chunk_,list) or isinstance(chunk_,np.ndarray)):
            if chunk_ == sentinel:
                break 
        
        chunk  = np.atleast_1d(chunk_)
        logger = logging.getLogger(__name__+'.chunk')
        for i,filepath in enumerate(chunk):
            self._single_file(filepath)
            hf.update_progress((i+1)/np.size(chunk),logger=logger)
        queue.task_done()
    queue.task_done()
def _work_on_chunk_pool(self,queue):
    '''
    Takes an item (list of files to process) from the queue and runs 
    _single_file on each file. Keeps a log.
    '''
    sentinel = None
    while True:
        chunk_ = queue.get()
        # only continue if provided with a list
        if not (isinstance(chunk_,list) or isinstance(chunk_,np.ndarray)):
            if chunk_ == sentinel:
                break 
        
        chunk  = np.atleast_1d(chunk_)
        logger = logging.getLogger(__name__+'.chunk')
        for i,filepath in enumerate(chunk):
            self._single_file(filepath)
            hf.update_progress((i+1)/np.size(chunk),logger=logger)
        queue.task_done()
    queue.task_done()
            
class Series(object):
    def __init__(self,outfile):
        self.output_file = outfile
        self._cache = {}
    def __getitem__(self,item):
        '''
        Returns an (N,M) shaped numpy array. 
        N = number of exposures
        M = number of values for this item.
        
        Caches the array if not already done.
        '''
        
        item, args, arg_sent = self._extract_item(item)
        assert item in ['flux','b2e','temp','exptime','date-obs','pressure',
                        'lfc_slmlevel','lfc_status']
        try:
            value = self._cache[item]
        except:
            if item not in self._cache:
                dct,n = io.mread_outfile_primheader(self.output_file,item)
                value = dct[item]
                if item=='date-obs':
                    value = np.ravel(value)
                self._cache[item] = value
        return value
         
    def _extract_item(self,item):
        arg_sent = False
        arg      = None
        if isinstance (item,tuple):
            nitem=len(item)
            if nitem == 1:
                ext=item[0]
            elif nitem > 1:
                arg_sent=True
                ext,*arg=item
        else:
            arg_sent=False
            ext=item
        return ext,arg,arg_sent
    def clear_cache(self):
        '''
        Deletes the content of the cache.
        '''
        self._cache = {}
        return
    def mread_header(self,item):
        items, args, arg_sent = self._extract_item(item)
        data, n = io.mread_outfile_primheader(self.output_file,item)
        
    def mread_outfile(self,extension,version):
        '''
        Returns a dictionary of shape {item:array}. 
        Each array is of shape (N,*(shape(array_exposure))), 
        where N is the number of exposures the rest of the shape is determined
        from the shape of the array for a single exposure.
        
        Item can be a list. 
        '''
#        extension, version, ver_sent = self._extract_item(item)
        data, n = io.mread_outfile(self.output_file,extension,version)
        return data
    def velshift_wavesol(self,fittype,version,sigma=3,refindex=0,**kwargs):
        '''
        Returns a structured numpy array with velocity shifts of each exposure
        with respect to the reference exsposure (default = first exposure).
        The velocity shift of a single exposure is the unweighted average of 
        velocity shifts of all wavelength calibrated pixels. 
        
        Args:
        -----
            fittype (str) : 'gauss' or 'lsf'
            version (int) : three-digit integer. First digit is the polynomial
                            order, second digit is binary 'gaps' flag, third
                            digit is the binary 'segmented' flag.
            sigma (float, list of float) : sigma-clipping of values 
                            around the mean. Default value = 3.
            refindex (int): exposure with respect to which velocity
                            shifts are calculated. Default value = 0
        Returns:
        -------
            rv (ndarray) : structured numpy array. 
                            Fields of the array are:
                             'flux'     : average flux per LFC line
                             'datetime' : UTC time of acquisition
                             'mean'     : mean velocity shift 
                             'sigma'    : uncertainty on the mean 
        
        '''
        ext     = ['wavesol_{}'.format(fittype),'datetime','avflux','avnoise']
        data,n  = io.mread_outfile(self.output_file,ext)
        waves2d = data['wavesol_{}'.format(fittype)]
        dates   = data['datetime']
        fluxes  = data['avflux']
        noises  = data['avnoise']
        rv      = vs.wavesol(waves2d,fittype,sigma,dates,fluxes,noises,refindex,
                          **kwargs)
        return rv
    def velshift_interpolate(self,use,fittype,sigma=3,refindex=0,
                          **kwargs):
        '''
        Returns a structured numpy array with velocity shifts of each exposure
        with respect to the reference exsposure (default = first exposure).
        The velocity shift of a single exposure is the weighted average of 
        velocity shifts of all LFC lines. Velocity shift of a line is 
        calculated as the distance from its expected frequency or position on
        the detector as interpolated by fitting a straight line between two 
        closest lines to this line.
        
        Args:
        -----
            use (str)     : 'freq' or 'centre'
            fittype (str) : 'gauss' or 'lsf'
            sigma (float, list of float) : sigma-clipping of values 
                            around the mean. Default value = 3.
            refindex (int): exposure with respect to which velocity
                            shifts are calculated. Default value = 0
        Returns:
        -------
            rv (ndarray) : structured numpy array. 
                            Fields of the array are:
                             'flux'     : average flux per LFC line
                             'datetime' : UTC time of acquisition
                             'mean'     : mean velocity shift 
                             'sigma'    : uncertainty on the mean 
        
        '''
        ext     = ['linelist','datetime','avflux']
        data,n  = io.mread_outfile(self.output_file,ext)
        linelist = data['linelist']
        dates    = data['datetime']
        fluxes   = data['avflux']
        rv       = vs.interpolate(linelist,fittype,sigma,dates,fluxes,use=use,
                          refindex=refindex)
        return rv
    def velshift_interpolate_freq(self,fittype,sigma=3,refindex=0,
                                  **kwargs):
        '''
        Wrapper around self.velshift_interpolate where use='freq'.
        '''
        rv = self.rv_interpolate('freq',fittype,sigma,refindex,**kwargs)
        return rv
    def velshift_interpolate_cent(self,fittype,sigma=3,refindex=0,
                                  **kwargs):
        '''
        Wrapper around self.velshift_interpolate where use='centre'.
        '''
        rv = self.velshift_interpolate('centre',fittype,
                                       sigma,refindex,**kwargs)
        return rv
    def velshift_coefficients(self,fittype,version,sigma=3,refindex=0,
                              **kwargs):
        '''
        Returns a structured numpy array with velocity shifts of each exposure
        with respect to the reference exsposure (default = first exposure).
        The velocity shift of a single exposure is the weighted average of 
        velocity shifts of all LFC lines. Velocity shift of a single LFC line 
        is calculated from the difference in the wavelength resulting from 
        shifts in the line's centre. This assumes the use of coefficients, 
        which are automatically calculated (see harps.velshift.coefficients).
        
        Args:
        -----
            fittype (str) : 'gauss' or 'lsf'
            version (int) : three-digit integer. First digit is the polynomial
                            order, second digit is binary 'gaps' flag, third
                            digit is the binary 'segmented' flag.
            sigma (float, list of float) : sigma-clipping of values 
                            around the mean. Default value = 3.
            refindex (int): exposure with respect to which velocity
                            shifts are calculated. Default value = 0
        Returns:
        -------
            rv (ndarray) : structured numpy array. 
                            Fields of the array are:
                             'flux'     : average flux per LFC line
                             'datetime' : UTC time of acquisition
                             'mean'     : mean velocity shift 
                             'sigma'    : uncertainty on the mean 
        
        '''
        ext     = ['linelist','datetime','avflux']
        data,n  = io.mread_outfile(self.output_file,ext)
        linelist = data['linelist']
        dates    = data['datetime']
        fluxes   = data['avflux']
        rv       = vs.coefficients(linelist,fittype,version,sigma,dates,fluxes,
                                refindex=refindex,**kwargs)
        return rv    
    
def get_base(filename):
    basename = os.path.basename(filename)
    return basename[0:29]  
