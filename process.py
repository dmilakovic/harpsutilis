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

from   harps.core import np, mp, json, os, gc, glob, time, plt
from   harps.settings import __version__ as hs_version
import logging
from   numpy.lib.recfunctions import append_fields

from   harps.spectrum import Spectrum
from   harps.wavesol import ThAr
from   harps.decorators import memoize

import harps.io as io
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
        self._cache = {}
        
#        self.nproc    = self.settings['nproc']
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
        self._thar = self.settings['thar']
        self.logger.info('ThAr calibration from {}'.format(self.thar))
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
                self.sOrder = self.settings['sorder'] 
                msg1 = "{}".format(self.sOrder)
            except:
                self.sOrder = None
                msg1 = "{} (default in harps.settings)".format(hs.sOrder)
            
            try:
                self.eOrder = self.settings['eorder']
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
        
        # --- overwrite
        self.overwrite  = self.settings['overwrite']
        self.logger.info('OVERWRITE existing files {}'.format(self.overwrite))
        # --- output file
        self.open_outfits() 
        
    def __len__(self):
        try:
            return self._numfiles
        except:
            fl = self.filelist
            return len(fl)
    
    def __call__(self,nproc=None,*args,**kwargs):
        '''
        Process exposures provided in the input file and produce files needed
        for other computations.
        '''
        if nproc is None:
            try:
                nproc   = self.settings['nproc']
            except:
                nproc   = mp.cpu_count()//2
        else:
            nproc = nproc
        return self.run(nproc,*args,**kwargs)
    
    def __getitem__(self,item):
        # make sure all input 'e2ds' files have appropriate 'out' files
        if len(self) > 0:
            self.__call__()
        
        item, args, arg_sent = self._extract_item(item)
        assert item in ['flux','b2e','temp','exptime','date-obs','pressure']
        if item not in self._cache:
            value = io.mread_outfile_primheader(self.output_file,item)[0][item]
            self._cache[item] = value
        else:
            value = self._cache[item]
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
    def init_logger(self):
        # -----------------   L O G G E R   -----------------
        # logger
        hs.setup_logging()
        
        return
    @property
    def settings(self):
        return self._settings.__dict__
        
    @settings.setter
    def settings(self,filepath):
        self._settings = hs.Settings(filepath)
    @property
    def output_dir(self):
        return self.settings['outdir']
    @property
    def output_file(self):
        return os.path.join(self.output_dir,self.json_file+'.dat')
    @property
    def json_file(self):
        return self.settings['selfname']
    @property
    def e2dslist(self):
        return io.read_textfile(self.settings['e2dslist'])
    @property
    def thar(self):
        return self._thar
    @property
    def nproc(self):
        try:
            return self._nproc
        except:
            self.nproc=None
            return self._nproc
    @nproc.setter
    def nproc(self,value):
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
            
    @staticmethod
    def get_base(filename):
            basename = os.path.basename(filename)
            return basename[0:29]    
    @property
    def filelist(self):
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
        return self._versions
    def mread_outfile(self,item):
        extension, version, ver_sent = hf.extract_item(item)
        data, n = io.mread_outfile(self.output_file,extension,version)
        return data
    def rv_wavesol(self,fittype,version,sigma=3,refindex=0,**kwargs):
        ext     = ['wavesol_{}'.format(fittype),'datetime','avflux','avnoise']
        data,n  = io.mread_outfile(self.output_file,ext)
        waves2d = data['wavesol_{}'.format(fittype)]
        dates   = data['datetime']
        fluxes  = data['avflux']
        noises  = data['avnoise']
        rv      = vs.wavesol(waves2d,fittype,sigma,dates,fluxes,noises,refindex,
                          **kwargs)
        return rv
    def rv_interpolate(self,use,fittype,version,sigma=3,refindex=0,**kwargs):
        ext     = ['linelist','datetime','avflux']
        data,n  = io.mread_outfile(self.output_file,ext)
        linelist = data['linelist']
        dates    = data['datetime']
        fluxes   = data['avflux']
        rv       = vs.interpolate(linelist,fittype,sigma,dates,fluxes,use=use,
                          refindex=refindex)
        return rv
    def rv_interpolate_freq(self,fittype,version,sigma=3,refindex=0,**kwargs):
        rv = self.rv_interpolate('freq',fittype,version,sigma,refindex,**kwargs)
        return rv
    def rv_interpolate_cent(self,fittype,version,sigma=3,refindex=0,**kwargs):
        rv = self.rv_interpolate('centre',fittype,version,sigma,refindex,**kwargs)
        return rv
    def rv_coefficients(self,fittype,version,sigma=3,refindex=0,**kwargs):
        ext     = ['linelist','datetime','avflux']
        data,n  = io.mread_outfile(self.output_file,ext)
        linelist = data['linelist']
        dates    = data['datetime']
        fluxes   = data['avflux']
        rv       = vs.coefficients(linelist,fittype,version,sigma,dates,fluxes,
                                refindex=refindex,fibre=self.fibre,**kwargs)
        return rv
    
    def run(self,nproc=None):
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
            print(i,chunks[i])
#            p = mp.Process(target=self._work_on_chunk,args=((chunk,)))
            p = mp.Process(target=self._work_on_chunk,args=(queue,))
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
            
        end       = time.time()
        worktime  = end - start
        logger.info("End time {}".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        logger.info("Total time "
                    "{0:02d}h {1:02d}m {2:02d}s".format(*hf.get_time(worktime)))
        return None
    def _item_to_version(self,item=None):
        return hf.item_to_version(item)
    
    def _read_filelist(self,filepath):
        return io.read_textfile(filepath)
    
#    def _extract_item(self,item):
#        return hf.extract_item(item)
    
    
    
    def _spec_kwargs(self):
        settings = self.settings
        
        kwargs = {}
        
        keywords = ['f0','fr','debug','dirpath','overwrite','sOrder','eOrder']
        
        for key in keywords:
            try:
                kwargs[key] = settings[key]
            except:
                kwargs[key] = None
        return kwargs
    
    def _single_file(self,filepath):
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
        versions  = self.version
        
        speckwargs = self._spec_kwargs() 
        
        spec       = Spectrum(filepath,**speckwargs)
        # replace ThAr with reference
        spec.ThAr = ThAr(self.settings['thar'],vacuum=True)
        
        #if self.orders is not None:
            #print("Orders: {}".format(self.orders))
        
        try:
            lsfpath = self.settings['lsf']
        except:
            lsfpath = None
        linelist = spec('linelist',order=(self.sOrder,self.eOrder),write=True,
                        fittype=self.fittype,
                        lsf=lsfpath,
                        remove_false=self.remove_false_lines)
     
        
        basic    = ['flux','error','envelope','background','weights'] 
        for item in basic:
            get_item(spec,item,None)
        
        combitems = []
        for fittype in np.atleast_1d(self.settings['fittype']):
            combitems = combitems + comb_specific(fittype) 
        for item in combitems:
            if item in ['model_lsf','model_gauss']:
                get_item(spec,item,None,
                         lsf=lsfpath)
            else:
                for version in versions:
                    get_item(spec,item,version)
            pass
            
            
        savepath = spec._outpath + '\n'
        with open(self.output_file,'a+') as outfile:
            outfile.write(savepath)
        logger.info('Spectrum {} FINISHED'.format(get_base(filepath)))
        del(spec); 
        #logger.info("Saved SPECTRUM {} ".format(Process.get_base(filepath)))
        #gc.collect()
        return savepath
    def _work_on_chunk(self,queue):
        sentinel = None
        while True:
            chunk_ = queue.get()
            # only continue if provided with a list
            print(type(chunk_))
            if not (isinstance(chunk_,list) or isinstance(chunk_,np.ndarray)):
                if chunk_ == sentinel:
                    continue 
            
            chunk  = np.atleast_1d(chunk_)
            logger = logging.getLogger(__name__+'.chunk')
            for i,filepath in enumerate(chunk):
                self._single_file(filepath)
                hf.update_progress((i+1)/np.size(chunk),logger=logger)
            queue.task_done()
        queue.task_done()
            
    
    
def get_base(filename):
    basename = os.path.basename(filename)
    return basename[0:29]  
