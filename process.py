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

from harps.core import np, mp, json, os, gc, glob, time
import logging

from harps.spectrum import Spectrum
from harps.wavesol import ThAr

import harps.io as io
import harps.functions as hf
import harps.settings as hs

#logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
#                    datefmt = '%Y-%m-%d %H:%M:%S')
__version__      = hs.__version__

class Process(object):
    def __init__(self,setting_json):
        self._settings = hs.Settings(setting_json)
        # -----------------   L O G G E R   -----------------
        # logger
        now       = time.strftime('-%Y-%m-%d_%H-%M-%S')
        self._log =  self.settings['log'] +now+ '.log'
        self.logger = logging.getLogger('process')
        self.logger.setLevel(logging.DEBUG)
        # file handler
        fh     = logging.FileHandler(self._log)
        self.filehandler = fh
        # formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                      '%(levelname)s: %(message)s')#,

        # add formatter to handler
        fh.setFormatter(formatter)
        # add handler to logger
        self.logger.addHandler(fh)
        self.logger.info("INITIALIZING NEW PROCESS")
        # -----------------   M A I N   -----------------
        
        self._outdir   = self.settings['outdir']
        self._jsonname = self.settings['selfname']
        self.logger.info("Using {}".format(self.settings['selfpath']))
        
        self._outfits  = os.path.join(self._outdir,self._jsonname+'.dat')
        self.logger.info("Saving .dat file path to {}.json".format(self._jsonname))
        
        self._settings.update(dict(outlist=self._outfits))
        self._settings.write()
        self._filelist = None
        self.open_outfits()
        self._nproc    = self.settings['nproc']
        try:
            self._versions  = np.atleast_1d(self.settings['version'])
        except:
            versdic = dict(polyord=self.settings['polyord'],
                           gaps=self.settings['gaps'],
                           segment=self.settings['segment'])
            versions = np.atleast_1d(hf.item_to_version(versdic))
            self._versions = versions
        self.logger.info('VERSION {}'.format(self.version))  
        # log reference
        self._reference = self.settings['refspec']
        
        try:
            self._sOrder = self.settings['sorder']
        except:
            self._sOrder = None
        try:
            self._eOrder = self.settings['eorder']
        except:
            self._eOrder = None
        
        self.logger.info('REFERENCE SPECTRUM {}'.format(self.reference))
        # log overwrite
        self.overwrite  = self.settings['overwrite']
        self.logger.info('OVERWRITE {}'.format(self.overwrite))
        
    def __len__(self):
        return self._numfiles
    @property
    def settings(self):
        return self._settings.__dict__
        
    @settings.setter
    def settings(self,filepath):
        self._settings = hs.Settings(filepath)
        
       
    def open_outfits(self):
        write_header = False
        success = hs.make_directory(self._outdir)
        if success:
            pass
        else:
            raise ValueError("Could not make directory")
        if os.path.isfile(self._outfits):
            mode = 'a'
        else:
            mode = 'w'
            write_header = True
        with open(self._outfits,mode) as outfile:
            if write_header:
                outfile.write("# Created by "
                              "{}\n".format(self.settings['selfpath']))
            self.logger.info("Opened {} file "
                             ", mode '{}'".format(self._outfits,mode))
            
    @staticmethod
    def get_base(filename):
            basename = os.path.basename(filename)
            return basename[0:29]    
    @property
    def filelist(self):
        logger = logging.getLogger('process.filelist')
        e2dslist = self.settings['e2dslist']
        logger.info("Reading filelist from {}".format(e2dslist))
        todo_full = np.sort(self._read_filelist(e2dslist))
        if self.overwrite :
            ff = todo_full
        else:
            logger.info("{} files to process".format(len(todo_full)))
            done_full = np.sort(self._read_filelist(self._outfits))
            logger.info("{} already processed".format(len(done_full)))
            done_base = np.array([Process.get_base(f) for f in done_full])
            todo_base = np.array([Process.get_base(f) for f in todo_full])
            index     = ~np.isin(todo_base,done_base)
            todo_now  = todo_full[index]
            ff = todo_now
        
        self._filelist = ff
        self._numfiles = len(ff)
        return self._filelist
        
  
    @property
    def version(self):
        return self._versions
#    @version.setter
#    def version(self,item):
#        self._version = self._item_to_version(item)
    
    def __call__(self):
        nproc   = self.settings['nproc']
        return self.run(nproc)
    
    def run(self,nproc):
        files   = self.filelist
        if len(files)<nproc:
            _nproc = len(files)
        else:
            _nproc = nproc
        logger = logging.getLogger('process.run')
        if not len(files)==0 :
            logger.info("Running {} files ".format(len(files)) + \
                             "on {} processors".format(_nproc))
        else:
            logger.info("All data already processed, exiting")
            return
        start       = time.time()
        logger.info("Start time {}".format(time.strftime('%Y-%m-%d_%H:%M:%S')))
        
        
        
        chunks = np.array_split(files,_nproc)
        
        self.processes = []
        self.queue     = mp.Queue()
        for i in range(nproc):
            chunk = chunks[i]
            if len(chunk)==0:
                continue
#            print(i,chunk)
            p = mp.Process(target=self._work_on_chunk,args=((chunk,)))
            p.start()
            self.processes.append(p)
        for p in self.processes:
            p.join()
            
        while True:
            time.sleep(5)
            if not mp.active_children():
                break
#        while self.queue.empty() == True:
#            time.sleep(5)
        for i in range(len(files)):
            outfits = self.queue.get()          
            print('{0:>5d} element extracted'.format(i))
        end       = time.time()
        worktime  = end - start
        logger.info("End time {}".format(time.strftime('%Y-%m-%d_%H:%M:%S')))
        logger.info("Total time "
                    "{0:2d}h{1:2d}m{2:2d}s".format(*hf.get_time(worktime)))
        logger.info("EXIT")
    def _item_to_version(self,item=None):
        return hf.item_to_version(item,default=self.version)
    
    def _read_filelist(self,filepath):
        return io.read_textfile(filepath)
    
    def _extract_item(self,item):
        return hf.extract_item(item,default=self.version)
    
    @property
    def reference(self):
        return self._reference

#    @property
#    def log(self):
    
    def _single_file(self,filepath):
        def get_item(spec,item,version):
            try:
                itemdata = spec[item,version]
                print("FILE {}, ext {} success".format(filepath,item))
                del(itemdata)
            except:
                itemdata = spec(item,version,write=True)
                print("FILE {}, ext {} fail".format(filepath,item))
                logger.error("{} failed {}".format(item.upper(),filepath))
                del(itemdata)
            finally:
                pass
            return
        def comb_specific(fittype):
            comb_items = ['coeff','wavesol','residuals','model']
            return ['{}_{}'.format(item,fittype) for item in comb_items]
        logger    = logging.getLogger('process.single_file')
        versions  = self.version
        
        anchoff   = self.settings['anchor_offset']
        dirpath   = self.settings['outfitsdir']
        spec      = Spectrum(filepath,LFC=self.settings['LFC'],
                             dirpath=dirpath,
                             overwrite=self.overwrite,
                             anchor_offset=anchoff)
        
        fb        = spec.meta['fibre']
        # replace ThAr with reference
        spec.ThAr = ThAr(self.settings['refspec']+\
                           "_e2ds_{fb}.fits".format(fb=fb),
                           vacuum=True)
        
        if self._sOrder is not None and self._eOrder is not None:
            print("Limit in orders: {}-{}".format(self._sOrder,self._eOrder))
            orders   = np.arange(self._sOrder,self._eOrder)
            linelist = spec('linelist',order=orders,write=True)
        else:
            linelist = spec['linelist']
        
        basic    = ['flux','background','error','weights'] 
        totitems = basic
        for fittype in np.atleast_1d(self.settings['fittype']):
            totitems = totitems + comb_specific(fittype) 
        for item in totitems:
            if item in ['model_lsf','model_gauss']:
                get_item(spec,item,None)
            else:
                for version in versions:
                    get_item(spec,item,version)
            pass
            
            
        savepath = spec._outfits + '\n'
        with open(self._outfits,'a+') as outfile:
            outfile.write(savepath)
        
        del(spec); 
        logger.info("Saved SPECTRUM {} ".format(Process.get_base(filepath)))
        #gc.collect()
        return savepath
    def _work_on_chunk(self,chunk):  
        chunk = np.atleast_1d(chunk)
        for i,filepath in enumerate(chunk):
            self._single_file(filepath)
            self.queue.put(filepath)
            hf.update_progress((i+1)/np.size(chunk))
        