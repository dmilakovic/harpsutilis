#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:18:41 2018

@author: dmilakov
"""

from harps.core import np, mp, json, os, gc, glob, time, logging

from harps.classes import Spectrum

import harps.functions as hf
import harps.settings as hs

#logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
#                    datefmt = '%Y-%m-%d %H:%M:%S')

class Process(object):
    def __init__(self,setting_json):
        self._settings = hs.Settings(setting_json)
        # -----------------   L O G G E R   -----------------
        # logger
        now       = time.strftime('%Y-%m-%d_%H-%M-%S')
        self._log = self.settings['log'] + now + '.log'
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
            self._version  = self.settings['version']
        except:
            versdic = dict(polyord=self.settings['polyord'],
                           gaps=self.settings['gaps'],
                           segment=self.settings['segment'])
            version = hf.item_to_version(versdic,default=300)
            self._version = version
        self.logger.info('VERSION {}'.format(self.version))  
        # log reference
        self._reference = self.settings['refspec']
        self.logger.info('REFERENCE SPECTRUM {}'.format(self.reference))
        # log overwrite
        self.overwrite  = self.settings['overwrite']
        self.logger.info('OVERWRITE {}'.format(self.overwrite))
        
  
    @property
    def settings(self):
        return self._settings.__dict__
        
    @settings.setter
    def settings(self,filepath):
        self._settings = hs.Settings(filepath)
        
       
    def open_outfits(self):
        write_header = False
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
        logger.info("{} files to process".format(len(todo_full)))
        done_full = np.sort(self._read_filelist(self._outfits))
        logger.info("{} already processed".format(len(done_full)))
        done_base = np.array([Process.get_base(f) for f in done_full])
        todo_base = np.array([Process.get_base(f) for f in todo_full])
        index     = ~np.isin(todo_base,done_base)
        todo_now  = todo_full[index]
        self._filelist = todo_now
        return self._filelist
        
  
    @property
    def version(self):
        return self._version
    @version.setter
    def version(self,item):
        self._version = self._item_to_version(item)
    
    def __call__(self):
        return self.run()
    
    def run(self):
        nproc   = self.settings['nproc']
        files   = self.filelist
        logger = logging.getLogger('process.run')
        if not len(files)==0 :
            logger.info("Running {} files ".format(len(files)) + \
                             "on {} processors".format(nproc))
        else:
            logger.info("All data already processed, exiting")
            return
        start       = time.time()
        logger.info("Start time {}".format(time.strftime('%Y-%m-%d_%H-%M-%S')))
        chunks = np.array_split(files,nproc)
        
        self.processes = []
        self.queue     = mp.Queue()
        for i in range(nproc):
            chunk = chunks[i]
            if len(chunk)==0:
                continue
#            print(i,chunk)
            p = mp.Process(target=self._work_on_chunk,args=((chunk,)))
            self.processes.append(p)
            p.start()

        for p in self.processes:
            p.join(timeout=2)  
        while self.queue.empty() == True:
            time.sleep(5)
        for i in range(len(files)):
            outfits = self.queue.get()          
            print('Queue element extracted')
        end       = time.time()
        worktime  = end - start
        logger.info("End time {}".format(time.strftime('%Y-%m-%d_%H-%M-%S')))
        logger.info("Total time {}".format(hf.get_worktime(worktime)))
        logger.info("EXIT")
    def _item_to_version(self,item=None):
        return hf.item_to_version(item,default=self.version)
    
    def _read_filelist(self,filepath):
        if os.path.isfile(filepath):
            mode = 'r+'
        else:
            mode = 'a+'
        filelist=[line.strip('\n') for line in open(filepath,mode)
                  if line[0]!='#']
        return filelist
    
    def _extract_item(self,item):
        return hf.extract_item(item,default=self.version)
    
    @property
    def reference(self):
        return self._reference

#    @property
#    def log(self):
        
    def _single_file(self,filepath):
        
        logger    = logging.getLogger('process.single_file')
        version   = self.version
        
        spec      = Spectrum(filepath,LFC=self.settings['LFC'],
                             clobber=self.overwrite)
        
        fb        = spec.meta['fibre']
        refspec   = Spectrum(self.settings['refspec']+\
                           "_e2ds_{fb}.fits".format(fb=fb))
        # replace ThAr with reference
        spec.tharsol = refspec._tharsol
        linelist = spec['linelist']
        coeff    = spec['coeff',version]
        combsol  = spec['wavesol_comb',version]
        resids   = spec['residuals',version]
        model    = spec['model_gauss']
            
        savepath = spec._outfits + '\n'
        with open(self._outfits,'a+') as outfile:
            outfile.write(savepath)
        
        del(spec);    del(linelist); del(coeff)
        del(combsol); del(resids);   del(model)
        logger.info("Saved SPECTRUM {} ".format(Process.get_base(filepath)))
        gc.collect()
        return savepath
    def _work_on_chunk(self,chunk):  
        if type(chunk)==np.int64:
            chunk=[chunk]
        outputs = [self._single_file(filepath) for filepath in chunk]
        self.queue.put(outputs)
        