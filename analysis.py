#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:10:20 2018

@author: dmilakov
"""
from harps.core import np, os, time, gc, glob, mp

import harps.classes as hc
import harps.settings as hs
import harps.functions as hf
import multiprocessing as mp

__version__=hs.__version__

sOrder = hs.sOrder
eOrder = hs.eOrder
default_sOrder = 43
default_eOrder = 72
nOrder = default_eOrder - default_sOrder


class Analyser(object):
    def __init__(self,manager,fibre,
                 filelim=None,LFC=None,LFC_reference=None,
                 use_reference=False,specnum_reference=0,
                 refspec_path = None,
                 savedir=None,line_dir=None,ws_dir=None,
                 savelines=True,savewavesol=True,
                 patches = True, gaps=False, 
                 polyord = 3,
                 anchor_offset = None,
                 fittype=['gauss','epsf'],
                 sOrder=None,eOrder=None):
        self.sOrder    = sOrder if sOrder is not None else None #default_sOrder
        self.eOrder    = eOrder if eOrder is not None else None #default_eOrder
        try:
            self.orders = np.arange(self.sOrder,self.eOrder,1)
        except:
            self.orders = None
        
        
        self.manager   = manager
        self.filepaths = manager.file_paths[fibre]
        self.reference = None
        self.nFiles    = len(self.filepaths)
        self.nOrder    = nOrder
        self.fibre     = fibre
        
        self.use_reference = use_reference
        self.refspec_path = refspec_path
        self.specnum_reference = specnum_reference
        self.savewavesol = savewavesol
        
        self.patches = patches
        self.gaps    = gaps
        self.fittype = fittype
        self.polyord = polyord
        
        
#        if filelim is not None:
#            self.LFC1    = LFC1 if LFC1 is not None else 'HARPS'
#            self.LFC2    = LFC2 if LFC2 is not None else 'HARPS'
#            self.filelim = filelim
#        else:
#            self.filelim = 'ZZZZZ'
#            self.LFC1    = LFC1 if LFC1 is not None else 'HARPS'
#            self.LFC2    = LFC2 if LFC2 is not None else 'HARPS'
        self.LFC=LFC
        self.anchor_offset = anchor_offset if anchor_offset is not None else 0.
        
        savedir  = savedir if savedir is not None else hs.harps_prod
        line_dir = line_dir if line_dir is not None else os.path.join(savedir,'lines')
        ws_dir   = ws_dir if ws_dir is not None else os.path.join(savedir,'LFCws')
        
        self.savedir = savedir
        self.line_dir = line_dir
        self.ws_dir   = ws_dir
            
        print('Saving lines to:   ',self.line_dir)
        print('Saving wavesols to:',self.ws_dir)
        reduced_filelist = self.reduce_filelist()
        self.filepaths    = reduced_filelist
        
        if len(self.filepaths)==0:
            print("All files processed, saved in :")
            print("\tlines:",self.line_dir)
            print("\tLFCws:",self.ws_dir)
        else:
            print("Files to process: {}".format(len(self.filepaths)))
            try:
                print("Spectral orders:", np.arange(sOrder,eOrder,1))
            except:
                pass
            
            self.initialize_reference()
        #print(self.filepaths)
        return
    def initialize_reference(self):
        if self.refspec_path is not None:
            spec0 = hc.Spectrum(self.refspec_path)
        else: 
            specnum = self.specnum_reference
            spec0 = hc.Spectrum(self.filepaths[specnum])
        thar0  = spec0.__get_wavesol__('ThAr')
        wavecoeff_air0 = spec0.wavecoeff_air
        wavecoeff_vac0 = spec0.wavecoeff_vacuum
        self.reference = dict(thar0=thar0,
                              wavecoeff_air0=wavecoeff_air0,
                              wavecoeff_vac0=wavecoeff_vac0)
        return 
    def get_reference(self,use_reference=None):
        use_reference = use_reference if use_reference is not None else self.use_reference
        if use_reference==False:
            return None
        else:
            try:
                reference = self.reference
            except:
                self.initialize_reference()
                reference = self.reference
            return reference
    def reduce_filelist(self):
        def get_base(filename):
            basename = os.path.basename(filename)
            return basename[0:36]
#        dirname          = os.path.dirname(self.reference)
        all_filepaths    = np.sort(self.filepaths)
        all_dirnames     = np.array([os.path.dirname(path) for path in all_filepaths])
        all_basenames    = np.array([get_base(file) for file in [os.path.basename(path) for path in all_filepaths]])
        
        existing_lines   = np.array([get_base(file) for file in \
                                     glob(os.path.join(self.line_dir,'*{}_lines.nc'.format(self.fibre)))])
        existing_ws      = np.array([get_base(file) for file in \
                                     glob(os.path.join(self.ws_dir,'*{}_LFCws.nc'.format(self.fibre)))])
        
        diff_lines       = np.setdiff1d(all_basenames,existing_lines)
        diff_ws          = np.setdiff1d(all_basenames,existing_ws)

        diff_basenames   = np.union1d(diff_lines,diff_ws)
       
        if len(diff_basenames) == 0:
            return []
    
        else:
            index            = np.isin(all_basenames,diff_basenames)
            diff_dirnames    = all_dirnames[index]
            reduced_filelist = [os.path.join(dirname,basename+'.fits') \
                                for dirname,basename in zip(diff_dirnames,diff_basenames)]
            return reduced_filelist
    def start_multiprocess(self,nproc=hs.nproc):
        self.nproc=nproc
        self.processes = []
        self.queue     = mp.Queue()
        print("Number of processes : ",nproc)
        if np.size(self.filepaths)>0:
            pass
        else:
            print("Nothing to do, exiting")
            return
        chunks = np.array_split(self.filepaths,self.nproc)
        for i in range(self.nproc):
            chunk = chunks[i]
            if len(chunk)==0:
                continue
            #print(i,chunk)
            p = mp.Process(target=self.work_on_chunk,args=((chunk,)))
            self.processes.append(p)
            #p.daemon=False
            p.start()
            #print('Processes started')
        #print('Pre-joining')
        for p in self.processes:
            print(p)
            p.join(timeout=2)  
        #print('Processes joined')
        #for p in self.processes:
            #print("Process is alive = ",p.is_alive())
        #print(data_out.empty())
        while self.queue.empty() == True:
            time.sleep(10)
        for i in range(self.nFiles):
            #print('Queue is empty = ',data_out.empty())
            elem = self.queue.get()
            print('Queue element {} extracted'.format(elem))

    def single_file(self,filepath,i):
        basename = os.path.basename(filepath)
        LFC = self.LFC
#        if basename < self.filelim:
#            LFC = self.LFC
#        else:
#            LFC = self.LFC2
        if LFC == 'HARPS':
            anchor_offset=self.anchor_offset#-100e6
            fibre_shape = 'round'
        elif LFC == 'FOCES':
            fibre_shape = 'round'
            anchor_offset=0
        
        spec = hc.Spectrum(filepath,LFC=self.LFC)
        spec.patches = self.patches
        spec.gaps    = self.gaps
        spec.fibre_shape = fibre_shape
        
        reference = self.get_reference()
        if self.reference is not None:
            spec.wavecoeff_air    = reference['wavecoeff_air0']
            spec.wavecoeff_vacuum = reference['wavecoeff_vac0']
            spec.wavesol_thar     = reference['thar0']
        else:
            spec.__get_wavesol__('ThAr')
        lines = spec.load_lines(self.line_dir)        
        if lines is None:
            #del(spec)
            #spec = hc.Spectrum(filepath,LFC=LFC)
            #spec.fibre_shape = fibre_shape
            print("{0:>4d}{1:>50s}{2:>8s}{3:>10s}".format(i,basename,LFC,' LINES working'))
            
            
            spec.detect_lines(order=self.orders)
            spec.fit_lines(order=self.orders,
                           fittype='gauss')
            spec.save_lines(self.line_dir)
            lines = spec.lines
        else:
            print("{0:>4d}{1:>50s}{2:>8s}{3:>10s}".format(i,basename,LFC,' LINES exists'))
        
        if self.savewavesol:
            ws    = spec.load_wavesol(self.ws_dir)
            if ws is None:
#                self.bad_orders=[]
                print("{0:>4d}{1:>50s}{2:>8s}{3:>10s}".format(i,basename,LFC,' WAVESOL working'))
                print('Patches = {0}\tGaps = {1}\tPolyord = {2}'.format(self.patches,self.gaps,self.polyord))
                spec.__get_wavesol__(calibrator='LFC',
                                     orders=self.orders,
                                     patches=self.patches,
                                     gaps=self.gaps,
                                     polyord=self.polyord,
                                     fittype=self.fittype,
                                     anchor_offset=anchor_offset)
                spec.save_wavesol(self.ws_dir)
                spec.save_lines(self.line_dir,replace=True)
            else:
                print("{0:>4d}{1:>50s}{2:>8s}{3:>10s}".format(i,basename,LFC,' WAVESOL exists'))
            
        del(spec)
        
        gc.collect()
    def start_singleprocess(self):
        reference = self.get_reference()
        for i,filepath in enumerate(self.filepaths):
            self.single_file(filepath,i)

    def work_on_chunk(self,chunk):  
        ''' Specific for April 2015 data'''
        #print(chunk,type(chunk))
        if type(chunk)==np.int64:
            chunk=[chunk]
        for i,filepath in enumerate(chunk):
            self.single_file(filepath,i)

            
            #### PUT QUEUE
            self.queue.put([i])
            
            #gc.collect()     
         