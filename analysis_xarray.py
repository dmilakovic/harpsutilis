#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:10:20 2018

@author: dmilakov
"""
import numpy as np
import xarray as xr
import os
import h5py

from harps import settings as hs
from harps import classes as hc
from harps import functions as hf
import multiprocessing as mp


sOrder = hs.sOrder
eOrder = hs.eOrder
#sOrder = 42
#eOrder = 72
nOrder = eOrder - sOrder

class Analyser(object):
    def __init__(self,manager,fibre,filelim=None,LFC1=None,LFC2=None):
        self.manager   = manager
        self.filepaths = manager.file_paths[fibre]
        self.nFiles    = len(self.files)
        self.nOrder    = nOrder
        self.fibre     = fibre
        if filelim is not None:
            self.LFC1    = LFC1 if LFC1 is not None else 'HARPS'
            self.LFC2    = LFC2 if LFC2 is not None else 'HARPS'
            self.filelim = filelim
        else:
            self.filelim = 'ZZZZZ'
            self.LFC1    = 'HARPS'
            self.LFC2    = 'HARPS'
        
        self.processes = []
        self.queue     = mp.Queue()
    def start_work(self,nproc=6):
        self.nproc=nproc
        print(nproc)
        chunks = np.array_split(self.files,self.nproc)
        print(chunks)
        for i in range(self.nproc):
            chunk = chunks[i]
            #print(i,chunk)
            p = mp.Process(target=self.work_on_chunk,args=((chunk,)))
            self.processes.append(p)
            #p.daemon=False
            p.start()
            print('Processes started')
        print('Pre-joining')
        for p in self.processes:
            print(p)
            p.join(timeout=2)  
        print('Processes joined')
        for p in self.processes:
            print("Process is alive = ",p.is_alive())
        #print(data_out.empty())
        while self.queue.empty() == True:
            time.sleep(10)
        for i in range(self.nFiles):
            #print('Queue is empty = ',data_out.empty())
            l,rv,wg,cal = self.queue.get()
            print('Queue element {} extracted'.format(i))
            self.data.rvdata.loc[dict(fb='A',ex=l)] = rv.loc[dict(fb='A',ex=l)]
            self.data.rvdata.loc[dict(fb='B',ex=l)] = rv.loc[dict(fb='B',ex=l)]
        
            self.data.weights.loc[dict(fb='A',ex=l)] = wg.loc[dict(fb='A',ex=l)]
            self.data.weights.loc[dict(fb='B',ex=l)] = wg.loc[dict(fb='B',ex=l)]
            
            self.data.wavesol_LFC.loc[dict(fb='A',ex=l)] = cal.loc[dict(fb='A',ex=l)]
            self.data.wavesol_LFC.loc[dict(fb='B',ex=l)] = cal.loc[dict(fb='B',ex=l)]
            #except:
            #    print(i, 'Empty exception raised on element')
    #for i in range(numfiles):
    def work_on_chunk(self,chunk):  
        ''' Specific for April 2015 data'''
        print(chunk,type(chunk))
        if type(chunk)==np.int64:
            chunk=[chunk]
        for i in chunk:
            filepath = self.filepaths[i]
            basename = os.path.basename(filepath)
                if basename < self.filelim:
                    LFC = self.LFC1
                else:
                    LFC = self.LFC2
            if LFC == 'HARPS':
                anchor_offset=-100e6
            print(i,filepath,LFC1)
           
            spec = hc.Spectrum(filepath,data=True,LFC=LFC)
            
      
            spec.__get_wavesol__(calibrator='LFC',patches=True,gaps=False)
            spec.save_lines()
            spec.save_wavesol()
            
            #### PUT QUEUE
            self.queue.put([i])
            
            #gc.collect()     
         