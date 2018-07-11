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
import time 
import sys

import gc

from harps import classes as hc
from harps import settings as hs
from harps import functions as hf
from glob import glob
import multiprocessing as mp

__version__='0.1.2'

sOrder = hs.sOrder
eOrder = hs.eOrder
sOrder = 42
eOrder = 72
nOrder = eOrder - sOrder

class Worker(object):   
    def __init__(self,filename=None,mode=None,manager=None,
                 orders=None):
        self.filename = filename
        self.open_file(self.filename,mode)
        self.manager = manager
        print(self.file)
        eo = self.check_exists("orders")
        if eo == False:
            if not orders:
                orders = np.arange(sOrder,eOrder)
            self.file.create_dataset("orders",data=orders)
        self.manager.get_file_paths('AB')
        
        
        print('Worker initialised')  
    def is_open(self):
        return self.open
    def open_file(self,filename=None,mode=None):
        if not filename:
            filename = self.filename
        e = os.path.isfile(filename)
        if not mode:
            if e:
                mode = "r+"
            else:
                mode = "w"
        print(filename,mode)
        self.file = h5py.File(filename,mode)
        self.open = True
    def dump_to_file(self):
        self.file.flush()
        return
    def close_file(self):
        self.file.close()
        self.open = False
        return
    def check_exists(self,node):
        e = False
        ds = "{}".format(node)
        if ds in self.file:
            e = True
        return e 
    def distortion_node(self,i,fibre='AB'):
        return ["{}/{}/{}".format(i,f,t) for f in list(fibre) 
                                 for t in ['wave','pix','rv']]
    def do_wavelength_stability(self,refA=None,refB=None,tharA=None,tharB=None,
                                LFC_A0='FOCES',LFC_B0='FOCES',filelim=None):
        if not refA:
            return "Reference wavelength solution for fibre A not given"
        if not refB:
            return "Reference wavelength solution for fibre B not given"
            
        o = self.is_open()
        if o == False:
            self.open_file()
        else:
            pass
        
        fileA0 = self.manager.file_paths['A'][0]
        fileB0 = self.manager.file_paths['B'][0]
        specA0 = hc.Spectrum(fileA0,data=True,LFC=LFC_A0)
        specB0 = hc.Spectrum(fileB0,data=True,LFC=LFC_B0)
        tharA  = specA0.__get_wavesol__('ThAr')
        tharB  = specB0.__get_wavesol__('ThAr')
        wavecoeff_airA = specA0.wavecoeff_air
        wavecoeff_airB = specB0.wavecoeff_air
        wavesol_refA  = specA0.__get_wavesol__('LFC')
        wavesol_refB  = specB0.__get_wavesol__('LFC')
        
        numfiles = self.manager.numfiles[0]
        for i in range(numfiles):
            e = self.check_exists("{}".format(i))
            
            nodes = ["{}/{}/{}".format(i,f,t) for f in ["A","B"] 
                     for t in ["wavesol_LFC","rv","weights","lines","coef"]]
            ne = [self.check_exists(node) for node in nodes]
               
            # THIS SECTION IS MEANT TO WORK FOR DATA FROM APRIL 17th ONLY!!
            filelim = {'A':self.manager.file_paths['A'][93], 
                       'B':self.manager.file_paths['B'][93]}
            if ((e == False) or (np.all(ne)==False)):
                fileA = self.manager.file_paths['A'][i]
                if fileA < filelim['A']:
                    LFC1 = 'FOCES'
                    LFC2 = 'FOCES'
                    anchor_offset=0e0
                else:
                    LFC1 = 'FOCES'
                    LFC2 = 'HARPS'
                    anchor_offset=-100e6
                fileB = self.manager.file_paths['B'][i]
                print(i,fileA,LFC1)
                print(i,fileB,LFC2)
                specA = hc.Spectrum(fileA,data=True,LFC=LFC1)
                specB = hc.Spectrum(fileB,data=True,LFC=LFC2)
        
                wavesolA = specA.__get_wavesol__(calibrator='LFC',
                                      wavesol_thar=tharA,
                                      wavecoeff_air=wavecoeff_airA)['epsf']
                wavesolB = specB.__get_wavesol__(calibrator='LFC',
                                      anchor_offset=anchor_offset,
                                      wavesol_thar=tharB,
                                      #orders=np.arange(sOrder+1,eOrder-1),
                                      wavecoeff_air=wavecoeff_airB)['epsf']
                
                rvA      = (wavesolA[sOrder:eOrder] - wavesol_refA)/wavesol_refA * 299792458
                rvB      = (wavesolB[sOrder:eOrder] - wavesol_refB)/wavesol_refB * 299792458
                
                  
                weightsA = specA.get_weights2d()[sOrder:eOrder]
                weightsB = specB.get_weights2d()[sOrder:eOrder]
                
                linesA   = specA.lines.values
                linesB   = specB.lines.values
                
                coefsA   = specA.wavecoef_LFC
                coefsB   = specB.wavecoef_LFC
                
                nodedata = [wavesolA,rvA,weightsA,linesA,coefsA,
                            wavesolB,rvB,weightsB,linesB,coefsB]
                for node,data in zip(nodes,nodedata):
                    node_exists = self.check_exists(node)
                    if node_exists==False:
                        self.file.create_dataset(node,data=data)
                        self.file.flush()
                    else:
                        pass
            
        return  
    def do_distortion_calculation(self,fibre='AB'):
        
        o = self.is_open()
        if o == False:
            self.open_file()
        else:
            pass
        
        
        for fi,f in enumerate(list(fibre)):
            numfiles = self.manager.numfiles[fi]
            for i in range(numfiles):   
                e = self.check_exists("{}".format(i))
                
                nodes = self.distortion_node(i,fibre=f)
                ne = [self.check_exists(node) for node in nodes]
#                print(nodes)
#                print(e, ne)
                if ((e == False) or (np.all(ne)==False)):
                    print("Working on {}/{}".format(i+1,numfiles))
                    spec = hc.Spectrum(self.manager.file_paths[f][i],LFC='HARPS')
                    spec.polyord = 8
                    spec.__get_wavesol__('LFC',gaps=False,patches=False)
                    #spec.plot_distortions(plotter=plotter,kind='lines',show=False)
                    dist = spec.get_distortions()
                    
                    wav  = dist.sel(typ='wave')
                    pix  = dist.sel(typ='pix')
                    rv   = dist.sel(typ='rv')
                    nodedata = [wav,pix,rv]
                    self.save_nodes(nodes,nodedata)
        return
    def read_distortion_file(self,filename=None):
        if filename is None:
            filename = self.filename
        self.open_file()
        
        l    = len(self.file)
        data = xr.DataArray(np.full((l,2,3,nOrder,500),np.nan),
                            dims=['fn','fbr','typ','od','val'],
                            coords = [np.arange(l),
                                      ['A','B'],
                                      ['wave','pix','rv'],
                                      np.arange(sOrder,eOrder),
                                      np.arange(500)])
        for i in range(l):
            nodes = self.distortion_node(i,fibre='AB')
            for node in nodes:
                print(node)
                e = self.check_exists(node)
                if e == True:
                    fn,fbr,typ = node.split('/')
                    fn = int(fn)
                    if fbr == 'A':
                        ods=np.arange(sOrder,eOrder)
                    elif fbr == 'B':
                        ods=np.arange(sOrder,eOrder-1)
                    data.loc[dict(fn=fn,fbr=fbr,typ=typ,od=ods)] = self.file[node][...]
        self.distortion_data = data
        return data  
    def save_nodes(self,nodenames,nodedata):
        for node,data in zip(nodenames,nodedata):
            node_exists = self.check_exists(node)
#            print(node,node_exists)
            if node_exists==False:
                print('Saving node:',node)
                self.file.create_dataset(node,data=data)
                self.file.flush()
            else:
                pass           
        return




class Analyser(object):
    def __init__(self,manager,fibre,filelim=None,LFC1=None,LFC2=None):
        self.manager   = manager
        self.filepaths = manager.file_paths[fibre]
        self.reference = manager.file_paths[fibre][0]
        self.nFiles    = len(self.filepaths)
        self.nOrder    = nOrder
        self.fibre     = fibre
        if filelim is not None:
            self.LFC1    = LFC1 if LFC1 is not None else 'HARPS'
            self.LFC2    = LFC2 if LFC2 is not None else 'HARPS'
            self.filelim = filelim
        else:
            self.filelim = 'ZZZZZ'
            self.LFC1    = LFC1 if LFC1 is not None else 'HARPS'
            self.LFC2    = LFC2 if LFC2 is not None else 'HARPS'
        self.ws_dir   = hs.harps_ws
        self.line_dir = hs.harps_lines
        
        reduced_filelist = self.reduce_filelist()
        self.filepaths    = reduced_filelist
        print("Number of files = {}".format(len(self.filepaths)))
        self.initialize_reference()
    def initialize_reference(self):
        spec0 = hc.Spectrum(self.filepaths[0],LFC=self.LFC1)
        thar0  = spec0.__get_wavesol__('ThAr')
        wavecoeff_air0 = spec0.wavecoeff_air
        wavecoeff_vac0 = spec0.wavecoeff_vacuum
        self.reference = dict(thar0=thar0,
                              wavecoeff_air0=wavecoeff_air0,
                              wavecoeff_vac0=wavecoeff_vac0)
        return self.reference
    def get_reference(self):
        try:
            reference = self.reference
        except:
            reference = self.initialize_reference()
        return reference
    def reduce_filelist(self):
        def get_base(filename):
            basename = os.path.basename(filename)
            return basename[0:36]
        dirname          = os.path.dirname(self.reference)
        all_basenames    = [get_base(file) for file in [os.path.basename(path) for path in self.filepaths]]
        existing_lines   = [get_base(file) for file in glob(os.path.join(self.line_dir,'*lines.nc'))]
        existing_ws      = [get_base(file) for file in glob(os.path.join(self.ws_dir,'*ws.nc'))]
        
        diff_lines       = np.setdiff1d(all_basenames,existing_lines)
        diff_ws          = np.setdiff1d(all_basenames,existing_ws)
        
        diff             = np.union1d(diff_lines,diff_ws)
        reduced_filelist = [os.path.join(dirname,basename+'.fits') for basename in diff]
        return reduced_filelist
    def start_multiprocess(self,nproc=hs.nproc):
        self.nproc=nproc
        self.processes = []
        self.queue     = mp.Queue()
        print("Number of processes : ",nproc)
        chunks = np.array_split(self.filepaths,self.nproc)
        
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
            elem = self.queue.get()
            print('Queue element {} extracted'.format(elem))
#            self.data.rvdata.loc[dict(fb='A',ex=l)] = rv.loc[dict(fb='A',ex=l)]
#            self.data.rvdata.loc[dict(fb='B',ex=l)] = rv.loc[dict(fb='B',ex=l)]
#        
#            self.data.weights.loc[dict(fb='A',ex=l)] = wg.loc[dict(fb='A',ex=l)]
#            self.data.weights.loc[dict(fb='B',ex=l)] = wg.loc[dict(fb='B',ex=l)]
#            
#            self.data.wavesol_LFC.loc[dict(fb='A',ex=l)] = cal.loc[dict(fb='A',ex=l)]
#            self.data.wavesol_LFC.loc[dict(fb='B',ex=l)] = cal.loc[dict(fb='B',ex=l)]
            #except:
            #    print(i, 'Empty exception raised on element')
    #for i in range(numfiles):
    def single_file(self,filepath,i):
        reference = self.get_reference()
        basename = os.path.basename(filepath)
        if basename < self.filelim:
            LFC = self.LFC1
        else:
            LFC = self.LFC2
        if LFC == 'HARPS':
            anchor_offset=-100e6
            fibre_shape = 'round'
        elif LFC == 'FOCES':
            fibre_shape = 'round'
            anchor_offset=0
        
       
        spec = hc.Spectrum(filepath,LFC=LFC)
        spec.fibre_shape = fibre_shape
        lines = spec.load_lines()
        ws    = spec.load_wavesol()
        if lines is None or ws is None:
            del(spec)
            spec = hc.Spectrum(filepath,LFC=LFC)
            spec.fibre_shape = fibre_shape
            print("{0:>4d}{1:>50s}{2:>8s}{3:>10s}".format(i,basename,LFC,'working'))
            spec.wavecoeff_air    = reference['wavecoeff_air0']
            spec.wavecoeff_vacuum = reference['wavecoeff_vac0']
            spec.wavesol_thar     = reference['thar0']
            print('Patches = ',spec.patches)
            spec.detect_lines()
            
            spec.__get_wavesol__(calibrator='LFC',patches=True,gaps=False,
                                 anchor_offset=anchor_offset)
            spec.save_lines()
            spec.save_wavesol()
        else:
            print("{0:>4d}{1:>50s}{2:>8s}{3:>10s}".format(i,basename,LFC,'exists'))
        del(spec)
        del(lines)
        del(ws)
        gc.collect()
    def start_singleprocess(self):
        reference = self.get_reference()
        for i,filepath in enumerate(self.filepaths):
            basename = os.path.basename(filepath)
            if basename < self.filelim:
                LFC = self.LFC1
            else:
                LFC = self.LFC2
            if LFC == 'HARPS':
                anchor_offset=-100e6
                fibre_shape = 'round'
            elif LFC == 'FOCES':
                fibre_shape = 'round'
                anchor_offset=0
            
           
            spec = hc.Spectrum(filepath,LFC=LFC)
            spec.fibre_shape = fibre_shape
            lines = spec.load_lines()
            ws    = spec.load_wavesol()
            if lines is None or ws is None:
                print("{0:>4d}{1:>50s}{2:>8s}{3:>10s}".format(i,basename,LFC,'working'))
                spec.wavecoeff_air    = reference['wavecoeff_air0']
                spec.wavecoeff_vacuum = reference['wavecoeff_vac0']
                spec.wavesol_thar     = reference['thar0']
                print('Patches = ',spec.patches)
                spec.detect_lines()
                
                spec.__get_wavesol__(calibrator='LFC',patches=True,gaps=False,
                                     anchor_offset=anchor_offset)
                spec.save_lines()
                spec.save_wavesol()
            else:
                print("{0:>4d}{1:>50s}{2:>8s}{3:>10s}".format(i,basename,LFC,'exists'))
            del(spec)
            del(lines)
            del(ws)
            gc.collect()
    def work_on_chunk(self,chunk):  
        ''' Specific for April 2015 data'''
        #print(chunk,type(chunk))
        if type(chunk)==np.int64:
            chunk=[chunk]
        for i,filepath in enumerate(chunk):
            self.single_file(filepath,i)
#            basename = os.path.basename(filepath)
#            if basename < self.filelim:
#                LFC = self.LFC1
#            else:
#                LFC = self.LFC2
#            if LFC == 'HARPS':
#                anchor_offset=-100e6
#                fibre_shape = 'round'
#            elif LFC == 'FOCES':
#                fibre_shape = 'round'
#                anchor_offset=0
#            print(i,filepath,LFC)
#           
#            spec = hc.Spectrum(filepath,LFC=LFC)
#            spec.fibre_shape = fibre_shape
#            print(fibre_shape)
#            spec.__get_wavesol__(calibrator='LFC',patches=True,gaps=False,
#                                 anchor_offset=anchor_offset)
#            spec.save_lines()
#            spec.save_wavesol()
            
            #### PUT QUEUE
            self.queue.put([i])
            
            #gc.collect()     
         