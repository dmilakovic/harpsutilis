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

__version__='0.1.3'

sOrder = hs.sOrder
eOrder = hs.eOrder
default_sOrder = 42
default_eOrder = 72
nOrder = default_eOrder - default_sOrder

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
    def __init__(self,manager,fibre,
                 filelim=None,LFC1=None,LFC2=None,
                 use_reference=False,specnum_reference=0,
                 refspec_path = None,
                 savedir=None,line_dir=None,ws_dir=None,
                 savelines=True,savewavesol=True,
                 patches = True, gaps=False, 
                 polyord = 3,
                 fittype=['gauss','epsf'],
                 sOrder=None,eOrder=None):
        self.sOrder    = sOrder if sOrder is not None else default_sOrder
        self.eOrder    = eOrder if eOrder is not None else default_eOrder
        self.orders    = np.arange(self.sOrder,self.eOrder,1)
        
        
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
        
        if filelim is not None:
            self.LFC1    = LFC1 if LFC1 is not None else 'HARPS'
            self.LFC2    = LFC2 if LFC2 is not None else 'HARPS'
            self.filelim = filelim
        else:
            self.filelim = 'ZZZZZ'
            self.LFC1    = LFC1 if LFC1 is not None else 'HARPS'
            self.LFC2    = LFC2 if LFC2 is not None else 'HARPS'
        
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
            print("Spectral orders:", np.arange(sOrder,eOrder,1))
            
            self.initialize_reference()
        #print(self.filepaths)
        return
    def initialize_reference(self):
        if self.refspec_path is not None:
            spec0 = hc.Spectrum(self.refspec_path,LFC=self.LFC1)
        else: 
            specnum = self.specnum_reference
            spec0 = hc.Spectrum(self.filepaths[specnum],LFC=self.LFC1)
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
        if basename < self.filelim:
            LFC = self.LFC1
        else:
            LFC = self.LFC2
        if LFC == 'HARPS':
            anchor_offset=0#-100e6
            fibre_shape = 'round'
        elif LFC == 'FOCES':
            fibre_shape = 'round'
            anchor_offset=0
        
       
        spec = hc.Spectrum(filepath,LFC=LFC)
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
         