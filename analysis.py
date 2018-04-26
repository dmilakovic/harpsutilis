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

import harps.settings as settings
import harps.classes as hc

sOrder = settings.sOrder
eOrder = settings.eOrder
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
                                      wavecoeff_air=wavecoeff_airA)
                wavesolB = specB.__get_wavesol__(calibrator='LFC',
                                      anchor_offset=anchor_offset,
                                      wavesol_thar=tharB,
                                      #orders=np.arange(sOrder+1,eOrder-1),
                                      wavecoeff_air=wavecoeff_airB)
                
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


