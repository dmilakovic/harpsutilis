#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:29:16 2018

@author: dmilakov
"""

from harps.classes import Spectrum
import harps.settings as hs
import harps.functions as hf

import numpy  as np
import xarray as xr
import pandas as pd
#mport matplotlib.pyplot as plt

import os
import datetime
from glob import glob
import urllib


###############################################################################
##############################   MANAGER   ####################################
###############################################################################
class Manager(object):
    '''
    Manager is able to find the files on local drive, read the selected data, 
    and perform bulk operations on the data.
    '''
    def __init__(self,
                 date=None,year=None,month=None,day=None,
                 begin=None,end=None,
                 run=None,sequence=None,
                 dirpath=None,
                 filelist=None,
                 get_file_paths=True):
        '''
        Ways to initialize object:
            (1) Provide a single date
            (2) Provide a begin and end date
            (3) Provide a run and sequence ID
            (4) Provide a path to a directory
            (5) Provide a path to a list of files to use
        date(yyyy-mm-dd)
        begin(yyyy-mm-dd)
        end(yyyy-mm-dd)
        sequence(day,sequence)
        '''
        def get_init_method():
            if date!=None or (year!=None and month!=None and day!=None):
                method = 1
            elif begin!=None and end!=None:
                method = 2
            elif run!=None and sequence!=None:
                method = 3
            elif dirpath!=None:
                method = 4
            elif filelist!=None:
                method = 5
            return method
        
        baseurl     = 'http://people.sc.eso.org/%7Eglocurto/COMB/'
        
        self.file_paths = []
        self.spectra    = []
        #harpsDataFolder = os.path.join("/Volumes/home/dmilakov/harps","data")
        harpsDataFolder = hs.harps_data#os.path.join("/Volumes/home/dmilakov/harps","data")
        self.harpsdir   = harpsDataFolder
        self.datadir_list = []
        
        method = get_init_method()
#        print(method)
        self.method = method

        if method==1:
            self.sequence_list_filepath = None
            if   date==None and (year!=None and month!=None and day!=None):
                self.dates = ["{y:4d}-{m:02d}-{d:02d}".format(y=year,m=month,d=day)]
                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
            elif date!=None:
                self.dates = [date]
                
                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
                print(self.datadir)
            elif date==None and (year==None or month==None or day==None) and (begin==None or end==None):
                raise ValueError("Invalid date input. Expected format is 'yyyy-mm-dd'.")
        elif method==2:
            
            by,bm,bd       = tuple(int(val) for val in begin.split('-'))
            ey,em,ed       = tuple(int(val) for val in end.split('-'))
            print(by,bm,bd)
            print(ey,em,ed)
            self.begindate = datetime.datetime.strptime(begin, "%Y-%m-%d")
            self.enddate   = datetime.datetime.strptime(end, "%Y-%m-%d")
            self.dates     = []
            def daterange(start_date, end_date):
                for n in range(int ((end_date - start_date).days)):
                    yield start_date + datetime.timedelta(n)
            for single_date in daterange(self.begindate, self.enddate):
                self.dates.append(single_date.strftime("%Y-%m-%d"))
                
            
            
            for date in self.dates:
                #print(date)
                datadir = os.path.join(harpsDataFolder,date)
                if os.path.isdir(datadir):
                    self.datadir_list.append(datadir)
        if method==3:
            run = run if run is not None else ValueError("No run selected")
            
            if type(sequence)==tuple:
                sequence_list_filepath = baseurl+'COMB_{}/day{}_seq{}.list'.format(run,*sequence)
                print(sequence_list_filepath)
                self.sequence_list_filepath = [sequence_list_filepath]
                self.sequence = [sequence]
            elif type(sequence)==list:
                self.sequence_list_filepath = []
                self.sequence = sequence
                for item in sequence:
                    sequence_list_filepath = baseurl+'COMB_{}/day{}_seq{}.list'.format(run,*item)
                    self.sequence_list_filepath.append(sequence_list_filepath)
                    
        elif method == 4:
            if type(dirpath)==str: 
                self.datadir_list.append(dirpath)
            elif type(dirpath)==list:
                for d in dirpath:
                    self.datadir_list.append(d)
        elif method == 5:
            path_list = pd.read_csv(filelist,comment='#',names=['file'])
            absolute = np.all([os.path.exists(filepath) for filepath in path_list.file])
            if not absolute:
                pass
                
#        else:
#            self.sequence_list_filepath = None
#            if   date==None and (year!=None and month!=None and day!=None) and (begin==None or end==None):
#                self.dates = ["{y:4d}-{m:02d}-{d:02d}".format(y=year,m=month,d=day)]
#                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
#            elif date!=None:
#                self.dates = [date]
#                
#                self.datadir = os.path.join(harpsDataFolder,self.dates[0])
#                print(self.datadir)
#            elif date==None and (year==None or month==None or day==None) and (begin==None or end==None):
#                raise ValueError("Invalid date input. Expected format is 'yyyy-mm-dd'.")
#            elif (begin!=None and end!=None):
#                by,bm,bd       = tuple(int(val) for val in begin.split('-'))
#                ey,em,ed       = tuple(int(val) for val in end.split('-'))
#                print(by,bm,bd)
#                print(ey,em,ed)
#                self.begindate = datetime.datetime.strptime(begin, "%Y-%m-%d")
#                self.enddate   = datetime.datetime.strptime(end, "%Y-%m-%d")
#                self.dates     = []
#                def daterange(start_date, end_date):
#                    for n in range(int ((end_date - start_date).days)):
#                        yield start_date + datetime.timedelta(n)
#                for single_date in daterange(self.begindate, self.enddate):
#                    self.dates.append(single_date.strftime("%Y-%m-%d"))
#                
#            self.datadirlist  = []
#            
#            for date in self.dates:
#                #print(date)
#                datadir = os.path.join(harpsDataFolder,date)
#                if os.path.isdir(datadir):
#                    self.datadirlist.append(datadir)
        self.orders = np.arange(hs.sOrder,hs.eOrder,1)
        if get_file_paths==True and len(self.file_paths)==0 :
            self.get_file_paths(fibre='AB')
    def _check_dtype(self,dtype):
        if dtype in ['lines','LFCws','series']:
            return dtype
        else:
            raise UserWarning('Data type unknown')
    def check_data(self,dtype=None):
        dtype = self._check_dtype(dtype)
        return hasattr(self,dtype)
    def get_data(self,dtype=None,*args):
        dtype_exists = self.check_data(dtype)
        if dtype_exists:
            return getattr(self,dtype) 
        else:
            return self.read_data(dtype,*args)
    def get_file_paths(self, fibre, ftype='e2ds',**kwargs):
        '''
        Function to find fits files of input type and input date in the 
        $HARPS environment variable.
        
        INPUT:
        ------
        date  = "yyyy-mm-dd" - string
        type  = ("e2ds", "s1d") - string
        fibre = ("A", "B", "AB") - string
        
        Additional arguments:
        condition = ("filename")
            "filename": requires additional arguments "first" and "last" 
            (filename)
        OUTPUT:
        ------
        fileList = list of fits file paths in the HARPS data folder for a 
                   specific date
        '''
        self.fibre = fibre
        self.ftype = ftype
        #if os.path.isdir(self.datadir)==False: raise OSError("Folder not found")
        filePaths        = {}
        
        if self.method==3:  
            # Run ID and sequence provided
            print(self.sequence_list_filepath)
            if type(self.sequence_list_filepath)==list:    
                # list to save paths to files on disk
                sequence_list = []
                
                for item,seq in zip(self.sequence_list_filepath,self.sequence):
                    # read files in the sequence from the internet
                    req = urllib.request.Request(item)
                    res = urllib.request.urlopen(req)
                    htmlBytes = res.read()
                    htmlStr   = htmlBytes.decode('utf8').split('\n')
                    filenamelist  = htmlStr[:-1]
                    # append filenames to a list
                    for filename in filenamelist:
                        sequence_list.append([seq,filename[0:29]])
                    # use the list to construct filepaths to files on disk
                    for fbr in list(fibre):
                        fitsfilepath_list = []
                        for seq,item in sequence_list:
                            date    = item.split('.')[1][0:10]
                            datadir = os.path.join("{date}".format(date=item.split('.')[1][0:10]),
                                                   "series {n:02d}".format(n=seq[1]))
                            time    = item.split('T')[1].split(':')
                            fitsfilepath = os.path.join(self.harpsdir,datadir,
                                "HARPS.{date}T{h}_{m}_{s}_{ft}_{f}.fits".format(date=date,h=time[0],m=time[1],s=time[2],ft=ftype,f=fbr))
                            fitsfilepath_list.append(fitsfilepath)
                            #print(date,time,fitsfilepath,os.path.isfile(fitsfilepath))
                        filePaths[fbr] = np.sort(fitsfilepath_list)
        elif self.method==5:
            pass
        else:
            for fbr in list(fibre):
                nestedlist = []
                for datadir in self.datadir_list:
                    try:
                        files_in_dir=np.array(glob(os.path.join(datadir,"*{ftp}*{fbr}.fits".format(ftp=ftype,fbr=fbr))))
                    except:
                        raise ValueError("No files of this type were found")
                    if "condition" in kwargs.keys():
                        self.condition = {"condition":kwargs["condition"]}
                        if kwargs["condition"] == "filename":
                            self.condition["first"] = kwargs["first"]
                            self.condition["last"]  = kwargs["last"]
                            ff = np.where(files_in_dir==os.path.join(datadir,
                                    "{base}_{ftp}_{fbr}.fits".format(base=kwargs["first"],ftp=self.ftype,fbr=fbr)))[0][0]
                            lf = np.where(files_in_dir==os.path.join(datadir,
                                    "{base}_{ftp}_{fbr}.fits".format(base=kwargs["last"],ftp=self.ftype,fbr=fbr)))[0][0]
                            selection = files_in_dir[ff:lf]
                            nestedlist.append(selection)
                    else:
                        nestedlist.append(files_in_dir)
                flatlist       = [item for sublist in nestedlist for item in sublist]   
                filePaths[fbr] = np.sort(flatlist)
              
        
        self.file_paths = filePaths
        basenames       = [[os.path.basename(file)[:-5] for file in filePaths[f]] for f in list(fibre)]
        self.basenames  = dict(zip(list(fibre),basenames))
        datetimes      = [[np.datetime64(bn.split('.')[1].replace('_',':')) for bn in self.basenames[fbr]] for fbr in list(fibre)]
        self.datetimes = dict(zip(list(fibre),datetimes))
        self.numfiles = [np.size(filePaths[fbr]) for fbr in list(fibre)]
        if np.sum(self.numfiles)==0:
            raise UserWarning("No files found in the specified location")
        return 
    def get_spectra(self, fibre, ftype='e2ds', header=False,data=False):
        # DOESN'T WORK PROPERLY!! DO NOT USE
        '''
        Function to get a list of Spectrum class objects for manipulation
        '''
        if not self.file_paths:
            print("Fetching file paths")
            self.get_file_paths(fibre=fibre, ftype=ftype)
        else:
            pass
        spectra = {}
        for fbr in list(fibre):
            fbr_list    = self.file_paths[fbr]
            fbr_spectra = []
            for path in fbr_list:
                spectrum = Spectrum(filepath=path,ftype=ftype,header=header,data=data)
                fbr_spectra.append(spectrum)
            spectra[fbr] = fbr_spectra
        self.spectra = spectra
        return self.spectra
    def read_data(self,dtype=None,fibre='AB',dirname=None,autoclose=True,
                  engine=None):
        ''' Reads lines and wavelength solutions of the spectra '''
        # Check that dtype is recognised
        dtype = self._check_dtype(dtype)
        # If already exists, return data
        data_exists = self.check_data(dtype)
        if data_exists:
            return getattr(self,dtype)
        # Otherwise, read the data from files
        else:  
            # make the fibre argument a list
            fibres = hf.to_list(fibre)
            if len(self.file_paths) == 0:
                self.get_file_paths(fibre=fibre)
            else:
                pass
            # get dirnames
            if dirname is not None:
                dirnames = dirname
            else:
                if dtype == 'lines':
                    dirnames = dict(zip(fibres,[hs.harps_lines for fbr in fibres]))
                elif dtype == 'LFCws':
                    dirnames = dict(zip(fibres,[hs.harps_ws for fbr in fibres]))
                else:
                    raise UserWarning("Uknown")
                    
    #        dirnames = dirname if dirname is not None else dict(zip(fibres,[hs.harps_lines for fbr in fibres]))
            if type(dirname)==dict:
                pass
            elif type(dirname)==str:
                dirnames = dict(zip(fibres,[dirname for fbr in fibres]))
            
            # get basename
            basenames = self.basenames
            
            filenames = [[os.path.join(dirnames[fbr],b+'_{}.nc'.format(dtype)) \
                                       for b in basenames[fbr]] for fbr in fibres]
            filelist = dict(zip(fibres,filenames))
            ll = []
            #print(self.orders)
            for fbr in fibres:
                idx   = pd.Index(self.datetimes[fbr],name='time')
                data_fibre = xr.open_mfdataset(filelist[fbr],
                                               concat_dim=idx,
                                               engine=engine,
                                               autoclose=autoclose)
                #print(data_fibre.coords['od'])
                data_fibre = data_fibre.sel(od=self.orders)
                data_fibre = data_fibre.sortby('time')
                data_fibre.expand_dims('fb')
                ll.append(data_fibre)
#            if len(fibres)>1:
#                data = xr.concat(ll,dim=pd.Index(['A','B'],name='fb'))
#            else:
#                data = ll[0]
            data = xr.concat(ll,dim=pd.Index(fibres,name='fb'))
            #print(data)
        return data
    def read_lines(self,fibre='AB',dirname=None,**kwargs):
        self.lines = self.read_data(dtype='lines',fibre=fibre,
                                    dirname=dirname,**kwargs)
        return self.lines
    def read_wavesol(self,fibre='AB',dirname=None,**kwargs):
        self.LFCws = self.read_data(dtype='LFCws',fibre=fibre,
                                    dirname=dirname,**kwargs)
        return self.LFCws
#    def read_series(self,fibre='AB',dirname=None,**kwargs):
#        self.LFCws = self.read_data(dtype='series',fibre=fibre,
#                                    dirname=dirname,**kwargs)
#        return self.LFCws
    def save_data(self,basename,dtype=None,dirname=None,engine=None):
        # Check that dtype is recognised
        dtype = self._check_dtype(dtype)
        # If already exists, return data
        data_exists = self.check_data(dtype)
        if not data_exists:
            raise UserWarning("Manager doesn't have {} data".format(dtype))
        else:
            data = self.get_data(dtype)
            # make file path
            # get directory path and create it if non-existent
            dirname  = dirname if dirname is not None else hs.harps_combined
            if not os.path.exists(dirname):
                os.makedirs(dirname,exist_ok=True)
            # get file basename
            basename = "{0}_{1}.nc".format(basename,dtype)
            filename = os.path.join(dirname,basename)
            
            print(data)
            try:
                data.to_netcdf(filename,engine='netcdf4')
                print("Dataset '{}' "
                      "successfully saved to {}".format(dtype,filename))
            except:
                print("Dataset {} could not be saved.")
        return
        

        
    
    

#    def get_spectrum(self,ftype,fibre,header=False,data=False):
#        return 0
#    def read_data(self, filename="datacube", **kwargs):
#        try:    
#            fibre       = kwargs["fibre"]
#            self.fibre  = fibre
#        except: fibre   = self.fibre
#        try:    
#            orders      = kwargs["orders"]
#            self.orders = orders
#        except: orders  = self.orders
#        # if there are conditions, change the filename to reflect the conditions
#        try:
#            for key,val in self.condition.items():
#                filename = filename+"_{key}={val}".format(key=key,val=val)
#        except:
#            pass
#        if not self.datadirlist:
#            self.get_file_paths(fibre=fibre)
#        # CREATE DATACUBE IF IT DOES NOT EXIST
#        if   len(self.dates)==1:
#            self.datafilepath = os.path.join(self.datadirlist[0],
#                                         "{name}_{fibre}.npy".format(name=filename, fibre=fibre))
#        elif len(self.dates)>1:
#            self.datafilepath = os.path.join(self.harpsdir,
#                                         "{name}_{fibre}_{begin}_{end}.npy".format(name=filename, fibre=fibre,
#                                             begin=self.begindate.strftime("%Y-%m-%d"), end=self.enddate.strftime("%Y-%m-%d")))
#        #self.datafilepath = os.path.join(self.harpsdir,"2015-04-18/datacube_condition=filename_first=HARPS.2015-04-18T01_35_46.748_last=HARPS.2015-04-18T13_40_42.580_AB.npy")
#        if os.path.isfile(self.datafilepath)==False:
#            #sys.exit()
#            print("Data at {date} is not prepared. Processing...".format(date=self.dates))
#            self.reduce_data(fibre=fibre, filename=filename)
#        else:
#            pass        
#        
#        datainfile  = np.load(self.datafilepath)
#        self.dtype  = [datainfile.dtype.fields[f][0].names for f in list(fibre)][0]
#        self.nfiles = [np.shape(datainfile[f]["FLX"])[1] for f in list(fibre)]
#        if kwargs["orders"]:
#            col         = hf.select_orders(orders)
#            subdtype  = Datatypes(nOrder=len(orders),nFiles=self.nfiles[0],fibre=fibre).specdata(add_corr=True)
#            data        = np.empty(shape=datainfile.shape, dtype=subdtype.data)
#            for f in list(fibre):
#                for dt in self.dtype:
#                    data[f][dt] = datainfile[f][dt][:,:,col]
#            
#        else:
#            data    = datainfile
#        self.data   = data
#        
#        
#        return
#    def reduce_data(self, fibre, ftype='e2ds', filename="datacube", **kwargs):
#        ''' Subroutine which prepares data for easier handling. 
#        Subroutine reads all spectra contained in Manager.file_paths and extracts detected counts for all orders (FLX), fits the 
#        envelope (ENV) and background (BKG) with a cubic spline, calculates the background-to-envelope ratio (B2R). The subroutine 
#        removes the background from original detected signal (FMB). This data is finally saved into a numpy pickled file.'''        
#        if np.size(self.file_paths)>0:
#            pass
#        else:
#            self.get_file_paths(fibre=fibre, ftype=ftype, **kwargs)
#        
#        # SOME PARAMETERS 
#        #nPix    = 4096               # number of pixels in image
#        #sOrder  = 40                # first order in image
#        #eOrder  = 72                # last order in image
#        #nOrder  = eOrder-sOrder     # number of orders in image
#        
#        #if   len(self.dates)==1:
#        #    self.datafilepath = os.path.join(self.datadirlist[0],
#        #                                 "{name}_{fibre}.npy".format(name=filename, fibre=fibre))
#        #elif len(self.dates)>1:
#        #    self.datafilepath = os.path.join(self.harpsdir,
#        #                                 "{name}_{fibre}_{begin}_{end}.npy".format(name=filename, fibre=fibre, 
#        #                                    begin=self.begindate.strftime("%Y-%m-%d"), end=self.enddate.strftime("%Y-%m-%d")))
#        fibres  = list(fibre)
#        #nFibres = len(fibres)
#        nFiles  = len(self.file_paths[fibres[0]])
#        print("Found {} files".format(nFiles))
#
#        data    = np.zeros((hs.nPix,),dtype=Datatypes(nFiles=nFiles,nOrder=nOrder,fibre=fibre).specdata(add_corr=True).data)
#        #print(np.shape(data))
#        
#        for f in fibres:
#            nFiles = np.size(self.file_paths[f])
#            for e in range(nFiles):
#                spec = Spectrum(filepath=self.file_paths[f][e],ftype='e2ds',header=True)
#                print(self.file_paths[f][e])
#                for order in range(sOrder,eOrder-1,1):
#                    o = order-sOrder
#                    envelope   = spec.get_envelope1d(order=order,scale='pixel',kind='spline')
#                    background = spec.get_background1d(order=order,scale='pixel',kind='spline')
#                    b2eRatio   = (background / envelope)
#                    #print(np.shape(envelope),np.shape(background),np.shape(b2eRatio), np.shape(data))
#                    #print(f,e,o, np.shape(envelope), np.shape(data[f]["ENV"]))
#                    data[f]["FLX"][:,e,o] = spec.extract1d(order=order)['flux']
#                    data[f]["ENV"][:,e,o] = envelope
#                    data[f]["BKG"][:,e,o] = background
#                    data[f]["B2E"][:,e,o] = b2eRatio
#                    data[f]["FMB"][:,e,o] = data[f]["FLX"][:,e,o] - background
#                    del(envelope); del(background); del(b2eRatio)
#                    gc.collect()
#                del(spec)
#        # SAVE TO FILE
#        
#        np.save(self.datafilepath,data)
#        print("Data saved to {0}".format(self.datafilepath))
#        return
#    def select_file_subset(self,condition,**kwargs):
#        '''
#        Select only those files which fulfill the condition. 
#        Condition keyword options:
#            filename: selects files between the two file (base)names. Requires additional keywords "first" and "last".
#
#        Returns a new Manager class object, with only selected filenames
#        '''
#        if not self.file_paths:
#            self.get_file_paths(self.fibre)
#        selection = {}
#        if condition == "filename":
#            for f in list(self.fibre):
#                filenames = np.array(self.file_paths[f])
#                print(os.path.join(self.datadir,
#                                "{base}_{ft}_{f}.fits".format(base=kwargs["first"],ft=self.ftype,f=f)))
#                ff = np.where(filenames==os.path.join(self.datadir,
#                                "{base}_{ft}_{f}.fits".format(base=kwargs["first"],ft=self.ftype,f=f)))[0][0]
#                lf = np.where(filenames==os.path.join(self.datadir,
#                                "{base}_{ft}_{f}.fits".format(base=kwargs["last"],ft=self.ftype,f=f)))[0][0]
#                print(ff,lf)
#                selection[f] = list(filenames[ff:lf])
#        newManager = Manager(date=self.dates[0])
#        newManager.fibre      = self.fibre
#        newManager.ftype      = self.ftype
#        newManager.file_paths = selection
#        return newManager
#    def calculate_medians(self,use="data",**kwargs):
#        '''
#        This subroutine calculates the medians for user-selected datatypes and orders, or for all data handled by the manager.
#        The operations are done using data in Manager.data and a new attribute, Manager.mediandata, is created by this subroutine.
#        '''
#        try:    fibre  = kwargs["fibre"]
#        except: fibre  = self.fibre
#        try:    dtype  = kwargs["dtype"]
#        except: dtype  = self.dtype
#        try:    orders = kwargs["orders"]
#        except: 
#            try: orders = self.orders
#            except: print("Keyword 'orders' not specified")
#        try:    errors = kwargs["errors"]
#        except: errors = False
#        
#        datatypes  = Datatypes().specdata(self.nfiles[0],nOrder=len(orders),fibre=fibre, add_corr=True)
#        if   use == "data":
#            dtuse = datatypes.median
#            data = self.data
#        elif use == "fourier":
#            dtuse = datatypes.ftmedian
#            try:
#                data = self.ftdata
#            except:
#                self.calculate_fft(**kwargs)
#                data = self.datafft
#        # if 'error' keyword is true, calculate 16th,50th,84th percentile
#        data50p = np.empty(shape=data.shape, dtype=dtuse)
#        if   errors == True:
#            q = [50,16,84]
#            data16p = np.empty(shape=data.shape, dtype=dtuse)
#            data84p = np.empty(shape=data.shape, dtype=dtuse)
#        # else, calculate only the 50th percentile (median)
#        else:
#            q = [50]
#        # for compatibility with other parts of the code, it is necessary to reshape the arrays (only needed if using a single order)
#        # reshaping from (npix,) -> (npix,1)
#        if np.size(orders)==1:
#            data50p=data50p[:,np.newaxis]
#            if errors == True:
#                data16p = data16p[:,np.newaxis]
#                data84p = data84p[:,np.newaxis]
#
#        # make a selection on orders
#        #col = hf.select_orders(orders)
#        # now use the selection to define a new object which contains median data
#        
#        for f in list(fibre):
#            for dt in dtype:
#                #print(f,dt,data[f][dt].shape)
#                subdata = data[f][dt]#[:,:,col]
#                auxdata = np.nanpercentile(subdata,q=q,axis=1)
#                print(auxdata.shape)
#                if   errors == True:
#                    data50p[f][dt] = auxdata[0]
#                    data16p[f][dt] = auxdata[1]
#                    data84p[f][dt] = auxdata[2]
#                elif errors == False:
#                    data50p[f][dt] = auxdata
#        if   use == "data":
#            self.data50p = data50p
#            if errors == True:
#                self.data84p = data84p
#                self.data16p = data16p
#        elif use == "fourier":
#            self.datafft50p = data50p
#            if errors == True:
#                self.datafft84p = data84p
#                self.datafft16p = data16p
#        return
#    def calculate_fft(self,**kwargs):
#        try:    fibre  = kwargs["fibre"]
#        except: fibre  = self.fibre
#        try:    dtype  = kwargs["dtype"]
#        except: dtype  = self.dtype
#        #try:    orders = kwargs["orders"]
#        #except: 
#        #    try: 
#        orders = self.orders
#        #    except: print("Keyword 'orders' not specified")
#        ############### FREQUENCIES ###############
#        n       = (2**2)*4096
#        freq    = np.fft.rfftfreq(n=n, d=1)
#        uppix   = 1./freq
#        # we only want to use periods lower that 4096 pixels (as there's no sense to use more)
#        cut     = np.where(uppix<=hs.nPix)
#        # prepare object for data input
#        datatypes = Datatypes(nFiles=self.nfiles[0],nOrder=np.size(orders),fibre=fibre).specdata(add_corr=True)
#        datafft   = np.zeros(shape=uppix.shape, dtype=datatypes.ftdata)
#        for f in list(fibre):
#            for dt in dtype: 
#                subdata = self.data[f][dt]
#                #print(f,dt,np.shape(subdata))
#                for i,o in enumerate(orders):
#                    for e in range(subdata.shape[1]):
#                        datafft[f][dt][:,e,i] = np.fft.rfft(subdata[:,e,i],n=n)
#        self.datafft = datafft[cut]
#        self.freq    = uppix[cut]
#        return