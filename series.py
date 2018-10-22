#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:34:16 2018

@author: dmilakov
"""

###############################################################################
###############################    SERIES  ####################################
###############################################################################
class Series(object):
    
    def __init__(self,settingsfile,initiate_manager=True):
        with open(settingsfile) as setup_file:
            settings = json.load(setup_file)
        for key,val in settings.items():
            setattr(self,key,val)
        # basename of the settings folder:
        setup_basename = os.path.basename(settingsfile)
        setup_noext    = setup_basename.split('.')[0]
        self.setupfile_noext = setup_noext
        # path to the spectra used for calculation
        #dirpath = settings['dirpath']
        
        # path to the directories with 'lines' and 'LFCws' files
        topdirname  = os.path.basename(settingsfile).split('.')[0]
        topdirpath  = os.path.join(settings['savedir'],topdirname)
        savedirpaths    = {}
        for dirname in ['lines','LFCws','series']:
            savedirpath = os.path.join(topdirpath,dirname)
            savedirpaths[dirname] = savedirpath
        self.savedirs = savedirpaths
        # use default orders?
        try:
            sOrder = settings['sOrder']
        except:
            sOrder = hs.sOrder
        try:
            # eOrder+1 is necessary to cover the specified eOrder
            eOrder = settings['eOrder'] + 1
        except:
            eOrder = hs.eOrder
        self.sOrder=sOrder
        self.eOrder=eOrder
        
        # manager associated to the series
        if initiate_manager:
            manager = Manager(dirpath=self.dirpath)
            self.manager = manager
        
        # processing stage
        # 0 - just initiated
        # 1 - ran calc_lambda
        # 2 - ran calc_rv
        self.stage = 0
        return
    def _check_dtype(self,dtype):
        if dtype in ['lines','LFCws','series']:
            return dtype
        else:
            raise UserWarning('Data type unknown')
    def _get_index(self,centers):
        ''' Input: dataarray with fitted positions of the lines
            Output: 1d array with indices that uniquely identify every line'''
        fac = 10000
        MOD = 2.
        od = centers.od.values[:,np.newaxis]*fac
        centers_round = np.rint(centers.values/MOD)*MOD
        centers_nonan = np.nan_to_num(centers_round)
        ce = np.asarray(centers_nonan,dtype=np.int)
        index0=np.ravel(od+ce)
        mask = np.where(index0%fac==0)
        index0[mask]=999999999
        return index0
    def _get_sorted(self,index1,index2):
        print('len indexes',len(index1),len(index2))
        # lines that are common for both spectra
        intersect=np.intersect1d(index1,index2)
        intersect=intersect[intersect>0]
    
        indsort=np.argsort(intersect)
        
        argsort1=np.argsort(index1)
        argsort2=np.argsort(index2)
        
        sort1 =np.searchsorted(index1[argsort1],intersect)
        sort2 =np.searchsorted(index2[argsort2],intersect)
        
        return argsort1[sort1],argsort2[sort2]
    
    def check_data(self):
        return hasattr(self,'data')
    def check_stage(self):
        return self.stage
    def read_data(self,filepath=None):
        if self.check_data==True:
            return getattr(self,'data')
        if filepath is not None:
            filepath = filepath  
        else:
            filepath = glob(os.path.join(self.savedirs['series'],
                            '{}_series.nc'.format(self.setupfile_noext)))
            if len(filepath)==1:
                filepath = filepath[0]
            elif len(filepath)==0:
                return None
        data = xr.open_dataset(filepath)
        if self.sOrder is None:
            self.sOrder = np.min(data.coords['od'])
        if self.eOrder is None:
            self.eOrder = np.min(data.coords['od'])
        self.data = data
        return data
    def save_data(self,basename=None,dirname=None):
        # Check that dtype is recognised
        # If already exists, return data
        data_exists = self.check_data()
        if not data_exists:
            raise UserWarning("Series has no data to save".format())
        else:
            data = self.data
            # make file path
            # get directory path and create it if non-existent
            if dirname is not None:
                dirname  = dirname
            else:
                dirname = self.savedirs['series']
            if not os.path.exists(dirname):
                os.makedirs(dirname,exist_ok=True)
            # get file basename
            if basename is not None:
                basename=basename
            else:
                basename = "{0}_{1}.nc".format(self.setupfile_noext,'series')
            filename = os.path.join(dirname,basename)
            
            try:
                data.to_netcdf(filename,engine='netcdf4')
                print("Dataset 'series' "
                      "successfully saved to {}".format(filename))
            except:
                print("Dataset {} could not be saved.")
        return
    def read_dtype_data(self,dtype,dirname=None,engine='netcdf4'):

        # Check that dtype is recognised
        dtype = self._check_dtype(dtype)
        # If already exists, return data
        if hasattr(self,dtype):
            return getattr(self,dtype)
        # Otherwise, read the data from files
        else:  
            # make the fibre argument a list
            fibres = hf.to_list(self.fibre)
            # get dirnames
            if dirname is not None:
                dirnames = dirname
            else:
                dirnames = dict(zip(fibres,[self.savedirs[dtype] for fbr in fibres]))
                
                    
            if type(dirname)==dict:
                pass
            elif type(dirname)==str:
                dirnames = dict(zip(fibres,[dirname for fbr in fibres]))
            
            ll = []
            for fbr in fibres:
                filenames = hf.return_filelist(dirnames[fbr],dtype,fbr,'nc')
                basenames = hf.return_basenames(filenames)
                datetimes = hf.basename_to_datetime(basenames)
                idx   = pd.Index(datetimes,name='time')
                data_fibre = xr.open_mfdataset(filenames,
                                               concat_dim=idx,
                                               engine=engine,
                                               autoclose=True)
                data_fibre = data_fibre.sortby('time')
                data_fibre.expand_dims('fb')
                ll.append(data_fibre)

            dtype_data = xr.concat(ll,dim=pd.Index(fibres,name='fb'))
        setattr(self,dtype,dtype_data)
        return dtype_data
    def save_dtype_data(self,basename,dtype=None,dirname=None,engine=None):
        # Check that dtype is recognised
        dtype = self._check_dtype(dtype)
        # If already exists, return data
        data_exists = hasattr(self,dtype)
        if not data_exists:
            raise UserWarning("Manager doesn't have {} data".format(dtype))
        else:
            dtype_data = getattr(self,dtype)
            # make file path
            # get directory path and create it if non-existent
            if dirname is not None:
                dirname  = dirname  
            else:
                dirname = self.savedirs[dtype]
            if not os.path.exists(dirname):
                os.makedirs(dirname,exist_ok=True)
            # get file basename
            basename = "{0}_{1}.nc".format(basename,dtype)
            filename = os.path.join(dirname,basename)
            
            try:
                dtype_data.to_netcdf(filename,engine='netcdf4')
                print("Dataset '{}' "
                      "successfully saved to {}".format(dtype,filename))
            except:
                print("Dataset {} could not be saved.")
        return
    def read_lines(self,dirname=None):
        return self.read_dtype_data('lines',dirname)
    def read_wavesol(self,dirname=None):
        return self.read_dtype_data('wavesol',dirname)
    def calculate_lambda(self,reference=None,ft='gauss',orders=None,flim=2e3):
        ''' Returns wavelength and wavelength error for the lines using 
            polynomial coefficients in wavecoef_LFC.
            
            Adapted from HARPS mai_compute_drift.py'''
            
        if self.stage>1:
            if self.check_data == True:
                return self.data 
            else:
                try:
                    data = self.read_data()
                    return data
                except:
                    pass
                
        else:
            
            if hasattr(self,'lines'):
                lines0 = self.lines
            else:
                lines0 = self.read_lines()
            
            
            # wavelength calibrations
            #ws    = self.check_and_get_wavesol()
            #wc    = self.LFCws['coef']
            # coordinates
            times = lines0.coords['time']
            ids   = lines0.coords['id']
            fibre = lines0.coords['fb']
            nspec = len(lines0.coords['time'])
            if orders is not None:
                orders = orders
            else:
                orders = lines0.coords['od'].values
            print(orders)
            # select only lines which have fluxes above the limit
            lines0 = lines0.where(lines0['pars'].sel(par='flx')>flim)
            
            
            # new dataset
            dims_coeff  = ['fb','od','polyord']
            dims_wave   = ['fb','time','od','id','val']
            dims_rv     = ['fb','time','par']
            dims_att    = ['fb','time','att']
            shape_coeff = (len(fibre),len(orders),4)
            shape_wave  = (len(fibre),len(times),len(orders),400,2)
            shape_rv    = (len(fibre),len(times),2)
            shape_att   = (len(fibre),len(times),2)
            variables   = {'wave':(dims_wave,np.full(shape_wave,np.nan)),
                           'rv':(dims_rv,np.full(shape_rv,np.nan)),
                           'stat':(dims_att,np.full(shape_att,np.nan)),
                           'coeff':(dims_coeff,np.full(shape_coeff,np.nan))}
            coords      = {'fb':fibre.values,
                           'time':times.values,
                           'od':orders,
                           'id':ids.values,
                           'att':['nlines','average_flux'],
                           'val':['wav','dwav'],
                           'par':['rv','rv_err'],
                           'polyord':np.arange(4)}
            series     = xr.Dataset(variables,coords)
    
            for fbr in fibre:
                if reference is not None:
                    reference = reference
                else:
                    reference = self.refspec+'_e2ds_{}.fits'.format(fbr.values)
                spec0 = Spectrum(reference)
                thar  = spec0.__get_wavesol__('ThAr')
                coef  = spec0.wavecoeff_vacuum[orders]
                for j,time in enumerate(times):
                    print(fbr.values,j)
                    idx = dict(fb=fbr,time=times[j],od=orders)
                    # all lines in this order, fittype, exposure, fibre
                    l0 = lines0.sel(od=orders,ft=ft,time=times[j],fb=fbr)
                    # centers and errors
                    x     = (l0['pars'].sel(par='cen')).values
                    x_err = (l0['pars'].sel(par='cen_err')).values
                   
    #                coef  = wc.sel(patch=0,od=orders,
    #                               ft=ft,time=times[iref],
    #                               fb=fbr).values
                    # wavelength of lines
                    wave = np.sum([coef[:,i]*(x.T**i) \
                                   for i in range(np.shape(coef)[-1])],axis=0).T
                    series['wave'].loc[dict(fb=fbr,time=times[j],od=orders,val='wav')] = wave
                 
                    # wavelength errors
                    dwave = np.sum([(i+1)*coef[:,i+1]*(x.T**(i)) \
                                    for i in range(np.shape(coef)[-1]-1)],axis=0).T*x_err
                    #print(np.shape(hf.ravel(x)),np.shape(hf.ravel(x_err)))
                    series['wave'].loc[dict(fb=fbr,time=times[j],od=orders,val='dwav')] = dwave
            self.stage = 1
            self.data  = series
            #self.save_data()
        return self.data
    
    def calculate_rv(self,orders=None,iref=0,sigma=5,ft='gauss'):
        plot=False
        if self.stage<2:
            # load things that have not been loaded yet
            data_exists = self.check_data()
            if not data_exists:
                # try reading data
                data_from_file = self.read_data()
                if data_from_file is not None:
                    series = data_from_file
                else:
                    series = self.calculate_lambda()
            else:
                series = self.data
            if hasattr(self,'lines'):
                lines = self.lines
            else:
                lines = self.read_lines()
            #series = self.calculate_lambda()
        else:
            pass
        
        # coordinates
        times = series.coords['time']
        ids   = series.coords['id']
        fibre = series.coords['fb']
        nspec = len(times)
        if orders is not None:
            orders = orders
        else:
            orders = series.coords['od']
      
        # Radial velocity calculations
        for fbr in fibre:
            
            cen_ref = lines['pars'].sel(fb=fbr,time=times[iref],
                            od=orders,par='cen',ft=ft)
            index_ref = hf._get_index(cen_ref)
            wavref = series['wave'].sel(fb=fbr,time=times[iref],
                                       od=orders,val='wav')
            dwref  = series['wave'].sel(fb=fbr,time=times[iref],
                                       od=orders,val='dwav')

            wavref = np.ravel(wavref)
            dwref  = np.ravel(dwref)
            for j,time in enumerate(times):
                total_flux = lines['stat'].sel(fb=fbr,time=times[j],
                                  od=orders,odpar='sumflux')
                total_flux = np.sum(total_flux)
                cen   = lines['pars'].sel(fb=fbr,time=times[j],
                            od=orders,par='cen',ft=ft)
                index = hf._get_index(cen)
                
                
                wav1  = series['wave'].sel(fb=fbr,time=times[j],
                                          od=orders,val='wav')
                dwav1 = series['wave'].sel(fb=fbr,time=times[j],
                                          od=orders,val='dwav')

                wav1  = np.ravel(wav1)
                dwav1 = np.ravel(dwav1)
                # global shift
                c=2.99792458e8
                sel1,sel2 = hf._get_sorted(index,index_ref)
                v=hf.ravel(c*(wav1[sel1]-wavref[sel2])/wav1[sel1])
                dwav=hf.ravel(np.sqrt((c*dwref[sel2]/wavref[sel2])**2+\
                                      (c*dwav1[sel1]/wav1[sel1])**2))
                
                if j!=iref:
                    m=hf.sig_clip2(v,sigma)
                else:
                    m=np.arange(len(v))
   	  
                average_flux = total_flux / len(m)
                global_dv    = np.sum(v[m]/(dwav[m])**2)/np.sum(1/(dwav[m])**2)
                global_sig_dv= (np.sum(1/(dwav[m])**2))**(-0.5)
                print(fbr.values,j,len(v), sum(m), global_dv, global_sig_dv)
                if plot:
                    plt.figure()
                    plt.title("fbr={}, spec={}".format(fbr.values,times[j].values))
                    plt.scatter(np.arange(len(v)),v,s=2)
                    plt.scatter(np.arange(len(v))[m],v[m],s=2)
                    #plt.plot(freq1d[sel1]-freq1d_ref[sel2])
                series['rv'].loc[dict(fb=fbr,time=times[j],par='rv')] = global_dv
                series['rv'].loc[dict(fb=fbr,time=times[j],par='rv_err')] = global_sig_dv
                series['stat'].loc[dict(fb=fbr,time=times[j],att='nlines')] = len(m)
                series['stat'].loc[dict(fb=fbr,time=times[j],att='average_flux')] = average_flux
        self.data=series
        self.stage=2
        #self.save_data()
        return self.data
    
    def plot_rv_time(self,fittype='gauss',fibre=None): 
        dv = self.data
        fibre = hf.to_list(fibre) if fibre is not None else dv.coords['fb']
        plotter = SpectrumPlotter(bottom=0.12)
        fig, ax = plotter.figure, plotter.axes
        t = dv.coords['time'].values
        t = np.arange(len(t))
        
        for f in fibre:  
            x = dv['stat'].sel(fb=f,att='average_flux')
            y = dv['rv'].sel(par='rv',fb=f)
            y_err = dv['rv'].sel(par='rv_err',fb=f)
            ax[0].errorbar(t,y,y_err,label=f.values,ls='',marker='o',ms=2)
        if len(fibre)>1:
            AmB= dv['rv'].sel(par='rv',fb='A') - \
                 dv['rv'].sel(par='rv',fb='B')
            AmB_err = np.sqrt(np.sum([dv['rv'].sel(par='rv_err',fb=f)**2 \
                                  for f in ['A','B']],axis=0))
            ax[0].errorbar(t,AmB,AmB_err,label='A-B',ls='',marker='x',ms=5)
        ax[0].axhline(0,ls='--',c='k',lw=0.7)
        #[ax[0].axvline((10*i),ls=':',lw=0.5,c='C0') for i in range(len(t)//10)]
        ax[0].legend()
        ax[0].set_xlabel("Exposure number")
        ax[0].set_ylabel("Shift [m/s]")
        return plotter
    def plot_rv_flux(self,fittype='gauss',fibre=None): 
        dv = self.data
        fibre = hf.to_list(fibre) if fibre is not None else dv.coords['fb']
        plotter = SpectrumPlotter()
        fig, ax = plotter.figure, plotter.axes
       
        #AmB= dv['rv'].sel(par='rv',fb='A') - \
        #     dv['rv'].sel(par='rv',fb='B')
        #AmB_err = np.sqrt(np.sum([dv['rv'].sel(par='rv_err',fb=f)**2 \
        #                          for f in ['A','B']],axis=0))
        x0 = np.average(dv['stat'].sel(att='average_flux'),axis=0)
        #ax[0].errorbar(x0,AmB,AmB_err,label='A-B',ls='',marker='x',ms=5)
        for f in fibre:  
            x = dv['stat'].sel(fb=f,att='average_flux')
            y = dv['rv'].sel(par='rv',fb=f)
            y_err = dv['rv'].sel(par='rv_err',fb=f)
            ax[0].errorbar(x,y,y_err,label=f.values,ls='',marker='o',ms=2)
        ax[0].axhline(0,ls='--',c='k',lw=0.7)
        ax[0].legend()
        return plotter