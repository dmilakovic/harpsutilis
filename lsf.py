#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
import harps.settings as hs
from   harps.spectrum import Spectrum
import harps.io as io
import harps.containers as container
import harps.plotter as plot
import harps.fit as hfit
from harps.core import os, np, plt

import errno

from scipy import interpolate
from scipy.optimize import leastsq

###############################################################################
###########################    LSF MODELLER   #################################
###############################################################################
class LSFModeller(object):
    def __init__(self,outfile,e2dsfile,orders=None,specnum=10,segnum=16,subnum=4,
                 niter=4,LFC='HARPS',fibreshape='round'):
        ''' Initializes the LSF Modeller
        
        Args:
        -----
            manager: Manager class object with paths to LFC spectra
            orders:  (scalar or list) echelle orders for which to 
                     perform LSF modelling
            specnum: (scalar) number of spectra to use 
            segnum:  (scalar) number of subdivisions of 4096 pixel 
            niter:   (scalar) number of iterations for LSF modelling
            fibre:   (str) fibres to perform LSF modelling 
            LFC:     (str) HARPS or FOCES
            fibreshape: (str) shape of the fibre (round or octagonal)
            
        '''
        if orders is not None:
            orders = hf.to_list(orders)
        else:
            orders = np.arange(hs.sOrder,hs.eOrder,1)
            
        self.outfile = outfile
        self.orders  = orders
        self.specnum = specnum
        self.segnum  = segnum
        self.subnum  = subnum
        self.LFC     = LFC
        self.niter   = niter
        self.fibreshape = fibreshape
        
        self.interpolate=True
        self._cache = {}
        
        self.savedir  = os.path.join(hs.dirnames['lsf'],'April2015_2')
    def read(self):
        cache    = io.mread_outfile(self.outfile,['linelist',
                                                  'flux','background'])
        self._cache = cache
        
    def get_dataset(self,dataset):
        try:
            data = self._cache[dataset]
        except:
            output  = io.mread_outfile(self.outfile,[dataset])
            data = output[dataset]
            self._cache[dataset] = data
        return data
    
    def return_eLSF(self,fibre,order):
        ''' Performs effective LSF reconstruction in totality'''
        manager = self.manager
        niter   = self.niter
        orders  = order#self.orders
        segnum  = self.segnum
        #subnum  = self.subnum
        specnum = self.specnum
        
        interpolate_local_psf = self.interpolate
        fit_gaussians = self.fit_gaussians
        
        #orders = orders if orders is not None else [45]
        
        data0   = self.initialize_dataset(order)
        data    = self.stack_lines_from_spectra(data0,fibre,first_iteration=True,fit_gaussians=fit_gaussians) 
        # j counts iterations
        j = 0
        # 
        data_with_pars = data_with_eLSF = data_recentered = data
        plot_elsf = False
        plot_cen  = False
        if plot_elsf:
            fig_elsf,ax_elsf = hf.get_fig_axes(segnum,alignment='grid',title='LSF iteration')
        if plot_cen:
            fig_cen,ax_cen = hf.get_fig_axes(1,title='Centeroid shifts')
        # iteratively derive LSF 
        while j < niter:
            data_with_eLSF  = self.construct_eLSF(data_recentered)
            data_with_pars  = self.solve_line_positions(data_with_eLSF,interpolate_local_psf)
            data_recentered = self.stack_lines_from_spectra(data_with_pars,fibre,False)       
            
            j +=1
        final_data = data_recentered
        return final_data
    def run(self):
        orders = self.orders
        fibres = self.fibres
        for fibre in list(fibres):
        
            for order in orders:
                
                filepath = self.get_filepath(order,fibre)
                #print(filepath)
                fileexists = os.path.isfile(filepath)
                if fileexists == True:
                    print('FIBRE {0}, ORDER {1} {2:>10}'.format(fibre,order,'exists'))
                    continue
                else:
                    print('FIBRE {0}, ORDER {1} {2:>10}'.format(fibre,order,'working'))
                    pass
                
                data=self.return_eLSF(fibre,order)
                self.save2file(data,fibre)
        return data
                
            
    def initialize_dataset(self,order):
        ''' Returns a new xarray dataset object of given shape.'''
#        orders  = order #self.orders
        specnum = self.specnum
        segnum  = self.segnum
        subnum  = self.subnum
        nOrders = 1
       
        # number of pixels each eLSF comprises of
        npix   = 17
        # make the subsampled grid where eLSF will be tabulated
        a      = divmod(npix,2)
        xrange = (-a[0],a[0]+a[1])
        pixels    = np.arange(xrange[0],xrange[1],1/subnum)
        # assume each segment contains 60 lines (not true for large segments!)
        lines_per_seg = 60
        # create a multi-index for data storage
        mdix      = pd.MultiIndex.from_product([np.arange(segnum),
                                            np.arange(specnum),
                                            np.arange(lines_per_seg)],
                                names=['sg','sp','id'])
        ndix      = specnum*segnum*lines_per_seg
        # axes for each line
        # x = x coords in eLSF reference frame
        # y = eLSF estimates in eLSF reference frame
        # pos = x coords on CCD
        # flx = counts extracted from CCD
        # err = sqrt(flx), backgroud included
        # lsf = eLSF estimate for this line
        # rsd = residuals between the model and flux
        # der = derivatives of eLSF
        # w   = weigths 
        # mod = model of the line
        axes   = ['x','y','pos','flx','err','bkg','lsf','rsd','der','w','mod']
        n_axes = len(axes)
        # values for each parameter
        values = ['cen','cen_err','flx','flx_err','sft','phi','bary','cen_1g']
        n_vals = len(values)
        # create xarray Dataset object to save the data
        data0   = xr.Dataset({'line': (['od','pix','ax','idx'], np.full((nOrders,npix*subnum,n_axes,ndix),np.nan)),
                         'resd': (['od','seg','pix'],np.full((nOrders,segnum,npix*subnum),np.nan)),
                         'elsf': (['od','seg','pix','ax'], np.full((nOrders,segnum,npix*subnum,n_axes),np.nan)),
                         'shft': (['od','seg'], np.full((nOrders,segnum),np.nan)),
                         'pars': (['od','idx','val'], np.full((nOrders,ndix,n_vals),np.nan)),
                         'gauss': (['ng','od','pix','ax','idx'], np.full((2,nOrders,npix*subnum,n_axes,ndix),np.nan))},
                         coords={'od' :[order], 
                                 'idx':mdix, 
                                 'pix':pixels,
                                 'seg':np.arange(segnum),
                                 'ax' :axes,
                                 'val':values,
                                 'ng' :[1,2]})
    
        return data0
    
    def stack_lines2d(self):
        linelist = self.get_linelist()
        
        numpix = np.max(linelist['pixr']-linelist['pixl'])
        #subseg = 
        
        
        
        
    
            
            
            
    def stack_lines_from_spectra(self,data,fibre='A',first_iteration=None,fit_gaussians=False):
        ''' Stacks LFC lines along their determined centre
        
            Stacks the LFC lines along their centre (or barycentre) using all the 
            spectra in the provided Manager object. Returns updated xarray dataset 
            (provided by the keyword data).
        '''
        manager = self.manager
        def get_idxs(barycenters,order,nspec):
            '''Returns a list of (segment,spectrum,index) for a given order.'''
            segs=np.asarray(np.array(barycenters[order])//s,np.int32)
            seg,frq = np.unique(segs,return_counts=True)
            nums=np.concatenate([np.arange(f) for s,f in zip(seg,frq)])
            idxs = [(s, nspec, i) for s,i in zip(segs,nums)]
            return idxs
        def return_n_filepaths(manager,N,fibre,skip=5):
            ''' Returns a list of length N with paths to HARPS spectra contained
                in the Manager object. Skips files so to improve dithering of
                lines.
            '''
            i = 0
            files = []
            while len(files) < N:
                spec = Spectrum(manager.file_paths[fibre][skip*i+1])
                spec.__get_wavesol__('ThAr')
                if np.sum(spec.wavesol_thar)==0:
                    pass
                else:
                    files.append(spec.filepath)
                i+=1
                
            return files
        if first_iteration == None:
            # check if data['pars'] is empty
            if np.size(data['pars'].dropna('val','all')) == 0:
                first_iteration = True
            else:
                first_iteration = False
            
        orders          = data.coords['od'].values
        pixels          = data.coords['pix'].values
        specnum         = np.unique(data.coords['sp'].values).size
        pix_step        = pixels[1]-pixels[0]
        pixelbins       = (pixels[1:]+pixels[:-1])/2
        segments        = np.unique(data.coords['seg'].values)
        N_seg           = len(segments)
        s               = 4096//N_seg
        pbar            = tqdm.tqdm(total=(specnum*len(orders)),desc="Centering spectra")
        files           = return_n_filepaths(manager,specnum,fibre)
        for i_spec, file in enumerate(files):
            #print("SPEC {0} {1}".format(fibre,i_spec+1))
            # use every 5th spectrum to improve the sampling of the PSF
            spec = Spectrum(file,LFC='FOCES')
            
            xdata,ydata,edata,bdata,barycenters =spec.cut_lines(orders,nobackground=False,
                                              columns=['pixel','flux','error','bkg','bary'])
            
            for o,order in enumerate(orders):
                idxs = get_idxs(barycenters,order,i_spec)
                numlines = len(barycenters[order])
                if first_iteration:
                    maxima      = spec.get_extremes(order,scale='pixel',extreme='max')['y']
                    lines_1g    = spec.fit_lines(order,fittype='gauss')
                # stack individual lines
                for i in range(numlines):
                    
                    line_pix = xdata[order][i]
                    line_flx = ydata[order][i]
                    line_err = edata[order][i]
                    line_bkg = bdata[order][i]
                    line_flx_nobkg = line_flx-line_bkg
                    idx = idxs[i]
                    # cen is the center of the ePSF!
                    # all lines are aligned so that their cen aligns
                    if first_iteration:
                        b        = barycenters[order][i]
                        cen      = barycenters[order][i]
                        cen_err  = 0
                        flux     = np.max(line_flx)#maxima.iloc[i]
                        flux_err = np.sqrt(flux)
                        shift    = 0
                        phase    = cen - int(cen+0.5)
                        try: cen_g1   = lines_1g.center.iloc[i]
                        except: cen_g1 = np.nan
                        pars     = (cen,cen_err,flux,flux_err,shift,phase,b,cen_g1)
                    else:
                        pars = data['pars'].sel(od=order,idx=idx).values
                        cen,cen_err,flux,flux_err,shift,phase,b,cen_g1 = pars
                    
                    data['pars'].loc[dict(idx=idx,od=order)] = np.array(pars)
                    
                    #-------- MOVING TO A COMMON FRAME--------
                    xline0 = line_pix - cen
                    pix = pixels[np.digitize(xline0,pixelbins,right=True)]
                    
                    #-------- LINE POSITIONS & FLUX --------
                    # first clear the previous estimates of where the line is located
                    # (new estimate of the line center might make the values in the  
                    # 'pix' array different from before, but this is not erased
                    # with each new iteration)
                    
                    data['line'].loc[dict(ax='pos',od=order,idx=idx)]=np.nan
                    data['line'].loc[dict(ax='flx',od=order,idx=idx)]=np.nan
                    data['line'].loc[dict(ax='err',od=order,idx=idx)]=np.nan
                    data['line'].loc[dict(ax='bkg',od=order,idx=idx)]=np.nan
                    data['line'].loc[dict(ax='x',od=order,idx=idx)]  =np.nan
                    data['line'].loc[dict(ax='w',od=order,idx=idx)]  =np.nan
                    # --------------------------------------
                    # Save new values
                    data['line'].loc[dict(ax='pos',od=order,idx=idx,pix=pix)]=line_pix
                    data['line'].loc[dict(ax='flx',od=order,idx=idx,pix=pix)]=line_flx
                    data['line'].loc[dict(ax='err',od=order,idx=idx,pix=pix)]=line_err
                    data['line'].loc[dict(ax='bkg',od=order,idx=idx,pix=pix)]=line_bkg
                    data['line'].loc[dict(ax='x',od=order,idx=idx,pix=pix)]  =line_pix-cen
                    
                    # --------WEIGHTS--------
                    # central 2.5 pixels on each side have weights = 1
                    central_pix = pix[np.where(abs(pix)<=2.5)[0]]
                    data['line'].loc[dict(ax='w',od=order,idx=idx,pix=central_pix)]=1.0
                    # pixels outside of 5.5 have weights = 0
                    outer_pix   = pix[np.where(abs(pix)>=5.5)[0]]
                    data['line'].loc[dict(ax='w',od=order,idx=idx,pix=outer_pix)]=0.0
                    # pixels with 2.5<abs(pix)<5.5 have weights between 0 and 1, linear
                    midleft_pix  = pix[np.where((pix>-5.5)&(pix<-2.5))[0]]
                    midleft_w   = np.array([(x+5.5)/3 for x in midleft_pix])
                    
                    midright_pix = pix[np.where((pix>2.5)&(pix<5.5))[0]]
                    midright_w   = np.array([(-x+5.5)/3 for x in midright_pix])
                    
                    data['line'].loc[dict(ax='w',od=order,idx=idx,pix=midleft_pix)] =midleft_w
                    data['line'].loc[dict(ax='w',od=order,idx=idx,pix=midright_pix)]=midright_w
                    
                    #-------- NORMALISE LINE ------
                    data['line'].loc[dict(ax='y',od=order,idx=idx,pix=pix)]  =(line_flx_nobkg)/np.sum(line_flx_nobkg)
                pbar.update(1)
        pbar.close()
        return data
    def construct_eLSF(self,data):
        n_iter   = self.niter
        orders   = data.coords['od'].values
        segments = np.unique(data.coords['seg'].values)
        N_seg    = len(segments)
        pixels   = data.coords['pix'].values
        N_sub    = round(len(pixels)/(pixels.max()-pixels.min()))
        
        clip     = 2.5
        
        plot = False
        pbar     = tqdm.tqdm(total=(len(orders)*N_seg),desc='Constructing eLSF')
        if plot:
            fig, ax = hf.get_fig_axes(N_seg,alignment='grid')
        for o,order in enumerate(orders):
            for n in segments:
                j = 0
                # select flux data for all lines in the n-th segment and the right order
                # drop all NaN values in pixel and 
                segment = data['line'].sel(sg=n,od=order).dropna('pix','all').dropna('idx','all')
                # extract data in x and y, corresponding coordinates and line idx
                y_data = segment.sel(ax='y')#.dropna('pix','all').dropna('idx','all')
                x_data = segment.sel(ax='x')#.dropna('pix','all').dropna('idx','all')
                x_coords = y_data.coords['pix'].values
                line_idx = [(n,*t) for t in y_data.coords['idx'].values]
                #line_idx = y_data.coords['idx'].values
                # initialise effective LSF of this segment as null values    
                data['elsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] = 0
                data['elsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')] = x_coords
                delta_x    = 0
                sum_deltax = 0
                
                
                while j<n_iter:
                    if np.isnan(delta_x):
                        print("delta_x is NaN!")
                        
                        return data
                    # read the latest eLSF array for this order and segment, drop NaNs
                    elsf_y  = data['elsf'].sel(od=order,seg=n,ax='y',pix=x_coords)
                    elsf_x  = data['elsf'].sel(od=order,seg=n,ax='x',pix=x_coords)
                    # construct the spline using x_coords and current eLSF, 
                    # evaluate eLSF for all points and save values and residuals
                    splr = interpolate.splrep(elsf_x.values,elsf_y.values)                    
                    sple = interpolate.splev(x_data.values,splr)
    #                print(sple)
                    #print(len(line_idx),len(sple))
                    rsd  = (y_data-sple)
                    data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='lsf')] = sple
                    data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='rsd')] = rsd.values
                    # calculate the mean of the residuals between the samplings and eLSF
                    testbins  = np.array([(l,u) for l,u in zip(x_coords-1/N_sub,x_coords+1/N_sub)])
                               
                    
                    rsd_muarr = np.zeros_like(x_coords)
                    rsd_sigarr = np.zeros_like(x_coords)
                    #-------- ITERATIVELY REJECT SAMPLIN MORE THAN 2.5sigma FROM THE MEAN --------
                    for i in range(rsd_muarr.size):
                        llim, ulim = testbins[i]
                        rsd_cut = rsd.where((x_data>llim)&(x_data<=ulim)).dropna('pix','all')     
                        if rsd_cut.size == 0:
                                break
                        sigma_old = 999
                        dsigma    = 999
                        while dsigma>1e-2:
                            mu = rsd_cut.mean(skipna=True).values
                            sigma = rsd_cut.std(skipna=True).values
                            if ((sigma == 0) or (np.isnan(sigma)==True)):
                                break
                            rsd_cut.clip(mu-sigma*clip,mu+sigma*clip).dropna('idx','all')
                            
                            dsigma = (sigma_old-sigma)/sigma
                            sigma_old = sigma
                        rsd_muarr[i]  =   mu
                        rsd_sigarr[i] =   sigma
                    rsd_mean = xr.DataArray(rsd_muarr,coords=[x_coords],dims=['pix']).dropna('pix','all')
                    #rsd_sigma= xr.DataArray(rsd_sigarr,coords=[x_coords],dims=['pix']).dropna('pix','all')
                    #rsd_coords = rsd_mean.coords['pix']
                    #print(rsd_coords==x_coords)
                    # adjust current model of the eLSF by the mean of the residuals
                    data['elsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')]  = x_coords
                    data['elsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] += rsd_mean
                    # re-read the new eLSF model: 
                    elsf_y = data['elsf'].sel(od=order,seg=n,ax='y',pix=x_coords)
                    elsf_x = data['elsf'].sel(od=order,seg=n,ax='x',pix=x_coords)
                    elsf_c = elsf_y.coords['pix']
                    # calculate the derivative of the new eLSF model
                    elsf_der = xr.DataArray(hf.derivative1d(elsf_y.values,elsf_x.values),coords=[elsf_c],dims=['pix'])
                    data['elsf'].loc[dict(od=order,seg=n,ax='der',pix=elsf_c)] =elsf_der
                    # calculate the shift to be applied to all samplings
                    # evaluate at pixel e
                    e = 0.5
                    elsf_neg     = elsf_y.sel(pix=-e,method='nearest').values
                    elsf_pos     = elsf_y.sel(pix=e,method='nearest').values
                    elsf_der_neg = elsf_der.sel(pix=-e,method='nearest').values
                    elsf_der_pos = elsf_der.sel(pix=e,method='nearest').values
                    delta_x      = (elsf_pos-elsf_neg)/(elsf_der_pos-elsf_der_neg)
  
                    if plot:
   
                        ax[n].scatter(elsf_x.values,elsf_y.values,marker='s',s=10,c='C{}'.format(j+1)) 
                        ax[n].axvline(0,ls='--',lw=1,c='C0')
                        ax[n].scatter(x_data.values,y_data.values,s=1,c='C{}'.format(j),marker='s',alpha=0.5)
  
                            
                    j+=1               
                    # shift the sampling by delta_x for the next iteration
                    x_data += delta_x
                    # add delta_x to total shift over all iterations
                    sum_deltax += delta_x
                    if np.isnan(delta_x):
                        
                        print("delta_x is NaN!")
                        print(x_data)
                data['shft'].loc[dict(seg=n,od=order)] = sum_deltax
               
                pbar.update(1)
                
        pbar.close()
        
        return data
    def solve_line_positions(self,data,interpolate_local_psf=True):
        ''' Solves for the flux of the line and the shift (Delta x) from the center
        of the brightest pixel'''
       
        
        orders          = data.coords['od'].values
        pixels          = data.coords['pix'].values
        midx            = data.coords['idx'].values
        segments        = np.unique(data.coords['seg'].values)
        segnum          = len(segments)
        s               = 4096//segnum
        segment_limits  = sl = np.linspace(0,4096,segnum+1)
        segment_centers = sc = (sl[1:]+sl[:-1])/2
        segment_centers[0] = 0
        segment_centers[-1] = 4096
        def residuals(x0,pixels,counts,weights,background,splr):
            # center, flux
            sft, flux = x0
            model = flux * interpolate.splev(pixels+sft,splr)
            # sigma_tot^2 = sigma_counts^2 + sigma_background^2
            # sigma_counts = sqrt(counts)     sigma_background = sqrt(background)
            error = np.sqrt(counts + background)
            resid = np.sqrt(line_w) * ((counts-background) - model) / error
            #resid = line_w * (counts- model)
            return resid
            
        for order in orders:
            for idx in midx:
                sg,sp,lid = idx
                line_pars = data['pars'].sel(idx=idx,od=order).values
                cen,cen_err, flx, flx_err, dx, phi, b, cen_1g = line_pars
                p0 = (dx,flx)
                if np.isnan(p0).any() == True:
                    continue
                line   = data['line'].sel(idx=idx,od=order)
                line_x = line.sel(ax='pos').dropna('pix')
                lcoords=line_x.coords['pix']
                line_x = line_x.values
                line_y = line.sel(ax='flx').dropna('pix').values
                line_b = line.sel(ax='bkg').dropna('pix').values
                line_w = line.sel(ax='w').dropna('pix').values
                if ((len(line_x)==0)or(len(line_y)==0)or(len(line_w)==0)):
                    continue
                cen_pix = line_x[np.argmax(line_y)]  
                
                elsf_x  = data['elsf'].sel(ax='x',od=order,seg=sg).dropna('pix')+cen_pix
                
                #---------- CONSTRUCT A LOCAL LSF ----------
                # find in which segment the line falls
                sg2     = np.digitize(cen_pix,segment_centers)
                sg1     = sg2-1
                elsf1   = data['elsf'].sel(ax='y',od=order,seg=sg1).dropna('pix') 
                elsf2   = data['elsf'].sel(ax='y',od=order,seg=sg2).dropna('pix')
                if interpolate_local_psf:
                    f1 = (sc[sg2]-cen_pix)/(sc[sg2]-sc[sg1])
                    f2 = (cen_pix-sc[sg1])/(sc[sg2]-sc[sg1])
                    elsf_y  = f1*elsf1 + f2*elsf2  
                else:
                    elsf_y  = data['elsf'].sel(ax='y',od=order,seg=sg).dropna('pix') 
                elsf_x  = elsf_x.sel(pix=elsf_y.coords['pix'])
                splr    = interpolate.splrep(elsf_x.values,elsf_y.values)
                popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                                                       args=(line_x,line_y,line_w,line_b,splr),
                                                       full_output=True)
                
                if ier not in [1, 2, 3, 4]:
                    print((3*("{:<3d}")).format(*idx),"Optimal parameters not found: " + errmsg)
                    popt = np.full_like(p0,np.nan)
                    pcov = None
                    success = False
                else:
                    success = True
                if success:
                    
                    sft, flx = popt
                    cost = np.sum(infodict['fvec']**2)
                    dof  = (len(line_x) - len(popt))
                    if pcov is not None:
                        pcov = pcov*cost/dof
                    else:
                        pcov = np.array([[np.inf,0],[0,np.inf]])
                    #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
                else:
                    continue
                #print('CHISQ = {0:15.5f}'.format(cost/dof))
                cen = line_x[np.argmax(line_y)]-sft
                cen_err, flx_err = [np.sqrt(pcov[i][i]) for i in range(2)]
                phi = cen - int(cen+0.5)
                save_pars = np.array([cen,cen_err,flx,flx_err, sft,phi,b,cen_1g])
                data['pars'].loc[dict(idx=idx,od=order)] = save_pars
                data['line'].loc[dict(idx=idx,od=order,ax='lsf',pix=elsf_y.coords['pix'])]=elsf_y
                
                # calculate residuals:
                model = flx * interpolate.splev(line_x+sft,splr) 
                resid = (line_y-line_b) - model
                data['line'].loc[dict(idx=idx,od=order,ax='mod',pix=lcoords)]=model
                data['line'].loc[dict(idx=idx,od=order,ax='rsd',pix=lcoords)]=resid
                
        return data
    def get_filepath(self,order,fibre):
        dirpath  = os.path.join(self.savedir,'fibre{}'.format(fibre))
        basepath = '{LFC}_fib{fb}{sh}_order_{od}.nc'.format(LFC=self.LFC,
                                                         fb=fibre,
                                                         sh=self.fibreshape,
                                                         od=order)
        filepath = os.path.join(dirpath,basepath)
        return filepath
    def save2file(self,data,fibre):
#        fibres = self.fibres
        order = int(data.coords['od'].values)
        filepath = self.get_filepath(order,fibre)
        data4file = data.unstack('idx')
        data4file.attrs['LFC'] = self.LFC
        data4file.attrs['fibreshape'] = self.fibreshape
        data4file.attrs['interpolate'] = int(self.interpolate)
        data4file.attrs['fit_gaussians'] = int(self.fit_gaussians)
        data4file.to_netcdf(filepath)
        print("Saved to {}".format(filepath))
        
        return
def stack_lines_multispec(linelists,fluxes,backgrounds,fittype):
    xx = []
    yy = []
    oo = []
#    fittype = 'lsf'
#    if first_iteration:
#        fittype='gauss'
    
    for l,f,b in zip(linelists,fluxes,backgrounds):
        xarray, yarray, orders = stack_lines_singlespec(fittype,l,f,b)
        xx.append(xarray)
        yy.append(yarray)
        oo.append(orders)
        #sys.exit()
    return np.dstack(xx), np.dstack(yy), orders
        
def stack_lines_singlespec(fittype,linelist,flux,background=None):
    xarray = np.zeros((72,4096))
    yarray = np.zeros((72,4096))
        
    orders = np.unique(linelist['order'])
    for od in orders:
        cut0   = np.where(linelist['order']==od)
        lines0 = linelist[cut0]
        #segs   = np.digitize(lines0[fittype][:,1],seglims)
        for j,line in enumerate(lines0):
            #segment = segs[j]
            pixl      = line['pixl']
            pixr      = line['pixr']
            lineflux  = flux[od,pixl:pixr]
            if background is not None:
                lineflux = lineflux - background[od,pixl:pixr]
            
            # move to frame centered at 0 
            #print(line[fittype])
            x_vals = np.arange(line['pixl'],line['pixr']) - line[fittype][1]
            #print(x_vals)
            y_vals = lineflux/np.sum(lineflux)
            xarray[od,pixl:pixr] = x_vals
            yarray[od,pixl:pixr] = y_vals

    return xarray,yarray,orders
def solve_multispec(linelists,fluxes,backgrounds,errors,lsf):
    new_linelist = []
    tot = len(linelists)
    for i,l,f,b,e in zip(np.arange(tot),linelists,fluxes,backgrounds,errors):
        linelist = solve_singlespec(l,f,b,e,lsf)
        new_linelist.append(linelist)
        hf.update_progress(i/(tot-1))
    return np.array(new_linelist)
def solve_singlespec(linelist,flux,background,error,lsf):
    """ Performs LSF fitting on already identified lines """
    for i,line in enumerate(linelist):
        # order 
        od   = line['order']
        seg  = line['segm']+1
        # mode edges
        lpix = line['pixl']
        rpix = line['pixr']
        flx  = flux[od,lpix:rpix]
        
        pix  = np.arange(lpix,rpix,1.) 
        bkg  = background[od,lpix:rpix]
        err  = error[od,lpix:rpix]
        wgt  = np.ones_like(pix)
        # barycenter
        bary = np.sum(flx*pix)/np.sum(flx)
        
        #pix  = pix-bary
        # initial guess
        p0 = (np.max(flx),0)
        cut = np.where((lsf['order']==od)&(lsf['segm']==seg))[0][0]
        lsf_local  = lsf[cut]
        pars,errs, chisq,model = hfit.lsf(pix-bary,flx,bkg,err,wgt,
                                          lsf_local,p0,output_model=True)
        amp, shift = pars
        center = bary + shift
#            print(amp,center)
        line['lsf'] = [amp,center,0]
        line['lsf_err'] = [*errs,0]
        line['lchisq']= chisq
        #print(line['lsf'])
    return linelist
def model(outlist,numiter=3):
    extensions  = ['linelist','flux','background','error']
    cache       = io.mread_outfile(outlist,extensions,version=None)
    linelists   = cache['linelist']
    fluxes      = cache['flux']
    backgrounds = cache['background']
    errors      = cache['background']
    fittype     = 'lsf'
    for i in range(numiter):
        if i == 0:
            fittype = 'gauss'
        xarrays, yarrays, orders = stack_lines_multispec(linelists,fluxes,
                                                 backgrounds,fittype)
        lsf_iter = construct_lsf(xarrays,yarrays,orders,numseg=16,numpix=30,
                                 subpix=4,numiter=5)
        linelists = solve_multispec(linelists,fluxes,backgrounds,errors,lsf_iter)
    lsf = lsf_iter
    return lsf
def construct_lsf(x, y, orders,
                  numseg=16,numpix=30,subpix=4,numiter=5,**kwargs):
    lst = []
    for i,od in enumerate(orders):
        if len(orders)>1:
            hf.update_progress(i/(len(orders)-1),'LSF fitting')
        lsf1d=(construct_lsf1d(x[od],y[od],numseg,numpix,subpix,numiter,**kwargs))
        lsf1d['order'] = od
        lst.append(lsf1d)
    lsf = np.hstack(lst)
    
    return lsf
def construct_lsf1d(xdata,ydata,numseg=16,numpix=30,subpix=4,numiter=5,**kwargs):
    """ Input: single order output of stack_lines_multispec"""
    do_plot    = kwargs.pop('plot',False)
    seglims = np.linspace(0,4096,numseg+1,dtype=int)
    totpix  = numpix*subpix+1
    
    pixcens = np.linspace(-numpix/2,numpix/2,totpix)
    pixlims = pixcens+0.5/subpix
    
    # lsf for the entire order, divided into segments
    lsf1d      = container.lsf(numseg,totpix)
    lsf1d['x'] = pixcens
    count = 0
    for i,lsf in enumerate(lsf1d):
        
        pixl = seglims[i]
        pixr = seglims[i+1]
        # save pixl and pixr
        lsf['pixl'] = pixl
        lsf['pixr'] = pixr
        x_vals = np.ravel(xdata[pixl:pixr])
        y_vals = np.ravel(ydata[pixl:pixr])
        
        # remove zeros
        zeros  = y_vals==0
        y_vals = y_vals[~zeros]
        x_vals = x_vals[~zeros]
        shift  = 0
        if do_plot:
            plotter=plot.Figure(2,figsize=(9,6),sharex=True,ratios=[2,1])
            ax = plotter.axes
            ax[0].scatter(x_vals,y_vals,s=1)
        for j in range(numiter):
            # shift the values along x-axis for improved centering
            x_vals = x_vals+shift
            # get current model of the LSF
            splr = interpolate.splrep(lsf['x'],lsf['y'])                    
            sple = interpolate.splev(x_vals,splr)
            # calculate residuals to the model
            rsd  = (y_vals-sple)
            
            # calculate mean of residuals for each pixel comprising the LSF
            means  = bin_means(x_vals,rsd,pixlims)
            lsf['y'] = lsf['y']+means
            
            # calculate derivative
            deriv = hf.derivative1d(lsf['y'],lsf['x'])
            lsf['dydx'] = deriv
            
            left  = np.where(lsf['x']==-0.5)[0]
            right = np.where(lsf['x']==0.5)[0]
            elsf_neg     = lsf['y'][left]
            elsf_pos     = lsf['y'][right]
            elsf_der_neg = lsf['dydx'][left]
            elsf_der_pos = lsf['dydx'][right]
            shift        = (elsf_pos-elsf_neg)/(elsf_der_pos-elsf_der_neg)
            count        +=1
            #hf.update_progress(count/(numseg*numiter-1),"LSF modelling")
            #print(i,j,left,right,elsf_neg,elsf_pos,elsf_der_neg,elsf_der_pos,shift)
            #print(i,j,shift)
            if do_plot:
                ax[1].scatter(x_vals,rsd,s=1)
    #            ax[1].scatter(pixcens,means,marker='s',s=2)
                
                ax[0].errorbar(lsf['x'],lsf['y'],xerr=0.5/subpix,ms=10,marker='x')
                ax[0].vlines(pixlims,0,0.25,linestyles=':',lw=0.4,colors='k')
    return lsf1d
def bin_means(x,y,xbins):
    inds  = np.digitize(x,xbins,right=False)
    means = np.zeros(len(xbins))
    for i in np.unique(inds):
        if i>=len(xbins):
            continue
        cut = np.where(inds==i)[0]
        if len(cut)<1:
            continue
        
        yi  = y[cut]
        means[i] = np.nanmean(yi)
    return means
    