#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:04:13 2018

@author: dmilakov
"""

import harps.utilis as h
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from scipy.interpolate import splrep,splev
from scipy.optimize import newton,leastsq
import tqdm
import os
import datetime

#%%
class Labeloffset():
    def __init__(self,  ax, label="", axis="y"):
        self.axis = {"y":ax.yaxis, "x":ax.xaxis}[axis]
        self.label=label
        ax.callbacks.connect(axis+'lim_changed', self.update)
        ax.figure.canvas.draw()
        self.update(None)

    def update(self, lim):
        fmt = self.axis.get_major_formatter()
        self.axis.offsetText.set_visible(False)
        self.axis.set_label_text(self.label + r"$\times$"+ fmt.get_offset() )

#%%
def solve(data,interpolate=True):
    ''' Solves for the flux of the line and the shift (Delta x) from the center
    of the brightest pixel'''
    #data = xr.open_dataset(path=filepath,chunks=chunks)
    
    orders          = data.coords['od'].values
    pixels          = data.coords['pix'].values
    midx            = data.coords['idx'].values
    segments        = np.unique(data.coords['seg'].values)
    N_seg           = len(segments)
    s               = 4096//N_seg
    segment_limits  = sl = np.linspace(0,4096,N_seg+1)
    segment_centers = sc = (sl[1:]+sl[:-1])/2
    segment_centers[0] = 0
    segment_centers[-1] = 4096
    def residuals(x0,pixels,counts,weights,background,splr):
        # center, flux
        sft, flux = x0
        model = flux * splev(pixels+sft,splr) 
        #print(counts)
        resid = np.sqrt(line_w) * ((counts-background) - model) / np.sqrt(np.abs(counts))
        #resid = line_w * (counts- model)
        return resid
        
    for order in orders:
        #fig, axes = plt.
        for idx in midx:
            sg,sp,lid = idx
            line_pars = data['pars'].sel(idx=idx,od=order).values
            cen,cen_err, flx, flx_err, dx, phi, b, cen_1g = line_pars
            p0 = (dx,flx)
            #print(idx)
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
            
            epsf_x  = data['epsf'].sel(ax='x',od=order,seg=sg).dropna('pix')+cen_pix
            
            #---------- CONSTRUCT A LOCAL PSF ----------
            # find in which segment the line falls
            sg2     = np.digitize(cen_pix,segment_centers)
            sg1     = sg2-1
            epsf1   = data['epsf'].sel(ax='y',od=order,seg=sg1).dropna('pix') 
            epsf2   = data['epsf'].sel(ax='y',od=order,seg=sg2).dropna('pix')
            if interpolate:
                f1 = (sc[sg2]-cen_pix)/(sc[sg2]-sc[sg1])
                f2 = (cen_pix-sc[sg1])/(sc[sg2]-sc[sg1])
                epsf_y  = f1*epsf1 + f2*epsf2  
            else:
                epsf_y  = data['epsf'].sel(ax='y',od=order,seg=sg).dropna('pix') 
            epsf_x  = epsf_x.sel(pix=epsf_y.coords['pix'])
            splr    = splrep(epsf_x.values,epsf_y.values)
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
            data['line'].loc[dict(idx=idx,od=order,ax='psf',pix=epsf_y.coords['pix'])]=epsf_y
            
            # calculate residuals:
            model = flx * splev(line_x+sft,splr) 
            resid = (line_y-line_b) - model
            data['line'].loc[dict(idx=idx,od=order,ax='mod',pix=lcoords)]=model
            data['line'].loc[dict(idx=idx,od=order,ax='rsd',pix=lcoords)]=resid
            #residuals(popt,line_x,line_y,line_w,line_b,splr)
    #data.to_netcdf(path=filepath,mode='a')
    return data
#%%
def stack_lines_from_spectra(manager,data,first_iteration=None,fit_gaussians=False):
#    n_spec = n_spec if n_spec is not None else manager.numfiles[0] 
    #data = xr.open_dataset(path=filepath,chunks=chunks)
    def get_idxs(barycenters,order,nspec):
        segs=np.asarray(np.array(barycenters[order])//s,np.int32)
        seg,frq = np.unique(segs,return_counts=True)
        nums=np.concatenate([np.arange(f) for s,f in zip(seg,frq)])
        idxs = [(s, nspec, i) for s,i in zip(segs,nums)]
        return idxs
    if first_iteration == None:
        # check if data['pars'] is empty
        if np.size(data['pars'].dropna('val','all')) == 0:
            first_iteration = True
        else:
            first_iteration = False
        
    orders          = data.coords['od'].values
    pixels          = data.coords['pix'].values
    n_spec          = np.unique(data.coords['sp'].values).size
    pix_step        = pixels[1]-pixels[0]
    pixelbins       = (pixels[1:]+pixels[:-1])/2
    segments        = np.unique(data.coords['seg'].values)
    N_seg           = len(segments)
    s               = 4096//N_seg
    pbar            = tqdm.tqdm(total=(n_spec*len(orders)),desc="Centering spectra")
    for i_spec in range(n_spec):
        
        
        #print("SPEC {}".format(i_spec+1))
        spec = h.Spectrum(manager.file_paths['A'][5*i_spec],LFC='HARPS')
        #print(type(orders))
        xdata,ydata,edata,bdata,barycenters =spec.cut_lines(orders,nobackground=False,
                                          columns=['pixel','flux','error','bkg','bary'])
        
        #barycenters = spec.get_barycenters(orders)    
        for o,order in enumerate(orders):
            idxs = get_idxs(barycenters,order,i_spec)
            #print("\t",order)
            # counting to know when we switch to the next segment
            
            if first_iteration:
                maxima      = spec.get_extremes(order,scale='pixel',extreme='max')['y']
                lines_1g    = spec.fit_lines(order,model='singlegaussian')
                #lines_2g    = spec.fit_lines(order,model='simplegaussian')
                if (len(barycenters[order])!=len(lines_1g)):
                    print("Spec {}, order {}, len(bary) {}, len(xdata){}".format(i_spec,order,len(barycenters[order]),len(lines_1g)))
            for i in range(len(barycenters[order])):
                
                line_pix = xdata[order][i]
                line_flx = ydata[order][i]
                line_err = edata[order][i]
                line_bkg = bdata[order][i]
                line_flx_nobkg = line_flx-line_bkg
                #print((4*("{:>8d}")).format(*[len(arr) for arr in [line_pix,line_flx,line_err,line_bkg]]))                
                idx = idxs[i]
                
                # cen is the center of the ePSF!
                # all lines are aligned so that their cen aligns
                if first_iteration:
                    b        = barycenters[order][i]
                    cen      = barycenters[order][i]
                    cen_err  = 0
                    flux     = maxima.iloc[i]
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
                #-------- LINE POSITIONS & FLUX --------
                data['line'].loc[dict(ax='pos',od=order,idx=idx,pix=pix)]=line_pix
                data['line'].loc[dict(ax='flx',od=order,idx=idx,pix=pix)]=line_flx
                data['line'].loc[dict(ax='err',od=order,idx=idx,pix=pix)]=line_err
                data['line'].loc[dict(ax='bkg',od=order,idx=idx,pix=pix)]=line_bkg
                data['line'].loc[dict(ax='x',od=order,idx=idx,pix=pix)]  =line_pix-cen
                #-------- SHOULD WE USE PEAK FLUX OR NORMALISE TO ONE? ------
                #data['line'].loc[dict(ax='y',od=order,idx=idx,pix=pix)]  =(line_flx)/peakflux
                data['line'].loc[dict(ax='y',od=order,idx=idx,pix=pix)]  =(line_flx_nobkg)/np.sum(line_flx_nobkg)
                #data['line'].loc[dict(ax='y',od=order,idx=idx,pix=pix)]  =(line_flx-line_bkg)/peakflux
                if first_iteration:
                    fitpars_1g   = lines_1g[['amplitude','cen','sigma']].iloc[i]
                    line_1g      = h.SingleGaussian(line_pix,line_flx)
                    model_1g     = line_1g.evaluate(pars=fitpars_1g,clipx=False)
                    #model_g1     = amp*np.exp(-0.5*((line_pix-cen)/sigma)**2)
                    resid_1g     = line_flx_nobkg - model_1g
                    data['gauss'].loc[dict(ng=1,ax='mod',od=order,idx=idx,pix=pix)]=model_1g
                    data['gauss'].loc[dict(ng=1,ax='rsd',od=order,idx=idx,pix=pix)]=resid_1g
#                    try:
#                        fitpars_2g   = lines_2g[['amplitude1','center1','sigma1',
#                                             'amplitude2','center2','sigma2']].loc[i]
#                    except:
#                        fitpars_2g   = (0,0,0,0,0,0)
#                    line_2g      = h.DoubleGaussian(line_pix,line_flx)
#                    model_2g     = line_2g.evaluate(pars=fitpars_2g,clipx=False)
#                    resid_2g     = line_flx_nobkg - model_2g
#                    data['gauss'].loc[dict(ng=2,ax='mod',od=order,idx=idx,pix=pix)]=model_2g
#                    data['gauss'].loc[dict(ng=2,ax='rsd',od=order,idx=idx,pix=pix)]=resid_2g
            pbar.update(1)
    pbar.close()
    #data.to_netcdf(path=filepath,mode='a')
    return data
def construct_ePSF(data):
    
    #data     = xr.open_dataset(filepath,chunks=chunks)
    
    n_iter   = 5
    orders   = data.coords['od'].values
    segments = np.unique(data.coords['seg'].values)
    N_seg    = len(segments)
    pixels   = data.coords['pix'].values
    N_sub    = round(len(pixels)/(pixels.max()-pixels.min()))
    
    clip     = 2.5
    
    plot = False
    pbar     = tqdm.tqdm(total=(len(orders)*N_seg),desc='Constructing ePSF')
    if plot:
        fig, ax = h.get_fig_axes(N_seg,alignment='grid')
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
            # initialise effective PSF of this segment as null values    
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] = 0
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')] = x_coords
            delta_x    = 0
            sum_deltax = 0
            
            
            while j<n_iter:
                if np.isnan(delta_x):
                    print("delta_x is NaN!")
                    
                    return data
                # read the latest ePSF array for this order and segment, drop NaNs
                epsf_y  = data['epsf'].sel(od=order,seg=n,ax='y',pix=x_coords)
                epsf_x  = data['epsf'].sel(od=order,seg=n,ax='x',pix=x_coords)
                # construct the spline using x_coords and current ePSF, 
                # evaluate ePSF for all points and save values and residuals
                splr = splrep(epsf_x.values,epsf_y.values)                    
                sple = splev(x_data.values,splr)
#                print(sple)
                #print(len(line_idx),len(sple))
                rsd  = (y_data-sple)
                data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='psf')] = sple
                data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='rsd')] = rsd.values
                # calculate the mean of the residuals between the samplings and ePSF
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
                # adjust current model of the ePSF by the mean of the residuals
                data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')]  = x_coords
                data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] += rsd_mean
                # re-read the new ePSF model: 
                epsf_y = data['epsf'].sel(od=order,seg=n,ax='y',pix=x_coords)
                epsf_x = data['epsf'].sel(od=order,seg=n,ax='x',pix=x_coords)
                epsf_c = epsf_y.coords['pix']
#                print(epsf_x)
#                print(epsf_y)
                # calculate the derivative of the new ePSF model
                epsf_der = xr.DataArray(h.derivative1d(epsf_y.values,epsf_x.values),coords=[epsf_c],dims=['pix'])
                data['epsf'].loc[dict(od=order,seg=n,ax='der',pix=epsf_c)] =epsf_der
                # calculate the shift to be applied to all samplings
                # evaluate at pixel e
                e = 0.5
                epsf_neg     = epsf_y.sel(pix=-e,method='nearest').values
                epsf_pos     = epsf_y.sel(pix=e,method='nearest').values
                epsf_der_neg = epsf_der.sel(pix=-e,method='nearest').values
                epsf_der_pos = epsf_der.sel(pix=e,method='nearest').values
                delta_x      = (epsf_pos-epsf_neg)/(epsf_der_pos-epsf_der_neg)
#                print((5*("{:>8.5f}")).format(float(epsf_neg), 
#                                              float(epsf_pos), 
#                                              float(epsf_der_neg), 
#                                              float(epsf_der_pos),
#                                              delta_x))
                #print(epsf_der)
                
                if plot:
#                        epsf_x0 = data['epsf'].sel(ax='x',seg=n,od=order).dropna('pix','all')
#                        epsf_y0 = data['epsf'].sel(ax='y',seg=n,od=order).dropna('pix','all')
                    ax[n].scatter(epsf_x.values,epsf_y.values,marker='s',s=10,c='C{}'.format(j+1)) 
                    ax[n].axvline(0,ls='--',lw=1,c='C0')
                    ax[n].scatter(x_data.values,y_data.values,s=1,c='C{}'.format(j),marker='s',alpha=0.5)
#                    ax[n].fill_between(x_coords,
#                                      epsf_y+clip*rsd_sigma,
#                                      epsf_y-clip*rsd_sigma,
#                                      alpha=0.3,
#                                      color='C{}'.format(j))
                        
                j+=1               
                # shift the sampling by delta_x for the next iteration
                x_data += delta_x
                # add delta_x to total shift over all iterations
                sum_deltax += delta_x
                if np.isnan(delta_x):
                    
                    print("delta_x is NaN!")
                    print(x_data)
            data['shft'].loc[dict(seg=n,od=order)] = sum_deltax
            #print("{0:2d}{1:>10.6f}".format(n,sum_deltax))
            #print("{0:=^20}".format(""))
            pbar.update(1)
            # save the recentered positions (in ePSF pixel frame)
            #data['line'].loc[dict(od=order,sg=n,ax='x')] += sum_deltax
    pbar.close()
    #data.to_netcdf(path=filepath,mode='a')
    #print('Finished ePSF construction')
    return data
def initialize_dataset(orders,N_seg,N_sub,n_spec):
    nOrders = len(orders)
   
    # number of pixels each ePSF comprises of
    npix   = 17
    # make the subsampled grid where ePSF will be tabulated
    a      = divmod(npix,2)
    xrange = (-a[0],a[0]+a[1])
    pixels    = np.arange(xrange[0],xrange[1],1/N_sub)
    # assume each segment contains 60 lines (not true for large segments!)
    lines_per_seg = 60
    # create a multi-index for data storage
    mdix      = pd.MultiIndex.from_product([np.arange(N_seg),
                                        np.arange(n_spec),
                                        np.arange(lines_per_seg)],
                            names=['sg','sp','id'])
    ndix      = n_spec*N_seg*lines_per_seg
    # axes for each line
    # x = x coords in ePSF reference frame
    # y = ePSF estimates in ePSF reference frame
    # pos = x coords on CCD
    # flx = counts extracted from CCD
    # err = sqrt(flx), backgroud included
    # psf = ePSF estimate for this line
    # rsd = residuals between the model and flux
    # der = derivatives of ePSF
    # w   = weigths 
    # mod = model of the line
    axes   = ['x','y','pos','flx','err','bkg','psf','rsd','der','w','mod']
    n_axes = len(axes)
    # values for each parameter
    values = ['cen','cen_err','flx','flx_err','sft','phi','bary','cen_1g']
    n_vals = len(values)
    # create xarray Dataset object to save the data
    data0   = xr.Dataset({'line': (['od','pix','ax','idx'], np.full((nOrders,npix*N_sub,n_axes,ndix),np.nan)),
                     'resd': (['od','seg','pix'],np.full((nOrders,N_seg,npix*N_sub),np.nan)),
                     'epsf': (['od','seg','pix','ax'], np.full((nOrders,N_seg,npix*N_sub,n_axes),np.nan)),
                     'shft': (['od','seg'], np.full((nOrders,N_seg),np.nan)),
                     'pars': (['od','idx','val'], np.full((nOrders,ndix,n_vals),np.nan)),
                     'gauss': (['ng','od','pix','ax','idx'], np.full((2,nOrders,npix*N_sub,n_axes,ndix),np.nan))},
                     coords={'od':orders, 
                             'idx':mdix, 
                             'pix':pixels,
                             'seg':np.arange(N_seg),
                             'ax' :axes,
                             'val':values,
                             'ng':[1,2]})
#    now = datetime.datetime.now()
#    filepath = os.path.join(h.harps_dtprod,
#                            "epsf",
#                            "{}.nc".format(now.strftime("%Y-%m-%d&%H%M")))
    #data0.to_netcdf(path=filepath)
    return data0

def return_ePSF(manager,niter=1,interpolate=True,
                orders=None,N_seg=16,N_sub=4,n_spec=None,fit_gaussians=False):
    orders = orders if orders is not None else [45]
    
    n_spec  = n_spec if n_spec is not None else manager.numfiles[0]
    data0   = initialize_dataset(orders,N_seg,N_sub,n_spec)
#    chunks  = {'idx':100}
#    data0   = data0.chunk(chunks)
    data    = stack_lines_from_spectra(manager,data0,first_iteration=True,fit_gaussians=fit_gaussians) 
    #data     = xr.open_dataset(filepath,chunks=chunks)
    j = 0
    data_with_pars = data_with_ePSF = data_recentered = data
    plot_epsf = False
    plot_cen  = False
    if plot_epsf:
        fig_epsf,ax_epsf = h.get_fig_axes(N_seg,alignment='grid',title='PSF iteration')
    if plot_cen:
        fig_cen,ax_cen = h.get_fig_axes(1,title='Centeroid shifts')
    while j < niter:
        
       
        
        data_with_ePSF  = construct_ePSF(data_recentered)
        data_with_pars  = solve(data_with_ePSF,interpolate)
        data_recentered = stack_lines_from_spectra(manager,data_with_pars,False)       
         
#        if plot_epsf:
#            midx = data.coords['idx'].values
#            for idx in midx:
#                sg, sp, li = idx
#                if j>0:
#                    data_s = data_with_pars['shft'].sel(seg=sg)
#                else:
#                    data_s = 0
#                data_x = data_with_pars['line'].sel(ax='x',idx=idx).dropna('pix')
#                data_y = data_with_pars['line'].sel(ax='y',idx=idx).dropna('pix')
#                ax_epsf[sg].scatter(data_x+data_s,data_y,s=1,c='C{}'.format(j),marker='s',alpha=0.3)
#                
#        if plot_epsf:
#            for n in range(N_seg):
#                epsf_x = data_with_ePSF['epsf'].sel(ax='x',seg=n).dropna('pix','all')
#                epsf_y = data_with_ePSF['epsf'].sel(ax='y',seg=n).dropna('pix','all')
#                ax_epsf[n].scatter(epsf_x,epsf_y,marker='x',s=20,c='C{}'.format(j),label='{}'.format(j)) 
#        if plot_cen:
#            barycenters = data_recentered['pars'].sel(val='bary')
#            centers     = data_recentered['pars'].sel(val='cen')
#            rel_shift   = (centers/barycenters) - 1
#            ax_cen[0].scatter(barycenters,rel_shift,c="C{}".format(j),s=3)
        j +=1
    final_data = data_with_pars
    return final_data
#%% ============================ PLOTTERS ====================================
def plot_residuals(data,plotter=None,spectra=None,model=None,normed=False,**kwargs):
    # line has format data.sel(od=order,idx=idx)
    orders = data.coords['od'].values
    #midx = data.coords['idx'].values
    if spectra is None:
        spectra = np.unique(data.coords['sp'].values)
    else:
        if type(spectra) == np.int:
            spectra = [spectra]
        elif type(spectra) == list:
            spectra = spectra
    n_spec  = len(spectra)
    models = to_list(model)
    cmap   = plt.get_cmap('Set1')
    colors = {model:cmap(x) for x,model in enumerate(models)}
    if plotter is None:
        plotter = h.SpectrumPlotter(naxes=n_spec,alignment='vertical',
                                    bottom=0.12,**kwargs)
    else:
        pass
#    figure = plotter.figure
    axes = plotter.axes
    ms=1
    model_data = get_model_data(data,model=models,axes=['rsd','pos','flx'])
    #print(model_data)
    for sp in spectra:
        for model in models:
            for order in orders:
                positions = model_data['epsf']['pos'].sel(od=order).dropna('idx','all')
                resids    = model_data[model]['rsd'].sel(od=order).dropna('idx','all')
                flux      = model_data['epsf']['flx'].sel(od=order).dropna('idx','all')
                ms = 1
                if not normed:
                    axes[sp].scatter(positions,resids,s=ms,label=model,c=colors[model])
                else:
                    axes[sp].scatter(positions,resids/flux,s=ms,label=model,c=colors[model])
    #[axes[0].axvline(segcen,lw=0.3,ls=':') for segcen in segment_centers]
    axes[0].legend()       
    return plotter
#%%
#def plot_residuals2(data,plotter=None,spectra=None,**kwargs):
#    # line has format data.sel(od=order,idx=idx)
#    orders = data.coords['od'].values
#    midx = data.coords['idx'].values
#    if spectra is None:
#        spectra = np.unique(data.coords['sp'].values)
#    else:
#        if type(spectra) == np.int:
#            spectra = [spectra]
#        elif type(spectra) == list:
#            spectra = spectra
#    n_spec  = len(spectra)
#    if plotter is None:
#        plotter = h.SpectrumPlotter(naxes=n_spec,alignment='vertical',
#                                    bottom=0.12,**kwargs)
#    else:
#        pass
#    #figure, axes = plotter.figure, 
#    axes = plotter.axes
#    for order in orders:
#        
#        for idx in midx:
#            sg, sp, lix     = idx
#            if sp not in spectra: 
#                continue
#            else:
#                pass
#            line            = data.sel(idx=idx,od=order)
#            cen,flx,sft,phi,b = line['pars']
#            
#            line_pix        = line['line'].sel(ax='pos').dropna('pix')    
#            if len(line_pix)==0:
#                continue
#            line_flx        = line['line'].sel(ax='flx').dropna('pix')
#            line_bkg        = line['line'].sel(ax='bkg').dropna('pix')
#            #line_psf        = flx * line['line'].sel(ax='psf').dropna('pix')
#            #line_err        = line['line'].sel(ax='err').dropna('pix')
#            
#            cen_pix         = line_pix[np.argmax(line_flx.values)]
#            
#            epsf_x          = line['epsf'].sel(seg=sg,ax='x').dropna('pix')+cen_pix-sft
#            epsf_y          = line['epsf'].sel(seg=sg,ax='y').dropna('pix')
#            splr            = splrep(epsf_x.values,epsf_y.values)
#            model           = flx.values * splev((line_pix).values,splr) + line_bkg
#            
#            line_rsd = model-line_flx
#            print("{0:>3d}{1:>3d}{2:>3d}{3:>5d} RMS(residuals): {4:>10.3f}".format(*idx,int(cen_pix.values),h.rms(line_rsd.values)))
#            #ms = 1
#            #widths = 1
#
#            #axes[0].plot(line_pix,line_flx,marker='d',ms=2,label='real',c='C0')
#            #axes[sp].scatter(epsf_x,flx*epsf_y,s=ms,label='epsf',c='C0')
#            #axes[sp].scatter(line_pix,model,s=10,label='model',marker='X',c='C1')
#            axes[sp].scatter(line_pix,line_rsd,s=3)
#    #[axes[0].axvline(segcen,lw=0.3,ls=':') for segcen in segment_centers]
#            
#    return plotter
def plot_hist(data,plotter=None,model=None,spectra=None,separate_spectra=False,
              orders=None, separate_orders=False, **kwargs):
    model = model if model is not None else 'epsf'
    models = to_list(model)
    keys = {'epsf':'cen','gauss':'cen_gauss'}
    
    
    segments  = np.unique(data.coords['seg'].values)
    #midx = data.coords['idx'].values
    
    bins = kwargs.pop('bins',None)
    histtype=kwargs.pop('histtype','step')
    xrange=kwargs.pop('range',None)
    normed = kwargs.pop('normed',None)
    label  = kwargs.pop('label',None)
    if orders is None:
        orders  = data.coords['od'].values
    else:
        orders  = to_list(orders)
    if spectra is None:
        spectra = np.unique(data.coords['sp'].values)
    else:
        spectra = to_list(spectra)
    iterables = {}
    if separate_spectra is True:
        naxes = len(spectra)
        if naxes>2:
            figsize = (12,12)
        elif naxes>5:
            figsize = (18,18)
        iterables['sp']=spectra
    if separate_orders is True:
        naxes = len(orders)
        if naxes>2:
            figsize = (12,12)
        elif naxes>5:
            figsize = (18,18)
        iterables['od']=orders
    else:
        naxes = 1
        figsize = (9,9)
    if plotter is None:
        plotter = h.SpectrumPlotter(naxes=naxes,figsize=figsize,
                                    alignment='grid',
                                    left=0.15,
                                    bottom=0.12,**kwargs)
    else:
        pass
    
    axes = plotter.axes
    
    resids = get_model_data(data,model=models,axes='rsd')
    iters = get_list_of_iters(iterables)
    if len(iters)>0:
        c = -1
        for it in iters:
            c+=1
            locdict=it
            for model in models:
                resids2 = resids[model]['rsd']
                if model == 'epsf':
                    resid    = resids2.loc[locdict]#.values.ravel()
                elif model == '1g':
                    resid = resids2.sel(ng=1).loc[locdict]#.values.ravel()
                elif model == '2g':
                    resid = resids2.sel(ng=2).loc[locdict]#.values.ravel()
                resid    = resid[~np.isnan(resids)]
                
                axes[c].hist(resids,bins,histtype=histtype,range=xrange,normed=normed,label=label+model)
    else: 
        for model in models:
            resids2 = resids[model]['rsd']     
            resid    = resids2.values.ravel()
            resid    = resid[~np.isnan(resid)]
            if label is not None:
                label2="{0:>4} {1:>4}".format(model,label)
            else:
                label2=model
            axes[0].hist(resid,bins,histtype=histtype,range=xrange,normed=normed,label=label2)
    #[axes[0].axvline(segcen,lw=0.3,ls=':') for segcen in segment_centers]
    axes[0].legend()
    return plotter
#%%
def plot_fits(data,plotter=None,spectra=None,model='epsf',**kwargs):
    # line has format data.sel(od=order,idx=idx)
    orders = data.coords['od'].values
    midx = data.coords['idx'].values
    if spectra is None:
        spectra = np.unique(data.coords['sp'].values)
    else:
        if type(spectra) == np.int:
            spectra = [spectra]
        elif type(spectra) == list:
            spectra = spectra
    n_spec  = len(spectra)
    
    
    if plotter is None:
        plotter = h.SpectrumPlotter(naxes=n_spec,alignment='vertical',
                                    bottom=0.12,**kwargs)
    else:
        pass
    figure, axes = plotter.figure, plotter.axes
    
    models = to_list(model)
    data4plot = get_model_data(data,model,['mod','rsd'])
    for order in orders:
        print("{0:>39}{1:>10}".format(*models))
        for idx in midx:
            sg, sp, lix     = idx
            if sp not in spectra: 
                continue
            else:
                pass
            line            = data.sel(idx=idx,od=order)
            pars            = ['cen','flx','sft','cen_gauss']
            cen,flx,sft,cen_gauss = line['pars'].sel(val=pars)
            
            line_pix        = line['line'].sel(ax='pos').dropna('pix')    
            if len(line_pix)==0:
                continue
            line_flx        = line['line'].sel(ax='flx').dropna('pix')
            line_bkg        = line['line'].sel(ax='bkg').dropna('pix')
            #line_err        = line['line'].sel(ax='err').dropna('pix')
            
            cen_pix         = line_pix[np.argmax(line_flx.values)]
            
#            epsf_x          = line['epsf'].sel(seg=sg,ax='x').dropna('pix')+cen_pix-sft
#            epsf_y          = line['epsf'].sel(seg=sg,ax='y').dropna('pix')
#            splr            = splrep(epsf_x.values,epsf_y.values)
            line_rms = []
            for j,model in enumerate(models):
                #line_model   = line['line'].sel(ax='mod').dropna('pix')
                line_model   = data4plot[model]['mod'].sel(od=order,idx=idx).dropna('pix') + line_bkg
                line_rsd     = line_model-line_flx
                line_rms.append(float(h.rms(line_rsd)))
                line_x       = line_pix.sel(pix=line_model.coords['pix'])
            #ms = 1
            #widths = 1

                axes[sp].plot(line_pix,line_flx,marker='d',ms=2,label='real',c='C0')
                #axes[sp].scatter(epsf_x,flx*epsf_y,s=ms,label='epsf',c='C0')
                axes[sp].scatter(line_x,line_model,s=10,label='model',marker='X',c='C{}'.format(j+1))
                #axes[sp].scatter(line_pix,line_rsd,s=3)
            print((3*("{:>3d}")).format(*idx),
                      "{0:>5d}".format(int(cen_pix.values)),
                      "RMS(residuals):",
                  (2*("{:>10.3f}")).format(*line_rms))
    return plotter

#%% 
def plot_velocity_difference(data,plotter=None,spectra=None,model=None,
                             xaxis='bary',**kwargs):
    #orders = data.coords['od'].values
    #midx = data.coords['idx'].values
    model = model if model is not None else ['epsf','gauss']
    if type(model)==str:
        models = [model]
    elif type(model)==list:
        models = model
    keys = {'epsf':'cen','gauss':'cen_gauss'}
    if spectra is None:
        spectra = np.unique(data.coords['sp'].values)
    else:
        if type(spectra) == np.int:
            spectra = [spectra]
        elif type(spectra) == list:
            spectra = spectra
    n_spec  = len(spectra)
    if plotter is None:
        plotter = h.SpectrumPlotter(naxes=n_spec,alignment='vertical',
                                    bottom=0.12,**kwargs)
    else:
        pass
    figure, axes = plotter.figure, plotter.axes
    
    centers = get_model_data(data,model,pars='cen')
    for sp in spectra:
        barycenters = data['pars'].sel(val='bary',sp=sp).dropna('idx','all')
        cen_epsf  = centers['epsf']['cen'].dropna('idx','any')
        cen_gauss = centers['gauss']['cen'] .dropna('idx','any')
        #gauss_cents = data['pars'].sel(val='cen_gauss',sp=sp).dropna('idx','all')
        #psf_centers = data['pars'].sel(val='cen',sp=sp).dropna('idx','all')
        axes[sp].scatter(cen_epsf,829*(cen_gauss-cen_epsf),s=3)
        axes[sp].set_ylabel("$\Delta$ center [m/s] \n(Gauss-ePSF)")
    axes[-1].set_xlabel("ePSF center")
    [axes[0].axvline(i*512,ls=':',lw=0.3) for i in range(9)]
    axes[0].legend()
    return plotter
#%%
def plot_ppe(data,fig=None,model=None):
    models   = to_list(model)
    orders   = data.coords['od'].values
    segments = np.unique(data.coords['seg'].values)
    N_seg    = len(segments)
    
    # pixel at which epsf is centered:
    e = 1.5
    
    # calculate mean positions from n_spec
    if fig is None:
        fig,ax = h.get_fig_axes(1)
    else:
        fig = fig
        ax  = fig.get_axes()
    data4plot = get_model_data(data,model=model,pars='cen')
    for model in models:
        centers = data4plot[model]['cen']
        for order in orders:
            #segment  = data['pars'].sel(od=order).unstack('idx')
            real_pos = centers.sel(od=order)
            mean_pos = real_pos.mean('sp')
            res = (real_pos - mean_pos).dropna('id','any')
            phi = real_pos-((real_pos+e).dropna('id','any')).astype(int)
            #phi = segment.sel(val='phi').dropna('id','all')
            ax[0].scatter(phi,res,alpha=0.3,s=1,label=model)
    ax[0].legend()
    return fig
#%%
def plot_epsf(data,fig=None):
    orders   = data.coords['od'].values
    segments = np.unique(data.coords['seg'].values)
    ids = np.unique(data.coords['id'].values)
    sgs = np.unique(data.coords['sg'].values)
    sps = np.unique(data.coords['sp'].values)
    
    midx  = pd.MultiIndex.from_product([sgs,sps,np.arange(60)],
                            names=['sg','sp','id'])
    if fig is None:
        fig,ax = h.get_fig_axes(len(segments),alignment='grid')
    else:
        fig = fig
        ax  = fig.get_axes()
    for order in orders:
        for idx in midx:
            sg, sp, li = idx
            data_s = data['shft'].sel(seg=sg,od=order)
            data_x = data['line'].sel(ax='x',sg=sg,sp=sp,id=li).dropna('pix')
            data_y = data['line'].sel(ax='y',sg=sg,sp=sp,id=li).dropna('pix')
            
            ax[sg].scatter(data_x+data_s,data_y,s=1,c='C0',marker='s',alpha=0.3)
        for n in segments:
            epsf_x = data['epsf'].sel(ax='x',seg=n).dropna('pix','all')
            epsf_y = data['epsf'].sel(ax='y',seg=n).dropna('pix','all')
            ax[n].scatter(epsf_x,epsf_y,marker='x',s=20,c='C1') 
    return fig 
def plot_psf(psf_ds,order=None,fig=None,**kwargs):
    if order is not None:
        orders = to_list(order)
    else:
        orders   = psf_ds.coords['od'].values
    segments = np.unique(psf_ds.coords['seg'].values)
    # provided figure?
    if fig is None:
        fig,ax = h.get_fig_axes(len(segments),alignment='grid')
    else:
        fig = fig
        ax  = fig.get_axes()
    # colormap
    if len(orders)>5:
        cmap = plt.get_cmap('jet')
    else:
        cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0,1,len(orders)))
    # line parameters
    ls = kwargs.pop('ls','')
    lw = kwargs.pop('lw',0.3)
    for o,order in enumerate(orders):
        for n in segments:
            epsf_y = psf_ds.sel(ax='y',seg=n,od=order).dropna('pix','all')
            epsf_x = psf_ds.sel(ax='x',seg=n,od=order).dropna('pix','all')
            #print(epsf_y, epsf_x)
            ax[n].plot(epsf_x,epsf_y,c=colors[o],ls=ls,lw=lw,marker='x',ms=3) 
            ax[n].set_title('{0}<pix<{1}'.format(n*256,(n+1)*256), )
    return fig 
#%%   
def plot_line(data,idx,model=None):
    sg, sp, lix     = idx
    line            = data.sel(idx=idx,od=45)
    cen,cen_err,flx,flx_err,sft,phi,b = line['pars']
    
    model = model if model is not None else 'epsf'
    if type(model)==list:
        models = model
    else:
        models = [model]
    
    
    line_flx        = line['line'].sel(ax='flx').dropna('pix')
    line_pix        = line['line'].sel(ax='pos').dropna('pix')     
    line_bkg        = line['line'].sel(ax='bkg').dropna('pix')
    line_flx        = line_flx #- line_bkg
    line_psf        = flx * line['line'].sel(ax='psf').dropna('pix')
    line_err        = line['line'].sel(ax='err').dropna('pix')
    cen_pix         = line_pix[np.argmax(line_flx.values)]
    
    epsf_x          = line['epsf'].sel(seg=sg,ax='x').dropna('pix')+cen_pix-sft
    epsf_y          = line['epsf'].sel(seg=sg,ax='y').dropna('pix')
    splr            = splrep(epsf_x.values,epsf_y.values)
    
    line_models     = {}
    line_resids     = {}
    if 'epsf' in models:
        line_model      = line['line'].sel(ax='mod').dropna('pix') + line_bkg
        line_models['epsf'] = line_model
        line_resids['epsf'] = line_model - line_flx
    if 'gauss' in models:
        line_model      = line['gauss'].sel(ax='mod').dropna('pix')+line_bkg
        line_models['gauss'] = line_model
        line_resids['gauss'] = line_model - line_flx
    
    #model           = flx.values * splev((line_pix).values,splr) + line_bkg
    
    line_rsd = model-line_flx
    print("RMS of residuals: {0:8.5f}".format(h.rms(line_rsd.values)))
    fig, ax = h.get_fig_axes(3,figsize=(12,9),ratios=[1,4,1],sharex=True,
                             alignment='vertical',left=0.15,sep=0.01)
    ms = 1
    widths = 1
    lp = labelpad = 10
    
    ax[1].bar(line_pix,line_flx,
          widths,align='center',alpha=0.3,color='C0')
    ax[1].errorbar(line_pix,line_flx,
                       yerr=line_err,fmt='o',color='C0',ms=5)
    ax[1].scatter(line_pix,line_flx,s=ms,label='')
    ax[1].scatter(line_pix,line_psf+line_bkg,s=ms,label='epsf')
    for model in models:
        line_mod = line_models[model]
        line_rsd = line_resids[model]
        ax[1].scatter(line_pix,line_mod,s=20,label=model,marker='X')
        ax[0].scatter(line_pix,line_rsd,s=3) 
        ax[2].scatter(line_pix,line_rsd/line_err,c="C0",s=5)  
    ax[1].axvline(cen,ls='--',lw=0.3)
    ax[1].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    lo = Labeloffset(ax[1],label='Intensity [counts]',axis='y')
    ax[1].legend() 
    

    ax[0].scatter(line_pix,line_rsd,c="C0",s=3)   
    ax[0].set_ylabel("Residuals\n[counts]")
    lims = 1.3*max(abs(line_rsd))
    ax[0].set_ylim(-lims,lims)
    ax[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax[0].axhline(0,ls=':',lw=0.3)
    
    ax[2].axhline(0,ls=':',lw=0.3)
    ax[2].set_ylabel("Residuals [$\sigma$]",labelpad=lp)
    lims = 1.3*max(abs(line_rsd/line_err))
    ax[2].set_ylim(-lims,lims)
    
    [plt.setp(ax[i].get_xticklabels(),visible=False) for i in [0,1]]
def print_stats(data,stat='res',idx=None):
    orders   = data.coords['od'].values
    midx = tuple([idx]) if idx is not None else data.coords['idx'].values
    if stat == 'res':
        selection_lines = data['line'].sel(ax='rsd')
        selection_cen  = data['pars'].sel(val='cen')
    for order in orders:
        sel_od = selection_lines.sel(od=order)
        for idx in midx:
            if ~np.isnan(selection_cen.sel(idx=idx)):
                print((3*("{:>3d}")).format(*idx),
                      "RMS of residuals: {0:10.5f}".format(h.rms(sel_od.sel(idx=idx).dropna('pix','all').values)))
    return 
#%%
def get_model_data(data,model=None,axes=None,pars=None):
    if ((axes is None)and(pars is None)):
        raise Exception("No axes/pars selected")
    else:
        pass
    model = model if model is not None else 'epsf'
    models = to_list(model)
    axes   = to_list(axes)
    pars   = to_list(pars)
    keys   = {'epsf':'line','1g':'gauss','2g':'gauss'}
    
    output = {}#model:{} for model in models}
    for model in models:
        model_dict = {}
        for ax in axes:
            if model=='1g':
                ax_data = data[keys[model]].sel(ax=ax,ng=1)
            elif model=='2g':
                ax_data = data[keys[model]].sel(ax=ax,ng=2)
            else:
                ax_data = data[keys[model]].sel(ax=ax)
            model_dict[ax] = ax_data
        for par in pars:
            if (par == 'cen')and(model=='1g'):
                model_dict[par] = data['pars'].sel(val='cen_1g')
            elif (par=='cen')and(model=='2g'):
                model_dict[par] = data['pars'].sel(val='cen_2g')
            else:
                model_dict[par] = data['pars'].sel(val=par)
        output[model]=model_dict
    return output

def to_list(item):
    if type(item)==str:
        items = [item]
    elif type(item) == list:
        items = item
    elif item is None:
        items = []
    elif type(item)==int:
        items = [item]
    else:
        print("Type provided {}".format(type(item)))
    return items
def get_list_of_iters(dictionary):
    iterkeys = dictionary.keys()
    iters    = itertools.product(*dictionary.values())
    list_of_dictionaries = []
    if len(iterkeys)>0:
        for it in iters:
            locdict = {}
            for k,i in zip(iterkeys,it):
                locdict[k]=i
            list_of_dictionaries.append(locdict)
    return list_of_dictionaries
#%%
    
manager =h.Manager(date='2016-10-23')
nspec = 3
#orders = np.arange(43,72,1)
orders=[60]#,64]
#data_int = return_ePSF(manager,orders=order,niter=3,n_spec=nspec,interpolate=True)
#data_noint = return_ePSF(manager,orders=order,niter=3,n_spec=nspec,interpolate=False)
for order in orders:
    data=return_ePSF(manager,orders=[order],niter=3,n_spec=nspec,interpolate=True,fit_gaussians=False)
    data4file = data.unstack('idx')
    data4file.to_netcdf('/Users/dmilakov/harps/psf_fit/harps_order_{}.nc'.format(order))
#data    = xr.open_dataset(filepath)
#data2 = return_ePSF(manager,niter=1,n_spec=nspec)
#%%
#fig1 = plot_ppe(data1)
#fig2 = plot_ppe(data2,fig1)
#flux1 = solve_for_fluxes(data)
#data2 = return_ePSF(manager,n_spec=nspec,line_fluxes=flux1)
#%%
#fig,ax = h.get_fig_axes(8,alignment='grid')
#midx = data.coords['idx'].values
#for n in range(8):
#    epsf_x = data['epsf'].sel(ax='x',seg=n).dropna('pix','all')
#    epsf_y = data['epsf'].sel(ax='y',seg=n).dropna('pix','all')
#    ax[n].scatter(epsf_x,epsf_y,marker='s',s=10,c='C1') 
#    ax[n].axvline(0,ls='--',lw=1,c='C0')
#for idx in midx:
#    sg, sp, li = idx
#    data_x = data['line'].sel(ax='x',idx=idx).dropna('pix')
#    data_y = data['line'].sel(ax='y',idx=idx).dropna('pix')
#    ax[sg].scatter(data_x,data_y,s=1,c='C0',marker='s')
