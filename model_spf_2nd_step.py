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


#def residuals(x0,line_x,line_y,line_w,splr):
#    # center, flux
#    shift, flux = x0
#    print(shift,flux)
def return_models(data):
    # line has format data.sel(od=order,idx=idx)
    orders = data.coords['od'].values
    midx = data.coords['idx'].values
    for order in orders:
        for idx in midx:
            sg, sp, lix     = idx
            line            = data.sel(idx=idx,od=order)
            cen,flx,sft,phi = line['pars']
            #line_pos        = line['line'].sel(ax='x').dropna('pix')+cen
            line_pix        = line['line'].sel(ax='pos').dropna('pix') 
            #line_dx        = np.unique(line_pos - line_pix)
            
            line_flx        = line['line'].sel(ax='flx').dropna('pix')
            line_psf        = flx * line['line'].sel(ax='psf').dropna('pix')
            
            epsf_x          = line['epsf'].sel(seg=sg,ax='x').dropna('pix')+cen+3.6*sft
            epsf_y          = line['epsf'].sel(seg=sg,ax='y').dropna('pix')
            splr            = splrep(epsf_x.values,epsf_y.values)
            model           = flx.values * splev((line_pix).values,splr)
            
            line_rsd = model-line_flx
            print(line_rsd)
            plt.figure(figsize=(8,8))
            ms = 1
            widths = 1
            plt.bar(line_pix,line_flx,
                  widths,align='center',alpha=0.3,color='C0')
            plt.scatter(line_pix,line_flx,s=ms,label='real')
            plt.scatter(epsf_x,flx*epsf_y,s=ms,label='epsf')
            plt.scatter(line_pix,model,s=ms,label='model')
            #plt.scatter(line_pos,line_psf,s=ms,label='sampling')
            plt.axvline(cen,ls='--',lw=0.3)
            plt.legend()
    
#    model = flux * splev(line_x+shift,splr)
#    resid = line_w * (model - line_y) / line_y
#    return resid
#def solve_for_fluxes(data):
#    num = ((data['line'].sel(ax='w')*data['line'].sel(ax='flx')**2*data['line'].sel(ax='psf')).sum(axis=1))
#    den = (data['line'].sel(ax='w')*data['line'].sel(ax='flx')*data['line'].sel(ax='psf')**2).sum(axis=1)
#    #orders = data['line'].coords['od'].values
#    flx = num/den
#    #print(flx)
#    #line_fluxes = {order:flx.sel(od=order).dropna('idx').values for order in orders}
#    return flx
def solve(data):
    orders          = data.coords['od'].values
    pixels          = data.coords['pix'].values
    midx = data.coords['idx'].values
    segments        = np.unique(data.coords['seg'].values)
    
    def residuals(x0,line_x,line_y,line_w,splr):
        # center, flux
        sft, flux = x0
        model = flux * splev(line_x+sft,splr)
        resid = line_w * (model - line_y) / line_y
        return resid
        
    for order in orders:
        for idx in midx:
            sg,sp,lid = idx
            cen, flx, dx, phi = data['pars'].sel(idx=idx,od=order).values
            p0 = (dx,flx)
            line_x = data['line'].sel(ax='pos',idx=idx,od=order).dropna('pix').values - cen
            line_y = data['line'].sel(ax='flx',idx=idx,od=order).dropna('pix').values
            line_w = data['line'].sel(ax='w',idx=idx,od=order).dropna('pix').values
            
            
            if np.isnan(p0).any() == True:
                continue
            
            
            epsf_x  = data['epsf'].sel(ax='x',od=order,seg=sg).dropna('pix')
            epsf_y  = data['epsf'].sel(ax='y',od=order,seg=sg).dropna('pix')
            #print(np.shape(epsf_x+cen),np.shape(epsf_y))
            splr    = splrep(epsf_x.values,epsf_y.values)
            popt,pcov,infodict,errmsg,ie = leastsq(residuals,x0=p0,
                                                   args=(line_x,line_y,line_w,splr),
                                                   full_output=True)
            
            sft, peakflux = popt
            cen = cen+sft
            phi = cen - int(cen+0.5)
            data['pars'].loc[dict(idx=idx,od=order)] = (cen,peakflux,sft,phi)
            #print(data['pars'].sel(idx=idx,od=order).values)
            # change positions and fluxes of the line
            line_pix = data['line'].sel(idx=idx,od=order).coords['pix'].values
            line_flx = data['line'].sel(idx=idx,od=order,ax='flx',pix=line_pix)
            data['line'].loc[dict(idx=idx,od=order,ax='x',pix=line_pix)] += sft
            data['line'].loc[dict(idx=idx,od=order,ax='y',pix=line_pix)] = line_flx/peakflux
    return data   
#%%
def stack_lines_from_spectra(manager,data,first_iteration=None,n_spec=None):
    n_spec = n_spec if n_spec is not None else manager.numfiles[0] 
    if first_iteration == None:
        # check if data['pars'] is empty
        if np.size(data['pars'].dropna('val','all')) == 0:
            first_iteration = True
        else:
            first_iteration = False
        
    orders          = data.coords['od'].values
    pixels          = data.coords['pix'].values
    pix_step        = pixels[1]-pixels[0]
    pixelbins       = (pixels[1:]+pixels[:-1])/2
    segments        = np.unique(data.coords['seg'].values)
    N_seg           = len(segments)
    s               = 4096//N_seg
    for i_spec in range(n_spec):
        print("SPEC {}".format(i_spec+1))
        spec = h.Spectrum(manager.file_paths['A'][i_spec],LFC='HARPS')
        #print(type(orders))
        xdata,ydata=spec.cut_lines(orders)
        barycenters = spec.get_barycenters(orders)
            
        for o,order in enumerate(orders):
            print("\t",order)
            # counting to know when we switch to the next segment
            old_j = 0
            k     = -1
            if first_iteration:
                maxima      = spec.get_extremes(order,scale='pixel',extreme='max')['y']
            for i in range(np.size(xdata[order])):
#                if first_iteration:
#                    data['pars'].loc[dict(od=order,idx=(j,i_spec,k) = 
#                    line_fluxes   = spec.get_extremes(order,scale='pixel',extreme='max')['y']
#                    line_shifts   = pd.Series(0,index=line_positions.index)
#                else:
#                    line_positions = line_positions.sel(sp=i_spec,od=order).dropna('idx')
#                    line_fluxes   = line_fluxes.sel(sp=i_spec,od=order).dropna('idx')
#                    line_shifts   = 

                line_pix = xdata[order][i]
                line_flx = ydata[order][i]
                b        = barycenters[order][i]     
                # j = segment cardinal number (0-7)
                # k = line cardinal number (0-59 for each segment)
                j = int(b//s)
                if j > old_j:
                    k = 0
                    old_j = j
                else:
                    k +=1
                idx = (j, i_spec, k)
                
                if first_iteration:
                    cen   = barycenters[order][i]
                    flux  = maxima.iloc[i]
                    shift = 0
                    phase = cen - int(cen+0.5)
                else:
                    cen,flux,shift,phase = data['pars'].sel(od=order,idx=idx).values
                peakflux = flux
                data['pars'].loc[dict(idx=idx,od=order)] = cen,flux,shift,phase
                #print(cen,line_pix)
                xline0 = line_pix - cen
                #yline /= np.sum(yline)
                
               
                    
                pix = pixels[np.digitize(xline0,pixelbins,right=True)]
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
                
                data['line'].loc[dict(ax='pos',od=order,idx=idx,pix=pix)]=line_pix
                data['line'].loc[dict(ax='flx',od=order,idx=idx,pix=pix)]=line_flx
                data['line'].loc[dict(ax='x',od=order,idx=idx,pix=pix)]  =line_pix-cen
                data['line'].loc[dict(ax='y',od=order,idx=idx,pix=pix)]  =line_flx/peakflux#line_flx.sum()
    return data
def construct_ePSF(data):
    n_iter   = 5
    orders   = data.coords['od'].values
    segments = np.unique(data.coords['seg'].values)
    pixels   = data.coords['pix'].values
    N_sub    = round(len(pixels)/(pixels.max()-pixels.min()))
    plot = False
    if plot:
        fig, ax = h.get_fig_axes(8,alignment='grid')
    for o,order in enumerate(orders):
        for n in segments:
            j = 0
            # select flux data for all lines in the n-th segment and the right order
            # drop all NaN values in pixel and 
            segment = data['line'].sel(sg=n,od=order).dropna('pix','all').dropna('idx','all')
            # extract data in x and y, corresponding coordinates and line idx
            y_data = segment.sel(ax='y').dropna('pix','all').dropna('idx','all')
            x_data = segment.sel(ax='x').dropna('pix','all').dropna('idx','all')
            x_coords = y_data.coords['pix'].values
            line_idx = [(n,*t) for t in y_data.coords['idx'].values]
            # initialise effective PSF of this segment as null values    
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] = 0
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')] = x_coords
            delta_x    = 0
            sum_deltax = 0
            
            
            while j<n_iter:
                if np.isnan(delta_x):
                    print("delta_x is NaN!")
                    return data
                print("X_DATA:",x_data.min().values,x_data.max().values)
                # read the latest ePSF array for this order and segment, drop NaNs
                epsf_y  = data['epsf'].sel(od=order,seg=n,ax='y',pix=x_coords)
                epsf_x  = data['epsf'].sel(od=order,seg=n,ax='x',pix=x_coords)
                print("EPSF_X:",epsf_x.min().values,epsf_x.max().values)
                print("EPSF_Y:",epsf_y.min().values,epsf_y.max().values)
                # construct the spline using x_coords and current ePSF, 
                # evaluate ePSF for all points and save values and residuals
                splr = splrep(epsf_x.values,epsf_y.values)    
                print(splr)
                sple = splev(x_data,splr)
                data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='psf')] = sple
                data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='rsd')] = y_data-sple
                print("SPLE:",sple.min(),sple.max())
                # calculate the mean of the residuals between the samplings and ePSF
                testbins  = np.array([(l,u) for l,u in zip(x_coords-1/N_sub,x_coords+1/N_sub)])
                rsd  = y_data - sple
                print("RSD:",rsd.min().values,rsd.max().values)
                rsd_array = np.zeros_like(x_coords)
                for i in range(rsd_array.size):
                    llim, ulim = testbins[i]
                    #print(np.isnan(x_data).values.all())
                    #[a[n].axvline(p,ls=':',lw=0.3,c='C0') for p in testbins[i]]
                    rsd_cut = rsd.where((x_data>llim)&(x_data<=ulim)).dropna('pix','all')
                    #print(llim,ulim,(rsd_cut).size)
                    rsd_array[i] = rsd_cut.mean(skipna=True).values
                    #print(rsd_array[i])
                #print(np.isnan(rsd_array))
                rsd_mean = xr.DataArray(rsd_array,coords=[x_coords],dims=['pix'])
                #rsd_mean = rsd.groupby('pix').mean()
                rsd_coords = rsd_mean.coords['pix']
                #print(rsd_mean, rsd_coords)
                # adjust current model of the ePSF by the mean of the residuals
                data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')]  = rsd_coords
                data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] += rsd_mean
                # re-read the new ePSF model: 
                epsf_y = data['epsf'].sel(od=order,seg=n,ax='y')
                epsf_x = data['epsf'].sel(od=order,seg=n,ax='x')
                epsf_c = epsf_x.coords['pix']
    
                # calculate the derivative of the new ePSF model
                epsf_der = xr.DataArray(h.derivative1d(epsf_y.values,epsf_x.values),coords=[epsf_c],dims=['pix'])
                data['epsf'].loc[dict(od=order,seg=n,ax='der')] =epsf_der
                # calculate the shift to be applied to all samplings
                # evaluate at pixel e
                e = 0.5
                epsf_neg     = epsf_y.sel(pix=-e,method='nearest').values
                epsf_pos     = epsf_y.sel(pix=e,method='nearest').values
                epsf_der_neg = epsf_der.sel(pix=-e,method='nearest').values
                epsf_der_pos = epsf_der.sel(pix=e,method='nearest').values
                delta_x      = (epsf_pos-epsf_neg)/(epsf_der_pos-epsf_der_neg)
                print((5*("{:>8.5f}")).format(float(epsf_neg), 
                                              float(epsf_pos), 
                                              float(epsf_der_neg), 
                                              float(epsf_der_pos),
                                              delta_x))
                
                if plot:
#                        epsf_x0 = data['epsf'].sel(ax='x',seg=n,od=order).dropna('pix','all')
#                        epsf_y0 = data['epsf'].sel(ax='y',seg=n,od=order).dropna('pix','all')
                    ax[n].scatter(epsf_x.values,epsf_y.values,marker='s',s=10,c='C{}'.format(j+1)) 
                    ax[n].axvline(0,ls='--',lw=1,c='C0')
                    ax[n].scatter(x_data.values,y_data.values,s=1,c='C{}'.format(j+1),marker='s',alpha=0.3)
                        
                #print("\t",delta_x.values)
                j+=1
                
                # shift the sampling by delta_x for the next iteration
                x_data += delta_x
                # add delta_x to total shift over all iterations
                sum_deltax += delta_x
                if np.isnan(delta_x):
                    
                    print("delta_x is NaN!")
                    print(x_data)
            data['shft'].loc[dict(seg=n,od=order)] = sum_deltax
            print(n,"{0}".format(sum_deltax) )
            # save the recentered positions (in ePSF pixel frame)
            #data['line'].loc[dict(od=order,sg=n,ax='x')] += sum_deltax
    return data
def initialize_dataset(orders,N_seg,N_sub,n_spec):
    # segment size
    nOrders = len(orders)
    # bins
    # bins   = np.linspace(0,4096,N_seg+1)
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
                                        np.arange(nspec),
                                        np.arange(lines_per_seg)],
                            names=['sg','sp','id'])
    ndix      = nspec*N_seg*lines_per_seg
    # create xarray Dataset object to save the data
    data0   = xr.Dataset({'line': (['od','pix','ax','idx'], np.full((nOrders,npix*N_sub,8,ndix),np.nan)),
                     'resd': (['od','seg','pix'],np.full((nOrders,N_seg,npix*N_sub),np.nan)),
                     'epsf': (['od','seg','pix','ax'], np.full((nOrders,N_seg,npix*N_sub,8),np.nan)),
                     'shft': (['od','seg'], np.full((nOrders,N_seg),np.nan)),
                     'pars': (['od','idx','val'], np.full((nOrders,ndix,4),np.nan)) },
                     coords={'od':orders, 
                             'idx':mdix, 
                             'pix':pixels,
                             'seg':np.arange(N_seg),
                             'ax':['x','y','pos','flx','psf','rsd','der','w'],
                             'val':['cen','flx','sft','phi']})
    return data0
def return_ePSF(manager,niter=3,line_positions=None,line_fluxes=None,
                orders=None,N_seg=8,N_sub=4,n_spec=None):
    orders = orders if orders is not None else [45]
    
    n_spec = n_spec if n_spec is not None else manager.numfiles[0]
    if line_positions is None:
        first_iteration = True
    if first_iteration:
        data0 = initialize_dataset(orders,N_seg,N_sub,n_spec)
    
    data1 = stack_lines_from_spectra(manager,data0,first_iteration,n_spec) 
    j = 0
    data_with_pars = data_with_ePSF = data1
    plot = False
    if plot:
        fig,ax = h.get_fig_axes(8,alignment='grid',title='PSF iteration')
    while j < niter:
        
        midx = data_with_pars.coords['idx'].values
        if plot:
            for idx in midx:
                sg, sp, li = idx
                if j>0:
                    data_s = data_with_pars['shft'].sel(seg=sg,od=order)
                else:
                    data_s = 0
                data_x = data_with_pars['line'].sel(ax='x',idx=idx).dropna('pix')
                data_y = data_with_pars['line'].sel(ax='y',idx=idx).dropna('pix')
                ax[sg].scatter(data_x+data_s,data_y,s=1,c='C{}'.format(j),marker='s',alpha=0.3)
             
        data_with_ePSF  = construct_ePSF(data_with_pars) 
        data_recentered = data_with_ePSF#stack_lines_from_spectra(manager,data_with_ePSF,False,n_spec)
        data_with_pars  = solve(data_recentered)
        #plot_ppe(data_with_pars)
        if plot:
            for n in range(8):
                epsf_x = data_with_ePSF['epsf'].sel(ax='x',seg=n).dropna('pix','all')
                epsf_y = data_with_ePSF['epsf'].sel(ax='y',seg=n).dropna('pix','all')
                ax[n].scatter(epsf_x,epsf_y,marker='x',s=20,c='C{}'.format(j),label='{}'.format(j)) 
    #            ax[n].axvline(0,ls='--',lw=1,c='C0')
            
        
        j +=1
#    ax[0].legend()
    final_data = data_with_pars
    return final_data
#%%
def plot_ppe(data,fig=None):
    orders   = data.coords['od'].values
    segments = np.unique(data.coords['seg'].values)
    # calculate mean positions from n_spec
    if fig is None:
        fig,ax = h.get_fig_axes(8,alignment='grid')
    else:
        fig = fig
        ax  = fig.get_axes()
    for order in orders:
        for n in segments:
            segment  = data['pars'].sel(od=order,sg=n).unstack('idx')
            real_pos = segment.sel(val='cen')
            mean_pos = real_pos.mean('sp')
            res = (real_pos - mean_pos).dropna('id','all')
            phi = segment.sel(val='phi').dropna('id','all')
            ax[n].scatter(phi,res,alpha=0.3,s=1)
    return fig
def plot_epsf(data,fig=None):
    orders   = data.coords['od'].values
    segments = np.unique(data.coords['seg'].values)
    midx = data.coords['idx'].values
    if fig is None:
        fig,ax = h.get_fig_axes(len(segments),alignment='grid')
    else:
        fig = fig
        ax  = fig.get_axes()
    for order in orders:
        for idx in midx:
            sg, sp, li = idx
            data_s = data['shft'].sel(seg=sg,od=order)
            data_x = data['line'].sel(ax='x',idx=idx).dropna('pix')
            data_y = data['line'].sel(ax='y',idx=idx).dropna('pix')
            ax[sg].scatter(data_x+data_s,data_y,s=1,c='C0',marker='s',alpha=0.3)
        for n in segments:
            epsf_x = data['epsf'].sel(ax='x',seg=n).dropna('pix','all')
            epsf_y = data['epsf'].sel(ax='y',seg=n).dropna('pix','all')
            ax[n].scatter(epsf_x,epsf_y,marker='x',s=20,c='C1') 
    return fig    
            
#%%
    
manager = manager=h.Manager(date='2016-10-23')
nspec = 1

data1 = return_ePSF(manager,niter=3,n_spec=nspec)
data2 = return_ePSF(manager,niter=1,n_spec=nspec)
#%%
fig1 = plot_ppe(data1)
fig2 = plot_ppe(data2,fig1)
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
