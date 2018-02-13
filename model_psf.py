#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:13:13 2018

@author: dmilakov
"""

import harps.utilis as h
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from scipy.interpolate import splrep,splev

manager = manager=h.Manager(date='2016-10-23')
#specfile='/Volumes/dmilakov/harps/data/2016-10-23/HARPS.2016-10-23T12:45:40.889_e2ds_A.fits'
nspec = 2

orders=[45]#,50,55,60,65,70]
nOrders=len(orders)
colours = plt.cm.jet(np.linspace(0, 1, len(orders)))

N_seg      = 8
N_sub      = 4
s          = 4096//N_seg
bins   = np.linspace(0,4096,N_seg+1)
npix   = 17
a      = divmod(npix,2)
xrange = (-a[0],a[0]+a[1])

pixels    = np.arange(xrange[0],xrange[1],1/N_sub)
pixelbins = (pixels[1:]+pixels[:-1])/2

lines_per_seg = 60
mdix      = pd.MultiIndex.from_product([np.arange(N_seg),
                                        np.arange(nspec),
                                        np.arange(lines_per_seg)],
                            names=['sg','sp','id'])
ndix      = nspec*N_seg*lines_per_seg

data   = xr.Dataset({'line': (['od','pix','ax','idx'], np.full((nOrders,npix*N_sub,5,ndix),np.nan)),
                     'bary': (['od','idx'], np.full((nOrders,ndix),np.nan)),
                     'segm': (['od','idx'], np.full((nOrders,ndix),np.nan)),
                     'resd': (['od','seg','pix'],np.full((nOrders,N_seg,npix*N_sub),np.nan)),
                     'epsf': (['od','seg','pix','ax'], np.full((nOrders,N_seg,npix*N_sub,5),np.nan))},
                     coords={#'sp':np.arange(nspec),
                             'od':orders, 
                             'idx':mdix, 
                             'pix':pixels,
                             'seg':np.arange(N_seg),
                             'ax':['x','y','psf','rsd','der']})
#barycenters = xr.DataArray(np.full((nOrders,400,1),np.nan),
#                           coords=[orders,np.arange(400),np.arange(1)],
#                           dims=['od','id','val'])
#barycenters = xr.DataArray(data=np.full((nOrders,400,1),np.nan),
#                            coords={'od':orders,'id':np.arange(400),'val':np.arange(1)},
#                            dims=['od','id','val'])
barycenters = pd.DataFrame(np.full((400,nOrders),np.nan),columns=orders)

#%%
plot = True
if plot:
    figs = [h.get_fig_axes(N_seg,alignment='grid') for i in range(nOrders)]
for i_spec in range(nspec):
    print("SPEC {}".format(i_spec+1))
    spec = h.Spectrum(manager.file_paths['A'][i_spec],LFC='HARPS')
    xdata,ydata=spec.cut_lines(orders)
    for o,order in enumerate(orders):
        print("\t order {}".format(order))
        a = figs[o][1]
        old_j = 0
        k     = 0
        for i in range(np.size(xdata[order])):
            xline = xdata[order][i]
            yline = ydata[order][i]
            b     = np.sum(xline * yline) / np.sum(yline)
            barycenters[order][i]=b
            xline -= b
            yline /= np.sum(yline)
            
            j = int(b//s)
            if j > old_j:
                k = 0
                old_j = j
            else:
                k +=1
            pix = pixels[np.digitize(xline,pixelbins,right=True)]
            data['line'].loc[dict(ax='x',od=order,idx=(j,i_spec,k),pix=pix)]=xline
            data['line'].loc[dict(ax='y',od=order,idx=(j,i_spec,k),pix=pix)]=yline
#            data['bary'].loc[dict(sp=k,od=order,id=i)]=b
            
            
#            data['segm'].loc[dict(sp=k,od=order,id=i)]=j
            if plot:
                a[j].scatter(data['line'].loc[dict(ax='x',od=order,idx=(j,i_spec,k))],
                              data['line'].loc[dict(ax='y',od=order,idx=(j,i_spec,k))],
                              s=1,c=colours[o])
#for i,a in enumerate(ax):
#    a.set_title("{}<x<{}".format(i*s,(i+1)*s),fontsize='small')
    #%%
## FOR TOMORROW:
##    Following §4.2.1 in Anderson & King 2000 paper:
##    Calculate the composite PSF in different sections of the single order spectrum.
##    Bin composite PSF in 5 subpixels. calculate the average value and perform
##    sigma-clipping (2.5) iteratively. find a way to smooth the function
#plot=True
#for o,order in enumerate(orders):
#    a = figs[o][1]
#    for n in range(N):
#        j = 0
#        subdata = data['line'].where(data['segm'].sel(od=order)==n).sel(od=order,ax='y').dropna('pix','all')
#        
#        y_est   = subdata.groupby('pix').mean()
#        y_arr   = y_est.values
#        x_est   = subdata.coords['pix']
#        x_arr   = x_est.values
#        epsf    = np.zeros_like(y_est)
#        while j<5:
#            j+=1
#            # calculate the median of the residuals and the extract the corresponding
#            # coordinates
#            epsf = y_est - epsf
#            # calculate the shift as per eq (9) in Anderson & King
#            epsf_der = xr.DataArray(h.derivative1d(epsf,x_est),coords=[x_arr],dims=['pix'])
#            delta_x = (epsf.sel(pix=0.5)-epsf.sel(pix=-0.5))/(epsf_der.sel(pix=0.5)+epsf_der.sel(pix=-0.5)).values
#            print(delta_x)
#            # shift every grid point by the amount of the shift
#            x_est = x_est + delta_x
#            if plot==True:
#                splr = splrep(x_est,y_est)
#                x_spl = np.linspace(x_est[0],x_est[-1],1000)
#                y_spl = splev(x_spl,splr)
#                a[n].scatter(x_est,y_est,marker='x',s=10,c='C1')
#                a[n].plot(x_spl,y_spl,c='C1')
#                a[n].plot(x_spl,epsf_der,c='C0')
#                [a[n].axvline(p,ls=':',lw=0.3,c='C0') for p in (pixels[1:]+pixels[:-1])/2]
#                
#%%

for o,order in enumerate(orders):
    a = figs[0][1]
    for n in range(N_seg):
        j = 0
        # select flux data for all lines in the n-th segment and the right order
        # drop all NaN values in pixel and 
        segment = data['line'].sel(sg=n,od=order).dropna('pix','all').dropna('idx','all')
        # extract data in x and y
        y_data = segment.sel(ax='y')#.dropna('pix','all').dropna('idx','all')
        x_data = segment.sel(ax='x')#.dropna('pix','all').dropna('idx','all')
        x_coords = y_data.coords['pix'].values
        line_idx = [(n,*t) for t in y_data.coords['idx'].values]
        # initialise effective PSF of this segment as null values         
        while j<5:
#            print(j)
            if j == 0:
                data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] = 0
                data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')] = x_coords
                delta_x    = 0
            # read the latest ePSF array for this order and segment, drop NaNs
            epsf_y  = data['epsf'].sel(od=order,seg=n,ax='y').dropna('pix','all')
            epsf_x  = data['epsf'].sel(od=order,seg=n,ax='x').dropna('pix','all')
            epsf_c  = epsf_y.coords['pix']
            
#            print(j,"\t",epsf_x.values)
#            print(j,"\t",epsf_y.values)
            # move the x_data by delta_x. first iteration, delta_x = 0

            # construct the spline using x_coords and epsf, evaluate it for all data and save
            splr = splrep(epsf_x.values,epsf_y.values)
            
            sple = splev(x_data,splr)
            data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='psf')] = sple
            data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='rsd')] = y_data-sple
            # calculate the mean of the residuals between the data and ePSF
            rsd  = y_data - sple
            rsd_mean = rsd.groupby('pix').mean()
            rsd_coords = rsd_mean.coords['pix']
            # adjust ePSF by the mean of the residuals
#            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')] = rsd_coords
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] += rsd_mean
#            j+=1
#        j = 0
#        while j<5:
            if j==0:
                delta_x = 0
            # calculate the derivative of the ePSF
#            epsf_y = data['epsf'].sel(od=order,seg=n,ax='y')
#            epsf_x = data['epsf'].sel(od=order,seg=n,ax='x')
#            epsf_c = epsf_y.coords['pix']
            if plot: 
                a[n].scatter(epsf_x,epsf_y,marker='x',s=10,c='C{}'.format(j+1))
                [a[n].axvline(p,ls=':',lw=0.3,c='C0') for p in (pixels[1:]+pixels[:-1])/2]
            x_data = x_data + delta_x
            
#            data['line'].loc[dict(idx=line_idx,od=order,pix=x_coords,ax='x')] = x_data
            epsf_der = xr.DataArray(h.derivative1d(epsf_y.values,epsf_x.values),coords=[epsf_c],dims=['pix'])
            delta_x = (epsf_y.sel(pix=0.5)-epsf_y.sel(pix=-0.5))/(epsf_der.sel(pix=0.5)+epsf_der.sel(pix=-0.5)).values
            print(j,"\t",(epsf_y.sel(pix=0.5)-epsf_y.sel(pix=-0.5)).values, (epsf_der.sel(pix=0.5)+epsf_der.sel(pix=-0.5)).values)
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')] += delta_x
            print(j,"\t",delta_x.values)
#            if j == 5:
                
            j+=1

        
#%%
fig_psf,ax_psf = h.get_fig_axes(N_seg,sharex=True,alignment='grid')
for n in range(N_seg):
    segment = data['line'].sel(sg=n,od=order).dropna('pix','all').dropna('idx','all')
    y_data = segment.sel(ax='y')
    x_data = segment.sel(ax='x')
    epsf = data['epsf'].sel(seg=n,od=order,ax='y')#.stack(z=('sp','pix'))
    epsf_coords = data['epsf'].sel(seg=n,od=order,ax='x')
    ax_psf[n].axvline(0,ls='--',lw=0.4)
    ax_psf[n].scatter(x_data,y_data,s=1)
    ax_psf[n].plot(epsf_coords,epsf.T)
    
