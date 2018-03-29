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

def in_ranges(x,bins):
    # https://stackoverflow.com/questions/44005652/how-do-i-efficiently-bin-values-into-overlapping-bins-using-pandas
    return np.array([((x>=b[0])&(x<=b[1])) for b in bins]).T

def stack_lines_from_spectra(manager,data,line_positions=None,n_spec=None):
    n_spec = n_spec if n_spec is not None else manager.numfiles[0] 
    first_iteration = True if line_positions is None else False
    for i_spec in range(n_spec):
        print("SPEC {}".format(i_spec+1))
        spec = h.Spectrum(manager.file_paths['A'][i_spec],LFC='HARPS')
        xdata,ydata=spec.cut_lines(orders)
        if first_iteration:
            line_positions = spec.get_barycenters(orders)
        for o,order in enumerate(orders):
            print("\t",order)
            # counting to know when we switch to the next segment
            old_j = 0
            k     = 0
            for i in range(np.size(xdata[order])):
                xline = xdata[order][i]
                yline = ydata[order][i]
                
                b     = line_positions[order][i]
                xline -= b
                #yline /= np.sum(yline)
                
                j = int(b//s)
                if j > old_j:
                    k = 0
                    old_j = j
                else:
                    k +=1
                pix = pixels[np.digitize(xline,pixelbins,right=True)]
                data['line'].loc[dict(ax='flx',od=order,idx=(j,i_spec,k),pix=pix)]=yline
                data['line'].loc[dict(ax='x',od=order,idx=(j,i_spec,k),pix=pix)]  =xline
                data['line'].loc[dict(ax='y',od=order,idx=(j,i_spec,k),pix=pix)]  =yline/np.sum(yline)
    return data
def construct_ePSF(data):
    orders   = data.coords['od'].values
    segments = np.unique(data.coords['seg'].values)
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
            # initialise effective PSF of this segment as null values    
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] = 0
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')] = x_coords
            delta_x    = 0
            sum_deltax = 0
            
            while j<n_iter:
                # read the latest ePSF array for this order and segment, drop NaNs
                epsf_y  = data['epsf'].sel(od=order,seg=n,ax='y',pix=x_coords)
                epsf_x  = data['epsf'].sel(od=order,seg=n,ax='x',pix=x_coords)
                # construct the spline using x_coords and current ePSF, 
                # evaluate ePSF for all points and save values and residuals
                splr = splrep(epsf_x.values,epsf_y.values)           
                sple = splev(x_data,splr)
                data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='psf')] = sple
                data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='rsd')] = y_data-sple
                # calculate the mean of the residuals between the samplings and ePSF
                testbins  = np.array([(l,u) for l,u in zip(x_coords-1/N_sub,x_coords+1/N_sub)])
                rsd  = y_data - sple
                rsd_array = np.zeros_like(x_coords)
                for i in range(rsd_array.size):
                    llim, ulim = testbins[i]
                    [a[n].axvline(p,ls=':',lw=0.3,c='C0') for p in testbins[i]]
                    rsd_array[i] = rsd.where((x_data>llim)&(x_data<=ulim)).mean().values
                rsd_mean = xr.DataArray(rsd_array,coords=[x_coords],dims=['pix'])
                #rsd_mean = rsd.groupby('pix').mean()
                rsd_coords = rsd_mean.coords['pix']
                # adjust current model of the ePSF by the mean of the residuals
                data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')]  = rsd_coords
                data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] += rsd_mean
                # re-read the new ePSF model: 
                epsf_y = data['epsf'].sel(od=order,seg=n,ax='y')
                epsf_x = data['epsf'].sel(od=order,seg=n,ax='x')
                epsf_c = epsf_x.coords['pix']
    
                # calculate the derivative of the new ePSF model
                epsf_der = xr.DataArray(h.derivative1d(epsf_y.values,epsf_x.values),coords=[epsf_c],dims=['pix'])
    
                # calculate the shift to be applied to all samplings
                # evaluate at pixel e
                e = 0.5
                epsf_neg     = epsf_y.sel(pix=-e,method='nearest')
                epsf_pos     = epsf_y.sel(pix=e,method='nearest')
                epsf_der_neg = epsf_der.sel(pix=-e,method='nearest')
                epsf_der_pos = epsf_der.sel(pix=e,method='nearest')
                delta_x      = (epsf_pos-epsf_neg)/(epsf_der_pos-epsf_der_neg)
                #print("\t",delta_x.values)
                j+=1
                
                # shift the sampling by delta_x for the next iteration
                x_data += delta_x
                # add delta_x to total shift over all iterations
                sum_deltax += delta_x
            data['shft'].loc[dict(seg=n,od=order)] = sum_deltax
            print(n,"{0}".format(sum_deltax.values) )
    return data
def return_ePSF(manager,line_positions=None,orders=None,N_seg=8,N_sub=4,n_spec=None):
    orders = orders if orders is not None else [45]
    n_spec = n_spec if n_spec is not None else manager.numfiles[0]
    if line_positions is None:
        first_iteration = True
    # segment size
    s          = 4096//N_seg
    # bins
    bins   = np.linspace(0,4096,N_seg+1)
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
    data0   = xr.Dataset({'line': (['od','pix','ax','idx'], np.full((nOrders,npix*N_sub,7,ndix),np.nan)),
                     'resd': (['od','seg','pix'],np.full((nOrders,N_seg,npix*N_sub),np.nan)),
                     'epsf': (['od','seg','pix','ax'], np.full((nOrders,N_seg,npix*N_sub,7),np.nan)),
                     'shft': (['od','seg'], np.full((nOrders,N_seg),np.nan))},
                     coords={'od':orders, 
                             'idx':mdix, 
                             'pix':pixels,
                             'seg':np.arange(N_seg),
                             'ax':['x','y','flx','psf','rsd','der','w']})
    
    data1 = stack_lines_from_spectra(manager,data0,line_positions=None,n_spec)            
    data2 = construct_ePSF(data1)
    return data2
    
#%%
manager = manager=h.Manager(date='2016-10-23')
#specfile='/Volumes/dmilakov/harps/data/2016-10-23/HARPS.2016-10-23T12:45:40.889_e2ds_A.fits'
nspec = 20

orders=[45]#,50,55,60,65,70]
nOrders=len(orders)


N_seg      = 8
N_sub      = 4
s          = 4096//N_seg
bins   = np.linspace(0,4096,N_seg+1)
npix   = 17
a      = divmod(npix,2)
xrange = (-a[0],a[0]+a[1])

colours = plt.cm.jet(np.linspace(0, 1, N_seg))

pixels    = np.arange(xrange[0],xrange[1],1/N_sub)
pixelbins = (pixels[1:]+pixels[:-1])/2
testbins  = np.array([(l,u) for l,u in zip(pixels-1/N_sub,pixels+1/N_sub)])
lines_per_seg = 60
mdix      = pd.MultiIndex.from_product([np.arange(N_seg),
                                        np.arange(nspec),
                                        np.arange(lines_per_seg)],
                            names=['sg','sp','id'])
ndix      = nspec*N_seg*lines_per_seg

data   = xr.Dataset({'line': (['od','pix','ax','idx'], np.full((nOrders,npix*N_sub,7,ndix),np.nan)),
                     'bary': (['od','idx'], np.full((nOrders,ndix),np.nan)),
                     'segm': (['od','idx'], np.full((nOrders,ndix),np.nan)),
                     'resd': (['od','seg','pix'],np.full((nOrders,N_seg,npix*N_sub),np.nan)),
                     'epsf': (['od','seg','pix','ax'], np.full((nOrders,N_seg,npix*N_sub,7),np.nan)),
                     'shft': (['od','seg'], np.full((nOrders,N_seg),np.nan))},
                     coords={#'sp':np.arange(nspec),
                             'od':orders, 
                             'idx':mdix, 
                             'pix':pixels,
                             'seg':np.arange(N_seg),
                             'ax':['x','y','flx','psf','rsd','der','w']})
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
            #yline /= np.sum(yline)
            
            j = int(b//s)
            if j > old_j:
                k = 0
                old_j = j
            else:
                k +=1
            test = in_ranges(xline,testbins)
            if 2 in test:
                print(test)
            pix = pixels[np.digitize(xline,pixelbins,right=True)]
            data['line'].loc[dict(ax='flx',od=order,idx=(j,i_spec,k),pix=pix)]=yline
            data['line'].loc[dict(ax='x',od=order,idx=(j,i_spec,k),pix=pix)]  =xline
            data['line'].loc[dict(ax='y',od=order,idx=(j,i_spec,k),pix=pix)]  =yline/np.sum(yline)
            
            
#            data['bary'].loc[dict(sp=k,od=order,id=i)]=b
            
            
#            data['segm'].loc[dict(sp=k,od=order,id=i)]=j
#            if plot:
#                a[j].scatter(data['line'].loc[dict(ax='x',od=order,idx=(j,i_spec,k))],
#                              data['line'].loc[dict(ax='y',od=order,idx=(j,i_spec,k))],
#                              s=1,c=colours[o])
#for i,a in enumerate(ax):
#    a.set_title("{}<x<{}".format(i*s,(i+1)*s),fontsize='small')

#%%
n_iter = 5
# Create the ePSF 
for o,order in enumerate(orders):
    a = figs[0][1]
    for n in range(N_seg):
        j = 0
        # select flux data for all lines in the n-th segment and the right order
        # drop all NaN values in pixel and 
        segment = data['line'].sel(sg=n,od=order).dropna('pix','all').dropna('idx','all')
        # extract data in x and y, corresponding coordinates and line idx
        y_data = segment.sel(ax='y')#.dropna('pix','all').dropna('idx','all')
        x_data = segment.sel(ax='x')#.dropna('pix','all').dropna('idx','all')
        x_coords = y_data.coords['pix'].values
        line_idx = [(n,*t) for t in y_data.coords['idx'].values]
        # initialise effective PSF of this segment as null values    
        data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] = 0
        data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')] = x_coords
        delta_x    = 0
        sum_deltax = 0
        #if plot:
            #a[n].scatter(x_data,y_data,s=1,c='C0')
            
        while j<n_iter:
            # read the latest ePSF array for this order and segment, drop NaNs
            epsf_y  = data['epsf'].sel(od=order,seg=n,ax='y',pix=x_coords)
            epsf_x  = data['epsf'].sel(od=order,seg=n,ax='x',pix=x_coords)
            # construct the spline using x_coords and current ePSF, 
            # evaluate ePSF for all points and save values and residuals
            splr = splrep(epsf_x.values,epsf_y.values)           
            sple = splev(x_data,splr)
            data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='psf')] = sple
            data['line'].loc[dict(od=order,pix=x_coords,idx=line_idx,ax='rsd')] = y_data-sple
            # calculate the mean of the residuals between the samplings and ePSF
            testbins  = np.array([(l,u) for l,u in zip(x_coords-1/N_sub,x_coords+1/N_sub)])
            rsd  = y_data - sple
            rsd_array = np.zeros_like(x_coords)
            for i in range(rsd_array.size):
                llim, ulim = testbins[i]
                [a[n].axvline(p,ls=':',lw=0.3,c='C0') for p in testbins[i]]
                rsd_array[i] = rsd.where((x_data>llim)&(x_data<=ulim)).mean().values
            rsd_mean = xr.DataArray(rsd_array,coords=[x_coords],dims=['pix'])
            #rsd_mean = rsd.groupby('pix').mean()
            rsd_coords = rsd_mean.coords['pix']
            # adjust current model of the ePSF by the mean of the residuals
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='x')]  = rsd_coords
            data['epsf'].loc[dict(od=order,seg=n,pix=x_coords,ax='y')] += rsd_mean
            # re-read the new ePSF model: 
            epsf_y = data['epsf'].sel(od=order,seg=n,ax='y')
            epsf_x = data['epsf'].sel(od=order,seg=n,ax='x')
            epsf_c = epsf_x.coords['pix']

            # calculate the derivative of the new ePSF model
            epsf_der = xr.DataArray(h.derivative1d(epsf_y.values,epsf_x.values),coords=[epsf_c],dims=['pix'])

            # calculate the shift to be applied to all samplings
            # evaluate at pixel e
            e = 0.5
            epsf_neg     = epsf_y.sel(pix=-e,method='nearest')
            epsf_pos     = epsf_y.sel(pix=e,method='nearest')
            epsf_der_neg = epsf_der.sel(pix=-e,method='nearest')
            epsf_der_pos = epsf_der.sel(pix=e,method='nearest')
            delta_x      = (epsf_pos-epsf_neg)/(epsf_der_pos-epsf_der_neg)
            #print("\t",delta_x.values)
            j+=1
            if plot:
                if j==1:
                    a[n].scatter(epsf_x,epsf_y,marker='s',s=10,c='C1')
                    a[n].scatter(x_data,y_data,s=1,c='C0',marker='s')
                    a[n].axvline(0,ls='--',lw=1,c='C0')
                    

                if j==n_iter:
                    
                    a[n].scatter(epsf_x,epsf_y,marker='x',s=10,c='C3')
                    a[n].scatter(x_data,y_data,s=1,c='C2',marker='x')
            # shift the sampling by delta_x for the next iteration
            x_data += delta_x
            # add delta_x to total shift over all iterations
            sum_deltax += delta_x
        data['shft'].loc[dict(seg=n,od=order)] = sum_deltax
        print(n,"{0}".format(sum_deltax.values) )

        
#%%
fig_psf,ax_psf = h.get_fig_axes(N_seg,sharex=True,sharey=True,alignment='grid')
for n in range(N_seg):
    segment = data['line'].sel(sg=n,od=order).dropna('pix','all').dropna('idx','all')
    shift   = data['shft'].sel(seg=n,od=order)
    y_data = segment.sel(ax='y')
    x_data = segment.sel(ax='x') + shift
    psf_data = segment.sel(ax='psf')
    epsf = data['epsf'].sel(seg=n,od=order,ax='y')#.stack(z=('sp','pix'))
    epsf_coords = data['epsf'].sel(seg=n,od=order,ax='x')
    testbins  = np.array([(l,u) for l,u in zip(epsf_coords-1/N_sub,epsf_coords+1/N_sub)])
    ax_psf[n].axvline(0,ls='--',lw=0.4)
    ax_psf[n].scatter(x_data,y_data,s=1)
    ax_psf[n].scatter(x_data,psf_data,s=1)
    [[ax_psf[n].axvline(a,ls=':',lw=0.3,c='C0'),
      ax_psf[n].axvline(b,ls=':',lw=0.3,c='C0')] for a,b in testbins]
    ax_psf[n].scatter(epsf_coords,epsf,marker='X',color='C1',s=10)
    
