#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
import harps.containers as container
import harps.fit as hfit
import harps.version as hv

import gc
import numpy as np
import harps.lsf.inout as io
import hashlib
import sys
import jax
import jax.numpy as jnp

from scipy import interpolate
from scipy.optimize import brentq
import scipy.stats as stats


def prepare_array(array):
    if array is not None:
        dim = len(np.shape(array))
        if dim == 2:
            output = np.moveaxis(np.atleast_3d(array),-1,0)
        if dim == 3:
            output = np.atleast_3d(array)
    else:
        output = None
    return output

def stack_1d(fittype,linelist,flx1d_in,x1d_in,err1d_in=None,
          bkg1d_in=None):
    # numex = np.shape(linelists)[0]
    ftpix = '{}_pix'.format(fittype)
    ftwav = '{}_wav'.format(fittype)
    
    x1d_in   = np.atleast_1d(x1d_in)
    flx1d_in = np.atleast_1d(flx1d_in)
    bkg1d_in = np.atleast_1d(bkg1d_in)
    err1d_in = np.atleast_1d(err1d_in)
    
    numpix = np.shape(flx1d_in)
    pix1d = np.zeros(numpix)
    flx1d = np.zeros(numpix)
    err1d = np.zeros(numpix)   
    vel1d = np.zeros(numpix) 
    
    for j,line in enumerate(linelist):
        pixl     = line['pixl']
        pixr     = line['pixr']
        pix1l    = np.arange(pixl,pixr) - line[ftpix][1]
        lineflux = flx1d_in[pixl:pixr]
        wav1l    = x1d_in[pixl:pixr]
        vel1l    = (wav1l - line[ftwav][1])/line[ftwav][1]*299792.458 #km/s
        if bkg1d_in is not None:
            linebkg  = bkg1d_in[pixl:pixr]
            lineflux = lineflux - linebkg
        if err1d_in is not None:
            lineerr = err1d_in[pixl:pixr]
            if bkg1d_in is not None:
                lineerr = np.sqrt(lineerr**2 + \
                                 bkg1d_in[pixl:pixr])
        # flux is Poissonian distributed, P(nu),  mean = variance = nu
        # Sum of fluxes is also Poissonian, P(sum(nu))
        #           mean     = sum(nu)
        #           variance = sum(nu)
        C_flux = np.sum(lineflux)
        C_flux_err = np.sqrt(C_flux)
        pix1d[pixl:pixr] = pix1l
        vel1d[pixl:pixr] = vel1l
        flx1d[pixl:pixr] = lineflux/C_flux
        err1d[pixl:pixr] = 1./C_flux*np.sqrt(lineerr**2 + \
                                        (lineflux*C_flux_err/C_flux)**2)
            
    pix1d = jnp.array(pix1d)
    vel1d = jnp.array(vel1d)
    flx1d = jnp.array(flx1d*100)
    err1d = jnp.array(err1d*100)
            
    return pix1d,vel1d,flx1d,err1d

def stack(fittype,linelists,flx3d_in,x3d_in,err3d_in=None,
          bkg3d_in=None,orders=None):
    # numex = np.shape(linelists)[0]
    ftpix = '{}_pix'.format(fittype)
    ftwav = '{}_wav'.format(fittype)
    
    x3d_in   = prepare_array(x3d_in)
    flx3d_in = prepare_array(flx3d_in)
    bkg3d_in = prepare_array(bkg3d_in)
    err3d_in = prepare_array(err3d_in)
    
    numex, numord, numpix = np.shape(flx3d_in)
    pix3d = np.zeros((numord,numpix,numex))
    flx3d = np.zeros((numord,numpix,numex))
    err3d = np.zeros((numord,numpix,numex))   
    vel3d = np.zeros((numord,numpix,numex)) 
    
    linelists = np.atleast_2d(linelists)
    for exp,linelist in enumerate(linelists):
        hf.update_progress((exp+1)/len(linelists),"Stack")
        if orders is not None:
            orders = np.atleast_1d(orders)
        else:
            orders = np.unique(linelist['order'])
        # print(orders)
        for j,line in enumerate(linelist):
            od       = line['order']
            if od not in orders:
                continue
            pixl     = line['pixl']
            pixr     = line['pixr']
            # print(pixl,pixr)
            f_star = line[ftpix][0]
            x_star = line[ftpix][1]
            pix1l = np.arange(pixl,pixr) - x_star
            
            pixpos = np.arange(pixl,pixr,1)
            
            lineflux = flx3d_in[exp,od,pixl:pixr]
            wav1l = x3d_in[exp,od,pixl:pixr]
            # print(exp,od,pixl,pixr,lineflux)
            vel1l = (wav1l - line[ftwav][1])/line[ftwav][1]*299792.458 #km/s
            if bkg3d_in is not None:
                linebkg  = bkg3d_in[exp,od,pixl:pixr]
                lineflux = lineflux - linebkg
                # lineerr  = np.sqrt(lineflux + linebkg)
            if err3d_in is not None:
                lineerr = err3d_in[exp,od,pixl:pixr]
                if bkg3d_in is not None:
                    lineerr = np.sqrt(lineerr**2 + \
                                     bkg3d_in[exp,od,pixl:pixr])
            # flux is Poissonian distributed, P(nu),  mean = variance = nu
            # Sum of fluxes is also Poissonian, P(sum(nu))
            #           mean     = sum(nu)
            #           variance = sum(nu)
            # C_flux = np.sum(lineflux)
            # C_flux_err = np.sqrt(C_flux)
            C_flux = f_star
            C_flux_err = 0.
            pix3d[od,pixl:pixr,exp] = pix1l
            vel3d[od,pixl:pixr,exp] = vel1l
            flx3d[od,pixl:pixr,exp] = lineflux/C_flux
            err3d[od,pixl:pixr,exp] = 1./C_flux*np.sqrt(lineerr**2 + \
                                            (lineflux*C_flux_err/C_flux)**2)
            
    pix3d = jnp.array(pix3d)
    vel3d = jnp.array(vel3d)
    flx3d = jnp.array(flx3d*100)
    err3d = jnp.array(err3d*100)
            
    return pix3d,vel3d,flx3d,err3d,orders

def stack_spectrum(spec,version,fittype=None):
    wav2d = spec.wavereference
    flx2d = spec.data
    bkg2d = spec.background
    err2d = spec.error
    
    item,fittype  = get_linelist_item_fittype(version)
    llist = spec[item]
    # orders = spec.prepare_orders(order)
    
    
    return stack(fittype,llist,flx2d,wav2d,err2d,bkg2d) 

def _prepare_lsf1s(n_data,n_sct,pars):
    lsf1s = get_empty_lsf(1,n_data,n_sct,pars)#[0]
    return lsf1s

def _calculate_shift(y,x):
    return -hf.derivative_zero(y,x,-1,1)

# @jax.jit
# def loss_(theta,X,Y,Y_err):
#     gp = build_gp(theta,X,Y_err)
#     return -gp.log_probability(Y)




    
def bin_means(x,y,xbins,minpts=10,value='mean',kind='spline',y_err=None,
              remove_outliers=False,return_population_variance=False,
              return_population_kurtosis=False,
              return_variance_variance=False):
    
    if return_variance_variance:
        calc_pop_var = True
        calc_pop_kurt = True
        calc_var_var = True
    if return_population_kurtosis:
        calc_pop_var = True
        calc_pop_kurt = True
    if return_population_variance:
        calc_pop_var = True
    def interpolate_bins(means,missing_xbins,kind):
        
        x = xbins[idx]
        y = means[idx]
        if kind == 'spline':
            splr  = interpolate.splrep(x,y)
            model = interpolate.splev(missing_xbins,splr)
        else:
            model = np.interp(missing_xbins,x,y)
        return model
   # which pixels have at least minpts points in them?
    hist, edges = np.histogram(x,xbins)
    bins  = np.where(hist>=minpts)[0]+1
    # sort the points into bins and use only the ones with at least minpts
    inds  = np.digitize(x,xbins,right=False)
    means = np.zeros(len(xbins))
    stds  = np.zeros(len(xbins))
    var_pop = np.zeros(len(xbins))
    kurt  = np.zeros(len(xbins))
    var_var = np.zeros(len(xbins))
    idx   = bins
    # first calculate means for bins in which data exists
    for i in idx:
        # skip if more right than the rightmost bin
        if i>=len(xbins):
            continue
        # select the points in the bin
        cut = np.where(inds==i)[0]
        if len(cut)<1:
            print("Deleting bin ",i)
            continue
        y1  = y[cut]
        
        if remove_outliers == True:
            outliers = hf.is_outlier(y1)
            y1=y1[~outliers]
        
        if value=='mean':
            means[i] = np.nanmean(y1)
            stds[i]  = np.nanstd(y1)
        elif value=='median':
            means[i] = np.nanmedian(y1)
            stds[i]  = np.nanstd(y1)
        elif value=='weighted_mean':
            assert y_err is not None
            means[i],stds[i] = hf.wmean(y1,y_err[cut])
        if calc_pop_var:
            var_pop[i] = np.sum((y1-np.mean(y1))**2)/(len(y1)-1)
        if calc_pop_kurt:
            # first calculate the biased sample 4th moment and correct for bias
            n        = len(y1)
            mom4_sam = stats.moment(y1,moment=4,nan_policy='omit') 
            mom4_pop = n/(n-1)*mom4_sam
            # kurtosis is the 4th population moment / standard deviation**4
            kurt[i]  = mom4_pop / np.power(var_pop[i],2.)
        if calc_var_var:
            n = len(y1)
            var_var[i] = (kurt[i] - (n-3)/(n-1))*np.power(var_pop[i],2.)/n
    # go back and interpolate means for empty bins
    idy   = hf.find_missing(idx)
    # interpolate if no points in the bin, but only pixels -5 to 5
    if len(idy)>0:
        idy = np.atleast_1d(idy)
        means[idy] = interpolate_bins(means,xbins[idy],kind)
        
    return_tuple = (means,stds,hist)
    if return_population_variance: # this is really variance, not standard dev
        return_tuple = return_tuple + (var_pop,)
    if return_population_kurtosis:   
        return_tuple = return_tuple + (kurt,)
    if return_variance_variance:   
        return_tuple = return_tuple + (var_var,)
    return return_tuple

def bin_statistics(x,y,minpts=10,
                   calculate = ['mean','std','pop_variance','pop_kurtosis',
                                'pop_variance_variance'],
                   remove_outliers=False):
    
    counts, bin_edges = bin_optimally(x,minpts)
    # means, stds = get_bin_mean_std(x, y, bin_edges)
    
    arrays = get_bin_stat(x, y, bin_edges,calculate=calculate, 
                          remove_outliers=remove_outliers)
    
    
    return arrays
    

def bin_optimally(x,minpts=10):
    
    # determine histogram limits, symmetric around zero
    # xmin = np.abs(np.min(x))
    # xmax = np.abs(np.max(x))
    xlim = np.max(np.abs(x))
    # choose a starting value for nbins and count the number of points in bins
    nbins = 50
    
    # stopping condition is defined as False for first iteration
    condition = False
    while condition == False:
        counts ,bin_edges = np.histogram(x,nbins,range=(-xlim,xlim))
        # remove bins with no points from consideration (there may be gaps in data)
        cut = np.where(counts!=0)[0]
        counts_ = counts[cut]
        condition = np.all(counts_>minpts)
        if not condition: nbins = nbins - 1
    
    return counts, bin_edges


        
def get_bin_stat(x,y,bin_edges,calculate=['mean','std'],remove_outliers=True):
    allowed = ['mean','std','sam_variance','sav_variance_variance',
               'pop_variance','pop_kurtosis','pop_variance_variance']
    calculate = np.atleast_1d(calculate)
    # assert calculate in allowed, 'input not recognised'
    indices = np.digitize(x,bin_edges)
    nbins  = len(bin_edges)-1
    arrays = {name:np.zeros(nbins) for name in calculate}
    functions = {'mean':np.nanmean,
                 'std':np.nanstd,
                 'sam_variance':np.nanvar,
                 'pop_variance':np.nanvar,
                 'sam_variance_variance':get_samvar_variance,
                 'pop_kurtosis':get_kurtosis,
                 'pop_variance_variance':get_popvar_variance}
    arguments = {'mean':{},
                 'std':{},
                 'sam_variance':{},
                 'sam_variance_variance':{},
                 'pop_variance':{'ddof':1},
                 'pop_kurtosis':{},
                 'pop_variance_variance':{}}
    # means = np.zeros(nbins)
    # stds  = np.zeros(nbins)
    # plt.scatter(x,y)
    # [plt.axvline(pos,ls=':',c='k') for pos in bin_edges]
    for i in range(nbins):
        cut = np.where(indices==i+1)[0]
        if len(cut)>0: 
            pass
        else:
            continue
        y_i = y[cut] 
        if remove_outliers == True:
            outliers = hf.is_outlier(y_i)
            y_i=y_i[~outliers]
        for name in calculate:
            arrays[name][i] = functions[name](y_i,**arguments[name])
    # plt.errorbar((bin_edges[1:]+bin_edges[:-1])/2,means,stds,ls='',marker='s',c='r')
    return arrays

def get_kurtosis(x,*args,**kwargs):
    n        = len(x)
    mom4_sam = stats.moment(x,moment=4,nan_policy='omit') 
    mom4_pop = n/(n-1)*mom4_sam
    var_pop  = np.nanvar(x,ddof=1)
    # kurtosis is the 4th population moment / standard deviation**4
    return mom4_pop / np.power(var_pop,2.)

def get_samvar_variance(x,*args,**kwargs):
    n        = len(x)
    mom4_sam = stats.moment(x,moment=4,nan_policy='omit') 
    var      = np.nanvar(x)
    
    return mom4_sam/n - var**2 * (n-3)/(n*(n-1))

def get_popvar_variance(x,*args,**kwargs):
    n = len(x)
    var_pop = np.nanvar(x,ddof=1)
    kurt    = get_kurtosis(x,*args,**kwargs)
    return (kurt - (n-3)/(n-1))*np.power(var_pop,2.)/n

def solve_1d(lsf2d,linelist1d,x1d,flx1d,bkg1d,err1d,fittype,scale='pix',
             interpolate=False):
    
    tot = len(linelist1d)
    scl = f"{scale[:3]}"
    for i, line in enumerate(linelist1d):
        od   = line['order']
        
        lpix = line['pixl']
        rpix = line['pixr']
        flx  = flx1d[lpix:rpix]
        x    = x1d[lpix:rpix]
        bkg  = bkg1d[lpix:rpix]
        err  = err1d[lpix:rpix]
        try:
            lsf1d  = lsf2d[od].values
        except:
            continue
        if len(lsf1d)>len(np.unique(lsf1d['segm'])):
            realnseg = len(lsf1d)
            expnseg  = len(np.unique(lsf1d['segm']))
            raise ValueError(f"Expected {expnseg} segments, got {realnseg}")
        success = False
        # print(x,flx,bkg,err)
        try:
            
            output = hfit.lsf(x,flx,bkg,err,lsf1d,interpolate=interpolate,
                              output_model=False)
            success, pars, errs, chisq, chisqnu = output
        except:
            pass
        # sys.exit()
        print('\nline',i,success,pars,chisq, chisqnu)
        # sys.exit()
        if not success:
            print('fail')
            pars = np.full(3,np.nan)
            errs = np.full(3,np.nan)
            chisq = np.nan
            continue
        else:
            pars[1] = pars[1] 
            line[f'lsf_{scl}']     = pars
            line[f'lsf_{scl}_err'] = errs
            line[f'lsf_{scl}_chisq']  = chisq
            line[f'lsf_{scl}_chisqnu']  = chisqnu
        
        hf.update_progress((i+1)/tot,"Solve")
    return linelist1d

def get_linelist_item_fittype(version,fittype=None):
    if version<=200:
        item = 'linelist'
        default_fittype = 'gauss'
    else:
        item = ('linelist',version-100)
        default_fittype = 'lsf'
        
    fittype = fittype if fittype is not None else default_fittype
    return item,fittype


def solve(out_filepath,lsf_filepath,iteration,order,scale='pixel',
          model_scatter=False,interpolate=False):
    from fitsio import FITS
    from harps.lsf.container import LSF
    # abbreviations
    scl = f'{scale[:3]}'
    version = hv.item_to_version(dict(iteration=iteration,
                                        model_scatter=model_scatter,
                                        interpolate=interpolate
                                        ),
                                   ftype='lsf'
                                   )
    print(f'version={version}')
    # READ LSF
    with FITS(lsf_filepath,'rw',clobber=False) as hdu:
        lsf2d = hdu[scale,version].read()
    LSF2d = LSF(lsf2d)
    
    # COPY LINELIST 
    io.copy_linelist_inplace(out_filepath, version)
    
    # READ OLD LINELIST AND DATA
    with FITS(out_filepath,'rw',clobber=False) as hdu:
        item,fittype = get_linelist_item_fittype(version)
        print(item,fittype)
        centres = hdu[item].read(columns=f'{fittype}_{scl}')[:,1]
            # linelist_im1 = hdu['linelist',iteration-1].read()
        linelist = hdu[item].read()
        flx2d = hdu['flux'].read()
        bkg2d = hdu['background'].read()
        err2d = hdu['error'].read()
        nbo,npix = np.shape(flx2d)
        x2d   = np.vstack([np.arange(npix) for od in range(nbo)])
    # sys.exit()
    # MAKE MODEL EXTENSION
    io.make_extension(out_filepath, 'model_lsf', version, flx2d.shape)
        
    orders = hf.prepare_orders(order, nbo, sOrder=39, eOrder=None)
    
    firstrow = int(1e6)
    tot = len(linelist)
    # new_linelist = []
    # model2d = np.zeros_like(flx2d)
    for i, line in enumerate(linelist):
        od   = line['order']
        if od not in orders:
            continue
        else:
            if i<firstrow: firstrow=i
            pass
        
        try:
            lsf1d  = LSF2d[od].values
        except:
            continue
        
        lpix   = line['pixl']
        rpix   = line['pixr']
        cent   = centres[i]
        flx1l  = flx2d[od,lpix:rpix]
        x1l    = x2d[od,lpix:rpix]
        bkg1l  = bkg2d[od,lpix:rpix]
        err1l  = err2d[od,lpix:rpix]
        success = False
        print(f"\ni={i:>05d} od={od:>2d} cent={cent:>4.2f} "+\
              f"pixrange={lpix:04d}-{rpix:04d} firstrow={firstrow:>04d}")
        try:
            
            output = hfit.lsf(x1l,flx1l,bkg1l,err1l,lsf1d,
                              interpolate=interpolate,
                              output_model=True)
            success, pars, errs, chisq, chisqnu, model1l = output
        except:
            pass
        # print('line',i,success,pars,chisq)
        if not success:
            print('fail')
            return x1l,flx1l,bkg1l,err1l,lsf1d,interpolate
            # sys.exit()
            pars = np.full(3,np.nan)
            errs = np.full(3,np.nan)
            chisq = np.nan
            chisqnu = np.nan
            continue
        else:
            # pars[1] = pars[1] 
            line[f'lsf_{scl}']     = pars
            line[f'lsf_{scl}_err'] = errs
            line[f'lsf_{scl}_chisq']  = chisq
            line[f'lsf_{scl}_chisqnu']  = chisqnu
        #print(line['lsf'])
        # new_linelist.append(line)
        with FITS(out_filepath,'rw',clobber=False) as hdu:
            hdu['model_lsf',version].write(np.array(model1l),start=[od,lpix])
            hdu['linelist',version].write(np.atleast_1d(line),firstrow=i)
        gc.collect()   
        hf.update_progress((i+1)/tot,"Solve")
    # return new_linelist
    # new_linelist = np.hstack(new_linelist)     
    with FITS(out_filepath,'rw',clobber=False) as hdu:
        hdu['linelist',version].write_key('ITERATION', iteration)
        hdu['linelist',version].write_key('SCATTER', model_scatter)
        hdu['linelist',version].write_key('INTERPOLATE', interpolate)
        # hdu['linelist',version].write_comment(f"Iteration {iteration}")
        # hdu.write(new_linelist,firstrow=firstrow)
    return linelist


def solve2(lsf2d,linelist,x2d,flx2d,bkg2d,err2d,order,fittype,scale='pix',
          interpolate=False):
    linelist = np.atleast_1d(linelist)
    x2d   = np.atleast_2d(x2d)
    flx2d = np.atleast_2d(flx2d)
    bkg2d = np.atleast_2d(bkg2d)
    err2d = np.atleast_2d(err2d)
    orders = np.atleast_1d(order)
    
    tot = len(linelist)
    for i, line in enumerate(linelist):
            od   = line['order']
            if od not in orders:
                continue
            else:
                pass
            try:
                lsf1d  = lsf2d[od].values
            except:
                continue
            segm = line['segm']
            # mode edges
            lpix = line['pixl']
            rpix = line['pixr']
            bary = line['bary']
            cent = line['{}_pix'.format(fittype)][1]
            flx1l  = flx2d[od,lpix:rpix]
            x1l    = x2d[od,lpix:rpix]
            bkg1l  = bkg2d[od,lpix:rpix]
            err1l  = err2d[od,lpix:rpix]
            success = False
            try:
                
                output = hfit.lsf(x1l,flx1l,bkg1l,err1l,lsf1d,
                                  interpolate=interpolate,
                                  output_model=False)
                success, pars, errs, chisq, chisqnu = output
            except:
                pass
            # print('line',i,success,pars,chisq)
            if not success:
                print('fail')
                
                pars = np.full(3,np.nan)
                errs = np.full(3,np.nan)
                chisq = np.nan
                chisqnu = np.nan
                continue
            else:
                # pars[1] = pars[1] 
                line[f'lsf_{scale}']     = pars
                line[f'lsf_{scale}_err'] = errs
                line[f'lsf_{scale}_chisq']  = chisq
                line[f'lsf_{scale}_chisqnu']  = chisqnu
            #print(line['lsf'])
            
            hf.update_progress((i+1)/tot,"Solve")
            
    return linelist



def solve_bk(lsf,linelists,x3d,flx3d,bkg3d,err3d,fittype,scale='pix',
          interpolate=False):
    
    x3d   = prepare_array(x3d)
    flx3d = prepare_array(flx3d)
    bkg3d = prepare_array(bkg3d)
    err3d = prepare_array(err3d)
    
    linelists = np.atleast_2d(linelists)
    tot = len(np.hstack(linelists))
    for exp,linelist in enumerate(linelists):
        print(exp)
        for i, line in enumerate(linelist):
            od   = line['order']
            segm = line['segm']
            # mode edges
            lpix = line['pixl']
            rpix = line['pixr']
            bary = line['bary']
            cent = line['{}_pix'.format(fittype)][1]
            flx  = flx3d[exp,od,lpix:rpix]
            x    = x3d[exp,od,lpix:rpix]
            # pix  = np.arange(lpix,rpix,1.) 
            bkg  = bkg3d[exp,od,lpix:rpix]
            err  = err3d[exp,od,lpix:rpix]
            # wgt  = np.ones_like(pix)
            # initial guess
            p0 = (np.max(flx),cent,1)
            try:
                lsf1d  = lsf[od].values
            except:
                continue
            # print('line=',i)
            # print(*[np.shape(_) for _ in [x,flx,bkg,err]])
            # success, pars, errors, chisq, chisqnu, model = hfit.lsf(x,flx,bkg,err,
            #                                   lsf1s,
            #                                   output_model=True,
            #                                   plot=False)
            # output=hfit.lsf(x,flx,bkg,err,lsf1d,output_model=False)
            # lsfcen = output[1][1]
            # print(f"{i:>4d}, gaussian={cent:4.2f}, lsf={lsfcen:4.2f}, diff={(lsfcen-cent)*829:4.2f}")
            # sys.exit()
            success = False
            try:
                
                output = hfit.lsf(x,flx,bkg,err,lsf1d,interpolate=interpolate,
                                  output_model=False)
                success, pars, errs, chisq, chisqnu = output
            except:
                pass
            # print('line',i,success,pars,chisq)
            if not success:
                print('fail')
                pars = np.full_like(p0,np.nan)
                errs = np.full_like(p0,np.nan)
                chisq = np.nan
                continue
            else:
                pars[1] = pars[1] 
                line[f'lsf_{scale}']     = pars
                line[f'lsf_{scale}_err'] = errs
                line[f'lsf_{scale}_chisq']  = chisq
                line[f'lsf_{scale}_chisqnu']  = chisqnu
            #print(line['lsf'])
            
            hf.update_progress((i+1)/tot,"Solve")
    return linelists
def shift_anderson(lsfx,lsfy):
    deriv = hf.derivative1d(lsfy,lsfx)
    
    left  = np.where(lsfx==-0.5)[0]
    right = np.where(lsfx==0.5)[0]
    elsf_neg     = lsfy[left]
    elsf_pos     = lsfy[right]
    elsf_der_neg = deriv[left]
    elsf_der_pos = deriv[right]
    shift        = float((elsf_pos-elsf_neg)/(elsf_der_pos-elsf_der_neg))
    return shift
def shift_zeroder(lsfx,lsfy):
    shift = -brentq(hf.derivative_eval,-1,1,args=(lsfy,lsfx))
    return shift    

    
def get_empty_lsf(numsegs,n_data,n_sct,pars=None):
    '''
    Returns an empty array for LSF model.
    
    Args:
    ----
        method:    string ('analytic','spline','gp')
        numsegs:   int, number of segments per range modelled
        n:         int, number of parameters (20 for analytic, 160 for spline, 2 for gp)
        pixcens:   array of pixel centers to save to field 'x'
    '''
    lsf_cont = container.lsf(numsegs,n_data,n_sct,pars)
        
    return lsf_cont


def clean_input(x1s,flx1s,err1s=None,filter=None,xrange=None,binsize=None,
                sort=True,verbose=False,plot=False,rng_key=None):
    '''
    Removes infinities, NaN and zeros from the arrays. If sort=True, sorts the
    data by pixel number. If filter is given, removes every nth element from
    the array, where n=filter value.
    

    Parameters
    ----------
    x1s : array-like
        X-axis array, either pixels or velocities.
    flx1s : array-like
        Flux array.
    err1s : array-like, optional
        Error array. The default is None.
    verbose : boolean, optional
        Prints messages. The default is False.
    sort : boolean, optional
        Sorts the array. The default is True.
    filter_every : int, optional
        Filters every int element from the arrays. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    x1s    = np.ravel(x1s)
    sorter = np.argsort(x1s)
    x1s    = x1s[sorter]
    flx1s  = np.ravel(flx1s)[sorter]
    if err1s is not None:
        err1s = np.ravel(err1s)[sorter]
    
    # notoutlier = ~hf.is_outlier_bins(flx1s,idx)
    # remove infinites, nans, zeros and outliers
    arr = np.array([np.isfinite(x1s),
                    np.isfinite(flx1s),
                    np.isfinite(err1s),
                    flx1s!=0
                    ])
    finite_ = np.logical_and.reduce(arr)
    cut     = np.where(finite_)[0]
    # optimal binning and outlier detection    
    # counts, bin_edges = bin_optimally(x1s[finite_],minpts=3)
    bin_edges = np.arange(-8,8+0.25,0.25)
    # counts, edges = np.histogram(x1s[finite_],bins=bin_edges)
    # print(counts,bin_edges)
    idx     = np.digitize(x1s[finite_],bin_edges)
    notout  = ~hf.is_outlier_bins(flx1s[finite_],idx,thresh=3.5)
    finite  = cut[notout]
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x1s,flx1s,marker='o')
        plt.scatter(x1s[~finite_],flx1s[~finite_],marker='x',c='g')
        plt.scatter(x1s[cut[~notout]],flx1s[cut[~notout]],marker='x',c='r')
        [plt.axvline(edge,ls=':') for edge in bin_edges]
    numpts  = np.size(flx1s)
    
     
    x      = x1s[finite]
    flx    = flx1s[finite]
    res    = (x,flx)
    if err1s is not None:
        err = np.ravel(err1s)[finite]
        res = res + (err,)
    
    if filter:
        rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(55873)
        shape   = (len(x)//filter,)
        choice  = jax.random.choice(rng_key,np.arange(len(x)),shape,False)
        res = tuple(array[choice] for array in res)
        # res = (array[::filter] for array in res)
    if sort:
        sorter = np.argsort(res[0])
        res = (array[sorter] for array in res)
    res = tuple(res)    
    if verbose:
        diff  = numpts-len(res[0])
        print("{0:5d}/{1:5d} ({2:5.2%}) kept ; ".format(len(res[0]),numpts,
                                                      len(res[0])/numpts) +\
              "{0:5d}/{1:5d} ({2:5.2%}) discarded".format(diff,numpts,
                                                        diff/numpts))
    return tuple(res)   

def lin2log(values,errors):
    '''
    Transforms the values and the errors from linear into log space. 

    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    errors : TYPE
        DESCRIPTION.

    Returns
    -------
    log_values : TYPE
        DESCRIPTION.
    err_log_values : TYPE
        DESCRIPTION.

    '''
    log_values = jnp.log(values)
    err_log_values = jnp.abs(1./values * errors)
    return log_values, err_log_values

def log2lin(values,errors):
    '''
    Transforms the values and the errors from log into linear space. 

    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    errors : TYPE
        DESCRIPTION.

    Returns
    -------
    lin_values : TYPE
        DESCRIPTION.
    err_lin_values : TYPE
        DESCRIPTION.

    '''
    lin_values = jnp.exp(values)
    err_lin_values = jnp.abs(values) * errors
    return lin_values, err_lin_values
    
def get_checksum(X,Y,Y_err,uniqueid=None):
    if uniqueid is not None:
        uniqueid = uniqueid 
    else:
        import random
        import time
        seed= random.seed(time.time())
        uniqueid = random.random()
    _ = np.sum([X,Y,Y_err]) + np.sum(np.atleast_1d(uniqueid))
    return hashlib.md5(_).hexdigest()
    