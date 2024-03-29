#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
import harps.functions.math as mathfunc
import harps.functions.outliers as outlier
import harps.functions.aux as auxfunc
import harps.functions.spectral as specfunc


import harps.containers as container
import harps.lsf.fit as hlsfit
import harps.version as hv
import harps.progress_bar as progress_bar
import harps.lines_aux as laux
import harps.settings as hs
import harps.wavesol as ws

import gc
import numpy as np
import harps.lsf.inout as io
import hashlib
import sys
import logging
import jax
import jax.numpy as jnp
import time
import multiprocessing
import copy
from   functools import partial

from scipy import interpolate
from scipy.optimize import brentq
import scipy.stats as stats

from fitsio import FITS

import matplotlib.pyplot as plt


def prepare_array(array):
    if array is not None:
        dim = len(np.shape(array))
        if dim == 3:
            output = np.atleast_3d(array)
        elif dim<3:
            output = np.moveaxis(np.atleast_3d(array),-1,0)
        else:
            raise Exception("Array has more than 3 dimensions.")
    else:
        output = None
    return output





def stack(*args,**kwargs):
    return stack_subbkg_divenv(*args,**kwargs)



def stack_subbkg_divenv(fittype,linelists,flx3d_in,x3d_in,err3d_in,
          env3d_in,bkg3d_in,orders=None,subbkg=hs.subbkg,divenv=hs.divenv):
    # numex = np.shape(linelists)[0]
    
    logging.info(f'subbkg, divenv = {subbkg}, {divenv}')
    
    ftpix = '{}_pix'.format(fittype)
    ftwav = '{}_wav'.format(fittype)
    
    x3d_in   = prepare_array(x3d_in)
    flx3d_in = prepare_array(flx3d_in)
    bkg3d_in = prepare_array(bkg3d_in)
    env3d_in = prepare_array(env3d_in)
    err3d_in = prepare_array(err3d_in)
    
    numex, numord, numpix = np.shape(flx3d_in)
    pix3d = np.zeros((numord,numpix,numex))
    flx3d = np.zeros((numord,numpix,numex))
    err3d = np.zeros((numord,numpix,numex))   
    vel3d = np.zeros((numord,numpix,numex)) 
    
    
    
    
    data, data_error, bkg_norm = laux.prepare_data(flx3d_in,env3d_in,bkg3d_in, 
                                         subbkg=subbkg, divenv=divenv)
    
    
    linelists = np.atleast_2d(linelists)
    for exp,linelist in enumerate(linelists):
        progress_bar.update((exp+1)/len(linelists),"Stack")
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
            f_star = line[f'{ftpix}_integral']
            x_star = line[ftpix][1]
            wav1l = x3d_in[exp,od,pixl:pixr]
            vel1l = (wav1l - line[ftwav][1])/line[ftwav][1]*299792.458 #km/s
            # if any of vel1l are nan or have velocities larger than 8 km/s
            # then set the entire velocity array to nan, it will be removed
            # later by ``clean_input''
            cond = np.any(np.isnan(vel1l)) or np.any(np.abs(vel1l)>8.0)
            if cond:
                vel1l = np.full_like(wav1l, np.nan)
            # flux is Poissonian distributed, P(nu),  mean = variance = nu
            # Sum of fluxes is also Poissonian, P(sum(nu))
            #           mean     = sum(nu)
            #           variance = sum(nu)
            # C_flux = np.sum(flx1l)
            C_flux = f_star
            C_flux_err = np.sqrt(C_flux)
            # C_flux_err = 0.
            
            data1l = data[exp,od,pixl:pixr]
            data1l_var_tmp = (data_error[exp,od,pixl:pixr])**2
            # data1l_var = laux.quotient_variance(data1l, data1l_var_tmp, 
            #                                     f_star, np.sqrt(f_star))
            flx1l = data1l/f_star
            p = flx1l
            data1l_var = p*(1-p)/f_star 
            data1l_err = np.sqrt(data1l_var)
            
            pix3d[od,pixl:pixr,exp] = np.arange(pixl,pixr) - x_star
            vel3d[od,pixl:pixr,exp] = vel1l
            flx3d[od,pixl:pixr,exp] = flx1l
            # error propagation from normalisation
            # N = F/C_flux
            # sigma_N = 1/C_flux * np.sqrt(sigma_F**2 + (N * C_flux_err)**2)
            err3d[od,pixl:pixr,exp] = data_error[exp,od,pixl:pixr]/f_star
            # err3d[od,pixl:pixr,exp] = data1l_err
            # err3d[od,pixl:pixr,exp] = 1./f_star*np.sqrt(data_error[exp,od,pixl:pixr]**2 + \
            #                                 data[exp,od,pixl:pixr]*f_star)
            
    pix3d = jnp.array(pix3d)
    vel3d = jnp.array(vel3d)
    flx3d = jnp.array(flx3d*100)
    err3d = jnp.array(err3d*100)
    
    
    return pix3d,vel3d,flx3d,err3d,orders


def stack_outpath(outpath,version,orders=None,subbkg=hs.subbkg,divenv=hs.subbkg,
                   **kwargs):
    # wav2d = spec.wavereference
    # wav2d = spec['wavesol_gauss',701] # this should be changed to a new wsol every iteration
    item,fittype  = get_linelist_item_fittype(version)
    logging.info(f"Stacking {item}, {fittype}")
    with FITS(outpath) as hdul:
        flx2d = hdul['flux'].read()
        bkg2d = hdul['background'].read()
        env2d = hdul['envelope'].read()
        err2d = np.sqrt(np.abs(flx2d+bkg2d))
        llist = hdul[item].read()
        wref  = hdul['wavereference'].read()
    nord, npix = np.shape(flx2d)
    if version//100==1:
        wav2d = wref
    else:
        wav2d = ws.comb_dispersion(linelist=llist, 
                                   version=701, 
                                   fittype=fittype,
                                   npix=npix, 
                                   nord=nord)
    # orders = spec.prepare_orders(order)
    return stack_subbkg_divenv(fittype,llist,flx2d,wav2d,err2d,env2d,bkg2d,
                               orders=orders,subbkg=subbkg,divenv=divenv,
                               **kwargs) 

def stack_spectrum(spec,version,orders=None,subbkg=hs.subbkg,divenv=hs.subbkg,
                   **kwargs):
    # wav2d = spec.wavereference
    # wav2d = spec['wavesol_gauss',701] # this should be changed to a new wsol every iteration
    flx2d = spec['flux']
    bkg2d = spec['background']
    env2d = spec['envelope']
    err2d = np.sqrt(np.abs(flx2d)+np.abs(bkg2d))
    
    item,fittype  = get_linelist_item_fittype(version)
    logging.info(f"Stacking {item}, {fittype}")
    llist = spec[item]
    nord, npix = np.shape(flx2d)
    if version//100==1:
        wav2d = spec['wavereference']
    else:
        wav2d = ws.comb_dispersion(linelist=llist, 
                                   version=701, 
                                   fittype=fittype,
                                   npix=npix, 
                                   nord=nord)
    # orders = spec.prepare_orders(order)
    return stack_subbkg_divenv(fittype,llist,flx2d,wav2d,err2d,env2d,bkg2d,
                               orders=orders,subbkg=subbkg,divenv=divenv,
                               **kwargs) 

def _prepare_lsf1s(n_data,n_sct,pars):
    lsf1s = get_empty_lsf(1,n_data,n_sct,pars)#[0]
    return lsf1s

def _calculate_shift(y,x):
    return -mathfunc.derivative_zero(y,x,-1,1)

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
            outliers = outlier.is_outlier(y1)
            y1=y1[~outliers]
        
        if value=='mean':
            means[i] = np.nanmean(y1)
            stds[i]  = np.nanstd(y1)
        elif value=='median':
            means[i] = np.nanmedian(y1)
            stds[i]  = np.nanstd(y1)
        elif value=='weighted_mean':
            assert y_err is not None
            means[i],stds[i] = mathfunc.wmean(y1,y_err[cut])
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
    idy   = auxfunc.find_missing(idx)
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


        
def get_bin_stat(x,y,bin_edges,calculate=['mean','std'],remove_outliers=True,
                 usejax=False):
    allowed = ['mean','std','sam_variance','sav_variance_variance',
               'pop_variance','pop_kurtosis','pop_variance_variance']
    if usejax:
        import jax.numpy as np
    else:
        import numpy as np
    if isinstance(calculate,list):
        pass
    else:
        calculate = np.atleast_1d(calculate)
    # assert calculate in allowed, 'input not recognised'
    indices = np.digitize(x,bin_edges)
    nbins  = len(bin_edges)-1
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    arrays = {name:np.full(nbins,np.nan) for name in calculate}
    arrays.update({'bin_centres':bin_centres})
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
    flagged = np.zeros(nbins,dtype=bool)
    for i in range(nbins):
        try:
            cut = np.where(indices==i+1)[0]
        except:
            import jax.numpy as jnp
            cut = jnp.where(indices==i+1,size=len(indices),fill_value=False)[0]
        if len(cut)>0: 
            pass
        else:
            flagged[i]=True
            continue
        y_i = y[cut] 
        if remove_outliers == True:
            outliers = outlier.is_outlier(y_i)
            try:
                y_i=y_i[~outliers]
            except:
                pass
        for name in calculate:
            try:
                val = functions[name](y_i,**arguments[name])
                if np.isfinite(val):
                    arrays[name][i] = val
                else:
                    # arrays[name][i] = np.nan
                    flagged[i] = True 
            except:
                val = functions[name](y_i,**arguments[name])
                arrays[name].at[i].set(val)
    output_dict = dict()
    for key,array in arrays.items():
        output_dict[key] = array[~flagged]
    # plt.errorbar((bin_edges[1:]+bin_edges[:-1])/2,means,stds,ls='',marker='s',c='r')
    return output_dict

def get_kurtosis(x,*args,**kwargs):
    n        = len(x)
    diff     = x-jnp.nanmean(x)
    # mom4_sam = 1./n * jnp.nansum(expt_rec(diff,4))
    mom4_sam = stats.moment(x,moment=4,nan_policy='omit') 
    mom4_pop = n/(n-1)*mom4_sam
    var_pop  = jnp.nanvar(x,ddof=1)
    # kurtosis is the 4th population moment / standard deviation**4
    return mom4_pop / jnp.power(var_pop,2.)

def get_samvar_variance(x,*args,**kwargs):
    '''
    Returns variance on the sample variance.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    n        = len(x)
    mom4_sam = stats.moment(x,moment=4,nan_policy='omit') 
    # diff     = x-jnp.nanmean(x)
    # mom4_sam = 1./n * jnp.nansum(expt_rec(diff,4))
    var      = jnp.nanvar(x)
    
    return mom4_sam/n - var**2 * (n-3)/(n*(n-1))

def expt_rec(a, b):
    if b == 0:
        return 1
    elif b % 2 == 1:
        return a * expt_rec(a, b - 1)
    else:
        p = expt_rec(a, b / 2)
        return p * p
    
def get_popvar_variance(x,*args,**kwargs):
    n = len(x)
    var_pop = jnp.nanvar(x,ddof=1)
    kurt    = get_kurtosis(x,*args,**kwargs)
    return (kurt - (n-3)/(n-1))*jnp.power(var_pop,2.)/n

# def solve_1d(lsf2d,linelist1d,x1d,flx1d,bkg1d,err1d,fittype,scale='pix',
#              interpolate=False):
    
#     tot = len(linelist1d)
#     scl = f"{scale[:3]}"
#     for i, line in enumerate(linelist1d):
#         od   = line['order']
        
#         lpix = line['pixl']
#         rpix = line['pixr']
#         flx  = flx1d[lpix:rpix]
#         x    = x1d[lpix:rpix]
#         bkg  = bkg1d[lpix:rpix]
#         err  = err1d[lpix:rpix]
#         try:
#             lsf1d  = lsf2d[od].values
#         except:
#             continue
#         if len(lsf1d)>len(np.unique(lsf1d['segm'])):
#             realnseg = len(lsf1d)
#             expnseg  = len(np.unique(lsf1d['segm']))
#             raise ValueError(f"Expected {expnseg} segments, got {realnseg}")
#         success = False
#         # print(x,flx,bkg,err)
#         try:
            
#             output = hfit.lsf(x,flx,bkg,err,lsf1d,interpolate=interpolate,
#                               output_model=False)
#             success, pars, errs, chisq, chisqnu = output
#         except:
#             pass
#         # sys.exit()
#         print('\nline',i,success,pars,chisq, chisqnu)
#         # sys.exit()
#         if not success:
#             print('fail')
#             pars = np.full(3,np.nan)
#             errs = np.full(3,np.nan)
#             chisq = np.nan
#             continue
#         else:
#             pars[1] = pars[1] 
#             line[f'lsf_{scl}']     = pars
#             line[f'lsf_{scl}_err'] = errs
#             line[f'lsf_{scl}_chisq']  = chisq
#             line[f'lsf_{scl}_chisqnu']  = chisqnu
        
#         progress_bar.update((i+1)/tot,"Solve")
#     return linelist1d

def get_linelist_item_fittype(version,fittype=None):
    if version==1:
        item = ('linelist',version)
        default_fittype = 'lsf'
    elif version>1 and version<=200:
        item = 'linelist'
        default_fittype = 'gauss'
    else:
        item = ('linelist',version-100)
        default_fittype = 'lsf'
    
        
    fittype = fittype if fittype is not None else default_fittype
    return item,fittype

def read_outfile4solve(out_filepath,version,scale):
    with FITS(out_filepath,'rw',clobber=False) as hdu:
        item,fittype = get_linelist_item_fittype(version)
        # print(item,fittype)
        # centres = hdu[item].read(columns=f'{fittype}_{scale[:3]}')[:,1]
            # linelist_im1 = hdu['linelist',iteration-1].read()
        linelist = hdu[item].read()
        flx2d = hdu['flux'].read()
        err2d = hdu['error'].read()
        env2d = hdu['envelope'].read()
        bkg2d = hdu['background'].read()
        
        nbo,npix = np.shape(flx2d)
        x2d   = np.vstack([np.arange(npix) for od in range(nbo)])
    return x2d,flx2d,err2d,env2d,bkg2d,linelist
        
def solve(out_filepath,lsf_filepath,iteration,order,force_version=None,
          model_scatter=False,interpolate=False,scale=['pixel','velocity'],
          npars = None,
          subbkg=hs.subbkg,divenv=hs.divenv,save2fits=True,logger=None):
    from fitsio import FITS
    from harps.lsf.container import LSF2d
    
    def bulk_fit(function):
        manager = multiprocessing.Manager()
        inq = manager.Queue()
        outq = manager.Queue()
    
        # construct the workers
        nproc = multiprocessing.cpu_count()
        logger.info(f"Using {nproc} workers")
        workers = [LineSolver(str(name+1), function,inq, outq) 
                   for name in range(nproc)]
        for worker in workers:
            worker.start()
    
        # add data to the queue for processing
        work_len = tot
        for item in cut:
            inq.put(item)
    
        while outq.qsize() != work_len:
            # waiting for workers to finish
            done = outq.qsize()
            progress = done/(work_len-1)
            time_elapsed = time.time() - time_start
            progress_bar.update(progress,name='lsf.aux.solve',
                               time=time_elapsed,
                               logger=None)
            
            time.sleep(1)
    
        # clean up
        for worker in workers:
            worker.terminate()
    
        # print the outputs
        results = []
        while not outq.empty():
            results.append(outq.get())
        return results
    
    if logger is not None:
        logger = logger.getChild('solve')
    else:
        logger = logging.getLogger(__name__).getChild('solve')
    # abbreviations
    # scl = f'{scale[:3]}'
    if force_version is not None:
        version = force_version
    else:
        version = hv.item_to_version(dict(iteration=iteration,
                                        model_scatter=model_scatter,
                                        interpolate=interpolate
                                        ),
                                   ftype='lsf'
                                   )
    scale = np.atleast_1d(scale)
    logger.info(f'version : {version}')
    # READ LSF
    with FITS(lsf_filepath,'r',clobber=False) as hdu:
        if 'pixel' in scale:
            lsf2d_pix = hdu['pixel_model',version].read()
            LSF2d_nm_pix = LSF2d(lsf2d_pix)
        if 'velocity' in scale:
            lsf2d_vel = hdu['velocity_model',version].read()
            LSF2d_nm_vel = LSF2d(lsf2d_vel)
    # lsf2d_gp = LSF2d_gp[order].values
    # lsf2d_numerical = hlsfit.numerical_model(lsf2d_gp,xrange=(-8,8),subpix=11)
    # LSF2d_numerical = LSF(lsf2d_numerical)
    
    
    # COPY LINELIST 
    io.copy_linelist_inplace(out_filepath, version)
    
    # READ OLD LINELIST AND DATA
    x2d,flx2d,err2d,env2d,bkg2d,linelist = read_outfile4solve(out_filepath,
                                                        version,
                                                        scale='pixel')
    flx_norm, err_norm, bkg_norm  = laux.prepare_data(flx2d,env2d,bkg2d, 
                                         subbkg=subbkg, divenv=divenv)
    
    
    # MAKE MODEL EXTENSION
    io.make_extension(out_filepath, 'model_lsf', version, flx2d.shape)
    
    nbo,npix = np.shape(flx2d)
    orders = specfunc.prepare_orders(order, nbo, sOrder=39, eOrder=None)
    
    # firstrow = int(1e6)
    cut_ = [np.ravel(np.where(linelist['order']==od)[0]) for od in orders]
    cut = np.hstack(cut_)
    tot = len(cut)
    logger.info(f"Number of lines to fit : {tot}")
    # new_linelist = []
    # model2d = np.zeros_like(flx2d)
    # def get_iterable()
    # lines = (line for line in linelist)
    time_start = time.time()
    
    option = 2
    if 'pixel' in scale:
        partial_function_pix = partial(solve_line,
                                       linelist=linelist,
                                       x2d=x2d,
                                       flx2d=flx_norm,
                                       err2d=err_norm,
                                       LSF2d_nm=LSF2d_nm_pix,
                                       ftype='lsf',
                                       scale='pixel',
                                       interpolate=interpolate,
                                       npars=npars)
        if option==1:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial_function_pix, cut)
        elif option==2:
            results = bulk_fit(partial_function_pix)
        new_llist, models = np.transpose(results)
    
        linelist[cut] = new_llist
    # delete these lines later. These were put in to skip re-doing the entire
    # calculations for pixel when also creating velocity models
    # with FITS(out_filepath,'r') as hdul:
        # linelist = hdul['linelist',version].read()
    # fit for wavelength positions
    if 'velocity' in scale:
        lsf_wavesol = ws.comb_dispersion(linelist, version=701, fittype='lsf', 
                                         npix=npix, 
                                         nord=nbo,
                                         ) 
        
        partial_function_vel= partial(solve_line,
                                       linelist=linelist,
                                       x2d=lsf_wavesol,
                                       flx2d=flx_norm,
                                       err2d=err_norm,
                                       LSF2d_nm=LSF2d_nm_vel,
                                       ftype='lsf',
                                       scale='velocity',
                                       interpolate=interpolate,
                                       npars=npars)
        
        if option==1:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial_function_pix, cut)
        elif option==2:
            results = bulk_fit(partial_function_vel)
        new_llist, models = np.transpose(results)
        linelist[cut] = new_llist
    worktime = (time.time() - time_start)
    h, m, s  = progress_bar.get_time(worktime)
    logger.info(f"Total time elapsed : {h:02d}h {m:02d}m {s:02d}s")
    
    if save2fits:
        for i,(ll,mod) in enumerate(zip(new_llist,models)):
            od   = ll['order']
            pixl = ll['pixl']
            row  = cut[i]
            with FITS(out_filepath,'rw',clobber=False) as hdu:
                hdu['model_lsf',version].write(np.array(mod),start=[od,pixl])
                hdu['linelist',version].write(np.atleast_1d(ll),firstrow=row)
        # if 'velocity' in scale:
        #     for i,(mod) in enumerate(zip(new_llist,models)):
        #         od   = ll['order']
        #         pixl = ll['pixl']
        #         row  = cut[i]
        #         with FITS(out_filepath,'rw',clobber=False) as hdu:
        #             hdu['model_lsf',version].write(np.array(mod),start=[od,pixl])
        #             # hdu['linelist',version].write(np.atleast_1d(ll),firstrow=row)
           
        with FITS(out_filepath,'rw',clobber=False) as hdu:
            hdu['linelist',version].write_key('ITER', iteration)
            hdu['linelist',version].write_key('SCT', model_scatter)
            hdu['linelist',version].write_key('INTP', interpolate)
    return linelist

class LineSolver(multiprocessing.Process):
    """
    Simple worker.
    """

    def __init__(self, name, function, in_queue, out_queue):
        super(LineSolver, self).__init__()
        self.name = name
        self.function = function
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.logger = logging.getLogger("worker_"+name)

    def run(self):
        while True:
            # grab work; do something to it (+1); then put the result on the output queue
            item = self.in_queue.get()
            # print(f'item after queue.get = {item}')
            result = self.function(item,logger=self.logger)
            self.out_queue.put(result)
            
def solve_line(i,linelist,x2d,flx2d,err2d,LSF2d_nm,ftype='gauss',scale='pix',
                interpolate=False,npars=None,logger=None):
    
    logger = logger if logger is not None else logging.getLogger(__name__)
    
    if scale[:3] =='pix':
        scl = 'pix'
    elif scale[:3]=='vel':
        scl = 'wav'
    line   = linelist[i]
    od     = line['order']
    lpix   = line['pixl']
    rpix   = line['pixr']
    bary   = line['bary']
    # cent   = line[f'{ftype}_{scl}'][1]
    flx1l  = flx2d[od,lpix:rpix]
    x1l    = x2d[od,lpix:rpix]
    err1l  = err2d[od,lpix:rpix]
    npars  = npars if npars is not None else hs.npars
    
    
    try: 
        LSF1d  = LSF2d_nm[od]
    except:
        logger.warning("LSF not found")
        return None
    
    success = False
    
    
    try:
        # logger.info(lsf1d.interpolate(bary))
        # output = hfit.lsf(x1l,flx1l,bkg1l,err1l,lsf1d,
        #                   interpolate=interpolate,
        #                   output_model=True)
        output = hlsfit.line(x1l,flx1l,err1l,bary,
                             LSF1d=LSF1d,
                             scale=scale,
                             interpolate=interpolate,
                             weight=True,
                             npars=npars,
                             method='scipy',
                             output_model=True)
        
        success, pars, errs, chisq, chisqnu, integral, model1l = output
    except:
        # logger.critical("failed")
        pass
    # print('line',i,success,pars,chisq)
    if not success:
        logger.critical('FAILED TO FIT LINE')
        logger.warning([i,od,bary,x1l,flx1l,err1l])
        # return x1l,flx1l,err1l,LSF1d,interpolate
        # sys.exit()
        pars = np.full(npars,np.nan)
        errs = np.full(npars,np.nan)
        chisq = np.nan
        chisqnu = np.nan
        integral = np.nan
        model1l = np.zeros_like(flx1l)
    # else:
        # pars[1] = pars[1] 
        # new_line = copy.deepcopy(line)
    line[f'lsf_{scl}'][:npars]     = pars[:npars]
    line[f'lsf_{scl}_err'][:npars] = errs[:npars]
    line[f'lsf_{scl}_chisq']       = chisq
    line[f'lsf_{scl}_chisqnu']     = chisqnu
    line[f'lsf_{scl}_integral']    = integral
    
    return line, model1l
    
def shift_anderson(lsfx,lsfy):
    deriv = mathfunc.derivative1d(lsfy,lsfx)
    
    left  = np.where(lsfx==-0.5)[0]
    right = np.where(lsfx==0.5)[0]
    elsf_neg     = lsfy[left]
    elsf_pos     = lsfy[right]
    elsf_der_neg = deriv[left]
    elsf_der_pos = deriv[right]
    shift        = float((elsf_pos-elsf_neg)/(elsf_der_pos-elsf_der_neg))
    return shift
def shift_zeroder(lsfx,lsfy):
    shift = -brentq(mathfunc.derivative_eval,-1,1,args=(lsfy,lsfx))
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
    # remove infinites, nans, zeros and outliers
    arr = np.array([np.isfinite(x1s),
                    np.abs(x1s<10),
                    np.isfinite(flx1s),
                    np.isfinite(err1s),
                    flx1s!=0,
                    # np.abs(x1s)<8.,
                    ])
    finite_ = np.logical_and.reduce(arr)
    cut     = np.where(finite_)[0]
    # optimal binning and outlier detection    
    # counts, bin_edges = bin_optimally(x1s[finite_],minpts=5)
    bin_edges = np.arange(-8,8+0.5,0.5)
    # counts, edges = np.histogram(x1s[finite_],bins=bin_edges)
    # print(counts,bin_edges)
    # idx     = np.digitize(x1s[finite_],bin_edges)
    # identify outliers and remove them
    # keep   = ~hf.is_outlier_from_linear(x1s[finite_],
    #                                     flx1s[finite_],
    #                                     idx,
    #                                     yerrs=err1s[finite_],
    #                                     thresh=3.5)
    # keep  = ~hf.is_outlier_bins(flx1s[finite_],idx,thresh=3.5)
    # finite  = cut[keep]
    # uncomment next line if no outliers should be removed
    finite  = finite_
    if plot:
        # import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x1s,flx1s,marker='o')
        plt.scatter(x1s[~finite_],flx1s[~finite_],marker='x',c='g')
        # plt.scatter(x1s[cut[~keep]],flx1s[cut[~keep]],marker='x',c='r')
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
    