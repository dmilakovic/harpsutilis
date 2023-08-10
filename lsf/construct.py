#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:39:20 2023

@author: dmilakov
"""
from fitsio import FITS
import numpy as np
import jax.numpy as jnp
import harps.functions as hf
import harps.peakdetect as pkd
import harps.lsf.aux as aux
import harps.lsf.gp_aux as gp_aux
import harps.lsf.plot as lsfplot
import harps.lsf.gp as lsfgp
import harps.lsf.inout as lio
# import harps.lsf.write as write
import harps.lsf.read as read
import harps.fit as hfit
import harps.inout as hio
import harps.version as hv
import harps.progress_bar as progress_bar
import harps.settings as hs
import hashlib
import matplotlib.pyplot as plt
# import scipy.interpolate as interpolate
import gc
import multiprocessing
from functools import partial
import time
import ctypes
import logging



# def models_1d(x2d,flx2d,err2d,numseg=16,numiter=5,minpts=10,model_scatter=False,
#               minpix=None,maxpix=None,filter=None,plot=True,save_plot=False,
#               metadata=None,*args,
#                     **kwargs):
#     '''
    

#     Parameters
#     ----------
#     x2d : 2d array
#         Array containing pixel or velocity (km/s) values.
#     flx2d : 2d array
#         Array containing normalised flux values.
#     err2d : 2d array
#         Array containing errors on flux.
#     method : str
#         Method to use for LSF reconstruction. Options: 'gp','spline','analytic'
#     numseg : int, optional
#         Number of segments along the main dispersion (x-axis) direction. 
#         The default is 16.
#     numpix : int, optional
#         Distance (in pixels or km/s) each side of the line centre to use. 
#         The default is 8 (assumes pixels).
#     subpix : int, optional
#         The number of divisions of each pixel or km/s bin. The default is 4.
#     numiter : int, optional
#         DESCRIPTION. The default is 5.
#     minpix : int, optional
#         DESCRIPTION. The default is 0.
#     minpts : int, optional
#         Only applies when using method='spline'. The minimum number of lines 
#         in each subpixel or velocity bin. The default is 10.
#     filter : int, optional
#         If given, the program will use every N=filter (x,y,e) values. 
#         The default is None, all values are used.
#     plot : bool, optional
#         Plots the models and saves to file. The default is True.
#     **kwargs : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     lsf1d : TYPE
#         DESCRIPTION.

#     '''
#     npix   = np.shape(x2d)[0]
#     minpix = minpix if minpix is not None else 0
#     maxpix = maxpix if maxpix is not None else npix
#     seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
#     # totpix  = 2*numpix*subpix+1
    
#     # pixcens = np.linspace(-numpix,numpix,totpix)
#     # lsf1d   = aux.get_empty_lsf('spline',numseg,totpix,pixcens)
#     parnames = gp_aux.parnames_lfc.copy()
#     if model_scatter:
#         parnames = gp_aux.parnames_all.copy()
#     lsf1d = aux.get_empty_lsf(numseg, n_data=500, n_sct=40, pars=parnames)
#     # lsf1d = []
#     count = 0
    
#     # x2d_shared    = multiprocessing.Array(ctypes.c_double, x2d)
#     # flx2d_shared  = multiprocessing.Array(ctypes.c_double, flx2d)
#     # err2d_shared  = multiprocessing.Array(ctypes.c_double, err2d)
    
#     time_start = time.time()
#     mproc = True
#     if mproc:
#         with multiprocessing.Pool() as pool:
#             results = pool.map(partial(model_1si,
#                                        seglims=seglims,
#                                        x2d=x2d,
#                                        flx2d=flx2d,
#                                        err2d=err2d,
#                                        numiter=numiter,
#                                        filter=filter,
#                                        model_scatter=model_scatter,
#                                        plot=plot,
#                                        save_plot=save_plot,
#                                        metadata=metadata,
#                                        ),
#                                    range(numseg))
            
        
        
#         for i,lsf1s_out in results:
#             lsf1d[i]=copy_lsf1s_data(lsf1s_out[0],lsf1d[i])
#     else:
#         for i in range(numseg):
#             pixl = seglims[i]
#             pixr = seglims[i+1]
#             x1s  = np.ravel(x2d[pixl:pixr])
#             flx1s = np.ravel(flx2d[pixl:pixr])
#             err1s = np.ravel(err2d[pixl:pixr])
#             checksum = aux.get_checksum(x1s, flx1s, err1s,uniqueid=i)
#             print(f"segment = {i+1}/{len(lsf1d)}")
#             # kwargs = {'numiter':numiter}
#             try:
#                 metadata.update({'segment':i+1,'checksum':checksum})
#             except:
#                 pass
#             out  = model_1s(x1s,flx1s,err1s,numiter=numiter,
#                             filter=filter,model_scatter=model_scatter,
#                             plot=plot,metadata=metadata,
#                             **kwargs)
#             if out is not None:
#                 pass
#             else:
#                 continue
#             lsf1s_out = out
#             # lsf1s_out['pixl'] = pixl
#             # lsf1s_out['pixr'] = pixr
#             # lsf1s_out['segm'] = i
#             lsf1d[i]=copy_lsf1s_data(lsf1s_out[0],lsf1d[i])
#             lsf1d[i]['ledge'] = pixl
#             lsf1d[i]['redge'] = pixr
#             lsf1d[i]['segm'] = i
#             # lsf1d.append(lsf1s)
#     time_pass = (time.time() - time_start)/60.
#     print(f"time = {time_pass:>8.3f} min")
    
#     return lsf1d

def worker(item, q):
    order, pixl, pixr = item
    

def model_1si(i,seglims,x2d,flx2d,err2d,numiter=5,filter=None,model_scatter=False,
                    plot=False,save_plot=False,metadata=None,
                    **kwargs):
    logger = logging.getLogger(__name__)
    pixl = seglims[i]
    pixr = seglims[i+1]
    x1s  = np.ravel(x2d[pixl:pixr])
    flx1s = np.ravel(flx2d[pixl:pixr])
    err1s = np.ravel(err2d[pixl:pixr])
    checksum = aux.get_checksum(x1s, flx1s, err1s,uniqueid=i)
    
    try:
        metadata.update({'segment':i+1,'checksum':checksum})
    except:
        pass
    out  = model_1s(x1s,flx1s,err1s,numiter=numiter,
                    filter=filter,model_scatter=model_scatter,
                    plot=plot,metadata=metadata,
                    **kwargs)
    if out is not None:
        out['ledge'] = pixl
        out['redge'] = pixr
        out['segm'] = i
    else:
        out = None
    return i, out

def model_1s_(od,pixl,pixr,x2d,flx2d,err2d,numiter=5,filter=None,model_scatter=False,
                    plot=False,save_plot=False,metadata=None,logger=None,
                    **kwargs):
    # pixl = seglims[i]
    pixr = pixr-1
    x1s  = np.ravel(x2d[od,pixl:pixr])
    flx1s = np.ravel(flx2d[od,pixl:pixr])
    err1s = np.ravel(err2d[od,pixl:pixr])
    checksum = aux.get_checksum(x1s, flx1s, err1s,uniqueid=pixl+pixr+od)
    # print(f"segment = {i+1}/{len(seglims)-1}")
    try:
        metadata.update({'checksum':checksum})
    except:
        pass
    metadata.update({'order':od})
    segm = int(divmod((pixl+pixr)/2.,(pixr-pixl))[0])
    metadata.update({'segment':segm})
    logger.info(f"Order, segment : {od}, {segm}")
    out  = model_1s(x1s,flx1s,err1s,numiter=numiter,
                    filter=filter,model_scatter=model_scatter,
                    plot=plot,save_plot=save_plot,
                    metadata=metadata,logger=logger,
                    **kwargs)
    if out is not None:
        out['ledge'] = pixl
        out['redge'] = pixr
        out['order'] = od
        out['segm'] = segm
    else:
        out = None
    return out

#@profile
def stack_segment(x_star,f_star,x1s,flx1s,err1s,minima_x,scale='pixel'):
    '''
    

    Parameters
    ----------
    x_star : list
        line centres.
    f_star : list
        line brightness.
    x1s : array-like
        data x-coordinates.
    flx1s : array-like
        data y-coordinates.
    err1s : array-like
        data y-coordinate errors.

    Returns
    -------
    x_stacked : TYPE
        DESCRIPTION.
    y_stacked : TYPE
        DESCRIPTION.
    err_stacked : TYPE
        DESCRIPTION.

    '''
    assert len(x1s)==len(flx1s)==len(err1s)
    assert len(x_star)==len(f_star)==(len(minima_x)-1)
    assert scale in ['pixel','velocity']
    
    N     = len(minima_x)-1
    X     = np.zeros_like(x1s,dtype=np.float32)
    Y     = np.zeros_like(flx1s)
    Y_err = np.zeros_like(err1s)
    
    for i in range(N):
        pixl,pixr = minima_x[i],minima_x[i+1]
        _         = slice(pixl,pixr)
        print(i,_,x_star[i])
        X[_]      = x1s[_] - x_star[i]
        Y[_]      = flx1s[_] / f_star[i]
        Y_err[_]  = err1s[_] / f_star[i]
        
    
    
    return X, Y, Y_err

def get_initial_guess(x1s,flx1s,err1s,minima_x):
    '''
    Returns the locations of minima between LFC lines and the initial guess
    for line positions and brightness. 

    Parameters
    ----------
    x1s : array
        data x-coordinates.
    flx1s : array
        data y-coordinates.
    err1s : array
        data y-coordinate error.
    minima_x : list
        list of points separating LFC lines.

    Returns
    -------
    minima_x : list
        A list of x-coordinates for minima in the input data.
    x_star_0 : list
        A list of LFC line centroids (centre of mass).
    f_star_0 : list
        A list of LFC line brigntess (sum of flux).

    '''
    # detect minima in the data, lines are between minima
    
    npix = len(x1s)
    nlines = len(minima_x)-1
    
    x_star_0 = np.zeros(nlines)
    f_star_0 = np.zeros(nlines)
    for i in range(nlines):
        lpix, rpix = minima_x[i], minima_x[i+1]
        if lpix==0:
            lpix = 1
        if rpix==npix-1:
            rpix = npix-2 
        
        x = x1s[lpix-1:rpix+1]
        f = flx1s[lpix-1:rpix+1]
        e = err1s[lpix-1:rpix+1]
        # bkgx = background[lpix-1:rpix+1]
        # envx = envelope[lpix-1:rpix+1]
        fit_result = hfit.gauss(x,f,e,line_model='SingleGaussian')
        success, pars,errs,chisq,chisqnu,integral = fit_result
        x_star_0[i] = pars[1]
        f_star_0[i] = integral
    
    # for i in range(nlines):
    #     l,r = minima_x[i], minima_x[i+1]
    #     x_star_0[i] = np.average(x1s[l:r],weights=flx1s[l:r])
    #     f_star_0[i] = np.sum(flx1s[l:r])
        
    return minima_x, x_star_0, f_star_0

# def segment(od,pixl,pixr,x2d,flx2d,err2d,iter_cent=10,iter_solve=10,filter=None,
#             model_scatter=False,remove_outliers=False,
#             plot=False,save_plot=False,metadata=None,logger=None,
#             debug=True,**kwargs):
#     logger = logger if logger is not None else logging.getLogger(__name__)
#     verbose          = kwargs.pop('verbose',False)

#     # slice the data appropriately
#     _     = [od,slice(pixl,pixr)]
#     x1s   = x2d[_]
#     flx1s = flx2d[_]
#     err1s = err2d[_]
#     # get minima positions in integer values
#     maxima,minima = pkd.peakdetect_derivatives(flx1s,x1s)
#     minima_x, minima_y = np.transpose(minima)
#     minima_x = minima_x.astype(int)
    
#     condition = False
#     for it in range(iter_solve):
#         if it==0:
#             x_star, f_star = get_initial_guess(x1s, flx1s, err1s)
#             X, Y, Y_err = stack_segment(x_star, f_star, 
#                                         x1s, flx1s, err1s, minima_x)
#         else:
#             x_star, f_star = solve_line()
            
#         lsf1d = 
#         if condition:
#             break
        
    
    
    
#     # lsf1s = model_1s_(od,pixl,pixr,x2d,flx2d,err2d,numiter=iter_cent,
#     #                   filter=filter,model_scatter=model_scatter,plot=plot,
#     #                   save_plot=save_plot,metadata=metadata,logger=logger,
#     #                   remove_outliers=remove_outliers,debug=debug)
    
#     return 

def model_1s(pix1s,flx1s,err1s,numiter=5,filter=None,model_scatter=False,
             remove_outliers=True,
             plot=False,save_plot=False,metadata=None,logger=None,
             debug=True,**kwargs):
    '''
    Constructs the LSF model for a single segment
    '''
    logger = logger if logger is not None else logging.getLogger(__name__)
    verbose          = kwargs.pop('verbose',False)
    # print(f'filter={filter}')
    pix1s, flx1s, err1s = aux.clean_input(pix1s,flx1s,err1s,sort=True,
                                      verbose=verbose,filter=filter)
    # print("shape pix1s = ",np.shape(pix1s))
    if len(pix1s)==0:
        return None
    
    
        
    shift    = 0
    oldshift = 1
    relchange = 1
    delta     = 100
    shift_j  = 0
    keep_full = np.full_like(pix1s, True, dtype=bool)
    keep_jm1  = keep_full
    args = {}
    metadata.update({'model_scatter':model_scatter})
    for j in range(numiter):
        # shift the values along x-axis for improved centering
        # remove outliers from last iteration
        if np.abs(shift)>1: shift=np.sign(shift)*0.25
        pix1s_j = (pix1s + shift)[keep_jm1]
        flx1s_j = flx1s[keep_jm1]
        err1s_j = err1s[keep_jm1]
        # pix1s = pix1s+shift  
        # pix1s = pix1s[keep] + shift
        # flx1s = flx1s[keep]
        # err1s = err1s[keep]
        # args.update({#'numpix':numpix,
        #              #'subpix':subpix,
        #              # 'metadata':metadata,
        #              'plot':plot,
        #              'filter':filter,
        #              #'minpts':minpts,
        #              'model_scatter':model_scatter})
        dictionary=construct_tinygp(pix1s_j,flx1s_j,err1s_j, 
                                    plot=plot,
                                    # metadata=metadata,
                                    filter=filter,model_scatter=model_scatter)
        lsf1s  = dictionary['lsf1s']
        # save shift from previous iteration
        shift_jm1 = shift_j
        # update this iterations shift
        shift_j  = dictionary['lsfcen']
        # update total shift
        shift += shift_j
        # shift = shift_j
        
        cenerr = dictionary['lsfcen_err']
        chisq  = dictionary['chisq']
        rsd    = dictionary['rsd']
        # remove outliers in residuals before proceeding with next iteration
        if remove_outliers:
            outliers_j   = hf.is_outlier_original(rsd)
            cut          = np.where(outliers_j==True)
            keep_full[cut] = False
            keep_jm1 =  keep_full
            keep_full = np.full_like(pix1s,True,dtype='bool')
        else:
            keep_jm1 = np.full_like(pix1s,True,dtype='bool')
        
        delta = np.abs(shift_j - shift_jm1)
        
        dictionary.update({'shift':shift})
        dictionary.update({'scale':metadata['scale'][:3]})
        if debug:
            logger.info(f"iter {j:2d}   shift={shift:+5.2e}  " + \
                  f"delta={delta:5.2e}   " +\
                  f"N={len(rsd)}  chisq={chisq:6.2f}")
        
        # break if
        # 1. change in LSF centre smaller than delta (pix) or
        # 2. total shift smaller than 1e-3 (pix)
        # 3. iteration number equal to iteration limit
        condition = (delta<1e-3 or shift<=1e-3 or j==numiter-1)
        if condition and j>0:
            # if debug:
                # logger.info('stopping condition satisfied')
            if plot:
                # logger.info(f'Plotting, plot={plot}, save_plot={save_plot}')
                plotfunction = lsfplot.plot_solution
                LSF_solution = dictionary['LSF_solution']
                scatter      = dictionary['scatter']
                plotkwargs = dict(params_LSF=LSF_solution, 
                                  scatter=scatter, 
                                  metadata=metadata, 
                                  save=save_plot,
                                  shift=shift,
                                  **kwargs)
                plotfunction(pix1s_j, flx1s_j, err1s_j, **plotkwargs)
                # if model_scatter==True: #plot also the solution without scatter
                #     LSF_solution = dictionary['LSF_solution_nosct']
                #     scatter      = None
                #     plotkwargs = dict(params_LSF=LSF_solution, 
                #                       scatter=None, 
                #                       metadata=metadata, 
                #                       save=save_plot,
                #                       shift=shift,
                #                       **kwargs)
                #     plotfunction(pix1s_j, flx1s_j, err1s_j, **plotkwargs)
                
                
            break
        else:
            for variable in [dictionary, lsf1s, shift, cenerr, chisq, rsd]:
                del(variable)
        
        # if plot and j==numiter-1:
    if debug:      
        logger.info(f'total shift : {shift*1e3:12.6f} mpix '+\
                    f'after {j} iterations, rmv_outliers:{remove_outliers}'+\
                    f' (delta={delta:+6.2f}, chisq={chisq:6.2f})')   
    
    # save the total number of points used
    lsf1s['numlines'] = len(pix1s_j)
    return lsf1s

# def model_1s_from_file(filename,order,pixl,pixr,scale='pix',fittype='gauss',
#                        numiter=5,model_scatter=True,plot=False,filter=None,
#                        **kwargs):
#     X,Y,Y_err = read.get_data(filename,order,pixl,pixr,
#                               scale=scale,fittype=fittype, filter=filter)
#     checksum = aux.get_checksum(X, Y, Y_err, uniqueid=None)
#     # print(f"segment = {i+1}/{len(seglims)-1}")
#     # try:
#     #     metadata.update({'pixl':pixl,'pixr':pixr,'checksum':checksum})
#     # except:
#     #     pass
#     out  = model_1s(X,Y,Y_err,numiter=numiter,
#                     filter=filter,model_scatter=model_scatter,
#                     plot=plot,metadata=metadata,
#                     **kwargs)
#     if out is not None:
#         out['pixl'] = pixl
#         out['pixr'] = pixr
#         # out['segm'] = i
#     else:
#         out = None
#     return i, out

def construct_tinygp(x,y,y_err,plot=False,
                     filter=None,N_test=20,model_scatter=False,**kwargs):
    '''
    Returns the LSF model for one segment using TinyGP framework

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    y_err : TYPE
        DESCRIPTION.
    numpix : TYPE
        DESCRIPTION.
    subpix : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    checksum : TYPE, optional
        DESCRIPTION. The default is None.
    filter : TYPE, optional
        DESCRIPTION. The default is 10.
    N_test : TYPE, optional
        DESCRIPTION. The default is 400.
    model_scatter : TYPE, optional
        DESCRIPTION. The default is True.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    out_dict : TYPE
        DESCRIPTION.

    '''
    X        = jnp.array(x)
    Y        = jnp.array(y)
    Y_err    = jnp.array(y_err)
    assert len(X)==len(Y)==len(Y_err)
    N_data   = len(X)
    
    LSF_solution_nosct = lsfgp.train_LSF_tinygp(X,Y,Y_err,scatter=None)
    
    if model_scatter:
        scatter = lsfgp.train_scatter_tinygp(X,Y,Y_err,LSF_solution_nosct)
        LSF_solution = lsfgp.train_LSF_tinygp(X,Y,Y_err,scatter)
    else:
        scatter=None
        LSF_solution = LSF_solution_nosct
        
    gp = lsfgp.build_LSF_GP(LSF_solution,X,Y,Y_err,scatter)
    
    # --------  Save output -------- 
    
    if scatter is not None:
        parnames = gp_aux.parnames_lfc.copy() + gp_aux.parnames_sct.copy()
        assert len(scatter[1])==len(scatter[2])
        N_sct = len(scatter[1])
    else:
        parnames = gp_aux.parnames_lfc.copy()
        N_sct = 0
        
    # Initialize an LSF for this segment
    lsf1s    = aux._prepare_lsf1s(N_data,N_sct,pars=parnames)
    # if model_scatter:
        # parnames_ = gp_aux.parnames_lfc.copy()
        # lsf1s_nosct = aux._prepare_lsf1s(N_data,N_sct=0,pars=parnames_)
    
    # Save parameters
    # The parameters are saved in gp_aux
    npars = 0
    # print(gp_aux.parnames_lfc)
    for parname in gp_aux.parnames_lfc:
        lsf1s[parname] = LSF_solution[parname]
        npars = npars + 1
    if scatter is not None:
        for parname in gp_aux.parnames_sct:
            lsf1s[parname] = scatter[0][parname]
            npars = npars + 1
        # for parname in gp_aux.parnames_lfc:
        #     lsf1s_nosct[parname] = LSF_solution_nosct[parname]
            # npars = npars + 1
        
    # Save data that was used to create the GP models (needed for conditioning)
    lsf1s['data_x']    = X
    lsf1s['data_y']    = Y
    lsf1s['data_yerr']    = Y_err
    if model_scatter:
        lsf1s['sct_x']     = scatter[1]
        lsf1s['sct_y']     = scatter[2]
        lsf1s['sct_yerr']  = scatter[3]
        
        
    
    Y_data_err = Y_err
    if scatter is not None:
        S, S_var = lsfgp.rescale_errors(scatter, X, Y_err)
        Y_data_err = S
        
        
    # # Now condition on the same grid as data to calculate residual
    
    logL, cond    = gp.condition(Y, X)
    lsf1s['logL'] = logL
    # Y_mod_err  = np.sqrt(cond.variance)
    # Y_tot_err  = jnp.sqrt(np.sum(np.power([Y_data_err,Y_mod_err],2.),axis=0))
    rsd        = lsfgp.get_residuals(X, Y, Y_data_err, LSF_solution)
    dof        = len(rsd) - npars
    chisq      = np.sum(rsd**2)
    chisqdof   = chisq / dof
    centre_estimator = lsfgp.estimate_centre_anderson
    # centre_estimator = lsfgp.estimate_centre_median
    # centre_estimator = lsfgp.estimate_centre_mean
    
    lsfcen, lsfcen_err = centre_estimator(X, Y, Y_err,
                                          LSF_solution,scatter=scatter)
    out_dict = dict(lsf1s=lsf1s, lsfcen=lsfcen, lsfcen_err=lsfcen_err,
                    chisq=chisqdof, rsd=rsd, 
                    LSF_solution=LSF_solution,
                    LSF_solution_nosct = LSF_solution_nosct)
    out_dict.update(dict(model_scatter=model_scatter))
    out_dict.update(dict(scatter=scatter))
        # out_dict.update(dict(lsf1s_nosct=lsf1s_nosct))
    gc.collect()
    return out_dict


def copy_lsf1s_data(copy_from,copy_to):
    # print(copy_from.dtype.names)
    # print(copy_to.dtype.names)
    assert copy_from.dtype.names == copy_to.dtype.names
    names = copy_from.dtype.names
    for name in names:
        try:
            # Data can be directly copied
            copy_to[name] = copy_from[name]
        except:
            # Array lengths do not match so copy only where needed
            len_data  = len(copy_from[name])
            copy_to[name] = np.nan
            copy_to[name][slice(0,len_data)] = copy_from[name]
            
    return copy_to



class SequenceIterator:
    # Based in part on https://realpython.com/python-iterators-iterables/
    def __init__(self,orders,seglimits):
        self._index = 0
        self._orders = orders
        self._seglimits = seglimits
        self._index_od   = 0
        self._current_od = self._orders[0]
        self._index_seg  = 0
        self._current_seg = self._seglimits[0],self._seglimits[1]
        self._current = (self._current_od,*self._current_seg)
        
        self._max_od = len(orders)
        self._max_seg = len(seglimits)-1
        self._max = len(orders)*(len(seglimits)-1)
    def __len__(self):
        return self._max
    def __iter__(self):
        return self
    def __next__(self):
        item = (self._current_od,*self._current_seg)
        self._index += 1
        cond1 = self._index_seg < self._max_seg-1
        cond2 = self._index_od < self._max_od-1
        cond3 = self._index < self._max + 1
        if cond3:
            try:
                self._next_segment()
            except:
                try:
                    self._next_order()
                except:
                    pass
            return item
            # if cond1: # is not last segment in the order
            #     self._next_segment()
            #     return item
            # elif not cond1 and cond2: # is last segment but is not last order
            #     self._next_order()
            #     return item
            # elif cond1 and not cond2: # is not last segment but is last order
            #     self._next_segment()
            #     return item
            # elif not cond1 and not cond2: # is last segment and is last order
            #     return item
        else:
            raise StopIteration
    def _next_order(self):
        self._index_od += 1
        self._current_od = self._orders[self._index_od]
        self._index_seg = 0
        self._current_seg = (self._seglimits[self._index_seg],
                             self._seglimits[self._index_seg+1])
    def _next_segment(self):
        self._index_seg +=1 
        self._current_seg = (self._seglimits[self._index_seg],
                             self._seglimits[self._index_seg+1])


def from_spectrum_2d(spec,orders,iteration,scale='pixel',iter_center=5,
                  numseg=16,minpix=None,maxpix=None,filter=None,
                  model_scatter=True,save_fits=True,clobber=False,
                  plot=False,save_plot=False,
                  interpolate=False,update_linelist=True):
    assert scale in ['pixel','velocity']
    assert iteration>0
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger(__name__)
    
    version = hv.item_to_version(dict(iteration=iteration,
                                        model_scatter=model_scatter,
                                        interpolate=interpolate
                                        ),
                                   ftype='lsf'
                                   )
    logging.info(f'{__name__}, subbkg = {hs.subbkg}, divenv = {hs.divenv} ')
    pix3d,vel3d,flx3d,err3d,orders_=aux.stack_spectrum(spec,version,
                                                       subbkg=hs.subbkg,
                                                       divenv=hs.divenv)
    if scale=='pixel':
        x2d = pix3d[:,:,0]
    elif scale=='velocity':
        x2d = vel3d[:,:,0]
    flx2d = flx3d[:,:,0]
    err2d = err3d[:,:,0]
    
    
    metadata = dict(
        scale=scale,
        # order=order,
        iteration=iteration,
        model_scatter=model_scatter,
        interpolate=interpolate
        )
    
    # print(np.shape(x2d))
    npix   = np.shape(x2d)[1]
    minpix = minpix if minpix is not None else 0
    maxpix = maxpix if maxpix is not None else npix
    seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
    iterator = SequenceIterator(orders,seglims)
    
    parnames = gp_aux.parnames_lfc.copy()
    if model_scatter:
        parnames = gp_aux.parnames_all.copy()
    lsf2d = aux.get_empty_lsf(len(iterator), 
                              n_data=600, n_sct=40, pars=parnames)
    
    
    time_start = time.time()
    
    
    
    option=2
    partial_function = partial(model_1s_,
                                x2d=x2d,
                                flx2d=flx2d,
                                err2d=err2d,
                                numiter=iter_center,
                                filter=filter,
                                model_scatter=model_scatter,
                                plot=plot,
                                save_plot=save_plot,
                                metadata=metadata,
                                )
    if option==1:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(partial_function,
                                    iterator)
    elif option==2:
        # job_queue = multiprocessing.Queue(maxsize=8)
        # results   = multiprocessing.Queue()
        # for item in iterator:
            # job_queue.put(item)
        logger.info('Starting LSF fitting')
        manager = multiprocessing.Manager()
        inq = manager.Queue()
        outq = manager.Queue()
    
        # construct the workers
        nproc = multiprocessing.cpu_count()
        logger.info(f"Using {nproc} workers")
        workers = [Worker(str(name+1), partial_function,inq, outq) 
                   for name in range(nproc)]
        for worker in workers:
            worker.start()
    
        # add data to the queue for processing
        work_len = len(iterator)
        for item in iterator:
            # print(f"Item before putting into queue: {item}")
            inq.put(item)
    
        while outq.qsize() < work_len:
            # waiting for workers to finish
            done = outq.qsize()
            progress = done/(work_len)
            time_elapsed = time.time() - time_start
            progress_bar.update(progress,name='LSF_2d',
                               time=time_elapsed,
                               logger=None)
            
            # print("Waiting for workers. Out queue size {}".format(outq.qsize()))
            time.sleep(1)
    
        # clean up
        for worker in workers:
            worker.terminate()
    
        # print the outputs
        results = []
        while not outq.empty():
            results.append(outq.get())
    
    
    for i,lsf1s_out in enumerate(results):
        lsf2d[i]=copy_lsf1s_data(lsf1s_out[0],lsf2d[i])
    worktime = (time.time() - time_start)
    h, m, s = progress_bar.get_time(worktime)
    logger.info(f"Total time elapsed = {h:02d}h {m:02d}m {s:02d}s")
    
    if save_fits:
        
        # Save GP parameters and data
        lsf_filepath = hio.get_fits_path('lsf',spec.filepath)
        lio.write_lsf_to_fits(lsf2d, lsf_filepath, f"{scale}_gp",
                              version=version,
                              clobber=clobber)   
        # Save LSF numerical models
        nummodel_lsf = numerical_models(lsf2d,xrange=(-6,6),subpix=50)
        lio.write_lsf_to_fits(nummodel_lsf, lsf_filepath, f"{scale}_model",
                              version=version,
                              clobber=clobber)   
    gc.collect()
    
    return lsf2d

# def worker(input_queue,output_queue,function):
#     item = input_queue.get(timeout=10)
#     result = function(*item)
#     output_queue.put(result)
#     return None

class Worker(multiprocessing.Process):
    """
    Simple worker.
    """

    def __init__(self, name, function, in_queue, out_queue):
        super(Worker, self).__init__()
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
            result = self.function(*item,logger=self.logger)
            self.out_queue.put(result)
            
            
def get_lsf1s_numerical_model(lsf1s_gp,x_array):
    y_array,sct_array = evaluate_lsf1s(lsf1s_gp,x_array)
    return y_array, sct_array

def numerical_models(lsf1d_gp,xrange=(-8,8),subpix=11):
    from harps.containers import lsf_spline
    x_min, x_max = xrange
    numsegs = len(lsf1d_gp)
    npts    = (x_max - x_min) * subpix + 1
    lsf1d_model = lsf_spline(numsegs, npts)
    
    x_array = np.linspace(x_min,x_max,npts)
    lsf1d_model['x']=x_array
    for i,lsf1s_gp in enumerate(lsf1d_gp):
        y_array,sct_array     = get_lsf1s_numerical_model(lsf1s_gp,x_array)
        lsf1d_model[i]['y'] = y_array
        lsf1d_model[i]['scatter'] = sct_array
        names = lsf1d_model.dtype.names
        for name in names:
            if name not in ['x','y','scatter']:
                lsf1d_model[i][name] = lsf1s_gp[name]
        progress_bar.update(i/(len(lsf1d_gp)-1),'numerical model')
    return lsf1d_model



def evaluate_GP(GP,y_data,x_test):
    _, cond = GP.condition(y_data,x_test)
    
    mean = cond.mean
    var  = jnp.sqrt(cond.variance)
    
    return mean, var

def build_scatter_GP_from_lsf1s(lsf1s):
    scatter    = read.scatter_from_lsf1s(lsf1s)
    scatter_gp = lsfgp.build_scatter_GP(scatter[0],
                                         X=scatter[1],
                                         Y_err=scatter[3])
    return scatter_gp

def evaluate_scatter_GP_from_lsf1s(lsf1s,x_test):
    theta_sct, sct_x, sct_y, sct_yerr  = read.scatter_from_lsf1s(lsf1s)
    sct_gp = lsfgp.build_scatter_GP(theta_sct,sct_x,sct_yerr)
   
    return evaluate_GP(sct_gp, sct_y, x_test)


    
def build_LSF_GP_from_lsf1s(lsf1s,return_scatter=False):
    theta_LSF, data_x, data_y, data_yerr = read.LSF_from_lsf1s(lsf1s)
    scatter  = read.scatter_from_lsf1s(lsf1s)
    LSF_gp = lsfgp.build_LSF_GP(theta_LSF, data_x, data_y, data_yerr,
                                 scatter=scatter)
    if return_scatter:
        return LSF_gp, scatter
    else:
        return LSF_gp

def evaluate_LSF_GP_from_lsf1s(lsf1s,x_test):
    theta_LSF, data_x, data_y, data_yerr = read.LSF_from_lsf1s(lsf1s)
    scatter = read.scatter_from_lsf1s(lsf1s)
    LSF_gp = lsfgp.build_LSF_GP(theta_LSF, data_x, data_y, data_yerr,
                                 scatter=scatter)
    
    return evaluate_GP(LSF_gp, data_y, x_test)




def evaluate_lsf1s(lsf1s_gp,x_test):
    return evaluate_LSF_GP_from_lsf1s(lsf1s_gp,x_test)
    

def lsf_1d(fittype,linelist1d,x1d_stacked,flx1d_stacked,err1d_stacked,
           iter_center=5,numseg=16,model_scatter=True,metadata=None):
    
    
    plot=False; save_plot=False
    # if scale=='pixel':
    #     x1d = pix1d
    # elif scale=='velocity':
    #     x1d = vel1d
    metadata_=dict(
        # order=od,
        # scale=scale,
        model_scatter=model_scatter,
        # iteration=iteration,
        )
    if metadata is not None:
        metadata.update(metadata_)
    else:
        metadata = metadata_
    lsf1d=models_1d(x1d_stacked,flx1d_stacked,err1d_stacked,
                              numseg=numseg,
                              numiter=iter_center,
                              minpts=15,
                              model_scatter=model_scatter,
                              minpix=None,maxpix=None,
                              filter=None,plot=plot,
                              metadata=metadata,
                              save_plot=save_plot)
    
    return lsf1d


import itertools

def get_most_likely_lsf2d(lsfpath,scale,nbo=72,nseg=16):
    data = {}
    with FITS(lsfpath) as hdul:
        for ext in hdul:
            extname = ext.get_extname()
            extver  = ext.get_extver()
            if extver==511: continue
            if extname==f'{scale}_gp':
                data[extver] = ext.read()
    numver = len(data)
    dtype = np.dtype([('version',int, (numver)),
                      ('order',int, ()),
                      ('segm',int, ()),
                      ('logL',np.float32, (numver)),
                      ('loc',int, (numver)),
                      ])
    
    array = np.zeros(nbo*nseg,dtype=dtype)
    comb=itertools.product(np.arange(nbo),np.arange(nseg))
    for i,(od,segm) in enumerate(comb):
        
        for j, (ver,lsf2d) in enumerate(data.items()):
            odver = np.unique(lsf2d['order'])
            segver = np.unique(lsf2d['segm'])
            if od in odver and segm in segver:
                pass
            else:
                continue
            array[i]['order']=od
            array[i]['segm']=segm
            cut = np.where((lsf2d['order']==od)&(lsf2d['segm']==segm))[0]
            print(i,od,segm,j,ver,cut)
            array['version'][i,j] = ver
            array['loc'][i,j] = cut
            
            try: 
                array['logL'][i,j] = lsf2d[cut]['logL']
            except:
                array['logL'][i,j] = gp_aux.get_likelihood_from_lsf1s(lsf2d[cut])
    nonzero = np.where(array['order']!=0)
    array = array[nonzero]
    # find the location of the maximum in log likelihood
    best = np.argmax(array['logL'],axis=1)
    
    most_likely_lsf2d = []
    for i,entry in enumerate(array):
        
        veritem = entry['version'][best[i]]
        locitem = entry['loc'][best[i]]
        print(i,entry['order'],entry['segm'],veritem,locitem)
        most_likely_lsf2d.append(data[veritem][locitem])
    
    return np.hstack(most_likely_lsf2d)

def save_most_likely(lsf_filepath,scale,nbo=72,nseg=16,clobber=False):
    most_likely_lsf2d = get_most_likely_lsf2d(lsf_filepath,scale,nbo=72,nseg=16)
    
    nummodel_lsf = numerical_models(most_likely_lsf2d,xrange=(-6,6),subpix=50)
    lio.write_lsf_to_fits(nummodel_lsf, lsf_filepath, f"{scale}_model",
                          version=1,
                          clobber=clobber)  
    