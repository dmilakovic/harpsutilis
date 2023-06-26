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
import hashlib
import matplotlib.pyplot as plt
# import scipy.interpolate as interpolate
import gc
import multiprocessing
from functools import partial
import time
import ctypes



def models_2d(x3d, flx3d, err3d, orders, filename, scale,
                  numseg=16,numpix=7,subpix=4,numiter=5,filter=None,**kwargs):
    assert scale in ['pixel','velocity'], "Scale not known"
    lst = []
    for i,od in enumerate(orders):
        print("order = {}".format(od))
        plot=False
        lsf1d=(models_1d(x3d[od],flx3d[od],err3d[od],numseg,numiter,
                         filter=filter,plot=plot,**kwargs))
        lsf1d['order'] = od
        lst.append(lsf1d)
        # filepath = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/'+\
        #            f'ESPRESSO_{od}_{scale}.fits'
        # with FITS(filepath,mode='rw') as hdu:
        #     hdu.write(lsf1d,extname='{}'.format(od))
        # hdu.close()
        # print("File saved to {}".format(filepath))
        if len(orders)>1:
            hf.update_progress((i+1)/len(orders),'Fit LSF')
    lsf = np.hstack(lst)
    
    return lsf

def models_1d(x2d,flx2d,err2d,numseg=16,numiter=5,minpts=10,model_scatter=False,
              minpix=None,maxpix=None,filter=None,plot=True,save_plot=False,
              metadata=None,*args,
                    **kwargs):
    '''
    

    Parameters
    ----------
    x2d : 2d array
        Array containing pixel or velocity (km/s) values.
    flx2d : 2d array
        Array containing normalised flux values.
    err2d : 2d array
        Array containing errors on flux.
    method : str
        Method to use for LSF reconstruction. Options: 'gp','spline','analytic'
    numseg : int, optional
        Number of segments along the main dispersion (x-axis) direction. 
        The default is 16.
    numpix : int, optional
        Distance (in pixels or km/s) each side of the line centre to use. 
        The default is 8 (assumes pixels).
    subpix : int, optional
        The number of divisions of each pixel or km/s bin. The default is 4.
    numiter : int, optional
        DESCRIPTION. The default is 5.
    minpix : int, optional
        DESCRIPTION. The default is 0.
    minpts : int, optional
        Only applies when using method='spline'. The minimum number of lines 
        in each subpixel or velocity bin. The default is 10.
    filter : int, optional
        If given, the program will use every N=filter (x,y,e) values. 
        The default is None, all values are used.
    plot : bool, optional
        Plots the models and saves to file. The default is True.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    lsf1d : TYPE
        DESCRIPTION.

    '''
    npix   = np.shape(x2d)[0]
    minpix = minpix if minpix is not None else 0
    maxpix = maxpix if maxpix is not None else npix
    seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
    # totpix  = 2*numpix*subpix+1
    
    # pixcens = np.linspace(-numpix,numpix,totpix)
    # lsf1d   = aux.get_empty_lsf('spline',numseg,totpix,pixcens)
    parnames = gp_aux.parnames_lfc.copy()
    if model_scatter:
        parnames = gp_aux.parnames_all.copy()
    lsf1d = aux.get_empty_lsf(numseg, n_data=500, n_sct=40, pars=parnames)
    # lsf1d = []
    count = 0
    
    # x2d_shared    = multiprocessing.Array(ctypes.c_double, x2d)
    # flx2d_shared  = multiprocessing.Array(ctypes.c_double, flx2d)
    # err2d_shared  = multiprocessing.Array(ctypes.c_double, err2d)
    
    time_start = time.time()
    mproc = True
    if mproc:
        with multiprocessing.Pool() as pool:
            results = pool.map(partial(model_1si,
                                       seglims=seglims,
                                       x2d=x2d,
                                       flx2d=flx2d,
                                       err2d=err2d,
                                       numiter=numiter,
                                       filter=filter,
                                       model_scatter=model_scatter,
                                       plot=plot,
                                       save_plot=save_plot,
                                       metadata=metadata,
                                       ),
                                   range(numseg))
            
        
        
        for i,lsf1s_out in results:
            lsf1d[i]=copy_lsf1s_data(lsf1s_out[0],lsf1d[i])
    else:
        for i in range(numseg):
            pixl = seglims[i]
            pixr = seglims[i+1]
            x1s  = np.ravel(x2d[pixl:pixr])
            flx1s = np.ravel(flx2d[pixl:pixr])
            err1s = np.ravel(err2d[pixl:pixr])
            checksum = aux.get_checksum(x1s, flx1s, err1s,uniqueid=i)
            print(f"segment = {i+1}/{len(lsf1d)}")
            # kwargs = {'numiter':numiter}
            try:
                metadata.update({'segment':i+1,'checksum':checksum})
            except:
                pass
            out  = model_1s(x1s,flx1s,err1s,numiter=numiter,
                            filter=filter,model_scatter=model_scatter,
                            plot=plot,metadata=metadata,
                            **kwargs)
            if out is not None:
                pass
            else:
                continue
            lsf1s_out = out
            # lsf1s_out['pixl'] = pixl
            # lsf1s_out['pixr'] = pixr
            # lsf1s_out['segm'] = i
            lsf1d[i]=copy_lsf1s_data(lsf1s_out[0],lsf1d[i])
            lsf1d[i]['pixl'] = pixl
            lsf1d[i]['pixr'] = pixr
            lsf1d[i]['segm'] = i
            # lsf1d.append(lsf1s)
    time_pass = (time.time() - time_start)/60.
    print(f"time = {time_pass:>8.3f} min")
    
    return lsf1d

def worker(item, q):
    order, pixl, pixr = item
    

def model_1si(i,seglims,x2d,flx2d,err2d,numiter=5,filter=None,model_scatter=False,
                    plot=False,save_plot=False,metadata=None,
                    **kwargs):
    pixl = seglims[i]
    pixr = seglims[i+1]
    x1s  = np.ravel(x2d[pixl:pixr])
    flx1s = np.ravel(flx2d[pixl:pixr])
    err1s = np.ravel(err2d[pixl:pixr])
    checksum = aux.get_checksum(x1s, flx1s, err1s,uniqueid=i)
    print(f"segment = {i+1}/{len(seglims)-1}")
    try:
        metadata.update({'segment':i+1,'checksum':checksum})
    except:
        pass
    out  = model_1s(x1s,flx1s,err1s,numiter=numiter,
                    filter=filter,model_scatter=model_scatter,
                    plot=plot,metadata=metadata,
                    **kwargs)
    if out is not None:
        out['pixl'] = pixl
        out['pixr'] = pixr
        out['segm'] = i
    else:
        out = None
    return i, out

def model_1s_(od,pixl,pixr,x2d,flx2d,err2d,numiter=5,filter=None,model_scatter=False,
                    plot=False,save_plot=False,metadata=None,
                    **kwargs):
    # pixl = seglims[i]
    # pixr = seglims[i+1]
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
    out  = model_1s(x1s,flx1s,err1s,numiter=numiter,
                    filter=filter,model_scatter=model_scatter,
                    plot=plot,metadata=metadata,
                    **kwargs)
    if out is not None:
        out['pixl'] = pixl
        out['pixr'] = pixr
        out['order'] = od
        out['segm'] = segm
    else:
        out = None
    return out

#@profile
def model_1s(pix1s,flx1s,err1s,numiter=5,filter=None,model_scatter=False,
                    plot=False,save_plot=False,metadata=None,
                    **kwargs):
    '''
    Constructs the LSF model for a single segment
    '''
    # totpix  = 2*numpix*subpix+1
    # pixcens = np.linspace(-numpix,numpix,totpix)
    # pixlims = (pixcens+0.5/subpix)
    ## other keywords
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
        shift_jm1 = shift_j
        shift_j  = dictionary['lsfcen']
        shift += shift_j
        # shift = shift_j
        
        cenerr = dictionary['lsfcen_err']
        chisq  = dictionary['chisq']
        rsd    = dictionary['rsd']
        # # remove outliers in residuals before proceeding with next iteration
        outliers_j   = hf.is_outlier_original(rsd)
        cut          = np.where(outliers_j==True)
        keep_full[cut] = False
        keep_jm1 =  keep_full
        keep_full = np.full_like(pix1s,True,dtype='bool')
        
        delta = np.abs(shift_j - shift_jm1)
        
        dictionary.update({'shift':shift})
        dictionary.update({'scale':metadata['scale'][:3]})
        
        print(f"iter {j:2d}   shift={shift:+5.2e}  " + \
              f"delta={delta:5.2e}   " +\
              f"N={len(rsd)}  chisq={chisq:6.2f}")
        
        # oldshift = shift
        if (delta<1e-3 or j==numiter-1) and j>0:
            print('stopping condition satisfied')
            if plot:
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
                if model_scatter==True: #plot also the solution without scatter
                    LSF_solution = dictionary['LSF_solution_nosct']
                    scatter      = None
                    plotkwargs = dict(params_LSF=LSF_solution, 
                                      scatter=None, 
                                      metadata=metadata, 
                                      save=save_plot,
                                      shift=shift,
                                      **kwargs)
                    plotfunction(pix1s_j, flx1s_j, err1s_j, **plotkwargs)
                
                
            break
        else:
            for variable in [dictionary, lsf1s, shift, cenerr, chisq, rsd]:
                del(variable)
        
        # if plot and j==numiter-1:
           
    print('total shift {0:12.6f} +/- {1:12.6f}'.format(shift*1e3,
                                                             cenerr*1e3))   
    
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
                     filter=None,N_test=10,model_scatter=False,**kwargs):
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
    
    _, cond    = gp.condition(Y, X)
    # Y_mod_err  = np.sqrt(cond.variance)
    # Y_tot_err  = jnp.sqrt(np.sum(np.power([Y_data_err,Y_mod_err],2.),axis=0))
    rsd        = lsfgp.get_residuals(X, Y, Y_data_err, LSF_solution)
    dof        = len(rsd) - npars
    chisq      = np.sum(rsd**2)
    chisqdof   = chisq / dof
    lsfcen, lsfcen_err = lsfgp.estimate_centre(X,Y,Y_err,
                                          LSF_solution,scatter=scatter,
                                          N=N_test)
    # lsfcen, lsfcen_err = lsfgp.estimate_centre_anderson(X, Y, Y_err, 
    #                                                     LSF_solution,
    #                                                     scatter=scatter)
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

def from_spectrum_1d(spec,order,iteration,scale='pixel',iter_center=5,
                  numseg=16,model_scatter=True,save_fits=True,clobber=False,
                  interpolate=False,update_linelist=True):
    from harps.lsf.container import LSF
    assert scale in ['pixel','velocity']
    assert iteration>0
    
    version = hv.item_to_version(dict(iteration=iteration,
                                        model_scatter=model_scatter,
                                        interpolate=interpolate
                                        ),
                                   ftype='lsf'
                                   )
    pix3d,vel3d,flx3d,err3d,orders=aux.stack_spectrum(spec,version)

    item, fittype = aux.get_linelist_item_fittype(version,fittype=None)
    print(item,fittype)
    llist = spec[item]
    cut = np.where(llist['order']==order)[0]
    linelist1d = llist[cut]
    # # print(linelist1d)
    # # sys.exit()
    
    # pix1d,vel1d,flx1d,err1d = aux.stack_1d(fittype,linelist1d,flx1d,
    #                                 wav1d,err1d,bkg1d)  
    flx1d = flx3d[order,:,0]
    # bkg1d = bkg3d[order,:,0]
    err1d = err3d[order,:,0]    
    if scale=='pixel':
        x1d = pix3d[order,:,0]
    elif scale=='velocity':
        x1d = vel3d[order,:,0]
        
    metadata = dict(
        scale=scale,
        order=order,
        iteration=iteration,
        model_scatter=model_scatter,
        interpolate=interpolate
        )
    
    lsf1d = lsf_1d(fittype,linelist1d,x1d,flx1d,err1d,iter_center=iter_center,
                   numseg=numseg,model_scatter=model_scatter,
                   metadata=metadata)
    lsf1d['order']=order
    if save_fits:
        # version = lio.convert_version(iteration,interpolate,model_scatter)
        
        lsf_filepath = hio.get_fits_path('lsf',spec.filepath)
        extname = f"{scale}"
        lio.write_lsf_to_fits(lsf1d, lsf_filepath, extname,version=version,
                          clobber=clobber)   
    gc.collect()
    if update_linelist:
        lsf_filepath = hio.get_fits_path('lsf',spec.filepath)
        new_llist = aux.solve(spec._outpath,lsf_filepath,iteration=iteration,
                              order=order,scale=scale,
                              model_scatter=model_scatter,
                              interpolate=interpolate)
    #     llist = aux.solve_1d(LSF(lsf1d),linelist1d,x1d,flx1d,
    #                             bkg1d,err1d,fittype,scale,interpolate)
    
        
    gc.collect()
    return lsf1d

class SequenceIterator:
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
    from harps.lsf.container import LSF
    assert scale in ['pixel','velocity']
    assert iteration>0
    
    version = hv.item_to_version(dict(iteration=iteration,
                                        model_scatter=model_scatter,
                                        interpolate=interpolate
                                        ),
                                   ftype='lsf'
                                   )
    pix3d,vel3d,flx3d,err3d,orders_=aux.stack_spectrum(spec,version)
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
                              n_data=500, n_sct=40, pars=parnames)
    
    
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
        manager = multiprocessing.Manager()
        inq = manager.Queue()
        outq = manager.Queue()
    
        # construct the workers
        nproc = multiprocessing.cpu_count()
        workers = [Worker(str(name), partial_function,inq, outq) for name in range(nproc)]
        for worker in workers:
            worker.start()
    
        # add data to the queue for processing
        work_len = len(iterator)
        for item in iterator:
            # print(f"Item before putting into queue: {item}")
            inq.put(item)
    
        while outq.qsize() != work_len:
            # waiting for workers to finish
            done = outq.qsize()
            progress = done/work_len
            time_elapsed = time.time() - time_start
            hf.update_progress(progress,name='LSF_2d',
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
    time_pass = (time.time() - time_start)/60.
    print(f"time = {time_pass:>8.3f} min")
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

    def run(self):
        while True:
            # grab work; do something to it (+1); then put the result on the output queue
            item = self.in_queue.get()
            # print(f'item after queue.get = {item}')
            result = self.function(*item)
            self.out_queue.put(result)
            
            
# def solve_from_spec(spec,order,lsf2d,fittype,scale='pix',interpolate=False):
    
#     if 'pix' in scale:
#         x2d = np.vstack([np.arange(spec.npix) for od in range(spec.nbo)])
#     elif 'wav' in scale:
#         x2d = spec.wavereference
#     flx2d = spec.data
#     bkg2d = spec.background
#     err2d = spec.error
#     orders = spec.prepare_orders(order)
#     linelist = spec['linelist']
    
#     updated_linelist = solve(lsf2d=lsf2d,linelist=linelist,
#                              x2d=x2d,flx2d=flx2d,bkg2d=bkg2d,err2d=err2d,
#                              order=orders,fittype=fittype,scale='pix',
#                              interpolate=False)
#     return updated_linelist

# def update_linelist_1d(spec,order,scale,iteration,fittype,save=False):
    
#     from harps.lsf.container import LSF
#     lsf_filepath = hio.get_fits_path('lsf', spec.filepath)
#     lsf2d = lio.from_fits(lsf_filepath, scale, iteration)
#     LSF2d = LSF(lsf2d)
#     # LSF1d = LSF2d[order]
#     # lsf1d = LSF1d.values
    
#     wav1d = spec.wavereference[order]
#     flx1d = spec.data[order]
#     bkg1d = spec.background[order]
#     err1d = spec.error[order]
#     # llist = spec['linelist']
#     # orders = spec.prepare_orders(order)
#     # fittype = 'lsf'
    
#     linelist = spec['linelist']
#     cut = np.where(linelist['order']==order)[0]
#     linelist1d = linelist[cut]
    
#     if scale=='pixel':
#         x1d = np.arange(spec.npix)
#     elif scale=='wave':
#         x1d = wav1d
    
#     llist1d = aux.solve_1d(LSF2d,linelist,x1d,flx1d,
#                             bkg1d,err1d,fittype,scale,interpolate)
    
#     # if save:
#     #     with FITS(spec._outpath,'rw') as hdu:
#     #         hdu.write(data=data,header=header,extname=ext,extver=ver)
#     #     status = " calculated."
        
#     return llist1d
# def from_spectrum_1d(spec,order,scale='pixel',iter_solve=2,iter_center=5,
#                   numseg=16,model_scatter=True,save=True,overwrite=False,
#                   start_iteration=0,fittype=None):
    

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


    