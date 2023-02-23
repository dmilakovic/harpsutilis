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
import harps.lsf.write as write
import harps.fit as hfit
import harps.io as hio
import hashlib
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import gc



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
              minpix=None,maxpix=None,filter=None,plot=True,metadata=None,*args,
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
    minpix = minpix if minpix is not None else 0
    maxpix = maxpix if maxpix is not None else np.shape(x2d)[0]
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
            metadata.update({'checksum':checksum,'segment':i+1})
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
    return lsf1d


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
    totshift  = 0
    args = {}
    metadata.update({'model_scatter':model_scatter})
    for j in range(numiter):
        # shift the values along x-axis for improved centering
        pix1s = pix1s+shift  
        args.update({#'numpix':numpix,
                     #'subpix':subpix,
                     'metadata':metadata,
                     'plot':plot,
                     'filter':filter,
                     #'minpts':minpts,
                     'model_scatter':model_scatter})
        dictionary=construct_tinygp(pix1s,flx1s,err1s, plot=plot,
                                    # metadata=metadata,
                                    filter=filter,model_scatter=model_scatter)
        lsf1s  = dictionary['lsf1s']
        shift  = -dictionary['lsfcen']
        cenerr = dictionary['lsfcen_err']
        chisq  = dictionary['chisq']
        rsd    = dictionary['rsd']
        
        
        delta = np.abs(shift - oldshift)
        relchange = np.abs(delta/oldshift)-1
        totshift += shift
        dictionary.update({'totshift':totshift})
        dictionary.update({'scale':metadata['scale'][:3]})
        
        print(f"iter {j:2d}   shift={shift:+5.2e}  " + \
              f"delta={delta:5.2e}   sum_shift={totshift:5.2e}   " +\
              f"relchange={relchange:5.2%}  chisq={chisq:6.2f}")
        
        oldshift = shift
        if (delta<1e-4 or np.abs(oldshift)<1e-4 or j==numiter-1) and j>0:
            print('stopping condition satisfied')
            if plot:
                plotfunction = lsfplot.plot_solution
                plotfunction(pix1s, flx1s, err1s, dictionary,
                                      metadata, save=save_plot,**kwargs)
            break
        else:
            pass
        
        # if plot and j==numiter-1:
           
    print('total shift {0:12.6f} +/- {1:12.6f}'.format(totshift*1e3,
                                                             cenerr*1e3))   
    
    # save the total number of points used
    lsf1s['numlines'] = len(pix1s)
    return lsf1s


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
    
    LSF_solution = lsfgp.train_LSF_tinygp(X,Y,Y_err,scatter=None)
    
    if model_scatter:
        scatter = lsfgp.train_scatter_tinygp(X,Y,Y_err,LSF_solution)
        LSF_solution = lsfgp.train_LSF_tinygp(X,Y,Y_err,scatter)
    else:
        scatter=None
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
    Y_mod_err  = np.sqrt(cond.variance)
    Y_tot_err  = jnp.sqrt(np.sum(np.power([Y_data_err,Y_mod_err],2.),axis=0))
    rsd        = lsfgp.get_residuals(X, Y, Y_tot_err, LSF_solution)
    dof        = len(rsd) - npars
    chisq      = np.sum(rsd**2)
    chisqdof   = chisq / dof
    lsfcen, lsfcen_err = lsfgp.estimate_centre(X,Y,Y_err,
                                          LSF_solution,scatter=scatter,
                                          N=N_test)
    out_dict = dict(lsf1s=lsf1s, lsfcen=lsfcen, lsfcen_err=lsfcen_err,
                    chisq=chisqdof, rsd=rsd, 
                    solution_LSF=LSF_solution)
    out_dict.update(dict(model_scatter=model_scatter))
    if model_scatter==True:
        out_dict.update(dict(solution_scatter=scatter))
    
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

def from_spectrum(spec,order=None,scale='pixel',iter_solve=2,iter_center=5,
                  numseg=16,model_scatter=True,save=True,overwrite=False):
    from harps.lsf.container import LSF
    
    assert scale in ['pixel','velocity']
    wav2d = spec.wavereference
    flx2d = spec.data
    bkg2d = spec.background
    err2d = spec.error
    llist = spec['linelist']
    orders = spec.prepare_orders(order)
    fittype     = 'lsf'
    n_steps = len(orders)*iter_solve
    for j,od in enumerate(orders):
        for i in range(iter_solve):
            step = i+j+1
            if i==0:
                linelist = spec['linelist']
                fittype = 'gauss'
            else:
                linelist = llist
            pix3d,vel3d,flx3d,err3d,_ = aux.stack(fittype,linelist,flx2d,
                                                wav2d,err2d,bkg2d,od)    
            plot=True; save_plot=True
            if scale=='pixel':
                x3d = pix3d
            elif scale=='velocity':
                x3d = vel3d
            metadata=dict(
                order=od,
                scale=scale,
                model_scatter=model_scatter,
                iteration=i,
                )
            lsf1d=models_1d(x3d[od],flx3d[od],err3d[od],
                                      numseg=numseg,
                                      numiter=iter_center,
                                      minpts=15,
                                      model_scatter=model_scatter,
                                      minpix=None,maxpix=None,
                                      filter=None,plot=plot,
                                      metadata=metadata,
                                      save_plot=save_plot)
            lsf1d['order'] = od
            if save:
                filename = hio.get_filename(spec.filepath,'lsf')
                extname = f"{scale}"
                write.lsf_to_file(lsf1d, filename, extname,overwrite=overwrite)
                
            if i < iter_solve-1:
                llist = aux.solve(LSF(lsf1d),linelist,flx2d,err2d,
                                    bkg2d,fittype,)
                # linelist = linelists_i
            hf.update_progress(step/n_steps,'Fit LSF')
            
                
            gc.collect()
    return None
