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
import harps.lsf.plot as lsfplot
import harps.lsf.gp as lsfgp
import harps.fit as hfit
import hashlib
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

def models_2d(vel3d, flx3d, err3d, orders, method, scale,
                  numseg=16,numpix=7,subpix=4,numiter=5,filter=None,**kwargs):
    assert scale in ['pixel','velocity'], "Scale not understood"
    lst = []
    for i,od in enumerate(orders):
        print("order = {}".format(od))
        plot=False
        lsf1d=(models_1d(vel3d[od],flx3d[od],err3d[od],
                               method,numseg,numpix,
                               subpix,numiter,filter=filter,plot=plot,**kwargs))
        lsf1d['order'] = od
        lst.append(lsf1d)
        filepath = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/'+\
                   f'ESPRESSO_{od}_{scale}.fits'
        with FITS(filepath,mode='rw') as hdu:
            hdu.write(lsf1d,extname='{}'.format(od))
        # hdu.close()
        print("File saved to {}".format(filepath))
        if len(orders)>1:
            hf.update_progress((i+1)/len(orders),'Fit LSF')
    lsf = np.hstack(lst)
    
    return lsf

def models_1d(x2d,flx2d,err2d,method,numseg=16,numpix=8,subpix=4,
                    numiter=5,minpix=0,minpts=10,filter=None,plot=True,
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
    maxpix  = np.shape(x2d)[0]
    seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
    totpix  = 2*numpix*subpix+1
    
    pixcens = np.linspace(-numpix,numpix,totpix)
    lsf1d   = aux.get_empty_lsf('spline',numseg,totpix,pixcens)
    count = 0
    for i in range(len(lsf1d)):
        pixl = seglims[i]
        pixr = seglims[i+1]
        x1s  = np.ravel(x2d[pixl:pixr])
        flx1s = np.ravel(flx2d[pixl:pixr])
        err1s = np.ravel(err2d[pixl:pixr])
        checksum = hashlib.md5(np.array([x1s,flx1s,err1s,
                                         np.full_like(x1s,i)])).hexdigest()
        print("segment = {0}/{1}".format(i+1,len(lsf1d)))
        out  = model_1s(x1s,flx1s,err1s,method,numiter,numpix,subpix,
                               minpts,filter,plot=plot,checksum=checksum,
                               **kwargs)
        if out is not None:
            pass
        else:
            continue
        lsf1s_out = out
        
        lsf1d[i]=lsf1s_out
        lsf1d[i]['pixl'] = pixl
        lsf1d[i]['pixr'] = pixr
        lsf1d[i]['segm'] = i
    return lsf1d


#@profile
def model_1s(pix1s,flx1s,err1s,method,
                    numiter=5,numpix=10,subpix=4,minpts=10,filter=None,
                    plot=False,save_plot=False,checksum=None,
                    model_scatter=False,**kwargs):
    '''
    Constructs the LSF model for a single segment
    '''
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    ## other keywords
    verbose          = kwargs.pop('verbose',False)
    print(f'filter={filter}')
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
    for j in range(numiter):
        
        
        # shift the values along x-axis for improved centering
        pix1s = pix1s+shift  
        
        if method == 'spline':
            function = construct_spline
            shift_method = kwargs.pop('shift_method',2)
            args.update({'numpix':numpix,'subpix':subpix,'minpts':minpts,
                         'shift_method':shift_method})
        elif method == 'analytic':
            function = construct_analytic
        elif method=='tinygp':
            function = construct_tinygp
            args.update({'numpix':numpix,
                         'subpix':subpix,
                         'checksum':checksum,
                         'plot':plot,
                         'filter':filter,
                         'minpts':minpts,
                         'model_scatter':model_scatter})
        dictionary=function(pix1s,flx1s,err1s,**args)
        lsf1s  = dictionary['lsf1s']
        shift  = dictionary['lsfcen']
        cenerr = dictionary['lsfcen_err']
        chisq  = dictionary['chisq']
        rsd    = dictionary['rsd']
        
        
        delta = np.abs(shift - oldshift)
        relchange = np.abs(delta/oldshift)
        totshift += shift
        dictionary.update({'totshift':totshift})
        print("iter {0:2d}   shift={1:+5.2e}  ".format(j,shift) + \
              "delta={0:5.2e}   sum_shift={1:5.2e}   ".format(delta,totshift) +\
              "relchange={0:5.2e}  chisq={1:6.2f}".format(relchange,chisq))
        
        oldshift = shift
        if (delta<1e-4 or np.abs(oldshift)<1e-4 or j==numiter-1) and j>0:
            print('stopping condition satisfied')
            if plot:
                plotfunction = lsfplot.plot_solution
                plotfunction(pix1s, flx1s, err1s, method, dictionary,
                                      checksum, save=save_plot,**kwargs)
            break
        else:
            pass
        
        # if plot and j==numiter-1:
           
    print('total shift {0:12.6f} +/- {1:12.6f} [m/s]'.format(totshift*1e3,
                                                             cenerr*1e3))   
    
    # save the total number of points used
    lsf1s['numlines'] = len(pix1s)
    return lsf1s


def construct_tinygp(x,y,y_err,numpix,subpix,plot=False,checksum=None,
                     filter=10,N_test=400,model_scatter=False,**kwargs):
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
    
    LSF_solution = lsfgp.train_LSF_tinygp(X,Y,Y_err,scatter=None)
    
    if model_scatter:
        scatter = lsfgp.train_scatter_tinygp(X,Y,Y_err,LSF_solution)
        LSF_solution = lsfgp.train_LSF_tinygp(X,Y,Y_err,scatter)
    else:
        scatter=None
    gp = lsfgp.build_LSF_GP(LSF_solution,X,Y,Y_err,scatter)
    
    # --------  Save output -------- 
    npars = len(LSF_solution) 
    if scatter is not None:
        npars = npars + len(scatter[0])
    
    # Initialize an LSF for this segment
    lsf1s    = aux._prepare_lsf1s(numpix,subpix,npars=npars)
    
    # Save parameters
    # The ordering of the parameters is:
    lsf1s['pars'][0] = LSF_solution['mf_amp']
    lsf1s['pars'][1] = LSF_solution['mf_loc']
    lsf1s['pars'][2] = LSF_solution['mf_log_sig']
    lsf1s['pars'][3] = LSF_solution['mf_const']
    lsf1s['pars'][4] = LSF_solution['gp_log_amp']
    lsf1s['pars'][5] = LSF_solution['gp_log_scale']
    lsf1s['pars'][6] = LSF_solution['log_var_add']
    
    if scatter is not None:
        lsf1s['pars'][7] = scatter[0]['sct_log_amp']
        lsf1s['pars'][8] = scatter[0]['sct_log_scale']
        lsf1s['pars'][9] = scatter[0]['sct_log_const']
    
    # Save LSF x-coordinates
    X_grid     = jnp.linspace(-numpix, numpix, 2*numpix*subpix+1)
    lsf1s['x'] = X_grid
    # Evaluate LSF model on X_grid and save it and the error
    
    _, cond    = gp.condition(Y, X_grid)
    lsf_y      = cond.loc
    lsf_yerr   = np.sqrt(cond.variance)
    lsf1s['y']    = lsf_y
    lsf1s['yerr'] = lsf_yerr
    # Now condition on the same grid as data to calculate residuals
    _, cond    = gp.condition(Y, X)
    Y_pred     = cond.loc
    # Y_pred_err = np.sqrt(cond.variance)
    # Y_tot_err  = jnp.sqrt(Y_err**2+Y_pred_err**2)
    rsd        = (Y - Y_pred)/Y_err
    # rsd        = (Y - Y_pred)/Y_tot_err
    dof        = len(rsd) - len(LSF_solution)
    chisq      = np.sum(rsd**2) / dof
    # Find x for which df/dx=0 (turning point)
    # grad   = jax.grad(gp.mean_function)
    # lsfcen = newton(grad,0.)
    # lsfcen_err = 0.0
    lsfcen, lsfcen_err = lsfgp.estimate_centre(X,Y,Y_err,
                                          LSF_solution,scatter=scatter,
                                          N=10)
    out_dict = dict(lsf1s=lsf1s, lsfcen=lsfcen, lsfcen_err=lsfcen_err,
                    chisq=chisq, rsd=rsd, 
                    solution_LSF=LSF_solution)
    out_dict.update(dict(model_scatter=model_scatter))
    if model_scatter==True:
        out_dict.update(dict(solution_scatter=scatter))
    
    return out_dict
def construct_spline(pix1s,flx1s,err1s,numpix,subpix,minpts,shift_method):
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    lsf1s = aux.get_empty_lsf(1,totpix,pixcens)[0]
        
    # get current model of the LSF
    splr = interpolate.splrep(lsf1s['x'],lsf1s['y']) 
    xx = pix1s                   
    prediction = interpolate.splev(pix1s,splr)
    prediction_err = np.zeros_like(prediction)
    # calculate residuals to the model
    rsd  = (flx1s-prediction)
    # return pix1s,rsd,pixlims,minpts
    # calculate mean of residuals for each pixel comprising the LSF
    means,stds, counts  = aux.bin_means(pix1s,rsd,pixlims,minpts)
    lsf1s['y'] = lsf1s['y']+means
    
    if shift_method==1:
        shift = aux.shift_anderson(lsf1s['x'],lsf1s['y'])
    elif shift_method==2:
        shift = aux.shift_zeroder(lsf1s['x'],lsf1s['y'])
    dof = len(rsd) - totpix
    chisq = np.sum((rsd/err1s)**2) / dof
    return dict(lsf1s=lsf1s, shift=shift, chisq=chisq, rsd=rsd, 
                mean=prediction, mean_err=prediction_err)

def construct_analytic(pix1s,flx1s,err1s):
    ngauss = 10
    lsf1s = aux.get_empty_lsf(1,ngauss)[0]
    
    # test parameters
    p0=(1,5)+ngauss*(0.1,)
    # set range in which to fit (set by the data)
    xmax = np.around(np.max(pix1s)*2)/2
    xmin = np.around(np.min(pix1s)*2)/2
    popt,pcov=hfit.curve(hf.gaussP,pix1s,flx1s,p0=p0,method='lm',)
#                                fargs={'xrange':(xmin,xmax)})
    prediction_err = hf.error_from_covar(hf.gaussP,popt,pcov,pix1s)
    if np.any(~np.isfinite(popt)):
        plt.figure()
        plt.scatter(pix1s,flx1s,s=3)
        plt.show()
    xx = np.linspace(-9,9,700)
    prediction,centers,sigma = hf.gaussP(xx,*popt,#xrange=(xmin,xmax),
                                 return_center=True,return_sigma=True)
    shift = -hf.derivative_zero(xx,prediction,-1,1)
    rsd = flx1s - hf.gaussP(pix1s,*popt)
    lsf1s['pars'] = popt
    lsf1s['errs'] = np.square(np.diag(pcov))
    dof           = len(rsd) - len(popt)
    chisq = np.sum((rsd/err1s)**2) / dof
    
    return dict(lsf1s=lsf1s, shift=shift, chisq=chisq, rsd=rsd, 
                mean=prediction, mean_err=prediction_err)