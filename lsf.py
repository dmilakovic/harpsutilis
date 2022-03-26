#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
from harps import functions as hf
from harps import settings as hs
from harps import io as io
from harps import containers as container
from harps import plotter as hplot
from harps import fit as hfit
#from .gaussprocess_class import HeteroskedasticGaussian
from harps.core import os, np, plt, FITS


import jax
from   jax import jit
import jax.numpy as jnp
import jaxopt
from tinygp import kernels, GaussianProcess, noise

import pyro
import numpyro 
import torch
import numpyro.distributions as dist
from functools import partial 

#import line_profiler

#import errno

from scipy import interpolate
from scipy.optimize import leastsq, brentq, curve_fit
import scipy.stats as stats

from matplotlib import ticker
import hashlib

#from sklearn.gaussian_process import GaussianProcessRegressor as GPR
#from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern, ExpSineSquared

#import gpflow
#from gpflow.utilities import print_summary
#from gpflow.ci_utils import ci_niter
#from gpflow.monitor import (
#    ImageToTensorBoard,
#    ModelToTensorBoard,
#    Monitor,
#    MonitorTaskGroup,
#    ScalarToTensorBoard,
#)
#import tensorflow as tf
#import tensorflow_probability as tfp
#import itertools
# =============================================================================
#    
#                        I N P U T  / O U T P U T
#    
# =============================================================================
    
def read_lsf(fibre,specifier,method,version=-1):
    # specifier must be either a string (['round','octog']) or a np.datetime64
    # instance. 
    if isinstance(specifier,str):
        shape = specifier[0:5]
    elif isinstance(specifier,np.datetime64):
        if specifier<=np.datetime64('2015-05-01'):
            shape = 'round'
        else:
            shape = 'octog'
    else:
        print("Fibre shape unknown")
    assert shape in ['round','octog']
    assert method in ['spline','analytic','gp']
    filename ='LSF_{f}_{s}_{m}.fits'.format(f=fibre,s=shape,m=method)
    hdu = FITS(os.path.join(hs.dirnames['lsf'],filename))
    lsf = hdu[-1].read()
    return LSF(lsf)

def from_file(filepath,nhdu=-1):
    hdu = FITS(filepath)
    lsf = hdu[nhdu].read()
    return LSF(lsf)
# =============================================================================
#    
#                        L S F    M O D E L L I N G
#    
# =============================================================================

class LSFModeller(object):
    def __init__(self,outfile,sOrder,eOrder,iter_solve=2,iter_center=5,
                 numseg=16,numpix=7,subpix=4,filter=10,method='gp'):
        self._outfile = outfile
        self._cache = {}
        self._iter_solve  = iter_solve
        self._iter_center = iter_center
        self._numseg  = numseg
        self._numpix  = numpix
        self._subpix  = subpix
        self._sOrder  = sOrder
        self._eOrder  = eOrder
        self._orders  = np.arange(sOrder,eOrder)
        self._method  = method
        self._filter  = filter
        self.iters_done = 0
    def __getitem__(self,extension):
        try:
            data = self._cache[extension]
        except:
            self._read_data_from_file()
            #self._cache.update({extension:data})
            data = self._cache[extension]
        return data
    def __setitem__(self,extension,data):
        self._cache.update({extension:data})
    def _read_data_from_file(self,start=None,stop=None,step=None,**kwargs):
        extensions = ['linelist','flux','background','error','wavereference']
        data, numfiles = io.mread_outfile(self._outfile,extensions,701,
                                start=start,stop=stop,step=step)
        self._cache.update(data)
        self.numfiles = numfiles
        return
    
    def __call__(self,verbose=False,filepath=None):
        """ Returns the LSF in an numpy array  """
        wavelengths = self['wavereference']
        fluxes      = self['flux']
        backgrounds = self['background']
        errors      = self['error']
        fittype     = 'lsf'
        for i in range(self._iter_solve):
            linelists = self['linelist']
            if i == 0:
                fittype = 'gauss'
            vel3d, wav3d, flx3d, err3d, orders = stack(fittype,linelists,fluxes,
                                                wavelengths,errors,backgrounds,
                                         self._orders)
            # lsf_i    = construct_lsf(pix3d,flx3d,err3d,self._orders,
            #                          numseg=self._segnum,
            #                          numpix=self._numpix,
            #                          subpix=self._subpix,
            #                          numiter=self._iter_center,
            #                          method=self._method,
            #                          filter=self._filter,
            #                          verbose=verbose)
            lst = []
            for j,od in enumerate(self._orders):
                print("order = {}".format(od))
                plot=True
                lsf1d=(construct_lsf1d(vel3d[od],flx3d[od],err3d[od],
                                       method=self._method,
                                       numseg=self._numseg,
                                       numpix=self._numpix,
                                       subpix=self._subpix,
                                       numiter=self._iter_center,
                                       plot=plot,
                                       verbose=verbose,
                                       filter=self._filter))
                lsf1d['order'] = od
                lst.append(lsf1d)
                
                if len(orders)>1:
                    hf.update_progress((j+1)/len(orders),'Fit LSF')
                if filepath is not None:
                    self.save(lsf1d,filepath,'{0:02d}'.format(j+1),False)
            lsf_i = LSF(np.hstack(lst))
            self._lsf_i = lsf_i
            setattr(self,'lsf_{}'.format(i),lsf_i)
            if i < self._iter_solve-1:
                linelists_i = solve(lsf_i,linelists,fluxes,errors,
                                    backgrounds,fittype,self._method)
                self['linelist'] = linelists_i
            self.iters_done += 1
        lsf_final = lsf_i
        self._lsf_final = lsf_final
        
        return lsf_final
    def stack(self,fittype='lsf'):
        fluxes      = self['flux']
        backgrounds = self['background']
        linelists   = self['linelist']
        errors      = self['error']
        wavelengths = self['wavereference']
        
        return stack(fittype,linelists,fluxes,wavelengths,errors,
                     backgrounds,self._orders)
        
    def save(self,data,filepath,version=None,overwrite=False):
        with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
            hdu.write(data,extname='LSF',extver=version)
        hdu.close()
        print("File saved to {}".format(filepath))
        return




def stack(fittype,linelists,fluxes,wavelengths,errors=None,
          backgrounds=None,orders=None):
    # numex = np.shape(linelists)[0]
    ftpix = '{}_pix'.format(fittype)
    ftwav = '{}_wav'.format(fittype)
    numex, numord, numpix = np.shape(fluxes)
    pix3d = np.zeros((numord,numpix,numex))
    flx3d = np.zeros((numord,numpix,numex))
    err3d = np.zeros((numord,numpix,numex))   
    vel3d = np.zeros((numord,numpix,numex))  
#    plt.figure()
    for exp,linelist in enumerate(linelists):
        hf.update_progress((exp+1)/len(linelists),"Stack")
        if orders is not None:
            orders = orders
        else:
            orders = np.unique(linelist['order'])
        for j,line in enumerate(linelist):
            segment  = line['segm']
            od       = line['order']
            if od not in orders:
                continue
            pixl     = line['pixl']
            pixr     = line['pixr']
            
            pix1l = np.arange(pixl,pixr) - line[ftpix][1]
            
            # normalise the flux by area under the central 16 pixels 
            # (8 pixels either side)
            central = np.where((pix1l>=-5) & (pix1l<=5))[0]
            pixpos = np.arange(pixl,pixr,1)
            
            lineflux = fluxes[exp,od,pixl:pixr]
            wav1l = wavelengths[exp,od,pixl:pixr]
            vel1l = (wav1l - line[ftwav][1])/line[ftwav][1]*299792.458
            if backgrounds is not None:
                linebkg  = backgrounds[exp,od,pixl:pixr]
                lineflux = lineflux - linebkg
                # lineerr  = np.sqrt(lineflux + linebkg)
            if errors is not None:
                lineerr = errors[exp,od,pixl:pixr]
                if backgrounds is not None:
                    lineerr = np.sqrt(lineerr + \
                                     backgrounds[exp,od,pixl:pixr])
            # flux is Poissonian distributed, P(nu),  mean = variance = nu
            # Sum of fluxes is also Poissonian, P(sum(nu))
            #           mean     = sum(nu)
            #           variance = sum(nu)
            # C_flux = np.sum(lineflux)
            # C_flux_err = np.sqrt(C_flux)
            C_flux = line[ftpix][0]
            C_flux_err = line[f'{ftpix}_err'][0]
            # print(err_norm)
            # flx1l = lineflux/normalisation
            pix3d[od,pixpos,exp] = pix1l
            vel3d[od,pixpos,exp] = vel1l
            flx3d[od,pixpos,exp] = lineflux/C_flux
            err3d[od,pixpos,exp] = 1./C_flux*np.sqrt(lineerr**2 + \
                                            (lineflux*C_flux_err/C_flux)**2)
    return pix3d,vel3d,flx3d,err3d,orders


def construct_lsf(vel3d, flx3d, err3d, orders, method,
                  numseg=16,numpix=7,subpix=4,numiter=5,filter=None,**kwargs):
    lst = []
    for i,od in enumerate(orders):
        print("order = {}".format(od))
        plot=True
        lsf1d=(construct_lsf1d(vel3d[od],flx3d[od],err3d[od],
                               method,numseg,numpix,
                               subpix,numiter,plot=plot,**kwargs))
        lsf1d['order'] = od
        lst.append(lsf1d)
        filepath = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/'+\
                   'ESPRESSO_{}_{}.fits'.format(od,'vel')
        with FITS(filepath,mode='rw') as hdu:
            hdu.write(lsf1d,extname='{}'.format(od))
        # hdu.close()
        print("File saved to {}".format(filepath))
        if len(orders)>1:
            hf.update_progress((i+1)/len(orders),'Fit LSF')
    lsf = np.hstack(lst)
    
    return LSF(lsf)
def construct_lsf1d(x2d,flx2d,err2d,method,numseg=16,numpix=8,subpix=4,
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
    lsf1d   = get_empty_lsf('spline',numseg,totpix,pixcens)
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
        out  = construct_lsf1s(x1s,flx1s,err1s,method,numiter,numpix,subpix,
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
def construct_lsf1s(pix1s,flx1s,err1s,method,
                    numiter=5,numpix=10,subpix=4,minpts=10,filter=None,
                    plot=False,save_plot=False,checksum=None,
                    model_scatter=True,**kwargs):
    '''
    Constructs the LSF model for a single segment
    '''
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    ## other keywords
    verbose          = kwargs.pop('verbose',False)
    
    pix1s, flx1s, err1s = clean_input(pix1s,flx1s,err1s,sort=True,
                                      verbose=verbose,filter=None)
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
        lsf1s = dictionary['lsf1s']
        shift = dictionary['shift']
        chisq = dictionary['chisq']
        rsd   = dictionary['rsd']
        
        
        delta = np.abs(shift - oldshift)
        relchange = np.abs(delta/oldshift)
        totshift += shift
        dictionary.update({'totshift':totshift})
        print("iter {0:2d}   shift={1:+5.2e}  ".format(j,shift) + \
              "delta={0:5.2e}   sum_shift={1:5.2e}   ".format(delta,totshift) +\
              "relchange={0:5.2e}  chisq={1:6.2f}".format(relchange,chisq))
        
        oldshift = shift
        if delta<1e-4 or np.abs(oldshift)<1e-4 or j==numiter-1:
            print('stopping condition satisfied')
            if plot:
                # plot_function = plot_lsf_model
                plotfunction = plot_solution
                plotfunction(pix1s, flx1s, err1s, method, dictionary,
                                      checksum, save=save_plot,**kwargs)
            break
        else:
            pass
        
        # if plot and j==numiter-1:
           
    print('total shift {0:12.6f} [m/s]'.format(totshift*1e3))   
    
    # save the total number of points used
    lsf1s['numlines'] = len(pix1s)
    return lsf1s

def _prepare_lsf1s(numpix,subpix):
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    lsf1s = get_empty_lsf('spline',1,totpix,pixcens)[0]
    return lsf1s

def _calculate_shift(y,x):
    return -hf.derivative_zero(y,x,-1,1)

# @jax.jit
# def loss_(theta,X,Y,Y_err):
#     gp = build_gp(theta,X,Y_err)
#     return -gp.log_probability(Y)

@jax.jit
def loss_LSF(theta,X,Y,Y_err,scatter=None):
    gp = build_LSF_GP(theta,X,Y_err,scatter)
    return -gp.log_probability(Y)
@jax.jit
def loss_scatter(theta,X,Y):
    gp = build_scatter_GP(theta,X)
    return -gp.log_probability(Y)


def train_LSF_tinygp(X,Y,Y_err,scatter=None):
    
    
    popt,pcov = curve_fit(hf.gauss4p,X,Y,sigma=Y_err,
                          absolute_sigma=False,p0=(np.max(Y),0,np.std(X),0))
    perr = np.sqrt(np.diag(pcov))

    theta = dict(
        log_error = 1.0,
        log_gp_amp=np.array(1.),
        log_gp_scale=np.array(1.),
        log_mf_amp   = np.log(np.abs(popt[0])),
        log_mf_width = np.log(np.abs(popt[2])),
        mf_const     = popt[3],
        mf_loc       = popt[1],
    )
    kappa = 5
    lower_bounds = dict(
        log_error = -10,
        log_gp_amp = -2.,
        log_gp_scale = np.log(0.2), # corresponds to 200 m/s
        log_mf_amp = np.log(np.abs(popt[0])-kappa*perr[0]),
        log_mf_width=np.log(np.abs(popt[2])-kappa*perr[2]),
        mf_const = popt[3]-kappa*perr[3],
        mf_loc = popt[1]-1*perr[1],
    )
    upper_bounds = dict(
        log_error = 0.,
        log_gp_amp = 2.,
        log_gp_scale = 3.,
        log_mf_amp = np.log(np.abs(popt[0])+kappa*perr[0]),
        log_mf_width=np.log(np.abs(popt[2])+kappa*perr[2]),
        mf_const = popt[3]+kappa*perr[3],
        mf_loc = popt[1]+1*perr[1],
    )
    bounds = (lower_bounds, upper_bounds)
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=partial(loss_LSF,
                                                     X=X,
                                                     Y=Y,
                                                     Y_err=Y_err,
                                                     scatter=scatter),
                                         method="l-bfgs-b")
    solution = lbfgsb.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
   
    print(f"Final negative log likelihood: {solution.state.fun_val}")
    return solution

def estimate_variance(X,Y,Y_err,theta_lsf,scale=7.0,step=0.4,minpts=10):
    gp = build_LSF_GP(theta_lsf,X,Y_err)
    _, cond = gp.condition(Y,X)
    mean_lsf = cond.loc
    rsdsq = np.exp(jnp.array(Y - mean_lsf))**2
    # Bin the residuals along the X-axis into (N+1) bins and calculate the
    # standard deviation in each. 
    # N = 40
    # xlims = np.linspace(X.min(),X.max(),N+1)
    # xdist = np.diff(xlims)[0] # size of the bin in km/s
    xlims = np.arange(-scale,scale+step,step)
    # bin_means takes right edges of pixels
    means, stds, counts = bin_means(X,rsdsq,xlims,minpts,kind='spline')
    
    # Remove empty bins
    cut = np.where(stds!=0)[0]
    # print(cut, stds[cut], counts)
    # The new observed dataset is {x_cens,logvar} where x_cens are the centres
    # of the bins and logvar is the log(std**2)
    # We remove the first element because there are N+1 limits to the N bins
    logvar_x = jnp.array((xlims - step/2)[cut])
    logvar_y = jnp.log(stds[cut])
    
    return logvar_x, logvar_y

def train_scatter_tinygp(X,Y,Y_err,theta_lsf,scale=7.0,step=0.4):
    '''
    Based on Kersting et al. 2007 :
        Most Likely Heteroscedastic Gaussian Process Regression

    '''
    # Calculate residuals to best fit LSF profile for observed {X,Y,Y_err}
    
    logvar_x, logvar_y = estimate_variance(X,Y,Y_err,theta_lsf,scale,step)
    
    theta = dict(
        log_sct_amp= 2.0,
        log_sct_scale = 1.0,
        )
    lower_bounds = dict(
        log_sct_amp= -5.0,
        log_sct_scale = np.log(step),
        )
    upper_bounds = dict(
        log_sct_amp= 10.0,
        log_sct_scale = 3.0,
        )
    bounds = (lower_bounds, upper_bounds)

    solver = jaxopt.ScipyBoundedMinimize(fun=partial(loss_scatter,
                                                      X=logvar_x,
                                                     Y=logvar_y
                                                     ),
                                         method="l-bfgs-b")
    solution = solver.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
    
    return solution, logvar_x, logvar_y


def gaussian_mean_function(theta, X):
    
    gauss = jnp.exp(
        -0.5 * jnp.square((X - theta["mf_loc"]) / jnp.exp(theta["log_mf_width"]))
    )
    
    beta = jnp.array([1, gauss])
    # beta = jnp.array([1,1])
    return jnp.array([theta["mf_const"],
                      jnp.exp(theta["log_mf_amp"])/jnp.sqrt(2*jnp.pi)]) @ beta

def build_scatter_GP(theta,X):
    # GP for the noise
    amp_sct    = jnp.exp(theta["log_sct_amp"])
    scale_sct  = jnp.exp(theta["log_sct_scale"])
    kernel_sct = amp_sct**2 * kernels.ExpSquared(scale_sct) # scatter kernel
    return GaussianProcess(
        kernel_sct,
        X,
        diag = 1e-8,
        mean = 0.0
    )

def build_LSF_GP(theta,X,Y_err,scatter=None):
    amp   = jnp.exp(theta["log_gp_amp"])
    scale = jnp.exp(theta["log_gp_scale"])
    kernel = amp**2 * kernels.ExpSquared(scale) # LSF kernel
    # Various variances (obs=observer, add=constant, tot=total)
    obs_var = jnp.power(Y_err,2)
    add_var = jnp.broadcast_to(jnp.exp(theta['log_error']),Y_err.shape)
    tot_var = add_var + obs_var
    # If scatter variance parameters are given, include them
    if scatter is not None:
        theta_scatter, logvar_x, logvar_y = scatter
        gp_scatter = build_scatter_GP(theta_scatter,logvar_x)
        _, gp_scatt_cond = gp_scatter.condition(logvar_y,X)
        inf_var = jnp.exp(gp_scatt_cond.loc)
        tot_var += inf_var
    return GaussianProcess(
        kernel,
        X,
        noise = noise.Diagonal(tot_var),
        mean=partial(gaussian_mean_function, theta),
    )

def construct_tinygp(x,y,y_err,numpix,subpix,plot=False,checksum=None,
                     filter=10,N_test=400,model_scatter=True,**kwargs):
    X        = jnp.array(x)
    Y        = jnp.array(y*100)
    Y_err    = jnp.array(y_err*100)
    
    
    LSF_solution = train_LSF_tinygp(X,Y,Y_err)
    if model_scatter==True:
        # print("INITIAL\n",LSF_solution.params)
        for i in range(1):
            scatter = train_scatter_tinygp(X, Y, Y_err, LSF_solution.params)
            scatter_solution, logvar_x, logvar_y  = scatter
            LSF_solution = train_LSF_tinygp(X,Y,Y_err, 
                                            (scatter_solution.params,
                                             logvar_x,
                                             logvar_y
                                             )
                                            )
        gp = build_LSF_GP(LSF_solution.params,X,Y_err,
                          (scatter_solution.params,
                           logvar_x,
                           logvar_y
                           )
                          )
        # print("FINAL:\n",LSF_solution.params)
    else:
        gp = build_LSF_GP(LSF_solution.params,X,Y_err)
    
    # Prepare to save output
    lsf1s    = _prepare_lsf1s(numpix,subpix)
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
    Y_pred_err = np.sqrt(cond.variance)
    Y_tot_err  = jnp.sqrt(Y_err**2+Y_pred_err**2)
    rsd        = (Y - Y_pred)/Y_tot_err
    dof        = len(rsd) - len(LSF_solution.params)
    chisq      = np.sum(rsd**2) / dof
    # Dense grid around centre
    X_central = jnp.arange(-2,2,0.01)
    _, cond   = gp.condition(Y, X_central)
    Y_central = cond.loc
    lsfcen    = _calculate_shift(Y_central,X_central)
    
    out_dict = dict(lsf1s=lsf1s, shift=lsfcen, chisq=chisq, rsd=rsd, 
                solution_LSF=LSF_solution)
    if model_scatter==True:
        out_dict.update(dict(solution_scatter=scatter))
    
    return out_dict



def estimate_centre(rng_key,X,Y,Y_err,LSF_solution,scatter=None,N=200):
    
    def value_(x):
        _, cond = gp.condition(Y,jnp.array([x]))
        sample = cond.sample(rng_key,shape=())
        return sample[0]
        # loc = gp.loc
        # scale = jnp.sqrt(jnp.diag(gp.variance))
        # numpyro.sample()
    # @partial(gp=cond,Y=Y)
    def derivative_(x):#,gp,Y,rng_key):
        # return jax.grad(partial(value_,gp=gp,Y=Y,rng_key=rng_key))(x)
        return jax.grad(value_,argnums=(0,1))(x)
    @jit
    def solve_():
        bisect = jaxopt.Bisection(derivative_,-1.,1.)
        return bisect.run().params
    # def nympyro_model(gp,Y,rng_key):
        
    
    # def pyro_model(gp,x,rng_key=None):
    #     if rng_key is None:
    #         rng_key = jax.random.PRNGKey(55873)
    #     # print(gp.loc)
    #     # lsf_model = numpyro.sample("gp", gp.numpyro_dist())
    #     mean = torch.tensor(gp.loc._value)
    #     sigma = torch.tensor(np.sqrt(gp.variance._value))
    #     lsf_model = numpyro.sample('gp',dist.Normal(mean,sigma))
    #     # print(x,lsf_model)
    #     centre = _calculate_shift(lsf_model,x)
    #     centre = numpyro.sample("obs",dist.Normal(centre,0.001),obs=centre)
    #     # return centre
    # # def numpyro_model(x,gp):
    # #     lsf_model = numpyro.sample("gp", gp.numpyro_dist())
    # #     centre = _calculate_shift(x,lsf_model)
    # #     return centre
    # def _wrapper(samples,x):
    #     # sample = gp.sample(rng_key,(1,))
    #     return -_calculate_shift(samples,x)
    
    if scatter is not None:
        scatter_solution, logvar_x, logvar_y  = scatter
        gp = build_LSF_GP(LSF_solution.params,X,Y_err,
                          (scatter_solution.params,
                           logvar_x,
                           logvar_y
                           )
                          )
    else:
        gp = build_LSF_GP(LSF_solution.params,X,Y_err)
    
    X_grid  = jnp.linspace(-1,1,100)
    _, cond = gp.condition(Y,X_grid)
    centre  = -_calculate_shift(cond.loc,X_grid)
    
    
    # derivative = partial(derivative_,)
    centres = np.empty(N)
    # keys = 
    for i in range(N):
        print(i)
        rng_key = jax.random.PRNGKey(i)
        # bisect = jaxopt.Bisection(derivative_,-1.,1.)#,gp=gp,Y=Y,rng_key=rng_key)
        # centres[i] = bisect.run().params
        centres[i] = solve_()
    # samples = cond.sample(rng_key,(N,))
    
    # deriv   = hf.derivative(samples)
    # argmin  = jnp.argmin(jnp.abs(deriv)[:,5:-5],axis=1)
    # centres = X_grid[argmin]

    # centres = np.empty(N)
    # for i in range(N):
    #     print(i)
    #     centres[i] = -_calculate_shift(samples[i],X_grid)
    #     
    plt.hist(centres.T,range=(-0.05,0.05),bins=20)
    mean, sigma = hf.average(centres)
    return centre, mean, sigma, centres
    # # rng_key = jax.random.PRNGKey(55873)
    # nuts_kernel = numpyro.infer.NUTS(pyro_model)
    # mcmc = numpyro.infer.MCMC(nuts_kernel, 
    #                           num_warmup=200,
    #                           num_samples=500)
    # mcmc.run(rng_key, cond,X_grid)
    # samples = mcmc.get_samples()    
    # return samples
        
        
    
    # centres = jax.vmap(_wrapper)(samples.T)
    # fun = partial(hf.derivative_eval,x_array=X_grid,y_array=samples.T)
    
    
    
    
    
    # print(dir(samples))
    # # print(mcmc.summary(prob=0.5))
    # q = np.percentile(samples["gp"], [5, 25, 50, 75, 95], axis=0)
    # plt.plot(X_grid, q[2], color="C0", label="MCMC inferred model")
    # plt.fill_between(X, q[0], q[-1], alpha=0.3, lw=0, color="C0")
    # plt.fill_between(X, q[1], q[-2], alpha=0.3, lw=0, color="C0")
    # # plt.plot(X, true_log_rate, "--", color="C1", label="true rate")
    # plt.errorbar(X, Y, Y_err, ls='', marker='.', c='k', label="data")
    # plt.legend(loc=2)
    # plt.xlabel("x")
    # _ = plt.ylabel("counts")

    # return
    return centre, sigma

def construct_spline(pix1s,flx1s,err1s,numpix,subpix,minpts,shift_method):
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    lsf1s = get_empty_lsf('spline',1,totpix,pixcens)[0]
        
    # get current model of the LSF
    splr = interpolate.splrep(lsf1s['x'],lsf1s['y']) 
    xx = pix1s                   
    prediction = interpolate.splev(pix1s,splr)
    prediction_err = np.zeros_like(prediction)
    # calculate residuals to the model
    rsd  = (flx1s-prediction)
    # return pix1s,rsd,pixlims,minpts
    # calculate mean of residuals for each pixel comprising the LSF
    means,stds, counts  = bin_means(pix1s,rsd,pixlims,minpts)
    lsf1s['y'] = lsf1s['y']+means
    
    if shift_method==1:
        shift = shift_anderson(lsf1s['x'],lsf1s['y'])
    elif shift_method==2:
        shift = shift_zeroder(lsf1s['x'],lsf1s['y'])
    dof = len(rsd) - totpix
    chisq = np.sum((rsd/err1s)**2) / dof
    return dict(lsf1s=lsf1s, shift=shift, chisq=chisq, rsd=rsd, 
                mean=prediction, mean_err=prediction_err)

def construct_analytic(pix1s,flx1s,err1s):
    ngauss = 10
    lsf1s = get_empty_lsf('analytic',1,ngauss)[0]
    
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


    
def bin_means(x,y,xbins,minpts=10,kind='spline'):
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
        means[i] = np.nanmean(y1)
        stds[i]  = np.nanstd(y1)
    # go back and interpolate means for empty bins
    idy   = hf.find_missing(idx)
    # interpolate if no points in the bin, but only pixels -5 to 5
    if len(idy)>0:
        idy = np.atleast_1d(idy)
        means[idy] = interpolate_bins(means,xbins[idy],kind)
    
    return means,stds,hist
def interpolate_local_spline(lsf,order,center):
    assert np.isfinite(center)==True, "Center not finite, {}".format(center)
    values  = lsf[order].values
    assert len(values)>0, "No LSF model for order {}".format(order)
    numseg,totpix  = np.shape(values['x'])
    
    segcens = (values['pixl']+values['pixr'])/2
    segcens[0]  = values['pixl'][0]
    segcens[-1] = values['pixr'][-1]
    # print(segcens)
    seg_r   = np.digitize(center,segcens)
    #assert seg_r<len(segcens), "Right segment 'too right', {}".format(seg_r)
    if seg_r<len(segcens):
        pass
    else:
        seg_r = len(segcens)-1
    seg_l   = seg_r-1
    
    lsf_l   = lsf[order,seg_l]
    lsf_r   = lsf[order,seg_r]
   
    f1      = (segcens[seg_r]-center)/(segcens[seg_r]-segcens[seg_l])
    f2      = (center-segcens[seg_l])/(segcens[seg_r]-segcens[seg_l])
    
    loc_lsf = container.lsf(1,totpix)
    loc_lsf['pixl'] = lsf_l.values['pixl']
    loc_lsf['pixr'] = lsf_l.values['pixr']
    loc_lsf['segm'] = lsf_l.values['segm']
    loc_lsf['x']    = lsf_l.values['x']
    loc_lsf['y']    = f1*lsf_l.y + f2*lsf_r.y

    
    return LSF(loc_lsf[0])
def interpolate_local_analytic(lsf,order,center):
    assert np.isfinite(center)==True, "Center not finite, {}".format(center)
    values  = lsf[order].values
    assert len(values)>0, "No LSF model for order {}".format(order)
    #numseg,totpix  = np.shape(values['x'])
    
    segcens = (values['pixl']+values['pixr'])/2
    segcens[0]  = values['pixl'][0]
    segcens[-1] = values['pixr'][-1]
    seg_r   = np.digitize(center,segcens)
    #assert seg_r<len(segcens), "Right segment 'too right', {}".format(seg_r)
    if seg_r<len(segcens):
        pass
    else:
        seg_r = len(segcens)-1
    seg_l   = seg_r-1
    
    lsf_l   = lsf[order,seg_l]
    lsf_r   = lsf[order,seg_r]
   
    f1      = (segcens[seg_r]-center)/(segcens[seg_r]-segcens[seg_l])
    f2      = (center-segcens[seg_l])/(segcens[seg_r]-segcens[seg_l])
    
    loc_lsf = np.zeros_like(lsf.values[0])
    loc_lsf['order'] = lsf_l.values['order']
    loc_lsf['optord'] = lsf_l.values['optord']
    loc_lsf['numlines'] = lsf_l.values['numlines']
    loc_lsf['pixl'] = lsf_l.values['pixl']
    loc_lsf['pixr'] = lsf_l.values['pixr']
    loc_lsf['segm'] = lsf_l.values['segm']
   
    loc_lsf['pars'] = f1*lsf_l.values['pars'] + f2*lsf_r.values['pars']
    loc_lsf['errs'] = np.sqrt((f1*lsf_l.values['errs'])**2 + \
                              (f2*lsf_r.values['errs'])**2)
    return LSF(loc_lsf)

def solve(lsf,linelists,fluxes,backgrounds,errors,fittype,method):
    tot = len(linelists)
    for exp,linelist in enumerate(linelists):
        for i, line in enumerate(linelist):
            od   = line['order']
            segm = line['segm']
            # mode edges
            lpix = line['pixl']
            rpix = line['pixr']
            bary = line['bary']
            cent = line['{}_pix'.format(fittype)][1]
            flx  = fluxes[exp,od,lpix:rpix]
            pix  = np.arange(lpix,rpix,1.) 
            bkg  = backgrounds[exp,od,lpix:rpix]
            err  = errors[exp,od,lpix:rpix]
            wgt  = np.ones_like(pix)
            # initial guess
            p0 = (np.max(flx),cent,1)
            try:
                lsf1s  = lsf[od,segm]
            except:
                continue
    #        print('line=',i)
            try:
                success,pars,errs,chisq,model = hfit.lsf(pix,flx,bkg,err,
                                                  lsf1s,p0,method,
                                                  output_model=True)
            except:
                continue
            if not success:
                print(line)
                pars = np.full_like(p0,np.nan)
                errs = np.full_like(p0,np.nan)
                chisq = np.nan
                continue
            else:
                line['lsf']     = pars
                line['lsf_err'] = errs
                line['lchisq']  = chisq
            #print(line['lsf'])
            
        hf.update_progress((exp+1)/tot,"Solve")
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
    
class LSF(object):
    def __init__(self,narray):
        self._values = narray
    def __getitem__(self,item):
        condict, segm_sent = self._extract_item(item)
        values  = self.values 
        condition = np.logical_and.reduce(tuple(values[key]==val \
                                       for key,val in condict.items()))
        cut = np.where(condition==True)[0]
        
        return LSF(values[cut])

    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning order plus segment.
        
        To be used with partial decorator
        """
        condict = {}
        if isinstance(item,dict):
            if len(item)==2: segm_sent=True
            condict.update(item)
        else:
            dict_sent=False
            if isinstance(item,tuple):
                
                nitem = len(item) 
                if nitem==2:
                    segm_sent=True
                    order,segm = item
                    
                elif nitem==1:
                    segm_sent=False
                    order = item[0]
            else:
                segm_sent=False
                order=item
            condict['order']=order
            if segm_sent:
                condict['segm']=segm
        return condict, segm_sent
    @property
    def values(self):
        return self._values
    @property
    def x(self):
        return self._values['x']
    @property
    def y(self):
        return self._values['y']
    @property
    def deriv(self):
        return self._values['dydx']
    @property
    def pars(self):
        return self._values['pars']
    
    def save(self,filepath,version=None,overwrite=False):
        with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
            hdu.write(self.values,extname='LSF',extver=version)
        hdu.close()
        print("File saved to {}".format(filepath))
        return
    def plot(self,ax=None,title=None,saveto=None,*args,**kwargs):
        if ax is not None:
            ax = ax  
        else:
            plotter = hplot.Figure2(1,1,left=0.08,bottom=0.12)
            figure, ax = plotter.fig, plotter.add_subplot(0,1,0,1)
        # try:
        ax = plot_spline_lsf(self.values,ax,title,saveto,*args,**kwargs)
        # except:
        #     ax = plot_analytic_lsf(self.values,ax,title,saveto,*args,**kwargs)
        
        ax.set_ylim(-0.03,0.35)
        ax.set_xlabel("Distance from center"+r" [kms$^{-1}$]")
        ax.set_ylabel("Relative intensity")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5,steps=[1,2,5]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#        ax.set_yticklabels([])
        ax.grid(True,ls=':',lw=1,which='both',axis='both')

        if title:
            ax.set_title(title)
        if saveto:
            figure.savefig(saveto)
        return ax
    def interpolate(self,order,center):
        
        # if method=='spline':
        return interpolate_local_spline(self,order,center)
        # elif method == 'analytic':
        #     return interpolate_local_analytic(self,order,center)
    
def plot_spline_lsf(values,ax,title=None,saveto=None,*args,**kwargs):
    nitems = len(values)    
    if nitems>1:
        numvals = len(values)
        colors = plt.cm.jet(np.linspace(0,1,numvals))
        for j,item in enumerate(values):
            # print('j=',j)
            color = colors[j]
            lw = kwargs.pop('lw',0.6)
            plot_lsf1s(ax,item,color=color,lw=lw)
    else:            
        color = kwargs.pop('color','C0')
        lw    = kwargs.pop('lw',2)
        plot_lsf1s(ax,values[0],color=color,lw=lw,*args,**kwargs)
    return ax

def plot_lsf1s(ax,item,color,lw,*args,**kwargs):
    npts   = item['y'].shape[-1]
    x = np.linspace(np.min(item['x']),np.max(item['x']),3*npts)
    splr = interpolate.splrep(item['x'],item['y'])                    
    sple = interpolate.splev(x,splr)
    ax.scatter(item['x'],item['y'],edgecolor='None',
                    c=[color])
    ax.plot(x,sple,lw=lw,c=color)
    try:
        ax.fill_between(item['x'],
                        item['y']-item['yerr'],
                        item['y']+item['yerr'],
                        color=color,alpha=0.3)
    except:
        pass
    return ax
def get_empty_lsf(method,numsegs=1,n=None,pixcens=None):
    '''
    Returns an empty array for LSF model.
    
    Args:
    ----
        method:    string ('analytic','spline','gp')
        numsegs:   int, number of segments per range modelled
        n:         int, number of parameters (20 for analytic, 160 for spline, 2 for gp)
        pixcens:   array of pixel centers to save to field 'x'
    '''
    assert method in ['analytic','spline','gp']
    if method == 'analytic':
        n     = n if n is not None else 20
        lsf_cont = container.lsf_analytic(numsegs,n)
    elif method == 'spline':
        n     = n if n is not None else 160
        lsf_cont = container.lsf(numsegs,n)
        lsf_cont['x'] = pixcens
    elif method == 'gp':
        n     = n if n is not None else 2
        lsf_cont = container.lsf_gp(numsegs,n)
    return lsf_cont


def clean_input(x1s,flx1s,err1s=None,filter=None,xrange=None,binsize=None,
                sort=True,verbose=False,rng_key=None):
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
    # bins the flx values into 33 bins along the pix direction and removes
    # outliers in each bin
    if xrange is not None:
        xrange = xrange
    else:
        delta = np.max(x1s)-np.min(x1s)
        if delta>1e3: # x1s is in velocity space
            xrange = 4000 # m/s either side of the line centre
            binsize = binsize if binsize is not None else 100 # m/s
        else: # x1s is in pixel space
            xrange = 8 # pixels
            binsize = binsize if binsize is not None else 0.25 # pixels 
    bins    = np.arange(-xrange,xrange+binsize,binsize)
    idx     = np.digitize(x1s[finite_],bins)
    notout  = ~hf.is_outlier_bins(flx1s[finite_],idx)
    finite  = cut[notout]
    numpts  = np.size(flx1s)
    
     
    x      = x1s[finite]
    flx    = flx1s[finite]
    res    = (x,flx)
    if err1s is not None:
        err = np.ravel(err1s)[finite]
        res = res + (err,)
    if sort:
        sorter = np.argsort(x)
        res = (array[sorter] for array in res)
    if filter:
        # rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(55873)
        # shape   = (len(x)//filter,)
        # choice  = jax.random.choice(rng_key,np.arange(len(x)),shape,False)
        # res = (array[choice] for array in res)
        res = (array[::filter] for array in res)
    res = tuple(res)    
    if verbose:
        diff  = numpts-len(res[0])
        print("{0:5d}/{1:5d} ({2:5.2%}) kept ; ".format(len(res[0]),numpts,
                                                      len(res[0])/numpts) +\
              "{0:5d}/{1:5d} ({2:5.2%}) discarded".format(diff,numpts,
                                                        diff/numpts))
    return tuple(res)   

def plot_lsf_model(pix1s,flx1s,err1s,method,dictionary,
                      checksum,save=False,**kwargs):


    plot_subpix_grid = kwargs.pop('plot_subpix',False)
    plot_model       = kwargs.pop('plot_model',True)
    rasterized       = kwargs.pop('rasterized',False)
    
    plotter=hplot.Figure2(3,1,figsize=(10,8),height_ratios=[3,1,1],
                          hspace=0.15)
    ax0 = plotter.add_subplot(0,1,0,1)
    ax1 = plotter.add_subplot(1,2,0,1,sharex=ax0)
    ax2 = plotter.add_subplot(2,3,0,1)
    ax  = [ax0,ax1]

    #ax[0].plot(pix1s,flx1s,ms=0.3,alpha=0.2,marker='o',ls='')
    
    ax[0].set_ylim(-5,35)
    ax[0].axhline(0,ls=':',c='k')
    ax[1].set_xlabel("Distance from center"+r" [kms$^{-1}$]")
    ax[0].grid()
    for a in ax:
        a.set_xlim(-11,11)
    # if method=='spline' or method=='gp':
    #     ax[0].scatter(lsf1s['x'],lsf1s['y'],marker='s',s=32,
    #               linewidths=0.2,edgecolors='k',c='C1',zorder=1000)
    # else:
    # ax[0].errorbar(pix1s,flx1s,yerr=err1s,ls='',ms=2,alpha=0.5,marker='.',
    #                color='C0',rasterized=rasterized,zorder=10)
    # ax[0].plot(xx,yy,lw=2,c='C1',zorder=2)
    # ax[0].fill_between(xx,np.ravel(yy)-np.ravel(yy_err),
    #                       np.ravel(yy)+np.ravel(yy_err),
    #       alpha=0.6,color='C1',zorder=1)
    if method == 'gpflow':
        plot_gpflow_distribution(pix1s, flx1s, err1s, 
                                 dictionary['mean'], dictionary['mean_err'],
                                 ax0)
        loss_fn = dictionary['loss_fn']
        ax2.plot(np.arange(len(loss_fn)),np.array(loss_fn))
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("Loss function")
    elif method == 'tinygp':
        plot_tinygp_model(pix1s,flx1s,err1s,dictionary['solution'],ax0)
    ax[1].axhline(0,ls=':',c='k')
    ax[1].scatter(pix1s,dictionary['rsd'],s=1)
    ax[1].set_ylabel('Normalised rsd')
    # if method=='spline':
    #     ax[1].errorbar(pixcens,means,ls='',
    #               xerr=0.5/subpix,ms=4,marker='s')
    
   
    if save:
        figname = '/Users/dmilakov/projects/lfc/plots/lsf/'+\
                  'ESPRESSO_{0}.pdf'.format(checksum)
        plotter.save(figname,rasterized=rasterized)
        plt.close(plotter.figure)   
        return None
    else:
        return plotter

def plot_tinygp_model(x,y,y_err,solution,ax):
    X = jnp.array(x)
    Y = jnp.array(y*100)
    Y_err = jnp.array(y_err*100)
    X_grid = jnp.linspace(X.min(),X.max(),400)
    
    gp = build_LSF_GP(solution.params,X,Y_err)
    _, cond = gp.condition(Y, X_grid)

    mu = cond.loc
    std = np.sqrt(cond.variance)
    ax.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
    ax.plot(X_grid, mu, label="Full model")
    for i in [1,3]:
        ax.fill_between(X_grid, mu + i*std, mu - i*std, color="C0", alpha=0.3)


    # Separate mean and GP
    _, cond_nomean = gp.condition(Y, X_grid, include_mean=False)

    mu_nomean = cond_nomean.loc #+ soln.params["mf_amps"][0] # second term is for nicer plot
    std_nomean = np.sqrt(cond_nomean.variance)

    # plt.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
    ax.plot(X_grid, mu_nomean, c='C0', ls='--', label="GP model")
    for i in [1,3]:
        ax.fill_between(X_grid, mu_nomean + i*std_nomean, mu_nomean - i*std_nomean,
                         color="C0", alpha=0.3)
    ax.plot(X_grid, jax.vmap(gp.mean_function)(X_grid), c="C1", label="Gaussian model")

  
    return None

def plot_solution(pix1s,flx1s,err1s,method,dictionary,
                      checksum,save=False,**kwargs):
    plot_subpix_grid = kwargs.pop('plot_subpix',False)
    plot_model       = kwargs.pop('plot_model',True)
    rasterized       = kwargs.pop('rasterized',False)
    
    
    total_shift = -dictionary['totshift']
    
    params_LSF = dictionary['solution_LSF'].params
    N_params = len(params_LSF)
    try:
        solution_scatter = dictionary['solution_scatter']
        params_sct = solution_scatter[0].params
        logvar_x   = solution_scatter[1]
        logvar_y   = solution_scatter[2]  
        scatter    = (params_sct,logvar_x,logvar_y)
        N_params += len(params_sct)
    except:
        scatter = None
    
    X = jnp.array(pix1s)
    Y = jnp.array(flx1s*100)
    Y_err = jnp.array(err1s*100)
    X_grid = jnp.linspace(X.min(),X.max(),400)

    gp = build_LSF_GP(params_LSF,X,Y_err,scatter=scatter)
    
    _, cond = gp.condition(Y, X_grid)
    
    mu = cond.loc
    std = np.sqrt(cond.variance)
    
    plotter = hplot.Figure2(4,2, figsize=(9,8),
                        height_ratios=[5,2,2,2],width_ratios=[5,1])
    
    ax_obs = plotter.add_subplot(0,1,0,1)
    ax_gp  = plotter.add_subplot(1,2,0,1,sharex=ax_obs)
    ax_var = plotter.add_subplot(2,3,0,1,sharex=ax_obs)
    ax_rsd = plotter.add_subplot(3,4,0,1,sharex=ax_obs)
    ax_hst = plotter.add_subplot(3,4,1,2)
    
    for ax in plotter.axes:
        ax.axhline(0,ls=':',c='grey',zorder=5)
    
    # First panel: data, full model and the gaussian model
    ax_obs.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
    ax_obs.plot(X_grid, mu, label="Full model",lw=2,zorder=5)
    for i in [1,3]:
        ax_obs.fill_between(X_grid, mu + i*std, mu - i*std, color="C0", alpha=0.3)
    ax_obs.plot(X_grid, jax.vmap(gp.mean_function)(X_grid), c="C1",ls='--',
             label="Gaussian model",lw=2,zorder=4)   
    # First panel: random samples from GP posterior 
    # rng_key = jax.random.PRNGKey(55873)
    # sampled_f = cond.sample(rng_key,(10,))
    # for f in sampled_f:
    #     ax_obs.plot(X_grid,f,c='C0',lw=0.5)
    
    # Second panel: the Gaussian process + residuals from Gaussian model
    _, cond_nomean = gp.condition(Y, X_grid, include_mean=False)
    
    mu_nomean = cond_nomean.loc 
    std_nomean = np.sqrt(cond_nomean.variance)
    ax_gp.plot(X_grid, mu_nomean, c='C0', ls='--', label="GP model",zorder=5)
    y2lims = [100,-100] # saves y limits for the middle panel
    for i in [1,3]:
        upper = mu_nomean + i*std_nomean
        lower = mu_nomean - i*std_nomean
        if np.max(lower)<y2lims[0]:
            y2lims[0]=np.min(lower)
        if np.max(upper)>y2lims[1]:
            y2lims[1]=np.max(upper)
        ax_gp.fill_between(X_grid, upper, lower,
                         color="C0", alpha=0.3,zorder=0)
    # Second panel: residuals from gaussian model
    _, cond_nomean_predict = gp.condition(Y, X, include_mean=False)
    std_nomean_predict = np.sqrt(cond_nomean_predict.variance)
    Y_gauss_rsd = Y - jax.vmap(gp.mean_function)(X)
    Y_gauss_err = jnp.sqrt(Y_err**2 + std_nomean_predict**2)
    ax_gp.errorbar(X, Y_gauss_rsd, Y_gauss_err, marker='.',color='grey',ls='')
    
    # Third panel: variances
    ax_var = plot_variances(ax_var, X,Y,Y_err,params_LSF,scatter=scatter)
    ax_var.legend(bbox_to_anchor=(1.02, 1.00),fontsize=8)
    
    # Fourth left panel: normalised residuals
    _, cond_predict = gp.condition(Y, X, include_mean=True)
    mu_predict = cond_predict.loc # second term is for nicer plot
    std_predict = np.sqrt(cond_predict.variance)
    
    
    Y_tot = jnp.sqrt(std_predict**2 + Y_err**2)
    rsd = (mu_predict-Y)/Y_tot
    ax_rsd.scatter(X,rsd,marker='.',c='grey')
    ax3_ylims = ax_rsd.get_ylim()
    
    # Fourth right panel: a histogram of normalised residuals
    n, bins, _ = ax_hst.hist(np.ravel(rsd),bins=10,range=ax3_ylims,
                          color='grey',orientation='horizontal',
                          histtype='step',lw=2)
    # Fourth right panel: print and plot horizontal lines for quantiles 
    median    = np.median(np.ravel(rsd))
    quantiles = np.quantile(np.ravel(rsd),[0.05,0.95])
    lower, upper = quantiles - median
    ax_hst.text(0.85,0.9,
             r'${{{0:+3.1f}}}_{{{1:+3.1f}}}^{{{2:+3.1f}}}$'.format(median,lower,upper),
             horizontalalignment='right',
             verticalalignment='center', 
             transform=ax_hst.transAxes, 
             fontsize=7)
    ax_hst.set_ylim(ax_rsd.get_ylim())
    for ax in [ax_rsd,ax_hst]:
        [ax.axhline(val,ls=(0,(10,5,10,5)),color='grey',lw=0.8) for val in [upper,lower]]
    
    chisq = np.sum(rsd**2)
    dof   = (len(Y)-N_params)
    aicc  = chisq + 2*len(Y)*N_params / (len(Y)-N_params-1)
    labels = ['Gaussian $\mu$',
              'Gaussian $\sigma$', 
              'Gaussian $A$', 
              '$y_0$',
              'GP $\sigma$', 
              'GP $l$', 
              'log(GP error)',
              r'log(<Y_err>)',
              '$N$',
              r'$\nu$',
              r'$\chi^2$',
              r'$\chi^2/\nu$',
              'AICc',
              '-log(probability)',
              'Centre']
    values = [params_LSF['mf_loc'], 
              np.exp(params_LSF['log_mf_width']), 
              np.exp(params_LSF['log_mf_amp']), 
              params_LSF['mf_const'], 
              np.exp(params_LSF['log_gp_amp']),
              np.exp(params_LSF['log_gp_scale']),
              params_LSF['log_error'],
              np.log(np.mean(Y)),
              len(Y),  
              dof, 
              chisq, 
              chisq/dof,
              aicc,
              loss_LSF(params_LSF,X,Y,Y_err,scatter),
              total_shift*1000]
    units  = [*2*(r'kms$^{-1}$',),
              *3*('arb.',),
              r'kms$^{-1}$',
              *8*('',),
              *1*(r'ms$^{-1}$',)]
    formats = [*8*('9.3f',),
               *2*('5d',),
               *10*('9.3f',),
               '+9.3f']
    for i,(l,v,m,u) in enumerate(zip(labels,values,formats,units)):
        text = (f"{l:>20} = {v:>{m}}")
        if len(u)>0:
            text+=f' [{u}]'
        ax_obs.text(1.26,0.9-i*0.08,text,
                 horizontalalignment='right',
                 verticalalignment='center', 
                 transform=ax_obs.transAxes, 
                 fontsize=7)
        print(text)
    
    
    Xlarger = np.max([X.max(), np.abs(X.min())])
    ax_obs.set_xlim(-1.025*Xlarger, 1.025*Xlarger)
    y2lim = np.max([*np.abs(y2lims),*Y_gauss_rsd])
    ax_gp.set_ylim(-1.5*y2lim,1.5*y2lim)
    # ax1.set_ylim(-0.05,0.25)
    plotter.axes[-1].set_xlabel("x "+r'[kms$^{-1}$]')
    ax_obs.set_ylabel("y")
    ax_gp.set_ylabel(r"Data $-$ Gaussian")
    ax_rsd.set_ylabel("Residuals\n"+r"$\sigma$")
    ax_hst.set_yticklabels([])
    ax_hst.set_xlabel('#')
    _ = ax_obs.legend()
    _ = ax_gp.legend()
    
    plotter.figure.align_ylabels()
    
    if save:
        figname = '/Users/dmilakov/projects/lfc/plots/lsf/'+\
                  'ESPRESSO_{0}.pdf'.format(checksum)
        plotter.save(figname,rasterized=rasterized)
        _ = plt.close(plotter.figure)   
        return None
    else:
        return plotter
 
def plot_variances(ax, X,Y,Y_err,theta_LSF,scatter=None):
    obs_var = jnp.power(Y_err,2)
    add_var = jnp.broadcast_to(jnp.exp(theta_LSF['log_error']),Y_err.shape)
    tot_var = add_var + obs_var
    ax.scatter(X,obs_var,label='Data',marker='.',c='k',s=2)
    ax.plot(X,add_var,label='GP constant',ls=(0,(1,2,1,2)))
    if scatter is not None:
        theta_scatter, logvar_x, logvar_y = scatter
        gp_scatter = build_scatter_GP(theta_scatter,logvar_x)
        _, cond    = gp_scatter.condition(logvar_y,X)
        inf_var    = jnp.exp(cond.loc)
        tot_var   += inf_var
        ax.plot(X,inf_var,label='Scatter (model)',ls=(0,(5,2,5)))
        ax.scatter(logvar_x,jnp.exp(logvar_y),marker='x',c='grey',
                   label='Scatter (measured)')
    ax.plot(X,tot_var,label='Total variance',ls='-')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\sigma^2$')
    return ax

def plot_analytic_lsf(values,ax,title=None,saveto=None,**kwargs):
    nitems = len(values.shape)
    npts   = 500
    x = np.linspace(-6,6,npts)
    plot_components=kwargs.pop('plot_components',False)
    if nitems>0:
        numvals = len(values)
        colors = plt.cm.jet(np.linspace(0,1,numvals))
        for j,item in enumerate(values):
            y = hf.gaussP(x,*item['pars'])
            ax.plot(x,y,lw=2,c=colors[j])
            if plot_components:
                ylist = hf.gaussP(x,*item['pars'],return_components=True)
                [ax.plot(x,y_,lw=0.6,ls='--',c=colors[j]) for y_ in ylist]
    else:            
        y = hf.gaussP(x,*values['pars'])
        ax.plot(x,y,lw=2)
    return ax
    
def plot_gp_lsf(values,ax,title=None,saveto=None,**kwargs):
    nitems = len(values.shape)
    npts   = 500
    x      = np.linspace(-10,10,npts)
    if nitems>0:
        numvals = len(values)
        colors = plt.cm.jet(np.linspace(0,1,numvals))
        for j, item in enumerate(values):
            pass

    return ax