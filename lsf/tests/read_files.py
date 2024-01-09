#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:07:40 2023

@author: dmilakov
"""

import numpy as np
import jax.numpy as jnp
import scipy.interpolate as intp
import matplotlib.pyplot as plt
from matplotlib import ticker
import hashlib
import fitsio
import os


class IP_1d(object):
    def __init__(self,narray):
        self._values = narray
        self._cache  = {}
    def __len__(self):
        return len(self._values)
    def __getitem__(self,item):
        values  = self.values 
        cut = np.where(values['segm']==item)
        return IP_1d(values[cut])

    @property
    def values(self):
        return self._values
    
    def plot(self,segm=None,ax=None,title=None,saveto=None,*args,**kwargs):
        
        return_fig = False
        if segm is not None:
            values = self[segm].values
        else:
            values = self.values
        if ax is not None:
            ax = ax  
        else:
            return_fig = True
            figure, ax = plt.subplots(1,1)
        ax = plot_numerical_model(ax,values,*args,**kwargs)
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel("Intensity (arb.)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5,steps=[1,2,5]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(True,ls=':',lw=1,which='both',axis='both')

        if title:
            ax.set_title(title)
        if return_fig:
            return figure, ax
        else:
            return ax
    def interpolate_ip(self,center,N=2):
        return interpolate_local_ip(center,self.values,N=N)
    def interpolate_scatter(self,center,N=2):
        return interpolate_local_scatter(center,self.values,N=N)
    
class IP_2d(object):
    def __init__(self,narray):
        self._values = narray
        self._cache  = {}
        
    def __len__(self):
        return len(self._values)
    def __getitem__(self,item):
        condict, segm_sent = _extract_item_(item)
        values  = self.values 
        condition = np.logical_and.reduce(tuple(values[key]==val \
                                       for key,val in condict.items()))
        cut = np.where(condition==True)[0]
        
        return IP_1d(values[cut])

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
    def segcens(self):
        return get_segment_centres(self.values)
    def plot(self,ax=None,title=None,saveto=None,*args,**kwargs):
        if ax is not None:
            ax = ax  
        else:
            figure, ax = plt.subplots(1,1,)
        
        ax = plot_numerical_model(ax,self._values,*args,**kwargs)
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel("Intensity (arb.)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5,steps=[1,2,5]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(True,ls=':',lw=1,which='both',axis='both')

        if title:
            ax.set_title(title)
        if saveto:
            figure.savefig(saveto)
        return ax

def plot_numerical_model(ax,nummodel,*args,**kwargs):
    if ax is not None:
        pass
    else:
        figure, ax  = plt.subplots(1,1)
    x = nummodel['x']
    y = nummodel['y']
    numseg_sent,npts = np.shape(x)
    if numseg_sent==1:
        x = x[0]
        y = y[0]
        ax.plot(x,y,*args,**kwargs)
    if numseg_sent>5:
        colors = plt.cm.jet(np.linspace(0, 1, numseg_sent))
        for i,(x_,y_) in enumerate(zip(x,y)):
            ax.plot(x_,y_,color=colors[i],*args,**kwargs,label=f'Segment {i+1}')
    
    return ax

def interpolate_local_ip(center,ip_1d_num,N=2):
    ''' Wrapper around interpolate_local with what='IP' '''
    return interpolate_local(center,'IP',ip_1d_num,N=N)
    
def interpolate_local_scatter(center,ip_1d_num,N=2):
    ''' Wrapper around interpolate_local with what='scatter' '''
    return interpolate_local(center,'scatter',ip_1d_num,N=N)
    
def interpolate_local(loc,what,ip_1d_arr,N=2):
    '''
    Interpolates the local IP/scatter model at the position x. 

    Parameters
    ----------
    loc : scalar
        position, x-coordinate in units detector pixels.
    what : string, 'IP' or 'scatter'.
        if 'IP', returns the local IP, psi(x).
        if 'scatter', returns the local modification to variances on the data, 
            g(x)
    ip_1d_arr : numpy structured array
        the array containing the numerical models of the LSF/scatter in a
        single echelle order.
    N : scalar, optional
        number of IP segments closest to 'loc' that should be considered.
        The default is 2.

    Returns
    -------
    loc_lsf_x : numpy array
        x-coordinates of the interpolated LSF/scatter model.
    loc_lsf_y : numpy array
        y-coordinates of the interpolated LSF/scatter model.

    '''
    assert np.isfinite(loc)==True, "Center not finite, {}".format(loc)
    order, (indices, weights) = get_segment_weights(ip_1d_arr, loc, N=N)
    if what=='IP':
        name = 'y'
    elif what=='scatter':
        name = 'scatter'
    lsf_array = [ip_1d_arr[i][name] for i in indices]
    loc_lsf_x    = ip_1d_arr[indices[0]]['x']
    loc_lsf_y    = helper_calculate_average(lsf_array,
                                            weights)
    return loc_lsf_x,loc_lsf_y

def helper_calculate_average(list_array,weights):
    '''
    A helper function to calculate the weighted average from a list of arrays
    with corresponding weights along one dimension

    Parameters
    ----------
    list_array : list
        list of arrays containing data.
    weights : list
        list of weights for each array in list_array.

    Returns
    -------
    average : array
        The new, weighted, array.

    '''
    N = len(list_array[0])
    weights_= np.vstack([np.full(N,w,dtype='float32') for w in weights])
    average = np.average(list_array,axis=0,weights=weights_) 
    return average


def _extract_item_(item):
    '''
    A helper function for extraction of the relevant values from IP_1d and
    IP_2d objects

    Parameters
    ----------
    item : tuple or dictionary
        if tuple, form (order,segment).
        if dictionary, must contain entry for 'order' (optionally also 'segm')

    Returns
    -------
    condict : dictionary
        dictionary with entries for order and segment.
    segm_sent : bool
        True if information on segment was sent as input.

    '''
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
            

def sort_array(func):
    '''
    Wraps a function to return a sorted output, where sorting is done 
    both in echelle order and segment within order.

    Parameters
    ----------
    func : function
        function to wrap.

    Returns
    -------
    wrapped function.

    '''
    def wrapped_func(ip_obj,*args,**kwargs):
        if isinstance(ip_obj,IP_1d) or isinstance(ip_obj,IP_2d):
            values = ip_obj.values
        elif isinstance(ip_obj,np.ndarray):
            values = ip_obj
        else:
            print('Class not recognised')
        orders = np.unique(values['order'])
        if len(orders)==1:
            sorter = np.argsort(values['segm'])
        else:
            sorter_ = []
            for od in orders:
                cut = np.where(values['order']==od)[0]
                _ = np.argsort(values[cut]['segm'])
                sorter_.append(cut[_])
            sorter = np.vstack(sorter_)
        return orders,func(values[sorter],*args,**kwargs)
    return wrapped_func

@sort_array
def get_segment_ledges(ip_obj):
    '''
    Returns left edges of segments

    Parameters
    ----------
    ip_obj : IP_1d or IP_2d or numpy.array
        contains values for the IP.

    '''
    return ip_obj['ledge']

@sort_array
def get_segment_redges(ip_obj):
    '''
    Returns right edges of segments

    Parameters
    ----------
    ip_obj : IP_1d or IP_2d or numpy.array
        contains values for the IP.

    '''
    return ip_obj['redge']

@sort_array 
def get_segment_centres(ip_obj):
    '''
    Returns centres of segments

    Parameters
    ----------
    ip_obj : IP_1d or IP_2d or numpy.array
        contains values for the IP.

    '''
    return (ip_obj['ledge']+ip_obj['redge'])/2

@sort_array
def get_segment_limits(ip_obj):
    '''
    Returns edges of segments

    Parameters
    ----------
    ip_obj : IP_1d or IP_2d or numpy.array
        contains values for the IP.

    '''
    ledges = np.array(ip_obj['ledge'])
    redges = np.array(ip_obj['redge'])
    ndim = len(np.shape(ledges))
    a      = ledges
    if ndim>1:
        b  = np.transpose(redges[:,-1])
        return np.append(a,b[:,None],axis=-1)
    else:
        b  = redges
        return np.append(a,b,axis=-1)
    
@sort_array
def get_segment_weights(ip_1d,loc,N=2):
    '''
    Returns weights at center from N neighbouring IP models. 

    Parameters
    ----------
    ip_1d : a structured numpy array 
        values of IP_1d class.
    loc : scalar, float
        location at which the weights should be calculated.
    N : scalar, int, optional
        number of IP segments closest to 'loc' that should be considered.
        The default is 2.

    Returns
    -------
    segments : array, len(N)
        indices of N segments closest to loc.
    weights : array, len(N)
        weights associated with those segments.

    '''
    sorter=np.argsort(ip_1d['segm'])
    orders, segcens   = get_segment_centres(ip_1d[sorter])
    segdist   = np.diff(segcens)[0] # assumes equally spaced segment centres
    distances = np.abs(loc-segcens)
    if N>1:
        used  = distances<segdist*(N-1)
    else:
        used  = distances<segdist/2.
    inv_dist  = 1./distances
    
    segments = np.where(used)[0]
    weights  = inv_dist[used]/np.sum(inv_dist[used])
    
    return segments, weights
#%% Construct GP
import jax
from functools import partial
from tinygp import GaussianProcess, noise, kernels

def F(x,gp,logvar_y):
    '''
    For a scalar input x and a Gaussian Process GP(mu,sigma**2), 
    returns a scalar output GP(mu (x))
    
    Parameters
    ----------
        x : float32
    Output:
        value : float32
    '''
    
    value = gp.condition(logvar_y,jnp.atleast_1d(x))[1].mean
    return value[0]

def transform(x, sigma, GP_mean, GP_sigma, GP, logvar_y):
    '''
    Rescales the old error value at x-coordinate x using the GP mean 
    and sigma evaluated at x.
    
    F ~ GP(mean, sigma^2)
    F(x=x_i) = log( S_i^2 / sigma_i^2 )
    ==> S_i = sqrt( exp( F(x=x_i) ) ) * sigma_i
            = sqrt( exp( GP_mean) ) * sigma_i
        because of the property of logarithms:
            = exp (GP_mean/2.) * sigma_i
    
    
    Propagation of error gives:
    sigma(S_i) = | S_i / 2 * d(F)/dx|_{x_i} * GP_sigma |
    
    where
    GP_mean = F(x=x_i)
    GP_sigma = sigma(F(x=x_i)) 

    Parameters
    ----------
    x : float32, array_like
        x-coordinate.
    sigma : float32, array_like
        error on the y-coordinate value at x.
    GP_mean : float32, array_like
        mean of the GP evaluated at x.
    GP_sigma : float32, array_like
        sigma of the GP evaluated at x.

    Returns
    -------
    S : float32, array_like
        rescaled error on the y-coordinate at x.
    S_var : float32, array_like
        variance on the rescaled error due to uncertainty on the GP mean.

    '''
    deriv = jax.grad(partial(F,gp=GP,logvar_y=logvar_y))
    dFdx  = jax.vmap(deriv)(x)
    S = sigma * jnp.exp(GP_mean/2.)
    S_var = jnp.power(S / 2. * dFdx * GP_sigma,2.)
    return S, S_var

def rescale_errors(scatter,X,Y_err,plot=False,ax=None):
    '''
    Performs error rescaling, as determined by the scatter parameters

    Parameters
    ----------
    scatter : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    Y_err : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    
    
    theta_scatter, logvar_x, logvar_y, logvar_err = scatter
    
    
    sct_gp        = build_scatter_GP(theta_scatter,logvar_x,logvar_err)
    _, sct_cond   = sct_gp.condition(logvar_y,X)
    F_mean  = sct_cond.mean
    F_sigma = jnp.sqrt(sct_cond.variance)
    
    S, S_var = transform(X,Y_err,F_mean,F_sigma,sct_gp,logvar_y)
    return S, S_var

def gaussian_mean_function(theta, X):
    '''
    Returns the Gaussian profile with parameters encapsulated in dictionary
    theta, evaluated a points in X

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    mean  = theta["mf_loc"]
    sigma = jnp.exp(theta["mf_log_sig"])
    gauss = jnp.exp(-0.5 * jnp.square((X - mean)/sigma)) \
            / jnp.sqrt(2*jnp.pi) / sigma
    beta = jnp.array([gauss,1])
    
    return jnp.array([theta['mf_amp'],theta['mf_const']]) @ beta

def build_scatter_GP(theta,X,Y_err=None):
    '''
    Returns Gaussian Process for the intrinsic scatter of points (beyond noise)

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    sct_const  = jnp.exp(theta['sct_log_const'])
    sct_amp    = jnp.exp(theta['sct_log_amp'])
    sct_scale  = jnp.exp(theta['sct_log_scale'])
    pred = Y_err!=None
    
    def true_func():
        return noise.Diagonal(jnp.power(Y_err,2.))
    def false_func():
        val = 1e-8
        return noise.Diagonal(jnp.full_like(X,val))
    noise1d = jax.lax.cond(pred,true_func,false_func)
    sct_kernel = sct_amp * kernels.ExpSquared(sct_scale) 
    return GaussianProcess(
        sct_kernel,
        X,
        noise= noise1d,
        mean = sct_const
    )
def build_IP_GP(theta,X,Y=None,Y_err=None,scatter=None):
    '''
    Returns a Gaussian Process for the IP. If scatter is not None, tries to 
    include a second GP for the intrinsic scatter of datapoints beyond the
    error on each individual point.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    Y_err : TYPE
        DESCRIPTION.
    scatter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    gp_amp   = jnp.exp(theta['gp_log_amp'])
    gp_scale = jnp.exp(theta["gp_log_scale"])
    kernel   = gp_amp * kernels.ExpSquared(gp_scale) 
    # Various variances (obs=observed, add=constant random noise, tot=total)
    var_add = jnp.exp(theta['log_var_add']) 
    
    if scatter is not None:   
        S, S_var = rescale_errors(scatter, X, Y_err)
        var_data  = jnp.power(S,2.)
    else:
        _pred_ = Y_err!=None
        def _true_func():
            return jnp.power(Y_err,2.)
        def _false_func():
            return jnp.full_like(X, 1e-8)
        var_data = jax.lax.cond(_pred_,_true_func,_false_func)
    var_tot = var_data + var_add
    noise2d = jnp.diag(var_tot)
    Noise2d = noise.Dense(noise2d)
    
    
    return GaussianProcess(
        kernel,
        X,
        noise = noise2d,
        mean=partial(gaussian_mean_function, theta),
    )

#%%

parnames_lfc = ['mf_amp','mf_loc','mf_log_sig','mf_const',
                'gp_log_amp','gp_log_scale','log_var_add']
parnames_sct = ['sct_log_amp','sct_log_scale','sct_log_const']
parnames_all = parnames_lfc + parnames_sct

def prepare_ip1s(ip1s):
    test = len(np.shape(ip1s))
    if test>0:
        return ip1s[0]
    else:
        return ip1s
    
def read_field_from_ip1s(ip1s,field):
    ip1s = prepare_ip1s(ip1s)
    data = ip1s[field] 
    cut  = np.where(~np.isnan(data))[0]
    return jnp.array(data[cut],dtype='float32')

def read_parameters_from_ip1s(ip1s,parnames=None):
    dictionary = {}
    if parnames is not None:
        parnames = np.atleast_1d(parnames)
    else:
        parnames = parnames_lfc + parnames_sct
    for parname in parnames:
        try:
            dictionary.update({parname:jnp.array(ip1s[parname][0])})
        except:
            try:
                dictionary.update({parname:jnp.array(ip1s[parname])})
            except:
                continue
    return dictionary


def read_from_ip1s(ip1s,what):
    if what == 'IP':
        desc = 'data'
        parnames = parnames_lfc
    elif what == 'scatter':
        desc = 'sct'
        parnames = parnames_sct
    
    pars = read_parameters_from_ip1s(ip1s,parnames)
    field_names = [f"{desc}_{coord}" for coord in ['x','y','yerr']]
    x, y, y_err = (read_field_from_ip1s(ip1s,field) for field in field_names)
    return (pars, x, y, y_err)

#%% ================             E X A M P L E S              =================

def read_file(filepath,scale,what):
    '''
    

    Parameters
    ----------
    file : string
        path to the FITS file containing the IP.
    scale : string
        'pixel' or 'velocity'.
    what : string
        'gp' returns the Gaussian Process parameters and training sets.
        'model' returns the numerical model of the IP

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    assert scale in ['pixel','velocity']
    assert what in ['gp','model']
    extname = f'{scale}_{what}'
    return fitsio.read(filepath,ext=extname)


def example_model_plot(filepath):
    '''
    Shows how to read IP data from the FITS file and how to use classes
    IP_2d and IP_1d. 

    Parameters
    ----------
    filepath : string
        path to the FITS file.
    '''
    od     = 50
    segm   = 4
    values = read_file(filepath,scale='pixel',what='model')
    ip2d   = IP_2d(values)
    ip1d   = ip2d[od]
    ip1s   = ip2d[od,segm]
    
    fig,axes = plt.subplots(1,3,figsize=(9,4))
    ip2d.plot(ax=axes[0],title='All IP models')
    ip1d.plot(ax=axes[1],title='Order 50')
    ip1s.plot(ax=axes[2],title='Order 50, segment 4')
    fig.tight_layout()
    return
    
def example_gp_plot(filepath):
    '''
    Shows how to read Gaussian Process hyperparameters from the FITS file, 
    including data. Also shows how to train a Gaussian Process with this data.

    Parameters
    ----------
    filepath : string
        path to the FITS file.
    '''
    od     = 50
    segm   = 4
    values = read_file(filepath,scale='pixel',what='gp')
    ip2d   = IP_2d(values)
    ip1s   = ip2d[od,segm]
    
    pars, x, y, y_err = read_from_ip1s(ip1s.values, 'IP')
    fig,ax = plt.subplots(1,1,figsize=(5,4))
    ax.errorbar(x,y,y_err,ls='',marker='.',capsize=2)
    
    x_test = np.linspace(-5,5,300)
    
    gp_noscatter = build_IP_GP(pars, X=x, Y_err=y_err, scatter = None)
    _, gp_ns_trained = gp_noscatter.condition(y,x_test)
    gp_ns_mean = gp_ns_trained.mean
    gp_ns_std  = np.sqrt(gp_ns_trained.variance)
    ax.plot(x_test,gp_ns_mean,label='No variance modification',c='C1')
    ax.fill_between(x_test,gp_ns_mean-gp_ns_std,gp_ns_mean+gp_ns_std,
                    color='C1',alpha=0.3)
    
    scatter = read_from_ip1s(ip1s.values, 'scatter')
    gp_scatter = build_IP_GP(pars, X=x, Y_err=y_err, scatter=scatter)
    _, gp_sct_trained = gp_scatter.condition(y,x_test)
    gp_sct_mean = gp_sct_trained.mean
    gp_sct_std  = np.sqrt(gp_sct_trained.variance)
    ax.plot(x_test,gp_sct_mean,label='Variance modification',c='C2')
    ax.fill_between(x_test,gp_sct_mean-gp_sct_std,gp_sct_mean+gp_sct_std,
                    color='C2',alpha=0.3)
    
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"Intensity (arb.)")
    ax.legend()
    return

def example_interpolation(filepath):
    '''
    Shows how interpolation is done using IP_1d class. 

    Parameters
    ----------
    filepath : string
        path to the FITS file.
    '''
    od     = 50
    segm   = 4
    loc    = 1111.1
    N      = 2
    values = read_file(filepath,scale='pixel',what='model')
    ip2d   = IP_2d(values)
    ip1d   = ip2d[od]
    
    x_ip_new, y_ip_new = ip1d.interpolate_ip(loc,N)
    
    _,(segments,weights) = get_segment_weights(ip1d, loc,N)
    seg1, seg2 = segments
    w1, w2 = weights
    label = f"{w1:5.2%}*seg{seg1} + {w2:5.2%}*seg{seg2}"
    
    fig,ax = plt.subplots(1,1)
    ax.plot(x_ip_new,y_ip_new,label=label)
    for seg in segments:
        ip1d[seg].plot(ax=ax,label=f'Segment {seg}')
    ax.legend()
    fig.tight_layout()
    
    return 
    
filepath = '/path/to/HARPS.2018-12-07T00:12:50.196_e2ds_A_lsf.fits' 
example_model_plot(filepath)
example_gp_plot(filepath)
example_interpolation(filepath)
