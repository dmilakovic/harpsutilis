#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:20:45 2023

@author: dmilakov
"""
import harps.lsf.read as read
import harps.lsf.gp as hlsfgp
import numpy as np
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp

# @partial()
def numpyro_model(x_test, y_data, y_err, lsf1d, N=2,rng_key=None):
    rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(55873)
    
    
    bary = jnp.average(x_test,weights=y_data)
    # N = 2 if interpolate == True else 1
    
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    # amp numbers
    amp_mean = jnp.max(y_data)
    amp_std  = 0.05*amp_mean
    # cen numbers
    cen_mean = bary
    cen_std  = 0.1
    # wid numbers
    wid_mean = 1.0
    wid_std  = 0.01
    
    # parameters to fit
    amp = numpyro.sample("amp",dist.Normal(amp_mean,amp_std),rng_key=rng_key)
    cen = numpyro.sample("cen",dist.Normal(cen_mean,cen_std),rng_key=rng_key)
    wid = numpyro.sample("wid",dist.Normal(wid_mean,wid_std),rng_key=rng_key)
    
    
    theta = dict(
        amp = amp,
        cen = cen, 
        wid = wid
    )
    model, error = return_model(theta,x_test,weights,LSF_data)
    # return model,error
    return numpyro.sample("obs",  dist.Normal(model, error), obs=y_data,
                          rng_key=rng_key)

def return_model(theta,x_test,weights,LSF_data):
    amp,cen,wid = helper_extract_params(theta)
    x = helper_rescale_xarray(theta, x_test)
    
    N = len(LSF_data)
    model_list = []
    error_list = []
    for i in range(N):
        LSF_theta, LSF_x, LSF_y, LSF_yerr = LSF_data[i]
        mean, error = hlsfgp.get_model(x,LSF_x,LSF_y,LSF_yerr,LSF_theta,
                                     scatter=None)
        model_list.append(mean)
        error_list.append(error)
        
    model_ = helper_calculate_average(model_list, weights,len(x_test))
    error_ = helper_sum_errors(*error_list)
    
    normalisation = amp / jnp.max(model_)
    model = model_ * normalisation
    error = error_ * normalisation
    return model, error    
 

def extract_lists(what,center,lsf1d,N=2):
    assert what in ['LSF','scatter']
    segments = jnp.arange(len(lsf1d))
    used, weights = get_segment_weights(center,lsf1d,N)
    # weights = get_segment_weights(center,lsf1d,N)
    ll = []
    for i,segm in enumerate(segments):
        if used[i]:
            pass
        else:
            continue
        print(f'segment {i}')
        data = read.from_lsf1s(lsf1d[i],what)
        ll.append(data)
    return ll,weights

    
def get_segment_centres(lsf1d):
    segcens = jnp.array((lsf1d['pixl']+lsf1d['pixr'])/2.)
    return segcens


def get_segment_weights(center,lsf1d,N=2):
    unique    = jnp.arange(len(lsf1d))
    segcens   = (lsf1d['pixl']+lsf1d['pixr'])/2.
    segdist   = jnp.diff(segcens)[0] # assumes equally spaced segment centres
    distances = jnp.abs(center-segcens)
    # condition = True if whitin the limits set by segcens
    cond1 = center>=segcens.min()
    cond2 = center<=segcens.max()
    cond  = jnp.logical_or(cond1,cond2)
    
    
    
    used    = jax.lax.cond(((cond1) | (cond2)) | N>1,
                          true_fun = lambda: distances<segdist*(N-1),
                          false_fun = lambda: distances<segdist/2.)
    # print(used)
    # M = 1 if cond else N
    # used  = jax.lax.cond(cond)
    # if M>1:
    #     # used  = jnp.where(distances<segdist*(N-1))[0]
    #     used = distances<segdist*(N-1)
    # else:
    #     # used  = jnp.where(distances<segdist/2.)[0]
    #     used = distances<segdist/2.
    # segments  = jnp.where(used>0)
    # 
    # weights = jnp.ones_like(segments)
    print(used)
    inv_dist  = 1./distances
    weights   = inv_dist[used]/jnp.sum(inv_dist[used])
    # return weights
    return used, weights

def helper_calculate_average(list_array,weights,N):
    weights_= jnp.vstack([jnp.full(N,w,dtype='float32') for w in weights])
    average = jnp.average(list_array,axis=0,weights=weights_) 
    return average

def helper_extract_params(theta):
    # print(theta)
    try:
        amp = theta['amp']
        cen = theta['cen']
        wid = jnp.abs(theta['wid'])
    except:
        amp, cen, wid = theta
        wid = jnp.abs(wid)
    return amp,cen,wid
def helper_rescale_errors(theta,x_test,y_err,sct_data,weights):
    
    amp,cen,wid = helper_extract_params(theta)
    x   = jnp.array((x_test-cen) * wid)
    
    S_list = []
    for scatter in sct_data:
        S, S_var = hlsfgp.rescale_errors(scatter,x,y_err,plot=False)
        S_list.append(S)
    average = helper_calculate_average(S_list,weights,len(x_test))   
     
    return average

def helper_rescale_xarray(theta,x_test):
    amp,cen,wid = helper_extract_params(theta)
    
    x   = jnp.array((x_test-cen) * wid)
    return x

# def helper_sum_errors(array1,array2):
def helper_sum_errors(*terms):    
    # squared1 = jnp.power(array1,2.)
    # squared2 = jnp.power(array2,2.)
    squared = jnp.vstack([jnp.power(_,2) for _ in terms])
    # X = jnp.sqrt(squared1+squared2)
    X = jnp.sqrt(squared)
    return X
