#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:00:05 2023

@author: dmilakov
"""
import numpy as np
import harps.lsf.gp as gp
import harps.read as hread
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
rng_key = jax.random.PRNGKey(55873)
import matplotlib.pyplot as plt
#%%

ftype='HARPS'
# ftype='ESPRESSO'
if ftype == 'HARPS':
    npix = 4096
    od = 39
    fname = '/Users/dmilakov/projects/lfc/dataprod/v_2.2/output/2018-12-05_0812.dat'
    # fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/harps/2015-04-17_1440.dat'
    # fname = '/Users/dmilakov/projects/lfc/list/2018-12-05_A.list'
    # fname = '/Users/dmilakov/projects/lfc/list/HARPS2018-12-10T0525.list'
    # checksum = f'{ftype}_'

if ftype == 'ESPRESSO':
    npix = 9111
    od = 120
    fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/single.dat'


segm = 9
pixl=npix//16*segm
pixr=npix//16*(segm+1)
scale = 'pixel'
# scale = 'velocity'
X_,Y_,Y_err_,fig = hread.get_data(fname,od,pixl,pixr,scale=scale,version=111,
                                  fittype='gauss',filter=None,plot=True)
X = jnp.array(X_)
Y = jnp.array(Y_)
Y_err = jnp.array(Y_err_)

# plt.figure()
# plt.errorbar(X,Y,Y_err,ls='',marker='.',color='grey')

#%%
scatter=False
soln=gp.train_LSF_and_scatter(X,Y,Y_err,scatter=scatter)
for key, val in soln.items():
    print(f'{key:>20} = {val:<20.8f}')
lsf_gp=gp.build_LSF_GP(soln,X,Y,Y_err,scatter=scatter)
_,cond=lsf_gp.condition(Y,X)

fig,(ax1,ax2,ax3)=plt.subplots(3,sharex=True,figsize=(8,6))
ax1.errorbar(X,Y,Y_err,c='C0')
ax1.plot(X,cond.mean,c='C1',lw=3.)
std=np.sqrt(cond.variance)
ax1.fill_between(X,cond.mean-std,cond.mean+std,alpha=0.2,color='C1')
rsd  = (cond.mean - Y)/Y_err
rsd2 = jnp.power(rsd,2.)
ax2.scatter(X,rsd)

_,cond_nomean=lsf_gp.condition(Y,X,include_mean=False)
ax1.plot(X,cond_nomean.mean,c='C2',lw=3.)

chisq = jnp.sum(rsd2)
dof   = len(X)-len(soln)+3

if scatter:
    

    g = gp.get_variance_multiplier(soln,X,Y,Y_err,absolute_err=False)
    gsqrt = np.sqrt(g)
    #sct_gp = gp.build_scatter_GP(soln,X)
    #_, cond_sct_gp = sct_gp.condition(rsd2,X)
    logrsd2 = jnp.log(rsd2)
    ax3.errorbar(X,logrsd2,np.exp(soln['sct_log_epsilon0']),capsize=2,marker='.')
    ax3.plot(X,np.log(g))
    ax3.plot(X,g)
    
    ax2.plot(X,gsqrt,c='C2',lw=3)
    new_Y_err = Y_err*gsqrt
    ax1.errorbar(X,Y,new_Y_err,c='C1',capsize=4,ls='')
    new_rsd  = (cond.mean - Y)/new_Y_err
    ax2.scatter(X,new_rsd)
    ax3.scatter(X,jnp.log(new_rsd**2),marker='.',c='C1')
    [ax2.axhline(_, ls=':',c='C0') for _ in [-1,1]]
    chisq = jnp.sum(jnp.power(new_rsd,2.))
    dof   = len(X)-len(soln)
print(f'Chisq = {chisq/dof}')