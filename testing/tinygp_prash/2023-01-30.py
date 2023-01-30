#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:39:11 2023

A collection of code used to make tests on the proper calculation of scatter

@author: dmilakov
"""

#%%
import numpy as np
import jax
import jax.numpy as jnp
import harps.lsf as hlsf
from .2023-01-18 import get_data
#%%
def get_data(od,pixl,pixr,filter=50):
    import harps.lsf as hlsf
    import harps.io as io
    fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/single.dat'
    modeller=hlsf.LSFModeller(fname,60,70,method='gp',subpix=10,
                              filter=None,numpix=8,iter_solve=1,iter_center=1)

    extensions = ['linelist','flux','background','error','wavereference']
    data, numfiles = io.mread_outfile(modeller._outfile,extensions,701,
                            start=None,stop=None,step=None)
    linelists=data['linelist']
    fluxes=data['flux']
    errors=data['error']
    backgrounds=data['background']
    # backgrounds=None
    wavelengths=data['wavereference']
    
    plt.figure()
    plt.plot(wavelengths[0,od,pixl:pixr],(fluxes/errors)[0,od,pixl:pixr])
    plt.ylabel('S/N')
    
    orders=np.arange(od,od+1)
    pix3d,vel3d,flx3d,err3d,orders=hlsf.stack('gauss',
                                              linelists,
                                              fluxes,
                                              wavelengths,
                                              errors,
                                              backgrounds,
                                              orders)


    pix1s=pix3d[od,pixl:pixr]
    vel1s=vel3d[od,pixl:pixr]
    flx1s=flx3d[od,pixl:pixr]
    err1s=err3d[od,pixl:pixr]

    # vel1s_ , flx1s_, err1s_ = vel1s, flx1s, err1s
    x = pix1s
    # x = vel1s
    vel1s_, flx1s_, err1s_ = hlsf.clean_input(x,flx1s,err1s,sort=False,
                                              verbose=True,filter=filter)
    
    X      = jnp.array(vel1s_)
    # X      = jnp.array(pix1s)
    Y      = jnp.array(flx1s_*100)
    Y_err  = jnp.array(err1s_*100)
    # Y      = jnp.array([flx1s_,err1s_])
    plt.figure(); plt.plot(X,Y/Y_err,'.k'); plt.ylabel('S/N')
    
    return X, Y, Y_err, 
# od=102
od = 120
seg = 12
pixl=9111//16*seg
pixr=9111//16*(seg+1)
# pixl=2235
# pixr=2737
# pixl = 3200
# pixr = 3500
X_,Y_,Y_err_ = get_data(od,pixl,pixr,None)
#%%
LSF_solution = hlsf.train_LSF_tinygp(X,Y,Y_err,scatter=None)
#%%
hlsf.estimate_variance_bin(X,Y,Y_err,LSF_solution,scale=15,nbins=50,minpts=10,plot=True)
#%%
scatter = hlsf.train_scatter_tinygp(X,Y,Y_err,LSF_solution,include_error=True)
scatter_pars, var_x, var_y, var_y_err = scatter
#%%
plt.figure()
plt.errorbar(var_x,var_y,var_y_err,ls='',marker='x',capsize=2)
sct_gp = hlsf.build_scatter_GP(scatter_pars,var_x,Y_err=var_y_err)
_,cond_sct_gp = sct_gp.condition(var_y,var_x)
mean  = cond_sct_gp.mean
sigma = np.sqrt(cond_sct_gp.variance)
plt.plot(var_x,mean,'-k')
for i in [1,3]:
    plt.fill_between(var_x,mean-i*sigma,mean+i*sigma,alpha=0.25,color='k')
#%%
LSF_solution_wscatter = hlsf.train_LSF_tinygp(X,Y,Y_err,scatter=scatter)
#%%
fig,ax=plt.subplots(1,1)
hlsf.plot_tinygp_model(X,Y,Y_err,LSF_solution_wscatter,ax,scatter)
#%%
fig,ax=plt.subplots(1,1)
hlsf.plot_tinygp_model(X,Y,Y_err,LSF_solution,ax,None)

#%%
ax=plt.subplot()
hlsf.plot_variances(ax,X,Y,Y_err,LSF_solution,scatter,yscale='linear')
