#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:27:41 2022

@author: dmilakov
"""

import jax
import jax.numpy as jnp
import jaxopt
from tinygp import kernels, GaussianProcess, transforms, noise
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
from harps.functions import gauss4p
import harps.lsf as hlsf

jax.config.update("jax_enable_x64", True)
rng_key = jax.random.PRNGKey(55873)
#%%
def get_data(od,pixl,pixr,filter=50):
    import harps.lsf as hlsf
    import harps.io as io
    fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/test.dat'
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
    vel1s_, flx1s_, err1s_ = hlsf.clean_input(vel1s,flx1s,err1s,sort=False,
                                              verbose=True,filter=filter)
    
    X      = jnp.array(vel1s_)
    # X      = jnp.array(pix1s)
    Y      = jnp.array(flx1s_*100)
    Y_err  = jnp.array(err1s_*100)
    # Y      = jnp.array([flx1s_,err1s_])
    return X, Y, Y_err, 
X_,Y_,Y_err_ = get_data(149,5000,5500,5)
#%%
X = X_
Y = Y_
Y_err = Y_err_
plt.figure()
plt.errorbar(X,Y,Y_err,ls='',marker='.',color='grey')
#%%
def gaussian_mean_function(theta, X):
    
    gauss = jnp.exp(
        -0.5 * jnp.square((X - theta["mf_loc"]) / jnp.exp(theta["log_mf_width"]))
    )
    
    beta = jnp.array([1, gauss])
    return jnp.array([theta["mf_const"],
                      jnp.exp(theta["log_mf_amp"])/jnp.sqrt(2*jnp.pi)]) @ beta


def build_gp_sct(theta,X):
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

def build_gp(theta,X,Y_err,scatter=None,plot=False):
    amp   = jnp.exp(theta["log_gp_amp"])
    scale = jnp.exp(theta["log_gp_scale"])
    kernel = amp**2 * kernels.ExpSquared(scale) # LSF kernel
    # amp_sct    = jnp.exp(theta["log_sct_amp"])
    # scale_sct  = jnp.exp(theta["log_sct_scale"])
    # kernel_sct = amp_sct**2 * kernels.ExpSquared(scale_sct) # scatter kernel
    obs_var = jnp.power(Y_err,2)
    add_var = jnp.broadcast_to(jnp.exp(theta['log_error']),Y_err.shape)
    tot_var = add_var + obs_var
    # print(tot_var)
    if scatter is not None:
        theta_scatter, logvar_x, logvar_y = scatter
        gp_scatter = build_gp_sct(theta_scatter,logvar_x)
        _, gp_scatt_cond = gp_scatter.condition(logvar_y,X)
        inf_var = jnp.exp(gp_scatt_cond.loc)
        tot_var += inf_var
        
        if plot:
            plt.figure()
            plt.plot(X,obs_var,label='(observed error)^2')
            plt.plot(X,add_var,label='exp(log_error)^2')
            plt.plot(X,inf_var,label='predicted scatter variance')
            plt.plot(X,tot_var,label='total variance')
            plt.legend()
    nn = noise.Diagonal(tot_var)
    return GaussianProcess(
        kernel,
        X,
        # noise = noise.Dense(kernel_sct(X,X)**2),
        # noise = noise.Dense(Y_err**2+kernel_sct(X,X)**2),
        # noise = noise.Diagonal(Y_err**2),
        noise = nn,
        mean=partial(gaussian_mean_function, theta),
        # mean = 0.0
    )
@jax.jit
def loss(theta,X,Y,Y_err,scatter):
    gp = build_gp(theta,X,Y_err,scatter)
    return -gp.log_probability(Y)
@jax.jit
def loss2(theta,X,Y):
    gp = build_gp_sct(theta,X)
    return -gp.log_probability(Y)
 
def train_tinygp(X,Y,Y_err,scatter=None):
    
   
    
    popt,pcov = curve_fit(gauss4p,X,Y,sigma=Y_err,
                          absolute_sigma=False,p0=(np.max(Y),0,np.std(X),0))
    perr = np.sqrt(np.diag(pcov))
    mean_params = dict(
        mf_const     = popt[3],
        log_mf_amp   = np.log(np.abs(popt[0])),
        mf_loc       = popt[1],
        log_mf_width = np.log(np.abs(popt[2])),
    )
    theta = dict(
        log_error = -1.,
        log_gp_amp=np.array(1.),
        log_gp_scale=np.array(1.),
        # log_sct_amp=np.array(0.1),
        # log_sct_scale=np.array(-1.),
        **mean_params
    )

    kappa = 10
    lower_bounds = dict(
        log_error = -10.,
        log_gp_amp = -2.,
        log_gp_scale = np.log(0.4), # corresponds to 400 m/s
        log_mf_amp = np.log(np.abs(popt[0])-kappa*perr[0]),
        log_mf_width=np.log(np.abs(popt[2])-kappa*perr[2]),
        # log_sct_amp= -1.,
        # log_sct_scale= -3.,
        mf_const = popt[3]-kappa*perr[3],
        mf_loc = popt[1]-3*perr[1],
    )
    upper_bounds = dict(
        log_error = 0.,
        log_gp_amp = 2.,
        log_gp_scale = 2.,
        log_mf_amp = np.log(np.abs(popt[0])+kappa*perr[0]),
        log_mf_width=np.log(np.abs(popt[2])+kappa*perr[2]),
        # log_sct_amp= 1.,
        # log_sct_scale= 3. ,
        mf_const = popt[3]+kappa*perr[3],
        mf_loc = popt[1]+3*perr[1],
    )
    bounds = (lower_bounds, upper_bounds)
    
    
    solver = jaxopt.ScipyBoundedMinimize(fun=partial(loss,X=X,Y=Y,
                                                     Y_err=Y_err,
                                                     scatter=scatter),
                                         method="l-bfgs-b")
    solution = solver.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
   
    return solution

def train_tinygp_sct(X,Y,Y_err,theta_lsf):
    '''
    Based on Kersting et al. 2007 :
        Most Likely Heteroscedastic Gaussian Process Regression

    '''
    
    gp = build_gp(theta_lsf,X,Y_err)
    _, cond = gp.condition(Y,X)
    
    mean, variance_ = cond.loc, np.sqrt(cond.variance)
    rsd = np.exp(jnp.array(Y - mean))
    n_bins = 20
    xlims = np.linspace(X.min(),X.max(),n_bins+1)
    xdist = np.diff(xlims)[0] # size of the bin in km/s
    xcens = (xlims - xdist/2)[1:]
    bin_means, bin_stds, counts = hlsf.bin_means(X,rsd,xlims,minpts=5,kind='spline')
    X_obs = jnp.array(xcens)
    logvar = jnp.log(bin_stds[1:]**2)
    # print(X_obs,logvar)
    # take s samples for each input X coordinate
    # X_obs = X
    # s = 100
    # samples = cond.sample(rng_key,(s,))
    # variance = 1./s * jnp.sum(0.5*(mean-samples)**2,axis=0)
    # logvar = jnp.log(variance)
    
    
    
    theta = dict(
        log_sct_amp= 2.0,
        log_sct_scale = 1.0,
        )
    lower_bounds = dict(
        log_sct_amp= -5.0,
        log_sct_scale = np.log(xdist),
        )
    upper_bounds = dict(
        log_sct_amp= 10.0,
        log_sct_scale = 3.0,
        )
    bounds = (lower_bounds, upper_bounds)

    solver = jaxopt.ScipyBoundedMinimize(fun=partial(loss2,
                                                    # X=X,
                                                      X=X_obs,
                                                     Y=logvar
                                                     ),
                                         method="l-bfgs-b")
    solution = solver.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
    # solution = solver.run(jax.tree_map(jnp.asarray, theta))
    # print(solution.params)
    X_grid = jnp.linspace(X_obs.min(),X_obs.max(),400)
    gp_sct = build_gp_sct(solution.params,X_obs)
    _, cond_sct = gp_sct.condition(logvar,X_grid)
    sct_mean = cond_sct.mean
    # print(sct_mean)
    sct_err  = np.sqrt(cond_sct.variance)
    # plt.figure()
    # plt.scatter(X_obs,logvar)
    # plt.plot(X_grid,sct_mean,c='C1')
    # for i in range(1):
    #     plt.fill_between(X_grid, sct_mean+i*sct_err, sct_mean-i*sct_err, 
    #                       color='C1',
    #                       alpha=0.5)
    return solution, X_obs, logvar


soln = train_tinygp(X,Y,Y_err)
for i in range(1):
    scatter_solution, logvar_x, logvar_y = train_tinygp_sct(X,Y,Y_err,soln.params)
    LSF_solution = train_tinygp(X,Y,Y_err,(scatter_solution.params,logvar_x, logvar_y))
    print("scatter\n", scatter_solution.params)
    print("LSF\n", LSF_solution.params)
#%%
for s in [soln,LSF_solution]:
    gp3 = build_gp(s.params,X,Y_err,(scatter_solution.params,logvar_x,logvar_y),plot=True)
    # gp3 = build_gp(soln.params,X,Y_err)
    X_grid = jnp.linspace(X.min(),X.max(),400)
    _,gp3_cond = gp3.condition(Y,X_grid)
    mu=gp3_cond.loc
    std = np.sqrt(gp3_cond.variance)
    plt.figure()
    plt.errorbar(X,Y,Y_err,ls='',marker='.')
    plt.plot(X_grid,mu,c='k')
    for i in range(3):
         plt.fill_between(X_grid, mu+i*std, mu-i*std, 
                          color='grey',
                          alpha=0.5)
    print(f"Final negative log likelihood: {soln.state.fun_val}")

#%%
import harps.plotter as hplt

def plot_solution(params,X,Y,Y_err):
    X_test = jnp.linspace(X.min(), X.max(), 400)


    gp = build_gp(params,X,Y_err)
    _, cond = gp.condition(Y, X_test)
    
    mu = cond.loc
    std = np.sqrt(cond.variance)
    
    fig = hplt.Figure2(3,2, figsize=(9,6), height_ratios=[5,2,2],width_ratios=[5,1])
    
    ax1 = fig.add_subplot(0,1,0,1)
    ax2 = fig.add_subplot(1,2,0,1,sharex=ax1)
    ax3 = fig.add_subplot(2,3,0,1,sharex=ax1)
    ax4 = fig.add_subplot(2, 3, 1, 2)
    
    for ax in fig.axes:
        ax.axhline(0,ls=':',c='grey',zorder=5)
    
    # Top panel: data, full model and the gaussian model
    ax1.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
    ax1.plot(X_test, mu, label="Full model",lw=2,zorder=5)
    for i in [1,3]:
        ax1.fill_between(X_test, mu + i*std, mu - i*std, color="C0", alpha=0.3)
    ax1.plot(X_test, jax.vmap(gp.mean_function)(X_test), c="C1",ls='--',
              label="Gaussian model",lw=2,zorder=4)    
    
    # samples = cond.sample(rng_key,(100,3))
    # for s in samples:
    #     ax1.plot(X_test,s.T)
    
    # Middle panel: the Gaussian process only
    _, cond_nomean = gp.condition(Y, X_test, include_mean=False)
    
    mu_nomean = cond_nomean.loc 
    std_nomean = np.sqrt(cond_nomean.variance)
    ax2.plot(X_test, mu_nomean, c='C0', ls='--', label="GP model")
    y2lims = [100,-100] # saves y limits for the middle panel
    for i in [1,3]:
        upper = mu_nomean + i*std_nomean
        lower = mu_nomean - i*std_nomean
        if np.max(lower)<y2lims[0]:
            y2lims[0]=np.min(lower)
        if np.max(upper)>y2lims[1]:
            y2lims[1]=np.max(upper)
        ax2.fill_between(X_test, upper, lower,
                          color="C0", alpha=0.3)
    # Middle panel: residuals from gaussian model
    Y_gauss_rsd = Y - jax.vmap(gp.mean_function)(X)
    ax2.scatter(X, Y_gauss_rsd, marker='.',c='grey')
    
    # Bottom left panel: normalised residuals
    _, cond_predict = gp.condition(Y, X, include_mean=True)
    mu_predict = cond_predict.loc # second term is for nicer plot
    std_predict = np.sqrt(cond_predict.variance)
    
    
    Y_tot = jnp.sqrt(std_predict**2 + Y_err**2)
    rsd = (mu_predict-Y)/Y_tot
    ax3.scatter(X,rsd,marker='.',c='grey')
    ax3_ylims = ax3.get_ylim()
    
    # Bottom right panel: a histogram of normalised residuals
    ax4.hist(np.ravel(rsd),bins=20,range=ax3_ylims,
              color='grey',orientation='horizontal',histtype='step',lw=2)
    
    
    chisq = np.sum(rsd**2)
    dof   = (len(Y)-len(params))
    labels = ['Gaussian $\mu$','Gaussian $\sigma$', 'Gaussian $A$', '$y_0$',
              'GP $\sigma$', 'GP $l$', 'log(GP error)','$N$', r'$\nu$',r'$\chi^2$',
              r'$\chi^2/\nu$','-log(probability)']
    values = [params['mf_loc'], np.exp(params['log_mf_width']), 
              np.exp(params['log_mf_amp']), params['mf_const'], 
              np.exp(params['log_gp_amp']),np.exp(params['log_gp_scale']),
              params['log_error'], len(Y),  dof, chisq, chisq/dof, 
              loss(params,X=X,Y=Y,Y_err=Y_err)]
    units  = [*2*(r'kms$^{-1}$',),*3*('arb.',),'km/s',*5*('',)]
    formats = [*7*('9.3f',),*2*('5d',),*2*('9.3f',)]
    for i,(l,v,m,u) in enumerate(zip(labels,values,formats,units)):
        text = (f"{l:>20} = {v:>{m}}")
        if len(u)>0:
            text+=f' [{u}]'
        ax1.text(1.26,0.9-i*0.08,text,horizontalalignment='right',
          verticalalignment='center', transform=ax1.transAxes, fontsize=7)
        print(text)
    
    
    Xlarger = np.max([X.max(), np.abs(X.min())])
    ax1.set_xlim(-1.025*Xlarger, 1.025*Xlarger)
    y2lim = np.max([np.abs(y2lims)])
    ax2.set_ylim(-1.5*y2lim,1.5*y2lim)
    # ax1.set_ylim(-0.05,0.25)
    ax3.set_xlabel("x "+r'[kms$^{-1}$]')
    ax1.set_ylabel("y")
    ax3.set_ylabel("Norm. resids")
    ax4.set_yticklabels([])
    ax4.set_xlabel('#')
    _ = ax1.legend()
    # _ = ax2.legend()
plot_solution(soln.params,X,Y,Y_err)
#%%
def model_numpyro(x, y=None):
    # The parameters of the GP model
    mf_loc       = numpyro.sample("mf_loc", dist.Normal(0.0, 2.0))
    log_mf_width = numpyro.sample("log_mf_width", dist.HalfNormal(2.0))
    mf_const     = numpyro.sample("mf_const", dist.Uniform(0.0,10.0))
    log_mf_amp   = numpyro.sample("log_mf_amp", dist.HalfNormal(10.0))
    log_gp_amp   = numpyro.sample("log_gp_amp", dist.HalfNormal(5.0))
    log_gp_scale = numpyro.sample("log_gp_scale", dist.Uniform(0.0,2.0))
    # sys.exit()
    
    # diag = 1e-5
    theta = dict(
        mf_loc=mf_loc,
        log_mf_width=log_mf_width,
        mf_const=mf_const,
        log_mf_amp=log_mf_amp,
        log_gp_diag=log_gp_diag,
        log_gp_amps=log_gp_amp,
        log_gp_scale=log_gp_scale,
    )
    # Set up the kernel and GP objects for the LSF
    kernel_lsf = jnp.exp(log_gp_amp) * kernels.ExpSquared(log_gp_scale)
    gp_lsf = GaussianProcess(kernel_lsf, 
                             x, 
                             diag=log_gp_diag, 
                             mean=partial(mean_function, theta))
    
    # LSF model
    inferred_lsf = numpyro.sample("gp_lsf", gp_lsf.numpyro_dist())
    
    # Intrinsic scatter
    log_sct_amp   = numpyro.sample("log_sct_amp", dist.HalfNormal(5.0))
    log_sct_scale = numpyro.sample("log_sct_scale", dist.Uniform(0.0,2.))
    log_sct_diag  = numpyro.sample("log_sct_diag", dist.Uniform(0.0,2.0))
    kernel_sct    = jnp.exp(log_sct_amp) * kernels.ExpSquared(log_sct_scale)

    gp_scatter = GaussianProcess(kernel_sct, 
                                 x, 
                                 diag=log_sct_diag, 
                                 mean=0)
    
    # log scatter model
    log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    
    numpyro.sample("obs", 
                   dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
                   obs=y)

def estimate_centre(rng_key,X,Y,Y_err,LSF_solution,scatter=None):
    if scatter is not None:
        scatter_solution, logvar_x, logvar_y  = scatter
        gp = build_gp(LSF_solution.params,X,Y_err,
                          (scatter_solution.params,
                           logvar_x,
                           logvar_y
                           )
                          )
    else:
        gp = build_gp(LSF_solution.params,X,Y_err)
    
    X_grid = jnp.arange(-2,2,0.01)
    _, cond = gp.condition(Y,X_grid)
    
    
    return centre, uncertainty
