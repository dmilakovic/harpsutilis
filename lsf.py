#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
from . import functions as hf
from . import settings as hs
from . import io as io
from . import containers as container
from . import plotter as hplot
from . import fit as hfit
from .gaussprocess_class import HeteroskedasticGaussian
from .core import os, np, plt, FITS

#import line_profiler

import errno

from scipy import interpolate
from scipy.optimize import leastsq, brentq, curve_fit
import scipy.stats as stats

from matplotlib import ticker
import hashlib

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern, ExpSineSquared

import gpflow
from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter
import tensorflow as tf
import tensorflow_probability as tfp
import itertools
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
                 segnum=16,numpix=7,subpix=4,filter=10,method='gp'):
        self._outfile = outfile
        self._cache = {}
        self._iter_solve  = iter_solve
        self._iter_center = iter_center
        self._segnum  = segnum
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
    
    def __call__(self,verbose=False):
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
            pix3d, wav3d, flx3d, err3d, orders = stack(fittype,linelists,fluxes,
                                                wavelengths,errors,backgrounds,
                                         self._orders)
            lsf_i    = construct_lsf(pix3d,flx3d,err3d,self._orders,
                                     numseg=self._segnum,
                                     numpix=self._numpix,
                                     subpix=self._subpix,
                                     numiter=self._iter_center,
                                     method=self._method,
                                     filter=self._filter,
                                     verbose=verbose)
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
        
    def save(self,filepath,version=None,overwrite=False):
        with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
            hdu.write(self._lsf_final.values,extname='LSF',extver=version)
        hdu.close()
        print("File saved to {}".format(filepath))
        return

@tf.function
def gaussian(x,loc,scale,amp):
    x_    = tf.cast(x, tf.float64)
    pi    = tf.constant(np.pi,tf.float64)
    # loc   = tf.constant(loc,tf.float32)
    # scale = tf.constant(scale,tf.float32)
    # amp   = tf.constant(amp,tf.float32)
    y     = amp/tf.sqrt(2*pi)/scale*tf.exp((-((x_-loc)/scale)**2)/2)
    return tf.cast(y,tf.float64)
class gaussian_mean_function(gpflow.mean_functions.MeanFunction):
    def __init__(self,loc=0,scale=1,amp=1,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.loc   = loc
        self.scale = scale
        self.amp   = amp
    
    def __call__(self,X):
        if isinstance(X,tf.Tensor):
            x_ = X.numpy()
        else:
            x_ = X
        x = x_.squeeze()
        y = gaussian(x, self.loc, self.scale, self.amp)
        # y = np.tanh(x)
        return tf.transpose([y,tf.zeros_like(y)])
        # return tf.constant(tf.transpose([y,tf.zeros_like(y)]),dtype=tf.float64)     
    


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
            pixl     = line['pixl']
            pixr     = line['pixr']
            
            pix1l = np.arange(pixl,pixr) - line[ftpix][1]
            
            # normalise the flux by area under the central 16 pixels 
            # (8 pixels either side)
            central = np.where((pix1l>=-10) & (pix1l<=10))[0]
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
            # move to frame centered at 0 
            
            normalisation = np.sum(lineflux)
            
            # flx1l = lineflux/normalisation
            pix3d[od,pixpos,exp] = pix1l
            vel3d[od,pixpos,exp] = vel1l
            flx3d[od,pixpos,exp] = lineflux/normalisation
            err3d[od,pixpos,exp] = lineerr/normalisation
    return pix3d,vel3d,flx3d,err3d,orders


def construct_lsf(vel3d, flx3d, err3d, orders, method,
                  numseg=16,numpix=7,subpix=4,numiter=5,**kwargs):
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
                    numiter=5,numpix=10,subpix=4,minpts=50,filter=None,
                    plot=False,checksum=None,**kwargs):
    '''
    Constructs the LSF model for a single segment
    '''
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    ## other keywords
    verbose          = kwargs.pop('verbose',False)
    
    pix1s, flx1s, err1s = clean_input(pix1s,flx1s,err1s,sort=True,
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
        if delta<1e-4:
            print('stopping condition satisfied')
            break
        else:
            pass
        oldshift = shift
        # shift the values along x-axis for improved centering
        pix1s = pix1s+shift  
        
        if method == 'spline':
            function = construct_spline
            shift_method = kwargs.pop('shift_method',2)
            args.update({'numpix':numpix,'subpix':subpix,'minpts':minpts,
                         'shift_method':shift_method})
        elif method == 'analytic':
            function = construct_analytic
        elif method=='gp':
            # function = construct_gaussprocess
            # function = construct_gpflow
            function = construct_gpflow2
            args.update({'numpix':numpix,'subpix':subpix,'checksum':checksum,
                         'plot':plot})
        lsf1s,shift,chisq,rsd,loc,scale=function(pix1s,flx1s,err1s,**args)
        delta = np.abs(shift - oldshift)
        relchange = np.abs(delta/oldshift)
        totshift += shift
        print("iter {0:2d}   shift={1:+5.2e}  ".format(j,shift) + \
              "delta={0:5.2e}   sum_shift={1:5.2e}   ".format(delta,totshift) +\
              "relchange={0:5.2e}  chisq={1:6.2f}".format(relchange,chisq))
        
        #count        +=1
        if plot:
            plot_gpflow_model(pix1s, flx1s, err1s, loc, scale, rsd,
                              checksum, **kwargs)
    print('total shift {0:12.6f}'.format(totshift))   
    
    # save the total number of points used
    lsf1s['numlines'] = len(pix1s)
    return lsf1s

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
    means,stds  = bin_means(pix1s,rsd,pixlims,minpts)
    lsf1s['y'] = lsf1s['y']+means
    
    if shift_method==1:
        shift = shift_anderson(lsf1s['x'],lsf1s['y'])
    elif shift_method==2:
        shift = shift_zeroder(lsf1s['x'],lsf1s['y'])
    dof = len(rsd) - totpix
    chisq = np.sum((rsd/err1s)**2) / dof
    return lsf1s, shift, chisq, rsd, xx, prediction, prediction_err

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
    
    return lsf1s, shift, chisq, rsd, xx, prediction, prediction_err

def construct_gaussprocess(pix1s,flx1s,err1s,numpix,subpix,plot=False,**kwargs):
    '''
    Uses gaussian process (GP) to estimate the LSF shape and error. 

    Parameters
    ----------
    pix1s : array 
        Pixel values.
    flx1s : array
        Flux values.
    err1s : array
        Flux error values.
    numpix : integer
        Number of pixels either side of the line centre that is considered
        when constructing the LSF.
    subpix : integer
        Number of subdivisions of each pixel.

    Returns
    -------
    lsf1s : structured array (see harps.containers)
        Contains the LSF.
    lsfcen : float
        The location of the centre of the LSF with respect to the zeropoint.
        Centre is defined as the position for which the derivative of the
        LSF profile is zero.
    chisq : float
        The chi-squared per degree of freedom for the fit.
        Degrees of freedom = len(data) - len(parameters of the GP kernel)
    rsd : array
        Residuals to the fit.
    xx : array
        A high resolution pixel array covering the same pixel range as the LSF.
        (Used for plotting in the top routine).
    prediction : array
        A high resolution LSF model array covering the same pixel range as the 
        LSF. (Used for plotting in the top routine).
    prediction_err : array
        A high resolution LSF model error array covering the same pixel range 
        as the LSF. (Used for plotting in the top routine).

    '''
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    # pixlims = (pixcens+0.5/subpix)
    lsf1s = get_empty_lsf('spline',1,totpix,pixcens)[0]
    
    white     = WhiteKernel(noise_level=1e0,
                            noise_level_bounds=(1e-10,1e0))
    matern    = Matern(nu=2.5,length_scale=5,
                       length_scale_bounds=(1e-1,5e1)) 
    rbf       = RBF(length_scale=10,length_scale_bounds=(10e0,1e3))
    periodic  = ExpSineSquared(length_scale=1e0, 
                               length_scale_bounds=(5,5e1),
                               periodicity=1.5e1, 
                               periodicity_bounds=(3e0,5e1))
    # kernel    = white + 1**1 * periodic + rbf
    # kernel = white + 1**1*periodic
    
    gpr       = GPR(kernel=matern,alpha=err1s**2)
    gp        = gpr.fit(pix1s[:,np.newaxis],flx1s[:,np.newaxis])
    print(gp.kernel_)
    
    # MODEL PREDICTION
    model     = gp.predict(pix1s[:,np.newaxis]) 
    
    rsd       = flx1s - np.ravel(model)
    xx        = np.linspace(-numpix,numpix,100)
    prediction,prediction_err = gp.predict(xx[:,np.newaxis],return_std=True)
    
    X             = np.linspace(-2,2,50)
    Y             = gp.predict(X[:,np.newaxis])
    
    lsfcen        = -hf.derivative_zero(X,np.ravel(Y),-2,2)
    y, yerr       = gp.predict(pixcens[:,np.newaxis],return_std=True)
    lsf1s['y']    = np.ravel(y)
    lsf1s['yerr'] = np.ravel(yerr)
    
    dof       = len(rsd) - len(gp.kernel_.theta)
    
    chisq = np.sum((rsd/err1s)**2) / dof
    return lsf1s, lsfcen, chisq, rsd, xx, prediction, prediction_err

def construct_gpflow(pix1s,flx1s,err1s,numpix,subpix,plot=False,checksum=None,
                     **kwargs):
    '''
    Uses gaussian process (GP) to estimate the LSF shape and error. 

    Parameters
    ----------
    pix1s : array 
        Pixel values.
    flx1s : array
        Flux values.
    err1s : array
        Flux error values.
    numpix : integer
        Number of pixels either side of the line centre that is considered
        when constructing the LSF.
    subpix : integer
        Number of subdivisions of each pixel.

    Returns
    -------
    lsf1s : structured array (see harps.containers)
        Contains the LSF.
    lsfcen : float
        The location of the centre of the LSF with respect to the zeropoint.
        Centre is defined as the position for which the derivative of the
        LSF profile is zero.
    chisq : float
        The chi-squared per degree of freedom for the fit.
        Degrees of freedom = len(data) - len(parameters of the GP kernel)
    rsd : array
        Residuals to the fit.
    xx : array
        A high resolution pixel array covering the same pixel range as the LSF.
        (Used for plotting in the top routine).
    prediction : array
        A high resolution LSF model array covering the same pixel range as the 
        LSF. (Used for plotting in the top routine).
    prediction_err : array
        A high resolution LSF model error array covering the same pixel range 
        as the LSF. (Used for plotting in the top routine).

    '''
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    lsf1s = get_empty_lsf('spline',1,totpix,pixcens)[0]
    
    # Data
    X, Y = (pix1s[:,np.newaxis], flx1s[:,np.newaxis])
    data = (X,Y)
    
    # Randomly chosen M points:
    N = len(err1s)
    M = np.random.randint(0,N,size=50)
    Z = pix1s[M,np.newaxis].copy() 
    # Kernel
    kernel = gpflow.kernels.Matern52() + gpflow.kernels.White()
    
    m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)

    
    elbo = tf.function(m.elbo)

    # TensorFlow re-traces & compiles a `tf.function`-wrapped method at *every* call if the arguments are numpy arrays instead of tf.Tensors. Hence:
    tensor_data = tuple(map(tf.convert_to_tensor, data))
    elbo(tensor_data)  # run it once to trace & compile
    
    minibatch_size = 100

    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(N)
    train_iter = iter(train_dataset.batch(minibatch_size))
    ground_truth = elbo(tensor_data).numpy()
    evals = [elbo(minibatch).numpy() for minibatch in itertools.islice(train_iter, 100)]
    
    gpflow.set_trainable(m.inducing_variable, False)
    
    maxiter = ci_niter(20000)

    logf = run_adam(m, train_dataset, maxiter, minibatch_size)
    
    
    model, model_err = m.predict_y(X) 
    
    # plt.figure()
    # plt.scatter(pix1s,flx1s)
    # plt.plot(pix1s,np.ravel(model))
    
    rsd       = flx1s - np.ravel(model)
    xx        = np.linspace(-numpix,numpix,100)
    prediction,prediction_err = m.predict_y(xx[:,np.newaxis]) 
    
    X             = np.linspace(-2,2,50)
    Y,Y_err       = m.predict_y(X[:,np.newaxis])
    
    lsfcen        = -hf.derivative_zero(X,np.ravel(Y),-2,2)
    y, yerr       = m.predict_y(pixcens[:,np.newaxis])
    lsf1s['y']    = np.ravel(y)
    lsf1s['yerr'] = np.ravel(yerr)
    
    
    dof       = len(rsd) - len(kernel.parameters)
    
    chisq = np.sum((rsd/err1s)**2) / dof
    return lsf1s, lsfcen, chisq, rsd, xx, prediction, prediction_err, maxiter,logf


def construct_gpflow2(pix1s,flx1s,err1s,numpix,subpix,plot=False,checksum=None,
                     **kwargs):
    '''
    Uses gaussian process (GP) to estimate the LSF shape and error. 

    Parameters
    ----------
    pix1s : array 
        Pixel values.
    flx1s : array
        Flux values.
    err1s : array
        Flux error values.
    numpix : integer
        Number of pixels either side of the line centre that is considered
        when constructing the LSF.
    subpix : integer
        Number of subdivisions of each pixel.

    Returns
    -------
    lsf1s : structured array (see harps.containers)
        Contains the LSF.
    lsfcen : float
        The location of the centre of the LSF with respect to the zeropoint.
        Centre is defined as the position for which the derivative of the
        LSF profile is zero.
    chisq : float
        The chi-squared per degree of freedom for the fit.
        Degrees of freedom = len(data) - len(parameters of the GP kernel)
    rsd : array
        Residuals to the fit.
    xx : array
        A high resolution pixel array covering the same pixel range as the LSF.
        (Used for plotting in the top routine).
    prediction : array
        A high resolution LSF model array covering the same pixel range as the 
        LSF. (Used for plotting in the top routine).
    prediction_err : array
        A high resolution LSF model error array covering the same pixel range 
        as the LSF. (Used for plotting in the top routine).

    '''
    
    
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    lsf1s = get_empty_lsf('spline',1,totpix,pixcens)[0]
    
    X = pix1s[:,np.newaxis]
    N = len(X)
    
    Y = flx1s[:,np.newaxis]*100
    E = err1s[:,np.newaxis]*100
    data = (X,Y) 
    
    # transform = np.exp
    # Compute loc and scale as functions of input X
    f1 = hf.gauss3p
    popt, pcov = curve_fit(f1,pix1s,flx1s,p0=(1,0,np.std(pix1s)))
    
    loc = f1(pix1s,*popt)
    scale = hf.error_from_covar(f1, popt, pcov, pix1s)
    
    # plot_distribution(pix1s, flx1s, err1s, loc, scale,plt.subplot(111))
    
    likelihood = gpflow.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
        scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
    )

    # print(f"Likelihood's expected latent_dim: {likelihood.latent_dim}")
    kernel = gpflow.kernels.SeparateIndependent(
        [
            gpflow.kernels.SquaredExponential(0.5,1),  # This is k1, the kernel of f1
            gpflow.kernels.SquaredExponential(0.5,2),  # this is k2, the kernel of f2
        ]
    )
    # The number of kernels contained in gpf.kernels.SeparateIndependent must be the same as likelihood.latent_dim

    M = 50  # Number of inducing variables for each f_i

    # Initial inducing points position Z
    Z = np.linspace(X.min(), X.max(), M)[:, None]  # Z must be of shape [M, 1]
    
    inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
        [
            gpflow.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
            gpflow.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
        ]
    )
    # mean function
    mean_function = gaussian_mean_function(popt[1],popt[2],100*popt[0])
                           
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_latent_gps=likelihood.latent_dim,
        mean_function=mean_function
    )

    # samples_ = model.predict_f_samples(Z, 20) /100.  # shape (10, 100, 1)
    
    # data = (X, Y)
    loss_fn = model.training_loss_closure(data)
    
    gpflow.utilities.set_trainable(model.q_mu, False)
    gpflow.utilities.set_trainable(model.q_sqrt, False)
    
    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.1) #0.1
    
    adam_vars = model.trainable_variables
    # print(adam_vars)
    adam_opt = tf.optimizers.Adam(0.01) # 0.01
    
    # print(type(X),type(Y),X.dtype,Y.dtype)
    @tf.function
    def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)

    epochs = 420
    log_freq = 20
    
    for epoch in range(1, epochs + 1):
        optimisation_step()
        # sys.exit('pass')
        # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
        # if epoch % log_freq == 0 and epoch > 0:
        #     print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")
        # if epoch == epochs-1:
        #     Ymean, Yvar = model.predict_y(X)
        #     Ymean = Ymean.numpy().squeeze()
        #     Ystd = tf.sqrt(Yvar).numpy().squeeze()
        #     plot_distribution(pix1s, flx1s, err1s, Ymean, Ystd)
            
    # Predict model for data
    Ymean, Yvar = model.predict_y(X)
    Ymean = Ymean.numpy().squeeze()/100.
    Ystd = tf.sqrt(Yvar).numpy().squeeze()/100.
    rsd       = flx1s - np.ravel(Ymean)
    # ax = plt.subplot(111)
    # plot_distribution(pix1s, flx1s, err1s, loc, scale, ax)
    # samples = model.predict_f_samples(Z, 20) /100.  # shape (10, 100, 1)
    # ax.plot(Z, samples_[:, :, 0].numpy().T, "C1", linewidth=0.5)
    # ax.plot(Z, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    # plot_distribution(pix1s, flx1s, err1s, Ymean, Ystd, ax)
    
    # # Predict model on a high resolution grid 
    # xx        = np.linspace(-numpix,numpix,100)
    # YYmean, YYvar = model.predict_y(xx[:,np.newaxis])
    # YYmean = YYmean.numpy().squeeze()
    # YYstd = tf.sqrt(YYvar).numpy().squeeze()
    
    # Predict model on a fine grid around zero and solve for the location
    # at which model's derivative is equal to zero (line centre)
    X_             = np.linspace(-2,2,50)
    Y,Y_var        = model.predict_y(X_[:,np.newaxis])
    lsfcen        = -hf.derivative_zero(X_,np.ravel(Y),-2,2)
    # Predict model for the subpixels
    lsfYmean, lsfYvar = model.predict_y(pixcens[:,np.newaxis])
    lsfYmean = lsfYmean.numpy().squeeze()/100.
    lsfYstd = tf.sqrt(lsfYvar).numpy().squeeze()/100.
    lsf1s['y']    = np.ravel(lsfYmean)
    lsf1s['yerr'] = np.ravel(lsfYstd)
    
    dof       = len(rsd) - len(kernel.parameters)
    
    chisq = np.sum((rsd/err1s)**2) / dof
    return lsf1s, lsfcen, chisq, rsd, Ymean, Ystd
def run_adam(model, train_dataset, iterations, minibatch_size):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf

def plot_distribution(X, Y, E, loc, scale, ax):
    # plt.figure(figsize=(15, 5))
    x = X.squeeze()
    for k in (1, 2):
        lb = (loc - k * scale).squeeze()
        ub = (loc + k * scale).squeeze()
        ax.fill_between(x, lb, ub, color="C1", alpha=1 - 0.05 * k ** 3)
        # ax.scatter(X,((Y-loc)/scale)/1e4-0.2,marker='.')
    ax.plot(x, lb, color="C1")
    ax.plot(x, ub, color="C1")
    ax.plot(X, loc, color="C2")
    ax.errorbar(X, Y, E, color="C0", alpha=0.8, marker='.',ls='')
    
    # plt.show()
    
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
    
    return means,stds
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
    shift = -brentq(hf.derivative_eval,-1,1,args=(lsfx,lsfy))
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
                sort=True,verbose=False):
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
        res = (array[::filter] for array in res)
    res = tuple(res)    
    if verbose:
        diff  = numpts-len(res[0])
        print("{0:5d}/{1:5d} ({2:5.2%}) kept ; ".format(len(res[0]),numpts,
                                                      len(res[0])/numpts) +\
              "{0:5d}/{1:5d} ({2:5.2%}) discarded".format(diff,numpts,
                                                        diff/numpts))
        
    return tuple(res)   

def plot_gpflow_model(pix1s,flx1s,err1s,loc,scale,rsd,
                      checksum,**kwargs):


    plot_subpix_grid = kwargs.pop('plot_subpix',False)
    plot_model       = kwargs.pop('plot_model',True)
    rasterized       = kwargs.pop('rasterized',False)
    
    plotter=hplot.Figure2(2,1,figsize=(10,8),height_ratios=[3,1],
                          hspace=0.15)
    ax0 = plotter.add_subplot(0,1,0,1)
    ax1 = plotter.add_subplot(1,2,0,1,sharex=ax0)
    # ax2 = plotter.add_subplot(2,3,0,1)
    ax  = [ax0,ax1]

    #ax[0].plot(pix1s,flx1s,ms=0.3,alpha=0.2,marker='o',ls='')
    
    ax[0].set_ylim(-0.05,0.35)
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
    
    plot_distribution(pix1s, flx1s, err1s, loc, scale, ax0)
    ax[1].axhline(0,ls=':',c='k')
    ax[1].scatter(pix1s,rsd/err1s,s=1)
    ax[1].set_ylabel('Normalised rsd')
    # if method=='spline':
    #     ax[1].errorbar(pixcens,means,ls='',
    #               xerr=0.5/subpix,ms=4,marker='s')
    # ax[2].plot(np.arange(maxiter)[::10], logf)
    # ax[2].set_xlabel("iteration")
    # ax[2].set_ylabel("ELBO")
   

    figname = '/Users/dmilakov/projects/lfc/plots/lsf/'+\
              'ESPRESSO_{0}.pdf'.format(checksum)
    plotter.save(figname,rasterized=rasterized)
    plt.close(plotter.figure)   
    return 

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