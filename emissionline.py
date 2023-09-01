#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:56:48 2018

@author: dmilakov
"""
from harps.core import np
from harps.core import plt
import warnings

#from harps.functions import get_fig_axes

from scipy.optimize import leastsq, brentq, least_squares, OptimizeWarning
from scipy.optimize._lsq.least_squares import prepare_bounds
from scipy.linalg import svd
from scipy.special import erf, erfc
#import scipy.interpolate as interpolate


class EmissionLine(object):
    def __init__(self):#,xdata,ydata,yerr,absolute_sigma=False,bounds=None):
        ''' Initialize the object using measured data
        
        Args:
        ----
            xdata: 1d array of x-axis data (pixels or wavelengts)
            ydata: 1d array of y-axis data (electron counts)
            weights:  1d array of weights calculated using Bouchy method
            kind: 'emission' or 'absorption'
        '''
       
#        self.barycenter = np.sum(self.xdata*self.ydata)/np.sum(self.ydata)

    def _get_parameters(self):
        ''' Method to check whether the fit has been successfully performed.
        If the fit was successful, returns the fit values. Otherwise, the 
        method performs the fitting procedure and returns the fit values.
        Returns:
        -------
            pfit: tuple with fitted (amplitude, mean, sigma) values
        '''
        if self.success == True:
            params = self.pars
            errors = self.errs
        else:
            pass
#            p0 = self._initialize_parameters()
#            pars, errors = self.fit(p0)
#            pfit = self.fit_parameters
#            errors = self.fit_errors
        return params, errors
    @property
    def params(self):
        return self._get_parameters()
    def _wrap_jac(self,jac,xdata=None,weights=None,*args):
        if xdata is None:
            xdata = self.xdata
        if weights is None:
            def jac_wrapped(xdata,params):
                return jac(xdata,*args)
        elif weights.ndim == 1:
            weights = self.weights[1:-1]
            def jac_wrapped(xdata,params):
                return weights[:, np.newaxis] * np.asarray(jac(xdata,*args))
        
        return jac_wrapped
    def residuals(self,pars,weights=None):
        ''' Returns the residuals of individual data points to the model.
        Args:
        ----
            pars: tuple (amplitude, mean, sigma) of the model
        Returns:
        -------
             1d array (len = len(xdata)) of residuals
        '''
#        model = self.model(*pars)
        if weights is not None:
            w = weights
        else:
            w = assign_weights(self.xdata[1:-1], pars[1], self.scale)
        obsdata = self.ydata[1:-1]
        resids  = ((obsdata - self.model(pars))/self.yerr[1:-1])*w
        return resids
    def chisq(self,pars=None,weights=None):
        ''' Returns the chi-square of data points to the model.
        Args:
        ----
            pars: tuple (amplitude, mean, sigma) of the model
        Returns:
        -------
            chisq
        '''
        if pars is None:
            pars = self._get_parameters()[0]
        return (self.residuals(pars)**2).sum() / self.dof
    def calc_R2(self,pars=None,weights=None):
        ''' Returns the R^2 estimator of goodness of fit to the model.
        Args:
        ----
            pars: tuple (A1, mu1, sigma1, fA, fm, sigma2) of the model
        Returns:
        -------
            chisq
        '''
        if pars is None:
            pars = self._get_parameters()[0]
        cdata = self.ydata[1:-1]
        weights = weights if weights is not None else self.weights[1:-1]
        SSR = 1 - np.sum(self.residuals(pars,weights)**2/np.std(cdata))
        SST = np.sum(weights*(cdata - np.mean(cdata))**2)
        rsq = 1 - SSR/SST
        return rsq
    
    def evaluate(self,pars=None,xdata=None):
        pars = pars if pars is not None else self._get_parameters()[0]
        xdata = xdata if xdata is not None else self.xdata
        return self.model(pars,xdata)

    def fit(self,xdata,ydata,error,p0=None,absolute_sigma=True, bounded=False,
            method=None, check_finite=True,full_output=False, **kwargs):
        ''' Performs the fitting of a Gaussian to the data. Acts as a wrapper 
        around the scipy.optimize `leastsq' function that minimizes the chisq 
        of the fit. 
        
        The function at each point is evaluated as an integral of the Gaussian 
        between the edges of the pixels (in case of wavelengths, boundary is 
        assumed to be in the midpoint between wavelength values). 
        
        The function calculates the fit parameters and the fit errors from the 
        covariance matrix provided by `leastsq'.
        
        Args:
        ----
            p0: tuple (amplitude, mean, sigma) with the initial guesses. 
                If None, is calculated from the data.
                
        Returns:
        -------
            pfit: tuple (amplitude, mean, sigma) of best fit parameters
            perror: tuple with errors on the best fit parameters
            
        Optional:
        --------
            return_full: Returns full output. Defaults to False
        '''
        
        def _unwrap_array_(array):
            try: 
                narray = array.values
            except:
                narray = array
            return narray
            
            
        self.xdata       = _unwrap_array_(xdata)
        self.xbounds     = (self.xdata[:-1]+self.xdata[1:])/2
        self.ydata       = _unwrap_array_(ydata)

        self.yerr        = _unwrap_array_(error)
        
        
        if p0 is None:
            p0 = self._initialize_parameters(xdata,ydata)
        p0 = np.atleast_1d(p0)
        n = p0.size  
        
        scale = kwargs.pop('scale',None)
        if scale is not None:
            assert scale in ['pixel','velocity']
        else:
            if np.min(self.xdata)>4900:
                scale = 'velocity'
            else:
                scale = 'pixel'
        self.scale = scale
        
        if bounded == True:
            try:
                bounds = self.bounds
            except:
                bounds = self._initialize_bounds()
                
        else:
            bounds=(-np.inf, np.inf)
        lb, ub = prepare_bounds(bounds, n)
        bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
        if method is None:
            if bounded_problem:
                method = 'trf'
            else:
                method = 'lm'
    
        if method == 'lm' and bounded_problem:
            raise ValueError("Method 'lm' only works for unconstrained problems. "
                             "Use 'trf' or 'dogbox' instead.")
        # NaNs can not be handled
        if check_finite:
            self.ydata = np.asarray_chkfinite(self.ydata)
        else:
            self.ydata = np.asarray(self.ydata)
    
        if isinstance(self.xdata, (list, tuple, np.ndarray)):
            # `xdata` is passed straight to the user-defined `f`, so allow
            # non-array_like `xdata`.
            if check_finite:
                self.xdata = np.asarray_chkfinite(self.xdata)
            else:
                self.xdata = np.asarray(self.xdata)
        if method != 'lm':
            #jac = '2-point'
            pass
        if method == 'lm':    
            #wrapped_jac = self._wrap_jac(self.jacobian,self.xdata,self.yerr)
            #
            res = leastsq(self.residuals,p0,Dfun=None,
                          full_output=True,col_deriv=False,**kwargs)
            pfit, pcov, infodict, errmsg, ier = res
            #print(errmsg)
            cost = np.sum(infodict['fvec']**2)
            if ier not in [1, 2, 3, 4]:
                #raise RuntimeError("Optimal parameters not found: " + errmsg)
                pfit = np.full_like(p0,np.nan)
                pcov = None
                success = False
            else:
                success = True
        else:
            #print('Bounded problem')
            res = least_squares(self.residuals, p0, jac=self.jacobian,
                                bounds=bounds, method=method,
                                **kwargs)
            if not res.success:
                #raise RuntimeError("Optimal parameters not found: " + res.message)
                pfit = np.full_like(p0,np.nan)
                pcov = None
                success = False
                cost = np.inf
            else:
                cost = 2 * res.cost  # res.cost is half sum of squares!
                pfit = res.x
            
                success = res.success
                # Do Moore-Penrose inverse discarding zero singular values.
                _, s, VT = svd(res.jac, full_matrices=False)
                threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
                s = s[s > threshold]
                VT = VT[:s.size]
                pcov = np.dot(VT.T / s**2, VT)
            #return_full = False    
        warn_cov = False
#        absolute_sigma=False
        dof  = (len(self.ydata) - len(pfit))
                
        if pcov is None:
            # indeterminate covariance
            pcov = np.zeros((len(pfit), len(pfit)), dtype=np.float)
            pcov.fill(np.inf)
            warn_cov = True         
        elif not absolute_sigma:
            if len(self.ydata) > len(pfit):
                
                #s_sq = cost / (self.ydata.size - pfit.size)
                s_sq = cost / dof
                pcov = pcov * s_sq
            else:
                pcov.fill(self.inf)
                warn_cov = True
        if warn_cov:
            warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning)    
        
        # From fit parameters, calculate the parameters of the two gaussians 
        # and corresponding errors
        if success == True:
            errors = [] 
            for i in range(len(pfit)):
                try:
                  errors.append(np.absolute(pcov[i][i])**0.5)
                except:
                  errors.append( 0.00 )
            self.pars = pfit
            self.errs = errors
        else:
            self.pars = np.full_like(pfit,np.nan)
            self.errs = np.full_like(pfit,np.nan)
            
            
        self.covar     = pcov
        self.rchi2     = cost / dof
        self.dof       = dof

        self.success = success
        self.cost = cost
        if full_output:
            return pfit, errors, infodict, errmsg, ier
        else:
            return pfit, errors
    
    def plot(self,fit=True,cofidence_intervals=True,ax=None,**kwargs):
        ''' Plots the line as a histogram of electron counts with corresponding
        errors. If `fit' flag is True, plots the result of the fitting 
        procedure.
        
        
        '''
        #import matplotlib.transforms as mtransforms
        if ax is None:
            fig = plt.figure(figsize=(9,9))
            if fit==True:
                ax1 = plt.subplot(211)
                ax2 = plt.subplot(212,sharex=ax1)
                ax  = [ax1,ax2]
            else:
                ax = [plt.subplot(111)]
            self.fig = fig
        elif type(ax) == plt.Axes:
            ax = [ax]
        elif type(ax) == list:
            pass
        self.ax_list  = [ax]
        widths = np.diff(self.xdata)[:-1]
        ax[0].bar(self.xdata[1:-1],self.ydata[1:-1],
                  widths,align='center',alpha=0.3,color='C0')
        ax[0].errorbar(self.xdata[1:-1],self.ydata[1:-1],
                       yerr=self.yerr[1:-1],fmt='o',color='C0')
        yeval = np.zeros_like(self.ydata)
        if fit is True:
            p,pe = self._get_parameters()
#            xeval = np.linspace(np.min(self.xdata),np.max(self.xdata),100)
            xeval = self.xdata[1:-1]
            yeval = self.evaluate(p)
           
            color = kwargs.pop('color','C1')
            label = kwargs.pop('label',None)
            ax[0].plot(xeval,yeval,color=color,marker='o',label=label)
            ax[0].set_ylabel('Flux [e-]')
            ax[1].axhline(0,ls='--',lw=0.5,c='k')
            ax[1].plot(xeval,(yeval-self.ydata[1:-1])/self.yerr[1:-1],
                      ls='',marker='o')
            ax[1].set_ylabel('Residuals [$\sigma$]')
            ax[1].set_xlabel('Pixel')
        if cofidence_intervals is True and fit is True:
#            xeval = self.xdata
            y,ylow,yhigh = self.confidence_band(confprob=0.05)
            ax[0].fill_between(xeval,ylow,yhigh,alpha=0.5,color='C1')
            y,ylow,yhigh = self.confidence_band(confprob=0.32)
            ax[0].fill_between(xeval,ylow,yhigh,alpha=0.2,
                              color='C1')
        ymax = np.max([1.2*np.percentile(yeval,95),1.2*np.max(self.ydata)])
        ax[0].set_ylim(-np.percentile(yeval,20),ymax)
        ax[0].set_xlabel('Pixel')
        ax[0].set_ylabel('Counts')
        ax[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        return
    def confidence_band(self, confprob=0.05, absolute_sigma=False):
        
        # https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html \
                            #confidence-and-prediction-intervals
        from scipy.stats import t
        # Given the confidence probability confprob = 100(1-alpha)
        # we derive for alpha: alpha = 1 - confprob
        alpha = 1.0 - confprob
        prb = 1.0 - alpha/2
        tval = t.ppf(prb, self.dof) #degrees of freedom
                    
        C = self.covar
       
        p,pe = self._get_parameters()
        x = self.xdata
        n = len(p)              # Number of parameters from covariance matrix
        N = len(x)
        if absolute_sigma:
            covscale = 1.0
        else:
            covscale = self.rchi2 * self.dof
          
        y = self.evaluate(p)
        dfdp = self.jacobian(p).T
        
        df2 = np.zeros(N-2)
        for j in range(n):
            for k in range(n):
                df2 += dfdp[j]*dfdp[k]*C[j,k]
#        df2 = np.dot(np.dot(dfdp.T,self.covar),dfdp).sum(axis=1)
        df = np.sqrt(covscale*df2)
        delta = tval * df
        upperband = y + delta
        lowerband = y - delta
        return y, upperband, lowerband       
    def calculate_center_uncertainty(self,pfit=None,covar=None,N=200):
        ''' 
        Returns the standard deviation of centres drawn from a random sample.
        
        Draws N samples by randomly sampling the provided fit parameteres and
        corresponding errors to construct N models of the line. Line centre 
        is calculated for each of the N models.         
        '''
        pfit = pfit if pfit is not None else self.params[0] 
        C = covar if covar is not None else self.covar
        # It is not possible to calculate center uncertainty if the covariance 
        # matrix contains infinite values
        if np.isinf(C).any() == True:
            return -1
        else:
            pass
        mdgN  = np.random.multivariate_normal(mean=pfit,cov=C,size=N)
#        cut   = np.where(mdgN[:,3]>0)[0]
#        mdgN  = mdgN[cut]
        centers = np.zeros(N)
        for i,pars in enumerate(mdgN):
            pgauss_i    = pars
            centers[i]  = self.calculate_center(pgauss_i)
        return centers.std()    
class SingleGaussian(EmissionLine):
    '''Single gaussian model of an emission line, with error function.'''
    def model(self,pars,xdata=None,separate=False):
        ''' Calculates the expected electron counts by assuming:
            (1) The PSF is a Gaussian function,
            (2) The number of photons falling on each pixel is equal to the 
                integral of the PSF between the pixel edges. (In the case of 
                wavelengths, the edges are calculated as midpoints between
                the wavelength of each pixel.)
        
        The integral of a Gaussian between two points, x1 and x2, is calculated
        as:
            
            Phi(x1,x2) = A * sigma * sqrt(pi/2) * [erf(t2) - erf(t1)]
        
        Where A and sigma are the amplitude and the variance of a Gaussian, 
        and 't' is defined as:
            
            t = (x - mu)/(sqrt(2) * sigma)
        
        Here, mu is the mean of the Gaussian.
        '''
        xdata = xdata if xdata is not None else self.xdata
        xb  = (xdata[1:]+xdata[:-1])/2
        A, mu, sigma = pars
        e1  = erf((xb[:-1]-mu)/(np.sqrt(2)*sigma))
        e2  = erf((xb[1:] -mu)/(np.sqrt(2)*sigma))
        y   = A*sigma*np.sqrt(np.pi/2)*(e2-e1)
        
        return y
    
        
    def _initialize_parameters(self,xdata,ydata):
        ''' Method to initialize parameters from data (pre-fit)
        Returns:
        ----
            p0: tuple with inital (amplitude, mean, sigma) values
        '''
        A0 = np.percentile(ydata,90)
        
        m0 = np.percentile(xdata,45)
        s0 = np.sqrt(np.var(xdata))/3
        p0 = (A0,m0,s0)
        self.initial_parameters = p0
        return p0
    def _initialize_bounds(self):
        ''' Method to initialize bounds from data (pre-fit)
        Returns:
        ----
            (lb,ub): tuple with bounds on the fitting parameters
        '''

        
        lb = (np.min(self.ydata), np.min(self.xdata), 0)
        ub = (np.max(self.ydata), np.max(self.xdata), self.sigmabound)
        self.bounds = (lb,ub)
        return (lb,ub)
    
    def jacobian(self,pars):
        '''
        Returns the Jacobian matrix of the __fitting__ function. 
        In the case x0 and weights are not provided, uses inital values.
        '''
        # Be careful not to put gaussian parameters instead of fit parameters!
        A, mu, sigma = pars
        x = self.xdata[1:-1]
        y = self.evaluate(pars) 
        dfdp = np.array([y/A,
                         -y*(x-mu)/(sigma**2),
                         y*(x-mu)**2/(sigma**3)]).T
#        return weights[:,None]*dfdp
        return dfdp
    def calculate_center(self,pgauss=None):
        '''
        Returns the x coordinate of the line center.
        Calculates the line center by solving for CDF(x_center)=0.5
        '''
        pgauss = pgauss if pgauss is not None else self._get_gauss_parameters()[0]
        A,m,s = pgauss
        
        def eq(x):
            cdf =  0.5*erfc((m-x)/(s*np.sqrt(2)))
            return  cdf - 0.5
        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
        return x

        
class SimpleGaussian(EmissionLine):
    '''Single gaussian model of an emission line, without error function'''
    def model(self,pars,separate=False):
        ''' Calculates the expected electron counts by assuming:
            (1) The PSF is a Gaussian function,
            (2) The number of photons falling on each pixel is equal to the 
                value of the Gaussian in the center of the pixel.
        
        The value of the Gaussian is calculated as:
            
            Phi(x) = A * exp(-1/2*((x-mu)/sigma)**2)
        
        Where A, mu, and sigma are the amplitude, mean, and the variance 
        of the Gaussian.
        '''
        x  = self.xdata
        A, mu, sigma = pars
        
        y   = A*np.exp(-0.5*(x-mu)**2/sigma**2)
        
        return y[1:-1]
        # return y
    def _fitpars_to_gausspars(self,pfit):
        '''
        Transforms fit parameteres into gaussian parameters.
        '''
        return pfit                
    def _initialize_parameters(self,xdata,ydata):
        ''' Method to initialize parameters from data (pre-fit)
        Returns:
        ----
            p0: tuple with inital (amplitude, mean, sigma) values
        '''
        A0 = np.percentile(ydata,90)
        
        m0 = np.percentile(xdata,45)
        s0 = np.sqrt(np.var(xdata))/3
        p0 = (A0,m0,s0)
        self.initial_parameters = p0
        return p0
    def _initialize_bounds(self):
        ''' Method to initialize bounds from data (pre-fit)
        Returns:
        ----
            (lb,ub): tuple with bounds on the fitting parameters
        '''

        
        lb = (np.min(self.ydata), np.min(self.xdata), 0)
        ub = (np.max(self.ydata), np.max(self.xdata), self.sigmabound)
        self.bounds = (lb,ub)
        return (lb,ub)
    
    def jacobian(self,pars,xdata=None,yerr=None):
        '''
        Returns the Jacobian matrix of the __fitting__ function. 
        In the case x0 and weights are not provided, uses inital values.
        '''
        # Be careful not to put gaussian parameters instead of fit parameters!
        A, mu, sigma = pars
        if xdata is not None:
            x = xdata
        else:
            x = self.xdata
        if yerr is not None:
            err = yerr
        else:
            err = self.yerr
        x = x[1:-1]
        err = err[1:-1]
        y = A*np.exp(-0.5*(x-mu)**2/sigma**2)
        
        dfdp = np.stack([y/A,
                         y*(x-mu)/(sigma**2),
                         y*(x-mu)**2/(sigma**3)
                         ],axis=1)
        return dfdp/-err[:,np.newaxis]
    def calculate_center(self,pgauss=None):
        '''
        Returns the x coordinate of the line center.
        Calculates the line center by solving for CDF(x_center)=0.5
        '''
        pgauss = pgauss if pgauss is not None else self._get_gauss_parameters()[0]
        A,m,s = pgauss
        
        def eq(x):
            cdf =  0.5*erfc((m-x)/(s*np.sqrt(2)))
            return  cdf - 0.5
        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
        return x    
    
def get_binlimits(xarray,center,scale):
    if scale[:3]=='pix':
        dx = np.array([-5,-2.5,2.5,5]) # units pix
        binlims = dx + center
    elif scale[:3]=='vel':
        # varray = (xarray-center)/center * 299792.458 # units km/s
        dv     = np.array([-4,-2,2,4]) # units km/s
        binlims = center * (1 + dv/299792.458) # units wavelength
    return binlims

def assign_weights(xarray,center,scale):
    def f(x,x1,x2): 
        # a linear function going through x1 and x2
        return np.abs((x-x1)/(x2-x1))
    
    weights  = np.zeros_like(xarray,dtype=np.float64)
    binlims = get_binlimits(xarray, center, scale)
        
    idx      = np.digitize(xarray,binlims)
    cut1     = np.where(idx==2)[0]
    cutl     = np.where(idx==1)[0]
    cutr     = np.where(idx==3)[0]
    # ---------------
    # weights are:
    #  = 1,           -2.5<=x<=2.5
    #  = 0,           -5.0>=x & x>=5.0
    #  = linear[0-1]  -5.0<x<-2.5 & 2.5>x>5.0 
    # ---------------
    weights[cutl] = f(xarray[cutl],binlims[0],binlims[1])
    weights[cutr] = f(xarray[cutr],binlims[3],binlims[2])
    weights[cut1] = 1
    return weights    
        