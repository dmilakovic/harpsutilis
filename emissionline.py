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
        obsdata = self.ydata[1:-1]
        resids  = ((obsdata - self.model(pars))/self.yerr[1:-1])
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
        
        if bounded == True:
            if self.bounds is None:
                bounds = self._initialize_bounds()
            else:
                bounds = self.bounds
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
        pfit = pfit if pfit is not None else self._get_fit_parameters()[0]    
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
        centers = np.zeros(mdgN.size)
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
#class DoubleGaussian(EmissionLine):
#    def model(self,pars):
#        ''' Calculates the expected electron counts by assuming:
#            (1) The PSF is a Gaussian function,
#            (2) The number of photons falling on each pixel is equal to the 
#                integral of the PSF between the pixel edges. (In the case of 
#                wavelengths, the edges are calculated as midpoints between
#                the wavelength of each pixel.)
#        
#        The integral of a Gaussian between two points, x1 and x2, is calculated
#        as:
#            
#            Phi(x1,x2) = A * sigma * sqrt(pi/2) * [erf(t2) - erf(t1)]
#        
#        Where A and sigma are the amplitude and the variance of a Gaussian, 
#        and 't' is defined as:
#            
#            t = (x - mu)/(sqrt(2) * sigma)
#        
#        Here, mu is the mean of the Gaussian.
#        '''
#        xb  = self.xbounds
#        #A1,mu1,sigma1,fA,fm,sigma2 = pars
#        #A2  = A1*fA
#        #mu2 = mu1 + fm*np.max([sigma1,sigma2])
#        #A1,mu1,sigma1,A2,mu2,sigma2 = self._fitpars_to_gausspars(pars)
#        A1,mu1,sigma1,A2,mu2,sigma2 = pars
#        
#        e11  = erf((xb[:-1]-mu1)/(np.sqrt(2)*sigma1))
#        e21  = erf((xb[1:] -mu1)/(np.sqrt(2)*sigma1))
#        y1   = A1*sigma1*np.sqrt(np.pi/2)*(e21-e11)
#        
#        e12  = erf((xb[:-1]-mu2)/(np.sqrt(2)*sigma2))
#        e22  = erf((xb[1:] -mu2)/(np.sqrt(2)*sigma2))
#        y2   = A2*sigma2*np.sqrt(np.pi/2)*(e22-e12)
#        
#        return y1+y2
#    
#    def _initialize_parameters(self):
#        ''' Method to initialize parameters from data (pre-fit)
#        Returns:
#        ----
#            p0: tuple with inital (amplitude, mean, sigma) values
#        '''
#        A0 = np.percentile(self.ydata,90)
#        
#        m0 = np.percentile(self.xdata,50)
#        s0 = np.sqrt(np.var(self.xdata))/3
#        p0 = (A0,m0,s0,A0,m0,s0)
#        self.initial_parameters = p0
#        return p0
#    def _initialize_bounds(self):
#        ''' Method to initialize bounds from data (pre-fit)
#        Returns:
#        ----
#            (lb,ub): tuple with bounds on the fitting parameters
#        '''
#        # ORIGINAL CONSTRAINTS
#        lb = (np.min(self.ydata), np.min(self.xdata), 0,
#              0, -3, 0)
#        ub = (np.max(self.ydata), np.max(self.xdata), self.sigmabound,
#              1, 3, self.sigmabound)
#        
#        # NO CONSTRAINTS
##        lb = (0., -np.inf, 0,         0, -np.inf, 0)
##        ub = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf )
#        
##        GASPARE'S CONSTRAINTS
##        lb = (0, -np.inf, 0,
##              0, -3, 0)
##        ub = (np.inf,  np.inf, np.inf,
##              1, 3, np.inf)
#        #  CONSTRAINTS
##        lb = (np.min(self.ydata), -np.inf, 0,         0, -np.inf, 0)
##        ub = (np.max(self.ydata), np.inf, np.inf, np.inf, np.inf, np.inf )
#        
##        lb = (0,-np.inf,0, 0,-np.inf,0)
##        ub = (np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)
#        
#        return (lb,ub)
#    def jacobian(self,pars,x0=None,weights=None):
#        '''
#        Returns the Jacobian matrix of the __fitting__ function. 
#        In the case x0 and weights are not provided, uses inital values.
#        '''
#        # Be careful not to put gaussian parameters instead of fit parameters!
#        A1, mu1, sigma1, A2, mu2, sigma2 = pars
#        
#        weights = weights[1:-1] if weights is not None else self.weights[1:-1]
#        if x0 is None:
#            x = self.xdata#[1:-1]
#            #y = self.ydata[1:-1]
#        else:
#            x = x0#[1:-1]
#        y1,y2 = self.evaluate(pars,x,separate=True) 
#        #y = A * np.exp(-1/2*((x-mu)/sigma)**2) 
#        x = x[1:-1]
#        dfdp = np.array([y1/A1,
#                         y1*(x-mu1)/(sigma1**2),
#                         y1*(x-mu1)**2/(sigma1**3),
#                         y2/A1,
#                         y2*(x-mu2)/(sigma2**2),
#                         y2*(x-mu2)**2/(sigma2**3)]).T
#        return weights[:,None]*dfdp
#    def calculate_center(self,pars=None):
#        '''
#        Returns the x coordinate of the line center.
#        Calculates the line center by solving for CDF(x_center)=0.5
#        '''
#        pars = pars if pars is not None else self._get_gauss_parameters()[0]
#        A1,m1,s1,A2,m2,s2 = pars
#        print(pars)
#        def eq(x):
#            cdf =  0.5*erfc((m1-x)/(s1*np.sqrt(2))) + \
#                  0.5*erfc((m2-x)/(s2*np.sqrt(2)))
#            return  cdf/2 - 0.5
#        print(eq(np.min(self.xdata)),eq(np.max(self.xdata)))
#        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
#        return x 
#class SimpleGaussian(DoubleGaussian):
#    def model(self,pars):
#        ''' Calculates the expected electron counts by assuming:
#            (1) The PSF is a sum of two Gaussian functions,
#            (2) The number of photons falling on each pixel is equal to the 
#                sum of the values of the two Gaussians, evaluated in the 
#                center of the pixel. 
#        
#        '''
#        x  = self.xdata[1:-1]
##        A1,mu1,sigma1,fA,fm,sigma2 = pars
#        A1,mu1,sigma1,A2,mu2,sigma2 = pars
#        
#        y1   = A1*np.exp(-0.5*(x-mu1)**2/sigma1**2)
#        y2   = A2*np.exp(-0.5*(x-mu2)**2/sigma2**2)
#                
#        return y1+y2
#
#    def evaluate(self,pars,x=None,separate=False,ptype='gauss'):
#        ''' Returns the evaluated Gaussian function along the provided `x' and 
#        for the provided Gaussian parameters `p'. 
#        
#        Args:
#        ---- 
#            x: 1d array along which to evaluate the Gaussian. Defaults to xdata
#            p: tuple (amplitude, mean, sigma) of Gaussian parameters. 
#               Defaults to the fit parameter values.
#        '''
#        if x is None:
#            x = self.xdata[1:-1]
#            xb = self.xbounds
#        else:
#            x = x[1:-1]
#            xb = (x[:-1]+x[1:])/2
#        p = pars if pars is not None else self._get_gauss_parameters()[0]
#        
##        g1 = A1 * np.exp(-1/2*((x-mu1)/sigma1)**2)[1:-1]
##        g2 = A2 * np.exp(-1/2*((x-mu2)/sigma2)**2)[1:-1]
#        pi = np.reshape(p,(-1,3))
#        N    = pi.shape[0]
#        Y    = []
#        for i in range(N):
#            A, mu, sigma = pi[i]
#            y    = A*np.exp(-0.5*(x-mu)**2/sigma**2)
#            Y.append(y)
#        
#        
#        if separate:
#            return tuple(Y)
#        else:
#            return np.sum(Y,axis=0)
#    def calculate_center(self,pgauss=None):
#        '''
#        Returns the x coordinate of the line center.
#        Calculates the line center by solving for CDF(x_center)=0.5
#        '''
#        pgauss = pgauss if pgauss is not None else self._get_gauss_parameters()[0]
#        A1,m1,s1,A2,m2,s2 = pgauss
##        print(pgauss)
#        def eq(x):
#            cdf =  0.5*erfc((m1-x)/(s1*np.sqrt(2))) + \
#                  0.5*erfc((m2-x)/(s2*np.sqrt(2)))
#            return  cdf - 1.5
#        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
#        return x 
#class SpectralLine2(object):
#    ''' Class with functions to fit LFC lines as pure Gaussians'''
#    def __init__(self,xdata,ydata,kind='emission',yerr=None, weights=None,
#                 absolute_sigma=True):
#        ''' Initialize the object using measured data
#        
#        Args:
#        ----
#            xdata: 1d array of x-axis data (pixels or wavelengts)
#            ydata: 1d array of y-axis data (electron counts)
#            weights:  1d array of weights calculated using Bouchy method
#            kind: 'emission' or 'absorption'
#        '''
#        def _unwrap_array_(array):
#            if type(array)==pd.Series:
#                narray = array.values
#            elif type(array)==np.ndarray:
#                narray = array
#            return narray
#            
#            
#        self.xdata   = _unwrap_array_(xdata)
#        self.xbounds = (self.xdata[:-1]+self.xdata[1:])/2
#        self.ydata   = _unwrap_array_(ydata)
#        self.kind    = kind
#        yerr         = yerr if yerr is not None else np.sqrt(np.abs(self.ydata))
#        weights      = weights if weights is not None else yerr #np.ones_like(xdata)
#        self.yerr    = _unwrap_array_(yerr)
#        self.weights = _unwrap_array_(weights)
#        
#        
#        self.success = False   
#    def _fitpars_to_gausspars(self,pfit):
#        '''
#        Transforms fit parameteres into gaussian parameters.
#        '''
#        A1, m1, s1, fA, fm, s2 = pfit
#        A2 = fA*A1
#        D  = np.max([s1,s2])
#        m2 = m1 + fm*D    
#        return (A1,m1,s1,A2,m2,s2)                
#    def _initialize_parameters(self):
#        ''' Method to initialize parameters from data (pre-fit)
#        Returns:
#        ----
#            p0: tuple with inital (amplitude, mean, sigma) values
#        '''
#        A0 = np.percentile(self.ydata,90)
#        
#        m0 = np.percentile(self.xdata,45)
#        D  = np.mean(np.diff(self.xdata))
#        s0 = np.sqrt(np.var(self.xdata))/3
#        p0 = (A0,m0,s0,0.9,D,s0)
#        self.initial_parameters = p0
#        return p0
#    def _initialize_bounds(self):
#        ''' Method to initialize bounds from data (pre-fit)
#        Returns:
#        ----
#            (lb,ub): tuple with bounds on the fitting parameters
#        '''
#        std = np.std(self.xdata)#/3
#        if self.kind == 'emission':
#            Amin = 0.
#            Amax = np.max(self.ydata)
#        elif self.kind == 'absorption':
#            Amin = np.min(self.ydata)
#            Amax = 0.
#        #peak = peakdetect(self.ydata,self.xdata,lookahead=2,delta=0)[0][0][0]
#        lb = (Amin, np.min(self.xdata), 0,
#              0, -2, 0)
#        ub = (Amax, np.max(self.xdata), std,
#              1, 2, std)
#        self.bounds = (lb,ub)
#        return (lb,ub)
#    def _get_fit_parameters(self):
#        ''' Method to check whether the fit has been successfully performed.
#        If the fit was successful, returns the fit values. Otherwise, the 
#        method performs the fitting procedure and returns the fit values.
#        Returns:
#        -------
#            pfit: tuple with fitted (amplitude, mean, sigma) values
#        '''
#        if self.success == True:
#            pfit = self.fit_parameters
#            errors = self.fit_errors
#        else:
#            p0 = self._initialize_parameters()
#            pars, errors = self.fit(p0)
#            pfit = self.fit_parameters
#            errors = self.fit_errors
#        return pfit, errors
#    def _get_gauss_parameters(self):
#        ''' Method to check whether the fit has been successfully performed.
#        If the fit was successful, returns the fit values. Otherwise, the 
#        method performs the fitting procedure and returns the fit values.
#        Returns:
#        -------
#            pfit: tuple with fitted (amplitude, mean, sigma) values
#        '''
#        if self.success == True:
#            pars = self.gauss_parameters
#            errors = self.gauss_errors
#        else:
#            p0 = self._initialize_parameters()
#            pars, errors = self.fit(p0)
#        return pars, errors
#    
#    def residuals(self,pars,weights=None):
#        ''' Returns the residuals of individual data points to the model.
#        Args:
#        ----
#            pars: tuple (amplitude, mean, sigma) of the model
#        Returns:
#        -------
#             1d array (len = len(xdata)) of residuals
#        '''
##        model = self.model(*pars)
#        cdata = self.ydata[1:-1]
#        weights = weights if weights is not None else self.weights[1:-1]
#        return weights * (self.model(*pars) - cdata)
#    def chisq(self,pars,weights=None):
#        ''' Returns the chi-square of data points to the model.
#        Args:
#        ----
#            pars: tuple (amplitude, mean, sigma) of the model
#        Returns:
#        -------
#            chisq
#        '''
#        return (self.residuals(pars)**2).sum()
#    def R2(self,pars=None,weights=None):
#        ''' Returns the R^2 estimator of goodness of fit to the model.
#        Args:
#        ----
#            pars: tuple (A1, mu1, sigma1, fA, fm, sigma2) of the model
#        Returns:
#        -------
#            chisq
#        '''
#        if pars is None:
#            pars = self._get_fit_parameters()[0]
#        cdata = self.ydata[1:-1]
#        weights = weights if weights is not None else self.weights[1:-1]
#        SSR = 1 - np.sum(self.residuals(pars,weights)**2/np.std(cdata))
#        SST = np.sum(weights*(cdata - np.mean(cdata))**2)
#        rsq = 1 - SSR/SST
#        return rsq
#    def log_prior(self,pars=None):
#        if pars is None:
#            pars = self._get_fit_parameters()[0]
#        A1,m1,s1,A2,m2,s2 = pars
#        xmin = np.min(self.xdata)
#        xmax = np.max(self.xdata)
#        D = max([s1,s2])
#        if ((s1<0) or (s2<0) or (A1<0) or (A2<0) or 
#            (m1<xmin) or (m1>xmax) or (m2<xmin) or (m2>xmax)):
#            return -np.inf # log(0)
#        else:
#            return - np.log(s1) - np.log(s2) - np.log(A1) - np.log(A2) 
#    def log_likelihood(self,pars=None):
#        if pars is None:
#            pars = self._get_fit_parameters()[0]
#        A1,m1,s1,A2,m2,s2 = pars
#        y_model = self.model(theta,x)
#        return np.sum(-0.5*np.log(2*np.pi*y_model) - (y[1:-1]-y_model)**2 / (2*y_model))
#    def log_posterior(theta,x,y):
#        lnprior = log_prior(theta,x)
#        if lnprior == -np.inf:
#            return -np.inf
#        else:
#            return lnprior + log_likelihood(theta,x,y)
#    def evaluate(self,p,x=None,separate=False,ptype='gauss'):
#        ''' Returns the evaluated Gaussian function along the provided `x' and 
#        for the provided Gaussian parameters `p'. 
#        
#        Args:
#        ---- 
#            x: 1d array along which to evaluate the Gaussian. Defaults to xdata
#            p: tuple (amplitude, mean, sigma) of Gaussian parameters. 
#               Defaults to the fit parameter values.
#        '''
#        if x is None:
#            x = self.xdata
#            xb = self.xbounds
#        else:
#            x = x
#            xb = (x[:-1]+x[1:])/2
#        if ptype=='gauss':
#            A1, mu1, sigma1, A2, mu2, sigma2 = p if p is not None else self._get_gauss_parameters()
#        elif ptype=='fit':
#            A1, mu1, sigma1, fA, fm, sigma2 = p if p is not None else self._get_fit_parameters()
#            A1, mu1, sigma1, A2, mu2, sigma2 = self._fitpars_to_gausspars(p)
##        g1 = A1 * np.exp(-1/2*((x-mu1)/sigma1)**2)[1:-1]
##        g2 = A2 * np.exp(-1/2*((x-mu2)/sigma2)**2)[1:-1]
#        
#        e11  = erf((xb[:-1]-mu1)/(np.sqrt(2)*sigma1))
#        e21  = erf((xb[1:] -mu1)/(np.sqrt(2)*sigma1))
#        y1   = A1*sigma1*np.sqrt(np.pi/2)*(e21-e11)
#        
#        e12  = erf((xb[:-1]-mu2)/(np.sqrt(2)*sigma2))
#        e22  = erf((xb[1:] -mu2)/(np.sqrt(2)*sigma2))
#        y2   = A2*sigma2*np.sqrt(np.pi/2)*(e22-e12)
#        
#        
#        if separate:
#            return y1,y2
#        else:
#            return y1+y2
#    def jacobian(self,fitpars,x0=None,weights=None):
#        '''
#        Returns the Jacobian matrix of the __fitting__ function. 
#        In the case x0 and weights are not provided, uses inital values.
#        '''
#        # Be careful not to put gaussian parameters instead of fit parameters!
#        A1, mu1, sigma1, fA, fm, sigma2 = fitpars
#        D   = np.max([sigma1,sigma2])
#        mu2 = mu1 + D*fm
#        weights = weights[1:-1] if weights is not None else self.weights[1:-1]
#        if x0 is None:
#            x = self.xdata#[1:-1]
#            #y = self.ydata[1:-1]
#        else:
#            x = x0#[1:-1]
#        y1,y2 = self.evaluate(fitpars,x,separate=True,ptype='fit') 
#        #y = A * np.exp(-1/2*((x-mu)/sigma)**2) 
#        x = x[1:-1]
#        dfdp = np.array([y1/A1 + y2/A1,
#                         y1*(x-mu1)/(sigma1**2) + y2*(x-mu2)/(sigma2**2),
#                         y1*(x-mu1)**2/(sigma1**3),
#                         y2/fA,
#                         y2*(x-mu2)/(sigma2**2)*D,
#                         y2*(x-mu2)**2/(sigma2**3)]).T
#        return weights[:,None]*dfdp
#    def hessian(self,p,x=None):
#        A, mu, sigma = p
#        x = x if x is not None else self.xdata[1:-1]
#        y = self.evaluate(p,x) if x is not None else self.ydata[1:-1]
#        N = len(x)
#        n = len(p)
#        hes = np.zeros((n,n,N))
#        hes[0,0] = 0
#        hes[0,1] = y/A*(x-mu)/sigma**2
#        hes[0,2] = y/A*(x-mu)**2/sigma**3
#        hes[1,0] = hes[0,1]
#        hes[1,1] = y*((x-mu)**2/sigma**4 - 1/sigma**2)
#        hes[1,2] = y*((x-mu)**3/sigma**5 - 2*(x-mu)/sigma**3)
#        hes[2,0] = hes[0,2]
#        hes[2,1] = hes[1,2]
#        hes[2,2] = y*((x-mu)**4/sigma**6 - 3*(x-mu)**2/sigma**4)
#        self.hess = hes
#        return hes
#    def model(self,A1,mu1,sigma1,fA,fm,sigma2):
#        ''' Calculates the expected electron counts by assuming:
#            (1) The PSF is a Gaussian function,
#            (2) The number of photons falling on each pixel is equal to the 
#                integral of the PSF between the pixel edges. (In the case of 
#                wavelengths, the edges are calculated as midpoints between
#                the wavelength of each pixel.)
#        
#        The integral of a Gaussian between two points, x1 and x2, is calculated
#        as:
#            
#            Phi(x1,x2) = A * sigma * sqrt(pi/2) * [erf(t2) - erf(t1)]
#        
#        Where A and sigma are the amplitude and the variance of a Gaussian, 
#        and 't' is defined as:
#            
#            t = (x - mu)/(sqrt(2) * sigma)
#        
#        Here, mu is the mean of the Gaussian.
#        '''
#        xb  = self.xbounds
#        
#        A2  = A1*fA
#        mu2 = mu1 + fm*np.max([sigma1,sigma2])
#        
#        e11  = erf((xb[:-1]-mu1)/(np.sqrt(2)*sigma1))
#        e21  = erf((xb[1:] -mu1)/(np.sqrt(2)*sigma1))
#        y1   = A1*sigma1*np.sqrt(np.pi/2)*(e21-e11)
#        
#        e12  = erf((xb[:-1]-mu2)/(np.sqrt(2)*sigma2))
#        e22  = erf((xb[1:] -mu2)/(np.sqrt(2)*sigma2))
#        y2   = A2*sigma2*np.sqrt(np.pi/2)*(e22-e12)
#        
#        return y1+y2
#    def fit(self,p0=None,absolute_sigma=True, bounded=True,
#            method=None, check_finite=True, **kwargs):
#        ''' Performs the fitting of a Gaussian to the data. Acts as a wrapper 
#        around the scipy.optimize `leastsq' function that minimizes the chisq 
#        of the fit. 
#        
#        The function at each point is evaluated as an integral of the Gaussian 
#        between the edges of the pixels (in case of wavelengths, boundary is 
#        assumed to be in the midpoint between wavelength values). 
#        
#        The function calculates the fit parameters and the fit errors from the 
#        covariance matrix provided by `leastsq'.
#        
#        Args:
#        ----
#            p0: tuple (amplitude, mean, sigma) with the initial guesses. 
#                If None, is calculated from the data.
#                
#        Returns:
#        -------
#            pfit: tuple (amplitude, mean, sigma) of best fit parameters
#            pcov: covariance matrix between the best fit parameters
#            
#        Optional:
#        --------
#            return_full: Returns full output. Defaults to False
#        '''
#        
#        if p0 is None:
#            p0 = self._initialize_parameters()
#        p0 = np.atleast_1d(p0)
#        n = p0.size  
#        
#        if bounded == True:
#            bounds = self._initialize_bounds()
#        else:
#            bounds=(-np.inf, np.inf)
#        lb, ub = prepare_bounds(bounds, n)
#        bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
#        if method is None:
#            if bounded_problem:
#                method = 'trf'
#            else:
#                method = 'lm'
#    
#        if method == 'lm' and bounded_problem:
#            raise ValueError("Method 'lm' only works for unconstrained problems. "
#                             "Use 'trf' or 'dogbox' instead.")
#        # NaNs can not be handled
#        if check_finite:
#            self.ydata = np.asarray_chkfinite(self.ydata)
#        else:
#            self.ydata = np.asarray(self.ydata)
#    
#        if isinstance(self.xdata, (list, tuple, np.ndarray)):
#            # `xdata` is passed straight to the user-defined `f`, so allow
#            # non-array_like `xdata`.
#            if check_finite:
#                self.xdata = np.asarray_chkfinite(self.xdata)
#            else:
#                self.xdata = np.asarray(self.xdata)
#        
#        if method != 'lm':
#            jac = '2-point'
#        if method == 'lm':    
#            return_full = kwargs.pop('full_output', False)
##            wrapped_jac = self._wrap_jac()
#            res = leastsq(self.residuals,p0,Dfun=None,full_output=1)#,col_deriv=True,**kwargs)
#            pfit, pcov, infodict, errmsg, ier = res
#            cost = np.sum(infodict['fvec']**2)
#            if ier not in [1, 2, 3, 4]:
#                #raise RuntimeError("Optimal parameters not found: " + errmsg)
#                pfit = np.full_like(p0,np.nan)
#                pcov = None
#                success = False
#            else:
#                success = True
#        else:
#            res = least_squares(self.residuals, p0, jac=self.jacobian, bounds=bounds, method=method,
#                                **kwargs)
#            if not res.success:
#                #raise RuntimeError("Optimal parameters not found: " + res.message)
#                pfit = np.full_like(p0,np.nan)
#                pcov = None
#                success = False
#                cost = np.inf
#            else:
#                cost = 2 * res.cost  # res.cost is half sum of squares!
#                pfit = res.x
#            
#                success = res.success
#                # Do Moore-Penrose inverse discarding zero singular values.
#                _, s, VT = svd(res.jac, full_matrices=False)
#                threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
#                s = s[s > threshold]
#                VT = VT[:s.size]
#                pcov = np.dot(VT.T / s**2, VT)
#            return_full = False    
#        warn_cov = False
##        absolute_sigma=False
#        dof  = (len(self.ydata) - len(pfit))
#        
#
#                
#        if pcov is None:
#            # indeterminate covariance
#            pcov = np.zeros((len(pfit), len(pfit)), dtype=np.float)
#            pcov.fill(np.inf)
#            warn_cov = True         
#        elif not absolute_sigma:
#            if len(self.ydata) > len(pfit):
#                
#                #s_sq = cost / (self.ydata.size - pfit.size)
#                s_sq = cost / dof
#                pcov = pcov * s_sq
#            else:
#                pcov.fill(self.inf)
#                warn_cov = True
#        if warn_cov:
#            warnings.warn('Covariance of the parameters could not be estimated',
#                      category=OptimizeWarning)    
#        error_fit_pars = [] 
#        for i in range(len(pfit)):
#            try:
#              error_fit_pars.append(np.absolute(pcov[i][i])**0.5)
#            except:
#              error_fit_pars.append( 0.00 )
#        
#        # From fit parameters, calculate the parameters of the two gaussians 
#        # and corresponding errors
#        if success == True:
#            A1, m1, s1, fA, fm, s2 = pfit
#            
#            A2 = fA*A1
#            D  = max([s1,s2])
#            m2 = m1 + fm*D
#            
#            error_A1, error_m1, error_s1, error_fA, error_fm, error_s2 = error_fit_pars
#            
#            error_A2 = np.sqrt((A2/A1*error_A1)**2 +  (A2/fA*error_fA)**2)
#            if D == s1:
#                error_D = error_s1
#            elif D==s2:
#                error_D = error_s2
#            error_m2 = np.sqrt(error_m1**2 + error_D**2)
#            
#            # Make the component with the smaller mean to be m1 and the  
#            # component with the larger mean to be m2. (i.e. m1<m2)
#            
#            if m1<m2:
#                gp_c1 = A1, m1, s1
#                gp_c2 = A2, m2, s2
#                gp_c1error = error_A1, error_m1, error_s1
#                gp_c2error = error_A2, error_m2, error_s2
#                
#                
#            elif m1>m2:
#                gp_c1 = A2, m2, s2
#                gp_c2 = A1, m1, s1
#                gp_c1error = error_A2, error_m2, error_s2  
#                gp_c2error = error_A1, error_m1, error_s1            
#            else:
#                print("m1=m2 ?", m1==m2)
#            gauss_parameters = np.array([*gp_c1,*gp_c2])
#            gauss_errors     = np.array([*gp_c1error,*gp_c2error])
#            fit_parameters   = pfit
#            fit_errors       = error_fit_pars
#            
#        else:
#            gauss_parameters = np.full_like(pfit,np.nan)
#            gauss_errors     = np.full_like(pfit,np.nan)
#            fit_parameters   = pfit
#            fit_errors       = error_fit_pars
#            
#        self.covar     = pcov
#        self.rchi2     = cost / dof
#        self.dof       = dof
#        
#        
#        self.gauss_parameters = gauss_parameters
#        self.gauss_errors     = gauss_errors
#        
#        
#        self.fit_parameters   = fit_parameters
#        self.fit_errors       = fit_errors
#        
#        self.center           = self.calculate_center(gauss_parameters)
##        self.center_error     = self.calculate_center_uncertainty(pfit,pcov)
#        self.center_mass      = np.sum(self.weights*self.xdata*self.ydata)/np.sum(self.weights*self.ydata)
#        
#        
#        
##        self.infodict = infodict
##        self.errmsg = errmsg
#        self.success = success
#        self.cost = cost
#        if return_full:
#            return gauss_parameters, gauss_errors, infodict, errmsg, ier
#        else:
#            return gauss_parameters, gauss_errors
#    def calculate_center(self,pgauss=None):
#        '''
#        Returns the x coordinate of the line center.
#        Calculates the line center by solving for CDF(x_center)=0.5
#        '''
#        pgauss = pgauss if pgauss is not None else self._get_gauss_parameters()[0]
#        A1,m1,s1,A2,m2,s2 = pgauss
#        
#        def eq(x):
#            cdf =  0.5*erfc((m1-x)/(s1*np.sqrt(2))) + \
#                  0.5*erfc((m2-x)/(s2*np.sqrt(2)))
#            return  cdf/2 - 0.5
#        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
#        return x
#    def calculate_center_uncertainty(self,pfit=None,covar=None,N=200):
#        ''' 
#        Returns the standard deviation of centres drawn from a random sample.
#        
#        Draws N samples by randomly sampling the provided fit parameteres and
#        corresponding errors to construct N models of the line. Line centre 
#        is calculated for each of the N models.         
#        '''
#        pfit = pfit if pfit is not None else self._get_fit_parameters()[0]    
#        C = covar if covar is not None else self.covar
#        # It is not possible to calculate center uncertainty if the covariance 
#        # matrix contains infinite values
#        if np.isinf(C).any() == True:
#            return -1
#        else:
#            pass
#        mdgN  = np.random.multivariate_normal(mean=pfit,cov=C,size=N)
#        cut   = np.where(mdgN[:,3]>0)[0]
#        mdgN  = mdgN[cut]
#        centers = np.zeros(cut.size)
#        for i,pars in enumerate(mdgN):
#            pgauss_i    = self._fitpars_to_gausspars(pars)
#            centers[i]  = self.calculate_center(pgauss_i)
#        return centers.std()
#    def calculate_photon_noise(self):
#        '''INCORRECT'''
#        deriv   = derivative1d(self.ydata,self.xdata)
#        weights = pd.Series(deriv**2*829**2/self.ydata)
#        weights = weights.replace([np.inf,-np.inf],np.nan)
#        weights = weights.dropna()
#        return 1./np.sqrt(weights.sum())*299792458e0
#    def plot(self,fit=True,cofidence_intervals=True,ax=None,**kwargs):
#        ''' Plots the line as a histogram of electron counts with corresponding
#        errors. If `fit' flag is True, plots the result of the fitting 
#        procedure.
#        
#        
#        '''
#        import matplotlib.transforms as mtransforms
#        if ax is None:
#            fig,ax = get_fig_axes(1,figsize=(9,9),bottom=0.12,left=0.15,**kwargs)
#            self.fig = fig
#        self.ax_list  = ax
#        widths = np.diff(self.xdata)[:-1]
#        ax[0].bar(self.xdata[1:-1],self.ydata[1:-1],
#                  widths,align='center',alpha=0.3,color='#1f77b4')
#        ax[0].errorbar(self.xdata[1:-1],self.ydata[1:-1],
#                       yerr=self.yerr[1:-1],fmt='o',color='#1f77b4')
#        yeval = np.zeros_like(self.ydata)
#        if fit is True:
#            p,pe = self._get_gauss_parameters()
##            xeval = np.linspace(np.min(self.xdata),np.max(self.xdata),100)
#            xeval = self.xdata
#            y1,y2 = self.evaluate(p,xeval,True,ptype='gauss')
#            yeval = y1+y2
#            fit = True
#            xeval = xeval[1:-1]
#            ax[0].plot(xeval,yeval,color='#ff7f0e',marker='o')
#            ax[0].plot(xeval,y1,color='#2ca02c',lw=0.7,ls='--')
#            ax[0].plot(xeval,y2,color='#2ca02c',lw=0.7,ls='--')
#            A1, m1, s1, A2, m2, s2 = p
#            ax[0].plot([m1,m1], [0,A1],ls='--',lw=0.7,color='#2ca02c')
#            ax[0].plot([m2,m2], [0,A2],ls='--',lw=0.7,color='#2ca02c')
#              
#            # calculate the center of the line and the 1-sigma uncertainty
##            cenx = self.center
##            ceny = self.evaluate(p,np.array([m1,cenx,m2]),ptype='gauss')
##            ax[0].plot([cenx,cenx],[0,ceny[1]],ls='--',lw=1,c='C1')
##            cend = self.center_error
#            
#            # shade the area around the center of line (1-sigma uncertainty)
##            xcenval = np.linspace(cenx-cend,cenx+cend,100)
##            ycenval = self.evaluate(p,xcenval,ptype='gauss')
##            ax[0].fill_between(xcenval,0,ycenval,color='C1',alpha=0.4,
##              where=((xcenval>=cenx-cend)&(xcenval<=cenx+cend)))
#        if cofidence_intervals is True and fit is True:
#            xeval = self.xdata
#            y,ylow,yhigh = self.confidence_band(xeval,confprob=0.05)
#            ax[0].fill_between(xeval[1:-1],ylow,yhigh,alpha=0.5,color='#ff7f0e')
#            y,ylow,yhigh = self.confidence_band(xeval,confprob=0.32)
#            ax[0].fill_between(xeval[1:-1],ylow,yhigh,alpha=0.2,
#                              color='#ff7f0e')
#        ymax = np.max([1.2*np.percentile(yeval,95),1.2*np.max(self.ydata)])
#        ax[0].set_ylim(-np.percentile(yeval,20),ymax)
#        ax[0].set_xlabel('Pixel')
#        ax[0].set_ylabel('Counts')
#        ax[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
#        return
#    def confidence_band(self, x, confprob=0.05, absolute_sigma=False):
#        
#        # https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html \
#                            #confidence-and-prediction-intervals
#        from scipy.stats import t
#        # Given the confidence probability confprob = 100(1-alpha)
#        # we derive for alpha: alpha = 1 - confprob
#        alpha = 1.0 - confprob
#        prb = 1.0 - alpha/2
#        tval = t.ppf(prb, self.dof) #degrees of freedom
#                    
#        C = self.covar
#       
#        p,pe = self._get_fit_parameters()
#        n = len(p)              # Number of parameters from covariance matrix
#        N = len(x)
#        if absolute_sigma:
#            covscale = 1.0
#        else:
#            covscale = self.rchi2 * self.dof
#          
#        y = self.evaluate(p,x,ptype='fit')
#        
#        # If the x array is larger than xdata, provide new weights
#        # for all points in x by linear interpolation
#        int_function = interpolate.interp1d(self.xdata,self.weights)
#        weights      = int_function(x)
#        dfdp = self.jacobian(p,x,weights).T
#        
#        df2 = np.zeros(N-2)
#        for j in range(n):
#            for k in range(n):
#                df2 += dfdp[j]*dfdp[k]*C[j,k]
##        df2 = np.dot(np.dot(dfdp.T,self.covar),dfdp).sum(axis=1)
#        df = np.sqrt(covscale*df2)
#        delta = tval * df
#        upperband = y + delta
#        lowerband = y - delta
#        return y, upperband, lowerband       
        