#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:07:32 2018

@author: dmilakov
"""

from harps.core import odr, np, os, plt, curve_fit, json, interpolate, leastsq

from harps.constants import c

import harps.settings as hs
import harps.emissionline as emline
import harps.containers as container
import harps.functions as hf
import harps.gaps as hg
import warnings

quiet = hs.quiet
version = hs.version
#==============================================================================
# Assumption: Frequencies are known with 1MHz accuracy
freq_err = 2e4


#==============================================================================
    
#                           G A P S    F I L E   
    
#==============================================================================
#def read_gaps(filepath=None):
#    if filepath is not None:
#        filepath = filepath  
#    else:
#        dirpath = hs.get_dirname('gaps')
#        filepath = os.path.join(dirpath,'gaps.json')
#    with open(filepath,'r') as json_file:
#        gaps_file = json.load(json_file)
#    gaps = []
#    for block in range(1,4):
#        orders  = gaps_file['orders{}'.format(block)]
#        norders = len(orders)
#        block_gaps = container.gaps(norders)
#        block_gaps['order'] = orders
#        block_gaps['gaps']  = gaps_file['block{}'.format(block)]
#        gaps.append(block_gaps)
#    gaps = np.hstack(gaps)
#    return np.sort(gaps)
#    
#    
#
#def get_gaps(order,filepath=None):
#    gapsfile  = read_gaps(filepath)
#    orders   = np.array(gapsfile[:,0],dtype='i4')
#    gaps2d   = np.array(gapsfile[:,1:],dtype='f8')
#    selection = np.where(orders==order)[0]
#    gaps1d    = gaps2d[selection]
#    return np.ravel(gaps1d)
#
#
#def introduce_gaps(centers,gaps1d,npix=4096):
#    if np.size(gaps1d)==0:
#        return centers
#    elif np.size(gaps1d)==1:
#        gap  = gaps1d
#        gaps = np.full((7,),gap)
#    else:
#        gaps = gaps1d
#    centc = np.copy(centers)
#    
#    for i,gap in enumerate(gaps):
#        ll = (i+1)*npix/(np.size(gaps)+1)
#        cut = np.where(centc>ll)[0]
#        centc[cut] = centc[cut]-gap
#    return centc
#==============================================================================
#
#                         L I N E      F I T T I N G                  
#
#==============================================================================
default_line = 'SingleGaussian'
def gauss(x,flux,bkg,error,model=default_line,output_model=False,
          *args,**kwargs):
    assert np.size(x)==np.size(flux)==np.size(bkg)
    line_model   = getattr(emline,model)
    line         = line_model()    
    try:
        pars, errors = line.fit(x,flux-bkg,error,bounded=False)
        chisqnu      = line.rchi2
        chisq        = line.cost
        model = line.evaluate(pars)
        success = True
    except:
#        plt.figure()
#        plt.plot(x,flux-bkg)
#        plt.plot(x,error)
        pars   = np.full(3,np.nan)
        errors = np.full(3,np.nan)
        chisq   = np.nan
        chisqnu = np.nan
        model  = np.full_like(flux,np.nan)
        success = False
    if output_model:
        
        return success, pars, errors, chisq, chiqnu, model
    else:
        return success, pars, errors, chisq, chisqnu
def assign_weights(pixels):
        weights  = np.zeros_like(pixels)
        binlims  = [-5,-2.5,2.5,5]
        idx      = np.digitize(pixels,binlims)
        cut1     = np.where(idx==2)[0]
        cutl     = np.where(idx==1)[0]
        cutr     = np.where(idx==3)[0]
        # ---------------
        # weights are:
        #  = 1,           -2.5<=x<=2.5
        #  = 0,           -5.0>=x & x>=5.0
        #  = linear[0-1]  -5.0<x<2.5 & 2.5>x>5.0 
        # ---------------
        weights[cut1] = 1
        weights[cutl] = 0.4*(5+pixels[cutl])
        weights[cutr] = 0.4*(5-pixels[cutr])
        return weights
def lsf(pix,flux,background,error,lsf1s,p0,
        output_model=False,*args,**kwargs):
    """
    lsf1d must be an instance of LSF class and contain only one segment 
    (see harps.lsf)
    """
    def residuals(x0,lsf1s):
        # flux, center
        amp, sft, s = x0
        sftpix   = pix-sft
        model    = lsf_model(lsf1s,x0,pix)#amp * interpolate.splev(sftpix,splr)
        weights  = np.ones_like(pix)
#        weights  = assign_weights(sftpix)
        resid = np.sqrt(weights) * ((flux-background) - model) / error
        #resid = line_w * (counts- model)
        return resid
    
    amp0,sft0,s0 = p0
    popt,pcov,infodict,errmsg,ier = leastsq(residuals,x0=p0,
                                        args=(lsf1s,),ftol=1e-10,
                                        full_output=True)
    if ier not in [1, 2, 3, 4]:
        print("Optimal parameters not found: " + errmsg)
        popt = np.full_like(p0,np.nan)
        pcov = None
        success = False
    else:
        success = True
    if success:
        
        amp, cen, wid = popt
        cost = np.sum(infodict['fvec']**2)
        dof  = (len(pix) - len(popt))
        if pcov is not None:
            pcov = pcov*cost/dof
        else:
            pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
        #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
    else:
        popt = np.full_like(p0,np.nan)
        pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
        cost = np.nan
        dof  = (len(pix) - len(popt))
        success=False
    pars    = popt
    errors  = np.sqrt(np.diag(pcov))
    chisqnu = cost/dof
    
    #pars[0]*interpolate.splev(pix+pars[1],splr)+background
#    plt.figure()
#    plt.title('fit.lsf')
#    plt.plot(pix,flux)
#    plt.plot(pix,model)
    if output_model:  
        model   = lsf_model(lsf1s,pars,pix)
        return success, pars, errors, cost, chisqnu, model
    else:
        return success, pars, errors, cost, chisqnu
def lsf_model(lsf1s,pars,pix):
    """
    Returns the model of the data from the LSF and parameters provided. 
    Does not include the background.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    amp, cen, wid = pars
    wid   = np.abs(wid)
    x     = lsf1s.x * wid
    y     = lsf1s.y / np.max(lsf1s.y)
    splr  = interpolate.splrep(x,y)
    model = amp*interpolate.splev(pix-cen,splr)
    return model
#==============================================================================
#
#        W A V E L E N G T H     D I S P E R S I O N      F I T T I N G                  
#
#==============================================================================
def poly(centers,wavelengths,cerror,werror,polyord):
    numcen = np.size(centers)
    assert numcen>polyord, "No. centers too low, {}".format(numcen)
    # beta0 is the initial guess
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=np.RankWarning)
        beta0 = np.flip(np.polyfit(centers,wavelengths,polyord,cov=False))
    data  = odr.RealData(centers,wavelengths,sx=cerror,sy=werror)
    model = odr.polynomial(order=polyord)
    ODR   = odr.ODR(data,model,beta0=beta0)
    out   = ODR.run()
    #pars  = out.beta
    #errs  = out.sd_beta
    #ssq   = out.sum_square
    return out #pars, errs, ssq
def segment(centers,wavelengths,cerror,werror,polyord,plot=False):
    """
    Fits a polynomial to the provided data and errors.
    Uses scipy's Orthogonal distance regression package in order to take into
    account the errors in both x and y directions.
    
    Returns:
    -------
        coef : len(polyord) array
        errs : len(polyord) array
    """
    numcen = np.size(centers)
    if numcen>polyord:
        pass
    else:
        pars    = np.full(polyord+1,np.nan)
        errs    = np.full(polyord+1,np.inf)
        chisq   = -1
        chisqnu = -1
        return pars, errs, chisq, chisqnu
    arenan = np.isnan(centers)
    centers     = centers[~arenan]/4095
    wavelengths = wavelengths[~arenan]
    cerror      = cerror[~arenan]/4095
    werror      = werror[~arenan]
#    if plot:
#        plt.figure()
#        plt.errorbar(centers,wavelengths,yerr=werror,xerr=cerror,ms=2,ls='',capsize=4)
#        [plt.axvline(512*i,ls='--',lw=0.3,c='k') for i in range(9)]
    # clip0: points kept in previous iteration
    clip0 = np.full_like(centers,False,dtype='bool')
    # clip1: points kept in this iteration
    clip1 = np.full_like(centers,True,dtype='bool')
    j = 0
    # iterate maximum 10 times
    while not np.sum(clip0)==np.sum(clip1) and j<10:
        j+=1
        
        clip0        = clip1
        centers0     = centers[clip0]
        wavelengths0 = wavelengths[clip0]
        cerror0      = cerror[clip0]
        werror0      = werror[clip0]
        try:
            model        = poly(centers0,wavelengths0,cerror0,werror0,polyord)   
            pars         = model.beta
            errs         = model.sd_beta
            chisqnu      = model.res_var
            nu           = len(wavelengths0) - len(pars)
            chisq        = chisqnu * nu
            residuals    = wavelengths-np.polyval(np.flip(pars),centers)
        except:
            pars         = np.full(polyord+1,np.nan)
            errs         = np.full(polyord+1,np.inf)
            chisqnu      = np.inf
            chisq        = np.inf
            residuals    = np.full_like(centers,np.nan)
            
        
        #derivpars    = (np.arange(len(pars))*pars)[1:]
        #errors       = np.polyval(np.flip(derivpars),centers)*cerror0
        
        # clip 5 sigma outliers from the residuals and check condition
        outliers     = hf.is_outlier(residuals)
        clip1        = ~outliers
        
        if plot:# and np.sum(outliers)>0:
            
#            plt.figure()
            #plt.plot(centers[clip1],np.polyval(pars[::-1],centers[clip1]))
            plt.scatter(centers,residuals,s=2)
            plt.scatter(centers[outliers],residuals[outliers],
                        s=16,marker='x',c='k')
    return pars, errs, chisq, chisqnu

def dispersion(linelist,version,fittype='gauss',f=1):
    """
    Fits the wavelength solution to the data provided in the linelist.
    Calls 'wavesol1d' for all orders in linedict.
    
    Uses Gaussian profiles as default input.
    
    Returns:
    -------
        wavesol2d : dictionary with coefficients for each order in linelist
        
    """
    orders  = np.unique(linelist['order'])
    polyord, gaps, do_segment = hf.version_to_pgs(version)
    disperlist = []
    if gaps:
        gaps2d     = hg.read_gaps(None)
    plot=False
    if plot and gaps:
        plt.figure()
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
    linelist0 = hf.remove_bad_fits(linelist,fittype)
    for i,order in enumerate(orders):
        linelis1d = linelist0[np.where(linelist0['order']==order)]
        # rescale the centers by the highest pixel number (4095)
        centers1d = linelis1d[fittype][:,1]
        cerrors1d = f*linelis1d['{fit}_err'.format(fit=fittype)][:,1]
        wavelen1d = hf.freq_to_lambda(linelis1d['freq'])
        werrors1d = 1e10*(c/((linelis1d['freq'])**2)) * freq_err
        if gaps:
            if plot:
                centersold = centers1d
            cut       = np.where(gaps2d['order']==order)
            gaps1d    = gaps2d[cut]['gaps'][0]
            centers1d = hg.introduce_gaps(centers1d,gaps1d)
            if plot:
                plt.scatter(centersold,centers1d-centersold,s=2,c=[colors[i]])
        else:
            pass
        di1d      = dispersion1d(centers1d,wavelen1d,
                              cerrors1d,werrors1d,
                              version)
        di1d['order'] = order
        disperlist.append(di1d)
    dispersion2d = np.hstack(disperlist)
    return dispersion2d
        
def dispersion1d(centers,wavelengths,cerror,werror,version):
    """
    Uses 'segment' to fit polynomials of degree given by polyord keyword.
    
    
    If version=xx1, divides the data into 8 segments, each 512 pix wide. 
    A separate polyonomial solution is derived for each segment.
    """
    polyord, gaps, do_segment = hf.version_to_pgs(version)
    if do_segment==True:
        numsegs = 8
    else:
        numsegs = 1
            
    npix = 4096
    # remove NaN
    #centers     = hf.removenan(centers)
    #wavelengths = hf.removenan(wavelengths)
    #cerror      = hf.removenan(cerror)
    #werror      = hf.removenan(werror)
    seglims  = np.linspace(npix//numsegs,npix,numsegs)
    binned   = np.digitize(centers,seglims)
    # new container
    coeffs = container.coeffs(polyord,numsegs)
    for i in range(numsegs):
        sel = np.where(binned==i)[0]
        output = segment(centers[sel],wavelengths[sel],
                               cerror[sel],werror[sel],polyord)
        pars, errs, chisq, chisqnu = output
        p = len(pars)
        n = len(sel)
        # not enough points for the fit
        if (n-p-1)<1:
            continue
        else:
            pass
        coeffs[i]['pixl']   = seglims[i]-npix//numsegs
        coeffs[i]['pixr']   = seglims[i]
        coeffs[i]['pars']   = pars
        coeffs[i]['errs']   = errs
        coeffs[i]['chisq']  = chisq
        coeffs[i]['chisqnu']= chisqnu
        coeffs[i]['npts']   = n
        coeffs[i]['aicc']   = chisq + 2*p + 2*p*(p+1)/(n-p-1)
    return coeffs
    


# =============================================================================
    
#                              S  P  L  I  N  E
    
# =============================================================================
# https://stackoverflow.com/questions/51321100/python-natural-smoothing-splines

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
# 

def spline(*args,**kwargs):
    return get_natural_cubic_spline_model(*args,**kwargs)

def get_natural_cubic_spline_model(x, y, minval=None, maxval=None, n_knots=None, knots=None):
    """
    Get a natural cubic spline model for the data.

    For the knots, give (a) `knots` (as an array) or (b) minval, maxval and n_knots.

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    x: np.array of float
        The input data
    y: np.array of float
        The outpur data
    minval: float 
        Minimum of interval containing the knots.
    maxval: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.

    Returns
    --------
    model: a model object
        The returned model will have following method:
        - predict(x):
            x is a numpy array. This will return the predicted y-values.
    """

    if knots:
        spline = NaturalCubicSpline(knots=knots)
    else:
        maxval = maxval if maxval is not None else np.max(x)
        minval = minval if minval is not None else np.min(x)
        spline = NaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)

    p = Pipeline([
        ('nat_cubic', spline),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    p.fit(x, y)

    return p


class AbstractSpline(BaseEstimator, TransformerMixin):
    """Base class for all spline basis expansions."""

    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
        if knots is None:
            if not n_knots:
                n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self


class NaturalCubicSpline(AbstractSpline):
    """Apply a natural cubic basis expansion to an array.
    The features created with this basis expansion can be used to fit a
    piecewise cubic function under the constraint that the fitted curve is
    linear *outside* the range of the knots..  The fitted curve is continuously
    differentiable to the second order at all of the knots.
    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots.
      - By specifying the cutpoints directly.  

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.
    Parameters
    ----------
    min: float 
        Minimum of interval containing the knots.
    max: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.
    """

    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = np.atleast_1d(X).squeeze()
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError: # For arrays with only one element
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            def ppart(t): return np.maximum(0, t)

            def cube(t): return t*t*t
            numerator = (cube(ppart(x - self.knots[knot_idx]))
                         - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
        return X_spl