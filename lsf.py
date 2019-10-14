#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
import harps.settings as hs
import harps.io as io
import harps.containers as container
import harps.plotter as hplot
import harps.fit as hfit
from harps.core import os, np, plt, FITS

import errno

from scipy import interpolate
from scipy.optimize import leastsq, brentq, curve_fit
import scipy.stats as stats
# =============================================================================
#    
#                        I N P U T  / O U T P U T
#    
# =============================================================================
    
def read_lsf(fibre,specifier,version=-1):
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
    filename ='LSF_{fibre}_{shape}.fits'.format(fibre=fibre,shape=shape)
    hdu = FITS(os.path.join(hs.dirnames['lsf'],filename))
    lsf = hdu[-1].read()
    return LSF(lsf)

def from_file(filepath):
    hdu = FITS(filepath)
    lsf = hdu[-1].read()
    return LSF(lsf)
# =============================================================================
#    
#                        L S F    M O D E L L I N G
#    
# =============================================================================

class LSFModeller(object):
    def __init__(self,outfile,sOrder,eOrder,iter_solve=2,iter_center=5,
                 segnum=16,numpix=20,subnum=4,method='analytic'):
        self._outfile = outfile
        self._cache = {}
        self._iter_solve  = iter_solve
        self._iter_center = iter_center
        self._segnum  = segnum
        self._numpix  = numpix
        self._subnum  = subnum
        self._sOrder  = sOrder
        self._eOrder  = eOrder
        self._orders  = np.arange(sOrder,eOrder)
        self._method  = method
        self.iters_done = 0
    def __getitem__(self,extension):
        try:
            data = self._cache[extension]
        except:
            self._read_from_file()
            #self._cache.update({extension:data})
            data = self._cache[extension]
        return data
    def __setitem__(self,extension,data):
        self._cache.update({extension:data})
    def _read_from_file(self,start=None,stop=None,step=None,**kwargs):
        extensions = ['linelist','flux','background','error']
        data, numfiles = io.mread_outfile(self._outfile,extensions,501,
                                start=start,stop=stop,step=step)
        self._cache.update(data)
        self.numfiles = numfiles
        return
    
    def __call__(self):
        """ Returns the LSF in an numpy array  """
        
        fluxes      = self['flux']
        backgrounds = self['background']
        errors      = self['error']
        fittype     = 'lsf'
        for i in range(self._iter_solve):
            linelists = self['linelist']
            if i == 0:
                fittype = 'gauss'
            pix3d, flx3d, orders = stack(fittype,linelists,fluxes,backgrounds,
                                         self._orders)
            lsf_i    = construct_lsf(pix3d,flx3d,self._orders,
                                     numseg=self._segnum,
                                     numpix=self._numpix,
                                     subpix=self._subnum,
                                     numiter=self._iter_center,
                                     method=self._method)
            self._lsf_i = lsf_i
            setattr(self,'lsf_{}'.format(i),lsf_i)
            if i < self._iter_solve-1:
                linelists_i = solve(lsf_i,linelists,fluxes,backgrounds,
                                    errors,fittype,self._method)
                self['linelist'] = linelists_i
            self.iters_done += 1
        lsf_final = lsf_i
        self._lsf_final = lsf_final
        
        return lsf_final
    def stack(self,fittype='lsf'):
        fluxes      = self['flux']
        backgrounds = self['background']
        linelists   = self['linelist']
        
        return stack(fittype,linelists,fluxes,backgrounds,self._orders)
        
    def save(self,filepath,version=None,overwrite=False):
        with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
            hdu.write(self._lsf_final.values,extname='LSF',extver=version)
        hdu.close()
        print("File saved to {}".format(filepath))
        return
        

def stack(fittype,linelists,fluxes,backgrounds=None,orders=None):
    numex = np.shape(linelists)[0]
    pix3d = np.zeros((72,4096,numex))
    flx3d = np.zeros((72,4096,numex))
        
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
            lineflux = fluxes[exp,od,pixl:pixr]
            if backgrounds is not None:
                lineflux = lineflux - backgrounds[exp,od,pixl:pixr]
            
            # move to frame centered at 0 
            pix1l = np.arange(line['pixl'],line['pixr']) - line[fittype][1]
            # normalise the flux
            flx1l = lineflux/np.sum(lineflux)
#            if (od==51)and(segment==2):
#                plt.plot(pix1l,flx1l,ls='',marker='o',ms=2)
            pix3d[od,pixl:pixr,exp] = pix1l
            flx3d[od,pixl:pixr,exp] = flx1l

    return pix3d,flx3d,orders


def construct_lsf(pix3d, flx3d, orders, method,
                  numseg=16,numpix=10,subpix=4,numiter=5,**kwargs):
    lst = []
    for i,od in enumerate(orders):
        plot=False
        lsf1d=(construct_lsf1d(pix3d[od],flx3d[od],method,numseg,numpix,
                               subpix,numiter,plot=plot,**kwargs))
        lsf1d['order'] = od
        lst.append(lsf1d)
        if len(orders)>1:
            hf.update_progress((i+1)/len(orders),'Fit LSF')
    lsf = np.hstack(lst)
    
    return LSF(lsf)
def construct_lsf1d(pix2d,flx2d,method,numseg=16,numpix=10,subpix=4,
                    numiter=5,minpix=0,maxpix=4096,minpts=50,
                    plot=False,plot_res=False,
                    **kwargs):
    """ Input: single order output of stack_lines_multispec"""

    seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
    totpix  = 2*numpix*subpix+1
    
    pixcens = np.linspace(-numpix,numpix,totpix)
#    pixlims = (pixcens+0.5/subpix)
    # lsf for the entire order, divided into segments
    if method == 'analytic':
        lsf1d = get_empty_lsf(method,numseg,20)
    elif method == 'spline':
        lsf1d = get_empty_lsf(method,numseg,totpix,pixcens)
    count = 0
    for i in range(len(lsf1d)):
        
        pixl = seglims[i]
        pixr = seglims[i+1]
        # save pixl and pixr
#        lsf1s['pixl'] = pixl
#        lsf1s['pixr'] = pixr
        pix1s = np.ravel(pix2d[pixl:pixr])
        flx1s = np.ravel(flx2d[pixl:pixr])
        if plot == True:
            lsf1d[i],plotter = construct_lsf1s(pix1s,flx1s,method,numiter,numpix,
                                               subpix,minpts,plot=plot,
                                               plot_residuals=plot_res,
                                               **kwargs)
        else:
            lsf1d[i] = construct_lsf1s(pix1s,flx1s,method,numiter,numpix,subpix,minpts,
                                       plot=False,plot_residuals=False,
                                       **kwargs)
        lsf1d[i]['pixl'] = pixl
        lsf1d[i]['pixr'] = pixr
        lsf1d[i]['segm'] = i

    return lsf1d
def get_empty_lsf(method,numsegs,n=None,pixcens=None):
    assert method in ['analytic','spline']
    if method == 'analytic':
        n     = n if n is not None else 20
        lsf_cont = container.lsf_analytic(numsegs,n)
    elif method == 'spline':
        n     = n if n is not None else 160
        lsf_cont = container.lsf(numsegs,n)
        lsf_cont['x'] = pixcens
    return lsf_cont
def clean_input(pix1s,flx1s):
    pix1s = np.ravel(pix1s)
    flx1s = np.ravel(flx1s)
    # remove infinites, nans and zeros
    finite  = np.logical_and(np.isfinite(flx1s),flx1s!=0)
    numpts = np.size(flx1s)
    diff  = numpts-np.sum(finite)
    print("{0:5d}/{1:5d} ({2:5.2%}) points removed".format(diff,numpts,diff/numpts))
    return pix1s[finite],flx1s[finite]
def construct_lsf1s(pix1s,flx1s,method,numiter=5,numpix=10,subpix=4,minpts=50,
                    plot=False,plot_residuals=False,**kwargs):
    '''
    Constructs the LSF model for a single segment
    '''
    
    ## plotting keywords
    plot_subpix_grid = kwargs.pop('plot_subpix',False)
    plot_model       = kwargs.pop('plot_model',True)
    rasterized       = kwargs.pop('rasterized',False)
    ## useful things
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    if method == 'analytic':
        lsf1s = get_empty_lsf(method,1,20)[0]
    elif method == 'spline':
        lsf1s = get_empty_lsf(method,1,totpix,pixcens)[0]
    pix1s, flx1s = clean_input(pix1s,flx1s)
    
    # save the total number of points used
    lsf1s['numlines'] = len(pix1s)
    
    if plot and plot_residuals:
        plotter=hplot.Figure2(2,1,figsize=(8,6),height_ratios=[2,1])
        ax = [plotter.add_subplot(0,1,0,1),plotter.add_subplot(1,2,0,1)]
    elif plot and not plot_residuals:
        plotter=hplot.Figure2(1,1,figsize=(8,6))
        ax = [plotter.add_subplot(0,1,0,1)]
        #ax[0].plot(pix1s,flx1s,ms=0.3,alpha=0.2,marker='o',ls='')
        
    shift = 0
    totshift = 0
    for j in range(numiter):
        # shift the values along x-axis for improved centering
        pix1s = pix1s+shift  
        
        if method == 'spline':
            shift_method = kwargs.pop('shift_method',2)
            # get current model of the LSF
            splr = interpolate.splrep(lsf1s['x'],lsf1s['y'])                    
            sple = interpolate.splev(pix1s,splr)
            # calculate residuals to the model
            rsd  = (flx1s-sple)
            # calculate mean of residuals for each pixel comprising the LSF
            means  = bin_means(pix1s,rsd,pixlims,minpts)
            lsf1s['y'] = lsf1s['y']+means
            
            if shift_method==1:
                shift = shift_anderson(lsf1s['x'],lsf1s['y'])
            elif shift_method==2:
                shift = shift_zeroder(lsf1s['x'],lsf1s['y'])
        elif method == 'analytic':
            p0=(1,5)+20*(0.1,)
            popt,pcov=curve_fit(hf.gaussP,pix1s,flx1s,p0=p0)
            xx = np.linspace(-8,8,500)
            yy,centers,sigma = hf.gaussP(xx,*popt,return_center=True,return_sigma=True)
            shift = -hf.derivative_zero(xx,yy,-1,1)
            rsd = flx1s - hf.gaussP(pix1s,*popt)
#            lsf1s['mu']  = centers
#            lsf1s['sig'] = np.hstack([popt[0],np.repeat(sigma,len(popt)-2)])
#            lsf1s['amp'] = popt[1:]
            lsf1s['pars'] = popt
            lsf1s['errs'] = np.square(np.diag(pcov))
        print("iter {0:2d} shift {1:12.6f}".format(j,shift))
        totshift += shift
        #count        +=1
    print('total shift {0:12.6f}'.format(totshift))   
    if plot:
        ax[0].scatter(pix1s,flx1s,s=4,alpha=0.2,marker='o',c='C0',
          rasterized=rasterized)
        ax[0].set_ylim(-0.05,0.35)
        ax[-1].set_xlabel("Distance from center [pix]")
        for a in ax:
            a.set_xlim(-11,11)
        if plot_model:
            if method=='spline':
                ax[0].scatter(lsf1s['x'],lsf1s['y'],marker='s',s=32,
                          linewidths=0.2,edgecolors='k',c='C1',zorder=1000)
            elif method == 'analytic':
                ax[0].plot(xx,yy,lw=2,c='C1',zorder=1000)
        if plot_residuals:
            ax[1].scatter(pix1s,rsd,s=1)
            if method=='spline':
                ax[1].errorbar(pixcens,means,ls='',
                          xerr=0.5/subpix,ms=4,marker='s')
        if plot_subpix_grid:
            for a in ax:
                [a.axvline(lim,ls=':',lw=0.4,color='k') for lim in pixlims]
                
        return lsf1s, plotter
    else:
        return lsf1s
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
        means[i] = np.nanmedian(y1)
    # go back and interpolate means for empty bins
    idy   = hf.find_missing(idx)
    # interpolate if no points in the bin, but only pixels -5 to 5
    if len(idy)>0:
        idy = np.atleast_1d(idy)
        means[idy] = interpolate_bins(means,xbins[idy],kind)
    
    return means
def interpolate_local(lsf,order,center):
    assert np.isfinite(center)==True, "Center not finite, {}".format(center)
    values  = lsf[order].values
    assert len(values)>0, "No LSF model for order {}".format(order)
    numseg,totpix  = np.shape(values['x'])
    
    segcens = (values['pixl']+values['pixr'])/2
    segcens[0]  = 0
    segcens[-1] = 4096
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
            cent = line[fittype][1]
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
    def save(self,filepath,version=None,overwrite=False):
        with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
            hdu.write(self.values,extname='LSF',extver=version)
        hdu.close()
        print("File saved to {}".format(filepath))
        return
    def plot(self,title=None,saveto=None,plotter=None):
        values = self.values
        if plotter is not None:
            plotter = plotter
        else:
            plotter = hplot.Figure2(1,1)
        figure, ax = plotter.fig, plotter.add_subplot(0,1,0,1)
        nitems = len(values.shape)
        npts   = values['y'].shape[-1]
        x = np.linspace(np.min(values['x']),np.max(values['x']),3*npts)
        if nitems>0:
            numvals = len(values)
            colors = plt.cm.jet(np.linspace(0,1,numvals))
            for j,item in enumerate(values):
                splr = interpolate.splrep(item['x'],item['y'])                    
                sple = interpolate.splev(x,splr)
                ax.scatter(item['x'],item['y'],edgecolor='None',
                                c=[colors[j]])
                ax.plot(x,sple,lw=0.6,c=colors[j])
        else:            
            splr = interpolate.splrep(values['x'],values['y'])                    
            sple = interpolate.splev(x,splr)
            ax.scatter(values['x'],values['y'],edgecolor='None')
            ax.plot(x,sple,lw=0.6)
        ax.set_ylim(-0.03,0.35)
        if title:
            ax.set_title(title)
        if saveto:
            figure.savefig(saveto)
        return plotter
    def interpolate(self,order,center):
        return interpolate_local(self,order,center)