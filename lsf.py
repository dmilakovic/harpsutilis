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
from scipy.optimize import leastsq
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
    def __init__(self,outfile,sOrder,eOrder,numiter=2,segnum=16,numpix=20,subnum=4):
        self._outfile = outfile
        self._cache = {}
        self._numiter = numiter
        self._segnum  = segnum
        self._numpix  = numpix
        self._subnum  = subnum
        self._sOrder  = sOrder
        self._eOrder  = eOrder
        self._orders  = np.arange(sOrder,eOrder)
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
        for i in range(self._numiter):
            linelists = self['linelist']
            if i == 0:
                fittype = 'gauss'
            pix3d, flx3d, orders = stack(fittype,linelists,fluxes,backgrounds,
                                         self._orders)
            lsf_i    = construct_lsf(pix3d,flx3d,self._orders,
                                     numseg=self._segnum,
                                     numpix=self._numpix,
                                     subpix=self._subnum,
                                     numiter=self._numiter)
            self._lsf_i = lsf_i
            setattr(self,'lsf_{}'.format(i),lsf_i)
            if i < self._numiter-1:
                linelists_i = solve(lsf_i,linelists,fluxes,backgrounds,
                                    errors,fittype)
                self['linelist'] = linelists_i
            self.iters_done += 1
        lsf_final = lsf_i
        self._lsf_final = lsf_final
        return lsf_final
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


def construct_lsf(pix3d, flx3d, orders,
                  numseg=16,numpix=10,subpix=4,numiter=5,**kwargs):
    lst = []
    for i,od in enumerate(orders):
        
        plot=False
        lsf1d=(construct_lsf1d(pix3d[od],flx3d[od],numseg,numpix,
                               subpix,numiter,plot=plot,**kwargs))
        lsf1d['order'] = od
        lst.append(lsf1d)
        if len(orders)>1:
            hf.update_progress((i+1)/len(orders),'Fit LSF')
    lsf = np.hstack(lst)
    
    return LSF(lsf)
def construct_lsf1d(pix2d,flx2d,numseg=16,numpix=10,subpix=4,
                    numiter=5,minpix=0,maxpix=4096,**kwargs):
    """ Input: single order output of stack_lines_multispec"""
    do_plot    = kwargs.pop('plot',False)
    seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
    totpix  = 2*numpix*subpix+1
    
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    # lsf for the entire order, divided into segments
    lsf1d      = container.lsf(numseg,totpix)
    lsf1d['x'] = pixcens
    count = 0
    for i,lsf1s in enumerate(lsf1d):
        pixl = seglims[i]
        pixr = seglims[i+1]
        # save pixl and pixr
        lsf1s['pixl'] = pixl
        lsf1s['pixr'] = pixr
        pix1s = np.ravel(pix2d[pixl:pixr])
        flx1s = np.ravel(flx2d[pixl:pixr])
        numlines = np.size(flx1s)
        lsf1s['numlines'] = np.size(flx1s)
        # remove infinites, nans and zeros
        finite  = np.logical_and(np.isfinite(flx1s),flx1s!=0)
        diff  = numlines-np.sum(finite)
        #print("{0:5d}/{1:5d} ({2:5.2%}) points removed".format(diff,numlines,diff/numlines))
        flx1s = flx1s[finite]
        pix1s = pix1s[finite]
        
        if np.size(flx1s)>0 and np.size(pix1s)>0:
            pass
        else:
            continue
        
        shift  = 0
        totshift = 0
        if do_plot:
            plotter=hplot.Figure(2,figsize=(9,6),sharex=True,ratios=[1,1])
            ax = plotter.axes
            #ax[0].scatter(pix1s,flx1s,s=2)
        for j in range(numiter):
            # shift the values along x-axis for improved centering
            pix1s = pix1s+shift  
            # get current model of the LSF
            splr = interpolate.splrep(lsf1s['x'],lsf1s['y'])                    
            sple = interpolate.splev(pix1s,splr)
            # calculate residuals to the model
            rsd  = (flx1s-sple)
            # calculate mean of residuals for each pixel comprising the LSF
            means  = bin_means(pix1s,rsd,pixlims)
            lsf1s['y'] = lsf1s['y']+means
            
            # calculate derivative
            deriv = hf.derivative1d(lsf1s['y'],lsf1s['x'])
            lsf1s['dydx'] = deriv
            
            left  = np.where(lsf1s['x']==-0.5)[0]
            right = np.where(lsf1s['x']==0.5)[0]
            elsf_neg     = lsf1s['y'][left]
            elsf_pos     = lsf1s['y'][right]
            elsf_der_neg = lsf1s['dydx'][left]
            elsf_der_pos = lsf1s['dydx'][right]
            shift        = float((elsf_pos-elsf_neg)/(elsf_der_pos-elsf_der_neg))
            totshift    += shift
            print("segm {0:2d} iter {1:2d} shift {2:12.6f} ({3:12.6f} cumul)".format(i,j,shift,totshift))
            count        +=1
            
            if do_plot:
                ax[0].plot(pix1s,flx1s,ms=0.3,alpha=0.2,marker='o',ls='')
                ax[0].scatter(lsf1s['x'],lsf1s['y'],marker='s',s=16,
                              linewidths=0.2,edgecolors='k')
                ax[1].scatter(pix1s,rsd,s=1)
                ax[1].errorbar(pixcens,means,ls='',
                          xerr=0.5/subpix,ms=4,marker='s')
                for a in ax:
                    a.vlines(pixlims,0,0.35,linestyles=':',lw=0.4,colors='k')
    return lsf1d
def construct_lsf1s(pix1s,flx1s,numiter=5,numpix=10,subpix=4,plot=False,
                    plot_residuals=False):
    totpix  = 2*numpix*subpix+1
    pixcens = np.linspace(-numpix,numpix,totpix)
    pixlims = (pixcens+0.5/subpix)
    lsf1s = container.lsf(1,totpix)[0]
    lsf1s['x'] = pixcens
    
    pix1s = np.ravel(pix1s)
    flx1s = np.ravel(flx1s)
    # remove infinites, nans and zeros
    finite  = np.logical_and(np.isfinite(flx1s),flx1s!=0)
    numpts = np.size(flx1s)
    diff  = numpts-np.sum(finite)
    print("{0:5d}/{1:5d} ({2:5.2%}) points removed".format(diff,numpts,diff/numpts))
    flx1s = flx1s[finite]
    pix1s = pix1s[finite]
    
    if plot and plot_residuals:
        plotter=hplot.Figure(2,figsize=(8,6),sharex=True,ratios=[2,1])
        ax = plotter.axes
    elif plot and not plot_residuals:
        plotter=hplot.Figure(1,figsize=(8,6))
        ax = plotter.axes
        #ax[0].plot(pix1s,flx1s,ms=0.3,alpha=0.2,marker='o',ls='')
        
    shift = 0
    totshift = 0
    for j in range(numiter):
        # shift the values along x-axis for improved centering
        pix1s = pix1s+shift  
        # get current model of the LSF
        splr = interpolate.splrep(lsf1s['x'],lsf1s['y'])                    
        sple = interpolate.splev(pix1s,splr)
        # calculate residuals to the model
        rsd  = (flx1s-sple)
        # calculate mean of residuals for each pixel comprising the LSF
        means  = bin_means(pix1s,rsd,pixlims)
        lsf1s['y'] = lsf1s['y']+means
        
        # calculate derivative
        deriv = hf.derivative1d(lsf1s['y'],lsf1s['x'])
        lsf1s['dydx'] = deriv
        
        left  = np.where(lsf1s['x']==-0.5)[0]
        right = np.where(lsf1s['x']==0.5)[0]
        elsf_neg     = lsf1s['y'][left]
        elsf_pos     = lsf1s['y'][right]
        elsf_der_neg = lsf1s['dydx'][left]
        elsf_der_pos = lsf1s['dydx'][right]
        shift        = float((elsf_pos-elsf_neg)/(elsf_der_pos-elsf_der_neg))
        print("iter {0:2d} shift {1:12.6f}".format(j,shift))
        totshift += shift
        #count        +=1
    print('total shift {0:12.6f}'.format(totshift))   
    if plot:
        ax[0].scatter(pix1s,flx1s,s=4,alpha=0.2,marker='o',c='C0')
        ax[0].scatter(lsf1s['x'],lsf1s['y'],marker='s',s=32,
                      linewidths=0.2,edgecolors='k',c='C1',zorder=1000)
        if plot_residuals:
            ax[1].scatter(pix1s,rsd,s=1)
            ax[1].errorbar(pixcens,means,ls='',
                      xerr=0.5/subpix,ms=4,marker='s')
        for a in ax:
            [a.axvline(lim,ls=':',lw=0.4,color='k') for lim in pixlims]
def bin_means(x,y,xbins,minpts=1):
    def interpolate_bins(means,missing_xbins,kind='spline'):
        
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
        means[i] = np.nanmean(y1)
    # go back and interpolate means for empty bins
    idy   = hf.find_missing(idx)
    # interpolate if no points in the bin, but only pixels -5 to 5
    if len(idy)>0:
        idy = np.atleast_1d(idy)
        means[idy] = interpolate_bins(means,xbins[idy])
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
    assert seg_r<len(segcens), "Right segment 'too right', {}".format(seg_r)
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

def solve(lsf,linelists,fluxes,backgrounds,errors,fittype):
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
                                                  lsf1s,p0,output_model=True)
            except:
                continue
            line['lsf']     = pars
            line['lsf_err'] = errs
            line['lchisq']  = chisq
            #print(line['lsf'])
            if not success:
                print(line)
        hf.update_progress((exp+1)/tot,"Solve")
    return linelists

class LSF(object):
    def __init__(self,narray):
        self._values = narray
    def __getitem__(self,item):
        condict, segm_sent = self._extract_item(item)
        values  = self.values 
        condition = np.logical_and.reduce(tuple(values[key]==val \
                                       for key,val in condict.items()))
        
        cut = np.where(condition==True)
        if segm_sent:
            return LSF(values[cut][0])
        else:
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
            plotter = hplot.Figure(1)
        figure, axes = plotter.fig, plotter.axes
        nitems = len(values.shape)
        npts   = values['y'].shape[-1]
        x = np.linspace(np.min(values['x']),np.max(values['x']),3*npts)
        if nitems>0:
            numvals = len(values)
            colors = plt.cm.jet(np.linspace(0,1,numvals))
            for j,item in enumerate(values):
                splr = interpolate.splrep(item['x'],item['y'])                    
                sple = interpolate.splev(x,splr)
                axes[0].scatter(item['x'],item['y'],edgecolor='None',
                                c=[colors[j]])
                axes[0].plot(x,sple,lw=0.6,c=colors[j])
        else:            
            splr = interpolate.splrep(values['x'],values['y'])                    
            sple = interpolate.splev(x,splr)
            axes[0].scatter(values['x'],values['y'],edgecolor='None')
            axes[0].plot(x,sple,lw=0.6)
        axes[0].set_ylim(-0.03,0.35)
        if title:
            axes[0].set_title(title)
        if saveto:
            figure.savefig(saveto)
        return plotter
    def interpolate(self,order,center):
        return interpolate_local(self,order,center)