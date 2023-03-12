#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:40:20 2018

@author: dmilakov
"""
#import matplotlib
#matplotlib.use('GTKAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import harps.functions as hf
#from harps.lines import Linelist
#import harps.wavesol as ws
#from harps.classes import Manager, Spectrum
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib import ticker
#------------------------------------------------------------------------------

#                                PLOTTER   

#------------------------------------------------------------------------------

class Figure2(object):
    def __init__(self,nrows,ncols,width_ratios=None,height_ratios=None,title=None,
                 figsize=None,sharex=None,sharey=None,grid=None,subtitles=None,
                 enforce_figsize=False,left=0.1,right=0.95,top=0.95,
                 bottom=0.10,wspace=0.05,hspace=0.05, **kwargs):
        
        
        fig         = plt.figure(figsize=figsize)
        
        self._fig   = fig
        self._figsize = figsize
        
        self.nrows  = nrows
        self.ncols  = ncols
        self.col    = 0
        self.row    = 0
        
        
        self.top    = top
        self.bottom = bottom
        self.left   = left
        self.right  = right
        self.wspace = wspace
        self.hspace = hspace
        
#        self.alignment = alignment
        
        if enforce_figsize:
            fig.set_size_inches(figsize)
        
        gs  = GridSpec(nrows=nrows,ncols=ncols,
                       width_ratios=width_ratios, height_ratios=height_ratios,
                       left=left,right=right,top=top,bottom=bottom,
                       wspace=wspace,hspace=hspace)
        self._grid = gs
        self._axes = []
        return
    
    def add_subplot(self,top,bottom,left,right,*args,**kwargs):
        ax = self.fig.add_subplot(self._grid[slice(top,bottom),
                                             slice(left,right)],*args,**kwargs)
        
        self._axes.append(ax)
        return ax
    def ax(self):
        col = len(self._axes)%self.ncols
        row = len(self._axes)//self.nrows
        # col = self.col%self.ncols
        # row = self.row%self.nrows
        print(col,row,self.ncols,self.nrows)
        # try adding a column
        # if col==self.ncols and row==self.nrows:
        #     print("Impossible to add new subplots")
        # if col==self.ncols-1:
        #     self.col=0
        #     self.row+=1
        # if row==self.nrows-1:
        #     self.col+=1
        #     self.row=0
        if col<self.ncols and row<self.nrows:
            return self.add_subplot(row,row+1,col,col+1)
            # print("Add subplot",row,row+1,col,col+1)
        else:
            return None
        
    @property
    def fig(self):
        return self._fig
    @property
    def figure(self):
        return self._fig
    @property
    def axes(self):
        return self._axes
    def save(self,path,*args,**kwargs):
        self.fig.savefig(path,*args,**kwargs)
        print('Figure saved to:', path)
    def ticks(self,axnum,scale='x',ticknum=None,minval=None,maxval=None):
        ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
            ticks on a given scale (x or y)'''
        axis = self.axes[axnum]
        ticknum = ticknum if ticknum is not None else 4
        if minval is None or maxval is None:
            if scale == 'x':
                minval,maxval = axis.get_xlim()
            elif scale=='y':
                minval,maxval = axis.get_ylim()
            valrange = (maxval-minval)
            minval = minval + 0.05*valrange
            maxval = maxval - 0.05*valrange
        if scale=='x':
            axis.set_xticks(np.linspace(minval,maxval,ticknum))
        elif scale=='y':
            axis.set_yticks(np.linspace(minval,maxval,ticknum))
        return 

    def ticks_(self,which,axnum,axis='x',tick_every=None,ticknum=None):
        ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
            ticks on a given scale (x or y)'''
            
        ax   = self.axes[axnum]
        axsc = getattr(ax,'{0}axis'.format(axis))
        func = getattr(axsc,'set_{0}_locator'.format(which))
        
        ticknum = ticknum if ticknum is not None else 4
        if tick_every is not None:
            ax_ticker=ticker.MultipleLocator(tick_every)
        else:
            ax_ticker=ticker.MaxNLocator(ticknum)
        func(ax_ticker)
        return  
    def major_ticks(self,axnum,axis='x',tick_every=None,ticknum=None):
        ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
            ticks on a given scale (x or y)'''
        return self.ticks_('major',axnum,axis,tick_every,ticknum)
    def minor_ticks(self,axnum,axis='x',tick_every=None,ticknum=None):
        ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
            ticks on a given scale (x or y)'''
        return self.ticks_('minor',axnum,axis,tick_every,ticknum)
    def scinotate(self,axnum,axis,exp=None,dec=1,bracket='round'):
        ax   = self.axes[axnum]
        axsc = getattr(ax,'{0}axis'.format(axis))
        
        braleft = '[' 
        brarigh = ']'
        if bracket == 'round':
            braleft = '('
            brarigh = ')'
        
        oldlbl = getattr(ax,'get_{0}label'.format(axis))()
        loc    = oldlbl.find(brarigh)
        axlim  = getattr(ax,'get_{0}lim'.format(axis))()
        exp    = exp if exp is not None else np.floor(np.log10(axlim[1]))
        axsc.set_major_formatter(ticker.FuncFormatter(lambda x,y : sciformat(x,y,exp,dec)))
        
        
        if loc > 0:
            newlbl = oldlbl[:loc] + \
                r' $\times 10^{{{exp:0.0f}}}${br}'.format(exp=exp,br=brarigh)
        else:
            newlbl = oldlbl + \
                r' {bl}$\times 10^{{{exp:.0f}}}${br}'.format(exp=exp,br=brarigh,bl=braleft)
        print (newlbl)
        set_lbl = getattr(ax,'set_{0}label'.format(axis))
        set_lbl(newlbl)
        return
def scinotate(ax,axis,exp=None,dec=1,bracket='round'):
    '''
    Args:
    ----
    ax (matplotlib.Axes instance)
    axis (str) : 'x' or 'y'
    exp (int) : exponential
    dec (int) : number of decimal points
    bracket (str) : 'round' or 'square'
    '''
    axsc = getattr(ax,'{0}axis'.format(axis))
    
    braleft = '[' 
    brarigh = ']'
    if bracket == 'round':
        braleft = '('
        brarigh = ')'
    
    oldlbl = getattr(ax,'get_{0}label'.format(axis))()
    loc    = oldlbl.find(brarigh)
    axlim  = getattr(ax,'get_{0}lim'.format(axis))()
    exp    = exp if exp is not None else np.floor(np.log10(axlim[1]))
    axsc.set_major_formatter(ticker.FuncFormatter(lambda x,y : sciformat(x,y,exp,dec)))
    
    
    if loc > 0:
        newlbl = oldlbl[:loc] + \
            r' $\times 10^{{{exp:0.0f}}}${br}'.format(exp=exp,br=brarigh)
    else:
        newlbl = oldlbl + \
            r' {bl}$\times 10^{{{exp:.0f}}}${br}'.format(exp=exp,br=brarigh,bl=braleft)
    print (newlbl)
    set_lbl = getattr(ax,'set_{0}label'.format(axis))
    set_lbl(newlbl)
    return 
def ticks(ax,which,axis='x',tick_every=None,ticknum=None):
    ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
        ticks on a given scale (x or y)
        
    Args:
    -----
        ax (plt.Axes instance)
        which (str): 'major' or 'minor'
        axis (str): 'x' or 'y'
        tick_every (int,float): distance between ticks
        ticknum (int) : number of ticks (incompatible with tick_every)
    '''
        
    
        
#        ax   = self.axes[axnum]
    axsc = getattr(ax,'{0}axis'.format(axis))
    func = getattr(axsc,'set_{0}_locator'.format(which))
    
    ticknum = ticknum if ticknum is not None else 4
    if tick_every is not None:
        ax_ticker=ticker.MultipleLocator(tick_every)
    else:
        ax_ticker=ticker.MaxNLocator(ticknum)
    func(ax_ticker)
    return  
#------------------------------------------------------------------------------

#                                PLOTTER   

#------------------------------------------------------------------------------
class SpectrumPlotter(object):
    def __init__(self,naxes=1,ratios=None,title=None,sep=0.05,figsize=(16,9),
                 alignment="vertical",sharex=None,sharey=None,**kwargs):
        fig, axes = hf.get_fig_axes(naxes,ratios=ratios,title=title,
                                 sep=sep,alignment=alignment,
                                 figsize=figsize,sharex=sharex,
                                 sharey=sharey,**kwargs)
        self.figure = fig
        self.axes   = axes   
    def show(self):
        self.figure.show()
        return


def sciformat(x,y,exp,dec):
    if x==0:
        return ('{num:.{width}f}'.format(num=x,width=dec))
    return ('{num:.{width}f}'.format(num=x/10**exp,width=dec))


# =============================================================================
#                         F  U  N  C  T  I  O  N  S
# =============================================================================

def ccd_from_linelist(linelist,desc,fittype='gauss',scale='pix',mean=False,column=None,
                      label=None,yscale='wave',centre_estimate=None,*args,**kwargs):
    centre_colname = f'{fittype}_{scale}'
    if mean:
        # NOTE: y is frequencies if yscale='wave'
        x, y, c = mean_val(linelist,
                           f'{desc}',
                           centre_colname,
                           column,
                           yscale)
        if yscale == 'wave': # convert Hz to nanometres
            y = hf.freq_to_lambda(y)/10
    else:
        if centre_estimate != 'bary':
            x = linelist[centre_colname][:,1]
        else:
            x = linelist['bary']
        if yscale != 'wave':
            y = linelist['order']
        else:
            y = hf.freq_to_lambda(linelist['freq'])/10 # nanometres
        if column is not None:
            c = linelist[desc][:,column]
        else:
            c = linelist[desc]
        if column==0:
            # normalise amplitudes by sigma
            sigma = linelist[desc][:,2]
            c = 1 / (sigma * np.sqrt(2*np.pi))
            print('Normalised amplitude')
    if column is not None:
        c_hist = c#linelist[desc][:,column]
    else:
        c_hist = c#linelist[desc]
    print(mean,column,c.shape,c_hist.shape)
    label = label if label is not None else get_label(desc,column)  
    return ccd(x,y,c,c_hist,label,yscale,centre_estimate=centre_estimate,*args,**kwargs)

def ccd(x,y,c,c_hist,label=None,yscale='wave',bins=20,figsize=(10,9),
        centre_estimate=None,*args,**kwargs):
    
    plotter = Figure2(nrows=2,ncols=2,left=0.12,top=0.93,right=0.9,bottom=0.08,
                      vspace=0.2,hspace=0.05,
                      height_ratios=[1,4],width_ratios=[30,1],
                      figsize=figsize)
    fig    = plotter.figure
    ax_top = plotter.add_subplot(0,1,0,1)
    ax_bot = plotter.add_subplot(1,2,0,1)
    ax_bar = plotter.add_subplot(1,2,1,2)
      
    sc = ax_bot.scatter(x,
                y,
                c=c,
                cmap='inferno',
                marker='s',s=16,rasterized=True)
    
    # cmap = sc.get_cmap()
    minlim,maxlim = np.nanpercentile(c,[0.05,99.5])
    xrange = kwargs.pop('range',(minlim,maxlim))
    sc.set_clim(*xrange)
    if centre_estimate != 'bary':
        ax_bot.set_xlabel("Line centre (pix)")
    else:
        ax_bot.set_xlabel("Line barycentre (pix)")
    if yscale == 'wave':
        ax_bot.set_ylabel(r"Wavelength (nm)")
    else:
        ax_bot.set_ylabel("Optical order")
        ax_bot.invert_yaxis()
    
    norm = Normalize(vmin=xrange[0], vmax=xrange[1])
    cb1 = ColorbarBase(ax=ax_bar,norm=norm,label=label,cmap=sc.get_cmap())
    
    bins = bins
    lw=3
    alpha=0.8
    ax_top.hist(c_hist,bins=bins,color='black',
        range=xrange,
        histtype='step',density=False,
        lw=lw)
    ax_top.set_ylabel("# of lines")
    ax_top.set_xlabel(label)
    ax_top.xaxis.tick_top()
    ax_top.xaxis.set_label_position('top') 
 
    
    fig.align_ylabels()
    
#    plotter.scinotate(0,'y',exp=3,dec=0)
#    plotter.major_ticks(0,'x',tick_every=25)#ticks(ax0,'x',5,0,4096)
#    plotter.minor_ticks(0,'x',tick_every=12.5)#ticks(ax0,'x',5,0,4096)
    
    plotter.major_ticks(1,'x',tick_every=1024)#ticks(ax0,'x',5,0,4096)
    plotter.minor_ticks(1,'x',tick_every=256)#ticks(ax0,'x',5,0,4096)
    plotter.minor_ticks(1,'y',tick_every=10)#ticks(ax0,'x',5,0,4096)

    return plotter
def get_label(desc,column=None):
    label = desc
    if desc == 'chisq':
        label = r'$\chi^2$'
    elif desc == 'chisqnu':
        label = r'$\chi_\nu^2$'
    else:
        if 'err' in desc:
            if column == 0:
                label = r'$\sigma_A$'
            elif column == 1:
                label = r'$\sigma_\mu$'
            elif column == 2:
                if 'lsf' in desc:
                    label = r'$\sigma_w$'
                elif 'gauss' in desc:
                    label = r'$\sigma_\sigma$'
        else:
            if column == 0:
                label = r'A'
            elif column == 1:
                label = r'$\mu$'
            elif column == 2:
                if 'lsf' in desc:
                    label = r'w'
                elif 'gauss' in desc:
                    label = r'$\sigma$'
    return label
def mean_val(linelist,desc,fittype,column,yscale):
    xpositions   = []
    values      = []
    ypositions = []
    orders=np.unique(linelist['order'])
    for j,od in enumerate(orders):
        cutod  = np.where(linelist['order']==od)[0]
        mlod = linelist[cutod]
        #print(len(mlod.values))
        modes = np.unique(mlod['freq'])
        #print(len(modes))
        for f in modes:
            cut = np.where((linelist['order']==od)&(linelist['freq']==f))[0]
            #print(od,f,np.mean(linelist[cut][desc]))
            
            if column is not None:
                val = np.mean(linelist[cut][desc][:,column],axis=0)
            else:
                val = np.mean(linelist[cut][desc],axis=0)
            
            pos = np.mean(linelist[cut][fittype][:,1])
            xpositions.append(pos)
            if yscale=='wave':
                ypositions.append(f)
            else:
                ypositions.append(od)
            try:
                values.append(tuple(val))
            except:
                values.append(val)
        hf.update_progress((j+1)/len(orders),desc)
    return np.array(xpositions),np.array(ypositions),np.array(values)

       
#def wavesolution(linelist,wavesol='polynomial',order=None,
#                      plotter=None,**kwargs):
#    '''
#    Plots the wavelength solution for the provided orders using the linelist.
#    '''
#    
#    # ----------------------      READ ARGUMENTS     ----------------------
#    orders  = np.unique(linelist['order'])
#    fittype = kwargs.pop('fittype','gauss')
#    ai      = kwargs.pop('axnum', 0)
#    plotter = plotter if plotter is not None else SpectrumPlotter(**kwargs)
#    axes    = plotter.axes
#    # ----------------------        READ DATA        ----------------------
#    # Check and retrieve the wavelength calibration
#    wavesol_type = wavesol
#    if wavesol_type == 'twopoint':
#        wavesol = ws.twopoint(linelist,fittype)       
#    else:
#        version = kwargs.pop('version',hf.item_to_version(None))
#        wavesol = ws.polynomial(linelist,version,fittype)
#   
#    
#    
#    fittype = hf.to_list(fittype)
#    
#    
#    frequencies = linelist['freq'] 
#    wavelengths = hf.freq_to_lambda(frequencies)
#    # Manage colors
#    #cmap   = plt.get_cmap('viridis')
#    colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
#    marker = kwargs.get('marker','x')
#    ms     = kwargs.get('markersize',5)
#    ls     = kwargs.get('ls','-')#{'epsf':'--','gauss':'-'}
#    lw     = kwargs.get('lw',0.5)
#    # Plot the line through the points?
#    plotline = kwargs.get('plot_line',True)
#    # Select line data    
#    for ft in fittype:
#        centers  = linelist[ft][:,1]
#        # Do plotting
#        for i,order in enumerate(orders):
#            cut = np.where(linelist['order']==order)
#            pix = centers[cut]
#            wav = wavelengths[cut]
#            axes[ai].scatter(pix,wav,s=ms,color=colors[i],marker=marker)
#            if plotline == True:
#                axes[ai].plot(wavesol[order],color=colors[i],ls=ls,lw=lw)
#    axes[ai].set_xlabel('Pixel')
#    axes[ai].set_ylabel('Wavelength [$\AA$]')
#    return plotter

#class ManagerPlotter(object):
#    """ IDEA: separate class for plotting data"""
#    def __init__(self,plot_object,figsize=(16,9),**kwargs):
#        if   plot_object.__class__ == Manager:
#            self.manager = plot_object
#            self.plot_object_class = Manager#.__class__
#            self.fibre = self.manager.fibre
#            self.orders = self.manager.orders
#        if   plot_object.__class__ == Spectrum:
#            self.spectrum = plot_object
#            self.plot_object_class = Spectrum#.__class__
#            self.fibre = plot_object.filepath[-6]
#        self.fig = plt.figure(figsize=figsize)
#        self.defaultparams = (0.1,0.1,0.85,0.85)	# position and size of the canvas
#        self.fontsize=12
#        self.axes = []
#        colours      = [Colours().palette for i in range(20)]
#        self.colours = [item for sublist in colours for item in sublist]
#        try:
#            self.dtype   = kwargs["dtype"]
#        except:
#            if self.plot_object_class == Manager:
#                self.dtype   = self.manager.dtype
#            elif self.plot_object_class == Spectrum:
#                self.dtype   = ["FLX","ENV","BKG","B2E","FMB"]
#        #self.datatypes = Datatypes(self.manager.nfiles[0],nOrder=self.manager.orders,fibre=self.manager.fibre, add_corr=True)
#    
#        
#        
#    def create_canvas(self,ctype,size,**kwargs):
#        if ctype == "SPECTRUM":
#            self.axes.append(self.fig.add_axes(size,**kwargs))
#        if ctype == "FOURIER":
#            self.axes.append(self.fig.add_axes(size,**kwargs))
#        if ctype == "RV":
#            self.axes.append(self.fig.add_axes(size,**kwargs))
#    def plot(self,dtype,ctype,**kwargs):
#        ctype = ctype.upper()
#        #additional plot arguments
#        try:
#            fibre  = list(kwargs["fibre"])
#        except: print("Please select fibre(s).")
#        try:
#            labels = kwargs["legend"]
#        except: pass
#        try: orders = kwargs["orders"]
#        except: 
#            try:
#                orders = self.orders
#            except:
#                print("Please specify orders.")
#        try: median = kwargs["median"]
#        except: median = False
#        
#        self.get_plot_params(dtype=self.dtype,orders=orders)
#        
#        if not self.axes:
#            naxes = len(fibre) #number of axes
#            top, bottom = (0.95,0.08)
#            left, right = (0.1,0.95)
#            W, H        = (right-left, top-bottom)
#            s           = 0.05
#            h           = H/naxes - (naxes-1)/naxes*s
#            for i in range(naxes):
#                down    = top - (i+1)*h - i*s
#                size = [left,down,W,h]
#                if i==0:
#                    self.create_canvas(ctype,size)
#                if i>0:
#                    self.create_canvas(ctype,size,sharex=self.axes[0],sharey=self.axes[0])
#        
#            #labels = [np.arange(np.shape(data[f])[1]) for f in fibre]    
#        ylims = []
#        if ctype=="SPECTRUM":
#            for fn,f in enumerate(fibre):
#                ax = self.axes[fn]
#                for dt in dtype:
#                    for i,o in enumerate(orders): 
#                        pargs = self.plot_params[f][dt][i]
#                        print(pargs["label"])
#                        if self.plot_object_class == Manager:
#                            if   median == True:
#                                data = self.manager.data50p[f][dt][:,i]
#                                if dt=="B2E":
#                                    data = data*100.
#                            elif median == False:
#                                data = self.manager.data[f][dt][:,i]
#                        elif self.plot_object_class == Spectrum:
#                            spec1d  = self.spectrum.extract1d(o)
#                            env     = self.spectrum.get_envelope1d(o)
#                            bkg     = self.spectrum.get_background1d(o)
#                            b2e     = bkg/env
#                            
#                            fmb         = spec1d['flux']-bkg
#                            if   dt == "FLX":
#                                data = spec1d['flux']
#                            elif dt == "ENV":
#                                data = env
#                            elif dt == "BKG":
#                                data = bkg
#                            elif dt == "B2E":
#                                data = b2e
#                            elif dt == "FMB":
#                                data = fmb 
#                        try:
#                            ax.plot(data, **pargs)
#                        except:
#                            print("Something went wrong")
#                        del(pargs)
#                    
#                        ylims.append(1.5*np.percentile(data,98))
#                    #print(np.percentile(self.manager.data[f][dt],98))
#                ax.set_xlim(0,4096)
#            print(ylims)
#            self.axes[-1].set_xlabel("Pixel")
#            self.axes[-1].set_ylim(0,max(ylims))
#            
#        if ctype=="FOURIER":
#            #print("Fourier")
#            lst = {"real":'-', "imag":'-.'}
#            for fn,f in enumerate(fibre):
#                ax = self.axes[fn]
#                for dt in dtype:
#                    for i,o in enumerate(orders):
#                        pargs = self.plot_params[f][dt][i]
#                        print(f,dt,i,o,pargs)
#                        if self.plot_object_class == Manager:
#                            if   median == True:
#                                data = self.manager.datafft50p[f][dt][:,i]
#                                freq = self.manager.freq
#                            elif median == False:
#                                data = self.manager.datafft[f][dt][:,i]
#                                freq = self.manager.freq
#                        elif self.plot_object_class == Spectrum:
#                            data = self.spectrum.calculate_fourier_transform()
#                            freq = self.spectrum.freq
#                        #print(data.real.shape, self.manager.freq.shape)
#                        #try:
#                        ax.plot(freq, data.real,lw=2.,**pargs)
#                        #   print("Plotted")
#                        #except:
#                        #   print("Something went wrong")                
#                ax.set_xscale('log')
#            self.axes[-1].set_xlabel("Period [Pixel$^{-1}$]")
#        if ctype == "RV":
#            bins = 100
#            fs = 12 #25 for posters
#            # An example of three data sets to compare
#            labels = [str(o) for o in orders]
#            data_sets = [self.spectrum.get_rv_diff(o) for o in orders]
#          
#            # Computed quantities to aid plotting
#            #hist_range = (np.min([np.min(dd) for dd in data_sets]),
#            #              np.max([np.max(dd) for dd in data_sets]))
#            hist_range = (-5,5)
#            binned_data_sets = [np.histogram(d, range=hist_range, bins=bins)[0]
#                                for d in data_sets]
#            binned_maximums = np.max(binned_data_sets, axis=1)
#            y_locations = np.linspace(0, 1.8*sum(binned_maximums), np.size(binned_maximums))           
#            # The bin_edges are the same for all of the histograms
#            bin_edges = np.linspace(hist_range[0], hist_range[1], bins+1)
#            centers = .5 * (bin_edges + np.roll(bin_edges, 1))[1:]
#            widths = np.diff(bin_edges)
#            # Cycle through and plot each histogram
#            for ax in self.axes:
#                ax.axvline(color='k',ls=':',lw=1.5)
#                i=0
#                for y_loc, binned_data in zip(y_locations, binned_data_sets):
#                    #ax.barh(centers, binned_data, height=heights, left=lefts)
#                    ax.bar(left=centers,height=binned_data,bottom=y_loc, width=widths,lw=0,color=self.colours[0],align='center')
#                    ax.axhline(xmin=0.05,xmax=0.95,y=y_loc,ls="-",color='k')
#                    dt = np.where((data_sets[i]>hist_range[0])&(data_sets[i]<hist_range[1]))[0]
#                    #ax.annotate("{0:5.3f}".format(median_rv[i]),xy=(median_rv[i],y_loc),xytext=(median_rv[i],y_loc+binned_maximums[i]))
#                    for dd in data_sets[i][dt]:
#                        ax.plot((dd,dd),(y_loc-2,y_loc-7),color=self.colours[0])
#                    i+=1
#                
#                ax.set_yticks(y_locations)
#                ax.set_yticklabels(labels)
#                ax.yaxis.set_tick_params(labelsize=fs)
#                ax.xaxis.set_tick_params(labelsize=fs)
#                
#                ax.set_xlabel("Radial velocity [m/s]",fontsize=fs)
#                ax.set_ylabel("Echelle order",fontsize=fs)
#                
#                ax.spines["left"].set_visible(False)
#                ax.spines["right"].set_visible(False)
#                ax.spines["top"].set_visible(False)
#                ax.tick_params(axis='both', direction='out')
#                ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
#                ax.get_yaxis().tick_left()
#                ax.set_ylim(-10,2.1*sum(binned_maximums))
#            plt.show()
#
#        for ax in self.axes:
#            ax.legend()
#    
#        plt.show()
#        return
#    def get_plot_params(self,orders,**kwargs):
#        fibre  = list(self.fibre)
#        try:
#            dtype = kwargs["dtype"]
#        except:
#            dtype = self.manager.dtype
#        real1d = np.dtype([((dtype[i],list,1)) for i in range(np.size(dtype))])
#        self.plot_params = np.empty(shape=np.shape(orders), dtype=np.dtype({"names":fibre, "formats":[real1d for f in fibre]}))
#        lstyles = {"FLX":"-","ENV":"--","BKG":":","B2E":"-","FMB":"-"}
#        lwidths = {"FLX":1.0,"ENV":2.0,"BKG":2.0,"B2E":1.5,"FMB":1.0}
#        #print(fibre,type(orders),orders.shape,self.plot_params)
#        for f in fibre:
#            j=0
#            for dt in dtype:
#                k=0
#                for i,o in enumerate(orders):
#                    label = "{f} {d} {o}".format(f=f,d=dt,o=o)
#                    #print(f,dt,o,"label=",label)
#                    c=self.colours[i]
#                    #print(lstyles[dt])
#                    pargs = {"label":label, "c":c,"ls":lstyles[dt], "lw":lwidths[dt]}
#                    
#                    self.plot_params[f][dt][i] = pargs
#                    del(pargs)
#                    k+=2
#                j+=5
#        return