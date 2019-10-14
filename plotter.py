#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:40:20 2018

@author: dmilakov
"""
#import matplotlib
#matplotlib.use('GTKAgg')

from harps.core import np, pd
from harps.core import plt

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
class Figure(object):
    def __init__(self,naxes,ratios=None,title=None,alignment="vertical",
                 figsize=None,sharex=None,sharey=None,grid=None,subtitles=None,
                 enforce_figsize=False,left=0.1,right=0.95,top=0.95,
                 bottom=0.10,wspace=0.05,hspace=0.05, **kwargs):
        
        
        fig         = plt.figure(figsize=figsize)
        
        self._fig   = fig
        self._figsize = figsize
        
        self.naxes  = naxes
        self.top    = top
        self.bottom = bottom
        self.left   = left
        self.right  = right
        self.wspace = wspace
        self.hspace = hspace
        
        self.alignment = alignment
        
        if enforce_figsize:
            fig.set_size_inches(figsize)
        # Change color scheme and text size if producing plots for a presentation
        # assuming black background
        
        
        # Share X axis
        share_x = self._shareax(sharex)
        
        # First item with sharex==True:
        try:
            self.firstx = share_x.index(True)
        except:
            self.firstx = None
        # Share Y axis  
        share_y = self._shareax(sharey)
        # First item with sharey==True:
        try:
            self.firsty = share_y.index(True)
        except:
            self.firsty = None
        
        self.share_xy = [(share_x[i],share_y[i]) for i in range(self.naxes)]
        
        
        # GRID
        if grid==None:
            grid = self._get_grid(alignment,naxes)
#            grid = self._grid_from_ratios(ratios)
        else:
            grid = np.array(grid,dtype=int)
        ncols,nrows = grid
        self.ncols = ncols
        self.nrows = nrows
#        print(ncols,nrows)
        self.grid     = GridSpec(nrows=ncols,ncols=nrows,figure=self._fig,
                                 left=self.left,right=self.right,top=self.top,
                                 bottom=self.bottom,wspace=self.wspace,
                                 hspace=self.hspace)
        self.ratios   = self._get_ratios(ratios)
#        colslice,rowslice = self._axes_slices(ratios)
#        for i,j in zip(colslice,rowslice):
#            print(i,j)
#        print(self.ratios)
        self._axsizes = self._get_axsizes()
#        print(self._axsizes)
        self._axes    = self._get_axes()
#        self._axes = [self._fig.add_subplot(self.grid[si,sj]) \
#                            for si,sj in zip(colslice,rowslice)]
#        if presentation:
#            self._make_presentable(self)
        return 
    
    def _ratios_arr(self,ratios):
        if ratios is not None:
            ratios = np.atleast_2d(ratios)
        else:
            ratios = np.atleast_2d([np.ones(self.naxes),np.ones(self.naxes)])
        return ratios
    def _get_grid(self,alignment,naxes):
        if alignment=="grid":
            ncols = np.int(round(np.sqrt(naxes)))
            nrows,lr = [np.int(k) for k in divmod(naxes,round(np.sqrt(naxes)))]
            if lr>0:
                nrows += 1     
        elif alignment=="vertical":
            ncols = 1
            nrows = naxes
        elif alignment=="horizontal":
            ncols = naxes
            nrows = 1
        grid = np.array([ncols,nrows],dtype=int)
        return grid
    def _get_ratios(self,ratios):
        if ratios==None:
            ratios = np.vstack([np.ones(self.ncols,dtype=int),
                                np.ones(self.nrows,dtype=int)])
        else:
            if len(np.shape(ratios))==1:
                # Ratios given in 1d
                if   self.alignment == 'vertical':
                    ratios = np.array([np.ones(self.ncols),ratios])
                elif self.alignment == 'horizontal':
                    ratios = np.array([ratios,np.ones(self.nrows)])
            elif len(np.shape(ratios))==2:
                # Ratios given in 2d
                ratios = np.array(ratios).reshape((self.ncols,self.nrows))
        return ratios
    def _grid_from_ratios(self,ratios):
        ratarr    = self._ratios_arr(ratios)
        sumratios = np.sum(ratarr,axis=-1,dtype=int)
#        print(np.shape(ratarr))
#        print(sumratios)
        if len(np.shape(ratarr))==2:
            ncols,nrows = sumratios
        else:
            # Ratios given in 1d
            if   self.alignment == 'vertical':
                ncols = 1
                nrows = sumratios
            elif self.alignment == 'horizontal':
                ncols = sumratios
                nrows = 1
        return ncols,nrows
    def _axes_slices(self,ratios):
        ratarr = self._ratios_arr(ratios)
        end = np.cumsum(ratarr,axis=1,dtype=int)
        start = end-ratarr
        cols,rows=np.dstack([start,end]).astype(int)
        colslice = [slice(*c) for c in cols]
        rowslice = [slice(*r) for r in rows]
        return colslice, rowslice
    def _get_axsizes(self):
        top, bottom = (self.top,self.bottom)
        left, right = (self.left,self.right)
        W, H        = (right-left, top-bottom)
        s           = self.hspace
#        h           = H/naxes - (naxes-1)/naxes*s
        
        h0          = (H - (self.nrows-1)*s)/np.sum(self.ratios[1])
        w0          = (W - (self.ncols-1)*s)/np.sum(self.ratios[0])
        axsize      = []
        for c in range(self.ncols):
            for r in range(self.nrows):
                ratiosc = self.ratios[0][:c]
                ratiosr = self.ratios[1][:r+1]
                w  = self.ratios[0][c]*w0
                h  = self.ratios[1][r]*h0
                l  = left + np.sum(ratiosc)*w0 + c*s
                d  = top - np.sum(ratiosr)*h0 - r*s
                size  = [l,d,w,h] 
                axsize.append(size)    
        return axsize
    def _get_axes(self):
        axes    = []
        axsizes = self._axsizes
        for i in range(self.naxes):   
            size   = axsizes[i]
            sharex,sharey = self.share_xy[i]
            if i==0:
                axes.append(self.fig.add_axes(size))
            else:
                kwargs = {}
                if   (sharex==True  and sharey==False):
                    kwargs["sharex"]=axes[self.firstx]
                    #axes.append(fig.add_axes(size,sharex=axes[firstx]))
                elif (sharex==False and sharey==True):
                    kwargs["sharey"]=axes[self.firsty]
                    #axes.append(fig.add_axes(size,sharey=axes[firsty]))
                elif (sharex==True  and sharey==True):
                    kwargs["sharex"]=axes[self.firstx]
                    kwargs["sharey"]=axes[self.firsty]
                    #axes.append(fig.add_axes(size,sharex=axes[firstx],sharey=axes[firsty]))
                elif (sharex==False and sharey==False): 
                    pass
                    #axes.append(fig.add_axes(size))
                axes.append(self.fig.add_axes(size,**kwargs))
        for a in axes:
            a.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))
        return axes
    def _shareax(self,share):
        if share!=None:
            if type(share)==list:
                pass
            else:
                share = list(share for i in range(self.naxes))
        elif share==None:
            share = list(False for i in range(self.naxes))
        return share
    
    def _make_presentable(self,**plotargs):
        spine_col  = plotargs.pop('spine_color','k')
        text_size  = plotargs.pop('text_size','20')
        hide_spine = plotargs.pop('hide_spine',[])
        spine_lw   = plotargs.pop('spine_lw','3')
        #spine_ec   = plotargs.pop('spine_ec','k')
        axes = self.axes
        for a in axes:
            #plt.setp(tuple(a.spines.values()), edgecolor=spine_ec)
            plt.setp(tuple(a.spines.values()), color=spine_col)
            plt.setp(tuple(a.spines.values()), linewidth=spine_lw)
            
            plt.setp(tuple(a.spines.values()), facecolor=spine_col)
            plt.setp([a.get_xticklines(), a.get_yticklines(),a.get_xticklabels(),a.get_yticklabels()], color=spine_col)
            plt.setp([a.get_xticklabels(),a.get_yticklabels()],size=text_size)
            for s in hide_spine:
                a.spines[s].set_visible(False)
                #plt.setp([a.get_xlabel(),a.get_ylabel()],color=spine_col,size=text_size)
            #plt.setp(a.get_yticklabels(),visible=False)
        return
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
    @property
    def fig(self):
        return self._fig
    @property
    def figure(self):
        return self._fig
    @property
    def axes(self):
        return self._axes
    
    
    @property
    def figsize(self):
        return self._figsize
    @figsize.setter
    def figsize(self,size):
        if isinstance(size,tuple):
            self._figsize = size
        else:
            raise TypeError("Size must be tuple")
        
    
    @property
    def title(self):
        return self._fig.get_title()
    @title.setter
    def title(self,text):
        self._fig.suptitle(text)
        return
        
        # Calculate canvas dimensions
        
    
    def plot(self,ax_index,*args,**kwargs):
        self.axes[ax_index].plot(*args,**kwargs)
        return
    
    def save(self,path,*args,**kwargs):
        self.fig.savefig(path,*args,**kwargs)
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
    def scinotate(self,axnum,axis,exp=None,dec=1):
        ax   = self.axes[axnum]
        axsc = getattr(ax,'{0}axis'.format(axis))
        
        
        oldlbl = getattr(ax,'get_{0}label'.format(axis))()
        loc    = oldlbl.find(']')
        axlim  = getattr(ax,'get_{0}lim'.format(axis))()
        exp    = exp if exp is not None else np.floor(np.log10(axlim[1]))
        axsc.set_major_formatter(ticker.FuncFormatter(lambda x,y : sciformat(x,y,exp,dec)))
        if loc > 0:
            newlbl = oldlbl[:loc] + r' $\times 10^{0:0.0f}$]'.format(exp)
        else:
            newlbl = oldlbl + r' [$\times 10^{0:.0f}$]'.format(exp)
        print (newlbl)
        set_lbl = getattr(ax,'set_{0}label'.format(axis))
        set_lbl(newlbl)
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
class LSFPlotter(object):
    def __init__(self,filepath):
        lsf = xr.open_dataset(filepath)
        self.lsf = lsf
        
    def initilize_plotter(self,naxes=1,ratios=None,title=None,sep=0.05,figsize=(9,9),
                 alignment="vertical",sharex=None,sharey=None,**kwargs):
        fig, axes = hf.get_fig_axes(naxes,ratios=ratios,title=title,
                                 sep=sep,alignment=alignment,
                                 figsize=figsize,sharex=sharex,
                                 sharey=sharey,**kwargs)
        self.figure = fig
        self.axes   = axes  
        
        return
    def plot_epsf(self,order=None,seg=None,plot_lsf=True,plot_points=False,fig=None,):
        ''' Plots the full data, including the lines used for reconstruction'''
        data = self.lsf
        if order is None:
            orders   = data.coords['od'].values
        else:
            orders   = hf.to_list(order)
        print(orders)
        if seg is None:    
            segments = np.unique(data.coords['seg'].values)
        else:
            segments = hf.to_list(seg)
        ids = np.unique(data.coords['id'].values)
        #sgs = np.unique(data.coords['sg'].values)
        sgs = segments
        sps = np.unique(data.coords['sp'].values)
        
        midx  = pd.MultiIndex.from_product([sgs,sps,np.arange(60)],
                                names=['sg','sp','id'])
        if fig is None:
            fig,ax = hf.get_fig_axes(1,figsize=(9,9))
        else:
            fig = fig
            ax  = fig.get_axes()
        for order in orders:

            
            for s in sgs:
                if plot_points:
                    data_s = data['shft'].sel(od=order,seg=s)
                    for lid in range(60):
                        data_x = data['line'].sel(ax='x',od=order,sg=s,id=lid).dropna('pix','all')
                        pix = data_x.coords['pix']
                        if np.size(pix)>0:
                            data_y = data['line'].sel(ax='y',od=order,sg=s,id=lid,pix=pix)
                            print(np.shape(data_x),np.shape(data_y),np.shape(data_s))
                            ax[0].scatter(data_x+data_s,data_y,s=5,c='C0',marker='s',alpha=0.3)
                        else:
                            continue
                if plot_lsf:
                    epsf_x = data['epsf'].sel(ax='x',od=order,seg=s).dropna('pix','all')
                    epsf_y = data['epsf'].sel(ax='y',od=order,seg=s).dropna('pix','all')
                    ax[0].scatter(epsf_x,epsf_y,marker='x',c='C1') 
        ax[0].set_xlabel('Pixel')
        ax[0].set_yticks([])
        return fig 
    def plot_psf(self,psf_ds,order=None,fig=None,**kwargs):
        '''Plots only the LSF'''
        if order is not None:
            orders = hf.to_list(order)
        else:
            orders   = psf_ds.coords['od'].values
        segments = np.unique(psf_ds.coords['seg'].values)
        # provided figure?
        if fig is None:
            fig,ax = hf.get_fig_axes(len(segments),alignment='grid')
        else:
            fig = fig
            ax  = fig.get_axes()
        # colormap
        if len(orders)>5:
            cmap = plt.get_cmap('jet')
        else:
            cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0,1,len(orders)))
        # line parameters
        ls = kwargs.pop('ls','')
        lw = kwargs.pop('lw',0.3)
        for o,order in enumerate(orders):
            for n in segments:
                epsf_y = psf_ds.sel(ax='y',seg=n,od=order).dropna('pix','all')
                epsf_x = psf_ds.sel(ax='x',seg=n,od=order).dropna('pix','all')
                #print(epsf_y, epsf_x)
                ax[n].plot(epsf_x,epsf_y,c=colors[o],ls=ls,lw=lw,marker='x',ms=3) 
                ax[n].set_title('{0}<pix<{1}'.format(n*256,(n+1)*256), )
        return fig 
    def to_list(self,item):
        if type(item)==str:
            items = [item]
        elif type(item) == list:
            items = item
        elif item is None:
            items = []
        elif type(item)==int:
            items = [item]
        else:
            print("Type provided {}".format(type(item)))
        return items
    
    

def sciformat(x,y,exp,dec):
    if x==0:
        return ('{num:.{width}f}'.format(num=x,width=dec))
    return ('{num:.{width}f}'.format(num=x/10**exp,width=dec))


# =============================================================================
#                         F  U  N  C  T  I  O  N  S
# =============================================================================

def ccd_from_linelist(linelist,desc,fittype='gauss',mean=False,column=None,
                      *args,**kwargs):
    if mean:
        x, y, c = mean_val(linelist,
                           '{}'.format(desc),
                           '{}'.format(fittype),
                           column)
    else:
        x = linelist[fittype][:,1]
        y = linelist['freq']
        if column is not None:
            c = linelist[desc][:,column]
        else:
            c = linelist[desc]
    return ccd(x,y,c,*args,**kwargs)

def ccd(x,y,c,yscale='wave',bins=20,figsize=(10,9)):
    
    plotter = Figure2(nrows=2,ncols=2,left=0.12,top=0.93,right=0.9,bottom=0.08,
                      vspace=0.2,hspace=0.03,
                      height_ratios=[1,4],width_ratios=[30,1],
                      figsize=figsize)
    fig    = plotter.figure
    ax_top = plotter.add_subplot(0,1,0,1)
    ax_bot = plotter.add_subplot(1,2,0,1)
    ax_bar = plotter.add_subplot(1,2,1,2)
      
    sc = ax_bot.scatter(x,
                hf.freq_to_lambda(y)/10,
                c=c,
                cmap='inferno',
                marker='s',s=16,rasterized=True)
    
    minlim,maxlim = np.percentile(c,[0.05,99.5])
    sc.set_clim(minlim,maxlim)
    ax_bot.set_xlabel("Line centre [pix]")
    if yscale == 'wave':
        ax_bot.set_ylabel(r"Wavelength [nm]")
    else:
        ax_bot.set_ylabel("Optical order")
        ax_bot.invert_yaxis()
    
    norm = Normalize(vmin=minlim, vmax=maxlim)
    cb1 = ColorbarBase(ax=ax_bar,norm=norm,label=r'$\chi_\nu^2$',cmap='inferno')
    
    bins = bins
    lw=3
    alpha=0.8
    
    ax_top.hist(c,bins=bins,color='black',
        range=(minlim,maxlim),
        histtype='step',density=False,
        lw=lw)
    ax_top.set_ylabel("Number of \nlines")
    ax_top.set_xlabel(r'$\chi_\nu^2$')
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

def mean_val(linelist,desc,fittype,column):
    positions   = []
    values      = []
    frequencies = []
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
            positions.append(pos)
            frequencies.append(f)
            try:
                values.append(tuple(val))
            except:
                values.append(val)
        hf.update_progress((j+1)/len(orders),desc)
    return np.array(positions),np.array(frequencies),np.array(values)

       
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