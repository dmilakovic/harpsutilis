#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:40:20 2018

@author: dmilakov
"""
from harps.core import np, pd, xr
from harps.core import plt

import harps.functions as hf
from harps.classes import Manager, Spectrum
###############################################################################
##############################   PLOTTER   ####################################
###############################################################################
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
class ManagerPlotter(object):
    """ IDEA: separate class for plotting data"""
    def __init__(self,plot_object,figsize=(16,9),**kwargs):
        if   plot_object.__class__ == Manager:
            self.manager = plot_object
            self.plot_object_class = Manager#.__class__
            self.fibre = self.manager.fibre
            self.orders = self.manager.orders
        if   plot_object.__class__ == Spectrum:
            self.spectrum = plot_object
            self.plot_object_class = Spectrum#.__class__
            self.fibre = plot_object.filepath[-6]
        self.fig = plt.figure(figsize=figsize)
        self.defaultparams = (0.1,0.1,0.85,0.85)	# position and size of the canvas
        self.fontsize=12
        self.axes = []
        colours      = [Colours().palette for i in range(20)]
        self.colours = [item for sublist in colours for item in sublist]
        try:
            self.dtype   = kwargs["dtype"]
        except:
            if self.plot_object_class == Manager:
                self.dtype   = self.manager.dtype
            elif self.plot_object_class == Spectrum:
                self.dtype   = ["FLX","ENV","BKG","B2E","FMB"]
        #self.datatypes = Datatypes(self.manager.nfiles[0],nOrder=self.manager.orders,fibre=self.manager.fibre, add_corr=True)
    
        
        
    def create_canvas(self,ctype,size,**kwargs):
        if ctype == "SPECTRUM":
            self.axes.append(self.fig.add_axes(size,**kwargs))
        if ctype == "FOURIER":
            self.axes.append(self.fig.add_axes(size,**kwargs))
        if ctype == "RV":
            self.axes.append(self.fig.add_axes(size,**kwargs))
    def plot(self,dtype,ctype,**kwargs):
        ctype = ctype.upper()
        #additional plot arguments
        try:
            fibre  = list(kwargs["fibre"])
        except: print("Please select fibre(s).")
        try:
            labels = kwargs["legend"]
        except: pass
        try: orders = kwargs["orders"]
        except: 
            try:
                orders = self.orders
            except:
                print("Please specify orders.")
        try: median = kwargs["median"]
        except: median = False
        
        self.get_plot_params(dtype=self.dtype,orders=orders)
        
        if not self.axes:
            naxes = len(fibre) #number of axes
            top, bottom = (0.95,0.08)
            left, right = (0.1,0.95)
            W, H        = (right-left, top-bottom)
            s           = 0.05
            h           = H/naxes - (naxes-1)/naxes*s
            for i in range(naxes):
                down    = top - (i+1)*h - i*s
                size = [left,down,W,h]
                if i==0:
                    self.create_canvas(ctype,size)
                if i>0:
                    self.create_canvas(ctype,size,sharex=self.axes[0],sharey=self.axes[0])
        
            #labels = [np.arange(np.shape(data[f])[1]) for f in fibre]    
        ylims = []
        if ctype=="SPECTRUM":
            for fn,f in enumerate(fibre):
                ax = self.axes[fn]
                for dt in dtype:
                    for i,o in enumerate(orders): 
                        pargs = self.plot_params[f][dt][i]
                        print(pargs["label"])
                        if self.plot_object_class == Manager:
                            if   median == True:
                                data = self.manager.data50p[f][dt][:,i]
                                if dt=="B2E":
                                    data = data*100.
                            elif median == False:
                                data = self.manager.data[f][dt][:,i]
                        elif self.plot_object_class == Spectrum:
                            spec1d  = self.spectrum.extract1d(o)
                            env     = self.spectrum.get_envelope1d(o)
                            bkg     = self.spectrum.get_background1d(o)
                            b2e     = bkg/env
                            
                            fmb         = spec1d['flux']-bkg
                            if   dt == "FLX":
                                data = spec1d['flux']
                            elif dt == "ENV":
                                data = env
                            elif dt == "BKG":
                                data = bkg
                            elif dt == "B2E":
                                data = b2e
                            elif dt == "FMB":
                                data = fmb 
                        try:
                            ax.plot(data, **pargs)
                        except:
                            print("Something went wrong")
                        del(pargs)
                    
                        ylims.append(1.5*np.percentile(data,98))
                    #print(np.percentile(self.manager.data[f][dt],98))
                ax.set_xlim(0,4096)
            print(ylims)
            self.axes[-1].set_xlabel("Pixel")
            self.axes[-1].set_ylim(0,max(ylims))
            
        if ctype=="FOURIER":
            #print("Fourier")
            lst = {"real":'-', "imag":'-.'}
            for fn,f in enumerate(fibre):
                ax = self.axes[fn]
                for dt in dtype:
                    for i,o in enumerate(orders):
                        pargs = self.plot_params[f][dt][i]
                        print(f,dt,i,o,pargs)
                        if self.plot_object_class == Manager:
                            if   median == True:
                                data = self.manager.datafft50p[f][dt][:,i]
                                freq = self.manager.freq
                            elif median == False:
                                data = self.manager.datafft[f][dt][:,i]
                                freq = self.manager.freq
                        elif self.plot_object_class == Spectrum:
                            data = self.spectrum.calculate_fourier_transform()
                            freq = self.spectrum.freq
                        #print(data.real.shape, self.manager.freq.shape)
                        #try:
                        ax.plot(freq, data.real,lw=2.,**pargs)
                        #   print("Plotted")
                        #except:
                        #   print("Something went wrong")                
                ax.set_xscale('log')
            self.axes[-1].set_xlabel("Period [Pixel$^{-1}$]")
        if ctype == "RV":
            bins = 100
            fs = 12 #25 for posters
            # An example of three data sets to compare
            labels = [str(o) for o in orders]
            data_sets = [self.spectrum.get_rv_diff(o) for o in orders]
          
            # Computed quantities to aid plotting
            #hist_range = (np.min([np.min(dd) for dd in data_sets]),
            #              np.max([np.max(dd) for dd in data_sets]))
            hist_range = (-5,5)
            binned_data_sets = [np.histogram(d, range=hist_range, bins=bins)[0]
                                for d in data_sets]
            binned_maximums = np.max(binned_data_sets, axis=1)
            y_locations = np.linspace(0, 1.8*sum(binned_maximums), np.size(binned_maximums))           
            # The bin_edges are the same for all of the histograms
            bin_edges = np.linspace(hist_range[0], hist_range[1], bins+1)
            centers = .5 * (bin_edges + np.roll(bin_edges, 1))[1:]
            widths = np.diff(bin_edges)
            # Cycle through and plot each histogram
            for ax in self.axes:
                ax.axvline(color='k',ls=':',lw=1.5)
                i=0
                for y_loc, binned_data in zip(y_locations, binned_data_sets):
                    #ax.barh(centers, binned_data, height=heights, left=lefts)
                    ax.bar(left=centers,height=binned_data,bottom=y_loc, width=widths,lw=0,color=self.colours[0],align='center')
                    ax.axhline(xmin=0.05,xmax=0.95,y=y_loc,ls="-",color='k')
                    dt = np.where((data_sets[i]>hist_range[0])&(data_sets[i]<hist_range[1]))[0]
                    #ax.annotate("{0:5.3f}".format(median_rv[i]),xy=(median_rv[i],y_loc),xytext=(median_rv[i],y_loc+binned_maximums[i]))
                    for dd in data_sets[i][dt]:
                        ax.plot((dd,dd),(y_loc-2,y_loc-7),color=self.colours[0])
                    i+=1
                
                ax.set_yticks(y_locations)
                ax.set_yticklabels(labels)
                ax.yaxis.set_tick_params(labelsize=fs)
                ax.xaxis.set_tick_params(labelsize=fs)
                
                ax.set_xlabel("Radial velocity [m/s]",fontsize=fs)
                ax.set_ylabel("Echelle order",fontsize=fs)
                
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.tick_params(axis='both', direction='out')
                ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
                ax.get_yaxis().tick_left()
                ax.set_ylim(-10,2.1*sum(binned_maximums))
            plt.show()

        for ax in self.axes:
            ax.legend()
    
        plt.show()
        return
    def get_plot_params(self,orders,**kwargs):
        fibre  = list(self.fibre)
        try:
            dtype = kwargs["dtype"]
        except:
            dtype = self.manager.dtype
        real1d = np.dtype([((dtype[i],list,1)) for i in range(np.size(dtype))])
        self.plot_params = np.empty(shape=np.shape(orders), dtype=np.dtype({"names":fibre, "formats":[real1d for f in fibre]}))
        lstyles = {"FLX":"-","ENV":"--","BKG":":","B2E":"-","FMB":"-"}
        lwidths = {"FLX":1.0,"ENV":2.0,"BKG":2.0,"B2E":1.5,"FMB":1.0}
        #print(fibre,type(orders),orders.shape,self.plot_params)
        for f in fibre:
            j=0
            for dt in dtype:
                k=0
                for i,o in enumerate(orders):
                    label = "{f} {d} {o}".format(f=f,d=dt,o=o)
                    #print(f,dt,o,"label=",label)
                    c=self.colours[i]
                    #print(lstyles[dt])
                    pargs = {"label":label, "c":c,"ls":lstyles[dt], "lw":lwidths[dt]}
                    
                    self.plot_params[f][dt][i] = pargs
                    del(pargs)
                    k+=2
                j+=5
        return