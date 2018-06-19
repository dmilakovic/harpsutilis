#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:07:36 2018

@author: dmilakov
"""
#%%
importlib.reload(hc)
spec=hc.Spectrum(manager.file_paths['A'][0],LFC='HARPS')
lines=spec.fit_lines(order=None,fittype='epsf')
spec2=hc.Spectrum(manager.file_paths['A'][1],LFC='HARPS')
lines2=spec2.fit_lines(order=None,fittype='epsf')
spec3=hc.Spectrum(manager.file_paths['A'][2],LFC='HARPS')
lines3=spec3.fit_lines(order=None,fittype='epsf')
#%%
importlib.reload(hc)
do_new=True
spec_to_use=[0,1,2,3]
#fibre_shape = 'round'
patches=False
if do_new:
    spectra = {}
    lines   = {}
    wavesols = {}
for i in spec_to_use:
    spec=hc.Spectrum(manager.file_paths['A'][i],LFC='HARPS')
    spec.fibre_shape = 'octogonal'
    if do_new == False:
        spec.lines=lines[i]
        spec.lineDetectionPerformed=True
        spec.lineFittingPerformed={'gauss':True,'epsf':True}
        fitlines = spec.lines
    else:
        fitlines=spec.fit_lines(order=None,fittype='epsf')
    ws=spec.__get_wavesol__('LFC',patches=patches)
    
    spectra[i]=spec
    lines[i]=fitlines
    wavesols[i]=ws
spectra_octogonal=spectra
lines_octogonal=lines
#%%
importlib.reload(hc)
patches=False
spec=hc.Spectrum(manager.file_paths['A'][0],LFC='HARPS')
spec.lines=lines
spec.lineDetectionPerformed=True
spec.lineFittingPerformed={'gauss':True,'epsf':True}
#lines=spec.fit_lines(order=None,fittype='epsf')
ws=spec.__get_wavesol__('LFC',patches=patches)

spec2=hc.Spectrum(manager.file_paths['A'][1],LFC='HARPS')
spec2.lines=lines2
spec2.lineDetectionPerformed=True
spec2.lineFittingPerformed={'gauss':True,'epsf':True}
#lines2=spec2.fit_lines(order=None,fittype='epsf')
ws2=spec2.__get_wavesol__('LFC',patches=patches)

spec3=hc.Spectrum(manager.file_paths['A'][2],LFC='HARPS')
spec3.lines=lines3
spec3.lineDetectionPerformed=True
spec3.lineFittingPerformed={'gauss':True,'epsf':True}
#lines3=spec3.fit_lines(order=None,fittype='epsf')
ws3=spec3.__get_wavesol__('LFC',patches=patches)
#%% PLOT RESIDUALS OF INTERPOLATED vs KNOWN LINE POSITIONS 
spectra_to_use=spectra_octogonal
orders = [55]
for order in orders:
    p1=hc.SpectrumPlotter(naxes=2,sharex=True,sharey=True)
    for i,ft in enumerate(['epsf','gauss']):
        for s in spectra_to_use.values():
            s.plot_residuals(order=order,fittype=ft,title=ft,mean=False,plotter=p1,axnum=i,photon_noise=True)
        p1.axes[i].set_title(ft)
 #%% PLOT DIFFERENCE BETWEEN PSF AND GAUSSIAN CENTERS
spectra_to_use=spectra_octogonal
orders = [46]
for order in orders:
    p1=hc.SpectrumPlotter(naxes=2,sharex=True,sharey=True)
#    for i,ft in enumerate(['epsf','gauss']):
    cen_gauss_list = []
    cen_psf_list = []
    for s in spectra_to_use.values():
        cen_gauss = s.lines['pars'].sel(ft='gauss',od=order,par='cen')
        cen_psf   = s.lines['pars'].sel(ft='epsf',od=order,par='cen')
        p1.axes[0].scatter(cen_psf,(cen_psf-cen_gauss),s=1)
#        trd_moment = s.lines['line'].
        cen_gauss_list.append(cen_gauss)
        cen_psf_list.append(cen_psf)
#        trd_moment_list.append(trd_moment)
    average = np.nanmean((np.array(cen_gauss_list) - np.array(cen_psf_list)),axis=0)
    phase = [cen_psf_list[0][i]-int(cen_psf_list[0][i]) for i in range(345)]
    p1.axes[1].scatter(phase,-average[:345],s=1)
#%% 
ft='epsf'
orders = np.arange(45,72)
#orders=[50]
fig,axes=hf.get_fig_axes(1)
for od in orders:
    axes[0].scatter(lines0['pars'].sel(ft=ft,od=od,par='cen'),
    (lines0['pars'].sel(ft=ft,od=od,par='cen')-lines2['pars'].sel(ft=ft,od=od,par='cen')),s=1)
    [axes[0].axvline(512*i,ls=':',lw=0.3) for i in range(9)]
axes[0].set_ylabel('$\Delta$ center [pix]')
axes[0].set_xlabel('center [pix]')    
    
#%% PLOT SPECTRA AND DIFFERENCE IN CENTERS
fig,axes=hf.get_fig_axes(3,sharex=True)
for lid in range(300):
    axes[0].plot(lines0['line'].sel(od=50,id=lid,ax='pix'),lines0['line'].sel(od=50,id=lid,ax='flx'),ls='-',c='C0',marker='o')

    axes[1].plot(lines2['line'].sel(od=50,id=lid,ax='pix'),lines2['line'].sel(od=50,id=lid,ax='flx'),c='C2',marker='x',ls=':')
axes[2].scatter(lines0['pars'].sel(ft=ft,od=od,par='cen'),
    (lines0['pars'].sel(ft=ft,od=od,par='cen')-lines2['pars'].sel(ft=ft,od=od,par='cen')),s=1)

#%% PLOT DIFFERENCE IN BARYCENTERS
fig,axes=hf.get_fig_axes(4,sharex=True,figsize=(10,10))
for lid in range(300):
    axes[0].plot(lines0['line'].sel(od=50,id=lid,ax='pix'),lines0['line'].sel(od=50,id=lid,ax='flx'),ls='-',c='C0',marker='o')

    axes[1].plot(lines2['line'].sel(od=50,id=lid,ax='pix'),lines2['line'].sel(od=50,id=lid,ax='flx'),c='C2',marker='x',ls=':')
    axes[2].plot(lines3['line'].sel(od=50,id=lid,ax='pix'),lines3['line'].sel(od=50,id=lid,ax='flx'),c='C3',marker='^',ls=':')
axes[3].scatter(lines['attr'].sel(od=od,att='bary'),
    (lines0['attr'].sel(od=od,att='bary')-lines2['attr'].sel(od=od,att='bary')),s=1,c='C2')
axes[3].scatter(lines['attr'].sel(od=od,att='bary'),
    (lines0['attr'].sel(od=od,att='bary')-lines3['attr'].sel(od=od,att='bary')),s=1,c='C3')
axes[3].set_ylabel('$\Delta$ barycenter [pix]')
axes[3].set_xlabel('Barycenter [pix]')
#%% PLOT DIFFERENCE IN SHIFTS
fig,axes=hf.get_fig_axes(1)
axes[0].scatter(lines0['pars'].sel(ft=ft,od=od,par='cen'),
    (lines0['pars'].sel(ft=ft,od=od,par='shift')-lines2['pars'].sel(ft=ft,od=od,par='shift')),s=1)
axes[0].scatter(lines0['pars'].sel(ft=ft,od=od,par='cen'),
    (lines0['pars'].sel(ft=ft,od=od,par='shift')-lines3['pars'].sel(ft=ft,od=od,par='shift')),s=1)
axes[3].set_ylabel('$\Delta$ shift [pix]')
axes[3].set_xlabel('Center [pix]')
#%% PLOT DIFFERENCE IN CENTERS, DIFFERENCE IN BARYCENTERS AND DIFFERENCE IN SHIFTS
ft = 'epsf'
fig,axes=hf.get_fig_axes(3,sharex=True)
#orders = [50]
def update_yscale(ax1,ax2):
    y1,y2 = ax1.get_ylim()
    ax2.set_ylim(y1*829,y2*829)
    ax2.figure.canvas.draw()
    return

axes[0].scatter(lines0['pars'].sel(ft=ft,par='cen'),
    (lines0['pars'].sel(ft=ft,par='cen')-lines2['pars'].sel(ft=ft,par='cen')),s=1)
axes[0].scatter(lines0['pars'].sel(ft=ft,par='cen'),
    (lines0['pars'].sel(ft=ft,par='cen')-lines3['pars'].sel(ft=ft,par='cen')),s=1)
[axes[0].axvline(512*i,ls=':',lw=0.3) for i in range(9)]
  
axes[1].scatter(lines0['pars'].sel(ft=ft,par='cen'),
        (lines0['attr'].sel(att='bary')-lines2['attr'].sel(att='bary')),s=1,c='C0')
axes[1].scatter(lines0['pars'].sel(ft=ft,par='cen'),
        (lines0['attr'].sel(att='bary')-lines3['attr'].sel(att='bary')),s=1,c='C1')
    
axes[2].scatter(lines0['pars'].sel(ft=ft,par='cen'),
        (lines0['pars'].sel(ft=ft,par='shift')-lines2['pars'].sel(ft=ft,par='shift')),s=1)
axes[2].scatter(lines0['pars'].sel(ft=ft,par='cen'),
        (lines0['pars'].sel(ft=ft,par='shift')-lines3['pars'].sel(ft=ft,par='shift')),s=1)
axes[0].set_ylabel('$\Delta$ center [pix]')
axes[1].set_ylabel('$\Delta$ barycenter [pix]')
axes[2].set_ylabel('$\Delta$ shift [pix]')
axes[2].set_xlabel('Center [pix]')