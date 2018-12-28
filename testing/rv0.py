#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:39:20 2018

@author: dmilakov
"""

import harps.classes as hc
import harps.functions as hf
import harps.fit as fit
import harps.io as io
import harps.plotter as hplt
import harps.containers as container
import scipy.stats as stats
import harps.wavesol as ws
import harps.compare as compare

from harps.core import np, os
from harps.core import FITS

from harps.constants import c

outfile = '/Users/dmilakov/harps/dataprod/output/v_0.5.5/combined_scigarn.txt'
outlist = np.sort(io.read_textfile(outfile))
  

wavesols = {'A':[],'B':[]}
files    = {'A':[],'B':[]}
lines    = {'A':[],'B':[]}

#wavelist = []
#%%
for i,file in enumerate(outlist):
    print("Filename = ",file)
    fibre = os.path.splitext(os.path.basename(file))[0][-1]
    with FITS(file,'r') as fits:        
        hdu1 = fits['wavesol_comb',500]
        wavs   = hdu1.read()
        hdu2 = fits['linelist']
        l    = hdu2.read()
    wl = wavesols[fibre]
    wl.append(wavs)
    
    fl = files[fibre]
    fl.append(file)
    
    
    ll = lines[fibre]
    ll.append(l)
#%% 
kind = 'wavesol'
refindex = 0
#%%   W A V E S O L
plot2d = False
shift_wave= {'A':None,'B':None}
for fibre in ['A','B']:
    wavefib  = np.array(wavesols[fibre])
    wsrv     = container.radial_velocity(len(wavefib))
    mask     = np.zeros_like(wavefib)
    mask[:,48,:]=1
    wavesol  = np.ma.array(wavefib,mask=mask)
    wavesol  = wavefib
    # take only orders 43 and up
    wavesol  = wavesol[:,43:,:]
    waveref  = wavesol[refindex]
    # RV shift in pixel values
    wavediff = (waveref - wavesol)/waveref * c
    
    for i,file in enumerate(files[fibre]):
        print(i,file,fibre)
        #fibre    = os.path.splitext(os.path.basename(file))[0][-1]
        datetime = hf.basename_to_datetime(str(file))
        
        wave_exp = np.ma.masked_where(np.abs(wavediff[i])>4,
                                             wavediff[i])
        
        #clipped  = stats.sigmaclip(wave_exp).clipped
        average_rv = np.average(wave_exp)
        print(average_rv)
        wsrv[i]['rv'] = average_rv
        wsrv[i]['datetime'] = datetime
        wsrv[i]['fibre']=fibre
        
        if plot2d==True:
            fig,ax=hf.figure(1)
            ax0 = ax[0].imshow(wave_exp,aspect='auto',vmin=-40,vmax=40)
            fig.colorbar(ax0)
    shift_wave[fibre] = wsrv
        
#%%    L I N E S

def get_index(linelist,fittype='gauss'):
    fac = 10000
    MOD = 2.
    
    centers = linelist[fittype][:,1]
    orders  = linelist['order']*fac
    cround  = np.around(centers/MOD)*MOD
    cint    = np.asarray(cround,dtype=np.int)
    index0  = orders+cint
    return index0

def get_sorted(index1,index2):
    #print('len indexes',len(index1),len(index2))
    # lines that are common for both spectra
    intersect=np.intersect1d(index1,index2)
    intersect=intersect[intersect>0]

    indsort=np.argsort(intersect)
    
    argsort1=np.argsort(index1)
    argsort2=np.argsort(index2)
    
    sort1 =np.searchsorted(index1[argsort1],intersect)
    sort2 =np.searchsorted(index2[argsort2],intersect)
    
    return argsort1[sort1],argsort2[sort2]

def calc_lambda(linelist,coeffs,fittype='gauss'):
    """
    Coeffiecents should be ThAr
    """
    lineord = linelist['order']
    centers = linelist[fittype][:,1]
    centerr = linelist['{ft}_err'.format(ft=fittype)][:,1]
    l       = np.zeros(len(centers))
    dl      = np.zeros(len(centers))
    polyord = np.shape(coeffs['pars'])[-1]
    for coeff in coeffs:
        order = coeff['order']
        
        inord = np.where(lineord==order)[0]
        if len(inord)==0:
            continue
        pars  = coeff['pars']
        order = coeff['order']
        segm  = coeff['segm']
        pixl  = coeff['pixl']
        pixr  = coeff['pixr']
        cut   = np.where((linelist['order']==order) & 
                         (centers >= pixl) &
                         (centers <= pixr))
        cen   = centers[cut]
        cerr  = centerr[cut]
        print(segm,pixl,pixr,np.shape(cut))

        l[cut] = np.ravel(np.sum([pars[i]*(cen[:,np.newaxis]**i) \
                                   for i in range(polyord)],axis=0))
        
        dl[cut] = np.ravel(np.sum([pars[i+1]*(cen[:,np.newaxis]**i) \
                                   for i in range(polyord-1)],axis=0))*cerr
    return l, dl
#%%
def interpolate_2combs(linelist1,linelist2,order,fittype='gauss',plot=False):
    """
    Interpolates between two comb linelists
    
    linelist1 : known
    linelist2 : to be interpolated
    """
    def extract_cen_freq(linelist,fittype):
        return linelist[fittype][:,1], linelist['freq']
    
    def interpolate(freq,index):
        if index > 0 and index < len(freq1):
            f1 = freq1[index-1]
            x1 = cen1[index-1]
            f2 = freq1[index]
            x2 = cen1[index]
            
#            x_int = x1+(freq-f1)*(x2-x1)/(f2-f1)
            x_int = x2+(freq-f2)*(x1-x2)/(f1-f2)
#            pars    = np.polyfit(x=[x_left,x_right],
#                                 y=[f_left,f_right],deg=1)
#            print("{0:5d}{1:>10.1f}{2:>10.3f}\t{3}".format(index,f_left-f_right,
#                                                      x_left-x_right,pars))
#            f_int   = np.polyval(pars,cen)
        else:
            x_int = np.nan
        return x_int
    
    inord1 = np.where(linelist1['order']==order)[0]
    inord2 = np.where(linelist2['order']==order)[0]        
    cen1, freq1 = extract_cen_freq(linelist1[inord1],fittype)
    cen2, freq2 = extract_cen_freq(linelist2[inord2],fittype)
    noise1  = linelist1[inord1]['noise']
    noise2  = linelist2[inord2]['noise']
    
    
    left_of = np.digitize(freq2,freq1,right=False)
    cen_int = np.array([interpolate(f,i) for f,i in zip(freq2,left_of)])
    #v = c*(freq2-freq_int)/freq2
    v0     = 829*(cen2-cen_int)
    notnan = ~np.isnan(v0)
    v      = v0[notnan]
    if plot:
        fig, ax = hf.figure(4,sharex=[True,True,False,False],
                            alignment='grid',ratios=[[3,1],[3,1]])
        ax[0].set_title('ORDER = {}'.format(order))
        ax[0].plot(freq1,cen1,marker='o',c="C0",label='LFC1')
        ax[0].plot(freq2,cen2,marker='s',c="C1",label='LFC2',ls='')
        ax[0].plot(freq2,cen_int,marker='X',c='C1',ms=5,label='INT',ls='')
        [ax[0].axhline(i*512,ls=':',lw=0.5,c='k') for i in range(9)]
        
        ax[1].scatter(freq2[notnan],v,c='C1',marker='x',s=20)
        #[ax[1].axhline(i*512,ls=':',lw=0.5,c='k') for i in range(9)]
        
        ax[2].hist(v)
        ax[3].text(0.5,0.8,'$\mu={0:8.5f}$ m/s'.format(np.average(v, weights=noise2[notnan])),
                   horizontalalignment='center')
        ax[3].text(0.5,0.5,'PN={0:8.5f} m/s'.format(1./np.sqrt(np.sum(1./noise2**2))),
                   horizontalalignment='center')
        
        ax[0].legend()
    return cen_int
#%%
def combsol_residuals(linelist1,linelist2,version=501,fittype='gauss'):
    """
    Calculates residuals of linelist2 to a wavelength solution derived from 
    the lines in linelist1
    """
    # Obtain wavelength calibration from LFC1 lines
    coefficients1 = fit.dispersion(linelist1,version,fittype)
    # Obtain positions and wavelengths of LFC2 lines
    centers2      = linelist2[fittype][:,1]
    wavelengths2  = hf.freq_to_lambda(linelist2['freq'])
    
    nlines        = len(linelist2)
    residuals     = container.residuals(nlines)
    for coeff in coefficients1:
        order = coeff['order']
        segm  = coeff['segm']
        pixl  = coeff['pixl']
        pixr  = coeff['pixr']
        cut   = np.where((linelist2['order']==order) & 
                         (centers2 >= pixl) &
                         (centers2 <= pixr))
        centsegm = centers2[cut]
        wavereal  = wavelengths2[cut]
        wavefit   = ws.evaluate(coeff['pars'],centsegm)
        residuals['order'][cut]=order
        residuals['segm'][cut]=segm
        residuals['residual'][cut]=(wavereal-wavefit)/wavereal*c
    residuals['gauss'] = centers2

    return residuals
#%%
def interpolate_comb(linelist1,linelist2,fittype='gauss'):
    
    minord  = np.min(tuple(np.min(f['order']) for f in [linelist1,linelist2]))
    maxord  = np.max(tuple(np.max(f['order']) for f in [linelist1,linelist2]))
    
    new_linelist = np.copy(linelist2)
    for order in range(minord,maxord+1,1):
        
        inord1 = np.where(linelist1['order']==order)[0]
        inord2 = np.where(linelist2['order']==order)[0]
        #print(order,len(inord1),len(inord2))
        cen_int=interpolate_2combs(linelist1,linelist2,order,fittype,plot=False)
#        wait=input('press key')
        new_linelist[fittype][:,1][inord2]= cen_int
    return new_linelist
#%%
refindex=0
shift_line= {'A':None,'B':None}

def remove_order(container,order):
    inorder = container['order']==order
    return container[~inorder]    

for fibre in ['A','B']:
    linfib    = np.array(lines[fibre])
    lnrv    = container.radial_velocity(len(wavefib))
    
#    tharsol = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/series01/'
#                    'HARPS.2015-04-17T'
#                    '15_19_59.010_e2ds_{fb}.fits'.format(fb=fibre),
#                    vacuum=True)
#    combsol = ws.Comb(hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/series01/'
#                    'HARPS.2015-04-17T'
#                    '15_19_59.010_e2ds_{fb}.fits'.format(fb=fibre)),
#                    version=500)
#    coeffs  = tharsol.coeffs
#    coeffs  = combsol.get_wavecoeff_comb()
    
    #coeffs   = remove_order(coeffs0,47)
    #linelist_ref = remove_order(linfib[refindex],47)
    linelist_ref = linfib[refindex]
#    coeffs = fit.dispersion(linelist_ref,501,'gauss')
#    wavref,dwref=calc_lambda(linelist_ref,coeffs)
    
    index_ref = get_index(linelist_ref)
    for i,file in enumerate(files[fibre]):
        if i == refindex:
            continue
        datetime = hf.basename_to_datetime(str(file))
        #linelist = remove_order(linfib[i],47)
        linelist = linfib[i]
#        wav1,dwav1=calc_lambda(linelist,coeffs)
        
#        index = get_index(linelist)
#        print(i,fibre)
#        if i>93 and fibre=='B':
##        if i>93:
#            linelist_int = interpolate_comb(linelist_ref,linelist)
#            isnan        = np.isnan(linelist_int['gauss'][:,1])
#            linelist_nonan = linelist_int[~isnan]
#            wav2,dwav2  = calc_lambda(linelist_int,coeffs)
#            index_int = get_index(linelist_int)
#            sel1,sel2 = get_sorted(index,index_int)
#            plotflag=True
#            v=c*(wav1[sel1]-wav2[sel2])/(wav1[sel1])
#        else:
#            wav2,dwav2  = wavref,dwref
#            index_ref = get_index(linelist_ref)
#            sel1,sel2 = get_sorted(index,index_ref)    
#            plotflag=False
#            v=c*(wav1[sel1]-wav2[sel2])/(wav1[sel1])
#        linelist_int = interpolate_comb(linelist_ref,linelist)
#        isnan        = np.isnan(linelist_int['gauss'][:,1])
#        linelist_nonan = linelist_int[~isnan]
#        wav2,dwav2  = calc_lambda(linelist_int,coeffs)
#        index_int = get_index(linelist_int)
#        sel1,sel2 = get_sorted(index,index_int)
#        plotflag=False
#        v=c*(wav1[sel1]-wav2[sel2])/(wav1[sel1])    
#        dwav=c*np.sqrt((dwav2[sel2]/wav2[sel2])**2+\
#                       (dwav1[sel1]/wav1[sel1])**2)
#        
        
        
#        if i!=refindex:
#            m=hf.sigclip1d(v,plot=False)
#        else:
#            m=np.arange(len(v))
#            
#        global_dv     = np.sum(v[m]/(dwav[m])**2)/np.sum(1/(dwav[m])**2)
#        global_sig_dv = (np.sum(1/(dwav[m])**2))**(-0.5)  
        global_dv, global_sig_dv = compare.two_linelists(linelist_ref,linelist)
        plotflag=False
        if plotflag:
            fig,ax = hf.figure(1,figsize=(12,6))
            ax[0].plot(v,marker='o',ms=2,ls='')
            ax[0].axhline(global_dv,c='r')
            ax[0].axhline(global_dv+global_sig_dv,c='r',ls='--')
            ax[0].axhline(global_dv-global_sig_dv,c='r',ls='--')
        print("FB={fb:<5s}EXP={exp:<5d}".format(fb=fibre,exp=i) + \
              "{t1:>8s}{rv:>8.3f}".format(t1="RV =",rv=global_dv) + \
              "{t2:>8s}{pn:>7.3f}".format(t2="PN =",pn=global_sig_dv))
        lnrv[i]['datetime'] = datetime
        lnrv[i]['rv'] = global_dv
        lnrv[i]['pn'] = global_sig_dv
        lnrv[i]['fibre'] = fibre
        
    shift_line[fibre] = lnrv

#kind='lines'
#kind='wavesol'
    
kinds = ['lines']
for kind in kinds:
    fig, ax = hf.figure(1)
    fig.suptitle(kind)
    if kind == "wavesol":
        array = shift_wave
    elif kind == 'lines':
        array = shift_line
    for fibre in ['A','B']:
        #cut  = np.where(shift['fibre']==fibre)
        data = array[fibre]['rv']
        ax[0].plot(data,label=fibre)
    #fig.axes[0].set_ylim(-1.5,1.5)   
    ax[0].plot(array['A']['rv']-array['B']['rv'])
    ax[0].axvline(refindex,ls=':',lw=0.8,c='k')
    ax[0].axhline(0,ls=':',lw=0.8,c='k')
    ax[0].legend()
    #fig.axes[0].set_ylim(-10,10)
