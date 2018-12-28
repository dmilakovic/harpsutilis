#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 19:35:58 2018

@author: dmilakov

Input: two line lists
"""
from harps.core import np, plt
from harps.constants import c
import harps.functions as hf

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
def extract_cen_freq(linelist,fittype):
    return linelist[fittype][:,1], linelist['freq']

def calc_lambda(linelist,coeffs,fittype='gauss'):
    """
    Coeffiecents should be ThAr or Comb
    """
    lineord = linelist['order']
    centers = linelist[fittype][:,1]
    centerr = linelist['{ft}_err'.format(ft=fittype)][:,1]
    l       = np.zeros(len(centers))
    dl      = np.zeros(len(centers))
    polyord = np.shape(coeffs['pars'])[-1]
    for coeff in coeffs:
        order = coeff['order']
        if order<43:
            continue
        inord = np.where(lineord==order)[0]
        if len(inord)==0:
            continue
        pars  = coeff['pars']
        cen   = centers[inord]
        cerr  = centerr[inord]
        if len(np.shape(pars))!=1:
            pars = pars[0]
        l[inord] = np.ravel(np.sum([pars[i]*(cen[:,np.newaxis]**i) \
                                   for i in range(polyord)],axis=0))
        
        dl[inord] = np.ravel(np.sum([pars[i+1]*(cen[:,np.newaxis]**i) \
                                   for i in range(polyord-1)],axis=0))*cerr
    return l, dl
def rv(linelist1,linelist2,fittype='gauss',kind='pixel',coeffs=None):
    
    index1 = get_index(linelist1)
    index2 = get_index(linelist2)
    sel1,sel2 = get_sorted(index1,index2)
    
    if kind == 'pixel':
        cent1, freq1 = extract_cen_freq(linelist1,fittype)
        cent2, freq2 = extract_cen_freq(linelist2,fittype)
        v=829*(cent1[sel1]-cent2[sel2])
    else:
        assert coeffs!=None, "Provide coefficients"
        wav1,dwav1 = calc_lambda(linelist1,coeffs,fittype)
        wav2,dwav2 = calc_lambda(linelist2,coeffs,fittype)
        v=c*(wav1[sel1]-wav2[sel2])/(wav1[sel1])
            
        dwav = c*np.sqrt((dwav2[sel2]/wav2[sel2])**2+\
                         (dwav1[sel1]/wav1[sel1])**2)
        
        m=hf.sigclip1d(v,plot=True)
        
        global_dv     = np.sum(v[m]/(dwav[m])**2)/np.sum(1/(dwav[m])**2)
        global_sig_dv = (np.sum(1/(dwav[m])**2))**(-0.5)  
        print("shift {0:7.5f} \t pn {1:7.5f}".format(global_dv,global_sig_dv))
    return v

#%%
import harps.classes as hc
import harps.wavesol as ws
spec1=hc.Spectrum('/Users/dmilakov/observations/HE0515-4414/102.A-0697/data/reduced/HARPS.2018-12-11T07:19:58.539_e2ds_A.fits')
spec2=hc.Spectrum('/Users/dmilakov/observations/HE0515-4414/102.A-0697/data/reduced/HARPS.2018-12-11T08:52:09.513_e2ds_A.fits')
cc1=ws.Comb(spec1,500).get_wavecoeff_comb()
cc2=ws.Comb(spec2,500).get_wavecoeff_comb()
tc1=ws.ThAr(spec1.filepath,True).coeffs
tc2=ws.ThAr(spec2.filepath,True).coeffs
#%%
lines1=spec1['linelist']
lines2=spec2['linelist']
#%%
#kind = 'pixel'
kind = 'wave'
v = rv(lines1,lines2,kind=kind,coeffs=tc1)
fig,ax=hf.figure(1,figsize=(12,5),title="RV shift {}".format(kind))
ax[0].plot(v,marker='o',ms=3)