#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:38:31 2018

@author: dmilakov
"""
import harps.classes as hc
import harps.functions as hf
import harps.wavesol as ws
import numpy as np
import matplotlib.ticker as ticker, formatter
ow=False
#%% 17 April 2015 FOCES HARPS
# reference FOCES
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/series01/HARPS.2015-04-17T14_40_26.910_e2ds_B.fits',LFC='FOCES',overwrite=ow)
linelist1 = spec1['linelist']
# target HARPS
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/series03/HARPS.2015-04-17T18_32_17.039_e2ds_B.fits',LFC='HARPS',overwrite=ow)
linelist2 = spec2['linelist']


# coefficients 
#thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/series03/HARPS.2015-04-17T18_32_17.039_e2ds_B.fits',True)
#coeffs = thar.coeffs

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)
fig.suptitle("17 April 2015 FOCES HARPS")
#%% 17 April 2015 HARPS FOCES density = 0
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T14_40_26.910_e2ds_A.fits',True)
coeffs = thar.coeffs
# reference HARPS
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T23_27_33.430_e2ds_A.fits',LFC='HARPS',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target HARPS
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T14_40_26.910_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)
fig.suptitle("17 April 2015 HARPS FOCES")
#%% 14 April 2015 FOCES HARPS
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-14/HARPS.2015-04-14T13_54_07.234_e2ds_A.fits',True)
coeffs = thar.coeffs
# reference FOCES
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-16/HARPS.2015-04-16T20_03_38.389_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target HARPS
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-16/HARPS.2015-04-16T06_12_49.141_e2ds_A.fits',LFC='HARPS',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)
fig.suptitle("14 April 2015 FOCES HARPS")
#%% 04 December 2018
# reference HARPS
spec1 = hc.Spectrum('/Users/dmilakov/observations/HE0515-4414/102.A-0697/data/reduced/HARPS.2018-12-04T07:07:51.523_e2ds_A.fits',overwrite=ow)
linelist1 = spec1['linelist']
# target HARPS
spec2 = hc.Spectrum('/Users/dmilakov/observations/HE0515-4414/102.A-0697/data/reduced/HARPS.2018-12-05T06:23:04.730_e2ds_A.fits',overwrite=ow)
linelist2 = spec2['linelist']


# coefficients 
thar = ws.ThAr('/Users/dmilakov/observations/HE0515-4414/102.A-0697/data/reduced/HARPS.2018-12-05T06:23:04.730_e2ds_A.fits',True)
coeffs = thar.coeffs

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)
fig.suptitle("04-05 December 2018 HARPS HARPS")
#%% 14 April 2015 HARPS HARPS
# reference
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-16/HARPS.2015-04-16T06_09_46.288_e2ds_A.fits',LFC='HARPS',overwrite=ow)
linelist1 = spec1['linelist']
# target
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-16/HARPS.2015-04-16T06_12_49.141_e2ds_A.fits',LFC='HARPS',overwrite=ow)
linelist2 = spec2['linelist']


# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-14/HARPS.2015-04-14T13_54_07.234_e2ds_A.fits',True)
coeffs = thar.coeffs

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)

fig.suptitle("14 April 2015 HARPS HARPS")
#%% 14 April 2015 FOCES FOCES
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-14/HARPS.2015-04-14T13_54_07.234_e2ds_A.fits',True)
coeffs = thar.coeffs

# reference
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-16/HARPS.2015-04-16T20_00_35.416_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-16/HARPS.2015-04-16T20_03_38.389_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)
fig.suptitle("14 April 2015 FOCES FOCES")

#%% 17 April 2015 FOCES FOCES
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T14_46_31.541_e2ds_A.fits',True)
coeffs = thar.coeffs
# reference FOCES
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T14_46_31.541_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target HARPS
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T14_51_36.460_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)
fig.suptitle("17 April 2015 FOCES FOCES density = 0.0")
#%% 17 April 2015 FOCES FOCES density = 0
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_20_27.593_e2ds_A.fits',True)
coeffs = thar.coeffs
# reference FOCES
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_20_27.593_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target FOCES
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_23_29.993_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)

fig.suptitle("17 April 2015 FOCES FOCES density = 0.0")

#%% 17 April 2015 FOCES FOCES density = 0.5
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_20_27.593_e2ds_A.fits',True)
coeffs = thar.coeffs
# reference FOCES
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_25_19.269_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target FOCES
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_28_29.791_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)
fig.suptitle("17 April 2015 FOCES FOCES density = 0.5")
#%% 17 April 2015 FOCES FOCES density = 1.0
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T14_46_31.541_e2ds_A.fits',True)
coeffs = thar.coeffs
# reference FOCES
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_31_22.511_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target FOCES
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_34_32.712_e2ds_A.fits',LFC='FOCES',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

#linelist_int=interpolate2d(linelist1,linelist2)
#fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig,ax = plot_rv(linelist1,linelist2)
fig.suptitle("17 April 2015 FOCES FOCES density = 1.0")
#%% 17 April 2015 HARPS HARPS density = 0
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_20_27.593_e2ds_B.fits',True)
coeffs = thar.coeffs
# reference HARPS
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_20_27.593_e2ds_B.fits',LFC='HARPS',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target HARPS
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_23_29.993_e2ds_B.fits',LFC='HARPS',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

linelist_int=interpolate2d(linelist1,linelist2)
fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig.suptitle("17 April 2015 HARPS HARPS density = 0.0")
#%% 17 April 2015 HARPS HARPS density = 0.5
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_20_27.593_e2ds_B.fits',True)
coeffs = thar.coeffs
# reference HARPS
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_25_19.269_e2ds_B.fits',LFC='HARPS',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target HARPS
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_28_29.791_e2ds_B.fits',LFC='HARPS',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

linelist_int=interpolate2d(linelist1,linelist2)
fig,ax = plot_rv_lambda(linelist_int,linelist2,coeffs)
fig.suptitle("17 April 2015 HARPS HARPS density = 0.5")
#%% 17 April 2015 HARPS HARPS density = 1.0
# coefficients 
thar = ws.ThAr('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T14_46_31.541_e2ds_B.fits',True)
coeffs = thar.coeffs
# reference HARPS
spec1 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_31_22.511_e2ds_B.fits',LFC='HARPS',overwrite=ow)
spec1.ThAr = thar
linelist1 = spec1['linelist']
# target HARPS
spec2 = hc.Spectrum('/Volumes/dmilakov/harps/data/2015-04-17/HARPS.2015-04-17T20_34_32.712_e2ds_B.fits',LFC='HARPS',overwrite=ow)
spec2.ThAr = thar
linelist2 = spec2['linelist']

fig,ax = plot_rv(linelist1,linelist2)
fig.suptitle("17 April 2015 HARPS HARPS density = 1.0")


#%%

linelist_int=interpolate2d(linelist1,linelist2)
plot_rv_lambda(linelist_int,linelist2,coeffs)


#%%
def extract_cen_freq(linelist,fittype):
    return linelist[fittype][:,1], linelist['freq'], linelist['noise']

def remove_nans(array):
    return array[~np.isnan(array)]

def interpolate1d(comb1lines,comb2lines,fittype='gauss'):
    """
    Returns the interpolated frequencies and the photon noise of COMB2 lines 
    from the known positions of COMB2 lines. Uses the frequencies and positions 
    of COMB1 lines to do the interpolation.
    """
    def interpolate_freq(x,nx,index):
        """
        Returns the interpolated frequency and the photon noise of a single 
        line. 
        
        Input: 
        -----
            x (float)   : position of COMB2 line in pixels
            nx (flaot)  : photon noise of the COMB2 line
            index (int) : index of the COMB1 line that is to the right of the 
                          COMB2 line 
        """
        
        if index > 0 and index < len(freq1):
            f1 = freq1[index-1]
            x1 = cen1[index-1]
            f2 = freq1[index]
            x2 = cen1[index]
            
            # Two point form of a line passing through (x1,f1) and (x2,f2):
            # f(x) =  f2 + (x-x2)*(f1-f2)/(x1-x2)
            f_int = f2 + (x-x2)*(f1-f2)/(x1-x2)
            
            # Noise is the square root of the sum of variances
            n1 = noise1[index-1]
            n2 = noise1[index]
            noise = np.sqrt(n1*n1 + n2*n2 + nx*nx)
            # 
            #print((n1/(x1-x2)**2)**2,(n2*(x-x1)/(x1-x2)**2)**2,(nx*(f1-f2)/(x1-x2))**2)
            #noise = np.sqrt((n1/(x1-x2)**2)**2 +\
            #                (n2*(x-x1)/(x1-x2)**2)**2)
        else:
            f_int = np.nan
            noise = np.nan
        return f_int, noise
    
    # COMB1 is used for interpolation of COMB2 lines
    cen1, freq1, noise1 = extract_cen_freq(comb1lines,fittype)
    cen2, freq2, noise2 = extract_cen_freq(comb2lines,fittype)
    
    # COMB2 lines are binned into bins defined by the positions of COMB1 lines
    left_of = np.digitize(cen2,cen1,right=False)
    
    # interpolate COMB2 frequencies from the positions of COMB1 and COMB2 lines
    freq_int, noise_int = np.transpose([interpolate_freq(c,n,i) \
                                        for c,n,i in zip(cen2,noise2,left_of)])
    
#    v=calculate_rv(freq2,freq_int)
#    plt.figure(figsize=(10,4))
#    plt.plot(v,marker='X',ms=5,c='C2',lw=0.3)
    return freq_int, noise_int
def interpolate2d(comb1lines,comb2lines,fittype='gauss'):
    
    minord = np.min(tuple(np.min(f['order']) for f in [comb1lines,comb2lines]))
    maxord = np.max(tuple(np.max(f['order']) for f in [comb1lines,comb2lines]))
    
    interpolated_freq  = np.full(len(comb2lines),np.nan)
    interpolated_noise = np.full(len(comb2lines),np.nan)
    for order in range(minord,maxord+1,1):
        inord1 = np.where(comb1lines['order']==order)[0]
        inord2 = np.where(comb2lines['order']==order)[0]
        #print(order,len(inord1),len(inord2))
        freq_int, noise_int =interpolate1d(comb1lines[inord1],
                              comb2lines[inord2],
                              fittype)
#        wait=input('press key')
        interpolated_freq[inord2] = freq_int
        interpolated_noise[inord2] = noise_int
    return interpolated_freq, interpolated_noise

def calculate_shift(shifts,noises):
    
    shift = np.nansum(shifts / noises**2) / np.nansum(1 / noises**2)
    noise = 1./ np.sqrt(np.nansum(1./noises**2))
    
    return shift, noise
#def plot_rv_pix(comb1lines,comb2lines,fittype='gauss'):
#    cen1, freq1 = extract_cen_freq(comb1lines,fittype)
#    cen2, freq2 = extract_cen_freq(comb2lines,fittype)
#    
#    fig, ax = hf.figure(2,figsize=(12,8))
#    v = calculate_rv(freq1,freq2)
#    ax[0].plot((cen2-cen1)*829,marker='o',c="C0",
#                label='Centers',ms=3,lw=0.3)
##    ax[1].plot(299792458*(freq2-freq1)/freq1,marker='o',c="C1",
##              label='Frequencies',lw=0.3)
#
#    return fig,ax
def plot_rv(comb1lines,comb2lines,fittype='gauss'):
    tru_cent, tru_freq, tru_noise  = extract_cen_freq(comb2lines,fittype)
    int_freq, int_noise = interpolate2d(comb1lines,comb2lines,fittype)
    shifts = hf.removenan(c*(tru_freq-int_freq)/tru_freq)
    noises = hf.removenan(int_noise)
    m = hf.sigclip1d(shifts)
#    m = np.arange(len(shifts))
    print(sum(m),len(m))
    rv, noise = calculate_shift(shifts[m],noises[m])
    
    fig,ax = hf.figure(1,figsize=(12,6))
    
    y = shifts[m]
    x = np.arange(len(y))
    print(np.shape(x),np.shape(y))
    ax[0].errorbar(x,y,yerr=noises[m],
                  marker='o',ms=3,lw=0.3,capsize=3)
    orderbreak=np.where(np.diff(comb2lines['order'])==1)[0]
    [ax[0].axvline(i,ls=':',lw=0.5) for i in orderbreak]
    
    ax[0].text(0.1,0.1,"RV=${0:5.3f}\pm{1:5.3}$ m/s".format(rv,noise),
               transform=ax[0].transAxes)
    return fig,ax

def compare(spec,refspec,fittype='gauss'):
    comb1lines = refspec['linelist']
    comb2lines = spec['linelist']
    
    true_cent, true_freq, true_noise  = extract_cen_freq(comb2lines,fittype)
    int_freq, int_noise = interpolate2d(comb1lines,comb2lines,fittype)
    
    shift = hf.removenan(c*(true_freq-int_freq)/true_freq)
    noise = hf.removenan(int_noise)
    
    m = hf.sigclip1d(shifts)
    
    rv, rv_sigma = calculate_shift(shift[m],noise[m])
    
    return rv, rv_sigma
#%%

def plot_rv_lambda(comb1lines,comb2lines,coefficients,
                   fittype='gauss',*args,**kwargs):
    
    """
    comblines1: interpolated
    comblines2: measured
    """
    def idx2lbd(index,pos):
        lbd = hf.freq_to_lambda(comb2lines['freq'][pos])
        return "{0:8.3f}".format(lbd)

    cen1, freq1 = extract_cen_freq(comb1lines,fittype)
    cen2, freq2 = extract_cen_freq(comb2lines,fittype)
    
    lambda1, lamerr1 = calculate_lambda(comb1lines,coefficients,fittype)
    lambda2, lamerr2 = calculate_lambda(comb2lines,coefficients,fittype)

    rv_center = ((cen2-cen1)*829)
    rv_lambda = remove_nans(299792458*(lambda2-lambda1)/lambda1)
    rv_freq   = calculate_rv(comb1lines['freq'],comb2lines['freq'])
    
    numlines = len(comb2lines)
    
    fig, ax = hf.figure(3,figsize=(14,9),sharex=True,sharey=True,sep=0.1,
                        *args,**kwargs)
   
    ax[0].set_title("Centers",loc='left',fontsize=10)
    ax[0].plot(rv_center,marker='o',c="C0",
                label='Centers',ms=2,lw=0.3)
#    ax[0].set_xlim(2000,4000)
#    ax[0].set_ylim(-50,50)

    ax[1].set_title("Wavelengths",loc='left',fontsize=10)
    
    ax[1].plot(rv_lambda,marker='o',c="C0",
                label='Wavelengths',ms=2,lw=0.3)

#    ax[1].set_xlim(0,numlines)
#    ax11=ax[1].twiny()
#    ax11.plot(rv_center,ls='')
#    ax11.xaxis.set_major_locator(ticker.IndexLocator(400,1))
#    ax11.xaxis.set_minor_locator(ticker.MultipleLocator(20))
#    ax11.xaxis.set_major_formatter(ticker.FuncFormatter(idx2lbd))
#    ax11.xaxis.set_minor_formatter(ticker.FuncFormatter(idx2lbd))
#    
#    print(ax11.get_xticks())
#    lineind = np.arange(0,numlines,100)
#    lambdas = hf.freq_to_lambda(comb2lines['freq'][lineind])
#    ax11_xlim = ax[1].get_xlim()
#    print(ax11_xlim)
    
#    ax11.set_xticks(lineind)
#    ax11.set_xticklabels(lambdas)
#    
    ax[2].set_title("Frequencies",loc='left',fontsize=10)
    ax[2].plot(rv_freq,marker='o',c="C1",
              label='Frequencies',lw=0.3)
    
    orderbreak=np.where(np.diff(comb1lines['order'])==1)[0]
    [[a.axvline(i,ls=':',lw=0.5) for i in orderbreak] for a in ax]
    
    fig.text(0.01,0.5,"RV [m/s]",verticalalignment='center',rotation=90,
             fontsize=12)
    ax[-1].set_xlabel("Line number")
    return fig,ax
#%%
def calculate_rv(frequencies1,frequencies2):
    # Frequencies in GHz
    return 299792458e0*(frequencies2-frequencies1)/frequencies2
def calculate_lambda(comblines,coefficients,fittype='gauss'):
    """
    Coeffiecents should be dtype=containers.coeffs
    """
    lineord = comblines['order']
    centers = comblines[fittype][:,1]
    centerr = comblines['{ft}_err'.format(ft=fittype)][:,1]
    l       = np.zeros(len(centers))
    dl      = np.zeros(len(centers))
    polyord = np.shape(coeffs['pars'])[-1]
    for coeff in coefficients:
        order = coeff['order']
        if order<43:
            #print("Continuing")
            continue
        pars  = coeff['pars']
        order = coeff['order']
        pixl  = coeff['pixl']
        pixr  = coeff['pixr']
        cut   = np.where((comblines['order']==order) & 
                         (centers >= pixl) &
                         (centers <= pixr))[0]
        if len(cut)==0:
            continue
        cen   = centers[cut]
        cerr  = centerr[cut]
        l[cut] = np.ravel(np.sum([pars[i]*(cen[:,np.newaxis]**i) \
                                   for i in range(polyord)],axis=0))
        
        dl[cut] = np.ravel(np.sum([pars[i+1]*(cen[:,np.newaxis]**i) \
                                   for i in range(polyord-1)],axis=0))*cerr
    return l, dl
#%%
linelist1=lines['B'][0]
linelist2=lines['B'][1]
linelist_int=interpolate2d(linelist1,linelist2)
plot_rv_lambda(linelist_int,linelist2,coeffs)