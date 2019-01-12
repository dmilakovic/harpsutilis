#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:59:42 2019

@author: dmilakov
"""

import harps.series as ser
from   harps.core import np, curve_fit,os
import harps.plotter as plot
import harps.functions as hf
#%%
seriesA_2015=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.6/scigarn/2015-04-17_fibreA_v501_series0103.dat','A',refindex=0,version=500)
seriesB_2015=ser.Series('/Users/dmilakov/harps/dataprod/output/'
                   'v_0.5.6/scigarn/2015-04-17_fibreB_v501_series0103.dat','B',refindex=0,version=500)
#%%
waveA_2015 = seriesA_2015.wavesol()
waveB_2015 = seriesB_2015.wavesol()

colors = {'A':'C0','B':'C1','A-B':'C2'}
data = {'A':waveA_2015, 'B':waveB_2015}
plotter = plot.Figure(3,figsize=(16,9),sharex=True,sharey=[True,True,False],
                      ratios=[5,5,2],top=0.92)
diff_uncorrected = data['A']-data['B']
diff_uncorrected.plot(plotter=plotter,label='A-B',axnum=0,c=colors['A-B'])

for f in ['A','B']:
    data[f].plot(plotter=plotter,axnum=0,label=f,
        c=colors[f])
    data[f].correct_cti(f)
    data[f].plot(plotter=plotter,axnum=1,label=f,
        c=colors[f])
    
diff_corrected = data['A']-data['B']
diff_corrected.plot(plotter=plotter,axnum=1,c=colors['A-B'],label='A-B')

handles,labels = plotter.axes[1].get_legend_handles_labels()
plotter.axes[0].legend(handles,labels,ncol=3,bbox_to_anchor=(0.5,1.3),
            loc='upper center',frameon=False)
plotter.axes[1].legend([],frameon=False)

text = ['Uncorrected','Corrected','Difference']
for i in range(3):
    ax = plotter.axes[i]
    ax.text(0.04,0.8,text[i],fontsize=10,horizontalalignment='left',
                       transform=ax.transAxes)
    if i!=2:
        step=0.5
        ax.hlines(np.arange(-0.5,1.+step,step),0,194,
                  linestyles='dotted',colors='k',lw=0.5)

(diff_corrected-diff_uncorrected).plot(plotter=plotter,axnum=2)

#specifier = '2015-04-shift_corrected'
figname   = '2015-04-shift_corrected.pdf'
folder    = '/Users/dmilakov/harps/dataprod/plots/CTI/'
plotter.fig.savefig(os.path.join(folder,figname))

#%%
plotter=plot.Figure(1,figsize=(16,8),bottom=0.12)
    
plotter.axes[0].plot(data['A']['flux'].values,marker='o')
plotter.axes[0].plot(data['B']['flux'].values,marker='s')
#%%