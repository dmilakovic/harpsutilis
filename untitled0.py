#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 12:13:25 2025

@author: dmilakov
"""
import numpy as np
import harps.spectrum as hs
#%%
spec_A=hs.HARPS('/Users/dmilakov/projects/lfc/data/harps/e2ds/2015-04-17/HARPS.2015-04-17T16:38:39.414_e2ds_A.fits',
                blazepath='/Users/dmilakov/projects/lfc/data/harps/e2ds/2015-04-17/blaze/HARPS.2015-04-17T22_36_13.842/HARPS.2015-04-17T22:36:13.842_blaze_A.fits',
                fr=25e9,f0=9.27e9,sOrder=39,overwrite=False,
                )
                
spec_B=hs.HARPS('/Users/dmilakov/projects/lfc/data/harps/e2ds/2015-04-17/HARPS.2015-04-17T16:38:39.414_e2ds_B.fits',
                blazepath='/Users/dmilakov/projects/lfc/data/harps/e2ds/2015-04-17/blaze/HARPS.2015-04-17T22_36_13.842/HARPS.2015-04-17T22:36:13.842_blaze_B.fits',
                fr=18e9,f0=5.7e9,sOrder=39,overwrite=False,
                )
linelist_A = spec_A['linelist']
linelist_B = spec_B['linelist']
#%%
runfile('/Users/dmilakov/software/harps/testing/run_echelle_analysis_zernike.py', 
        wdir='/Users/dmilakov/software/harps/testing', 
        args='/Users/dmilakov/projects/lfc/data/harps/raw/HARPS.2015-04-17T16:38:39.414.fits -d red -O /Users/dmilakov/projects/spectroperf/dataprod/ --plot_comparison --plot_heatmap --plot_detector --clobber')
#%%
from harps.twodim.analyzer import EchelleAnalyzer

lfc_filename = '/Users/dmilakov/projects/lfc/data/harps/raw/HARPS.2015-04-17T16:38:39.414.fits'
output_fits_path = '/Users/dmilakov/projects/spectroperf/dataprod/HARPS.2015-04-17T16:38:39.414_red_zernike_results.fits'
analyzer = EchelleAnalyzer(lfc_filename=lfc_filename, bias_filename=None)
analyzer.output_fits_path = output_fits_path
#%%
catalog = analyzer.read_peak_catalog(save_internally=True)
#%%
# --- Your original selection and part of the processing ---
choices = np.random.choice(catalog, size=40) # Note: np.random.choice on structured array returns a view

# --- Step 1: Calculate 'm' and 'y_transformed' for all items in 'choices' ---
# We need to add a new field for 'm' or create a temporary array
# For simplicity, let's create temporary arrays for m and y_transformed

m_values = np.zeros(len(choices), dtype=int)
y_transformed_values = np.zeros(len(choices), dtype=int)
w_values = np.zeros(len(choices), dtype=float)

for i, item in enumerate(choices):
    img_type_val = item['IMGTYPE']
    y_transformed_values[i] = 4096 - item['PEAK_Y'] # Calculate transformed Y

    if img_type_val == 0: # 'A'
        od_idx = item['ORDER_NUM'] + 45
        linelist = linelist_A
        if 0 <= od_idx < len(array1_mask):
            m_values[i] = array1_mask[od_idx]
        else:
            # Handle out-of-bounds index for array1_mask, e.g., assign a special value or skip
            m_values[i] = -1 # Or np.nan, or raise error
            print(f"Warning: Out-of-bounds index for A: od_idx={od_idx}, item={item}")
    elif img_type_val == 1: # 'B'
        od_idx = item['ORDER_NUM'] + 43
        linelist = linelist_B
        if 0 <= od_idx < len(array2_mask):
            m_values[i] = array2_mask[od_idx]
        else:
            # Handle out-of-bounds index for array2_mask
            m_values[i] = -1 # Or np.nan, or raise error
            print(f"Warning: Out-of-bounds index for B: od_idx={od_idx}, item={item}")
    cut = np.where( (linelist['optord']==m_values[i]) & (np.abs(linelist['gauss_pix'][:,1]-y_transformed_values[i])<=2) )[0]
    if len(cut): w_values[i]  = linelist[cut]['gauss_wav'][0][1]


# --- Step 2: Perform lexsort ---
# Sort keys are applied from right to left:
# Primary sort: IMGTYPE
# Secondary sort: m_values
# Tertiary sort: y_transformed_values
# So, lexsort input is (tertiary_key, secondary_key, primary_key)
sorter = np.lexsort((y_transformed_values, m_values, choices['IMGTYPE']))

# --- Iterate through the sorted choices ---
print("Sorted by IMGTYPE, then m, then y_transformed:")
print("Img |  m  | PeakX | PeakY | WAVE")
print("----|-----|-------|-------|-----")

for i in sorter:
    item = choices[i] # Get the item using the sorted index
    img_char = 'A' if item['IMGTYPE'] == 0 else 'B'
    x_val  = item['PEAK_X']
    y_val  = item['PEAK_Y']
    # y_val is already calculated in y_transformed_values[i]
    # m_val is already calculated in m_values[i]

    # Get the pre-calculated m and y for this sorted item
    current_m = m_values[i]
    current_y = y_transformed_values[i]
    current_w = w_values[i]

    # Skip printing if m was -1 (indicating an out-of-bounds issue earlier)
    if current_m == -1:
        print(item)
        print(f"Skipping item due to m=-1 (original index in choices: {i})")
        continue

    print(f"{img_char:1s},{current_m:3d},{x_val:5d},{y_val:5d}, {current_w:8.3f}, {current_y:5d}")