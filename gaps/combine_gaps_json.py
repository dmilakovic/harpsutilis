#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:55:38 2019

@author: dmilakov


Reads the gap size measurements of the individual blocks into a single file.

"""
#%%
import argparse
import os, json, time
import harps.settings as hs

#%%
orders = {1:[61,62,63,64,65,66,67,68,69,70,71,72],
          2:[46,47,48,49,50,51,52,53,54,55,56,57,58,59,60],
          3:[27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]}
def main(args):
    
    input_filename = args.file
    input_basename = os.path.basename(input_filename)
    
    dirname  = hs.get_dirname('gaps') 
    output_basename = "{}_gaps.json".format(os.path.splitext(input_basename)[0])
    output_filename = os.path.join(dirname,output_basename)
    output = {"dataset": input_filename,
              "created_on":time.strftime("%Y-%m-%dT%H_%M_%S")}
    for block in [1,2,3]:
        gapsname = "{}_{}_gaps.json".format(os.path.splitext(input_basename)[0],block)
        gapspath = os.path.join(dirname,gapsname)
        
        with open(gapspath,'r') as file:
            data = json.load(file)
        output['block{}'.format(block)] = data['gaps_pix']
        output['orders{}'.format(block)] = orders[block]
    if os.path.isfile(gapspath):
        mode = 'a'
    else:
        mode = 'w'
    with open(output_filename,mode) as file:
        json.dump(output,file,indent=4)
    
#%% M A I N    P A R T
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Calculate gaps.')
    parser.add_argument('file',type=str, 
                        help='Path to the settings file')

    args = parser.parse_args()
    main(args)
