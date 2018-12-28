#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 23:18:21 2018

@author: dmilakov

Combines outfiles of several processes/settings

"""
import argparse, sys
import harps.io as io
import harps.functions as hf
def main(args):
    inputs = args.input
    output = args.output[0]
    
    print("DATA READ FROM :")
    print("{0:->20}".format(""))
    for file in inputs:
        print("{0}".format(file))
    
    data_in = hf.flatten_list([io.read_textfile(file) for file in inputs])
    header =  io.to_string(['# {file}'.format(file=file) for file in inputs])
    print("{0} LINES SAVED TO :".format(len(data_in)))
    print("{0}".format(output))
    io.write_textfile(data_in,output,header)
    
    return
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Combine "outlist" files ')
    parser.add_argument('input',nargs='+',
                        help='Paths to the input outlist files')
    parser.add_argument('--output',nargs=1,default=sys.stdout,
                        help='Path to the output file')
    args = parser.parse_args()
    main(args)