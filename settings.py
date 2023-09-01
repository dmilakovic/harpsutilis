#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:59:15 2018

@author: dmilakov
"""
import os, errno, json, logging, datetime

__version__ = '2.3'
version     = 'v{vers}'.format(vers=__version__)


harps_home   = os.environ['HARPSHOME']
harps_data   = os.environ['HARPSDATA']
harps_dtprod = os.environ['HARPSDATAPROD']

def prod_version(version):
    return os.path.join(*[harps_dtprod,version])
#harps_prod     = prod_version(version)
#harps_psf      = os.path.join(*[harps_prod,'psf_fit'])
#harps_ws       = os.path.join(*[prod_version(version),'fits','wavesol'])
#harps_lines    = os.path.join(*[harps_prod,'xrlines'])
#harps_rv       = os.path.join(*[harps_prod,'rv'])
#harps_combined = os.path.join(*[harps_prod,'combined_datasets'])
#harps_plots    = os.path.join(*[prod_version(version),'plots'])
#harps_linelist = os.path.join(*[prod_version(version),'fits','linelist'])
#harps_coeff    = os.path.join(*[prod_version(version),'fits','coeff'])
#harps_fits     = os.path.join(*[prod_version(version),'fits'])

# DIRECTORY TREE
harps_prod = harps_dtprod
harps_fits = os.path.join(*[harps_prod,'fits',version])
harps_objs = os.path.join(*[harps_prod,'objspec',version])
harps_logs = os.path.join(harps_prod,'logs')
harps_sett = os.path.join(harps_prod,'settings')
harps_inpt = os.path.join(harps_prod,'input')
harps_outp = os.path.join(*[harps_prod,'output',version])
harps_lsf  = os.path.join(*[harps_prod,'lsf',version])
harps_vpip = os.path.join(*[harps_prod,'vpfit_ip',version])
harps_plot = os.path.join(harps_prod,'plots')
harps_gaps = os.path.join(*[harps_prod,'output',version,'gaps'])
harps_dset = os.path.join(*[harps_prod,'dataset',version])
harps_sers = os.path.join(*[harps_prod,'series',version])
harps_cti  = os.path.join(*[harps_prod,'cti'])
harps_sims = os.path.join(*[harps_home,'simulations'])


dirnames = {'home':harps_home,
            'data':harps_data,
            'dtprod':harps_dtprod,
            'prod':harps_prod,
            'vpfit_ip':harps_vpip,
            'fits':harps_fits,
            'gaps':harps_gaps,
            'objspec':harps_objs,
#            'wavesol':harps_ws,
#            'linelist':harps_linelist,
#            'lines':harps_lines,
#            'coeff':harps_coeff,
            'plots':harps_plot,
            'lsf':harps_lsf,
            'dataset':harps_dset,
            'series':harps_sers,
            'cti':harps_cti,
            'sims':harps_sims}

def get_dirname(name,vers=None):
    if vers is not None:
        version = 'v{vers}'.format(vers=vers) 
    else:
        version = 'v{vers}'.format(vers=__version__)
    
    if name == 'prod':
        dirname = harps_dtprod
    elif name == 'fits':
        dirname = os.path.join(*[harps_prod,version,'fits'])
    elif name == 'objspec':
        dirname = os.path.join(*[harps_prod,version,'objspec',version])
    elif name == 'logs':
        dirname = os.path.join(harps_prod,'logs')
    elif name == 'settings':
        dirname = os.path.join(harps_prod,'settings')
    elif name == 'input':
        dirname = os.path.join(harps_prod,'input')
    elif name == 'output':
        dirname = os.path.join(*[harps_prod,version,'output'])
    elif name == 'lsf':
        dirname = os.path.join(*[harps_prod,version,'lsf'])
    elif name == 'plots':
        dirname = os.path.join(*[harps_prod,version,'plots'])
    elif name == 'gaps':
        dirname = os.path.join(*[harps_prod,version,'output','gaps'])
    elif name == 'dataset':
        dirname = os.path.join(*[harps_prod,version,'dataset'])
    elif name == 'series':
        dirname = os.path.join(*[harps_prod,version,'series'])
    elif name == 'cti':
        dirname  = os.path.join(*[harps_prod,version,'cti'])
    elif name == 'sims':
        dirname  = harps_sims
    else:
        dirname = None
    return dirname

rexp = 1e5

## 
nproc = 4

quiet = True

## first and last order in a spectrum
chip   = 'both'
if chip == 'red':
    sOrder = 45   
    eOrder = 71
elif chip == 'blue':
    sOrder = 39
    eOrder = 44
elif chip == 'both':
    sOrder = 39
    eOrder = 71
nOrder = eOrder - sOrder
nPix   = 4096
##


# =============================================================================
# settings for line fitting and stacking
# subbkg = True will subtract the background from the flux values
# divenv = True will divide the background subtracted flux values by the
#          background subtracted envelope
# error propagation is also performed. 
# Take a look at lines_aux.prepare_data
subbkg=True
divenv=False

# =============================================================================
def make_directory(dirpath):
    try:
        os.makedirs(dirpath)
        print("Making directory: ",dirpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return 1
def make_directory_tree(version=__version__):
    directories = (harps_prod,harps_fits,harps_inpt,harps_outp,
                   harps_plot,harps_sett,harps_logs,harps_lsf,harps_gaps,
                   harps_sers,harps_dset)
    [make_directory(d) for d in directories]

#------------------------------------------------------------------------------
#
#                       S E T T I N G S     C L A S S 
#
#------------------------------------------------------------------------------
class Settings(object):
    """
    Contains all the settings used when handling HARPS LFC spectra
    Self-updates with paths to various files that are produced in the process 
    and initiates a logger
    
    The JSON file needs to contain the following data:
        fibre (str)     : A / B
        e2dslist (str)  : path to the text file which contains paths to all the 
                          data to use
        LFC (str)       : HARPS / FOCES
        polyord (int)   : polynomial order of the wavelength solution (should
                          not be higher than 20)
        gaps (bool)     : shift the positions of the lines using the knowledge
                          of the gaps
        segment (bool)  : divide the 4096 pixels into 8 individual segments 
                          when calculating the wavelength solution
        fittype (str)   : gauss / lsf
        refspec (str)   : path to the spectrum whose ThAr coefficients will be 
                          used
        nproc (int)     : number of processors to use for calculations
        anchor_offset   : offset in the anchor frequency of the LFC              
    """
    def __init__(self,filepath):
        self.selfpath = filepath
        with open(self.selfpath,'r') as json_file:
            settings = json.load(json_file)
        for key,val in settings.items():
            setattr(self,key,val)
        # basename of the settings folder:
        setup_basename  = os.path.basename(filepath)
        setup_noext     = setup_basename.split('.')[0]
        self.append('selfname',setup_noext)
        # path to log
#        logdir         = hs.get_dirname('logs')
#        self.append('log',logfile)
        # output directory
        self.append('outdir',harps_outp)
        self.append('outfitsdir',harps_fits)
        self.write()
        
    def __str__(self):
        title = "{0:^80}\n".format("S E T T I N G S") +\
               "{0:-<80}\n".format("")
        mess = ""
        for key, val in self.__dict__.items():
            mess = mess+"{key:<20}:{val:>60}\n".format(key=key,val=str(val))
        return title+mess
    def update(self,newdict):
        for key,val in newdict.items():
            setattr(self,key,val)
        return
    def append(self,key,val):
        
        if not hasattr(self,key):
            print((3*("{:<15}")).format("Append", key, val))
            setattr(self,key,val)
        self.write()
        return
    def write(self):
        with open(self.selfpath,'w') as json_file:
            json.dump(self.__dict__,json_file,indent=4)
        return
logset_file = os.path.join(*[harps_logs,'config','logconfig.json'])
import logging.config
def read_logconfig(default_path=logset_file,default_level=logging.INFO,
                  env_key='LOG_CFG'):
    path = default_path if default_path is not None else logset_file
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
    else:
        config = None
    return config

def setup_logging(default_path=logset_file,default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """
    Setup logging configuration

    """
    config = read_logconfig(default_path=default_path,
                            default_level=default_level,
                            env_key=env_key)
    if config:
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return

def setup_log_files(default_path=logset_file,default_level=logging.INFO,
                  env_key='LOG_CFG'):
    logdir = get_dirname('logs')
    now    = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    info_logname  = os.path.join(logdir,f'{now}_info.log')
    error_logname = os.path.join(logdir,f'{now}_error.log')
    debug_logname = os.path.join(logdir,f'{now}_debug.log')
    
    CONFIG = read_logconfig(default_path=default_path,
                            default_level=default_level,
                            env_key=env_key)
    CONFIG['handlers']['info_file_handler']['filename']=info_logname
    CONFIG['handlers']['error_file_handler']['filename']=error_logname
    CONFIG['handlers']['debug_file_handler']['filename']=debug_logname
    
    logging.config.dictConfig(CONFIG)
    return
# def setup_logger():
#     logdir = hs.get_dirname('logs')
#     now    = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
#     logname = f'{now}_construct_from_spectrum_2d.log'