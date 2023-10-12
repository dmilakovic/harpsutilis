#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:58:08 2023

@author: dmilakov
"""
import multiprocessing, time, logging
from   functools import partial

import numpy as np
import harps.settings as hs
import harps.progress_bar as progress_bar
import harps.version as hv
import harps.functions.spectral as specfunc
from harps.containers import npars


from fitsio import FITS
import harps.lsf.fit as fit_ip
from harps.lsf.container import LSF2d

class LineSolver(multiprocessing.Process):
    """
    Simple worker.
    """

    def __init__(self, name, function, in_queue, out_queue):
        super(LineSolver, self).__init__()
        self.name = name
        self.function = function
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.logger = logging.getLogger("worker_"+name)

    def run(self):
        while True:
            # grab work; do something to it (+1); then put the result on the output queue
            item = self.in_queue.get()
            # print(f'item after queue.get = {item}')
            result = self.function(item,logger=self.logger)
            self.out_queue.put(result)
            
def single_line(i,linelist,x2d,flx2d,err2d,LSF2d_nm,ftype='gauss',scale='pix',
                interpolate=False,logger=None):
    
    logger = logger if logger is not None else logging.getLogger(__name__)
    
    if scale[:3] =='pix':
        scl = 'pix'
    elif scale[:3]=='vel':
        scl = 'wav'
    line   = linelist[i]
    od     = line['order']
    lpix   = line['pixl']
    rpix   = line['pixr']
    bary   = line['bary']
    # cent   = line[f'{ftype}_{scl}'][1]
    
    if len(np.shape(x2d))>1:
        
        flx1l  = flx2d[od,lpix:rpix]
        x1l    = x2d[od,lpix:rpix]
        err1l  = err2d[od,lpix:rpix]
        try: 
            LSF1d  = LSF2d_nm[od]
        except:
            logger.warning("LSF not found")
            return None
    elif len(np.shape(x2d))==1:
        flx1l  = flx2d[lpix:rpix]
        x1l    = x2d[lpix:rpix]
        err1l  = err2d[lpix:rpix]
        LSF1d  = LSF2d_nm
        
    success = False
    
    
    try:
        # logger.info(lsf1d.interpolate(bary))
        # output = hfit.lsf(x1l,flx1l,bkg1l,err1l,lsf1d,
        #                   interpolate=interpolate,
        #                   output_model=True)
        output = fit_ip.line(x1l,flx1l,err1l,bary,
                             LSF1d=LSF1d,
                             scale=scale,
                             interpolate=interpolate,
                             output_model=True)
        
        success, pars, errs, chisq, chisqnu, integral, model1l = output
    except:
        # logger.critical("failed")
        pass
    # print('line',i,success,pars,chisq)
    if not success:
        logger.critical('FAILED TO FIT LINE')
        logger.warning([i,od,x1l,flx1l,err1l])
        # return x1l,flx1l,err1l,LSF1d,interpolate
        # sys.exit()
        pars = np.full(npars,np.nan)
        errs = np.full(npars,np.nan)
        chisq = np.nan
        chisqnu = np.nan
        integral = np.nan
        model1l = np.zeros_like(flx1l)
    # else:
        # pars[1] = pars[1] 
        # new_line = copy.deepcopy(line)
    # npars = len(pars)
    line[f'lsf_{scl}']          = pars
    line[f'lsf_{scl}_err']      = errs
    line[f'lsf_{scl}_chisq']    = chisq
    line[f'lsf_{scl}_chisqnu']  = chisqnu
    line[f'lsf_{scl}_integral'] = integral
    
    return line, model1l

def ip_bulk(linelist,data,lsf_filepath,version=1,scale=['pixel','wave'],
            interpolate=True,logger=None):
    '''
    

    Parameters
    ----------
    linelist : TYPE
        DESCRIPTION.
    data : dictionary
        Contains keywords "flx", "err".
    lsf_filepath : TYPE
        DESCRIPTION.
    version : TYPE, optional
        DESCRIPTION. The default is 1.
    scale : TYPE, optional
        DESCRIPTION. The default is ['pixel','wave'].
    interpolate : TYPE, optional
        DESCRIPTION. The default is True.
    logger : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    def distribute(function):
        manager = multiprocessing.Manager()
        inq = manager.Queue()
        outq = manager.Queue()
    
        # construct the workers
        nproc = multiprocessing.cpu_count()
        logger.info(f"Using {nproc} workers")
        workers = [LineSolver(str(name+1), function,inq, outq) 
                   for name in range(nproc)]
        for worker in workers:
            worker.start()
    
        # add data to the queue for processing
        work_len = tot
        for item in cut:
            inq.put(item)
    
        while outq.qsize() != work_len:
            # waiting for workers to finish
            done = outq.qsize()
            progress = done/(work_len-1)
            time_elapsed = time.time() - time_start
            progress_bar.update(progress,name='lsf.aux.solve',
                               time=time_elapsed,
                               logger=None)
            
            time.sleep(1)
    
        # clean up
        for worker in workers:
            worker.terminate()
    
        # print the outputs
        results = []
        while not outq.empty():
            results.append(outq.get())
        return results
    if logger is not None:
        logger = logger.getChild('solve')
    else:
        logger = logging.getLogger(__name__).getChild('solve')
    # abbreviations
    # scl = f'{scale[:3]}'
    
    scale = np.atleast_1d(scale)
    # logger.info(f'version : {version}')
    # READ LSF
    with FITS(lsf_filepath,'r',clobber=False) as hdu:
        if 'pixel' in scale or 'pix' in scale:
            lsf2d_pix = hdu['pixel_model',version].read()
            LSF2d_nm_pix = LSF2d(lsf2d_pix)
        if 'velocity' in scale or 'wave' in scale:
            lsf2d_vel = hdu['velocity_model',version].read()
            LSF2d_nm_vel = LSF2d(lsf2d_vel)
    
    # READ OLD LINELIST AND DATA
    flx2d = data['flx']
    err2d = data['err']
    if len(np.shape(flx2d))>1:
        nbo,npix = np.shape(flx2d)
        pix2d = np.tile(np.arange(npix),nbo).reshape(nbo,npix)
    elif len(np.shape(flx2d))==1:
        nbo=1
        npix=len(flx2d)
        pix2d = np.arange(npix)
    
    
    # orders = np.unique(linelist['order'])
    
    # firstrow = int(1e6)
    orders = np.unique(linelist['order'])
    cut_ = [np.ravel(np.where(linelist['order']==od)[0]) for od in orders]
    cut = np.hstack(cut_)
    tot = len(cut)
    logger.info(f"Number of lines to fit : {tot}")
    # new_linelist = []
    # model2d = np.zeros_like(flx2d)
    # def get_iterable()
    # lines = (line for line in linelist)
    time_start = time.time()
    
    option = 2
    if 'pixel' in scale or 'pix' in scale:
        partial_function_pix = partial(single_line,
                                       linelist=linelist,
                                       x2d=pix2d,
                                       flx2d=flx2d,
                                       err2d=err2d,
                                       LSF2d_nm=LSF2d_nm_pix,
                                       ftype='lsf',
                                       scale='pixel',
                                       interpolate=interpolate)
        if option==1:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial_function_pix, cut)
        elif option==2:
            results = distribute(partial_function_pix)
        new_llist, models = np.transpose(results)
        # linelist=new_llist
        linelist[cut] = new_llist
    # delete these lines later. These were put in to skip re-doing the entire
    # calculations for pixel when also creating velocity models
    # with FITS(out_filepath,'r') as hdul:
        # linelist = hdul['linelist',version].read()
    # fit for wavelength positions
    if 'velocity' in scale or 'wave' in scale:
        import harps.wavesol as ws
        lsf_wavesol = ws.comb_dispersion(linelist, version=701, fittype='lsf', 
                                         npix=npix, 
                                         nord=nbo,
                                         ) 
        
        partial_function_vel= partial(single_line,
                                       linelist=linelist,
                                       x2d=lsf_wavesol,
                                       flx2d=flx2d,
                                       err2d=err2d,
                                       LSF2d_nm=LSF2d_nm_vel,
                                       ftype='lsf',
                                       scale='velocity',
                                       interpolate=interpolate)
        
        if option==1:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial_function_pix, np.arange(len(linelist)))
        elif option==2:
            results = distribute(partial_function_vel)
        new_llist, models = np.transpose(results)
        linelist[cut] = new_llist
    worktime = (time.time() - time_start)
    h, m, s  = progress_bar.get_time(worktime)
    logger.info(f"Total time elapsed : {h:02d}h {m:02d}m {s:02d}s")
    return linelist

def solve_1d(i,linelist1d,x1d,flx1d,err1d,LSF1d,ftype='gauss',scale='pix',
                interpolate=False,logger=None):
    
    logger = logger if logger is not None else logging.getLogger(__name__)
    
    if scale[:3] =='pix':
        scl = 'pix'
    elif scale[:3]=='vel' or scale[:3]=='wav':
        scl = 'wav'
    line   = linelist1d[i]
    lpix   = line['pixl']
    rpix   = line['pixr']
    bary   = line['bary']
    # cent   = line[f'{ftype}_{scl}'][1]
    flx1l  = flx1d[lpix:rpix]
    x1l    = x1d[lpix:rpix]
    err1l  = err1d[lpix:rpix]
    
    
    
    success = False
    
    
    try:
        # logger.info(lsf1d.interpolate(bary))
        # output = hfit.lsf(x1l,flx1l,bkg1l,err1l,lsf1d,
        #                   interpolate=interpolate,
        #                   output_model=True)
        output = fit_ip.line(x1l,flx1l,err1l,bary,
                             LSF1d=LSF1d,
                             scale=scale,
                             interpolate=interpolate,
                             output_model=True)
        
        success, pars, errs, chisq, chisqnu, integral, model1l = output
    except:
        # logger.critical("failed")
        pass
    # print('line',i,success,pars,chisq)
    if not success:
        logger.critical('FAILED TO FIT LINE')
        logger.warning([i,x1l,flx1l,err1l])
        # return x1l,flx1l,err1l,LSF1d,interpolate
        # sys.exit()
        pars = np.full(3,np.nan)
        errs = np.full(3,np.nan)
        chisq = np.nan
        chisqnu = np.nan
        integral = np.nan
        model1l = np.zeros_like(flx1l)
    # else:
        # pars[1] = pars[1] 
        # new_line = copy.deepcopy(line)
    line[f'lsf_{scl}']     = pars
    line[f'lsf_{scl}_err'] = errs
    line[f'lsf_{scl}_chisq']    = chisq
    line[f'lsf_{scl}_chisqnu']  = chisqnu
    line[f'lsf_{scl}_integral'] = integral
    
    return line, model1l

def solve(out_filepath,lsf_filepath,iteration,order,force_version=None,
          model_scatter=False,interpolate=False,scale=['pixel','velocity'],
          subbkg=hs.subbkg,divenv=hs.divenv,save2fits=True,logger=None):
    
    
    def bulk_fit(function):
        manager = multiprocessing.Manager()
        inq = manager.Queue()
        outq = manager.Queue()
    
        # construct the workers
        nproc = multiprocessing.cpu_count()
        logger.info(f"Using {nproc} workers")
        workers = [LineSolver(str(name+1), function,inq, outq) 
                   for name in range(nproc)]
        for worker in workers:
            worker.start()
    
        # add data to the queue for processing
        work_len = tot
        for item in cut:
            inq.put(item)
    
        while outq.qsize() != work_len:
            # waiting for workers to finish
            done = outq.qsize()
            progress = done/(work_len-1)
            time_elapsed = time.time() - time_start
            progress_bar.update(progress,name='lsf.aux.solve',
                               time=time_elapsed,
                               logger=None)
            
            time.sleep(1)
    
        # clean up
        for worker in workers:
            worker.terminate()
    
        # print the outputs
        results = []
        while not outq.empty():
            results.append(outq.get())
        return results
    
    if logger is not None:
        logger = logger.getChild('solve')
    else:
        logger = logging.getLogger(__name__).getChild('solve')
    # abbreviations
    # scl = f'{scale[:3]}'
    if force_version is not None:
        version = force_version
    else:
        version = hv.item_to_version(dict(iteration=iteration,
                                        model_scatter=model_scatter,
                                        interpolate=interpolate
                                        ),
                                   ftype='lsf'
                                   )
    scale = np.atleast_1d(scale)
    logger.info(f'version : {version}')
    # READ LSF
    with FITS(lsf_filepath,'r',clobber=False) as hdu:
        if 'pixel' in scale:
            lsf2d_pix = hdu['pixel_model',version].read()
            LSF2d_nm_pix = LSF2d(lsf2d_pix)
        if 'velocity' in scale:
            lsf2d_vel = hdu['velocity_model',version].read()
            LSF2d_nm_vel = LSF2d(lsf2d_vel)
    # lsf2d_gp = LSF2d_gp[order].values
    # lsf2d_numerical = hlsfit.numerical_model(lsf2d_gp,xrange=(-8,8),subpix=11)
    # LSF2d_numerical = LSF(lsf2d_numerical)
    
    
    # COPY LINELIST 
    io.copy_linelist_inplace(out_filepath, version)
    
    # READ OLD LINELIST AND DATA
    x2d,flx2d,err2d,env2d,bkg2d,linelist = read_outfile4solve(out_filepath,
                                                        version,
                                                        scale='pixel')
    flx_norm, err_norm, bkg_norm  = laux.prepare_data(flx2d,err2d,env2d,bkg2d, 
                                         subbkg=subbkg, divenv=divenv)
    
    
    # MAKE MODEL EXTENSION
    io.make_extension(out_filepath, 'model_lsf', version, flx2d.shape)
    
    nbo,npix = np.shape(flx2d)
    orders = specfunc.prepare_orders(order, nbo, sOrder=39, eOrder=None)
    
    # firstrow = int(1e6)
    cut_ = [np.ravel(np.where(linelist['order']==od)[0]) for od in orders]
    cut = np.hstack(cut_)
    tot = len(cut)
    logger.info(f"Number of lines to fit : {tot}")
    # new_linelist = []
    # model2d = np.zeros_like(flx2d)
    # def get_iterable()
    # lines = (line for line in linelist)
    time_start = time.time()
    
    option = 2
    if 'pixel' in scale:
        partial_function_pix = partial(solve_line,
                                       linelist=linelist,
                                       x2d=x2d,
                                       flx2d=flx_norm,
                                       err2d=err_norm,
                                       LSF2d_nm=LSF2d_nm_pix,
                                       ftype='lsf',
                                       scale='pixel',
                                       interpolate=interpolate)
        if option==1:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial_function_pix, cut)
        elif option==2:
            results = bulk_fit(partial_function_pix)
        new_llist, models = np.transpose(results)
    
        linelist[cut] = new_llist
    # delete these lines later. These were put in to skip re-doing the entire
    # calculations for pixel when also creating velocity models
    # with FITS(out_filepath,'r') as hdul:
        # linelist = hdul['linelist',version].read()
    # fit for wavelength positions
    if 'velocity' in scale:
        lsf_wavesol = ws.comb_dispersion(linelist, version=701, fittype='lsf', 
                                         npix=npix, 
                                         nord=nbo,
                                         ) 
        
        partial_function_vel= partial(solve_line,
                                       linelist=linelist,
                                       x2d=lsf_wavesol,
                                       flx2d=flx_norm,
                                       err2d=err_norm,
                                       LSF2d_nm=LSF2d_nm_vel,
                                       ftype='lsf',
                                       scale='velocity',
                                       interpolate=interpolate)
        
        if option==1:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial_function_pix, cut)
        elif option==2:
            results = bulk_fit(partial_function_vel)
        new_llist, models = np.transpose(results)
        linelist[cut] = new_llist
    worktime = (time.time() - time_start)
    h, m, s  = progress_bar.get_time(worktime)
    logger.info(f"Total time elapsed : {h:02d}h {m:02d}m {s:02d}s")
    
    if save2fits:
        for i,(ll,mod) in enumerate(zip(new_llist,models)):
            od   = ll['order']
            pixl = ll['pixl']
            row  = cut[i]
            with FITS(out_filepath,'rw',clobber=False) as hdu:
                hdu['model_lsf',version].write(np.array(mod),start=[od,pixl])
                hdu['linelist',version].write(np.atleast_1d(ll),firstrow=row)
        # if 'velocity' in scale:
        #     for i,(mod) in enumerate(zip(new_llist,models)):
        #         od   = ll['order']
        #         pixl = ll['pixl']
        #         row  = cut[i]
        #         with FITS(out_filepath,'rw',clobber=False) as hdu:
        #             hdu['model_lsf',version].write(np.array(mod),start=[od,pixl])
        #             # hdu['linelist',version].write(np.atleast_1d(ll),firstrow=row)
           
        with FITS(out_filepath,'rw',clobber=False) as hdu:
            hdu['linelist',version].write_key('ITER', iteration)
            hdu['linelist',version].write_key('SCT', model_scatter)
            hdu['linelist',version].write_key('INTP', interpolate)
    return linelist