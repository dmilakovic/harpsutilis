#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:32:58 2018

@author: dmilakov
"""
from harps.core import np, xr
import harps.functions as hf

lineAxes      = ['pix','flx','bkg','err','rsd',
                 'sigma_v','wgt','mod','gauss_mod','wave']
fitPars       = ['cen','cen_err','flx','flx_err','sigma','sigma_err','chisq']
wavPars       = ['val','err','rsd']
fitTypes      = ['epsf','gauss']
lineAttrs     = ['bary','n','freq','freq_err','seg','pn','snr']
orderPars     = ['sumflux']
    

#==============================================================================
#
#                         D A T A    C O N T A I N E R S                  
#
#==============================================================================
#datatypes={'index':'u4',
#           'pixl':'u4',
#           'pixr':'u4',
#           'segm':'u4',
#           'bary':'float32',
#           'freq':'float32',
#           'mode':'uint16',
#           'noise':'float32',
#           'snr':'float32',
#           'gauss':'float32',
#           'gcen':'float32',
#           'gsig':'float32',
#           'gamp':'float32',
#           'gcenerr':'float32',
#           'gsigerr':'float32',
#           'gamperr':'float32',
#           'lcen':'float32',
#           'lsig':'float32',
#           'lamp':'float32',
#           'lcenerr':'float32',
#           'lsigerr':'float32',
#           'lamperr':'float32'}
datashapes={'index':('index','u4',()),
           'pixl':('pixl','u4',()),
           'pixr':('pixr','u4',()),
           'segm':('segm','u4',()),
           'bary':('bary','float32',()),
           'freq':('freq','float32',()),
           'mode':('mode','uint16',()),
           'noise':('noise','float32',()),
           'snr':('snr','float32',()),
           'gauss':('gauss','float32',(3,)),
           'gauss_err':('gauss_err','float32',(3,)),
           'gchisq':('gchisq','float32',()),
           'lsf':('lsf','float32',(3,)),
           'lsf_err':('lsf_err','float32',(3,)),
           'lchisq':('lchisq','float32',())}
def create_dtype(name,fmt,shape):
    return (name,fmt,shape)
def array_dtype(arraytype):
    assert arraytype in ['linelist','linepars']
    if arraytype=='linelist':
        names = ['index','pixl','pixr',
                 'segm','bary','freq','mode','noise','snr',
                 'gauss','gauss_err','gchisq',
                 'lsf','lsf_err','lchisq']
    elif arraytype == 'linepars':
        names = ['index','gcen','gsig','gamp','gcenerr','gsigerr','gamperr',
                         'lcen','lsig','lamp','lcenerr','lsigerr','lamperr']
    elif arraytype == 'wavesol':
        names = ['order',]
    dtypes = [datashapes[name] for name in names]
    #formats = [datatypes[name] for name in names]
    #shapes  = [datashapes[name] for name in names]
    #return np.dtype({'names':names,'formats':formats, 'shapes':shapes})
    return np.dtype(dtypes)
def narray(nlines,arraytype):
    dtype=array_dtype(arraytype)
    narray = np.zeros(nlines,dtype=dtype)
    narray['index'] = np.arange(1,nlines+1)
    return narray
        
def linelist(nlines):
    linelist = narray(nlines,'linelist')
    return linelist

def coeffs(polydeg,numpatch):
    dtype = np.dtype([('patch','u4',()),
                      ('pixl','u4',()),
                      ('pixr','u4',()),
                      ('pars','float32',(polydeg+1,)),
                      ('errs','float32',(polydeg+1,))])
    narray = np.zeros(numpatch,dtype=dtype)
    narray['patch']=np.arange(1,numpatch+1)
    return narray
def linepars(nlines):
    # g is for gauss
    # l is for line-spread-function
    linepars = narray(nlines,'linepars')
    linepars['index']=np.arange(1,nlines+1)
    return linepars
def return_empty_wavesol():
    return
def return_empty_dataset(order=None,pixPerLine=22,names=None):

    if names is None:
        varnames = {'line':'line','pars':'pars',
                    'wave':'wave',
                    'attr':'attr','model':'model',
                    'stat':'stat'}
    else:
        varnames = dict()
        varnames['line'] = names.pop('line','line')
        varnames['pars'] = names.pop('pars','pars')
        varnames['wave'] = names.pop('wave','wave')
        varnames['attr']  = names.pop('attr','attr')
        varnames['model'] = names.pop('model','model')
        varnames['stat'] = names.pop('stat','stat')
        
    dataarrays = [dataarray(name,order,pixPerLine) 
        for name in varnames.values()]
        

    dataset = xr.merge(dataarrays)
    return dataset
def dataarray(name=None,order=None,pixPerLine=22):
    linesPerOrder = 400

    if name is None:
        raise ValueError("Type not specified")
    else:pass
    orders = hf.prepare_orders(order)
    dict_coords = {'od':orders,
                   'id':np.arange(linesPerOrder),
                   'ax':lineAxes,
                   'pid':np.arange(pixPerLine),
                   'ft':fitTypes,
                   'par':fitPars,
                   'wav':wavPars,
                   'att':lineAttrs,
                   'odpar':orderPars}
    dict_sizes  = {'od':len(orders),
                   'id':linesPerOrder,
                   'ax':len(lineAxes),
                   'pid':pixPerLine,
                   'ft':len(fitTypes),
                   'par':len(fitPars),
                   'wav':len(wavPars),
                   'att':len(lineAttrs),
                   'odpar':len(orderPars)}
    if name=='line':
        dims   = ['od','id','ax','pid']
    elif name=='pars':
        dims   = ['od','id','par','ft']
    elif name=='wave':
        dims   = ['od','id','wav','ft']
    elif name=='attr':
        dims   = ['od','id','att']
    elif name=='model':
        dims = ['od','id','ft','pid']
    elif name=='stat':
        dims = ['od','odpar']
    
    if orders is None:
        dims.remove('od')
    else:
        pass
    shape  = tuple([dict_sizes[key] for key in dims])
    coords = [dict_coords[key] for key in dims]
    dataarray = xr.DataArray(np.full(shape,np.nan),coords=coords,dims=dims,
                             name=name)
    return dataarray