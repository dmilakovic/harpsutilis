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

datashapes={'order':('order','u4',()),
           'index':('index','u4',()),
           'pixl':('pixl','u4',()),
           'pixr':('pixr','u4',()),
           'segm':('segm','u4',()),
           'bary':('bary','float32',()),
           'freq':('freq','float64',()),
           'mode':('mode','uint16',()),
           'anchor':('anchor','float64',()),
           'reprate':('reprate','float64',()),
           'noise':('noise','float64',()),
           'snr':('snr','float32',()),
           'gauss':('gauss','float64',(3,)),
           'gauss_err':('gauss_err','float64',(3,)),
           'gchisq':('gchisq','float32',()),
           'chisq':('chisq','float64',()),
           'residual':('residual','float64',()),
           'lsf':('lsf','float64',(3,)),
           'lsf_err':('lsf_err','float64',(3,)),
           'lchisq':('lchisq','float32',()),
           'shift':('shift','float64',()),
           'fibre':('fibre','U1',())}
def create_dtype(name,fmt,shape):
    return (name,fmt,shape)
def array_dtype(arraytype):
    assert arraytype in ['linelist','residuals','radial_velocity']
    if arraytype=='linelist':
        names = ['order','index','pixl','pixr',
                 'segm','bary','freq','mode',
                 #'anchor','reprate',
                 'noise','snr',
                 'gauss','gauss_err','gchisq',
                 'lsf','lsf_err','lchisq']
    elif arraytype == 'linepars':
        names = ['index','gcen','gsig','gamp','gcenerr','gsigerr','gamperr',
                         'lcen','lsig','lamp','lcenerr','lsigerr','lamperr']
    elif arraytype == 'wavesol':
        names = ['order',]
    elif arraytype == 'coeffs':
        names = ['order','segm','pixl','pixr','chisq','pars','errs']
    elif arraytype == 'residuals':
        names = ['order','index','segm','residual','bary','noise']
    else:
        names = []
    dtypes = [datashapes[name] for name in names]
    #formats = [datatypes[name] for name in names]
    #shapes  = [datashapes[name] for name in names]
    #return np.dtype({'names':names,'formats':formats, 'shapes':shapes})
    return np.dtype(dtypes)

_dtype = {'linelist':array_dtype('linelist'),
              'radial_velocity':array_dtype('radial_velocity')}

def narray(nlines,arraytype):
    dtype=array_dtype(arraytype)
    narray = np.zeros(nlines,dtype=dtype)
    narray['index'] = np.arange(1,nlines+1)
    return narray
        
def linelist(nlines):
    linelist = narray(nlines,'linelist')
    return linelist

def coeffs(polydeg,numsegs):
    dtype = np.dtype([('order','u4',()),
                      ('segm','u4',()),
                      ('pixl','u4',()),
                      ('pixr','u4',()),
                      ('chi2','float64',()),
                      ('pars','float64',(polydeg+1,)),
                      ('errs','float64',(polydeg+1,))])
    narray = np.zeros(numsegs,dtype=dtype)
    narray['segm']=np.arange(1,numsegs+1)
    return narray
def linepars(nlines):
    # g is for gauss
    # l is for line-spread-function
    linepars = narray(nlines,'linepars')
    linepars['index']=np.arange(1,nlines+1)
    return linepars
def residuals(nlines):
    dtype = np.dtype([('order','u4',()),
                      ('index','u4',()),
                      ('segm','u4',()),
                      ('residual','float64',()),
                      ('gauss','float64',()), # center
                      ('noise','float64',())]) 
    narray = np.zeros(nlines,dtype=dtype)
    narray['index']=np.arange(1,nlines+1)
    return narray
def gaps():
    pass

def radial_velocity(nexposures):
    dtype = np.dtype([('index','u4',()),
                      ('shift','float64',()),
                      ('noise','float64',()),
                      ('datetime','datetime64[s]',()),
                      ('fibre','U1',())])
    narray = np.zeros(nexposures,dtype=dtype)
    narray['index'] = np.arange(1,nexposures+1)
    return narray
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