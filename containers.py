#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:32:58 2018

@author: dmilakov
"""
from harps.core import np
#import harps.functions as hf

lineAxes      = ['pix','flx','bkg','err','rsd',
                 'sigma_v','wgt','mod','gauss_mod','wave']
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
            'optord':('optord','u4',()),
           'index':('index','u4',()),
           'pixl':('pixl','u4',()),
           'pixr':('pixr','u4',()),
           'segm':('segm','u4',()),
           'bary':('bary','float32',()),
           'skew':('skew','float32',()),
           'freq':('freq','float64',()),
           'mode':('mode','uint16',()),
           'anchor':('anchor','float64',()),
           'reprate':('reprate','float64',()),
           'noise':('noise','float64',()),
           'snr':('snr','float32',()),
           'gauss':('gauss','float64',(3,)),
           'gauss_err':('gauss_err','float64',(3,)),
           'gchisq':('gchisq','float32',()),
           'gchisqnu':('gchisqnu','float32',()),
           'chisq':('chisq','float64',()),
           'chisqnu':('chisqnu','float64',()),
           'residual':('residual','float64',()),
           'lsf':('lsf','float64',(3,)),
           'lsf_err':('lsf_err','float64',(3,)),
           'lchisq':('lchisq','float32',()),
           'lchisqnu':('lchisqnu','float32',()),
           'shift':('shift','float64',()),
           'fibre':('fibre','U1',()),
           'pars':('pars','float64',(3,)),
           'errs':('errs','float64',(3,)),
           'success':('success','b',(2,)),
           'conv':('conv','b',())} # (gauss, lsf)
def create_dtype(name,fmt,shape):
    return (name,fmt,shape)
def array_dtype(arraytype):
    assert arraytype in ['linelist','residuals','radial_velocity','linepars']
    if arraytype=='linelist':
        names = ['order','optord','index','pixl','pixr',
                 'segm','bary','skew','freq','mode',
                 #'anchor','reprate',
                 'noise','snr',
                 'gauss','gauss_err','gchisq','gchisqnu',
                 'lsf','lsf_err','lchisq','lchisqnu',
                 'success']
#    elif arraytype == 'linepars':
#        names = ['index','gcen','gsig','gamp','gcenerr','gsigerr','gamperr',
#                         'lcen','lsig','lamp','lcenerr','lsigerr','lamperr']
    elif arraytype == 'wavesol':
        names = ['order',]
    elif arraytype == 'coeffs':
        names = ['order','segm','pixl','pixr','chisq','chisqnu','pars','errs']
    elif arraytype == 'residuals':
        names = ['order','index','segm','residual','bary','noise']
    elif arraytype == 'linepars':
        names = ['index','pars','errs','chisq','chisqnu','conv']
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
    narray['index'] = np.arange(nlines)
    return narray
        
def linelist(nlines):
    linelist = narray(nlines,'linelist')
    return linelist
def linepars(nlines,npars=3):
    dtype=np.dtype([('index','u4',()),
                    ('pars','float64',(npars,)),
                    ('errs','float64',(npars,)),
                    ('chisq','float64',()),
                    ('chisqnu','float64',()),
                    ('conv','b',())])
    fitpars = np.zeros(nlines,dtype=dtype)
    fitpars['index'] = np.arange(nlines)
    return fitpars
def coeffs(polydeg,numsegs):
    dtype = np.dtype([('order','u4',()),
                      ('optord','u4',()),
                      ('segm','u4',()),
                      ('pixl','float64',()),
                      ('pixr','float64',()),
                      ('chisq','float64',()),
                      ('chisqnu','float64',()),
                      ('aicc','float64',()),
                      ('npts','uint16',()),
                      ('pars','float64',(polydeg+1,)),
                      ('errs','float64',(polydeg+1,))])
    narray = np.zeros(numsegs,dtype=dtype)
    narray['segm']=np.arange(numsegs)
    return narray

def residuals(nlines):
    dtype = np.dtype([('order','u4',()),
                      ('optord','u4',()),
                      ('index','u4',()),
                      ('segm','u4',()),
                      ('residual_mps','float64',()), # residual in units m/s
                      ('residual_A','float64',()), # residual in units Ansgtrom
                      ('wavefit','float64',()),
                      ('waverr','float64',()),
                      ('gauss','float64',()), 
                      ('lsf','float64',()),
                      ('cenerr','float64',()),# center
                      ('noise','float64',())]) 
    narray = np.zeros(nlines,dtype=dtype)
    narray['index']=np.arange(nlines)
    return narray
def gaps(norders):
    dtype = np.dtype([('order','u4',()),
                      ('gaps','float64',(7,))])
    narray= np.zeros(norders,dtype=dtype)
    return narray
def radial_velocity(nexposures):
    dtype = np.dtype([('index','u4',()),
                      ('shift','float64',()),
                      ('noise','float64',()),
                      ('datetime','datetime64[s]',()),
                      ('fibre','U3',()),
                      ('flux','float64',()),
                      ('fittype','U5',())])
    narray = np.zeros(nexposures,dtype=dtype)
    narray['index'] = np.arange(nexposures)
    return narray
def lsf(numsegs,npix):
    dtype = np.dtype([('order','u4',()),
                      ('optord','u4',()),
                      ('segm','u4',()),
                      ('pixl','u4',()),
                      ('pixr','u4',()),
                      ('x','float64',(npix,)),
                      ('y','float64',(npix,)),
                      ('dydx','float64',(npix,)),
                      ('numlines','u4',())])
    narray = np.full(numsegs,0,dtype=dtype)
    narray['segm'] = np.arange(numsegs)
    return narray
def datetime(numtim):
    dtype = np.dtype([('year','u4',()),
                      ('month','u4',()),
                      ('day','u4',()),
                      ('hour','u4',()),
                      ('min','u4',()),
                      ('sec','u4',())])
    narray = np.zeros(numtim,dtype=dtype)
    return narray

def add_field(a, descr):
    # https://stackoverflow.com/questions/1201817/
    # adding-a-field-to-a-structured-numpy-array
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("A must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b

class Generic(object):
    def __init__(self,narray):
        self._values = narray
    def __getitem__(self,item):
        condict, segm_sent = self._extract_item(item)
        return self.select(condict)
    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning order plus segment.
        
        To be used with partial decorator
        """
        condict = {}
        segm_sent = False
        if isinstance(item,dict):
            if len(item)==2: segm_sent=True
            
            condict.update(item)
        else:
            dict_sent=False
            if isinstance(item,tuple):
                
                nitem = len(item) 
                if nitem==2:
                    segm_sent=True
                    order,segm = item
                    
                elif nitem==1:
                    segm_sent=False
                    order = item[0]
            else:
                segm_sent=False
                order=item
            condict['order']=order
            if segm_sent:
                condict['segm']=segm
        return condict, segm_sent
    def __len__(self):
        return len(self.values)
    @property
    def values(self):
        return self._values
    def select(self,condict):
        cut  = self.cut(condict) 
        
        return Generic(self.values[cut])
    def cut(self,condict):
        values  = self.values 
        condtup = tuple(values[key]==val for key,val in condict.items())
        
        condition = np.logical_and.reduce(condtup)
        
        cut = np.where(condition==True)
        return cut
#def dataset(order=None,pixPerLine=22,names=None):
#
#    if names is None:
#        varnames = {'line':'line','pars':'pars',
#                    'wave':'wave',
#                    'attr':'attr','model':'model',
#                    'stat':'stat'}
#    else:
#        varnames = dict()
#        varnames['line'] = names.pop('line','line')
#        varnames['pars'] = names.pop('pars','pars')
#        varnames['wave'] = names.pop('wave','wave')
#        varnames['attr']  = names.pop('attr','attr')
#        varnames['model'] = names.pop('model','model')
#        varnames['stat'] = names.pop('stat','stat')
#        
#    dataarrays = [dataarray(name,order,pixPerLine) 
#        for name in varnames.values()]
#        
#
#    dataset = xr.merge(dataarrays)
#    return dataset
#def dataarray(name=None,order=None,pixPerLine=22):
#    linesPerOrder = 400
#
#    if name is None:
#        raise ValueError("Type not specified")
#    else:pass
#    orders = hf.prepare_orders(order)
#    dict_coords = {'od':orders,
#                   'id':np.arange(linesPerOrder),
#                   'ax':lineAxes,
#                   'pid':np.arange(pixPerLine),
#                   'ft':fitTypes,
#                   'par':fitPars,
#                   'wav':wavPars,
#                   'att':lineAttrs,
#                   'odpar':orderPars}
#    dict_sizes  = {'od':len(orders),
#                   'id':linesPerOrder,
#                   'ax':len(lineAxes),
#                   'pid':pixPerLine,
#                   'ft':len(fitTypes),
#                   'par':len(fitPars),
#                   'wav':len(wavPars),
#                   'att':len(lineAttrs),
#                   'odpar':len(orderPars)}
#    if name=='line':
#        dims   = ['od','id','ax','pid']
#    elif name=='pars':
#        dims   = ['od','id','par','ft']
#    elif name=='wave':
#        dims   = ['od','id','wav','ft']
#    elif name=='attr':
#        dims   = ['od','id','att']
#    elif name=='model':
#        dims = ['od','id','ft','pid']
#    elif name=='stat':
#        dims = ['od','odpar']
#    
#    if orders is None:
#        dims.remove('od')
#    else:
#        pass
#    shape  = tuple([dict_sizes[key] for key in dims])
#    coords = [dict_coords[key] for key in dims]
#    dataarray = xr.DataArray(np.full(shape,np.nan),coords=coords,dims=dims,
#                             name=name)
#    return dataarray