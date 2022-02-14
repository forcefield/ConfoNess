import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize

import sys
sys.path.insert( 0, '../')

from rxnet import KeqFromRXCycle
from ness import ness
from conformers import driven_conformer_rxs

BASE_RATES = {
    'M_u=M_f' : (1., 1.),
    'M_u+H_o=M_u.H_o' : (1., 1.),
    'M_f+H_o=M_f.H_o' : (1., 1.),
    'M_u.H_o=M_u.H_c' : (1., 1.),
    'M_f.H_o=M_f.H_c' : (1., 1.),
    'H_o=H_c' : (1., 1.)
    }
    

def ATP_driven_chaperone_folding( rates, folding_in_closed='folding_in_open'):
    ksrx = dict()
    
    rx = 'M_u=M_f'
    ksrx[rx] = rates[rx]

    for m in ['u', 'f']:
        rxc = 'M_%(m)s.H_o=M_%(m)s.H_c' % { 'm' : m }
        ksrx[rxc] = rates[rx]

    rx = 'M_u+H_o=M_u.H_o'
    ksrx[rx] = rates[rx]
    rx = 'M_f+H_o=M_f.H_o'
    ksrx[rx] = rates[rx]

    rx = 'M_u.H_o=M_f.H_o'
    K = KeqFromRXCycle( ksrx,
                        [ ('M_u=M_f', 1),
                          ('M_u+H_o=M_u.H_o', -1),
                          ('M_f+H_o=M_f.H_o', 1)])
    kf = ksrx['M_u=M_f'][0]
    kr = kf/K
    ksrx[rx] = (kf, kr)

    rx = 'M_u.H_c=M_f.H_c'
    if ('folding_in_open' == folding_in_closed):
        ksrx[rx] = ksrx['M_u.H_o=M_f.H_o']
    else:
        ksrx[rx] = ksrx['M_u=M_f']

    rx = 'H_o=H_c'
    ksrx[rx] = rates[rx]

    rxs = list( ksrx.keys())
    return rxs, np.array( [ ksrx[rx] for rx in ksrx ])

def chaperone_folding_analysis():
    from sympy import symbols, lambdify, diff

    kaU, kdU, kaF, kdF, kcU, koU, kcF, koF = symbols(
        'kaU kdU kaF kdF kcU koU kcF koF')

    # variables to investigate
    ys = [ kaU, kdU, kaF, kdF, kcU, koU, kcF, koF ]
    
    rates = {
        'M_u=M_f' : (1., 1.),
        'M_u+H_o=M_u.H_o' : (kaU, kdU),
        'M_f+H_o=M_f.H_o' : (kaF, kdF),
        'M_u.H_o=M_u.H_c' : (kcU, koU),
        'M_f.H_o=M_f.H_c' : (kcF, koF),
        'H_o=H_c' : (1., 1.)
        }

    concs = dict( M = 1., H = 1.)
    
    rxs, ks = ATP_driven_chaperone_folding( rates)
    Crx, ks, Nh, Nhmt0, mols, comps = driven_conformer_rxs( rxs, ks, concs, [])
    molid = dict( [(m, i) for i, m in enumerate( mols) ])
    
    U, S, Vh = svd( Crx)

    kofyfunc = lambdify( tuple(ys), ks, 'numpy')

    def kofy( y):
        return np.array(
            kofyfunc( y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]))

    foldid = molid[ 'M_f']
    
    def folded( y):
        ks = kofy( y)
        mss = ness( Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=Nhmt0)
        return mss.x[foldid]

    def optfunc( lny):
        y = np.exp( lny)
        return 1./folded( y)

    lny0 = np.ones( len(ys))
    res = minimize( optfunc, lny0)
        
    return res
