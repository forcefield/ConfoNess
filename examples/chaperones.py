import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize

import sys
sys.path.insert( 0, '../')

from rxnet import KeqFromRXCycle
from ness import ness, dness_dy, kofy_and_dkdy
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

    rx = 'M_f=M_n'
    if rx in rates:
        ksrx[rx] = rates[rx]

    rx = 'M_d=M_u'
    if rx in rates:
        ksrx[rx] = rates[rx]
        
    rx = 'M_u=M_f'
    ksrx[rx] = rates[rx]

    for m in ['u', 'f']:
        rx = 'M_%(m)s.H_o=M_%(m)s.H_c' % { 'm' : m }
        ksrx[rx] = rates[rx]

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

def chaperone_folding( states=['U', 'F']):
    '''
    Return two functions folded( ks) and noneq( ks), that compute the
    folded concentration and nonequilibrium ratio [U]/[F] given a set
    of kinetic parameters.

    Also return the list of variables to the functions.

    '''
    
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
        'H_o=H_c' : (1., 10.)
        }

    if 'N' in states:
        rates.update( { 'M_f=M_n' : (1000., 1.) })
    if 'D' in states:
        rates.update( { 'M_d=M_u' : (1., 1000.) })

    concs = dict( M = 1., H = 10.)
    
    rxs, ks = ATP_driven_chaperone_folding( rates, folding_in_closed='solution')
    Crx, ks, Nh, Nhmt0, mols, comps = driven_conformer_rxs( rxs, ks, concs, [])
    molid = dict( [(m, i) for i, m in enumerate( mols) ])

    U, S, Vh = svd( Crx)

    kofy, dkdy = kofy_and_dkdy( ks, ys)
    
    if 'N' in states:
        foldid = molid[ 'M_n']
    else:
        foldid = molid[ 'M_f']

    neqid = ( molid[ 'M_f'], molid[ 'M_u'])
    
    def noneq( y):
        ks = kofy( y)
        mss = ness( Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=Nhmt0)
        return mss.x[neqid[0]]/mss.x[neqid[1]]

    def folded( y):
        ks = kofy( y)
        mss = ness( Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=Nhmt0)
        return mss.x[foldid]

    def dfolded( y):
        ks = kofy( y)
        dks = dkdy( y)
        mss = ness( Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=Nhmt0)
        dmdy = dness_dy( Crx, ks, dks, len(y), mss.x, U=U, S=S, Vh=Vh, Nh=Nh)
        return dmdy[foldid, :]
    
    return folded, dfolded, noneq, ys

def main( args=None):

    fourstates = ['D', 'U', 'F', 'N']
    twostates = ['U', 'F']
    
    folded, dfolded, noneq, ys = chaperone_folding( states=twostates)

    def optfunc( lny):
        y = np.exp( lny)
        # print( folded(y))
        return 1./folded( y)

    def doptfunc( lny):
        y = np.exp( lny)
        f = folded( y)
        df = dfolded( y)
        # print( -y*df/(f*f))
        return -y*df/(f*f)

    from scipy.optimize import check_grad

    err = check_grad( optfunc, doptfunc, np.random.rand( len(ys)))
    print( 'grad err = ', err)
    
    print( ys)
    lny0 = np.random.rand( len(ys))
    res = minimize( optfunc, lny0, jac=doptfunc)

    print( res)
    return res

if '__main__' == __name__:
    main()
