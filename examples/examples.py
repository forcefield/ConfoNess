import sys
sys.path.insert( 0, '..')

import numpy as np
from rxnet import *
from ness import *
from conformers import mass_equations, driven_conformer_rxs

def ATP_driven_conformational_cycle( kIA, shift, kATP):
    ksrx = {
        'M_i.ADP+ATP=M_i.ATP+ADP': (1., 1.),
        'M_a.ADP+ATP=M_a.ATP+ADP': (1., None),
        'M_i.ADP=M_a.ADP': kIA,
        'M_i.ATP=M_a.ATP': (kIA[0], kIA[1]/shift),
        'M_i.ATP=M_i.ADP+Pi': kATP,
        'M_a.ATP=M_a.ADP+Pi': (kATP[0], None),
    }
    
    # The following thermodynamic cycle closures must be satisfied:
    #
    # [m.ATP]/([m.ADP][Pi]) = [m.ATP][ADP]/([m.ADP][ATP])
    #                       * [ATP]/([ADP][Pi])
    # for m = M_i, M_a     
    #
    # [M_a.ATP][ADP]/([M_a.ADP][ATP]) = [M_a.ATP]/[M_i.ATP]
    #                                 * [M_i.ATP][ADP]/([M_i.ADP][ATP])
    #                                 * [M_i.ADP]/[M_a.ADP]

    Kx = KeqFromRXCycle( ksrx,
                         [('M_i.ATP=M_a.ATP', 1),
                          ('M_i.ADP+ATP=M_i.ATP+ADP', 1),
                          ('M_i.ADP=M_a.ADP', -1)])
    rx = 'M_a.ADP+ATP=M_a.ATP+ADP'
    kf, kr = ksrx[rx]
    ksrx[rx] = kf, kf/Kx

    rx0 = 'ATP=ADP+Pi'
    Kh0 = KeqFromRXCycle( ksrx,
                          [ ('M_i.ATP=M_i.ADP+Pi', 1),
                            ('M_i.ADP+ATP=M_i.ATP+ADP', 1) ])
    Kh = Kh0/Kx
    rx = 'M_a.ATP=M_a.ADP+Pi'
    kf, kr = ksrx[rx]
    ksrx[rx] = (kf, kf/Kh)

    rxs = ksrx.keys()
    return rxs, np.array( [ksrx[rx] for rx in ksrx])

def ATP_driven_conformational_cycle_layout():
    rmols = {
        'M_i.ADP' : (-1, 0),
        'M_a.ADP' : (1, 0),
        'M_i.ATP' : (-1, -1),
        'M_a.ATP' : (1, -1)
    }
    for m in rmols: rmols[m] = np.array( rmols[m])
    return rmols

def ATP_driven_conformational_noneq( kIA, shift, kATP, concs):
    rxs, ks = ATP_driven_conformational_cycle( kIA, shift, kATP)

    cstmols = ['ATP', 'ADP', 'Pi']
    Crx, ks, Nh, Nhmt0, mols, comps = driven_conformer_rxs( rxs, ks, concs, cstmols)

    molid = dict( [ (m, i) for i, m in enumerate( mols) ])
    mss = ness( Crx, ks, Nh=Nh, Nhmt0=Nhmt0)
    c = mss.x
    noneq = (c[molid['M_a.ADP']]/c[molid['M_i.ADP']])/(kIA[1]/kIA[0])
    return c, mols, noneq

def test_ATP_driven_conformational_cycle( tol=1e-9):
    kIA = (1., 1.)
    shift = 10.
    kATP = (1.e2, 1.)

    success = True

    rxs, ks0 = ATP_driven_conformational_cycle( kIA, shift, kATP)
    success &= testCycleClosure( rxs, ks0, tol)

    concs = { 'M': 1.,
              'ATP': 1.e3,
              'ADP': 1.,
              'Pi': 1. }
    cmols = dict({ 'M_i.ADP' : 0.2,
                   'M_a.ADP' : 0.3,
                   'M_i.ATP' : 0.1,
                   'M_a.ATP' : 0.5 }, **concs)

    nucs = ['ATP', 'ADP', 'Pi']
    Crx0, mols0 = reactionMatrix( rxs)
    molid = dict([(m, i) for i, m in enumerate(mols0)])
    dmdt0 = rate_of_change( Crx0, ks0, np.array([cmols[m] for m in mols0]))
    dmdt0 = np.delete( dmdt0, [ molid[m] for m in nucs ])
    
    Crx, ks, Nh, Nhmt0, mols, cmps = driven_conformer_rxs(rxs, ks0, concs, nucs)

    dmdt = rate_of_change( Crx, ks, np.array([cmols[m] for m in mols]))
    diffs = dmdt - dmdt0
    diff = np.max( np.abs( diffs))
    report = 'dm/dt for driven reactions computed two ways: delta = %g' % diff
    if diff < tol:
        print 'SUCCESS: %s < %g' % (report, tol)
    else:
        print 'FAIL: %s > %g' % (report, tol)
        success = False

    c, mols, noneq = ATP_driven_conformational_noneq( kIA, shift, kATP, concs)
    print 'ATP-driven conformational nonequilibrium: %f' % noneq
            
    return success

if __name__ == '__main__':
    sep = '='*60
    print 'Testing ATP-driven conformational cycle...'
    test_ATP_driven_conformational_cycle()
    print sep
    
