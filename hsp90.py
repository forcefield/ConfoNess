import numpy as np
import pickle as pickle
from scipy.linalg import svd, lstsq
from scipy.optimize import check_grad
from scipy.optimize import least_squares
from ness import ness, kinetics, dness_dy
from ness import consumption_rate
from rxnet import KeqFromk, KeqFromRXCycle, testCycleClosure, reactionMatrix
from rxnet import reactionsWithConstMols
from conformers import mass_equations, driven_conformer_rxs
from conformers import consumptive_reactions

# The concentration and time units in the rates are uM and seconds
BASE_RATES = {
    # Client binding to Hsp90.
    'M_d+H_O=M_d.H_O': (1., 5.6e-3),  # (1, 5.09e2),
    'M_m+H_O=M_m.H_O': (1., 6.15e-1),

    # ATP-driven conformational cycle of Hsp90.
    'H_O+ATP=H_O.ATP': (6.66e-4, 0.145),
    'H_O.ATP=H_C.ATP': (1.67e-2, 5.e-5),
    'H_C.ADP=H_O+ADP': (1, 1.e-1),
    'H_C.ATP=H_C.ADP+Pi': (3.3e-1, 9.28e-14),

    # The deactivating client may induce a different rate of O=C conversion
    # in the Hps90 than the maturing client, because its binding to the Hsp90
    # may hinder the latter's conformational change.
    'M_d.H_O.ATP=M_d.H_C.ATP': (3.33e-1, 5.49e-3),
    'M_m.H_O.ATP=M_m.H_C.ATP': (3.33e-1, 5.e-5), # (1.67e-2, 5.e-5), #  

    # Client conformational dynamics.
    'M_i=M_d': (1, 5e2),
    'M_d=M_m': (1e-1, 1e-1), # (1.e-2, 1.27), # 
    'M_m=M_a': (6.51e1, 1),

    # The binding of CDC37 to the Hsp90.
    'H_O+Q=Q.H_O': (1., 1.4),

    'M_d+Q=Q.M_d': (1., 5.12e-3), # (1, 7.56e-2),

    # The cis-binding between CDC37 and Hsp90 within the CDC37-client-Hsp90
    # ternary complex.  This is not explicitly modeled, but provides a parameter
    # to other reactions.
    # 'Q.M_d.H_O=1Q.M_d.H_O1': (1e-4, 1.4),
    
    'M_d.H_O+Q=Q.M_d.H_O': (2.e-1, 2.42e-2),
    
    'M_d.H_O.ATP+Q=Q.M_d.H_O.ATP': (1., 2.52e-2),

    # CDC37 has to adopt a different conformation in the closed Hsp90 ternary
    # complex.  This may entail a thermodynamic penalty.
    'M_d.H_C.ATP+Q=Q.M_d.H_C.ATP': (1.e-1, 1.e-1),

    # AHA1 binding to Hsp90
    'H_O+A=A.H_O': (1., 1.2),
    'H_C.ATP+A=A.H_C.ATP': (1., 0.16),
    'A.H_O.ATP=A.H_C.ATP': (1.67e-1, 1.996e-4)
}
    
# Equilibrium constant of ATP=ADP+Pi: Keq = [ADP][Pi]/[ATP]
ATP_HYDROLYSIS_KEQ = 5.45e13

def ATP_driven_cycle( rates):
    # The ATP hydrolysis cycle satisfies the following thermodynamic 
    # cycle closure:
    #
    # [H_C.ADP][Pi]/[H_C.ATP] = [H_C.ADP]/([H_O][ADP])
    #                         * [H_O][ATP]/[H_O.ATP]
    #                         * [H_O.ATP]/[H_C.ATP]
    #                         * [ADP][Pi]/[ATP]
    rx = 'H_C.ATP=H_C.ADP+Pi'
    KATP = ATP_HYDROLYSIS_KEQ*KeqFromRXCycle( rates,
                                              [('H_C.ADP=H_O+ADP', -1),
                                               ('H_O+ATP=H_O.ATP', -1),
                                               ('H_O.ATP=H_C.ATP', -1)])
    kf, kr = rates[rx]
    kr = kf/KATP
    rates[rx] = (kf, kr)
    rxs = ['H_O+ATP=H_O.ATP', 'H_O.ATP=H_C.ATP', 'H_C.ATP=H_C.ADP+Pi',
           'H_C.ADP=H_O+ADP']
    return rxs, np.array( [rates[rx] for rx in rxs])
    
def ATP_cochaperone_conformational_cycle( rates, include_AHA1=False):
    ksrx = dict()

    # This updates the rates to close the thermodynamic cycle.
    rxs, ks = ATP_driven_cycle( rates)

    for rx, k in zip( rxs, ks):
        ksrx[rx] = k

    # The deactivated and mature clients can enter intermediate and
    # activating states, respectively.
    for rx in ['M_i=M_d', 'M_m=M_a']:
        ksrx[rx] = rates[rx]
        
    # deactivating client may have different affinity for the closed
    # conformation than the maturing client.
    deltaKOCdm = KeqFromRXCycle( rates,
                                 [('M_d.H_O.ATP=M_d.H_C.ATP', -1),
                                  ('M_m.H_O.ATP=M_m.H_C.ATP', 1)])

    for m in [ 'M_d', 'M_m' ]:
        # client-chaperone binding
        rx = '%(m)s+%(h)s=%(m)s.%(h)s' % {'m': m, 'h': 'H_O'}
        ksrx[rx] = rates['%(m)s+H_O=%(m)s.H_O' % {'m': m}]
        
        # ATP-binding to open Hsp90
        rx = '%(m)s.H_O+ATP=%(m)s.H_O.ATP' % {'m': m}
        ksrx[rx] = rates['H_O+ATP=H_O.ATP']
        
        rx = '%(m)s+H_O.ATP=%(m)s.H_O.ATP' % {'m': m}
        # The client binding to H_O.ATP satisfies the following 
        # thermodynamic cycle
        #
        # [M][[H_O.ATP]/[M.H_O.ATP] = [M][H_O]/[M.H_O]
        #                           * [M.H_O][ATP]/[M.H_O.ATP]
        #                           * [H_O.ATP]/([H_O][ATP])
        KD = KeqFromRXCycle(ksrx,
                            [('%(m)s+H_O=%(m)s.H_O' % {'m': m}, -1),
                             ('%(m)s.H_O+ATP=%(m)s.H_O.ATP' % {'m': m}, -1),
                             ('H_O+ATP=H_O.ATP', 1)])
        ka, kd = rates['%(m)s+H_O=%(m)s.H_O' % {'m': m}]
        kd = ka*KD
        ksrx[rx] = (ka, kd)

        # ATP-bound Hsp90 transition between open and closed conformations
        rx = '%(m)s.H_O.ATP=%(m)s.H_C.ATP' % {'m': m}
        kf, kr = ksrx[rx] = rates[rx]

        # ATP hydrolysis
        rx = '%(m)s.H_C.ATP=%(m)s.H_C.ADP+Pi' % {'m': m}
        ksrx[rx] = rates['H_C.ATP=H_C.ADP+Pi']

        # Dissociation of ADP and return to open state of chaperone
        rx = '%(m)s.H_C.ADP=%(m)s.H_O+ADP' % {'m': m}
        # The hydrolysis satisfies the following thermodynamic cycle
        # [M.H_O][ADP]/[M.H_C.ADP] = [M.H_O][ATP]/[M.H_O.ATP]
        #                          * [M.H_C.ATP]/([M_H_C.ADP][Pi])
        #                          * [M.H_O.ATP]/[M.H_C.ATP]
        #                          * [Pi][ADP]/[ATP]
        KD = ATP_HYDROLYSIS_KEQ* \
             KeqFromRXCycle( ksrx,
                             [('%(m)s.H_O+ATP=%(m)s.H_O.ATP' % {'m': m}, -1),
                              ('%(m)s.H_C.ATP=%(m)s.H_C.ADP+Pi' % {'m': m}, -1),
                              ('%(m)s.H_O.ATP=%(m)s.H_C.ATP' % {'m': m}, -1)])
        kd, ka = rates['H_C.ADP=H_O+ADP']
        ka = kd/KD
        ksrx[rx] = (kd, ka)

    # client conformational transitions
    rx = 'M_d=M_m'
    kdm = rates[rx]
    ksrx[rx] = kdm
    
    # client can change conformation in open and ATP-bound closed state.
    # Because the deactivating and maturing clients may have different binding
    # affinities for Hsp90, the conformational equilibrium when they are
    # Hsp90-bound will be different from in apo.  They satisify the following
    # thermodynamic cycle closure:
    #
    # [M_m.H]/[M_d.H] = [M_m.H]/([M_m][H])
    #                 = [M_d][H]/[M_d.H]
    #                 = [M_m]/[M_d]
    Kdm = KeqFromRXCycle( ksrx,
                         [('M_m+H_O=M_m.H_O', 1),
                          ('M_d+H_O=M_d.H_O', -1),
                          ('M_d=M_m', 1)])
    #for h in ['H_O', 'H_O.ATP']:
    #    rx = 'M_d.%(h)s=M_m.%(h)s' % {'h': h}
    #    kf, kr = kdm
    #    ksrx[rx] = kf, kf/Kdm

    # The maturing and deactivating clients may have different preferences
    # for the closed state, affecting the conformational transitions when
    # bound to the closed state.  The following thermodynamic cycle closure 
    # should be satisfied:
    #
    # [M_m.H_C.ATP]/[M_d.H_C.ATP] = [M_m.H_C.ATP]/[M_m.H_O.ATP]
    #                             * [M_d.H_O.ATP]/[M_d.H_C.ATP]
    #                             * [M_m.H_O.ATP]/[M_d.H_O.ATP]
    Kdm *= KeqFromRXCycle( ksrx,
                           [('M_m.H_O.ATP=M_m.H_C.ATP', 1),
                            ('M_d.H_O.ATP=M_d.H_C.ATP', -1)])
    for h in ['H_C.ATP']: # , 'H_C.ADP']:
        rx = 'M_d.%(h)s=M_m.%(h)s' % {'h': h}
        kf, kr = kdm
        ksrx[rx] = kf, kf/Kdm

    ###################################
    ##### CDC37-related reactions #####
    ###################################
    rx = 'H_O+Q=Q.H_O'
    ksrx[rx] = rates[rx]

    rx = 'M_d+Q=Q.M_d'
    ksrx[rx] = rates[rx]

    # The cochaperone binds M_d.H_O with 2 interfaces. Dissociation requires 
    # that the Q first breaks the interface with H_O. The dissociation rate
    # can be approximated by
    # 
    # k(Q.M.H->M.H+Q) = [Q.M.H]/([1Q.M.H1]+[Q.M.H])
    #                 * k(M.Q->M+Q)
    # KAQHt = KeqFromk( rates['Q.M_d.H_O=1Q.M_d.H_O1'])
    rx = 'M_d.H_O+Q=Q.M_d.H_O'
    # ka, kd = ksrx['M_d+Q=Q.M_d']
    # ksrx[rx] = (ka, kd/(KAQHt + 1.))
    ksrx[rx] = rates[rx]

    # client binding to cochaperone-hsp90 complex satisfies the following 
    # thermodynamic cycle:
    #
    # [M_d][Q.H_O]/[Q.M_d.H_O] = [M_d][H_O]/[M_d.H_O]
    #                          * [Q.H_O]/([Q][H_O])
    #                          * [M_d.H_O][Q]/[Q.M_d.H_O]
    rx = 'M_d+Q.H_O=Q.M_d.H_O'
    KD = KeqFromRXCycle( ksrx,
                         [('M_d+H_O=M_d.H_O', -1),
                          ('H_O+Q=Q.H_O', 1),
                          ('M_d.H_O+Q=Q.M_d.H_O', -1)])
    ka, kd = ksrx['M_d+Q=Q.M_d']
    kd = ka*KD
    ksrx[rx] = (ka, kd)

    # cochaperone-client complex binding to hsp90 satisfies the following
    # thermodynamic cycle:
    #
    # [Q.M_d][H_O]/[Q.M_d.H_O] = [Q.M_d]/([Q][M_d])
    #                          * [Q][H_O]/[Q.H_O]
    #                          * [Q.H_O][M_d]/[Q.M_d.H_O]
    rx = 'Q.M_d+H_O=Q.M_d.H_O'
    KD = KeqFromRXCycle( ksrx,
                         [('M_d+Q=Q.M_d', 1),
                          ('H_O+Q=Q.H_O', -1),
                          ('M_d+Q.H_O=Q.M_d.H_O', -1)])
    ka, kd = ksrx['H_O+Q=Q.H_O']
    kd = KD*ka
    ksrx[rx] = (ka, kd)
    
    # cochaperone binding to ATP-bound Hsp90-client complex should be
    # similar to its binding to the client, as ATP-bound Hsp90
    # eliminates the CDC37-Hsp90 interface.  
    #
    rx = 'M_d.H_O.ATP+Q=Q.M_d.H_O.ATP'
    ksrx[rx] = rates[rx]

    # The following thermodynamic cycle should be satisifed:
    #
    # [Q.M_d.H_O][ATP]/[Q.M_d.H_O.ATP] = [Q.M_d.H_O]/([Q][M_d.H_O])
    #                                  * [Q][M_d.H_O.ATP]/[Q.M_d.H_O.ATP]
    #                                  * [M_d.H_O][ATP]/[M_d.H_O.ATP]
    rx = 'Q.M_d.H_O+ATP=Q.M_d.H_O.ATP'
    KD = KeqFromRXCycle( ksrx,
                         [('M_d.H_O+Q=Q.M_d.H_O', 1),
                          ('M_d.H_O.ATP+Q=Q.M_d.H_O.ATP', -1),
                          ('M_d.H_O+ATP=M_d.H_O.ATP', -1)])
    ka, kd = ksrx['H_O+ATP=H_O.ATP']
    kd = KD*ka
    ksrx[rx] = (ka, kd)

    rx = 'Q.M_d+H_O.ATP=Q.M_d.H_O.ATP'
    # The following thermodynamic cycle need to be satisified:
    # 
    # [Q.M][H_O.ATP]/[Q.M.H_O.ATP] = [Q.M][H_O]/[Q.M.H_O]
    #                              * [Q.M.H_O][ATP]/[Q.M.H_O.ATP]
    #                              * [H_O.ATP]/([H_O][ATP])
    KD = KeqFromRXCycle( ksrx,
                         [('Q.M_d+H_O=Q.M_d.H_O', -1),
                          ('Q.M_d.H_O+ATP=Q.M_d.H_O.ATP', -1),
                          ('H_O+ATP=H_O.ATP', 1)])
    ka, kd = ksrx['Q.M_d+H_O=Q.M_d.H_O']
    kd = ka*KD
    ksrx[rx] = (ka, kd)

    rx = 'M_d.H_C.ATP+Q=Q.M_d.H_C.ATP'
    ksrx[rx] = rates[rx]

    # cochaperone binding to the closed client-Hsp90 complex.
    # The binding satisfies the following thermodynamic cycle:
    #
    # [Q.M_d.H_C.ATP]/[Q.M_d.H_O.ATP] = [Q][M_d.H_O.ATP]/[Q.M_d.H_O.ATP]
    #                                 * [M_d.H_C.ATP]/[M_d.H_O.ATP]
    #                                 * [Q.M_d.H_C.ATP]/([Q][M_d.H_C.ATP])
    rx = 'Q.M_d.H_O.ATP=Q.M_d.H_C.ATP'
    Keq = KeqFromRXCycle( ksrx,
                          [('M_d.H_O.ATP+Q=Q.M_d.H_O.ATP', -1),
                           ('M_d.H_O.ATP=M_d.H_C.ATP', 1),
                           ('M_d.H_C.ATP+Q=Q.M_d.H_C.ATP', 1)])
    kf, kr = ksrx['M_d.H_O.ATP=M_d.H_C.ATP']
    kr = kf/Keq
    ksrx[rx] = (kf, kr)
    
    if (include_AHA1):
        ksrx = add_AHA1_to_hsp90_cycle( ksrx, rates)

    rxs = list(ksrx.keys())
    return rxs, np.array( [ksrx[rx] for rx in ksrx])

def add_AHA1_to_hsp90_cycle( ksrx, rates):
    '''
    Include AHA1 in the conformational cycle.
    '''
    ka, kd = rates['H_O+A=A.H_O']
    for h in [ 'H_O', 'H_O.ATP' ]:
        rx = '%(h)s+A=A.%(h)s' % { 'h' : h }
        ksrx[rx] = (ka, kd)
        for m in [ 'M_d', 'M_m', 'Q.M_d' ]:
            rx = '%(m)s.%(h)s+A=A.%(m)s.%(h)s' % { 'h' : h, 'm' : m }
            ksrx[rx] = (ka, kd)
    
    ka, kd = rates['H_C.ATP+A=A.H_C.ATP']
    h = 'H_C.ATP'
    rx = '%(h)s+A=A.%(h)s' % { 'h' : h }
    ksrx[rx] = (ka, kd)
    for m in [ 'M_d', 'M_m', 'Q.M_d' ]:
        rx = '%(m)s.%(h)s+A=A.%(m)s.%(h)s' % { 'h' : h, 'm' : m }
        ksrx[rx] = (ka, kd)

    # ATP binding
    rx = 'A.H_O+ATP=A.H_O.ATP'
    ksrx[rx] = ksrx['H_O+ATP=H_O.ATP']
    for m in ['M_d', 'M_m', 'Q.M_d']:
        rx0 = '%(m)s.H_O+ATP=%(m)s.H_O.ATP' % {'m': m}
        rx = 'A.%(m)s.H_O+ATP=A.%(m)s.H_O.ATP' % {'m': m}
        ksrx[rx] = ksrx[rx0]
    
    # Hsp90 open to close conformational transition.
    # The following cycle must be closed
    #
    # [A.H_C.ATP]/[A.H_O.ATP] = [A.H_C.ATP]/([A][H_C.ATP])
    #                         * [A][H_O.ATP]/[A.H_O.ATP]
    #                         * [H_C.ATP]/[H_O.ATP]
    kOC0, kCO0 = ksrx[rx0]
    KeqOC0 = kOC0/kCO0
    rx = 'A.H_O.ATP=A.H_C.ATP'
    kOCA, kCOA = rates['A.H_O.ATP=A.H_C.ATP']
    KeqOCA = KeqFromRXCycle( ksrx, 
                          [ ('H_C.ATP+A=A.H_C.ATP', 1),
                            ('H_O.ATP+A=A.H_O.ATP', -1),
                            ('H_O.ATP=H_C.ATP', 1) ])
    kCOA = kOCA/KeqOCA
    ksrx[rx] = (kOCA, kCOA)

    # The following thermodynamic cycle should be satisified
    #
    # [A.M.H_C.ATP]/[A.M.H_O.ATP] = [A.M.H_C.ATP]/([A][M.H_C.ATP])
    #                             * [A][M.H_O.ATP]/[A.M.H_O.ATP]
    #                             * [M.H_C.ATP]/[M.H_O.ATP]
    for m in ['M_d', 'M_m', 'Q.M_d']:
        KOC = KeqFromRXCycle(ksrx,
                             [('%(m)s.H_C.ATP+A=A.%(m)s.H_C.ATP' % {'m':m}, 1),
                              ('%(m)s.H_O.ATP+A=A.%(m)s.H_O.ATP' % {'m':m}, -1),
                              ('%(m)s.H_O.ATP=%(m)s.H_C.ATP' % {'m':m}, 1)])
        rx0 = '%(m)s.H_O.ATP=%(m)s.H_C.ATP' % {'m': m}
        kOCm, kCOm = ksrx[rx0]

        kOC = max( kOCm, kOCA)
        kCO = kOC/KOC
        rx = 'A.%(m)s.H_O.ATP=A.%(m)s.H_C.ATP' % {'m': m}
        ksrx[rx] = (kOC, kCO)

    # Client binding to Hsp90
    for m in ['M_d', 'M_m', 'Q.M_d']:
        for h in ['H_O', 'H_O.ATP']:
            rx0 = '%(m)s+%(h)s=%(m)s.%(h)s' % {'m': m, 'h': h}
            rx = '%(m)s+A.%(h)s=%(m)s.A.%(h)s' % {'m': m, 'h': h}
            ksrx[rx] = ksrx[rx0]

    # Cochaperone binding to client-Hsp90 complex
    for h in ['H_O', 'H_O.ATP', 'H_C.ATP']:
        rx0 = 'M_d.%(h)s+Q=Q.M_d.%(h)s' % {'h': h}
        rx = 'A.M_d.%(h)s+Q=A.Q.M_d.%(h)s' % {'h': h}
        ksrx[rx] = ksrx[rx0]

    # Conformational change of the client
    rx0 = 'M_d.H_C.ATP=M_m.H_C.ATP'
    rx = 'A.M_d.H_C.ATP=A.M_m.H_C.ATP'
    ksrx[rx] = ksrx[rx0]
    
    return ksrx

def ATP_driven_conformer_rxs( rxs, ks, concs):
    cstmols = ['ATP', 'ADP', 'Pi']
    # Use the concentration of Hsp90 dimer
    _concs = dict(concs, **{'H': 0.5*concs['H']}) 
    return driven_conformer_rxs( rxs, ks, _concs, cstmols)

def ATP_cochaperone_driven_conformational_noneq( rates, concs):
    Keq = KeqFromRXCycle( rates,
                          [('M_i=M_d', 1),
                           ('M_d=M_m', 1),
                           ('M_m=M_a', 1)])
    include_AHA1 = (concs.get( 'A', 0) > 0)
    rxs, ks = ATP_cochaperone_conformational_cycle( rates, include_AHA1)
    
    Crx, ks, Nh, Nhmt0, mols, comps = ATP_driven_conformer_rxs( rxs, ks, concs)
    mss = ness( Crx, ks, Nh=Nh, Nhmt0=Nhmt0, tinit=None, check_steady_state=True)
    c = mss.x
    mol2id = dict( [(m, i) for i, m in enumerate(mols)])
    noneq = (c[mol2id['M_a']]/c[mol2id['M_i']])/Keq
    return noneq, c, mol2id

def ness_vs_krx( rates, concs, rx, krx):
    '''Compute the steady state molecular concentrations at different rates 
    for a given reaction.

    Args:

    rates: dictionary mapping reactions to forward and reverse rate constants.

    concs: concentrations of molecular components.

    rx: the reaction whose rates to be changed in different
    conditions. rx may be a list of reactions.

    krx: Tx2 matrix, where krx[t,0] and krx[t,1] are the forward and
    reverse rate constants in condition t. If rx is a list of reactions,
    krx should be a T x len(rx) x 2 matrix. 

    Returns:

    cmols: TxM matrix of steady state molecular concentrations.

    mols: length M array of the molecular species, in the same order as cmols.

    noneq: length T of floats, giving nonequilibrium factor [M_a]/[M_i] / Keq

    '''
    include_AHA1 = (concs.get('A', 0)>0)
    rxs, ks = ATP_cochaperone_conformational_cycle( rates, include_AHA1)
    Crx, ks, Nh, cc, mols, cmps = ATP_driven_conformer_rxs(rxs, ks, concs)
    molid = dict([(m,i) for i, m in enumerate(mols)])
    U, S, Vh = svd( Crx)
    
    cmols = np.zeros( (len(krx), len(mols)))
    noneq = np.zeros( len(krx))
    mt0 = None
    for t, k in enumerate(krx):
        if isinstance( rx, list):
            _rates = dict( rates, **dict( list(zip(rx, k))))
        else:
            _rates = dict( rates, **{rx: k})
        Keq = KeqFromRXCycle( _rates, 
                              [ ('M_i=M_d', 1),
                                ('M_d=M_m', 1),
                                ('M_m=M_a', 1) ])
        rxs, ks = ATP_cochaperone_conformational_cycle( _rates, include_AHA1)
        Crx, ks, Nh, cc, mols, cmps = ATP_driven_conformer_rxs(rxs, ks, concs)
        if (mt0 is None):
            tinit = 1.
            mss = ness(Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=cc, tinit=tinit)
        else:
            mss = ness( Crx, ks, mt0=mt0, U=U, S=S, Vh=Vh, Nh=Nh)   
        cmols[t,:] = mss.x
        noneq[t] = cmols[t,molid['M_a']]/cmols[t,molid['M_i']] / Keq
        mt0 = mss.x
    return cmols, mols, noneq

def ness_vs_concs( rates, concs, components, ccs):
    '''Compute the steady state molecular concentrations at different
    concentrations for a given component.

    Args:

    rates: dictionary mapping reactions to forward and reverse rate constants.

    concs: concentrations of molecular components.

    components: string or list of strings, the components whose
    concentrations will be varied.

    ccs: length TxC array of floats, the concentrations of the components.

    Returns:

    cmols: TxM matrix of steady state molecular concentrations.

    mols: length M array of the molecular species, in the same order as cmols.

    noneq: length T of floats, giving nonequilibrium factor [M_a]/[M_i] / Keq

    '''
    include_AHA1 = (concs.get('A', 0)>0)
    rxs, ks0 = ATP_cochaperone_conformational_cycle( rates, include_AHA1)
    
    Crx, ks, Nh, cc, mols, cmps = ATP_driven_conformer_rxs(rxs, ks0, concs)
    molid = dict([(m,i) for i, m in enumerate(mols)])
    U, S, Vh = svd( Crx)
    
    Keq = KeqFromRXCycle( rates, 
                          [ ('M_i=M_d', 1),
                            ('M_d=M_m', 1),
                            ('M_m=M_a', 1) ])
    cmols = np.zeros( (len(ccs), len(mols)))
    noneq = np.zeros( len(ccs))
    for t, cs in enumerate(ccs):
        tinit = 10.
        _concs = dict( concs, **dict( list(zip(components, [cs]*len(components)))))
        Crx, ks, Nh, cc, mols, cmps = ATP_driven_conformer_rxs(rxs, ks0, _concs)
        mss = ness( Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=cc, tinit=tinit)   
        cmols[t,:] = mss.x
        noneq[t] = cmols[t,molid['M_a']]/cmols[t,molid['M_i']] / Keq
        mt0 = mss.x
    return cmols, mols, noneq

def dATP_vs_krx( rates, concs, rx, krx):
    ''' Compute the consumption rate of ATP at steady state.

    Args:

    rates: dictionary mapping reactions to forward and reverse rate constants.

    concs: concentrations of molecular components.

    rx: the reaction whose rates to be changed in different
    conditions. rx may be a list of reactions.

    krx: Tx2 matrix, where krx[t,0] and krx[t,1] are the forward and
    reverse rate constants in condition t. If rx is a list of reactions,
    krx should be a T x len(rx) x 2 matrix. 

    Returns:

    dATP: length T array, dATP/dt at steady state.
    '''
    include_AHA1 = (concs.get('A', 0)>0)
    rxs, ks = ATP_cochaperone_conformational_cycle( rates, include_AHA1)
    nATPs = consumptive_reactions( rxs, 'ATP')
    Crx, ks, Nh, cc, mols, cmps = ATP_driven_conformer_rxs( rxs, ks, concs)
    
    molid = dict([(m,i) for i, m in enumerate(mols)])
    U, S, Vh = svd( Crx)
    
    dATP = np.zeros( len(krx))
    mt0 = None
    for t, k in enumerate(krx):
        if isinstance( rx, list):
            _rates = dict( rates, **dict( list(zip(rx, k))))
        else:
            _rates = dict( rates, **{rx: k})
        rxs, ks = ATP_cochaperone_conformational_cycle( _rates, include_AHA1)
        Crx, ks, Nh, cc, mols, cmps = ATP_driven_conformer_rxs(rxs, ks, concs)
        if (mt0 is None):
            tinit = 1.
            mss = ness(Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=cc, tinit=tinit)
        else:
            mss = ness( Crx, ks, mt0=mt0, U=U, S=S, Vh=Vh, Nh=Nh)   
        mt0 = mss.x
        
        dATP[t] = consumption_rate( Crx, ks, nATPs, mss.x)
    
    return dATP
    
def dATP_vs_concs( rates, concs, components, ccs):
    ''' Compute the consumption rate of ATP at steady state.

    Args:

    rates: dictionary mapping reactions to forward and reverse rate constants.

    concs: concentrations of molecular components.

    components: string or list of strings, the components whose
    concentrations will be varied.

    ccs: length TxC array of floats, the concentrations of the components.

    Returns:

    dATP: length T array, dATP/dt at steady state.
    '''
    include_AHA1 = (concs.get('A', 0)>0) or ('A' in components)
    rxs, ks0 = ATP_cochaperone_conformational_cycle( rates, include_AHA1)
    nATPs = consumptive_reactions( rxs, 'ATP')
    Crx, ks, Nh, cc, mols, cmps = ATP_driven_conformer_rxs(rxs, ks0, concs)
    molid = dict([(m,i) for i, m in enumerate(mols)])
    U, S, Vh = svd( Crx)
    
    Keq = KeqFromRXCycle( rates, 
                          [ ('M_i=M_d', 1),
                            ('M_d=M_m', 1),
                            ('M_m=M_a', 1) ])
    dATP = np.zeros( len(ccs))
    for t, cs in enumerate(ccs):
        tinit = 10.
        _concs = dict( concs, **dict( list(zip(components, [cs]*len(components)))))
        Crx, ks, Nh, cc, mols, cmps = ATP_driven_conformer_rxs(rxs, ks0, _concs)
        mss = ness( Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=cc, tinit=tinit)   
        mt0 = mss.x

        dATP[t] = consumption_rate( Crx, ks, nATPs, mss.x)
    return dATP

def conformational_equilibrium( rates, M=1.):
    '''Compute the equilibrium distribution of the client conformations,
    without Hsp90.  Returns the equilibrium concentrations of M_i,
    M_d, M_m, M_a, in that order.

    '''
    kid, kdi = rates['M_i=M_d']
    kdm, kmd = rates['M_d=M_m']
    kma, kam = rates['M_m=M_a']
    
    Kdi = kdi/kid
    Kmd = kmd/kdm
    Kam = kam/kma

    a = M/(1 + Kam + Kam*Kmd + Kam*Kmd*Kdi)
    m = Kam*a
    d = Kmd*m
    i = Kdi*d

    return (i, d, m, a)

def state_distribution( cmols, molid):
    '''
    Compute the populations of apo open, ATP-bound open, ATP-bound closed, 
    and ADP-bound closed states of Hsp90.
    '''
    states = [ 'H_O', 'H_O.ATP', 'H_C.ATP', 'H_C.ADP' ]
    mols = list(molid.keys())
    statemols = {
        'H_O': [m for m in mols if 'H_O' in m and 'H_O.ATP' not in m],
        'H_O.ATP': [m for m in mols if 'H_O.ATP' in m],
        'H_C.ATP': [m for m in mols if 'H_C.ATP' in m],
        'H_C.ADP': [m for m in mols if 'H_C.ADP' in m] }

    ndim = cmols.ndim
    if 1==ndim:
        cmols = cmols.reshape( (1, len(cmols)))
    cstates = np.array([ np.sum( cmols[:,[ molid[m] for m in statemols[state]]],
                                 axis=1)
                         for state in states ])
    pstates = cstates/np.sum(cstates, axis=0)
    if 1==ndim:
        pstates.reshape( (len(states),))
    pstates = dict([(state, pstates[s]) for s, state in enumerate(states)])
    return pstates

def ATP_cochaperone_conformational_cycle_layout():

    rmol = { 'M_d': (0, 0),
             'M_d.H_O': (0, -0.5),
             'M_d.H_O.ATP': (-0.25, -1.0),
             'M_d.H_C.ATP': (-0.05, -1.35),
             'M_d.H_C.ADP': (0.2, -0.85)}
    for m in rmol: rmol[m] = np.array( rmol[m])

    dr = np.array( [ 0.65, 0.])
    for i, ri in list(rmol.items()):
        a = i.replace('M_d', 'M_m')
        rmol[a] = dr + ri

    dr = np.array( [ -0.75, 0 ])
    rmol['M_i'] = dr + rmol['M_d']
    dr = np.array( [ 0.75, 0 ])
    rmol['M_a'] = dr + rmol['M_m']

    dr = np.array( [-0.38, -0.15])
    for i in [ 'M_d', 'M_d.H_O', 'M_d.H_O.ATP', 'M_d.H_C.ATP' ]:
        q = 'Q.' + i
        rmol[q] = rmol[i] + dr

    return rmol

def parameterize_rates( rates, y):
    kddH, kma, kdQM, kdQMH, kdQMHA, kCOd = y

    y0 = np.array( [
        rates['M_d+H_O=M_d.H_O'][1],
        # rates['M_d=M_m'][1],
        rates['M_m=M_a'][0],
        rates['M_d+Q=Q.M_d'][1],
        rates['M_d.H_O.ATP=M_d.H_C.ATP'][1],
        rates['M_d.H_O.ATP+Q=Q.M_d.H_O.ATP'][1],
        rates['M_d.H_O+Q=Q.M_d.H_O'][1]])

    kaMH = rates['M_d+H_O=M_d.H_O'][0]
    # kdm = rates['M_d=M_m'][0]
    kam = rates['M_m=M_a'][1]
    kaQM = rates['M_d+Q=Q.M_d'][0]
    kaQMH = rates['M_d.H_O+Q=Q.M_d.H_O'][0]
    kaQMHA = rates['M_d.H_O.ATP+Q=Q.M_d.H_O.ATP'][0]
    kOCd = rates['M_d.H_O.ATP=M_d.H_C.ATP'][0]
    
    rates['M_d+H_O=M_d.H_O'] = (kaMH, kddH)
    # rates['M_d=M_m'] = (kdm, kmd)
    rates['M_m=M_a'] = (kma, kam)
    # kdm, kmd = rates['M_d=M_m']
    rates['M_d+Q=Q.M_d'] = (kaQM, kdQM)
    rates['M_d.H_O+Q=Q.M_d.H_O'] = (kaQMH, kdQMH)
    rates['M_d.H_O.ATP+Q=Q.M_d.H_O.ATP'] = (kaQMHA, kdQMHA)
    rates['M_d.H_O.ATP=M_d.H_C.ATP'] = (kOCd, kCOd)

    # Impose the constraint that M_d.H_C.ATP=M_m.H_C.ATP has an equilibrium
    # constant of 1. The two intermediate client conformations should be 
    # of roughly equal stability when clamped by Hsp90.
    kdmH = kddH*(kCOd/rates['M_m.H_O.ATP=M_m.H_C.ATP'][1]) # *(kdm/kmd)
    rates['M_m+H_O=M_m.H_O'] = (kaMH, kdmH)
    
    return rates, y0

def fit_to_normalized_client_activity( outpkl):
    '''
    We assume that the normalized activity is given by 

    A = [M_a](H, Q)/[M_a](H=0, Q=0).
    '''
    # Load the experimental data from Fig. 1B of Boczek et al. 2015.
    import os
    DIR = 'data/hsp90/'
    def loaddata( fname):
        return np.loadtxt( os.path.join( DIR, fname), delimiter=',')
    data = dict(
        H=loaddata( 'vSrc-Hsp90.Boczek2015.csv'),
        QH=loaddata( 'vSrc-Hsp90-CDC37E.Boczek2015.csv'))

    # Actual ATP=20uM
    concs = dict( ATP=20., ADP=1., Pi=1., M=0.32, Q=1.3, H=1.3)
    
    if (True):
        npts = len(data['H']) + len(data['QH'])
        Qs = np.concatenate( [ np.zeros( len(data['H'])), data['QH'][:,0] ])
        Hs = np.concatenate( [ data['H'][:,0], data['QH'][:,0] ])
        As = np.concatenate( [ data['H'][:,1], data['QH'][:,1] ])
    else:
        npts = len(data['QH'])
        Qs = data['QH'][:,0]
        Hs = data['QH'][:,0]
        As = data['QH'][:,1]

    Hs *= 0.5  # Hsp90 dimer concentration

    # Set up the fitting problem
    rates = BASE_RATES

    # We will use sympy to carry out symbolic calculations including
    # differentiations.
    from sympy import symbols, lambdify, diff

    kddH, kma, kdQM, kdQMH, kdQMHA, kCOd = symbols( 'kddH kma kdQM kdQMH kdQMHA kCOd')
    ys = [ kddH, kma, kdQM, kdQMH, kdQMHA, kCOd ]
    rates, ys0 = parameterize_rates( rates, ys)

    rxs, ks = ATP_cochaperone_conformational_cycle( rates)
    
    Crx, ks, Nh, cc, mols, comps = ATP_driven_conformer_rxs( rxs, ks, concs)
    molid = dict( [(m, i) for i, m in enumerate(mols)])
    compid = dict( [(c, i) for i, c in enumerate(comps)])

    U, S, Vh = svd( Crx)
    
    for r, k in zip( rxs, ks):
        print(r, k)

    kofyfunc = lambdify( tuple(ys), ks, 'numpy')

    # Find the non-zero dk/dy, store them as tuples (r, d, j, dk[r,d]/dy[j])
    nzdkdys = []
    for r, k in enumerate( ks):
        # go through forward and reverse
        for d, kd in enumerate( k):
            for j, y in enumerate( ys):
                grad = diff( kd, y)
                if grad == 0: continue
                nzdkdys.append( (r, d, j, lambdify( ys, grad)))

    # print nzdkdys
    def kofy( y):
        return np.array( kofyfunc( y[0], y[1], y[2], y[3], y[4], y[5]))
        
    def dkdy( y):
        dkvals = [ (r, d, j, dk( y[0], y[1], y[2], y[3], y[4], y[5])) 
                   for r, d, j, dk in nzdkdys ]
        return dkvals

    tinit = 10.
    mt0s = [ None for t in range( npts) ]

    def activity( y, t):
        ks = kofy( y)
        cc[compid['Q']] = Qs[t]
        cc[compid['H']] = Hs[t]
        if (mt0s[t] is None):
            mss = ness( Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=cc, tinit=tinit)
            mt0s[t] = mss.x
        else:
            mss = ness( Crx, ks, mt0=mt0s[t], U=U, S=S, Vh=Vh, Nh=Nh)
        return mss.x[molid['M_a']]

    def dactivity( y, t):
        ks = kofy( y)
        cc[compid['Q']] = Qs[t]
        cc[compid['H']] = Hs[t]
        dks = dkdy( y)
        if (mt0s[t] is None):
            mss = ness( Crx, ks, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=cc, tinit=tinit)
            mt0s[t] = mss.x
        else:
            mss = ness( Crx, ks, mt0=mt0s[t], U=U, S=S, Vh=Vh, Nh=Nh)
            
        dmdy = dness_dy( Crx, ks, dks, len(y), mss.x, U=U, S=S, Vh=Vh, Nh=Nh)
                             
        return dmdy[molid['M_a'],:]

    tol = 3e-6
    diff = check_grad( lambda ly: activity( np.exp(ly), -1),
                       lambda ly: dactivity( np.exp(ly), -1)*np.exp(ly),
                       np.log(ys0))
    report = 'Gradient check for activity %g' % diff
    if (diff<tol):
        print('SUCCESS: %s < %g' % (report, tol))
    else:
        print('FAIL: %s > %g' % (report, tol))

    def residual( lny, t):
        y = np.exp( lny)
        a0 = activity( y, 0) # the first data point has Q=H=0.
        a = activity( y, t)
        return a/a0 - As[t]

    def dresidual( lny, t):
        y = np.exp( lny)
        a0 = activity( y, 0) # the first data point has Q=H=0.
        ia0 = 1./a0
        da0 = dactivity( y, 0)*y
        a = activity( y, t)
        da = dactivity( y, t)*y
        return ia0*(da - a*da0*ia0)

    diff = check_grad( lambda ly: residual( ly, -1),
                       lambda ly: dresidual( ly, -1),
                       np.log(ys0))
    report = 'Gradient check for residual %g' % diff
    if (diff<tol):
        print('SUCCESS: %s < %g' % (report, tol))
    else:
        print('FAIL: %s > %g' % (report, tol))

    result = least_squares( lambda ly: [residual(ly, t) for t in range(npts)],
                            np.log(ys0),
                            lambda ly: [dresidual(ly, t) for t in range(npts)],
                            method = 'trf')

    result.x = np.exp(result.x)
    print(result)
    pickle.dump( result, file( outpkl, 'wb'))
    return result

def test_ATP_cochaperone_conformational_cycle( Q, H, AHA1=0):
    include_AHA1 = (AHA1>0)
    rates = BASE_RATES
    rxs, ks = ATP_cochaperone_conformational_cycle( rates, include_AHA1)
    ksrx = dict( list(zip(rxs, ks)))
    rxs_sorted = list(ksrx.keys())
    rxs_sorted.sort()
    print('%d reactions in total:' % len(rxs_sorted))
    for rx in rxs_sorted:
        kf, kr = ksrx[rx]
        print('%25s %.3e %.3e' % (rx, kf, kr))

    print('Testing ATP cycle')
    atpcycle = [ 'H_O+ATP=H_O.ATP',
                 'H_O.ATP=H_C.ATP',
                 'H_C.ATP=H_C.ADP+Pi',
                 'H_C.ADP=H_O+ADP' ]
    kscycle = [ ksrx[rx] for rx in atpcycle ]
    testCycleClosure( atpcycle + [ 'ATP=ADP+Pi' ],
                      np.concatenate( [kscycle, [ (ATP_HYDROLYSIS_KEQ, 1.) ]]))

    print('Testing d-to-m pathway cycle')
    dtom = [ 'M_d+H_O=M_d.H_O',
             'M_d.H_O+ATP=M_d.H_O.ATP',
             'M_d.H_O.ATP=M_d.H_C.ATP',
             'M_d.H_C.ATP=M_m.H_C.ATP',
             'M_m.H_C.ATP=M_m.H_C.ADP+Pi',
             'M_m.H_C.ADP=M_m.H_O+ADP',
             'M_m+H_O=M_m.H_O',
             'M_d=M_m' ]
    ksdtom = [ ksrx[rx] for rx in dtom ]
    testCycleClosure( dtom + [ 'ATP=ADP+Pi' ],
                      np.concatenate( [ksdtom, [ (ATP_HYDROLYSIS_KEQ, 1.) ]]))
    
    print('Testing all reactions')
    testCycleClosure( rxs + [ 'ATP=ADP+Pi' ], 
                      np.concatenate( [ks, [ (ATP_HYDROLYSIS_KEQ, 1.) ]]))

    concs = {'M': 0.32,
             'H': H,
             'Q': Q,
             'ATP': 20,
             'ADP': 1,
             'Pi': 1}

    if include_AHA1: concs.update( **{'A': AHA1})

    neq, c, molid = ATP_cochaperone_driven_conformational_noneq( rates, concs)
    for m in molid: print('[%s] = %g' % (m, c[molid[m]]))
    pstates = state_distribution( c, molid)
    for s in pstates: print('p(%s) = %g' % (s, pstates[s]))

    print('ATP and cochaperone-driven conformational nonequilibrium: %f' % neq)
    print('[Active]=%.3g' % c[molid['M_a']])
    return
    for m in molid:
        print(m, c[molid[m]])
    for rx, k in zip(rxs, ks):
        print(rx, k)

import argparse

def opts():
    parser = argparse.ArgumentParser(
        description='Analysis of Hsp90-mediated nonequilibrium activation of client kinases.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '-Q', '--CDC37', type=float, default=1.3,
                         help='CDC37 concentration.')
    parser.add_argument( '-H', '--Hsp90', type=float, default=1.3,
                         help='Hsp90 concentration.')
    parser.add_argument( '--AHA1', type=float, default=0,
                         help='AHA1 concentration')
    parser.add_argument( '--fit-pkl', default='params.pkl',
                         help='Pickle filename to output fit parameters.')
    return parser

def main( args):
    test_ATP_cochaperone_conformational_cycle( args.CDC37, args.Hsp90,
                                               args.AHA1)
    if args.fit_pkl != 'no':
        fit_to_normalized_client_activity( args.fit_pkl)
    
if __name__ == '__main__':
    main( opts().parse_args())
