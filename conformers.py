'''
Utilities for analyzing conformational changes.
'''

import numpy as np
from rxnet import *
from functools import reduce

def mass_equations( mols, not_conserved=None, delimiter='_'):
    '''
    Construct the set of equations for the mass conservation of molecules.
    
    Args:

    mols: a list of molecular species containing the components.

    not_conserved: a list of the components that are not conserved 
    (such as ATP, ADP, Pi, which may be regenerated in the system).

    delimiter: '_' if a molecular species is represented by COMPONENT_CONFORMER.

    Returns:

    masseq: a C x M numpy array of integers, so that nc = masseq[c] 
    gives the mass conservation of component c in the form of
    c_0 = \sum_i nc[i]*mols[i]
    
    components: a list of components among the molecular species.
    '''
    molid = dict( [ (m, i) for i, m in enumerate(mols) ])
    def components( mol):
        '''
        Return the components of the molecular species. 
        '''
        # Remove the branch binding indicators.
        mol = mol.replace( '(', '').replace( ')', '')
        return [_f for _f in [ c.lstrip(r'0123456789').split( delimiter)[0] 
                               for c in mol.split( '.') ] if _f]
    
    molcomps = [ components(m) for m in mols ]
    allcomps = set( reduce( lambda x, y: x + y, molcomps))

    if not_conserved is not None:
        for c in not_conserved:
            try:
                allcomps.remove( c)
            except KeyError:
                pass

    allcomps = list( allcomps)
    compid = dict( [ (c, i) for i, c in enumerate( allcomps) ])
    
    masseq = np.zeros( (len(allcomps), len(mols)), dtype=int)
    for i, m in enumerate( mols):
        for c in molcomps[i]:
            cid = compid.get( c, None)
            if cid is None: continue
            masseq[cid][i] = 1

    return np.array(masseq), allcomps

def symbolic_mass_equations( masseq, mols, components):
    '''
    Construct a symbolic representation of the mass conservation equations.

    Args:
    
    masseq: a two-dimensional numpy array of integers, so that nc = masseq[c] 
    gives the mass conservation of component c in the form of
    c_0 = \sum_i nc[i]*mols[i]
    
    components: a list of components among the molecular species.

    Returns:

    symboleq: a list of symbolic equations representing the mass conservations.
    '''
    def symbolic_eq( eq):
        return ' + '.join( [ 
            ('' if eq[i]==1 else str(eq[i])) + ('[%s]' % mols[i]) 
            for i in range(len(mols)) if eq[i] != 0 ])

    masseqsymbol = []
    for c, eq in enumerate( masseq):
        mass = '[%s]_0' % components[c]
        mass = mass + ' = ' + symbolic_eq( eq)
        masseqsymbol.append( mass)
    
    return masseqsymbol

def driven_conformer_rxs( rxs, ks, concs, cstmols):
    '''Construct the reaction system where the conformational changes are driven
    by some molecules whose concentrations are hold constant (e.g. by 
    regenerative input).

    Args:

    rxs: a length R list of strings each representing a reaction.

    ks: a Rx2 array of forward and reverse rate constants.

    concs: a dictionary of molecular concentrations.

    cstmols: a list of molecules that are hold at constant concentrations.

    Returns:

    Crx: the R x M reaction coefficient matrix

    ks: the R x 2 array of reaction rate constants, absorbing the
    constant molecular concentrations.

    Nh: C x M array, where Nh.m = cc, m is the concentration
    of the molecules, cc is the total concentrations of components. 

    cc: length C array of total concentrations of components.

    mols: a list of molecules, in the same order the columns of Crx.

    comps: a list of components, each component can be found in several 
    different molecules.
    '''
    Crx0, mols = reactionMatrix( rxs)
    molid = dict([(m, i) for i, m in enumerate(mols)])
    cstconcs = [(molid[m], concs[m]) for m in cstmols]
    
    Crx, ks = reactionsWithConstMols( Crx0, ks, cstconcs)
    mols = [m for m in mols if m not in cstmols]
    Nh, comps = mass_equations( mols, cstmols)
    Nhmt0 = np.array( [ concs[c] for c in comps ])

    return Crx, ks, Nh, Nhmt0, mols, comps
    
def consumptive_reactions( rxs, consumed):
    '''Find all the reactions that consume certain molecular components
    (e.g. ATP) and produce the corresponding products (e.g. ADP and
    Pi).

    Args:

    rxs: a list of strings representing the set of reactions.

    consumed: string, the name of the consumed molecular component,
    such as 'ATP'.

    Returns:

    nCs: a len(rxs) array of numbers, nCs[i] is the number of consumptions of
    the consumed molecule in the i'th reaction.

    '''
    nCs = np.zeros( len(rxs), dtype=int)
    for j, rx in enumerate( rxs):
        (rxtants, products), ns = reactants( rx, True)
        mols = rxtants + products
        Cs = [ consumed in m for m in mols ]
        nCs[j] = np.sum( ns[Cs])
    return nCs

def test_mass_equations():
    mols = [ 'M_i', 'M_i.ADP', 'M_i.ATP', 'M_a', 'M_a.ADP', 'M_a.ATP',
             'H_O.M_i', 'H_C.M_a', 'H_O', 'H_C' ]
    eq, comps = mass_equations( mols, ['ADP', 'ATP'])
    symbolic = symbolic_mass_equations( eq, mols, comps)
    for se in symbolic:
        print(se)

    print('Conserved components are', comps)
    success = True
    
    eq0 = dict(M=[ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 ],
               H=[ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 ])
    
    Nh = np.array( [ eq0[c] for c in comps ])
    
    print('Nh0 = \n', Nh)
    print('Nh = \n', eq)
    if np.any(eq - Nh):
        print('FAIL: Conservation equation disagree with expected.')
        success = False
    else:
        print('SUCCESS: Conservation equation agrees with the expected.')

    return success

if __name__ == '__main__':
    test_mass_equations()
