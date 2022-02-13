import numpy as np
from scipy.linalg import null_space
from sympy import Matrix
import networkx as nx

def reactants( rx, return_coeff=False):
    '''
    Get the reactants and products of the reaction.
    
    Args:

    rx: string, representing the reaction, in the form of, e.g.,
    R1 + n2*R2 + R3 = P1 + P2

    Return:

    (rxtant, products): tuple of list, rxtant are the reactants and products
    are the products of the reaction.

    coeffs: Optional, list of numbers, each representing the
    coefficient in front of the molecule in the reaction.  Products
    have negative coefficients. Only returned when return_coeff=True.

    '''
    rx = rx.replace( ' ', '')
    has_coeff = (rx.find( '*')>=0)

    lhs, rhs = rx.split('=')
    lhs, rhs = lhs.split('+'), rhs.split('+')
        
    coeffmol = [ dict(), dict() ]
    rxtants, products = [], []
    for mols, rxmols, sgn in zip([ lhs, rhs ], [rxtants, products], [1, -1]):
        side = (1 - sgn)//2 # 0 for left side and 1 for right side.
        for i, mol in enumerate( mols):
            d = mol.split( '*')
            if len(d) == 2:
                c, m = float(d[0]), d[1]
                if m in coeffmol[side]:
                    coeffmol[side][m] += sgn*c
                else:
                    coeffmol[side][m] = sgn*c
                    rxmols.append( m)
            else:
                m = d[0]
                if m in coeffmol[side]:
                    coeffmol[side][m] += sgn
                else:
                    coeffmol[side][m] = sgn
                    rxmols.append( m)
            
    n = 0
    coeffs = np.zeros(len(rxtants) + len(products), dtype=int)
    for side, rxmols in enumerate([rxtants, products]):
        for mol in rxmols:
            coeffs[n] = coeffmol[side][mol]
            n += 1

    if return_coeff:
        return (rxtants, products), coeffs
    else:
        return rxtants, products

def allMolecules( rxs):
    '''
    Return the list of all molecules involved in a set of reactions.
    '''
    mols = set()
    for r in rxs:
        rxtants, products = reactants( r)
        mols.update( rxtants + products)
    return mols

def reactionMatrix( rxs):
    '''
    Create the reaction matrix from the list of reactions.  The
    reaction matrix N is a RxM matrix, where R=len(rxs) is the number
    of reactions and M is the total number of molecular species
    participating in the reactions.  The element N_{r,m} gives the
    coefficient of molecule m in reaction r (N_{r,m}!=0 indicates that
    m is involved in reaction r.  For example, in the following set of
    reactions:

    A + B2 = A.B2
    2*B = B2

    There are R=2 reactions involving M=4 molecular species (A, B, B2, A.B2),
    and the corresponding reaction matrix is 

        / 1   0   1   -1 \   
    N = |                |
        \ 0   2  -1    0 /

    Note that the products are represented by negative coefficients.

    Args:
    
    rxs: a list of strings, each string representing a reaction.

    Returns:
    
    N: a numpy matrix of integers. The reaction matrix.

    molecules: the list of molecular species in the reactions, in the same
    order as the columns of the reaction matrix.

    '''
    molecules = []
    mol2id = dict()
    # coeffs is a list of (r, m, n), which is a sparse matrix representation
    # of N[r,m] = n.
    coeffs = []
    for r, rx in enumerate( rxs):
        (rxtants, products), ns = reactants( rx, return_coeff=True)
        for i, mol in enumerate( list(rxtants) + list(products)):
            m = mol2id.get( mol, None)
            if m is None:
                m = len(molecules)
                molecules.append( mol)
                mol2id[mol] = m
            coeffs.append( (r, m, ns[i]))
    
    # Construct the dense numpy matrix from the sparse matrix.
    N = np.zeros( (len(rxs), len(molecules)), dtype=float)
    for r, m, n in coeffs:
        N[r,m] = n
    
    return N, molecules
    
def reactionCycles( rxmat):
    '''
    Find all reaction cycles in a reaction matrix.  A reaction cycle
    is a set of reactions, when taken together, return all products
    back to the reactants.  For example, the three reactions
    
    A+B=C+D
    
    C=A
    
    D=B

    form a reaction cycle.

    The reaction cycles are found using the null space of the reaction matrix.

    Args:

    rxmat: the reaction matrix representing a list of reactions.

    Returns:

    rc: numpy matrix, each rc[c] representing a reaction cycle such that 
    \sum_r rc[c,r] rxs[r] = 0.
    '''
    Nt = Matrix( rxmat.T.astype( int))
    rc = Nt.nullspace()
    if not rc: return None
    else: return np.array( rc)

def thermodynamicCycleClosure( rxs, ks):
    '''
    Check thermodynamic cycle closure for a list of reactions.  For each
    reaction cycle, calculate the sum of n*ln(kf/kr) over the cycle; 
    thermodynamics dictate that the sum should be 0 for each cycle.
    '''
    N, mols = reactionMatrix( rxs)
    rxcycles = reactionCycles( N)
    
    if rxcycles is None: return None

    # Equilibrium free energy change for each reaction in standard state.
    dGs = np.log(ks[:,0]/ks[:,1])
    ddGs = rxcycles.dot( dGs)

    return ddGs

def symbolicReactionCycle( rxs, cycle):
    '''
    Construct a symbolic representation of the reaction cycle.

    Args:

    rxs: a list of strings, each string representing a reaction.

    cycle: a numpy array, the r'th element gives the coefficient of
    the reaction r in the cycle, such that \sum_r cycle[r] rxs[r] = 0.

    Return:

    A list of strings representing the reactions forming the cycle.
    '''
    crxs = []
    for r, c in enumerate( cycle):
        if c!=0:
            crxs.append( '%d*(%s)' % (c, rxs[r]))
    return crxs

def KeqFromk( ks):
    '''Return the equilibrium constant of the reaction, given the forward
    and reverse rate constants.
    '''
    kf, kr = ks
    return kf/kr

def KeqFromRXCycle( ks, rxs):
    '''
    Compute the product of the equilibrium constants of a series of reactions
    in part of a reaction cycle. 
    
    Args:

    ks: a dictionary of rx: (kf, kr).

    rxs: a list of reactions and their directions (rx, direction) in the cycle,
    where direction=1 indicates forward reaction and direction=-1 indicates
    reverse reaction.

    Returns:

    Keq: the products of (kf/kr)^direction.
    '''
    Keq = 1.
    for rx, d in rxs:
        kf, kr = ks[rx]
        if 0==kf and 0==kr: continue
        if 1==d: 
            Keq *= (kf/kr) # TODO: this can be generalized to (kf/kr)^d for d>1
        else: 
            Keq *= (kr/kf)
    return Keq

def fluxNet( Jrx, rxs, mols):
    '''Construct the reactive flux among the given set of molecules.

    Args:

    Jrx: a Rx3 matrix of numbers, or length R array.  If Rx3 matrix,
    Jrx[r][0,1,2] gives the forward, reverse, and net flux of reaction
    r; if length R array, Jrx[r] gives the net flux of reaction r.

    rxs: a list of strings, rxs[r] gives the r'th reaction.

    mols: a list of strings, representing the molecular species in the network

    Return:

    J: a NxN antisymmetric matrix, where J[i,j] = -J[j,i] gives the 
    net flux from molecule i to molecule j.

    '''
    molid = dict( [ (m, i) for i, m in enumerate( mols) ])
    molset = set( mols)
    
    if (Jrx.ndim==2 and Jrx.shape[1]==3):
        Js = Jrx[:,-1]
    else:
        Js = Jrx
    J = np.zeros( (len(mols), len(mols)))
    for k, rx in enumerate( rxs):
        reax, prods = reactants( rx)
        mi = molset.intersection( reax)
        mj = molset.intersection( prods)
        if not (len(mi)==1 and len(mj)==1): continue
        mi = mi.pop()
        mj = mj.pop()
        i, j = molid[mi], molid[mj]
        J[i,j] = Js[k]
        J[j,i] = -J[i,j]
    return J
    
def testCycleClosure( rxs, ks, tol=1e-9):
    '''
    Test the thermodynamic cycle closure for a set of reactions and their
    kinetic rates.  Report unclosed reaction cycles.
    '''
    ddG = thermodynamicCycleClosure( rxs, ks)

    rxcycles = reactionCycles( reactionMatrix( rxs)[0])
    success = True
    for c, cycle in enumerate( rxcycles):
        symbol = symbolicReactionCycle( rxs, cycle)
        if np.abs( ddG[c]) > tol:
            print('FAIL: reaction cycle violates thermodynamic closure!')
            success = False
        print(symbol)
        print('ddG = %g' % ddG[c])
    if success:
        print('SUCCESS: all reaction cycles satisfy thermodynamic closure.')
    
    return success

def reactionsWithConstMols( Crx, ks, constconcs):
    '''Construct the reaction matrix and the associated rate constants,
    when some of the molecules are hold at constant concentrations.
    
    Args:

    Crx: R x M matrix of reaction coefficients.

    ks: R x 2 matrix of reaction rates.

    constconcs: a list of 2-tuples (i, m), where i is the index of the
    molecule and m its concentration that is hold constant.

    Return:

    Crxc: R x (M - len(constconcs)) matrix of reaction coefficients,
    removing the molecules whose concentrations are constant.

    ksc: R x 2 matrix of reaction rates, which have absorbed the
    constant concentrations.

    '''
    ksc = ks.copy()
    for i, m in constconcs:
        Ci = Crx[:,i]
        # Forward reaction, k[r, 0] = k0[r, 0]*m^C[r,i]
        f = (Ci>0)
        ksc[f, 0] *= np.power( m, Ci[f])
        # Reverse reaction, k[r, 1] = k0[r, 1]*m^(-C[r,i])
        r = (Ci<0)
        ksc[r, 1] *= np.power( m, Ci[r])

    Crxc = np.delete( Crx, [ i for i, m in constconcs ], axis=1)

    return Crxc, ksc

def reaction_network_as_graph( mols, rx):
    '''
    Return the reaction network as a graph.  Each edge is associated with a 
    reaction or, sometimes, a list of reactions.
    '''
    g = nx.Graph()
    g.add_nodes_from( mols)
    for r in rx:
        R, P = reactants( r[0])
        if (g.has_edge(R[0], P[0])):
            if (isinstance( g[R[0]][P[0]]['reaction'], list)): 
                g[R[0]][P[0]]['reaction'].append( r)
            else:
                g[R[0]][P[0]]['reaction'] = [ g[R[0]][P[0]]['reaction'], r]
        else:
            g.add_edge( R[0], P[0], reaction=r)
    return g

def flux_network_as_graph( mols, J, n=None, epsilon=1e-15):
    '''
    Represent a network of reactive flux, given by the antisymmetric
    matrix J[i,j], as a directed graph, with each edge assigned weight
    J[i,j].  Optionally, each node i can be assigned the molecular
    fraction n[i].

    '''
    g = nx.DiGraph()
    if (n is not None):
        g.add_nodes_from( list(zip(mols, [{'weight': w} for w in n])))
    else:
        g.add_nodes_from( mols)
    for i, mi in enumerate(mols):
        for j in range(i+1, len(mols)):
            mj = mols[j]
            if (J[i,j] > epsilon):
                g.add_edge( mi, mj, weight=J[i,j])
            elif (J[i,j] < -epsilon):
                g.add_edge( mj, mi, weight=J[j,i])
    return g

def reaction_free_energy_change( kf, kr, rx, concs):
    '''
    Compute the free energy change of a reaction \sum_j C_j X_j = 0:

    \Delta G = -sum_j C_j ln[X_j] - ln(kf/kr).
    '''
    (rxtants, products), ns = reactants( rx, return_coeff=True)
    dG = -np.log( kf/kr)
    for i, mol in enumerate( list(rxtants) + list(products)):
        ci = ns[i]
        dG -= ci*np.log( concs[mol])
    return dG

def pathway_free_energy_landscape( mols_in_path, rxs, ks, concs):
    '''Compute the free energy landscape along a reaction pathway. For a 
    reaction step

    C_i X_i + C_{i+1} X_{i+1} + \sum_{j!=i and j!=i+1} C_j Y_j = 0

    the pathway free energy change is defined by

    \Delta G_i = -\sum_{j!=i and j!=i+1} C_j ln[Y_j] -ln(kf/kr)

    Args:

    mols_in_path: list of strings, the molecules in the pathway

    rxs: list of strings, all the reactions

    ks: 2-d numpy array. ks[r] gives the forward and reverse rates of the r'th reaction.

    concs: the concentrations of the molecules.

    Returns:

    dG: array of floats, dG[i] is the cumulative pathway free energy
    change up to the i'th molecule.

    '''
    molset = set(mols_in_path)
    if mols_in_path[-1] == mols_in_path[0]: 
        mols_in_path = mols_in_path[:-1]
        cyclic = True
    else:
        cyclic = False
    nddG = len(mols_in_path)  # Include the zero'th ddG = 0.
    if cyclic: nddG += 1

    mol2id = dict( [(m, i) for i, m in enumerate(mols_in_path)])
    ddG = np.zeros( nddG)
    # Set concentrations of all pathway molecules to standard condition.
    standard = 1. 
    concs = dict( concs, **dict( [ (m, standard) for m in mols_in_path ]))
    for rx, (kf, kr) in zip(rxs, ks):
        (rxtants, products), ns = reactants( rx, return_coeff=True)
        rmol = set(rxtants).intersection( molset)
        pmol = set(products).intersection( molset)
        if len(rmol)==0 or len(pmol)==0: continue
        if len(rmol)>1:
            raise ValueError('%d mols appear on the reactant sides of reaction %s: %s' % (len(rmol), rx, ', '.join( list(rmol))))
        if len(pmol)>1:
            raise ValueError('%d mols appear on the product sides of reaction %s: %s' % (len(pmol), rx, ', '.join( list(pmol))))
        rmol = rmol.pop()
        pmol = pmol.pop()
        rid = mol2id[rmol]
        pid = mol2id[pmol]
        di = pid - rid
        if cyclic:
            if abs(di)>1 and not abs(di) == len(mols_in_path) - 1: continue
        elif abs(di) > 1 : continue
        rxdG = reaction_free_energy_change( kf, kr, rx, concs)
        if abs(di)>1:
            if di > 0:
                ddG[pid+1] = -rxdG
            else:
                ddG[rid+1] = rxdG
        else:
            if 1==di:
                ddG[rid+1] = rxdG
            else:
                ddG[pid+1] = -rxdG
    dG = np.cumsum( ddG)
    return dG

def test_reaction_matrix( tol=1e-5):
    
    rxs = [
        'A + B2 + C = D + E',
        '2*B = B2',
        'E = A + 1*B + B',
        'D = C'
    ]

    ks = np.array( [
        [ 1.5, 0.5 ],
        [ 2.0, 1.0 ],
        [ 1.0, 0.8 ],
        [ 0.3, 0.3*(1.5/0.5*2.0/1.0*1.0/0.8) ] ])

    mols0 = [ 'A', 'B2', 'C', 'D', 'E', 'B' ]
    N0 = np.array( [
        [ 1, 1, 1, -1, -1, 0 ],
        [ 0, -1, 0, 0, 0, 2 ],
        [ -1, 0, 0, 0, 1, -2 ],
        [ 0, 0, -1, 1, 0, 0 ] ], dtype=float)
    
    N, mols = reactionMatrix( rxs)
    # print N
    # print mols
    if np.all(N == N0) and mols == mols0:
        print('Reaction matrix test PASSED!')
    else:
        print('Reaction matrix test FAILED!')

    # print reactionCycles( N)

    testCycleClosure( rxs, ks)

if __name__ == '__main__':
    test_reaction_matrix()

