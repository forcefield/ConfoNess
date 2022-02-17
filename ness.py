'''
Nonequilibrium steady state analysis.
'''

import numpy as np
from scipy.optimize import root, check_grad
from scipy.optimize import least_squares, nnls
from scipy.linalg import svd, null_space
from scipy.linalg import lstsq
from scipy.integrate import solve_ivp, cumtrapz
from scipy.stats import linregress

from rxnet import reactants, allMolecules
from functools import reduce

def flux_in_reaction( crx, krx, cmols):
    '''Compute the reactive flux in a given reaction.

    Args:
    
    crx: length M array of reaction coefficient. crx[i] gives the
    coefficient of molecule i in the reaction.

    krx: length 2 list of floats, forward and reverse rate constants.

    cmols: length M array of molecular concentrations.

    Return:

    J: float, instantaneous flux in the reaction.
    '''
    kf, kr = krx
    M = len(cmols)
    reaxants = np.arange( M)[crx > 0]
    products = np.arange( M)[crx < 0]
    # forward flux: k_f \prod_{j|c_j > 0} m_j^{c_j}
    Jf = kf*np.prod( np.power(cmols[reaxants], crx[reaxants]))
    # reverse flux: k_r \prod_{j|c_j < 0} m_j^{-c_j}
    Jr = kr*np.prod( np.power(cmols[products], -crx[products]))

    return Jf - Jr

def dflux_dcmol( crx, krx, cmols):
    '''Compute the derivative of the reactive flux with respect to the 
    molecular concentrations.

    Args:
    
    crx: length M array of reaction coefficient. crx[i] gives the
    coefficient of molecule i in the reaction.

    krx: length 2 list of floats, forward and reverse rate constants.

    cmols: length M array of molecular concentrations.

    Return:

    dJ/dm_i: length M vector of dJ/dm_i.
    '''
    kf, kr = krx
    M = len(cmols)
    reaxants = np.arange( M)[crx > 0]
    products = np.arange( M)[crx < 0]
    # forward flux: k_f \prod_{j|c_j > 0} m_j^{c_j}
    Jf = kf*np.prod( np.power(cmols[reaxants], crx[reaxants]))
    # reverse flux: k_r \prod_{j|c_j < 0} m_j^{-c_j}
    Jr = kr*np.prod( np.power(cmols[products], -crx[products]))
    
    icmols = np.zeros( M)
    nz = np.nonzero( cmols)
    icmols[nz] = 1./cmols[nz]

    dJ = np.zeros( M)
    dJ[reaxants] += Jf*crx[reaxants]*icmols[reaxants]
    dJ[products] += Jr*crx[products]*icmols[products]

    # For m_i = 0 and c_{i} == 1
    # dJ/dm_i = kf c_i \prod_{j!=i|c_{kj}>0} m_j^{c_j} if c_i = 1
    # dJ/dm_i = kr c_i \prod_{j!=i|c_{kj}<0} m_j^{c_j} if c_j = -1
    for p, i in enumerate(reaxants):
        if cmols[i]==0 and crx[i]==1:
            nz = list(reaxants[:p]) + list(reaxants[p+1:])
            dJ[i] = kf*np.prod( np.power(cmols[nz], crx[nz]))
    for p, i in enumerate(products):
        if cmols[i]==0 and crx[i]==-1:
            nz = list(products[:p]) + list(products[p+1:])
            dJ[i] = -kr*np.prod( np.power(cmols[nz], -crx[nz]))

    return dJ
   
def dflux_dcmol_Jacobian( Crx, krx, cmols):
    '''Compute the Jacobian matrix of dJ/dm, where J is the length R vector
    of reactive flux in the R reactions, and m is the length M vector of 
    molecular concentrations of M molecules.

    Args:

    Crx: RxM matrix of reaction coefficients, where R is the number of
    reactions and M the number of molecules.  The Crx[r,i] gives the
    coefficient of molecule i in reaction r.  Positive coefficient
    indicates that molecule i is on the reactant side, negative
    coefficient indicates that it is on the product side.

    krx: Rx2 matrix of reaction rate constants. krx[r,0] and krx[r,1]
    are the forward and reverse rate constants of reaction r.

    cmols: a length M array of molecular concentrations.

    Return:

    dJ/dm: RxM Jacobian matrix dJ/dm, where (dJ/dm)_{ri} = dJ_{r}/dm_{i}.

    '''
    return np.array( [ dflux_dcmol( Crx[k,:], krx[k], cmols)
                       for k in range(Crx.shape[0]) ])

def rate_of_change( Crx, krx, cmols):
    '''Compute the instantaneous rate of change for the molecules
    according to the reactions and the current concentrations.

    Args:

    Crx: RxM matrix of reaction coefficients, where R is the number of
    reactions and M the number of molecules.  The Crx[r,i] gives the
    coefficient of molecule i in reaction r.  Positive coefficient
    indicates that molecule i is on the reactant side, negative
    coefficient indicates that it is on the product side.

    krx: Rx2 matrix of reaction rate constants. krx[r,0] and krx[r,1]
    are the forward and reverse rate constants of reaction r.

    cmols: a length M array of molecular concentrations.

    Return:

    dm/dt: a length M array of the instantaneous rate of change for
    each molecule.  
    '''
    # Compute the reactive flux for each reaction
    R = Crx.shape[0]
    Jrx = np.array( [ flux_in_reaction( Crx[r], krx[r], cmols)
                      for r in range(R) ])
    # The rate of change is given by 
    # dm/dt = - C^t\cdot J
    dmdt = -Crx.T.dot( Jrx)
    return dmdt

def drate_of_change_dcmol_Jacobian( Crx, krx, cmols):
    '''Compute the Jacobian d(dm/dt)/dm.

    Args:

    Crx: RxM matrix of reaction coefficients, where R is the number of
    reactions and M the number of molecules.  The Crx[r,i] gives the
    coefficient of molecule i in reaction r.  Positive coefficient
    indicates that molecule i is on the reactant side, negative
    coefficient indicates that it is on the product side.

    krx: Rx2 matrix of reaction rate constants. krx[r,0] and krx[r,1]
    are the forward and reverse rate constants of reaction r.

    cmols: a length M array of molecular concentrations.

    Return:

    d(dm/dt)/dm: MxM Jacobian matrix d(dm/dt)/dm, where
    (d(dm/dt)/dm)_{ij} = d(dm_i/dt)/dm_j.
    '''
    # dm/dt = - C^t\cdot J
    # Hence d(dm/dt)/dm = - C^t\cdot dJ/dm
    dJdm = dflux_dcmol_Jacobian( Crx, krx, cmols)
    ddmdt_dm = - Crx.transpose().dot( dJdm)
    return ddmdt_dm

def kinetics( Crx, krx, cmolst0, tmax, method='LSODA'):
    '''Solve the kinetic equation for the time evolution of m(t).

    Args:

    Crx: RxM matrix of reaction coefficients, where R is the number of
    reactions and M the number of molecules.  The Crx[r,i] gives the
    coefficient of molecule i in reaction r.  Positive coefficient
    indicates that molecule i is on the reactant side, negative
    coefficient indicates that it is on the product side.

    krx: Rx2 matrix of reaction rate constants. krx[r,0] and krx[r,1]
    are the forward and reverse rate constants of reaction r.

    cmolst0: a length M array of molecular concentrations at time t=0.

    tmax: number, the end time of the kinetic evolution.

    method: integration method: 'RK45', 'LSODA', etc. see
    scipy.integrate.solve_ivp.

    Return:

    moft: the solution of kinetic equation up to time tmax.

    '''
    moft = solve_ivp(
        lambda t, x: rate_of_change( Crx, krx, x),
        (0, tmax), cmolst0,
        method=method,
        jac=lambda t, x: drate_of_change_dcmol_Jacobian( Crx, krx, x))
    return moft

def null_from_svd( s, vh, R, M, rcond):
    '''Construct the nullspace of C from its SVD.  Code taken from
    https://github.com/scipy/scipy/blob/master/scipy/linalg/decomp_svd.py
    '''
    if rcond is None:
        rcond = np.finfo(vh.dtype).eps * max(R, M)
    tol = np.amax( s)*rcond
    num = np.sum( s>tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

def orth_from_svd( s, u, R, M, rcond):
    '''Construct an orthonormal basis for the range of C from its
    SVD. Code taken from
    https://github.com/scipy/scipy/blob/master/scipy/linalg/decomp_svd.py
    '''
    if rcond is None:
        rcond = np.finfo(u.dtype).eps * max(R, M)
    tol = np.amax( s)*rcond
    num = np.sum( s>tol, dtype=int)
    Q = u[:, :num]
    return Q

def ness( Crx, krx, mt0=None, U=None, S=None, Vh=None, Nh=None, Nhmt0=None,
          rcond=None, tinit=None, maxtrials=10, check_steady_state=False):
    '''Solve the nonequilibrium steady state equation of the molecular
    concentrations: dm/dt = 0.

    Args:

    Crx: RxM matrix of reaction coefficients, where R is the number of
    reactions and M the number of molecules.  The Crx[r,i] gives the
    coefficient of molecule i in reaction r.  Positive coefficient
    indicates that molecule i is on the reactant side, negative
    coefficient indicates that it is on the product side.

    krx: Rx2 matrix of reaction rate constants. krx[r,0] and krx[r,1]
    are the forward and reverse rate constants of reaction r.

    mt0: length M array of initial molecular concentrations at t=0.  If 
    mt0 is None, Nh and Nhmt0 must be provided.

    U: RxR matrix, C = U S Vh is the singular value decomposition of C.
    
    S: length rank(C) array of the singular values of matrix Crx.

    Vh: MxM matrix, C = U S Vh is the singular value decomposition of C.

    Nh: (M-rank(C))xM matrix, where N = Nh.T is the null space of C.

    Nhmt0: length (M-rank(C)) array of numbers, Nh\cdot m(t=0)
    
    rcond: float, relative condition number. Singular values s smaller
    than rcond * max(s) are considered zero. Default: floating point
    eps * max(R,M).  

    tinit: float, optional, solve kinetic equations up to this time
    and use the resulting concentrations as initial guess for solving
    the steady state equations.

    Return:

    mss: the solution of the ness equation dm/dt = 0.

    '''
    R, M = Crx.shape

    if U is None or S is None or Vh is None:
        U, S, Vh = svd( Crx)

    if Nh is None:
        N = null_from_svd( S, Vh, R, M, rcond)
        Nh = N.T
        
    if mt0 is not None and Nhmt0 is not None:
        raise ValueError('Either mt0 or Nhmt0 should be supplied as the initial condition, but not both!')

    if mt0 is None and Nhmt0 is None:
        raise ValueError('Either mt0 or Nhmt0 must be supplied as the initial condition!')

    if Nhmt0 is None:
        Nhmt0 = Nh.dot( mt0)

    # If we do not know the initial concentrations, but are given the
    # conserved quantities, we use the non-negative least square solution of
    # ||N.T.dot(m) - Nhmt0|| as a guess of initial condition for root
    # finding.
    if mt0 is None:
        mt0, _residues = nnls( Nh, Nhmt0)
    
    if tinit is not None:
        moft = kinetics( Crx, krx, mt0, tinit)
        mt0 = moft.y[:,-1]

    krx, alphas = normalize_rate_constants( Crx, krx)
    Nh = Nh.dot( np.diag( alphas))
    mt0 = mt0/alphas

    Us = orth_from_svd( S, U, R, M, rcond)
    Ush = Us.T
    
    # The independent equations at steady state are
    # Us.T.dot( J) = 0
    # and
    # Nh.dot( m) - Nh.dot( m(t=0)) = 0
    def eqss( m):
        J = np.array( [flux_in_reaction( Crx[r], krx[r], m) for r in range(R)])
        residuals = np.zeros( M)
        nj = Ush.shape[0]
        residuals[:nj] = Ush.dot( J)
        residuals[nj:] = Nh.dot( m) - Nhmt0
        return residuals

    # The Jacobian of the steady state equations
    # ( Us.T.dot( dJ/dm) )
    # (       Nh         )
    def Deqss( m):
        D = np.zeros( (M, M))
        dJdm = dflux_dcmol_Jacobian( Crx, krx, m)
        nj = Ush.shape[0]
        D[:nj,:] = Ush.dot( dJdm)
        D[nj:,:] = Nh
        return D

    sol = root( eqss, mt0, method='lm', jac=Deqss, options={'xtol': 1e-10})

    tol = 1e-10
    if (np.any( sol.x < -tol)):
        raise ValueError('Negative concentrations encountered in NESS solution: min(cmol) = cmol[%d] = %f < %g' % (np.argmin( sol.x), np.min( sol.x), tol))
 
    if (check_steady_state):
        dmdt = alphas*rate_of_change( Crx, krx, sol.x)
        print('max(|J|) = %g' % np.max( np.abs(dmdt)))

    sol.x *= alphas
    return sol
    
def dness_dkinetic_rates( Crx, krx, cmols, theta, 
                         U=None, S=None, Vh=None, Nh=None, rcond=None):
    '''Compute the dm_{ss}/dtheta, where theta are kinetic rate constants.

    Args:

    Crx: RxM matrix of reaction coefficients, where R is the number of
    reactions and M the number of molecules.  The Crx[r,i] gives the
    coefficient of molecule i in reaction r.  Positive coefficient
    indicates that molecule i is on the reactant side, negative
    coefficient indicates that it is on the product side.

    krx: Rx2 matrix of reaction rate constants. krx[r,0] and krx[r,1]
    are the forward and reverse rate constants of reaction r.

    cmols: length M array of steady state concentrations.

    theta: list of 2-tuples of indices.  Each tuple (r, d)--where r\in [0, R)
    and d = 0, 1--indicates derivative to be taken w.r.t. krx[r,d].

    U: RxR matrix, C = U S Vh is the singular value decomposition of C.
    
    S: length rank(C) array of the singular values of matrix Crx.

    Vh: MxM matrix, C = U S Vh is the singular value decomposition of C.

    Nh: (M-rank(C))xM matrix, where N = Nh.T is the null space of C.

    rcond: float, relative condition number. Singular values s smaller
    than rcond * max(s) are considered zero. Default: floating point
    eps * max(R,M).  
    
    Return:

    dmdtheta: M x len(theta) Jacobian matrix dm/dtheta: (dm/dtheta)[i,t]
    = dm[i]/dtheta[t]

    '''
    R, M = Crx.shape

    if U is None or S is None:
        U, S, Vh = svd( Crx)
    Us = orth_from_svd( S, U, R, M, rcond)
    Ush = Us.T

    if Nh is None:
        if Vh is None:
            _U, _S, Vh = svd( Crx)
        N = null_from_svd( S, Vh, R, M, rcond)
        Nh = N.T

    dJdm = dflux_dcmol_Jacobian( Crx, krx, cmols)
    
    dJdtheta = np.zeros( (R, len(theta)))
    for t, (r, d) in enumerate(theta):
        # if d==0, sgn = 1 for forward; if d==1, sgn = -1 for reverse
        sgn = 1 - 2*d 
        crx = sgn*Crx[r]
        # Select the reactants if sgn=1, select the products if sgn=-1
        mols = np.arange( M)[crx>0]
        dJdtheta[r, t] = sgn*np.prod( np.power( cmols[mols], crx[mols]))
    
    lhs = np.block( [ [ Nh ], 
                      [ Ush.dot( dJdm) ] ])
    rhs = np.block( [ [ np.zeros( (Nh.shape[0], len(theta)) ) ], 
                      [ -Ush.dot( dJdtheta) ] ])
    dmdtheta, _residues, _rank, _svs = lstsq( lhs, rhs)
    
    return dmdtheta

def kofy_and_dkdy( ks, ys):
    '''
    If the rate constants k are funtions of some parameters y, construct the
    function k(y) and its derivatives dk/dy.
    '''
    from sympy import symbols, lambdify, diff

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
        return np.array( kofyfunc( *tuple(y)))
        
    def dkdy( y):
        dkvals = [ (r, d, j, dk( *tuple(y))) 
                   for r, d, j, dk in nzdkdys ]
        return dkvals

    return kofy, dkdy
    
def dness_dy( Crx, kofy, dkdy, ny, cmols, U, S, Vh, Nh): 
    '''Jacobian of the molecular concentrations at steady state with
    respect to some parameters y.

    Args:

    Crx: RxM matrix of reaction coefficients, where R is the number of
    reactions and M the number of molecules.  The Crx[r,i] gives the
    coefficient of molecule i in reaction r.  Positive coefficient
    indicates that molecule i is on the reactant side, negative
    coefficient indicates that it is on the product side.

    kofy: Rx2 matrix of reaction rates given the parameters y.
 
    dkdy: a list of tuples, a sparse matrix
    representation of the Jacobian dk/dy given
    the parameters y: a list of tuples (r, d, j, dk[r,d]/dy[j]).

    ny: int, the number of parameters = len(y).

    cmols: length M array of steady state concentrations.

    U: RxR matrix, C = U S Vh is the singular value decomposition of C.
    
    S: length rank(C) array of the singular values of matrix Crx.

    Vh: MxM matrix, C = U S Vh is the singular value decomposition of C.

    Nh: (M-rank(C))xM matrix, where N = Nh.T is the null space of C.

    Nhmt0: length (M-rank(C)) array of numbers, Nh\cdot m(t=0)
    
    mt0: length M array of numbers of m(t=0)

    tinit: float, optional, solve kinetic equations up to this time
    and use the resulting concentrations as initial guess for solving
    the steady state equations.

    Return:

    dmdy: M x len(y) Jacobian dm/dy.  (dm/dy)[i,j] = dm[i]/dy[j]

    '''
    krx = kofy
    dks = dkdy
    # the same kinetic rate constant (r,d) may depend on several 
    # parameters.
    theta = list(set( [ (r, d) for r, d, _j, _dk in dks ]))
    rd2theta = dict( [ ((r, d), p) for p, (r, d) in enumerate( theta)])
    # dm/d\theta
    dmdtheta = dness_dkinetic_rates( Crx, krx, cmols, theta,
                                     U, S, Vh, Nh)
    dmdy = np.zeros( (len(cmols), ny))
    for r, d, j, dk in dks:
        p = rd2theta[(r,d)]
        dmdy[:, j] += dmdtheta[:,p]*dk
    return dmdy

def normalize_rate_constants( Crx, krx):
    '''Scale the reaction rate constants so that the rate constants
    involved in the change in each molecular species are approximately
    on the same order-of-magnitude.

    Args:

    Crx: RxM matrix of reaction coefficients, where R is the number of
    reactions and M the number of molecules.  The Crx[r,i] gives the
    coefficient of molecule i in reaction r.  Positive coefficient
    indicates that molecule i is on the reactant side, negative
    coefficient indicates that it is on the product side.

    krx: Rx2 matrix of reaction rate constants. krx[r,0] and krx[r,1]
    are the forward and reverse rate constants of reaction r.

    Returns:
    
    k: Rx2 matrix of scaled reaction rate constants.

    alpha: length M array of scaling factors for the concentration of
    each molecule.  The original concentrations and the scaled
    concentrations are related by m[i] (original) = alpha[i]*m'[i]
    (scaled)

    '''
    Cp, Cm = Crx.copy(), Crx.copy()
    Cp[Cp<0] = 0
    Cm[Cm>0] = 0
    Cpm = np.block( [ [ Cp ],
                      [ Cm ] ])
    epsilon = 1e-12
    lnf = np.log( krx[:,0] + epsilon)
    lnr = np.log( krx[:,1] + epsilon)
    lnfr = np.concatenate( [-lnf, lnr])
    lnalpha, _residues, _rank, _svs = lstsq( Cpm, lnfr)

    lnfp = lnf + Cp.dot( lnalpha)
    lnrp = lnr - Cm.dot( lnalpha)

    k = np.zeros( krx.shape)
    k[:,0] = np.exp( lnfp)
    k[:,1] = np.exp( lnrp)
    return k, np.exp( lnalpha)

def fit_kinetic_rates_to_ness( Crx, kofy, dkdy, y0,
                               Nexp, Nexpms, Nh, Nhmt0s):
    '''Fit kinetic rate parameters to the measured steady state concentrations.

    Args:

    Crx: RxM matrix of reaction coefficients, where R is the number of
    reactions and M the number of molecules.  The Crx[r,i] gives the
    coefficient of molecule i in reaction r.  Positive coefficient
    indicates that molecule i is on the reactant side, negative
    coefficient indicates that it is on the product side.

    kofy: callable function, kofy(y, t) returns a Rx2 matrix of
    reaction rates in the t'th experiment, given the parameters y.
 
    dkdy: callable function, dkdy(y, t) returns a sparse matrix
    representation of the Jacobian dk/dy in the t'th experiment, given
    the parameters y: a list of tuples (r, d, j, dk[r,d]/dy[j]).

    y0: array of numbers, the initial guess of the parameter values.

    Nexp: ExM matrix. E is the number of experimentally measured
    concentrations. Nexp.dot(m) should compute the measured computations
    from the molecular concentrations.

    Nexpms: ExD matrix. D is the number of experimental data
    points. Nexpms[j, d] gives the i'th measured concentration in
    d'th experiment.

    Nh: (M-rank(C))xM matrix, where N=Nh.T is the null space of C.

    Nhmt0s: (M-rank(C))xD matrix. Nhmt0s[:,d] = Nh.dot( m(t=0)[d]) for
    experimental d.

    Return:

    y: length len(y0) array. The fitted parameters.

    '''
    U, S, Vh = svd( Crx)
    E, D = Nexpms.shape[1]

    if (Nexp.ndim == 1):
        Nexp = Nexp.reshape( 1, Nexp.shape[0])

    def residuals( y):
        delta = np.zeros( (E, D))
        for t in range( D):
            krx = kofy( y, t)
            mss = ness( Crx, krx, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=Nhmt0s[t])
            delta[:,t] = Nexp.dot( mss.x) - Nexpms[:,t]
        return np.sum( np.square( delta))

    def residuals_Jacobian( y):
        delta = np.zeros( (E, D))
        ddeltady = np.zeros( (E, D, len(y)))
        for t in range( D):
            krx = kofy( y, t)
            mss = ness( Crx, krx, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=Nhmt0s[t])
            # \delta = m - m_{exp}
            delta[:,t] = mto.dot( mss.x) - Nexpms[:,t]
            dks = dkdy( y, t)
            # the same kinetic rate constant (r,d) can depend on several 
            # parameters.
            theta = list(set( [ (r, d) for r, d, _j, _dk in dks ]))
            rd2theta = dict( [ ((r, d), p) for p, (r, d) in enumerate( theta)])
            # dm/d\theta
            dmdtheta = dness_dkinetic_rates( Crx, krx, mss.x, theta,
                                             U, S, Vh, Nh)
            # d\delta/d\theta = d\delta/dm . dm/d\theta = N_{exp} . dm/d\theta
            ddeltadtheta = Nexp.dot( dmdtheta)
            # d\delta/dy = d\delta/d\theta . d\theta/dy
            for r, d, j, dk in dks:
                p = rd2theta[(r,d)]
                ddeltady[:, t, j] += ddeltadtheta[:,p]*dk
        jac = delta.dot( ddeltady)

def consumption_rate( Crx, ks, nCs, ms):
    '''Compute the consumption rate of a consumed molecular component
    (e.g. ATP).

    Args:

    Crx: the RxM matrix of reaction coefficients.

    ks: Rx2 matrix of reaction rate constants.

    nCs: length R array of numbers, nCs[r] is the number of molecules consumed
    in reaction r.

    ms: length M array of numbers, ms[i] gives the concentration of
    i'th molecular species.

    Returns:

    dDeltaCdt: number, the consumption rate d\DeltaC/dt

    '''
    R, M = Crx.shape
    rxs = np.arange( R)[nCs!=0]
    dCdt = 0.
    for r in rxs:
        J = flux_in_reaction( Crx[r], ks[r], ms)
        dCdt += nCs[r]*J
    return dCdt

def cumulative_consumption( Crx, ks, nCs, ts, mts):
    '''Compute the cumulative consumption of a consumed molecular
    component (e.g. ATP at a series of given time points.

    Args:

    Crx: the RxM matrix of reaction coefficients.

    ks: Rx2 matrix of reaction rate constants.

    nCs: length R array of numbers, nCs[r] is the number of molecules consumed
    in reaction r.

    ts: length T array of numbers, ts[t] gives the time at t'th time point.

    mts: T x M matrix of numbers, ms[i, t] gives the concentration of
    i'th molecular species at the t'th time point.

    Returns:

    DeltaCt: a length T array of numbers, DeltaCt[t] gives the
    consumption up to t'th time point.

    '''
    dCdts = np.array( [ consumption_rate( Crx, ks, nCs, mts[:,t])
                        for t in range(len(ts))])
    DeltaCt = cumtrapz( dCdts, ts, initial=0.)
    return DeltaCt
 
def Michaelis_Menten( s, v):
    '''Fit the substrate concentration and reaction rate to the
    Michaelis-Menten equation, return KM and Vmax.
    '''
    oneovers = 1./s
    oneoverv = 1./v
    slope, intercept, rval, pval, stderr = linregress( oneovers, oneoverv)
    Vmax = 1./intercept
    Km = slope*Vmax
    return Km, Vmax

def test_dflux_dcmol( tol=1e-5):
    crx = np.array( [ 2, -1, 1, -2, 0 ])
    krx = np.array( [ 2.5, 1.3 ])
    cmols = [ np.array( [ 1., 2., 0., 1.2, 2. ]),
              np.array( [ 1., 2., 1.1, 1.2, 1. ]),
              np.array( [ 1., 0., 1.1, 1.2, 0. ]) ]

    diffs = [ check_grad( lambda x: flux_in_reaction( crx, krx, x),
                          lambda x: dflux_dcmol( crx, krx, x),
                          c) for c in cmols ]
    print('Test dJ/dm: |dJ/dm - DJ/Dm| = %s' \
        % ('(' + ','.join( map(str, diffs)) + ')'))
    if np.max( diffs) < tol:
        print('SUCCESS: dJ/dm test passed!')
        return True
    else:
        print('FAIL: dJ/dm has an error of %g' % np.max( diffs))
        return False

def test_rate_of_change_Jacobian( tol=1e-5):
    Crx = np.array( [ [2, -1, 0, 0],
                      [0, 1, -1, -1] ])
    krx = np.array( [ [2.5, 1.3 ],
                      [1.8, 1.2 ] ])
    cmols = np.array( [1., 2., 0.1, 2.2])
    
    diffs = [
        check_grad( lambda x: rate_of_change( Crx, krx, x)[i],
                    lambda x: drate_of_change_dcmol_Jacobian( Crx, krx, x)[i],
                    cmols)
        for i in range(len(cmols)) ]
    print('Test d(dm/dt)/dm: | d(dm/dt)/dm - D(dm/dt)/Dm | = %g' % np.max(diffs))
    if np.max(diffs) < tol:
        print('SUCCESS: d(dm/dt)/dm test passed!')
        return True
    else:
        print('FAIL: d(dm/dt)/dm has an error of %g' % np.max(diffs))
        return False

def test_kinetics( dt=1e-4, tol=1e-5):
    Crx = np.array( [ [ 2, -1, 0, 0 ],
                      [ 0, 1, -1, -1 ] ])
    krx = np.array( [ [2.5, 1.3],
                      [1.8, 1.2] ])
    cmolst0 = np.array( [1., 0., 0., 0.1])
    tmax = 1.

    moft = kinetics( Crx, krx, cmolst0, tmax)
    moftp = kinetics( Crx, krx, moft.y[:,-1], dt)
    DmDt = (moftp.y[:,-1] - moftp.y[:,0])/dt
    dmdt = rate_of_change( Crx, krx, 0.5*(moftp.y[:,-1] + moftp.y[:,0]))
    print('Testing solution of dm/dt...')
    print('%25s' % 'dm/dt =', dmdt)
    print('%25s' % '(m(t+dt) - m(t))/dt =', DmDt)
    diff = np.max( np.abs(DmDt - dmdt))
    diffrep = 'max(|dm(t)/dt - (m(t+dt)-m(t))/dt|) = %.3g' % diff
    if diff < tol:
        print('SUCCESS: m(t) agrees with dm/dt: %s < %g' % (diffrep, tol))
        return True
    else:
        print('FAIL: %s > %g' % (diffrep, tol))
        return False

def test_ness( tol=1e-9):
    if (False):
        # A simple model
        # H.ADP + ATP = H.ATP + ADP
        # H.ATP = H.ADP + Pi
        # mols = [ H.ADP, H.ATP ]
        Crx = np.array( [ [ 1, -1 ],
                          [ -1, 1 ] ])
        krx = np.array( [ [ 1.5, 2. ],
                          [ 1.3, 0.8] ])
        mt0 = np.array( [ 1., 0.1 ])
        Nh = np.array( [ [ 1., 1. ] ])
    else:
        # A more complex model
        # H.ADP + ATP = H.ATP + ADP
        # 2 H.ATP = (H.ATP)2
        # (H.ATP)2 = (H.ADP)2 + 2 Pi
        # (H.ADP)2 = 2 H.ADP
        # mols = [ H.ADP, H.ATP, (H.ADP)2, (H.ATP)2 ]
        Crx = np.array( [ [1, -1, 0, 0],
                          [0, 2, 0, -1],
                          [0, 0, -1, 1],
                          [-2, 0, 1, 0] ])
        krx = np.array( [ [ 1.5, 2. ],
                          [ 1.3, 0.8 ],
                          [ 0.5, 1e-3],
                          [ 0.8, 1e-4 ] ])
        mt0 = [ 1., 0, 0, 0 ]
        Nh = np.array( [ [ 1, 1, 2, 2 ] ])

    Nhmt0 = Nh.dot( mt0)

    print('Testing solution to steady state...')
    success = True
    # Test two different initial conditions
    mss = ness( Crx, krx, mt0)
    mss1 = ness( Crx, krx, Nh=Nh, Nhmt0=Nhmt0)
    diff = np.max( np.abs( mss.x - mss1.x))
    report = 'Different representation of initial conditions max(|delta m|) = %.3g' % diff
    if (diff < tol):
        print('SUCCESS: %s < %g' % (report, tol))
    else:
        print('FAIL: %s > %g' % (report, tol))
        success = False
    
    # Also run the kinetics until we reach steady state.
    tmax = 1.
    mt = mt0[:]
    while True:
        moft = kinetics( Crx, krx, mt, tmax)
        if np.max( np.abs( moft.y[:,-1] - mt)) > tol:
            tmax *= 2
            mt = moft.y[:,-1]
        else:
            break

    # Check that dm/dt = 0 at steady state.
    dmdt = rate_of_change( Crx, krx, mss.x)
    
    diff = np.max( np.abs( dmdt))
    report = 'At steady state, max(|dm/dt|) = %.3g' % diff
    if (diff < tol):
        print('SUCCESS: %s < %g' % (report, tol))
    else:
        print('FAIL: %s > %g' % (report, tol))
        success = False

    diff = np.max( np.abs(moft.y[:,-1] - mss.x))
    report = 'Solution to dm/dt = 0 differs from m(t=\inf) by %.3g' % diff
    if (diff < tol):
        print('SUCCESS: %s < %g' % (report, tol))
    else:
        print('FAIL: %s > %g' % (report, tol))
        success = False

    return success
    
def test_dness_dkinetic_rate( tol=1e-7):
    # see test_ness for a description of the models
    if (False):
        Crx = np.array( [ [ 1, -1 ],
                          [ -1, 1 ] ])
        krx = np.array( [ [ 1.5, 2. ],
                          [ 1.3, 0.8] ])
        mt0 = np.array( [ 1., 0.1 ])
        
        mss = ness( Crx, krx, mt0)
    else:
        Crx = np.array( [ [1, -1, 0, 0],
                          [0, 2, 0, -1],
                          [0, 0, -1, 1],
                          [-2, 0, 1, 0] ])
        krx = np.array( [ [ 1.5, 2. ],
                          [ 1.3, 0.8 ],
                          [ 0.5, 1e-3],
                          [ 0.8, 1e-4 ] ])
        mt0 = [ 1., 0, 0, 0 ]
        Nh = np.array( [ [ 1, 1, 2, 2 ] ])

    def mofk( x):
        krx = np.array( [ [ x[2*r], x[2*r+1] ] for r in range(len(x)//2) ])
        mss = ness( Crx, krx, mt0)
        return mss.x

    def dmofk( x):
        krx = np.array( [ [ x[2*r], x[2*r+1] ] for r in range(len(x)//2) ])
        theta = reduce( lambda x, y: x + y,
                        [ [ (r, 0), (r, 1) ] for r in range(len(x)//2) ])
        m = ness( Crx, krx, mt0).x
        dmdk = dness_dkinetic_rates( Crx, krx, m, theta)
        return dmdk

    print('Testing dm/dtheta...')

    diffs = [ check_grad( lambda x: mofk( x)[i], lambda x: dmofk( x)[i],
                          krx.flatten()) for i in range(len(mt0)) ]
    diff = np.max( diffs)
    report = '| dm/dtheta - Dm/Dtheta | = %g' % diff
    if diff < tol:
        print('SUCCESS: %s < %g' % (report, tol))
        return True
    else:
        print('FAIL: %s > %g' % (report, tol))
        return False

def test_fit_ness_to_data( tol=1e-5):
    # Consider the set of reactions:
    # H.ADP + ATP = H.ATP + ADP
    # 2 H.ATP = (H.ATP)2
    # (H.ATP)2 = (H.ADP)2 + 2 Pi
    # (H.ADP)2 = 2 H.ADP
    # mols = [ H.ADP, H.ATP, (H.ADP)2, (H.ATP)2 ]
    Crx = np.array( [ [1, -1, 0, 0],
                      [0, 2, 0, -1],
                      [0, 0, -1, 1],
                      [-2, 0, 1, 0] ])
    krx0 = np.array( [ [ 1.5, 2. ],
                       [ 1.3, 0.8 ],
                       [ 0.5, 1e-3],
                       [ 0.8, 1e-4 ] ])
    Nh = np.array( [ [ 1, 1, 2, 2 ] ])
    # Let's assume that we have measured the activity of ATP-bound H, such that
    # data = gamma*([H.ATP] + 2[(H.ATP)2])
    #
    # The experiments are performed at a series [ATP], [ADP], [Pi], and
    # total H = [H.ATP] + [H.ADP] + 2[(H.ATP)2] + 2[(H.ADP)2].
    npts = 10
    mt0s = np.array([ [ h, 0, 0, 0 ] for h in np.linspace( 1, 10, npts) ]).T
    nucleotides = 2*( np.random.rand( 2, npts) + 0.1)
    pi = 0.1
    pi2 = pi*pi

    # We will fit the ATP/ADP-exchange rates and the ATP hydrolysis rate.
    # y = [ k[0,0], k[0,1], k[2,0] ]
    theta = [ (0,0), (0,1), (2,0) ]
    def kofy( y, t):
        krx = krx0.copy()
        krx[0,0] = y[0]*nucleotides[0, t]
        krx[0,1] = y[1]*nucleotides[1, t]
        krx[2,0] = y[2]
        krx[2,1] *= pi2
        return krx

    def dkdy( y, t):
        dks = [
            (0, 0, 0, nucleotides[0, t]),
            (0, 1, 1, nucleotides[1, t]),
            (2, 0, 2, 1.) ]
        return dks

    U, S, Vh = svd( Crx)
    # Generate reference data
    Nhmt0s = Nh.dot( mt0s)
    gamma0 = 1.2
    y0 = np.array([ krx0[r,d] for r, d in theta ] + [gamma0])

    initmt0s = [ None for t in range(npts) ]

    mtod = np.array( [ [ 0, 1, 0, 2 ] ])
    data = [ 
        ness(Crx, kofy(y0[:3], t), U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=Nhmt0s[:,t]).x
        for t in range( npts) ]
    data = np.array( data).T
    data = gamma0*mtod.dot( data).flatten()

    def residual( y, t):
        # y[0:3] are the kinetic rate parameters, y[3] is gamma.
        krx = kofy( y[:3], t)
        if (initmt0s[t] is None):
            mss = ness( Crx, krx, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=Nhmt0s[:,t])
            initmt0s[t] = mss.x
        else:
            mss = ness( Crx, krx, mt0=initmt0s[t], U=U, S=S, Vh=Vh, Nh=Nh)
        delta = y[-1]*mtod.dot( mss.x).flatten()[0] - data[t]
        return delta

    def residual_jac( y, t):
        jac = np.zeros( len(y))
        krx = kofy( y[:3], t)
        mss = ness( Crx, krx, U=U, S=S, Vh=Vh, Nh=Nh, Nhmt0=Nhmt0s[:,t])
        md = mtod.dot( mss.x).flatten()
        dmdy3 = dness_dy( Crx,
                          kofy( y, t), dkdy( y, t), 3, mss.x,
                          U=U, S=S, Vh=Vh, Nh=Nh)
        jac[:3] = y[-1]*mtod.dot( dmdy3).flatten()
        jac[-1] = md
        return jac

    print('Testing fitting kinetic parameters to steady state data...')

    success = True
    diffs = np.array( [ check_grad( lambda y: residual(y, t), 
                                    lambda y: residual_jac(y, t), y0)
                        for t in range( npts) ])
    diff = np.max( diffs)
    report = 'Jacobian of residual from data: |d - D| = %g' % diff
    if diff < tol:
        print('SUCCESS: %s < %g' % (report, tol))
    else:
        print('FAIL: %s > %g' % (report, tol))
        success = False
    
    result = least_squares(lambda y: [residual(y, t) for t in range( npts)],
                           y0*(1 + 0.5*(np.random.rand( len(y0))-0.5)),
                           lambda y: [residual_jac(y, t) for t in range(npts)],
                           method='lm')

    success &= result.success
    diffs = result.x - y0
    diff = np.max( np.abs( diffs))
    report = 'Fit parameters differ from reference: %g' % diff
    if diff < tol:
        print('SUCCESS: %s < %g' % (report, tol))
    else:
        print('FAIL: %s > %g' % (report, tol))
        success = False
                           
    return success

def test_consumption( tol=2e-3):
    # Consider the following simple reactions:
    #
    #       f1
    # M.ATP = M.ADP + Pi  
    #       r1
    #             f2
    # M.ADP + ATP = M.ATP + ADP
    #             r2
    #
    # The rate-of-change equations are
    #
    # d[M.ATP]/dt = r1[Pi][M.ADP] - f1[M.ATP] + f2[ATP][M.ADP] - r2[ADP][M.ATP]
    #
    # [M.ATP] + [M.ADP] = M0
    #
    # We have
    # 
    # d[M.ATP]/dt = (r1[Pi] + f2[ATP])M0 
    #             - (f1 + r1[Pi] + f2[ATP] + r2[ADP])[M.ATP]
    # The solution is
    # [M.ATP] = ([M.ATP](t=0) - [M.ATP](t=\infty))exp(-kt) + [M.ATP](t=\infty)
    # where
    # k = f1 + r1[Pi] + f2[ATP] + r2[ADP]
    # and
    # [M.ATP](t=\infty) = M0(r1[Pi]+f2[ATP])/k
    #
    # The ATP consumption rate is
    # dATP/dt = f1[M.ATP] - r1[Pi][M.ADP]
    #         = (f1 + r1[Pi])[M.ATP] - r1[Pi]M0
    #         = (f1 + r1[Pi])[M.ATP](t=\infty) 
    #         + (f1 + r1[Pi])([M.ATP](t=0)-[M.ATP](t=\infty))exp(-k t) 
    #         - r1[Pi]M0
    # And the cumulative consumption is
    # ATP = \int dt dATP/dt
    #     = (f1 + r1[Pi])[M.ATP](t=\infty)*t
    #     + (f1 + r1[Pi])([M.ATP](t=0) - [M.ATP](t=\infty))(1 - exp(-kt))/k 
    #     - r1[Pi]M0 t
    M0 = 1.

    Crx = np.array( [ [ 1, -1 ],
                      [-1,  1 ] ])
    ks = np.array( [ [ 2.5, 1.2 ],
                     [ 3.,  4.0 ] ])
    nATPs = np.array( [ 1, 0 ])
    mt0 = np.array( [ M0, 0. ])
    
    ktot = np.sum( ks)
    tmax = -np.log(1e-2)/ktot
    moft = kinetics( Crx, ks, mt0, tmax)
    t = moft.t
    mtinfty = M0*(ks[0,1] + ks[1,0])/ktot

    mt = (mt0[0] - mtinfty)*np.exp(-ktot*t) + mtinfty

    ATPs0 = (ks[0,0] + ks[0,1])*mtinfty*t + \
            (ks[0,0] + ks[0,1])*(mt0[0] - mtinfty)*(1 - np.exp(-ktot*t))/ktot \
            - ks[0,1]*M0*t

    ATPs = cumulative_consumption( Crx, ks, nATPs, t, moft.y)

    dATP = ATPs - ATPs0
    delta = np.max( np.abs(dATP))

    report = 'max(|ATP_exact(t) - ATP_solve(t)|) = %g' % delta
    success = True
    if (delta <= tol):
        print('SUCCESS: %s <= %g' % (report, tol))
    else:
        print('FAIL: %s > %g' % (report, tol))
        success = False

    return success

def unit_test():
    sep = '='*60
    success = True
    print(sep)
    success &= test_dflux_dcmol()
    print(sep)
    success &= test_rate_of_change_Jacobian()
    print(sep)
    success &= test_kinetics()
    print(sep)
    success &= test_ness()
    print(sep)
    success &= test_dness_dkinetic_rate()
    print(sep)
    success &= test_fit_ness_to_data()
    print(sep)
    success &= test_consumption()

    if success:
        print('All unit tests passed!')
    else:
        print('Some unit tests failed!')

if __name__ == '__main__':
    unit_test()
    
