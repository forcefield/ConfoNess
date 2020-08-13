import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from rxnet import flux_network_as_graph
import networkx as nx

def latex_float( f, fmt="{0:.2g}"):
    '''
    Formatting float number in LaTex.
    '''
    fstr = fmt.format( f)
    if "e" in fstr:
        base, exponent = fstr.split('e')
        return r'{0} \times 10^{{{1}}}'.format( base, int(exponent))
    else:
        return fstr

def reaction_arrows( kf, kr, rr, rp, s, lnk0, d, chirality=1):
    '''
    Draw a reaction from the reactant to the product, with the arrow
    length be s*ln(kf/k0) for the forward reaction and s*ln(kr/k0) for the
    backward reaction, and with the reactant positioned at rr and the
    product at rp.  The forward and backward arrows will be separated by 2d.
    Return [ farrow, rarrow ], parameters for the forward and reaction arrows, 
    which can be used in axis.arrow( a[0][0], a[0][1], a[1][0], a[1][1])
    to plot the arrows (where a=farrow, rarrow).
    '''
    dr = rp - rr
    if (dr[0] != 0):
        alpha = dr[1]/dr[0]  # slope
        sgn = np.sign(dr[0]) # direction
        cos_t = sgn/np.sqrt(1 + alpha*alpha)
        sin_t = alpha*cos_t
    else:
        sgn = np.sign(dr[1])
        cos_t = 0
        sin_t = sgn
    rm = 0.5*(rp + rr)
    dc = chirality*d*np.array([-sin_t, cos_t])
    # The centers of the reaction arrows
    rcf = rm + dc  # forward
    rcb = rm - dc  # backward
    # The lengths of the arrows
    lf = s*(np.log(kf) - lnk0) if kf>0 else 0
    lb = s*(np.log(kr) - lnk0) if kr>0 else 0
    dlf = lf*np.array( [cos_t, sin_t])
    dlb = lb*np.array( [cos_t, sin_t])
    # The end points of the forward reaction arrow 
    farrow = [ rcf - 0.5*dlf, dlf ] if lf>0 else None
    # The end points of the backward reaction arrow
    rarrow = [ rcb + 0.5*dlb, -dlb ] if lb>0 else None
    return [farrow, rarrow]

def plot_reaction_network( mols, mpos, kij, chirality=dict(),
                           highlights = [],
                           eta=5, kmin=None, kmax=None,
                           node_symbol='o', label_nodes=True, margin=0.2, 
                           sepfactor=0.06,
                           ax=None):
    '''Place the molecule i at mpos[i], and draw reaction arrows between i
    and j whose lengths are proportional to the reaction rates
    ln(kij).
    
    Args:

    mols: a list of string, mols[i] gives the name (label) of molecule i.

    mpos: a Nx2 numpy array, mpos[i] gives the 2D coordinates of molecule i.

    kij: a list of tuples (i, j, kf, kr).

    eta: float, specifies the ratio of the longest reaction arrow to
    the shortest.

    sepfactor: float, specifies the separation between the forward and reverse
    arrows.

    highlights: list of pairs (i,j), highlight the arrow from i to j
    in a different color.

    Returns:

    fig: a matplotlib figure

    '''
    max_text_size=16
    
    xmin, xmax = np.min(mpos[:,0]), np.max(mpos[:,0])
    ymin, ymax = np.min(mpos[:,1]), np.max(mpos[:,1])
    ratio = (ymax-ymin)/(xmax-xmin)
    width = 5*(xmax - xmin)
    height = width*ratio

    # Determine the appropriate k0 and scaling factor for arrow lengths.
    kfs = map(lambda k: k[2], kij)
    krs = map(lambda k: k[3], kij)
    kfs = filter(lambda k: k>0, kfs)
    krs = filter(lambda k: k>0, krs)
    kkmax = max(max(kfs), max(krs))
    kkmin = min(min(kfs), min(krs))

    # Round to the next order-of-magnitudes
    #kkmax = np.power(10., np.ceil(np.log10(kmax))) 
    #kkmin = np.power(10., np.floor(np.log10(kmin)))

    if (kmin is None): kmin = kkmin
    if (kmax is None): kmax = kkmax
    
    # We want the arrow length for kmax to be amax and for kmin to be amin.
    # i.e.  s ln(kmax/k0) = amax and s ln(kmin/k0) = amin
    # hence  s = (amax - amin)/(ln(kmax/kmin)) and
    #        lnk0 = (amax*ln(kmin) - amin*ln(kmax))/(amax - amin)

    # We want to find the largest amax such that
    # aij = s ln(kij/k0) <= dij = (1-margin)*rij for all i, j
    # where rij is the distance between nodes i and j.

    # We fix amin = amax/eta, where eta > 1.
    #
    # lnk0 = (eta*ln(kmin) - ln(kmax))/(eta - 1)
    #
    # aij = s (ln(kij) - ln(k0))
    #     = (eta - 1)/(ln(kmax/kmin))*amin*(ln(kij) - lnk0)
    #
    # Let c = (eta - 1)/ln(kmax/kmin) > 0
    #  
    # aij <= dij <=>
    # amin < dij/(c*(ln(kij) - ln(k0)))
    
    dlnk = np.log(kmax/kmin)
    _c = (eta - 1)/dlnk
    lnk0 = (eta*np.log(kmin) - np.log(kmax))/(eta - 1)
    def max_amin_ij( kij, dij):
        lnk = np.log(kij)
        return dij/(_c*(lnk - lnk0)) 

    ds = []
    amin = 1E10
    for i, j, kf, kr in kij:
        rij = mpos[j] - mpos[i]
        dij = np.sqrt(rij[0]*rij[0] + rij[1]*rij[1])
        ds.append( dij)
        dij *= (1 - margin)
        _k = max(min(max(kf, kr), kmax), kmin)
        if (_k > 0):
            amin = min(amin, max_amin_ij( _k, dij))

    s = (eta - 1)/dlnk*amin
    
    # The forward and backward arrows should be separated by 2*d, which
    # should be no more than 0.06*min({d_ij}).
    d = sepfactor*np.min(ds)

    arrow_params = {'length_includes_head': True, 'shape': 'right',
                    'head_starts_at_zero': True,
                    'color': 'black'}
    
    def draw_reaction_arrow( ax, a, chirality=1, highlight=False):
        if chirality==-1: arrow_params['shape'] = 'left'
        else: arrow_params['shape'] = 'right'
        if (highlight):
            arrow_params['color'] = 'red'
        else:
            arrow_params['color'] = 'black'
        ax.arrow( a[0][0], a[0][1], a[1][0], a[1][1],
                  width=0.0075, head_width=0.05, head_length=0.075,
                  **arrow_params)
        
    if ax is None:
        fig, ax = plt.subplots( figsize=(width, height))
    else:
        fig = ax.figure

    if node_symbol is not None:
        ax.plot( mpos[:,0], mpos[:,1], node_symbol, color='k', mfc='none')
    if (label_nodes):
        text_params = {'ha': 'center', 'va': 'bottom', 'family': 'sans-serif',
                       'fontweight': 'normal', 'fontsize': 24,
                       'color': '#377eb8'}
        if (isinstance(label_nodes, list)):
            labeled = filter( lambda (i, m): m in label_nodes, enumerate(mols))
        else:
            labeled = enumerate(mols)
        for i, m in labeled:
            ax.text( mpos[i,0], mpos[i,1]+d, m, zorder=-10,
                     **text_params)

    for i, j, kf, kr in kij:
        c = chirality.get( (i, j), 1)
        if (kf>0): kf = max(min(kf, kmax), kmin)
        if (kr>0): kr = max(min(kr, kmax), kmin)
        ax.plot( mpos[[i,j], 0], mpos[[i,j],1], '--', lw=0.5)
        af, ar = reaction_arrows( kf, kr, mpos[i], mpos[j], s, lnk0, d, c)
        if af is not None:
            draw_reaction_arrow( ax, af, c, (i, j) in highlights)
        if ar is not None:
            draw_reaction_arrow( ax, ar, c, (j, i) in highlights)

    # Put length legend on the plot
    amax = amin*eta
    xright = 1.02*xmax - 0.02*xmin
    ymed = 0.8*ymin+0.2*ymax
    dy = 0.15*(ymax - ymin)
    legpos = np.array( [ [ xright, ymed ],
                         [ xright + amax, ymed ] ])
    text_params = {'ha': 'left', 'va': 'center', 'family': 'sans-serif',
                   'fontweight': 'normal', 'fontsize': 18, 'color': 'black'}
    kmed = np.sqrt(kmin*kmax)
    # print kmin, kmed, kmax
    if (kkmin < kmin):
        skmin = ' $\leq %s$' % latex_float(kmin)
    else:
        skmin = ' $%s$' % latex_float(kmin)
    if (kkmax > kmax):
        skmax = ' $\geq %s$' % latex_float(kmax)
    else:
        skmax = ' $%s$' % latex_float(kmax)
    skmed = ' $%s$' % latex_float( kmed)
    legk = [ skmin, skmed, skmax ]    
    for k, leg in zip([kmin, kmed, kmax], legk):
        af, _ = reaction_arrows( k, 0, legpos[0], legpos[1], s, lnk0, 0)
        draw_reaction_arrow( ax, af)
        ax.text( legpos[1,0]+2*d, legpos[1,1], leg, **text_params)
        legpos += np.array( [0, dy])
        
    xmin0, xmax0 = ax.get_xlim()
    xmin = min(xmin - 2.5*d, xmin0)
    xmax = max(xright+amax+2.5*d, xmax0)
    ax.set_xlim( xmin, xmax)
    ymin0, ymax0 = ax.get_ylim()
    ymin = min(ymin0, ymin-3*d)
    ymax = max(ymax0, ymax+3*d)
    ax.set_ylim( ymin, ymax)
    ax.set_xticks( [])
    ax.set_yticks( [])
    plt.axis('off')
    
    return fig

def scale_weight( w, smin, smax, w0 = None):
    '''
    Scale the weight so that the values lie between smin and smax, and
    all values below w0 lie at smin.
    '''
    wmin, wmax = np.min(w), np.max(w)
    if w0 is None:
        wp = w
    else:
        wmin = max(wmin, w0)
        wp = np.maximum( w, w0)
        
    return ((smin*wmax - smax*wmin) + wp*(smax - smin))/(wmax - wmin)

def plot_reaction_flux( mols, mpos, n, J, ax, label_nodes=[],
                        highlights = [],
                        node_size_min=1, node_size_max=600,
                        edge_width_min=0.1, edge_width_max=10,
                        nmin=None, Jmin=None, epsilon=1e-15,
                        arrow_width=1):
    '''
    Plot the flux in a reaction network.  Return a directed graph
    representing the flux network and the axis object for the plot.

    Args:

    mols: a list of string, mols[i] gives the name (label) of molecule i.

    mpos: a dict mapping string to a tuple of 2 numbers, mpos[mol] gives the 
    2D coordinates of molecule mol.

    n: array of numbers, n[i] gives the molar fraction of molecule i.

    J: NxN antisymmetric matrix, J[i,j] gives the flux from molecule i to j.

    Returns:

    g: a directed graph representing the reactive flux.

    ax: a matplotlib axis plotting the reactive flux. 
    '''
    colors = ['#e41a1c','#377eb8','#4daf4a']
    g = flux_network_as_graph( mols, J, n, epsilon)
    nodesize = np.log( [ g.node[a]['weight'] for a in g.nodes() ])
    edgewidth = np.log( [ g[a][b]['weight'] for a, b in g.edges() ])
    smin = None if nmin is None else np.log(nmin)
    nodesize = scale_weight( nodesize, node_size_min, node_size_max, smin)
    wmin = None if Jmin is None else np.log(Jmin)
    edgewidth = scale_weight( edgewidth, edge_width_min, edge_width_max, wmin)
    nodes = nx.draw_networkx_nodes( g, pos=mpos, ax=ax, node_color='white', node_size=nodesize)
    if (label_nodes is None):
        labels = None
    else:
        labels=dict([ (m, m) for m in label_nodes ]) 
    nodes.set_edgecolor( 'k')
    nx.draw_networkx_labels( g, pos=mpos, labels=labels, ax=ax, font_color=colors[1], font_weight='bold')
    nx.draw_networkx_edges( g, pos=mpos, ax=ax, arrows=False, width=edgewidth)
    if (highlights):
        nohilite = filter( lambda e: not e in highlights and not (e[1], e[0]) in highlights, g.edges())
        nx.draw_networkx_edges( g, pos=mpos, edgelist=nohilite, ax=ax,
                                arrows=True,
                                edge_color=colors[2], width=arrow_width)
        nx.draw_networkx_edges( g, pos=mpos, edgelist=highlights, ax=ax,
                                arrows=True,
                                edge_color=colors[0], width=arrow_width)
    else:
        nx.draw_networkx_edges( g, pos=mpos, ax=ax, arrows=True, edge_color=colors[2], width=arrow_width)
    return g, ax
