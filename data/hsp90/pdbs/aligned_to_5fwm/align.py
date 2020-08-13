from atomsel import atomsel

def overlap_residues( asegs, asel, mola, bsegs, bsel, molb):
    '''
    Find mapped residues between the segments of A and the segments of B.
    '''
    delta = [ pa[0] - pb[0] for pa, pb in zip(asegs, bsegs) ]
 
    a = [ atomsel( '(%s) and resid %d to %d' % (asel, start, stop), mola)
          for start, stop in asegs ]
    b = [ atomsel( '(%s) and resid %d to %d' % (bsel, start, stop), molb)
          for start, stop in bsegs ]
    aids = [ s.get( 'resid') for s in a ]
    bids = [ s.get( 'resid') for s in b ]
    
    bidsp = [ [ i + d for i in s ] for s, d in zip(bids, delta) ]

    common = [ list(set(aid).intersection( set(bid))) for aid, bid in
               zip( aids, bidsp) ]
    commonb = [ [ i - d for i in s ] for s, d in zip(common, delta) ]

    aids = reduce( lambda x, y: x + y, common)
    bids = reduce( lambda x, y: x + y, commonb)

    return aids, bids

