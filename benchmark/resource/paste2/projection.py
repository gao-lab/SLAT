import numpy as np

"""
Functions for projecting slices onto the same coordinate system from partial alignment matrices obtained by PASTE2/
"""


def partial_stack_slices_pairwise(slices, pis):
    """
    Projecting all slices onto the same 2D coordinate system.
    
    In other words, project: 
    
        slices[0] --> slices[1] --> slices[2] --> ...
    
    param: slices - list of slices (AnnData Object)
    param: pis - list of pi (partial_pairwise_align output) between consecutive slices
    
    Return: new_slices - list of slices (AnnData Object) with new spatial coordinates.
    """
    
    assert len(slices) == len(pis) + 1, "'slices' should have length one more than 'pis'. Please double check."
    assert len(slices) > 1, "You should have at least 2 layers."

    new_coor = []
    S1, S2 = partial_procrustes_analysis(slices[0].obsm['spatial'], slices[1].obsm['spatial'], pis[0])
    new_coor.append(S1)
    new_coor.append(S2)
    for i in range(1, len(slices) - 1):
        x, y = partial_procrustes_analysis(new_coor[i], slices[i + 1].obsm['spatial'], pis[i])
        shift = new_coor[i][0,:] - x[0,:]
        y = y + shift
        new_coor.append(y)
    new_slices = []
    for i in range(len(slices)):
        s = slices[i].copy()
        s.obsm['spatial'] = new_coor[i]
        new_slices.append(s)
    return new_slices


def partial_procrustes_analysis(X, Y, pi):
    """
    Finds and applies optimal rotation between spatial coordinates of two slices given a partial alignment matrix.
    
    param: X - np array of spatial coordinates (e.g.: sliceA.obs['spatial'])
    param: Y - np array of spatial coordinates (e.g.: sliceB.obs['spatial'])
    param: pi - alignment matrix between the two slices output by PASTE2

    Return: projected spatial coordinates of X, Y
    """
    m = np.sum(pi)
    Z = (X - pi.sum(axis=1).dot(X) * (1.0 / m)).T
    W = (Y - pi.sum(axis=0).dot(Y) * (1.0 / m)).T
    H = W.dot(pi.T.dot(Z.T))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    W = R.dot(W)
    return Z.T, W.T
