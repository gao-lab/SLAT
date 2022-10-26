from typing import List
from scipy.spatial import Delaunay
import numpy as np
from itertools import chain

def alpha_shape(points, alpha, only_outer=True)->List:
    """
    Compute the alpha shape (concave hull) of a set of points.
    
    Parameters
    ----------
    points
        np.array of shape (n,2) points.
    alpha
        alpha value.
    only_outer
    boolean value to specify if we keep only the outer border or also inner edges.
    
    Return
    ----------
    Set of (i,j) pairs representing edges of the alpha-shape. (i,j) are the indices in the points array.
    
    Refer
    ----------
    https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    circum_r_list = []
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        circum_r_list.append(circum_r)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    boundary = list(set(list(chain.from_iterable(list(edges)))))
    return boundary, edges, circum_r_list


def _find_edges_with(i, edge_set):
    i_first = [j for (x,j) in edge_set if x==i]
    i_second = [j for (j,x) in edge_set if x==i]
    return i_first,i_second


def _stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i,j = last_edge
            j_first, j_second = _find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst


def rotate_via_numpy(xy, radians)->np.ndarray:
    """
    Use numpy to build a rotation matrix and take the dot product.
    
    Parameters
    ----------
    xy
        coordinate 
    radians
        rotation radians
        
    Return
    ----------
    Rotated coordinate 
        
    Refer
    ----------
    https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
    """
    print(f"Rotation {radians * 180 / np.pi} degree")
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, xy.T).T
    return np.array(m)