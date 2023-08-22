import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.spatial.distance import cdist
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from helper import intersect, to_dense_array, extract_data_matrix, glmpca_distance
from PASTE2 import partial_pairwise_align_given_cost_matrix


"""
Functions for deciding the overlap percentage between two partially overlapped slices.
"""


def create_graph(adata, degree=4):
    """
    Converts spatial coordinates into graph using networkx library.

    param: adata - ST Slice
    param: degree - number of edges per vertex

    return: 1) G - networkx graph
            2) node_dict - dictionary mapping nodes to spots
    """
    D = cdist(adata.obsm['spatial'], adata.obsm['spatial'])
    # Get column indexes of the degree+1 lowest values per row
    idx = np.argsort(D, 1)[:, 0:degree + 1]
    # Remove first column since it results in self loops
    idx = idx[:, 1:]

    G = nx.Graph()
    for r in range(len(idx)):
        for c in idx[r]:
            G.add_edge(r, c)

    node_dict = dict(zip(range(adata.shape[0]), adata.obs.index))
    return G, node_dict


def generate_graph_from_labels(adata, labels_dict):
    """
    Creates and returns the graph and dictionary {node: cluster_label}
    """
    g, node_to_spot = create_graph(adata)
    spot_to_cluster = labels_dict

    # remove any nodes that are not mapped to a cluster
    removed_nodes = []
    for node in node_to_spot.keys():
        if (node_to_spot[node] not in spot_to_cluster.keys()):
            removed_nodes.append(node)

    for node in removed_nodes:
        del node_to_spot[node]
        g.remove_node(node)

    labels = dict(zip(g.nodes(), [spot_to_cluster[node_to_spot[node]] for node in g.nodes()]))
    return g, labels


def edge_inconsistency_score(g, labels):
    # construct contiguity matrix C which counts pairs of cluster edges
    cluster_names = np.unique(list(labels.values()))
    C = pd.DataFrame(0, index=cluster_names, columns=cluster_names)

    for e in g.edges():
        C[labels[e[0]]][labels[e[1]]] += 1

    C_sum = C.values.sum()
    diagonal = 0
    for i in range(len(cluster_names)):
        diagonal += C[cluster_names[i]][cluster_names[i]]

    return float(C_sum - diagonal) / C_sum


def calculate_convex_hull_edge_inconsistency(sliceA, sliceB, pi):
    sliceA = sliceA.copy()

    source_split = []
    source_mass = np.sum(pi, axis=1)
    for i in range(len(source_mass)):
        if source_mass[i] > 0:
            source_split.append("true")
        else:
            source_split.append("false")
    sliceA.obs["aligned"] = source_split

    source_mapped_points = []
    source_mass = np.sum(pi, axis=1)
    for i in range(len(source_mass)):
        if source_mass[i] > 0:
            source_mapped_points.append(sliceA.obsm['spatial'][i])
    source_mapped_points = np.array(source_mapped_points)
    source_hull = ConvexHull(source_mapped_points)
    source_hull_path = Path(source_mapped_points[source_hull.vertices])
    source_hull_adata = sliceA[sliceA.obs.index[source_hull_path.contains_points(sliceA.obsm['spatial'])]]

    g_A, l_A = generate_graph_from_labels(source_hull_adata, source_hull_adata.obs['aligned'])
    measure_A = edge_inconsistency_score(g_A, l_A)


    sliceB = sliceB.copy()

    target_split = []
    target_mass = np.sum(pi, axis=0)
    for i in range(len(target_mass)):
        if target_mass[i] > 0:
            target_split.append("true")
        else:
            target_split.append("false")
    sliceB.obs["aligned"] = target_split

    target_mapped_points = []
    target_mass = np.sum(pi, axis=0)
    for i in range(len(target_mass)):
        if target_mass[i] > 0:
            target_mapped_points.append(sliceB.obsm['spatial'][i])
    target_mapped_points = np.array(target_mapped_points)
    target_hull = ConvexHull(target_mapped_points)
    target_hull_path = Path(target_mapped_points[target_hull.vertices])
    target_hull_adata = sliceB[sliceB.obs.index[target_hull_path.contains_points(sliceB.obsm['spatial'])]]

    g_B, l_B = generate_graph_from_labels(target_hull_adata, target_hull_adata.obs['aligned'])
    measure_B = edge_inconsistency_score(g_B, l_B)

    return measure_A, measure_B


"""
Main function.
"""
def select_overlap_fraction(sliceA, sliceB, alpha=0.1):
    """
    Estimates the overlap percentage of two ST slices.

    param: sliceA - AnnData object
    param: sliceB - AnnData object
    param: alpha - Alignment tuning parameter. Note: 0 ≤ alpha ≤ 1
    
    return: estimation of the overlap percentage between sliceA and sliceB
    """
    print("PASTE2 model selection procedure.")
    overlap_to_check=[0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]
    # Get transport cost matrix
    A_X, B_X = to_dense_array(extract_data_matrix(sliceA, None)), to_dense_array(extract_data_matrix(sliceB, None))
    M = glmpca_distance(A_X, B_X, latent_dim=50, filter=True, verbose=True)
    # Get an alignment for each overlap percentage
    m_to_pi = {}
    for m in overlap_to_check:
        print("Running PASTE2 with s = " + str(m) + "...")
        pi, log = partial_pairwise_align_given_cost_matrix(sliceA, sliceB, s=m, M=M, alpha=alpha, armijo=False,
                                                        norm=True, return_obj=True, verbose=False)
        m_to_pi[m] = pi
    # Model selection based on edge inconsistency score
    m_to_edge_inconsistency_A = []
    m_to_edge_inconsistency_B = []
    for m in overlap_to_check:
        pi = m_to_pi[m]
        sliceA_measure, sliceB_measure = calculate_convex_hull_edge_inconsistency(sliceA, sliceB, pi)
        m_to_edge_inconsistency_A.append(sliceA_measure)
        m_to_edge_inconsistency_B.append(sliceB_measure)

    half_estimate_A = overlap_to_check[m_to_edge_inconsistency_A.index(max(m_to_edge_inconsistency_A))]
    half_estimate_B = overlap_to_check[m_to_edge_inconsistency_B.index(max(m_to_edge_inconsistency_B))]

    print("Estimation of overlap percentage is " + str(min(2 * min(half_estimate_A, half_estimate_B), 1)))
    return min(2 * min(half_estimate_A, half_estimate_B), 1)



def plot_edge_curve(m_list, source_list, target_list):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(m_list, source_list)
    ax1.set_xlim(1, 0)
    ax1.set_xticks([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ax1.set_xlabel('m')
    ax1.set_ylabel('Edge inconsistency score')
    ax1.set_title('Source slice')
    
    ax2.plot(m_list, target_list)
    ax2.set_xlim(1, 0)
    ax2.set_xticks([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ax2.set_xlabel('m')
    ax2.set_ylabel('Edge inconsistency score')
    ax2.set_title('Target slice')
    
    plt.show()


def select_overlap_fraction_plotting(sliceA, sliceB, alpha=0.1):
    overlap_to_check=[0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]
    # Get transport cost matrix
    A_X, B_X = to_dense_array(extract_data_matrix(sliceA, None)), to_dense_array(extract_data_matrix(sliceB, None))
    M = glmpca_distance(A_X, B_X, latent_dim=50, filter=True)

    m_to_pi = {}
    for m in overlap_to_check:
        print("Running PASTE2 with s = " + str(m) + "...")
        pi, log = partial_pairwise_align_given_cost_matrix(sliceA, sliceB, s=m, M=M, alpha=alpha, armijo=False,
                                                        norm=True, return_obj=True, verbose=False)
        m_to_pi[m] = pi

    m_to_edge_inconsistency_A = []
    m_to_edge_inconsistency_B = []
    for m in overlap_to_check:
        pi = m_to_pi[m]
        sliceA_measure, sliceB_measure = calculate_convex_hull_edge_inconsistency(sliceA, sliceB, pi)
        m_to_edge_inconsistency_A.append(sliceA_measure)
        m_to_edge_inconsistency_B.append(sliceB_measure)

    plot_edge_curve(overlap_to_check, m_to_edge_inconsistency_A, m_to_edge_inconsistency_B)

    half_estimate_A = overlap_to_check[m_to_edge_inconsistency_A.index(max(m_to_edge_inconsistency_A))]
    half_estimate_B = overlap_to_check[m_to_edge_inconsistency_B.index(max(m_to_edge_inconsistency_B))]

    print("Estimation of overlap percentage is " + str(min(2 * min(half_estimate_A, half_estimate_B), 1)))
    return min(2 * min(half_estimate_A, half_estimate_B), 1)




