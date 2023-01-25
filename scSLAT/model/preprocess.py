r"""
Data preprocess and build graph
"""
from typing import Optional, Union

import pandas as pd
import numpy as np
import scanpy as sc
import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
from anndata import AnnData


def Cal_Spatial_Net(adata:AnnData,
                    rad_cutoff:Optional[Union[None,int]]=None,
                    k_cutoff:Optional[Union[None,int]]=None, 
                    model:Optional[str]='Radius',
                    return_data:Optional[bool]=False,
                    verbose:Optional[bool]=True
    ) -> None:
    r"""
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. 
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('Calculating spatial neighbor graph ...')

    if model == 'KNN':
        edge_index = knn_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                                k=k_cutoff, loop=True, num_workers=8)
        edge_index = to_undirected(edge_index, num_nodes=adata.shape[0]) # ensure the graph is undirected
    elif model == 'Radius':
        edge_index = radius_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                                    r=rad_cutoff, loop=True, num_workers=8) 

    graph_df = pd.DataFrame(edge_index.numpy().T, columns=['Cell1', 'Cell2'])
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    graph_df['Cell1'] = graph_df['Cell1'].map(id_cell_trans)
    graph_df['Cell2'] = graph_df['Cell2'].map(id_cell_trans)
    adata.uns['Spatial_Net'] = graph_df
    
    if verbose:
        print(f'The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{graph_df.shape[0]/adata.n_obs} neighbors per cell on average.')

    if return_data:
        return adata


def scanpy_workflow(adata:AnnData,
                    n_top_genes:Optional[int]=2500,
                    n_comps:Optional[int]=50
    ) -> AnnData:
    r"""
    Scanpy workflow using Seurat HVG
    
    Parameters
    ----------
    adata
        adata
    n_top_genes
        n top genes
    n_comps
        n PCA components
        
    Return
    ----------
    anndata object
    """
    if 'counts' not in adata.layers.keys():
        adata.layers["counts"] = adata.X.copy()
    if "highly_variable" not in adata.var_keys() and adata.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    if n_comps > 0:
        sc.tl.pca(adata, n_comps=n_comps, svd_solver="auto")
    return adata