r"""
Extract cell features from anndata
"""
from random import random
from typing import Optional, List, Union

import scanpy as sc
import numpy as np
import scipy
import scipy.sparse
from anndata import AnnData
import torch
from torch_geometric.data import Data


def Transfer_pyg_Data(adata: AnnData,
                      feature:Optional[str] = 'PCA'
    ) -> Data:
    r"""
    Transfer an adata with spatial info into PyG dataset
    
    Parameters:
    ----------
    adata
        Anndata object
    feature
        use which data to build graph
        - PCA
        - Harmony
        - Your customized embeddings
        
    Note:
    ----------
    Only support 'Spatial_Net' which store in uns yet
    """
    adata = adata.copy()
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    # build Adjacent Matrix
    G = scipy.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + scipy.sparse.eye(G.shape[0])

    edgeList = np.nonzero(G)
    
    # select feature
    assert feature in ['HVG','PCA','RAW','GAN']
    if feature == 'RAW':
        if type(adata.X) == np.ndarray:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
        return data
    elif feature in ['PCA','HVG']:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        if feature == 'HVG':
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))
        sc.pp.scale(adata, max_value=10)
        print('Use PCA to format graph')
        sc.tl.pca(adata, svd_solver='arpack')
        data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['X_pca'].copy()))
        return data, adata.varm['PCs']
    elif feature == 'GAN':
        raise NotImplementedError('Ongoing')


def Transfer_pyg_Datas(adatas: List[AnnData],
                      feature:Optional[str] = 'PCA',
                      hvg:Optional[str] = 'seurat',
                      join:Optional[str] = 'inner'
    ) -> List[Data]:
    r"""
    Transfer adatas with spatial info into PyG datasets
    
    Parameters:
    ----------
    adatas
        List of Anndata objects
    feature
        use which data to build graph
        - PCA
        - Harmony
        - GLUE (**NOTE**: only suitable for multi-omics integration)
        - Your customized embeddings
    hvg
        method used to calculate HVGs
    join
        how to concatenate two adata

        
    Note:
    ----------
    Only support 'Spatial_Net' which store in uns yet
    """
    assert len(adatas) == 2
    assert feature in ['HVG','PCA','RAW','GAN','Harmony','harmony','GLUE','glue','scglue']
    
    adatas = [adata.copy() for adata in adatas ] # May consume more memory
    edgeLists = []
    for adata in adatas:
        adata = adata.copy()
        G_df = adata.uns['Spatial_Net'].copy()
        cells = np.array(adata.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
        G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

        # build Adjacent Matrix
        G = scipy.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), 
                                    shape=(adata.n_obs, adata.n_obs))
        G = G + scipy.sparse.eye(G.shape[0])
        edgeList = np.nonzero(G)
        edgeLists.append(edgeList)

    # select feature
    datas = []
    print(f'Use {feature} feature to format graph')
    if feature == 'RAW':
        for i, adata in enumerate(adatas):
            if type(adata.X) == np.ndarray:
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])),
                            x=torch.FloatTensor(adata.X))  # .todense()
            else:
                data = Data(edge_index=torch.LongTensor(np.array(
                    [edgeLists[i][0], edgeLists[i][1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
            datas.append(data)
        return datas
    
    elif feature in ['glue','GLUE','scglue']:
        for i, adata in enumerate(adatas):
            assert 'X_glue' in adata.obsm.keys()
            data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])),
                        x=torch.FloatTensor(adata.obsm['X_glue']))
            datas.append(data)
        return datas
    
    elif feature in ['PCA','HVG','Harmony','harmony']:
        if 'counts' not in adata.layers.keys():
            adata.layers["counts"] = adata.X.copy()
        adata_all = adatas[0].concatenate(adatas[1], join=join) # join can not be 'outer' !
        if hvg.lower() in ['seurat','seurat_v3']:
            sc.pp.highly_variable_genes(adata_all, n_top_genes=2500, flavor="seurat_v3")
        sc.pp.normalize_total(adata_all, target_sum=1e4)
        sc.pp.log1p(adata_all)
        if hvg.lower() not in ['seurat','seurat_v3']:
            sc.pp.highly_variable_genes(adata_all, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_all = adata_all[:, adata_all.var.highly_variable]
        if feature == 'HVG':
            for i in len(adatas):
                adata = adata_all[adata_all.obs['batch'] == str(i)]
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])), 
                            x=torch.FloatTensor(adata.X.todense()))
                datas.append(data)
            return datas
        sc.pp.scale(adata_all)
        sc.tl.pca(adata_all, svd_solver='auto')
        if feature == 'PCA':
            for i in range(len(adatas)):
                adata = adata_all[adata_all.obs['batch'] == str(i)]
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])), 
                            x=torch.FloatTensor(adata.obsm['X_pca'].copy()))
                datas.append(data)
            return datas
        elif feature == 'harmony' or feature == 'Harmony':
            from harmony import harmonize
            gpu_flag = True if torch.cuda.is_available() and adata_all.shape[0] > 6000 else False
            if gpu_flag:
                print('Harmony is using GPU, which may lead to not duplicature result')
            Z = harmonize(adata_all.obsm['X_pca'], adata_all.obs, random_state=0, max_iter_harmony=30,
                          batch_key='batch', use_gpu=gpu_flag)
            adata_all.obsm['X_harmony'] = Z
            for i in range(len(adatas)):
                adata = adata_all[adata_all.obs['batch'] == str(i)]
                data = Data(edge_index=torch.LongTensor(np.array([edgeLists[i][0], edgeLists[i][1]])), 
                            x=torch.FloatTensor(adata.obsm['X_harmony'].copy()))
                datas.append(data) 
        return datas
    
    elif feature == 'GAN':
        raise NotImplementedError('Ongoing...')


def load_anndata(
    adata: AnnData,
    feature: Optional[str] = 'PCA',
    noise_level: Optional[float] = 0,
    noise_type: Optional[str] = 'uniform',
    edge_homo_ratio: Optional[float] = 0.9,
    return_PCs: Optional[bool] = False
    ) -> List:
    r"""
    Create 2 graphs from single anndata for test
    
    Parameters
    ----------
    adata
        Anndata object
    feature
        feature to use to build graph and align, now support
        - `PCA`
        - `Harmony` (default)
    noise_level
        node noise 
    noise_type
        type of noise, support 'uniform' and 'normal'
    edge_homo_ratio
        ratio of edge in graph2
    return_PCs
        if return adata.varm['PCs'] if use feature 'PCA' (just for benchmark)
        
    Warning
    ----------
    This function is only for test. It generates `two` graphs 
    from single anndata by data augmentation
    """
    if feature=='PCA':
        dataset, PCs = Transfer_pyg_Data(adata, feature=feature)
    else:
        dataset = Transfer_pyg_Data(adata, feature=feature)
    edge1 = dataset.edge_index
    feature1 = dataset.x
    edge2 = edge1.clone()
    ledge = edge2.size(1) # get edge numbers
    edge2 = edge2[:, torch.randperm(ledge)[:int(ledge*edge_homo_ratio)]]
    perm = torch.randperm(feature1.size(0))
    perm_back = torch.tensor(list(range(feature1.size(0))))
    perm_mapping = torch.stack([perm_back, perm])
    edge2 = perm[edge2.view(-1)].view(2, -1) # reset edge order 
    edge2 = edge2[:, torch.argsort(edge2[0])]
    feature2 = torch.zeros(feature1.size())
    feature2[perm] = feature1.clone()
    if noise_type == 'uniform':
        feature2 = feature2 + 2 * (torch.rand(feature2.size())-0.5) * noise_level
    elif noise_type == 'normal':
        feature2 = feature2 + torch.randn(feature2.size()) * noise_level
    if feature=='PCA' and return_PCs:
        return edge1, feature1, edge2, feature2, perm_mapping, PCs
    return edge1, feature1, edge2, feature2, perm_mapping


def load_anndatas(
    adatas: List[AnnData],
    feature: Optional[str] = 'PCA',
    check_order: Optional[bool] = True,
    hvg: Optional[str] = 'seurat',
    join: Optional[str] = 'inner'
    ) -> List:
    r"""
    Extract features and edges from anndata object list 
    
    Parameters
    ----------
    adatas
        Anndata object list
    feature
        feature to use to build graph and align, now support
        - `PCA`
        - `Harmony` (default)
        - `GLUE` (**NOTE**: only suitable for multi-omics integration)
        - Your customized embeddings
    check_order
        if check order of two datasets, ideal order is [large, small]
    hvg
        method to calculate highly variable genes
    join
        `inner` or `outer` when concat two dataset, `inner` is suitable for most situation
    """
    if check_order and adatas[0].shape[0] < adatas[1].shape[0]:
        raise ValueError('Please change the order of adata1 and adata2 or set `check_order=False`')
    datasets = Transfer_pyg_Datas(adatas,feature=feature,hvg=hvg,join=join)
    edges = [dataset.edge_index for dataset in datasets]
    features = [dataset.x for dataset in datasets]
    
    return edges, features
    
 