import matplotlib.pyplot as plt
import numpy as np
import os
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix
import pandas as pd
import torch
from .ST_utils import mclust_R
from .typehint import RandomState
from .utils import get_rs
from .metrics import mean_average_precision, normalized_mutual_info, avg_silhouette_width, graph_connectivity, neighbor_conservation
from sklearn.metrics import adjusted_rand_score as ari_score
import colorcet as cc
from collections import Counter
import sklearn
from matplotlib import rcParams
import math


def assign_color(adata_region, adata_cluster, region_palette, region_key, cluster_key, cluster_list):
    cluster_palette = {k:None for k in cluster_list}
    region_list = list(region_palette.keys())
    for region_name in region_list:
        adata_region_obs = adata_region[adata_region.obs[region_key].isin([region_name])].obs_names
        overlap_max = 0
        aligned_cluster_name = None
        for cluster_name in cluster_list:
            adata_cluster_obs = adata_cluster[adata_cluster.obs[cluster_key].isin([cluster_name])].obs_names
            overlap_num = len(set(adata_cluster_obs).intersection(set(adata_region_obs)))
            if overlap_num > overlap_max:
                overlap_max = overlap_num
                aligned_cluster_name = cluster_name
        if aligned_cluster_name == None:
            aligned_cluster_name = cluster_list[-1]
        cluster_list.remove(aligned_cluster_name)

        cluster_palette[aligned_cluster_name] = region_palette[region_name]

    return cluster_palette



def merge_embedding(adata_dict, key_umap='STACAME', if_annotation = True):
    k = 0
    for species_id, adata in adata_dict.items():
        if k == 0:
            embedding_X = adata.obsm[key_umap]
            embedding_spatial = adata.obsm['spatial']
            embedding_obs_name = list(adata.obs_names)
            embedding_slice_name = list(adata.obs['slice_name']) 
            embedding_batch_name = list(adata.obs['batch_name'])
            embedding_species_id = list(adata.obs['species_id'])
            if 'annotation' in adata.obs:
                embedding_annotation = list(adata.obs['annotation']) 
        else:
            embedding_X = np.concatenate((embedding_X, adata.obsm[key_umap]), axis=0)
    
            embedding_spatial = np.concatenate((embedding_spatial, adata.obsm['spatial']), axis=0)
    
            embedding_obs_name = embedding_obs_name + list(adata.obs_names)
            embedding_slice_name = embedding_slice_name + list(adata.obs['slice_name']) 
            embedding_batch_name = embedding_batch_name + list(adata.obs['batch_name'])
            embedding_species_id = embedding_species_id + list(adata.obs['species_id'])
            if 'annotation' in adata.obs and 'embedding_annotation' in locals():
                embedding_annotation = embedding_annotation + list(adata.obs['annotation'])
            
        k += 1        
    
    adata_embedding = ad.AnnData(X = embedding_X)
    adata_embedding.obs_names = embedding_obs_name
    adata_embedding.obsm['spatial'] = embedding_spatial
    adata_embedding.obs['slice_name'] = embedding_slice_name
    adata_embedding.obs['batch_name'] = embedding_batch_name
    adata_embedding.obs['species_id'] = embedding_species_id
    if 'annotation' in adata_embedding.obs or if_annotation == True:
        if 'embedding_annotation' in locals():
            adata_embedding.obs['annotation'] = embedding_annotation
    if 'region_name' in adata_embedding.obs:
        if 'embedding_annotation' in locals():
            adata_embedding.obs['region_name'] = embedding_annotation
    adata_embedding.obsm[key_umap] = embedding_X
    return adata_embedding


def rotate_spots(spatial_mat, theta):
    '''
    Suppose the angle we want two rotate is theta
    1. Compute the center of the coordinates: a1 = mean(xlim), b1 = mean(ylim);
    2. For any spot a2, b2, vector z = (a2 - a1, b2 - b1) = (x, y)
    3. The vector z' = (a3 - a1, b3 - b1) = (x', y'), then
        x' = x*cos(theta) - y*sin(theta)
        y' = x*sin(theta) - y*cos(theta)
    4. Therefore, 
        a3 = x' + a1 = x*cos(theta) - y*sin(theta) + a1
        b3 = y' + b1 = x*sin(theta) - y*cos(theta) + b1
    '''
    spatial_mat_rotate = np.zeros(spatial_mat.shape)
    x_lim_min = np.min(spatial_mat[:, 0])
    x_lim_max = np.max(spatial_mat[:, 0])
    y_lim_min = np.min(spatial_mat[:, 1])
    y_lim_max = np.max(spatial_mat[:, 1])
    a1 = (x_lim_max - x_lim_min)/2 + x_lim_min
    b1 = (y_lim_max - y_lim_min)/2 + y_lim_min
    for i in range(spatial_mat.shape[0]):
        a2 = spatial_mat[i, 0]
        b2 = spatial_mat[i, 1]
        #print(a2, b2)
        x = a2 - a1
        y = b2 - b1
        a3 = x*math.cos(theta) - y*math.sin(theta) + a1
        b3 = x*math.sin(theta) + y*math.cos(theta) + b1
        spatial_mat_rotate[i, 0] = a3
        spatial_mat_rotate[i, 1] = b3
    return spatial_mat_rotate



def convert_dict2adata(adata_dict, key_umap='STACAME'):
    k = 0
    for species_id, adata in adata_dict.items():
        if species_id == 'Zebrafish':
            adata.obs['annotation'] = adata.obs['layer_annotation']
        if k == 0:
            embedding_X = adata.obsm[key_umap]
            embedding_spatial = adata.obsm['spatial']
            embedding_obs_name = list(adata.obs_names)
            embedding_slice_name = list(adata.obs['slice_name']) 
            embedding_batch_name = list(adata.obs['batch_name'])
            embedding_species_id = list(adata.obs['species_id'])
            embedding_annotation = list(adata.obs['annotation'])
            embedding_mclust_separate = list(adata.obs['mclust_separate']) 
            embedding_mclust = list(adata.obs['mclust']) 
        else:
            embedding_X = np.concatenate((embedding_X, adata.obsm[key_umap]), axis=0)
    
            embedding_spatial = np.concatenate((embedding_spatial, adata.obsm['spatial']), axis=0)
    
            embedding_obs_name = embedding_obs_name + list(adata.obs_names)
            embedding_slice_name = embedding_slice_name + list(adata.obs['slice_name']) 
            embedding_batch_name = embedding_batch_name + list(adata.obs['batch_name'])
            embedding_species_id = embedding_species_id + list(adata.obs['species_id'])
            embedding_annotation = embedding_annotation + list(adata.obs['annotation'])
            embedding_mclust_separate = embedding_mclust_separate + list(adata.obs['mclust_separate']) 
            embedding_mclust = embedding_mclust + list(adata.obs['mclust']) 

        
        k += 1

    adata_embedding = ad.AnnData(X = embedding_X, obs=embedding_obs_name)
    adata_embedding.obs_names = embedding_obs_name
    adata_embedding.obsm['spatial'] = embedding_spatial
    adata_embedding.obsm[key_umap] = embedding_X
    adata_embedding.obs['slice_name'] = embedding_slice_name
    adata_embedding.obs['batch_name'] = embedding_batch_name
    adata_embedding.obs['species_id'] = embedding_species_id
    adata_embedding.obs['annotation'] = embedding_annotation
    adata_embedding.obs['mclust_separate'] = embedding_mclust_separate
    adata_embedding.obs['mclust'] = embedding_mclust

    return adata_embedding



def get_alignment_score(adata, type_name, method_name):
    '''
    Compute alignemnt score
    '''
    type_labels = [x for x in Counter(adata.obs[type_name]).keys()]
    start_label = type_labels[0]
    label_num = len(type_labels)
    type_labels_remain = type_labels
    type_labels_remain.pop(0)
    start_adata = adata[adata.obs[type_name].isin([start_label])]
    #print(start_adata)
    alignment_score = 0
    for label_remain in type_labels_remain:
        adata_remain = adata[adata.obs[type_name].isin([label_remain])]
        #print(adata_remain)
        X = np.concatenate([start_adata.obsm[method_name], adata_remain.obsm[method_name]], axis=0)
        Y = np.concatenate([np.zeros((start_adata.n_obs, 1)), np.ones((adata_remain.n_obs, 1))], axis=0)
        alignment_score = alignment_score + seurat_alignment_score(X, Y, neighbor_frac=0.05, n_repeats=10)
    
    alignment_score = alignment_score / (label_num - 1)
    return alignment_score


def get_score_annotation(adata, type_name, method_name, annotation_name, annotation_dict, metric='Seurat_alignment_score'):
    '''
    Compute score related to batch and annotation
    '''
    type_labels = [x for x in Counter(adata.obs[type_name]).keys()]
    start_label = type_labels[0]
    label_num = len(type_labels)
    type_labels_remain = type_labels
    type_labels_remain.pop(0)
    alignment_score = 0
    region_num = len(annotation_dict[start_label])
    k = 0
    for start_region in annotation_dict[start_label]:
        region_alignment_score = 0
        start_adata = adata[adata.obs[type_name].isin([start_label])]
        start_adata = start_adata[start_adata.obs[annotation_name].isin([start_region])]
        #print(start_adata)
        alignment_score = 0
        for label_remain in type_labels_remain:
            region_remain = annotation_dict[label_remain][k]
            adata_remain = adata[adata.obs[type_name].isin([label_remain])]
            adata_remain = adata_remain[adata_remain.obs[annotation_name].isin([region_remain])]
            #print(adata_remain)
            X = np.concatenate([start_adata.obsm[method_name], adata_remain.obsm[method_name]], axis=0)
            Y = np.concatenate([np.zeros((start_adata.n_obs, 1)), np.ones((adata_remain.n_obs, 1))], axis=0)
            region_alignment_score = region_alignment_score + seurat_alignment_score(X, Y, neighbor_frac=0.2, n_repeats=10)
        
        region_alignment_score = region_alignment_score / (label_num - 1)
        alignment_score = alignment_score + region_alignment_score
        k += 1
    alignment_score = alignment_score / region_num
    return alignment_score


def get_alignment_score_annotation(adata, type_name, method_name, annotation_name, annotation_dict, neighbor_frac=0.02):
    '''
    Compute alignemnt score
    '''
    type_labels = [x for x in Counter(adata.obs[type_name]).keys()]
    start_label = type_labels[0]
    label_num = len(type_labels)
    type_labels_remain = type_labels
    type_labels_remain.pop(0)

    alignment_score = 0

    region_num = len(annotation_dict[start_label])

    k = 0

    alignment_score = 0.0
    for start_region in annotation_dict[start_label]:
        region_alignment_score = 0.0
        start_adata = adata[adata.obs[type_name].isin([start_label])]
        start_adata = start_adata[start_adata.obs[annotation_name].isin([start_region])]
        for label_remain in type_labels_remain:
            region_remain = annotation_dict[label_remain][k]
            adata_remain = adata[adata.obs[type_name].isin([label_remain])]
            adata_remain = adata_remain[adata_remain.obs[annotation_name].isin([region_remain])]
            X = np.concatenate([start_adata.obsm[method_name], adata_remain.obsm[method_name]], axis=0)
            Y = np.concatenate([np.zeros((start_adata.n_obs, 1)), np.ones((adata_remain.n_obs, 1))], axis=0)
            region_alignment_score = region_alignment_score + seurat_alignment_score(X, Y, neighbor_frac=neighbor_frac, n_repeats=10)
        
        region_alignment_score = region_alignment_score / (label_num - 1)
        alignment_score = alignment_score + region_alignment_score
        k += 1
    
    alignment_score = alignment_score / region_num
    return alignment_score



def seurat_alignment_score(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01,
        n_repeats: int = 4, random_state: RandomState = None, **kwargs
) -> float:
    r"""
    Seurat alignment score

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    neighbor_frac
        Nearest neighbor fraction
    n_repeats
        Number of subsampling repeats
    random_state
        Random state
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    sas
        Seurat alignment score
    """
    rs = get_rs(random_state)
    idx_list = [np.where(y == u)[0] for u in np.unique(y)]
    min_size = min(idx.size for idx in idx_list)
    repeat_scores = []
    for _ in range(n_repeats):
        subsample_idx = np.concatenate([
            rs.choice(idx, min_size, replace=False)
            for idx in idx_list
        ])
        subsample_x = x[subsample_idx]
        subsample_y = y[subsample_idx]
        k = max(round(subsample_idx.size * neighbor_frac), 1)
        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k + 1, **kwargs
        ).fit(subsample_x)
        nni = nn.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
            subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1)
        ).sum(axis=1).mean()
        repeat_score = (k - same_y_hits) * len(idx_list) / (k * (len(idx_list) - 1))
        repeat_scores.append(min(repeat_score, 1))  # score may exceed 1, if same_y_hits is lower than expected by chance
    return np.mean(repeat_scores).item()





def clustering_umap(adata_dict, key_umap='STACAME'):
    '''
    Do UMAP for each slice in each species for obsm[key_umap]
    '''
    k = 0
    for species_id, adata in adata_dict.items():
        if species_id == 'Zebrafish':
            adata.obs['annotation'] = adata.obs['layer_annotation']
        if k == 0:
            embedding_X = adata.obsm[key_umap]
            embedding_spatial = adata.obsm['spatial']
            embedding_obs_name = list(adata.obs_names)
            embedding_slice_name = list(adata.obs['slice_name']) 
            embedding_batch_name = list(adata.obs['batch_name'])
            embedding_species_id = list(adata.obs['species_id'])
            embedding_annotation = list(adata.obs['annotation']) 
        else:
            embedding_X = np.concatenate((embedding_X, adata.obsm[key_umap]), axis=0)
    
            embedding_spatial = np.concatenate((embedding_spatial, adata.obsm['spatial']), axis=0)
    
            embedding_obs_name = embedding_obs_name + list(adata.obs_names)
            embedding_slice_name = embedding_slice_name + list(adata.obs['slice_name']) 
            embedding_batch_name = embedding_batch_name + list(adata.obs['batch_name'])
            embedding_species_id = embedding_species_id + list(adata.obs['species_id'])
            embedding_annotation = embedding_annotation + list(adata.obs['annotation'])
            
        k += 1
        # Visualize UMAP of each species
        sc.pp.neighbors(adata,  n_neighbors=20, use_rep=key_umap, metric='cosine',  random_state=666)
        sc.tl.leiden(adata, random_state=666, key_added="leiden", resolution=0.5)
        sc.tl.umap(adata, min_dist=1, random_state=666)
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams["figure.figsize"] = (3, 3)
        plt.rcParams['font.size'] = 10
        sc.pl.umap(adata, color=['batch_name', 'leiden', 'annotation'], ncols=3, wspace=0.7, show=False)
        fig = plt.gcf()
        fig.suptitle(species_id)
        plt.show()
            
    
    adata_embedding = ad.AnnData(X = embedding_X, obs=embedding_obs_name)
    adata_embedding.obsm['spatial'] = embedding_spatial
    adata_embedding.obs['slice_name'] = embedding_slice_name
    adata_embedding.obs['batch_name'] = embedding_batch_name
    adata_embedding.obs['species_id'] = embedding_species_id
    adata_embedding.obs['annotation'] = embedding_annotation
    
    sc.pp.neighbors(adata_embedding,  n_neighbors=20, use_rep='X', metric='cosine',  random_state=666)
    sc.tl.leiden(adata_embedding, random_state=666, key_added="leiden", resolution=0.5)
    
    sc.tl.umap(adata_embedding, min_dist=1, random_state=666)

    species_ids = list(adata_dict.keys())
    
    species_color = ['#4778FA', '#8A1C62', '#ED7A43']
    species_color_dict = dict(zip(species_ids, species_color))
    adata_embedding.uns['species_colors'] = [species_color_dict[x] for x in adata_embedding.obs.species_id]
    
    
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams["figure.figsize"] = (3, 3)
    plt.rcParams['font.size'] = 10

    # mclust clustering 
    sc.pl.umap(adata_embedding, color=['species_id', 'batch_name', 'leiden', 'annotation'], ncols=2, wspace=0.5, show=True)

    fig, axes = plt.subplots(len(species_ids), 1,  figsize=(3, 3*len(species_ids))) #, dpi=500
    for i in range(len(species_ids)):
        species_id = species_ids[i]
        axes[i].set_title(species_id)
        adata_mh = adata_embedding[adata_embedding.obs['species_id'].isin([species_id])]
        ax = sc.pl.umap(adata_embedding, show=False, ax=axes[i])
        sc.pl.umap(adata_mh, color='annotation', ax=ax,  wspace=0.5, show=False, size=10, legend_loc='on data') 
    plt.show()



def clustering_umap_spatial(adata_dict, key_umap='STAGATE'):
    k = 0
    for species_id, adata in adata_dict.items():
        if k == 0:
            embedding_X = adata.obsm[key_umap]
            embedding_spatial = adata.obsm['spatial']
            embedding_obs_name = list(adata.obs_names)
            embedding_slice_name = list(adata.obs['slice_name']) 
            embedding_batch_name = list(adata.obs['batch_name'])
            embedding_species_id = list(adata.obs['species_id'])
            embedding_annotation = list(adata.obs['annotation']) 
        else:
            embedding_X = np.concatenate((embedding_X, adata.obsm[key_umap]), axis=0)
            embedding_spatial = np.concatenate((embedding_spatial, adata.obsm['spatial']), axis=0)
            embedding_obs_name = embedding_obs_name + list(adata.obs_names)
            embedding_slice_name = embedding_slice_name + list(adata.obs['slice_name']) 
            embedding_batch_name = embedding_batch_name + list(adata.obs['batch_name'])
            embedding_species_id = embedding_species_id + list(adata.obs['species_id'])
            embedding_annotation = embedding_annotation + list(adata.obs['annotation'])

        k += 1
        #Visualize UMAP of each species
        sc.pp.neighbors(adata,  n_neighbors=20, use_rep=key_umap, metric='cosine',  random_state=666)
        sc.tl.leiden(adata, random_state=666, key_added="leiden", resolution=0.5)
        sc.tl.umap(adata, min_dist=1, random_state=666)
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams["figure.figsize"] = (3, 3)
        plt.rcParams['font.size'] = 10
        num_clusters = len(adata.obs['annotation'].unique())
        mclust_R(adata, num_cluster=num_clusters, used_obsm=key_umap)
        print('mclust, ARI = %01.3f' % ari_score(adata.obs['annotation'], adata.obs['mclust']))
        sc.pl.umap(adata, color=['batch_name', 'leiden', 'annotation', 'mclust'], ncols=3, wspace=0.7, show=False)
        fig = plt.gcf()
        fig.suptitle(species_id)
        plt.show()

        adata_dict[species_id] = adata

    adata_embedding = ad.AnnData(X = embedding_X, obs=embedding_obs_name)
    adata_embedding.obsm['spatial'] = embedding_spatial
    adata_embedding.obs['slice_name'] = embedding_slice_name
    adata_embedding.obs['batch_name'] = embedding_batch_name
    adata_embedding.obs['species_id'] = embedding_species_id
    adata_embedding.obs['annotation'] = embedding_annotation
    
    return adata_dict, adata_embedding
    

def spatial_annotation_species(adata_dict, img_key=None, scale_factor=None, spot_size=1, title_size=12):
    Batch_list = []
    for species_id in adata_dict.keys():
        Batch_list.append(adata_dict[species_id])

    species_num = len(list(adata_dict.keys()))

    species_list = list(adata_dict.keys())
    ARI_list = []
    for bb in range(species_num):
        ARI_list.append(round(ari_score(Batch_list[bb].obs['annotation'], Batch_list[bb].obs['mclust']), 2))
    
    fig, ax = plt.subplots(species_num, 2, figsize=(10, 10), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    for ax_r in range(species_num):
        species_id = species_list[ax_r]
        _sc_0 = sc.pl.spatial(Batch_list[ax_r], img_key=img_key, scale_factor=scale_factor, color=['mclust'], title=[''], legend_loc=None, legend_fontsize=12, show=False, ax=ax[ax_r][0], frameon=False,
                              spot_size=spot_size, add_outline=False, alpha=1, edges=False)
        _sc_0[0].set_title("ARI=" + str(ARI_list[ax_r]), size=title_size)
        _sc_1 = sc.pl.spatial(Batch_list[ax_r], img_key=img_key, color=['annotation'], title=[species_id + ' Annotation'], scale_factor=scale_factor,
                              legend_loc=None, legend_fontsize=12, show=False, ax=ax[ax_r][1], frameon=False,
                              spot_size=spot_size, add_outline=False, alpha=1, edges=False)

    plt.show()