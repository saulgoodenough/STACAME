from .STACAME import STACAME
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
from STACAME import create_dictionary_mnn
from STACAME import STACAME
import scanpy as sc
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import k_hop_subgraph
from math import ceil
import anndata as ad
from collections import Counter
from .utils_OT import *
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import random
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import NeighborSampler
from torch.optim.lr_scheduler import StepLR

def random_list(N, subsampling_rate):
    num = int(N * subsampling_rate)
    result = random.sample(range(0, N), num)
    result = list(set(result))
    result.sort()
    return result


def clustering_umap(adata_dict, key_umap='STACAME'):
    """
    Multi‑species joint UMAP visualization and leiden clustering.

    Parameters
    ----------
    adata_dict : dict
        Keys are species identifiers, values are AnnData objects.
    key_umap : str
        Key in ``obsm`` that stores the embedding to be used for
        neighbourhood graph construction.

    Returns
    -------
    None
        All results are displayed as matplotlib figures.
    """
    # Determine automatic colour palette for species
    species_ids = list(adata_dict.keys())
    n_species = len(species_ids)
    if n_species <= 10:
        species_color = sns.color_palette("tab10", n_colors=n_species).as_hex()
    elif n_species <= 20:
        species_color = sns.color_palette("tab20", n_colors=n_species).as_hex()
    else:
        species_color = sns.color_palette(cc.glasbey, n_colors=n_species).as_hex()

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
                embedding_annotation = ['None'] * adata.n_obs
        else:
            embedding_X = np.concatenate((embedding_X, adata.obsm[key_umap]), axis=0)
            embedding_spatial = np.concatenate((embedding_spatial, adata.obsm['spatial']), axis=0)
            embedding_obs_name = embedding_obs_name + list(adata.obs_names)
            embedding_slice_name = embedding_slice_name + list(adata.obs['slice_name'])
            embedding_batch_name = embedding_batch_name + list(adata.obs['batch_name'])
            embedding_species_id = embedding_species_id + list(adata.obs['species_id'])
            if 'annotation' in adata.obs:
                embedding_annotation = embedding_annotation + list(adata.obs['annotation'])
            else:
                embedding_annotation = embedding_annotation + ['None'] * adata.n_obs

        k += 1
        # Per‑species UMAP (visualization is optional; we skip it here for brevity)
        sc.pp.neighbors(adata, n_neighbors=20, use_rep=key_umap, metric='cosine', random_state=666)
        sc.tl.leiden(adata, random_state=666, key_added="leiden", resolution=0.5)
        sc.tl.umap(adata, min_dist=1, random_state=666)
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams["figure.figsize"] = (3, 3)
        plt.rcParams['font.size'] = 10

    # Build joint AnnData object
    adata_embedding = ad.AnnData(X=embedding_X,
                                  obs=pd.DataFrame(index=embedding_obs_name))
    adata_embedding.obsm['spatial'] = embedding_spatial
    adata_embedding.obs['slice_name'] = embedding_slice_name
    adata_embedding.obs['batch_name'] = embedding_batch_name
    adata_embedding.obs['species_id'] = embedding_species_id
    if 'annotation' in adata.obs:
        adata_embedding.obs['annotation'] = embedding_annotation

    sc.pp.neighbors(adata_embedding, n_neighbors=20, use_rep='X', metric='cosine', random_state=666)
    sc.tl.leiden(adata_embedding, random_state=666, key_added="leiden", resolution=0.5)

    print(adata_embedding.X.shape)

    sc.tl.umap(adata_embedding, min_dist=1, random_state=666)

    # Assign the automatically chosen species colours
    species_color_dict = dict(zip(species_ids, species_color))
    adata_embedding.uns['species_colors'] = [species_color_dict[x] for x in adata_embedding.obs.species_id]

    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = 10

    if 'annotation' in adata.obs:
        sc.pl.umap(adata_embedding,
                   color=['species_id', 'batch_name', 'leiden', 'annotation'],
                   ncols=2, wspace=0.5, show=True)
    else:
        sc.pl.umap(adata_embedding,
                   color=['species_id', 'batch_name', 'leiden'],
                   ncols=2, wspace=0.5, show=True)

    if 'annotation' in adata.obs:
        fig, axes = plt.subplots(n_species, 1, figsize=(5, 4 * n_species))
        if n_species == 1:
            axes = [axes]
        for i, species_id in enumerate(species_ids):
            axes[i].set_title(species_id)
            adata_mh = adata_embedding[adata_embedding.obs['species_id'] == species_id]
            unique_annot = adata_mh.obs['annotation'].unique()
            color_list = sns.color_palette(cc.glasbey, n_colors=len(unique_annot)).as_hex()
            palette = dict(zip(unique_annot, color_list))
            ax = sc.pl.umap(adata_embedding, show=False, ax=axes[i])
            sc.pl.umap(adata_mh, color='annotation', ax=ax, wspace=0.5,
                       show=False, legend_loc='on data', palette=palette)
        plt.show()

        fig, axes = plt.subplots(n_species, 1, figsize=(5, 4 * n_species))
        if n_species == 1:
            axes = [axes]
        for i, species_id in enumerate(species_ids):
            axes[i].set_title(species_id)
            adata_mh = adata_embedding[adata_embedding.obs['species_id'] == species_id]
            unique_annot = adata_mh.obs['annotation'].unique()
            color_list = sns.color_palette(cc.glasbey, n_colors=len(unique_annot)).as_hex()
            palette = dict(zip(unique_annot, color_list))
            ax = sc.pl.umap(adata_embedding, show=False, ax=axes[i])
            sc.pl.umap(adata_mh, color='annotation', ax=ax, wspace=0.5,
                       show=False, legend_loc='right margin', palette=palette)
        plt.show()

    return None


def clustering_umap_downsampling(adata_dict, key_umap='STACAME', downsampling_rate=0.1):
    """
    Multi‑species UMAP with random down‑sampling per species.

    Parameters
    ----------
    adata_dict : dict
        Keys are species identifiers, values are AnnData objects.
    key_umap : str
        Key in ``obsm`` for the embedding.
    downsampling_rate : float
        Fraction of cells to retain from each species (default 0.1).

    Returns
    -------
    None
    """
    species_ids = list(adata_dict.keys())
    n_species = len(species_ids)

    # Automatic species colour assignment
    if n_species <= 10:
        species_color = sns.color_palette("tab10", n_colors=n_species).as_hex()
    elif n_species <= 20:
        species_color = sns.color_palette("tab20", n_colors=n_species).as_hex()
    else:
        species_color = sns.color_palette(cc.glasbey, n_colors=n_species).as_hex()

    k = 0
    for species_id, adata_full in adata_dict.items():
        # Down‑sample using the user‑supplied rate (not hard‑coded 0.1)
        adata = sc.pp.subsample(adata_full, fraction=downsampling_rate, copy=True)

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

        sc.pp.neighbors(adata, n_neighbors=20, use_rep=key_umap, metric='cosine', random_state=666)
        sc.tl.leiden(adata, random_state=666, key_added="leiden", resolution=0.5)
        sc.tl.umap(adata, min_dist=1, random_state=666)
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams["figure.figsize"] = (3, 3)
        plt.rcParams['font.size'] = 10
        if 'annotation' in adata.obs:
            sc.pl.umap(adata, color=['batch_name', 'leiden', 'annotation'],
                       ncols=3, wspace=0.7, show=False)
        else:
            sc.pl.umap(adata, color=['batch_name', 'leiden'],
                       ncols=3, wspace=0.7, show=False)
        fig = plt.gcf()
        fig.suptitle(species_id)
        plt.show()

    adata_embedding = ad.AnnData(X=embedding_X,
                                  obs=pd.DataFrame(index=embedding_obs_name))
    adata_embedding.obsm['spatial'] = embedding_spatial
    adata_embedding.obs['slice_name'] = embedding_slice_name
    adata_embedding.obs['batch_name'] = embedding_batch_name
    adata_embedding.obs['species_id'] = embedding_species_id
    if 'annotation' in adata.obs and 'embedding_annotation' in locals():
        adata_embedding.obs['annotation'] = embedding_annotation

    sc.pp.neighbors(adata_embedding, n_neighbors=20, use_rep='X', metric='cosine', random_state=666)
    sc.tl.leiden(adata_embedding, random_state=666, key_added="leiden", resolution=0.5)

    print(adata_embedding.X.shape)

    sc.tl.umap(adata_embedding, min_dist=1, random_state=666)

    species_color_dict = dict(zip(species_ids, species_color))
    adata_embedding.uns['species_colors'] = [species_color_dict[x] for x in adata_embedding.obs.species_id]

    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = 10
    if 'annotation' in adata.obs and 'embedding_annotation' in locals():
        sc.pl.umap(adata_embedding,
                   color=['species_id', 'batch_name', 'leiden', 'annotation'],
                   ncols=2, wspace=0.5, show=True)
    else:
        sc.pl.umap(adata_embedding,
                   color=['species_id', 'batch_name', 'leiden'],
                   ncols=2, wspace=0.5, show=True)

    if 'annotation' in adata.obs and 'embedding_annotation' in locals():
        fig, axes = plt.subplots(n_species, 1, figsize=(5, 4 * n_species))
        if n_species == 1:
            axes = [axes]
        for i, species_id in enumerate(species_ids):
            axes[i].set_title(species_id)
            adata_mh = adata_embedding[adata_embedding.obs['species_id'] == species_id]
            unique_annot = adata_mh.obs['annotation'].unique()
            color_list = sns.color_palette(cc.glasbey, n_colors=len(unique_annot)).as_hex()
            palette = dict(zip(unique_annot, color_list))
            ax = sc.pl.umap(adata_embedding, show=False, ax=axes[i])
            sc.pl.umap(adata_mh, color='annotation', ax=ax, wspace=0.5,
                       show=False, legend_loc='on data', palette=palette)
        plt.show()

        fig, axes = plt.subplots(n_species, 1, figsize=(5, 4 * n_species))
        if n_species == 1:
            axes = [axes]
        for i, species_id in enumerate(species_ids):
            axes[i].set_title(species_id)
            adata_mh = adata_embedding[adata_embedding.obs['species_id'] == species_id]
            unique_annot = adata_mh.obs['annotation'].unique()
            color_list = sns.color_palette(cc.glasbey, n_colors=len(unique_annot)).as_hex()
            palette = dict(zip(unique_annot, color_list))
            ax = sc.pl.umap(adata_embedding, show=False, ax=axes[i])
            sc.pl.umap(adata_mh, color='annotation', ax=ax, wspace=0.5,
                       show=False, legend_loc='right margin', palette=palette)
        plt.show()

    return None


def train_STACAME(adata_species_dict,
                  triplet_ind_species_dict,
                  edge_ndarray_species,
                  triplet_ind_sections_dict=None,
                  edge_ndarray_sections=None,
                  hidden_dims=[512, 30],
                  stagate_epoch=500,
                  n_epochs_species=2000,
                  lr=0.001,
                  key_added='STACAME',
                  gradient_clipping=5.,
                  weight_decay=0.0001,
                  lr_wd=0.001,
                  weight_decay_wd=5e-4,
                  margin=1.0,
                  margin_species=1.0,
                  lr_species=0.001,
                  beta=1,
                  verbose=False,
                  random_seed=666,
                  iter_comb=None,
                  knn_neigh=20,
                  device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),
                  pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                  mse_beta=1,
                  tri_beta=1,
                  mmd_beta=1,
                  mmd_batch_size=2048,
                  concate_pca_dim=None,
                  if_knn_mnn_graph=True,
                  if_integrate_within_species=False,
                  if_return_loss=False,
                  structure_beta=0.0,
                  structure_sampling_ratio=1.0,
                  max_structure_edges=10000):
    """
    Train a graph-attention autoencoder and align cross-species spots via
    triplet constraints in the latent embedding space.

    The function first pretrains a STAGATE model for each species individually.
    Then, a joint decoder is trained on the concatenated latent representations
    using MSE reconstruction loss, cross-species triplet margin loss, optional
    within-species slice triplet loss, a maximum mean discrepancy (MMD) term
    that encourages mixing of species in the latent space, and an optional
    geometric structure preservation loss that keeps the local adjacency
    structure of each species consistent between original expression and
    latent embeddings.

    Parameters
    ----------
    adata_species_dict : dict of {str: anndata.AnnData}
        Dictionary mapping species names to AnnData objects. Each object must
        have ``.X`` (dense or sparse), ``.obs['batch_name']``, ``.uns['edgeList']``,
        and optionally ``.var['highly_variable']`` or ``.uns['highly_variable']``.
    triplet_ind_species_dict : dict
        Cross-species triplet indices with keys ``'anchor_ind_species'``,
        ``'positive_ind_species'``, ``'negative_ind_species'``. Each value is
        a 1D array of indices into the concatenated spot order.
    edge_ndarray_species : np.ndarray
        Cross-species mutual nearest neighbour (MNN) graph edges, shape (2, n_edges).
    triplet_ind_sections_dict : dict, optional
        Within-species slice-level triplet indices with analogous keys
        ``'anchor_ind_sections'``, ``'positive_ind_sections'``,
        ``'negative_ind_sections'``. Required if ``if_integrate_within_species``
        is True.
    edge_ndarray_sections : np.ndarray, optional
        Within-species slice MNN graph edges, shape (2, n_edges). Required if
        ``if_integrate_within_species`` and ``if_knn_mnn_graph`` are both True.
    hidden_dims : list of int, optional
        Hidden layer dimensions for the encoder and decoder. The first element
        is the dimension after the input layer, the second is the bottleneck
        dimension.
    stagate_epoch : int or dict, optional
        Number of pretraining epochs per species. If an int, it is used for all
        species; if a dict, it must map species names to epoch counts.
    n_epochs_species : int, optional
        Number of epochs for the cross-species joint training phase.
    lr : float, optional
        Learning rate for the per-species STAGATE pretraining.
    key_added : str, optional
        Key under which the final latent embedding is stored in
        ``adata_species_dict[species].obsm``.
    gradient_clipping : float, optional
        Maximum gradient norm for clipping during joint training.
    weight_decay : float, optional
        Weight decay for the Adam optimiser during pretraining.
    lr_wd : float, optional
        (Unused in this light version; kept for compatibility.)
    weight_decay_wd : float, optional
        (Unused in this light version; kept for compatibility.)
    margin : float, optional
        Margin for the within-species slice triplet loss.
    margin_species : float, optional
        Margin for the cross-species triplet loss.
    lr_species : float, optional
        Learning rate for the joint cross-species training phase.
    beta : float, optional
        Weight of the within-species slice triplet loss in the total loss
        (only active when ``if_integrate_within_species==True``).
    verbose : bool, optional
        If True, log training progress every 100 epochs.
    random_seed : int, optional
        Random seed for reproducibility.
    iter_comb : tuple or None, optional
        (Reserved for multi-batch ordering; unused in the light version.)
    knn_neigh : int, optional
        Number of nearest neighbours used when constructing MNN graphs.
    device : torch.device, optional
        Device for the cross-species training.
    pretrain_device : torch.device, optional
        Device for the per-species pretraining.
    mse_beta : float, optional
        Weight of the MSE reconstruction loss.
    tri_beta : float, optional
        Weight of the cross-species triplet loss.
    mmd_beta : float, optional
        Weight of the MMD loss.
    mmd_batch_size : int, optional
        Batch size used for MMD computation.
    if_knn_mnn_graph : bool, optional
        If True, add cross-species MNN edges to the combined adjacency graph.
    if_integrate_within_species : bool, optional
        If True, enable within-species slice integration using additional
        triplet constraints.
    if_return_loss : bool, optional
        If True, return a dictionary recording loss values over epochs.

    Returns
    -------
    adata_species_dict : dict
        The input dictionary with the final embedding stored in
        ``obsm[key_added]`` for each species.
    loss_dict : dict, optional
        Only returned if ``if_return_loss=True``. Contains keys:
        ``'Loss name'``, ``'Epoch'``, ``'Loss value'``, each a list of
        recorded values at every epoch.
    structure_beta : float, optional
        Weight of the per-species geometric structure preservation loss.
        Set to 0 to disable this loss. Default is 0.
    structure_sampling_ratio : float, optional
        Fraction of intra-species edges to sample when computing the structure
        loss. Use a value < 1.0 to reduce memory usage. Default is 1.0 (use all).
    max_structure_edges : int, optional
        Upper bound on the number of intra-species edges used in a single
        epoch for the structure loss. Ignored if the number of available
        intra-species edges is smaller. Default is 10000.

    Returns
    -------
    adata_species_dict : dict
        The input dictionary with the final embedding stored in
        ``obsm[key_added]`` for each species.
    loss_dict : dict, optional
        Only returned if ``if_return_loss=True``.
    """
    # ---------- Reproducibility and GPU cleanup ----------
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.autograd.set_detect_anomaly(True)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    # Allow species-specific pretraining epochs if a dict is provided
    if not isinstance(stagate_epoch, dict):
        stagate_epoch_dict = {k: stagate_epoch for k in adata_species_dict.keys()}
    else:
        stagate_epoch_dict = stagate_epoch

    # ---------- Per-species STAGATE pretraining ----------
    z_dict = {k: 0 for k in adata_species_dict.keys()}
    species_order = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.uns['highly_variable']]
        print(f'For {species_id}, using {len(adata.var_names)} genes for training.')
        # Convert to PyG Data object
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    prune_edge_index=torch.LongTensor(np.array([])),
                    x=torch.FloatTensor(adata.X.todense()))
        data = data.to(pretrain_device)

        if species_order == 0:
            model = STACAME.STACAME(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)
        species_order += 1

        print('Pretrain with STAGATE_multiple...')
        for epoch in tqdm(range(0, stagate_epoch_dict[species_id])):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)
            loss = F.mse_loss(data.x, out)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        print(f'mse loss = {loss.item():.4f}')
        with torch.no_grad():
            z, _ = model(data.x, data.edge_index)
        adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
        z_dict[species_id] = z.cpu().detach()

        if species_order >= len(adata_species_dict.keys()):
            del model, optimizer, data, z, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print('-------------------------------------------------------------------------------')
    print('Train with STACAME...')
    anchor_ind_species = triplet_ind_species_dict['anchor_ind_species']
    positive_ind_species = triplet_ind_species_dict['positive_ind_species']
    negative_ind_species = triplet_ind_species_dict['negative_ind_species']

    # ---------- Build concatenated graph and feature matrix ----------
    k_add = 0
    species_add_dict = {k: None for k in z_dict.keys()}
    for species_id in z_dict.keys():
        species_add_dict[species_id] = int(k_add)
        k_add = int(k_add + adata_species_dict[species_id].n_obs)

    adata = adata_species_dict[list(adata_species_dict.keys())[0]]
    edgeList = adata.uns['edgeList']
    edge_ndarray = np.array([edgeList[0], edgeList[1]])
    S = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if S != 0:
            edge_arr_temp = np.array([edgeList[0], edgeList[1]]) + species_add_dict[species_id]
            edge_ndarray = np.concatenate((edge_ndarray, edge_arr_temp), axis=1)
        else:
            S = S + 1
    edge_ndarray_species = np.array([edge_ndarray_species[0], edge_ndarray_species[1]])

    if if_knn_mnn_graph:
        edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_species), axis=1)

    S = 0
    for species_id, z_input in z_dict.items():
        if S == 0:
            X = z_dict[species_id].cpu().detach().numpy()
        else:
            X = np.concatenate((X, z_dict[species_id].cpu().detach().numpy()), axis=0)
        S = S + 1

    print('Pretrain with STAGATE_multiple...')
    print('Train with cross species STACAME...')
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')
    L1_loss = torch.nn.HuberLoss()
    z = torch.FloatTensor(X)

    species_list = list(adata_species_dict.keys())
    n_species = len(species_list)
    ref_species = species_list[0]
    n_homo_genes = len(adata_species_dict[ref_species].uns['homo_highly_variable'])
    species_specific_n_genes = {
        sp: len(adata_species_dict[sp].uns['species_specific'])
        for sp in species_list
    }
    max_specific_genes = max(species_specific_n_genes.values())
    total_cols = n_homo_genes + max_specific_genes * n_species
    merge_X = None
    mask_matrix = None

    for sp_idx, species_id in enumerate(species_list):
        adata = adata_species_dict[species_id]
        homo_genes = adata.uns['homo_highly_variable']
        x_homo = adata[:, homo_genes].X.todense()
        specific_genes = adata.uns['species_specific']
        x_specific = adata[:, specific_genes].X.todense()
        n_cells = x_homo.shape[0]
        x_current = np.zeros((n_cells, total_cols))
        mask_current = np.zeros((n_cells, total_cols))
        x_current[:, :n_homo_genes] = x_homo
        mask_current[:, :n_homo_genes] = 1
        specific_start_col = n_homo_genes + sp_idx * max_specific_genes
        specific_end_col = specific_start_col + species_specific_n_genes[species_id]
        x_current[:, specific_start_col:specific_end_col] = x_specific
        mask_current[:, specific_start_col:specific_end_col] = 1

        if merge_X is None:
            merge_X = x_current
            mask_matrix = mask_current
        else:
            merge_X = np.concatenate((merge_X, x_current), axis=0)
            mask_matrix = np.concatenate((mask_matrix, mask_current), axis=0)

    if concate_pca_dim is not None:
        adata_X = ad.AnnData(merge_X)
        sc.pp.scale(adata_X)
        sc.tl.pca(adata_X, n_comps=concate_pca_dim)
        merge_X = adata_X.obsm["X_pca"]
    merge_X = torch.FloatTensor(merge_X).to(device)

    # ---------- Joint decoder and optimiser ----------
    model = STACAME.STACAME_Decoder(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_species, weight_decay=weight_decay, foreach=False)

    if if_integrate_within_species:
        anchor_ind_sections = triplet_ind_sections_dict['anchor_ind_sections']
        positive_ind_sections = triplet_ind_sections_dict['positive_ind_sections']
        negative_ind_sections = triplet_ind_sections_dict['negative_ind_sections']
        if if_knn_mnn_graph:
            edge_ndarray_sections = np.array([edge_ndarray_sections[0], edge_ndarray_sections[1]])
            edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_sections), axis=1)
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)
    else:
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)

    # ---------- Precompute species sizes for weighting ----------
    species_n_cells = {sp: adata_species_dict[sp].n_obs for sp in adata_species_dict.keys()}

    # ---------- Extract intra-species edges for manifold preserving loss ----------
    intra_edges = None
    if structure_beta != 0.0:
        total_nodes = z.shape[0]
        node_species = -1 * np.ones(total_nodes, dtype=np.int64)
        for sp_idx, species_id in enumerate(adata_species_dict.keys()):
            start = species_add_dict[species_id]
            n_cells = adata_species_dict[species_id].n_obs
            node_species[start:start+n_cells] = sp_idx

        edge_index_cpu = data.edge_index.cpu().numpy()
        src, dst = edge_index_cpu[0], edge_index_cpu[1]
        mask = (node_species[src] == node_species[dst])
        intra_edges_np = edge_index_cpu[:, mask]
        print(f'[Manifold] Found {intra_edges_np.shape[1]} intra-species edges '
              f'(out of {edge_index_cpu.shape[1]} total).')

        if intra_edges_np.shape[1] > 0:
            intra_edges = torch.LongTensor(intra_edges_np).to(device)

    if if_return_loss:
        loss_dict = {'Loss name': [], 'Epoch': [], 'Loss value': []}

    plot_epoch = n_epochs_species // 3
    scheduler = StepLR(optimizer, step_size=1000, gamma=1)

    # ---------- Helper: Manifold preserving loss (with optional edge weights) ----------
    def manifold_preserving_loss(latent_vectors, ref_vectors, same_species_edges, edge_weights=None):
        """
        Weighted Laplacian loss to preserve local geometry from ref_vectors.
        latent_vectors : (N, d) tensor
        ref_vectors    : (N, d_ref) tensor (data.x, the pretrained embeddings)
        same_species_edges : (2, E) tensor (int64) of edges within same species
        edge_weights   : (E,) tensor, optional per‑edge importance weights
        """
        src, dst = same_species_edges[0], same_species_edges[1]
        d2_ref = torch.sum((ref_vectors[src] - ref_vectors[dst]) ** 2, dim=1)
        sigma = torch.sqrt(torch.median(d2_ref) + 1e-8)
        w = torch.exp(-d2_ref / (2 * sigma ** 2 + 1e-8))
        d2_lat = torch.sum((latent_vectors[src] - latent_vectors[dst]) ** 2, dim=1)
        if edge_weights is None:
            edge_weights = torch.ones_like(w)
        return torch.mean(edge_weights * w * d2_lat)

    # ---------- Cross-species training loop ----------
    for epoch in tqdm(range(0, n_epochs_species)):
        current_seed = random_seed + epoch
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)

        # Update per-species latent arrays in adata
        k_add = 0
        for species_id in z_dict.keys():
            species_add_dict[species_id] = int(k_add)
            adata_species_dict[species_id].obsm['STAGATE'] = z[k_add:int(k_add + adata_species_dict[species_id].n_obs),
                                                             :].cpu().detach().numpy()
            z_dict[species_id] = adata_species_dict[species_id].obsm['STAGATE']
            k_add = int(k_add + adata_species_dict[species_id].n_obs)

        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)

        # 1) MSE reconstruction loss
        mse_loss = F.mse_loss(merge_X, out)

        # 2) Within-species slice triplet loss (optional)
        if if_integrate_within_species:
            anchor_arr = z[anchor_ind_sections,]
            positive_arr = z[positive_ind_sections,]
            negative_arr = z[negative_ind_sections,]
            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        else:
            tri_output = torch.tensor(0.0, device=device)

        # 3) Cross-species triplet loss
        anchor_arr_species = z[anchor_ind_species,]
        positive_arr_species = z[positive_ind_species,]
        negative_arr_species = z[negative_ind_species,]
        triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin_species, p=2, reduction='mean')
        tri_output_species = triplet_loss_species(anchor_arr_species, positive_arr_species, negative_arr_species)

        # 4) MMD loss
        mmd_loss = STACAME.MMDLoss(kernel=STACAME.RBF(device=device), device=device).to(device)
        mmd_loss_sum = 0
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            remain_list = list(
                set(list(range(z.shape[0]))) - set(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))))
            random.seed(epoch)
            ind_1 = random.sample(list(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))), mmd_batch_size)
            ind_2 = random.sample(remain_list, mmd_batch_size)
            mmd_loss_sum = mmd_loss_sum + mmd_loss(z[ind_1,].to(device), z[ind_2,].to(device)).to(device)

        # 5) Manifold preserving loss with species-size weighting
        geom_structure_loss = torch.tensor(0.0, device=device)
        if structure_beta != 0.0 and intra_edges is not None:
            num_intra = intra_edges.shape[1]
            if num_intra > 0:
                # Sampling: apply ratio then cap with max_structure_edges
                n_sample = min(int(num_intra * structure_sampling_ratio), max_structure_edges)
                n_sample = max(1, n_sample)
                if n_sample < num_intra:
                    perm = torch.randperm(num_intra, device=device)[:n_sample]
                    sampled_edges = intra_edges[:, perm]
                else:
                    sampled_edges = intra_edges
                # --- compute species‑size‑based edge weights ---
                src_nodes = sampled_edges[0].cpu().numpy()
                # node_species gives species index for each node
                edge_species = node_species[src_nodes]  # still numpy array
                inv_n = {sp: 1.0 / species_n_cells[sp] for sp in species_n_cells}
                # map species index back to species name using adata_species_dict.keys() order
                species_list = list(adata_species_dict.keys())
                raw_weights = torch.tensor([inv_n[species_list[sp_idx]] for sp_idx in edge_species],
                                           device=device, dtype=torch.float32)
                # normalise so that the average weight equals 1 (keeps loss magnitude stable)
                edge_w = raw_weights * (len(raw_weights) / raw_weights.sum())
                # -------------------------------------------------

                geom_structure_loss = manifold_preserving_loss(z, data.x, sampled_edges, edge_weights=edge_w)

        sampling_num_spe = anchor_arr_species.shape[0]

        # Compose total loss (with manifold loss)
        if if_integrate_within_species:
            loss = (mse_beta * mse_loss
                    + tri_beta * tri_output_species
                    + beta * tri_output
                    + mmd_beta * mmd_loss_sum
                    + structure_beta * geom_structure_loss)
        else:
            loss = (mse_beta * mse_loss
                    + tri_beta * tri_output_species
                    + mmd_beta * mmd_loss_sum
                    + structure_beta * geom_structure_loss)

        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # ---------- Record losses ----------
        if if_return_loss:
            loss_dict['Loss name'].append('Loss sum')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(loss.item())
            loss_dict['Loss name'].append('MSE')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mse_loss.item())
            loss_dict['Loss name'].append('Cross-species triplet')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(tri_output_species.item())
            loss_dict['Loss name'].append('MMD')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mmd_loss_sum.item())
            if structure_beta != 0.0:
                loss_dict['Loss name'].append('ManifoldPreserve')
                loss_dict['Epoch'].append(epoch)
                loss_dict['Loss value'].append(geom_structure_loss.item())

        # ---------- Logging ----------
        if verbose and epoch % 100 == 0:
            cos_sim = cosine_loss(anchor_arr_species, positive_arr_species,
                                  torch.ones(sampling_num_spe).to(device)).item()
            print(f'---------------------------------Epoch {epoch:4d}-----------------------------------')
            print(f'Total loss: {loss.item():.4f}',
                  f'MSE (weighted) : {mse_beta * mse_loss.item():.4f}',
                  f'Cross-species Tri: {tri_beta * tri_output_species.item():.4f}',
                  f'MMD (weighted): {mmd_beta * mmd_loss_sum.item():.4f}')
            if if_integrate_within_species:
                print(f'Cross-slices Tri: {beta * tri_output.item():.4f}',
                      f'Cosine sim (mon.): {cos_sim:.4f}')
                if if_return_loss:
                    loss_dict['Loss name'].append('Cross-slices triplet')
                    loss_dict['Epoch'].append(epoch)
                    loss_dict['Loss value'].append(tri_output.item())
            if structure_beta != 0.0:
                print(f'Manifold (weighted): {structure_beta * geom_structure_loss.item():.4f}')
            current_lr = scheduler.get_last_lr()[0]
            print(f'  Learning rate     : {current_lr:.6f}')

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        # Update the final embedding in adata objects
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[k_add:int(k_add + adata_species_dict[species_id].n_obs),
                                                             :].cpu().detach().numpy()

        # Periodic UMAP visualisation
        if verbose and epoch % plot_epoch == 0 and n_epochs_species - epoch >= plot_epoch:
            if z.shape[0] >= 50000:
                clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
            else:
                clustering_umap(adata_species_dict, key_umap=key_added)

    print('Clustering and UMAP of Cross Species STACAME:')
    if z.shape[0] >= 50000:
        clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
    else:
        clustering_umap(adata_species_dict, key_umap=key_added)

    # ---------- Finalise and cleanup ----------
    with torch.no_grad():
        z_final, out_final = model(data.x, data.edge_index)
        out_np = out_final.cpu().detach().numpy()

    del model, optimizer, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict

'''
def train_STACAME(adata_species_dict,
                  triplet_ind_species_dict,
                  edge_ndarray_species,
                  triplet_ind_sections_dict=None,
                  edge_ndarray_sections=None,
                  hidden_dims=[512, 30],
                  stagate_epoch=500,
                  n_epochs_species=2000,
                  lr=0.001,
                  key_added='STACAME',
                  gradient_clipping=5.,
                  weight_decay=0.0001,
                  lr_wd=0.001,
                  weight_decay_wd=5e-4,
                  margin=1.0,
                  margin_species=1.0,
                  lr_species=0.001,
                  beta=1,
                  verbose=False,
                  random_seed=666,
                  iter_comb=None,
                  knn_neigh=20,
                  device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),
                  pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                  mse_beta=1,
                  tri_beta=1,
                  mmd_beta=1,
                  mmd_batch_size=2048,
                  concate_pca_dim=None,
                  if_knn_mnn_graph=True,
                  if_integrate_within_species=False,
                  if_return_loss=False,
                  structure_beta=0.0,           
                  structure_sampling_ratio=1.0,  
                  max_structure_edges=10000):
    """
   Train a graph-attention autoencoder and align cross-species spots via
    triplet constraints in the latent embedding space.

    The function first pretrains a STAGATE model for each species individually.
    Then, a joint decoder is trained on the concatenated latent representations
    using MSE reconstruction loss, cross-species triplet margin loss, optional
    within-species slice triplet loss, a maximum mean discrepancy (MMD) term
    that encourages mixing of species in the latent space, and an optional
    geometric structure preservation loss that keeps the local adjacency
    structure of each species consistent between original expression and
    latent embeddings.

    Parameters
    ----------
    adata_species_dict : dict of {str: anndata.AnnData}
        Dictionary mapping species names to AnnData objects. Each object must
        have ``.X`` (dense or sparse), ``.obs['batch_name']``, ``.uns['edgeList']``,
        and optionally ``.var['highly_variable']`` or ``.uns['highly_variable']``.
    triplet_ind_species_dict : dict
        Cross-species triplet indices with keys ``'anchor_ind_species'``,
        ``'positive_ind_species'``, ``'negative_ind_species'``. Each value is
        a 1D array of indices into the concatenated spot order.
    edge_ndarray_species : np.ndarray
        Cross-species mutual nearest neighbour (MNN) graph edges, shape (2, n_edges).
    triplet_ind_sections_dict : dict, optional
        Within-species slice-level triplet indices with analogous keys
        ``'anchor_ind_sections'``, ``'positive_ind_sections'``,
        ``'negative_ind_sections'``. Required if ``if_integrate_within_species``
        is True.
    edge_ndarray_sections : np.ndarray, optional
        Within-species slice MNN graph edges, shape (2, n_edges). Required if
        ``if_integrate_within_species`` and ``if_knn_mnn_graph`` are both True.
    hidden_dims : list of int, optional
        Hidden layer dimensions for the encoder and decoder. The first element
        is the dimension after the input layer, the second is the bottleneck
        dimension.
    stagate_epoch : int or dict, optional
        Number of pretraining epochs per species. If an int, it is used for all
        species; if a dict, it must map species names to epoch counts.
    n_epochs_species : int, optional
        Number of epochs for the cross-species joint training phase.
    lr : float, optional
        Learning rate for the per-species STAGATE pretraining.
    key_added : str, optional
        Key under which the final latent embedding is stored in
        ``adata_species_dict[species].obsm``.
    gradient_clipping : float, optional
        Maximum gradient norm for clipping during joint training.
    weight_decay : float, optional
        Weight decay for the Adam optimiser during pretraining.
    lr_wd : float, optional
        (Unused in this light version; kept for compatibility.)
    weight_decay_wd : float, optional
        (Unused in this light version; kept for compatibility.)
    margin : float, optional
        Margin for the within-species slice triplet loss.
    margin_species : float, optional
        Margin for the cross-species triplet loss.
    lr_species : float, optional
        Learning rate for the joint cross-species training phase.
    beta : float, optional
        Weight of the within-species slice triplet loss in the total loss
        (only active when ``if_integrate_within_species==True``).
    verbose : bool, optional
        If True, log training progress every 100 epochs.
    random_seed : int, optional
        Random seed for reproducibility.
    iter_comb : tuple or None, optional
        (Reserved for multi-batch ordering; unused in the light version.)
    knn_neigh : int, optional
        Number of nearest neighbours used when constructing MNN graphs.
    device : torch.device, optional
        Device for the cross-species training.
    pretrain_device : torch.device, optional
        Device for the per-species pretraining.
    mse_beta : float, optional
        Weight of the MSE reconstruction loss.
    tri_beta : float, optional
        Weight of the cross-species triplet loss.
    mmd_beta : float, optional
        Weight of the MMD loss.
    mmd_batch_size : int, optional
        Batch size used for MMD computation.
    if_knn_mnn_graph : bool, optional
        If True, add cross-species MNN edges to the combined adjacency graph.
    if_integrate_within_species : bool, optional
        If True, enable within-species slice integration using additional
        triplet constraints.
    if_return_loss : bool, optional
        If True, return a dictionary recording loss values over epochs.

    Returns
    -------
    adata_species_dict : dict
        The input dictionary with the final embedding stored in
        ``obsm[key_added]`` for each species.
    loss_dict : dict, optional
        Only returned if ``if_return_loss=True``. Contains keys:
        ``'Loss name'``, ``'Epoch'``, ``'Loss value'``, each a list of
        recorded values at every epoch.
    structure_beta : float, optional
        Weight of the per-species geometric structure preservation loss.
        Set to 0 to disable this loss. Default is 0.
    structure_sampling_ratio : float, optional
        Fraction of intra-species edges to sample when computing the structure
        loss. Use a value < 1.0 to reduce memory usage. Default is 1.0 (use all).
    max_structure_edges : int, optional
        Upper bound on the number of intra-species edges used in a single
        epoch for the structure loss. Ignored if the number of available
        intra-species edges is smaller. Default is 10000.

    Returns
    -------
    adata_species_dict : dict
        The input dictionary with the final embedding stored in
        ``obsm[key_added]`` for each species.
    loss_dict : dict, optional
        Only returned if ``if_return_loss=True``.
    """

    # ---------- Reproducibility and GPU cleanup ----------
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.autograd.set_detect_anomaly(True)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    # Allow species-specific pretraining epochs if a dict is provided
    if not isinstance(stagate_epoch, dict):
        stagate_epoch_dict = {k: stagate_epoch for k in adata_species_dict.keys()}
    else:
        stagate_epoch_dict = stagate_epoch

    # ---------- Per-species STAGATE pretraining ----------
    z_dict = {k: 0 for k in adata_species_dict.keys()}
    species_order = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.uns['highly_variable']]
        print(f'For {species_id}, using {len(adata.var_names)} genes for training.')
        # Convert to PyG Data object
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    prune_edge_index=torch.LongTensor(np.array([])),
                    x=torch.FloatTensor(adata.X.todense()))
        data = data.to(pretrain_device)

        if species_order == 0:
            model = STACAME.STACAME(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)
        species_order += 1

        print('Pretrain with STAGATE_multiple...')
        for epoch in tqdm(range(0, stagate_epoch_dict[species_id])):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)
            loss = F.mse_loss(data.x, out)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        print(f'mse loss = {loss.item():.4f}')
        with torch.no_grad():
            z, _ = model(data.x, data.edge_index)
        adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
        z_dict[species_id] = z.cpu().detach()

        if species_order >= len(adata_species_dict.keys()):
            del model, optimizer, data, z, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print('-------------------------------------------------------------------------------')
    print('Train with STACAME...')
    anchor_ind_species = triplet_ind_species_dict['anchor_ind_species']
    positive_ind_species = triplet_ind_species_dict['positive_ind_species']
    negative_ind_species = triplet_ind_species_dict['negative_ind_species']

    # ---------- Build concatenated graph and feature matrix ----------
    k_add = 0
    species_add_dict = {k: None for k in z_dict.keys()}
    for species_id in z_dict.keys():
        species_add_dict[species_id] = int(k_add)
        k_add = int(k_add + adata_species_dict[species_id].n_obs)

    adata = adata_species_dict[list(adata_species_dict.keys())[0]]
    edgeList = adata.uns['edgeList']
    edge_ndarray = np.array([edgeList[0], edgeList[1]])
    S = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if S != 0:
            edge_arr_temp = np.array([edgeList[0], edgeList[1]]) + species_add_dict[species_id]
            edge_ndarray = np.concatenate((edge_ndarray, edge_arr_temp), axis=1)
        else:
            S = S + 1
    edge_ndarray_species = np.array([edge_ndarray_species[0], edge_ndarray_species[1]])

    if if_knn_mnn_graph:
        edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_species), axis=1)

    S = 0
    for species_id, z_input in z_dict.items():
        if S == 0:
            X = z_dict[species_id].cpu().detach().numpy()
        else:
            X = np.concatenate((X, z_dict[species_id].cpu().detach().numpy()), axis=0)
        S = S + 1

    print('Pretrain with STAGATE_multiple...')
    print('Train with cross species STACAME...')
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')
    L1_loss = torch.nn.HuberLoss()
    z = torch.FloatTensor(X)

    species_list = list(adata_species_dict.keys())
    n_species = len(species_list)
    ref_species = species_list[0]
    n_homo_genes = len(adata_species_dict[ref_species].uns['homo_highly_variable'])
    species_specific_n_genes = {
        sp: len(adata_species_dict[sp].uns['species_specific'])
        for sp in species_list
    }
    max_specific_genes = max(species_specific_n_genes.values())
    total_cols = n_homo_genes + max_specific_genes * n_species
    merge_X = None
    mask_matrix = None

    for sp_idx, species_id in enumerate(species_list):
        adata = adata_species_dict[species_id]
        homo_genes = adata.uns['homo_highly_variable']
        x_homo = adata[:, homo_genes].X.todense()
        specific_genes = adata.uns['species_specific']
        x_specific = adata[:, specific_genes].X.todense()
        n_cells = x_homo.shape[0]
        x_current = np.zeros((n_cells, total_cols))
        mask_current = np.zeros((n_cells, total_cols))
        x_current[:, :n_homo_genes] = x_homo
        mask_current[:, :n_homo_genes] = 1
        specific_start_col = n_homo_genes + sp_idx * max_specific_genes
        specific_end_col = specific_start_col + species_specific_n_genes[species_id]
        x_current[:, specific_start_col:specific_end_col] = x_specific
        mask_current[:, specific_start_col:specific_end_col] = 1

        if merge_X is None:
            merge_X = x_current
            mask_matrix = mask_current
        else:
            merge_X = np.concatenate((merge_X, x_current), axis=0)
            mask_matrix = np.concatenate((mask_matrix, mask_current), axis=0)

    if concate_pca_dim is not None:
        adata_X = ad.AnnData(merge_X)
        sc.pp.scale(adata_X)
        sc.tl.pca(adata_X, n_comps=concate_pca_dim)
        merge_X = adata_X.obsm["X_pca"]
    merge_X = torch.FloatTensor(merge_X).to(device)

    # ---------- Joint decoder and optimiser ----------
    model = STACAME.STACAME_Decoder(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_species, weight_decay=weight_decay, foreach=False)

    if if_integrate_within_species:
        anchor_ind_sections = triplet_ind_sections_dict['anchor_ind_sections']
        positive_ind_sections = triplet_ind_sections_dict['positive_ind_sections']
        negative_ind_sections = triplet_ind_sections_dict['negative_ind_sections']
        if if_knn_mnn_graph:
            edge_ndarray_sections = np.array([edge_ndarray_sections[0], edge_ndarray_sections[1]])
            edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_sections), axis=1)
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)
    else:
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)

    # ---------- Extract intra-species edges for manifold preserving loss ----------
    intra_edges = None
    if structure_beta != 0.0:
        total_nodes = z.shape[0]
        node_species = -1 * np.ones(total_nodes, dtype=np.int64)
        for sp_idx, species_id in enumerate(adata_species_dict.keys()):
            start = species_add_dict[species_id]
            n_cells = adata_species_dict[species_id].n_obs
            node_species[start:start+n_cells] = sp_idx

        edge_index_cpu = data.edge_index.cpu().numpy()
        src, dst = edge_index_cpu[0], edge_index_cpu[1]
        mask = (node_species[src] == node_species[dst])
        intra_edges_np = edge_index_cpu[:, mask]
        print(f'[Manifold] Found {intra_edges_np.shape[1]} intra-species edges '
              f'(out of {edge_index_cpu.shape[1]} total).')

        if intra_edges_np.shape[1] > 0:
            intra_edges = torch.LongTensor(intra_edges_np).to(device)

    if if_return_loss:
        loss_dict = {'Loss name': [], 'Epoch': [], 'Loss value': []}

    plot_epoch = n_epochs_species // 3
    scheduler = StepLR(optimizer, step_size=1000, gamma=1)

    # ---------- Helper: Manifold preserving loss ----------
    def manifold_preserving_loss(latent_vectors, ref_vectors, same_species_edges):
        """
        Weighted Laplacian loss to preserve local geometry from ref_vectors.
        latent_vectors : (N, d) tensor
        ref_vectors    : (N, d_ref) tensor (data.x, the pretrained embeddings)
        same_species_edges : (2, E) tensor (int64) of edges within same species
        """
        src, dst = same_species_edges[0], same_species_edges[1]
        d2_ref = torch.sum((ref_vectors[src] - ref_vectors[dst]) ** 2, dim=1)
        # Adaptive bandwidth: sqrt of median distance (stabilised)
        sigma = torch.sqrt(torch.median(d2_ref) + 1e-8)
        w = torch.exp(-d2_ref / (2 * sigma ** 2 + 1e-8))
        d2_lat = torch.sum((latent_vectors[src] - latent_vectors[dst]) ** 2, dim=1)
        return torch.mean(w * d2_lat)

    # ---------- Cross-species training loop ----------
    for epoch in tqdm(range(0, n_epochs_species)):
        current_seed = random_seed + epoch
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)

        # Update per-species latent arrays in adata
        k_add = 0
        for species_id in z_dict.keys():
            species_add_dict[species_id] = int(k_add)
            adata_species_dict[species_id].obsm['STAGATE'] = z[k_add:int(k_add + adata_species_dict[species_id].n_obs),
                                                             :].cpu().detach().numpy()
            z_dict[species_id] = adata_species_dict[species_id].obsm['STAGATE']
            k_add = int(k_add + adata_species_dict[species_id].n_obs)

        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)

        # 1) MSE reconstruction loss
        mse_loss = F.mse_loss(merge_X, out)

        # 2) Within-species slice triplet loss (optional)
        if if_integrate_within_species:
            anchor_arr = z[anchor_ind_sections,]
            positive_arr = z[positive_ind_sections,]
            negative_arr = z[negative_ind_sections,]
            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        else:
            tri_output = torch.tensor(0.0, device=device)

        # 3) Cross-species triplet loss
        anchor_arr_species = z[anchor_ind_species,]
        positive_arr_species = z[positive_ind_species,]
        negative_arr_species = z[negative_ind_species,]
        triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin_species, p=2, reduction='mean')
        tri_output_species = triplet_loss_species(anchor_arr_species, positive_arr_species, negative_arr_species)

        # 4) MMD loss
        mmd_loss = STACAME.MMDLoss(kernel=STACAME.RBF(device=device), device=device).to(device)
        mmd_loss_sum = 0
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            remain_list = list(
                set(list(range(z.shape[0]))) - set(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))))
            random.seed(epoch)
            ind_1 = random.sample(list(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))), mmd_batch_size)
            ind_2 = random.sample(remain_list, mmd_batch_size)
            mmd_loss_sum = mmd_loss_sum + mmd_loss(z[ind_1,].to(device), z[ind_2,].to(device)).to(device)

        # 5) Manifold preserving loss (replaces old Gaussian affinity)
        geom_structure_loss = torch.tensor(0.0, device=device)
        if structure_beta != 0.0 and intra_edges is not None:
            num_intra = intra_edges.shape[1]
            if num_intra > 0:
                # Sampling: apply ratio then cap with max_structure_edges
                n_sample = min(int(num_intra * structure_sampling_ratio), max_structure_edges)
                n_sample = max(1, n_sample)  # ensure at least 1
                if n_sample < num_intra:
                    perm = torch.randperm(num_intra, device=device)[:n_sample]
                    sampled_edges = intra_edges[:, perm]
                else:
                    sampled_edges = intra_edges
                # reference is data.x (pretrained embeddings, fixed)
                geom_structure_loss = manifold_preserving_loss(z, data.x, sampled_edges)

        sampling_num_spe = anchor_arr_species.shape[0]

        # Compose total loss (with manifold loss)
        if if_integrate_within_species:
            loss = (mse_beta * mse_loss
                    + tri_beta * tri_output_species
                    + beta * tri_output
                    + mmd_beta * mmd_loss_sum
                    + structure_beta * geom_structure_loss)
        else:
            loss = (mse_beta * mse_loss
                    + tri_beta * tri_output_species
                    + mmd_beta * mmd_loss_sum
                    + structure_beta * geom_structure_loss)

        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # ---------- Record losses ----------
        if if_return_loss:
            loss_dict['Loss name'].append('Loss sum')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(loss.item())
            loss_dict['Loss name'].append('MSE')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mse_loss.item())
            loss_dict['Loss name'].append('Cross-species triplet')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(tri_output_species.item())
            loss_dict['Loss name'].append('MMD')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mmd_loss_sum.item())
            if structure_beta != 0.0:
                loss_dict['Loss name'].append('ManifoldPreserve')
                loss_dict['Epoch'].append(epoch)
                loss_dict['Loss value'].append(geom_structure_loss.item())

        # ---------- Logging ----------
        if verbose and epoch % 100 == 0:
            cos_sim = cosine_loss(anchor_arr_species, positive_arr_species,
                                  torch.ones(sampling_num_spe).to(device)).item()
            print(f'---------------------------------Epoch {epoch:4d}-----------------------------------')
            print(f'Total loss: {loss.item():.4f}',
                  f'MSE (weighted) : {mse_beta * mse_loss.item():.4f}',
                  f'Cross-species Tri: {tri_beta * tri_output_species.item():.4f}',
                  f'MMD (weighted): {mmd_beta * mmd_loss_sum.item():.4f}')
            if if_integrate_within_species:
                print(f'Cross-slices Tri: {beta * tri_output.item():.4f}',
                      f'Cosine sim (mon.): {cos_sim:.4f}')
                if if_return_loss:
                    loss_dict['Loss name'].append('Cross-slices triplet')
                    loss_dict['Epoch'].append(epoch)
                    loss_dict['Loss value'].append(tri_output.item())
            if structure_beta != 0.0:
                print(f'Manifold (weighted): {structure_beta * geom_structure_loss.item():.4f}')
            current_lr = scheduler.get_last_lr()[0]
            print(f'  Learning rate     : {current_lr:.6f}')

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        # Update the final embedding in adata objects
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[k_add:int(k_add + adata_species_dict[species_id].n_obs),
                                                             :].cpu().detach().numpy()

        # Periodic UMAP visualisation
        if verbose and epoch % plot_epoch == 0 and n_epochs_species - epoch >= plot_epoch:
            if z.shape[0] >= 50000:
                clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
            else:
                clustering_umap(adata_species_dict, key_umap=key_added)

    print('Clustering and UMAP of Cross Species STACAME:')
    if z.shape[0] >= 50000:
        clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
    else:
        clustering_umap(adata_species_dict, key_umap=key_added)

    # ---------- Finalise and cleanup ----------
    with torch.no_grad():
        z_final, out_final = model(data.x, data.edge_index)
        out_np = out_final.cpu().detach().numpy()

    del model, optimizer, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict
'''

'''
## Standard version of STACAME with GAN loss and auxiliary model
def train_STACAME_GAN(adata_species_dict,
                      triplet_ind_species_dict,
                      edge_ndarray_species,
                      triplet_ind_sections_dict=None,
                      edge_ndarray_sections=None,
                      hidden_dims=[256, 20],
                      stagate_epoch=500,
                      n_epochs_species=2000,
                      lr=0.001,
                      key_added='STACAME',
                      gradient_clipping=5.,
                      weight_decay=0.0001,
                      lr_wd=0.001,
                      weight_decay_wd=5e-4,
                      margin=1.0,
                      margin_species=1.0,
                      lr_species=0.001,
                      beta=1,
                      verbose=False,
                      random_seed=666,
                      iter_comb=None,
                      knn_neigh=10,
                      device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                      pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                      mse_beta=1,
                      tri_beta=5,
                      mmd_beta=1,
                      gan_beta=0,
                      gan_epoch=3,
                      ot_beta=0,
                      mmd_batch_size=1024,
                      if_knn_mnn_graph=False,
                      if_integrate_within_species=False,
                      if_return_loss=False,
                      adata_whole=None,
                      concate_pca_dim=64,
                      if_use_light_model=False,
                      structure_beta=0.0,            # weight for the manifold preserving loss
                      structure_sampling_ratio=1.0,  # fraction of intra-species edges to sample
                      max_structure_edges=10000):    # max number of edges per epoch for this loss
    """
    Train a graph-attention auto-encoder with GAN-based domain confusion
    and an auxiliary model for cross-species spatial transcriptomics integration.

    Compared to the light version, this standard version additionally includes:
      - A multi-class discriminator (domain classifier) that adversarially
        encourages species mixing in the latent space (GAN loss).
      - An auxiliary model sharing the same architecture, trained jointly with
        its own discriminator to provide a complementary alignment signal.
      - Optional unbalanced optimal transport (OT) loss between batches.
      - (New) Manifold preserving loss that keeps the local neighbourhood structure
        of each species consistent between the pretrained embeddings and the
        current latent representations of both the primary and auxiliary models.

    The training proceeds in two stages:
      1. Per-species STAGATE pretraining (with optional within-species triplet updates).
      2. Joint cross-species training with MSE, triplet, MMD, GAN, OT and manifold losses.

    Parameters
    ----------
    adata_species_dict : dict of {str: anndata.AnnData}
        Dictionary mapping species names to AnnData objects. Each object must
        contain ``.X``, ``.obs['batch_name']``, ``.uns['edgeList']``, and
        optionally ``.var['highly_variable']`` or ``.uns['highly_variable']``.
    triplet_ind_species_dict : dict
        Cross-species triplet indices with keys ``'anchor_ind_species'``,
        ``'positive_ind_species'``, ``'negative_ind_species'``.
    edge_ndarray_species : np.ndarray
        Cross-species MNN graph edge array, shape (2, n_edges).
    triplet_ind_sections_dict : dict, optional
        Within-species slice-level triplet indices with analogous keys.
    edge_ndarray_sections : np.ndarray, optional
        Within-species slice MNN graph edges.
    hidden_dims : list
        Hidden layer dimensions: [encoder_output_dim, bottleneck_dim].
    stagate_epoch : int or dict
        Number of pretraining epochs per species. If an int, used for all;
        if a dict, maps species names to epoch counts.
    n_epochs_species : int
        Total number of epochs for the cross-species joint training phase.
    lr : float
        Learning rate for the per-species pretraining.
    key_added : str
        Key under which the final latent embedding is stored in
        ``adata_species_dict[species].obsm``.
    gradient_clipping : float
        Max gradient norm for clipping during joint training.
    weight_decay : float
        Weight decay for the pretraining optimizer.
    lr_wd : float
        (Unused; kept for compatibility.)
    weight_decay_wd : float
        (Unused; kept for compatibility.)
    margin : float
        Triplet loss margin for the within-species slice constraint.
    margin_species : float
        Triplet loss margin for the cross-species constraint.
    lr_species : float
        Learning rate for the cross-species joint training phase.
    beta : float
        Weight of the within-species slice triplet loss in the total loss.
    verbose : bool
        If True, print a detailed loss breakdown every 100 epochs.
    random_seed : int
        Random seed for reproducibility.
    iter_comb : tuple or None
        Order of slice integration for within-species MNN construction.
    knn_neigh : int
        Number of nearest neighbours for MNN graph construction.
    device : torch.device
        Device for the cross-species training.
    pretrain_device : torch.device
        Device for the per-species pretraining.
    mse_beta : float
        Weight of the MSE reconstruction loss.
    tri_beta : float
        Weight of the combined triplet loss (auxiliary + cross-species).
    mmd_beta : float
        Weight of the MMD loss.
    gan_beta : float
        Weight of the GAN domain confusion loss.
    gan_epoch : int
        Number of discriminator update steps per generator step.
    ot_beta : float
        Weight of the optional optimal transport loss (0 to disable).
    mmd_batch_size : int
        Batch size used for MMD computation.
    if_knn_mnn_graph : bool
        Whether to add cross-species MNN edges to the graph.
    if_integrate_within_species : bool
        Whether to enable within-species slice integration.
    if_return_loss : bool
        If True, return a dictionary recording loss values over epochs.
    adata_whole : anndata.AnnData
        Concatenated AnnData object across all species for MNN search
        and domain labels during training.
    concate_pca_dim : int
        PCA dimension to which merged gene expression is reduced before
        feeding to the decoder.
    structure_beta : float, optional
        Weight of the manifold preserving loss. Set to 0 to disable.
    structure_sampling_ratio : float, optional
        Fraction of intra-species edges to sample for the structure loss.
    max_structure_edges : int, optional
        Maximum number of intra-species edges used per epoch for the
        manifold loss.

    Returns
    -------
    adata_species_dict : dict
        Updated dictionary with final embedding in ``obsm[key_added]``.
    loss_dict : dict, optional
        Only returned when ``if_return_loss=True``. Keys: ``'Loss name'``,
        ``'Epoch'``, ``'Loss value'``.
    """
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---------- Set all random seeds for reproducibility ----------
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.autograd.set_detect_anomaly(True)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    # Allow per-species pretraining epochs via dict
    if not isinstance(stagate_epoch, dict):
        stagate_epoch_dict = {k: stagate_epoch for k in adata_species_dict.keys()}
    else:
        stagate_epoch_dict = stagate_epoch

    # ---------- Per-species STAGATE pretraining ----------
    z_dict = {k: 0 for k in adata_species_dict.keys()}
    species_order = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.uns['highly_variable']]
        print(f'For {species_id}, using {len(adata.var_names)} genes for training.')

        # Build PyG Data object for the species
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    prune_edge_index=torch.LongTensor(np.array([])),
                    x=torch.FloatTensor(adata.X.todense()))
        data = data.to(pretrain_device)

        if species_order == 0:
            # Instantiate STAGATE model (shared across species)
            model = STACAME.STACAME(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)

        species_order += 1
        print('Pretrain with STAGATE_multiple...')
        for epoch in tqdm(range(0, stagate_epoch_dict[species_id])):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)

            # If within-species integration is on, periodically refresh triplets
            if if_integrate_within_species:
                if epoch % 10 == 0 and epoch >= 500:
                    if verbose:
                        print('Update spot triplets at epoch ' + str(epoch))
                    adata.obsm['STAGATE'] = z.cpu().detach().numpy()
                    mnn_dict = create_dictionary_mnn(adata, use_rep='STAGATE', batch_name='batch_name', k=knn_neigh,
                                                     iter_comb=iter_comb, verbose=0)
                    anchor_ind = []
                    positive_ind = []
                    negative_ind = []
                    for batch_pair in mnn_dict.keys():
                        batchname_list = adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                        cellname_by_batch_dict = dict()
                        for batch_id in range(len(section_ids)):
                            cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                                adata.obs['batch_name'] == section_ids[batch_id]].values
                        anchor_list = []
                        positive_list = []
                        negative_list = []
                        for anchor in mnn_dict[batch_pair].keys():
                            anchor_list.append(anchor)
                            positive_spot = mnn_dict[batch_pair][anchor][0]
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(
                                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
                        batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                        anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                        positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                        negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

                    anchor_arr = z[anchor_ind,]
                    positive_arr = z[positive_ind,]
                    negative_arr = z[negative_ind,]
                    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                    tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
                    loss = F.mse_loss(data.x.to(pretrain_device), out) + beta * tri_output
                else:
                    loss = F.mse_loss(data.x.to(pretrain_device), out)
            else:
                loss = F.mse_loss(data.x.to(pretrain_device), out)

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
        print(f'Pretrain mse loss = {loss.item():.4f}')

        # Extract and store the pretrained latent representation
        with torch.no_grad():
            z, _ = model(data.x, data.edge_index)
        adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
        z_dict[species_id] = z.cpu().detach()

        # Clean up after the last species
        if species_order >= len(adata_species_dict.keys()):
            del model, optimizer, data, z, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    print('-------------------------------------------------------------------------------')
    print('Train with STACAME...')
    anchor_ind_species = triplet_ind_species_dict['anchor_ind_species']
    positive_ind_species = triplet_ind_species_dict['positive_ind_species']
    negative_ind_species = triplet_ind_species_dict['negative_ind_species']

    # ---------- Build the concatenated graph ----------
    k_add = 0
    species_add_dict = {k: None for k in z_dict.keys()}
    for species_id in z_dict.keys():
        species_add_dict[species_id] = int(k_add)
        k_add = int(k_add + adata_species_dict[species_id].n_obs)

    # Start with the within-species edges of the first species
    adata = adata_species_dict[list(adata_species_dict.keys())[0]]
    edgeList = adata.uns['edgeList']
    edge_ndarray = np.array([edgeList[0], edgeList[1]])
    S = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if S != 0:
            edge_arr_temp = np.array([edgeList[0], edgeList[1]]) + species_add_dict[species_id]
            edge_ndarray = np.concatenate((edge_ndarray, edge_arr_temp), axis=1)
        else:
            S = S + 1
    edge_ndarray_species = np.array([edge_ndarray_species[0], edge_ndarray_species[1]])

    # Optionally include cross-species MNN edges
    if if_knn_mnn_graph:
        edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_species), axis=1)

    # Concatenate pretrained latent vectors for the primary model input
    S = 0
    for species_id, z_input in z_dict.items():
        if S == 0:
            X = z_dict[species_id].cpu().detach().numpy()
        else:
            X = np.concatenate((X, z_dict[species_id].cpu().detach().numpy()), axis=0)
        S = S + 1

    print('Pretrain with STAGATE_multiple...')
    print('Train with cross species STACAME...')
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    z = torch.FloatTensor(X)

    # ---------- Merge raw gene expression and reduce dimensionality with PCA ----------
    species_list = list(adata_species_dict.keys())
    n_species = len(species_list)
    ref_species = species_list[0]
    n_homo_genes = len(adata_species_dict[ref_species].uns['homo_highly_variable'])
    species_specific_n_genes = {
        sp: len(adata_species_dict[sp].uns['species_specific'])
        for sp in species_list
    }
    max_specific_genes = max(species_specific_n_genes.values())
    total_cols = n_homo_genes + max_specific_genes * n_species
    merge_X = None
    mask_matrix = None
    for sp_idx, species_id in enumerate(species_list):
        adata = adata_species_dict[species_id]
        homo_genes = adata.uns['homo_highly_variable']
        x_homo = adata[:, homo_genes].X.todense()
        specific_genes = adata.uns['species_specific']
        x_specific = adata[:, specific_genes].X.todense()
        n_cells = x_homo.shape[0]
        x_current = np.zeros((n_cells, total_cols))
        mask_current = np.zeros((n_cells, total_cols))
        x_current[:, :n_homo_genes] = x_homo
        mask_current[:, :n_homo_genes] = 1
        specific_start_col = n_homo_genes + sp_idx * max_specific_genes
        specific_end_col = specific_start_col + species_specific_n_genes[species_id]
        x_current[:, specific_start_col:specific_end_col] = x_specific
        mask_current[:, specific_start_col:specific_end_col] = 1
        if merge_X is None:
            merge_X = x_current
            mask_matrix = mask_current
        else:
            merge_X = np.concatenate((merge_X, x_current), axis=0)
            mask_matrix = np.concatenate((mask_matrix, mask_current), axis=0)

    if concate_pca_dim is not None:
        adata_X = ad.AnnData(merge_X)
        sc.pp.scale(adata_X)
        sc.tl.pca(adata_X, n_comps=concate_pca_dim)
        merge_X = adata_X.obsm["X_pca"]
    merge_X = torch.FloatTensor(merge_X).to(device)

    # ---------- Build models and discriminators ----------
    species_ids = list(adata_species_dict.keys())
    # Primary decoder (receives latent code from the main graph)
    if if_use_light_model:
        model = STACAME.STACAMEDecoder_light(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    else:
        model = STACAME.STACAME_Decoder(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)

    # Auxiliary model uses raw PCA features as input
    auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'])
    auxiliary_model = STACAME.STACAME(hidden_dims=[auxiliary_X.shape[1], hidden_dims[0]//4, hidden_dims[1]]).to(device)

    # Joint optimizer for both primary decoder and auxiliary model
    optimizer = torch.optim.Adam(list(model.parameters()) + list(auxiliary_model.parameters()), lr=lr_species,
                                 weight_decay=weight_decay, foreach=False)

    # Domain discriminators for adversarial training
    auxiliary_D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    auxiliary_optimizer_D = torch.optim.Adam(list(auxiliary_D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)
    auxiliary_D_Z.train()

    D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    optimizer_D = torch.optim.Adam(list(D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)
    D_Z.train()

    # Create ground-truth domain labels for GAN
    species_list_gt = []
    for species_id, adata in adata_species_dict.items():
        species_list_gt = species_list_gt + [species_id] * adata.n_obs
    true_dom = torch.LongTensor(pd.Series(species_list_gt).astype('category').cat.codes.values).to(device)

    # Auxiliary data object
    auxiliary_data = Data(edge_index=torch.LongTensor(edge_ndarray),
                          prune_edge_index=torch.LongTensor(np.array([])), x=auxiliary_X)
    auxiliary_data = auxiliary_data.to(device)

    # ---------- Build joint data object for primary path ----------
    if if_integrate_within_species:
        anchor_ind_sections = triplet_ind_sections_dict['anchor_ind_sections']
        positive_ind_sections = triplet_ind_sections_dict['positive_ind_sections']
        negative_ind_sections = triplet_ind_sections_dict['negative_ind_sections']
        if if_knn_mnn_graph:
            edge_ndarray_sections = np.array([edge_ndarray_sections[0], edge_ndarray_sections[1]])
            edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_sections), axis=1)
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)
    else:
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)

    # ---- Extract intra-species edges for manifold preserving loss ----
    intra_edges = None
    if structure_beta != 0.0:
        total_nodes = z.shape[0]
        node_species = -1 * np.ones(total_nodes, dtype=np.int64)
        for sp_idx, species_id in enumerate(adata_species_dict.keys()):
            start = species_add_dict[species_id]
            n_cells = adata_species_dict[species_id].n_obs
            node_species[start:start+n_cells] = sp_idx

        edge_index_cpu = data.edge_index.cpu().numpy()
        src, dst = edge_index_cpu[0], edge_index_cpu[1]
        mask = (node_species[src] == node_species[dst])
        intra_edges_np = edge_index_cpu[:, mask]
        print(f'[Manifold] Found {intra_edges_np.shape[1]} intra-species edges '
              f'(out of {edge_index_cpu.shape[1]} total).')
        if intra_edges_np.shape[1] > 0:
            intra_edges = torch.LongTensor(intra_edges_np).to(device)

    if if_return_loss:
        loss_dict = {'Loss name': [], 'Epoch': [], 'Loss value': []}

    plot_epoch = n_epochs_species // 3

    # ---------- Helper: manifold preserving loss ----------
    def manifold_preserving_loss(latent_vectors, ref_vectors, same_species_edges):
        """
        Weighted Laplacian loss to preserve local geometry from ref_vectors.
        latent_vectors : (N, d) tensor (current latent representation)
        ref_vectors    : (N, d_ref) tensor (pretrained embeddings, fixed)
        same_species_edges : (2, E) tensor of intra-species edges
        """
        src, dst = same_species_edges[0], same_species_edges[1]
        d2_ref = torch.sum((ref_vectors[src] - ref_vectors[dst]) ** 2, dim=1)
        # Adaptive bandwidth: sqrt of median squared distance
        sigma = torch.sqrt(torch.median(d2_ref) + 1e-8)
        w = torch.exp(-d2_ref / (2 * sigma ** 2 + 1e-8))
        d2_lat = torch.sum((latent_vectors[src] - latent_vectors[dst]) ** 2, dim=1)
        return torch.mean(w * d2_lat)

    # ---------- Cross-species joint training ----------
    for epoch in tqdm(range(0, n_epochs_species)):
        # Update per-species latent arrays for tracking
        k_add = 0
        for species_id in z_dict.keys():
            species_add_dict[species_id] = int(k_add)
            adata_species_dict[species_id].obsm['STAGATE'] = z[
                k_add:int(k_add + adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
            z_dict[species_id] = adata_species_dict[species_id].obsm['STAGATE']
            k_add = int(k_add + adata_species_dict[species_id].n_obs)
        if epoch == 0:
            adata_whole.obsm['auxiliary'] = z.cpu().detach().numpy()

        model.train()
        auxiliary_model.train()
        optimizer.zero_grad()
        auxiliary_z, auxiliary_out = auxiliary_model(auxiliary_data.x, auxiliary_data.edge_index)
        z, out = model(data.x, data.edge_index)

        # 1) MSE reconstruction loss
        mse_loss = F.mse_loss(merge_X, out) + F.mse_loss(auxiliary_data.x, auxiliary_out)

        # 2) Within-species slice triplet loss (optional)
        if if_integrate_within_species:
            anchor_arr = z[anchor_ind_sections,]
            positive_arr = z[positive_ind_sections,]
            negative_arr = z[negative_ind_sections,]
            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        else:
            tri_output = torch.tensor(0.0, device=device)

        # 3) Cross-species triplet loss
        anchor_arr_species = z[anchor_ind_species,]
        positive_arr_species = z[positive_ind_species,]
        negative_arr_species = z[negative_ind_species,]
        triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin_species, p=2, reduction='mean')
        tri_output_species = triplet_loss_species(anchor_arr_species, positive_arr_species, negative_arr_species)

        # 4) MMD loss (main + auxiliary)
        mmd_loss = STACAME.MMDLoss(kernel=STACAME.RBF(device=device), device=device).to(device)
        mmd_loss_sum = 0
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            remain_list = list(
                set(list(range(z.shape[0]))) - set(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))))
            ind_1 = random.sample(list(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))), mmd_batch_size)
            ind_2 = random.sample(remain_list, mmd_batch_size)
            mmd_loss_sum = mmd_loss_sum + mmd_loss(z[ind_1,].to(device), z[ind_2,].to(device)).to(device)
            mmd_loss_sum = mmd_loss_sum + mmd_loss(auxiliary_z[ind_1,].to(device), auxiliary_z[ind_2,].to(device)).to(device)

        # 5) Optional optimal transport loss
        loss_ot = torch.tensor(0.0, device=device)
        if ot_beta != 0:
            z_A = z[ind_1,].to(device)
            z_B = z[ind_2,].to(device)
            x_A = auxiliary_X[ind_1,].to(device)
            x_B = auxiliary_X[ind_2,].to(device)
            c_cross = pairwise_correlation_distance(x_A.detach(), x_B.detach()).to(device)
            T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=device)
            z_dist = torch.mean((z_A.view(mmd_batch_size, 1, -1) - z_B.view(1, mmd_batch_size, -1)) ** 2, dim=2)
            loss_ot = torch.sum(T * z_dist) / torch.sum(T)

        # 6) Periodically update cross-species triplets via MNN
        sampling_num_spe = anchor_arr_species.shape[0]
        if epoch % 100 == 0:
            mnn_dict = create_dictionary_mnn(adata_whole, use_rep='auxiliary', batch_name='species_id', k=knn_neigh,
                                             iter_comb=iter_comb, verbose=0)
            anchor_ind = []
            positive_ind = []
            negative_ind = []
            for batch_pair in mnn_dict.keys():
                batchname_list = adata_whole.obs['species_id'][mnn_dict[batch_pair].keys()]
                cellname_by_batch_dict = dict()
                for batch_id in range(len(species_ids)):
                    cellname_by_batch_dict[species_ids[batch_id]] = adata_whole.obs_names[
                        adata_whole.obs['species_id'] == species_ids[batch_id]].values
                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in mnn_dict[batch_pair].keys():
                    anchor_list.append(anchor)
                    positive_spot = mnn_dict[batch_pair][anchor][0]
                    positive_list.append(positive_spot)
                    section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                    negative_list.append(
                        cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
                batch_as_dict = dict(zip(list(adata_whole.obs_names), range(0, adata_whole.shape[0])))
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

        # 7) Auxiliary triplet loss (based on updated MNN)
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_auxiliary = triplet_loss(z[anchor_ind,], z[positive_ind,], z[negative_ind,]) + triplet_loss(
            auxiliary_z[anchor_ind,], auxiliary_z[positive_ind,], auxiliary_z[negative_ind,])

        # 8) GAN domain confusion
        if gan_beta != 0:
            for _ in range(gan_epoch):
                optimizer_D.zero_grad()
                logits_D = D_Z(z)
                loss_D = F.cross_entropy(logits_D, true_dom)
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
            for _ in range(gan_epoch):
                auxiliary_optimizer_D.zero_grad()
                auxiliary_logits_D = auxiliary_D_Z(auxiliary_z)
                auxiliary_loss_D = F.cross_entropy(auxiliary_logits_D, true_dom)
                auxiliary_loss_D.backward(retain_graph=True)
                auxiliary_optimizer_D.step()

        loss_G_GAN = -F.cross_entropy(D_Z(z), true_dom) - F.cross_entropy(auxiliary_D_Z(auxiliary_z), true_dom)

        # 9) Manifold preserving loss (new)
        geom_structure_loss = torch.tensor(0.0, device=device)
        if structure_beta != 0.0 and intra_edges is not None:
            num_intra = intra_edges.shape[1]
            if num_intra > 0:
                n_sample = min(int(num_intra * structure_sampling_ratio), max_structure_edges)
                n_sample = max(1, n_sample)
                if n_sample < num_intra:
                    perm = torch.randperm(num_intra, device=device)[:n_sample]
                    sampled_edges = intra_edges[:, perm]
                else:
                    sampled_edges = intra_edges
                # Compute loss for both primary and auxiliary latent spaces
                loss_z = manifold_preserving_loss(z, data.x, sampled_edges)
                loss_aux = manifold_preserving_loss(auxiliary_z, data.x, sampled_edges)
                geom_structure_loss = loss_z + loss_aux

        # ---- Compose total loss ----
        if if_integrate_within_species:
            loss = (mse_beta * mse_loss
                    + tri_beta * (tri_auxiliary + 0.1 * tri_output_species)
                    + beta * tri_output
                    + mmd_beta * mmd_loss_sum
                    + gan_beta * loss_G_GAN
                    + ot_beta * loss_ot
                    + structure_beta * geom_structure_loss)
        else:
            loss = (mse_beta * mse_loss
                    + tri_beta * (tri_auxiliary + 0.1 * tri_output_species)
                    + mmd_beta * mmd_loss_sum
                    + gan_beta * loss_G_GAN
                    + ot_beta * loss_ot
                    + structure_beta * geom_structure_loss)

        loss.backward(retain_graph=True)
        optimizer.step()

        # ---------- Record losses ----------
        if if_return_loss:
            loss_dict['Loss name'].append('Loss sum')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(loss.item())
            loss_dict['Loss name'].append('MSE')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mse_loss.item())
            loss_dict['Loss name'].append('Cross-species triplet')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(tri_output_species.item())
            loss_dict['Loss name'].append('MMD')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mmd_loss_sum.item())
            if ot_beta > 0:
                loss_dict['Loss name'].append('OT')
                loss_dict['Epoch'].append(epoch)
                loss_dict['Loss value'].append(loss_ot.item())
            if structure_beta != 0.0:
                loss_dict['Loss name'].append('ManifoldPreserve')
                loss_dict['Epoch'].append(epoch)
                loss_dict['Loss value'].append(geom_structure_loss.item())

        # ---------- Verbose logging ----------
        if verbose and epoch % 100 == 0:
            cos_sim = cosine_loss(anchor_arr_species, positive_arr_species,
                                  torch.ones(sampling_num_spe).to(device)).item()
            print(f'---------------------------------Epoch {epoch:4d}-----------------------------------')
            print(f'Total loss: {loss.item():.4f}|',
                  f'MSE (weighted): {mse_beta * mse_loss.item():.4f}|',
                  f'Cross-species Tri: {tri_beta * tri_output_species.item():.4f}|',
                  f'Auxiliary Tri: {tri_beta * tri_auxiliary.item():.4f}|',
                  f'MMD (weighted): {mmd_beta * mmd_loss_sum.item():.4f}|',
                  f'GAN (weighted): {gan_beta * loss_G_GAN.item():.4f}')
            if ot_beta > 0:
                print(f'OT (weighted): {ot_beta * loss_ot.item():.4f}')
            if structure_beta != 0.0:
                print(f'Manifold (weighted): {structure_beta * geom_structure_loss.item():.4f}')
            if if_integrate_within_species:
                print(f'Cross-slices Tri: {beta * tri_output.item():.4f}')
                print(f'Cosine sim (mon.): {cos_sim:.4f}')
                if if_return_loss:
                    loss_dict['Loss name'].append('Cross-slices triplet')
                    loss_dict['Epoch'].append(epoch)
                    loss_dict['Loss value'].append(tri_output.item())

        # Update final embedding in AnnData objects
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[
                k_add:int(k_add + adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
        adata_whole.obsm['auxiliary'] = auxiliary_z.cpu().detach().numpy()

        # Periodic UMAP for visual inspection (downsampled if >50000 spots)
        if verbose and epoch % plot_epoch == 0 and n_epochs_species - epoch >= plot_epoch:
            if z.shape[0] >= 50000:
                clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
            else:
                clustering_umap(adata_species_dict, key_umap=key_added)

    print('Clustering and UMAP of Cross Species STACAME:')
    if z.shape[0] >= 50000:
        clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
    else:
        clustering_umap(adata_species_dict, key_umap=key_added)

    # ---------- Cleanup ----------
    del model, auxiliary_model, optimizer, D_Z, auxiliary_D_Z, data, auxiliary_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict

'''


def train_STACAME_GAN(adata_species_dict,
                      triplet_ind_species_dict,
                      edge_ndarray_species,
                      triplet_ind_sections_dict=None,
                      edge_ndarray_sections=None,
                      hidden_dims=[256, 20],
                      stagate_epoch=500,
                      n_epochs_species=2000,
                      lr=0.001,
                      key_added='STACAME',
                      gradient_clipping=5.,
                      weight_decay=0.0001,
                      lr_wd=0.001,
                      weight_decay_wd=5e-4,
                      margin=1.0,
                      margin_species=1.0,
                      lr_species=0.001,
                      beta=1,
                      verbose=False,
                      random_seed=666,
                      iter_comb=None,
                      knn_neigh=10,
                      device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                      pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                      mse_beta=1,
                      tri_beta=5,
                      mmd_beta=1,
                      gan_beta=0,
                      gan_epoch=3,
                      ot_beta=0,
                      mmd_batch_size=1024,
                      if_knn_mnn_graph=False,
                      if_integrate_within_species=False,
                      if_return_loss=False,
                      adata_whole=None,
                      concate_pca_dim=64,
                      if_use_light_model=False,
                      structure_beta=0.0,
                      structure_sampling_ratio=1.0,
                      max_structure_edges=10000):
    """
     Train a graph-attention auto-encoder with GAN-based domain confusion
    and an auxiliary model for cross-species spatial transcriptomics integration.

    Compared to the light version, this standard version additionally includes:
      - A multi-class discriminator (domain classifier) that adversarially
        encourages species mixing in the latent space (GAN loss).
      - An auxiliary model sharing the same architecture, trained jointly with
        its own discriminator to provide a complementary alignment signal.
      - Optional unbalanced optimal transport (OT) loss between batches.
      - (New) Manifold preserving loss that keeps the local neighbourhood structure
        of each species consistent between the pretrained embeddings and the
        current latent representations of both the primary and auxiliary models.

    The training proceeds in two stages:
      1. Per-species STAGATE pretraining (with optional within-species triplet updates).
      2. Joint cross-species training with MSE, triplet, MMD, GAN, OT and manifold losses.

    Parameters
    ----------
    adata_species_dict : dict of {str: anndata.AnnData}
        Dictionary mapping species names to AnnData objects. Each object must
        contain ``.X``, ``.obs['batch_name']``, ``.uns['edgeList']``, and
        optionally ``.var['highly_variable']`` or ``.uns['highly_variable']``.
    triplet_ind_species_dict : dict
        Cross-species triplet indices with keys ``'anchor_ind_species'``,
        ``'positive_ind_species'``, ``'negative_ind_species'``.
    edge_ndarray_species : np.ndarray
        Cross-species MNN graph edge array, shape (2, n_edges).
    triplet_ind_sections_dict : dict, optional
        Within-species slice-level triplet indices with analogous keys.
    edge_ndarray_sections : np.ndarray, optional
        Within-species slice MNN graph edges.
    hidden_dims : list
        Hidden layer dimensions: [encoder_output_dim, bottleneck_dim].
    stagate_epoch : int or dict
        Number of pretraining epochs per species. If an int, used for all;
        if a dict, maps species names to epoch counts.
    n_epochs_species : int
        Total number of epochs for the cross-species joint training phase.
    lr : float
        Learning rate for the per-species pretraining.
    key_added : str
        Key under which the final latent embedding is stored in
        ``adata_species_dict[species].obsm``.
    gradient_clipping : float
        Max gradient norm for clipping during joint training.
    weight_decay : float
        Weight decay for the pretraining optimizer.
    lr_wd : float
        (Unused; kept for compatibility.)
    weight_decay_wd : float
        (Unused; kept for compatibility.)
    margin : float
        Triplet loss margin for the within-species slice constraint.
    margin_species : float
        Triplet loss margin for the cross-species constraint.
    lr_species : float
        Learning rate for the cross-species joint training phase.
    beta : float
        Weight of the within-species slice triplet loss in the total loss.
    verbose : bool
        If True, print a detailed loss breakdown every 100 epochs.
    random_seed : int
        Random seed for reproducibility.
    iter_comb : tuple or None
        Order of slice integration for within-species MNN construction.
    knn_neigh : int
        Number of nearest neighbours for MNN graph construction.
    device : torch.device
        Device for the cross-species training.
    pretrain_device : torch.device
        Device for the per-species pretraining.
    mse_beta : float
        Weight of the MSE reconstruction loss.
    tri_beta : float
        Weight of the combined triplet loss (auxiliary + cross-species).
    mmd_beta : float
        Weight of the MMD loss.
    gan_beta : float
        Weight of the GAN domain confusion loss.
    gan_epoch : int
        Number of discriminator update steps per generator step.
    ot_beta : float
        Weight of the optional optimal transport loss (0 to disable).
    mmd_batch_size : int
        Batch size used for MMD computation.
    if_knn_mnn_graph : bool
        Whether to add cross-species MNN edges to the graph.
    if_integrate_within_species : bool
        Whether to enable within-species slice integration.
    if_return_loss : bool
        If True, return a dictionary recording loss values over epochs.
    adata_whole : anndata.AnnData
        Concatenated AnnData object across all species for MNN search
        and domain labels during training.
    concate_pca_dim : int
        PCA dimension to which merged gene expression is reduced before
        feeding to the decoder.
    structure_beta : float, optional
        Weight of the manifold preserving loss. Set to 0 to disable.
    structure_sampling_ratio : float, optional
        Fraction of intra-species edges to sample for the structure loss.
    max_structure_edges : int, optional
        Maximum number of intra-species edges used per epoch for the
        manifold loss.

    Returns
    -------
    adata_species_dict : dict
        Updated dictionary with final embedding in ``obsm[key_added]``.
    loss_dict : dict, optional
        Only returned when ``if_return_loss=True``. Keys: ``'Loss name'``,
        ``'Epoch'``, ``'Loss value'``.
    """
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---------- Set all random seeds for reproducibility ----------
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.autograd.set_detect_anomaly(True)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    if not isinstance(stagate_epoch, dict):
        stagate_epoch_dict = {k: stagate_epoch for k in adata_species_dict.keys()}
    else:
        stagate_epoch_dict = stagate_epoch

    # ---------- Per-species STAGATE pretraining ----------
    z_dict = {k: 0 for k in adata_species_dict.keys()}
    species_order = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.uns['highly_variable']]
        print(f'For {species_id}, using {len(adata.var_names)} genes for training.')

        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    prune_edge_index=torch.LongTensor(np.array([])),
                    x=torch.FloatTensor(adata.X.todense()))
        data = data.to(pretrain_device)

        if species_order == 0:
            model = STACAME.STACAME(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)

        species_order += 1
        print('Pretrain with STAGATE_multiple...')
        for epoch in tqdm(range(0, stagate_epoch_dict[species_id])):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)

            if if_integrate_within_species:
                if epoch % 10 == 0 and epoch >= 500:
                    if verbose:
                        print('Update spot triplets at epoch ' + str(epoch))
                    adata.obsm['STAGATE'] = z.cpu().detach().numpy()
                    mnn_dict = create_dictionary_mnn(adata, use_rep='STAGATE', batch_name='batch_name', k=knn_neigh,
                                                     iter_comb=iter_comb, verbose=0)
                    anchor_ind = []
                    positive_ind = []
                    negative_ind = []
                    for batch_pair in mnn_dict.keys():
                        batchname_list = adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                        cellname_by_batch_dict = dict()
                        for batch_id in range(len(section_ids)):
                            cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                                adata.obs['batch_name'] == section_ids[batch_id]].values
                        anchor_list = []
                        positive_list = []
                        negative_list = []
                        for anchor in mnn_dict[batch_pair].keys():
                            anchor_list.append(anchor)
                            positive_spot = mnn_dict[batch_pair][anchor][0]
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(
                                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
                        batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                        anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                        positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                        negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

                    anchor_arr = z[anchor_ind,]
                    positive_arr = z[positive_ind,]
                    negative_arr = z[negative_ind,]
                    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                    tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
                    loss = F.mse_loss(data.x.to(pretrain_device), out) + beta * tri_output
                else:
                    loss = F.mse_loss(data.x.to(pretrain_device), out)
            else:
                loss = F.mse_loss(data.x.to(pretrain_device), out)

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
        print(f'Pretrain mse loss = {loss.item():.4f}')

        with torch.no_grad():
            z, _ = model(data.x, data.edge_index)
        adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
        z_dict[species_id] = z.cpu().detach()

        if species_order >= len(adata_species_dict.keys()):
            del model, optimizer, data, z, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    print('-------------------------------------------------------------------------------')
    print('Train with STACAME...')
    anchor_ind_species = triplet_ind_species_dict['anchor_ind_species']
    positive_ind_species = triplet_ind_species_dict['positive_ind_species']
    negative_ind_species = triplet_ind_species_dict['negative_ind_species']

    # ---------- Build the concatenated graph ----------
    k_add = 0
    species_add_dict = {k: None for k in z_dict.keys()}
    for species_id in z_dict.keys():
        species_add_dict[species_id] = int(k_add)
        k_add = int(k_add + adata_species_dict[species_id].n_obs)

    adata = adata_species_dict[list(adata_species_dict.keys())[0]]
    edgeList = adata.uns['edgeList']
    edge_ndarray = np.array([edgeList[0], edgeList[1]])
    S = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if S != 0:
            edge_arr_temp = np.array([edgeList[0], edgeList[1]]) + species_add_dict[species_id]
            edge_ndarray = np.concatenate((edge_ndarray, edge_arr_temp), axis=1)
        else:
            S = S + 1
    edge_ndarray_species = np.array([edge_ndarray_species[0], edge_ndarray_species[1]])

    if if_knn_mnn_graph:
        edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_species), axis=1)

    S = 0
    for species_id, z_input in z_dict.items():
        if S == 0:
            X = z_dict[species_id].cpu().detach().numpy()
        else:
            X = np.concatenate((X, z_dict[species_id].cpu().detach().numpy()), axis=0)
        S = S + 1

    print('Pretrain with STAGATE_multiple...')
    print('Train with cross species STACAME...')
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    z = torch.FloatTensor(X)

    # ---------- Merge raw gene expression and reduce dimensionality with PCA ----------
    species_list = list(adata_species_dict.keys())
    n_species = len(species_list)
    ref_species = species_list[0]
    n_homo_genes = len(adata_species_dict[ref_species].uns['homo_highly_variable'])
    species_specific_n_genes = {
        sp: len(adata_species_dict[sp].uns['species_specific'])
        for sp in species_list
    }
    max_specific_genes = max(species_specific_n_genes.values())
    total_cols = n_homo_genes + max_specific_genes * n_species
    merge_X = None
    mask_matrix = None
    for sp_idx, species_id in enumerate(species_list):
        adata = adata_species_dict[species_id]
        homo_genes = adata.uns['homo_highly_variable']
        x_homo = adata[:, homo_genes].X.todense()
        specific_genes = adata.uns['species_specific']
        x_specific = adata[:, specific_genes].X.todense()
        n_cells = x_homo.shape[0]
        x_current = np.zeros((n_cells, total_cols))
        mask_current = np.zeros((n_cells, total_cols))
        x_current[:, :n_homo_genes] = x_homo
        mask_current[:, :n_homo_genes] = 1
        specific_start_col = n_homo_genes + sp_idx * max_specific_genes
        specific_end_col = specific_start_col + species_specific_n_genes[species_id]
        x_current[:, specific_start_col:specific_end_col] = x_specific
        mask_current[:, specific_start_col:specific_end_col] = 1
        if merge_X is None:
            merge_X = x_current
            mask_matrix = mask_current
        else:
            merge_X = np.concatenate((merge_X, x_current), axis=0)
            mask_matrix = np.concatenate((mask_matrix, mask_current), axis=0)

    if concate_pca_dim is not None:
        adata_X = ad.AnnData(merge_X)
        sc.pp.scale(adata_X)
        sc.tl.pca(adata_X, n_comps=concate_pca_dim)
        merge_X = adata_X.obsm["X_pca"]
    merge_X = torch.FloatTensor(merge_X).to(device)

    # ---------- Build models and discriminators ----------
    species_ids = list(adata_species_dict.keys())
    if if_use_light_model:
        model = STACAME.STACAMEDecoder_light(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    else:
        model = STACAME.STACAME_Decoder(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)

    auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'])
    auxiliary_model = STACAME.STACAME(hidden_dims=[auxiliary_X.shape[1], hidden_dims[0]//4, hidden_dims[1]]).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(auxiliary_model.parameters()), lr=lr_species,
                                 weight_decay=weight_decay, foreach=False)

    # --- Precompute species sizes for manifold loss weighting ---
    species_n_cells = {sp: adata_species_dict[sp].n_obs for sp in adata_species_dict.keys()}

    auxiliary_D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    auxiliary_optimizer_D = torch.optim.Adam(list(auxiliary_D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)
    auxiliary_D_Z.train()

    D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    optimizer_D = torch.optim.Adam(list(D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)
    D_Z.train()

    species_list_gt = []
    for species_id, adata in adata_species_dict.items():
        species_list_gt = species_list_gt + [species_id] * adata.n_obs
    true_dom = torch.LongTensor(pd.Series(species_list_gt).astype('category').cat.codes.values).to(device)

    auxiliary_data = Data(edge_index=torch.LongTensor(edge_ndarray),
                          prune_edge_index=torch.LongTensor(np.array([])), x=auxiliary_X)
    auxiliary_data = auxiliary_data.to(device)

    if if_integrate_within_species:
        anchor_ind_sections = triplet_ind_sections_dict['anchor_ind_sections']
        positive_ind_sections = triplet_ind_sections_dict['positive_ind_sections']
        negative_ind_sections = triplet_ind_sections_dict['negative_ind_sections']
        if if_knn_mnn_graph:
            edge_ndarray_sections = np.array([edge_ndarray_sections[0], edge_ndarray_sections[1]])
            edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_sections), axis=1)
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)
    else:
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)

    # ---- Extract intra-species edges for manifold preserving loss ----
    intra_edges = None
    if structure_beta != 0.0:
        total_nodes = z.shape[0]
        node_species = -1 * np.ones(total_nodes, dtype=np.int64)
        for sp_idx, species_id in enumerate(adata_species_dict.keys()):
            start = species_add_dict[species_id]
            n_cells = adata_species_dict[species_id].n_obs
            node_species[start:start+n_cells] = sp_idx

        edge_index_cpu = data.edge_index.cpu().numpy()
        src, dst = edge_index_cpu[0], edge_index_cpu[1]
        mask = (node_species[src] == node_species[dst])
        intra_edges_np = edge_index_cpu[:, mask]
        print(f'[Manifold] Found {intra_edges_np.shape[1]} intra-species edges '
              f'(out of {edge_index_cpu.shape[1]} total).')
        if intra_edges_np.shape[1] > 0:
            intra_edges = torch.LongTensor(intra_edges_np).to(device)

    if if_return_loss:
        loss_dict = {'Loss name': [], 'Epoch': [], 'Loss value': []}

    plot_epoch = n_epochs_species // 3

    # ---------- Helper: manifold preserving loss (now with edge weights) ----------
    def manifold_preserving_loss(latent_vectors, ref_vectors, same_species_edges, edge_weights=None):
        """
        Weighted Laplacian loss to preserve local geometry from ref_vectors.
        latent_vectors : (N, d) tensor
        ref_vectors    : (N, d_ref) tensor
        same_species_edges : (2, E) tensor
        edge_weights   : (E,) tensor, optional
        """
        src, dst = same_species_edges[0], same_species_edges[1]
        d2_ref = torch.sum((ref_vectors[src] - ref_vectors[dst]) ** 2, dim=1)
        sigma = torch.sqrt(torch.median(d2_ref) + 1e-8)
        w = torch.exp(-d2_ref / (2 * sigma ** 2 + 1e-8))
        d2_lat = torch.sum((latent_vectors[src] - latent_vectors[dst]) ** 2, dim=1)
        if edge_weights is None:
            edge_weights = torch.ones_like(w)
        return torch.mean(edge_weights * w * d2_lat)

    # ---------- Cross-species joint training ----------
    for epoch in tqdm(range(0, n_epochs_species)):
        k_add = 0
        for species_id in z_dict.keys():
            species_add_dict[species_id] = int(k_add)
            adata_species_dict[species_id].obsm['STAGATE'] = z[
                k_add:int(k_add + adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
            z_dict[species_id] = adata_species_dict[species_id].obsm['STAGATE']
            k_add = int(k_add + adata_species_dict[species_id].n_obs)
        if epoch == 0:
            adata_whole.obsm['auxiliary'] = z.cpu().detach().numpy()

        model.train()
        auxiliary_model.train()
        optimizer.zero_grad()
        auxiliary_z, auxiliary_out = auxiliary_model(auxiliary_data.x, auxiliary_data.edge_index)
        z, out = model(data.x, data.edge_index)

        # 1) MSE
        mse_loss = F.mse_loss(merge_X, out) + F.mse_loss(auxiliary_data.x, auxiliary_out)

        # 2) Within-species slice triplet
        if if_integrate_within_species:
            anchor_arr = z[anchor_ind_sections,]
            positive_arr = z[positive_ind_sections,]
            negative_arr = z[negative_ind_sections,]
            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        else:
            tri_output = torch.tensor(0.0, device=device)

        # 3) Cross-species triplet
        anchor_arr_species = z[anchor_ind_species,]
        positive_arr_species = z[positive_ind_species,]
        negative_arr_species = z[negative_ind_species,]
        triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin_species, p=2, reduction='mean')
        tri_output_species = triplet_loss_species(anchor_arr_species, positive_arr_species, negative_arr_species)

        # 4) MMD
        mmd_loss = STACAME.MMDLoss(kernel=STACAME.RBF(device=device), device=device).to(device)
        mmd_loss_sum = 0
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            remain_list = list(
                set(list(range(z.shape[0]))) - set(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))))
            ind_1 = random.sample(list(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))), mmd_batch_size)
            ind_2 = random.sample(remain_list, mmd_batch_size)
            mmd_loss_sum = mmd_loss_sum + mmd_loss(z[ind_1,].to(device), z[ind_2,].to(device)).to(device)
            mmd_loss_sum = mmd_loss_sum + mmd_loss(auxiliary_z[ind_1,].to(device), auxiliary_z[ind_2,].to(device)).to(device)

        # 5) OT
        loss_ot = torch.tensor(0.0, device=device)
        if ot_beta != 0:
            z_A = z[ind_1,].to(device)
            z_B = z[ind_2,].to(device)
            x_A = auxiliary_X[ind_1,].to(device)
            x_B = auxiliary_X[ind_2,].to(device)
            c_cross = pairwise_correlation_distance(x_A.detach(), x_B.detach()).to(device)
            T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=device)
            z_dist = torch.mean((z_A.view(mmd_batch_size, 1, -1) - z_B.view(1, mmd_batch_size, -1)) ** 2, dim=2)
            loss_ot = torch.sum(T * z_dist) / torch.sum(T)

        # 6) Refresh cross-species triplets
        sampling_num_spe = anchor_arr_species.shape[0]
        if epoch % 100 == 0:
            mnn_dict = create_dictionary_mnn(adata_whole, use_rep='auxiliary', batch_name='species_id', k=knn_neigh,
                                             iter_comb=iter_comb, verbose=0)
            anchor_ind = []
            positive_ind = []
            negative_ind = []
            for batch_pair in mnn_dict.keys():
                batchname_list = adata_whole.obs['species_id'][mnn_dict[batch_pair].keys()]
                cellname_by_batch_dict = dict()
                for batch_id in range(len(species_ids)):
                    cellname_by_batch_dict[species_ids[batch_id]] = adata_whole.obs_names[
                        adata_whole.obs['species_id'] == species_ids[batch_id]].values
                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in mnn_dict[batch_pair].keys():
                    anchor_list.append(anchor)
                    positive_spot = mnn_dict[batch_pair][anchor][0]
                    positive_list.append(positive_spot)
                    section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                    negative_list.append(
                        cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
                batch_as_dict = dict(zip(list(adata_whole.obs_names), range(0, adata_whole.shape[0])))
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

        # 7) Auxiliary triplet
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_auxiliary = triplet_loss(z[anchor_ind,], z[positive_ind,], z[negative_ind,]) + triplet_loss(
            auxiliary_z[anchor_ind,], auxiliary_z[positive_ind,], auxiliary_z[negative_ind,])

        # 8) GAN
        if gan_beta != 0:
            for _ in range(gan_epoch):
                optimizer_D.zero_grad()
                logits_D = D_Z(z)
                loss_D = F.cross_entropy(logits_D, true_dom)
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
            for _ in range(gan_epoch):
                auxiliary_optimizer_D.zero_grad()
                auxiliary_logits_D = auxiliary_D_Z(auxiliary_z)
                auxiliary_loss_D = F.cross_entropy(auxiliary_logits_D, true_dom)
                auxiliary_loss_D.backward(retain_graph=True)
                auxiliary_optimizer_D.step()

        loss_G_GAN = -F.cross_entropy(D_Z(z), true_dom) - F.cross_entropy(auxiliary_D_Z(auxiliary_z), true_dom)

        # 9) Manifold preserving loss with species-size weighting
        geom_structure_loss = torch.tensor(0.0, device=device)
        if structure_beta != 0.0 and intra_edges is not None:
            num_intra = intra_edges.shape[1]
            if num_intra > 0:
                n_sample = min(int(num_intra * structure_sampling_ratio), max_structure_edges)
                n_sample = max(1, n_sample)
                if n_sample < num_intra:
                    perm = torch.randperm(num_intra, device=device)[:n_sample]
                    sampled_edges = intra_edges[:, perm]
                else:
                    sampled_edges = intra_edges

                # ---- compute per‑edge weights based on species size ----
                src_nodes = sampled_edges[0].cpu().numpy()
                edge_species = node_species[src_nodes]          # species index for each source node
                inv_n = {sp: 1.0 / species_n_cells[sp] for sp in species_n_cells}
                species_list_local = list(adata_species_dict.keys())
                raw_weights = torch.tensor([inv_n[species_list_local[sp_idx]] for sp_idx in edge_species],
                                           device=device, dtype=torch.float32)
                edge_w = raw_weights * (len(raw_weights) / raw_weights.sum())  # normalise mean to 1
                # -------------------------------------------------------

                loss_z = manifold_preserving_loss(z, data.x, sampled_edges, edge_weights=edge_w)
                loss_aux = manifold_preserving_loss(auxiliary_z, data.x, sampled_edges, edge_weights=edge_w)
                geom_structure_loss = loss_z + loss_aux

        # ---- Total loss ----
        if if_integrate_within_species:
            loss = (mse_beta * mse_loss
                    + tri_beta * (tri_auxiliary + 0.1 * tri_output_species)
                    + beta * tri_output
                    + mmd_beta * mmd_loss_sum
                    + gan_beta * loss_G_GAN
                    + ot_beta * loss_ot
                    + structure_beta * geom_structure_loss)
        else:
            loss = (mse_beta * mse_loss
                    + tri_beta * (tri_auxiliary + 0.1 * tri_output_species)
                    + mmd_beta * mmd_loss_sum
                    + gan_beta * loss_G_GAN
                    + ot_beta * loss_ot
                    + structure_beta * geom_structure_loss)

        loss.backward(retain_graph=True)
        optimizer.step()

        # ---------- Record losses ----------
        if if_return_loss:
            loss_dict['Loss name'].append('Loss sum')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(loss.item())
            loss_dict['Loss name'].append('MSE')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mse_loss.item())
            loss_dict['Loss name'].append('Cross-species triplet')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(tri_output_species.item())
            loss_dict['Loss name'].append('MMD')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mmd_loss_sum.item())
            if ot_beta > 0:
                loss_dict['Loss name'].append('OT')
                loss_dict['Epoch'].append(epoch)
                loss_dict['Loss value'].append(loss_ot.item())
            if structure_beta != 0.0:
                loss_dict['Loss name'].append('ManifoldPreserve')
                loss_dict['Epoch'].append(epoch)
                loss_dict['Loss value'].append(geom_structure_loss.item())

        # ---------- Logging ----------
        if verbose and epoch % 100 == 0:
            cos_sim = cosine_loss(anchor_arr_species, positive_arr_species,
                                  torch.ones(sampling_num_spe).to(device)).item()
            print(f'---------------------------------Epoch {epoch:4d}-----------------------------------')
            print(f'Total loss: {loss.item():.4f}|',
                  f'MSE (weighted): {mse_beta * mse_loss.item():.4f}|',
                  f'Cross-species Tri: {tri_beta * tri_output_species.item():.4f}|',
                  f'Auxiliary Tri: {tri_beta * tri_auxiliary.item():.4f}|',
                  f'MMD (weighted): {mmd_beta * mmd_loss_sum.item():.4f}|',
                  f'GAN (weighted): {gan_beta * loss_G_GAN.item():.4f}')
            if ot_beta > 0:
                print(f'OT (weighted): {ot_beta * loss_ot.item():.4f}')
            if structure_beta != 0.0:
                print(f'Manifold (weighted): {structure_beta * geom_structure_loss.item():.4f}')
            if if_integrate_within_species:
                print(f'Cross-slices Tri: {beta * tri_output.item():.4f}')
                print(f'Cosine sim (mon.): {cos_sim:.4f}')
                if if_return_loss:
                    loss_dict['Loss name'].append('Cross-slices triplet')
                    loss_dict['Epoch'].append(epoch)
                    loss_dict['Loss value'].append(tri_output.item())

        # Update final embedding
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[
                k_add:int(k_add + adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
        adata_whole.obsm['auxiliary'] = auxiliary_z.cpu().detach().numpy()

        if verbose and epoch % plot_epoch == 0 and n_epochs_species - epoch >= plot_epoch:
            if z.shape[0] >= 50000:
                clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
            else:
                clustering_umap(adata_species_dict, key_umap=key_added)

    print('Clustering and UMAP of Cross Species STACAME:')
    if z.shape[0] >= 50000:
        clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
    else:
        clustering_umap(adata_species_dict, key_umap=key_added)

    # ---------- Cleanup ----------
    del model, auxiliary_model, optimizer, D_Z, auxiliary_D_Z, data, auxiliary_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict



def train_STACAME_subgraph_auxiliary(adata_species_dict,
                                     triplet_ind_species_dict,
                                     edge_ndarray_species,
                                     triplet_ind_sections_dict=None,
                                     edge_ndarray_sections=None,
                                     hidden_dims=[256, 30],
                                     stagate_epoch=500,
                                     n_epochs_species=2000,
                                     lr=0.001,
                                     key_added='STACAME',
                                     gradient_clipping=5.,
                                     weight_decay=0.0001,
                                     lr_wd=0.001,
                                     weight_decay_wd=5e-4,
                                     margin=1.0,
                                     margin_species=1.0,
                                     lr_species=0.001,
                                     beta=1,
                                     verbose=False,
                                     random_seed=666,
                                     iter_comb=None,
                                     knn_neigh=10,
                                     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                     pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                     mse_beta=1,
                                     tri_beta=1,
                                     mmd_beta=1,
                                     gan_beta=0,
                                     gan_epoch=3,
                                     ot_beta=0,
                                     mmd_batch_size=2048,
                                     if_knn_mnn_graph=False,
                                     if_integrate_within_species=False,
                                     if_return_loss=False,
                                     if_batch_pretrain=False,
                                     batch_size_dict={'Mouse': 10000, 'Marmoset': 10000, 'Macaque': 10000},
                                     batch_size=2048,
                                     concate_pca_dim=None,
                                     umap_downsampling_rate=0.1,
                                     adata_whole=None,
                                     if_use_light_model=False,
                                     structure_beta=0.0,
                                     structure_sampling_ratio=1.0):
    """
    Subgraph‑based training of STACAME with GAN loss and an auxiliary model.

    This function performs cross‑species spatial transcriptomics integration
    using a graph‑attention auto‑encoder. It first pretrains a STAGATE model
    per species (with optional mini‑batch training), then jointly trains a
    primary decoder and an auxiliary model on concatenated data containing
    multiple species. The training leverages subgraph sampling to handle
    large‑scale datasets and incorporates several loss components:
    - MSE reconstruction loss (primary + auxiliary)
    - Cross‑species triplet loss
    - Within‑species slice triplet loss (optional)
    - Maximum Mean Discrepancy (MMD) loss
    - GAN‑based domain confusion loss (optional)
    - Unbalanced optimal transport (OT) loss (optional)

    Parameters
    ----------
    adata_species_dict : dict
        Dictionary mapping species names to AnnData objects.
    triplet_ind_species_dict : dict
        Cross‑species triplet indices (keys: 'anchor_ind_species',
        'positive_ind_species', 'negative_ind_species').
    edge_ndarray_species : np.ndarray
        Cross‑species MNN edge array of shape (2, n_edges).
    triplet_ind_sections_dict : dict, optional
        Within‑species slice triplet indices.
    edge_ndarray_sections : np.ndarray, optional
        Within‑species slice MNN edge array.
    hidden_dims : list
        Encoder/decoder hidden dimensions [output_dim, bottleneck_dim].
    stagate_epoch : int or dict
        Pretraining epochs per species. If int, applies to all.
    n_epochs_species : int
        Number of joint cross‑species training epochs.
    lr : float
        Learning rate for pretraining.
    key_added : str
        Key in ``obsm`` where final embeddings are stored.
    gradient_clipping : float
        Max gradient norm for clipping.
    weight_decay : float
        Weight decay for pretraining optimizer.
    lr_wd : float
        (Unused; kept for compatibility.)
    weight_decay_wd : float
        (Unused; kept for compatibility.)
    margin : float
        Triplet margin for within‑species constraints.
    margin_species : float
        Triplet margin for cross‑species constraints.
    lr_species : float
        Learning rate for joint training.
    beta : float
        Weight for within‑species triplet loss.
    verbose : bool
        If True, print loss details every 100 epochs.
    random_seed : int
        Random seed.
    iter_comb : tuple or None
        Slice integration order.
    knn_neigh : int
        Neighbours for MNN construction.
    device : torch.device
        Device for joint training.
    pretrain_device : torch.device
        Device for pretraining.
    mse_beta : float
        Weight for MSE loss.
    tri_beta : float
        Weight for cross‑species triplet loss.
    mmd_beta : float
        Weight for MMD loss.
    gan_beta : float
        Weight for GAN loss (0 to disable).
    gan_epoch : int
        Discriminator training steps per generator step.
    ot_beta : float
        Weight for OT loss (0 to disable).
    mmd_batch_size : int
        Sample size for MMD computation.
    if_knn_mnn_graph : bool
        Whether to add cross‑species MNN edges to the graph.
    if_integrate_within_species : bool
        Whether to enable within‑species slice integration.
    if_return_loss : bool
        If True, return a loss tracking dictionary.
    if_batch_pretrain : bool
        Whether to pretrain with mini‑batch sampling.
    batch_size_dict : dict
        Mini‑batch sizes for pretraining per species.
    batch_size : int
        Batch size for subgraph sampling during joint training.
    concate_pca_dim : int or None
        PCA dimension for concatenated gene expression; if None, raw
        expression is used.
    umap_downsampling_rate : float
        Downsampling fraction for UMAP visualisation.
    adata_whole : AnnData
        Concatenated AnnData object across species.
    structure_beta : float
        Weight for per-species geometric structure preservation loss (0 to disable).
    structure_sampling_ratio : float
        Fraction of intra-species edges used for structure loss (1.0 = all).
    Returns
    -------
    adata_species_dict : dict
        Updated AnnData dict with final embedding in ``obsm[key_added]``.
    loss_dict : dict, optional
        Recorded losses per epoch, returned when ``if_return_loss=True``.
    """
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ---------- Set all random seeds for reproducibility ----------
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.autograd.set_detect_anomaly(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    if not isinstance(stagate_epoch, dict):
        stagate_epoch_dict = {k: stagate_epoch for k in adata_species_dict.keys()}
    else:
        stagate_epoch_dict = stagate_epoch

    # ---------- Per‑species STAGATE pretraining ----------
    z_dict = {k: 0 for k in adata_species_dict.keys()}
    species_order = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.uns['highly_variable']]
        print(f'For {species_id}, using {len(adata.var_names)} genes for training.')

        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    prune_edge_index=torch.LongTensor(np.array([])),
                    x=torch.FloatTensor(adata.X.todense()))

        if if_batch_pretrain:
            if species_order == 0:
                model = STACAME.STACAME_minibatch(
                    hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)

            train_loader = NeighborSampler(data.edge_index,
                                           node_idx=torch.LongTensor(np.array([i for i in range(adata.n_obs)])),
                                           sizes=[8, 4], batch_size=batch_size_dict[species_id], shuffle=True,
                                           drop_last=True)
            subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=batch_size_dict[species_id],
                                             shuffle=False)

            print('Pretrain with STAGATE (Minibatch)...')
            for epoch in tqdm(range(stagate_epoch_dict[species_id])):
                total_loss = 0
                for batchsize, n_id, adjs in train_loader:
                    adjs = [adj.to(pretrain_device) for adj in adjs]
                    optimizer.zero_grad()
                    z_batch, out_batch = model(data.x[n_id, :].to(pretrain_device), adjs, mode='batch')
                    x_batch = data.x[n_id, :].to(pretrain_device)

                    n_id_list = n_id.cpu().detach().numpy()
                    batch_id_list = adata.obs['batch_name'][n_id_list]
                    x_batch_cpu = z_batch.cpu().detach().numpy()
                    x_batch_adata = ad.AnnData(X=x_batch_cpu, obs=pd.DataFrame({"batch_name": batch_id_list}))
                    x_batch_adata.obsm['STAGATE'] = x_batch_cpu
                    section_ids = np.array(x_batch_adata.obs['batch_name'].unique())
                    mnn_dict = create_dictionary_mnn(x_batch_adata, use_rep='STAGATE', batch_name='batch_name',
                                                     k=knn_neigh, iter_comb=iter_comb, verbose=0)

                    anchor_ind = []
                    positive_ind = []
                    negative_ind = []
                    for batch_pair in mnn_dict.keys():
                        batchname_list = x_batch_adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                        cellname_by_batch_dict = dict()
                        for batch_id in range(len(section_ids)):
                            cellname_by_batch_dict[section_ids[batch_id]] = x_batch_adata.obs_names[
                                x_batch_adata.obs['batch_name'] == section_ids[batch_id]].values
                        anchor_list = []
                        positive_list = []
                        negative_list = []
                        for anchor in mnn_dict[batch_pair].keys():
                            anchor_list.append(anchor)
                            positive_spot = mnn_dict[batch_pair][anchor][0]
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(
                                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
                        batch_as_dict = dict(zip(list(x_batch_adata.obs_names), range(0, x_batch_adata.shape[0])))
                        anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                        positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                        negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

                    anchor_arr = z_batch[anchor_ind,]
                    positive_arr = z_batch[positive_ind,]
                    negative_arr = z_batch[negative_ind,]

                    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                    tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
                    loss = F.mse_loss(out_batch, x_batch) + beta * tri_output
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            with torch.no_grad():
                z_list = []
                out_list = []
                for batch in subgraph_loader:
                    batch.to(pretrain_device)
                    z, out = model(batch.x, batch.edge_index, mode='all')
                    z_list.append(z[:batch.batch_size].cpu())
                    out_list.append(out[:batch.batch_size].cpu())
            z_all = torch.cat(z_list, dim=0)
            out_all = torch.cat(out_list, dim=0)
            adata_species_dict[species_id].obsm['STAGATE'] = z_all.cpu().detach().numpy()
            z_dict[species_id] = z_all.cpu().detach()

            if species_order >= len(adata_species_dict.keys()):
                del model, optimizer, data, z_batch, out_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        else:
            print('Pretrain with STAligner...')
            if species_order == 0:
                model = STACAME.STACAME(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(
                    pretrain_device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)
            species_order += 1
            print('Pretrain with STAGATE_multiple...')
            for epoch in tqdm(range(0, stagate_epoch_dict[species_id])):
                model.train()
                optimizer.zero_grad()
                z, out = model(data.x.to(pretrain_device), data.edge_index.to(pretrain_device))

                if epoch % 50 == 0 and epoch >= stagate_epoch_dict[species_id] // 2:
                    adata.obsm['STAGATE'] = z.cpu().detach().numpy()
                    mnn_dict = create_dictionary_mnn(adata, use_rep='STAGATE', batch_name='batch_name', k=knn_neigh,
                                                     iter_comb=iter_comb, verbose=0)
                    anchor_ind = []
                    positive_ind = []
                    negative_ind = []
                    for batch_pair in mnn_dict.keys():
                        batchname_list = adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                        cellname_by_batch_dict = dict()
                        for batch_id in range(len(section_ids)):
                            cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                                adata.obs['batch_name'] == section_ids[batch_id]].values
                        anchor_list = []
                        positive_list = []
                        negative_list = []
                        for anchor in mnn_dict[batch_pair].keys():
                            anchor_list.append(anchor)
                            positive_spot = mnn_dict[batch_pair][anchor][0]
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(
                                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
                        batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                        anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                        positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                        negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

                    anchor_arr = z[anchor_ind,]
                    positive_arr = z[positive_ind,]
                    negative_arr = z[negative_ind,]
                    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                    tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
                    loss = F.mse_loss(data.x.to(pretrain_device), out) + beta * tri_output
                else:
                    loss = F.mse_loss(data.x.to(pretrain_device), out)
                loss.backward(retain_graph=True)
                optimizer.step()

            with torch.no_grad():
                z, _ = model(data.x.to(pretrain_device), data.edge_index.to(pretrain_device))
            adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
            z_dict[species_id] = z.cpu().detach()

    del model, optimizer, data, z, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    print('-------------------------------------------------------------------------------')
    print('Train with STACAME...')
    anchor_ind_species = triplet_ind_species_dict['anchor_ind_species']
    positive_ind_species = triplet_ind_species_dict['positive_ind_species']
    negative_ind_species = triplet_ind_species_dict['negative_ind_species']

    # ---------- Build concatenated graph ----------
    k_add = 0
    species_add_dict = {k: None for k in z_dict.keys()}
    for species_id in z_dict.keys():
        species_add_dict[species_id] = int(k_add)
        k_add = int(k_add + adata_species_dict[species_id].n_obs)

    adata = adata_species_dict[list(adata_species_dict.keys())[0]]
    edgeList = adata.uns['edgeList']
    edge_ndarray = np.array([edgeList[0], edgeList[1]])
    S = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if S != 0:
            edge_arr_temp = np.array([edgeList[0], edgeList[1]]) + species_add_dict[species_id]
            edge_ndarray = np.concatenate((edge_ndarray, edge_arr_temp), axis=1)
        else:
            S = S + 1

    edge_ndarray_species = np.array([edge_ndarray_species[0], edge_ndarray_species[1]])
    if if_knn_mnn_graph:
        edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_species), axis=1)

    S = 0
    for species_id, z_input in z_dict.items():
        if S == 0:
            X = z_dict[species_id].cpu().detach().numpy()
        else:
            X = np.concatenate((X, z_dict[species_id].cpu().detach().numpy()), axis=0)
        S = S + 1

    print('Pretrain with STAGATE_multiple...')
    print('Train with cross species STACAME...')
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    z = torch.FloatTensor(X)

    # Merge gene expression across species
    species_list = list(adata_species_dict.keys())
    n_species = len(species_list)
    ref_species = species_list[0]
    n_homo_genes = len(adata_species_dict[ref_species].uns['homo_highly_variable'])
    species_specific_n_genes = {
        sp: len(adata_species_dict[sp].uns['species_specific'])
        for sp in species_list
    }
    max_specific_genes = max(species_specific_n_genes.values())
    total_cols = n_homo_genes + max_specific_genes * n_species
    merge_X = None
    mask_matrix = None
    for sp_idx, species_id in enumerate(species_list):
        adata = adata_species_dict[species_id]
        homo_genes = adata.uns['homo_highly_variable']
        x_homo = adata[:, homo_genes].X.todense()
        specific_genes = adata.uns['species_specific']
        x_specific = adata[:, specific_genes].X.todense()
        n_cells = x_homo.shape[0]
        x_current = np.zeros((n_cells, total_cols))
        mask_current = np.zeros((n_cells, total_cols))
        x_current[:, :n_homo_genes] = x_homo
        mask_current[:, :n_homo_genes] = 1
        specific_start_col = n_homo_genes + sp_idx * max_specific_genes
        specific_end_col = specific_start_col + species_specific_n_genes[species_id]
        x_current[:, specific_start_col:specific_end_col] = x_specific
        mask_current[:, specific_start_col:specific_end_col] = 1
        if merge_X is None:
            merge_X = x_current
            mask_matrix = mask_current
        else:
            merge_X = np.concatenate((merge_X, x_current), axis=0)
            mask_matrix = np.concatenate((mask_matrix, mask_current), axis=0)

    if concate_pca_dim is not None:
        adata_X = ad.AnnData(merge_X)
        sc.pp.scale(adata_X)
        sc.tl.pca(adata_X, n_comps=concate_pca_dim)
        merge_X = adata_X.obsm["X_pca"]
    merge_X = torch.FloatTensor(merge_X).to(device)

    if hasattr(adata_whole.obsm['X_pca'], 'todense'):
        auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'].todense())
    else:
        auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'])

    # ---------- Build models ----------
    if if_use_light_model:
        model = STACAME.STACAME_lightdecoder_minibatch(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    else:
        model = STACAME.STACAMEDecoder_minibatch(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    auxiliary_model = STACAME.STACAME_minibatch(
        hidden_dims=[auxiliary_X.shape[1], hidden_dims[0] // 2, hidden_dims[1]]).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(auxiliary_model.parameters()), lr=lr_species,
                                 weight_decay=weight_decay, foreach=False)

    species_n_cells = {sp: adata_species_dict[sp].n_obs for sp in adata_species_dict.keys()}

    if if_integrate_within_species:
        anchor_ind_sections = triplet_ind_sections_dict['anchor_ind_sections']
        positive_ind_sections = triplet_ind_sections_dict['positive_ind_sections']
        negative_ind_sections = triplet_ind_sections_dict['negative_ind_sections']
        if if_knn_mnn_graph:
            edge_ndarray_sections = np.array([edge_ndarray_sections[0], edge_ndarray_sections[1]])
            edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_sections), axis=1)
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)
    else:
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)

    for spe, adata in adata_species_dict.items():
        print(spe, adata.n_obs)

    id_species_dict = {k: None for k in range(merge_X.shape[0])}
    k_add = 0
    for spe_id in adata_species_dict.keys():
        for id_s in range(k_add, k_add + adata_species_dict[spe_id].n_obs):
            id_species_dict[id_s] = spe_id
        k_add = k_add + adata_species_dict[spe_id].n_obs

    subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=batch_size * 2, shuffle=False)

    if if_return_loss:
        loss_dict = {'Loss name': [], 'Epoch': [], 'Loss value': []}

    plot_epoch = n_epochs_species // 3

    D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    optimizer_D = torch.optim.Adam(list(D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)
    D_Z.train()

    auxiliary_D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    auxiliary_optimizer_D = torch.optim.Adam(list(auxiliary_D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)
    auxiliary_D_Z.train()

    species_list_gt = []
    for species_id, adata in adata_species_dict.items():
        species_list_gt = species_list_gt + [species_id] * adata.n_obs
    true_dom = torch.LongTensor(pd.Series(species_list_gt).astype('category').cat.codes.values)

    auxiliary_data = Data(edge_index=torch.LongTensor(edge_ndarray),
                          prune_edge_index=torch.LongTensor(np.array([])), x=auxiliary_X)
    auxiliary_subgraph_loader = NeighborLoader(auxiliary_data, num_neighbors=[-1], batch_size=batch_size * 2,
                                               shuffle=False)

    ite_N = max(int((len(anchor_ind_species) // batch_size)) + 1, 1)
    print('ite_N', ite_N)
    species_ids = list(adata_species_dict.keys())
    species_id_list = list(adata_species_dict.keys())

    # ---------- Helper: Manifold preserving loss ----------
    def manifold_preserving_loss(latent_vectors, ref_vectors, same_species_edges, edge_weights=None):
        """
        Weighted Laplacian loss. Optional edge_weights tensor of shape (E,)
        gives per‑edge importance (e.g. inverse species size).
        """
        src, dst = same_species_edges[0], same_species_edges[1]
        d2_ref = torch.sum((ref_vectors[src] - ref_vectors[dst]) ** 2, dim=1)
        sigma = torch.sqrt(torch.median(d2_ref) + 1e-8)
        w = torch.exp(-d2_ref / (2 * sigma ** 2 + 1e-8))
        d2_lat = torch.sum((latent_vectors[src] - latent_vectors[dst]) ** 2, dim=1)
        if edge_weights is None:
            edge_weights = torch.ones_like(w)
        return torch.mean(edge_weights * w * d2_lat)

    # ---------- Cross‑species joint training ----------
    for epoch in tqdm(range(0, n_epochs_species)):
        k_add = 0
        for species_id in z_dict.keys():
            adata_species_dict[species_id].obsm['STAGATE'] = z[k_add:int(k_add + adata_species_dict[species_id].n_obs),
                                                             :].cpu().detach().numpy()
            z_dict[species_id] = adata_species_dict[species_id].obsm['STAGATE']
            k_add = int(k_add + adata_species_dict[species_id].n_obs)

        if epoch == 0:
            anchor_ind_species_ = anchor_ind_species
            positive_ind_species_ = positive_ind_species
            negative_ind_species_ = negative_ind_species
            if hasattr(adata_whole.obsm['X_pca'], 'todense'):
                adata_whole.obsm['auxiliary'] = adata_whole.obsm['X_pca'].todense()
            else:
                adata_whole.obsm['auxiliary'] = adata_whole.obsm['X_pca']

        if epoch % 100 == 0 and epoch > 0:
            mnn_dict = create_dictionary_mnn(adata_whole, use_rep='auxiliary', batch_name='species_id', k=knn_neigh,
                                             iter_comb=iter_comb, verbose=0)
            anchor_ind_species_ = []
            positive_ind_species_ = []
            negative_ind_species_ = []
            for batch_pair in mnn_dict.keys():
                batchname_list = adata_whole.obs['species_id'][mnn_dict[batch_pair].keys()]
                cellname_by_batch_dict = dict()
                for batch_id in range(len(species_ids)):
                    cellname_by_batch_dict[species_ids[batch_id]] = adata_whole.obs_names[
                        adata_whole.obs['species_id'] == species_ids[batch_id]].values
                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in mnn_dict[batch_pair].keys():
                    anchor_list.append(anchor)
                    positive_spot = mnn_dict[batch_pair][anchor][0]
                    positive_list.append(positive_spot)
                    section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                    negative_list.append(
                        cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
                batch_as_dict = dict(zip(list(adata_whole.obs_names), range(0, adata_whole.shape[0])))
                anchor_ind_species_ = np.append(anchor_ind_species_, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind_species_ = np.append(positive_ind_species_,
                                                  list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind_species_ = np.append(negative_ind_species_,
                                                  list(map(lambda _: batch_as_dict[_], negative_list)))
            anchor_ind_species_ = np.concatenate([anchor_ind_species, anchor_ind_species_])
            positive_ind_species_ = np.concatenate([positive_ind_species, positive_ind_species_])
            negative_ind_species_ = np.concatenate([negative_ind_species, negative_ind_species_])

        for ite_ in range(ite_N):
            triples_N = len(anchor_ind_species)
            tri_ind_list = random.sample(list(range(triples_N)), min(triples_N, batch_size))

            anchor_ind_species_batch = [anchor_ind_species_[x] for x in tri_ind_list]
            positive_ind_species_batch = [positive_ind_species_[x] for x in tri_ind_list]
            negative_ind_species_batch = [negative_ind_species_[x] for x in tri_ind_list]
            ind_list_init = list(
                set(anchor_ind_species_batch + positive_ind_species_batch + negative_ind_species_batch))

            if if_integrate_within_species:
                triples_N_sec = len(anchor_ind_sections)
                tri_ind_list_sec = random.sample(list(range(triples_N_sec)), min(batch_size, triples_N_sec))
                anchor_ind_sections_batch = [anchor_ind_sections[x] for x in tri_ind_list_sec]
                positive_ind_sections_batch = [positive_ind_sections[x] for x in tri_ind_list_sec]
                negative_ind_sections_batch = [negative_ind_sections[x] for x in tri_ind_list_sec]
                ind_list_init = list(set(ind_list_init + list(
                    set(anchor_ind_sections_batch + positive_ind_sections_batch + negative_ind_sections_batch))))

            idx_subset, edge_index_batch, mapping, edge_mask = k_hop_subgraph(node_idx=torch.LongTensor(ind_list_init),
                                                                              num_hops=1, edge_index=data.edge_index,
                                                                              relabel_nodes=True)
            idx_subset_list = [int(x) for x in idx_subset]
            idx_map = {k: v for k, v in zip(idx_subset_list, range(len(idx_subset_list)))}

            model.train()
            optimizer.zero_grad()
            z_batch, out = model(data.x[idx_subset_list,].to(device), edge_index_batch.to(device), mode='whole')
            auxiliary_z_batch, auxiliary_out = auxiliary_model(auxiliary_data.x[idx_subset_list,].to(device),
                                                               edge_index_batch.to(device), mode='whole')
            # Structural reference: data.x for manifold preserving
            ref_x_sub = data.x[idx_subset_list,].to(device)  # (num_sub, d_pretrain)

            # 1) MSE loss
            mse_loss = F.mse_loss(merge_X[idx_subset_list,].to(device), out) + F.mse_loss(
                auxiliary_X[idx_subset_list,].to(device), auxiliary_out)

            # 2) Within-species slice triplet loss (optional)
            if if_integrate_within_species:
                anchor_arr = z_batch[[idx_map[x] for x in anchor_ind_sections_batch],]
                positive_arr = z_batch[[idx_map[x] for x in positive_ind_sections_batch],]
                negative_arr = z_batch[[idx_map[x] for x in negative_ind_sections_batch],]
                triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
            else:
                tri_output = torch.tensor(0.0, device=device)

            # 3) Cross-species triplet loss
            anchor_arr_species = z_batch[[idx_map[x] for x in anchor_ind_species_batch],]
            positive_arr_species = z_batch[[idx_map[x] for x in positive_ind_species_batch],]
            negative_arr_species = z_batch[[idx_map[x] for x in negative_ind_species_batch],]
            triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin_species, p=2, reduction='mean')
            tri_output_species = triplet_loss_species(anchor_arr_species, positive_arr_species, negative_arr_species)

            # 4) MMD loss
            z_ind_species_dict = {k: [] for k in adata_species_dict.keys()}
            for n_id_temp in idx_subset_list:
                n_id_species = id_species_dict[n_id_temp]
                z_ind_species_dict[n_id_species].append(n_id_temp)

            mmd_loss = STACAME.MMDLoss(kernel=STACAME.RBF(device=device), device=device).to(device)
            mmd_loss_sum = 0

            spe_id = random.sample(species_id_list, 1)[0]
            spe_id_list = [idx_map[x] for x in z_ind_species_dict[spe_id]]
            bsize = min(len(spe_id_list), len(idx_subset_list) - len(spe_id_list))

            z_A = z_batch[spe_id_list[0:bsize],]
            z_B_ind_list = random.sample(list(set(range(0, len(idx_subset_list))) - set(spe_id_list)), bsize)
            z_B = z_batch[z_B_ind_list,]

            # Not used elsewhere but kept for possible OT loss
            x_batch = auxiliary_X[idx_subset_list,].to(device)
            x_A = x_batch[spe_id_list[0:bsize],]
            x_B_ind_list = random.sample(list(set(range(0, len(idx_subset_list))) - set(spe_id_list)), bsize)
            x_B = x_batch[x_B_ind_list,]

            auxiliary_z_A = auxiliary_z_batch[spe_id_list[0:bsize],]
            auxiliary_z_B = auxiliary_z_batch[z_B_ind_list,]
            anchor_arr_species_X = x_batch[[idx_map[x] for x in anchor_ind_species_batch],]
            positive_arr_species_X = x_batch[[idx_map[x] for x in positive_ind_species_batch],]

            # 5) GAN loss
            if gan_beta != 0:
                for _ in range(gan_epoch):
                    optimizer_D.zero_grad()
                    logits_D = D_Z(z_batch)
                    loss_D = F.cross_entropy(logits_D, true_dom[idx_subset_list,].to(device))
                    loss_D.backward(retain_graph=True)
                    optimizer_D.step()
                for _ in range(gan_epoch):
                    auxiliary_optimizer_D.zero_grad()
                    auxiliary_logits_D = auxiliary_D_Z(auxiliary_z_batch)
                    auxiliary_loss_D = F.cross_entropy(auxiliary_logits_D, true_dom[idx_subset_list,].to(device))
                    auxiliary_loss_D.backward(retain_graph=True)
                    auxiliary_optimizer_D.step()

            loss_G_GAN = -F.cross_entropy(D_Z(z_batch), true_dom[idx_subset_list,].to(device)) - F.cross_entropy(
                auxiliary_D_Z(auxiliary_z_batch), true_dom[idx_subset_list,].to(device))

            mmd_loss_sum = mmd_loss(z_A[0:mmd_batch_size, :], z_B[0:mmd_batch_size, :]).to(device) + \
                           mmd_loss(auxiliary_z_A[0:mmd_batch_size, :], auxiliary_z_B[0:mmd_batch_size, :]).to(device)

            # OT loss (optional)
            loss_ot_value = 0.0
            loss_ot = torch.tensor(0.0, device=device)
            if ot_beta != 0:
                c_cross = pairwise_correlation_distance(anchor_arr_species_X.detach(),
                                                        positive_arr_species_X.detach()).to(device)
                T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=device)
                z_dist = torch.mean((anchor_arr_species.view(anchor_arr_species_X.shape[0], 1,
                                                             -1) - positive_arr_species_X.view(
                    1, anchor_arr_species_X.shape[0], -1)) ** 2, dim=2)
                loss_ot = torch.sum(T * z_dist) / torch.sum(T)
                loss_ot_value = loss_ot.item()

            # ---- Manifold preserving loss (replaces old Gaussian affinity) ----
            # ---- Manifold preserving loss (with species size weighting) ----
            geom_structure_loss = torch.tensor(0.0, device=device)
            if structure_beta != 0.0:
                new_idx_to_species = {new: id_species_dict[orig]
                                      for orig, new in zip(idx_subset_list, range(len(idx_subset_list)))}
                edge_np = edge_index_batch.cpu().numpy()
                species_mask = np.array([
                    new_idx_to_species.get(int(src), None) is not None and
                    new_idx_to_species.get(int(dst), None) is not None and
                    new_idx_to_species[int(src)] == new_idx_to_species[int(dst)]
                    for src, dst in edge_np.T
                ])
                species_edges = edge_index_batch[:, species_mask]
                if species_edges.shape[1] > 0:
                    n_edges = species_edges.shape[1]
                    n_sample = max(1, int(n_edges * structure_sampling_ratio))
                    perm = torch.randperm(n_edges, device=device)[:n_sample]
                    sampled_edges = species_edges[:, perm]
            
                    # --- compute species‑size‑based edge weights ---
                    src_nodes = sampled_edges[0].cpu().numpy()
                    edge_species = [new_idx_to_species[int(node)] for node in src_nodes]
                    # inverse of number of cells for that species
                    inv_n = {sp: 1.0 / species_n_cells[sp] for sp in species_n_cells}
                    raw_weights = torch.tensor([inv_n[sp] for sp in edge_species],
                                               device=device, dtype=torch.float32)
                    # normalise so that the average weight equals 1 (keeps loss magnitude stable)
                    edge_w = raw_weights * (len(raw_weights) / raw_weights.sum())
                    # ------------------------------------------------
            
                    loss_z = manifold_preserving_loss(z_batch, ref_x_sub, sampled_edges, edge_weights=edge_w)
                    loss_aux = manifold_preserving_loss(auxiliary_z_batch, ref_x_sub, sampled_edges, edge_weights=edge_w)
                    geom_structure_loss = loss_z + loss_aux

            # Total loss
            if if_integrate_within_species:
                loss = mse_beta * mse_loss + tri_beta * tri_output_species + beta * tri_output + \
                       mmd_beta * mmd_loss_sum + gan_beta * loss_G_GAN + ot_beta * loss_ot + \
                       structure_beta * geom_structure_loss
            else:
                loss = mse_beta * mse_loss + tri_beta * tri_output_species + \
                       mmd_beta * mmd_loss_sum + gan_beta * loss_G_GAN + ot_beta * loss_ot + \
                       structure_beta * geom_structure_loss

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

        # ---------- Record and print losses ----------
        if if_return_loss:
            loss_dict['Loss name'].append('Loss sum')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(loss.item())
            loss_dict['Loss name'].append('MSE')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mse_loss.item())
            loss_dict['Loss name'].append('Cross-species triplet')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(tri_output_species.item())
            loss_dict['Loss name'].append('MMD')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mmd_loss_sum.item())
            if structure_beta != 0.0:
                loss_dict['Loss name'].append('ManifoldPreserve')
                loss_dict['Epoch'].append(epoch)
                loss_dict['Loss value'].append(geom_structure_loss.item())

        if verbose and epoch % 100 == 0:
            cos_sim = cosine_loss(anchor_arr_species, positive_arr_species,
                                  torch.ones(len(anchor_arr_species)).to(device)).item()
            out_str = (f'Epoch {epoch:4d} | Total: {loss.item():.4f} | '
                       f'MSE: {mse_beta * mse_loss.item():.4f} | '
                       f'Cross-species Tri: {tri_beta * tri_output_species.item():.4f} | '
                       f'MMD: {mmd_beta * mmd_loss_sum.item():.4f} | '
                       f'GAN: {gan_beta * loss_G_GAN.item():.4f} | '
                       f'OT: {ot_beta * loss_ot_value:.4f} | '
                       f'CosSim: {cos_sim:.4f}')
            if if_integrate_within_species:
                out_str += f' | Slice Tri: {beta * tri_output.item():.4f}'
            if structure_beta != 0.0:
                out_str += f' | Manifold: {structure_beta * geom_structure_loss.item():.4f}'
            print(out_str)

        with torch.no_grad():
            z_list = []
            out_list = []
            for batch in subgraph_loader:
                batch.to(device)
                z, out = model(batch.x, batch.edge_index, mode='all')
                z_list.append(z[:batch.batch_size].cpu())
                out_list.append(out[:batch.batch_size].cpu())

            auxiliary_z_list = []
            for batch in auxiliary_subgraph_loader:
                batch.to(device)
                auxiliary_z, auxiliary_out = auxiliary_model(batch.x, batch.edge_index, mode='all')
                auxiliary_z_list.append(auxiliary_z[:batch.batch_size].cpu())

        z = torch.cat(z_list, dim=0)
        out_all = torch.cat(out_list, dim=0)

        auxiliary_z = torch.cat(auxiliary_z_list, dim=0)
        adata_whole.obsm['auxiliary'] = auxiliary_z.cpu().detach().numpy()
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[
                k_add:int(k_add + adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()

        if epoch % plot_epoch == 0 and n_epochs_species - epoch >= plot_epoch:
            if z.shape[0] >= 50000:
                clustering_umap_downsampling(adata_species_dict, key_umap=key_added,
                                             downsampling_rate=umap_downsampling_rate)
            else:
                clustering_umap(adata_species_dict, key_umap=key_added)

    print('Clustering and UMAP of Cross Species STACAME:')
    if z.shape[0] >= 50000:
        clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=umap_downsampling_rate)
    else:
        clustering_umap(adata_species_dict, key_umap=key_added)

    del model, optimizer, D_Z, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict



## Subgraph and light version of STACAME without GAN loss and auxiliary model
def train_STACAME_subgraph(adata_species_dict, 
                  triplet_ind_species_dict, 
                  edge_ndarray_species,
                  triplet_ind_sections_dict = None, 
                  edge_ndarray_sections = None,
                  hidden_dims=[512, 30], 
                  stagate_epoch=500,   
                  n_epochs_species=2000,  
                  lr=0.001, 
                  key_added='STACAME',
                  gradient_clipping=5.,
                  weight_decay=0.0001, 
                  lr_wd=0.001, 
                  weight_decay_wd=5e-4,
                  margin=1.0, 
                  margin_species=1.0, 
                  lr_species = 0.001,
                  beta=1,
                  verbose=False,
                  random_seed=666, 
                  iter_comb=None, 
                  knn_neigh=20, 
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                  pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                  mse_beta = 1, 
                  tri_beta = 1, 
                  mmd_beta = 1, 
                  gan_beta = 1,
                  mmd_batch_size = 2048, 
                  if_knn_mnn_graph = True, 
                  if_integrate_within_species = False, 
                  if_return_loss = False, 
                  batch_size_dict = {'Mouse': 10000, 'Marmoset':10000, 'Macaque':4096}, 
                  batch_size = 2048, 
                  umap_downsampling_rate = 0.1, 
                  mode = 'spatial_domain', 
                  concate_pca_dim = None):
    """\
    Train graph attention auto-encoder and use spot triplets across species to perform batch correction in the embedding space.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    margin
        Margin is used in triplet loss to enforce the distance between positive and negative pairs.
        Larger values result in more aggressive correction.
    iter_comb
        For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
        For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    knn_neigh
        The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
    device
        See torch.device.

    Returns
    -------
    AnnData dict, storing AnnData for each species
    """

    import gc
    # 初始清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    #torch.use_deterministic_algorithms(True)
    if not isinstance(stagate_epoch, dict):
        stagate_epoch_dict = {k:stagate_epoch for k in adata_species_dict.keys()}
    else:
        stagate_epoch_dict = stagate_epoch


    # Initializing the hidden units, model and optimizer dicts
    z_dict = {k:0 for k in adata_species_dict.keys()}
    # Train a STAGATE model for each species
    species_order = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.uns['highly_variable']]
        print(f'For {species_id}, using {len(adata.var_names)} genes for training.')
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    prune_edge_index=torch.LongTensor(np.array([])),
                    x=torch.FloatTensor(adata.X.todense()))

        if species_order == 0:
            model = STACAME.STACAME_minibatch(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)

        # Minibatch
        train_loader = NeighborSampler(data.edge_index, node_idx=torch.LongTensor(np.array([i for i in range(adata.n_obs)])),
                               sizes=[8, 4], batch_size=batch_size_dict[species_id], shuffle=True, drop_last = True)
        subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=batch_size_dict[species_id],
                                          shuffle=False)
    
        print('Pretrain with STAGATE (Minibatch)...')
        for epoch in tqdm(range(stagate_epoch_dict[species_id])):
            total_loss = 0
            for batchsize, n_id, adjs in train_loader:
                adjs = [adj.to(pretrain_device) for adj in adjs]
                optimizer.zero_grad()
                z_batch, out_batch = model(data.x[n_id, :].to(pretrain_device), adjs, mode='batch')
                # get batch data
                x_batch = data.x[n_id,:].to(pretrain_device)
                loss = F.mse_loss(out_batch, x_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()
                total_loss += loss.item()
            #print(f'Epoch = {epoch}, mse loss = {total_loss/len(train_loader)}')

        with torch.no_grad():
            z_list = []
            out_list = []
            for batch in subgraph_loader:
                batch.to(pretrain_device)
                z, out = model(batch.x, batch.edge_index, mode='all')
                z_list.append(z[:batch.batch_size].cpu())
                out_list.append(out[:batch.batch_size].cpu())
        
        # z, _ = model(data.x, data.edge_index)
        z_all = torch.cat(z_list, dim=0)
        out_all = torch.cat(out_list, dim=0)
        adata_species_dict[species_id].obsm['STAGATE'] = z_all.cpu().detach().numpy()
        z_dict[species_id] = z_all.cpu().detach()

        # ===================== 核心：释放预训练显存（修改点2）=====================
    del model, optimizer, data, z_batch, out_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
 
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    print('-------------------------------------------------------------------------------')
    print('Train with STACAME...')
    anchor_ind_species = triplet_ind_species_dict['anchor_ind_species']
    positive_ind_species = triplet_ind_species_dict['positive_ind_species']
    negative_ind_species = triplet_ind_species_dict['negative_ind_species']
    ##########################################-Cross Species####################################################
    k_add = 0
    species_add_dict = {k:None for k in z_dict.keys()}
    for species_id in z_dict.keys():
        species_add_dict[species_id] = int(k_add)
        k_add = int(k_add+adata_species_dict[species_id].n_obs)
    
    adata = adata_species_dict[list(adata_species_dict.keys())[0]]
    edgeList = adata.uns['edgeList']
    edge_ndarray = np.array([edgeList[0], edgeList[1]])
    #if verbose:
    S = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if S != 0:
            edge_arr_temp = np.array([edgeList[0], edgeList[1]]) + species_add_dict[species_id]
            edge_ndarray = np.concatenate((edge_ndarray, edge_arr_temp), axis=1)
        else:
            S = S + 1
    edge_ndarray_species = np.array([edge_ndarray_species[0], edge_ndarray_species[1]])
    
    # Choose whether build the knn and mnn cross species neigbors into the graph
    if if_knn_mnn_graph == True:
        edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_species), axis=1)
    S = 0
    for species_id, z_input in z_dict.items():
        if S==0:
            X = z_dict[species_id].cpu().detach().numpy()
        else:
            X = np.concatenate((X, z_dict[species_id].cpu().detach().numpy()), axis=0)
        S = S + 1
    # ---------------------------Init graph-----------------------------
    print('Pretrain with STAGATE_multiple...')
    #----------------------------Create model------------------------------
    print('Train with cross species STACAME...')
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    
    z = torch.FloatTensor(X)
    # Merge X and get cross-species features
    species_list = list(adata_species_dict.keys())
    n_species = len(species_list)
    ref_species = species_list[0]
    n_homo_genes = len(adata_species_dict[ref_species].uns['homo_highly_variable'])
    species_specific_n_genes = {
        sp: len(adata_species_dict[sp].uns['species_specific'])
        for sp in species_list
    }
    max_specific_genes = max(species_specific_n_genes.values())
    total_cols = n_homo_genes + max_specific_genes * n_species
    merge_X = None
    #merge_X_count = None
    mask_matrix = None
    for sp_idx, species_id in enumerate(species_list):
        adata = adata_species_dict[species_id]
        homo_genes = adata.uns['homo_highly_variable']
        x_homo = adata[:, homo_genes].X.todense()  # shape: (n_cells, n_homo_genes)
        #x_count_homo = adata.obsm['counts_hvg_share'].todense()
        specific_genes = adata.uns['species_specific']
        x_specific = adata[:, specific_genes].X.todense()  # shape: (n_cells, n_specific_genes)
        #x_count_specific = adata.obsm['counts_hvg_specific'].todense()
        n_cells = x_homo.shape[0]
        x_current = np.zeros((n_cells, total_cols))
        #x_count_current = np.zeros((n_cells, total_cols))
        mask_current = np.zeros((n_cells, total_cols))  
       
        x_current[:, :n_homo_genes] = x_homo
        #x_count_current[:, :n_homo_genes] = x_count_homo
        mask_current[:, :n_homo_genes] = 1 
        specific_start_col = n_homo_genes + sp_idx * max_specific_genes
        specific_end_col = specific_start_col + species_specific_n_genes[species_id]

        x_current[:, specific_start_col:specific_end_col] = x_specific
        #x_count_current[:, specific_start_col:specific_end_col] = x_count_specific
        mask_current[:, specific_start_col:specific_end_col] = 1  # 特异区域mask为1
        if merge_X is None:
            merge_X = x_current
            #merge_X_count = x_count_current
            mask_matrix = mask_current
        else:
            merge_X = np.concatenate((merge_X, x_current), axis=0)
            #merge_X_count = np.concatenate((merge_X_count, x_count_current), axis=0)
            mask_matrix = np.concatenate((mask_matrix, mask_current), axis=0)


    if concate_pca_dim != None:
        adata_X = ad.AnnData(merge_X)
        sc.pp.scale(adata_X)
        sc.tl.pca(adata_X, n_comps=concate_pca_dim)
        merge_X = adata_X.obsm["X_pca"]
    merge_X = torch.FloatTensor(merge_X).to(device)

    ##-----------------------------------------------------------
    model = STACAME.STACAMEDecoder_minibatch(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_species, weight_decay=weight_decay, foreach=False)
    
    if if_integrate_within_species == True:
        anchor_ind_sections = triplet_ind_sections_dict['anchor_ind_sections']
        positive_ind_sections = triplet_ind_sections_dict['positive_ind_sections']
        negative_ind_sections = triplet_ind_sections_dict['negative_ind_sections']
        if if_knn_mnn_graph == True:
            edge_ndarray_sections = np.array([edge_ndarray_sections[0], edge_ndarray_sections[1]])
            edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_sections), axis=1)
            data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
            data = data.to(device)
        else:
            data = Data(edge_index=torch.LongTensor(edge_ndarray),
                prune_edge_index=torch.LongTensor(np.array([])), x=z)
            data = data.to(device)
    else:
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
            prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)

    #print('adata_species_dict.keys()', adata_species_dict.keys())
    for spe, adata in adata_species_dict.items():
        print(spe, adata.n_obs)

    id_species_dict = {k:None for k in range(merge_X.shape[0])}
    k_add = 0
    for spe_id in adata_species_dict.keys():
        for id_s in range(k_add, k_add + adata_species_dict[spe_id].n_obs):
            id_species_dict[id_s] = spe_id
        k_add = k_add + adata_species_dict[spe_id].n_obs #species_add_dict[spe_id]

    # The predict data loader, where the number 5 should be adjusted depended on the GPU memory
    subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=batch_size*5, shuffle=False)

    if if_return_loss:
        loss_dict = {'Loss name':[], 'Epoch':[], 'Loss value':[]}

    plot_epoch = n_epochs_species // 3

    D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    optimizer_D = torch.optim.Adam(list(D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)
    D_Z.train()

    species_list = []
    #celltype_all_list = []
    for species_id, adata in adata_species_dict.items():
        species_list = species_list + [species_id] * adata.n_obs

    true_dom = torch.LongTensor(pd.Series(species_list).astype('category').cat.codes.values)#.to(device)  

    ite_N = int(0.5*(len(anchor_ind_species) // batch_size))
    
    for epoch in tqdm(range(0, n_epochs_species)):
        # subsampling anchor, positive, and negative
        k_add = 0
        for species_id in z_dict.keys():
            #species_add_dict[species_id] = int(k_add)
            adata_species_dict[species_id].obsm['STAGATE'] = z[k_add:int(k_add+adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
            z_dict[species_id] = adata_species_dict[species_id].obsm['STAGATE']
            k_add = int(k_add + adata_species_dict[species_id].n_obs)
      
        mse_loss_mean = 0
        mse_loss = 0
        tri_output_species_mean = 0
        tri_output_slice_mean = 0
        mmd_loss_sum_mean = 0
        batch_num = 0
        species_id_list = list(adata_species_dict.keys())


        for ite_ in range(ite_N):
            triples_N = len(anchor_ind_species)
            tri_ind_list = random.sample(list(range(triples_N)), batch_size)

            anchor_ind_species_batch = [anchor_ind_species[x] for x in tri_ind_list]
            positive_ind_species_batch = [positive_ind_species[x] for x in tri_ind_list]
            negative_ind_species_batch = [negative_ind_species[x] for x in tri_ind_list]

            ind_list_init = list(set(anchor_ind_species_batch + positive_ind_species_batch + negative_ind_species_batch))

            if if_integrate_within_species == True:
                triples_N_sec = len(anchor_ind_sections)
                tri_ind_list_sec = random.sample(list(range(triples_N_sec)), batch_size)

                anchor_ind_sections_batch = [anchor_ind_sections[x] for x in tri_ind_list_sec]
                positive_ind_sections_batch = [positive_ind_sections[x] for x in tri_ind_list_sec]
                negative_ind_sections_batch = [negative_ind_sections[x] for x in tri_ind_list_sec]
                
                ind_list_init = list(set(ind_list_init + list(set(anchor_ind_sections_batch + positive_ind_sections_batch + negative_ind_sections_batch))))

                
            idx_subset, edge_index_batch, mapping, edge_mask = k_hop_subgraph(node_idx = torch.LongTensor(ind_list_init), num_hops = 1, edge_index=data.edge_index,   relabel_nodes=True)

            idx_subset_list = [int(x) for x in idx_subset]
            idx_map = {k:v for k,v in zip(idx_subset_list, range(len(idx_subset_list)))}

            model.train()
            optimizer.zero_grad()
            z_batch, out = model(data.x[idx_subset_list,].to(device), edge_index_batch.to(device), mode='whole')
            mse_loss = F.mse_loss(merge_X[idx_subset_list,].to(device), out)
            #l1_loss = L1_loss(merge_X[idx_subset,].to(device), out)
            if if_integrate_within_species == True:
                anchor_arr = z_batch[[idx_map[x] for x in anchor_ind_sections_batch],]
                positive_arr = z_batch[[idx_map[x] for x in positive_ind_sections_batch],]
                negative_arr = z_batch[[idx_map[x] for x in negative_ind_sections_batch],]
                triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
            else:
                pass
            anchor_arr_species = z_batch[[idx_map[x] for x in anchor_ind_species_batch],]
            positive_arr_species = z_batch[[idx_map[x] for x in positive_ind_species_batch],]
            negative_arr_species = z_batch[[idx_map[x] for x in negative_ind_species_batch],]
            triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin_species, p=2, reduction='mean')
            tri_output_species = triplet_loss_species(anchor_arr_species, positive_arr_species, negative_arr_species)

            #print(f'Number of triplets: {len(anchor_ind_species_batch)}')

            z_ind_species_dict = {k: [] for k in adata_species_dict.keys()}
            for n_id_temp in idx_subset_list:
                n_id_species = id_species_dict[n_id_temp]
                z_ind_species_dict[n_id_species].append(n_id_temp)
            
            mmd_loss = STACAME.MMDLoss(kernel=STACAME.RBF(device=device), device=device).to(device)
            mmd_loss_sum = 0
            
            
            spe_id = random.sample(species_id_list, 1)[0]
            #print(spe_id)
            spe_id_list = [idx_map[x] for x in z_ind_species_dict[spe_id]]

            bsize = min(len(spe_id_list), len(idx_subset_list) - len(spe_id_list))
            
            z_A = z_batch[spe_id_list[0:bsize],]
            z_B_ind_list = random.sample(list(set(range(0, len(idx_subset_list))) - set(spe_id_list)), bsize)    
            z_B = z_batch[z_B_ind_list,]

            if gan_beta != 0:
                for _ in range(5):
                    optimizer_D.zero_grad()
                    logits_D = D_Z(z_batch) 
                    loss_D = F.cross_entropy(logits_D, true_dom[idx_subset_list, ].to(device))
                    loss_D.backward(retain_graph=True)
                    optimizer_D.step()
                
            loss_G_GAN = -F.cross_entropy(D_Z(z_batch) , true_dom[idx_subset_list, ].to(device))
            mmd_loss_sum = mmd_loss(z_A[0:mmd_batch_size], z_B[0:mmd_batch_size]).to(device)
            if if_integrate_within_species == True:     
                loss =  mse_beta * mse_loss + tri_beta * tri_output_species + beta * tri_output  + mmd_beta * mmd_loss_sum +  gan_beta * loss_G_GAN 
            else:
                loss =  mse_beta * mse_loss + tri_beta * tri_output_species + mmd_beta * mmd_loss_sum + gan_beta * loss_G_GAN

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            

        if if_return_loss:
            loss_dict['Loss name'].append('Loss sum')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(loss.item())
            loss_dict['Loss name'].append('MSE')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mse_loss.item())
            loss_dict['Loss name'].append('Cross-species triplet')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(tri_output_species.item())
            loss_dict['Loss name'].append('MMD')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mmd_loss_sum.item())
        if verbose == True:
            print(f'---------------------------------Epoch {epoch}-----------------------------------')
            print(f'Total loss: {loss.item()}')
            print(
                f'MSE:{mse_beta * mse_loss.item()}, Cross species triplets:{tri_beta * tri_output_species.item()}, '
                f'MMD:{mmd_beta * mmd_loss_sum.item()}, GAN: {gan_beta * loss_G_GAN.item()}')
            if if_integrate_within_species == True:
                print(f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(len(anchor_arr_species)).to(device)).item()}, Cross slices triplets: {tri_output.item()}')
                if if_return_loss:
                    loss_dict['Loss name'].append('Cross-slices triplet')
                    loss_dict['Epoch'].append(epoch)
                    loss_dict['Loss value'].append(tri_output.item())
            else:
                print(f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(len(anchor_arr_species)).to(device)).item()}')


        with torch.no_grad():
            z_list = []
            out_list = []
            for batch in subgraph_loader:
                batch.to(device)
                z, out = model(batch.x, batch.edge_index, mode='all')
                z_list.append(z[:batch.batch_size].cpu())
                out_list.append(out[:batch.batch_size].cpu())
        
        z = torch.cat(z_list, dim=0)
        out_all = torch.cat(out_list, dim=0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[k_add:int(k_add+adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
      
    print('Clustering and UMAP of Cross Species STACAME:')
    clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate = umap_downsampling_rate)

    del model, optimizer, D_Z,  data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict


## Subgraph and light version of STACAME with GAN loss and but without auxiliary model
def train_STACAME_subgraph_GAN(adata_species_dict,
                  triplet_ind_species_dict, 
                  edge_ndarray_species,
                  triplet_ind_sections_dict = None, 
                  edge_ndarray_sections = None,
                  hidden_dims=[512, 30], 
                  stagate_epoch=500, 
                  n_epochs_species=2000,  
                  lr=0.001, 
                  key_added='STACAME',
                  gradient_clipping=5.,
                  weight_decay=0.0001, 
                  lr_wd=0.001, 
                  weight_decay_wd=5e-4,
                  margin=1.0, 
                  margin_species=1.0, 
                  lr_species = 0.001,
                  beta=1,
                  verbose=False,
                  random_seed=666, 
                  iter_comb=None, 
                  knn_neigh=100, 
                  device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'), 
                  pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                  mse_beta = 1, 
                  tri_beta = 1, 
                  mmd_beta = 1, 
                  gan_beta = 1,
                  gan_epoch = 1,
                  ot_beta = 0,
                  mmd_batch_size = 2048, 
                  if_knn_mnn_graph = True, 
                  if_integrate_within_species = False, 
                  if_return_loss = False, 
                  if_batch_pretrain = False, 
                  batch_size_dict = {'Mouse': 20000, 'Marmoset':12000, 'Macaque':4096}, 
                  batch_size = 2048, 
                  concate_pca_dim = None,
                  umap_downsampling_rate = 0.1, 
                  adata_whole=None):
    """\
    Train graph attention auto-encoder and use spot triplets across species to perform batch correction in the embedding space.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    margin
        Margin is used in triplet loss to enforce the distance between positive and negative pairs.
        Larger values result in more aggressive correction.
    iter_comb
        For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
        For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    knn_neigh
        The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
    device
        See torch.device.

    Returns
    -------
    AnnData dict, storing AnnData for each species
    """

    import gc
    # 初始清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    #torch.use_deterministic_algorithms(True)
    if not isinstance(stagate_epoch, dict):
        stagate_epoch_dict = {k:stagate_epoch for k in adata_species_dict.keys()}
    else:
        stagate_epoch_dict = stagate_epoch


    # Initializing the hidden units, model and optimizer dicts
    z_dict = {k:0 for k in adata_species_dict.keys()}
    # Train a STAGATE model for each species
    species_order = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.uns['highly_variable']]
        print(f'For {species_id}, using {len(adata.var_names)} genes for training.')
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    prune_edge_index=torch.LongTensor(np.array([])),
                    x=torch.FloatTensor(adata.X.todense()))

        # If use mini-batch training 
        if if_batch_pretrain:
            if species_order == 0:
                model = STACAME.STACAME_minibatch(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)
            # Minibatch
            train_loader = NeighborSampler(data.edge_index, node_idx=torch.LongTensor(np.array([i for i in range(adata.n_obs)])),
                                   sizes=[8, 4], batch_size=batch_size_dict[species_id], shuffle=True, drop_last = True)
            subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=batch_size_dict[species_id],
                                              shuffle=False)
        
            print('Pretrain with STAGATE (Minibatch)...')
            for epoch in tqdm(range(stagate_epoch_dict[species_id])):
                total_loss = 0
                for batchsize, n_id, adjs in train_loader:
                    adjs = [adj.to(pretrain_device) for adj in adjs]
                    optimizer.zero_grad()
                    z_batch, out_batch = model(data.x[n_id, :].to(pretrain_device), adjs, mode='batch')
                    # get batch data
                    x_batch = data.x[n_id,:].to(pretrain_device)                        
                    n_id_list = n_id.cpu().detach().numpy()
                    batch_id_list = adata.obs['batch_name'][n_id_list]
                    x_batch_cpu = z_batch.cpu().detach().numpy()
                    x_batch_adata = ad.AnnData(X=x_batch_cpu, obs=pd.DataFrame({"batch_name": batch_id_list}))
                    x_batch_adata.obsm['STAGATE'] = x_batch_cpu
                    section_ids = np.array(x_batch_adata.obs['batch_name'].unique())
                    mnn_dict = create_dictionary_mnn(x_batch_adata, use_rep='STAGATE', batch_name='batch_name', k=knn_neigh,
                                                       iter_comb=iter_comb, verbose=0)
                    
                    anchor_ind = []
                    positive_ind = []
                    negative_ind = []
                    for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                        batchname_list = x_batch_adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                        cellname_by_batch_dict = dict()
                        for batch_id in range(len(section_ids)):
                            cellname_by_batch_dict[section_ids[batch_id]] = x_batch_adata.obs_names[
                                x_batch_adata.obs['batch_name'] == section_ids[batch_id]].values
        
                        anchor_list = []
                        positive_list = []
                        negative_list = []
                        for anchor in mnn_dict[batch_pair].keys():
                            anchor_list.append(anchor)
                            ## np.random.choice(mnn_dict[batch_pair][anchor])
                            positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(
                                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
        
                        batch_as_dict = dict(zip(list(x_batch_adata.obs_names), range(0, x_batch_adata.shape[0])))
                        anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                        positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                        negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
    
                    anchor_arr = z_batch[anchor_ind,]
                    positive_arr = z_batch[positive_ind,]
                    negative_arr = z_batch[negative_ind,]
    
                    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                    tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
                    loss = F.mse_loss(out_batch, x_batch) + beta*tri_output
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()
                    total_loss += loss.item()
                #print(f'Epoch = {epoch}, mse loss = {total_loss/len(train_loader)}')
    
            with torch.no_grad():
                z_list = []
                out_list = []
                for batch in subgraph_loader:
                    batch.to(pretrain_device)
                    z, out = model(batch.x, batch.edge_index, mode='all')
                    z_list.append(z[:batch.batch_size].cpu())
                    out_list.append(out[:batch.batch_size].cpu())
            
            z_all = torch.cat(z_list, dim=0)
            out_all = torch.cat(out_list, dim=0)
            adata_species_dict[species_id].obsm['STAGATE'] = z_all.cpu().detach().numpy()
            z_dict[species_id] = z_all.cpu().detach()

            if species_order >= len(adata_species_dict.keys()):
                del model, optimizer, data, z_batch, out_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Not use minibatch training
        else:
            print('Pretrain with STAligner...')
            if species_order == 0:
                model = STACAME.STACAME(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)
            species_order += 1
            print('Pretrain with STAGATE_multiple...')
            for epoch in tqdm(range(0,  stagate_epoch_dict[species_id])):
                model.train()
                optimizer.zero_grad()
                z, out = model(data.x.to(pretrain_device), data.edge_index.to(pretrain_device))

                if epoch % 10 == 0 and epoch >= stagate_epoch_dict[species_id]//2:
                    adata.obsm['STAGATE'] = z.cpu().detach().numpy()
                    mnn_dict = create_dictionary_mnn(adata, use_rep='STAGATE', batch_name='batch_name', k=knn_neigh,
                                                               iter_comb=iter_comb, verbose=0)
                    anchor_ind = []
                    positive_ind = []
                    negative_ind = []
                    for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                        batchname_list = adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                        cellname_by_batch_dict = dict()
                        for batch_id in range(len(section_ids)):
                            cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                                adata.obs['batch_name'] == section_ids[batch_id]].values
        
                        anchor_list = []
                        positive_list = []
                        negative_list = []
                        for anchor in mnn_dict[batch_pair].keys():
                            anchor_list.append(anchor)
                            ## np.random.choice(mnn_dict[batch_pair][anchor])
                            positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(
                                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
        
                        batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                        anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                        positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                        negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
                
                    anchor_arr = z[anchor_ind,]
                    positive_arr = z[positive_ind,]
                    negative_arr = z[negative_ind,]
    
                    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                    tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
                    loss = F.mse_loss(data.x.to(pretrain_device), out) + beta * tri_output
                else:
                    loss = F.mse_loss(data.x.to(pretrain_device), out)
                loss.backward(retain_graph=True)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            #print(f'mse loss = {loss.item()}')
            with torch.no_grad():
                z, _ = model(data.x.to(pretrain_device), data.edge_index.to(pretrain_device))
    
            adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
            z_dict[species_id] = z.cpu().detach()

    #if species_order >= len(adata_species_dict.keys()):
    del model, optimizer, data, z, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    print('-------------------------------------------------------------------------------')
    print('Train with STACAME...')
    anchor_ind_species = triplet_ind_species_dict['anchor_ind_species']
    positive_ind_species = triplet_ind_species_dict['positive_ind_species']
    negative_ind_species = triplet_ind_species_dict['negative_ind_species']
    ##########################################-Cross Species####################################################
    k_add = 0
    species_add_dict = {k:None for k in z_dict.keys()}
    for species_id in z_dict.keys():
        species_add_dict[species_id] = int(k_add)
        k_add = int(k_add+adata_species_dict[species_id].n_obs)
    
    adata = adata_species_dict[list(adata_species_dict.keys())[0]]
    edgeList = adata.uns['edgeList']
    edge_ndarray = np.array([edgeList[0], edgeList[1]])
    #if verbose:
    S = 0
    for species_id, adata in adata_species_dict.items():
        section_ids = np.array(adata.obs['batch_name'].unique())
        edgeList = adata.uns['edgeList']
        if S != 0:
            edge_arr_temp = np.array([edgeList[0], edgeList[1]]) + species_add_dict[species_id]
            edge_ndarray = np.concatenate((edge_ndarray, edge_arr_temp), axis=1)
        else:
            S = S + 1
    edge_ndarray_species = np.array([edge_ndarray_species[0], edge_ndarray_species[1]])
    
    # Choose whether build the knn and mnn cross species neigbors into the graph
    if if_knn_mnn_graph == True:
        edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_species), axis=1)
    S = 0
    for species_id, z_input in z_dict.items():
        if S==0:
            X = z_dict[species_id].cpu().detach().numpy()
        else:
            X = np.concatenate((X, z_dict[species_id].cpu().detach().numpy()), axis=0)
        S = S + 1
    # ---------------------------Init graph-----------------------------
    print('Pretrain with STAGATE_multiple...')
    #----------------------------Create model------------------------------
    print('Train with cross species STACAME...')
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    
    z = torch.FloatTensor(X)
    # Merge X and get cross-species features
    #-------------------------------------------------------
    species_list = list(adata_species_dict.keys())
    n_species = len(species_list)
    ref_species = species_list[0]
    n_homo_genes = len(adata_species_dict[ref_species].uns['homo_highly_variable'])
    species_specific_n_genes = {
        sp: len(adata_species_dict[sp].uns['species_specific'])
        for sp in species_list
    }
    max_specific_genes = max(species_specific_n_genes.values())
    total_cols = n_homo_genes + max_specific_genes * n_species
    merge_X = None
    #merge_X_count = None
    mask_matrix = None
    for sp_idx, species_id in enumerate(species_list):
        adata = adata_species_dict[species_id]
        homo_genes = adata.uns['homo_highly_variable']
        x_homo = adata[:, homo_genes].X.todense()  # shape: (n_cells, n_homo_genes)
        #x_count_homo = adata.obsm['counts_hvg_share'].todense()
        specific_genes = adata.uns['species_specific']
        x_specific = adata[:, specific_genes].X.todense()  # shape: (n_cells, n_specific_genes)
        #x_count_specific = adata.obsm['counts_hvg_specific'].todense()
        n_cells = x_homo.shape[0]
        x_current = np.zeros((n_cells, total_cols))
        #x_count_current = np.zeros((n_cells, total_cols))
        mask_current = np.zeros((n_cells, total_cols))  
       
        x_current[:, :n_homo_genes] = x_homo
        #x_count_current[:, :n_homo_genes] = x_count_homo
        mask_current[:, :n_homo_genes] = 1 
        specific_start_col = n_homo_genes + sp_idx * max_specific_genes
        specific_end_col = specific_start_col + species_specific_n_genes[species_id]

        x_current[:, specific_start_col:specific_end_col] = x_specific
        #x_count_current[:, specific_start_col:specific_end_col] = x_count_specific
        mask_current[:, specific_start_col:specific_end_col] = 1  # 特异区域mask为1
        if merge_X is None:
            merge_X = x_current
            #merge_X_count = x_count_current
            mask_matrix = mask_current
        else:
            merge_X = np.concatenate((merge_X, x_current), axis=0)
            #merge_X_count = np.concatenate((merge_X_count, x_count_current), axis=0)
            mask_matrix = np.concatenate((mask_matrix, mask_current), axis=0)


    if concate_pca_dim != None:
        adata_X = ad.AnnData(merge_X)
        sc.pp.scale(adata_X)
        sc.tl.pca(adata_X, n_comps=concate_pca_dim)
        merge_X = adata_X.obsm["X_pca"]
    merge_X = torch.FloatTensor(merge_X).to(device)

    if hasattr(adata_whole.obsm['X_pca'], 'todense'):
        auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'].todense())
    else:
        auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'])

    ##-----------------------------------------------------------
    model = STACAME.STACAMEDecoder_minibatch(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_species, weight_decay=weight_decay, foreach=False)
    
    if if_integrate_within_species == True:
        anchor_ind_sections = triplet_ind_sections_dict['anchor_ind_sections']
        positive_ind_sections = triplet_ind_sections_dict['positive_ind_sections']
        negative_ind_sections = triplet_ind_sections_dict['negative_ind_sections']
        if if_knn_mnn_graph == True:
            edge_ndarray_sections = np.array([edge_ndarray_sections[0], edge_ndarray_sections[1]])
            edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_sections), axis=1)
            data = Data(edge_index=torch.LongTensor(edge_ndarray),
                    prune_edge_index=torch.LongTensor(np.array([])), x=z)
            data = data.to(device)
        else:
            data = Data(edge_index=torch.LongTensor(edge_ndarray),
                prune_edge_index=torch.LongTensor(np.array([])), x=z)
            data = data.to(device)
    else:
        data = Data(edge_index=torch.LongTensor(edge_ndarray),
            prune_edge_index=torch.LongTensor(np.array([])), x=z)
        data = data.to(device)

    for spe, adata in adata_species_dict.items():
        print(spe, adata.n_obs)

    id_species_dict = {k:None for k in range(merge_X.shape[0])}
    k_add = 0
    for spe_id in adata_species_dict.keys():
        for id_s in range(k_add, k_add + adata_species_dict[spe_id].n_obs):
            id_species_dict[id_s] = spe_id
        k_add = k_add + adata_species_dict[spe_id].n_obs #species_add_dict[spe_id]

    # The predict data loader, where the number 5 should be adjusted depended on the GPU memory
    subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=batch_size*5, shuffle=False)

    if if_return_loss:
        loss_dict = {'Loss name':[], 'Epoch':[], 'Loss value':[]}

    plot_epoch = n_epochs_species // 3

    D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    optimizer_D = torch.optim.Adam(list(D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)
    D_Z.train()

    species_list = []
    #celltype_all_list = []
    for species_id, adata in adata_species_dict.items():
        species_list = species_list + [species_id] * adata.n_obs

    true_dom = torch.LongTensor(pd.Series(species_list).astype('category').cat.codes.values)#.to(device)  

    ite_N = max(int(0.5*(len(anchor_ind_species) // batch_size)), 1)
    print('ite_N', ite_N)
    
    for epoch in tqdm(range(0, n_epochs_species)):
        # subsampling anchor, positive, and negative
        k_add = 0
        for species_id in z_dict.keys():
            adata_species_dict[species_id].obsm['STAGATE'] = z[k_add:int(k_add+adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
            z_dict[species_id] = adata_species_dict[species_id].obsm['STAGATE']
            k_add = int(k_add + adata_species_dict[species_id].n_obs)
      
        mse_loss_mean = 0
        mse_loss = 0
        tri_output_species_mean = 0
        tri_output_slice_mean = 0
        mmd_loss_sum_mean = 0
        batch_num = 0
        species_id_list = list(adata_species_dict.keys())

        for ite_ in range(ite_N):
            triples_N = len(anchor_ind_species)
            tri_ind_list = random.sample(list(range(triples_N)), batch_size)

            anchor_ind_species_batch = [anchor_ind_species[x] for x in tri_ind_list]
            positive_ind_species_batch = [positive_ind_species[x] for x in tri_ind_list]
            negative_ind_species_batch = [negative_ind_species[x] for x in tri_ind_list]

            ind_list_init = list(set(anchor_ind_species_batch + positive_ind_species_batch + negative_ind_species_batch))

            if if_integrate_within_species == True:
                triples_N_sec = len(anchor_ind_sections)
                tri_ind_list_sec = random.sample(list(range(triples_N_sec)), batch_size)

                anchor_ind_sections_batch = [anchor_ind_sections[x] for x in tri_ind_list_sec]
                positive_ind_sections_batch = [positive_ind_sections[x] for x in tri_ind_list_sec]
                negative_ind_sections_batch = [negative_ind_sections[x] for x in tri_ind_list_sec]
                
                ind_list_init = list(set(ind_list_init + list(set(anchor_ind_sections_batch + positive_ind_sections_batch + negative_ind_sections_batch))))

            idx_subset, edge_index_batch, mapping, edge_mask = k_hop_subgraph(node_idx = torch.LongTensor(ind_list_init), num_hops = 1, edge_index=data.edge_index,   relabel_nodes=True)

            idx_subset_list = [int(x) for x in idx_subset]
            idx_map = {k:v for k,v in zip(idx_subset_list, range(len(idx_subset_list)))}

            model.train()
            optimizer.zero_grad()
            z_batch, out = model(data.x[idx_subset_list,].to(device), edge_index_batch.to(device), mode='whole')
            mse_loss = F.mse_loss(merge_X[idx_subset_list,].to(device), out)
            if if_integrate_within_species == True:
                anchor_arr = z_batch[[idx_map[x] for x in anchor_ind_sections_batch],]
                positive_arr = z_batch[[idx_map[x] for x in positive_ind_sections_batch],]
                negative_arr = z_batch[[idx_map[x] for x in negative_ind_sections_batch],]
                triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
            else:
                pass
            anchor_arr_species = z_batch[[idx_map[x] for x in anchor_ind_species_batch],]
            positive_arr_species = z_batch[[idx_map[x] for x in positive_ind_species_batch],]
            negative_arr_species = z_batch[[idx_map[x] for x in negative_ind_species_batch],]
            triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin_species, p=2, reduction='mean')
            tri_output_species = triplet_loss_species(anchor_arr_species, positive_arr_species, negative_arr_species)

            z_ind_species_dict = {k: [] for k in adata_species_dict.keys()}
            for n_id_temp in idx_subset_list:
                n_id_species = id_species_dict[n_id_temp]
                z_ind_species_dict[n_id_species].append(n_id_temp)
            
            mmd_loss = STACAME.MMDLoss(kernel=STACAME.RBF(device=device), device=device).to(device)
            mmd_loss_sum = 0
            
            
            spe_id = random.sample(species_id_list, 1)[0]
            spe_id_list = [idx_map[x] for x in z_ind_species_dict[spe_id]]

            bsize = min(len(spe_id_list), len(idx_subset_list) - len(spe_id_list))
            
            z_A = z_batch[spe_id_list[0:bsize],]
            z_B_ind_list = random.sample(list(set(range(0, len(idx_subset_list))) - set(spe_id_list)), bsize)    
            z_B = z_batch[z_B_ind_list,]

            x_batch = auxiliary_X[idx_subset_list,].to(device)
            x_A = x_batch[spe_id_list[0:bsize],]
            x_B_ind_list = random.sample(list(set(range(0, len(idx_subset_list))) - set(spe_id_list)), bsize)    
            x_B = x_batch[x_B_ind_list,]

            anchor_arr_species_X = x_batch[[idx_map[x] for x in anchor_ind_species_batch],]
            positive_arr_species_X = x_batch[[idx_map[x] for x in positive_ind_species_batch],]

            if gan_beta != 0:
                for _ in range(gan_epoch):
                    optimizer_D.zero_grad()
                    logits_D = D_Z(z_batch) 
                    loss_D = F.cross_entropy(logits_D, true_dom[idx_subset_list, ].to(device))
                    loss_D.backward(retain_graph=True)
                    optimizer_D.step()
                
            loss_G_GAN = -F.cross_entropy(D_Z(z_batch) , true_dom[idx_subset_list, ].to(device))
            mmd_loss_sum = mmd_loss(z_A[0:mmd_batch_size, :], z_B[0:mmd_batch_size, :]).to(device)

            c_cross = pairwise_correlation_distance(x_A[0:mmd_batch_size, :].detach(), x_B[0:mmd_batch_size, :].detach()).to(device)
            T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=device)

            loss_ot = 0
            loss_ot_value = 0
            if ot_beta != 0:
                c_cross = pairwise_correlation_distance(anchor_arr_species_X.detach(), positive_arr_species_X.detach()).to(device)
                T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=device)
                z_dist = torch.mean((anchor_arr_species.view(anchor_arr_species_X.shape[0], 1, -1) - positive_arr_species.view(1, anchor_arr_species_X.shape[0], -1)) ** 2, dim=2)
                loss_ot = torch.sum(T * z_dist) / torch.sum(T)

                loss_ot_value = loss_ot.item()

            if if_integrate_within_species == True:     
                loss =  mse_beta * mse_loss + tri_beta * tri_output_species + beta * tri_output  + mmd_beta * mmd_loss_sum +  gan_beta * loss_G_GAN + ot_beta * loss_ot
            else:
                loss =  mse_beta * mse_loss + tri_beta * tri_output_species + mmd_beta * mmd_loss_sum + gan_beta * loss_G_GAN + ot_beta * loss_ot

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            

        if if_return_loss:
            loss_dict['Loss name'].append('Loss sum')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(loss.item())
            loss_dict['Loss name'].append('MSE')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mse_loss.item())
            loss_dict['Loss name'].append('Cross-species triplet')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(tri_output_species.item())
            loss_dict['Loss name'].append('MMD')
            loss_dict['Epoch'].append(epoch)
            loss_dict['Loss value'].append(mmd_loss_sum.item())
        if verbose == True and epoch % 100 == 0:
            print(f'---------------------------------Epoch {epoch}-----------------------------------')
            print(f'Total loss: {loss.item()}')
            print(
                f'MSE:{mse_beta * mse_loss.item()}, Cross species triplets:{tri_beta * tri_output_species.item()}, '
                f'MMD:{mmd_beta * mmd_loss_sum.item()}, GAN: {gan_beta * loss_G_GAN.item()}, OT: {ot_beta * loss_ot_value}')
            if if_integrate_within_species == True:
                print(f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(len(anchor_arr_species)).to(device)).item()}, Cross slices triplets: {tri_output.item()}')
                if if_return_loss:
                    loss_dict['Loss name'].append('Cross-slices triplet')
                    loss_dict['Epoch'].append(epoch)
                    loss_dict['Loss value'].append(tri_output.item())
                else:
                    print(f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(len(anchor_arr_species)).to(device)).item()}')

        with torch.no_grad():
            z_list = []
            out_list = []
            for batch in subgraph_loader:
                batch.to(device)
                z, out = model(batch.x, batch.edge_index, mode='all')
                z_list.append(z[:batch.batch_size].cpu())
                out_list.append(out[:batch.batch_size].cpu())
        
        z = torch.cat(z_list, dim=0)
        out_all = torch.cat(out_list, dim=0)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[k_add:int(k_add+adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()

        if epoch % plot_epoch == 0 and n_epochs_species - epoch >= plot_epoch:
            if z.shape[0] >= 50000:
                clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=umap_downsampling_rate)
            else:
                clustering_umap(adata_species_dict, key_umap=key_added)

    print('Clustering and UMAP of Cross Species STACAME:')
    if z.shape[0] >= 50000:
        clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=umap_downsampling_rate)
    else:
        clustering_umap(adata_species_dict, key_umap=key_added)


    del model, optimizer, D_Z, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict


