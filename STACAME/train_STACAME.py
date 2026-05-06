
from .STACAME import STACAME
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
from STACAME import create_dictionary_mnn
from STACAME import STACAME
import scanpy as sc
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import k_hop_subgraph
from math import ceil
import anndata as ad
from collections import Counter
from STACAME import STALIGNER
from .utils_OT import *
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import random
import os
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
            if 'annotation' in adata.obs:
                embedding_annotation = embedding_annotation + list(adata.obs['annotation'])
            
        k += 1
        # Visualize UMAP of each species
        sc.pp.neighbors(adata, n_neighbors=20, use_rep=key_umap, metric='cosine',  random_state=666)
        sc.tl.louvain(adata, random_state=666, key_added="louvain", resolution=0.5)
        sc.tl.umap(adata, min_dist=1, random_state=666)
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams["figure.figsize"] = (3, 3)
        plt.rcParams['font.size'] = 10

        
    adata_embedding = ad.AnnData(X = embedding_X, obs=embedding_obs_name)
    adata_embedding.obsm['spatial'] = embedding_spatial
    adata_embedding.obs['slice_name'] = embedding_slice_name
    adata_embedding.obs['batch_name'] = embedding_batch_name
    adata_embedding.obs['species_id'] = embedding_species_id
    if 'annotation' in adata.obs:
        adata_embedding.obs['annotation'] = embedding_annotation
    
    sc.pp.neighbors(adata_embedding,  n_neighbors=20, use_rep='X', metric='cosine',  random_state=666)
    sc.tl.louvain(adata_embedding, random_state=666, key_added="louvain", resolution=0.5)
    
    print(adata_embedding.X.shape)

    sc.tl.umap(adata_embedding, min_dist=1, random_state=666)

    species_ids = list(adata_dict.keys())
    
    species_color = ['#e64b35', '#4dbbd5', '#00a087', '#f39b7f', '#3c5488']#['#4778FA', '#8A1C62', '#ED7A43'] #['#ff7f0e', '#1f77b4']
    species_color_dict = dict(zip(species_ids, species_color))
    adata_embedding.uns['species_colors'] = [species_color_dict[x] for x in adata_embedding.obs.species_id]
    
    plt.rcParams['font.sans-serif'] = "Arial"
    #plt.rcParams["figure.figsize"] = (3, 3)
    plt.rcParams['font.size'] = 10
    if 'annotation' in adata.obs:
        sc.pl.umap(adata_embedding, color=['species_id', 'batch_name', 'louvain', 'annotation'], ncols=2, wspace=0.5, show=True)
    else:
        sc.pl.umap(adata_embedding, color=['species_id', 'batch_name', 'louvain'], ncols=2, wspace=0.5, show=True)

    if 'annotation' in adata.obs:
        fig, axes = plt.subplots(len(species_ids), 1,  figsize=(5, 4*len(species_ids))) #, dpi=500
        for i in range(len(species_ids)):
            species_id = species_ids[i]
            adata_mh = adata_embedding[adata_embedding.obs['species_id'].isin([species_id])]
            color_list = sns.color_palette(cc.glasbey, n_colors=len(adata_mh.obs['annotation'].unique()))
            region_list = [x for x in Counter(adata_mh.obs['annotation']).keys()]
            palette = {k:v for k,v in zip(region_list, color_list)}
            ax = sc.pl.umap(adata_embedding, show=False, ax=axes[i])
            sc.pl.umap(adata_mh, color='annotation', ax=ax,  wspace=0.5, show=False, legend_loc='on data', palette=palette)
        plt.show()
    
        fig, axes = plt.subplots(len(species_ids), 1,  figsize=(5, 4*len(species_ids))) #, dpi=500
        for i in range(len(species_ids)):
            species_id = species_ids[i]
            adata_mh = adata_embedding[adata_embedding.obs['species_id'].isin([species_id])]
            color_list = sns.color_palette(cc.glasbey, n_colors=len(adata_mh.obs['annotation'].unique()))
            region_list = [x for x in Counter(adata_mh.obs['annotation']).keys()]
            palette = {k:v for k,v in zip(region_list, color_list)}
            ax = sc.pl.umap(adata_embedding, show=False, ax=axes[i])
            sc.pl.umap(adata_mh, color='annotation', ax=ax,  wspace=0.5, show=False, legend_loc='right margin', palette=palette)
        plt.show()
    return None



def clustering_umap_downsampling(adata_dict, key_umap='STACAME', downsampling_rate = 0.1):
    
    k = 0
    for species_id, adata_ in adata_dict.items():
        adata = sc.pp.subsample(adata_, fraction=0.1, copy=True)
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
        # Visualize UMAP of each species
        
        sc.pp.neighbors(adata, n_neighbors=20, use_rep=key_umap, metric='cosine',  random_state=666)
        sc.tl.louvain(adata, random_state=666, key_added="louvain", resolution=0.5)
        sc.tl.umap(adata, min_dist=1, random_state=666)
        plt.rcParams['font.sans-serif'] = "Arial"
        plt.rcParams["figure.figsize"] = (3, 3)
        plt.rcParams['font.size'] = 10
        if 'annotation' in adata.obs:
            sc.pl.umap(adata, color=['batch_name', 'louvain', 'annotation'], ncols=3, wspace=0.7, show=True)
        else:
            sc.pl.umap(adata, color=['batch_name', 'louvain'], ncols=3, wspace=0.7, show=True)
            
    
    adata_embedding = ad.AnnData(X = embedding_X, obs=embedding_obs_name)
    adata_embedding.obsm['spatial'] = embedding_spatial
    adata_embedding.obs['slice_name'] = embedding_slice_name
    adata_embedding.obs['batch_name'] = embedding_batch_name
    adata_embedding.obs['species_id'] = embedding_species_id
    if 'annotation' in adata.obs and 'embedding_annotation' in locals():
        adata_embedding.obs['annotation'] = embedding_annotation
    
    sc.pp.neighbors(adata_embedding,  n_neighbors=20, use_rep='X', metric='cosine',  random_state=666)
    sc.tl.louvain(adata_embedding, random_state=666, key_added="louvain", resolution=0.5)
    
    print(adata_embedding.X.shape)

    sc.tl.umap(adata_embedding, min_dist=1, random_state=666)

    species_ids = list(adata_dict.keys())
    
    species_color = ['#e64b35', '#4dbbd5', '#00a087', '#f39b7f', '#3c5488']#['#4778FA', '#8A1C62', '#ED7A43'] #['#ff7f0e', '#1f77b4']
    species_color_dict = dict(zip(species_ids, species_color))
    adata_embedding.uns['species_colors'] = [species_color_dict[x] for x in adata_embedding.obs.species_id]
    
    plt.rcParams['font.sans-serif'] = "Arial"
    #plt.rcParams["figure.figsize"] = (3, 3)
    plt.rcParams['font.size'] = 10
    if 'annotation' in adata.obs and 'embedding_annotation' in locals():
        sc.pl.umap(adata_embedding, color=['species_id', 'batch_name', 'louvain', 'annotation'], ncols=2, wspace=0.5, show=True)
    else:
        sc.pl.umap(adata_embedding, color=['species_id', 'batch_name', 'louvain'], ncols=2, wspace=0.5, show=True)

    if 'annotation' in adata.obs and 'embedding_annotation' in locals():
        fig, axes = plt.subplots(len(species_ids), 1,  figsize=(5, 4*len(species_ids))) #, dpi=500
        for i in range(len(species_ids)):
            species_id = species_ids[i]
            adata_mh = adata_embedding[adata_embedding.obs['species_id'].isin([species_id])]
            color_list = sns.color_palette(cc.glasbey, n_colors=len(adata_mh.obs['annotation'].unique()))
            region_list = [x for x in Counter(adata_mh.obs['annotation']).keys()]
            palette = {k:v for k,v in zip(region_list, color_list)}
            ax = sc.pl.umap(adata_embedding, show=False, ax=axes[i])
            sc.pl.umap(adata_mh, color='annotation', ax=ax,  wspace=0.5, show=False, legend_loc='on data', palette=palette)
        plt.show()
    
        fig, axes = plt.subplots(len(species_ids), 1,  figsize=(5, 4*len(species_ids))) #, dpi=500
        for i in range(len(species_ids)):
            species_id = species_ids[i]
            adata_mh = adata_embedding[adata_embedding.obs['species_id'].isin([species_id])]
            color_list = sns.color_palette(cc.glasbey, n_colors=len(adata_mh.obs['annotation'].unique()))
            region_list = [x for x in Counter(adata_mh.obs['annotation']).keys()]
            palette = {k:v for k,v in zip(region_list, color_list)}
            ax = sc.pl.umap(adata_embedding, show=False, ax=axes[i])
            sc.pl.umap(adata_mh, color='annotation', ax=ax,  wspace=0.5, show=False, legend_loc='right margin', palette=palette)
        plt.show()
    return None


## Light version of STACAME without GAN loss and auxiliary model
def train_STACAME(adata_species_dict, 
                  triplet_ind_species_dict, 
                  edge_ndarray_species,
                  triplet_ind_sections_dict = None, 
                  edge_ndarray_sections = None,
                  hidden_dims=[512, 30], 
                  stagate_epoch=500, 
                  n_epochs=1000,  
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
                  alpha=1, 
                  beta=1,
                  verbose=False,
                  random_seed=666, 
                  iter_comb=None, 
                  knn_neigh=100, 
                  device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'), 
                  pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                  mse_beta = 1, 
                  tri_beta = 1, 
                  mmd_beta = 2, 
                  mmd_batch_size = 4000, 
                  if_knn_mnn_graph = True, 
                  if_integrate_within_species = False, 
                  if_return_loss = False):
    """
    Train graph attention auto-encoder and utilize cross-species spot triplets
    for cross-species batch integration in latent embedding space.

    Parameters
    ----------
    adata_species_dict : dict
        Dictionary contains AnnData object of each species.
    triplet_ind_species_dict : dict
        Cross-species anchor/positive/negative triplet index.
    edge_ndarray_species : np.ndarray
        Cross-species MNN graph edge index.
    triplet_ind_sections_dict : dict, optional
        Within-species slice-level triplet index.
    edge_ndarray_sections : np.ndarray, optional
        Within-species slice MNN graph edge index.
    hidden_dims : list
        MLP hidden dimension of encoder and decoder.
    stagate_epoch : int | dict
        Pre-training epoch for single species STAGATE.
    n_epochs : int
        General training epoch (reserved).
    n_epochs_species : int
        Total epoch for cross-species STACAME training.
    lr : float
        Learning rate for pre-training.
    key_added : str
        Key name for storing final embedding in obsm[key_added].
    gradient_clipping : float
        Max norm for gradient clipping.
    weight_decay : float
        Weight decay for Adam optimizer.
    margin : float
        Triplet loss margin for within-species constraint.
    margin_species : float
        Triplet loss margin for cross-species constraint.
    lr_species : float
        Learning rate for cross-species joint training.
    alpha, beta : float
        Loss weight coefficients.
    verbose : bool
        Whether print training log every 100 epochs.
    random_seed : int
        Global random seed for reproducibility.
    iter_comb : tuple | None
        Slice integration order for multi-batch alignment.
    knn_neigh : int
        KNN neighbor number for MNN construction.
    device : torch.device
        Main GPU device for cross-species training.
    pretrain_device : torch.device
        GPU device for single species pre-training.
    mse_beta, tri_beta, mmd_beta, gan_beta, ot_beta : float
        Weight for each loss component.
    gan_epoch : int
        Iteration number for discriminator update in GAN.
    mmd_batch_size : int
        Batch size for MMD calculation.
    if_knn_mnn_graph : bool
        Whether add cross-species MNN edges to global graph.
    if_integrate_within_species : bool
        Whether enable within-species slice integration.
    if_return_loss : bool
        Whether return loss record dictionary.
    adata_whole : AnnData
        Whole concatenated AnnData for cross-species MNN search.
    concate_pca_dim : int
        PCA dimension for concatenated raw feature input.

    Returns
    -------
    adata_species_dict : dict
        Updated dict with STACAME embedding in obsm[key_added].
    loss_dict : dict, optional
        Loss record, returned if if_return_loss=True.
    """

    # seed_everything()
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
        data = data.to(pretrain_device)

        if species_order == 0:
            model = STACAME.STACAME(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        print(f'mse loss = {loss.item()}')
        with torch.no_grad():
            z, _ = model(data.x, data.edge_index)
        # Update the save hidden units
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
    ##########################################-Cross Species####################################################
    k_add = 0
    species_add_dict = {k:None for k in z_dict.keys()}
    for species_id in z_dict.keys():
        species_add_dict[species_id] = int(k_add)
        k_add = int(k_add+adata_species_dict[species_id].n_obs)
    
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
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduction='mean')
    L1_loss = torch.nn.HuberLoss()
    z = torch.FloatTensor(X)
    # Merge X and get cross-species features
    #-------------------------------------------------------
    # 记录每个物种的基因位置，用于后续拆分
    species_gene_ranges = {}
    current_gene_idx = 0
    S = 0
    for species_id, adata in adata_species_dict.items():
        if 'highly_variable' in adata.var.columns:
            sub_adata = adata[:, adata.uns['highly_variable']]
        else:
            sub_adata = adata
        
        x = sub_adata.X.todense()
        n_genes = x.shape[1]
        species_gene_ranges[species_id] = (current_gene_idx, current_gene_idx + n_genes)
        current_gene_idx += n_genes

        if S != 0:
            merge_X = np.concatenate((merge_X, x), axis=0)
        else:
            merge_X = x
            S = S + 1
    merge_X = torch.FloatTensor(merge_X).to(device)
    ##-----------------------------------------------------------
    model = STACAME.STACAME_Decoder(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_species, weight_decay=weight_decay)

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

    if if_return_loss:
        loss_dict = {'Loss name':[], 'Epoch':[], 'Loss value':[]}

    plot_epoch = n_epochs_species // 3
    scheduler = StepLR(optimizer, step_size=1000, gamma=1)
    
    for epoch in tqdm(range(0, n_epochs_species)):
        current_seed = random_seed + epoch
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        # subsampling anchor, positive, and negative
        k_add = 0
        for species_id in z_dict.keys():
            species_add_dict[species_id] = int(k_add)
            adata_species_dict[species_id].obsm['STAGATE'] = z[k_add:int(k_add+adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
            z_dict[species_id] = adata_species_dict[species_id].obsm['STAGATE']
            k_add = int(k_add + adata_species_dict[species_id].n_obs)

        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        mse_loss = F.mse_loss(merge_X, out)
        if if_integrate_within_species == True:
            anchor_arr = z[anchor_ind_sections,]
            positive_arr = z[positive_ind_sections,]
            negative_arr = z[negative_ind_sections,]
            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        else:
            pass
        anchor_arr_species = z[anchor_ind_species,]
        positive_arr_species = z[positive_ind_species,]
        negative_arr_species = z[negative_ind_species,]
        triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin_species, p=2, reduction='mean')
        tri_output_species = triplet_loss_species(anchor_arr_species, positive_arr_species, negative_arr_species)
        mmd_loss = STACAME.MMDLoss(kernel=STACAME.RBF(device=device), device=device).to(device)
        mmd_loss_sum = 0
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            remain_list = list(set(list(range(z.shape[0]))) - set(range(k_add,int(k_add+adata_species_dict[species_id].n_obs))))
            random.seed(epoch)
            ind_1 = random.sample(list(range(k_add, int(k_add+adata_species_dict[species_id].n_obs))), mmd_batch_size)
            ind_2 = random.sample(remain_list, mmd_batch_size)
            mmd_loss_sum = mmd_loss_sum +  mmd_loss(z[ind_1, ].to(device), z[ind_2, ].to(device)).to(device)
        sampling_num_spe = anchor_arr_species.shape[0]

        if if_integrate_within_species == True:     
            loss =  mse_beta * mse_loss  + tri_beta * tri_output_species + beta * tri_output + mmd_beta * mmd_loss_sum
        else:
            loss =  mse_beta * mse_loss  + tri_beta * tri_output_species + mmd_beta * mmd_loss_sum
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

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
                f'MMD:{mmd_beta * mmd_loss_sum.item()}')
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}, LR: {current_lr:.6f}")
            if if_integrate_within_species == True:
                print(f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(sampling_num_spe).to(device)).item()}, Cross slices triplets: {tri_output.item()}')
                if if_return_loss:
                    loss_dict['Loss name'].append('Cross-slices triplet')
                    loss_dict['Epoch'].append(epoch)
                    loss_dict['Loss value'].append(tri_output.item())
            else:
                print(f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(sampling_num_spe).to(device)).item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[k_add:int(k_add+adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
        if epoch % plot_epoch == 0 and n_epochs_species - epoch >= plot_epoch:
            if z.shape[0] >= 50000:
                clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate = 0.1)
            else:
                clustering_umap(adata_species_dict, key_umap=key_added)

    print('Clustering and UMAP of Cross Species STACAME:')
    if z.shape[0] >= 50000:
        clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate = 0.1)
    else:
        clustering_umap(adata_species_dict, key_umap=key_added)

    # ===================== Save results =====================
    with torch.no_grad():
        z_final, out_final = model(data.x, data.edge_index)
        out_np = out_final.cpu().detach().numpy()
    # ==================================================================================
    del model,  optimizer, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict


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
                      device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),
                      pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                      mse_beta=1,
                      tri_beta=1,
                      mmd_beta=1,
                      gan_beta=1,
                      gan_epoch=3,
                      ot_beta=0,
                      mmd_batch_size=1024,
                      if_knn_mnn_graph=True,
                      if_integrate_within_species=False,
                      if_return_loss=False,
                      adata_whole=None, concate_pca_dim=40):
    """
    Train graph attention auto-encoder and utilize cross-species spot triplets
    for cross-species batch integration in latent embedding space.

    Parameters
    ----------
    adata_species_dict : dict
        Dictionary contains AnnData object of each species.
    triplet_ind_species_dict : dict
        Cross-species anchor/positive/negative triplet index.
    edge_ndarray_species : np.ndarray
        Cross-species MNN graph edge index.
    triplet_ind_sections_dict : dict, optional
        Within-species slice-level triplet index.
    edge_ndarray_sections : np.ndarray, optional
        Within-species slice MNN graph edge index.
    hidden_dims : list
        MLP hidden dimension of encoder and decoder.
    stagate_epoch : int | dict
        Pre-training epoch for single species STAGATE.
    n_epochs_species : int
        Total epoch for cross-species STACAME training.
    lr : float
        Learning rate for pre-training.
    key_added : str
        Key name for storing final embedding in obsm[key_added].
    gradient_clipping : float
        Max norm for gradient clipping.
    weight_decay : float
        Weight decay for Adam optimizer.
    margin : float
        Triplet loss margin for within-species constraint.
    margin_species : float
        Triplet loss margin for cross-species constraint.
    lr_species : float
        Learning rate for cross-species joint training.
    alpha, beta : float
        Loss weight coefficients.
    verbose : bool
        Whether print training log every 100 epochs.
    random_seed : int
        Global random seed for reproducibility.
    iter_comb : tuple | None
        Slice integration order for multi-batch alignment.
    knn_neigh : int
        KNN neighbor number for MNN construction.
    device : torch.device
        Main GPU device for cross-species training.
    pretrain_device : torch.device
        GPU device for single species pre-training.
    mse_beta, tri_beta, mmd_beta, gan_beta, ot_beta : float
        Weight for each loss component.
    gan_epoch : int
        Iteration number for discriminator update in GAN.
    mmd_batch_size : int
        Batch size for MMD calculation.
    if_knn_mnn_graph : bool
        Whether add cross-species MNN edges to global graph.
    if_integrate_within_species : bool
        Whether enable within-species slice integration.
    if_return_loss : bool
        Whether return loss record dictionary.
    adata_whole : AnnData
        Whole concatenated AnnData for cross-species MNN search.
    concate_pca_dim : int
        PCA dimension for concatenated raw feature input.

    Returns
    -------
    adata_species_dict : dict
        Updated dict with STACAME embedding in obsm[key_added].
    loss_dict : dict, optional
        Loss record, returned if if_return_loss=True.
    """
    import gc
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.use_deterministic_algorithms(True)

    if not isinstance(stagate_epoch, dict):
        stagate_epoch_dict = {k: stagate_epoch for k in adata_species_dict.keys()}
    else:
        stagate_epoch_dict = stagate_epoch

    # Initializing the hidden units, model and optimizer dicts
    z_dict = {k: 0 for k in adata_species_dict.keys()}
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
        data = data.to(pretrain_device)

        if species_order == 0:
            model = STACAME.STACAME(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        print(f'mse loss = {loss.item()}')
        with torch.no_grad():
            z, _ = model(data.x, data.edge_index)

        adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
        z_dict[species_id] = z.cpu().detach()

        if species_order >= len(adata_species_dict.keys()):
            del model, optimizer, data, z, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        # del model, optimizer, data, z, out
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # gc.collect()
        # ========================================================================

    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    print('-------------------------------------------------------------------------------')
    print('Train with STACAME...')
    anchor_ind_species = triplet_ind_species_dict['anchor_ind_species']
    positive_ind_species = triplet_ind_species_dict['positive_ind_species']
    negative_ind_species = triplet_ind_species_dict['negative_ind_species']
    ##########################################-Cross Species####################################################
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

    # Choose whether build the knn and mnn cross species neigbors into the graph
    if if_knn_mnn_graph == True:
        edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_species), axis=1)
    S = 0
    for species_id, z_input in z_dict.items():
        if S == 0:
            X = z_dict[species_id].cpu().detach().numpy()
        else:
            X = np.concatenate((X, z_dict[species_id].cpu().detach().numpy()), axis=0)
        S = S + 1
    # ---------------------------Init graph-----------------------------
    print('Pretrain with STAGATE_multiple...')
    # ----------------------------Create model------------------------------
    print('Train with cross species STACAME...')
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    z = torch.FloatTensor(X)
    # Merge X and get cross-species features
    # -------------------------------------------------------
    S = 0
    for species_id, adata in adata_species_dict.items():
        if S != 0:
            if 'highly_variable' in adata.var.columns:
                x = adata[:, adata.uns['highly_variable']].X.todense()
            else:
                x = adata.X.todense()
            merge_X = np.concatenate((merge_X, x), axis=0)
        else:
            if 'highly_variable' in adata.var.columns:
                merge_X = adata[:, adata.uns['highly_variable']].X.todense()
            else:
                merge_X = adata.X.todense()
            S = S + 1

    adata_X = ad.AnnData(merge_X)
    sc.pp.scale(adata_X)
    sc.tl.pca(adata_X, n_comps=concate_pca_dim)
    merge_X = adata_X.obsm["X_pca"]

    merge_X = torch.FloatTensor(merge_X).to(device)

    ##-----------------------------------------------------------
    species_ids = list(adata_species_dict.keys())
    model = STACAME.STACAME_Decoder(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)

    auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'])
    auxiliary_model = STACAME.STACAME(hidden_dims=[auxiliary_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(auxiliary_model.parameters()), lr=lr_species,
                                 weight_decay=weight_decay)

    auxiliary_D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    auxiliary_optimizer_D = torch.optim.Adam(list(auxiliary_D_Z.parameters()), lr=0.001, weight_decay=0.001)
    auxiliary_D_Z.train()

    D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    optimizer_D = torch.optim.Adam(list(D_Z.parameters()), lr=0.001, weight_decay=0.001)
    D_Z.train()

    species_list = []
    for species_id, adata in adata_species_dict.items():
        species_list = species_list + [species_id] * adata.n_obs

    true_dom = torch.LongTensor(pd.Series(species_list).astype('category').cat.codes.values).to(device)
    auxiliary_data = Data(edge_index=torch.LongTensor(edge_ndarray),
                          prune_edge_index=torch.LongTensor(np.array([])), x=auxiliary_X)
    auxiliary_data = auxiliary_data.to(device)

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

    if if_return_loss:
        loss_dict = {'Loss name': [], 'Epoch': [], 'Loss value': []}

    plot_epoch = n_epochs_species // 3

    for epoch in tqdm(range(0, n_epochs_species)):
        # subsampling anchor, positive, and negative
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

        mse_loss = F.mse_loss(merge_X, out) + F.mse_loss(auxiliary_data.x, auxiliary_out)
        if if_integrate_within_species == True:
            anchor_arr = z[anchor_ind_sections,]
            positive_arr = z[positive_ind_sections,]
            negative_arr = z[negative_ind_sections,]
            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        else:
            pass
        anchor_arr_species = z[anchor_ind_species,]
        positive_arr_species = z[positive_ind_species,]
        negative_arr_species = z[negative_ind_species,]
        triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin_species, p=2, reduction='mean')
        tri_output_species = triplet_loss_species(anchor_arr_species, positive_arr_species, negative_arr_species)
        mmd_loss = STACAME.MMDLoss(kernel=STACAME.RBF(device=device), device=device).to(device)
        mmd_loss_sum = 0
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            remain_list = list(
                set(list(range(z.shape[0]))) - set(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))))
            ind_1 = random.sample(list(range(k_add, int(k_add + adata_species_dict[species_id].n_obs))), mmd_batch_size)
            ind_2 = random.sample(remain_list, mmd_batch_size)
            mmd_loss_sum = mmd_loss_sum + mmd_loss(z[ind_1,].to(device), z[ind_2,].to(device)).to(device)
            mmd_loss_sum = mmd_loss_sum + mmd_loss(auxiliary_z[ind_1,].to(device), auxiliary_z[ind_2,].to(device)).to(
                device)

            loss_ot = 0
            if ot_beta != 0:
                z_A = z[ind_1,].to(device)
                z_B = z[ind_2,].to(device)
                x_A = auxiliary_X[ind_1,].to(device)
                x_B = auxiliary_X[ind_2,].to(device)
                c_cross = pairwise_correlation_distance(x_A.detach(), x_B.detach()).to(device)
                T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=device)

                z_dist = torch.mean((z_A.view(mmd_batch_size, 1, -1) - z_B.view(1, mmd_batch_size, -1)) ** 2, dim=2)
                loss_ot = torch.sum(T * z_dist) / torch.sum(T)

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

        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_auxiliary = triplet_loss(z[anchor_ind,], z[positive_ind,], z[negative_ind,]) + triplet_loss(
            auxiliary_z[anchor_ind,], auxiliary_z[positive_ind,], auxiliary_z[negative_ind,])

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
        if if_integrate_within_species == True:
            loss = mse_beta * mse_loss + tri_beta * (
                        tri_auxiliary + 0.1 * tri_output_species) + beta * tri_output + mmd_beta * mmd_loss_sum + gan_beta * loss_G_GAN + ot_beta * loss_ot
        else:
            loss = mse_beta * mse_loss + tri_beta * (
                        tri_auxiliary + 0.1 * tri_output_species) + mmd_beta * mmd_loss_sum + gan_beta * loss_G_GAN + ot_beta * loss_ot
        loss.backward(retain_graph=True)
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
            print(
                f'MSE loss:{mse_beta * mse_loss.item()},  Cross species triplets:{tri_beta * tri_output_species.item()}, MMD loss:{mmd_beta * mmd_loss_sum.item()}, GAN loss:{gan_beta * loss_G_GAN.item()}')
            if if_integrate_within_species == True:
                print(
                    f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(sampling_num_spe).to(device)).item()}, Cross slices triplets: {tri_output.item()}')
                if if_return_loss:
                    loss_dict['Loss name'].append('Cross-slices triplet')
                    loss_dict['Epoch'].append(epoch)
                    loss_dict['Loss value'].append(tri_output.item())
            else:
                print(
                    f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(sampling_num_spe).to(device)).item()}')

        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[
                k_add:int(k_add + adata_species_dict[species_id].n_obs), :].cpu().detach().numpy()
        adata_whole.obsm['auxiliary'] = auxiliary_z.cpu().detach().numpy()
        if epoch % plot_epoch == 0 and n_epochs_species - epoch >= plot_epoch:
            if z.shape[0] >= 50000:
                clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
            else:
                clustering_umap(adata_species_dict, key_umap=key_added)

    print('Clustering and UMAP of Cross Species STACAME:')
    if z.shape[0] >= 50000:
        clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate=0.1)
    else:
        clustering_umap(adata_species_dict, key_umap=key_added)

    del model, auxiliary_model, optimizer, D_Z, auxiliary_D_Z, data, auxiliary_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict



## Subgraph training of standard version of STACAME with GAN loss and auxiliary model
def train_STACAME_subgraph_auxiliary(adata_species_dict,
                                     triplet_ind_species_dict,
                                     edge_ndarray_species,
                                     triplet_ind_sections_dict=None,
                                     edge_ndarray_sections=None,
                                     hidden_dims=[512, 30],
                                     stagate_epoch=500,
                                     n_epochs=1000,
                                     n_epochs_species=2000,
                                     lr=0.001,
                                     key_added='STACAME',
                                     gradient_clipping=10.,
                                     weight_decay=0.0001,
                                     lr_wd=0.001,
                                     weight_decay_wd=5e-4,
                                     margin=1.0,
                                     margin_species=1.0,
                                     lr_species=0.001,
                                     alpha=1,
                                     beta=1,
                                     verbose=False,
                                     random_seed=666,
                                     iter_comb=None,
                                     knn_neigh=100,
                                     device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),
                                     pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                     mse_beta=1,
                                     tri_beta=1,
                                     mmd_beta=2,
                                     gan_beta=1,
                                     gan_epoch=1,
                                     ot_beta=0,
                                     mmd_batch_size=2048,
                                     if_knn_mnn_graph=True,
                                     if_integrate_within_species=False,
                                     if_return_loss=False,
                                     if_batch_pretrain=False,
                                     batch_size_dict={'Mouse': 20000, 'Marmoset': 12000, 'Macaque': 4096},
                                     batch_size=2048,
                                     concate_pca_dim=None,
                                     umap_downsampling_rate=0.1,
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
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # torch.use_deterministic_algorithms(True)
    if not isinstance(stagate_epoch, dict):
        stagate_epoch_dict = {k: stagate_epoch for k in adata_species_dict.keys()}
    else:
        stagate_epoch_dict = stagate_epoch

    # Initializing the hidden units, model and optimizer dicts
    z_dict = {k: 0 for k in adata_species_dict.keys()}
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
                model = STACAME.STACAME_minibatch_large(
                    hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            # Minibatch
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
                    # get batch data
                    x_batch = data.x[n_id, :].to(pretrain_device)
                    # if epoch % 2 == 0:
                    # n_id_batch = n_id[0:batchsize]

                    n_id_list = n_id.cpu().detach().numpy()
                    batch_id_list = adata.obs['batch_name'][n_id_list]
                    x_batch_cpu = z_batch.cpu().detach().numpy()
                    x_batch_adata = ad.AnnData(X=x_batch_cpu, obs=pd.DataFrame({"batch_name": batch_id_list}))
                    x_batch_adata.obsm['STAGATE'] = x_batch_cpu
                    section_ids = np.array(x_batch_adata.obs['batch_name'].unique())
                    mnn_dict = create_dictionary_mnn(x_batch_adata, use_rep='STAGATE', batch_name='batch_name',
                                                     k=knn_neigh,
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
                    loss = F.mse_loss(out_batch, x_batch) + beta * tri_output
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()
                    total_loss += loss.item()
                # print(f'Epoch = {epoch}, mse loss = {total_loss/len(train_loader)}')

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

            if species_order >= len(adata_species_dict.keys()):
                del model, optimizer, data, z_batch, out_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Not use minibatch training
        else:
            print('Pretrain with STAligner...')
            if species_order == 0:
                model = STACAME.STACAME(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(
                    pretrain_device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            species_order += 1
            print('Pretrain with STAGATE_multiple...')
            for epoch in tqdm(range(0, stagate_epoch_dict[species_id])):
                model.train()
                optimizer.zero_grad()
                z, out = model(data.x.to(pretrain_device), data.edge_index.to(pretrain_device))

                if epoch % 10 == 0 and epoch >= stagate_epoch_dict[species_id] // 2:
                    # if verbose:
                    #     print('Update spot triplets at epoch ' + str(epoch))
                    adata.obsm['STAGATE'] = z.cpu().detach().numpy()
                    # If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
                    # not all points have MNN achors
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
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            # print(f'mse loss = {loss.item()}')
            with torch.no_grad():
                z, _ = model(data.x.to(pretrain_device), data.edge_index.to(pretrain_device))

            adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
            z_dict[species_id] = z.cpu().detach()

    # if species_order >= len(adata_species_dict.keys()):
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
    species_add_dict = {k: None for k in z_dict.keys()}
    for species_id in z_dict.keys():
        species_add_dict[species_id] = int(k_add)
        k_add = int(k_add + adata_species_dict[species_id].n_obs)

    adata = adata_species_dict[list(adata_species_dict.keys())[0]]
    edgeList = adata.uns['edgeList']
    edge_ndarray = np.array([edgeList[0], edgeList[1]])
    # if verbose:
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
        if S == 0:
            X = z_dict[species_id].cpu().detach().numpy()
        else:
            X = np.concatenate((X, z_dict[species_id].cpu().detach().numpy()), axis=0)
        S = S + 1
    # ---------------------------Init graph-----------------------------
    print('Pretrain with STAGATE_multiple...')
    # ----------------------------Create model------------------------------
    print('Train with cross species STACAME...')
    cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=True)
    triplet_loss_species = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')

    z = torch.FloatTensor(X)
    # Merge X and get cross-species features
    # -------------------------------------------------------
    S = 0
    for species_id, adata in adata_species_dict.items():
        if S != 0:
            if 'highly_variable' in adata.var.columns:
                x = adata[:, adata.uns['highly_variable']].X.todense()
            else:
                x = adata.X.todense()
            merge_X = np.concatenate((merge_X, x), axis=0)
        else:
            if 'highly_variable' in adata.var.columns:
                merge_X = adata[:, adata.uns['highly_variable']].X.todense()
            else:
                merge_X = adata.X.todense()
            S = S + 1

    if hasattr(adata_whole.obsm['X_pca'], 'todense'):
        auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'].todense())
    else:
        auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'])

    if concate_pca_dim != None:
        adata_X = ad.AnnData(merge_X)
        sc.pp.scale(adata_X)
        sc.tl.pca(adata_X, n_comps=concate_pca_dim)
        merge_X = adata_X.obsm["X_pca"]  # shape: (n_cells, n_comps)
    merge_X = torch.FloatTensor(merge_X)
    ##-----------------------------------------------------------
    model = STACAME.STACAMEDecoder_minibatch(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)

    auxiliary_model = STACAME.STACAME_minibatch(
        hidden_dims=[auxiliary_X.shape[1], hidden_dims[0] // 2, hidden_dims[1]]).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(auxiliary_model.parameters()), lr=lr_species,
                                 weight_decay=weight_decay)

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

    # print('adata_species_dict.keys()', adata_species_dict.keys())
    for spe, adata in adata_species_dict.items():
        print(spe, adata.n_obs)

    id_species_dict = {k: None for k in range(merge_X.shape[0])}
    k_add = 0
    for spe_id in adata_species_dict.keys():
        for id_s in range(k_add, k_add + adata_species_dict[spe_id].n_obs):
            id_species_dict[id_s] = spe_id
        k_add = k_add + adata_species_dict[spe_id].n_obs  # species_add_dict[spe_id]

    # The predict data loader, where the number 5 should be adjusted depended on the GPU memory
    subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=batch_size * 2, shuffle=False)

    if if_return_loss:
        loss_dict = {'Loss name': [], 'Epoch': [], 'Loss value': []}

    plot_epoch = n_epochs_species // 3

    D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    optimizer_D = torch.optim.Adam(list(D_Z.parameters()), lr=0.001, weight_decay=0.001)
    D_Z.train()

    auxiliary_D_Z = STACAME.MultiClassDiscriminator(hidden_dims[1], len(adata_species_dict.keys())).to(device)
    auxiliary_optimizer_D = torch.optim.Adam(list(auxiliary_D_Z.parameters()), lr=0.001, weight_decay=0.001)
    auxiliary_D_Z.train()

    species_list = []
    # celltype_all_list = []
    for species_id, adata in adata_species_dict.items():
        species_list = species_list + [species_id] * adata.n_obs

    true_dom = torch.LongTensor(pd.Series(species_list).astype('category').cat.codes.values)  # .to(device)
    species_list = []
    for species_id, adata in adata_species_dict.items():
        species_list = species_list + [species_id] * adata.n_obs

    auxiliary_data = Data(edge_index=torch.LongTensor(edge_ndarray),
                          prune_edge_index=torch.LongTensor(np.array([])), x=auxiliary_X)
    auxiliary_subgraph_loader = NeighborLoader(auxiliary_data, num_neighbors=[-1], batch_size=batch_size * 2,
                                               shuffle=False)

    ite_N = max(int((len(anchor_ind_species) // batch_size)) + 1, 1)
    # ite_N = max(int((auxiliary_X.shape[0] // batch_size)) + 1, 1)
    print('ite_N', ite_N)
    species_ids = list(adata_species_dict.keys())

    for epoch in tqdm(range(0, n_epochs_species)):
        # subsampling anchor, positive, and negative
        k_add = 0
        for species_id in z_dict.keys():
            # species_add_dict[species_id] = int(k_add)
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

        if epoch % 50 == 0 and epoch > 0:
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

        mse_loss_mean = 0
        mse_loss = 0
        tri_output_species_mean = 0
        tri_output_slice_mean = 0
        mmd_loss_sum_mean = 0
        batch_num = 0
        species_id_list = list(adata_species_dict.keys())

        for ite_ in range(ite_N):
            triples_N = len(anchor_ind_species)
            tri_ind_list = random.sample(list(range(triples_N)), min(triples_N, batch_size))

            anchor_ind_species_batch = [anchor_ind_species_[x] for x in tri_ind_list]
            positive_ind_species_batch = [positive_ind_species_[x] for x in tri_ind_list]
            negative_ind_species_batch = [negative_ind_species_[x] for x in tri_ind_list]

            ind_list_init = list(
                set(anchor_ind_species_batch + positive_ind_species_batch + negative_ind_species_batch))

            if if_integrate_within_species == True:
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
            # print('-----------------------------')
            # print(auxiliary_X[idx_subset_list,].shape)
            # print(auxiliary_model)
            auxiliary_z_batch, auxiliary_out = auxiliary_model(auxiliary_data.x[idx_subset_list,].to(device),
                                                               edge_index_batch.to(device), mode='whole')

            mse_loss = F.mse_loss(merge_X[idx_subset_list,].to(device), out) + F.mse_loss(
                auxiliary_X[idx_subset_list,].to(device), auxiliary_out)

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
            # print(f'Number of triplets: {len(anchor_ind_species_batch)}')

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

            auxiliary_z_A = auxiliary_z_batch[spe_id_list[0:bsize],]
            auxiliary_z_B = auxiliary_z_batch[z_B_ind_list,]

            anchor_arr_species_X = x_batch[[idx_map[x] for x in anchor_ind_species_batch],]
            positive_arr_species_X = x_batch[[idx_map[x] for x in positive_ind_species_batch],]

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
            # mmd_loss_sum = mmd_loss(z_A[0:mmd_batch_size, :], z_B[0:mmd_batch_size, :]).to(device) + mmd_loss_sum + mmd_loss(auxiliary_z_A[0:mmd_batch_size, :], auxiliary_z_B[0:mmd_batch_size, :]).to(device)
            mmd_loss_sum = mmd_loss(z_A[0:mmd_batch_size, :], z_B[0:mmd_batch_size, :]).to(
                device) + mmd_loss_sum + mmd_loss(auxiliary_z_A[0:mmd_batch_size, :],
                                                  auxiliary_z_B[0:mmd_batch_size, :]).to(device)

            loss_ot = 0
            loss_ot_value = 0
            if ot_beta != 0:
                c_cross = pairwise_correlation_distance(anchor_arr_species_X.detach(),
                                                        positive_arr_species_X.detach()).to(device)
                T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=device)
                z_dist = torch.mean((anchor_arr_species.view(anchor_arr_species_X.shape[0], 1,
                                                             -1) - positive_arr_species.view(1,
                                                                                             anchor_arr_species_X.shape[
                                                                                                 0], -1)) ** 2, dim=2)
                loss_ot = torch.sum(T * z_dist) / torch.sum(T)
                loss_ot_value = loss_ot.item()

            sampling_num_spe = anchor_arr_species.shape[0]

            # modality align loss
            # z_dist = torch.mean((z_A[0:mmd_batch_size].view(mmd_batch_size, 1, -1) - z_B[0:mmd_batch_size].view(1, mmd_batch_size, -1))**2, dim=2)
            # loss_ot = torch.sum(T * z_dist) / torch.sum(T)
            # loss_G_GAN = -(torch.log(1 + torch.exp(-D_Z(z_A))) + torch.log(1 + torch.exp(D_Z(z_B)))).mean()
            if if_integrate_within_species == True:
                loss = mse_beta * mse_loss + tri_beta * tri_output_species + beta * tri_output + mmd_beta * mmd_loss_sum + gan_beta * loss_G_GAN + ot_beta * loss_ot
            else:
                loss = mse_beta * mse_loss + tri_beta * tri_output_species + mmd_beta * mmd_loss_sum + gan_beta * loss_G_GAN + ot_beta * loss_ot

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
            print(
                f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(len(anchor_arr_species)).to(device)).item()}, Cross slices triplets: {tri_output.item()}')
            if if_return_loss:
                loss_dict['Loss name'].append('Cross-slices triplet')
                loss_dict['Epoch'].append(epoch)
                loss_dict['Loss value'].append(tri_output.item())
            else:
                print(
                    f'Cosine cross species loss:{cosine_loss(anchor_arr_species, positive_arr_species, torch.ones(len(anchor_arr_species)).to(device)).item()}')

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
        # triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        # tri_auxiliary = triplet_loss(z[anchor_ind,], z[positive_ind,], z[negative_ind,]) + triplet_loss(
        #     auxiliary_z[anchor_ind,], auxiliary_z[positive_ind,], auxiliary_z[negative_ind,])
        # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        for species_id in z_dict.keys():
            k_add = species_add_dict[species_id]
            adata_species_dict[species_id].obsm[key_added] = z[k_add:int(k_add + adata_species_dict[species_id].n_obs),
                                                             :].cpu().detach().numpy()

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

    # print('Clustering and UMAP of Cross Species STACAME:')
    # clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate = umap_downsampling_rate)
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
                  n_epochs=1000,  
                  n_epochs_species=2000,  
                  lr=0.001, 
                  key_added='STACAME',
                  gradient_clipping=10., 
                  weight_decay=0.0001, 
                  lr_wd=0.001, 
                  weight_decay_wd=5e-4,
                  margin=1.0, 
                  margin_species=1.0, 
                  lr_species = 0.001,
                  alpha=1, 
                  beta=1,
                  verbose=False,
                  random_seed=666, 
                  iter_comb=None, 
                  knn_neigh=100, 
                  device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'), 
                  pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                  mse_beta = 1, 
                  tri_beta = 1, 
                  mmd_beta = 2, 
                  gan_beta = 1,
                  mmd_batch_size = 2048, 
                  if_knn_mnn_graph = True, 
                  if_integrate_within_species = False, 
                  if_return_loss = False, 
                  batch_size_dict = {'Mouse': 20000, 'Marmoset':12000, 'Macaque':4096}, 
                  batch_size = 2048, 
                  umap_downsampling_rate = 0.1, 
                  mode = 'spatial_domain', 
                  annotation_species = ['Mouse']):
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
            model = STACAME.STACAME_minibatch_large(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
                #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
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
    #-------------------------------------------------------
    S = 0
    for species_id, adata in adata_species_dict.items():
        if S != 0:
            if 'highly_variable' in adata.var.columns:
                x=adata[:, adata.uns['highly_variable']].X.todense()
            else:
                x=adata.X.todense()
            merge_X = np.concatenate((merge_X, x), axis=0)
        else:
            if 'highly_variable' in adata.var.columns:
                merge_X=adata[:, adata.uns['highly_variable']].X.todense()
            else:
                merge_X=adata.X.todense()
            S = S + 1
    merge_X = torch.FloatTensor(merge_X)
    ##-----------------------------------------------------------
    model = STACAME.STACAMEDecoder_minibatch(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_species, weight_decay=weight_decay)
    
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
    optimizer_D = torch.optim.Adam(list(D_Z.parameters()), lr=0.001, weight_decay=0.001)
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
            #loss_G_GAN = -(torch.log(1 + torch.exp(-D_Z(z_A))) + torch.log(1 + torch.exp(D_Z(z_B)))).mean()
            if if_integrate_within_species == True:     
                loss =  mse_beta * mse_loss + tri_beta * tri_output_species + beta * tri_output  + mmd_beta * mmd_loss_sum +  gan_beta * loss_G_GAN 
            else:
                loss =  mse_beta * mse_loss + tri_beta * tri_output_species + mmd_beta * mmd_loss_sum + gan_beta * loss_G_GAN

            loss.backward(retain_graph=True)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
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
        #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
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
                  n_epochs=1000,  
                  n_epochs_species=2000,  
                  lr=0.001, 
                  key_added='STACAME',
                  gradient_clipping=10., 
                  weight_decay=0.0001, 
                  lr_wd=0.001, 
                  weight_decay_wd=5e-4,
                  margin=1.0, 
                  margin_species=1.0, 
                  lr_species = 0.001,
                  alpha=1, 
                  beta=1,
                  verbose=False,
                  random_seed=666, 
                  iter_comb=None, 
                  knn_neigh=100, 
                  device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'), 
                  pretrain_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                  mse_beta = 1, 
                  tri_beta = 1, 
                  mmd_beta = 2, 
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
                model = STACAME.STACAME_minibatch_large(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(pretrain_device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                    #if epoch % 2 == 0:
                        #n_id_batch = n_id[0:batchsize]
                        
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
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
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
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            species_order += 1
            print('Pretrain with STAGATE_multiple...')
            for epoch in tqdm(range(0,  stagate_epoch_dict[species_id])):
                model.train()
                optimizer.zero_grad()
                z, out = model(data.x.to(pretrain_device), data.edge_index.to(pretrain_device))

                if epoch % 10 == 0 and epoch >= stagate_epoch_dict[species_id]//2:
                    # if verbose:
                    #     print('Update spot triplets at epoch ' + str(epoch))
                    adata.obsm['STAGATE'] = z.cpu().detach().numpy()
                    # If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
                    # not all points have MNN achors
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
    S = 0
    for species_id, adata in adata_species_dict.items():
        if S != 0:
            if 'highly_variable' in adata.var.columns:
                x=adata[:, adata.uns['highly_variable']].X.todense()
            else:
                x=adata.X.todense()
            merge_X = np.concatenate((merge_X, x), axis=0)
        else:
            if 'highly_variable' in adata.var.columns:
                merge_X=adata[:, adata.uns['highly_variable']].X.todense()
            else:
                merge_X=adata.X.todense()
            S = S + 1

    if hasattr(adata_whole.obsm['X_pca'], 'todense'):
        auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'].todense())
    else:
        auxiliary_X = torch.FloatTensor(adata_whole.obsm['X_pca'])

    if concate_pca_dim != None:
        adata_X = ad.AnnData(merge_X)
        sc.pp.scale(adata_X)
        sc.tl.pca(adata_X, n_comps=concate_pca_dim)
        merge_X = adata_X.obsm["X_pca"]   # shape: (n_cells, n_comps)
    merge_X = torch.FloatTensor(merge_X)
    ##-----------------------------------------------------------
    model = STACAME.STACAMEDecoder_minibatch(hidden_dims=[merge_X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_species, weight_decay=weight_decay)
    
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
    optimizer_D = torch.optim.Adam(list(D_Z.parameters()), lr=0.001, weight_decay=0.001)
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

            # modality align loss
            # z_dist = torch.mean((z_A[0:mmd_batch_size].view(mmd_batch_size, 1, -1) - z_B[0:mmd_batch_size].view(1, mmd_batch_size, -1))**2, dim=2)
            # loss_ot = torch.sum(T * z_dist) / torch.sum(T)
            #loss_G_GAN = -(torch.log(1 + torch.exp(-D_Z(z_A))) + torch.log(1 + torch.exp(D_Z(z_B)))).mean()
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


    # print('Clustering and UMAP of Cross Species STACAME:')
    # clustering_umap_downsampling(adata_species_dict, key_umap=key_added, downsampling_rate = umap_downsampling_rate)

    del model, optimizer, D_Z, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if if_return_loss:
        return adata_species_dict, loss_dict
    return adata_species_dict


