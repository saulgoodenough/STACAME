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
from .train_STACAME import random_list, clustering_umap, clustering_umap_downsampling
import gc


class STACAME_subgraph_trainer:
    """
    Subgraph‑based trainer for cross‑species STACAME with GAN loss and an auxiliary model.

    This class encapsulates per‑species STAGATE pretraining, joint subgraph
    training, checkpoint saving and resuming, and best‑loss model selection.
    It first pretrains a STAGATE model
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
    model_save_path : str, optional
        Directory to save the best model checkpoint. If None, no file is saved.
    resume_from_checkpoint : str, optional
        Path to a checkpoint file to resume training from. If provided,
        pretraining is skipped and the state is loaded.
    Returns
    -------
    adata_species_dict : dict
        Updated AnnData dict with the best epoch embedding in ``obsm[key_added]``.
    loss_dict : dict, optional
        Recorded losses per epoch, returned when ``if_return_loss=True``.
    """

    def __init__(self,
                 adata_species_dict,
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
                 structure_sampling_ratio=1.0,
                 model_save_path=None,
                 resume_from_checkpoint=None):
        # Store all configuration parameters
        self.adata_species_dict = adata_species_dict
        self.triplet_ind_species_dict = triplet_ind_species_dict
        self.edge_ndarray_species = edge_ndarray_species
        self.triplet_ind_sections_dict = triplet_ind_sections_dict
        self.edge_ndarray_sections = edge_ndarray_sections
        self.hidden_dims = hidden_dims
        self.stagate_epoch = stagate_epoch
        self.n_epochs_species = n_epochs_species
        self.lr = lr
        self.key_added = key_added
        self.gradient_clipping = gradient_clipping
        self.weight_decay = weight_decay
        self.lr_wd = lr_wd
        self.weight_decay_wd = weight_decay_wd
        self.margin = margin
        self.margin_species = margin_species
        self.lr_species = lr_species
        self.beta = beta
        self.verbose = verbose
        self.random_seed = random_seed
        self.iter_comb = iter_comb
        self.knn_neigh = knn_neigh
        self.device = device
        self.pretrain_device = pretrain_device
        self.mse_beta = mse_beta
        self.tri_beta = tri_beta
        self.mmd_beta = mmd_beta
        self.gan_beta = gan_beta
        self.gan_epoch = gan_epoch
        self.ot_beta = ot_beta
        self.mmd_batch_size = mmd_batch_size
        self.if_knn_mnn_graph = if_knn_mnn_graph
        self.if_integrate_within_species = if_integrate_within_species
        self.if_return_loss = if_return_loss
        self.if_batch_pretrain = if_batch_pretrain
        self.batch_size_dict = batch_size_dict
        self.batch_size = batch_size
        self.concate_pca_dim = concate_pca_dim
        self.umap_downsampling_rate = umap_downsampling_rate
        self.adata_whole = adata_whole
        self.if_use_light_model = if_use_light_model
        self.structure_beta = structure_beta
        self.structure_sampling_ratio = structure_sampling_ratio
        self.model_save_path = model_save_path
        self.resume_from_checkpoint = resume_from_checkpoint

        # Internal state to be initialized during training
        self.model = None
        self.auxiliary_model = None
        self.D_Z = None
        self.auxiliary_D_Z = None
        self.optimizer = None
        self.optimizer_D = None
        self.auxiliary_optimizer_D = None
        self.z_dict = {}
        self.species_add_dict = {}
        self.edge_ndarray = None
        self.data = None
        self.auxiliary_data = None
        self.subgraph_loader = None
        self.auxiliary_subgraph_loader = None
        self.merge_X = None
        self.auxiliary_X = None
        self.true_dom = None
        self.id_species_dict = {}
        self.species_n_cells = {}
        self.anchor_ind_species = None
        self.positive_ind_species = None
        self.negative_ind_species = None
        self.anchor_ind_sections = None
        self.positive_ind_sections = None
        self.negative_ind_sections = None
        self.loss_dict = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_embeddings = None
        self.best_auxiliary_embedding = None
        self.start_epoch = 0
        self.ite_N = 1
        self.species_ids = []
        self.species_id_list = []

    def set_random_seeds(self):
        """Set all random seeds for reproducibility."""
        seed = self.random_seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.autograd.set_detect_anomaly(True)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

    def pretrain_stage(self):
        """Per‑species STAGATE pretraining (mini‑batch or full‑batch)."""
        if not isinstance(self.stagate_epoch, dict):
            stagate_epoch_dict = {k: self.stagate_epoch for k in self.adata_species_dict.keys()}
        else:
            stagate_epoch_dict = self.stagate_epoch

        self.z_dict = {k: 0 for k in self.adata_species_dict.keys()}
        species_order = 0
        for species_id, adata in self.adata_species_dict.items():
            section_ids = np.array(adata.obs['batch_name'].unique())
            edgeList = adata.uns['edgeList']
            if 'highly_variable' in adata.var.columns:
                adata = adata[:, adata.uns['highly_variable']]
            print(f'For {species_id}, using {len(adata.var_names)} genes for training.')

            data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                        prune_edge_index=torch.LongTensor(np.array([])),
                        x=torch.FloatTensor(adata.X.todense()))

            if self.if_batch_pretrain:
                if species_order == 0:
                    model = STACAME.STACAME_minibatch(
                        hidden_dims=[data.x.shape[1], self.hidden_dims[0], self.hidden_dims[1]]
                    ).to(self.pretrain_device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                                 weight_decay=self.weight_decay, foreach=False)

                train_loader = NeighborSampler(data.edge_index,
                                               node_idx=torch.LongTensor(np.arange(adata.n_obs)),
                                               sizes=[8, 4],
                                               batch_size=self.batch_size_dict[species_id],
                                               shuffle=True, drop_last=True)
                subgraph_loader = NeighborLoader(data, num_neighbors=[-1],
                                                 batch_size=self.batch_size_dict[species_id],
                                                 shuffle=False)

                print('Pretrain with STAGATE (Minibatch)...')
                for epoch in tqdm(range(stagate_epoch_dict[species_id])):
                    total_loss = 0
                    for batchsize, n_id, adjs in train_loader:
                        adjs = [adj.to(self.pretrain_device) for adj in adjs]
                        optimizer.zero_grad()
                        z_batch, out_batch = model(data.x[n_id, :].to(self.pretrain_device), adjs, mode='batch')
                        x_batch = data.x[n_id, :].to(self.pretrain_device)

                        # Within‑species triplet loss computation (mini‑batch)
                        n_id_list = n_id.cpu().detach().numpy()
                        batch_id_list = adata.obs['batch_name'][n_id_list]
                        x_batch_cpu = z_batch.cpu().detach().numpy()
                        x_batch_adata = ad.AnnData(X=x_batch_cpu,
                                                   obs=pd.DataFrame({"batch_name": batch_id_list}))
                        x_batch_adata.obsm['STAGATE'] = x_batch_cpu
                        section_ids = np.array(x_batch_adata.obs['batch_name'].unique())
                        mnn_dict = create_dictionary_mnn(x_batch_adata, use_rep='STAGATE',
                                                         batch_name='batch_name',
                                                         k=self.knn_neigh,
                                                         iter_comb=self.iter_comb, verbose=0)

                        anchor_ind, positive_ind, negative_ind = [], [], []
                        for batch_pair in mnn_dict.keys():
                            batchname_list = x_batch_adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                            cellname_by_batch_dict = dict()
                            for batch_id in range(len(section_ids)):
                                cellname_by_batch_dict[section_ids[batch_id]] = x_batch_adata.obs_names[
                                    x_batch_adata.obs['batch_name'] == section_ids[batch_id]].values
                            for anchor in mnn_dict[batch_pair].keys():
                                anchor_list = [anchor]
                                positive_spot = mnn_dict[batch_pair][anchor][0]
                                section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                                negative_list = [
                                    cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)]
                                ]
                                batch_as_dict = dict(zip(list(x_batch_adata.obs_names), range(x_batch_adata.shape[0])))
                                anchor_ind.append(batch_as_dict[anchor_list[0]])
                                positive_ind.append(batch_as_dict[positive_spot])
                                negative_ind.append(batch_as_dict[negative_list[0]])

                        if len(anchor_ind) > 0:
                            anchor_arr = z_batch[anchor_ind,]
                            positive_arr = z_batch[positive_ind,]
                            negative_arr = z_batch[negative_ind,]
                            triplet_loss_fn = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')
                            tri_output = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
                            loss = F.mse_loss(out_batch, x_batch) + self.beta * tri_output
                        else:
                            loss = F.mse_loss(out_batch, x_batch)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                with torch.no_grad():
                    z_list, out_list = [], []
                    for batch in subgraph_loader:
                        batch.to(self.pretrain_device)
                        z, out = model(batch.x, batch.edge_index, mode='all')
                        z_list.append(z[:batch.batch_size].cpu())
                        out_list.append(out[:batch.batch_size].cpu())
                z_all = torch.cat(z_list, dim=0)
                self.adata_species_dict[species_id].obsm['STAGATE'] = z_all.cpu().detach().numpy()
                self.z_dict[species_id] = z_all.cpu().detach()

                if species_order >= len(self.adata_species_dict.keys()):
                    del model, optimizer, data, z_batch, out_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            else:
                print('Pretrain with STAligner...')
                if species_order == 0:
                    model = STACAME.STACAME(
                        hidden_dims=[data.x.shape[1], self.hidden_dims[0], self.hidden_dims[1]]
                    ).to(self.pretrain_device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                                 weight_decay=self.weight_decay, foreach=False)
                species_order += 1
                print('Pretrain with STAGATE_multiple...')
                for epoch in tqdm(range(0, stagate_epoch_dict[species_id])):
                    model.train()
                    optimizer.zero_grad()
                    z, out = model(data.x.to(self.pretrain_device),
                                   data.edge_index.to(self.pretrain_device))

                    if epoch % 50 == 0 and epoch >= stagate_epoch_dict[species_id] // 2:
                        adata.obsm['STAGATE'] = z.cpu().detach().numpy()
                        mnn_dict = create_dictionary_mnn(adata, use_rep='STAGATE', batch_name='batch_name',
                                                         k=self.knn_neigh, iter_comb=self.iter_comb, verbose=0)
                        anchor_ind, positive_ind, negative_ind = [], [], []
                        for batch_pair in mnn_dict.keys():
                            batchname_list = adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                            cellname_by_batch_dict = dict()
                            for batch_id in range(len(section_ids)):
                                cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                                    adata.obs['batch_name'] == section_ids[batch_id]].values
                            for anchor in mnn_dict[batch_pair].keys():
                                anchor_list = [anchor]
                                positive_spot = mnn_dict[batch_pair][anchor][0]
                                section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                                negative_list = [
                                    cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)]
                                ]
                                batch_as_dict = dict(zip(list(adata.obs_names), range(adata.shape[0])))
                                anchor_ind.append(batch_as_dict[anchor_list[0]])
                                positive_ind.append(batch_as_dict[positive_spot])
                                negative_ind.append(batch_as_dict[negative_list[0]])

                        if len(anchor_ind) > 0:
                            anchor_arr = z[anchor_ind,]
                            positive_arr = z[positive_ind,]
                            negative_arr = z[negative_ind,]
                            triplet_loss_fn = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')
                            tri_output = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
                            loss = F.mse_loss(data.x.to(self.pretrain_device), out) + self.beta * tri_output
                        else:
                            loss = F.mse_loss(data.x.to(self.pretrain_device), out)
                    else:
                        loss = F.mse_loss(data.x.to(self.pretrain_device), out)
                    loss.backward(retain_graph=True)
                    optimizer.step()

                with torch.no_grad():
                    z, _ = model(data.x.to(self.pretrain_device), data.edge_index.to(self.pretrain_device))
                self.adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
                self.z_dict[species_id] = z.cpu().detach()

        del model, optimizer, data, z, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def prepare_joint_training(self):
        """Build concatenated graph, merge expression, models and subgraph loaders."""
        print('-------------------------------------------------------------------------------')
        print('Prepare joint training...')
        self.anchor_ind_species = self.triplet_ind_species_dict['anchor_ind_species']
        self.positive_ind_species = self.triplet_ind_species_dict['positive_ind_species']
        self.negative_ind_species = self.triplet_ind_species_dict['negative_ind_species']

        # Build concatenated graph
        k_add = 0
        self.species_add_dict = {k: None for k in self.z_dict.keys()}
        for species_id in self.z_dict.keys():
            self.species_add_dict[species_id] = int(k_add)
            k_add = int(k_add + self.adata_species_dict[species_id].n_obs)

        adata = self.adata_species_dict[list(self.adata_species_dict.keys())[0]]
        edgeList = adata.uns['edgeList']
        edge_ndarray = np.array([edgeList[0], edgeList[1]])
        S = 0
        for species_id, adata in self.adata_species_dict.items():
            edgeList = adata.uns['edgeList']
            if S != 0:
                edge_arr_temp = np.array([edgeList[0], edgeList[1]]) + self.species_add_dict[species_id]
                edge_ndarray = np.concatenate((edge_ndarray, edge_arr_temp), axis=1)
            else:
                S += 1

        edge_ndarray_species = np.array([self.edge_ndarray_species[0], self.edge_ndarray_species[1]])
        if self.if_knn_mnn_graph:
            edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_species), axis=1)
        self.edge_ndarray = edge_ndarray

        # Concatenate pretrained embeddings
        X = np.concatenate([self.z_dict[sp].cpu().numpy() for sp in self.z_dict], axis=0)
        z = torch.FloatTensor(X)

        # Merge gene expression across species
        species_list = list(self.adata_species_dict.keys())
        n_species = len(species_list)
        ref_species = species_list[0]
        n_homo_genes = len(self.adata_species_dict[ref_species].uns['homo_highly_variable'])
        species_specific_n_genes = {
            sp: len(self.adata_species_dict[sp].uns['species_specific']) for sp in species_list
        }
        max_specific_genes = max(species_specific_n_genes.values())
        total_cols = n_homo_genes + max_specific_genes * n_species
        merge_X = None
        for sp_idx, species_id in enumerate(species_list):
            adata = self.adata_species_dict[species_id]
            homo_genes = adata.uns['homo_highly_variable']
            x_homo = adata[:, homo_genes].X.todense()
            specific_genes = adata.uns['species_specific']
            x_specific = adata[:, specific_genes].X.todense()
            n_cells = x_homo.shape[0]
            x_current = np.zeros((n_cells, total_cols))
            x_current[:, :n_homo_genes] = x_homo
            specific_start_col = n_homo_genes + sp_idx * max_specific_genes
            specific_end_col = specific_start_col + species_specific_n_genes[species_id]
            x_current[:, specific_start_col:specific_end_col] = x_specific
            if merge_X is None:
                merge_X = x_current
            else:
                merge_X = np.concatenate((merge_X, x_current), axis=0)

        if self.concate_pca_dim is not None:
            adata_X = ad.AnnData(merge_X)
            sc.pp.scale(adata_X)
            sc.tl.pca(adata_X, n_comps=self.concate_pca_dim)
            merge_X = adata_X.obsm["X_pca"]
        self.merge_X = torch.FloatTensor(merge_X)

        if hasattr(self.adata_whole.obsm['X_pca'], 'todense'):
            self.auxiliary_X = torch.FloatTensor(self.adata_whole.obsm['X_pca'].todense())
        else:
            self.auxiliary_X = torch.FloatTensor(self.adata_whole.obsm['X_pca'])

        # Build models
        if self.if_use_light_model:
            self.model = STACAME.STACAME_lightdecoder_minibatch(
                hidden_dims=[self.merge_X.shape[1], self.hidden_dims[0], self.hidden_dims[1]]
            ).to(self.device)
        else:
            self.model = STACAME.STACAMEDecoder_minibatch(
                hidden_dims=[self.merge_X.shape[1], self.hidden_dims[0], self.hidden_dims[1]]
            ).to(self.device)
        self.auxiliary_model = STACAME.STACAME_minibatch(
            hidden_dims=[self.auxiliary_X.shape[1], self.hidden_dims[0] // 2, self.hidden_dims[1]]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.auxiliary_model.parameters()),
            lr=self.lr_species, weight_decay=self.weight_decay, foreach=False
        )

        self.species_n_cells = {sp: self.adata_species_dict[sp].n_obs for sp in self.adata_species_dict.keys()}

        # Within‑species triplets
        if self.if_integrate_within_species:
            self.anchor_ind_sections = self.triplet_ind_sections_dict['anchor_ind_sections']
            self.positive_ind_sections = self.triplet_ind_sections_dict['positive_ind_sections']
            self.negative_ind_sections = self.triplet_ind_sections_dict['negative_ind_sections']
            if self.if_knn_mnn_graph and self.edge_ndarray_sections is not None:
                edge_ndarray_sections = np.array([self.edge_ndarray_sections[0], self.edge_ndarray_sections[1]])
                self.edge_ndarray = np.concatenate((self.edge_ndarray, edge_ndarray_sections), axis=1)

        # Create PyG Data and subgraph loaders
        self.data = Data(edge_index=torch.LongTensor(self.edge_ndarray),
                         prune_edge_index=torch.LongTensor(np.array([])), x=z)
        self.auxiliary_data = Data(edge_index=torch.LongTensor(self.edge_ndarray),
                                   prune_edge_index=torch.LongTensor(np.array([])),
                                   x=self.auxiliary_X)
        self.subgraph_loader = NeighborLoader(self.data, num_neighbors=[-1],
                                              batch_size=self.batch_size * 2, shuffle=False)
        self.auxiliary_subgraph_loader = NeighborLoader(self.auxiliary_data, num_neighbors=[-1],
                                                        batch_size=self.batch_size * 2, shuffle=False)

        # Species id mapping for every node
        self.id_species_dict = {}
        k_add = 0
        for spe_id in self.adata_species_dict.keys():
            for id_s in range(k_add, k_add + self.adata_species_dict[spe_id].n_obs):
                self.id_species_dict[id_s] = spe_id
            k_add += self.adata_species_dict[spe_id].n_obs

        # Discriminators
        self.D_Z = STACAME.MultiClassDiscriminator(self.hidden_dims[1], n_species).to(self.device)
        self.optimizer_D = torch.optim.Adam(list(self.D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)
        self.D_Z.train()
        self.auxiliary_D_Z = STACAME.MultiClassDiscriminator(self.hidden_dims[1], n_species).to(self.device)
        self.auxiliary_optimizer_D = torch.optim.Adam(list(self.auxiliary_D_Z.parameters()), lr=0.001,
                                                      weight_decay=0.001, foreach=False)
        self.auxiliary_D_Z.train()

        species_list_gt = []
        for species_id, adata in self.adata_species_dict.items():
            species_list_gt.extend([species_id] * adata.n_obs)
        self.true_dom = torch.LongTensor(pd.Series(species_list_gt).astype('category').cat.codes.values)

        # Iterations per epoch
        self.ite_N = max(int((len(self.anchor_ind_species) // self.batch_size)) + 1, 1)
        self.species_ids = list(self.adata_species_dict.keys())
        self.species_id_list = list(self.adata_species_dict.keys())

        if self.if_return_loss:
            self.loss_dict = {'Loss name': [], 'Epoch': [], 'Loss value': []}

    def manifold_preserving_loss(self, latent_vectors, ref_vectors, same_species_edges, edge_weights=None):
        """Weighted Laplacian loss to preserve local geometry."""
        src, dst = same_species_edges[0], same_species_edges[1]
        d2_ref = torch.sum((ref_vectors[src] - ref_vectors[dst]) ** 2, dim=1)
        sigma = torch.sqrt(torch.median(d2_ref) + 1e-8)
        w = torch.exp(-d2_ref / (2 * sigma ** 2 + 1e-8))
        d2_lat = torch.sum((latent_vectors[src] - latent_vectors[dst]) ** 2, dim=1)
        if edge_weights is None:
            edge_weights = torch.ones_like(w)
        return torch.mean(edge_weights * w * d2_lat)

    def joint_train(self):
        """Cross‑species subgraph joint training loop."""
        plot_epoch = self.n_epochs_species // 3

        for epoch in tqdm(range(self.start_epoch, self.n_epochs_species)):
            # Update STAGATE arrays in adata (recording)
            k_add = 0
            for species_id in self.z_dict.keys():
                start = int(k_add)
                end = int(k_add + self.adata_species_dict[species_id].n_obs)
                self.adata_species_dict[species_id].obsm['STAGATE'] = self.data.x[start:end].cpu().detach().numpy()
                self.z_dict[species_id] = self.adata_species_dict[species_id].obsm['STAGATE']
                k_add = end

            if epoch == self.start_epoch:
                anchor_ind_species_ = self.anchor_ind_species
                positive_ind_species_ = self.positive_ind_species
                negative_ind_species_ = self.negative_ind_species
                if hasattr(self.adata_whole.obsm['X_pca'], 'todense'):
                    self.adata_whole.obsm['auxiliary'] = self.adata_whole.obsm['X_pca'].todense()
                else:
                    self.adata_whole.obsm['auxiliary'] = self.adata_whole.obsm['X_pca']
            else:
                if epoch % 50 == 0 and epoch > 0:
                    mnn_dict = create_dictionary_mnn(self.adata_whole, use_rep='auxiliary',
                                                     batch_name='species_id', k=self.knn_neigh,
                                                     iter_comb=self.iter_comb, verbose=0)
                    anchor_ind_species_ = list(self.anchor_ind_species)
                    positive_ind_species_ = list(self.positive_ind_species)
                    negative_ind_species_ = list(self.negative_ind_species)
                    for batch_pair in mnn_dict.keys():
                        batchname_list = self.adata_whole.obs['species_id'][mnn_dict[batch_pair].keys()]
                        cellname_by_batch_dict = dict()
                        for batch_id in range(len(self.species_ids)):
                            cellname_by_batch_dict[self.species_ids[batch_id]] = self.adata_whole.obs_names[
                                self.adata_whole.obs['species_id'] == self.species_ids[batch_id]].values
                        for anchor in mnn_dict[batch_pair].keys():
                            pos = mnn_dict[batch_pair][anchor][0]
                            neg = cellname_by_batch_dict[batchname_list[anchor]][
                                np.random.randint(len(cellname_by_batch_dict[batchname_list[anchor]]))
                            ]
                            batch_as_dict = dict(zip(list(self.adata_whole.obs_names), range(self.adata_whole.shape[0])))
                            anchor_ind_species_.append(batch_as_dict[anchor])
                            positive_ind_species_.append(batch_as_dict[pos])
                            negative_ind_species_.append(batch_as_dict[neg])
                    anchor_ind_species_ = np.array(anchor_ind_species_)
                    positive_ind_species_ = np.array(positive_ind_species_)
                    negative_ind_species_ = np.array(negative_ind_species_)

            triples_N = len(anchor_ind_species_)
            ite_N = max(int((triples_N // self.batch_size)) + 1, 1)
            # Iterate over subgraph mini‑batches
            for ite_ in range(self.ite_N):
                triples_N = len(anchor_ind_species_)
                tri_ind_list = random.sample(list(range(triples_N)), min(triples_N, self.batch_size))

                anchor_ind_species_batch = [anchor_ind_species_[x] for x in tri_ind_list]
                positive_ind_species_batch = [positive_ind_species_[x] for x in tri_ind_list]
                negative_ind_species_batch = [negative_ind_species_[x] for x in tri_ind_list]
                ind_list_init = list(set(anchor_ind_species_batch + positive_ind_species_batch +
                                         negative_ind_species_batch))

                if self.if_integrate_within_species:
                    triples_N_sec = len(self.anchor_ind_sections)
                    tri_ind_list_sec = random.sample(list(range(triples_N_sec)),
                                                     min(self.batch_size, triples_N_sec))
                    anchor_ind_sections_batch = [self.anchor_ind_sections[x] for x in tri_ind_list_sec]
                    positive_ind_sections_batch = [self.positive_ind_sections[x] for x in tri_ind_list_sec]
                    negative_ind_sections_batch = [self.negative_ind_sections[x] for x in tri_ind_list_sec]
                    ind_list_init = list(set(ind_list_init + anchor_ind_sections_batch +
                                             positive_ind_sections_batch + negative_ind_sections_batch))

                idx_subset, edge_index_batch, mapping, edge_mask = k_hop_subgraph(
                    node_idx=torch.LongTensor(ind_list_init), 
                    num_hops=1, edge_index=self.data.edge_index, relabel_nodes=True
                )
                idx_subset_list = [int(x) for x in idx_subset]
                idx_map = {k: v for k, v in zip(idx_subset_list, range(len(idx_subset_list)))}

                self.model.train()
                self.auxiliary_model.train()
                self.optimizer.zero_grad()

                z_batch, out = self.model(self.data.x[idx_subset_list,].to(self.device),
                                          edge_index_batch.to(self.device), mode='whole')
                auxiliary_z_batch, auxiliary_out = self.auxiliary_model(
                    self.auxiliary_data.x[idx_subset_list,].to(self.device),
                    edge_index_batch.to(self.device), mode='whole'
                )
                ref_x_sub = self.data.x[idx_subset_list,].to(self.device)

                # 1) MSE
                mse_loss = F.mse_loss(self.merge_X[idx_subset_list,].to(self.device), out) + \
                           F.mse_loss(self.auxiliary_X[idx_subset_list,].to(self.device), auxiliary_out)

                # 2) Within‑species slice triplet
                if self.if_integrate_within_species:
                    anchor_arr = z_batch[[idx_map[x] for x in anchor_ind_sections_batch],]
                    positive_arr = z_batch[[idx_map[x] for x in positive_ind_sections_batch],]
                    negative_arr = z_batch[[idx_map[x] for x in negative_ind_sections_batch],]
                    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')
                    tri_output = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
                else:
                    tri_output = torch.tensor(0.0, device=self.device)

                # 3) Cross‑species triplet
                anchor_arr_species = z_batch[[idx_map[x] for x in anchor_ind_species_batch],]
                positive_arr_species = z_batch[[idx_map[x] for x in positive_ind_species_batch],]
                negative_arr_species = z_batch[[idx_map[x] for x in negative_ind_species_batch],]
                tri_output_species = torch.nn.TripletMarginLoss(
                    margin=self.margin_species, p=2, reduction='mean'
                )(anchor_arr_species, positive_arr_species, negative_arr_species)

                # 4) MMD
                z_ind_species_dict = {k: [] for k in self.adata_species_dict.keys()}
                for n_id_temp in idx_subset_list:
                    z_ind_species_dict[self.id_species_dict[n_id_temp]].append(n_id_temp)

                mmd_loss_fn = STACAME.MMDLoss(kernel=STACAME.RBF(device=self.device), device=self.device).to(self.device)
                mmd_loss_sum = 0.0

                spe_id = random.sample(self.species_id_list, 1)[0]
                spe_id_list = [idx_map[x] for x in z_ind_species_dict[spe_id]]
                bsize = min(len(spe_id_list), len(idx_subset_list) - len(spe_id_list))

                z_A = z_batch[spe_id_list[:bsize],]
                z_B_ind_list = random.sample(list(set(range(len(idx_subset_list))) - set(spe_id_list)), bsize)
                z_B = z_batch[z_B_ind_list,]

                auxiliary_z_A = auxiliary_z_batch[spe_id_list[:bsize],]
                auxiliary_z_B = auxiliary_z_batch[z_B_ind_list,]

                x_batch = self.auxiliary_X[idx_subset_list,].to(self.device)
                anchor_arr_species_X = x_batch[[idx_map[x] for x in anchor_ind_species_batch],]
                positive_arr_species_X = x_batch[[idx_map[x] for x in positive_ind_species_batch],]

                # 5) GAN
                # if self.gan_beta != 0:
                #     for _ in range(self.gan_epoch):
                #         self.optimizer_D.zero_grad()
                #         logits_D = self.D_Z(z_batch)
                #         loss_D = F.cross_entropy(logits_D, self.true_dom[idx_subset_list,].to(self.device))
                #         loss_D.backward(retain_graph=True)
                #         self.optimizer_D.step()
                #     for _ in range(self.gan_epoch):
                #         self.auxiliary_optimizer_D.zero_grad()
                #         auxiliary_logits_D = self.auxiliary_D_Z(auxiliary_z_batch)
                #         auxiliary_loss_D = F.cross_entropy(auxiliary_logits_D,
                #                                            self.true_dom[idx_subset_list,].to(self.device))
                #         auxiliary_loss_D.backward(retain_graph=True)
                #         self.auxiliary_optimizer_D.step()
                if self.gan_beta != 0:
                    z_detached = z_batch.detach()          # 切断梯度，避免生成器计算图被保留
                    for _ in range(self.gan_epoch):
                        self.optimizer_D.zero_grad()
                        logits_D = self.D_Z(z_detached)
                        loss_D = F.cross_entropy(logits_D, self.true_dom[idx_subset_list].to(self.device))
                        loss_D.backward()                 
                        self.optimizer_D.step()
                    auxiliary_z_detached = auxiliary_z_batch.detach()
                    for _ in range(self.gan_epoch):
                        self.auxiliary_optimizer_D.zero_grad()
                        auxiliary_logits_D = self.auxiliary_D_Z(auxiliary_z_detached)
                        auxiliary_loss_D = F.cross_entropy(auxiliary_logits_D, self.true_dom[idx_subset_list].to(self.device))
                        auxiliary_loss_D.backward()
                        self.auxiliary_optimizer_D.step()

                loss_G_GAN = -F.cross_entropy(self.D_Z(z_batch), self.true_dom[idx_subset_list,].to(self.device)) - \
                             F.cross_entropy(self.auxiliary_D_Z(auxiliary_z_batch),
                                             self.true_dom[idx_subset_list,].to(self.device))

                mmd_loss_sum = mmd_loss_fn(z_A[:self.mmd_batch_size], z_B[:self.mmd_batch_size]) + \
                               mmd_loss_fn(auxiliary_z_A[:self.mmd_batch_size], auxiliary_z_B[:self.mmd_batch_size])

                # 6) OT
                loss_ot = torch.tensor(0.0, device=self.device)
                if self.ot_beta != 0:
                    c_cross = pairwise_correlation_distance(anchor_arr_species_X.detach(),
                                                            positive_arr_species_X.detach()).to(self.device)
                    T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=self.device)
                    z_dist = torch.mean((anchor_arr_species.view(len(anchor_arr_species), 1, -1) -
                                         positive_arr_species_X.view(1, len(anchor_arr_species), -1)) ** 2, dim=2)
                    loss_ot = torch.sum(T * z_dist) / torch.sum(T)

                # 7) Manifold preserving loss
                geom_structure_loss = torch.tensor(0.0, device=self.device)
                if self.structure_beta != 0.0:
                    new_idx_to_species = {new: self.id_species_dict[orig]
                                          for orig, new in zip(idx_subset_list, range(len(idx_subset_list)))}
                    edge_np = edge_index_batch.cpu().numpy()
                    species_mask = np.array([
                        new_idx_to_species.get(int(src), None) is not None and
                        new_idx_to_species.get(int(dst), None) is not None and
                        new_idx_to_species[int(src)] == new_idx_to_species[int(dst)]
                        for src, dst in edge_np.T
                    ])
                    species_edges = edge_index_batch[:, species_mask].to(self.device)
                    if species_edges.shape[1] > 0:
                        n_edges = species_edges.shape[1]
                        n_sample = max(1, int(n_edges * self.structure_sampling_ratio))
                        perm = torch.randperm(n_edges, device=self.device)[:n_sample]
                        sampled_edges = species_edges[:, perm]
                        src_nodes = sampled_edges[0].cpu().numpy()
                        edge_species = [new_idx_to_species[int(node)] for node in src_nodes]
                        inv_n = {sp: 1.0 / self.species_n_cells[sp] for sp in self.species_n_cells}
                        raw_weights = torch.tensor([inv_n[sp] for sp in edge_species],
                                                   device=self.device, dtype=torch.float32)
                        edge_w = raw_weights * (len(raw_weights) / raw_weights.sum())
                        loss_z = self.manifold_preserving_loss(z_batch, ref_x_sub, sampled_edges, edge_weights=edge_w)
                        loss_aux = self.manifold_preserving_loss(auxiliary_z_batch, ref_x_sub, sampled_edges,
                                                                 edge_weights=edge_w)
                        geom_structure_loss = loss_z + loss_aux

                # Total loss
                if self.if_integrate_within_species:
                    loss = (self.mse_beta * mse_loss + self.tri_beta * tri_output_species +
                            self.beta * tri_output + self.mmd_beta * mmd_loss_sum +
                            self.gan_beta * loss_G_GAN + self.ot_beta * loss_ot +
                            self.structure_beta * geom_structure_loss)
                else:
                    loss = (self.mse_beta * mse_loss + self.tri_beta * tri_output_species +
                            self.mmd_beta * mmd_loss_sum + self.gan_beta * loss_G_GAN +
                            self.ot_beta * loss_ot + self.structure_beta * geom_structure_loss)

                loss.backward(retain_graph=True) #
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.auxiliary_model.parameters()),
                                               self.gradient_clipping)
                self.optimizer.step()

            # ---------- Record losses ----------
            if self.if_return_loss:
                self.loss_dict['Loss name'].extend(['Loss sum', 'MSE', 'Cross-species triplet', 'MMD'])
                self.loss_dict['Epoch'].extend([epoch] * 4)
                self.loss_dict['Loss value'].extend(
                    [loss.item(), mse_loss.item(), tri_output_species.item(), mmd_loss_sum.item()]
                )
                if self.structure_beta != 0.0:
                    self.loss_dict['Loss name'].append('ManifoldPreserve')
                    self.loss_dict['Epoch'].append(epoch)
                    self.loss_dict['Loss value'].append(geom_structure_loss.item())

            if self.verbose and epoch % 100 == 0:
                out_str = (f'Epoch {epoch:4d} | Total: {loss.item():.4f} | '
                           f'MSE: {self.mse_beta * mse_loss.item():.4f} | '
                           f'Cross-species Tri: {self.tri_beta * tri_output_species.item():.4f} | '
                           f'MMD: {self.mmd_beta * mmd_loss_sum.item():.4f} | '
                           f'GAN: {self.gan_beta * loss_G_GAN.item():.4f} | '
                           f'OT: {self.ot_beta * loss_ot.item():.4f}')
                if self.if_integrate_within_species:
                    out_str += f' | Slice Tri: {self.beta * tri_output.item():.4f}'
                if self.structure_beta != 0.0:
                    out_str += f' | Manifold: {self.structure_beta * geom_structure_loss.item():.4f}'
                print(out_str)

            # Full‑graph inference
            with torch.no_grad():
                z_list, out_list = [], []
                for batch in self.subgraph_loader:
                    batch.to(self.device)
                    z, out = self.model(batch.x, batch.edge_index, mode='all')
                    z_list.append(z[:batch.batch_size].cpu())
                    out_list.append(out[:batch.batch_size].cpu())
                z = torch.cat(z_list, dim=0)

                auxiliary_z_list = []
                for batch in self.auxiliary_subgraph_loader:
                    batch.to(self.device)
                    aux_z, _ = self.auxiliary_model(batch.x, batch.edge_index, mode='all')
                    auxiliary_z_list.append(aux_z[:batch.batch_size].cpu())
                auxiliary_z = torch.cat(auxiliary_z_list, dim=0)

            self.adata_whole.obsm['auxiliary'] = auxiliary_z.cpu().detach().numpy()
            for species_id in self.z_dict.keys():
                start = self.species_add_dict[species_id]
                end = start + self.adata_species_dict[species_id].n_obs
                self.adata_species_dict[species_id].obsm[self.key_added] = z[start:end].cpu().detach().numpy()

            # ---------- Check best model ----------
            current_loss = loss.item()
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_epoch = epoch
                self.best_embeddings = {}
                for species_id in self.adata_species_dict:
                    self.best_embeddings[species_id] = self.adata_species_dict[species_id].obsm[self.key_added].copy()
                self.best_auxiliary_embedding = self.adata_whole.obsm['auxiliary'].copy()
                if self.model_save_path is not None:
                    self.save_checkpoint(os.path.join(self.model_save_path, 'best_checkpoint.pth'))

            # Periodic UMAP
            if self.verbose and epoch % plot_epoch == 0 and self.n_epochs_species - epoch >= plot_epoch:
                if z.shape[0] >= 50000:
                    clustering_umap_downsampling(self.adata_species_dict, key_umap=self.key_added,
                                                 downsampling_rate=self.umap_downsampling_rate)
                else:
                    clustering_umap(self.adata_species_dict, key_umap=self.key_added)

    def save_checkpoint(self, path):
        """Save full training state to a checkpoint file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'auxiliary_model_state_dict': self.auxiliary_model.state_dict(),
            'D_Z_state_dict': self.D_Z.state_dict(),
            'auxiliary_D_Z_state_dict': self.auxiliary_D_Z.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'auxiliary_optimizer_D_state_dict': self.auxiliary_optimizer_D.state_dict(),
            'best_loss': self.best_loss,
            'best_embeddings': self.best_embeddings,
            'best_auxiliary_embedding': self.best_auxiliary_embedding,
            'z_dict': {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in self.z_dict.items()},
            'species_add_dict': self.species_add_dict,
            'edge_ndarray': self.edge_ndarray,
            'start_epoch': self.best_epoch + 1,
            'adata_whole_auxiliary': self.adata_whole.obsm['auxiliary'].copy()
        }
        torch.save(checkpoint, path)
        #print(f'Checkpoint saved to {path}')

    def load_checkpoint(self, path):
        """
        Load training state from a checkpoint file and prepare to resume.
        The models, optimizers, and internal state are restored so that
        training can continue from the next epoch.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.auxiliary_model.load_state_dict(checkpoint['auxiliary_model_state_dict'])
        self.D_Z.load_state_dict(checkpoint['D_Z_state_dict'])
        self.auxiliary_D_Z.load_state_dict(checkpoint['auxiliary_D_Z_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.auxiliary_optimizer_D.load_state_dict(checkpoint['auxiliary_optimizer_D_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.best_epoch = checkpoint['epoch']
        self.start_epoch = checkpoint['start_epoch']
        self.best_embeddings = checkpoint['best_embeddings']
        self.best_auxiliary_embedding = checkpoint['best_auxiliary_embedding']
        self.z_dict = {k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v
                       for k, v in checkpoint['z_dict'].items()}
        self.species_add_dict = checkpoint['species_add_dict']
        self.edge_ndarray = checkpoint['edge_ndarray']

        # Rebuild the data.x tensor from z_dict
        z_vals = [self.z_dict[sp].cpu().numpy() if torch.is_tensor(self.z_dict[sp]) else self.z_dict[sp]
                  for sp in self.z_dict]
        z = torch.FloatTensor(np.concatenate(z_vals, axis=0))
        self.data = Data(edge_index=torch.LongTensor(self.edge_ndarray), x=z)
        self.subgraph_loader = NeighborLoader(self.data, num_neighbors=[-1],
                                              batch_size=self.batch_size * 2, shuffle=False)

        if 'adata_whole_auxiliary' in checkpoint:
            self.adata_whole.obsm['auxiliary'] = checkpoint['adata_whole_auxiliary']

        print(f'Checkpoint loaded from {path}. Resuming from epoch {self.start_epoch}.')

    def finalize(self):
        """Overwrite final output with best embeddings and perform final visualization."""
        if self.best_embeddings is not None:
            for species_id, emb in self.best_embeddings.items():
                self.adata_species_dict[species_id].obsm[self.key_added] = emb
            self.adata_whole.obsm['auxiliary'] = self.best_auxiliary_embedding
            z_best = np.concatenate(list(self.best_embeddings.values()), axis=0)
            print(f'Best model at epoch {self.best_epoch} with loss {self.best_loss:.4f} used for output.')
        else:
            z_best = self.data.x.cpu().detach().numpy()
            print('No best model found. Using final model.')

        print('Clustering and UMAP of Cross Species STACAME:')
        if z_best.shape[0] >= 50000:
            clustering_umap_downsampling(self.adata_species_dict, key_umap=self.key_added,
                                         downsampling_rate=self.umap_downsampling_rate)
        else:
            clustering_umap(self.adata_species_dict, key_umap=self.key_added)

    def run(self):
        """Execute the complete training pipeline."""
        self.set_random_seeds()

        if self.resume_from_checkpoint is None:
            self.pretrain_stage()
            self.prepare_joint_training()
        else:
            print('Resuming from checkpoint, skipping pretraining stage.')
            self.prepare_joint_training()
            self.load_checkpoint(self.resume_from_checkpoint)

        self.joint_train()
        self.finalize()

        # Cleanup
        del self.model, self.auxiliary_model, self.optimizer, self.D_Z, self.auxiliary_D_Z, self.data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if self.if_return_loss:
            return self.adata_species_dict, self.loss_dict
        return self.adata_species_dict
