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
from .train_STACAME import random_list, clustering_umap, clustering_umap_downsampling
import gc

class STACAME_trainer:
    """
    Trainer class for cross-species spatial transcriptomics integration using
    STACAME with GAN-based domain confusion, auxiliary model, and optional
    manifold preserving loss. Supports checkpoint saving and resuming.

    # New traninng
    trainer = STACAME_trainer(
        adata_species_dict, triplet_ind_species_dict, edge_ndarray_species,
        model_save_path='./checkpoints',
        if_return_loss=True,
        verbose=True
    )
    result = trainer.run()  # return adata_species_dict 和 loss_dict（若 if_return_loss=True）
    
    # Break down and continue tranining
    trainer_resume = STACAME_trainer(
        adata_species_dict, triplet_ind_species_dict, edge_ndarray_species,
        model_save_path='./checkpoints',
        resume_from_checkpoint='./checkpoints/best_checkpoint.pth',
        if_return_loss=True,
        verbose=True
    )
    result_resume = trainer_resume.run()

    Parameters
    ----------
    adata_species_dict : dict
        Dictionary mapping species names to AnnData objects.
    triplet_ind_species_dict : dict
        Cross-species triplet indices.
    edge_ndarray_species : np.ndarray
        Cross-species MNN graph edges (2 x n_edges).
    triplet_ind_sections_dict : dict, optional
        Within-species slice-level triplet indices.
    edge_ndarray_sections : np.ndarray, optional
        Within-species slice MNN graph edges.
    hidden_dims : list
        Hidden layer dimensions [encoder_output_dim, bottleneck_dim].
    stagate_epoch : int or dict
        Pretraining epochs per species.
    n_epochs_species : int
        Total epochs for cross-species joint training.
    lr : float
        Learning rate for pretraining.
    key_added : str
        Key under which the final latent embedding is stored in adata.obsm.
    gradient_clipping : float
        Max gradient norm for clipping during joint training.
    weight_decay : float
        Weight decay for pretraining optimizer.
    margin : float
        Triplet loss margin for within-species slices.
    margin_species : float
        Triplet loss margin for cross-species.
    lr_species : float
        Learning rate for cross-species joint training.
    beta : float
        Weight of within-species slice triplet loss.
    verbose : bool
        If True, print detailed loss breakdown.
    random_seed : int
        Random seed for reproducibility.
    iter_comb : tuple or None
        Order of slice integration for within-species MNN.
    knn_neigh : int
        Number of nearest neighbours for MNN graph construction.
    device : torch.device
        Device for cross-species training.
    pretrain_device : torch.device
        Device for per-species pretraining.
    mse_beta : float
        Weight of MSE reconstruction loss.
    tri_beta : float
        Weight of combined triplet loss (auxiliary + cross-species).
    mmd_beta : float
        Weight of MMD loss.
    gan_beta : float
        Weight of GAN domain confusion loss.
    gan_epoch : int
        Number of discriminator updates per generator step.
    ot_beta : float
        Weight of optimal transport loss (0 to disable).
    mmd_batch_size : int
        Batch size for MMD computation.
    if_knn_mnn_graph : bool
        Whether to add cross-species MNN edges to the graph.
    if_integrate_within_species : bool
        Whether to enable within-species slice integration.
    if_return_loss : bool
        If True, return a loss dictionary.
    adata_whole : anndata.AnnData
        Concatenated AnnData object across all species.
    concate_pca_dim : int
        PCA dimension for merged gene expression before decoding.
    if_use_light_model : bool
        Whether to use a lightweight decoder variant.
    structure_beta : float
        Weight of manifold preserving loss (0 to disable).
    structure_sampling_ratio : float
        Fraction of intra-species edges sampled for structure loss.
    max_structure_edges : int
        Maximum intra-species edges per epoch for manifold loss.
    model_save_path : str, optional
        Directory to save best model checkpoint. If None, no file is saved.
    resume_from_checkpoint : str, optional
        Path to a checkpoint file to resume training from. If provided,
        pretraining is skipped and the state is loaded.
    """

    def __init__(self,
                 adata_species_dict,
                 triplet_ind_species_dict,
                 edge_ndarray_species,
                 triplet_ind_sections_dict=None,
                 edge_ndarray_sections=None,
                 hidden_dims=[256, 30],
                 stagate_epoch=500,
                 n_epochs_species=1500,
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
                 mmd_beta=5,
                 gan_beta=5,
                 gan_epoch=3,
                 ot_beta=0,
                 mmd_batch_size=1024,
                 if_knn_mnn_graph=False,
                 if_integrate_within_species=False,
                 if_return_loss=False,
                 adata_whole=None,
                 concate_pca_dim=200,
                 if_use_light_model=False,
                 structure_beta=0.0,
                 structure_sampling_ratio=1.0,
                 max_structure_edges=10000,
                 model_save_path=None,
                 resume_from_checkpoint=None):
        # Store all configuration parameters as instance attributes
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
        self.adata_whole = adata_whole
        self.concate_pca_dim = concate_pca_dim
        self.if_use_light_model = if_use_light_model
        self.structure_beta = structure_beta
        self.structure_sampling_ratio = structure_sampling_ratio
        self.max_structure_edges = max_structure_edges
        self.model_save_path = model_save_path
        self.resume_from_checkpoint = resume_from_checkpoint

        # Initialize internal training state
        self.model = None
        self.auxiliary_model = None
        self.D_Z = None
        self.auxiliary_D_Z = None
        self.optimizer = None
        self.auxiliary_optimizer_D = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_embeddings = None
        self.best_auxiliary_embedding = None
        self.start_epoch = 0
        self.species_add_dict = {}
        self.z_dict = {}
        self.edge_ndarray = None
        self.data = None
        self.auxiliary_data = None
        self.intra_edges = None
        self.node_species = None
        self.loss_dict = None

    def set_random_seeds(self):
        """Set all random seeds for reproducibility."""
        seed = self.random_seed
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def pretrain_stage(self):
        """Per‑species STAGATE pretraining to obtain z_dict."""
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
            data = data.to(self.pretrain_device)

            if species_order == 0:
                model = STACAME.STACAME(
                    hidden_dims=[data.x.shape[1], self.hidden_dims[0], self.hidden_dims[1]]
                ).to(self.pretrain_device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, foreach=False)

            species_order += 1
            print('Pretrain with STAGATE_multiple...')
            for epoch in tqdm(range(0, stagate_epoch_dict[species_id])):
                model.train()
                optimizer.zero_grad()
                z, out = model(data.x, data.edge_index)

                if self.if_integrate_within_species:
                    if epoch % 10 == 0 and epoch >= 500:
                        if self.verbose:
                            print('Update spot triplets at epoch ' + str(epoch))
                        adata.obsm['STAGATE'] = z.cpu().detach().numpy()
                        mnn_dict = create_dictionary_mnn(
                            adata, use_rep='STAGATE', batch_name='batch_name',
                            k=self.knn_neigh, iter_comb=self.iter_comb, verbose=0
                        )
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
                                    cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)]
                                )
                            batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                            anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                            positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                            negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

                        anchor_arr = z[anchor_ind,]
                        positive_arr = z[positive_ind,]
                        negative_arr = z[negative_ind,]
                        triplet_loss = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')
                        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
                        loss = F.mse_loss(data.x.to(self.pretrain_device), out) + self.beta * tri_output
                    else:
                        loss = F.mse_loss(data.x.to(self.pretrain_device), out)
                else:
                    loss = F.mse_loss(data.x.to(self.pretrain_device), out)

                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
                optimizer.step()
            print(f'Pretrain mse loss = {loss.item():.4f}')

            with torch.no_grad():
                z, _ = model(data.x, data.edge_index)
            self.adata_species_dict[species_id].obsm['STAGATE'] = z.cpu().detach().numpy()
            self.z_dict[species_id] = z.cpu().detach()

            if species_order >= len(self.adata_species_dict.keys()):
                del model, optimizer, data, z, out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    def prepare_joint_training(self):
        """Build data, models, and optimizers for the joint training phase."""
        # Extract cross-species triplet indices
        self.anchor_ind_species = self.triplet_ind_species_dict['anchor_ind_species']
        self.positive_ind_species = self.triplet_ind_species_dict['positive_ind_species']
        self.negative_ind_species = self.triplet_ind_species_dict['negative_ind_species']

        # ---- Build concatenated graph ----
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

        # Optionally add within-species slice edges
        if self.if_integrate_within_species and self.edge_ndarray_sections is not None:
            self.anchor_ind_sections = self.triplet_ind_sections_dict['anchor_ind_sections']
            self.positive_ind_sections = self.triplet_ind_sections_dict['positive_ind_sections']
            self.negative_ind_sections = self.triplet_ind_sections_dict['negative_ind_sections']
            edge_ndarray_sections = np.array([self.edge_ndarray_sections[0], self.edge_ndarray_sections[1]])
            edge_ndarray = np.concatenate((edge_ndarray, edge_ndarray_sections), axis=1)
        self.edge_ndarray = edge_ndarray

        # ---- Concatenate pretrained embeddings z ----
        S = 0
        for species_id, z_input in self.z_dict.items():
            if S == 0:
                X = self.z_dict[species_id].cpu().detach().numpy()
            else:
                X = np.concatenate((X, self.z_dict[species_id].cpu().detach().numpy()), axis=0)
            S += 1
        z = torch.FloatTensor(X)

        # ---- Merge gene expression and reduce dimensionality with PCA ----
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
            import anndata as ad
            import scanpy as sc
            adata_X = ad.AnnData(merge_X)
            sc.pp.scale(adata_X)
            sc.tl.pca(adata_X, n_comps=self.concate_pca_dim)
            merge_X = adata_X.obsm["X_pca"]
        self.merge_X = torch.FloatTensor(merge_X).to(self.device)

        # ---- Build models and discriminators ----
        if self.if_use_light_model:
            self.model = STACAME.STACAMEDecoder_light(
                hidden_dims=[self.merge_X.shape[1], self.hidden_dims[0], self.hidden_dims[1]]
            ).to(self.device)
        else:
            self.model = STACAME.STACAME_Decoder(
                hidden_dims=[self.merge_X.shape[1], self.hidden_dims[0], self.hidden_dims[1]]
            ).to(self.device)

        auxiliary_X = torch.FloatTensor(self.adata_whole.obsm['X_pca'])
        self.auxiliary_model = STACAME.STACAME(
            hidden_dims=[auxiliary_X.shape[1], self.hidden_dims[0] // 4, self.hidden_dims[1]]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.auxiliary_model.parameters()),
            lr=self.lr_species, weight_decay=self.weight_decay, foreach=False
        )

        self.auxiliary_D_Z = STACAME.MultiClassDiscriminator(self.hidden_dims[1], n_species).to(self.device)
        self.auxiliary_optimizer_D = torch.optim.Adam(
            list(self.auxiliary_D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False
        )
        self.D_Z = STACAME.MultiClassDiscriminator(self.hidden_dims[1], n_species).to(self.device)
        self.optimizer_D = torch.optim.Adam(list(self.D_Z.parameters()), lr=0.001, weight_decay=0.001, foreach=False)

        # ---- Ground truth domain labels ----
        species_list_gt = []
        for species_id, adata in self.adata_species_dict.items():
            species_list_gt.extend([species_id] * adata.n_obs)
        self.true_dom = torch.LongTensor(
            pd.Series(species_list_gt).astype('category').cat.codes.values
        ).to(self.device)

        # ---- Data containers for joint training ----
        self.auxiliary_data = Data(
            edge_index=torch.LongTensor(edge_ndarray),
            prune_edge_index=torch.LongTensor(np.array([])),
            x=auxiliary_X
        ).to(self.device)

        self.data = Data(
            edge_index=torch.LongTensor(edge_ndarray),
            prune_edge_index=torch.LongTensor(np.array([])),
            x=z
        ).to(self.device)

        # ---- Extract intra-species edges for manifold preserving loss ----
        self.intra_edges = None
        self.node_species = None
        if self.structure_beta != 0.0:
            total_nodes = z.shape[0]
            self.node_species = -1 * np.ones(total_nodes, dtype=np.int64)
            for sp_idx, species_id in enumerate(self.adata_species_dict.keys()):
                start = self.species_add_dict[species_id]
                n_cells = self.adata_species_dict[species_id].n_obs
                self.node_species[start:start + n_cells] = sp_idx
            edge_index_cpu = self.data.edge_index.cpu().numpy()
            src, dst = edge_index_cpu[0], edge_index_cpu[1]
            mask = (self.node_species[src] == self.node_species[dst])
            intra_edges_np = edge_index_cpu[:, mask]
            print(f'[Manifold] Found {intra_edges_np.shape[1]} intra-species edges '
                  f'(out of {edge_index_cpu.shape[1]} total).')
            if intra_edges_np.shape[1] > 0:
                self.intra_edges = torch.LongTensor(intra_edges_np).to(self.device)

        # ---- Other training utilities ----
        self.species_n_cells = {sp: self.adata_species_dict[sp].n_obs for sp in self.adata_species_dict.keys()}
        if self.if_return_loss:
            self.loss_dict = {'Loss name': [], 'Epoch': [], 'Loss value': []}

    def manifold_preserving_loss(self, latent_vectors, ref_vectors, edges, edge_weights=None):
        """
        Weighted Laplacian loss to preserve local geometry from ref_vectors.
        """
        src, dst = edges[0], edges[1]
        d2_ref = torch.sum((ref_vectors[src] - ref_vectors[dst]) ** 2, dim=1)
        sigma = torch.sqrt(torch.median(d2_ref) + 1e-8)
        w = torch.exp(-d2_ref / (2 * sigma ** 2 + 1e-8))
        d2_lat = torch.sum((latent_vectors[src] - latent_vectors[dst]) ** 2, dim=1)
        if edge_weights is None:
            edge_weights = torch.ones_like(w)
        return torch.mean(edge_weights * w * d2_lat)

    def joint_train(self):
        """Execute the cross-species joint training loop."""
        plot_epoch = self.n_epochs_species // 3
        for epoch in tqdm(range(self.start_epoch, self.n_epochs_species)):
            # Update STAGATE embeddings in adata (for recording only)
            k_add = 0
            for species_id in self.z_dict.keys():
                self.species_add_dict[species_id] = int(k_add)
                start = int(k_add)
                end = int(k_add + self.adata_species_dict[species_id].n_obs)
                self.adata_species_dict[species_id].obsm['STAGATE'] = self.data.x[start:end].cpu().detach().numpy()
                self.z_dict[species_id] = self.adata_species_dict[species_id].obsm['STAGATE']
                k_add = end
            if epoch == self.start_epoch:
                self.adata_whole.obsm['auxiliary'] = self.data.x.cpu().detach().numpy()

            self.model.train()
            self.auxiliary_model.train()
            self.optimizer.zero_grad()

            auxiliary_z, auxiliary_out = self.auxiliary_model(self.auxiliary_data.x, self.auxiliary_data.edge_index)
            z, out = self.model(self.data.x, self.data.edge_index)

            # 1) MSE reconstruction loss
            mse_loss = F.mse_loss(self.merge_X, out) + F.mse_loss(self.auxiliary_data.x, auxiliary_out)

            # 2) Within-species slice triplet loss
            if self.if_integrate_within_species:
                anchor_arr = z[self.anchor_ind_sections,]
                positive_arr = z[self.positive_ind_sections,]
                negative_arr = z[self.negative_ind_sections,]
                tri_output = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')(
                    anchor_arr, positive_arr, negative_arr)
            else:
                tri_output = torch.tensor(0.0, device=self.device)

            # 3) Cross-species triplet loss
            anchor_arr_species = z[self.anchor_ind_species,]
            positive_arr_species = z[self.positive_ind_species,]
            negative_arr_species = z[self.negative_ind_species,]
            tri_output_species = torch.nn.TripletMarginLoss(margin=self.margin_species, p=2, reduction='mean')(
                anchor_arr_species, positive_arr_species, negative_arr_species)

            # 4) MMD loss
            mmd_loss_fn = STACAME.MMDLoss(kernel=STACAME.RBF(device=self.device), device=self.device).to(self.device)
            mmd_loss_sum = 0.0
            for species_id in self.z_dict.keys():
                k_add = self.species_add_dict[species_id]
                end = int(k_add + self.adata_species_dict[species_id].n_obs)
                remain = list(set(range(z.shape[0])) - set(range(k_add, end)))
                ind_1 = random.sample(list(range(k_add, end)), self.mmd_batch_size)
                ind_2 = random.sample(remain, self.mmd_batch_size)
                mmd_loss_sum += mmd_loss_fn(z[ind_1,], z[ind_2,])
                mmd_loss_sum += mmd_loss_fn(auxiliary_z[ind_1,], auxiliary_z[ind_2,])

            # 5) Optimal transport loss (optional)
            loss_ot = torch.tensor(0.0, device=self.device)
            if self.ot_beta != 0:
                z_A = z[ind_1,]
                z_B = z[ind_2,]
                x_A = self.auxiliary_data.x[ind_1,]
                x_B = self.auxiliary_data.x[ind_2,]
                c_cross = pairwise_correlation_distance(x_A.detach(), x_B.detach()).to(self.device)
                T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=self.device)
                z_dist = torch.mean((z_A.view(self.mmd_batch_size, 1, -1) - z_B.view(1, self.mmd_batch_size, -1)) ** 2,
                                    dim=2)
                loss_ot = torch.sum(T * z_dist) / torch.sum(T)

            # 6) Update auxiliary triplets every 100 epochs
            sampling_num_spe = anchor_arr_species.shape[0]
            if epoch % 100 == 0:
                mnn_dict = create_dictionary_mnn(self.adata_whole, use_rep='auxiliary',
                                                 batch_name='species_id', k=self.knn_neigh,
                                                 iter_comb=self.iter_comb, verbose=0)
                anchor_ind = []
                positive_ind = []
                negative_ind = []
                species_ids = list(self.adata_species_dict.keys())
                for batch_pair in mnn_dict.keys():
                    batchname_list = self.adata_whole.obs['species_id'][mnn_dict[batch_pair].keys()]
                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(species_ids)):
                        cellname_by_batch_dict[species_ids[batch_id]] = self.adata_whole.obs_names[
                            self.adata_whole.obs['species_id'] == species_ids[batch_id]].values
                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for anchor in mnn_dict[batch_pair].keys():
                        anchor_list.append(anchor)
                        positive_list.append(mnn_dict[batch_pair][anchor][0])
                        section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                        negative_list.append(
                            cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)]
                        )
                    batch_as_dict = dict(zip(list(self.adata_whole.obs_names), range(0, self.adata_whole.shape[0])))
                    anchor_ind.extend(map(lambda _: batch_as_dict[_], anchor_list))
                    positive_ind.extend(map(lambda _: batch_as_dict[_], positive_list))
                    negative_ind.extend(map(lambda _: batch_as_dict[_], negative_list))

                # Store updated triplets for subsequent epochs
                current_anchor = torch.LongTensor(anchor_ind).to(self.device)
                current_pos = torch.LongTensor(positive_ind).to(self.device)
                current_neg = torch.LongTensor(negative_ind).to(self.device)
                self.triplet_cache = (current_anchor, current_pos, current_neg)
            else:
                # Use previously stored triplet indices
                if hasattr(self, 'triplet_cache'):
                    current_anchor, current_pos, current_neg = self.triplet_cache
                else:
                    # Fallback for the very first call (should not happen if update logic runs first)
                    current_anchor = self.anchor_ind_species
                    current_pos = self.positive_ind_species
                    current_neg = self.negative_ind_species

            # 7) Auxiliary triplet loss
            tri_auxiliary = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')(
                z[current_anchor,], z[current_pos,], z[current_neg,]
            ) + torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')(
                auxiliary_z[current_anchor,], auxiliary_z[current_pos,], auxiliary_z[current_neg,]
            )

            # 8) GAN domain confusion loss
            if self.gan_beta != 0:
                for _ in range(self.gan_epoch):
                    self.optimizer_D.zero_grad()
                    logits_D = self.D_Z(z)
                    loss_D = F.cross_entropy(logits_D, self.true_dom)
                    loss_D.backward(retain_graph=True)
                    self.optimizer_D.step()
                for _ in range(self.gan_epoch):
                    self.auxiliary_optimizer_D.zero_grad()
                    auxiliary_logits_D = self.auxiliary_D_Z(auxiliary_z)
                    auxiliary_loss_D = F.cross_entropy(auxiliary_logits_D, self.true_dom)
                    auxiliary_loss_D.backward(retain_graph=True)
                    self.auxiliary_optimizer_D.step()
            # 8) GAN domain confusion loss
            # if self.gan_beta != 0:
            #     z_detached = z.detach()
            #     auxiliary_z_detached = auxiliary_z.detach()
            
            #     for _ in range(self.gan_epoch):
            #         self.optimizer_D.zero_grad()
            #         logits_D = self.D_Z(z_detached)
            #         loss_D = F.cross_entropy(logits_D, self.true_dom)
            #         loss_D.backward()          
            #         self.optimizer_D.step()
            
            #     for _ in range(self.gan_epoch):
            #         self.auxiliary_optimizer_D.zero_grad()
            #         auxiliary_logits_D = self.auxiliary_D_Z(auxiliary_z_detached)
            #         auxiliary_loss_D = F.cross_entropy(auxiliary_logits_D, self.true_dom)
            #         auxiliary_loss_D.backward()
            #         self.auxiliary_optimizer_D.step()
            
            # Generator adversarial loss uses the original z (with gradients) to update the generator
            loss_G_GAN = -F.cross_entropy(self.D_Z(z), self.true_dom) - F.cross_entropy(
                self.auxiliary_D_Z(auxiliary_z), self.true_dom)

            # 9) Manifold preserving loss
            geom_structure_loss = torch.tensor(0.0, device=self.device)
            if self.structure_beta != 0.0 and self.intra_edges is not None:
                num_intra = self.intra_edges.shape[1]
                if num_intra > 0:
                    n_sample = min(int(num_intra * self.structure_sampling_ratio), self.max_structure_edges)
                    n_sample = max(1, n_sample)
                    if n_sample < num_intra:
                        perm = torch.randperm(num_intra, device=self.device)[:n_sample]
                        sampled_edges = self.intra_edges[:, perm]
                    else:
                        sampled_edges = self.intra_edges
                    src_nodes = sampled_edges[0].cpu().numpy()
                    edge_species = self.node_species[src_nodes]
                    inv_n = {sp: 1.0 / self.species_n_cells[sp] for sp in self.species_n_cells}
                    species_list_local = list(self.adata_species_dict.keys())
                    raw_weights = torch.tensor([inv_n[species_list_local[sp_idx]] for sp_idx in edge_species],
                                               device=self.device, dtype=torch.float32)
                    edge_w = raw_weights * (len(raw_weights) / raw_weights.sum())
                    loss_z = self.manifold_preserving_loss(z, self.data.x, sampled_edges, edge_weights=edge_w)
                    loss_aux = self.manifold_preserving_loss(auxiliary_z, self.data.x, sampled_edges, edge_weights=edge_w)
                    geom_structure_loss = loss_z + loss_aux

            # ---- Total loss ----
            if self.if_integrate_within_species:
                loss = (self.mse_beta * mse_loss +
                        self.tri_beta * (tri_auxiliary + 0.1 * tri_output_species) +
                        self.beta * tri_output +
                        self.mmd_beta * mmd_loss_sum +
                        self.gan_beta * loss_G_GAN +
                        self.ot_beta * loss_ot +
                        self.structure_beta * geom_structure_loss)
            else:
                loss = (self.mse_beta * mse_loss +
                        self.tri_beta * (tri_auxiliary + 0.1 * tri_output_species) +
                        self.mmd_beta * mmd_loss_sum +
                        self.gan_beta * loss_G_GAN +
                        self.ot_beta * loss_ot +
                        self.structure_beta * geom_structure_loss)

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.auxiliary_model.parameters()),
                self.gradient_clipping
            )
            self.optimizer.step()

            # Record losses if requested
            if self.if_return_loss:
                self.loss_dict['Loss name'].extend(['Loss sum', 'MSE', 'Cross-species triplet', 'MMD'])
                self.loss_dict['Epoch'].extend([epoch, epoch, epoch, epoch])
                self.loss_dict['Loss value'].extend(
                    [loss.item(), mse_loss.item(), tri_output_species.item(), mmd_loss_sum.item()]
                )
                if self.ot_beta > 0:
                    self.loss_dict['Loss name'].append('OT')
                    self.loss_dict['Epoch'].append(epoch)
                    self.loss_dict['Loss value'].append(loss_ot.item())
                if self.structure_beta != 0.0:
                    self.loss_dict['Loss name'].append('ManifoldPreserve')
                    self.loss_dict['Epoch'].append(epoch)
                    self.loss_dict['Loss value'].append(geom_structure_loss.item())

            # Update embeddings in AnnData objects
            for species_id in self.z_dict.keys():
                start = self.species_add_dict[species_id]
                end = start + self.adata_species_dict[species_id].n_obs
                self.adata_species_dict[species_id].obsm[self.key_added] = z[start:end].cpu().detach().numpy()
            self.adata_whole.obsm['auxiliary'] = auxiliary_z.cpu().detach().numpy()

            # Check and save best model
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

            # Verbose logging
            if self.verbose and epoch % 100 == 0:
                print(f'---------------------------------Epoch {epoch:4d}-----------------------------------')
                print(f'Total loss: {loss.item():.4f}| '
                      f'MSE (weighted): {self.mse_beta * mse_loss.item():.4f}| '
                      f'Cross-species Tri: {self.tri_beta * tri_output_species.item():.4f}| '
                      f'Auxiliary Tri: {self.tri_beta * tri_auxiliary.item():.4f}| '
                      f'MMD (weighted): {self.mmd_beta * mmd_loss_sum.item():.4f}| '
                      f'GAN (weighted): {self.gan_beta * loss_G_GAN.item():.4f}')
                if self.ot_beta > 0:
                    print(f'OT (weighted): {self.ot_beta * loss_ot.item():.4f}')
                if self.structure_beta != 0.0:
                    print(f'Manifold (weighted): {self.structure_beta * geom_structure_loss.item():.4f}')
                if self.if_integrate_within_species:
                    print(f'Cross-slices Tri: {self.beta * tri_output.item():.4f}')
                    if self.if_return_loss:
                        self.loss_dict['Loss name'].append('Cross-slices triplet')
                        self.loss_dict['Epoch'].append(epoch)
                        self.loss_dict['Loss value'].append(tri_output.item())

            # Periodic UMAP visualization
            if self.verbose and epoch % plot_epoch == 0 and self.n_epochs_species - epoch >= plot_epoch:
                if z.shape[0] >= 50000:
                    clustering_umap_downsampling(self.adata_species_dict, key_umap=self.key_added,
                                                 downsampling_rate=0.1)
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
            'auxiliary_optimizer_D_state_dict': self.auxiliary_optimizer_D.state_dict(),
            'best_loss': self.best_loss,
            'best_embeddings': self.best_embeddings,
            'best_auxiliary_embedding': self.best_auxiliary_embedding,
            'z_dict': {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in self.z_dict.items()},
            'species_add_dict': self.species_add_dict,
            'edge_ndarray': self.edge_ndarray,
            'node_species': self.node_species,
            'start_epoch': self.best_epoch + 1,
            'adata_whole_auxiliary': self.adata_whole.obsm['auxiliary'].copy() if 'auxiliary' in self.adata_whole.obsm else None,
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
        self.node_species = checkpoint['node_species']

        # Reconstruct data.x from the loaded z_dict
        z_vals = [self.z_dict[sp].cpu().numpy() if torch.is_tensor(self.z_dict[sp]) else self.z_dict[sp]
                  for sp in self.z_dict]
        z = torch.FloatTensor(np.concatenate(z_vals, axis=0)).to(self.device)
        self.data = Data(edge_index=torch.LongTensor(self.edge_ndarray), x=z).to(self.device)

        # Restore embeddings in AnnData objects
        for species_id in self.z_dict:
            start = self.species_add_dict[species_id]
            end = start + self.adata_species_dict[species_id].n_obs
            self.adata_species_dict[species_id].obsm['STAGATE'] = z[start:end].cpu().detach().numpy()

        if 'adata_whole_auxiliary' in checkpoint and checkpoint['adata_whole_auxiliary'] is not None:
            self.adata_whole.obsm['auxiliary'] = checkpoint['adata_whole_auxiliary']

        # Rebuild intra_edges if structure loss is enabled
        if self.structure_beta != 0.0 and self.node_species is not None:
            edge_index_cpu = self.data.edge_index.cpu().numpy()
            src, dst = edge_index_cpu[0], edge_index_cpu[1]
            mask = (self.node_species[src] == self.node_species[dst])
            intra_edges_np = edge_index_cpu[:, mask]
            if intra_edges_np.shape[1] > 0:
                self.intra_edges = torch.LongTensor(intra_edges_np).to(self.device)

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

        print('Final clustering and UMAP:')
        if z_best.shape[0] >= 50000:
            clustering_umap_downsampling(self.adata_species_dict, key_umap=self.key_added, downsampling_rate=0.1)
        else:
            clustering_umap(self.adata_species_dict, key_umap=self.key_added)

    def run(self):
        """Execute the complete training pipeline."""
        self.set_random_seeds()

        if self.resume_from_checkpoint is None:
            # Fresh training: pretrain first, then prepare and train
            self.pretrain_stage()
            self.prepare_joint_training()
        else:
            # Resume from checkpoint: skip pretraining, prepare structures and load state
            print('Resuming from checkpoint, skipping pretraining stage.')
            self.prepare_joint_training()
            self.load_checkpoint(self.resume_from_checkpoint)

        self.joint_train()
        self.finalize()

        # Clean up
        del self.model, self.auxiliary_model, self.optimizer, self.D_Z, self.auxiliary_D_Z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if self.if_return_loss:
            return self.adata_species_dict, self.loss_dict
        return self.adata_species_dict