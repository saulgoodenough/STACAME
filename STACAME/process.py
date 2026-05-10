
import os
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix, coo_matrix
import sklearn
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import random
from tqdm import tqdm
import sys
sys.path.append('../../STACAME/')
import STACAME
from STACAME import create_dictionary_mnn, get_species_triples
from STACAME.train_STACAME import train_STACAME
from collections import Counter
from scipy.sparse import hstack
import timeit
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler, MinMaxScaler, StandardScaler


class STACAME_processer():
    def __init__(self, 
                 root_data_path,
                 Gene_map_raw_path, 
                 species_section_ids, 
                 species_ortholog_column_dict, 
                 species_ortholog_type_dict, 
                 species_id_map, 
                 rad_cutoff_dict,
                 gene_cap_upper_dict = {'Mouse':'capitalize', 'Macaque':'upper'},
                 total_normalize = 1e4,
                 log_normalize_dict = None,
                 min_cells = 20,
                 Down_sampling_adata = None, 
                 n_top_genes = 2000, 
                 homo_n_top_genes = 4000, 
                 cross_species_neibors_K_mnn = 20,
                 cross_sections_neibors_K_mnn = 20,
                 cross_species_neibors_K_knn = 1, 
                 Smooth_spatial_neighbors = None, 
                 knn_triplets = False, 
                 knn_triplets_ratio=0.05, 
                 if_hvg_before_mnn = False, 
                 if_combat_mnn = False, 
                 if_pca_before_mnn = False, 
                 gene_save_path = './output_STACAME/', 
                 if_integrate_within_species = False, 
                 if_return_concat_adata = False, 
                 pca_dim_before_mnn = 50, 
                 graph_construct_key = 'spatial'):
    
        """ Run the main data preprocess of STACAME, for integrating multiple datasets from multiple species
        Parameters
        ----------
    
        Gene_map_raw_path
            file path of gene orthologs of multiple species, e.g., './Data/Mouse_Macaque.tsv'
        species_section_ids
            dict of sections for each species, e.g., species_section_ids = {'Mouse':['T315'], 'Macaque':['T36']}
        species_ortholog_column_dict
            dict of gene column name in the Gene_map_raw_path, e.g., species_ortholog_column_dict = {'Mouse':'Gene name', 'Macaque':'Macaque gene name'}
        species_ortholog_type_dict
            dict of gene homology type column name, e.g., species_ortholog_type_dict = {'Macaque':'Macaque homology type'}
        species_id_map
            ID of each species, e.g., species_id_map = {'Mouse':0, 'Macaque':1}):
        rad_cutoff_dict
            An important hyper-paramater, adjust the range of MNN. We advice to check the nearbor number to determine it. For example,  rad_cutoff_dict = {'Mouse':1.3, 'Macaque':1.3}.
        gene_cap_upper_dict
            Gene name characater form, e.g., gene_cap_upper_dict = {'Mouse':'capitalize', 'Macaque':'upper'}.
        log_normalize_dict
            Choose whether to apply log on the data, e.g., log_normalize_dict = {'Mouse':True, 'Macaque':False}
        Down_sampling_adata
            Downsampling rate of adata for light running, e.g., 0.1. Default: None.
        n_top_genes
            HVG number for each species (including homologous genes and species-specific genes)
        homo_n_top_genes
            Aligned one-to-one homologous genes
        cross_species_neibors_K_mnn
            MNN K parameter. This parameter needs to be well-adjusted.
        cross_species_neibors_K_knn
            KNN triplets paramater, and we advice to use 1, since KNN triplets increase rapidly as K increases. Default 1.
        Smooth_spatial_neighbors
            Neigborhood range for average when computing cross-species distance on multi-multi orthologs. Default 2.
        knn_triplets
            Whether to choose to use KNN triplets. Default True.
        knn_triplets_ratio
            The downsampling ratio of KNN triplets when using KNN triplets.
        Returns
        -------
        outputs: dict of anndata, the keys are species names 
        """
        self.root_data_path = root_data_path
        self.Gene_map_raw_path = Gene_map_raw_path
        self.species_section_ids = species_section_ids
        self.species_ortholog_column_dict = species_ortholog_column_dict
        self.species_ortholog_type_dict = species_ortholog_type_dict
        self.species_id_map = species_id_map
        self.rad_cutoff_dict = rad_cutoff_dict
        self.gene_cap_upper_dict = gene_cap_upper_dict
        self.Down_sampling_adata = Down_sampling_adata
        self.n_top_genes = n_top_genes
        self.homo_n_top_genes = homo_n_top_genes
        self.cross_species_neibors_K_mnn = cross_species_neibors_K_mnn
        self.cross_sections_neibors_K_mnn = cross_sections_neibors_K_mnn
        self.cross_species_neibors_K_knn = cross_species_neibors_K_knn
        self.Smooth_spatial_neighbors = Smooth_spatial_neighbors
        self.knn_triplets = knn_triplets
        self.knn_triplets_ratio = knn_triplets_ratio
        self.if_hvg_before_mnn = if_hvg_before_mnn
        self.if_combat_mnn = if_combat_mnn
        self.if_pca_before_mnn = if_pca_before_mnn
        self.total_normalize = total_normalize
        self.gene_save_path = gene_save_path
        self.if_integrate_within_species = if_integrate_within_species
        self.if_return_concat_adata = if_return_concat_adata
        self.pca_dim_before_mnn = pca_dim_before_mnn
        self.graph_construct_key = graph_construct_key
        
        if total_normalize == None:
            self.total_normalize_dict = {}
            for species_id in species_section_ids.keys():
                self.total_normalize_dict[species_id] = None
        else:
            self.total_normalize_dict = total_normalize
        if log_normalize_dict == None:
            self.log_normalize_dict = {}
            for species_id in species_section_ids.keys():
                self.log_normalize_dict[species_id] = True
        else:
            self.log_normalize_dict = log_normalize_dict

        if Smooth_spatial_neighbors == None:
            self.Smooth_spatial_neighbors = {}
            for species_id in species_section_ids.keys():
                self.Smooth_spatial_neighbors[species_id] = 0
        else:
            self.Smooth_spatial_neighbors = Smooth_spatial_neighbors

        self.min_cells = min_cells

        # Process of self.rad_cutoff_dict
        for k, v in rad_cutoff_dict.items():
            if isinstance(v, (int, float)):
                self.rad_cutoff_dict[k] = {x:y for x,y in zip(self.species_section_ids[k], [v] * len(self.species_section_ids[k]))}
            elif len(v) == len(self.species_section_ids[k]):
                pass
            else:
                raise Exception("The element value of rad_cutoff_dict should be a dict of length of 1 or number of sections.")
        print('self.rad_cutoff_dict:', self.rad_cutoff_dict)
    
    def load_process_adata(self):
        start = timeit.default_timer()
        ## Load gene orthologs and remove rows with na
        Gene_map_raw_path = self.Gene_map_raw_path
        Gene_map_raw_df = pd.read_csv(Gene_map_raw_path, sep='\t')
        Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df['Gene name'].notna()]
        for v in self.species_ortholog_column_dict.values():
            Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df[v].notna()]

        species_common_hvg_dict = {k:[] for k in self.species_section_ids.keys()}
        ## load adata and pick multi-to-multi homologous genes
        species_common_gene_list_dict = {}
        for species_id in self.species_section_ids.keys():
            section_ids = self.species_section_ids[species_id]
            print(f'--------------------------Species-{species_id}-------------------------------')
            gene_set = []
            for section_id in section_ids:
                print('Species:', species_id, 'Section:', section_id)
                adata = sc.read_h5ad(os.path.join(f'{self.root_data_path}{species_id}', section_id + '.h5ad'))
                print(adata.obsm[self.graph_construct_key].shape)
                adata.X = csr_matrix(adata.X)
                adata.var_names_make_unique(join="++")
                print('Before flitering: ', adata.shape)
                sc.pp.filter_genes(adata, min_cells=self.min_cells)
                print('After flitering: ', adata.shape)
                if len(gene_set) == 0:
                    gene_set = adata.var_names
                else:
                    gene_set = list(set(gene_set).intersection(set(adata.var_names)))
                print('Number of genes:', len(adata.var_names))
            species_common_gene_list_dict[species_id] = gene_set
            # make spot name unique
            hvg_set = []
            for section_id in section_ids:
                adata = sc.read_h5ad(os.path.join(f'{self.root_data_path}{species_id}', section_id + '.h5ad'))
                adata.X = csr_matrix(adata.X)
                adata.var_names_make_unique(join="++")
                print('Before flitering: ', adata.shape)
                sc.pp.filter_genes(adata, min_cells=self.min_cells)
                print('After flitering: ', adata.shape)
                adata.obs_names = [x+'_'+species_id + '_' + section_id for x in adata.obs_names]
                # Normalization
                adata = adata[:, gene_set]
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=np.min([len(gene_set), 5000]))
                adata = adata[:, adata.var['highly_variable']]
                if len(hvg_set) == 0:
                    hvg_set = adata.var_names
                else:
                    hvg_set = list(set(hvg_set).union(set(adata.var_names)))
                print('Number of hvgs:', len(adata.var_names))
            species_common_hvg_dict[species_id] = hvg_set
            print('Number of common hvgs:', len(hvg_set))


        ## Union of ortholog genes
        Gene_map_common_df = Gene_map_dropna_df
        for species_id, column_name in self.species_ortholog_column_dict.items():
            common_gene_list = species_common_gene_list_dict[species_id]
            Gene_map_common_df = Gene_map_common_df[Gene_map_common_df[column_name].isin(common_gene_list)]
        
        species_orthologs_hvg_union_dict = {}
        for species_id, column_name in self.species_ortholog_column_dict.items():
            orthologs_gene_list = set(Gene_map_common_df[column_name])
            hvg_list = set(list(species_common_hvg_dict[species_id]))
            species_orthologs_hvg_union_dict[species_id] = list(hvg_list.union(orthologs_gene_list))

        ## Normalizing data
        print('Normalizing data and get spatial neigbors...')
        Batch_dict = {k:[] for k in self.species_section_ids.keys()}
        A_adj_dict = {k:[] for k in self.species_section_ids.keys()}        
        for species_id, column_name in self.species_ortholog_column_dict.items():
            section_ids = self.species_section_ids[species_id]
            print(f'--------------------------Species-{species_id}-------------------------------')
            gene_set = species_orthologs_hvg_union_dict[species_id]
            hvg_set = list(species_common_hvg_dict[species_id])
            orthologs_set = Gene_map_common_df[column_name]
            for section_id in section_ids:
                adata = sc.read_h5ad(os.path.join(f'{self.root_data_path}{species_id}', section_id + '.h5ad'))
                print(f'---------Section-{section_id}---------')
                # Downsampling the datasets
                if self.Down_sampling_adata != None and self.Down_sampling_adata < 1:
                    sc.pp.subsample(adata, fraction=self.Down_sampling_adata)
                adata.var_names_make_unique(join="++")
                #sc.pp.filter_genes(adata, min_cells=50)
                adata.X = csr_matrix(adata.X)
                adata.obs_names = [x+'_'+species_id + '_' + section_id for x in adata.obs_names]
                # Select common genes for different slices
                adata = adata[:, gene_set]
                STACAME.Cal_Spatial_Net(adata, rad_cutoff=self.rad_cutoff_dict[species_id][section_id], use_key = self.graph_construct_key)
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
                if self.total_normalize != None:
                    if self.total_normalize_dict[species_id] != None:
                        sc.pp.normalize_total(adata, target_sum=self.total_normalize_dict[species_id])
                if self.log_normalize_dict[species_id] == True:
                    sc.pp.log1p(adata)
                Batch_dict[species_id].append(adata)
                A_adj_dict[species_id].append(adata.uns['adj'])

        ## Aquire all the spot name and map them to integrar ID
        print('Aquire all the spot name and map them to integrar ID...')
        spot_name_all_list = []
        spot_name_species_dict = {k:[] for k in self.species_section_ids.keys()}
        for species_id, sections_adata in Batch_dict.items():
            species_spot_list = []
            for adata in sections_adata:
                spot_name_all_list = spot_name_all_list + list(adata.obs_names)
                species_spot_list = species_spot_list + list(adata.obs_names)
            spot_name_species_dict[species_id] = species_spot_list
        spotname2id = {k:v for k,v in zip(spot_name_all_list, range(len(spot_name_all_list)))}
        id2spotname = {k:v for k,v in zip(range(len(spot_name_all_list)), spot_name_all_list)}

        ##################################################################
        ## Concat the scanpy objects for multiple slices of the same species
        print('Concat the scanpy objects for multiple slices of the same species...')
        adata_dict = {k:[] for k in self.species_section_ids.keys()}
        adata_species_section_num_dict = {k:[] for k in self.species_section_ids.keys()}
        for species_id, Batch_list in Batch_dict.items():
            section_ids = self.species_section_ids[species_id]
            adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
            adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
            adata_concat.obs["species_id"] = species_id
            #print('adata_concat.shape: ', adata_concat.shape)
            adata_dict[species_id] = adata_concat
            for section_adata in Batch_list:
                adata_species_section_num_dict[species_id].append(section_adata.n_obs)
        
        for species_id, adata_concat in adata_dict.items():
            sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=self.n_top_genes)
            if self.if_integrate_within_species:
                #sc.pp.combat(adata_concat, key='slice_name', covariates=None, inplace=True)
                sc.tl.pca(adata_concat, svd_solver='arpack', n_comps=64)
                adata_concat.X = scipy.sparse.csr_matrix(adata_concat.X)
            adata_dict[species_id] = adata_concat
        
        #####################################################
        ## Concat the spatial network for multiple slices of the same species
        print('Concat the spatial network for multiple slices of the same species...')
        #adj_concat = np.asarray(adj_list[0].todense())
        k = 0
        for species_id, adj_list in A_adj_dict.items():
            section_ids = self.species_section_ids[species_id]
            if k == 0:
                adj_concat_total = adj_list[0]
                adj_concat = adj_list[0]
                for batch_id in range(1, len(section_ids)):
                    adj_concat = scipy.sparse.block_diag((adj_concat, adj_list[batch_id]))
                    adj_concat_total = scipy.sparse.block_diag((adj_concat_total, adj_list[batch_id]))
                adata_dict[species_id].uns['edgeList'] = adj_concat.nonzero()
            else:
                adj_concat = adj_list[0]
                adj_concat_total = scipy.sparse.block_diag((adj_concat_total, adj_list[0]))
                for batch_id in range(1, len(section_ids)):
                    adj_concat = scipy.sparse.block_diag((adj_concat, adj_list[batch_id]))
                    adj_concat_total = scipy.sparse.block_diag((adj_concat_total, adj_list[batch_id]))
                adata_dict[species_id].uns['edgeList'] = adj_concat.nonzero()
            k += 1
        
        #########################################################
        ## Get KNN cross-species triplets
        # Save the cosine similarity matrix for each pair of species
        Species_cross_B_dict = {k:{} for k in adata_dict.keys()}
        N_spot = 0
        species_spot_num_dict = {}
        species_spot_add_dict = {}
        for species_id, adata in adata_dict.items():
            species_spot_add_dict[species_id] = N_spot
            N_spot += adata_dict[species_id].n_obs
            species_spot_num_dict[species_id] = adata_dict[species_id].n_obs
        print(f'N_spot = {N_spot}')
        species_id_list = list(adata_dict.keys())
        # X,Y,V for crs_matrix
        Row = []
        Col = []
        Val = []
        edge_ndarray_species = None
        if self.knn_triplets:
            print('------------Get KNN cross-species triplets--------------')
            # Here, the adjacent graph should be consistent with the cross-species triplets
            for k_species_order in range(len(species_id_list)-1):
                k_species = species_id_list[k_species_order]
                k_adata = adata_dict[k_species]
                spatial_coords_k = np.array(k_adata.obsm[self.graph_construct_key])
                k_column_name = self.species_ortholog_column_dict[k_species]
                k_orthologs = list(Gene_map_common_df[k_column_name])
                mat_k = k_adata[:, k_orthologs].X
                if self.Smooth_spatial_neighbors[k_species] >= 1:
                    mat_k = average_spatial_neighbors(np.array(mat_k.todense()), spatial_coords_k, self.Smooth_spatial_neighbors[k_species])
                else:
                    mat_k = np.array(mat_k.todense())
                spot_id_list_k = [spotname2id[s] for s in k_adata.obs_names]
                k_section_num_list = adata_species_section_num_dict[k_species]
                
                for v_species_other_order in range(k_species_order+1, len(species_id_list)):
                    v_species = species_id_list[v_species_other_order]
                    v_adata = adata_dict[v_species]
                    spatial_coords_v = np.array(v_adata.obsm[self.graph_construct_key])
                    v_column_name = self.species_ortholog_column_dict[v_species]
                    v_orthologs = list(Gene_map_common_df[v_column_name])
                    # Calculate distance and convert them to triples
                    mat_v = v_adata[:, v_orthologs].X
                    if self.Smooth_spatial_neighbors[v_species] >= 1:
                        mat_v = average_spatial_neighbors(np.array(mat_v.todense()), spatial_coords_v, self.Smooth_spatial_neighbors[v_species])
                    else:
                        mat_v = np.array(mat_v.todense())
    
                    spot_id_list_v = [spotname2id[s] for s in v_adata.obs_names]
                    v_section_num_list = adata_species_section_num_dict[v_species]
                    # The dimension is not supported to be too high
                    # How to calculate distance is a problem worthy of deep consideration, 
                    # since distance in high-dimension is not working.
                    ## matrix removing batch effect
                    mat_k_num = np.shape(mat_k)[0]
                    mat_v_num = np.shape(mat_v)[0]
                    adata_kv = ad.AnnData(np.concatenate((mat_k, mat_v), axis=0))
                    adata_kv.obs['species'] = ['species_1'] * mat_k_num + ['species_2'] * mat_v_num
                    sc.pp.combat(adata_kv, key='species', covariates=None, inplace=True)
                    
                    mat_kv_all = adata_kv.X
                    mat_k = mat_kv_all[0:mat_k_num]
                    mat_v = mat_kv_all[mat_k_num: mat_k_num + mat_v_num]
                    
                    kv_sparse_mat = cosine_similarity(mat_k, mat_v, dense_output=False)
                    # Save submatrix for each pair of species
                    Species_cross_B_dict[k_species][v_species] = sparse.csr_matrix(kv_sparse_mat)
                    Species_cross_B_dict[v_species][k_species] = sparse.csr_matrix(kv_sparse_mat).T
                    k_add = spot_id_list_k[0]
                    v_add = spot_id_list_v[0]
                    # Select the k top neigbors for each species pairs
                    ######################################################
                    k_num_sum = 0
                    for k_num in k_section_num_list:
                        v_num_sum = 0
                        for v_num in v_section_num_list:
                            zero_mat_prod = np.zeros(kv_sparse_mat.shape)
                            zero_mat_prod[:, v_num_sum:int(v_num_sum+v_num)] = 1
                            kv_sparse_mat_temp = kv_sparse_mat * zero_mat_prod
                            indices_max_kv = np.argsort(kv_sparse_mat_temp, axis=1)[:, -self.cross_species_neibors_K_knn:]
                            print('indices_max_kv.shape:', indices_max_kv.shape)
                            kv_mat_ones = np.zeros(kv_sparse_mat_temp.shape)
                            for i in range(indices_max_kv.shape[0]):
                                for j in range(indices_max_kv.shape[1]):
                                    kv_mat_ones[i][indices_max_kv[i, j]] = 1
                            edge_ndarray_s = np.nonzero(kv_mat_ones)
                            print('edge_ndarray_s', edge_ndarray_s)
                            edge_ndarray_temp = list(edge_ndarray_s)
                            edge_ndarray_temp[0] = edge_ndarray_temp[0] + k_add
                            edge_ndarray_temp[1] = edge_ndarray_temp[1] + v_add
                            if edge_ndarray_species == None:
                                edge_ndarray_species = edge_ndarray_temp
                            else:
                                edge_ndarray_species = [np.concatenate((edge_ndarray_species[0], edge_ndarray_temp[0])), \
                                        np.concatenate((edge_ndarray_species[1], edge_ndarray_temp[1]))]
                            v_num_sum = v_num_sum + v_num
                        k_num_sum = k_num_sum + k_num
                    ## Transpose matrix
                    v_num_sum = 0
                    for v_num in v_section_num_list:
                        k_num_sum = 0
                        for k_num in k_section_num_list:
                            zero_mat_prod = np.zeros(kv_sparse_mat.T.shape)
                            zero_mat_prod[:, k_num_sum:int(k_num_sum+k_num)] = 1
                            kv_sparse_mat_temp = kv_sparse_mat.T * zero_mat_prod
                            indices_max_kv = np.argsort(kv_sparse_mat_temp, axis=1)[:, -self.cross_species_neibors_K_knn:]
                            kv_mat_ones = np.zeros(kv_sparse_mat_temp.shape)
                            for i in range(indices_max_kv.shape[0]):
                                for j in range(indices_max_kv.shape[1]):
                                    kv_mat_ones[i][indices_max_kv[i, j]] = 1
                            edge_ndarray_s = np.nonzero(kv_mat_ones)
                            edge_ndarray_temp = list(edge_ndarray_s)
                            edge_ndarray_temp[0] = edge_ndarray_temp[0] + v_add
                            edge_ndarray_temp[1] = edge_ndarray_temp[1] + k_add
                            if edge_ndarray_species == None:
                                edge_ndarray_species = edge_ndarray_temp
                            else:
                                edge_ndarray_species = [np.concatenate((edge_ndarray_species[0], edge_ndarray_temp[0])), \
                                        np.concatenate((edge_ndarray_species[1], edge_ndarray_temp[1]))]
                            k_num_sum = k_num_sum + k_num
                        v_num_sum = v_num_sum + v_num
            
            
            edge_ndarray_species = tuple(edge_ndarray_species)
        else:
            print('------------Skipped KNN cross-species triplets--------------')
            pass
        ##########################################################
        ## Generate cross-species MNN triplets
        print('Generate cross-species MNN triplets...')
        spe = 0
        for species_id, adata in adata_dict.items():
            homologs_column_name = self.species_ortholog_column_dict[species_id]
            orthologs_name = list(Gene_map_common_df[homologs_column_name])
            if spe == 0:
                whole_obs_name = list(adata.obs_names)
                whole_homologs_X = adata[:, orthologs_name].X
                print(whole_homologs_X.shape)
                spatial_coords = np.array(adata.obsm[self.graph_construct_key])
                if self.Smooth_spatial_neighbors[species_id] >= 1:
                    whole_homologs_X = whole_homologs_X.todense()
                    whole_homologs_X = average_spatial_neighbors(whole_homologs_X, spatial_coords, self.Smooth_spatial_neighbors[species_id])
                    whole_homologs_X = csr_matrix(whole_homologs_X)
                whole_species_name = list(adata.obs['species_id'])
                whole_slice_name = list(adata.obs['slice_name']) 
                whole_batch_name = list(adata.obs['batch_name'])
                whole_species_id = list(adata.obs['species_id'])
                whole_spatial = adata.obsm[self.graph_construct_key]
                if 'annotation' in adata.obs:
                    whole_annotation = list(adata.obs['annotation']) 
            else:
                whole_obs_name = whole_obs_name + list(adata.obs_names)
                whole_homologs_X_temp = adata[:, orthologs_name].X #.todense()
                print(whole_homologs_X_temp.shape)
                spatial_coords = np.array(adata.obsm[self.graph_construct_key])
                if self.Smooth_spatial_neighbors[species_id] >= 1:
                    whole_homologs_X_temp = whole_homologs_X_temp.todense()
                    whole_homologs_X_temp = average_spatial_neighbors(whole_homologs_X_temp, spatial_coords, self.Smooth_spatial_neighbors[species_id])
                    whole_homologs_X_temp = csr_matrix(whole_homologs_X_temp)
      
                whole_homologs_X = sparse.vstack([whole_homologs_X, whole_homologs_X_temp])
                whole_species_name = whole_species_name + list(adata.obs['species_id'])
                whole_slice_name = whole_slice_name + list(adata.obs['slice_name']) 
                whole_batch_name = whole_batch_name + list(adata.obs['batch_name'])
                whole_species_id = whole_species_id + list(adata.obs['species_id'])
                whole_spatial = np.concatenate([whole_spatial, adata.obsm[self.graph_construct_key]], axis=0)
                if 'annotation' in adata.obs and 'whole_annotation' in locals():
                    whole_annotation = whole_annotation + list(adata.obs['annotation']) 
            spe += 1

        adata_whole = ad.AnnData(X = whole_homologs_X)
        adata_whole.obs_names = whole_obs_name
        adata_whole.obsm['homologs'] = whole_homologs_X
        adata_whole.obs['slice_name'] = whole_slice_name
        adata_whole.obs['batch_name'] = whole_batch_name
        adata_whole.obs['species_id'] = whole_species_id
        adata_whole.obsm[self.graph_construct_key] = whole_spatial
        if 'annotation' in adata.obs and 'whole_annotation' in locals():
            adata_whole.obs['annotation'] = whole_annotation
        # Batch correction
        
        if self.if_hvg_before_mnn:
            adata_whole.X = scipy.sparse.csr_matrix(adata_whole.X)
            sc.pp.highly_variable_genes(adata_whole, flavor="seurat_v3", n_top_genes=self.homo_n_top_genes, inplace=True)
            adata_whole = adata_whole[:, adata_whole.var['highly_variable']]
            if self.if_combat_mnn:
                sc.pp.combat(adata_whole, key='species_id', covariates=None, inplace=True)
                if self.if_pca_before_mnn:
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim_before_mnn)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X
            else:
                if self.if_pca_before_mnn:
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim_before_mnn)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X
        else:
            if self.if_combat_mnn:
                sc.pp.combat(adata_whole, key='species_id', covariates=None, inplace=True)
                if self.if_pca_before_mnn:
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim_before_mnn)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X
            else:
                if self.if_pca_before_mnn:
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim_before_mnn)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X

        
        adata_whole_dict = {k:None for k in adata_dict.keys()}
        for k in adata_whole_dict.keys():
            adata_whole_dict[k] = adata_whole[adata_whole.obs['species_id'].isin([k])]

        if not self.if_combat_mnn and sp.issparse(adata_whole.obsm['homologs']):
            adata_whole.obsm['homologs'] = adata_whole.obsm['homologs'].todense()
        mnn_dict = create_dictionary_mnn(adata_whole, use_rep='homologs', \
                                    batch_name='species_id', k=self.cross_species_neibors_K_mnn, iter_comb=None, verbose=0)
        section_ids = list(adata_dict.keys())
        anchor_ind_species = []
        positive_ind_species = []
        negative_ind_species = []
        for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
            batchname_list = adata_whole.obs['species_id'][mnn_dict[batch_pair].keys()]
            cellname_by_batch_dict = dict()
            for batch_id in range(len(section_ids)):
                cellname_by_batch_dict[section_ids[batch_id]] = adata_whole.obs_names[adata_whole.obs['species_id'] == section_ids[batch_id]].values
            anchor_list = []
            positive_list = []
            negative_list = []
            for anchor in mnn_dict[batch_pair].keys():
                anchor_list.append(anchor)
                positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                positive_list.append(positive_spot)
                section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                negative_list.append(
                    cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
            batch_as_dict = dict(zip(list(adata_whole.obs_names), range(0, adata_whole.shape[0])))
            anchor_ind_species = np.append(anchor_ind_species, list(map(lambda _: batch_as_dict[_], anchor_list)))
            positive_ind_species = np.append(positive_ind_species, list(map(lambda _: batch_as_dict[_], positive_list)))
            negative_ind_species = np.append(negative_ind_species, list(map(lambda _: batch_as_dict[_], negative_list)))
            

        if self.if_integrate_within_species:
            anchor_ind_sections = []
            positive_ind_sections = []
            negative_ind_sections = []
            for species_id in adata_dict.keys():
                k_add = species_spot_add_dict[species_id]
                section_ids = adata_dict[species_id].obs['slice_name'].unique()
                adata = adata_dict[species_id]
                sc.pp.combat(adata, key='slice_name', covariates=None, inplace=True)
                adata.X = scipy.sparse.csr_matrix(adata.X)
                mnn_dict = create_dictionary_mnn(adata, use_rep='X_pca',batch_name='slice_name', k=self.cross_sections_neibors_K_mnn, iter_comb=None, verbose=0)

                for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                    batchname_list = adata.obs['slice_name'][mnn_dict[batch_pair].keys()]
                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(section_ids)):
                        cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[adata.obs['slice_name'] == section_ids[batch_id]].values
    
                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for anchor in mnn_dict[batch_pair].keys():
                        anchor_list.append(anchor)
                        positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                        positive_list.append(positive_spot)
                        section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                        negative_list.append(
                            cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
    
                    batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                    anchor_ind_sections = np.append(anchor_ind_sections, [int(k_add + x) for x in list(map(lambda _: batch_as_dict[_], anchor_list))])
                    positive_ind_sections = np.append(positive_ind_sections, [int(k_add + x) for x in list(map(lambda _: batch_as_dict[_], positive_list))])
                    negative_ind_sections = np.append(negative_ind_sections, [int(k_add + x) for x in list(map(lambda _: batch_as_dict[_], negative_list))])


            triplet_ind_sections_dict = {'anchor_ind_sections':anchor_ind_sections,
                                   'positive_ind_sections':positive_ind_sections,
                                   'negative_ind_sections':negative_ind_sections}
            print('Number of cross-sections triplets:', len(anchor_ind_sections))
         
            
        triplet_ind_species_dict = {'anchor_ind_species':anchor_ind_species,
                                   'positive_ind_species':positive_ind_species,
                                   'negative_ind_species':negative_ind_species}

        
        # # Update edge_ndarray_species
        if self.knn_triplets:
            edge_1 = list(anchor_ind_species) + list(positive_ind_species) + list(list(edge_ndarray_species)[0])
            edge_2 = list(positive_ind_species) + list(anchor_ind_species) + list(list(edge_ndarray_species)[1])
        else:
            edge_1 = list(anchor_ind_species) + list(positive_ind_species)
            edge_2 = list(positive_ind_species) + list(anchor_ind_species)
        edge_ndarray_species = tuple([np.array(edge_1), np.array(edge_2)])


        if self.if_integrate_within_species:
            edge_1 = list(anchor_ind_sections) + list(positive_ind_sections)
            edge_2 = list(positive_ind_sections) + list(anchor_ind_sections)
            edge_ndarray_sections = tuple([np.array(edge_1), np.array(edge_2)])

        ###################################################################################
        ## Merge KNN and MNN triplets
        if self.knn_triplets:
            for k_species_order in range(len(species_id_list)-1):
                k_species = species_id_list[k_species_order]
                for v_species_other_order in range(k_species_order+1, len(species_id_list)):
                    v_species = species_id_list[v_species_other_order]
                    print(Species_cross_B_dict[k_species][v_species].toarray().shape)
            #####
            species_parameters = {'Species_cross_B_dict': Species_cross_B_dict,
                          'spot_name_species_dict':spot_name_species_dict, 
                          'spotname2id':spotname2id, 
                          'id2spotname':id2spotname, 
                          'knn_neigh_species': self.cross_species_neibors_K_knn,
                          'adata_species_section_num_dict':adata_species_section_num_dict}
    
            anchor_ind_species_all, positive_ind_species_all, negative_ind_species_all = STACAME.mnn_utils.get_species_triples(species_parameters)
            print('lenghth of anchor_ind_species:', len(anchor_ind_species_all))
            ## Running STACAME
            
            anchor_ind_all_num_dict = {k:0 for k in anchor_ind_species_all.keys()}
            
            subsampling_rate = self.knn_triplets_ratio
            
            anchor_ind_species = {}
            positive_ind_species = {}
            negative_ind_species = {}
            for s, anchor_list_s in anchor_ind_species_all.items():
                anchor_ind_all_num_dict[s] = len(anchor_list_s)
            
                anchor_ind_species[s] = []
                positive_ind_species[s] = []
                negative_ind_species[s] = []
            
            for spe_id in anchor_ind_all_num_dict.keys():
                spe_indices = random_list(anchor_ind_all_num_dict[spe_id], subsampling_rate)
            
                anchor_ind_species[spe_id] = [anchor_ind_species_all[spe_id][i] for i in spe_indices] 
                positive_ind_species[spe_id] = [positive_ind_species_all[spe_id][i] for i in spe_indices] 
                negative_ind_species[spe_id] = [negative_ind_species_all[spe_id][i] for i in spe_indices] 
            
            anchor_arr_species_ind = []
            positive_arr_species_ind = []
            negative_arr_species_ind = []
            
            k_add = 0
            species_add_dict = {k:None for k in adata_dict.keys()}
            for species_id in adata_dict.keys():
                species_add_dict[species_id] = int(k_add)
                k_add = int(k_add+adata_dict[species_id].n_obs)
            
            for species_id in adata_dict.keys():
                k_add = species_add_dict[species_id]
                anchor_arr_species_ind = anchor_arr_species_ind + \
                    [int(anch + k_add) for anch in anchor_ind_species[species_id]]
                for t in range(len(positive_ind_species[species_id])):
                    p_tuple = positive_ind_species[species_id][t]
                    positive_arr_species_ind.append(int(species_add_dict[p_tuple[0]] + p_tuple[1]))
                    n_tuple = negative_ind_species[species_id][t]
                    negative_arr_species_ind.append(int(species_add_dict[n_tuple[0]] + n_tuple[1]))
        
        if self.knn_triplets:
            triplet_ind_species_dict = {'anchor_ind_species':list(triplet_ind_species_dict['anchor_ind_species']) + 
                                    list(anchor_arr_species_ind), 
                                    'positive_ind_species':list(triplet_ind_species_dict['positive_ind_species']) + 
                                    list(positive_arr_species_ind), 
                                    'negative_ind_species':list(triplet_ind_species_dict['negative_ind_species']) + 
                                    list(negative_arr_species_ind)}
        else:
            triplet_ind_species_dict = {'anchor_ind_species':list(triplet_ind_species_dict['anchor_ind_species']),  
                                   'positive_ind_species':list(triplet_ind_species_dict['positive_ind_species']), 
                                   'negative_ind_species':list(triplet_ind_species_dict['negative_ind_species'])}
        
        print('Triplrts number = ', len(triplet_ind_species_dict['anchor_ind_species']))

        ###########################################################################################
        ## Add aligne specific number of homolgous genes and append them into the features
        print('Add aligne specific number of homolgous genes and append them into the features...')
        Gene_map_raw_df = pd.read_csv(Gene_map_raw_path, sep='\t')
        Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df['Gene name'].notna()]
        for v in self.species_ortholog_column_dict.values():
            Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df[v].notna()]
        for col in self.species_ortholog_type_dict.values():
            Gene_map_dropna_df = Gene_map_dropna_df[Gene_map_dropna_df[col].isin(['ortholog_one2one'])]
        ############
        Gene_map_common_df = Gene_map_dropna_df
        for species_id, column_name in self.species_ortholog_column_dict.items():
            common_gene_list = species_common_gene_list_dict[species_id]
            Gene_map_common_df = Gene_map_common_df[Gene_map_common_df[column_name].isin(common_gene_list)]
            print(Gene_map_common_df.shape)
            
        # Create gene map, use the first species genes as reference
        print('Create gene map, use the first species genes as reference...')
        gene_map_dict = {k:{} for k in adata_dict.keys()} 
        k_gene_map = 0
        for species_id, column_name in self.species_ortholog_column_dict.items():
            if k_gene_map == 0:
                gene_set = Gene_map_common_df[column_name].values
                gene_map_dict[species_id] = {k:v for k,v in zip(gene_set, gene_set)}
            else:
                gene_map_dict[species_id] = {k:v for k,v in zip(gene_set, Gene_map_common_df[column_name])}
            k_gene_map += 1

        
        k = 0
        adata_homo_dict = {k:{} for k in adata_dict.keys()} 
        for species_id, column_name in self.species_ortholog_column_dict.items():
            if k == 0:
                adata_homo = adata_dict[species_id][:, list(Gene_map_common_df[column_name].values)]
                adata_homo.var_names = list(Gene_map_common_df[column_name].values)
                adata_homo_dict[species_id] = adata_homo
            else:
                adata_homo = adata_dict[species_id][:, list(Gene_map_common_df[column_name].values)]
                adata_homo.var_names = list(Gene_map_common_df[column_name].values)
                adata_homo_dict[species_id] = adata_homo
            k += 1

        print('Concatenate gene matrix...')
        k = 0
        for adata in adata_homo_dict.values():
            if k == 0:
                expre_X = csr_matrix(adata.X)
                expre_spatial = adata.obsm[self.graph_construct_key]
                expre_obs_name = list(adata.obs_names)
                expre_slice_name = list(adata.obs['slice_name'])
                expre_batch_name = list(adata.obs['batch_name'])
                expre_species_id = list(adata.obs['species_id'])
                if 'annotation' in adata.obs:
                    expre_annotation = list(adata.obs['annotation'])
                expre_var_name = list(adata.var_names)
            else:
                print(f'Concatenate {k}-th gene matrix...')
                expre_X = sparse.vstack([expre_X, csr_matrix(adata.X)])
                expre_spatial = np.concatenate((expre_spatial, adata.obsm[self.graph_construct_key]), axis=0)
                expre_obs_name = expre_obs_name + list(adata.obs_names)
                expre_slice_name = expre_slice_name + list(adata.obs['slice_name'])
                expre_batch_name = expre_batch_name + list(adata.obs['batch_name'])
                expre_species_id = expre_species_id + list(adata.obs['species_id'])
                if 'annotation' in adata.obs and 'expre_annotation' in locals():
                    expre_annotation = expre_annotation + list(adata.obs['annotation'])
                
            k += 1
        adata_concat = ad.AnnData(X = sp.csr_matrix(expre_X))
        adata_concat.obsm[self.graph_construct_key] = expre_spatial
        adata_concat.obs['slice_name'] = expre_slice_name
        adata_concat.obs['batch_name'] = expre_batch_name
        adata_concat.obs['species_id'] = expre_species_id
        if 'annotation' in adata.obs and 'expre_annotation' in locals():
            adata_concat.obs['annotation'] = expre_annotation
        adata_concat.var_names = expre_var_name
        adata_concat.obs_names = expre_obs_name

        print('Select homologous HVGS...')
        sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=self.homo_n_top_genes)

        df = adata_concat.var['highly_variable']
        homo_gene_ref = df[df == True].index.tolist()
        for species_id, adata in adata_dict.items():
            adata_dict[species_id].uns['homo_highly_variable'] = [gene_map_dict[species_id][g] for g in homo_gene_ref]

        #############################################################
        ## Reorder the genes such that the homologous genes are aligned well
        print('Reorder the genes such that the homologous genes are aligned well...')
        align_gene_k = 0
        reference_hvg_vec = None
        hvg_dict = {k: [] for k in adata_dict.keys()}

        gene_cap_upper_dict = self.gene_cap_upper_dict
        upper2origin_dict = {k: {} for k in adata_dict.keys()}
        for species_id in upper2origin_dict.keys():
            upper2origin_dict[species_id] = {k: v for k, v in zip([x.upper() for x in adata_dict[species_id].var_names],
                                                                  adata_dict[species_id].var_names)}

        for species_id, adata in adata_dict.items():
            df = adata.var['highly_variable']
            hvg_order = df[df == True].index.tolist()
            hvg_dict[species_id] = [x for x in hvg_order]
            hvg_dict[species_id].sort()

        hvg_intersect_set = set()
        for species_id, adata in adata_dict.items():
            if align_gene_k == 0:
                hvg_intersect_set = set([x.upper() for x in hvg_dict[species_id]])
                align_gene_k += 1
            else:
                hvg_intersect_set = hvg_intersect_set.intersection(set(hvg_dict[species_id]))
                align_gene_k += 1

        hvg_intersect_set = list(hvg_intersect_set)
        print('Size of hvg intersect set: ', len(hvg_intersect_set))

        hvg_aligned_dict = {k: [] for k in adata_dict.keys()}

        gene_name_dict = {k: {'species_specific': [], 'homo_highly_variable': []} for k in adata_dict.keys()}

        for species_id, adata in adata_dict.items():
            orthlogs = [upper2origin_dict[species_id][x] for x in hvg_intersect_set]
            hvg_aligned_dict[species_id] = adata_dict[species_id].uns['homo_highly_variable'] + list(
                set(hvg_dict[species_id]) - set(orthlogs))
            adata_dict[species_id].uns['highly_variable'] = hvg_aligned_dict[species_id]
            adata_dict[species_id].uns['species_specific'] = list(set(hvg_dict[species_id]) - set(orthlogs))

            hvg_genes = adata_dict[species_id].uns['highly_variable']
            hvg_genes_share = adata_dict[species_id].uns['homo_highly_variable']
            hvg_genes_specific = adata_dict[species_id].uns['species_specific']

            # adata_dict[species_id].obsm['counts_hvg'] = adata_dict[species_id][:, hvg_genes].layers['counts'].copy()
            # adata_dict[species_id].obsm['counts_hvg_share'] = adata_dict[species_id][:, hvg_genes_share].layers[
            #     'counts'].copy()
            # adata_dict[species_id].obsm['counts_hvg_specific'] = adata_dict[species_id][:, hvg_genes_specific].layers[
            #     'counts'].copy()

            gene_name_dict[species_id]['species_specific'] = list(set(hvg_dict[species_id]) - set(orthlogs))
            gene_name_dict[species_id]['homo_highly_variable'] = adata_dict[species_id].uns['homo_highly_variable']

        if self.gene_save_path != None:
            if not os.path.exists(self.gene_save_path):
                os.makedirs(self.gene_save_path)
            np.save(self.gene_save_path + 'gene_name_dict.npy', gene_name_dict)
        #################################################################################
        print('Processing data finished.')
        stop = timeit.default_timer()
        print('Time used: ', stop - start)

        if self.if_return_concat_adata:
            if self.if_integrate_within_species:
                return adata_dict, triplet_ind_species_dict, edge_ndarray_species, triplet_ind_sections_dict, edge_ndarray_sections, adata_whole
            else:
                return adata_dict, triplet_ind_species_dict, edge_ndarray_species, adata_whole
        if self.if_integrate_within_species:
            return adata_dict, triplet_ind_species_dict, edge_ndarray_species, triplet_ind_sections_dict, edge_ndarray_sections
        else:
            return adata_dict, triplet_ind_species_dict, edge_ndarray_species

        # print('Select homologous HVGS...')
        # sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=self.homo_n_top_genes)
        #
        # df = adata_concat.var['highly_variable']
        # homo_gene_ref = df[df == True].index.tolist()
        # for species_id, adata in adata_dict.items():
        #     adata_dict[species_id].uns['homo_highly_variable'] = [gene_map_dict[species_id][g] for g in homo_gene_ref]
        #
        # #############################################################
        # ## Reorder the genes such that the homologous genes are aligned well
        # print('Reorder the genes such that the homologous genes are aligned well...')
        # align_gene_k = 0
        # reference_hvg_vec = None
        # hvg_dict = {k:[] for k in adata_dict.keys()}
        #
        # gene_cap_upper_dict = self.gene_cap_upper_dict
        # upper2origin_dict = {k:{} for k in adata_dict.keys()}
        # for species_id in upper2origin_dict.keys():
        #     upper2origin_dict[species_id] = {k:v for k,v in zip([x.upper() for x in adata_dict[species_id].var_names], adata_dict[species_id].var_names)}
        #
        # for species_id, adata in adata_dict.items():
        #     df = adata.var['highly_variable']
        #     hvg_order = df[df == True].index.tolist()
        #     hvg_dict[species_id] = [x for x in hvg_order]
        #     hvg_dict[species_id].sort()
        #
        # hvg_intersect_set = set()
        # for species_id, adata in adata_dict.items():
        #     if align_gene_k == 0:
        #         hvg_intersect_set = set([x.upper() for x in hvg_dict[species_id]])
        #         align_gene_k += 1
        #     else:
        #         hvg_intersect_set = hvg_intersect_set.intersection(set(hvg_dict[species_id]))
        #         align_gene_k += 1
        #
        # hvg_intersect_set = list(hvg_intersect_set)
        # print('Size of hvg intersect set: ', len(hvg_intersect_set))
        #
        # hvg_aligned_dict = {k:[] for k in adata_dict.keys()}
        #
        # gene_name_dict = {k:{'species_specific':[], 'homo_highly_variable':[]} for k in adata_dict.keys()}
        #
        # for species_id, adata in adata_dict.items():
        #     orthlogs = [upper2origin_dict[species_id][x] for x in hvg_intersect_set]
        #     hvg_aligned_dict[species_id] = adata_dict[species_id].uns['homo_highly_variable'] + list(set(hvg_dict[species_id]) - set(orthlogs))
        #     adata_dict[species_id].uns['highly_variable'] = hvg_aligned_dict[species_id]
        #
        #     gene_name_dict[species_id]['species_specific'] = list(set(hvg_dict[species_id]) - set(orthlogs))
        #     gene_name_dict[species_id]['homo_highly_variable'] = adata_dict[species_id].uns['homo_highly_variable']
        #
        # if self.gene_save_path != None:
        #     if not os.path.exists(self.gene_save_path):
        #         os.makedirs(self.gene_save_path)
        #     np.save(self.gene_save_path + 'gene_name_dict.npy', gene_name_dict)
        # #################################################################################
        # print('Processing data finished.')
        # stop = timeit.default_timer()
        # print('Time used: ', stop - start)
        #
        # if self.if_return_concat_adata:
        #     if self.if_integrate_within_species:
        #         return adata_dict, triplet_ind_species_dict, edge_ndarray_species, triplet_ind_sections_dict, edge_ndarray_sections, adata_whole
        #     else:
        #         return adata_dict, triplet_ind_species_dict, edge_ndarray_species, adata_whole
        # if self.if_integrate_within_species:
        #     return adata_dict, triplet_ind_species_dict, edge_ndarray_species, triplet_ind_sections_dict, edge_ndarray_sections
        # else:
        #     return adata_dict, triplet_ind_species_dict, edge_ndarray_species






class STACAME_processer_subgraph():
    def __init__(self, 
                 root_data_path,
                 Gene_map_raw_path, 
                 species_section_ids, 
                 species_ortholog_column_dict, 
                 species_ortholog_type_dict, 
                 species_id_map, 
                 rad_cutoff_dict,
                 gene_cap_upper_dict = {'Mouse':'capitalize', 'Macaque':'upper'},
                 total_normalize = 1e4,
                 log_normalize_dict = None,
                 Down_sampling_adata = None, 
                 n_top_genes = 2000, 
                 homo_n_top_genes = 4000, 
                 cross_species_neibors_K_mnn = 20,
                 cross_sections_neibors_K_mnn = 30,
                 cross_species_neibors_K_knn = 1, 
                 Smooth_spatial_neighbors = 2, 
                 knn_triplets = True, 
                 knn_triplets_ratio=0.05, 
                 if_hvg_before_mnn = False, 
                 if_combat_mnn = False, 
                 if_pca_before_mnn = False, 
                 gene_save_path = './output_STACAME/', 
                 if_integrate_within_species = False, pca_dim_before_mnn = 128, graph_construct_key = 'spatial'):
    
        """ Run the main data preprocess of STACAME, for integrating multiple datasets from multiple species
        Parameters
        ----------
    
        Gene_map_raw_path
            file path of gene orthologs of multiple species, e.g., './Data/Mouse_Macaque.tsv'
        species_section_ids
            dict of sections for each species, e.g., species_section_ids = {'Mouse':['T315'], 'Macaque':['T36']}
        species_ortholog_column_dict
            dict of gene column name in the Gene_map_raw_path, e.g., species_ortholog_column_dict = {'Mouse':'Gene name', 'Macaque':'Macaque gene name'}
        species_ortholog_type_dict
            dict of gene homology type column name, e.g., species_ortholog_type_dict = {'Macaque':'Macaque homology type'}
        species_id_map
            ID of each species, e.g., species_id_map = {'Mouse':0, 'Macaque':1}):
        rad_cutoff_dict
            An important hyper-paramater, adjust the range of MNN. We advice to check the nearbor number to determine it. For example,  rad_cutoff_dict = {'Mouse':1.3, 'Macaque':1.3}.
        gene_cap_upper_dict
            Gene name characater form, e.g., gene_cap_upper_dict = {'Mouse':'capitalize', 'Macaque':'upper'}.
        log_normalize_dict
            Choose whether to apply log on the data, e.g., log_normalize_dict = {'Mouse':True, 'Macaque':False}
        Down_sampling_adata
            Downsampling rate of adata for light running, e.g., 0.1. Default: None.
        n_top_genes
            HVG number for each species (including homologous genes and species-specific genes)
        homo_n_top_genes
            Aligned one-to-one homologous genes
        cross_species_neibors_K_mnn
            MNN K parameter. This parameter needs to be well-adjusted.
        cross_species_neibors_K_knn
            KNN triplets paramater, and we advice to use 1, since KNN triplets increase rapidly as K increases. Default 1.
        Smooth_spatial_neighbors
            Neigborhood range for average when computing cross-species distance on multi-multi orthologs. Default 2.
        knn_triplets
            Whether to choose to use KNN triplets. Default True.
        knn_triplets_ratio
            The downsampling ratio of KNN triplets when using KNN triplets.
        Returns
        -------
        outputs: dict of anndata, the keys are species names 
        """
        self.root_data_path = root_data_path
        self.Gene_map_raw_path = Gene_map_raw_path
        self.species_section_ids = species_section_ids
        self.species_ortholog_column_dict = species_ortholog_column_dict
        self.species_ortholog_type_dict = species_ortholog_type_dict
        self.species_id_map = species_id_map
        self.rad_cutoff_dict = rad_cutoff_dict
        self.gene_cap_upper_dict = gene_cap_upper_dict
        self.Down_sampling_adata = Down_sampling_adata
        self.n_top_genes = n_top_genes
        self.homo_n_top_genes = homo_n_top_genes
        self.cross_species_neibors_K_mnn = cross_species_neibors_K_mnn
        self.cross_sections_neibors_K_mnn = cross_sections_neibors_K_mnn
        self.cross_species_neibors_K_knn = cross_species_neibors_K_knn
        self.Smooth_spatial_neighbors = Smooth_spatial_neighbors
        self.knn_triplets = knn_triplets
        self.knn_triplets_ratio = knn_triplets_ratio

        self.if_hvg_before_mnn = if_hvg_before_mnn

        self.if_combat_mnn = if_combat_mnn

        self.if_pca_before_mnn = if_pca_before_mnn

        self.total_normalize = total_normalize

        self.gene_save_path = gene_save_path

        self.if_integrate_within_species = if_integrate_within_species

        self.pca_dim = pca_dim_before_mnn

        self.graph_construct_key = graph_construct_key
        
        if total_normalize == None:
            self.total_normalize_dict = {}
            for species_id in species_section_ids.keys():
                self.total_normalize_dict[species_id] = None
        else:
            self.total_normalize_dict = total_normalize
        if log_normalize_dict == None:
            self.log_normalize_dict = {}
            for species_id in species_section_ids.keys():
                self.log_normalize_dict[species_id] = True
        else:
            self.log_normalize_dict = log_normalize_dict

        # Process of self.rad_cutoff_dict
        for k, v in rad_cutoff_dict.items():
            if isinstance(v, (int, float)):
                self.rad_cutoff_dict[k] = {x:y for x,y in zip(self.species_section_ids[k], [v] * len(self.species_section_ids[k]))}
            elif len(v) == len(self.species_section_ids[k]):
                pass
            else:
                raise Exception("The element value of rad_cutoff_dict should be a dict of length of 1 or number of sections.")
        print('self.rad_cutoff_dict:', self.rad_cutoff_dict)
    
    def load_process_adata(self):
        
        start = timeit.default_timer()

        ## Load gene orthologs and remove rows with na
        Gene_map_raw_path = self.Gene_map_raw_path
        Gene_map_raw_df = pd.read_csv(Gene_map_raw_path, sep='\t')
        Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df['Gene name'].notna()]
        for v in self.species_ortholog_column_dict.values():
            Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df[v].notna()]
            

        species_common_hvg_dict = {k:[] for k in self.species_section_ids.keys()}
        ## load adata and pick multi-to-multi homologous genes
        species_common_gene_list_dict = {}
        for species_id in self.species_section_ids.keys():
            section_ids = self.species_section_ids[species_id]
            print(f'--------------------------Species-{species_id}-------------------------------')
            gene_set = []
            for section_id in section_ids:
                print('Species:', species_id, 'Section:', section_id)
                adata = sc.read_h5ad(os.path.join(f'{self.root_data_path}{species_id}', section_id + '.h5ad'))
                print(adata.obsm[self.graph_construct_key].shape)
                adata.X = csr_matrix(adata.X)
                adata.var_names_make_unique(join="++")
                print('Before flitering: ', adata.shape)
                sc.pp.filter_genes(adata, min_cells=20)
                print('After flitering: ', adata.shape)
                if len(gene_set) == 0:
                    gene_set = adata.var_names
                else:
                    gene_set = list(set(gene_set).intersection(set(adata.var_names)))
                print('Number of genes:', len(adata.var_names))
            species_common_gene_list_dict[species_id] = gene_set
            # make spot name unique
            hvg_set = []
            for section_id in section_ids:
                adata = sc.read_h5ad(os.path.join(f'{self.root_data_path}{species_id}', section_id + '.h5ad'))
                adata.X = csr_matrix(adata.X)
                adata.var_names_make_unique(join="++")
                print('Before flitering: ', adata.shape)
                sc.pp.filter_genes(adata, min_cells=20)
                print('After flitering: ', adata.shape)
                adata.obs_names = [x+'_'+species_id + '_' + section_id for x in adata.obs_names]
                # Normalization
                adata = adata[:, gene_set]
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=np.min([len(gene_set), 5000]))
                adata = adata[:, adata.var['highly_variable']]
                if len(hvg_set) == 0:
                    hvg_set = adata.var_names
                else:
                    hvg_set = list(set(hvg_set).union(set(adata.var_names)))
                print('Number of hvgs:', len(adata.var_names))
            species_common_hvg_dict[species_id] = hvg_set
            print('Number of common hvgs:', len(hvg_set))


        ## Union of ortholog genes
        Gene_map_common_df = Gene_map_dropna_df
        for species_id, column_name in self.species_ortholog_column_dict.items():
            common_gene_list = species_common_gene_list_dict[species_id]
            Gene_map_common_df = Gene_map_common_df[Gene_map_common_df[column_name].isin(common_gene_list)]
        
        species_orthologs_hvg_union_dict = {}
        for species_id, column_name in self.species_ortholog_column_dict.items():
            orthologs_gene_list = set(Gene_map_common_df[column_name])
            hvg_list = set(list(species_common_hvg_dict[species_id]))
            species_orthologs_hvg_union_dict[species_id] = list(hvg_list.union(orthologs_gene_list))
    
        ## Normalizing data
        print('Normalizing data and get spatial neigbors...')
        Batch_dict = {k:[] for k in self.species_section_ids.keys()}
        A_adj_dict = {k:[] for k in self.species_section_ids.keys()}        
        for species_id, column_name in self.species_ortholog_column_dict.items():
            section_ids = self.species_section_ids[species_id]
            print(f'--------------------------Species-{species_id}-------------------------------')
            gene_set = species_orthologs_hvg_union_dict[species_id]
            hvg_set = list(species_common_hvg_dict[species_id])
            orthologs_set = Gene_map_common_df[column_name]
            for section_id in section_ids:
                adata = sc.read_h5ad(os.path.join(f'{self.root_data_path}{species_id}', section_id + '.h5ad'))
                print(f'---------Section-{section_id}---------')
                # Downsampling the datasets
                if self.Down_sampling_adata != None and self.Down_sampling_adata < 1:
                    sc.pp.subsample(adata, fraction=self.Down_sampling_adata)

                adata.X = csr_matrix(adata.X)
                adata.obs_names = [x+'_'+species_id + '_' + section_id for x in adata.obs_names]
                # Select common genes for different slices
                adata = adata[:, gene_set]
                STACAME.Cal_Spatial_Net(adata, rad_cutoff=self.rad_cutoff_dict[species_id][section_id], use_key = self.graph_construct_key)
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
                if self.total_normalize != None:
                    if self.total_normalize_dict[species_id] != None:
                        sc.pp.normalize_total(adata, target_sum=self.total_normalize_dict[species_id])
                if self.log_normalize_dict[species_id] == True:
                    sc.pp.log1p(adata)
                #adata.X = MaxAbsScaler().fit_transform(adata.X)
                #sc.pp.scale(adata, max_value=1)
                Batch_dict[species_id].append(adata)
                A_adj_dict[species_id].append(adata.uns['adj'])

        ## Aquire all the spot name and map them to integrar ID
        print('Aquire all the spot name and map them to integrar ID...')
        spot_name_all_list = []
        spot_name_species_dict = {k:[] for k in self.species_section_ids.keys()}
        for species_id, sections_adata in Batch_dict.items():
            species_spot_list = []
            for adata in sections_adata:
                spot_name_all_list = spot_name_all_list + list(adata.obs_names)
                species_spot_list = species_spot_list + list(adata.obs_names)
            spot_name_species_dict[species_id] = species_spot_list
        spotname2id = {k:v for k,v in zip(spot_name_all_list, range(len(spot_name_all_list)))}
        id2spotname = {k:v for k,v in zip(range(len(spot_name_all_list)), spot_name_all_list)}

        ##################################################################
        ## Concat the scanpy objects for multiple slices of the same species
        print('Concat the scanpy objects for multiple slices of the same species...')
        adata_dict = {k:[] for k in self.species_section_ids.keys()}
        adata_species_section_num_dict = {k:[] for k in self.species_section_ids.keys()}
        for species_id, Batch_list in Batch_dict.items():
            section_ids = self.species_section_ids[species_id]
            adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
            adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
            adata_concat.obs["species_id"] = species_id
            #print('adata_concat.shape: ', adata_concat.shape)
            adata_dict[species_id] = adata_concat
            for section_adata in Batch_list:
                adata_species_section_num_dict[species_id].append(section_adata.n_obs)
        
        for species_id, adata_concat in adata_dict.items():
            sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=self.n_top_genes)
            if self.if_integrate_within_species:
                #sc.pp.combat(adata_concat, key='slice_name', covariates=None, inplace=True)
                sc.tl.pca(adata_concat, svd_solver='arpack', n_comps=64)
                adata_concat.X = scipy.sparse.csr_matrix(adata_concat.X)
            adata_dict[species_id] = adata_concat
        
        #####################################################
        ## Concat the spatial network for multiple slices of the same species
        print('Concat the spatial network for multiple slices of the same species...')
        #adj_concat = np.asarray(adj_list[0].todense())
        k = 0
        for species_id, adj_list in A_adj_dict.items():
            section_ids = self.species_section_ids[species_id]
            # for batch_id in range(1, len(section_ids)):
            #     adj_list[batch_id] = csr_matrix(adj_list[batch_id])
            if k == 0:
                adj_concat_total = adj_list[0]
                adj_concat = adj_list[0]
                for batch_id in range(1, len(section_ids)):
                    adj_concat = scipy.sparse.block_diag((adj_concat, adj_list[batch_id]))
                    adj_concat_total = scipy.sparse.block_diag((adj_concat_total, adj_list[batch_id]))
                adata_dict[species_id].uns['edgeList'] = adj_concat.nonzero()
            else:
                adj_concat = adj_list[0]
                adj_concat_total = scipy.sparse.block_diag((adj_concat_total, adj_list[0]))
                for batch_id in range(1, len(section_ids)):
                    adj_concat = scipy.sparse.block_diag((adj_concat, adj_list[batch_id]))
                    adj_concat_total = scipy.sparse.block_diag((adj_concat_total, adj_list[batch_id]))
                adata_dict[species_id].uns['edgeList'] = adj_concat.nonzero()
            k += 1
        
        #########################################################
        ## Get KNN cross-species triplets
        # Save the cosine similarity matrix for each pair of species
        Species_cross_B_dict = {k:{} for k in adata_dict.keys()}
        N_spot = 0
        species_spot_num_dict = {}
        species_spot_add_dict = {}
        for species_id, adata in adata_dict.items():
            species_spot_add_dict[species_id] = N_spot
            N_spot += adata_dict[species_id].n_obs
            species_spot_num_dict[species_id] = adata_dict[species_id].n_obs
        print(f'N_spot = {N_spot}')
        species_id_list = list(adata_dict.keys())
        # X,Y,V for crs_matrix
        Row = []
        Col = []
        Val = []

        edge_ndarray_species = None
        if self.knn_triplets:
            print('------------Get KNN cross-species triplets--------------')
            # Here, the adjacent graph should be consistent with the cross-species triplets
            for k_species_order in range(len(species_id_list)-1):
                k_species = species_id_list[k_species_order]
                k_adata = adata_dict[k_species]
                spatial_coords_k = np.array(k_adata.obsm[self.graph_construct_key])
                k_column_name = self.species_ortholog_column_dict[k_species]
                k_orthologs = list(Gene_map_common_df[k_column_name])
                mat_k = k_adata[:, k_orthologs].X
                if self.Smooth_spatial_neighbors[k_species] >= 1:
                    mat_k = average_spatial_neighbors(np.array(mat_k.todense()), spatial_coords_k, self.Smooth_spatial_neighbors[k_species])
                else:
                    mat_k = np.array(mat_k.todense())
                spot_id_list_k = [spotname2id[s] for s in k_adata.obs_names]
                k_section_num_list = adata_species_section_num_dict[k_species]
                
                for v_species_other_order in range(k_species_order+1, len(species_id_list)):
                    v_species = species_id_list[v_species_other_order]
                    v_adata = adata_dict[v_species]
                    spatial_coords_v = np.array(v_adata.obsm[self.graph_construct_key])
                    v_column_name = self.species_ortholog_column_dict[v_species]
                    v_orthologs = list(Gene_map_common_df[v_column_name])
                    # Calculate distance and convert them to triples
                    mat_v = v_adata[:, v_orthologs].X
                    if self.Smooth_spatial_neighbors[v_species] >= 1:
                        mat_v = average_spatial_neighbors(np.array(mat_v.todense()), spatial_coords_v, self.Smooth_spatial_neighbors[v_species])
                    else:
                        mat_v = np.array(mat_v.todense())
    
                    spot_id_list_v = [spotname2id[s] for s in v_adata.obs_names]
                    v_section_num_list = adata_species_section_num_dict[v_species]
                    # The dimension is not supported to be too high
                    # How to calculate distance is a problem worthy of deep consideration, 
                    # since distance in high-dimension is not working.
                    ## matrix removing batch effect
                    mat_k_num = np.shape(mat_k)[0]
                    mat_v_num = np.shape(mat_v)[0]
                    adata_kv = ad.AnnData(np.concatenate((mat_k, mat_v), axis=0))
                    adata_kv.obs['species'] = ['species_1'] * mat_k_num + ['species_2'] * mat_v_num
                    sc.pp.combat(adata_kv, key='species', covariates=None, inplace=True)
                    
                    mat_kv_all = adata_kv.X
                    mat_k = mat_kv_all[0:mat_k_num]
                    mat_v = mat_kv_all[mat_k_num: mat_k_num + mat_v_num]
                    
                    kv_sparse_mat = cosine_similarity(mat_k, mat_v, dense_output=False)
                    # Save submatrix for each pair of species
                    Species_cross_B_dict[k_species][v_species] = sparse.csr_matrix(kv_sparse_mat)
                    Species_cross_B_dict[v_species][k_species] = sparse.csr_matrix(kv_sparse_mat).T
                    k_add = spot_id_list_k[0]
                    v_add = spot_id_list_v[0]
                    # Select the k top neigbors for each species pairs
                    ######################################################
                    k_num_sum = 0
                    for k_num in k_section_num_list:
                        v_num_sum = 0
                        for v_num in v_section_num_list:
                            zero_mat_prod = np.zeros(kv_sparse_mat.shape)
                            zero_mat_prod[:, v_num_sum:int(v_num_sum+v_num)] = 1
                            kv_sparse_mat_temp = kv_sparse_mat * zero_mat_prod
                            indices_max_kv = np.argsort(kv_sparse_mat_temp, axis=1)[:, -self.cross_species_neibors_K_knn:]
                            print('indices_max_kv.shape:', indices_max_kv.shape)
                            kv_mat_ones = np.zeros(kv_sparse_mat_temp.shape)
                            for i in range(indices_max_kv.shape[0]):
                                for j in range(indices_max_kv.shape[1]):
                                    kv_mat_ones[i][indices_max_kv[i, j]] = 1
                            edge_ndarray_s = np.nonzero(kv_mat_ones)
                            print('edge_ndarray_s', edge_ndarray_s)
                            edge_ndarray_temp = list(edge_ndarray_s)
                            edge_ndarray_temp[0] = edge_ndarray_temp[0] + k_add
                            edge_ndarray_temp[1] = edge_ndarray_temp[1] + v_add
                            if edge_ndarray_species == None:
                                edge_ndarray_species = edge_ndarray_temp
                            else:
                                edge_ndarray_species = [np.concatenate((edge_ndarray_species[0], edge_ndarray_temp[0])), \
                                        np.concatenate((edge_ndarray_species[1], edge_ndarray_temp[1]))]
                            v_num_sum = v_num_sum + v_num
                        k_num_sum = k_num_sum + k_num
                    ## Transpose matrix
                    v_num_sum = 0
                    for v_num in v_section_num_list:
                        k_num_sum = 0
                        for k_num in k_section_num_list:
                            zero_mat_prod = np.zeros(kv_sparse_mat.T.shape)
                            zero_mat_prod[:, k_num_sum:int(k_num_sum+k_num)] = 1
                            kv_sparse_mat_temp = kv_sparse_mat.T * zero_mat_prod
                            indices_max_kv = np.argsort(kv_sparse_mat_temp, axis=1)[:, -self.cross_species_neibors_K_knn:]
                            kv_mat_ones = np.zeros(kv_sparse_mat_temp.shape)
                            for i in range(indices_max_kv.shape[0]):
                                for j in range(indices_max_kv.shape[1]):
                                    kv_mat_ones[i][indices_max_kv[i, j]] = 1
                            edge_ndarray_s = np.nonzero(kv_mat_ones)
                            edge_ndarray_temp = list(edge_ndarray_s)
                            edge_ndarray_temp[0] = edge_ndarray_temp[0] + v_add
                            edge_ndarray_temp[1] = edge_ndarray_temp[1] + k_add
                            if edge_ndarray_species == None:
                                edge_ndarray_species = edge_ndarray_temp
                            else:
                                edge_ndarray_species = [np.concatenate((edge_ndarray_species[0], edge_ndarray_temp[0])), \
                                        np.concatenate((edge_ndarray_species[1], edge_ndarray_temp[1]))]
                            k_num_sum = k_num_sum + k_num
                        v_num_sum = v_num_sum + v_num
            
            
            edge_ndarray_species = tuple(edge_ndarray_species)
        else:
            print('------------Skipped KNN cross-species triplets--------------')
            pass
        ##########################################################
        ## Generate cross-species MNN triplets
        print('Generate cross-species MNN triplets...')
        spe = 0
        for species_id, adata in adata_dict.items():
            homologs_column_name = self.species_ortholog_column_dict[species_id]
            orthologs_name = list(Gene_map_common_df[homologs_column_name])
            if spe == 0:
                whole_obs_name = list(adata.obs_names)
                whole_homologs_X = adata[:, orthologs_name].X
                print(whole_homologs_X.shape)
                spatial_coords = np.array(adata.obsm[self.graph_construct_key])
                if self.Smooth_spatial_neighbors[species_id] >= 1:
                    whole_homologs_X = whole_homologs_X.todense()
                    whole_homologs_X = average_spatial_neighbors(whole_homologs_X, spatial_coords, self.Smooth_spatial_neighbors[species_id])
                    whole_homologs_X = csr_matrix(whole_homologs_X)
                whole_species_name = list(adata.obs['species_id'])
                whole_slice_name = list(adata.obs['slice_name']) 
                whole_batch_name = list(adata.obs['batch_name'])
                whole_species_id = list(adata.obs['species_id'])
                if 'annotation' in adata.obs:
                    whole_annotation = list(adata.obs['annotation']) 
            else:
                whole_obs_name = whole_obs_name + list(adata.obs_names)
                whole_homologs_X_temp = adata[:, orthologs_name].X #.todense()
                print(whole_homologs_X_temp.shape)
                spatial_coords = np.array(adata.obsm[self.graph_construct_key])
                if self.Smooth_spatial_neighbors[species_id] >= 1:
                    whole_homologs_X_temp = whole_homologs_X_temp.todense()
                    whole_homologs_X_temp = average_spatial_neighbors(whole_homologs_X_temp, spatial_coords, self.Smooth_spatial_neighbors[species_id])
                    whole_homologs_X_temp = csr_matrix(whole_homologs_X_temp)
      
                whole_homologs_X = sparse.vstack([whole_homologs_X, whole_homologs_X_temp])
                whole_species_name = whole_species_name + list(adata.obs['species_id'])
                whole_slice_name = whole_slice_name + list(adata.obs['slice_name']) 
                whole_batch_name = whole_batch_name + list(adata.obs['batch_name'])
                whole_species_id = whole_species_id + list(adata.obs['species_id'])
                if 'annotation' in adata.obs:
                    whole_annotation = whole_annotation + list(adata.obs['annotation']) 
            spe += 1

        adata_whole = ad.AnnData(X = whole_homologs_X)
        adata_whole.obs_names = whole_obs_name
        adata_whole.obsm['homologs'] = whole_homologs_X
        adata_whole.obs['slice_name'] = whole_slice_name
        adata_whole.obs['batch_name'] = whole_batch_name
        adata_whole.obs['species_id'] = whole_species_id
        if 'annotation' in adata.obs:
            adata_whole.obs['annotation'] = whole_annotation
        # Batch correction
        
        if self.if_hvg_before_mnn:
            print('Run selecting hvgs before mnn...')
            adata_whole.X = scipy.sparse.csr_matrix(adata_whole.X)
            sc.pp.highly_variable_genes(adata_whole, flavor="seurat_v3", n_top_genes=self.homo_n_top_genes, batch_key='species_id', inplace=True)
            adata_whole = adata_whole[:, adata_whole.var['highly_variable']]
            if self.if_combat_mnn:
                sc.pp.combat(adata_whole, key='species_id', covariates=None, inplace=True)
                if self.if_pca_before_mnn:
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X
            else:
                if self.if_pca_before_mnn:
                    print('pca...')
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X
        else:
            if self.if_combat_mnn:
                print('Run combat...')
                sc.pp.combat(adata_whole, key='species_id', covariates=None, inplace=True)
                if self.if_pca_before_mnn:
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X
            else:
                if self.if_pca_before_mnn:
                    #adata_whole.X = np.asarray(adata_whole.X)
                    print('pca...')
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X

        adata_whole_dict = {k:None for k in adata_dict.keys()}
        for k in adata_whole_dict.keys():
            adata_whole_dict[k] = adata_whole[adata_whole.obs['species_id'].isin([k])]

        if not self.if_combat_mnn and sp.issparse(adata_whole.obsm['homologs']):
            adata_whole.obsm['homologs'] = adata_whole.obsm['homologs'].todense()
        
        print('create_dictionary_mnn...')
        mnn_dict = create_dictionary_mnn(adata_whole, use_rep='homologs', \
                                    batch_name='species_id', k=self.cross_species_neibors_K_mnn, iter_comb=None, verbose=0)
        section_ids = list(adata_dict.keys())
        anchor_ind_species = []
        positive_ind_species = []
        negative_ind_species = []
        for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
            batchname_list = adata_whole.obs['species_id'][mnn_dict[batch_pair].keys()]
            cellname_by_batch_dict = dict()
            for batch_id in range(len(section_ids)):
                cellname_by_batch_dict[section_ids[batch_id]] = adata_whole.obs_names[adata_whole.obs['species_id'] == section_ids[batch_id]].values
            anchor_list = []
            positive_list = []
            negative_list = []
            for anchor in mnn_dict[batch_pair].keys():
                anchor_list.append(anchor)
                positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                positive_list.append(positive_spot)
                section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                negative_list.append(
                    cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
            batch_as_dict = dict(zip(list(adata_whole.obs_names), range(0, adata_whole.shape[0])))
            anchor_ind_species = np.append(anchor_ind_species, list(map(lambda _: batch_as_dict[_], anchor_list)))
            positive_ind_species = np.append(positive_ind_species, list(map(lambda _: batch_as_dict[_], positive_list)))
            negative_ind_species = np.append(negative_ind_species, list(map(lambda _: batch_as_dict[_], negative_list)))

        
        print('Finished finding cross species triplets.')

       
        if self.if_integrate_within_species:
            print('Beginning to find cross section triplets...')
            anchor_ind_sections = []
            positive_ind_sections = []
            negative_ind_sections = []
            for species_id in adata_dict.keys():
                k_add = species_spot_add_dict[species_id]
                section_ids = adata_dict[species_id].obs['slice_name'].unique()
                print('slice_name:', section_ids)
                adata = adata_dict[species_id]
                #sc.pp.combat(adata, key='slice_name', covariates=None, inplace=True)
                adata.X = scipy.sparse.csr_matrix(adata.X)
                mnn_dict = create_dictionary_mnn(adata, use_rep='X_pca',batch_name='slice_name', k=self.cross_sections_neibors_K_mnn, iter_comb=None, verbose=0)

                for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                    batchname_list = adata.obs['slice_name'][mnn_dict[batch_pair].keys()]
                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(section_ids)):
                        cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[adata.obs['slice_name'] == section_ids[batch_id]].values
    
                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for anchor in mnn_dict[batch_pair].keys():
                        anchor_list.append(anchor)
                        positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                        positive_list.append(positive_spot)
                        section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                        negative_list.append(
                            cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
    
                    batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                    anchor_ind_sections = np.append(anchor_ind_sections, [int(k_add + x) for x in list(map(lambda _: batch_as_dict[_], anchor_list))])
                    positive_ind_sections = np.append(positive_ind_sections, [int(k_add + x) for x in list(map(lambda _: batch_as_dict[_], positive_list))])
                    negative_ind_sections = np.append(negative_ind_sections, [int(k_add + x) for x in list(map(lambda _: batch_as_dict[_], negative_list))])


            triplet_ind_sections_dict = {'anchor_ind_sections':anchor_ind_sections,
                                   'positive_ind_sections':positive_ind_sections,
                                   'negative_ind_sections':negative_ind_sections}
            print('Number of cross-sections triplets:', len(anchor_ind_sections))
         
            
        triplet_ind_species_dict = {'anchor_ind_species':anchor_ind_species,
                                   'positive_ind_species':positive_ind_species,
                                   'negative_ind_species':negative_ind_species}

        
        # # Update edge_ndarray_species
        if self.knn_triplets:
            edge_1 = list(anchor_ind_species) + list(positive_ind_species) + list(list(edge_ndarray_species)[0])
            edge_2 = list(positive_ind_species) + list(anchor_ind_species) + list(list(edge_ndarray_species)[1])
        else:
            edge_1 = list(anchor_ind_species) + list(positive_ind_species)
            edge_2 = list(positive_ind_species) + list(anchor_ind_species)
        edge_ndarray_species = tuple([np.array(edge_1), np.array(edge_2)])


        if self.if_integrate_within_species:
            edge_1 = list(anchor_ind_sections) + list(positive_ind_sections)
            edge_2 = list(positive_ind_sections) + list(anchor_ind_sections)
            edge_ndarray_sections = tuple([np.array(edge_1), np.array(edge_2)])

        ###################################################################################
        ## Merge KNN and MNN triplets
        if self.knn_triplets:
            for k_species_order in range(len(species_id_list)-1):
                k_species = species_id_list[k_species_order]
                for v_species_other_order in range(k_species_order+1, len(species_id_list)):
                    v_species = species_id_list[v_species_other_order]
                    print(Species_cross_B_dict[k_species][v_species].toarray().shape)
            #####
            species_parameters = {'Species_cross_B_dict': Species_cross_B_dict,
                          'spot_name_species_dict':spot_name_species_dict, 
                          'spotname2id':spotname2id, 
                          'id2spotname':id2spotname, 
                          'knn_neigh_species': self.cross_species_neibors_K_knn,
                          'adata_species_section_num_dict':adata_species_section_num_dict}
    
            anchor_ind_species_all, positive_ind_species_all, negative_ind_species_all = STACAME.mnn_utils.get_species_triples(species_parameters)
            print('lenghth of anchor_ind_species:', len(anchor_ind_species_all))
            ## Running STACAME
            
            anchor_ind_all_num_dict = {k:0 for k in anchor_ind_species_all.keys()}
            
            subsampling_rate = self.knn_triplets_ratio
            
            anchor_ind_species = {}
            positive_ind_species = {}
            negative_ind_species = {}
            for s, anchor_list_s in anchor_ind_species_all.items():
                anchor_ind_all_num_dict[s] = len(anchor_list_s)
            
                anchor_ind_species[s] = []
                positive_ind_species[s] = []
                negative_ind_species[s] = []
            
            for spe_id in anchor_ind_all_num_dict.keys():
                spe_indices = random_list(anchor_ind_all_num_dict[spe_id], subsampling_rate)
            
                anchor_ind_species[spe_id] = [anchor_ind_species_all[spe_id][i] for i in spe_indices] 
                positive_ind_species[spe_id] = [positive_ind_species_all[spe_id][i] for i in spe_indices] 
                negative_ind_species[spe_id] = [negative_ind_species_all[spe_id][i] for i in spe_indices] 
            
            anchor_arr_species_ind = []
            positive_arr_species_ind = []
            negative_arr_species_ind = []
            
            k_add = 0
            species_add_dict = {k:None for k in adata_dict.keys()}
            for species_id in adata_dict.keys():
                species_add_dict[species_id] = int(k_add)
                k_add = int(k_add+adata_dict[species_id].n_obs)
            
            for species_id in adata_dict.keys():
                k_add = species_add_dict[species_id]
                anchor_arr_species_ind = anchor_arr_species_ind + \
                    [int(anch + k_add) for anch in anchor_ind_species[species_id]]
                for t in range(len(positive_ind_species[species_id])):
                    p_tuple = positive_ind_species[species_id][t]
                    positive_arr_species_ind.append(int(species_add_dict[p_tuple[0]] + p_tuple[1]))
                    n_tuple = negative_ind_species[species_id][t]
                    negative_arr_species_ind.append(int(species_add_dict[n_tuple[0]] + n_tuple[1]))
        
        if self.knn_triplets:
            triplet_ind_species_dict = {'anchor_ind_species':list(triplet_ind_species_dict['anchor_ind_species']) + 
                                    list(anchor_arr_species_ind), 
                                    'positive_ind_species':list(triplet_ind_species_dict['positive_ind_species']) + 
                                    list(positive_arr_species_ind), 
                                    'negative_ind_species':list(triplet_ind_species_dict['negative_ind_species']) + 
                                    list(negative_arr_species_ind)}
        else:
            triplet_ind_species_dict = {'anchor_ind_species':list(triplet_ind_species_dict['anchor_ind_species']),  
                                   'positive_ind_species':list(triplet_ind_species_dict['positive_ind_species']), 
                                   'negative_ind_species':list(triplet_ind_species_dict['negative_ind_species'])}
        
        print('Triplrts number = ', len(triplet_ind_species_dict['anchor_ind_species']))

        ###########################################################################################
        ## Add aligne specific number of homolgous genes and append them into the features
        print('Add aligne specific number of homolgous genes and append them into the features...')
        Gene_map_raw_df = pd.read_csv(Gene_map_raw_path, sep='\t')
        Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df['Gene name'].notna()]
        for v in self.species_ortholog_column_dict.values():
            Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df[v].notna()]
        for col in self.species_ortholog_type_dict.values():
            Gene_map_dropna_df = Gene_map_dropna_df[Gene_map_dropna_df[col].isin(['ortholog_one2one'])]
        ############
        Gene_map_common_df = Gene_map_dropna_df
        for species_id, column_name in self.species_ortholog_column_dict.items():
            common_gene_list = species_common_gene_list_dict[species_id]
            Gene_map_common_df = Gene_map_common_df[Gene_map_common_df[column_name].isin(common_gene_list)]
            print(Gene_map_common_df.shape)
            
        # Create gene map, use the first species genes as reference
        print('Create gene map, use the first species genes as reference...')
        gene_map_dict = {k:{} for k in adata_dict.keys()} 
        k_gene_map = 0
        for species_id, column_name in self.species_ortholog_column_dict.items():
            if k_gene_map == 0:
                gene_set = Gene_map_common_df[column_name].values
                gene_map_dict[species_id] = {k:v for k,v in zip(gene_set, gene_set)}
            else:
                gene_map_dict[species_id] = {k:v for k,v in zip(gene_set, Gene_map_common_df[column_name])}
            k_gene_map += 1

        
        k = 0
        adata_homo_dict = {k:{} for k in adata_dict.keys()} 
        for species_id, column_name in self.species_ortholog_column_dict.items():
            if k == 0:
                adata_homo = adata_dict[species_id][:, list(Gene_map_common_df[column_name].values)]
                adata_homo.var_names = list(Gene_map_common_df[column_name].values)
                adata_homo_dict[species_id] = adata_homo
            else:
                adata_homo = adata_dict[species_id][:, list(Gene_map_common_df[column_name].values)]
                adata_homo.var_names = list(Gene_map_common_df[column_name].values)
                adata_homo_dict[species_id] = adata_homo
            k += 1

        print('Concatenate gene matrix...')
        k = 0
        for adata in adata_homo_dict.values():
            if k == 0:
                expre_X = csr_matrix(adata.X)
                expre_spatial = adata.obsm[self.graph_construct_key]
                expre_obs_name = list(adata.obs_names)
                expre_slice_name = list(adata.obs['slice_name'])
                expre_batch_name = list(adata.obs['batch_name'])
                expre_species_id = list(adata.obs['species_id'])
                if 'annotation' in adata.obs:
                    expre_annotation = list(adata.obs['annotation'])
                expre_var_name = list(adata.var_names)
            else:
                print(f'Concatenate {k}-th gene matrix...')
                expre_X = sparse.vstack([expre_X, csr_matrix(adata.X)])
                expre_spatial = np.concatenate((expre_spatial, adata.obsm[self.graph_construct_key]), axis=0)
                expre_obs_name = expre_obs_name + list(adata.obs_names)
                expre_slice_name = expre_slice_name + list(adata.obs['slice_name'])
                expre_batch_name = expre_batch_name + list(adata.obs['batch_name'])
                expre_species_id = expre_species_id + list(adata.obs['species_id'])
                if 'annotation' in adata.obs:
                    expre_annotation = expre_annotation + list(adata.obs['annotation'])
                
            k += 1
        adata_concat = ad.AnnData(X = sp.csr_matrix(expre_X))
        adata_concat.obsm[self.graph_construct_key] = expre_spatial
        adata_concat.obs['slice_name'] = expre_slice_name
        adata_concat.obs['batch_name'] = expre_batch_name
        adata_concat.obs['species_id'] = expre_species_id
        if 'annotation' in adata.obs:
            adata_concat.obs['annotation'] = expre_annotation
        adata_concat.var_names = expre_var_name
        adata_concat.obs_names = expre_obs_name

        print('Select homologous HVGS...')
        sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=self.homo_n_top_genes)
        
        df = adata_concat.var['highly_variable']
        homo_gene_ref = df[df == True].index.tolist()
        for species_id, adata in adata_dict.items():
            adata_dict[species_id].uns['homo_highly_variable'] = [gene_map_dict[species_id][g] for g in homo_gene_ref]
            #print(adata_dict[species_id].uns['homo_highly_variable'][:100])
        
        #############################################################
        ## Reorder the genes such that the homologous genes are aligned well
        print('Reorder the genes such that the homologous genes are aligned well...')
        align_gene_k = 0
        reference_hvg_vec = None
        hvg_dict = {k:[] for k in adata_dict.keys()}
        
        gene_cap_upper_dict = self.gene_cap_upper_dict
        upper2origin_dict = {k:{} for k in adata_dict.keys()}
        for species_id in upper2origin_dict.keys():
            upper2origin_dict[species_id] = {k:v for k,v in zip([x.upper() for x in adata_dict[species_id].var_names], adata_dict[species_id].var_names)}
        
        for species_id, adata in adata_dict.items():
            df = adata.var['highly_variable']
            hvg_order = df[df == True].index.tolist()
            hvg_dict[species_id] = [x for x in hvg_order]
            hvg_dict[species_id].sort()
            #print(hvg_dict[species_id][1:100])
        
        hvg_intersect_set = set()
        for species_id, adata in adata_dict.items():
            #print(len(hvg_dict[species_id]))
            if align_gene_k == 0:
                hvg_intersect_set = set([x.upper() for x in hvg_dict[species_id]])
                align_gene_k += 1
            else:
                hvg_intersect_set = hvg_intersect_set.intersection(set(hvg_dict[species_id]))
                align_gene_k += 1
        
        hvg_intersect_set = list(hvg_intersect_set)
        print('Size of hvg intersect set: ', len(hvg_intersect_set))
        
        hvg_aligned_dict = {k:[] for k in adata_dict.keys()}

        gene_name_dict = {k:{'species_specific':[], 'homo_highly_variable':[]} for k in adata_dict.keys()}
        
        for species_id, adata in adata_dict.items():
            orthlogs = [upper2origin_dict[species_id][x] for x in hvg_intersect_set]
            hvg_aligned_dict[species_id] = adata_dict[species_id].uns['homo_highly_variable'] + list(set(hvg_dict[species_id]) - set(orthlogs))
            adata_dict[species_id].uns['highly_variable'] = hvg_aligned_dict[species_id]

            gene_name_dict[species_id]['species_specific'] = list(set(hvg_dict[species_id]) - set(orthlogs))
            gene_name_dict[species_id]['homo_highly_variable'] = adata_dict[species_id].uns['homo_highly_variable']

        if self.gene_save_path != None:
            if not os.path.exists(self.gene_save_path):
                os.makedirs(self.gene_save_path)
            np.save(self.gene_save_path + 'gene_name_dict.npy', gene_name_dict)
        #################################################################################
        print('Processing data finished.')
        stop = timeit.default_timer()
        print('Time used: ', stop - start)  

        if self.if_integrate_within_species:
            return adata_dict, triplet_ind_species_dict, edge_ndarray_species, triplet_ind_sections_dict, edge_ndarray_sections
        else:
            return adata_dict, triplet_ind_species_dict, edge_ndarray_species



    def load_process_adata_3d(self):
        start = timeit.default_timer()

        ## Load gene orthologs and remove rows with na
        Gene_map_raw_path = self.Gene_map_raw_path
        Gene_map_raw_df = pd.read_csv(Gene_map_raw_path, sep='\t')
        Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df['Gene name'].notna()]
        for v in self.species_ortholog_column_dict.values():
            Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df[v].notna()]
            

        species_common_hvg_dict = {k:[] for k in self.species_section_ids.keys()}
        ## load adata and pick multi-to-multi homologous genes
        species_common_gene_list_dict = {}
        for species_id in self.species_section_ids.keys():
            section_ids = self.species_section_ids[species_id]
            print(f'--------------------------Species-{species_id}-------------------------------')
            gene_set = []
            for section_id in section_ids:
                print('Species:', species_id, 'Section:', section_id)
                adata = sc.read_h5ad(os.path.join(f'{self.root_data_path}{species_id}', section_id + '.h5ad'))
                print(adata.obsm[self.graph_construct_key].shape)
                adata.X = csr_matrix(adata.X)
                adata.var_names_make_unique(join="++")
                print('Before flitering: ', adata.shape)
                sc.pp.filter_genes(adata, min_cells=20)
                print('After flitering: ', adata.shape)
                if len(gene_set) == 0:
                    gene_set = adata.var_names
                else:
                    gene_set = list(set(gene_set).intersection(set(adata.var_names)))
                print('Number of genes:', len(adata.var_names))
            species_common_gene_list_dict[species_id] = gene_set
            # make spot name unique
            hvg_set = []
            for section_id in section_ids:
                adata = sc.read_h5ad(os.path.join(f'{self.root_data_path}{species_id}', section_id + '.h5ad'))
                adata.X = csr_matrix(adata.X)
                adata.var_names_make_unique(join="++")
                print('Before flitering: ', adata.shape)
                sc.pp.filter_genes(adata, min_cells=20)
                print('After flitering: ', adata.shape)
                adata.obs_names = [x+'_'+species_id + '_' + section_id for x in adata.obs_names]
                # Normalization
                adata = adata[:, gene_set]
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=np.min([len(gene_set), 5000]))
                adata = adata[:, adata.var['highly_variable']]
                if len(hvg_set) == 0:
                    hvg_set = adata.var_names
                else:
                    hvg_set = list(set(hvg_set).union(set(adata.var_names)))
                print('Number of hvgs:', len(adata.var_names))
            species_common_hvg_dict[species_id] = hvg_set
            print('Number of common hvgs:', len(hvg_set))


        ## Union of ortholog genes
        Gene_map_common_df = Gene_map_dropna_df
        for species_id, column_name in self.species_ortholog_column_dict.items():
            common_gene_list = species_common_gene_list_dict[species_id]
            Gene_map_common_df = Gene_map_common_df[Gene_map_common_df[column_name].isin(common_gene_list)]
            #print(Gene_map_common_df.shape)
        
        species_orthologs_hvg_union_dict = {}
        for species_id, column_name in self.species_ortholog_column_dict.items():
            orthologs_gene_list = set(Gene_map_common_df[column_name])
            hvg_list = set(list(species_common_hvg_dict[species_id]))
            species_orthologs_hvg_union_dict[species_id] = list(hvg_list.union(orthologs_gene_list))
            #print(f'{species_id}gene number = {len(species_orthologs_hvg_union_dict[species_id])}')
    

        ## Normalizing data
        print('Normalizing data and get spatial neigbors...')
        Batch_dict = {k:[] for k in self.species_section_ids.keys()}
        A_adj_dict = {k:[] for k in self.species_section_ids.keys()}        
        for species_id, column_name in self.species_ortholog_column_dict.items():
            section_ids = self.species_section_ids[species_id]
            print(f'--------------------------Species-{species_id}-------------------------------')
            gene_set = species_orthologs_hvg_union_dict[species_id]
            hvg_set = list(species_common_hvg_dict[species_id])
            orthologs_set = Gene_map_common_df[column_name]
            for section_id in section_ids:
                adata = sc.read_h5ad(os.path.join(f'{self.root_data_path}{species_id}', section_id + '.h5ad'))
                print(f'---------Section-{section_id}---------')
                # Downsampling the datasets
                if self.Down_sampling_adata != None and self.Down_sampling_adata < 1:
                    sc.pp.subsample(adata, fraction=self.Down_sampling_adata)
                adata.var_names_make_unique(join="++")
                # print('Before flitering: ', adata.shape)
                #sc.pp.filter_genes(adata, min_cells=50)
                # print('After flitering: ', adata.shape)
                adata.X = csr_matrix(adata.X)
                adata.obs_names = [x+'_'+species_id + '_' + section_id for x in adata.obs_names]
                # Select common genes for different slices
                adata = adata[:, gene_set]
                STACAME.Cal_Spatial_Net(adata, rad_cutoff=self.rad_cutoff_dict[species_id][section_id], use_key = self.graph_construct_key)
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=self.n_top_genes)
                if self.total_normalize != None:
                    sc.pp.normalize_total(adata, target_sum=self.total_normalize_dict[species_id])
                if self.log_normalize_dict[species_id] == True:
                    sc.pp.log1p(adata)
                #adata.X = MaxAbsScaler().fit_transform(adata.X)
                #sc.pp.scale(adata, max_value=1)
                Batch_dict[species_id].append(adata)
                A_adj_dict[species_id].append(adata.uns['adj'])

        ## Aquire all the spot name and map them to integrar ID
        print('Aquire all the spot name and map them to integrar ID...')
        spot_name_all_list = []
        spot_name_species_dict = {k:[] for k in self.species_section_ids.keys()}
        for species_id, sections_adata in Batch_dict.items():
            species_spot_list = []
            for adata in sections_adata:
                spot_name_all_list = spot_name_all_list + list(adata.obs_names)
                species_spot_list = species_spot_list + list(adata.obs_names)
            spot_name_species_dict[species_id] = species_spot_list
        spotname2id = {k:v for k,v in zip(spot_name_all_list, range(len(spot_name_all_list)))}
        id2spotname = {k:v for k,v in zip(range(len(spot_name_all_list)), spot_name_all_list)}

        ##################################################################
        ## Concat the scanpy objects for multiple slices of the same species
        print('Concat the scanpy objects for multiple slices of the same species...')
        adata_dict = {k:[] for k in self.species_section_ids.keys()}
        adata_species_section_num_dict = {k:[] for k in self.species_section_ids.keys()}
        for species_id, Batch_list in Batch_dict.items():
            section_ids = self.species_section_ids[species_id]
            adata_concat = Batch_list[0]#ad.concat(Batch_list, label="slice_name", keys=section_ids)
            adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
            adata_concat.obs["species_id"] = species_id
            #print('adata_concat.shape: ', adata_concat.shape)
            adata_dict[species_id] = adata_concat
            for section_adata in Batch_list:
                adata_species_section_num_dict[species_id].append(section_adata.n_obs)
        
        for species_id, adata_concat in adata_dict.items():
            sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=self.n_top_genes)
            if self.if_integrate_within_species:
                #sc.pp.combat(adata_concat, key='slice_name', covariates=None, inplace=True)
                sc.tl.pca(adata_concat, svd_solver='arpack', n_comps=64)
                adata_concat.X = scipy.sparse.csr_matrix(adata_concat.X)
            adata_dict[species_id] = adata_concat
        
        #####################################################
        ## Concat the spatial network for multiple slices of the same species
        print('Concat the spatial network for multiple slices of the same species...')
        #adj_concat = np.asarray(adj_list[0].todense())
        k = 0
        for species_id, adj_list in A_adj_dict.items():
            section_ids = self.species_section_ids[species_id]
            # for batch_id in range(1, len(section_ids)):
            #     adj_list[batch_id] = csr_matrix(adj_list[batch_id])
            if k == 0:
                adj_concat_total = adj_list[0]
                adj_concat = adj_list[0]
                for batch_id in range(1, len(section_ids)):
                    adj_concat = scipy.sparse.block_diag((adj_concat, adj_list[batch_id]))
                    adj_concat_total = scipy.sparse.block_diag((adj_concat_total, adj_list[batch_id]))
                adata_dict[species_id].uns['edgeList'] = adj_concat.nonzero()
            else:
                adj_concat = adj_list[0]
                adj_concat_total = scipy.sparse.block_diag((adj_concat_total, adj_list[0]))
                for batch_id in range(1, len(section_ids)):
                    adj_concat = scipy.sparse.block_diag((adj_concat, adj_list[batch_id]))
                    adj_concat_total = scipy.sparse.block_diag((adj_concat_total, adj_list[batch_id]))
                adata_dict[species_id].uns['edgeList'] = adj_concat.nonzero()
            k += 1
        
        #########################################################
        ## Get KNN cross-species triplets
        # Save the cosine similarity matrix for each pair of species
        Species_cross_B_dict = {k:{} for k in adata_dict.keys()}
        N_spot = 0
        species_spot_num_dict = {}
        species_spot_add_dict = {}
        for species_id, adata in adata_dict.items():
            species_spot_add_dict[species_id] = N_spot
            N_spot += adata_dict[species_id].n_obs
            species_spot_num_dict[species_id] = adata_dict[species_id].n_obs
        print(f'N_spot = {N_spot}')
        species_id_list = list(adata_dict.keys())
        # X,Y,V for crs_matrix
        Row = []
        Col = []
        Val = []

        edge_ndarray_species = None
        if self.knn_triplets:
            print('------------Get KNN cross-species triplets--------------')
            # Here, the adjacent graph should be consistent with the cross-species triplets
            for k_species_order in range(len(species_id_list)-1):
                k_species = species_id_list[k_species_order]
                k_adata = adata_dict[k_species]
                spatial_coords_k = np.array(k_adata.obsm[self.graph_construct_key])
                k_column_name = self.species_ortholog_column_dict[k_species]
                k_orthologs = list(Gene_map_common_df[k_column_name])
                mat_k = k_adata[:, k_orthologs].X
                if self.Smooth_spatial_neighbors[k_species] >= 1:
                    mat_k = average_spatial_neighbors(np.array(mat_k.todense()), spatial_coords_k, self.Smooth_spatial_neighbors[k_species])
                else:
                    mat_k = np.array(mat_k.todense())
                spot_id_list_k = [spotname2id[s] for s in k_adata.obs_names]
                k_section_num_list = adata_species_section_num_dict[k_species]
                
                for v_species_other_order in range(k_species_order+1, len(species_id_list)):
                    v_species = species_id_list[v_species_other_order]
                    v_adata = adata_dict[v_species]
                    spatial_coords_v = np.array(v_adata.obsm[self.graph_construct_key])
                    v_column_name = self.species_ortholog_column_dict[v_species]
                    v_orthologs = list(Gene_map_common_df[v_column_name])
                    # Calculate distance and convert them to triples
                    mat_v = v_adata[:, v_orthologs].X
                    if self.Smooth_spatial_neighbors[v_species] >= 1:
                        mat_v = average_spatial_neighbors(np.array(mat_v.todense()), spatial_coords_v, self.Smooth_spatial_neighbors[v_species])
                    else:
                        mat_v = np.array(mat_v.todense())
    
                    spot_id_list_v = [spotname2id[s] for s in v_adata.obs_names]
                    v_section_num_list = adata_species_section_num_dict[v_species]
                    # The dimension is not supported to be too high
                    # How to calculate distance is a problem worthy of deep consideration, 
                    # since distance in high-dimension is not working.
                    ## matrix removing batch effect
                    mat_k_num = np.shape(mat_k)[0]
                    mat_v_num = np.shape(mat_v)[0]
                    adata_kv = ad.AnnData(np.concatenate((mat_k, mat_v), axis=0))
                    adata_kv.obs['species'] = ['species_1'] * mat_k_num + ['species_2'] * mat_v_num
                    sc.pp.combat(adata_kv, key='species', covariates=None, inplace=True)
                    
                    mat_kv_all = adata_kv.X
                    mat_k = mat_kv_all[0:mat_k_num]
                    mat_v = mat_kv_all[mat_k_num: mat_k_num + mat_v_num]
                    
                    kv_sparse_mat = cosine_similarity(mat_k, mat_v, dense_output=False)
                    # Save submatrix for each pair of species
                    Species_cross_B_dict[k_species][v_species] = sparse.csr_matrix(kv_sparse_mat)
                    Species_cross_B_dict[v_species][k_species] = sparse.csr_matrix(kv_sparse_mat).T
                    k_add = spot_id_list_k[0]
                    v_add = spot_id_list_v[0]
                    # Select the k top neigbors for each species pairs
                    ######################################################
                    k_num_sum = 0
                    for k_num in k_section_num_list:
                        v_num_sum = 0
                        for v_num in v_section_num_list:
                            zero_mat_prod = np.zeros(kv_sparse_mat.shape)
                            zero_mat_prod[:, v_num_sum:int(v_num_sum+v_num)] = 1
                            kv_sparse_mat_temp = kv_sparse_mat * zero_mat_prod
                            indices_max_kv = np.argsort(kv_sparse_mat_temp, axis=1)[:, -self.cross_species_neibors_K_knn:]
                            print('indices_max_kv.shape:', indices_max_kv.shape)
                            kv_mat_ones = np.zeros(kv_sparse_mat_temp.shape)
                            for i in range(indices_max_kv.shape[0]):
                                for j in range(indices_max_kv.shape[1]):
                                    kv_mat_ones[i][indices_max_kv[i, j]] = 1
                            edge_ndarray_s = np.nonzero(kv_mat_ones)
                            print('edge_ndarray_s', edge_ndarray_s)
                            edge_ndarray_temp = list(edge_ndarray_s)
                            edge_ndarray_temp[0] = edge_ndarray_temp[0] + k_add
                            edge_ndarray_temp[1] = edge_ndarray_temp[1] + v_add
                            if edge_ndarray_species == None:
                                edge_ndarray_species = edge_ndarray_temp
                            else:
                                edge_ndarray_species = [np.concatenate((edge_ndarray_species[0], edge_ndarray_temp[0])), \
                                        np.concatenate((edge_ndarray_species[1], edge_ndarray_temp[1]))]
                            v_num_sum = v_num_sum + v_num
                        k_num_sum = k_num_sum + k_num
                    ## Transpose matrix
                    v_num_sum = 0
                    for v_num in v_section_num_list:
                        k_num_sum = 0
                        for k_num in k_section_num_list:
                            zero_mat_prod = np.zeros(kv_sparse_mat.T.shape)
                            zero_mat_prod[:, k_num_sum:int(k_num_sum+k_num)] = 1
                            kv_sparse_mat_temp = kv_sparse_mat.T * zero_mat_prod
                            indices_max_kv = np.argsort(kv_sparse_mat_temp, axis=1)[:, -self.cross_species_neibors_K_knn:]
                            kv_mat_ones = np.zeros(kv_sparse_mat_temp.shape)
                            for i in range(indices_max_kv.shape[0]):
                                for j in range(indices_max_kv.shape[1]):
                                    kv_mat_ones[i][indices_max_kv[i, j]] = 1
                            edge_ndarray_s = np.nonzero(kv_mat_ones)
                            edge_ndarray_temp = list(edge_ndarray_s)
                            edge_ndarray_temp[0] = edge_ndarray_temp[0] + v_add
                            edge_ndarray_temp[1] = edge_ndarray_temp[1] + k_add
                            if edge_ndarray_species == None:
                                edge_ndarray_species = edge_ndarray_temp
                            else:
                                edge_ndarray_species = [np.concatenate((edge_ndarray_species[0], edge_ndarray_temp[0])), \
                                        np.concatenate((edge_ndarray_species[1], edge_ndarray_temp[1]))]
                            k_num_sum = k_num_sum + k_num
                        v_num_sum = v_num_sum + v_num
            
            
            edge_ndarray_species = tuple(edge_ndarray_species)
        else:
            print('------------Skipped KNN cross-species triplets--------------')
            pass
        ##########################################################
        ## Generate cross-species MNN triplets
        print('Generate cross-species MNN triplets...')
        spe = 0
        for species_id, adata in adata_dict.items():
            homologs_column_name = self.species_ortholog_column_dict[species_id]
            orthologs_name = list(Gene_map_common_df[homologs_column_name])
            if spe == 0:
                whole_obs_name = list(adata.obs_names)
                whole_homologs_X = adata[:, orthologs_name].X
                print(whole_homologs_X.shape)
                spatial_coords = np.array(adata.obsm[self.graph_construct_key])
                if self.Smooth_spatial_neighbors[species_id] >= 1:
                    whole_homologs_X = whole_homologs_X.todense()
                    whole_homologs_X = average_spatial_neighbors(whole_homologs_X, spatial_coords, self.Smooth_spatial_neighbors[species_id])
                    whole_homologs_X = csr_matrix(whole_homologs_X)
                whole_species_name = list(adata.obs['species_id'])
                whole_slice_name = list(adata.obs['slice_name']) 
                whole_batch_name = list(adata.obs['batch_name'])
                whole_species_id = list(adata.obs['species_id'])
                if 'annotation' in adata.obs:
                    whole_annotation = list(adata.obs['annotation']) 
            else:
                whole_obs_name = whole_obs_name + list(adata.obs_names)
                whole_homologs_X_temp = adata[:, orthologs_name].X #.todense()
                print(whole_homologs_X_temp.shape)
                spatial_coords = np.array(adata.obsm[self.graph_construct_key])
                if self.Smooth_spatial_neighbors[species_id] >= 1:
                    whole_homologs_X_temp = whole_homologs_X_temp.todense()
                    whole_homologs_X_temp = average_spatial_neighbors(whole_homologs_X_temp, spatial_coords, self.Smooth_spatial_neighbors[species_id])
                    whole_homologs_X_temp = csr_matrix(whole_homologs_X_temp)
      
                whole_homologs_X = sparse.vstack([whole_homologs_X, whole_homologs_X_temp])
                whole_species_name = whole_species_name + list(adata.obs['species_id'])
                whole_slice_name = whole_slice_name + list(adata.obs['slice_name']) 
                whole_batch_name = whole_batch_name + list(adata.obs['batch_name'])
                whole_species_id = whole_species_id + list(adata.obs['species_id'])
                if 'annotation' in adata.obs:
                    whole_annotation = whole_annotation + list(adata.obs['annotation']) 
            spe += 1

        adata_whole = ad.AnnData(X = whole_homologs_X)
        adata_whole.obs_names = whole_obs_name
        adata_whole.obsm['homologs'] = whole_homologs_X
        adata_whole.obs['slice_name'] = whole_slice_name
        adata_whole.obs['batch_name'] = whole_batch_name
        adata_whole.obs['species_id'] = whole_species_id
        if 'annotation' in adata.obs:
            adata_whole.obs['annotation'] = whole_annotation
        # Batch correction
        
        if self.if_hvg_before_mnn:
            print('Run selecting hvgs before mnn...')
            adata_whole.X = scipy.sparse.csr_matrix(adata_whole.X)
            sc.pp.highly_variable_genes(adata_whole, flavor="seurat_v3", n_top_genes=self.homo_n_top_genes, inplace=True)
            adata_whole = adata_whole[:, adata_whole.var['highly_variable']]
            if self.if_combat_mnn:
                sc.pp.combat(adata_whole, key='species_id', covariates=None, inplace=True)
                if self.if_pca_before_mnn:
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X
            else:
                if self.if_pca_before_mnn:
                    print('pca...')
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X
        else:
            if self.if_combat_mnn:
                print('Run combat...')
                sc.pp.combat(adata_whole, key='species_id', covariates=None, inplace=True)
                if self.if_pca_before_mnn:
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X
            else:
                if self.if_pca_before_mnn:
                    #adata_whole.X = np.asarray(adata_whole.X)
                    print('pca...')
                    sc.tl.pca(adata_whole, svd_solver='arpack', n_comps=self.pca_dim)
                    adata_whole.obsm['homologs'] = adata_whole.obsm['X_pca']
                else:
                    adata_whole.obsm['homologs'] = adata_whole.X

        adata_whole_dict = {k:None for k in adata_dict.keys()}
        for k in adata_whole_dict.keys():
            adata_whole_dict[k] = adata_whole[adata_whole.obs['species_id'].isin([k])]

        print('create_dictionary_mnn...')
        mnn_dict = create_dictionary_mnn(adata_whole, use_rep='homologs', \
                                    batch_name='species_id', k=self.cross_species_neibors_K_mnn, iter_comb=None, verbose=0)
        section_ids = list(adata_dict.keys())
        anchor_ind_species = []
        positive_ind_species = []
        negative_ind_species = []
        for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
            batchname_list = adata_whole.obs['species_id'][mnn_dict[batch_pair].keys()]
            cellname_by_batch_dict = dict()
            for batch_id in range(len(section_ids)):
                cellname_by_batch_dict[section_ids[batch_id]] = adata_whole.obs_names[adata_whole.obs['species_id'] == section_ids[batch_id]].values
            anchor_list = []
            positive_list = []
            negative_list = []
            for anchor in mnn_dict[batch_pair].keys():
                anchor_list.append(anchor)
                positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                positive_list.append(positive_spot)
                section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                negative_list.append(
                    cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
            batch_as_dict = dict(zip(list(adata_whole.obs_names), range(0, adata_whole.shape[0])))
            anchor_ind_species = np.append(anchor_ind_species, list(map(lambda _: batch_as_dict[_], anchor_list)))
            positive_ind_species = np.append(positive_ind_species, list(map(lambda _: batch_as_dict[_], positive_list)))
            negative_ind_species = np.append(negative_ind_species, list(map(lambda _: batch_as_dict[_], negative_list)))

        
        print('Finished finding cross species triplets.')

       
        if self.if_integrate_within_species:
            print('Beginning to find cross section triplets...')
            anchor_ind_sections = []
            positive_ind_sections = []
            negative_ind_sections = []
            for species_id in adata_dict.keys():
                k_add = species_spot_add_dict[species_id]
                section_ids = adata_dict[species_id].obs['slice_name'].unique()
                print('slice_name:', section_ids)
                adata = adata_dict[species_id]
                #sc.pp.combat(adata, key='slice_name', covariates=None, inplace=True)
                adata.X = scipy.sparse.csr_matrix(adata.X)
                mnn_dict = create_dictionary_mnn(adata, use_rep='X_pca',batch_name='slice_name', k=self.cross_sections_neibors_K_mnn, iter_comb=None, verbose=0)

                for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                    batchname_list = adata.obs['slice_name'][mnn_dict[batch_pair].keys()]
                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(section_ids)):
                        cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[adata.obs['slice_name'] == section_ids[batch_id]].values
    
                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for anchor in mnn_dict[batch_pair].keys():
                        anchor_list.append(anchor)
                        positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                        positive_list.append(positive_spot)
                        section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                        negative_list.append(
                            cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])
    
                    batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                    anchor_ind_sections = np.append(anchor_ind_sections, [int(k_add + x) for x in list(map(lambda _: batch_as_dict[_], anchor_list))])
                    positive_ind_sections = np.append(positive_ind_sections, [int(k_add + x) for x in list(map(lambda _: batch_as_dict[_], positive_list))])
                    negative_ind_sections = np.append(negative_ind_sections, [int(k_add + x) for x in list(map(lambda _: batch_as_dict[_], negative_list))])


            triplet_ind_sections_dict = {'anchor_ind_sections':anchor_ind_sections,
                                   'positive_ind_sections':positive_ind_sections,
                                   'negative_ind_sections':negative_ind_sections}
            print('Number of cross-sections triplets:', len(anchor_ind_sections))
         
            
        triplet_ind_species_dict = {'anchor_ind_species':anchor_ind_species,
                                   'positive_ind_species':positive_ind_species,
                                   'negative_ind_species':negative_ind_species}

        
        # # Update edge_ndarray_species
        if self.knn_triplets:
            edge_1 = list(anchor_ind_species) + list(positive_ind_species) + list(list(edge_ndarray_species)[0])
            edge_2 = list(positive_ind_species) + list(anchor_ind_species) + list(list(edge_ndarray_species)[1])
        else:
            edge_1 = list(anchor_ind_species) + list(positive_ind_species)
            edge_2 = list(positive_ind_species) + list(anchor_ind_species)
        edge_ndarray_species = tuple([np.array(edge_1), np.array(edge_2)])


        if self.if_integrate_within_species:
            edge_1 = list(anchor_ind_sections) + list(positive_ind_sections)
            edge_2 = list(positive_ind_sections) + list(anchor_ind_sections)
            edge_ndarray_sections = tuple([np.array(edge_1), np.array(edge_2)])

        ###################################################################################
        ## Merge KNN and MNN triplets
        if self.knn_triplets:
            for k_species_order in range(len(species_id_list)-1):
                k_species = species_id_list[k_species_order]
                for v_species_other_order in range(k_species_order+1, len(species_id_list)):
                    v_species = species_id_list[v_species_other_order]
                    print(Species_cross_B_dict[k_species][v_species].toarray().shape)
            #####
            species_parameters = {'Species_cross_B_dict': Species_cross_B_dict,
                          'spot_name_species_dict':spot_name_species_dict, 
                          'spotname2id':spotname2id, 
                          'id2spotname':id2spotname, 
                          'knn_neigh_species': self.cross_species_neibors_K_knn,
                          'adata_species_section_num_dict':adata_species_section_num_dict}
    
            anchor_ind_species_all, positive_ind_species_all, negative_ind_species_all = STACAME.mnn_utils.get_species_triples(species_parameters)
            print('lenghth of anchor_ind_species:', len(anchor_ind_species_all))
            ## Running STACAME
            
            anchor_ind_all_num_dict = {k:0 for k in anchor_ind_species_all.keys()}
            
            subsampling_rate = self.knn_triplets_ratio
            
            anchor_ind_species = {}
            positive_ind_species = {}
            negative_ind_species = {}
            for s, anchor_list_s in anchor_ind_species_all.items():
                anchor_ind_all_num_dict[s] = len(anchor_list_s)
            
                anchor_ind_species[s] = []
                positive_ind_species[s] = []
                negative_ind_species[s] = []
            
            for spe_id in anchor_ind_all_num_dict.keys():
                spe_indices = random_list(anchor_ind_all_num_dict[spe_id], subsampling_rate)
            
                anchor_ind_species[spe_id] = [anchor_ind_species_all[spe_id][i] for i in spe_indices] 
                positive_ind_species[spe_id] = [positive_ind_species_all[spe_id][i] for i in spe_indices] 
                negative_ind_species[spe_id] = [negative_ind_species_all[spe_id][i] for i in spe_indices] 
            
            anchor_arr_species_ind = []
            positive_arr_species_ind = []
            negative_arr_species_ind = []
            
            k_add = 0
            species_add_dict = {k:None for k in adata_dict.keys()}
            for species_id in adata_dict.keys():
                species_add_dict[species_id] = int(k_add)
                k_add = int(k_add+adata_dict[species_id].n_obs)
            
            for species_id in adata_dict.keys():
                k_add = species_add_dict[species_id]
                anchor_arr_species_ind = anchor_arr_species_ind + \
                    [int(anch + k_add) for anch in anchor_ind_species[species_id]]
                for t in range(len(positive_ind_species[species_id])):
                    p_tuple = positive_ind_species[species_id][t]
                    positive_arr_species_ind.append(int(species_add_dict[p_tuple[0]] + p_tuple[1]))
                    n_tuple = negative_ind_species[species_id][t]
                    negative_arr_species_ind.append(int(species_add_dict[n_tuple[0]] + n_tuple[1]))
        
        if self.knn_triplets:
            triplet_ind_species_dict = {'anchor_ind_species':list(triplet_ind_species_dict['anchor_ind_species']) + 
                                    list(anchor_arr_species_ind), 
                                    'positive_ind_species':list(triplet_ind_species_dict['positive_ind_species']) + 
                                    list(positive_arr_species_ind), 
                                    'negative_ind_species':list(triplet_ind_species_dict['negative_ind_species']) + 
                                    list(negative_arr_species_ind)}
        else:
            triplet_ind_species_dict = {'anchor_ind_species':list(triplet_ind_species_dict['anchor_ind_species']),  
                                   'positive_ind_species':list(triplet_ind_species_dict['positive_ind_species']), 
                                   'negative_ind_species':list(triplet_ind_species_dict['negative_ind_species'])}
        
        print('Triplrts number = ', len(triplet_ind_species_dict['anchor_ind_species']))

        ###########################################################################################
        ## Add aligne specific number of homolgous genes and append them into the features
        print('Add aligne specific number of homolgous genes and append them into the features...')
        Gene_map_raw_df = pd.read_csv(Gene_map_raw_path, sep='\t')
        Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df['Gene name'].notna()]
        for v in self.species_ortholog_column_dict.values():
            Gene_map_dropna_df = Gene_map_raw_df[Gene_map_raw_df[v].notna()]
        for col in self.species_ortholog_type_dict.values():
            Gene_map_dropna_df = Gene_map_dropna_df[Gene_map_dropna_df[col].isin(['ortholog_one2one'])]
        ############
        Gene_map_common_df = Gene_map_dropna_df
        for species_id, column_name in self.species_ortholog_column_dict.items():
            common_gene_list = species_common_gene_list_dict[species_id]
            Gene_map_common_df = Gene_map_common_df[Gene_map_common_df[column_name].isin(common_gene_list)]
            print(Gene_map_common_df.shape)
            
        # Create gene map, use the first species genes as reference
        print('Create gene map, use the first species genes as reference...')
        gene_map_dict = {k:{} for k in adata_dict.keys()} 
        k_gene_map = 0
        for species_id, column_name in self.species_ortholog_column_dict.items():
            if k_gene_map == 0:
                gene_set = Gene_map_common_df[column_name].values
                gene_map_dict[species_id] = {k:v for k,v in zip(gene_set, gene_set)}
            else:
                gene_map_dict[species_id] = {k:v for k,v in zip(gene_set, Gene_map_common_df[column_name])}
            k_gene_map += 1

        
        k = 0
        adata_homo_dict = {k:{} for k in adata_dict.keys()} 
        for species_id, column_name in self.species_ortholog_column_dict.items():
            if k == 0:
                adata_homo = adata_dict[species_id][:, list(Gene_map_common_df[column_name].values)]
                adata_homo.var_names = list(Gene_map_common_df[column_name].values)
                adata_homo_dict[species_id] = adata_homo
            else:
                adata_homo = adata_dict[species_id][:, list(Gene_map_common_df[column_name].values)]
                adata_homo.var_names = list(Gene_map_common_df[column_name].values)
                adata_homo_dict[species_id] = adata_homo
            k += 1

        print('Concatenate gene matrix...')
        k = 0
        for adata in adata_homo_dict.values():
            if k == 0:
                expre_X = csr_matrix(adata.X)
                expre_spatial = adata.obsm[self.graph_construct_key]
                expre_obs_name = list(adata.obs_names)
                expre_slice_name = list(adata.obs['slice_name'])
                expre_batch_name = list(adata.obs['batch_name'])
                expre_species_id = list(adata.obs['species_id'])
                if 'annotation' in adata.obs:
                    expre_annotation = list(adata.obs['annotation'])
                expre_var_name = list(adata.var_names)
            else:
                print(f'Concatenate {k}-th gene matrix...')
                expre_X = sparse.vstack([expre_X, csr_matrix(adata.X)])
                expre_spatial = np.concatenate((expre_spatial, adata.obsm[self.graph_construct_key]), axis=0)
                expre_obs_name = expre_obs_name + list(adata.obs_names)
                expre_slice_name = expre_slice_name + list(adata.obs['slice_name'])
                expre_batch_name = expre_batch_name + list(adata.obs['batch_name'])
                expre_species_id = expre_species_id + list(adata.obs['species_id'])
                if 'annotation' in adata.obs:
                    expre_annotation = expre_annotation + list(adata.obs['annotation'])
                
            k += 1
        adata_concat = ad.AnnData(X = sp.csr_matrix(expre_X))
        adata_concat.obsm[self.graph_construct_key] = expre_spatial
        adata_concat.obs['slice_name'] = expre_slice_name
        adata_concat.obs['batch_name'] = expre_batch_name
        adata_concat.obs['species_id'] = expre_species_id
        if 'annotation' in adata.obs:
            adata_concat.obs['annotation'] = expre_annotation
        adata_concat.var_names = expre_var_name
        adata_concat.obs_names = expre_obs_name

        print('Select homologous HVGS...')
        sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=self.homo_n_top_genes)
        
        df = adata_concat.var['highly_variable']
        homo_gene_ref = df[df == True].index.tolist()
        for species_id, adata in adata_dict.items():
            adata_dict[species_id].uns['homo_highly_variable'] = [gene_map_dict[species_id][g] for g in homo_gene_ref]
            #print(adata_dict[species_id].uns['homo_highly_variable'][:100])
        
        #############################################################
        ## Reorder the genes such that the homologous genes are aligned well
        print('Reorder the genes such that the homologous genes are aligned well...')
        align_gene_k = 0
        reference_hvg_vec = None
        hvg_dict = {k:[] for k in adata_dict.keys()}
        
        gene_cap_upper_dict = self.gene_cap_upper_dict
        upper2origin_dict = {k:{} for k in adata_dict.keys()}
        for species_id in upper2origin_dict.keys():
            upper2origin_dict[species_id] = {k:v for k,v in zip([x.upper() for x in adata_dict[species_id].var_names], adata_dict[species_id].var_names)}
        
        for species_id, adata in adata_dict.items():
            df = adata.var['highly_variable']
            hvg_order = df[df == True].index.tolist()
            hvg_dict[species_id] = [x for x in hvg_order]
            hvg_dict[species_id].sort()
            #print(hvg_dict[species_id][1:100])
        
        hvg_intersect_set = set()
        for species_id, adata in adata_dict.items():
            #print(len(hvg_dict[species_id]))
            if align_gene_k == 0:
                hvg_intersect_set = set([x.upper() for x in hvg_dict[species_id]])
                align_gene_k += 1
            else:
                hvg_intersect_set = hvg_intersect_set.intersection(set(hvg_dict[species_id]))
                align_gene_k += 1
        
        hvg_intersect_set = list(hvg_intersect_set)
        print('Size of hvg intersect set: ', len(hvg_intersect_set))
        
        hvg_aligned_dict = {k:[] for k in adata_dict.keys()}

        gene_name_dict = {k:{'species_specific':[], 'homo_highly_variable':[]} for k in adata_dict.keys()}
        
        for species_id, adata in adata_dict.items():
            orthlogs = [upper2origin_dict[species_id][x] for x in hvg_intersect_set]
            hvg_aligned_dict[species_id] = adata_dict[species_id].uns['homo_highly_variable'] + list(set(hvg_dict[species_id]) - set(orthlogs))
            adata_dict[species_id].uns['highly_variable'] = hvg_aligned_dict[species_id]

            gene_name_dict[species_id]['species_specific'] = list(set(hvg_dict[species_id]) - set(orthlogs))
            gene_name_dict[species_id]['homo_highly_variable'] = adata_dict[species_id].uns['homo_highly_variable']

        if self.gene_save_path != None:
            if not os.path.exists(self.gene_save_path):
                os.makedirs(self.gene_save_path)
            np.save(self.gene_save_path + 'gene_name_dict.npy', gene_name_dict)
        #################################################################################
        print('Processing data finished.')
        stop = timeit.default_timer()
        print('Time used: ', stop - start)  

        if self.if_integrate_within_species:
            return adata_dict, triplet_ind_species_dict, edge_ndarray_species, triplet_ind_sections_dict, edge_ndarray_sections
        else:
            return adata_dict, triplet_ind_species_dict, edge_ndarray_species




def random_list(N, subsampling_rate):
    num = int(N * subsampling_rate)
    result = random.sample(range(0, N), num)
    result = list(set(result))
    result.sort()
    return result

def average_spatial_neighbors(X, coords, K):
    nn = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(coords)
    distances, indices = nn.kneighbors(coords)
    new_features = []
    for i in range(len(X)):
        nearest_indices = indices[i, :]
        nearest_features = X[nearest_indices]
        mean_features = np.mean(nearest_features, axis=0)
        new_features.append(mean_features)
    new_features = np.array(new_features).reshape(X.shape)
    return new_features
