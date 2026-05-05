import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import itertools
import networkx as nx
import hnswlib

import numpy as np
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


def acquire_pairs(X, Y, k=30, metric='angular'):
    # This function was modified from iMAP: https://github.com/Svvord/iMAP/blob/master/imap/stage2.py
    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat).astype(int)
    return mnn_mat

def random_indices(Sim_mat, knn_neigh_species):
    
    X_rows, X_cols = Sim_mat.shape
    indices_random = []
    for i in range(X_rows):
        Y_vec = np.random.choice(X_cols, knn_neigh_species, replace=False)
        indices_random.append(Y_vec)
    indices_random = np.array(indices_random)
    #print(Y)
    return indices_random

    
def get_species_triples(species_parameters):
    r'''
     params: species_parameters = { 'Species_cross_B_dict': Species_cross_B_dict,
                      'spot_name_species_dict':spot_name_species_dict, 
                      'spotname2id':spotname2id, 
                      'id2spotname':id2spotname, 
                      'knn_neigh_species': 20, 
                      'adata_species_section_num_dict':adata_species_section_num_dict}
     Generate cross-species triplets loss positive, negative samples index
     Use example: 
     - anchor_arr_species[species_id] = [23, 43, 45, ...]
     - positive_arr_species[species_id] = [(species_id1, 34), (species_id2, 300), (species_id2, 600), ...]
    '''
    #B_dist = species_parameters['B_dist']
    spot_name_species_dict = species_parameters['spot_name_species_dict']
    spotname2id = species_parameters['spotname2id']
    id2spotname = species_parameters['id2spotname']
    Species_cross_B_dict = species_parameters['Species_cross_B_dict']
    knn_neigh_species = int(species_parameters['knn_neigh_species'])

    adata_species_section_num_dict = species_parameters['adata_species_section_num_dict']
    
    N_spot = len(spotname2id)

    anchor_ind_species = {k:[] for k in Species_cross_B_dict.keys()}
    positive_ind_species = {k:[] for k in Species_cross_B_dict.keys()}
    negative_ind_species = {k:[] for k in Species_cross_B_dict.keys()}

    triple_set = []

    for species_id_k in Species_cross_B_dict.keys():
        triple_set_temp = []
        k_section_num_list = adata_species_section_num_dict[species_id_k]
        for species_id_v in Species_cross_B_dict.keys():
            if species_id_k != species_id_v:
                v_section_num_list = adata_species_section_num_dict[species_id_v]
                Sim_mat = Species_cross_B_dict[species_id_k][species_id_v].toarray()
                k_num_sum = 0
                for k_num in k_section_num_list:
                    v_num_sum = 0
                    Sim_mat_k = Sim_mat[k_num_sum:int(k_num_sum+k_num), :]
                    for v_num in v_section_num_list:
                        
                        zero_mat_prod = np.zeros(Sim_mat_k.shape)
                        zero_mat_prod[:, v_num_sum:int(v_num_sum+v_num)] = 1
                        Sim_mat_temp = Sim_mat_k * zero_mat_prod
                       
                        indices_max = np.argsort(Sim_mat_temp, axis=1)[:, -knn_neigh_species:]
                        indices_min = random_indices(Sim_mat_temp[:, v_num_sum:int(v_num_sum+v_num)], knn_neigh_species)

                        for i in range(Sim_mat_temp.shape[0]):
                            for j in range(knn_neigh_species):
                                triple_set_temp.append([species_id_k, i, species_id_v, indices_max[i][j], species_id_v, int(indices_min[i][j]+v_num_sum)])
                        v_num_sum = v_num_sum + v_num
                    
                    k_num_sum = k_num_sum + k_num

        
        triple_set = triple_set + triple_set_temp

    
    for i in range(len(triple_set)):
        triple_set[i] = tuple(triple_set[i])
    triple_set = list(set(triple_set))
    
    print(f'Cross-species triple number:{len(triple_set)}')

    for i in range(len(triple_set)):
        tri_list = triple_set[i]
        species_id = tri_list[0]
        anchor_ind_species[species_id].append(tri_list[1])
        positive_ind_species[species_id].append((tri_list[2], tri_list[3]))
        negative_ind_species[species_id].append((tri_list[4], tri_list[5]))

    return anchor_ind_species, positive_ind_species, negative_ind_species



def create_dictionary_mnn(adata, use_rep, batch_name, k = 50, save_on_disk = True, approx = True, verbose = 1, iter_comb = None):

    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    mnns = dict()

    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[key_name1] = {} # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        if(verbose > 0):
            print('Processing datasets {}'.format((i, j)))

        new = list(cells[j])
        ref = list(cells[i])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk = save_on_disk, approx = approx)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            mnns[key_name1][key]= names
    return(mnns)

def validate_sparse_labels(Y):
    if not zero_indexed(Y):
        raise ValueError('Ensure that your labels are zero-indexed')
    if not consecutive_indexed(Y):
        raise ValueError('Ensure that your labels are indexed consecutively')


def zero_indexed(Y):
    if min(abs(Y)) != 0:
        return False
    return True


def consecutive_indexed(Y):
    """ Assumes that Y is zero-indexed. """
    n_classes = len(np.unique(Y[Y != np.array(-1)]))
    if max(Y) >= n_classes:
        return False
    return True


def nn_approx(ds1, ds2, names1, names2, knn=50, random_seed=42):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M=16, random_seed=random_seed)
    p.set_ef(10)
    p.add_items(ds2)
    ind, distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


# def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
#     # Find nearest neighbors of first dataset.
#     nn_ = NearestNeighbors(knn, p=metric_p)
#     nn_.fit(ds2)
#     ind = nn_.kneighbors(ds1, return_distance=False)

#     match = set()
#     for a, b in zip(range(ds1.shape[0]), ind):
#         for b_i in b:
#             match.add((names1[a], names2[b_i]))

#     return match

def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(n_neighbors=knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def nn_annoy(ds1, ds2, names1, names2, knn = 20, metric='euclidean', n_trees = 50, save_on_disk = True):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    if(save_on_disk):
        a.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def mnn(ds1, ds2, names1, names2, knn = 20, save_on_disk = True, approx = True):
    if approx: 
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual
