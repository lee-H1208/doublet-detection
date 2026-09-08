import numpy as np
import scanpy as sc

import dbldec_utils as utils

'''Adapted naive doublet score from RADO'''

def calculate_naive_doublet_score(adata):
    n_neighbors = int(np.sqrt(adata.shape[0]))

    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=30, metric="euclidean")
    
    simulated_score = []
    simulated = np.array(adata.obs['type'] == 'synthetic')

    neighbors = (adata.obsp['distances'].todense() > 0)
    simulated_score = [
        (neighbors[i, :] & simulated).sum() / n_neighbors
        for i in range(len(adata))
    ]

    return np.array(simulated_score).reshape(-1, 1)

def add_naive_doublet_score(adata):
    naive_score = calculate_naive_doublet_score(adata)
    adata.obs['naive_doublet_score'] = naive_score

def remove_naive_doublets(adata):
    print('Removing naive doublets...')
    original_n = adata.n_obs

    naive_score = calculate_naive_doublet_score(adata)

    adata.obs['naive_doublet_score'] = naive_score
    adata.obs['naive_doublet_call'] = np.where(naive_score > 0.5, 1, 0)

    n_removed = np.sum((adata.obs['type'] == 'real') & (adata.obs['naive_doublet_call'] == 1))
    print(f"Naive doublets (real cells called doublet): {n_removed} / {original_n}")

    mask = ~((adata.obs['type'] == 'real') & (adata.obs['naive_doublet_call'] == 1))
    adata = adata[mask].copy()

    print(f"Remaining cells after removal: {adata.n_obs}")

    adata.X = adata.raw.X
    utils.normalize(adata)
    utils.log_transform(adata)
    print('Finished!\n')