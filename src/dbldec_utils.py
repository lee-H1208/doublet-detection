import math
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import sparse
from scipy.stats import binom

def normalize(adata):
    sc.pp.normalize_total(adata, exclude_highly_expressed=True, target_sum=1e4)

def log_transform(adata):
    sc.pp.log1p(adata)

def preprocess(adata, compute_umap=False):
    n_neighbors = int(np.sqrt(adata.shape[0]))

    sc.pp.pca(adata, svd_solver='arpack', n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=30, metric='euclidean')
    if compute_umap:
        sc.tl.umap(adata)

def visualize_real_doublets(adata):
    pca_coords = adata.obsm['X_pca'][:, :2]
    
    labels = adata.obs['y_true'].astype(int)
    
    cmap = ListedColormap(['lightgrey', 'red'])

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap=cmap, alpha=0.7, s=1)
    plt.colorbar(scatter, label='Real Doublet (1) / Singlet (0)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA plot with real doublets highlighted')
    plt.show()

def visualize_generated_doublets(adata, simdoublet_labels):    
    preprocess(adata)
    
    _, ax = plt.subplots(figsize=(6, 8))
    
    sc.pl.pca(
        adata,
        color=simdoublet_labels,
        size=15,
        ax=ax,
        palette=['lightgrey', 'red'],
        show=False
    )
    ax.set_title('Simulated Heterotypic Doublets')
    plt.tight_layout()
    plt.show()

    adata.X = adata.raw.X.copy()
    normalize(adata)
    log_transform(adata)

def optimize_threshold(doublet_probs, y_proba, thresholds, n_cells):
    expected_dbl = (n_cells ** 2) / 1e5
    FNR_arr = []
    FPR_arr = []
    size_discrep = []
    
    for t in thresholds:
        pot_dbl = np.sum(doublet_probs >= t)
        FNR_arr.append((np.sum(y_proba < t)/len(y_proba)))
        FPR_arr.append((pot_dbl / len(doublet_probs)))   
        size_discrep.append(((pot_dbl - expected_dbl) / expected_dbl)**2)
        
    cost = np.array(FNR_arr) + np.array(FPR_arr) + np.array(size_discrep)

    return thresholds[np.argmin(cost)]

def calc_lib_sizes(adata):
    lib_sizes = np.array(adata.raw.X.sum(axis=1)).flatten()
    lib_sizes = np.log1p(lib_sizes)
    low, high = np.percentile(lib_sizes, [5, 95])
    lib_sizes = np.clip(lib_sizes, low, high)

    adata.obs['lib_sizes'] = lib_sizes


'''From scDblFinder'''

def selFeatures(adata, clusters=None, nfeatures=1000, propMarkers=0, FDR_max=0.05):
    n_genes = adata.shape[1]
    
    if(n_genes <= nfeatures):
        return adata.var_names

    if clusters is None:
        propMarkers = 0

    '''ng is used for two cases, one with clusters, one without'''
    ng = math.ceil((1 - propMarkers) * nfeatures)

    if(ng > 0):
        if clusters is None:
            X = adata.X
            mean_expr = np.asarray(X.mean(axis=0)).ravel()
            top_gene_indices = np.argsort(mean_expr)[::-1][:ng]
            g = adata.var_names[top_gene_indices]
        '''Normally there would be an else for the clusters mode'''
    
    return g

def cxds2(adata, whichDbls=None, ntop=500, binThresh=None):
    x = adata.X.T
    if sparse.issparse(x):
        x = x.toarray()
    
    x = np.nan_to_num(x, nan=0.0)

    if binThresh is None:
        pNonZero = np.sum(x > 0) / x.size
        if pNonZero > 0.5:
            pNonZero = (x > 0).mean(axis=1)
            top_idx = np.argsort(pNonZero)[:ntop]
            x_sub = x[top_idx, :]
            thresh = np.median(x_sub)
            binThresh = max(1, int(thresh))
        else:
            binThresh = 1
    else:
        x_sub = x

    x_bin = (x >= binThresh).astype(int)
    ps = x_bin.mean(axis=1)

    if x_bin.shape[0] > ntop:
        score = ps * (1 - ps)
        top_hvg_idx = np.argsort(score)[::-1][:ntop]
        x_bin = x_bin[top_hvg_idx, :]
        ps = ps[top_hvg_idx]

    mask = np.ones(x_bin.shape[1], dtype=bool)
    if whichDbls is not None and len(whichDbls) > 0:
        mask[whichDbls] = False
    Bp = x_bin[:, mask]

    prb = np.outer(ps, 1 - ps)
    prb = prb + prb.T

    obs = Bp @ (1 - Bp.T)
    obs = obs + obs.T

    with np.errstate(divide='ignore'):
        S = binom.logsf(obs - 1, n=Bp.shape[1], p=prb)

    if np.isinf(S).any():
        finite_min = np.min(S[np.isfinite(S)])
        S[np.isinf(S)] = finite_min

    s_partial = -np.sum(x_bin * (S @ x_bin), axis=0)
    s_partial = s_partial - np.min(s_partial)
    if np.max(s_partial) > 0:
        s_partial = s_partial / np.max(s_partial)
    else:
        s_partial = np.zeros_like(s_partial)

    adata.obs['cxds_scores'] = s_partial
    return adata.obs['cxds_scores']
