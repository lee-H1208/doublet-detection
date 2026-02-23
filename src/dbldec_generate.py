import numpy as np
from itertools import product
from scipy.sparse import csr_matrix, issparse

import anndata

def generate_doublets(adata, random_state, cluster_key='main', group_key='leiden', n_doublets=None, simple_sum_prop=0.1, verbose=0):
    var_names = adata.raw.var_names
    var_data = adata.raw.var

    rng = np.random.default_rng(random_state)
    adata_use = adata[~adata.obs['density_outlier'], :].copy()

    if n_doublets is None:
        n_doublets = adata_use.n_obs

    X_use = adata_use.raw.X
    lib_pool = X_use.sum(axis=1)

    cluster_labels = adata_use.obs[cluster_key].values
    group_labels = adata_use.obs[group_key].values

    doublet_data = []
    parent_indices = []
    doublet_type = []

    if verbose:
        print(f"Generating {n_doublets} doublets using cluster filter: {cluster_key} vs {group_key}")

    attempts = 0
    max_attempts = n_doublets * 20

    while len(doublet_data) < n_doublets and attempts < max_attempts:
        idx1, idx2 = rng.choice(adata_use.n_obs, 2, replace=False)
        
        if cluster_labels[idx1] == cluster_labels[idx2]:
            attempts += 1
            continue

        if group_labels[idx1] == group_labels[idx2]:
            attempts += 1
            continue

        c1, c2 = X_use[idx1], X_use[idx2]

        if rng.random() < simple_sum_prop:
            doublet = (c1 + c2) / 2
        else:
            w = rng.beta(2, 2)
            doublet = w * c1 + (1.5 - w) * c2

        target_sum = rng.choice(lib_pool)
        doublet = doublet / doublet.sum() * target_sum

        doublet_data.append(doublet)
        parent_indices.append((idx1, idx2))
        doublet_type.append("confident_heterotypic")
        attempts += 1

    if verbose:
        print(f"Generated {len(doublet_data)} doublets after {attempts} attempts")

    doublets = np.vstack(doublet_data)

    doublet_adata = anndata.AnnData(X=doublets)
    doublet_adata.obs['type'] = 'synthetic'
    doublet_adata.obs['src'] = 'artificial'
    doublet_adata.obs['doublet_type'] = doublet_type
    doublet_adata.obs['density_outlier'] = False
    doublet_adata.var_names = var_names.copy()

    real_adata = anndata.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(), var=var_data.copy())
    real_adata.obs['type'] = 'real'
    real_adata.obs['src'] = 'real'
    real_adata.obs['doublet_type'] = 'real'
    real_adata.obs['density_outlier'] = adata.obs['density_outlier'].copy()

    combined_adata = anndata.concat(
        [real_adata, doublet_adata],
        label='origin',
        keys=['real', 'synthetic'],
        index_unique=None
    )
    combined_adata.obs['is_doublet'] = combined_adata.obs['type'].map({'real': 0, 'synthetic': 1})

    combined_raw_X = np.vstack([adata.raw.X.copy(), doublets])
    combined_adata.raw = anndata.AnnData(
        X=combined_raw_X,
        obs=combined_adata.obs.copy(),
        var=var_data.copy()
    )

    if verbose > 0:
        print(f"Successfully generated {len(doublets)} high-confidence heterotypic doublets.")

    return combined_adata

def generate_scdblfinder_doublets(adata, n_doublets=None, random_state=1234):
    adata_use = adata[~adata.obs['density_outlier'], :].copy()

    if n_doublets is None:
        n_doublets = adata_use.n_obs

    X_use = adata_use.raw.X
    var_names = adata.raw.var_names
    var_data = adata.raw.var

    df = getArtificialDoublets(x=X_use.T, propRandom=1, n=n_doublets, random_state=random_state)
    doublet_data = [d.T for d in df['counts']]
    doublets = np.vstack(doublet_data)
    doublets = doublets.T

    doublet_adata = anndata.AnnData(X=doublets)
    doublet_adata.obs['type'] = 'synthetic'
    doublet_adata.obs['src'] = 'artificial'
    doublet_adata.obs['density_outlier'] = False
    doublet_adata.var_names = var_names.copy()

    real_adata = anndata.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(), var=var_data.copy())
    real_adata.obs_names = adata.obs_names.copy()
    real_adata.obs['type'] = 'real'
    real_adata.obs['src'] = 'real'
    real_adata.obs['density_outlier'] = adata.obs['density_outlier'].copy()

    combined_adata = anndata.concat(
        [real_adata, doublet_adata],
        label='origin',
        keys=['real', 'synthetic'],
        index_unique=None
    )
    combined_adata.obs['is_doublet'] = combined_adata.obs['type'].map({'real': 0, 'synthetic': 1})

    combined_raw_X = np.vstack([adata.raw.X.copy(), doublets])
    combined_adata.raw = anndata.AnnData(
        X=combined_raw_X,
        obs=combined_adata.obs.copy(),
        var=var_data.copy()
    )

    return combined_adata


'''From scDblFinder'''
# x would probably be adata.X.T or something equivalent
def getArtificialDoublets(x, n=3000, clusters=None, 
                          resamp=0.25, halfSize=0.25, adjustSize=0.25,
                          propRandom=0.1, selMode="proportional",
                          random_state=1234,
                          trim_q=(0.05,0.95)):
    rng = np.random.default_rng(random_state)
    selMode = selMode.lower()
    valid_modes = ("proportional","uniform","sqrt")
    if selMode not in valid_modes:
        raise ValueError(f"Invalid selMode '{selMode}'. Must be one of {valid_modes}")
    
    # obtain library sizes
    ls = np.array(x.sum(axis=0)).flatten()

    # validate trim_q
    assert isinstance(trim_q, (tuple, list)) and len(trim_q) == 2
    assert all(isinstance(q, (float, int)) for q in trim_q)

    # get quantile thresholds
    lower_q, upper_q = np.quantile(ls, trim_q)

    # mask for filtering
    w = np.where((ls > 0) & (ls >= lower_q) & (ls <= upper_q))[0]

    # subset x to keep only selected cells
    x = x[:, w]

    # create random proportions
    if clusters is None or propRandom == 1:
        num_cells = x.shape[1]

        if num_cells**2 <= n:
            doublet_indices = np.array(list(product(range(num_cells), repeat=2)))
        else:
            doublet_indices = rng.integers(0, num_cells, size=(2 * n,)).reshape(-1, 2)
        
        doublet_indices = doublet_indices[doublet_indices[:, 0] != doublet_indices[:, 1]]

        if len(doublet_indices) > n:
            doublet_indices = doublet_indices[:n]

        ad_m, colnames = createDoublets(x, doublet_indices, adjustSize=False, resamp=resamp, halfSize=halfSize, prefix="rDbl.", random_state=random_state)
        oc = [None] * ad_m.shape[1]
        
        return {
            "counts": ad_m,
            "origins": np.array(oc),
            "colnames": colnames
        }
    
    # propRandom != 1, recursively call with propRandom=1
    # number of doublets to generate randomly in each step
    nr = int(np.ceil(n * propRandom))
    if nr > 0:
        result_dict = getArtificialDoublets(x, n=nr, clusters=clusters, halfSize=halfSize, 
                                          resamp=resamp, propRandom=1, adjustSize=adjustSize,
                                          random_state=random_state)
        ad_m = result_dict["counts"]
        colnames = result_dict["colnames"] 
        oc = result_dict["origins"]

        # adjust n
        n = int(np.ceil(n * (1 - propRandom)))
    else:
        n_genes = x.shape[0]
        if issparse(x):
            ad_m = csr_matrix((n_genes, 0), dtype=x.dtype)
        else:
            ad_m = np.zeros((n_genes, 0), dtype=x.dtype)
        
        oc = []
        colnames = []  
    
    # TODO: Add cluster-based doublet generation for remaining n doublets
    # For now, just return what we have from the random portion
    
    return {
        "counts": ad_m,
        "origins": np.array(oc),
        "colnames": colnames
    }

'''
dbl_idx: index df where each row represents a pair of cell indices to be combined into a doublet
'''
def createDoublets(x, dbl_idx, resamp=0.5,
                   halfSize=0.5, adjustSize=False, prefix="dbl.", random_state=1234):
    rng = np.random.default_rng(random_state)

    def check_prop_arg(val):
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        val = float(val)
        if val < 0 or val > 1:
            raise ValueError("Proportion arguments must be between 0 and 1.")
        return val

    adjustSize = check_prop_arg(adjustSize)
    halfSize = check_prop_arg(halfSize)
    resamp = check_prop_arg(resamp)

    num_pairs = dbl_idx.shape[0]
    n_adjust = int(round(adjustSize * num_pairs))

    # Resampled subset
    wAd = rng.choice(num_pairs, size=n_adjust, replace=False) 
   
    # Simple Summed subset
    wNad = np.setdiff1d(np.arange(num_pairs), wAd)

    # get matrix where each column is the sum of two cell's expression profiles
    x1 = x[:, dbl_idx[wNad, 0]] + x[:, dbl_idx[wNad, 1]]

    x = x1

    # Initialize half_indices to handle variable scope
    half_indices = np.array([], dtype=int)
    
    if halfSize > 0:
        n_cols = x.shape[1]
        half_indices = rng.choice(n_cols, size=int(np.ceil(halfSize * n_cols)), replace=False)
        
        if issparse(x):
            dense_cols = x[:, half_indices].toarray() / 2
            # Create new sparse matrix with modified columns
            x_dense = x.toarray()
            x_dense[:, half_indices] = dense_cols
            x = csr_matrix(x_dense)
        else:
            x[:, half_indices] = x[:, half_indices] / 2
    
    # poisson resampling
    if resamp > 0:
        if resamp != halfSize:
            n_cols = x.shape[1]
            resamp_indices = rng.choice(n_cols, size=int(np.ceil(resamp * n_cols)), replace=False)
        else:
            resamp_indices = half_indices
    
        if len(resamp_indices) > 0:
            if issparse(x):
                x_dense = x.toarray()
                sampled = rng.poisson(x_dense[:, resamp_indices])
                x_dense[:, resamp_indices] = sampled
                x = csr_matrix(x_dense)
            else:
                current_vals = x[:, resamp_indices]
                sampled = rng.poisson(current_vals)
                x[:, resamp_indices] = sampled
    else:
        if issparse(x):
            x.data = np.round(x.data)
        else:
            x = np.round(x)

    n_cols = x.shape[1]
    colnames = [f"{prefix}{i+1}" for i in range(n_cols)]

    return x, colnames
