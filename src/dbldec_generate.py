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

def generate_scdblfinder_doublets(adata, n_doublets=None, random_state=1234,
                                  cluster_key='main', propRandom=0.5,
                                  selMode='sqrt'):
    """Generate artificial doublets, faithful to scDblFinder's cluster-aware scheme.

    A fraction ``propRandom`` of doublets is drawn from fully-random (but still
    heterotypic, when clusters are available) cell pairs; the remaining
    ``1 - propRandom`` are drawn from *cross-cluster* pairs whose per-combination
    counts follow the expected-doublet model E_ij = p_i p_j * dbr * N
    (see getExpectedDoublets). Restricting to heterotypic pairs removes the
    homotypic label-noise that pure random pairing injects (a homotypic doublet
    is ~indistinguishable from a singlet after library-size normalization).
    """
    adata_use = adata[~adata.obs['density_outlier'], :].copy()

    if n_doublets is None:
        n_doublets = adata_use.n_obs

    X_use = adata_use.raw.X
    var_names = adata.raw.var_names
    var_data = adata.raw.var

    # cluster labels for heterotypic / cluster-aware pairing (aligned to X_use cols)
    clusters = None
    if cluster_key is not None and cluster_key in adata_use.obs:
        clusters = np.asarray(adata_use.obs[cluster_key].values)

    df = getArtificialDoublets(x=X_use.T, clusters=clusters, propRandom=propRandom,
                               selMode=selMode, n=n_doublets, random_state=random_state)
    doublet_data = [d.T for d in df['counts']]
    doublets = np.vstack(doublet_data)
    doublets = doublets.T
    origins = df.get('origins', None)

    doublet_adata = anndata.AnnData(X=doublets)
    doublet_adata.obs['type'] = 'synthetic'
    doublet_adata.obs['src'] = 'artificial'
    doublet_adata.obs['density_outlier'] = False
    if origins is not None and len(origins) == doublets.shape[0]:
        doublet_adata.obs['doublet_origin'] = [str(o) for o in origins]
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

def getExpectedDoublets(clusters, dbr=None, only_heterotypic=True, dbr_per1k=0.008):
    """Expected number of doublets per cluster pair (scDblFinder, misc.R).

    Under Poisson loading of droplets, a doublet contains two cells drawn
    independently from the cell-type distribution, so the probability that a
    doublet is of combination (i, j) is p_i * p_j. The expected count is

        E_ij = p_i * p_j * dbr * N,

    where p_i is cluster i's proportion, N the number of cells, and dbr the
    overall doublet rate (default ~0.8% per 1000 cells). For heterotypic-only
    combinations we keep i < j and double the value to fold in the symmetric
    (j, i) term. Returns {(i, j): E_ij}, the sorted unique levels, and the
    integer cluster code per cell.
    """
    clusters = np.asarray(clusters)
    lvls, inv = np.unique(clusters, return_inverse=True)
    ncells = len(clusters)
    K = len(lvls)
    if dbr is None:
        dbr = dbr_per1k * ncells / 1000.0
    if K <= 1:
        return {}, lvls, inv

    cs = np.bincount(inv, minlength=K) / ncells      # cluster proportions p_i
    E = np.outer(cs, cs) * dbr * ncells              # E_ij = p_i p_j dbr N

    expected = {}
    for i in range(K):
        for j in range(K):
            if only_heterotypic:
                if i < j:
                    expected[(i, j)] = 2.0 * E[i, j]
            else:
                key = (min(i, j), max(i, j))
                expected[key] = expected.get(key, 0.0) + E[i, j]
    return expected, lvls, inv


def getCellPairs(clusters, n, selMode="proportional", soft_min=5, random_state=1234):
    """Select ~n heterotypic cell pairs weighted by the expected-doublet model.

    Each cluster combination (i, j) is allocated a number of pairs proportional
    to a weight derived from E_ij: 'proportional' uses E_ij directly, 'sqrt'
    dampens the dominance of abundant-abundant combinations (improving coverage
    of rarer cross-type doublets), and 'uniform' allocates equally. A soft floor
    (soft_min) guarantees every heterotypic combination is represented so the
    classifier sees the full space of possible doublet states.
    """
    rng = np.random.default_rng(random_state)
    expected, lvls, inv = getExpectedDoublets(clusters)
    if len(expected) == 0:
        return np.empty((0, 2), dtype=int), []

    keys = list(expected.keys())
    vals = np.array([expected[k] for k in keys], dtype=float)
    if selMode == "sqrt":
        vals = np.sqrt(vals)
    elif selMode == "uniform":
        vals = np.ones_like(vals)

    total = vals.sum()
    alloc = np.ceil(vals * n / total).astype(int) if total > 0 else np.zeros(len(keys), int)
    alloc = np.maximum(alloc, soft_min)

    idx_by_cluster = {c: np.where(inv == c)[0] for c in range(len(lvls))}

    pairs, origins = [], []
    for (i, j), cnt in zip(keys, alloc):
        ci, cj = idx_by_cluster[i], idx_by_cluster[j]
        if len(ci) == 0 or len(cj) == 0 or cnt == 0:
            continue
        a = rng.choice(ci, size=cnt, replace=True)
        b = rng.choice(cj, size=cnt, replace=True)
        for x_, y_ in zip(a, b):
            pairs.append((x_, y_))
            origins.append(f"{lvls[i]}+{lvls[j]}")

    if len(pairs) == 0:
        return np.empty((0, 2), dtype=int), []

    pairs = np.array(pairs, dtype=int)
    order = rng.permutation(len(pairs))
    pairs = pairs[order]
    origins = [origins[k] for k in order]
    if len(pairs) > n:
        pairs = pairs[:n]
        origins = origins[:n]
    return pairs, origins


# x would probably be adata.X.T or something equivalent (genes x cells)
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

    # mask for filtering extreme-library-size cells
    w = np.where((ls > 0) & (ls >= lower_q) & (ls <= upper_q))[0]

    # subset x (and clusters) to keep only selected cells
    x = x[:, w]
    if clusters is not None:
        clusters = np.asarray(clusters)[w]

    # ---- fully random branch (propRandom == 1 or no clusters) ----
    if clusters is None or propRandom >= 1:
        doublet_indices, oc = _random_pairs(x.shape[1], n, clusters, rng)
        ad_m, colnames = createDoublets(x, doublet_indices, adjustSize=False,
                                        resamp=resamp, halfSize=halfSize,
                                        prefix="rDbl.", random_state=random_state)
        return {"counts": ad_m, "origins": np.array(oc, dtype=object), "colnames": colnames}

    # ---- mixed branch: propRandom random + (1-propRandom) cluster-based ----
    nr = int(np.ceil(n * propRandom))
    nc = int(np.ceil(n * (1 - propRandom)))

    all_idx = []
    all_origins = []

    clus_idx, clus_oc = (np.empty((0, 2), dtype=int), [])
    if nc > 0:
        clus_idx, clus_oc = getCellPairs(clusters, nc, selMode=selMode,
                                         random_state=random_state)

    # any cluster-branch shortfall (e.g. degenerate single-cluster data) is
    # routed to the random branch so we still return ~n doublets.
    nr = nr + (nc - len(clus_idx))

    if nr > 0:
        rand_idx, rand_oc = _random_pairs(x.shape[1], nr, clusters, rng, enforce_het=False)
        all_idx.append(rand_idx)
        all_origins.extend(list(rand_oc))

    if len(clus_idx) > 0:
        all_idx.append(clus_idx)
        all_origins.extend(list(clus_oc))

    doublet_indices = np.vstack(all_idx) if all_idx else np.empty((0, 2), dtype=int)

    ad_m, colnames = createDoublets(x, doublet_indices, adjustSize=False,
                                    resamp=resamp, halfSize=halfSize,
                                    prefix="dbl.", random_state=random_state)

    return {"counts": ad_m, "origins": np.array(all_origins, dtype=object), "colnames": colnames}


def _origins_for(pairs, clusters):
    """Origin label per pair: 'i+j' for heterotypic, None for homotypic."""
    if clusters is None:
        return [None] * len(pairs)
    return [f"{clusters[a]}+{clusters[b]}" if clusters[a] != clusters[b] else None
            for a, b in pairs]


def _random_pairs(num_cells, n, clusters, rng, enforce_het=False, max_tries=20):
    """Draw ~n random cell-index pairs.

    ``enforce_het=False`` (scDblFinder-faithful, default here) keeps pairs fully
    random, so the natural homotypic fraction sum(p_i^2) is retained. ``True``
    keeps only heterotypic (different-cluster) pairs, back-filling from the
    unrestricted pool if the heterotypic pool is too small (single-cluster data).
    """
    if num_cells ** 2 <= n and clusters is None:
        idx = np.array(list(product(range(num_cells), repeat=2)))
    else:
        idx = rng.integers(0, num_cells, size=(2 * n * max_tries,)).reshape(-1, 2)

    idx = idx[idx[:, 0] != idx[:, 1]]

    if not enforce_het or clusters is None:
        idx = idx[:n]
        return idx, _origins_for(idx, clusters)

    het = idx[clusters[idx[:, 0]] != clusters[idx[:, 1]]]
    if len(het) >= n:
        het = het[:n]
        return het, _origins_for(het, clusters)

    # not enough heterotypic pairs -> back-fill with unrestricted random pairs
    shortfall = idx[:n - len(het)]
    combined = np.vstack([het, shortfall]) if len(het) else shortfall
    return combined, _origins_for(combined, clusters)

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

    half_indices = np.array([], dtype=int)
    
    if halfSize > 0:
        n_cols = x.shape[1]
        half_indices = rng.choice(n_cols, size=int(np.ceil(halfSize * n_cols)), replace=False)
        
        if issparse(x):
            dense_cols = x[:, half_indices].toarray() / 2
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