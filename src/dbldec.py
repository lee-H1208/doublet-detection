import dbldec_utils as utils
import dbldec_clustering as clustering
import dbldec_density as density
import dbldec_generate as generate
import dbldec_naive as naive

import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy import sparse

def dbl_dec(adata, n_features=1000, random_state=1234, verbose=0):
    adata_copy = adata.copy()
    print(f"Selecting {n_features} features...")
    sel_features = utils.selFeatures(adata_copy, nfeatures=n_features, propMarkers=0)
    adata_copy = adata_copy[:, sel_features].copy()
    adata_copy.raw = adata_copy.copy();
    print("Finished!")

    # normalize
    print("Preprocessing...")
    utils.normalize(adata_copy)
    utils.log_transform(adata_copy)
    utils.preprocess(adata_copy, compute_umap=(verbose > 0))
    print("Finished!")

    # run clustering
    clustering.simple_clustering(adata_copy, verbose=verbose)
    group_key = 'leiden'

    # OPTIONAL: visualize real doublets
    if verbose == 4:
        print('Visualizing real doublets...')
        utils.visualize_real_doublets(adata_copy)

    # identify density outliers
    adata_clean = density.isolate_density_outliers(adata_copy, cluster_key=group_key, verbose=verbose)

    # generate artificial doublets
    n_cells = adata.n_obs
    print(f"Generating {n_cells} Doublets...")

    combined_adata = generate.generate_scdblfinder_doublets(adata_clean, n_doublets=n_cells, random_state=random_state)

    utils.normalize(combined_adata)
    utils.log_transform(combined_adata)
    print("Done!")

    # add cxds scores
    whichDbls = np.where(combined_adata.obs['type'] == 'synthetic')[0]
    utils.cxds2(combined_adata, whichDbls=whichDbls)

    # OPTIONAL: visualize artificial doublets
    if verbose == 3 or verbose == 4: 
        print("Visualizing Doublets...")
        utils.visualize_generated_doublets(combined_adata, 'type')

    combined_adata.X = sparse.csr_matrix(combined_adata.X)

    doublet_probs, doublet_preds = xgb_classifier(adata=combined_adata, verbose=verbose)
    # doublet_probs, doublet_preds = xgb_classifier_iterative(adata=combined_adata, verbose=verbose)

    return doublet_probs, doublet_preds

def split_anndata_stratified(adata, obs_key='cell_type', test_size=0.3, random_state=1234, verbose=0):
    """Split AnnData object with stratification based on an observation variable"""
    
    train_idx, test_idx = train_test_split(
        np.arange(adata.n_obs),
        test_size=test_size,
        random_state=random_state,
        stratify=adata.obs[obs_key]  # Stratify by cell type
    )
    
    # Create train and test AnnData objects
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()
    
    if verbose > 0:
        # Print stats about the split
        print(f"Training set: {train_adata.shape[0]} cells")
        print(f"Test set: {test_adata.shape[0]} cells")
        
        # Count cell types in each split
        print("\nCell type distribution:")

    train_counts = train_adata.obs[obs_key].value_counts()
    test_counts = test_adata.obs[obs_key].value_counts()
    
    for cell_type in sorted(train_adata.obs[obs_key].unique()):
        train_count = train_counts.get(cell_type, 0)
        test_count = test_counts.get(cell_type, 0)
        train_pct = train_count / train_adata.n_obs * 100
        test_pct = test_count / test_adata.n_obs * 100

        if verbose > 0: print(f"  {cell_type}: Train {train_count} ({train_pct:.1f}%), Test {test_count} ({test_pct:.1f}%)")
    
    return train_adata, test_adata

def get_features(adata, bdata=None, use_original=False):
    X_mat = adata.X

    # add naive scores
    if use_original:
        X_scores = sparse.csr_matrix(bdata.obs.loc[adata.obs_names, 'naive_doublet_score'].values.reshape(-1, 1))
    else:
        naive_scores_train = adata.obs['naive_doublet_score'].values.reshape(-1, 1)
        X_scores = sparse.csr_matrix(naive_scores_train)

    # add cxds scores
    cxds_scores = adata.obs['cxds_scores'].values.reshape(-1, 1)
    X_cxds = sparse.csr_matrix(cxds_scores)

    # add pca coordinates
    pca_arr = adata.obsm['X_pca']
    X_pca = sparse.csr_matrix(pca_arr)
    
    # add library sizes
    lib_sizes = adata.obs['lib_sizes'].values.reshape(-1, 1)
    X_lib = sparse.csr_matrix(lib_sizes)
    
    X_full = sparse.hstack([X_mat, X_scores, X_pca, X_cxds, X_lib])

    return X_full

def xgb_classifier(adata, verbose=0):
    # Extract training set
    if verbose > 0: print("\n=== Stratified Split (by origin) ===")

    naive.add_naive_doublet_score(adata)
    utils.calc_lib_sizes(adata)
    adata_use = adata[~adata.obs['density_outlier'], :].copy()
    train_data, test_data = split_anndata_stratified(adata=adata_use, obs_key='origin', verbose=verbose)
    
    train_X_ext = get_features(adata=train_data, use_original=False)
    test_X_ext = get_features(adata=test_data, use_original=False)

    # Prepare real data
    real_mask = adata.obs['type'] == 'real'
    real_names = adata.obs_names[real_mask]
    raw_adata = adata[real_names].copy()
    X_full = get_features(adata=raw_adata, use_original=False)

    # Train a Gradient Boosting classifier for doublet detection
    print("\nTraining Gradient Boosting classifier for heterotypic doublet detection...")

    y_test = test_data.obs.is_doublet

    clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=1,
        gamma=0,
        subsample=0.5,
        colsample_bytree=0.5,
        objective='binary:logistic',
        eval_metric=['aucpr', 'logloss'],
        nthread=4,
        scale_pos_weight=1,
        seed=42,
        verbosity=0)

    clf.fit(train_X_ext, train_data.obs.is_doublet)

    # Evaluate on test set
    y_pred = clf.predict(test_X_ext)
    y_proba = clf.predict_proba(test_X_ext)[:, 1]

    # OPTIONAL: Print classification report
    if verbose > 0:
        print("\nClassification Report for Heterotypic Doublet Detection:")
        print(classification_report(y_test, y_pred))

    # Predict Real Data
    doublet_probs = clf.predict_proba(X_full)[:, 1]

    # Print optimized threshold
    threshold = 0.5

    print(f'Threshold found: {threshold}')

    # Apply threshold
    doublet_preds = doublet_probs > threshold

    num_detected = doublet_preds.sum()
    print(f"Number of doublets detected by classifier: {num_detected}")

    return doublet_probs, doublet_preds

def xgb_classifier_iterative(adata, num_rounds=5, verbose=0):
    # Extract training set
    if verbose > 0: print("\n=== Stratified Split (by origin) ===")

    naive.add_naive_doublet_score(adata)
    utils.calc_lib_sizes(adata)
    adata_use = adata[~adata.obs['density_outlier'], :].copy()
    train_data, test_data = split_anndata_stratified(adata=adata_use, obs_key='origin', verbose=verbose)
    
    train_X_ext = get_features(adata=train_data, use_original=False)
    test_X_ext = get_features(adata=test_data, use_original=False)

    # Prepare real data
    real_mask = adata.obs['type'] == 'real'
    real_names = adata.obs_names[real_mask]
    raw_adata = adata[real_names].copy()
    X_full = get_features(adata=raw_adata, use_original=False)
    y_test = test_data.obs.is_doublet

    # Train a Gradient Boosting classifier for doublet detection
    print("\nTraining Gradient Boosting classifier for heterotypic doublet detection...")

    clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric=['aucpr', 'logloss'],
        nthread=4,
        scale_pos_weight=1,
        seed=42,
        verbosity=0)

    for i in range(num_rounds):
        clf.fit(train_X_ext, train_data.obs.is_doublet)

        # Evaluate on test set
        y_pred = clf.predict(test_X_ext)
        y_proba = clf.predict_proba(test_X_ext)[:, 1]

        # OPTIONAL: Print classification report
        if verbose > 0:
            print("\nClassification Report for Heterotypic Doublet Detection:")
            print(classification_report(y_test, y_pred))


    # Predict Real Data
    doublet_probs = clf.predict_proba(X_full)[:, 1]

    # Print optimized threshold
    threshold = 0.5

    print(f'Threshold found: {threshold}')

    # Apply threshold
    doublet_preds = doublet_probs > threshold

    num_detected = doublet_preds.sum()
    print(f"Number of doublets detected by classifier: {num_detected}")

    return doublet_probs, doublet_preds

def hyperparamter_optimization(X_train, y_train, X_test, y_test):
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

    space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

    def objective(space, X_train, y_train, X_test, y_test):
        clf = XGBClassifier(
                        n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
                        reg_alpha=int(space['reg_alpha']), reg_lambda=space['reg_lambda'],
                        min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=space['colsample_bytree'],
                        objective='binary:logistic',
                        eval_metric=['aucpr', 'logloss'],
                        early_stopping_rounds=10,
                        seed=space['seed'])

        evaluation = [( X_train, y_train), ( X_test, y_test)]

        clf.fit(X_train, y_train, eval_set=evaluation, verbose=False)

        pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, pred>0.5)
        print ("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK }

    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 100,
                            trials = trials)
    
    return best_hyperparams