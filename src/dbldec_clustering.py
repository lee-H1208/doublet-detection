import warnings
import scanpy as sc
import matplotlib.pyplot as plt

def simple_clustering(adata, verbose=0):
    print('Running Main Clustering...')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, message='.*default backend for leiden.*')
        sc.tl.leiden(adata, resolution=2.0, key_added='main', flavor='leidenalg')

        print('Running Leiden Clustering...')
        sc.tl.leiden(adata, resolution=0.1, key_added='leiden', flavor='leidenalg')

    if verbose == 3:
        _, axes = plt.subplots(1, 2, figsize=(20, 8))

        sc.pl.umap(adata, color='leiden', size=15, ax=axes[0], show=False)
        axes[0].set_title('UMAP')

        sc.pl.pca(adata, color='leiden', size=15, ax=axes[1], show=False)
        axes[1].set_title('PCA')

        plt.show()

