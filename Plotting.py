# Function for aggregating adata and plotting heatmaps grid
def heatmap_stat_adata(
    adata,
    layer,
    gene_dict,
    ag_column,
    func="median",  # use string instead of list
    ncols=3,
    nrows=3,
    figsize=(14, 10)
):
    """
    Plot heatmaps of gene expression grouped by a metadata column using mean or median aggregation.

    Parameters:
    - adata: AnnData object
    - layer: str, layer name in adata to use for aggregation
    - gene_dict: dict, dictionary of gene groups (e.g., {"T_cells": [gene1, gene2, ...]})
    - ag_column: str, column in adata.obs to group by (e.g., "cell_type")
    - func: str, aggregation function ("mean" or "median")
    - ncols: int, number of subplot columns
    - nrows: int, number of subplot rows
    - figsize: tuple, figure size
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc

    # Aggregate expression data
    adata_ag = sc.get.aggregate(adata, by=ag_column, func=func, layer=layer)
    df = pd.DataFrame(
        adata_ag.layers[func],  # assumes the aggregated result is stored in .layers[func]
        columns=adata_ag.var_names,
        index=adata_ag.obs_names
    )

    # Create subplots
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    axes = axes.flatten()

    # Plot each gene group
    for n, (cell_type, markers) in enumerate(gene_dict.items()):
        ax = axes[n]
        selected_genes = [g for g in markers if g in df.columns]
        df_subset = df.loc[:, selected_genes]
        sns.heatmap(df_subset, cmap="viridis", ax=ax)
        ax.set_title(cell_type)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Remove unused subplots
    for i in range(len(gene_dict), ncols * nrows):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.show()


# Function for aggregating adata and plotting clustermaps


def clustermap_stat_adata(
    adata,
    layer,
    gene_dict,
    ag_column,
    func=["median", "mean"],
    nrows=3,
    ncols=3,
    figsize=(6, 6),
    z_score=1,  # whether to z-score across rows (0) or columns (1) or None
    cmap="viridis"
):
    """
    Plot clustered heatmaps of gene expression grouped by a metadata column using mean or median aggregation.

    Parameters:
    - adata: AnnData object
    - layer: str, layer name in adata to use for aggregation
    - gene_dict: dict, dictionary of gene groups (e.g., {"T_cells": [gene1, gene2, ...]})
    - ag_column: str, column in adata.obs to group by (e.g., "cell_type")
    - func: str, aggregation function ("mean" or "median")
    - ncols: int, number of subplot columns
    - nrows: int, number of subplot rows
    - figsize: tuple, figure size for the whole figure (not each clustermap)
    - z_score: int or None, apply z-score normalization across rows (0) or columns (1)
    - cmap: str, colormap
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc
    import math

    # Aggregate expression data
    adata_ag = sc.get.aggregate(adata, by=ag_column, func=func, layer=layer)
    df = pd.DataFrame(
        adata_ag.layers[func],
        columns=adata_ag.var_names,
        index=adata_ag.obs_names
    )

    for i, (cell_type, markers) in enumerate(gene_dict.items()):
        ax = axes[i]
        df_subset = df.loc[:, [g for g in markers if g in df.columns]]
        # sns.clustermap creates its own figure, so plotting on ax is tricky.
        # Workaround: create clustermap, then move its axes to the subplot position.
        cg = sns.clustermap(
            df_subset,
            cmap=cmap,
            z_score=z_score,
            xticklabels=True,
            figsize=figsize,
            yticklabels=True,
            cbar=True
        )
        plt.title(cell_type)
        plt.show()
