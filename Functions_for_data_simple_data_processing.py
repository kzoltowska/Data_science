def random_boxplot(df,n):
  """ This function takes dataframe as an input and an integer with number of columns to plot"""
    import random
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.subplots(figsize=(20,5))
    random.seed(0)
    columns=[random.randint(0,df.shape[1]) for i in range(n)]
    df.iloc[:,columns].boxplot(grid=False, showfliers=False)
    plt.xticks(rotation=90)
    plt.show()


def adata_subsample(adata, column:str, target_obs:int):
#"""This function takes adata object, column name with class to subsample based on
#and number of target observations per class"""
    import scanpy as sc
    import anndata as ad
    adatas = [adata[adata.obs[column]==i] for i in adata.obs[column].unique()]
    for dat in adatas:
        if dat.shape[0]>target_obs:
            sc.pp.sample(dat, n=target_obs)
    adata_downsampled = ad.concat(adatas)
    return adata_downsampled
