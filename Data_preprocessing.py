def get_residuals(adata, design: str):
  """ This function takes adata with X containing the data to be processed
  and design the design to be used for model fitting.
  The function loops over genes (columns), fitting a model per gene and saves residuals to
  a new dataframe that is then placed in a new layer of anndata object """
    
    import statsmodels.formula.api as smf
    import pandas as pd
    import numpy as np

    # Fix gene names (hyphens are not allowed in formulas)
    adata.var_names = adata.var_names.str.replace('-', '_', regex=True)

    # Convert X to DataFrame 
    X_df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)

    # Combine expression data with design variables
    data = pd.concat([X_df, adata.obs], axis=1)

    # Pre-allocate array for residuals - it is faster with numpy
    residuals_matrix = np.zeros((adata.n_obs, adata.n_vars))

    # Iterate over genes and compute residuals
    for n, gene in enumerate(adata.var_names):
        formula = f"{gene} ~ {design}"
        model = smf.ols(formula=formula, data=data).fit()
        residuals_matrix[:, n] = model.resid.values

    # Convert to DataFrame
    residuals_df = pd.DataFrame(residuals_matrix, columns=adata.var_names, index=adata.obs_names)

    # Save to adata in a sepearate layer to keep the row data as well
    adata.layers["residuals"] = residuals_df

    return adata


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
