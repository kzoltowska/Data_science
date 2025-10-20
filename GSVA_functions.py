# This function performs differential analysis of GSVA scores between 2 groups
def diff_gsva(gsva_df, aligned_meta, group_col="Phenoconversion_phenoconv", group1="CTRL", group2="PD"):
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    if gsva.columns!=aligned_meta.index:
        print("GSVA data is not aligned with meta data")
        return None
    else:
        # Split samples into groups
        meta=aligned_meta.copy()
        group1 = meta[meta[group_col] == group1].index
        group2 = meta[meta[group_col] == group2].index

        # Perform t-tests for each gene set
        p_values = []
        u_stats = []
        group1_medians=[]
        group2_medians=[]

        for pathway in gsva_df.index:
            scores1 = gsva_df.loc[pathway, group1]
            scores2 = gsva_df.loc[pathway, group2]
            
            u_stat, p_val = mannwhitneyu(scores1.to_list(), scores2.to_list(), alternative='two-sided')
            
            p_values.append(p_val)
            u_stats.append(u_stat)
            group1_median=scores1.median()
            group2_median=scores2.median()
            group1_medians.append(group1_median)
            group2_medians.append(group2_median)

        # Multiple testing correction (Benjamini-Hochberg FDR)
        adjusted_pvals = multipletests(p_values, method='fdr_bh')[1]

        # Compile results
        results_df = pd.DataFrame({
            "pathway": gsva_df.index,
            "u_stat": u_stats,
            "p_value": p_values,
            "adj_p_value": adjusted_pvals,
            "{group1}_medians":group1_medians,
            "{group2}_medians":group2_medians
        })

        # Sort by adjusted p-value
        results_df = results_df.sort_values("p_value")

        return results_df
