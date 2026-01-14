import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_dual_volcano(
    df,
    lfc_comp1,
    lfc_comp2,
    pval_comp1,
    pval_comp2,
    alpha=0.05,
    fc_cutoff=1,
    label_comp1="Comparison 1",
    label_comp2="Comparison 2",
    title=None,
    size_scale=30
):
    df = df.copy()

    # Significance flags
    df["sig_1"] = df[pval_comp1] < alpha
    df["sig_2"] = df[pval_comp2] < alpha
    df[["sig_1", "sig_2"]] = df[["sig_1", "sig_2"]].fillna(False)

    # Assign colors
    def assign_color(row):
        if row["sig_1"] and row["sig_2"]:
            return "green"
        elif row["sig_2"]:
            return "blue"
        elif row["sig_1"]:
            return "orange"
        else:
            return "lightgrey"

    df["color"] = df.apply(assign_color, axis=1)

    # Point size scaled by significance
    min_p = df[[pval_comp1, pval_comp2]].min(axis=1)
    size = -np.log10(min_p.replace(0, np.nan))
    size = size.fillna(0)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(
        x=df[lfc_comp1],
        y=df[lfc_comp2],
        c=df["color"],
        s=size * size_scale + 10,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4
    )

    # Zero lines
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)

    # Fold-change cutoffs
    for val in [fc_cutoff, -fc_cutoff]:
        plt.axvline(val, color="red", linestyle=":", linewidth=2)
        plt.axhline(val, color="red", linestyle=":", linewidth=2)

    # Labels
    plt.xlabel(f"log2 Fold Change ({label_comp1})")
    plt.ylabel(f"log2 Fold Change ({label_comp2})")

    if title is None:
        title = f'Dual Volcano Plot: {label_comp1} vs {label_comp2}'
    plt.title(title)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Significant in both',
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label=f'Significant in {label_comp2}',
               markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label=f'Significant in {label_comp1}',
               markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Not significant',
               markerfacecolor='lightgrey', markersize=8),
    ]

    plt.legend(handles=legend_elements, frameon=False)
    plt.tight_layout()
    plt.show()

