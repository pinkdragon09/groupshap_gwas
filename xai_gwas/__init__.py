"""xai_gwas — Grouped Shapley and KAN explainability for GWAS / QTL mapping.

Public API surface:

    from xai_gwas.attribution import grouped_shapley_all, grouped_interaction_all
    from xai_gwas.attribution import summarize_grouped_shapley, conditional_permutation_importance, ale_effect
    from xai_gwas.grouping import build_ld_block_groups, build_gene_window_groups, build_cis_window_groups
    from xai_gwas.grouping import load_group_manifest
    from xai_gwas.data import load_tsv, load_hdf5_snps
    from xai_gwas.models import fit_random_forest, KanPredictShim
    from xai_gwas.viz import manhattan_plot, interaction_heatmap, ale_plot
"""

__version__ = "0.1.0"
