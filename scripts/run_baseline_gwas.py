import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Load merged data
data_path = 'outputs/baseline_gwas/ft16_chr1_subset.tsv'
df = pd.read_csv(data_path, sep='\t')

# Target and SNP columns
y = df['FT16'].values
snp_cols = [c for c in df.columns if c.startswith('Chr1_')]

# Run GWAS: Simple Linear Regression for each SNP
results = []
print(f"Running linear GWAS on {len(snp_cols)} SNPs...")

for snp in snp_cols:
    x = df[snp].values
    # Skip if no variation
    if len(np.unique(x)) < 2:
        continue
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    results.append({
        'SNP': snp,
        'POS': int(snp.split('_')[1]),
        'BETA': slope,
        'P': p_value,
        'NEG_LOG10_P': -np.log10(p_value) if p_value > 0 else 300.0
    })

results_df = pd.DataFrame(results).sort_values('P')

# Save results
outdir = Path('outputs/baseline_gwas')
results_df.to_csv(outdir / 'ft16_baseline_gwas_results.tsv', sep='\t', index=False)

print(f"GWAS completed. Top 10 SNPs:")
print(results_df.head(10))
