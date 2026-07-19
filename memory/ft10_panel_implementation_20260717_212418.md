# FT10 canonical-panel implementation

**Note date:** 2026-07-17 21:24:18

`scripts/history/run_ft10_biocomplete.py` implemented an FT10 RF feature panel for
135 matched AraPheno Study 1 accessions. It combined canonical flowering-time
windows, 200 PC-adjusted GWAS hits and three kinship PCs. The numerical outputs
are retained under `outputs/history/ft10/ft10_biocomplete/`.

Canonical windows used Araport11 gene-body coordinates plus 50 kb on each side.
A minor-allele count of at least five was required among the matched accessions.
The canonical SNP counts were:

| Gene | Gene ID | SNPs |
|---|---|---:|
| FLC | AT5G10140 | 1,138 |
| FRI | AT4G00650 | 1,583 |
| FT | AT1G65480 | 2,831 |
| VIN3 | AT5G57380 | 1,356 |
| DOG1 | AT5G45830 | 3,003 |

No canonical window was empty, so a 100 kb fallback was unnecessary.
The 9,911 canonical SNPs and 200 GWAS-background SNPs had no overlap, yielding
10,111 unique SNPs plus the three PCs. Genotype indices were mapped to chromosome
and position using the HDF5 `positions.attrs["chr_regions"]` ranges.

The configurable parameters were `--canonical-window-bp 50000`,
`--top-gwas-snps 200` and `--min-mac 5`. These counts document feature extraction,
not predictive performance or evidence for particular biological interactions.
