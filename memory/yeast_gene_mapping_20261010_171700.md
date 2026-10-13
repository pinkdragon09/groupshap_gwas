# Bloom yeast locus-to-gene annotation

**Note date:** 2026-10-10 17:17:00

`scripts/history/yeast_bloom_gene_mapping.py` implemented deterministic locus-to-gene
annotation without using deletion-interaction evidence to choose mappings.
Seventeen boundary, reference and naming tests were recorded as passing.

The annotation used an Ensembl release 99 SGD-import R64-1-1 GFF, on the same
sequence assembly as Bloom SacCer3, with October 2018 annotation. It contained
6,572 nuclear protein-coding spans; excluding 645 explicitly dubious ORFs left
5,927 records not flagged as dubious. That filter did not infer separate
verified/uncharacterized statuses or mix in R64-3-1 gene coordinates.

The primary mapping used exact gene-body overlap. Sensitivity mappings used
1, 5, 10 and 50 kb windows plus nearest-gene ties, retaining ambiguous candidates.

The additive-QTL table's SNP positions and refined intervals were not interchangeable.
Only 14 interaction endpoints exactly matched the additive SNP positions.
Explicit closest-additive indices linked 410 endpoints to additive records, but
15 links crossed chromosomes and only 152 linked intervals contained the endpoint.
Those links were therefore kept as separate sensitivity annotations, not treated
as interaction confidence intervals. This record explains the mapping checks;
it does not claim biological validation of the annotated pairs.
