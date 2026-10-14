# Yeast BYxRM (Bloom et al. 2013, 2015)

> Files described here are not redistributed in this archive. Fetch them with `scripts/download_data.sh`; see `data/README.md` for sources, citation and terms.

## Overview

Genotype, phenotype, and published QTL × QTL interaction tables from
the Bloom-lab BYxRM yeast cross. Two papers contribute the load-bearing
artifacts:

- **Bloom et al. 2013** (*Nature* 494:234–237, doi:10.1038/nature11867).
  Foundational cross: 1,008 haploid segregants from a BY × RM cross,
  46 quantitative growth traits. Genotype and phenotype tables are
  hosted as plain TSV at the Princeton Yeast Cross BYxRM resource page.
- **Bloom et al. 2015** (*Nature Communications* 6:8712,
  doi:10.1038/ncomms9712). Extended cross with 4,390 segregants and
  28,880 markers, plus the published two-locus (QTL × QTL) interaction
  list (205 interactions across 19 traits). Genotypes and phenotypes
  ship as RData inside a supplementary zip; the interaction list is
  Supplementary Data 4 (XLS, also converted to CSV here).

This dataset feeds the yeast-Bloom epistasis-recovery
analysis.
The two papers complement each other: 2013 gives the easy-to-load TSV
genotype/phenotype matrices (used for the smoke test); 2015 gives the
ground-truth interaction list for the actual epistasis-recovery
evaluation.

## Species

*Saccharomyces cerevisiae*. Cross between BY (lab strain S288C
derivative, "B" allele in genotype encoding) and RM
(vineyard isolate RM11-1a, "R" allele).

## Loci

- **2013 (Princeton TSV):** 11,623 markers across 16 chromosomes
  (chr01–chr16). Marker ID convention: `<index>_<chr>_<pos>_<ref>_<alt>`,
  e.g. `27915_chr01_27915_T_C`. Coordinates are `sacCer3` reference.
- **2015 (RData):** 28,880 markers (declared as 4,390 × 28,880
  segregant × marker matrix) — denser by linkage interpolation.
  Coordinates also `sacCer3`.

## Population

- **2013:** 1,008 F1 haploid segregants (column IDs `A01_01` …; an
  in-paper subset of 1,056 sequenced). Two-strain cross → very low LD
  outside immediate marker neighborhoods.
- **2015:** 4,390 F1 haploid segregants, expanding the 2013 panel.

## Strain / Accession

Parents: BY4716 (BY background) × RM11-1a (RM background).
Reference genome: `sacCer3`. Segregant IDs follow the lab's internal
plate-well scheme (e.g. `A01_01` = plate A01, well 01).

## Traits

- **2013 (Princeton phenotype matrix):** 46 traits, all colony-size
  growth measurements normalized for control-media growth. Trait
  columns include: `Cadmium_Chloride`, `Caffeine`, `Calcium_Chloride`,
  `Cisplatin`, `Cobalt_Chloride`, `Congo_red`, `Copper`,
  `Cycloheximide`, `Diamide`, `E6_Berbamine`, `Ethanol`, `Formamide`,
  `Galactose`, `Hydrogen_Peroxide`, `Hydroquinone`, `Hydroxyurea`,
  `Indoleacetic_Acid`, `Lactate`, `Lactose`, `Lithium_Chloride`,
  `Magnesium_Chloride`, `Magnesium_Sulfate`, `Maltose`, `Mannose`,
  `Menadione`, `Neomycin`, `Paraquat`, `Raffinose`, `SDS`, `Sorbitol`,
  `Trehalose`, `Tunicamycin`, `x4-Hydroxybenzaldehyde`, `x4NQO`,
  `x5-Fluorocytosine`, `x5-Fluorouracil`, `x6-Azauracil`, `Xylose`,
  `YNB`, `YNB:ph3`, `YNB:ph8`, `YPD`, `YPD:15C`, `YPD:37C`, `YPD:4C`,
  `Zeocin`. Each cell is the per-segregant trait mean across replicate
  measurements; raw per-replicate values live in `pheno_raw.Rdata`.
- **2015 published interaction list (Supp Data 4):** 205 QTL × QTL
  interactions across 19 traits. Trait names use a no-underscore
  convention (e.g. `CobaltChloride`, `CopperSulfate`,
  `ManganeseSulfate`); they map onto the 2013 Princeton trait names
  with minor renaming. Per-trait interaction counts:
  ManganeseSulfate (25), CopperSulfate (23), Neomycin (18),
  E6-Berbamine (15), Lactate (13), YPD (13), Formamide (11),
  IndolaceticAcid (11), Raffinose (11), Lactose (10), Zeocin (10),
  CobaltChloride (9), Ethanol (7), Diamide (6), Menadione (6),
  Xylose (6), YNB (5), MagnesiumChloride (3), Trehalose (3).

## Controls

The phenotype values in `BYxRM_PhenoData.txt` are *colony sizes
already normalized for growth on control media*, per the Princeton
README. No additional negative-control set is shipped with these files;
`pheno_raw.Rdata` contains the per-replicate raw measurements if
reanalysis of the normalization is needed.

## Data Formats

- `BYxRM_GenoData.txt` (Princeton, 22.6 MB): tab-delimited.
  Row 1 = header (`marker` + 1,008 segregant IDs). Each subsequent row
  is one marker; cells are `B` (BY allele) or `R` (RM allele). To
  feed sklearn, transpose and recode `B → 0`, `R → 1` (or `±1`).
- `BYxRM_PhenoData.txt` (Princeton, 779 KB): tab-delimited.
  Row 1 = header (empty cell + 46 trait names). Rows 2– = one segregant
  per row (segregant ID in col 0, then 46 numeric phenotype columns).
  Some cells are `NA`.
- `cross.Rdata` (Princeton, 21.2 MB): R/QTL `cross` object with
  averaged phenotypes + genotype calls + genetic map.
- `pheno_raw.Rdata` (Princeton, 853 KB): R list `pheno_raw`, one
  element per phenotype, containing per-replicate measurements.
- `Bloom2015_SuppData5_genotypes_phenotypes.zip` (12.9 MB): outer zip
  containing `genotypes.zip` (→ `genotypes.RData`, 30.6 MB extracted)
  and `phenotypes.zip` (→ `phenotypes.RData`, 1.5 MB extracted).
  Both extracted to `processed/Supplementary_Data_5/`.
- `Bloom2015_SuppData{1..4}.xls` (Excel BIFF8): supplementary tables
  from the 2015 paper. Converted to CSV via LibreOffice
  (`processed/*.csv`). The interaction list lives in **Supp Data 4**:
  columns `trait, Q1_index, Q2_index, Q1_chr, Q1_pos, Q2_chr, Q2_pos,
  LOD, lm_coefficient, variance_explained, closest.additive.Q1,
  closest.additive.Q2, number of interactors that are not additve QTL`.
- `Bloom2015_SuppInfo_Figs1-5.pdf` (206 KB): Supplementary Figures 1–5
  from the 2015 paper.

The Princeton TSV files load into pandas in seconds. The 2015 RData
files require either `pyreadr` (Python) or R itself; neither is
currently on the project Python environment, so any analysis that needs
the 4,390-segregant expansion is **deferred** until the loader can be
extended (see "Intended Use" below). The CSV-converted Supp Data 4 is
sufficient for the v1 epistasis-recovery target since the interaction
list is the actual ground truth.

## Intended Use

Yeast-Bloom epistasis-recovery analysis
(smoke test → analysis):

1. **Smoke test (pass criterion):** train RandomForestRegressor
   (200 trees, 5-fold CV, seed 42) on the 2013 genotype matrix
   (1,008 × 11,623) → one phenotype with published 2-locus
   interactions. The default smoke-test phenotype is `Raffinose`
   (11 published interactions in Supp Data 4, present in Princeton
   phenotype table). Confirm CV R² > 0 above marginal variance.
2. **Analysis (separate work):** train RF / MLP on the
   four highest-interaction traits and compute pairwise
   interaction-Shapley values across the top-N markers; compare
   top-K predicted pairs against the published Supp Data 4 list.
   Report precision + recall at K ∈ {10, 25, 50}.
3. **Optional 2015 expansion:** load the 4,390 × 28,880 RData matrices
   (requires pyreadr) once a denser segregant panel is justified.

This dataset is **not** under the dbGaP-class access restrictions
— both Princeton TSVs and Bloom
2015 supplementary files are openly hosted, no DUA required.

## Source

All URLs verified with `curl -I` returning HTTP 200 on **2026-04-25**
in this acquisition session (see `MANIFEST.tsv` for SHA256, sizes,
ETags, and `Last-Modified` timestamps; manifest is the authoritative
provenance record).

- **Princeton Yeast Cross BYxRM (Bloom 2013 data):**
  - Landing: <http://genomics-pubs.princeton.edu/YeastCross_BYxRM/data.shtml>
  - `BYxRM_GenoData.txt`: <http://genomics-pubs.princeton.edu/YeastCross_BYxRM/data/BYxRM_GenoData.txt>
  - `BYxRM_PhenoData.txt`: <http://genomics-pubs.princeton.edu/YeastCross_BYxRM/data/BYxRM_PhenoData.txt>
  - `cross.Rdata`: <http://genomics-pubs.princeton.edu/YeastCross_BYxRM/data/cross.Rdata>
  - `pheno_raw.Rdata`: <http://genomics-pubs.princeton.edu/YeastCross_BYxRM/data/pheno_raw.Rdata>
- **Bloom 2015 supplementary data (Nature Communications, Springer
  static-content):**
  - Article: <https://www.nature.com/articles/ncomms9712> (doi:10.1038/ncomms9712)
  - Supp Info PDF: <https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9712/MediaObjects/41467_2015_BFncomms9712_MOESM724_ESM.pdf>
  - Supp Data 1 (XLS): <https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9712/MediaObjects/41467_2015_BFncomms9712_MOESM725_ESM.xls>
  - Supp Data 2 (XLS): <https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9712/MediaObjects/41467_2015_BFncomms9712_MOESM726_ESM.xls>
  - Supp Data 3 (XLS, additive-QTL list): <https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9712/MediaObjects/41467_2015_BFncomms9712_MOESM727_ESM.xls>
  - **Supp Data 4 (XLS, QTL × QTL interaction list):** <https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9712/MediaObjects/41467_2015_BFncomms9712_MOESM728_ESM.xls>
  - Supp Data 5 (ZIP, genotypes + phenotypes RData): <https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9712/MediaObjects/41467_2015_BFncomms9712_MOESM729_ESM.zip>

### Citations

- Bloom JS, Ehrenreich IM, Loo WT, Lite T-LV, Kruglyak L. Finding the
  sources of missing heritability in a yeast cross. *Nature*
  494:234–237 (2013). doi:10.1038/nature11867.
- Bloom JS, Kotenko I, Sadhu MJ, Treusch S, Albert FW, Kruglyak L.
  Genetic interactions contribute less than additive effects to
  quantitative trait variation in yeast. *Nature Communications*
  6:8712 (2015). doi:10.1038/ncomms9712.

### Out-of-scope sources

The Bloom 2019 *eLife* paper (8:e49212, "Rare variants contribute
disproportionately to quantitative trait variation in yeast") was
considered as a fallback in case the 2013 and 2015 datasets did not
provide the interaction list at locus resolution. Supp Data 4 in the
2015 paper *does* provide locus-resolution interactions
(`Q1_chr/Q1_pos/Q2_chr/Q2_pos`), so the 2019 paper is not needed for
this analysis and was not downloaded.

The 44 GB `BYxRM_segs_saccer3.bam` raw-reads file on the Princeton
page is excluded by the per-file 500 MB cap (and is unnecessary for
the planned analysis — processed genotype calls are sufficient).
