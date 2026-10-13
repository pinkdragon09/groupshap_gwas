# Dataset: 1001 Genomes — Arabidopsis thaliana Natural Accessions

## Overview
Whole-genome sequencing data from 1,001 natural accessions of *Arabidopsis thaliana*. Provides the genotype backbone for GWAS and population genomics studies in this plant model organism.

## Species
- **Species**: *Arabidopsis thaliana* (thale cress)
- **Common name**: Thale cress / Mouse-ear cress
- **Kingdom**: Plantae
- **Reference assembly**: TAIR10

## Loci
- **Coverage**: Whole genome (all chromosomes)
- **Variant type**: SNPs (biallelic)
- **Pipeline**: Intersection of SHORE and GATK variant calling pipelines
- **Quality**: SHORE-based quality reference; population-level intersection calls
- **Minor allele count filter**: MAC ≥ 5 (for kinship matrix)

## Population
- **Sample size**: 1,001 natural accessions (lines)
- **Geographic origin**: Worldwide collection of natural *A. thaliana* accessions
- **Accession metadata**: `metadata/accessions.csv` (1,134 rows including non-sequenced entries)
  - Fields: accession ID, name, country, latitude, longitude, collector, collection date, CS number, species, genotype count, phenotype count

## Strain / Accession
- **Type**: Natural inbred accessions (near-homozygous lines)
- **Accession IDs**: Numeric IDs cross-referenced with TAIR and NASC stock collections (CS numbers)
- **Notable accessions**: Includes Col-0 (reference), Ler, and 999 other natural isolates

## Traits / Phenotypes
- Genotype-only dataset in this directory
- Paired phenotype data available via AraPheno (`../arapheno/`)
- Documented traits in associated studies: flowering time, defense, stress responses, morphology, climate adaptation

## Controls
- N/A (natural accession panel, not case-control)
- Kinship matrix (`kinship_ibs_mac5.hdf5`) provided to control for population structure in GWAS

## Data Formats
| File | Format | Description |
|------|--------|-------------|
| `genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5` | HDF5 (lzf-compressed) | Binary imputed SNP matrix, full genome (877 MB) |
| `genotypes/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5` | HDF5 (lzf-compressed) | IBS kinship matrix, MAC ≥ 5 (9.8 MB) |
| `genotypes/1001_SNP_MATRIX.tar.gz` | TAR.GZ archive | Compressed archive of above (317 MB) |
| `metadata/accessions.csv` | CSV (no header) | Accession metadata (1134 rows) |
| `metadata/GMI-MPI_RELEASE_README.txt` | Text | Pipeline and release notes |

### HDF5 Structure
- Library: h5py
- Compression: lzf filter
- Access tools: `tools/h5m2csv.py` (convert to CSV), `tools/hdf5_demo.py` (usage demo)

### Accession CSV Columns (inferred from API export)
Raw export from tools API — columns include: pk, name, country, latitude, longitude, collector, collection_date, cs_number, species, genotypes, count_phenotypes

## Imputation
- Method: Imputation from intersected SHORE + GATK calls
- Binary encoding used in HDF5 matrix

## Intended Use
- Genome-wide association studies (GWAS) in *A. thaliana*
- Population structure and demographic inference
- Kinship-corrected mixed model association (e.g., EMMA, GEMMA, FaST-LMM)
- Paired with AraPheno phenotypes for multi-trait analyses

## Source
- Project: 1001 Genomes Project (*A. thaliana*)
- URL: https://1001genomes.org/
- Data provider: GMI (Gregor Mendel Institute) / MPI (Max Planck Institute)
- Publication: 1001 Genomes Consortium 2016, *Cell* 166:481–491
