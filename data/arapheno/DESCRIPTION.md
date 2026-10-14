# Dataset: AraPheno — Arabidopsis thaliana Phenotype Catalog

> Files described here are not redistributed in this archive. Fetch them with `scripts/download_data.sh`; see `data/README.md` for sources, citation and terms. Note: the export is now placed at `bulk/database.zip` (and linked at `database.zip`); per-study CSVs for Studies 1 and 43 are extracted into `phenotypes/`.

## Overview
Community phenotype repository for *Arabidopsis thaliana*, aggregating 537 phenotypes from 30 published GWAS/QTL studies across 7,425 accessions. Serves as the primary phenotype resource for GWAS analyses paired with 1001 Genomes genotype data.

## Species
- **Species**: *Arabidopsis thaliana* (thale cress)
- **Common name**: Thale cress / Mouse-ear cress
- **Kingdom**: Plantae
- **Reference assembly**: TAIR10 (genotype side)

## Loci
- N/A (phenotype-only dataset)
- Intended to be paired with 1001 Genomes genotype data (`../1001genomes/`) for GWAS

## Population
- **Accessions**: 7,425 entries in accession list (across all studies)
- **Geographic origin**: Worldwide natural *A. thaliana* collection
- **Accession metadata fields**: pk, name, country, latitude, longitude, collector, collection_date, cs_number, species, genotypes, count_phenotypes

## Strain / Accession
- **Type**: Natural inbred accessions
- **Accession IDs**: Numeric IDs cross-referenced with TAIR/NASC (CS numbers)
- **Species included**: Primarily *A. thaliana*; some entries for related species

## Traits / Phenotypes
- **Total phenotypes**: 537
- **Total studies**: 30
- **Ontology annotations** per phenotype:
  - `TO` (Plant Trait Ontology): trait term, name, definition
  - `EO` (Environment Ontology): growth condition term, name, definition
  - `UO` (Unit Ontology): measurement unit
- **Phenotype categories** (representative):
  - Flowering time and vernalization
  - Defense and pathogen resistance
  - Abiotic stress responses (drought, salt, temperature)
  - Root system architecture
  - Shoot and leaf morphology
  - Climate adaptation traits
  - Iron deficiency response (FRO2)
  - Phytohormone responses

### Phenotype File Columns (`metadata/phenotype_list.csv`)
```
species, phenotype_id, name, doi, study, scoring, source, type, growth_conditions,
integration_date, number_replicates,
to_term, to_name, to_definition, to_comment, to_source_acronym, to_source_name, to_source_url,
eo_term, eo_name, eo_definition, eo_comment, eo_source_acronym, eo_source_name, eo_source_url,
uo_term, uo_name, uo_definition, uo_comment, uo_source_acronym, uo_source_name, uo_source_url,
num_values
```

## Controls
- Replication information available via `number_replicates` field per phenotype
- Growth conditions documented via EO ontology terms
- Study-level experimental design details in `metadata/study_list.csv`

## Data Formats
| File | Format | Description |
|------|--------|-------------|
| `metadata/phenotype_list.csv` | CSV | 537 phenotypes with ontology annotations (641 KB) |
| `metadata/study_list.csv` | CSV | 30 studies with descriptions and phenotype counts (12 KB) |
| `metadata/accession_list.csv` | CSV | 7,425 accession entries with geographic metadata (538 KB) |
| `bulk/database.zip` | ZIP | Full AraPheno database export; per-study CSVs and PLINK-format files (1.6 MB) |
| `phenotypes/` | (empty) | Reserved for extracted per-phenotype data files |

### Bulk Database Contents
- Per-study phenotype value files in CSV format
- PLINK-compatible phenotype files for direct GWAS input

## Intended Use
- Phenotype input for GWAS using 1001 Genomes genotypes
- Multi-trait and multi-study meta-analyses
- Ontology-guided trait clustering and pleiotropy analysis
- QTL replication across studies and environments

## Source
- Database: AraPheno (https://arapheno.1001genomes.org/)
- Publications:
  - Seren et al. 2017, *Nucleic Acids Research* (AraPheno database)
  - Togninalli et al. 2020 (AraPheno + AraGWAS integration)
