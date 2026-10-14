#!/usr/bin/env bash
# Fetch the third-party input data that this archive does not redistribute.
#
# Usage:  scripts/download_data.sh [--kinship-only] [--skip-arabidopsis] [--skip-yeast]
#
# Every downloaded or derived file is checked against data/CHECKSUMS.sha256.
# Sources, citations and terms of use are documented in data/README.md.
# The Araport11 annotation cannot be fetched by script (TAIR requires a login);
# this script prints instructions for that one file.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

KINSHIP_ONLY=0; DO_ARABIDOPSIS=1; DO_YEAST=1
for arg in "$@"; do
  case "$arg" in
    --kinship-only) KINSHIP_ONLY=1 ;;
    --skip-arabidopsis) DO_ARABIDOPSIS=0 ;;
    --skip-yeast) DO_YEAST=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

fetch() {  # fetch URL DEST
  mkdir -p "$(dirname "$2")"
  if [ -s "$2" ]; then echo "  exists   $2"; return 0; fi
  echo "  download $2"
  curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

check() {  # check PATH...  (verifies listed paths against data/CHECKSUMS.sha256)
  local p
  for p in "$@"; do
    if grep -q "  $p\$" data/CHECKSUMS.sha256; then
      grep "  $p\$" data/CHECKSUMS.sha256 | sha256sum -c --quiet - && echo "  verified $p"
    else
      echo "  (no checksum recorded for $p)"
    fi
  done
}

if [ "$DO_ARABIDOPSIS" = 1 ]; then
  echo "== AraPheno phenotypes (Studies 1, 27, 43) =="
  # AraPheno regenerates this export, so its own checksum changes; the member files are verified instead.
  fetch https://arapheno.1001genomes.org/static/database.zip data/arapheno/bulk/database.zip
  # Some runners expect the export at data/arapheno/database.zip; provide the same bytes there too.
  ln -f data/arapheno/bulk/database.zip data/arapheno/database.zip 2>/dev/null || cp data/arapheno/bulk/database.zip data/arapheno/database.zip
  mkdir -p data/arapheno/phenotypes
  for m in 1/study_1_values.csv 1/study_1_phenotypes.csv 43/study_43_values.csv 43/study_43_phenotypes.csv; do
    unzip -p data/arapheno/bulk/database.zip "$m" > "data/arapheno/phenotypes/$(basename "$m")"
  done
  check data/arapheno/phenotypes/study_1_values.csv data/arapheno/phenotypes/study_1_phenotypes.csv \
        data/arapheno/phenotypes/study_43_values.csv data/arapheno/phenotypes/study_43_phenotypes.csv
  unzip -p data/arapheno/bulk/database.zip 27/study_27_values.csv > /dev/null && echo "  present  27/study_27_values.csv (inside the export)"
  fetch https://arapheno.1001genomes.org/rest/phenotype/list.csv data/arapheno/metadata/phenotype_list.csv
  fetch https://arapheno.1001genomes.org/rest/study/list.csv     data/arapheno/metadata/study_list.csv
  check data/arapheno/metadata/phenotype_list.csv data/arapheno/metadata/study_list.csv || \
    echo "  note: AraPheno metadata lists are live exports; a checksum difference means the catalog changed since 2026-03-31."

  echo "== 1001 Genomes imputed SNP matrix and kinship (332 MB download) =="
  TAR=data/1001genomes/genotypes/1001_SNP_MATRIX.tar.gz
  fetch https://1001genomes.org/data/GMI-MPI/releases/v3.1/SNP_matrix_imputed_hdf5/1001_SNP_MATRIX.tar.gz "$TAR"
  check "$TAR"
  if [ "$KINSHIP_ONLY" = 1 ]; then
    tar -xzf "$TAR" -C data/1001genomes/genotypes 1001_SNP_MATRIX/kinship_ibs_mac5.hdf5 1001_SNP_MATRIX/README 1001_SNP_MATRIX/md5sum.txt
  else
    tar -xzf "$TAR" -C data/1001genomes/genotypes 1001_SNP_MATRIX/kinship_ibs_mac5.hdf5 1001_SNP_MATRIX/imputed_snps_binary.hdf5 1001_SNP_MATRIX/README 1001_SNP_MATRIX/md5sum.txt
    (cd data/1001genomes/genotypes/1001_SNP_MATRIX && md5sum -c --quiet md5sum.txt && echo "  verified imputed_snps_binary.hdf5 (md5 from the release)")
  fi
  check data/1001genomes/genotypes/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5

  echo "== Araport11 annotation (manual step) =="
  GFF=data/reference/Araport11_GFF3_genes_transposons.Mar92021.gff.gz
  if [ -s "$GFF" ]; then
    check "$GFF"
  else
    mkdir -p data/reference
    if curl -fsL --retry 1 -o "$GFF.part" "https://www.arabidopsis.org/download_files/Genes/Araport11_genome_release/Araport11_GFF3_genes_transposons.Mar92021.gff.gz" 2>/dev/null; then
      mv "$GFF.part" "$GFF"; check "$GFF"
    else
      rm -f "$GFF.part"
      cat <<EOF
  TAIR does not allow scripted downloads. Log in at https://www.arabidopsis.org (free account),
  open Download > Genes > Araport11_genome_release, download
  Araport11_GFF3_genes_transposons.Mar92021.gff.gz and save it as
    $GFF
  then rerun this script to verify its checksum. TAIR's terms require citing a TAIR publication.
EOF
    fi
  fi
fi

if [ "$DO_YEAST" = 1 ]; then
  echo "== Bloom lab BYxRM yeast cross (Bloom et al. 2013) =="
  fetch http://genomics-pubs.princeton.edu/YeastCross_BYxRM/data/BYxRM_GenoData.txt  data/yeast-byxrm/raw/BYxRM_GenoData.txt
  fetch http://genomics-pubs.princeton.edu/YeastCross_BYxRM/data/BYxRM_PhenoData.txt data/yeast-byxrm/raw/BYxRM_PhenoData.txt
  check data/yeast-byxrm/raw/BYxRM_GenoData.txt data/yeast-byxrm/raw/BYxRM_PhenoData.txt

  echo "== Bloom et al. 2015 Supplementary Data 4 (QTL x QTL interactions) =="
  XLS=data/yeast-byxrm/raw/Bloom2015_SuppData4_QTLxQTL_interactions.xls
  CSV=data/yeast-byxrm/processed/Bloom2015_SuppData4_QTLxQTL_interactions.csv
  fetch "https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9712/MediaObjects/41467_2015_BFncomms9712_MOESM728_ESM.xls" "$XLS"
  check "$XLS"
  if [ ! -s "$CSV" ]; then
    if command -v soffice >/dev/null; then
      mkdir -p data/yeast-byxrm/processed
      TMP=$(mktemp -d)
      soffice --headless --convert-to csv --outdir "$TMP" "$XLS" >/dev/null 2>&1
      mv "$TMP"/*.csv "$CSV"; rmdir "$TMP"
    else
      echo "  LibreOffice (soffice) not found; convert the first sheet of $XLS to CSV as $CSV" >&2
    fi
  fi
  [ -s "$CSV" ] && check "$CSV"
fi

echo "done."
