# GroupSHAP GWAS reproducibility archive

This archive contains numerical analysis code, frozen inputs, and saved selected output.
Source data are only referenced but not included in this public release. 


RF and KAN are the primary fitted models. Transformer material is historical.
ALE uses `PyALE==1.2.0`.

## Setup and checks

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/run_study27_fixed_pc_baseline.py --outdir /tmp/study27_pc_new_run
```

`requirements.txt` pins PyALE and pykan; other dependencies have version ranges.
Original numerical run manifests record the environments used for their outputs.
Exact floating-point reproduction may depend on those environments.

The two Study 27 support runners, `run_study27_fixed_pc_baseline.py` and
`run_study27_training_groupshap.py`, load the 81-by-203 panel through
`study27_fixed_panel.py`; build it first with `scripts/build_study27_fixed_panel.py`. They require a fresh output directory and do not import
the obsolete Phase 5 workflow. The panel contains 200 SNPs and three kinship PCs;
the SNP selection used the full phenotype before cross-validation.

Other inherited runners retain their original source paths and defaults. Adjust
those paths to the archive layout before a fresh run, and write results to a new directory. Their inclusion does not mean
that every historical entry point has been made portable or rerun in this archive.
Third-party inputs (1001 Genomes genotypes and kinship, AraPheno phenotypes, the
Araport11 annotation, and the Bloom yeast cross data) are not redistributed in this
archive. Run `scripts/download_data.sh` to fetch them from their providers; sources,
citations, terms of use and checksums are listed in `data/README.md`. The derived
Study 27 panel is rebuilt from those inputs rather than distributed.

## Scope

Stored scientific outputs are preserved as evidence of their recorded runs.
Archive assembly does not rerun or independently validate all reported analyses.

## Citation

If you use this codebase, please cite:

Qin, Helen. (2026). *GroupSHAP-GWAS* [Computer software]. GitHub. https://github.com/pinkdragon09/groupshap_gwas

```bibtex
@software{qin2026groupshapgwas,
  author = {Qin, Helen},
  title = {GroupSHAP-GWAS},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/pinkdragon09/groupshap_gwas}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License

The code, documentation and project-generated result files in this repository are
released under the MIT License (see `LICENSE`). Third-party datasets are not
redistributed; the inputs fetched by `scripts/download_data.sh` remain under their
providers' terms, summarized in `data/README.md`.
