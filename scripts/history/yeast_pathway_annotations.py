#!/usr/bin/env python3
"""Freeze independent SGD pathway and GO Slim annotations for interpretation.

This does not refit a model, alter SNP groups, or test pathway enrichment.
GO Slim biological processes are broad categories, not biochemical pathways.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data/yeast-deletion-reference/pathways"
BASE_URL = "https://downloads.yeastgenome.org/curation/literature/"
FILES = ("biochemical_pathways.tab", "go_slim_mapping.tab")
FIELDS = ("gene_id", "pathway_id", "pathway_name", "annotation_type", "source_url")
GENERIC_IDS = {"GO:0008150", "GO:0000004"}
GENERIC_LABELS = {"biological process", "biological_process", "other", "not yet annotated", "not_yet_annotated"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tab_rows(text: str, width: int):
    for row_number, row in enumerate(csv.reader(io.StringIO(text), delimiter="\t"), 1):
        if not row or row[0].startswith(("#", "!")):
            continue
        if len(row) != width:
            raise ValueError(f"Row {row_number}: expected {width} columns, found {len(row)}")
        yield row


def parse_go_slim(text: str, source_url: str):
    """Return process rows, symbol-to-ORF candidates, and transparent exclusions."""
    records = set()
    symbols = defaultdict(set)
    excluded = Counter()
    for gene_id, name, sgd_id, aspect, label, go_id, feature_type in tab_rows(text, 7):
        if feature_type.split("|")[0] != "ORF":
            excluded["non_ORF_features"] += 1
            continue
        symbols[gene_id].add(gene_id)
        if name:
            symbols[name].add(gene_id)
        if aspect != "P":
            excluded["other_GO_aspects"] += 1
            continue
        if go_id in GENERIC_IDS or label.strip().lower() in GENERIC_LABELS:
            excluded["generic_process_annotations"] += 1
            continue
        if not go_id.startswith("GO:"):
            raise ValueError(f"Invalid GO identifier: {go_id!r}")
        records.add((gene_id, go_id, label, "go_biological_process", source_url))
    return records, dict(symbols), dict(excluded)


def local_pathway_id(name: str) -> str:
    """The SGD TAB has names, not canonical pathway IDs: label our local IDs."""
    return "SGD_METABOLIC_NAME_SHA256:" + hashlib.sha256(name.encode()).hexdigest()[:16]


def parse_metabolic(text: str, symbols: dict[str, set[str]], source_url: str):
    records = set()
    unresolved = Counter()
    for pathway, enzyme, ec_number, name, references in tab_rows(text, 5):
        matches = symbols.get(name, set())
        if len(matches) != 1:
            unresolved[name] += 1
            continue
        gene_id = next(iter(matches))
        records.add((gene_id, local_pathway_id(pathway), pathway, "metabolic_pathway", source_url))
    return records, dict(sorted(unresolved.items()))


def acquire(raw: Path, refresh: bool = False):
    raw.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = raw / filename
        if target.exists() and not refresh:
            continue
        url = BASE_URL + filename
        with urlopen(url, timeout=45) as response:
            data = response.read()
            headers = dict(response.headers)
            resolved_url = response.geturl()
        # Validate before replacing a cached source (reject HTML error pages).
        list(tab_rows(data.decode("utf-8"), 5 if filename.startswith("biochemical") else 7))
        target.write_bytes(data)
        (raw / (filename + ".headers.json")).write_text(json.dumps(headers, indent=2) + "\n")
        (raw / (filename + ".retrieval.json")).write_text(json.dumps({
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": url, "resolved_url": resolved_url,
        }, indent=2) + "\n")


def candidate_coverage(records, candidate_path: Path):
    if not candidate_path.exists():
        return {"available": False}
    genes = set()
    pairs = set()
    with candidate_path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["mapping_radius_bp"] != "0":
                continue
            for pair in json.loads(row["supported_gene_pairs_stringent"]):
                genes.update(pair)
                pairs.add((row["trait"], *sorted(pair)))
    by_type = defaultdict(set)
    for gene_id, pathway_id, label, kind, source in records:
        by_type[kind].add(gene_id)
    covered = set().union(*by_type.values())
    return {"available": True, "scope": "direct-overlap stringent top100 supported candidates",
            "gene_count": len(genes), "trait_gene_pair_count": len(pairs),
            "any_annotation_gene_count": len(genes & covered),
            "missing_all_annotations": sorted(genes - covered),
            "by_type": {kind: {"covered_genes": sorted(genes & ids),
                               "covered_gene_count": len(genes & ids)}
                        for kind, ids in sorted(by_type.items())}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download", action="store_true", help="Download missing public official sources")
    parser.add_argument("--refresh", action="store_true", help="Replace cached sources explicitly")
    args = parser.parse_args(argv)
    raw = args.output / "raw"
    if args.download or args.refresh:
        acquire(raw, args.refresh)
    go_records, symbols, go_exclusions = parse_go_slim(
        (raw / FILES[1]).read_text(), BASE_URL + FILES[1])
    metabolic, unresolved = parse_metabolic(
        (raw / FILES[0]).read_text(), symbols, BASE_URL + FILES[0])
    records = sorted(go_records | metabolic)
    processed = args.output / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    target = processed / "gene_pathways.tsv"
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(FIELDS)
        writer.writerows(records)
    sources = []
    for filename in FILES:
        headers_path = raw / (filename + ".headers.json")
        headers = json.loads(headers_path.read_text()) if headers_path.exists() else {}
        headers_lower = {key.lower(): value for key, value in headers.items()}
        retrieval_path = raw / (filename + ".retrieval.json")
        retrieval = json.loads(retrieval_path.read_text()) if retrieval_path.exists() else {
            "retrieved_utc": headers_lower.get("date"),
            "retrieval_date_basis": "server HTTP Date captured with initial download",
        }
        sources.append({"filename": filename, "source_url": BASE_URL + filename,
                        "sha256": sha256(raw / filename), "bytes": (raw / filename).stat().st_size,
                        "last_modified": headers_lower.get("last-modified"),
                        "s3_version_id": headers_lower.get("x-amz-version-id"),
                        **retrieval})
    counts = Counter(row[3] for row in records)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "interpretation_scope": "Current 2026 SGD annotations used for post hoc functional interpretation, not a reconstruction of annotations available in 2015/2016; no model fitting or new grouping.",
        "independence_limit": "Groups were not inferred from Bloom or Costanzo interactions. Curated GO annotations can contain later experimental literature, so functional annotations themselves are not an independent validation test.",
        "pathway_id_note": "GO identifiers are canonical. The metabolic TAB contains pathway names only; SGD_METABOLIC_NAME_SHA256 identifiers are stable local hashes of the exact names, not SGD canonical pathway IDs.",
        "annotation_type_definitions": {"metabolic_pathway": "Curated SGD biochemical pathway membership (only metabolic enzymes/genes)", "go_biological_process": "Broad SGD GO Slim biological-process categories; not biochemical pathways"},
        "excluded_generic_go_ids": sorted(GENERIC_IDS),
        "excluded_generic_go_labels": sorted(GENERIC_LABELS), "go_exclusion_counts": go_exclusions,
        "unresolved_metabolic_symbols": unresolved,
        "annotation_counts": dict(counts),
        "annotated_gene_counts": {kind: len({row[0] for row in records if row[3] == kind}) for kind in counts},
        "term_counts": {kind: len({row[1] for row in records if row[3] == kind}) for kind in counts},
        "candidate_coverage": candidate_coverage(records, ROOT / "outputs/yeast_bloom_deletion_validation/supported_top100_candidates.tsv"),
        "output": {"path": str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target), "sha256": sha256(target), "rows": len(records)},
        "software": {"script": "scripts/yeast_pathway_annotations.py", "script_sha256": sha256(Path(__file__))},
        "documentation_sources": ["https://pathway.yeastgenome.org/", "https://wiki.yeastgenome.org/index.php/What%27s_New_in_SGD_in_2007"],
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(records), "counts": dict(counts), "unresolved_symbols": len(unresolved), "candidate_coverage": manifest["candidate_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
