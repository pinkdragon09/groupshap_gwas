"""Map Bloom SacCer3 loci to candidate ORFs without interaction evidence.

All coordinates are 1-based inclusive. Gene overlap and fixed windows identify
candidates, never causal genes. Published additive intervals are kept separate
from interaction peaks, including when Supp4 names a closest additive QTL.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION = ROOT / "data/yeast-deletion-reference/annotation"
GFF_NAME = "Saccharomyces_cerevisiae.R64-1-1.99.gff3.gz"
GFF_URL = "https://ftp.ensembl.org/pub/release-99/gff3/saccharomyces_cerevisiae/" + GFF_NAME
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI")
WINDOWS = (0, 1000, 5000, 10000, 50000)
GENE_FIELDS = ("gene_id", "gene_name", "chrom", "start", "end", "strand", "feature_type", "status")


@lru_cache(maxsize=128)
def normalize_chrom(chrom):
    """Return chr01..chr16; reject mitochondrial/unknown chromosomes."""
    value = re.sub(r"^(?:chr|chromosome)", "", str(chrom).strip(), flags=re.I).upper()
    if value in ROMAN:
        number = ROMAN.index(value) + 1
    elif value.isdigit():
        number = int(value)
    else:
        raise ValueError(f"Unknown nuclear yeast chromosome: {chrom!r}")
    if not 1 <= number <= 16:
        raise ValueError(f"Unknown nuclear yeast chromosome: {chrom!r}")
    return f"chr{number:02d}"


def _records(values):
    return values.to_dict("records") if hasattr(values, "to_dict") else list(values)


def map_interval(chrom, start, end, genes):
    """Return every overlapping gene; preserve ambiguity and stable ordering."""
    chrom, start, end = normalize_chrom(chrom), int(start), int(end)
    if start < 1 or end < start:
        raise ValueError("Expected a valid 1-based inclusive interval")
    matches = [dict(g) for g in _records(genes)
               if normalize_chrom(g["chrom"]) == chrom and int(g["start"]) <= end and int(g["end"]) >= start]
    return sorted(matches, key=lambda g: (int(g["start"]), int(g["end"]), g["gene_id"]))


def map_locus(chrom, pos, radius_bp, genes):
    """Return all ORFs overlapping [pos-radius,pos+radius], clipped at 1."""
    pos, radius_bp = int(pos), int(radius_bp)
    if pos < 1 or radius_bp < 0:
        raise ValueError("Positive position and nonnegative window required")
    return map_interval(chrom, max(1, pos-radius_bp), pos+radius_bp, genes)


def nearest_genes(chrom, pos, genes):
    """Separate sensitivity: all distance-tied nearest genes on this chromosome."""
    chrom, pos = normalize_chrom(chrom), int(pos)
    if pos < 1:
        raise ValueError("Position must be 1-based")
    local = [dict(g) for g in _records(genes) if normalize_chrom(g["chrom"]) == chrom]
    if not local:
        return []
    for g in local:
        g["distance_bp"] = max(int(g["start"])-pos, pos-int(g["end"]), 0)
    distance = min(g["distance_bp"] for g in local)
    return sorted([g for g in local if g["distance_bp"] == distance], key=lambda g: (int(g["start"]), g["gene_id"]))


def load_genes(path=ANNOTATION / "processed/genes.tsv", include_dubious=False):
    with Path(path).open() as handle:
        genes = list(csv.DictReader(handle, delimiter="\t"))
    for g in genes:
        g["start"], g["end"] = int(g["start"]), int(g["end"])
    return [g for g in genes if include_dubious or g["status"] != "dubious"]


def load_additive_intervals(path=ROOT / "data/yeast-byxrm/processed/Bloom2015_SuppData3.csv"):
    with Path(path).open() as handle:
        next(handle)  # Published title row; the second row contains column names.
        return list(csv.DictReader(handle))


def _interval_record(row, endpoint_chrom, endpoint_pos, method):
    left, peak, right = (int(row[k]) for k in ("pos.left_refineqtl", "pos.peak_refineqtl", "pos.right_refineqtl"))
    if left < 1 or not left <= peak <= right:
        raise ValueError("Malformed published additive QTL interval")
    chrom = normalize_chrom(row["chr"])
    return dict(additive_index=int(row["index"]), additive_chrom=chrom,
                original_additive_position=int(row["pos"]), interval_start=left,
                refined_peak=peak, interval_end=right,
                candidate_gene_ids=sorted(set(str(row.get("systematic name", "")).split())),
                mapping_method=method, same_chromosome=chrom == normalize_chrom(endpoint_chrom),
                endpoint_in_interval=chrom == normalize_chrom(endpoint_chrom) and left <= int(endpoint_pos) <= right)


def match_published_interval(trait, chrom, pos, additive_df):
    """Exact original additive position match only; return all matches (not nearest)."""
    chrom, pos = normalize_chrom(chrom), int(pos)
    return [_interval_record(r, chrom, pos, "exact_original_additive_position")
            for r in _records(additive_df)
            if r["trait"] == trait and normalize_chrom(r["chr"]) == chrom and int(r["pos"]) == pos]


def linked_additive_interval(trait, chrom, pos, additive_index, additive_df):
    """Follow the published closest.additive index; do not assume localization.

    Supp4 also assigns these indices to nonadditive endpoints and sometimes to
    another chromosome. Consumers must inspect same_chromosome and containment.
    This is a separate sensitivity annotation, never a primary interaction CI.
    """
    return [_interval_record(r, chrom, pos, "published_closest_additive_index")
            for r in _records(additive_df)
            if r["trait"] == trait and int(r["index"]) == int(additive_index)]


def parse_gff(path):
    """Retain nuclear protein-coding gene spans; separate explicit dubious ORFs."""
    genes, headers, excluded = [], [], Counter()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                headers.append(line.strip())
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attributes = fields
            if feature not in {"gene", "ncRNA_gene", "pseudogene", "transposable_element_gene"}:
                continue
            attrs = {k: unquote(v) for k, v in (part.split("=", 1) for part in attributes.split(";") if "=" in part)}
            if feature != "gene" or attrs.get("biotype") != "protein_coding":
                excluded[feature] += 1
                continue
            try:
                chrom = normalize_chrom(chrom)
            except ValueError:
                excluded["non_nuclear_gene"] += 1
                continue
            gene_id = attrs["gene_id"]
            genes.append(dict(gene_id=gene_id, gene_name=attrs.get("Name", ""), chrom=chrom,
                              start=int(start), end=int(end), strand=strand, feature_type="protein_coding_gene",
                              status="dubious" if "dubious open reading frame" in attrs.get("description", "").lower() else "not_flagged"))
    if "#!genome-version R64-1-1" not in headers:
        raise ValueError("Annotation must declare the Bloom-compatible R64-1-1 assembly")
    if len({g["gene_id"] for g in genes}) != len(genes):
        raise ValueError("Duplicate gene IDs in GFF")
    return sorted(genes, key=lambda g: (g["chrom"], g["start"], g["gene_id"])), headers, dict(excluded)


def _write_table(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_annotation(annotation_dir=ANNOTATION):
    annotation_dir = Path(annotation_dir)
    raw = annotation_dir / "raw" / GFF_NAME
    genes, headers, excluded = parse_gff(raw)
    _write_table(annotation_dir / "processed/genes.tsv", genes, GENE_FIELDS)
    eligible = [g for g in genes if g["status"] != "dubious"]
    additive = load_additive_intervals()
    with (ROOT / "data/yeast-byxrm/processed/Bloom2015_SuppData4_QTLxQTL_interactions.csv").open() as handle:
        interactions = list(csv.DictReader(handle))
    mapping, interval_links = [], []
    for ref_index, row in enumerate(interactions):
        for side in (1, 2):
            chrom, pos = normalize_chrom(row[f"Q{side}_chr"]), int(row[f"Q{side}_pos"])
            base = dict(reference_index=ref_index, trait=row["trait"], side=side, chrom=chrom, position=pos)
            for radius in WINDOWS:
                candidates = map_locus(chrom, pos, radius, eligible)
                mapping.append(dict(**base, mapping_method="exact_gene_overlap" if radius == 0 else "fixed_window_sensitivity",
                                    radius_bp=radius, candidate_gene_ids=";".join(g["gene_id"] for g in candidates), candidate_count=len(candidates)))
            candidates = nearest_genes(chrom, pos, eligible)
            mapping.append(dict(**base, mapping_method="nearest_gene_sensitivity", radius_bp="",
                                candidate_gene_ids=";".join(g["gene_id"] for g in candidates), candidate_count=len(candidates)))
            for method, matches in (("exact", match_published_interval(row["trait"], chrom, pos, additive)),
                                    ("linked", linked_additive_interval(row["trait"], chrom, pos, row[f"closest.additive.Q{side}"], additive))):
                for match in matches:
                    interval_links.append({**base, **match, "candidate_gene_ids": ";".join(match["candidate_gene_ids"])})
    _write_table(annotation_dir / "processed/bloom_reference_locus_genes.tsv", mapping, tuple(mapping[0]))
    _write_table(annotation_dir / "processed/bloom_published_additive_interval_links.tsv", interval_links, tuple(interval_links[0]))
    declared = {g for r in additive for g in r["systematic name"].split()}
    ids = {g["gene_id"] for g in genes}
    summary = dict(source_url=GFF_URL, sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), bytes=raw.stat().st_size,
                   processed_utc=datetime.now(timezone.utc).isoformat(), assembly="R64-1-1", assembly_accession="GCA_000146045.2",
                   ensembl_release=99, annotation_updated="2018-10", coordinate_system="1-based inclusive; no liftover",
                   build_headers=[h for h in headers if h.startswith("#!")], gene_count=len(genes), primary_gene_count=len(eligible),
                   status_counts=dict(Counter(g["status"] for g in genes)), excluded_features=excluded,
                   bloom_reference_interactions=len(interactions), mapping_windows_bp=list(WINDOWS),
                   exact_additive_position_matches=sum(r["mapping_method"] == "exact_original_additive_position" for r in interval_links),
                   linked_additive_index_matches=sum(r["mapping_method"] == "published_closest_additive_index" for r in interval_links),
                   linked_same_chromosome=sum(r["mapping_method"] == "published_closest_additive_index" and r["same_chromosome"] for r in interval_links),
                   linked_containing_endpoint=sum(r["mapping_method"] == "published_closest_additive_index" and r["endpoint_in_interval"] for r in interval_links),
                   published_gene_ids_absent_from_annotation=sorted(declared-ids),
                   input_sha256={name: hashlib.sha256((ROOT / "data/yeast-byxrm/processed" / name).read_bytes()).hexdigest()
                                 for name in ("Bloom2015_SuppData3.csv", "Bloom2015_SuppData4_QTLxQTL_interactions.csv")},
                   method_sources={"bloom_saccer3": "https://www.nature.com/articles/ncomms9712",
                                   "sgd_import_and_assembly": "https://www.ensembl.org/Saccharomyces_cerevisiae/Info/Annotation"},
                   limitations=["Gene candidates are not causal assignments.", "Fixed windows and nearest genes are sensitivities, not confidence intervals.",
                                "Additive confidence intervals do not establish interaction confidence intervals.",
                                "Status not_flagged does not establish verified or uncharacterized SGD classification.",
                                "Annotation is release99 on matching R64-1-1 sequence coordinates, not annotation contemporaneous with Bloom2015."])
    (annotation_dir / "provenance.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, default=ANNOTATION)
    args = parser.parse_args()
    print(json.dumps(build_annotation(args.annotation_dir), indent=2))
