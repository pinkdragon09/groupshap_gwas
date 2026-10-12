#!/usr/bin/env python3
"""Freeze the public Costanzo 2016 direct deletion/deletion interaction reference.

Pair calls are unions of calls on individual observed assays. They are NOT
computed by combining the smallest p-value with the largest absolute score,
and are not multiple-assay-adjusted gene-pair hypothesis tests.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/yeast-deletion-reference"
SOURCE_URL = (
    "https://thecellmap.org/costanzo2016/data_files/"
    "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
)
SOURCE_SHA256 = "05adf2aa309e5336cd1c3045eb033a2bc7b24e6f249d57a0cc25549ec6772522"
THRESHOLD_SOURCE = "https://boonelab.ccbr.utoronto.ca/supplement/costanzo2016/"
HEADER = [
    "Query Strain ID", "Query allele name", "Array Strain ID", "Array allele name",
    "Arraytype/Temp", "Genetic interaction score (ε)", "P-value",
    "Query single mutant fitness (SMF)", "Array SMF", "Double mutant fitness",
    "Double mutant fitness standard deviation",
]
PAIR_FIELDS = [
    "gene_a", "gene_b", "n_assays", "min_p", "max_abs_epsilon", "lenient",
    "intermediate", "stringent", "negative_stringent", "positive_stringent",
    "deletion_only", "n_valid_assays", "n_temperatures", "temperatures",
    "n_directions", "negative_intermediate", "positive_intermediate",
    "stringent_sign_conflict",
]
ASSAY_FIELDS = [
    "gene_a", "gene_b", "query_gene", "array_gene", "query_strain_id",
    "array_strain_id", "query_allele", "array_allele", "array_type",
    "temperature_c", "epsilon", "p_value", "query_smf", "array_smf",
    "double_mutant_fitness", "double_mutant_fitness_sd", "valid_assay",
    "lenient", "intermediate", "stringent", "negative_stringent",
    "positive_stringent", "source_member", "source_line",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(value) -> float | None:
    if value is None or str(value).strip().lower() in {"", "na", "nan", "n/a", "null"}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def gene_from_strain(strain_id: str) -> str:
    gene = strain_id.split("_", 1)[0].upper()
    if not re.fullmatch(r"Y[A-P][LR]\d{3}[CW](?:-[A-Z])?", gene):
        raise ValueError(f"Unexpected yeast systematic ORF identifier: {strain_id}")
    return gene


def assay_calls(epsilon, p_value) -> dict[str, int]:
    epsilon, p_value = numeric(epsilon), numeric(p_value)
    valid = epsilon is not None and p_value is not None
    if p_value is not None and not 0 <= p_value <= 1:
        raise ValueError(f"Invalid p-value {p_value}")
    significant = valid and p_value < 0.05
    negative = significant and epsilon < -0.12
    positive = significant and epsilon > 0.16
    return {
        "valid_assay": int(valid), "lenient": int(significant),
        "intermediate": int(significant and abs(epsilon) > 0.08),
        "stringent": int(negative or positive), "negative_stringent": int(negative),
        "positive_stringent": int(positive),
        "negative_intermediate": int(significant and epsilon < -0.08),
        "positive_intermediate": int(significant and epsilon > 0.08),
    }


@dataclass(slots=True)
class PairStats:
    n_assays: int = 0
    n_valid_assays: int = 0
    min_p: float = math.inf
    max_abs_epsilon: float = -math.inf
    lenient: int = 0
    intermediate: int = 0
    stringent: int = 0
    negative_stringent: int = 0
    positive_stringent: int = 0
    negative_intermediate: int = 0
    positive_intermediate: int = 0
    temperatures_mask: int = 0
    directions_mask: int = 0


class PairAggregator:
    """Deduplicate assay identities, retaining reciprocal/temperature assays.

    Identity is directed query strain, array strain, temperature/array type.
    Repeated identical rows are ignored; conflicting rows for the same identity
    raise rather than silently selecting a score. Fingerprints cover all raw
    assay fields (SHA256). This in-memory strategy fits the available workstation.
    """

    def __init__(self):
        self.pairs: dict[tuple[str, str], PairStats] = {}
        self.seen: dict[tuple[int, int, int], bytes] = {}
        self.strain_codes: dict[str, int] = {}
        self.condition_codes: dict[str, int] = {}
        self.temperature_codes: dict[int, int] = {}
        self.genes: dict[str, str] = {}
        self.duplicate_rows = 0
        self.self_rows = 0

    def add(self, row: list[str], source_member: str, source_line: int):
        if len(row) != 11:
            raise ValueError(f"{source_member}:{source_line}: expected 11 columns")
        query, q_allele, array, a_allele, condition = row[:5]
        # This pipeline only reads SGA_NxN, whose queries and arrays are deletions.
        # Require the DMA array designation as an additional schema check.
        if not re.fullmatch(r"DMA\d+", condition):
            raise ValueError(f"Unexpected deletion/deletion assay condition {condition}")
        qgene = self.genes.setdefault(query, gene_from_strain(query))
        agene = self.genes.setdefault(array, gene_from_strain(array))
        if qgene == agene:
            self.self_rows += 1
            return None
        temperature = int(condition[3:])
        key = (
            self.strain_codes.setdefault(query, len(self.strain_codes)),
            self.strain_codes.setdefault(array, len(self.strain_codes)),
            self.condition_codes.setdefault(condition, len(self.condition_codes)),
        )
        fingerprint = hashlib.sha256("\t".join(row).encode()).digest()
        if key in self.seen:
            if self.seen[key] != fingerprint:
                raise ValueError(f"Conflicting duplicate assay {query}, {array}, {condition}")
            self.duplicate_rows += 1
            return None
        self.seen[key] = fingerprint
        a, b = sorted((qgene, agene))
        # Intern ORF strings through a dict to avoid retaining copies per pair.
        a = sys.intern(a)
        b = sys.intern(b)
        stats = self.pairs.setdefault((a, b), PairStats())
        epsilon, p_value = numeric(row[5]), numeric(row[6])
        calls = assay_calls(epsilon, p_value)
        stats.n_assays += 1
        stats.n_valid_assays += calls["valid_assay"]
        if p_value is not None:
            stats.min_p = min(stats.min_p, p_value)
        if epsilon is not None:
            stats.max_abs_epsilon = max(stats.max_abs_epsilon, abs(epsilon))
        for name in ("lenient", "intermediate", "stringent", "negative_stringent",
                     "positive_stringent", "negative_intermediate", "positive_intermediate"):
            setattr(stats, name, getattr(stats, name) | calls[name])
        temp_bit = self.temperature_codes.setdefault(temperature, len(self.temperature_codes))
        stats.temperatures_mask |= 1 << temp_bit
        stats.directions_mask |= 1 if qgene == a else 2
        return dict(zip(ASSAY_FIELDS, [
            a, b, qgene, agene, query, array, q_allele, a_allele, "DMA", temperature,
            "" if epsilon is None else epsilon, "" if p_value is None else p_value,
            *row[7:], calls["valid_assay"], calls["lenient"], calls["intermediate"],
            calls["stringent"], calls["negative_stringent"], calls["positive_stringent"],
            source_member, source_line,
        ]))

    def rows(self):
        for (a, b), stats in sorted(self.pairs.items()):
            temps = sorted(t for t, i in self.temperature_codes.items()
                           if stats.temperatures_mask & (1 << i))
            yield {
                "gene_a": a, "gene_b": b, "n_assays": stats.n_assays,
                "min_p": "" if not math.isfinite(stats.min_p) else stats.min_p,
                "max_abs_epsilon": "" if not math.isfinite(stats.max_abs_epsilon)
                    else stats.max_abs_epsilon,
                "lenient": stats.lenient, "intermediate": stats.intermediate,
                "stringent": stats.stringent, "negative_stringent": stats.negative_stringent,
                "positive_stringent": stats.positive_stringent, "deletion_only": 1,
                "n_valid_assays": stats.n_valid_assays, "n_temperatures": len(temps),
                "temperatures": ",".join(map(str, temps)),
                "n_directions": stats.directions_mask.bit_count(),
                "negative_intermediate": stats.negative_intermediate,
                "positive_intermediate": stats.positive_intermediate,
                "stringent_sign_conflict": stats.negative_stringent & stats.positive_stringent,
            }


def validate_archive(path: Path, expected_sha256: str = SOURCE_SHA256) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Archive is incomplete or invalid: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"Archive does not match frozen May 2016 snapshot: {observed}")


def download(data_dir: Path) -> Path:
    path = data_dir / "raw/costanzo2016_pairwise.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        validate_archive(path)
        return path
    temporary = path.with_suffix(".zip.partial")
    print(f"Downloading {SOURCE_URL}", flush=True)
    urllib.request.urlretrieve(SOURCE_URL, temporary)
    validate_archive(temporary)
    os.replace(temporary, path)
    return path


def normalize(archive: Path, data_dir: Path) -> dict:
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pairs_path = processed / "costanzo2016_pairs.tsv"
    assays_path = processed / "costanzo2016_assays.tsv.gz"
    pairs_tmp = pairs_path.with_suffix(".tsv.partial")
    assays_tmp = assays_path.with_suffix(".gz.partial")
    aggregation = PairAggregator()
    counts = Counter()
    source_digest = sha256_file(archive)
    with zipfile.ZipFile(archive) as package:
        members = [name for name in package.namelist() if name.endswith("/SGA_NxN.txt")]
        if len(members) != 1:
            raise ValueError("Expected exactly one SGA_NxN.txt deletion network")
        member = members[0]
        member_hash = hashlib.sha256()
        # Stream raw bytes once to both validate member checksum and parse rows.
        with assays_tmp.open("wb") as raw_out, gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_out, mtime=0, compresslevel=3) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=ASSAY_FIELDS, delimiter="\t")
                writer.writeheader()
                with package.open(member) as input_file:
                    first = input_file.readline()
                    member_hash.update(first)
                    if first.decode("utf-8").rstrip("\r\n").split("\t") != HEADER:
                        raise ValueError("Official input header differs from frozen schema")
                    for line_number, line in enumerate(input_file, start=2):
                        member_hash.update(line)
                        row = line.decode("utf-8").rstrip("\r\n").split("\t")
                        counts["source_rows"] += 1
                        normalized = aggregation.add(row, member, line_number)
                        if normalized is not None:
                            writer.writerow(normalized)
                            counts["unique_assays"] += 1
                            counts["valid_assays"] += normalized["valid_assay"]
                            counts[f"temperature_{normalized['temperature_c']}"] += 1
                        if counts["source_rows"] % 1_000_000 == 0:
                            print(f"Parsed {counts['source_rows']:,} rows; "
                                  f"{len(aggregation.pairs):,} unordered pairs", flush=True)
        with pairs_tmp.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=PAIR_FIELDS, delimiter="\t")
            writer.writeheader()
            for row in aggregation.rows():
                writer.writerow(row)
                counts["unordered_pairs"] += 1
                counts["valid_unordered_pairs"] += int(row["n_valid_assays"] > 0)
                for name in ("lenient", "intermediate", "stringent", "negative_stringent",
                             "positive_stringent", "stringent_sign_conflict"):
                    counts[f"pairs_{name}"] += row[name]
                counts[f"pairs_n_assays_{row['n_assays']}"] += 1
        members_info = [{"member": item.filename, "bytes": item.file_size,
                         "zip_crc32": f"{item.CRC:08x}"} for item in package.infolist()]
    os.replace(assays_tmp, assays_path)
    os.replace(pairs_tmp, pairs_path)
    counts["identical_duplicate_rows"] = aggregation.duplicate_rows
    counts["excluded_self_gene_rows"] = aggregation.self_rows
    manifest = {
        "dataset": "Costanzo2016 complete direct pair-wise SGA reference, NxN subset",
        "source_url": SOURCE_URL, "landing_page": "https://thecellmap.org/yeast/costanzo2016/",
        "source_snapshot": "May 2016 complete dataset",
        "retrieved_utc": datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).isoformat(),
        "normalized_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {"path": str(archive.relative_to(data_dir)), "bytes": archive.stat().st_size,
                    "sha256": source_digest, "members": members_info},
        "selected_member": member, "selected_member_sha256": member_hash.hexdigest(),
        "deletion_only": True,
        "filter": "SGA_NxN.txt only, require DMA array type; exclude same-ORF self rows",
        "thresholds": {"lenient": "p < 0.05", "intermediate": "p < 0.05 and |epsilon| > 0.08",
                       "stringent": "p < 0.05 and (epsilon < -0.12 or epsilon > 0.16)",
                       "source": THRESHOLD_SOURCE,
                       "clarification": "Stringent inequality on 2016 HTML is a typographical "
                       "error; original Costanzo2010 Supplement Table S6 specifies the two tails.",
                       "original_source": "https://boonelab.ccbr.utoronto.ca/supplement/costanzo2009/"
                       "Costanzo%20et%20al_supplement.pdf"},
        "pair_aggregation": "Any individual valid directed strain/temperature assay passes. "
            "Reciprocal directions and temperatures retained; identical duplicate rows removed; "
            "conflicting same-identity rows rejected. Pair flags are descriptive unions, not "
            "multiplicity-adjusted hypothesis tests. min_p and max_abs_epsilon are independent "
            "descriptive summaries and never combined to produce a call.",
        "coverage_rule": "n_valid_assays > 0 defines usable screened coverage; missing score/p "
            "is preserved as an assay but cannot produce a call. Absent pairs are untested, "
            "and tested non-significant pairs are not confirmed biological negatives.",
        "counts": dict(sorted(counts.items())),
        "outputs": [{"path": str(path.relative_to(data_dir)), "bytes": path.stat().st_size,
                     "sha256": sha256_file(path)} for path in (pairs_path, assays_path)],
        "software": {"python": sys.version, "script": "scripts/yeast_deletion_reference.py",
                     "script_sha256": sha256_file(Path(__file__))},
    }
    (data_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--archive", type=Path, help="Already downloaded official ZIP")
    args = parser.parse_args()
    archive = args.archive or download(args.data_dir)
    if args.archive is not None:
        validate_archive(archive)
    if not args.download_only:
        normalize(archive, args.data_dir)


if __name__ == "__main__":
    main()
