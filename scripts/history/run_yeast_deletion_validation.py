#!/usr/bin/env python3
"""Annotate frozen Bloom rankings with independent direct deletion-screen evidence."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_yeast_interactions import chromosome, map_refs
from yeast_deletion_evaluation import bh_adjust, evaluate_annotations

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def candidates(genes, chrom, start, end):
    """Closed one-based interval overlap; no nearest-gene causal assignment."""
    sub = genes[genes.chrom.map(chromosome) == chromosome(chrom)]
    return sorted(sub.loc[(sub.start <= end) & (sub.end >= start), 'gene_id'].unique())


def gene_pairs(left, right):
    return sorted({tuple(sorted((a, b))) for a in left for b in right if a != b})


def read_deletions(path, relevant):
    lookup = {}; degrees = Counter(); assayed_degrees = Counter(); counts = Counter()
    boolean_columns = ['lenient', 'intermediate', 'stringent']
    for chunk in pd.read_csv(path, sep='\t', chunksize=200000):
        for col in boolean_columns:
            if chunk[col].dtype != bool:
                chunk[col] = chunk[col].astype(str).str.lower().isin(['true', '1'])
        valid = chunk.n_valid_assays > 0
        counts['all_pairs'] += len(chunk); counts['valid_pairs'] += int(valid.sum())
        for col in ['gene_a', 'gene_b']:
            assayed_degrees.update(chunk.loc[valid, col].value_counts().to_dict())
            degrees.update(chunk.loc[valid & chunk.stringent, col].value_counts().to_dict())
        sub = chunk[valid & chunk.gene_a.isin(relevant) & chunk.gene_b.isin(relevant)]
        for row in sub.itertuples(index=False):
            key = tuple(sorted((row.gene_a, row.gene_b)))
            if key in lookup:
                raise ValueError(f'Duplicate normalized deletion gene pair: {key}')
            lookup[key] = dict(lenient=bool(row.lenient), intermediate=bool(row.intermediate),
                stringent=bool(row.stringent), n_assays=int(row.n_assays),
                n_valid_assays=int(row.n_valid_assays),
                min_p=float(row.min_p), max_abs_epsilon=float(row.max_abs_epsilon))
    return lookup, degrees, assayed_degrees, dict(counts)


def evidence(left, right, lookup, degrees, supported_degrees=None):
    pairs = gene_pairs(left, right)
    assayed = [pair for pair in pairs if pair in lookup]
    result = dict(genes_a=json.dumps(left), genes_b=json.dumps(right),
        n_candidate_gene_pairs=len(pairs), n_screened_gene_pairs=len(assayed),
        n_screened_assays=sum(lookup[p].get('n_valid_assays', 1) for p in assayed),
        screened_fraction=len(assayed)/len(pairs) if pairs else 0.0,
        degree_score=float(np.mean([degrees[g] for g in set(left + right)])) if left or right else 0.0,
        ambiguous_gene_mapping=len(left) > 1 or len(right) > 1,
        mapping_missing=not left or not right)
    if supported_degrees is not None:
        result['stringent_degree_score'] = float(np.mean([supported_degrees[g] for g in set(left + right)])) if left or right else 0.0
    for threshold in ['lenient', 'intermediate', 'stringent']:
        supported = [p for p in assayed if lookup[p][threshold]]
        result[f'supported_{threshold}'] = bool(supported)
        result[f'n_supported_{threshold}_gene_pairs'] = len(supported)
        if threshold == 'stringent':
            result['supported_gene_pairs_stringent'] = json.dumps(supported)
    return result


def frozen_rankings(source):
    records = []; paths = set(); panels = {}
    for trait in ['Neomycin', 'Lactate']:
        path = source/trait/'selected_features.json'; paths.add(path)
        selected = json.loads(path.read_text())
        for fold in range(1, 6):
            panel = selected[f'outer{fold}']; panels[trait, fold] = panel
            refs = map_refs(trait, panel)
            published = {tuple(sorted((r['q1_panel_index'], r['q2_panel_index'])))
                for r in refs if r['eligible']}
            for method in ['rf', 'kan', 'marginal_product']:
                if method == 'marginal_product':
                    frame = pd.DataFrame([dict(j_idx=j, k_idx=k,
                        score=panel[j]['neg_log10_p']+panel[k]['neg_log10_p'])
                        for j, k in itertools.combinations(range(len(panel)), 2)])
                    frame = frame.sort_values(['score', 'j_idx', 'k_idx'], ascending=[False, True, True])
                else:
                    path = source/trait/f'fold_{fold}'/'interactions'/f'{method}.csv'
                    paths.add(path); frame = pd.read_csv(path)
                    # Published files already contain the frozen rank order, including ties.
                    frame['score'] = frame.mean_abs_interaction
                seen = set()
                for rank, row in enumerate(frame.itertuples(index=False), 1):
                    j, k = sorted((int(row.j_idx), int(row.k_idx)))
                    if j == k or (j, k) in seen:
                        raise ValueError('Invalid frozen marker-pair universe')
                    seen.add((j, k))
                    a, b = panel[j], panel[k]
                    records.append(dict(trait=trait, fold=fold, method=method,
                        pair_id=f'{j}:{k}', j_idx=j, k_idx=k, rank=rank, score=float(row.score),
                        marker_a=a['marker_id'], marker_b=b['marker_id'],
                        chromosome_a=a['chromosome'], chromosome_b=b['chromosome'],
                        position_a=a['position'], position_b=b['position'],
                        same_chromosome=chromosome(a['chromosome']) == chromosome(b['chromosome']),
                        bloom_reference_match_50kb=(j, k) in published))
                if len(seen) != len(panel)*(len(panel)-1)//2:
                    raise ValueError('Incomplete frozen pair ranking')
    return pd.DataFrame(records), panels, paths


def reference_maps(reference, additive, genes, radii):
    records = []; cache = {}
    for ri, row in reference.iterrows():
        for radius in radii:
            left = candidates(genes, row.Q1_chr, int(row.Q1_pos)-radius, int(row.Q1_pos)+radius)
            right = candidates(genes, row.Q2_chr, int(row.Q2_pos)-radius, int(row.Q2_pos)+radius)
            cache[ri, str(radius)] = (left, right)
            records.append(dict(reference_row=int(ri), trait=row.trait, mapping_policy=f'window_{radius}bp',
                mapping_radius_bp=radius, q1_chr=row.Q1_chr, q1_pos=int(row.Q1_pos),
                q2_chr=row.Q2_chr, q2_pos=int(row.Q2_pos), LOD=float(row.LOD),
                endpoint_a_in_linked_interval=None, endpoint_b_in_linked_interval=None))
        linked = []; inside = []; intervals = []
        for which in [1, 2]:
            index = row[f'closest.additive.Q{which}']; chrom = chromosome(row[f'Q{which}_chr'])
            match = additive[(additive.trait == row.trait) & (additive['index'] == index)
                             & (additive.chr.map(chromosome) == chrom)]
            if len(match) > 1:
                raise ValueError('Ambiguous published additive index join')
            if match.empty or match.iloc[0][['pos.left_refineqtl', 'pos.right_refineqtl']].isna().any():
                linked.append([]); inside.append(None); intervals.append(None); continue
            a = match.iloc[0]; lo = int(a['pos.left_refineqtl']); hi = int(a['pos.right_refineqtl'])
            if hi < lo:
                raise ValueError('Reversed published QTL interval')
            # These are explicitly linked additive intervals, not confidence intervals for the interaction peaks.
            supplied = str(a.get('systematic name', '')).split()
            linked.append(sorted(set(g for g in supplied if g in set(genes.gene_id))))
            inside.append(lo <= int(row[f'Q{which}_pos']) <= hi); intervals.append([lo, hi])
        cache[ri, 'linked'] = tuple(linked)
        records.append(dict(reference_row=int(ri), trait=row.trait, mapping_policy='linked_additive_interval',
            mapping_radius_bp=-1, q1_chr=row.Q1_chr, q1_pos=int(row.Q1_pos),
            q2_chr=row.Q2_chr, q2_pos=int(row.Q2_pos), LOD=float(row.LOD),
            endpoint_a_in_linked_interval=inside[0], endpoint_b_in_linked_interval=inside[1],
            linked_interval_a=json.dumps(intervals[0]), linked_interval_b=json.dumps(intervals[1])))
    return records, cache


def run(args):
    out = args.outdir; out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists() and not args.overwrite:
        raise ValueError('Completed output exists; use a new directory or --overwrite')
    genes = pd.read_csv(args.genes, sep='\t')
    if 'status' in genes and not args.include_dubious:
        genes = genes[genes.status != 'dubious'].copy()
    reference_path = ROOT/'data/yeast-byxrm/processed/Bloom2015_SuppData4_QTLxQTL_interactions.csv'
    additive_path = ROOT/'data/yeast-byxrm/processed/Bloom2015_SuppData3.csv'
    refs = pd.read_csv(reference_path); additive = pd.read_csv(additive_path, header=1)
    frozen, panels, paths = frozen_rankings(args.source)
    paths.update([reference_path, additive_path, args.genes, args.deletions])
    before = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}
    maps = {}; relevant = set(); marker_maps = []
    for (trait, fold), panel in panels.items():
        for radius in args.radii:
            for j, feature in enumerate(panel):
                gs = candidates(genes, feature['chromosome'], feature['position']-radius, feature['position']+radius)
                maps[trait, fold, radius, j] = gs; relevant.update(gs)
                marker_maps.append(dict(trait=trait, fold=fold, mapping_radius_bp=radius,
                    marker_id=feature['marker_id'], candidate_genes=json.dumps(gs), n_candidate_genes=len(gs)))
    ref_records, ref_cache = reference_maps(refs, additive, genes, args.radii)
    for left, right in ref_cache.values():
        relevant.update(left); relevant.update(right)
    lookup, degrees, assayed_degrees, deletion_counts = read_deletions(args.deletions, relevant)
    pd.DataFrame(marker_maps).to_csv(out/'marker_gene_maps.tsv', sep='\t', index=False)
    published_rows = []
    for row in ref_records:
        key = 'linked' if row['mapping_radius_bp'] == -1 else str(row['mapping_radius_bp'])
        ev = evidence(*ref_cache[row['reference_row'], key], lookup, assayed_degrees, degrees)
        published_rows.append(dict(row, **ev, evidence_category=('both' if ev['supported_stringent'] else
            'missing_deletion_coverage' if not ev['n_screened_gene_pairs'] else 'bloom_only')))
    published = pd.DataFrame(published_rows)
    published.to_csv(out/'published_reference_annotations.tsv', sep='\t', index=False)
    all_metrics = []; all_summaries = []; eval_metadata = []; examples = []; coverage = []
    for radius in args.radii:
        rows = []
        for row in frozen.to_dict('records'):
            left = maps[row['trait'], row['fold'], radius, row['j_idx']]
            right = maps[row['trait'], row['fold'], radius, row['k_idx']]
            ev = evidence(left, right, lookup, assayed_degrees, degrees)
            category = ('both' if row['bloom_reference_match_50kb'] else 'deletion_only') if ev['supported_stringent'] else (
                'missing_deletion_coverage' if not ev['n_screened_gene_pairs'] else
                'bloom_only' if row['bloom_reference_match_50kb'] else 'neither_detected')
            rows.append(dict(row, mapping_radius_bp=radius, **ev, evidence_category=category))
        frame = pd.DataFrame(rows)
        frame.to_csv(out/f'ranked_annotations_{radius}bp.tsv.gz', sep='\t', index=False, compression='gzip')
        by_fold, summary, metadata = evaluate_annotations(frame, n_permutations=args.permutations, seed=args.seed)
        by_fold['mapping_radius_bp'] = radius
        summary['mapping_radius_bp'] = radius
        all_metrics.append(by_fold); all_summaries.append(summary); eval_metadata.append(dict(radius_bp=radius, **metadata))
        examples.append(frame[(frame['rank'] <= 100) & frame.supported_stringent])
        for (trait, fold), group in frame[frame.method == 'rf'].groupby(['trait', 'fold']):
            coverage.append(dict(trait=trait, fold=int(fold), mapping_radius_bp=radius, total_pairs=len(group),
                assayed_pairs=int((group.n_screened_gene_pairs > 0).sum()),
                mapping_missing_pairs=int(group.mapping_missing.sum()),
                ambiguous_pairs=int(group.ambiguous_gene_mapping.sum()),
                stringent_supported_pairs=int(group.supported_stringent.sum())))
    metrics = pd.concat(all_metrics, ignore_index=True); summary = pd.concat(all_summaries, ignore_index=True)
    metrics['matched_bh_q_within_radius'] = metrics.matched_bh_q
    metrics['matched_bh_q'] = bh_adjust(metrics.matched_empirical_p)
    # Summary inferential fields stay explicitly within-radius; overall adjusted values live in by-fold output.
    summary = summary.rename(columns={'matched_bh_q_by_fold': 'matched_bh_q_within_radius_by_fold'})
    metrics.to_csv(out/'support_by_fold.tsv', sep='\t', index=False)
    summary.to_csv(out/'support_summary.tsv', sep='\t', index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage.tsv', sep='\t', index=False)
    pd.concat(examples, ignore_index=True).to_csv(out/'supported_top100_candidates.tsv', sep='\t', index=False)
    pub_summary = published.groupby(['trait', 'mapping_policy'], sort=True).agg(
        reference_rows=('reference_row', 'size'), assayed_rows=('n_screened_gene_pairs', lambda x: int((x > 0).sum())),
        stringent_supported_rows=('supported_stringent', 'sum'), ambiguous_rows=('ambiguous_gene_mapping', 'sum')).reset_index()
    pub_summary.to_csv(out/'published_reference_summary.tsv', sep='\t', index=False)
    after = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}
    if before != after:
        raise RuntimeError('Input changed during analysis')
    config = dict(source=str(args.source.relative_to(ROOT)), radii_bp=args.radii,
        primary_radius_bp=0, primary_threshold='stringent', permutations=args.permutations, seed=args.seed,
        include_dubious=args.include_dubious, deletion_scope='nonessential deletion x deletion',
        mapping_coordinates='one-based closed R64-1-1 / SacCer3', reuse_frozen_ranks=True, gpu_used=False,
        interval_policy='Explicit closest.additive indexes; linked intervals are separate hypotheses, not interaction confidence intervals.',
        input_sha256=before, original_inputs_unchanged=True, deletion_counts=deletion_counts,
        relevant_genes=len(relevant), retained_valid_deletion_pairs=len(lookup),
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'scripts/yeast_deletion_evaluation.py']})
    write_json(out/'provenance.json', config); write_json(out/'evaluation_metadata.json', eval_metadata)
    lines = ['# Bloom interactions: independent deletion-screen support', '',
        'This analysis reuses frozen matched five-fold marker-pair rankings; it does not retrain models.', '',
        'Primary mapping is direct gene overlap (0 bp); larger fixed windows are sensitivity analyses. '
        'Published linked additive-QTL intervals are analyzed separately and do not establish the localization of interaction peaks.', '',
        '## Published interaction coverage and support', '', pub_summary.to_csv(index=False),
        '## Reading the ranking results', '',
        '`support_summary.tsv` reports RF, KAN and marginal-product support@K, assayed coverage and enrichment. '
        '`support_by_fold.tsv` supplies matched opportunity-null tests and exploratory multiplicity adjustment. '
        'Check coverage before comparing support rates. Original top-K ranks are retained when assays are missing.', '',
        'Fold variability is descriptive; repeated gene evidence across folds is not independent replication. '
        'The matched null controls only its recorded opportunity strata; shared genes and linkage can induce additional dependence. '
        'Deletion epsilon is not a Shapley interaction value, and signs are not compared. '
        'The deletion screen uses different growth conditions from Neomycin and Lactate. '
        'Any supported marker pair identifies a candidate functional connection, not a causal SNP pair. '
        'Screened nonsignificant and untested gene pairs are distinguished.', '',
        '## Reproduce', '', '```bash',
        '.venv-kan/bin/python scripts/yeast_deletion_reference.py --help',
        '.venv-kan/bin/python scripts/yeast_bloom_gene_mapping.py --help',
        f'.venv-kan/bin/python scripts/run_yeast_deletion_validation.py --outdir {out.relative_to(ROOT)}',
        '```', '', 'See `provenance.json` for fixed configuration, source checksums and proof original inputs stayed unchanged.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'COMPLETE.json', dict(status='complete', original_inputs_unchanged=True,
        output_sha256={p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'COMPLETE.json'}))
    print(f'Completed deletion validation: {out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'outputs/fivefold_v1/yeast_bloom_20260913_matched')
    p.add_argument('--genes', type=Path, default=ROOT/'data/yeast-deletion-reference/annotation/processed/genes.tsv')
    p.add_argument('--deletions', type=Path, default=ROOT/'data/yeast-deletion-reference/processed/costanzo2016_pairs.tsv')
    p.add_argument('--outdir', type=Path, default=ROOT/'outputs/yeast_bloom_deletion_validation')
    p.add_argument('--radii', nargs='+', type=int, default=[0, 1000, 5000, 10000])
    p.add_argument('--permutations', type=int, default=2000); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--include-dubious', action='store_true'); p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    for name in ['source', 'genes', 'deletions', 'outdir']:
        setattr(args, name, getattr(args, name).resolve())
    if 0 not in args.radii or any(r < 0 for r in args.radii):
        p.error('Include primary radius 0; all radii must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
