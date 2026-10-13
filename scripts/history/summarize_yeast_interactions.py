#!/usr/bin/env python3
"""Recompute canonical reference mapping and summarize completed yeast rankings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fivefold_common import digest, fingerprint, write_json
from yeast_fivefold_common import TRAITS, load_interactions_2015


def roman_to_int(text):
    values={'I':1,'V':5,'X':10,'L':50,'C':100}; total=prev=0
    for char in reversed(text.upper()):
        value=values[char]; total += -value if value < prev else value; prev=value
    return total


def chromosome(value):
    text=str(value).strip()
    if text.lower().startswith('chr'): text=text[3:]
    number=int(text) if text.isdigit() else roman_to_int(text)
    if not 1 <= number <= 16: raise ValueError(f'Invalid yeast chromosome: {value}')
    return number


def map_refs(trait,selection,distance=50000):
    refs=load_interactions_2015(); refs=refs[refs.trait==trait].reset_index(drop=True); rows=[]
    for ri,row in refs.iterrows():
        mapped=[]
        for which in [1,2]:
            chrom=chromosome(row[f'Q{which}_chr']); pos=int(row[f'Q{which}_pos'])
            candidates=[(i,abs(int(feature['position'])-pos)) for i,feature in enumerate(selection)
                        if chromosome(feature['chromosome'])==chrom]
            mapped.append(min(candidates,key=lambda item:(item[1],item[0])) if candidates else (None,None))
        eligible=all(i is not None and d<=distance for i,d in mapped) and mapped[0][0]!=mapped[1][0]
        rows.append(dict(reference_index=ri,q1_panel_index=mapped[0][0],q1_distance_bp=mapped[0][1],
                         q2_panel_index=mapped[1][0],q2_distance_bp=mapped[1][1],eligible=eligible))
    return rows


def metrics(trait,fold,frames,selection,refs):
    eligible=[r for r in refs if r['eligible']]
    mapped=[tuple(sorted((int(r['q1_panel_index']),int(r['q2_panel_index'])))) for r in eligible]
    unique=set(mapped); total=len(refs); neglog=[r['neg_log10_p'] for r in selection]
    pairs=[dict(j_idx=j,k_idx=k,score=neglog[j]+neglog[k]) for j in range(len(selection)) for k in range(j+1,len(selection))]
    frames=dict(frames,marginal_product=pd.DataFrame(pairs).sort_values(['score','j_idx','k_idx'],ascending=[False,True,True]))
    rows=[]
    for method,frame in frames.items():
        for k in [10,25,50,100]:
            top={tuple(sorted((int(r.j_idx),int(r.k_idx)))) for r in frame.head(k).itertuples()}
            tp=len(top&unique); recovered=sum(pair in top for pair in mapped)
            rows.append(dict(trait=trait,fold=fold,method=method,k=k,total_reference_rows=total,
                eligible_reference_rows=len(eligible),unique_eligible_pairs=len(unique),collisions=len(mapped)-len(unique),
                true_positive_unique_pairs=tp,conditional_recall=tp/len(unique) if unique else None,
                recovered_reference_rows=recovered,unconditional_recovery=recovered/total))
    return rows


def run(out):
    rows=[]; mappings={}; model_trait_folds=0
    for trait in TRAITS:
        td=out/trait; selected=json.loads((td/'selected_features.json').read_text())
        for fold in range(1,6):
            inter=td/f'fold_{fold}/interactions'; rf=inter/'rf.csv'; kan=inter/'kan.csv'
            refs=map_refs(trait,selected[f'outer{fold}']); mappings[f'{trait}_fold_{fold}']=refs
            frames={}
            if rf.exists(): frames['rf']=pd.read_csv(rf)
            if kan.exists(): frames['kan']=pd.read_csv(kan)
            if frames: model_trait_folds += 1
            rows.extend(metrics(trait,fold,frames,selected[f'outer{fold}'],refs))
    identity=dict(source_sha256=digest(Path(__file__)),prediction_fingerprint=json.loads((out/'manifest.json').read_text())['fingerprint'],
                  mapping_distance_bp=50000)
    write_json(out/'interaction_reference_mapping_canonical.json',mappings)
    write_json(out/'interaction_recall_by_fold_canonical.json',rows)
    frame=pd.DataFrame(rows); summary=[]
    for (trait,method,k),group in frame.groupby(['trait','method','k'],sort=True):
        summary.append(dict(trait=trait,method=method,k=int(k),folds=int(len(group)),
            mean_conditional_recall=float(group.conditional_recall.mean()),
            sd_conditional_recall=float(group.conditional_recall.std(ddof=1)) if len(group)>1 else None,
            micro_conditional_recall=float(group.true_positive_unique_pairs.sum()/group.unique_eligible_pairs.sum()),
            mean_unconditional_recovery=float(group.unconditional_recovery.mean()),
            eligible_rows_by_fold=group.eligible_reference_rows.astype(int).tolist()))
    write_json(out/'interaction_recall_summary.json',summary)
    write_json(out/'interaction_summary_manifest.json',dict(**identity,fingerprint=fingerprint(identity),
        baseline_trait_folds=10,model_trait_folds=model_trait_folds))
    print(f'Canonical interaction summaries: {len(rows)} rows; baseline complete for 10/10 trait-folds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--outdir',type=Path,required=True)
    run(p.parse_args().outdir.resolve())
