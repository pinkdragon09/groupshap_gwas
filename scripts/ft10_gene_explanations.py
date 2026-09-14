"""FT10 annotation partition and balanced-donor permutation-path Shapley."""
import gzip
import re

import numpy as np


def load_genes(path):
    records = []
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.startswith('#'):
                continue
            row = line.rstrip('\n').split('\t')
            if len(row) != 9 or row[2] != 'gene':
                continue
            match = re.search(r'(?:^|;)ID=([^;]+)', row[8])
            if match and row[0].lower() in ['chr1','chr2','chr3','chr4','chr5']:
                records.append(dict(gene=match[1].split('.')[0], chromosome=row[0].lower(),
                                    start=int(row[3]), end=int(row[4])))
    if not records:
        raise ValueError('No nuclear gene annotations loaded')
    return records


def assign_genes(features, genes, window=25000):
    """Disjoint nearest-centre partition, preserving GFF order for distance ties."""
    labels = []
    for feature in features:
        pos, chrom = feature['position'], feature['chromosome']
        candidates = [g for g in genes if g['chromosome']==chrom and
                      g['start']-window <= pos <= g['end']+window]
        if candidates:
            gene = min(candidates, key=lambda g:abs((g['start']+g['end'])//2-pos))['gene']
        else:
            gene = f'intergenic_{chrom}_{pos}'
        labels.append(gene)
    names = sorted(set(labels)) + ['PCs']
    groups = [np.flatnonzero(np.array(labels)==name) for name in names[:-1]]
    groups.append(np.arange(len(features),len(features)+3))
    return names, groups, labels


def canonical_coverage(features, genes, assignments):
    result = []
    for symbol, gene in [('FLC','AT5G10140'),('FRI','AT4G00650'),('FT','AT1G65480'),
                         ('VIN3','AT5G57380'),('DOG1','AT5G45830')]:
        match = [g for g in genes if g['gene']==gene]
        if len(match)!=1:
            raise ValueError('Canonical annotation missing or duplicated: '+gene)
        g = match[0]
        indices = [i for i, f in enumerate(features) if f['chromosome']==g['chromosome'] and
                   g['start']-25000 <= f['position'] <= g['end']+25000]
        result.append(dict(symbol=symbol,gene=gene,window_selected_columns=indices,
                           assigned_columns=[i for i,a in enumerate(assignments) if a==gene]))
    return result


def permutation_path_game(predict, background, evaluation, groups, seed=42, blocks=8):
    """Each block uses every background once, with independent group orders.

    A path replaces whole groups from one donor to one evaluation row. Averaging
    paths estimates the finite-background marginal Shapley game. Telescoping
    enforces efficiency even before MC convergence. All models use equal paths.
    Blocks measure permutation-order uncertainty conditional on fixed donors;
    they do not measure finite-background sampling uncertainty.
    """
    nb, ne, m = len(background), len(evaluation), len(groups)
    if nb < 2 or blocks < 2 or not ne:
        raise ValueError('Insufficient explanation budget')
    if sorted(int(i) for group in groups for i in group) != list(range(evaluation.shape[1])):
        raise ValueError('Groups must partition all model inputs')
    if any(not len(group) for group in groups):
        raise ValueError('Empty group')
    rng = np.random.default_rng(seed)
    orders = np.array([[rng.permutation(m) for _ in range(nb)] for _ in range(blocks)])
    block_phi = np.zeros((blocks,ne,m))
    donor_phi = np.zeros((nb,ne,m))
    for b in range(blocks):
        for donor in range(nb):
            current = np.tile(background[donor], (ne,1))
            path = [current.copy()]
            for j in orders[b,donor]:
                current[:,groups[j]] = evaluation[:,groups[j]]
                path.append(current.copy())
            values = predict(np.concatenate(path)).reshape(m+1,ne)
            contribution = np.zeros((ne,m))
            contribution[:,orders[b,donor]] = np.diff(values,axis=0).T
            block_phi[b] += contribution/nb
            donor_phi[donor] += contribution/blocks
    phi = block_phi.mean(0)
    endpoint = predict(evaluation)-np.mean(predict(background))
    error = float(np.max(np.abs(phi.sum(1)-endpoint)))
    if error > 1e-5*max(1.,float(np.max(np.abs(endpoint)))):
        raise ValueError('Permutation-path efficiency failed')
    return dict(phi=phi, block_phi=block_phi, donor_phi=donor_phi, orders=orders,
        local_mc_se=block_phi.std(0,ddof=1)/np.sqrt(blocks), endpoint=endpoint,
        efficiency_error=error,
        half_budget_max_local_change=float(np.max(np.abs(block_phi[:blocks//2].mean(0)-phi))))
