# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# hydrochem_workbench/advanced.py
# the further analyses: end-member mixing with ion deviations, PCA, hierarchical
# clustering and group comparisons. No Streamlit in here.
#
# mixing: each sample is taken as a mix of 2 or 3 end members, with fractions found
# from conservative tracers (constrained to sum to 1, least squares when there are
# more tracers than needed). The deviation of every other ion from conservative
# mixing, measured - sum(f_k c_k), points to reactions along the flow path
# (Appelo and Postma, Geochemistry, Groundwater and Pollution)

# %%
import numpy as np
import pandas as pd

from hydrochem_workbench.chem import ANALYTES, hollow_mask

LARGE_DATASET = 200          # above this many samples the app warns that plots may be slow or crowded
LOR_SHARE_WARN = 0.25        # variables with more than this share below the LOR are left out by default
MULTI_MAJORS = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4']
TRANSFORMS = ['Centred log-ratio (clr)', 'log10, then standardised']


def mmol_col(a):
    return f'{a} (mmol/L)'


def analyte_of(col):
    # 'Ca (mmol/L)' -> 'Ca'; anything else -> None
    name = col.split(' (')[0]
    return name if name in ANALYTES and col.endswith(')') else None


def measured(v, cols):
    # the columns with values that rest on a filled pH, temperature or alkalinity set to NaN
    out = v[cols].astype(float).copy()
    for c in cols:
        out.loc[hollow_mask(v, [c]), c] = np.nan
    return out


def lor_share(v, replaced):
    # share of the samples in v with a below-LOR (replaced) value, for each analyte
    if replaced is None or not len(replaced):
        return {}
    rep = replaced[replaced['Sample'].isin(v['Sample'])]
    return {a: rep.loc[rep['Column'] == a, 'Sample'].nunique() / max(len(v), 1) for a in ANALYTES}


# %% mixing
def member_options(v):
    # every sample, then every group mean, as text the user picks from
    return [f'Sample: {s}' for s in v['Sample'].astype(str)] + \
           [f'Group mean: {g}' for g in pd.unique(v['Label'])]


def member_rows(v, choice):
    # the rows of v that make up an end member
    kind, name = choice.split(': ', 1)
    if kind == 'Sample':
        rows = v.index[v['Sample'].astype(str) == name]
        return rows[:1]
    return v.index[v['Label'] == name]


def member_values(v, choice, cols):
    # end-member composition: the sample's values, or the group mean of the measured values
    return measured(v.loc[member_rows(v, choice)], cols).mean()


def fractions(samples, members, tracers):
    # samples: (n x t) tracer values; members: (m x t). Returns (n x m) fractions summing to 1
    # and the rms misfit in scaled tracer units (zero when the tracers exactly fix the fractions).
    # Tracers are scaled by the spread of the end members so that, e.g., Cl in mmol/L and d18O
    # in permil count alike when there are more tracers than needed
    S = np.asarray(samples, dtype=float)
    E = np.asarray(members, dtype=float)
    m = E.shape[0]
    scale = E.max(axis=0) - E.min(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    B = ((E[:-1] - E[-1]) / scale).T                 # t x (m-1)
    if np.linalg.matrix_rank(B) < m - 1:
        raise ValueError('The end members cannot be told apart with these tracers. Pick end members '
                         'with different tracer values, or add a tracer.')
    Y = ((S - E[-1]) / scale).T                      # t x n
    f, *_ = np.linalg.lstsq(B, np.nan_to_num(Y), rcond=None)
    f = f.T                                          # n x (m-1)
    F = np.column_stack([f, 1 - f.sum(axis=1)])
    resid = np.sqrt(np.mean(((F @ E - S) / scale) ** 2, axis=1))
    bad = ~np.isfinite(S).all(axis=1)
    F[bad] = np.nan
    resid[bad] = np.nan
    return F, resid


def run_mixing(v_all, v, choices, tracers, ions, mc=None, outside_tol=0.02):
    # v_all: every sample (end members can come from groups that are not plotted); v: the samples
    # to evaluate. choices: end members as from member_options. Returns the end-member values,
    # fractions, misfit, deviations from conservative mixing (mmol/L), the exchange check
    # (meq/L) and, with the uncertainty run, 5th and 95th percentiles of fractions and deviations
    if len(set(choices)) != len(choices):
        raise ValueError('Pick a different sample or group for each end member.')
    if len(tracers) < len(choices) - 1:
        raise ValueError(f'{len(choices)} end members need at least {len(choices) - 1} tracer(s).')
    cols = list(dict.fromkeys(list(tracers) + list(ions)))
    names = [f'EM{k + 1}' for k in range(len(choices))]
    E = pd.DataFrame([member_values(v_all, c, cols) for c in choices], index=names)
    if E[tracers].isna().any().any():
        raise ValueError('An end member has no value for a tracer.')
    S = measured(v, cols)
    F, misfit = fractions(S[tracers].to_numpy(), E[tracers].to_numpy(), tracers)
    frac = pd.DataFrame(F, index=v.index, columns=[f'f {n}' for n in names])
    pred = pd.DataFrame(F @ E[ions].to_numpy(), index=v.index, columns=ions)
    dev = S[ions] - pred
    outside = ((frac < -outside_tol) | (frac > 1 + outside_tol)).any(axis=1) & frac.notna().all(axis=1)
    out = dict(members=E, member_choices=dict(zip(names, choices)), tracers=list(tracers), ions=list(ions),
               fractions=frac, misfit=pd.Series(misfit, index=v.index), predicted=pred,
               deviations=dev, outside=outside, exchange=exchange_check(dev))
    if mc is not None and 'conc' in mc:
        out.update(_mixing_mc(v_all, v, choices, tracers, ions, mc, names))
    return out


def exchange_check(dev):
    # deviations in meq/L: (Na + K) against (Ca + Mg). Cation exchange moves them in opposite
    # directions, on the 1:-1 line
    z = {a: abs(ANALYTES[a][1]) for a in ANALYTES}
    def meq(names):
        cols = [mmol_col(a) for a in names if mmol_col(a) in dev.columns]
        if len(cols) < len(names):
            return None
        return sum(dev[c] * z[analyte_of(c)] for c in cols)
    nak, camg = meq(['Na', 'K']), meq(['Ca', 'Mg'])
    if nak is None or camg is None:
        return None
    return pd.DataFrame({'dNa+K (meq/L)': nak, 'dCa+Mg (meq/L)': camg})


def _mixing_mc(v_all, v, choices, tracers, ions, mc, names):
    # the same calculation for every realisation of the uncertainty analysis
    conc = mc['conc']
    cols = list(dict.fromkeys(list(tracers) + list(ions)))
    missing = [c for c in cols if c not in conc]
    if missing:
        return dict(mc_note='No uncertainty for ' + ', '.join(missing) + ' (not in the uncertainty run).')
    pos = {i: k for k, i in enumerate(mc['index'])}
    rows_s = [pos[i] for i in v.index]
    member_pos = [[pos[i] for i in member_rows(v_all, c)] for c in choices]
    hollow = {c: hollow_mask(v_all, [c]) for c in cols}
    n = mc['n']
    fr, dv = [], []
    for r in range(n):
        real = {}
        for c in cols:
            x = conc[c][r].copy()
            x[hollow[c].reindex(mc['index']).to_numpy()] = np.nan
            real[c] = x
        E = np.array([[np.nanmean(real[c][p]) if np.isfinite(real[c][p]).any() else np.nan for c in cols]
                      for p in member_pos])
        S = np.array([[real[c][k] for c in cols] for k in rows_s])
        ti = [cols.index(t) for t in tracers]
        ii = [cols.index(i) for i in ions]
        try:
            F, _ = fractions(S[:, ti], E[:, ti], tracers)
        except ValueError:
            continue
        fr.append(F)
        dv.append(S[:, ii] - F @ E[:, ii])
    if not fr:
        return dict(mc_note='The uncertainty run gave no usable realisations for the mixing.')
    fr, dv = np.stack(fr), np.stack(dv)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            q = lambda a, p: np.nanpercentile(a, p, axis=0)
            f_lo, f_hi, d_lo, d_hi = q(fr, 5), q(fr, 95), q(dv, 5), q(dv, 95)
    fcols = [f'f {n}' for n in names]
    return dict(frac_lo=pd.DataFrame(f_lo, index=v.index, columns=fcols),
                frac_hi=pd.DataFrame(f_hi, index=v.index, columns=fcols),
                dev_lo=pd.DataFrame(d_lo, index=v.index, columns=ions),
                dev_hi=pd.DataFrame(d_hi, index=v.index, columns=ions),
                mc_used=len(fr))


def mixing_table(v, res):
    # one row per sample: fractions (with intervals), misfit, flags and deviations
    t = pd.DataFrame({'Sample': v['Sample'], 'Label': v['Label']}, index=v.index)
    for c in res['fractions'].columns:
        t[c] = res['fractions'][c]
        if 'frac_lo' in res:
            t[f'{c} 5th pct'] = res['frac_lo'][c]
            t[f'{c} 95th pct'] = res['frac_hi'][c]
    if len(res['tracers']) > len(res['members']) - 1:
        t['Tracer misfit (scaled rms)'] = res['misfit']
    t['Outside the end members'] = res['outside'].map({True: 'Yes', False: ''})
    for c in res['ions']:
        name = f'd{analyte_of(c) or c} (mmol/L)'
        t[name] = res['deviations'][c]
        if 'dev_lo' in res:
            lo, hi = res['dev_lo'][c], res['dev_hi'][c]
            t[f'{name} 5th pct'] = lo
            t[f'{name} 95th pct'] = hi
            t[f'{name} differs from 0'] = np.where((lo > 0) | (hi < 0), 'Yes', '')
    if res['exchange'] is not None:
        for c in res['exchange'].columns:
            t[c] = res['exchange'][c]
    return t


# %% PCA and clustering
def prepare(v, cols):
    # the measured values of the chosen analytes, samples with any gap or non-positive value dropped
    X = measured(v, cols)
    keep = X.notna().all(axis=1) & (X > 0).all(axis=1)
    return X[keep], v.loc[~keep, 'Sample'].astype(str).tolist()


def transform(X, how):
    # clr: log, centred on each sample's mean log (removes the closure of concentrations),
    # then centred on each variable. log10: log10, then standardised (mean 0, SD 1)
    L = np.log(X.to_numpy(dtype=float))
    if how.startswith('Centred'):
        Z = L - L.mean(axis=1, keepdims=True)
        Z = Z - Z.mean(axis=0)
    else:
        L = L / np.log(10)
        sd = L.std(axis=0, ddof=1)
        Z = (L - L.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def pca(X, how):
    Z = transform(X, how)
    U, s, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    var = s ** 2 / np.sum(s ** 2)
    k = len(s)
    pcs = [f'PC{i + 1}' for i in range(k)]
    # sign so that the largest loading on each component is positive (stable between runs)
    sign = np.sign(Vt[np.arange(k), np.abs(Vt).argmax(axis=1)])
    Vt = Vt * sign[:, None]
    U = U * sign[None, :]
    scores = pd.DataFrame(U * s, index=X.index, columns=pcs)
    loadings = pd.DataFrame(Vt.T, index=X.columns, columns=pcs)
    return dict(scores=scores, loadings=loadings, explained=pd.Series(var, index=pcs), transform=how)


def clusters(X, how, k):
    # Ward linkage on Euclidean distance between the transformed samples (Guler et al. 2002)
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    Z = transform(X, how)
    link = linkage(Z.to_numpy(), method='ward')
    raw = fcluster(link, t=k, criterion='maxclust')
    # number the clusters in the order they appear along the dendrogram
    order = leaves_list(link)
    seen = list(dict.fromkeys(raw[order]))
    renum = {c: i + 1 for i, c in enumerate(seen)}
    lab = pd.Series([f'Cluster {renum[c]}' for c in raw], index=X.index)
    return dict(link=link, labels=lab, k=k, transform=how, names=list(Z.index))


def crosstab(v, labels):
    t = pd.crosstab(labels, v.loc[labels.index, 'Label'])
    t.index.name = None
    t.columns.name = None
    return t


def cluster_means(v, labels, cols):
    X = measured(v.loc[labels.index], cols)
    out = X.groupby(labels).mean()
    out.insert(0, 'Samples', labels.value_counts().reindex(out.index))
    return out


# %% group comparison
def holm(p):
    # Holm step-down adjustment of a list of p values
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


MIN_PER_GROUP = 3


def group_tests(v, cols):
    # Kruskal-Wallis across the groups for each variable, then two-sided Mann-Whitney U
    # tests between each pair, Holm-adjusted within the variable. Groups with fewer than
    # MIN_PER_GROUP values for a variable are left out of its tests
    from scipy.stats import kruskal, mannwhitneyu
    kw, pairs = [], []
    X = measured(v, cols)
    groups = list(pd.unique(v['Label']))
    for c in cols:
        vals = {g: X.loc[v['Label'] == g, c].dropna() for g in groups}
        used = [g for g in groups if len(vals[g]) >= MIN_PER_GROUP]
        row = {'Variable': c, 'Groups tested': len(used),
               'Groups left out (< 3 values)': ', '.join(str(g) for g in groups if g not in used)}
        if len(used) >= 2 and len({x for g in used for x in vals[g]}) > 1:
            h, p = kruskal(*[vals[g] for g in used])
            row.update({'Kruskal-Wallis H': h, 'p': p})
            prs = [(a, b) for i, a in enumerate(used) for b in used[i + 1:]]
            ps = [mannwhitneyu(vals[a], vals[b], alternative='two-sided').pvalue for a, b in prs]
            for (a, b), praw, padj in zip(prs, ps, holm(ps)):
                pairs.append({'Variable': c, 'Group A': a, 'Group B': b,
                              'Median A': vals[a].median(), 'Median B': vals[b].median(),
                              'p (Mann-Whitney)': praw, 'p (Holm-adjusted)': padj})
        else:
            row.update({'Kruskal-Wallis H': np.nan, 'p': np.nan})
        kw.append(row)
    return pd.DataFrame(kw), pd.DataFrame(pairs, columns=['Variable', 'Group A', 'Group B', 'Median A',
                                                          'Median B', 'p (Mann-Whitney)', 'p (Holm-adjusted)'])


def spearman(v, cols):
    # pairwise-complete Spearman rank correlations and the number of samples behind each
    X = measured(v, cols)
    X = X.loc[:, X.notna().sum() >= 4]
    r = X.corr(method='spearman', min_periods=4)
    n = X.notna().astype(int).T @ X.notna().astype(int)
    return r, n
