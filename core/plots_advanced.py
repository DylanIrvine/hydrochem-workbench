# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# core/plots_advanced.py
# figures for the further analyses (mixing, PCA, clusters, group comparison), in the
# same style as core/plots.py

# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from core import plots
from core.plots import STYLE, CM, PALETTE, MARKERS, pretty
from core.advanced import analyte_of
from core.chem import hollow_mask

EM_STYLE = dict(marker='*', s=150, color='white', edgecolors='black', linewidths=1.0, zorder=60)
PANELS_PER_PAGE = 6


def _short(col):
    # 'Ca (mmol/L)' -> 'Ca' with subscripts
    return pretty(analyte_of(col) or col)


def _plain_log(ax, xlog, ylog):
    # plain numbers on log axes, with the 2s and 5s labelled when the axis spans little more than a decade
    for axis, log, lim in [(ax.xaxis, xlog, ax.get_xlim()), (ax.yaxis, ylog, ax.get_ylim())]:
        if log:
            axis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val:g}'))
            if lim[0] > 0 and np.log10(lim[1] / lim[0]) < 1.3:
                axis.set_minor_formatter(FuncFormatter(
                    lambda val, pos: f'{val:g}' if f'{val:e}'[0] in '25' else ''))
            else:
                axis.set_minor_formatter(NullFormatter())


def _em_handle():
    return Line2D([], [], ls='', marker='*', mfc='white', mec='black', mew=1.0, ms=10, label='End member')


def _mix_handle():
    return Line2D([], [], ls='--', color='0.35', lw=0.9, label='Conservative mixing')


# %% mixing
def mixing_panels(v, res, panels, log=False, means=False, letters=None, letters_pos=plots.PANEL_LABEL_POS[0]):
    # panels: list of (x column, y column). End members as stars, the mixing line (2) or
    # triangle (3) dashed; every point inside it can be made by mixing alone
    E = res['members']
    n = len(panels)
    if not n:
        return None
    nrows = int(np.ceil(n / 2))
    height = 4.6 * nrows + 1.6
    styles = plots.group_styles(v)
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, height * CM), dpi=150)
        axs = np.atleast_1d(fig.subplots(nrows, 2)).ravel()
        fig.subplots_adjust(left=0.1, right=0.98, top=1 - 0.3 / height, bottom=2.2 / height,
                            hspace=0.42, wspace=0.3)
        any_hollow = False
        for ax, (xc, yc) in zip(axs, panels):
            x, y = v[xc].astype(float), v[yc].astype(float)
            if log:
                x, y = x.where(x > 0), y.where(y > 0)
            hollow = hollow_mask(v, [xc, yc])
            plots._scatter(ax, v, x, y, hollow, styles, legend=False, means=means)
            any_hollow = any_hollow or bool((hollow & x.notna() & y.notna()).any())
            ex, ey = E[xc].to_numpy(float), E[yc].to_numpy(float)
            # edges sampled along their length so they stay right on log axes
            idx = list(range(len(E))) + ([0] if len(E) > 2 else [])
            for a, b in zip(idx[:-1], idx[1:]):
                t = np.linspace(0, 1, 60)
                ax.plot(ex[a] + t * (ex[b] - ex[a]), ey[a] + t * (ey[b] - ey[a]), ls='--', color='0.35',
                        lw=0.9, zorder=5)
            ax.scatter(ex, ey, **EM_STYLE)
            for k, name in enumerate(E.index):
                ax.annotate(name, (ex[k], ey[k]), xytext=(5, 4), textcoords='offset points', fontsize=7,
                            zorder=61)
            if log:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set_xlabel(pretty(xc))
            ax.set_ylabel(pretty(yc))
        if log:
            for ax in axs[:n]:
                _plain_log(ax, True, True)
        for ax in axs[n:]:
            ax.axis('off')
        plots.panel_letters(axs[:n], letters, letters_pos)
        handles = plots._legend_handles(styles, any_hollow, means) + [_em_handle(), _mix_handle()]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=min(4, len(handles)),
                   frameon=False)
    return fig


def deviation_plot(v, res, symlog=False):
    # deviation of each ion from conservative mixing, one column per ion, the groups side by side
    dev = res['deviations']
    ions = [c for c in res['ions'] if dev[c].notna().any()]
    if not ions:
        return None
    styles = plots.group_styles(v)
    groups = list(styles)
    width = 0.7
    step = width / max(len(groups), 1)
    rng = np.random.default_rng(7)
    unc = 'dev_lo' in res
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, 9 * CM), dpi=150)
        ax = fig.add_axes([0.1, 0.14, 0.62, 0.8])
        ax.axhline(0, color='0.35', lw=0.8, zorder=1)
        for i, c in enumerate(ions):
            for g, lab in enumerate(groups):
                col, mk = styles[lab]
                rows = v.index[(v['Label'] == lab) & dev[c].notna()]
                if not len(rows):
                    continue
                x0 = i - width / 2 + step * (g + 0.5)
                xs = x0 + rng.uniform(-0.3, 0.3, len(rows)) * step
                if unc:
                    for xx, r in zip(xs, rows):
                        ax.plot([xx, xx], [res['dev_lo'].at[r, c], res['dev_hi'].at[r, c]],
                                color=plots.UNC_COLOUR, lw=plots.UNC_LW, zorder=2)
                ax.scatter(xs, dev.loc[rows, c], marker=mk, s=22, color=col, alpha=0.85, edgecolors='black',
                           linewidths=0.5, zorder=10)
        ax.set_xticks(range(len(ions)))
        ax.set_xticklabels([_short(c) for c in ions])
        ax.set_xlim(-0.6, len(ions) - 0.4)
        if symlog:
            # linear within +-lin, log beyond; lin is a round number near the small deviations
            a = np.abs(dev[ions].to_numpy(float))
            a = a[np.isfinite(a) & (a > 0)]
            lin = 10 ** np.floor(np.log10(np.percentile(a, 25))) if len(a) else 0.01
            ax.set_yscale('symlog', linthresh=lin, linscale=0.6)
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda val, pos: '0' if val == 0 else ('' if abs(val) < lin * 1.5 else f'{val:g}')))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel('Measured minus conservative mixing (mmol/L)')
        handles = plots._legend_handles(styles, False)
        if unc:
            handles.append(plots._unc_handle())
        handles += [Line2D([], [], ls='', marker=''), Line2D([], [], ls='', marker='', label='Above 0: gained'),
                    Line2D([], [], ls='', marker='', label='Below 0: lost')]
        fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.74, 0.94))
    return fig


def exchange_plot(v, res):
    # (Na + K) against (Ca + Mg) deviations, meq/L. Points along the 1:-1 line: cation exchange
    ex = res['exchange']
    if ex is None or not ex.notna().all(axis=1).any():
        return None
    x, y = ex['dCa+Mg (meq/L)'], ex['dNa+K (meq/L)']
    styles = plots.group_styles(v)
    hollow = pd.Series(False, index=v.index)
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, 10 * CM), dpi=150)
        ax = fig.add_axes([0.1, 0.13, 0.55, 0.83])
        plots._scatter(ax, v, x, y, hollow, styles, legend=False)
        ok = x.notna() & y.notna()
        lim = 1.15 * max(float(np.nanmax(np.abs(np.r_[x[ok], y[ok]]))), 1e-3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.axhline(0, color='0.8', lw=0.5, zorder=0)
        ax.axvline(0, color='0.8', lw=0.5, zorder=0)
        ax.plot([-lim, lim], [lim, -lim], ls='--', color='0.35', lw=0.8, zorder=1)
        kw = dict(transform=ax.transAxes, fontsize=6, color='0.25')
        ax.text(0.03, 0.97, 'Na + K gained, Ca + Mg lost\n(exchange during freshening)', ha='left', va='top', **kw)
        ax.text(0.97, 0.03, 'Ca + Mg gained, Na + K lost\n(reverse exchange, e.g. salinisation)',
                ha='right', va='bottom', **kw)
        ax.set_xlabel('Ca + Mg, measured minus mixing (meq/L)')
        ax.set_ylabel('Na + K, measured minus mixing (meq/L)')
        handles = plots._legend_handles(styles, False) + [
            Line2D([], [], ls='--', color='0.35', lw=0.8, label='1:-1 (cation exchange)')]
        fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.66, 0.96))
    return fig


# %% PCA
def pca_biplot(v, res, a='PC1', b='PC2', means=False):
    # sample scores by group, variable loadings as arrows, and the share of variance per component
    sc, ld, ex = res['scores'], res['loadings'], res['explained']
    vv = v.loc[sc.index]
    styles = plots.group_styles(vv)
    hollow = pd.Series(False, index=vv.index)
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, 11 * CM), dpi=150)
        ax = fig.add_axes([0.09, 0.12, 0.56, 0.84])
        ax.axhline(0, color='0.85', lw=0.5, zorder=0)
        ax.axvline(0, color='0.85', lw=0.5, zorder=0)
        plots._scatter(ax, vv, sc[a], sc[b], hollow, styles, legend=False, means=means)
        reach = np.nanmax(np.abs(sc[[a, b]].to_numpy()))
        arm = np.nanmax(np.hypot(ld[a], ld[b]))
        k = 0.85 * reach / arm if arm > 0 else 1.0
        for var in ld.index:
            dx, dy = ld.at[var, a] * k, ld.at[var, b] * k
            ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color='0.25', lw=0.8, shrinkA=0, shrinkB=0), zorder=20)
            ax.text(dx * 1.08, dy * 1.08, _short(var), fontsize=7, color='0.15', ha='center', va='center',
                    zorder=21, bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.7))
        lim = 1.15 * max(reach, 1e-9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'{a} ({100 * ex[a]:.1f} % of variance)')
        ax.set_ylabel(f'{b} ({100 * ex[b]:.1f} % of variance)')
        # share of variance for each component
        ax2 = fig.add_axes([0.74, 0.62, 0.23, 0.3])
        ax2.bar(range(len(ex)), 100 * ex.to_numpy(), color='0.6', edgecolor='black', lw=0.4)
        ax2.plot(range(len(ex)), 100 * ex.cumsum().to_numpy(), '-o', color='0.2', ms=2.5, lw=0.8)
        ax2.set_xticks(range(len(ex)))
        ax2.set_xticklabels([p.replace('PC', '') for p in ex.index], fontsize=6)
        ax2.set_xlabel('Component', fontsize=7)
        ax2.set_ylabel('Variance (%)', fontsize=7)
        ax2.set_ylim(0, 100)
        ax2.tick_params(labelsize=6)
        handles = plots._legend_handles(styles, False, means)
        fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.72, 0.5))
    return fig


# %% clusters
def dendrogram_fig(v, res, max_labels=60):
    # Ward dendrogram, branches in the cluster colours, the cut as a dashed line and the
    # sample names in their group colours
    from scipy.cluster.hierarchy import dendrogram
    link, labels, k = res['link'], res['labels'], res['k']
    n = len(labels)
    cl_col = cluster_colours(labels)
    leaf_cl = labels.to_numpy()
    # cluster of every node (None where the branch joins two clusters)
    node_cl = {i: leaf_cl[i] for i in range(n)}
    for j, row in enumerate(link):
        a, b = node_cl[int(row[0])], node_cl[int(row[1])]
        node_cl[n + j] = a if a == b else None
    colour = lambda node: cl_col.get(node_cl.get(node), '0.55')
    styles = plots.group_styles(v)
    cut = 0.5 * (link[-k, 2] + link[-k + 1, 2]) if 1 < k <= n - 1 else None
    names = v.loc[labels.index, 'Sample'].astype(str).tolist()
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, 10 * CM), dpi=150)
        ax = fig.add_axes([0.08, 0.25 if n <= max_labels else 0.1, 0.72, 0.7])
        d = dendrogram(link, ax=ax, labels=names, link_color_func=colour, leaf_rotation=90,
                       leaf_font_size=max(4, min(7, 400 / max(n, 1))), no_labels=n > max_labels)
        if n <= max_labels:
            group_of = v.loc[labels.index, 'Label'].to_numpy()
            for tick, leaf in zip(ax.get_xticklabels(), d['leaves']):
                tick.set_color(styles[group_of[leaf]][0])
        if cut is not None:
            ax.axhline(cut, ls='--', color='0.35', lw=0.8)
        ax.set_ylabel('Ward linkage distance')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        handles = [matplotlib.patches.Patch(color=c, label=l) for l, c in cl_col.items()]
        handles.append(Line2D([], [], ls='--', color='0.35', lw=0.8, label='Cut'))
        if n <= max_labels:
            handles += [Line2D([], [], ls='', marker='')] + [
                Line2D([], [], ls='', marker='s', mfc=styles[g][0], mec='none', ms=5, label=f'{g} (name colour)')
                for g in styles]
        fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.81, 0.95), fontsize=7)
    return fig


# clusters start from the warm end of the palette, so they do not look like the groups
# (which start from green)
CLUSTER_PALETTE = ['#F37021', '#CC004C', '#FCB711', '#6460AA', '#0089D0', '#0DB14B', '#808080',
                   '#a88958', '#00bdd0', '#0055d0']


def cluster_colours(labels):
    names = sorted(pd.unique(labels), key=lambda s: int(s.split()[-1]))
    return {nm: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, nm in enumerate(names)}


def cluster_piper(v, labels, means=False):
    # the Piper diagram with the samples grouped by cluster instead of Label
    w = v.loc[labels.index].copy()
    cols = cluster_colours(labels)
    names = list(cols)
    w['Label'] = pd.Categorical(labels, categories=names).astype(str)
    w = w.iloc[np.argsort([names.index(x) for x in w['Label']], kind='stable')]
    w['Color'] = w['Label'].map(cols)
    w['Marker'] = w['Label'].map(lambda s: MARKERS[names.index(s) % len(MARKERS)])
    return plots.piper(w, means=means)


# %% group comparison
def _p_text(p):
    if not np.isfinite(p):
        return 'not tested'
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def box_page(v, cols, kw, log=False, letters=None, letters_pos=plots.PANEL_LABEL_POS[0]):
    # one panel per variable: a box for each group with the samples over it, Kruskal-Wallis p in the corner
    from core.advanced import measured
    n = len(cols)
    if not n:
        return None
    styles = plots.group_styles(v)
    groups = list(styles)
    X = measured(v, cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    height = 5.0 * nrows + 1.4
    pvals = kw.set_index('Variable')['p'] if len(kw) else pd.Series(dtype=float)
    rng = np.random.default_rng(11)
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, height * CM), dpi=150)
        axs = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
        fig.subplots_adjust(left=0.08, right=0.98, top=1 - 0.4 / height, bottom=1.9 / height,
                            hspace=0.35, wspace=0.42)
        for ax, c in zip(axs, cols):
            data = [X.loc[v['Label'] == g, c].dropna() for g in groups]
            if log:
                data = [d[d > 0] for d in data]
            pos = [i for i, d in enumerate(data) if len(d)]
            if pos:
                bp = ax.boxplot([data[i] for i in pos], positions=pos, widths=0.6, patch_artist=True,
                                showfliers=False, medianprops=dict(color='black', lw=1.0),
                                whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
                for patch, i in zip(bp['boxes'], pos):
                    patch.set_facecolor(mcolors.to_rgba(styles[groups[i]][0], 0.35))
                for i in pos:
                    col, mk = styles[groups[i]]
                    xs = i + rng.uniform(-0.18, 0.18, len(data[i]))
                    ax.scatter(xs, data[i], marker=mk, s=10, color=col, edgecolors='black', linewidths=0.3,
                               zorder=10)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([''] * len(groups))
            ax.set_xlim(-0.6, len(groups) - 0.4)
            if log:
                ax.set_yscale('log')
                _plain_log(ax, False, True)
            ax.set_ylabel(pretty(c))
            ax.text(0.02, 0.98, 'Kruskal-Wallis\n' + _p_text(float(pvals.get(c, np.nan))),
                    transform=ax.transAxes, ha='left', va='top', fontsize=6, color='0.25',
                    bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8), zorder=30)
        for ax in axs[n:]:
            ax.axis('off')
        plots.panel_letters(axs[:n], letters, letters_pos)
        handles = [matplotlib.patches.Patch(facecolor=mcolors.to_rgba(styles[g][0], 0.35), edgecolor='black',
                                            lw=0.5, label=g) for g in groups]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=min(4, len(handles)),
                   frameon=False)
    return fig


def corr_heatmap(r, n, min_n=4):
    # Spearman rank correlations, with the coefficient in each cell; cells with too few samples are blank
    k = len(r)
    if k < 2:
        return None
    vals = r.to_numpy(float).copy()
    vals[n.to_numpy() < min_n] = np.nan
    size = min(17, 3 + 0.9 * k)
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, size * CM), dpi=150)
        ax = fig.add_axes([0.18, 0.2, 0.6, 0.75])
        cmap = matplotlib.colormaps['RdBu_r'].with_extremes(bad='0.92')
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=-1, vmax=1)
        names = [pretty(c) for c in r.columns]
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticklabels(names)
        if k <= 16:
            for i in range(k):
                for j in range(k):
                    if np.isfinite(vals[i, j]):
                        ax.text(j, i, f'{vals[i, j]:.2f}', ha='center', va='center', fontsize=5.5,
                                color='white' if abs(vals[i, j]) > 0.6 else 'black')
        cax = fig.add_axes([0.82, 0.3, 0.02, 0.5])
        fig.colorbar(im, cax=cax).set_label('Spearman rank correlation')
    return fig
