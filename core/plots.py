# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.
#
# Portions of this file are adapted from WQChartPy (triangle_piper.py),
# Copyright (C) the WQChartPy authors (Yang et al., 2022), licensed under
# GPL-3.0: https://github.com/jyangfsu/WQChartPy
# Modified 2024 (D. Irvine: coloured fields, water-type labels) and 2026
# (returns a Figure, grouped drawing, hollow markers for filled inputs).

# core/plots.py
# all of the figures
#
#   piper()            Piper diagram
#   schoeller()        Schoeller
#   stiff()            Stiff diagrams
#   wqchartpy_figure() Durov, Gibbs, Chadha, HFE-D from WQChartPy
#   gallery_page()     x-y panels with reference lines
#   custom()           user-defined x-y plot
#
# nb. uses Figure() and not plt.figure(), otherwise figures pile up in memory
# each time Streamlit reruns (same as HydroSTITCH)

# %%
import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from core.chem import SEAWATER_MMOL, hollow_mask

CM = 1.0 / 2.54

# %% style
# Arial if available, otherwise Liberation Sans
STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'mathtext.default': 'regular',
}

# colours and markers for the groups
PALETTE = ['#0089d0', '#F37021', '#CC004C', '#7ea070', '#665e8e', '#FCB711',
           '#00bdd0', '#a88958', '#0055d0', '#7a7c78']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h']

# Piper colours
PIPER = {'Ca': '#FCB711', 'NaK': '#F37021', 'Mg': '#CC004C',
         'HCO3': '#00bdd0', 'Cl': '#0089d0', 'SO4': '#0055d0',
         'Ca-Cl': '#7ea070', 'Na-HCO3': '#a88958', 'Mg-HCO3': '#665e8e',
         'Na-Cl': '#7a7c78', 'Mixed': '0.9'}

# subscripts for axis labels
_PRETTY = [('HCO3', 'HCO$_3$'), ('SO4', 'SO$_4$'), ('NO3', 'NO$_3$'),
           ('CO3', 'CO$_3$'), ('SiO2', 'SiO$_2$'), ('pCO2', 'pCO$_2$')]


def pretty(name):
    out = name
    for raw, nice in _PRETTY:
        out = out.replace(raw, nice)
    return out


def group_styles(v):
    # colour and marker for each Label (Color/Marker columns override)
    styles = {}
    for k, lab in enumerate(pd.unique(v['Label'])):
        rows = v[v['Label'] == lab]
        col = next((c for c in rows['Color'] if c and mcolors.is_color_like(c)), None)
        mk = next((m for m in rows['Marker'] if m), None)
        styles[lab] = (col or PALETTE[k % len(PALETTE)], mk or MARKERS[k % len(MARKERS)])
    return styles


def _scatter(ax, v, x, y, hollow, styles, cvals=None, size=22, legend=True, zorder=10):
    # scatter by group, hollow markers where hollow is True
    # if cvals is given, colour by that variable instead (viridis)
    mappable = None
    if cvals is not None:
        ok = cvals.notna()
        norm = mcolors.Normalize(vmin=cvals[ok].min(), vmax=cvals[ok].max()) if ok.any() else None
    for lab, (col, mk) in styles.items():
        g = (v['Label'] == lab) & x.notna() & y.notna()
        solid = g & ~hollow
        open_ = g & hollow
        if cvals is not None and norm is not None:
            sel = solid & cvals.notna()
            if sel.any():
                mappable = ax.scatter(x[sel], y[sel], c=cvals[sel], cmap='viridis', norm=norm,
                                      marker=mk, s=size, edgecolors='black', linewidths=0.5,
                                      zorder=zorder, label=lab if legend else None)
            sel = open_ | (solid & cvals.isna())
            if sel.any():
                ax.scatter(x[sel], y[sel], marker=mk, s=size, facecolors='none',
                           edgecolors='0.3', linewidths=0.9, zorder=zorder)
        else:
            if solid.any():
                ax.scatter(x[solid], y[solid], marker=mk, s=size, color=col, alpha=0.85,
                           edgecolors='black', linewidths=0.5, zorder=zorder,
                           label=lab if legend else None)
            if open_.any():
                ax.scatter(x[open_], y[open_], marker=mk, s=size, facecolors='none',
                           edgecolors=col, linewidths=1.0, zorder=zorder,
                           label=lab if (legend and not solid.any()) else None)
    return mappable


def _legend_handles(styles, any_hollow):
    handles = [Line2D([], [], ls='', marker=mk, mfc=col, mec='black', mew=0.5, ms=5, label=lab)
               for lab, (col, mk) in styles.items()]
    if any_hollow:
        handles.append(Line2D([], [], ls='', marker='o', mfc='none', mec='0.3', mew=1.0, ms=5,
                              label='Filled input (hollow)'))
    return handles


# %% Piper
def piper(v, color_by=None, water_legend=False, show_types=True):
    # Piper diagram, adapted from WQChartPy triangle_piper with coloured fields
    with matplotlib.rc_context(STYLE):
        offset = 0.10
        h = 0.5 * np.tan(np.pi / 3.0)
        t3 = np.tan(np.pi / 3)
        a = 0.5   # alpha for the fields

        fig = Figure(figsize=(16 * CM, 16 * CM), dpi=150)
        ax = fig.add_subplot(111, aspect='equal', frameon=False, xticks=[], yticks=[])

        # outlines
        ltx = np.array([0, 0.5, 1, 0])
        lty = np.array([0, h, 0, 0])
        ax.plot(ltx, lty, '-k', lw=1.0)
        ax.plot(ltx + 2 * offset + 1, lty, '-k', lw=1.0)
        ax.plot(np.array([0.5, 1, 1.5, 1, 0.5]) + offset,
                h * np.array([1, 2, 1, 0, 1]) + offset * t3, '-k', lw=1.0)

        # grid lines and tick labels
        ticklabels = ['0', '20', '40', '60', '80', '100']
        R = 1 + 2 * offset   # shift for the right triangle
        for i, x in enumerate(np.linspace(0, 1, 6)):
            inner = i in [1, 2, 3, 4]
            for sx, lab_bottom, lab_right, lab_left in [
                    (0, ticklabels[-i - 1], ticklabels[i], ticklabels[i]),
                    (R, ticklabels[i], ticklabels[-i - 1], ticklabels[-i - 1])]:
                ax.plot([x + sx, x - x / 2.0 + sx], [0, x / 2.0 * t3], 'k-', lw=0.5)
                ax.plot([x + sx, (1 - x) / 2.0 + x + sx], [0, (1 - x) / 2.0 * t3], 'k-', lw=0.5)
                ax.plot([x / 2 + sx, 1 - x / 2 + sx], [x / 2 * t3, x / 2 * t3], 'k-', lw=0.5)
                if inner:
                    ax.text(x + sx, -0.03, lab_bottom, ha='center', va='center')
                    ax.text((1 - x) / 2.0 + x + sx + 0.026, (1 - x) / 2.0 * t3 + 0.015,
                            lab_right, ha='center', va='center', rotation=-60)
                    ax.text(x / 2 + sx - 0.026, x / 2 * t3 + 0.015,
                            lab_left, ha='center', va='center', rotation=60)
            # diamond
            dx0 = 0.5 + offset + 0.5 * x
            dx1 = 1 + offset + 0.5 * x
            ax.plot([dx0, dx1], [h + offset * t3 + 0.5 * x * t3, offset * t3 + 0.5 * x * t3], 'k-', lw=0.5)
            ax.plot([dx0, dx1], [h + offset * t3 - 0.5 * x * t3, 2 * h + offset * t3 - 0.5 * x * t3], 'k-', lw=0.5)
            if inner:
                ax.text(dx0 - 0.026, h + offset * t3 + 0.5 * x * t3 + 0.015, ticklabels[i],
                        ha='center', va='center', rotation=60)
                ax.text(dx1 + 0.026, offset * t3 + 0.5 * x * t3 - 0.015, ticklabels[-i - 1],
                        ha='center', va='center', rotation=60)
                ax.text(dx0 - 0.026, h + offset * t3 - 0.5 * x * t3 - 0.015, ticklabels[i],
                        ha='center', va='center', rotation=-60)
                ax.text(dx1 + 0.026, 2 * h + offset * t3 - 0.5 * x * t3 + 0.015, ticklabels[-i - 1],
                        ha='center', va='center', rotation=-60)

        # axis titles
        c30, s30 = np.cos(np.pi / 30), np.sin(np.pi / 30)
        ax.text(0.5, -offset, 'Ca$^{2+}$ (%)', ha='center', va='center')
        ax.text(R + 0.5, -offset, 'Cl$^{-}$ (%)', ha='center', va='center')
        ax.text(0.25 - offset * c30, 0.25 * t3 + offset * s30, 'Mg$^{2+}$ (%)',
                ha='center', va='center', rotation=60)
        ax.text(1.75 + 2 * offset + offset * c30, 0.25 * t3 + offset * s30, 'SO$_4^{2-}$ (%)',
                ha='center', va='center', rotation=-60)
        ax.text(0.75 + offset * c30, 0.25 * t3 + offset * s30, 'Na$^+$+K$^+$ (%)',
                ha='center', va='center', rotation=-60)
        ax.text(R + 0.25 - offset * c30, 0.25 * t3 + offset * s30, 'HCO$_3^-$+CO$_3^{2-}$ (%)',
                ha='center', va='center', rotation=60)
        ax.text(0.5 + 1.5 * offset + offset * c30, h + offset * t3 + 0.25 * t3 + offset * s30,
                'SO$_4^{2-}$+Cl$^-$ (%)', ha='center', va='center', rotation=60)
        ax.text(1.25 + offset + offset * c30, h + offset * t3 + 0.25 * t3 + offset * s30,
                'Ca$^{2+}$+Mg$^{2+}$ (%)', ha='center', va='center', rotation=-60)

        # fill the cation triangle
        ax.fill([0.25, 0.5, 0.75], [h / 2, 0, h / 2], color=PIPER['Mixed'], zorder=0, alpha=a, lw=0)
        ax.fill([0.25, 0.5, 0.75], [h / 2, h, h / 2], color=PIPER['Mg'], zorder=0, alpha=a, lw=0)
        ax.fill([0.0, 0.5, 0.25], [0, 0, h / 2], color=PIPER['Ca'], zorder=0, alpha=a, lw=0)
        ax.fill([0.5, 1.0, 0.75], [0, 0, h / 2], color=PIPER['NaK'], zorder=0, alpha=a, lw=0)
        # anion triangle
        ax.fill([R + 0.25, R + 0.5, R + 0.75], [h / 2, 0, h / 2], color=PIPER['Mixed'], zorder=0, lw=0)
        ax.fill([R, R + 0.5, R + 0.25], [0, 0, h / 2], color=PIPER['HCO3'], zorder=0, alpha=a, lw=0)
        ax.fill([R + 0.5, R + 1.0, R + 0.75], [0, 0, h / 2], color=PIPER['Cl'], zorder=0, alpha=a, lw=0)
        ax.fill([R + 0.25, R + 0.75, R + 0.5], [h / 2, h / 2, h], color=PIPER['SO4'], zorder=0, alpha=a, lw=0)
        # diamond
        x0 = 0.5 + offset + 0.25
        yc = h + offset * t3   # middle of the diamond
        s60 = 0.5 * np.sin(np.pi / 3)
        ax.fill([x0, x0 + 0.5, x0 + 0.25], [yc - s60, yc - s60, yc], color=PIPER['Mixed'], zorder=0, lw=0)
        ax.fill([x0, x0 + 0.25, x0 + 0.5], [yc + s60, yc, yc + s60], color=PIPER['Mixed'], zorder=0, lw=0)
        ax.fill([x0, x0 + 0.5, x0 + 0.25], [yc - s60 + h, yc - s60 + h, yc + h],
                color=PIPER['Ca-Cl'], zorder=0, alpha=a, lw=0)
        ax.fill([x0, x0 + 0.25, x0 + 0.5], [yc + s60 - h, yc - h, yc + s60 - h],
                color=PIPER['Na-HCO3'], zorder=0, alpha=a, lw=0)
        ax.fill([0.5 + offset, x0, 0.5 + offset + 0.5, x0],
                [1.2 * h, 1.7 * h, 1.2 * h, 0.7 * h], color=PIPER['Mg-HCO3'], zorder=0, alpha=a, lw=0)
        ax.fill([1.0 + offset, x0 + 0.5, 1.5 + offset, x0 + 0.5],
                [1.2 * h, 1.7 * h, 1.2 * h, 0.7 * h], color=PIPER['Na-Cl'], zorder=0, alpha=a, lw=0)

        if show_types:
            kw = dict(fontweight='bold', fontsize=10, ha='center')
            ax.text(x0 + 0.25, yc + h, 'Ca-Cl', color=PIPER['Ca-Cl'], **kw)
            ax.text(0.3 + offset, 1.2 * h, 'Mg-HCO$_3$', color=PIPER['Mg-HCO3'], **kw)
            ax.text(1.62 + offset, 1.2 * h - 0.02, 'Na-Cl', color=PIPER['Na-Cl'], **kw)
            ax.text(1.0 + offset, yc + s60 - h - 0.49, 'Na-HCO$_3$', color=PIPER['Na-HCO3'], **kw)

        # plot the samples
        styles = group_styles(v)
        hollow = hollow_mask(v, ['piper'])
        cvals = v[color_by] if color_by else None
        mappable = None
        for xc, yc_ in [('piper_cat_x', 'piper_cat_y'), ('piper_an_x', 'piper_an_y'),
                        ('piper_d_x', 'piper_d_y')]:
            m = _scatter(ax, v, v[xc], v[yc_], hollow, styles, cvals=cvals, size=28,
                         legend=False, zorder=100)
            mappable = m or mappable

        plotted = v['piper_d_x'].notna()
        if color_by and mappable is not None:
            cb = fig.colorbar(mappable, ax=ax, orientation='vertical', fraction=0.025, pad=0.02,
                              shrink=0.5)
            cb.set_label(pretty(color_by))
            handles = [Line2D([], [], ls='', marker=mk, mfc='0.6', mec='black', mew=0.5, ms=5, label=lab)
                       for lab, (_, mk) in styles.items()]
            if (hollow & plotted).any():
                handles.append(Line2D([], [], ls='', marker='o', mfc='none', mec='0.3', ms=5,
                                      label='Filled input (hollow)'))
        else:
            handles = _legend_handles(styles, (hollow & plotted).any())
        ax.legend(handles=handles, frameon=False, labelspacing=0.3, handletextpad=0.3,
                  loc='upper right', bbox_to_anchor=(1.02, 1.0))

        if water_legend:
            ax2 = fig.add_axes([0.0, 0.52, 0.2, 0.38])
            ax2.axis('off')
            for mk, key, lab in [('^', 'Mg', 'Mg'), ('^', 'Ca', 'Ca'), ('^', 'NaK', 'Na + K'),
                                 ('^', 'HCO3', 'HCO$_3$ + CO$_3$'), ('^', 'Cl', 'Cl'),
                                 ('^', 'SO4', 'SO$_4$'), ('^', 'Mixed', 'Mixed'),
                                 ('^', 'Ca-Cl', 'Ca-Cl'), ('d', 'Na-Cl', 'Na-Cl'),
                                 ('v', 'Na-HCO3', 'Na-HCO$_3$'), ('d', 'Mg-HCO3', 'Mg-HCO$_3$')]:
                ax2.plot([], [], mk, ms=9, mfc=PIPER[key], alpha=1.0 if key == 'Mixed' else a,
                         mew=0, label=lab)
            ax2.legend(title='Fields', frameon=False, labelspacing=0.8, loc='upper left')
    return fig


# %% Schoeller
def schoeller(v):
    # Schoeller diagram (meq/L, log scale)
    ions = ['Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'HCO3']
    ticks = ['Ca$^{2+}$', 'Mg$^{2+}$', 'Na$^+$', 'K$^+$', 'Cl$^-$', 'SO$_4^{2-}$', 'HCO$_3^-$']
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, 9 * CM), dpi=150)
        ax = fig.add_axes([0.09, 0.1, 0.68, 0.86])
        styles = group_styles(v)
        hollow = hollow_mask(v, ['HCO3 (meq/L)'])
        xs = np.arange(1, 8)
        for i in v.index:
            y = np.array([v.at[i, f'{a} (meq/L)'] for a in ions], dtype=float)
            y[y <= 0] = np.nan
            if np.isfinite(y).sum() < 2:
                continue
            col, mk = styles[v.at[i, 'Label']]
            ax.plot(xs, y, color=col, marker=mk, ms=3.5, lw=0.8, alpha=0.8, mec='black', mew=0.3,
                    ls='--' if hollow[i] else '-')
        ax.set_yscale('log')
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks)
        ax.set_xlim(0.8, 7.2)
        for x in xs:
            ax.axvline(x, lw=0.5, color='0.7', ls='--', zorder=0)
        ax.set_ylabel('Concentration (meq/L)')
        handles = [Line2D([], [], color=col, marker=mk, ms=4, mec='black', mew=0.3, lw=0.8, label=lab)
                   for lab, (col, mk) in styles.items()]
        if hollow.any():
            handles.append(Line2D([], [], color='0.3', ls='--', lw=0.8, label='Filled HCO$_3$ (dashed)'))
        ax.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    return fig


# %% Stiff
def stiff(v, samples, ncols=4):
    # grid of Stiff diagrams, all on the same meq/L scale
    sub = v[v['Sample'].isin(samples)]
    n = len(sub)
    if n == 0:
        return None
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    styles = group_styles(v)
    need = ['Na+K (meq/L)', 'Ca (meq/L)', 'Mg (meq/L)', 'Cl (meq/L)', 'HCO3 (meq/L)', 'SO4 (meq/L)']
    cmax = float(np.nanmax(sub[need].values)) if sub[need].notna().any().any() else 1.0
    hollow = hollow_mask(v, ['HCO3 (meq/L)'])
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, (3.6 * nrows + 1.2) * CM), dpi=150)
        axs = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
        fig.subplots_adjust(left=0.07, right=0.93, top=1 - 0.5 / (3.6 * nrows + 1.2),
                            bottom=1.3 / (3.6 * nrows + 1.2), wspace=0.55, hspace=0.75)
        for ax, i in zip(axs, sub.index):
            r = sub.loc[i, need].fillna(0).astype(float)
            x = [-r.iloc[0], -r.iloc[1], -r.iloc[2], r.iloc[5], r.iloc[4], r.iloc[3], -r.iloc[0]]
            y = [3, 2, 1, 1, 2, 3, 3]
            col, _ = styles[sub.at[i, 'Label']]
            if hollow[i]:
                ax.fill(x, y, facecolor='none', edgecolor=col, lw=1.0, hatch='///')
            else:
                ax.fill(x, y, facecolor=col, alpha=0.6, edgecolor='black', lw=0.7)
            ax.plot([0, 0], [0.8, 3.2], color='black', lw=0.6)
            ax.set_xlim(-cmax * 1.05, cmax * 1.05)
            ax.set_ylim(0.7, 3.3)
            ax.set_yticks([])
            for sp in ['left', 'right', 'top']:
                ax.spines[sp].set_visible(False)
            tk = np.array([-cmax, 0, cmax])
            ax.set_xticks(tk)
            ax.set_xticklabels([f'{abs(t):.3g}' for t in tk], fontsize=6)
            ax.set_title(str(sub.at[i, 'Sample']), fontsize=8, pad=2)
            for yy, lab in zip([3, 2, 1], ['Na+K', 'Ca', 'Mg']):
                ax.text(-cmax * 1.08, yy, lab, ha='right', va='center', fontsize=6)
            for yy, lab in zip([3, 2, 1], ['Cl', 'HCO$_3$', 'SO$_4$']):
                ax.text(cmax * 1.08, yy, lab, ha='left', va='center', fontsize=6)
        for ax in axs[n:]:
            ax.axis('off')
        fig.text(0.5, 0.01, 'meq/L (common scale). Hatched = filled HCO$_3$.', ha='center', fontsize=7)
    return fig


# %% WQChartPy diagrams
WQ_DIAGRAMS = {'Durov': 'durvo', 'Gibbs': 'gibbs', 'Chadha': 'chadha', 'HFE-D': 'hfed'}


def wqchartpy_figure(v, kind):
    # Durov, Gibbs, Chadha, HFE-D from WQChartPy as is
    # WQChartPy saves a file itself, so send that to a temp folder and keep the figure
    import importlib
    import matplotlib.pyplot as plt

    styles = group_styles(v)
    w = pd.DataFrame({'Sample': v['Sample'], 'Label': v['Label']})
    w['Color'] = [styles[l][0] for l in v['Label']]
    w['Marker'] = [styles[l][1] for l in v['Label']]
    w['Size'] = 30
    w['Alpha'] = 0.8
    w['pH'] = v['pH']
    for a in ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4']:
        w[a] = v[f'{a} (mg/L)']
    w['CO3'] = w['CO3'].fillna(0)
    w['K'] = w['K'].fillna(0)
    w['TDS'] = v['TDS (mg/L)']
    w = w.dropna(subset=['pH', 'Ca', 'Mg', 'Na', 'HCO3', 'Cl', 'SO4', 'TDS']).reset_index(drop=True)
    note = ''
    if kind == 'Durov':
        # the WQChartPy Durov TDS axis stops at 4000 mg/L, so drop anything above
        over = w['TDS'] > 4000
        if over.any():
            note = f'{int(over.sum())} sample(s) with TDS above 4000 mg/L are not shown: the WQChartPy Durov has a fixed TDS axis.'
            w = w[~over].reset_index(drop=True)
    if w.empty:
        return None, note

    module = importlib.import_module('wqchartpy.' + WQ_DIAGRAMS[kind])
    before = set(plt.get_fignums())
    with tempfile.TemporaryDirectory() as tmp:
        module.plot(w, unit='mg/L', figname=os.path.join(tmp, 'fig'), figformat='png')
    new = [n for n in plt.get_fignums() if n not in before]
    if not new:
        return None, note
    fig = plt.figure(new[-1])
    for n in new[:-1]:
        plt.close(n)
    return fig, note


# %% gallery
# x and y are column names in v
# lines:
#   ('slope', m, label)   y = m*x
#   ('h', value, label)   horizontal line
#   ('v', value, label)   vertical line
#   ('sw', ion)           seawater ratio line (ion vs Cl)
def _sw(ion):
    return SEAWATER_MMOL[ion] / SEAWATER_MMOL['Cl']


def _ion_vs_cl(ion):
    lines = [('sw', ion)] if ion in SEAWATER_MMOL else []
    if ion == 'Na':
        lines.append(('slope', 1.0, '1:1 (halite)'))
    return dict(x='Cl (mmol/L)', y=f'{ion} (mmol/L)', lines=lines, conc=True)


def _ratio_vs_cl(ratio, sw=None):
    lines = [('h', sw, 'Seawater')] if sw else []
    if ratio == 'Na/Cl':
        lines.append(('h', 1.0, 'Halite'))
    return dict(x='Cl (mmol/L)', y=ratio, lines=lines, xlog=True)


_SI_MAIN = ['Calcite', 'Dolomite', 'Gypsum', 'Halite', 'Fluorite', 'Chalcedony']
_SI_MORE = ['Aragonite', 'Anhydrite', 'Quartz', 'Barite', 'Celestite', 'Strontianite']
_SI_ALL = _SI_MAIN + _SI_MORE
# SI within +/- this is treated as near equilibrium (shaded band)
SI_BAND = 0.5
_ZERO = [('h', 0.0, 'Saturation')]
_BAND = (-SI_BAND, SI_BAND, f'SI = 0 $\\pm$ {SI_BAND}')


def _si(x, mineral, **kw):
    return dict(x=x, y=f'SI {mineral}', lines=_ZERO, band=_BAND, **kw)


_NACL_LINES = [('h', 1.0, 'Halite dissolution (1:1)'), ('h', _sw('Na'), 'Seawater / rainfall (0.86)')]
_NACL_NOTES = ['Above 1: silicate weathering or cation exchange',
               'Below 0.86: Na lost (reverse exchange)']

GALLERY = {
    'Major ions against Cl': [_ion_vs_cl(i) for i in ['Na', 'K', 'Ca', 'Mg', 'HCO3', 'SO4']],
    'Minor ions against Cl': [_ion_vs_cl(i) for i in ['Br', 'F', 'NO3', 'Sr', 'SiO2', 'Ba']],
    'Ion/Cl ratios against Cl': [
        dict(x='Cl (mmol/L)', y='Na/Cl', lines=_NACL_LINES, notes=_NACL_NOTES, xlog=True, include=True),
        _ratio_vs_cl('K/Cl', _sw('K')),
        _ratio_vs_cl('Ca/Cl', _sw('Ca')), _ratio_vs_cl('Mg/Cl', _sw('Mg')),
        _ratio_vs_cl('SO4/Cl', _sw('SO4')), _ratio_vs_cl('HCO3/Cl', _sw('HCO3')),
        dict(x='Cl (mmol/L)', y='Cl/Br', lines=[('h', 1.0 / _sw('Br'), 'Seawater (~650)')], xlog=True, ylog=True,
             include=True,
             notes=['Rising with Cl: halite dissolution',
                    'Flat with rising Cl: evaporation']),
        _ratio_vs_cl('Sr/Cl', _sw('Sr')),
        _ratio_vs_cl('F/Cl', _sw('F')), _ratio_vs_cl('NO3/Cl'),
        _ratio_vs_cl('Ca/Mg', SEAWATER_MMOL['Ca'] / SEAWATER_MMOL['Mg']),
        _ratio_vs_cl('Na/K', SEAWATER_MMOL['Na'] / SEAWATER_MMOL['K']),
    ],
    'Source of solutes': [
        dict(x='TDS (mg/L)', y='Na/Cl', lines=_NACL_LINES, notes=_NACL_NOTES, xlog=True, include=True),
        dict(x='Na/Cl', y='HCO3/sum anions (meq)', include=True,
             lines=[('v', 1.0, 'Halite dissolution (Na/Cl = 1)'), ('v', _sw('Na'), 'Seawater / rainfall'),
                    ('h', 0.5, None)],
             notes=['High HCO$_3$, Na/Cl > 1: silicate weathering',
                    'Low HCO$_3$, Na/Cl near 1: evaporites, marine salts']),
        dict(x='TDS (mg/L)', y='Ca/Mg', xlog=True, include=True,
             lines=[('h', 1.0, 'Dolomite (Ca/Mg = 1)'), ('h', 2.0, 'Above 2: calcite or silicates')]),
        dict(x='TDS (mg/L)', y='HCO3/SiO2', xlog=True, ylog=True, include=True,
             lines=[('h', 10.0, 'Above 10: carbonate weathering'), ('h', 5.0, 'Below 5: silicate weathering')]),
        dict(x='TDS (mg/L)', y='Na/(Na+Cl) (meq)', xlog=True, include=True,
             lines=[('h', 0.5, 'Halite (0.5)')],
             notes=['> 0.5: other Na source (albite, ion exchange)',
                    '< 0.5: reverse softening, seawater']),
        dict(x='TDS (mg/L)', y='Ca/(Ca+SO4) (meq)', xlog=True, include=True,
             lines=[('h', 0.5, 'Gypsum (0.5)')],
             notes=['> 0.5: Ca from carbonates or silicates',
                    '< 0.5: Ca removed (exchange, calcite)']),
        dict(x='TDS (mg/L)', y='Mg/(Ca+Mg) (meq)', xlog=True, include=True,
             lines=[('h', 0.5, 'Dolomite (0.5)')],
             notes=['< 0.5: limestone-dolomite',
                    '> 0.5: dolomite dissolution, calcite precipitation, or seawater']),
        dict(x='TDS (mg/L)', y='SiO2/(Na+K-Cl) (mmol/meq)', xlog=True, include=True,
             lines=[('h', 1.0, 'Below 1: cation exchange'), ('h', 2.0, 'Above 2: ferromagnesian minerals')],
             notes=['1 to 2: albite weathering']),
        dict(x='TDS (mg/L)', y='HCO3/Cl', lines=[], xlog=True, ylog=True),
        dict(x='TDS (mg/L)', y='SO4/Cl', lines=[('h', _sw('SO4'), 'Seawater')], xlog=True, ylog=True),
        dict(x='TDS (mg/L)', y='Cl/Br', lines=[('h', 1.0 / _sw('Br'), 'Seawater (~650)')], xlog=True, ylog=True,
             include=True),
        dict(x='TDS (mg/L)', y='(Ca+Mg)/(HCO3+SO4)', lines=[('h', 1.0, '1:1')], xlog=True, include=True,
             notes=['> 1: reverse exchange',
                    '< 1: cation exchange (Ca, Mg lost)']),
    ],
    'Weathering end members (Gaillardet)': [
        # end-member boxes from Gaillardet et al. (1999), molar ratios
        dict(x='Ca/Na', y='HCO3/Na', xlog=True, ylog=True, include=True, lines=[],
             boxes=[(0.08, 0.26, 0.05, 0.6, 'Evaporites'), (0.2, 0.5, 1.0, 3.0, 'Silicates'),
                    (30, 70, 80, 160, 'Carbonates')]),
        dict(x='Ca/Na', y='Mg/Na', xlog=True, ylog=True, include=True, lines=[],
             boxes=[(0.08, 0.26, 0.01, 0.03, 'Evaporites'), (0.2, 0.5, 0.12, 0.36, 'Silicates'),
                    (30, 70, 5, 15, 'Carbonates')]),
    ],
    'Gibbs (controls on chemistry)': [
        dict(x='Na/(Na+Ca) (mg/L basis)', y='TDS (mg/L)', ylog=True, lines=[], gibbs=True),
        dict(x='Cl/(Cl+HCO3) (mg/L basis)', y='TDS (mg/L)', ylog=True, lines=[], gibbs=True),
    ],
    'Water-rock processes': [
        dict(x='Cl (meq/L)', y='Na (meq/L)', lines=[('slope', 1.0, '1:1 (halite)'), ('slope', _sw('Na'), 'Seawater')]),
        dict(x='HCO3+SO4 (meq/L)', y='Ca+Mg (meq/L)', lines=[('slope', 1.0, '1:1 (carbonate + gypsum)')],
             notes=['Above: reverse exchange',
                    'Below: cation exchange or silicate weathering']),
        dict(x='HCO3 (meq/L)', y='Ca+Mg (meq/L)', lines=[('slope', 1.0, '1:1 (carbonate dissolution)')]),
        dict(x='SO4 (meq/L)', y='Ca (meq/L)', lines=[('slope', 1.0, '1:1 (gypsum)')]),
        dict(x='HCO3 (meq/L)', y='Ca (meq/L)', lines=[('slope', 1.0, '1:1 (calcite)'), ('slope', 0.5, '1:2 (dolomite)')]),
        dict(x='Ca (mmol/L)', y='Mg (mmol/L)', lines=[('slope', 1.0, '1:1 (dolomite)')]),
        dict(x='Ca+Mg-HCO3-SO4 (meq/L)', y='Na+K-Cl (meq/L)', lines=[('slope', -1.0, 'Slope -1 (cation exchange)')], origin=True),
        dict(x='Cl (mmol/L)', y='CAI-1', lines=[('h', 0.0, 'No exchange')], xlog=True, include=True,
             notes=['> 0: reverse exchange',
                    '< 0: cation exchange']),
        dict(x='SO4 (meq/L)', y='Ca+Mg-HCO3 (meq/L)', lines=[('slope', 1.0, '1:1 (gypsum)')], origin=True),
        dict(x='SiO2 (mmol/L)', y='HCO3 (mmol/L)', lines=[]),
        dict(x='Na (mmol/L)', y='SiO2 (mmol/L)', lines=[]),
        dict(x='SiO2 (mmol/L)', y='Na+K-Cl (meq/L)', lines=[('h', 0.0, 'Na + K = Cl')]),
    ],
    'Saturation indices against TDS': [_si('TDS (mg/L)', m, xlog=True, include=True) for m in _SI_ALL],
    'Saturation indices against Cl': [_si('Cl (mmol/L)', m, xlog=True, include=True) for m in _SI_ALL],
    'Saturation indices against pH': [_si('pH', m, include=True) for m in _SI_ALL],
    'Saturation indices against their own ions': [
        _si('Ca (mmol/L)', 'Calcite', xlog=True, include=True), _si('HCO3 (mmol/L)', 'Calcite', xlog=True, include=True),
        _si('Mg (mmol/L)', 'Dolomite', xlog=True, include=True), _si('Ca/Mg', 'Dolomite', xlog=True, include=True),
        _si('SO4 (mmol/L)', 'Gypsum', xlog=True, include=True), _si('Ca (mmol/L)', 'Gypsum', xlog=True, include=True),
        _si('F (mmol/L)', 'Fluorite', xlog=True, include=True), _si('Ca (mmol/L)', 'Fluorite', xlog=True, include=True),
        _si('SiO2 (mmol/L)', 'Chalcedony', xlog=True, include=True), _si('SiO2 (mmol/L)', 'Quartz', xlog=True, include=True),
        _si('Ba (mmol/L)', 'Barite', xlog=True, include=True), _si('Sr (mmol/L)', 'Celestite', xlog=True, include=True),
    ],
    'Saturation indices against each other': [
        dict(x='SI Calcite', y='SI Dolomite', lines=[('h', 0.0, None), ('v', 0.0, None), ('slope', 2.0, '2:1')],
             origin=True, include=True),
        dict(x='SI Gypsum', y='SI Calcite', lines=[('h', 0.0, None), ('v', 0.0, None)], include=True,
             notes=['Calcite near 0 while gypsum rises: dedolomitisation']),
        dict(x='SI Gypsum', y='SI Dolomite', lines=[('h', 0.0, None), ('v', 0.0, None)], include=True),
        dict(x='SI Calcite', y='SI Aragonite', lines=[('h', 0.0, None), ('v', 0.0, None)], include=True),
        dict(x='SI Gypsum', y='SI Anhydrite', lines=[('h', 0.0, None), ('v', 0.0, None)], include=True),
        dict(x='SI Chalcedony', y='SI Quartz', lines=[('h', 0.0, None), ('v', 0.0, None)], include=True),
    ],
    'Carbonate system': [
        dict(x='pH', y='log pCO2', lines=[('h', -3.38, 'Atmosphere (~420 ppm)'), ('h', -1.5, 'Typical soil zone (-1.5)')],
             include=True),
        _si('log pCO2', 'Calcite', include=True),
        _si('log pCO2', 'Dolomite', include=True),
        dict(x='HCO3 (mmol/L)', y='log pCO2', lines=[('h', -3.38, 'Atmosphere')], include=True),
        _si('HCO3 (mmol/L)', 'Calcite', include=True),
        dict(x='TDS (mg/L)', y='log pCO2', lines=[('h', -3.38, 'Atmosphere')], xlog=True, include=True),
    ],
    'Irrigation quality': [
        dict(x='EC', y='SAR', lines=[('h', 10.0, 'S1/S2 at low EC (10)'), ('h', 18.0, 'S2/S3 (18)'), ('h', 26.0, 'S3/S4 (26)')],
             xlog=True),
        dict(x='EC', y='Na (%)', lines=[('h', 60.0, '60 %')], xlog=True, include=True),
        dict(x='TDS (mg/L)', y='SAR', lines=[], xlog=True),
    ],
    'Quality checks': [
        dict(x='TDS (mg/L)', y='CBE (%)', lines=[('h', 5.0, '+/- 5 %'), ('h', -5.0, None)], xlog=True, include=True),
        dict(x='Sum anions (meq/L)', y='Sum cations (meq/L)', lines=[('slope', 1.0, '1:1')]),
        dict(x='TDS calc (mg/L)', y='TDS (mg/L)', lines=[('slope', 1.0, '1:1')]),
        dict(x='EC', y='TDS (mg/L)', lines=[('slope', 0.55, 'TDS = 0.55 EC'), ('slope', 0.75, 'TDS = 0.75 EC')]),
        dict(x='EC', y='SC calc (uS/cm)', lines=[('slope', 1.0, '1:1')]),
        dict(x='Cl (mmol/L)', y='Ionic strength (mol/kgw)', lines=[('h', 0.5, 'phreeqc.dat caution')], xlog=True, ylog=True),
    ],
}

_LINE_STYLES = ['--', ':', '-.']


def available(v, panel):
    # only draw a panel if both variables have data
    x, y = panel['x'], panel['y']
    if x not in v.columns or y not in v.columns:
        return False
    return bool((v[x].notna() & v[y].notna()).any())


def _draw_panel(ax, v, panel, styles, log_conc, cvals=None):
    x, y = v[panel['x']], v[panel['y']]
    xlog = panel.get('xlog', False) or (log_conc and panel.get('conc', False))
    ylog = panel.get('ylog', False) or (log_conc and panel.get('conc', False))
    if xlog:
        x = x.where(x > 0)
    if ylog:
        y = y.where(y > 0)
    hollow = hollow_mask(v, [panel['x'], panel['y']])
    mappable = _scatter(ax, v, x, y, hollow, styles, cvals=cvals, legend=False)

    ok = x.notna() & y.notna()
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    # set the axis limits from the data before adding reference lines
    if ok.any() and not xlog and not ylog and not panel.get('origin', False):
        if x[ok].min() >= 0:
            ax.set_xlim(left=0)
        if y[ok].min() >= 0 and not panel['y'].startswith(('SI ', 'log ', 'CBE', 'CAI')):
            ax.set_ylim(bottom=0)
    ax.margins(0.08)
    ax.autoscale_view()
    xl, yl = list(ax.get_xlim()), list(ax.get_ylim())
    # stretch the limits so the reference lines and boxes are in view
    if panel.get('include', False):
        ys = [l[1] for l in panel.get('lines', []) if l[0] == 'h']
        xv = [l[1] for l in panel.get('lines', []) if l[0] == 'v']
        if 'band' in panel:
            ys += list(panel['band'][:2])
        for bx in panel.get('boxes', []):
            xv += [bx[0], bx[1]]
            ys += [bx[2], bx[3]]
        for vals, lim, log in [(xv, xl, xlog), (ys, yl, ylog)]:
            if vals:
                lo, hi = min(vals), max(vals)
                if log:
                    lim[0], lim[1] = min(lim[0], lo / 1.5), max(lim[1], hi * 1.5)
                else:
                    pad = 0.06 * (max(lim[1], hi) - min(lim[0], lo))
                    lim[0], lim[1] = min(lim[0], lo - pad), max(lim[1], hi + pad)
    if panel.get('gibbs', False):
        xl = [0, 1]
        yl = [min(yl[0], 5), max(yl[1], 50000)]
        for fx, fy, txt in [(0.97, 0.08, 'Precipitation\ndominance'), (0.03, 0.5, 'Rock\ndominance'),
                            (0.97, 0.92, 'Evaporation\ndominance')]:
            ax.text(fx, fy, txt, transform=ax.transAxes, fontsize=6, color='0.35', style='italic',
                    ha='right' if fx > 0.5 else 'left', va='center', zorder=2)
    if 'band' in panel:
        ax.axhspan(panel['band'][0], panel['band'][1], color='0.85', alpha=0.6, lw=0, zorder=0,
                   label=panel['band'][2])
    for bx in panel.get('boxes', []):
        ax.fill([bx[0], bx[1], bx[1], bx[0]], [bx[2], bx[2], bx[3], bx[3]], facecolor='0.85',
                edgecolor='0.4', lw=0.6, alpha=0.6, zorder=0)
        mid = np.sqrt(bx[0] * bx[1]) if xlog else 0.5 * (bx[0] + bx[1])
        ax.text(mid, bx[2], bx[4], fontsize=6, color='0.3', ha='center', va='top', zorder=2)
    xs = np.geomspace(max(xl[0], 1e-12), xl[1], 50) if xlog else np.linspace(xl[0], xl[1], 50)

    k = 0
    for line in panel.get('lines', []):
        kind = line[0]
        ls = _LINE_STYLES[k % len(_LINE_STYLES)]
        if kind == 'sw':
            ax.plot(xs, xs * _sw(line[1]), ls=ls, color='0.35', lw=0.8, zorder=1, label='Seawater ratio')
        elif kind == 'slope':
            ax.plot(xs, xs * line[1], ls=ls, color='0.35', lw=0.8, zorder=1, label=line[2])
        elif kind == 'h':
            ax.axhline(line[1], ls=ls, color='0.35', lw=0.8, zorder=1, label=line[2])
        elif kind == 'v':
            ax.axvline(line[1], ls=ls, color='0.35', lw=0.8, zorder=1, label=line[2])
        k += 1
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    if panel.get('origin', False):
        ax.axhline(0, color='0.8', lw=0.5, zorder=0)
        ax.axvline(0, color='0.8', lw=0.5, zorder=0)

    ax.set_xlabel(pretty(panel['x']) + (' (molar)' if '/' in panel['x'] and '(' not in panel['x'] else ''))
    ax.set_ylabel(pretty(panel['y']) + (' (molar)' if '/' in panel['y'] and '(' not in panel['y'] else ''))
    # plain numbers on log axes
    for axis, log, lim in [(ax.xaxis, xlog, xl), (ax.yaxis, ylog, yl)]:
        if log:
            axis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val:g}'))
            if lim[0] > 0 and np.log10(lim[1] / lim[0]) < 1.2:
                axis.set_minor_formatter(FuncFormatter(
                    lambda val, pos: f'{val:g}' if str(f'{val:e}')[0] in '25' else ''))
            else:
                axis.set_minor_formatter(NullFormatter())
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h_, l_) for h_, l_ in zip(handles, labels) if l_ and not l_.startswith('_')]
    # interpretation notes go in the legend as text-only lines
    keep += [(Line2D([], [], ls='', marker=''), n) for n in panel.get('notes', [])]
    if keep:
        ax.legend(*zip(*keep), frameon=False, fontsize=6, loc='best', handlelength=2.2, labelspacing=0.3)
    return mappable, hollow & ok


def gallery_page(v, panels, log_conc=False, color_by=None):
    # up to 6 panels (2 x 3) with one legend along the bottom
    panels = [p for p in panels if available(v, p)]
    if not panels:
        return None
    n = len(panels)
    nrows = int(np.ceil(n / 2))
    height = 4.6 * nrows + 1.6
    styles = group_styles(v)
    cvals = v[color_by] if color_by else None
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, height * CM), dpi=150)
        axs = np.atleast_1d(fig.subplots(nrows, 2)).ravel()
        fig.subplots_adjust(left=0.1, right=0.86 if color_by else 0.98, top=1 - 0.3 / height,
                            bottom=2.2 / height, hspace=0.42, wspace=0.3)
        any_hollow = False
        mappable = None
        for ax, p in zip(axs, panels):
            m, hol = _draw_panel(ax, v, p, styles, log_conc, cvals=cvals)
            mappable = m or mappable
            any_hollow = any_hollow or bool(hol.any())
        for ax in axs[n:]:
            ax.axis('off')
        if color_by and mappable is not None:
            cax = fig.add_axes([0.89, 0.25, 0.018, 0.5])
            fig.colorbar(mappable, cax=cax).set_label(pretty(color_by))
            handles = [Line2D([], [], ls='', marker=mk, mfc='0.6', mec='black', mew=0.5, ms=5, label=lab)
                       for lab, (_, mk) in styles.items()]
            if any_hollow:
                handles.append(Line2D([], [], ls='', marker='o', mfc='none', mec='0.3', ms=5,
                                      label='Filled input (hollow)'))
        else:
            handles = _legend_handles(styles, any_hollow)
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=min(4, len(handles)), frameon=False)
    return fig


# %% custom plot
def custom(v, x, y, xlog=False, ylog=False, color_by=None, one_to_one=False, zero_line=False,
           annotate=False):
    # user picks x and y
    panel = dict(x=x, y=y, xlog=xlog, ylog=ylog, lines=[], origin=True)
    if one_to_one:
        panel['lines'].append(('slope', 1.0, '1:1'))
    if zero_line:
        panel['lines'].append(('h', 0.0, 'y = 0'))
    styles = group_styles(v)
    cvals = v[color_by] if color_by else None
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, 10 * CM), dpi=150)
        ax = fig.add_axes([0.1, 0.13, 0.58, 0.83])
        mappable, hol = _draw_panel(ax, v, panel, styles, False, cvals=cvals)
        if annotate:
            ok = v[x].notna() & v[y].notna()
            for i in v.index[ok]:
                ax.annotate(str(v.at[i, 'Sample']), (v.at[i, x], v.at[i, y]), fontsize=5,
                            xytext=(3, 3), textcoords='offset points')
        if color_by and mappable is not None:
            cax = fig.add_axes([0.72, 0.13, 0.015, 0.35])
            fig.colorbar(mappable, cax=cax).set_label(pretty(color_by))
            handles = [Line2D([], [], ls='', marker=mk, mfc='0.6', mec='black', mew=0.5, ms=5, label=lab)
                       for lab, (_, mk) in styles.items()]
            if hol.any():
                handles.append(Line2D([], [], ls='', marker='o', mfc='none', mec='0.3', ms=5,
                                      label='Filled input (hollow)'))
        else:
            handles = _legend_handles(styles, bool(hol.any()))
        fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.69, 0.96))
    return fig


# %% png for download
def fig_png(fig, dpi=300):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()
