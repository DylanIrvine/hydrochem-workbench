# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# hydrochem_workbench/plots_quality.py
# figures for hydrochem_workbench/quality.py: the USSL and Wilcox irrigation diagrams,
# the analysis checks and the CCME index

# %%
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from hydrochem_workbench import plots, quality as Q
from hydrochem_workbench.chem import hollow_mask

CM = plots.CM
FIELD = '0.35'       # class boundaries and their labels


def _points(ax, v, x, y, cols):
    styles = plots.group_styles(v)
    hol = hollow_mask(v, cols)
    plots._scatter(ax, v, x, y, hol, styles, legend=False)
    return styles, bool((hol & x.notna() & y.notna()).any())


def _legend(fig, v, styles, any_hollow, extra=(), x=0.69, y=0.96):
    used = {lab: st for lab, st in styles.items() if (v['Label'] == lab).any()}
    handles = plots._legend_handles(used, any_hollow) + plots.ring_handles(v) + list(extra)
    fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(x, y))


# %% irrigation
def ussl(v):
    # US Salinity Laboratory diagram (Richards, 1954): SAR against EC, with the salinity (C1 to C4) and
    # sodium (S1 to S4) hazard classes. Returns None if no sample has both
    ec, source = Q.ec_for_irrigation(v)
    sar = v['SAR'].astype(float)
    ok = ec.notna() & sar.notna() & (ec > 0)
    if not ok.any():
        return None
    with matplotlib.rc_context(plots.STYLE):
        fig = Figure(figsize=(17 * CM, 10.5 * CM), dpi=150)
        ax = fig.add_axes([0.1, 0.13, 0.56, 0.8])
        x_hi = max(5000.0, 1.3 * float(ec[ok].max()))
        x_lo = min(100.0, 0.8 * float(ec[ok].min()))
        y_hi = max(30.0, 1.1 * float(sar[ok].max()))
        for lim in Q.USSL_EC:
            ax.axvline(lim, color=FIELD, lw=0.6)
        xs = np.geomspace(x_lo, x_hi, 100)
        for a, b in Q.USSL_SAR:
            ax.plot(xs, np.clip(a - b * np.log10(xs), 0, None), color=FIELD, lw=0.6)
        edges = [x_lo] + Q.USSL_EC + [x_hi]
        for k in range(4):
            mid = np.sqrt(edges[k] * edges[k + 1])
            ax.text(mid, y_hi * 0.985, f'C{k + 1}', ha='center', va='top', color=FIELD, fontsize=8)
        for k, name in enumerate(['S1', 'S2', 'S3', 'S4']):
            lo = 0 if k == 0 else max(Q.USSL_SAR[k - 1][0] - Q.USSL_SAR[k - 1][1] * np.log10(x_lo), 0)
            hi = y_hi if k == 3 else Q.USSL_SAR[k][0] - Q.USSL_SAR[k][1] * np.log10(x_lo)
            ax.text(x_lo * 1.06, 0.5 * (lo + min(hi, y_hi)), name, ha='left', va='center', color=FIELD, fontsize=8)
        styles, hol = _points(ax, v, ec, sar, ['SAR'])
        ax.set_xscale('log')
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, y_hi)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{val:g}'))
        ax.set_xlabel('EC (µS/cm at 25 °C)' + (', calculated where not measured' if (source[ok] == 'Calculated').any()
                                              else ''))
        ax.set_ylabel('Sodium adsorption ratio, SAR ((meq/L)$^{0.5}$)')
        _legend(fig, v[ok], styles, hol)
        fig.text(0.69, 0.13, 'Salinity hazard: C1 low,\nC2 medium, C3 high, C4 very high.\nSodium hazard: S1 low,\n'
                 'S2 medium, S3 high, S4 very high.\nAfter Richards (1954).', fontsize=6, color='0.3', va='bottom')
    return fig


WILCOX_AXIS = 3500.0        # the EC axis of the original figure ends here


def wilcox(v):
    # Wilcox (1948, Fig. 1): percent sodium against EC, with the five fields of the original diagram.
    # The curves pass through the points read from the original figure (Q.WILCOX_CURVES).
    # Returns None if no sample has both
    ec, source = Q.ec_for_irrigation(v)
    na = Q.wilcox_sodium(v)
    ok = ec.notna() & na.notna() & (ec >= 0)
    if not ok.any():
        return None
    with matplotlib.rc_context(plots.STYLE):
        fig = Figure(figsize=(17 * CM, 10.5 * CM), dpi=150)
        ax = fig.add_axes([0.1, 0.13, 0.56, 0.8])
        # the axis of the original figure. More saline samples are all unsuitable and are counted in the note
        x_hi = WILCOX_AXIS
        off = ok & (ec > x_hi)
        xa, xb, xc = np.linspace(0, 750, 60), np.linspace(750, 2000, 80), np.linspace(2000, 3000, 40)
        xab = np.concatenate([xa, xb[1:]])
        lower, middle, upper = (lambda x, k=k: Q.wilcox_curve(k, x) for k in range(3))
        shades = ['1.0', '0.955', '0.90', '0.84', '0.77']
        ax.fill_between(xa, 0, lower(xa), color=shades[0], lw=0, zorder=0)
        ax.fill_between(xb, 0, lower(xb), color=shades[1], lw=0, zorder=0)
        ax.fill_between(xab, lower(xab), middle(xab), color=shades[2], lw=0, zorder=0)
        xall = np.concatenate([xab, xc[1:]])
        floor = np.concatenate([middle(xab), np.zeros(len(xc) - 1)])
        floor[len(xab) - 1] = 0.0          # the step down at 2000 uS/cm
        ax.fill_between(np.insert(xall, len(xab) - 1, 2000.0), np.insert(floor, len(xab) - 1, middle(2000.0)),
                        np.insert(upper(xall), len(xab) - 1, upper(2000.0)), color=shades[3], lw=0, zorder=0)
        ax.fill_between(xall, upper(xall), 100, color=shades[4], lw=0, zorder=0)
        ax.fill_between([3000, x_hi], 0, 100, color=shades[4], lw=0, zorder=0)
        ax.plot(xab, lower(xab), color=FIELD, lw=0.6)
        ax.plot(xab, middle(xab), color=FIELD, lw=0.6)
        ax.plot(xall, upper(xall), color=FIELD, lw=0.6)
        for x, k in ((750, 0), (2000, 1), (3000, 2)):
            ax.plot([x, x], [0, float(Q.wilcox_curve(k, x))], color=FIELD, lw=0.6)
        for x, y, name in ((375, 9, 'Excellent\nto good'), (1375, 9, 'Good to\npermissible'),
                           (2500, 9, 'Doubtful to\nunsuitable'), (0.5 * (3000 + x_hi), 9, 'Unsuitable'),
                           (1250, 69, 'Permissible\nto doubtful'), (1250, 89.5, 'Doubtful to unsuitable')):
            ax.text(x, y, name, ha='center', va='center', color=FIELD, fontsize=6, linespacing=1.1)
        styles, hol = _points(ax, v, ec.where(~off), na, ['Na (meq/L)'])
        ax.set_xlim(0, x_hi)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{val:,.0f}'))
        ax.set_xlabel('EC (µS/cm at 25 °C)' + (', calculated where not measured' if (source[ok] == 'Calculated').any()
                                              else ''))
        ax.set_ylabel('Percent sodium, Na / (Ca + Mg + Na + K), meq/L')
        _legend(fig, v[ok & ~off], styles, hol)
        note = 'Fields of Wilcox (1948, Fig. 1).\nCurves drawn through the points\nread from the original figure.'
        if off.any():
            note += f'\n{int(off.sum())} sample(s) above {x_hi:,.0f} µS/cm\nare off the axis (all unsuitable).'
        fig.text(0.69, 0.13, note, fontsize=6, color='0.3', va='bottom')
    return fig


# %% analysis checks
def qa_checks(v):
    # calculated against measured EC, and measured against calculated TDS, with the acceptance range
    # as dashed lines. Returns None if neither was measured
    panels = []
    ec, sc = v['EC'].astype(float), v['SC calc (uS/cm)'].astype(float)
    if (ec.notna() & sc.notna()).any():
        panels.append((ec, sc, 'Measured EC (µS/cm)', 'Calculated EC (µS/cm)', Q.QA_RANGES['EC calculated / measured'],
                       ['SC calc (uS/cm)']))
    tm = v['TDS (mg/L)'].astype(float).where(v['TDS is measured'].astype(bool))
    tc = v['TDS calc (mg/L)'].astype(float)
    if (tm.notna() & tc.notna()).any():
        lo, hi = Q.QA_RANGES['TDS measured / calculated']
        # drawn with the calculated value on the y axis, so the range is turned around
        panels.append((tm, tc, 'Measured TDS (mg/L)', 'Calculated TDS (mg/L)', (1 / hi, 1 / lo), ['TDS calc (mg/L)']))
    if not panels:
        return None
    with matplotlib.rc_context(plots.STYLE):
        fig = Figure(figsize=(17 * CM, 8.5 * CM), dpi=150)
        any_h = False
        for k, (x, y, xl, yl, (lo, hi), cols) in enumerate(panels):
            ax = fig.add_axes([0.08 + 0.36 * k, 0.15, 0.27, 0.78])
            styles, hol = _points(ax, v, x, y, cols)
            any_h |= hol
            ok = x.notna() & y.notna() & (x > 0) & (y > 0)
            lim = [0.8 * float(min(x[ok].min(), y[ok].min())), 1.25 * float(max(x[ok].max(), y[ok].max()))]
            ax.plot(lim, lim, color='0.3', lw=0.7)
            for f in (lo, hi):
                ax.plot(lim, [f * lim[0], f * lim[1]], color='0.3', lw=0.6, ls='--')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
        extra = [Line2D([], [], color='0.3', lw=0.7, label='1:1'),
                 Line2D([], [], color='0.3', lw=0.6, ls='--', label='Acceptance range')]
        _legend(fig, v, styles, any_h, extra, x=0.08 + 0.36 * len(panels) - 0.05)
    return fig


# %% CCME water quality index
CCME_COLOURS = {'Excellent': '#0089D0', 'Good': '#0DB14B', 'Fair': '#FCB711', 'Marginal': '#F37021', 'Poor': '#CC004C'}


def ccme_bars(table, title=''):
    # the index for each group as a bar, on the five CCME categories
    t = table[table['CCME WQI'].notna()]
    if t.empty:
        return None
    with matplotlib.rc_context(plots.STYLE):
        fig = Figure(figsize=(17 * CM, (3.2 + 0.75 * len(t)) * CM), dpi=150)
        ax = fig.add_axes([0.3, 1.5 / (3.2 + 0.75 * len(t)), 0.5, 1 - 2.1 / (3.2 + 0.75 * len(t))])
        edges = [0] + [lim for lim, _ in reversed(Q.CCME_CLASSES)][1:] + [100]
        names = [name for _, name in reversed(Q.CCME_CLASSES)]
        for k, name in enumerate(names):
            ax.axvspan(edges[k], edges[k + 1], color=CCME_COLOURS[name], alpha=0.16, lw=0, zorder=0)
            ax.text(0.5 * (edges[k] + edges[k + 1]), 1.01, name, transform=ax.get_xaxis_transform(), ha='center',
                    va='bottom', fontsize=6, color='0.3')
        y = np.arange(len(t))[::-1]
        ax.barh(y, t['CCME WQI'], height=0.6, color=[CCME_COLOURS[c] for c in t['Category']], edgecolor='black',
                linewidth=0.5, zorder=2)
        for yy, (_, row) in zip(y, t.iterrows()):
            ax.text(min(row['CCME WQI'] + 1.5, 92), yy, f"{row['CCME WQI']:.0f}" + ('' if row['Meets CCME minimum']
                                                                                      else ' *'),
                    va='center', ha='left', fontsize=7, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{g} (n = {n})" for g, n in zip(t['Group'], t['Samples'])])
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(t) - 0.4)
        ax.set_xlabel('CCME Water Quality Index' + (f': {title}' if title else ''))
        if not t['Meets CCME minimum'].all():
            fig.text(0.3, 0.01, '* fewer than 4 variables or 4 samples: below the CCME minimum', fontsize=6,
                     color='0.3', va='bottom')
    return fig
