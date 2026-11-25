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

# hydrochem_workbench/plots.py
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
import contextlib
import io
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
from matplotlib import font_manager
import matplotlib.text
import matplotlib.patches
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea
import matplotlib.image as mimage
import matplotlib.transforms as mtransforms
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from hydrochem_workbench.chem import SEAWATER_MMOL, hollow_mask, CB_RING, CB_LABEL
from hydrochem_workbench import chem, gibbs_fields

CM = 1.0 / 2.54

# no italics in maths text (e.g. HCO$_3$), on any figure
matplotlib.rcParams['mathtext.default'] = 'regular'

# %% style
# Arial if available, otherwise Liberation Sans (same letter widths, installed
# on Streamlit Cloud through packages.txt), otherwise the matplotlib default.
# Picked once here so matplotlib does not log a warning for every label
_have = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ['Arial', 'Liberation Sans'] if f in _have), 'DejaVu Sans')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': [FONT],
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

# colours and markers for the groups: the six NBC colours from green back to gold,
# then grey and three extras
PALETTE = ['#0DB14B', '#0089D0', '#6460AA', '#CC004C', '#F37021', '#FCB711', '#808080',
           '#00bdd0', '#a88958', '#0055d0']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h']
# names for the app. Filled shapes only, so hollow markers still work
MARKER_NAMES = {'o': 'Circle', 's': 'Square', '^': 'Triangle up', 'D': 'Diamond', 'v': 'Triangle down',
                'P': 'Plus (filled)', 'X': 'Cross (filled)', '<': 'Triangle left', '>': 'Triangle right',
                'h': 'Hexagon', 'p': 'Pentagon', '*': 'Star', 'd': 'Thin diamond', '8': 'Octagon'}

# Piper colours
PIPER = {'Ca': '#FCB711', 'NaK': '#F37021', 'Mg': '#CC004C',
         'HCO3': '#00bdd0', 'Cl': '#0089d0', 'SO4': '#0055d0',
         'Ca-Cl': '#7ea070', 'Na-HCO3': '#a88958', 'Mg-HCO3': '#665e8e',
         'Na-Cl': '#7a7c78', 'Mixed': '0.9'}
# how strong the field colours are (0 = white, 1 = solid). Was 0.5 originally
PIPER_ALPHA = 0.4

# subscripts for axis labels
_PRETTY = [('HCO3', 'HCO$_3$'), ('SO4', 'SO$_4$'), ('NO3', 'NO$_3$'),
           ('CO3', 'CO$_3$'), ('SiO2', 'SiO$_2$'), ('pCO2', 'pCO$_2$')]


# whole-name axis labels: units for the field columns, and the SAR definition
_LABELS = {'EC': 'EC (µS/cm)', 'Temp': 'Temperature (°C)', 'Screen_depth': 'Screen depth (m below surface)',
           'SAR': 'SAR = Na/$\\sqrt{(Ca+Mg)/2}$\n(ions in meq/L)'}


def pretty(name):
    if name in _LABELS:
        return _LABELS[name]
    out = name.replace('uS/cm', 'µS/cm')
    for raw, nice in _PRETTY:
        out = out.replace(raw, nice)
    return out


# panel letters for multipanel figures
PANEL_LABELS = {'(a)': '({})', 'a)': '{})', 'a': '{}', 'a.': '{}.', 'A': '{}'}
PANEL_LABEL_POS = ['Top right, inside the panel', 'Top left, inside the panel',
                   'Top right, above the panel', 'Top left, above the panel']


def panel_letters(axs, style=None, pos=PANEL_LABEL_POS[0], headroom=True, outside_only=False):
    # style is a key of PANEL_LABELS (None for no letters)
    # headroom: with letters inside, raise the top of each y axis so the band the
    # letter sits in is clear of data. outside_only: always put the letters above
    # the panels (for diagrams whose axes should not be changed)
    if not style or style not in PANEL_LABELS:
        return
    right = 'right' in pos
    inside = 'inside' in pos and not outside_only
    if inside and headroom:
        for ax in axs:
            y0, y1 = ax.get_ylim()
            if ax.get_yscale() == 'log' and y0 > 0 and y1 > y0:
                ax.set_ylim(y0, 10 ** (np.log10(y0) + (np.log10(y1) - np.log10(y0)) / 0.84))
            elif ax.get_yscale() == 'linear' and y1 > y0:
                ax.set_ylim(y0, y0 + (y1 - y0) / 0.84)
    for k, ax in enumerate(axs):
        letter = chr(ord('a') + k) if k < 26 else chr(ord('a') + k // 26 - 1) + chr(ord('a') + k % 26)
        if style == 'A':
            letter = letter.upper()
        text = PANEL_LABELS[style].format(letter)
        # inside: centred near (0.95, 0.92) or (0.05, 0.92) in axes fractions, on a light
        # backing so it stays readable over data
        if inside:
            x, y, va = (0.95 if right else 0.05), 0.92, 'center'
        else:
            x, y, va = (1.0 if right else 0.0), 1.02, 'bottom'
        # same size as the figure's axis labels (WQChartPy figures use larger text)
        size = max(9.0, float(ax.xaxis.label.get_size()))
        ax.text(x, y, text, transform=ax.transAxes, ha='right' if right else 'left', va=va,
                fontsize=size, zorder=30,
                bbox=dict(fc='white', ec='none', pad=1.0, alpha=0.8) if inside else None)


def group_styles(v):
    # colour and marker for each Label (Color/Marker columns override)
    styles = {}
    for k, lab in enumerate(pd.unique(v['Label'])):
        rows = v[v['Label'] == lab]
        col = next((c for c in rows['Color'] if c and mcolors.is_color_like(c)), None)
        mk = next((m for m in rows['Marker'] if m), None)
        styles[lab] = (col or PALETTE[k % len(PALETTE)], mk or MARKERS[k % len(MARKERS)])
    return styles


def _scatter(ax, v, x, y, hollow, styles, cvals=None, size=22, legend=True, zorder=10, means=False):
    # scatter by group, hollow markers where hollow is True
    # if cvals is given, colour by that variable instead (viridis)
    # means: add each group's mean as a large marker (solid points only, as in the group means table)
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
    _rings(ax, v, x, y, styles, size, zorder)
    if means:
        _group_means(ax, v, x, y, hollow, styles, cvals, size, zorder)
    return mappable


# %% charge balance rings
RING_SCALE = 2.4    # area of the ring relative to the marker it goes around
RING_WIDTH = 1.4
# the default ring colour is the one of these furthest from the group colours
RING_CHOICES = ['#FF00FF', '#00E5FF', '#FF0000', '#FFE600', '#00FF66', '#000000']


def ring_default(colours):
    # a ring colour that is not one of the group colours
    used = [mcolors.to_rgb(c) for c in colours if mcolors.is_color_like(c)]
    if not used:
        return RING_CHOICES[0]
    gap = lambda c: min(sum((a - b) ** 2 for a, b in zip(mcolors.to_rgb(c), u)) for u in used)
    return max(RING_CHOICES, key=gap)


def _rings(ax, v, x, y, styles, size=22, zorder=10):
    # a thick ring in another colour around each sample marked for its charge balance
    if CB_RING not in v.columns:
        return
    marked = (v[CB_RING] != '') & x.notna() & y.notna()
    for lab, (_, mk) in styles.items():
        g = marked & (v['Label'] == lab)
        if g.any():
            ax.scatter(x[g], y[g], marker=mk, s=size * RING_SCALE, facecolors='none',
                       edgecolors=v.loc[g, CB_RING].iloc[0], linewidths=RING_WIDTH, zorder=zorder + 0.6)


def ring_handles(v):
    # legend entry for the rings, if any sample in v is marked
    if CB_RING not in v.columns or not (v[CB_RING] != '').any():
        return []
    row = v[v[CB_RING] != ''].iloc[0]
    return [Line2D([], [], ls='', marker='o', mfc='none', mec=row[CB_RING], mew=RING_WIDTH, ms=7,
                   label=row[CB_LABEL])]


MEAN_SCALE = 5      # area of a group-mean marker relative to a sample marker


def _group_means(ax, v, x, y, hollow, styles, cvals, size, zorder):
    # mean of the plotted x and y for each group, drawn on top as a large version of its marker.
    # With colour-by, the marker takes the colour of the group's mean of that variable
    norm = None
    if cvals is not None and cvals.notna().any():
        norm = mcolors.Normalize(vmin=cvals.min(), vmax=cvals.max())
    for lab, (col, mk) in styles.items():
        g = (v['Label'] == lab) & x.notna() & y.notna() & ~hollow
        if not g.any():
            continue
        face = col
        if cvals is not None:
            cv = cvals[g].dropna()
            face = matplotlib.colormaps['viridis'](norm(cv.mean())) if (norm is not None and len(cv)) else '0.6'
        ax.scatter([x[g].mean()], [y[g].mean()], marker=mk, s=size * MEAN_SCALE, color=face,
                   edgecolors='black', linewidths=1.3, zorder=zorder + 1)


def _mean_handle():
    return Line2D([], [], ls='', marker='o', mfc='0.85', mec='black', mew=1.3, ms=8,
                  label='Group mean (large marker)')


def _legend_handles(styles, any_hollow, means=False):
    handles = [Line2D([], [], ls='', marker=mk, mfc=col, mec='black', mew=0.5, ms=5, label=lab)
               for lab, (col, mk) in styles.items()]
    if any_hollow:
        handles.append(Line2D([], [], ls='', marker='o', mfc='none', mec='0.3', mew=1.0, ms=5,
                              label='Filled input (hollow)'))
    if means:
        handles.append(_mean_handle())
    return handles


# %% uncertainty
# light grey for every uncertainty mark (interval lines, Piper clouds and outlines)
UNC_COLOUR = '0.4'
UNC_LW = 0.7
PIPER_UNC_MODES = ['Clouds', 'Outlines', 'Off']


def unc_colour(unc):
    # the colour chosen in the app (kept in the uncertainty result), or the default
    return (unc or {}).get('colour') or UNC_COLOUR


def _unc_cols(unc, v, col):
    # 5th and 95th percentiles of one column, lined up with v (None if not available)
    if unc is None or col not in unc['lo'].columns:
        return None, None
    return unc['lo'][col].reindex(v.index), unc['hi'][col].reindex(v.index)


def _interval_lines(ax, x, y, xlo=None, xhi=None, ylo=None, yhi=None, xlog=False, ylog=False,
                    colour=UNC_COLOUR):
    # light grey interval lines through each point (x and y). The axis limits are
    # left to the points, so a very wide interval runs off the panel
    segs = []
    ok = x.notna() & y.notna()
    for lo, hi, along_x, log in [(xlo, xhi, True, xlog), (ylo, yhi, False, ylog)]:
        if lo is None:
            continue
        val = x if along_x else y
        lo = lo.where(lo.notna(), val)
        hi = hi.where(hi.notna(), val)
        if log:
            lo = lo.where(lo > 0, val)
        for i in val.index[ok]:
            if lo[i] == hi[i]:
                continue
            if along_x:
                segs.append([(lo[i], y[i]), (hi[i], y[i])])
            else:
                segs.append([(x[i], lo[i]), (x[i], hi[i])])
    if segs:
        ax.add_collection(LineCollection(segs, colors=colour, linewidths=UNC_LW, zorder=5), autolim=False)
    return bool(segs)


def _unc_handle(label=None, colour=UNC_COLOUR):
    return Line2D([], [], color=colour, lw=1.0, label=label or 'Uncertainty (90 % interval)')


def _hull(pts):
    # convex hull (monotone chain), counter-clockwise, first point not repeated
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def central_outline(px, py, keep=0.90):
    # outline of the central 90 % of one sample's realisations: the convex hull of the
    # points with the smallest Mahalanobis distance from their mean
    pts = np.column_stack([px, py])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 5:
        return None
    c = pts - pts.mean(axis=0)
    cov = np.cov(c.T)
    d = np.einsum('ij,jk,ik->i', c, np.linalg.pinv(cov), c)
    inner = pts[d <= np.quantile(d, keep)]
    hull = _hull(np.round(inner, 9))
    if len(hull) < 2:
        return None
    return np.array(hull + [hull[0]])


def _piper_unc(ax, v, unc, mode):
    # realisations on the Piper: all of them as small grey dots (clouds), or an
    # outline round the central 90 % of each sample (outlines)
    if unc is None or mode == 'Off' or not unc.get('piper'):
        return None
    col = unc_colour(unc)
    pairs = [('piper_cat_x', 'piper_cat_y'), ('piper_an_x', 'piper_an_y'), ('piper_d_x', 'piper_d_y')]
    drawn = False
    for xc, yc in pairs:
        if xc not in unc['piper'] or yc not in unc['piper']:
            continue
        px = unc['piper'][xc].reindex(v.index).to_numpy(dtype=float)
        py = unc['piper'][yc].reindex(v.index).to_numpy(dtype=float)
        if mode == 'Clouds':
            ok = np.isfinite(px) & np.isfinite(py)
            if ok.any():
                ax.scatter(px[ok], py[ok], s=1.2, color=col, lw=0, zorder=50)
                drawn = True
        else:
            for k in range(len(px)):
                line = central_outline(px[k], py[k])
                if line is not None:
                    ax.plot(line[:, 0], line[:, 1], color=col, lw=UNC_LW + 0.1, zorder=50)
                    drawn = True
    if not drawn:
        return None
    if mode == 'Clouds':
        return Line2D([], [], ls='', marker='o', ms=2, mfc=col, mec='none',
                      label='Monte Carlo realisations')
    return Line2D([], [], color=col, lw=1.0, label='Central 90 % of realisations')


# %% Piper
def piper(v, color_by=None, water_legend=False, show_types=True, unc=None, unc_mode='Clouds', means=False):
    # Piper diagram, adapted from WQChartPy triangle_piper with coloured fields
    with matplotlib.rc_context(STYLE):
        offset = 0.10
        h = 0.5 * np.tan(np.pi / 3.0)
        t3 = np.tan(np.pi / 3)
        a = PIPER_ALPHA

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
            kw = dict(fontsize=10, ha='center')
            ax.text(x0 + 0.25, yc + h, 'Ca-Cl', color=PIPER['Ca-Cl'], **kw)
            ax.text(0.3 + offset, 1.2 * h, 'Mg-HCO$_3$', color=PIPER['Mg-HCO3'], **kw)
            ax.text(1.62 + offset, 1.2 * h - 0.02, 'Na-Cl', color=PIPER['Na-Cl'], **kw)
            ax.text(1.0 + offset, yc + s60 - h - 0.49, 'Na-HCO$_3$', color=PIPER['Na-HCO3'], **kw)

        # uncertainty behind the samples
        unc_handle = _piper_unc(ax, v, unc, unc_mode)

        # plot the samples
        styles = group_styles(v)
        hollow = hollow_mask(v, ['piper'])
        cvals = v[color_by] if color_by else None
        mappable = None
        for xc, yc_ in [('piper_cat_x', 'piper_cat_y'), ('piper_an_x', 'piper_an_y'),
                        ('piper_d_x', 'piper_d_y')]:
            m = _scatter(ax, v, v[xc], v[yc_], hollow, styles, cvals=cvals, size=28,
                         legend=False, zorder=100, means=means)
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
            if means:
                handles.append(_mean_handle())
        else:
            handles = _legend_handles(styles, (hollow & plotted).any(), means)
        handles += ring_handles(v[plotted])
        if unc_handle is not None:
            handles.append(unc_handle)
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
def schoeller(v, unc=None):
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
            if unc is not None:
                segs = []
                for xk, a in zip(xs, ions):
                    lo, hi = _unc_cols(unc, v, f'{a} (meq/L)')
                    if lo is not None and np.isfinite(y[xk - 1]) and lo[i] > 0 and hi[i] > lo[i]:
                        segs.append([(xk, lo[i]), (xk, hi[i])])
                if segs:
                    ax.add_collection(LineCollection(segs, colors=unc_colour(unc), linewidths=UNC_LW + 0.4,
                                                     zorder=1), autolim=False)
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
        if unc is not None:
            handles.append(_unc_handle(colour=unc_colour(unc)))
        ax.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    return fig


# %% Stiff
def stiff(v, samples, ncols=4, letters=None, letters_pos=PANEL_LABEL_POS[0], scale_samples=None):
    # grid of Stiff diagrams, all on the same meq/L scale. scale_samples: the samples that set that scale
    # (e.g. every sample, when they are split over several pages), otherwise those on this page
    sub = v[v['Sample'].isin(samples)]
    n = len(sub)
    if n == 0:
        return None
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    styles = group_styles(v)
    need = ['Na+K (meq/L)', 'Ca (meq/L)', 'Mg (meq/L)', 'Cl (meq/L)', 'HCO3 (meq/L)', 'SO4 (meq/L)']
    scl = v[v['Sample'].isin(scale_samples)] if scale_samples is not None else sub
    cmax = float(np.nanmax(scl[need].values)) if scl[need].notna().any().any() else 1.0
    hollow = hollow_mask(v, ['HCO3 (meq/L)'])
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, (3.6 * nrows + 1.2) * CM), dpi=150)
        axs = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
        fig.subplots_adjust(left=0.07, right=0.93, top=1 - 0.5 / (3.6 * nrows + 1.2),
                            bottom=1.3 / (3.6 * nrows + 1.2), wspace=0.55, hspace=0.75)
        for ax, i in zip(axs, sub.index):
            r = sub.loc[i, need].astype(float).fillna(0)
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
        panel_letters(axs[:n], letters, letters_pos)
        fig.text(0.5, 0.01, 'meq/L (common scale). Hatched = filled HCO$_3$.', ha='center', fontsize=7)
    return fig


# %% WQChartPy diagrams
WQ_DIAGRAMS = {'Durov': 'durvo', 'Gibbs': 'gibbs', 'Chadha': 'chadha', 'HFE-D': 'hfed'}


def _wq_frame(v, kind):
    # the table WQChartPy wants (mg/L), with '_i' = row of v. Rows it cannot draw are dropped
    w = pd.DataFrame({'_i': v.index})
    w['pH'] = v['pH'].to_numpy()
    for a in ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4']:
        w[a] = v[f'{a} (mg/L)'].to_numpy()
    w['CO3'] = w['CO3'].fillna(0)
    w['K'] = w['K'].fillna(0)
    w['TDS'] = v['TDS (mg/L)'].to_numpy()
    w = w.dropna(subset=['pH', 'Ca', 'Mg', 'Na', 'HCO3', 'Cl', 'SO4', 'TDS'])
    if kind == 'Durov':
        w = w[w['TDS'] <= 4000]
    return w.reset_index(drop=True)


def _wq_realisation_points(w, kind, unc):
    # where WQChartPy would put every realisation of every sample.
    # WQChartPy draws one scatter call per row, so it is run once on the realisations
    # with Axes.scatter swapped for a recorder. Each row gets a unique colour that
    # encodes (sample, realisation); the recorder reads it back, with the order of the
    # calls within a row (some diagrams draw a sample in several places).
    # returns [(axes number, sample row in w, call number in the row, x, y), ...]
    import importlib
    import matplotlib.axes
    import matplotlib.pyplot as plt

    real = unc.get('real') or {}
    if not real:
        return []
    n = unc['n']
    rows = []
    for k, i in enumerate(w['_i']):
        r = pd.DataFrame({'pH': real['pH'].loc[i].to_numpy(), 'TDS': real['TDS (mg/L)'].loc[i].to_numpy()})
        for a in ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4']:
            r[a] = real[f'{a} (mg/L)'].loc[i].to_numpy()
        r['_row'] = k * n + np.arange(n) + 1
        rows.append(r)
    d = pd.concat(rows, ignore_index=True)
    d['CO3'] = d['CO3'].fillna(0)
    d['K'] = d['K'].fillna(0)
    d = d.dropna(subset=['pH', 'Ca', 'Mg', 'Na', 'HCO3', 'Cl', 'SO4', 'TDS'])
    if kind == 'Durov':
        d = d[d['TDS'] <= 4000]
    if d.empty:
        return []
    d['Color'] = ['#%06x' % r for r in d['_row']]
    if kind == 'Durov':
        # the Durov pH axis runs between the whole pH units either side of the data. Keep
        # the realisations inside the samples' range and put the samples in the run (not
        # recorded), so both runs have the same pH axis
        lo, hi = np.floor(w['pH'].min()), np.ceil(w['pH'].max())
        d['pH'] = d['pH'].clip(lo, hi)
        samples = w.drop(columns='_i').copy()
        samples['Color'] = 'black'
        d = pd.concat([d, samples], ignore_index=True)
    d['Sample'] = ''
    d['Label'] = ''
    d['Marker'] = 'o'
    d['Size'] = 1
    d['Alpha'] = 1.0
    d = d.reset_index(drop=True)

    found = []
    calls = {}
    empty = {}
    original = matplotlib.axes.Axes.scatter

    def recorder(ax, x, y, *args, **kw):
        col = kw.get('color', kw.get('c'))
        if isinstance(col, str) and len(col) == 7 and col.startswith('#'):
            rid = int(col[1:], 16)
            j = calls.get(rid, 0)
            calls[rid] = j + 1
            found.append((ax, (rid - 1) // n, j, float(np.ravel(x)[0]), float(np.ravel(y)[0])))
            # pyplot wants a collection that belongs to the axes back: one empty one per axes
            if ax not in empty:
                empty[ax] = original(ax, [], [])
            return empty[ax]
        return original(ax, x, y, *args, **kw)

    module = importlib.import_module('wqchartpy.' + WQ_DIAGRAMS[kind])
    before = set(plt.get_fignums())
    matplotlib.axes.Axes.scatter = recorder
    try:
        import warnings
        # the recording run has no labelled points, so WQChartPy's legend call warns
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore')
            module.plot(d, unit='mg/L', figname=os.path.join(tmp, 'fig'), figformat='png')
    finally:
        matplotlib.axes.Axes.scatter = original
        new = [f for f in plt.get_fignums() if f not in before]
        axes = plt.figure(new[-1]).axes if new else []
        for f in new:
            plt.close(f)
    return [(axes.index(ax), k, j, x, y) for ax, k, j, x, y in found if ax in axes]


def _far_from_cluster(x, y, k=8.0):
    # True for points far from the median, in units of the median absolute deviation
    # (with a floor, so a tight cluster does not flag everything)
    out = np.zeros(len(x), dtype=bool)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return ~ok
    d = np.zeros(len(x))
    for z in (x, y):
        med = np.median(z[ok])
        mad = np.median(np.abs(z[ok] - med))
        span = np.ptp(z[ok])
        d = np.maximum(d, np.abs(z - med) / max(mad, 0.01 * span, 1e-9))
    out[ok] = d[ok] > k
    return out | ~ok


def _wq_add_uncertainty(fig, pts, mode, colour=UNC_COLOUR, kind=None):
    # draw the recorded realisations on the WQChartPy figure, under the samples
    if not pts:
        return False
    p = pd.DataFrame(pts, columns=['ax', 'k', 'j', 'x', 'y'])
    p = p[np.isfinite(p['x']) & np.isfinite(p['y'])]
    drawn = False
    for a, g in p.groupby('ax'):
        if a >= len(fig.axes):
            continue
        ax = fig.axes[a]
        # samples above the realisations
        for c in ax.collections:
            c.set_zorder(max(c.get_zorder(), 3))
        xl, yl = ax.get_xlim(), ax.get_ylim()
        if mode == 'Clouds':
            ax.scatter(g['x'], g['y'], s=1.2, color=colour, lw=0, zorder=2.5)
            drawn = True
        else:
            xlog, ylog = ax.get_xscale() == 'log', ax.get_yscale() == 'log'
            stray = []
            for _, h in g.groupby(['k', 'j']):
                hx = (np.log10(h['x'].where(h['x'] > 0)) if xlog else h['x']).to_numpy(dtype=float)
                hy = (np.log10(h['y'].where(h['y'] > 0)) if ylog else h['y']).to_numpy(dtype=float)
                # some diagrams (HFE-D) jump when the dominant ion changes. Realisations far
                # from the sample's main cluster are drawn as dots, not inside its outline
                far = _far_from_cluster(hx, hy)
                if far.any():
                    stray.append(h[far])
                line = central_outline(hx[~far], hy[~far])
                if line is None:
                    continue
                lx = 10 ** line[:, 0] if xlog else line[:, 0]
                ly = 10 ** line[:, 1] if ylog else line[:, 1]
                ax.plot(lx, ly, color=colour, lw=UNC_LW + 0.1, zorder=2.5)
                drawn = True
            if stray:
                st_ = pd.concat(stray)
                ax.scatter(st_['x'], st_['y'], s=1.2, color=colour, lw=0, zorder=2.5)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
    if drawn and kind == 'HFE-D':
        h = (Line2D([], [], ls='', marker='o', ms=2, mfc=colour, mec='none') if mode == 'Clouds'
             else Line2D([], [], color=colour, lw=1.0))
        _hfed_legend(fig, (h, 'Monte Carlo realisations' if mode == 'Clouds' else 'Central 90 % of realisations'))
    elif drawn:
        # a small legend entry just under WQChartPy's own legend (which is left as it is)
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            fig.canvas.draw()
            box = leg.get_window_extent().transformed(fig.transFigure.inverted())
            size = leg.get_texts()[0].get_fontsize() if leg.get_texts() else 10
            h = (Line2D([], [], ls='', marker='o', ms=2, mfc=colour, mec='none') if mode == 'Clouds'
                 else Line2D([], [], color=colour, lw=1.0))
            ax.add_artist(leg)
            ax.legend([h], ['Monte Carlo realisations' if mode == 'Clouds' else 'Central 90 % of realisations'],
                      loc='upper left', bbox_to_anchor=(box.x0, box.y0), bbox_transform=fig.transFigure,
                      frameon=False, fontsize=size, borderaxespad=0)
            break
    return drawn


def _hfed_legend(fig, extra=None):
    # WQChartPy puts the HFE-D legend over the top of the diagram. Move it (with the
    # uncertainty entry, if any) under the list of hydrochemical facies on the right
    ax = next((a for a in fig.axes if a.get_legend() is not None), None)
    if ax is None:
        return
    leg = ax.get_legend()
    handles, labels = list(leg.legend_handles), [t.get_text() for t in leg.get_texts()]
    if extra is not None:
        handles.append(extra[0])
        labels.append(extra[1])
    fig.canvas.draw()
    last = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().startswith('16:')]
    leg.remove()
    if last:
        box = last[0].get_window_extent().transformed(fig.transFigure.inverted())
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(box.x0, box.y0 - 0.04),
                  bbox_transform=fig.transFigure, frameon=False, fontsize=12, borderaxespad=0,
                  handletextpad=0.4, markerscale=1.4)
    else:
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 0.3), frameon=False, fontsize=12)


def wqchartpy_figure(v, kind, letters=None, letters_pos=PANEL_LABEL_POS[0], unc=None, unc_mode='Clouds'):
    # Durov, Gibbs, Chadha, HFE-D drawn by WQChartPy, then plain text and the uncertainty added
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
    pts = []
    if unc is not None and unc_mode != 'Off':
        pts = _wq_realisation_points(_wq_frame(v, kind), kind, unc)

    module = importlib.import_module('wqchartpy.' + WQ_DIAGRAMS[kind])
    before = set(plt.get_fignums())
    # WQChartPy prints a message each time; keep it out of the server log
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        module.plot(w, unit='mg/L', figname=os.path.join(tmp, 'fig'), figformat='png')
    new = [n for n in plt.get_fignums() if n not in before]
    if not new:
        return None, note
    fig = plt.figure(new[-1])
    for n in new[:-1]:
        plt.close(n)
    # WQChartPy uses bold and italic text and changes the global settings. Put
    # every label back to plain text
    matplotlib.rcParams['mathtext.default'] = 'regular'
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontweight('normal')
        t.set_fontstyle('normal')
        t.set_fontfamily(FONT)
    matplotlib.rcdefaults()
    matplotlib.rcParams['mathtext.default'] = 'regular'
    drawn = False
    if unc is not None and unc_mode != 'Off':
        drawn = _wq_add_uncertainty(fig, pts, unc_mode, colour=unc_colour(unc), kind=kind)
        if not drawn:
            note = (note + ' ' if note else '') + 'The uncertainty could not be drawn on this diagram.'
    if kind == 'HFE-D' and not drawn:
        _hfed_legend(fig)
    # panel letters for the diagrams with more than one panel (the Gibbs pair)
    if len(fig.axes) > 1:
        panel_letters(fig.axes, letters, letters_pos, outside_only=True)
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

_LINE_STYLES = ['--', ':', '-.', (0, (6, 2, 1, 2, 1, 2)), (0, (12, 3))]


def available(v, panel):
    # only draw a panel if both variables have data
    x, y = panel['x'], panel['y']
    if x not in v.columns or y not in v.columns:
        return False
    return bool((v[x].notna() & v[y].notna()).any())


def _clear_spot(ax, x, y, candidates):
    # pick the label position furthest from the plotted points
    # candidates: [(x, y, ha, va), ...] in axes fractions (0 to 1)
    ok = x.notna() & y.notna()
    if not ok.any():
        return candidates[0]
    pts = ax.transAxes.inverted().transform(ax.transData.transform(np.column_stack([x[ok], y[ok]])))
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return candidates[0]
    best, best_d = candidates[0], -1.0
    for c in candidates:
        # the text runs away from its anchor, so test a point a little way into it
        cx = c[0] + {'left': 0.10, 'right': -0.10, 'center': 0.0}[c[2]]
        cy = c[1] + {'bottom': 0.04, 'top': -0.04, 'center': 0.0}[c[3]]
        d = np.min(np.hypot((pts[:, 0] - cx) * 1.6, pts[:, 1] - cy))
        if d > best_d + 0.02:
            best, best_d = c, d
    return best


def _gibbs_field(ax, outline):
    # grey Gibbs envelope (the boomerang), closed along the right-hand axis
    lower, notch, upper = outline['lower'], outline['notch'], outline['upper']
    pts = (lower + [(1.0, lower[-1][1]), (1.0, notch[0][1])] + notch +
           [(1.0, notch[-1][1]), (1.0, upper[0][1])] + upper)
    xs, ys = zip(*pts)
    ax.fill(xs, ys, facecolor='0.85', edgecolor='0.4', lw=0.6, ls='--', alpha=0.6, zorder=0)


# scatter axes that fit the data at both ends instead of starting at 0
_FIT_TO_DATA = {'pH'}


def _panel_xlog(panel, log_conc):
    return panel.get('xlog', False) or (log_conc and panel.get('conc', False))


def _draw_panel(ax, v, panel, styles, log_conc, cvals=None, unc=None, means=False, xlim=None):
    # xlim: x limits to use instead of the panel's own (so panels with the same x share them)
    x, y = v[panel['x']], v[panel['y']]
    xlog = _panel_xlog(panel, log_conc)
    ylog = panel.get('ylog', False) or (log_conc and panel.get('conc', False))
    if xlog:
        x = x.where(x > 0)
    if ylog:
        y = y.where(y > 0)
    hollow = hollow_mask(v, [panel['x'], panel['y']])
    mappable = _scatter(ax, v, x, y, hollow, styles, cvals=cvals, legend=False, means=means)
    xlo, xhi = _unc_cols(unc, v, panel['x'])
    ylo, yhi = _unc_cols(unc, v, panel['y'])
    has_unc = _interval_lines(ax, x, y, xlo, xhi, ylo, yhi, xlog=xlog, ylog=ylog, colour=unc_colour(unc))

    ok = x.notna() & y.notna()
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    # set the axis limits from the data before adding reference lines
    if ok.any() and not xlog and not ylog and not panel.get('origin', False):
        # non-negative axes start at 0. fixing one end turns autoscaling off, so pad the other end here
        # (as margins(0.08) would). pH axes are left to fit the data, padded at both ends: no natural
        # water is near pH 0, so an axis from 0 is mostly empty
        if x[ok].min() >= 0 and panel['x'] not in _FIT_TO_DATA:
            ax.set_xlim(0, float(x[ok].max()) * 1.08 or 1)
        if (y[ok].min() >= 0 and panel['y'] not in _FIT_TO_DATA
                and not panel['y'].startswith(('SI ', 'log ', 'CBE', 'CAI'))):
            ax.set_ylim(0, float(y[ok].max()) * 1.08 or 1)
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
    if xlim is not None:
        xl = list(xlim)
        ax.set_xlim(xl)
    if panel.get('gibbs', False):
        xl = [0, 1]
        yl = [1, 100000]
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        _gibbs_field(ax, gibbs_fields.NA_CA if panel['x'].startswith('Na') else gibbs_fields.CL_HCO3)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        # each label has a few possible positions; use the one clear of the data
        for txt, cands in [
                ('Evaporation\ndominance', [(0.97, 0.96, 'right', 'top'), (0.60, 0.96, 'right', 'top'),
                                            (0.30, 0.96, 'right', 'top')]),
                ('Rock\ndominance', [(0.03, 0.47, 'left', 'center'), (0.03, 0.72, 'left', 'center'),
                                     (0.03, 0.14, 'left', 'center'), (0.30, 0.47, 'left', 'center')]),
                ('Precipitation\ndominance', [(0.97, 0.04, 'right', 'bottom'), (0.60, 0.04, 'right', 'bottom'),
                                              (0.30, 0.04, 'right', 'bottom')])]:
            fx, fy, ha, va = _clear_spot(ax, x, y, cands)
            ax.text(fx, fy, txt, transform=ax.transAxes, fontsize=6, color='0.25', ha=ha, va=va, zorder=20)
    if 'band' in panel:
        ax.axhspan(panel['band'][0], panel['band'][1], color='0.85', alpha=0.6, lw=0, zorder=0,
                   label=panel['band'][2])
    for bx in panel.get('boxes', []):
        ax.fill([bx[0], bx[1], bx[1], bx[0]], [bx[2], bx[2], bx[3], bx[3]], facecolor='0.85',
                edgecolor='0.4', lw=0.6, alpha=0.6, zorder=0)
        # label beside the box, on whichever side is clear of the data
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        (fx0, fy0), (fx1, fy1) = ax.transAxes.inverted().transform(
            ax.transData.transform([(bx[0], bx[2]), (bx[1], bx[3])]))
        fxm, fym = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)
        cands = [(fxm, fy0 - 0.02, 'center', 'top'), (fxm, fy1 + 0.02, 'center', 'bottom'),
                 (fx1 + 0.02, fym, 'left', 'center'), (fx0 - 0.02, fym, 'right', 'center')]
        cands = [c for c in cands if 0.06 < c[1] < 0.94 and (c[2] != 'left' or c[0] < 0.80)
                 and (c[2] != 'right' or c[0] > 0.20)] or cands[:1]
        fx, fy, ha, va = _clear_spot(ax, x, y, cands)
        ax.text(fx, fy, bx[4], transform=ax.transAxes, fontsize=6, color='0.25', ha=ha, va=va, zorder=20)
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
    ax._has_unc = has_unc
    return mappable, hollow & ok


def shared_xlims(v, panels, log_conc=False, unc=None, means=False):
    # x limits shared by every panel with the same x variable (and scale): each panel is drawn on a
    # scratch axes to get the limits it would set itself, and those are combined
    # returns {(x column, log): (lo, hi)}
    out = {}
    styles = group_styles(v)
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(8 * CM, 5 * CM), dpi=72)
        for p in panels:
            if not available(v, p):
                continue
            ax = fig.add_subplot()
            _draw_panel(ax, v, p, styles, log_conc, unc=unc, means=means)
            lo, hi = ax.get_xlim()
            key = (p['x'], _panel_xlog(p, log_conc))
            if key in out:
                lo, hi = min(lo, out[key][0]), max(hi, out[key][1])
            out[key] = (lo, hi)
            fig.delaxes(ax)
    return out


def gallery_page(v, panels, log_conc=False, color_by=None, letters=None, letters_pos=PANEL_LABEL_POS[0],
                 unc=None, means=False, share_panels=None, xlims=None):
    # up to 6 panels (2 x 3) with one legend along the bottom
    # panels with the same x variable share their x limits: xlims from shared_xlims (worked out once for
    # every page of a gallery set), or worked out here across share_panels, or across this page
    panels = [p for p in panels if available(v, p)]
    if not panels:
        return None
    if xlims is None:
        xlims = shared_xlims(v, share_panels if share_panels is not None else panels, log_conc, unc=unc,
                             means=means)
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
        any_unc = False
        mappable = None
        for ax, p in zip(axs, panels):
            m, hol = _draw_panel(ax, v, p, styles, log_conc, cvals=cvals, unc=unc, means=means,
                                 xlim=xlims.get((p['x'], _panel_xlog(p, log_conc))))
            mappable = m or mappable
            any_hollow = any_hollow or bool(hol.any())
            any_unc = any_unc or ax._has_unc
        for ax in axs[n:]:
            ax.axis('off')
        panel_letters(axs[:n], letters, letters_pos)
        if color_by and mappable is not None:
            cax = fig.add_axes([0.89, 0.25, 0.018, 0.5])
            fig.colorbar(mappable, cax=cax).set_label(pretty(color_by))
            handles = [Line2D([], [], ls='', marker=mk, mfc='0.6', mec='black', mew=0.5, ms=5, label=lab)
                       for lab, (_, mk) in styles.items()]
            if any_hollow:
                handles.append(Line2D([], [], ls='', marker='o', mfc='none', mec='0.3', ms=5,
                                      label='Filled input (hollow)'))
            if means:
                handles.append(_mean_handle())
        else:
            handles = _legend_handles(styles, any_hollow, means)
        handles += ring_handles(v)
        if any_unc:
            handles.append(_unc_handle(colour=unc_colour(unc)))
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=min(4, len(handles)), frameon=False)
    return fig


# %% custom plot
# reference lines through the origin for the custom plot: name -> slope (y = slope * x)
RATIO_LINES = {'1:1 (y = x)': 1.0, '2:1 (y = 2x)': 2.0, '1:2 (y = x/2)': 0.5}
SEAWATER_LINE = 'Seawater ratio'
CUSTOM_RATIO = 'Custom ratio'


def _ion_unit(col):
    # 'Ca (mmol/L)' -> ('Ca', 'mmol/L'); anything else -> (None, None)
    if not col.endswith(')') or ' (' not in col:
        return None, None
    name, unit = col[:-1].split(' (', 1)
    return (name, unit) if name in chem.ANALYTES and unit in chem.UNITS else (None, None)


def seawater_slope(x, y):
    # y/x in standard mean seawater, in the unit of the two columns; None unless both are ions with a
    # seawater value, in the same unit
    (ix, ux), (iy, uy) = _ion_unit(x), _ion_unit(y)
    if ix is None or iy is None or ux != uy or ix not in chem.SEAWATER_MGL or iy not in chem.SEAWATER_MGL:
        return None
    if ux == 'mg/L':
        return chem.SEAWATER_MGL[iy] / chem.SEAWATER_MGL[ix]
    r = SEAWATER_MMOL[iy] / SEAWATER_MMOL[ix]
    if ux == 'meq/L':
        zx, zy = abs(chem.ANALYTES[ix][1]), abs(chem.ANALYTES[iy][1])
        if not (zx and zy):
            return None
        r *= zy / zx
    return r


def custom(v, x, y, xlog=False, ylog=False, color_by=None, one_to_one=False, zero_line=False,
           annotate=False, unc=None, means=False, ratio_lines=(), custom_ratio=None, invert_y=False):
    # user picks x and y. invert_y: y increases downwards (e.g. depth)
    # ratio_lines: names from RATIO_LINES, SEAWATER_LINE and CUSTOM_RATIO (the slope is custom_ratio,
    # y:x); one_to_one is the older switch for the 1:1 line
    panel = dict(x=x, y=y, xlog=xlog, ylog=ylog, lines=[], origin=True)
    names = list(ratio_lines)
    if one_to_one and '1:1 (y = x)' not in names:
        names.insert(0, '1:1 (y = x)')
    for name in names:
        if name in RATIO_LINES:
            panel['lines'].append(('slope', RATIO_LINES[name], name.split(' ')[0]))
        elif name == SEAWATER_LINE:
            sw = seawater_slope(x, y)
            if sw is not None:
                panel['lines'].append(('slope', sw, f'Seawater ratio ({sw:.3g})'))
        elif name == CUSTOM_RATIO and custom_ratio is not None and custom_ratio > 0:
            panel['lines'].append(('slope', float(custom_ratio), f'{custom_ratio:g}:1'))
    if zero_line:
        panel['lines'].append(('h', 0.0, 'y = 0'))
    styles = group_styles(v)
    cvals = v[color_by] if color_by else None
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(17 * CM, 10 * CM), dpi=150)
        ax = fig.add_axes([0.1, 0.13, 0.58, 0.83])
        mappable, hol = _draw_panel(ax, v, panel, styles, False, cvals=cvals, unc=unc, means=means)
        if invert_y:
            ax.invert_yaxis()
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
            if means:
                handles.append(_mean_handle())
        else:
            handles = _legend_handles(styles, bool(hol.any()), means)
        handles += ring_handles(v[v[x].notna() & v[y].notna()])
        if ax._has_unc:
            handles.append(_unc_handle(colour=unc_colour(unc)))
        fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(0.69, 0.96))
    return fig


# %% bar chart against a guideline
def guideline_bars(v, var, lines, bar_colour=None, highlight=None, line_colour='#222222',
                   log=False, sort=False, title=None, unc=None):
    # one bar per sample, with the guideline as dashed line(s)
    # lines: list of (label, per-sample threshold Series) from guidelines.chart_lines
    # bar_colour None = group colours; highlight = hatch colour for bars outside the guideline
    # (hatched, so the group colour still shows)
    styles_all = group_styles(v)   # before sorting, so each group keeps its colour
    d = v[v[var].notna()]
    if sort:
        d = d.sort_values(var, kind='stable')
    if d.empty:
        return None
    x = np.arange(len(d))
    styles = {l: styles_all[l] for l in styles_all if l in set(d['Label'])}
    if bar_colour:
        colours = [bar_colour] * len(d)
    else:
        colours = [styles[l][0] for l in d['Label']]
    outside = pd.Series(False, index=d.index)
    if highlight and lines:
        if len(lines) == 2 and lines[0][0].startswith('Lower limit'):
            lo, hi = lines[0][1].reindex(d.index), lines[1][1].reindex(d.index)
            outside = (lo.notna() & (d[var] < lo)) | (hi.notna() & (d[var] > hi))
        else:
            first = lines[0][1].reindex(d.index)
            outside = first.notna() & (d[var] > first)
    width = max(12, min(30, 0.45 * len(d) + 6))
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(width * CM, 9 * CM), dpi=150)
        ax = fig.add_axes([0.1, 0.3, 0.62 if width < 20 else 0.72, 0.62])
        ax.bar(x, d[var], color=colours, edgecolor='black', linewidth=0.4, width=0.75, zorder=2)
        if outside.any():
            ox = x[outside.to_numpy()]
            ax.bar(ox, d.loc[outside, var], color='none', edgecolor=highlight, hatch='////', linewidth=0,
                   width=0.75, zorder=2.5)
            ax.bar(ox, d.loc[outside, var], color='none', edgecolor='black', linewidth=0.4, width=0.75,
                   zorder=2.6)
        unc_drawn = False
        ulo, uhi = _unc_cols(unc, d, var)
        if ulo is not None:
            val = d[var]
            down = (val - ulo).clip(lower=0).fillna(0)
            if log:
                down = down.where(ulo > 0, 0)
            up = (uhi - val).clip(lower=0).fillna(0)
            if (down > 0).any() or (up > 0).any():
                ax.errorbar(x, val, yerr=[down.to_numpy(), up.to_numpy()], fmt='none', ecolor=unc_colour(unc),
                            elinewidth=UNC_LW + 0.3, capsize=2, capthick=UNC_LW + 0.3, zorder=2.8)
                unc_drawn = True
        dashes = ['--', (0, (6, 2, 1, 2))]
        handles = []
        for k, (label, thr) in enumerate(lines):
            t = thr.reindex(d.index)
            ls = dashes[k % 2] if not label.startswith(('Lower', 'Upper')) else '--'
            if t.notna().all() and t.nunique() == 1:
                ax.axhline(t.iloc[0], color=line_colour, ls=ls, lw=1.2, zorder=3)
            else:
                for xi, ti in zip(x, t):
                    if pd.notna(ti):
                        ax.plot([xi - 0.45, xi + 0.45], [ti, ti], color=line_colour, ls=ls, lw=1.2, zorder=3)
            if not (label.startswith('Upper') and k == 1):
                text = 'Guideline range' if label.startswith('Lower') else label
                handles.append(Line2D([], [], color=line_colour, ls=ls, lw=1.2, label=text))
        ax.set_xticks(x)
        ax.set_xticklabels(d['Sample'].astype(str), rotation=90, fontsize=7)
        ax.set_xlim(-0.6, len(d) - 0.4)
        ax.set_ylabel(pretty(var))
        if log:
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=min(0, float(d[var].min()) * 1.05))
            tops = [float(t.max()) for _, t in lines if t.notna().any()]
            if ulo is not None and uhi.notna().any():
                tops.append(float(uhi.max()))
            top = max([float(d[var].max())] + tops)
            ax.set_ylim(top=top * 1.08 if top > 0 else None)
        if title:
            ax.set_title(title, fontsize=8, loc='left')
        if not bar_colour:
            handles = [matplotlib.patches.Patch(facecolor=styles[l][0], edgecolor='black', lw=0.4, label=l)
                       for l in styles] + handles
        if highlight and outside.any():
            handles.append(matplotlib.patches.Patch(facecolor='white', edgecolor=highlight, hatch='////',
                                                    lw=0.4, label='Outside the guideline'))
        if unc_drawn:
            handles.append(_unc_handle(colour=unc_colour(unc)))
        if handles:
            ax.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      fontsize=7)
    return fig


# %% png for download
BADGE_PT = 14       # height of the app icon in the 'Made using' badge, in points


def _badge_at(fig, x, y, transform):
    # 'Made using the Hydrochem Workbench' with the app icon, right-aligned at (x, y). Text only if the icon is missing
    if getattr(fig, '_badged', False):
        return
    try:
        img = mimage.imread(BADGE_ICON)
    except (FileNotFoundError, OSError):
        img = None
    parts = [TextArea('Made using the Hydrochem Workbench', textprops=dict(fontsize=6, color='0.5'))]
    if img is not None:
        parts.append(OffsetImage(img, zoom=BADGE_PT / img.shape[0]))
    packed = HPacker(children=parts, align='center', pad=0, sep=3)
    fig.add_artist(AnnotationBbox(packed, (x, y), xycoords=transform, box_alignment=(1, 0.5),
                                  frameon=False, pad=0, annotation_clip=False))


BADGE_ICON = Path(__file__).resolve().parent / 'data' / 'HydroCHEM_badge.png'


def add_badge(fig):
    # bottom right of what is drawn (legends included), in a strip just below it so it can be cropped off
    if getattr(fig, '_badged', False):
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not hasattr(fig.canvas, 'get_renderer'):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    below = fig.transFigure + mtransforms.ScaledTranslation(0, -(BADGE_PT / 2 + 4) / 72, fig.dpi_scale_trans)
    # a credit line (the map's imagery attribution) goes in the same strip, on the left
    credit = getattr(fig, '_credit', None)
    if credit:
        fig.text(box.x0 / width, box.y0 / height, credit, transform=below, ha='left', va='center',
                 fontsize=5.5, color='0.4', linespacing=1.15)
    _badge_at(fig, box.x1 / width, box.y0 / height, below)
    fig._badged = True


def fig_png(fig, dpi=300):
    add_badge(fig)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()
