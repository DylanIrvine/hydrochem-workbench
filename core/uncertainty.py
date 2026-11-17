# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# core/uncertainty.py
# Monte Carlo propagation of measurement uncertainty through the whole calculation
# (unit conversion, fills, PHREEQC, derived variables), after JCGM 101:2008
#
# for each measured value x of a variable with LOR L, relative uncertainty r and
# absolute uncertainty a, the standard uncertainty is
#     u(x) = sqrt((L / sqrt(12))^2 + (r x)^2 + a^2)
# L / sqrt(12) is the rounding of reported results to the LOR. Values are drawn from
# a normal distribution with that SD; negative draws are drawn again. Values that
# were below the LOR ("<L" or negative on the lab report) are drawn from
# Uniform(0, L) instead.
#
# the plotted point is still the measured (deterministic) result; the realisations
# give the interval around it

# %%
import numpy as np
import pandas as pd

from core import chem

# percentiles of the reported interval (90 %)
P_LO, P_HI = 5.0, 95.0
INTERVAL = '90 %'
DEFAULT_N = 200
SEED = 20261118

TABLE_COLS = ['Variable', 'LOR', 'Relative SD (%)', 'Absolute SD', 'Source']
LOR_NOTE = ('LOR in mg/L as on the lab report: NO3 and NH4 as N, PO4 as P, alkalinity as CaCO3, '
            'DIC as C. EC in uS/cm. Relative and absolute SD are 1 standard deviation.')
_QC = 'Laboratory QC (19 reports)'
_PLACE = 'Placeholder (not from QC data)'
_YSI = 'Field probe specification (YSI ProDSS)'

# name in the table: (data column(s), LOR, relative SD %, absolute SD, source)
# relative SDs for the major ions, alkalinity, NO3, Br and F are the root mean square
# deviation of laboratory control sample recoveries from 100 % (between-run
# precision and bias) in 19 QC reports from an Australian laboratory, 2021 to 2026.
# LORs are typical laboratory values. Field values are the YSI ProDSS accuracy
# (rectangular, a / sqrt(3))
_DEFAULTS = [
    ('Ca', ['Ca'], 1.0, 3.7, 0.0, _QC),
    ('Mg', ['Mg'], 1.0, 4.1, 0.0, _QC),
    ('Na', ['Na'], 1.0, 2.9, 0.0, _QC),
    ('K', ['K'], 1.0, 3.0, 0.0, _QC),
    ('Cl', ['Cl'], 1.0, 6.9, 0.0, _QC),
    ('SO4', ['SO4'], 1.0, 4.9, 0.0, _QC),
    ('Alkalinity (as CaCO3)', ['Total_Alk_CaCO3', 'Bicarb_Alk_CaCO3', 'Carb_Alk_CaCO3'], 1.0, 4.4, 0.0, _QC),
    ('HCO3 and CO3 (if entered)', ['HCO3', 'CO3'], 1.0, 4.4, 0.0, 'As alkalinity'),
    ('NO3', ['NO3'], 0.01, 3.7, 0.0, _QC),
    ('Br', ['Br'], 0.01, 3.7, 0.0, _QC),
    ('F', ['F'], 0.1, 7.4, 0.0, _QC),
    ('NH4', ['NH4'], 0.01, 5.0, 0.0, _PLACE),
    ('PO4', ['PO4'], 0.01, 5.0, 0.0, _PLACE),
    ('SiO2', ['SiO2'], 0.1, 5.0, 0.0, _PLACE),
    ('Sr', ['Sr'], 0.001, 5.0, 0.0, _PLACE),
    ('Ba', ['Ba'], 0.001, 5.0, 0.0, _PLACE),
    ('Fe', ['Fe'], 0.05, 5.0, 0.0, _PLACE),
    ('Mn', ['Mn'], 0.001, 5.0, 0.0, _PLACE),
    ('B', ['B'], 0.05, 5.0, 0.0, _PLACE),
    ('DIC', ['DIC'], 1.0, 5.0, 0.0, _PLACE),
    ('TDS (measured)', ['TDS'], 10.0, 5.0, 0.0, _PLACE),
    ('pH', ['pH'], 0.0, 0.0, round(0.2 / np.sqrt(3), 3), _YSI),
    ('Temp', ['Temp'], 0.0, 0.0, round(0.2 / np.sqrt(3), 3), _YSI),
    ('EC', ['EC'], 0.0, round(0.5 / np.sqrt(3), 2), 0.0, _YSI),
]
COLUMNS = {name: cols for name, cols, *_ in _DEFAULTS}


def default_table():
    return pd.DataFrame([(name, lor, rel, ab, src) for name, _, lor, rel, ab, src in _DEFAULTS],
                        columns=TABLE_COLS)


def check_table(table):
    # the table as edited in the app: known rows only, numbers >= 0, missing rows
    # put back from the defaults
    d = default_table().set_index('Variable')
    t = pd.DataFrame(table).copy()
    t = t[t['Variable'].isin(d.index)].drop_duplicates('Variable').set_index('Variable')
    for c in ['LOR', 'Relative SD (%)', 'Absolute SD']:
        t[c] = pd.to_numeric(t[c], errors='coerce').clip(lower=0)
    t = t.reindex(d.index)
    for c in TABLE_COLS[1:]:
        t[c] = t[c].where(t[c].notna(), d[c])
    return t.reset_index()[TABLE_COLS]


def mark_edited(table):
    # rows changed from the defaults are marked as the user's values
    t = table.copy()
    d = default_table().set_index('Variable').reindex(t['Variable'])
    num = ['LOR', 'Relative SD (%)', 'Absolute SD']
    changed = ~np.isclose(t[num].to_numpy(dtype=float), d[num].to_numpy(dtype=float)).all(axis=1)
    t.loc[changed, 'Source'] = 'Entered by user'
    return t


# %% LOR in the data's units
def lor_in_units(name, col, lor, unit, n_as_n=False, p_as_p=False, dic_unit='mg/L as C'):
    # the table's LOR (mg/L as on the lab report) in the unit of the data column
    if lor == 0 or col in ('pH', 'Temp', 'EC', 'TDS') or col in chem.ALK_COLS:
        return lor
    if col == chem.DIC_COL:
        return lor if dic_unit == 'mg/L as C' else lor / chem.MW_C
    if col in ('HCO3', 'CO3'):
        # the LOR is for alkalinity as CaCO3; as the ion, in the ion unit
        mw, z, _ = chem.ANALYTES[col]
        meq = lor / chem.CACO3_PER_MEQ
        mmol = meq / abs(z)
        return {'mg/L': mmol * mw, 'mmol/L': mmol, 'meq/L': meq}[unit]
    mw, z, _ = chem.ANALYTES[col]
    # lab reports give NO3 and NH4 as N and PO4 as P
    if col in ('NO3', 'NH4'):
        mmol = lor / chem.MW_N
        as_reported = n_as_n
    elif col == 'PO4':
        mmol = lor / chem.MW_P
        as_reported = p_as_p
    else:
        mmol = lor / mw
        as_reported = False
    if unit == 'mg/L':
        return lor if as_reported else mmol * mw
    if unit == 'mmol/L':
        return mmol
    return mmol * abs(z) if z != 0 else mmol


def sigmas(df, table, unit, n_as_n=False, p_as_p=False, dic_unit='mg/L as C'):
    # standard uncertainty for every measured cell (NaN where not measured)
    t = check_table(table).set_index('Variable')
    out = pd.DataFrame(index=df.index)
    lors = {}
    for name, cols in COLUMNS.items():
        lor, rel, ab = (float(t.at[name, c]) for c in ['LOR', 'Relative SD (%)', 'Absolute SD'])
        for col in cols:
            if col not in df.columns:
                continue
            x = pd.to_numeric(df[col], errors='coerce')
            lor_u = lor_in_units(name, col, lor, unit, n_as_n, p_as_p, dic_unit)
            lors[col] = lor_u
            out[col] = np.sqrt((lor_u / np.sqrt(12)) ** 2 + (rel / 100 * x) ** 2 + ab ** 2).where(x.notna())
    return out, lors


# %% one realisation of the inputs
def perturb(df, sig, below, rng):
    # df: cleaned data. sig: from sigmas(). below: {(row, column): LOR} for values that
    # were below the LOR on the report (drawn from Uniform(0, LOR))
    d = df.copy()
    for col in sig.columns:
        x = d[col].astype(float).to_numpy()
        s = sig[col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(s) & (s > 0)
        if not ok.any():
            continue
        new = x.copy()
        draw = x[ok] + rng.standard_normal(ok.sum()) * s[ok]
        if col not in ('pH', 'Temp'):
            # concentrations cannot be negative: draw again, then clip what is left
            for _ in range(20):
                neg = draw < 0
                if not neg.any():
                    break
                draw[neg] = x[ok][neg] + rng.standard_normal(neg.sum()) * s[ok][neg]
            draw = np.clip(draw, 0, None)
        new[ok] = draw
        d[col] = new
    for (i, col), lor in below.items():
        if col in d.columns and i in d.index and pd.notna(d.at[i, col]):
            d.at[i, col] = rng.uniform(0.0, lor)
    return d


# %% the Monte Carlo run
def run(df, below, table, settings, pp, n=DEFAULT_N, seed=SEED, progress=None):
    # df: cleaned data (after the below-LOR substitution), below: {(row, col): LOR}
    # settings: dict(unit, n_as_n, p_as_p, dic_unit, rules, extra_cols)
    # returns dict with lo, hi (DataFrames, same index and numeric columns as the
    # results table), the summary table, the Piper realisations and run notes
    rng = np.random.default_rng(seed)
    sig, _ = sigmas(df, table, settings['unit'], settings['n_as_n'], settings['p_as_p'], settings['dic_unit'])
    stack = []
    wtype, chadha = [], []
    n_fail = 0
    cols = None
    for r in range(n):
        d = perturb(df, sig, below, rng)
        v, errors = chem.derive(d, pp, **settings)
        n_fail += len(errors)
        if cols is None:
            cols = [c for c in v.columns if pd.api.types.is_numeric_dtype(v[c]) and v[c].dtype != bool
                    and c not in chem.INTERNAL_COLS]
        stack.append(v[cols].to_numpy(dtype=float))
        wtype.append(v['Water type'].to_numpy())
        chadha.append(v['Chadha field'].to_numpy())
        if progress is not None:
            progress((r + 1) / n)
    arr = np.stack(stack)   # (realisations, samples, columns)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lo = np.nanpercentile(arr, P_LO, axis=0)
            hi = np.nanpercentile(arr, P_HI, axis=0)
    lo = pd.DataFrame(lo, index=df.index, columns=cols)
    hi = pd.DataFrame(hi, index=df.index, columns=cols)
    piper = {c: pd.DataFrame(arr[:, :, cols.index(c)].T, index=df.index)
             for c in cols if c.startswith('piper_')}
    return dict(lo=lo, hi=hi, piper=piper, n=n, seed=seed, fails=n_fail,
                wtype=np.stack(wtype), chadha=np.stack(chadha),
                si=pd.DataFrame({c: (arr[:, :, cols.index(c)] > 0).mean(axis=0)
                                 for c in cols if c.startswith('SI ')}, index=df.index),
                si_n=pd.DataFrame({c: np.isfinite(arr[:, :, cols.index(c)]).sum(axis=0)
                                   for c in cols if c.startswith('SI ')}, index=df.index),
                cbe=pd.Series((np.abs(arr[:, :, cols.index('CBE (%)')]) <= 5).mean(axis=0), index=df.index),
                cbe_n=pd.Series(np.isfinite(arr[:, :, cols.index('CBE (%)')]).sum(axis=0), index=df.index))


def summary(v, mc):
    # per sample: how often the realisations agree with the measured result
    s = pd.DataFrame({'Sample': v['Sample'], 'Label': v['Label']})
    s['Water type'] = v['Water type']
    s['Same water type (%)'] = [100 * np.mean(mc['wtype'][:, k] == t) if t else np.nan
                                for k, t in enumerate(v['Water type'])]
    s['Chadha field'] = v['Chadha field']
    s['Same Chadha field (%)'] = [100 * np.mean(mc['chadha'][:, k] == f) if f else np.nan
                                  for k, f in enumerate(v['Chadha field'])]
    s['CBE (%)'] = v['CBE (%)']
    s['CBE 5th pct'] = mc['lo']['CBE (%)']
    s['CBE 95th pct'] = mc['hi']['CBE (%)']
    s['|CBE| <= 5 % (% of realisations)'] = (100 * mc['cbe']).where(mc['cbe_n'] > 0)
    for c in mc['si'].columns:
        if v[c].notna().any():
            s[f'{c} > 0 (% of realisations)'] = (100 * mc['si'][c]).where(mc['si_n'][c] > 0)
    return s


def interval_table(v, mc):
    # results with the 5th and 95th percentiles beside each numeric column
    out = {'Sample': v['Sample'], 'Label': v['Label']}
    for c in mc['lo'].columns:
        if c.startswith('piper_') or not v[c].notna().any():
            continue
        out[c] = v[c]
        out[f'{c} 5th pct'] = mc['lo'][c]
        out[f'{c} 95th pct'] = mc['hi'][c]
    return pd.DataFrame(out, index=v.index)


def method_text(n, seed):
    return (f'Monte Carlo propagation of measurement uncertainty (JCGM 101:2008), {n} realisations, '
            f'random seed {seed}. Each measured value is drawn from a normal distribution with standard '
            'deviation sqrt((LOR/sqrt(12))^2 + (relative SD x value)^2 + absolute SD^2); negative draws '
            'are drawn again. Values below the LOR on the lab report are drawn from Uniform(0, LOR). '
            'Every realisation goes through the same unit conversion, fills, PHREEQC speciation and '
            f'derived variables as the measured data. Intervals are {INTERVAL} (5th to 95th percentile '
            'of the realisations); points are the measured results.')
