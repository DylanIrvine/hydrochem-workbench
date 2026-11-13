# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.
#
# piper_coords() uses the coordinate transform from WQChartPy
# (triangle_piper.py), Copyright (C) the WQChartPy authors (Yang et al., 2022),
# GPL-3.0: https://github.com/jyangfsu/WQChartPy. Modified 2026.

# core/chem.py
# template, reading the upload, unit conversion, filling gaps, PHREEQC,
# and the table of derived variables used by the plots

# %%
import io

import numpy as np
import pandas as pd

# %% analytes
# name: (g/mol, charge, PHREEQC name)
# SiO2 has no charge, so it is read as mmol/L if the input unit is meq/L
ANALYTES = {
    'Ca':   (40.078,  +2, 'Ca'),
    'Mg':   (24.305,  +2, 'Mg'),
    'Na':   (22.990,  +1, 'Na'),
    'K':    (39.098,  +1, 'K'),
    'HCO3': (61.017,  -1, None),      # passed as Alkalinity
    'CO3':  (60.009,  -2, None),      # passed as Alkalinity
    'Cl':   (35.453,  -1, 'Cl'),
    'SO4':  (96.06,   -2, 'S(6)'),
    'NO3':  (62.004,  -1, 'N(5)'),
    'Br':   (79.904,  -1, 'Br'),
    'F':    (18.998,  -1, 'F'),
    'SiO2': (60.084,   0, 'Si'),
    'Sr':   (87.62,   +2, 'Sr'),
    'Ba':   (137.327, +2, 'Ba'),
    'Fe':   (55.845,  +2, 'Fe(2)'),
    'Mn':   (54.938,  +2, 'Mn(2)'),
}

MAJORS = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4']
CATIONS = [a for a, (_, z, _) in ANALYTES.items() if z > 0]
ANIONS = [a for a, (_, z, _) in ANALYTES.items() if z < 0]

ID_COLS = ['Sample', 'Label']
FIELD_COLS = ['pH', 'Temp', 'EC', 'TDS']
STYLE_COLS = ['Color', 'Marker']
TEMPLATE_COLS = ID_COLS + FIELD_COLS + list(ANALYTES) + STYLE_COLS

UNITS = ['mg/L', 'mmol/L', 'meq/L']

# alkalinity as CaCO3 -> HCO3 (2 x 61.017 / 100.087)
CACO3_TO_HCO3 = 1.2193
# NO3-N -> NO3 (62.004 / 14.007)
N_TO_NO3 = 4.4267

# seawater in mg/L (Hem, 1985). Only used for reference lines
SEAWATER_MGL = {'Cl': 19350, 'Na': 10760, 'SO4': 2710, 'Mg': 1290, 'Ca': 411,
                'K': 399, 'HCO3': 142, 'Br': 67, 'Sr': 8.0, 'F': 1.3}
SEAWATER_MMOL = {k: val / ANALYTES[k][0] for k, val in SEAWATER_MGL.items()}

# minerals to report, and the ions each one needs
# no Fe/Mn/sulfide phases as there is no pe or Eh in the template
MINERALS = {
    'Calcite':      ['Ca', 'HCO3'],
    'Aragonite':    ['Ca', 'HCO3'],
    'Dolomite':     ['Ca', 'Mg', 'HCO3'],
    'Gypsum':       ['Ca', 'SO4'],
    'Anhydrite':    ['Ca', 'SO4'],
    'Halite':       ['Na', 'Cl'],
    'Fluorite':     ['Ca', 'F'],
    'Quartz':       ['SiO2'],
    'Chalcedony':   ['SiO2'],
    'Barite':       ['Ba', 'SO4'],
    'Celestite':    ['Sr', 'SO4'],
    'Strontianite': ['Sr', 'HCO3'],
}

# warn above this ionic strength (phreeqc.dat is not for saline water, seawater ~0.7)
IONIC_STRENGTH_WARN = 0.5


# %% template
def template_bytes():
    # xlsx template: Data sheet + Notes sheet
    example = example_data().head(3)
    notes = pd.DataFrame({'Column': [
        'Sample', 'Label', 'pH', 'Temp', 'EC', 'TDS',
        'Ca ... Mn', 'HCO3', 'CO3', 'NO3', 'SiO2', 'Color', 'Marker',
        '(any other column)', '', 'Units', 'Blank cells', 'Below detection'],
        'What goes in it': [
            'Unique sample name. Required.',
            'Group used for colours and legends (aquifer, site, campaign). Optional.',
            'Field pH if you have it, otherwise lab pH.',
            'Water temperature, degrees C.',
            'Electrical conductivity, uS/cm. Optional, used for plotting only.',
            'Total dissolved solids, mg/L. Optional. If blank, it is calculated from the ions.',
            'Concentrations, all in the single unit you choose in the app (mg/L, mmol/L or meq/L).',
            'Bicarbonate as HCO3, or total alkalinity as CaCO3 (tell the app which).',
            'Carbonate as CO3. Leave blank if alkalinity is entered as CaCO3.',
            'Nitrate as NO3, or as N (tell the app which).',
            'Silica as SiO2. Neutral, so if the unit is meq/L enter it in mmol/L.',
            'Optional matplotlib colour for this row (e.g. #0089d0). Blank = automatic by Label.',
            'Optional matplotlib marker for this row (e.g. o, s, ^, D). Blank = automatic by Label.',
            'Any extra numeric column (depth, d18O, 87Sr/86Sr ...) is carried through and can be plotted.',
            '',
            'One unit for every concentration column. Do not mix.',
            'Leave a cell blank if not measured. Missing pH, Temp and HCO3 are filled by the rule you pick in the app and drawn hollow.',
            'Values such as <0.05 or "bd" are not numbers. They are dropped and listed in the app. Decide what to do with them before uploading.',
        ]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xl:
        example.to_excel(xl, sheet_name='Data', index=False)
        notes.to_excel(xl, sheet_name='Notes', index=False)
        ws = xl.sheets['Data']
        for i, col in enumerate(example.columns, start=1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = max(9, len(col) + 3)
        xl.sheets['Notes'].column_dimensions['A'].width = 20
        xl.sheets['Notes'].column_dimensions['B'].width = 110
    return buf.getvalue()


def example_data():
    # synthetic example data (not real samples), 3 groups, a few gaps left in on purpose
    rng = np.random.default_rng(7)
    rows = []
    for k in range(10):   # carbonate
        f = rng.uniform(0.6, 1.4)
        rows.append(['CB%02d' % (k + 1), 'Carbonate (synthetic)',
                     rng.uniform(6.9, 7.5), rng.uniform(28, 32), np.nan, np.nan,
                     62 * f, 31 * f, 8 * rng.uniform(0.5, 2), 1.5, 330 * f, 0,
                     9 * rng.uniform(0.5, 2.5), 6 * rng.uniform(0.5, 2),
                     0.8, 0.03, 0.15, 14, 0.12, 0.05, np.nan, np.nan])
    for k in range(10):   # sandstone
        f = rng.uniform(0.5, 2.5)
        rows.append(['SS%02d' % (k + 1), 'Sandstone (synthetic)',
                     rng.uniform(6.2, 7.9), rng.uniform(29, 34), np.nan, np.nan,
                     18 * rng.uniform(0.4, 1.6), 9 * rng.uniform(0.4, 1.6),
                     70 * f, 6, 150 * rng.uniform(0.6, 1.5), 0,
                     75 * f, 28 * rng.uniform(0.4, 2.5),
                     1.2, 0.26 * f, 0.4, 32, 0.3, 0.08, 0.4, 0.05])
    for k in range(6):    # saline
        f = rng.uniform(1, 6)
        rows.append(['SA%02d' % (k + 1), 'Saline (synthetic)',
                     rng.uniform(7.0, 7.8), rng.uniform(27, 31), np.nan, np.nan,
                     140 * rng.uniform(0.7, 2), 85 * rng.uniform(0.7, 2),
                     900 * f, 30 * f, 260, 0, 1500 * f,
                     420 * rng.uniform(0.7, 3), 2.0, 5.0 * f, 0.6, 20, 3.5, 0.04,
                     np.nan, np.nan])
    cols = ['Sample', 'Label', 'pH', 'Temp', 'EC', 'TDS', 'Ca', 'Mg', 'Na', 'K',
            'HCO3', 'CO3', 'Cl', 'SO4', 'NO3', 'Br', 'F', 'SiO2', 'Sr', 'Ba',
            'Fe', 'Mn']
    df = pd.DataFrame(rows, columns=cols).round(3)
    # leave some gaps
    df.loc[3, 'pH'] = np.nan
    df.loc[14, 'HCO3'] = np.nan
    df.loc[22, 'Temp'] = np.nan
    df['Color'] = ''
    df['Marker'] = ''
    return df[TEMPLATE_COLS]


# %% read and clean
def read_upload(file):
    # reads the Data sheet, or a csv
    name = getattr(file, 'name', '') or ''
    if name.lower().endswith('.csv'):
        raw = pd.read_csv(file, dtype=object)
    else:
        sheets = pd.read_excel(file, sheet_name=None, dtype=object)
        raw = sheets['Data'] if 'Data' in sheets else next(iter(sheets.values()))
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw.dropna(how='all').reset_index(drop=True)


def clean(raw):
    # convert to numbers. Text cells (<0.05, bd etc.) go to NaN and are listed in dropped
    problems = []
    df = raw.copy()

    if 'Sample' not in df.columns:
        problems.append('There is no "Sample" column. Use the template.')
        return df, pd.DataFrame(), [], problems
    missing_majors = [c for c in MAJORS if c not in df.columns and c != 'CO3']
    if missing_majors:
        problems.append('Missing major-ion columns: ' + ', '.join(missing_majors))
        return df, pd.DataFrame(), [], problems

    df['Sample'] = df['Sample'].astype(str).str.strip()
    if 'Label' not in df.columns:
        df['Label'] = ''
    df['Label'] = df['Label'].fillna('').astype(str).str.strip()
    df.loc[df['Label'] == '', 'Label'] = 'Samples'
    for c in STYLE_COLS:
        if c not in df.columns:
            df[c] = ''
        df[c] = df[c].fillna('').astype(str).str.strip()
    for c in FIELD_COLS + list(ANALYTES):
        if c not in df.columns:
            df[c] = np.nan

    if df['Sample'].duplicated().any():
        dups = df.loc[df['Sample'].duplicated(), 'Sample'].unique()
        problems.append('Duplicate sample names: ' + ', '.join(dups[:10]))

    fixed = set(ID_COLS + STYLE_COLS)
    dropped = []
    extra_cols = []
    for c in df.columns:
        if c in fixed:
            continue
        num = pd.to_numeric(df[c], errors='coerce')
        bad = num.isna() & df[c].notna() & (df[c].astype(str).str.strip() != '')
        known = c in FIELD_COLS or c in ANALYTES
        if known:
            for i in df.index[bad]:
                dropped.append({'Sample': df.at[i, 'Sample'], 'Column': c,
                                'Value': df.at[i, c]})
            df[c] = num
        elif num.notna().sum() >= max(1, int(0.5 * df[c].notna().sum())):
            # extra numeric column, keep it for plotting
            for i in df.index[bad]:
                dropped.append({'Sample': df.at[i, 'Sample'], 'Column': c,
                                'Value': df.at[i, c]})
            df[c] = num
            extra_cols.append(c)

    # negative concentrations are dropped too
    for c in list(ANALYTES):
        neg = df[c] < 0
        for i in df.index[neg]:
            dropped.append({'Sample': df.at[i, 'Sample'], 'Column': c,
                            'Value': df.at[i, c]})
        df.loc[neg, c] = np.nan

    return df, pd.DataFrame(dropped, columns=['Sample', 'Column', 'Value']), extra_cols, problems


# %% units
def to_mmol(df, unit, alk_as_caco3=False, no3_as_n=False):
    # everything to mmol/L
    mmol = pd.DataFrame(index=df.index)
    for a, (mw, z, _) in ANALYTES.items():
        x = df[a].astype(float)
        if unit == 'mg/L':
            mmol[a] = x / mw
        elif unit == 'mmol/L':
            mmol[a] = x
        else:
            mmol[a] = x / abs(z) if z != 0 else x

    if alk_as_caco3:
        if unit == 'mg/L':
            mmol['HCO3'] = df['HCO3'].astype(float) * CACO3_TO_HCO3 / ANALYTES['HCO3'][0]
        elif unit == 'mmol/L':
            mmol['HCO3'] = df['HCO3'].astype(float) * 2.0
        # in meq/L nothing to do
    if no3_as_n and unit == 'mg/L':
        mmol['NO3'] = df['NO3'].astype(float) * N_TO_NO3 / ANALYTES['NO3'][0]
    return mmol


# %% fill
def fill_missing(df, mmol, rules):
    # fill missing pH, Temp, HCO3. rules = {'pH': ('mean' or 'value', number), ...}
    # HCO3 can also be 'balance' (from charge balance). HCO3 value is in mmol/L
    df = df.copy()
    mmol = mmol.copy()
    filled = pd.DataFrame(False, index=df.index, columns=['pH', 'Temp', 'HCO3'])

    for var in ['pH', 'Temp']:
        how, val = rules[var]
        gap = df[var].isna()
        if not gap.any():
            continue
        fill = df[var].mean() if how == 'mean' else val
        if pd.isna(fill):
            fill = val
        df.loc[gap, var] = fill
        filled.loc[gap, var] = True

    how, val = rules['HCO3']
    has_other = mmol[['Ca', 'Mg', 'Na', 'Cl']].notna().any(axis=1)
    gap = mmol['HCO3'].isna() & has_other
    if gap.any():
        if how == 'balance':
            cat = sum(mmol[a].fillna(0) * ANALYTES[a][1] for a in CATIONS)
            an = sum(mmol[a].fillna(0) * abs(ANALYTES[a][1]) for a in ANIONS if a != 'HCO3')
            fill = (cat - an).clip(lower=0)[gap]
        elif how == 'mean':
            fill = mmol['HCO3'].mean()
            if pd.isna(fill):
                fill = val
        else:
            fill = val
        mmol.loc[gap, 'HCO3'] = fill
        filled.loc[gap, 'HCO3'] = True

    return df, mmol, filled


# %% PHREEQC
def run_phreeqc(df, mmol, pp):
    # run each sample through phreeqc.dat. Alkalinity is passed as meq (HCO3 + 2*CO3)
    out = pd.DataFrame(index=df.index, dtype=float)
    cols = ['SI ' + m for m in MINERALS] + ['log pCO2', 'Ionic strength (mol/kgw)',
                                            'SC calc (uS/cm)']
    for c in cols:
        out[c] = np.nan
    errors = []

    for i in df.index:
        row = mmol.loc[i]
        if pd.isna(df.at[i, 'pH']) or row[['Ca', 'Mg', 'Na', 'Cl']].isna().all():
            continue
        sol = {'units': 'mmol/l', 'pH': float(df.at[i, 'pH']),
               'temp': float(df.at[i, 'Temp']) if pd.notna(df.at[i, 'Temp']) else 25.0}
        for a, (_, _, key) in ANALYTES.items():
            if key is not None and pd.notna(row[a]) and row[a] > 0:
                sol[key] = float(row[a])
        alk = np.nansum([row['HCO3'], 2 * row['CO3']])
        if alk > 0:
            sol['Alkalinity'] = float(alk)

        try:
            s = pp.add_solution(sol)
        except Exception as exc:   # did not converge
            errors.append({'Sample': df.at[i, 'Sample'], 'Error': str(exc)[:200]})
            continue

        for m, needs in MINERALS.items():
            if all(pd.notna(row[n]) and row[n] > 0 for n in needs):
                si = s.si(m)
                out.at[i, 'SI ' + m] = si if si > -99 else np.nan
        if alk > 0:
            out.at[i, 'log pCO2'] = s.si('CO2(g)')
        out.at[i, 'Ionic strength (mol/kgw)'] = s.I
        out.at[i, 'SC calc (uS/cm)'] = s.sc
        s.forget()

    return out, pd.DataFrame(errors, columns=['Sample', 'Error'])


# %% derived variables
def piper_coords(meq):
    # x-y coordinates on the Piper, same geometry as WQChartPy triangle_piper
    offset = 0.10
    h = 0.5 * np.tan(np.pi / 3.0)
    ca, mg = meq['Ca'], meq['Mg']
    nak = meq['Na'] + meq['K'].fillna(0)
    hco3 = meq['HCO3'] + meq['CO3'].fillna(0)
    cl, so4 = meq['Cl'], meq['SO4']
    sumcat = ca + mg + nak
    suman = hco3 + cl + so4

    cat_x = 0.5 * (2 * nak / sumcat + mg / sumcat)
    cat_y = h * mg / sumcat
    an_x = 1 + 2 * offset + 0.5 * (2 * cl / suman + so4 / suman)
    an_y = h * so4 / suman
    d_x = an_y / (4 * h) + 0.5 * an_x - cat_y / (4 * h) + 0.5 * cat_x
    d_y = 0.5 * an_y + h * an_x + 0.5 * cat_y - h * cat_x
    return pd.DataFrame({'cat_x': cat_x, 'cat_y': cat_y, 'an_x': an_x,
                         'an_y': an_y, 'd_x': d_x, 'd_y': d_y})


def water_type(p, q):
    # water type from the shaded field in the Piper diamond
    # p = (Ca+Mg)/cations, q = (SO4+Cl)/anions, in meq
    # nb. the rule in the old PiperScript.py did not match the shaded fields
    if pd.isna(p) or pd.isna(q):
        return ''
    if p + q > 1.5:
        return 'Ca-Cl'
    if p + q < 0.5:
        return 'Na-HCO3'
    if p > 0.5 and q < 0.5:
        return 'Mg-HCO3'
    if p < 0.5 and q > 0.5:
        return 'Na-Cl'
    return 'Mixed'


def _ratio(a, b):
    b = b.where(b > 0)
    return a / b


def build_table(df, mmol, filled, phreeqc, extra_cols):
    # the one table (v) that all the plots use. Units are in the column names
    v = {}
    v['Sample'] = df['Sample']
    v['Label'] = df['Label']
    v['Color'] = df['Color']
    v['Marker'] = df['Marker']
    for c in ['pH', 'Temp', 'EC']:
        v[c] = df[c]

    meq = pd.DataFrame(index=df.index)
    mgl = pd.DataFrame(index=df.index)
    for a, (mw, z, _) in ANALYTES.items():
        meq[a] = mmol[a] * abs(z)
        mgl[a] = mmol[a] * mw
        v[f'{a} (mmol/L)'] = mmol[a]
    for a, (mw, z, _) in ANALYTES.items():
        if z != 0:
            v[f'{a} (meq/L)'] = meq[a]
    for a in ANALYTES:
        v[f'{a} (mg/L)'] = mgl[a]

    # charge balance error (positive = excess cations)
    sumcat = meq[CATIONS].sum(axis=1, min_count=1)
    suman = meq[ANIONS].sum(axis=1, min_count=1)
    v['Sum cations (meq/L)'] = sumcat
    v['Sum anions (meq/L)'] = suman
    v['CBE (%)'] = (sumcat - suman) / (sumcat + suman) * 100

    # TDS: measured if given, otherwise sum of ions (HCO3 x 0.4917)
    tds_calc = mgl.drop(columns=['HCO3']).sum(axis=1, min_count=1) + 0.4917 * mgl['HCO3'].fillna(0)
    v['TDS calc (mg/L)'] = tds_calc
    v['TDS (mg/L)'] = df['TDS'].where(df['TDS'].notna(), tds_calc)
    v['TDS is measured'] = df['TDS'].notna()

    # molar ratios
    for a in ['Na', 'K', 'Ca', 'Mg', 'SO4', 'HCO3', 'NO3', 'F', 'Sr']:
        v[f'{a}/Cl'] = _ratio(mmol[a], mmol['Cl'])
    v['Cl/Br'] = _ratio(mmol['Cl'], mmol['Br'])
    v['Ca/Mg'] = _ratio(mmol['Ca'], mmol['Mg'])
    v['Mg/Ca'] = _ratio(mmol['Mg'], mmol['Ca'])
    v['Ca/SO4'] = _ratio(mmol['Ca'], mmol['SO4'])
    v['Na/K'] = _ratio(mmol['Na'], mmol['K'])
    v['Sr/Ca'] = _ratio(mmol['Sr'], mmol['Ca'])
    v['HCO3/SiO2'] = _ratio(mmol['HCO3'], mmol['SiO2'])
    v['SO4/HCO3'] = _ratio(mmol['SO4'], mmol['HCO3'])

    # sums and differences in meq/L
    k0 = meq['K'].fillna(0)
    v['Na+K (meq/L)'] = meq['Na'] + k0
    v['Ca+Mg (meq/L)'] = meq['Ca'] + meq['Mg']
    v['HCO3+SO4 (meq/L)'] = meq['HCO3'] + meq['SO4']
    v['Na+K-Cl (meq/L)'] = meq['Na'] + k0 - meq['Cl']
    v['Ca+Mg-HCO3-SO4 (meq/L)'] = meq['Ca'] + meq['Mg'] - meq['HCO3'] - meq['SO4']
    v['Ca+Mg-HCO3 (meq/L)'] = meq['Ca'] + meq['Mg'] - meq['HCO3']
    v['(Ca+Mg)/(HCO3+SO4)'] = _ratio(meq['Ca'] + meq['Mg'], meq['HCO3'] + meq['SO4'])
    # chloro-alkaline index (CAI-1)
    v['CAI-1'] = _ratio(meq['Cl'] - (meq['Na'] + k0), meq['Cl'])
    # Gibbs ratios use mg/L
    v['Na/(Na+Ca) (mg/L basis)'] = _ratio(mgl['Na'], mgl['Na'] + mgl['Ca'])
    v['Cl/(Cl+HCO3) (mg/L basis)'] = _ratio(mgl['Cl'], mgl['Cl'] + mgl['HCO3'])

    # PHREEQC
    for c in phreeqc.columns:
        v[c] = phreeqc[c]

    # Piper coordinates and water type
    pc = piper_coords(meq)
    for c in pc.columns:
        v['piper_' + c] = pc[c]
    pcat = meq['Ca'] + meq['Mg'] + meq['Na'] + k0
    pan = meq['HCO3'] + meq['CO3'].fillna(0) + meq['Cl'] + meq['SO4']
    v['Water type'] = [water_type(a, b) for a, b in
                       zip((meq['Ca'] + meq['Mg']) / pcat, (meq['SO4'] + meq['Cl']) / pan)]

    for c in extra_cols:
        v[c] = df[c]

    for c in filled.columns:
        v['filled_' + c] = filled[c]
    v['Filled'] = [', '.join(c for c in filled.columns if filled.at[i, c]) for i in filled.index]
    return pd.DataFrame(v, index=df.index)


# %% hollow markers
_HCO3_DEPENDENT = {'CBE (%)', 'Sum anions (meq/L)', 'TDS calc (mg/L)',
                   'Ca+Mg-HCO3-SO4 (meq/L)', 'Ca+Mg-HCO3 (meq/L)',
                   '(Ca+Mg)/(HCO3+SO4)', 'Cl/(Cl+HCO3) (mg/L basis)'}
_PHREEQC_EXTRA = {'log pCO2', 'Ionic strength (mol/kgw)', 'SC calc (uS/cm)'}


def hollow_mask(v, cols):
    # True where a filled value affects the plot (so it gets a hollow marker)
    mask = pd.Series(False, index=v.index)
    for c in cols:
        if c.startswith('SI ') or c in _PHREEQC_EXTRA:
            mask |= v['filled_pH'] | v['filled_Temp'] | v['filled_HCO3']
        elif 'HCO3' in c or c in _HCO3_DEPENDENT:
            mask |= v['filled_HCO3']
        elif c == 'TDS (mg/L)':
            mask |= v['filled_HCO3'] & ~v['TDS is measured']
        elif c == 'pH':
            mask |= v['filled_pH']
        elif c == 'Temp':
            mask |= v['filled_Temp']
        elif c == 'piper':
            mask |= v['filled_HCO3']
    return mask


INTERNAL_COLS = {'Sample', 'Label', 'Color', 'Marker', 'Water type', 'Filled',
                 'TDS is measured', 'filled_pH', 'filled_Temp', 'filled_HCO3'}


def plottable(v):
    # numeric columns for the custom plot dropdowns
    out = []
    for c in v.columns:
        if c in INTERNAL_COLS or c.startswith('piper_'):
            continue
        if pd.api.types.is_numeric_dtype(v[c]) and v[c].notna().any():
            out.append(c)
    return out
