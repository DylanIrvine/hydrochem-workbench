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

# hydrochem_workbench/chem.py
# template, reading the upload, unit conversion, filling gaps, PHREEQC,
# and the table of derived variables used by the plots

# %%
import io
import re

import numpy as np
import pandas as pd

# %% analytes
# name: (g/mol, charge, PHREEQC name)
# SiO2, B and the others with a charge of 0 here are read as mmol/L if the input unit is meq/L
ANALYTES = {
    'Ca':   (40.078,  +2, 'Ca'),
    'Mg':   (24.305,  +2, 'Mg'),
    'Na':   (22.990,  +1, 'Na'),
    'K':    (39.098,  +1, 'K'),
    'HCO3': (61.017,  -1, None),      # used as given if entered, otherwise from PHREEQC
    'CO3':  (60.009,  -2, None),      # used as given if entered, otherwise from PHREEQC
    'Cl':   (35.453,  -1, 'Cl'),
    'SO4':  (96.06,   -2, 'S(6)'),
    'NO3':  (62.004,  -1, 'N(5)'),
    'NO2':  (46.005,  -1, 'N(3)'),
    'NH4':  (18.039,  +1, 'Amm'),
    'PO4':  (94.971,   0, 'P'),       # charge left out of the balance (pH dependent, small)
    'Br':   (79.904,  -1, 'Br'),
    'F':    (18.998,  -1, 'F'),
    'SiO2': (60.084,   0, 'Si'),
    'Sr':   (87.62,   +2, 'Sr'),
    'Ba':   (137.327, +2, 'Ba'),
    'Fe':   (55.845,  +2, 'Fe(2)'),
    'Mn':   (54.938,  +2, 'Mn(2)'),
    'B':    (10.81,    0, 'B'),       # optional; boric acid, left out of the charge balance
    # minor and trace constituents, all optional. Those with a PHREEQC name are in phreeqc.dat and
    # go into the speciation; the rest are carried through for the tables, plots and guidelines
    'Li':   (6.94,    +1, 'Li'),
    'Al':   (26.982,   0, 'Al'),      # charge left out of the balance (pH dependent: Al(OH)3, Al(OH)4-)
    'Zn':   (65.38,   +2, 'Zn'),
    'Cu':   (63.546,  +2, 'Cu(2)'),
    'Cd':   (112.414, +2, 'Cd'),
    'Pb':   (207.2,   +2, 'Pb'),
    'Ni':   (58.693,  +2, None),      # not in phreeqc.dat; counted in the charge balance
    'I':    (126.904, -1, None),      # iodide; not in phreeqc.dat; counted in the charge balance
    'As':   (74.922,   0, None),      # not in phreeqc.dat; oxyanions, left out of the charge balance
    'Cr':   (51.996,   0, None),
    'Se':   (78.971,   0, None),
    'U':    (238.029,  0, None),
    'Hg':   (200.592,  0, None),
}
# these get columns in the results only if the dataset has them, so a dataset of major ions
# does not come back with dozens of empty columns
OPTIONAL = ['NO2', 'Li', 'Al', 'Zn', 'Cu', 'Cd', 'Pb', 'Ni', 'I', 'As', 'Cr', 'Se', 'U', 'Hg']
MW_SI = 28.0855

MAJORS = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'CO3', 'Cl', 'SO4']
CATIONS = [a for a, (_, z, _) in ANALYTES.items() if z > 0]
ANIONS = [a for a, (_, z, _) in ANALYTES.items() if z < 0]

ID_COLS = ['Sample', 'Label']
# optional, for the map: decimal degrees (south and west negative) and the depth
# of the bore screen in m below surface
LOC_COLS = ['Lat', 'Lon', 'Screen_depth']
# other spellings that are read as the same thing (compared in lower case)
LOC_ALIASES = {
    'Lat': ['lat', 'latitude', 'lat_dd', 'latitude_dd'],
    'Lon': ['lon', 'long', 'lng', 'longitude', 'lon_dd', 'long_dd', 'longitude_dd'],
    'Screen_depth': ['screen_depth', 'screen depth', 'screen_depth_m', 'screen depth (m)', 'depth', 'depth_m',
                     'depth (m)'],
}


def location_columns(columns):
    # which column holds Lat, Lon and Screen_depth, by the names above. None if not there
    lower = {str(c).strip().lower(): c for c in columns}
    return {k: next((lower[a] for a in names if a in lower), None) for k, names in LOC_ALIASES.items()}
FIELD_COLS = ['pH', 'Temp', 'EC', 'TDS']
# alkalinity as reported by most labs, always in mg/L as CaCO3
ALK_COLS = ['Total_Alk_CaCO3', 'Bicarb_Alk_CaCO3', 'Carb_Alk_CaCO3']
# dissolved inorganic carbon, used when there is no alkalinity
DIC_COL = 'DIC'
STYLE_COLS = ['Color', 'Marker']
# HCO3 and CO3 are optional. If entered they are used as given, otherwise PHREEQC
# calculates them from alkalinity (or DIC) and pH
_ions = [a for a in ANALYTES if a not in ('HCO3', 'CO3')]
_k = _ions.index('Cl')
TEMPLATE_COLS = (ID_COLS + LOC_COLS + FIELD_COLS + _ions[:_k] + ALK_COLS + [DIC_COL] + ['HCO3', 'CO3'] + _ions[_k:]
                 + STYLE_COLS)

UNITS = ['mg/L', 'mmol/L', 'meq/L']

# units the EC column can be in, with the factor to uS/cm. Everything after the input is in uS/cm
EC_UNITS = {'uS/cm': 1.0, 'mS/cm': 1000.0, 'mS/m': 10.0}
EC_LABELS = {'uS/cm': 'µS/cm', 'mS/cm': 'mS/cm (the same as dS/m)', 'mS/m': 'mS/m'}
_EC_ALIASES = {'µs/cm': 'uS/cm', 'us/cm': 'uS/cm', 'ms/cm': 'mS/cm', 'ds/m': 'mS/cm', 'ms/m': 'mS/m'}


def ec_unit_key(unit):
    # 'dS/m', 'µS/cm' and so on to the key used in EC_UNITS
    key = _EC_ALIASES.get(str(unit).strip().lower().replace('μ', 'µ'))
    if key is None:
        raise ValueError(f"EC unit must be one of {list(EC_UNITS)} (or 'dS/m'), not {unit!r}")
    return key


def ec_unit_check(v, ec_unit='uS/cm'):
    # compares the measured EC (already converted to uS/cm) with the EC expected from the ions: the conductivity
    # PHREEQC calculates, or 100 x the sum of cations in meq/L where the database has no conductivity data
    # returns a message if they differ by more than a factor of 3 (median of the samples), otherwise ''
    ec = v['EC'].astype(float)
    expected = v['SC calc (uS/cm)'].astype(float)
    expected = expected.where(expected > 0, 100.0 * v['Sum cations (meq/L)'].astype(float))
    ok = (ec > 0) & (expected > 0)
    if ok.sum() < 3:
        return ''
    ratio = float((ec[ok] / expected[ok]).median())
    if 1 / 3 <= ratio <= 3:
        return ''
    msg = (f'Measured EC is about {ratio:.2g} times the EC expected from the ions (median of {int(ok.sum())} '
           f'samples), with the EC column read as {EC_LABELS[ec_unit_key(ec_unit)]}.')
    # the unit that would bring the two into line, if there is one
    implied = EC_UNITS[ec_unit_key(ec_unit)] / ratio
    for key, factor in EC_UNITS.items():
        if factor / 3 <= implied <= factor * 3 and key != ec_unit_key(ec_unit):
            return msg + f' The values look like {EC_LABELS[key]}. Check the EC unit.'
    return msg + ' Check the EC column and its unit.'

# mg/L as CaCO3 per meq/L of alkalinity (100.087 / 2)
CACO3_PER_MEQ = 50.0435
# as N or as P -> the ion
N_TO_NO3 = 4.4267    # 62.004 / 14.007
N_TO_NH4 = 1.2879    # 18.039 / 14.007
P_TO_PO4 = 3.0662    # 94.971 / 30.974
MW_N, MW_P, MW_C, MW_NH3 = 14.007, 30.974, 12.011, 17.031

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
    'Hydroxyapatite': ['Ca', 'PO4'],
}

# warn above this ionic strength (phreeqc.dat is not for saline water, seawater ~0.7)
IONIC_STRENGTH_WARN = 0.5


# %% template
def template_bytes():
    # xlsx template: Data sheet + Notes sheet
    example = example_data().head(3)
    notes = pd.DataFrame({'Column': [
        'Sample', 'Label', 'Lat', 'Lon', 'Screen_depth', 'pH', 'Temp', 'EC', 'TDS',
        'Ca, Mg, Na, K, Cl, SO4 ...',
        'ALKALINITY', 'Total_Alk_CaCO3', 'Bicarb_Alk_CaCO3', 'Carb_Alk_CaCO3', 'DIC', '',
        'HCO3', 'CO3',
        'NO3, NO2, NH4', 'PO4', 'SiO2', 'B', 'Li, Al, Zn, Cu, Cd, Pb', 'Ni, I, As, Cr, Se, U, Hg', 'Color', 'Marker',
        '(any other column)', '', 'Units', 'Blank cells', 'Below detection'],
        'What goes in it': [
            'Unique sample name. Required.',
            'Group used for colours and legends (aquifer, site, campaign). Optional.',
            'Latitude in decimal degrees, WGS84 or GDA94/GDA2020 (south is negative, e.g. -14.465). Optional, used for the map only.',
            'Longitude in decimal degrees (west is negative, e.g. 132.264). Optional, used for the map only.',
            'Depth of the bore screen in m below surface (one value, e.g. the middle of the screen). Optional. Can set the marker size on the map; leave blank for surface water.',
            'Field pH if you have it, otherwise lab pH.',
            'Water temperature, degrees C.',
            'Electrical conductivity at 25 degrees C. Optional. In uS/cm unless you choose mS/cm (dS/m) or mS/m in the app. Used for the irrigation classes, the analysis checks and plots.',
            'Total dissolved solids, mg/L. Optional. If blank, it is calculated from the ions.',
            'Concentrations, all in the single unit you choose in the app (mg/L, mmol/L or meq/L).',
            'Enter alkalinity as it appears on the lab report, in mg/L as CaCO3. Do not convert it. PHREEQC uses it (with pH) for the speciation, and calculates HCO3 and CO3 from it unless you enter them.',
            'Total alkalinity, mg/L as CaCO3. Use this column if the report gives it.',
            'Bicarbonate alkalinity, mg/L as CaCO3. Use this (with the next column) if the report has no total.',
            'Carbonate alkalinity, mg/L as CaCO3. Often below detection: leave blank.',
            'Dissolved inorganic carbon, for samples with no alkalinity. mg/L as C or mmol/L (tell the app which). PHREEQC calculates HCO3 and CO3 from DIC and pH.',
            'If total alkalinity is filled it is used. Otherwise bicarbonate + carbonate alkalinity, otherwise HCO3 + CO3 (next columns), otherwise DIC. The alkalinity columns are always mg/L as CaCO3, whatever unit is chosen for the ions.',
            'Optional. Bicarbonate as the ion, in the same unit as the other ions, if you already have it (e.g. from the lab report). Used as given. Leave blank to have PHREEQC calculate it.',
            'Optional. Carbonate as the ion, same unit as the other ions. Used as given. If both HCO3 and CO3 are blank, PHREEQC calculates both. If only one is entered, the other is the rest of the alkalinity (HCO3 + 2 CO3 = alkalinity, in meq/L), so carbonate is not counted twice. A CO3 below detection can be entered as 0 or as <LOR.',
            'Nitrate, nitrite and ammonium (total ammonia), as the ion or as N (tell the app which, one choice for all three). Lab reports usually give NO3-N, NO2-N and NH3-N, i.e. as N. Nitrite is optional.',
            'Phosphate (filterable reactive phosphorus), as PO4 or as P (tell the app which).',
            'Silica as SiO2. Neutral, so if the unit is meq/L enter it in mmol/L. If your report gives silicon, add a column called Si instead and leave SiO2 blank: it is converted.',
            'Boron, as B. Optional; used for the drinking water and irrigation guidelines. Neutral, so if the unit is meq/L enter it in mmol/L.',
            'Optional minor and trace constituents, as the element, in the same unit as the other ions (trace metals are often reported in ug/L: divide by 1000 for mg/L). They go into the PHREEQC speciation. Li, Zn, Cu, Cd and Pb are counted in the charge balance; Al is not (its charge depends on pH). Delete the columns you do not need.',
            'Optional trace constituents, as the element, same unit. Not in the phreeqc.dat database, so they are not in the speciation; they are carried through to the tables, plots and statistics. Ni and I are counted in the charge balance; the others are not. If the unit is meq/L, enter As, Cr, Se, U and Hg (and Al) in mmol/L.',
            'Optional matplotlib colour for this row (e.g. #0089d0). Blank = automatic by Label.',
            'Optional matplotlib marker for this row (e.g. o, s, ^, D). Blank = automatic by Label.',
            'Any extra numeric column (depth, d18O, 87Sr/86Sr ...) is carried through and can be plotted.',
            '',
            'One unit for every ion column. Do not mix. (Alkalinity as CaCO3 is the exception, see above.)',
            'Leave a cell blank if not measured. Missing pH, Temp and alkalinity are filled by the rule you pick in the app and drawn hollow.',
            'Values such as <0.05, and negative values used to mean below the LOR, can be replaced in the app by a fraction of the LOR (half by default), or left out. Text with no number (e.g. "bd") is always left out and listed in the app.',
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
    # alkalinity as a lab would report it (some as total, some as bicarbonate)
    alk = (df['HCO3'] / ANALYTES['HCO3'][0] * CACO3_PER_MEQ).round(1)
    for c in ALK_COLS:
        df[c] = np.nan
    sand = df['Label'].str.startswith('Sandstone')
    df.loc[~sand, 'Total_Alk_CaCO3'] = alk[~sand]
    df.loc[sand, 'Bicarb_Alk_CaCO3'] = alk[sand]
    # one sample with DIC and no alkalinity
    df[DIC_COL] = np.nan
    df.loc[7, DIC_COL] = round(df.loc[7, 'HCO3'] / ANALYTES['HCO3'][0] * MW_C * 1.08, 1)
    df.loc[7, ALK_COLS] = np.nan
    # nutrients (as the ion), with one high nitrate
    df['NH4'] = rng.uniform(0.01, 0.12, len(df)).round(3)
    df['PO4'] = rng.uniform(0.005, 0.06, len(df)).round(3)
    df['B'] = np.where(df['Label'].str.startswith('Saline'), rng.uniform(1.0, 4.0, len(df)),
                       rng.uniform(0.02, 0.4, len(df))).round(3)
    df.loc[12, 'NO3'] = 62.0
    # HCO3 and CO3 as a lab would report them (alkalinity x 1.219, CO3 below detection),
    # for two samples only. The rest are left to PHREEQC
    lab = df.index.isin([4, 5])
    df.loc[lab, 'HCO3'] = (alk[lab] * 1.2192).round(1)
    df.loc[lab, 'CO3'] = 0.0
    df.loc[~lab, ['HCO3', 'CO3']] = np.nan
    df['Color'] = ''
    df['Marker'] = ''
    # made-up locations and screen depths for the map, one cluster for each group.
    # A separate generator, so the chemistry above does not change
    loc = np.random.default_rng(11)
    centres = {'Carbonate': (-14.62, 132.48, 20, 80), 'Sandstone': (-14.30, 132.62, 40, 150),
               'Saline': (-14.52, 132.95, 5, 30)}
    for name, (lat0, lon0, d0, d1) in centres.items():
        g = df['Label'].str.startswith(name)
        df.loc[g, 'Lat'] = (lat0 + loc.normal(0, 0.06, g.sum())).round(4)
        df.loc[g, 'Lon'] = (lon0 + loc.normal(0, 0.07, g.sum())).round(4)
        df.loc[g, 'Screen_depth'] = loc.uniform(d0, d1, g.sum()).round(0)
    df.loc[24, 'Screen_depth'] = np.nan    # one sample with no depth
    # the optional minor and trace constituents are left blank in the example
    for c in OPTIONAL:
        df[c] = np.nan
    return df[TEMPLATE_COLS]


# %% below the limit of reporting
# "<0.05", "< 0.05", "<=0.05", "<0.05 mg/L"
_BELOW = re.compile(r'^\s*<\s*=?\s*((?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*[^\d\s]*\s*$')
# columns where a negative number can only mean below the LOR
CONC_COLS = list(ANALYTES) + ALK_COLS + [DIC_COL]
# never touched: identifiers, styles and the field values that are not concentrations
_NOT_CONC = set(ID_COLS + STYLE_COLS + ['pH', 'Temp', 'EC'])


def below_lor(raw, fraction=0.5, text=True, negative=True, rows=False):
    # replace below-LOR values with fraction x LOR, before clean()
    # text: "<0.05" style cells, in the concentration columns, TDS and any extra column
    # negative: negative numbers in the concentration columns (labs that report -0.05 for <0.05)
    # returns (raw with the values replaced, table of what was replaced)
    # rows=True adds a Row column (the row index), used by the uncertainty analysis
    out = raw.copy()
    found = []
    sample = out['Sample'] if 'Sample' in out.columns else pd.Series(out.index, index=out.index)
    for c in out.columns:
        if c in _NOT_CONC:
            continue
        for i, val in out[c].items():
            lor = None
            if text and isinstance(val, str):
                m = _BELOW.match(val)
                if m:
                    lor = float(m.group(1))
            if lor is None and negative and c in CONC_COLS:
                num = pd.to_numeric(val, errors='coerce')
                if pd.notna(num) and num < 0:
                    lor = abs(float(num))
            if lor is not None:
                out.at[i, c] = lor * fraction
                found.append({'Sample': sample[i], 'Column': c, 'Value': val, 'LOR': lor,
                              'Used': lor * fraction, 'Row': i})
    cols = ['Sample', 'Column', 'Value', 'LOR', 'Used'] + (['Row'] if rows else [])
    return out, pd.DataFrame(found, columns=['Sample', 'Column', 'Value', 'LOR', 'Used', 'Row'])[cols]


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
    missing_majors = [c for c in MAJORS if c not in df.columns and c not in ('HCO3', 'CO3')]
    if not any(c in df.columns for c in ['HCO3', DIC_COL] + ALK_COLS):
        missing_majors.append('alkalinity (Total_Alk_CaCO3 or Bicarb_Alk_CaCO3) or DIC')
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
    for c in FIELD_COLS + ALK_COLS + [DIC_COL] + list(ANALYTES):
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
        known = c in FIELD_COLS or c in ALK_COLS or c == DIC_COL or c in ANALYTES
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
    for c in list(ANALYTES) + ALK_COLS + [DIC_COL]:
        neg = df[c] < 0
        for i in df.index[neg]:
            dropped.append({'Sample': df.at[i, 'Sample'], 'Column': c,
                            'Value': df.at[i, c]})
        df.loc[neg, c] = np.nan

    return df, pd.DataFrame(dropped, columns=['Sample', 'Column', 'Value']), extra_cols, problems


# %% units
def to_mmol(df, unit, n_as_n=False, p_as_p=False, dic_unit='mg/L as C'):
    # ions to mmol/L and alkalinity to meq/L (mmol['Alk'])
    # returns (mmol, where the alkalinity came from for each sample)
    mmol = pd.DataFrame(index=df.index)
    for a, (mw, z, _) in ANALYTES.items():
        x = df[a].astype(float)
        if unit == 'mg/L':
            mmol[a] = x / mw
        elif unit == 'mmol/L':
            mmol[a] = x
        else:
            mmol[a] = x / abs(z) if z != 0 else x

    # silica entered as silicon (a column called Si) is used where SiO2 is blank
    if 'Si' in df.columns:
        si = pd.to_numeric(df['Si'], errors='coerce')
        mmol['SiO2'] = mmol['SiO2'].where(mmol['SiO2'].notna(), si / MW_SI if unit == 'mg/L' else si)

    # alkalinity in meq/L: total if given, otherwise bicarbonate + carbonate.
    # the as CaCO3 columns are always mg/L
    total = df['Total_Alk_CaCO3'].astype(float) / CACO3_PER_MEQ
    bicarb = df['Bicarb_Alk_CaCO3'].astype(float) / CACO3_PER_MEQ
    carb = df['Carb_Alk_CaCO3'].astype(float) / CACO3_PER_MEQ
    # HCO3 and CO3 entered by the user: kept to replace the PHREEQC values later,
    # and turned into alkalinity for samples with no alkalinity columns
    mmol['HCO3_in'] = mmol['HCO3']
    mmol['CO3_in'] = mmol['CO3']
    ions = mmol['HCO3'] + 2 * mmol['CO3'].fillna(0)

    mmol['Alk'] = np.nan
    source = pd.Series('', index=df.index)
    for vals, name in [(ions, 'HCO3 and CO3 columns'),
                       (bicarb + carb.fillna(0), 'Bicarbonate + carbonate alkalinity'),
                       (total, 'Total alkalinity')]:
        ok = vals.notna()
        mmol.loc[ok, 'Alk'] = vals[ok]
        source[ok] = name
    # HCO3 and CO3 are set later, from PHREEQC (or the user's columns)
    mmol['HCO3'] = np.nan
    mmol['CO3'] = np.nan

    # DIC is only used where there is no alkalinity
    dic = df[DIC_COL].astype(float)
    mmol['DIC'] = dic / MW_C if dic_unit == 'mg/L as C' else dic
    use_d = mmol['Alk'].isna() & mmol['DIC'].notna()
    source[use_d] = 'DIC'
    mmol.loc[~use_d, 'DIC'] = np.nan

    # as N / as P only changes anything for mg/L (a mole of NO3-N is a mole of NO3)
    if unit == 'mg/L':
        if n_as_n:
            mmol['NO3'] = df['NO3'].astype(float) / MW_N
            mmol['NH4'] = df['NH4'].astype(float) / MW_N
            mmol['NO2'] = df['NO2'].astype(float) / MW_N
        if p_as_p:
            mmol['PO4'] = df['PO4'].astype(float) / MW_P
    return mmol, source


# %% fill
def fill_missing(df, mmol, rules):
    # fill missing pH, Temp and alkalinity. rules = {'pH': ('mean' or 'value', number), ...}
    # alkalinity ('HCO3' key) can also be 'balance' (from charge balance)
    # the alkalinity value is in mg/L as CaCO3
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
    val = val / CACO3_PER_MEQ
    has_other = mmol[['Ca', 'Mg', 'Na', 'Cl']].notna().any(axis=1)
    gap = mmol['Alk'].isna() & mmol['DIC'].isna() & has_other
    if gap.any():
        if how == 'balance':
            cat = sum(mmol[a].fillna(0) * ANALYTES[a][1] for a in CATIONS)
            an = sum(mmol[a].fillna(0) * abs(ANALYTES[a][1]) for a in ANIONS if a not in ('HCO3', 'CO3'))
            fill = (cat - an).clip(lower=0)[gap]
        elif how == 'mean':
            fill = mmol['Alk'].mean()
            if pd.isna(fill):
                fill = val
        else:
            fill = val
        mmol.loc[gap, 'Alk'] = fill
        filled.loc[gap, 'HCO3'] = True

    return df, mmol, filled


# %% PHREEQC
def run_phreeqc(df, mmol, pp):
    # run each sample through phreeqc.dat with pH and alkalinity (or DIC) as inputs.
    # HCO3 and CO3 come back from the speciation and are written into mmol:
    #   HCO3 = HCO3- plus its ion pairs (CaHCO3+, MgHCO3+, NaHCO3 ...)
    #   CO3  = CO3-2 plus its ion pairs (CaCO3, MgCO3, NaCO3- ...)
    # so HCO3 + 2*CO3 is the carbonate alkalinity
    mmol = mmol.copy()
    keys = phreeqc_keys(pp)
    out = pd.DataFrame(index=df.index, dtype=float)
    cols = ['SI ' + m for m in MINERALS] + ['log pCO2', 'DIC (mmol/L)', 'Ionic strength (mol/kgw)',
                                            'SC calc (uS/cm)']
    for c in cols:
        out[c] = np.nan
    errors = []

    for i in df.index:
        row = mmol.loc[i]
        alk = row['Alk'] if pd.notna(row['Alk']) else 0.0
        dic = row['DIC'] if pd.notna(row['DIC']) else 0.0
        if pd.isna(df.at[i, 'pH']) or row[['Ca', 'Mg', 'Na', 'Cl']].isna().all():
            continue
        sol = {'units': 'mmol/l', 'pH': float(df.at[i, 'pH']),
               'temp': float(df.at[i, 'Temp']) if pd.notna(df.at[i, 'Temp']) else 25.0}
        for a, key in keys.items():
            if pd.notna(row[a]) and row[a] > 0:
                sol[key] = float(row[a])
        if alk > 0:
            sol['Alkalinity'] = float(alk)
        elif dic > 0:
            sol['C(4)'] = float(dic)

        try:
            s = pp.add_solution(sol)
        except Exception as exc:   # did not converge
            errors.append({'Sample': df.at[i, 'Sample'], 'Error': str(exc)[:200]})
            # fall back to all alkalinity as HCO3 so the sample still plots
            if alk > 0 or dic > 0:
                mmol.at[i, 'HCO3'] = alk if alk > 0 else dic
                mmol.at[i, 'CO3'] = 0.0
            continue

        if alk > 0 or dic > 0:
            moles = s.species_moles
            # per kg of water, the same basis PHREEQC used for the inputs
            per_l = 1000.0 / s.mass
            hco3 = sum(m for k, m in moles.items() if 'HCO3' in k) * per_l
            co3 = sum(m for k, m in moles.items() if 'CO3' in k and 'HCO3' not in k) * per_l
            co2 = sum(m * (2 if k == '(CO2)2' else 1) for k, m in moles.items() if 'CO2' in k) * per_l
            mmol.at[i, 'HCO3'] = hco3
            mmol.at[i, 'CO3'] = co3
            out.at[i, 'DIC (mmol/L)'] = hco3 + co3 + co2
            out.at[i, 'log pCO2'] = s.si('CO2(g)')
            if alk <= 0:
                # DIC sample: report the carbonate alkalinity that goes with it
                mmol.at[i, 'Alk'] = hco3 + 2 * co3
        for m, needs in MINERALS.items():
            have = [max(alk, dic) if n == 'HCO3' else row[n] for n in needs]
            if all(pd.notna(x) and x > 0 for x in have):
                try:
                    si = s.si(m)
                except Exception:      # the mineral is not in this database
                    si = -999.0
                out.at[i, 'SI ' + m] = si if si > -99 else np.nan
        out.at[i, 'Ionic strength (mol/kgw)'] = s.I
        # 0 means the database has no conductivity data (wateq4f)
        out.at[i, 'SC calc (uS/cm)'] = s.sc if s.sc > 0 else np.nan
        s.forget()

    # entered HCO3 and CO3 replace the calculated values. If only one is entered,
    # the other is what is left of the alkalinity (meq/L), so HCO3 + 2 CO3 still adds
    # up to it and carbonate is not counted twice
    h, c, alk = mmol['HCO3_in'], mmol['CO3_in'], mmol['Alk']
    both = h.notna() & c.notna()
    h_only = h.notna() & c.isna()
    c_only = c.notna() & h.isna()
    mmol.loc[both, 'HCO3'] = h[both]
    mmol.loc[both, 'CO3'] = c[both]
    mmol.loc[h_only, 'HCO3'] = h[h_only]
    mmol.loc[h_only, 'CO3'] = ((alk - h) / 2).clip(lower=0)[h_only]
    mmol.loc[c_only, 'CO3'] = c[c_only]
    mmol.loc[c_only, 'HCO3'] = (alk - 2 * c).clip(lower=0)[c_only]
    return out, pd.DataFrame(errors, columns=['Sample', 'Error']), mmol


# %% databases
# phreeqc.dat is the default. pitzer.dat (for saline water and brines) ships with phreeqpython;
# wateq4f.dat (more trace elements) is the USGS file, in hydrochem_workbench/data
DATABASES = {'phreeqc.dat': 'phreeqc.dat (fresh water; the default)',
             'wateq4f.dat': 'wateq4f.dat (fresh water, more trace elements)',
             'pitzer.dat': 'pitzer.dat (saline water and brines)'}
# constituents that are not in phreeqc.dat but are in wateq4f.dat, and are not redox sensitive
_OTHER_KEYS = {'Ni': 'Ni', 'I': 'I'}


def database_elements(name):
    # the elements a database defines (its SOLUTION_MASTER_SPECIES)
    from pathlib import Path
    import phreeqpython
    own = Path(__file__).resolve().parent / 'data' / name
    path = own if own.exists() else Path(phreeqpython.__file__).resolve().parent / 'database' / name
    text = path.read_text(errors='ignore')
    start = text.index('SOLUTION_MASTER_SPECIES')
    block = text[start:text.index('SOLUTION_SPECIES', start + 30)]
    return {line.split()[0].split('(')[0] for line in block.splitlines()[1:] if line.split('#')[0].split()}


def phreeqc_keys(pp):
    # {analyte: PHREEQC name} for the constituents this database can take.
    # pp.hw_elements is set by workflow.get_phreeqc(); without it the database is taken to be phreeqc.dat
    elements = getattr(pp, 'hw_elements', None)
    keys = {}
    for a, (_, _, key) in ANALYTES.items():
        if key is None and elements is not None:
            key = _OTHER_KEYS.get(a)
        if key == 'Amm' and elements is not None and 'Amm' not in elements:
            key = 'N(-3)'      # phreeqc.dat keeps ammonium apart from the nitrogen redox system; others do not
        if key is not None and (elements is None or key.split('(')[0] in elements):
            keys[a] = key
    return keys


def left_out_of_speciation(df, pp):
    # the constituents the dataset has that this database cannot take (they stay in the tables and plots)
    keys = phreeqc_keys(pp)
    return [a for a in ANALYTES if a not in keys and a not in ('HCO3', 'CO3') and a in df.columns
            and pd.to_numeric(df[a], errors='coerce').notna().any()]


# %% the whole calculation for one dataset
def derive(df, pp, unit='mg/L', n_as_n=False, p_as_p=False, dic_unit='mg/L as C', rules=None,
           extra_cols=()):
    # cleaned data -> results table. Used for the measured data and for every
    # realisation of the uncertainty analysis, so both go through the same steps
    rules = dict(rules) if rules is not None else {'pH': ('mean', 7.0), 'Temp': ('mean', 25.0),
                                                    'HCO3': ('mean', 100.0)}
    mmol, alk_source = to_mmol(df, unit, n_as_n, p_as_p, dic_unit)
    df, mmol, filled = fill_missing(df, mmol, rules)
    phreeqc, errors, mmol = run_phreeqc(df, mmol, pp)
    return build_table(df, mmol, filled, phreeqc, list(extra_cols), alk_source), errors


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


# Chadha (1999) fields 5 to 8 (the quadrants of the diagram). Fields 1 to 4 are
# the half-planes, so every sample is also in two of them. Many later papers number
# the quadrants 1 to 4 instead, so the type is given with the number
CHADHA_FIELDS = {
    5: 'Ca-Mg-HCO3 type: alkaline earths exceed alkali metals and weak acid anions exceed strong acid anions',
    6: 'Ca-Mg-Cl/SO4 type: alkaline earths exceed alkali metals and strong acid anions exceed weak acid anions',
    7: 'Na-Cl/SO4 type: alkali metals exceed alkaline earths and strong acid anions exceed weak acid anions',
    8: 'Na-HCO3 type: alkali metals exceed alkaline earths and weak acid anions exceed strong acid anions',
}


def chadha_field(x, y):
    # x = meq% (Ca+Mg) - (Na+K) of cations; y = meq% (HCO3+CO3) - (Cl+SO4) of anions
    if pd.isna(x) or pd.isna(y):
        return ''
    if x == 0 or y == 0:
        return 'On a field boundary'
    if x > 0:
        return '5' if y > 0 else '6'
    return '8' if y > 0 else '7'


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


def build_table(df, mmol, filled, phreeqc, extra_cols, alk_source=None):
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
    shown = [a for a in ANALYTES if a not in OPTIONAL or mmol[a].notna().any()]
    for a in shown:
        v[f'{a} (mmol/L)'] = mmol[a]
    for a in shown:
        if ANALYTES[a][1] != 0:
            v[f'{a} (meq/L)'] = meq[a]
    for a in shown:
        v[f'{a} (mg/L)'] = mgl[a]

    # forms used by the water quality guidelines
    v['NO3-N (mg/L)'] = mmol['NO3'] * MW_N
    v['NH4-N (mg/L)'] = mmol['NH4'] * MW_N
    v['Ammonia as NH3 (mg/L)'] = mmol['NH4'] * MW_NH3
    v['PO4-P (mg/L)'] = mmol['PO4'] * MW_P
    if mmol['NO2'].notna().any():
        v['NO2-N (mg/L)'] = mmol['NO2'] * MW_N
    v['Hardness (mg/L as CaCO3)'] = (mmol['Ca'] + mmol['Mg'].fillna(0)) * 100.087
    # the same hardness in French, German and English (Clark) degrees
    v['Hardness (°fH)'] = v['Hardness (mg/L as CaCO3)'] / 10.0
    v['Hardness (°dH)'] = v['Hardness (mg/L as CaCO3)'] / 17.848
    v['Hardness (°e)'] = v['Hardness (mg/L as CaCO3)'] / 14.254
    v['Alkalinity (meq/L)'] = mmol['Alk']
    v['Alkalinity (mg/L as CaCO3)'] = mmol['Alk'] * CACO3_PER_MEQ

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
    # source-rock style ratios (after Hounslow, 1995)
    v['HCO3/sum anions (meq)'] = _ratio(meq['HCO3'], meq['HCO3'] + meq['CO3'].fillna(0) + meq['Cl'] + meq['SO4'])
    v['Na/(Na+Cl) (meq)'] = _ratio(meq['Na'], meq['Na'] + meq['Cl'])
    v['Ca/(Ca+SO4) (meq)'] = _ratio(meq['Ca'], meq['Ca'] + meq['SO4'])
    v['Mg/(Ca+Mg) (meq)'] = _ratio(meq['Mg'], meq['Ca'] + meq['Mg'])
    v['SiO2/(Na+K-Cl) (mmol/meq)'] = _ratio(mmol['SiO2'], meq['Na'] + k0 - meq['Cl'])
    # Na-normalised ratios for the Gaillardet mixing diagrams
    v['Ca/Na'] = _ratio(mmol['Ca'], mmol['Na'])
    v['Mg/Na'] = _ratio(mmol['Mg'], mmol['Na'])
    v['HCO3/Na'] = _ratio(mmol['HCO3'], mmol['Na'])
    # irrigation: sodium adsorption ratio and percent sodium (meq)
    v['SAR'] = meq['Na'] / np.sqrt((meq['Ca'] + meq['Mg']) / 2).where(lambda z: z > 0)
    v['Na (%)'] = _ratio(meq['Na'] + k0, meq['Ca'] + meq['Mg'] + meq['Na'] + k0) * 100
    # chloro-alkaline indices (Schoeller): CAI-1 and CAI-2
    c0, n0 = meq['CO3'].fillna(0), meq['NO3'].fillna(0)
    v['CAI-1'] = _ratio(meq['Cl'] - (meq['Na'] + k0), meq['Cl'])
    v['CAI-2'] = _ratio(meq['Cl'] - (meq['Na'] + k0), meq['SO4'] + meq['HCO3'] + c0 + n0)
    # irrigation indices (meq/L): residual sodium carbonate (Eaton, 1950), Kelley's ratio (Kelley, 1940),
    # permeability index (Doneen, 1964) and magnesium hazard (Szabolcs and Darab, 1964)
    v['RSC (meq/L)'] = (meq['HCO3'] + c0) - (meq['Ca'] + meq['Mg'])
    v['Kelley ratio'] = _ratio(meq['Na'], meq['Ca'] + meq['Mg'])
    v['Permeability index (%)'] = _ratio(meq['Na'] + np.sqrt(meq['HCO3']), meq['Ca'] + meq['Mg'] + meq['Na']) * 100
    v['Magnesium hazard (%)'] = _ratio(meq['Mg'], meq['Ca'] + meq['Mg']) * 100
    # Gibbs ratios use mg/L
    v['Na/(Na+Ca) (mg/L basis)'] = _ratio(mgl['Na'], mgl['Na'] + mgl['Ca'])
    v['Cl/(Cl+HCO3) (mg/L basis)'] = _ratio(mgl['Cl'], mgl['Cl'] + mgl['HCO3'])

    # PHREEQC
    for c in phreeqc.columns:
        v[c] = phreeqc[c]
    # scaling and corrosion: Langelier (1936) and Ryznar (1944) indices. pHs, the pH at which the water
    # would be saturated with calcite, comes from the PHREEQC calcite saturation index
    v['pHs (calcite)'] = df['pH'] - phreeqc['SI Calcite']
    v['Langelier index'] = df['pH'] - v['pHs (calcite)']
    v['Ryznar index'] = 2 * v['pHs (calcite)'] - df['pH']

    # Piper coordinates and water type
    pc = piper_coords(meq)
    for c in pc.columns:
        v['piper_' + c] = pc[c]
    pcat = meq['Ca'] + meq['Mg'] + meq['Na'] + k0
    pan = meq['HCO3'] + meq['CO3'].fillna(0) + meq['Cl'] + meq['SO4']
    v['Water type'] = [water_type(a, b) for a, b in
                       zip((meq['Ca'] + meq['Mg']) / pcat, (meq['SO4'] + meq['Cl']) / pan)]
    # Chadha (1999) diagram coordinates and field
    v['Chadha x (meq%)'] = 100 * (meq['Ca'] + meq['Mg'] - meq['Na'] - k0) / pcat
    v['Chadha y (meq%)'] = 100 * (meq['HCO3'] + meq['CO3'].fillna(0) - meq['Cl'] - meq['SO4']) / pan
    v['Chadha field'] = [chadha_field(a, b) for a, b in zip(v['Chadha x (meq%)'], v['Chadha y (meq%)'])]

    for c in extra_cols:
        v[c] = df[c]

    for c in filled.columns:
        v['filled_' + c] = filled[c]
    v['Alkalinity from'] = alk_source if alk_source is not None else ''
    v['Alkalinity from'] = ['Filled' if f else a for a, f in zip(v['Alkalinity from'], filled['HCO3'])]
    hco3_in = mmol['HCO3_in'].notna() if 'HCO3_in' in mmol else pd.Series(False, index=df.index)
    co3_in = mmol['CO3_in'].notna() if 'CO3_in' in mmol else pd.Series(False, index=df.index)
    v['HCO3 is input'] = hco3_in
    v['CO3 is input'] = co3_in
    v['HCO3, CO3 from'] = ['Entered' if h and c else 'HCO3 entered, CO3 = alkalinity - HCO3' if h else
                           'CO3 entered, HCO3 = alkalinity - 2 CO3' if c else 'PHREEQC'
                           for h, c in zip(hco3_in, co3_in)]
    names = {'pH': 'pH', 'Temp': 'Temp', 'HCO3': 'Alkalinity'}
    v['Filled'] = [', '.join(names[c] for c in filled.columns if filled.at[i, c]) for i in filled.index]
    return pd.DataFrame(v, index=df.index)


# %% hollow markers
_HCO3_DEPENDENT = {'CBE (%)', 'Sum anions (meq/L)', 'TDS calc (mg/L)', 'CAI-2', 'RSC (meq/L)',
                   'Permeability index (%)'}
_PHREEQC_EXTRA = {'log pCO2', 'DIC (mmol/L)', 'Ionic strength (mol/kgw)', 'SC calc (uS/cm)', 'pHs (calcite)',
                  'Langelier index', 'Ryznar index'}


def hollow_mask(v, cols):
    # True where a filled value affects the plot (so it gets a hollow marker)
    mask = pd.Series(False, index=v.index)
    for c in cols:
        if c.startswith('SI ') or c in _PHREEQC_EXTRA:
            mask |= v['filled_pH'] | v['filled_Temp'] | v['filled_HCO3']
        elif c.startswith('Alkalinity') or c in _HCO3_DEPENDENT:
            mask |= v['filled_HCO3']
        elif 'CO3' in c:
            # the HCO3 / CO3 split comes from pH, unless the user entered either ion
            entered = v['HCO3 is input'] | v['CO3 is input']
            mask |= v['filled_HCO3'] | (v['filled_pH'] & ~entered)
        elif c == 'TDS (mg/L)':
            mask |= v['filled_HCO3'] & ~v['TDS is measured']
        elif c == 'pH':
            mask |= v['filled_pH']
        elif c == 'Temp':
            mask |= v['filled_Temp']
        elif c == 'piper':
            mask |= v['filled_HCO3']
    return mask


def group_means(v):
    # mean of each analyte (mg/L) and TDS for every Label, with the number of values behind each mean.
    # Values that rest on a filled pH, temperature or alkalinity are left out (the same values that
    # are hollow on the plots); below-LOR values count at their replaced value
    cols = [f'{a} (mg/L)' for a in ANALYTES if f'{a} (mg/L)' in v.columns and v[f'{a} (mg/L)'].notna().any()]
    cols += ['TDS (mg/L)']
    means, counts = [], []
    for lab in pd.unique(v['Label']):
        g = v[v['Label'] == lab]
        m = {'Label': lab, 'Samples': len(g)}
        n = {'Label': lab, 'Samples': len(g)}
        for c in cols:
            vals = g[c].where(~hollow_mask(g, [c])).dropna()
            m[c] = vals.mean() if len(vals) else np.nan
            n[c] = len(vals)
        means.append(m)
        counts.append(n)
    return pd.DataFrame(means), pd.DataFrame(counts)


STAT_COLS = ['n', 'Missing (%)', 'Min', 'P25', 'Median', 'Mean', 'P75', 'Max', 'SD', 'CV (%)']
ALL_SAMPLES = 'All samples'


def group_stats(v, cols, by_group=True):
    # descriptive statistics of the chosen columns, for all samples and for each Label
    # one row for each group and variable. As for the group means, values that rest on a filled pH,
    # temperature or alkalinity are left out, and below-LOR values count at their replaced value
    groups = [(ALL_SAMPLES, v)]
    if by_group and v['Label'].nunique() > 1:
        groups += [(lab, v[v['Label'] == lab]) for lab in pd.unique(v['Label'])]
    rows = []
    for name, g in groups:
        for c in cols:
            if c not in g.columns:
                continue
            x = pd.to_numeric(g[c], errors='coerce').where(~hollow_mask(g, [c])).dropna()
            row = {'Group': name, 'Variable': c, 'n': len(x),
                   'Missing (%)': 100 * (1 - len(x) / len(g)) if len(g) else np.nan}
            if len(x):
                q = x.quantile([0.25, 0.5, 0.75])
                sd = x.std(ddof=1) if len(x) > 1 else np.nan
                row.update({'Min': x.min(), 'P25': q[0.25], 'Median': q[0.5], 'Mean': x.mean(), 'P75': q[0.75],
                            'Max': x.max(), 'SD': sd,
                            'CV (%)': 100 * sd / x.mean() if x.mean() != 0 and pd.notna(sd) else np.nan})
            rows.append(row)
    return pd.DataFrame(rows, columns=['Group', 'Variable'] + STAT_COLS)


def stat_defaults(v):
    # the field values and major ions that were measured
    cols = ['pH', 'EC', 'TDS (mg/L)'] + [f'{a} (mg/L)' for a in MAJORS]
    return [c for c in cols if c in v.columns and v[c].notna().any()]


# %% charge balance check
CB_COL = 'Charge balance check'
CB_WITHIN, CB_BEYOND, CB_NA = 'Within limit', 'Beyond limit', 'Not assessable'
# set by the app for the plots: the ring colour of a marked sample ('' if not marked) and its legend text
CB_RING = 'cb_ring'
CB_LABEL = 'cb_ring_label'


def charge_balance_check(v, limit=5.0, lo=None, hi=None):
    # each sample as within the limit, beyond it, or not assessable
    # limit: charge balance error, in %, either side of zero
    # lo, hi: the 5th and 95th percentiles of CBE from the uncertainty analysis. If given, a sample is
    #   beyond the limit only if its whole interval is, so measurement uncertainty cannot bring it within.
    #   Samples with no interval are judged on the calculated value
    # not assessable: no CBE, or the alkalinity was filled (from the charge balance itself, which
    #   forces CBE to zero, or with a mean or fixed value that was not measured)
    cbe = v['CBE (%)'].astype(float)
    beyond = cbe.abs() > limit
    if lo is not None and hi is not None:
        lo = lo.reindex(v.index).astype(float)
        hi = hi.reindex(v.index).astype(float)
        have = lo.notna() & hi.notna()
        beyond = beyond.where(~have, (lo > limit) | (hi < -limit))
    out = pd.Series(np.where(beyond, CB_BEYOND, CB_WITHIN), index=v.index)
    out[cbe.isna() | v['filled_HCO3'].astype(bool)] = CB_NA
    return out


INTERNAL_COLS = {CB_RING, CB_LABEL, 'Sample', 'Label', 'Color', 'Marker', 'Water type', 'Chadha field', 'Filled', 'Alkalinity from',
                 'HCO3, CO3 from', 'HCO3 is input', 'CO3 is input',
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
