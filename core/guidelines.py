# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# core/guidelines.py
# Australian water quality guideline values and the comparison against them.
# All values are in the table below so they are easy to check and update.
# 'var' is a column in the results table (v). Concentrations are mg/L.

# %%
import numpy as np
import pandas as pd

# %% sources
ADWG = 'NHMRC, Australian Drinking Water Guidelines, Table 10.6 (online version, checked Nov 2026)'
ANZECC = 'ANZECC & ARMCANZ (2000), Table 3.4.1'
ANZG_NO3 = 'ANZG (2025) technical brief: nitrate in freshwater'
TROP = 'ANZECC & ARMCANZ (2000), Tables 3.3.4 and 3.3.5 (tropical Australia)'

# %% drinking water and toxicants
# kind: 'max' (flag above value), 'range' (flag outside low-high)
# hard: (min, max) hardness in mg/L as CaCO3 for which the row applies
SETS = {
    'Drinking water: health (ADWG)': [
        dict(var='NO3 (mg/L)', name='Nitrate (as NO3)', kind='max', value=50, source=ADWG,
             note='Protects bottle-fed infants under 3 months; up to 100 mg/L for others.'),
        dict(var='F (mg/L)', name='Fluoride', kind='max', value=1.5, source=ADWG),
        dict(var='Mn (mg/L)', name='Manganese', kind='max', value=0.1, source=ADWG,
             note='Lowered from 0.5 mg/L in the 2025 update.'),
        dict(var='Ba (mg/L)', name='Barium', kind='max', value=2, source=ADWG),
    ],
    'Drinking water: aesthetic (ADWG)': [
        dict(var='TDS (mg/L)', name='Total dissolved solids', kind='max', value=600, source=ADWG),
        dict(var='pH', name='pH', kind='range', low=6.5, high=8.5, source=ADWG),
        dict(var='Hardness (mg/L as CaCO3)', name='Hardness (as CaCO3)', kind='max', value=200, source=ADWG),
        dict(var='Na (mg/L)', name='Sodium', kind='max', value=180, source=ADWG),
        dict(var='Cl (mg/L)', name='Chloride', kind='max', value=250, source=ADWG),
        dict(var='SO4 (mg/L)', name='Sulfate', kind='max', value=250, source=ADWG,
             note='Taste threshold. Above 500 mg/L can have purgative effects.'),
        dict(var='Fe (mg/L)', name='Iron', kind='max', value=0.3, source=ADWG),
        dict(var='Mn (mg/L)', name='Manganese', kind='max', value=0.05, source=ADWG),
        dict(var='SiO2 (mg/L)', name='Silica', kind='max', value=80, source=ADWG),
        dict(var='Ammonia as NH3 (mg/L)', name='Ammonia (as NH3)', kind='max', value=0.5, source=ADWG),
    ],
    'Aquatic ecosystems: freshwater toxicants, 95% protection (ANZG)': [
        dict(var='NO3-N (mg/L)', name='Nitrate (as N), soft water', kind='max', value=1.1, hard=(0, 30),
             source=ANZG_NO3, note='Hardness below 30 mg/L as CaCO3.'),
        dict(var='NO3-N (mg/L)', name='Nitrate (as N), moderately hard water', kind='max', value=2.6,
             hard=(30, 150), source=ANZG_NO3, note='Hardness 30 to 150 mg/L as CaCO3.'),
        dict(var='NO3-N (mg/L)', name='Nitrate (as N), hard water', kind='max', value=29, hard=(150, np.inf),
             source=ANZG_NO3, note='Hardness above 150 mg/L as CaCO3.'),
        dict(var='NH4-N (mg/L)', name='Total ammonia (as N)', kind='max', value=0.9, source=ANZECC,
             note='Value is for pH 8. The guideline is lower above pH 8 and higher below it; not adjusted here.'),
        dict(var='Mn (mg/L)', name='Manganese', kind='max', value=1.9, source=ANZECC),
    ],
    'Aquatic ecosystems: marine toxicants, 95% protection (ANZG)': [
        dict(var='NH4-N (mg/L)', name='Total ammonia (as N)', kind='max', value=0.91, source=ANZECC,
             note='Value is for pH 8.'),
        dict(var='Mn (mg/L)', name='Manganese', kind='max', value=0.08, source=ANZECC,
             note='Low reliability value. A 2025 ANZG draft proposes 0.30 mg/L.'),
    ],
}

# %% physical and chemical stressors, tropical Australia (slightly disturbed ecosystems)
# nutrients in ug/L in the source, stored here in mg/L. Where the source gives a
# range (e.g. wetlands FRP 5-25) the upper end is used
# (FRP, NOx, total ammonia N, pH low, pH high, EC low, EC high)
TROPICAL = {
    'Upland river':        (0.005, 0.030, 0.006, 6.0, 7.5, 20, 250),
    'Lowland river':       (0.004, 0.010, 0.010, 6.0, 8.0, 20, 250),
    'Lakes and reservoirs': (0.005, 0.010, 0.010, 6.0, 8.0, 90, 900),
    'Wetlands':            (0.025, 0.010, 0.010, 6.0, 8.0, 90, 900),
    'Estuaries':           (0.005, 0.030, 0.015, 7.0, 8.5, None, None),
    'Marine, inshore':     (0.005, 0.008, 0.010, 8.0, 8.4, None, None),
    'Marine, offshore':    (0.005, 0.004, 0.006, 8.2, 8.2, None, None),
}
TROPICAL_NAME = 'Aquatic ecosystems: nutrients, pH and EC, tropical Australia (ANZECC 2000)'


def tropical_rows(ecosystem):
    frp, nox, nh4, ph_lo, ph_hi, ec_lo, ec_hi = TROPICAL[ecosystem]
    rows = [
        dict(var='PO4-P (mg/L)', name='Filterable reactive P', kind='max', value=frp, source=TROP),
        dict(var='NO3-N (mg/L)', name='Oxidised N (NOx), from nitrate', kind='max', value=nox, source=TROP,
             note='Nitrate only: nitrite is not in the template. NT value for lowland rivers and lakes is 0.005.'),
        dict(var='NH4-N (mg/L)', name='Total ammonia (as N)', kind='max', value=nh4, source=TROP),
        dict(var='pH', name='pH', kind='range', low=ph_lo, high=ph_hi, source=TROP),
    ]
    if ec_lo is not None:
        rows.append(dict(var='EC', name='EC (uS/cm)', kind='range', low=ec_lo, high=ec_hi, source=TROP))
    return rows


def set_names():
    return list(SETS) + [TROPICAL_NAME]


# %% comparison
def assess(v, chosen, ecosystem='Lowland river'):
    # returns (summary, flags)
    #   summary: one row per guideline value, with how many samples exceed it
    #   flags:   one row per sample, text listing what it exceeds
    hardness = v['Hardness (mg/L as CaCO3)']
    summary = []
    per_sample = {i: [] for i in v.index}
    for set_name in chosen:
        rows = tropical_rows(ecosystem) if set_name == TROPICAL_NAME else SETS[set_name]
        short = set_name.split(':')[0] + (' (' + ecosystem + ')' if set_name == TROPICAL_NAME else '')
        for g in rows:
            if g['var'] not in v.columns:
                continue
            x = v[g['var']]
            applies = x.notna()
            if 'hard' in g:
                applies &= hardness.notna() & (hardness >= g['hard'][0]) & (hardness < g['hard'][1])
            if g['kind'] == 'max':
                over = applies & (x > g['value'])
                limit = f"{g['value']:g}"
            else:
                over = applies & ((x < g['low']) | (x > g['high']))
                limit = f"{g['low']:g} to {g['high']:g}"
            summary.append({'Guideline set': set_name, 'Parameter': g['name'], 'Guideline value': limit,
                            'Samples assessed': int(applies.sum()), 'Outside guideline': int(over.sum()),
                            'Which samples': ', '.join(v.loc[over, 'Sample'].astype(str)),
                            'Note': g.get('note', ''), 'Source': g['source']})
            for i in v.index[over]:
                per_sample[i].append(g['name'])
    flags = pd.DataFrame({'Sample': v['Sample'], 'Label': v['Label'],
                          'Outside guideline for': ['; '.join(dict.fromkeys(per_sample[i])) for i in v.index]})
    flags['Count'] = [len(dict.fromkeys(per_sample[i])) for i in v.index]
    return pd.DataFrame(summary), flags
