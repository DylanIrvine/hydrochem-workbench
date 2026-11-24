# -*- coding: utf-8 -*-
# compare_with_hydrochem.py
# the same dataset through the Hydrochem Workbench and through the R package hydrochem
# (Cinkus and Bondu, 2026), compared variable by variable.
#
# run run_hydrochem.R first: it writes data_qc_raw.csv and hydrochem_results.csv here.
# The dataset is the Quebec groundwater data that ships with hydrochem (Cloutier, PhD thesis,
# INRS; 146 samples).

# %% libs
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import hydrochem_workbench as hw
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))    # not installed: use the repository copy
    import hydrochem_workbench as hw

# %% CONFIG
RAW_FILE = 'data_qc_raw.csv'
R_FILE = 'hydrochem_results.csv'
OUT_TABLE = 'comparison_summary.csv'
OUT_SAMPLES = 'comparison_by_sample.csv'
OUT_FIG = 'comparison.png'
DPI = 300

# Workbench column: (hydrochem column, what to expect)
PAIRS = {
    'CBE (%)': ('ib_major_minor', 'Same formula. hydrochem also counts PO4 and Al; the Workbench also counts Cd and Pb'),
    'SAR': ('sar', 'Same formula'),
    'Hardness (mg/L as CaCO3)': ('hardness', 'Same formula; 50.04 against 50 mg CaCO3 per meq'),
    'Alkalinity (mg/L as CaCO3)': ('alkalinity', 'hydrochem adds OH- and subtracts H+; 50.04 against 50'),
    'CAI-1': ('cai1', 'Same formula'),
    'Na (%)': ('percent_sodium', 'Different definition: the Workbench has Na + K in the numerator'),
    'TDS calc (mg/L)': ('tds_all', 'Different definition: the Workbench counts HCO3 x 0.4917 and fewer ions'),
}

# %% the dataset, in the Workbench template's columns
raw = pd.read_csv(RAW_FILE, dtype=object)
data = pd.DataFrame({'Sample': raw['Sample_ID'], 'Label': raw['Cluster'], 'pH': raw['pH'],
                     'Temp': raw['T'], 'EC': raw['EC']})
for ion in ['Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'HCO3', 'CO3', 'Br', 'F', 'SiO2', 'Sr', 'Ba', 'Fe', 'Mn', 'B',
            'Li', 'Al', 'Zn', 'Cu', 'Cd', 'Pb', 'Ni', 'I', 'As', 'Cr', 'Se', 'U', 'Hg']:
    data[ion] = raw[ion]
# these three are reported as N and as P
data['NO3'] = raw['NO3_N']
data['NH4'] = raw['NH4_N']
data['PO4'] = raw['PO4_P']

# %% the Workbench
result = hw.run(data, unit='mg/L', n_as_n=True, p_as_p=True, lor_fraction=0.5)
v = result.table
print(result)
print(f'{len(result.replaced)} values below the detection limit replaced')

# %% compare
theirs = pd.read_csv(R_FILE)
both = v.merge(theirs, on='Sample', how='inner', validate='one_to_one')
print(f'{len(both)} samples in both')

rows = []
for ours, (r_col, note) in PAIRS.items():
    a, b = both[ours].astype(float), both[r_col].astype(float)
    ok = a.notna() & b.notna()
    diff = (a - b)[ok]
    rows.append({'Workbench': ours, 'hydrochem': r_col, 'n': int(ok.sum()),
                 'Mean difference': diff.mean(), 'Max absolute difference': diff.abs().max(),
                 'Max relative difference (%)': 100 * (diff.abs() / b[ok].abs().where(b[ok] != 0)).max(),
                 'Pearson r': np.corrcoef(a[ok], b[ok])[0, 1], 'Note': note})
summary = pd.DataFrame(rows)
summary.to_csv(OUT_TABLE, index=False)
print(summary.drop(columns='Note').round(4).to_string(index=False))

# how many samples each tool would flag at 5 %
print('Charge balance beyond 5 %: Workbench', int((both['CBE (%)'].abs() > 5).sum()),
      ', hydrochem (major and minor ions)', int((both['ib_major_minor'].abs() > 5).sum()),
      ', hydrochem (major ions)', int((both['ib_major'].abs() > 5).sum()))

cols = ['Sample'] + [c for pair in PAIRS.items() for c in (pair[0], pair[1][0])] + ['ib_major', 'Water type',
                                                                                 'water_type']
both[cols].to_csv(OUT_SAMPLES, index=False)

# %% figure: one panel for each variable, Workbench against hydrochem
fig, axs = plt.subplots(2, 4, figsize=(17 / 2.54, 9 / 2.54))
for ax, (ours, (r_col, _)) in zip(axs.flat, PAIRS.items()):
    a, b = both[ours].astype(float), both[r_col].astype(float)
    ax.plot(b, a, ls='', marker='o', ms=2.5, mfc='#0089D0', mec='none', alpha=0.7)
    lim = [np.nanmin([a.min(), b.min()]), np.nanmax([a.max(), b.max()])]
    ax.plot(lim, lim, color='0.3', lw=0.6)
    ax.set_title(ours, fontsize=7)
    ax.tick_params(labelsize=6)
for ax in axs.flat[len(PAIRS):]:
    ax.axis('off')
fig.supxlabel('hydrochem (R)', fontsize=8)
fig.supylabel('Hydrochem Workbench', fontsize=8)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=DPI)
print('written:', OUT_TABLE, OUT_SAMPLES, OUT_FIG)
