# -*- coding: utf-8 -*-
# run_workbench.py
# the Hydrochem Workbench from a script: read a filled template, run PHREEQC, check the
# charge balance, write the results workbook and save the standard figures.
# Gives the same numbers as the web app with the same settings.
#
# install the package once, from the repository folder: pip install .
# Without installing, this file still runs from the examples folder of the repository

# %% libs
import os
import sys

try:
    import hydrochem_workbench as hw
except ImportError:
    sys.path.insert(0, os.path.abspath('..'))    # not installed: use the repository copy
    import hydrochem_workbench as hw
from hydrochem_workbench import plots

# %% CONFIG
DATA_FILE = 'hydrochem_workbench_synthetic_20.xlsx'   # a filled template (.xlsx or .csv)
OUT_FOLDER = 'workbench_output'

UNIT = 'mg/L'                 # unit of every ion column: 'mg/L', 'mmol/L' or 'meq/L'
N_AS_N = False                # True if NO3 and NH4 are entered as N
P_AS_P = False                # True if PO4 is entered as P
EC_UNIT = 'uS/cm'             # unit of the EC column: 'uS/cm', 'mS/cm' (or 'dS/m') or 'mS/m'

LOR_FRACTION = 0.5            # values below the LOR are replaced with this fraction of the LOR

MISSING_PH = 'mean'           # 'mean' of the dataset, or a number
MISSING_TEMP = 'mean'         # 'mean', or a number (deg C)
MISSING_ALKALINITY = 'mean'   # 'mean', 'balance' (from the charge balance), or a number (mg/L as CaCO3)

UNCERTAINTY_N = 0             # realisations of the uncertainty analysis. 0 = off, the app uses 200

CB_LIMIT = 5.0                # limit on the charge balance error, %
CB_ACTION = hw.MARK           # hw.MARK (ring on the plots), hw.EXCLUDE or hw.NO_ACTION
CB_BASIS = hw.VALUE           # hw.VALUE, or hw.INTERVAL (needs the uncertainty analysis)

DPI = 300
FIG_FORMAT = 'png'            # 'png', 'pdf', 'svg', 'tiff' ...

# %% run
result = hw.run(DATA_FILE, unit=UNIT, n_as_n=N_AS_N, p_as_p=P_AS_P, ec_unit=EC_UNIT, lor_fraction=LOR_FRACTION,
                missing_ph=MISSING_PH, missing_temp=MISSING_TEMP, missing_alkalinity=MISSING_ALKALINITY,
                uncertainty_n=UNCERTAINTY_N, cb_limit=CB_LIMIT, cb_action=CB_ACTION, cb_basis=CB_BASIS)
print(result)

v = result.table          # every sample: one row each, all derived variables
for p in result.problems:
    print('Note:', p)
if result.ec_warning:
    print('Note:', result.ec_warning)
print(v[['Sample', 'Label', 'Water type', 'CBE (%)', 'Charge balance check']].to_string(index=False))

# %% outputs
os.makedirs(OUT_FOLDER, exist_ok=True)
hw.save_results(result, os.path.join(OUT_FOLDER, 'results.xlsx'))
files, notes = hw.save_figures(result, os.path.join(OUT_FOLDER, 'figures'), dpi=DPI, fmt=FIG_FORMAT)
print(f'{len(files)} figures saved to {OUT_FOLDER}')
for note in notes:
    print('Note:', note)

# %% a plot of your own
# any numeric column of the results table can go on either axis
fig = plots.custom(result.plotted, 'Cl (mmol/L)', 'Na (mmol/L)', xlog=True, ylog=True, one_to_one=True,
                   unc=result.mc)
fig.savefig(os.path.join(OUT_FOLDER, 'na_vs_cl.png'), dpi=DPI, bbox_inches='tight')

# %% the map (needs the template's Lat and Lon columns, and "pip install .[map]")
# from hydrochem_workbench import maps
# pts, notes = maps.locations(result.plotted)
# extent = maps.fit_extent(maps.data_extent(pts))
# basemap = maps.fetch_basemap(extent, maps.zoom_choices(extent)[0])     # or None for a white background
# fig = maps.sample_map(pts, result.styles, extent, basemap=basemap, by_depth=True,
#                       inset=dict(corner='Top left', factor=12), credit=maps.attribution(maps.IMAGERY))
# fig.savefig(os.path.join(OUT_FOLDER, 'map.png'), dpi=DPI, bbox_inches='tight')
