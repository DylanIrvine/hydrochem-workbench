# Hydrochem Workbench

https://hydrochem-workbench.streamlit.app/

Upload a hydrochemistry spreadsheet, run every sample through PHREEQC, and get
the plots a hydrogeochemist makes first. Built to sit alongside HydroSILO and
HydroSTITCH: the app does the tedious part so the user (or student) can spend
their time on interpretation.

## Run

    pip install -r requirements.txt
    streamlit run app.py

Tests: `pip install -r requirements-dev.txt` then `pytest`.

## What it does

1. **Template in.** One row per sample, one unit for every ion (mg/L, mmol/L or
   meq/L, chosen in the app). Alkalinity goes in as the lab reports it, in mg/L
   as CaCO3: total alkalinity, or bicarbonate and carbonate alkalinity. HCO3 and
   CO3 are optional: where entered (in the ion unit) they are used as given,
   and where both are blank PHREEQC calculates them from alkalinity and pH. If
   only one is entered, the other is the rest of the alkalinity (HCO3 + 2 CO3 =
   alkalinity in meq/L). The results table says where each came from. With no
   alkalinity columns, HCO3 + CO3 are converted to alkalinity for PHREEQC. Where there is no alkalinity, DIC (mg/L as C
   or mmol/L) can be given instead and PHREEQC works from DIC and pH. Nitrate
   and ammonium can be as the ion or as N, phosphate as PO4 or as P. Any extra
   numeric column (depth, isotopes) is carried through and can be plotted.
2. **Missing values.** Missing pH, temperature and alkalinity are filled (dataset
   mean, a fixed value, or for alkalinity the charge-balance value). Filled samples
   are drawn hollow, but only on plots the fill actually affects. Values below
   the LOR (`<0.05`, or negative values) can be replaced by a chosen fraction
   of the LOR (half by default); other text cells are dropped and listed.
3. **PHREEQC.** `phreeqpython` with `phreeqc.dat`. Saturation indices for
   calcite, aragonite, dolomite, gypsum, anhydrite, halite, fluorite, quartz,
   chalcedony, barite, celestite and strontianite (each only where its elements
   were measured), log pCO2, ionic strength and calculated conductivity. A
   warning appears above 0.5 mol/kgw ionic strength.
4. **Guidelines.** Each sample is compared with Australian guideline values:
   drinking water (ADWG health and aesthetic), ANZG toxicant default guideline
   values for fresh and marine ecosystems (95% protection), and the ANZECC 2000
   nutrient, pH and EC values by region (tropical, south-east, south-west and
   south central Australia) and ecosystem type, including estuaries. Two
   international sets are also available: the WHO Guidelines for drinking-water
   quality (2026; health and acceptability values) and the FAO irrigation
   guidelines (Ayers and Westcot 1985), rated as no, slight to moderate, or
   severe restriction on use, including infiltration from SAR and EC together.
   A bar chart plots any parameter against any of these guideline values, with
   user-chosen colours. All the numbers are in `core/guidelines.py` with their
   sources. It is a screening comparison, not a compliance assessment.
5. **Plots.**
   - Piper, in the coloured-field style from `PiperScript.py`
   - Schoeller and Stiff, native, same style
   - Durov, Gibbs, Chadha and HFE-D drawn by WQChartPy, with plain-text labels
     and the uncertainty added
   - Diagnostic gallery (about 120 panels in 15 sets), with reference lines,
     shaded near-equilibrium bands and interpretation notes: ions and ion/Cl
     ratios against Cl, source-of-solute ratios (Na/Cl against TDS, HCO3
     fraction against Na/Cl, Hounslow-style ratios), Gaillardet end-member
     diagrams, Gibbs, water-rock process plots, saturation indices against
     TDS, Cl, pH, their own ions and each other, the carbonate system,
     irrigation quality (SAR, percent Na) and quality checks
   - Custom plot: any variable against any other
   - Optional group means on the Piper, gallery and custom plots, drawn as a
     larger version of each group's marker (the mean of what is on each axis)
6. **Uncertainty (optional).** Measurement uncertainty is carried through the
   whole calculation by Monte Carlo (JCGM 101:2008), 200 realisations by default.
   Each measured value is drawn from a normal distribution with standard
   deviation sqrt((LOR/sqrt(12))^2 + (relative SD x value)^2 + absolute SD^2),
   and values below the LOR on the lab report from Uniform(0, LOR). Every
   realisation goes through the same unit conversion, fills, PHREEQC and derived
   variables as the measured data. Results are 90 % intervals (5th to 95th
   percentile), drawn as light grey lines on the gallery, custom, Schoeller and
   guideline bar plots, and as grey clouds or outlines (or off) on the Piper and
   the WQChartPy diagrams (Durov, Gibbs, Chadha, HFE-D); the points stay at the
   measured values. For the WQChartPy diagrams, WQChartPy is run once on the
   realisations with matplotlib's scatter swapped for a recorder, so the
   realisations land exactly where WQChartPy would put them. A per-sample summary gives the share of
   realisations with the same water type and Chadha field, a charge balance
   within 5 % and each saturation index above zero. Every input (LOR, relative
   and absolute SD) is editable in the app. Defaults: major ions, alkalinity,
   NO3, Br and F from laboratory control sample recoveries in 19 quality-control
   reports from an Australian laboratory (surface water and groundwater, 2021 to
   2026); pH, temperature and EC from the YSI ProDSS specifications (rectangular,
   a/sqrt(3)); the rest are placeholders. `core/uncertainty.py` has the method.
7. **Export.** Results table with every derived variable, a Settings sheet with
   every option used, the group colours and markers, and any replaced below-LOR
   values, and the mean of each analyte for each group (with the number of
   values behind each mean); with the uncertainty analysis on, the inputs used, the 5th and 95th
   percentiles beside every derived variable, and the per-sample summary. A zip
   of all figures, in classical, gallery and custom plots folders. File names start with the dataset name.
8. **Further analyses**, each in its own view:
   - *Mixing.* Two or three end members (samples or group means) and one or
     more conservative tracers (Cl, Br, isotopes or any extra column). Mixing
     fractions are solved with the fractions summing to 1 (least squares, with
     each tracer scaled by the end-member spread, when there are more tracers
     than needed). Every other ion is compared with conservative mixing
     (measured minus mixing, mmol/L), with a cation-exchange check of Na + K
     against Ca + Mg (meq/L). Samples outside the end members are flagged. With
     the uncertainty analysis on, the fractions and deviations get 90 %
     intervals from the same realisations, so a deviation can be judged against
     the measurement uncertainty.
   - *PCA.* Centred log-ratio (default) or log10 and standardised; biplot with
     loadings and the variance on each component.
   - *Clusters.* Ward linkage on Euclidean distance (Guler et al. 2002), with a
     dendrogram, a Piper coloured by cluster and a table of clusters against
     groups.
   - *Group comparison.* Box plots by group, Kruskal-Wallis tests and pairwise
     Mann-Whitney U tests (Holm-adjusted), and a Spearman correlation heat map.

   Warnings appear for few samples, few samples per variable, and variables
   that are mostly below the LOR. Above 200 samples the app warns that figures
   may be slow or crowded; everything still runs.
9. **Several datasets, same settings.** "Load another dataset" replaces the data
   and keeps every setting (units, below-LOR, missing values, plot options,
   group colours and markers, guidelines), so several sites, or an edited sheet,
   can be run the same way.

## Privacy

The app keeps an uploaded dataset in memory for the browser session only and
never writes it to disk. The one thing it records is a running total of
sessions that processed data (`core/usage.py`). For that total to survive a
restart on Streamlit Cloud, add Upstash Redis secrets in either layout:

    [upstash]
    url = "https://....upstash.io"
    token = "..."

or `[redis]` with `rest_url` and `rest_token` (as in HydroSTITCH). Without them
the count lives in a local file that is wiped whenever the app restarts, and the
app says so under the count. The key defaults to `hydrochem:runs`, so it can
share a database with the other apps' counters. `.streamlit/config.toml` turns off Streamlit's browser
usage statistics.

## Layout

    app.py            Streamlit front end
    core/chem.py      template, cleaning, units, fill rules, PHREEQC, derived table
    core/guidelines.py  guideline values and the comparison
    core/gibbs_fields.py  outline of the Gibbs envelope (from WQChartPy)
    core/plots.py     all figures
    core/advanced.py      mixing, PCA, clusters and group tests (no Streamlit)
    core/plots_advanced.py  their figures
    core/uncertainty.py  Monte Carlo uncertainty analysis and its default inputs
    core/usage.py     the usage counter
    packages.txt      system font for Streamlit Cloud (Liberation Sans)
    logo/             header logo, browser-tab icon and svg source
    tests/            pytest: chemistry checks and a headless run of the app

`core/` has no Streamlit in it and can be used from Spyder directly.

## Known limits (version 0.1)

- Freshwater only. No Pitzer option.
- No redox: there is no pe/Eh column, so no iron, manganese or sulfide phases.
- The four WQChartPy diagrams keep WQChartPy's styling, and its Durov has a
  fixed TDS axis (0 to 4000 mg/L).
- Total N and total P are not in the template, so those guideline values are
  not checked. State and site-specific guideline values are not included.
- The bundled example data are synthetic.

## Licence

GPL-3.0-or-later (see `LICENSE`). The Piper diagram code is adapted from
WQChartPy, which is GPL-3.0, so this project is released under the same terms.
You are free to use, modify and share it; if you distribute a modified version
you must release its source under the GPL as well. There is no warranty: the
outputs are an aid to interpretation and should be checked before they are
relied on.

## How to cite

Irvine, D.J. (2026). Hydrochem Workbench (version 0.13.0)
[Web application]. Charles Darwin University. https://hydrochem-workbench.streamlit.app/

Please also cite PHREEQC and WQChartPy (below).

## Credits and third-party software

- **WQChartPy** (GPL-3.0), Yang et al. (2022), Groundwater,
  doi:10.1111/gwat.13185. Source of the Piper geometry (modified) and of the
  Durov, Gibbs, Chadha and HFE-D diagrams (called unmodified).
- **phreeqpython** (Apache-2.0), Vitens. Python wrapper for PHREEQC.
- **PHREEQC** and `phreeqc.dat`, Parkhurst and Appelo, U.S. Geological Survey
  (public domain).
- Streamlit (Apache-2.0), pandas and NumPy (BSD-3-Clause), Matplotlib
  (Matplotlib licence, BSD-style), openpyxl (MIT).
