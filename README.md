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
   nitrite and ammonium can be as the ion or as N, phosphate as PO4 or as P.
   Optional minor and trace constituents: Li, Al, Zn, Cu, Cd and Pb (in
   phreeqc.dat, so they go into the speciation) and Ni, I, As, Cr, Se, U and Hg
   (carried through to the tables, plots and statistics). Li, Zn, Cu, Cd, Pb, Ni,
   NO2 and I are counted in the charge balance; Al, As, Cr, Se, U and Hg are
   not, as their charge depends on pH or redox state. Silica can be given as
   SiO2 or, in a column called `Si`, as silicon. Any extra numeric column
   (depth, isotopes) is carried through and can be plotted.
   `Lat`, `Lon` and `Screen_depth` are optional and used for the map.
   EC can be in uS/cm, mS/cm (dS/m) or mS/m; it is converted to uS/cm, and the
   app warns if the EC and the ions point to a different unit.
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
   user-chosen colours. All the numbers are in `hydrochem_workbench/guidelines.py` with their
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
   - Custom plot: any variable against any other, with reference lines (1:1, 2:1, 1:2, seawater, a
     custom ratio), log or inverted axes, your own axis labels and axis limits
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
   a/sqrt(3)); the rest are placeholders. `hydrochem_workbench/uncertainty.py` has the method.
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
10. **Analysis checks.** Beside the charge balance: calculated against measured
    EC, measured against calculated TDS, TDS to EC, and the ion sums against EC,
    with the acceptance ranges of APHA Standard Methods 1030E. Outliers are
    listed one variable at a time (modified z-score) and, on the PCA view, for
    all variables together (robust Mahalanobis distance), where they can be
    marked or left out. For variables with values below the LOR, a table gives
    the mean, SD and quartiles by substitution, by regression on order
    statistics (ROS) and by Kaplan-Meier.
11. **Irrigation and scaling.** The Irrigation view has SAR, percent sodium, RSC,
    Kelley ratio, magnesium hazard and permeability index with their classes,
    and the USSL (Richards 1954) and Wilcox (1948) diagrams. Each sample gets
    the field it falls in on the Wilcox diagram and, separately, the grade from
    the table in the same bulletin, which is stricter for fresh, sodium-rich
    water. The Scaling view
    has the Langelier and Ryznar indices, with pHs taken from the PHREEQC
    calcite saturation index. The Guidelines view can also give the CCME Water
    Quality Index for a chosen guideline set, with a note on its limits.
12. **PHREEQC database.** phreeqc.dat by default; wateq4f.dat (adds Ni and I to
    the speciation; no calculated EC) or pitzer.dat (saline water and brines;
    no nitrogen, phosphorus, fluoride, aluminium or trace metals) can be chosen.
    Constituents a database lacks are left out of the speciation and stay in
    the tables and plots.
13. **Charge balance.** Each sample is checked against a limit on the charge
    balance error (5 % by default), either on the calculated value or, with the
    uncertainty analysis on, on its 5th to 95th percentile interval (beyond the
    limit only if the whole interval is). Samples beyond the limit are marked
    with a thick ring in another colour on the Piper diagram, the gallery,
    custom, mixing and PCA plots and the map (the default), or excluded from
    every plot, the group means and the further analyses, or left alone. All
    samples stay in the tables with a "Charge balance check" column. A sample
    whose alkalinity was filled is "Not assessable".
14. **Map.** If the template has `Lat` and `Lon` (decimal degrees, optional),
    the Map view draws the samples with their group colour and marker over
    satellite imagery (Esri World Imagery) or a white background, with a scale
    bar, north arrow and a locator inset. Marker size can follow `Screen_depth`
    (m below surface, optional), with the sizes in the legend (automatic depths, or
    up to four of your own). The marker edge can be any grey from black to white,
    at a chosen thickness. Optional extras:
    a csv of towns (Name, Lat, Lon; template in the app), up to three polygon or line
    layers, such as a boundary, rivers and roads (shapefile as a .zip or its
    separate files, GeoPackage, GeoJSON or KML), each with a chosen edge colour,
    fill colour and opacity, and your own outline for the
    inset. Polygon files are converted to latitude and longitude from whatever
    coordinate reference system they state; a shapefile with no .prj asks for
    its EPSG code. The imagery zoom level starts at a value that suits the
    extent and is capped at 100 tiles. The imagery credit is drawn in a strip
    below the map, so it can be cropped off and given in the caption instead.
    If the imagery cannot be fetched the map is drawn on white.

## Privacy

The app keeps an uploaded dataset in memory for the browser session only and
never writes it to disk. With satellite imagery chosen on the Map view, the app
asks Esri's tile server for the tiles that cover the map, so that server sees
the rectangle requested (not the sample locations or any data). The image is
held in memory for the session, not written to disk. With the white background
nothing is requested. A polygon upload is unpacked to a temporary folder to be read and
deleted straight away. The one thing the app records is a running total of
sessions that processed data (`hydrochem_workbench/usage.py`). For that total to survive a
restart on Streamlit Cloud, add Upstash Redis secrets in either layout:

    [upstash]
    url = "https://....upstash.io"
    token = "..."

or `[redis]` with `rest_url` and `rest_token` (as in HydroSTITCH). Without them
the count lives in a local file that is wiped whenever the app restarts, and the
app says so under the count. The key defaults to `hydrochem:runs`, so it can
share a database with the other apps' counters. `.streamlit/config.toml` turns off Streamlit's browser
usage statistics.

## Use from Python, without the app

Everything the app calculates and draws is in the `hydrochem_workbench` package,
which has no Streamlit in it. Install it from the repository folder:

    pip install .            # the calculations and figures
    pip install .[map]       # with the sample map (geopandas, contextily)

or straight from GitHub:

    pip install git+https://github.com/DylanIrvine/hydrochem-workbench.git

Then, from a script, Spyder or a notebook:

    import hydrochem_workbench as hw

    result = hw.run('my_data.xlsx', unit='mg/L')      # a filled template
    v = result.table                                  # every sample, every derived variable
    hw.save_results(result, 'results.xlsx')           # the results workbook
    hw.save_figures(result, 'figures', dpi=300)       # Piper, Schoeller, Stiff, Durov, Gibbs, Chadha, HFE-D, gallery

`hw.run()` takes the same choices as the app's sidebar: units, the below-LOR
fraction, how to fill a missing pH, temperature or alkalinity, the uncertainty
analysis (`uncertainty_n=200`) and the charge balance limit and action. The
app calls the same functions (`hydrochem_workbench/workflow.py`), so a script
and the app give the same numbers for the same settings; a test checks this.
Any single figure can be drawn with the functions in `hydrochem_workbench.plots`
(for example `plots.piper(result.plotted)`), which return a matplotlib figure.
`examples/run_workbench.py` is a complete script with the settings in a block
at the top.

The functions named above (`run`, `save_results`, `save_figures`, and `process`,
`fill_rules`, `style_groups`, `mark_charge_balance`) are the ones intended for
scripts. Other functions in the package may change between versions.

## Layout

    app.py            Streamlit front end
    pyproject.toml    for installing the package (not used by the app)
    hydrochem_workbench/workflow.py  the steps of a run, for scripts and the app: run(), process(), outputs
    hydrochem_workbench/chem.py      template, cleaning, units, fill rules, PHREEQC, derived table
    hydrochem_workbench/guidelines.py  guideline values and the comparison
    hydrochem_workbench/gibbs_fields.py  outline of the Gibbs envelope (from WQChartPy)
    hydrochem_workbench/plots.py     all figures
    hydrochem_workbench/maps.py      the sample map: coordinates, towns, polygon files, imagery, figure
    hydrochem_workbench/data/        country and state outlines for the map inset (Natural Earth), badge icon
    hydrochem_workbench/advanced.py      mixing, PCA, clusters and group tests
    hydrochem_workbench/plots_advanced.py  their figures
    hydrochem_workbench/uncertainty.py  Monte Carlo uncertainty analysis and its default inputs
    hydrochem_workbench/quality.py   irrigation classes, scaling, analysis checks, outliers, below-LOR statistics, CCME index
    hydrochem_workbench/plots_quality.py  their figures (USSL, Wilcox, checks, CCME)
    hydrochem_workbench/usage.py     the usage counter
    examples/         a synthetic dataset and run_workbench.py, the package used from a script
    packages.txt      system font for Streamlit Cloud (Liberation Sans)
    logo/             header logo, browser-tab icon and svg source
    tests/            pytest: chemistry checks, the package from a script, and a headless run of the app

## Known limits (version 0.1)

- Freshwater only. No Pitzer option.
- No redox: there is no pe/Eh column, so no iron, manganese or sulfide phases.
- The four WQChartPy diagrams keep WQChartPy's styling, and its Durov has a
  fixed TDS axis (0 to 4000 mg/L).
- Total N and total P are not in the template, so those guideline values are
  not checked. State and site-specific guideline values are not included.
- The bundled example data are synthetic, including the sample locations.
- The guideline comparison does not yet cover the trace constituents added in
  version 0.17.0 (Al, Zn, Cu, Cd, Pb, Ni, As, Cr, Se, U, Hg).
- Trace metals are entered in the same unit as the other ions. Laboratory
  reports often give them in ug/L, which has to be converted first.
- The map is drawn in longitude and latitude, not in a projected coordinate
  system. The scale bar is exact at the latitude of the middle of the map, which
  is fine for a study area but not for a map spanning many degrees of latitude.
  A map that crosses the 180 degree meridian is not handled.
- The satellite imagery comes from an outside server (Esri World Imagery) and is
  subject to Esri's terms of use, which require attribution. Check those terms
  before using a map with imagery commercially. The capture date of the imagery
  is not known to the app.
- Town names are not moved to avoid each other or the samples.
- The curves of the Wilcox diagram pass through points measured from a scan of
  Figure 1 of Wilcox (1948), good to about 0.5 percent sodium. A sample close to
  a curve could fall either side. The axis stops at 3,500 uS/cm as in the
  original; more saline samples are counted in a note on the figure and are all
  classed unsuitable.
- Percent sodium on the Wilcox diagram is Na / (Ca + Mg + Na + K), as Wilcox
  defines it. The `Na (%)` column elsewhere has Na + K on top, the form most
  later papers use, so the two differ slightly where potassium is high.
- The irrigation classes assume EC in uS/cm at 25 degrees C. The EC unit is set
  in the sidebar, and the app warns when the EC and the ions disagree by a
  factor that looks like a unit, but it cannot tell whether a field reading was
  temperature compensated.
- The CCME index treats each group as a site and its samples as visits, which
  is how it is commonly applied to surveys but not what it was designed for.
  With more than one group it is given per group only: F1 (scope) counts a
  variable as failed if it fails in any sample, so a pooled all-samples index
  is lower than the group values and not comparable with them.
- As, Cr, Se, U and Hg are not passed to PHREEQC under any database, as their
  redox state is not known from a total concentration.

## Licence

GPL-3.0-or-later (see `LICENSE`). The Piper diagram code is adapted from
WQChartPy, which is GPL-3.0, so this project is released under the same terms.
You are free to use, modify and share it; if you distribute a modified version
you must release its source under the GPL as well. There is no warranty: the
outputs are an aid to interpretation and should be checked before they are
relied on.

## How to cite

Irvine, D.J. (2026). Hydrochem Workbench (version 0.19.0)
[Web application]. Charles Darwin University. https://hydrochem-workbench.streamlit.app/

Please also cite PHREEQC and WQChartPy (below).

## Credits and third-party software

- **WQChartPy** (GPL-3.0), Yang et al. (2022), Groundwater,
  doi:10.1111/gwat.13185. Source of the Piper geometry (modified) and of the
  Durov, Gibbs, Chadha and HFE-D diagrams (called unmodified).
- **phreeqpython** (Apache-2.0), Vitens. Python wrapper for PHREEQC.
- **PHREEQC** and `phreeqc.dat`, Parkhurst and Appelo, U.S. Geological Survey
  (public domain).
- **Esri World Imagery**, the satellite imagery behind the map. Fetched from
  Esri's server when a map is drawn (not part of this repository) and credited
  on the figure.
- **wateq4f.dat**, the PHREEQC database from the USGS PHREEQC distribution
  (public domain), in `hydrochem_workbench/data/`. phreeqc.dat and pitzer.dat
  come with phreeqpython.
- **Natural Earth** (public domain), 1:50m country and state outlines for the
  map inset, in `hydrochem_workbench/data/natural_earth_50m.gpkg` with coordinates rounded to
  0.01 degrees.
- GeoPandas, contextily and pyproj (BSD-3-Clause/MIT) for the map.
- Streamlit (Apache-2.0), pandas and NumPy (BSD-3-Clause), Matplotlib
  (Matplotlib licence, BSD-style), openpyxl (MIT).
