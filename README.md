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
   CO3 are not inputs. PHREEQC calculates both from alkalinity and pH, and those
   values are used in every plot. Nitrate can be as NO3 or as N. Any extra
   numeric column (depth, isotopes) is carried through and can be plotted.
2. **Missing values.** Missing pH, temperature and alkalinity are filled (dataset
   mean, a fixed value, or for alkalinity the charge-balance value). Filled samples
   are drawn hollow, but only on plots the fill actually affects. Text cells
   such as `<0.05` are dropped and listed; dealing with them is the user's job.
3. **PHREEQC.** `phreeqpython` with `phreeqc.dat`. Saturation indices for
   calcite, aragonite, dolomite, gypsum, anhydrite, halite, fluorite, quartz,
   chalcedony, barite, celestite and strontianite (each only where its elements
   were measured), log pCO2, ionic strength and calculated conductivity. A
   warning appears above 0.5 mol/kgw ionic strength.
4. **Plots.**
   - Piper, in the coloured-field style from `PiperScript.py`
   - Schoeller and Stiff, native, same style
   - Durov, Gibbs, Chadha and HFE-D passed through from WQChartPy as is
   - Diagnostic gallery (about 120 panels in 15 sets), with reference lines,
     shaded near-equilibrium bands and interpretation notes: ions and ion/Cl
     ratios against Cl, source-of-solute ratios (Na/Cl against TDS, HCO3
     fraction against Na/Cl, Hounslow-style ratios), Gaillardet end-member
     diagrams, Gibbs, water-rock process plots, saturation indices against
     TDS, Cl, pH, their own ions and each other, the carbonate system,
     irrigation quality (SAR, percent Na) and quality checks
   - Custom plot: any variable against any other
5. **Export.** Results table with every derived variable and the settings used;
   a zip of all figures.

## Privacy

The app keeps an uploaded dataset in memory for the browser session only and
never writes it to disk. The one thing it records is a running total of
sessions that processed data (`core/usage.py`). For that total to survive a
redeploy on Streamlit Cloud, add Upstash Redis secrets:

    [redis]
    rest_url = "https://....upstash.io"
    rest_token = "..."

The key defaults to `hydrochem:runs`, so it can share a database with the
HydroSTITCH counter. `.streamlit/config.toml` turns off Streamlit's browser
usage statistics.

## Layout

    app.py            Streamlit front end
    core/chem.py      template, cleaning, units, fill rules, PHREEQC, derived table
    core/plots.py     all figures
    core/usage.py     the usage counter
    logo/             hex badge (svg, png) and favicon
    tests/            pytest: chemistry checks and a headless run of the app

`core/` has no Streamlit in it and can be used from Spyder directly.

## Known limits (version 0.1)

- Freshwater only. No Pitzer option.
- No redox: there is no pe/Eh column, so no iron, manganese or sulfide phases.
- The four WQChartPy diagrams keep WQChartPy's styling, and its Durov has a
  fixed TDS axis (0 to 4000 mg/L).
- The bundled example data are synthetic.

## Licence

GPL-3.0-or-later (see `LICENSE`). The Piper diagram code is adapted from
WQChartPy, which is GPL-3.0, so this project is released under the same terms.
You are free to use, modify and share it; if you distribute a modified version
you must release its source under the GPL as well. There is no warranty: the
outputs are an aid to interpretation and should be checked before they are
relied on.

## How to cite

Irvine, D. (2026). Hydrochem Workbench (version 0.4) [Web application].
Charles Darwin University. https://hydrochem-workbench.streamlit.app/

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
