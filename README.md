# HydroCHEM (working name)

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
   meq/L, chosen in the app). Two "reported as" switches: HCO3 or alkalinity as
   CaCO3, and nitrate as NO3 or as N. Any extra numeric column (depth, isotopes)
   is carried through and can be plotted.
2. **Missing values.** Missing pH, temperature and HCO3 are filled (dataset
   mean, a fixed value, or for HCO3 the charge-balance value). Filled samples
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
   - Diagnostic gallery (about 80 panels in 10 sets): ions against Cl, ion/Cl
     ratios, ratios against TDS, water-rock process plots, saturation indices
     against Cl, TDS and pH, the carbonate system, and quality checks
   - Custom plot: any variable against any other
5. **Export.** Results table with every derived variable and the settings used;
   a zip of all figures.

## Layout

    app.py            Streamlit front end
    core/chem.py      template, cleaning, units, fill rules, PHREEQC, derived table
    core/plots.py     all figures
    tests/            pytest: chemistry checks and a headless run of the app

`core/` has no Streamlit in it and can be used from Spyder directly.

## Known limits (version 0.1)

- Freshwater only. No Pitzer option.
- No redox: there is no pe/Eh column, so no iron, manganese or sulfide phases.
- The four WQChartPy diagrams keep WQChartPy's styling, and its Durov has a
  fixed TDS axis (0 to 4000 mg/L).
- The bundled example data are synthetic.

## Credits

Piper geometry and the pass-through diagrams: WQChartPy, Yang et al. (2022),
Groundwater, doi:10.1111/gwat.13185. Speciation: PHREEQC (Parkhurst and Appelo,
USGS) through phreeqpython (Vitens).
