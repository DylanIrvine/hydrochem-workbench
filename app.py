# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# app.py
# HydroCHEM: hydrochemistry plots and PHREEQC saturation indices
#
# to run: streamlit run app.py
#
# calculations are in core/chem.py, figures in core/plots.py

# %%
import hashlib
import io
import os
import zipfile

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core import chem, plots
from core.usage import increment as _record_use, status as _usage_status

# %% CONFIG
APP_NAME = 'HydroCHEM'
APP_VERSION = '0.2'
PANELS_PER_PAGE = 6
DEFAULT_PH = 7.0
DEFAULT_TEMP = 25.0
DEFAULT_HCO3_MMOL = 2.0
MAX_STIFF = 24

# theme (dark, red) is set in .streamlit/config.toml
st.set_page_config(page_title=APP_NAME, layout='centered')
st.title(APP_NAME)
st.caption(f'Version {APP_VERSION}')

# %% intro
st.write(
    'HydroCHEM takes a spreadsheet of water chemistry, runs every sample through PHREEQC, and draws '
    'the plots a hydrogeochemist would make first. Download the template from the sidebar, paste '
    'your data in (one row per sample, one unit for every ion), choose the unit and upload it.\n\n'
    'What comes back: charge balance, water type, saturation indices and pCO2, a Piper diagram, the '
    'classical diagrams, a gallery of diagnostic ion, ratio and saturation-index plots with '
    'reference lines, and a builder for any plot the gallery does not have. The database is '
    'phreeqc.dat, so results are for fresh water; saline samples are flagged.')


# %% usage counter
# counts sessions that process a dataset. Uses Upstash Redis if the secrets
# are set (as for HydroSTITCH), otherwise a local file. See core/usage.py
def _counter_secret(name, env):
    try:
        sec = st.secrets
        if 'redis' in sec and name in sec['redis']:
            return str(sec['redis'][name])
        if env in sec:
            return str(sec[env])
    except Exception:
        pass
    return os.environ.get(env)


def _counter_args():
    return dict(redis_url=_counter_secret('rest_url', 'UPSTASH_REDIS_REST_URL'),
                redis_token=_counter_secret('rest_token', 'UPSTASH_REDIS_REST_TOKEN'),
                redis_key=_counter_secret('key', 'HYDROCHEM_RUN_COUNT_KEY') or 'hydrochem:runs')


@st.cache_data(ttl=90, show_spinner=False)
def _counter_status():
    return _usage_status(**_counter_args())


def count_this_session():
    # add 1 to the counter, once per session
    if st.session_state.get('counted'):
        return
    st.session_state['counted'] = True
    try:
        total = _record_use(**_counter_args())
    except Exception:
        total = None
    if total is not None:
        st.session_state['use_count'] = total


_use_count_slot = st.empty()


def show_use_count():
    status = _counter_status()
    total = st.session_state.get('use_count') or status.get('count')
    with _use_count_slot.container():
        if total:
            st.caption(f'{APP_NAME} sessions that have processed data: {total:,}')
        if status['backend'] == 'redis-error':
            st.caption(f":orange[Usage counter: Redis is configured but not reachable ({status['detail']}).]")


show_use_count()

# %% privacy
with st.expander('**Privacy and data handling**'):
    st.markdown(
        '**Your data**\n'
        '- The file you upload is read into memory for your session, processed, and discarded when '
        'you close or refresh the page. The app does not save it to a database or to any permanent '
        'storage.\n'
        '- Nothing is sent to a third party. PHREEQC runs inside the app, not as an outside service.\n'
        '- Sample coordinates, site names or anything else in your spreadsheet are treated the '
        'same way: used for your plots, then gone. The results exist only in the files you download.\n\n'
        '**You**\n'
        '- No login, and no record kept by this app of who you are or where you are. The app does '
        'not log IP addresses or locations.\n'
        '- The only thing recorded is a single running total: the number of sessions that have '
        'processed a dataset (and the time it was last updated). It has no link to any user or '
        'file.\n\n'
        '**What this does not cover**\n'
        '- Processing happens on the server that hosts the app, not on your computer. If you are '
        'using the public web version, your file travels to that host (which may be outside '
        'Australia) for the length of your session. The hosting provider and the Streamlit '
        'framework may use their own cookies and keep their own server logs, under their own '
        'privacy policies.\n'
        '- If your data are confidential or subject to a data agreement, run the app on your own '
        'machine. The code is open source and the README has the two commands needed.')

# %% citations
with st.expander('**How to cite and references**'):
    st.markdown(
        'If you use output from this app, please cite the software it is built on.\n\n'
        '**Geochemical calculations (PHREEQC)**\n\n'
        'Parkhurst, D.L. and Appelo, C.A.J. (2013). Description of input and examples for PHREEQC '
        'version 3: a computer program for speciation, batch-reaction, one-dimensional transport, '
        'and inverse geochemical calculations. U.S. Geological Survey Techniques and Methods, '
        'book 6, chapter A43, 497 p. https://pubs.usgs.gov/tm/06/a43/\n\n'
        'PHREEQC is called through phreeqpython (Vitens): https://github.com/Vitens/phreeqpython\n\n'
        '**Diagrams (WQChartPy)**\n\n'
        'Yang, J., Liu, H., Tang, Z., Peeters, L. and Ye, M. (2022). Visualization of aqueous '
        'geochemical data using Python and WQChartPy. Groundwater, 60(4), 555-564. '
        'https://doi.org/10.1111/gwat.13185\n\n'
        'The Piper diagram here is adapted from WQChartPy; the Durov, Gibbs, Chadha and HFE-D '
        'diagrams are drawn by WQChartPy directly.\n\n'
        '**Original diagram references**\n\n'
        'Piper, A.M. (1944). A graphic procedure in the geochemical interpretation of '
        'water-analyses. Eos, Transactions American Geophysical Union, 25(6), 914-928. '
        'https://doi.org/10.1029/TR025i006p00914\n\n'
        'Stiff, H.A. (1951). The interpretation of chemical water analysis by means of patterns. '
        'Journal of Petroleum Technology, 3(10), 15-17. https://doi.org/10.2118/951376-G\n\n'
        'Durov, S.A. (1948). Natural waters and graphic representation of their composition. '
        'Doklady Akademii Nauk SSSR, 59, 87-90.\n\n'
        'Gibbs, R.J. (1970). Mechanisms controlling world water chemistry. Science, 170(3962), '
        '1088-1090. https://doi.org/10.1126/science.170.3962.1088\n\n'
        'Chadha, D.K. (1999). A proposed new diagram for geochemical classification of natural '
        'waters and interpretation of chemical data. Hydrogeology Journal, 7, 431-439. '
        'https://doi.org/10.1007/s100400050216\n\n'
        'Gimenez-Forcada, E. (2010). Dynamic of sea water interface using hydrochemical facies '
        'evolution diagram. Groundwater, 48(2), 212-216. '
        'https://doi.org/10.1111/j.1745-6584.2009.00649.x')


# %% functions
@st.cache_resource
def get_phreeqc():
    # load the database once
    import phreeqpython
    return phreeqpython.PhreeqPython(database='phreeqc.dat')


def process(raw, unit, alk_as_caco3, no3_as_n, rules):
    # clean, convert, fill, run PHREEQC
    # results are kept in session_state (not st.cache_data) so nothing is held after the user leaves
    key = hashlib.sha1((raw.to_csv() + repr((unit, alk_as_caco3, no3_as_n, rules))).encode()).hexdigest()
    cached = st.session_state.get('result')
    if cached is not None and cached[0] == key:
        return cached[1]

    with st.spinner('Running PHREEQC ...'):
        df, dropped, extra_cols, problems = chem.clean(raw)
        # duplicates are flagged but still run
        if any(not p.startswith('Duplicate') for p in problems):
            out = (None, dropped, None, problems)
        else:
            mmol = chem.to_mmol(df, unit, alk_as_caco3, no3_as_n)
            df, mmol, filled = chem.fill_missing(df, mmol, dict(rules))
            phreeqc, errors = chem.run_phreeqc(df, mmol, get_phreeqc())
            out = (chem.build_table(df, mmol, filled, phreeqc, extra_cols), dropped, errors, problems)
    st.session_state['result'] = (key, out)
    return out


def show(fig, name, key):
    st.pyplot(fig, use_container_width=True)
    st.download_button('Download PNG (600 dpi)', plots.fig_png(fig, dpi=600),
                       file_name=name + '.png', mime='image/png', key=key)


# %% sidebar: data in
with st.sidebar:
    st.header('1. Data')
    st.download_button('Download the template', chem.template_bytes(),
                       file_name='HydroCHEM_template.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    uploaded = st.file_uploader('Upload the filled template', type=['xlsx', 'csv'])
    use_example = st.checkbox('Use the synthetic example data', value=False,
                              help='Made-up samples for trying the app. Not real data.')

    st.header('2. Units')
    unit = st.selectbox('Concentration unit (all ions)', chem.UNITS, index=0)
    alk_choice = st.selectbox('HCO3 column holds', ['Bicarbonate as HCO3', 'Total alkalinity as CaCO3'])
    no3_choice = st.selectbox('NO3 column holds', ['Nitrate as NO3', 'Nitrate as N'])

    st.header('3. Missing values')
    st.caption('Missing pH, temperature and bicarbonate are filled so every sample can be '
               'speciated. Filled samples are drawn hollow wherever the fill affects the plot.')
    ph_how = st.selectbox('Missing pH', ['Mean of the dataset', 'Fixed value'])
    ph_val = st.number_input('Fixed pH', value=DEFAULT_PH, step=0.1) if ph_how == 'Fixed value' else DEFAULT_PH
    t_how = st.selectbox('Missing temperature', ['Mean of the dataset', 'Fixed value'])
    t_val = st.number_input('Fixed temperature (deg C)', value=DEFAULT_TEMP, step=0.5) if t_how == 'Fixed value' else DEFAULT_TEMP
    h_how = st.selectbox('Missing HCO3', ['Mean of the dataset', 'From charge balance', 'Fixed value'])
    h_val = st.number_input('Fixed HCO3 (mmol/L)', value=DEFAULT_HCO3_MMOL, step=0.1) if h_how == 'Fixed value' else DEFAULT_HCO3_MMOL

    st.header('4. Plot options')
    log_conc = st.checkbox('Log axes on concentration plots', value=False,
                           help='Useful when fresh and saline samples share a plot.')

HOW = {'Mean of the dataset': 'mean', 'Fixed value': 'value', 'From charge balance': 'balance'}
rules = (('pH', (HOW[ph_how], ph_val)), ('Temp', (HOW[t_how], t_val)), ('HCO3', (HOW[h_how], h_val)))

# %% load
if use_example:
    raw = chem.example_data().astype(object)
    st.info('Showing the synthetic example dataset. These are made-up samples.')
elif uploaded is not None:
    try:
        raw = chem.read_upload(uploaded)
    except Exception as exc:
        st.error(f'Could not read that file: {exc}')
        st.stop()
else:
    st.info('Upload a filled template, or tick the synthetic example in the sidebar, to begin.')
    st.stop()

v, dropped, errors, problems = process(raw, unit, alk_choice.startswith('Total'),
                                       no3_choice.endswith('as N'), rules)
for p in problems:
    st.error(p)
if v is None:
    st.stop()

count_this_session()
show_use_count()

# %% warnings
n_dropped = 0 if dropped is None else len(dropped)
if n_dropped:
    st.warning(f'{n_dropped} cell(s) were not numbers (for example "<0.05") and were treated as '
               'missing. They are listed on the Data tab. Decide how you want to handle them and '
               're-upload if needed.')
saline = v['Ionic strength (mol/kgw)'] > chem.IONIC_STRENGTH_WARN
if saline.any():
    st.warning(f'{int(saline.sum())} sample(s) have ionic strength above {chem.IONIC_STRENGTH_WARN} '
               'mol/kgw (' + ', '.join(v.loc[saline, 'Sample'].astype(str).head(8)) +
               ('...' if saline.sum() > 8 else '') + '). phreeqc.dat is a freshwater database; '
               'saturation indices for these samples are approximate and a Pitzer database would be '
               'the right tool.')
n_filled = int((v['Filled'] != '').sum())
if n_filled:
    st.info(f'{n_filled} sample(s) had a missing pH, temperature or HCO3 filled. They are hollow on '
            'the plots where the fill matters. Remove those rows and re-upload if you would rather '
            'not show them.')
if errors is not None and len(errors):
    st.warning(f'PHREEQC did not converge for {len(errors)} sample(s); see the Data tab.')

numeric_choices = chem.plottable(v)
# radio buttons and not st.tabs, as tabs redraw every figure on every click
VIEWS = ['Data and checks', 'Piper', 'Classical diagrams', 'Diagnostic gallery', 'Custom plot', 'Export']
view = st.radio('View', VIEWS, horizontal=True, label_visibility='collapsed')

# %% view: data
if view == 'Data and checks':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Samples', len(v))
    c2.metric('Groups', v['Label'].nunique())
    cbe_ok = (v['CBE (%)'].abs() <= 5).sum()
    c3.metric('Charge balance within 5 %', f'{int(cbe_ok)} of {int(v["CBE (%)"].notna().sum())}')
    c4.metric('Filled samples', n_filled)

    summary_cols = ['Sample', 'Label', 'Water type', 'CBE (%)', 'TDS (mg/L)', 'pH', 'log pCO2',
                    'Ionic strength (mol/kgw)'] + [c for c in v.columns if c.startswith('SI ') and v[c].notna().any()] + ['Filled']
    st.subheader('Summary')
    st.dataframe(v[summary_cols].round(3), use_container_width=True, hide_index=True)
    st.caption('CBE is signed: positive means excess cations. TDS is the measured value where '
               'given, otherwise the sum of ions with bicarbonate x 0.4917.')

    if n_dropped:
        st.subheader('Cells treated as missing')
        st.dataframe(dropped.astype(str), hide_index=True)
    if errors is not None and len(errors):
        st.subheader('PHREEQC errors')
        st.dataframe(errors, hide_index=True)

# %% view: Piper
if view == 'Piper':
    c1, c2, c3 = st.columns(3)
    p_color = c1.selectbox('Colour by', ['Group (Label)'] + numeric_choices, key='piper_color')
    p_types = c2.checkbox('Water-type labels', value=True)
    p_legend = c3.checkbox('Legend for the coloured fields', value=False)
    fig = plots.piper(v, color_by=None if p_color.startswith('Group') else p_color,
                      water_legend=p_legend, show_types=p_types)
    show(fig, 'piper', 'dl_piper')
    not_shown = int(v['piper_d_x'].isna().sum())
    if not_shown:
        st.caption(f'{not_shown} sample(s) are missing a major ion and are not on the diagram.')

# %% view: classical
if view == 'Classical diagrams':
    kind = st.selectbox('Diagram', ['Schoeller', 'Stiff'] + list(plots.WQ_DIAGRAMS))
    if kind == 'Schoeller':
        show(plots.schoeller(v), 'schoeller', 'dl_schoeller')
    elif kind == 'Stiff':
        chosen = st.multiselect('Samples', list(v['Sample']), default=list(v['Sample'][:12]),
                                max_selections=MAX_STIFF)
        fig = plots.stiff(v, chosen)
        if fig is not None:
            show(fig, 'stiff', 'dl_stiff')
    else:
        st.caption('Drawn by WQChartPy (Yang et al., 2022, Groundwater) in its own style. '
                   'Not yet restyled to match the rest of the app.')
        fig, note = plots.wqchartpy_figure(v, kind)
        if note:
            st.warning(note)
        if fig is None:
            st.info('Not enough complete samples for this diagram (it needs pH, TDS and all major ions).')
        else:
            show(fig, kind.lower(), 'dl_wq')
            plt.close(fig)

# %% view: gallery
if view == 'Diagnostic gallery':
    c1, c2 = st.columns(2)
    group = c1.selectbox('Plot set', list(plots.GALLERY))
    g_color = c2.selectbox('Colour by', ['Group (Label)'] + numeric_choices, key='gallery_color')
    panels = [p for p in plots.GALLERY[group] if plots.available(v, p)]
    skipped = len(plots.GALLERY[group]) - len(panels)
    if not panels:
        st.info('None of the variables for this set are in the data.')
    for k in range(0, len(panels), PANELS_PER_PAGE):
        fig = plots.gallery_page(v, panels[k:k + PANELS_PER_PAGE], log_conc=log_conc,
                                 color_by=None if g_color.startswith('Group') else g_color)
        show(fig, f'{group.lower().replace(" ", "_").replace("/", "-")}_{k // PANELS_PER_PAGE + 1}',
             f'dl_gallery_{k}')
    if skipped:
        st.caption(f'{skipped} panel(s) skipped because the data do not include those variables.')
    st.caption('Ratios are molar. Seawater lines use standard mean seawater. Reference lines are '
               'guides to interpretation, not fitted to the data.')

# %% view: custom
if view == 'Custom plot':
    c1, c2, c3 = st.columns(3)
    cx = c1.selectbox('x axis', numeric_choices,
                      index=numeric_choices.index('Cl (mmol/L)') if 'Cl (mmol/L)' in numeric_choices else 0)
    cy = c2.selectbox('y axis', numeric_choices,
                      index=numeric_choices.index('Na (mmol/L)') if 'Na (mmol/L)' in numeric_choices else 0)
    cc = c3.selectbox('Colour by', ['Group (Label)'] + numeric_choices, key='custom_color')
    c1, c2, c3, c4, c5 = st.columns(5)
    xlog = c1.checkbox('Log x')
    ylog = c2.checkbox('Log y')
    one = c3.checkbox('1:1 line')
    zero = c4.checkbox('y = 0 line')
    ann = c5.checkbox('Label samples')
    fig = plots.custom(v, cx, cy, xlog=xlog, ylog=ylog, one_to_one=one, zero_line=zero, annotate=ann,
                       color_by=None if cc.startswith('Group') else cc)
    show(fig, 'custom_plot', 'dl_custom')

# %% view: export
if view == 'Export':
    st.subheader('Results table')
    out_cols = [c for c in v.columns if not c.startswith('piper_') and c not in ('Color', 'Marker')]
    settings = pd.DataFrame({'Setting': ['App version', 'Database', 'Input unit', 'HCO3 column', 'NO3 column',
                                         'Missing pH', 'Missing temperature', 'Missing HCO3'],
                             'Value': [APP_VERSION, 'phreeqc.dat (phreeqpython)', unit, alk_choice, no3_choice,
                                       ph_how + (f' ({ph_val})' if ph_how == 'Fixed value' else ''),
                                       t_how + (f' ({t_val})' if t_how == 'Fixed value' else ''),
                                       h_how + (f' ({h_val} mmol/L)' if h_how == 'Fixed value' else '')]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xl:
        v[out_cols].to_excel(xl, sheet_name='Results', index=False)
        settings.to_excel(xl, sheet_name='Settings', index=False)
        if n_dropped:
            dropped.astype(str).to_excel(xl, sheet_name='Cells treated as missing', index=False)
    st.download_button('Download results (.xlsx)', buf.getvalue(), file_name='HydroCHEM_results.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.caption('Every derived variable, saturation index and flag, plus the settings used, so the '
               'run can be reproduced.')

    st.subheader('All figures')
    dpi = st.selectbox('Resolution', [300, 600], index=0)
    if st.button('Build a zip of every figure'):
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('piper.png', plots.fig_png(plots.piper(v), dpi))
            z.writestr('schoeller.png', plots.fig_png(plots.schoeller(v), dpi))
            for gname, gpanels in plots.GALLERY.items():
                gp = [p for p in gpanels if plots.available(v, p)]
                for k in range(0, len(gp), PANELS_PER_PAGE):
                    f = plots.gallery_page(v, gp[k:k + PANELS_PER_PAGE], log_conc=log_conc)
                    fname = gname.lower().replace(' ', '_').replace('/', '-')
                    z.writestr(f'gallery/{fname}_{k // PANELS_PER_PAGE + 1}.png', plots.fig_png(f, dpi))
        st.download_button('Download figures (.zip)', zbuf.getvalue(), file_name='HydroCHEM_figures.zip',
                           mime='application/zip')
