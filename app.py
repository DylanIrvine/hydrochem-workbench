# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# app.py
# Hydrochem Workbench: hydrochemistry plots and PHREEQC saturation indices
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

from core import chem, guidelines, plots
from core.usage import increment as _record_use, status as _usage_status

# %% CONFIG
APP_NAME = 'Hydrochem Workbench'
APP_URL = 'https://hydrochem-workbench.streamlit.app/'
APP_VERSION = '0.6'
PANELS_PER_PAGE = 6
DEFAULT_PH = 7.0
DEFAULT_TEMP = 25.0
DEFAULT_ALK = 100.0   # mg/L as CaCO3
MAX_STIFF = 24

# theme (dark, red) is set in .streamlit/config.toml
st.set_page_config(page_title=APP_NAME, page_icon='logo/favicon.png', layout='centered')

# %% header
head_text, head_logo = st.columns([3, 1], vertical_alignment='center')
with head_text:
    st.title(APP_NAME)
    st.subheader('Hydrochemistry plots and PHREEQC saturation indices')
    st.write('Dylan Irvine, Charles Darwin University.')
    st.caption(f'Version {APP_VERSION}')
with head_logo:
    st.image('logo/hydrochem_badge.png', width='stretch')

# %% intro
st.write(
    'The Hydrochem Workbench takes a spreadsheet of water chemistry, runs every sample through PHREEQC, and draws '
    'the plots a hydrogeochemist would make first. Download the template from the sidebar, paste '
    'your data in (one row per sample, one unit for every ion, alkalinity as CaCO3 straight from '
    'the lab report), choose the unit and upload it.\n\n'
    'What comes back: charge balance, water type, saturation indices and pCO2, a comparison with '
    'Australian water quality guidelines, a Piper diagram, the '
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
        '**This app**\n\n'
        f'Irvine, D. (2026). Hydrochem Workbench (version {APP_VERSION}) [Web application]. '
        f'Charles Darwin University. {APP_URL}\n\n'
        'Please also cite the software it is built on.\n\n'
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
        'Gaillardet, J., Dupre, B., Louvat, P. and Allegre, C.J. (1999). Global silicate weathering '
        'and CO2 consumption rates deduced from the chemistry of large rivers. Chemical Geology, '
        '159, 3-30. https://doi.org/10.1016/S0009-2541(99)00031-5\n\n'
        'Hounslow, A.W. (1995). Water quality data: analysis and interpretation. CRC Lewis '
        'Publishers, Boca Raton. (Source of the source-rock ratio thresholds.)\n\n'
        'Richards, L.A. (ed.) (1954). Diagnosis and improvement of saline and alkali soils. USDA '
        'Agriculture Handbook 60. (Sodium adsorption ratio.)\n\n'
        '**Water quality guidelines**\n\n'
        'ANZG (2018). Australian and New Zealand Guidelines for Fresh and Marine Water Quality. '
        'Australian and New Zealand Governments and Australian state and territory governments, '
        'Canberra. https://www.waterquality.gov.au/anz-guidelines\n\n'
        'ANZECC and ARMCANZ (2000). Australian and New Zealand Guidelines for Fresh and Marine Water '
        'Quality. Volume 1. Australian and New Zealand Environment and Conservation Council and '
        'Agriculture and Resource Management Council of Australia and New Zealand, Canberra.\n\n'
        'ANZG (2025). Toxicant default guideline values for aquatic ecosystem protection: nitrate in '
        'freshwater. Technical brief.\n\n'
        'NHMRC and NRMMC (2011, with later updates). Australian Drinking Water Guidelines 6. National '
        'Health and Medical Research Council, Canberra. '
        'https://guidelines.nhmrc.gov.au/australian-drinking-water-guidelines\n\n'
        '**Other**\n\n'
        'Gimenez-Forcada, E. (2010). Dynamic of sea water interface using hydrochemical facies '
        'evolution diagram. Groundwater, 48(2), 212-216. '
        'https://doi.org/10.1111/j.1745-6584.2009.00649.x')


# %% functions
@st.cache_resource
def get_phreeqc():
    # load the database once
    import phreeqpython
    return phreeqpython.PhreeqPython(database='phreeqc.dat')


def process(raw, unit, n_as_n, p_as_p, dic_unit, rules):
    # clean, convert, fill, run PHREEQC
    # results are kept in session_state (not st.cache_data) so nothing is held after the user leaves
    key = hashlib.sha1((raw.to_csv() + repr((unit, n_as_n, p_as_p, dic_unit, rules))).encode()).hexdigest()
    cached = st.session_state.get('result')
    if cached is not None and cached[0] == key:
        return cached[1]

    with st.spinner('Running PHREEQC ...'):
        df, dropped, extra_cols, problems = chem.clean(raw)
        # duplicates are flagged but still run
        if any(not p.startswith('Duplicate') for p in problems):
            out = (None, dropped, None, problems)
        else:
            mmol, alk_source = chem.to_mmol(df, unit, n_as_n, p_as_p, dic_unit)
            df, mmol, filled = chem.fill_missing(df, mmol, dict(rules))
            phreeqc, errors, mmol = chem.run_phreeqc(df, mmol, get_phreeqc())
            out = (chem.build_table(df, mmol, filled, phreeqc, extra_cols, alk_source), dropped, errors, problems)
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
                       file_name='hydrochem_workbench_template.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    uploaded = st.file_uploader('Upload the filled template', type=['xlsx', 'csv'])
    use_example = st.checkbox('Use the synthetic example data', value=False,
                              help='Made-up samples for trying the app. Not real data.')

    st.header('2. Units')
    unit = st.selectbox('Concentration unit (ion columns)', chem.UNITS, index=0)
    st.caption('**Alkalinity** goes in as it appears on the lab report: total alkalinity, or '
               'bicarbonate and carbonate alkalinity, in mg/L as CaCO3 (whatever unit the ions are '
               'in). Do not convert it. HCO3 and CO3 are calculated by PHREEQC from alkalinity and pH, '
               'or from DIC and pH where there is no alkalinity.')
    n_choice = st.selectbox('NO3 and NH4 columns hold', ['The ion (NO3, NH4)', 'As N (NO3-N, NH3-N)'])
    p_choice = st.selectbox('PO4 column holds', ['Phosphate as PO4', 'Phosphate as P'])
    dic_unit = st.selectbox('DIC column is in', ['mg/L as C', 'mmol/L'],
                            help='DIC is only used for samples with no alkalinity.')

    st.header('3. Missing values')
    st.caption('Missing pH, temperature and alkalinity are filled so every sample can be '
               'speciated. Filled samples are drawn hollow wherever the fill affects the plot.')
    ph_how = st.selectbox('Missing pH', ['Mean of the dataset', 'Fixed value'])
    ph_val = st.number_input('Fixed pH', value=DEFAULT_PH, step=0.1) if ph_how == 'Fixed value' else DEFAULT_PH
    t_how = st.selectbox('Missing temperature', ['Mean of the dataset', 'Fixed value'])
    t_val = st.number_input('Fixed temperature (deg C)', value=DEFAULT_TEMP, step=0.5) if t_how == 'Fixed value' else DEFAULT_TEMP
    h_how = st.selectbox('Missing alkalinity', ['Mean of the dataset', 'From charge balance', 'Fixed value'])
    h_val = st.number_input('Fixed alkalinity (mg/L as CaCO3)', value=DEFAULT_ALK, step=10.0) if h_how == 'Fixed value' else DEFAULT_ALK

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

v, dropped, errors, problems = process(raw, unit, n_choice.startswith('As N'), p_choice.endswith('as P'),
                                       dic_unit, rules)
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
    st.info(f'{n_filled} sample(s) had a missing pH, temperature or alkalinity filled. They are hollow on '
            'the plots where the fill matters. Remove those rows and re-upload if you would rather '
            'not show them.')
if errors is not None and len(errors):
    st.warning(f'PHREEQC did not converge for {len(errors)} sample(s); see the Data tab.')

numeric_choices = chem.plottable(v)
# radio buttons and not st.tabs, as tabs redraw every figure on every click
VIEWS = ['Data and checks', 'Guidelines', 'Piper', 'Classical diagrams', 'Diagnostic gallery', 'Custom plot',
         'Export']
view = st.radio('View', VIEWS, horizontal=True, label_visibility='collapsed')

# %% view: data
if view == 'Data and checks':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Samples', len(v))
    c2.metric('Groups', v['Label'].nunique())
    cbe_ok = (v['CBE (%)'].abs() <= 5).sum()
    c3.metric('Charge balance within 5 %', f'{int(cbe_ok)} of {int(v["CBE (%)"].notna().sum())}')
    c4.metric('Filled samples', n_filled)

    summary_cols = ['Sample', 'Label', 'Water type', 'CBE (%)', 'TDS (mg/L)', 'pH',
                    'Alkalinity (mg/L as CaCO3)', 'HCO3 (mg/L)', 'CO3 (mg/L)', 'log pCO2',
                    'Ionic strength (mol/kgw)'] + [c for c in v.columns if c.startswith('SI ') and v[c].notna().any()] + ['Filled']
    st.subheader('Summary')
    st.dataframe(v[summary_cols].round(3), use_container_width=True, hide_index=True)
    st.caption('HCO3 and CO3 are calculated by PHREEQC from the alkalinity and pH you supplied '
               '(each includes its ion pairs, so HCO3 + CO3 adds up to the carbonate alkalinity). '
               'CBE is signed: positive means excess cations. TDS is the measured value where '
               'given, otherwise the sum of ions with bicarbonate x 0.4917.')

    if n_dropped:
        st.subheader('Cells treated as missing')
        st.dataframe(dropped.astype(str), hide_index=True)
    if errors is not None and len(errors):
        st.subheader('PHREEQC errors')
        st.dataframe(errors, hide_index=True)

# %% view: guidelines
GUIDE_DEFAULT = guidelines.set_names()[:3] + [guidelines.REGIONAL_NAME]
guide_sets = st.session_state.get('guide_sets', GUIDE_DEFAULT)
guide_region = st.session_state.get('guide_region', 'Tropical Australia')
guide_eco = st.session_state.get('guide_eco', 'Lowland river')
if guide_eco not in guidelines.REGIONS[guide_region]:
    guide_eco = list(guidelines.REGIONS[guide_region])[0]
if view == 'Guidelines':
    st.subheader('Comparison with Australian water quality guidelines')
    st.write('Drinking water guidelines and environmental (aquatic ecosystem) guidelines for fresh, '
             'estuarine and marine waters. Choose the sets that suit the water body.')
    guide_sets = st.multiselect('Guideline sets', guidelines.set_names(), default=guide_sets, key='guide_sets')
    if guidelines.REGIONAL_NAME in guide_sets:
        c1, c2 = st.columns(2)
        guide_region = c1.selectbox('Region', list(guidelines.REGIONS),
                                    index=list(guidelines.REGIONS).index(guide_region), key='guide_region')
        ecos = list(guidelines.REGIONS[guide_region])
        guide_eco = c2.selectbox('Ecosystem type', ecos,
                                 index=ecos.index(guide_eco) if guide_eco in ecos else 0, key='guide_eco',
                                 help='Estuaries are the nearest match for brackish water.')
    if not guide_sets:
        st.info('Choose at least one guideline set.')
    else:
        g_summary, g_flags = guidelines.assess(v, guide_sets, guide_region, guide_eco)
        c1, c2 = st.columns(2)
        c1.metric('Samples outside at least one value', f"{int((g_flags['Count'] > 0).sum())} of {len(g_flags)}")
        c2.metric('Guideline values checked', len(g_summary))
        st.markdown('**By guideline value**')
        st.dataframe(g_summary.style.apply(
            lambda r: ['background-color: rgba(255, 75, 75, 0.25)' if r['Outside guideline'] > 0 else ''] * len(r),
            axis=1), use_container_width=True, hide_index=True)
        st.markdown('**By sample**')
        st.dataframe(g_flags[g_flags['Count'] > 0], use_container_width=True, hide_index=True)
    st.caption(
        'This is a screening comparison of single samples, not a compliance assessment. '
        'Fresh, estuarine or marine is your choice: pick the sets, region and ecosystem type that match '
        'the water body. There are no national default values for brackish water as such; the estuary '
        'values are the nearest match. '
        'The aquatic ecosystem values were derived for surface waters, not for groundwater or for '
        'brackish or hypersaline water, and are meant to be compared with the median of monitoring '
        'data, with state or site-specific values taking precedence where they exist. '
        'Where the source gives a range, the upper end is used. '
        'Total N and total P are not in the template, so those guideline values are not checked. '
        'Guideline values are revised from time to time: check the current source before relying on them.')

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
    st.caption('Ratios are molar unless the label says meq. Seawater lines use standard mean seawater. '
               'Reference lines, shaded bands and the notes in the legends are guides to interpretation, '
               'not fitted to the data: check them against what you know of the geology.')

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
    settings = pd.DataFrame({'Setting': ['App version', 'Database', 'Input unit', 'Alkalinity', 'N species', 'PO4 column', 'DIC unit',
                                         'Missing pH', 'Missing temperature', 'Missing alkalinity'],
                             'Value': [APP_VERSION, 'phreeqc.dat (phreeqpython)', unit, 'mg/L as CaCO3; HCO3 and CO3 from PHREEQC (see "Alkalinity from" column)', n_choice, p_choice, dic_unit,
                                       ph_how + (f' ({ph_val})' if ph_how == 'Fixed value' else ''),
                                       t_how + (f' ({t_val})' if t_how == 'Fixed value' else ''),
                                       h_how + (f' ({h_val} mg/L as CaCO3)' if h_how == 'Fixed value' else '')]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xl:
        v[out_cols].to_excel(xl, sheet_name='Results', index=False)
        settings.to_excel(xl, sheet_name='Settings', index=False)
        if n_dropped:
            dropped.astype(str).to_excel(xl, sheet_name='Cells treated as missing', index=False)
        if guide_sets:
            g_summary, g_flags = guidelines.assess(v, guide_sets, guide_region, guide_eco)
            g_summary.to_excel(xl, sheet_name='Guidelines', index=False)
            g_flags.to_excel(xl, sheet_name='Guidelines by sample', index=False)
    st.download_button('Download results (.xlsx)', buf.getvalue(), file_name='hydrochem_workbench_results.xlsx',
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
        st.download_button('Download figures (.zip)', zbuf.getvalue(), file_name='hydrochem_workbench_figures.zip',
                           mime='application/zip')
