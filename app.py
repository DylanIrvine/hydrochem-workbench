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
from matplotlib import colors as mcolors

import core
from core import chem, guidelines, plots, uncertainty, usage

# %% CONFIG
APP_NAME = 'Hydrochem Workbench'
APP_URL = 'https://hydrochem-workbench.streamlit.app/'
APP_VERSION = '0.11.1'   # keep the same as VERSION in core/__init__.py

# after an update on Streamlit Cloud the server can keep old copies of the core
# modules in memory while running the new app.py. If the versions do not match,
# reload them
if getattr(core, 'VERSION', None) != APP_VERSION:
    import importlib
    importlib.reload(core)
    for _m in (chem, guidelines, plots, uncertainty, usage):
        importlib.reload(_m)

_record_use, _usage_status = usage.increment, usage.status

PANELS_PER_PAGE = 6
DEFAULT_PH = 7.0
DEFAULT_TEMP = 25.0
DEFAULT_ALK = 100.0   # mg/L as CaCO3
MAX_STIFF = 24

# theme (dark, red) is set in .streamlit/config.toml
st.set_page_config(page_title=APP_NAME, page_icon='logo/HydroCHEM_icon.png', layout='centered')

# %% settings that survive
# Streamlit drops the value of any widget that is not drawn on a run (another view,
# or no data loaded). Writing the values back each run keeps every setting for the
# whole session, so a new dataset is processed with the same choices
KEEP = ('opt_', 'grp_show_', 'grp_col_', 'grp_mk_', 'guide_', 'piper_color', 'gallery_color',
        'custom_color')
U = uncertainty
for _k in list(st.session_state.keys()):
    if _k.startswith(KEEP):
        st.session_state[_k] = st.session_state[_k]


def opt(key, default, options=None):
    # set a widget's starting value through session_state (so widgets take no
    # value/index argument) and drop a kept value that is no longer an option
    if key in st.session_state and options is not None:
        cur = st.session_state[key]
        if isinstance(cur, list):
            st.session_state[key] = [x for x in cur if x in options]
        elif cur not in options:
            del st.session_state[key]
    if key not in st.session_state:
        st.session_state[key] = default
    return key


# %% header
head_text, head_logo = st.columns([3, 1], vertical_alignment='center')
with head_text:
    st.title(APP_NAME)
    st.subheader('Hydrochemistry plots and PHREEQC saturation indices')
    st.write('Dylan Irvine, Charles Darwin University.')
    st.caption(f'Version {APP_VERSION}')
with head_logo:
    st.image('logo/HydroCHEM_logo.png', width='stretch')

# %% intro
st.write(
    'The Hydrochem Workbench takes a spreadsheet of water chemistry, runs every sample through PHREEQC, and draws '
    'the plots a hydrogeochemist would make first. Download the template from the sidebar, paste '
    'your data in (one row per sample, one unit for every ion, alkalinity as CaCO3 straight from '
    'the lab report), choose the unit and upload it.\n\n'
    'What comes back: charge balance, water type, saturation indices and pCO2, a comparison with '
    'Australian, WHO and FAO (irrigation) water quality guidelines with a bar chart against any of them, a Piper diagram, the '
    'classical diagrams, a gallery of diagnostic ion, ratio and saturation-index plots with '
    'reference lines, and a builder for any plot the gallery does not have. The database is '
    'phreeqc.dat, so results are for fresh water; saline samples are flagged.\n\n'
    'Optionally, the measurement uncertainty of every input (laboratory uncertainty, values below the '
    'limit of reporting, field probe accuracy) can be carried through the whole calculation by Monte '
    'Carlo, so saturation indices, ratios, charge balance and water type come with an interval, or a '
    'probability, instead of a single number.')


# %% usage counter
# counts sessions that process a dataset. Uses Upstash Redis if the secrets
# are set, otherwise a local file that resets when the app restarts. See core/usage.py
# Takes either layout used across the apps:
#   [redis] rest_url / rest_token   (HydroSTITCH)
#   [upstash] url / token           (HydroSILO, HydroPOINT, HydroCAPTURE)
# or top-level UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN, or the same
# names as environment variables
def _counter_secret(names, env):
    try:
        sec = st.secrets
        for section in ('redis', 'upstash'):
            if section in sec:
                for name in names:
                    if name in sec[section] and str(sec[section][name]).strip():
                        return str(sec[section][name]).strip()
        if env in sec and str(sec[env]).strip():
            return str(sec[env]).strip()
    except Exception:
        pass
    return os.environ.get(env) or None


def _counter_args():
    return dict(redis_url=_counter_secret(('rest_url', 'url'), 'UPSTASH_REDIS_REST_URL'),
                redis_token=_counter_secret(('rest_token', 'token'), 'UPSTASH_REDIS_REST_TOKEN'),
                redis_key=_counter_secret(('key',), 'HYDROCHEM_RUN_COUNT_KEY') or 'hydrochem:runs')


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
    # whichever is larger: this session's increment, or the latest read
    total = max(st.session_state.get('use_count') or 0, status.get('count') or 0)
    with _use_count_slot.container():
        if total:
            st.caption(f'{APP_NAME} sessions that have processed data: {total:,}')
        if status['backend'] == 'redis-error':
            st.caption(f":orange[Usage counter: Redis is configured but not reachable ({status['detail']}).]")
        elif status['backend'] in ('file', 'memory'):
            st.caption(':gray[Usage counter is not connected to its database, so this count restarts '
                       'whenever the app restarts.]')


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
        'The Piper diagram here is adapted from WQChartPy. The Durov, Gibbs, Chadha and HFE-D '
        'diagrams are drawn by WQChartPy, with plain-text labels and the uncertainty added by this app.\n\n'
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
        'World Health Organization (2026). Guidelines for drinking-water quality: fourth edition '
        'incorporating the first, second and third addenda. WHO, Geneva. '
        'https://www.who.int/publications/i/item/9789240121225\n\n'
        'Ayers, R.S. and Westcot, D.W. (1985). Water quality for agriculture. FAO Irrigation and '
        'Drainage Paper 29 Rev. 1. Food and Agriculture Organization of the United Nations, Rome. '
        'https://www.fao.org/3/t0234e/t0234e00.htm\n\n'
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
        '**Uncertainty analysis**\n\n'
        'JCGM (2008). Evaluation of measurement data: Supplement 1 to the Guide to the expression of '
        'uncertainty in measurement. Propagation of distributions using a Monte Carlo method. JCGM 101:2008. '
        'https://www.bipm.org/en/committees/cc/wg/jcgm\n\n'
        'JCGM (2008). Evaluation of measurement data: Guide to the expression of uncertainty in '
        'measurement. JCGM 100:2008. (Rectangular distributions for instrument specifications.)\n\n'
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


def process(raw, unit, n_as_n, p_as_p, dic_unit, rules, lor):
    # below-LOR substitution, clean, convert, fill, run PHREEQC
    # lor = (fraction, text, negative)
    # results are kept in session_state (not st.cache_data) so nothing is held after the user leaves
    # returns (results, dropped cells, PHREEQC errors, problems, replaced below-LOR values,
    #          what the uncertainty analysis needs)
    key = hashlib.sha1((raw.to_csv() + repr((unit, n_as_n, p_as_p, dic_unit, rules, lor))).encode()).hexdigest()
    cached = st.session_state.get('result')
    if cached is not None and cached[0] == key:
        return cached[1]

    with st.spinner('Running PHREEQC ...'):
        fraction, lor_text, lor_neg = lor
        raw, replaced = chem.below_lor(raw, fraction, text=lor_text, negative=lor_neg, rows=True)
        below = {(r.Row, r.Column): float(r.LOR) for r in replaced.itertuples()}
        replaced = replaced.drop(columns='Row')
        df, dropped, extra_cols, problems = chem.clean(raw)
        # duplicates are flagged but still run
        if any(not p.startswith('Duplicate') for p in problems):
            out = (None, dropped, None, problems, replaced, None)
        else:
            settings = dict(unit=unit, n_as_n=n_as_n, p_as_p=p_as_p, dic_unit=dic_unit, rules=rules,
                            extra_cols=tuple(extra_cols))
            v, errors = chem.derive(df, get_phreeqc(), **settings)
            out = (v, dropped, errors, problems, replaced,
                   dict(key=key, df=df, below=below, settings=settings))
    st.session_state['result'] = (key, out)
    return out


def monte_carlo(base, table, n):
    # the uncertainty analysis, kept for the session until an input or setting changes
    # the app version is in the key, so a result kept from an older version is not reused
    key = hashlib.sha1((APP_VERSION + base['key'] + table.to_csv() + str(n)).encode()).hexdigest()
    cached = st.session_state.get('mc')
    if cached is not None and cached[0] == key:
        return cached[1]
    bar = st.progress(0.0, text=f'Uncertainty analysis: {n} realisations through PHREEQC ...')
    mc = U.run(base['df'], base['below'], table, base['settings'], get_phreeqc(), n=n,
               progress=lambda f: bar.progress(f, text=f'Uncertainty analysis: {int(round(f * n))} of {n} '
                                                       'realisations'))
    bar.empty()
    st.session_state['mc'] = (key, mc)
    return mc


def show(fig, name, key):
    st.pyplot(fig, width='stretch')
    st.download_button('Download PNG (600 dpi)', plots.fig_png(fig, dpi=600),
                       file_name=name + '.png', mime='image/png', key=key)


# %% sidebar: data in
# the loaded table is kept in session_state, so the uploader is only needed to
# bring in a dataset (the first one, or the next one)
st.session_state.setdefault('upload_n', 0)
with st.sidebar:
    st.header('1. Data')
    st.download_button('Download the template', chem.template_bytes(),
                       file_name='hydrochem_workbench_template.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    use_example = st.checkbox('Use the synthetic example data', key=opt('opt_example', False),
                              help='Made-up samples for trying the app. Not real data.')
    dataset = st.session_state.get('dataset')   # (file name, raw table)
    if not use_example:
        if dataset is None or st.session_state.get('replacing'):
            up = st.file_uploader('Upload the filled template' if dataset is None else 'Upload the next dataset',
                                  type=['xlsx', 'csv'], key=f"uploader_{st.session_state['upload_n']}")
            if dataset is not None:
                st.caption(f'This replaces the current data ({dataset[0]}). All settings are kept.')
                if st.button('Keep the current data', key='keep_data'):
                    st.session_state['replacing'] = False
                    st.rerun()
            if up is not None:
                try:
                    st.session_state['dataset'] = (up.name, chem.read_upload(up))
                except Exception as exc:
                    st.error(f'Could not read that file: {exc}')
                else:
                    st.session_state['replacing'] = False
                    st.session_state['upload_n'] += 1
                    st.rerun()
        else:
            st.markdown(f'**Loaded:** {dataset[0]} ({len(dataset[1])} rows)')
            if st.button('Load another dataset', key='load_another'):
                st.session_state['replacing'] = True
                st.rerun()
            st.caption('Replaces the current data with a new file. Every setting (units, below-LOR, '
                       'missing values, plot options, group colours and markers, guidelines) is kept, '
                       'so several sites, or an edited sheet, can be run the same way.')

    st.header('2. Units')
    unit = st.selectbox('Concentration unit (ion columns)', chem.UNITS, key=opt('opt_unit', chem.UNITS[0]))
    st.caption('**Alkalinity** goes in as it appears on the lab report: total alkalinity, or '
               'bicarbonate and carbonate alkalinity, in mg/L as CaCO3 (whatever unit the ions are '
               'in). Do not convert it. **HCO3 and CO3** are optional, in the ion unit: where you '
               'enter them they are used as given. If both are blank, PHREEQC calculates them from '
               'alkalinity and pH (or DIC and pH). If only one is entered, the other is the rest of '
               'the alkalinity.')
    N_OPTS = ['The ion (NO3, NH4)', 'As N (NO3-N, NH3-N)']
    P_OPTS = ['Phosphate as PO4', 'Phosphate as P']
    DIC_OPTS = ['mg/L as C', 'mmol/L']
    n_choice = st.selectbox('NO3 and NH4 columns hold', N_OPTS, key=opt('opt_n_form', N_OPTS[0]))
    p_choice = st.selectbox('PO4 column holds', P_OPTS, key=opt('opt_p_form', P_OPTS[0]))
    dic_unit = st.selectbox('DIC column is in', DIC_OPTS, key=opt('opt_dic_unit', DIC_OPTS[0]),
                            help='DIC is only used for samples with no alkalinity.')

    st.header('3. Below the LOR')
    st.caption('Values below the limit of reporting (LOR) can be replaced by a fraction of the LOR. '
               'Anything not replaced is treated as missing and listed on the Data tab.')
    lor_text = st.toggle('Replace "<" values (e.g. <0.05)', key=opt('opt_lor_text', True),
                         help='Concentration columns, TDS and any extra numeric column.')
    lor_neg = st.toggle('Replace negative concentrations', key=opt('opt_lor_neg', True),
                        help='Some labs report -0.05 to mean <0.05. Only the ion, alkalinity and DIC columns.')
    LOR_FRACTIONS = {'0.5 x LOR (half)': 0.5, '0.25 x LOR': 0.25, '0.1 x LOR': 0.1,
                     '1 x LOR (the LOR itself)': 1.0, 'Other': None}
    lor_pick = st.selectbox('Replace with', list(LOR_FRACTIONS), key=opt('opt_lor_pick', '0.5 x LOR (half)'),
                            disabled=not (lor_text or lor_neg))
    lor_fraction = LOR_FRACTIONS[lor_pick]
    if lor_fraction is None:
        lor_fraction = st.number_input('Fraction of the LOR', min_value=0.001, max_value=1.0,
                                       step=0.05, format='%.3f', key=opt('opt_lor_frac', 0.5),
                                       disabled=not (lor_text or lor_neg))

    st.header('4. Missing values')
    st.caption('Missing pH, temperature and alkalinity are filled so every sample can be '
               'speciated. Filled samples are drawn hollow wherever the fill affects the plot.')
    FILL2 = ['Mean of the dataset', 'Fixed value']
    FILL3 = ['Mean of the dataset', 'From charge balance', 'Fixed value']
    ph_how = st.selectbox('Missing pH', FILL2, key=opt('opt_ph_how', FILL2[0]))
    ph_val = st.number_input('Fixed pH', step=0.1, key=opt('opt_ph_val', DEFAULT_PH)) if ph_how == 'Fixed value' else DEFAULT_PH
    t_how = st.selectbox('Missing temperature', FILL2, key=opt('opt_t_how', FILL2[0]))
    t_val = st.number_input('Fixed temperature (deg C)', step=0.5, key=opt('opt_t_val', DEFAULT_TEMP)) if t_how == 'Fixed value' else DEFAULT_TEMP
    h_how = st.selectbox('Missing alkalinity', FILL3, key=opt('opt_h_how', FILL3[0]))
    h_val = st.number_input('Fixed alkalinity (mg/L as CaCO3)', step=10.0, key=opt('opt_h_val', DEFAULT_ALK)) if h_how == 'Fixed value' else DEFAULT_ALK

    st.header('5. Plot options')
    log_conc = st.checkbox('Log axes on concentration plots', key=opt('opt_log_conc', False),
                           help='Useful when fresh and saline samples share a plot.')
    letters = st.selectbox('Panel letters (multipanel figures)', ['None'] + list(plots.PANEL_LABELS),
                           key=opt('opt_letters', 'None', ['None'] + list(plots.PANEL_LABELS)),
                           help='For every figure with more than one panel: the gallery pages, the Stiff '
                                'grid and the Gibbs diagram.')
    letters = None if letters == 'None' else letters
    letters_pos = st.selectbox('Panel letter position', plots.PANEL_LABEL_POS,
                               key=opt('opt_letters_pos', plots.PANEL_LABEL_POS[0], plots.PANEL_LABEL_POS),
                               disabled=letters is None)

    st.header('6. Uncertainty')
    use_unc = st.checkbox('Include uncertainty analysis', key=opt('opt_unc', False),
                          help='Carries the measurement uncertainty of every input through the whole '
                               'calculation (Monte Carlo). Intervals are drawn as light grey lines.')
    unc_n = st.number_input('Realisations', min_value=20, max_value=2000, step=50,
                            key=opt('opt_unc_n', U.DEFAULT_N), disabled=not use_unc,
                            help='More realisations give smoother intervals and take longer. '
                                 f'{U.DEFAULT_N} is enough for the 5th and 95th percentiles.')
    unc_col = st.color_picker('Uncertainty colour', key=opt('opt_unc_colour', '#666666'), disabled=not use_unc,
                              help='Colour of the interval lines, error bars, clouds and outlines on every plot.')
    show_unc = st.checkbox('Show and edit the uncertainty values', key=opt('opt_unc_show', False),
                           disabled=not use_unc,
                           help='Shows the table of uncertainties used for each input, how the defaults were '
                                'determined, and how to estimate your own.')
    if use_unc:
        st.caption(f'Intervals are {U.INTERVAL} (5th to 95th percentile). '
                   + ('Edited values are in use.' if (U.mark_edited(U.check_table(
                       st.session_state.get('unc_current', st.session_state.get('unc_base', U.default_table()))))
                       ['Source'] == 'Entered by user').any() else 'Default values are in use.'))

HOW = {'Mean of the dataset': 'mean', 'Fixed value': 'value', 'From charge balance': 'balance'}
rules = (('pH', (HOW[ph_how], ph_val)), ('Temp', (HOW[t_how], t_val)), ('HCO3', (HOW[h_how], h_val)))

# %% load
if use_example:
    raw = chem.example_data().astype(object)
    data_name = 'Synthetic example data'
    st.info('Showing the synthetic example dataset. These are made-up samples.')
elif dataset is not None:
    data_name, raw = dataset
    st.markdown(f'**Dataset:** {data_name}')
else:
    st.info('Upload a filled template, or tick the synthetic example in the sidebar, to begin.')
    st.stop()

v, dropped, errors, problems, replaced, base = process(raw, unit, n_choice.startswith('As N'), p_choice.endswith('as P'),
                                                       dic_unit, rules, (lor_fraction, lor_text, lor_neg))
for p in problems:
    st.error(p)
if v is None:
    st.stop()

count_this_session()
show_use_count()

# %% warnings
n_dropped = 0 if dropped is None else len(dropped)
n_replaced = 0 if replaced is None else len(replaced)
if n_replaced:
    st.info(f'{n_replaced} value(s) below the LOR were replaced with {lor_fraction:g} x LOR. '
            'They are listed on the Data tab.')
if n_dropped:
    st.warning(f'{n_dropped} cell(s) were not usable numbers and were treated as missing. '
               'They are listed on the Data tab. The "Below the LOR" options in the sidebar handle '
               '"<" and negative values; anything else (e.g. "bd", "n/a") needs fixing in the file.')
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

# %% groups: colour, marker, show or hide
# defaults come from the palette, or the Color/Marker columns in the file
labels = list(pd.unique(v['Label']))
default_styles = plots.group_styles(v)
marker_codes = list(plots.MARKER_NAMES)
marker_names = list(plots.MARKER_NAMES.values())
with st.expander(f'**Groups (Label): colours, markers and which to plot** ({len(labels)} group(s))',
                 expanded=True):
    if st.button('Reset to defaults', key='grp_reset'):
        for lab in labels:
            for pre in ('grp_show_', 'grp_col_', 'grp_mk_'):
                st.session_state.pop(pre + lab, None)
    h1, h2, h3, h4 = st.columns([1, 3, 1.4, 2.6])
    h1.caption('Plot')
    h2.caption('Label')
    h3.caption('Colour')
    h4.caption('Marker')
    group_style = {}
    shown = []
    for lab in labels:
        col0, mk0 = default_styles[lab]
        names = marker_names if mk0 in marker_codes else marker_names + [mk0]
        codes = marker_codes if mk0 in marker_codes else marker_codes + [mk0]
        c1, c2, c3, c4 = st.columns([1, 3, 1.4, 2.6], vertical_alignment='center')
        if c1.checkbox('Plot', key=opt('grp_show_' + lab, True), label_visibility='collapsed'):
            shown.append(lab)
        c2.write(lab)
        col = c3.color_picker('Colour', key=opt('grp_col_' + lab, mcolors.to_hex(col0)),
                              label_visibility='collapsed')
        mk_name = c4.selectbox('Marker', names, key=opt('grp_mk_' + lab, names[codes.index(mk0)], names),
                               label_visibility='collapsed')
        group_style[lab] = (col, codes[names.index(mk_name)])
    st.caption('Unticked groups are left off every plot. The tables and the results download '
               'keep all samples.')

# %% uncertainty inputs and the Monte Carlo run
ss = st.session_state
ss.setdefault('unc_base', U.default_table())
mc = None
unc_table = U.mark_edited(U.check_table(ss.get('unc_current', ss['unc_base'])))
show_inputs = use_unc and show_unc
if not show_inputs:
    # the table is not drawn: keep any edits for when it is shown again
    ss['unc_base'] = unc_table
    ss.pop('unc_editor', None)
else:
    st.subheader('Uncertainty inputs')
    st.write('These are the uncertainties drawn for every measured value. Edit any cell to use values for '
             'your laboratory and instruments; the analysis reruns with them. ' + U.LOR_NOTE)
    if st.button('Reset to defaults', key='unc_reset'):
        ss['unc_base'] = U.default_table()
        ss.pop('unc_editor', None)
        ss.pop('unc_current', None)
    edited = st.data_editor(
        ss['unc_base'], key='unc_editor', hide_index=True, num_rows='fixed', width='stretch',
        disabled=['Variable', 'Source'],
        column_config={'LOR': st.column_config.NumberColumn(min_value=0.0, format='%g'),
                       'Relative SD (%)': st.column_config.NumberColumn(min_value=0.0, format='%g'),
                       'Absolute SD': st.column_config.NumberColumn(min_value=0.0, format='%g')})
    ss['unc_current'] = edited
    unc_table = U.mark_edited(U.check_table(edited))
    with st.expander('**How the default values were determined**'):
        st.markdown(U.DEFAULTS_TEXT)
    with st.expander('**How to estimate your own values**'):
        st.markdown(U.OWN_VALUES_TEXT)
    st.caption(U.method_text(unc_n, U.SEED))
if use_unc:
    # the colour goes with the result to the plots (a new dict, the kept result is not changed)
    mc = dict(monte_carlo(base, unc_table, int(unc_n)), colour=unc_col)
    if mc['fails']:
        st.warning(f"PHREEQC did not converge in {mc['fails']} sample-realisation(s); those are left out of "
                   'the intervals.')

v = v.copy()
v['Color'] = v['Label'].map(lambda l: group_style[l][0])
v['Marker'] = v['Label'].map(lambda l: group_style[l][1])
# vp is what the plots see
vp = v[v['Label'].isin(shown)]

numeric_choices = chem.plottable(v)
color_opts = ['Group (Label)'] + numeric_choices
# radio buttons and not st.tabs, as tabs redraw every figure on every click
VIEWS = ['Data and checks', 'Guidelines', 'Piper', 'Classical diagrams', 'Diagnostic gallery', 'Custom plot',
         'Export']
view = st.radio('View', VIEWS, horizontal=True, label_visibility='collapsed', key=opt('opt_view', VIEWS[0], VIEWS))

# %% view: data
if view == 'Data and checks':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Samples', len(v))
    c2.metric('Groups', v['Label'].nunique())
    cbe_ok = (v['CBE (%)'].abs() <= 5).sum()
    c3.metric('Charge balance within 5 %', f'{int(cbe_ok)} of {int(v["CBE (%)"].notna().sum())}')
    c4.metric('Filled samples', n_filled)

    summary_cols = ['Sample', 'Label', 'Water type', 'Chadha field', 'CBE (%)', 'TDS (mg/L)', 'pH',
                    'Alkalinity (mg/L as CaCO3)', 'HCO3 (mg/L)', 'CO3 (mg/L)', 'HCO3, CO3 from', 'log pCO2',
                    'Ionic strength (mol/kgw)'] + [c for c in v.columns if c.startswith('SI ') and v[c].notna().any()] + ['Filled']
    st.subheader('Summary')
    st.dataframe(v[summary_cols].round(3), width='stretch', hide_index=True)
    st.caption('HCO3 and CO3 are your values where you entered them. Where you entered only one, '
               'the other is the rest of the alkalinity. Where you entered neither, PHREEQC calculates '
               'both from the alkalinity and pH (each includes its ion pairs, so HCO3 + 2 CO3, in '
               'meq/L, adds up to the carbonate alkalinity). "HCO3, CO3 from" shows which. '
               'Saturation indices and pCO2 always come from the alkalinity (or DIC) and pH. '
               'CBE is signed: positive means excess cations. TDS is the measured value where '
               'given, otherwise the sum of ions with bicarbonate x 0.4917.')

    if mc is not None:
        st.subheader('Uncertainty')
        st.dataframe(U.summary(v, mc).round(1), width='stretch', hide_index=True)
        st.caption(f"How often the {mc['n']} realisations agree with the measured result: the same water type "
                   '(Piper), the same Chadha field, a charge balance within 5 %, and each saturation index above '
                   'zero (oversaturated). Values near 50 % mean the data cannot say which side the sample is on. '
                   f'CBE 5th and 95th percentiles bound the {U.INTERVAL} interval. Samples with a filled pH, '
                   'temperature or alkalinity carry no uncertainty for the filled value itself.')

    if n_replaced:
        st.subheader('Values below the LOR that were replaced')
        st.dataframe(replaced.astype({'Value': str}), hide_index=True)
    if n_dropped:
        st.subheader('Cells treated as missing')
        st.dataframe(dropped.astype(str), hide_index=True)
    if errors is not None and len(errors):
        st.subheader('PHREEQC errors')
        st.dataframe(errors, hide_index=True)

# %% view: guidelines
GUIDE_DEFAULT = guidelines.set_names()[:3] + [guidelines.REGIONAL_NAME]



def _eco_key(region):
    ecos = list(guidelines.REGIONS[region])
    return opt('guide_eco', 'Lowland river' if 'Lowland river' in ecos else ecos[0], ecos), ecos


guide_sets = st.session_state[opt('guide_sets', GUIDE_DEFAULT, guidelines.set_names())]
guide_region = st.session_state[opt('guide_region', 'Tropical Australia', list(guidelines.REGIONS))]
guide_eco = st.session_state[_eco_key(guide_region)[0]]
if view == 'Guidelines':
    st.subheader('Comparison with water quality guidelines')
    st.write('Australian drinking water (ADWG) and aquatic ecosystem (ANZG) guidelines for fresh, '
             'estuarine and marine waters, plus two international sets: the WHO Guidelines for '
             'drinking-water quality, and the FAO guidelines for irrigation water (Ayers and Westcot 1985). '
             'Choose the sets that suit the water body and its use.')
    guide_sets = st.multiselect('Guideline sets', guidelines.set_names(), key='guide_sets')
    if guidelines.REGIONAL_NAME in guide_sets:
        c1, c2 = st.columns(2)
        guide_region = c1.selectbox('Region', list(guidelines.REGIONS), key='guide_region')
        eco_key, ecos = _eco_key(guide_region)
        guide_eco = c2.selectbox('Ecosystem type', ecos, key=eco_key,
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
            axis=1), width='stretch', hide_index=True)
        st.markdown('**By sample**')
        st.dataframe(g_flags[g_flags['Count'] > 0], width='stretch', hide_index=True)
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
        'WHO values are an international reference from which countries set their own standards; they are '
        'not standards themselves. The FAO irrigation guidelines rate the degree of restriction on use as '
        'none, slight to moderate, or severe; a sample is counted as outside the guideline from slight to '
        'moderate upward, and severe cases are marked. FAO values assume semi-arid to arid conditions with '
        'good drainage and are not absolute limits. '
        'Guideline values are revised from time to time: check the current source before relying on them.')

    # bar chart of one parameter against one guideline
    st.subheader('Bar chart against a guideline')
    if vp.empty:
        st.info('No groups are ticked to plot. Tick at least one in the Groups panel above.')
    else:
        all_rows = guidelines.chart_rows(guide_region, guide_eco)
        chart_vars = list(dict.fromkeys(g['var'] for _, _, g in all_rows
                                        if g['var'] in vp.columns and vp[g['var']].notna().any()))
        if not chart_vars:
            st.info('None of the parameters with guideline values are in the data.')
        else:
            c1, c2 = st.columns([1, 2])
            g_var = c1.selectbox('Parameter', chart_vars, format_func=lambda c: c,
                                 key=opt('opt_gchart_var', chart_vars[0], chart_vars))
            rows_here = [(lab, g) for lab, _, g in all_rows if g['var'] == g_var]
            rule_labels = [lab for lab, _ in rows_here]
            g_rule = c2.selectbox('Guideline', rule_labels, key=opt('opt_gchart_rule', rule_labels[0], rule_labels),
                                  help='Every guideline for this parameter, from all the sets.')
            g_row = dict(rows_here)[g_rule]
            c1, c2, c3, c4 = st.columns(4)
            BAR_MODES = ['Group colours', 'One colour']
            bar_mode = c1.radio('Bars', BAR_MODES, key=opt('opt_gchart_barmode', BAR_MODES[0], BAR_MODES))
            bar_col = c2.color_picker('Bar colour', key=opt('opt_gchart_barcol', '#0089D0'),
                                      disabled=bar_mode != 'One colour')
            line_col = c3.color_picker('Guideline line colour', key=opt('opt_gchart_linecol', '#222222'))
            hl_col = c4.color_picker('Hatching colour', key=opt('opt_gchart_hlcol', '#000000'),
                                     help='Bars outside the guideline are hatched in this colour.')
            c1, c2, c3 = st.columns(3)
            hl = c1.checkbox('Hatch bars outside the guideline', key=opt('opt_gchart_hl', False))
            g_log = c2.checkbox('Log y axis', key=opt('opt_gchart_log', False))
            g_sort = c3.checkbox('Sort by value', key=opt('opt_gchart_sort', False))
            fig = plots.guideline_bars(vp, g_var, guidelines.chart_lines(vp, g_row),
                                       bar_colour=bar_col if bar_mode == 'One colour' else None,
                                       highlight=hl_col if hl else None, line_colour=line_col,
                                       log=g_log, sort=g_sort, unc=mc)
            if fig is None:
                st.info('No plotted sample has a value for this parameter.')
            else:
                safe = g_var.split(' (')[0].replace('/', '-').replace(' ', '_')
                show(fig, f'guideline_bars_{safe}', 'dl_gchart')
                st.caption(f"Source: {g_row['source']}. " + (g_row.get('note') or '') +

                           (' Samples with a value below zero or none are left out.' if g_log else ''))

# %% plots need at least one group
PLOT_VIEWS = ('Piper', 'Classical diagrams', 'Diagnostic gallery', 'Custom plot')
if view in PLOT_VIEWS and vp.empty:
    st.info('No groups are ticked to plot. Tick at least one in the Groups panel above.')
    st.stop()

# %% view: Piper
if view == 'Piper':
    c1, c2, c3 = st.columns(3)
    p_color = c1.selectbox('Colour by', color_opts, key=opt('piper_color', color_opts[0], color_opts))
    p_types = c2.checkbox('Water-type labels', key=opt('opt_piper_types', True))
    p_legend = c3.checkbox('Legend for the coloured fields', key=opt('opt_piper_legend', False))
    p_unc = 'Off'
    if mc is not None:
        p_unc = st.radio('Uncertainty on the Piper', plots.PIPER_UNC_MODES, horizontal=True,
                         key=opt('opt_piper_unc', 'Clouds', plots.PIPER_UNC_MODES),
                         help='Clouds: every realisation as a small grey dot. Outlines: a line round the '
                              'central 90 % of each sample\'s realisations. Off is cleaner for large datasets.')
    fig = plots.piper(vp, color_by=None if p_color.startswith('Group') else p_color,
                      water_legend=p_legend, show_types=p_types, unc=mc, unc_mode=p_unc)
    show(fig, 'piper', 'dl_piper')
    not_shown = int(vp['piper_d_x'].isna().sum())
    if not_shown:
        st.caption(f'{not_shown} sample(s) are missing a major ion and are not on the diagram.')

# %% view: classical
if view == 'Classical diagrams':
    kinds = ['Schoeller', 'Stiff'] + list(plots.WQ_DIAGRAMS)
    kind = st.selectbox('Diagram', kinds, key=opt('opt_classical', kinds[0], kinds))
    if kind == 'Schoeller':
        show(plots.schoeller(vp, unc=mc), 'schoeller', 'dl_schoeller')
    elif kind == 'Stiff':
        samples = list(vp['Sample'])
        chosen = st.multiselect('Samples', samples, key=opt('opt_stiff_samples', samples[:12], samples),
                                max_selections=MAX_STIFF)
        fig = plots.stiff(vp, chosen, letters=letters, letters_pos=letters_pos)
        if fig is not None:
            show(fig, 'stiff', 'dl_stiff')
    else:
        st.caption('Diagram geometry and sample positions from WQChartPy (Yang et al., 2022, Groundwater). '
                   'This app sets the text to plain type and adds the uncertainty; the layout and '
                   'markers are otherwise WQChartPy\'s.')
        wq_unc = 'Off'
        if mc is not None:
            wq_unc = st.radio('Uncertainty on this diagram', plots.PIPER_UNC_MODES, horizontal=True,
                              key=opt('opt_wq_unc', 'Clouds', plots.PIPER_UNC_MODES),
                              help='Clouds: every realisation as a small grey dot. Outlines: a line round the '
                                   'central 90 % of each sample\'s realisations. Applies to the Durov, Gibbs, '
                                   'Chadha and HFE-D diagrams.')
        fig, note = plots.wqchartpy_figure(vp, kind, letters=letters, letters_pos=letters_pos, unc=mc,
                                           unc_mode=wq_unc)
        if note:
            st.warning(note)
        if fig is None:
            st.info('Not enough complete samples for this diagram (it needs pH, TDS and all major ions).')
        else:
            show(fig, kind.lower(), 'dl_wq')
            plt.close(fig)
            if mc is not None and wq_unc != 'Off' and kind == 'HFE-D':
                st.caption('The HFE-D position jumps when the dominant ion changes, so realisations of a '
                           'sample near a boundary can land in another field, away from the sample. '
                           'Those are drawn as dots in both modes: the sample\'s facies is uncertain.')
        if kind == 'Chadha':
            st.markdown(
                'The axes are differences in milliequivalent percentages: (Ca + Mg) less (Na + K) as a '
                'percentage of the cations, and (HCO3 + CO3) less (Cl + SO4) as a percentage of the anions. '
                'Chadha (1999) numbered eight fields:\n'
                '- **1** (right half): alkaline earths exceed alkali metals.\n'
                '- **2** (left half): alkali metals exceed alkaline earths.\n'
                '- **3** (upper half): weak acid anions (HCO3 + CO3) exceed strong acid anions (Cl + SO4).\n'
                '- **4** (lower half): strong acid anions exceed weak acid anions.\n'
                + ''.join(f'- **{k}** ({q}): {chem.CHADHA_FIELDS[k]}.\n' for k, q in
                          [(5, 'upper right'), (6, 'lower right'), (7, 'lower left'), (8, 'upper left')]))
            st.caption(
                'Fields 5 to 8 are often read as processes: 5 as recharge water, 6 as reverse ion exchange, '
                '7 as end-member (e.g. seawater-influenced or evaporated) water, and 8 as base (normal) ion '
                'exchange. Those readings are interpretations and need supporting evidence, such as '
                'chloro-alkaline indices or Na/Cl ratios. Many later papers number the four quadrants 1 to 4 '
                'instead of 5 to 8, so check which scheme a paper uses. Samples close to the axes are only weakly '
                'one type or the other. The field of each sample is in the Chadha field column of the results '
                'download.')
            counts = vp['Chadha field'].value_counts()
            st.dataframe(pd.DataFrame({'Field': counts.index, 'Samples': counts.values}),
                         hide_index=True)

# %% view: gallery
if view == 'Diagnostic gallery':
    c1, c2 = st.columns(2)
    sets = list(plots.GALLERY)
    group = c1.selectbox('Plot set', sets, key=opt('opt_gallery_set', sets[0], sets))
    g_color = c2.selectbox('Colour by', color_opts, key=opt('gallery_color', color_opts[0], color_opts))
    panels = [p for p in plots.GALLERY[group] if plots.available(vp, p)]
    skipped = len(plots.GALLERY[group]) - len(panels)
    if not panels:
        st.info('None of the variables for this set are in the data.')
    for k in range(0, len(panels), PANELS_PER_PAGE):
        fig = plots.gallery_page(vp, panels[k:k + PANELS_PER_PAGE], log_conc=log_conc,
                                 letters=letters, letters_pos=letters_pos,
                                 color_by=None if g_color.startswith('Group') else g_color, unc=mc)
        show(fig, f'{group.lower().replace(" ", "_").replace("/", "-")}_{k // PANELS_PER_PAGE + 1}',
             f'dl_gallery_{k}')
    if skipped:
        st.caption(f'{skipped} panel(s) skipped because the data do not include those variables.')
    st.caption('Ratios are molar unless the label says meq. Seawater lines use standard mean seawater. '
               'Reference lines, shaded bands and the notes in the legends are guides to interpretation, '
               'not fitted to the data: check them against what you know of the geology.' +
               (f' Grey lines are {U.INTERVAL} intervals from the uncertainty analysis; the axes are set by the '
                'points, so a wide interval can run off the panel.' if mc is not None else ''))

# %% view: custom
if view == 'Custom plot':
    c1, c2, c3 = st.columns(3)
    nc = numeric_choices
    cx = c1.selectbox('x axis', nc, key=opt('opt_custom_x', 'Cl (mmol/L)' if 'Cl (mmol/L)' in nc else nc[0], nc))
    cy = c2.selectbox('y axis', nc, key=opt('opt_custom_y', 'Na (mmol/L)' if 'Na (mmol/L)' in nc else nc[0], nc))
    cc = c3.selectbox('Colour by', color_opts, key=opt('custom_color', color_opts[0], color_opts))
    c1, c2, c3, c4, c5 = st.columns(5)
    xlog = c1.checkbox('Log x', key=opt('opt_custom_xlog', False))
    ylog = c2.checkbox('Log y', key=opt('opt_custom_ylog', False))
    one = c3.checkbox('1:1 line', key=opt('opt_custom_one', False))
    zero = c4.checkbox('y = 0 line', key=opt('opt_custom_zero', False))
    ann = c5.checkbox('Label samples', key=opt('opt_custom_labels', False))
    fig = plots.custom(vp, cx, cy, xlog=xlog, ylog=ylog, one_to_one=one, zero_line=zero, annotate=ann,
                       color_by=None if cc.startswith('Group') else cc, unc=mc)
    show(fig, 'custom_plot', 'dl_custom')

# %% view: export
if view == 'Export':
    st.subheader('Results table')
    out_cols = [c for c in v.columns if not c.startswith('piper_') and c not in ('Color', 'Marker')]
    lor_used = ('Not replaced' if not (lor_text or lor_neg) else
                f'{lor_fraction:g} x LOR for ' + ' and '.join(
                    [w for w, on in (('"<" values', lor_text), ('negative concentrations', lor_neg)) if on]))
    ss = st.session_state
    guide_txt = '; '.join(guide_sets) if guide_sets else 'None'
    if guidelines.REGIONAL_NAME in guide_sets:
        guide_txt += f' (region: {guide_region}; ecosystem: {guide_eco})'
    rows = [
        ('Run', 'App version', APP_VERSION),
        ('Run', 'Run at (UTC)', pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M')),
        ('Run', 'Dataset', data_name),
        ('Run', 'Samples', len(v)),
        ('Run', 'Database', 'phreeqc.dat (phreeqpython)'),
        ('Units', 'Ion columns', unit),
        ('Units', 'Alkalinity', 'mg/L as CaCO3; HCO3 and CO3 as entered where given, otherwise from '
                                'PHREEQC (see the "Alkalinity from" and "HCO3, CO3 from" columns)'),
        ('Units', 'NO3 and NH4 columns', n_choice),
        ('Units', 'PO4 column', p_choice),
        ('Units', 'DIC column', dic_unit),
        ('Below the LOR', 'Replace "<" values', 'Yes' if lor_text else 'No'),
        ('Below the LOR', 'Replace negative concentrations', 'Yes' if lor_neg else 'No'),
        ('Below the LOR', 'Replaced with', f'{lor_fraction:g} x LOR' if (lor_text or lor_neg) else 'Not replaced'),
        ('Missing values', 'Missing pH', ph_how + (f' ({ph_val})' if ph_how == 'Fixed value' else '')),
        ('Missing values', 'Missing temperature', t_how + (f' ({t_val} deg C)' if t_how == 'Fixed value' else '')),
        ('Missing values', 'Missing alkalinity', h_how + (f' ({h_val} mg/L as CaCO3)' if h_how == 'Fixed value' else '')),
        ('Groups', 'Plotted', ', '.join(shown) if shown else 'None'),
        ('Groups', 'Not plotted', ', '.join(l for l in labels if l not in shown) or 'None'),
        ('Groups', 'Colours and markers', 'See the Groups sheet'),
        ('Plots', 'Log axes on concentration plots', 'Yes' if log_conc else 'No'),
        ('Plots', 'Panel letters', letters or 'None'),
        ('Plots', 'Panel letter position', letters_pos if letters else ''),
        ('Plots', 'Piper: colour by', ss.get('piper_color', 'Group (Label)')),
        ('Plots', 'Piper: water-type labels', 'Yes' if ss.get('opt_piper_types', True) else 'No'),
        ('Plots', 'Piper: legend for the coloured fields', 'Yes' if ss.get('opt_piper_legend', False) else 'No'),
        ('Plots', 'Gallery: colour by', ss.get('gallery_color', 'Group (Label)')),
        ('Plots', 'Custom plot', f"x = {ss.get('opt_custom_x', '')}, y = {ss.get('opt_custom_y', '')}, "
                                 f"colour by {ss.get('custom_color', 'Group (Label)')}"),
        ('Plots', 'Figure zip resolution (dpi)', ss.get('opt_dpi', 300)),
        ('Uncertainty', 'Included', 'Yes' if mc is not None else 'No'),
        ('Uncertainty', 'Method', U.method_text(mc['n'], mc['seed']) if mc is not None else ''),
        ('Uncertainty', 'Inputs', 'See the Uncertainty inputs sheet' if mc is not None else ''),
        ('Uncertainty', 'Colour', unc_col if mc is not None else ''),
        ('Uncertainty', 'Piper display', ss.get('opt_piper_unc', 'Clouds') if mc is not None else ''),
        ('Uncertainty', 'Durov, Gibbs, Chadha, HFE-D display', ss.get('opt_wq_unc', 'Clouds') if mc is not None else ''),
        ('Guidelines', 'Sets', guide_txt),
        ('Guidelines', 'Bar chart', f"{ss.get('opt_gchart_var', '')}: {ss.get('opt_gchart_rule', '')}"),
    ]
    settings = pd.DataFrame(rows, columns=['Section', 'Setting', 'Value'])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xl:
        v[out_cols].to_excel(xl, sheet_name='Results', index=False)
        settings.to_excel(xl, sheet_name='Settings', index=False)
        pd.DataFrame({'Label': labels, 'Plotted': [lab in shown for lab in labels],
                      'Colour': [group_style[lab][0] for lab in labels],
                      'Marker': [group_style[lab][1] for lab in labels]}).to_excel(
            xl, sheet_name='Groups', index=False)
        if n_replaced:
            replaced.astype({'Value': str}).to_excel(xl, sheet_name='Below LOR replaced', index=False)
        if n_dropped:
            dropped.astype(str).to_excel(xl, sheet_name='Cells treated as missing', index=False)
        if guide_sets:
            g_summary, g_flags = guidelines.assess(v, guide_sets, guide_region, guide_eco)
            g_summary.to_excel(xl, sheet_name='Guidelines', index=False)
            g_flags.to_excel(xl, sheet_name='Guidelines by sample', index=False)
        if mc is not None:
            U.summary(v, mc).to_excel(xl, sheet_name='Uncertainty summary', index=False)
            U.interval_table(v, mc).to_excel(xl, sheet_name='Uncertainty intervals', index=False)
            unc_table.to_excel(xl, sheet_name='Uncertainty inputs', index=False)
    stem = os.path.splitext(data_name)[0].replace(' ', '_') if not use_example else 'example'
    st.download_button('Download results (.xlsx)', buf.getvalue(), file_name=f'{stem}_hydrochem_results.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.caption('Every derived variable, saturation index and flag for all samples (including any '
               'groups not plotted). The Settings sheet records every option used (units, below-LOR, '
               'missing values, groups, plot options, guidelines), with the group colours and markers '
               'and any replaced below-LOR values on their own sheets, so the run can be reproduced. '
               'With the uncertainty analysis on, three more sheets: the inputs used, the 5th and 95th '
               'percentiles beside every derived variable, and a per-sample summary (probability of the '
               'same water type, Chadha field, charge balance within 5 % and each SI above zero). '
               'The figure zip has every figure the app draws, for the plotted groups only and with the '
               'options chosen on each view, in three folders: classical (Piper, Schoeller, Stiff, Durov, '
               'Gibbs, Chadha, HFE-D), gallery, and custom plots (the custom plot and the guideline bar '
               'chart). File names start with the dataset name.')

    st.subheader('All figures')
    dpi = st.selectbox('Resolution', [300, 600], key=opt('opt_dpi', 300, [300, 600]))
    if vp.empty:
        st.info('No groups are ticked to plot, so there are no figures to build.')
    elif st.button('Build a zip of every figure'):
        # every figure the app draws, with the choices made on each view
        # (colour-by, Piper options, custom plot and guideline chart selections)
        ss = st.session_state
        by = lambda key: None if ss.get(key, 'Group (Label)') in ('Group (Label)', None) or \
            ss.get(key) not in vp.columns else ss.get(key)
        notes = []
        zbuf = io.BytesIO()
        with st.spinner('Drawing every figure ...'), zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
            def add(name, fig):
                if fig is None:
                    notes.append(f'{name}: not drawn (not enough data).')
                    return
                z.writestr(name, plots.fig_png(fig, dpi))
                plt.close(fig)

            add('classical/piper.png', plots.piper(vp, color_by=by('piper_color'),
                                         water_legend=ss.get('opt_piper_legend', False),
                                         show_types=ss.get('opt_piper_types', True),
                                         unc=mc, unc_mode=ss.get('opt_piper_unc', 'Clouds')))
            add('classical/schoeller.png', plots.schoeller(vp, unc=mc))
            # Stiff: every plotted sample, MAX_STIFF to a page
            samples = list(vp['Sample'])
            for k in range(0, len(samples), MAX_STIFF):
                add(f'classical/stiff_{k // MAX_STIFF + 1}.png',
                    plots.stiff(vp, samples[k:k + MAX_STIFF], letters=letters, letters_pos=letters_pos))
            for kind in plots.WQ_DIAGRAMS:
                fig, note = plots.wqchartpy_figure(vp, kind, letters=letters, letters_pos=letters_pos, unc=mc,
                                                   unc_mode=ss.get('opt_wq_unc', 'Clouds'))
                if note:
                    notes.append(f'{kind}: {note}')
                add(f'classical/{kind.lower()}.png', fig)
            for gname, gpanels in plots.GALLERY.items():
                gp = [p for p in gpanels if plots.available(vp, p)]
                for k in range(0, len(gp), PANELS_PER_PAGE):
                    f = plots.gallery_page(vp, gp[k:k + PANELS_PER_PAGE], log_conc=log_conc,
                                          color_by=by('gallery_color'), letters=letters, letters_pos=letters_pos,
                                          unc=mc)
                    fname = gname.lower().replace(' ', '_').replace('/', '-')
                    add(f'gallery/{fname}_{k // PANELS_PER_PAGE + 1}.png', f)
            # custom plot, as last set on its view
            nc = numeric_choices
            cx = ss.get('opt_custom_x', 'Cl (mmol/L)')
            cy = ss.get('opt_custom_y', 'Na (mmol/L)')
            if cx in nc and cy in nc:
                add('custom plots/custom_plot.png', plots.custom(
                    vp, cx, cy, xlog=ss.get('opt_custom_xlog', False), ylog=ss.get('opt_custom_ylog', False),
                    one_to_one=ss.get('opt_custom_one', False), zero_line=ss.get('opt_custom_zero', False),
                    annotate=ss.get('opt_custom_labels', False), color_by=by('custom_color'), unc=mc))
            # guideline bar chart, as last set on the Guidelines view
            rows = {lab: g for lab, _, g in guidelines.chart_rows(guide_region, guide_eco)}
            g_row = rows.get(ss.get('opt_gchart_rule'))
            if g_row is None:
                notes.append('Guideline bar chart: not included (open the Guidelines view and choose one).')
            elif g_row['var'] in vp.columns:
                mode_one = ss.get('opt_gchart_barmode') == 'One colour'
                add('custom plots/guideline_bar_chart.png', plots.guideline_bars(
                    vp, g_row['var'], guidelines.chart_lines(vp, g_row),
                    bar_colour=ss.get('opt_gchart_barcol', '#0089D0') if mode_one else None,
                    highlight=ss.get('opt_gchart_hlcol', '#000000') if ss.get('opt_gchart_hl') else None,
                    line_colour=ss.get('opt_gchart_linecol', '#222222'),
                    log=ss.get('opt_gchart_log', False), sort=ss.get('opt_gchart_sort', False), unc=mc))
            if notes:
                z.writestr('notes.txt', '\n'.join(notes) + '\n')
            figure_names = z.namelist()
        st.caption(f'{len([n for n in figure_names if n.endswith(".png")])} figures in the zip.')
        st.session_state['zip_contents'] = figure_names   # for the tests
        st.download_button('Download figures (.zip)', zbuf.getvalue(), file_name=f'{stem}_hydrochem_figures.zip',
                           mime='application/zip')
        if notes:
            st.caption('Some figures were left out or trimmed (also listed in notes.txt in the zip): '
                       + ' '.join(notes))
