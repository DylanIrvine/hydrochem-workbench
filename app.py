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
# calculations are in core/chem.py, figures in core/plots.py, the map in core/maps.py

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
from core import chem, guidelines, plots, uncertainty, usage, advanced, plots_advanced, maps

# %% CONFIG
APP_NAME = 'Hydrochem Workbench'
APP_URL = 'https://hydrochem-workbench.streamlit.app/'
APP_VERSION = '0.15.0'   # keep the same as VERSION in core/__init__.py

# after an update on Streamlit Cloud the server can keep old copies of the core
# modules in memory while running the new app.py. If the versions do not match,
# reload them
if getattr(core, 'VERSION', None) != APP_VERSION:
    import importlib
    importlib.reload(core)
    for _m in (chem, guidelines, plots, uncertainty, usage, advanced, plots_advanced, maps):
        importlib.reload(_m)

_record_use, _usage_status = usage.increment, usage.status
A, PA = advanced, plots_advanced

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
    st.write('Dylan J. Irvine, Charles Darwin University.')
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
    'reference lines, a builder for any plot the gallery does not have, and, if the samples have '
    'coordinates, a location map over satellite imagery. The database is '
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
        '- Your data are not sent to a third party. PHREEQC runs inside the app, not as an outside service.\n'
        '- Sample coordinates, site names or anything else in your spreadsheet are treated the '
        'same way: used for your plots, then gone. The results exist only in the files you download. '
        'The same goes for a towns file or polygon files uploaded for the map.\n'
        '- One exception, only if you open the Map view with satellite imagery as the background: the '
        'app asks an outside map server (Esri World Imagery) for the image tiles covering the map, so that '
        'server sees which area was requested, as a rectangle around your samples. It does not receive the sample '
        'locations themselves, their names or any chemistry. Choose the white background and no request '
        'is made.\n\n'
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
        f'Irvine, D.J. (2026). Hydrochem Workbench (version {APP_VERSION}) '
        '[Web application]. '
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
        '**Map**\n\n'
        'Satellite imagery: Esri World Imagery, credited below each map. Inset outlines: Natural Earth '
        '(naturalearthdata.com, public domain).\n\n'
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
        '**Further analyses**\n\n'
        'Appelo, C.A.J. and Postma, D. (2005). Geochemistry, groundwater and pollution, 2nd edition. '
        'A.A. Balkema, Leiden. (Mixing with conservative tracers and ion deviations; cation exchange.)\n\n'
        'Aitchison, J. (1986). The statistical analysis of compositional data. Chapman and Hall, London. '
        '(Centred log-ratio transform.)\n\n'
        'Guler, C., Thyne, G.D., McCray, J.E. and Turner, A.K. (2002). Evaluation of graphical and '
        'multivariate statistical methods for classification of water chemistry data. Hydrogeology '
        'Journal, 10, 455-474. https://doi.org/10.1007/s10040-002-0196-6\n\n'
        'Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal '
        'of Statistics, 6(2), 65-70.\n\n'
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
    plots.add_badge(fig)
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
    show_means = st.checkbox('Show group means as large markers', key=opt('opt_means', False),
                             help='Adds the mean of each group to the Piper, gallery and custom plots, as a '
                                  'larger version of the group marker. The means are listed on the Data tab.')
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

    st.header('7. Charge balance')
    CB_BASES = ['The calculated value', 'The uncertainty interval']
    CB_ACTIONS = ['Mark on the plots', 'Exclude from the plots and analyses', 'No action']
    cb_limit = st.number_input('Limit on the charge balance error (%)', min_value=0.0, max_value=50.0, step=1.0,
                               key=opt('opt_cb_limit', 5.0),
                               help='Either side of zero. Samples whose alkalinity was filled are not assessed: '
                                    'filling from the charge balance forces the error to zero, and a mean or '
                                    'fixed value was not measured.')
    cb_basis = st.selectbox('Judged on', CB_BASES, key=opt('opt_cb_basis', CB_BASES[0], CB_BASES),
                            disabled=not use_unc,
                            help='Needs the uncertainty analysis. With "The uncertainty interval" a sample is '
                                 'beyond the limit only if its whole 5th to 95th percentile interval of charge '
                                 'balance error is, that is, if measurement uncertainty cannot bring it within '
                                 'the limit. With a limit of 0 this flags every sample whose interval does not '
                                 'include zero.')
    if not use_unc:
        cb_basis = CB_BASES[0]
    cb_action = st.selectbox('Samples beyond the limit', CB_ACTIONS, key=opt('opt_cb_action', CB_ACTIONS[0], CB_ACTIONS),
                             help='Marked samples get a thick ring in another colour (set with the group '
                                  'colours) on the Piper diagram, the gallery, custom, mixing and PCA plots and '
                                  'the map. Excluded samples are left off every plot and out of the group means '
                                  'and the further analyses. Either way every sample stays in the tables and '
                                  'the guideline comparison, with its status in the "Charge balance check" '
                                  'column.')

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
if len(v) > A.LARGE_DATASET:
    st.warning(f'{len(v)} samples. Above {A.LARGE_DATASET} samples the figures can be slow to draw, crowded, '
               'or in the worst case fail to draw. Every figure is still available; the tables and the '
               'results download are not affected. Unticking groups in the Groups panel makes the plots lighter.')
n_filled = int((v['Filled'] != '').sum())
if n_filled:
    st.info(f'{n_filled} sample(s) had a missing pH, temperature or alkalinity filled. They are hollow on '
            'the plots where the fill matters. Remove those rows and re-upload if you would rather '
            'not show them.')
if errors is not None and len(errors):
    st.warning(f'PHREEQC did not converge for {len(errors)} sample(s); see the Data tab.')

# %% plotting and analysis: everything below works on the loaded dataset
st.divider()
st.header('Plotting and analysis')
st.write('Set the colour and marker of each group and choose which groups to plot. The plot options '
         '(log axes, panel letters, group means) are in the sidebar.')

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
    cb_col = '#FF00FF'
    if cb_action == CB_ACTIONS[0]:
        c1, c2 = st.columns([4, 4], vertical_alignment='center')
        c1.write('Ring around samples beyond the charge balance limit')
        cb_col = c2.color_picker('Ring colour', label_visibility='collapsed',
                                 key=opt('opt_cb_col', plots.ring_default([c for c, _ in group_style.values()])))
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

# %% charge balance check
cb_unc = mc is not None and cb_basis == CB_BASES[1]
v[chem.CB_COL] = chem.charge_balance_check(v, cb_limit, mc['lo']['CBE (%)'] if cb_unc else None,
                                           mc['hi']['CBE (%)'] if cb_unc else None)
cb_beyond = v[chem.CB_COL] == chem.CB_BEYOND
cb_exclude = cb_beyond & (cb_action == CB_ACTIONS[1])
cb_text = (f'Charge balance beyond {cb_limit:g} %' + (', allowing for uncertainty' if cb_unc else ''))
v[chem.CB_RING] = [cb_col if (b and cb_action == CB_ACTIONS[0]) else '' for b in cb_beyond]
v[chem.CB_LABEL] = cb_text
if cb_beyond.any():
    done = {CB_ACTIONS[0]: 'They are marked with a ring on the plots.',
            CB_ACTIONS[1]: 'They are left off every plot and out of the group means and the further analyses, '
                           'and stay in the tables.',
            CB_ACTIONS[2]: 'No action is taken (see "7. Charge balance" in the sidebar).'}[cb_action]
    names = list(v.loc[cb_beyond, 'Sample'].astype(str))
    st.info(f"{int(cb_beyond.sum())} sample(s) have a charge balance error beyond {cb_limit:g} %"
            + (' even allowing for measurement uncertainty' if cb_unc else '') + ': '
            + ', '.join(names[:8]) + (' ...' if len(names) > 8 else '') + '. ' + done)
# vp is what the plots see
vp = v[v['Label'].isin(shown) & ~cb_exclude]

numeric_choices = chem.plottable(v)
color_opts = ['Group (Label)'] + numeric_choices
st.subheader('Checks, plots and analyses')
st.caption('Choose a view. Each one uses the groups ticked above.')
# radio buttons and not st.tabs, as tabs redraw every figure on every click
VIEWS = ['Data and checks', 'Guidelines', 'Piper', 'Classical diagrams', 'Diagnostic gallery', 'Custom plot',
         'Map', 'Mixing', 'PCA', 'Clusters', 'Group comparison', 'Export']
view = st.radio('View', VIEWS, horizontal=True, label_visibility='collapsed', key=opt('opt_view', VIEWS[0], VIEWS))

# %% view: data
if view == 'Data and checks':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Samples', len(v))
    c2.metric('Groups', v['Label'].nunique())
    cbe_ok = (v[chem.CB_COL] == chem.CB_WITHIN).sum()
    c3.metric(f'Charge balance within {cb_limit:g} %',
              f'{int(cbe_ok)} of {int((v[chem.CB_COL] != chem.CB_NA).sum())}',
              help='Of the samples that can be assessed (a measured alkalinity). Set in "7. Charge balance" '
                   'in the sidebar.')
    c4.metric('Filled samples', n_filled)

    summary_cols = ['Sample', 'Label', 'Water type', 'Chadha field', 'CBE (%)', chem.CB_COL, 'TDS (mg/L)', 'pH',
                    'Alkalinity (mg/L as CaCO3)', 'HCO3 (mg/L)', 'CO3 (mg/L)', 'HCO3, CO3 from', 'log pCO2',
                    'Ionic strength (mol/kgw)'] + [c for c in v.columns if c.startswith('SI ') and v[c].notna().any()] + ['Filled']
    st.subheader('Summary')
    st.dataframe(v[summary_cols].round(3), width='stretch', hide_index=True)
    st.caption('HCO3 and CO3 are your values where you entered them. Where you entered only one, '
               'the other is the rest of the alkalinity. Where you entered neither, PHREEQC calculates '
               'both from the alkalinity and pH (each includes its ion pairs, so HCO3 + 2 CO3, in '
               'meq/L, adds up to the carbonate alkalinity). "HCO3, CO3 from" shows which. '
               'Saturation indices and pCO2 always come from the alkalinity (or DIC) and pH. '
               'CBE is signed: positive means excess cations. "Charge balance check" compares it with the '
               f'limit of {cb_limit:g} %' + (' using the 5th to 95th percentile interval from the uncertainty '
                                             'analysis' if cb_unc else '') +
               '; samples with a filled alkalinity are not assessable. TDS is the measured value where '
               'given, otherwise the sum of ions with bicarbonate x 0.4917.')

    st.subheader('Group means')
    g_means, g_counts = chem.group_means(v[~cb_exclude])
    st.dataframe(g_means.round(3), width='stretch', hide_index=True)
    st.caption('Arithmetic mean of each analyte (mg/L) and TDS for each group, all groups included. '
               + (f'{int(cb_exclude.sum())} sample(s) excluded for their charge balance are left out. '
                  if cb_exclude.any() else '') +
               'Values below the LOR count at their replaced value. Values that rest on a filled pH, '
               'temperature or alkalinity (HCO3, CO3, and TDS where it was calculated) are left out, as they '
               'are drawn hollow on the plots. Samples is the group size; the number of values behind each '
               'mean is on the Group mean counts sheet of the results download. With "Show group means" '
               'ticked, the large markers on the plots are the mean of whatever is on each axis, so on a '
               'ratio plot the marker is the mean of the ratios, not the ratio of the mean concentrations.')

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
PLOT_VIEWS = ('Piper', 'Classical diagrams', 'Diagnostic gallery', 'Custom plot', 'Mixing', 'PCA', 'Clusters',
              'Group comparison')
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
                      water_legend=p_legend, show_types=p_types, unc=mc, unc_mode=p_unc, means=show_means)
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
                                 color_by=None if g_color.startswith('Group') else g_color, unc=mc,
                                 means=show_means)
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
                       color_by=None if cc.startswith('Group') else cc, unc=mc, means=show_means)
    show(fig, 'custom_plot', 'dl_custom')


# %% further analyses: shared choices
# each analysis reads its settings from session_state, so the Export view can redo it with the
# same choices for the download
def mmol_choices(w):
    return [A.mmol_col(a) for a in chem.ANALYTES if A.mmol_col(a) in w.columns and w[A.mmol_col(a)].notna().any()]


def multi_defaults(w):
    # majors that were measured, without those mostly below the LOR
    share = A.lor_share(w, replaced)
    return [A.mmol_col(a) for a in A.MULTI_MAJORS
            if A.mmol_col(a) in mmol_choices(w) and share.get(a, 0) <= A.LOR_SHARE_WARN
            and (w[A.mmol_col(a)] > 0).sum() >= 3]


def multi_warnings(w, cols, n_used):
    share = A.lor_share(w, replaced)
    heavy = [A.analyte_of(c) for c in cols if share.get(A.analyte_of(c), 0) > A.LOR_SHARE_WARN]
    if heavy:
        st.warning('More than a quarter of the samples are below the LOR for ' + ', '.join(heavy) +
                   '. Every replaced value is the same number, which adds structure that is not in the water. '
                   'Consider leaving these out.')
    if n_used < 10:
        st.warning(f'Only {n_used} samples. With so few, the result can change a lot if one sample is added '
                   'or removed; treat it as a description of these samples, not a general finding.')
    elif n_used < 3 * len(cols):
        st.warning(f'{n_used} samples for {len(cols)} variables. A common rule of thumb is at least three '
                   'samples per variable; the components or clusters may not be stable.')


ION_LABEL = lambda c: A.analyte_of(c) or c
# coordinates and screen depth are not tracers
_loc = set(chem.location_columns(v.columns).values())
extra_numeric = [c for c in (base['settings']['extra_cols'] if base else [])
                 if c in numeric_choices and c not in _loc]
mix_options = A.member_options(v)
_groups = list(pd.unique(v['Label']))


def _em_default(k):
    return f'Group mean: {_groups[k]}' if len(_groups) > k else mix_options[min(k, len(mix_options) - 1)]


def compute_mixing():
    n_em = ss.get('opt_mix_n', 2)
    choices = [ss.get(f'opt_mix_em{k}', _em_default(k)) for k in range(n_em)]
    tracers = ss.get('opt_mix_tracers', ['Cl (mmol/L)'])
    ions = ss.get('opt_mix_ions', [])
    return A.run_mixing(v, vp, choices, tracers, ions, mc=mc)


def mixing_panel_list(res):
    t = res['tracers']
    out = [(t[0], t[1])] if len(t) >= 2 else []
    return out + [(t[0], c) for c in res['ions']]


def compute_pca():
    cols = ss.get('opt_pca_vars', multi_defaults(vp))
    if len(cols) < 2:
        raise ValueError('Choose at least two variables.')
    X, dropped_s = A.prepare(vp, cols)
    if len(X) < 3:
        raise ValueError('Fewer than three samples have every chosen variable.')
    return A.pca(X, ss.get('opt_pca_transform', A.TRANSFORMS[0])), X, dropped_s


def compute_clusters():
    cols = ss.get('opt_hca_vars', multi_defaults(vp))
    if len(cols) < 2:
        raise ValueError('Choose at least two variables.')
    X, dropped_s = A.prepare(vp, cols)
    k = int(ss.get('opt_hca_k', 3))
    if len(X) <= k:
        raise ValueError(f'{len(X)} samples have every chosen variable, too few for {k} clusters.')
    return A.clusters(X, ss.get('opt_hca_transform', A.TRANSFORMS[1]), k), X, dropped_s


CMP_DEFAULT = [f'{a} (mg/L)' for a in ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'Cl', 'SO4']]

# %% view: mixing
if view == 'Mixing':
    st.subheader('Mixing between end members')
    st.write('Each sample is treated as a mix of two or three end members (for example rainfall, river water '
             'and regional groundwater). The mixing fractions come from conservative tracers, ones not added '
             'or taken away along the flow path (usually Cl, Br or the water isotopes). Every other ion is then '
             'compared with what mixing alone would give. Measured minus mixing above zero means the ion was '
             'gained (dissolution, exchange, oxidation); below zero means it was lost (precipitation, '
             'exchange, uptake, reduction).')
    n_em = st.radio('Number of end members', [2, 3], horizontal=True, key=opt('opt_mix_n', 2, [2, 3]))
    cols_em = st.columns(n_em)
    for k in range(n_em):
        cols_em[k].selectbox(f'End member {k + 1} (EM{k + 1})', mix_options,
                             key=opt(f'opt_mix_em{k}', _em_default(k), mix_options))
    tr_opts = mmol_choices(v) + extra_numeric
    tr_default = ['Cl (mmol/L)'] if 'Cl (mmol/L)' in tr_opts else tr_opts[:1]
    if n_em == 3:
        tr_default += [c for c in ['Br (mmol/L)'] + extra_numeric if c in tr_opts and c not in tr_default][:1]
    st.multiselect('Conservative tracers', tr_opts, key=opt('opt_mix_tracers', tr_default, tr_opts),
                   help=f'{n_em} end members need at least {n_em - 1} tracer(s). With more, the fractions are a '
                        'least-squares fit, each tracer scaled by the spread of the end members so that units do '
                        'not matter.')
    ion_opts = [c for c in mmol_choices(v) if c not in ss.get('opt_mix_tracers', [])]
    ion_default = [A.mmol_col(a) for a in ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'SO4'] if A.mmol_col(a) in ion_opts]
    st.multiselect('Ions to compare with mixing', ion_opts, key=opt('opt_mix_ions', ion_default, ion_opts))
    c1, c2 = st.columns(2)
    mix_log = c1.checkbox('Log axes on the mixing plots', key=opt('opt_mix_log', False))
    dev_log = c2.checkbox('Symmetric log axis on the deviation plot', key=opt('opt_mix_symlog', False),
                          help='Linear close to zero and logarithmic beyond, for data that spans fresh and saline water.')
    try:
        mix = compute_mixing()
    except ValueError as err:
        st.error(str(err))
        mix = None
    if mix is not None:
        em = mix['members'].copy()
        em.insert(0, 'Chosen as', [mix['member_choices'][n] for n in em.index])
        st.dataframe(em.rename(columns=ION_LABEL).round(4), width='stretch')
        st.caption('End-member compositions (mmol/L, extra columns in their own units). A group mean is the mean '
                   'of that group\'s measured values.')
        if mix.get('mc_note'):
            st.info(mix['mc_note'])
        n_out = int(mix['outside'].sum())
        if n_out:
            st.info(f'{n_out} sample(s) have a mixing fraction below 0 or above 1: they cannot be made by mixing '
                    'these end members. Another source, an end member that is not quite right, or a tracer that '
                    'is not conservative are the usual reasons. They are flagged in the table.')
        pl = mixing_panel_list(mix)
        for k in range(0, len(pl), PANELS_PER_PAGE):
            show(PA.mixing_panels(vp, mix, pl[k:k + PANELS_PER_PAGE], log=mix_log, means=show_means,
                                  letters=letters, letters_pos=letters_pos),
                 f'mixing_{k // PANELS_PER_PAGE + 1}', f'dl_mix_{k}')
        st.caption('Stars are the end members. Samples on the dashed line (two end members) or inside the dashed '
                   'triangle (three) can be explained by mixing alone in that pair of variables; samples off it '
                   'need a process or another source.')
        fig = PA.deviation_plot(vp, mix, symlog=dev_log)
        if fig is not None:
            show(fig, 'mixing_deviations', 'dl_mix_dev')
        fig = PA.exchange_plot(vp, mix)
        if fig is not None:
            show(fig, 'mixing_cation_exchange', 'dl_mix_exch')
            st.caption('Cation exchange swaps Na + K for Ca + Mg charge for charge, so exchange alone puts samples '
                       'on the 1:-1 line. Upper left: the water gained Na and lost Ca + Mg, as when fresh water '
                       'flushes a formerly saline aquifer. Lower right: the reverse, as in salinisation.')
        st.subheader('Mixing results')
        st.dataframe(A.mixing_table(vp, mix).round(4), width='stretch', hide_index=True)
        st.caption('Fractions are f EM1, f EM2 (and f EM3), summing to 1. Deviations (d) are measured minus '
                   'conservative mixing, in mmol/L. Values that rest on a filled pH, temperature or alkalinity are '
                   'left out. ' +
                   ('With the uncertainty analysis on, each fraction and deviation has a 90 % interval from the '
                    'realisations (end members drawn again in each one), and "differs from 0" marks deviations '
                    'whose interval does not include zero. Extra columns such as isotopes have no uncertainty in '
                    'the realisations. ' if 'dev_lo' in mix else
                    'Tick "Include uncertainty analysis" in the sidebar to see which deviations are larger than '
                    'the measurement uncertainty. ') +
                   'The mixing method follows Appelo and Postma (2005), Geochemistry, Groundwater and Pollution.')

# %% view: PCA
if view == 'PCA':
    st.subheader('Principal component analysis')
    st.write('PCA finds the few directions (components) that carry most of the variation among the samples. '
             'The biplot shows the samples on two components and each variable as an arrow: samples lie toward '
             'the variables they are rich in, and arrows pointing the same way vary together.')
    var_opts = mmol_choices(vp)
    c1, c2 = st.columns([3, 2])
    pca_vars = c1.multiselect('Variables', var_opts, key=opt('opt_pca_vars', multi_defaults(vp), var_opts),
                              format_func=ION_LABEL)
    c2.selectbox('Transform', A.TRANSFORMS, key=opt('opt_pca_transform', A.TRANSFORMS[0], A.TRANSFORMS),
                 help='Centred log-ratio (clr): the standard for compositions, where every concentration is part '
                      'of a whole (Aitchison 1986). log10, then standardised: each variable on a log scale with '
                      'mean 0 and SD 1, so every variable counts equally.')
    try:
        pca, X_pca, dropped_s = compute_pca()
    except ValueError as err:
        st.error(str(err))
        pca = None
    if pca is not None:
        multi_warnings(vp, pca_vars, len(X_pca))
        if dropped_s:
            st.info(f'{len(dropped_s)} sample(s) left out for a missing, zero or filled value in a chosen '
                    'variable: ' + ', '.join(dropped_s[:12]) + (' ...' if len(dropped_s) > 12 else ''))
        pcs = list(pca['explained'].index)
        c1, c2 = st.columns(2)
        a = c1.selectbox('x axis', pcs, key=opt('opt_pca_x', pcs[0], pcs))
        b = c2.selectbox('y axis', pcs, key=opt('opt_pca_y', pcs[1], pcs))
        show(PA.pca_biplot(vp, pca, a, b, means=show_means), 'pca_biplot', 'dl_pca')
        ld = pca['loadings'].rename(index=ION_LABEL)
        ld.loc['Variance explained (%)'] = 100 * pca['explained']
        st.dataframe(ld.round(3), width='stretch')
        st.caption('Loadings: how much each variable contributes to each component (the arrows on the biplot). '
                   'The bar chart shows the share of the variance on each component. With clr the last component '
                   'always carries no variance (the clr values of a sample sum to zero). PCA describes these '
                   'samples; it does not test anything. Values that rest on a filled alkalinity, pH or '
                   'temperature are left out.')

# %% view: clusters
if view == 'Clusters':
    st.subheader('Hierarchical cluster analysis')
    st.write('Groups the samples by chemical similarity alone, without your labels: Ward linkage on the Euclidean '
             'distance between samples, the approach recommended for water chemistry by Guler et al. (2002, '
             'Hydrogeology Journal 10, 455-474). The table compares the clusters with your groups.')
    var_opts = mmol_choices(vp)
    c1, c2, c3 = st.columns([3, 2, 1])
    hca_vars = c1.multiselect('Variables', var_opts, key=opt('opt_hca_vars', multi_defaults(vp), var_opts),
                              format_func=ION_LABEL)
    c2.selectbox('Transform', A.TRANSFORMS, key=opt('opt_hca_transform', A.TRANSFORMS[1], A.TRANSFORMS))
    c3.number_input('Clusters', min_value=2, max_value=10, step=1, key=opt('opt_hca_k', 3))
    try:
        hca, X_hca, dropped_s = compute_clusters()
    except ValueError as err:
        st.error(str(err))
        hca = None
    if hca is not None:
        multi_warnings(vp, hca_vars, len(X_hca))
        if dropped_s:
            st.info(f'{len(dropped_s)} sample(s) left out for a missing, zero or filled value in a chosen '
                    'variable: ' + ', '.join(dropped_s[:12]) + (' ...' if len(dropped_s) > 12 else ''))
        show(PA.dendrogram_fig(vp, hca), 'cluster_dendrogram', 'dl_dend')
        st.caption('Branches are coloured by cluster, and the dashed line is where the tree is cut. Sample names '
                   'are in the colour of their group, so a cluster that mixes colours mixes your groups. A long '
                   'vertical step just above the cut means well-separated clusters; short steps mean the number '
                   'of clusters is a matter of choice.')
        show(PA.cluster_piper(vp, hca['labels'], means=show_means), 'cluster_piper', 'dl_cpiper')
        st.markdown('**Clusters against your groups** (number of samples)')
        st.dataframe(A.crosstab(vp, hca['labels']), width='stretch')
        mg = [c.replace('(mmol/L)', '(mg/L)') for c in hca_vars]
        st.markdown('**Cluster means** (mg/L)')
        st.dataframe(A.cluster_means(vp, hca['labels'], mg).rename(columns=lambda c: c.replace(' (mg/L)', '')).round(2),
                     width='stretch')
        st.caption('Clustering always returns clusters, whether or not the water falls into distinct types; '
                   'check them against the dendrogram, the Piper and what you know of the site.')

# %% view: group comparison
if view == 'Group comparison':
    st.subheader('Group comparison')
    st.write('Box plots of each variable by group, with a Kruskal-Wallis test of whether the groups differ, and '
             'Mann-Whitney U tests between each pair of groups. Both are rank-based, so they suit skewed '
             'concentrations and small groups better than ANOVA and t-tests.')
    cmp_opts = numeric_choices
    c1, c2 = st.columns([4, 1])
    cmp_vars = c1.multiselect('Variables', cmp_opts,
                              key=opt('opt_cmp_vars', [c for c in CMP_DEFAULT if c in cmp_opts], cmp_opts))
    cmp_log = c2.checkbox('Log axes', key=opt('opt_cmp_log', False))
    if vp['Label'].nunique() < 2:
        st.info('Tick at least two groups to compare.')
    elif cmp_vars:
        kw, pairs = A.group_tests(vp, cmp_vars)
        for k in range(0, len(cmp_vars), PANELS_PER_PAGE):
            show(PA.box_page(vp, cmp_vars[k:k + PANELS_PER_PAGE], kw, log=cmp_log, letters=letters,
                             letters_pos=letters_pos), f'group_boxplots_{k // PANELS_PER_PAGE + 1}', f'dl_box_{k}')
        st.caption('Boxes span the 25th to 75th percentile with the median as a line; whiskers reach the furthest '
                   'sample within 1.5 times the box height. Every sample is drawn over its box.')
        st.markdown('**Kruskal-Wallis tests**')
        st.dataframe(kw.round(4), width='stretch', hide_index=True)
        if len(pairs):
            st.markdown('**Pairs of groups (Mann-Whitney U)**')
            st.dataframe(pairs.round(4), width='stretch', hide_index=True)
        st.caption(f'Groups with fewer than {A.MIN_PER_GROUP} values of a variable are left out of its tests. The '
                   'pairwise p values are Holm-adjusted within each variable for the number of pairs, not across '
                   'variables: with many variables, some small p values are expected by chance. Values that rest '
                   'on a filled pH, temperature or alkalinity are left out.')
        if st.checkbox('Correlation heat map of these variables', key=opt('opt_cmp_corr', True)):
            r, n_pairs = A.spearman(vp, cmp_vars)
            fig = PA.corr_heatmap(r, n_pairs)
            if fig is not None:
                show(fig, 'spearman_correlations', 'dl_corr')
                st.caption('Spearman rank correlations across all plotted samples, each from the samples with both '
                           'values (grey: fewer than 4). Correlation across groups often reflects the groups '
                           'themselves (e.g. fresh against saline) rather than a process within any one of them.')

# %% view: map
# every map setting is read from session_state, so the Export view can draw the same map
MAP_DEFAULTS = {
    'opt_map_base': maps.IMAGERY, 'opt_map_size': 30, 'opt_map_depth': False, 'opt_map_reverse': False,
    'opt_map_labels': False, 'opt_map_margin': 15, 'opt_map_manual': False,
    'opt_map_scale': 'Bottom right', 'opt_map_north': 'Top right', 'opt_map_inset': 'Auto',
    'opt_map_inset_factor': 12,
    'opt_map_legend_depths': '', 'opt_map_edge_width': 0.5,
    # marker edge, 0 (black) to 100 (white): one setting for each background
    'opt_map_edge_img': 100, 'opt_map_edge_plain': 0,
}
INSET_CHOICES = ['Auto'] + maps.CORNERS + ['Off']
# up to three polygon or line layers (e.g. a boundary, rivers, roads), each with its own style
LAYER_COLOURS = ['#FCB711', '#0089D0', '#CC004C']
for _k in range(maps.MAX_LAYERS):
    MAP_DEFAULTS.update({f'opt_map_poly{_k}_edge': LAYER_COLOURS[_k], f'opt_map_poly{_k}_face': LAYER_COLOURS[_k],
                         f'opt_map_poly{_k}_alpha': 0.25, f'opt_map_poly{_k}_width': 1.2,
                         f'opt_map_poly{_k}_label': f'Layer {_k + 1}'})


def map_layers():
    # the loaded layers with their styles, in order
    ss = st.session_state
    out = []
    for k in range(maps.MAX_LAYERS):
        if ss.get(f'map_poly_{k}'):
            out.append(dict(gdf=ss[f'map_poly_{k}'][1], file=ss[f'map_poly_{k}'][0],
                            edge=mget(f'opt_map_poly{k}_edge'), face=mget(f'opt_map_poly{k}_face'),
                            alpha=mget(f'opt_map_poly{k}_alpha'), width=mget(f'opt_map_poly{k}_width'),
                            label=mget(f'opt_map_poly{k}_label') or f'Layer {k + 1}'))
    return out


def mget(key):
    return st.session_state.get(key, MAP_DEFAULTS[key])


def mopt(key, options=None):
    return opt(key, MAP_DEFAULTS[key], options)


def map_edge_key():
    return 'opt_map_edge_img' if mget('opt_map_base') == maps.IMAGERY else 'opt_map_edge_plain'


def map_points():
    # the samples of the plotted groups that have coordinates
    pts, notes = maps.locations(v)
    return pts[pts['Label'].isin(shown)], notes


def map_extent(pts):
    ss = st.session_state
    if mget('opt_map_manual') and all(k in ss for k in ('opt_map_w', 'opt_map_e', 'opt_map_s', 'opt_map_n')):
        w, e, s, n = ss['opt_map_w'], ss['opt_map_e'], ss['opt_map_s'], ss['opt_map_n']
        if e > w and n > s:
            return maps.fit_extent((w, e, s, n))
    return maps.fit_extent(maps.data_extent(pts, mget('opt_map_margin') / 100))


def map_zoom(extent):
    # (zoom in use, default, lowest, highest, tiles at each zoom, key of the slider)
    default, lo, hi, counts = maps.zoom_choices(extent)
    key = f'opt_map_zoom_{default}_{hi}'
    zoom = st.session_state.get(key, default)
    return (zoom if lo <= zoom <= hi else default), default, lo, hi, counts, key


def map_basemap(extent, zoom):
    # imagery for the map, kept for the session until the extent or zoom changes
    # returns (basemap or None, message if it could not be fetched)
    ss = st.session_state
    key = (tuple(round(float(x), 5) for x in extent), zoom)
    kept = ss.get('map_tiles')
    if kept is not None and kept[0] == key:
        return kept[1], None
    if ss.get('map_tiles_failed', (None, ''))[0] == key:
        return None, ss['map_tiles_failed'][1]
    try:
        with st.spinner('Fetching the imagery ...'):
            basemap = maps.fetch_basemap(extent, zoom)
    except Exception as exc:
        msg = f'{type(exc).__name__}: {exc}'[:300]
        ss['map_tiles_failed'] = (key, msg)
        return None, msg
    ss['map_tiles'] = (key, basemap)
    ss.pop('map_tiles_failed', None)
    return basemap, None


def build_map():
    # returns (figure or None, notes, credit line)
    ss = st.session_state
    pts, notes = map_points()
    if pts.empty:
        return None, notes, ''
    extent = map_extent(pts)
    basemap, credit = None, ''
    if mget('opt_map_base') == maps.IMAGERY:
        basemap, failed = map_basemap(extent, map_zoom(extent)[0])
        if basemap is None:
            notes = notes + ['The imagery could not be fetched, so the map has a white background '
                             f'({failed}).']
        else:
            credit = maps.attribution(maps.IMAGERY)
    towns = ss['map_towns'][1] if ss.get('map_towns') else None
    poly = map_layers() or None
    inset = None
    corner = mget('opt_map_inset')
    if corner != 'Off':
        if corner == 'Auto':
            corner = maps.emptiest_corner(pts, extent, taken=(mget('opt_map_scale'), mget('opt_map_north')),
                                          towns=towns)
        inset = dict(corner=corner, factor=mget('opt_map_inset_factor'),
                     outline=ss['map_outline'][1] if ss.get('map_outline') else None)
    # if the imagery could not be fetched the map is white, so the edge setting for white is used
    edge_key = 'opt_map_edge_plain' if basemap is None else 'opt_map_edge_img'
    fig = maps.sample_map(pts, {lab: group_style[lab] for lab in shown}, extent, basemap=basemap,
                          size=mget('opt_map_size'), by_depth=mget('opt_map_depth'),
                          reverse=mget('opt_map_reverse'), labels=mget('opt_map_labels'), towns=towns, poly=poly,
                          inset=inset, scale_corner=mget('opt_map_scale'), north_corner=mget('opt_map_north'),
                          credit=credit, legend_depths=maps.parse_depths(mget('opt_map_legend_depths'))[0] or None,
                          edge=f'{mget(edge_key) / 100:.2f}', edge_width=mget('opt_map_edge_width'))
    return fig, notes, credit


def layer_upload(slot, label, types, reader, help=None):
    # one uploaded map layer kept in session_state[slot] as (description, data). A file with no
    # coordinate reference system waits in slot + '_pending' until the user gives an EPSG code
    ss = st.session_state
    ss.setdefault(slot + '_n', 0)
    if ss.get(slot):
        st.markdown(f'**Loaded:** {ss[slot][0]}')
        if st.button('Remove', key=slot + '_remove'):
            ss.pop(slot)
            st.rerun()
        return
    pending = ss.get(slot + '_pending')
    if pending:
        st.warning(pending[1])
        epsg = st.number_input('EPSG code of the file (e.g. 28353 for GDA94 / MGA zone 53, 7853 for GDA2020 / '
                               'MGA zone 53, 4283 for GDA94 latitude and longitude)', min_value=1024,
                               max_value=99999, value=4326, step=1, key=slot + '_epsg')
        c1, c2 = st.columns(2)
        if c1.button('Use this EPSG code', key=slot + '_use'):
            try:
                ss[slot] = tuple(reversed(reader(pending[0], epsg=int(epsg))))
            except maps.MapError as exc:
                st.error(str(exc))
            else:
                ss.pop(slot + '_pending')
                st.rerun()
        if c2.button('Cancel', key=slot + '_cancel'):
            ss.pop(slot + '_pending')
            st.rerun()
        return
    multi = len(types) > 2
    up = st.file_uploader(label, type=types, accept_multiple_files=multi, key=f'{slot}_up_{ss[slot + "_n"]}',
                          help=help)
    files = up if multi else ([up] if up is not None else [])
    if files:
        kept = [maps.Upload(f.name, f.getvalue()) for f in files]
        try:
            out = reader(kept) if multi else reader(io.BytesIO(kept[0].data), kept[0].name)
        except maps.NeedsCRS as exc:
            ss[slot + '_pending'] = (kept, str(exc))
        except maps.MapError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f'Could not read that file: {exc}')
            return
        else:
            ss[slot] = tuple(reversed(out)) if multi else out
        ss[slot + '_n'] += 1
        st.rerun()


def _read_towns(buf, name):
    buf.name = name
    towns = maps.read_towns(buf)
    return f'{name} ({len(towns)} town(s))', towns


if view == 'Map':
    st.subheader('Sample map')
    try:
        import geopandas   # noqa: F401
        import contextily  # noqa: F401
        map_ready = True
    except ImportError:
        map_ready = False
        st.error('The map needs the geopandas and contextily packages, which are not installed here. '
                 'Run "pip install -r requirements.txt".')
    pts, map_notes = map_points() if map_ready else (pd.DataFrame(), [])
    if map_ready and pts.empty:
        for note in map_notes:
            st.info(note)
        if not map_notes or 'no Lat and Lon' not in map_notes[0]:
            st.info('None of the plotted groups has a sample with coordinates.')
    elif map_ready:
        st.write('Samples with a latitude and longitude, drawn with the colour and marker of their group. '
                 'North is up. Open the panels below for the extent, towns, polygons or lines, and the layout.')
        c1, c2 = st.columns(2)
        base = c1.selectbox('Background', maps.BASEMAPS, key=mopt('opt_map_base', maps.BASEMAPS))
        c2.slider('Marker size', 5, 150, key=mopt('opt_map_size'), step=5,
                  help='Marker area in points squared, as in matplotlib. With "Size by screen depth" this is the '
                       'middle of the range.')
        has_depth = bool(pts['depth'].notna().any())
        c1, c2, c3 = st.columns(3)
        by_depth = c1.checkbox('Size by screen depth', key=mopt('opt_map_depth'), disabled=not has_depth,
                               help='Needs the Screen_depth column (m below surface). Samples with no depth '
                                    'keep the chosen marker size and are listed in the legend as "No depth".')
        c2.checkbox('Shallow = large', key=mopt('opt_map_reverse'), disabled=not (by_depth and has_depth),
                    help='By default the deepest screen has the largest marker. Tick to turn that around.')
        c3.checkbox('Label the samples', key=mopt('opt_map_labels'))
        if by_depth and has_depth:
            d_text = st.text_input(f'Depths in the legend (m): up to {maps.MAX_LEGEND_DEPTHS}, separated by commas',
                                   key=mopt('opt_map_legend_depths'), placeholder='e.g. 10, 50, 100, 200',
                                   help='Leave blank for the shallowest and deepest screen with round values '
                                        f"between them (here {pts['depth'].min():g} to {pts['depth'].max():g} m). "
                                        'A depth outside that range widens the size scale to include it.')
            d_msg = maps.parse_depths(d_text)[1]
            if d_msg:
                st.warning(d_msg)
        c1, c2 = st.columns(2)
        c1.slider('Marker edge: black (0) to white (100)', 0, 100, key=opt(map_edge_key(), MAP_DEFAULTS[map_edge_key()]),
                  step=5, help='Shades of grey in between. Kept separately for the imagery and the white '
                               'background. Legend markers keep a dark edge so they show on the page.')
        c2.slider('Marker edge thickness (points)', 0.0, 2.0, key=mopt('opt_map_edge_width'), step=0.25,
                  help='0 draws the markers with no edge.')

        with st.expander('**Extent and imagery detail**'):
            auto = maps.data_extent(pts, mget('opt_map_margin') / 100)
            manual = st.checkbox('Set the extent by hand', key=mopt('opt_map_manual'))
            if not manual:
                st.slider('Margin around the samples (% of their spread)', 0, 200, key=mopt('opt_map_margin'),
                          step=5)
            else:
                c1, c2, c3, c4 = st.columns(4)
                for col, key, lab, val in ((c1, 'opt_map_w', 'West', auto[0]), (c2, 'opt_map_e', 'East', auto[1]),
                                           (c3, 'opt_map_s', 'South', auto[2]), (c4, 'opt_map_n', 'North', auto[3])):
                    col.number_input(lab, min_value=-180.0, max_value=180.0, step=0.05, format='%.4f',
                                     key=opt(key, round(float(val), 4)))
                ss = st.session_state
                if not (ss['opt_map_e'] > ss['opt_map_w'] and ss['opt_map_n'] > ss['opt_map_s']):
                    st.warning('East has to be greater than West, and North greater than South. Using the '
                               'extent of the samples instead.')
            extent = map_extent(pts)
            st.caption(f'Map extent: {extent[0]:.4f} to {extent[1]:.4f} E, {extent[2]:.4f} to {extent[3]:.4f} N '
                       f'(about {maps.extent_width_km(extent):.3g} km wide). The extent is widened where needed '
                       'so the map is not a thin strip.')
            if base == maps.IMAGERY:
                zoom, z_default, z_lo, z_hi, counts, z_key = map_zoom(extent)
                if z_hi > z_lo:
                    zoom = st.slider('Imagery zoom level', z_lo, z_hi, key=opt(z_key, z_default),
                                     help='Each step up doubles the detail and needs about four times as many '
                                          f'image tiles. The starting value ({z_default}) suits this extent. '
                                          f'The app does not go above {maps.MAX_TILES} tiles.')
                st.caption(f'Zoom {zoom}: {counts[zoom]} tile(s) to download.')
                if counts[zoom] > maps.WARN_TILES:
                    st.warning(f'{counts[zoom]} tiles. High zoom levels take a long time to download and draw, '
                               'and the map is redrawn each time a setting changes. Lower the zoom while '
                               'setting the map up, and raise it at the end.')
                if st.session_state.get('map_tiles_failed') and st.button('Try the imagery again'):
                    st.session_state.pop('map_tiles_failed')
                    st.rerun()

        towns_kept = st.session_state.get('map_towns')
        with st.expander('**Towns**' + (f' ({len(towns_kept[1])} loaded)' if towns_kept else '')):
            st.write('A csv with the columns Name, Lat and Lon (decimal degrees). Towns inside the map extent '
                     'are drawn as black squares with their name.')
            st.download_button('Download the towns template', maps.towns_template_bytes(),
                               file_name='hydrochem_workbench_towns_template.csv', mime='text/csv')
            layer_upload('map_towns', 'Upload the towns file', ['csv', 'xlsx'], _read_towns)

        n_layers = len(map_layers())
        with st.expander('**Polygons and lines**' + (f' ({n_layers} loaded)' if n_layers else '')):
            st.write(f'Up to {maps.MAX_LAYERS} layers of polygons or lines, for example a region boundary, rivers '
                     'and roads. Each is a shapefile as a .zip or as its separate files (.shp, .shx, .dbf and '
                     f'.prj together), a GeoPackage, GeoJSON or KML, up to {maps.MAX_UPLOAD_MB} MB, and is '
                     'converted to latitude and longitude from whatever coordinate reference system the file '
                     'states. Layer 1 is drawn first, with the others on top. Lines take the edge colour.')
            loaded = [bool(st.session_state.get(f'map_poly_{k}')) for k in range(maps.MAX_LAYERS)]
            for k in range(maps.MAX_LAYERS):
                # the loaded layers, and one empty slot after them
                if not (loaded[k] or all(loaded[:k])):
                    continue
                st.markdown(f'**Layer {k + 1}**')
                layer_upload(f'map_poly_{k}', f'Upload the file(s) for layer {k + 1}', maps.POLY_TYPES,
                             maps.read_vector)
                if loaded[k]:
                    c1, c2, c3 = st.columns(3)
                    c1.color_picker('Edge colour', key=mopt(f'opt_map_poly{k}_edge'))
                    c2.color_picker('Fill colour', key=mopt(f'opt_map_poly{k}_face'))
                    c3.slider('Fill opacity (alpha)', 0.0, 1.0, key=mopt(f'opt_map_poly{k}_alpha'), step=0.05,
                              help='0 is no fill (outline only), 1 is solid. Not used for lines.')
                    c1, c2 = st.columns(2)
                    c1.slider('Edge width', 0.2, 4.0, key=mopt(f'opt_map_poly{k}_width'), step=0.2)
                    c2.text_input('Name in the legend', key=mopt(f'opt_map_poly{k}_label'))

        with st.expander('**Scale bar, north arrow and locator inset**'):
            c1, c2, c3 = st.columns(3)
            c1.selectbox('Scale bar', maps.CORNERS, key=mopt('opt_map_scale', maps.CORNERS))
            c2.selectbox('North arrow', maps.CORNERS, key=mopt('opt_map_north', maps.CORNERS))
            inset_at = c3.selectbox('Locator inset', INSET_CHOICES, key=mopt('opt_map_inset', INSET_CHOICES),
                                    help='A small map showing where the main map is. Auto puts it in the '
                                         'corner with the fewest samples.')
            if inset_at != 'Off':
                if not st.session_state.get('map_outline'):
                    st.slider('Inset width (times the width of the map)', 3, 100, key=mopt('opt_map_inset_factor'),
                              help='The inset shows country and state outlines from Natural Earth (1:50m).')
                st.write('To use your own outline in the inset instead (a country, state or catchment), '
                         'upload it here. The inset then covers that outline.')
                layer_upload('map_outline', 'Upload an outline for the inset', maps.POLY_TYPES, maps.read_vector)

        fig, map_notes, credit = build_map()
        for note in map_notes:
            st.info(note)
        if fig is not None:
            show(fig, 'sample_map', 'dl_map')
            caption = 'The scale bar is exact at the latitude of the middle of the map.'
            if credit:
                caption += (' The imagery credit is in a strip below the map, so it can be cropped off if the '
                            f'credit is given in the figure caption instead: "{credit}". The imagery shows the '
                            'surface on the date it was captured, which differs from place to place.')
            st.caption(caption)

# %% view: export
if view == 'Export':
    st.subheader('Results table')
    out_cols = [c for c in v.columns if not c.startswith('piper_')
                and c not in ('Color', 'Marker', chem.CB_RING, chem.CB_LABEL)]
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
        ('Charge balance', 'Limit (%)', cb_limit),
        ('Charge balance', 'Judged on', (cb_basis + ' (5th to 95th percentile wholly beyond the limit)')
         if cb_unc else CB_BASES[0]),
        ('Charge balance', 'Samples beyond the limit', cb_action),
        ('Charge balance', 'Beyond the limit', ', '.join(v.loc[cb_beyond, 'Sample'].astype(str)) or 'None'),
        ('Charge balance', 'Not assessable (alkalinity filled, or no CBE)',
         ', '.join(v.loc[v[chem.CB_COL] == chem.CB_NA, 'Sample'].astype(str)) or 'None'),
        ('Charge balance', 'Ring colour', cb_col if cb_action == CB_ACTIONS[0] else ''),
        ('Groups', 'Plotted', ', '.join(shown) if shown else 'None'),
        ('Groups', 'Not plotted', ', '.join(l for l in labels if l not in shown) or 'None'),
        ('Groups', 'Colours and markers', 'See the Groups sheet'),
        ('Plots', 'Log axes on concentration plots', 'Yes' if log_conc else 'No'),
        ('Plots', 'Group means as large markers', 'Yes' if show_means else 'No'),
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
    # the map, if its view was opened
    if 'opt_map_base' in ss:
        m_pts, _ = map_points()
        rows += [('Map', 'Samples with coordinates (plotted groups)', len(m_pts))]
        if len(m_pts):
            m_ext = map_extent(m_pts)
            rows += [
                ('Map', 'Extent (W, E, S, N; decimal degrees)', ', '.join(f'{x:.4f}' for x in m_ext)),
                ('Map', 'Extent set by', 'Hand' if mget('opt_map_manual') else
                 f"Samples, with a {mget('opt_map_margin')} % margin"),
                ('Map', 'Background', mget('opt_map_base')),
                ('Map', 'Imagery zoom level', map_zoom(m_ext)[0] if mget('opt_map_base') == maps.IMAGERY else ''),
                ('Map', 'Marker size', mget('opt_map_size')),
                ('Map', 'Size by screen depth', ('Yes, shallow = large' if mget('opt_map_reverse') else
                                                 'Yes, deep = large') if mget('opt_map_depth') else 'No'),
                ('Map', 'Depths in the legend (m)', (mget('opt_map_legend_depths') or 'Automatic')
                 if mget('opt_map_depth') else ''),
                ('Map', 'Marker edge', f"{mget(map_edge_key())} on a scale of 0 (black) to 100 (white), "
                                       f"{mget('opt_map_edge_width'):g} pt"),
                ('Map', 'Sample labels', 'Yes' if mget('opt_map_labels') else 'No'),
                ('Map', 'Towns', ss['map_towns'][0] if ss.get('map_towns') else 'None'),
                ('Map', 'Polygon and line layers', len(map_layers()) or 'None'),
                *[('Map', f"Layer: {m['label']}", f"{m['file']}; edge {m['edge']}, fill {m['face']}, "
                                                  f"alpha {m['alpha']:g}, width {m['width']:g}")
                  for m in map_layers()],
                ('Map', 'Scale bar, north arrow', f"{mget('opt_map_scale')}, {mget('opt_map_north')}"),
                ('Map', 'Locator inset', mget('opt_map_inset') + (
                    '' if mget('opt_map_inset') == 'Off' else
                    f"; outline: {ss['map_outline'][0]}" if ss.get('map_outline') else
                    f"; Natural Earth outlines, {mget('opt_map_inset_factor')} x the map width")),
            ]
    else:
        rows.append(('Map', 'Included', 'No (view not opened)'))
    # the further analyses, each only if its view was opened (so its choices were made)
    ran = {'Mixing': 'opt_mix_tracers' in ss, 'PCA': 'opt_pca_vars' in ss, 'Clusters': 'opt_hca_vars' in ss,
           'Group comparison': 'opt_cmp_vars' in ss}
    lab = lambda cols: ', '.join(ION_LABEL(c) for c in cols)
    if ran['Mixing']:
        n_em = ss.get('opt_mix_n', 2)
        rows += [('Mixing', 'End members', '; '.join(f"EM{k + 1} = {ss.get(f'opt_mix_em{k}', _em_default(k))}"
                                                      for k in range(n_em))),
                 ('Mixing', 'Tracers', lab(ss.get('opt_mix_tracers', []))),
                 ('Mixing', 'Ions compared', lab(ss.get('opt_mix_ions', [])))]
    if ran['PCA']:
        rows += [('PCA', 'Variables', lab(ss.get('opt_pca_vars', []))),
                 ('PCA', 'Transform', ss.get('opt_pca_transform', A.TRANSFORMS[0]))]
    if ran['Clusters']:
        rows += [('Clusters', 'Variables', lab(ss.get('opt_hca_vars', []))),
                 ('Clusters', 'Transform', ss.get('opt_hca_transform', A.TRANSFORMS[1])),
                 ('Clusters', 'Method', 'Ward linkage, Euclidean distance'),
                 ('Clusters', 'Number of clusters', ss.get('opt_hca_k', 3))]
    if ran['Group comparison']:
        rows += [('Group comparison', 'Variables', ', '.join(ss.get('opt_cmp_vars', []))),
                 ('Group comparison', 'Tests', 'Kruskal-Wallis; pairwise Mann-Whitney U, Holm-adjusted within '
                                               'each variable')]
    for name, done in ran.items():
        if not done:
            rows.append((name, 'Included', 'No (view not opened)'))
    settings = pd.DataFrame(rows, columns=['Section', 'Setting', 'Value'])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xl:
        v[out_cols].to_excel(xl, sheet_name='Results', index=False)
        settings.to_excel(xl, sheet_name='Settings', index=False)
        pd.DataFrame({'Label': labels, 'Plotted': [lab in shown for lab in labels],
                      'Colour': [group_style[lab][0] for lab in labels],
                      'Marker': [group_style[lab][1] for lab in labels]}).to_excel(
            xl, sheet_name='Groups', index=False)
        g_means, g_counts = chem.group_means(v[~cb_exclude])
        g_means.to_excel(xl, sheet_name='Group means', index=False)
        g_counts.to_excel(xl, sheet_name='Group mean counts', index=False)
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
        if not vp.empty:
            if ran['Mixing']:
                try:
                    mix = compute_mixing()
                    A.mixing_table(vp, mix).to_excel(xl, sheet_name='Mixing', index=False)
                    em = mix['members'].copy()
                    em.insert(0, 'Chosen as', [mix['member_choices'][n] for n in em.index])
                    em.to_excel(xl, sheet_name='Mixing end members')
                except ValueError:
                    pass
            if ran['PCA']:
                try:
                    pca, X_pca, _ = compute_pca()
                    sc = pca['scores'].copy()
                    sc.insert(0, 'Label', vp.loc[sc.index, 'Label'])
                    sc.insert(0, 'Sample', vp.loc[sc.index, 'Sample'])
                    sc.to_excel(xl, sheet_name='PCA scores', index=False)
                    ld = pca['loadings'].copy()
                    ld.loc['Variance explained (%)'] = 100 * pca['explained']
                    ld.to_excel(xl, sheet_name='PCA loadings')
                except ValueError:
                    pass
            if ran['Clusters']:
                try:
                    hca, _, _ = compute_clusters()
                    pd.DataFrame({'Sample': vp.loc[hca['labels'].index, 'Sample'],
                                  'Label': vp.loc[hca['labels'].index, 'Label'],
                                  'Cluster': hca['labels']}).to_excel(xl, sheet_name='Clusters', index=False)
                    A.crosstab(vp, hca['labels']).to_excel(xl, sheet_name='Clusters vs groups')
                except ValueError:
                    pass
            if ran['Group comparison'] and ss.get('opt_cmp_vars') and vp['Label'].nunique() >= 2:
                kw, pairs = A.group_tests(vp, ss['opt_cmp_vars'])
                kw.to_excel(xl, sheet_name='Group tests', index=False)
                pairs.to_excel(xl, sheet_name='Group tests pairwise', index=False)
    stem = os.path.splitext(data_name)[0].replace(' ', '_') if not use_example else 'example'
    st.download_button('Download results (.xlsx)', buf.getvalue(), file_name=f'{stem}_hydrochem_results.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.caption('Every derived variable, saturation index and flag for all samples (including any '
               'groups not plotted). The Settings sheet records every option used (units, below-LOR, '
               'missing values, groups, plot options, guidelines, the map), with the group colours and markers '
               'and any replaced below-LOR values on their own sheets, so the run can be reproduced. '
               'The Group means sheet has the mean of each analyte for each group, and Group mean counts '
               'the number of values behind each mean. '
               'With the uncertainty analysis on, three more sheets: the inputs used, the 5th and 95th '
               'percentiles beside every derived variable, and a per-sample summary (probability of the '
               'same water type, Chadha field, charge balance within 5 % and each SI above zero). '
               'Mixing, PCA, Clusters and Group comparison add their own sheets once their view has been '
               'opened, with the choices made there. '
               'The figure zip has every figure the app draws, for the plotted groups only and with the '
               'options chosen on each view, in folders: classical (Piper, Schoeller, Stiff, Durov, '
               'Gibbs, Chadha, HFE-D), gallery, custom plots (the custom plot and the guideline bar '
               'chart), map (if the Map view has been opened) and further analyses (from the views that '
               'have been opened). File names start with '
               'the dataset name.')

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
                                         unc=mc, unc_mode=ss.get('opt_piper_unc', 'Clouds'), means=show_means))
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
                                          unc=mc, means=show_means)
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
                    annotate=ss.get('opt_custom_labels', False), color_by=by('custom_color'), unc=mc,
                    means=show_means))
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
            # the map, as last set on its view
            if 'opt_map_base' in ss:
                try:
                    m_fig, m_notes, _ = build_map()
                except Exception as exc:
                    m_fig, m_notes = None, [f'{type(exc).__name__}: {exc}']
                if m_fig is not None:
                    add('map/sample_map.png', m_fig)
                notes.extend(f'Map: {note}' for note in m_notes)
            # the further analyses, as last set on their views
            fa = 'further analyses/'
            if ran['Mixing']:
                try:
                    mix = compute_mixing()
                    pl = mixing_panel_list(mix)
                    for k in range(0, len(pl), PANELS_PER_PAGE):
                        add(f'{fa}mixing_{k // PANELS_PER_PAGE + 1}.png', PA.mixing_panels(
                            vp, mix, pl[k:k + PANELS_PER_PAGE], log=ss.get('opt_mix_log', False), means=show_means,
                            letters=letters, letters_pos=letters_pos))
                    add(f'{fa}mixing_deviations.png', PA.deviation_plot(vp, mix, symlog=ss.get('opt_mix_symlog', False)))
                    add(f'{fa}mixing_cation_exchange.png', PA.exchange_plot(vp, mix))
                except ValueError as err:
                    notes.append(f'Mixing: not drawn ({err})')
            if ran['PCA']:
                try:
                    pca, _, _ = compute_pca()
                    pcs = list(pca['explained'].index)
                    a = ss.get('opt_pca_x', 'PC1')
                    b = ss.get('opt_pca_y', 'PC2')
                    add(f'{fa}pca_biplot.png', PA.pca_biplot(vp, pca, a if a in pcs else pcs[0],
                                                             b if b in pcs else pcs[1], means=show_means))
                except ValueError as err:
                    notes.append(f'PCA: not drawn ({err})')
            if ran['Clusters']:
                try:
                    hca, _, _ = compute_clusters()
                    add(f'{fa}cluster_dendrogram.png', PA.dendrogram_fig(vp, hca))
                    add(f'{fa}cluster_piper.png', PA.cluster_piper(vp, hca['labels'], means=show_means))
                except ValueError as err:
                    notes.append(f'Clusters: not drawn ({err})')
            cmp_vars = ss.get('opt_cmp_vars', [])
            if ran['Group comparison'] and cmp_vars and vp['Label'].nunique() >= 2:
                kw, _ = A.group_tests(vp, cmp_vars)
                for k in range(0, len(cmp_vars), PANELS_PER_PAGE):
                    add(f'{fa}group_boxplots_{k // PANELS_PER_PAGE + 1}.png', PA.box_page(
                        vp, cmp_vars[k:k + PANELS_PER_PAGE], kw, log=ss.get('opt_cmp_log', False),
                        letters=letters, letters_pos=letters_pos))
                if ss.get('opt_cmp_corr', True):
                    add(f'{fa}spearman_correlations.png', PA.corr_heatmap(*A.spearman(vp, cmp_vars)))
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
