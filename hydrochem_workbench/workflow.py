# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# hydrochem_workbench/workflow.py
# the steps of a run, in order, for use from a script and by the app:
#   process()               below-LOR values, cleaning, units, fills, PHREEQC
#   style_groups()          colour and marker for each Label
#   mark_charge_balance()   the charge balance check
# and for scripts only:
#   run()                   all of the above from a file, in one call
#   save_results()          the results workbook
#   save_figures()          the standard figures
#
# the app calls process(), style_groups() and mark_charge_balance(), so a script and
# the app give the same numbers

# %%
import os
import warnings

import pandas as pd

from hydrochem_workbench import chem, plots, uncertainty, quality, plots_quality

# %% CONFIG
PANELS_PER_PAGE = 6     # gallery panels on a page
MAX_STIFF = 24          # Stiff diagrams on a page

MARK, EXCLUDE, NO_ACTION = 'mark', 'exclude', 'none'
VALUE, INTERVAL = 'value', 'interval'


# %% PHREEQC
def get_phreeqc(database='phreeqc.dat'):
    # a PHREEQC instance with one of chem.DATABASES loaded
    from pathlib import Path
    import phreeqpython
    if database not in chem.DATABASES:
        raise ValueError(f'database must be one of {list(chem.DATABASES)}')
    own = Path(__file__).resolve().parent / 'data'
    pp = (phreeqpython.PhreeqPython(database=database, database_directory=own) if (own / database).exists()
          else phreeqpython.PhreeqPython(database=database))
    pp.hw_database = database
    pp.hw_elements = chem.database_elements(database)
    return pp


# %% the calculation
def process(raw, pp, unit='mg/L', n_as_n=False, p_as_p=False, dic_unit='mg/L as C', rules=None,
            lor=(0.5, True, True), ec_unit='uS/cm'):
    # below-LOR substitution, clean, convert, fill, run PHREEQC
    # raw: the table as read from the template (chem.read_upload)
    # rules: how to fill a missing pH, temperature and alkalinity, see fill_rules()
    # lor = (fraction of the LOR, replace "<" values, replace negative concentrations)
    # ec_unit: unit of the EC column ('uS/cm', 'mS/cm', 'dS/m' or 'mS/m'); it is converted to uS/cm here
    # returns dict with
    #   table     the results (None if the data could not be run, see problems)
    #   dropped   cells treated as missing
    #   errors    samples PHREEQC did not converge for
    #   problems  messages. Duplicate sample names are reported but still run
    #   replaced  the below-LOR values that were replaced
    #   base      what the uncertainty analysis needs (cleaned data, LORs, settings)
    rules = fill_rules() if rules is None else rules
    fraction, lor_text, lor_neg = lor
    raw, replaced = chem.below_lor(raw, fraction, text=lor_text, negative=lor_neg, rows=True)
    below = {(r.Row, r.Column): float(r.LOR) for r in replaced.itertuples()}
    replaced = replaced.drop(columns='Row')
    df, dropped, extra_cols, problems = chem.clean(raw)
    if any(not p.startswith('Duplicate') for p in problems):
        return dict(table=None, dropped=dropped, errors=None, problems=problems, replaced=replaced, base=None)
    ec_factor = chem.EC_UNITS[chem.ec_unit_key(ec_unit)]
    if ec_factor != 1.0:
        df = df.copy()
        df['EC'] = df['EC'].astype(float) * ec_factor
    settings = dict(unit=unit, n_as_n=n_as_n, p_as_p=p_as_p, dic_unit=dic_unit, rules=rules,
                    extra_cols=tuple(extra_cols))
    v, errors = chem.derive(df, pp, **settings)
    return dict(table=v, dropped=dropped, errors=errors, problems=problems, replaced=replaced,
                base=dict(df=df, below=below, settings=settings))


def fill_rules(ph='mean', temp='mean', alkalinity='mean'):
    # how to fill a missing pH, temperature and alkalinity
    # each is 'mean' (mean of the dataset) or a number (that fixed value: pH, deg C, mg/L as CaCO3).
    # alkalinity can also be 'balance' (from the charge balance)
    def rule(how, default, allow_balance=False):
        if isinstance(how, str):
            if how == 'mean' or (how == 'balance' and allow_balance):
                return (how, default)
            raise ValueError(f"'{how}' is not a fill rule. Use 'mean'" +
                             (", 'balance'" if allow_balance else '') + ' or a number.')
        return ('value', float(how))

    return (('pH', rule(ph, 7.0)), ('Temp', rule(temp, 25.0)), ('HCO3', rule(alkalinity, 100.0, True)))


# %% styles and the charge balance check
def style_groups(v, styles=None):
    # sets the Color and Marker of every sample from its Label
    # styles: {Label: (colour, marker)}. Default: the palette, or the Color/Marker columns of the file
    # returns (a copy of v with the styles set, the styles)
    styles = plots.group_styles(v) if styles is None else styles
    v = v.copy()
    v['Color'] = v['Label'].map(lambda lab: styles[lab][0])
    v['Marker'] = v['Label'].map(lambda lab: styles[lab][1])
    return v, styles


def mark_charge_balance(v, limit=5.0, action=MARK, basis=VALUE, mc=None, ring=None):
    # adds the charge balance check to v (in place) and says which samples are beyond the limit
    # limit: charge balance error in %, either side of zero
    # action: MARK (a ring on the plots), EXCLUDE (left off the plots and out of the means and
    #         analyses) or NO_ACTION
    # basis: VALUE (the calculated CBE) or INTERVAL (beyond the limit only if the whole 5th to 95th
    #        percentile interval from the uncertainty analysis mc is)
    # ring: colour of the ring. Default: one that is not a group colour
    # returns (beyond, exclude, legend text): two boolean Series on v's index
    use_mc = mc is not None and basis == INTERVAL
    v[chem.CB_COL] = chem.charge_balance_check(v, limit, mc['lo']['CBE (%)'] if use_mc else None,
                                               mc['hi']['CBE (%)'] if use_mc else None)
    beyond = v[chem.CB_COL] == chem.CB_BEYOND
    exclude = beyond & (action == EXCLUDE)
    text = f'Charge balance beyond {limit:g} %' + (', allowing for uncertainty' if use_mc else '')
    if ring is None:
        ring = plots.ring_default(list(pd.unique(v['Color'])) if 'Color' in v.columns else [])
    v[chem.CB_RING] = [ring if (b and action == MARK) else '' for b in beyond]
    v[chem.CB_LABEL] = text
    return beyond, exclude, text


# %% a run from a script
class Result:
    # what run() returns
    #   table      every sample: inputs, derived variables, saturation indices, flags
    #   plotted    the samples that go on the plots (those excluded for charge balance left out)
    #   styles     {Label: (colour, marker)}
    #   mc         the uncertainty analysis (None if not run): pass as unc=result.mc to the plots
    #   dropped, replaced, errors, problems   as from process()
    #   settings   every option of the run, as a table
    #   base       what the uncertainty analysis used
    #   ec_warning a message if the measured EC and the ions disagree by a unit-sized factor, otherwise ''
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f'<hydrochem_workbench Result: {len(self.table)} samples, {self.table["Label"].nunique()} group(s), '
                f'{len(self.plotted)} plotted, uncertainty {"on" if self.mc is not None else "off"}>')


def run(data, unit='mg/L', n_as_n=False, p_as_p=False, dic_unit='mg/L as C',
        lor_fraction=0.5, replace_text=True, replace_negative=True,
        missing_ph='mean', missing_temp='mean', missing_alkalinity='mean',
        uncertainty_n=0, uncertainty_table=None, seed=uncertainty.SEED,
        cb_limit=5.0, cb_action=MARK, cb_basis=VALUE, cb_ring=None,
        styles=None, phreeqc=None, database='phreeqc.dat', ec_unit='uS/cm'):
    # the whole calculation from a filled template
    # data: path of the .xlsx or .csv, or a table already read
    # unit: unit of every ion column ('mg/L', 'mmol/L' or 'meq/L')
    # n_as_n, p_as_p: NO3, NO2 and NH4 are entered as N, PO4 as P
    # dic_unit: 'mg/L as C' or 'mmol/L'
    # ec_unit: unit of the EC column: 'uS/cm', 'mS/cm' (or 'dS/m') or 'mS/m'. Results are in uS/cm
    # lor_fraction: below-LOR values are replaced with this fraction of the LOR.
    #   replace_text for "<0.05" style cells, replace_negative for negative concentrations
    # missing_ph, missing_temp, missing_alkalinity: see fill_rules()
    # uncertainty_n: realisations of the uncertainty analysis (0 = not run; the app uses 200)
    # uncertainty_table: the uncertainty inputs, default uncertainty.default_table()
    # cb_limit, cb_action, cb_basis, cb_ring: see mark_charge_balance()
    # styles: {Label: (colour, marker)}, default automatic
    # database: 'phreeqc.dat', 'wateq4f.dat' or 'pitzer.dat' (see chem.DATABASES)
    # phreeqc: a PHREEQC instance from get_phreeqc() to reuse (its database is then the one used)
    name = data if isinstance(data, (str, os.PathLike)) else 'table'
    if isinstance(data, pd.DataFrame):
        raw = data.astype(object)
    elif isinstance(data, (str, os.PathLike)):
        with open(data, 'rb') as f:
            raw = chem.read_upload(f)
    else:
        raw = chem.read_upload(data)
    if unit not in chem.UNITS:
        raise ValueError(f"unit must be one of {chem.UNITS}")
    pp = get_phreeqc(database) if phreeqc is None else phreeqc
    database = getattr(pp, 'hw_database', 'phreeqc.dat')
    rules = fill_rules(missing_ph, missing_temp, missing_alkalinity)
    ec_unit = chem.ec_unit_key(ec_unit)
    out = process(raw, pp, unit, n_as_n, p_as_p, dic_unit, rules, (lor_fraction, replace_text, replace_negative),
                  ec_unit)
    if out['table'] is None:
        raise ValueError('The data could not be run: ' + ' '.join(out['problems']))

    mc, table_used = None, None
    if uncertainty_n:
        table_used = uncertainty.mark_edited(uncertainty.check_table(
            uncertainty.default_table() if uncertainty_table is None else uncertainty_table))
        mc = uncertainty.run(out['base']['df'], out['base']['below'], table_used, out['base']['settings'], pp,
                             n=int(uncertainty_n), seed=seed)
    v, styles = style_groups(out['table'], styles)
    ec_warning = chem.ec_unit_check(v, ec_unit)
    if ec_warning:
        warnings.warn(ec_warning)
    beyond, exclude, _ = mark_charge_balance(v, cb_limit, cb_action, cb_basis, mc, cb_ring)

    from hydrochem_workbench import VERSION
    describe = {'mean': 'Mean of the dataset', 'balance': 'From charge balance'}
    fill = lambda how: describe.get(how, f'Fixed value ({how})') if isinstance(how, str) else f'Fixed value ({how:g})'
    settings = pd.DataFrame([
        ('Run', 'Version', VERSION),
        ('Run', 'Run at (UTC)', pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M')),
        ('Run', 'Dataset', os.path.basename(str(name))),
        ('Run', 'Samples', len(v)),
        ('Run', 'Database', database),
        ('Run', 'Not in this database (left out of the speciation)',
         ', '.join(chem.left_out_of_speciation(out['base']['df'], pp)) or 'None'),
        ('Units', 'Ion columns', unit),
        ('Units', 'NO3, NO2 and NH4 columns', 'As N' if n_as_n else 'As the ion'),
        ('Units', 'PO4 column', 'As P' if p_as_p else 'As PO4'),
        ('Units', 'DIC column', dic_unit),
        ('Units', 'EC column', chem.EC_LABELS[ec_unit] + ('' if ec_unit == 'uS/cm' else ', converted to µS/cm')),
        ('Units', 'EC against the ions', ec_warning or 'Consistent'),
        ('Below the LOR', 'Replace "<" values', 'Yes' if replace_text else 'No'),
        ('Below the LOR', 'Replace negative concentrations', 'Yes' if replace_negative else 'No'),
        ('Below the LOR', 'Replaced with', f'{lor_fraction:g} x LOR' if (replace_text or replace_negative)
         else 'Not replaced'),
        ('Missing values', 'Missing pH', fill(missing_ph)),
        ('Missing values', 'Missing temperature', fill(missing_temp)),
        ('Missing values', 'Missing alkalinity', fill(missing_alkalinity)),
        ('Charge balance', 'Limit (%)', cb_limit),
        ('Charge balance', 'Judged on', 'The uncertainty interval (5th to 95th percentile wholly beyond the limit)'
         if (mc is not None and cb_basis == INTERVAL) else 'The calculated value'),
        ('Charge balance', 'Samples beyond the limit', {MARK: 'Marked on the plots', EXCLUDE: 'Excluded from the '
                                                        'plots and analyses', NO_ACTION: 'No action'}[cb_action]),
        ('Charge balance', 'Beyond the limit', ', '.join(v.loc[beyond, 'Sample'].astype(str)) or 'None'),
        ('Charge balance', 'Not assessable (alkalinity filled, or no CBE)',
         ', '.join(v.loc[v[chem.CB_COL] == chem.CB_NA, 'Sample'].astype(str)) or 'None'),
        ('Uncertainty', 'Included', 'Yes' if mc is not None else 'No'),
        ('Uncertainty', 'Method', uncertainty.method_text(mc['n'], mc['seed']) if mc is not None else ''),
    ], columns=['Section', 'Setting', 'Value'])
    return Result(table=v, plotted=v[~exclude], styles=styles, mc=mc, dropped=out['dropped'],
                  replaced=out['replaced'], errors=out['errors'], problems=out['problems'], settings=settings,
                  base=out['base'], uncertainty_table=table_used, excluded=exclude, ec_warning=ec_warning)


# %% outputs
def save_results(result, path, guideline_sets=None, region='Tropical Australia', ecosystem='Lowland river',
                 stat_columns=None):
    # the results workbook: Results, Settings, Groups, Group means, Statistics and, where they apply, the replaced
    # and dropped cells, the guideline comparison and the uncertainty sheets
    # guideline_sets: names from guidelines.set_names(), or None for no comparison
    # stat_columns: columns for the Statistics sheet, default the field values and major ions
    v = result.table
    out_cols = [c for c in v.columns if not c.startswith('piper_')
                and c not in ('Color', 'Marker', chem.CB_RING, chem.CB_LABEL)]
    with pd.ExcelWriter(path, engine='openpyxl') as xl:
        v[out_cols].to_excel(xl, sheet_name='Results', index=False)
        result.settings.to_excel(xl, sheet_name='Settings', index=False)
        pd.DataFrame({'Label': list(result.styles), 'Colour': [c for c, _ in result.styles.values()],
                      'Marker': [m for _, m in result.styles.values()]}).to_excel(xl, sheet_name='Groups',
                                                                                  index=False)
        g_means, g_counts = chem.group_means(v[~result.excluded])
        g_means.to_excel(xl, sheet_name='Group means', index=False)
        g_counts.to_excel(xl, sheet_name='Group mean counts', index=False)
        chem.group_stats(v[~result.excluded], chem.stat_defaults(v) if stat_columns is None else
                         [c for c in stat_columns if c in v.columns]).to_excel(xl, sheet_name='Statistics',
                                                                              index=False)
        quality.irrigation_table(v).to_excel(xl, sheet_name='Irrigation', index=False)
        quality.scaling_table(v).to_excel(xl, sheet_name='Scaling', index=False)
        quality.qa_table(v).to_excel(xl, sheet_name='Analysis checks', index=False)
        quality.univariate_outliers(v[~result.excluded], chem.stat_defaults(v) if stat_columns is None else
                                    [c for c in stat_columns if c in v.columns]).to_excel(
            xl, sheet_name='Outliers', index=False)
        if len(result.replaced):
            quality.censored_stats(result.base['df'], result.base['below']).to_excel(
                xl, sheet_name='Below LOR statistics', index=False)
            result.replaced.astype({'Value': str}).to_excel(xl, sheet_name='Below LOR replaced', index=False)
        if len(result.dropped):
            result.dropped.astype(str).to_excel(xl, sheet_name='Cells treated as missing', index=False)
        if guideline_sets:
            from hydrochem_workbench import guidelines
            g_summary, g_flags = guidelines.assess(v, list(guideline_sets), region, ecosystem)
            g_summary.to_excel(xl, sheet_name='Guidelines', index=False)
            g_flags.to_excel(xl, sheet_name='Guidelines by sample', index=False)
            wqi = pd.concat([quality.ccme_wqi(v[~result.excluded], guidelines.regional_rows(region, ecosystem)
                                              if name == guidelines.REGIONAL_NAME else guidelines.SETS[name])
                            .assign(**{'Guideline set': name}) for name in guideline_sets])
            wqi.to_excel(xl, sheet_name='CCME WQI', index=False)
        if result.mc is not None:
            uncertainty.summary(v, result.mc).to_excel(xl, sheet_name='Uncertainty summary', index=False)
            uncertainty.interval_table(v, result.mc).to_excel(xl, sheet_name='Uncertainty intervals', index=False)
            result.uncertainty_table.to_excel(xl, sheet_name='Uncertainty inputs', index=False)
    return path


def save_figures(result, folder, dpi=300, fmt='png', log_conc=False, means=False, letters=None,
                 letters_pos=plots.PANEL_LABEL_POS[0]):
    # the standard figures, for the plotted samples: Piper, Schoeller, Stiff, Durov, Gibbs, Chadha,
    # HFE-D, every gallery page, the USSL and Wilcox irrigation diagrams and the analysis checks
    # fmt: any format matplotlib writes ('png', 'pdf', 'svg', 'tiff' ...)
    # returns (files written, notes on figures that were left out or trimmed)
    v, mc = result.plotted, result.mc
    os.makedirs(folder, exist_ok=True)
    written, notes = [], []

    def add(name, fig):
        if fig is None:
            notes.append(f'{name}: not drawn (not enough data).')
            return
        plots.add_badge(fig)
        path = os.path.join(folder, f'{name}.{fmt}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        written.append(path)

    if v.empty:
        return written, ['No samples to plot.']
    add('piper', plots.piper(v, unc=mc, means=means))
    add('schoeller', plots.schoeller(v, unc=mc))
    samples = list(v['Sample'])
    for k in range(0, len(samples), MAX_STIFF):
        add(f'stiff_{k // MAX_STIFF + 1}', plots.stiff(v, samples[k:k + MAX_STIFF], letters=letters,
                                                       letters_pos=letters_pos, scale_samples=samples))
    for kind in plots.WQ_DIAGRAMS:
        fig, note = plots.wqchartpy_figure(v, kind, letters=letters, letters_pos=letters_pos, unc=mc)
        if note:
            notes.append(f'{kind}: {note}')
        add(kind.lower(), fig)
    for gname, gpanels in plots.GALLERY.items():
        gp = [p for p in gpanels if plots.available(v, p)]
        gx = plots.shared_xlims(v, gp, log_conc, unc=mc, means=means)
        for k in range(0, len(gp), PANELS_PER_PAGE):
            fname = gname.lower().replace(' ', '_').replace('/', '-')
            add(f'gallery_{fname}_{k // PANELS_PER_PAGE + 1}',
                plots.gallery_page(v, gp[k:k + PANELS_PER_PAGE], log_conc=log_conc, xlims=gx, letters=letters,
                                   letters_pos=letters_pos, unc=mc, means=means))
    add('irrigation_ussl', plots_quality.ussl(v))
    add('irrigation_wilcox', plots_quality.wilcox(v))
    checks = plots_quality.qa_checks(v)
    if checks is not None:
        add('analysis_checks', checks)
    return written, notes
