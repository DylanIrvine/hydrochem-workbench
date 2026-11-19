# tests for core/chem.py
import numpy as np
import pandas as pd
import pytest

from core import chem


def one_sample(**kw):
    row = {c: np.nan for c in chem.TEMPLATE_COLS}
    row.update({'Sample': 'S1', 'Label': 'A', 'Color': '', 'Marker': ''})
    row.update(kw)
    return pd.DataFrame([row]).astype(object)


def run(raw, unit='mg/L', rules=None, **kw):
    import phreeqpython
    rules = rules or {'pH': ('mean', 7), 'Temp': ('mean', 25), 'HCO3': ('mean', 100)}
    df, dropped, extra, problems = chem.clean(raw)
    mmol, source = chem.to_mmol(df, unit, **kw)
    df, mmol, filled = chem.fill_missing(df, mmol, rules)
    ph, err, mmol = chem.run_phreeqc(df, mmol, phreeqpython.PhreeqPython(database='phreeqc.dat'))
    return chem.build_table(df, mmol, filled, ph, extra, source), dropped


def test_units_agree():
    # 40.078 mg/L Ca = 1 mmol/L = 2 meq/L
    a, _ = run(one_sample(pH=7, Temp=25, Ca=40.078, Mg=0, Na=0, K=0, HCO3=122.034, Cl=0, SO4=0), 'mg/L')
    b, _ = run(one_sample(pH=7, Temp=25, Ca=1.0, Mg=0, Na=0, K=0, HCO3=2.0, Cl=0, SO4=0), 'mmol/L')
    c, _ = run(one_sample(pH=7, Temp=25, Ca=2.0, Mg=0, Na=0, K=0, HCO3=2.0, Cl=0, SO4=0), 'meq/L')
    for t in (a, b, c):
        assert t.at[0, 'Ca (mmol/L)'] == pytest.approx(1.0, rel=1e-6)
        assert t.at[0, 'Ca (meq/L)'] == pytest.approx(2.0, rel=1e-6)
        assert t.at[0, 'CBE (%)'] == pytest.approx(0.0, abs=0.05)
    assert a.at[0, 'SI Calcite'] == pytest.approx(b.at[0, 'SI Calcite'], abs=1e-4)


def test_alkalinity_columns_and_nitrate_as_n():
    # 100 mg/L as CaCO3 = 2 meq/L, entered as total or as bicarbonate + carbonate
    base = dict(pH=7.5, Temp=25, Ca=40, Mg=1, Na=1, K=1, Cl=1, SO4=1)
    a, _ = run(one_sample(Total_Alk_CaCO3=100, NO3=10, NH4=1, PO4=1, **base), 'mg/L', n_as_n=True, p_as_p=True)
    b, _ = run(one_sample(Bicarb_Alk_CaCO3=80, Carb_Alk_CaCO3=20, **base), 'mg/L')
    for t in (a, b):
        assert t.at[0, 'Alkalinity (meq/L)'] == pytest.approx(1.998, rel=1e-3)
    assert a.at[0, 'Alkalinity from'] == 'Total alkalinity'
    assert b.at[0, 'Alkalinity from'] == 'Bicarbonate + carbonate alkalinity'
    assert a.at[0, 'NO3 (mg/L)'] == pytest.approx(44.27, rel=1e-3)
    assert a.at[0, 'NO3-N (mg/L)'] == pytest.approx(10.0, rel=1e-6)
    assert a.at[0, 'NH4 (mg/L)'] == pytest.approx(1.288, rel=1e-3)
    assert a.at[0, 'PO4 (mg/L)'] == pytest.approx(3.066, rel=1e-3)
    # the alkalinity columns are mg/L as CaCO3 even when the ions are in meq/L
    c, _ = run(one_sample(pH=7.5, Temp=25, Ca=2, Mg=0.1, Na=0.1, K=0.1, Cl=0.1, SO4=0.1,
                          Total_Alk_CaCO3=100), 'meq/L')
    assert c.at[0, 'Alkalinity (meq/L)'] == pytest.approx(1.998, rel=1e-3)


def test_hco3_and_co3_come_from_phreeqc():
    # same alkalinity, different pH: the split moves but HCO3 + 2 CO3 stays put
    base = dict(Temp=25, Ca=20, Mg=5, Na=60, K=2, Cl=30, SO4=10, Total_Alk_CaCO3=150)
    lo, _ = run(one_sample(pH=7.0, **base))
    hi, _ = run(one_sample(pH=9.5, **base))
    assert lo.at[0, 'CO3 (meq/L)'] < 0.01 * lo.at[0, 'HCO3 (meq/L)']
    assert hi.at[0, 'CO3 (meq/L)'] > 0.2 * hi.at[0, 'HCO3 (meq/L)']
    for t in (lo, hi):
        total = t.at[0, 'HCO3 (meq/L)'] + t.at[0, 'CO3 (meq/L)']
        assert total == pytest.approx(t.at[0, 'Alkalinity (meq/L)'], rel=0.03)
    assert 'HCO3' in chem.TEMPLATE_COLS and 'CO3' in chem.TEMPLATE_COLS
    assert set(lo['HCO3, CO3 from']) == {'PHREEQC'}


def test_text_cells_are_dropped_and_listed():
    raw = one_sample(pH=7, Temp=25, Ca=40, Mg=5, Na=10, K=1, HCO3=150, Cl=10, SO4=5, Br='<0.01', F='bd')
    t, dropped = run(raw)
    assert set(dropped['Column']) == {'Br', 'F'}
    assert np.isnan(t.at[0, 'Br (mmol/L)'])


def test_fill_rules_flag_the_right_samples():
    raw = pd.concat([one_sample(Sample='A', pH=7.0, Temp=20, Ca=40, Mg=5, Na=10, K=1, HCO3=122.034, Cl=10, SO4=5),
                     one_sample(Sample='B', pH=8.0, Temp=30, Ca=40, Mg=5, Na=10, K=1, HCO3=61.017, Cl=10, SO4=5),
                     one_sample(Sample='C', Ca=40, Mg=5, Na=10, K=1, Cl=10, SO4=5)], ignore_index=True)
    t, _ = run(raw)
    assert t.at[2, 'pH'] == pytest.approx(7.5)
    assert t.at[2, 'Temp'] == pytest.approx(25.0)
    assert t.at[2, 'Alkalinity (meq/L)'] == pytest.approx(1.5, rel=1e-4)
    assert t.at[2, 'Filled'] == 'pH, Temp, Alkalinity'
    assert t.at[0, 'Filled'] == ''
    # hollow markers
    assert not chem.hollow_mask(t, ['Na (mmol/L)', 'Cl (mmol/L)'])[2]
    assert chem.hollow_mask(t, ['SI Calcite', 'Cl (mmol/L)'])[2]
    assert not chem.hollow_mask(t, ['SI Calcite'])[0]

    t2, _ = run(raw, rules={'pH': ('value', 6.5), 'Temp': ('value', 22), 'HCO3': ('balance', 0)})
    assert t2.at[2, 'pH'] == 6.5
    assert t2.at[2, 'CBE (%)'] == pytest.approx(0.0, abs=0.5)


def test_piper_corners_and_water_types():
    def pos(**meq):
        base = dict(Ca=0, Mg=0, Na=0, K=0, HCO3=0, CO3=0, Cl=0, SO4=0)
        base.update(meq)
        return chem.piper_coords(pd.DataFrame([base], dtype=float)).iloc[0]
    h = 0.5 * np.tan(np.pi / 3)
    p = pos(Ca=1, HCO3=1)   # left corner
    assert (p.cat_x, p.cat_y) == pytest.approx((0, 0))
    assert (p.d_x, p.d_y) == pytest.approx((0.6, h + 0.1 * np.tan(np.pi / 3)))
    p = pos(Na=1, Cl=1)   # right corner
    assert p.d_x == pytest.approx(1.6)


def test_water_type_follows_the_shaded_fields():
    assert chem.water_type(1.0, 0.0) == 'Mg-HCO3'
    assert chem.water_type(0.0, 1.0) == 'Na-Cl'
    assert chem.water_type(1.0, 1.0) == 'Ca-Cl'
    assert chem.water_type(0.0, 0.0) == 'Na-HCO3'
    assert chem.water_type(0.5, 0.5) == 'Mixed'
    assert chem.water_type(0.9, 0.3) == 'Mg-HCO3'
    assert chem.water_type(0.6, 0.7) == 'Mixed'


def test_phreeqc_recovers_calcite_equilibrium():
    # water equilibrated with calcite in PHREEQC should come back with SI = 0
    import phreeqpython
    pp = phreeqpython.PhreeqPython(database='phreeqc.dat')
    s = pp.add_solution({'units': 'mmol/l', 'pH': 7.0, 'temp': 25, 'Na': 1, 'Cl': 1})
    s.equalize(['Calcite', 'CO2(g)'], [0.0, -2.0])
    ca = s.total('Ca', units='mmol')
    alk = 2 * ca
    raw = one_sample(pH=s.pH, Temp=25, Ca=ca, Mg=0, Na=1, K=0, HCO3=alk, Cl=1, SO4=0)
    t, _ = run(raw, 'mmol/L')
    assert t.at[0, 'SI Calcite'] == pytest.approx(0.0, abs=0.02)
    assert t.at[0, 'log pCO2'] == pytest.approx(-2.0, abs=0.02)
    assert np.isnan(t.at[0, 'SI Gypsum'])   # no SO4


def test_template_round_trip():
    import io
    raw = chem.read_upload(io.BytesIO(chem.template_bytes()))
    df, dropped, extra, problems = chem.clean(raw)
    assert problems == [] and len(dropped) == 0 and len(df) == 3


def test_extra_numeric_column_is_plottable():
    raw = one_sample(pH=7, Temp=25, Ca=40, Mg=5, Na=10, K=1, HCO3=150, Cl=10, SO4=5)
    raw['d18O'] = -6.2
    t, _ = run(raw)
    assert 'd18O' in chem.plottable(t)


def test_dic_gives_the_same_answer_as_alkalinity():
    # speciate with alkalinity, read off the DIC, feed that back in as DIC
    base = dict(pH=7.4, Temp=25, Ca=40, Mg=10, Na=20, K=2, Cl=25, SO4=15)
    a, _ = run(one_sample(Total_Alk_CaCO3=150, **base))
    dic_mgC = a.at[0, 'DIC (mmol/L)'] * 12.011
    b, _ = run(one_sample(DIC=dic_mgC, **base))
    c, _ = run(one_sample(DIC=a.at[0, 'DIC (mmol/L)'], **base), dic_unit='mmol/L')
    for t in (b, c):
        assert t.at[0, 'Alkalinity from'] == 'DIC'
        assert t.at[0, 'HCO3 (mmol/L)'] == pytest.approx(a.at[0, 'HCO3 (mmol/L)'], rel=2e-3)
        assert t.at[0, 'SI Calcite'] == pytest.approx(a.at[0, 'SI Calcite'], abs=0.01)
        assert t.at[0, 'Filled'] == ''


def test_guidelines():
    from core import guidelines
    raw = pd.concat([
        one_sample(Sample='ok', pH=7.2, Temp=25, Ca=20, Mg=5, Na=30, K=2, Cl=40, SO4=20, Total_Alk_CaCO3=60,
                   NO3=5, F=0.3, Mn=0.01, Fe=0.05),
        one_sample(Sample='bad', pH=9.0, Temp=25, Ca=150, Mg=60, Na=400, K=10, Cl=700, SO4=300,
                   Total_Alk_CaCO3=200, NO3=80, F=2.0, Mn=0.2, Fe=1.0),
        # soft water: 2 mg/L NO3-N is over the 1.1 soft-water value but not the hard-water one
        one_sample(Sample='soft', pH=7.0, Temp=25, Ca=4, Mg=1, Na=5, K=1, Cl=5, SO4=2, Total_Alk_CaCO3=10,
                   NO3=8.85),
    ], ignore_index=True)
    t, _ = run(raw)
    summary, flags = guidelines.assess(t, guidelines.set_names(), 'Tropical Australia', 'Lowland river')
    by = lambda name, s: summary[(summary['Parameter'] == name) & summary['Guideline set'].str.startswith(s)].iloc[0]
    assert by('Nitrate (as NO3)', 'Drinking water: health')['Which samples'] == 'bad'
    assert by('Fluoride', 'Drinking water: health')['Which samples'] == 'bad'
    assert by('Manganese', 'Drinking water: health')['Guideline value'] == '0.1'
    assert by('pH', 'Drinking water: aesthetic')['Which samples'] == 'bad'
    assert by('Total dissolved solids', 'Drinking water: aesthetic')['Which samples'] == 'bad'
    assert by('Nitrate (as N), soft water', 'Aquatic')['Which samples'] == 'soft'
    assert by('Nitrate (as N), hard water', 'Aquatic')['Samples assessed'] == 1
    assert by('Nitrate (as N), hard water', 'Aquatic')['Outside guideline'] == 0
    assert flags.set_index('Sample').at['ok', 'Count'] <= 2      # only the tropical nutrient values
    # every region and ecosystem type runs, and the values differ between regions
    for region, ecos in guidelines.REGIONS.items():
        for eco in ecos:
            guidelines.assess(t, [guidelines.REGIONAL_NAME], region, eco)
    se, _ = guidelines.assess(t, [guidelines.REGIONAL_NAME], 'South-east Australia', 'Lowland river')
    assert se.set_index('Parameter').at['EC (uS/cm)', 'Guideline value'] == '125 to 2200'
    assert se.set_index('Parameter').at['Filterable reactive P', 'Guideline value'] == '0.02'
    assert 'Fluoride' in flags.set_index('Sample').at['bad', 'Outside guideline for']


def test_below_lor_text_and_negative():
    raw = one_sample(pH=7, Temp=25, Ca=40, Mg=5, Na=10, K='<0.5', Cl=10, SO4=-2, Total_Alk_CaCO3=100,
                     F='< 0.1 mg/L', Br='bd')
    out, rep = chem.below_lor(raw, 0.5)
    assert out.at[0, 'K'] == pytest.approx(0.25)
    assert out.at[0, 'SO4'] == pytest.approx(1.0)
    assert out.at[0, 'F'] == pytest.approx(0.05)
    assert out.at[0, 'Br'] == 'bd'   # no number, left for clean() to drop
    assert set(rep['Column']) == {'K', 'SO4', 'F'}
    df, dropped, extra, problems = chem.clean(out)
    assert list(dropped['Column']) == ['Br']


def test_below_lor_switches_and_fraction():
    raw = one_sample(pH=7, K='<0.5', SO4=-2)
    out, rep = chem.below_lor(raw, 0.1, text=False, negative=True)
    assert out.at[0, 'K'] == '<0.5' and out.at[0, 'SO4'] == pytest.approx(0.2)
    out, rep = chem.below_lor(raw, 1.0, text=True, negative=False)
    assert out.at[0, 'K'] == pytest.approx(0.5) and out.at[0, 'SO4'] == -2


def test_below_lor_leaves_ph_and_isotopes_alone():
    raw = one_sample(pH='-1', Ca=40)
    raw['d18O'] = -6.2
    out, rep = chem.below_lor(raw, 0.5)
    assert out.at[0, 'd18O'] == -6.2 and out.at[0, 'pH'] == '-1' and rep.empty


def test_entered_hco3_and_co3_are_used():
    # lab alkalinity drives PHREEQC, but the entered ions are reported
    base = dict(Temp=25, Ca=20, Mg=5, Na=60, K=2, Cl=30, SO4=10, Total_Alk_CaCO3=150, pH=9.5)
    t, _ = run(one_sample(HCO3=150.0, CO3=12.0, **base))
    assert t.at[0, 'HCO3 (mg/L)'] == pytest.approx(150.0)
    assert t.at[0, 'CO3 (mg/L)'] == pytest.approx(12.0)
    assert t.at[0, 'Alkalinity (mg/L as CaCO3)'] == pytest.approx(150.0)
    assert t.at[0, 'HCO3, CO3 from'] == 'Entered'
    calc, _ = run(one_sample(**base))
    assert t.at[0, 'SI Calcite'] == pytest.approx(calc.at[0, 'SI Calcite'], abs=1e-6)


def test_one_entered_ion_takes_the_rest_of_the_alkalinity():
    base = dict(Temp=25, Ca=20, Mg=5, Na=60, K=2, Cl=30, SO4=10, Total_Alk_CaCO3=150, pH=9.5)
    alk = 150 / chem.CACO3_PER_MEQ
    t, _ = run(one_sample(HCO3=140.0, **base))
    assert t.at[0, 'HCO3 (mg/L)'] == pytest.approx(140.0)
    assert t.at[0, 'HCO3 (meq/L)'] + t.at[0, 'CO3 (meq/L)'] == pytest.approx(alk, rel=1e-6)
    t, _ = run(one_sample(CO3=6.0, **base))
    assert t.at[0, 'CO3 (mg/L)'] == pytest.approx(6.0)
    assert t.at[0, 'HCO3 (meq/L)'] + t.at[0, 'CO3 (meq/L)'] == pytest.approx(alk, rel=1e-6)
    # HCO3 alone, no alkalinity columns: becomes the alkalinity, CO3 is zero
    t, _ = run(one_sample(pH=7, Temp=25, Ca=40.078, Mg=0, Na=0, K=0, HCO3=122.034, Cl=0, SO4=0))
    assert t.at[0, 'CO3 (meq/L)'] == pytest.approx(0.0, abs=1e-9)
    assert t.at[0, 'CBE (%)'] == pytest.approx(0.0, abs=0.01)


def test_fao_classes_by_hand():
    from core import guidelines
    # SAR, EC (uS/cm) -> expected infiltration class (FAO 29 Table 1)
    cases = [(2, 800, 0), (2, 500, 1), (2, 150, 2), (5, 1300, 0), (5, 250, 2), (15, 2000, 1),
             (15, 1000, 2), (30, 6000, 0), (50, 1000, np.nan), (np.nan, 1000, np.nan)]
    v = pd.DataFrame({'SAR': [c[0] for c in cases], 'EC': [c[1] for c in cases]})
    row = [g for g in guidelines.SETS['Irrigation: degree of restriction on use (FAO)'] if g['kind'] == 'sar_ec'][0]
    got = guidelines.classify(v, row)
    for (sar, ec, want), g in zip(cases, got):
        assert (np.isnan(want) and np.isnan(g)) or g == want, (sar, ec, g, want)
    # graded single variable: chloride, surface irrigation, meq/L
    cl = [g for g in guidelines.SETS['Irrigation: degree of restriction on use (FAO)']
          if g['name'].startswith('Chloride toxicity, surface')][0]
    got = guidelines.classify(pd.DataFrame({'Cl (meq/L)': [3.9, 4.1, 10.5]}), cl)
    assert list(got) == [0, 1, 2]


def test_who_and_fao_sets_in_assess():
    from core import guidelines
    raw = pd.concat([
        one_sample(Sample='fresh', pH=7.2, Temp=25, Ca=20, Mg=5, Na=30, K=2, Cl=40, SO4=20, Total_Alk_CaCO3=60,
                   EC=300, B=0.1),
        one_sample(Sample='salty', pH=7.5, Temp=25, Ca=60, Mg=40, Na=900, K=10, Cl=1400, SO4=300,
                   Total_Alk_CaCO3=200, EC=5000, B=3.5, Ba=1.5),
    ], ignore_index=True)
    t, _ = run(raw)
    sets = ['Drinking water: health (WHO)', 'Drinking water: acceptability (WHO)',
            'Irrigation: degree of restriction on use (FAO)']
    summary, flags = guidelines.assess(t, sets)
    by = lambda name: summary.set_index('Parameter').loc[name]
    assert by('Barium')['Which samples'] == 'salty'          # WHO 1.3, ADWG would be 2
    assert by('Boron')['Which samples'] == 'salty'
    assert by('Boron toxicity')['Which samples'] == 'salty (severe)'
    assert by('Salinity, ECw (uS/cm)')['Which samples'] == 'salty (severe)'
    assert by('Chloride')['Which samples'] == 'salty'
    assert 'semi-arid' in by('Boron toxicity')['Note']
    assert 'Boron toxicity (severe)' in flags.set_index('Sample').at['salty', 'Outside guideline for']


def test_chart_lines_and_bars():
    from core import guidelines, plots
    v = pd.DataFrame({'Sample': ['a', 'b', 'c'], 'Label': ['x', 'x', 'y'], 'Color': '', 'Marker': '',
                      'NO3-N (mg/L)': [0.5, 2.0, 5.0], 'Hardness (mg/L as CaCO3)': [10, 100, 300],
                      'pH': [6.0, 7.0, 9.0], 'EC': [500, 1000, 4000]})
    soft = [g for g in guidelines.SETS['Aquatic ecosystems: freshwater, toxicants at 95% protection (ANZG)']
            if 'soft' in g['name']][0]
    (_, thr), = guidelines.chart_lines(v, soft)
    assert thr.notna().tolist() == [True, False, False]      # only the soft-water sample
    # in the chart list the three hardness rows are one, with a value for every sample
    combined = [g for lab, _, g in guidelines.chart_rows() if g['kind'] == 'hard_max'][0]
    (_, thr), = guidelines.chart_lines(v, combined)
    assert thr.tolist() == [1.1, 2.6, 29]
    assert not any('soft water' in lab for lab, _, _ in guidelines.chart_rows())
    ph = guidelines.chart_lines(v, dict(var='pH', name='pH', kind='range', low=6.5, high=8.5, source=''))
    assert [l for l, _ in ph] == ['Lower limit: 6.5', 'Upper limit: 8.5']
    ec = [g for g in guidelines.SETS['Irrigation: degree of restriction on use (FAO)'] if g['var'] == 'EC'][0]
    assert len(guidelines.chart_lines(v, ec)) == 2
    for var, row in [('NO3-N (mg/L)', soft), ('pH', dict(var='pH', name='pH', kind='range', low=6.5, high=8.5,
                                                        source='')), ('EC', ec)]:
        fig = plots.guideline_bars(v, var, guidelines.chart_lines(v, row), highlight='#ff0000', sort=True)
        assert fig is not None
    assert plots.guideline_bars(v, 'EC', guidelines.chart_lines(v, ec), bar_colour='#123456', log=True)
    labels = [lab for lab, _, _ in guidelines.chart_rows()]
    assert all('Infiltration' not in l for l in labels)       # the SAR with EC table is not a bar chart


def test_chadha_fields():
    assert chem.chadha_field(40, 30) == '5' and chem.chadha_field(40, -30) == '6'
    assert chem.chadha_field(-40, -30) == '7' and chem.chadha_field(-40, 30) == '8'
    assert chem.chadha_field(0, 10) == 'On a field boundary' and chem.chadha_field(np.nan, 1) == ''
    # a Ca-HCO3 water lands in 5, a Na-Cl water in 7
    t, _ = run(pd.concat([
        one_sample(Sample='caco3', pH=7.5, Temp=25, Ca=60, Mg=10, Na=10, K=1, Cl=10, SO4=10, Total_Alk_CaCO3=180),
        one_sample(Sample='nacl', pH=7.5, Temp=25, Ca=10, Mg=5, Na=300, K=5, Cl=450, SO4=40, Total_Alk_CaCO3=50),
    ], ignore_index=True))
    assert list(t['Chadha field']) == ['5', '7']
    assert t.loc[0, 'Chadha x (meq%)'] > 0 and t.loc[1, 'Chadha y (meq%)'] < 0


def test_panel_letters_styles_and_gibbs():
    from matplotlib.figure import Figure
    from core import plots
    fig = Figure()
    axs = fig.subplots(1, 3)
    for style, want in [('(a)', ['(a)', '(b)', '(c)']), ('a)', ['a)', 'b)', 'c)']), ('a', ['a', 'b', 'c']),
                        ('a.', ['a.', 'b.', 'c.']), ('A', ['A', 'B', 'C'])]:
        for ax in axs:
            for t in list(ax.texts):
                t.remove()
        plots.panel_letters(axs, style)
        assert [ax.texts[0].get_text() for ax in axs] == want
        x, y = axs[0].texts[0].get_position()
        assert (x, y) == (0.95, 0.92)          # default: top right, inside
    plots.panel_letters(axs, None)              # no letters: nothing added
    # the Gibbs pair gets letters; single-panel diagrams do not
    t, _ = run(pd.concat([one_sample(Sample=f's{i}', pH=7.2, Temp=25, Ca=20 + i, Mg=5, Na=30, K=2, Cl=40, SO4=20,
                                     Total_Alk_CaCO3=60, TDS=200 + i) for i in range(3)], ignore_index=True))
    gibbs, _ = plots.wqchartpy_figure(t, 'Gibbs', letters='(a)')
    assert [a.texts[-1].get_text() for a in gibbs.axes] == ['(a)', '(b)']
    chadha, _ = plots.wqchartpy_figure(t, 'Chadha', letters='(a)')
    assert not any(tx.get_text() == '(a)' for a in chadha.axes for tx in a.texts)


def test_group_means_skip_filled_values():
    rows = [one_sample(Sample=f'S{k}', Label=lab, pH=ph, Temp=25, Ca=ca, Mg=5, Na=20, K=2,
                       Total_Alk_CaCO3=alk, Cl=30, SO4=10)
            for k, (lab, ca, ph, alk) in enumerate([('A', 10, 7.0, 100), ('A', 30, 7.2, np.nan),
                                                      ('B', 50, 7.4, 120)])]
    v, _ = run(pd.concat(rows, ignore_index=True))
    means, counts = chem.group_means(v)
    a = means.set_index('Label').loc['A']
    n = counts.set_index('Label').loc['A']
    assert a['Samples'] == 2
    assert a['Ca (mg/L)'] == pytest.approx(20.0)
    assert n['Ca (mg/L)'] == 2
    # the second A sample has a filled alkalinity, so only the first counts for HCO3
    assert n['HCO3 (mg/L)'] == 1
    assert a['HCO3 (mg/L)'] == pytest.approx(v.loc[0, 'HCO3 (mg/L)'])
    assert 'NO3 (mg/L)' not in means.columns           # not measured, not listed


def test_group_mean_markers_are_drawn():
    from core import plots
    v = chem.example_data()
    df, _, extra, _ = chem.clean(v.astype(object))
    import phreeqpython
    mmol, src = chem.to_mmol(df, 'mg/L')
    df, mmol, filled = chem.fill_missing(df, mmol, {'pH': ('mean', 7), 'Temp': ('mean', 25),
                                                    'HCO3': ('mean', 100)})
    ph, err, mmol = chem.run_phreeqc(df, mmol, phreeqpython.PhreeqPython(database='phreeqc.dat'))
    t = chem.build_table(df, mmol, filled, ph, extra, src)
    n_groups = t['Label'].nunique()
    base = plots.custom(t, 'Cl (mmol/L)', 'Na (mmol/L)')
    withm = plots.custom(t, 'Cl (mmol/L)', 'Na (mmol/L)', means=True)
    count = lambda f: len(f.axes[0].collections)
    assert count(withm) - count(base) == n_groups
    labels = [x.get_text() for x in withm.legends[0].get_texts()]
    assert 'Group mean (large marker)' in labels
    assert plots.piper(t, means=True) is not None
    assert plots.gallery_page(t, list(plots.GALLERY.values())[0][:2], means=True) is not None
    assert plots.custom(t, 'Cl (mmol/L)', 'Na (mmol/L)', color_by='pH', means=True) is not None
