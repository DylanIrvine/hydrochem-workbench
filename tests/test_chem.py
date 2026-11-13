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
    assert 'HCO3' not in chem.TEMPLATE_COLS and 'CO3' not in chem.TEMPLATE_COLS


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
    summary, flags = guidelines.assess(t, guidelines.set_names(), 'Lowland river')
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
    assert 'Fluoride' in flags.set_index('Sample').at['bad', 'Outside guideline for']
