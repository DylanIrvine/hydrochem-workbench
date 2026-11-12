# Checks on the chemistry: unit conversion, charge balance, fill rules, Piper
# geometry and a PHREEQC sanity test that does not depend on our own numbers.
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
    rules = rules or {'pH': ('mean', 7), 'Temp': ('mean', 25), 'HCO3': ('mean', 2)}
    df, dropped, extra, problems = chem.clean(raw)
    mmol = chem.to_mmol(df, unit, **kw)
    df, mmol, filled = chem.fill_missing(df, mmol, rules)
    ph, err = chem.run_phreeqc(df, mmol, phreeqpython.PhreeqPython(database='phreeqc.dat'))
    return chem.build_table(df, mmol, filled, ph, extra), dropped


def test_units_agree():
    # 40.078 mg/L Ca = 1 mmol/L = 2 meq/L, whichever way it is entered
    a, _ = run(one_sample(pH=7, Temp=25, Ca=40.078, Mg=0, Na=0, K=0, HCO3=122.034, Cl=0, SO4=0), 'mg/L')
    b, _ = run(one_sample(pH=7, Temp=25, Ca=1.0, Mg=0, Na=0, K=0, HCO3=2.0, Cl=0, SO4=0), 'mmol/L')
    c, _ = run(one_sample(pH=7, Temp=25, Ca=2.0, Mg=0, Na=0, K=0, HCO3=2.0, Cl=0, SO4=0), 'meq/L')
    for t in (a, b, c):
        assert t.at[0, 'Ca (mmol/L)'] == pytest.approx(1.0, rel=1e-6)
        assert t.at[0, 'Ca (meq/L)'] == pytest.approx(2.0, rel=1e-6)
        assert t.at[0, 'CBE (%)'] == pytest.approx(0.0, abs=1e-4)
    assert a.at[0, 'SI Calcite'] == pytest.approx(b.at[0, 'SI Calcite'], abs=1e-4)


def test_alkalinity_as_caco3_and_nitrate_as_n():
    # 100 mg/L as CaCO3 = 2 meq/L = 121.9 mg/L HCO3; 10 mg/L NO3-N = 44.27 mg/L NO3
    t, _ = run(one_sample(pH=7, Temp=25, Ca=40, Mg=1, Na=1, K=1, HCO3=100, Cl=1, SO4=1, NO3=10),
               'mg/L', alk_as_caco3=True, no3_as_n=True)
    assert t.at[0, 'HCO3 (meq/L)'] == pytest.approx(2.0, rel=2e-3)
    assert t.at[0, 'NO3 (mg/L)'] == pytest.approx(44.27, rel=1e-3)


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
    assert t.at[2, 'HCO3 (mmol/L)'] == pytest.approx(1.5, rel=1e-4)
    assert t.at[2, 'Filled'] == 'pH, Temp, HCO3'
    assert t.at[0, 'Filled'] == ''
    # hollow only where the fill matters
    assert not chem.hollow_mask(t, ['Na (mmol/L)', 'Cl (mmol/L)'])[2]
    assert chem.hollow_mask(t, ['SI Calcite', 'Cl (mmol/L)'])[2]
    assert not chem.hollow_mask(t, ['SI Calcite'])[0]

    t2, _ = run(raw, rules={'pH': ('value', 6.5), 'Temp': ('value', 22), 'HCO3': ('balance', 0)})
    assert t2.at[2, 'pH'] == 6.5
    assert t2.at[2, 'CBE (%)'] == pytest.approx(0.0, abs=1e-6)


def test_piper_corners_and_water_types():
    def pos(**meq):
        base = dict(Ca=0, Mg=0, Na=0, K=0, HCO3=0, CO3=0, Cl=0, SO4=0)
        base.update(meq)
        return chem.piper_coords(pd.DataFrame([base], dtype=float)).iloc[0]
    h = 0.5 * np.tan(np.pi / 3)
    p = pos(Ca=1, HCO3=1)                      # pure Ca-HCO3: left corner of the diamond
    assert (p.cat_x, p.cat_y) == pytest.approx((0, 0))
    assert (p.d_x, p.d_y) == pytest.approx((0.6, h + 0.1 * np.tan(np.pi / 3)))
    p = pos(Na=1, Cl=1)                        # pure Na-Cl: right corner
    assert p.d_x == pytest.approx(1.6)


def test_water_type_follows_the_shaded_fields():
    assert chem.water_type(1.0, 0.0) == 'Mg-HCO3'      # left corner (Ca-Mg-HCO3 waters)
    assert chem.water_type(0.0, 1.0) == 'Na-Cl'        # right corner
    assert chem.water_type(1.0, 1.0) == 'Ca-Cl'        # top
    assert chem.water_type(0.0, 0.0) == 'Na-HCO3'      # bottom
    assert chem.water_type(0.5, 0.5) == 'Mixed'        # centre
    assert chem.water_type(0.9, 0.3) == 'Mg-HCO3'
    assert chem.water_type(0.6, 0.7) == 'Mixed'        # upper grey triangle


def test_phreeqc_recovers_calcite_equilibrium():
    # Make a water that PHREEQC itself has equilibrated with calcite, read its
    # composition back, push that through our pipeline, and expect SI = 0.
    import phreeqpython
    pp = phreeqpython.PhreeqPython(database='phreeqc.dat')
    s = pp.add_solution({'units': 'mmol/l', 'pH': 7.0, 'temp': 25, 'Na': 1, 'Cl': 1})
    s.equalize(['Calcite', 'CO2(g)'], [0.0, -2.0])
    ca = s.total('Ca', units='mmol')
    alk = 2 * ca                                 # all alkalinity came from calcite
    raw = one_sample(pH=s.pH, Temp=25, Ca=ca, Mg=0, Na=1, K=0, HCO3=alk, Cl=1, SO4=0)
    t, _ = run(raw, 'mmol/L')
    assert t.at[0, 'SI Calcite'] == pytest.approx(0.0, abs=0.02)
    assert t.at[0, 'log pCO2'] == pytest.approx(-2.0, abs=0.02)
    assert np.isnan(t.at[0, 'SI Gypsum'])                # no sulfate, so no gypsum SI


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
