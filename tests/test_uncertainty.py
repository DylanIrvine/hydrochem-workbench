# uncertainty analysis: inputs, sampling and the Monte Carlo run
import numpy as np
import pandas as pd
import phreeqpython
import pytest

from core import chem, uncertainty as U

RULES = (('pH', ('mean', 7.0)), ('Temp', ('mean', 25.0)), ('HCO3', ('mean', 100.0)))


@pytest.fixture(scope='module')
def pp():
    return phreeqpython.PhreeqPython(database='phreeqc.dat')


def _setup(unit='mg/L'):
    raw = chem.example_data().astype(object)
    raw.loc[0, 'SO4'] = '<1'
    raw.loc[2, 'K'] = '<1'
    raw, rep = chem.below_lor(raw, 0.5, rows=True)
    df, _, extra, problems = chem.clean(raw)
    assert not problems
    below = {(r.Row, r.Column): r.LOR for r in rep.itertuples()}
    settings = dict(unit=unit, n_as_n=False, p_as_p=False, dic_unit='mg/L as C', rules=RULES,
                    extra_cols=tuple(extra))
    return df, below, settings


def test_default_table_complete():
    t = U.default_table()
    assert list(t.columns) == U.TABLE_COLS
    assert (t[['LOR', 'Relative SD (%)', 'Absolute SD']] >= 0).all().all()
    # every concentration column in the template has a row
    covered = {c for cols in U.COLUMNS.values() for c in cols}
    for c in list(chem.ANALYTES) + chem.ALK_COLS + [chem.DIC_COL, 'pH', 'Temp', 'EC', 'TDS']:
        assert c in covered, c
    # GUM rectangular for the probe specs
    assert t.set_index('Variable').at['pH', 'Absolute SD'] == pytest.approx(0.2 / np.sqrt(3), abs=1e-3)


def test_check_and_mark_edited():
    t = U.default_table()
    t.loc[t['Variable'] == 'Cl', 'Relative SD (%)'] = 10.0
    t.loc[t['Variable'] == 'Ca', 'LOR'] = -1          # not allowed, clipped to 0
    t = t[t['Variable'] != 'Br']                      # a dropped row comes back
    out = U.mark_edited(U.check_table(t)).set_index('Variable')
    assert out.at['Cl', 'Source'] == 'Entered by user'
    assert out.at['Ca', 'LOR'] == 0
    assert out.at['Br', 'Relative SD (%)'] == pytest.approx(3.7)
    assert out.at['Na', 'Source'] != 'Entered by user'


def test_lor_units():
    # 1 mg/L Ca -> mmol/L and meq/L; NO3 LOR is as N on the lab report
    assert U.lor_in_units('Ca', 'Ca', 1.0, 'mmol/L') == pytest.approx(1 / 40.078)
    assert U.lor_in_units('Ca', 'Ca', 1.0, 'meq/L') == pytest.approx(2 / 40.078)
    assert U.lor_in_units('NO3', 'NO3', 0.01, 'mg/L', n_as_n=True) == pytest.approx(0.01)
    assert U.lor_in_units('NO3', 'NO3', 0.01, 'mg/L', n_as_n=False) == pytest.approx(0.01 * chem.N_TO_NO3, rel=1e-3)
    assert U.lor_in_units('Alk', 'Total_Alk_CaCO3', 1.0, 'mmol/L') == 1.0


def test_sigma_formula():
    df, _, _ = _setup()
    sig, lors = U.sigmas(df, U.default_table(), 'mg/L')
    x = df.at[5, 'Cl']
    expect = np.sqrt((1 / np.sqrt(12)) ** 2 + (0.069 * x) ** 2)
    assert sig.at[5, 'Cl'] == pytest.approx(expect)
    assert sig.at[5, 'pH'] == pytest.approx(0.115, abs=1e-3)
    # not measured -> no sigma
    assert sig['Cl'].isna().equals(df['Cl'].isna())


def test_perturb_below_lor_and_positive():
    df, below, _ = _setup()
    sig, _ = U.sigmas(df, U.default_table(), 'mg/L')
    rng = np.random.default_rng(1)
    so4, k = [], []
    for _ in range(300):
        d = U.perturb(df, sig, below, rng)
        conc = d[['Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4']]
        assert (conc.fillna(0) >= 0).all().all()
        so4.append(d.at[0, 'SO4'])
        k.append(d.at[2, 'K'])
    # below-LOR values are Uniform(0, LOR)
    for vals in (so4, k):
        vals = np.array(vals)
        assert vals.min() >= 0 and vals.max() <= 1.0
        assert vals.mean() == pytest.approx(0.5, abs=0.06)


def test_run_example(pp):
    df, below, settings = _setup()
    v, _ = chem.derive(df, pp, **settings)
    mc = U.run(df, below, U.default_table(), settings, pp, n=40, seed=3)
    lo, hi = mc['lo'], mc['hi']
    assert lo.index.equals(v.index)
    for c in ['SI Calcite', 'CBE (%)', 'Na/Cl', 'piper_d_x']:
        ok = lo[c].notna()
        assert ok.any()
        assert (lo[c][ok] <= hi[c][ok] + 1e-12).all()
    # measured SI near the middle of its interval for samples with no filled input
    full = v['Filled'] == ''
    inside = (v.loc[full, 'SI Calcite'] >= lo.loc[full, 'SI Calcite'] - 0.05) & \
             (v.loc[full, 'SI Calcite'] <= hi.loc[full, 'SI Calcite'] + 0.05)
    assert inside.mean() > 0.9
    s = U.summary(v, mc)
    assert s['Same water type (%)'].between(0, 100).all()
    assert 'SI Calcite > 0 (% of realisations)' in s.columns
    t = U.interval_table(v, mc)
    assert 'SI Calcite 5th pct' in t.columns and 'SI Calcite 95th pct' in t.columns
    # same seed, same answer
    mc2 = U.run(df, below, U.default_table(), settings, pp, n=40, seed=3)
    pd.testing.assert_frame_equal(mc['lo'], mc2['lo'])


def test_zero_uncertainty_gives_no_spread(pp):
    df, _, settings = _setup()
    t = U.default_table()
    t[['LOR', 'Relative SD (%)', 'Absolute SD']] = 0.0
    v, _ = chem.derive(df, pp, **settings)
    mc = U.run(df, {}, t, settings, pp, n=5)
    np.testing.assert_allclose(mc['lo']['SI Calcite'], v['SI Calcite'], atol=1e-9)
    np.testing.assert_allclose(mc['hi']['SI Calcite'], v['SI Calcite'], atol=1e-9)


def test_other_units_run(pp):
    for unit in ['mmol/L', 'meq/L']:
        df, below, settings = _setup(unit)
        mc = U.run(df, below, U.default_table(), settings, pp, n=5)
        assert mc['lo']['CBE (%)'].notna().any()


def test_outline_and_hull():
    from core import plots
    rng = np.random.default_rng(0)
    line = plots.central_outline(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
    assert line is not None and np.allclose(line[0], line[-1])
    # roughly 90 % of a fresh sample falls inside
    from matplotlib.path import Path
    p = Path(line)
    pts = np.column_stack([rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)])
    assert 0.8 < p.contains_points(pts).mean() < 0.97
    assert plots.central_outline(np.ones(10), np.ones(10)) is None
