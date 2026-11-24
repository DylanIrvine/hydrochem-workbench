# irrigation classes, scaling indices, analysis checks, outliers, statistics with values below
# the LOR, the CCME index and the choice of PHREEQC database
import numpy as np
import pandas as pd
import pytest

import hydrochem_workbench as hw
from hydrochem_workbench import advanced as A, chem, guidelines, plots, plots_quality as PQ, quality as Q, workflow


@pytest.fixture(scope='module')
def pp():
    return workflow.get_phreeqc()


@pytest.fixture(scope='module')
def v(pp):
    return hw.run(chem.example_data(), phreeqc=pp).table


@pytest.fixture(scope='module')
def v_measured(pp, v):
    # the example with an EC and a TDS added, 5 % above and 10 % above the calculated values
    d = chem.example_data()
    d['EC'] = (v['SC calc (uS/cm)'] * 1.05).round(1)
    d['TDS'] = (v['TDS calc (mg/L)'] * 1.10).round(1)
    return hw.run(d, phreeqc=pp).table


def test_ussl_and_wilcox_classes():
    # the corners of the USSL diagram: SAR limits of 10, 18, 26 at 100 uS/cm and 4, 9, 14 at 2250
    for a, b in Q.USSL_SAR:
        assert a - b * np.log10(100) == pytest.approx([10, 18, 26][Q.USSL_SAR.index((a, b))], abs=0.05)
        assert a - b * np.log10(2250) == pytest.approx([4, 9, 14][Q.USSL_SAR.index((a, b))], abs=0.05)
    assert Q.ussl_class([200, 500, 1000, 3000, np.nan], [2, 12, 12, 30, 5]) == ['C1-S1', 'C2-S2', 'C3-S3', 'C4-S4', '']
    assert Q.wilcox_class([200, 200, 900, 2500, 3500, np.nan], [10, 70, 30, 30, 10, 10]) == [
        'Excellent', 'Doubtful', 'Permissible', 'Doubtful', 'Unsuitable', '']


def test_wilcox_diagram_fields():
    # the curves pass through the points of Wilcox (1948, Fig. 1) and end where the verticals meet them
    assert float(Q.wilcox_curve(0, 750)) == pytest.approx(64.5) and float(Q.wilcox_curve(0, 2000)) == pytest.approx(48)
    assert float(Q.wilcox_curve(1, 2000)) == pytest.approx(76.5) and float(Q.wilcox_curve(2, 3000)) == pytest.approx(92.1)
    assert np.isnan(Q.wilcox_curve(0, 2500)) and np.isnan(Q.wilcox_curve(2, 3200))
    for k in range(3):                                   # each curve falls all the way, and they do not cross
        x = np.linspace(0, 2000, 400)
        assert (np.diff(Q.wilcox_curve(k, x)) <= 0).all()
        if k:
            assert (Q.wilcox_curve(k, x[1:]) > Q.wilcox_curve(k - 1, x[1:])).all()
    # the worked example of the bulletin: sample 16642, 39 percent sodium at 12.06 e.p.m. (1206 uS/cm)
    assert Q.wilcox_field([1206], [39]) == ['Good to permissible']
    # points labelled on the original figure: Elephant Butte (42 % at about 650), County Line (60 % at about 2900)
    assert Q.wilcox_field([650, 2900], [42, 60]) == ['Excellent to good', 'Doubtful to unsuitable']
    ec = [500, 500, 1000, 1000, 1500, 1500, 2500, 2500, 3200, 200, np.nan, 1000]
    na = [70, 73, 59, 70, 85, 97, 50, 95, 10, 99.9, 50, np.nan]
    assert Q.wilcox_field(ec, na) == [
        'Excellent to good', 'Permissible to doubtful', 'Good to permissible', 'Permissible to doubtful',
        'Doubtful to unsuitable', 'Unsuitable', 'Doubtful to unsuitable', 'Unsuitable', 'Unsuitable', 'Unsuitable',
        '', '']
    # fresh, sodium-rich water: lenient on the diagram, strict in the table
    assert Q.wilcox_field([500], [70]) == ['Excellent to good'] and Q.wilcox_class([500], [70]) == ['Doubtful']


def test_wilcox_percent_sodium(v):
    # Na / (Ca + Mg + Na + K): potassium in the total only, so never above the Na + K form
    w = Q.wilcox_sodium(v)
    assert (w <= v['Na (%)'] + 1e-9).all() and (w > 0).all()
    k = v['K (meq/L)'].fillna(0)
    total = v['Ca (meq/L)'] + v['Mg (meq/L)'] + v['Na (meq/L)'] + k
    assert np.allclose(w, v['Na (meq/L)'] / total * 100)
    t = Q.irrigation_table(v)
    assert {'Percent sodium (Wilcox)', 'Wilcox diagram field', 'Wilcox table grade'} <= set(t.columns)
    assert set(t['Wilcox diagram field']) <= set(Q.WILCOX_FIELDS)


def test_ec_units(pp):
    d = chem.example_data()
    base = hw.run(d, phreeqc=pp)
    d['EC'] = base.table['SC calc (uS/cm)'].round(0)
    right = hw.run(d, phreeqc=pp)
    assert right.ec_warning == '' and chem.ec_unit_check(right.table) == ''
    assert dict(zip(right.settings['Setting'], right.settings['Value']))['EC against the ions'] == 'Consistent'
    # the same water reported in mS/cm, dS/m and mS/m gives the same EC and the same irrigation classes
    for unit, factor in (('mS/cm', 1000.0), ('dS/m', 1000.0), ('mS/m', 10.0)):
        e = d.copy()
        e['EC'] = d['EC'] / factor
        r = hw.run(e, phreeqc=pp, ec_unit=unit)
        assert np.allclose(r.table['EC'].astype(float), right.table['EC'].astype(float)) and r.ec_warning == ''
        assert (Q.irrigation_table(r.table)['USSL class'] == Q.irrigation_table(right.table)['USSL class']).all()
        # read as uS/cm instead, it is caught, and the message names the likely unit
        with pytest.warns(UserWarning, match='EC'):
            wrong = hw.run(e, phreeqc=pp)
        assert chem.EC_LABELS[chem.ec_unit_key(unit)] in wrong.ec_warning
    # uS/cm data with the unit set to mS/cm
    with pytest.warns(UserWarning):
        assert chem.EC_LABELS['uS/cm'] in hw.run(d, phreeqc=pp, ec_unit='mS/cm').ec_warning
    with pytest.raises(ValueError):
        hw.run(d, phreeqc=pp, ec_unit='S/m')
    # too few samples with an EC to judge, and no EC at all: no message
    assert base.ec_warning == ''


def test_irrigation_and_scaling_tables(v):
    t = Q.irrigation_table(v)
    assert len(t) == len(v) and (t['EC is'] == 'Calculated').all()        # no EC in the example
    assert t['USSL class'].str.match(r'C[1-4]-S[1-4]').all()
    assert set(t['RSC class']) <= {'Safe', 'Marginal', 'Unsuitable'}
    assert set(t.loc[t['Kelly ratio'] > 1, 'Kelly class']) == {'Unsuitable'}
    s = Q.scaling_table(v)
    ok = s['Langelier index'].notna()
    assert ok.sum() >= 20
    # Langelier is the calcite saturation index; Ryznar = pH - 2 x Langelier
    assert np.allclose(s.loc[ok, 'Langelier index'], v.loc[ok, 'SI Calcite'])
    assert np.allclose(s.loc[ok, 'Ryznar index'], v.loc[ok, 'pH'] - 2 * v.loc[ok, 'SI Calcite'])
    assert set(s.loc[s['Langelier index'] > 0.5, 'Langelier tendency']) <= {'Scale forming'}
    assert s['Filled input'].sum() >= 1 and 'Langelier index' in chem._PHREEQC_EXTRA
    for fig in (PQ.ussl(v), PQ.wilcox(v)):
        assert plots.fig_png(fig, dpi=50)[:4] == b'\x89PNG'


def test_analysis_checks(v, v_measured):
    none = Q.qa_table(v)
    assert none['Checks possible'].sum() == 0 and PQ.qa_checks(v) is None
    t = Q.qa_table(v_measured)
    assert np.allclose(t['EC calculated / measured'], 1 / 1.05, atol=1e-3)
    assert np.allclose(t['TDS measured / calculated'], 1.10, atol=1e-3)
    assert (t['Checks possible'] == 5).all()
    assert not t['Outside the range for'].str.contains('EC calculated').any()
    assert not t['Outside the range for'].str.contains('TDS measured').any()
    assert t['Outside the range for'].str.startswith('CBE').sum() == 4
    assert plots.fig_png(PQ.qa_checks(v_measured), dpi=50)[:4] == b'\x89PNG'


def test_univariate_outliers(v):
    out = Q.univariate_outliers(v, ['Ca (mg/L)', 'NO3 (mg/L)', 'pH', 'not a column'])
    assert list(out['Sample']) == ['SS03'] and out['Variable'].iloc[0] == 'NO3 (mg/L)'     # the 62 mg/L nitrate
    assert out['Modified z'].iloc[0] > Q.OUTLIER_Z and out['Scale'].iloc[0] == 'log10'
    assert Q.univariate_outliers(v.head(4), ['NO3 (mg/L)']).empty                          # too few samples


def test_multivariate_outliers():
    rng = np.random.default_rng(3)
    Z = pd.DataFrame(rng.normal(size=(80, 4)), columns=list('abcd'))
    Z.iloc[:3] += 8                                    # three samples far from the rest
    out, limit, note = Q.multivariate_outliers(Z)
    assert note == '' and limit == pytest.approx(np.sqrt(11.143), abs=0.01)
    assert out['Outlier'].iloc[:3].all() and out['Outlier'].iloc[3:].sum() <= 6     # about 2.5 % expected
    # a direction with no variance (as in clr data) is dropped, not inverted
    Z2 = Z.sub(Z.mean(axis=1), axis=0)
    out2, _, note2 = Q.multivariate_outliers(Z2)
    assert note2 == '' and np.isfinite(out2['Distance']).all()
    few, _, note3 = Q.multivariate_outliers(Z.head(6))
    assert 'too few' in note3 and not few['Outlier'].any()


def test_ros_and_kaplan_meier_without_censoring_are_the_sample_statistics():
    x = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
    none = np.zeros(6, bool)
    vals, mass = Q._kaplan_meier(x, none)
    mean, sd, p25, med, p75 = Q._weighted(vals, mass)
    assert mean == pytest.approx(x.mean()) and sd == pytest.approx(x.std())
    assert np.allclose(mass, 1 / 6)


def test_ros_recovers_a_lognormal():
    rng = np.random.default_rng(5)
    true = np.exp(rng.normal(0.0, 1.0, 400))
    lor = np.where(np.arange(400) % 2 == 0, 0.5, 1.0)          # two detection limits
    cen = true < lor
    est = Q._ros(np.where(cen, lor, true), cen)
    assert 0.3 < cen.mean() < 0.6
    assert est[~cen] == pytest.approx(true[~cen])              # detected values are not touched
    assert (est[cen] < lor[cen] * 1.0001).mean() > 0.95        # estimates sit below their limit
    assert est.mean() == pytest.approx(true.mean(), rel=0.05)
    assert np.median(est) == pytest.approx(np.median(true), rel=0.1)
    # substitution at half the limit does worse on the spread
    half = np.where(cen, lor / 2, true)
    assert abs(np.log(est).std() - np.log(true).std()) < abs(np.log(half).std() - np.log(true).std())
    vals, mass = Q._kaplan_meier(np.where(cen, lor, true), cen)
    assert mass.sum() == pytest.approx(1.0)
    assert Q._weighted(vals, mass)[0] == pytest.approx(true.mean(), rel=0.1)


def test_censored_stats_table(pp):
    d = chem.example_data().astype(object)
    d.loc[[0, 1, 2, 5, 9], 'Br'] = '<0.5'
    d.loc[[3, 4], 'Br'] = '<0.2'
    d.loc[d.index[:24], 'Fe'] = '<0.05'                         # nearly all below the LOR
    r = hw.run(d, phreeqc=pp)
    t = Q.censored_stats(r.base['df'], r.base['below'])
    assert list(t.columns) == Q.CENSORED_COLS and set(t['Variable']) == {'Br', 'Fe'}
    br = t[t['Variable'] == 'Br'].set_index('Method')
    assert (br['n'] == 26).all() and (br['Below LOR'] == 7).all()
    assert br.loc['ROS', 'Mean'] == pytest.approx(br.loc['Substitution (as used in the app)', 'Mean'], rel=0.05)
    assert br.loc['Kaplan-Meier', 'Median'] > 0 and br.loc['Kaplan-Meier', 'Note'] == ''   # lowest value is detected
    fe = t[t['Variable'] == 'Fe'].set_index('Method')
    assert np.isnan(fe.loc['ROS', 'Mean']) and 'Not estimated' in fe.loc['ROS', 'Note']
    assert np.isfinite(fe.loc['Substitution (as used in the app)', 'Mean'])


def test_ccme_index_by_hand():
    # 4 variables, 5 samples = 20 tests. One variable fails twice, at 2x and 4x its limit
    rows = [dict(var=c, name=c, kind='max', value=10.0, source='') for c in 'ABC'] + \
           [dict(var='pH', name='pH', kind='range', low=6.5, high=8.5, source='')]
    t = pd.DataFrame({'Sample': list('12345'), 'Label': 'G', 'A': [1, 2, 20, 40, 5], 'B': 1.0, 'C': 2.0, 'pH': 7.0,
                      'Hardness (mg/L as CaCO3)': 100.0})
    w = Q.ccme_wqi(t, rows).iloc[0]
    f1, f2 = 25.0, 10.0
    nse = (1 + 3) / 20
    f3 = nse / (0.01 * nse + 0.01)
    assert (w['Variables'], w['Tests'], w['Failed variables'], w['Failed tests']) == (4, 20, 1, 2)
    assert w['F1 (scope)'] == f1 and w['F2 (frequency)'] == f2 and w['F3 (amplitude)'] == pytest.approx(f3)
    assert w['CCME WQI'] == pytest.approx(100 - np.sqrt(f1 ** 2 + f2 ** 2 + f3 ** 2) / 1.732)
    assert w['Category'] == 'Good' and bool(w['Meets CCME minimum'])
    # a pH below its range is an excursion of low / value - 1
    t2 = t.assign(pH=[7, 7, 7, 7, 5.2])
    assert Q.ccme_wqi(t2, rows[3:]).iloc[0]['F3 (amplitude)'] == pytest.approx(
        (6.5 / 5.2 - 1) / 5 / (0.01 * (6.5 / 5.2 - 1) / 5 + 0.01))
    clean = Q.ccme_wqi(t.assign(A=1.0), rows).iloc[0]
    assert clean['CCME WQI'] == 100 and clean['Category'] == 'Excellent'


def test_ccme_on_example(v):
    w = Q.ccme_wqi(v, guidelines.SETS[guidelines.set_names()[0]])
    assert list(w['Group']) == [chem.ALL_SAMPLES] + list(pd.unique(v['Label']))
    assert w['CCME WQI'].between(0, 100).all()
    assert plots.fig_png(PQ.ccme_bars(w, 'test'), dpi=50)[:4] == b'\x89PNG'
    # graded guidelines (FAO irrigation) are left out, so that set gives no index
    fao = [n for n in guidelines.set_names() if 'FAO' in n][0]
    graded = Q.ccme_wqi(v, [g for g in guidelines.SETS[fao] if g['kind'] not in ('max', 'range')])
    assert graded['Tests'].sum() == 0 and PQ.ccme_bars(graded) is None


def test_databases(v):
    assert list(chem.DATABASES)[0] == 'phreeqc.dat'
    with pytest.raises(ValueError):
        workflow.get_phreeqc('minteq.dat')
    full = chem.example_data().assign(Ni=0.02, I=0.05, Li=0.01)
    w = hw.run(full, database='wateq4f.dat')
    assert w.errors is None or len(w.errors) == 0
    # wateq4f takes Ni and I into the speciation and has no conductivity data
    assert {'Ni', 'I'} <= set(chem.phreeqc_keys(workflow.get_phreeqc('wateq4f.dat')))
    assert not {'Ni', 'I'} & set(chem.phreeqc_keys(workflow.get_phreeqc('phreeqc.dat')))
    assert w.table['SC calc (uS/cm)'].isna().all()
    ok = v['SI Calcite'].notna()
    assert np.abs(w.table.loc[ok, 'SI Calcite'] - v.loc[ok, 'SI Calcite']).max() < 0.05
    left = w.settings.set_index('Setting').at['Not in this database (left out of the speciation)', 'Value']
    assert left == 'None'
    # pitzer: no nitrogen, phosphorus or fluoride, some minerals missing, saline samples differ most
    p = hw.run(full, database='pitzer.dat')
    left = p.settings.set_index('Setting').at['Not in this database (left out of the speciation)', 'Value']
    assert {'NO3', 'NH4', 'PO4', 'F', 'Ni', 'I'} <= set(left.split(', '))
    assert p.table['SI Fluorite'].isna().all() and p.table['SI Calcite'].notna().sum() == ok.sum()
    assert p.table['SC calc (uS/cm)'].notna().any()
    # the charge balance does not depend on the database
    assert np.allclose(p.table['Sum cations (meq/L)'], w.table['Sum cations (meq/L)'])
