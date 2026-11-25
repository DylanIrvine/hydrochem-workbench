# tests for hydrochem_workbench/advanced.py (mixing, PCA, clusters, group tests)
import numpy as np
import pandas as pd
import pytest

from hydrochem_workbench import advanced as A


def frame(rows, label='A'):
    # a minimal results table: concentrations in mmol/L and the flags hollow_mask reads
    v = pd.DataFrame(rows)
    v.insert(0, 'Sample', [f'S{k}' for k in range(len(v))])
    if 'Label' not in v:
        v['Label'] = label
    for c in ['filled_pH', 'filled_Temp', 'filled_HCO3', 'HCO3 is input', 'CO3 is input', 'TDS is measured']:
        v[c] = False
    return v


def test_two_end_members_by_hand():
    # Cl 1 and 101: a sample with Cl 26 is a quarter EM2. Na mixes conservatively except in the
    # last sample, which has gained 2 mmol/L
    em = [{'Cl (mmol/L)': 1.0, 'Na (mmol/L)': 1.0}, {'Cl (mmol/L)': 101.0, 'Na (mmol/L)': 90.0}]
    f2 = np.array([0.25, 0.5, 0.9])
    rows = [{'Cl (mmol/L)': 1 + 100 * f, 'Na (mmol/L)': 1 + 89 * f} for f in f2]
    rows[-1]['Na (mmol/L)'] += 2.0
    v = frame(em + rows)
    res = A.run_mixing(v, v.iloc[2:], ['Sample: S0', 'Sample: S1'], ['Cl (mmol/L)'], ['Na (mmol/L)'])
    assert res['fractions']['f EM2'].to_numpy() == pytest.approx(f2)
    assert res['fractions'].sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert res['deviations']['Na (mmol/L)'].to_numpy() == pytest.approx([0, 0, 2.0], abs=1e-9)
    assert not res['outside'].any()


def test_three_end_members_and_outside_flag():
    E = np.array([[1.0, 0.01, 0.5], [50.0, 0.20, 3.0], [10.0, 0.01, 8.0]])   # Cl, Br, Ca
    F = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [1.4, -0.1, -0.3]])      # last is outside
    S = F @ E
    cols = ['Cl (mmol/L)', 'Br (mmol/L)', 'Ca (mmol/L)']
    v = frame([dict(zip(cols, r)) for r in np.vstack([E, S])])
    res = A.run_mixing(v, v.iloc[3:], ['Sample: S0', 'Sample: S1', 'Sample: S2'], cols[:2], cols[2:])
    assert res['fractions'].to_numpy() == pytest.approx(F, abs=1e-9)
    assert res['deviations']['Ca (mmol/L)'].abs().max() < 1e-9
    assert res['outside'].tolist() == [False, False, True]


def test_group_mean_end_member_and_bad_choices():
    v = frame([{'Cl (mmol/L)': c, 'Na (mmol/L)': c} for c in [1, 3, 100, 100]],
              label=None)
    v['Label'] = ['fresh', 'fresh', 'sea', 'sea']
    res = A.run_mixing(v, v, ['Group mean: fresh', 'Group mean: sea'], ['Cl (mmol/L)'], ['Na (mmol/L)'])
    assert res['members'].loc['EM1', 'Cl (mmol/L)'] == pytest.approx(2.0)
    with pytest.raises(ValueError, match='different'):
        A.run_mixing(v, v, ['Group mean: sea', 'Group mean: sea'], ['Cl (mmol/L)'], [])
    with pytest.raises(ValueError, match='told apart'):
        A.run_mixing(v, v, ['Sample: S2', 'Sample: S3'], ['Cl (mmol/L)'], [])
    with pytest.raises(ValueError, match='tracer'):
        A.run_mixing(v, v, ['Sample: S0', 'Sample: S1', 'Sample: S2'], ['Cl (mmol/L)'], [])


def test_mixing_intervals_from_realisations():
    em = [{'Cl (mmol/L)': 1.0, 'Na (mmol/L)': 1.0}, {'Cl (mmol/L)': 101.0, 'Na (mmol/L)': 101.0}]
    v = frame(em + [{'Cl (mmol/L)': 51.0, 'Na (mmol/L)': 56.0}])
    rng = np.random.default_rng(1)
    n = 400
    base = v[['Cl (mmol/L)', 'Na (mmol/L)']].to_numpy()
    real = base[None] * (1 + 0.01 * rng.standard_normal((n,) + base.shape))
    mc = dict(n=n, index=list(v.index), conc={'Cl (mmol/L)': real[:, :, 0], 'Na (mmol/L)': real[:, :, 1]})
    res = A.run_mixing(v, v.iloc[2:], ['Sample: S0', 'Sample: S1'], ['Cl (mmol/L)'], ['Na (mmol/L)'], mc=mc)
    lo, hi = res['dev_lo'].iloc[0, 0], res['dev_hi'].iloc[0, 0]
    assert lo < 5.0 < hi and lo > 0          # a 5 mmol/L gain, clearly above 1 % noise
    t = A.mixing_table(v.iloc[2:], res)
    assert t['dNa (mmol/L) differs from 0'].iloc[0] == 'Yes'


def test_pca_and_clr():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(np.exp(rng.normal(size=(30, 4))), columns=[f'{a} (mmol/L)' for a in ['Ca', 'Mg', 'Na', 'Cl']])
    for how in A.TRANSFORMS:
        p = A.pca(X, how)
        assert p['explained'].sum() == pytest.approx(1.0)
        # scores times loadings give back the transformed data
        Z = A.transform(X, how).to_numpy()
        assert p['scores'].to_numpy() @ p['loadings'].to_numpy().T == pytest.approx(Z, abs=1e-9)
    # clr rows sum to zero, so the last component is empty
    assert A.pca(X, A.TRANSFORMS[0])['explained'].iloc[-1] < 1e-12


def test_clusters_find_separate_waters():
    rng = np.random.default_rng(5)
    cols = [f'{a} (mmol/L)' for a in ['Ca', 'Na', 'Cl', 'HCO3']]
    a = np.exp(rng.normal(np.log([2, 0.5, 0.5, 4]), 0.1, (8, 4)))
    b = np.exp(rng.normal(np.log([1, 40, 45, 2]), 0.1, (8, 4)))
    v = frame([dict(zip(cols, r)) for r in np.vstack([a, b])])
    v['Label'] = ['fresh'] * 8 + ['saline'] * 8
    X, dropped = A.prepare(v, cols)
    c = A.clusters(X, A.TRANSFORMS[1], 2)
    t = A.crosstab(v, c['labels'])
    assert sorted(t.to_numpy().ravel().tolist()) == [0, 0, 8, 8]


def test_holm_and_group_tests():
    assert A.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(9)
    v = frame([{'Na (mg/L)': x} for x in np.r_[rng.normal(10, 1, 8), rng.normal(30, 1, 8), [5.0, 6.0]]])
    v['Label'] = ['a'] * 8 + ['b'] * 8 + ['c'] * 2
    kw, pairs = A.group_tests(v, ['Na (mg/L)'])
    assert kw.loc[0, 'p'] < 0.01
    assert kw.loc[0, 'Groups tested'] == 2                 # c has only 2 values
    assert 'c' in kw.loc[0, 'Groups left out (< 3 values)']
    assert len(pairs) == 1 and pairs.loc[0, 'p (Holm-adjusted)'] < 0.01


def test_pca_biplot_panel_letters():
    from hydrochem_workbench import plots, plots_advanced as PA
    rng = np.random.default_rng(3)
    cols = [f'{a} (mmol/L)' for a in ['Ca', 'Mg', 'Na', 'Cl']]
    X = pd.DataFrame(np.exp(rng.normal(size=(30, 4))), columns=cols)
    v = X.assign(Sample=[f'S{i}' for i in range(30)], Label=['A', 'B', 'C'] * 10,
                 Color=['#1f77b4', '#ff7f0e', '#2ca02c'] * 10, Marker=['o', 's', '^'] * 10)
    p = A.pca(X, A.TRANSFORMS[0])
    texts = lambda fig: [t.get_text() for ax in fig.axes for t in ax.texts]
    plain = PA.pca_biplot(v, p)
    assert '(a)' not in texts(plain) and plain.axes[1].get_ylim() == (0, 100)
    for pos in plots.PANEL_LABEL_POS:
        fig = PA.pca_biplot(v, p, letters='(a)', letters_pos=pos)
        assert '(a)' in [t.get_text() for t in fig.axes[0].texts]
        assert '(b)' in [t.get_text() for t in fig.axes[1].texts]
        # the biplot stays square and symmetric; the variance panel gets headroom only for inside letters
        assert fig.axes[0].get_xlim() == fig.axes[0].get_ylim()
        assert (fig.axes[1].get_ylim()[1] > 100) == ('inside' in pos)
