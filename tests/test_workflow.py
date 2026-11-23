# the package used from a script: run(), the outputs, and agreement with the app
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import hydrochem_workbench as hw
from hydrochem_workbench import chem, workflow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def pp():
    return workflow.get_phreeqc()


@pytest.fixture(scope='module')
def result(pp):
    return hw.run(chem.example_data(), phreeqc=pp)


def test_public_names():
    assert hw.__version__ == hw.VERSION
    for name in ('run', 'save_results', 'save_figures', 'process', 'fill_rules', 'style_groups',
                 'mark_charge_balance', 'MARK', 'EXCLUDE', 'NO_ACTION', 'VALUE', 'INTERVAL'):
        assert hasattr(hw, name)


def test_run_on_example(result):
    v = result.table
    assert len(v) == 26 and len(result.plotted) == 26
    assert {'CBE (%)', 'Water type', 'SI Calcite', chem.CB_COL, 'Color', 'Marker'} <= set(v.columns)
    assert (v[chem.CB_COL] == chem.CB_BEYOND).sum() == 4 and (v[chem.CB_COL] == chem.CB_NA).sum() == 1
    assert (v[chem.CB_RING] != '').sum() == 4
    assert set(result.styles) == set(v['Label']) and result.mc is None
    assert 'Result: 26 samples' in repr(result)
    assert list(result.settings.columns) == ['Section', 'Setting', 'Value']


def test_run_from_files(tmp_path, pp, result):
    xlsx, csv = tmp_path / 'data.xlsx', tmp_path / 'data.csv'
    with pd.ExcelWriter(xlsx) as xl:
        chem.example_data().to_excel(xl, sheet_name='Data', index=False)
    chem.example_data().to_csv(csv, index=False)
    for path in (xlsx, str(csv)):
        r = hw.run(path, phreeqc=pp)
        pd.testing.assert_series_equal(r.table['CBE (%)'], result.table['CBE (%)'], check_exact=False, rtol=1e-9)
    with pytest.raises(ValueError):
        hw.run(csv, unit='ppm', phreeqc=pp)
    with pytest.raises(ValueError, match='could not be run'):
        hw.run(chem.example_data().drop(columns='Sample'), phreeqc=pp)


def test_run_options(pp, result):
    ex = hw.run(chem.example_data(), phreeqc=pp, cb_action=hw.EXCLUDE)
    assert len(ex.table) == 26 and len(ex.plotted) == 22 and (ex.table[chem.CB_RING] == '').all()
    none = hw.run(chem.example_data(), phreeqc=pp, cb_action=hw.NO_ACTION, cb_limit=10)
    assert len(none.plotted) == 26 and (none.table[chem.CB_COL] == chem.CB_BEYOND).sum() < 4
    # fill rules: a fixed pH changes the filled sample only
    fixed = hw.run(chem.example_data(), phreeqc=pp, missing_ph=6.0, missing_alkalinity='balance')
    assert fixed.table.loc[3, 'pH'] == 6.0 and fixed.table.loc[0, 'pH'] == result.table.loc[0, 'pH']
    assert workflow.fill_rules(7.5, 'mean', 'balance') == (('pH', ('value', 7.5)), ('Temp', ('mean', 25.0)),
                                                           ('HCO3', ('balance', 100.0)))
    with pytest.raises(ValueError):
        workflow.fill_rules(ph='balance')
    styled = hw.run(chem.example_data(), phreeqc=pp,
                    styles={lab: ('#123456', 'D') for lab in pd.unique(chem.example_data()['Label'])})
    assert set(styled.table['Color']) == {'#123456'} and set(styled.table['Marker']) == {'D'}


def test_run_with_uncertainty(pp):
    r = hw.run(chem.example_data(), phreeqc=pp, uncertainty_n=30, cb_basis=hw.INTERVAL)
    assert r.mc is not None and r.mc['n'] == 30
    assert (r.table[chem.CB_COL] == chem.CB_BEYOND).sum() <= 4
    assert 'allowing for uncertainty' in r.table[chem.CB_LABEL].iloc[0]


def test_save_results_and_figures(tmp_path, pp, result):
    from hydrochem_workbench import guidelines
    path = hw.save_results(result, tmp_path / 'results.xlsx', guideline_sets=guidelines.set_names()[:2])
    sheets = pd.read_excel(path, sheet_name=None)
    assert {'Results', 'Settings', 'Groups', 'Group means', 'Group mean counts', 'Guidelines'} <= set(sheets)
    assert len(sheets['Results']) == 26 and chem.CB_COL in sheets['Results'].columns
    assert not {'Color', 'Marker', chem.CB_RING, chem.CB_LABEL} & set(sheets['Results'].columns)
    files, notes = hw.save_figures(result, tmp_path / 'figures', dpi=50)
    names = {os.path.basename(f) for f in files}
    assert {'piper.png', 'schoeller.png', 'stiff_1.png', 'stiff_2.png', 'durov.png', 'gibbs.png', 'chadha.png',
            'hfe-d.png'} <= names
    assert any(n.startswith('gallery_') for n in names)
    assert all(os.path.getsize(f) > 1000 for f in files)
    pdf, _ = hw.save_figures(hw.run(chem.example_data().head(6), phreeqc=pp), tmp_path / 'pdf', fmt='pdf')
    assert all(f.endswith('.pdf') for f in pdf)


def test_app_and_script_agree(tmp_path, monkeypatch, result):
    # the app on the example data gives the table that run() gives with the default settings
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=120)
    at.run()
    at.checkbox[0].check().run()
    assert not at.exception
    app_table = at.session_state['result'][1][0]
    ours = result.table[app_table.columns]
    num = app_table.select_dtypes('number').columns
    np.testing.assert_allclose(ours[num].to_numpy(float), app_table[num].to_numpy(float), rtol=1e-12, equal_nan=True)
    other = [c for c in app_table.columns if c not in num and c not in ('Color', 'Marker')]
    assert ours[other].fillna('').astype(str).equals(app_table[other].fillna('').astype(str))


def test_package_installs_and_runs_outside_the_repository(tmp_path):
    # pip install into an empty folder, then import from somewhere else: the data files must come along
    target = tmp_path / 'site'
    done = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--quiet', '--target', str(target),
                           ROOT], capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip('pip could not build the package here: ' + done.stderr[-300:])
    code = ('import hydrochem_workbench as hw, os;'
            'from hydrochem_workbench import maps, plots;'
            'assert os.path.exists(maps.NE_FILE) and os.path.exists(plots.BADGE_ICON);'
            'r = hw.run(hw.chem.example_data());'
            'print(hw.__file__, len(r.table))')
    env = dict(os.environ, PYTHONPATH=str(target))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert out.returncode == 0, out.stderr[-800:]
    assert str(target) in out.stdout and out.stdout.strip().endswith('26')
