# runs the app on the example data and steps through the options
import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app.py')


def test_app_runs_on_example_data(tmp_path, monkeypatch):
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    at.checkbox[0].check().run()   # example data
    assert not at.exception
    assert len(at.error) == 0
    assert any('filled' in i.value for i in at.info)
    assert any('sessions that have processed data: 1' in c.value for c in at.caption)
    assert [e.label for e in at.expander][:2] == ['**Privacy and data handling**', '**How to cite and references**']

    def go(view):
        at.radio[0].set_value(view).run()
        assert not at.exception

    for label in ['From charge balance', 'Fixed value']:
        [s for s in at.selectbox if s.label == 'Missing alkalinity'][0].select(label).run()
        assert not at.exception
    go('Guidelines')
    ms = [m for m in at.multiselect if m.label == 'Guideline sets'][0]
    ms.set_value(ms.options).run()
    assert not at.exception
    go('Piper')
    go('Classical diagrams')
    for diagram in ['Stiff', 'Durov', 'Gibbs', 'Chadha', 'HFE-D']:
        [s for s in at.selectbox if s.label == 'Diagram'][0].select(diagram).run()
        assert not at.exception
    go('Diagnostic gallery')
    plot_set = [s for s in at.selectbox if s.label == 'Plot set'][0]
    for group in plot_set.options:
        plot_set.select(group).run()
        assert not at.exception
    [s for s in at.selectbox if s.key == 'gallery_color'][0].select('pH').run()
    assert not at.exception
    go('Custom plot')
    go('Export')
    assert any(b.label == 'Build a zip of every figure' for b in at.button)
    # should still be 1
    assert any('sessions that have processed data: 1' in c.value for c in at.caption)


def test_app_recovers_from_stale_core_modules(tmp_path, monkeypatch):
    # what Streamlit Cloud did after an update: new app.py, old core modules still in memory
    import core
    from core import guidelines
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    monkeypatch.setattr(core, 'VERSION', '0.6')
    monkeypatch.delattr(guidelines, 'REGIONAL_NAME')
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.checkbox[0].check().run()
    at.radio[0].set_value('Guidelines').run()
    assert not at.exception
