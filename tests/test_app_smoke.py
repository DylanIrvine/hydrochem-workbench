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
    # bar chart: every parameter, then every guideline for one of them, colours and options
    for var in at.selectbox(key='opt_gchart_var').options:
        at.selectbox(key='opt_gchart_var').select(var).run()
        assert not at.exception
    at.selectbox(key='opt_gchart_var').select('Cl (mg/L)').run()
    for rule in at.selectbox(key='opt_gchart_rule').options:
        at.selectbox(key='opt_gchart_rule').select(rule).run()
        assert not at.exception
    at.radio(key='opt_gchart_barmode').set_value('One colour').run()
    at.color_picker(key='opt_gchart_linecol').pick('#00ff00').run()
    at.checkbox(key='opt_gchart_hl').check().run()
    at.checkbox(key='opt_gchart_log').check().run()
    at.checkbox(key='opt_gchart_sort').check().run()
    assert not at.exception
    assert any(b.label == 'Download PNG (600 dpi)' for b in at.get('download_button'))
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
    [s for s in at.selectbox if s.label.startswith('Panel letters')][0].select('(a)').run()
    assert not at.exception
    [s for s in at.selectbox if s.label == 'Panel letter position'][0].select(
        'Top left, inside the panel').run()
    assert not at.exception
    go('Custom plot')
    # one group only, then a new colour and marker for it
    groups = [c for c in at.checkbox if c.key and c.key.startswith('grp_show_')]
    assert len(groups) == 3
    for c in groups[1:]:
        c.uncheck()
    at.run()
    assert not at.exception
    at.color_picker(key=groups[0].key.replace('show', 'col')).pick('#123456').run()
    at.selectbox(key=groups[0].key.replace('show', 'mk')).select('Star').run()
    assert not at.exception
    go('Piper')
    go('Diagnostic gallery')
    for c in groups[:1]:
        at.checkbox(key=c.key).uncheck()
    at.run()
    assert any('No groups are ticked' in i.value for i in at.info)
    at.button(key='grp_reset').click().run()
    assert not at.exception
    assert all(at.checkbox(key=c.key).value for c in groups)
    # below-LOR options
    [t for t in at.toggle if t.label.startswith('Replace negative')][0].set_value(False).run()
    [s for s in at.selectbox if s.label == 'Replace with'][0].select('Other').run()
    assert not at.exception
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


def test_settings_survive_loading_another_dataset(tmp_path, monkeypatch):
    # set things up on one dataset, load another, and nothing should be lost
    from core import chem
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    first = chem.example_data().astype(object)
    second = chem.example_data().astype(object).iloc[::2].reset_index(drop=True)
    second['Sample'] = 'B' + second['Sample'].astype(str)
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state['dataset'] = ('site_a.xlsx', first)
    at.run()
    assert not at.exception
    assert any(b.label == 'Load another dataset' for b in at.button)

    lab = first['Label'].iloc[0]
    at.selectbox(key='opt_lor_pick').select('0.1 x LOR').run()
    at.selectbox(key='opt_letters').select('a)').run()
    at.color_picker(key='grp_col_' + lab).pick('#123456').run()
    at.radio(key='opt_view').set_value('Guidelines').run()
    at.selectbox(key='guide_region').select(at.selectbox(key='guide_region').options[0]).run()
    region = at.selectbox(key='guide_region').value
    at.radio(key='opt_view').set_value('Piper').run()
    at.selectbox(key='piper_color').select('pH').run()
    assert not at.exception

    # the button shows the uploader and says what it does; keeping the data goes back
    at.button(key='load_another').click().run()
    assert any('replaces the current data' in c.value for c in at.caption)
    at.button(key='keep_data').click().run()
    assert any(b.label == 'Load another dataset' for b in at.button)

    # a new dataset arrives (as the uploader would leave it)
    at.session_state['dataset'] = ('site_b.xlsx', second)
    at.run()
    assert not at.exception
    assert any('site_b.xlsx' in m.value for m in at.markdown)
    assert at.selectbox(key='opt_lor_pick').value == '0.1 x LOR'
    assert at.selectbox(key='opt_letters').value == 'a)'
    assert at.color_picker(key='grp_col_' + lab).value == '#123456'
    assert at.radio(key='opt_view').value == 'Piper'
    assert at.selectbox(key='piper_color').value == 'pH'
    at.radio(key='opt_view').set_value('Guidelines').run()
    assert at.selectbox(key='guide_region').value == region
    at.radio(key='opt_view').set_value('Export').run()
    assert not at.exception
    assert any(b.label == 'Download results (.xlsx)' for b in at.get('download_button'))
