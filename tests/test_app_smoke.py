# runs the app on the example data and steps through the options
import os

from streamlit.testing.v1 import AppTest


def plots_gallery_sets():
    from hydrochem_workbench import plots
    return list(plots.GALLERY)

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
    [b for b in at.button if b.label == 'Build a zip of every figure'][0].click().run()
    assert not at.exception
    names = at.session_state['zip_contents']
    for want in ['classical/piper.png', 'classical/schoeller.png', 'classical/stiff_1.png', 'classical/durov.png',
                 'classical/gibbs.png', 'classical/chadha.png', 'classical/hfe-d.png', 'custom plots/custom_plot.png',
                 'custom plots/guideline_bar_chart.png']:
        assert want in names, want
    assert sum(n.startswith('gallery/') for n in names) >= len(plots_gallery_sets())
    # should still be 1
    assert any('sessions that have processed data: 1' in c.value for c in at.caption)


def test_app_recovers_from_stale_core_modules(tmp_path, monkeypatch):
    # what Streamlit Cloud did after an update: new app.py, old core modules still in memory
    import hydrochem_workbench
    from hydrochem_workbench import guidelines
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    monkeypatch.setattr(hydrochem_workbench, 'VERSION', '0.6')
    monkeypatch.delattr(guidelines, 'REGIONAL_NAME')
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.checkbox[0].check().run()
    at.radio[0].set_value('Guidelines').run()
    assert not at.exception


def test_settings_survive_loading_another_dataset(tmp_path, monkeypatch):
    # set things up on one dataset, load another, and nothing should be lost
    from hydrochem_workbench import chem
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


def test_app_with_uncertainty(tmp_path, monkeypatch):
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.checkbox(key='opt_example').check().run()
    at.checkbox(key='opt_unc').check().run()
    at.number_input(key='opt_unc_n').set_value(20).run()
    assert not at.exception
    # the table and the explanations only when asked for
    assert not any(e.label.startswith('**How the default') for e in at.expander)
    at.color_picker(key='opt_unc_colour').pick('#aa0000').run()
    assert not at.exception
    at.checkbox(key='opt_unc_show').check().run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert '**How the default values were determined**' in labels
    assert '**How to estimate your own values**' in labels
    assert any(s.value == 'Uncertainty inputs' for s in at.subheader)
    at.button(key='unc_reset').click().run()
    assert not at.exception
    at.checkbox(key='opt_unc_show').uncheck().run()
    assert not at.exception
    for view in ['Data and checks', 'Guidelines', 'Piper', 'Classical diagrams', 'Diagnostic gallery',
                 'Custom plot', 'Export']:
        at.radio(key='opt_view').set_value(view).run()
        assert not at.exception
    at.radio(key='opt_view').set_value('Piper').run()
    for mode in ['Outlines', 'Off', 'Clouds']:
        at.radio(key='opt_piper_unc').set_value(mode).run()
        assert not at.exception
    # off and on again keeps working
    at.checkbox(key='opt_unc').uncheck().run()
    assert not at.exception
    at.checkbox(key='opt_unc').check().run()
    assert not at.exception


def test_wqchartpy_uncertainty(tmp_path, monkeypatch):
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.checkbox(key='opt_example').check().run()
    at.checkbox(key='opt_unc').check().run()
    at.number_input(key='opt_unc_n').set_value(20).run()
    at.radio(key='opt_view').set_value('Classical diagrams').run()
    for diagram in ['Durov', 'Gibbs', 'Chadha', 'HFE-D']:
        [s for s in at.selectbox if s.label == 'Diagram'][0].select(diagram).run()
        for mode in ['Outlines', 'Off', 'Clouds']:
            at.radio(key='opt_wq_unc').set_value(mode).run()
            assert not at.exception


def test_further_analyses(tmp_path, monkeypatch):
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.checkbox(key='opt_example').check().run()
    at.checkbox(key='opt_means').check().run()
    for view in ['Mixing', 'PCA', 'Clusters', 'Group comparison']:
        at.radio(key='opt_view').set_value(view).run()
        assert not at.exception
        assert not at.error
    # three end members with two tracers, on log axes
    at.radio(key='opt_view').set_value('Mixing').run()
    at.radio(key='opt_mix_n').set_value(3).run()
    assert any('at least 2 tracer' in e.value for e in at.error)
    at.multiselect(key='opt_mix_tracers').set_value(['Cl (mmol/L)', 'Br (mmol/L)']).run()
    at.checkbox(key='opt_mix_log').check().run()
    assert not at.exception and not at.error
    at.radio(key='opt_view').set_value('Export').run()
    [b for b in at.button if 'zip of every figure' in b.label][0].click().run()
    assert not at.exception
    names = at.session_state['zip_contents']
    for f in ['mixing_1.png', 'mixing_deviations.png', 'pca_biplot.png', 'cluster_dendrogram.png',
              'group_boxplots_1.png']:
        assert f'further analyses/{f}' in names


def test_mixing_with_uncertainty(tmp_path, monkeypatch):
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.checkbox(key='opt_example').check().run()
    at.checkbox(key='opt_unc').check().run()
    at.number_input(key='opt_unc_n').set_value(20).run()
    at.radio(key='opt_view').set_value('Mixing').run()
    assert not at.exception and not at.error
    assert any('differs from 0' in c.value for c in at.caption)


def test_large_dataset_warning(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from hydrochem_workbench import chem
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    small = chem.example_data()
    rng = np.random.default_rng(2)
    big = pd.concat([small] * 10, ignore_index=True)
    big['Sample'] = [f'X{k:03d}' for k in range(len(big))]
    for c in ['Ca', 'Mg', 'Na', 'Cl']:
        big[c] = pd.to_numeric(big[c], errors='coerce') * rng.uniform(0.9, 1.1, len(big))
    monkeypatch.setattr(chem, 'example_data', lambda: big)
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    at.checkbox(key='opt_example').check().run()
    assert any('Above 200 samples' in w.value for w in at.warning)
    for view in ['Mixing', 'PCA', 'Clusters', 'Group comparison']:
        at.radio(key='opt_view').set_value(view).run()
        assert not at.exception


def test_map_view(tmp_path, monkeypatch):
    # the Map view on the example data, with a made-up image in place of the imagery download
    import numpy as np
    from hydrochem_workbench import maps
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    calls = []

    def fake_basemap(extent, zoom):
        calls.append(zoom)
        return np.full((120, 160, 4), 120, dtype=np.uint8), extent

    monkeypatch.setattr(maps, 'fetch_basemap', fake_basemap)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.checkbox[0].check().run()
    at.radio[0].set_value('Map').run()
    assert not at.exception and len(at.error) == 0
    assert len(calls) == 1
    assert any(b.label == 'Download the towns template' for b in at.get('download_button'))
    assert any('Esri' in c.value for c in at.caption)
    # a change of style does not download the imagery again
    at.checkbox(key='opt_map_depth').check().run()
    at.checkbox(key='opt_map_reverse').check().run()
    at.checkbox(key='opt_map_labels').check().run()
    at.slider(key='opt_map_size').set_value(60).run()
    at.text_input(key='opt_map_legend_depths').set_value('10, 50, 100, 200, 300').run()
    assert not at.exception and any('first 4' in w.value for w in at.warning)
    at.text_input(key='opt_map_legend_depths').set_value('10, 50, 100').run()
    at.slider(key='opt_map_edge_img').set_value(50).run()
    at.slider(key='opt_map_edge_width').set_value(0.0).run()
    assert not at.exception
    for corner in at.selectbox(key='opt_map_inset').options:
        at.selectbox(key='opt_map_inset').select(corner).run()
        assert not at.exception
    assert len(calls) == 1
    # a higher zoom does, and warns
    zoom = [s for s in at.slider if s.label == 'Imagery zoom level'][0]
    zoom.set_value(zoom.max).run()
    assert not at.exception and len(calls) == 2 and calls[-1] == zoom.max
    assert any('tiles' in w.value for w in at.warning)
    # extent by hand
    at.checkbox(key='opt_map_manual').check().run()
    at.number_input(key='opt_map_w').set_value(132.0).run()
    at.number_input(key='opt_map_e').set_value(131.0).run()     # east of west: falls back, with a warning
    assert not at.exception and any('East has to be greater' in w.value for w in at.warning)
    at.number_input(key='opt_map_e').set_value(133.5).run()
    assert not at.exception
    # white background: nothing is downloaded
    n = len(calls)
    at.selectbox(key='opt_map_base').select(maps.PLAIN).run()
    assert not at.exception and len(calls) == n
    assert not any('Esri' in c.value for c in at.caption)
    # the map is in the figure zip and its settings are in the results
    at.radio[0].set_value('Export').run()
    [b for b in at.button if b.label == 'Build a zip of every figure'][0].click().run()
    assert not at.exception
    assert 'map/sample_map.png' in at.session_state['zip_contents']


def test_map_view_when_imagery_fails(tmp_path, monkeypatch):
    # no connection to the map server: a white map and a note, and no retry on every rerun
    from hydrochem_workbench import maps
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    calls = []

    def no_basemap(extent, zoom):
        calls.append(zoom)
        raise ConnectionError('no route to the map server')

    monkeypatch.setattr(maps, 'fetch_basemap', no_basemap)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.checkbox[0].check().run()
    at.radio[0].set_value('Map').run()
    assert not at.exception
    assert any('could not be fetched' in i.value for i in at.info)
    at.checkbox(key='opt_map_labels').check().run()
    assert len(calls) == 1
    [b for b in at.button if b.label == 'Try the imagery again'][0].click().run()
    assert not at.exception and len(calls) == 2


def test_map_view_without_coordinates(tmp_path, monkeypatch):
    from hydrochem_workbench import chem
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    plain = chem.example_data().drop(columns=chem.LOC_COLS)
    monkeypatch.setattr(chem, 'example_data', lambda: plain)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.checkbox[0].check().run()
    at.radio[0].set_value('Map').run()
    assert not at.exception
    assert any('no Lat and Lon' in i.value for i in at.info)


def test_map_layers(tmp_path, monkeypatch):
    # towns, a polygon and an inset outline, and a shapefile with no .prj waiting for its EPSG code
    import io
    import os
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Polygon
    from hydrochem_workbench import maps
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    monkeypatch.setattr(maps, 'fetch_basemap', lambda extent, zoom: (np.full((120, 160, 4), 120, dtype=np.uint8), extent))
    zone = gpd.GeoDataFrame(geometry=[Polygon([(132.4, -14.7), (132.8, -14.7), (132.8, -14.3), (132.4, -14.3)])],
                            crs=4326)
    zone.to_crs(28353).to_file(tmp_path / 'zone.shp')
    parts = [maps.Upload(n, (tmp_path / n).read_bytes()) for n in os.listdir(tmp_path)
             if n.startswith('zone.') and not n.endswith('.prj')]

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.checkbox[0].check().run()
    at.session_state['map_towns'] = ('towns.csv (3 town(s))', maps.read_towns(io.BytesIO(maps.towns_template_bytes())))
    at.session_state['map_poly_0_pending'] = (parts, 'zone.shp does not say what its coordinates are')
    nt = maps.natural_earth()
    at.session_state['map_outline'] = ('nt.gpkg', nt[nt['name'] == 'Northern Territory'])
    at.radio[0].set_value('Map').run()
    assert not at.exception
    assert any('does not say what its coordinates are' in w.value for w in at.warning)
    at.number_input(key='map_poly_0_epsg').set_value(28353).run()
    at.button(key='map_poly_0_use').click().run()
    assert not at.exception and len(at.error) == 0
    info, gdf = at.session_state['map_poly_0']
    assert 'EPSG:28353' in info and abs(gdf.total_bounds[0] - 132.4) < 1e-3
    at.color_picker(key='opt_map_poly0_edge').pick('#000000').run()
    at.slider(key='opt_map_poly0_alpha').set_value(0.0).run()
    at.text_input(key='opt_map_poly0_label').set_value('Management zone').run()
    assert not at.exception
    at.button(key='map_poly_0_remove').click().run()
    assert not at.exception and 'map_poly_0' not in at.session_state


def test_map_uploads(tmp_path, monkeypatch):
    # the uploaders themselves: a towns csv, a zipped shapefile, and a file that is not a layer
    import io
    import zipfile
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Polygon
    from hydrochem_workbench import maps
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    monkeypatch.setattr(maps, 'fetch_basemap', lambda extent, zoom: (np.full((120, 160, 4), 120, dtype=np.uint8), extent))
    zone = gpd.GeoDataFrame(geometry=[Polygon([(132.4, -14.7), (132.8, -14.7), (132.8, -14.3), (132.4, -14.3)])],
                            crs=4326)
    zone.to_crs(28353).to_file(tmp_path / 'zone.shp')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for p in tmp_path.glob('zone.*'):
            z.write(p, p.name)

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.checkbox[0].check().run()
    at.radio[0].set_value('Map').run()
    at.file_uploader(key='map_towns_up_0').upload('towns.csv', maps.towns_template_bytes(), 'text/csv').run()
    assert not at.exception and at.session_state['map_towns'][0] == 'towns.csv (3 town(s))'
    at.file_uploader(key='map_poly_0_up_0').upload('readme.geojson', b'{"type": "nothing"}').run()
    assert not at.exception and len(at.error) == 1 and 'map_poly_0' not in at.session_state
    at.file_uploader(key='map_poly_0_up_0').upload('zone.zip', buf.getvalue(), 'application/zip').run()
    assert not at.exception and 'EPSG:28353' in at.session_state['map_poly_0'][0]
    # a second and a third layer (lines), then the first is removed and the others stay where they are
    from shapely.geometry import LineString
    for k, name in ((1, 'rivers'), (2, 'roads')):
        path = tmp_path / f'{name}.geojson'
        gpd.GeoDataFrame(geometry=[LineString([(132.3, -14.8 + 0.1 * k), (133.0, -14.2)])], crs=4326).to_file(
            path, driver='GeoJSON')
        at.file_uploader(key=f'map_poly_{k}_up_0').upload(path.name, path.read_bytes()).run()
        assert not at.exception and name in at.session_state[f'map_poly_{k}'][0]
    at.text_input(key='opt_map_poly1_label').set_value('Rivers').run()
    at.color_picker(key='opt_map_poly2_edge').pick('#444444').run()
    assert not at.exception
    assert not any(f.key.startswith('map_poly_') for f in at.get('file_uploader'))     # all three slots are full
    at.button(key='map_poly_0_remove').click().run()
    assert not at.exception and 'map_poly_0' not in at.session_state and 'map_poly_2' in at.session_state
    at.radio[0].set_value('Export').run()
    assert not at.exception
    at.radio[0].set_value('Map').run()
    at.button(key='map_towns_remove').click().run()
    assert not at.exception and 'map_towns' not in at.session_state


def test_charge_balance_options(tmp_path, monkeypatch):
    # mark (the default), exclude and no action; then judged on the uncertainty interval
    import numpy as np
    from hydrochem_workbench import chem, maps
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    monkeypatch.setattr(maps, 'fetch_basemap', lambda extent, zoom: (np.full((60, 80, 4), 120, dtype=np.uint8), extent))
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.checkbox[0].check().run()
    assert not at.exception
    assert any('4 sample(s) have a charge balance error beyond 5 %' in i.value and 'marked with a ring' in i.value
               for i in at.info)
    assert at.color_picker(key='opt_cb_col').value.lower() not in [
        c.value.lower() for c in at.color_picker if c.key.startswith('grp_col_')]
    for view in ('Piper', 'Diagnostic gallery', 'Custom plot', 'Map', 'PCA'):
        at.radio[0].set_value(view).run()
        assert not at.exception
    at.radio[0].set_value('Data and checks').run()
    means_marked = [d.value for d in at.dataframe][1].copy()

    at.selectbox(key='opt_cb_action').select('Exclude from the plots and analyses').run()
    assert not at.exception and any('left off every plot' in i.value for i in at.info)
    assert not [c for c in at.color_picker if c.key == 'opt_cb_col']
    tables = [d.value for d in at.dataframe]
    assert len(tables[0]) == 26                                  # every sample is still in the summary
    assert (tables[0][chem.CB_COL] == chem.CB_BEYOND).sum() == 4
    assert (tables[0][chem.CB_COL] == chem.CB_NA).sum() == 1     # the sample with a filled alkalinity
    sand = lambda t: t[t['Label'].str.startswith('Sandstone')].iloc[0]
    assert sand(tables[1])['Samples'] == sand(means_marked)['Samples'] - 4
    assert any('excluded for their charge balance' in c.value for c in at.caption)
    for view in ('Piper', 'Map', 'Clusters', 'Export'):
        at.radio[0].set_value(view).run()
        assert not at.exception

    at.selectbox(key='opt_cb_action').select('No action').run()
    assert not at.exception and any('No action is taken' in i.value for i in at.info)
    at.number_input(key='opt_cb_limit').set_value(25.0).run()
    assert not at.exception and not any('charge balance error beyond' in i.value for i in at.info)

    # with the uncertainty analysis, a sample is beyond the limit only if its whole interval is
    at.number_input(key='opt_cb_limit').set_value(5.0).run()
    at.selectbox(key='opt_cb_action').select('Mark on the plots').run()
    at.checkbox(key='opt_unc').check().run()
    at.number_input(key='opt_unc_n').set_value(40).run()
    at.selectbox(key='opt_cb_basis').select('The uncertainty interval').run()
    assert not at.exception
    note = [i.value for i in at.info if 'charge balance error beyond' in i.value]
    assert note and 'even allowing for measurement uncertainty' in note[0]
    assert int(note[0].split()[0]) <= 4
    at.radio[0].set_value('Piper').run()
    assert not at.exception


def test_descriptive_statistics(tmp_path, monkeypatch):
    from hydrochem_workbench import chem
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.checkbox[0].check().run()
    assert not at.exception
    stats = [d.value for d in at.dataframe if list(d.value.columns[:2]) == ['Group', 'Variable']]
    assert len(stats) == 1 and set(stats[0]['Group']) == {chem.ALL_SAMPLES} | set(chem.example_data()['Label'])
    at.multiselect(key='opt_stat_vars').set_value(['pH', 'SAR', 'RSC (meq/L)', 'Hardness (°dH)']).run()
    at.checkbox(key='opt_stat_groups').uncheck().run()
    assert not at.exception
    stats = [d.value for d in at.dataframe if list(d.value.columns[:2]) == ['Group', 'Variable']][0]
    assert list(stats['Variable']) == ['pH', 'SAR', 'RSC (meq/L)', 'Hardness (°dH)']
    # nitrite shares the as-N switch
    assert [s for s in at.selectbox if s.label == 'NO3, NO2 and NH4 columns hold']


def test_quality_views_and_databases(tmp_path, monkeypatch):
    # irrigation, scaling, analysis checks, outliers, below-LOR statistics, the CCME index, PCA outliers
    # and the other PHREEQC databases
    import numpy as np
    from hydrochem_workbench import chem, maps
    monkeypatch.setenv('HYDROCHEM_RUN_COUNT_PATH', str(tmp_path / 'count.json'))
    monkeypatch.setattr(maps, 'fetch_basemap', lambda extent, zoom: (np.full((60, 80, 4), 120, dtype=np.uint8), extent))
    data = chem.example_data().astype(object)
    data.loc[[0, 1, 2, 5, 9], 'Br'] = '<0.5'
    data['EC'] = [500 + 40 * k for k in range(len(data))]
    monkeypatch.setattr(chem, 'example_data', lambda: data.copy())

    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.checkbox[0].check().run()
    assert not at.exception and len(at.error) == 0
    heads = [h.value for h in at.subheader]
    for name in ('Statistics with values below the LOR', 'Analysis checks', 'Outliers, one variable at a time'):
        assert name in heads
    tables = [d.value for d in at.dataframe]
    assert any('Method' in t.columns and set(t['Variable']) == {'Br'} for t in tables)
    assert any('EC calculated / measured' in t.columns for t in tables)
    assert any(list(t.columns[:3]) == ['Sample', 'Label', 'Variable'] and 'Modified z' in t.columns for t in tables)

    def go(view):
        at.radio[0].set_value(view).run()
        assert not at.exception and len(at.error) == 0

    go('Guidelines')
    assert 'CCME water quality index' in [h.value for h in at.subheader]
    assert any('CCME WQI' in d.value.columns for d in at.dataframe)
    assert any('debated' in e.label for e in at.expander)
    go('Irrigation')
    assert any({'USSL class', 'Wilcox diagram field', 'Wilcox table grade'} <= set(d.value.columns) for d in at.dataframe)
    at.radio(key='opt_irr_plot').set_value('Wilcox diagram (EC and percent sodium)').run()
    assert not at.exception
    go('Scaling')
    assert any('Ryznar index' in d.value.columns for d in at.dataframe)
    go('PCA')
    assert at.checkbox(key='opt_pca_mark').value is True
    at.checkbox(key='opt_pca_drop').check().run()
    assert not at.exception
    at.checkbox(key='opt_pca_mark').uncheck().run()
    assert not at.exception

    go('Export')
    [b for b in at.button if b.label == 'Build a zip of every figure'][0].click().run()
    assert not at.exception
    names = at.session_state['zip_contents']
    for name in ('irrigation/ussl.png', 'irrigation/wilcox.png', 'irrigation/langelier_index.png',
                 'checks/analysis_checks.png', 'checks/ccme_wqi.png'):
        assert name in names, name

    # the EC unit: the EC of this dataset is in uS/cm, so any other choice is warned about and names uS/cm
    assert not [w for w in at.warning if 'Measured EC' in w.value]
    for ec_unit in ('mS/cm', 'mS/m'):
        at.selectbox(key='opt_ec_unit').select(ec_unit).run()
        assert not at.exception and len(at.error) == 0
        assert [w for w in at.warning if 'Measured EC' in w.value and 'look like µS/cm' in w.value]
    at.selectbox(key='opt_ec_unit').select('uS/cm').run()
    assert not at.exception and not [w for w in at.warning if 'Measured EC' in w.value]

    # the other databases
    for db in ('wateq4f.dat', 'pitzer.dat', 'phreeqc.dat'):
        at.selectbox(key='opt_database').select(db).run()
        assert not at.exception and len(at.error) == 0
        left_out = [i.value for i in at.info if 'left out of the speciation' in i.value]
        assert bool(left_out) == (db == 'pitzer.dat')
        saline = [w.value for w in at.warning if 'ionic strength above' in w.value]
        assert not (saline and db == 'pitzer.dat')
