# the sample map: coordinates, towns, polygon files, extent, zoom limits and the figure.
# No test downloads imagery; the basemap is a made-up image
import io
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from core import chem, maps, plots

gpd = pytest.importorskip('geopandas')
from shapely.geometry import LineString, Polygon


def example_points():
    return maps.locations(chem.example_data())


def styles_for(pts):
    return {lab: (plots.PALETTE[k], plots.MARKERS[k]) for k, lab in enumerate(pd.unique(pts['Label']))}


def square(crs, x0, y0, size):
    return gpd.GeoDataFrame(geometry=[Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])],
                            crs=crs)


def shapefile_parts(gdf, tmp_path, name='zone'):
    gdf.to_file(tmp_path / f'{name}.shp')
    return [maps.Upload(n, (tmp_path / n).read_bytes()) for n in sorted(os.listdir(tmp_path))
            if n.startswith(name + '.')]


def test_template_has_location_columns():
    raw = chem.read_upload(io.BytesIO(chem.template_bytes()))
    assert all(c in raw.columns for c in chem.LOC_COLS)
    assert chem.location_columns(['Sample', 'Latitude', 'LONGITUDE', 'Depth (m)']) == {
        'Lat': 'Latitude', 'Lon': 'LONGITUDE', 'Screen_depth': 'Depth (m)'}
    assert chem.location_columns(['Sample', 'Ca']) == {'Lat': None, 'Lon': None, 'Screen_depth': None}


def test_locations_from_example():
    pts, notes = example_points()
    assert len(pts) == 26 and notes == []
    assert pts['depth'].isna().sum() == 1
    assert pts['lat'].between(-15, -14).all() and pts['lon'].between(132, 134).all()


def test_locations_flags_bad_coordinates():
    d = chem.example_data()
    d.loc[0, 'Lat'] = np.nan                       # missing
    d.loc[1, ['Lat', 'Lon']] = [132.5, -14.5]      # swapped
    d.loc[2, 'Lat'] = 14.6                         # missing minus sign
    pts, notes = maps.locations(d)
    assert len(pts) == 24
    text = ' '.join(notes)
    assert '1 sample(s) have no latitude or longitude' in text
    assert 'look swapped' in text and 'CB02' in text
    assert 'both sides of the equator' in text and 'CB03' in text


def test_locations_flags_eastings_and_far_points():
    d = chem.example_data()
    d.loc[0, ['Lat', 'Lon']] = [8380000.0, 250000.0]
    d.loc[1, 'Lon'] = 152.0
    pts, notes = maps.locations(d)
    text = ' '.join(notes)
    assert 'eastings and northings' in text
    assert 'Far from the other samples' in text and 'CB02' in text


def test_locations_without_columns():
    pts, notes = maps.locations(chem.example_data().drop(columns=['Lat', 'Lon']))
    assert pts.empty and 'no Lat and Lon' in notes[0]


def test_towns_template_round_trip():
    towns = maps.read_towns(io.BytesIO(maps.towns_template_bytes()))
    assert list(towns.columns) == maps.TOWN_COLS and len(towns) == 3
    with pytest.raises(maps.MapError):
        maps.read_towns(io.BytesIO(b'Town,Easting\nA,1\n'))
    other = maps.read_towns(io.BytesIO(b'town,latitude,longitude\nA,-14.5,132.3\nB,bad,132\n,-14,132\n'))
    assert list(other['Name']) == ['A']


def test_extent_and_zoom_limits():
    pts, _ = example_points()
    extent = maps.fit_extent(maps.data_extent(pts))
    w, e, s, n = extent
    assert w < pts['lon'].min() and e > pts['lon'].max() and s < pts['lat'].min() and n > pts['lat'].max()
    # a north-south line of points is widened, an east-west one is heightened
    tall = maps.fit_extent((132.0, 132.01, -16.0, -14.0))
    wide = maps.fit_extent((130.0, 134.0, -14.01, -14.0))
    for ext in (tall, wide):
        ratio = (ext[3] - ext[2]) * 110.57 / maps.extent_width_km(ext)
        assert maps.RATIO[0] - 0.01 <= ratio <= maps.RATIO[1] + 0.01
    # one sample still gives a map
    one = maps.data_extent(pts.iloc[:1])
    assert maps.extent_width_km(one) > 2
    default, lo, hi, counts = maps.zoom_choices(extent)
    assert lo <= default <= hi
    assert counts[default] <= maps.DEFAULT_TILES and counts[hi] <= maps.MAX_TILES
    with pytest.raises(maps.MapError):
        maps.fetch_basemap(extent, hi + 2)


def test_scale_bar_length():
    assert [maps.nice_length(x) for x in (0.7, 1, 3.4, 19.9, 20, 260)] == [0.5, 1, 2, 10, 20, 200]


def test_depth_sizes():
    depth = pd.Series([10.0, 60.0, 110.0, np.nan])
    areas, area_of = maps.depth_sizes(depth, 30)
    assert areas[0] == pytest.approx(30 * maps.SIZE_RANGE[0]) and areas[2] == pytest.approx(30 * maps.SIZE_RANGE[1])
    assert areas[3] == 30                      # no depth: the chosen size
    rev, _ = maps.depth_sizes(depth, 30, reverse=True)
    assert rev[0] == pytest.approx(areas[2]) and rev[2] == pytest.approx(areas[0])
    same, _ = maps.depth_sizes(pd.Series([20.0, 20.0]), 30)
    assert same.nunique() == 1
    vals = maps.depth_legend_values(depth)
    assert vals[0] == 10 and vals[-1] == 110 and vals == sorted(vals)


def test_read_shapefile_zip_in_utm(tmp_path):
    # GDA94 / MGA zone 53, as a .zip
    utm = square(4326, 132.4, -14.7, 0.4).to_crs(28353)
    parts = shapefile_parts(utm, tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for p in parts:
            z.writestr('folder/' + p.name, p.data)
        z.writestr('readme.txt', 'not a layer')
    gdf, info = maps.read_vector([maps.Upload('zone.zip', buf.getvalue())])
    assert gdf.crs.to_epsg() == 4326 and 'EPSG:28353' in info
    assert gdf.total_bounds == pytest.approx([132.4, -14.7, 132.8, -14.3], abs=1e-4)


def test_read_shapefile_parts_and_missing_prj(tmp_path):
    utm = square(4326, 132.4, -14.7, 0.4).to_crs(28353)
    parts = shapefile_parts(utm, tmp_path)
    gdf, _ = maps.read_vector(parts)
    assert gdf.total_bounds == pytest.approx([132.4, -14.7, 132.8, -14.3], abs=1e-4)
    no_prj = [p for p in parts if not p.name.endswith('.prj')]
    with pytest.raises(maps.NeedsCRS):
        maps.read_vector(no_prj)
    gdf, info = maps.read_vector(no_prj, epsg=28353)
    assert gdf.total_bounds == pytest.approx([132.4, -14.7, 132.8, -14.3], abs=1e-4)
    # the wrong code puts the shapes off the Earth, or somewhere else: either way no silent pass
    with pytest.raises(maps.MapError):
        maps.read_vector(no_prj, epsg=4326)
    with pytest.raises(maps.MapError):
        maps.read_vector([p for p in parts if p.name.endswith('.shp')])


@pytest.mark.parametrize('driver,ext', [('GPKG', 'gpkg'), ('GeoJSON', 'geojson'), ('KML', 'kml')])
def test_read_other_formats(tmp_path, driver, ext):
    path = tmp_path / f'zone.{ext}'
    square(4326, 132.4, -14.7, 0.4).to_file(path, driver=driver)
    gdf, _ = maps.read_vector([maps.Upload(path.name, path.read_bytes())])
    assert len(gdf) == 1 and gdf.total_bounds == pytest.approx([132.4, -14.7, 132.8, -14.3], abs=1e-4)


def test_read_vector_rejects_rubbish():
    with pytest.raises(maps.MapError):
        maps.read_vector([maps.Upload('zone.zip', b'not a zip')])
    with pytest.raises(maps.MapError):
        maps.read_vector([maps.Upload('zone.geojson', b'{"type": "nothing"}')])
    with pytest.raises(maps.MapError):
        maps.read_vector([maps.Upload('big.geojson', b' ' * int(maps.MAX_UPLOAD_MB * 1e6 + 1))])


def test_natural_earth_file():
    ne = maps.natural_earth()
    assert set(ne['level']) == {'country', 'state'}
    assert 'Northern Territory' in set(ne['name']) and 'Australia' in set(ne['name'])


def test_inset_extent_and_corner():
    pts, _ = example_points()
    extent = maps.fit_extent(maps.data_extent(pts))
    w, e, s, n = maps.inset_extent(extent, 12)
    assert w < extent[0] and e > extent[1] and s < extent[2] and n > extent[3]
    assert (e - w) == pytest.approx(12 * (extent[1] - extent[0]), rel=0.3)
    nt = maps.natural_earth()
    nt = nt[nt['name'] == 'Northern Territory']
    w, e, s, n = maps.inset_extent(extent, 12, outline=nt)
    b = nt.total_bounds
    assert w <= b[0] and e >= b[2] and s <= b[1] and n >= b[3]
    corner = maps.emptiest_corner(pts, extent, taken=('Bottom right', 'Top right'))
    assert corner in ('Top left', 'Bottom left')


def figure_texts(fig):
    out = [t.get_text() for t in fig.findobj(lambda a: hasattr(a, 'get_text'))]
    return [t for t in out if t]


def test_map_figure_with_everything():
    pts, _ = example_points()
    extent = maps.fit_extent(maps.data_extent(pts, 0.6))
    img = np.full((200, 300, 4), 110, dtype=np.uint8)
    towns = maps.read_towns(io.BytesIO(maps.towns_template_bytes()))
    poly = dict(gdf=square(4326, 132.4, -14.7, 0.4), edge='#FCB711', face='#FCB711', alpha=0.25, width=1.2,
                label='Management zone')
    fig = maps.sample_map(pts, styles_for(pts), extent, basemap=(img, extent), by_depth=True, labels=True,
                          towns=towns, poly=poly, inset=dict(corner='Top left', factor=12), credit='Basemap: test')
    texts = figure_texts(fig)
    assert 'N' in texts and any(t.endswith(' km') for t in texts)
    assert 'Katherine' in texts and 'Management zone' in texts
    assert 'Screen depth' in texts and 'No depth' in texts
    assert 'CB01' in texts
    assert len(fig.axes) == 2                  # map and inset
    assert fig._credit == 'Basemap: test'
    png = plots.fig_png(fig, dpi=80)
    assert png[:4] == b'\x89PNG'
    # the map panel has the same scale east-west and north-south
    ax = fig.axes[0]
    box = ax.get_window_extent()
    km_x = maps.extent_width_km(extent)
    km_y = (extent[3] - extent[2]) * 110.57
    assert (box.height / box.width) == pytest.approx(km_y / km_x, rel=0.01)


def test_map_figure_plain_and_lines():
    pts, _ = example_points()
    extent = maps.fit_extent(maps.data_extent(pts))
    line = gpd.GeoDataFrame(geometry=[LineString([(132.4, -14.7), (133.0, -14.2)])], crs=4326)
    poly = dict(gdf=line, edge='#0089D0', face='#0089D0', alpha=0.0, width=1.0, label='River')
    fig = maps.sample_map(pts, styles_for(pts), extent, poly=poly, scale_corner='Top left',
                          north_corner='Bottom left')
    texts = figure_texts(fig)
    assert 'River' in texts and 'Screen depth' not in texts and len(fig.axes) == 1
    assert not hasattr(fig, '_credit')
    # only one group inside a hand-set extent: the legend lists that group only
    small = maps.fit_extent((132.85, 133.05, -14.6, -14.45))
    fig = maps.sample_map(pts, styles_for(pts), small)
    texts = figure_texts(fig)
    assert 'Saline (synthetic)' in texts and 'Carbonate (synthetic)' not in texts


def test_fetch_basemap_orientation(monkeypatch):
    # contextily and the warp for real, with the tile server replaced: each tile is coloured by its
    # column (red) and row (green), so the image must get redder to the east and greener to the south
    import re
    import mercantile
    from PIL import Image
    from contextily import tile as ctx_tile

    extent = maps.fit_extent((132.37, 133.11, -14.80, -14.16))
    zoom = 11
    tiles = list(mercantile.tiles(extent[0], extent[2], extent[1], extent[3], [zoom]))
    x0, y0 = min(t.x for t in tiles), min(t.y for t in tiles)
    asked = []

    class Reply:
        status_code = 200

        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, timeout=None):
        z, y, x = (int(g) for g in re.search(r'/tile/(\d+)/(\d+)/(\d+)', url).groups())
        asked.append((z, timeout))
        buf = io.BytesIO()
        Image.new('RGB', (256, 256), (40 * (x - x0) + 20, 40 * (y - y0) + 20, 0)).save(buf, format='PNG')
        return Reply(buf.getvalue())

    monkeypatch.setattr(ctx_tile.requests, 'get', fake_get)
    img, (w, e, s, n) = maps.fetch_basemap(extent, zoom)
    assert len(asked) == len(tiles) and all(a == (zoom, maps.TILE_TIMEOUT) for a in asked)
    # the image covers the map
    assert w <= extent[0] and e >= extent[1] and s <= extent[2] and n >= extent[3]
    assert img.ndim == 3 and img.shape[2] == 4
    rows, cols = img.shape[:2]
    mid_r, mid_c = rows // 2, cols // 2
    assert img[mid_r, cols - 20, 0] > img[mid_r, 20, 0]      # east is redder
    assert img[rows - 20, mid_c, 1] > img[20, mid_c, 1]      # the bottom of the image is south
