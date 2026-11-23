# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# core/maps.py
# sample location map: points by group (size by screen depth if wanted), satellite
# imagery behind, towns, up to three polygon or line layers, a locator inset, scale bar and north arrow
#
# the map is drawn in longitude and latitude (EPSG:4326). geopandas and contextily
# are imported inside the functions that need them, so the rest of the app does not
# load them
#
# core/data/natural_earth_50m.gpkg: country and state outlines from Natural Earth
# (1:50m, public domain, naturalearthdata.com), coordinates rounded to 0.01 degrees

# %%
import io
import math
import os
import tempfile
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

from core import chem, plots

CM = plots.CM

# %% CONFIG
NE_FILE = Path(__file__).resolve().parent / 'data' / 'natural_earth_50m.gpkg'

IMAGERY = 'Satellite imagery (Esri World Imagery)'
PLAIN = 'None (white background)'
BASEMAPS = [IMAGERY, PLAIN]

DEFAULT_TILES = 40      # the starting zoom is the highest that needs no more tiles than this
WARN_TILES = 60         # the app warns above this
MAX_TILES = 100         # and does not go above this
MAX_ZOOM = 18
TILE_TIMEOUT = 15       # seconds, for each tile

MAX_UPLOAD_MB = 20      # polygon and outline files
MAX_VERTICES = 200_000  # simplified above this

CORNERS = ['Top left', 'Top right', 'Bottom left', 'Bottom right']
FIG_W = 17.0            # cm
MAP_W = 10.4            # cm
MAP_LEFT = 1.5          # cm
RATIO = (0.45, 1.25)    # height / width of the map panel is kept in this range
SIZE_RANGE = (0.4, 3.0)  # marker area of the smallest and largest depth, times the chosen size
MAX_LEGEND_DEPTHS = 4
MAX_LAYERS = 3           # polygon or line layers on the map
NORTH_SIZE = 13          # north arrow triangle, points

TOWN_COLS = ['Name', 'Lat', 'Lon']
POLY_TYPES = ['zip', 'shp', 'shx', 'dbf', 'prj', 'cpg', 'gpkg', 'geojson', 'json', 'kml']
_SHP_PARTS = ('.shp', '.shx', '.dbf', '.prj', '.cpg')
_VECTOR = ('.shp', '.gpkg', '.geojson', '.json', '.kml')


class MapError(ValueError):
    pass


class Upload:
    # an uploaded file kept in memory (name and bytes), so it can be read again later
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def getvalue(self):
        return self.data


class NeedsCRS(MapError):
    # the file has no coordinate reference system, so the user has to give one
    pass


# %% sample locations
def locations(v):
    # Sample, Label, lat, lon, depth for the rows that can be mapped, and notes for the user
    cols = chem.location_columns(v.columns)
    notes = []
    empty = pd.DataFrame(columns=['Sample', 'Label', 'lat', 'lon', 'depth'])
    if cols['Lat'] is None or cols['Lon'] is None:
        return empty, ['There are no Lat and Lon columns in this dataset. Add them (decimal degrees) to map '
                       'the samples; the template has them.']
    pts = pd.DataFrame({'Sample': v['Sample'].astype(str), 'Label': v['Label'],
                        'lat': pd.to_numeric(v[cols['Lat']], errors='coerce'),
                        'lon': pd.to_numeric(v[cols['Lon']], errors='coerce')})
    pts['depth'] = pd.to_numeric(v[cols['Screen_depth']], errors='coerce') if cols['Screen_depth'] else np.nan

    has = pts['lat'].notna() & pts['lon'].notna()
    if (~has).any():
        notes.append(f'{int((~has).sum())} sample(s) have no latitude or longitude and are not on the map.')
    pts = pts[has]
    bad = (pts['lat'].abs() > 90) | (pts['lon'].abs() > 180)
    if bad.any():
        swapped = bad & (pts['lon'].abs() <= 90) & (pts['lat'].abs() <= 180)
        msg = (f'{int(bad.sum())} sample(s) have coordinates that are not decimal degrees (latitude beyond 90 or '
               f'longitude beyond 180) and are not on the map: ' + _some(pts.loc[bad, 'Sample']) + '.')
        if swapped.sum() > 0.5 * bad.sum():
            msg += ' Latitude and longitude look swapped.'
        elif (pts.loc[bad, ['lat', 'lon']].abs().max(axis=1) > 1000).any():
            msg += ' They look like eastings and northings; convert them to decimal degrees first.'
        notes.append(msg)
    pts = pts[~bad]
    if pts.empty:
        return empty, notes

    if (pts['lat'] > 0).any() and (pts['lat'] < 0).any():
        notes.append('Latitudes are on both sides of the equator. If that is not right, check for a missing '
                     'minus sign: ' + _some(pts.loc[_minority(pts['lat'] > 0), 'Sample']) + '.')
    if (pts['lon'] > 0).any() and (pts['lon'] < 0).any():
        notes.append('Longitudes are on both sides of the prime meridian. If that is not right, check for a '
                     'missing minus sign: ' + _some(pts.loc[_minority(pts['lon'] > 0), 'Sample']) + '.')
    # a point a long way from the rest stretches the map
    if len(pts) >= 4:
        d = np.hypot(pts['lat'] - pts['lat'].median(), pts['lon'] - pts['lon'].median())
        far = d > max(1.0, 10 * d.median())
        if far.any() and far.sum() <= 0.25 * len(pts):
            notes.append('Far from the other samples (check the coordinates, or set the map extent by hand): '
                         + _some(pts.loc[far, 'Sample']) + '.')
    if pts['depth'].notna().any() and (pts['depth'] < 0).any():
        notes.append('Some screen depths are negative. Depth is in m below surface, so it should be positive.')
    return pts.reset_index(drop=True), notes


def _some(names, n=8):
    names = [str(x) for x in names]
    return ', '.join(names[:n]) + (' ...' if len(names) > n else '')


def _minority(mask):
    return mask if mask.sum() <= (~mask).sum() else ~mask


# %% towns
def towns_template_bytes():
    example = pd.DataFrame({'Name': ['Katherine', 'Mataranka', 'Pine Creek'],
                            'Lat': [-14.465, -14.923, -13.821], 'Lon': [132.264, 133.066, 131.834]})
    return example.to_csv(index=False).encode()


def read_towns(file):
    # csv (or xlsx) with Name, Lat, Lon in decimal degrees
    name = (getattr(file, 'name', '') or '').lower()
    raw = pd.read_excel(file, dtype=object) if name.endswith(('.xlsx', '.xls')) else pd.read_csv(file, dtype=object)
    raw.columns = [str(c).strip() for c in raw.columns]
    cols = chem.location_columns(raw.columns)
    lower = {c.lower(): c for c in raw.columns}
    name_col = next((lower[k] for k in ('name', 'town', 'place', 'label') if k in lower), None)
    if name_col is None or cols['Lat'] is None or cols['Lon'] is None:
        raise MapError('The towns file needs the columns Name, Lat and Lon. Use the template.')
    out = pd.DataFrame({'Name': raw[name_col].fillna('').astype(str).str.strip(),
                        'Lat': pd.to_numeric(raw[cols['Lat']], errors='coerce'),
                        'Lon': pd.to_numeric(raw[cols['Lon']], errors='coerce')})
    out = out[out['Lat'].notna() & out['Lon'].notna() & (out['Name'] != '') & (out['Name'] != 'nan')]
    out = out[(out['Lat'].abs() <= 90) & (out['Lon'].abs() <= 180)]
    if out.empty:
        raise MapError('No rows in the towns file have a name with a latitude and longitude in decimal degrees.')
    return out.reset_index(drop=True)


# %% polygons
def read_vector(files, epsg=None):
    # one polygon (or line) layer from the uploaded file(s), in longitude and latitude
    # files: a .zip holding a shapefile, the parts of a shapefile, or one .gpkg/.geojson/.kml
    # epsg: only used when the file does not say what its coordinates are
    # returns (GeoDataFrame with a geometry column only, short description)
    import geopandas as gpd

    files = list(files) if isinstance(files, (list, tuple)) else [files]
    if not files:
        raise MapError('No file.')
    with tempfile.TemporaryDirectory() as tmp:
        total = 0
        for f in files:
            data = f.getvalue() if hasattr(f, 'getvalue') else f.read()
            fname = os.path.basename(getattr(f, 'name', 'upload'))
            total += len(data)
            if total > MAX_UPLOAD_MB * 1e6:
                raise MapError(f'The file is larger than {MAX_UPLOAD_MB} MB. Simplify it or cut it down first.')
            if fname.lower().endswith('.zip'):
                _unzip(data, tmp)
            else:
                with open(os.path.join(tmp, fname), 'wb') as out:
                    out.write(data)
        found = sorted(n for n in os.listdir(tmp) if n.lower().endswith(_VECTOR))
        if not found:
            raise MapError('No shapefile (.shp), GeoPackage, GeoJSON or KML was found in the upload.')
        path = os.path.join(tmp, found[0])
        if path.lower().endswith('.shp'):
            stem = os.path.splitext(found[0])[0].lower()
            have = {os.path.splitext(n)[1].lower() for n in os.listdir(tmp)
                    if os.path.splitext(n)[0].lower() == stem}
            missing = [e for e in ('.shx', '.dbf') if e not in have]
            if missing:
                raise MapError('A shapefile is several files. Missing: ' + ', '.join(missing) +
                               '. Upload them together (with the .prj), or as one .zip.')
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            raise MapError(f'Could not read {found[0]}: {exc}')
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise MapError(f'{found[0]} has no shapes in it.')
    if gdf.crs is None:
        if epsg is None:
            raise NeedsCRS(f'{found[0]} does not say what its coordinates are (no .prj). Enter the EPSG code.')
        try:
            gdf = gdf.set_crs(epsg=int(epsg))
        except Exception:
            raise MapError(f'EPSG:{epsg} is not a known coordinate reference system.')
    source_crs = gdf.crs.to_string()
    try:
        gdf = gdf.to_crs(epsg=4326)
    except Exception as exc:
        raise MapError(f'Could not convert {found[0]} from {source_crs} to longitude and latitude: {exc}')
    gdf = gpd.GeoDataFrame(geometry=gdf.geometry.reset_index(drop=True), crs=4326)
    b = gdf.total_bounds
    if not np.all(np.isfinite(b)) or b[0] < -180.5 or b[2] > 180.5 or b[1] < -90.5 or b[3] > 90.5:
        raise MapError(f'After conversion from {source_crs} the shapes are not on the Earth. The coordinate '
                       'reference system of the file is probably wrong.')
    n_vert = int(gdf.geometry.count_coordinates().sum())
    if n_vert > MAX_VERTICES:
        # drop detail that could not be seen at the size of the figure
        gdf['geometry'] = gdf.geometry.simplify(max(b[2] - b[0], b[3] - b[1]) / 5000, preserve_topology=True)
    kinds = sorted({t.replace('Multi', '') for t in gdf.geom_type.unique()})
    info = f"{found[0]}: {len(gdf)} shape(s) ({', '.join(kinds).lower()}), converted from {source_crs}"
    return gdf, info


def _unzip(data, folder):
    # only the files a layer is made of, into one flat folder
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise MapError('That .zip could not be opened.')
    size = 0
    for item in z.infolist():
        name = os.path.basename(item.filename)
        if item.is_dir() or not name.lower().endswith(_SHP_PARTS + _VECTOR) or name.startswith('.'):
            continue
        size += item.file_size
        if size > 10 * MAX_UPLOAD_MB * 1e6:
            raise MapError('The .zip is too large once unpacked.')
        with open(os.path.join(folder, name), 'wb') as out:
            out.write(z.read(item))


def natural_earth(extent=None):
    # country and state outlines, only those that touch the extent (west, east, south, north) if given.
    # Read each time and not kept, as the whole file takes about 50 MB of memory
    import geopandas as gpd
    if extent is None:
        return gpd.read_file(NE_FILE)
    w, e, s, n = extent
    return gpd.read_file(NE_FILE, bbox=(w, s, e, n))


# %% extent
def _km_per_deg_lon(lat):
    return 111.32 * max(math.cos(math.radians(lat)), 0.01)


def data_extent(pts, margin=0.15):
    # (west, east, south, north) around the points, with a margin and a minimum size of about 2 km
    w, e, s, n = pts['lon'].min(), pts['lon'].max(), pts['lat'].min(), pts['lat'].max()
    lat0 = 0.5 * (s + n)
    dx = max(e - w, 2.0 / _km_per_deg_lon(lat0))
    dy = max(n - s, 2.0 / 110.57)
    cx, cy = 0.5 * (w + e), 0.5 * (s + n)
    return (cx - dx * (0.5 + margin), cx + dx * (0.5 + margin),
            max(cy - dy * (0.5 + margin), -89.0), min(cy + dy * (0.5 + margin), 89.0))


def fit_extent(extent):
    # widen or heighten the extent so the map panel is not a sliver
    w, e, s, n = extent
    k = _km_per_deg_lon(0.5 * (s + n)) / 110.57
    ratio = (n - s) / ((e - w) * k)
    if ratio < RATIO[0]:
        grow = 0.5 * ((e - w) * k * RATIO[0] - (n - s))
        s, n = max(s - grow, -89.5), min(n + grow, 89.5)
    elif ratio > RATIO[1]:
        grow = 0.5 * ((n - s) / (k * RATIO[1]) - (e - w))
        w, e = w - grow, e + grow
    return (w, e, s, n)


def extent_width_km(extent):
    w, e, s, n = extent
    return (e - w) * _km_per_deg_lon(0.5 * (s + n))


# %% basemap
def tile_counts(extent):
    # number of imagery tiles needed at each zoom level, up to the limit
    import mercantile
    w, e, s, n = extent
    counts = {}
    for z in range(1, MAX_ZOOM + 1):
        counts[z] = sum(1 for _ in mercantile.tiles(w, s, e, n, [z]))
        if counts[z] > MAX_TILES:
            break
    return counts


def zoom_choices(extent):
    # (default zoom, lowest, highest, tiles at each zoom)
    counts = tile_counts(extent)
    allowed = [z for z, c in counts.items() if c <= MAX_TILES] or [1]
    default = max([z for z in allowed if counts[z] <= DEFAULT_TILES] or [allowed[0]])
    return default, max(min(allowed), default - 4), max(allowed), counts


def attribution(basemap):
    if basemap != IMAGERY:
        return ''
    import contextily as ctx
    return 'Basemap: ' + ctx.providers.Esri.WorldImagery['attribution'].replace('(C)', '©').replace(' -- ', '. ')


def fetch_basemap(extent, zoom):
    # imagery for the extent, warped to longitude and latitude
    # returns (image array, (west, east, south, north) of the image)
    import contextily as ctx
    w, e, s, n = extent
    counts = tile_counts(extent)
    if counts.get(zoom, MAX_TILES + 1) > MAX_TILES:
        raise MapError(f'Zoom {zoom} needs more than {MAX_TILES} tiles for this extent.')
    # use_cache=False: four downloads at a time on threads. With the cache on, contextily starts
    # extra processes and writes the tiles to disk; the app keeps the image for the session instead
    img, ext = ctx.bounds2img(w, s, e, n, zoom=zoom, source=ctx.providers.Esri.WorldImagery, ll=True,
                              n_connections=4, use_cache=False, max_retries=1, timeout=TILE_TIMEOUT)
    img, ext = ctx.warp_tiles(img, ext, t_crs='EPSG:4326')
    return np.ascontiguousarray(img), tuple(ext)


# %% pieces of the figure
def _halo(colour, width=2.0):
    return [pe.withStroke(linewidth=width, foreground=colour)]


def nice_length(target_km):
    # 1, 2 or 5 times a power of ten, at or below the target
    if target_km <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(target_km))
    return max(m * p for m in (1, 2, 5) if m * p <= target_km * 1.0001)


def _corner_xy(corner, dx, dy):
    # axes-fraction position dx, dy in from a corner
    x = dx if 'left' in corner else 1 - dx
    y = 1 - dy if corner.startswith('Top') else dy
    return x, y


def emptiest_corner(pts, extent, taken=(), towns=None):
    # the corner of the map with the fewest points in it, for the inset
    w, e, s, n = extent
    lon, lat = list(pts['lon']), list(pts['lat'])
    if towns is not None and len(towns):
        lon, lat = lon + list(towns['Lon']), lat + list(towns['Lat'])
    fx = (np.asarray(lon) - w) / (e - w)
    fy = (np.asarray(lat) - s) / (n - s)
    best, fewest = None, None
    for corner in CORNERS:
        if corner in taken:
            continue
        in_x = fx < 0.38 if 'left' in corner else fx > 0.62
        in_y = fy > 0.5 if corner.startswith('Top') else fy < 0.5
        count = int((in_x & in_y).sum())
        if fewest is None or count < fewest:
            best, fewest = corner, count
    return best or CORNERS[0]


def scale_bar(ax, extent, corner, fg, bg):
    from pyproj import Geod
    w, e, s, n = extent
    lat0 = 0.5 * (s + n)
    km = nice_length(0.25 * extent_width_km(extent))
    # the length is right at the latitude of the middle of the map
    lon2, _, _ = Geod(ellps='WGS84').fwd(0.5 * (w + e), lat0, 90.0, km * 1000.0)
    deg = lon2 - 0.5 * (w + e)
    fx, fy = _corner_xy(corner, 0.05, 0.06 if corner.startswith('Bottom') else 0.12)
    x1 = w + fx * (e - w) if 'right' in corner else w + fx * (e - w) + deg
    x0 = x1 - deg
    y = s + fy * (n - s)
    ax.plot([x0, x1], [y, y], color=fg, lw=2.2, solid_capstyle='butt', zorder=25,
            path_effects=[pe.Stroke(linewidth=3.8, foreground=bg), pe.Normal()])
    label = f'{km:g} km' if km >= 1 else f'{km * 1000:g} m'
    ax.annotate(label, (0.5 * (x0 + x1), y), xytext=(0, 3), textcoords='offset points', ha='center',
                va='bottom', color=fg, fontsize=8, zorder=25, path_effects=_halo(bg))
    return km


def north_arrow(ax, corner, fg, bg):
    # a triangle pointing up the page (north) with N above it
    x, y = _corner_xy(corner, 0.06, 0.12 if corner.startswith('Top') else 0.20)
    ax.plot([x], [y], transform=ax.transAxes, ls='', marker='^', ms=NORTH_SIZE, mfc=fg, mec=bg, mew=0.9,
            zorder=25, clip_on=False)
    ax.annotate('N', (x, y), xycoords='axes fraction', xytext=(0, 0.5 * NORTH_SIZE + 2),
                textcoords='offset points', ha='center', va='bottom', color=fg, fontsize=10, zorder=25,
                path_effects=_halo(bg))


def parse_depths(text):
    # depths typed for the legend ('10, 50, 100'): up to MAX_LEGEND_DEPTHS numbers, sorted
    # returns (list, message if something was left out)
    parts = [p for p in str(text or '').replace(';', ',').replace(' ', ',').split(',') if p]
    vals, bad = [], []
    for p in parts:
        try:
            x = float(p)
        except ValueError:
            bad.append(p)
            continue
        if np.isfinite(x) and x >= 0 and x not in vals:
            vals.append(x)
        elif not (np.isfinite(x) and x >= 0):
            bad.append(p)
    msg = ''
    if bad:
        msg = 'Not used (not a depth): ' + ', '.join(bad) + '. '
    if len(vals) > MAX_LEGEND_DEPTHS:
        msg += f'Only the first {MAX_LEGEND_DEPTHS} depths are used.'
        vals = vals[:MAX_LEGEND_DEPTHS]
    return sorted(vals), msg.strip()


def depth_sizes(depth, size, reverse=False, include=()):
    # marker area for each depth: the shallowest and deepest get SIZE_RANGE x size
    # include: depths chosen for the legend, which widen the range if they lie outside the data
    # returns (areas, function from depth to area)
    d = depth.astype(float)
    lo = min([d.min()] + list(include))
    hi = max([d.max()] + list(include))

    def area(x):
        t = 0.5 if not np.isfinite(hi - lo) or hi == lo else (np.asarray(x, dtype=float) - lo) / (hi - lo)
        t = 1 - t if reverse else t
        return size * (SIZE_RANGE[0] + (SIZE_RANGE[1] - SIZE_RANGE[0]) * t)

    out = pd.Series(area(d), index=d.index)
    return out.where(d.notna(), size), area


def depth_legend_values(depth):
    d = depth.dropna()
    lo, hi = float(d.min()), float(d.max())
    if lo == hi:
        return [lo]
    # the shallowest and the deepest, with round values between them
    gap = 0.15 * (hi - lo)
    inner = [float(t) for t in MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
             if lo + gap <= t <= hi - gap]
    return [lo] + inner + [hi]


def inset_extent(extent, factor=12.0, outline=None):
    # a square (in km) around the map, factor times as wide, or around the uploaded outline
    w, e, s, n = extent
    if outline is not None:
        b = outline.total_bounds
        w2, e2, s2, n2 = min(b[0], w), max(b[2], e), min(b[1], s), max(b[3], n)
        cx, cy = 0.5 * (w2 + e2), 0.5 * (s2 + n2)
        half_x, half_y = 0.55 * (e2 - w2), 0.55 * (n2 - s2)
    else:
        cx, cy = 0.5 * (w + e), 0.5 * (s + n)
        half_x, half_y = 0.5 * (e - w) * factor, 0.5 * (n - s) * factor
    k = _km_per_deg_lon(cy) / 110.57
    half = min(max(half_x * k, half_y), 85.0)
    half_x = min(half / k, 180.0)
    return (cx - half_x, cx + half_x, max(cy - half, -89.9), min(cy + half, 89.9))


def _draw_inset(fig, ax, extent, inset, map_h_cm):
    # locator: land in grey, the map extent as a red box
    side = min(0.30 * MAP_W, 0.42 * map_h_cm)          # cm
    ext = inset_extent(extent, inset.get('factor', 12.0), inset.get('outline'))
    pos = ax.get_position()
    fw, fh = fig.get_size_inches() / CM
    pad = 0.15
    x = pos.x0 + pad / fw if 'left' in inset['corner'] else pos.x1 - (side + pad) / fw
    y = pos.y1 - (side + pad) / fh if inset['corner'].startswith('Top') else pos.y0 + pad / fh
    axi = fig.add_axes([x, y, side / fw, side / fh], zorder=30)
    axi.set_facecolor('white')
    w, e, s, n = ext
    if inset.get('outline') is not None:
        inset['outline'].plot(ax=axi, facecolor='0.88', edgecolor='0.35', linewidth=0.4, aspect=None)
    else:
        ne = natural_earth(ext)
        land = ne[ne['level'] == 'country']
        states = ne[ne['level'] == 'state']
        if len(land):
            land.plot(ax=axi, facecolor='0.88', edgecolor='0.45', linewidth=0.3, aspect=None)
        if len(states):
            states.plot(ax=axi, facecolor='none', edgecolor='0.6', linewidth=0.2, aspect=None)
    mw, me, ms, mn = extent
    axi.add_patch(Rectangle((mw, ms), me - mw, mn - ms, facecolor='none', edgecolor='#CC004C', linewidth=0.9,
                            zorder=5))
    if (me - mw) < 0.05 * (e - w):
        # too small to see as a box
        axi.plot([0.5 * (mw + me)], [0.5 * (ms + mn)], marker='s', ms=3.5, color='#CC004C', zorder=6)
    axi.set_xlim(w, e)
    axi.set_ylim(s, n)
    axi.set_aspect('auto')
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_xlabel('')
    axi.set_ylabel('')
    for sp in axi.spines.values():
        sp.set_linewidth(0.8)
    return axi


# %% the map
def sample_map(pts, styles, extent, basemap=None, size=30.0, by_depth=False, reverse=False, labels=False,
               towns=None, poly=None, inset=None, scale_corner='Bottom right', north_corner='Top right',
               credit='', legend_depths=None, edge=None, edge_width=0.5):
    # pts: Sample, Label, lat, lon, depth (from locations())
    # styles: {Label: (colour, marker)}
    # extent: (west, east, south, north), already through fit_extent()
    # basemap: (image, image extent) from fetch_basemap(), or None for a white background
    # towns: table with Name, Lat, Lon
    # poly: dict(gdf, edge, face, alpha, width, label), or a list of them (drawn in order, first at the bottom)
    # inset: dict(corner, factor, outline) or None
    # legend_depths: depths (m) to show in the size legend, or None for automatic ones
    # edge: colour of the marker edges (default white on imagery, black on white); edge_width in points
    w, e, s, n = extent
    lat0 = 0.5 * (s + n)
    k = _km_per_deg_lon(lat0) / 110.57
    map_h = MAP_W * (n - s) / ((e - w) * k)
    fig_h = map_h + 1.7
    dark = basemap is not None
    fg, bg = ('white', 'black') if dark else ('black', 'white')
    if edge is None:
        edge = 'white' if dark else 'black'

    inside = pts[(pts['lon'] >= w) & (pts['lon'] <= e) & (pts['lat'] >= s) & (pts['lat'] <= n)]
    use_depth = by_depth and inside['depth'].notna().any()
    if use_depth:
        areas, area_of = depth_sizes(inside['depth'], size, reverse, include=legend_depths or ())
    else:
        areas = pd.Series(size, index=inside.index)

    with matplotlib.rc_context(plots.STYLE):
        fig = Figure(figsize=(FIG_W * CM, fig_h * CM), dpi=150)
        ax = fig.add_axes([MAP_LEFT / FIG_W, 1.3 / fig_h, MAP_W / FIG_W, map_h / fig_h])
        if dark:
            img, (iw, ie, is_, in_) = basemap
            ax.imshow(img, extent=(iw, ie, is_, in_), origin='upper', aspect='auto', interpolation='bilinear',
                      zorder=0)
        elif extent_width_km(extent) > 100:
            # some context on a white map. Too coarse to show on a small area
            ne = natural_earth(extent)
            if len(ne):
                ne.plot(ax=ax, facecolor='none', edgecolor='0.7', linewidth=0.4, aspect=None, zorder=1)

        polys = [] if poly is None else [poly] if isinstance(poly, dict) else list(poly)
        for k, layer in enumerate(polys):
            gdf = layer['gdf']
            lines = gdf.geom_type.str.contains('Line')
            if (~lines).any():
                gdf[~lines].plot(ax=ax, facecolor=mcolors.to_rgba(layer['face'], layer['alpha']),
                                 edgecolor=layer['edge'], linewidth=layer['width'], aspect=None, zorder=5 + 0.1 * k)
            if lines.any():
                gdf[lines].plot(ax=ax, color=layer['edge'], linewidth=layer['width'], aspect=None,
                                zorder=5 + 0.1 * k)

        # largest first, so a small marker is not hidden under a big one at the same bore
        order = areas.sort_values(ascending=False).index
        if use_depth and len(inside) <= 400:
            for i in order:
                col, mk = styles[inside.at[i, 'Label']]
                ax.scatter([inside.at[i, 'lon']], [inside.at[i, 'lat']], s=areas[i], marker=mk, color=col,
                           edgecolors=edge, linewidths=edge_width, alpha=0.9, zorder=10)
        else:
            for lab, (col, mk) in styles.items():
                g = inside[inside['Label'] == lab]
                if len(g):
                    ax.scatter(g['lon'], g['lat'], s=areas[g.index], marker=mk, color=col, edgecolors=edge,
                               linewidths=edge_width, alpha=0.9, zorder=10)
        if labels:
            for i in inside.index:
                ax.annotate(inside.at[i, 'Sample'], (inside.at[i, 'lon'], inside.at[i, 'lat']), xytext=(3, 3),
                            textcoords='offset points', fontsize=5.5, color=fg, zorder=12,
                            path_effects=_halo(bg, 1.5))

        n_towns = 0
        if towns is not None and len(towns):
            t = towns[(towns['Lon'] >= w) & (towns['Lon'] <= e) & (towns['Lat'] >= s) & (towns['Lat'] <= n)]
            n_towns = len(t)
            ax.plot(t['Lon'], t['Lat'], ls='', marker='s', ms=3.5, mfc='black', mec='white', mew=0.6, zorder=20)
            for _, row in t.iterrows():
                # to the left of the marker near the right edge, so the name stays on the map
                left = row['Lon'] > w + 0.8 * (e - w)
                ax.annotate(row['Name'], (row['Lon'], row['Lat']), xytext=(-4 if left else 4, 3),
                            textcoords='offset points', ha='right' if left else 'left', fontsize=7,
                            color='black', zorder=21, path_effects=_halo('white', 2.2))

        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_aspect('auto')    # the panel was sized for this extent, so the scale is the same both ways
        decimals = 0 if (e - w) > 8 else 1 if (e - w) > 0.8 else 2 if (e - w) > 0.08 else 3
        fmt = FuncFormatter(lambda val, _: f'{val:.{decimals}f}')
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            axis.set_major_formatter(fmt)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        km = scale_bar(ax, extent, scale_corner, fg, bg)
        north_arrow(ax, north_corner, fg, bg)
        if inset is not None:
            _draw_inset(fig, ax, extent, inset, map_h)

        # legend to the right of the map: groups, then towns and the polygon
        used = [lab for lab in styles if (inside['Label'] == lab).any()]
        # legend markers keep a dark edge (a white one would not show on the page), at the chosen width
        mec = edge if mcolors.to_rgb(edge)[0] < 0.6 else 'black'
        handles = [Line2D([], [], ls='', marker=styles[lab][1], mfc=styles[lab][0], mec=mec, mew=edge_width, ms=5,
                          label=lab) for lab in used]
        if n_towns:
            handles.append(Line2D([], [], ls='', marker='s', mfc='black', mec='0.6', mew=0.6, ms=4, label='Town'))
        for layer in polys:
            if layer['gdf'].geom_type.str.contains('Line').all():
                handles.append(Line2D([], [], color=layer['edge'], lw=layer['width'], label=layer['label']))
            else:
                handles.append(Patch(facecolor=mcolors.to_rgba(layer['face'], layer['alpha']),
                                     edgecolor=layer['edge'], linewidth=layer['width'], label=layer['label']))
        lx = (MAP_LEFT + MAP_W + 0.3) / FIG_W
        top = (1.3 + map_h) / fig_h
        leg = None
        if handles:
            leg = fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(lx, top),
                             borderaxespad=0)
        if use_depth:
            # marker sizes in the legend are the sizes on the map
            vals = list(legend_depths) if legend_depths else depth_legend_values(inside['depth'])
            biggest = math.sqrt(float(np.max(area_of(vals))))
            dh = [Line2D([], [], ls='', marker='o', mfc='0.75', mec=mec, mew=edge_width,
                         ms=math.sqrt(float(area_of(val))), label=f'{val:g} m') for val in vals]
            if inside['depth'].isna().any():
                dh.append(Line2D([], [], ls='', marker='o', mfc='white', mec='black', mew=max(edge_width, 0.5),
                                 ms=math.sqrt(size),
                                 label='No depth'))
            y = top
            if leg is not None:
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                FigureCanvasAgg(fig)
                fig.canvas.draw()
                y = fig.transFigure.inverted().transform(leg.get_window_extent())[0, 1] - 0.25 / fig_h
            leg2 = fig.legend(handles=dh, title='Screen depth', frameon=False, loc='upper left',
                              bbox_to_anchor=(lx, y), borderaxespad=0, labelspacing=max(0.4, biggest / 9),
                              handletextpad=max(0.6, biggest / 10), borderpad=0.2)
            leg2._legend_box.align = 'left'
        if credit:
            fig._credit = textwrap.fill(credit, 100)
    fig._scale_km = km
    return fig
