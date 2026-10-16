# Fire-perimeter evolution

From a KMZ of timed fire perimeters to a `TIGN_G` ignition-time array that WRF-SFIRE
will accept, with the validation numbers and figures that go with it.

The method: between two observed perimeters, compute a travel-time field outward from
the earlier one and inward from the later one under a metric, and blend them as
`s = T_f / (T_f + T_b)`. With a Euclidean metric this is the linear prototype; with the
fire ellipse built from the observed rate-of-spread (ROS) vector it is the physics the
prototypes were missing. Both perimeters are recovered exactly, burned area only grows,
and the output is a single continuous arrival-time field — which is what `TIGN_G` is.

| Script | Purpose | Section |
|---|---|---|
| `kmz_to_shapefile.py` | KMZ/KML → one timed perimeter shapefile per observation, ROS in the attribute table | [1](#1-kmz_to_shapefilepy) |
| `validation.py` | elliptical vs Euclidean on a pair or a whole series; held-out test | [2](#2-validationpy) |
| `tign.py` | evolve, chain, and write `TIGN_G` (NetCDF, GeoTIFF, or straight into `wrfinput`) | [3](#3-tignpy) |
| `figures.py` | filmstrip, arrival-time map, comparison figure | [4](#4-figurespy) |
| `chain.py` | run a whole series from one YAML file; one combined output | [5](#5-chainpy) |
| `make_chain.py` | write that YAML from a folder of timestamped shapefiles | [6](#6-make_chainpy) |

Everything is run from this directory. Library modules are listed at the end.

---

## Environment

The intended environment is the conda one in `env.yaml`:

```bash
conda env create -f env.yaml
conda activate wrfsfire
python -m pytest tests -q        # expect: 298 passed
```

Without conda, the same packages install with pip: `numpy scipy shapely pyproj
geopandas pyogrio rasterio netCDF4 xarray matplotlib pyyaml pytest`.

`env.yaml` is an osx-arm64 lockfile with exact build strings, so on another platform the
pins have to be relaxed (strip the build strings; `scikit-fmm` has no aarch64 build and
nothing here uses it).

---

## Quick start

```bash
# 1. KMZ -> shapefiles (say what unit the ROS values are in if they carry none)
python kmz_to_shapefile.py mill_fire.kmz -o perimeters/ --speed-units ch/h

# 2. Does the ROS vector change the answer, and is it right?
python validation.py perimeters/*.shp --lb 3

# 3. TIGN_G for the whole series
python tign.py perimeters/*.shp --simulation-start 2024-07-23T00:00:00Z --dt 2 --lb 3 \
       --netcdf tign_g.nc --geotiff tign_g.tif

# 4. Figures for one pair
python figures.py perimeters/p1.shp perimeters/p2.shp --lb 3 --out filmstrip.png

# 5-6. Or drive the whole series from one YAML file
python make_chain.py perimeters/ --lb 3 --dt 2
python chain.py perimeters/chain.yaml
```

Options shared by every script in steps 2–6:

| Option | Meaning | Default |
|---|---|---|
| `--cell-size` | grid resolution in metres | 25 |
| `--padding` | fraction of the fire's extent added around the grid | 0.1 |
| `--lb` | length-to-breadth ratio of the fire ellipse | none — **required for an elliptical metric**; the shapefiles carry a ROS vector but no LB |
| `--wind-mps` | midflame wind speed; supplies LB via Finney (1998) Eq. 13 when `--lb` is absent | none |
| `--infer-from-geometry` | estimate the ROS vector from the pair itself when neither perimeter carries one | off |

**Which vector a pair uses.** A perimeter's ROS vector describes the interval that
*ends* at it (which is why the first observation of a series never carries one). So the
metric for the pair *A → B* is built from **B's vector**; A's belongs to the interval
before and is used only if B has none, in which case the provenance says so. The two
are never averaged.

Finer cells cost more: each solve is cells × boundary vertices. 25 m is comfortable for
a fire a few kilometres across.

---

## 1. `kmz_to_shapefile.py`

Convert temporally-referenced fire-perimeter KMZ/KML files into shapefiles.

Each polygon in the KMZ becomes its own shapefile, named after the timestamp attached
to that polygon. Velocity ("rate of spread") vectors that accompany the perimeters are
matched to their perimeter by time and encoded into the attribute table alongside the
time reference itself. The attribute table is what every downstream script reads.

### What the script does

- **Unpacks the KMZ.** Every `.kml` document inside the archive is parsed
  namespace-agnostically, so exports from ArcGIS, QGIS, Google Earth, WRF-SFIRE and
  friends all read the same way. Bare `.kml` files are accepted directly.
- **Walks the document tree.** `Document`/`Folder` nesting is traversed recursively and
  placemarks inherit a folder's time when they don't carry their own, which is how most
  fire-progression KMZs are laid out (one folder per timestep).
- **Resolves a time for every perimeter.** Time comes from `TimeStamp/when` or
  `TimeSpan/begin`+`end`. Failing that, the script digs a timestamp out of the placemark
  name, the ExtendedData, or the folder label. Polygons with no time at all fall back to
  sequence numbering (`perimeter_unknown_0003.shp`) rather than being dropped.
- **Separates perimeters from vectors.** Polygons become perimeters; lines and points
  become velocity vectors. Vectors are matched to a perimeter by exact timestamp first,
  then by shared folder, then by nearest time within `--time-tolerance`.
- **Derives the velocity.** Speed is taken from ExtendedData when present — keys matching
  `ros`, `speed`, `velocity`, `rate of spread`, `spread rate`, with unit strings such as
  `m/s`, `km/h`, `mph`, `ft/min`, `ch/h`, `kt` normalised to m s⁻¹. Otherwise it is
  derived geodesically from the drawn vectors: bearing and displacement per segment,
  divided by the vector's own `TimeSpan`, `--vector-duration`, or the inferred perimeter
  interval. The two sources merge field by field, so an explicit speed attribute and a
  drawn bearing both get used.
- **Measures the polygon** on the WGS84 ellipsoid (area and perimeter), independent of
  the output CRS.
- **Writes metadata.** Attributes go into the DBF; each shapefile also gets a `.json`
  sidecar with the full untruncated record and the field-name mapping, plus an
  FGDC-style `.shp.xml`.

#### Attribute table

| Field | Type | Meaning |
|---|---|---|
| `step_idx` | int | 1-based index of the perimeter in the time-sorted series |
| `time_iso` | text | Perimeter time, ISO 8601 UTC |
| `time_begin` / `time_end` | text | `TimeSpan` bounds, ISO 8601 UTC (`time_end` null for a `TimeStamp`) |
| `time_epoch` | int | Unix seconds |
| `time_label` | text | Compact stamp used in the filename, e.g. `20240812T140000Z` |
| `duration_s` | float | Length of the perimeter's `TimeSpan`, seconds |
| `name` | text | Placemark name |
| `folder` | text | KML folder path the placemark came from |
| `src_kmz` / `src_kml` | text | Source archive and document inside it |
| `speed_mps` / `speed_kph` | float | Spread speed |
| `speed_min` / `speed_max` | float | Range across the matched vectors (geometry-derived only) |
| `dir_deg` | float | Direction of travel, degrees clockwise from true north |
| `vel_u_mps` / `vel_v_mps` | float | Eastward / northward components |
| `vec_len_m` | float | Mean vector displacement, metres |
| `n_vectors` | int | Number of velocity vectors matched to this perimeter |
| `vel_src` | text | `extended_data`, `vector_geometry`, a `+`-joined combination, or `none` |
| `area_m2` / `area_ha` | float | Geodesic area |
| `perim_m` | float | Geodesic perimeter |
| *(passthrough)* | text | Every ExtendedData key found, truncated to the 10-character DBF limit with collisions resolved |

#### Output files

For an output named `perimeter_20240812T140000Z`:

```
perimeter_20240812T140000Z.shp / .shx / .dbf / .prj / .cpg   the shapefile
perimeter_20240812T140000Z.json                              full metadata + field-name map
perimeter_20240812T140000Z.shp.xml                           FGDC-style metadata
perimeter_20240812T140000Z_vectors.shp                       with --write-vectors
```

Additive-mode files carry a `_cumulative` suffix before the extension.

---

### Modes

| `--mode` | Result |
|---|---|
| `individual` *(default)* | One perimeter per shapefile: the polygon observed at that time and nothing else. |
| `additive` | File *N* contains perimeters 1…*N* as separate features — the running union — written as `*_cumulative.shp`. |
| `both` | Writes both sets side by side into the same directory. |

Inputs accept files, glob patterns, or directories (searched recursively), and multiple
KMZs merge into a single time-sorted series before anything is written.

#### Which mode to use

It depends on what the polygons in the KMZ represent:

- **Full footprints** — each polygon is the whole burned area at its time, so each
  later one encloses the earlier ones. This is the usual case for mapped fire
  perimeters. Use `individual`; `additive` would only add redundant interior features.
- **New-growth patches** — each polygon is only the area burned *since the previous
  observation*. An individual file is then not a footprint at all. Use `additive`: the
  union of patches 1…*N* is the burned area at time *N*, which is what the file is
  named for.

If you are not sure, open two consecutive individual files: if the later one does not
contain the earlier one, you have patches.

#### How the two modes are read downstream

Both feed the evolution pipeline (steps 2–5 below) without any change in the
commands. `Perimeter.from_shapefile` dissolves every feature in a file into one
footprint and takes the time and ROS vector from the feature with the **latest** time,
so a cumulative file reads identically to the individual file for the same instant.
`make_chain.py` recognises the `_cumulative` suffix after the timestamp when pairing
files by name.

Attributes in a cumulative file are per feature, so a GIS table view shows one row per
perimeter, earliest first; the row that describes the file as a whole is the last one.

---

### Speed units

Every speed the script writes (`speed_mps`, `vel_u_mps`, `vel_v_mps`) is in metres per
second. The unit of an *input* speed is taken from, in order of precedence:

1. the value itself — `"556 ch/h"`, `"10 km/h"`, `"3 m/s"`;
2. the attribute's name — `ros_chph`, `Spread rate (ch/h)`;
3. `--speed-units`, for values that name no unit anywhere. The default is `m/s`.

So a KML whose ExtendedData holds bare numbers in chains per hour — the standard US
fire-behaviour unit, and easy to mistake for m/s since nothing in the file says
otherwise — is converted with:

```bash
python kmz_to_shapefile.py fire.kmz -o out/ --speed-units ch/h
```

Recognised units: `m/s`, `km/h`, `mph`, `ft/s`, `ft/min`, `m/min`, `m/h`, `ch/h`
(also `ch/hr`, `chph`, `chains/hour`, `chains per hour`), `ch/min`, `kt`. One chain is
20.1168 m, so 1 ch/h = 0.005588 m/s. Speeds derived geodesically from drawn vectors are
already in m/s and are not affected. The unit assumed is recorded in the JSON sidecar as
`velocity.input_speed_unit_assumed` and printed at the start of the run.

A quick sanity check after converting: `validation.py` reports an *implied* head rate
from the perimeter geometry next to the observed |v|; their ratio should be of order 1.
A ratio near 0.006 is chains per hour read as m/s.

---

### Command-line reference

```
kmz_to_shapefile.py [-h] -o OUTPUT_DIR [-m {individual,additive,both}]
                    [--prefix PREFIX] [--crs CRS] [--speed-units SPEED_UNITS]
                    [--vector-duration VECTOR_DURATION]
                    [--time-tolerance TIME_TOLERANCE] [--write-vectors]
                    [--no-metadata] [--overwrite] [--dry-run]
                    inputs [inputs ...]
```

| Argument | Default | Description |
|---|---|---|
| `inputs` | *(required)* | One or more `.kmz`/`.kml` files, directories, or glob patterns. Directories are searched recursively; duplicates are removed. |
| `-o`, `--output-dir` | *(required)* | Directory to write the shapefiles into. Created if it does not exist. |
| `-m`, `--mode` | `individual` | `individual`, `additive`, or `both` — see **Modes** above. |
| `--prefix` | `perimeter_` | Filename prefix for the output shapefiles. |
| `--crs` | `EPSG:4326` | CRS to write in. KML is always WGS84 lon/lat; anything else triggers a reprojection. Areas and speeds are measured geodesically regardless. |
| `--speed-units` | `m/s` | Unit of speed values in the KML that carry no unit of their own — e.g. `ch/h` for chains per hour. Output is always m/s. See **Speed units** above. |
| `--vector-duration` | *(inferred)* | Seconds a drawn velocity vector represents. Defaults to the vector's own `TimeSpan`, else the median interval between perimeters. Set this when the vectors are drawn to a fixed forecast horizon. |
| `--time-tolerance` | `0.0` | Seconds of slack when matching vectors to perimeters by time. Only consulted after exact-time and same-folder matching fail. |
| `--write-vectors` | off | Also write the matched velocity vectors as companion line shapefiles (`*_vectors.shp`). |
| `--no-metadata` | off | Skip the `.json` / `.shp.xml` sidecars. |
| `--overwrite` | off | Overwrite existing shapefiles. Without it, existing outputs are skipped with a warning. |
| `--dry-run` | off | Report what would be written, without writing it or creating the output directory. |

### Examples

```bash
# One shapefile per perimeter
python kmz_to_shapefile.py fire.kmz -o out/

# ROS values are bare numbers in chains per hour
python kmz_to_shapefile.py fire.kmz -o out/ --speed-units ch/h

# The KMZ polygons are new-growth patches, not footprints
python kmz_to_shapefile.py fire.kmz -o out/ --mode additive

# Both sets, plus the velocity vectors as their own line shapefiles
python kmz_to_shapefile.py *.kmz -o out/ --mode both --write-vectors

# Vectors drawn to a 30-minute horizon rather than the perimeter interval
python kmz_to_shapefile.py fire.kmz -o out/ --vector-duration 1800

# See what would happen first
python kmz_to_shapefile.py fire.kmz -o out/ --mode both --dry-run
```

Exit status is `0` on success, `1` if no input files were found or the input contained
no polygons.

### Notes and caveats

- The velocity-vector conventions are inferred from a description of the data, not from a
  real file — no sample KMZ was available. Testing used two synthetic KMZs covering the
  two structures the parser is built for: folder-per-timestep with polygons plus drawn
  ROS vectors, and flat placemarks with `TimeSpan` and speed/direction in `SchemaData`.
  Check `vel_src` and `n_vectors` on a real conversion to confirm the velocities were
  resolved the way you expect.
- Shapefile DBF field names are capped at 10 characters. Passthrough ExtendedData fields
  are truncated and de-duplicated; the `.json` sidecar records the full mapping.
- Text fields are truncated to 254 characters, the DBF limit.
- Shapefiles cannot store a datetime, so all times are written as ISO 8601 strings (plus
  `time_epoch` as an integer for sorting and joins).
- If two perimeters share a timestamp, the second and later filenames get a `_02`, `_03`
  suffix so nothing is silently overwritten.

---

## 2. `validation.py`

Compare the elliptical metric against a Euclidean control on the same perimeters, and
report by how much the bulk ROS vector changes the answer — and, with three or more
perimeters, whether it changes it for the better.

### What the script does

- **Two files** — runs the pair under both metrics on one grid, same clock, same
  frames, so any difference is the metric alone, and prints the full report:
  - *Endpoints and monotonicity* — Jaccard of the s = 0 and s = 1 frames against the
    observations (should be 1.000), frames that lost area (should be 0), overshoot past
    s = 1 (should be 0). These come from the clock, not the metric, and pass identically.
  - *Timing* — mean, rms and max change in a cell's ignition time between the metrics,
    as a fraction of the interval and in minutes; the signed mean; and the
    *redistribution*: mean change downwind and upwind of the initial perimeter's
    centroid along the heading, and their separation (should be positive). A fire that
    grew only downwind has no upwind half, and the report says so instead of failing.
  - *Extent* — per-frame Jaccard between the two runs' fronts, decomposed into a
    **size** share (different amounts burned by the same level) and a **placement**
    share (same amount, different cells), which is what the write-up's
    "redistribution not extent" prediction is tested against; plus the mean front
    separation in metres.
  - *Head/back advance ratio* — the C6 diagnostic over a ladder of frame counts, with
    the Euclidean column as the floor; a dash means the upwind extent did not move by
    two cells and the ratio is unresolvable.
  - *Implied versus observed ROS* — head advance measured from the perimeters ÷ head
    rate of the supplied vector. ≈1: the vector describes the growth. ≈0: the fire did
    not advance along the vector's heading. 0.006: chains per hour read as m/s.
  - A `WARNING` line in the header if the initial perimeter is not inside the final one.
- **Three or more files** — one line per consecutive pair with the numbers above
  (`--full` for every full report), then the **held-out** test on every triplet:
  evolve *i−1 → i+1* without seeing *i*, score the prediction at time *i* against the
  real perimeter under both metrics. Each Jaccard is reported with a skill score
  against the better of two trivial guesses — `static` (nothing moved since *i−1*) and
  `satur.` (everything up to *i+1* already burned) — and the block ends with the mean
  and how often the ellipse won. This is the only check that measures the method rather
  than the constraint.
- **Sparse vectors** — a pair with a vector at either end uses it; a pair with none
  **borrows** the vector of the nearest later observation that has one and the table says
  so (`own` / `borrowed`). `--no-borrow` disables this, in which case such pairs are
  skipped as isotropic.

### Command-line reference

```
validation.py [-h] [--cell-size CELL_SIZE] [--padding PADDING] [--lb LB]
              [--wind-mps WIND_MPS] [--infer-from-geometry] [--no-borrow]
              [--num-steps NUM_STEPS] [--hold-out HOLD_OUT] [--full]
              perimeters [perimeters ...]
```

| Argument | Default | Description |
|---|---|---|
| `perimeters` | *(required)* | Perimeter shapefiles in time order (a glob is fine; they are sorted by their times). |
| `--lb`, `--wind-mps`, `--cell-size`, `--padding`, `--infer-from-geometry` | | See the shared options above. |
| `--no-borrow` | off | Do not borrow a vector from the nearest observation when a pair has none. |
| `--num-steps` | `20` | Frames to compare across. |
| `--hold-out` | all interior | Score only this observation's triplet. |
| `--full` | off | Print the full report for every pair, not one line each. |

### Examples

```bash
python validation.py p1.shp p2.shp --lb 3                 # one pair, full report
python validation.py perimeters/*.shp --lb 3              # series: pair table + held-out
python validation.py perimeters/*.shp --lb 3 --hold-out 4 # one triplet
python validation.py p1.shp p2.shp --wind-mps 5           # LB from wind
```

### Reading the results

- If the ellipse and the control are nearly indistinguishable, that is a finding about
  how much information a bulk vector carries once the endpoints are pinned. On short
  intervals (minutes) expect small differences; the metric's leverage grows with the gap.
- The held-out skill is often negative for *both* metrics on real series. Look at which
  trivial guess won: `satur.` ≫ `static` means the fire did its growing early in the span
  and stalled, the reverse means it moved late. The two-sided clock assumes steady
  progress in time; bursty growth defeats that regardless of metric. The fair reading is
  elliptical against Euclidean on the same clock.
- The ellipse helps where `impl/obs` is of order 1 and hurts where the vector points
  away from the growth (`impl/obs` ≈ 0). That column tells you in advance which pairs
  to trust it on.

---

## 3. `tign.py`

Evolve a series and write `TIGN_G`, the ignition-time array WRF-SFIRE consumes.

### What the script does

- **Evolves every consecutive pair** on one shared grid and **chains** them: each
  segment's progress is mapped to absolute seconds from `--simulation-start`, and the
  elementwise minimum is taken across segments. Before chaining, each segment's final
  mask is compared cell for cell with the next segment's initial mask, and the series is
  checked for nesting; a mismatch means the observations disagree with each other and is
  refused rather than smoothed over.
- **Fills every cell.** Inside the first perimeter: that observation's time (`--inside
  flat`), or a burn-in ramp reaching the centre earliest (`--inside extend`). Outside the
  last perimeter: the field continued outward as distance ÷ head rate (`--outside
  extend`, the default — it gives SFIRE a signed-distance-like level-set seed), or a
  constant (`--outside sentinel --sentinel 1e6`).
- **Runs SFIRE's acceptance check before writing.** `module_fr_sfire_core.F`'s
  `init_no_fire` sets `lfn = tign − (time_start + dt)` and crashes unless `lfn > 0` at
  every fire-mesh node — *"ignition time must be after the end of the first time step"*.
  The same loop is reimplemented here; if it fails, the message says how much earlier
  the simulation must start. This is why `--simulation-start` must precede the first
  perimeter by more than `--dt`: every cell already burned at the first observation
  carries that observation's time.
- **Writes** a standalone NetCDF (with cell-centre `XLONG`/`XLAT`), a GeoTIFF, and/or the
  `TIGN_G` variable of an existing `wrfinput_d0X`: the array is resampled onto the file's
  `FXLONG`/`FXLAT` fire mesh, WRF's padding rows and columns are filled, the check is run
  at the file's own `DT`, and nothing is written if it fails. Then set
  `fire_tign_in_time` (or `fire_perimeter_time`) in `namelist.input` to the last
  observation's time in seconds.

### Command-line reference

```
tign.py [-h] --simulation-start SIMULATION_START --dt DT [--cell-size CELL_SIZE]
        [--padding PADDING] [--lb LB] [--wind-mps WIND_MPS] [--infer-from-geometry]
        [--outside {extend,sentinel}] [--inside {flat,extend}] [--speed SPEED]
        [--sentinel SENTINEL] [--netcdf NETCDF] [--geotiff GEOTIFF]
        [--wrfinput WRFINPUT] [--method {nearest,linear}]
        perimeters [perimeters ...]
```

| Argument | Default | Description |
|---|---|---|
| `perimeters` | *(required)* | Perimeter shapefiles in time order; two or more. |
| `--simulation-start` | *(required)* | ISO 8601 UTC instant the WRF run begins, e.g. `2024-07-23T00:00:00Z`. Must precede the first perimeter by more than `--dt`. |
| `--dt` | *(required)* | Model time step in seconds, for the acceptance check. |
| `--lb`, `--wind-mps`, `--cell-size`, `--padding`, `--infer-from-geometry` | | See the shared options above. |
| `--outside` | `extend` | What cells beyond the last perimeter get: `extend` (distance ÷ speed continuation) or `sentinel` (a constant). |
| `--inside` | `flat` | What cells inside the first perimeter get: `flat` (its time) or `extend` (a burn-in ramp). |
| `--speed` | ellipse head rate | Metres per second for either extension. Required if the metric is Euclidean. |
| `--sentinel` | none | The constant for `--outside sentinel`; must be later than the last observation. |
| `--netcdf` | none | Standalone NetCDF to write (`TIGN_G`, `XLONG`, `XLAT`, `x`, `y`). |
| `--geotiff` | none | GeoTIFF to write, in the grid's CRS. |
| `--wrfinput` | none | Existing `wrfinput_d0X` whose `TIGN_G` to overwrite in place. |
| `--method` | `nearest` | Resampling onto the fire mesh: `nearest` or `linear`. |

### Examples

```bash
python tign.py perimeters/*.shp --simulation-start 2024-07-23T00:00:00Z --dt 2 --lb 3 \
       --netcdf tign_g.nc --geotiff tign_g.tif

python tign.py perimeters/*.shp --simulation-start 2024-07-23T00:00:00Z --dt 2 --lb 3 \
       --wrfinput /run/wrfinput_d01

python tign.py p1.shp p2.shp --simulation-start 2024-07-23T00:00:00Z --dt 2 \
       --speed 0.5 --outside sentinel --sentinel 1e6 --netcdf tign_g.nc   # Euclidean
```

---

## 4. `figures.py`

Draw the figures for one perimeter pair.

### What the script does

- **Filmstrip** (`--out`) — one row, one panel per progress level (default 0.25, 0.5,
  0.75, 1.0): burned area filled (grey = burned before the interval, orange = burned
  during it), the two observed perimeters dashed, the evolved front solid, the fire
  ellipse drawn to scale in the emptiest corner with its heading and a caption, a scale
  bar, and the final match printed as a Jaccard index on the last panel. This is the row
  that replaces the poster's two.
- **Arrival-time map** (`--arrival`) — the continuous field itself with labelled
  iso-arrival contours. This is what SFIRE receives; the filmstrip is contours of it.
- **Comparison filmstrip** (`--comparison`) — the elliptical row over the Euclidean row
  for the same pair, each control panel labelled with its Jaccard against the panel
  above. The §8 figure.
- **Animation** (`--gif`) — the filmstrip as a GIF: one frame per progress level from
  s = 0 to 1 (`--frames`, default 40; `--fps`, default 8), same drawing as the panels,
  drawn at a constant rate (`--hold-frames`, default 0; raise it to hold the first
  and last frames so the endpoints can be read before the loop restarts).

For a whole chained series, `chain.py` produces the same pictures across every segment
(`chain_animate`, `chain_filmstrip`, `chain_arrival_map` in the module). There the
**animation is the default**: with dozens of segments a filmstrip is a wall of postage
stamps, while the GIF shows the front landing on each observation in turn with the
clock in the title.

#### The chained arrival map's colour axis

`chain_arrival_map` draws the whole chain's field — one raster, every evolution in it.
The colour axis matters more than it looks: a real chain's intervals span three orders
of magnitude (the Mill Fire chain runs from 3-minute gaps to one of 5133 minutes), and
an axis linear in time then spends **74% of the ramp on that one interval** while the
first thirty segments — where 64% of the cells actually ignite — share the bottom 11%
and render as a single flat shade. `figure: {arrival_scale: ...}` picks the mapping:

| `arrival_scale` | Colour is linear in | Read it as |
| --- | --- | --- |
| `segments` (default) | evolution index | which interval a cell ignited in; every evolution equally legible |
| `area` | burned area (the field's quantiles) | colour resolution follows where the fire is |
| `linear` | time | true rate, at the cost of legibility on a ragged chain |

The stretch is never hidden: the colourbar is ticked at real UTC times, and the default
contours are placed evenly along the *colour* axis rather than evenly in time, so they
spread over the map instead of bunching inside the longest interval. Set
`figure: {contour_minutes: N}` for fixed spacing in time instead. The exported TIGN_G is
untouched by any of this — it stays seconds from the simulation start.

### Command-line reference

```
figures.py [-h] [--out OUT] [--arrival ARRIVAL] [--comparison COMPARISON]
           [--gif GIF] [--frames FRAMES] [--fps FPS] [--hold-frames HOLD_FRAMES]
           [--levels LEVELS [LEVELS ...]] [--cell-size CELL_SIZE] [--padding PADDING]
           [--lb LB] [--wind-mps WIND_MPS] [--infer-from-geometry] [--title TITLE]
           [--dpi DPI]
           initial final
```

| Argument | Default | Description |
|---|---|---|
| `initial`, `final` | *(required)* | The two perimeter shapefiles. |
| `--out` | `filmstrip.png` | Filmstrip PNG. |
| `--arrival` | none | Arrival-time map PNG. |
| `--comparison` | none | Comparison filmstrip PNG. |
| `--gif` | none | Animated GIF of the evolution. |
| `--frames` | `40` | Frames in the GIF, s = 0 to 1. |
| `--fps` | `8` | GIF playback rate. |
| `--levels` | `0.25 0.5 0.75 1.0` | Progress levels, one panel each, in (0, 1]. |
| `--title` | times and metric | Figure title. |
| `--dpi` | `300` | Output resolution; `110` for quick looks. |
| `--lb`, `--wind-mps`, `--cell-size`, `--padding`, `--infer-from-geometry` | | See the shared options above. |

### Examples

```bash
python figures.py p1.shp p2.shp --lb 3 --out filmstrip.png
python figures.py p1.shp p2.shp --lb 3 --out strip.png --arrival arrival.png \
       --comparison comparison.png --levels 0.33 0.66 1
python figures.py p1.shp p2.shp --lb 3 --gif evolution.gif --frames 60 --fps 10
```

Synthetic examples of every figure are in `figures_out/`.

---

## 5. `chain.py`

Run any number of evolutions in sequence from one YAML file, combine them into one
arrival-time field, and write the outputs the file asks for. A file with one evolution
is the way to run a single pair; several is chain mode.

### What the script does

- Loads every perimeter once (the one shared by two consecutive evolutions is the same
  object in both), applying the YAML's `time`, `ros` and `lb` overrides.
- Refuses a time gap between evolutions when the file is read, and a geometry mismatch
  at a join before anything is written. Checks nesting with a tolerance (default 1%) that
  absorbs the few boundary cells reprojection and rasterisation leave outside a successor
  on real data, and prints the actual loss per pair.
- Evolves each pair with its own metric — the final perimeter's vector, or the nearest later one when
  it has none (`borrow_ros`) — and prints endpoints, violations and provenance per segment.
  With `validate: true`, also runs the elliptical-vs-Euclidean comparison per pair.
- Chains the segments (`tign.py`'s rules) and writes: an **animation** of the whole
  series (the default picture — one frame per instant, frames spaced within each segment
  rather than uniformly in time so long gaps don't dominate, each observation held for a
  few frames with its Jaccard shown), optionally a filmstrip with one panel per absolute
  time, the combined arrival-time map, a text report, and optionally the `TIGN_G` arrays
  and `wrfinput`.

### The YAML file

```yaml
grid:                        # optional
  cell_size: 25              # metres
  padding: 0.1
metric:                      # defaults for every evolution; optional
  lb: 3.0                    # or wind_mps: 5.0, or infer_from_geometry: true
simulation_start: 2024-07-23T00:00:00Z   # optional; defaults to the first perimeter's time

evolutions:
  - initial: {path: perimeters/p1.shp, time: 2024-07-23T01:37:31Z}
    final:   {path: perimeters/p2.shp, time: 2024-07-23T02:50:41Z}
  - initial: {path: perimeters/p2.shp, time: 2024-07-23T02:50:41Z}
    final:   {path: perimeters/p3.shp, time: 2024-07-23T04:10:00Z}
    lb: 4.0                  # per-evolution override of anything in `metric`

outputs:
  animation: chain.gif       # the readable form for a long series
  # filmstrip: chain_filmstrip.png   # one panel per instant; fine for a few segments
  arrival_map: chain_arrival.png
  report: chain_report.txt
  netcdf: tign_g.nc          # optional
  geotiff: tign_g.tif        # optional
  wrfinput: wrfinput_d01     # optional, written in place
  dt: 2.0                    # model time step, for the SFIRE check
tign:                        # optional; see tign.py --outside / --inside
  outside: extend
  inside: flat

figure:                      # optional
  frames_per_segment: 8      # animation: frames between consecutive observations
  fps: 6
  hold_frames: 0             # extra frames on each observation (0: no pause at a join)
  arrival_scale: segments    # arrival map colour axis: segments|area|linear
  # contour_minutes: 60      # fixed contour spacing instead of even-in-colour
  gif_dpi: 100
  panels_per_segment: 3      # filmstrip, if enabled
  max_columns: 4
  dpi: 300                   # PNGs
  # times: [2024-07-23T02:00:00Z, ...]   # explicit frame/panel times instead

validate: true               # also run the elliptical-vs-Euclidean comparison per pair
nesting_tolerance: 0.01      # fraction of a perimeter allowed outside its successor
borrow_ros: true             # a pair with no vector at either end borrows the nearest later one
```

Paths are relative to the YAML file. A perimeter's `time` overrides whatever its
shapefile carries; a bare timestamp is read as UTC. A perimeter entry may also give
`ros: {speed_mps: 0.3, dir_deg: 60}` (or `ros: [u, v]`) and `lb`. An evolution may set
`metric: euclidean` to force the control.

### Command-line reference

```
chain.py [-h] [--quiet] config
```

| Argument | Description |
|---|---|
| `config` | The YAML chain description. |
| `--quiet` | Suppress progress lines (the report file is still written). |

### Examples

```bash
python chain.py chain.yaml
python chain.py chain.yaml --quiet && cat chain_report.txt
```

A worked example with four synthetic perimeters is in `figures_out/chain_demo/`.

---

## 6. `make_chain.py`

Write a `chain.yaml` from a folder of timestamped perimeter shapefiles.

### What the script does

- Reads the time from the end of each filename — `…YYYYMMDD_HHMMSSZ.shp`, or the
  `…YYYYMMDDTHHMMSSZ.shp` form `kmz_to_shapefile.py` writes, with or without its
  `_cumulative` suffix. Files with no stamp are skipped with a note (`--strict` makes
  that an error); `_vectors.shp` companions are ignored; two files with the same time are
  refused.
- Sorts by that time, pairs consecutively, and writes a runnable YAML next to the files
  with paths relative to it, so the folder can be moved intact. Every block is commented
  for hand-editing.
- Without `--lb` or `--wind-mps` the metric block is written as a commented `REQUIRED`
  reminder and a warning is printed, because otherwise every evolution would silently run
  Euclidean.

### Command-line reference

```
make_chain.py [-h] [-o OUT] [--pattern PATTERN] [--cell-size CELL_SIZE] [--lb LB]
              [--wind-mps WIND_MPS] [--simulation-start SIMULATION_START]
              [--lead-hours LEAD_HOURS] [--dt DT] [--prefix PREFIX] [--no-validate]
              [--tign] [--panels-per-segment PANELS_PER_SEGMENT] [--dpi DPI]
              [--nesting-tolerance NESTING_TOLERANCE] [--strict] [--overwrite]
              folder
```

| Argument | Default | Description |
|---|---|---|
| `folder` | *(required)* | Directory of perimeter shapefiles. |
| `-o`, `--out` | `<folder>/chain.yaml` | YAML to write; paths stay relative to it. |
| `--pattern` | `*.shp` | Glob within the folder. |
| `--cell-size` | `25` | Grid resolution in metres. |
| `--lb` / `--wind-mps` | none | The metric block for every evolution. |
| `--simulation-start` | `--lead-hours` before the first perimeter | ISO 8601 UTC. |
| `--lead-hours` | `1` | Lead time before the first perimeter when `--simulation-start` is not given. |
| `--dt` | `2` | Model time step for the SFIRE check. |
| `--prefix` | `chain` | Output filenames are `<prefix>_filmstrip.png` and so on. |
| `--no-validate` | off | Write `validate: false`. |
| `--tign` | off | Also request the NetCDF and GeoTIFF arrays. |
| `--panels-per-segment` | `3` | Filmstrip panels per segment. |
| `--dpi` | `300` | Figure resolution. |
| `--nesting-tolerance` | `0.01` | Fraction of a perimeter allowed outside its successor. |
| `--strict` | off | Fail on a shapefile with no timestamp instead of skipping it. |
| `--overwrite` | off | Replace an existing YAML. |

### Examples

```bash
python make_chain.py perimeters/ --lb 3 --dt 2
python make_chain.py perimeters/ --wind-mps 5 --tign --prefix mill -o runs/mill.yaml
python chain.py perimeters/chain.yaml
```

---

## Using the modules directly

Everything the scripts do is a few lines in Python:

```python
from perimeter import PerimeterSeries
from evolution import GaugeEvolution
from validation import MetricComparison
from tign import ArrivalTime
from figures import filmstrip

series = PerimeterSeries.from_shapefiles("perimeters/")
grid = series.common_grid(cell_size=25.0)
series = series.on_grid(grid)

evo = GaugeEvolution(series[0], series[1], grid, lb=3.0)
print(evo.summary())                       # endpoint Jaccard, violations, ...
s = evo.progress                           # 0 on the first perimeter, 1 on the last

print(MetricComparison(series[0], series[1], grid, lb=3.0).report())

arrival = ArrivalTime.from_series(series, grid, "2024-07-23T00:00:00Z", lb=3.0)
arrival.check(dt=2.0)                      # raises if SFIRE would
arrival.to_netcdf("tign_g.nc")

filmstrip(evo, path="filmstrip.png")
```

## Files

| File | Role |
|---|---|
| `kmz_to_shapefile.py` | KMZ/KML → timed perimeter shapefiles with ROS attributes |
| `validation.py` | the §8 checks; elliptical-vs-Euclidean comparison; held-out test; series mode |
| `tign.py` | `ArrivalTime`: chaining, `TIGN_G` export, the SFIRE acceptance check |
| `figures.py` | filmstrip, animation, arrival-time map, comparison figure; chain versions of the first three |
| `chain.py` | YAML-driven chain mode |
| `make_chain.py` | folder of shapefiles → `chain.yaml` |
| `grid.py` | `GridSpec`: the shared metre grid, rasterisation, signed distance, projection checks |
| `perimeter.py` | `Perimeter`, `PerimeterSeries`: loading, ROS/time attributes, projection, nesting |
| `wavelet.py` | `FireEllipse`: the fire ellipse from a ROS vector and LB (Finney Eqs. 13–14) |
| `gauge.py` | the closed-form arrival-time solver |
| `metrics.py` | Euclidean and elliptical metrics; choosing one for a pair |
| `clock.py` | how travel time becomes 0–1 progress (two-sided = exact endpoints) |
| `evolution.py` | `GaugeEvolution`: one pair → progress field, frames, diagnostics |
| `tests/` | 298 tests; `python -m pytest tests -q` |
| `figures_out/` | synthetic example figures and the chain demo |
| `perimeter_evolution.py`, `fire_perimeter_evolution.py`, `shapefile_loader.py` | the original prototypes, kept for the Euclidean regression test |
| `perimeter_evolution_ros_recommendation_v2.docx` | the method write-up this implements |
