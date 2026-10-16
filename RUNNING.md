# Running the perimeter-evolution pipeline

From a KMZ of timed fire perimeters to a `TIGN_G` array WRF-SFIRE will accept,
plus the validation numbers and figures that go with it. Every step is one
command, and step 5 runs a whole series from one YAML file. `kmz_to_shapefile.py`
has its own detailed reference in [README.md](README.md); this file is the
end-to-end walkthrough.

## 0. Environment

The intended environment is the conda one in `env.yaml`:

```bash
conda env create -f env.yaml
conda activate wrfsfire
```

Without conda, the same packages install with pip
(`numpy scipy shapely pyproj geopandas rasterio netCDF4 xarray matplotlib pyyaml pytest`).

Check everything works before touching data:

```bash
python -m pytest tests -q        # expect: 298 passed
```

All commands below are run from this directory.

## 1. KMZ → perimeter shapefiles

```bash
python kmz_to_shapefile.py mill_fire.kmz -o perimeters/
```

You get one shapefile per timed perimeter, named by timestamp
(`perimeters/perimeter_20240723T013731Z.shp`, …), each carrying the observation
time and the matched ROS vector in its attribute table. Everything downstream
reads those attributes, so nothing needs re-typing.

If the ROS values in the KMZ are bare numbers in a unit other than m/s — chains per
hour is the common case — say so, and they are converted on the way out:

```bash
python kmz_to_shapefile.py mill_fire.kmz -o perimeters/ --speed-units ch/h
```

Everything downstream expects m/s. The `Implied versus observed ROS` block of the
validation report is the check: a ratio far from 1 (0.006 for ch/h read as m/s)
means the unit is wrong.

If the KMZ has no velocity vectors, the shapefiles still work — see the
`--infer-from-geometry` note in step 2.

Either converter mode feeds the pipeline. `individual` (the default) writes
one polygon per file; `additive` writes each file as that perimeter plus every
earlier one (`*_cumulative.shp`). The loader dissolves a file's features into
one footprint and takes the time and ROS vector from the newest feature, so a
cumulative file reads the same as the individual one for the same instant.
Use `additive` when the KMZ polygons are new-growth patches rather than full
footprints — the union then *is* the footprint — and `individual` otherwise.

## 2. Does the ROS vector change the answer? (validation)

Run the elliptical-versus-Euclidean comparison on a consecutive pair:

```bash
python validation.py perimeters/perimeter_20240723T013731Z.shp \
                     perimeters/perimeter_20240723T025041Z.shp \
                     --lb 3 --cell-size 25
```

The metric for a pair *A → B* is built from **B's** ROS vector: a perimeter's vector
describes the interval that ends at it, so A's belongs to the previous interval and
is used only if B has none (the report's `provenance` line says which). They are
never averaged.

`--lb` is the length-to-breadth ratio of the fire ellipse. The shapefiles carry
a ROS vector but not an LB, so you must supply one, either directly or from
wind speed (`--wind-mps 5` uses Finney's Eq. 13). Without it the metric falls
back to Euclidean and the script says so rather than comparing nothing.

Read the report top to bottom:

- **Endpoints and monotonicity** — both Jaccards should be 1.000 and violations 0
  under both metrics. Anything else is a plumbing problem, not a result.
- **Timing** — how far the vector moves arrival times, in minutes.
  `separation` should be positive (downwind burns earlier than upwind).
- **Extent** — whether the disagreement is *placement* (same area, different
  cells — a redistribution) or *size* (one metric simply burns more by a given
  level). The report says which dominates; both happen.
- **Head/back ratio** — a ladder over frame counts. Read the quotient column
  and its spread, not the absolute numbers: on a pinned pair the Euclidean
  column reads the perimeters' own lopsidedness.
- **Implied vs observed ROS** — how much of the growth along the heading the
  bulk vector explains. A ratio far from 1 is a finding to report.

With three or more perimeters the output changes shape: one line per
consecutive pair (interval, mean progress change, front offset, size/placement
split, implied-over-observed ratio, and which vector was used), then the
**held-out** test on every triplet — evolve *i−1 → i+1* and score the withheld
*i* under both metrics:

```bash
python validation.py perimeters/*.shp --lb 3          # add --full for every pair's full report
python validation.py perimeters/*.shp --lb 3 --hold-out 4   # one triplet only
```

Each held-out Jaccard is reported against two trivial baselines (nothing moved,
everything burned) as a skill score, with a mean and a win count at the bottom.
This is the only check that measures the method rather than the constraint.

Real series carry ROS vectors on some perimeters and not others. A pair with a
vector on its final perimeter uses it; a pair with none **borrows the vector of the
nearest later observation** that has one and the table says so (`borrowed` vs `own`).
`--no-borrow` turns that off, in which case such pairs are skipped. `chain.py`
does the same (`borrow_ros: true` in the YAML) and prints the donor per segment.

## 3. Write TIGN_G for WRF-SFIRE

```bash
python tign.py perimeters/*.shp \
       --simulation-start 2024-07-23T00:00:00Z --dt 2 --lb 3 \
       --netcdf tign_g.nc --geotiff tign_g.tif
```

- `--simulation-start` is when the WRF run begins (ISO 8601, UTC). It must be
  *before* the first perimeter by more than one time step — every cell already
  burned at the first observation is stamped with that observation's time, and
  SFIRE rejects any ignition time that is not after the end of its first step.
  The script runs that exact check and tells you how much earlier to start if
  it fails.
- `--dt` is the model time step, for that check.
- Consecutive pairs are evolved and chained automatically. The series is
  refused if any later perimeter drops part of an earlier one — that is a data
  problem to fix upstream, not something to smooth over.
- Outside the last perimeter the field is continued outward as
  distance ÷ head rate (`--outside extend`, the default), which gives SFIRE a
  well-shaped level-set seed. `--outside sentinel --sentinel 1e6` writes a
  constant instead if your workflow expects one.

To write straight into an existing `wrfinput_d01`'s fire mesh:

```bash
python tign.py perimeters/*.shp \
       --simulation-start 2024-07-23T00:00:00Z --dt 2 --lb 3 \
       --wrfinput /path/to/wrfinput_d01
```

The array is resampled onto `FXLONG`/`FXLAT`, checked at the file's own `DT`,
and nothing is written if the check fails. Then set `fire_tign_in_time` (or
`fire_perimeter_time`) in `namelist.input` to the last observation's time in
seconds and run WRF as usual.

## 4. Figures

```bash
python figures.py perimeters/perimeter_20240723T013731Z.shp \
                  perimeters/perimeter_20240723T025041Z.shp \
                  --lb 3 --out filmstrip.png \
                  --arrival arrival_time.png \
                  --comparison comparison.png
```

- `filmstrip.png` — the one row for the poster: burned area and front at
  s = 0.25, 0.5, 0.75, 1.0, with the fire ellipse drawn to scale and the final
  match printed as a Jaccard index. Change the levels with `--levels 0.33 0.66 1`.
- `arrival_time.png` — the continuous field itself, with iso-arrival contours.
  This is what SFIRE actually receives.
- `comparison.png` — elliptical row over Euclidean row, each control panel
  labelled with its agreement against the panel above.
- add `--gif evolution.gif` for the same thing as an animation (`--frames 40`,
  `--fps 8` by default).

Default output is 300 dpi; `--dpi 110` for quick looks. Synthetic examples of
all of them are in `figures_out/`.

## 5. Chain mode: a whole series from one YAML file

`chain.py` runs any number of evolutions in sequence from a single file,
checks that each ends where the next begins, combines them into one
arrival-time field, and writes the outputs the file asks for. The same file
with one evolution is how to run a single pair.

```yaml
# chain.yaml
grid:
  cell_size: 25                 # metres
metric:
  lb: 3.0                       # defaults for every evolution (or wind_mps, infer_from_geometry)
simulation_start: 2024-07-23T00:00:00Z     # optional; defaults to the first perimeter's time

evolutions:
  - initial: {path: perimeters/perimeter_20240723T013731Z.shp, time: 2024-07-23T01:37:31Z}
    final:   {path: perimeters/perimeter_20240723T025041Z.shp, time: 2024-07-23T02:50:41Z}
  - initial: {path: perimeters/perimeter_20240723T025041Z.shp, time: 2024-07-23T02:50:41Z}
    final:   {path: perimeters/perimeter_20240723T041000Z.shp, time: 2024-07-23T04:10:00Z}
    lb: 4.0                     # per-evolution override

outputs:
  animation: chain.gif               # the whole sequence as a GIF (the default picture)
  # filmstrip: chain_filmstrip.png   # one panel per instant; readable for a few segments
  arrival_map: chain_arrival.png     # the combined field, every evolution in it
  report: chain_report.txt
  netcdf: tign_g.nc                  # optional; needs dt
  dt: 2.0

figure:
  frames_per_segment: 8         # animation frames between observations
  fps: 6
  hold_frames: 0                # extra frames on each observation (0: no pause at a join)
  arrival_scale: segments       # arrival map colour axis: segments|area|linear
  panels_per_segment: 3         # filmstrip, if enabled
  max_columns: 4
  dpi: 300

validate: true                  # also run the elliptical-vs-Euclidean comparison per pair
nesting_tolerance: 0.01         # fraction of a perimeter allowed outside its successor
```

```bash
python chain.py chain.yaml
```

### Generating the YAML from a folder

You don't have to write the file by hand. `make_chain.py` reads the time from
the end of each shapefile's name — `…YYYYMMDD_HHMMSSZ.shp` (the
`…YYYYMMDDTHHMMSSZ.shp` form `kmz_to_shapefile.py` writes is accepted too) —
sorts the files by time, pairs them consecutively, and writes a runnable
`chain.yaml` next to them:

```bash
python make_chain.py perimeters/ --lb 3 --dt 2
python chain.py perimeters/chain.yaml
```

Useful flags: `-o other.yaml` to write elsewhere (paths stay relative to the
YAML), `--simulation-start 2024-07-23T00:00:00Z` (default: one hour before the
first perimeter; `--lead-hours` changes that), `--tign` to request the NetCDF
and GeoTIFF arrays, `--prefix mill` for the output filenames, `--strict` to
fail on a file with no timestamp instead of skipping it, `--overwrite`.
Without `--lb` or `--wind-mps` the metric block is written as a commented
reminder and every evolution would run Euclidean, so set one.

Notes:

- Paths are relative to the YAML file. A perimeter's `time` overrides
  whatever the shapefile carries; a bare timestamp is read as UTC. An entry may
  also give `ros: {speed_mps: 0.3, dir_deg: 60}` and `lb` to override those.
- Each evolution must begin at the instant the previous one ended, with the
  same perimeter. A gap is refused when the file is read; a geometry mismatch
  at a join is refused before anything is written.
- `nesting_tolerance` absorbs the few boundary cells that reprojection and
  rasterisation leave outside a successor perimeter on real data. Anything
  above it is treated as the observations contradicting each other, which is a
  data problem to fix upstream. The actual loss per pair is printed.
- The animation has `frames_per_segment` frames between consecutive
  observations — spaced within each segment rather than uniformly in time, so
  an 85-hour gap doesn't swallow the whole GIF. `hold_frames` repeats each
  observation frame; it is 0 by default, because the arrival-time field is one
  continuous raster across the joins and a dwell there makes the front look as
  though it stops at every observation, which it does not. Raise it to read
  each observation's Jaccard in the title. The filmstrip is
  the same at `panels_per_segment` panels per segment; with more than a few
  segments it is unreadable, which is why `make_chain.py` writes it commented
  out. Pass explicit `figure: {times: [2024-07-23T02:00:00Z, ...]}` to choose
  the frames or panels yourself.
- The arrival map's colour axis is stretched by `figure: {arrival_scale: ...}`:
  `segments` (the default) gives every evolution an equal share of the ramp,
  `area` gives equal colour to equal burned area, `linear` is linear in time.
  The default exists because a real chain's intervals are ragged — the Mill
  Fire chain ends with a 5133-minute interval that would otherwise take 74%
  of the colour range for 15% of the burned cells, leaving the first twelve
  hours of fire as one flat shade. The colourbar is ticked in UTC either way,
  and contours are placed evenly along the colour axis unless
  `figure: {contour_minutes: N}` asks for fixed spacing in time. None of this
  touches the exported TIGN_G.
- `outputs` accepts every writer `tign.py` has: `netcdf`, `geotiff`,
  `wrfinput` (written in place, checked at the file's own `DT`), and `tign:
  {outside: extend|sentinel, inside: flat|extend, speed, sentinel}` for the
  array options.

A worked example with synthetic perimeters is in `figures_out/chain_demo/`.

## Common options

| Option | Meaning | Default |
|---|---|---|
| `--cell-size` | grid resolution in metres | 25 |
| `--padding` | fraction of the fire's extent added around the grid | 0.1 |
| `--lb` | length-to-breadth ratio of the fire ellipse | none — required for an elliptical metric |
| `--wind-mps` | midflame wind speed; supplies LB via Finney Eq. 13 when `--lb` is absent | none |
| `--infer-from-geometry` | estimate the ROS vector from the pair itself when neither perimeter carries one | off |

Finer cells cost more: each solve is cells × boundary vertices. 25 m is
comfortable for a fire a few kilometres across; go to 10 m only if you need it.

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

## What each file is

| File | Role |
|---|---|
| `kmz_to_shapefile.py` | KMZ/KML → timed perimeter shapefiles with ROS attributes |
| `grid.py` | `GridSpec`: the shared metre grid, rasterisation, signed distance |
| `perimeter.py` | `Perimeter`, `PerimeterSeries`: loading, projection, nesting checks |
| `wavelet.py` | `FireEllipse`: the fire ellipse from a ROS vector and LB |
| `gauge.py` | the closed-form arrival-time solver |
| `metrics.py` | Euclidean and elliptical metrics; choosing one for a pair |
| `clock.py` | how travel time becomes 0–1 progress (two-sided = exact endpoints) |
| `evolution.py` | `GaugeEvolution`: one pair → progress field, frames, diagnostics |
| `validation.py` | the §8 checks and the elliptical-vs-Euclidean comparison |
| `tign.py` | `ArrivalTime`: chaining, TIGN_G export, the SFIRE acceptance check |
| `figures.py` | filmstrip, animation, arrival-time map, comparison figure; chain versions of the first three |
| `chain.py` | YAML-driven chain mode: many evolutions in sequence, one combined output |
| `make_chain.py` | writes that YAML from a folder of timestamped shapefiles |
| `tests/` | 298 tests; `python -m pytest tests -q` |
| `perimeter_evolution.py`, `fire_perimeter_evolution.py`, `shapefile_loader.py` | the original prototypes, kept for the Euclidean regression test |
