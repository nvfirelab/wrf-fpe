"""
Module running a chain of perimeter evolutions from a YAML description.

One file names every evolution to perform - the initial and final perimeter
of each, with the UTC time of each observation - and the script evolves them
in order, checks that they join, combines them into one arrival-time field,
and writes whatever outputs the file asks for. A file with a single evolution
is the way to run one pair; a file with several is chain mode.

    grid:                      # optional
      cell_size: 25            # metres
      padding: 0.1
    metric:                    # defaults for every evolution; optional
      lb: 3.0                  # or wind_mps: 5.0, or infer_from_geometry: true
    simulation_start: 2024-07-23T00:00:00Z   # optional; defaults to the first perimeter
    evolutions:
      - initial: {path: perimeters/p1.shp, time: 2024-07-23T01:37:31Z}
        final:   {path: perimeters/p2.shp, time: 2024-07-23T02:50:41Z}
      - initial: {path: perimeters/p2.shp, time: 2024-07-23T02:50:41Z}
        final:   {path: perimeters/p3.shp, time: 2024-07-23T04:10:00Z}
        lb: 4.0                # per-evolution override of anything in `metric`
    outputs:
      animation: chain.gif     # the readable form for a long series
      filmstrip: chain_filmstrip.png   # one panel per instant; fine for a few segments
      arrival_map: chain_arrival.png
      report: chain_report.txt
      netcdf: tign_g.nc        # optional
      geotiff: tign_g.tif      # optional
      wrfinput: wrfinput_d01   # optional, written in place
      dt: 2.0                  # model time step, for the SFIRE check
    figure:                    # optional
      frames_per_segment: 8    # animation
      fps: 6
      hold_frames: 0           # extra frames on each observation (0: no pause at a join)
      panels_per_segment: 3    # filmstrip
      max_columns: 4
      arrival_scale: segments  # arrival map colour axis: segments|area|linear
      dpi: 300                 # PNGs; gif_dpi (default 100) for the animation
    validate: true             # also run the elliptical-vs-Euclidean comparison per pair
    nesting_tolerance: 0.01    # fraction of a perimeter allowed outside its successor
    borrow_ros: true           # a pair with no vector at either end borrows the nearest later observation's

A perimeter's `time` overrides whatever its shapefile carries; a perimeter
entry may also give `ros: {speed_mps: 0.3, dir_deg: 60}` and `lb` to
override those. Times must be timezone-aware; a bare timestamp is read as
UTC.

Chaining needs each evolution to end where the next one begins - the same
time, and the same geometry. A gap or a disagreement is refused, not
smoothed over: it means the observations contradict each other.
"""

import os
from datetime import date, datetime, timezone

import numpy as np

from evolution import GaugeEvolution
from perimeter import Perimeter, PerimeterSeries, ros_from_speed_bearing
from tign import ArrivalTime, _aware

METRIC_KEYS = ("lb", "wind_mps", "infer_from_geometry", "metric")


# ----------------------------------------------------------------------
# Reading the file
# ----------------------------------------------------------------------

def load_config(path):
    """
    Read and validate a chain description.

    Parameters:
    -----------
    path : str
        The YAML file

    Returns:
    --------
    config : dict
        With paths resolved relative to the file, times parsed, and every
        evolution's metric options merged with the defaults
    """
    import yaml

    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    base = os.path.dirname(os.path.abspath(path))

    evolutions = raw.get("evolutions")
    if not evolutions or not isinstance(evolutions, list):
        raise ValueError(f"{path}: 'evolutions' must be a non-empty list")

    defaults = dict(raw.get("metric") or {})
    unknown = set(defaults) - set(METRIC_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown metric option(s) {sorted(unknown)}")

    parsed = []
    for k, entry in enumerate(evolutions):
        if not isinstance(entry, dict) or "initial" not in entry or "final" not in entry:
            raise ValueError(f"{path}: evolution {k} needs 'initial' and 'final'")
        options = dict(defaults)
        for key in METRIC_KEYS:
            if key in entry:
                options[key] = entry[key]
        parsed.append({
            "initial": _perimeter_spec(entry["initial"], base, f"evolution {k} initial"),
            "final": _perimeter_spec(entry["final"], base, f"evolution {k} final"),
            "options": options,
        })

    _check_continuity(parsed)

    outputs = dict(raw.get("outputs") or {})
    for key in ("filmstrip", "animation", "arrival_map", "report", "netcdf", "geotiff",
                "wrfinput"):
        if outputs.get(key):
            outputs[key] = _resolve(outputs[key], base)

    # Checked here rather than at drawing time: the figures are written last,
    # and a typo should not cost a whole chain's solve before it is reported.
    figure = dict(raw.get("figure") or {})
    if figure.get("arrival_scale") is not None:
        from figures import ARRIVAL_SCALES
        if figure["arrival_scale"] not in ARRIVAL_SCALES:
            raise ValueError(
                f"figure.arrival_scale must be one of {ARRIVAL_SCALES}, not "
                f"{figure['arrival_scale']!r}")

    return {
        "grid": {"cell_size": float((raw.get("grid") or {}).get("cell_size", 25.0)),
                 "padding": float((raw.get("grid") or {}).get("padding", 0.1)),
                 "crs": (raw.get("grid") or {}).get("crs")},
        "simulation_start": (_time(raw["simulation_start"], "simulation_start")
                             if raw.get("simulation_start") is not None else None),
        "evolutions": parsed,
        "outputs": outputs,
        "figure": figure,
        "validate": bool(raw.get("validate", False)),
        "nesting_tolerance": float(raw.get("nesting_tolerance", 0.01)),
        "borrow": bool(raw.get("borrow_ros", True)),
        "tign": dict(raw.get("tign") or {}),
        "source": os.path.abspath(path),
    }


def _perimeter_spec(entry, base, label):
    """One perimeter entry: path, optional time / ros / lb overrides."""
    if isinstance(entry, str):
        entry = {"path": entry}
    if not isinstance(entry, dict) or "path" not in entry:
        raise ValueError(f"{label}: needs a 'path'")
    spec = {"path": _resolve(entry["path"], base), "time": None, "ros": None, "lb": None}
    if entry.get("time") is not None:
        spec["time"] = _time(entry["time"], f"{label} time")
    if entry.get("ros") is not None:
        ros = entry["ros"]
        if isinstance(ros, dict) and {"speed_mps", "dir_deg"} <= set(ros):
            spec["ros"] = ros_from_speed_bearing(float(ros["speed_mps"]), float(ros["dir_deg"]))
        elif isinstance(ros, (list, tuple)) and len(ros) == 2:
            spec["ros"] = np.asarray(ros, dtype=float)
        else:
            raise ValueError(f"{label}: ros must be {{speed_mps, dir_deg}} or [u, v]")
    if entry.get("lb") is not None:
        spec["lb"] = float(entry["lb"])
    return spec


def _time(value, label):
    """A timezone-aware UTC datetime from what YAML hands back."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            # PyYAML returns bare timestamps naive; the file is documented
            # as UTC, so that is what a bare one means.
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        try:
            return _aware(text)
        except ValueError:
            return _aware(text + "+00:00")
    raise ValueError(f"{label}: cannot read a time from {value!r}")


def _resolve(path, base):
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def _check_continuity(evolutions):
    """
    Each evolution must begin when the previous one ended.

    Only the times are checked here. Two different files at the same instant
    are allowed if they rasterise identically - ArrivalTime.chain compares
    the masks cell for cell - so the geometry check waits until the grid
    exists.
    """
    for k, (a, b) in enumerate(zip(evolutions[:-1], evolutions[1:])):
        end, start = a["final"], b["initial"]
        if end["time"] is not None and start["time"] is not None and end["time"] != start["time"]:
            raise ValueError(
                f"evolution {k} ends at {end['time'].isoformat()} but evolution "
                f"{k + 1} begins at {start['time'].isoformat()}; a chain has to be "
                "continuous. Run the two as separate files if they are separate.")


# ----------------------------------------------------------------------
# Running it
# ----------------------------------------------------------------------

def run(config, log=print):
    """
    Evolve every pair in a configuration and write its outputs.

    Parameters:
    -----------
    config : dict
        From `load_config`
    log : callable
        Where progress lines go

    Returns:
    --------
    result : dict
        arrival (ArrivalTime), evolutions (list of GaugeEvolution),
        comparisons (list, when validate is on), report (str), and the paths
        written
    """
    perimeters = _load_perimeters(config)
    series = PerimeterSeries(perimeters)
    grid = series.common_grid(cell_size=config["grid"]["cell_size"],
                              padding=config["grid"]["padding"],
                              crs=config["grid"]["crs"])
    projected = [p.to_crs(grid.crs) for p in perimeters]
    log(f"  grid            {grid!r}")

    # Nesting, with a tolerance. Reprojection and rasterisation leave a few
    # boundary cells of any real perimeter outside its successor, and those
    # are noise, not a contradiction; a loss above the tolerance is one.
    nesting = PerimeterSeries(projected).check_nesting(grid)
    tolerance = config["nesting_tolerance"]
    bad = [r for r in nesting if r["fraction_lost"] > tolerance]
    if bad:
        worst = max(bad, key=lambda r: r["fraction_lost"])
        raise ValueError(
            f"{len(bad)} pair(s) lose more than {100 * tolerance:g}% of the earlier "
            f"perimeter; worst loses {100 * worst['fraction_lost']:.2f}% between "
            f"{worst['earlier']} and {worst['later']}. Fix the observations before "
            "chaining, or raise nesting_tolerance if this is rasterisation noise.")
    for r in nesting:
        if r["fraction_lost"] > 0:
            log(f"  nesting         {100 * r['fraction_lost']:.2f}% of the "
                f"{r['earlier'].strftime('%H:%MZ')} perimeter lies outside the "
                f"{r['later'].strftime('%H:%MZ')} one (within tolerance)")

    lines = [f"Chain of {len(config['evolutions'])} evolution(s) from {config['source']}",
             f"grid: {grid!r}", ""]
    evolutions, comparisons = [], []
    projected_series = PerimeterSeries(projected)
    for k, (a, b) in enumerate(zip(projected[:-1], projected[1:])):
        raw_options = config["evolutions"][k]["options"]
        options = _evolution_kwargs(raw_options)
        provenance = None
        if "metric" not in options and config["borrow"]:
            # Sparse vectors: a pair with none at either end borrows the
            # nearest later one in the series, and the report says whose.
            from validation import series_metric
            options["metric"], provenance = series_metric(
                projected_series, k, lb=options.get("lb"),
                wind_mps=options.get("wind_mps"),
                infer_from_geometry=options.get("infer_from_geometry", False),
                grid=grid, borrow=True)
            options = {"metric": options["metric"]}
        ev = GaugeEvolution(a, b, grid, **options)
        if provenance is not None:
            ev.metric_provenance = provenance
        start_j, end_j = ev.endpoint_jaccard()
        head = (f"segment {k}: {a.time.strftime('%Y-%m-%d %H:%M:%SZ')} -> "
                f"{b.time.strftime('%H:%M:%SZ')} ({ev.interval / 60:.0f} min)")
        body = (f"  metric          {ev.metric_provenance}\n"
                f"  endpoints       jaccard {start_j:.4f} / {end_j:.4f}\n"
                f"  violations      {ev.monotonicity_violations}\n"
                f"  shortest path   {ev.shortest_journey():.1f}")
        log("  " + head)
        for line in body.splitlines():
            log("  " + line)
        lines += [head, body]
        evolutions.append(ev)

        if config["validate"] and ev.metric.is_anisotropic:
            from validation import MetricComparison
            comparison = MetricComparison(a, b, grid, metric=ev.metric,
                                          num_steps=ev.num_steps)
            comparisons.append(comparison)
            lines += ["", comparison.report()]
        elif config["validate"]:
            lines.append("  (validation skipped: the metric is isotropic)")
        lines.append("")

    simulation_start = config["simulation_start"] or projected[0].time
    arrival = ArrivalTime.chain(evolutions, simulation_start)
    lines += ["combined field", *(f"  {k:24} {v}" for k, v in arrival.summary().items())]
    log(f"  combined        {arrival!r}")

    written = {}
    outputs = config["outputs"]
    figure = config["figure"]
    dpi = int(figure.get("dpi", 300))
    if outputs.get("filmstrip"):
        from figures import chain_filmstrip
        chain_filmstrip(arrival, times=figure.get("times"),
                        panels_per_segment=int(figure.get("panels_per_segment", 3)),
                        max_columns=int(figure.get("max_columns", 4)),
                        path=outputs["filmstrip"], title=figure.get("title"), dpi=dpi)
        written["filmstrip"] = outputs["filmstrip"]
    if outputs.get("animation"):
        from figures import chain_animate
        chain_animate(arrival, outputs["animation"],
                      frames_per_segment=int(figure.get("frames_per_segment", 8)),
                      fps=int(figure.get("fps", 6)),
                      hold_frames=int(figure.get("hold_frames", 0)),
                      times=figure.get("times"), title=figure.get("title"),
                      dpi=int(figure.get("gif_dpi", 100)))
        written["animation"] = outputs["animation"]
    if outputs.get("arrival_map"):
        from figures import chain_arrival_map
        chain_arrival_map(arrival, path=outputs["arrival_map"],
                          contour_minutes=figure.get("contour_minutes"),
                          scale=str(figure.get("arrival_scale", "segments")),
                          title=figure.get("title"), dpi=dpi)
        written["arrival_map"] = outputs["arrival_map"]

    tign_kwargs = {k: v for k, v in config["tign"].items()
                   if k in ("outside", "inside", "speed", "sentinel")}
    wants_array = any(outputs.get(k) for k in ("netcdf", "geotiff", "wrfinput"))
    if wants_array:
        dt = outputs.get("dt")
        if dt is None and not outputs.get("wrfinput"):
            raise ValueError("outputs.dt (the model time step) is needed to check the array")
        if dt is not None:
            report = arrival.check(float(dt), **tign_kwargs)
            lines.append(f"  init_no_fire check passes at dt={float(dt):g} "
                         f"(min tign {report['min_tign']:.1f} s)")
        tign = arrival.tign_g(**tign_kwargs)
        if outputs.get("netcdf"):
            written["netcdf"] = arrival.to_netcdf(outputs["netcdf"], tign=tign)
        if outputs.get("geotiff"):
            written["geotiff"] = arrival.to_geotiff(outputs["geotiff"], tign=tign)
        if outputs.get("wrfinput"):
            r = arrival.write_wrfinput(outputs["wrfinput"], tign=tign,
                                       method=outputs.get("method", "nearest"),
                                       dt=float(dt) if dt is not None else None)
            written["wrfinput"] = outputs["wrfinput"]
            lines.append(f"  wrfinput: {r['nodes_inside_grid']} nodes sampled, "
                         f"{r['nodes_filled']} filled, check passes at DT={r['dt']:g}")

    report = "\n".join(lines)
    if outputs.get("report"):
        with open(outputs["report"], "w") as handle:
            handle.write(report + "\n")
        written["report"] = outputs["report"]
    for name, path in written.items():
        log(f"  wrote {name:12} {path}")

    return {"arrival": arrival, "evolutions": evolutions, "comparisons": comparisons,
            "report": report, "written": written, "grid": grid}


def _load_perimeters(config):
    """
    Every distinct observation, in order, loaded once.

    The perimeter shared by two consecutive evolutions is the same object in
    both, which is what makes the join check exact rather than approximate.
    """
    specs = []
    for k, ev in enumerate(config["evolutions"]):
        if k == 0:
            specs.append(ev["initial"])
        specs.append(ev["final"])

    perimeters = []
    for spec in specs:
        perimeter = Perimeter.from_shapefile(spec["path"], time=spec["time"],
                                             ros=spec["ros"], lb=spec["lb"])
        if perimeter.time is None:
            raise ValueError(f"{spec['path']} carries no time and none was given in the YAML")
        perimeters.append(perimeter)

    for k, (a, b) in enumerate(zip(perimeters[:-1], perimeters[1:])):
        if b.time <= a.time:
            raise ValueError(
                f"evolution {k} runs backwards or nowhere: {a.time.isoformat()} -> "
                f"{b.time.isoformat()}")
    return perimeters


def _evolution_kwargs(options):
    """Translate the YAML metric block into GaugeEvolution arguments."""
    kwargs = {}
    if options.get("lb") is not None:
        kwargs["lb"] = float(options["lb"])
    if options.get("wind_mps") is not None:
        kwargs["wind_mps"] = float(options["wind_mps"])
    if options.get("infer_from_geometry"):
        kwargs["infer_from_geometry"] = True
    if options.get("metric") == "euclidean":
        from metrics import EuclideanMetric
        kwargs["metric"] = EuclideanMetric()
    elif options.get("metric") not in (None, "elliptical", "auto"):
        raise ValueError(f"unknown metric {options['metric']!r}; use 'euclidean' or leave it out")
    return kwargs


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def main(argv=None):
    """Run a chain described in a YAML file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evolve a chain of perimeter pairs described in a YAML file, "
                    "combine them into one arrival-time field, and write the "
                    "outputs the file asks for.")
    parser.add_argument("config", help="the YAML chain description")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run(config, log=(lambda *_: None) if args.quiet else print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
