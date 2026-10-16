#!/usr/bin/env python3
"""
Were the ROS vectors in a KMZ measured, or calculated from the perimeters?

For every consecutive perimeter pair A -> B this recomputes the speeds and
bearings that the usual "calculate it from the perimeters" recipes would give,
and sets each one against the ROS value the file carries. A vector that was
derived from the perimeters reproduces one recipe almost exactly (ratio ~1,
tiny scatter, bearing agreeing); a measured or modelled vector scatters
against all of them.

Perimeters are paired consecutively in time order, whatever the input:

  KMZ/KML (preferred)  each perimeter exactly as drawn, the raw values, and
                       any drawn arrows. This is the question's natural input:
                       a vector calculated from the perimeters was calculated
                       from these shapes, not from the running union the
                       chain's _cumulative shapefiles hold.
  chain.yaml           exactly the evolutions chain.py would run (paths,
                       times and any ros overrides included).
  shapefiles/folder    as validation.py reads them. Speeds already in m/s.

Only derived numbers are printed - no coordinates - so the output can be
shared without sharing the perimeters.

Usage:
    python ros_forensics.py mill_fire.kmz --speed-units ch/h
    python ros_forensics.py test_chain.yaml
    python ros_forensics.py ../perimeters/            # folder, like validation.py
    python ros_forensics.py test_chain.yaml --csv ros_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import os

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from pyproj import CRS, Transformer
from shapely.ops import transform

from kmz_to_shapefile import (
    KML_CRS, UNIT_TO_MPS, collect_inputs, match_vectors, normalise_unit,
    read_source, split_features, velocity_from_attributes,
)
from perimeter import Perimeter, PerimeterSeries, ros_from_speed_bearing


# ----------------------------------------------------------------------
# Geometry helpers (everything in a local metric CRS)
# ----------------------------------------------------------------------

def bearing_of(dx, dy):
    """Degrees clockwise from grid north."""
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def bearing_gap(a, b):
    """Smallest angle between two bearings, degrees."""
    if a is None or b is None:
        return float("nan")
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def boundary_points(geom, spacing):
    """Points every `spacing` metres along a polygon's boundary."""
    pts = []
    for line in getattr(geom.boundary, "geoms", [geom.boundary]):
        n = max(int(line.length // spacing), 1)
        pts.extend(line.interpolate(i * line.length / n) for i in range(n))
    return pts


def directed_advance(a, b, spacing):
    """
    Furthest any point of B's boundary lies outside A: max over B of dist(p, A).
    Returns (distance, bearing from the nearest point of A to that p).
    """
    best, best_bearing = 0.0, None
    for p in boundary_points(b, spacing):
        d = a.distance(p)
        if d > best:
            q = nearest_points(a, p)[0]
            best, best_bearing = d, bearing_of(p.x - q.x, p.y - q.y)
    return best, best_bearing


def support_advance(a, b, bearing):
    """Head advance along a bearing: furthest extent of B minus that of A."""
    th = math.radians(bearing)
    ux, uy = math.sin(th), math.cos(th)

    def extent(g):
        xs, ys = [], []
        for poly in getattr(g, "geoms", [g]):
            x, y = poly.exterior.coords.xy
            xs.extend(x)
            ys.extend(y)
        return max(ux * x + uy * y for x, y in zip(xs, ys))

    return extent(b) - extent(a)


def ray_advance(a, b, bearing):
    """
    Distance along the bearing from A's leading point to B's boundary - what
    you get by drawing an arrow from the old head to the new edge.
    """
    th = math.radians(bearing)
    ux, uy = math.sin(th), math.cos(th)
    best, head = -math.inf, None
    for poly in getattr(a, "geoms", [a]):
        for x, y in poly.exterior.coords:
            s = ux * x + uy * y
            if s > best:
                best, head = s, (x, y)
    reach = 4.0 * max(math.sqrt(b.area), 1.0) + b.hausdorff_distance(a)
    ray = LineString([head, (head[0] + ux * reach, head[1] + uy * reach)])
    hit = ray.intersection(b.boundary)
    if hit.is_empty:
        return 0.0
    pts = [hit] if hit.geom_type == "Point" else [
        g for g in getattr(hit, "geoms", []) if g.geom_type == "Point"]
    if not pts:
        return 0.0
    return max(Point(head).distance(p) for p in pts)


def equivalent_radius(g):
    return math.sqrt(g.area / math.pi)


def arrow_lines(geom):
    """Every line in a (multi)line geometry, all vertices kept."""
    if geom is None:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "LineString"]


def arrow_parts(line):
    """
    Tail, tip, length and bearing of one drawn arrow.

    The tip is the vertex furthest from the tail rather than the last one: an
    arrow drawn with its arrowhead as part of the line ends on a barb, which
    sits short of the tip and off to one side.
    """
    coords = list(line.coords)
    tail = coords[0]
    tip = max(coords, key=lambda c: math.hypot(c[0] - tail[0], c[1] - tail[1]))
    length = math.hypot(tip[0] - tail[0], tip[1] - tail[1])
    return (Point(tail[:2]), Point(tip[:2]), Point(coords[-1][:2]), length,
            bearing_of(tip[0] - tail[0], tip[1] - tail[1]) if length else None)


def mean_bearing(bearings):
    """Circular mean, degrees."""
    th = np.radians(bearings)
    return (math.degrees(math.atan2(np.sin(th).sum(), np.cos(th).sum())) + 360.0) % 360.0


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------

KMZ_SUFFIXES = (".kmz", ".kml")


def is_kmz_input(inputs):
    """KMZ/KML files, or a folder holding them."""
    def kmz(path):
        return path.lower().endswith(KMZ_SUFFIXES) or (
            os.path.isdir(path) and any(f.lower().endswith(KMZ_SUFFIXES)
                                        for f in os.listdir(path)))
    return all(kmz(p) for p in inputs)


def load_kmz(inputs, speed_units, tolerance_s):
    """
    Perimeters straight from the KMZ, each exactly as drawn.

    Read with kmz_to_shapefile.py's parser, but never dissolved into the
    running union the chain's _cumulative files hold: a vector calculated
    from the perimeters was calculated from these shapes.
    """
    features = []
    for path in collect_inputs(inputs):
        features.extend(read_source(path))
    polygons, vectors = split_features(features)
    polygons = [f for f in polygons if f.time is not None]

    # Two polygons at one time are parts of one perimeter.
    by_time = {}
    for f in polygons:
        by_time.setdefault(f.time, []).append(f)

    perimeters, extras = [], []
    for when in sorted(by_time):
        parts = by_time[when]
        geometry = shapely.union_all([f.geometry for f in parts])
        # Timed vectors: exact (or tolerance) time only. The folder fallback in
        # match_vectors hands an arrow-less perimeter every arrow in the folder.
        if any(v.time for v in vectors):
            matched = [v for v in vectors if v.time and
                       abs((v.time - when).total_seconds()) <= tolerance_s]
        else:
            matched = match_vectors(parts[0], vectors, tolerance_s)
        vel = next((v for f in parts
                    if (v := velocity_from_attributes(f.extended, speed_units))), None)
        if vel is None:
            vel = next((v for m in matched
                        if (v := velocity_from_attributes(m.extended, speed_units))), None)
        ros = (ros_from_speed_bearing(vel.speed_mps, vel.direction_deg)
               if vel and vel.speed_mps and vel.direction_deg is not None else None)
        perimeters.append(Perimeter(geometry=geometry, crs=CRS.from_user_input(KML_CRS),
                                    time=when, ros=ros,
                                    source=parts[0].name or when.isoformat()))
        extras.append({
            "lines": [ln for m in matched for ln in arrow_lines(m.geometry)],
            "keys": set().union(*(f.extended for f in parts), *(m.extended for m in matched)),
            "speed_only": vel.speed_mps if vel and ros is None else None,
        })
    return PerimeterSeries(perimeters), extras


def load(inputs, speed_units="m/s", tolerance_s=0.0):
    """
    The perimeter series, projected onto the grid CRS the chain would use.

    KMZ/KML goes through load_kmz; a single .yaml/.yml through chain.py's
    own loader; anything else through PerimeterSeries.from_shapefiles, as
    validation.py does.
    """
    grid_crs, extras = None, None
    if is_kmz_input(inputs):
        series, extras = load_kmz(inputs, speed_units, tolerance_s)
        source = f"KMZ ({', '.join(os.path.basename(p.rstrip('/')) for p in inputs)})"
    elif len(inputs) == 1 and inputs[0].lower().endswith((".yaml", ".yml")):
        from chain import _load_perimeters, load_config
        config = load_config(inputs[0])
        series = PerimeterSeries(_load_perimeters(config))
        grid_crs = config["grid"]["crs"]
        source = f"{len(config['evolutions'])} evolutions in {inputs[0]}"
    else:
        target = inputs[0] if len(inputs) == 1 and (
            os.path.isdir(inputs[0]) or any(c in inputs[0] for c in "*?[")) else inputs
        series = PerimeterSeries.from_shapefiles(target)
        source = f"{len(series)} shapefiles"
    if len(series) < 2 or any(p.time is None for p in series):
        raise SystemExit("need at least two perimeters, all with a time")

    grid = series.common_grid(cell_size=25.0, crs=grid_crs)
    true_bearings = [p.ros_bearing for p in series]
    projected = series.on_grid(grid)
    rows = [{
        "time": p.time,
        "geometry_m": p.geometry.buffer(0),
        "speed": p.ros_speed,
        # Grid bearing: to_crs has rotated the vector by the meridian
        # convergence, so it lines up with the projected geometry.
        "bearing": p.ros_bearing,
        # As written in the file, for the rounding check.
        "true_bearing": tb,
        "source": os.path.basename(p.source or ""),
        "arrows_m": [],
        "bearing_src": "attribute" if p.ros_bearing is not None else None,
        "keys": set(),
    } for p, tb in zip(projected, true_bearings)]

    if extras:
        to_grid = Transformer.from_crs(KML_CRS, grid.crs, always_xy=True).transform
        for row, extra in zip(rows, extras):
            row["arrows_m"] = [transform(to_grid, ln) for ln in extra["lines"]]
            row["keys"] = extra["keys"]
            if row["bearing"] is None and row["arrows_m"]:
                # No direction attribute: the arrows are the only direction
                # there is. Grid bearing, since the arrows are projected too.
                brgs = [b for b in (arrow_parts(g)[4] for g in row["arrows_m"])
                        if b is not None]
                if brgs:
                    row["bearing"] = mean_bearing(brgs)
                    row["bearing_src"] = "arrow"
            if row["speed"] is None and extra["speed_only"]:
                # A speed with no direction still has a magnitude to compare.
                row["speed"] = extra["speed_only"]
    return rows, grid.crs, source


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------

RECIPES = {
    "support":  "head advance along the stated bearing (extent B - extent A) / dt",
    "ray":      "arrow from A's head along the bearing to B's edge / dt",
    "hausdorff_dir": "max distance of B's edge outside A / dt",
    "hausdorff_dir_cum": "max distance of B's edge outside the fire so far "
                         "(every perimeter up to A) / dt",
    "hausdorff": "symmetric Hausdorff distance A<->B / dt",
    "centroid": "centroid displacement / dt",
    "radius":   "equivalent-circle radius growth / dt",
}


def analyse(rows, spacing, which):
    """One record per pair. `which` = 'end' uses B's vector, 'start' uses A's."""
    out = []
    # The whole fire so far: the same as A for cumulative perimeters, but not
    # when each polygon is only the growth since the one before.
    footprint = [rows[0]["geometry_m"]]
    for r in rows[1:]:
        footprint.append(footprint[-1].union(r["geometry_m"]))

    for iA, (A, B) in enumerate(zip(rows, rows[1:])):
        dt = (B["time"] - A["time"]).total_seconds()
        if dt <= 0:
            continue
        owner = B if which == "end" else A
        stated, bearing = owner["speed"], owner["bearing"]
        a, b = A["geometry_m"], B["geometry_m"]

        rec = {"pair": len(out), "bearing_src": owner["bearing_src"], "file_start": A["source"], "file_end": B["source"],
               "t_start": A["time"].isoformat(), "t_end": B["time"].isoformat(),
               "dt_s": dt, "stated_mps": stated, "stated_bearing": bearing,
               "area_growth_ha": (b.area - a.area) / 1e4,
               # 0 when B contains A; 1 when they do not overlap at all
               "a_outside_b": a.difference(b).area / max(a.area, 1.0)}

        hd, hd_bearing = directed_advance(a, b, spacing)
        hdc, hdc_bearing = directed_advance(footprint[iA], b, spacing)
        ca, cb = a.centroid, b.centroid
        cand = {
            "hausdorff_dir": (hd / dt, hd_bearing),
            "hausdorff_dir_cum": (hdc / dt, hdc_bearing),
            "hausdorff": (shapely.hausdorff_distance(a, b, densify=0.05) / dt, None),
            "centroid": (ca.distance(cb) / dt, bearing_of(cb.x - ca.x, cb.y - ca.y)),
            "radius": ((equivalent_radius(b) - equivalent_radius(a)) / dt, None),
        }
        if bearing is not None:
            cand["support"] = (support_advance(a, b, bearing) / dt, bearing)
            cand["ray"] = (ray_advance(a, b, bearing) / dt, bearing)
        for name, (v, brg) in cand.items():
            rec[f"{name}_mps"] = v
            rec[f"{name}_ratio"] = v / stated if stated else float("nan")
            rec[f"{name}_dbearing"] = bearing_gap(brg, bearing)

        # Where drawn arrows sit: tail on A's edge and tip on B's means they
        # were drawn from the perimeters. A tail that is not on A is traced to
        # the earlier perimeter it does sit on, and the interval back to it.
        arrows = [arrow_parts(g) for g in owner["arrows_m"]]
        if arrows:
            med = lambda xs: float(np.median(xs))
            tails = [t for t, *_ in arrows]
            length = med([ln for *_, ln, _ in arrows])
            origin = []
            for t in tails:
                dists = [rows[k]["geometry_m"].boundary.distance(t) for k in range(iA + 1)]
                k = int(np.argmin(dists))
                origin.append((iA - k, dists[k], (B["time"] - rows[k]["time"]).total_seconds()))
            lag, odist, span = (med([o[i] for o in origin]) for i in range(3))
            rec.update({
                "arrow_tail_to_A_m": med([a.boundary.distance(t) for t in tails]),
                "arrow_tail_to_fire_m": med([footprint[iA].distance(t) for t in tails]),
                "arrow_tip_to_B_m": med([b.boundary.distance(tp) for _, tp, *_ in arrows]),
                "arrow_last_to_B_m": med([b.boundary.distance(lv) for _, _, lv, *_ in arrows]),
                "arrow_len_m": length,
                "arrow_len_over_stated_dt": length / (stated * dt) if stated else float("nan"),
                # 0 = tail on A, 1 = on the perimeter before A, ...
                "arrow_tail_perims_back": lag,
                "arrow_tail_origin_m": odist,
                "arrow_origin_span_s": span,
                "arrow_len_over_stated_span": (length / (stated * span)
                                               if stated and span else float("nan")),
            })
        out.append(rec)
    return out


def summarise(records, label):
    print(f"\n=== Stated vector taken from the {label} perimeter "
          f"({len(records)} pairs) ===")
    usable = [r for r in records if r["stated_mps"]]
    if not usable:
        print("  no stated speeds - nothing to compare")
        return
    print(f"  {'recipe':17s} {'median':>8s} {'IQR':>13s} {'within5%':>9s} "
          f"{'r(log)':>7s} {'|dBrg|':>7s}")
    stated = np.array([r["stated_mps"] for r in usable])
    for name, desc in RECIPES.items():
        key = f"{name}_ratio"
        pairs = [(r[key], r["stated_mps"], r[f"{name}_dbearing"])
                 for r in usable if key in r and np.isfinite(r[key])]
        if not pairs:
            continue
        ratio = np.array([p[0] for p in pairs])
        s = np.array([p[1] for p in pairs])
        calc = ratio * s
        good = (calc > 0) & (s > 0)
        r_log = (np.corrcoef(np.log(calc[good]), np.log(s[good]))[0, 1]
                 if good.sum() > 2 else float("nan"))
        dbrg = np.nanmedian([p[2] for p in pairs]) if any(
            np.isfinite(p[2]) for p in pairs) else float("nan")
        q1, q3 = np.percentile(ratio, [25, 75])
        within = np.mean(np.abs(ratio - 1) < 0.05)
        print(f"  {name:17s} {np.median(ratio):8.3f} {q1:6.3f}-{q3:6.3f} "
              f"{within:9.0%} {r_log:7.3f} {dbrg:7.1f}")
    arrows = [r for r in usable if "arrow_tail_to_A_m" in r]
    if arrows:
        med = lambda k: np.median([r[k] for r in arrows])
        on_a = [r for r in arrows if r["arrow_tail_to_A_m"] < 5.0]
        print(f"  drawn arrows ({len(arrows)} pairs, tail on A's edge in {len(on_a)}): "
              f"tail->A {med('arrow_tail_to_A_m'):.0f} m, tail->fire so far "
              f"{med('arrow_tail_to_fire_m'):.0f} m")
        print(f"    tip->B edge {med('arrow_tip_to_B_m'):.0f} m (last vertex "
              f"{med('arrow_last_to_B_m'):.0f} m), length/(stated*dt) "
              f"{med('arrow_len_over_stated_dt'):.3f}")


def pair_table(records):
    """One line per pair: the columns that separate matching pairs from the rest."""
    print("\n=== Per pair (vector from the LATER perimeter) ===")
    print(f"  {'pair':>4s} {'t_end (UTC)':>17s} {'dt_s':>6s} {'stated':>7s} "
          f"{'hd':>7s} {'hd_cum':>7s} {'tail>A':>7s} {'back':>4s} {'tail>P':>6s} "
          f"{'tip>B':>6s} {'L/sdt':>6s} {'L/sT':>6s} {'A_out':>5s}")
    f = lambda v, w, d: f"{v:{w}.{d}f}" if v is not None and np.isfinite(v) else f"{'-':>{w}s}"
    for r in records:
        g = r.get
        print(f"  {r['pair']:4d} {r['t_end'][:16]:>17s} {r['dt_s']:6.0f} "
              f"{f(g('stated_mps'), 7, 3)} {f(g('hausdorff_dir_ratio'), 7, 3)} "
              f"{f(g('hausdorff_dir_cum_ratio'), 7, 3)} {f(g('arrow_tail_to_A_m'), 7, 0)} "
              f"{f(g('arrow_tail_perims_back'), 4, 0)} {f(g('arrow_tail_origin_m'), 6, 0)} "
              f"{f(g('arrow_tip_to_B_m'), 6, 0)} {f(g('arrow_len_over_stated_dt'), 6, 3)} "
              f"{f(g('arrow_len_over_stated_span'), 6, 3)} {f(g('a_outside_b'), 5, 2)}")
    print("  stated m/s; hd, hd_cum = recipe/stated; tail>A, tail>P = arrow tail to A's edge and\n"
          "  to the nearest earlier perimeter's edge (m), 'back' = how many perimeters before A that edge was first mapped;\n"
          "  tip>B m; L/sdt = arrow length/(stated*dt); L/sT = the same over the interval\n"
          "  back to the tail's perimeter; A_out = share of A outside B (1 = no overlap)")


def value_texture(rows, speed_units):
    """Rounding and repetition in the stated numbers themselves."""
    raw = [r["speed"] / UNIT_TO_MPS[speed_units] for r in rows if r["speed"]]
    brg = [r["true_bearing"] for r in rows if r["true_bearing"] is not None]
    print(f"\n=== Stated values ({len(raw)} speeds, {len(brg)} bearings) ===")
    if raw:
        raw = np.array(raw)
        print(f"  speed in {speed_units}: min {raw.min():.3f}  median {np.median(raw):.3f}  "
              f"max {raw.max():.3f}  distinct {len(np.unique(np.round(raw, 6)))}")
        print(f"  integer-valued {np.mean(np.isclose(raw, np.round(raw))):.0%}, "
              f"multiple of 5 {np.mean(np.isclose(raw % 5, 0) | np.isclose(raw % 5, 5)):.0%}")
    keys = sorted({k for r in rows for k in r["keys"]})
    if keys:
        print(f"  attribute keys in the KMZ: {', '.join(keys)}")
    from_arrows = sum(1 for r in rows if r["bearing_src"] == "arrow")
    if from_arrows:
        print(f"  no direction attribute: bearing taken from the drawn arrows for "
              f"{from_arrows} perimeters (so support/ray and |dBrg| test the arrows)")
    if brg:
        brg = np.array(brg)
        print(f"  bearings on a 5-deg grid {np.mean(np.isclose(brg % 5, 0) | np.isclose(brg % 5, 5)):.0%}, "
              f"16-point compass (22.5) {np.mean(np.isclose(brg % 22.5, 0) | np.isclose(brg % 22.5, 22.5)):.0%}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="KMZ/KML files or a folder of them (preferred); or a "
                         "chain.yaml, or perimeter shapefiles / a folder of them")
    ap.add_argument("--speed-units", default="m/s", type=normalise_unit,
                    help="unit the KMZ's ROS values are written in when they "
                         "carry none (Mill Fire: ch/h). Also the unit the "
                         "rounding check shows speeds in")
    ap.add_argument("--time-tolerance", type=float, default=0.0,
                    help="KMZ only: seconds of slack when matching drawn arrows "
                         "to perimeters")
    ap.add_argument("--spacing", type=float, default=10.0,
                    help="metres between boundary samples for the Hausdorff recipes")
    ap.add_argument("--table", action="store_true",
                    help="also print one line per pair (later-perimeter convention)")
    ap.add_argument("--csv", help="optionally also write every pair's numbers "
                                  "(one row per pair and convention) to this file")
    args = ap.parse_args(argv)

    rows, crs, source = load(args.inputs, args.speed_units, args.time_tolerance)
    print(f"{source}: {len(rows)} perimeters, {len(rows) - 1} pairs, "
          f"{sum(1 for r in rows if r['speed'])} with a stated speed; working in {crs}")

    value_texture(rows, args.speed_units)
    results = {}
    for which, label in (("end", "LATER (interval ending at it)"),
                         ("start", "EARLIER (interval starting at it)")):
        results[which] = analyse(rows, args.spacing, which)
        summarise(results[which], label)

    if args.table:
        pair_table(results["end"])

    print("\nRecipes:")
    for name, desc in RECIPES.items():
        print(f"  {name:17s} {desc}")
    print("\nReading it: a calculated vector shows up as one recipe with median ~1.000,\n"
          "a tight IQR, most pairs within 5%, r(log) ~1 and |dBrg| ~0 - and drawn\n"
          "arrows whose tail sits on A's edge and tip on B's. A measured vector\n"
          "scatters against every recipe (IQR wide, r well below 1).")

    if args.csv:
        recs = [dict(r, vector_from=w) for w, rs in results.items() for r in rs]
        fields = sorted({k for r in recs for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(recs)
        print(f"\nWrote {len(recs)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
