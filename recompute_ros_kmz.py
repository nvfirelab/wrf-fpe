#!/usr/bin/env python3
"""
Recompute every pair's ROS vector from the perimeters, and write it into a
copy of the KMZ next to the original arrows.

The calculation is the chain's own geometric inference, metrics.infer_ros,
run on each consecutive pair of whole-fire footprints (the union of every
polygon up to that time - what the chain's _cumulative shapefiles hold):

  1. new growth  = cells burned in B but not in A
  2. advance     = each new cell's distance from A's edge
  3. leading edge = the furthest 1% of new cells (--quantile 0.99)
  4. speed       = their mean advance / the pair's own interval
  5. direction   = advance-weighted mean direction from the nearest point of
                   A's edge out to those cells

Each vector is drawn as an arrow from where that growth left A's edge, as
long as the advance it represents (speed x interval), and stamped with the
later perimeter's time so the Google Earth time slider shows it alongside
the original arrow for the same interval.

The output is the input KMZ unchanged plus one folder, "Recomputed ROS".
Its attribute names deliberately avoid the words kmz_to_shapefile.py and
ros_forensics.py read as speeds and directions, so run those on the
ORIGINAL file, not this one.

Usage:
    python recompute_ros_kmz.py mill_fire.kmz --speed-units ch/h
    python recompute_ros_kmz.py mill_fire.kmz --speed-units ch/h -o compare.kmz
"""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import shapely
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Point
from shapely.ops import transform

from kmz_to_shapefile import KML_CRS, UNIT_TO_MPS, normalise_unit
from metrics import infer_ros_detail
from perimeter import Perimeter, PerimeterSeries
from ros_forensics import load_kmz

GEOD = Geod(ellps="WGS84")
KML_NS = "http://www.opengis.net/kml/2.2"
COLOUR = "ffffffff"          # KML is aabbggrr: opaque white


# ----------------------------------------------------------------------
# The calculation
# ----------------------------------------------------------------------

def stated_arrow(lines):
    """Tail, tip (furthest vertex - the last one is an arrowhead barb), length, azimuth."""
    best = None
    for line in lines:
        coords = list(line.coords)
        lon0, lat0 = coords[0][:2]
        for lon, lat in coords[1:]:
            az, _, dist = GEOD.inv(lon0, lat0, lon, lat)
            if best is None or dist > best[2]:
                best = ((lon0, lat0), (lon, lat), dist, az % 360.0)
    return best


def recompute(path, speed_units, cell_size, quantile, tolerance_s):
    patches, extras = load_kmz([str(path)], speed_units, tolerance_s)
    if len(patches) < 2:
        raise SystemExit("need at least two timed perimeters")

    # Whole-fire footprints, as the chain sees them.
    footprints, running = [], None
    for p in patches:
        running = p.geometry if running is None else shapely.union_all([running, p.geometry])
        footprints.append(Perimeter(geometry=running, crs=CRS.from_user_input(KML_CRS),
                                    time=p.time))
    grid = PerimeterSeries(footprints).common_grid(cell_size=cell_size)
    to_ll = Transformer.from_crs(grid.crs, KML_CRS, always_xy=True).transform
    to_grid = Transformer.from_crs(KML_CRS, grid.crs, always_xy=True).transform
    patches_m = [transform(to_grid, p.geometry) for p in patches]

    results = []
    for k in range(len(patches) - 1):
        a, b = footprints[k], footprints[k + 1]
        dt = (b.time - a.time).total_seconds()
        stated = patches[k + 1].ros_speed or extras[k + 1]["speed_only"]
        rec = {"pair": k, "t_start": a.time, "t_end": b.time, "dt_s": dt,
               "stated_mps": stated}

        arrow = stated_arrow(extras[k + 1]["lines"])
        if arrow:
            rec["stated_az"] = arrow[3]
            rec["stated_len_m"] = arrow[2]
            # Which earlier polygon the stated arrow starts on, and so the
            # interval the producer divided by.
            tail = Point(to_grid(*arrow[0]))
            dists = [patches_m[j].boundary.distance(tail) for j in range(k + 1)]
            j = int(np.argmin(dists))
            rec["stated_back"] = k - j
            rec["stated_span_s"] = (b.time - patches[j].time).total_seconds()

        try:
            d = infer_ros_detail(a.mask(grid), b.mask(grid), grid, dt, quantile=quantile)
        except ValueError as exc:
            rec["skipped"] = str(exc)
            results.append(rec)
            continue

        u, v = d["ros"]
        speed = math.hypot(u, v)
        tx, ty = d["tail"]
        hx, hy = tx + u / speed * d["advance_m"], ty + v / speed * d["advance_m"]
        tail_ll, tip_ll = to_ll(tx, ty), to_ll(hx, hy)
        az, _, _ = GEOD.inv(*tail_ll, *tip_ll)
        rec.update({
            "recomp_mps": speed, "recomp_az": az % 360.0,
            "advance_m": d["advance_m"], "n_cells": d["n_cells"],
            "tail_ll": tail_ll, "tip_ll": tip_ll,
            "head_ll": arrowhead(tx, ty, hx, hy, to_ll),
            "ratio": speed / stated if stated else None,
        })
        results.append(rec)
    return results, grid


def arrowhead(tx, ty, hx, hy, to_ll):
    """Two barbs at the tip, in the grid's metres, returned as lon/lat."""
    length = math.hypot(hx - tx, hy - ty)
    barb = min(max(0.2 * length, 15.0), 200.0)
    back = math.atan2(ty - hy, tx - hx)
    return [to_ll(hx + barb * math.cos(back + s), hy + barb * math.sin(back + s))
            for s in (math.radians(25), -math.radians(25))]


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def _coords(points):
    return " ".join(f"{lon:.8f},{lat:.8f},0" for lon, lat in points)


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def placemark(rec, unit, cell_size, quantile):
    per = UNIT_TO_MPS[unit]
    fmt = lambda v, d=3: "" if v is None else f"{v:.{d}f}"
    stated_u = rec["stated_mps"] / per if rec["stated_mps"] else None
    recomp_u = rec["recomp_mps"] / per
    rows = [
        ("Interval", f"{_iso(rec['t_start'])} to {_iso(rec['t_end'])} ({rec['dt_s']:.0f} s)"),
        (f"Recomputed ROS ({unit})", fmt(recomp_u)),
        (f"Stated ROS ({unit})", fmt(stated_u)),
        ("Recomputed / stated", fmt(rec["ratio"])),
        ("Recomputed azimuth (deg true)", fmt(rec["recomp_az"], 1)),
        ("Stated arrow azimuth (deg true)", fmt(rec.get("stated_az"), 1)),
        ("Advance drawn (m)", fmt(rec["advance_m"], 0)),
        # The arrow always starts on the fire's edge at the previous time; the
        # edge it starts on may have been mapped earlier, and the stated
        # speed is divided by the time since then.
        ("Stated arrow starts on", "" if "stated_back" not in rec else
         ("the previous perimeter's new edge" if rec["stated_back"] == 0 else
          f"fire edge first mapped {rec['stated_back']} perimeters earlier; stated "
          f"speed divides by {rec['stated_span_s']:.0f} s, not {rec['dt_s']:.0f} s")),
        ("Method", f"metrics.infer_ros, {cell_size:g} m cells, quantile {quantile:g}, "
                   f"{rec['n_cells']} leading cells"),
    ]
    table = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>"
                    for k, v in rows)
    data = {
        "pair": rec["pair"], "t_start": _iso(rec["t_start"]), "t_end": _iso(rec["t_end"]),
        "dt_s": f"{rec['dt_s']:.0f}",
        # Names kept clear of the speed/direction patterns the converter reads.
        f"recomp_{unit.replace('/', 'p')}": fmt(recomp_u, 4),
        **({} if unit == "m/s" else {"recomp_mps": fmt(rec["recomp_mps"], 5)}),
        "recomp_az_deg": fmt(rec["recomp_az"], 2),
        f"stated_{unit.replace('/', 'p')}": fmt(stated_u, 4),
        "stated_az_deg": fmt(rec.get("stated_az"), 2),
        "recomp_over_stated": fmt(rec["ratio"], 4),
        "stated_perims_back": str(rec.get("stated_back", "")),
        "advance_m": fmt(rec["advance_m"], 1),
    }
    ext = "".join(f'<Data name="{k}"><value>{html.escape(str(v))}</value></Data>'
                  for k, v in data.items())
    name = f"{rec['pair']}: {recomp_u:.1f} {unit}"
    if stated_u:
        name += f" (stated {stated_u:.1f})"
    return (f"<Placemark><name>{html.escape(name)}</name>"
            f"<description><![CDATA[<table>{table}</table>]]></description>"
            f"<TimeStamp><when>{_iso(rec['t_end'])}</when></TimeStamp>"
            f"<Style><LineStyle><color>{COLOUR}</color><width>3</width></LineStyle></Style>"
            f"<ExtendedData>{ext}</ExtendedData>"
            "<MultiGeometry>"
            f"<LineString><tessellate>1</tessellate><coordinates>"
            f"{_coords([rec['tail_ll'], rec['tip_ll']])}</coordinates></LineString>"
            f"<LineString><tessellate>1</tessellate><coordinates>"
            f"{_coords([rec['head_ll'][0], rec['tip_ll'], rec['head_ll'][1]])}"
            "</coordinates></LineString>"
            "</MultiGeometry></Placemark>")


def folder(results, unit, cell_size, quantile):
    marks = [placemark(r, unit, cell_size, quantile) for r in results if "recomp_mps" in r]
    note = ("ROS recomputed from the perimeters with the chain's geometric inference "
            "(metrics.infer_ros) on whole-fire footprints. White arrows; the original "
            "arrows are untouched elsewhere in this file.")
    return (f'<Folder xmlns="{KML_NS}"><name>Recomputed ROS</name>'
            f"<description>{html.escape(note)}</description>{''.join(marks)}</Folder>")


def insert_folder(kml_text, fragment):
    """Put the new folder inside the document, whatever its root looks like."""
    closes = list(re.finditer(r"</(?:\w+:)?Document\s*>", kml_text))
    if closes:
        at = closes[-1].start()
        return kml_text[:at] + fragment + kml_text[at:]
    # No Document: wrap the existing root feature and the new folder in one.
    open_tag = re.search(r"<(?:\w+:)?kml\b[^>]*>", kml_text)
    close_tag = list(re.finditer(r"</(?:\w+:)?kml\s*>", kml_text))
    if not open_tag or not close_tag:
        raise SystemExit("could not find the <kml> root to add the folder to")
    a, b = open_tag.end(), close_tag[-1].start()
    return (kml_text[:a] + f'<Document xmlns="{KML_NS}">' + kml_text[a:b]
            + fragment + "</Document>" + kml_text[b:])


def write_output(src, dst, fragment):
    if src.suffix.lower() == ".kml":
        text = insert_folder(src.read_text(encoding="utf-8"), fragment)
        if dst.suffix.lower() == ".kml":
            dst.write_text(text, encoding="utf-8")
        else:
            with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
                zout.writestr("doc.kml", text)
        return
    with zipfile.ZipFile(src) as zin:
        kmls = [n for n in zin.namelist() if n.lower().endswith(".kml")]
        if not kmls:
            raise SystemExit(f"{src.name} holds no .kml document")
        # Google Earth opens doc.kml if there is one, else the first .kml.
        main = next((n for n in kmls if n.lower().endswith("doc.kml")), kmls[0])
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == main:
                    data = insert_folder(data.decode("utf-8"), fragment).encode("utf-8")
                zout.writestr(item, data)


# ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="the original KMZ or KML")
    ap.add_argument("-o", "--output", help="output KMZ (default <input>_recomputed.kmz)")
    ap.add_argument("--speed-units", default="m/s", type=normalise_unit,
                    help="unit the KMZ's ROS values are written in when they carry "
                         "none (Mill Fire: ch/h); the new vectors are written in it too")
    ap.add_argument("--cell-size", type=float, default=25.0,
                    help="grid resolution in metres - the chain's default")
    ap.add_argument("--quantile", type=float, default=0.99,
                    help="leading-edge quantile - infer_ros's default")
    ap.add_argument("--time-tolerance", type=float, default=0.0,
                    help="seconds of slack when matching stated arrows to perimeters")
    args = ap.parse_args(argv)

    src = Path(args.input).expanduser()
    dst = Path(args.output).expanduser() if args.output else \
        src.with_name(src.stem + "_recomputed.kmz")
    if dst.resolve() == src.resolve():
        raise SystemExit("refusing to overwrite the input; pass a different -o")

    results, grid = recompute(src, args.speed_units, args.cell_size, args.quantile,
                              args.time_tolerance)
    per = UNIT_TO_MPS[args.speed_units]
    unit = args.speed_units
    print(f"{len(results) + 1} perimeters, {len(results)} pairs; grid {grid!r}\n")
    print(f"  {'pair':>4s} {'t_end (UTC)':>17s} {'dt_s':>6s} {'stated':>8s} "
          f"{'recomp':>8s} {'ratio':>7s} {'st_az':>6s} {'rc_az':>6s} {'back':>4s}")
    f = lambda v, w, d: f"{v:{w}.{d}f}" if v is not None else f"{'-':>{w}s}"
    for r in results:
        if "skipped" in r:
            print(f"  {r['pair']:4d} {_iso(r['t_end'])[:16]:>17s} {r['dt_s']:6.0f}  "
                  f"skipped: {r['skipped']}")
            continue
        print(f"  {r['pair']:4d} {_iso(r['t_end'])[:16]:>17s} {r['dt_s']:6.0f} "
              f"{f(r['stated_mps'] / per if r['stated_mps'] else None, 8, 2)} "
              f"{f(r['recomp_mps'] / per, 8, 2)} {f(r['ratio'], 7, 3)} "
              f"{f(r.get('stated_az'), 6, 1)} {f(r['recomp_az'], 6, 1)} "
              f"{f(r.get('stated_back'), 4, 0)}")
    print(f"\n  speeds in {unit}; azimuths degrees true; back = perimeters before the "
          "previous one that the edge the stated arrow starts on was first mapped")

    write_output(src, dst, folder(results, unit, args.cell_size, args.quantile))
    drawn = sum(1 for r in results if "recomp_mps" in r)
    print(f"\nWrote {dst} ({drawn} recomputed vectors in the 'Recomputed ROS' folder)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
