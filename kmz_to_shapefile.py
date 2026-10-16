#!/usr/bin/env python
"""Convert temporally-referenced fire-perimeter KMZ/KML files into shapefiles.

Each polygon in the KMZ is written out as its own shapefile, named after the
timestamp attached to the polygon.  Velocity ("rate of spread") vectors that
accompany the perimeters are matched to their perimeter by time and encoded
into the attribute table, alongside the time reference itself.

Two output modes are supported:

  individual  each shapefile holds exactly one perimeter (the default)
  additive    each shapefile holds that perimeter plus every earlier one
  both        write both sets side by side

Example
-------
    python kmz_to_shapefile.py fire.kmz -o out/ --mode both --write-vectors
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from xml.etree import ElementTree as ET

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

KML_CRS = "EPSG:4326"          # KML coordinates are always WGS84 lon/lat
GEOD = Geod(ellps="WGS84")

# ExtendedData / description keys that carry velocity information.
SPEED_KEYS = re.compile(r"(speed|velocit|\bros\b|rate.?of.?spread|spread.?rate)", re.I)
DIR_KEYS = re.compile(r"(direction|bearing|heading|azimuth|\bdir\b|\baz\b)", re.I)
U_KEYS = re.compile(r"^(u|vx|u_?vel|vel_?u|east(ward)?(_?vel)?)$", re.I)
V_KEYS = re.compile(r"^(v|vy|v_?vel|vel_?v|north(ward)?(_?vel)?)$", re.I)

# Units appearing next to a speed value, normalised to metres per second.
CHAIN_M = 20.1168  # one surveyor's chain, exactly 66 ft

UNIT_TO_MPS = {
    "m/s": 1.0, "mps": 1.0, "ms-1": 1.0, "m s-1": 1.0,
    "km/h": 1 / 3.6, "kph": 1 / 3.6, "kmh": 1 / 3.6,
    "mph": 0.44704, "mi/h": 0.44704,
    "ft/s": 0.3048, "fps": 0.3048,
    "ft/min": 0.00508, "ft/m": 0.00508,
    "m/min": 1 / 60.0, "m/h": 1 / 3600.0,
    # Chains per hour, the standard US fire-behaviour unit (FARSITE, BehavePlus).
    "ch/h": CHAIN_M / 3600.0, "ch/hr": CHAIN_M / 3600.0, "chph": CHAIN_M / 3600.0,
    "ch h-1": CHAIN_M / 3600.0, "chains/hour": CHAIN_M / 3600.0,
    "chains/h": CHAIN_M / 3600.0, "chains per hour": CHAIN_M / 3600.0,
    "ch/min": CHAIN_M / 60.0, "chains/min": CHAIN_M / 60.0,
    "kt": 0.514444, "kts": 0.514444, "knot": 0.514444, "knots": 0.514444,
}

# Longest names first, so "ch/hr" is not matched as "ch/h" + "r" and "km/h"
# is not matched as "m/h".
UNITS_BY_LENGTH = sorted(UNIT_TO_MPS, key=len, reverse=True)


def normalise_unit(text: str) -> str:
    """The canonical spelling of a speed unit, or a ValueError naming the options."""
    key = str(text).strip().lower().replace("_", "/")
    if key in UNIT_TO_MPS:
        return key
    for unit in UNITS_BY_LENGTH:
        if key == unit.replace(" ", ""):
            return unit
    raise ValueError(f"unknown speed unit {text!r}; one of {', '.join(sorted(UNIT_TO_MPS))}")


def unit_in(text) -> str | None:
    """A speed unit written somewhere in a string, if any."""
    if text is None:
        return None
    lowered = str(text).lower()
    for unit in UNITS_BY_LENGTH:
        if unit in lowered:
            return unit
    return None

TIMESTAMP_PATTERNS = [
    re.compile(r"(\d{4})-(\d{2})-(\d{2})[T _](\d{2}):?(\d{2})(?::?(\d{2}))?"),
    re.compile(r"(\d{4})(\d{2})(\d{2})[T_ ]?(\d{2})(\d{2})(\d{2})?"),
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def local_name(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def child_text(elem: ET.Element, name: str) -> str | None:
    for kid in elem:
        if local_name(kid.tag) == name and kid.text and kid.text.strip():
            return kid.text.strip()
    return None


def iter_children(elem: ET.Element, name: str) -> Iterator[ET.Element]:
    for kid in elem:
        if local_name(kid.tag) == name:
            yield kid


def parse_time(text: str | None) -> datetime | None:
    """Parse a KML time value, or dig a timestamp out of free text."""
    if not text:
        return None
    raw = text.strip()
    iso = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y", "%Y%m%d", "%Y%m%d%H%M", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    for pattern in TIMESTAMP_PATTERNS:
        m = pattern.search(raw)
        if m:
            y, mo, d, hh, mm, ss = m.groups()
            return datetime(int(y), int(mo), int(d), int(hh), int(mm),
                            int(ss or 0), tzinfo=timezone.utc)
    return None


def to_float(value) -> float | None:
    """Pull a number out of a value that may carry a unit suffix."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(m.group()) if m else None


def speed_to_mps(value, default_unit: str = "m/s", key: str | None = None) -> float | None:
    """
    Convert a speed to m/s.

    The unit is taken from, in order: the value itself ("556 ch/h"), the
    attribute's name ("ros_chph", "Speed (ch/h)"), and finally ``default_unit``
    - which is what --speed-units sets, for files that carry bare numbers in
    some unit other than m/s.
    """
    number = to_float(value)
    if number is None:
        return None
    unit = unit_in(value) or unit_in(key) or normalise_unit(default_unit)
    return number * UNIT_TO_MPS[unit]


def stamp(dt: datetime | None, index: int) -> str:
    """Filename-safe timestamp; falls back to a sequence number."""
    if dt is None:
        return f"unknown_{index:04d}"
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------
# KML parsing
# --------------------------------------------------------------------------

@dataclass
class Feature:
    """One KML Placemark, flattened."""
    name: str | None
    description: str | None
    geometry: BaseGeometry | None
    time_begin: datetime | None
    time_end: datetime | None
    extended: dict[str, str] = field(default_factory=dict)
    folder: str = ""
    source_kmz: str = ""
    source_kml: str = ""

    @property
    def time(self) -> datetime | None:
        return self.time_begin or self.time_end

    @property
    def duration_s(self) -> float | None:
        if self.time_begin and self.time_end and self.time_end > self.time_begin:
            return (self.time_end - self.time_begin).total_seconds()
        return None


def parse_coordinates(text: str | None) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    if not text:
        return coords
    for token in text.replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return coords


def parse_geometry(elem: ET.Element) -> BaseGeometry | None:
    """Build a shapely geometry from a KML geometry element."""
    tag = local_name(elem.tag)

    if tag == "Point":
        coords = parse_coordinates(child_text(elem, "coordinates"))
        return Point(coords[0]) if coords else None

    if tag in ("LineString", "LinearRing"):
        coords = parse_coordinates(child_text(elem, "coordinates"))
        return LineString(coords) if len(coords) >= 2 else None

    if tag == "Polygon":
        shell: list[tuple[float, float]] = []
        holes: list[list[tuple[float, float]]] = []
        for outer in iter_children(elem, "outerBoundaryIs"):
            for ring in iter_children(outer, "LinearRing"):
                shell = parse_coordinates(child_text(ring, "coordinates"))
        for inner in iter_children(elem, "innerBoundaryIs"):
            for ring in iter_children(inner, "LinearRing"):
                hole = parse_coordinates(child_text(ring, "coordinates"))
                if len(hole) >= 4:
                    holes.append(hole)
        if len(shell) < 4:
            return None
        try:
            poly = Polygon(shell, holes)
            return poly if poly.is_valid else poly.buffer(0)
        except Exception:
            return None

    if tag == "MultiGeometry":
        parts = [g for kid in elem if (g := parse_geometry(kid)) is not None]
        return combine(parts)

    if tag == "Track":  # gx:Track — coordinates live in gx:coord elements
        coords = []
        for kid in iter_children(elem, "coord"):
            if kid.text:
                bits = kid.text.split()
                if len(bits) >= 2:
                    coords.append((float(bits[0]), float(bits[1])))
        return LineString(coords) if len(coords) >= 2 else None

    if tag in ("MultiTrack", "Model"):
        parts = [g for kid in elem if (g := parse_geometry(kid)) is not None]
        return combine(parts)

    return None


def combine(parts: Sequence[BaseGeometry]) -> BaseGeometry | None:
    """Merge geometry parts, keeping polygons separate from lines/points."""
    parts = [p for p in parts if p is not None and not p.is_empty]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    polys = [p for p in parts if p.geom_type in ("Polygon", "MultiPolygon")]
    if polys:
        merged = unary_union(polys)
        if merged.geom_type == "Polygon":
            merged = MultiPolygon([merged])
        return merged
    lines = [p for p in parts if p.geom_type in ("LineString", "MultiLineString")]
    if lines and len(lines) == len(parts):
        return MultiLineString([ln for p in lines for ln in
                                (p.geoms if p.geom_type == "MultiLineString" else [p])])
    points = [p for p in parts if p.geom_type == "Point"]
    if points and len(points) == len(parts):
        return MultiPoint(points)
    return unary_union(parts)


def parse_extended_data(elem: ET.Element) -> dict[str, str]:
    """Read ExtendedData in all three KML flavours."""
    data: dict[str, str] = {}
    for ext in iter_children(elem, "ExtendedData"):
        for kid in ext:
            tag = local_name(kid.tag)
            if tag == "Data":
                key = kid.get("name")
                val = child_text(kid, "value")
                if key and val is not None:
                    data[key] = val
            elif tag == "SchemaData":
                for simple in iter_children(kid, "SimpleData"):
                    key = simple.get("name")
                    if key and simple.text:
                        data[key] = simple.text.strip()
            elif kid.text and kid.text.strip():
                data[tag] = kid.text.strip()
    return data


def parse_description_table(description: str | None) -> dict[str, str]:
    """Many KML exporters bury attributes in an HTML table in <description>."""
    if not description or "<" not in description:
        return {}
    data: dict[str, str] = {}
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", description, re.I | re.S)
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.I | re.S)
        if len(cells) >= 2:
            key = re.sub(r"<[^>]+>", "", cells[0]).strip()
            val = re.sub(r"<[^>]+>", "", cells[1]).strip()
            if key:
                data.setdefault(key, val)
    return data


def parse_time_primitive(elem: ET.Element) -> tuple[datetime | None, datetime | None]:
    for ts in iter_children(elem, "TimeStamp"):
        when = parse_time(child_text(ts, "when"))
        if when:
            return when, None
    for span in iter_children(elem, "TimeSpan"):
        begin = parse_time(child_text(span, "begin"))
        end = parse_time(child_text(span, "end"))
        if begin or end:
            return begin, end
    for track in iter_children(elem, "Track"):
        whens = [parse_time(k.text) for k in iter_children(track, "when")]
        whens = [w for w in whens if w]
        if whens:
            return min(whens), max(whens) if len(whens) > 1 else None
    return None, None


def walk_kml(elem: ET.Element, folder: str, inherited: tuple, ctx: dict) -> Iterator[Feature]:
    """Recursively yield Placemarks, letting folders supply names and times."""
    for kid in elem:
        tag = local_name(kid.tag)
        if tag == "Placemark":
            yield build_feature(kid, folder, inherited, ctx)
        elif tag in ("Document", "Folder"):
            label = child_text(kid, "name") or tag
            sub_folder = f"{folder}/{label}" if folder else label
            times = parse_time_primitive(kid)
            yield from walk_kml(kid, sub_folder,
                                times if any(times) else inherited, ctx)
        elif tag in ("kml", "NetworkLink"):
            yield from walk_kml(kid, folder, inherited, ctx)


def build_feature(pm: ET.Element, folder: str, inherited: tuple, ctx: dict) -> Feature:
    name = child_text(pm, "name")
    description = child_text(pm, "description")
    begin, end = parse_time_primitive(pm)
    extended = parse_extended_data(pm)
    extended.update({k: v for k, v in parse_description_table(description).items()
                     if k not in extended})

    if begin is None and end is None:
        begin, end = inherited

    if begin is None:  # last resort: a timestamp written into the name or data
        for candidate in (name, extended.get("time"), extended.get("Time"),
                          extended.get("timestamp"), extended.get("date"), folder):
            begin = parse_time(candidate)
            if begin:
                break

    geoms = [g for kid in pm if (g := parse_geometry(kid)) is not None]
    return Feature(
        name=name,
        description=description,
        geometry=combine(geoms),
        time_begin=begin,
        time_end=end,
        extended=extended,
        folder=folder,
        source_kmz=ctx.get("kmz", ""),
        source_kml=ctx.get("kml", ""),
    )


def read_kml_bytes(data: bytes, kmz_name: str, kml_name: str) -> list[Feature]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        print(f"  ! skipping {kml_name}: not parseable XML ({exc})", file=sys.stderr)
        return []
    ctx = {"kmz": kmz_name, "kml": kml_name}
    return list(walk_kml(root, "", (None, None), ctx))


def read_source(path: Path) -> list[Feature]:
    """Read every KML document inside a .kmz, or a bare .kml file."""
    if path.suffix.lower() == ".kmz":
        features: list[Feature] = []
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not members:
                print(f"  ! {path.name} contains no .kml document", file=sys.stderr)
            for member in sorted(members):
                features.extend(read_kml_bytes(zf.read(member), path.name, member))
        return features
    return read_kml_bytes(path.read_bytes(), path.name, path.name)


# --------------------------------------------------------------------------
# velocity handling
# --------------------------------------------------------------------------

@dataclass
class Velocity:
    speed_mps: float | None = None
    direction_deg: float | None = None   # degrees clockwise from true north
    u_mps: float | None = None           # eastward component
    v_mps: float | None = None           # northward component
    length_m: float | None = None        # displacement encoded by the vector
    speed_min: float | None = None
    speed_max: float | None = None
    count: int = 0
    source: str = "none"

    def fill_components(self) -> None:
        if self.u_mps is None and self.speed_mps is not None and self.direction_deg is not None:
            theta = math.radians(self.direction_deg)
            self.u_mps = self.speed_mps * math.sin(theta)
            self.v_mps = self.speed_mps * math.cos(theta)
        if self.speed_mps is None and self.u_mps is not None and self.v_mps is not None:
            self.speed_mps = math.hypot(self.u_mps, self.v_mps)
            self.direction_deg = (math.degrees(math.atan2(self.u_mps, self.v_mps)) + 360) % 360


def velocity_from_attributes(data: dict[str, str],
                             default_unit: str = "m/s") -> Velocity | None:
    """Look for speed / direction / u / v already written into the KML.

    ``default_unit`` applies to any speed or component that names no unit of
    its own, in its value or its key.
    """
    vel = Velocity(source="extended_data")
    for key, raw in data.items():
        k = key.strip()
        if vel.speed_mps is None and SPEED_KEYS.search(k) and not DIR_KEYS.search(k):
            vel.speed_mps = speed_to_mps(raw, default_unit, key=k)
        elif vel.direction_deg is None and DIR_KEYS.search(k):
            vel.direction_deg = to_float(raw)
        elif vel.u_mps is None and U_KEYS.match(k):
            vel.u_mps = speed_to_mps(raw, default_unit, key=k)
        elif vel.v_mps is None and V_KEYS.match(k):
            vel.v_mps = speed_to_mps(raw, default_unit, key=k)
    if vel.speed_mps is None and vel.u_mps is None:
        return None
    vel.fill_components()
    vel.count = 1
    return vel


def vector_segments(geom: BaseGeometry) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """First and last vertex of every line in a (multi)line geometry."""
    lines: list[LineString] = []
    if geom is None:
        return []
    if geom.geom_type == "LineString":
        lines = [geom]
    elif geom.geom_type == "MultiLineString":
        lines = list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        lines = [g for g in geom.geoms if g.geom_type == "LineString"]
    out = []
    for line in lines:
        coords = list(line.coords)
        if len(coords) >= 2:
            out.append((coords[0][:2], coords[-1][:2]))
    return out


def velocity_from_geometry(features: Sequence[Feature], dt_s: float | None) -> Velocity | None:
    """Derive speed and bearing from the drawn vectors themselves.

    ``dt_s`` is the time the vector represents.  Without it we can still report
    the displacement and bearing, but not a speed.
    """
    us: list[float] = []
    vs: list[float] = []
    lengths: list[float] = []
    for feat in features:
        span = feat.duration_s or dt_s
        for (lon0, lat0), (lon1, lat1) in vector_segments(feat.geometry):
            az, _, dist = GEOD.inv(lon0, lat0, lon1, lat1)
            if not math.isfinite(dist) or dist == 0:
                continue
            lengths.append(dist)
            magnitude = dist / span if span else dist
            theta = math.radians(az)
            us.append(magnitude * math.sin(theta))
            vs.append(magnitude * math.cos(theta))
    if not lengths:
        return None

    u_mean, v_mean = float(np.mean(us)), float(np.mean(vs))
    magnitudes = [math.hypot(u, v) for u, v in zip(us, vs)]
    have_time = any(f.duration_s for f in features) or bool(dt_s)
    vel = Velocity(
        speed_mps=float(np.mean(magnitudes)) if have_time else None,
        direction_deg=(math.degrees(math.atan2(u_mean, v_mean)) + 360) % 360,
        u_mps=u_mean if have_time else None,
        v_mps=v_mean if have_time else None,
        length_m=float(np.mean(lengths)),
        speed_min=float(np.min(magnitudes)) if have_time else None,
        speed_max=float(np.max(magnitudes)) if have_time else None,
        count=len(magnitudes),
        source="vector_geometry",
    )
    return vel


def merge_velocities(*candidates: "Velocity | None") -> Velocity:
    """Combine velocity estimates, earlier candidates winning field by field.

    An explicit ``speed`` attribute on the perimeter and a bearing taken from
    the drawn vector are perfectly happy to come from different places, so the
    fields are merged rather than picking one estimate wholesale.
    """
    merged = Velocity()
    used: list[str] = []
    for cand in candidates:
        if cand is None:
            continue
        contributed = False
        for attr in ("speed_mps", "direction_deg", "length_m",
                     "speed_min", "speed_max"):
            if getattr(merged, attr) is None and getattr(cand, attr) is not None:
                setattr(merged, attr, getattr(cand, attr))
                contributed = True
        if not merged.count and cand.count:
            merged.count = cand.count
        if contributed:
            used.append(cand.source)
    merged.source = "+".join(dict.fromkeys(used)) or "none"
    merged.fill_components()  # u/v always recomputed, so they match the speed used
    return merged


def match_vectors(perimeter: Feature, vectors: Sequence[Feature],
                  tolerance_s: float) -> list[Feature]:
    """Find the velocity vectors belonging to one perimeter.

    Matching is by timestamp first, then by the folder the features live in,
    then by nearest time within ``tolerance_s``.
    """
    if not vectors:
        return []
    t = perimeter.time
    if t is not None:
        exact = [v for v in vectors if v.time == t]
        if exact:
            return exact
    same_folder = [v for v in vectors if v.folder and v.folder == perimeter.folder]
    if same_folder:
        return same_folder
    if t is not None and tolerance_s > 0:
        near = [v for v in vectors
                if v.time and abs((v.time - t).total_seconds()) <= tolerance_s]
        if near:
            closest = min(abs((v.time - t).total_seconds()) for v in near)
            return [v for v in near
                    if abs((v.time - t).total_seconds()) <= closest + 1e-6]
    return []


# --------------------------------------------------------------------------
# attribute assembly
# --------------------------------------------------------------------------

def sanitize_field(name: str, taken: set[str]) -> str:
    """Shapefile DBF field names are limited to 10 characters and must be unique."""
    clean = re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_") or "field"
    if clean[0].isdigit():
        clean = "f_" + clean
    clean = clean[:10]
    candidate, n = clean, 1
    while candidate.lower() in taken:
        suffix = str(n)
        candidate = clean[:10 - len(suffix)] + suffix
        n += 1
    taken.add(candidate.lower())
    return candidate


def measure(geom: BaseGeometry) -> tuple[float | None, float | None]:
    """Area (m^2) and perimeter (m) measured on the ellipsoid."""
    if geom is None or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
        return None, None
    try:
        area, perim = GEOD.geometry_area_perimeter(geom)
        return abs(area), abs(perim)
    except Exception:
        return None, None


def build_record(perimeter: Feature, velocity: Velocity, index: int,
                 extra_fields: Sequence[str]) -> dict:
    t = perimeter.time
    area_m2, perim_m = measure(perimeter.geometry)
    record = {
        "step_idx": index,
        "time_iso": t.astimezone(timezone.utc).isoformat() if t else None,
        "time_begin": perimeter.time_begin.astimezone(timezone.utc).isoformat()
                      if perimeter.time_begin else None,
        "time_end": perimeter.time_end.astimezone(timezone.utc).isoformat()
                    if perimeter.time_end else None,
        "time_epoch": int(t.timestamp()) if t else None,
        "time_label": stamp(t, index),
        "duration_s": perimeter.duration_s,
        "name": (perimeter.name or "")[:254],
        "folder": perimeter.folder[:254],
        "src_kmz": perimeter.source_kmz[:254],
        "src_kml": perimeter.source_kml[:254],
        "speed_mps": velocity.speed_mps,
        "speed_kph": velocity.speed_mps * 3.6 if velocity.speed_mps is not None else None,
        "speed_min": velocity.speed_min,
        "speed_max": velocity.speed_max,
        "dir_deg": velocity.direction_deg,
        "vel_u_mps": velocity.u_mps,
        "vel_v_mps": velocity.v_mps,
        "vec_len_m": velocity.length_m,
        "n_vectors": velocity.count,
        "vel_src": velocity.source,
        "area_m2": area_m2,
        "area_ha": area_m2 / 1e4 if area_m2 is not None else None,
        "perim_m": perim_m,
    }
    taken = {k.lower() for k in record}
    for key in extra_fields:
        record[sanitize_field(key, taken)] = str(perimeter.extended[key])[:254] \
            if key in perimeter.extended else None
    return record


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

NUMERIC_FIELDS = ("step_idx", "time_epoch", "duration_s", "speed_mps", "speed_kph",
                  "speed_min", "speed_max", "dir_deg", "vel_u_mps", "vel_v_mps",
                  "vec_len_m", "n_vectors", "area_m2", "area_ha", "perim_m")


def coerce_numeric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep the numeric columns numeric, so an all-empty speed column still
    lands in the DBF as a number field rather than text."""
    for column in NUMERIC_FIELDS:
        if column in gdf.columns:
            gdf[column] = pd.to_numeric(gdf[column], errors="coerce")
    return gdf


def write_shapefile(gdf: gpd.GeoDataFrame, path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"  ! {path.name} exists, skipping (use --overwrite)", file=sys.stderr)
        return False
    coerce_numeric(gdf).to_file(path, driver="ESRI Shapefile", engine="pyogrio")
    return True


def write_metadata(path: Path, records: Sequence[dict], mode: str,
                   crs: str, field_map: dict[str, str], speed_units: str = "m/s") -> None:
    """Sidecar metadata: a JSON dump plus an FGDC-flavoured .shp.xml."""
    meta = {
        "shapefile": path.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "kmz_to_shapefile.py",
        "mode": mode,
        "crs": crs,
        "feature_count": len(records),
        "time_reference": {
            "begin": min((r["time_iso"] for r in records if r["time_iso"]), default=None),
            "end": max((r["time_iso"] for r in records if r["time_iso"]), default=None),
            "timezone": "UTC",
        },
        "velocity": {
            "units": {"speed_mps": "m s-1", "speed_kph": "km h-1",
                      "dir_deg": "degrees clockwise from true north",
                      "vel_u_mps": "eastward m s-1", "vel_v_mps": "northward m s-1",
                      "vec_len_m": "vector displacement, metres"},
            "sources": sorted({r["vel_src"] for r in records}),
            "input_speed_unit_assumed": speed_units,
        },
        "truncated_field_names": field_map,
        "features": records,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))

    begin = meta["time_reference"]["begin"] or ""
    end = meta["time_reference"]["end"] or begin
    speeds = [r["speed_mps"] for r in records if r["speed_mps"] is not None]
    summary = (f"mean speed {np.mean(speeds):.3f} m/s over {len(speeds)} perimeter(s)"
               if speeds else "no velocity vectors resolved")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<metadata>
  <idinfo>
    <citation><citeinfo>
      <title>{path.stem}</title>
      <pubdate>{datetime.now(timezone.utc).strftime('%Y%m%d')}</pubdate>
      <origin>kmz_to_shapefile.py</origin>
    </citeinfo></citation>
    <descript>
      <abstract>Fire perimeter polygon(s) extracted from KMZ, written in
      {mode} mode. Attributes carry the perimeter time reference and the
      associated rate-of-spread velocity vector ({summary}).</abstract>
      <purpose>Time-referenced fire perimeter with encoded spread velocity.</purpose>
    </descript>
    <timeperd><timeinfo><rngdates>
      <begdate>{begin}</begdate>
      <enddate>{end}</enddate>
    </rngdates></timeinfo><current>observation time (UTC)</current></timeperd>
  </idinfo>
  <spref><horizsys><cordsysn><geogcsn>{crs}</geogcsn></cordsysn></horizsys></spref>
</metadata>
"""
    path.with_suffix(".shp.xml").write_text(xml)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def collect_inputs(paths: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            found.extend(sorted(q for q in p.rglob("*")
                                if q.suffix.lower() in (".kmz", ".kml")))
        elif any(ch in raw for ch in "*?["):
            found.extend(sorted(Path().glob(raw)))
        elif p.exists():
            found.append(p)
        else:
            print(f"  ! input not found: {raw}", file=sys.stderr)
    seen, unique = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique


def split_features(features: Sequence[Feature]) -> tuple[list[Feature], list[Feature]]:
    """Separate perimeter polygons from velocity vectors."""
    perimeters, vectors = [], []
    for feat in features:
        if feat.geometry is None or feat.geometry.is_empty:
            continue
        if feat.geometry.geom_type in ("Polygon", "MultiPolygon"):
            perimeters.append(feat)
        else:
            vectors.append(feat)
    return perimeters, vectors


def infer_step_seconds(perimeters: Sequence[Feature]) -> float | None:
    """Median spacing between perimeter times, used when vectors carry no span."""
    times = sorted(p.time for p in perimeters if p.time)
    if len(times) < 2:
        return None
    deltas = [(b - a).total_seconds() for a, b in zip(times, times[1:])
              if (b - a).total_seconds() > 0]
    return float(np.median(deltas)) if deltas else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert time-referenced fire perimeter KMZ/KML files to shapefiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+",
                        help="input .kmz/.kml files, directories, or glob patterns")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="directory to write the shapefiles into")
    parser.add_argument("-m", "--mode", choices=("individual", "additive", "both"),
                        default="individual",
                        help="individual: one perimeter per shapefile; "
                             "additive: each file also contains all earlier perimeters")
    parser.add_argument("--prefix", default="perimeter_",
                        help="filename prefix for the output shapefiles")
    parser.add_argument("--crs", default=KML_CRS,
                        help="CRS to write the shapefiles in")
    parser.add_argument("--speed-units", default="m/s", type=normalise_unit,
                        help="unit of speed values in the KML that carry no unit of "
                             "their own, e.g. ch/h for chains per hour; every output "
                             "is converted to m/s regardless")
    parser.add_argument("--vector-duration", type=float, default=None,
                        help="seconds a drawn velocity vector represents; defaults to "
                             "the vector's own TimeSpan, else the perimeter interval")
    parser.add_argument("--time-tolerance", type=float, default=0.0,
                        help="seconds of slack when matching vectors to perimeters")
    parser.add_argument("--write-vectors", action="store_true",
                        help="also write the matched velocity vectors as line shapefiles")
    parser.add_argument("--no-metadata", action="store_true",
                        help="skip the .json / .shp.xml metadata sidecars")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing shapefiles")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written without writing it")
    args = parser.parse_args(argv)

    inputs = collect_inputs(args.inputs)
    if not inputs:
        print("No input KMZ/KML files found.", file=sys.stderr)
        return 1

    features: list[Feature] = []
    for path in inputs:
        print(f"Reading {path}")
        found = read_source(path)
        print(f"  {len(found)} placemark(s)")
        features.extend(found)

    perimeters, vectors = split_features(features)
    if not perimeters:
        print("No polygon features found in the input.", file=sys.stderr)
        return 1

    undated = sum(1 for p in perimeters if p.time is None)
    if undated:
        print(f"  ! {undated} perimeter(s) have no time reference; "
              f"they will be named by sequence number", file=sys.stderr)

    perimeters.sort(key=lambda f: (f.time is None,
                                   f.time or datetime.min.replace(tzinfo=timezone.utc),
                                   f.name or ""))
    step_s = args.vector_duration or infer_step_seconds(perimeters)
    print(f"\n{len(perimeters)} perimeter(s), {len(vectors)} velocity vector feature(s)"
          + (f", step {step_s:g}s" if step_s else ""))

    # Two perimeters can share a timestamp; keep their filenames distinct.
    labels: dict[str, int] = {}

    # Every ExtendedData key seen anywhere becomes a column, so the tables line up.
    extra_fields = sorted({k for p in perimeters for k in p.extended})

    out_dir = Path(args.output_dir).expanduser()
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.speed_units != "m/s":
        print(f"  unit-less speeds in the KML read as {args.speed_units} "
              f"(x{UNIT_TO_MPS[args.speed_units]:.6g} to m/s); units written in a "
              "value or its key still take precedence")

    records: list[dict] = []
    geometries: list[BaseGeometry] = []
    matched_vectors: list[list[Feature]] = []
    for i, perimeter in enumerate(perimeters, start=1):
        matched = match_vectors(perimeter, vectors, args.time_tolerance)
        velocity = merge_velocities(
            velocity_from_attributes(perimeter.extended, args.speed_units),
            next((v for m in matched
                  if (v := velocity_from_attributes(m.extended, args.speed_units))), None),
            velocity_from_geometry(matched, step_s),
        )
        record = build_record(perimeter, velocity, i, extra_fields)
        seen = labels.get(record["time_label"], 0) + 1
        labels[record["time_label"]] = seen
        if seen > 1:
            record["time_label"] = f"{record['time_label']}_{seen:02d}"
        records.append(record)
        geometries.append(perimeter.geometry)
        matched_vectors.append(matched)

    field_map = {k: v for k, v in zip(extra_fields,
                                      list(records[0].keys())[-len(extra_fields):])} \
        if extra_fields else {}

    written = 0
    modes = ("individual", "additive") if args.mode == "both" else (args.mode,)
    for mode in modes:
        for i in range(len(records)):
            rows = records[: i + 1] if mode == "additive" else [records[i]]
            geoms = geometries[: i + 1] if mode == "additive" else [geometries[i]]
            label = records[i]["time_label"]
            suffix = "_cumulative" if mode == "additive" else ""
            target = out_dir / f"{args.prefix}{label}{suffix}.shp"

            gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=KML_CRS)
            if args.crs and args.crs != KML_CRS:
                gdf = gdf.to_crs(args.crs)

            speed = records[i]["speed_mps"]
            speed_txt = f"{speed:.3f} m/s" if speed is not None else "n/a"
            print(f"  {target.name}: {len(gdf)} polygon(s), speed {speed_txt}, "
                  f"{records[i]['n_vectors']} vector(s)")
            if args.dry_run:
                continue
            if write_shapefile(gdf, target, args.overwrite):
                written += 1
                if not args.no_metadata:
                    write_metadata(target, rows, mode, str(gdf.crs), field_map,
                                   args.speed_units)

            if args.write_vectors:
                vec_rows, vec_geoms = [], []
                idx_range = range(i + 1) if mode == "additive" else [i]
                for j in idx_range:
                    for feat in matched_vectors[j]:
                        if feat.geometry is None or feat.geometry.is_empty:
                            continue
                        vec_rows.append({
                            "step_idx": records[j]["step_idx"],
                            "time_iso": records[j]["time_iso"],
                            "name": (feat.name or "")[:254],
                            "speed_mps": records[j]["speed_mps"],
                            "dir_deg": records[j]["dir_deg"],
                        })
                        vec_geoms.append(feat.geometry)
                if vec_rows:
                    vec_gdf = gpd.GeoDataFrame(pd.DataFrame(vec_rows),
                                               geometry=vec_geoms, crs=KML_CRS)
                    if args.crs and args.crs != KML_CRS:
                        vec_gdf = vec_gdf.to_crs(args.crs)
                    write_shapefile(vec_gdf,
                                    out_dir / f"{args.prefix}{label}{suffix}_vectors.shp",
                                    args.overwrite)

    if args.dry_run:
        print("\nDry run — nothing written.")
    else:
        print(f"\nWrote {written} shapefile(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
