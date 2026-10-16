"""
Tests for speed units in kmz_to_shapefile.py.

The real Mill Fire vectors turned out to be in chains per hour with no unit
written next to them, and were read as m/s - a factor of 179. These pin the
three places a unit can come from (the value, the key, --speed-units) and
their precedence, and run a KMZ through the CLI to check the DBF comes out
in m/s.
"""

import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kmz_to_shapefile import (CHAIN_M, UNIT_TO_MPS, main, normalise_unit, speed_to_mps,
                              unit_in, velocity_from_attributes)

CHPH = CHAIN_M / 3600.0


def test_chains_per_hour_is_the_surveyors_chain():
    assert CHAIN_M == pytest.approx(66 * 0.3048)
    assert 556.43 * CHPH == pytest.approx(3.109, abs=1e-3)


def test_bare_numbers_take_the_default_unit():
    assert speed_to_mps("556.43") == pytest.approx(556.43)
    assert speed_to_mps("556.43", "ch/h") == pytest.approx(556.43 * CHPH)
    assert speed_to_mps(556.43, "ch/h") == pytest.approx(556.43 * CHPH)
    assert speed_to_mps("2", "mph") == pytest.approx(0.89408)


def test_a_unit_in_the_value_beats_the_default():
    assert speed_to_mps("10 km/h", "ch/h") == pytest.approx(10 / 3.6)
    assert speed_to_mps("3 m/s", "ch/h") == pytest.approx(3.0)


def test_a_unit_in_the_key_beats_the_default_but_not_the_value():
    assert speed_to_mps("100", "m/s", key="ros_chph") == pytest.approx(100 * CHPH)
    assert speed_to_mps("100", "m/s", key="Spread rate (ch/h)") == pytest.approx(100 * CHPH)
    assert speed_to_mps("5 mph", "m/s", key="ros_chph") == pytest.approx(5 * 0.44704)
    # Keys with no unit in them fall through to the default.
    assert speed_to_mps("100", "ch/h", key="speed") == pytest.approx(100 * CHPH)


def test_longer_unit_names_win_over_their_substrings():
    assert unit_in("12 km/h") == "km/h"        # not m/h
    assert unit_in("12 ft/min") == "ft/min"    # not ft/m
    assert unit_in("12 ch/hr") == "ch/hr"      # not ch/h
    assert unit_in("just a number 12") is None


@pytest.mark.parametrize("spelling", ["ch/h", "CH/H", "ch/hr", "chph", "ch_h", "chains/hour",
                                      "chains per hour", "ch h-1"])
def test_chain_spellings_normalise(spelling):
    unit = normalise_unit(spelling)
    assert UNIT_TO_MPS[unit] == pytest.approx(CHPH)


def test_unknown_units_are_refused_with_the_options():
    with pytest.raises(ValueError, match="furlongs.*one of"):
        normalise_unit("furlongs/fortnight")


def test_components_are_converted_too():
    vel = velocity_from_attributes({"u": "300", "v": "400"}, "ch/h")
    assert vel.u_mps == pytest.approx(300 * CHPH)
    assert vel.v_mps == pytest.approx(400 * CHPH)
    assert vel.speed_mps == pytest.approx(500 * CHPH)


def _kmz(path, speed_text):
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Folder><name>a</name>
 <Placemark><name>perimeter</name><TimeStamp><when>2024-07-23T01:37:31Z</when></TimeStamp>
  <ExtendedData><Data name="ROS"><value>{speed_text}</value></Data>
                <Data name="direction"><value>38</value></Data></ExtendedData>
  <Polygon><outerBoundaryIs><LinearRing><coordinates>
  -122.445,41.495,0 -122.435,41.495,0 -122.435,41.505,0 -122.445,41.505,0 -122.445,41.495,0
  </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
</Folder>
<Folder><name>b</name>
 <Placemark><name>perimeter</name><TimeStamp><when>2024-07-23T01:40:35Z</when></TimeStamp>
  <ExtendedData><Data name="ROS"><value>{speed_text}</value></Data>
                <Data name="direction"><value>38</value></Data></ExtendedData>
  <Polygon><outerBoundaryIs><LinearRing><coordinates>
  -122.450,41.490,0 -122.425,41.490,0 -122.425,41.512,0 -122.450,41.512,0 -122.450,41.490,0
  </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
</Folder>
</Document></kml>"""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("doc.kml", kml)
    return str(path)


def test_cli_converts_bare_chains_per_hour_to_mps(tmp_path):
    import geopandas as gpd

    from perimeter import Perimeter

    kmz = _kmz(tmp_path / "fire.kmz", "556.43")
    out = tmp_path / "out"
    assert main([kmz, "-o", str(out), "--speed-units", "ch/h", "--no-metadata"]) == 0
    shp = sorted(out.glob("*.shp"))[0]
    row = gpd.read_file(shp).iloc[0]
    assert row["speed_mps"] == pytest.approx(556.43 * CHPH, rel=1e-6)
    assert row["dir_deg"] == pytest.approx(38.0)
    assert Perimeter.from_shapefile(str(shp)).ros_speed == pytest.approx(3.109, abs=1e-3)


def test_cli_still_honours_a_unit_written_in_the_value(tmp_path):
    import geopandas as gpd

    kmz = _kmz(tmp_path / "fire.kmz", "3 m/s")
    out = tmp_path / "out"
    assert main([kmz, "-o", str(out), "--speed-units", "ch/h", "--no-metadata"]) == 0
    row = gpd.read_file(sorted(out.glob("*.shp"))[0]).iloc[0]
    assert row["speed_mps"] == pytest.approx(3.0)


def test_cli_records_the_assumed_unit_in_the_metadata(tmp_path):
    import json

    kmz = _kmz(tmp_path / "fire.kmz", "556.43")
    out = tmp_path / "out"
    assert main([kmz, "-o", str(out), "--speed-units", "chph"]) == 0
    meta = json.loads(sorted(out.glob("*.json"))[0].read_text())
    assert meta["velocity"]["input_speed_unit_assumed"] == "chph"
    assert meta["velocity"]["units"]["speed_mps"] == "m s-1"


def test_cli_rejects_an_unknown_unit(tmp_path):
    kmz = _kmz(tmp_path / "fire.kmz", "1")
    with pytest.raises(SystemExit):
        main([kmz, "-o", str(tmp_path / "out"), "--speed-units", "leagues/day"])
