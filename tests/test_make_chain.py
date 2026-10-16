"""Tests for make_chain.py: filename timestamps, ordering, pairing, the YAML it writes."""

import os
from datetime import datetime, timezone

import pytest

from chain import load_config
from make_chain import find_perimeters, main, render_yaml, time_from_filename


def _touch(folder, *names):
    for name in names:
        open(os.path.join(folder, name), "w").close()


def test_time_is_read_from_the_end_of_the_filename():
    when = time_from_filename("/x/mill_fire_20240723_013731Z.shp")
    assert when == datetime(2024, 7, 23, 1, 37, 31, tzinfo=timezone.utc)
    # kmz_to_shapefile's own form is accepted too.
    assert time_from_filename("perimeter_20240723T013731Z.shp") == when
    # The stamp must be at the end, and must be a real time.
    assert time_from_filename("20240723_013731Z_perimeter.shp") is None
    assert time_from_filename("mill_20240723_013731.shp") is None
    assert time_from_filename("mill_20241399_013731Z.shp") is None
    assert time_from_filename("notes.shp") is None


def test_perimeters_are_sorted_by_time_not_by_name(tmp_path):
    _touch(tmp_path, "b_20240723_020000Z.shp", "a_20240723_030000Z.shp",
           "c_20240723_010000Z.shp", "readme.shp", "x_20240723_010000Z_vectors.shp")
    found, skipped = find_perimeters(str(tmp_path))
    assert [os.path.basename(p) for _, p in found] == [
        "c_20240723_010000Z.shp", "b_20240723_020000Z.shp", "a_20240723_030000Z.shp"]
    assert [os.path.basename(p) for p in skipped] == ["readme.shp"]


def test_strict_mode_refuses_unstamped_files(tmp_path):
    _touch(tmp_path, "a_20240723_010000Z.shp", "b_20240723_020000Z.shp", "odd.shp")
    with pytest.raises(ValueError, match="odd.shp"):
        find_perimeters(str(tmp_path), strict=True)


def test_duplicate_times_are_refused(tmp_path):
    _touch(tmp_path, "a_20240723_010000Z.shp", "b_20240723_010000Z.shp")
    with pytest.raises(ValueError, match="same time"):
        find_perimeters(str(tmp_path))


def test_yaml_pairs_consecutively_and_loads_in_chain(tmp_path):
    _touch(tmp_path, "p_20240723_010000Z.shp", "p_20240723_013000Z.shp",
           "p_20240723_020000Z.shp", "p_20240723_030000Z.shp")
    found, _ = find_perimeters(str(tmp_path))
    text = render_yaml(found, str(tmp_path), lb=3.0, dt=2.0)
    path = tmp_path / "chain.yaml"
    path.write_text(text)

    config = load_config(str(path))
    evs = config["evolutions"]
    assert len(evs) == 3
    times = [e["initial"]["time"] for e in evs] + [evs[-1]["final"]["time"]]
    assert times == [w for w, _ in found]
    for a, b in zip(evs[:-1], evs[1:]):
        assert a["final"]["path"] == b["initial"]["path"]
        assert a["final"]["time"] == b["initial"]["time"]
    assert evs[0]["options"] == {"lb": 3.0}
    assert config["outputs"]["dt"] == 2.0
    assert config["validate"] is True
    # Paths are relative to the YAML, so the file survives being moved with its folder.
    assert "path: p_20240723_010000Z.shp" in text


def test_simulation_start_defaults_to_a_lead_before_the_first(tmp_path):
    _touch(tmp_path, "p_20240723_010000Z.shp", "p_20240723_020000Z.shp")
    found, _ = find_perimeters(str(tmp_path))
    text = render_yaml(found, str(tmp_path), lb=3.0, lead_hours=2.0)
    assert "simulation_start: 2024-07-22T23:00:00Z" in text
    with pytest.raises(ValueError, match="before the first perimeter"):
        render_yaml(found, str(tmp_path), lb=3.0,
                    simulation_start=datetime(2024, 7, 23, 1, 30, tzinfo=timezone.utc))


def test_missing_lb_is_left_as_a_commented_reminder(tmp_path):
    _touch(tmp_path, "p_20240723_010000Z.shp", "p_20240723_020000Z.shp")
    found, _ = find_perimeters(str(tmp_path))
    text = render_yaml(found, str(tmp_path))
    assert "# lb: 3.0" in text and "REQUIRED" in text
    config = load_config(str(_write(tmp_path, text)))
    assert config["evolutions"][0]["options"] == {}
    text = render_yaml(found, str(tmp_path), wind_mps=5.0)
    assert "wind_mps: 5" in text


def test_cli_writes_the_file_and_refuses_to_clobber(tmp_path, capsys):
    _touch(tmp_path, "p_20240723_010000Z.shp", "p_20240723_020000Z.shp", "stray.shp")
    assert main([str(tmp_path), "--lb", "3"]) == 0
    out = capsys.readouterr().out
    assert "skipping stray.shp" in out and "1 evolution(s)" in out
    assert (tmp_path / "chain.yaml").exists()
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--lb", "3"])
    assert main([str(tmp_path), "--lb", "3", "--overwrite", "--tign", "--prefix", "mill"]) == 0
    text = (tmp_path / "chain.yaml").read_text()
    assert "netcdf: mill_tign_g.nc" in text
    # The animation is the default picture; the filmstrip is offered commented out.
    assert "animation: mill_animation.gif" in text
    assert "# filmstrip: mill_filmstrip.png" in text


def test_cli_needs_two_perimeters(tmp_path):
    _touch(tmp_path, "p_20240723_010000Z.shp")
    with pytest.raises(SystemExit):
        main([str(tmp_path)])


def _write(folder, text):
    path = folder / "chain.yaml"
    path.write_text(text)
    return path


def test_cumulative_suffix_is_allowed_after_the_stamp():
    when = time_from_filename("perimeter_20240723T013731Z_cumulative.shp")
    assert when == datetime(2024, 7, 23, 1, 37, 31, tzinfo=timezone.utc)
    assert time_from_filename("perimeter_20240723T013731Z_other.shp") is None
