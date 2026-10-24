import datetime as dt

import pytest
from pydantic import ValidationError

from firewatch.config import load_settings
from firewatch.evaluation.agent_eval import circle
from firewatch.evaluation.firms import match_run, read_firms_csv
from firewatch.evaluation.nrsc import compare, fire_area_km2, run_areas
from firewatch.geo.grid import WGS84, transformer
from firewatch.services.kv import SQLiteKV
from firewatch.services.repository import Repository


def test_environment_overrides(tmp_path):
    extra = tmp_path / "extra.yaml"
    extra.write_text("risk:\n  high: 0.7\n")
    s = load_settings(extra, environ={"FIREWATCH__AGENT__PLANNER": "bedrock", "FIREWATCH__ALIGNMENT__PREFERENCE": "[orb, sift]"})
    assert s.risk.high == 0.7
    assert s.agent.planner == "bedrock"
    assert s.alignment.preference == ["orb", "sift"]


def test_unknown_keys_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("risk:\n  hgih: 0.7\n")
    with pytest.raises(ValidationError):
        load_settings(bad, environ={})


def test_config_files_in_repo_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "config"
    for path in sorted(root.glob("*.yaml")):
        if path.name.startswith("routes"):
            continue
        load_settings(path, environ={})


def _repo_with_runs(crs):
    repo = Repository(SQLiteKV(":memory:"))
    to_map = transformer(WGS84, crs)
    x, y = to_map.transform(86.40, 23.75)
    repo.put_run({"run_id": "R1", "scene_id": "LC08_A", "acquired": "2020-03-01T04:50:00+00:00", "source": "landsat", "status": "done", "cloud_fraction": 0.05})
    repo.put_run({"run_id": "R2", "scene_id": "LC08_B", "acquired": "2020-03-17T04:50:00+00:00", "source": "landsat", "status": "done", "cloud_fraction": 0.6})
    repo.put_run({"run_id": "S1", "scene_id": "SYN", "acquired": "2020-03-01T04:50:00+00:00", "source": "synthetic", "status": "done", "cloud_fraction": 0.0})
    repo.put_hotspot("R1", {"hotspot_id": "fire", "decision": "close", "polygon": circle(x, y, 500.0), "centroid": (x, y)})
    repo.put_hotspot("R1", {"hotspot_id": "overlap", "decision": "log", "polygon": circle(x + 500.0, y, 500.0), "centroid": (x + 500.0, y)})
    repo.put_hotspot("R1", {"hotspot_id": "plant", "decision": "exclude", "polygon": circle(x + 5000.0, y, 300.0), "centroid": (x + 5000.0, y)})
    return repo, (x, y)


def test_nrsc_area_uses_union_and_skips_cloudy_and_industrial(settings):
    repo, _ = _repo_with_runs(settings.aoi.crs)
    rows = run_areas(repo)
    assert [r["run_id"] for r in rows] == ["R1"]
    single = fire_area_km2([{"decision": "log", "polygon": circle(0, 0, 500.0)}])
    assert single < rows[0]["area_km2"] < 2 * single
    result = {c["label"]: c for c in compare(rows)}
    assert result["2020"]["passes"] == 1 and result["2017"]["passes"] == 0
    assert result["2020"]["ratio"] == pytest.approx(rows[0]["area_km2"] / 1.89, abs=0.01)


def test_firms_matching(settings, tmp_path):
    repo, (x, y) = _repo_with_runs(settings.aoi.crs)
    to_ll = transformer(settings.aoi.crs, WGS84)
    near_lon, near_lat = to_ll.transform(x + 700.0, y)
    plant_lon, plant_lat = to_ll.transform(x + 5000.0, y)
    far_lon, far_lat = to_ll.transform(x - 9000.0, y)
    csv_path = tmp_path / "firms.csv"
    lines = ["latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight"]
    for lat, lon in ((near_lat, near_lon), (plant_lat, plant_lon), (far_lat, far_lon)):
        lines.append(f"{lat:.5f},{lon:.5f},340.1,0.4,0.4,2020-03-02,0820,N,VIIRS,n,2.0NRT,295.0,5.1,D")
    lines.append("10.0,10.0,340.1,0.4,0.4,2020-03-02,0820,N,VIIRS,n,2.0NRT,295.0,5.1,D")
    csv_path.write_text("\n".join(lines) + "\n")
    detections = read_firms_csv(csv_path, settings.aoi.bbox)
    assert len(detections) == 3 and detections[0]["date"] == dt.date(2020, 3, 2)
    result = match_run(repo.list_hotspots("R1"), detections, settings.aoi.crs)
    assert result == {"detections": 3, "near_fire": 1, "near_industrial": 1, "unmatched": 1, "fire_hotspots": 2, "fire_hotspots_with_firms": 2}
