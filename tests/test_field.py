"""Field location, weather parsing and saved-field tests. No network."""

import pytest

import agri_core as core


def test_geolocation_success():
    result = core.parse_geolocation(
        {"coords": {"latitude": 17.385, "longitude": 78.4867, "accuracy": 35.0}, "timestamp": 1}
    )
    assert result == {"ok": True, "lat": 17.385, "lon": 78.4867, "accuracy": 35.0}


@pytest.mark.parametrize("code, phrase", [(1, "denied"), (2, "could not determine"), (3, "timed out"), (0, "does not support")])
def test_geolocation_errors_are_explained(code, phrase):
    result = core.parse_geolocation({"error": {"code": code, "message": "x"}})
    assert result["ok"] is False
    assert phrase in result["message"]


@pytest.mark.parametrize("bad", [None, "text", {"coords": {}}, {"coords": {"latitude": 200, "longitude": 0}}])
def test_geolocation_rejects_malformed(bad):
    assert core.parse_geolocation(bad)["ok"] is False


def test_polygon_area_of_one_hectare_square():
    # ~100 m x ~100 m square near Hyderabad
    lat, lon = 17.4, 78.4
    dlat = 100 / 111_195
    dlon = 100 / (111_195 * __import__("math").cos(__import__("math").radians(lat)))
    ring = [[lon, lat], [lon + dlon, lat], [lon + dlon, lat + dlat], [lon, lat + dlat]]
    assert core.polygon_area_hectares(ring) == pytest.approx(1.0, rel=0.01)


def test_polygon_area_degenerate():
    assert core.polygon_area_hectares([[0, 0], [1, 1]]) == 0.0


SAMPLE = {
    "timezone": "Asia/Kolkata",
    "timezone_abbreviation": "GMT+5:30",
    "current_units": {"temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation": "mm"},
    "current": {"time": "2026-10-17T10:15", "temperature_2m": 28.4, "relative_humidity_2m": 62,
                "precipitation": 0.0, "weather_code": 2},
    "hourly": {"time": ["2026-10-17T09:00", "2026-10-17T10:00"], "soil_moisture_3_to_9cm": [0.07, 0.08]},
}


def test_parse_weather_values_units_and_source():
    weather = core.parse_weather(SAMPLE)
    assert weather["temp"] == "28.4°C"
    assert weather["humidity"] == "62%"
    assert weather["precip"] == "0.0 mm"
    assert weather["condition"] == "Partly cloudy"
    assert weather["soil_moisture"] == 0.08
    assert weather["soil_moisture_band"] == "Dry"
    assert weather["observed"] == "2026-10-17T10:15"
    assert weather["timezone"] == "GMT+5:30"
    assert "Open-Meteo" in weather["source"]


def test_parse_weather_leaves_missing_values_unavailable():
    weather = core.parse_weather({"current": {"time": "2026-10-17T10:15"}})
    assert weather["temp"] is None and weather["humidity"] is None
    assert weather["condition"] is None and weather["soil_moisture"] is None


def test_field_record_rounds_coordinates_and_requires_name():
    record = core.field_record("  North plot ", 17.3850123, 78.4867456, "Rice", "2026-07-01",
                               [[78.48671234, 17.38501234], [78.4868, 17.385], [78.4868, 17.3851]], "Paddy")
    assert record["name"] == "North plot"
    assert record["lat"] == 17.38501 and record["lon"] == 78.48675
    assert record["boundary"][0] == [78.486712, 17.385012]
    with pytest.raises(ValueError):
        core.field_record(" ", 1, 2)


def test_load_field_records_skips_invalid_and_extra_keys():
    data = {"fields": [
        {"name": "A", "lat": 1, "lon": 2, "secret": "drop me"},
        {"name": "", "lat": 1, "lon": 2},
        {"lat": 1},
        "junk",
    ]}
    records = core.load_field_records(data)
    assert len(records) == 1
    assert set(records[0]) == set(core.FIELD_KEYS)
