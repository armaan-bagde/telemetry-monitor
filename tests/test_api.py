"""
Automated PyTest test suite for the Machine Telemetry Ingestion & Aggregation Service.

Verifies HTTP status codes, defensive input validation, SQLite persistence,
and parameterized SQL aggregations (GROUP BY, AVG, MIN, MAX).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from database import init_db
from main import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide a TestClient backed by an isolated temporary SQLite database."""
    db_file = tmp_path / "test_telemetry.db"
    monkeypatch.setenv("TELEMETRY_DB_PATH", str(db_file))
    init_db(str(db_file))
    with TestClient(app) as test_client:
        yield test_client


def _sample_payload(**overrides: object) -> dict:
    base = {
        "machine_id": "CNC-MILL-001",
        "timestamp": "2026-10-29T10:15:00Z",
        "temperature_c": 68.5,
        "vibration_mm_s": 2.1,
        "pressure_psi": 120.0,
        "power_kw": 15.2,
        "status": "OPERATIONAL",
    }
    base.update(overrides)
    return base


def test_health_check(client: TestClient) -> None:
    """Verify the service health check endpoint returns HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "telemetry-monitor"}


def test_ingest_valid_telemetry_record(client: TestClient) -> None:
    """Verify valid telemetry payloads are persisted and return HTTP 201 Created."""
    payload = _sample_payload()
    response = client.post("/telemetry", json=payload)
    assert response.status_code == 201

    body = response.json()
    assert body["id"] == 1
    assert body["machine_id"] == "CNC-MILL-001"
    assert body["timestamp"] == "2026-10-29T10:15:00Z"
    assert body["temperature_c"] == 68.5
    assert body["vibration_mm_s"] == 2.1
    assert body["pressure_psi"] == 120.0
    assert body["power_kw"] == 15.2
    assert body["status"] == "OPERATIONAL"
    assert "created_at" in body


def test_reject_invalid_machine_id(client: TestClient) -> None:
    """Verify malformed or SQL-injection-like machine IDs are rejected with HTTP 422."""
    for bad_id in ["", "A", "DROP TABLE telemetry_records;--", "bad id with spaces", "@invalid"]:
        response = client.post("/telemetry", json=_sample_payload(machine_id=bad_id))
        assert response.status_code == 422, f"Expected 422 for machine_id={bad_id!r}"


def test_reject_future_and_malformed_timestamps(client: TestClient) -> None:
    """Verify future timestamps and invalid datetime strings are rejected with HTTP 422."""
    future_ts = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    res_future = client.post("/telemetry", json=_sample_payload(timestamp=future_ts))
    assert res_future.status_code == 422

    res_ancient = client.post("/telemetry", json=_sample_payload(timestamp="1995-01-01T00:00:00Z"))
    assert res_ancient.status_code == 422

    res_malformed = client.post("/telemetry", json=_sample_payload(timestamp="not-a-timestamp"))
    assert res_malformed.status_code == 422


def test_reject_out_of_range_numeric_metrics(client: TestClient) -> None:
    """Verify out-of-range numeric readings and booleans are rejected with HTTP 422."""
    invalid_cases = [
        {"temperature_c": -100.0},
        {"temperature_c": 450.0},
        {"vibration_mm_s": -0.5},
        {"vibration_mm_s": 999.0},
        {"pressure_psi": -10.0},
        {"pressure_psi": 8000.0},
        {"power_kw": -1.0},
        {"power_kw": 25000.0},
        {"temperature_c": True},
    ]
    for override in invalid_cases:
        response = client.post("/telemetry", json=_sample_payload(**override))
        assert response.status_code == 422, f"Expected 422 for override={override}"


def test_reject_unknown_extra_fields(client: TestClient) -> None:
    """Verify unexpected fields in payload are rejected due to strict schema configuration."""
    payload = _sample_payload(unexpected_field="malicious_value")
    response = client.post("/telemetry", json=payload)
    assert response.status_code == 422


def test_aggregated_machine_health_summary(client: TestClient) -> None:
    """Verify GROUP BY, AVG, MIN, MAX calculations across multiple machines and records."""
    records = [
        _sample_payload(
            machine_id="TURBINE-A1",
            timestamp="2026-10-29T08:00:00Z",
            temperature_c=70.0,
            vibration_mm_s=2.0,
            pressure_psi=100.0,
            power_kw=10.0,
        ),
        _sample_payload(
            machine_id="TURBINE-A1",
            timestamp="2026-10-29T09:00:00Z",
            temperature_c=80.0,
            vibration_mm_s=4.0,
            pressure_psi=150.0,
            power_kw=20.0,
        ),
        _sample_payload(
            machine_id="TURBINE-A1",
            timestamp="2026-10-29T10:00:00Z",
            temperature_c=90.0,
            vibration_mm_s=6.0,
            pressure_psi=200.0,
            power_kw=30.0,
        ),
        _sample_payload(
            machine_id="PRESS-B2",
            timestamp="2026-10-29T09:30:00Z",
            temperature_c=125.0,
            vibration_mm_s=12.0,
            pressure_psi=300.0,
            power_kw=45.0,
            status="CRITICAL",
        ),
    ]
    for record in records:
        res = client.post("/telemetry", json=record)
        assert res.status_code == 201

    summary_res = client.get("/telemetry/summary")
    assert summary_res.status_code == 200
    data = summary_res.json()

    assert data["total_machines"] == 2
    assert data["total_records"] == 4

    by_machine = {m["machine_id"]: m for m in data["machines"]}
    turbine = by_machine["TURBINE-A1"]
    assert turbine["sample_count"] == 3
    assert turbine["first_seen"] == "2026-10-29T08:00:00Z"
    assert turbine["last_seen"] == "2026-10-29T10:00:00Z"
    assert turbine["temperature_c"] == {"avg": 80.0, "min": 70.0, "max": 90.0}
    assert turbine["vibration_mm_s"] == {"avg": 4.0, "min": 2.0, "max": 6.0}
    assert turbine["pressure_psi"] == {"avg": 150.0, "min": 100.0, "max": 200.0}
    assert turbine["power_kw"] == {"avg": 20.0, "min": 10.0, "max": 30.0}
    assert turbine["health_status"] == "ELEVATED"

    press = by_machine["PRESS-B2"]
    assert press["sample_count"] == 1
    assert press["temperature_c"] == {"avg": 125.0, "min": 125.0, "max": 125.0}
    assert press["health_status"] == "CRITICAL"


def test_single_machine_summary_and_404(client: TestClient) -> None:
    """Verify single-machine summary endpoint returns 200 when found and 404 when absent."""
    client.post("/telemetry", json=_sample_payload(machine_id="ROBOT-ARM-07", temperature_c=55.0))

    res_ok = client.get("/telemetry/summary/ROBOT-ARM-07")
    assert res_ok.status_code == 200
    body = res_ok.json()
    assert body["machine_id"] == "ROBOT-ARM-07"
    assert body["sample_count"] == 1
    assert body["health_status"] == "HEALTHY"

    res_missing = client.get("/telemetry/summary/NON-EXISTENT-99")
    assert res_missing.status_code == 404


def test_list_telemetry_with_filter_and_limit(client: TestClient) -> None:
    """Verify listing telemetry records supports machine_id filtering and row limits."""
    client.post("/telemetry", json=_sample_payload(machine_id="M-100", timestamp="2026-10-29T08:00:00Z"))
    client.post("/telemetry", json=_sample_payload(machine_id="M-100", timestamp="2026-10-29T09:00:00Z"))
    client.post("/telemetry", json=_sample_payload(machine_id="M-200", timestamp="2026-10-29T10:00:00Z"))

    res_filtered = client.get("/telemetry", params={"machine_id": "M-100"})
    assert res_filtered.status_code == 200
    items = res_filtered.json()
    assert len(items) == 2
    assert items[0]["timestamp"] == "2026-10-29T09:00:00Z"

    res_limited = client.get("/telemetry", params={"limit": 1})
    assert res_limited.status_code == 200
    assert len(res_limited.json()) == 1
