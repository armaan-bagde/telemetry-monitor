"""
FastAPI application and routes for the Machine Telemetry Ingestion & Aggregation Service.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status

from database import get_db, init_db
from models import (
    AggregatedSummaryResponse,
    MachineHealthSummary,
    MetricStats,
    TelemetryPayload,
    TelemetryRecordResponse,
    _validate_machine_id_value,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ensure the SQLite database tables and indexes exist on application startup."""
    init_db()
    yield


app = FastAPI(
    title="Machine Telemetry Ingestion & Aggregation Service",
    description=(
        "Production REST API for ingesting, validating, and aggregating "
        "industrial machine telemetry records in SQLite."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.on_event("startup")
def _startup_fallback() -> None:
    """Fallback startup hook for older Starlette/FastAPI versions."""
    init_db()


def _evaluate_health_status(
    avg_temp: float,
    max_temp: float,
    avg_vib: float,
    max_vib: float,
) -> str:
    """Derive an overall machine health indicator from aggregated telemetry metrics."""
    if max_temp >= 120.0 or max_vib >= 25.0 or avg_temp >= 100.0 or avg_vib >= 15.0:
        return "CRITICAL"
    if max_temp >= 90.0 or max_vib >= 10.0 or avg_temp >= 80.0 or avg_vib >= 7.5:
        return "ELEVATED"
    return "HEALTHY"


def _row_to_summary(row: sqlite3.Row) -> MachineHealthSummary:
    """Convert an aggregated SQLite row into a MachineHealthSummary response model."""
    avg_temp = round(float(row["avg_temp"]), 2)
    min_temp = round(float(row["min_temp"]), 2)
    max_temp = round(float(row["max_temp"]), 2)

    avg_vib = round(float(row["avg_vibration"]), 2)
    min_vib = round(float(row["min_vibration"]), 2)
    max_vib = round(float(row["max_vibration"]), 2)

    avg_pres = round(float(row["avg_pressure"]), 2)
    min_pres = round(float(row["min_pressure"]), 2)
    max_pres = round(float(row["max_pressure"]), 2)

    avg_pow = round(float(row["avg_power"]), 2)
    min_pow = round(float(row["min_power"]), 2)
    max_pow = round(float(row["max_power"]), 2)

    return MachineHealthSummary(
        machine_id=row["machine_id"],
        sample_count=int(row["sample_count"]),
        first_seen=row["first_seen"],
        last_seen=row["last_seen"],
        temperature_c=MetricStats(avg=avg_temp, min=min_temp, max=max_temp),
        vibration_mm_s=MetricStats(avg=avg_vib, min=min_vib, max=max_vib),
        pressure_psi=MetricStats(avg=avg_pres, min=min_pres, max=max_pres),
        power_kw=MetricStats(avg=avg_pow, min=min_pow, max=max_pow),
        health_status=_evaluate_health_status(avg_temp, max_temp, avg_vib, max_vib),
    )


@app.get("/health", tags=["System"])
def health_check() -> dict:
    """Service readiness and liveness endpoint."""
    return {"status": "ok", "service": "telemetry-monitor"}


@app.post(
    "/telemetry",
    response_model=TelemetryRecordResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Telemetry"],
    summary="Ingest a validated machine telemetry record",
)
def ingest_telemetry(
    payload: TelemetryPayload,
    db: sqlite3.Connection = Depends(get_db),
) -> TelemetryRecordResponse:
    """
    Validate and store a single machine telemetry reading in SQLite using parameterized SQL.
    """
    iso_timestamp = payload.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO telemetry_records (
            machine_id,
            timestamp,
            temperature_c,
            vibration_mm_s,
            pressure_psi,
            power_kw,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.machine_id,
            iso_timestamp,
            payload.temperature_c,
            payload.vibration_mm_s,
            payload.pressure_psi,
            payload.power_kw,
            payload.status.value,
        ),
    )
    db.commit()
    record_id = cursor.lastrowid

    cursor.execute(
        """
        SELECT id, machine_id, timestamp, temperature_c, vibration_mm_s,
               pressure_psi, power_kw, status, created_at
        FROM telemetry_records
        WHERE id = ?
        """,
        (record_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve persisted telemetry record.",
        )

    return TelemetryRecordResponse(**dict(row))


@app.get(
    "/telemetry",
    response_model=List[TelemetryRecordResponse],
    tags=["Telemetry"],
    summary="List recent telemetry records with optional machine filter",
)
def list_telemetry(
    machine_id: Optional[str] = Query(
        default=None,
        description="Optional machine_id filter.",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of records to return (1-1000).",
    ),
    db: sqlite3.Connection = Depends(get_db),
) -> List[TelemetryRecordResponse]:
    """Fetch stored telemetry records ordered by most recent timestamp first."""
    cursor = db.cursor()
    if machine_id is not None:
        try:
            validated_id = _validate_machine_id_value(machine_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        cursor.execute(
            """
            SELECT id, machine_id, timestamp, temperature_c, vibration_mm_s,
                   pressure_psi, power_kw, status, created_at
            FROM telemetry_records
            WHERE machine_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (validated_id, limit),
        )
    else:
        cursor.execute(
            """
            SELECT id, machine_id, timestamp, temperature_c, vibration_mm_s,
                   pressure_psi, power_kw, status, created_at
            FROM telemetry_records
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cursor.fetchall()
    return [TelemetryRecordResponse(**dict(row)) for row in rows]


@app.get(
    "/telemetry/summary",
    response_model=AggregatedSummaryResponse,
    tags=["Aggregation"],
    summary="Compute real-time aggregated machine health summaries",
)
def get_telemetry_summary(
    machine_id: Optional[str] = Query(
        default=None,
        description="Optional machine_id to filter the aggregation.",
    ),
    start_time: Optional[str] = Query(
        default=None,
        description="Optional ISO-8601 lower bound timestamp filter.",
    ),
    end_time: Optional[str] = Query(
        default=None,
        description="Optional ISO-8601 upper bound timestamp filter.",
    ),
    db: sqlite3.Connection = Depends(get_db),
) -> AggregatedSummaryResponse:
    """
    Compute per-machine telemetry aggregations (COUNT, AVG, MIN, MAX) using
    parameterized SQL GROUP BY queries.
    """
    conditions: List[str] = []
    params: List[object] = []

    if machine_id is not None:
        try:
            validated_id = _validate_machine_id_value(machine_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        conditions.append("machine_id = ?")
        params.append(validated_id)

    if start_time is not None:
        conditions.append("timestamp >= ?")
        params.append(start_time)

    if end_time is not None:
        conditions.append("timestamp <= ?")
        params.append(end_time)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            machine_id,
            COUNT(*) AS sample_count,
            MIN(timestamp) AS first_seen,
            MAX(timestamp) AS last_seen,
            AVG(temperature_c) AS avg_temp,
            MIN(temperature_c) AS min_temp,
            MAX(temperature_c) AS max_temp,
            AVG(vibration_mm_s) AS avg_vibration,
            MIN(vibration_mm_s) AS min_vibration,
            MAX(vibration_mm_s) AS max_vibration,
            AVG(pressure_psi) AS avg_pressure,
            MIN(pressure_psi) AS min_pressure,
            MAX(pressure_psi) AS max_pressure,
            AVG(power_kw) AS avg_power,
            MIN(power_kw) AS min_power,
            MAX(power_kw) AS max_power
        FROM telemetry_records
        {where_clause}
        GROUP BY machine_id
        ORDER BY machine_id ASC
    """

    cursor = db.cursor()
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()

    summaries = [_row_to_summary(row) for row in rows]
    total_records = sum(item.sample_count for item in summaries)

    return AggregatedSummaryResponse(
        total_machines=len(summaries),
        total_records=total_records,
        machines=summaries,
    )


@app.get(
    "/telemetry/summary/{machine_id}",
    response_model=MachineHealthSummary,
    tags=["Aggregation"],
    summary="Compute real-time health summary for a single machine",
)
def get_single_machine_summary(
    machine_id: str,
    db: sqlite3.Connection = Depends(get_db),
) -> MachineHealthSummary:
    """
    Compute real-time SQL aggregations (COUNT, AVG, MIN, MAX) for a specific machine_id.
    Returns 404 if no telemetry records exist for the requested machine.
    """
    try:
        validated_id = _validate_machine_id_value(machine_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    cursor = db.cursor()
    cursor.execute(
        """
        SELECT
            machine_id,
            COUNT(*) AS sample_count,
            MIN(timestamp) AS first_seen,
            MAX(timestamp) AS last_seen,
            AVG(temperature_c) AS avg_temp,
            MIN(temperature_c) AS min_temp,
            MAX(temperature_c) AS max_temp,
            AVG(vibration_mm_s) AS avg_vibration,
            MIN(vibration_mm_s) AS min_vibration,
            MAX(vibration_mm_s) AS max_vibration,
            AVG(pressure_psi) AS avg_pressure,
            MIN(pressure_psi) AS min_pressure,
            MAX(pressure_psi) AS max_pressure,
            AVG(power_kw) AS avg_power,
            MIN(power_kw) AS min_power,
            MAX(power_kw) AS max_power
        FROM telemetry_records
        WHERE machine_id = ?
        GROUP BY machine_id
        """,
        (validated_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No telemetry records found for machine '{validated_id}'.",
        )

    return _row_to_summary(row)
