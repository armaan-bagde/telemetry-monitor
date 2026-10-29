"""
Pydantic schemas and defensive input validation for Machine Telemetry payloads.

Compatible with both Pydantic v2 and v1 runtimes.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional

import pydantic
from pydantic import BaseModel, Field

PYDANTIC_V2 = int(pydantic.__version__.split(".")[0]) >= 2

if PYDANTIC_V2:
    from pydantic import ConfigDict, field_validator
else:
    from pydantic import validator

MACHINE_ID_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")

# Physical operating bounds for defensive telemetry validation
MIN_TEMPERATURE_C = -50.0
MAX_TEMPERATURE_C = 300.0
MIN_VIBRATION_MM_S = 0.0
MAX_VIBRATION_MM_S = 500.0
MIN_PRESSURE_PSI = 0.0
MAX_PRESSURE_PSI = 5000.0
MIN_POWER_KW = 0.0
MAX_POWER_KW = 10000.0
MAX_FUTURE_SKEW_SECONDS = 300  # Allow up to 5 minutes of clock drift


class MachineStatus(str, Enum):
    """Allowed operational states for reporting machines."""

    OPERATIONAL = "OPERATIONAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    MAINTENANCE = "MAINTENANCE"


def _validate_machine_id_value(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("machine_id must be a string.")
    cleaned = value.strip()
    if not MACHINE_ID_REGEX.match(cleaned):
        raise ValueError(
            "machine_id must be 3-64 characters, start with an alphanumeric character, "
            "and contain only letters, digits, underscores, or hyphens."
        )
    return cleaned


def _validate_timestamp_value(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("timestamp must be a valid ISO-8601 datetime.")
    if value.tzinfo is None:
        normalized = value.replace(tzinfo=timezone.utc)
    else:
        normalized = value.astimezone(timezone.utc)

    now_utc = datetime.now(timezone.utc)
    if normalized > now_utc + timedelta(seconds=MAX_FUTURE_SKEW_SECONDS):
        raise ValueError("timestamp cannot be in the future beyond allowed clock skew (5 minutes).")
    if normalized.year < 2000:
        raise ValueError("timestamp year must be 2000 or later.")
    return normalized


def _validate_finite_range(value: float, field_name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a valid number.")
    numeric_val = float(value)
    if math.isnan(numeric_val) or math.isinf(numeric_val):
        raise ValueError(f"{field_name} must be a finite number (NaN/Infinity not allowed).")
    if numeric_val < minimum or numeric_val > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return round(numeric_val, 4)


class TelemetryPayload(BaseModel):
    """Schema for incoming machine telemetry ingestion requests."""

    machine_id: str = Field(
        ...,
        description="Unique identifier of the machine (3-64 alphanumeric/underscore/hyphen chars).",
        example="CNC-MILL-001",
    )
    timestamp: datetime = Field(
        ...,
        description="ISO-8601 timestamp when the telemetry reading was captured.",
        example="2026-10-29T12:00:00Z",
    )
    temperature_c: float = Field(
        ...,
        description="Operating temperature in degrees Celsius (-50.0 to 300.0).",
        example=72.5,
    )
    vibration_mm_s: float = Field(
        ...,
        description="RMS vibration velocity in mm/s (0.0 to 500.0).",
        example=2.35,
    )
    pressure_psi: float = Field(
        ...,
        description="System hydraulic/pneumatic pressure in PSI (0.0 to 5000.0).",
        example=145.0,
    )
    power_kw: float = Field(
        ...,
        description="Instantaneous power consumption in kilowatts (0.0 to 10000.0).",
        example=18.4,
    )
    status: MachineStatus = Field(
        default=MachineStatus.OPERATIONAL,
        description="Current operational health status reported by the machine.",
        example=MachineStatus.OPERATIONAL,
    )

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

        @field_validator("machine_id", mode="before")
        @classmethod
        def validate_machine_id(cls, v: str) -> str:
            return _validate_machine_id_value(v)

        @field_validator("timestamp", mode="after")
        @classmethod
        def validate_timestamp(cls, v: datetime) -> datetime:
            return _validate_timestamp_value(v)

        @field_validator("temperature_c", mode="before")
        @classmethod
        def validate_temperature(cls, v: float) -> float:
            return _validate_finite_range(v, "temperature_c", MIN_TEMPERATURE_C, MAX_TEMPERATURE_C)

        @field_validator("vibration_mm_s", mode="before")
        @classmethod
        def validate_vibration(cls, v: float) -> float:
            return _validate_finite_range(v, "vibration_mm_s", MIN_VIBRATION_MM_S, MAX_VIBRATION_MM_S)

        @field_validator("pressure_psi", mode="before")
        @classmethod
        def validate_pressure(cls, v: float) -> float:
            return _validate_finite_range(v, "pressure_psi", MIN_PRESSURE_PSI, MAX_PRESSURE_PSI)

        @field_validator("power_kw", mode="before")
        @classmethod
        def validate_power(cls, v: float) -> float:
            return _validate_finite_range(v, "power_kw", MIN_POWER_KW, MAX_POWER_KW)

    else:

        class Config:
            extra = "forbid"
            anystr_strip_whitespace = True

        @validator("machine_id", pre=True)
        def validate_machine_id(cls, v: str) -> str:
            return _validate_machine_id_value(v)

        @validator("timestamp")
        def validate_timestamp(cls, v: datetime) -> datetime:
            return _validate_timestamp_value(v)

        @validator("temperature_c", pre=True)
        def validate_temperature(cls, v: float) -> float:
            return _validate_finite_range(v, "temperature_c", MIN_TEMPERATURE_C, MAX_TEMPERATURE_C)

        @validator("vibration_mm_s", pre=True)
        def validate_vibration(cls, v: float) -> float:
            return _validate_finite_range(v, "vibration_mm_s", MIN_VIBRATION_MM_S, MAX_VIBRATION_MM_S)

        @validator("pressure_psi", pre=True)
        def validate_pressure(cls, v: float) -> float:
            return _validate_finite_range(v, "pressure_psi", MIN_PRESSURE_PSI, MAX_PRESSURE_PSI)

        @validator("power_kw", pre=True)
        def validate_power(cls, v: float) -> float:
            return _validate_finite_range(v, "power_kw", MIN_POWER_KW, MAX_POWER_KW)


class TelemetryRecordResponse(BaseModel):
    """Schema for a stored telemetry record returned by the API."""

    id: int
    machine_id: str
    timestamp: str
    temperature_c: float
    vibration_mm_s: float
    pressure_psi: float
    power_kw: float
    status: MachineStatus
    created_at: str


class MetricStats(BaseModel):
    """Aggregated SQL statistics (AVG, MIN, MAX) for a single numeric telemetry metric."""

    avg: float
    min: float
    max: float


class MachineHealthSummary(BaseModel):
    """Real-time aggregated health summary for an individual machine."""

    machine_id: str
    sample_count: int
    first_seen: str
    last_seen: str
    temperature_c: MetricStats
    vibration_mm_s: MetricStats
    pressure_psi: MetricStats
    power_kw: MetricStats
    health_status: str = Field(
        ...,
        description="Evaluated aggregate health state (HEALTHY, ELEVATED, CRITICAL).",
    )


class AggregatedSummaryResponse(BaseModel):
    """Envelope returned by the GET /telemetry/summary aggregation endpoint."""

    total_machines: int
    total_records: int
    machines: List[MachineHealthSummary]
