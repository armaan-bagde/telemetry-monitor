"""
Database setup and connection management for the Machine Telemetry Service.

Provides context-managed SQLite connections with Write-Ahead Logging (WAL),
foreign key enforcement, and parameterized query execution.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

DEFAULT_DB_PATH = os.getenv("TELEMETRY_DB_PATH", "telemetry.db")


def get_db_path() -> str:
    """Return the configured SQLite database file path."""
    return os.getenv("TELEMETRY_DB_PATH", DEFAULT_DB_PATH)


def init_db(db_path: str | None = None) -> None:
    """
    Initialize the SQLite database schema and indexes if they do not exist.

    Args:
        db_path: Optional override path for the SQLite database file.
    """
    target_path = db_path or get_db_path()
    if target_path != ":memory:":
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                temperature_c REAL NOT NULL,
                vibration_mm_s REAL NOT NULL,
                pressure_psi REAL NOT NULL,
                power_kw REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT ( strftime('%Y-%m-%dT%H:%M:%SZ', 'now') )
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_telemetry_machine_timestamp
            ON telemetry_records (machine_id, timestamp)
            """
        )
        conn.commit()


@contextmanager
def get_db_connection(db_path: str | None = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager yielding a thread-safe SQLite connection with row factory enabled.

    Ensures transactions are rolled back on error and connections are closed cleanly.
    """
    target_path = db_path or get_db_path()
    conn = sqlite3.connect(target_path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        if target_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency that yields a managed SQLite connection per request."""
    with get_db_connection() as conn:
        yield conn
