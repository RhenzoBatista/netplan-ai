"""SQLite reference database of NR bands and numerology, plus a read-only query layer.

Values are reference data summarized from 3GPP TS 38.101-1/-2 and TS 38.211.
Brazilian spectrum notes are informative. Always confirm against the current
3GPP release and Anatel regulation before a real deployment.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE nr_bands (
    band TEXT PRIMARY KEY,
    freq_range TEXT NOT NULL,          -- FR1 or FR2
    duplex TEXT NOT NULL,              -- TDD, FDD, SDL
    ul_low_mhz REAL, ul_high_mhz REAL,
    dl_low_mhz REAL NOT NULL, dl_high_mhz REAL NOT NULL,
    scs_allowed_khz TEXT NOT NULL,     -- comma separated
    scs_typical_khz INTEGER NOT NULL
);
CREATE TABLE numerology (
    mu INTEGER PRIMARY KEY,
    scs_khz INTEGER NOT NULL,
    slot_ms REAL NOT NULL,
    slots_per_subframe INTEGER NOT NULL,
    freq_range TEXT NOT NULL,
    max_carrier_bw_mhz INTEGER
);
CREATE TABLE brazil_spectrum (
    band TEXT NOT NULL,
    range_mhz TEXT NOT NULL,
    usage TEXT NOT NULL
);
"""

NR_BANDS = [
    ("n1", "FR1", "FDD", 1920, 1980, 2110, 2170, "15,30,60", 15),
    ("n3", "FR1", "FDD", 1710, 1785, 1805, 1880, "15,30,60", 15),
    ("n7", "FR1", "FDD", 2500, 2570, 2620, 2690, "15,30,60", 15),
    ("n28", "FR1", "FDD", 703, 748, 758, 803, "15,30", 15),
    ("n40", "FR1", "TDD", 2300, 2400, 2300, 2400, "15,30,60", 30),
    ("n41", "FR1", "TDD", 2496, 2690, 2496, 2690, "15,30,60", 30),
    ("n77", "FR1", "TDD", 3300, 4200, 3300, 4200, "15,30,60", 30),
    ("n78", "FR1", "TDD", 3300, 3800, 3300, 3800, "15,30,60", 30),
    ("n79", "FR1", "TDD", 4400, 5000, 4400, 5000, "30,60", 30),
    ("n257", "FR2", "TDD", 26500, 29500, 26500, 29500, "60,120", 120),
    ("n258", "FR2", "TDD", 24250, 27500, 24250, 27500, "60,120", 120),
    ("n260", "FR2", "TDD", 37000, 40000, 37000, 40000, "60,120", 120),
]

NUMEROLOGY = [
    (0, 15, 1.0, 1, "FR1", 50),
    (1, 30, 0.5, 2, "FR1", 100),
    (2, 60, 0.25, 4, "FR1/FR2", 200),
    (3, 120, 0.125, 8, "FR2", 400),
    (4, 240, 0.0625, 16, "FR2 (SSB only)", None),
]

BRAZIL = [
    ("n28", "703-748 / 758-803", "700 MHz, nationwide coverage layer"),
    ("n78", "3300-3700", "Main 5G band, 2021 auction"),
    ("n78", "3700-3800", "Private networks (SLP), confirm current Anatel rules"),
    ("n40", "2300-2400", "Regional and capacity layer"),
    ("n258", "24300-27500", "26 GHz mmWave, 2021 auction"),
]


def build_db(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO nr_bands VALUES (?,?,?,?,?,?,?,?,?)", NR_BANDS)
        con.executemany("INSERT INTO numerology VALUES (?,?,?,?,?,?)", NUMEROLOGY)
        con.executemany("INSERT INTO brazil_spectrum VALUES (?,?,?)", BRAZIL)
    return path


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|truncate)\b",
    re.IGNORECASE,
)


class ReadOnlyTelecomDB:
    """SQL access for agents. Read-only connection plus a SELECT-only guard."""

    def __init__(self, path: str | Path, max_rows: int = 50):
        self.path = Path(path)
        self.max_rows = max_rows
        if not self.path.exists():
            build_db(self.path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def schema(self) -> str:
        with self._connect() as con:
            rows = con.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
        return "\n".join(r[0] for r in rows)

    def query(self, sql: str) -> list[dict]:
        stmt = sql.strip().rstrip(";")
        if ";" in stmt:
            raise ValueError("Only a single statement is allowed")
        if not re.match(r"^\s*(select|with)\b", stmt, re.IGNORECASE) or _FORBIDDEN.search(stmt):
            raise ValueError("Only read-only SELECT queries are allowed")
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(stmt).fetchmany(self.max_rows)
        return [dict(r) for r in rows]

    def band_for_frequency(self, freq_mhz: float) -> list[dict]:
        """Bands whose DL or UL range contains the frequency, most specific first."""
        freq_mhz = float(freq_mhz)
        return self.query(
            "SELECT * FROM nr_bands WHERE "
            f"({freq_mhz} BETWEEN dl_low_mhz AND dl_high_mhz) OR "
            f"({freq_mhz} BETWEEN ul_low_mhz AND ul_high_mhz) "
            "ORDER BY (dl_high_mhz - dl_low_mhz) ASC"
        )
