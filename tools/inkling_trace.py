"""Summarize one warmed Inkling request from an Nsight Systems SQLite export."""
import argparse
import json
from pathlib import Path
import sqlite3


def active_ms(intervals):
    """Elapsed time with at least one kernel active, counting overlaps once."""
    total = 0
    until = 0
    for start, end in intervals:
        total += max(0, end - max(until, start))
        until = max(until, end)
    return total / 1e6


def summarize(db):
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "NVTX_EVENTS" not in tables:
        raise ValueError("No NVTX events. Capture cuda,nvtx with a marked Tandem binary.")
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        raise ValueError("No CUDA kernels. Capture cuda with --cuda-graph-trace=node.")
    db.execute("""
        CREATE TEMP VIEW steps AS
        SELECT start, end FROM NVTX_EVENTS
        WHERE text = 'Inkling decode step' AND end > start
    """)
    if not db.execute("SELECT COUNT(*) FROM steps").fetchone()[0]:
        raise ValueError("No Inkling decode steps. Capture cuda,nvtx with a marked Tandem binary.")
    db.execute("""
        CREATE TEMP VIEW decode_kernels AS
        SELECT s.value AS name, k.deviceId, MAX(r.start, k.start) AS start, MIN(r.end, k.end) AS end
        FROM steps r JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.start < r.end AND k.end > r.start
        JOIN StringIds s ON k.shortName = s.id
    """)
    phases = [dict(r) for r in db.execute("""
        SELECT text AS name, COUNT(*) AS calls, SUM(end-start)/1e6 AS ms
        FROM NVTX_EVENTS WHERE text LIKE 'Inkling %' AND end > start
        GROUP BY text ORDER BY ms DESC
    """)]
    kernels = [dict(r) for r in db.execute("""
        SELECT name, COUNT(*) AS events, SUM(end-start)/1e6 AS ms
        FROM decode_kernels GROUP BY name ORDER BY ms DESC
    """)]
    if not kernels:
        raise ValueError("No individual GPU kernels inside decode. Capture with --cuda-graph-trace=node.")
    intervals = "SELECT start,end FROM decode_kernels"
    by_device = {
        row[0]: active_ms(db.execute(intervals + " WHERE deviceId=? ORDER BY start", (row[0],)))
        for row in db.execute("SELECT DISTINCT deviceId FROM decode_kernels ORDER BY deviceId")
    }
    return {
        "phase_elapsed_ms": phases,
        "any_gpu_active_ms": active_ms(db.execute(intervals + " ORDER BY start")),
        "device_active_ms": by_device,
        "kernel_duration_ms_including_overlap": kernels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_export", type=Path)
    args = parser.parse_args()
    try:
        with sqlite3.connect(args.sqlite_export.resolve().as_uri() + "?mode=ro", uri=True) as db:
            print(json.dumps(summarize(db), indent=2))
    except (sqlite3.Error, ValueError) as exc:
        parser.exit(1, f"Cannot summarize this trace: {exc}\n")
