"""
analyze_nsight.py — Parse Nsight Systems SQLite exports and print GPU utilization metrics.

Usage:
    nsys export --type=sqlite profiling/case_N.nsys-rep
    python profiling/analyze_nsight.py profiling/case_N.sqlite
    # Or all at once:
    python profiling/analyze_nsight.py profiling/case_*.sqlite
"""

import sqlite3
import sys
from pathlib import Path


def analyze_nsight_db(db_path):
    """Open a Nsight Systems SQLite export and query GPU activity tables.

    Parameters
    ----------
    db_path : str or Path
        Path to the .sqlite file exported from an .nsys-rep file.

    Returns
    -------
    dict
        Dictionary containing kernel and memcpy statistics, or None fields
        when the relevant tables are absent.
    """
    db_path = str(db_path)
    results = {
        "db_path": db_path,
        "kernel": None,
        "memcpy_htod": None,
        "memcpy_dtoh": None,
        "top_kernels": None,
    }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # --- GPU Kernel activity ---
    try:
        cur.execute(
            """
            SELECT
                COUNT(*)                        AS kernel_count,
                SUM(end - start)                AS total_duration_ns,
                AVG(end - start)                AS avg_duration_ns,
                MIN(end - start)                AS min_duration_ns,
                MAX(end - start)                AS max_duration_ns,
                MIN(start)                      AS first_start_ns,
                MAX(end)                        AS last_end_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            """
        )
        row = cur.fetchone()
        if row and row["kernel_count"]:
            span_ns = row["last_end_ns"] - row["first_start_ns"]
            gpu_util = (
                row["total_duration_ns"] / span_ns if span_ns > 0 else 0.0
            )
            results["kernel"] = {
                "count": row["kernel_count"],
                "total_ms": row["total_duration_ns"] / 1e6,
                "avg_ms": row["avg_duration_ns"] / 1e6,
                "min_ms": row["min_duration_ns"] / 1e6,
                "max_ms": row["max_duration_ns"] / 1e6,
                "span_ms": span_ns / 1e6,
                "gpu_utilization": gpu_util,
            }

        # Top 10 kernels by cumulative time
        cur.execute(
            """
            SELECT
                demangledName                   AS name,
                COUNT(*)                        AS count,
                SUM(end - start)                AS total_duration_ns,
                AVG(end - start)                AS avg_duration_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY demangledName
            ORDER BY total_duration_ns DESC
            LIMIT 10
            """
        )
        rows = cur.fetchall()
        results["top_kernels"] = [
            {
                "name": r["name"],
                "count": r["count"],
                "total_ms": r["total_duration_ns"] / 1e6,
                "avg_ms": r["avg_duration_ns"] / 1e6,
            }
            for r in rows
        ]
    except sqlite3.OperationalError:
        # Table does not exist in this export
        pass

    # --- Memory copy activity ---
    try:
        # HtoD copies
        cur.execute(
            """
            SELECT
                COUNT(*)            AS count,
                SUM(end - start)    AS total_duration_ns,
                SUM(bytes)          AS total_bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind LIKE '%HtoD%'
            """
        )
        row = cur.fetchone()
        if row and row["count"]:
            results["memcpy_htod"] = {
                "count": row["count"],
                "total_ms": row["total_duration_ns"] / 1e6,
                "total_bytes": row["total_bytes"],
            }

        # DtoH copies
        cur.execute(
            """
            SELECT
                COUNT(*)            AS count,
                SUM(end - start)    AS total_duration_ns,
                SUM(bytes)          AS total_bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind LIKE '%DtoH%'
            """
        )
        row = cur.fetchone()
        if row and row["count"]:
            results["memcpy_dtoh"] = {
                "count": row["count"],
                "total_ms": row["total_duration_ns"] / 1e6,
                "total_bytes": row["total_bytes"],
            }
    except sqlite3.OperationalError:
        # Table does not exist in this export
        pass

    conn.close()
    return results


def _fmt_bytes(n_bytes):
    """Return a human-readable byte count string."""
    if n_bytes is None:
        return "N/A"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024.0:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.2f} PB"


def print_nsight_summary(results):
    """Print a formatted summary of Nsight Systems profiling results.

    Parameters
    ----------
    results : dict
        Output from :func:`analyze_nsight_db`.
    """
    sep = "=" * 72
    thin = "-" * 72

    print(sep)
    print(f"  Nsight Systems Profile: {results['db_path']}")
    print(sep)

    # --- Kernel summary ---
    k = results.get("kernel")
    if k:
        print("\n  GPU Kernel Activity")
        print(thin)
        print(f"  {'Kernel time (ms)':<35} {k['total_ms']:.3f}")
        print(f"  {'Kernel span (ms)':<35} {k['span_ms']:.3f}")
        print(f"  {'GPU utilization':<35} {k['gpu_utilization'] * 100:.1f}%")
        print(f"  {'Kernel launches':<35} {k['count']}")
        print(f"  {'Avg kernel duration (ms)':<35} {k['avg_ms']:.4f}")
        print(f"  {'Min kernel duration (ms)':<35} {k['min_ms']:.4f}")
        print(f"  {'Max kernel duration (ms)':<35} {k['max_ms']:.4f}")
    else:
        print("\n  GPU Kernel Activity: no data (table absent or empty)")

    # --- Memcpy summary ---
    htod = results.get("memcpy_htod")
    dtoh = results.get("memcpy_dtoh")
    if htod or dtoh:
        print("\n  Memory Copy Activity")
        print(thin)
        for label, m in (("HtoD (host -> device)", htod), ("DtoH (device -> host)", dtoh)):
            if m:
                print(f"  {label}")
                print(f"    {'Count':<33} {m['count']}")
                print(f"    {'Total time (ms)':<33} {m['total_ms']:.3f}")
                print(f"    {'Total bytes':<33} {_fmt_bytes(m['total_bytes'])}")
            else:
                print(f"  {label}: no transfers recorded")
    else:
        print("\n  Memory Copy Activity: no data (table absent or empty)")

    # --- Top kernels table ---
    top = results.get("top_kernels")
    if top:
        print("\n  Top 10 Kernels by Cumulative Time")
        print(thin)
        header = f"  {'Kernel Name':<60}  {'Count':>8}  {'Total ms':>10}  {'Avg ms':>10}"
        print(header)
        print("  " + "-" * 94)
        for entry in top:
            name = entry["name"] or "<unnamed>"
            truncated = name[:60] if len(name) <= 60 else name[:57] + "..."
            print(
                f"  {truncated:<60}  {entry['count']:>8}  "
                f"{entry['total_ms']:>10.3f}  {entry['avg_ms']:>10.4f}"
            )
    else:
        print("\n  Top 10 Kernels: no data")

    print(sep)
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python profiling/analyze_nsight.py profiling/case_N.sqlite [...]",
            file=sys.stderr,
        )
        sys.exit(1)

    paths = sys.argv[1:]
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"Warning: file not found: {path}", file=sys.stderr)
            continue
        results = analyze_nsight_db(p)
        print_nsight_summary(results)
