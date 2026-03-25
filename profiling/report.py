"""Combined profiling report for DESC GPU profiling cases.

Usage:
    python profiling/report.py
"""

import glob
import json


def load_results(pattern="profiling/results_case_*.json"):
    """Glob for JSON result files, load and return a sorted list of result dicts."""
    files = sorted(glob.glob(pattern))
    results = []
    for path in files:
        with open(path) as f:
            results.append(json.load(f))
    return results


def print_report(results):
    """Print a combined profiling report for all result dicts."""
    if not results:
        print("No results found.")
        return

    for res in results:
        case = res.get("case", "unknown")
        wall = res.get("wall_time_s", float("nan"))
        dofs = res.get("dofs", "?")
        dim_f = res.get("dim_f", "?")
        call_log = res.get("call_log", [])
        obj_breakdown = res.get("objective_breakdown", [])

        # Compute step breakdown from call_log
        fun_total = sum(t for kind, t in call_log if kind == "fun")
        jac_total = sum(t for kind, t in call_log if kind == "jac")
        other = wall - fun_total - jac_total

        def pct(val):
            if wall and wall > 0:
                return 100.0 * val / wall
            return float("nan")

        print("=" * 70)
        print(f"Case:        {case}")
        print(f"DOFs:        {dofs}")
        print(f"dim_f:       {dim_f}  (residuals)")
        print(f"Wall time:   {wall:.2f} s")
        print()
        print("Step breakdown:")
        print(f"  Jacobian:  {jac_total:.2f} s  ({pct(jac_total):.1f}%)")
        print(f"  Obj eval:  {fun_total:.2f} s  ({pct(fun_total):.1f}%)")
        print(f"  Other:     {other:.2f} s  ({pct(other):.1f}%)")
        print()

        if obj_breakdown:
            # Sort by median_ms descending
            sorted_objs = sorted(
                obj_breakdown, key=lambda x: x.get("median_ms", 0), reverse=True
            )
            jac_ms_total = sum(o.get("median_ms", 0) for o in sorted_objs)

            print("Per-objective Jacobian breakdown (sorted by cost):")
            header = f"  {'Objective':<35} {'median_ms':>10} {'%':>7} {'[dim_f x dim_x]':>16}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for obj in sorted_objs:
                name = obj.get("name", "unknown")
                med = obj.get("median_ms", 0.0)
                obj_dim_f = obj.get("dim_f", "?")
                obj_dim_x = obj.get("dim_x", "?")
                share = (100.0 * med / jac_ms_total) if jac_ms_total > 0 else float("nan")
                shape_str = f"[{obj_dim_f} x {obj_dim_x}]"
                print(f"  {name:<35} {med:>10.1f} {share:>6.1f}% {shape_str:>16}")
        else:
            print("  (no per-objective breakdown available)")

        print()

    print("=" * 70)
    print(f"Total cases: {len(results)}")


if __name__ == "__main__":
    results = load_results()
    print_report(results)
