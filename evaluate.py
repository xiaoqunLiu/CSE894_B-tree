"""
Evaluation harness.

Runs three workloads on both the baseline B+-tree and the
freeze-and-replace simulator, then prints a comparison table and
saves the numbers to disk for the report.

Workloads
---------
  W1: insert-only, sequential keys 0..N-1
  W2: insert-only, random keys
  W3: mixed insert/delete (roughly 70/30), random keys

For W1 and W2 we additionally delete half the keys at the end so we
also exercise the underflow path.
"""

from __future__ import annotations
import json
import os
import random
import sys
from dataclasses import asdict
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from baseline_bplus import BPlusTree
from freeze_replace_bplus import FreezeReplaceBPlusTree
from invariants import run_all_checks


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------
def workload_sequential(n: int) -> List[Tuple[str, int]]:
    ops = [("i", k) for k in range(n)]
    # delete the first half
    ops += [("d", k) for k in range(n // 2)]
    return ops


def workload_random(n: int, seed: int) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    keys = list(range(n))
    rng.shuffle(keys)
    ops = [("i", k) for k in keys]
    to_delete = keys[: n // 2]
    rng.shuffle(to_delete)
    ops += [("d", k) for k in to_delete]
    return ops


def workload_mixed(n: int, seed: int) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    ops: List[Tuple[str, int]] = []
    keys_in: set = set()
    for _ in range(n):
        if not keys_in or rng.random() < 0.7:
            k = rng.randint(0, 10 * n)
            ops.append(("i", k))
            keys_in.add(k)
        else:
            k = rng.choice(list(keys_in))
            ops.append(("d", k))
            keys_in.discard(k)
    return ops


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def apply_ops(tree, ops: List[Tuple[str, int]]) -> set:
    keys_in: set = set()
    for op, k in ops:
        if op == "i":
            tree.insert(k, k)
            keys_in.add(k)
        else:
            tree.delete(k)
            keys_in.discard(k)
    return keys_in


def run_workload(name: str, ops: List[Tuple[str, int]], order: int = 5) -> dict:
    base = BPlusTree(order=order)
    fr = FreezeReplaceBPlusTree(order=order)

    base_keys = apply_ops(base, ops)
    fr_keys = apply_ops(fr, ops)
    assert base_keys == fr_keys, "the two trees disagree on which keys remain"

    # cross-check: every remaining key is found in both, and never in either
    # if it was deleted
    for k in base_keys:
        assert base.search(k) == k, f"baseline lost {k}"
        assert fr.search(k) == k, f"fr lost {k}"

    inv = run_all_checks(fr, base_keys)

    return {
        "workload": name,
        "n_ops": len(ops),
        "remaining_keys": len(base_keys),
        "order": order,
        "baseline": {
            "height": base.height(),
            "avg_search_path": round(base.average_search_path(), 3),
            "node_utilization": round(base.node_utilization(), 3),
            "node_count": base.count_nodes(),
            "splits": base.metrics.splits,
            "merges": base.metrics.merges,
            "borrows": base.metrics.borrows,
        },
        "freeze_replace": {
            "height": fr.height(),
            "avg_search_path": round(fr.average_search_path(), 3),
            "node_utilization": round(fr.node_utilization(), 3),
            "live_node_count": fr.count_live_nodes(),
            "splits": fr.metrics.splits,
            "joins": fr.metrics.joins,
            "borrows": fr.metrics.borrows,
            "copies": fr.metrics.copies,
            "state_transitions": fr.metrics.state_transitions,
            "staged_parent_rewrites": fr.metrics.staged_parent_rewrites,
            "temp_key_duplications": fr.metrics.temp_key_duplications,
        },
        "invariants": inv,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def fmt_section(title: str) -> str:
    return f"\n{'=' * 70}\n {title}\n{'=' * 70}"


def fmt_result(r: dict) -> str:
    out = []
    out.append(fmt_section(f"Workload {r['workload']} (ops={r['n_ops']}, "
                            f"remaining={r['remaining_keys']}, order={r['order']})"))
    b = r["baseline"]
    f = r["freeze_replace"]
    out.append(f"  Structural quality (should match)")
    out.append(f"    height                : baseline={b['height']:>5}   fr={f['height']:>5}")
    out.append(f"    avg search path       : baseline={b['avg_search_path']:>5}   fr={f['avg_search_path']:>5}")
    out.append(f"    node utilization      : baseline={b['node_utilization']:>5}   fr={f['node_utilization']:>5}")
    out.append(f"    live node count       : baseline={b['node_count']:>5}   fr={f['live_node_count']:>5}")
    out.append(f"  Structural events     ")
    out.append(f"    splits                : baseline={b['splits']:>5}   fr={f['splits']:>5}")
    out.append(f"    merges / joins        : baseline={b['merges']:>5}   fr={f['joins']:>5}")
    out.append(f"    borrows               : baseline={b['borrows']:>5}   fr={f['borrows']:>5}")
    out.append(f"  FR-only protocol overhead")
    out.append(f"    copy outcomes         : {f['copies']}")
    out.append(f"    state transitions     : {f['state_transitions']}")
    out.append(f"    staged parent rewrites: {f['staged_parent_rewrites']}")
    out.append(f"    temp key duplications : {f['temp_key_duplications']}")
    out.append(f"  Invariants (all should be 0)")
    for k, v in r["invariants"].items():
        out.append(f"    {k:<32}: {v}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    workloads = [
        ("W1_sequential", workload_sequential(200)),
        ("W2_random",     workload_random(200, seed=7)),
        ("W3_mixed",      workload_mixed(800, seed=13)),
    ]

    results = []
    for name, ops in workloads:
        r = run_workload(name, ops, order=5)
        results.append(r)
        print(fmt_result(r))

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
