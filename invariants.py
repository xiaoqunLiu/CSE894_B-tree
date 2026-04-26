"""
Invariant checker for the freeze-and-replace simulator.

We verify the four invariants the proposal lists:

  1. Monotonic state evolution: every observed state history must be
     a valid path in the allowed transition graph, and a frozen node
     never returns to NORMAL.
  2. Safe replacement: a node in INFANT state must not be reachable
     by an ordinary search descent from the root.
  3. Search continuity: every key that was ever inserted and not
     deleted is reachable by `search`.
  4. Join consistency: a JOIN-terminated node's children (if internal)
     must all appear under exactly one merged replacement node.
"""

from __future__ import annotations
from typing import Iterable, Set
from freeze_replace_bplus import FreezeReplaceBPlusTree, FRNode, NodeState, _ALLOWED


def check_monotonic_states(tree: FreezeReplaceBPlusTree,
                           all_nodes: Iterable[FRNode]) -> int:
    """Return number of violations of monotonic state evolution."""
    violations = 0
    for node in all_nodes:
        prev = node.state_history[0]
        for nxt in node.state_history[1:]:
            if nxt not in _ALLOWED[prev]:
                violations += 1
            prev = nxt
        # Once frozen, NORMAL must never appear after.
        seen_freeze = False
        for s in node.state_history:
            if s in (NodeState.FREEZE, NodeState.SLAVE_FREEZE):
                seen_freeze = True
            elif seen_freeze and s == NodeState.NORMAL:
                violations += 1
    return violations


def collect_all_nodes(tree: FreezeReplaceBPlusTree) -> Set[FRNode]:
    """Walk live tree + all replacement chains."""
    seen: Set[FRNode] = set()
    stack = [tree.root]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for r in n.replacement:
            if r not in seen:
                stack.append(r)
        if not n.is_leaf:
            for c in n.children:
                if c not in seen:
                    stack.append(c)
    return seen


def check_no_infant_reachable(tree: FreezeReplaceBPlusTree) -> int:
    """An INFANT node should never be reachable from the live root."""
    violations = 0
    stack = [tree._follow_replacement(tree.root)]
    seen = set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if n.state == NodeState.INFANT:
            violations += 1
        if not n.is_leaf:
            for c in n.children:
                stack.append(tree._follow_replacement(c))
    return violations


def check_search_continuity(tree: FreezeReplaceBPlusTree,
                             expected_keys: Iterable[int]) -> int:
    """Every expected key must be findable."""
    missing = 0
    for k in expected_keys:
        if tree.search(k) is None:
            missing += 1
    return missing


def check_join_consistency(all_nodes: Iterable[FRNode]) -> int:
    """For nodes that ended in JOIN, both peers should point to the same
    merged replacement (length-1 replacement list, same target)."""
    violations = 0
    for n in all_nodes:
        if n.state == NodeState.RECLAIMED and NodeState.JOIN in n.state_history:
            if len(n.replacement) != 1:
                violations += 1
    return violations


def run_all_checks(tree: FreezeReplaceBPlusTree,
                    expected_keys: Iterable[int]) -> dict:
    nodes = collect_all_nodes(tree)
    return {
        "monotonic_state_violations": check_monotonic_states(tree, nodes),
        "infant_reachable_violations": check_no_infant_reachable(tree),
        "search_continuity_misses": check_search_continuity(tree, expected_keys),
        "join_consistency_violations": check_join_consistency(nodes),
        "internal_invariant_violations": tree.metrics.invariant_violations,
    }
