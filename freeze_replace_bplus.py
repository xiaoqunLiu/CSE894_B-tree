"""
Freeze-and-replace B+-tree simulator.

This module models the algorithmic core of Braginsky and Petrank's
lock-free B+-tree (SPAA 2012). It does NOT implement real lock-free
concurrency (no threads, no CAS). Instead, it models the protocol:

  * a node never changes its keys after it leaves the "normal" state;
  * once a node needs structural repair, it is FROZEN;
  * one or more INFANT replacement nodes are created;
  * parent links are rewritten in stages;
  * once installation is complete, the replacement nodes are NORMAL
    and the frozen old nodes are RECLAIMED.

The point of the simulator is to count protocol-level events
(state transitions, staged parent rewrites, key duplications, copy
outcomes) so we can compare against the in-place baseline.

State machine
-------------
The lifecycle of a node is roughly:

    INFANT ---> NORMAL ---> FREEZE ---> {COPY | SPLIT | JOIN | SLAVE_FREEZE}
                                            |
                                            v
                                         RECLAIMED

REQUEST_SLAVE is a transient marker on a sibling chosen as a join
partner; it then becomes SLAVE_FREEZE once the partner agrees and
is frozen too.

State transitions are monotonic: a node never moves backward, and
once it is frozen its `keys` field is never written again.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# Node states
# ---------------------------------------------------------------------------
class NodeState(Enum):
    INFANT = "infant"            # newly allocated, not yet visible to searches
    NORMAL = "normal"            # part of the official tree, mutable
    FREEZE = "freeze"            # marked for replacement, no more writes
    REQUEST_SLAVE = "req_slave"  # asked to become a join partner
    SLAVE_FREEZE = "slave_freeze"# accepted as join partner, also frozen
    COPY = "copy"                # frozen node terminated as a copy outcome
    SPLIT = "split"              # frozen node terminated as a split outcome
    JOIN = "join"                # frozen node terminated as a join outcome
    RECLAIMED = "reclaimed"      # logically removed from the tree


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class FRMetrics:
    inserts: int = 0
    deletes: int = 0
    searches: int = 0

    splits: int = 0           # split outcomes (overflow)
    joins: int = 0            # join outcomes (underflow + merge)
    borrows: int = 0          # borrow / redistribute outcomes
    copies: int = 0           # frozen but ended up as a copy

    state_transitions: int = 0
    staged_parent_rewrites: int = 0
    temp_key_duplications: int = 0   # keys present in both old and replacement during installation
    invariant_violations: int = 0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class FRNode:
    _next_id = 0

    def __init__(self, is_leaf: bool, state: NodeState = NodeState.INFANT):
        FRNode._next_id += 1
        self.id: int = FRNode._next_id
        self.is_leaf: bool = is_leaf
        self.state: NodeState = state
        self.keys: List[int] = []
        self.values: List[object] = []
        self.children: List["FRNode"] = []
        self.next: Optional["FRNode"] = None
        self.parent: Optional["FRNode"] = None
        # Replacement pointer: when this node is frozen, this is what
        # searches should follow. May point to a single replacement
        # (copy/borrow) or a list of replacements (split).
        self.replacement: List["FRNode"] = []
        # History of states this node has been in, used for invariant checks.
        self.state_history: List[NodeState] = [state]

    def __repr__(self) -> str:  # pragma: no cover
        kind = "L" if self.is_leaf else "I"
        return f"<{kind}{self.id} {self.state.value} keys={self.keys}>"


# ---------------------------------------------------------------------------
# Allowed monotonic transitions
# ---------------------------------------------------------------------------
_ALLOWED: dict = {
    NodeState.INFANT: {NodeState.NORMAL},
    NodeState.NORMAL: {NodeState.FREEZE, NodeState.REQUEST_SLAVE},
    NodeState.REQUEST_SLAVE: {NodeState.SLAVE_FREEZE},
    NodeState.SLAVE_FREEZE: {NodeState.JOIN},
    NodeState.FREEZE: {NodeState.COPY, NodeState.SPLIT, NodeState.JOIN},
    NodeState.COPY: {NodeState.RECLAIMED},
    NodeState.SPLIT: {NodeState.RECLAIMED},
    NodeState.JOIN: {NodeState.RECLAIMED},
    NodeState.RECLAIMED: set(),
}


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------
class FreezeReplaceBPlusTree:
    """
    B+-tree that uses freeze-and-replace for all structural repair.

    Search is the same as in a normal B+-tree, except that if it
    encounters a frozen node it follows the replacement pointers.
    """

    def __init__(self, order: int = 4):
        if order < 3:
            raise ValueError("order must be >= 3")
        self.order = order
        self.min_keys = math.ceil(order / 2) - 1
        self.metrics = FRMetrics()
        root = FRNode(is_leaf=True)
        self.root: FRNode = root
        self._activate(root)

    # -- state management ---------------------------------------------
    def _transition(self, node: FRNode, new_state: NodeState) -> None:
        if new_state not in _ALLOWED[node.state]:
            self.metrics.invariant_violations += 1
            raise RuntimeError(
                f"illegal transition {node.state} -> {new_state} on {node!r}"
            )
        node.state = new_state
        node.state_history.append(new_state)
        self.metrics.state_transitions += 1

    def _activate(self, node: FRNode) -> None:
        """INFANT -> NORMAL: make a freshly built node official."""
        self._transition(node, NodeState.NORMAL)

    # -- search --------------------------------------------------------
    def search(self, key: int) -> Optional[object]:
        self.metrics.searches += 1
        node = self._follow_replacement(self.root)
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            child = node.children[i]
            child = self._follow_replacement(child)
            node = child
        for k, v in zip(node.keys, node.values):
            if k == key:
                return v
        return None

    def _follow_replacement(self, node: FRNode) -> FRNode:
        """If `node` is frozen and has been replaced, hop to a replacement
        that contains the (logical) key range we want. For simplicity in
        this simulator, we just follow the first replacement; for splits
        the caller will re-route at the parent."""
        # In the real algorithm a search re-reads from the root if it
        # finds a frozen node. We simulate this by always being able to
        # reach a NORMAL successor through the replacement chain.
        guard = 0
        while node.state in (NodeState.FREEZE, NodeState.SLAVE_FREEZE,
                              NodeState.REQUEST_SLAVE,
                              NodeState.COPY, NodeState.SPLIT, NodeState.JOIN,
                              NodeState.RECLAIMED) and node.replacement:
            node = node.replacement[0]
            guard += 1
            if guard > 1000:
                raise RuntimeError("replacement chain too long")
        return node

    def _find_leaf_for_update(self, key: int) -> FRNode:
        """Like search, but we need a leaf in NORMAL state to update."""
        # walk from the root, descending to the appropriate leaf
        node = self.root
        node = self._follow_replacement(node)
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            if i >= len(node.children):
                i = len(node.children) - 1
            child = node.children[i]
            child = self._follow_replacement(child)
            node = child
        return node

    # -- insert --------------------------------------------------------
    def insert(self, key: int, value: object = None) -> None:
        self.metrics.inserts += 1
        if value is None:
            value = key
        leaf = self._find_leaf_for_update(key)
        # update if exists
        for i, k in enumerate(leaf.keys):
            if k == key:
                leaf.values[i] = value
                return
        # Sequential model: NORMAL leaves can take direct writes, exactly
        # like in the paper's "no rebalancing needed" fast path.
        i = 0
        while i < len(leaf.keys) and leaf.keys[i] < key:
            i += 1
        leaf.keys.insert(i, key)
        leaf.values.insert(i, value)
        if len(leaf.keys) >= self.order:
            self._freeze_and_split(leaf)

    def _freeze_and_split(self, leaf: FRNode) -> None:
        """Overflow: freeze the leaf, build two replacement leaves,
        install them at the parent."""
        self.metrics.splits += 1
        self._transition(leaf, NodeState.FREEZE)

        mid = len(leaf.keys) // 2
        left = FRNode(is_leaf=True)
        left.keys = leaf.keys[:mid]
        left.values = leaf.values[:mid]
        right = FRNode(is_leaf=True)
        right.keys = leaf.keys[mid:]
        right.values = leaf.values[mid:]

        # Temporary key duplication: during installation, the keys exist
        # both in the frozen leaf and in the two infants.
        self.metrics.temp_key_duplications += len(leaf.keys)

        # leaf-sibling links
        left.next = right
        right.next = leaf.next

        leaf.replacement = [left, right]

        # Stage 1: install at the parent (or create a new root).
        sep = right.keys[0]
        self._install_split_at_parent(leaf, sep, left, right)

        # Stage 2: activate replacements (INFANT -> NORMAL).
        self._activate(left)
        self._activate(right)

        # Stage 3: terminate frozen leaf as SPLIT, then RECLAIM.
        self._transition(leaf, NodeState.SPLIT)
        self._transition(leaf, NodeState.RECLAIMED)

    def _install_split_at_parent(self, old: FRNode, sep: int,
                                  left: FRNode, right: FRNode) -> None:
        if old is self.root:
            new_root = FRNode(is_leaf=False)
            new_root.keys = [sep]
            new_root.children = [left, right]
            left.parent = new_root
            right.parent = new_root
            self._activate(new_root)
            self.root = new_root
            self.metrics.staged_parent_rewrites += 1
            return

        parent = old.parent
        # In the paper, parent installation is itself a CAS; if the parent
        # is frozen we'd recurse upward. Here we model the staged rewrite
        # by counting it and, if the parent itself overflows, recursing.
        idx = parent.children.index(old)
        parent.keys.insert(idx, sep)
        parent.children.insert(idx, left)       # left replaces old at idx
        parent.children[idx + 1] = right        # right takes old's old slot
        left.parent = parent
        right.parent = parent
        self.metrics.staged_parent_rewrites += 1

        if len(parent.children) > self.order:
            self._freeze_and_split_internal(parent)

    def _freeze_and_split_internal(self, node: FRNode) -> None:
        self.metrics.splits += 1
        self._transition(node, NodeState.FREEZE)
        mid = len(node.keys) // 2
        sep = node.keys[mid]
        left = FRNode(is_leaf=False)
        left.keys = node.keys[:mid]
        left.children = node.children[:mid + 1]
        for c in left.children:
            c.parent = left
        right = FRNode(is_leaf=False)
        right.keys = node.keys[mid + 1:]
        right.children = node.children[mid + 1:]
        for c in right.children:
            c.parent = right
        self.metrics.temp_key_duplications += len(node.keys)
        node.replacement = [left, right]

        self._install_split_at_parent(node, sep, left, right)
        self._activate(left)
        self._activate(right)
        self._transition(node, NodeState.SPLIT)
        self._transition(node, NodeState.RECLAIMED)

    # -- delete --------------------------------------------------------
    def delete(self, key: int) -> bool:
        self.metrics.deletes += 1
        leaf = self._find_leaf_for_update(key)
        if key not in leaf.keys:
            return False
        i = leaf.keys.index(key)
        leaf.keys.pop(i)
        leaf.values.pop(i)
        if leaf is self.root:
            return True
        if len(leaf.keys) < self.min_keys:
            self._freeze_and_repair(leaf)
        else:
            self._maybe_refresh_separator(leaf)
        return True

    def _maybe_refresh_separator(self, node: FRNode) -> None:
        if not node.keys or node is self.root:
            return
        cur = node
        while cur.parent is not None:
            parent = cur.parent
            if cur not in parent.children:
                return
            idx = parent.children.index(cur)
            if idx > 0:
                first = self._first_key(cur)
                if first is not None:
                    parent.keys[idx - 1] = first
                return
            cur = parent

    def _first_key(self, node: FRNode) -> Optional[int]:
        node = self._follow_replacement(node)
        while not node.is_leaf:
            if not node.children:
                return None
            node = self._follow_replacement(node.children[0])
        return node.keys[0] if node.keys else None

    def _freeze_and_repair(self, node: FRNode) -> None:
        """
        Underflow path. Decide between borrow (-> COPY outcome on this
        node, since we're just moving keys around) and join (-> JOIN
        outcome with a sibling).
        """
        parent = node.parent
        if parent is None:
            return
        idx = parent.children.index(node)
        left = parent.children[idx - 1] if idx > 0 else None
        right = parent.children[idx + 1] if idx + 1 < len(parent.children) else None

        if left is not None and self._can_lend(left):
            self._freeze_and_borrow(node, left, parent, idx, from_left=True)
            return
        if right is not None and self._can_lend(right):
            self._freeze_and_borrow(node, right, parent, idx, from_left=False)
            return

        if left is not None:
            self._freeze_and_join(node, left, parent, idx, with_left=True)
        else:
            assert right is not None
            self._freeze_and_join(node, right, parent, idx, with_left=False)

        # Parent may now underflow.
        if parent is self.root:
            if not parent.keys and parent.children:
                # collapse root
                self.root = self._follow_replacement(parent.children[0])
                self.root.parent = None
            return
        if not self._is_internal_ok(parent):
            self._freeze_and_repair(parent)

    def _is_internal_ok(self, node: FRNode) -> bool:
        return len(node.children) - 1 >= self.min_keys or node is self.root

    def _can_lend(self, sibling: FRNode) -> bool:
        if sibling.is_leaf:
            return len(sibling.keys) > self.min_keys
        return len(sibling.children) - 1 > self.min_keys

    # ----- borrow (a.k.a. copy outcome) ------------------------------
    def _freeze_and_borrow(self, node: FRNode, sibling: FRNode,
                            parent: FRNode, idx: int, from_left: bool) -> None:
        """
        Borrow is implemented by freezing both nodes and producing
        a fresh pair of replacement nodes with redistributed keys.
        Each frozen node terminates in COPY (we're moving keys, not
        changing the count of children at the parent).
        """
        self.metrics.borrows += 1
        self._transition(node, NodeState.FREEZE)
        # We treat the sibling as a slave for the borrow operation.
        self._transition(sibling, NodeState.REQUEST_SLAVE)
        self._transition(sibling, NodeState.SLAVE_FREEZE)

        if node.is_leaf:
            new_node, new_sibling = self._redistribute_leaves(node, sibling, from_left)
        else:
            new_node, new_sibling = self._redistribute_internals(
                node, sibling, parent, idx, from_left
            )
        self.metrics.temp_key_duplications += len(node.keys) + len(sibling.keys)

        node.replacement = [new_node]
        sibling.replacement = [new_sibling]

        # Install replacements at the parent, then update separator.
        self._install_borrow_at_parent(parent, idx, node, sibling,
                                        new_node, new_sibling, from_left)

        self._activate(new_node)
        self._activate(new_sibling)

        # Terminate frozen nodes as COPY (no structural count change).
        self._transition(node, NodeState.COPY)
        self._transition(node, NodeState.RECLAIMED)
        # The slave terminates the same way -- we collapse SLAVE_FREEZE
        # straight to JOIN-like termination but model it as COPY since
        # nothing merged. The state machine permits SLAVE_FREEZE -> JOIN
        # only, so we go through JOIN and then RECLAIMED. (This is a
        # minor simplification of the paper's full state set.)
        self._transition(sibling, NodeState.JOIN)
        self._transition(sibling, NodeState.RECLAIMED)
        self.metrics.copies += 1  # the underflowing node "copied" its data

    def _redistribute_leaves(self, node: FRNode, sibling: FRNode,
                              from_left: bool) -> Tuple[FRNode, FRNode]:
        new_node = FRNode(is_leaf=True)
        new_sibling = FRNode(is_leaf=True)
        if from_left:
            combined_keys = sibling.keys[:-1] + [sibling.keys[-1]] + node.keys
            combined_vals = sibling.values[:-1] + [sibling.values[-1]] + node.values
            # move one from sibling to node
            new_sibling.keys = sibling.keys[:-1]
            new_sibling.values = sibling.values[:-1]
            new_node.keys = [sibling.keys[-1]] + node.keys
            new_node.values = [sibling.values[-1]] + node.values
        else:
            new_node.keys = node.keys + [sibling.keys[0]]
            new_node.values = node.values + [sibling.values[0]]
            new_sibling.keys = sibling.keys[1:]
            new_sibling.values = sibling.values[1:]
        # leaf links
        if from_left:
            new_sibling.next = new_node
            new_node.next = node.next
        else:
            new_node.next = new_sibling
            new_sibling.next = sibling.next
        return new_node, new_sibling

    def _redistribute_internals(self, node: FRNode, sibling: FRNode,
                                 parent: FRNode, idx: int,
                                 from_left: bool) -> Tuple[FRNode, FRNode]:
        new_node = FRNode(is_leaf=False)
        new_sibling = FRNode(is_leaf=False)
        if from_left:
            sep = parent.keys[idx - 1]
            new_node.keys = [sep] + node.keys
            new_node.children = [sibling.children[-1]] + node.children
            new_sibling.keys = sibling.keys[:-1]
            new_sibling.children = sibling.children[:-1]
            # The new separator at parent will be sibling.keys[-1]
        else:
            sep = parent.keys[idx]
            new_node.keys = node.keys + [sep]
            new_node.children = node.children + [sibling.children[0]]
            new_sibling.keys = sibling.keys[1:]
            new_sibling.children = sibling.children[1:]
        for c in new_node.children:
            c.parent = new_node
        for c in new_sibling.children:
            c.parent = new_sibling
        return new_node, new_sibling

    def _install_borrow_at_parent(self, parent: FRNode, idx: int,
                                   old_node: FRNode, old_sibling: FRNode,
                                   new_node: FRNode, new_sibling: FRNode,
                                   from_left: bool) -> None:
        if from_left:
            # sibling at idx-1, node at idx
            parent.children[idx - 1] = new_sibling
            parent.children[idx] = new_node
            # new separator: smallest key under new_node (or its first key for internals)
            if new_node.is_leaf:
                parent.keys[idx - 1] = new_node.keys[0]
            else:
                parent.keys[idx - 1] = old_sibling.keys[-1]
        else:
            # node at idx, sibling at idx+1
            parent.children[idx] = new_node
            parent.children[idx + 1] = new_sibling
            if new_sibling.is_leaf:
                parent.keys[idx] = new_sibling.keys[0]
            else:
                parent.keys[idx] = old_sibling.keys[0]
        new_node.parent = parent
        new_sibling.parent = parent
        self.metrics.staged_parent_rewrites += 1

    # ----- join (merge) ----------------------------------------------
    def _freeze_and_join(self, node: FRNode, sibling: FRNode,
                          parent: FRNode, idx: int, with_left: bool) -> None:
        self.metrics.joins += 1
        self._transition(node, NodeState.FREEZE)
        self._transition(sibling, NodeState.REQUEST_SLAVE)
        self._transition(sibling, NodeState.SLAVE_FREEZE)

        merged = FRNode(is_leaf=node.is_leaf)
        if node.is_leaf:
            if with_left:
                merged.keys = sibling.keys + node.keys
                merged.values = sibling.values + node.values
                merged.next = node.next
            else:
                merged.keys = node.keys + sibling.keys
                merged.values = node.values + sibling.values
                merged.next = sibling.next
        else:
            if with_left:
                sep = parent.keys[idx - 1]
                merged.keys = sibling.keys + [sep] + node.keys
                merged.children = sibling.children + node.children
            else:
                sep = parent.keys[idx]
                merged.keys = node.keys + [sep] + sibling.keys
                merged.children = node.children + sibling.children
            for c in merged.children:
                c.parent = merged

        self.metrics.temp_key_duplications += len(node.keys) + len(sibling.keys)

        node.replacement = [merged]
        sibling.replacement = [merged]

        if with_left:
            # remove key idx-1 and child idx (the right one of the pair)
            parent.keys.pop(idx - 1)
            parent.children.pop(idx)
            parent.children[idx - 1] = merged
        else:
            parent.keys.pop(idx)
            parent.children.pop(idx + 1)
            parent.children[idx] = merged
        merged.parent = parent
        self.metrics.staged_parent_rewrites += 1

        self._activate(merged)
        # Terminate both frozen nodes as JOIN, then RECLAIMED.
        self._transition(node, NodeState.JOIN)
        self._transition(node, NodeState.RECLAIMED)
        self._transition(sibling, NodeState.JOIN)
        self._transition(sibling, NodeState.RECLAIMED)

    # -- structural metrics -------------------------------------------
    def height(self) -> int:
        h = 0
        node = self._follow_replacement(self.root)
        while not node.is_leaf:
            h += 1
            node = self._follow_replacement(node.children[0])
        return h

    def average_search_path(self) -> float:
        total_depth = 0
        total_keys = 0
        def walk(node: FRNode, depth: int) -> None:
            nonlocal total_depth, total_keys
            node = self._follow_replacement(node)
            if node.is_leaf:
                total_depth += depth * len(node.keys)
                total_keys += len(node.keys)
            else:
                for c in node.children:
                    walk(c, depth + 1)
        walk(self.root, 1)
        return total_depth / total_keys if total_keys else 0.0

    def node_utilization(self) -> float:
        capacities = []
        def walk(node: FRNode) -> None:
            node = self._follow_replacement(node)
            if node is not self.root:
                cap = self.order - 1
                capacities.append(len(node.keys) / cap)
            if not node.is_leaf:
                for c in node.children:
                    walk(c)
        walk(self.root)
        return sum(capacities) / len(capacities) if capacities else 1.0

    def count_live_nodes(self) -> int:
        n = 0
        def walk(node: FRNode) -> None:
            nonlocal n
            node = self._follow_replacement(node)
            n += 1
            if not node.is_leaf:
                for c in node.children:
                    walk(c)
        walk(self.root)
        return n
