"""
Baseline sequential B+-tree.

This is a straightforward B+-tree that supports search, insert, delete,
split (on overflow), and merge/borrow (on underflow). It performs
in-place rebalancing, which is the standard approach.

This serves as the reference point for comparing against the
freeze-and-replace simulator.

Conventions
-----------
- Each node has at most ORDER children (for internals) or ORDER keys (for leaves).
- A node underflows when it has fewer than ceil(ORDER/2) children/keys
  (except the root, which is allowed to be smaller).
- Leaves are linked left-to-right to support range scans (we do not
  exercise range scans here, but the link is maintained for completeness).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# Metrics container (counts how many structural events happened)
# ---------------------------------------------------------------------------
@dataclass
class BaselineMetrics:
    splits: int = 0
    merges: int = 0
    borrows: int = 0
    inserts: int = 0
    deletes: int = 0
    searches: int = 0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class BNode:
    """A node in the sequential B+-tree."""

    def __init__(self, is_leaf: bool):
        self.is_leaf: bool = is_leaf
        # For leaves: keys[i] has value values[i]. For internals: keys
        # are separators and children[i] is the subtree for keys < keys[i]
        # (with children[len(keys)] being the rightmost subtree).
        self.keys: List[int] = []
        self.values: List[object] = []          # leaf-only
        self.children: List["BNode"] = []       # internal-only
        self.next: Optional["BNode"] = None     # leaf-only sibling link
        self.parent: Optional["BNode"] = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        kind = "L" if self.is_leaf else "I"
        return f"<{kind} keys={self.keys}>"


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------
class BPlusTree:
    """Sequential B+-tree with in-place rebalancing."""

    def __init__(self, order: int = 4):
        if order < 3:
            raise ValueError("order must be >= 3")
        self.order = order
        self.min_keys = math.ceil(order / 2) - 1  # min keys in non-root node
        self.root: BNode = BNode(is_leaf=True)
        self.metrics = BaselineMetrics()

    # -- search ---------------------------------------------------------
    def search(self, key: int) -> Optional[object]:
        self.metrics.searches += 1
        leaf = self._find_leaf(key)
        for k, v in zip(leaf.keys, leaf.values):
            if k == key:
                return v
        return None

    def _find_leaf(self, key: int) -> BNode:
        node = self.root
        while not node.is_leaf:
            # find the child to descend into
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        return node

    # -- insert ---------------------------------------------------------
    def insert(self, key: int, value: object = None) -> None:
        self.metrics.inserts += 1
        if value is None:
            value = key
        leaf = self._find_leaf(key)
        # if key already present, just update
        for i, k in enumerate(leaf.keys):
            if k == key:
                leaf.values[i] = value
                return
        # otherwise insert in sorted order
        i = 0
        while i < len(leaf.keys) and leaf.keys[i] < key:
            i += 1
        leaf.keys.insert(i, key)
        leaf.values.insert(i, value)
        if len(leaf.keys) >= self.order:
            self._split_leaf(leaf)

    def _split_leaf(self, leaf: BNode) -> None:
        self.metrics.splits += 1
        mid = len(leaf.keys) // 2
        new_leaf = BNode(is_leaf=True)
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        # maintain sibling link
        new_leaf.next = leaf.next
        leaf.next = new_leaf
        # promote the first key of new_leaf as the separator
        sep = new_leaf.keys[0]
        self._insert_into_parent(leaf, sep, new_leaf)

    def _insert_into_parent(self, left: BNode, sep: int, right: BNode) -> None:
        if left is self.root:
            new_root = BNode(is_leaf=False)
            new_root.keys = [sep]
            new_root.children = [left, right]
            left.parent = new_root
            right.parent = new_root
            self.root = new_root
            return
        parent = left.parent
        # locate left in parent.children
        idx = parent.children.index(left)
        parent.keys.insert(idx, sep)
        parent.children.insert(idx + 1, right)
        right.parent = parent
        if len(parent.children) > self.order:
            self._split_internal(parent)

    def _split_internal(self, node: BNode) -> None:
        self.metrics.splits += 1
        mid = len(node.keys) // 2
        sep = node.keys[mid]
        new_node = BNode(is_leaf=False)
        new_node.keys = node.keys[mid + 1:]
        new_node.children = node.children[mid + 1:]
        for c in new_node.children:
            c.parent = new_node
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        self._insert_into_parent(node, sep, new_node)

    # -- delete ---------------------------------------------------------
    def delete(self, key: int) -> bool:
        self.metrics.deletes += 1
        leaf = self._find_leaf(key)
        if key not in leaf.keys:
            return False
        i = leaf.keys.index(key)
        leaf.keys.pop(i)
        leaf.values.pop(i)
        if leaf is self.root:
            return True
        if len(leaf.keys) < self.min_keys:
            self._fix_underflow(leaf)
        else:
            # if we removed the leftmost key we may want to update
            # the separator in the parent; for our metrics this isn't
            # critical because search only needs ordering invariants
            self._maybe_refresh_separator(leaf)
        return True

    def _maybe_refresh_separator(self, node: BNode) -> None:
        # walk up while we are the leftmost child; the separator that
        # points to us in some ancestor may need updating after a delete
        if not node.keys or node is self.root:
            return
        cur = node
        while cur.parent is not None:
            parent = cur.parent
            idx = parent.children.index(cur)
            if idx > 0:
                # the separator at idx-1 separates children[idx-1] and us.
                # It should be >= our smallest key. Update it to our smallest.
                first_key = self._first_key(cur)
                if first_key is not None:
                    parent.keys[idx - 1] = first_key
                return
            cur = parent

    def _first_key(self, node: BNode) -> Optional[int]:
        while not node.is_leaf:
            if not node.children:
                return None
            node = node.children[0]
        return node.keys[0] if node.keys else None

    def _fix_underflow(self, node: BNode) -> None:
        parent = node.parent
        if parent is None:
            return
        idx = parent.children.index(node)
        left = parent.children[idx - 1] if idx > 0 else None
        right = parent.children[idx + 1] if idx + 1 < len(parent.children) else None

        # try to borrow from a sibling first
        if left is not None and self._can_lend(left):
            self._borrow_from_left(node, left, parent, idx)
            self.metrics.borrows += 1
            return
        if right is not None and self._can_lend(right):
            self._borrow_from_right(node, right, parent, idx)
            self.metrics.borrows += 1
            return

        # otherwise merge with a sibling
        if left is not None:
            self._merge_with_left(node, left, parent, idx)
        else:
            assert right is not None
            self._merge_with_right(node, right, parent, idx)
        self.metrics.merges += 1

        # parent may now underflow
        if parent is self.root:
            if not parent.keys:
                # root has only one child left; collapse it
                if parent.children:
                    self.root = parent.children[0]
                    self.root.parent = None
            return
        if len(parent.keys) < self.min_keys:
            self._fix_underflow(parent)

    def _can_lend(self, sibling: BNode) -> bool:
        if sibling.is_leaf:
            return len(sibling.keys) > self.min_keys
        return len(sibling.children) - 1 > self.min_keys

    def _borrow_from_left(self, node: BNode, left: BNode, parent: BNode, idx: int) -> None:
        if node.is_leaf:
            node.keys.insert(0, left.keys.pop())
            node.values.insert(0, left.values.pop())
            parent.keys[idx - 1] = node.keys[0]
        else:
            # rotate through parent
            sep = parent.keys[idx - 1]
            node.keys.insert(0, sep)
            moved_child = left.children.pop()
            node.children.insert(0, moved_child)
            moved_child.parent = node
            parent.keys[idx - 1] = left.keys.pop()

    def _borrow_from_right(self, node: BNode, right: BNode, parent: BNode, idx: int) -> None:
        if node.is_leaf:
            node.keys.append(right.keys.pop(0))
            node.values.append(right.values.pop(0))
            parent.keys[idx] = right.keys[0]
        else:
            sep = parent.keys[idx]
            node.keys.append(sep)
            moved_child = right.children.pop(0)
            node.children.append(moved_child)
            moved_child.parent = node
            parent.keys[idx] = right.keys.pop(0)

    def _merge_with_left(self, node: BNode, left: BNode, parent: BNode, idx: int) -> None:
        if node.is_leaf:
            left.keys.extend(node.keys)
            left.values.extend(node.values)
            left.next = node.next
        else:
            sep = parent.keys[idx - 1]
            left.keys.append(sep)
            left.keys.extend(node.keys)
            for c in node.children:
                c.parent = left
            left.children.extend(node.children)
        parent.keys.pop(idx - 1)
        parent.children.pop(idx)

    def _merge_with_right(self, node: BNode, right: BNode, parent: BNode, idx: int) -> None:
        if node.is_leaf:
            node.keys.extend(right.keys)
            node.values.extend(right.values)
            node.next = right.next
        else:
            sep = parent.keys[idx]
            node.keys.append(sep)
            node.keys.extend(right.keys)
            for c in right.children:
                c.parent = node
            node.children.extend(right.children)
        parent.keys.pop(idx)
        parent.children.pop(idx + 1)

    # -- structural metrics -------------------------------------------
    def height(self) -> int:
        h = 0
        node = self.root
        while not node.is_leaf:
            h += 1
            node = node.children[0]
        return h

    def average_search_path(self) -> float:
        # average over all keys in all leaves
        total_depth = 0
        total_keys = 0
        def walk(node: BNode, depth: int) -> None:
            nonlocal total_depth, total_keys
            if node.is_leaf:
                total_depth += depth * len(node.keys)
                total_keys += len(node.keys)
            else:
                for c in node.children:
                    walk(c, depth + 1)
        walk(self.root, 1)
        return total_depth / total_keys if total_keys else 0.0

    def node_utilization(self) -> float:
        # average (keys / capacity) across all non-root nodes
        capacities = []
        def walk(node: BNode) -> None:
            if node is not self.root:
                cap = self.order - 1
                capacities.append(len(node.keys) / cap)
            if not node.is_leaf:
                for c in node.children:
                    walk(c)
        walk(self.root)
        return sum(capacities) / len(capacities) if capacities else 1.0

    def count_nodes(self) -> int:
        n = 0
        def walk(node: BNode) -> None:
            nonlocal n
            n += 1
            if not node.is_leaf:
                for c in node.children:
                    walk(c)
        walk(self.root)
        return n
