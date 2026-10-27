"""Model-independent retrospective content grouping and deterministic pool allocation."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import random

PROTOCOL = "SSI-ContentGroup-Sparse3-v1"
SPLIT_SEED = 20261023


def connected_groups(nodes, edges):
    parent = {n: n for n in nodes}
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for a, b in edges:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    components = defaultdict(list)
    for n in sorted(nodes):
        components[find(n)].append(n)
    result = {}
    for members in components.values():
        gid = "CG_" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        for n in members:
            result[n] = gid
    return result


def allocate_groups(rows, seed=SPLIT_SEED):
    """Largest groups first; squared class-count error, seeded ties; no model inputs."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    rng = random.Random(seed)
    order = sorted(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))
    splits = ["train", "validation", "test"]
    totals = [sum(r["label"] == c for r in rows) for c in [0, 1]]
    targets = {s: [t * ratio for t in totals] for s, ratio in zip(splits, [.6, .2, .2])}
    counts = {s: [0, 0] for s in splits}
    assignment = {}
    for g in order:
        inc = [sum(r["label"] == c for r in groups[g]) for c in [0, 1]]
        tie = splits.copy()
        rng.shuffle(tie)
        def cost(s):
            return sum(((counts[s][c] + inc[c] - targets[s][c]) ** 2 -
                        (counts[s][c] - targets[s][c]) ** 2) / max(1, targets[s][c]) for c in [0, 1])
        chosen = min(tie, key=cost)
        assignment[g] = chosen
        counts[chosen] = [counts[chosen][c] + inc[c] for c in [0, 1]]
    return assignment


def diverse_select(rows, count, seed):
    """Seeded group round-robin; choose one image per group before second images."""
    groups = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["image_path"]):
        groups[r["group_id"]].append(r)
    rng = random.Random(seed)
    order = sorted(groups)
    rng.shuffle(order)
    for group in order:
        rng.shuffle(groups[group])
    selected = []
    while len(selected) < min(count, len(rows)):
        for group in order:
            if groups[group]:
                selected.append(groups[group].pop())
                if len(selected) == min(count, len(rows)):
                    break
    return selected


def freeze_selection(rows, seed=SPLIT_SEED):
    assignment = allocate_groups(rows, seed)
    selected = {}
    for split, budgets in [("train", {0: 336, 1: 3}), ("validation", {0: 84, 1: 20})]:
        selected[split] = []
        for label, count in budgets.items():
            pool = [r for r in rows if assignment[r["group_id"]] == split and r["label"] == label]
            selected[split] += diverse_select(pool, count, seed + label + (100 if split == "validation" else 0))
    selected["test"] = [r for r in rows if assignment[r["group_id"]] == "test"]
    for split in selected:
        selected[split] = sorted(selected[split], key=lambda r: (r["label"], r["image_path"]))
    defects = [r for r in selected["validation"] if r["label"]]
    one = diverse_select(defects, 1, seed + 1000)[0]["image_path"]
    return assignment, selected, one
