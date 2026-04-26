"""Generate a simple bar chart comparing baseline vs FR overhead."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results", "results.json")
OUT_DIR = os.path.join(HERE, "results")

with open(RESULTS) as fh:
    data = json.load(fh)

# Plot 1: structural events match across baseline and FR.
fig, ax = plt.subplots(figsize=(7.5, 3.6))
labels = [r["workload"] for r in data]
splits_b = [r["baseline"]["splits"] for r in data]
splits_f = [r["freeze_replace"]["splits"] for r in data]
merges_b = [r["baseline"]["merges"] for r in data]
merges_f = [r["freeze_replace"]["joins"] for r in data]
borrows_b = [r["baseline"]["borrows"] for r in data]
borrows_f = [r["freeze_replace"]["borrows"] for r in data]

x = np.arange(len(labels))
w = 0.13
ax.bar(x - 2.5 * w, splits_b, w, label="splits (baseline)", color="#4C72B0")
ax.bar(x - 1.5 * w, splits_f, w, label="splits (FR)", color="#A6BDDB")
ax.bar(x - 0.5 * w, merges_b, w, label="merges (baseline)", color="#DD8452")
ax.bar(x + 0.5 * w, merges_f, w, label="joins (FR)", color="#F4C28A")
ax.bar(x + 1.5 * w, borrows_b, w, label="borrows (baseline)", color="#55A868")
ax.bar(x + 2.5 * w, borrows_f, w, label="borrows (FR)", color="#A8CFA0")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("count")
ax.set_title("Structural events: baseline vs freeze-and-replace (matching)")
ax.legend(fontsize=8, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_structural_events.png"), dpi=130)

# Plot 2: protocol overhead unique to FR.
fig, ax = plt.subplots(figsize=(7.5, 3.6))
trans = [r["freeze_replace"]["state_transitions"] for r in data]
rewr = [r["freeze_replace"]["staged_parent_rewrites"] for r in data]
dups = [r["freeze_replace"]["temp_key_duplications"] for r in data]

w = 0.25
ax.bar(x - w, trans, w, label="state transitions", color="#8172B2")
ax.bar(x, rewr, w, label="staged parent rewrites", color="#937860")
ax.bar(x + w, dups, w, label="temp key duplications", color="#DA8BC3")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("count")
ax.set_title("Protocol-specific overhead (FR only)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_protocol_overhead.png"), dpi=130)

print("wrote charts to", OUT_DIR)
