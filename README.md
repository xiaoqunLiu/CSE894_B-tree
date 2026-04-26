# Freeze-and-Replace B+-Tree Project

A lightweight algorithmic study of the freeze-and-replace rebalancing
protocol from Braginsky and Petrank, *A Lock-Free B+tree* (SPAA 2012).

## Files

```
src/
  baseline_bplus.py         standard sequential B+-tree (in-place rebalancing)
  freeze_replace_bplus.py   simulator that models the freeze-and-replace protocol
  invariants.py             checker for the four correctness invariants
  evaluate.py               runs both trees on three workloads, prints results
  make_charts.py            builds the comparison figures used in the report
  build_report.py           builds the final PDF report
  results/
    results.json              raw numbers from the evaluation
    fig_structural_events.png chart used in the report
    fig_protocol_overhead.png chart used in the report
```

## How to reproduce

From the project root:

```
python3 src/evaluate.py        # run both trees, save results.json, print summary
python3 src/make_charts.py     # build the two PNG charts
python3 src/build_report.py    # build project_report.pdf
```

`evaluate.py` cross-checks at the end of each workload that
(a) both trees agree on which keys remain,
(b) every remaining key is found by `search`, and
(c) the simulator's invariant checker reports zero violations.

## Workloads

* **W1 sequential**: insert 0..199, delete 0..99
* **W2 random**: insert 200 random keys, delete 100 of them
* **W3 mixed**: 800 ops, ~70% inserts and ~30% deletes (random keys)
