"""CV-0: the calibration instrument (design CV-0 card).

Four sub-experiments -> four artifacts + a single machine-readable verdict:
    (a) determinism_audit  -> envelope.json (epsilon floors) + Figure S1
    (b) capability_screen  -> eligibility.parquet + capability_report.md
    (c) potency_screen     -> potency.json
    (d) variance_estimation-> re-solved power table
    gates + report         -> cv0_verdict.json (PASS/FAIL/KILL per gate)
"""

from __future__ import annotations
