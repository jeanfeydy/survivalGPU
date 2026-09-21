Examples
========

Small, self-contained scripts showing how to use survivalGPU's two models —
**CoxPH** and **WCE** — with and without bootstrap resampling.

All examples fit on synthetic data (via ``common.py``, which uses
survivalGPU's own simulator so the "true" coefficients / hazard ratio are
known) and run on **CPU by default**, so they work on any machine. Each
script has a comment showing how to switch to ``device="cuda"`` to run on a
GPU if one is available.
