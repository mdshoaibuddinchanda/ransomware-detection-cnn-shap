[![Research status](https://img.shields.io/badge/research-needs--revision-orange)](#current-status)
[![CI](https://github.com/mdshoaibuddinchanda/ransomware-detection-cnn2D/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mdshoaibuddinchanda/ransomware-detection-cnn2D/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-research%20prototype-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# 🔐 Ransomware Detection using CNN2D with Explainable AI and Robustness Evaluation

This is a behavior-based ransomware-detection research project developed as a
major project. It studies whether processor/HPC, memory, and disk-I/O telemetry
can identify ransomware-like behavior early while keeping false positives and
operational cost explicit.

The project preserves the original identity and research motivation while
upgrading the evaluation design. CNN2D is an evaluated candidate, not a preferred
answer chosen in advance. A model earns that position only through fair,
leakage-safe comparison against strong classical and temporal baselines.

## Project team

- **md shoaib uddin chanda** — 160922748092
- **Mohammed Asim** — 160922748108
- **Maimona Jaweed** — 160922748083

## Research question

Can a calibrated, resource-bounded temporal detector identify ransomware-like
behavior under a declared false-positive budget and remain useful across unseen
families, workloads, hosts, hardware configurations, time periods, and independent
datasets?

The project also asks two separate questions:

1. **What happens?** How do detection recall, false-positive rate, calibration,
   latency, and resource cost change across models and operating conditions?
2. **Why does it happen?** Which telemetry groups, temporal patterns, missing-data
   effects, and representation choices explain the observed behavior?

## System concept

```text
Authorized inert telemetry or safe replay
        ↓
Schema validation and provenance checks
        ↓
Session-aware temporal windows
        ↓
Fold-local preprocessing and model training
        ↓
Calibrated ransomware-risk score
        ↓
Persistence, cooldown, severity, and audit policy
        ↓
Offline notification adapter or audited alert record
```

The system is defensive research software. It does not execute ransomware, collect
live endpoint data, contact third-party systems, or claim production protection.

## Updated methodology

### Data and provenance

Every dataset must record its source, license, schema, labels, collection context,
hashes, missingness, class balance, and the independent units available for splitting.
Raw public data remains local and is never treated as evidence merely because it was
downloaded.

The checked-in HPC CSV contains 6,000 labeled rows and 13 telemetry features with
balanced labels. It has no host, capture/session, family, or timestamp field, so it
supports diagnostic row-level comparisons only. It cannot support a strong claim of
temporal or cross-family generalization by itself.

### Temporal representation

The unit of inference is a timestamped window belonging to one host or capture
session. A semantic CNN2D representation must document its axes, feature groups,
time step, stride, channel meaning, normalization, and missingness handling. An
arbitrary reshape of flat columns into an image is not accepted as a scientific
CNN2D contribution.

### Model comparison

The comparison ladder includes, when supported by the admitted data:

- majority and logistic-regression baselines;
- random forest, gradient boosting, and XGBoost;
- MLP;
- temporal CNN/TCN and recurrent models;
- semantic CNN2D; and
- a compact attention model.

All candidates use comparable data, preprocessing boundaries, split definitions,
seed sets, tuning budgets, stopping rules, and reporting metrics. A failed model is
reported as failed; it is never silently replaced by another model.

### Evaluation and explanation

Development uses grouped or temporal splits whenever those fields exist. If no
independent groups exist, the default diagnostic protocol is 5-fold cross-validation
repeated 4 times with fixed published seeds and an explicit limitation.

Final evidence requires an untouched external, family-held-out, host-held-out, or
future-time holdout. Results report precision, recall, specificity, F1, balanced
accuracy, PR-AUC, ROC-AUC, false-positive rate, Brier score, calibration, uncertainty
intervals, detection latency, and resource cost.

Explainability is tied to the evaluated model and its preprocessing. Attribution
plots are treated as model-behavior evidence, not proof of causality. Mechanistic
claims require stability checks and controlled interventions.

## Current status

The project is **needs revision**, not a final conference-ready result. The original
README's hard-coded accuracy and CNN2D-superiority statements are retained only as
historical project context and are not current publishable evidence.

The major remaining scientific gates are:

- a sufficiently large independent temporal benchmark;
- a frozen untouched holdout inaccessible during tuning;
- window-length, feature-group, layout, missing-telemetry, and alert-policy
  ablations;
- seed sensitivity and uncertainty analysis;
- a declared damage/onset proxy for early-warning latency;
- model-faithful explanation stability and mechanism tests; and
- clean-checkout reproduction of every reported table and figure.

## Historical baseline and updated implementation

The preserved historical snapshot contains the original `Main.py`, `run.bat`,
notebook, dataset files, saved models, and figures. Those artifacts document the
major-project baseline but should not be used as proof of current scientific claims.

The updated implementation is being reconstructed as a headless, testable research
pipeline with explicit schemas, temporal windows, fold-local preprocessing,
calibration, holdout evaluation, latency measurement, alert replay, manifests, and
CI checks. The migration remains intentionally separate from the preserved GitHub
history until it is explicitly accepted.

## Continuous integration

Every push and pull request runs a clean-checkout workflow. It checks Python syntax,
project-file integrity, notebook JSON structure, CSV columns and numeric values,
label validity, and the declared dependency manifest across Python 3.10 through
3.13.

The preserved GUI creates a Tk window at import time and uses legacy TensorFlow and
Keras pins, so CI does not pretend to train those models headlessly. The workflow
reports exactly what it verifies; model-level tests will be added with the modern
package migration.

## Safety and reproducibility

Use only inert datasets or explicitly authorized, isolated, disposable simulations.
Never commit secrets, private prompts, victim data, live malware, or destructive
payloads. Apply the repository's safety controls before adding collectors or
external data.

See [LICENSE](LICENSE) for the project license. No commit, push, or Git timestamp
rewrite is implied by this README update.
