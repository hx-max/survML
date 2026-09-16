"""survML — Survival Analysis task for FLAML (end-to-end demo).

This script demonstrates the survival-analysis task added on top of FLAML:

    automl.fit(X, y, task="survival", metric="c_index")

where ``y`` is a right-censored target with two columns ``(duration, event)``.
It runs automatic model selection + hyperparameter tuning and reports the
best concordance index (C-index).

Run from the project root:

    pip install lifelines
    python survival_demo.py
"""
import numpy as np
from lifelines.datasets import load_rossi

from flaml import AutoML

# rossi recidivism dataset: `week` = survival time, `arrest` = event (0/1).
df = load_rossi()
feature_cols = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
X = df[feature_cols]
y = df[["week", "arrest"]]  # (duration, event) — right-censored

automl = AutoML()
automl.fit(
    X,
    y,
    task="survival",
    metric="c_index",
    time_budget=15,
    max_iter=3,
    verbose=1,
)

print("\n=== survML result ===")
print("best estimator :", automl.best_estimator)
print("best C-index   :", round(1 - automl.best_loss, 4))
print("best config    :", automl.best_config)

risk = automl.predict(X)
print("risk score shape:", np.asarray(risk).shape)
print("sample risk (higher = higher hazard):", np.round(np.asarray(risk)[:5].ravel(), 3))
