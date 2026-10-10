"""Finite-difference gradient check for the CORRECTED symmetric-KL backward pass.

Two checks are run on tiny dense fixtures (no dataset, no training):

1. symmetric-KL logit gradients (the required correction):
   loss(z_o, z_r) = mean_B 0.5*(KL(p||q) + KL(q||p)), p = softmax(z_o),
   q = softmax(z_r), with epsilon clipping. The analytical gradients
   g_p = 0.5/B * [ p*(log(p)-log(q)-KL(p||q)) + (p-q) ]
   g_q = 0.5/B * [ q*(log(q)-log(p)-KL(q||p)) + (q-p) ]
   (implemented in student.symmetric_kl_logit_grads) are compared against
   central finite differences of the loss w.r.t. every logit entry.

2. full student parameter gradients (wiring check for CE + symmetric KL +
   shared layer + heads): a tiny TfidfStudent (dropout=0) is built in float64;
   the analytical parameter gradients from `loss_and_grads` are compared
   against central finite differences of `loss_value` w.r.t. every parameter
   element.

Outputs (saved under ../gradient_check/):
- symmetric_kl_logit_gradients.csv  (fixture logits, numerical, analytical)
- parameter_gradient_check.csv      (per-parameter max abs error)
- gradient_check_results.json       (max abs error, tolerance, PASS/FAIL)
- GRADIENT_CHECK.md                 (human-readable summary)

The pipeline does NOT proceed to training if either check FAILS.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (GRADIENT_CHECK_DIR, RUN_LABEL, RUN_DATE, LABEL_SPACES,
                    CONSISTENCY_TASKS, write_json, ensure_dirs)
from student import TfidfStudent, symmetric_kl_value, symmetric_kl_logit_grads, _softmax

TOLERANCE_KL = 1e-6       # max abs error allowed for the symmetric-KL check
TOLERANCE_PARAM = 1e-5    # max abs error allowed for the parameter check
FD_H = 1e-6               # central finite-difference step (float64 fixtures)

FIXTURE_SEED = 20260722
FIXTURE_B = 5             # batch rows
FIXTURE_C = 7             # classes


def check_symmetric_kl():
    """Check the corrected symmetric-KL logit gradients vs finite differences."""
    rng = np.random.default_rng(FIXTURE_SEED)
    z_o = rng.standard_normal((FIXTURE_B, FIXTURE_C)) * 1.7
    z_r = rng.standard_normal((FIXTURE_B, FIXTURE_C)) * 1.7

    def loss(zo, zr):
        return float(symmetric_kl_value(_softmax(zo), _softmax(zr)).mean())

    p = _softmax(z_o)
    q = _softmax(z_r)
    g_p, g_q = symmetric_kl_logit_grads(p, q)

    rows = []
    max_err = 0.0
    for name, z, g in [("z_original", z_o, g_p), ("z_redacted", z_r, g_q)]:
        for i in range(FIXTURE_B):
            for j in range(FIXTURE_C):
                zp = z.copy(); zp[i, j] += FD_H
                zm = z.copy(); zm[i, j] -= FD_H
                if name == "z_original":
                    num = (loss(zp, z_r) - loss(zm, z_r)) / (2 * FD_H)
                else:
                    num = (loss(z_o, zp) - loss(z_o, zm)) / (2 * FD_H)
                ana = float(g[i, j])
                err = abs(num - ana)
                max_err = max(max_err, err)
                rows.append({"logits": name, "row": i, "col": j,
                             "logit_value": float(z[i, j]),
                             "numerical_gradient": num,
                             "analytical_gradient": ana,
                             "abs_error": err})
    df = pd.DataFrame(rows)
    return df, float(max_err), float(loss(z_o, z_r))


def check_student_parameters():
    """Check full loss_and_grads parameter gradients vs finite differences."""
    rng = np.random.default_rng(FIXTURE_SEED + 1)
    feature_dim, shared_dim, B = 12, 8, 6
    model = TfidfStudent(feature_dim, seed=FIXTURE_SEED + 2,
                         shared_dim=shared_dim, dropout=0.0)
    model.set_dtype(np.float64)  # float64 fixture for clean finite differences

    X_o = rng.standard_normal((B, feature_dim))
    X_r = rng.standard_normal((B, feature_dim))
    # make the redacted view slightly different (as redaction would)
    X_r = X_r + 0.05 * rng.standard_normal((B, feature_dim))
    y_o, y_r = {}, {}
    for t in LABEL_SPACES:
        n_cls = len(LABEL_SPACES[t])
        y_o[t] = rng.integers(0, n_cls, size=B)
        if t == "pii_risk":
            y_r[t] = rng.integers(0, n_cls, size=B)  # view-specific risk target
        else:
            y_r[t] = y_o[t]                          # shared semantic target
    lam = 0.07

    loss, kl_val, grads = model.loss_and_grads(X_o, X_r, y_o, y_r, lam)

    rows = []
    max_err = 0.0
    for k in sorted(model.params.keys()):
        param = model.params[k]
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + FD_H
            lp = model.loss_value(X_o, X_r, y_o, y_r, lam)
            param[idx] = orig - FD_H
            lm = model.loss_value(X_o, X_r, y_o, y_r, lam)
            param[idx] = orig
            num_grad[idx] = (lp - lm) / (2 * FD_H)
            it.iternext()
        ana_grad = np.asarray(grads[k], dtype=np.float64)
        err = float(np.max(np.abs(num_grad - ana_grad)))
        max_err = max(max_err, err)
        rows.append({"parameter": k, "shape": "x".join(str(s) for s in param.shape),
                     "num_elements": int(param.size),
                     "max_abs_error": err})
    df = pd.DataFrame(rows)
    return df, float(max_err), float(loss), float(kl_val)


def main():
    ensure_dirs()
    print(f"[{RUN_LABEL}] finite-difference gradient check (tolerance: "
          f"KL {TOLERANCE_KL}, params {TOLERANCE_PARAM}, fd step {FD_H})")

    # ---- check 1: symmetric-KL logit gradients (the required correction) ----
    kl_df, kl_max_err, kl_loss = check_symmetric_kl()
    kl_pass = bool(kl_max_err <= TOLERANCE_KL)
    print(f"symmetric-KL logit grads: max_abs_error={kl_max_err:.3e} "
          f"tolerance={TOLERANCE_KL:.1e} -> {'PASS' if kl_pass else 'FAIL'}")

    # ---- check 2: full student parameter gradients (wiring check) ----
    param_df, param_max_err, train_loss, kl_val = check_student_parameters()
    param_pass = bool(param_max_err <= TOLERANCE_PARAM)
    print(f"student parameter grads:  max_abs_error={param_max_err:.3e} "
          f"tolerance={TOLERANCE_PARAM:.1e} -> {'PASS' if param_pass else 'FAIL'}")

    overall = kl_pass and param_pass

    kl_df.to_csv(GRADIENT_CHECK_DIR / "symmetric_kl_logit_gradients.csv", index=False)
    param_df.to_csv(GRADIENT_CHECK_DIR / "parameter_gradient_check.csv", index=False)

    results = {
        "run_label": RUN_LABEL,
        "date": RUN_DATE,
        "purpose": "finite-difference verification of the CORRECTED symmetric-KL "
                   "backward pass (and full student parameter gradients)",
        "fixture": {
            "seed": FIXTURE_SEED,
            "symmetric_kl": {"batch_rows": FIXTURE_B, "classes": FIXTURE_C,
                             "logit_scale": 1.7},
            "student": {"feature_dim": 12, "shared_dim": 8, "batch_rows": 6,
                        "dropout": 0.0, "dtype": "float64", "lambda": 0.07},
        },
        "method": {
            "analytical": "student.symmetric_kl_logit_grads / TfidfStudent.loss_and_grads",
            "numerical": f"central finite differences, step h={FD_H}",
            "loss": "batch-mean 0.5*(KL(p||q)+KL(q||p)) with epsilon clipping "
                    "(p=softmax(z_original), q=softmax(z_redacted))",
        },
        "symmetric_kl_logit_gradient_check": {
            "fixture_loss": kl_loss,
            "max_abs_error": kl_max_err,
            "tolerance": TOLERANCE_KL,
            "result": "PASS" if kl_pass else "FAIL",
            "detail_file": "gradient_check/symmetric_kl_logit_gradients.csv",
        },
        "student_parameter_gradient_check": {
            "fixture_train_loss": train_loss,
            "fixture_consistency_kl": kl_val,
            "max_abs_error": param_max_err,
            "tolerance": TOLERANCE_PARAM,
            "result": "PASS" if param_pass else "FAIL",
            "detail_file": "gradient_check/parameter_gradient_check.csv",
        },
        "overall_result": "PASS" if overall else "FAIL",
    }
    write_json(GRADIENT_CHECK_DIR / "gradient_check_results.json", results)

    md = []
    md.append("# Gradient Check — Corrected Symmetric-KL Backward Pass")
    md.append("")
    md.append(f"- Run label: `{RUN_LABEL}` | Date: {RUN_DATE}")
    md.append("- The reference `code_reference/student.py` ships an INCOMPLETE "
              "symmetric-KL backward pass (`lam*0.5*(p_o-p_r)` / `lam*0.5*(p_r-p_o)`). "
              "This run implements the complete batch-mean logit gradients and verifies "
              "them by central finite differences before any training.")
    md.append("")
    md.append("## Corrected gradient (implemented in `src/student.py`)")
    md.append("")
    md.append("For p = softmax(z_original), q = softmax(z_redacted), "
              "L = 0.5*(KL(p||q) + KL(q||p)), batch size B, epsilon clipping:")
    md.append("")
    md.append("```text")
    md.append("g_p = 0.5/B * [ p*(log(p) - log(q) - KL(p||q)) + (p - q) ]")
    md.append("g_q = 0.5/B * [ q*(log(q) - log(p) - KL(q||p)) + (q - p) ]")
    md.append("```")
    md.append("")
    md.append("## Check 1 — symmetric-KL logit gradients (required correction)")
    md.append("")
    md.append(f"- Fixture: B={FIXTURE_B} rows x C={FIXTURE_C} classes, "
              f"seed {FIXTURE_SEED}, logits ~ N(0, 1.7^2); loss value {kl_loss:.6f}")
    md.append(f"- Numerical gradient: central finite differences, h={FD_H}")
    md.append(f"- Maximum absolute error: **{kl_max_err:.3e}** (tolerance {TOLERANCE_KL:.1e})")
    md.append(f"- Result: **{'PASS' if kl_pass else 'FAIL'}**")
    md.append(f"- Per-entry numerical/analytical gradients: `symmetric_kl_logit_gradients.csv`")
    md.append("")
    md.append("## Check 2 — full student parameter gradients (wiring check)")
    md.append("")
    md.append("- Fixture: tiny dense student (feature_dim=12, shared_dim=8, B=6, "
              "dropout=0, float64, lambda=0.07)")
    md.append(f"- Analytical gradients: `TfidfStudent.loss_and_grads` "
              f"(supervised CE on both views + corrected symmetric KL)")
    md.append(f"- Numerical gradients: central finite differences of "
              f"`TfidfStudent.loss_value`, h={FD_H}")
    md.append(f"- Maximum absolute error: **{param_max_err:.3e}** "
              f"(tolerance {TOLERANCE_PARAM:.1e})")
    md.append(f"- Result: **{'PASS' if param_pass else 'FAIL'}**")
    md.append(f"- Per-parameter errors: `parameter_gradient_check.csv`")
    md.append("")
    md.append(f"## Overall: **{'PASS' if overall else 'FAIL'}** "
              "(training does not proceed on FAIL)")
    md.append("")
    md.append("Machine-readable results: `gradient_check_results.json`.")
    with open(GRADIENT_CHECK_DIR / "GRADIENT_CHECK.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"overall gradient check: {'PASS' if overall else 'FAIL'}")
    print(f"saved under {GRADIENT_CHECK_DIR}")
    if not overall:
        print("GRADIENT CHECK FAILED - refusing to proceed to training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
