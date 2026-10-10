# Gradient Check — Corrected Symmetric-KL Backward Pass

- Run label: `CORRECTED_FAITHFUL_REIMPLEMENTATION` | Date: 2026-10-10
- The reference `code_reference/student.py` ships an INCOMPLETE symmetric-KL backward pass (`lam*0.5*(p_o-p_r)` / `lam*0.5*(p_r-p_o)`). This run implements the complete batch-mean logit gradients and verifies them by central finite differences before any training.

## Corrected gradient (implemented in `src/student.py`)

For p = softmax(z_original), q = softmax(z_redacted), L = 0.5*(KL(p||q) + KL(q||p)), batch size B, epsilon clipping:

```text
g_p = 0.5/B * [ p*(log(p) - log(q) - KL(p||q)) + (p - q) ]
g_q = 0.5/B * [ q*(log(q) - log(p) - KL(q||p)) + (q - p) ]
```

## Check 1 — symmetric-KL logit gradients (required correction)

- Fixture: B=5 rows x C=7 classes, seed 20260722, logits ~ N(0, 1.7^2); loss value 1.872328
- Numerical gradient: central finite differences, h=1e-06
- Maximum absolute error: **3.036e-10** (tolerance 1.0e-06)
- Result: **PASS**
- Per-entry numerical/analytical gradients: `symmetric_kl_logit_gradients.csv`

## Check 2 — full student parameter gradients (wiring check)

- Fixture: tiny dense student (feature_dim=12, shared_dim=8, B=6, dropout=0, float64, lambda=0.07)
- Analytical gradients: `TfidfStudent.loss_and_grads` (supervised CE on both views + corrected symmetric KL)
- Numerical gradients: central finite differences of `TfidfStudent.loss_value`, h=1e-06
- Maximum absolute error: **2.949e-09** (tolerance 1.0e-05)
- Result: **PASS**
- Per-parameter errors: `parameter_gradient_check.csv`

## Overall: **PASS** (training does not proceed on FAIL)

Machine-readable results: `gradient_check_results.json`.
