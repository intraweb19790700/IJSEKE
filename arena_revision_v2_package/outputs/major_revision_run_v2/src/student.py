"""RC-EdgeDistill TF-IDF student (CORRECTED faithful reimplementation, pure numpy).

Architecture (per manuscript IJSEKE-D-26-00185 Section 3.3):
- input: concatenated char_wb (2,5) TF-IDF (max 6,000) + word (1,2) TF-IDF
  (max 4,000), fitted on paired TRAIN views only (original + redacted);
- shared projection to 384 dimensions (ReLU - assumed, recorded);
- dropout 0.15 on the shared representation during training (inverted);
- four heads: service_domain (10), sentiment (3), priority (3), pii_risk (2);
- supervised cross-entropy for all four heads on BOTH views
  (original view: pii target = pii_original; redacted view: pii target =
  pii_redacted; the three semantic tasks share their targets across views);
- symmetric KL consistency = 0.5*(KL(p_o||p_r) + KL(p_r||p_o)) for
  service_domain, sentiment, priority ONLY (pii_risk excluded);
- total loss = sum of supervised losses + lambda * sum of symmetric KL;
- training: 160 FULL-BATCH epochs, Adam lr=0.006, weight decay 1e-4
  (decoupled / AdamW-style - assumed, recorded), betas (0.9, 0.999), eps 1e-8.

CORRECTION vs code_reference/student.py
---------------------------------------
The reference backward pass used the one-directional approximation
    dlogits_o += lam * 0.5 * (p_o - p_r)
    dlogits_r += lam * 0.5 * (p_r - p_o)
which is NOT the gradient of the symmetric KL. For p = softmax(z_o),
q = softmax(z_r) and L = 0.5*(KL(p||q) + KL(q||p)) with batch size B and
epsilon clipping, the COMPLETE batch-mean logit gradients are

    g_p = 0.5/B * [ p*(log(p) - log(q) - KL(p||q)) + (p - q) ]
    g_q = 0.5/B * [ q*(log(q) - log(p) - KL(q||p)) + (q - p) ]

The first term is the numerator-distribution derivative and the second term
is the denominator-distribution derivative. This module implements exactly
that (see `symmetric_kl_value` / `symmetric_kl_logit_grads`, verified by a
finite-difference check in ../gradient_check/ and src/gradient_check.py).
"""
import numpy as np

from common import (LABEL_SPACES, CONSISTENCY_TASKS, ALL_TASKS, SHARED_DIM,
                    DROPOUT, LR, WEIGHT_DECAY)

EPS = 1e-12  # epsilon clipping for numerical stability


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def symmetric_kl_value(p, q, eps=EPS):
    """Batch-mean L = 0.5*(KL(p||q) + KL(q||p)) with epsilon clipping.

    p, q: (B, C) probability rows (softmax outputs). Returns a scalar.
    """
    log_p = np.log(np.clip(p, eps, None))
    log_q = np.log(np.clip(q, eps, None))
    kl_pq = np.sum(p * (log_p - log_q), axis=1)
    kl_qp = np.sum(q * (log_q - log_p), axis=1)
    return 0.5 * (kl_pq + kl_qp)  # (B,) per-sample symmetric KL


def symmetric_kl_logit_grads(p, q, eps=EPS):
    """Complete batch-mean logit gradients of the symmetric KL (THE CORRECTION).

    For L = mean_B 0.5*(KL(p||q) + KL(q||p)), p = softmax(z_p), q = softmax(z_q):

        g_p = 0.5/B * [ p*(log(p) - log(q) - KL(p||q)) + (p - q) ]
        g_q = 0.5/B * [ q*(log(q) - log(p) - KL(q||p)) + (q - p) ]

    Returns (g_p, g_q), each (B, C), the gradients w.r.t. the logits z_p, z_q.
    """
    B = p.shape[0]
    log_p = np.log(np.clip(p, eps, None))
    log_q = np.log(np.clip(q, eps, None))
    kl_pq = np.sum(p * (log_p - log_q), axis=1, keepdims=True)  # (B, 1)
    kl_qp = np.sum(q * (log_q - log_p), axis=1, keepdims=True)  # (B, 1)
    g_p = 0.5 / B * (p * (log_p - log_q - kl_pq) + (p - q))
    g_q = 0.5 / B * (q * (log_q - log_p - kl_qp) + (q - p))
    return g_p, g_q


class TfidfStudent:
    def __init__(self, feature_dim, seed, shared_dim=SHARED_DIM,
                 dropout=DROPOUT, lr=LR, weight_decay=WEIGHT_DECAY):
        self.feature_dim = feature_dim
        self.shared_dim = shared_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        rng = np.random.default_rng(seed)
        # He init for the shared projection (ReLU), Glorot-style for heads.
        self.W1 = (rng.standard_normal((feature_dim, shared_dim)) *
                   np.sqrt(2.0 / feature_dim)).astype(np.float32)
        self.b1 = np.zeros(shared_dim, dtype=np.float32)
        self.heads = {}
        for task, labels in LABEL_SPACES.items():
            n_cls = len(labels)
            w = (rng.standard_normal((shared_dim, n_cls)) *
                 np.sqrt(2.0 / (shared_dim + n_cls))).astype(np.float32)
            b = np.zeros(n_cls, dtype=np.float32)
            self.heads[task] = {"W": w, "b": b}
        self.params = {"W1": self.W1, "b1": self.b1}
        for task, h in self.heads.items():
            self.params[f"{task}.W"] = h["W"]
            self.params[f"{task}.b"] = h["b"]
        self._sync_refs()
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self.rng = np.random.default_rng(seed + 1)

    def _sync_refs(self):
        self.W1 = self.params["W1"]
        self.b1 = self.params["b1"]
        for task in self.heads:
            self.heads[task]["W"] = self.params[f"{task}.W"]
            self.heads[task]["b"] = self.params[f"{task}.b"]

    def set_dtype(self, dtype):
        """Cast all parameters to `dtype` (used by the float64 gradient check)."""
        for k in self.params:
            self.params[k] = self.params[k].astype(dtype)
        self._sync_refs()
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}

    # ---------------------------------------------------------- forward
    def _shared(self, X, train):
        H = X @ self.W1 + self.b1
        A = np.maximum(H, 0.0)
        mask = None
        if train and self.dropout > 0:
            mask = (self.rng.random(A.shape) >= self.dropout).astype(A.dtype)
            mask /= (1.0 - self.dropout)
            A = A * mask
        return H, A, mask

    def forward(self, X, train=False):
        H, A, mask = self._shared(X, train)
        logits = {task: A @ h["W"] + h["b"] for task, h in self.heads.items()}
        return logits, H, A, mask

    def predict_idx(self, X):
        logits, _, _, _ = self.forward(X, train=False)
        return {task: logits[task].argmax(axis=1) for task in LABEL_SPACES}

    # ------------------------------------------------------ loss + grads
    @staticmethod
    def _ce_terms(p, y, B):
        """Per-sample cross-entropy and its logit gradient (softmax + CE)."""
        ce = -np.log(np.clip(p[np.arange(B), y], EPS, None))          # (B,)
        dlogits = (p - np.eye(p.shape[1], dtype=p.dtype)[y]) / B      # (B, C)
        return ce, dlogits

    def loss_value(self, X_o, X_r, y_o, y_r, lam):
        """Scalar training loss (no gradients) - used by the gradient check."""
        B = X_o.shape[0]
        logits_o, _, _, _ = self.forward(X_o, train=False)
        logits_r, _, _, _ = self.forward(X_r, train=False)
        p_o = {t: _softmax(logits_o[t]) for t in LABEL_SPACES}
        p_r = {t: _softmax(logits_r[t]) for t in LABEL_SPACES}
        loss = 0.0
        for t in LABEL_SPACES:
            ce_o, _ = self._ce_terms(p_o[t], y_o[t], B)
            ce_r, _ = self._ce_terms(p_r[t], y_r[t], B)
            loss += float(ce_o.mean()) + float(ce_r.mean())
        for t in CONSISTENCY_TASKS:
            loss += float(lam * symmetric_kl_value(p_o[t], p_r[t]).mean())
        return float(loss)

    def loss_and_grads(self, X_o, X_r, y_o, y_r, lam):
        """Full-batch loss and parameter gradients.

        y_o / y_r: dict task -> int label array (B,). For the original view
        y_o["pii_risk"] is the pii_original target; for the redacted view
        y_r["pii_risk"] is the pii_redacted target; the three semantic tasks
        share targets across views.
        """
        B = X_o.shape[0]
        logits_o, H_o, A_o, mask_o = self.forward(X_o, train=True)
        logits_r, H_r, A_r, mask_r = self.forward(X_r, train=True)
        p_o = {t: _softmax(logits_o[t]) for t in LABEL_SPACES}
        p_r = {t: _softmax(logits_r[t]) for t in LABEL_SPACES}

        loss = 0.0
        dlogits_o = {}
        dlogits_r = {}
        for t in LABEL_SPACES:
            ce_o, dlogits_o[t] = self._ce_terms(p_o[t], y_o[t], B)
            ce_r, dlogits_r[t] = self._ce_terms(p_r[t], y_r[t], B)
            loss += float(ce_o.mean()) + float(ce_r.mean())

        # ---- symmetric KL consistency (semantic tasks only) - CORRECTED ----
        kl_val = 0.0
        for t in CONSISTENCY_TASKS:
            kl_per_sample = symmetric_kl_value(p_o[t], p_r[t])
            kl = float(kl_per_sample.mean())
            kl_val += kl
            g_p, g_q = symmetric_kl_logit_grads(p_o[t], p_r[t])
            dlogits_o[t] = dlogits_o[t] + lam * g_p
            dlogits_r[t] = dlogits_r[t] + lam * g_q
        loss += lam * kl_val

        # ---------------------------------------------------------- backward
        grads = {}
        dA_o = np.zeros_like(A_o)
        dA_r = np.zeros_like(A_r)
        for t in ALL_TASKS:
            h = self.heads[t]
            grads[f"{t}.W"] = A_o.T @ dlogits_o[t] + A_r.T @ dlogits_r[t]
            grads[f"{t}.b"] = dlogits_o[t].sum(axis=0) + dlogits_r[t].sum(axis=0)
            dA_o += dlogits_o[t] @ h["W"].T
            dA_r += dlogits_r[t] @ h["W"].T
        if mask_o is not None:
            dA_o = dA_o * mask_o
            dA_r = dA_r * mask_r
        dH_o = dA_o * (H_o > 0)
        dH_r = dA_r * (H_r > 0)
        grads["W1"] = X_o.T @ dH_o + X_r.T @ dH_r
        grads["b1"] = dH_o.sum(axis=0) + dH_r.sum(axis=0)
        return float(loss), kl_val, grads

    # ----------------------------------------------------------- adam
    def step(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * self.weight_decay * p          # decoupled weight decay
            p -= self.lr * mhat / (np.sqrt(vhat) + eps)
            self.params[k] = p
        self._sync_refs()
