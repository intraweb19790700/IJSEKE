"""RC-EdgeDistill TF-IDF student (FAITHFUL_REIMPLEMENTATION) in pure numpy.

Architecture (per submitted manuscript Section 3.3):
- input: concatenated char_wb (2,5) TF-IDF (max 6000) + word (1,2) TF-IDF
  (max 4000), fitted on paired train views;
- shared projection to 384 dimensions (ReLU - assumed, recorded);
- dropout 0.15 on the shared representation during training (inverted);
- four heads: service_domain (10), sentiment (3), priority (3),
  pii_risk (2);
- supervised cross-entropy for all four heads on both views;
- same semantic targets for original/redacted service_domain, sentiment,
  priority; view-specific pii_risk target (pii_original for the original
  view, pii_redacted for the redacted view);
- symmetric KL consistency = 0.5*(KL(p_o||p_r) + KL(p_r||p_o)) for
  service_domain, sentiment, priority only (gradients flow to each side's own
  logits; the other side is detached - assumed, recorded);
- total loss = sum of supervised losses + lambda * symmetric KL;
- Adam lr=0.006, weight_decay=1e-4 (decoupled / AdamW-style - assumed,
  recorded), betas (0.9, 0.999), eps 1e-8.
"""
import numpy as np

from common import (LABEL_SPACES, CONSISTENCY_TASKS, SHARED_DIM, DROPOUT,
                    LR, WEIGHT_DECAY)


class TfidfStudent:
    def __init__(self, feature_dim, seed, shared_dim=SHARED_DIM,
                 dropout=DROPOUT, lr=LR, weight_decay=WEIGHT_DECAY):
        self.feature_dim = feature_dim
        self.shared_dim = shared_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        rng = np.random.default_rng(seed)
        # He init for the shared projection (ReLU), Glorot for heads.
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
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self.rng = np.random.default_rng(seed + 1)

    # ---------------------------------------------------------- forward
    def _shared(self, X, train):
        H = X @ self.W1 + self.b1
        A = np.maximum(H, 0.0)
        mask = None
        if train and self.dropout > 0:
            mask = (self.rng.random(A.shape) >= self.dropout).astype(np.float32)
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
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _kl(p, q):
        # KL(p || q), p and q are probability rows
        return np.sum(p * (np.log(np.clip(p, 1e-12, None)) -
                           np.log(np.clip(q, 1e-12, None))), axis=1).mean()

    def loss_and_grads(self, X_o, X_r, y_o, y_r, lam):
        """y_o/y_r: dict task -> int label array (batch,)."""
        B = X_o.shape[0]
        logits_o, H_o, A_o, mask_o = self.forward(X_o, train=True)
        logits_r, H_r, A_r, mask_r = self.forward(X_r, train=True)
        p_o = {t: self._softmax(logits_o[t]) for t in LABEL_SPACES}
        p_r = {t: self._softmax(logits_r[t]) for t in LABEL_SPACES}

        loss = 0.0
        dlogits_o = {t: (p_o[t] - np.eye(p_o[t].shape[1], dtype=np.float32)[y_o[t]]) / B
                     for t in LABEL_SPACES}
        dlogits_r = {t: (p_r[t] - np.eye(p_r[t].shape[1], dtype=np.float32)[y_r[t]]) / B
                     for t in LABEL_SPACES}
        for t in LABEL_SPACES:
            ce_o = -np.log(np.clip(p_o[t][np.arange(B), y_o[t]], 1e-12, None)).mean()
            ce_r = -np.log(np.clip(p_r[t][np.arange(B), y_r[t]], 1e-12, None)).mean()
            loss += float(ce_o) + float(ce_r)

        # symmetric KL consistency for semantic tasks only
        kl_val = 0.0
        for t in CONSISTENCY_TASKS:
            kl = 0.5 * (self._kl(p_o[t], p_r[t]) + self._kl(p_r[t], p_o[t]))
            kl_val += float(kl)
            # gradients: each direction flows to its own logits (other detached)
            dlogits_o[t] = dlogits_o[t] + lam * 0.5 * (p_o[t] - p_r[t])
            dlogits_r[t] = dlogits_r[t] + lam * 0.5 * (p_r[t] - p_o[t])
        loss += lam * kl_val

        # backward
        grads = {}
        dA_o = np.zeros_like(A_o)
        dA_r = np.zeros_like(A_r)
        for t in LABEL_SPACES:
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

    # ------------------------------------------------------ save / load
    def save(self, path):
        np.savez_compressed(path, t=self.t, **{k: v for k, v in self.params.items()})

    def load(self, path):
        data = np.load(path)
        self.t = int(data["t"])
        for k in self.params:
            self.params[k] = data[k].astype(np.float32)
        self.W1 = self.params["W1"]
        self.b1 = self.params["b1"]
        for task in self.heads:
            self.heads[task]["W"] = self.params[f"{task}.W"]
            self.heads[task]["b"] = self.params[f"{task}.b"]
