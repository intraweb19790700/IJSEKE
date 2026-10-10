# Timing Protocol (common end-to-end boundary)

- Run label: `CORRECTED_FAITHFUL_REIMPLEMENTATION` | Date: 2026-10-10
- Boundary per record (identical for every method): input normalization (NFKC + whitespace collapse) -> local redaction (regex rules) -> TF-IDF construction (fitted vectorizer transform) -> inference -> decoding (argmax/label -> string) -> output serialization (JSON).
- Model-only latency is NOT mixed in; training/fitting time and vectorizer fitting are excluded.
- Records: 299 locked-test records in file order; per-record (batch=1); 1 warm-up pass + 5 measured repetitions; timer `time.perf_counter`; threads=1 (`OMP_NUM_THREADS=1`).
- CPU: Intel(R) Xeon(R) Processor @ 2.60GHz (2 cores); Python 3.11.2; numpy 2.4.6; pandas 3.0.6; scipy 1.17.1; scikit-learn 1.9.1.
- Timing predictions are never used for any metric.
- Aggregate results: `end_to_end_timing.csv`; environment manifest: `timing_manifest.json`. Raw per-record timing files are intentionally NOT saved (storage constraint).

| method | mean ms/record | p50 | p95 | throughput (rec/s) |
|---|---:|---:|---:|---:|
| rc_edgedistill_lambda_0.25 | 1.526 | 1.453 | 2.038 | 655.2 |
| tfidf_logistic_balanced | 4.304 | 4.127 | 5.654 | 232.3 |
| tfidf_linear_svc_balanced | 2.405 | 2.309 | 3.661 | 415.8 |
| tfidf_sgd_modified_huber | 2.213 | 2.142 | 2.790 | 451.9 |
| tfidf_ridge_balanced | 2.237 | 2.151 | 2.776 | 447.0 |
| tfidf_complement_nb | 2.317 | 2.255 | 2.954 | 431.6 |
