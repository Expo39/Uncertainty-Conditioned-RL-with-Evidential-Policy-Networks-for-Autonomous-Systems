"""
@file __init__.py
@brief Evaluation analysis tooling (run via make targets, not imported by the pipeline).

Three tools that turn an evaluation run into the dissertation's input-covariance
claims:

  - covariance_probe.py: in-container (torch) causal probe - does the trained
    policy mechanically respond to the EKF covariance input?
  - ablation_analyser.py: host-side cross-arm contrast of the four ablation arms
    from their episode_records.csv (covariance deltas + degradation slope).
  - gate_roc.py: host-side EKF-std vs evidential-epistemic safety-gate ROC.

@author Antonio Galdes
"""
