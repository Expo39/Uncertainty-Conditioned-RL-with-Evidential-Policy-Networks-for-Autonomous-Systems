"""
@file __init__.py
@brief Evaluation across uncertainty levels.
"""

# Lazy imports: evaluate.py requires torch which may not be installed in CI
# environments (dependencies are optional). These are only imported when
# actually used (at runtime), not at package load time.

__all__ = [
    "EvaluationMetrics",
    "evaluate_agent",
    "evaluate_across_conditions",
]


def __getattr__(name):
    """
    @brief Lazy load evaluation symbols on first access.
    @param name: Name of the attribute being accessed.
    @return The requested attribute from its defining module.

    EvaluationMetrics has no torch / stable-baselines3 dependency, so it loads
    from its dedicated module; the two evaluation-loop entry points pull in
    those optional deps via evaluate. Everything stays lazy so importing the
    package never forces them in CI.
    """
    if name == "EvaluationMetrics":
        from uncertainty_rl.evaluation.metrics import EvaluationMetrics

        return EvaluationMetrics
    if name in ("evaluate_agent", "evaluate_across_conditions"):
        from uncertainty_rl.evaluation.evaluate import (  # noqa: E402
            evaluate_across_conditions,
            evaluate_agent,
        )

        return {
            "evaluate_agent": evaluate_agent,
            "evaluate_across_conditions": evaluate_across_conditions,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
