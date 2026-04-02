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
    "plot_evaluation_results",
]


def __getattr__(name):
    """
    @brief Lazy load evaluation functions on first access.
    @param name: Name of the attribute being accessed.
    @return The requested attribute from evaluate module.
    """
    if name in __all__:
        from uncertainty_rl.evaluation.evaluate import (  # noqa: E402
            EvaluationMetrics,
            evaluate_across_conditions,
            evaluate_agent,
            plot_evaluation_results,
        )

        attrs = {
            "EvaluationMetrics": EvaluationMetrics,
            "evaluate_agent": evaluate_agent,
            "evaluate_across_conditions": evaluate_across_conditions,
            "plot_evaluation_results": plot_evaluation_results,
        }
        return attrs[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
