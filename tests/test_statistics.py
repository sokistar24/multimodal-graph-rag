from multimodal_graph_rag.evaluation.statistics import (
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_delta,
)


def test_paired_bootstrap_reports_observed_delta():
    estimate = paired_bootstrap_delta(
        [0, 0, 1, 1], [0, 1, 1, 1], iterations=500, seed=3
    )
    assert estimate.delta == 0.25
    assert estimate.lower <= estimate.delta <= estimate.upper


def test_mcnemar_uses_only_discordant_pairs():
    result = mcnemar_exact([1, 0, 0, 1], [0, 1, 1, 1])
    assert result["baseline_only"] == 1
    assert result["treatment_only"] == 2
    assert result["p_value"] == 1.0


def test_holm_adjust_is_monotone_in_sorted_order():
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
