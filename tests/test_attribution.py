from multimodal_graph_rag.evaluation.attribution import (
    EvidenceState,
    classify_evidence_state,
)


def test_correct_answer_with_non_gold_support_is_not_called_parametric_memory():
    state = classify_evidence_state(
        evidence_available=True,
        gold_found=1,
        gold_required=2,
        answer_correct=True,
        non_gold_support=True,
    )
    assert state is EvidenceState.NON_GOLD_SUPPORTED


def test_complete_context_wrong_answer_is_evidence_use_failure():
    state = classify_evidence_state(
        evidence_available=True,
        gold_found=2,
        gold_required=2,
        answer_correct=False,
    )
    assert state is EvidenceState.EVIDENCE_USE_FAILURE
