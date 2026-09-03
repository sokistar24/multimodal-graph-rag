from multimodal_graph_rag.evaluation.integrity import audit_answer_recoverability


def test_numeric_equivalence_flags_formatted_values():
    audit = audit_answer_recoverability(
        "1,234.50", "The table reports 1234.50 participants."
    )
    assert audit.numeric_equivalent is True
    assert audit.needs_human_review is True


def test_unflagged_result_is_not_called_verified():
    audit = audit_answer_recoverability(
        "Macrophage", "The caption describes immune cells."
    )
    assert audit.status == "screened_unflagged"
    assert audit.needs_human_review is False
