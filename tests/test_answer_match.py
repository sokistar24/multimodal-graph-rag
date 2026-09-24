from multimodal_graph_rag.evaluation.answer_match import (
    contains_reference,
    exact_match,
    normalise_answer,
    token_f1,
)


def test_normalisation_ignores_case_articles_and_punctuation():
    assert normalise_answer("The Conversation.") == "conversation"
    assert normalise_answer("  35,124  ") == "35124"


def test_exact_match_is_insensitive_to_wrapping_but_not_to_content():
    assert exact_match("The Conversation", "the conversation.") == 1
    assert exact_match("The Guardian", "The Conversation") == 0


def test_token_f1_gives_partial_credit_for_a_sentence_answer():
    score = token_f1(
        "The answer is Colonel Charles Hastings Judd.",
        "Colonel Charles Hastings Judd",
    )
    assert 0.6 < score < 1.0
    assert (
        token_f1("Colonel Charles Hastings Judd", "Colonel Charles Hastings Judd")
        == 1.0
    )


def test_token_f1_of_disjoint_answers_is_zero():
    assert token_f1("yes", "no") == 0.0


def test_empty_answers_match_only_each_other():
    assert token_f1("", "") == 1.0
    assert token_f1("", "yes") == 0.0


def test_contains_reference_is_lenient_where_exact_match_is_not():
    prediction = "Based on the context, the answer is 35,124 households."
    assert exact_match(prediction, "35,124") == 0
    assert contains_reference(prediction, "35,124") == 1
    assert contains_reference(prediction, "41,000") == 0
