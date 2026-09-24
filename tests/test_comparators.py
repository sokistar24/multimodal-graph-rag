"""Published-comparator adapter: precomputed retrievals served as passages."""

from types import SimpleNamespace

import pytest

from multimodal_graph_rag.corpora.loaders import iter_corpus_chunks
from multimodal_graph_rag.errors import ConfigurationError, MissingEvidenceError
from multimodal_graph_rag.retrieval.comparators import (
    PrecomputedRetrievals,
    comparator_report,
    precompute_hipporag,
    write_retrievals,
)
from multimodal_graph_rag.retrieval.hipporag_index import main
from multimodal_graph_rag.schemas import load_questions


def _write(tmp_path, corpus_dir, question_file):
    questions = load_questions(question_file)
    retrievals = {
        questions[0].id: [
            {"source": "alpha.txt", "text": "alpha chunk", "score": 0.9},
            {"source": "gamma.txt", "text": "gamma chunk", "score": 0.4},
        ],
        questions[1].id: [
            {"source": "beta.txt", "text": "beta chunk", "score": 0.8},
            {"source": "alpha.txt", "text": "alpha chunk", "score": 0.7},
            {"source": "delta.txt", "text": "delta chunk", "score": 0.1},
        ],
        questions[2].id: [
            {"source": "delta.txt", "text": "delta chunk", "score": 0.5},
        ],
    }
    path = tmp_path / "hipporag_retrievals.json"
    write_retrievals(
        path,
        retrievals=retrievals,
        corpus_dir=corpus_dir,
        question_file=question_file,
        comparator="hipporag2",
        num_to_retrieve=3,
        metadata={"package_version": "2.0.0"},
    )
    return path, questions


def test_precomputed_retrievals_round_trip_at_the_candidate_budget(
    tmp_path, corpus_dir, question_file
):
    path, questions = _write(tmp_path, corpus_dir, question_file)
    loaded = PrecomputedRetrievals.load(path, corpus_dir=corpus_dir)
    passages = loaded.retrieve(questions[1].id, k=2)
    assert [p.source for p in passages] == ["beta.txt", "alpha.txt"]
    assert [p.rank for p in passages] == [1, 2]
    assert passages[0].score == pytest.approx(0.8)
    assert passages[0].text == "beta chunk"
    assert loaded.comparator == "hipporag2"
    assert loaded.metadata["package_version"] == "2.0.0"


def test_precomputed_retrievals_refuse_a_different_corpus(
    tmp_path, corpus_dir, question_file
):
    path, _ = _write(tmp_path, corpus_dir, question_file)
    (corpus_dir / "iota.txt").write_text("A new document changes the corpus.")
    with pytest.raises(ConfigurationError, match="fingerprint"):
        PrecomputedRetrievals.load(path, corpus_dir=corpus_dir)


def test_unknown_question_is_missing_evidence(tmp_path, corpus_dir, question_file):
    path, _ = _write(tmp_path, corpus_dir, question_file)
    loaded = PrecomputedRetrievals.load(path)
    with pytest.raises(MissingEvidenceError, match="no precomputed"):
        loaded.retrieve("not-a-question", k=3)


class _StubHippoRAG:
    """Mimics the hipporag.HippoRAG surface the adapter relies on."""

    def __init__(self, answers):
        self.answers = answers
        self.indexed = None

    def index(self, docs):
        self.indexed = list(docs)

    def retrieve(self, queries, num_to_retrieve):
        return [
            SimpleNamespace(
                question=q,
                docs=self.answers[q][:num_to_retrieve],
                doc_scores=[1.0 / (i + 1) for i in range(len(self.answers[q]))][
                    :num_to_retrieve
                ],
            )
            for q in queries
        ]

    def get_graph_info(self):
        return {"num_phrase_nodes": 3}


def test_precompute_maps_returned_chunk_texts_back_to_sources(
    tmp_path, corpus_dir, question_file
):
    questions = load_questions(question_file)
    chunks = list(iter_corpus_chunks(corpus_dir))
    text_of = {c.source: c.text for c in chunks}
    engine = _StubHippoRAG(
        {
            questions[0].question: [text_of["alpha.txt"], text_of["gamma.txt"]],
            questions[1].question: [text_of["beta.txt"], text_of["alpha.txt"]],
            questions[2].question: [text_of["delta.txt"]],
        }
    )
    out = tmp_path / "out.json"
    precompute_hipporag(
        engine,
        corpus_dir=corpus_dir,
        question_file=question_file,
        output=out,
        num_to_retrieve=2,
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        package_version="2.0.0",
    )
    assert engine.indexed == [c.text for c in chunks]
    loaded = PrecomputedRetrievals.load(out, corpus_dir=corpus_dir)
    assert [p.source for p in loaded.retrieve(questions[0].id, k=2)] == [
        "alpha.txt",
        "gamma.txt",
    ]
    assert loaded.metadata["graph_info"] == {"num_phrase_nodes": 3}
    assert loaded.metadata["llm_model"] == "gpt-4o-mini"
    assert loaded.metadata["embedding_model"] == "text-embedding-3-small"


def test_precompute_refuses_a_text_it_cannot_attribute(
    tmp_path, corpus_dir, question_file
):
    questions = load_questions(question_file)
    engine = _StubHippoRAG({q.question: ["not a corpus chunk"] for q in questions})
    with pytest.raises(MissingEvidenceError, match="attribute"):
        precompute_hipporag(
            engine,
            corpus_dir=corpus_dir,
            question_file=question_file,
            output=tmp_path / "out.json",
            num_to_retrieve=1,
            llm_model="m",
            embedding_model="e",
        )


def test_comparator_report_counts_completeness_at_k(
    tmp_path, corpus_dir, question_file
):
    path, questions = _write(tmp_path, corpus_dir, question_file)
    loaded = PrecomputedRetrievals.load(path)
    report = comparator_report(questions, loaded, k=2)
    # q0 gold alpha: hit; q1 gold alpha+beta: both in top-2; q2 gold gamma: missed.
    assert report["summary"] == {"questions": 3, "complete": 2, "k": 2}
    assert report["outcomes"][2]["missing_gold"] == ["gamma.txt"]


def test_index_entry_point_builds_engine_from_arguments_and_writes_file(
    tmp_path, corpus_dir, question_file
):
    questions = load_questions(question_file)
    chunks = list(iter_corpus_chunks(corpus_dir))
    text_of = {c.source: c.text for c in chunks}
    seen = {}

    def factory(*, save_dir, llm_model_name, embedding_model_name):
        seen.update(
            save_dir=save_dir,
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
        )
        return _StubHippoRAG(
            {q.question: [text_of[q.gold_sources[0]]] for q in questions}
        )

    out = tmp_path / "retrievals.json"
    code = main(
        [
            "--corpus-dir",
            str(corpus_dir),
            "--questions",
            str(question_file),
            "--output",
            str(out),
            "--save-dir",
            str(tmp_path / "index"),
            "--llm",
            "gpt-4o-mini",
            "--embedding",
            "text-embedding-3-small",
            "--num-to-retrieve",
            "1",
        ],
        engine_factory=factory,
    )
    assert code == 0
    assert seen == {
        "save_dir": str(tmp_path / "index"),
        "llm_model_name": "gpt-4o-mini",
        "embedding_model_name": "text-embedding-3-small",
    }
    loaded = PrecomputedRetrievals.load(out, corpus_dir=corpus_dir)
    assert loaded.num_to_retrieve == 1
    assert loaded.metadata["llm_model"] == "gpt-4o-mini"
