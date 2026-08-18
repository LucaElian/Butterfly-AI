import json

from butterfly.corpus.verified_experiences import build_verified_experience_rows
from butterfly.learning.dynamic_exam import normalize_surface
from butterfly.learning.evaluator import benchmark_surface_prompts
from butterfly.learning.study_exam import study_surface_prompts


def _experience(**overrides):
    row = {
        "id": 7,
        "task": "Aprender que una API define como dos programas intercambian datos.",
        "context": json.dumps({
            "curriculum_node": "programming.api_basics",
            "dynamic_family": "api",
            "source_url": "https://developer.mozilla.org/en-US/docs/Glossary/API",
            "context": "Fuente oficial o documentacion tecnica revisada.",
        }),
        "actions": "[]",
        "result": "API significa Application Programming Interface.",
        "lesson": "Una API define reglas para que programas pidan datos o funciones de forma documentada.",
        "quality": 0.95,
    }
    row.update(overrides)
    return row


def test_verified_experience_rows_match_curriculum_node_and_keep_provenance():
    rows, ids = build_verified_experience_rows(
        [_experience()],
        {
            "curriculum_node": "programming.api_basics",
            "dynamic_family": "api",
            "corpus_aliases": ["api"],
        },
        limit=8,
    )
    assert ids == [7]
    assert len(rows) == 1
    assert rows[0]["source"] == "verified_experience"
    assert rows[0]["skill"] == "verified_experience:api"
    assert rows[0]["provenance"]["experience_id"] == 7
    assert rows[0]["provenance"]["source"].startswith("https://")


def test_verified_experience_rows_ignore_unrelated_material():
    rows, ids = build_verified_experience_rows(
        [_experience()],
        {
            "curriculum_node": "conversation.thanks",
            "dynamic_family": "thanks",
            "corpus_aliases": ["thanks"],
        },
        limit=8,
    )
    assert rows == []
    assert ids == []


def test_verified_experience_rows_are_disjoint_from_held_out_prompts():
    rows, _ = build_verified_experience_rows(
        [_experience()],
        {
            "curriculum_node": "programming.api_basics",
            "dynamic_family": "api",
            "corpus_aliases": ["api"],
        },
        limit=8,
    )
    held_out = benchmark_surface_prompts()
    held_out |= {normalize_surface(prompt) for prompt in study_surface_prompts()}
    assert rows
    assert all(normalize_surface(row["user"]) not in held_out for row in rows)

def test_verified_experience_packets_accept_minimum_quality_argument(monkeypatch):
    import butterfly.corpus.verified_experiences as vx

    class Store:
        def approved_experiences(self, limit=5000, minimum_quality=0.7):
            assert minimum_quality == 0.9
            return []

    monkeypatch.setattr(vx, "MemoryStore", lambda: Store())
    rows, ids = vx.build_verified_experience_packets(
        {"curriculum_node": "programming.api_basics", "dynamic_family": "api"},
        minimum_quality=0.9,
    )
    assert rows == []
    assert ids == []