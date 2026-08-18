from butterfly.learning.research_seed import (
    CURATED_RESEARCH_SEED,
    install_curated_research_seed,
)
from butterfly.memory import MemoryStore


def test_curated_research_seed_has_provenance_and_curriculum_targets():
    assert len(CURATED_RESEARCH_SEED) >= 8
    for row in CURATED_RESEARCH_SEED:
        assert row["curriculum_node"]
        assert row["dynamic_family"]
        assert row["source_url"].startswith("https://")
        assert row["source_title"]
        assert row["lesson"]


def test_research_seed_installs_verified_experiences_once(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    marked = []
    first = install_curated_research_seed(
        store=store,
        nodes=["programming.api_basics"],
        mark_material_func=lambda node, status: marked.append((node, status)),
    )
    second = install_curated_research_seed(
        store=store,
        nodes=["programming.api_basics"],
        mark_material_func=lambda node, status: marked.append((node, status)),
    )

    assert len(first["added"]) == 2
    assert len(second["added"]) == 0
    assert len(second["skipped"]) == 2
    assert marked.count(("programming.api_basics", "verified_packet")) >= 2

    rows = store.approved_experiences(limit=10, minimum_quality=0.9)
    assert len(rows) == 2
    assert all(row[5] for row in rows)


def test_research_seed_dry_run_does_not_write(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    report = install_curated_research_seed(
        store=store,
        limit=1,
        dry_run=True,
        mark_material_func=lambda *_: (_ for _ in ()).throw(AssertionError("should not mark")),
    )
    assert len(report["added"]) == 1
    assert store.approved_experiences(limit=10) == []