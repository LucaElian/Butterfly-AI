from butterfly.config import ROOT


def test_corpus_manifest_writes_portable_relative_paths():
    source = (ROOT / "butterfly" / "corpus" / "deliberate.py").read_text(encoding="utf-8")
    assert "project_relpath(train_path)" in source
    assert "project_relpath(valid_path)" in source
    assert "str(train_path.relative_to(ROOT))" not in source
    assert "str(valid_path.relative_to(ROOT))" not in source


def test_corpus_filters_internal_study_exam_surfaces():
    source = (ROOT / "butterfly" / "corpus" / "deliberate.py").read_text(encoding="utf-8")
    assert "study_surface_prompts" in source
