import hashlib
import json

from butterfly.config import ROOT
from butterfly.corpus.focus_packets import answer_for_case, build_focus_packets
from butterfly.learning.dynamic_exam import generate_bank, normalize_surface
from butterfly.learning.evaluator import benchmark_surface_prompts
from butterfly.learning.study_exam import study_surface_prompts


def test_focus_packets_generate_train_rows_for_dynamic_family():
    rows = build_focus_packets("file", 12345, count=12)
    assert len(rows) == 12
    assert {row["source"] for row in rows} == {"focus_packet"}
    assert all(row["skill"] == "dynamic_packet:file" for row in rows)
    assert all("archivo" in row["assistant"].casefold() for row in rows)


def test_focus_packets_are_disjoint_from_fixed_held_out_surfaces():
    rows = build_focus_packets("folder", 7788, count=16)
    held_out = benchmark_surface_prompts()
    held_out |= {normalize_surface(prompt) for prompt in study_surface_prompts()}
    assert all(normalize_surface(row["user"]) not in held_out for row in rows)


def test_focus_packet_mode_is_disjoint_from_selection_transfer_acceptance():
    packet_rows = build_focus_packets("api", "seed-demo", count=16)
    packet_surfaces = {normalize_surface(row["user"]) for row in packet_rows}
    for mode in ("selection", "transfer", "acceptance"):
        bank = generate_bank("api", "seed-demo", count=16, mode=mode)
        bank_surfaces = {normalize_surface(row["prompt"]) for row in bank["cases"]}
        assert not (packet_surfaces & bank_surfaces)


def test_instruction_packet_answers_follow_dynamic_case_shape():
    cases = generate_bank("two_steps", "steps-demo", count=4, mode="learning_packet")["cases"]
    for case in cases:
        answer = answer_for_case(case)
        assert answer.startswith("1. ")
        assert "\n2. " in answer


def test_deliberate_manifest_records_focus_packet_rows():
    source = (ROOT / "butterfly" / "corpus" / "deliberate.py").read_text(encoding="utf-8")
    assert "build_focus_packets" in source
    assert "focus_packet_rows" in source


def test_deliberate_manifest_records_verified_experience_rows():
    source = (ROOT / "butterfly" / "corpus" / "deliberate.py").read_text(encoding="utf-8")
    assert "build_verified_experience_packets" in source
    assert "verified_experience_rows" in source

def test_deliberate_defers_verified_experience_consumption_until_acceptance():
    source = (ROOT / "butterfly" / "corpus" / "deliberate.py").read_text(encoding="utf-8")
    build_body = source.split("def build_corpus", 1)[1].split("def verified_experience_ids_for_experiment", 1)[0]
    assert "mark_verified_experiences_used" not in build_body
    assert "verified_experience_ids" in build_body