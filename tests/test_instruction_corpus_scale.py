from butterfly.corpus.skills.instruction_following import build


def test_instruction_corpus_has_real_scale():
    train, valid = build(123456)
    assert len(train) >= 700
    assert len(valid) >= 100


def test_instruction_train_valid_families_are_disjoint():
    train, valid = build(123456)
    train_families = {row["family"] for row in train}
    valid_families = {row["family"] for row in valid}
    assert train_families.isdisjoint(valid_families)


def test_instruction_train_valid_prompt_surfaces_are_disjoint():
    train, valid = build(123456)
    train_prompts = {row["user"].casefold().strip() for row in train}
    valid_prompts = {row["user"].casefold().strip() for row in valid}
    assert train_prompts.isdisjoint(valid_prompts)
