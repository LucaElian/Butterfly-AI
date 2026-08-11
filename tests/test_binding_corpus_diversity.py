from butterfly.corpus.skills.binding import build

def test_binding_target_split_is_disjoint():
    train, valid = build(99881)
    assert {r["assistant"] for r in train}.isdisjoint({r["assistant"] for r in valid})

def test_binding_has_multiple_code_shapes():
    train, _ = build(99881)
    values = {r["assistant"] for r in train}
    assert any(len(x) == 3 and x[:2].isalpha() and x[2].isdigit() for x in values)
    assert any(len(x) == 4 and x[0].isalpha() and x[1].isdigit() and x[2].isalpha() and x[3].isdigit() for x in values)
    assert any(len(x) == 4 and x[0].isalpha() and x[1:3].isdigit() and x[3].isalpha() for x in values)
