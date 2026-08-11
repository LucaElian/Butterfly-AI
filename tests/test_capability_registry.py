from butterfly.runtime import load_capabilities


def test_deterministic_capabilities_are_not_trainable():
    caps = load_capabilities()["capabilities"]
    assert caps["exact_binding"]["implementation"] == "deterministic_runtime"
    assert caps["exact_binding"]["trainable"] is False
    assert caps["arithmetic"]["implementation"] == "deterministic_runtime"
    assert caps["arithmetic"]["trainable"] is False


def test_instruction_format_remains_trainable():
    caps = load_capabilities()["capabilities"]
    assert caps["instruction_format"]["implementation"] == "neural"
    assert caps["instruction_format"]["trainable"] is True
