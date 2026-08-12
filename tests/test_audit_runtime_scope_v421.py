from butterfly import audit


def test_hardcode_audit_excludes_runtime_training_state(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    runtime = tmp_path / "training_state" / "deliberate"
    runtime.mkdir(parents=True)
    version_literal = "0." + "12345"
    (runtime / "progress.json").write_text(
        '{"target_version": "' + version_literal + '"}',
        encoding="utf-8",
    )

    assert audit.audit_hardcodes() == []


def test_hardcode_audit_still_catches_version_literal_in_operational_code(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    version_literal = "0." + "12345"
    code = tmp_path / "example.py"
    code.write_text('VALUE = "' + version_literal + '"\n', encoding="utf-8")

    findings = audit.audit_hardcodes()
    assert len(findings) >= 1
    assert findings[0]["file"] == "example.py"
