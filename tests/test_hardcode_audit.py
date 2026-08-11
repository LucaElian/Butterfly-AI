from butterfly.audit import audit_hardcodes


def test_operational_code_has_no_brain_or_suite_version_hardcodes():
    findings = audit_hardcodes()
    details = "\n".join(
        f"{item['file']}:{item['line']} [{item['kind']}] {item['text']}"
        for item in findings
    )
    assert not findings, "Operational hardcodes found:\n" + details
