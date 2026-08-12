from butterfly.audit import PATTERNS, audit_hardcodes


def test_operational_code_has_no_brain_or_suite_version_hardcodes():
    findings = audit_hardcodes()
    details = "\n".join(
        f"{item['file']}:{item['line']} [{item['kind']}] {item['text']}"
        for item in findings
    )
    assert not findings, "Operational hardcodes found:\n" + details


def test_audit_detects_versioned_butterfly_user_agent():
    sample = 'USER_AGENT = "ButterflyAI/' + "0." + "0004" + ' educational local research agent"'
    matches = [label for label, pattern in PATTERNS if pattern.search(sample)]
    assert "ButterflyAI user-agent version literal" in matches
