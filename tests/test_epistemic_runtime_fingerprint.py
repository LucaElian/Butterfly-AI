from butterfly.runtime import runtime_fingerprint_payload

def test_epistemic_routes_participate_in_suite_fingerprint():
    payload = runtime_fingerprint_payload()
    assert "parse_arithmetic_truth_claim" in payload
    assert "parse_explicit_unknown" in payload
