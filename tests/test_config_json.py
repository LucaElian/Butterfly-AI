from butterfly.config import load_json


def test_load_json_accepts_utf8_bom(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text('\ufeff{"ok": true}', encoding="utf-8")

    assert load_json(path) == {"ok": True}
