from butterfly.config import ROOT


def test_evaluation_path_writers_use_portable_helper():
    for path in (
        ROOT / "butterfly" / "upgrade.py",
        ROOT / "butterfly" / "deliberate.py",
        ROOT / "butterfly" / "cli.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "project_relpath" in source
        assert "str(report_path.relative_to(ROOT))" not in source
