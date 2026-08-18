from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "SETUP_LINUX.sh",
    "RUN_AUTONOMY.sh",
    "RUN_AUTONOMY_LOOP.sh",
    "STOP_AUTONOMY.sh",
    "TEACHER_LESSONS.sh",
    "ADD_RESEARCH_SEED.sh",
    "AUTONOMY_STATUS.sh",
    "INSTALL_AUTONOMY_SERVICE.sh",
]


def test_linux_entrypoints_exist_and_use_lf():
    for name in SCRIPTS:
        data = (ROOT / name).read_bytes()
        assert data.startswith(b"#!/usr/bin/env bash\n"), name
        assert b"\r\n" not in data, name


def test_linux_cloud_service_uses_loop_not_raw_autonomy():
    service_installer = (ROOT / "INSTALL_AUTONOMY_SERVICE.sh").read_text(encoding="utf-8")
    loop = (ROOT / "RUN_AUTONOMY_LOOP.sh").read_text(encoding="utf-8")

    assert "RUN_AUTONOMY_LOOP.sh" in service_installer
    assert "BUTTERFLY_AUTONOMY_CONSOLE=compact" in service_installer
    assert "STOP_AUTONOMY" in loop
    assert "TEACHER_LESSONS.sh" in loop