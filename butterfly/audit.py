from __future__ import annotations

import re

from .config import ROOT

EXCLUDED_PARTS = {
    ".git", ".venv", "benchmarks", "logs", "reports", "release",
    "data", "models", ".butterfly", "__pycache__", "_general_refactor_payload",
}
EXCLUDED_FILES = {"CHANGELOG.md", "README.md", "_general_refactor_apply.py", "INSTALL_GENERAL_REFACTOR.bat"}

PATTERNS = [
    ("brain/suite version literal", re.compile(r"\bv0\.\d{4,}\b", re.I)),
    ("quoted version-like literal", re.compile(r"""["']0\.\d{4,}["']""")),
    ("old compact version token", re.compile(r"\bv000\d+\b", re.I)),
    ("version-bound tokenizer constant", re.compile(r"\bV\d+_TOKENIZER_PATH\b")),
    ("literal benchmark suite assignment", re.compile(r"""BENCHMARK_SUITE_(?:VERSION|ID)\s*=\s*["']""")),
    ("absolute Butterfly path", re.compile(r"[A-Za-z]:\\[^\"'\r\n]*ButterflyAI", re.I)),
]


def iter_operational_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if path.name in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if path.suffix.lower() not in {".py", ".json", ".bat", ".toml"}:
            continue
        yield path


def audit_hardcodes():
    findings = []
    for path in iter_operational_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append({
                        "file": str(path.relative_to(ROOT)),
                        "line": line_no,
                        "kind": label,
                        "text": line.strip(),
                    })
    return findings


def print_audit():
    findings = audit_hardcodes()
    if not findings:
        print("Hardcode audit: PASS")
        print("No brain/suite version literals or absolute Butterfly paths found in operational code.")
        return 0
    print("Hardcode audit: FAIL")
    for item in findings:
        print(f"{item['file']}:{item['line']} [{item['kind']}] {item['text']}")
    return 1
