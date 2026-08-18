from __future__ import annotations
import sqlite3
import json
from datetime import datetime, timezone
from .config import DB_PATH, ensure_dirs


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, path=DB_PATH):
        ensure_dirs()
        self.path = path
        self._init()

    def connect(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self.connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                task TEXT NOT NULL,
                context TEXT,
                actions TEXT,
                result TEXT NOT NULL,
                lesson TEXT,
                verified INTEGER NOT NULL DEFAULT 0,
                quality REAL NOT NULL DEFAULT 0.0,
                used_for_training INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                claim TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                method TEXT,
                sources TEXT,
                evidence TEXT,
                last_verified_at TEXT
            );
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                rule TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 50,
                source TEXT
            );
            """)

    def add_experience(self, task, result, context="", actions=None, lesson="", verified=False, quality=0.0):
        with self.connect() as c:
            cur = c.execute(
                "INSERT INTO experiences(created_at,task,context,actions,result,lesson,verified,quality) VALUES(?,?,?,?,?,?,?,?)",
                (utcnow(), task, context, json.dumps(actions or [], ensure_ascii=False), result, lesson, int(verified), quality),
            )
            return int(cur.lastrowid)

    def add_claim(self, claim, status, confidence, method="", sources=None, evidence=""):
        with self.connect() as c:
            c.execute(
                "INSERT INTO claims(created_at,claim,status,confidence,method,sources,evidence,last_verified_at) VALUES(?,?,?,?,?,?,?,?)",
                (utcnow(), claim, status, confidence, method, json.dumps(sources or [], ensure_ascii=False), evidence, utcnow()),
            )

    def approved_experiences(self, limit=5000, minimum_quality=0.7):
        with self.connect() as c:
            rows = c.execute(
                "SELECT id,task,context,actions,result,lesson,quality FROM experiences WHERE verified=1 AND quality>=? AND used_for_training=0 ORDER BY id LIMIT ?",
                (float(minimum_quality), limit),
            ).fetchall()
        return rows

    def mark_used(self, ids):
        if not ids:
            return
        with self.connect() as c:
            c.executemany("UPDATE experiences SET used_for_training=1 WHERE id=?", [(i,) for i in ids])
