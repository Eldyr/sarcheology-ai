"""Local, append-only expert feedback. Survey references live outside this store."""
from contextlib import closing
from datetime import datetime,timezone
from itertools import combinations
import sqlite3

import pandas as pd

DECISIONS = ("plausible","rejected","needs_investigation")
SCHEMA = """CREATE TABLE IF NOT EXISTS feedback (
    event_id INTEGER PRIMARY KEY, catalogue_sha256 TEXT NOT NULL, candidate_id TEXT NOT NULL,
    reviewer TEXT NOT NULL, revision INTEGER NOT NULL, decision TEXT NOT NULL
    CHECK(decision IN ('plausible','rejected','needs_investigation')),
    note TEXT NOT NULL, created_utc TEXT NOT NULL,
    UNIQUE(catalogue_sha256,candidate_id,reviewer,revision))"""


def initialize(db):
    db.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def save_feedback(db,run,candidate,reviewer,decision,note,expected_revision,allowed_ids):
    reviewer = reviewer.strip()
    if not reviewer or len(reviewer)>100 or candidate not in allowed_ids or decision not in DECISIONS:
        raise ValueError("Valid reviewer, catalogue candidate and decision required")
    initialize(db)
    with closing(sqlite3.connect(db,timeout=15)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT COALESCE(MAX(revision),0) FROM feedback WHERE catalogue_sha256=? "
                               "AND candidate_id=? AND reviewer=?",(run,candidate,reviewer)).fetchone()[0]
        if current!=expected_revision:
            raise ValueError("This review changed in another session. Reload before saving.")
        conn.execute("INSERT INTO feedback(catalogue_sha256,candidate_id,reviewer,revision,decision,note,created_utc) "
                     "VALUES(?,?,?,?,?,?,?)",(run,candidate,reviewer,current+1,decision,note.strip(),
                                               datetime.now(timezone.utc).isoformat()))
        conn.commit()


def feedback(db,run):
    initialize(db)
    with closing(sqlite3.connect(db)) as conn:
        history = pd.read_sql_query("SELECT * FROM feedback WHERE catalogue_sha256=? ORDER BY event_id",conn,params=(run,))
    latest = history.drop_duplicates(["candidate_id","reviewer"],keep="last")
    agreements = []
    for (a,b) in combinations(sorted(latest.reviewer.unique()),2):
        first = latest[latest.reviewer==a].set_index("candidate_id").decision
        second = latest[latest.reviewer==b].set_index("candidate_id").decision
        common = first.index.intersection(second.index)
        if common.empty:
            continue
        x,y = first.loc[common],second.loc[common]
        actual = float((x==y).mean())
        expected = sum((x==k).mean()*(y==k).mean() for k in DECISIONS)
        agreements.append(dict(reviewer_a=a,reviewer_b=b,shared=len(common),agreement=actual,
                               kappa=(actual-expected)/(1-expected) if expected<1 else None,
                               disagreements=list(common[x!=y])))
    return history,latest,agreements
