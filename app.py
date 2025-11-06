# streamlit_pbi_table_migration_planner.py
# -------------------------------------------------
# Plan and track moving tables from an OLD SQL Server database
# to a NEW SQL Server database, prioritized by Power BI reports.
#
# What it does
#  - Capture dataset/report, priority, and the exact table mapping
#    (old server/db/schema.table  ->  new server/db/schema.table)
#  - One-click **T‑SQL script generation** per row (safe templates):
#      • CREATE SCHEMA (if not exists)
#      • Create empty table (like source) via SELECT TOP(0)
#      • INSERT...SELECT in batches (optional)
#    (keeps constraints & indexes as TODO notes, since those need DBA review)
#  - Durable storage in SQLite under ~/Documents/PBI_MigrationPlanner + CSV mirror
#  - Filter, edit status/notes, and export filtered views
#
# How to run
#   pip install streamlit pandas sqlalchemy pydantic filelock
#   streamlit run streamlit_pbi_table_migration_planner.py

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from filelock import FileLock
import uuid
import tempfile
import shutil

## lets make it secure 

import os
import streamlit as st

# -------- Simple password gate (single shared password) --------
def _get_app_password() -> str:
    pw ="lesmillsreport"
    return pw

def password_gate():
    # Already unlocked?
    if st.session_state.get("authed", False):
        return True

    # Basic rate-limit: 5 tries, then 60s cool-down
    st.session_state.setdefault("tries", 0)
    st.session_state.setdefault("lock_until", 0)

    import time
    now = int(time.time())
    if now < st.session_state["lock_until"]:
        wait = st.session_state["lock_until"] - now
        st.error(f"Too many attempts. Try again in {wait}s.")
        st.stop()

    st.title("🔐 Sign in")
    pwd = st.text_input("Password", type="password", key="__app_pwd")
    submitted = st.button("Unlock", type="primary")

    if submitted:
        if not _get_app_password():
            st.error("APP_PASSWORD is not configured in secrets or environment.")
            st.stop()

        if pwd == _get_app_password():
            st.session_state["authed"] = True
            st.session_state["tries"] = 0
            st.rerun()
        else:
            st.session_state["tries"] += 1
            if st.session_state["tries"] >= 5:
                st.session_state["lock_until"] = now + 60  # 60s cool-down
                st.error("Incorrect password. Locked for 60 seconds.")
            else:
                left = 5 - st.session_state["tries"]
                st.error(f"Incorrect password. {left} attempt(s) left.")
    st.stop()

password_gate()  # <-- Gate everything below this line

# Optional: add a logout button in the sidebar (after gate)
with st.sidebar:
    if st.session_state.get("authed"):
        if st.button("Logout"):
            for k in ("authed", "tries", "lock_until", "__app_pwd"):
                st.session_state.pop(k, None)
            st.rerun()
# ---------------------------------------------------------------








APP_NAME = "Database Tables MigrationPlanner"
BASE_DIR = Path.home() / "Documents" / APP_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "planner.sqlite3"
CSV_PATH = BASE_DIR / "entries.csv"
LOCK_PATH = str(BASE_DIR / "entries.csv.lock")

STATUS_OPTIONS = [
    "Planned",
    "In Progress",
    "Validated on NEW",
    "Switched in Power BI",
    "Done"
]

PRIORITY_OPTIONS = ["P0 – Critical", "P1 – High", "P2 – Medium", "P3 – Low"]

MIGRATION_METHODS = [
    "Create+Insert (safe)",
    "SELECT INTO (quick copy)",
]

# ---------------- DB helpers ----------------

def get_engine() -> Engine:
    return create_engine(f"sqlite:///{DB_PATH.as_posix()}", future=True)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entries (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  priority TEXT,
  dataset_report TEXT,
  status TEXT,
  old_server TEXT, old_database TEXT, old_schema TEXT, old_table TEXT,
  new_server TEXT, new_database TEXT, new_schema TEXT, new_table TEXT,
  migration_method TEXT,
  comment_sql TEXT
);
"""

def ensure_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_SQL)


def insert_entry(engine: Engine, row: dict) -> None:
    row = row.copy()
    now = datetime.utcnow().isoformat(timespec="seconds")
    row.setdefault("id", str(uuid.uuid4()))
    row.setdefault("created_at", now)
    row["updated_at"] = now
    sql = text(
        """
        INSERT INTO entries (
          id, created_at, updated_at, priority, dataset_report, status,
          old_server, old_database, old_schema, old_table,
          new_server, new_database, new_schema, new_table,
          migration_method, comment_sql
        ) VALUES (
          :id, :created_at, :updated_at, :priority, :dataset_report, :status,
          :old_server, :old_database, :old_schema, :old_table,
          :new_server, :new_database, :new_schema, :new_table,
          :migration_method, :comment_sql
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, row)


def update_entry(engine: Engine, row: dict) -> None:
    row = row.copy()
    row["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    sql = text(
        """
        UPDATE entries SET
          updated_at=:updated_at,
          priority=:priority,
          dataset_report=:dataset_report,
          status=:status,
          old_server=:old_server, old_database=:old_database, old_schema=:old_schema, old_table=:old_table,
          new_server=:new_server, new_database=:new_database, new_schema=:new_schema, new_table=:new_table,
          migration_method=:migration_method,
          comment_sql=:comment_sql
        WHERE id=:id
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, row)


def fetch_df(engine: Engine) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql_query("SELECT * FROM entries ORDER BY created_at DESC", conn)

# --------------- CSV export (durable) ---------------

def atomic_write_csv(df: pd.DataFrame, csv_path: Path, lock_path: str) -> None:
    lock = FileLock(lock_path)
    with lock:
        tmp_dir = tempfile.mkdtemp()
        tmp_file = Path(tmp_dir) / "entries.tmp.csv"
        df.to_csv(tmp_file, index=False)
        shutil.move(str(tmp_file), str(csv_path))
        shutil.rmtree(tmp_dir, ignore_errors=True)

# --------------- Script generator ---------------

def fqdn(server: str, db: str, schema: str, table: str) -> str:
    return f"[{server}].[{db}].[{schema}].[{table}]"


def src_ref(db: str, schema: str, table: str) -> str:
    # For USE ... then [schema].[table]
    return f"[{schema}].[{table}]"


def make_tsql_create_insert(old_server, old_db, old_schema, old_table,
                            new_server, new_db, new_schema, new_table) -> str:
    # Template uses two connections (run on NEW): USE new_db; then 3 steps
    return f"""
-- Run ON NEW server: {new_server}
-- 1) Ensure schema exists
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'{new_schema}')
    EXEC('CREATE SCHEMA [{new_schema}]');
GO

-- 2) Create empty table with source shape
--    You may need a linked server or run this from a machine that can query OLD.
--    Replace OPENQUERY or three-part naming with your connectivity approach.
DROP TABLE IF EXISTS [{new_schema}].[{new_table}];
SELECT TOP(0) *
INTO   [{new_schema}].[{new_table}]
FROM   OPENQUERY([{old_server}], 'SELECT * FROM [{old_db}].[{old_schema}].[{old_table}]');

-- TODO: Recreate PKs, indexes, constraints, identities, defaults, collation, computed columns.
--       Consider scripting from SSMS or using DacPac for full fidelity.

-- 3) Insert data (optionally in batches)
DECLARE @batch BIGINT = 100000; -- tune for size
DECLARE @rowcount BIGINT;
SET NOCOUNT ON;

-- Simple one-shot copy (replace if table is very large):
INSERT INTO [{new_schema}].[{new_table}]
SELECT *
FROM OPENQUERY([{old_server}], 'SELECT * FROM [{old_db}].[{old_schema}].[{old_table}]');

SELECT @@ROWCOUNT AS rows_copied;
""".strip()


def make_tsql_select_into(old_server, old_db, old_schema, old_table,
                          new_server, new_db, new_schema, new_table) -> str:
    return f"""
-- Run ON NEW server: {new_server}
-- Creates table and copies data in one step (fast) but does NOT bring indexes/constraints.
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'{new_schema}')
    EXEC('CREATE SCHEMA [{new_schema}]');
GO

DROP TABLE IF EXISTS [{new_schema}].[{new_table}];
SELECT *
INTO   [{new_schema}].[{new_table}]
FROM   OPENQUERY([{old_server}], 'SELECT * FROM [{old_db}].[{old_schema}].[{old_table}]');

-- TODO: Recreate PKs, indexes, constraints after copy.
""".strip()


def build_script(row: dict) -> str:
    m = row.get("migration_method", MIGRATION_METHODS[0])
    args = (
        row.get("old_server",""), row.get("old_database",""), row.get("old_schema","dbo"), row.get("old_table",""),
        row.get("new_server",""), row.get("new_database",""), row.get("new_schema","dbo"), row.get("new_table",""),
    )
    if m == "SELECT INTO (quick copy)":
        return make_tsql_select_into(*args)
    return make_tsql_create_insert(*args)

# ---------------- UI ----------------

st.set_page_config(page_title="Table Migration Planner", page_icon="📤", layout="wide")
st.title("📤 Table Migration Planner")
st.caption(f"Data folder: {BASE_DIR}")
engine = get_engine(); ensure_db(engine)

with st.expander("➕ Add table move", expanded=True):
    with st.form("new_row"):
        c0, c1, c2 = st.columns([1,2,1])
        priority = c0.selectbox("Priority", PRIORITY_OPTIONS, index=1)
        dataset_report = c1.text_input("Dataset/Report name *", placeholder="e.g., Customer (Service)")
        status = c2.selectbox("Status", STATUS_OPTIONS, index=0)

        st.markdown("**Old (source)**")
        o1,o2,o3,o4 = st.columns(4)
        old_server = o1.text_input("Old server", placeholder="LMNZLREPORT01\\LM_RPT")
        old_db     = o2.text_input("Old database", placeholder="LMNZ_Report")
        old_schema = o3.text_input("Old schema", value="dbo")
        old_table  = o4.text_input("Old table *", placeholder="Customer")

        st.markdown("**New (target)**")
        n1,n2,n3,n4 = st.columns(4)
        new_server = n1.text_input("New server", placeholder="LMNZLRPT001\\LM_RPT_NEW")
        new_db     = n2.text_input("New database", placeholder="LMNZ_Report_New")
        new_schema = n3.text_input("New schema", value="dbo")
        new_table  = n4.text_input("New table *", placeholder="Customer")

        migration_method = st.radio("Migration method", MIGRATION_METHODS, horizontal=True)
        comment_sql = st.text_area("Notes / extra SQL (optional)", height=140)

        submitted = st.form_submit_button("Add to plan")
        if submitted:
            rec = dict(
                priority=priority,
                dataset_report=dataset_report.strip(),
                status=status,
                old_server=old_server.strip(), old_database=old_db.strip(), old_schema=old_schema.strip(), old_table=old_table.strip(),
                new_server=new_server.strip(), new_database=new_db.strip(), new_schema=new_schema.strip(), new_table=new_table.strip(),
                migration_method=migration_method,
                comment_sql=comment_sql.strip(),
            )
            if not rec["dataset_report"] or not rec["old_table"] or not rec["new_table"]:
                st.error("Dataset/Report, Old table, and New table are required.")
            else:
                insert_entry(engine, rec)
                df = fetch_df(engine)
                atomic_write_csv(df, CSV_PATH, LOCK_PATH)
                st.success("Saved to plan and mirrored to CSV.")

st.divider()

df = fetch_df(engine)

with st.expander("🔎 Filter, view, export", expanded=True):
    f1,f2,f3,f4 = st.columns(4)
    f_pri = f1.multiselect("Priority", PRIORITY_OPTIONS)
    f_stat= f2.multiselect("Status", STATUS_OPTIONS)
    f_ds  = f3.text_input("Dataset/Report contains")
    f_tbl = f4.text_input("Table contains (old/new)")

    view = df.copy()
    if f_pri: view = view[view["priority"].isin(f_pri)]
    if f_stat: view = view[view["status"].isin(f_stat)]
    if f_ds: view = view[view["dataset_report"].str.contains(f_ds, case=False, na=False)]
    if f_tbl:
        mask = (
            view["old_table"].str.contains(f_tbl, case=False, na=False) |
            view["new_table"].str.contains(f_tbl, case=False, na=False)
        )
        view = view[mask]

    st.dataframe(view, use_container_width=True)
    if st.button("⬇️ Export filtered view to CSV"):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "Download now",
            data=view.to_csv(index=False),
            file_name=f"PBI_TableMigration_{ts}.csv",
            mime="text/csv"
        )

st.divider()

with st.expander("🧰 Generate T‑SQL for a selected row", expanded=False):
    if df.empty:
        st.info("No rows yet.")
    else:
        df["label"] = df.apply(lambda r: f"{r['priority']} | {r['dataset_report']} | {r['old_database']}.{r['old_schema']}.{r['old_table']} → {r['new_database']}.{r['new_schema']}.{r['new_table']} [{r['status']}]", axis=1)
        choice = st.selectbox("Pick an item", df["label"].tolist())
        row = df.loc[df["label"]==choice].iloc[0].to_dict()
        script = build_script(row)
        st.code(script, language="sql")
        st.download_button("Download .sql", data=script, file_name=f"migrate_{row['new_schema']}_{row['new_table']}.sql")

st.divider()

with st.expander("✏️ Update status / notes", expanded=False):
    if df.empty:
        st.info("No rows yet.")
    else:
        df["label2"] = df.apply(lambda r: f"{r['dataset_report']} → {r['new_schema']}.{r['new_table']} ({r['status']})", axis=1)
        pick = st.selectbox("Choose row", df["label2"].tolist())
        rid = df.loc[df["label2"]==pick, "id"].iloc[0]
        rec = df.loc[df["id"]==rid].iloc[0].to_dict()
        c1,c2 = st.columns([2,1])
        new_notes = c1.text_area("Notes / SQL", value=rec.get("comment_sql") or "", height=160)
        new_status = c2.selectbox("Status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(rec["status"]) if rec["status"] in STATUS_OPTIONS else 0)
        if st.button("Save changes"):
            rec["comment_sql"] = new_notes
            rec["status"] = new_status
            update_entry(engine, rec)
            df2 = fetch_df(engine)
            atomic_write_csv(df2, CSV_PATH, LOCK_PATH)
            st.success("Updated and mirrored to CSV.")

with st.expander("ℹ️ Notes on fidelity & best practice"):
    st.markdown(
        """
        **Choosing a method**
        - *Create+Insert (safe)*: preserves structure by creating an empty table from the source shape (via `TOP(0)`), then inserts data. Still requires recreating PKs, indexes, constraints.
        - *SELECT INTO (quick copy)*: fastest for first load, but you'll rebuild all objects after.

        **Indexes & constraints**
        - Use SSMS "Script Table as" or DacPac (sqlpackage) to script keys/indexes. Apply **after** data copy.

        **Power BI switch-over**
        - Once validated on NEW, update data source/parameters in the dataset (or use deployment rules) and refresh.

        **Durability**
        - Data is stored in SQLite at `{DB_PATH}` and mirrored to CSV `{CSV_PATH}` with atomic writes, so sudden Windows shutdowns won't corrupt your plan.
        """
    )
