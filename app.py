# streamlit_pbi_table_migration_planner.py
# -------------------------------------------------
# Plan and track moving tables from an OLD SQL Server database
# to a NEW SQL Server database, prioritized by Power BI reports.
#
# (No Priority/Status; duplicate mappings are blocked.)

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import tempfile, shutil, uuid

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from filelock import FileLock


# --- Defaults for your environment ---
SCHEMA_CHOICES     = ["dim", "dbo", "repo", "pbi", "mart"]
DEFAULT_SCHEMA     = "dbo"

OLD_SERVER_DEFAULT = r"LMNZLRPT001\LM_RPT"
OLD_DB_DEFAULT     = "LesMills_Reporting"

NEW_SERVER_DEFAULT = r"LMNZLREPORT01\LM_RPT"
NEW_DB_DEFAULT     = "LesMills_Report"


# -------- Simple password gate (single shared password) --------
def _get_app_password() -> str:
    # Keep this in secrets/env in production; hardcoded here per your ask
    return "lesmillsreport"

def password_gate():
    if st.session_state.get("authed", False):
        return True
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
            st.error("APP_PASSWORD not configured.")
            st.stop()
        if pwd == _get_app_password():
            st.session_state["authed"] = True
            st.session_state["tries"] = 0
            st.rerun()
        else:
            st.session_state["tries"] += 1
            if st.session_state["tries"] >= 5:
                st.session_state["lock_until"] = now + 60
                st.error("Incorrect password. Locked for 60 seconds.")
            else:
                left = 5 - st.session_state["tries"]
                st.error(f"Incorrect password. {left} attempt(s) left.")
    st.stop()

password_gate()

with st.sidebar:
    if st.session_state.get("authed"):
        if st.button("Logout"):
            for k in ("authed", "tries", "lock_until", "__app_pwd"):
                st.session_state.pop(k, None)
            st.rerun()

# ---------------- App constants ----------------
APP_NAME = "Database Tables MigrationPlanner"
BASE_DIR = Path.home() / "Documents" / APP_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "planner.sqlite3"
CSV_PATH = BASE_DIR / "entries.csv"
LOCK_PATH = str(BASE_DIR / "entries.csv.lock")

# ---------------- DB helpers ----------------
def get_engine() -> Engine:
    return create_engine(f"sqlite:///{DB_PATH.as_posix()}", future=True)

# --- schema & integrity helpers ---

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entries (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  dataset_report TEXT,
  old_server TEXT, old_database TEXT, old_schema TEXT, old_table TEXT,
  new_server TEXT, new_database TEXT, new_schema TEXT, new_table TEXT,
  migration_method TEXT,
  comment_sql TEXT
);
"""

def dedupe_entries(engine: Engine) -> int:
    """
    Remove duplicate mappings, keeping the earliest created row per mapping.
    Returns number of rows deleted.
    """
    # Delete everything that is not the MIN(created_at,id) within each mapping group
    # (works on all modern SQLite versions)
    sql = text("""
        DELETE FROM entries
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id
                FROM entries e
                JOIN (
                    SELECT
                        old_server, old_database, old_schema, old_table,
                        new_server, new_database, new_schema, new_table,
                        MIN(created_at) AS min_created
                    FROM entries
                    GROUP BY
                        old_server, old_database, old_schema, old_table,
                        new_server, new_database, new_schema, new_table
                ) g
                  ON e.old_server  = g.old_server
                 AND e.old_database= g.old_database
                 AND e.old_schema  = g.old_schema
                 AND e.old_table   = g.old_table
                 AND e.new_server  = g.new_server
                 AND e.new_database= g.new_database
                 AND e.new_schema  = g.new_schema
                 AND e.new_table   = g.new_table
                 AND e.created_at  = g.min_created
            )
        );
    """)
    with engine.begin() as conn:
        res = conn.exec_driver_sql("SELECT COUNT(*) FROM entries").scalar()
        conn.execute(sql)
        res2 = conn.exec_driver_sql("SELECT COUNT(*) FROM entries").scalar()
    return (res - res2)

def ensure_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_SQL)

    # 1) Dedupe existing rows before adding unique constraint
    removed = dedupe_entries(engine)
    if removed:
        st.warning(f"Removed {removed} duplicate mapping(s) found in existing data.")

    # 2) Now add the UNIQUE index (safe on clean data)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_entries_mapping
                ON entries (
                  old_server, old_database, old_schema, old_table,
                  new_server, new_database, new_schema, new_table
                );
            """)
    except Exception as e:
        # As a fallback, surface a helpful message
        st.error(f"Failed to create unique index. Reason: {e}")

def mapping_exists(engine: Engine, rec: dict) -> bool:
    sql = text("""
        SELECT 1 FROM entries
        WHERE old_server=:old_server AND old_database=:old_database
          AND old_schema=:old_schema AND old_table=:old_table
          AND new_server=:new_server AND new_database=:new_database
          AND new_schema=:new_schema AND new_table=:new_table
        LIMIT 1
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, rec).fetchone()
    return row is not None

def insert_entry(engine: Engine, row: dict) -> tuple[bool, str]:
    """Returns (ok, message). Blocks duplicates."""
    row = row.copy()
    now = datetime.utcnow().isoformat(timespec="seconds")
    row.setdefault("id", str(uuid.uuid4()))
    row.setdefault("created_at", now)
    row["updated_at"] = now

    if mapping_exists(engine, row):
        return False, "This OLD→NEW table mapping already exists — not added."

    sql = text("""
        INSERT INTO entries (
          id, created_at, updated_at, dataset_report,
          old_server, old_database, old_schema, old_table,
          new_server, new_database, new_schema, new_table,
          migration_method, comment_sql
        ) VALUES (
          :id, :created_at, :updated_at, :dataset_report,
          :old_server, :old_database, :old_schema, :old_table,
          :new_server, :new_database, :new_schema, :new_table,
          :migration_method, :comment_sql
        )
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, row)
        return True, "Saved to plan."
    except Exception as e:
        # In case of race, unique index will throw; surface a friendly message
        msg = str(e)
        if "ux_entries_mapping" in msg or "UNIQUE" in msg.upper():
            return False, "This OLD→NEW table mapping already exists — not added."
        return False, f"Insert failed: {e}"

def update_entry(engine: Engine, row: dict) -> None:
    row = row.copy()
    row["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    sql = text("""
        UPDATE entries SET
          updated_at=:updated_at,
          dataset_report=:dataset_report,
          old_server=:old_server, old_database=:old_database, old_schema=:old_schema, old_table=:old_table,
          new_server=:new_server, new_database=:new_database, new_schema=:new_schema, new_table=:new_table,
          migration_method=:migration_method,
          comment_sql=:comment_sql
        WHERE id=:id
    """)
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
MIGRATION_METHODS = [
    "Create+Insert (safe)",
    "SELECT INTO (quick copy)",
]

def make_tsql_create_insert(old_server, old_db, old_schema, old_table,
                            new_server, new_db, new_schema, new_table) -> str:
    return f"""
-- Run ON NEW server: {new_server}
-- 1) Ensure schema exists
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'{new_schema}')
    EXEC('CREATE SCHEMA [{new_schema}]');
GO

-- 2) Create empty table with source shape
DROP TABLE IF EXISTS [{new_schema}].[{new_table}];
SELECT TOP(0) *
INTO   [{new_schema}].[{new_table}]
FROM   OPENQUERY([{old_server}], 'SELECT * FROM [{old_db}].[{old_schema}].[{old_table}]');

-- TODO: Recreate PKs, indexes, constraints, identities, defaults, collation, computed columns.

-- 3) Insert data (simple one-shot; replace with batching for very large tables)
INSERT INTO [{new_schema}].[{new_table}]
SELECT *
FROM OPENQUERY([{old_server}], 'SELECT * FROM [{old_db}].[{old_schema}].[{old_table}]');

SELECT @@ROWCOUNT AS rows_copied;
""".strip()

def make_tsql_select_into(old_server, old_db, old_schema, old_table,
                          new_server, new_db, new_schema, new_table) -> str:
    return f"""
-- Run ON NEW server: {new_server}
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
        dataset_report = st.text_input("Dataset/Report name *", placeholder="e.g., Customer (Service)")
        st.markdown("**Old (source)**")
        o1,o2,o3,o4 = st.columns(4)
        old_server = o1.text_input("Old server", value=OLD_SERVER_DEFAULT)
        old_db     = o2.text_input("Old database", value=OLD_DB_DEFAULT)
        old_schema = o3.selectbox(
            "Old schema",
            SCHEMA_CHOICES,
            index=SCHEMA_CHOICES.index(DEFAULT_SCHEMA) if DEFAULT_SCHEMA in SCHEMA_CHOICES else 0,
        )
        old_table  = o4.text_input("Old table *", placeholder="Customer")

        st.markdown("**New (target)**")
        n1,n2,n3,n4 = st.columns(4)
        new_server = n1.text_input("New server", value=NEW_SERVER_DEFAULT)
        new_db     = n2.text_input("New database", value=NEW_DB_DEFAULT)
        new_schema = n3.selectbox(
            "New schema",
            SCHEMA_CHOICES,
            index=SCHEMA_CHOICES.index(DEFAULT_SCHEMA) if DEFAULT_SCHEMA in SCHEMA_CHOICES else 0,
        )
        new_table  = n4.text_input("New table *", placeholder="Customer")

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
                dataset_report=dataset_report.strip(),
                old_server=old_server.strip(), old_database=old_db.strip(), old_schema=old_schema.strip(), old_table=old_table.strip(),
                new_server=new_server.strip(), new_database=new_db.strip(), new_schema=new_schema.strip(), new_table=new_table.strip(),
                migration_method=migration_method,
                comment_sql=comment_sql.strip(),
            )
            if not rec["dataset_report"] or not rec["old_table"] or not rec["new_table"]:
                st.error("Dataset/Report, Old table, and New table are required.")
            else:
                ok, msg = insert_entry(engine, rec)
                if ok:
                    df = fetch_df(engine)
                    atomic_write_csv(df, CSV_PATH, LOCK_PATH)
                    st.success("Saved to plan and mirrored to CSV.")
                else:
                    st.warning(msg)

st.divider()

df = fetch_df(engine)

with st.expander("🔎 Filter, view, export", expanded=True):
    f1,f2 = st.columns(2)
    f_ds  = f1.text_input("Dataset/Report contains")
    f_tbl = f2.text_input("Table contains (old/new)")

    view = df.copy()
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

with st.expander("🧰 Generate T-SQL for a selected row", expanded=False):
    if df.empty:
        st.info("No rows yet.")
    else:
        df["label"] = df.apply(
            lambda r: f"{r['dataset_report']} | {r['old_database']}.{r['old_schema']}.{r['old_table']} → {r['new_database']}.{r['new_schema']}.{r['new_table']}",
            axis=1
        )
        choice = st.selectbox("Pick an item", df["label"].tolist())
        row = df.loc[df["label"]==choice].iloc[0].to_dict()
        script = build_script(row)
        st.code(script, language="sql")
        st.download_button("Download .sql", data=script, file_name=f"migrate_{row['new_schema']}_{row['new_table']}.sql")

st.divider()

with st.expander("✏️ Update notes / mapping", expanded=False):
    if df.empty:
        st.info("No rows yet.")
    else:
        df["label2"] = df.apply(lambda r: f"{r['dataset_report']} → {r['new_schema']}.{r['new_table']}", axis=1)
        pick = st.selectbox("Choose row", df["label2"].tolist())
        rid = df.loc[df["label2"]==pick, "id"].iloc[0]
        rec = df.loc[df["id"]==rid].iloc[0].to_dict()

        cA,cB,cC,cD = st.columns(4)
        rec["old_server"]   = cA.text_input("Old server", value=rec["old_server"])
        rec["old_database"] = cB.text_input("Old database", value=rec["old_database"])
        rec["old_schema"]   = cC.selectbox(
            "Old schema",
            SCHEMA_CHOICES,
            index=SCHEMA_CHOICES.index(rec.get("old_schema", DEFAULT_SCHEMA))
                if rec.get("old_schema", DEFAULT_SCHEMA) in SCHEMA_CHOICES else 0,
            key=f"old_schema_{rid}",
        )
        rec["old_table"]    = cD.text_input("Old table", value=rec["old_table"])

        dA,dB,dC,dD = st.columns(4)
        rec["new_server"]   = dA.text_input("New server", value=rec["new_server"])
        rec["new_database"] = dB.text_input("New database", value=rec["new_database"])
        rec["new_schema"]   = dC.selectbox(
            "New schema",
            SCHEMA_CHOICES,
            index=SCHEMA_CHOICES.index(rec.get("new_schema", DEFAULT_SCHEMA))
                if rec.get("new_schema", DEFAULT_SCHEMA) in SCHEMA_CHOICES else 0,
            key=f"new_schema_{rid}",
        )
        rec["new_table"]    = dD.text_input("New table", value=rec["new_table"])


        rec["dataset_report"] = st.text_input("Dataset/Report", value=rec["dataset_report"])
        rec["migration_method"] = st.radio("Migration method", MIGRATION_METHODS,
                                           index=MIGRATION_METHODS.index(rec.get("migration_method") or MIGRATION_METHODS[0]),
                                           horizontal=True)
        new_notes = st.text_area("Notes / SQL", value=rec.get("comment_sql") or "", height=160)

        if st.button("Save changes"):
            rec["comment_sql"] = new_notes

            # Prevent updates that would create a duplicate mapping
            if mapping_exists(engine, rec) and df.loc[df["id"]==rid].empty is False:
                # If mapping exists on a different id, block
                existing = df[(df["old_server"]==rec["old_server"]) &
                              (df["old_database"]==rec["old_database"]) &
                              (df["old_schema"]==rec["old_schema"]) &
                              (df["old_table"]==rec["old_table"]) &
                              (df["new_server"]==rec["new_server"]) &
                              (df["new_database"]==rec["new_database"]) &
                              (df["new_schema"]==rec["new_schema"]) &
                              (df["new_table"]==rec["new_table"])]
                if not existing.empty and existing.iloc[0]["id"] != rid:
                    st.warning("This OLD→NEW table mapping already exists — not updated.")
                    st.stop()

            update_entry(engine, rec)
            df2 = fetch_df(engine)
            atomic_write_csv(df2, CSV_PATH, LOCK_PATH)
            st.success("Updated and mirrored to CSV.")

with st.expander("ℹ️ Notes"):
    st.markdown(
        f"""
        **T-SQL generation**
        - *Create+Insert (safe)* builds empty target via `TOP(0)` and inserts rows; recreate PKs/indexes after.
        - *SELECT INTO (quick copy)* fastest first load; recreate PKs/indexes after.

        **Switch-over**
        - After validating on NEW, point your Power BI dataset to the new server/db and refresh.

        **Durability**
        - SQLite at `{DB_PATH}` is the point of truth; CSV mirror at `{CSV_PATH}` uses atomic writes.
        """
    )
