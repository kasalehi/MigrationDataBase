# streamlit_pbi_table_migration_planner.py
# -------------------------------------------------
# Plan & track moving tables from an OLD SQL Server database to a NEW one.
# - Two-status workflow: "In Progress" / "Completed" (default = In Progress)
# - Duplicate prevention & cleanup: ONLY by old_table (case/space-insensitive)
# - Durable SQLite storage (~/Documents/Database Tables MigrationPlanner)
# - CSV mirror with atomic writes
# - T-SQL generation helpers
# - Single shared password gate

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import uuid
import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from filelock import FileLock

# =========================
# App defaults & constants
# =========================
SCHEMA_CHOICES     = ["dim", "dbo", "repo", "pbi", "mart"]
DEFAULT_SCHEMA     = "dbo"

OLD_SERVER_DEFAULT = r"LMNZLRPT001\LM_RPT"
OLD_DB_DEFAULT     = "LesMills_Reporting"

NEW_SERVER_DEFAULT = r"LMNZLREPORT01\LM_RPT"
NEW_DB_DEFAULT     = "LesMills_Report"

STATUS_CHOICES  = ["In Progress", "Completed"]
DEFAULT_STATUS  = "In Progress"

APP_NAME = "Database Tables MigrationPlanner"
BASE_DIR = Path.home() / "Documents" / APP_NAME
BASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH  = BASE_DIR / "planner.sqlite3"
CSV_PATH = BASE_DIR / "entries.csv"
LOCK_PATH = str(BASE_DIR / "entries.csv.lock")

# =========================
# Auth (single shared pwd)
# =========================
def _get_app_password() -> str:
    # Move to st.secrets["APP_PASSWORD"] or env var for production
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

# =========================
# DB layer
# =========================
def get_engine() -> Engine:
    return create_engine(f"sqlite:///{DB_PATH.as_posix()}", future=True)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entries (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  dataset_report TEXT,
  old_server   TEXT,
  old_database TEXT,
  old_schema   TEXT,
  old_table    TEXT,
  new_server   TEXT,
  new_database TEXT,
  new_schema   TEXT,
  new_table    TEXT,
  migration_method TEXT,
  comment_sql  TEXT,
  status       TEXT
);
"""

def dedupe_entries(engine: Engine) -> int:
    """
    Keep the earliest row per normalized old_table (LOWER(TRIM(old_table))).
    Delete all other duplicates regardless of server/db/schema/targets.
    """
    sql = text("""
        DELETE FROM entries
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT e.id
                FROM entries e
                JOIN (
                    SELECT
                        LOWER(TRIM(old_table)) AS t_norm,
                        MIN(created_at)        AS min_created
                    FROM entries
                    GROUP BY LOWER(TRIM(old_table))
                ) g
                  ON LOWER(TRIM(e.old_table)) = g.t_norm
                 AND e.created_at             = g.min_created
            )
        );
    """)
    with engine.begin() as conn:
        before = conn.exec_driver_sql("SELECT COUNT(*) FROM entries").scalar()
        conn.execute(sql)
        after = conn.exec_driver_sql("SELECT COUNT(*) FROM entries").scalar()
    return (before - after)

def ensure_db(engine: Engine) -> None:
    # 1) Table
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_SQL)

    # 2) Ensure `status` column populated if DB existed before
    with engine.begin() as conn:
        cols = conn.exec_driver_sql("PRAGMA table_info(entries);").fetchall()
        names = [c[1] for c in cols]
        if "status" not in names:
            conn.exec_driver_sql("ALTER TABLE entries ADD COLUMN status TEXT;")
        conn.exec_driver_sql(
            "UPDATE entries SET status = :dflt WHERE status IS NULL OR TRIM(status)='';",
            {"dflt": DEFAULT_STATUS},
        )

    # 3) Dedupe by old_table only
    removed = dedupe_entries(engine)
    if removed:
        st.warning(f"Removed {removed} duplicate row(s) by old_table.")

    # 4) Uniqueness by old_table only (case/space-insensitive)
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_entries_oldtable
            ON entries (LOWER(TRIM(old_table)));
        """)

def mapping_exists_by_old_table(engine: Engine, old_table: str) -> bool:
    sql = text("""
        SELECT 1
        FROM entries
        WHERE LOWER(TRIM(old_table)) = LOWER(TRIM(:t))
        LIMIT 1
    """)
    with engine.begin() as conn:
        return conn.execute(sql, {"t": (old_table or "")}).fetchone() is not None

def insert_entry(engine: Engine, row: dict) -> tuple[bool, str]:
    """
    Insert a row; block duplicates by old_table only.
    Returns (ok, message).
    """
    row = row.copy()
    now = datetime.utcnow().isoformat(timespec="seconds")
    row.setdefault("id", str(uuid.uuid4()))
    row.setdefault("created_at", now)
    row["updated_at"] = now

    # normalize critical fields
    row["old_table"] = (row.get("old_table") or "").strip()
    row["old_schema"] = (row.get("old_schema") or DEFAULT_SCHEMA).strip().lower()
    row["new_schema"] = (row.get("new_schema") or DEFAULT_SCHEMA).strip().lower()
    row["status"] = (row.get("status") or DEFAULT_STATUS).strip()

    if not row["dataset_report"] or not row["old_table"] or not row["new_table"]:
        return False, "Dataset/Report, Old table, and New table are required."

    # Block: duplicate by old_table only
    if mapping_exists_by_old_table(engine, row["old_table"]):
        return False, "Duplicate by old_table: an entry for this old_table already exists."

    sql = text("""
        INSERT INTO entries (
          id, created_at, updated_at, dataset_report,
          old_server, old_database, old_schema, old_table,
          new_server, new_database, new_schema, new_table,
          migration_method, comment_sql, status
        ) VALUES (
          :id, :created_at, :updated_at, :dataset_report,
          :old_server, :old_database, :old_schema, :old_table,
          :new_server, :new_database, :new_schema, :new_table,
          :migration_method, :comment_sql, :status
        )
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, row)
        return True, "Saved to plan."
    except Exception as e:
        if "ux_entries_oldtable" in str(e) or "UNIQUE" in str(e).upper():
            return False, "Duplicate by old_table: an entry for this old_table already exists."
        return False, f"Insert failed: {e}"

def update_entry(engine: Engine, row: dict) -> tuple[bool, str]:
    """
    Update row fields. Prevent changing to an old_table that already exists on a different id.
    """
    row = row.copy()
    row["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    row["old_schema"] = (row.get("old_schema") or DEFAULT_SCHEMA).strip().lower()
    row["new_schema"] = (row.get("new_schema") or DEFAULT_SCHEMA).strip().lower()
    row["status"]     = (row.get("status") or DEFAULT_STATUS).strip()
    row["old_table"]  = (row.get("old_table") or "").strip()
    row["new_table"]  = (row.get("new_table") or "").strip()

    # If changing old_table, ensure uniqueness by old_table
    with engine.begin() as conn:
        existing = conn.exec_driver_sql(
            """
            SELECT id FROM entries
            WHERE LOWER(TRIM(old_table)) = LOWER(TRIM(:t))
            """,
            {"t": row["old_table"]},
        ).fetchall()
    # If there's a different id with same old_table, block
    if any(eid[0] != row["id"] for eid in existing):
        return False, "Duplicate by old_table: another row already uses this old_table."

    sql = text("""
        UPDATE entries SET
          updated_at=:updated_at,
          dataset_report=:dataset_report,
          old_server=:old_server, old_database=:old_database, old_schema=:old_schema, old_table=:old_table,
          new_server=:new_server, new_database=:new_database, new_schema=:new_schema, new_table=:new_table,
          migration_method=:migration_method,
          comment_sql=:comment_sql,
          status=:status
        WHERE id=:id
    """)
    with engine.begin() as conn:
        conn.execute(sql, row)
    return True, "Updated."

def delete_entries(engine: Engine, ids: list[str]) -> int:
    if not ids:
        return 0
    sql = text("DELETE FROM entries WHERE id = :id")
    with engine.begin() as conn:
        for _id in ids:
            conn.execute(sql, {"id": _id})
    return len(ids)

def fetch_df(engine: Engine) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql_query("SELECT * FROM entries ORDER BY created_at DESC", conn)

# =========================
# CSV durable mirror
# =========================
def atomic_write_csv(df: pd.DataFrame, csv_path: Path, lock_path: str) -> None:
    lock = FileLock(lock_path)
    with lock:
        tmp_dir = tempfile.mkdtemp()
        tmp_file = Path(tmp_dir) / "entries.tmp.csv"
        df.to_csv(tmp_file, index=False)
        shutil.move(str(tmp_file), str(csv_path))
        shutil.rmtree(tmp_dir, ignore_errors=True)

# =========================
# T-SQL script generator
# =========================
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

# =========================
# UI
# =========================
st.set_page_config(page_title="Table Migration Planner", page_icon="📤", layout="wide")
st.title("📤 Table Migration Planner")
st.caption(f"Data folder: {BASE_DIR}")

engine = get_engine(); ensure_db(engine)

# ---- Add mapping
with st.expander("➕ Add table move", expanded=True):
    with st.form("new_row"):
        dataset_report = st.text_input("Dataset/Report name *", placeholder="e.g., Customer (Service)")

        st.markdown("**Old (source)**")
        o1,o2,o3,o4 = st.columns(4)
        old_server = o1.text_input("Old server", value=OLD_SERVER_DEFAULT)
        old_db     = o2.text_input("Old database", value=OLD_DB_DEFAULT)
        old_schema = o3.selectbox("Old schema", SCHEMA_CHOICES,
                                  index=SCHEMA_CHOICES.index(DEFAULT_SCHEMA))
        old_table  = o4.text_input("Old table *", placeholder="Customer")

        st.markdown("**New (target)**")
        n1,n2,n3,n4 = st.columns(4)
        new_server = n1.text_input("New server", value=NEW_SERVER_DEFAULT)
        new_db     = n2.text_input("New database", value=NEW_DB_DEFAULT)
        new_schema = n3.selectbox("New schema", SCHEMA_CHOICES,
                                  index=SCHEMA_CHOICES.index(DEFAULT_SCHEMA))
        new_table  = n4.text_input("New table *", placeholder="Customer")

        migration_method = st.radio("Migration method", MIGRATION_METHODS, horizontal=True)
        comment_sql = st.text_area("Notes / extra SQL (optional)", height=140)
        status_val  = st.selectbox("Status", STATUS_CHOICES, index=STATUS_CHOICES.index(DEFAULT_STATUS))

        submitted = st.form_submit_button("Add to plan")
        if submitted:
            rec = dict(
                dataset_report=dataset_report.strip(),
                old_server=old_server.strip(),
                old_database=old_db.strip(),
                old_schema=old_schema.strip().lower(),
                old_table=old_table.strip(),
                new_server=new_server.strip(),
                new_database=new_db.strip(),
                new_schema=new_schema.strip().lower(),
                new_table=new_table.strip(),
                migration_method=migration_method,
                comment_sql=comment_sql.strip(),
                status=status_val.strip(),
            )
            ok, msg = insert_entry(engine, rec)
            if ok:
                df = fetch_df(engine); atomic_write_csv(df, CSV_PATH, LOCK_PATH)
                st.success("Saved to plan and mirrored to CSV.")
            else:
                st.warning(msg)

st.divider()

# ---- Filter, inline status edit, export
def normalize_status_values(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        df["status"] = DEFAULT_STATUS
        return df
    df["status"] = df["status"].fillna("").astype(str).str.strip()
    df.loc[~df["status"].isin(STATUS_CHOICES), "status"] = DEFAULT_STATUS
    return df

with st.expander("🔎 Filter, edit status inline, export", expanded=True):
    raw = fetch_df(engine).copy()
    view = normalize_status_values(raw)

    f1, f2, f3 = st.columns(3)
    f_ds  = f1.text_input("Dataset/Report contains")
    f_tbl = f2.text_input("Table contains (old/new)")
    f_st  = f3.multiselect("Status", STATUS_CHOICES, default=[])

    if f_ds:
        view = view[view["dataset_report"].str.contains(f_ds, case=False, na=False)]
    if f_tbl:
        mask = (
            view["old_table"].str.contains(f_tbl, case=False, na=False) |
            view["new_table"].str.contains(f_tbl, case=False, na=False)
        )
        view = view[mask]
    if f_st:
        view = view[view["status"].isin(f_st)]

    preferred_order = [
        "status",
        "dataset_report",
        "old_server","old_database","old_schema","old_table",
        "new_server","new_database","new_schema","new_table",
        "migration_method",
        "comment_sql",
        "created_at","updated_at","id",
    ]
    cols_in_view = [c for c in preferred_order if c in view.columns] + \
                   [c for c in view.columns if c not in preferred_order]
    view = view[cols_in_view].reset_index(drop=True)

    st.caption("Edit statuses directly in the grid, then click **Save edits** to persist.")
    edited = st.data_editor(
        view,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "status": st.column_config.SelectboxColumn(
                "Status",
                options=STATUS_CHOICES,
                help="Only two options are allowed.",
                width="small",
                required=True,
                default=DEFAULT_STATUS,
            )
        },
        disabled=[
            "dataset_report","old_server","old_database","old_schema","old_table",
            "new_server","new_database","new_schema","new_table",
            "migration_method","comment_sql","created_at","updated_at","id"
        ],
        hide_index=True,
        key="edit_view",
    )

    if st.button("💾 Save edits"):
        try:
            edited = normalize_status_values(edited)
            changes = edited[["id","status"]].merge(
                raw[["id","status"]], on="id", how="left", suffixes=("_new","_old")
            )
            to_update = changes[changes["status_new"] != changes["status_old"]][["id","status_new"]]
            if not to_update.empty:
                with engine.begin() as conn:
                    for _id, _st in to_update.itertuples(index=False, name=None):
                        conn.execute(
                            text("UPDATE entries SET status=:s, updated_at=:u WHERE id=:i"),
                            {"s": _st, "u": datetime.utcnow().isoformat(timespec="seconds"), "i": _id},
                        )
                df2 = fetch_df(engine); atomic_write_csv(df2, CSV_PATH, LOCK_PATH)
                st.success(f"Saved {len(to_update)} status change(s) and updated CSV.")
                st.rerun()
            else:
                st.info("No changes to save.")
        except Exception as e:
            st.error(f"Save failed: {e}")

    if not edited.empty:
        counts = edited["status"].value_counts().reindex(STATUS_CHOICES, fill_value=0)
        cA, cB = st.columns(2)
        cA.metric("In Progress", int(counts.get("In Progress", 0)))
        cB.metric("Completed",   int(counts.get("Completed", 0)))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "⬇️ Download filtered CSV (includes Status)",
        data=edited.to_csv(index=False),
        file_name=f"PBI_TableMigration_{ts}.csv",
        mime="text/csv"
    )

st.divider()

# ---- Generate T-SQL for a selected row
df_all = fetch_df(engine)
with st.expander("🧰 Generate T-SQL for a selected row", expanded=False):
    if df_all.empty:
        st.info("No rows yet.")
    else:
        df_all = df_all.copy()
        df_all["label"] = df_all.apply(
            lambda r: f"{r['dataset_report']} | {r['old_database']}.{r['old_schema']}.{r['old_table']} → {r['new_database']}.{r['new_schema']}.{r['new_table']}  [{r.get('status','')}]",
            axis=1
        )
        choice = st.selectbox("Pick an item", df_all["label"].tolist())
        row = df_all.loc[df_all["label"]==choice].iloc[0].to_dict()
        script = build_script(row)
        st.code(script, language="sql")
        st.download_button("Download .sql", data=script, file_name=f"migrate_{row['new_schema']}_{row['new_table']}.sql")

st.divider()

# ---- Update mapping / notes + single delete
with st.expander("✏️ Update mapping / notes", expanded=False):
    df = fetch_df(engine)
    if df.empty:
        st.info("No rows yet.")
    else:
        df["label2"] = df.apply(lambda r: f"{r['dataset_report']} → {r['new_schema']}.{r['new_table']}  [{r.get('status','')}]",
                                axis=1)
        pick = st.selectbox("Choose row", df["label2"].tolist())
        rid = df.loc[df["label2"]==pick, "id"].iloc[0]
        rec = df.loc[df["id"]==rid].iloc[0].to_dict()

        cA,cB,cC,cD = st.columns(4)
        rec["old_server"]   = cA.text_input("Old server", value=rec["old_server"])
        rec["old_database"] = cB.text_input("Old database", value=rec["old_database"])
        rec["old_schema"]   = cC.selectbox(
            "Old schema", SCHEMA_CHOICES,
            index=SCHEMA_CHOICES.index(rec.get("old_schema", DEFAULT_SCHEMA))
                if rec.get("old_schema", DEFAULT_SCHEMA) in SCHEMA_CHOICES else 0,
            key=f"old_schema_{rid}",
        )
        rec["old_table"]    = cD.text_input("Old table", value=rec["old_table"])

        dA,dB,dC,dD = st.columns(4)
        rec["new_server"]   = dA.text_input("New server", value=rec["new_server"])
        rec["new_database"] = dB.text_input("New database", value=rec["new_database"])
        rec["new_schema"]   = dC.selectbox(
            "New schema", SCHEMA_CHOICES,
            index=SCHEMA_CHOICES.index(rec.get("new_schema", DEFAULT_SCHEMA))
                if rec.get("new_schema", DEFAULT_SCHEMA) in SCHEMA_CHOICES else 0,
            key=f"new_schema_{rid}",
        )
        rec["new_table"]    = dD.text_input("New table", value=rec["new_table"])

        rec["dataset_report"]   = st.text_input("Dataset/Report", value=rec["dataset_report"])
        rec["migration_method"] = st.radio(
            "Migration method", MIGRATION_METHODS,
            index=MIGRATION_METHODS.index(rec.get("migration_method") or MIGRATION_METHODS[0]),
            horizontal=True
        )
        rec["status"] = st.selectbox("Status", STATUS_CHOICES,
                                     index=STATUS_CHOICES.index(rec.get("status", DEFAULT_STATUS)))
        new_notes = st.text_area("Notes / SQL", value=rec.get("comment_sql") or "", height=160)

        c1, c2 = st.columns([1,1])
        if c1.button("💾 Save changes"):
            rec["comment_sql"] = new_notes
            ok, msg = update_entry(engine, rec)
            if ok:
                df2 = fetch_df(engine); atomic_write_csv(df2, CSV_PATH, LOCK_PATH)
                st.success("Updated and mirrored to CSV.")
                st.rerun()
            else:
                st.warning(msg)

        if c2.button("🗑️ Delete this row"):
            confirm = st.checkbox("Yes, delete this row permanently", key=f"confirm_del_{rid}")
            if confirm and st.button("Delete now", type="primary", key=f"del_now_{rid}"):
                delete_entries(engine, [rid])
                df2 = fetch_df(engine); atomic_write_csv(df2, CSV_PATH, LOCK_PATH)
                st.success("Row deleted and CSV updated.")
                st.rerun()

st.divider()

# ---- Multi-delete
with st.expander("🗑️ Delete rows", expanded=False):
    df = fetch_df(engine)
    if df.empty:
        st.info("No rows to delete.")
    else:
        df["label_del"] = df.apply(
            lambda r: f"{r['dataset_report']} | {r['old_database']}.{r['old_schema']}.{r['old_table']} → {r['new_database']}.{r['new_schema']}.{r['new_table']}  [{r.get('status','')}]  (id={r['id'][:8]}…)",
            axis=1
        )
        to_delete_labels = st.multiselect("Select one or more rows to delete", df["label_del"].tolist())
        ids_to_delete = df.loc[df["label_del"].isin(to_delete_labels), "id"].tolist()
        colA, colB = st.columns([1,2])
        sure = colA.checkbox("I understand this is permanent")
        do_delete = colB.button("Delete selected", type="primary", disabled=not ids_to_delete)

        if do_delete:
            if not sure:
                st.warning("Please tick the confirmation checkbox first.")
            else:
                n = delete_entries(engine, ids_to_delete)
                df2 = fetch_df(engine); atomic_write_csv(df2, CSV_PATH, LOCK_PATH)
                st.success(f"Deleted {n} row(s) and updated CSV.")
                st.rerun()

# ---- Notes
with st.expander("ℹ️ Notes"):
    st.markdown(
        f"""
        **Status tracking**
        - Only two statuses exist: *In Progress* (default) and *Completed*. You can edit inline and export with Status.

        **Deduplication & Uniqueness**
        - Duplicates are removed/blocked **only by** `old_table` (case/space-insensitive), regardless of server/db/schema.

        **T-SQL generation**
        - *Create+Insert (safe)*: Create empty target via `TOP(0)` then insert. Rebuild PKs/indexes after.
        - *SELECT INTO (quick copy)*: Fast first load; rebuild PKs/indexes after.

        **Durability**
        - SQLite at `{DB_PATH}` is the source of truth; CSV mirror at `{CSV_PATH}` uses atomic writes.
        """
    )
