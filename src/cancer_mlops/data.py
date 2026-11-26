"""DuckDB ingestion, SQL data audit, schema validation, and deterministic partitioning."""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from cancer_mlops.config import PROJECT_ROOT, AppConfig
from cancer_mlops.storage import StorageBackend, get_storage

SQL_DIR = PROJECT_ROOT / "sql"


class DataValidationError(ValueError):
    """Raised when raw input fails required schema or integrity checks."""


def snake_case(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_")


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_sql_queries(
    con: duckdb.DuckDBPyConnection,
    sql_path: Path,
    params: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Execute a multi-statement templated SQL script and collect SELECT outputs."""
    text = sql_path.read_text(encoding="utf-8")
    for key, value in (params or {}).items():
        text = text.replace("{{" + key + "}}", value)

    results: dict[str, list[dict]] = {}
    pending_name: str | None = None
    for raw in text.split(";"):
        body: list[str] = []
        stmt_name: str | None = None
        for line in raw.splitlines():
            s = line.strip()
            if s.startswith("-- NAME:"):
                stmt_name = s.split(":", 1)[1].strip()
            elif s.startswith("-- TITLE:"):
                stmt_name = s.split(":", 1)[1].strip()
            elif s and not s.startswith("--"):
                body.append(line)
        if pending_name is None:
            pending_name = stmt_name
        sql = "\n".join(body).strip()
        if not sql:
            continue
        if sql.upper().startswith(("CREATE", "DROP", "BEGIN")):
            con.execute(sql)
        else:
            df = con.execute(sql).fetchdf()
            key_name = pending_name or f"query_{len(results) + 1}"
            results[key_name] = df.to_dict(orient="records")
        pending_name = None
    return results


def validate_raw_schema(df: pd.DataFrame, cfg: AppConfig) -> dict:
    """Validate dataset schema, uniqueness, target values, and finiteness."""
    id_col = cfg.data.id_col
    target_col = cfg.data.target_col
    pos = cfg.data.positive_label

    if df.empty:
        raise DataValidationError("Input dataset is empty.")
    for required in (id_col, target_col):
        if required not in df.columns:
            raise DataValidationError(f"Missing required column: {required}")

    if df[id_col].isna().any():
        raise DataValidationError(f"Column {id_col} contains null values.")
    if df[id_col].nunique() != len(df):
        raise DataValidationError(f"Column {id_col} contains duplicate IDs.")
    if df[target_col].isna().any():
        raise DataValidationError(f"Target column {target_col} contains null values.")

    classes = set(df[target_col].astype(str).unique())
    if pos not in classes or len(classes) != 2:
        raise DataValidationError(
            f"Expected binary target containing '{pos}', found classes: {sorted(classes)}"
        )

    excluded = {id_col, target_col, "y_m", "partition"}
    numeric_cols = [c for c in df.columns if c not in excluded]
    if not numeric_cols:
        raise DataValidationError("No numeric feature columns detected.")

    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Feature '{col}' is not numeric (dtype={df[col].dtype}).")
        vals = df[col].to_numpy(dtype=float)
        if np.isinf(vals).any():
            raise DataValidationError(f"Feature '{col}' contains non-finite values.")

    total_nulls = int(df[numeric_cols].isna().sum().sum())
    unique_vectors = int(len(df.drop_duplicates(subset=numeric_cols)))
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_numeric_features": len(numeric_cols),
        "unique_feature_vectors": unique_vectors,
        "total_nulls": total_nulls,
        "class_counts": {str(k): int(v) for k, v in df[target_col].value_counts().items()},
    }


def load_and_partition(
    cfg: AppConfig,
    storage: StorageBackend | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Ingest dataset via storage backend into DuckDB, audit via SQL, and split 70/30."""
    data_uri = cfg.resolved_data_uri()
    backend = storage or get_storage(data_uri)
    cache_dir = cfg.resolve_path(cfg.artifacts.root_dir) / "cache"
    local_csv = backend.resolve_local_file(data_uri, cache_dir=cache_dir)
    data_sha256 = compute_file_sha256(local_csv)

    csv_posix = str(local_csv).replace("\\", "/")
    con = duckdb.connect()
    try:
        desc = con.execute(
            f"DESCRIBE SELECT * FROM read_csv_auto('{csv_posix}', header=true)"
        ).fetchall()
        cols = [r[0] for r in desc]
        numeric = [
            snake_case(r[0])
            for r in desc
            if str(r[1]).upper() in ("DOUBLE", "BIGINT", "INTEGER", "FLOAT")
            and snake_case(r[0]) not in {cfg.data.id_col, cfg.data.target_col}
        ]
        select_parts = ", ".join(f'"{c}" AS "{snake_case(c)}"' for c in cols)
        con.execute(
            f"CREATE OR REPLACE VIEW src AS SELECT {select_parts} "
            f"FROM read_csv_auto('{csv_posix}', header=true)"
        )

        sql_params = {
            "numeric_cols": ", ".join(numeric),
            "target_col": cfg.data.target_col,
            "positive_label": cfg.data.positive_label,
            "id_col": cfg.data.id_col,
            "train_pct": str(cfg.data.train_pct),
            "hash_mod": str(cfg.data.hash_mod),
        }
        audit_results = run_sql_queries(con, SQL_DIR / "01_audit.sql", sql_params)
        raw_row_count = int(con.execute("SELECT COUNT(*) FROM src").fetchone()[0])

        if cfg.data.deduplicate_features:
            col_expr = ", ".join(f'"{c}"' for c in numeric)
            conflicts = int(
                con.execute(
                    f'SELECT COUNT(*) FROM (SELECT {col_expr} FROM src '
                    f'GROUP BY {col_expr} HAVING COUNT(DISTINCT "{cfg.data.target_col}") > 1)'
                ).fetchone()[0]
            )
            if conflicts > 0:
                raise DataValidationError(
                    f"Detected {conflicts} identical feature vectors with conflicting labels."
                )
            con.execute(
                "CREATE OR REPLACE VIEW src_clean AS "
                "SELECT * FROM src "
                f"QUALIFY ROW_NUMBER() OVER ("
                f'PARTITION BY {col_expr} ORDER BY "{cfg.data.id_col}") = 1'
            )
        else:
            con.execute("CREATE OR REPLACE VIEW src_clean AS SELECT * FROM src")

        part_results = run_sql_queries(con, SQL_DIR / "02_partition.sql", sql_params)

        full_df = con.execute(
            f'SELECT * FROM vw_dataset ORDER BY "{cfg.data.id_col}"'
        ).fetchdf()
    finally:
        con.close()

    schema_stats = validate_raw_schema(full_df, cfg)
    full_df["partition"] = full_df["partition"].astype(str)
    full_df["y_m"] = full_df["y_m"].astype(int)

    train_df = full_df[full_df["partition"] == "train"].copy().reset_index(drop=True)
    test_df = full_df[full_df["partition"] == "test"].copy().reset_index(drop=True)

    id_col = cfg.data.id_col
    if not set(train_df[id_col]).isdisjoint(set(test_df[id_col])):
        raise DataValidationError("Partition overlap detected between train and test.")
    if train_df.empty or test_df.empty:
        raise DataValidationError("Empty train or test partition after hash split.")

    raw_numeric_features = [
        c for c in full_df.columns if c not in {id_col, cfg.data.target_col, "y_m", "partition"}
    ]
    if cfg.data.deduplicate_features:
        tr_rows = train_df[raw_numeric_features].astype(float).itertuples(index=False, name=None)
        te_rows = test_df[raw_numeric_features].astype(float).itertuples(index=False, name=None)
        tr_sigs = set(map(tuple, tr_rows))
        te_sigs = set(map(tuple, te_rows))
        overlap_cnt = len(tr_sigs & te_sigs)
        if overlap_cnt > 0:
            raise DataValidationError(
                f"Detected {overlap_cnt} duplicate feature vectors leaking across train/test."
            )

    duplicates_removed = raw_row_count - int(len(full_df))
    audit_summary = {
        "data_uri": data_uri,
        "data_sha256": data_sha256,
        "raw_rows": raw_row_count,
        "deduplicate_features": cfg.data.deduplicate_features,
        "duplicates_removed": duplicates_removed,
        "schema": schema_stats,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "raw_numeric_features": raw_numeric_features,
        "sql_audit": audit_results,
        "partition_counts": part_results.get("partition_counts", []),
    }
    return train_df, test_df, audit_summary
