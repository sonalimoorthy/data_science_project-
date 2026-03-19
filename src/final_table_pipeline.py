"""
Create final Country-Year table from cleaned interim datasets.

Outputs written under data/final:
- master_country_year.csv
- master_country_year.parquet
- merge_diagnostics.json
- final_table_schema.json
- security_scan_report.json
- evidence_ledger.jsonl (append-only)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

import pandas as pd


FINAL_COLUMNS = [
    "Country",
    "Year",
    "Region",
    "Num_Attacks",
    "Fatalities",
    "Injuries",
    "Attack_Type_Mode",
    "Target_Type_Mode",
    "Perpetrator_Mode",
    "Conflict",
    "Conflict_Level",
    "Conflict_Intensity_deaths",
    "Conflict_Region",
    "Population",
    "GDP_per_capita",
    "has_conflict_data",
    "has_population_data",
    "has_gdp_data",
    "country_mapping_applied",
]

STRING_COLUMNS = [
    "Country",
    "Region",
    "Attack_Type_Mode",
    "Target_Type_Mode",
    "Perpetrator_Mode",
    "Conflict_Region",
]


@dataclass(frozen=True)
class Paths:
    gtd_country_year: Path
    conflict_cleaned: Path
    population_cleaned: Path
    gdp_cleaned: Path
    country_mapping: Path
    schema_source: Path
    output_csv: Path
    output_parquet: Path
    diagnostics_json: Path
    security_scan_report_json: Path
    output_schema_json: Path
    evidence_ledger_jsonl: Path


def resolve_paths(project_root: Path) -> Paths:
    interim = project_root / "data" / "interim"
    final_dir = project_root / "data" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        gtd_country_year=interim / "gtd_dataclean" / "gtd_country_year.csv",
        conflict_cleaned=interim / "conflict_dataclean" / "conflict_cleaned.csv",
        population_cleaned=interim / "population+gdp_dataclean" / "population_cleaned.csv",
        gdp_cleaned=interim / "population+gdp_dataclean" / "gdp_cleaned.csv",
        country_mapping=interim / "country_name_mapping.csv",
        schema_source=project_root / "src" / "schemas" / "final_table_schema.json",
        output_csv=final_dir / "master_country_year.csv",
        output_parquet=final_dir / "master_country_year.parquet",
        diagnostics_json=final_dir / "merge_diagnostics.json",
        security_scan_report_json=final_dir / "security_scan_report.json",
        output_schema_json=final_dir / "final_table_schema.json",
        evidence_ledger_jsonl=final_dir / "evidence_ledger.jsonl",
    )


def normalize_country(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def safe_mode(series: pd.Series) -> object:
    cleaned = series.dropna()
    if cleaned.empty:
        return pd.NA
    modes = cleaned.mode(dropna=True)
    if modes.empty:
        return pd.NA
    return modes.iloc[0]


def compute_file_hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(data: object) -> str:
    return sha256(json.dumps(data, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def load_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing mapping file: {path}. Create data/interim/country_name_mapping.csv first."
        )

    mapping = pd.read_csv(path, low_memory=False)
    required = {"source_country", "canonical_country", "source_dataset", "active_flag"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"country_name_mapping.csv missing required columns: {sorted(missing)}")

    mapping = mapping.copy()
    mapping["source_country"] = mapping["source_country"].astype("string").str.strip()
    mapping["canonical_country"] = mapping["canonical_country"].astype("string").str.strip()
    mapping["source_dataset"] = mapping["source_dataset"].astype("string").str.strip().str.casefold()
    mapping["active_flag"] = pd.to_numeric(mapping["active_flag"], errors="coerce").fillna(0).astype("int64")
    mapping = mapping[mapping["active_flag"] == 1].copy()
    mapping["source_country_key"] = mapping["source_country"].map(normalize_country)
    return mapping


def apply_country_mapping(df: pd.DataFrame, mapping: pd.DataFrame, source_dataset: str) -> pd.DataFrame:
    out = df.copy()
    out["Country"] = out["Country"].astype("string").str.strip()
    out["Country_key"] = out["Country"].map(normalize_country)

    source_map = mapping[mapping["source_dataset"] == source_dataset.casefold()][
        ["source_country_key", "canonical_country"]
    ].drop_duplicates(subset=["source_country_key"], keep="first")

    out = out.merge(
        source_map,
        how="left",
        left_on="Country_key",
        right_on="source_country_key",
    )
    out["Country_std"] = out["canonical_country"].fillna(out["Country"])
    out["mapping_applied"] = out["canonical_country"].notna().astype("int64")
    return out.drop(columns=["Country_key", "source_country_key", "canonical_country"])


def aggregate_gtd(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Country_std", "Year"], as_index=False)
        .agg(
            Region=("Region", safe_mode),
            Num_Attacks=("Num_Attacks", "sum"),
            Fatalities=("Fatalities", "sum"),
            Injuries=("Injuries", "sum"),
            Attack_Type_Mode=("Attack_Type_Mode", safe_mode),
            Target_Type_Mode=("Target_Type_Mode", safe_mode),
            Perpetrator_Mode=("Perpetrator_Mode", safe_mode),
            country_mapping_applied=("mapping_applied", "max"),
        )
        .sort_values(["Country_std", "Year"], ignore_index=True)
    )


def aggregate_conflict(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Country_std", "Year"], as_index=False)
        .agg(
            Conflict_Region=("Region", safe_mode),
            Conflict=("Conflict", "max"),
            Conflict_Level=("Conflict_Level", "max"),
            Conflict_Intensity_deaths=("Conflict_Intensity_deaths", "sum"),
        )
        .sort_values(["Country_std", "Year"], ignore_index=True)
    )


def aggregate_single_value(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        df.groupby(["Country_std", "Year"], as_index=False)
        .agg(**{value_column: (value_column, "first")})
        .sort_values(["Country_std", "Year"], ignore_index=True)
    )


def load_and_prepare(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gtd = pd.read_csv(paths.gtd_country_year, low_memory=False)
    conflict = pd.read_csv(paths.conflict_cleaned, low_memory=False)
    population = pd.read_csv(paths.population_cleaned, low_memory=False)
    gdp = pd.read_csv(paths.gdp_cleaned, low_memory=False)
    mapping = load_mapping(paths.country_mapping)

    for frame in [gtd, conflict, population, gdp]:
        frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
        frame.dropna(subset=["Country", "Year"], inplace=True)
        frame["Year"] = frame["Year"].astype("int64")

    return gtd, conflict, population, gdp, mapping


def build_final_table(paths: Paths) -> tuple[pd.DataFrame, dict[str, object], str]:
    gtd, conflict, population, gdp, mapping = load_and_prepare(paths)

    gtd_mapped = apply_country_mapping(gtd, mapping, "gtd")
    conflict_mapped = apply_country_mapping(conflict, mapping, "conflict")
    population_mapped = apply_country_mapping(population, mapping, "population")
    gdp_mapped = apply_country_mapping(gdp, mapping, "gdp")

    gtd_base = aggregate_gtd(gtd_mapped)
    conflict_ready = aggregate_conflict(conflict_mapped)
    population_ready = aggregate_single_value(population_mapped, "Population")
    gdp_ready = aggregate_single_value(gdp_mapped, "GDP_per_capita")

    final_df = gtd_base.merge(conflict_ready, on=["Country_std", "Year"], how="left")
    final_df = final_df.merge(population_ready, on=["Country_std", "Year"], how="left")
    final_df = final_df.merge(gdp_ready, on=["Country_std", "Year"], how="left")

    final_df["Country"] = final_df["Country_std"]
    final_df["Conflict"] = final_df["Conflict"].fillna(0).astype("int64")
    final_df["Conflict_Level"] = final_df["Conflict_Level"].fillna(0).astype("int64")
    final_df["Conflict_Intensity_deaths"] = final_df["Conflict_Intensity_deaths"].fillna(0.0)
    final_df["has_conflict_data"] = final_df["Conflict_Region"].notna().astype("int64")
    final_df["has_population_data"] = final_df["Population"].notna().astype("int64")
    final_df["has_gdp_data"] = final_df["GDP_per_capita"].notna().astype("int64")
    final_df["country_mapping_applied"] = final_df["country_mapping_applied"].fillna(0).astype("int64")
    final_df = final_df.drop(columns=["Country_std"])

    for col in STRING_COLUMNS:
        final_df[col] = final_df[col].astype("string")

    final_df = final_df[FINAL_COLUMNS].sort_values(["Country", "Year"], ignore_index=True)
    duplicate_keys = int(final_df.duplicated(subset=["Country", "Year"]).sum())
    if duplicate_keys:
        raise ValueError(f"Duplicate Country-Year keys found: {duplicate_keys}")

    diagnostics = {
        "row_count": int(len(final_df)),
        "distinct_countries": int(final_df["Country"].nunique()),
        "year_min": int(final_df["Year"].min()),
        "year_max": int(final_df["Year"].max()),
        "duplicate_country_year_keys": duplicate_keys,
        "match_rates": {
            "conflict": float(final_df["has_conflict_data"].mean()),
            "population": float(final_df["has_population_data"].mean()),
            "gdp": float(final_df["has_gdp_data"].mean()),
        },
    }

    diff_hash_inputs = [
        paths.gtd_country_year,
        paths.conflict_cleaned,
        paths.population_cleaned,
        paths.gdp_cleaned,
        paths.country_mapping,
    ]
    diff_hash = sha256("".join(compute_file_hash(p) for p in diff_hash_inputs).encode("utf-8")).hexdigest()

    return final_df, diagnostics, diff_hash


def validate_schema(df: pd.DataFrame, schema_source: Path) -> dict[str, object]:
    schema = json.loads(schema_source.read_text(encoding="utf-8"))
    if list(df.columns) != schema["column_order"]:
        raise ValueError("Column order mismatch vs schema.")

    for required_col in schema["required_columns"]:
        if required_col not in df.columns:
            raise ValueError(f"Missing required column: {required_col}")

    for col, dtype in schema["dtypes"].items():
        if dtype == "string" and not pd.api.types.is_string_dtype(df[col]):
            raise ValueError(f"Column {col} expected string dtype.")
        if dtype == "int64" and not pd.api.types.is_integer_dtype(df[col]):
            raise ValueError(f"Column {col} expected int dtype.")
        if dtype == "float64" and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} expected numeric dtype.")

    return schema


def run_security_scan(project_root: Path) -> dict[str, object]:
    secret_patterns = [
        ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
        ("google_api_key", re.compile(r"AIza[0-9A-Za-z\\-_]{35}")),
        ("github_pat", re.compile(r"ghp_[0-9A-Za-z]{36}")),
        ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ]

    dangerous_patterns = [
        ("DROP TABLE", re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE)),
        ("DELETE FROM without WHERE", re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE)),
        ("auth middleware disable/bypass", re.compile(r"(disable|bypass).{0,20}(auth|middleware)", re.IGNORECASE)),
        ("iam wildcard permissions", re.compile(r"\"Action\"\s*:\s*\"\\*\"|\"Resource\"\s*:\s*\"\\*\"", re.IGNORECASE)),
        ("logging headers/cookies/tokens", re.compile(r"(print|log).{0,40}(header|cookie|token|authorization)", re.IGNORECASE)),
    ]

    findings: list[dict[str, object]] = []
    scan_files = list((project_root / "src").glob("*.py"))
    # Exclude this scanner file itself to avoid literal-pattern self-matches.
    scan_files = [p for p in scan_files if p.name != "final_table_pipeline.py"]

    for file_path in scan_files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            for label, pattern in secret_patterns:
                if pattern.search(line):
                    findings.append(
                        {
                            "type": "secret_pattern",
                            "reason": label,
                            "file": str(file_path),
                            "line": i,
                            "content": line.strip()[:240],
                        }
                    )
            for label, pattern in dangerous_patterns:
                if pattern.search(line):
                    if label == "DELETE FROM without WHERE" and "WHERE" in line.upper():
                        continue
                    findings.append(
                        {
                            "type": "dangerous_diff",
                            "reason": label,
                            "file": str(file_path),
                            "line": i,
                            "content": line.strip()[:240],
                        }
                    )

    return {"blocked": bool(findings), "findings": findings}


def write_outputs(paths: Paths, final_df: pd.DataFrame, diagnostics: dict[str, object], schema: dict[str, object]) -> None:
    final_df.to_csv(paths.output_csv, index=False)
    final_df.to_parquet(paths.output_parquet, index=False)
    paths.diagnostics_json.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=True), encoding="utf-8")
    paths.output_schema_json.write_text(json.dumps(schema, indent=2, ensure_ascii=True), encoding="utf-8")


def append_evidence(
    paths: Paths,
    diff_hash: str,
    diagnostics: dict[str, object],
    security_scan_report: dict[str, object],
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": "cursor_agent",
        "agentRole": "BUILDER",
        "actionType": "build_final_table",
        "resourcesTouched": [
            str(paths.gtd_country_year),
            str(paths.conflict_cleaned),
            str(paths.population_cleaned),
            str(paths.gdp_cleaned),
            str(paths.country_mapping),
            str(paths.output_csv),
            str(paths.output_parquet),
            str(paths.diagnostics_json),
            str(paths.output_schema_json),
            str(paths.security_scan_report_json),
        ],
        "diffHash": diff_hash,
        "testHashes": {
            "mergeDiagnosticsHash": stable_hash(diagnostics),
            "securityScanHash": stable_hash(security_scan_report),
        },
        "approvals": [],
    }
    with paths.evidence_ledger_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(project_root)

    final_df, diagnostics, diff_hash = build_final_table(paths)
    schema = validate_schema(final_df, paths.schema_source)
    security_scan_report = run_security_scan(project_root)
    paths.security_scan_report_json.write_text(
        json.dumps(security_scan_report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if security_scan_report["blocked"]:
        raise RuntimeError(
            "Security scan blocked pipeline. See data/final/security_scan_report.json"
        )

    write_outputs(paths, final_df, diagnostics, schema)
    append_evidence(paths, diff_hash, diagnostics, security_scan_report)

    print("Final table created successfully.")
    print(f"Rows: {len(final_df)}")
    print(f"CSV: {paths.output_csv}")
    print(f"Parquet: {paths.output_parquet}")


if __name__ == "__main__":
    main()
