"""
preprocessing.py — Robust ingestion and preprocessing for heterogeneous
lipidomics / metabolomics exports (MSDIAL, LipidSearch, Lipotype, etc.).

This module sits between file I/O and the existing analysis pipeline.
It auto-detects column roles, strips metadata, validates abundance data,
parses diverse lipid names, and produces the canonical DataFrame contracts
expected by all downstream functions.

Output contract (identical to legacy pipeline):
    df_raw  : columns = ["Sample Name", <abundance cols …>]
    df_meta : columns = ["Sample Name", "Head Group", "Head Group 2",
                          "Acyl Chain Length", "Unsaturation", "Unsaturation 2"]
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_BLACKLIST = {
    "average rt(min)", "average mz", "adduct type", "formula",
    "ontology", "inchikey", "smiles", "ms/ms spectrum",
    "alignment id", "comment", "reference rt", "reference m/z",
    "total score", "s/n average", "msms spectrum",
    "post curation result", "fill %", "dot product",
    "reverse dot product", "fragment presence %",
    "spectrum reference file name", "ms1 isotopic spectrum",
    "ms/ms assigned", "quantmass", "rt left(min)", "rt right(min)",
}

LIPID_ID_HINTS = [
    "metabolite name", "lipid species", "lipid name",
    "compound name", "shorthand notation", "sample name",
]

NON_LIPID_DENYLIST = {
    "vitamin", "unknown", "metabolite", "adenosine",
    "cholecalciferol", "w/o", "could not", "unparsed",
    "unidentified", "noise",
    # Chemical contaminants / reagents / non-biological compounds
    "cyclopentasiloxane", "cyclohexasiloxane", "phthalate",
    "dioctyl", "elaidylphosphocholine", "coq",
    "docosenamide", "siloxane",
}


# ---------------------------------------------------------------------------
# PreprocessReport — structured validation summary
# ---------------------------------------------------------------------------

@dataclass
class PreprocessReport:
    """Collects all warnings / diagnostics from a preprocessing run."""
    detected_lipid_column: str = ""
    detected_sample_columns: List[str] = field(default_factory=list)
    removed_metadata_columns: List[str] = field(default_factory=list)
    dropped_nonlipid_rows: int = 0
    dropped_nonlipid_categories: List[str] = field(default_factory=list)
    dropped_allzero_columns: List[str] = field(default_factory=list)
    dropped_nonnumeric_columns: List[str] = field(default_factory=list)
    dropped_duplicate_columns: List[str] = field(default_factory=list)
    failed_numeric_coercions: int = 0
    failed_lipid_parses: int = 0
    duplicates_collapsed: int = 0
    inferred_cohorts: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

    def summary_lines(self) -> List[str]:
        """Return human-readable summary lines for UI display."""
        lines = []
        if self.detected_lipid_column:
            lines.append(f"Lipid ID column: {self.detected_lipid_column}")
        if self.detected_sample_columns:
            lines.append(f"Sample/abundance columns detected: "
                         f"{len(self.detected_sample_columns)}")
        if self.removed_metadata_columns:
            lines.append(f"Metadata columns removed: "
                         f"{len(self.removed_metadata_columns)} "
                         f"({', '.join(self.removed_metadata_columns[:5])}"
                         f"{'…' if len(self.removed_metadata_columns) > 5 else ''})")
        if self.dropped_allzero_columns:
            lines.append(f"All-zero columns dropped: "
                         f"{len(self.dropped_allzero_columns)}")
        if self.dropped_nonnumeric_columns:
            lines.append(f"Non-numeric columns dropped: "
                         f"{len(self.dropped_nonnumeric_columns)}")
        if self.dropped_duplicate_columns:
            lines.append(f"Duplicate columns dropped: "
                         f"{len(self.dropped_duplicate_columns)}")
        if self.failed_numeric_coercions:
            lines.append(f"Cells failed numeric coercion (set to 0): "
                         f"{self.failed_numeric_coercions}")
        if self.failed_lipid_parses:
            lines.append(f"Lipid names that could not be parsed: "
                         f"{self.failed_lipid_parses}")
        if self.duplicates_collapsed > 0:
            lines.append(f"Duplicate lipid rows consolidated (sum): "
                         f"{self.duplicates_collapsed}")
        if self.dropped_nonlipid_rows:
            cats = ", ".join(self.dropped_nonlipid_categories[:8])
            lines.append(f"Non-lipid rows removed: "
                         f"{self.dropped_nonlipid_rows} ({cats})")
        for w in self.warnings:
            lines.append(f"⚠ {w}")
        return lines


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def detect_lipid_id_column(
    df: pd.DataFrame,
    user_hint: Optional[str] = None,
) -> str:
    """
    Return the column name most likely holding lipid identifiers.

    Priority:
      1. Exact match to *user_hint* (if provided and present).
      2. Case-insensitive substring match against LIPID_ID_HINTS.
      3. Fallback: first column.
    """
    # 1. User-supplied hint
    if user_hint and str(user_hint).strip():
        hint = str(user_hint).strip()
        if hint in df.columns:
            return hint
        # Case-insensitive fallback
        for col in df.columns:
            if col.lower().strip() == hint.lower().strip():
                return col

    # 2. Scan known hint strings
    lowered = {c: c.lower().strip() for c in df.columns}
    for hint_str in LIPID_ID_HINTS:
        for col, low in lowered.items():
            if hint_str in low:
                return col

    # 3. Fallback: first column
    return df.columns[0]


def detect_abundance_columns(
    df: pd.DataFrame,
    lipid_col: str,
    report: Optional[PreprocessReport] = None,
) -> List[str]:
    """
    Return column names that are numeric abundance data.

    Excludes:
      - The lipid identifier column.
      - Columns whose name (lowered) is in METADATA_BLACKLIST.
      - Columns where >80% of values fail pd.to_numeric coercion.
    """
    abundance = []
    report = report or PreprocessReport()

    for col in df.columns:
        if col == lipid_col:
            continue

        # Blacklist check (case-insensitive)
        if col.lower().strip() in METADATA_BLACKLIST:
            report.removed_metadata_columns.append(col)
            continue

        # Numeric viability check
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        valid_frac = numeric_series.notna().mean()
        if valid_frac < 0.20:
            report.dropped_nonnumeric_columns.append(col)
            continue

        abundance.append(col)

    report.detected_sample_columns = list(abundance)
    return abundance


# ---------------------------------------------------------------------------
# Abundance validation
# ---------------------------------------------------------------------------

def validate_abundance_df(
    df: pd.DataFrame,
    abundance_cols: List[str],
    report: Optional[PreprocessReport] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Coerce abundance columns to numeric, drop invalid columns, return cleaned df.

    Returns (df_clean, kept_cols).
    """
    report = report or PreprocessReport()
    df = df.copy()
    kept = []

    for col in abundance_cols:
        original = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Count coercion failures
        n_failed = int(original.notna().sum() - df[col].notna().sum())
        report.failed_numeric_coercions += n_failed

        df[col] = df[col].fillna(0)

        # Drop all-zero columns
        if (df[col] == 0).all():
            report.dropped_allzero_columns.append(col)
            continue

        kept.append(col)

    # Detect exact-duplicate columns (same name after dedup, same values)
    seen_data = {}
    final_kept = []
    for col in kept:
        col_hash = pd.util.hash_pandas_object(df[col]).sum()
        if col_hash in seen_data and col in seen_data[col_hash]:
            report.dropped_duplicate_columns.append(col)
            continue
        seen_data.setdefault(col_hash, set()).add(col)
        final_kept.append(col)

    return df, final_kept


# ---------------------------------------------------------------------------
# Lipid name parsing — 4-tier regex cascade
# ---------------------------------------------------------------------------

# Pre-compiled patterns (most-specific first)
_RE_TIER1_CER = re.compile(
    r"^(?P<hg>[A-Za-z0-9_]+(?:Cer|HexCer|Hex2Cer|LacCer|ASG|AHexCer|Gb3))"
    r"\s*[\(\s]*[dteOP-]*"
    r"(?P<cl>\d+):(?P<us>\d+)",
    re.IGNORECASE,
)

_RE_TIER1_STD = re.compile(
    r"^(?P<hg>[A-Za-z]+)\s+"
    r"(?:[OP]-?)?"
    r"(?P<cl>\d+):(?P<us>\d+)",
)

_RE_TIER2_LOOSE = re.compile(
    r"^(?P<hg>[A-Za-z]+).*?"
    r"(?P<cl>\d+):(?P<us>\d+)",
)

_RE_TIER3_HG_ONLY = re.compile(
    r"^(?P<hg>[A-Za-z][A-Za-z0-9_]*)",
)


def _normalise_head_group(hg: str) -> str:
    """
    Produce 'Head Group 2' — normalised lipid class.

    - Strip O/P plasmalogen suffix
    - Strip trailing digits (GD1 → GD)
    - Normalise HexCer variants
    """
    hg2 = hg.strip()
    hg2 = re.sub(r"\s+[OP]$", "", hg2).strip()
    hg2 = re.sub(r"\d+$", "", hg2).strip()
    hg2 = hg2.replace("HexCer", "Hex_Cer")
    return hg2 if hg2 else "Unparsed"


def parse_lipid_name(name: str) -> dict:
    """
    Parse a lipid name string into head group, chain length, unsaturation.

    Uses a 4-tier regex cascade. Never returns None for any field.

    Returns
    -------
    dict with keys: head_group, head_group_2, chain_length, unsaturation,
                    unsaturation_2
    """
    name = str(name).strip()
    hg, cl, us = "Unparsed", 0, 0

    # Tier 1a: Ceramide / sphingolipid / glyco-lipid class names
    m = _RE_TIER1_CER.match(name)
    if m:
        hg = m.group("hg")
        cl = int(m.group("cl"))
        us = int(m.group("us"))
    else:
        # Tier 1b: Standard lipid  (PC 34:1, PE O-36:2, SM 42:2;O)
        m = _RE_TIER1_STD.match(name)
        if m:
            hg = m.group("hg")
            cl = int(m.group("cl"))
            us = int(m.group("us"))
        else:
            # Tier 2: Loose — head-group anywhere + chain:unsat anywhere
            m = _RE_TIER2_LOOSE.match(name)
            if m:
                hg = m.group("hg")
                cl = int(m.group("cl"))
                us = int(m.group("us"))
            else:
                # Tier 3: Head-group only (no chain info)
                m = _RE_TIER3_HG_ONLY.match(name)
                if m:
                    hg = m.group("hg")
                # else: stays "Unparsed"

    hg2 = _normalise_head_group(hg)
    us2 = str(us) if us < 3 else ">=3"

    return {
        "head_group": hg,
        "head_group_2": hg2,
        "chain_length": cl,
        "unsaturation": us,
        "unsaturation_2": us2,
    }


# ---------------------------------------------------------------------------
# Metadata extraction (robust replacement for analysis.extract_metadata)
# ---------------------------------------------------------------------------

def extract_metadata_robust(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'Sample Name' column using the robust lipid parser.

    Returns DataFrame with identical columns to the legacy extract_metadata():
        Sample Name | Head Group | Head Group 2 |
        Acyl Chain Length | Unsaturation | Unsaturation 2
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    names = df_raw["Sample Name"].tolist()
    records = []
    n_failed = 0

    for name in names:
        parsed = parse_lipid_name(name)
        if parsed["head_group"] == "Unparsed":
            n_failed += 1
        records.append({
            "Sample Name":      str(name).strip(),
            "Head Group":       parsed["head_group"],
            "Head Group 2":     parsed["head_group_2"],
            "Acyl Chain Length": parsed["chain_length"],
            "Unsaturation":     parsed["unsaturation"],
            "Unsaturation 2":   parsed["unsaturation_2"],
        })

    if n_failed:
        logger.info("Lipid parser: %d / %d names could not be fully parsed "
                     "(marked as 'Unparsed')", n_failed, len(names))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Ontology / non-lipid filtering
# ---------------------------------------------------------------------------

def filter_non_lipids(
    df_raw: pd.DataFrame,
    df_meta: pd.DataFrame,
    ontology_series: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Remove non-lipid rows based on parsed head groups and optional ontology data.

    Parameters
    ----------
    ontology_series : optional Series aligned with df_meta index, containing
                      the ontology annotation from the original raw file.

    Returns
    -------
    (df_raw_filtered, df_meta_filtered, report_dict)
    where report_dict = {"removed_count": N, "removed_categories": [...]}
    """
    if df_raw.empty or df_meta.empty:
        return df_raw, df_meta, {"removed_count": 0, "removed_categories": []}

    mask = pd.Series(True, index=df_meta.index)

    # 1. Head-group deny-list check
    hg_lower = df_meta["Head Group"].str.lower().str.strip()
    for deny in NON_LIPID_DENYLIST:
        mask &= ~hg_lower.str.contains(deny, na=False)

    # 2. Ontology column check (if available)
    if ontology_series is not None and not ontology_series.empty:
        ont_lower = ontology_series.str.lower().str.strip()
        # Keep rows where ontology is NaN (unknown) or contains lipid-related terms
        lipid_terms = ["lipid", "sphingo", "glycero", "sterol", "fatty"]
        ont_is_lipid = ont_lower.isna()
        for term in lipid_terms:
            ont_is_lipid = ont_is_lipid | ont_lower.str.contains(term, na=False)
        mask &= ont_is_lipid

    removed_meta = df_meta[~mask]
    removed_cats = sorted(removed_meta["Head Group"].unique().tolist())
    removed_count = int((~mask).sum())

    keep_names = set(df_meta.loc[mask, "Sample Name"])
    df_raw_f = df_raw[df_raw["Sample Name"].isin(keep_names)].copy()
    df_meta_f = df_meta[mask].copy()

    return df_raw_f, df_meta_f, {
        "removed_count": removed_count,
        "removed_categories": removed_cats,
    }


# ---------------------------------------------------------------------------
# Heuristic cohort inference from sample column names
# ---------------------------------------------------------------------------

# Common instrument prefixes to strip
_INSTRUMENT_PREFIXES = re.compile(
    r"^(?:Neg|Pos)(?:MSMS(?:ALL)?|MS1|DDA)?[-_]",
    re.IGNORECASE,
)

# Trailing replicate suffixes:  _A, _B, -1, -2, -01, -02, -03
_REPLICATE_SUFFIX = re.compile(
    r"[-_]\s*(?:[A-Za-z]|\d{1,3})\s*$"
)


def heuristic_cohort_from_name(col_name: str) -> Tuple[str, str]:
    """
    Infer (mutation/cohort, replicate) from a sample column name.

    Strips:
      1. Known instrument prefixes (NegMSMSALL-, PosMSMSALL-, etc.)
      2. Trailing replicate suffixes (_A, _B, -1, -2, -01, etc.)

    Returns
    -------
    (mutation: str, replicate: str)
    """
    s = str(col_name).strip()

    # Strip deduplication suffix (.1, .2) added by pandas/our dedup
    base = re.sub(r"\.\d+$", "", s)

    # Strip instrument prefix
    base = _INSTRUMENT_PREFIXES.sub("", base)

    # Extract replicate suffix
    m = _REPLICATE_SUFFIX.search(base)
    if m:
        replicate = m.group(0).lstrip("-_").strip()
        mutation = base[:m.start()].strip()
    else:
        mutation = base
        replicate = "1"

    # If mutation is empty after stripping, use the full name
    if not mutation:
        mutation = base if base else s
        replicate = "1"

    return mutation, replicate


# ---------------------------------------------------------------------------
# Multi-level aggregation
# ---------------------------------------------------------------------------

def aggregate_by_hierarchy(
    df_p: pd.DataFrame,
    df_exps: pd.DataFrame,
    levels: Sequence[str] = ("Mutation",),
    method: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate per-experiment data across specified hierarchy levels.

    When called with levels=["Mutation"] this produces identical output to
    the legacy aggregate_by_cohort().

    Parameters
    ----------
    df_p     : DataFrame with 'Sample Name' col + one col per experiment
               (columns may be repeated mutation names = technical replicates)
    df_exps  : DataFrame from extract_experiments()
    levels   : Sequence of column names in df_exps to group by
    method   : 'mean' | 'median' | 'sum'

    Returns
    -------
    df_agg : DataFrame  index='Sample Name', cols=unique group labels
    """
    if df_p.empty:
        return pd.DataFrame()

    # For the standard single-level case (most common), use the
    # proven aggregation logic from the legacy pipeline.
    if len(levels) == 1 and levels[0] == "Mutation":
        mutations = df_exps["Mutation"].unique().tolist()
        agg = {}
        for mtn in mutations:
            cols = [c for c in df_p.columns if c == mtn]
            if not cols:
                continue
            sub = df_p[cols]
            if method == "median":
                agg[mtn] = sub.median(axis=1)
            elif method == "sum":
                agg[mtn] = sub.sum(axis=1)
            else:
                agg[mtn] = sub.mean(axis=1)

        df_agg = pd.DataFrame(agg)
        df_agg.index = df_p["Sample Name"].values
        df_agg.index.name = "Sample Name"
        return df_agg

    # Multi-level aggregation (future use)
    # Build mapping: experiment → group label (concatenated hierarchy)
    group_cols = [l for l in levels if l in df_exps.columns]
    if not group_cols:
        group_cols = ["Mutation"]

    exp_to_group = {}
    for _, row in df_exps.iterrows():
        label = " | ".join(str(row.get(g, "")) for g in group_cols)
        exp_to_group[row["Exp"]] = label

    groups = sorted(set(exp_to_group.values()))
    agg = {}
    for grp in groups:
        member_exps = [e for e, g in exp_to_group.items() if g == grp]
        member_cols = [c for c in df_p.columns
                       if c != "Sample Name" and c in member_exps]
        if not member_cols:
            continue
        sub = df_p[member_cols]
        if method == "median":
            agg[grp] = sub.median(axis=1)
        elif method == "sum":
            agg[grp] = sub.sum(axis=1)
        else:
            agg[grp] = sub.mean(axis=1)

    df_agg = pd.DataFrame(agg)
    df_agg.index = df_p["Sample Name"].values
    df_agg.index.name = "Sample Name"
    return df_agg


# ---------------------------------------------------------------------------
# Central preprocessing function
# ---------------------------------------------------------------------------

def preprocess_raw_metabolomics_export(
    file_infos,
    *,
    sheet_name: Union[int, str] = 0,
    idx_col: Optional[str] = None,
    num_idx: int = 0,
    skip_rows: int = 0,
) -> Tuple[pd.DataFrame, PreprocessReport]:
    """
    Load raw CSV/XLS/XLSX exports, auto-detect columns, strip metadata,
    validate abundance data.

    Returns
    -------
    (df_clean, report) where df_clean has:
        columns = ["Sample Name", <abundance columns only>]
        all abundance columns are numeric (float64), NaN filled with 0.
    """
    # Lazy import to avoid circular dependency
    try:
        from app.analysis import load_file_any_format, deduplicate_columns
    except ImportError:
        from analysis import load_file_any_format, deduplicate_columns

    report = PreprocessReport()

    dfs = []
    for fi in file_infos:
        path = fi["datapath"] if isinstance(fi, dict) else fi
        try:
            df = load_file_any_format(path, skip_rows=skip_rows,
                                       sheet_name=sheet_name)
        except Exception as e:
            report.add_warning(f"Failed to read file {path}: {e}")
            continue

        if df.empty:
            report.add_warning(f"File {path} produced an empty DataFrame")
            continue

        # --- Detect lipid ID column ---
        lipid_col = detect_lipid_id_column(df, user_hint=idx_col)
        report.detected_lipid_column = lipid_col

        # --- Drop leading metadata columns (num_idx) ---
        if num_idx > 0:
            cols_to_drop = list(df.columns[:num_idx])
            # Preserve the lipid column even if it's in the drop range
            if lipid_col in cols_to_drop:
                cols_to_drop.remove(lipid_col)
            dropped_meta = [c for c in cols_to_drop]
            report.removed_metadata_columns.extend(dropped_meta)
            df = df.drop(columns=cols_to_drop, errors="ignore")

        # --- Auto-detect abundance vs metadata columns ---
        abundance_cols = detect_abundance_columns(df, lipid_col, report)

        # --- Rename lipid column → "Sample Name" ---
        df = df.rename(columns={lipid_col: "Sample Name"})

        # --- Keep only Sample Name + abundance columns ---
        keep_cols = ["Sample Name"] + abundance_cols
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        dfs.append(df)

    if not dfs:
        report.add_warning("No valid data files could be loaded")
        return pd.DataFrame(), report

    # Combine files
    combined = pd.concat(dfs, axis=0, ignore_index=True)

    # Deduplicate column names (preserves the intentional duplicate behavior)
    combined.columns = deduplicate_columns(list(combined.columns))

    # Validate and coerce abundance columns
    abundance_cols = [c for c in combined.columns if c != "Sample Name"]
    combined, kept_cols = validate_abundance_df(combined, abundance_cols, report)

    # Final output: Sample Name + validated abundance columns only
    result = combined[["Sample Name"] + kept_cols].copy()

    # --- Consolidate duplicate lipid species (sum abundances) ---
    n_before = len(result)
    num_cols = [c for c in result.columns if c != "Sample Name"]
    result = result.groupby("Sample Name", as_index=False)[num_cols].sum()
    n_collapsed = n_before - len(result)
    if n_collapsed > 0:
        report.duplicates_collapsed = n_collapsed

    return result, report
