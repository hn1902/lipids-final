"""
tests/test_preprocessing.py
Test suite for the robust preprocessing pipeline.
Run with: python -m pytest tests/test_preprocessing.py -v
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.preprocessing import (
    detect_lipid_id_column,
    detect_abundance_columns,
    validate_abundance_df,
    parse_lipid_name,
    extract_metadata_robust,
    filter_non_lipids,
    heuristic_cohort_from_name,
    aggregate_by_hierarchy,
    preprocess_raw_metabolomics_export,
    PreprocessReport,
    METADATA_BLACKLIST,
)
from app.analysis import (
    extract_experiments,
    make_df_p,
    aggregate_by_cohort,
    extract_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_with_metadata():
    """DataFrame mixing lipid ID, metadata, and abundance columns."""
    return pd.DataFrame({
        "Metabolite name": ["PC 34:1", "PE 36:2", "SM 18:1"],
        "Average Rt(min)": ["5.2", "6.1", "4.8"],
        "Formula": ["C42H82NO8P", "C41H78NO8P", "C39H79N2O6P"],
        "Ontology": ["Lipid", "Lipid", "Lipid"],
        "Sample_A": ["100", "200", "50"],
        "Sample_B": ["110", "190", "55"],
        "Sample_C": ["80", "220", "60"],
    })


@pytest.fixture
def simple_df_raw():
    """Clean raw DataFrame matching the downstream contract."""
    return pd.DataFrame({
        "Sample Name": ["PC 34:1", "PE 36:2", "SM 18:1"],
        "CAS9_A":      [100.0, 200.0, 50.0],
        "CAS9_B":      [110.0, 190.0, 55.0],
        "WT_A":        [80.0,  220.0, 60.0],
        "WT_B":        [90.0,  210.0, 45.0],
    })


@pytest.fixture
def simple_df_exps():
    return pd.DataFrame({
        "Exp":      ["CAS9_A", "CAS9_B", "WT_A", "WT_B"],
        "Mutation": ["CAS9",   "CAS9",   "WT",   "WT"],
        "Replicate":["A",      "B",      "A",    "B"],
    })


# ---------------------------------------------------------------------------
# 1. detect_lipid_id_column
# ---------------------------------------------------------------------------

def test_detect_lipid_id_column_metabolite_name(df_with_metadata):
    assert detect_lipid_id_column(df_with_metadata) == "Metabolite name"


def test_detect_lipid_id_column_user_hint(df_with_metadata):
    assert detect_lipid_id_column(df_with_metadata, user_hint="Metabolite name") == "Metabolite name"


def test_detect_lipid_id_column_fallback():
    df = pd.DataFrame({"ColA": [1, 2], "ColB": [3, 4]})
    assert detect_lipid_id_column(df) == "ColA"


def test_detect_lipid_id_column_case_insensitive():
    df = pd.DataFrame({"LIPID NAME": ["PC 34:1"], "val": [100]})
    assert detect_lipid_id_column(df) == "LIPID NAME"


# ---------------------------------------------------------------------------
# 2. detect_abundance_columns
# ---------------------------------------------------------------------------

def test_detect_abundance_columns_strips_metadata(df_with_metadata):
    lipid_col = "Metabolite name"
    report = PreprocessReport()
    abundance = detect_abundance_columns(df_with_metadata, lipid_col, report)
    # Should exclude Average Rt(min), Formula, Ontology
    assert "Average Rt(min)" not in abundance
    assert "Formula" not in abundance
    assert "Ontology" not in abundance
    assert "Sample_A" in abundance
    assert "Sample_B" in abundance
    assert "Sample_C" in abundance
    # Check report captured the metadata columns
    assert "Average Rt(min)" in report.removed_metadata_columns


def test_detect_abundance_columns_non_numeric():
    df = pd.DataFrame({
        "ID": ["PC 34:1", "PE 36:2"],
        "text_col": ["hello", "world"],
        "num_col": ["100", "200"],
    })
    report = PreprocessReport()
    abundance = detect_abundance_columns(df, "ID", report)
    assert "text_col" in report.dropped_nonnumeric_columns
    assert "num_col" in abundance


# ---------------------------------------------------------------------------
# 3. validate_abundance_df
# ---------------------------------------------------------------------------

def test_validate_abundance_df_coercion():
    df = pd.DataFrame({
        "Sample Name": ["A", "B"],
        "col1": ["100", "abc"],
        "col2": ["200", "300"],
    })
    report = PreprocessReport()
    df_out, kept = validate_abundance_df(df, ["col1", "col2"], report)
    assert report.failed_numeric_coercions >= 1
    assert df_out["col1"].iloc[1] == 0  # "abc" → NaN → 0


def test_validate_abundance_df_allzero_dropped():
    df = pd.DataFrame({
        "Sample Name": ["A", "B"],
        "zero_col": [0, 0],
        "good_col": [1, 2],
    })
    report = PreprocessReport()
    _, kept = validate_abundance_df(df, ["zero_col", "good_col"], report)
    assert "zero_col" in report.dropped_allzero_columns
    assert "good_col" in kept


def test_validate_abundance_df_duplicates():
    df = pd.DataFrame({
        "Sample Name": ["A", "B"],
        "col1": [100, 200],
        "col2": [100, 200],  # exact duplicate values
    })
    report = PreprocessReport()
    _, kept = validate_abundance_df(df, ["col1", "col2"], report)
    # Both should survive since they have different column names
    # (duplicate-column dedup is by hash + name, not just hash)
    assert "col1" in kept


# ---------------------------------------------------------------------------
# 4. parse_lipid_name
# ---------------------------------------------------------------------------

def test_parse_lipid_name_standard():
    r = parse_lipid_name("PC 34:1")
    assert r["head_group"] == "PC"
    assert r["chain_length"] == 34
    assert r["unsaturation"] == 1


def test_parse_lipid_name_ceramide():
    r = parse_lipid_name("Cer(d18:1/24:0)")
    assert r["head_group"] == "Cer"
    assert r["chain_length"] == 18
    assert r["unsaturation"] == 1


def test_parse_lipid_name_semicolon():
    r = parse_lipid_name("SM 42:2;O")
    assert r["head_group"] == "SM"
    assert r["chain_length"] == 42
    assert r["unsaturation"] == 2


def test_parse_lipid_name_nae():
    r = parse_lipid_name("NAE 13:1")
    assert r["head_group"] == "NAE"
    assert r["chain_length"] == 13
    assert r["unsaturation"] == 1


def test_parse_lipid_name_unparsable():
    r = parse_lipid_name("???")
    assert r["head_group"] == "Unparsed"
    assert r["chain_length"] == 0
    assert r["unsaturation"] == 0


def test_parse_lipid_name_plasmalogen():
    r = parse_lipid_name("PC O-34:1+HCOO")
    assert r["head_group"] == "PC"
    assert r["chain_length"] == 34
    assert r["unsaturation"] == 1


def test_parse_lipid_name_spb():
    r = parse_lipid_name("SPB 15:0;O2")
    assert r["head_group"] == "SPB"
    assert r["chain_length"] == 15
    assert r["unsaturation"] == 0


def test_parse_lipid_name_ahexcer():
    r = parse_lipid_name("AHexCer 42:1;O3")
    assert r["head_group"] == "AHexCer"
    assert r["chain_length"] == 42
    assert r["unsaturation"] == 1


def test_parse_lipid_name_hexcer():
    r = parse_lipid_name("HexCer d18:1/22:0")
    assert "Cer" in r["head_group"] or "Hex" in r["head_group"]
    assert r["chain_length"] == 18
    assert r["unsaturation"] == 1


def test_parse_lipid_name_unsaturation_2():
    r = parse_lipid_name("PC 34:5")
    assert r["unsaturation_2"] == ">=3"
    r2 = parse_lipid_name("PC 34:2")
    assert r2["unsaturation_2"] == "2"


def test_parse_lipid_name_head_group_only():
    r = parse_lipid_name("Cholesterol")
    assert r["head_group"] == "Cholesterol"
    assert r["chain_length"] == 0


# ---------------------------------------------------------------------------
# 5. extract_metadata_robust
# ---------------------------------------------------------------------------

def test_extract_metadata_robust_contract(simple_df_raw):
    meta = extract_metadata_robust(simple_df_raw)
    expected_cols = {"Sample Name", "Head Group", "Head Group 2",
                     "Acyl Chain Length", "Unsaturation", "Unsaturation 2"}
    assert expected_cols.issubset(set(meta.columns))
    assert len(meta) == len(simple_df_raw)


def test_extract_metadata_robust_empty():
    meta = extract_metadata_robust(pd.DataFrame())
    assert meta.empty


def test_extract_metadata_robust_matches_old(simple_df_raw):
    """The robust parser should produce equivalent results for simple names."""
    meta_new = extract_metadata_robust(simple_df_raw)
    # Check that PC 34:1 parses correctly
    pc_row = meta_new[meta_new["Sample Name"] == "PC 34:1"].iloc[0]
    assert pc_row["Head Group"] == "PC"
    assert pc_row["Acyl Chain Length"] == 34
    assert pc_row["Unsaturation"] == 1


# ---------------------------------------------------------------------------
# 6. filter_non_lipids
# ---------------------------------------------------------------------------

def test_filter_non_lipids_removes_vitamins():
    df_raw = pd.DataFrame({
        "Sample Name": ["PC 34:1", "25-hydroxycholecalciferol", "SM 18:1"],
        "s1": [100.0, 50.0, 80.0],
    })
    df_meta = extract_metadata_robust(df_raw)
    dr, dm, report = filter_non_lipids(df_raw, df_meta)
    assert report["removed_count"] >= 1
    assert "25-hydroxycholecalciferol" not in dr["Sample Name"].values


def test_filter_non_lipids_keeps_lipids():
    df_raw = pd.DataFrame({
        "Sample Name": ["PC 34:1", "PE 36:2", "SM 18:1"],
        "s1": [100.0, 200.0, 80.0],
    })
    df_meta = extract_metadata_robust(df_raw)
    dr, dm, report = filter_non_lipids(df_raw, df_meta)
    assert report["removed_count"] == 0
    assert len(dr) == 3


def test_filter_non_lipids_empty():
    dr, dm, report = filter_non_lipids(pd.DataFrame(), pd.DataFrame())
    assert report["removed_count"] == 0


# ---------------------------------------------------------------------------
# 7. heuristic_cohort_from_name
# ---------------------------------------------------------------------------

def test_heuristic_cohort_strips_replicate():
    mut, rep = heuristic_cohort_from_name("LPS-2")
    assert mut == "LPS"
    assert rep == "2"


def test_heuristic_cohort_strips_prefix():
    mut, rep = heuristic_cohort_from_name("NegMSMSALL-CAS9_A")
    assert mut == "CAS9"
    assert rep == "A"


def test_heuristic_cohort_complex_name():
    mut, rep = heuristic_cohort_from_name("LN-25-MBCD-3")
    assert mut == "LN-25-MBCD"
    assert rep == "3"


def test_heuristic_cohort_simple_name():
    mut, rep = heuristic_cohort_from_name("ut-01")
    assert mut == "ut"
    assert rep == "01"


def test_heuristic_cohort_dedup_suffix():
    mut, rep = heuristic_cohort_from_name("CAS9_A.1")
    # Should strip .1 dedup suffix first, then parse
    assert mut == "CAS9"
    assert rep == "A"


# ---------------------------------------------------------------------------
# 8. aggregate_by_hierarchy
# ---------------------------------------------------------------------------

def test_aggregate_by_hierarchy_single_level(simple_df_raw, simple_df_exps):
    df_p = make_df_p(simple_df_raw, simple_df_exps)
    result = aggregate_by_hierarchy(df_p, simple_df_exps, levels=["Mutation"])
    assert "CAS9" in result.columns
    assert "WT" in result.columns
    assert result.index.name == "Sample Name"
    # CAS9_A=100, CAS9_B=110 → mean=105 for PC 34:1
    assert abs(result.loc["PC 34:1", "CAS9"] - 105.0) < 1e-9


def test_aggregate_by_hierarchy_wrapper_compat(simple_df_raw, simple_df_exps):
    """aggregate_by_cohort (wrapper) should produce identical output."""
    df_p = make_df_p(simple_df_raw, simple_df_exps)
    result_new = aggregate_by_hierarchy(df_p, simple_df_exps, levels=["Mutation"])
    result_old = aggregate_by_cohort(df_p, simple_df_exps)
    pd.testing.assert_frame_equal(result_new, result_old)


def test_aggregate_by_hierarchy_empty():
    result = aggregate_by_hierarchy(pd.DataFrame(), pd.DataFrame())
    assert result.empty


# ---------------------------------------------------------------------------
# 9. extract_experiments with heuristic
# ---------------------------------------------------------------------------

def test_extract_experiments_heuristic():
    """Non-standard column names should be parsed by the heuristic."""
    df = pd.DataFrame({
        "Sample Name": ["PC 34:1", "PE 36:2"],
        "LPS-1": [100.0, 200.0],
        "LPS-2": [110.0, 190.0],
        "ut-01": [80.0, 220.0],
        "ut-02": [90.0, 210.0],
    })
    exps = extract_experiments(df)
    assert set(exps.columns) == {"Exp", "Mutation", "Replicate"}
    assert len(exps) == 4
    mutations = set(exps["Mutation"])
    # Should group LPS-1/LPS-2 into "LPS" and ut-01/ut-02 into "ut"
    assert "LPS" in mutations
    assert "ut" in mutations


# ---------------------------------------------------------------------------
# 10. Full pipeline end-to-end
# ---------------------------------------------------------------------------

def test_preprocess_full_pipeline_csv(tmp_path):
    """Write a CSV with metadata + abundance, preprocess, verify contract."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "Metabolite name": ["PC 34:1", "PE 36:2", "SM 18:1"],
        "Average Rt(min)": [5.2, 6.1, 4.8],
        "Formula": ["C42H82NO8P", "C41H78NO8P", "C39H79N2O6P"],
        "Sample_A": [100, 200, 50],
        "Sample_B": [110, 190, 55],
    })
    df.to_csv(csv_path, index=False)

    result, report = preprocess_raw_metabolomics_export(
        [{"datapath": str(csv_path)}],
        idx_col="Metabolite name",
    )
    assert "Sample Name" in result.columns
    assert "Average Rt(min)" not in result.columns
    assert "Formula" not in result.columns
    assert "Sample_A" in result.columns
    assert "Sample_B" in result.columns
    assert len(result) == 3
    assert result["Sample_A"].dtype in [np.float64, np.int64]


def test_preprocess_fallback_on_failure():
    """Bad file info → empty df, no crash."""
    result, report = preprocess_raw_metabolomics_export(
        [{"datapath": "/nonexistent/file.csv"}],
    )
    assert result.empty
    assert len(report.warnings) > 0


# ---------------------------------------------------------------------------
# 11. Downstream contract preservation
# ---------------------------------------------------------------------------

def test_downstream_contract_preserved(simple_df_raw, simple_df_exps):
    """Verify that the full pipeline df_raw → df_meta → df_p → df_cohort
    still produces correct shapes and columns with the new code paths."""
    # Use the new extract_metadata (which delegates to robust parser)
    df_meta = extract_metadata(simple_df_raw)
    assert set(df_meta.columns) == {
        "Sample Name", "Head Group", "Head Group 2",
        "Acyl Chain Length", "Unsaturation", "Unsaturation 2",
    }

    # df_p should have mutation-named columns (duplicates from make_df_p's
    # rename+select pattern: each mutation in keep list pulls all cols with
    # that name, so 2 CAS9 exps → 4 CAS9-named cols in df_p)
    df_p = make_df_p(simple_df_raw, simple_df_exps)
    assert "Sample Name" in df_p.columns
    cas9_cols = [c for c in df_p.columns if c == "CAS9"]
    assert len(cas9_cols) >= 2  # At least 2 CAS9 columns (replicates)

    # df_cohort should have unique mutation columns
    df_cohort = aggregate_by_cohort(df_p, simple_df_exps)
    assert df_cohort.index.name == "Sample Name"
    assert "CAS9" in df_cohort.columns
    assert "WT" in df_cohort.columns
    assert abs(df_cohort.loc["PC 34:1", "CAS9"] - 105.0) < 1e-9


# ---------------------------------------------------------------------------
# 12. PreprocessReport
# ---------------------------------------------------------------------------

def test_preprocess_report_summary_lines():
    report = PreprocessReport(
        detected_lipid_column="Metabolite name",
        detected_sample_columns=["S1", "S2", "S3"],
        removed_metadata_columns=["Rt", "Mz"],
        failed_lipid_parses=2,
    )
    lines = report.summary_lines()
    assert any("Metabolite name" in l for l in lines)
    assert any("3" in l for l in lines)  # 3 sample columns
    assert any("2" in l for l in lines)  # 2 metadata cols or 2 failed parses
