"""
tests/test_analysis.py
Comprehensive test suite for all analysis functions (old + new).
Run with: python -m pytest tests/ -v
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.analysis import (
    deduplicate_columns,
    load_and_deduplicate_data,
    load_and_deduplicate_data_v2,
    extract_experiments,
    make_df_p,
    aggregate_by_cohort,
    extract_metadata,
    filter_data,
    norm_col,
    norm_row,
    fold_change,
    z_score,
    groupby_norm,
    chain_length_analysis,
    unsaturation_analysis,
    headgroup_analysis,
    lipid_class_analysis,
    odd_chain_fraction,
    subset_headgroup_by_chain,
    subset_headgroup_by_unsat,
    perform_pca,
    holm_sidak_correction,
    bonferroni_correction,
    benjamini_hochberg_correction,
    statistical_analysis_v2,
    plot_pca_2d_replicates,
    plot_cl_gaussian_fit,
    identify_cl_outliers,
    plot_odd_chain_kde,
    pointwise_stat_test,
    subgroup_analysis,
    summarise_top_changes,
    SUBGROUP_MADAG,
    SUBGROUP_SPHINGOLIPIDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df_raw():
    """A minimal raw DataFrame (3 lipids, 4 experiments over 2 mutations)."""
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


@pytest.fixture
def simple_df_p(simple_df_raw, simple_df_exps):
    return make_df_p(simple_df_raw, simple_df_exps)


@pytest.fixture
def simple_df_cohort(simple_df_p, simple_df_exps):
    return aggregate_by_cohort(simple_df_p, simple_df_exps)


@pytest.fixture
def simple_df_meta(simple_df_raw):
    return extract_metadata(simple_df_raw)


# ---------------------------------------------------------------------------
# 1. deduplicate_columns
# ---------------------------------------------------------------------------

def test_deduplicate_columns_no_dups():
    assert deduplicate_columns(["A", "B", "C"]) == ["A", "B", "C"]


def test_deduplicate_columns_with_dups():
    result = deduplicate_columns(["A", "A", "B", "A"])
    assert result == ["A", "A.1", "B", "A.2"]


# ---------------------------------------------------------------------------
# 2. extract_experiments (regex fallback)
# ---------------------------------------------------------------------------

def test_extract_experiments_regex(simple_df_raw):
    df = extract_experiments(simple_df_raw)
    assert set(df.columns) == {"Exp", "Mutation", "Replicate"}
    # Without the NegMSMSALL prefix, the regex doesn't match, so the full
    # column name is used as the mutation label (expected fallback behavior)
    assert set(df["Exp"]) == {"CAS9_A", "CAS9_B", "WT_A", "WT_B"}
    assert len(df) == 4


def test_extract_experiments_header_file(simple_df_raw):
    header_df = pd.DataFrame([
        ["shortname", "CAS9_A", "CAS9_B", "WT_A", "WT_B"],
        ["fullname",  "CAS9 rep A", "CAS9 rep B", "WT rep A", "WT rep B"],
        ["cohort",    "CAS9", "CAS9", "WT", "WT"],
    ])
    df = extract_experiments(simple_df_raw, header_df=header_df, cohort_row_idx=2)
    assert set(df["Mutation"]) == {"CAS9", "WT"}


# ---------------------------------------------------------------------------
# 3. make_df_p  /  aggregate_by_cohort
# ---------------------------------------------------------------------------

def test_make_df_p_columns(simple_df_p):
    assert "Sample Name" in simple_df_p.columns
    assert "CAS9" in simple_df_p.columns
    assert "WT" in simple_df_p.columns


def test_aggregate_by_cohort_mean(simple_df_p, simple_df_exps):
    dc = aggregate_by_cohort(simple_df_p, simple_df_exps, method="mean")
    # CAS9_A=100, CAS9_B=110 → mean=105 for PC 34:1
    assert abs(dc.loc["PC 34:1", "CAS9"] - 105.0) < 1e-9


def test_aggregate_by_cohort_median(simple_df_p, simple_df_exps):
    dc = aggregate_by_cohort(simple_df_p, simple_df_exps, method="median")
    assert abs(dc.loc["PC 34:1", "CAS9"] - 105.0) < 1e-9


# ---------------------------------------------------------------------------
# 4. extract_metadata
# ---------------------------------------------------------------------------

def test_extract_metadata_columns(simple_df_meta):
    expected = {"Sample Name", "Head Group", "Head Group 2",
                "Acyl Chain Length", "Unsaturation", "Unsaturation 2"}
    assert expected.issubset(set(simple_df_meta.columns))


def test_extract_metadata_values(simple_df_meta):
    pc_row = simple_df_meta[simple_df_meta["Sample Name"] == "PC 34:1"].iloc[0]
    assert pc_row["Head Group"] == "PC"
    assert pc_row["Acyl Chain Length"] == 34
    assert pc_row["Unsaturation"] == 1


# ---------------------------------------------------------------------------
# 5. filter_data — blank keyword removal
# ---------------------------------------------------------------------------

def test_filter_data_blank_keywords(simple_df_raw, simple_df_meta):
    dr, dm = filter_data(simple_df_raw, simple_df_meta,
                         blank_keywords=["WT"])
    assert "WT_A" not in dr.columns
    assert "WT_B" not in dr.columns
    assert "CAS9_A" in dr.columns


def test_filter_data_no_keywords(simple_df_raw, simple_df_meta):
    dr, dm = filter_data(simple_df_raw, simple_df_meta)
    assert dr.shape == simple_df_raw.shape


# ---------------------------------------------------------------------------
# 6. norm_col / norm_row
# ---------------------------------------------------------------------------

def test_norm_col():
    df = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 2.0]})
    nc = norm_col(df)
    assert abs(nc["A"].sum() - 1.0) < 1e-9
    assert abs(nc["B"].sum() - 1.0) < 1e-9


def test_norm_row():
    df = pd.DataFrame({"A": [1.0, 3.0], "B": [3.0, 1.0]})
    nr = norm_row(df)
    assert abs(nr.iloc[0].sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 7. Multiple testing corrections
# ---------------------------------------------------------------------------

def test_holm_sidak_basic():
    pvals = [0.001, 0.04, 0.3]
    res = holm_sidak_correction(pvals, alpha=0.05)
    assert res["reject"][0] == True   # smallest p passes
    assert len(res["p_adjusted"]) == 3


def test_bonferroni_correction():
    pvals = [0.01, 0.04, 0.5]
    res = bonferroni_correction(pvals, alpha=0.05)
    assert len(res["p_adjusted"]) == 3
    # Bonferroni: p_adj = p * m
    assert abs(res["p_adjusted"][0] - 0.03) < 1e-9
    assert res["reject"][0] == True
    assert res["reject"][2] == False


def test_benjamini_hochberg_correction():
    pvals = [0.001, 0.02, 0.15, 0.6]
    res = benjamini_hochberg_correction(pvals, alpha=0.05)
    assert len(res["p_adjusted"]) == 4
    assert res["reject"][0] == True


def test_corrections_empty():
    for fn in (holm_sidak_correction, bonferroni_correction, benjamini_hochberg_correction):
        res = fn([], alpha=0.05)
        assert len(res["p_adjusted"]) == 0


# ---------------------------------------------------------------------------
# 8. statistical_analysis_v2
# ---------------------------------------------------------------------------

def test_statistical_analysis_v2_runs(
    simple_df_meta, simple_df_p, simple_df_cohort
):
    anova_df, posthoc_df = statistical_analysis_v2(
        simple_df_meta, simple_df_p, simple_df_cohort,
        "Head Group 2", alpha=0.05, correction="bonferroni"
    )
    assert isinstance(anova_df, pd.DataFrame)
    assert isinstance(posthoc_df, pd.DataFrame)


def test_statistical_analysis_v2_bh(
    simple_df_meta, simple_df_p, simple_df_cohort
):
    anova_df, _ = statistical_analysis_v2(
        simple_df_meta, simple_df_p, simple_df_cohort,
        "Head Group 2", alpha=0.05, correction="bh"
    )
    assert "PR(>F)" in anova_df.columns


# ---------------------------------------------------------------------------
# 9. PCA replicate scatter with ellipses
# ---------------------------------------------------------------------------

def test_plot_pca_2d_replicates_runs(simple_df_p, simple_df_exps):
    fig = plot_pca_2d_replicates(simple_df_p, simple_df_exps)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_pca_2d_replicates_empty():
    import matplotlib.pyplot as plt
    fig = plot_pca_2d_replicates(pd.DataFrame(), pd.DataFrame())
    assert fig is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# 10. Gaussian curve fit
# ---------------------------------------------------------------------------

def test_plot_cl_gaussian_fit_runs(simple_df_meta, simple_df_p, simple_df_cohort):
    import matplotlib.pyplot as plt
    d = chain_length_analysis(simple_df_meta, simple_df_p, simple_df_cohort)
    fig = plot_cl_gaussian_fit(d.get("long"))
    assert fig is not None
    plt.close("all")


def test_plot_cl_gaussian_fit_empty():
    import matplotlib.pyplot as plt
    fig = plot_cl_gaussian_fit(None)
    assert fig is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# 11. Chain length outlier identification
# ---------------------------------------------------------------------------

def test_identify_cl_outliers_runs(simple_df_meta, simple_df_p, simple_df_cohort):
    d = chain_length_analysis(simple_df_meta, simple_df_p, simple_df_cohort)
    out = identify_cl_outliers(d)
    assert isinstance(out, pd.DataFrame)
    if not out.empty:
        assert "Cohort" in out.columns
        assert "JS_Divergence" in out.columns
        assert "Rank" in out.columns


def test_identify_cl_outliers_empty():
    out = identify_cl_outliers({})
    assert out.empty


# ---------------------------------------------------------------------------
# 12. Odd-chain KDE
# ---------------------------------------------------------------------------

def test_plot_odd_chain_kde_runs(simple_df_meta, simple_df_p):
    import matplotlib.pyplot as plt
    fig = plot_odd_chain_kde(simple_df_meta, simple_df_p)
    assert fig is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# 13. Per-bin statistical test
# ---------------------------------------------------------------------------

def test_pointwise_stat_test_runs(simple_df_meta, simple_df_p):
    result = pointwise_stat_test(
        simple_df_meta, simple_df_p, "Acyl Chain Length", "CAS9", alpha=0.05
    )
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "Cohort" in result.columns
        assert "pval" in result.columns


def test_pointwise_stat_test_no_ctrl(simple_df_meta, simple_df_p):
    result = pointwise_stat_test(simple_df_meta, simple_df_p, "Acyl Chain Length", "")
    assert result.empty


# ---------------------------------------------------------------------------
# 14. Subgroup analysis
# ---------------------------------------------------------------------------

def test_subgroup_analysis_madag_returns_dict(
    simple_df_meta, simple_df_p, simple_df_cohort
):
    result = subgroup_analysis(
        simple_df_meta, simple_df_p, simple_df_cohort, SUBGROUP_MADAG, "CAS9"
    )
    # No MADAG lipids in fixture → empty dict or empty prop
    assert isinstance(result, dict)


def test_subgroup_analysis_sphingolipids(simple_df_meta, simple_df_p, simple_df_cohort):
    result = subgroup_analysis(
        simple_df_meta, simple_df_p, simple_df_cohort, SUBGROUP_SPHINGOLIPIDS, "CAS9"
    )
    # SM is in fixture
    assert isinstance(result, dict)
    if result:
        assert "prop" in result


# ---------------------------------------------------------------------------
# 15. summarise_top_changes
# ---------------------------------------------------------------------------

def test_summarise_top_changes_runs(simple_df_meta, simple_df_p, simple_df_cohort):
    g, s = summarise_top_changes(simple_df_meta, simple_df_p, simple_df_cohort, "CAS9", top_n=5)
    assert isinstance(g, pd.DataFrame)
    assert isinstance(s, pd.DataFrame)


def test_summarise_top_changes_no_ctrl(simple_df_meta, simple_df_p, simple_df_cohort):
    g, s = summarise_top_changes(simple_df_meta, simple_df_p, simple_df_cohort, "FAKE")
    assert g.empty
    assert s.empty


def test_summarise_top_changes_global_n(simple_df_meta, simple_df_p, simple_df_cohort):
    g, _ = summarise_top_changes(simple_df_meta, simple_df_p, simple_df_cohort, "CAS9", top_n=2)
    assert len(g) <= 2


# ---------------------------------------------------------------------------
# 16. fold_change / z_score / odd_chain / subsets (regression)
# ---------------------------------------------------------------------------

def test_fold_change_returns_df(simple_df_meta, simple_df_p):
    df_log = fold_change(simple_df_meta, simple_df_p, "Head Group 2", "CAS9")
    assert isinstance(df_log, pd.DataFrame)


def test_z_score_returns_df(simple_df_meta, simple_df_p):
    dfz = z_score(simple_df_meta, simple_df_p, "Head Group 2", "CAS9")
    assert isinstance(dfz, pd.DataFrame)


def test_odd_chain_fraction(simple_df_meta, simple_df_cohort):
    result = odd_chain_fraction(simple_df_meta, simple_df_cohort)
    if not result.empty:
        assert "FractionOdd" in result.columns


def test_subset_headgroup_by_chain(simple_df_meta, simple_df_cohort):
    result = subset_headgroup_by_chain(simple_df_meta, simple_df_cohort, lambda x: x >= 30)
    assert isinstance(result, pd.DataFrame)


def test_subset_headgroup_by_unsat(simple_df_meta, simple_df_cohort):
    result = subset_headgroup_by_unsat(simple_df_meta, simple_df_cohort, lambda x: x == 1)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# 17. perform_pca
# ---------------------------------------------------------------------------

def test_perform_pca_returns_scores(simple_df_cohort):
    df_pca, var, loadings = perform_pca(simple_df_cohort)
    assert isinstance(df_pca, pd.DataFrame)
    assert len(var) > 0
    assert abs(var.sum() - 1.0) < 1e-6
