"""
app.py — Shiny for Python application for Lipidomics Data Analysis.

Data model (matches reference lipids_dataset_analysis):
  df_raw    : raw concatenated CSV  (Sample Name col + experiment cols)
  df_exps   : experiment metadata   (Exp, Mutation, Replicate)
  df_p      : per-experiment data, cols = Mutation names (replicates share column names)
  df_cohort : cohort-aggregated     index=Sample Name, cols=unique mutations
  df_meta   : lipid metadata        Sample Name, Head Group, Head Group 2,
                                    Acyl Chain Length, Unsaturation, Unsaturation 2

Running:
  python -m shiny run --port 8000 app/app.py
"""

import io
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as mt
import shinyswatch
from dotenv import load_dotenv

# Load environment variables (e.g. GROQ_API_KEY) from .env if present
load_dotenv()

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

try:
    from app.analysis import (
        load_and_deduplicate_data,
        load_and_deduplicate_data_v2,
        extract_experiments,
        make_df_p,
        aggregate_by_cohort,
        extract_metadata,
        filter_data,
        perform_pca,
        groupby_norm,
        fold_change,
        z_score,
        chain_length_analysis,
        unsaturation_analysis,
        headgroup_analysis,
        lipid_class_analysis,
        odd_chain_fraction,
        subset_headgroup_by_chain,
        subset_headgroup_by_unsat,
        statistical_analysis,
        statistical_analysis_v2,
        norm_col,
        bonferroni_correction,
        benjamini_hochberg_correction,
        plot_pca_variance,
        plot_pca_2d,
        plot_pca_3d,
        plot_pca_2d_replicates,
        plot_kde_histogram,
        plot_zscore_heatmap,
        plot_correlation_heatmap,
        plot_fold_change_heatmap,
        plot_heatmap_general,
        plot_donut_chart,
        plot_pie_chart,
        plot_odd_chain_bar,
        plot_cl_gaussian_fit,
        plot_odd_chain_kde,
        identify_cl_outliers,
        pointwise_stat_test,
        subgroup_analysis,
        summarise_top_changes,
        get_excel_sheet_names,
        plot_pca_scree,
        plot_pca_loadings,
        plot_cohort_dendrogram,
        plot_replicate_correlation,
        plot_volcano,
        plot_lipid_boxplot,
        SUBGROUP_MADAG,
        SUBGROUP_SPHINGOLIPIDS,
    )
except ImportError:
    from analysis import (
        load_and_deduplicate_data,
        load_and_deduplicate_data_v2,
        extract_experiments,
        make_df_p,
        aggregate_by_cohort,
        extract_metadata,
        filter_data,
        perform_pca,
        groupby_norm,
        fold_change,
        z_score,
        chain_length_analysis,
        unsaturation_analysis,
        headgroup_analysis,
        lipid_class_analysis,
        odd_chain_fraction,
        subset_headgroup_by_chain,
        subset_headgroup_by_unsat,
        statistical_analysis,
        statistical_analysis_v2,
        norm_col,
        bonferroni_correction,
        benjamini_hochberg_correction,
        plot_pca_variance,
        plot_pca_2d,
        plot_pca_3d,
        plot_pca_2d_replicates,
        plot_kde_histogram,
        plot_zscore_heatmap,
        plot_correlation_heatmap,
        plot_fold_change_heatmap,
        plot_heatmap_general,
        plot_donut_chart,
        plot_pie_chart,
        plot_odd_chain_bar,
        plot_cl_gaussian_fit,
        plot_odd_chain_kde,
        identify_cl_outliers,
        pointwise_stat_test,
        subgroup_analysis,
        summarise_top_changes,
        get_excel_sheet_names,
        plot_pca_scree,
        plot_pca_loadings,
        plot_cohort_dendrogram,
        plot_replicate_correlation,
        plot_volcano,
        plot_lipid_boxplot,
        SUBGROUP_MADAG,
        SUBGROUP_SPHINGOLIPIDS,
    )

try:
    from app.preprocessing import (
        preprocess_raw_metabolomics_export,
        filter_non_lipids,
        suggest_cohort_mapping,
        coerce_bool_series,
    )
except ImportError:
    from preprocessing import (
        preprocess_raw_metabolomics_export,
        filter_non_lipids,
        suggest_cohort_mapping,
        coerce_bool_series,
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _card(title, *contents, id=None):
    return ui.card(ui.card_header(title), *contents,
                   id=id, full_screen=True)


def plot_toolbar_ui(plot_id):
    return ui.output_ui(f"tb_{plot_id}")


def bulk_export_controls(tab_id):
    return ui.div(
        ui.layout_columns(
            ui.div(
                ui.tags.span("📦 Bulk Export: ", style="font-weight: bold;"),
                "Download all plots in this tab as a single ZIP archive.",
                style="display: flex; align-items: center; height: 100%;"
            ),
            ui.div(
                ui.div(
                    ui.input_select(f"bulk_fmt_{tab_id}", "", {
                        "png300": "PNG (300 DPI)",
                        "png600": "PNG (600 DPI)",
                        "pdf": "PDF (Vector)",
                        "svg": "SVG (Vector)",
                        "eps": "EPS (Vector - No Transparency)"
                    }, selected="eps"),
                    style="width: 220px; display: inline-block; margin-bottom: 0;"
                ),
                ui.download_button(f"dl_zip_{tab_id}", "Download All (ZIP)", class_="btn-sm btn-primary ms-2"),
                style="display: flex; align-items: center; justify-content: flex-end;"
            ),
            col_widths=(8, 4),
            class_="align-items-center"
        ),
        class_="card p-2 bg-light mb-4"
    )



# ---------------------------------------------------------------------------
# UI definition
# ---------------------------------------------------------------------------

app_ui = ui.page_navbar(
    ui.head_content(
        ui.tags.style("""
            .plot-toolbar-container {
                display: flex;
                justify-content: flex-end;
                padding-top: 8px;
                border-top: 1px solid #f0f0f0;
                margin-top: 5px;
            }
            .dropdown-menu {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                border: 1px solid rgba(0, 0, 0, 0.15);
                border-radius: 6px;
                z-index: 1050;
            }
            .dropdown-item {
                font-size: 0.85rem;
                padding: 6px 16px;
                cursor: pointer;
            }
            .dropdown-item:hover {
                background-color: #f8f9fa;
            }
            .dropdown-divider {
                margin: 4px 0;
            }
            .shiny-plot-output img {
                object-fit: contain !important;
                height: 100% !important;
                width: 100% !important;
            }
        """)
    ),

    # ===========================  Tab 1: Upload  ===========================
    ui.nav_panel(
        "📁 Upload Data",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Data Files"),
                ui.input_file("data_files",
                              "Upload CSV / XLS / XLSX file(s):",
                              accept=".csv,.xls,.xlsx", multiple=True),
                ui.hr(),
                ui.h5("Comparison Mode"),
                ui.input_checkbox("wt_cas9_mode",
                                  "WT vs. CAS9 mode (locks control & shows only WT/CAS9)",
                                  False),
                ui.hr(),
                ui.h5("Aggregation"),
                ui.input_select("agg_method", "Replicate aggregation:",
                                {"mean": "Mean", "median": "Median", "sum": "Sum"}),
                ui.hr(),
                ui.h5("Filters"),
                ui.input_selectize("filter_hg",
                                   "Keep head groups (blank = all):",
                                   choices=[], multiple=True),
                ui.input_numeric("min_chain", "Min acyl chain length:", 0, min=0),
                ui.input_numeric("max_unsat", "Max unsaturation:", 99, min=0),
                ui.input_checkbox("remove_blank", "Remove blank samples", False),
                ui.input_numeric("blank_threshold",
                                 "Blank threshold (fraction of mean):",
                                 0.05, min=0.0, max=1.0, step=0.01),
                ui.input_text("blank_keywords",
                              "Blank sample keywords (comma-separated):",
                              placeholder="e.g. RAJU, blank, QC"),
                ui.hr(),
                ui.input_action_button("btn_process", "Process / Apply Filters",
                                       class_="btn-primary btn-sm w-100"),
                width=290,
            ),
            ui.layout_column_wrap(
                _card("Advanced / Emergency Preprocessing Options",
                      ui.p("Use these settings if your file has a non-standard layout (e.g., Lipotype/MSDIAL XLSX files).",
                           class_="text-muted mb-2"),
                      ui.layout_columns(
                          ui.input_text("adv_idx_col",
                                        "Lipid Species Column Name:",
                                        placeholder="e.g. Metabolite name"),
                          ui.input_numeric("adv_skip_rows",
                                           "Header Row Offset (Skip N rows):",
                                           0, min=0),
                          ui.input_numeric("adv_num_idx",
                                           "Drop N initial metadata columns:",
                                           0, min=0),
                          col_widths=(4, 4, 4)
                      ),
                      ui.hr(),
                      ui.h6("Excel Sheet Selection"),
                      ui.p("For XLSX files, select which sheet contains your data. "
                           "Upload the file first, then pick the sheet.",
                           class_="text-muted small mb-1"),
                      ui.input_select("adv_sheet_name",
                                      "Sheet name:",
                                      choices=["(auto — first sheet)"],
                                      selected="(auto — first sheet)"),
                ),
                width=1,
            ),
            ui.layout_column_wrap(
                _card("Optional Header File",
                      ui.input_file("header_file", "Upload custom Header File (CSV) to map column names to cohorts manually:", 
                                    accept=".csv", multiple=False),
                      ui.input_numeric("cohort_row", "Row containing Cohorts (1-based index):", 2, min=1),
                      ui.p("If no header file is provided, the app will automatically extract cohorts from the data file column names.", class_="text-muted mt-2")
                ),
                width=1,
            ),
            ui.layout_column_wrap(
                _card("Preprocessing Options",
                      ui.input_switch("enable_robust_preprocess",
                                      "Enable robust preprocessing",
                                      value=True),
                      ui.p("Automatically detects and strips metadata columns, "
                           "validates abundance data, and handles non-standard formats. "
                           "Disable to use legacy loading for already-clean datasets.",
                           class_="text-muted small mb-2"),
                      ui.input_checkbox("keep_lipids_only",
                                        "Keep only lipid species (remove vitamins, unknowns)",
                                        value=True),
                      ui.hr(),
                      ui.h6("Preprocessing Report"),
                      ui.output_ui("preprocess_report"),
                ),
                width=1,
            ),
            ui.layout_column_wrap(
                _card("★ Cohort / Group Mapping",
                      ui.p("Review and edit the cohort assignments below. "
                           "Each sample column is auto-assigned to a cohort group. "
                           "Edit the 'Cohort' column to change assignments, or uncheck 'Include' to remove a sample.",
                           class_="text-muted small mb-2"),
                      ui.output_ui("cohort_confidence_badge"),
                      ui.output_data_frame("tbl_cohort_mapping"),
                      ui.layout_columns(
                          ui.input_action_button("btn_confirm_cohorts",
                                                 "✅ Confirm Cohort Mapping",
                                                 class_="btn-success btn-sm"),
                          ui.input_action_button("btn_reset_cohorts",
                                                 "↺ Reset to Auto-Detected",
                                                 class_="btn-outline-secondary btn-sm"),
                          col_widths=(6, 6),
                      ),
                ),
                width=1,
            ),
            ui.output_ui("unconfirmed_banner"),
            ui.layout_column_wrap(
                _card("Processed / Filtered Data",
                      ui.output_data_frame("tbl_filtered"),
                      ui.download_button("dl_filtered", "Download CSV",
                                         class_="btn-sm btn-outline-secondary mt-2")),
                _card("Sample / Experiment Table",
                      ui.output_data_frame("tbl_exps")),
                width=1,
            ),

            ui.layout_column_wrap(
                _card("Others (Non-Lipid / Non-Parseable Compounds)",
                      ui.p("Compounds removed from the main analysis pipeline "
                           "(chemical contaminants, IUPAC names, unparseable entries).",
                           class_="text-muted small mb-2"),
                      ui.output_data_frame("tbl_others")),
                width=1,
            ),
        ),
    ),

    # ===========================  Tab 2: PCA  ==============================
    ui.nav_panel(
        "📊 PCA",
        bulk_export_controls("pca"),
        ui.layout_column_wrap(
            _card("Explained Variance", ui.output_plot("plt_pca_var"), plot_toolbar_ui("plt_pca_var")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("PCA — 2D Scores (PC1 vs PC2)", ui.output_plot("plt_pca_2d"), plot_toolbar_ui("plt_pca_2d")),
            _card("PCA — 3D Scores",              ui.output_plot("plt_pca_3d"), plot_toolbar_ui("plt_pca_3d")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("PCA — Replicate Scatter with 95% Confidence Ellipses",
                  ui.output_plot("plt_pca_ellipse"), plot_toolbar_ui("plt_pca_ellipse")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("PCA — Scree Plot", ui.output_plot("plt_pca_scree"), plot_toolbar_ui("plt_pca_scree")),
            _card("PCA — Top Lipid Loadings per Component", ui.output_plot("plt_pca_loadings"), plot_toolbar_ui("plt_pca_loadings")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Cohort Clustering Dendrogram", ui.output_plot("plt_pca_dendro"), plot_toolbar_ui("plt_pca_dendro")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Replicate Correlation Matrix (QC)",
                  ui.p("This computes the Pearson correlation matrix across all replicates. For large datasets, this can take a few minutes.", class_="text-muted small mb-2"),
                  ui.input_action_button("btn_run_qc", "Run QC (Replicate Correlation)", class_="btn-warning btn-sm mb-3", width="300px"),
                  ui.output_plot("plt_rep_corr"),
                  plot_toolbar_ui("plt_rep_corr")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Downloads",
                  ui.download_button("dl_pca_scores",   "PCA Scores CSV",
                                     class_="btn-sm btn-outline-secondary"),
                  ui.download_button("dl_pca_variance", "Explained Variance CSV",
                                     class_="btn-sm btn-outline-secondary ms-2")),
            width=1,
        ),
    ),

    # ========================  Tab 3: Chain Length  ========================
    ui.nav_panel(
        "⛓ Chain Length",
        bulk_export_controls("cl"),
        ui.layout_column_wrap(
            _card("KDE / Histogram — Chain Length Distribution",
                  ui.output_plot("plt_cl_kde"), plot_toolbar_ui("plt_cl_kde")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Z-score Heatmap",         ui.output_plot("plt_cl_zscore"), plot_toolbar_ui("plt_cl_zscore")),
            _card("Correlation Matrix",      ui.output_plot("plt_cl_corr"), plot_toolbar_ui("plt_cl_corr")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Proportions Heatmap",     ui.output_plot("plt_cl_prop"), plot_toolbar_ui("plt_cl_prop")),
            _card("Fold Change vs Control",
                  ui.layout_sidebar(
                      ui.sidebar(ui.input_select("cl_ctrl", "Control:", choices=[]),
                                 width=200),
                      ui.output_plot("plt_cl_fc"), plot_toolbar_ui("plt_cl_fc"))),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Gaussian Fit — Chain Length Distribution",
                  ui.output_plot("plt_cl_gauss"), plot_toolbar_ui("plt_cl_gauss")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Odd-Chain Lipid Fraction by Cohort",
                  ui.output_plot("plt_odd_chain"), plot_toolbar_ui("plt_odd_chain")),
            _card("Odd-Chain Length KDE",
                  ui.output_plot("plt_odd_cl_kde"), plot_toolbar_ui("plt_odd_cl_kde")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Chain Length Distribution Outliers",
                  ui.output_data_frame("tbl_cl_outliers"),
                  ui.p("Ranked by Jensen-Shannon divergence from mean distribution.",
                       class_="text-muted mt-1 small")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Head Groups — Chain Length ≥ 50",
                  ui.output_plot("plt_cl_ge50"), plot_toolbar_ui("plt_cl_ge50")),
            _card("Head Groups — Chain Length ≤ 30",
                  ui.output_plot("plt_cl_le30"), plot_toolbar_ui("plt_cl_le30")),
            _card("Head Groups — Chain Length ≤ 20",
                  ui.output_plot("plt_cl_le20"), plot_toolbar_ui("plt_cl_le20")),
            width="1/3",
        ),
    ),

    # =======================  Tab 4: Unsaturation  ========================
    ui.nav_panel(
        "〰 Unsaturation",
        bulk_export_controls("us"),
        ui.layout_column_wrap(
            _card("KDE / Histogram — Unsaturation Distribution",
                  ui.output_plot("plt_us_kde"), plot_toolbar_ui("plt_us_kde")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Z-score Heatmap",    ui.output_plot("plt_us_zscore"), plot_toolbar_ui("plt_us_zscore")),
            _card("Correlation Matrix", ui.output_plot("plt_us_corr"), plot_toolbar_ui("plt_us_corr")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Proportions Heatmap", ui.output_plot("plt_us_prop"), plot_toolbar_ui("plt_us_prop")),
            _card("Fold Change vs Control",
                  ui.layout_sidebar(
                      ui.sidebar(ui.input_select("us_ctrl", "Control:", choices=[]),
                                 width=200),
                      ui.output_plot("plt_us_fc"), plot_toolbar_ui("plt_us_fc"))),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Saturated (0 db) — Head Group Distribution",
                  ui.output_plot("plt_us_sat"), plot_toolbar_ui("plt_us_sat")),
            _card("Monounsaturated (1–2 db) — Head Group Distribution",
                  ui.output_plot("plt_us_mono"), plot_toolbar_ui("plt_us_mono")),
            _card("Polyunsaturated (≥3 db) — Head Group Distribution",
                  ui.output_plot("plt_us_poly"), plot_toolbar_ui("plt_us_poly")),
            width="1/3",
        ),
    ),

    # ========================  Tab 5: Head Group  ==========================
    ui.nav_panel(
        "🧬 Head Group",
        bulk_export_controls("hg"),
        ui.layout_column_wrap(
            _card("Head Group Distribution (Donut)",
                  ui.output_plot("plt_hg_donut"), plot_toolbar_ui("plt_hg_donut")),
            _card("Z-score Heatmap",
                  ui.output_plot("plt_hg_zscore"), plot_toolbar_ui("plt_hg_zscore")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Correlation Matrix",
                  ui.output_plot("plt_hg_corr"), plot_toolbar_ui("plt_hg_corr")),
            _card("Fold Change vs Control",
                  ui.layout_sidebar(
                      ui.sidebar(ui.input_select("hg_ctrl", "Control:", choices=[]),
                                 width=200),
                      ui.output_plot("plt_hg_fc"), plot_toolbar_ui("plt_hg_fc"))),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Proportions Heatmap",
                  ui.output_plot("plt_hg_prop"), plot_toolbar_ui("plt_hg_prop")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Bar Plot — Selected Head Group by Cohort",
                  ui.layout_sidebar(
                      ui.sidebar(
                          ui.input_select("hg_bar_group", "Head Group:", choices=[]),
                          width=200),
                      ui.output_plot("plt_hg_bar"), plot_toolbar_ui("plt_hg_bar"))),
            width=1,
        ),
    ),

    # ========================  Tab 6: Lipid Class  =========================
    ui.nav_panel(
        "🫧 Lipid Class",
        bulk_export_controls("lc"),
        ui.layout_column_wrap(
            _card("Lipid Class Distribution (Pie)",
                  ui.output_plot("plt_lc_pie"), plot_toolbar_ui("plt_lc_pie")),
            _card("Z-score Heatmap",
                  ui.output_plot("plt_lc_zscore"), plot_toolbar_ui("plt_lc_zscore")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Normalised Proportions Heatmap",
                  ui.output_plot("plt_lc_prop"), plot_toolbar_ui("plt_lc_prop")),
            _card("Fold Change vs Control",
                  ui.layout_sidebar(
                      ui.sidebar(ui.input_select("lc_ctrl", "Control:", choices=[]),
                                 width=200),
                      ui.output_plot("plt_lc_fc"), plot_toolbar_ui("plt_lc_fc"))),
            width="1/2",
        ),
    ),

    # =======================  Tab 7: Statistics  ===========================
    ui.nav_panel(
        "📈 Statistics",
        bulk_export_controls("stat"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("ANOVA / Post-hoc Controls"),
                ui.input_select("stat_var", "Group variable:",
                                {"Head Group 2":      "Head Group 2",
                                 "Acyl Chain Length": "Chain Length",
                                 "Unsaturation":      "Unsaturation"}),
                ui.input_numeric("stat_alpha", "Significance level (α):",
                                 0.05, min=0.001, max=0.5, step=0.001),
                ui.input_select("stat_correction", "Multiple testing correction:",
                                {"holm_sidak": "Holm–Sidak",
                                 "bonferroni":  "Bonferroni",
                                 "bh":          "Benjamini–Hochberg (FDR)"}),
                ui.input_select("stat_ctrl", "Control cohort (for per-bin tests):",
                                choices=[]),
                ui.input_action_button("btn_run_stat", "Run Analysis",
                                       class_="btn-primary btn-sm w-100"),
                width=260,
            ),
            ui.layout_column_wrap(
                _card("One-Way ANOVA Results",
                      ui.output_data_frame("tbl_anova"),
                      ui.download_button("dl_anova", "Download ANOVA CSV",
                                         class_="btn-sm btn-outline-secondary mt-2")),
                width=1,
            ),
            ui.layout_column_wrap(
                _card("Pairwise Post-hoc (corrected)",
                      ui.output_data_frame("tbl_posthoc"),
                      ui.download_button("dl_posthoc", "Download Post-hoc CSV",
                                         class_="btn-sm btn-outline-secondary mt-2")),
                width=1,
            ),
            ui.layout_column_wrap(
                _card("Per-Bin t-test vs Control",
                      ui.output_data_frame("tbl_pw_stat"),
                      ui.download_button("dl_pw_stat", "Download Per-Bin CSV",
                                         class_="btn-sm btn-outline-secondary mt-2")),
                width=1,
            ),
            ui.layout_column_wrap(
                _card("Volcano Plot", ui.output_plot("plt_volcano"), plot_toolbar_ui("plt_volcano")),
                width=1,
            ),
        ),
    ),

    # =======================  Tab: Lipid Explorer  =========================
    ui.nav_panel(
        "🔬 Lipid Explorer",
        bulk_export_controls("lipid"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.output_ui("lipid_select_ui"),
                ui.hr(),
                ui.p("Select any lipid to see its abundance across all cohorts and replicates.",
                     class_="text-muted small"),
                width=260,
            ),
            ui.layout_column_wrap(
                _card("Per-Lipid Abundance — Box / Strip Plot",
                      ui.output_plot("plt_lipid_box"),
                      plot_toolbar_ui("plt_lipid_box")),
                width=1,
            ),
        ),
    ),


    # =======================  Tab 8: Sub-groups  ===========================
    ui.nav_panel(
        "🧫 Sub-groups",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Sub-group Controls"),
                ui.input_select("sg_ctrl", "Control cohort:", choices=[]),
                width=220,
            ),
            bulk_export_controls("sg"),
            ui.h4("Storage Lipids (MAG / DAG / TAG)"),
            ui.layout_column_wrap(
                _card("Storage Lipids — Proportions Heatmap",
                      ui.output_plot("plt_sg_madag_prop"), plot_toolbar_ui("plt_sg_madag_prop")),
                _card("Storage Lipids — Log Fold Change",
                      ui.output_plot("plt_sg_madag_fc"), plot_toolbar_ui("plt_sg_madag_fc")),
                _card("Storage Lipids — Z-score",
                      ui.output_plot("plt_sg_madag_z"), plot_toolbar_ui("plt_sg_madag_z")),
                width="1/3",
            ),
            ui.hr(),
            ui.h4("Sphingolipids (Cer / HexCer / SM / GM)"),
            ui.layout_column_wrap(
                _card("Sphingolipids — Proportions Heatmap",
                      ui.output_plot("plt_sg_sph_prop"), plot_toolbar_ui("plt_sg_sph_prop")),
                _card("Sphingolipids — Log Fold Change",
                      ui.output_plot("plt_sg_sph_fc"), plot_toolbar_ui("plt_sg_sph_fc")),
                _card("Sphingolipids — Z-score",
                      ui.output_plot("plt_sg_sph_z"), plot_toolbar_ui("plt_sg_sph_z")),
                width="1/3",
            ),
        ),
    ),

    # =======================  Tab 9: Insights  =============================
    ui.nav_panel(
        "🔍 Insights",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Insight Controls"),
                ui.input_select("ins_ctrl", "Control cohort:", choices=[]),
                ui.input_numeric("ins_top_n", "Top N lipids:", 20, min=5, max=100),
                ui.input_action_button("btn_run_insights", "Run Insights",
                                       class_="btn-primary btn-sm w-100"),
                width=230,
            ),
            ui.layout_column_wrap(
                _card("Universally Changed Lipids",
                      ui.p("Lipids with highest mean |log2FC| across ALL non-control mutations, in a consistent direction.",
                           class_="text-muted small"),
                      ui.output_data_frame("tbl_global_changes"),
                      ui.download_button("dl_global_changes", "Download CSV",
                                         class_="btn-sm btn-outline-secondary mt-2")),
                width=1,
            ),
            ui.layout_column_wrap(
                _card("Mutation-Specific Signatures",
                      ui.p("Lipids with the largest fold change unique to a single mutation.",
                           class_="text-muted small"),
                      ui.layout_sidebar(
                          ui.sidebar(
                              ui.input_select("ins_mut_filter", "Filter by mutation:",
                                              choices=["(all)"]),
                              width=180),
                          ui.output_data_frame("tbl_specific_changes")),
                      ui.download_button("dl_specific_changes", "Download CSV",
                                         class_="btn-sm btn-outline-secondary mt-2")),
                width=1,
            ),
        ),
    ),

    ui.nav_panel(
        "✨ AI Report",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Report Settings"),
                ui.input_password("ai_api_key", "Groq API Key:", value=""),
                ui.input_select("ai_audience", "Target Audience:", 
                                choices=["Layperson", "Clinician", "Researcher"],
                                selected="Researcher"),
                ui.p("Reports will automatically include statistical significance if 'Run Analysis' was pressed in the Statistics tab.", class_="text-muted small"),
                ui.input_action_button("btn_generate_report", "Generate Report",
                                       class_="btn-primary w-100"),
                ui.hr(),
                ui.h5("Downloads"),
                ui.download_button("dl_report_md", "Download Markdown (.md)", class_="btn-sm btn-outline-secondary w-100 mb-2"),
                ui.download_button("dl_report_txt", "Download Text (.txt)", class_="btn-sm btn-outline-secondary w-100"),
                width=250,
            ),
            ui.div(
                ui.output_ui("ai_report_content"),
                style="padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #333;"
            )
        )
    ),

    title=ui.tags.span(
        ui.tags.b("Lipidomics"),
        ui.tags.span(" Data Analysis", style="font-weight:300"),
    ),
    bg="#1a1a2e",
    inverse=True,
)


def plot_hg_abundance_bar(d, grp):
    if not d or not grp:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Select a head group", ha="center", va="center")
        return fig
    raw = d["cohort_raw"]
    if grp not in raw.index:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"'{grp}' not found", ha="center", va="center")
        return fig
    row    = raw.loc[grp]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (coh, val) in enumerate(row.items()):
        ax.bar(coh, val, color=colors[i % len(colors)], edgecolor="white")
    ax.set_ylabel("Abundance (sum)")
    ax.set_title(f"{grp} — Abundance by Cohort")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); return fig


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session):

    @reactive.effect
    async def _ws_keepalive():
        reactive.invalidate_later(25)  # every 25s — below any proxy's idle threshold
        await session.send_custom_message("keepalive", {"t": "ping"})

    # ------------------------------------------------------------------ #
    #  REACTIVE CALCULATIONS                                               #
    # ------------------------------------------------------------------ #

    current_enlarged_plot = reactive.value(None)

    # Reactive value to hold preprocessing report lines for the UI
    _preprocess_report_lines = reactive.value([])

    # AI Report State
    ai_report_markdown = reactive.value("Click **Generate Report** to begin.")
    
    @reactive.effect
    def _invalidate_stale_report():
        raw_df()
        _cohort_confirmed()
        ai_report_markdown.set("Click **Generate Report** to begin.")
    
    @reactive.effect
    @reactive.event(input.btn_generate_report)
    async def generate_ai_report():
        if not _cohort_confirmed():
            ai_report_markdown.set("⚠️ **Error:** Cohort mapping must be confirmed before generating a report.")
            return
            
        dm = df_meta()
        dp = df_p()
        dc = df_cohort()
        if dm.empty or dp.empty or dc.empty:
            ai_report_markdown.set("⚠️ **Error:** Missing dataset.")
            return
            
        # Get comparison cohorts
        comparison_cohorts = dc.columns.tolist()
        if len(comparison_cohorts) < 2:
            ai_report_markdown.set("⚠️ **Error:** At least 2 cohorts are required to generate a comparison report.")
            return
            
        # Comparison basis (global control or None)
        comparison_basis = input.stat_ctrl() if input.stat_ctrl() else None 
        
        from report_context import build_llm_context
        from llm_service import ReportGenerator
        
        # Try to get stats if computed
        df_stats = pd.DataFrame()
        df_pw_stats = pd.DataFrame()
        try:
            with reactive.isolate():
                from shiny.types import SilentException, SilentCancelOutputException
                df_stats, df_pw_stats = stat_result()
        except Exception:
            pass
            
        # Try to get PCA if computed
        df_pca = None
        variance = None
        try:
            df_pca, variance, _ = pca_result()
        except Exception:
            pass
            
        # Fetch detailed tab analyses
        c_data, u_data, h_data, l_data = {}, {}, {}, {}
        try:
            with reactive.isolate():
                c_data = cl_data()
                u_data = us_data()
                h_data = hg_data()
                l_data = lc_data()
        except Exception as e:
            from shiny.types import SilentException, SilentCancelOutputException
            if isinstance(e, (SilentException, SilentCancelOutputException)):
                ai_report_markdown.set("⚠️ **Error:** Please visit the Chain Length, Unsaturation, Head Group, and Lipid Class tabs to initialize them before generating a report.")
                return
            ai_report_markdown.set(f"⚠️ **Error fetching detailed analysis:** {str(e)}")
            return
            
        with reactive.isolate():
            try:
                odd_chain = odd_chain_fraction(dm, dc)
            except Exception:
                odd_chain = pd.DataFrame()
        
        ai_report_markdown.set("⏳ **Building context and analyzing findings...**")
        
        # Build Context
        try:
            ctx = build_llm_context(
                df_meta=dm,
                df_cohort=dc,
                comparison_cohorts=comparison_cohorts,
                df_pca=df_pca,
                variance=variance,
                comparison_basis=comparison_basis,
                df_stats=df_stats,
                df_pw_stats=df_pw_stats,
                c_data=c_data,
                u_data=u_data,
                h_data=h_data,
                l_data=l_data,
                odd_chain=odd_chain
            )
        except Exception as e:
            ai_report_markdown.set(f"⚠️ **Error building context:** {str(e)}")
            return
        
        ai_report_markdown.set("⏳ **Context built. Sending to AI model...**")
        
        # Generate Report
        try:
            import os
            api_key = input.ai_api_key() or os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                ai_report_markdown.set("⚠️ **Error:** No Groq API Key provided. Please enter it in the sidebar or set the GROQ_API_KEY environment variable.")
                return
            llm = ReportGenerator(api_key=api_key)
        except Exception as e:
            print(f"LLM init error: {e}", flush=True)
            ai_report_markdown.set("⚠️ **Error initializing the AI client.** Check that your API key is valid and try again.")
            return
            
        audience = input.ai_audience()
        
        stream_chunks = []
        async for chunk in llm.generate_report_stream(ctx, audience):
            stream_chunks.append(chunk)
            ai_report_markdown.set("".join(stream_chunks) + " ▌")
            
        ai_report_markdown.set("".join(stream_chunks))

    @render.download(filename="lipidomics_report.md")
    def dl_report_md():
        yield ai_report_markdown()
        
    @render.download(filename="lipidomics_report.txt")
    def dl_report_txt():
        yield ai_report_markdown()

    @render.ui
    def ai_report_content():
        return ui.markdown(ai_report_markdown())

    @reactive.calc
    def raw_df():
        files = input.data_files()
        if files is None:
            return pd.DataFrame()

        adv_idx = input.adv_idx_col()
        num_idx = input.adv_num_idx()
        skip    = input.adv_skip_rows()
        sheet_choice = input.adv_sheet_name()
        sheet_name = 0 if sheet_choice == "(auto — first sheet)" else sheet_choice

        if input.enable_robust_preprocess():
            try:
                df, report = preprocess_raw_metabolomics_export(
                    files,
                    sheet_name=sheet_name,
                    idx_col=adv_idx,
                    num_idx=int(num_idx) if num_idx else 0,
                    skip_rows=int(skip) if skip else 0,
                )
                _preprocess_report_lines.set(report.summary_lines())
                if df.empty and report.warnings:
                    ui.notification_show(
                        "Robust preprocessing returned empty data. "
                        "Falling back to legacy loader.",
                        type="warning", duration=6,
                    )
                    return load_and_deduplicate_data_v2(
                        files,
                        idx_col=adv_idx,
                        num_idx=int(num_idx) if num_idx else 0,
                        skip_rows=int(skip) if skip else 0,
                        sheet_name=sheet_name,
                    )
                return df
            except Exception as e:
                ui.notification_show(
                    f"Robust preprocessing failed: {e}. "
                    f"Falling back to legacy loader.",
                    type="warning", duration=8,
                )
                _preprocess_report_lines.set(
                    [f"⚠ Preprocessing error: {e}",
                     "Fell back to legacy pipeline."]
                )
                return load_and_deduplicate_data_v2(
                    files,
                    idx_col=adv_idx,
                    num_idx=int(num_idx) if num_idx else 0,
                    skip_rows=int(skip) if skip else 0,
                    sheet_name=sheet_name,
                )
        else:
            _preprocess_report_lines.set(
                ["Robust preprocessing disabled — using legacy loader."]
            )
            return load_and_deduplicate_data_v2(
                files,
                idx_col=adv_idx,
                num_idx=int(num_idx) if num_idx else 0,
                skip_rows=int(skip) if skip else 0,
                sheet_name=sheet_name,
            )

    @reactive.effect
    def _update_sheet_choices():
        """Whenever a file is uploaded, populate the sheet name dropdown."""
        files = input.data_files()
        if not files:
            ui.update_select("adv_sheet_name",
                             choices=["(auto — first sheet)"],
                             selected="(auto — first sheet)",
                             session=session)
            return
        # Use the first uploaded file to read sheet names
        path = files[0]["datapath"]
        sheets = get_excel_sheet_names(path)
        if sheets:
            choices = sheets  # real sheet names only
            selected = sheets[0]
            ui.update_select("adv_sheet_name",
                             choices=choices,
                             selected=selected,
                             session=session)
        else:
            # CSV — no sheet selection needed
            ui.update_select("adv_sheet_name",
                             choices=["(auto — first sheet)"],
                             selected="(auto — first sheet)",
                             session=session)

    @reactive.calc
    def header_df():
        hf = input.header_file()
        if not hf:
            return None
        try:
            return pd.read_csv(hf[0]["datapath"], header=None)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  COHORT MAPPING & INFERENCE                                          #
    # ------------------------------------------------------------------ #

    _cohort_mapping_df = reactive.value(pd.DataFrame())
    _cohort_confirmed = reactive.Value(False)
    _qc_run = reactive.Value(False)
    _rep_corr_run = reactive.Value(False)
    _cohort_confidence = reactive.value(0.0)

    def _compute_initial_mapping(df, hdr, crow_idx):
        if df.empty:
            return pd.DataFrame(), 0.0
        sample_cols = [c for c in df.columns if c != "Sample Name"]
        if hdr is not None and not hdr.empty:
            exps_legacy = extract_experiments(df, header_df=hdr, cohort_row_idx=crow_idx)
            if not exps_legacy.empty:
                mapping = pd.DataFrame({
                    "Include": [True] * len(exps_legacy),
                    "Sample": exps_legacy["Exp"],
                    "Cohort": exps_legacy["Mutation"],
                    "Replicate": exps_legacy["Replicate"]
                })
                return mapping, 1.0 # Header file is high confidence
        return suggest_cohort_mapping(sample_cols)

    @reactive.effect
    @reactive.event(raw_df, header_df, input.cohort_row)
    def _auto_suggest_cohorts():
        df = raw_df()
        hdr = header_df()
        crow = max(0, int(input.cohort_row()) - 1) if input.cohort_row() else 1
        mapping, conf = _compute_initial_mapping(df, hdr, crow)
        
        _cohort_mapping_df.set(mapping)
        _cohort_confirmed.set(False)
        _cohort_confidence.set(conf)

    @reactive.effect
    @reactive.event(input.btn_confirm_cohorts)
    def _confirm_cohorts():
        print("DEBUG: CONFIRM CLICKED", flush=True)
        _cohort_confirmed.set(True)

    @reactive.effect
    @reactive.event(input.btn_run_qc)
    def _run_qc():
        _qc_run.set(True)

    @reactive.effect
    @reactive.event(input.btn_reset_cohorts)
    def _reset_cohorts():
        df = raw_df()
        hdr = header_df()
        crow = max(0, int(input.cohort_row()) - 1) if input.cohort_row() else 1
        mapping, conf = _compute_initial_mapping(df, hdr, crow)
        
        _cohort_mapping_df.set(mapping)
        _cohort_confirmed.set(False)
        _cohort_confidence.set(conf)

    @reactive.calc
    def finalized_cohort_mapping():
        from shiny.types import SilentException, SilentCancelOutputException
        try:
            edited = tbl_cohort_mapping.data_view()
            if edited is not None and not edited.empty:
                # Validate that the edited dataframe actually has our expected columns
                if "Sample" not in edited.columns or "Cohort" not in edited.columns:
                    print(f"DEBUG: edited dataframe has missing columns: {edited.columns.tolist()}. Falling back.")
                    return _cohort_mapping_df()
                return edited
        except (AttributeError, SilentException, SilentCancelOutputException):
            pass
        return _cohort_mapping_df()

    @reactive.calc
    def df_exps():
        mapping = finalized_cohort_mapping()
        if mapping.empty:
            return pd.DataFrame()
            
        # Robust boolean coercion
        if "Include" in mapping.columns:
            inc = coerce_bool_series(mapping["Include"])
            mapping_included = mapping[inc]
        else:
            mapping_included = mapping
        
        if mapping_included.empty:
            return pd.DataFrame()
            
        exps = pd.DataFrame({
            "Exp":       mapping_included["Sample"].tolist(),
            "Mutation":  mapping_included["Cohort"].tolist(),
            "Replicate": mapping_included["Replicate"].tolist(),
        })
        return exps

    @reactive.calc
    def df_p_full():
        """Per-experiment data, cols = Mutation names (cols shared between replicates)."""
        df = raw_df()
        exps = df_exps()
        if df.empty or exps.empty:
            return pd.DataFrame()
        return make_df_p(df, exps)

    @reactive.calc
    def df_meta_full():
        df = raw_df()
        if df.empty:
            return pd.DataFrame()
        return extract_metadata(df)

    # Reactive value to hold rows removed by ontology filtering ("Others")
    _others_df = reactive.value(pd.DataFrame())

    @reactive.calc
    def df_filtered_pair():
        input.btn_process()  # trigger on button press
        dr = raw_df()
        dm = df_meta_full()
        if dr.empty or dm.empty:
            return pd.DataFrame(), pd.DataFrame()

        # --- Ontology filtering (remove non-lipid rows) ---
        if input.keep_lipids_only():
            dr_f, dm_f, ont_report = filter_non_lipids(dr, dm)
            # Capture removed rows for the "Others" table
            removed_names = set(dr["Sample Name"]) - set(dr_f["Sample Name"])
            if removed_names:
                others = dr[dr["Sample Name"].isin(removed_names)].copy()
                _others_df.set(others)
            else:
                _others_df.set(pd.DataFrame())
            if ont_report["removed_count"] > 0:
                cats = ", ".join(ont_report["removed_categories"][:5])
                ui.notification_show(
                    f"Removed {ont_report['removed_count']} non-lipid rows "
                    f"({cats}{'…' if len(ont_report['removed_categories']) > 5 else ''}).",
                    type="message", duration=5,
                )
            dr, dm = dr_f, dm_f
        else:
            _others_df.set(pd.DataFrame())

        fhg = list(input.filter_hg()) if input.filter_hg() else None
        mc  = int(input.min_chain())
        mu  = int(input.max_unsat())
        rb  = bool(input.remove_blank())
        bth = float(input.blank_threshold())
        kw_raw = input.blank_keywords()
        bkw = [k.strip() for k in kw_raw.split(",") if k.strip()] if kw_raw else None

        return filter_data(dr, dm, filter_hg=fhg, min_chain=mc, max_unsat=mu,
                           remove_blank=rb, blank_threshold=bth, blank_keywords=bkw)

    @reactive.calc
    def df_raw_filt():
        dr, _ = df_filtered_pair()
        return dr

    @reactive.calc
    def df_meta():
        _, dm = df_filtered_pair()
        return dm

    @reactive.calc
    def df_p():
        """Filtered per-experiment data."""
        dr = df_raw_filt()
        exps = df_exps()
        if dr.empty or exps.empty:
            return pd.DataFrame()
        return make_df_p(dr, exps)

    @reactive.calc
    def df_cohort():
        print("DEBUG: df_cohort() called", flush=True)
        dp = df_p()
        exps = df_exps()
        if dp.empty or exps.empty:
            return pd.DataFrame()
        dc = aggregate_by_cohort(dp, exps, method=input.agg_method())
        # WT/CAS9 locked mode: restrict to only WT and CAS9 cohorts
        if input.wt_cas9_mode() and not dc.empty:
            wt_cas9_cols = [c for c in dc.columns
                            if "cas9" in c.lower() or "wt" in c.lower()]
            if wt_cas9_cols:
                dc = dc[wt_cas9_cols]
        return dc

    @reactive.calc
    def pca_result():
        print("DEBUG: pca_result() called", flush=True)
        dc = df_cohort()
        if dc.empty:
            return pd.DataFrame(), [], pd.DataFrame()
        return perform_pca(dc)

    @reactive.calc
    def cl_data():
        if not _cohort_confirmed(): return {}
        return chain_length_analysis(df_meta(), df_p(), df_cohort())

    @reactive.calc
    def us_data():
        if not _cohort_confirmed(): return {}
        try:
            return unsaturation_analysis(df_meta(), df_p(), df_cohort())
        except Exception as e:
            import traceback, sys
            print(f"ERROR IN us_data: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return {}

    @reactive.calc
    def hg_data():
        if not _cohort_confirmed(): return {}
        try:
            return headgroup_analysis(df_meta(), df_p(), df_cohort())
        except Exception as e:
            import traceback, sys
            print(f"ERROR IN hg_data: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return {}

    @reactive.calc
    def lc_data():
        if not _cohort_confirmed(): return {}
        try:
            return lipid_class_analysis(df_meta(), df_cohort())
        except Exception as e:
            import traceback, sys
            print(f"ERROR IN lc_data: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return {}

    @reactive.calc
    def stat_result():
        if not _cohort_confirmed(): return pd.DataFrame(), pd.DataFrame()
        input.btn_run_stat()
        dm = df_meta()
        dp = df_p()
        dc = df_cohort()
        if dm.empty or dp.empty or dc.empty:
            return pd.DataFrame(), pd.DataFrame()
        var_map = {"Head Group 2": "Head Group 2",
                   "Acyl Chain Length": "Acyl Chain Length",
                   "Unsaturation": "Unsaturation"}
        var        = var_map.get(input.stat_var(), "Head Group 2")
        alpha      = float(input.stat_alpha())
        correction = input.stat_correction()
        return statistical_analysis_v2(dm, dp, dc, var, alpha, correction)

    @reactive.calc
    def pw_stat_result():
        if not _cohort_confirmed(): return pd.DataFrame()
        input.btn_run_stat()
        dm = df_meta()
        dp = df_p()
        if dm.empty or dp.empty:
            return pd.DataFrame()
        var_map = {"Head Group 2": "Head Group 2",
                   "Acyl Chain Length": "Acyl Chain Length",
                   "Unsaturation": "Unsaturation"}
        var    = var_map.get(input.stat_var(), "Head Group 2")
        ctrl   = input.stat_ctrl()
        alpha  = float(input.stat_alpha())
        corr   = input.stat_correction()
        return pointwise_stat_test(dm, dp, var, ctrl, alpha, corr)

    @reactive.calc
    @reactive.event(input.btn_run_stat)
    def volcano_data():
        """Compute fold-change data for volcano plot, gated on Run Analysis button."""
        pw = pw_stat_result()
        if pw is None or pw.empty:
            return None
        var_map = {"Head Group 2": "Head Group 2",
                   "Acyl Chain Length": "Acyl Chain Length",
                   "Unsaturation": "Unsaturation"}
        var = var_map.get(input.stat_var(), "Head Group 2")
        ctrl = input.stat_ctrl()
        try:
            logfc = fold_change(df_meta(), df_p(), var, ctrl) / np.log(2)
        except Exception:
            return None
        return (pw, logfc, var)

    @reactive.calc
    def sg_madag_result():
        if not _cohort_confirmed(): return {}
        ctrl = input.sg_ctrl()
        return subgroup_analysis(df_meta(), df_p(), df_cohort(), SUBGROUP_MADAG, ctrl)

    @reactive.calc
    def sg_sph_result():
        if not _cohort_confirmed(): return {}
        ctrl = input.sg_ctrl()
        return subgroup_analysis(df_meta(), df_p(), df_cohort(), SUBGROUP_SPHINGOLIPIDS, ctrl)

    @reactive.calc
    def insights_result():
        if not _cohort_confirmed(): return pd.DataFrame(), pd.DataFrame()
        input.btn_run_insights()
        ctrl  = input.ins_ctrl()
        top_n = int(input.ins_top_n())
        return summarise_top_changes(df_meta(), df_p(), df_cohort(), ctrl, top_n)

    # ------------------------------------------------------------------ #
    #  DYNAMIC CHOICE UPDATES                                              #
    # ------------------------------------------------------------------ #

    @reactive.effect
    def _update_cohort_choices():
        dc = df_cohort()
        cohorts = dc.columns.tolist() if not dc.empty else []
        # Guess CAS9 or WT as default control
        ctrl = cohorts[0] if cohorts else None
        for c in cohorts:
            if "cas9" in c.lower() or "wt" in c.lower():
                ctrl = c; break
        for widget_id in ("cl_ctrl", "us_ctrl", "hg_ctrl", "lc_ctrl",
                          "sg_ctrl", "ins_ctrl", "stat_ctrl"):
            ui.update_select(widget_id, choices=cohorts, selected=ctrl, session=session)
        # Insights mutation filter
        non_ctrl = [c for c in cohorts if c != ctrl] if ctrl else cohorts
        ui.update_select("ins_mut_filter",
                         choices=["(all)"] + non_ctrl,
                         selected="(all)", session=session)

    @reactive.effect
    def _update_hg_filter():
        dm = df_meta_full()
        if not dm.empty and "Head Group 2" in dm.columns:
            hgs = sorted(dm["Head Group 2"].unique().tolist())
            ui.update_selectize("filter_hg", choices=hgs, session=session)
            ui.update_select("hg_bar_group", choices=hgs, session=session)

    # ------------------------------------------------------------------ #
    #  TAB 1 — UPLOAD                                                      #
    # ------------------------------------------------------------------ #

    @render.data_frame
    def tbl_filtered():
        dr = df_raw_filt()
        dm = df_meta()
        if dr.empty:
            return render.DataGrid(pd.DataFrame(
                {"Info": ["Upload CSV files to begin."]}))
        try:
            out = dm.merge(dr, on="Sample Name")
        except Exception:
            out = dr
        return render.DataGrid(out.head(500), width="100%")

    @render.data_frame
    def tbl_exps():
        exps = df_exps()
        if exps is None or exps.empty:
            return render.DataGrid(pd.DataFrame({"Info": ["No files loaded."]}))
        return render.DataGrid(exps, width="100%")

    @render.data_frame
    def tbl_others():
        others = _others_df()
        if others is None or others.empty:
            return render.DataGrid(pd.DataFrame(
                {"Info": ["No non-lipid compounds removed (or filtering disabled)."]}))
        # Show just the first few abundance columns for readability
        cols = list(others.columns[:6])
        return render.DataGrid(others[cols], width="100%")

    @render.data_frame
    def tbl_cohort_mapping():
        mapping = _cohort_mapping_df()
        if mapping.empty:
            return render.DataGrid(pd.DataFrame({"Info": ["Upload a file to see cohort mapping."]}))
        return render.DataTable(mapping, editable=True)

    @render.ui
    def cohort_confidence_badge():
        mapping = _cohort_mapping_df()
        if mapping.empty:
            return ui.p("")
        
        confidence = _cohort_confidence()
        
        if confidence >= 0.8:
            badge = ui.tags.span("Clear replicate naming detected", class_="badge bg-success")
            msg = "Replicate structure easily mapped."
        elif confidence >= 0.4:
            badge = ui.tags.span("Ambiguous sample naming", class_="badge bg-warning")
            msg = "Some samples could not be grouped into replicates."
        else:
            badge = ui.tags.span("No clear replicates detected", class_="badge bg-danger")
            msg = "Most samples treated as singletons. Please define cohorts manually."
        
        confirmed = _cohort_confirmed()
        status = (ui.tags.span(" ✅ Confirmed", class_="badge bg-success ms-2") if confirmed 
                  else ui.tags.span(" ⏳ Not confirmed", class_="badge bg-secondary ms-2"))
        
        return ui.div(badge, status, ui.p(msg, class_="text-muted small mt-1 mb-3"))

    @render.ui
    def unconfirmed_banner():
        if raw_df().empty or _cohort_confirmed():
            return ui.p("")
        return ui.div(
            ui.h6("⚠ Action Required", class_="alert-heading"),
            ui.p("Cohort mapping not yet confirmed — statistical analyses are temporarily disabled.", class_="mb-0"),
            class_="alert alert-warning mt-3 mb-3"
        )

    @render.download(filename="filtered_data.csv")
    def dl_filtered():
        dr = df_raw_filt()
        dm = df_meta()
        if dr.empty:
            yield ""; return
        try:
            out = dm.merge(dr, on="Sample Name")
        except Exception:
            out = dr
        yield out.to_csv(index=False)


    @render.ui
    def preprocess_report():
        lines = _preprocess_report_lines()
        if not lines:
            return ui.p("No preprocessing report yet — upload a file to begin.",
                        class_="text-muted small")
        items = []
        for line in lines:
            if line.startswith("\u26a0"):
                items.append(ui.tags.li(line, class_="text-warning small"))
            else:
                items.append(ui.tags.li(line, class_="small"))
        return ui.tags.ul(*items, class_="list-unstyled mb-0")

    # ------------------------------------------------------------------ #
    #  TAB 2 — PCA                                                         #
    # ------------------------------------------------------------------ #




    @render.download(filename="pca_scores.csv")
    def dl_pca_scores():
        df_pca, _, _ = pca_result()
        yield df_pca.to_csv() if not df_pca.empty else ""


    @render.download(filename="pca_variance.csv")
    def dl_pca_variance():
        import numpy as np
        _, var, _ = pca_result()
        if len(var) == 0:
            yield ""; return
        dv = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(var))],
                           "Variance": var,
                           "Cumulative": np.cumsum(var)})
        yield dv.to_csv(index=False)

    # ------------------------------------------------------------------ #
    #  TAB 3 — CHAIN LENGTH                                                #
    # ------------------------------------------------------------------ #

    def _empty_plot(msg="No data"):
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, msg, ha="center", va="center")
        return fig











    @render.data_frame
    def tbl_cl_outliers():
        d = cl_data()
        out = identify_cl_outliers(d) if d else pd.DataFrame()
        if out.empty:
            return render.DataGrid(pd.DataFrame({"Info": ["No data yet."]}))
        return render.DataGrid(out, width="100%")





    # ------------------------------------------------------------------ #
    #  TAB 4 — UNSATURATION                                                #
    # ------------------------------------------------------------------ #










    # ------------------------------------------------------------------ #
    #  TAB 5 — HEAD GROUP                                                  #
    # ------------------------------------------------------------------ #








    # ------------------------------------------------------------------ #
    #  TAB 6 — LIPID CLASS                                                 #
    # ------------------------------------------------------------------ #






    # ------------------------------------------------------------------ #
    #  TAB 7 — STATISTICS                                                  #
    # ------------------------------------------------------------------ #

    @render.data_frame
    def tbl_anova():
        anova_df, _ = stat_result()
        if anova_df is None or anova_df.empty:
            return render.DataGrid(pd.DataFrame(
                {"Info": ["Run analysis or no results."]}))
        return render.DataGrid(anova_df, width="100%")

    @render.data_frame
    def tbl_posthoc():
        _, ph_df = stat_result()
        if ph_df is None or ph_df.empty:
            return render.DataGrid(pd.DataFrame(
                {"Info": ["No significant pairwise comparisons."]}))
        return render.DataGrid(ph_df, width="100%")

    @render.data_frame
    def tbl_pw_stat():
        df = pw_stat_result()
        if df is None or df.empty:
            return render.DataGrid(pd.DataFrame(
                {"Info": ["Run analysis with a control cohort selected."]}))
        return render.DataGrid(df, width="100%")

    @render.download(filename="anova_results.csv")
    def dl_anova():
        anova_df, _ = stat_result()
        yield (anova_df.to_csv(index=False)
               if anova_df is not None and not anova_df.empty else "")

    @render.download(filename="posthoc_results.csv")
    def dl_posthoc():
        _, ph_df = stat_result()
        yield (ph_df.to_csv(index=False)
               if ph_df is not None and not ph_df.empty else "")

    @render.download(filename="perbin_stat.csv")
    def dl_pw_stat():
        df = pw_stat_result()
        yield (df.to_csv(index=False) if df is not None and not df.empty else "")

    # ------------------------------------------------------------------ #
    #  TAB 8 — SUB-GROUPS                                                  #
    # ------------------------------------------------------------------ #

    def _sg_plot(result, key, title):
        d = result()
        df = d.get(key) if d else None
        if key == "prop":
            return plot_heatmap_general(df, title, cmap="YlOrRd")
        if key == "logfc":
            return plot_fold_change_heatmap(df, title)
        return plot_zscore_heatmap(df, title)








    # ------------------------------------------------------------------ #
    #  TAB 9 — INSIGHTS                                                    #
    # ------------------------------------------------------------------ #

    @render.data_frame
    def tbl_global_changes():
        global_df, _ = insights_result()
        if global_df is None or global_df.empty:
            return render.DataGrid(pd.DataFrame(
                {"Info": ["Click 'Run Insights' with a control cohort selected."]}))
        return render.DataGrid(global_df, width="100%")

    @render.data_frame
    def tbl_specific_changes():
        _, spec_df = insights_result()
        if spec_df is None or spec_df.empty:
            return render.DataGrid(pd.DataFrame(
                {"Info": ["Click 'Run Insights' with a control cohort selected."]}))
        mut_filter = input.ins_mut_filter()
        if mut_filter and mut_filter != "(all)":
            spec_df = spec_df[spec_df["Mutation"] == mut_filter]
        return render.DataGrid(spec_df, width="100%")

    @render.download(filename="global_changes.csv")
    def dl_global_changes():
        global_df, _ = insights_result()
        yield (global_df.to_csv(index=False)
               if global_df is not None and not global_df.empty else "")

    @render.download(filename="specific_changes.csv")
    def dl_specific_changes():
        _, spec_df = insights_result()
        yield (spec_df.to_csv(index=False)
               if spec_df is not None and not spec_df.empty else "")

    def _safe_input(name, default=None):
        try:
            val = getattr(input, name)()
            return val if val else default
        except Exception:
            return default

    def _cl_fc():
        ctrl = _safe_input("cl_ctrl")
        dm = df_meta(); dp = df_p()
        if dm.empty or dp.empty or not ctrl: return _empty_plot("Select a control cohort")
        return plot_fold_change_heatmap(fold_change(dm, dp, "Acyl Chain Length", ctrl), f"Chain Length log FC vs {ctrl}")

    def _cl_ge50():
        hg = subset_headgroup_by_chain(df_meta(), df_cohort(), lambda x: x >= 50)
        return plot_heatmap_general(hg, 'Head Groups — Chain Length ≥ 50', cmap='YlOrRd')

    def _cl_le30():
        hg = subset_headgroup_by_chain(df_meta(), df_cohort(), lambda x: x <= 30)
        return plot_heatmap_general(hg, 'Head Groups — Chain Length ≤ 30', cmap='YlOrRd')

    def _cl_le20():
        hg = subset_headgroup_by_chain(df_meta(), df_cohort(), lambda x: x <= 20)
        return plot_heatmap_general(hg, 'Head Groups — Chain Length ≤ 20', cmap='YlOrRd')

    def _us_fc():
        ctrl = _safe_input("us_ctrl")
        dm = df_meta(); dp = df_p()
        if dm.empty or dp.empty or not ctrl: return _empty_plot("Select a control cohort")
        return plot_fold_change_heatmap(fold_change(dm, dp, "Unsaturation", ctrl), f"Unsaturation log FC vs {ctrl}")

    def _us_sat():
        hg = subset_headgroup_by_unsat(df_meta(), df_cohort(), lambda x: x == 0)
        return plot_heatmap_general(hg, 'Head Groups — Saturated (0 db)', cmap='YlOrRd')

    def _us_mono():
        hg = subset_headgroup_by_unsat(df_meta(), df_cohort(), lambda x: x.isin([1, 2]))
        return plot_heatmap_general(hg, 'Head Groups — Monounsaturated (1–2 db)', cmap='YlOrRd')

    def _us_poly():
        hg = subset_headgroup_by_unsat(df_meta(), df_cohort(), lambda x: x >= 3)
        return plot_heatmap_general(hg, 'Head Groups — Polyunsaturated (≥3 db)', cmap='YlOrRd')

    def _hg_donut():
        d = hg_data()
        if not d: return _empty_plot()
        return plot_donut_chart(d["cohort_prop"].mean(axis=1).sort_values(ascending=False), 'Average Head Group Distribution')

    def _hg_fc():
        ctrl = _safe_input("hg_ctrl")
        dm = df_meta(); dp = df_p()
        if dm.empty or dp.empty or not ctrl: return _empty_plot("Select a control cohort")
        return plot_fold_change_heatmap(fold_change(dm, dp, "Head Group 2", ctrl), f"Head Group log FC vs {ctrl}")

    def _lc_pie():
        d = lc_data()
        if not d: return _empty_plot()
        return plot_pie_chart(d["prop"].mean(axis=1).sort_values(ascending=False), 'Lipid Class Distribution')

    def _lc_fc():
        ctrl = _safe_input("lc_ctrl")
        dm = df_meta(); dp = df_p()
        if dm.empty or dp.empty or not ctrl: return _empty_plot("Select a control cohort")
        return plot_fold_change_heatmap(fold_change(dm, dp, "Head Group", ctrl), f"Lipid Class log FC vs {ctrl}")

    plot_registry = {
        # PCA Tab
        "plt_pca_var": lambda: plot_pca_variance(pca_result()[1]),
        "plt_pca_2d": lambda: plot_pca_2d(pca_result()[0]),
        "plt_pca_3d": lambda: plot_pca_3d(pca_result()[0]),
        "plt_pca_ellipse": lambda: plot_pca_2d_replicates(df_p(), df_exps()) if _cohort_confirmed() else _empty_plot("Replicate structure unconfirmed"),

        # Chain Length Tab
        "plt_cl_kde": lambda: plot_kde_histogram(cl_data()['long'], 'Acyl Chain Length', 'Mutation', 'Chain Length Distribution', 'Acyl Chain Length') if cl_data() else _empty_plot(),
        "plt_cl_zscore": lambda: plot_zscore_heatmap(cl_data().get('cohort_z'), 'Chain Length Z-scores') if cl_data() else _empty_plot(),
        "plt_cl_corr": lambda: plot_correlation_heatmap(cl_data().get('cohort_raw'), 'Correlation — Chain Lengths') if cl_data() else _empty_plot(),
        "plt_cl_prop": lambda: plot_heatmap_general(cl_data().get('cohort_prop'), 'Chain Length Proportions', cmap='YlOrRd') if cl_data() else _empty_plot(),
        "plt_cl_fc": _cl_fc,
        "plt_cl_gauss": lambda: plot_cl_gaussian_fit(cl_data().get('long')) if cl_data() else _empty_plot(),
        "plt_odd_chain": lambda: plot_odd_chain_bar(odd_chain_fraction(df_meta(), df_cohort())) if not df_cohort().empty else _empty_plot(),
        "plt_odd_cl_kde": lambda: plot_odd_chain_kde(df_meta(), df_p()) if not df_p().empty else _empty_plot(),
        "plt_cl_ge50": _cl_ge50,
        "plt_cl_le30": _cl_le30,
        "plt_cl_le20": _cl_le20,

        # Unsaturation Tab
        "plt_us_kde": lambda: plot_kde_histogram(us_data()['long'], 'Unsaturation', 'Mutation', 'Unsaturation Distribution', 'Unsaturation (# db)') if us_data() else _empty_plot(),
        "plt_us_zscore": lambda: plot_zscore_heatmap(us_data().get('cohort_z'), 'Unsaturation Z-scores') if us_data() else _empty_plot(),
        "plt_us_corr": lambda: plot_correlation_heatmap(us_data().get('cohort_raw'), 'Correlation — Unsaturation Levels') if us_data() else _empty_plot(),
        "plt_us_prop": lambda: plot_heatmap_general(us_data().get('cohort_prop'), 'Unsaturation Proportions', cmap='YlOrRd') if us_data() else _empty_plot(),
        "plt_us_fc": _us_fc,
        "plt_us_sat": _us_sat,
        "plt_us_mono": _us_mono,
        "plt_us_poly": _us_poly,

        # Head Group Tab
        "plt_hg_donut": _hg_donut,
        "plt_hg_zscore": lambda: plot_zscore_heatmap(hg_data().get('cohort_z'), 'Head Group Z-scores') if hg_data() else _empty_plot(),
        "plt_hg_corr": lambda: plot_correlation_heatmap(hg_data().get('cohort_raw'), 'Correlation — Head Groups') if hg_data() else _empty_plot(),
        "plt_hg_fc": _hg_fc,
        "plt_hg_prop": lambda: plot_heatmap_general(hg_data().get('cohort_prop'), 'Head Group Proportions', cmap='YlOrRd') if hg_data() else _empty_plot(),
        "plt_hg_bar": lambda: plot_hg_abundance_bar(hg_data(), _safe_input("hg_bar_group")),

        # Lipid Class Tab
        "plt_lc_pie": _lc_pie,
        "plt_lc_zscore": lambda: plot_zscore_heatmap(lc_data().get('zscore'), 'Lipid Class Z-scores') if lc_data() else _empty_plot(),
        "plt_lc_prop": lambda: plot_heatmap_general(lc_data().get('prop'), 'Lipid Class Normalised Proportions', cmap='YlOrRd') if lc_data() else _empty_plot(),
        "plt_lc_fc": _lc_fc,

        # Sub-groups Tab
        "plt_sg_madag_prop": lambda: plot_heatmap_general(sg_madag_result().get('prop'), 'Storage Lipids — Proportions', cmap='YlOrRd') if sg_madag_result() else _empty_plot(),
        "plt_sg_madag_fc": lambda: plot_fold_change_heatmap(sg_madag_result().get('logfc'), 'Storage Lipids — Log Fold Change') if sg_madag_result() else _empty_plot(),
        "plt_sg_madag_z": lambda: plot_zscore_heatmap(sg_madag_result().get('zscore'), 'Storage Lipids — Z-score') if sg_madag_result() else _empty_plot(),
        "plt_sg_sph_prop": lambda: plot_heatmap_general(sg_sph_result().get('prop'), 'Sphingolipids — Proportions', cmap='YlOrRd') if sg_sph_result() else _empty_plot(),
        "plt_sg_sph_fc": lambda: plot_fold_change_heatmap(sg_sph_result().get('logfc'), 'Sphingolipids — Log Fold Change') if sg_sph_result() else _empty_plot(),
        "plt_sg_sph_z": lambda: plot_zscore_heatmap(sg_sph_result().get('zscore'), 'Sphingolipids — Z-score') if sg_sph_result() else _empty_plot(),
        # New features
        "plt_pca_scree": lambda: plot_pca_scree(pca_result()[1]),
        "plt_pca_loadings": lambda: plot_pca_loadings(pca_result()[2], pca_result()[1]),
        "plt_pca_dendro": lambda: plot_cohort_dendrogram(df_cohort()) if not df_cohort().empty else _empty_plot(),
        "plt_volcano": lambda: plot_volcano(volcano_data()[0], volcano_data()[1], volcano_data()[2]) if volcano_data() is not None else _empty_plot("Run statistics first"),
        "plt_lipid_box": lambda: plot_lipid_boxplot(df_p(), df_exps(), input.selected_lipid()) if not df_p().empty and _cohort_confirmed() and hasattr(input, 'selected_lipid') and input.selected_lipid() else _empty_plot("Select a lipid"),
        "plt_rep_corr": lambda: plot_replicate_correlation(df_p(), df_exps()) if not df_p().empty else _empty_plot(),
    }

    plot_metadata = {
        "plt_pca_var": {"filename": "pca_explained_variance", "gated": False},
        "plt_pca_2d": {"filename": "pca_2d_scores", "gated": False},
        "plt_pca_3d": {"filename": "pca_3d_scores", "gated": False},
        "plt_pca_ellipse": {"filename": "pca_replicate_ellipses", "gated": True},
        "plt_cl_kde": {"filename": "chain_length_kde", "gated": True},
        "plt_cl_zscore": {"filename": "chain_length_zscore", "gated": True},
        "plt_cl_corr": {"filename": "chain_length_correlation", "gated": True},
        "plt_cl_prop": {"filename": "chain_length_proportions", "gated": True},
        "plt_cl_fc": {"filename": "chain_length_fold_change", "gated": True},
        "plt_cl_gauss": {"filename": "chain_length_gaussian_fit", "gated": True},
        "plt_odd_chain": {"filename": "odd_chain_lipid_fraction", "gated": True},
        "plt_odd_cl_kde": {"filename": "odd_chain_length_kde", "gated": True},
        "plt_cl_ge50": {"filename": "chain_length_ge50_headgroups", "gated": True},
        "plt_cl_le30": {"filename": "chain_length_le30_headgroups", "gated": True},
        "plt_cl_le20": {"filename": "chain_length_le20_headgroups", "gated": True},
        "plt_us_kde": {"filename": "unsaturation_kde", "gated": True},
        "plt_us_zscore": {"filename": "unsaturation_zscore", "gated": True},
        "plt_us_corr": {"filename": "unsaturation_correlation", "gated": True},
        "plt_us_prop": {"filename": "unsaturation_proportions", "gated": True},
        "plt_us_fc": {"filename": "unsaturation_fold_change", "gated": True},
        "plt_us_sat": {"filename": "unsaturation_sat_headgroups", "gated": True},
        "plt_us_mono": {"filename": "unsaturation_mono_headgroups", "gated": True},
        "plt_us_poly": {"filename": "unsaturation_poly_headgroups", "gated": True},
        "plt_hg_donut": {"filename": "head_group_donut_chart", "gated": True},
        "plt_hg_zscore": {"filename": "head_group_zscore", "gated": True},
        "plt_hg_corr": {"filename": "head_group_correlation", "gated": True},
        "plt_hg_fc": {"filename": "head_group_fold_change", "gated": True},
        "plt_hg_prop": {"filename": "head_group_proportions", "gated": True},
        "plt_hg_bar": {"filename": "head_group_abundance_bar", "gated": True},
        "plt_lc_pie": {"filename": "lipid_class_pie_chart", "gated": True},
        "plt_lc_zscore": {"filename": "lipid_class_zscore", "gated": True},
        "plt_lc_prop": {"filename": "lipid_class_proportions", "gated": True},
        "plt_lc_fc": {"filename": "lipid_class_fold_change", "gated": True},
        "plt_sg_madag_prop": {"filename": "subgroup_madag_proportions", "gated": True},
        "plt_sg_madag_fc": {"filename": "subgroup_madag_fold_change", "gated": True},
        "plt_sg_madag_z": {"filename": "subgroup_madag_zscore", "gated": True},
        "plt_sg_sph_prop": {"filename": "subgroup_sphingolipids_proportions", "gated": True},
        "plt_sg_sph_fc": {"filename": "subgroup_sphingolipids_fold_change", "gated": True},
        "plt_sg_sph_z": {"filename": "subgroup_sphingolipids_zscore", "gated": True},
        # New features
        "plt_pca_scree": {"filename": "pca_scree_plot", "gated": False},
        "plt_pca_loadings": {"filename": "pca_loadings", "gated": False},
        "plt_pca_dendro": {"filename": "cohort_dendrogram", "gated": True},
        "plt_volcano": {"filename": "volcano_plot", "gated": True},
        "plt_lipid_box": {"filename": "lipid_boxplot", "gated": True},
        "plt_rep_corr": {"filename": "replicate_correlation_qc", "gated": True},
    }

    HEATMAPS = {
        "plt_cl_zscore", "plt_cl_corr", "plt_cl_prop", "plt_cl_fc", "plt_cl_ge50", "plt_cl_le30", "plt_cl_le20",
        "plt_us_zscore", "plt_us_corr", "plt_us_prop", "plt_us_fc", "plt_us_sat", "plt_us_mono", "plt_us_poly",
        "plt_hg_zscore", "plt_hg_corr", "plt_hg_fc", "plt_hg_prop",
        "plt_lc_zscore", "plt_lc_prop", "plt_lc_fc",
        "plt_sg_madag_prop", "plt_sg_madag_fc", "plt_sg_madag_z",
        "plt_sg_sph_prop", "plt_sg_sph_fc", "plt_sg_sph_z"
    }

    # Modal observers and renderers
    @reactive.effect
    @reactive.event(input.enlarge_plot_id)
    def _show_enlargement_modal():
        pid = input.enlarge_plot_id()
        if not pid:
            return
        current_enlarged_plot.set(pid)
        ui.modal_show(
            ui.modal(
                ui.div(
                    ui.output_ui("scale_style"),
                    ui.div(
                        ui.input_slider("enlarge_scale", "Enlarge Scale", min=100, max=800, value=100, step=50, post="%"),
                        style="margin-bottom: 15px;"
                    ),
                    ui.div(
                        ui.div(
                            ui.output_ui("enlarged_plot_content"),
                            id="enlarged_plot_inner",
                            style="margin: 0 auto; padding: 10px; transition: width 0.1s ease-out;"
                        ),
                        style="overflow: auto; max-height: 70vh; max-width: 100%; border: 1px solid #ccc; background: white; text-align: center; position: relative;"
                    ),
                    style="display: flex; flex-direction: column;"
                ),
                title=f"Enlarge Plot — {plot_metadata[pid]['filename']}",
                size="xl",
                easy_close=True,
                footer=ui.modal_button("Close")
            )
        )

    @render.ui
    def scale_style():
        scale = input.enlarge_scale() or 100
        width_px = int(600 * (scale / 100.0))
        return ui.tags.style(f"""
            #enlarged_plot_inner {{
                width: {width_px}px !important;
            }}
        """)

    @render.ui
    def enlarged_plot_content():
        import base64
        pid = current_enlarged_plot()
        if not pid:
            return ui.p("No plot selected")
        
        # Get figure from registry
        fig = plot_registry[pid]()
        
        # Check hybrid enlargement strategy
        is_heatmap = pid in HEATMAPS
        
        if is_heatmap:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=600, bbox_inches="tight")
            plt.close(fig)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return ui.HTML(f'<img src="data:image/png;base64,{img_b64}" style="width: 100%; height: auto;" />')
        else:
            buf = io.StringIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
            plt.close(fig)
            svg_data = buf.getvalue()
            svg_data = svg_data.replace('<svg ', '<svg style="width: 100%; height: auto;" ', 1)
            return ui.HTML(svg_data)

    # Helper function to generate download renderers dynamically
    def make_download_handler(pid, fmt, dpi_val=None):
        meta = plot_metadata[pid]
        base_name = meta["filename"]
        ext = "eps" if fmt == "eps" else ("svg" if fmt == "svg" else ("pdf" if fmt == "pdf" else "png"))
        
        if dpi_val:
            fname = f"{base_name}_{dpi_val}dpi.{ext}"
        else:
            fname = f"{base_name}.{ext}"
            
        def download_fn():
            import sys, traceback
            try:
                print(f"ENTERED DOWNLOAD: {pid} FORMAT: {fmt}", file=sys.stderr, flush=True)
                if meta["gated"] and not _cohort_confirmed():
                    print("COHORT NOT CONFIRMED - YIELDING EMPTY", file=sys.stderr, flush=True)
                    yield b""
                    return
                print(f"CALLING REGISTRY FOR {pid}", file=sys.stderr, flush=True)
                fig = plot_registry[pid]()
                print("FIG GENERATED", file=sys.stderr, flush=True)

                buf = io.BytesIO()
                if fmt == "eps":
                    fig.savefig(buf, format="eps", bbox_inches="tight")
                elif fmt == "pdf":
                    fig.savefig(buf, format="pdf", bbox_inches="tight")
                elif fmt == "svg":
                    fig.savefig(buf, format="svg", bbox_inches="tight")
                else:
                    fig.savefig(buf, format="png", dpi=dpi_val, bbox_inches="tight")
                plt.close(fig)
                print(f"BUFFER SIZE: {len(buf.getvalue())}", file=sys.stderr, flush=True)
                yield buf.getvalue()
            except Exception as e:
                import sys, traceback
                print(f"EXCEPTION IN DOWNLOAD {pid}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                raise
        
        download_fn.__name__ = f"dl_{pid}_{fmt}"
        return render.download(filename=fname)(download_fn)

    # Helper function to generate toolbar renderers dynamically
    def make_toolbar_renderer(pid):
        @render.ui
        def toolbar_fn():
            meta = plot_metadata[pid]
            is_gated = meta["gated"]
            if is_gated and not _cohort_confirmed():
                return ui.tags.button(
                    "Export (Requires Cohort Confirmation)",
                    class_="btn btn-sm btn-outline-secondary disabled w-100",
                    type="button",
                    disabled=True
                )
            else:
                # Flat Export dropdown menu
                return ui.div(
                    ui.div(
                        ui.tags.button(
                            "Export ▾",
                            class_="btn btn-sm btn-outline-secondary dropdown-toggle",
                            type="button",
                            data_bs_toggle="dropdown",
                            aria_expanded="false"
                        ),
                        ui.tags.ul(
                            ui.tags.li(
                                ui.tags.a(
                                    "Enlarge Plot",
                                    class_="dropdown-item",
                                    href="#",
                                    onclick=f"Shiny.setInputValue('enlarge_plot_id', '{pid}', {{priority: 'event'}}); return false;"
                                )
                            ),
                            ui.tags.li(ui.tags.hr(class_="dropdown-divider")),
                            ui.tags.li(ui.download_link(f"dl_{pid}_png300", "Download PNG (300 DPI)", class_="dropdown-item")),
                            ui.tags.li(ui.download_link(f"dl_{pid}_png600", "Download PNG (600 DPI)", class_="dropdown-item")),
                            ui.tags.li(ui.download_link(f"dl_{pid}_pdf", "Download PDF (Vector)", class_="dropdown-item")),
                            ui.tags.li(ui.download_link(f"dl_{pid}_svg", "Download SVG (Vector)", class_="dropdown-item")),
                            ui.tags.li(ui.download_link(f"dl_{pid}_eps", "Download EPS (Vector - No Transparency)", class_="dropdown-item")),
                            class_="dropdown-menu dropdown-menu-end"
                        ),
                        class_="dropdown"
                    ),
                    class_="plot-toolbar-container"
                )
        return toolbar_fn

    # Loop to register all toolbar UI outputs and download handlers

    # Explicit @render.plot definitions generated

    @render.plot
    def plt_pca_var():
        if plot_metadata["plt_pca_var"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_pca_var"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_pca_var: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_pca_2d():
        if plot_metadata["plt_pca_2d"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_pca_2d"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_pca_2d: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_pca_3d():
        if plot_metadata["plt_pca_3d"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_pca_3d"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_pca_3d: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_pca_ellipse():
        if plot_metadata["plt_pca_ellipse"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_pca_ellipse"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_pca_ellipse: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_cl_kde():
        if plot_metadata["plt_cl_kde"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_cl_kde"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_cl_kde: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_cl_zscore():
        if plot_metadata["plt_cl_zscore"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_cl_zscore"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_cl_zscore: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_cl_corr():
        if plot_metadata["plt_cl_corr"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_cl_corr"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_cl_corr: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_cl_prop():
        if plot_metadata["plt_cl_prop"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_cl_prop"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_cl_prop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_cl_fc():
        if plot_metadata["plt_cl_fc"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_cl_fc"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_cl_fc: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_cl_gauss():
        if plot_metadata["plt_cl_gauss"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_cl_gauss"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_cl_gauss: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_odd_chain():
        if plot_metadata["plt_odd_chain"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_odd_chain"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_odd_chain: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_odd_cl_kde():
        if plot_metadata["plt_odd_cl_kde"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_odd_cl_kde"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_odd_cl_kde: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_cl_ge50():
        if plot_metadata["plt_cl_ge50"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_cl_ge50"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_cl_ge50: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_cl_le30():
        if plot_metadata["plt_cl_le30"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_cl_le30"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_cl_le30: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_cl_le20():
        if plot_metadata["plt_cl_le20"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_cl_le20"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_cl_le20: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_us_kde():
        if plot_metadata["plt_us_kde"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_us_kde"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_us_kde: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_us_zscore():
        if plot_metadata["plt_us_zscore"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_us_zscore"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_us_zscore: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_us_corr():
        if plot_metadata["plt_us_corr"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_us_corr"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_us_corr: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_us_prop():
        if plot_metadata["plt_us_prop"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_us_prop"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_us_prop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_us_fc():
        if plot_metadata["plt_us_fc"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_us_fc"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_us_fc: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_us_sat():
        if plot_metadata["plt_us_sat"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_us_sat"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_us_sat: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_us_mono():
        if plot_metadata["plt_us_mono"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_us_mono"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_us_mono: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_us_poly():
        if plot_metadata["plt_us_poly"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_us_poly"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_us_poly: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_hg_donut():
        if plot_metadata["plt_hg_donut"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_hg_donut"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_hg_donut: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_hg_zscore():
        if plot_metadata["plt_hg_zscore"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_hg_zscore"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_hg_zscore: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_hg_corr():
        if plot_metadata["plt_hg_corr"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_hg_corr"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_hg_corr: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_hg_fc():
        if plot_metadata["plt_hg_fc"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_hg_fc"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_hg_fc: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_hg_prop():
        if plot_metadata["plt_hg_prop"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_hg_prop"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_hg_prop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_hg_bar():
        if plot_metadata["plt_hg_bar"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_hg_bar"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_hg_bar: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_lc_pie():
        if plot_metadata["plt_lc_pie"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_lc_pie"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_lc_pie: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_lc_zscore():
        if plot_metadata["plt_lc_zscore"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_lc_zscore"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_lc_zscore: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_lc_prop():
        if plot_metadata["plt_lc_prop"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_lc_prop"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_lc_prop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_lc_fc():
        if plot_metadata["plt_lc_fc"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_lc_fc"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_lc_fc: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_sg_madag_prop():
        if plot_metadata["plt_sg_madag_prop"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_sg_madag_prop"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_sg_madag_prop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_sg_madag_fc():
        if plot_metadata["plt_sg_madag_fc"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_sg_madag_fc"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_sg_madag_fc: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_sg_madag_z():
        if plot_metadata["plt_sg_madag_z"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_sg_madag_z"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_sg_madag_z: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_sg_sph_prop():
        if plot_metadata["plt_sg_sph_prop"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_sg_sph_prop"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_sg_sph_prop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_sg_sph_fc():
        if plot_metadata["plt_sg_sph_fc"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_sg_sph_fc"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_sg_sph_fc: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_sg_sph_z():
        if plot_metadata["plt_sg_sph_z"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_sg_sph_z"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_sg_sph_z: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_pca_scree():
        print("DEBUG: plt_pca_scree render called", flush=True)
        if plot_metadata["plt_pca_scree"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_pca_scree"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_pca_scree: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_pca_loadings():
        print("DEBUG: plt_pca_loadings render called", flush=True)
        if plot_metadata["plt_pca_loadings"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_pca_loadings"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_pca_loadings: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_pca_dendro():
        print("DEBUG: plt_pca_dendro render called", flush=True)
        if plot_metadata["plt_pca_dendro"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_pca_dendro"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_pca_dendro: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_volcano():
        print("DEBUG: plt_volcano render called", flush=True)
        if plot_metadata["plt_volcano"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_volcano"]()
        except Exception as e:
            if type(e).__name__ == "SilentException":
                return _empty_plot("Run statistics first")
            import sys, traceback
            print(f"CRASH IN PLOT plt_volcano: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_lipid_box():
        print("DEBUG: plt_lipid_box render called", flush=True)
        if plot_metadata["plt_lipid_box"]["gated"] and not _cohort_confirmed():
            return _empty_plot("Replicate structure unconfirmed")
        try:
            return plot_registry["plt_lipid_box"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_lipid_box: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.plot
    def plt_rep_corr():
        print("DEBUG: plt_rep_corr render called", flush=True)
        if not _cohort_confirmed():
            return _empty_plot("Confirm cohort mapping first")
        if not _qc_run():
            return _empty_plot("Click 'Run QC' above to generate correlation matrix")
        try:
            return plot_registry["plt_rep_corr"]()
        except Exception as e:
            import sys, traceback
            print(f"CRASH IN PLOT plt_rep_corr: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return _empty_plot(f"Error rendering: {e}")

    @render.ui
    def lipid_select_ui():
        print("DEBUG: lipid_select_ui render called", flush=True)
        dp = df_p()
        if dp.empty:
            return ui.p("Upload data first.", class_="text-muted")
        if "Sample Name" in dp.columns:
            choices = sorted(dp["Sample Name"].unique().tolist())
        else:
            choices = sorted(dp.index.tolist())
        return ui.input_selectize(
            "selected_lipid", "Select Lipid:", choices=choices,
            selected=choices[0] if choices else None
        )

    for plot_id in plot_registry.keys():
        output(make_toolbar_renderer(plot_id), id=f"tb_{plot_id}")
        output(make_download_handler(plot_id, "png300", 300), id=f"dl_{plot_id}_png300")
        output(make_download_handler(plot_id, "png600", 600), id=f"dl_{plot_id}_png600")
        output(make_download_handler(plot_id, "pdf"), id=f"dl_{plot_id}_pdf")
        output(make_download_handler(plot_id, "svg"), id=f"dl_{plot_id}_svg")
        output(make_download_handler(plot_id, "eps"), id=f"dl_{plot_id}_eps")

    # Helper function to generate tab bulk ZIP downloads
    def make_zip_handler(tab_name, pids):
        import zipfile
        def zip_fn():
            import sys, traceback
            try:
                fmt = getattr(input, f"bulk_fmt_{tab_name}")()
                ext = "eps" if fmt == "eps" else ("svg" if fmt == "svg" else ("pdf" if fmt == "pdf" else "png"))
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for pid in pids:
                        meta = plot_metadata[pid]
                        if meta["gated"] and not _cohort_confirmed():
                            continue
                            
                        try:
                            fig = plot_registry[pid]()
                            
                            # If plot_registry returns None due to another error internally
                            if fig is None:
                                zip_file.writestr(f"ERROR_{pid}.txt", "Plot returned None.")
                                continue
                                
                            img_buf = io.BytesIO()
                            if fmt == "png300":
                                fig.savefig(img_buf, format="png", dpi=300, bbox_inches="tight")
                            elif fmt == "png600":
                                fig.savefig(img_buf, format="png", dpi=600, bbox_inches="tight")
                            elif fmt == "pdf":
                                fig.savefig(img_buf, format="pdf", bbox_inches="tight")
                            elif fmt == "svg":
                                fig.savefig(img_buf, format="svg", bbox_inches="tight")
                            elif fmt == "eps":
                                fig.savefig(img_buf, format="eps", bbox_inches="tight")
                            plt.close(fig)
                            
                            zip_file.writestr(f"{meta['filename']}.{ext}", img_buf.getvalue())
                        except Exception as e:
                            import traceback
                            error_msg = f"Error generating {pid}: {str(e)}\n\n{traceback.format_exc()}"
                            zip_file.writestr(f"ERROR_{pid}.txt", error_msg)
                            
                yield zip_buffer.getvalue()
            except Exception as e:
                import sys, traceback
                print(f"EXCEPTION IN ZIP DOWNLOAD {tab_name}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                raise
        
        zip_fn.__name__ = f"dl_zip_{tab_name}"
        return render.download(filename=f"lipidomics_{tab_name}_plots.zip")(zip_fn)

    # Register ZIP downloads for all 6 tabs
    tabs_plots = {
        "pca": ["plt_pca_var", "plt_pca_2d", "plt_pca_3d", "plt_pca_ellipse", "plt_pca_scree", "plt_pca_loadings", "plt_pca_dendro", "plt_rep_corr"],
        "cl": ["plt_cl_kde", "plt_cl_zscore", "plt_cl_corr", "plt_cl_prop", "plt_cl_fc", "plt_cl_gauss", "plt_odd_chain", "plt_odd_cl_kde", "plt_cl_ge50", "plt_cl_le30", "plt_cl_le20"],
        "us": ["plt_us_kde", "plt_us_zscore", "plt_us_corr", "plt_us_prop", "plt_us_fc", "plt_us_sat", "plt_us_mono", "plt_us_poly"],
        "hg": ["plt_hg_donut", "plt_hg_zscore", "plt_hg_corr", "plt_hg_fc", "plt_hg_prop", "plt_hg_bar"],
        "lc": ["plt_lc_pie", "plt_lc_zscore", "plt_lc_prop", "plt_lc_fc"],
        "sg": ["plt_sg_madag_prop", "plt_sg_madag_fc", "plt_sg_madag_z", "plt_sg_sph_prop", "plt_sg_sph_fc", "plt_sg_sph_z"],
        "stat": ["plt_volcano"],
        "lipid": ["plt_lipid_box"],
    }
    
    for tname, pids in tabs_plots.items():
        output(make_zip_handler(tname, pids), id=f"dl_zip_{tname}")


# ---------------------------------------------------------------------------
# App object
# ---------------------------------------------------------------------------

app = App(app_ui, server)
app.sanitize_errors = True

