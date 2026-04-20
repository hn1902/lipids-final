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
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

try:
    from app.analysis import (
        load_and_deduplicate_data,
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
        norm_col,
        # plots
        plot_pca_variance,
        plot_pca_2d,
        plot_pca_3d,
        plot_kde_histogram,
        plot_zscore_heatmap,
        plot_correlation_heatmap,
        plot_fold_change_heatmap,
        plot_heatmap_general,
        plot_donut_chart,
        plot_pie_chart,
        plot_odd_chain_bar,
    )
except ImportError:
    from analysis import (
        load_and_deduplicate_data,
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
        norm_col,
        plot_pca_variance,
        plot_pca_2d,
        plot_pca_3d,
        plot_kde_histogram,
        plot_zscore_heatmap,
        plot_correlation_heatmap,
        plot_fold_change_heatmap,
        plot_heatmap_general,
        plot_donut_chart,
        plot_pie_chart,
        plot_odd_chain_bar,
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _card(title, *contents, id=None):
    return ui.card(ui.card_header(title), *contents,
                   id=id, full_screen=True)


# ---------------------------------------------------------------------------
# UI definition
# ---------------------------------------------------------------------------

app_ui = ui.page_navbar(

    # ===========================  Tab 1: Upload  ===========================
    ui.nav_panel(
        "📁 Upload Data",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Data Files"),
                ui.input_file("data_files",
                              "Upload CSV file(s) (one per lipid class):",
                              accept=".csv", multiple=True),
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
                ui.hr(),
                ui.input_action_button("btn_process", "Process / Apply Filters",
                                       class_="btn-primary btn-sm w-100"),
                width=290,
            ),
            ui.layout_column_wrap(
                _card("Advanced Data Format Options",
                      ui.p("Use these settings if your CSV has a non-standard layout (e.g., Lipotype files).", class_="text-muted mb-2"),
                      ui.layout_columns(
                          ui.input_text("adv_idx_col", "Lipid Species Column Name:", placeholder="e.g. Shorthand Notation"),
                          ui.input_numeric("adv_skip_rows", "Header Row Offset (Skip N rows):", 0, min=0),
                          ui.input_numeric("adv_num_idx", "Drop N initial metadata columns:", 0, min=0),
                          col_widths=(4, 4, 4)
                      )
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
                _card("Processed / Filtered Data",
                      ui.output_data_frame("tbl_filtered"),
                      ui.download_button("dl_filtered", "Download CSV",
                                         class_="btn-sm btn-outline-secondary mt-2")),
                _card("Sample / Experiment Table",
                      ui.output_data_frame("tbl_exps")),
                width=1,
            ),
        ),
    ),

    # ===========================  Tab 2: PCA  ==============================
    ui.nav_panel(
        "📊 PCA",
        ui.layout_column_wrap(
            _card("Explained Variance", ui.output_plot("plt_pca_var")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("PCA — 2D Scores (PC1 vs PC2)", ui.output_plot("plt_pca_2d")),
            _card("PCA — 3D Scores",              ui.output_plot("plt_pca_3d")),
            width="1/2",
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
        ui.layout_column_wrap(
            _card("KDE / Histogram — Chain Length Distribution",
                  ui.output_plot("plt_cl_kde")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Z-score Heatmap",         ui.output_plot("plt_cl_zscore")),
            _card("Correlation Matrix",      ui.output_plot("plt_cl_corr")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Proportions Heatmap",     ui.output_plot("plt_cl_prop")),
            _card("Fold Change vs Control",
                  ui.layout_sidebar(
                      ui.sidebar(ui.input_select("cl_ctrl", "Control:", choices=[]),
                                 width=200),
                      ui.output_plot("plt_cl_fc"))),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Odd-Chain Lipid Fraction by Cohort",
                  ui.output_plot("plt_odd_chain")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Head Groups — Chain Length ≥ 50",
                  ui.output_plot("plt_cl_ge50")),
            _card("Head Groups — Chain Length ≤ 30",
                  ui.output_plot("plt_cl_le30")),
            _card("Head Groups — Chain Length ≤ 20",
                  ui.output_plot("plt_cl_le20")),
            width="1/3",
        ),
    ),

    # =======================  Tab 4: Unsaturation  ========================
    ui.nav_panel(
        "〰 Unsaturation",
        ui.layout_column_wrap(
            _card("KDE / Histogram — Unsaturation Distribution",
                  ui.output_plot("plt_us_kde")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Z-score Heatmap",    ui.output_plot("plt_us_zscore")),
            _card("Correlation Matrix", ui.output_plot("plt_us_corr")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Proportions Heatmap", ui.output_plot("plt_us_prop")),
            _card("Fold Change vs Control",
                  ui.layout_sidebar(
                      ui.sidebar(ui.input_select("us_ctrl", "Control:", choices=[]),
                                 width=200),
                      ui.output_plot("plt_us_fc"))),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Saturated (0 db) — Head Group Distribution",
                  ui.output_plot("plt_us_sat")),
            _card("Monounsaturated (1–2 db) — Head Group Distribution",
                  ui.output_plot("plt_us_mono")),
            _card("Polyunsaturated (≥3 db) — Head Group Distribution",
                  ui.output_plot("plt_us_poly")),
            width="1/3",
        ),
    ),

    # ========================  Tab 5: Head Group  ==========================
    ui.nav_panel(
        "🧬 Head Group",
        ui.layout_column_wrap(
            _card("Head Group Distribution (Donut)",
                  ui.output_plot("plt_hg_donut")),
            _card("Z-score Heatmap",
                  ui.output_plot("plt_hg_zscore")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Correlation Matrix",
                  ui.output_plot("plt_hg_corr")),
            _card("Fold Change vs Control",
                  ui.layout_sidebar(
                      ui.sidebar(ui.input_select("hg_ctrl", "Control:", choices=[]),
                                 width=200),
                      ui.output_plot("plt_hg_fc"))),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Proportions Heatmap",
                  ui.output_plot("plt_hg_prop")),
            width=1,
        ),
        ui.layout_column_wrap(
            _card("Bar Plot — Selected Head Group by Cohort",
                  ui.layout_sidebar(
                      ui.sidebar(
                          ui.input_select("hg_bar_group", "Head Group:", choices=[]),
                          width=200),
                      ui.output_plot("plt_hg_bar"))),
            width=1,
        ),
    ),

    # ========================  Tab 6: Lipid Class  =========================
    ui.nav_panel(
        "🫧 Lipid Class",
        ui.layout_column_wrap(
            _card("Lipid Class Distribution (Pie)",
                  ui.output_plot("plt_lc_pie")),
            _card("Z-score Heatmap",
                  ui.output_plot("plt_lc_zscore")),
            width="1/2",
        ),
        ui.layout_column_wrap(
            _card("Normalised Proportions Heatmap",
                  ui.output_plot("plt_lc_prop")),
            _card("Fold Change vs Control",
                  ui.layout_sidebar(
                      ui.sidebar(ui.input_select("lc_ctrl", "Control:", choices=[]),
                                 width=200),
                      ui.output_plot("plt_lc_fc"))),
            width="1/2",
        ),
    ),

    # =======================  Tab 7: Statistics  ===========================
    ui.nav_panel(
        "📈 Statistics",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("ANOVA / Post-hoc Controls"),
                ui.input_select("stat_var", "Group variable:",
                                {"Head Group 2":   "Head Group 2",
                                 "Acyl Chain Length": "Chain Length",
                                 "Unsaturation":   "Unsaturation"}),
                ui.input_numeric("stat_alpha", "Significance level (α):",
                                 0.05, min=0.001, max=0.5, step=0.001),
                ui.input_action_button("btn_run_stat", "Run Analysis",
                                       class_="btn-primary btn-sm w-100"),
                width=250,
            ),
            ui.layout_column_wrap(
                _card("One-Way ANOVA Results",
                      ui.output_data_frame("tbl_anova"),
                      ui.download_button("dl_anova", "Download ANOVA CSV",
                                         class_="btn-sm btn-outline-secondary mt-2")),
                width=1,
            ),
            ui.layout_column_wrap(
                _card("Pairwise Post-hoc (Holm–Sidak)",
                      ui.output_data_frame("tbl_posthoc"),
                      ui.download_button("dl_posthoc", "Download Post-hoc CSV",
                                         class_="btn-sm btn-outline-secondary mt-2")),
                width=1,
            ),
        ),
    ),

    title=ui.tags.span(
        ui.tags.b("Lipidomics"),
        ui.tags.span(" Data Analysis", style="font-weight:300"),
    ),
    bg="#1a1a2e",
    inverse=True,
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session):

    # ------------------------------------------------------------------ #
    #  REACTIVE CALCULATIONS                                               #
    # ------------------------------------------------------------------ #

    @reactive.calc
    def raw_df():
        files = input.data_files()
        if files is None:
            return pd.DataFrame()
            
        adv_idx = input.adv_idx_col()
        num_idx = input.adv_num_idx()
        skip = input.adv_skip_rows()
        
        return load_and_deduplicate_data(
            files,
            idx_col=adv_idx,
            num_idx=int(num_idx) if num_idx else 0,
            skip_rows=int(skip) if skip else 0
        )

    @reactive.calc
    def header_df():
        hf = input.header_file()
        if not hf:
            return None
        try:
            return pd.read_csv(hf[0]["datapath"], header=None)
        except Exception:
            return None

    @reactive.calc
    def df_exps():
        df = raw_df()
        if df.empty:
            return pd.DataFrame()
        hdr = header_df()
        crow = max(0, int(input.cohort_row()) - 1) if input.cohort_row() else 1
        return extract_experiments(df, header_df=hdr, cohort_row_idx=crow)

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

    @reactive.calc
    def df_filtered_pair():
        input.btn_process()  # trigger on button press
        dr = raw_df()
        dm = df_meta_full()
        if dr.empty or dm.empty:
            return pd.DataFrame(), pd.DataFrame()

        fhg = list(input.filter_hg()) if input.filter_hg() else None
        mc  = int(input.min_chain())
        mu  = int(input.max_unsat())
        rb  = bool(input.remove_blank())
        bth = float(input.blank_threshold())

        return filter_data(dr, dm, filter_hg=fhg, min_chain=mc, max_unsat=mu,
                           remove_blank=rb, blank_threshold=bth)

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
        dp = df_p()
        exps = df_exps()
        if dp.empty or exps.empty:
            return pd.DataFrame()
        return aggregate_by_cohort(dp, exps, method=input.agg_method())

    @reactive.calc
    def pca_result():
        dc = df_cohort()
        if dc.empty:
            return pd.DataFrame(), [], pd.DataFrame()
        return perform_pca(dc)

    @reactive.calc
    def cl_data():
        return chain_length_analysis(df_meta(), df_p(), df_cohort())

    @reactive.calc
    def us_data():
        return unsaturation_analysis(df_meta(), df_p(), df_cohort())

    @reactive.calc
    def hg_data():
        return headgroup_analysis(df_meta(), df_p(), df_cohort())

    @reactive.calc
    def lc_data():
        return lipid_class_analysis(df_meta(), df_cohort())

    @reactive.calc
    def stat_result():
        input.btn_run_stat()
        dm = df_meta()
        dp = df_p()
        dc = df_cohort()
        if dm.empty or dp.empty or dc.empty:
            return pd.DataFrame(), pd.DataFrame()
        var_map = {"Head Group 2": "Head Group 2",
                   "Acyl Chain Length": "Acyl Chain Length",
                   "Unsaturation": "Unsaturation"}
        var   = var_map.get(input.stat_var(), "Head Group 2")
        alpha = float(input.stat_alpha())
        return statistical_analysis(dm, dp, dc, var, alpha)

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
        for widget_id in ("cl_ctrl", "us_ctrl", "hg_ctrl", "lc_ctrl"):
            ui.update_select(widget_id, choices=cohorts, selected=ctrl, session=session)

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

    # ------------------------------------------------------------------ #
    #  TAB 2 — PCA                                                         #
    # ------------------------------------------------------------------ #

    @render.plot
    def plt_pca_var():
        _, var, _ = pca_result()
        return plot_pca_variance(var)

    @render.plot
    def plt_pca_2d():
        df_pca, _, _ = pca_result()
        return plot_pca_2d(df_pca)

    @render.plot
    def plt_pca_3d():
        df_pca, _, _ = pca_result()
        return plot_pca_3d(df_pca)

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

    @render.plot
    def plt_cl_kde():
        d = cl_data()
        if not d:
            return _empty_plot()
        return plot_kde_histogram(d["long"], "Acyl Chain Length", "Mutation",
                                  "Chain Length Distribution", "Acyl Chain Length")

    @render.plot
    def plt_cl_zscore():
        d = cl_data()
        return plot_zscore_heatmap(d.get("cohort_z") if d else None,
                                   "Chain Length Z-scores")

    @render.plot
    def plt_cl_corr():
        d = cl_data()
        return plot_correlation_heatmap(d.get("cohort_raw") if d else None,
                                        "Correlation — Chain Lengths")

    @render.plot
    def plt_cl_prop():
        d = cl_data()
        return plot_heatmap_general(d.get("cohort_prop") if d else None,
                                    "Chain Length Proportions", cmap="YlOrRd")

    @render.plot
    def plt_cl_fc():
        ctrl = input.cl_ctrl()
        dm = df_meta(); dp = df_p()
        if dm.empty or dp.empty or not ctrl:
            return _empty_plot("Select a control cohort")
        df_log = fold_change(dm, dp, "Acyl Chain Length", ctrl)
        return plot_fold_change_heatmap(df_log, f"Chain Length log FC vs {ctrl}")

    @render.plot
    def plt_odd_chain():
        return plot_odd_chain_bar(odd_chain_fraction(df_meta(), df_cohort()))

    @render.plot
    def plt_cl_ge50():
        hg = subset_headgroup_by_chain(df_meta(), df_cohort(), lambda x: x >= 50)
        return plot_heatmap_general(hg, "Head Groups — Chain Length ≥ 50",
                                    cmap="YlOrRd")

    @render.plot
    def plt_cl_le30():
        hg = subset_headgroup_by_chain(df_meta(), df_cohort(), lambda x: x <= 30)
        return plot_heatmap_general(hg, "Head Groups — Chain Length ≤ 30",
                                    cmap="YlOrRd")

    @render.plot
    def plt_cl_le20():
        hg = subset_headgroup_by_chain(df_meta(), df_cohort(), lambda x: x <= 20)
        return plot_heatmap_general(hg, "Head Groups — Chain Length ≤ 20",
                                    cmap="YlOrRd")

    # ------------------------------------------------------------------ #
    #  TAB 4 — UNSATURATION                                                #
    # ------------------------------------------------------------------ #

    @render.plot
    def plt_us_kde():
        d = us_data()
        if not d:
            return _empty_plot()
        return plot_kde_histogram(d["long"], "Unsaturation", "Mutation",
                                  "Unsaturation Distribution", "Unsaturation (# db)")

    @render.plot
    def plt_us_zscore():
        d = us_data()
        return plot_zscore_heatmap(d.get("cohort_z") if d else None,
                                   "Unsaturation Z-scores")

    @render.plot
    def plt_us_corr():
        d = us_data()
        return plot_correlation_heatmap(d.get("cohort_raw") if d else None,
                                        "Correlation — Unsaturation Levels")

    @render.plot
    def plt_us_prop():
        d = us_data()
        return plot_heatmap_general(d.get("cohort_prop") if d else None,
                                    "Unsaturation Proportions", cmap="YlOrRd")

    @render.plot
    def plt_us_fc():
        ctrl = input.us_ctrl()
        dm = df_meta(); dp = df_p()
        if dm.empty or dp.empty or not ctrl:
            return _empty_plot("Select a control cohort")
        df_log = fold_change(dm, dp, "Unsaturation", ctrl)
        return plot_fold_change_heatmap(df_log, f"Unsaturation log FC vs {ctrl}")

    @render.plot
    def plt_us_sat():
        hg = subset_headgroup_by_unsat(df_meta(), df_cohort(), lambda x: x == 0)
        return plot_heatmap_general(hg, "Head Groups — Saturated (0 db)",
                                    cmap="YlOrRd")

    @render.plot
    def plt_us_mono():
        hg = subset_headgroup_by_unsat(df_meta(), df_cohort(),
                                       lambda x: x.isin([1, 2]))
        return plot_heatmap_general(hg, "Head Groups — Monounsaturated (1–2 db)",
                                    cmap="YlOrRd")

    @render.plot
    def plt_us_poly():
        hg = subset_headgroup_by_unsat(df_meta(), df_cohort(), lambda x: x >= 3)
        return plot_heatmap_general(hg, "Head Groups — Polyunsaturated (≥3 db)",
                                    cmap="YlOrRd")

    # ------------------------------------------------------------------ #
    #  TAB 5 — HEAD GROUP                                                  #
    # ------------------------------------------------------------------ #

    @render.plot
    def plt_hg_donut():
        d = hg_data()
        if not d:
            return _empty_plot()
        mean_prop = d["cohort_prop"].mean(axis=1).sort_values(ascending=False)
        return plot_donut_chart(mean_prop, "Average Head Group Distribution")

    @render.plot
    def plt_hg_zscore():
        d = hg_data()
        return plot_zscore_heatmap(d.get("cohort_z") if d else None,
                                   "Head Group Z-scores")

    @render.plot
    def plt_hg_corr():
        d = hg_data()
        return plot_correlation_heatmap(d.get("cohort_raw") if d else None,
                                        "Correlation — Head Groups")

    @render.plot
    def plt_hg_fc():
        ctrl = input.hg_ctrl()
        dm = df_meta(); dp = df_p()
        if dm.empty or dp.empty or not ctrl:
            return _empty_plot("Select a control cohort")
        df_log = fold_change(dm, dp, "Head Group 2", ctrl)
        return plot_fold_change_heatmap(df_log, f"Head Group log FC vs {ctrl}")

    @render.plot
    def plt_hg_prop():
        d = hg_data()
        return plot_heatmap_general(d.get("cohort_prop") if d else None,
                                    "Head Group Proportions", cmap="YlOrRd")

    @render.plot
    def plt_hg_bar():
        d   = hg_data()
        grp = input.hg_bar_group()
        if not d or not grp:
            return _empty_plot("Select a head group")
        raw = d["cohort_raw"]
        if grp not in raw.index:
            return _empty_plot(f"'{grp}' not found")
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

    # ------------------------------------------------------------------ #
    #  TAB 6 — LIPID CLASS                                                 #
    # ------------------------------------------------------------------ #

    @render.plot
    def plt_lc_pie():
        d = lc_data()
        if not d:
            return _empty_plot()
        mean_prop = d["prop"].mean(axis=1).sort_values(ascending=False)
        return plot_pie_chart(mean_prop, "Lipid Class Distribution")

    @render.plot
    def plt_lc_zscore():
        d = lc_data()
        return plot_zscore_heatmap(d.get("zscore") if d else None,
                                   "Lipid Class Z-scores")

    @render.plot
    def plt_lc_prop():
        d = lc_data()
        return plot_heatmap_general(d.get("prop") if d else None,
                                    "Lipid Class Normalised Proportions",
                                    cmap="YlOrRd")

    @render.plot
    def plt_lc_fc():
        ctrl = input.lc_ctrl()
        dm = df_meta(); dp = df_p()
        if dm.empty or dp.empty or not ctrl:
            return _empty_plot("Select a control cohort")
        df_log = fold_change(dm, dp, "Head Group", ctrl)
        return plot_fold_change_heatmap(df_log, f"Lipid Class log FC vs {ctrl}")

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


# ---------------------------------------------------------------------------
# App object
# ---------------------------------------------------------------------------

app = App(app_ui, server)
