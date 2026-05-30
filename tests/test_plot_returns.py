import os
import sys
import ast
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.analysis import (
    plot_pca_variance,
    plot_pca_2d,
    plot_pca_3d,
    plot_pca_2d_replicates,
    plot_kde_histogram,
    plot_zscore_heatmap,
    plot_correlation_heatmap,
    plot_heatmap_general,
    plot_fold_change_heatmap,
    plot_donut_chart,
    plot_pie_chart,
    plot_odd_chain_bar,
    plot_cl_gaussian_fit,
    plot_odd_chain_kde,
    perform_pca,
    make_df_p,
    extract_metadata,
    aggregate_by_cohort,
    chain_length_analysis,
    unsaturation_analysis,
    headgroup_analysis,
    lipid_class_analysis,
    odd_chain_fraction,
)
from app.app import plot_hg_abundance_bar
from app.preprocessing import preprocess_raw_metabolomics_export, suggest_cohort_mapping

def test_registry_ast_coverage():
    app_py_path = os.path.join(os.path.dirname(__file__), "../app/app.py")
    with open(app_py_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    
    render_plots = set()
    ui_output_plots = set()
    registry_keys = set()
    
    # 1. Extract @render.plot functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Attribute) and dec.attr == "plot" and isinstance(dec.value, ast.Name) and dec.value.id == "render") or \
                   (isinstance(dec, ast.Name) and dec.id == "plot"):
                    render_plots.add(node.name)
            
            # 2. Extract plot_registry assignment inside server()
            if node.name == "server":
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Assign):
                        for target in subnode.targets:
                            if isinstance(target, ast.Name) and target.id == "plot_registry":
                                if isinstance(subnode.value, ast.Dict):
                                    for k in subnode.value.keys:
                                        if isinstance(k, ast.Constant):
                                            registry_keys.add(k.value)
                                        elif isinstance(k, ast.Str): # Support older Python
                                            registry_keys.add(k.s)
    
    # 3. Extract ui.output_plot("id") calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_output_plot = False
            if isinstance(func, ast.Attribute) and func.attr == "output_plot" and isinstance(func.value, ast.Name) and func.value.id == "ui":
                is_output_plot = True
            elif isinstance(func, ast.Name) and func.id == "output_plot":
                is_output_plot = True
                
            if is_output_plot and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant):
                    ui_output_plots.add(arg.value)
                elif isinstance(arg, ast.Str):
                    ui_output_plots.add(arg.s)
                    
    assert render_plots == ui_output_plots, f"Mismatch between render.plot and ui.output_plot: Render: {render_plots - ui_output_plots}, UI: {ui_output_plots - render_plots}"
    assert render_plots == registry_keys, f"Mismatch between render.plot and plot_registry: Render: {render_plots - registry_keys}, Registry: {registry_keys - render_plots}"


def load_dataset_and_run_analysis(file_path, sheet_name=0):
    # Load using preprocess_raw_metabolomics_export
    file_infos = [{"name": os.path.basename(file_path), "datapath": file_path}]
    df_clean, report = preprocess_raw_metabolomics_export(file_infos, sheet_name=sheet_name)
    
    # Generate cohort mapping
    sample_cols = [c for c in df_clean.columns if c != "Sample Name"]
    mapping_df, conf = suggest_cohort_mapping(sample_cols)
    
    # Standardize dataframes for analysis
    exps = pd.DataFrame({
        "Exp": mapping_df["Sample"].tolist(),
        "Mutation": mapping_df["Cohort"].tolist(),
        "Replicate": mapping_df["Replicate"].tolist()
    })
    
    df_p = make_df_p(df_clean, exps)
    df_cohort = aggregate_by_cohort(df_p, exps)
    df_meta = extract_metadata(df_clean)
    
    # Calculate everything
    pca_res = perform_pca(df_cohort)
    cl_res = chain_length_analysis(df_meta, df_p, df_cohort)
    us_res = unsaturation_analysis(df_meta, df_p, df_cohort)
    hg_res = headgroup_analysis(df_meta, df_p, df_cohort)
    lc_res = lipid_class_analysis(df_meta, df_cohort)
    
    return {
        "df_clean": df_clean,
        "df_p": df_p,
        "exps": exps,
        "df_cohort": df_cohort,
        "df_meta": df_meta,
        "pca_res": pca_res,
        "cl_res": cl_res,
        "us_res": us_res,
        "hg_res": hg_res,
        "lc_res": lc_res
    }

# We run Tier 1 on example_data.csv and Tier 2 on Total_20260405.xlsx
@pytest.mark.parametrize("file_path,sheet_name", [
    ("example_data.csv", 0),
    ("Total_20260405.xlsx", "negative-All samples")
])
def test_all_plotting_functions_return_figures(file_path, sheet_name):
    # Skip Tier 2 if the large file is missing (to be safe, though we know it is there)
    if not os.path.exists(file_path):
        pytest.skip(f"File {file_path} not found")
        
    data = load_dataset_and_run_analysis(file_path, sheet_name)
    
    # 1. PCA plots
    fig = plot_pca_variance(data["pca_res"][1])
    assert isinstance(fig, Figure)
    
    fig = plot_pca_2d(data["pca_res"][0])
    assert isinstance(fig, Figure)
    
    fig = plot_pca_3d(data["pca_res"][0])
    assert isinstance(fig, Figure)
    
    fig = plot_pca_2d_replicates(data["df_p"], data["exps"])
    assert isinstance(fig, Figure)
    
    # 2. Chain Length plots
    cl = data["cl_res"]
    fig = plot_kde_histogram(cl['long'], 'Acyl Chain Length', 'Mutation', 'Chain Length Distribution', 'Acyl Chain Length')
    assert isinstance(fig, Figure)
    
    fig = plot_zscore_heatmap(cl.get('cohort_z'), 'Chain Length Z-scores')
    assert isinstance(fig, Figure)
    
    fig = plot_correlation_heatmap(cl.get('cohort_raw'), 'Correlation — Chain Lengths')
    assert isinstance(fig, Figure)
    
    fig = plot_heatmap_general(cl.get('cohort_prop'), 'Chain Length Proportions', cmap='YlOrRd')
    assert isinstance(fig, Figure)
    
    # cl_fc_data calculation
    from app.analysis import fold_change
    ctrl = data["exps"]["Mutation"].unique()[0]
    df_fc = fold_change(data["df_meta"], data["df_p"], "Acyl Chain Length", ctrl)
    fig = plot_fold_change_heatmap(df_fc, 'Chain Length log FC')
    assert isinstance(fig, Figure)
    
    fig = plot_cl_gaussian_fit(cl.get('long'))
    assert isinstance(fig, Figure)
    
    fig = plot_odd_chain_bar(odd_chain_fraction(data["df_meta"], data["df_cohort"]))
    assert isinstance(fig, Figure)
    
    fig = plot_odd_chain_kde(data["df_meta"], data["df_p"])
    assert isinstance(fig, Figure)
    
    # 3. Unsaturation plots
    us = data["us_res"]
    fig = plot_kde_histogram(us['long'], 'Unsaturation', 'Mutation', 'Unsaturation Distribution', 'Unsaturation (# db)')
    assert isinstance(fig, Figure)
    
    fig = plot_zscore_heatmap(us.get('cohort_z'), 'Unsaturation Z-scores')
    assert isinstance(fig, Figure)
    
    fig = plot_correlation_heatmap(us.get('cohort_raw'), 'Correlation — Unsaturation Levels')
    assert isinstance(fig, Figure)
    
    fig = plot_heatmap_general(us.get('cohort_prop'), 'Unsaturation Proportions', cmap='YlOrRd')
    assert isinstance(fig, Figure)
    
    # 4. Head Group plots
    hg = data["hg_res"]
    fig = plot_donut_chart(hg["cohort_prop"].mean(axis=1), 'Average Head Group Distribution')
    assert isinstance(fig, Figure)
    
    fig = plot_zscore_heatmap(hg.get('cohort_z'), 'Head Group Z-scores')
    assert isinstance(fig, Figure)
    
    fig = plot_correlation_heatmap(hg.get('cohort_raw'), 'Correlation — Head Groups')
    assert isinstance(fig, Figure)
    
    fig = plot_heatmap_general(hg.get('cohort_prop'), 'Head Group Proportions', cmap='YlOrRd')
    assert isinstance(fig, Figure)
    
    fig = plot_hg_abundance_bar(hg, hg["cohort_raw"].index[0])
    assert isinstance(fig, Figure)
    
    # 5. Lipid Class plots
    lc = data["lc_res"]
    fig = plot_pie_chart(lc["prop"].mean(axis=1), 'Lipid Class Distribution')
    assert isinstance(fig, Figure)
    
    fig = plot_zscore_heatmap(lc.get('zscore'), 'Lipid Class Z-scores')
    assert isinstance(fig, Figure)
    
    fig = plot_heatmap_general(lc.get('prop'), 'Lipid Class Normalised Proportions', cmap='YlOrRd')
    assert isinstance(fig, Figure)
