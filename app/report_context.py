import datetime
import numpy as np
import pandas as pd

def compute_pca_heuristics(df_pca, variance):
    """
    Computes simple PCA separation heuristics based on the pre-computed cohort-level PCA.
    """
    try:
        if df_pca is None or df_pca.empty or len(variance) < 2:
            return ["PCA data unavailable or insufficient components."]
            
        findings = [f"PC1 explains {variance[0]*100:.1f}% and PC2 explains {variance[1]*100:.1f}% of the dataset variance."]
        
        cohorts = df_pca.index.tolist()
        if len(cohorts) > 1 and "PC1" in df_pca.columns and "PC2" in df_pca.columns:
            distances = []
            for i in range(len(cohorts)):
                for j in range(i+1, len(cohorts)):
                    c1, c2 = cohorts[i], cohorts[j]
                    p1 = df_pca.loc[c1, ["PC1", "PC2"]].values
                    p2 = df_pca.loc[c2, ["PC1", "PC2"]].values
                    dist = np.linalg.norm(p1 - p2)
                    distances.append((dist, c1, c2))
            
            distances.sort(reverse=True)
            furthest = distances[0]
            findings.append(f"In PCA space, {furthest[1]} and {furthest[2]} are the most distinct from each other.")
            
            if len(distances) > 1:
                closest = distances[-1]
                findings.append(f"Conversely, {closest[1]} and {closest[2]} are the most similar in their overall lipidomic profiles.")
                
        return findings
    except Exception as e:
        return [f"Could not compute PCA separation: {str(e)}"]

def extract_top_changes(df_cohort, comparison_basis):
    """
    Extracts top changes. If comparison_basis is None, uses the dataset mean.
    Returns sentences for cohort specific outliers.
    """
    if df_cohort.empty:
        return [], []
        
    global_changes = []
    specific_sentences = []
    
    dc = df_cohort.copy()
    cohorts = dc.columns.tolist()
    
    if comparison_basis and comparison_basis in cohorts:
        ref_col = dc[comparison_basis].replace(0, np.nan)
        test_cohorts = [c for c in cohorts if c != comparison_basis]
    else:
        ref_col = dc.mean(axis=1).replace(0, np.nan)
        test_cohorts = cohorts
        
    if not test_cohorts:
        return [], []
        
    log2fc = pd.DataFrame(index=dc.index)
    for c in test_cohorts:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = dc[c] / ref_col
        log2fc[c] = np.log2(ratio.replace(0, np.nan).astype(float))
        
    # Global changes
    if len(test_cohorts) > 0:
        log2fc_clean = log2fc.dropna()
        if not log2fc_clean.empty:
            is_up = (log2fc_clean > 0.5).all(axis=1)
            is_down = (log2fc_clean < -0.5).all(axis=1)
            consistent = log2fc_clean[is_up | is_down].copy()
            consistent["avg_fc"] = consistent.mean(axis=1)
            consistent["abs_avg"] = consistent["avg_fc"].abs()
            top_global = consistent.sort_values("abs_avg", ascending=False).head(10)
            
            for lipid, row in top_global.iterrows():
                global_changes.append({
                    "lipid": str(lipid),
                    "log2FC_avg": round(row["avg_fc"], 2),
                    "trend": "Increased" if row["avg_fc"] > 0 else "Decreased"
                })
                
    # Specific changes: must be uniquely shifted in ONE cohort
    found_any_specific = False
    for c in test_cohorts:
        if len(test_cohorts) > 1:
            other_cols = [col for col in test_cohorts if col != c]
            c_fc = log2fc[c].abs()
            others_max = log2fc[other_cols].abs().max(axis=1)
            diff = c_fc - others_max
            # Hard filter > 0.5 specificity diff
            is_specific = diff > 0.5
            top_spec = diff[is_specific].sort_values(ascending=False).head(3)
        else:
            top_spec = log2fc[c].abs().sort_values(ascending=False).head(3)
            
        specific_lipids = []
        for lipid in top_spec.index:
            val = log2fc.loc[lipid, c]
            if pd.notna(val) and abs(val) > 0.5:
                specific_lipids.append(f"{lipid} (Log2FC: {val:.2f})")
                found_any_specific = True
                
        if specific_lipids:
            specific_sentences.append(f"In cohort {c}, the most uniquely distinct changes were {', '.join(specific_lipids)}.")

    if not found_any_specific:
        specific_sentences.append("No lipids showed changes uniquely specific to one cohort; all major changes were shared across groups.")
                
    return global_changes, specific_sentences


def summarize_chain_length(c_data, odd_chain):
    if not c_data or "cohort_prop" not in c_data:
        return [], "Chain length analysis was not available."
    
    sentences = []
    prop = c_data["cohort_prop"]
    
    # Dominant lengths
    mean_prop = prop.mean(axis=1).sort_values(ascending=False)
    top_lengths = mean_prop.head(3)
    sentences.append(f"The most dominant acyl chain lengths across all cohorts were {', '.join([str(i) for i in top_lengths.index])}.")
    
    # Greatest variance/shifts
    if "cohort_z" in c_data and len(prop.columns) > 1:
        z = c_data["cohort_z"].abs().max(axis=1)
        top_shifts = z.sort_values(ascending=False).head(3)
        sentences.append(f"The chain lengths that exhibited the largest relative shifts between cohorts were {', '.join([str(i) for i in top_shifts.index])}.")

    # Odd chain handler
    if not odd_chain.empty and "FractionOdd" in odd_chain.columns:
        if odd_chain["FractionOdd"].max() == 0.0 or odd_chain["FractionOdd"].isna().all():
            sentences.append("Odd-chain lipids were not detected in this dataset.")
        else:
            sentences.append("Odd-chain lipids were detected, with variable fractions across the tested cohorts.")
            
    return sentences, None


def summarize_unsaturation(u_data):
    if not u_data or "cohort_prop" not in u_data:
        return [], "Unsaturation analysis was not available."
        
    sentences = []
    prop = u_data["cohort_prop"]
    
    mean_prop = prop.mean(axis=1).sort_values(ascending=False)
    top_unsat = mean_prop.head(3)
    sentences.append(f"The most common unsaturation states (double bond counts) were {', '.join([str(i) for i in top_unsat.index])}.")
    
    if "cohort_z" in u_data and len(prop.columns) > 1:
        z = u_data["cohort_z"].abs().max(axis=1)
        top_shifts = z.sort_values(ascending=False).head(3)
        sentences.append(f"The unsaturation states showing the most significant variation between cohorts were {', '.join([str(i) for i in top_shifts.index])}.")
        
    return sentences, None


def summarize_lipid_class(l_data):
    if not l_data or "zscore" not in l_data:
        return [], "Lipid class analysis was not available."
        
    sentences = []
    zscore = l_data["zscore"]
    
    if not zscore.empty and len(zscore.columns) > 1:
        top_shifts = zscore.abs().max(axis=1).sort_values(ascending=False).head(3)
        sentences.append(f"The lipid classes with the most dramatic relative shifts between cohorts were {', '.join([str(i) for i in top_shifts.index])}.")
        
    return sentences, None


def summarize_headgroup(h_data):
    if not h_data or "cohort_z" not in h_data:
        return [], "Head group analysis was not available."
        
    sentences = []
    zscore = h_data["cohort_z"]
    
    if not zscore.empty and len(zscore.columns) > 1:
        top_shifts = zscore.abs().max(axis=1).sort_values(ascending=False).head(3)
        sentences.append(f"The specific head groups showing the largest abundance changes were {', '.join([str(i) for i in top_shifts.index])}.")
        
    return sentences, None


def build_llm_context(df_meta, df_cohort, comparison_cohorts, df_pca=None, variance=None, comparison_basis=None, df_stats=None, df_pw_stats=None, c_data=None, u_data=None, h_data=None, l_data=None, odd_chain=None, available_images=None):
    """
    Builds the structured dictionary payload for the LLM report generation.
    """
    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # 1. Overview
    total_samples = len(df_meta) if not df_meta.empty else 0
    replicates = df_meta["Cohort"].value_counts().to_dict() if "Cohort" in df_meta.columns else {}
    
    has_stats = df_stats is not None and not df_stats.empty and df_pw_stats is not None and not df_pw_stats.empty
    report_level = "statistical" if has_stats else "descriptive"
    
    overview = {
        "generated_at": generated_at,
        "analysis_version": "v1.2.0",
        "report_level": report_level,
        "comparison_cohorts": comparison_cohorts,
        "comparison_basis": comparison_basis,
        "total_samples": total_samples,
        "replicates_per_cohort": replicates
    }
    
    # 2. Limitations
    limitations = []
    if any(count < 3 for count in replicates.values()):
        limitations.append("Statistical power is limited due to having fewer than 3 replicates for one or more cohorts.")
    if not comparison_basis:
        limitations.append("No explicit control baseline was designated; findings represent relative shifts rather than baseline changes.")
    if not has_stats:
        limitations.append("Statistical significance testing (p-values, FDR) was not performed. All changes should be interpreted as descriptive trends only.")
        
    # 3. PCA Findings
    pca_findings = compute_pca_heuristics(df_pca, variance) if df_pca is not None else []
    
    # 4. Top Changes
    global_changes, specific_changes = extract_top_changes(df_cohort, comparison_basis)
    
    # 5. Summarizers
    cl_sentences, cl_lim = summarize_chain_length(c_data, odd_chain if odd_chain is not None else pd.DataFrame())
    if cl_lim: limitations.append(cl_lim)
        
    us_sentences, us_lim = summarize_unsaturation(u_data)
    if us_lim: limitations.append(us_lim)
        
    lc_sentences, lc_lim = summarize_lipid_class(l_data)
    if lc_lim: limitations.append(lc_lim)
        
    hg_sentences, hg_lim = summarize_headgroup(h_data)
    if hg_lim: limitations.append(hg_lim)
    
    # 6. Statistical Confidence
    stat_conf = None
    sig_shifts = []
    if has_stats:
        try:
            p_cols = [c for c in df_pw_stats.columns if "p-value" in c.lower() or "pval" in c.lower()]
            fdr_cols = [c for c in df_pw_stats.columns if "fdr" in c.lower() or "q-val" in c.lower()]
            
            p05_count = 0
            fdr_count = 0
            
            if p_cols:
                mask = (df_pw_stats[p_cols] < 0.05).any(axis=1)
                p05_count = int(mask.sum())
                
                sig_rows = df_pw_stats[mask].head(3)
                for idx, row in sig_rows.iterrows():
                    sig_shifts.append({
                        "feature": str(idx),
                        "p_value": round(row[p_cols[0]], 4) if pd.notna(row[p_cols[0]]) else "Unknown"
                    })
                    
            if fdr_cols:
                mask_fdr = (df_pw_stats[fdr_cols] < 0.05).any(axis=1)
                fdr_count = int(mask_fdr.sum())
                
            stat_conf = {
                "total_significant_findings_p05": p05_count,
                "total_surviving_fdr": fdr_count
            }
        except Exception:
            stat_conf = {"error": "Could not parse statistical confidence metrics"}

    return {
        "experiment_overview": overview,
        "statistical_confidence": stat_conf,
        "pca_clustering_findings": pca_findings,
        "acyl_chain_length_shifts": cl_sentences,
        "unsaturation_shifts": us_sentences,
        "lipid_class_shifts": lc_sentences,
        "headgroup_shifts": hg_sentences,
        "top_consistent_changes_across_cohorts": global_changes,
        "cohort_specific_outliers": specific_changes,
        "significant_class_shifts": sig_shifts,
        "calculated_limitations": limitations,
        "available_images": available_images if available_images else []
    }
