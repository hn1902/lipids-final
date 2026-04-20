"""
analysis.py — Core data processing and analysis functions for the Lipidomics app.
Faithfully reproduces the logic from the reference implementation in
lipidsop/lipids_dataset_analysis/app/{functions.py, analysis.py}.

Data model
----------
Raw CSV format (from the mass-spec instrument export):
  • Row 0 = column headers:  "Sample Name", "NegMSMSALL-CAS9_A", "NegMSMSALL-CAS9_A", ...
  • Rows 1+ = data:           "PC 34:1+HCOO (LPC pe)", 100, 120, ...

After loading:
  df_raw   : rows=lipids, first col="Sample Name", remaining cols=experiment names
  df_meta  : rows=lipids, cols=[Sample Name, Head Group, Head Group 2,
                                 Acyl Chain Length, Unsaturation, Unsaturation 2]
  df_p     : rows=lipids, col "Sample Name" is a regular column,
             remaining cols = experiment names (still per-replicate)
  df_exps  : rows=experiments, cols=[Exp, Mutation, Replicate]
             where Mutation is the cohort label (e.g. "CAS9", "WT")
  df_cohort: rows=lipids, cols=cohort names (averaged over replicates),
             index="Sample Name"
"""

import re
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Colour palette (up to 10 cohorts)
# ---------------------------------------------------------------------------
COHORT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# ---------------------------------------------------------------------------
# Column deduplication (identical to reference functions.py)
# ---------------------------------------------------------------------------

def deduplicate_columns(columns):
    """Append .1, .2 … to duplicate column names so they are all unique."""
    seen = {}
    new_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_deduplicate_data(file_infos, idx_col=None, num_idx=0, skip_rows=0):
    """
    Read one or more uploaded CSV files and concatenate them row-wise.

    Each CSV has:
      • Row 0: headers (can be offset by skip_rows)
      • Rows 1+: lipid rows

    Returns
    -------
    pd.DataFrame  Combined DataFrame with "Sample Name" first column.
    """
    dfs = []
    for fi in file_infos:
        path = fi["datapath"] if isinstance(fi, dict) else fi
        try:
            df = pd.read_csv(path, dtype=str, skiprows=skip_rows)
            
            actual_idx_col = None
            if idx_col and str(idx_col).strip() and str(idx_col).strip() in df.columns:
                actual_idx_col = str(idx_col).strip()
            else:
                actual_idx_col = df.columns[0]
                
            if num_idx > 0:
                cols_to_drop = list(df.columns[:num_idx])
                if actual_idx_col in cols_to_drop:
                    cols_to_drop.remove(actual_idx_col)
                df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
                
            df.set_index(actual_idx_col, inplace=True)
            df.index.name = "Sample Name"
            dfs.append(df.reset_index())
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    # Stack vertically (each file is one lipid class)
    combined = pd.concat(dfs, axis=0, ignore_index=True)

    # Deduplicate column names
    combined.columns = deduplicate_columns(list(combined.columns))

    # Convert numeric columns
    for col in combined.columns[1:]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

    return combined


# ---------------------------------------------------------------------------
# Experiment metadata extraction
# ---------------------------------------------------------------------------

def extract_experiments(df_raw, header_df=None, cohort_row_idx=1):
    """
    Parse the experiment column names of the raw DataFrame into a table.

    If a custom `header_df` is provided (from an uploaded header CSV), it maps
    the experiment columns to the cohorts defined in `header_df` at `cohort_row_idx` (0-indexed).
    Assuming header_df has sample names in row 0, and metadata in subsequent rows.
    
    If no header is provided, parses expected column format: ``NegMSMSALL-{Mutation}_{Replicate}``
    e.g.  ``NegMSMSALL-CAS9_A``  →  Mutation="CAS9", Replicate="A"

    Returns
    -------
    df_exps : pd.DataFrame  cols = ['Exp', 'Mutation', 'Replicate']
    """
    exp_cols = [c for c in df_raw.columns if c != "Sample Name"]
    rows = []
    
    if header_df is not None and not header_df.empty:
        # Transpose so index = sample names (from row 0). 
        # But if header_df already has it as columns, we handle it as rows = metadata levels
        try:
            # We assume header_df: Row 0 = Sample names, Row N = Cohorts
            sample_names_row = header_df.iloc[0].astype(str).tolist()
            cohorts_row = header_df.iloc[cohort_row_idx].astype(str).tolist()
            
            # Create mapping dictionary stripping trailing suffix from deduplication in raw if needed
            hmap = {}
            for sname, coh in zip(sample_names_row, cohorts_row):
                if pd.isna(sname): continue
                hmap[str(sname).strip()] = str(coh).strip()
                
            for col in exp_cols:
                base = re.sub(r"\.\d+$", "", col)
                # Try exact match first, then base match
                if col in hmap:
                    mut = hmap[col]
                elif base in hmap:
                    mut = hmap[base]
                else:
                    # Skip columns not found in the custom header file (e.g. metadata cols)
                    continue
                rows.append({"Exp": col, "Mutation": mut, "Replicate": "1"})
            return pd.DataFrame(rows)
        except Exception:
            pass # Fall back to regex if parsing fails

    # Fallback / Regex Parsing
    for col in exp_cols:
        # Strip deduplication suffix (.1, .2 ...) for display
        base = re.sub(r"\.\d+$", "", col)
        # Try to parse NegMSMSALL-{Mutation}_{Replicate} pattern
        m = re.match(r".*?-(.+?)_([A-Za-z])$", base)
        if m:
            mutation  = m.group(1)
            replicate = m.group(2)
        else:
            # Fallback: use whole base name as mutation
            mutation  = base
            replicate = "1"
        rows.append({"Exp": col, "Mutation": mutation, "Replicate": replicate})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# df_p: rename experiments to mutations (matching reference functions.df_p)
# ---------------------------------------------------------------------------

def make_df_p(df_raw, df_exps):
    """
    Rename experiment columns → mutation names.  Keeps "Sample Name" as a
    regular column (not the index) to match the reference merge pattern.

    Returns
    -------
    pd.DataFrame  columns = ['Sample Name', Mutation1, Mutation2, ...]
    """
    rename_map = dict(zip(df_exps["Exp"], df_exps["Mutation"]))
    df = df_raw.rename(columns=rename_map)
    # Keep only Sample Name + mutation columns
    keep = ["Sample Name"] + df_exps["Mutation"].tolist()
    keep = [c for c in keep if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Cohort aggregation  (average over replicates within each mutation)
# ---------------------------------------------------------------------------

def aggregate_by_cohort(df_p, df_exps, method="mean"):
    """
    Average (or median/sum) per-experiment data within each mutation group.

    Parameters
    ----------
    df_p    : DataFrame with 'Sample Name' col + one col per experiment
              (columns may be repeated mutation names = technical replicates)
    df_exps : DataFrame from extract_experiments()
    method  : 'mean' | 'median' | 'sum'

    Returns
    -------
    df_cohort : DataFrame  index='Sample Name', cols=unique mutations
    """
    if df_p.empty:
        return pd.DataFrame()

    # Build mapping of mutation → list of column indices
    mutations = df_exps["Mutation"].unique().tolist()
    agg = {}
    for mtn in mutations:
        # All columns in df_p that belong to this mutation
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

    df_cohort = pd.DataFrame(agg)
    df_cohort.index = df_p["Sample Name"].values
    df_cohort.index.name = "Sample Name"
    return df_cohort


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
# Lipid name examples from real data:
#   "PC 34:1+HCOO (LPC pe)"
#   "LPC 20:4+HCOO (LPC pe)"
#   "PC O-34:1+HCOO (FA 16:0)"
#   "Cer d18:1/C24:0"
#   "TG 50:3"
#
# Head Group = everything before the first space
# Chain info  = the token immediately after the first space (e.g. "34:1+HCOO")
#               → take the part before "+" if present
# Chain Length = integer before ":"
# Unsaturation = integer after ":"

def extract_metadata(df_raw):
    """
    Parse the "Sample Name" column of the raw DataFrame into metadata.

    Returns
    -------
    pd.DataFrame  cols = ['Sample Name', 'Head Group', 'Head Group 2',
                          'Acyl Chain Length', 'Unsaturation', 'Unsaturation 2']
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    names = df_raw["Sample Name"].tolist()
    records = []

    for name in names:
        name = str(name).strip()
        parts = name.split()
        hg = parts[0] if parts else "Unknown"

        # Chain spec is the second token (may be "34:1+HCOO", "d18:1/C24:0", etc.)
        chain_spec = parts[1] if len(parts) > 1 else "0:0"
        # Remove adduct suffix (+HCOO, +NH4, etc.)
        chain_spec = chain_spec.split("+")[0]
        # Handle plasmalogen prefix like "O-34:1"
        chain_spec = re.sub(r"^[OP]-", "", chain_spec)
        # Handle Cer-style "d18:1/C24:0" — use first number pair
        chain_spec = chain_spec.split("/")[0]
        # Remove leading "d" or "t"
        chain_spec = re.sub(r"^[a-zA-Z]", "", chain_spec)

        match = re.match(r"(\d+):(\d+)", chain_spec)
        cl    = int(match.group(1)) if match else 0
        unsat = int(match.group(2)) if match else 0

        # Head Group 2 — normalised class
        hg2 = hg
        hg2 = re.sub(r"\s+[OP]$", "", hg2).strip()   # remove O/P plasmalogen notation
        hg2 = re.sub(r"\d+$", "", hg2).strip()         # remove trailing digits (GD1→GD)
        hg2 = hg2.replace("HexCer", "Hex_Cer")

        unsat2 = str(unsat) if unsat < 3 else ">=3"

        records.append({
            "Sample Name":    name,
            "Head Group":     hg,
            "Head Group 2":   hg2,
            "Acyl Chain Length": cl,
            "Unsaturation":   unsat,
            "Unsaturation 2": unsat2,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_data(df_raw, df_meta,
                filter_hg=None,
                min_chain=0,
                max_unsat=99,
                remove_blank=False,
                blank_threshold=0.05,
                blank_keywords=None):
    """
    Filter the raw DataFrame based on metadata criteria.

    Parameters
    ----------
    blank_keywords : list[str] | None
        If provided, any sample column whose name contains any keyword
        (case-insensitive) is dropped before analysis (e.g. ["RAJU", "blank"]).

    Returns
    -------
    (df_raw_filt, df_meta_filt) — both aligned on 'Sample Name'.
    """
    if df_raw.empty or df_meta.empty:
        return df_raw, df_meta

    dm = df_meta.copy()

    # Head Group filter
    if filter_hg:
        dm = dm[dm["Head Group 2"].isin(filter_hg)]

    # Chain length filter
    dm = dm[dm["Acyl Chain Length"] >= min_chain]

    # Unsaturation filter
    dm = dm[dm["Unsaturation"] <= max_unsat]

    # Apply filter to raw df
    keep = set(dm["Sample Name"])
    dr = df_raw[df_raw["Sample Name"].isin(keep)].copy()
    dm = dm[dm["Sample Name"].isin(set(dr["Sample Name"]))].copy()

    # Remove blank columns by name keyword
    if blank_keywords:
        kws = [k.strip().lower() for k in blank_keywords if k.strip()]
        if kws:
            num_cols = [c for c in dr.columns if c != "Sample Name"]
            drop_cols = [c for c in num_cols
                         if any(kw in c.lower() for kw in kws)]
            dr = dr.drop(columns=drop_cols, errors="ignore")

    # Remove blank columns (samples) by signal threshold
    if remove_blank and not dr.empty:
        num_cols = [c for c in dr.columns if c != "Sample Name"]
        col_sums = dr[num_cols].sum(axis=0)
        mean_sum = col_sums.mean()
        blank_threshold_val = blank_threshold * mean_sum
        keep_cols = ["Sample Name"] + col_sums[col_sums >= blank_threshold_val].index.tolist()
        dr = dr[[c for c in keep_cols if c in dr.columns]]

    return dr, dm


# ---------------------------------------------------------------------------
# Normalisation helpers (matching reference functions.py)
# ---------------------------------------------------------------------------

def norm_col(df):
    """Normalise each column by its sum (column-wise proportions)."""
    return df / df.sum()


def norm_row(df):
    """Normalise each row by its sum (row-wise proportions)."""
    return df.div(df.sum(axis=1), axis=0)


# ---------------------------------------------------------------------------
# groupby_norm (mirrors reference functions.groupby_norm)
# ---------------------------------------------------------------------------

def groupby_norm(df_meta, df_p, var, drop_var=None, drop_mutation=None,
                 norm_exp=True, norm_var=True):
    """
    Group df_p by a metadata variable and normalise.

    Uses the same merge-on-'Sample Name' pattern as the reference.

    Returns
    -------
    pd.DataFrame  rows=var levels, cols=mutations
    """
    drop_var = drop_var or []
    drop_mutation = drop_mutation or []

    df = df_meta[["Sample Name", var]].merge(df_p, on="Sample Name").set_index("Sample Name")
    df = df.groupby(var).sum(numeric_only=True)

    if norm_exp:
        df = norm_col(df)

    if drop_var:
        df = df[~df.index.isin(drop_var)]
    if drop_mutation:
        df = df.drop(columns=[c for c in drop_mutation if c in df.columns])

    df.columns.names = ["Mutation"]

    # Average over replicates: T.groupby('Mutation').mean().T
    df = df.T.groupby("Mutation").mean().T

    if norm_var:
        df = norm_row(df)

    return df


# ---------------------------------------------------------------------------
# fold_change and z_score (mirrors reference functions.py)
# ---------------------------------------------------------------------------

def fold_change(df_meta, df_p, var, control_mtn, drop_var=None, drop_mutation=None,
                norm_exp=True):
    """
    Compute log(fold change) vs the control mutation.

    Returns
    -------
    df_logfc : pd.DataFrame
    """
    drop_var = drop_var or []
    drop_mutation = drop_mutation or []

    df = df_meta[["Sample Name", var]].merge(df_p, on="Sample Name").set_index("Sample Name")
    df = df.groupby(var).sum(numeric_only=True)

    if norm_exp:
        df = norm_col(df)
    if drop_var:
        df = df[~df.index.isin(drop_var)]
    if drop_mutation:
        df = df.drop(columns=[c for c in drop_mutation if c in df.columns])

    df.columns.names = ["Mutation"]
    df = df.T.groupby("Mutation").mean().T

    if control_mtn not in df.columns:
        return pd.DataFrame()

    df_fc = df.div(df[control_mtn], axis=0)
    df_fc.fillna(1, inplace=True)
    non_inf_max = df_fc[df_fc != np.inf].max().max()
    df_fc.replace(np.inf, non_inf_max ** 10, inplace=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        df_log = np.log(df_fc)

    non_inf_min = df_log[df_log != -np.inf].min().min()
    df_log.replace(-np.inf, non_inf_min * 10, inplace=True)
    df_log.replace(np.inf, np.log(non_inf_max) * 10, inplace=True)

    return df_log


def z_score(df_meta, df_p, var, control_mtn, drop_var=None, drop_mutation=None,
            norm_exp=True):
    """
    Compute z-score vs control mutation using pooled SEM.

    Returns
    -------
    dfz : pd.DataFrame
    """
    drop_var = drop_var or []
    drop_mutation = drop_mutation or []

    df = df_meta[["Sample Name", var]].merge(df_p, on="Sample Name").set_index("Sample Name")
    df = df.groupby(var).sum(numeric_only=True)

    if norm_exp:
        df = norm_col(df)
    if drop_var:
        df = df[~df.index.isin(drop_var)]
    if drop_mutation:
        df = df.drop(columns=[c for c in drop_mutation if c in df.columns])

    df.columns.names = ["Mutation"]

    dfm  = df.T.groupby("Mutation").mean().T
    dfse = df.T.groupby("Mutation").sem().T

    if control_mtn not in dfm.columns:
        return pd.DataFrame()

    dfm_diff = dfm.sub(dfm[control_mtn], axis=0)
    dfse2    = dfse ** 2
    dfse_pooled = np.sqrt(dfse2.add(dfse2[control_mtn], axis=0))
    dfz = dfm_diff / dfse_pooled

    dfz.fillna(0, inplace=True)
    dfz.replace([np.inf, -np.inf], 0, inplace=True)
    return dfz


# ---------------------------------------------------------------------------
# Chain Length analysis (mirrors reference analysis.chain_length_group_tables)
# ---------------------------------------------------------------------------

def chain_length_analysis(df_meta, df_p, df_cohort):
    """
    Returns dict with keys:
      'cohort_raw'  : chain len × cohort sums
      'cohort_prop' : column-normalised proportions
      'cohort_z'    : z-score across cohorts (row-wise)
      'long'        : long-form for KDE/histogram
    """
    if df_meta.empty or df_p.empty or df_cohort.empty:
        return {}

    dc = df_cohort.reset_index()  # Sample Name becomes column
    merged = df_meta.merge(dc, on="Sample Name")
    cohort_cols = [c for c in df_cohort.columns]

    cl_raw  = merged.groupby("Acyl Chain Length")[cohort_cols].sum()
    cl_prop = cl_raw.div(cl_raw.sum(axis=0), axis=1)
    cl_z    = cl_raw.apply(lambda x: (x - x.mean()) / (x.std() + 1e-12), axis=1)

    # Long form using df_p (per-experiment, for individual sample KDE)
    dp = df_p.copy()
    merged2 = df_meta.merge(dp, on="Sample Name")
    exp_cols = [c for c in dp.columns if c != "Sample Name"]
    long_df = merged2.melt(
        id_vars=["Sample Name", "Acyl Chain Length",
                 "Head Group", "Head Group 2", "Unsaturation", "Unsaturation 2"],
        value_vars=exp_cols,
        var_name="Mutation", value_name="Abundance",
    )

    return {"cohort_raw": cl_raw, "cohort_prop": cl_prop,
            "cohort_z": cl_z, "long": long_df}


# ---------------------------------------------------------------------------
# Unsaturation analysis (mirrors reference analysis.unsaturation_group_tables)
# ---------------------------------------------------------------------------

def unsaturation_analysis(df_meta, df_p, df_cohort):
    if df_meta.empty or df_p.empty or df_cohort.empty:
        return {}

    dc = df_cohort.reset_index()
    merged = df_meta.merge(dc, on="Sample Name")
    cohort_cols = list(df_cohort.columns)

    unsat_raw  = merged.groupby("Unsaturation")[cohort_cols].sum()
    unsat_prop = unsat_raw.div(unsat_raw.sum(axis=0), axis=1)
    unsat_z    = unsat_raw.apply(lambda x: (x - x.mean()) / (x.std() + 1e-12), axis=1)

    dp = df_p.copy()
    merged2 = df_meta.merge(dp, on="Sample Name")
    exp_cols = [c for c in dp.columns if c != "Sample Name"]
    long_df = merged2.melt(
        id_vars=["Sample Name", "Unsaturation",
                 "Head Group", "Head Group 2", "Acyl Chain Length"],
        value_vars=exp_cols,
        var_name="Mutation", value_name="Abundance",
    )

    return {"cohort_raw": unsat_raw, "cohort_prop": unsat_prop,
            "cohort_z": unsat_z, "long": long_df}


# ---------------------------------------------------------------------------
# Head Group analysis
# ---------------------------------------------------------------------------

def headgroup_analysis(df_meta, df_p, df_cohort):
    if df_meta.empty or df_p.empty or df_cohort.empty:
        return {}

    dc = df_cohort.reset_index()
    merged = df_meta.merge(dc, on="Sample Name")
    cohort_cols = list(df_cohort.columns)

    hg_raw  = merged.groupby("Head Group 2")[cohort_cols].sum()
    hg_prop = hg_raw.div(hg_raw.sum(axis=0), axis=1)
    hg_z    = hg_raw.apply(lambda x: (x - x.mean()) / (x.std() + 1e-12), axis=1)

    long_rows = []
    dp = df_p.copy()
    merged2 = df_meta.merge(dp, on="Sample Name")
    exp_cols = [c for c in dp.columns if c != "Sample Name"]
    long_df = merged2.melt(
        id_vars=["Sample Name", "Head Group 2", "Acyl Chain Length", "Unsaturation"],
        value_vars=exp_cols,
        var_name="Mutation", value_name="Abundance",
    )

    return {"cohort_raw": hg_raw, "cohort_prop": hg_prop,
            "cohort_z": hg_z, "long": long_df}


# ---------------------------------------------------------------------------
# Lipid class analysis (uses 'Head Group' — un-normalised)
# ---------------------------------------------------------------------------

def lipid_class_analysis(df_meta, df_cohort):
    if df_meta.empty or df_cohort.empty:
        return {}

    dc = df_cohort.reset_index()
    merged = df_meta.merge(dc, on="Sample Name")
    cohort_cols = list(df_cohort.columns)

    lc_raw  = merged.groupby("Head Group")[cohort_cols].sum()
    lc_prop = lc_raw.div(lc_raw.sum(axis=0), axis=1)
    lc_z    = lc_raw.apply(lambda x: (x - x.mean()) / (x.std() + 1e-12), axis=1)

    return {"raw": lc_raw, "prop": lc_prop, "zscore": lc_z}


# ---------------------------------------------------------------------------
# Odd-chain fraction (mirrors reference analysis.odd_chain_fraction)
# ---------------------------------------------------------------------------

def odd_chain_fraction(df_meta, df_cohort):
    """
    Returns pd.DataFrame indexed by Cohort with cols [Odd, Even, FractionOdd].
    """
    if df_meta.empty or df_cohort.empty:
        return pd.DataFrame()

    dc = df_cohort.reset_index()
    merged = df_meta.merge(dc, on="Sample Name")
    cohort_cols = list(df_cohort.columns)

    long_df = merged.melt(
        id_vars=["Sample Name", "Acyl Chain Length"],
        value_vars=cohort_cols,
        var_name="Cohort", value_name="Abundance",
    )
    long_df["Odd"] = long_df["Acyl Chain Length"] % 2 == 1
    s = long_df.groupby(["Cohort", "Odd"])["Abundance"].sum().unstack(fill_value=0)
    s.rename(columns={True: "Odd", False: "Even"}, inplace=True)
    if "Odd"  not in s.columns: s["Odd"]  = 0
    if "Even" not in s.columns: s["Even"] = 0
    s["FractionOdd"] = s["Odd"] / (s["Odd"] + s["Even"]).replace(0, np.nan)
    s.fillna(0, inplace=True)
    return s


# ---------------------------------------------------------------------------
# Subset head group by chain length (mirrors analysis.subset_headgroup_by_chain)
# ---------------------------------------------------------------------------

def subset_headgroup_by_chain(df_meta, df_cohort, condition):
    """
    condition: lambda applied to df_meta['Acyl Chain Length'],  e.g. lambda x: x >= 50
    Returns normalised head group proportions (Head Group 2 × cohorts).
    """
    if df_meta.empty or df_cohort.empty:
        return pd.DataFrame()

    mask = condition(df_meta["Acyl Chain Length"])
    keep = df_meta.loc[mask, "Sample Name"].values

    dc_filt = df_cohort[df_cohort.index.isin(keep)]
    if dc_filt.empty:
        return pd.DataFrame()

    dc_reset = dc_filt.reset_index()
    dm_local = df_meta[df_meta["Sample Name"].isin(keep)]
    merged   = dm_local.merge(dc_reset, on="Sample Name")
    cohort_cols = list(df_cohort.columns)

    hg = merged.groupby("Head Group 2")[cohort_cols].sum()
    hg_norm = hg.div(hg.sum(axis=0), axis=1)
    return hg_norm


# ---------------------------------------------------------------------------
# Subset head group by unsaturation (mirrors analysis.subset_headgroup_by_unsat)
# ---------------------------------------------------------------------------

def subset_headgroup_by_unsat(df_meta, df_cohort, condition):
    """
    condition: lambda applied to df_meta['Unsaturation'], e.g. lambda x: x == 0
    Returns normalised head group proportions.
    """
    if df_meta.empty or df_cohort.empty:
        return pd.DataFrame()

    mask = condition(df_meta["Unsaturation"])
    keep = df_meta.loc[mask, "Sample Name"].values

    dc_filt = df_cohort[df_cohort.index.isin(keep)]
    if dc_filt.empty:
        return pd.DataFrame()

    dc_reset = dc_filt.reset_index()
    dm_local = df_meta[df_meta["Sample Name"].isin(keep)]
    merged   = dm_local.merge(dc_reset, on="Sample Name")
    cohort_cols = list(df_cohort.columns)

    hg = merged.groupby("Head Group 2")[cohort_cols].sum()
    hg_norm = hg.div(hg.sum(axis=0), axis=1)
    return hg_norm


# ---------------------------------------------------------------------------
# PCA (on the cohort-level data)
# ---------------------------------------------------------------------------

def perform_pca(df_cohort):
    """
    PCA on cohort-aggregated data.

    df_cohort : index=Sample Name, cols=mutations

    Returns
    -------
    df_pca   : DataFrame  index=mutations, cols=PC1/PC2/PC3
    variance : array
    loadings : DataFrame
    """
    if df_cohort is None or df_cohort.empty:
        return pd.DataFrame(), np.array([]), pd.DataFrame()

    X      = df_cohort.T.values
    labels = df_cohort.columns.tolist()

    n_components = min(3, X.shape[0], X.shape[1])
    if n_components < 1:
        return pd.DataFrame(), np.array([]), pd.DataFrame()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca    = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    cols   = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(scores, index=labels, columns=cols)
    df_pca.index.name = "Mutation"

    loadings = pd.DataFrame(
        pca.components_.T,
        index=df_cohort.index,
        columns=cols,
    )
    return df_pca, pca.explained_variance_ratio_, loadings


# ---------------------------------------------------------------------------
# Holm–Sidak correction (identical to reference analysis.holm_sidak_correction)
# ---------------------------------------------------------------------------

def holm_sidak_correction(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    m     = len(pvals)
    if m == 0:
        return {"reject": np.array([], dtype=bool),
                "p_adjusted": np.array([]), "thresholds": []}

    order        = np.argsort(pvals)
    p_sorted     = pvals[order]
    reject_sorted = np.zeros(m, dtype=bool)
    thresholds    = np.zeros(m)

    for i in range(m):
        thresh = 1 - (1 - alpha) ** (1 / (m - i))
        thresholds[i] = thresh
        if p_sorted[i] <= thresh:
            reject_sorted[i] = True
        else:
            break  # step-down: once fails, rest not rejected

    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted

    p_adj_sorted = np.empty(m)
    for i in range(m):
        p_adj_sorted[i] = min(1.0, p_sorted[i] * (m - i))
    p_adjusted = np.empty(m)
    p_adjusted[order] = p_adj_sorted

    return {"reject": reject, "p_adjusted": p_adjusted, "thresholds": thresholds}


# ---------------------------------------------------------------------------
# Statistical analysis (mirrors reference analysis.anova_per_level)
# ---------------------------------------------------------------------------

def statistical_analysis(df_meta, df_p, df_cohort, var, alpha=0.05):
    """
    One-way ANOVA + Holm-Sidak post-hoc pairwise t-tests.

    Uses the cohort-level data for ANOVA (one value per cohort per level).
    The model is: Abundance ~ C(Cohort) fitted with statsmodels OLS.

    Parameters
    ----------
    df_meta   : metadata DataFrame
    df_p      : per-experiment DataFrame (with 'Sample Name' column)
    df_cohort : cohort-aggregated DataFrame (index=Sample Name, cols=cohorts)
    var       : grouping variable ('Head Group 2', 'Acyl Chain Length', etc.)
    alpha     : significance level

    Returns
    -------
    anova_df   : DataFrame indexed by var-level
    posthoc_df : long-form pairwise comparisons DataFrame
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    if df_meta.empty or df_cohort.empty:
        return pd.DataFrame(), pd.DataFrame()

    dc = df_cohort.reset_index()
    merged = df_meta.merge(dc, on="Sample Name")
    cohort_cols = list(df_cohort.columns)
    df_long = merged.melt(
        id_vars=["Sample Name", var],
        value_vars=cohort_cols,
        var_name="Cohort",
        value_name="Abundance",
    )

    levels     = df_long[var].dropna().unique()
    anova_rows = []
    posthoc_rows = []

    for lvl in levels:
        sub = df_long[df_long[var] == lvl]
        groups = sub.groupby("Cohort")["Abundance"].apply(list)
        if groups.shape[0] < 2:
            anova_rows.append({"level": lvl, "F": np.nan, "PR(>F)": np.nan,
                               "n_groups": groups.shape[0]})
            continue

        try:
            model     = smf.ols("Abundance ~ C(Cohort)", data=sub).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)
            F    = aov_table.loc["C(Cohort)", "F"]     if "C(Cohort)" in aov_table.index else np.nan
            pval = aov_table.loc["C(Cohort)", "PR(>F)"] if "C(Cohort)" in aov_table.index else np.nan
        except Exception:
            try:
                grp_lists = [g for _, g in sub.groupby("Cohort")["Abundance"]]
                F, pval = stats.f_oneway(*grp_lists)
            except Exception:
                F, pval = np.nan, np.nan

        anova_rows.append({"level": lvl, "F": F, "PR(>F)": pval,
                           "n_groups": groups.shape[0]})

        if np.isnan(pval) or pval > alpha:
            continue

        cohort_names = groups.index.tolist()
        pvals, pairs, tstats = [], [], []
        for i in range(len(cohort_names)):
            for j in range(i + 1, len(cohort_names)):
                g1, g2 = groups[cohort_names[i]], groups[cohort_names[j]]
                try:
                    t, p = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
                except Exception:
                    t, p = np.nan, np.nan
                pvals.append(p)
                pairs.append((cohort_names[i], cohort_names[j]))
                tstats.append(t)

        if pvals:
            res        = holm_sidak_correction(pvals, alpha=alpha)
            p_adjusted = res["p_adjusted"]
            reject     = res["reject"]
            for k, (c1, c2) in enumerate(pairs):
                posthoc_rows.append({
                    var: lvl,
                    "Group1": c1, "Group2": c2,
                    "t-stat": tstats[k],
                    "pval":   pvals[k],
                    "p_adj":  p_adjusted[k],
                    "reject": bool(reject[k]),
                })

    anova_df   = pd.DataFrame(anova_rows)
    posthoc_df = pd.DataFrame(posthoc_rows)

    if not anova_df.empty and "PR(>F)" in anova_df.columns:
        anova_df = anova_df.sort_values("PR(>F)")
    if not posthoc_df.empty and "p_adj" in posthoc_df.columns:
        posthoc_df = posthoc_df.sort_values("p_adj")

    return anova_df, posthoc_df


# ---------------------------------------------------------------------------
# Confidence ellipse (matches reference functions.confidence_ellipse)
# ---------------------------------------------------------------------------

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    if len(x) < 2:
        return
    x, y = np.asarray(x), np.asarray(y)
    cov     = np.cov(x, y)
    pearson = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]) + 1e-12)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


# ---------------------------------------------------------------------------
# =-=-=-=-=-= PLOTTING HELPERS =-=-=-=-=-=
# ---------------------------------------------------------------------------

def plot_pca_variance(variance):
    fig, ax = plt.subplots(figsize=(6, 4))
    if variance is None or len(variance) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    pcs   = [f"PC{i+1}" for i in range(len(variance))]
    cumv  = np.cumsum(variance)
    bars  = ax.bar(pcs, variance * 100, color=COHORT_COLORS[:len(pcs)], edgecolor="white")
    ax.plot(pcs, cumv * 100, "k--o", ms=6, label="Cumulative")
    for bar, v in zip(bars, variance):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA — Explained Variance")
    ax.legend(fontsize=9); ax.set_ylim(0, 115)
    fig.tight_layout(); return fig


def plot_pca_2d(df_pca):
    fig, ax = plt.subplots(figsize=(7, 5))
    if df_pca is None or df_pca.empty or "PC1" not in df_pca.columns:
        ax.text(0.5, 0.5, "No PCA data", ha="center", va="center"); return fig
    cohorts = df_pca.index.tolist()
    colors  = {c: COHORT_COLORS[i % len(COHORT_COLORS)] for i, c in enumerate(cohorts)}
    for coh in cohorts:
        row = df_pca.loc[coh]
        x, y = row["PC1"], row.get("PC2", 0)
        ax.scatter(x, y, color=colors[coh], s=120, zorder=3,
                   label=coh, edgecolors="white", linewidths=0.8)
        ax.annotate(coh, (x, y), fontsize=7, ha="left",
                    xytext=(5, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA — 2D Scores")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(); return fig


def plot_pca_3d(df_pca):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    if df_pca is None or df_pca.empty or "PC1" not in df_pca.columns:
        ax.text(0, 0, 0, "No PCA data"); return fig
    cohorts = df_pca.index.tolist()
    colors  = {c: COHORT_COLORS[i % len(COHORT_COLORS)] for i, c in enumerate(cohorts)}
    for coh in cohorts:
        row = df_pca.loc[coh]
        ax.scatter(row.get("PC1", 0), row.get("PC2", 0), row.get("PC3", 0),
                   color=colors[coh], s=80, label=coh)
        ax.text(row.get("PC1", 0), row.get("PC2", 0), row.get("PC3", 0),
                f"  {coh}", fontsize=7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("PCA — 3D Scores")
    ax.legend(fontsize=7); fig.tight_layout(); return fig


def plot_kde_histogram(long_df, x_col, cohort_col, title, xlabel):
    """KDE + histogram weighted by abundance per cohort (mirrors reference kde_hist_plot)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if long_df is None or long_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return fig

    cohorts = long_df[cohort_col].unique()
    pal = sns.color_palette("tab10", n_colors=len(cohorts))

    for i, cohort in enumerate(cohorts):
        sub = long_df[long_df[cohort_col] == cohort]
        if sub["Abundance"].sum() == 0:
            continue
        try:
            xvals = sub[x_col].astype(float)
            w     = sub["Abundance"].values
            sns.kdeplot(x=xvals, weights=w, label=cohort, ax=ax, color=pal[i])
            sns.histplot(x=xvals, weights=w, stat="density",
                         alpha=0.2, color=pal[i], ax=ax,
                         bins=range(int(xvals.min()), int(xvals.max()) + 2))
        except Exception:
            ax.scatter(sub[x_col], sub["Abundance"],
                       label=cohort, color=pal[i], alpha=0.6, s=20)

    ax.set_xlabel(xlabel); ax.set_ylabel("Density (weighted)")
    ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout(); return fig


def plot_zscore_heatmap(df, title, cmap="coolwarm"):
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return fig
    g = sns.clustermap(
        df.fillna(0),
        cmap=cmap, center=0,
        figsize=(max(6, df.shape[1] * 0.8 + 2),
                 max(4, df.shape[0] * 0.4 + 2)),
        linewidths=0.3, dendrogram_ratio=(0.1, 0.15),
        cbar_kws={"shrink": 0.6},
    )
    g.fig.suptitle(title, y=1.01, fontsize=11); return g.fig


def plot_correlation_heatmap(df, title):
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return fig
    corr = df.T.corr()
    fig, ax = plt.subplots(figsize=(max(5, len(corr) * 0.5 + 2),
                                   max(4, len(corr) * 0.5 + 1.5)))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", vmin=-1, vmax=1,
                annot=len(corr) <= 15, fmt=".2f", linewidths=0.3, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title); fig.tight_layout(); return fig


def plot_fold_change_heatmap(df_log, title, cmap="RdBu_r"):
    if df_log is None or df_log.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return fig
    data = df_log.fillna(0)
    vabs = max(abs(data.values.min()), abs(data.values.max()), 0.1)
    fig, ax = plt.subplots(figsize=(max(5, data.shape[1] * 0.8 + 2.5),
                                    max(4, data.shape[0] * 0.4 + 2)))
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    sns.heatmap(data, ax=ax, cmap=cmap, norm=norm, linewidths=0.3,
                cbar_kws={"shrink": 0.8, "label": "log FC"})
    ax.set_title(title); fig.tight_layout(); return fig


def plot_heatmap_general(df, title, cmap="YlOrRd", vmin=None, vmax=None):
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return fig
    fig, ax = plt.subplots(figsize=(max(5, df.shape[1] * 0.75 + 2.5),
                                    max(4, df.shape[0] * 0.4 + 2)))
    sns.heatmap(df.fillna(0), ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                linewidths=0.3, cbar_kws={"shrink": 0.8})
    ax.set_title(title); fig.tight_layout(); return fig


def plot_donut_chart(series, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    if series is None or series.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return fig
    vals   = series.values
    labels = series.index.tolist()
    colors = [COHORT_COLORS[i % len(COHORT_COLORS)] for i in range(len(labels))]
    ax.pie(vals, labels=labels, colors=colors,
           autopct="%1.1f%%", pctdistance=0.82,
           wedgeprops=dict(width=0.5, edgecolor="white"), startangle=90)
    ax.set_title(title); fig.tight_layout(); return fig


def plot_pie_chart(series, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    if series is None or series.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return fig
    vals   = series.values
    labels = series.index.tolist()
    colors = [COHORT_COLORS[i % len(COHORT_COLORS)] for i in range(len(labels))]
    ax.pie(vals, labels=labels, colors=colors,
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white"))
    ax.set_title(title); fig.tight_layout(); return fig


def plot_odd_chain_bar(odd_frac_df, title="Odd-Chain Lipid Fraction by Cohort"):
    fig, ax = plt.subplots(figsize=(7, 4))
    if odd_frac_df is None or odd_frac_df.empty or "FractionOdd" not in odd_frac_df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return fig
    cohorts = odd_frac_df.index.tolist()
    values  = odd_frac_df["FractionOdd"].values
    colors  = [COHORT_COLORS[i % len(COHORT_COLORS)] for i in range(len(cohorts))]
    bars = ax.bar(cohorts, values * 100, color=colors, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Odd-Chain Fraction (%)")
    ax.set_title(title)
    ax.set_xticklabels(cohorts, rotation=30, ha="right")
    ax.set_ylim(0, max(values * 100) * 1.25 + 2 if values.max() > 0 else 10)
    fig.tight_layout(); return fig


# ===========================================================================
# NEW FEATURES — Phase 1–4 additions (all additive, no existing code changed)
# ===========================================================================


# ---------------------------------------------------------------------------
# XLS / XLSX aware loader
# ---------------------------------------------------------------------------

def load_file_any_format(path, skip_rows=0):
    """Read CSV, XLS or XLSX into a raw DataFrame (dtype=str)."""
    ext = str(path).lower().split(".")[-1]
    if ext in ("xls", "xlsx"):
        return pd.read_excel(path, dtype=str, skiprows=skip_rows, engine="openpyxl")
    return pd.read_csv(path, dtype=str, skiprows=skip_rows)


def load_and_deduplicate_data_v2(file_infos, idx_col=None, num_idx=0, skip_rows=0):
    """
    Like load_and_deduplicate_data but also handles XLS/XLSX.
    Drop-in replacement when XLSX files are uploaded.
    """
    dfs = []
    for fi in file_infos:
        path = fi["datapath"] if isinstance(fi, dict) else fi
        try:
            df = load_file_any_format(path, skip_rows=skip_rows)

            actual_idx_col = None
            if idx_col and str(idx_col).strip() and str(idx_col).strip() in df.columns:
                actual_idx_col = str(idx_col).strip()
            else:
                actual_idx_col = df.columns[0]

            if num_idx > 0:
                cols_to_drop = list(df.columns[:num_idx])
                if actual_idx_col in cols_to_drop:
                    cols_to_drop.remove(actual_idx_col)
                df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

            df.set_index(actual_idx_col, inplace=True)
            df.index.name = "Sample Name"
            dfs.append(df.reset_index())
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    combined.columns = deduplicate_columns(list(combined.columns))
    for col in combined.columns[1:]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)
    return combined


# ---------------------------------------------------------------------------
# Multiple testing corrections
# ---------------------------------------------------------------------------

def bonferroni_correction(pvals, alpha=0.05):
    """Bonferroni multiple-testing correction."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return {"reject": np.array([], dtype=bool),
                "p_adjusted": np.array([]), "thresholds": []}
    p_adjusted = np.clip(pvals * m, 0, 1)
    reject = p_adjusted <= alpha
    thresholds = [alpha / m] * m
    return {"reject": reject, "p_adjusted": p_adjusted, "thresholds": thresholds}


def benjamini_hochberg_correction(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return {"reject": np.array([], dtype=bool),
                "p_adjusted": np.array([]), "thresholds": []}
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    p_adj_sorted = np.minimum(1, p_sorted * m / (np.arange(m) + 1))
    for i in range(m - 2, -1, -1):
        p_adj_sorted[i] = min(p_adj_sorted[i], p_adj_sorted[i + 1])
    p_adjusted = np.empty(m)
    p_adjusted[order] = p_adj_sorted
    reject = p_adjusted <= alpha
    thresholds = (alpha * (np.arange(m) + 1) / m).tolist()
    return {"reject": reject, "p_adjusted": p_adjusted, "thresholds": thresholds}


def _apply_correction(pvals, alpha, method):
    """Dispatch multiple-testing correction by name."""
    if method == "bonferroni":
        return bonferroni_correction(pvals, alpha)
    if method == "bh":
        return benjamini_hochberg_correction(pvals, alpha)
    return holm_sidak_correction(pvals, alpha)


def statistical_analysis_v2(df_meta, df_p, df_cohort, var, alpha=0.05,
                             correction="holm_sidak"):
    """
    One-way ANOVA + pairwise post-hoc with selectable multiple-testing correction.
    correction: 'holm_sidak' | 'bonferroni' | 'bh'
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    if df_meta.empty or df_cohort.empty:
        return pd.DataFrame(), pd.DataFrame()

    dc = df_cohort.reset_index()
    merged = df_meta.merge(dc, on="Sample Name")
    cohort_cols = list(df_cohort.columns)
    df_long = merged.melt(
        id_vars=["Sample Name", var],
        value_vars=cohort_cols,
        var_name="Cohort",
        value_name="Abundance",
    )

    levels = df_long[var].dropna().unique()
    anova_rows = []
    posthoc_rows = []

    for lvl in levels:
        sub = df_long[df_long[var] == lvl]
        groups = sub.groupby("Cohort")["Abundance"].apply(list)
        if groups.shape[0] < 2:
            anova_rows.append({"level": lvl, "F": np.nan, "PR(>F)": np.nan,
                               "n_groups": groups.shape[0]})
            continue

        try:
            model     = smf.ols("Abundance ~ C(Cohort)", data=sub).fit()
            aov_table = sm.stats.anova_lm(model, typ=2)
            F    = aov_table.loc["C(Cohort)", "F"]     if "C(Cohort)" in aov_table.index else np.nan
            pval = aov_table.loc["C(Cohort)", "PR(>F)"] if "C(Cohort)" in aov_table.index else np.nan
        except Exception:
            try:
                grp_lists = [g for _, g in sub.groupby("Cohort")["Abundance"]]
                F, pval = stats.f_oneway(*grp_lists)
            except Exception:
                F, pval = np.nan, np.nan

        anova_rows.append({"level": lvl, "F": F, "PR(>F)": pval,
                           "n_groups": groups.shape[0]})

        if np.isnan(pval) or pval > alpha:
            continue

        cohort_names = groups.index.tolist()
        pvals_list, pairs, tstats = [], [], []
        for i in range(len(cohort_names)):
            for j in range(i + 1, len(cohort_names)):
                g1, g2 = groups[cohort_names[i]], groups[cohort_names[j]]
                try:
                    t, p = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
                except Exception:
                    t, p = np.nan, np.nan
                pvals_list.append(p)
                pairs.append((cohort_names[i], cohort_names[j]))
                tstats.append(t)

        if pvals_list:
            res        = _apply_correction(pvals_list, alpha, correction)
            p_adjusted = res["p_adjusted"]
            reject     = res["reject"]
            for k, (c1, c2) in enumerate(pairs):
                posthoc_rows.append({
                    var: lvl,
                    "Group1": c1, "Group2": c2,
                    "t-stat": tstats[k],
                    "pval":   pvals_list[k],
                    "p_adj":  p_adjusted[k],
                    "reject": bool(reject[k]),
                })

    anova_df   = pd.DataFrame(anova_rows)
    posthoc_df = pd.DataFrame(posthoc_rows)

    if not anova_df.empty and "PR(>F)" in anova_df.columns:
        anova_df = anova_df.sort_values("PR(>F)")
    if not posthoc_df.empty and "p_adj" in posthoc_df.columns:
        posthoc_df = posthoc_df.sort_values("p_adj")

    return anova_df, posthoc_df


# ---------------------------------------------------------------------------
# PCA with replicate points + 95% confidence ellipses
# ---------------------------------------------------------------------------

def plot_pca_2d_replicates(df_p, df_exps):
    """
    PCA on per-replicate data with individual points coloured by mutation and
    95% confidence ellipses around each mutation cluster.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if df_p is None or df_p.empty or df_exps is None or df_exps.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    exp_cols = [c for c in df_p.columns if c != "Sample Name"]
    if len(exp_cols) < 2:
        ax.text(0.5, 0.5, "Need ≥2 experiments for PCA", ha="center", va="center")
        return fig

    X = df_p[exp_cols].T.values  # n_experiments × n_lipids
    mut_map = dict(zip(df_exps["Exp"], df_exps["Mutation"]))
    mutations_per_exp = [mut_map.get(e, e) for e in exp_cols]

    n_components = min(2, X.shape[0], X.shape[1])
    if n_components < 2:
        ax.text(0.5, 0.5, "Not enough data for 2-component PCA",
                ha="center", va="center")
        return fig

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
    except Exception as e:
        ax.text(0.5, 0.5, f"PCA error: {e}", ha="center", va="center", fontsize=8)
        return fig

    unique_muts = list(dict.fromkeys(mutations_per_exp))
    colors = {m: COHORT_COLORS[i % len(COHORT_COLORS)] for i, m in enumerate(unique_muts)}
    var_exp = pca.explained_variance_ratio_

    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax.set_title("PCA — Replicate Scatter with 95% Confidence Ellipses")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")

    for mut in unique_muts:
        idx = [i for i, m in enumerate(mutations_per_exp) if m == mut]
        xs = scores[idx, 0]
        ys = scores[idx, 1]
        col = colors[mut]
        ax.scatter(xs, ys, color=col, s=80, label=mut,
                   edgecolors="white", linewidths=0.8, zorder=3)
        for k, (xi, yi) in enumerate(zip(xs, ys)):
            ax.annotate(exp_cols[idx[k]], (xi, yi),
                        fontsize=6, xytext=(4, 2), textcoords="offset points",
                        color="grey")
        if len(xs) >= 2:
            confidence_ellipse(xs, ys, ax, n_std=1.96,
                               edgecolor=col, linewidth=1.5, linestyle="--")

    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gaussian curve fitting on chain length histogram
# ---------------------------------------------------------------------------

def _gaussian(x, amp, mu, sigma):
    sigma = max(abs(sigma), 1e-6)
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def _double_gaussian(x, a1, m1, s1, a2, m2, s2):
    return _gaussian(x, a1, m1, s1) + _gaussian(x, a2, m2, s2)


def plot_cl_gaussian_fit(long_df):
    """
    Weighted chain length histogram per cohort with Gaussian curve fitting overlay.
    Tries double-Gaussian first, falls back to single.
    """
    from scipy.optimize import curve_fit

    fig, ax = plt.subplots(figsize=(9, 5))
    if long_df is None or long_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    cohorts = long_df["Mutation"].unique()
    pal = sns.color_palette("tab10", n_colors=len(cohorts))
    all_cl = long_df["Acyl Chain Length"].dropna().astype(int)
    if all_cl.empty:
        ax.text(0.5, 0.5, "No chain length data", ha="center", va="center")
        return fig

    bins = np.arange(all_cl.min(), all_cl.max() + 2)
    x_fine = np.linspace(bins[0], bins[-1], 300)

    for i, cohort in enumerate(cohorts):
        sub = long_df[long_df["Mutation"] == cohort]
        if sub["Abundance"].sum() == 0:
            continue
        cl_vals = sub["Acyl Chain Length"].astype(int).values
        weights = sub["Abundance"].values
        hist, edges = np.histogram(cl_vals, bins=bins, weights=weights, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        col = pal[i]
        ax.bar(centers, hist, width=0.7, color=col, alpha=0.25, label=None)

        try:
            mu_est = np.average(centers, weights=hist + 1e-12)
            sig_est = max(np.sqrt(np.average((centers - mu_est) ** 2,
                                             weights=hist + 1e-12)), 1.0)
            p0_d = [hist.max() * 0.6, mu_est - sig_est, sig_est,
                    hist.max() * 0.4, mu_est + sig_est, sig_est]
            popt, _ = curve_fit(_double_gaussian, centers, hist, p0=p0_d, maxfev=5000)
            ax.plot(x_fine, _double_gaussian(x_fine, *popt),
                    color=col, lw=2, label=f"{cohort} fit")
            ax.axvline(popt[1], color=col, ls=":", lw=1, alpha=0.7)
            ax.axvline(popt[4], color=col, ls=":", lw=1, alpha=0.7)
        except Exception:
            try:
                mu_est = np.average(centers, weights=hist + 1e-12)
                sig_est = max(np.sqrt(np.average(
                    (centers - mu_est) ** 2, weights=hist + 1e-12)), 1.0)
                popt1, _ = curve_fit(_gaussian, centers, hist,
                                     p0=[hist.max(), mu_est, sig_est], maxfev=3000)
                ax.plot(x_fine, _gaussian(x_fine, *popt1),
                        color=col, lw=2,
                        label=f"{cohort} (μ={popt1[1]:.1f})")
            except Exception:
                ax.plot(centers, hist, color=col, lw=1.5,
                        label=f"{cohort} (no fit)")

    ax.set_xlabel("Acyl Chain Length")
    ax.set_ylabel("Density (weighted)")
    ax.set_title("Chain Length Distribution — Gaussian Fit")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Chain length distribution outlier identification
# ---------------------------------------------------------------------------

def identify_cl_outliers(cl_data_dict):
    """
    Rank cohorts by Jensen-Shannon divergence from the mean chain length
    distribution. Most-divergent cohorts are flagged as outliers.

    Returns
    -------
    pd.DataFrame  cols = ['Cohort', 'JS_Divergence', 'Rank']
    """
    prop = cl_data_dict.get("cohort_prop") if cl_data_dict else None
    if prop is None or prop.empty:
        return pd.DataFrame()

    from scipy.spatial.distance import jensenshannon

    df = prop.fillna(0).copy()
    mean_dist = df.mean(axis=1).values
    total = mean_dist.sum()
    mean_dist = mean_dist / total if total > 0 else mean_dist

    rows = []
    for col in df.columns:
        q = df[col].values
        q_total = q.sum()
        q = q / q_total if q_total > 0 else q
        js = float(jensenshannon(mean_dist, q) ** 2)
        rows.append({"Cohort": col, "JS_Divergence": round(js, 5)})

    out = pd.DataFrame(rows).sort_values("JS_Divergence", ascending=False)
    out["Rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Odd-chain length KDE (distribution of odd chain lengths only)
# ---------------------------------------------------------------------------

def plot_odd_chain_kde(df_meta, df_p):
    """
    KDE of chain length values restricted to odd-chain lipids, per mutation.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if df_meta is None or df_meta.empty or df_p is None or df_p.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    odd_names = df_meta.loc[
        df_meta["Acyl Chain Length"] % 2 == 1, "Sample Name"
    ].values

    dp_odd = df_p[df_p["Sample Name"].isin(odd_names)].copy()
    if dp_odd.empty:
        ax.text(0.5, 0.5, "No odd-chain lipids found", ha="center", va="center")
        return fig

    dm_odd = df_meta[df_meta["Sample Name"].isin(odd_names)]
    exp_cols = [c for c in dp_odd.columns if c != "Sample Name"]
    long_df = dm_odd.merge(dp_odd, on="Sample Name").melt(
        id_vars=["Sample Name", "Acyl Chain Length"],
        value_vars=exp_cols,
        var_name="Mutation", value_name="Abundance"
    )
    plt.close(fig)  # close blank figure before returning new one
    return plot_kde_histogram(
        long_df, "Acyl Chain Length", "Mutation",
        "Odd-Chain Lipids — Chain Length KDE",
        "Acyl Chain Length (odd only)"
    )


# ---------------------------------------------------------------------------
# Per-bin (point-wise) statistical testing
# ---------------------------------------------------------------------------

def pointwise_stat_test(df_meta, df_p, var, control_mtn, alpha=0.05,
                        correction="holm_sidak"):
    """
    For each unique level of `var`, run Welch t-test vs control and correct.

    Returns
    -------
    pd.DataFrame  cols = [var, 'Cohort', 't-stat', 'pval', 'p_adj', 'significant']
    """
    if df_meta.empty or df_p.empty or not control_mtn:
        return pd.DataFrame()

    df = df_meta[["Sample Name", var]].merge(df_p, on="Sample Name")
    exp_cols = [c for c in df_p.columns if c != "Sample Name"]
    df_long = df.melt(
        id_vars=["Sample Name", var],
        value_vars=exp_cols,
        var_name="Cohort", value_name="Abundance"
    )

    non_ctrl = [c for c in df_long["Cohort"].unique() if c != control_mtn]
    levels   = sorted(df_long[var].dropna().unique())

    raw_rows = []
    for lvl in levels:
        sub = df_long[df_long[var] == lvl]
        ctrl_vals = sub.loc[sub["Cohort"] == control_mtn, "Abundance"].dropna().tolist()
        if not ctrl_vals:
            continue
        for coh in non_ctrl:
            coh_vals = sub.loc[sub["Cohort"] == coh, "Abundance"].dropna().tolist()
            if not coh_vals:
                continue
            try:
                t, p = stats.ttest_ind(coh_vals, ctrl_vals, equal_var=False)
            except Exception:
                t, p = np.nan, np.nan
            raw_rows.append({var: lvl, "Cohort": coh, "t-stat": t, "pval": p})

    if not raw_rows:
        return pd.DataFrame()

    result = pd.DataFrame(raw_rows)
    valid_mask = result["pval"].notna()
    if valid_mask.sum() > 0:
        corr = _apply_correction(result.loc[valid_mask, "pval"].tolist(), alpha, correction)
        result.loc[valid_mask, "p_adj"] = corr["p_adjusted"]
        result.loc[valid_mask, "significant"] = corr["reject"]
    result["p_adj"] = result.get("p_adj", pd.Series(np.nan, index=result.index))
    result["significant"] = result.get("significant", pd.Series(False, index=result.index))
    return result.sort_values(["Cohort", var]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Subgroup analysis (MADAG / Sphingolipids / any subset)
# ---------------------------------------------------------------------------

SUBGROUP_MADAG         = ["MAG", "DAG", "TAG", "MAG-O", "DAG-O", "TG", "DG", "MG"]
SUBGROUP_SPHINGOLIPIDS = ["Cer", "HexCer", "Hex_Cer", "SM", "GM", "GD", "GT",
                          "Hex2Cer", "LacCer", "Gb3"]


def subgroup_analysis(df_meta, df_p, df_cohort, head_groups, control_mtn):
    """
    Focused analysis on a subset of lipid head groups.

    Returns
    -------
    dict  keys: 'prop', 'logfc', 'zscore', 'meta_sub', 'cohort_sub'
    """
    if df_meta.empty or df_p.empty or df_cohort.empty:
        return {}

    dm_sub = df_meta[
        df_meta["Head Group 2"].isin(head_groups) |
        df_meta["Head Group"].isin(head_groups)
    ].copy()
    if dm_sub.empty:
        return {}

    keep_lipids = set(dm_sub["Sample Name"])
    dp_sub = df_p[df_p["Sample Name"].isin(keep_lipids)].copy()
    dc_sub = df_cohort[df_cohort.index.isin(keep_lipids)].copy()

    if dp_sub.empty or dc_sub.empty:
        return {}

    cohort_cols = list(df_cohort.columns)
    hg_raw = dm_sub.merge(dc_sub.reset_index(), on="Sample Name")
    hg_sum = hg_raw.groupby("Head Group 2")[cohort_cols].sum()
    col_totals = hg_sum.sum(axis=0).replace(0, np.nan)
    hg_prop = hg_sum.div(col_totals, axis=1)

    lfc = pd.DataFrame()
    zsc = pd.DataFrame()
    if control_mtn and control_mtn in dp_sub.columns:
        try:
            lfc = fold_change(dm_sub, dp_sub, "Head Group 2", control_mtn)
        except Exception:
            pass
        try:
            zsc = z_score(dm_sub, dp_sub, "Head Group 2", control_mtn)
        except Exception:
            pass

    return {"prop": hg_prop, "logfc": lfc, "zscore": zsc,
            "meta_sub": dm_sub, "cohort_sub": dc_sub}


# ---------------------------------------------------------------------------
# Global change summary: universally + mutation-specifically changed lipids
# ---------------------------------------------------------------------------

def summarise_top_changes(df_meta, df_p, df_cohort, control_mtn, top_n=20):
    """
    Identify lipids most consistently (global) or uniquely (specific) changed.

    Returns
    -------
    global_df   : top_n lipids changed in same direction across ALL non-ctrl mutations
    specific_df : per-mutation top_n most mutation-specific lipids
    """
    if df_meta.empty or df_p.empty or df_cohort.empty or not control_mtn:
        return pd.DataFrame(), pd.DataFrame()
    if control_mtn not in df_cohort.columns:
        return pd.DataFrame(), pd.DataFrame()

    dc = df_cohort.copy()
    ctrl_col = dc[control_mtn].replace(0, np.nan)
    non_ctrl_cols = [c for c in dc.columns if c != control_mtn]
    if not non_ctrl_cols:
        return pd.DataFrame(), pd.DataFrame()

    with np.errstate(divide="ignore", invalid="ignore"):
        lfc = np.log2(dc[non_ctrl_cols].div(ctrl_col, axis=0))
    lfc.replace([np.inf, -np.inf], np.nan, inplace=True)
    lfc.index.name = "Sample Name"
    lfc = lfc.reset_index()

    # Global: highest mean |log2FC| with consistent sign
    lfc["mean_abs_lfc"] = lfc[non_ctrl_cols].abs().mean(axis=1)
    signs = np.sign(lfc[non_ctrl_cols].fillna(0))
    lfc["sign_consistency"] = signs.sum(axis=1).abs() / len(non_ctrl_cols)
    global_df = lfc.nlargest(top_n, "mean_abs_lfc")[
        ["Sample Name", "mean_abs_lfc", "sign_consistency"] + non_ctrl_cols
    ].round(4).reset_index(drop=True)

    # Mutation-specific
    spec_rows = []
    lfc_vals = lfc.set_index("Sample Name")[non_ctrl_cols]
    for mut in non_ctrl_cols:
        others = [c for c in non_ctrl_cols if c != mut]
        other_mean = lfc_vals[others].abs().mean(axis=1) if others else pd.Series(0, index=lfc_vals.index)
        specificity = lfc_vals[mut].abs() / (other_mean + 1e-6)
        top_idx = specificity.nlargest(top_n).index
        for lipid in top_idx:
            spec_rows.append({
                "Sample Name":        lipid,
                "Mutation":           mut,
                "log2FC":             round(float(lfc_vals.loc[lipid, mut]), 3)
                                      if not pd.isna(lfc_vals.loc[lipid, mut]) else np.nan,
                "Specificity Score":  round(float(specificity.loc[lipid]), 3),
            })
    specific_df = pd.DataFrame(spec_rows)

    return global_df, specific_df
