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
                blank_threshold=0.05):
    """
    Filter the raw DataFrame based on metadata criteria.

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

    # Remove blank columns (samples)
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
