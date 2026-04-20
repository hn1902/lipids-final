import pandas as pd
import sys
from app.analysis import load_and_deduplicate_data, extract_experiments

def test_advanced_parsing():
    print("Testing advanced parsing with a1_pmol.csv and a1_header.csv...")
    path_data = r"c:\Users\Hayag\lipidsop\lipids_dataset_analysis\csv\dgat\a1_pmol.csv"
    path_head = r"c:\Users\Hayag\lipidsop\lipids_dataset_analysis\csv\dgat\a1_header.csv"

    # Use skip_rows=0 because the actual sample names are in row 0 of a1_pmol.csv
    try:
        df = load_and_deduplicate_data([path_data], idx_col="feature", num_idx=3, skip_rows=0)
        print(f"Data DataFrame Shape: {df.shape}")
        print(f"First 10 columns: {list(df.columns[:10])}")
        
        df_head = pd.read_csv(path_head, header=None)
        print(f"Header DataFrame Shape: {df_head.shape}")
        
        # cohort_row_idx=2 because "cohort" is the 3rd row in a1_header.csv
        df_exps = extract_experiments(df, header_df=df_head, cohort_row_idx=2)
        print(f"Exps Shape: {df_exps.shape}")
        print("Experiment Extraction Results:")
        print(df_exps.head(10))
        
        # Verify that samples correctly mapped to cohorts
        print("\nCohorts found in extraction:")
        print(df_exps['Mutation'].value_counts())
        
        print("\nSUCCESS!")
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_advanced_parsing()
