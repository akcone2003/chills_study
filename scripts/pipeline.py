import pandas as pd
from AutoClean import AutoClean  # py-AutoClean library
from scripts.scoring_functions import ScaleScorer
from scripts.helpers import normalize_column_name

def preprocess(df):
    """
    Preprocesses the dataframe using AutoClean.
    """
    return AutoClean(df, missing_num='auto', missing_categ=False, encode_categ='auto')

def sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag'):
    """
    Perform a sanity check for inconsistencies between 'chills_column' and 'chills_intensity_column'.
    If 'chills_column' is 0, but 'chills_intensity_column' > intensity_threshold, flag or drop the row.
    """
    inconsistent_mask = (df[chills_column] == 0) & (df[chills_intensity_column] > intensity_threshold)

    if mode == 'flag':
        df = df.assign(Sanity_Flag=inconsistent_mask)
    elif mode == 'drop':
        df = df.loc[~inconsistent_mask]

    return df

def generate_qa_report(df):
    """
    Generate a Quality Assurance (QA) report with missing value statistics.
    """
    missing_counts = df.isnull().sum()
    excessive_missing_rows = df[df.isnull().sum(axis=1) >= 3]

    return {
        'missing_values': missing_counts.to_dict(),
        'rows_with_3_or_more_missing_values': {
            'count': len(excessive_missing_rows),
            'row_indices': excessive_missing_rows.index.tolist()
        }
    }

def normalize_columns(df):
    """
    Normalize column names for consistency.
    """
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df

def process_data_pipeline(input_df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag',
                          user_column_mappings=None):
    """
    Main pipeline function integrating all processing steps.
    """
    # Step 1: Clean and preprocess the data
    cleaned_df = preprocess(input_df)

    # Step 2: Generate a QA report
    qa_report = generate_qa_report(cleaned_df)

    # Step 3: Perform sanity check on chills data
    cleaned_df = sanity_check_chills(
        cleaned_df, chills_column, chills_intensity_column, intensity_threshold, mode
    )

    # Step 4: Normalize column names for consistency
    cleaned_df = normalize_columns(cleaned_df)

    # Step 5: Calculate scores using ScaleScorer
    scorer = ScaleScorer(cleaned_df, user_column_mappings)
    final_df = scorer.calculate_all_scales()

    return final_df, cleaned_df, str(qa_report)
