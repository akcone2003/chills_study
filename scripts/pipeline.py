import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from scripts.scoring_functions import ScaleScorer
from scripts.helpers import normalize_column_name

# Scale mappings inferred from PDFs
SCALE_MAPPINGS = {
    "Likert_5": {1: "Strongly Disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly Agree"},
    "Likert_7": {
        1: "Strongly Disagree", 2: "Disagree", 3: "Disagree Somewhat",
        4: "Neither Agree nor Disagree", 5: "Agree Somewhat", 6: "Agree", 7: "Strongly Agree"
    },
    "Frequency": {
        "Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4,
        "Very Often": 5, "Always": 6, "Once a month": 2, "2-3 times a week": 4, "Two or more times per day": 5
    },
    "Anxiety_9": {1: "Not at all", 9: "Severely"}
}


def detect_ordinal_pattern(series):
    """
    Detect if the column matches one of the known ordinal patterns.
    """
    unique_vals = series.dropna().unique()
    for scale_name, mapping in SCALE_MAPPINGS.items():
        if all(str(val).strip() in mapping.values() for val in unique_vals):
            print(f"[INFO] Column '{series.name}' matches '{scale_name}' scale.")
            reversed_mapping = {v: k for k, v in mapping.items()}
            return reversed_mapping
    return None


def encode_column(series):
    """
    Encode a column dynamically based on its content.
    """
    mapping = detect_ordinal_pattern(series)
    if mapping:
        return series.map(mapping).fillna(series)

    # If not ordinal, apply OneHotEncoding
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded = onehot_encoder.fit_transform(series.to_frame())
    return pd.DataFrame(
        encoded, columns=[f"{series.name}_{cat}" for cat in onehot_encoder.categories_[0]]
    )


def preprocess_for_output(df):
    """
    Preprocess the DataFrame by encoding ordinal and categorical columns.
    """
    encoded_dfs = []
    for col in df.select_dtypes(include='object').columns:
        print(f"[DEBUG] Encoding column: {col}")
        try:
            encoded_col = encode_column(df[col])
            encoded_dfs.append(encoded_col)
        except Exception as e:
            print(f"[ERROR] Failed to encode column '{col}': {e}")

    # Concatenate all encoded columns and ensure numeric columns are properly typed
    encoded_df = pd.concat([df.select_dtypes(exclude='object')] + encoded_dfs, axis=1)
    encoded_df.columns = [normalize_column_name(col) for col in encoded_df.columns]
    return encoded_df


def handle_missing_values(df):
    """
    Handle missing values by filling numeric columns with mean values
    and categorical columns with 'Missing'.
    """
    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = num_cols.fillna(num_cols.mean())

    cat_cols = df.select_dtypes(include='object')
    for col in cat_cols:
        df[col] = df[col].fillna('Missing')

    return df


def detect_outliers(df, column_name, threshold=3):
    """
    Detect outliers using Z-scores.
    """
    col_data = df[[column_name]].dropna()
    if col_data.std().iloc[0] == 0:
        return 0

    z_scores = np.abs(StandardScaler().fit_transform(col_data))
    return int((z_scores > threshold).sum())


def generate_qa_report(df):
    """
    Generate a QA report summarizing missing values and outliers.
    """
    report = {'missing_values': df.isnull().sum().to_dict()}
    outliers_report = {}
    for col in df.select_dtypes(include=np.number).columns:
        outliers_count = detect_outliers(df, col)
        if outliers_count > 0:
            outliers_report[col] = outliers_count
    report['outliers'] = outliers_report
    return report


def sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag'):
    """
    Check for inconsistencies between chills and intensity columns.
    """
    inconsistent_rows = (df[chills_column] == 0) & (df[chills_intensity_column] > intensity_threshold)
    if mode == 'flag':
        df['Sanity_Flag'] = inconsistent_rows
    elif mode == 'drop':
        df = df[~inconsistent_rows]
    return df


def process_data_pipeline(input_df, chills_column, chills_intensity_column,
                          intensity_threshold=0, mode='flag', user_column_mappings=None):
    """
    Main data processing pipeline for encoding and scoring.
    """
    df = handle_missing_values(input_df)
    df = sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold, mode)
    final_df = preprocess_for_output(df)

    qa_report = generate_qa_report(final_df)
    scorer = ScaleScorer(final_df, user_column_mappings)
    scored_df = scorer.calculate_all_scales()

    return scored_df, final_df, str(qa_report)
