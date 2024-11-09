import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from scripts.scoring_functions import ScaleScorer
from scripts.utils import normalize_column_name, ORDERED_KEYWORD_SET
import torch
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


@st.cache_resource
def load_model():
    """Load BERT tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased', torch_dtype=torch.float32).to('cpu')
    return tokenizer, model


# Use the cached resources within your functions when needed.
tokenizer, model = load_model()


def get_embedding(text):
    """
    Get the BERT-based embedding for a given text.
    """
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)

    # Use the [CLS] token as the sentence embedding
    embedding = output.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return embedding


def handle_missing_values(df):
    """
    Fill missing values in numerical columns with their mean and in
    categorical columns with 'Missing'.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame with potential missing values.

    Returns:
    -------
    pd.DataFrame
        DataFrame with filled missing values.
    """
    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = df[num_cols.columns].fillna(num_cols.mean())

    cat_cols = df.select_dtypes(include='object')
    df[cat_cols.columns] = df[cat_cols.columns].fillna('Missing')
    return df


def detect_column_types(df):
    """
    Classify columns into nominal, ordinal, or free text based on data patterns.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing mixed data types.

    Returns:
    -------
    dict
        Dictionary with column classifications.
    """
    column_types = {'nominal': [], 'ordinal': [], 'free_text': []}
    cat_cols = df.select_dtypes(include='object')

    for col in cat_cols.columns:
        unique_values = df[col].nunique()
        total_rows = len(df)

        # Classify as 'free_text' if many unique values
        if unique_values / total_rows > 0.3:
            column_types['free_text'].append(col)
        else:
            # Check if the column values match any known scale in ORDERED_KEYWORD_SET
            if any(
                    keyword in [normalize_column_name(val) for val in df[col].unique()]
                    for scale in ORDERED_KEYWORD_SET.values()
                    for keyword in scale
            ):
                column_types['ordinal'].append(col)
            else:
                column_types['nominal'].append(col)

    return column_types


def determine_category_order(col_values):
    """
    Dynamically determine the correct order of categories for a given column.

    Parameters:
    ----------
    col_values : list
        List of unique values in the column.

    Returns:
    -------
    list
        Sorted list of categories in the determined order.
    """
    lower_col_values = [normalize_column_name(val) for val in col_values]
    best_match = None
    best_match_count = 0

    # Try to match the column values with the known ordered keyword sets
    for scale_name, keywords in ORDERED_KEYWORD_SET.items():
        match_count = sum(1 for val in lower_col_values if val in keywords)
        if match_count > best_match_count:
            best_match = keywords
            best_match_count = match_count

    # Sort based on the matched order, or use semantic similarity if no match
    if best_match:
        print(f"[DEBUG] Best Match Found: {best_match}")
        return sorted(col_values,
                      key=lambda x: best_match.index(x.lower()) if x.lower() in best_match else float('inf'))
    else:
        print("[WARN] No predefined match found. Using semantic similarity.")
        embeddings = {val: get_embedding(val) for val in col_values}
        ref_min = get_embedding("least")
        ref_max = get_embedding("most")

        scores = {val: cosine_similarity([embedding], [ref_max])[0][0] -
                       cosine_similarity([embedding], [ref_min])[0][0]
                  for val, embedding in embeddings.items()}
        return sorted(col_values, key=lambda x: scores[x])


def encode_columns(df, column_types):
    """
    Encode ordinal and nominal columns to make them interpretable for statistical analysis.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column_types : dict
        Dictionary containing the classified column types.

    Returns:
    -------
    pd.DataFrame
        DataFrame with encoded columns.
    """
    # Step 1: Handle ordinal columns (using predefined or dynamically detected categories)
    for col in column_types['ordinal']:
        try:
            if col in ORDERED_KEYWORD_SET:
                # Use predefined categories for known ordinal columns
                categories = [ORDERED_KEYWORD_SET[col]]
            else:
                # Dynamically determine categories for unseen ordinal columns
                unique_values = df[col].dropna().unique()
                categories = [determine_category_order(unique_values)]
                print(f"\n\n[DEBUG] Determined Order for {col}: {categories}")  # Track category order

            encoder = OrdinalEncoder(categories=categories)
            df[col] = encoder.fit_transform(df[[col]]) + 1  # +1 to avoid 0-based indexing
        except Exception as e:
            print(f"[ERROR] Ordinal encoding failed for '{col}': {e}")

    # Step 2: Handle nominal columns (label encoding to keep single column structure)
    for col in column_types['nominal']:
        if df[col].nunique() == 2:  # Handle binary columns explicitly
            df[col] = df[col].map({'No': 0, 'Yes': 1})  # Adjust based on actual values
        else:
            # Use LabelEncoder for other nominal columns
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to handle non-string categories

    return df


def sanity_check_chills(df, chills_column, chills_intensity_column, threshold=0):
    """
    Sanity check to flag inconsistencies between chills columns.

    This function checks for inconsistencies where a subject reports no chills
    (indicated by a 0 in the chills column) but records a non-zero intensity
    (greater than the given threshold) in the chills intensity column.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame to be checked for inconsistencies.
    chills_column : str
        Name of the column that indicates whether chills were experienced (0 or 1).
    chills_intensity_column : str
        Name of the column representing the intensity of chills.
    threshold : int, optional
        The minimum value for chills intensity to flag an inconsistency.
        Default is 0, meaning any non-zero intensity will trigger the flag.

    Returns:
    -------
    pd.DataFrame
        A copy of the input DataFrame with an additional column 'Sanity_Flag'.
        This column contains 1 for inconsistent rows and 0 for consistent ones.
    """
    inconsistent_rows = (df[chills_column] == 0) & (df[chills_intensity_column] > threshold)
    df['Sanity_Flag'] = inconsistent_rows.astype(int)
    return df


def preprocess_data(df):
    """
    Normalize column names, detect types, and encode data for statistical analysis and aggregating behavioral measure scores.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame to be preprocessed.

    Returns:
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for analysis.
    """
    # Normalize column names right at the beginning
    df.columns = [normalize_column_name(col) for col in df.columns]

    # Detect column types
    column_types = detect_column_types(df)

    # Encode columns
    df = encode_columns(df, column_types)

    # Ensure numeric columns are consistent in type (float64)
    df = df.astype({col: 'float64' for col in df.select_dtypes(include=[np.int64, np.float64]).columns})
    return df


def generate_qa_report(df):
    """
    Generate a QA report identifying missing values, outliers, and rows with many missing values.

    This function scans the DataFrame for missing values and numerical outliers. It also identifies
    rows that contain a significant number (3 or more) of missing values to assist with quality assurance.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame to be analyzed for QA issues.

    Returns:
    -------
    dict
        A dictionary containing the QA report with the following keys:
        - 'missing_values': Dictionary with column names as keys and count of missing values as values.
        - 'outliers': Dictionary with column names as keys and count of detected outliers as values.
        - 'rows_with_3_or_more_missing_values': Dictionary with:
            - 'count': Number of rows containing 3 or more missing values.
            - 'row_indices': List of indices of these rows.
    """

    report = {'missing_values': df.isnull().sum().to_dict()}

    outliers_report = {}
    num_cols = df.select_dtypes(include=np.number)
    for col in num_cols:
        outliers_count = detect_outliers(df, col)
        if outliers_count > 0:
            outliers_report[col] = outliers_count
    report['outliers'] = outliers_report

    rows_with_many_missing = df[df.isnull().sum(axis=1) >= 3]
    report['rows_with_3_or_more_missing_values'] = {
        'count': len(rows_with_many_missing),
        'row_indices': rows_with_many_missing.index.tolist()
    }

    return report


def detect_outliers(df, column_name, threshold=3):
    """
    Detect outliers in a numerical column using Z-scores.

    This function calculates Z-scores for each value in a specified column.
    Outliers are defined as values with an absolute Z-score greater than the
    given threshold.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the column to analyze.
    column_name : str
        The name of the numerical column to check for outliers.
    threshold : int, optional
        The Z-score threshold to identify outliers. Default is 3.

    Returns:
    -------
    int
        The number of outliers found in the specified column.
    """
    col_data = df[[column_name]].dropna()
    if col_data.std().iloc[0] == 0:
        return 0

    scaler = StandardScaler()
    z_scores = scaler.fit_transform(col_data)
    return (np.abs(z_scores) > threshold).sum()


# Full pipeline
def process_data_pipeline(input_df, chills_column=None, chills_intensity_column=None, intensity_threshold=0, mode='flag',
                          user_column_mappings=None):
    """
    Main pipeline function that handles QA, sanity checks, encoding, and scoring.

    This function serves as the main entry point for processing data. It handles
    missing values, generates a QA report, performs a sanity check for the
    'chills' columns, preprocesses the data, and calculates scores using a
    provided scorer.

    Parameters:
    ----------
    input_df : pd.DataFrame
        The raw input DataFrame to process.
    chills_column : str
        The column representing whether chills were experienced (e.g., 0 or 1).
    chills_intensity_column : str
        The column representing the intensity of chills.
    intensity_threshold : int, optional
        The threshold for intensity to flag inconsistencies in the sanity check.
        Default is 0.
    mode : str, optional
        Processing mode; not currently used but reserved for future extensions.
        Default is 'flag'.
    user_column_mappings : dict, optional
        Custom mappings provided by the user for scoring purposes.

    Returns:
    -------
    tuple
        - final_df (pd.DataFrame): DataFrame with all calculated scale scores.
        - intermediate_df (pd.DataFrame): Preprocessed DataFrame before scoring.
        - qa_report (str): JSON-like string representation of the QA report. This is downloaded as a .txt file in the application.
    """
    # Step 1: Handle missing values
    df = handle_missing_values(input_df)

    # Step 2: Run automated QA and generate QA report
    qa_report = generate_qa_report(df)

    # Step 3: Perform sanity check for chills response if present in data
    if chills_column and chills_intensity_column:
        df = sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold)

    # Step 4: Preprocess data
    intermediate_df = preprocess_data(df.copy())

    # Step 5: Calculate the scores
    scorer = ScaleScorer(intermediate_df, user_column_mappings)
    final_df = scorer.calculate_all_scales()

    return final_df, intermediate_df, str(qa_report)


# Testing Ground
if __name__ == "__main__":

    # Sample subset of data for testing
    sample_data = {
        "When was the last time you felt moved or touched?": [
            "Within the last month", "Within the last year", "Within the last month",
            "Within the last month", "Within the last 24 hours", "Within the last month",
            "Within the last month"
        ],
        "How often do you feel moved or touched?": [
            "Once a month", "Less than once a month", "2-3 times a week",
            "Once a month", "Once a month", "2-3 times a week", "2-3 times a month"
        ],
        "How often do you get: [Choked-up from a moving or touching experience?]": [
            "Less than once a month", "Less than once a month", "2-3 times a month",
            "Less than once a month", "Once a week", "Once a month", "Once a month"
        ],
        "How often do you get: [Tears or moist eyes from a moving or touching experience?]": [
            "2-3 times a month", "Less than once a month", "2-3 times a month",
            "Less than once a month", "About once a day", "Once a month", "2-3 times a week"
        ],
        "How often do you get: [Shivers (chills) or goosebumps from a moving or touching experience?]": [
            "Less than once a month", "Less than once a month", "2-3 times a month",
            "Less than once a month", "Once a month", "Once a month", "Once a week"
        ],
        "How often do you get: [A warmth in your chest from a moving or touching experience?]": [
            "2-3 times a month", "Less than once a month", "2-3 times a month",
            "Less than once a month", "About once a day", "Once a week", "Once a week"
        ],
        "Would you describe yourself as someone who gets easily moved or touched?": [
            "Somewhat", "Not at all", "Extremely", "Somewhat", "Somewhat", "Somewhat", "Extremely"
        ]
    }

    # Convert dictionary to DataFrame
    test_df = pd.DataFrame(sample_data)

    # Run through normalization to test functionality
    try:
        processed_df = process_data_pipeline(test_df)
        print("Processed DataFrame:\n", processed_df)
    except Exception as e:
        print(f"An error occurred: {e}")