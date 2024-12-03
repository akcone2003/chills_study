import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from scripts.scoring_functions import ScaleScorer
from scripts.utils import normalize_column_name, ORDERED_KEYWORD_SET
import streamlit as st


# def handle_missing_values(df):
#     """
#     Fill missing values in numerical columns with their mean and in
#     categorical columns with 'Missing'.
#
#     Parameters:
#     ----------
#     df : pd.DataFrame
#         The input DataFrame with potential missing values.
#
#     Returns:
#     -------
#     pd.DataFrame
#         DataFrame with filled missing values.
#     """
#     num_cols = df.select_dtypes(include=np.number)
#     df[num_cols.columns] = df[num_cols.columns].fillna(num_cols.mean())
#
#     cat_cols = df.select_dtypes(include='object')
#     df[cat_cols.columns] = df[cat_cols.columns].fillna('Missing')
#
#     print("\n[DEBUG] Function: handle_missing_values Completed")
#     return df


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

        # Check conditions
        if (unique_values / total_rows > 0.3) and (unique_values > 8):
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

    print("\n[DEBUG] Function: detect_column_types Completed")
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
        # Convert dictionary keys to list if `keywords` is a dictionary
        keyword_list = list(keywords.keys()) if isinstance(keywords, dict) else keywords
        match_count = sum(1 for val in lower_col_values if val in keyword_list)

        # Update best match if this keyword set has more matches
        if match_count > best_match_count:
            best_match = keywords
            best_match_count = match_count

    # Sort based on the matched order, or use semantic similarity if no match
    if best_match:
        print(f"[DEBUG] Best Match Found: {best_match}")

        # If best_match is a dictionary, sort col_values based on dictionary values
        if isinstance(best_match, dict):
            print("\n[DEBUG] Function: determine_category_order Completed")

            return sorted(col_values, key=lambda x: best_match.get(x.lower(), float('inf')))
        else:
            print("\n[DEBUG] Function: determine_category_order Completed")

            return sorted(col_values,
                          key=lambda x: best_match.index(x.lower()) if x.lower() in best_match else float('inf'))


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
            # Dynamically determine categories for unseen ordinal columns (assume list)
            unique_values = df[col].dropna().unique()
            categories = [determine_category_order(unique_values)]
            print(f"\n\n[DEBUG] Determined Order for {col}: {categories}")  # Track category order

            encoder = OrdinalEncoder(categories=categories)
            df[col] = encoder.fit_transform(df[[col]]) + 1  # +1 to avoid 0-based indexing
        except Exception as e:
            print(f"[ERROR] Ordinal encoding failed for '{col}': {e}")

    # Step 2: Handle nominal columns (label encoding to keep single column structure)
    for col in column_types['nominal']:
        try:
            if df[col].nunique() == 2:  # Handle binary columns explicitly
                df[col] = df[col].map({'no': 0, 'yes': 1})  # Adjust based on actual values
            else:
                # Use LabelEncoder for other nominal columns
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to handle non-string categories
        except Exception as e:
            print(f"[ERROR] Nominal encoding failed for '{col}': {e}")

    print("\n[DEBUG] Function: encode_columns Completed")

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

    print("\n[DEBUG] Function: sanity_check_chills Completed")

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

    print("\n[DEBUG] Function: preprocess_data Completed\n")

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
    report['outliers'] = outliers_report

    rows_with_many_missing = df[df.isnull().sum(axis=1) >= 3]
    report['rows_with_3_or_more_missing_values'] = {
        'count': len(rows_with_many_missing),
        'row_indices': rows_with_many_missing.index.tolist()
    }

    return report


# def detect_outliers(df, column_name, threshold=3):
#     """
#     Detect outliers in a numerical column using Z-scores.
#
#     This function calculates Z-scores for each value in a specified column.
#     Outliers are defined as values with an absolute Z-score greater than the
#     given threshold.
#
#     Parameters:
#     ----------
#     df : pd.DataFrame
#         The DataFrame containing the column to analyze.
#     column_name : str
#         The name of the numerical column to check for outliers.
#     threshold : int, optional
#         The Z-score threshold to identify outliers. Default is 3.
#
#     Returns:
#     -------
#     int
#         The number of outliers found in the specified column.
#     """
#     col_data = df[[column_name]].dropna()
#     if col_data.std().iloc[0] == 0:
#         return 0
#
#     scaler = StandardScaler()
#     z_scores = scaler.fit_transform(col_data)
#     return (np.abs(z_scores) > threshold).sum()


# Full pipeline
def process_data_pipeline(input_df, chills_column=None, chills_intensity_column=None, intensity_threshold=0,
                          mode='flag',
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
    # df = handle_missing_values(input_df)

    df = input_df

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

    print("\n[DEBUG] Function: process_data_pipeline Completed")

    return final_df, intermediate_df, str(qa_report)


# Testing Ground
if __name__ == "__main__":
    # Test data for the ASI-3 questionnaire
    test_data = {
        "It is important for me not to appear nervous.": [
            "Very little", "Some", "A little", "A little", "Very little", "Some", "A little", "A little",
            "Very little", "Some", "A little", "A little", "Very little", "Some", "A little", "A little",
            "Very little", "Some", "A little", "A little", "Very little", "Some", "A little", "A little",
            "Very little", "Some", "A little", "A little", "Very little", "Some", "A little", "A little"
        ],
        "When I cannot keep my mind on a task, I worry that I might be going crazy.": [
            "Some", "Very much", "Some", "Very little", "Some", "Very much", "Some", "Very little",
            "Some", "Very much", "Some", "Very little", "Some", "Very much", "Some", "Very little",
            "Some", "Very much", "Some", "Very little", "Some", "Very much", "Some", "Very little",
            "Some", "Very much", "Some", "Very little", "Some", "Very much", "Some", "Very little"
        ],
        "It scares me when my heart beats rapidly.": [
            "A little", "Much", "Some", "Very little", "A little", "Much", "Some", "Very little",
            "A little", "Much", "Some", "Very little", "A little", "Much", "Some", "Very little",
            "A little", "Much", "Some", "Very little", "A little", "Much", "Some", "Very little",
            "A little", "Much", "Some", "Very little", "A little", "Much", "Some", "Very little"
        ],
        "When my stomach is upset, I worry that I might be seriously ill.": [
            "Very much", "A little", "Some", "Much", "Very much", "A little", "Some", "Much",
            "Very much", "A little", "Some", "Much", "Very much", "A little", "Some", "Much",
            "Very much", "A little", "Some", "Much", "Very much", "A little", "Some", "Much",
            "Very much", "A little", "Some", "Much", "Very much", "A little", "Some", "Much"
        ],
        "It scares me when I am unable to keep my mind on a task.": [
            "Some", "Some", "Very much", "A little", "Some", "Some", "Very much", "A little",
            "Some", "Some", "Very much", "A little", "Some", "Some", "Very much", "A little",
            "Some", "Some", "Very much", "A little", "Some", "Some", "Very much", "A little",
            "Some", "Some", "Very much", "A little", "Some", "Some", "Very much", "A little"
        ],
        "When I tremble in the presence of others, I fear what people might think of me.": [
            "Much", "Some", "Very little", "Some", "Much", "Some", "Very little", "Some",
            "Much", "Some", "Very little", "Some", "Much", "Some", "Very little", "Some",
            "Much", "Some", "Very little", "Some", "Much", "Some", "Very little", "Some",
            "Much", "Some", "Very little", "Some", "Much", "Some", "Very little", "Some"
        ],
        "When my chest feels tight, I get scared that I won't be able to breathe properly.": [
            "Very much", "Much", "A little", "Some", "Very much", "Much", "A little", "Some",
            "Very much", "Much", "A little", "Some", "Very much", "Much", "A little", "Some",
            "Very much", "Much", "A little", "Some", "Very much", "Much", "A little", "Some",
            "Very much", "Much", "A little", "Some", "Very much", "Much", "A little", "Some"
        ],
        "When I feel pain in my chest, I worry that I'm going to have a heart attack.": [
            "Some", "Very much", "Much", "A little", "Some", "Very much", "Much", "A little",
            "Some", "Very much", "Much", "A little", "Some", "Very much", "Much", "A little",
            "Some", "Very much", "Much", "A little", "Some", "Very much", "Much", "A little",
            "Some", "Very much", "Much", "A little", "Some", "Very much", "Much", "A little"
        ],
        "I worry that other people will notice my anxiety.": [
            "Much", "Some", "A little", "Very little", "Much", "Some", "A little", "Very little",
            "Much", "Some", "A little", "Very little", "Much", "Some", "A little", "Very little",
            "Much", "Some", "A little", "Very little", "Much", "Some", "A little", "Very little",
            "Much", "Some", "A little", "Very little", "Much", "Some", "A little", "Very little"
        ],
        "When I feel 'spacey' or spaced out I worry that I may be mentally ill.": [
            "Very little", "Much", "Some", "Some", "Very little", "Much", "Some", "Some",
            "Very little", "Much", "Some", "Some", "Very little", "Much", "Some", "Some",
            "Very little", "Much", "Some", "Some", "Very little", "Much", "Some", "Some",
            "Very little", "Much", "Some", "Some", "Very little", "Much", "Some", "Some"
        ],
        "It scares me when I blush in front of people.": [
            "A little", "Some", "Very little", "Much", "A little", "Some", "Very little", "Much",
            "A little", "Some", "Very little", "Much", "A little", "Some", "Very little", "Much",
            "A little", "Some", "Very little", "Much", "A little", "Some", "Very little", "Much",
            "A little", "Some", "Very little", "Much", "A little", "Some", "Very little", "Much"
        ],
        "When I notice my heart skipping a beat, I worry that there is something seriously wrong with me.": [
            "Some", "Very little", "Much", "A little", "Some", "Very little", "Much", "A little",
            "Some", "Very little", "Much", "A little", "Some", "Very little", "Much", "A little",
            "Some", "Very little", "Much", "A little", "Some", "Very little", "Much", "A little",
            "Some", "Very little", "Much", "A little", "Some", "Very little", "Much", "A little"
        ],
        "When I begin to sweat in a social situation, I fear people will think negatively of me.": [
            "Very much", "Some", "Some", "A little", "Very much", "Some", "Some", "A little",
            "Very much", "Some", "Some", "A little", "Very much", "Some", "Some", "A little",
            "Very much", "Some", "Some", "A little", "Very much", "Some", "Some", "A little",
            "Very much", "Some", "Some", "A little", "Very much", "Some", "Some", "A little"
        ],
        "When my thoughts seem to speed up, I worry that I might be going crazy.": [
            "Some", "Much", "Very much", "Some", "Some", "Much", "Very much", "Some",
            "Some", "Much", "Very much", "Some", "Some", "Much", "Very much", "Some",
            "Some", "Much", "Very much", "Some", "Some", "Much", "Very much", "Some",
            "Some", "Much", "Very much", "Some", "Some", "Much", "Very much", "Some"
        ],
        "When my throat feels tight, I worry that I could choke to death.": [
            "A little", "Very much", "Much", "Very little", "A little", "Very much", "Much", "Very little",
            "A little", "Very much", "Much", "Very little", "A little", "Very much", "Much", "Very little",
            "A little", "Very much", "Much", "Very little", "A little", "Very much", "Much", "Very little",
            "A little", "Very much", "Much", "Very little", "A little", "Very much", "Much", "Very little"
        ],
        "When I have trouble thinking clearly, I worry that there is something wrong with me.": [
            "Very much", "Some", "A little", "Some", "Very much", "Some", "A little", "Some",
            "Very much", "Some", "A little", "Some", "Very much", "Some", "A little", "Some",
            "Very much", "Some", "A little", "Some", "Very much", "Some", "A little", "Some",
            "Very much", "Some", "A little", "Some", "Very much", "Some", "A little", "Some"
        ],
        "I think it would be horrible for me to faint in public.": [
            "Some", "Some", "Very much", "Much", "Some", "Some", "Very much", "Much",
            "Some", "Some", "Very much", "Much", "Some", "Some", "Very much", "Much",
            "Some", "Some", "Very much", "Much", "Some", "Some", "Very much", "Much",
            "Some", "Some", "Very much", "Much", "Some", "Some", "Very much", "Much"
        ],
        "When my mind goes blank, I worry there is something terribly wrong with me.": [
            "A little", "Very little", "Some", "Much", "A little", "Very little", "Some", "Much",
            "A little", "Very little", "Some", "Much", "A little", "Very little", "Some", "Much",
            "A little", "Very little", "Some", "Much", "A little", "Very little", "Some", "Much",
            "A little", "Very little", "Some", "Much", "A little", "Very little", "Some", "Much"
        ]
    }

    # Convert the test data to a DataFrame
    test_df = pd.DataFrame(test_data)

    print(test_df.head())

    # User-defined column mappings
    user_column_mappings = test_df.columns

    # Run the pipeline
    final_df, intermediate_df, qa_report = process_data_pipeline(
        input_df=test_df,  # Input DataFrame
        chills_column=None,  # No chills columns provided
        chills_intensity_column=None,
        intensity_threshold=0,  # Not used here
        user_column_mappings=user_column_mappings  # Response mappings for scoring
    )

    # Print outputs
    print("Final DataFrame:\n", final_df)
    print("\nIntermediate DataFrame:\n", intermediate_df)
    print("\nQA Report:\n", qa_report)
