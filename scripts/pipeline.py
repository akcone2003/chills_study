"""
data_pipeline.py
================

This script is designed to process a CSV dataset by handling missing values,
detecting outliers, generating a Quality Assurance (QA) report, simplifying
gender categories, and preprocessing data for statistical analysis.

The main function of this script is `process_data_pipeline`, which integrates
all individual processing steps to ensure the data is cleaned and ready for analysis.

Functions
---------
1. handle_missing_values(df)
2. detect_outliers(df, column_name, threshold=3)
3. generate_qa_report(df)
4. simplify_gender(gender)
5. preprocess_for_output(df)
6. process_data_pipeline(input_file, output_file, qa_report_file)

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import sys


def handle_missing_values(df):
    """
    Handle missing values in the dataframe using appropriate imputation methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential missing values in numerical and categorical columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing numerical values filled with their respective column means
        and missing categorical values filled with the placeholder 'Missing'.

    Description
    -----------
    This function addresses missing values separately for numerical and categorical columns:

    - For numerical columns: Missing values are replaced with the mean of the respective column.

    - For categorical columns: Missing values are filled with a placeholder string 'Missing'. Additionally,
      if a column has more than 50% of its values missing, a warning message is printed indicating the high percentage
      of missing values.

    This approach ensures that no data is dropped and maintains the integrity of both numerical and categorical columns
    for further processing.
    """

    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = num_cols.fillna(num_cols.mean())

    # Fill missing values in categorical columns with 'Missing' only if they are sparse.
    cat_cols = df.select_dtypes(include='object')
    for col in cat_cols:
        if df[col].isnull().mean() > 0.5:  # Flag columns with more than 50% missing
            print(f"Warning: Column '{col}' has a high percentage of missing values.")
        df[col] = df[col].fillna('Missing')

    return df


def detect_outliers(df, column_name, threshold=3):
    """
    Detect outliers in a numerical column using Z-score calculations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the column to be analyzed.
    column_name : str
        The name of the numerical column in which to detect outliers.
    threshold : int, optional, default=3
        The number of standard deviations from the mean to consider as an outlier.
        Values with Z-scores greater than this threshold will be flagged as outliers.

    Returns
    -------
    int
        The count of outliers in the specified column based on the given threshold.

    Description
    -----------
    This function identifies outliers in a numerical column by calculating Z-scores using
    `StandardScaler` from the `scikit-learn` library. It performs the following steps:

    - Drops any `NaN` values from the column to ensure accurate calculations.

    - Checks if the column has zero variance (i.e., no variation), in which case outlier detection is skipped.

    - Computes the Z-scores for each value in the column.

    - Counts the number of values with absolute Z-scores exceeding the specified threshold, indicating outliers.

    If the column has zero variance, the function returns 0, since no outliers can be detected in a constant column.
    """

    col_data = df[[column_name]].dropna()

    # Check if the column has zero variance to avoid issues with scaling
    if col_data.std().iloc[0] == 0:
        return 0  # No outliers if there is no variation

    scaler = StandardScaler()
    z_scores = scaler.fit_transform(col_data)

    outliers_count = (np.abs(z_scores) > threshold).sum()
    return int(outliers_count)


def generate_qa_report(df):
    """
    Generate a Quality Assurance (QA) report that identifies missing values and outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to generate the QA report.

    Returns
    -------
    dict
        A dictionary containing missing values and outliers information.

    Description
    -----------
    This function generates a QA report for the given DataFrame. The report
    includes a summary of columns with missing values and the count of outliers
    in numerical columns. It uses the `detect_outliers` function to flag
    outliers based on the standard deviation method.
    """
    report = {}

    # 1. Missing values report
    missing_values_report = df.isnull().sum()
    report['missing_values'] = missing_values_report[missing_values_report > 0].to_dict()

    # 2. Outlier detection report (on numerical columns)
    outliers_report = {}
    num_cols = df.select_dtypes(include=np.number)
    for col in num_cols.columns:
        outliers_count = detect_outliers(df, col)
        if outliers_count > 0:
            outliers_report[col] = outliers_count
    report['outliers'] = outliers_report

    # 3. Rows with 3 or more missing values
    rows_with_excessive_missing = df[df.isnull().sum(axis=1) >= 3]
    report['rows_with_3_or_more_missing_values'] = {
        'count': rows_with_excessive_missing.shape[0],
        'row_indices': rows_with_excessive_missing.index.tolist()
    }

    return report

# Will revisit simplifying gender if we need to
# Currently not being accounted for and genders are being coded nominally
# def simplify_gender(gender):
#     """
#     Simplify gender categories into broader groups.
#
#     Parameters
#     ----------
#     gender : str
#         A string representing the gender category from the input data.
#
#     Returns
#     -------
#     str
#         The simplified gender category as 'Female', 'Male', 'Non-Binary', or 'Other'.
#
#     Description
#     -----------
#     This function simplifies gender values by grouping them into broader categories:
#     'Female', 'Male', 'Non-Binary', and 'Other'. It detects specific keywords within
#     the input string to assign the category.
#     """
#     if 'Female' in gender:
#         return 'Female'
#     elif 'Male' in gender:
#         return 'Male'
#     elif 'Non-Binary' in gender or 'genderqueer' in gender:
#         return 'Non-Binary'
#     else:
#         return 'Other'


def detect_column_types(df):
    """
    Detect column types and classify them as nominal, ordinal, or free text.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    dict
        A dictionary containing detected column types:
        {
            'nominal': list of nominal columns,
            'ordinal': list of ordinal columns,
            'free_text': list of free text columns,
            'timestamp': list of timestamp columns
        }
    """
    column_types = {
        'nominal': [],
        'ordinal': [],
        'free_text': [],
        'timestamp': []
    }

    cat_cols = df.select_dtypes(include='object')

    for col in cat_cols.columns:
        unique_values = df[col].nunique()
        total_rows = len(df)

        # Detect if a column is a timestamp
        if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0.9 * len(df):
            column_types['timestamp'].append(col)
            continue

        # Heuristic for free text columns (many unique values compared to the number of rows)
        if unique_values / total_rows > 0.3:
            column_types['free_text'].append(col)
        else:
            # Determine if a categorical column is ordinal based on ordered keywords
            ordered_keywords = ['never', 'rarely', 'sometimes', 'often', 'always',
                                'poor', 'fair', 'good', 'agree', 'strongly',
                                'low', 'medium', 'high', 'none', 'basic', 'advanced']

            if any(keyword in [str(val).lower() for val in df[col].unique()] for keyword in ordered_keywords):
                column_types['ordinal'].append(col)
            else:
                column_types['nominal'].append(col)

    return column_types


def preprocess_for_output(df):
    """
    Preprocess the DataFrame by handling categorical columns dynamically
    based on detected types. Skip scaling for categorical and free text columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be preprocessed.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame, ready for statistical analysis.
    """

    # Step 1: Detect column types automatically
    column_types = detect_column_types(df)

    # Step 2: Handle ordinal columns by using OrdinalEncoder to preserve order
    ordinal_encoder = OrdinalEncoder()
    if column_types['ordinal']:
        df[column_types['ordinal']] = ordinal_encoder.fit_transform(df[column_types['ordinal']])

    # Step 3: Handle nominal columns by converting to numeric codes
    for col in column_types['nominal']:
        df[col] = df[col].astype('category').cat.codes

    # Step 4: Free text and timestamp columns are left unchanged
    return df


def sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag'):
    """
    Perform a sanity check for inconsistencies between 'chills_column' and 'chills_intensity_column'.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the chills response columns.
    chills_column : str
        The name of the column representing whether chills were experienced (0 or 1).
    chills_intensity_column : str
        The name of the column representing the intensity of chills.
    intensity_threshold : int, optional
        The threshold value above which intensity is considered non-trivial.
    mode : str, optional, default='flag'
        The mode of handling inconsistent rows. Options:
        - 'flag': Mark inconsistent rows with a new column 'Sanity_Flag' as True.
        - 'drop': Remove the inconsistent rows from the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with the inconsistencies handled based on the specified mode.
    """
    # Identify rows where Chills is 0 but Chills_Intensity exceeds the threshold
    inconsistent_rows = (df[chills_column] == 0) & (df[chills_intensity_column] > intensity_threshold)

    if mode == 'flag':
        # Create a new column 'Sanity_Flag' to mark these rows
        df['Sanity_Flag'] = inconsistent_rows
    elif mode == 'drop':
        # Drop these rows from the DataFrame
        df = df[~inconsistent_rows]

    return df

# Full pipeline
def process_data_pipeline(input_df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag'):
    """
    Main pipeline function to handle the following:
    1. Perform automated QA on the input DataFrame.
    2. Generate a QA report.
    3. Perform sanity checks for chills response.
    4. Preprocess the data for output.

    Parameters
    ----------
    input_df : pd.DataFrame
        The input DataFrame to be processed.
    chills_column : str
        The column representing chills response (0 or 1).
    chills_intensity_column : str
        The column representing the intensity of chills.
    intensity_threshold : int, optional
        The threshold for flagging/removing inconsistent rows.
    mode : str, optional, default='flag'
        The mode of handling inconsistent rows ('flag' or 'drop').

    Returns
    -------
    processed_df : pd.DataFrame
        The preprocessed DataFrame, ready for analysis.
    qa_report : str
        A string representation of the QA report.
    """
    # Step 1: Handle missing values
    df = handle_missing_values(input_df)

    # Step 2: Run automated QA and generate QA report
    qa_report = generate_qa_report(df)

    # Step 3: Perform sanity check for chills response using dynamic columns
    df = sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold, mode)

    # Step 4: Preprocess the data for output
    processed_df = preprocess_for_output(df)

    return processed_df, str(qa_report)  # Return the processed DataFrame and QA report string


if __name__ == "__main__":
    # Check if the correct number of arguments is provided (input and output file paths)
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <input_file>")
    else:
        # Read the input CSV file from the command-line arguments
        input_file = sys.argv[1]

        # Load the CSV file into a DataFrame
        input_df = pd.read_csv(input_file)

        # Process the DataFrame using the pipeline
        processed_df, qa_report = process_data_pipeline(input_df)

        # Save the results back to a file
        processed_output_file = input_file.replace(".csv", "_processed.csv")
        qa_report_file = input_file.replace(".csv", "_qa_report.txt")

        # Save the processed data and QA report to respective files
        processed_df.to_csv(processed_output_file, index=False)
        with open(qa_report_file, 'w') as f:
            f.write(qa_report)

        print(f"Processed data saved to: {processed_output_file}")
        print(f"QA report saved to: {qa_report_file}")

# TODO - figure out a way to automate converting the questionnaire scales
