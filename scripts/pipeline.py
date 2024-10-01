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
from sklearn.preprocessing import StandardScaler
import sys


def handle_missing_values(df):
    """
    Handle missing values in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing numerical values filled with their column means
        and missing categorical values filled with the placeholder 'Missing'.

    Description
    -----------
    This function fills in missing numerical values with the column mean,
    and missing categorical (object) values with the placeholder string 'Missing'.
    """
    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = num_cols.fillna(num_cols.mean())

    cat_cols = df.select_dtypes(include='object')
    df[cat_cols.columns] = cat_cols.fillna('Missing')
    # TODO - Check with Leo if we just want to straight up drop these rows

    return df


def detect_outliers(df, column_name, threshold=3):
    """
    Detect outliers in a numerical column using StandardScaler from scikit-learn for Z-score calculation.

    :param df: Pandas DataFrame containing the data
    :param column_name: The name of the column to detect outliers in
    :param threshold: The number of standard deviations from the mean to consider as an outlier
    :return: Count of outliers in the specified column
    """
    # Extract the column data and reshape it for the scaler (required by StandardScaler)
    col_data = df[[column_name]].dropna()

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the data to calculate Z-scores
    z_scores = scaler.fit_transform(col_data)

    # Count the values where the absolute Z-score exceeds the threshold
    outliers_count = (np.abs(z_scores) > threshold).sum()

    return int(outliers_count)  # Return as an integer
    # TODO - Check with Leo what we want to do with outliers


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

    # Missing values report
    missing_values_report = df.isnull().sum()
    report['missing_values'] = missing_values_report[missing_values_report > 0].to_dict()

    # Outlier detection report (on numerical columns)
    outliers_report = {}
    num_cols = df.select_dtypes(include=np.number)
    for col in num_cols.columns:
        outliers_count = detect_outliers(df, col)
        if outliers_count > 0:
            outliers_report[col] = outliers_count
    report['outliers'] = outliers_report

    return report


def simplify_gender(gender):
    """
    Simplify gender categories into broader groups.

    Parameters
    ----------
    gender : str
        A string representing the gender category from the input data.

    Returns
    -------
    str
        The simplified gender category as 'Female', 'Male', 'Non-Binary', or 'Other'.

    Description
    -----------
    This function simplifies gender values by grouping them into broader categories:
    'Female', 'Male', 'Non-Binary', and 'Other'. It detects specific keywords within
    the input string to assign the category.
    """
    if 'Female' in gender:
        return 'Female'
    elif 'Male' in gender:
        return 'Male'
    elif 'Non-Binary' in gender or 'genderqueer' in gender:
        return 'Non-Binary'
    else:
        return 'Other'
    # TODO - Verify this is how we want to handle gender


def preprocess_for_output(df):
    """
    Preprocess the DataFrame for output to statistical software.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be preprocessed.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame, with categorical columns converted to
        numeric codes and numerical columns normalized.

    Description
    -----------
    This function simplifies the 'Gender' column using the `simplify_gender`
    function, converts all categorical columns into numeric codes, and normalizes
    the numerical columns by subtracting the mean and dividing by the standard deviation.
    """
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].apply(simplify_gender)


    # Handle categorical columns by converting to numeric codes
    cat_cols = df.select_dtypes(include='object')
    df[cat_cols.columns] = cat_cols.apply(lambda col: col.astype('category').cat.codes)

    # Normalize numerical columns
    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = (df[num_cols.columns] - df[num_cols.columns].mean()) / df[num_cols.columns].std()

    return df


def process_data_pipeline(input_file):
    """
    Main pipeline function to process the data.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame, str
        The preprocessed DataFrame and QA report as a formatted string.
    """
    # Step 1: Load raw data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Generate QA report
    qa_report = generate_qa_report(df)

    # Step 4: Preprocess the data
    processed_df = preprocess_for_output(df)

    return processed_df, qa_report


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_pipeline.py <input_file> <output_file> <qa_report_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        qa_report_file = sys.argv[3]
        process_data_pipeline(input_file, output_file, qa_report_file)
