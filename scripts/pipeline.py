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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys


def handle_missing_values(df):
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
    Detect outliers in a numerical column using StandardScaler from scikit-learn for Z-score calculation.

    :param df: Pandas DataFrame containing the data
    :param column_name: The name of the column to detect outliers in
    :param threshold: The number of standard deviations from the mean to consider as an outlier
    :return: Count of outliers in the specified column
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
    # TODO - code trans-female and trans-male into 'Trans'


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
        numeric codes and numerical columns normalized. Free text columns
        are left unmodified.

    Description
    -----------
    This function simplifies the 'Gender' column using the `simplify_gender`
    function, converts standard categorical columns (non-free-text) into numeric codes,
    and normalizes the numerical columns by subtracting the mean and dividing by the standard deviation.
    Free text columns are left as-is.
    """
    # Step 1: Simplify the 'Gender' column if present
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].apply(simplify_gender)

    # TODO - gender and age are being converted

    # Step 2: Identify free text columns (e.g., long descriptions)
    # We'll assume free text columns have a high number of unique values compared to the number of rows
    text_cols = df.select_dtypes(include='object')
    free_text_cols = [col for col in text_cols.columns if df[col].nunique() > (0.3 * len(df))]

    # Step 3: Handle other categorical columns (excluding free text columns) by converting to numeric codes
    cat_cols = text_cols.drop(columns=free_text_cols)  # Exclude free text columns
    df[cat_cols.columns] = cat_cols.apply(lambda col: col.astype('category').cat.codes)

    # Step 4: Normalize numerical columns
    scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = scaler.fit_transform(df[num_cols.columns])

    return df

# Modify process_data_pipeline to handle DataFrames instead of files
def process_data_pipeline(input_df):
    """
    Main pipeline function to handle the following:
    1. Perform automated QA on the input DataFrame.
    2. Generate a QA report.
    3. Preprocess the data for output.

    Parameters
    ----------
    input_df : pd.DataFrame
        The input DataFrame to be processed.

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

    # Step 3: Preprocess the data for output
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

