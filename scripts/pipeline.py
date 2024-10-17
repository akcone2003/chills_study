import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scripts.scoring_functions import calculate_all_scales
from scripts.helpers import normalize_column_name


def handle_missing_values(df):
    """
    Handle missing values in the dataframe using appropriate imputation methods.
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


def sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag'):
    """
    Perform a sanity check for inconsistencies between 'chills_column' and 'chills_intensity_column'.
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

    # Step 4: Normalize dataframe columns
    df.columns = [normalize_column_name(col) for col in df.columns]

    return df


# Full pipeline
def process_data_pipeline(input_df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag',
                          user_column_mappings=None):
    """
    Main pipeline function that handles QA, sanity checks, encoding, and scoring.
    """
    # Step 1: Handle missing values
    df = handle_missing_values(input_df)

    # Step 2: Run automated QA and generate QA report
    qa_report = generate_qa_report(df)

    # Step 3: Perform sanity check for chills response using dynamic columns
    df = sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold, mode)

    # Step 4: Save the data encoded dataset with all columns intact and no aggregation
    intermediate_df = preprocess_for_output(df.copy())

    # Step 5: Final processing with only score columns (question columns dropped)
    final_df = calculate_all_scales(intermediate_df, user_column_mappings, mid_processing=False)

    return final_df, intermediate_df, str(qa_report)
