import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from scripts.scoring_functions import ScaleScorer
from scripts.helpers import normalize_column_name


def handle_missing_values(df):
    """
    Handle missing values by filling numerical columns with the mean
    and categorical columns with 'Missing'.
    """
    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = num_cols.fillna(num_cols.mean())

    cat_cols = df.select_dtypes(include='object')
    df[cat_cols.columns] = cat_cols.fillna('Missing')
    return df


def detect_column_types(df):
    """
    Detect column types as nominal, ordinal, free text, or timestamp.
    """
    column_types = {
        'nominal': [],
        'ordinal': [],
        'free_text': [],
        'timestamp': []
    }

    cat_cols = df.select_dtypes(include='object')

    # Heuristic to classify free text or categorical columns
    for col in cat_cols.columns:
        unique_values = df[col].nunique()
        total_rows = len(df)

        if unique_values / total_rows > 0.3:
            column_types['free_text'].append(col)
        else:
            # Identify ordinal columns using common patterns
            ordered_keywords = ['never', 'rarely', 'sometimes', 'often', 'always', 'Strongly disagree',
                                'Disagree', 'Disagree somewhat','Neither agree nor disagree', 'Agree somewhat','Agree', 'Strongly agree','Never', 'Less than once a month', 'Once a month',
                                '2-3 times a month', 'Once a week', '2-3 times a week',
                                'About once a day', 'Two or more times per day',
                                'Not at all', 'A little', 'Moderately', 'Quite a bit', 'Extremely',
                                'Not at all', 'Somewhat', 'Extremely']

            if any(keyword in [str(val).lower() for val in df[col].unique()]
                   for keyword in ordered_keywords):
                column_types['ordinal'].append(col)
            else:
                column_types['nominal'].append(col)

    return column_types


def encode_nominal_columns(df, column_types):
    """
    Encode nominal columns using LabelEncoder, with special handling for binary columns.
    """
    for col in column_types['nominal']:
        if df[col].nunique() == 2:  # Handle binary columns explicitly
            df[col] = df[col].map({'No': 0, 'Yes': 1})  # Adjust based on actual values
        else:
            # Use LabelEncoder for other nominal columns
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to handle non-string categories
        print(f"[DEBUG] Encoding nominal column: {col}")
    return df


def encode_columns_for_jamovi(df, column_types):
    """
    Encode ordinal columns with OrdinalEncoder and nominal columns with codes,
    keeping the data interpretable for Jamovi.
    """
    # Step 1: Handle ordinal columns (keeping order intact)
    for col in column_types['ordinal']:
        print(f"[DEBUG] Encoding ordinal column: {col}")
        try:
            encoder = OrdinalEncoder(categories=[['never', 'rarely', 'sometimes', 'often', 'always'],
                                                 ['Strongly disagree', 'Disagree', 'Disagree somewhat',
                                                   'Neither agree nor disagree', 'Agree somewhat',
                                                   'Agree', 'Strongly agree'],
                                                 ['Never', 'Less than once a month', 'Once a month',
                                                  '2-3 times a month', 'Once a week', '2-3 times a week',
                                                  'About once a day', 'Two or more times per day'],
                                                 ['Not at all', 'A little', 'Moderately', 'Quite a bit', 'Extremely'],
                                                 ['Not at all', 'Somewhat', 'Extremely']], dtype=int)
            df[col] = encoder.fit_transform(df[[col]]) + 1  # +1 to avoid 0-based indexing
        except Exception as e:
            print(f"[ERROR] Ordinal encoding failed for '{col}': {e}")

    # Step 2: Handle nominal columns (label encoding to keep single column structure)
    df = encode_nominal_columns(df, column_types)

    return df


def sanity_check_chills(df, chills_column, chills_intensity_column, threshold=0):
    """
    Sanity check to flag inconsistencies between chills columns.
    """
    inconsistent_rows = (df[chills_column] == 0) & (df[chills_intensity_column] > threshold)
    df['Sanity_Flag'] = inconsistent_rows.astype(int)
    return df


def preprocess_data(df):
    """
    Preprocess the DataFrame for Jamovi analysis with proper encodings.
    """
    column_types = detect_column_types(df)
    df = encode_columns_for_jamovi(df, column_types)

    # Normalize column names
    df.columns = [normalize_column_name(col) for col in df.columns]

    df = df.astype({col: 'float64' for col in df.select_dtypes(include=[np.int64, np.float64]).columns})
    return df


def generate_qa_report(df):
    """
    Generate a QA report identifying missing values and outliers.
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
    """
    col_data = df[[column_name]].dropna()
    if col_data.std().iloc[0] == 0:
        return 0

    scaler = StandardScaler()
    z_scores = scaler.fit_transform(col_data)
    return (np.abs(z_scores) > threshold).sum()


def process_data_pipeline(input_df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag',
                          user_column_mappings=None):
    """
    Full data processing pipeline, tailored for Jamovi analysis.
    """
    df = handle_missing_values(input_df)
    qa_report = generate_qa_report(df)
    df = sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold)
    intermediate_df = preprocess_data(df.copy())

    scorer = ScaleScorer(intermediate_df, user_column_mappings)
    final_df = scorer.calculate_all_scales()

    return final_df, intermediate_df, str(qa_report)
