import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
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
    Detect and classify columns as nominal or ordinal.
    """
    column_types = {'nominal': [], 'ordinal': []}
    cat_cols = df.select_dtypes(include='object')

    # Identify ordinal columns based on keywords or known patterns
    for col in cat_cols.columns:
        values = [str(v).lower() for v in df[col].unique()]
        ordinal_keywords = ['never', 'rarely', 'sometimes', 'often', 'always']

        if any(keyword in values for keyword in ordinal_keywords):
            column_types['ordinal'].append(col)
        else:
            column_types['nominal'].append(col)

    print(f"[DEBUG] Detected column types: {column_types}")
    return column_types

def encode_columns(df, column_types):
    """
    Encode nominal and ordinal columns with human-readable values.
    """
    # Encode ordinal columns with their natural order
    for col in column_types['ordinal']:
        print(f"[DEBUG] Encoding ordinal column: {col}")
        unique_values = sorted(df[col].dropna().unique(), key=str)
        encoder = OrdinalEncoder(categories=[unique_values], dtype=int)
        df[col] = encoder.fit_transform(df[[col]]) + 1  # +1 to avoid 0-indexing

    # Encode nominal columns with simple label encoding
    for col in column_types['nominal']:
        print(f"[DEBUG] Encoding nominal column: {col}")
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col]) + 1  # +1 to avoid 0-indexing

    return df

def sanity_check_chills(df, chills_column, chills_intensity_column, threshold=0):
    """
    Sanity check to flag inconsistent chills data.
    """
    inconsistent_rows = (df[chills_column] == 0) & (df[chills_intensity_column] > threshold)
    df['Sanity_Flag'] = inconsistent_rows
    return df

def process_data_pipeline(input_df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag'):
    """
    Main pipeline for preparing data for Jamovi analysis.
    """
    # Step 1: Handle missing values
    df = handle_missing_values(input_df)

    # Step 2: Detect column types
    column_types = detect_column_types(df)

    # Step 3: Perform sanity check on chills columns
    df = sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold)

    # Step 4: Encode columns for human readability
    df = encode_columns(df, column_types)

    # Step 5: Normalize column names (optional, for neatness)
    df.columns = [normalize_column_name(col) for col in df.columns]

    return df
