from sklearn.cluster import KMeans
import spacy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scripts.scoring_functions import ScaleScorer
from scripts.helpers import normalize_column_name

# Load spaCy language model
nlp = spacy.load("en_core_web_md")

def encode_with_embeddings(phrase):
    """
    Encode a phrase using spaCy embeddings by extracting the vector norm.
    """
    doc = nlp(str(phrase))
    return doc.vector_norm

def detect_dynamic_scale(series, max_clusters=10):
    """
    Use KMeans clustering to dynamically detect the appropriate scale range.
    """
    values = series.dropna().unique().reshape(-1, 1)

    # Use KMeans to detect clusters
    kmeans = KMeans(n_clusters=min(max_clusters, len(values)), random_state=42).fit(values)
    num_clusters = len(np.unique(kmeans.labels_))

    # Determine the appropriate scale range based on clusters
    if num_clusters <= 5:
        return 1, 5  # Likert 1-5
    elif num_clusters <= 7:
        return 1, 7  # Likert 1-7
    else:
        return series.min(), series.max()  # Use the data's own min and max as the range

def rescale_values(series, min_value, max_value):
    """
    Rescale a numeric series to fit within a specified min and max range.
    """
    normalized = (series - series.min()) / (series.max() - series.min())  # Normalize to 0-1
    rescaled = normalized * (max_value - min_value) + min_value  # Scale to target range
    return rescaled

def handle_missing_values(df):
    """
    Handle missing values in the dataframe using appropriate imputation methods.
    """
    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = num_cols.fillna(num_cols.mean())

    cat_cols = df.select_dtypes(include='object')
    for col in cat_cols:
        if df[col].isnull().mean() > 0.5:
            print(f"Warning: Column '{col}' has a high percentage of missing values.")
        df[col] = df[col].fillna('Missing')

    return df

def detect_column_types(df):
    """
    Detect column types and classify them as nominal, ordinal, or free text.
    """
    column_types = {'nominal': [], 'ordinal': [], 'free_text': []}
    cat_cols = df.select_dtypes(include='object')

    for col in cat_cols.columns:
        unique_values = df[col].nunique()
        total_rows = len(df)

        print(f"[DEBUG] Column '{col}' unique values: {unique_values}")

        if unique_values / total_rows > 0.3:
            column_types['free_text'].append(col)
        else:
            ordered_keywords = [
                'never', 'rarely', 'sometimes', 'occasionally', 'often', 'always',
                'poor', 'fair', 'good', 'very good', 'excellent',
                'dimly vivid', 'moderately vivid', 'realistically vivid', 'perfectly realistic',
                'low', 'medium', 'high', 'agree', 'disagree', 'strongly agree', 'strongly disagree',
                'none', 'basic', 'advanced', 'no image',
            ]
            values = [str(val).lower() for val in df[col].unique()]
            if any(keyword in values for keyword in ordered_keywords):
                column_types['ordinal'].append(col)
            else:
                column_types['nominal'].append(col)

    print(f"[DEBUG] Detected column types: {column_types}")
    return column_types

def preprocess_for_output(df, user_column_scales=None):
    """
    Preprocess the DataFrame by dynamically encoding nominal and ordinal columns.
    """
    column_types = detect_column_types(df)

    # Handle Ordinal Columns with Embedding-based Encoding and Rescaling
    for col in column_types['ordinal']:
        print(f"[DEBUG] Ordinal encoding column: {col} with unique values: {df[col].unique()}")
        try:
            df[col] = df[col].apply(encode_with_embeddings)

            # Use clustering to detect the appropriate scale range
            min_value, max_value = detect_dynamic_scale(df[col])

            # Rescale the values to the detected scale range
            df[col] = rescale_values(df[col], min_value, max_value)
        except Exception as e:
            print(f"[ERROR] Failed to encode ordinal column '{col}': {e}")

    # Handle Nominal Columns by Converting to Category Codes
    for col in column_types['nominal']:
        print(f"[DEBUG] Nominal encoding column: {col} with unique values: {df[col].unique()}")
        try:
            df[col] = df[col].astype('category').cat.codes
        except Exception as e:
            print(f"[ERROR] Failed to encode nominal column '{col}': {e}")

    df.columns = [normalize_column_name(col) for col in df.columns]
    numeric_cols = df.select_dtypes(include=[np.int64, np.float64]).columns
    df[numeric_cols] = df[numeric_cols].astype('float64')

    return df


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
    inconsistent_rows = (df[chills_column] == 0) & (df[chills_intensity_column] > intensity_threshold)

    if mode == 'flag':
        df['Sanity_Flag'] = inconsistent_rows
    elif mode == 'drop':
        df = df[~inconsistent_rows]

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

def process_data_pipeline(input_df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag',
                          user_column_mappings=None):
    """
    Main pipeline function that handles QA, sanity checks, encoding, and scoring.
    """
    df = handle_missing_values(input_df)
    qa_report = generate_qa_report(df)
    df = sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold, mode)
    intermediate_df = preprocess_for_output(df.copy())
    scorer = ScaleScorer(intermediate_df, user_column_mappings)
    final_df = scorer.calculate_all_scales()

    return final_df, intermediate_df, str(qa_report)
