import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from scripts.scoring_functions import ScaleScorer
from scripts.utils import normalize_column_name, ORDERED_KEYWORD_SET
import streamlit as st
import logging
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Configure logging for better error tracking and debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and converts various forms of missing values to proper NaN.
    This ensures missing values are consistently handled throughout the pipeline.
    
    Args:
        df: Input DataFrame with potential missing values
        
    Returns:
        DataFrame with standardized missing value representation
    """
    df = df.copy()
    
    # Common text representations of missing values
    missing_values = ['nan', 'NaN', 'N/A', 'n/a', 'NA', 'na', 'missing', 
                     'MISSING', 'None', 'none', '', ' ', 'nil', 'NULL', 'null']
    
    # Replace string representations with proper NaN
    df = df.replace(missing_values, np.nan)
    
    # Handle whitespace-only strings as missing
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)
    
    # Convert numeric columns with improper missing values
    for col in df.columns:
        try:
            # Attempt to convert to numeric, forcing non-numeric to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            # If conversion fails, keep the column as is
            pass
    
    # Log information about detected missing values
    missing_counts = df.isna().sum()
    if missing_counts.any():
        logger.info(f"Detected and standardized missing values: {missing_counts[missing_counts > 0].to_dict()}")
        
    return df

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classifies columns into nominal, ordinal, or free text based on data patterns.
    Now handles column name normalization and matching more robustly.
    """
    column_types = {'nominal': [], 'ordinal': [], 'free_text': []}
    cat_cols = df.select_dtypes(include='object')

    # Pre-calculate metrics to avoid repeated computation
    total_rows = len(df)
    MAX_TEXT_LENGTH = 100

    # Process each column to determine its type
    for col in cat_cols.columns:
        # Calculate average text length for the column
        avg_length = df[col].astype(str).str.len().mean()

        # Handle long text fields separately
        if avg_length > MAX_TEXT_LENGTH:
            column_types['free_text'].append(col)
            continue

        unique_ratio = df[col].nunique() / total_rows
        if unique_ratio > 0.3 and df[col].nunique() > 8:
            column_types['free_text'].append(col)
        else:
            # Check for ordinal patterns in the data
            if any(keyword in [normalize_column_name(val) for val in df[col].unique()]
                   for scale in ORDERED_KEYWORD_SET.values()
                   for keyword in scale):
                column_types['ordinal'].append(col)
            else:
                column_types['nominal'].append(col)

    logger.info(f"Column classification complete: {len(column_types['nominal'])} nominal, "
                f"{len(column_types['ordinal'])} ordinal, {len(column_types['free_text'])} free text")
    return column_types


def determine_category_order(col_values: List[str]) -> List[str]:
    """
    Determines the correct order of categories for ordinal encoding.
    Uses cached patterns when possible to improve performance.
    """
    lower_col_values = [normalize_column_name(val) for val in col_values]
    best_match = None
    best_match_count = 0

    # Match against known patterns
    for scale_name, keywords in ORDERED_KEYWORD_SET.items():
        keyword_list = list(keywords.keys()) if isinstance(keywords, dict) else keywords
        match_count = sum(1 for val in lower_col_values if val in keyword_list)

        if match_count > best_match_count:
            best_match = keywords
            best_match_count = match_count

    # Sort based on the best matching pattern
    if best_match:
        if isinstance(best_match, dict):
            return sorted(col_values, key=lambda x: best_match.get(normalize_column_name(x), float('inf')))
        return sorted(col_values,
                      key=lambda x: best_match.index(normalize_column_name(x))
                      if normalize_column_name(x) in best_match else float('inf'))
    return col_values


def encode_columns(df: pd.DataFrame, column_types: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Encodes columns based on their type with improved error handling and normalization.
    Uses parallel processing for large datasets to improve performance.
    """
    df = df.copy()
    encoder_cache = {}

    def encode_ordinal_column(col: str) -> pd.Series:
        """Helper function to encode a single ordinal column"""
        try:
            # Only use non-missing values for determining categories
            unique_values = df[col].dropna().unique()
            categories = [determine_category_order(unique_values)]

            if col not in encoder_cache:
                encoder_cache[col] = OrdinalEncoder(categories=categories, 
                                                   handle_unknown='use_encoded_value',
                                                   unknown_value=np.nan)

            # Keep track of NaN positions
            nan_mask = df[col].isna()
            
            # Only transform non-NaN values
            if nan_mask.all():
                return pd.Series(np.nan, index=df.index, name=col)
            
            encoded_values = encoder_cache[col].fit_transform(df.loc[~nan_mask, [col]]) + 1
            
            # Create result series with NaNs in the right places
            result = pd.Series(np.nan, index=df.index, name=col)
            result.loc[~nan_mask] = encoded_values.ravel()
            
            return result

        except Exception as e:
            logger.warning(f"Could not encode ordinal column {col}: {e}")
            return df[col]

    def encode_nominal_column(col: str) -> pd.Series:
        """Helper function to encode a single nominal column"""
        try:
            # Handle missing values
            nan_mask = df[col].isna()
            
            # If all values are missing, return all NaNs
            if nan_mask.all():
                return pd.Series(np.nan, index=df.index, name=col)
            
            if df[col].nunique() == 2:
                # For binary columns, use categorical encoding
                encoded = pd.Categorical(df[col]).codes
                # Convert -1 (missing) to NaN
                encoded = pd.Series(encoded, index=df.index)
                encoded[encoded == -1] = np.nan
                return encoded
            else:
                if col not in encoder_cache:
                    encoder_cache[col] = LabelEncoder()
                
                # Create a temporary series for encoding
                temp_series = df[col].copy()
                # Mark missing values with a placeholder
                if nan_mask.any():
                    temp_series.loc[nan_mask] = '__MISSING__'
                
                # Encode non-missing values
                encoded = pd.Series(
                    encoder_cache[col].fit_transform(temp_series.astype(str)),
                    index=df.index,
                    name=col
                )
                
                # Convert the missing value placeholder back to NaN
                if nan_mask.any():
                    missing_code = encoder_cache[col].transform(['__MISSING__'])[0]
                    encoded[encoded == missing_code] = np.nan
                
                return encoded
        except Exception as e:
            logger.warning(f"Could not encode nominal column {col}: {e}")
            return df[col]

    # Process columns in parallel if dataset is large enough
    if len(df) > 1000:
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Process ordinal columns
            if column_types['ordinal']:
                ordinal_results = list(executor.map(encode_ordinal_column, column_types['ordinal']))
                for col, result in zip(column_types['ordinal'], ordinal_results):
                    df[col] = result

            # Process nominal columns
            if column_types['nominal']:
                nominal_results = list(executor.map(encode_nominal_column, column_types['nominal']))
                for col, result in zip(column_types['nominal'], nominal_results):
                    df[col] = result
    else:
        # Process sequentially for smaller datasets
        for col in column_types['ordinal']:
            df[col] = encode_ordinal_column(col)
        for col in column_types['nominal']:
            df[col] = encode_nominal_column(col)

    return df


def normalize_and_match_columns(df: pd.DataFrame, columns_to_match: List[str]) -> List[str]:
    """
    Finds matching columns in the DataFrame even with slight naming differences.
    Returns the list of matched original column names.
    """
    df_cols = {normalize_column_name(col): col for col in df.columns}
    matched_cols = []

    for col in columns_to_match:
        norm_col = normalize_column_name(col)
        if norm_col in df_cols:
            matched_cols.append(df_cols[norm_col])

    return matched_cols


def generate_qa_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a quality assurance report with missing value analysis
    and potential data quality issues.
    """
    return {
        'missing_values': df.isnull().sum().to_dict(),
        'rows_with_3_or_more_missing_values': {
            'count': len(df[df.isnull().sum(axis=1) >= 3]),
            'row_indices': df[df.isnull().sum(axis=1) >= 3].index.tolist()
        }
    }


def sanity_check_chills(df: pd.DataFrame, chills_column: str,
                        chills_intensity_column: str, threshold: int = 0) -> pd.DataFrame:
    """
    Performs consistency checks on chills-related data.
    Flags or removes inconsistent responses based on the specified mode.
    """
    df = df.copy()
    try:
        inconsistent_rows = (df[chills_column] == 0) & (df[chills_intensity_column] >= threshold)
        df['Sanity_Flag'] = inconsistent_rows.astype(int)
    except Exception as e:
        logger.error(f"Error in chills sanity check: {e}")
        df['Sanity_Flag'] = 0
    return df


def process_data_pipeline(
        input_df: pd.DataFrame,
        chills_column: Optional[str] = None,
        chills_intensity_column: Optional[str] = None,
        intensity_threshold: int = 0,
        mode: str = 'flag',
        user_column_mappings: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Main pipeline function that coordinates all data processing steps.
    Handles column normalization, encoding, and scoring with improved error handling.
    """
    try:
        # Create a working copy of the input data
        df = input_df.copy()
        
        # Handle missing values first - convert to proper NaN format
        df = handle_missing_values(df)

        # Normalize column names in the DataFrame
        df.columns = [normalize_column_name(col) for col in df.columns]

        # Process user column mappings to match normalized format
        if user_column_mappings:
            normalized_mappings = {}
            for scale, mapping in user_column_mappings.items():
                normalized_mapping = {}
                for q_num, col_name in mapping.items():
                    normalized_col = normalize_column_name(col_name)
                    matching_cols = normalize_and_match_columns(df, [col_name])
                    if matching_cols:
                        normalized_mapping[q_num] = matching_cols[0]
                if normalized_mapping:
                    normalized_mappings[scale] = normalized_mapping
            user_column_mappings = normalized_mappings

        # Generate QA report
        qa_report = generate_qa_report(df)

        # Detect and encode column types
        column_types = detect_column_types(df)
        intermediate_df = encode_columns(df, column_types)

        # Perform chills analysis if requested
        if chills_column and chills_intensity_column:
            intermediate_df = sanity_check_chills(
                intermediate_df,
                normalize_column_name(chills_column),
                normalize_column_name(chills_intensity_column),
                intensity_threshold
            )

        # Calculate behavioral scores
        scorer = ScaleScorer(intermediate_df, user_column_mappings)
        final_df = scorer.calculate_all_scales()

        return final_df, intermediate_df, str(qa_report)

    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise