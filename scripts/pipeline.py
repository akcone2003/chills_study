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
    Handles empty cells and common missing value representations in the dataframe.
    
    Args:
        df: Input DataFrame potentially containing empty cells
        
    Returns:
        DataFrame with consistent NaN representation for missing values
    """
    # Make a copy to avoid modifying the input
    df = df.copy()
    
    # When pandas reads a CSV, empty cells are typically already converted to NaN
    # But we'll handle additional cases that might be interpreted as strings
    
    # Common missing value indicators in surveys
    na_values = ['nan', 'NaN', 'NA', 'N/A', '', 'None']
    
    # Replace these values with np.nan
    df = df.replace(na_values, np.nan)
    
    # Log how many missing values we found
    missing_count = df.isna().sum().sum()
    logger.info(f"Identified {missing_count} missing values across all columns")
    
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
    Uses the predefined scales in ORDERED_KEYWORD_SET from utils.py for encoding.
    """
    df = df.copy()

    def encode_ordinal_column(col: str) -> pd.Series:
        """Helper function to encode a single ordinal column using ORDERED_KEYWORD_SET"""
        try:
            # Get non-NaN values
            non_na_mask = ~df[col].isna()
            if not non_na_mask.any():
                logger.warning(f"Column {col} has all missing values, skipping encoding")
                return df[col]
            
            # Create a result series to store encoded values
            result = pd.Series(np.nan, index=df.index, name=col)
            
            # Get all unique normalized values
            unique_values = [normalize_column_name(str(val)) for val in df[col].dropna().unique()]
            
            # Find the best matching scale
            best_scale = None
            best_match_count = 0
            
            for scale_name, scale in ORDERED_KEYWORD_SET.items():
                if isinstance(scale, dict):
                    keywords = list(scale.keys())
                else:
                    keywords = scale
                
                match_count = sum(1 for val in unique_values if val in keywords)
                if match_count > best_match_count:
                    best_scale = scale_name
                    best_match_count = match_count
            
            # If we found a matching scale, use it to encode
            if best_scale:
                logger.info(f"Encoding column '{col}' using scale '{best_scale}'")
                scale = ORDERED_KEYWORD_SET[best_scale]
                
                # Apply encoding
                for idx, val in enumerate(df.loc[non_na_mask, col]):
                    norm_val = normalize_column_name(str(val))
                    if isinstance(scale, dict):
                        # Use dictionary mapping
                        encoded_val = scale.get(norm_val)
                        if encoded_val is not None:
                            result.iloc[df.index.get_indexer([df.index[non_na_mask].tolist()[idx]])[0]] = encoded_val
                    else:
                        # Use list ordering
                        try:
                            encoded_val = scale.index(norm_val)
                            result.iloc[df.index.get_indexer([df.index[non_na_mask].tolist()[idx]])[0]] = encoded_val
                        except ValueError:
                            pass
            else:
                # If no matching scale, just use OrdinalEncoder as fallback
                logger.warning(f"No matching scale found for column '{col}', using fallback encoding")
                unique_vals = df[col].dropna().unique()
                categories = [determine_category_order(unique_vals)]
                encoder = OrdinalEncoder(categories=categories)
                encoded_vals = encoder.fit_transform(df.loc[non_na_mask, [col]]) + 1
                result.loc[non_na_mask] = encoded_vals.ravel()
            
            return result
                
        except Exception as e:
            logger.warning(f"Could not encode ordinal column {col}: {str(e)}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            return df[col]

    def encode_nominal_column(col: str) -> pd.Series:
        """Helper function to encode a single nominal column"""
        try:
            # Skip completely empty columns
            if df[col].isna().all():
                return df[col]
                
            # Create a copy for the result
            result = pd.Series(np.nan, index=df.index, name=col)
            non_na_mask = ~df[col].isna()
            
            # For binary columns, use 0 and 1
            if df[col].dropna().nunique() == 2:
                unique_vals = df[col].dropna().unique()
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                result.loc[non_na_mask] = df.loc[non_na_mask, col].map(mapping)
            else:
                # For multi-value columns, use LabelEncoder
                encoder = LabelEncoder()
                encoder.fit(df.loc[non_na_mask, col].astype(str))
                result.loc[non_na_mask] = encoder.transform(df.loc[non_na_mask, col].astype(str))
                
            return result
        except Exception as e:
            logger.warning(f"Could not encode nominal column {col}: {str(e)}")
            return df[col]

    # Process each column
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

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive preprocessing function that:
    1. Handles missing values (empty cells)
    2. Converts string numbers to actual numeric types
    3. Preserves NaN values appropriately
    
    Args:
        df: Input DataFrame with potential missing values and string numerics
        
    Returns:
        DataFrame with standardized data types and missing values
    """
    # Make a copy to avoid modifying the input
    df = df.copy()
    
    # First, identify columns that should be numeric
    potentially_numeric_cols = []
    for col in df.columns:
        # Skip columns that are clearly not numeric or scale items
        if col.lower() in ['subj id', 'unnamed: 0'] or 'record id' in col.lower():
            continue
            
        # Check if column contains values that look like numbers
        sample_vals = df[col].dropna().head(10).astype(str)
        numeric_pattern = r'^[-+]?[0-9]*\.?[0-9]+$'
        if any(sample_vals.str.match(numeric_pattern)):
            potentially_numeric_cols.append(col)
    
    # Process each column
    for col in df.columns:
        # Skip columns that are definitely non-numeric
        if col not in potentially_numeric_cols:
            continue
            
        # Try to convert to numeric, keeping NaN values intact
        try:
            # Convert to numeric with coercion (strings that don't look like numbers become NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Log successful conversion
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                logger.info(f"Converted column '{col}' to numeric with values: {sorted(unique_vals)}")
        except Exception as e:
            logger.warning(f"Could not convert column '{col}' to numeric: {str(e)}")
    
    # Handle empty cells and other missing value indicators
    na_values = ['nan', 'NaN', 'NA', 'N/A', '', None]
    df = df.replace(na_values, np.nan)
    
    # Log missing value counts
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if not cols_with_missing.empty:
        logger.info(f"Columns with missing values: {cols_with_missing.to_dict()}")
    
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
        
        # Preprocess the dataframe - handle missing values and convert data types
        df = preprocess_dataframe(df)

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