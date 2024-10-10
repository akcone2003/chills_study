import pandas as pd
from scripts.helpers import normalize_column_name

# Function to score the MODTAS scale
def score_modtas(df, column_mapping):
    """
    Calculate the MODTAS (Modified Tellegen Absorption Scale) average score for each row in the DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame with columns corresponding to the MODTAS questions.
    column_mapping : dict
        Dictionary mapping the original question to the corresponding column in the input DataFrame.

    Returns:
    -------
    pd.Series
        A Series containing the average MODTAS scores for each row in the DataFrame.
    """
    # Normalize column mapping to ensure consistency
    modtas_questions = [normalize_column_name(col) for col in column_mapping.values()]

    # Normalize the DataFrame columns as well
    df.columns = [normalize_column_name(col) for col in df.columns]

    print(f"\n\n\nMODTAS Questions (Expected Columns): {modtas_questions}")  # Debug: Show expected MODTAS questions
    print(f"\n\n\nDataFrame Columns: {df.columns.tolist()}")  # Debug: Show DataFrame columns

    # Check if the necessary questions are in the DataFrame
    missing_columns = [q for q in modtas_questions if q not in df.columns]
    if missing_columns:
        print(f"\n\n\nMissing columns for MODTAS scoring: {missing_columns}")
        return pd.Series(['Missing Columns'] * len(df))  # Return None values for rows if columns are missing

    # Calculate the average of all MODTAS questions for each row
    return df[modtas_questions].mean(axis=1)

# TODO - add more functions for scoring the other behavioral measures
# Make sure that the sub scores are being represented


# CHANGED: Updated calculate_all_scales to support mid-processing and final outputs
def calculate_all_scales(df, user_column_mappings, mid_processing=False):
    """
    Calculate all available scale scores for the input DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns corresponding to multiple scales.
    user_column_mappings : dict
        Dictionary mapping scales to question-to-column mappings.
    mid_processing : bool, optional
        If True, keeps question columns and returns encoded DataFrame (Default: False)

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the original columns with additional columns for each calculated scale.
        All question columns are removed after scoring unless mid_processing is set to True.
    """
    df_scored = df.copy()

    # Dictionary of scale scoring functions
    scoring_functions = {
        'MODTAS': score_modtas,  # Add other scale functions here as needed
    }

    question_columns_to_drop = []

    # Calculate each scale score and add it as a new column
    for scale_name, scoring_fn in scoring_functions.items():
        if scale_name not in user_column_mappings:
            continue

        column_mapping = user_column_mappings[scale_name]
        df_scored[scale_name + '_Score'] = scoring_fn(df_scored, column_mapping)
        question_columns_to_drop.extend(list(column_mapping.values()))

    # CHANGED: Return the encoded DataFrame for mid-processing step
    if mid_processing:
        return df_scored

    # CHANGED: Remove question columns for final output
    df_scored = df_scored.drop(columns=question_columns_to_drop, errors='ignore')

    return df_scored

# TODO - look into starting with string associated with behavioral measure
# will remove need for user input, just create checks in pipeline for if a question starts with the string for behavioral measure, it will call that
# scoring function





