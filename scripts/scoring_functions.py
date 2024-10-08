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


def calculate_all_scales(df, user_column_mappings):
    """
    Calculate all available scale scores for the input DataFrame and drop question columns after scoring.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns corresponding to multiple scales.
    user_column_mappings : dict
        Dictionary mapping scales to question-to-column mappings.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the original columns with additional columns for each calculated scale.
        All question columns are removed after scoring.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df_scored = df.copy()

    # Dictionary of scale scoring functions
    scoring_functions = {
        'MODTAS': score_modtas,  # Add other scale functions here as needed
    }

    # Track all question columns to be dropped
    question_columns_to_drop = []

    # Normalize the DataFrame columns once at the beginning
    df_scored.columns = [normalize_column_name(col) for col in df_scored.columns]

    # Calculate each scale score and add it as a new column
    for scale_name, scoring_fn in scoring_functions.items():
        if scale_name not in user_column_mappings:
            print(f"\n\n\nSkipping {scale_name} because no user mappings are provided.")
            continue

        try:
            # Get the user-provided column mappings for this scale
            column_mapping = user_column_mappings[scale_name]

            # Normalize the column mapping to ensure consistency
            column_mapping = {k: normalize_column_name(v) for k, v in column_mapping.items()}
            print(f"\n\n\nMapping for {scale_name}: {column_mapping}")  # Debug: Show mappings

            # Calculate the score using the mapped columns
            df_scored[scale_name + '_Score'] = scoring_fn(df_scored, column_mapping)
            print(f"\n\n\nSuccessfully scored {scale_name}")

            # Add the columns used in this scale to the drop list
            question_columns_to_drop.extend(list(column_mapping.values()))

        except Exception as e:
            # Debug output: Which columns are missing?
            print(f"\n\n\nSkipping {scale_name} due to error: {e}")
            print(f"DataFrame columns: {df.columns.tolist()}")  # Print current columns for debug

    # Remove the columns used for scoring
    print(f"\n\nColumns to be dropped: {question_columns_to_drop}")  # Debug: Show columns to be dropped
    df_scored = df_scored.drop(columns=question_columns_to_drop, errors='ignore')

    # Debug: Print the final columns after scoring
    print("Columns after scoring and dropping questions:", df_scored.columns.tolist())
    return df_scored





