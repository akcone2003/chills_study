from scales.scale_mappings import get_scale_questions
import pandas as pd

# Function to score the MODTAS scale
def score_modtas(df, column_mapping):
    """
    Calculate the MODTAS (Modified Tellegen Absorption Scale) average score for each row in the DataFrame.

    Parameters:
    df : pd.DataFrame
        Input DataFrame with columns corresponding to the MODTAS questions.
    column_mapping : dict
        Dictionary mapping the original question to the corresponding column in the input DataFrame.

    Returns:
    pd.Series
        A Series containing the average MODTAS scores for each row in the DataFrame.
    """
    # Get the mapped column names for the MODTAS scale
    modtas_questions = list(column_mapping.values())

    # Check if the necessary questions are in the dataframe
    missing_columns = [q for q in modtas_questions if q not in df.columns]
    if missing_columns:
        print(f"Missing columns for MODTAS scoring: {missing_columns}")
        return pd.Series([None] * len(df))  # Return None values for rows if columns are missing

    # Calculate the average of all MODTAS questions for each row
    return df[modtas_questions].mean(axis=1)


# TODO - make functions for other scales in the questionnaire


def calculate_all_scales(df, user_column_mappings):
    """
    Calculate all available scale scores for the input DataFrame and drop question columns after scoring.

    Parameters:
    df : pd.DataFrame
        Input DataFrame containing columns corresponding to multiple scales.
    user_column_mappings : dict
        Dictionary mapping scales to question-to-column mappings.

    Returns:
    pd.DataFrame
        A DataFrame containing the original columns with additional columns for each calculated scale.
        All question columns are removed after scoring.
    """
    # Debug: Print the columns before scoring
    print("Columns before scoring:", df.columns.tolist())

    df_scored = df.copy()
    scoring_functions = {
        'MODTAS': score_modtas,  # Add other scale functions here as needed
    }

    # Track all question columns to be dropped
    question_columns_to_drop = []

    # Calculate each scale score and add as a new column
    for scale_name, scoring_fn in scoring_functions.items():
        if scale_name not in user_column_mappings:
            print(f"Skipping {scale_name} because no user mappings are provided.")
            continue

        try:
            # Get the user column mappings for this scale
            column_mapping = user_column_mappings[scale_name]
            print(f"Mapping for {scale_name}: {column_mapping}")  # Debug: Show mappings

            # Calculate the score using the mapped columns
            df_scored[scale_name + '_Score'] = scoring_fn(df, column_mapping)
            print(f"Successfully scored {scale_name}")

            # Add the columns used in this scale to the drop list
            question_columns_to_drop.extend(list(column_mapping.values()))

        except Exception as e:
            # Debug output: Which columns are missing?
            print(f"Skipping {scale_name} due to error: {e}")
            print(f"DataFrame columns: {df.columns.tolist()}")  # Print current columns for debug

    # Remove the columns used for scoring
    print(f"Columns to be dropped: {question_columns_to_drop}")  # Debug: Show columns to be dropped
    df_scored = df_scored.drop(columns=question_columns_to_drop, errors='ignore')

    # Debug: Print the final columns after scoring
    print("Columns after scoring and dropping questions:", df_scored.columns.tolist())
    return df_scored





