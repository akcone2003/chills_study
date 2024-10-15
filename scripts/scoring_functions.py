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

    # Check if the necessary questions are in the DataFrame
    missing_columns = [q for q in modtas_questions if q not in df.columns]
    if missing_columns:
        print(f"\n\n\nMissing columns for MODTAS scoring: {missing_columns}")
        return pd.Series(['Missing Columns'] * len(df))  # Return None values for rows if columns are missing

    # Calculate the average of all MODTAS questions for each row
    return df[modtas_questions].mean(axis=1)


def score_tipi(df, column_mapping): # TODO - unsure how to code this lowkey
    """
    Calculate the TIPI (ten-item personality inventory) average score for each row in the DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame with columns corresponding to the MODTAS questions.
    column_mapping : dict
        Dictionary mapping the original question to the corresponding column in the input DataFrame.

    Returns:
    -------
    pd.Series
        A Series containing the TIPI scores for each row in the DataFrame.
    """

    def recode_reverse_score(item_score):
        """
        Recode the reverse-scored items.
        Reverse scoring is done by subtracting the original score from 8.
        """
        return 8 - item_score

    # Normalize the DataFrame columns as well
    df.columns = [normalize_column_name(col) for col in df.columns]

    # Recode reverse-scored items within the DataFrame using column mappings for reverse-scored items
    df[column_mapping[2]] = df[column_mapping[2]].apply(recode_reverse_score)
    df[column_mapping[4]] = df[column_mapping[4]].apply(recode_reverse_score)
    df[column_mapping[6]] = df[column_mapping[6]].apply(recode_reverse_score)
    df[column_mapping[8]] = df[column_mapping[8]].apply(recode_reverse_score)
    df[column_mapping[10]] = df[column_mapping[10]].apply(recode_reverse_score)

    # Calculate the average scores for each personality dimension
    df['Extraversion'] = df[[column_mapping[1], column_mapping[6]]].mean(axis=1)
    df['Agreeableness'] = df[[column_mapping[2], column_mapping[7]]].mean(axis=1)
    df['Conscientiousness'] = df[[column_mapping[3], column_mapping[8]]].mean(axis=1)
    df['Emotional_Stability'] = df[[column_mapping[4], column_mapping[9]]].mean(axis=1)
    df['Openness_to_Experience'] = df[[column_mapping[5], column_mapping[10]]].mean(axis=1)

    # Return a DataFrame with the calculated scores for each row
    return df[['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness_to_Experience']]


def score_vviq(df, column_mapping):
    """
    Calculate the VVIQ (Vividness of Visual Imagery Questionnaire) average score for each row in the DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame with columns corresponding to the VVIQ questions.
    column_mapping : dict
        Dictionary mapping the VVIQ questions to their corresponding columns in the input DataFrame.

    Returns:
    -------
    pd.Series
        A Series containing the average VVIQ score for each row in the DataFrame.
    """

    # Normalize the DataFrame columns
    df.columns = [normalize_column_name(col) for col in df.columns]

    # Print the column names in the DataFrame for debugging
    print("Columns in DataFrame:", df.columns)
    print("\n\nVVIQ Column Mappings:", column_mapping)

    try:
        # Collect the 16 VVIQ columns specified in column_mapping
        vviq_columns = [normalize_column_name(col) for col in column_mapping.values()]

        # Ensure that we have all the required VVIQ columns
        if len(vviq_columns) != 16:
            missing_columns = set(range(1, 17)) - set(column_mapping.keys())
            raise ValueError(f"Missing columns for VVIQ items: {missing_columns}")

        # Calculate the average VVIQ score for each row
        return df[vviq_columns].mean(axis=1)

    except KeyError as e:
        raise KeyError(f"Column mapping is missing or invalid for VVIQ: {e}")
    except ValueError as e:
        raise ValueError(f"An error occurred with VVIQ scoring: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during VVIQ scoring: {e}")




# TODO - add more functions for scoring the other behavioral measures
# Make sure that the sub scores are being represented


# Updated calculate_all_scales to support mid-processing and final outputs
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
        'MODTAS': score_modtas,  # TODO - Add other scale functions here as needed
        'TIPI': score_tipi,
        'VVIQ': score_vviq
    }

    question_columns_to_drop = []

    print("\n\nColumns in DataFrame:", df.columns)

    # Calculate each scale score and add it as a new column
    for scale_name, scoring_fn in scoring_functions.items():
        if scale_name not in user_column_mappings:
            continue

        column_mapping = user_column_mappings[scale_name]
        df_scored[scale_name + '_Score'] = scoring_fn(df_scored, column_mapping)
        question_columns_to_drop.extend(list(column_mapping.values()))

    # Return the encoded DataFrame for mid-processing step
    if mid_processing:
        return df_scored

    # Remove question columns for final output
    df_scored = df_scored.drop(columns=question_columns_to_drop, errors='ignore')

    return df_scored

# TODO - look into starting with string associated with behavioral measure
# will remove need for user input, just create checks in pipeline for if a question starts with the string for behavioral measure, it will call that
# scoring function
