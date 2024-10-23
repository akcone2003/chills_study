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

    # Check if the necessary questions are in the DataFrame
    missing_columns = [q for q in modtas_questions if q not in df.columns]
    if missing_columns:
        print(f"\n\n\nMissing columns for MODTAS scoring: {missing_columns}")
        return pd.Series(['Missing Columns'] * len(df))  # Return None values for rows if columns are missing

    # Calculate the average of all MODTAS questions for each row
    return df[modtas_questions].mean(axis=1)


def score_tipi(df, column_mapping):  # TODO - unsure how to code this lowkey
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

    # Get questions
    df.columns = list(column_mapping.values())

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


def score_vviq(df, column_mapping):  # TODO - need help
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

    # Print the column names in the DataFrame for debugging
    print("Columns in DataFrame:", df.columns)
    print("\n\nVVIQ Column Mappings:", column_mapping)

    try:
        # Collect the 16 VVIQ columns specified in column_mapping
        vviq_columns = list(column_mapping.values())

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


def score_kamf(df, column_mapping):  # TODO - need help with this
    pass


def score_dpes_awe(df, column_mapping):
    """
        Calculate the DPES-Awe (Dispositional Positive Emotion Scale) score for each row in the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame
        column_mapping : dict
            Dictionary mapping the DPES-Awe questions to their corresponding columns in the input DataFrame.

        Returns:
        -------
        pd.Series
            A Series containing the score of DPES-Awe for each row in the DataFrame.
    """
    dpes_questions = [normalize_column_name(col) for col in column_mapping.values()]

    # Check if the necessary questions are in the DataFrame
    missing_columns = [q for q in dpes_questions if q not in df.columns]
    if missing_columns:
        print(f"\n\n\nMissing columns for DPES-Awe scoring: {missing_columns}")
        return pd.Series(['Missing Columns'] * len(df))  # Return None values for rows if columns are missing

    # Return dataframe with aggregated score for each row
    return df[dpes_questions].sum(axis=1)


def score_maia(df, column_mapping):
    """
       Calculate the MAIA (Multidimensional Assessment of Interoceptive Awareness) score for each row in the DataFrame.

       Parameters:
       ----------
       df : pd.DataFrame
           Input DataFrame
       column_mapping : dict
           Dictionary mapping the MAIA questions to their corresponding columns in the input DataFrame.

       Returns:
       -------
       pd.Series
           A series containing the score of MAIA for each row in the DataFrame.
    """
    maia_questions = [normalize_column_name(col) for col in column_mapping.values()]

    # Check if the necessary questions are in the DataFrame
    missing_columns = [q for q in maia_questions if q not in df.columns]
    if missing_columns:
        print(f"\n\n\nMissing columns for MAIA scoring: {missing_columns}")
        return pd.Series(['Missing Columns'] * len(df))  # Return None values for rows if columns are missing

    # Return dataframe with aggregated score for each row
    return df[maia_questions].sum(axis=1)


def score_ego_dissolution(df, column_mapping):
    """
        Calculate the Ego-Dissolution score for each row in the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
        Input DataFrame
        column_mapping : dict
        Dictionary mapping the ego-dissolution questions to their corresponding columns in the input DataFrame.

        Returns:
        -------
        pd.Series
        A series containing the score of dissolution for each row in the DataFrame.
    """
    ego_dissolution_questions = [normalize_column_name(col) for col in column_mapping.values()]

    # Check if the necessary questions are in the DataFrame
    missing_columns = [q for q in ego_dissolution_questions if q not in df.columns]
    if missing_columns:
        print(f"\n\n\nMissing columns for Ego-Dissolution scoring: {missing_columns}")
        return pd.Series(['Missing Columns'] * len(df))  # Return 'Missing Columns' for rows if columns are missing

    # Return dataframe with aggregated score for each row
    return df[ego_dissolution_questions].sum(axis=1)


def score_smes(df, column_mapping):
    """
       Calculate the SMES score for each row in the DataFrame.

       Parameters:
       ----------
       df : pd.DataFrame
           Input DataFrame
       column_mapping : dict
           Dictionary mapping the SMES questions to their corresponding columns in the input DataFrame.

       Returns:
       -------
       pd.Series
           A series containing the score of SMES for each row in the DataFrame.
    """
    smes_questions = [normalize_column_name(col) for col in column_mapping.values()]

    # Check if the necessary questions are in the DataFrame
    missing_columns = [q for q in smes_questions if q not in df.columns]
    if missing_columns:
        print(f"\n\n\nMissing columns for SMES scoring: {missing_columns}")
        return pd.Series(['Missing Columns'] * len(df))  # Return None values for rows if columns are missing

    # Return dataframe with aggregated score for each row
    return df[smes_questions].sum(axis=1)


def score_emotional_breakthrough(df, column_mapping):
    """
       Calculate the Ego-Dissolution score for each row in the DataFrame.

       Parameters:
       ----------
       df : pd.DataFrame
           Input DataFrame
       column_mapping : dict
           Dictionary mapping the ego-dissolution questions to their corresponding columns in the input DataFrame.

       Returns:
       -------
       pd.Series
           A series containing the score of dissolution for each row in the DataFrame.
     """
    emot_break_questions = [normalize_column_name(col) for col in column_mapping.values()]

    # Check if the necessary questions are in the DataFrame
    missing_columns = [q for q in emot_break_questions if q not in df.columns]
    if missing_columns:
        print(f"\n\n\nMissing columns for Emotional Breakthrough scoring: {missing_columns}")
        return pd.Series(['Missing Columns'] * len(df))  # Return None values for rows if columns are missing

    # Return dataframe with aggregated score for each row
    return df[emot_break_questions].sum(axis=1)


def score_psychological_insight(df, column_mapping):
    """
       Calculate the Ego-Dissolution score for each row in the DataFrame.

       Parameters:
       ----------
       df : pd.DataFrame
           Input DataFrame
       column_mapping : dict
           Dictionary mapping the ego-dissolution questions to their corresponding columns in the input DataFrame.

       Returns:
       -------
       pd.Series
           A series containing the score of dissolution for each row in the DataFrame.
     """
    psych_insight_questions = [normalize_column_name(col) for col in column_mapping.values()]

    # Check if the necessary questions are in the DataFrame
    missing_columns = [q for q in psych_insight_questions if q not in df.columns]
    if missing_columns:
        print(f"\n\n\nMissing columns for Psychological Insight scoring: {missing_columns}")
        return pd.Series(['Missing Columns'] * len(df))  # Return None values for rows if columns are missing

    # Return dataframe with aggregated score for each row
    return df[psych_insight_questions].sum(axis=1)


# TODO - add more functions for scoring the other behavioral measures


# Updated calculate_all_scales to support mid-processing and final outputs
# Main function to detect scales and calculate scores
def calculate_all_scales(df, mid_processing=False):
    """
    Dynamically detect scales based on column prefixes and calculate scores.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns for multiple behavioral measures.
    mid_processing : bool, optional
        If True, retains the original columns. Default is False.

    Returns:
    -------
    pd.DataFrame
        DataFrame with calculated scale scores and (optionally) question columns removed.
    """
    df_scored = df.copy()

    # Define the mapping between scale names (prefix) and their scoring functions
    # TODO - add here
    scoring_functions = {
        'MODTAS': score_modtas,
        'TIPI': score_tipi,
        'VVIQ': score_vviq,
        'KAMF': score_kamf,
        'DPES-Awe': score_dpes_awe,
        'MAIA': score_maia,
        'Ego-Dissolution': score_ego_dissolution,
        'SMES': score_smes,
        'Emotional-Breakthrough': score_emotional_breakthrough,
        # TODO - need to check on this if we have a hyphen or not in google form
    }

    question_columns_to_drop = []

    # Loop through all scoring functions and process relevant columns
    for scale_name, scoring_fn in scoring_functions.items():
        # Find all columns starting with the scale prefix
        matching_columns = [col for col in df.columns if col.startswith(scale_name)]

        if matching_columns:
            print(f"Calculating scores for: {scale_name} (Columns: {len(matching_columns)})")

            # Calculate the scale score and add it as a new column
            df_scored[scale_name] = scoring_fn(df_scored, matching_columns)
            question_columns_to_drop.extend(matching_columns)

    # Optionally drop question columns for the final output
    if not mid_processing:
        df_scored = df_scored.drop(columns=question_columns_to_drop, errors='ignore')

    return df_scored
