import pandas as pd


def normalize_column_name(df_or_name):
    """
    Normalize column names for a DataFrame or an individual column name.

    Parameters
    ----------
    df_or_name : pd.DataFrame or str
        If a DataFrame is provided, it normalizes all columns.
        If a string is provided, it normalizes the single column name.

    Returns
    -------
    pd.DataFrame or str
        The DataFrame with normalized columns, or a normalized string.
    """

    def _normalize(name):
        return (
            name.strip()  # Remove leading and trailing spaces
            .replace('“', '"')
            .replace('”', '"')
            .replace('\u00A0', ' ')  # Replace non-breaking space with regular space
            .replace('\n', ' ')  # Replace newlines with spaces
            .replace('\t', ' ')  # Replace tabs with spaces
            .lower()  # Ensure consistent lowercasing
            .replace('  ', ' ')  # Collapse multiple spaces into one
        )

    if isinstance(df_or_name, pd.DataFrame):
        # Apply normalization to all column names in the DataFrame
        df_or_name.columns = [_normalize(col) for col in df_or_name.columns]
        return df_or_name
    elif isinstance(df_or_name, str):
        return _normalize(df_or_name)
    else:
        raise TypeError("Input must be a DataFrame or a column name string.")


def add_behavioral_score_prefix(df, behavioral_score_mappings):
    """
    Add behavioral score prefixes to the questions in the DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing survey questions.
    behavioral_score_mappings : dict
        A dictionary where the key is the behavioral score (e.g., "MODTAS")
        and the value is a list of questions associated with that score.

    Returns:
    -------
    pd.DataFrame
        DataFrame with updated column names, where each question is prefixed
        with the respective behavioral score and question number.
    """
    # Create a new dictionary to store updated column names
    updated_columns = {}

    # Loop through each behavioral score and its associated questions
    for score, questions in behavioral_score_mappings.items():
        for idx, question in enumerate(questions, 1):
            # Build the new column name with prefix
            new_column_name = f"{score} Question {idx}: {question}"
            updated_columns[question] = new_column_name

    # Rename the DataFrame columns using the new mapping
    df = df.rename(columns=updated_columns)

    return df


def normalize_column_input(pasted_text):
    """
    Clean pasted input by removing extra spaces, handling mixed delimiters,
    and ensuring consistent formatting of column names.

    Columns are split based on newlines to avoid breaking on embedded commas.
    """
    # Split only by newlines or tabs to avoid splitting columns with embedded commas
    clean_text = pasted_text.replace('\t', '\n')  # Convert tabs to newlines
    columns = [col.strip() for col in clean_text.splitlines() if col.strip()]
    return columns


def get_score_from_mapping(value, scale_type):
    """Retrieve the score from a mapping based on the scale type."""
    scale_mapping = ORDERED_KEYWORD_SET.get(scale_type)
    if isinstance(scale_mapping, dict):
        return scale_mapping.get(value.lower(), None)
    elif isinstance(scale_mapping, list):
        # For lists, return the index as a score, if the value is in the list
        try:
            return scale_mapping.index(value.lower()) * 25  # Example scale
        except ValueError:
            return None
    return None

# Define multiple ordered keyword lists for different types of scales
ORDERED_KEYWORD_SET = {
    # Recency Scales
    'recency': ['cannot remember', 'within the last year', 'within the last month', 'within the last 24 hours'],
    # Frequency Scales
    'frequency_01': ['never', 'rarely', 'sometimes', 'often', 'always'],
    'frequency_02': ['never', 'less than once a month', 'once a month',
                     '2-3 times a month', 'once a week', '2-3 times a week',
                     'about once a day', 'two or more times per day'],
    'frequency_03': ['never', 'rarely', 'occasionally', 'often', 'very often'],
    'frequency_04': ['almost always', 'very frequently', 'somewhat frequently',
                     'somewhat infrequently', 'very infrequently', 'almost never'],
    'frequency_05': ['never or very rarely true', 'rarely true', 'sometimes true', 'often true',
                     'very often or always true'],
    'frequency_06': ['none of the time', 'rarely', 'some of the time', 'often', 'all of the time'],
    'frequency_07': ['rarely/not at all', 'sometimes', 'often', 'almost always'],
    # Dictionaries for Burnout Scales
    'frequency_08': {'always': 100, 'often': 75, 'sometimes': 50, 'seldom': 25, 'never/almost never': 0},
    'frequency_09': {'to a very high degree': 100, 'to a high degree': 75,
                     'somewhat': 50, 'to a low degree': 25, 'to a very low degree': 0},
    # Agreement Scales
    'agreement_01': ['strongly disagree', 'disagree', 'neither agree nor disagree', 'agree', 'strongly agree'],
    'agreement_02': ['strongly disagree', 'disagree', 'somewhat disagree', 'neutral', 'somewhat agree', 'agree',
                     'strongly agree'],
    'agreement_03': ['completely untrue of me', 'mostly untrue of me', 'slightly more true than untrue',
                     'moderately true of me', 'mostly true of me', 'describes me perfectly'],
    # Intensity Scales
    'intensity_01': ['not at all', 'a little', 'moderately', 'quite a bit', 'extremely'],
    'intensity_02': ['not at all', 'somewhat', 'extremely'],
    'intensity_03': ['very slightly or not at all', 'a little', 'moderately', 'quite a bit', 'extremely'],
    'intensity_04': ['not at all', 'a little', 'somewhat', 'very much'],
    # Mood Scales
    'positivity': ['poor', 'fair', 'good', 'very good', 'excellent']
}