def normalize_column_names(df):
    """
    Normalize column names in the DataFrame to ensure consistency.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame with potentially inconsistent column names.

    Returns:
    -------
    pd.DataFrame
        DataFrame with normalized column names.
    """
    # Strip leading/trailing spaces, replace special quotes, and convert to lowercase (optional)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace('“', '"', regex=False)
        .str.replace('”', '"', regex=False)
    )
    return df


# Import the dictionary from `scale_questions.py`
from scales.scale_questions import ALL_SCALES

# Use this dictionary to map the scales
def get_scale_questions(scale_name):
    """
    Retrieve the list of questions for a given scale.

    Parameters:
    scale_name (str): Name of the scale to look up.

    Returns:
    list: List of questions for the scale if found, else an empty list.
    """
    return ALL_SCALES.get(scale_name, [])


