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
    if isinstance(df_or_name, pd.DataFrame):
        df_or_name.columns = (
            df_or_name.columns
            .str.strip()
            .str.replace('“', '"', regex=False)
            .str.replace('”', '"', regex=False)
        )
        return df_or_name
    elif isinstance(df_or_name, str):
        return (
            df_or_name.strip()
            .replace('“', '"')
            .replace('”', '"')
        )
    else:
        raise TypeError("Input must be a DataFrame or a column name string.")
