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
