import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the student success dataset.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    df = df.copy()

    # Rename columns
    df.rename(columns={'Nacionality': 'Nationality'}, inplace=True)

    # Map Target column if it exists and is categorical
    if 'Target' in df.columns and df['Target'].dtype == 'object':
        df['Target'] = df['Target'].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

    return df
