import numpy as np

def add_colunms_to_df(df, column_names_list):
    """This function takes a dataframe and list of columns

    Args:
        df (df): The first parameter.
        column_names_list (list of str): The second parameter.

    Returns:
        df: Will return a dataframe with the list of columns added
    """
    for name in column_names_list:
        df[name] = np.nan
    return df