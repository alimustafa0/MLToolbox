import pandas as pd

def load_data(file_path):
    """
    Load data from a file and return a Pandas DataFrame.
    """
    return pd.read_csv(file_path)