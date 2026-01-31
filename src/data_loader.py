# Data loading utilities 
import pandas as pd


def load_data(filepath, nrows=None):
    df = pd.read_csv(filepath, nrows=nrows)
    return df