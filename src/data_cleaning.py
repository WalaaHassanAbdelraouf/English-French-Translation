# Data cleaning utilities 
import pandas as pd

# Explore the dataset and print basic statistics.
def explore_data(df):
    data_shape = df.shape
    data_info = df.info()
    missing_vals = df.isnull().sum()
    duplicated_vals = df.duplicated().sum()
    print('Data Shape: ', data_shape)
    print('Data Info: ', data_info)
    print('Missing Values: ', missing_vals)
    print('duplicated Values: ', duplicated_vals)


# Clean the dataset by removing missing values and showing text statistics.
def data_cleaning(df):
    df = df.dropna()
    print('Text Statistics')
    print('English Language')
    print(f"Average length: {df['en'].astype(str).str.len().mean():.1f} characters")
    print(f"Min length: {df['en'].astype(str).str.len().min()} characters")
    print(f"Max length: {df['en'].astype(str).str.len().max()} characters")
    
    print('\nFrench Language')
    print(f"Average length: {df['fr'].astype(str).str.len().mean():.1f} characters")
    print(f"Min length: {df['fr'].astype(str).str.len().min()} characters")
    print(f"Max length: {df['fr'].astype(str).str.len().max()} characters")
    return df


# Apply text cleaning including length filtering and normalization.
def text_cleaning(df):
    # Length filtering 
    df = df[
        (df['en'].str.split().str.len() >= 1) & 
        (df['en'].str.split().str.len() <= 128) &
        (df['fr'].str.split().str.len() >= 1) & 
        (df['fr'].str.split().str.len() <= 128)
    ]
    # Normalize: lowercase, strip whitespace
    df['en'] = df['en'].str.lower().str.strip()
    df['fr'] = df['fr'].str.lower().str.strip()
    return df