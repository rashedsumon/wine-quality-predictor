import os
import glob
import pandas as pd
import kagglehub

def load_wine_data():
    """
    Downloads the wine reviews dataset from Kaggle using kagglehub
    and loads the primary CSV file into a Pandas DataFrame.
    """
    print("Checking/Downloading dataset from Kaggle...")
    # Download latest version of the dataset
    path = kagglehub.dataset_download("zynicide/wine-reviews")
    print("Path to dataset files:", path)
    
    # Locate the CSV file in the downloaded path
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the downloaded path: {path}")
    
    # We want the main file, usually 'winemag-data-130k-v2.csv' if available, otherwise grab the first one
    main_file = next((f for f in csv_files if "130k" in f), csv_files[0])
    print(f"Loading data from: {main_file}")
    
    # Load dataset, selecting only the columns we need for the model
    df = pd.read_csv(main_file, usecols=['country', 'price', 'variety', 'description', 'points'])
    
    # Drop rows where critical fields like description or points are missing
    df = df.dropna(subset=['description', 'points'])
    
    # Fill missing values for categorical/numerical columns
    df['country'] = df['country'].fillna('Unknown')
    df['variety'] = df['variety'].fillna('Unknown')
    df['price'] = df['price'].fillna(df['price'].median()) # Baseline imputation
    
    return df

if __name__ == "__main__":
    # Test execution
    df = load_wine_data()
    print(f"Dataset successfully loaded! Shape: {df.shape}")