import os
import pandas as pd

def load_all_data(data_dir):
    """
    Loads and concatenates all .pkl transaction files from a directory.
    Assumes files are named by date (e.g., '2018-04-02.pkl').
    """
    all_data = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.pkl'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_pickle(file_path)
            all_data.append(df)
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df
