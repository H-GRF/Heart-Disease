from src.processing import load_and_preprocess
import pandas as pd
import os

def test_data_loading():
    """Verify that the data loads and encoders are created."""
    data_path = 'data/heart.csv'
    if os.path.exists(data_path):
        df, encoders = load_and_preprocess(data_path)
        assert isinstance(df, pd.DataFrame)
        assert len(encoders) > 0