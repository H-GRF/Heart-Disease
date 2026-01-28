import os
from src.processing import load_and_preprocess

def test_data_integrity():
    """Verify data folder and preprocessing logic."""
    data_path = 'data/heart.csv'
    assert os.path.exists(data_path), "Dataset file is missing from data/ folder"
    df, encoders = load_and_preprocess(data_path)
    assert not df.empty
    assert 'Sex' in encoders